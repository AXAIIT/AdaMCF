# only_train_once/hesso.py
import torch
import numpy as np
import random
from torch.optim.optimizer import Optimizer, required
import torch.nn.functional as F
from collections import deque, defaultdict
from only_train_once.transform import TensorTransform, index_transformation_param_group
from .base_hybrid_sparse_optimizer import BaseHybridSparseOptimizer
from only_train_once.graph import Graph
from only_train_once.transform import tensor_transformation


# === 引入 α 调度器（集中管理超参与更新逻辑） ===
from only_train_once.optimizer.alpha_scheduler import LossGuidedAlphaScheduler, AlphaScheduleConfig


class HESSO(BaseHybridSparseOptimizer):
    @staticmethod
    def _seed_everything(seed: int):
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _build_period_quota(total_groups: int, periods: int, mode: str = "cosine_front"):
        import numpy as np
        if periods <= 1 or total_groups <= 0:
            return [max(total_groups, 0)]
        if mode == "uniform":
            allot = [total_groups // periods] * periods
            allot[-1] += total_groups - sum(allot)
            return allot
        if mode == "uniform_random":
            base = [total_groups // periods] * periods
            remainder = total_groups % periods
            indices = list(range(periods))
            random.shuffle(indices)
            for i in range(remainder):
                base[indices[i]] += 1
            return base
        if mode == "cosine_front":
            x = np.linspace(0, 1, periods, endpoint=True)
            w = 1 - np.cos(np.pi * (1 - x))  # 前期权重更大
            w = w / w.sum()
            alloc = np.floor(w * total_groups).astype(int)
            diff = total_groups - int(alloc.sum())
            if diff > 0:
                order = np.argsort(-w)
                for i in range(diff):
                    alloc[order[i % periods]] += 1
            elif diff < 0:
                order = np.argsort(w)
                for i in range(-diff):
                    idx = order[i % periods]
                    if alloc[idx] > 0:
                        alloc[idx] -= 1
            return alloc.tolist()
        raise ValueError(f"Unknown pruning quota mode: {mode}")


    def __init__(self, params, variant='sgd', lr=required, first_momentum=None, second_momentum=None, \
                 dampening=None, weight_decay=None, target_group_sparsity=0.5, \
                 start_pruning_step=0, pruning_steps=None, pruning_periods=200, \
                 group_divisible=1, importance_score_criteria='default', device='cuda', alpha_scheduler_cfg=None, seed=None):

        '''
        first_momentum (一阶动量)：SGD, Adam, AdamW
        second_momentum (二阶动量)：Adam, AdamW
        weight_decay (权重衰减)：SGD, Adam, AdamW
        dampening (阻尼)：SGD
        '''
        
        print("\n--- HESSO Optimizer Initialization ---")
        print(f"  {'Optimizer Variant:':<30} {variant}")
        print(f"  {'Learning Rate (lr):':<30} {lr}")
        print(f"  {'Target Group Sparsity:':<30} {target_group_sparsity}")
        print(f"  {'Device:':<30} {device}")
        print(f"  {'Start Pruning Step:':<30} {start_pruning_step}")
        print(f"  {'Pruning Steps (total):':<30} {pruning_steps}")
        print(f"  {'Pruning Periods:':<30} {pruning_periods}")
        print(f"  {'Group Divisible:':<30} {group_divisible}")
        print(f"  {'Importance Score Criteria:':<30} {importance_score_criteria}")

        self.start_pruning_step = start_pruning_step
        self.pruning_periods = int(max(1, pruning_periods))
        self.pruning_steps = pruning_steps
        assert self.pruning_steps is not None and self.pruning_steps > 0, "pruning_steps must be > 0"
        self.pruning_period_duration = self.pruning_steps // self.pruning_periods
        assert self.pruning_period_duration > 0, "pruning_period_duration must be > 0"
        self.curr_pruning_period = 0
        self.device = device
        self.seed = seed
        if self.seed is not None:
            self._seed_everything(self.seed)
        self.pruned_group_idxes = list()

        if importance_score_criteria == 'default':
            self.importance_score_criteria = {
                'l1_avg_magnitude': 0.2, 
                'l2_avg_magnitude': 0.2, 
                'cosine_similarity': 0.2, 
                'fisher': 0.2, 
                'grad_weight_product': 0.2
            }
        else:
            self.importance_score_criteria = dict(importance_score_criteria)

        super(HESSO, self).__init__(params=params, variant=variant, lr=lr, first_momentum=first_momentum, second_momentum=second_momentum, \
                                    dampening=dampening, weight_decay=weight_decay, target_group_sparsity=target_group_sparsity, \
                                    group_divisible=group_divisible)

        for param_group in self.param_groups:
            param_group['important_idxes'] = [i for i in range(param_group['num_groups'])]
            param_group['active_redundant_idxes'] = list()
            param_group['pruned_idxes'] = list()
            param_group['importance_scores'] = dict()
            param_group['_raw_scores'] = dict()

        self.active_num_redundant_groups = self._build_period_quota(
            total_groups=self.target_num_redundant_groups,
            periods=self.pruning_periods,
            mode="cosine_front"
        )
        print(f"Target redundant groups per period: ", self.active_num_redundant_groups)

        # === α 调度器（超参在 alpha_scheduler.py 内部集中管理） ===
        self._alpha_keys = list(self.importance_score_criteria.keys())
        init_alpha = [self.importance_score_criteria[k] for k in self._alpha_keys]
        # default_alpha_cfg = {
        #     "lr": 0.04,
        #     "eps": 0.03,
        #     "momentum": 0.0,
        #     "ema_gamma": 0.4,
        #     "g_ema_gamma": 0.3,
        #     "max_step_l1": 0.25,
        #     "eval_drop_ratio": 0.0,
        #     "start_after_period": 2,
        #     "update_stride": 1,
        #     "min_gap_std": 0.2,
        #     "eval_batch_limit": 64,
        #     "eval_batch_grow_every": 2,
        #     "n_alpha_val": 5000,
        #     "alpha_seed": self.seed if self.seed is not None else 42,
        # }
        default_alpha_cfg = {
            "lr": 0.02,
            "eps": 0.03,
            "momentum": 0.0,
            "ema_gamma": 0.6,
            "g_ema_gamma": 0.5,
            "max_step_l1": 0.18,
            "eval_drop_ratio": 0.0,
            "start_after_period": 0,
            "update_stride": 1,
            "min_gap_std": 0.2,
            "eval_batch_limit": 64,
            "eval_batch_grow_every": 2,
            "n_alpha_val": 5000,
            "alpha_seed": self.seed if self.seed is not None else 42,
        }
        if alpha_scheduler_cfg:
            default_alpha_cfg.update(alpha_scheduler_cfg)
        if self.seed is not None:
            default_alpha_cfg.setdefault("alpha_seed", self.seed)
        self.alpha_scheduler_cfg = AlphaScheduleConfig(**default_alpha_cfg)
        self.alpha_scheduler = LossGuidedAlphaScheduler(
            keys=self._alpha_keys,
            init_weights=init_alpha,
            device=self.device,
            cfg=self.alpha_scheduler_cfg,
        )
        self.alpha_scheduler.attach_backend(self)

        # === α 预热调度 ===
        wu = getattr(self.alpha_scheduler_cfg, "warmup_updates", 0)
        if self.start_pruning_step > 0 and wu > 0:
            # 在 [1, start_pruning_step] 区间里平均做 wu 次 α 更新
            self.alpha_warmup_stride = max(1, self.start_pruning_step // wu)
        else:
            self.alpha_warmup_stride = None

    def will_enter_new_period_on_next_step(self) -> bool:
        next_steps = self.num_steps + 1

        # ---------- 1) α 预热阶段：step < start_pruning_step ----------
        # 在这个阶段，step() 里因为 next_steps < start_pruning_step ，
        # 不会真正 commit 剪枝，只是正常更新权重；我们只想利用同一批数据
        # 额外触发 alpha_scheduler.update(...) 做虚拟剪枝评估。
        if next_steps < self.start_pruning_step and self.alpha_warmup_stride:
            return (next_steps % self.alpha_warmup_stride) == 0

        # ---------- 2) 正式剪枝阶段：沿用原来的 period 判断 ----------
        if self.curr_pruning_period >= self.pruning_periods:
            return False
        if next_steps < self.start_pruning_step:
            return False
        return ((next_steps - self.start_pruning_step - 1) % self.pruning_period_duration) == 0


    def _select_naive(self, order_all: np.ndarray, scores_np: np.ndarray, K: int, pool_size: int = None, tie_policy: str = 'current'):
        """
        原始方案：直接按分数从小到大取 K 个未被剪的全局索引。
        """
        # 1) 候选序列（升序）：默认用传入的 order_all
        idx_seq = np.asarray(order_all, dtype=np.int64)
        if pool_size is not None:
            idx_seq = idx_seq[:int(pool_size)]

        # 2) 去掉已经剪过的
        if self.pruned_group_idxes:
            used = np.asarray(self.pruned_group_idxes, dtype=idx_seq.dtype)
            mask = ~np.isin(idx_seq, used)
            idx_seq = idx_seq[mask]

        # 3) 取前 K 个（若不足则取能取到的那么多）
        picked = idx_seq[:int(K)].tolist()
        return picked, {"method": "orig_minK", "picked": len(picked)}


    def identify_redundant_groups(self):
        # 确定冗余组（使用 self.global_scores 对应的 overall 分数）
        global_scores = torch.cat(self.global_scores, dim=0)
        curr_active_num_redundant_groups = self.active_num_redundant_groups[self.curr_pruning_period]
        # 统一升序排序 + 排除已剪
        scores_np = global_scores.detach().cpu().numpy()
        order_all = np.argsort(scores_np).tolist()  # 分数小→更冗余
        if self.pruned_group_idxes:
            used = set(self.pruned_group_idxes)
            order_all = [i for i in order_all if i not in used]

        K_final = int(curr_active_num_redundant_groups)
        K_final = min(K_final, len(order_all))

        picked_global, info = self._select_naive(order_all, scores_np, K_final, pool_size=None, tie_policy='current')

        top_indices = np.array(picked_global, dtype=np.int64)
        self.pruned_group_idxes.extend(picked_global)

        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                global_active_redundant_idx = np.intersect1d(top_indices, group['global_idxes'])

                group['active_redundant_idxes'] = (global_active_redundant_idx - group['global_start_idx']).tolist()
                group['active_redundant_idxes'] = [i for i in group['active_redundant_idxes'] if 0 <= i < group['num_groups']]

                if group['num_groups'] < self.group_divisible:
                    group['active_redundant_idxes'].clear()
                    group['important_idxes'] = [i for i in range(group['num_groups']) if i not in group['pruned_idxes']]
                    continue
                else:
                    curr_num_important_groups = len(group['important_idxes'])
                    trial_num_important_groups = curr_num_important_groups - len(group['active_redundant_idxes'])

                    if trial_num_important_groups % self.group_divisible != 0 or trial_num_important_groups <= 0:
                        ratio = trial_num_important_groups // self.group_divisible + 1
                        if ratio <= 1 or trial_num_important_groups == 0:
                            refined_num_important_groups = max(int(self.group_divisible), 1)
                        else:
                            refined_num_important_groups = max(int(ratio * self.group_divisible), int(self.group_divisible))
                        refined_num_important_groups = min(group['num_groups'], refined_num_important_groups)
                        refined_num_active_redundant_groups = group['num_groups'] - len(group['pruned_idxes']) - refined_num_important_groups

                        self.target_num_redundant_groups += (refined_num_active_redundant_groups - len(group['active_redundant_idxes']))
                        group['active_redundant_idxes'] = group['active_redundant_idxes'][:refined_num_active_redundant_groups]

                group['important_idxes'] = [i for i in group['important_idxes'] if (i not in group['active_redundant_idxes'] and i not in group['pruned_idxes'])]

    def commit_redundant_idxes(self):
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                group['pruned_idxes'].extend(group['active_redundant_idxes'].copy())
                group['active_redundant_idxes'].clear()
                group['important_idxes'] = [i for i in range(group['num_groups']) if i not in group['pruned_idxes']]
                group['importance_scores'].clear()
                group['_raw_scores'].clear()  # 伴随清空，下一轮 compute_importance_scores 会刷新

    def step(self, loss=None, closure=None):
        if closure is not None:
            loss = closure()

        self.num_steps += 1
        if not getattr(self, "_grad_variant_ready", False):
            self.compute_grad_variant()
        else:
            self._grad_variant_ready = False

        if self.num_steps >= self.start_pruning_step and self.curr_pruning_period < self.pruning_periods:
            if (self.num_steps - self.start_pruning_step - 1) % self.pruning_period_duration == 0:
                self.commit_redundant_idxes()
                self.compute_importance_scores()
                self.identify_redundant_groups()
                self.curr_pruning_period += 1

        t = (self.num_steps - self.start_pruning_step) % self.pruning_period_duration
        for group in self.param_groups:
            if not group['is_prunable'] or len(group['active_redundant_idxes']) == 0:
                self.gradient_descent_step(group)
            elif group['is_prunable'] and len(group['active_redundant_idxes']) > 0:
                for (p_name, p, p_transform) in zip(group['p_names'], group['params'], group['p_transform']):
                    if p_name not in group['grad_variant']:
                        continue
                    if group['weight_decay'] is not None and group['variant'] == 'adamw':
                        p.data.add_(group['weight_decay'] * p.data, alpha=-group['lr'])
                    p.data.add_(group['grad_variant'][p_name], alpha=-group['lr'])

                    active_redundant_idxes = index_transformation_param_group(group['active_redundant_idxes'], p_transform, group)
                    if p_transform == TensorTransform.TRANSPOSE and len(p.data.shape) > 1:
                        p.data[:, active_redundant_idxes, ...] *= (self.pruning_period_duration - t - 1.0) / (self.pruning_period_duration - t)
                    else:
                        p.data[active_redundant_idxes] *= (self.pruning_period_duration - t - 1.0) / (self.pruning_period_duration - t)

                    # 处理辅助节点参数
                    for ng_id, offset in group['auxiliary_ngs']:
                        active_redundant_aux_idxes = [i + offset for i in active_redundant_idxes]
                        for aux_p in self.auxiliary_param_groups[ng_id]['params']:
                            # 评估阶段我们会 no_grad 临时缩放和恢复，这里训练期仍保持原逻辑
                            if aux_p.grad is None:
                                continue
                            valid_idxes = [i for i in active_redundant_aux_idxes if 0 <= i < aux_p.data.size(0)]
                            if not valid_idxes:
                                continue
                            aux_p.data[valid_idxes, ...] *= (self.pruning_period_duration - t - 1.0) / (self.pruning_period_duration - t)
            self.fix_pruned_groups_as_zeros(group)

        if self.num_steps >= self.start_pruning_step and t == self.pruning_period_duration - 1:
            self.commit_redundant_idxes()

    def compute_importance_scores(self, **kwargs):
        """
        三步：
        1) 计算各代理 raw 分数（调用 calculate_importance_score 写入 group['importance_scores'][proxy]）
        2) 对每个代理做全局 L2 归一化，并缓存到 group['_raw_scores'][proxy]
        3) 用“当前 α（存于 self.importance_score_criteria 或 alpha_scheduler.get_alpha()）”聚合 overall
        """
        global_start_idx = 0
        self.global_scores = list()

        # 1) 计算 raw
        from .importance_score import calculate_importance_score
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                calculate_importance_score(self.importance_score_criteria, group)  # 写入 group['importance_scores'][proxy]
                ref_param = group['params'][0]
                for proxy_name in self.importance_score_criteria:
                    score = group['importance_scores'].get(proxy_name, None)
                    if score is None:
                        group['importance_scores'][proxy_name] = torch.zeros(
                            group['num_groups'],
                            device=ref_param.device,
                            dtype=ref_param.dtype,
                        )

        # 2) 全局归一化统计
        normalization_denoms = dict.fromkeys(self.importance_score_criteria.keys(), self.safe_guard)
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                for proxy_name in self.importance_score_criteria:
                    if proxy_name not in group['importance_scores']:
                        continue
                    normalization_denoms[proxy_name] += torch.sum(group['importance_scores'][proxy_name] ** 2, dim=0).item()
        for proxy_name in normalization_denoms:
            normalization_denoms[proxy_name] = np.sqrt(normalization_denoms[proxy_name]) + self.safe_guard

        # 3) 缓存“归一化 raw”，并按当前 α 聚合 overall
        # 当前 α：以 importance_score_criteria 的 key 顺序为准
        alpha_vec = torch.tensor(
            [self.importance_score_criteria[k] for k in self._alpha_keys],
            dtype=torch.float32, device=self.device
        )
        # 归一化到单纯形（保险）
        with torch.no_grad():
            alpha_vec = torch.clamp(alpha_vec, min=1e-8)
            alpha_vec = alpha_vec / alpha_vec.sum()

        global_start_idx = 0
        self.global_scores = []
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                group['global_start_idx'] = global_start_idx
                group['global_idxes'] = np.arange(global_start_idx, global_start_idx + group['num_groups'])

                # 清空并缓存“归一化 raw”
                group['_raw_scores'].clear()
                overall = None
                for i, proxy_name in enumerate(self._alpha_keys):
                    if proxy_name not in group['importance_scores']:
                        continue
                    # 先得到“归一化 raw”
                    normalized_raw = group['importance_scores'][proxy_name] / normalization_denoms[proxy_name]
                    group['_raw_scores'][proxy_name] = normalized_raw.detach().clone()

                    # importance_scores[proxy] 仍可保留“权重后的样子”（保持兼容）
                    group['importance_scores'][proxy_name] = normalized_raw * float(alpha_vec[i].item())

                    if overall is None:
                        overall = group['importance_scores'][proxy_name].clone()
                    else:
                        overall += group['importance_scores'][proxy_name]

                # 记录 overall
                group['importance_scores']['overall'] = overall
                self.global_scores.append(overall)
                global_start_idx += group['num_groups']

    @torch.no_grad()
    def export_kendall_raw_scores(self, refresh=True):
        """
        导出当前时刻用于 Kendall 分析的 5 个准则一维原始分数。

        返回:
            {
                'group_ids': [str, ...],
                'score_map': {
                    'L1': [float, ...],
                    'L2': [float, ...],
                    'Fisher': [float, ...],
                    'GW': [float, ...],
                    'Trend': [float, ...],
                }
            }

        说明:
        - 数据源直接来自 group['_raw_scores']，与内部实际剪枝准则一致。
        - 这里的 Trend 对应内部的 cosine_similarity。
        - 每个元素对应一个真实可剪枝结构组，因此 5 个数组长度一致，均为 total_num_groups。
        """
        if refresh:
            self.compute_grad_variant()
            self.compute_importance_scores()

        proxy_name_map = {
            'L1': 'l1_avg_magnitude',
            'L2': 'l2_avg_magnitude',
            'Fisher': 'fisher',
            'GW': 'grad_weight_product',
            'Trend': 'cosine_similarity',
        }

        score_map = {name: [] for name in proxy_name_map}
        group_ids = []

        for group_idx, group in enumerate(self.param_groups):
            if not group.get('is_prunable', False) or group.get('is_auxiliary', False):
                continue

            raw_scores = group.get('_raw_scores', {})
            if not raw_scores:
                raise RuntimeError('HESSO 当前没有可导出的 `_raw_scores`，请先刷新 importance scores。')

            base_group_id = group.get('id', f'param_group_{group_idx:04d}')
            global_idxes = group.get('global_idxes', np.arange(group['num_groups']))

            missing = [proxy for proxy in proxy_name_map.values() if proxy not in raw_scores]
            if missing:
                raise RuntimeError(
                    f'HESSO 缺少 Kendall 所需原始分数: {missing}，group_id={base_group_id}'
                )

            for local_idx in range(group['num_groups']):
                global_idx = int(global_idxes[local_idx]) if len(global_idxes) > local_idx else local_idx
                group_ids.append(f'{base_group_id}::local_{local_idx:04d}::global_{global_idx:06d}')
                for display_name, proxy_name in proxy_name_map.items():
                    score_tensor = raw_scores[proxy_name]
                    score_map[display_name].append(float(score_tensor[local_idx].detach().item()))

        if len(group_ids) == 0:
            raise RuntimeError('HESSO 未导出任何可剪枝结构组分数，无法执行 Kendall 分析。')

        return {
            'group_ids': group_ids,
            'score_map': score_map,
        }

    # =============== 供 AlphaScheduler 调用的后端接口 ===============

    @torch.no_grad()
    def _reaggregate_overall_from_alpha(self, alpha_vec: torch.Tensor):
        """
        用缓存的“归一化 raw” + 新 α，直接重聚合 overall，不重算 raw。
        """
        # 归一化 α 到单纯形
        alpha = alpha_vec.detach().to(self.device)
        alpha = torch.clamp(alpha, min=1e-8)
        alpha = alpha / alpha.sum()

        global_scores_new = []
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                overall = None
                for i, proxy_name in enumerate(self._alpha_keys):
                    if proxy_name not in group['_raw_scores']:
                        continue
                    contrib = group['_raw_scores'][proxy_name] * float(alpha[i].item())
                    if overall is None:
                        overall = contrib.clone()
                    else:
                        overall += contrib
                    # 为兼容，下次查看 importance_scores[proxy] 也按新 α
                    group['importance_scores'][proxy_name] = contrib.clone()
                group['importance_scores']['overall'] = overall
                global_scores_new.append(overall)
        self.global_scores = global_scores_new

    @torch.no_grad()
    def _dry_pick_active_by_alpha(self, alpha_vec: torch.Tensor):
        """
        在不改内部状态的情况下：用给定 α 计算“候选 overall”并干跑一次 top-K，
        返回 {id(group): [local_idx, ...]} 的映射，仅用于临时缩放评估。
        """
        # 构造“候选 overall”（不写回 importance_scores / global_scores）
        cand_global = []
        per_group_overall = {}
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                overall = None
                for i, proxy_name in enumerate(self._alpha_keys):
                    if proxy_name not in group['_raw_scores']:
                        continue
                    contrib = group['_raw_scores'][proxy_name] * float(alpha_vec[i].item())
                    overall = contrib.clone() if overall is None else (overall + contrib)
                per_group_overall[id(group)] = overall
                cand_global.append(overall)
        if len(cand_global) == 0:
            return {}

        global_scores = torch.cat(cand_global, dim=0)

        # 复用当前 period 的 quota 和“已剪集合”
        curr_active_num = self.active_num_redundant_groups[self.curr_pruning_period]
        curr_K = len(self.pruned_group_idxes) + curr_active_num
        _, top_indices = torch.topk(-global_scores, curr_K)
        top_indices = top_indices.cpu().numpy()
        top_indices = np.setdiff1d(top_indices, self.pruned_group_idxes)[:curr_active_num].tolist()

        # 映射到每个 group 的局部索引
        plan = {}
        global_cursor = 0
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                gs = group['global_start_idx']
                ge = gs + group['num_groups']
                mask = (np.array(top_indices) >= gs) & (np.array(top_indices) < ge)
                picked_global = np.array(top_indices)[mask]
                local = (picked_global - gs).tolist()
                if len(local) > 0:
                    plan[id(group)] = local
                global_cursor += group['num_groups']
        return plan

    @torch.no_grad()
    def _apply_ratio_and_backup(self, group, local_idxes, ratio: float):
        """
        对某个 group 的局部索引做临时缩放（乘以 ratio），并返回备份以便恢复。
        兼容 TRANSPOSE/BASIC；辅助参数按你原训练期逻辑对齐处理。
        """
        backups = []
        for p, p_transform in zip(group['params'], group['p_transform']):
            idx = index_transformation_param_group(local_idxes, p_transform, group)
            if len(idx) == 0:
                continue
            idx_t = torch.as_tensor(idx, dtype=torch.long, device=p.data.device)
            if p_transform == TensorTransform.TRANSPOSE and len(p.data.shape) > 1:
                orig = p.data[:, idx_t, ...].clone()
                p.data[:, idx_t, ...] *= ratio
                backups.append(("param_col", p, idx_t, orig))
            else:
                orig = p.data[idx_t].clone()
                p.data[idx_t] *= ratio
                backups.append(("param_row", p, idx_t, orig))

        # 辅助参数：按 BASIC 对齐 + 边界检查
        for ng_id, offset in group['auxiliary_ngs']:
            aux_local = index_transformation_param_group(local_idxes, TensorTransform.BASIC, group)
            aux_local = [i + offset for i in aux_local]
            for aux_p in self.auxiliary_param_groups[ng_id]['params']:
                if aux_p.data.dim() == 0 or aux_p.data.size(0) == 0:
                    continue
                valid = [i for i in aux_local if 0 <= i < aux_p.data.size(0)]
                if not valid:
                    continue
                idx_t = torch.as_tensor(valid, dtype=torch.long, device=aux_p.data.device)
                orig = aux_p.data[idx_t, ...].clone()
                aux_p.data[idx_t, ...] *= ratio
                backups.append(("aux_row", aux_p, idx_t, orig))
        return backups

    @torch.no_grad()
    def _restore_from_backups(self, backups):
        """
        恢复 _apply_ratio_and_backup 做过的所有临时改动。
        """
        for kind, tensor, idx_t, orig in reversed(backups):
            if kind == "param_col":
                tensor.data[:, idx_t, ...] = orig
            else:
                tensor.data[idx_t] = orig

