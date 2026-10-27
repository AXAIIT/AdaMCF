"""AdaMCF 五准则/α 与 CRIC 的兼容层；原 HESSO 和模型算子保持原样。"""
from dataclasses import asdict, replace
import copy
import math
import torch
from torch.optim.optimizer import required
from .alpha_scheduler import ALPHA_CFG, AlphaScheduleConfig
from .base_hybrid_sparse_optimizer import BaseHybridSparseOptimizer
from .cric_alpha import CRICAlphaScheduler
from .hesso import HESSO
from only_train_once.transform import (
    TensorTransform, index_transformation_param_group, pruning_axis,
)


class AdaptiveCRICMixin:
    """只覆盖需要适配的行为，预热/投影/采样/终止的阶段顺序沿用 CRIC。"""

    _reaggregate_overall_from_alpha = HESSO._reaggregate_overall_from_alpha
    _STATE_VERSION = 1
    _CONTROL_FIELDS = (
        'num_steps', 'curr_cycle_period', 'is_cric_terminated',
        'redundant_groups_identified', 'cric_terminated_step', 'ref_loss',
        'target_num_redundant_groups', 'target_group_sparsity', 'start_cric_step',
        'start_global_sampling_step', 'sampling_steps', 'hybrid_training_steps',
        'max_cycle_period', 'tolerance', 'group_divisible',
    )

    def __init__(self, params, variant='sgd', lr=required, first_momentum=None,
                 second_momentum=None, dampening=None, weight_decay=None,
                 target_group_sparsity=0.5, tolerance=0, group_divisible=1,
                 start_cric_step=0, max_cycle_period=10, sampling_steps=None,
                 hybrid_training_steps=None, importance_score_criteria='default',
                 *, device=None, alpha_scheduler_cfg=None, adaptive_alpha=True):
        params = list(params)
        if not params or not any(group['params'] for group in params):
            raise ValueError('CRIC 需要非空参数组。')
        if not 0 <= target_group_sparsity < 1:
            raise ValueError('target_group_sparsity 必须在 [0, 1) 内。')
        for name, value in [('sampling_steps', sampling_steps),
                            ('hybrid_training_steps', hybrid_training_steps),
                            ('group_divisible', group_divisible),
                            ('max_cycle_period', max_cycle_period)]:
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} 必须显式设置为正整数。')
        if not isinstance(start_cric_step, int) or start_cric_step < 0:
            raise ValueError('start_cric_step 必须为非负整数。')
        if importance_score_criteria is None or importance_score_criteria == 'default':
            criteria = dict(l1_avg_magnitude=.2, l2_avg_magnitude=.2,
                            cosine_similarity=.2, fisher=.2, grad_weight_product=.2)
            criteria['loss'] = 1.0
        else:
            aliases = {'magnitude': 'l2_magnitude', 'avg_magnitude': 'l2_avg_magnitude'}
            criteria = {}
            for key, value in dict(importance_score_criteria).items():
                key = aliases.get(key, key)
                if key in criteria:
                    raise ValueError(f'评分别名重复：{key}')
                criteria[key] = float(value)
            criteria.setdefault('loss', 1.0)
        supported = {'l1_magnitude', 'l1_avg_magnitude', 'l2_magnitude',
                     'l2_avg_magnitude', 'cosine_similarity', 'fisher',
                     'grad_weight_product', 'taylor_first_order', 'taylor_second_order'}
        score_keys = [key for key in criteria if key != 'loss']
        if not score_keys or not set(score_keys) <= supported:
            raise ValueError(f'CRIC 不支持的评分项：{set(score_keys) - supported}')
        if any(not math.isfinite(value) or value < 0 for value in criteria.values()):
            raise ValueError('评分及 loss 校正权重必须为有限非负数。')
        if sum(criteria[key] for key in score_keys) <= 0:
            raise ValueError('至少一个评分权重必须为正。')
        actual_devices = {p.device for group in params for p in group['params']}
        if len(actual_devices) != 1:
            raise ValueError('CRIC 当前要求参数组位于同一个设备。')
        actual = next(iter(actual_devices))
        requested = torch.device(device) if device is not None else actual
        if requested.type != actual.type or requested.index is not None and requested.index != actual.index:
            raise ValueError('CRIC 的 device 与参数所在设备不一致。')
        self.device = actual
        self._alpha_keys = score_keys
        self.adaptive_alpha = bool(adaptive_alpha)

        # 外部 step 从 1 计数；0 表示首步开始，避免旧代码跳过参考损失初始化。
        super().__init__(params=params, variant=variant, lr=lr,
                         first_momentum=first_momentum, second_momentum=second_momentum,
                         dampening=dampening, weight_decay=weight_decay,
                         target_group_sparsity=target_group_sparsity, tolerance=tolerance,
                         group_divisible=group_divisible, start_cric_step=max(1, start_cric_step),
                         max_cycle_period=max_cycle_period, sampling_steps=sampling_steps,
                         hybrid_training_steps=hybrid_training_steps,
                         importance_score_criteria=criteria)
        for group in self.param_groups:
            group['_raw_scores'] = {}
            if group['is_prunable'] and not group['is_auxiliary']:
                # 最终重融合只需每周期的未加权分数和/计数，无须保存每步 GPU 副本。
                group['raw_score_sums'] = {}
                group['raw_score_counts'] = {}
        cfg = (replace(ALPHA_CFG) if alpha_scheduler_cfg is None else
               replace(alpha_scheduler_cfg) if isinstance(alpha_scheduler_cfg, AlphaScheduleConfig)
               else AlphaScheduleConfig(**alpha_scheduler_cfg))
        self.alpha_scheduler = CRICAlphaScheduler(
            keys=score_keys, init_weights=[criteria[k] for k in score_keys],
            device=self.device, cfg=cfg)
        self.alpha_scheduler.attach_backend(self)
        for key, value in zip(score_keys, self.alpha_scheduler.alpha):
            self.importance_score_criteria[key] = float(value)

    def alpha_update_context(self):
        """返回下一训练步的 α 评估轮次；CRIC 周期内返回 None，保持 α 冻结。"""
        if not self.adaptive_alpha or self.is_cric_terminated or not self.target_num_redundant_groups:
            return None
        step = self.num_steps + 1
        if step < self.start_cric_step:
            updates = self.alpha_scheduler.cfg.warmup_updates
            if updates <= 0 or step % max(1, self.start_cric_step // updates):
                return None
            return {'step': step, 'round': 0, 'warmup': True}
        if step == self.start_cric_step:
            return {'step': step, 'round': 0, 'warmup': False}
        if step >= self.start_global_sampling_step:
            offset = step - self.start_global_sampling_step
            cycle = 1 + offset // self.sampling_steps
            if offset % self.sampling_steps == 0 and cycle <= self.max_cycle_period:
                return {'step': step, 'round': cycle, 'warmup': False}
        return None

    def will_update_alpha_on_next_step(self):
        return self.alpha_update_context() is not None

    def will_enter_new_period_on_next_step(self):
        """兼容既有训练循环；这里指 α 的 CRIC 边界，不是 HESSO 预算周期。"""
        return self.will_update_alpha_on_next_step()

    @torch.no_grad()
    def preview_grad_variant(self):
        """计算与下一真实 step 相同的梯度变体，但不修改动量或步数。

        BaseOptimizer 的首个动量样本直接采用梯度，故预览也保留该既有约定，
        而不是在适配 CRIC 时改变整个框架的 Adam/SGD 数值规则。
        """
        step = self.num_steps + 1
        def moment(store, name, beta, dampening, value):
            # 克隆后沿用基类的原地运算顺序，避免预览引入额外舍入差异。
            if beta > 0 and name in store:
                return store[name].clone().mul_(beta).add_(value, alpha=1 - dampening)
            return value
        for i, group in enumerate(self.param_groups):
            adam = group['variant'] in ('adam', 'adamw')
            group['grad_variant'] = {}
            for j, (name, p) in enumerate(zip(group['p_names'], group['params'])):
                if p.grad is None:
                    continue
                grad = p.grad.detach().clone()
                if group['weight_decay'] is not None and group['variant'] != 'adamw':
                    grad += group['weight_decay'] * p.detach()
                first_name = f'grad_first_moment_buffer_group_{i}_param_{j}'
                if not adam:
                    result = moment(self.first_moment_grads, first_name,
                                    group['first_momentum'], group['dampening'], grad)
                else:
                    b1, b2 = group['first_momentum'], group['second_momentum']
                    first = moment(self.first_moment_grads, first_name, b1, b1, grad)
                    second = moment(self.second_moment_grads,
                                    f'grad_second_moment_buffer_group_{i}_param_{j}',
                                    b2, b2, grad.square())
                    result = (first / (1 - b1 ** step)) / ((second / (1 - b2 ** step)).sqrt() + self.safe_guard)
                group['grad_variant'][name] = result

    @torch.no_grad()
    def compute_importance_scores(self, **kwargs):
        # 复用已验证的五准则计算和全局归一化，不调用 HESSO 的构造或时间流程。
        HESSO.compute_importance_scores(self, **kwargs)

    @torch.no_grad()
    def _dry_pick_active_by_alpha(self, alpha_vec):
        self._reaggregate_overall_from_alpha(alpha_vec)
        scores = torch.cat(self.global_scores)
        count = min(self.target_num_redundant_groups, scores.numel())
        chosen = set(torch.topk(-scores, count).indices.cpu().tolist())
        return {id(group): [i for i in range(group['num_groups'])
                            if group['global_start_idx'] + i in chosen]
                for group in self.param_groups
                if group['is_prunable'] and not group['is_auxiliary']}

    def _pruning_slices(self, group, indexes):
        slices = []
        for p, transform in zip(group['params'], group['p_transform']):
            if transform in (TensorTransform.NO_PRUNE, TensorTransform.NO_UPDATE):
                continue
            physical = index_transformation_param_group(indexes, transform, group)
            if not physical:
                continue
            axis = pruning_axis(transform) % p.ndim
            if min(physical) < 0 or max(physical) >= p.shape[axis]:
                raise IndexError(f'CRIC 参数物理索引越界：{group["id"]}')
            slices.append((p, axis, torch.tensor(physical, device=p.device, dtype=torch.long)))
        # 辅助张量的合同是逻辑组 + offset，不能复用某个主参数展开后的物理索引。
        for group_id, offset in group['auxiliary_ngs']:
            for p in self.auxiliary_param_groups[group_id]['params']:
                if p.ndim == 0:
                    continue
                physical = sorted({int(i) + offset for i in indexes if 0 <= int(i) + offset < p.shape[0]})
                if physical:
                    slices.append((p, 0, torch.tensor(physical, device=p.device, dtype=torch.long)))
        return slices

    @torch.no_grad()
    def _scale_group(self, group, indexes, ratio):
        for p, axis, physical in self._pruning_slices(group, indexes):
            p.data.index_copy_(axis, physical, p.data.index_select(axis, physical) * ratio)

    def fix_pruned_groups_as_zeros(self, group):
        # 训练结束后的冻结也使用相同索引合同，保留 NO_PRUNE/NO_UPDATE 参数。
        self._scale_group(group, group['pruned_idxes'], 0.0)

    @torch.no_grad()
    def _apply_ratio_and_backup(self, group, indexes, ratio):
        backups = []
        try:
            for p, axis, physical in self._pruning_slices(group, indexes):
                saved = p.data.index_select(axis, physical).clone()
                backups.append((p, axis, physical, saved))
                p.data.index_copy_(axis, physical, saved * ratio)
        except Exception:
            # 即使本组尚未返回备份列表，部分参数已缩放后出错也必须恢复。
            self._restore_from_backups(backups)
            raise
        return backups

    @torch.no_grad()
    def _restore_from_backups(self, backups):
        for p, axis, physical, saved in reversed(backups):
            p.data.index_copy_(axis, physical, saved)

    def proj_trial_group_sparsity(self, param_group, trial_group_sparsity):
        count = max(1, min(int(param_group['num_groups'] * trial_group_sparsity), param_group['num_groups']))
        indexes = torch.topk(-param_group['importance_scores']['overall'], count).indices.cpu().tolist()
        self._scale_group(param_group, indexes, 0.0)

    def cric_step(self):
        self.compute_importance_scores()
        self.commit_important_scores(self.curr_cycle_period)
        if (self.num_steps - self.start_global_sampling_step) % self.sampling_steps == 0:
            self.curr_cycle_period += 1
            self.update_violating_set(self.curr_cycle_period)
            self.update_historical_violating_set()
            self.reset_params()
        self.update_trial_violating_set()
        t = (self.num_steps - self.start_global_sampling_step - 1) % self.sampling_steps
        ratio = (self.sampling_steps - t - 1.0) / (self.sampling_steps - t)
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                self.gradient_descent_step(group)
                # 采样阶段同样必须展开多头/INDEX_GROUPS 的物理索引。
                self._scale_group(group, group['active_violating_idxes'], ratio)

    def hybrid_step(self):
        t = self.num_steps - self.cric_terminated_step - 1
        end = self.cric_terminated_step + self.hybrid_training_steps
        for group in self.param_groups:
            self.gradient_descent_step(group)
            if group['is_prunable'] and group['active_redundant_idxes'] and self.num_steps <= end:
                self._scale_group(group, group['active_redundant_idxes'],
                                  (self.hybrid_training_steps - t - 1.0) / (self.hybrid_training_steps - t))
            if self.num_steps == end:
                group['pruned_idxes'].extend(group['active_redundant_idxes'])
                group['active_redundant_idxes'].clear()
            if self.num_steps > end:
                self.fix_pruned_groups_as_zeros(group)

    def step(self, loss=None, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        next_step = self.num_steps + 1
        if self.target_num_redundant_groups == 0:
            self.num_steps += 1
            self.compute_grad_variant()
            return self.basic_step()
        if self.start_cric_step <= next_step < self.start_global_sampling_step:
            if loss is None or (isinstance(loss, torch.Tensor) and loss.numel() != 1):
                raise ValueError('CRIC 投影阶段必须传入标量 loss，例如 step(loss=loss.detach())。')
            value = float(loss.detach()) if isinstance(loss, torch.Tensor) else float(loss)
            if not math.isfinite(value):
                raise ValueError('CRIC loss 必须为有限值。')
            loss = torch.as_tensor(value, device=self.device)
        return super().step(loss=loss)

    def proj_step(self, loss):
        # 保留原投影轮次，只为零参考损失提供稳定分母。
        if self.ref_loss == 0:
            self.ref_loss = self.safe_guard
        return super().proj_step(loss)

    @torch.no_grad()
    def commit_important_scores(self, cycle_period):
        for group in self.param_groups:
            if not group['is_prunable'] or group['is_auxiliary']:
                continue
            sums = group['raw_score_sums'].setdefault(cycle_period, {})
            for key in self._alpha_keys:
                value = group['_raw_scores'][key].detach().float().cpu()
                sums[key] = sums[key] + value if key in sums else value.clone()
            counts = group['raw_score_counts']
            counts[cycle_period] = counts.get(cycle_period, 0) + 1

    @torch.no_grad()
    def compute_accumulate_saliency_score(self):
        for group in self.param_groups:
            if not group['is_prunable'] or group['is_auxiliary']:
                continue
            count = sum(group['raw_score_counts'].values())
            if not count:
                raise RuntimeError('CRIC 没有可用于最终融合的评分历史。')
            score = torch.zeros(group['num_groups'], device=self.device)
            # 先平均未加权的归一化分数，再用最终 α 统一融合；不混合旧 overall。
            for key in self._alpha_keys:
                mean = sum(cycle[key] for cycle in group['raw_score_sums'].values()) / count
                score.add_(mean.to(self.device), alpha=self.importance_score_criteria[key])
            for cycle, indexes in group['active_violating_idxes_collection'].items():
                losses = group['loss_collection'][cycle]
                if indexes and losses:
                    score[indexes] += self.importance_score_criteria['loss'] * sum(losses) / len(losses) / len(indexes)
            group['accumulated_importance_score'] = score

    def state_dict(self):
        result = super().state_dict()
        result['cric_state'] = {
            'version': self._STATE_VERSION,
            'control': {k: getattr(self, k) for k in self._CONTROL_FIELDS if hasattr(self, k)},
            'criteria': dict(self.importance_score_criteria),
            'keys': list(self._alpha_keys), 'adaptive_alpha': self.adaptive_alpha,
            'cache_parameters': self.cache_parameters,
            'first_moment_grads': self.first_moment_grads,
            'second_moment_grads': self.second_moment_grads,
            'alpha_config': asdict(self.alpha_scheduler.cfg),
            'alpha_scheduler': self.alpha_scheduler.state_dict(),
        }
        return result

    def load_state_dict(self, state_dict):
        state = copy.deepcopy(state_dict)
        extra = state.pop('cric_state', None)
        if extra is None or extra.get('version') != self._STATE_VERSION:
            raise ValueError('检查点缺少完整 CRIC 状态，不能作为可继续训练的检查点加载。')
        if extra['keys'] != self._alpha_keys:
            raise ValueError('检查点评分项与当前 CRIC 配置不一致。')
        for old, new in zip(state['param_groups'], self.param_groups):
            if old['p_names'] != new['p_names'] or old['num_groups'] != new['num_groups']:
                raise ValueError('检查点参数组布局不一致。')
        super().load_state_dict(state)
        for key, value in extra['control'].items():
            setattr(self, key, value)
        self.importance_score_criteria = extra['criteria']
        self.adaptive_alpha = extra['adaptive_alpha']
        for name in ('cache_parameters', 'first_moment_grads', 'second_moment_grads'):
            setattr(self, name, {key: value.to(self.device) for key, value in extra[name].items()})
        self.alpha_scheduler.cfg = AlphaScheduleConfig(**extra['alpha_config'])
        self.alpha_scheduler.load_state_dict(extra['alpha_scheduler'])
        self.prunable_param_group_dict = {g['id']: g for g in self.param_groups if g['is_prunable'] and not g['is_auxiliary']}
        self.param_group_ids = list(self.prunable_param_group_dict)
        self.auxiliary_param_groups = {g['id']: g for g in self.param_groups if g['is_auxiliary']}
        for group in self.param_groups:
            group['grad_variant'] = {}
            group['_raw_scores'] = {}
            if group['is_prunable'] and not group['is_auxiliary']:
                # torch.load(map_location='cuda') 及 Optimizer.load_state_dict
                # 都可能把历史搬到参数设备；评分累计合同始终是 CPU float32。
                group['raw_score_sums'] = {
                    cycle: {key: value.detach().float().cpu() for key, value in scores.items()}
                    for cycle, scores in group['raw_score_sums'].items()
                }

    def compute_metrics(self, **summary_stats):
        # 共用指标保留原模型适配；CRIC 的额外统计仍由原实现计算。
        BaseHybridSparseOptimizer.compute_metrics(self, **summary_stats)
        return super().compute_metrics()
