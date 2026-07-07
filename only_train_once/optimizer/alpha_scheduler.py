import os
import torch
from dataclasses import dataclass, replace
from typing import Optional, List, Tuple
from torch.utils.data import DataLoader, Subset

Tensor = torch.Tensor

# 使用 dataclass 来定义 Alpha 调度器的配置项
@dataclass
class AlphaScheduleConfig:
    # 优化器参数
    # lr: float = 0.08              # α 的学习率（模拟 SGD 更新）
    lr: float = 0.02                # 通用值
    eps: float = 0.03               # 中心差分法中用于扰动 α 的小量，eps 设为 0.03，0.025 作为次优备选
    momentum: float = 0.0           # α 更新的动量系数（0 表示不使用动量），量项利用历史梯度的移动平均值来更新权重，有助于减少震荡，使优化过程更稳定。
    ema_gamma: float = 0.6          # 对 α 本身的 EMA 平滑系数（0~1，越大越稳定），控制 α 权重本身的指数移动平均，当前新的 α 由 40% 的历史值和 60% 的新计算值混合而成
    g_ema_gamma: float = 0.5        # 对差分梯度 g 的 EMA 平滑（0 表示关闭），控制 差分梯度 g 的指数移动平均（EMA）平滑，当前有效梯度由 50% 的历史梯度平均值 和 50% 的当前估算梯度 混合而成

    # 信赖域限制：0.15~0.22都可用
    max_step_l1: float = 0.18       # 单次 Δα 的 L1 范数上限（防止更新过大，破坏优化的稳定性），设置 α 权重单次更新幅度的信赖域（Trust Region）限制，

    # 探索性评估参数
    eval_drop_ratio: Optional[float] = 0.0  # 评估阶段的剪枝比例（None 表示与训练阶段一致）

    # 调整策略相关参数
    start_after_period: int = 0  # 前几个周期不更新 α
    update_stride: int = 1        # 每隔多少周期更新一次 α
    min_gap_std: float = 0.2      # 当边界 Gap 超过标准差的 0.2 倍时，停止更新 α。

    # 多微批平均策略
    eval_batch_limit: int = 64       # 每次更新最多使用多少个微批
    eval_batch_grow_every: int = 2   # 每隔多少周期，微批数量递增

    # α 评估数据相关超参 
    n_alpha_val: int = 5000          # 从 trainset 采样多少样本作为 α 子集
    alpha_seed: int = 42           # 采样子集的随机种子

    warmup_updates: int = 5        # 预热 α 权重

# 全局默认配置实例
ALPHA_CFG = AlphaScheduleConfig()

def override_alpha_cfg_from_env():
    """
    从环境变量读取若干 α 超参并覆盖 ALPHA_CFG。
    例如:
      export OTO_ALPHA_LR=0.05
      export OTO_ALPHA_EPS=0.01
    """
    global ALPHA_CFG
    mapping = {
        "OTO_ALPHA_LR": ("lr", float),
        "OTO_ALPHA_EPS": ("eps", float),
        "OTO_ALPHA_MAX_STEP_L1": ("max_step_l1", float),
        "OTO_ALPHA_G_EMA_GAMMA": ("g_ema_gamma", float),
        "OTO_ALPHA_EMA_GAMMA": ("ema_gamma", float),
        "OTO_ALPHA_MOMENTUM": ("momentum", float),
        "OTO_ALPHA_START_AFTER_PERIOD": ("start_after_period", int),
        "OTO_ALPHA_UPDATE_STRIDE": ("update_stride", int),
        "OTO_ALPHA_MIN_GAP_STD": ("min_gap_std", float),
        "OTO_ALPHA_EVAL_BATCH_LIMIT": ("eval_batch_limit", int),
        "OTO_ALPHA_WARMUP_UPDATES": ("warmup_updates", int),
    }
    for env_key, (field, cast) in mapping.items():
        if env_key in os.environ:
            try:
                val = cast(os.environ[env_key])
                setattr(ALPHA_CFG, field, val)
            except Exception as e:
                print(f"[AlphaCFG] Failed to parse {env_key}='{os.environ[env_key]}': {e}")

# 模块导入时立即应用一次环境变量覆盖
override_alpha_cfg_from_env()

# 损失引导的 Alpha 调度器类
class LossGuidedAlphaScheduler:
    """
    该类用于学习一组权重 α（重要性代理），通过中心差分法估计梯度并进行优化。
    α 会被投影到单纯形（所有分量非负且和为 1）上。
    """

    def __init__(self, keys, init_weights=None, device="cuda", cfg: AlphaScheduleConfig = None):
        # 初始化设备
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.keys = list(keys)

        # 初始化 α 权重
        if init_weights is None:
            init_weights = torch.ones(len(self.keys), dtype=torch.float32)

        self.alpha = torch.as_tensor(init_weights, dtype=torch.float32, device=self.device)
        self.alpha = torch.clamp(self.alpha, min=1e-8)  # 避免为 0
        self.alpha = self.alpha / self.alpha.sum()      # 归一化至单纯形

        # 初始化动量和梯度 EMA 缓冲
        self.m = torch.zeros_like(self.alpha)      # α 的动量
        self.g_ema = torch.zeros_like(self.alpha)  # 差分梯度的 EMA

        self.backend = None  # 后端接口（由 HESSO 提供）

        if cfg is None:
            self.cfg = replace(ALPHA_CFG)
        elif isinstance(cfg, AlphaScheduleConfig):
            self.cfg = replace(cfg)
        else:
            self.cfg = AlphaScheduleConfig(**cfg)

    # --- 外部接口 ---
    def attach_backend(self, backend):
        # 绑定后端（提供实际模型和优化相关接口）
        self.backend = backend

    def get_alpha(self):
        # 获取当前的 α 权重
        return self.alpha

    def state_dict(self):
        # 保存调度器状态（用于 checkpoint）
        return {
            "alpha": self.alpha.detach().cpu(),
            "m": self.m.detach().cpu(),
            "g_ema": self.g_ema.detach().cpu(),
            "keys": self.keys,
        }

    def load_state_dict(self, sd):
        self.keys = list(sd["keys"])
        self.alpha = torch.as_tensor(sd["alpha"], dtype=torch.float32, device=self.device)
        self.alpha = torch.clamp(self.alpha, min=1e-8)
        self.alpha = self.alpha / self.alpha.sum()
        self.m = torch.as_tensor(sd["m"], dtype=torch.float32, device=self.device)
        self.g_ema = torch.as_tensor(sd.get("g_ema", torch.zeros_like(self.alpha)), dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def update(self, model, criterion, inputs, targets, t_in_period=0):
        """
        使用中心有限差分法更新 α 权重。
        inputs/targets 支持单个张量或微批列表。
        """
        assert self.backend is not None, "AlphaScheduler: backend is not attached."
        cfg = self.cfg

        is_warmup = self.backend.num_steps < self.backend.start_pruning_step

        # 1) 构建梯度 variant
        self.backend.compute_grad_variant()
        self.backend._grad_variant_ready = True

        # 2) 刷新重要性分数
        self.backend.compute_importance_scores()

        # 节流策略：跳过某些周期的更新
        cur_p = self.backend.curr_pruning_period
        if cur_p < cfg.start_after_period or (cur_p % max(1, cfg.update_stride) != 0):
            return

        # 置信门限策略：边界清晰则跳过更新
        s = torch.cat(self.backend.global_scores, dim=0).detach()
        k_new = self.backend.active_num_redundant_groups[self.backend.curr_pruning_period]
        k_all = len(self.backend.pruned_group_idxes) + k_new
        if 1 <= k_all < s.numel():
            s_sorted, _ = torch.sort(s)
            margin = (s_sorted[k_all] - s_sorted[k_all - 1]).abs()
            sigma = s.std().clamp_min(1e-8)
            if (margin / sigma) > cfg.min_gap_std:
                return

        # 设置评估阶段的 drop ratio
        # T = max(1, int(self.backend.pruning_period_duration))
        # den = float(max(1, T - t_in_period))
        # ratio_train = (den - 1.0) / den
        # ratio_eval = cfg.eval_drop_ratio if cfg.eval_drop_ratio is not None else ratio_train

        if is_warmup:
            ratio_eval = 0.7    # 预热阶段使用的剪枝比例
        else:
            ratio_eval = cfg.eval_drop_ratio

        # 选择评估用的微批（动态增长 batchsize）
        if isinstance(inputs, (list, tuple)):
            grow_k = max(0, cur_p - cfg.start_after_period) // max(1, cfg.eval_batch_grow_every)
            want = min(cfg.eval_batch_limit, 1 + grow_k)
            N = len(inputs)
            use_n = min(N, max(1, want))

            if not hasattr(self, "_eval_cursor"):
                self._eval_cursor = 0
            start = self._eval_cursor % N
            idxs = [(start + i) % N for i in range(use_n)]
            eval_batches = [inputs[i] for i in idxs]
            self._eval_cursor += use_n
        else:
            eval_batches = [(inputs, targets)]

        def eval_with_alpha(a_vec: Tensor) -> Tensor:
            self.backend._reaggregate_overall_from_alpha(a_vec)
            plan = self.backend._dry_pick_active_by_alpha(a_vec)

            total_loss = 0.0
            n = 0
            for (vx, vy) in eval_batches:
                backups_all = []
                for group in self.backend.param_groups:
                    if group.get('is_prunable', False) and not group.get('is_auxiliary', False):
                        local = plan.get(id(group), [])
                        if local:
                            backups_all += self.backend._apply_ratio_and_backup(group, local, ratio_eval)

                was_training = model.training
                use_train_fwd = False
                L = None

                # 定义一个内部辅助函数来处理 loss 返回值可能是 tuple 的情况
                def get_loss_value(output, target):
                    ret = criterion(output, target)
                    if isinstance(ret, (tuple, list)):
                        return ret[0]  # YOLO ComputeLoss 返回 (loss, items)
                    return ret

                try:
                    model.eval()
                    if isinstance(vx, dict):
                        out = model(**vx)
                    elif isinstance(vx, (list, tuple)):
                        out = model(*vx)
                    else:
                        out = model(vx)
                    
                    # 修改：使用 get_loss_value 处理 tuple 返回值
                    L = get_loss_value(out, vy).detach()

                except (IndexError, RuntimeError, ValueError) as e:
                    # 捕获 YOLO 在 eval 模式下可能的 build_targets 越界等问题
                    use_train_fwd = True

                if use_train_fwd:
                    model.train()
                    if isinstance(vx, dict):
                        out = model(**vx)
                    elif isinstance(vx, (list, tuple)):
                        out = model(*vx)
                    else:
                        out = model(vx)
                    
                    # 修改：同样使用 get_loss_value
                    L = get_loss_value(out, vy).detach()

                # 恢复原始模式
                if not was_training:
                    model.eval()
                else:
                    model.train()

                if was_training:
                    model.train()

                self.backend._restore_from_backups(backups_all)

                total_loss += L
                n += 1
            return total_loss / max(1, n)

        # 中心差分法估计梯度
        K = len(self.alpha)
        g = torch.zeros_like(self.alpha)
        for k in range(K):
            a_plus  = self.alpha.clone(); a_plus[k]  += cfg.eps
            a_minus = self.alpha.clone(); a_minus[k] -= cfg.eps
            a_plus  = torch.clamp(a_plus,  min=1e-8); a_plus  = a_plus  / a_plus.sum()
            a_minus = torch.clamp(a_minus, min=1e-8); a_minus = a_minus / a_minus.sum()

            Lp = eval_with_alpha(a_plus)
            Lm = eval_with_alpha(a_minus)
            g[k] = (Lp - Lm) / (2.0 * cfg.eps)

        # 梯度 EMA 平滑
        if cfg.g_ema_gamma > 0:
            self.g_ema = cfg.g_ema_gamma * self.g_ema + (1 - cfg.g_ema_gamma) * g
            g_eff = self.g_ema
        else:
            g_eff = g

        # 动量 + SGD 更新
        self.m = cfg.momentum * self.m + (1 - cfg.momentum) * g_eff if cfg.momentum > 0 else g_eff
        raw_new = self.alpha - cfg.lr * self.m
        raw_new = torch.clamp(raw_new, min=1e-8)
        raw_new = raw_new / raw_new.sum()

        # 信赖域裁剪（L1）
        if cfg.max_step_l1 is not None and cfg.max_step_l1 > 0:
            delta = raw_new - self.alpha
            l1 = delta.abs().sum()
            if l1 > cfg.max_step_l1:
                delta = delta * (cfg.max_step_l1 / (l1 + 1e-12))
            raw_new = self.alpha + delta
            raw_new = torch.clamp(raw_new, min=1e-8)
            raw_new = raw_new / raw_new.sum()

        # α EMA 平滑
        if cfg.ema_gamma > 0:
            self.alpha = (1 - cfg.ema_gamma) * self.alpha + cfg.ema_gamma * raw_new
            self.alpha = torch.clamp(self.alpha, min=1e-8)
            self.alpha = self.alpha / self.alpha.sum()
        else:
            self.alpha = raw_new

        # 更新后端的权重和聚合
        for i, k in enumerate(self.keys):
            self.backend.importance_score_criteria[k] = float(self.alpha[i].item())
        self.backend._reaggregate_overall_from_alpha(self.alpha)


# === 下面是给外部脚本使用的工具函数 ===

def build_alpha_loader(
    trainset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    n_alpha_val: Optional[int] = None,
    seed: Optional[int] = None,
):
    """
    从 trainset 固定随机采样若干样本，构造用于 α 更新的 DataLoader。
    注意：这里只用 Subset 视图，原始 trainset 仍可正常参与训练。
    """
    cfg = ALPHA_CFG
    if n_alpha_val is None:
        n_alpha_val = cfg.n_alpha_val
    if seed is None:
        seed = cfg.alpha_seed

    g = torch.Generator().manual_seed(seed)
    take = min(n_alpha_val, len(trainset))
    alpha_indices = torch.randperm(len(trainset), generator=g)[:take].tolist()

    alpha_subset = Subset(trainset, alpha_indices)
    alpha_loader = DataLoader(
        alpha_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return alpha_loader, alpha_subset


@torch.no_grad()
def cache_val_batches(
    loader,
    n: Optional[int],
    device: torch.device,
):
    """
    从给定的 loader 中缓存 n 个微批到指定设备，用于 α 更新。
    会循环遍历 loader，不会消耗原 loader 的数据。
    """
    if n is None:
        n = ALPHA_CFG.eval_batch_limit

    val_batches = []
    it = iter(loader)
    for _ in range(n):
        try:
            vx, vy = next(it)
        except StopIteration:
            it = iter(loader)
            vx, vy = next(it)
        vx = vx.to(device, non_blocking=True)
        vy = vy.to(device, non_blocking=True)
        val_batches.append((vx, vy))
    return val_batches


def prepare_alpha_eval_batches(
    trainset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
):
    """
    使用 ALPHA_CFG 中的超参，一站式准备 α 更新所需的:
      - α-Subset DataLoader
      - 缓存好的 eval 微批 (list[(vx, vy), ...])

    返回:
      val_batches, info_dict
    """
    cfg = ALPHA_CFG

    alpha_loader, alpha_subset = build_alpha_loader(
        trainset=trainset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        n_alpha_val=cfg.n_alpha_val,
        seed=cfg.alpha_seed,
    )

    val_batches = cache_val_batches(
        loader=alpha_loader,
        n=cfg.eval_batch_limit,
        device=device,
    )

    info = {
        "n_alpha_val": len(alpha_subset),
        "alpha_batch_size": batch_size,
        "num_val_microbatches": len(val_batches),
    }
    return val_batches, info
