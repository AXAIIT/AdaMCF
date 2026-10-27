"""CRIC 周期边界上的损失引导 α 更新，不改变原 HESSO 调度器。"""
import math
import torch
from .alpha_scheduler import LossGuidedAlphaScheduler


class CRICAlphaScheduler(LossGuidedAlphaScheduler):
    """复用 α 的数值更新规则，为 CRIC 提供独立的触发和评估协议。

    调用位置与 HESSO 相同：训练批次 backward 后、optimizer.step 前。
    inputs 可以是单个输入，也可以是 [(inputs_dict, labels), ...] 微批列表。
    criterion=None 时，使用 Hugging Face 输出的 loss；否则由 criterion 处理输出。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_count = 0
        self._last_update_step = -1
        self._eval_cursor = 0
        if self.cfg.eps <= 0 or self.cfg.eval_batch_limit < 1:
            raise ValueError('CRIC α: eps 必须为正，eval_batch_limit 必须至少为 1。')
        if not math.isfinite(self.cfg.eps) or not math.isfinite(self.cfg.lr) or self.cfg.lr < 0:
            raise ValueError('CRIC α: eps、lr 必须有限，lr 必须非负。')
        for name in ('momentum', 'ema_gamma', 'g_ema_gamma'):
            value = getattr(self.cfg, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'CRIC α: {name} 必须在 [0, 1] 内。')
        if self.cfg.eval_drop_ratio is not None and not 0 <= self.cfg.eval_drop_ratio <= 1:
            raise ValueError('CRIC α: eval_drop_ratio 必须在 [0, 1] 内。')

    def state_dict(self):
        result = super().state_dict()
        result.update(update_count=self.update_count,
                      last_update_step=self._last_update_step,
                      eval_cursor=self._eval_cursor)
        return result

    def load_state_dict(self, state):
        saved_alpha = torch.as_tensor(state['alpha'], device=self.device).clone()
        if (saved_alpha.shape != self.alpha.shape
                or not torch.isfinite(saved_alpha).all() or not (saved_alpha > 0).all()
                or abs(saved_alpha.sum().item() - 1) > 1e-6):
            raise ValueError('CRIC α 检查点的权重必须为正、有限且总和接近 1。')
        super().load_state_dict(state)
        # 已通过验证的检查点不再次归一化，避免恢复时产生 1 ULP 漂移。
        self.alpha = saved_alpha
        self.update_count = int(state.get('update_count', 0))
        self._last_update_step = int(state.get('last_update_step', -1))
        self._eval_cursor = int(state.get('eval_cursor', 0))

    @staticmethod
    def _loss(model, criterion, inputs, targets):
        if isinstance(inputs, dict):
            output = model(**inputs)
        elif isinstance(inputs, (list, tuple)):
            output = model(*inputs)
        else:
            output = model(inputs)
        if criterion is None:
            value = output.get('loss') if isinstance(output, dict) else getattr(output, 'loss', None)
            if value is None:
                raise ValueError('CRIC α: criterion=None 时模型输出必须包含 loss。')
        else:
            value = criterion(output, targets)
        if isinstance(value, (tuple, list)):
            value = value[0]
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise ValueError('CRIC α: 评估损失必须是标量 Tensor。')
        if not torch.isfinite(value).all():
            raise ValueError('CRIC α: 评估损失不是有限值。')
        return value.detach().float().reshape(())

    @torch.no_grad()
    def update(self, model, criterion, inputs, targets=None, t_in_period=0):
        backend = self.backend
        if backend is None:
            raise RuntimeError('CRIC α: 尚未绑定优化器。')
        context = backend.alpha_update_context()
        if context is None or context['step'] == self._last_update_step:
            return False
        cfg = self.cfg
        cycle = context['round']
        if cycle < cfg.start_after_period or cycle % max(1, cfg.update_stride):
            return False

        # 只预览下一步梯度变体；不推进 Adam/SGD 动量，真实 step 只更新一次状态。
        backend.preview_grad_variant()
        backend.compute_importance_scores()
        scores = torch.cat(backend.global_scores).detach()
        k = backend.target_num_redundant_groups
        if 0 < k < scores.numel():
            ordered = scores.sort().values
            gap = (ordered[k] - ordered[k - 1]).abs()
            if gap / scores.std(unbiased=False).clamp_min(1e-8) > cfg.min_gap_std:
                return False

        batches = list(inputs) if isinstance(inputs, (list, tuple)) else [(inputs, targets)]
        if not batches or any(not isinstance(b, (list, tuple)) or len(b) != 2 for b in batches):
            raise ValueError('CRIC α: 微批必须是非空的 (inputs, targets) 列表。')
        count = min(len(batches), cfg.eval_batch_limit,
                    1 + max(0, cycle - cfg.start_after_period) // max(1, cfg.eval_batch_grow_every))
        selected = [batches[(self._eval_cursor + i) % len(batches)] for i in range(count)]
        ratio = 0.7 if context['warmup'] else (cfg.eval_drop_ratio or 0.0)
        old_alpha = self.alpha.clone()

        def evaluate(alpha):
            backend._reaggregate_overall_from_alpha(alpha)
            plan = backend._dry_pick_active_by_alpha(alpha)
            values = []
            for vx, vy in selected:
                backups = []
                modes = [(module, module.training) for module in model.modules()]
                buffers = [(buf, buf.detach().clone()) for buf in model.buffers()]
                devices = [self.device.index or 0] if self.device.type == 'cuda' else []
                try:
                    for group in backend.param_groups:
                        indexes = plan.get(id(group), [])
                        if indexes:
                            backups.extend(backend._apply_ratio_and_backup(group, indexes, ratio))
                    # 评估与检测模型的 train 回退都不能污染随机序列或 BatchNorm 状态。
                    with torch.random.fork_rng(devices=devices):
                        model.eval()
                        try:
                            loss = self._loss(model, criterion, vx, vy)
                        except (IndexError, RuntimeError, ValueError) as error:
                            if isinstance(error, torch.cuda.OutOfMemoryError):
                                raise
                            model.train()
                            loss = self._loss(model, criterion, vx, vy)
                    values.append(loss)
                finally:
                    backend._restore_from_backups(backups)
                    for buffer, saved in buffers:
                        buffer.copy_(saved)
                    for module, was_training in modes:
                        module.training = was_training
            return torch.stack(values).mean()

        try:
            gradient = torch.zeros_like(self.alpha)
            for index in range(len(self.alpha)):
                plus, minus = self.alpha.clone(), self.alpha.clone()
                plus[index] += cfg.eps
                minus[index] -= cfg.eps
                plus = plus.clamp_min(1e-8); plus = plus / plus.sum()
                minus = minus.clamp_min(1e-8); minus = minus / minus.sum()
                gradient[index] = (evaluate(plus) - evaluate(minus)) / (2 * cfg.eps)
            if not torch.isfinite(gradient).all():
                raise ValueError('CRIC α: 有限差分梯度不是有限值。')

            # 下列 EMA、动量、单纯形归一化及 L1 信赖域规则与 HESSO 一致。
            new_g_ema = cfg.g_ema_gamma * self.g_ema + (1 - cfg.g_ema_gamma) * gradient
            effective = new_g_ema if cfg.g_ema_gamma > 0 else gradient
            new_m = cfg.momentum * self.m + (1 - cfg.momentum) * effective if cfg.momentum > 0 else effective
            proposal = (self.alpha - cfg.lr * new_m).clamp_min(1e-8)
            proposal /= proposal.sum()
            if cfg.max_step_l1 is not None and cfg.max_step_l1 > 0:
                delta = proposal - self.alpha
                delta *= min(1.0, cfg.max_step_l1 / (delta.abs().sum().item() + 1e-12))
                proposal = (self.alpha + delta).clamp_min(1e-8)
                proposal /= proposal.sum()
            new_alpha = ((1 - cfg.ema_gamma) * self.alpha + cfg.ema_gamma * proposal
                         if cfg.ema_gamma > 0 else proposal)
            new_alpha = new_alpha.clamp_min(1e-8); new_alpha /= new_alpha.sum()
            if not torch.isfinite(new_alpha).all():
                raise ValueError('CRIC α: 更新后的权重不是有限值。')
        except Exception:
            backend._reaggregate_overall_from_alpha(old_alpha)
            raise

        self.alpha, self.m = new_alpha, new_m
        if cfg.g_ema_gamma > 0:
            self.g_ema = new_g_ema
        for index, name in enumerate(self.keys):
            backend.importance_score_criteria[name] = float(self.alpha[index])
        backend._reaggregate_overall_from_alpha(self.alpha)
        self._eval_cursor += count
        self._last_update_step = context['step']
        self.update_count += 1
        return True
