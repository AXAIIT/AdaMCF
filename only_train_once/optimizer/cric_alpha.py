"""CRIC 周期边界上的损失引导 α 更新，不改变原 HESSO 调度器。"""
import math
from dataclasses import asdict, dataclass, replace
import torch
from .alpha_scheduler import AlphaScheduleConfig, ALPHA_CFG, LossGuidedAlphaScheduler, advance_alpha, alpha_boundary_confident
from .alpha_data import (CRICAlphaPool, AlphaBatch, AlphaEvaluationPool, _to_device,
                         select_alpha_batches, commit_alpha_selection)


@dataclass
class CRICAlphaScheduleConfig(AlphaScheduleConfig):
    """CRIC 独立的取样配置；HESSO 继续使用原 AlphaScheduleConfig。

    数量约束按分层计算。默认不设硬上限，避免无声覆盖百分比；数据太大时
    调用方可以设置上限，并从日志读取实际比例。微批大小只控制显存。
    """
    sampling_policy: str = 'ratio'
    pool_ratio: float = 0.1
    update_ratio: float = 0.5
    align_power_of_two: bool = True
    min_pool_samples: int = 1
    max_pool_samples: int | None = None
    min_update_samples: int = 1
    max_update_samples: int | None = None

    def __post_init__(self):
        if self.sampling_policy not in ('ratio', 'legacy'):
            raise ValueError('CRIC α sampling_policy 必须为 ratio 或 legacy。')
        for name in ('pool_ratio', 'update_ratio'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f'CRIC α {name} 必须在 (0, 1] 内。')
        if not isinstance(self.align_power_of_two, bool):
            raise ValueError('CRIC α align_power_of_two 必须为 bool。')
        for role in ('pool', 'update'):
            low, high = getattr(self, f'min_{role}_samples'), getattr(self, f'max_{role}_samples')
            if not isinstance(low,int) or isinstance(low,bool) or low < 1:
                raise ValueError('CRIC α 样本下限必须为正整数。')
            if high is not None and (not isinstance(high,int) or isinstance(high,bool) or high < low):
                raise ValueError('CRIC α 样本上限必须为不低于下限的整数。')


def make_cric_alpha_config(value=None, *, checkpoint=False):
    """新建默认使用百分比；显式旧配置对象/旧检查点保留原渐增策略。"""
    if isinstance(value, CRICAlphaScheduleConfig):
        return replace(value)
    if isinstance(value, AlphaScheduleConfig):
        return CRICAlphaScheduleConfig(**asdict(value), sampling_policy='legacy')
    values = asdict(ALPHA_CFG) if value is None else dict(value)
    if checkpoint and 'sampling_policy' not in values:
        values['sampling_policy'] = 'legacy'
    return CRICAlphaScheduleConfig(**values)


class CRICAlphaScheduler(LossGuidedAlphaScheduler):
    """复用 α 的数值更新规则，为 CRIC 提供独立的触发和评估协议。

    调用位置与 HESSO 相同：训练批次 backward 后、optimizer.step 前。
    inputs 可以是单个输入，也可以是 [(inputs_dict, labels), ...] 微批列表。
    criterion=None 时，使用 Hugging Face 输出的 loss；否则由 criterion 处理输出。
    """

    def __init__(self, *args, **kwargs):
        if len(args) < 4:
            kwargs['cfg'] = make_cric_alpha_config(kwargs.get('cfg'))
        super().__init__(*args, **kwargs)
        self.data_pool = None
        self._pending_pool_state = None
        self.last_skip_reason = None
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
        result['data_pool'] = self.data_pool.state_dict() if isinstance(self.data_pool, CRICAlphaPool) else self._pending_pool_state
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
        self._pending_pool_state = state.get('data_pool')

    def attach_data_pool(self, pool):
        if self._pending_pool_state is not None:
            if not isinstance(pool, CRICAlphaPool):
                raise ValueError('检查点含百分比池状态，恢复时必须提供 CRICAlphaPool。')
            pool.load_state_dict(self._pending_pool_state)
            self._pending_pool_state = None
        self.data_pool = pool

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
    def update(self, model, criterion, inputs=None, targets=None, t_in_period=0):
        backend = self.backend
        if backend is None:
            raise RuntimeError('CRIC α: 尚未绑定优化器。')
        if inputs is None:
            inputs = self.data_pool
        if isinstance(inputs, CRICAlphaPool):
            self.attach_data_pool(inputs)
        context = backend.alpha_update_context()
        self.last_skip_reason = None
        if context is None or context['step'] == self._last_update_step:
            self.last_skip_reason = 'not_boundary' if context is None else 'already_updated'
            return False
        cfg = self.cfg
        self.last_eval_stats = None
        cycle = context['round']
        if cycle < cfg.start_after_period or cycle % max(1, cfg.update_stride):
            self.last_skip_reason = 'stride_or_start'
            return False

        # 只预览下一步梯度变体；不推进 Adam/SGD 动量，真实 step 只更新一次状态。
        backend.preview_grad_variant()
        backend.compute_importance_scores()
        scores = torch.cat(backend.global_scores).detach()
        k = backend.target_num_redundant_groups
        if alpha_boundary_confident(scores,k,cfg.min_gap_std):
            self.last_skip_reason = 'confident_boundary'
            return False

        pool_state = inputs.state_dict() if isinstance(inputs, CRICAlphaPool) else None
        try:
            if getattr(cfg, 'sampling_policy', 'legacy') == 'ratio':
                if isinstance(inputs, CRICAlphaPool):
                    selected = inputs.take(cfg)
                    selected = [AlphaBatch(_to_device(b.inputs,self.device), _to_device(b.targets,self.device),
                                           b.sample_ids,b.supervised_tokens) for b in selected]
                else:
                    # 旧模型传批次缓存时只能按批次百分比回退，不能声称精确控制
                    # 独立样本量。新入口应使用 CRICAlphaPool 的样本级协议。
                    size = len(inputs) if isinstance(inputs,(AlphaEvaluationPool,list,tuple)) else 1
                    count = max(1, math.ceil(size * cfg.update_ratio))
                    fixed = replace(cfg, eval_batch_limit=count, eval_batch_grow_every=1,
                                    start_after_period=0)
                    selected = select_alpha_batches(inputs,targets,count-1,fixed,self.device,self._eval_cursor)
            else:
                selected = select_alpha_batches(inputs,targets,cycle,cfg,self.device,self._eval_cursor)
        except BaseException:
            if pool_state is not None:
                inputs.load_state_dict(pool_state)
            raise
        count = len(selected)
        ratio = 0.7 if context['warmup'] else (cfg.eval_drop_ratio or 0.0)
        old_alpha = self.alpha.clone()

        def evaluate(alpha):
            backend._reaggregate_overall_from_alpha(alpha)
            plan = backend._dry_pick_active_by_alpha(alpha)
            values = []
            for batch in selected:
                vx,vy=batch.inputs,batch.targets
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
            # 与 HESSO 保持同一累加顺序，不额外引入 mean kernel 的舍入差异。
            total=0.0
            for value in values:total+=value
            return total/len(values)

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

            new_alpha,new_m,new_g_ema=advance_alpha(self.alpha,self.m,self.g_ema,gradient,cfg)
            if not torch.isfinite(new_alpha).all():
                raise ValueError('CRIC α: 更新后的权重不是有限值。')
        except BaseException:
            backend._reaggregate_overall_from_alpha(old_alpha)
            if pool_state is not None:
                inputs.load_state_dict(pool_state)
            raise

        self.alpha, self.m, self.g_ema = new_alpha, new_m, new_g_ema
        for index, name in enumerate(self.keys):
            backend.importance_score_criteria[name] = float(self.alpha[index])
        backend._reaggregate_overall_from_alpha(self.alpha)
        self._eval_cursor += count
        self._last_update_step = context['step']
        self.update_count += 1
        self.last_eval_stats=commit_alpha_selection(inputs,selected)
        if getattr(cfg, 'sampling_policy', 'legacy') == 'ratio':
            self.last_eval_stats.update(sampling_policy='ratio',
                selection_unit='sample' if isinstance(inputs,CRICAlphaPool) else 'microbatch',
                alpha_gradient=gradient.detach().cpu().tolist(),
                alpha_delta_l1=float((self.alpha-old_alpha).abs().sum()),
                alpha_value_changed=not torch.equal(self.alpha,old_alpha),
                finite_difference_forward_batches=2*len(self.alpha)*count)
        return True
