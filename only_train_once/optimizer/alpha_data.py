"""模型无关的 α 评估批次协议。HESSO/CRIC 共用取批和覆盖记录。"""
from dataclasses import dataclass
from collections.abc import Mapping
from collections import defaultdict
import copy
import hashlib
import math
import torch
from torch.utils.data import Subset, default_collate


@dataclass
class AlphaBatch:
    """模型脚本只负责把任务数据适配到此协议，不决定 α 的取批策略。"""
    inputs: object
    targets: object = None
    sample_ids: tuple = ()
    supervised_tokens: int = 0


def _to_device(value, device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,Mapping):return {k:_to_device(v,device) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(_to_device(v,device) for v in value)
    if isinstance(value,list):return [_to_device(v,device) for v in value]
    return value


def _as_batch(item):
    if isinstance(item,AlphaBatch):return item
    if isinstance(item,dict):return AlphaBatch(item)
    if isinstance(item,(tuple,list)) and len(item)==2:return AlphaBatch(*item)
    raise ValueError('α 批次必须为 AlphaBatch、输入 dict 或 (inputs, targets)。')


class AlphaEvaluationPool:
    """持续遍历调用方提供的 loader，仅将本次评估批次搬到设备。

    loader 可用独立种子的随机 sampler，或固定顺序。数据集选择/切分归调用方，
    α 的取批数量、评估内复用和累计覆盖归优化器。旧缓存列表仍支持。
    """
    def __init__(self, loader):
        if len(loader)==0:raise ValueError('α loader 不能为空。')
        self.loader=loader
        self.iterator=iter(loader)
        self.used_sample_ids=set()

    def __len__(self):return len(self.loader)

    def take(self, count):
        result=[]
        for _ in range(count):
            try:item=next(self.iterator)
            except StopIteration:
                self.iterator=iter(self.loader);item=next(self.iterator)
            result.append(_as_batch(item))
        return result


def microbatch_count(size, period, config):
    """保留 HESSO 的渐增规则；CRIC 传入自己的周期编号。"""
    if size<1:raise ValueError('α 评估池/缓存不能为空。')
    grow=max(0,period-config.start_after_period)//max(1,config.eval_batch_grow_every)
    return min(size,max(1,min(config.eval_batch_limit,1+grow)))


def select_alpha_batches(source, targets, period, config, device, cursor):
    """一次更新只取一次批次；所有正/负扰动复用返回的同一列表。"""
    if isinstance(source,AlphaEvaluationPool):
        selected=source.take(microbatch_count(len(source),period,config))
    elif isinstance(source,(list,tuple)):
        count=microbatch_count(len(source),period,config)
        selected=[_as_batch(source[(cursor+i)%len(source)]) for i in range(count)]
    else:
        selected=[_as_batch(source) if isinstance(source,AlphaBatch) else AlphaBatch(source,targets)]
    selected=[AlphaBatch(_to_device(b.inputs,device),_to_device(b.targets,device),
                         tuple(b.sample_ids),b.supervised_tokens) for b in selected]
    return selected


def commit_alpha_selection(source, selected):
    """仅成功更新后提交覆盖量；gap/stride 跳过时不消耗数据。"""
    ids=[row for batch in selected for row in batch.sample_ids]
    if isinstance(source,(AlphaEvaluationPool, CRICAlphaPool)):
        source.used_sample_ids.update(ids)
        unique=len(source.used_sample_ids)
    else:unique=None
    result = {'sample_ids':ids,'microbatches':len(selected),
            'supervised_tokens':sum(batch.supervised_tokens for batch in selected),
            'unique_samples_seen':unique}
    if isinstance(source, CRICAlphaPool):
        result.update(source.last_selection)
    return result


def _ratio_count(size, ratio, align, minimum=1, maximum=None):
    """先计算样本预算，再可选地就近对齐；可用数量和显式上限优先。"""
    if size < 1:
        raise ValueError('α 数据集或分层不能为空。')
    count = max(1, math.ceil(size * ratio))
    if align and ratio < 1:
        # 按绝对距离取最近的 2 次幂，等距时向上；不复制样本补齐。
        lower = 1 << (count.bit_length() - 1)
        upper = lower * 2
        count = lower if count - lower < upper - count else upper
    count = max(minimum, count)
    return min(size, count, maximum if maximum is not None else size)


def _group_indices(size, labels):
    if labels is not None and len(labels) != size:
        raise ValueError('α 分层标签数必须与数据集长度一致。')
    groups = defaultdict(list)
    display_labels = {}
    for index in range(size):
        label = labels[index] if labels is not None else 'all'
        try:
            display = str(label)
            if display in display_labels and display_labels[display] != label:
                raise ValueError('α 分层标签的字符串表示必须唯一。')
            display_labels[display] = label
            groups[label].append(index)
        except TypeError as error:
            raise ValueError('α 分层标签必须可哈希。') from error
    return dict(groups)


class CRICAlphaPool:
    """CRIC 的样本级百分比池；不改变 HESSO 的缓存/微批协议。

    每层独立随机轮换，单次更新内无重复；数据留在 CPU，选中后才按微批
    collate。任务/类别标签由调用方提供，没有标签时自然退化为单任务。
    """
    def __init__(self, dataset, *, batch_size=8, collate_fn=None, groups=None, seed=42):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError('α 微批大小必须为正整数。')
        if len(dataset) < 1:
            raise ValueError('α 数据集不能为空。')
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn or default_collate
        self.groups = _group_indices(len(dataset), groups)
        self.generator = torch.Generator().manual_seed(seed)
        self.orders, self.cursors = {}, {}
        self.used_sample_ids = set()
        self.last_selection = {}
        for label, indexes in self.groups.items():
            self.orders[label] = self._permutation(indexes)
            self.cursors[label] = 0

    def __len__(self):
        return len(self.dataset)  # 此类长度的单位是样本，不是微批。

    def _permutation(self, indexes):
        return [indexes[i] for i in torch.randperm(len(indexes), generator=self.generator).tolist()]

    def _draw_group(self, label, count):
        chosen, seen = [], set()
        while len(chosen) < count:
            if self.cursors[label] == len(self.orders[label]):
                self.orders[label] = self._permutation(self.groups[label])
                self.cursors[label] = 0
            index = self.orders[label][self.cursors[label]]
            self.cursors[label] += 1
            if index not in seen:
                chosen.append(index)
                seen.add(index)
        return chosen

    def take(self, config):
        batches, counts = [], {}
        for label, indexes in self.groups.items():
            count = _ratio_count(len(indexes), config.update_ratio,
                                 config.align_power_of_two, config.min_update_samples,
                                 config.max_update_samples)
            chosen = self._draw_group(label, count)
            counts[str(label)] = count
            for start in range(0, count, self.batch_size):
                micro = chosen[start:start + self.batch_size]
                batch = _as_batch(self.collate_fn([self.dataset[i] for i in micro]))
                if batch.sample_ids and len(batch.sample_ids) != len(micro):
                    raise ValueError('AlphaBatch.sample_ids 数量与实际样本数不符。')
                batches.append(AlphaBatch(batch.inputs, batch.targets,
                    batch.sample_ids or tuple(micro), batch.supervised_tokens))
        total = sum(counts.values())
        self.last_selection = {'selection_unit': 'sample', 'pool_samples': len(self),
            'selected_samples': total, 'selected_by_group': counts,
            'requested_update_ratio': config.update_ratio,
            'actual_update_ratio': total / len(self)}
        return batches

    def state_dict(self):
        return copy.deepcopy({'groups': self.groups, 'orders': self.orders,
            'batch_size': self.batch_size,
            'cursors': self.cursors, 'rng': self.generator.get_state(),
            'used_sample_ids': list(self.used_sample_ids), 'last_selection': self.last_selection})

    def load_state_dict(self, state):
        if state['groups'] != self.groups or state['batch_size'] != self.batch_size:
            raise ValueError('恢复 α 池时，样本索引、分层或微批大小已改变。')
        for label, indexes in self.groups.items():
            if (sorted(state['orders'][label]) != indexes or
                    not 0 <= state['cursors'][label] <= len(indexes)):
                raise ValueError('α 池检查点中的轮换索引不合法。')
        self.orders = copy.deepcopy(state['orders'])
        self.cursors = copy.deepcopy(state['cursors'])
        self.generator.set_state(state['rng'].cpu())
        self.used_sample_ids = set(state['used_sample_ids'])
        self.last_selection = copy.deepcopy(state['last_selection'])


def prepare_cric_alpha_data(train_dataset, *, alpha_dataset=None, batch_size=8,
                            collate_fn=None, train_groups=None, alpha_groups=None,
                            config=None, seed=42, holdout=True):
    """返回 (梯度训练数据, α 池, 清单)，供任何模型复用。

    显式 α 数据优先且不再按 pool_ratio 缩减。回退时按每层百分比固定取样；
    默认从梯度训练中移除这些行，holdout=False 可显式选择旧式共享训练样本。
    分层上、下限均按每层计算；不提供标签时只有一层。
    """
    from .cric_alpha import make_cric_alpha_config
    config = make_cric_alpha_config(config)
    if config.sampling_policy != 'ratio':
        raise ValueError('prepare_cric_alpha_data 用于 ratio 策略；legacy 使用原评估批次接口。')
    if alpha_dataset is not None:
        selected = list(range(len(alpha_dataset)))
        pool_dataset, groups = alpha_dataset, alpha_groups
        training = train_dataset
        source = 'explicit'
    else:
        grouped = _group_indices(len(train_dataset), train_groups)
        generator = torch.Generator().manual_seed(seed)
        selected = []
        for label, indexes in grouped.items():
            available = len(indexes) - int(holdout)
            if available < 1:
                raise ValueError('独立划出 α 池时，每个分层至少需要两条训练样本。')
            count = min(available, _ratio_count(len(indexes), config.pool_ratio,
                config.align_power_of_two, config.min_pool_samples, config.max_pool_samples))
            selected.extend(indexes[i] for i in torch.randperm(len(indexes), generator=generator)[:count].tolist())
        selected.sort()
        pool_dataset = Subset(train_dataset, selected)
        groups = [train_groups[i] for i in selected] if train_groups is not None else None
        held = set(selected)
        training = Subset(train_dataset, [i for i in range(len(train_dataset)) if i not in held]) if holdout else train_dataset
        source = 'training_holdout' if holdout else 'training_shared'
    pool = CRICAlphaPool(pool_dataset, batch_size=batch_size, collate_fn=collate_fn,
                         groups=groups, seed=seed)
    info = {'source': source, 'source_train_samples': len(train_dataset),
        'gradient_train_samples': len(training), 'pool_samples': len(pool),
        'pool_indices': selected, 'pool_ratio': config.pool_ratio,
        'actual_pool_ratio': None if source == 'explicit' else len(pool) / len(train_dataset),
        'update_ratio': config.update_ratio, 'batch_size': batch_size,
        'align_power_of_two': config.align_power_of_two, 'seed': seed,
        'pool_by_group': {str(k): len(v) for k,v in pool.groups.items()},
        'index_sha256': hashlib.sha256(repr(selected).encode('utf-8')).hexdigest()}
    return training, pool, info
