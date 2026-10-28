"""模型无关的 α 评估批次协议。HESSO/CRIC 共用取批和覆盖记录。"""
from dataclasses import dataclass
from collections.abc import Mapping
import torch


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
    if isinstance(source,AlphaEvaluationPool):
        source.used_sample_ids.update(ids)
        unique=len(source.used_sample_ids)
    else:unique=None
    return {'sample_ids':ids,'microbatches':len(selected),
            'supervised_tokens':sum(batch.supervised_tokens for batch in selected),
            'unique_samples_seen':unique}
