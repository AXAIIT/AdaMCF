"""模型适配分派。可选的 Transformers 依赖只在匹配原生 Phi 时加载。"""
import json
from pathlib import Path


def is_native_phi(model):
    cls = type(model)
    return (cls.__name__ == 'PhiForCausalLM'
            and cls.__module__ == 'transformers.models.phi.modeling_phi'
            and getattr(getattr(model, 'config', None), 'model_type', None) == 'phi')


def create_model_adapter(model, selection='auto', pruning_mode='joint'):
    """旧模型返回 None，继续原有 Graph；显式 graph 可关闭自动分派。"""
    if selection == 'graph':
        return None
    if selection not in ('auto', 'native_phi'):
        raise ValueError('model_adapter 必须为 auto/graph/native_phi。')
    if is_native_phi(model):
        from .phi import NativePhiAdapter
        return NativePhiAdapter(model, pruning_mode)
    if selection == 'native_phi':
        raise TypeError('native_phi 只接受 Transformers 原生 PhiForCausalLM。')
    return None


def load_subnet(directory, device='cpu'):
    """加载带结构合同的安全模型包；不隐式反序列化 torch Python 对象。"""
    directory = Path(directory)
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    if config.get('model_type') == 'phi':
        from .phi import load_pruned_package
        return load_pruned_package(directory, device)
    raise ValueError('该模型包尚未注册结构加载器；旧 torch 子网沿用原有加载方式。')
