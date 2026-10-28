"""原生 Phi 的显式依赖组、物理压缩和可重载结构包。

FFN 连动 fc1 的行/bias 与 fc2 的列；注意力连动
Q/K/V 的完整头与 dense 的对应输入列，保持 residual hidden_size 不变。
"""
import json
import copy
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file, save_file
from transformers import PhiConfig, PhiForCausalLM
from ..operation.native_phi import (dimension_units,
                                    subset_linear, NativePhiAttentionOTO)

from ..transform import TensorTransform


def validate_native_phi(model):
    config = model.config
    if config.model_type != 'phi' or config.num_attention_heads != config.num_key_value_heads:
        raise ValueError('此适配只支持原生 Phi MHA，不支持 GQA 或其他模型类型。')
    if config.hidden_act not in ('gelu_new', 'gelu', 'relu', 'silu'):
        raise ValueError('FFN 剪枝要求激活函数满足 activation(0)=0。')
    for layer in model.model.layers:
        attn, mlp = layer.self_attn, layer.mlp
        if any(p.bias is None for p in (attn.q_proj, attn.k_proj, attn.v_proj, mlp.fc1)):
            raise ValueError('当前原生 Phi 适配要求 Q/K/V 和 fc1 带 bias。')
        if not all(isinstance(p, nn.Linear) for p in (
                attn.q_proj, attn.k_proj, attn.v_proj, attn.dense, mlp.fc1, mlp.fc2)):
            raise TypeError('投影层必须为未量化的 nn.Linear。')
        if (attn.q_proj.out_features != attn.k_proj.out_features
                or attn.q_proj.out_features != attn.v_proj.out_features
                or attn.dense.in_features != attn.q_proj.out_features):
            raise ValueError('Q/K/V/dense 的头布局不一致。')
        if attn.q_proj.out_features % attn.head_dim or mlp.fc1.out_features != mlp.fc2.in_features:
            raise ValueError('注意力头宽度或 FFN 连动关系错误。')


def build_param_groups(model, pruning_mode='joint'):
    """由实际模块自动构造依赖组，全部可训练参数恰好归属一次。"""
    validate_native_phi(model)
    if pruning_mode not in ('joint', 'ffn', 'num_head', 'head_dim'):
        raise ValueError('pruning_mode 必须为 joint/ffn/num_head/head_dim。')
    if pruning_mode == 'head_dim' and model.config.qk_layernorm:
        raise ValueError('head_dim 模式不支持 qk_layernorm=True。')
    names = {id(p): name for name, p in model.named_parameters()}
    groups, owned = [], set()

    def add(layer_index, kind, params, transforms, count, prunable=True):
        for p in params:
            if id(p) in owned:
                raise ValueError('参数被多个剪枝组重复拥有。')
            owned.add(id(p))
        groups.append({
            'id': f'layer_{layer_index:02d}_{kind}', 'layer_index': layer_index,
            'phi_kind': kind, 'params': params, 'p_names': [names[id(p)] for p in params],
            'p_transform': transforms, 'num_groups': count,
            'is_prunable': prunable, 'is_auxiliary': False, 'auxiliary_ngs': [],
        })

    for index, layer in enumerate(model.model.layers):
        mlp, attn = layer.mlp, layer.self_attn
        add(index, 'ffn', [mlp.fc1.weight, mlp.fc1.bias, mlp.fc2.weight],
            [TensorTransform.BASIC, TensorTransform.BASIC, TensorTransform.TRANSPOSE],
            mlp.fc1.out_features, prunable=pruning_mode in ('joint', 'ffn'))
        params, transforms, count, units = NativePhiAttentionOTO(attn).group_spec(pruning_mode)
        if pruning_mode == 'head_dim':
            add(index, 'head_dim', params, transforms, count)
            groups[-1]['phi_dimension_units'] = units
        else:
            add(index, 'attention', params, transforms, count,
                prunable=pruning_mode in ('joint', 'num_head'))

    # 输出 bias、embedding、LayerNorm 等保持可训练，但不改变其结构维度。
    rest = [p for p in model.parameters() if id(p) not in owned]
    groups.append({
        'id': 'unprunable', 'params': rest, 'p_names': [names[id(p)] for p in rest],
        'p_transform': [TensorTransform.NO_PRUNE] * len(rest), 'num_groups': 1,
        'is_prunable': False, 'is_auxiliary': False, 'auxiliary_ngs': [],
    })
    all_ids = [id(p) for group in groups for p in group['params']]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(names):
        raise ValueError('参数归属覆盖检查失败。')
    return groups


def group_summary(groups):
    return [{**{key: group[key] for key in ('id', 'num_groups', 'p_names', 'is_prunable')},
             'kind': group.get('phi_kind', 'unprunable')}
            for group in groups]


@torch.no_grad()
def collect_pruning_plan(model, optimizer):
    """只允许导出已经精确置零的最终组，防止把未结束的稀疏化当作压缩。"""
    if not optimizer.is_cric_terminated:
        raise RuntimeError('CRIC 尚未结束，不能物理导出。')
    plan = {'format_version': 2, 'pruning_mode': getattr(optimizer, 'phi_pruning_mode', 'joint'), 'layers': []}
    by_layer = {}
    # 未参与当前模式的结构使用恒等合同，继续训练但不会被误导出为删除。
    for index, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        heads = attn.q_proj.out_features // attn.head_dim
        ffn = layer.mlp.fc1.out_features
        by_layer[index] = {
            'layer_index': index,
            'ffn': {'original': ffn, 'keep': list(range(ffn)), 'pruned': []},
            'attention': {'original': heads, 'keep': list(range(heads)), 'pruned': []},
            'head_dim': attn.head_dim, 'head_dim_keep': list(range(attn.head_dim)),
            'head_dim_pruned': [], 'head_dim_pruned_units': [],
            'rotary_ndims': attn.rotary_ndims, 'attention_scaling': attn.scaling,
        }
    for group in optimizer.param_groups:
        if not group['is_prunable']:
            continue
        if group['active_redundant_idxes']:
            raise RuntimeError('混合稀疏化尚未结束。')
        pruned = sorted(set(group['pruned_idxes']))
        keep = [i for i in range(group['num_groups']) if i not in set(pruned)]
        if not keep:
            raise RuntimeError(f'{group["id"]} 没有保留组。')
        for p, axis, physical in optimizer._pruning_slices(group, pruned):
            if torch.count_nonzero(p.index_select(axis, physical)).item():
                raise RuntimeError(f'{group["id"]} 的被剪参数没有精确归零。')
        entry = by_layer[group['layer_index']]
        if group['phi_kind'] == 'head_dim':
            removed = sorted(dim for unit in pruned for dim in group['phi_dimension_units'][unit])
            entry['head_dim_pruned'] = removed
            entry['head_dim_keep'] = [i for i in range(entry['head_dim']) if i not in set(removed)]
            entry['head_dim_pruned_units'] = pruned
        else:
            entry[group['phi_kind']] = {'original': group['num_groups'], 'keep': keep, 'pruned': pruned}
    plan['layers'] = [by_layer[index] for index in range(len(model.model.layers))]
    return plan


def validate_plan(model, plan):
    if plan.get('format_version') not in (1, 2) or len(plan['layers']) != len(model.model.layers):
        raise ValueError('结构清单版本或层数不一致。')
    for index, (layer, entry) in enumerate(zip(model.model.layers, plan['layers'])):
        expected = {'ffn': layer.mlp.fc1.out_features,
                    'attention': layer.self_attn.q_proj.out_features // layer.self_attn.head_dim}
        if entry['layer_index'] != index or entry['head_dim'] != layer.self_attn.head_dim:
            raise ValueError('结构清单的层号或 head_dim 不一致。')
        for kind, count in expected.items():
            item = entry[kind]
            keep, pruned = item['keep'], item['pruned']
            if (item['original'] != count or not keep or keep != sorted(set(keep))
                    or pruned != sorted(set(pruned)) or set(keep).intersection(pruned)
                    or sorted(keep + pruned) != list(range(count))):
                raise ValueError(f'第 {index} 层 {kind} 的结构索引非法。')
        kept = entry.get('head_dim_keep', list(range(entry['head_dim'])))
        removed = entry.get('head_dim_pruned', [])
        if (not kept or kept != sorted(set(kept)) or removed != sorted(set(removed))
                or set(kept).intersection(removed)
                or sorted(kept + removed) != list(range(entry['head_dim']))):
            raise ValueError(f'第 {index} 层 head_dim 的索引非法。')
        if removed:
            if layer.self_attn.qk_layernorm:
                raise ValueError('head_dim 剪枝不能改变 QK LayerNorm 的归一化统计。')
            rotary = entry.get('rotary_ndims', layer.self_attn.rotary_ndims)
            if rotary != layer.self_attn.rotary_ndims:
                raise ValueError('原始旋转维度与模型不一致。')
            for unit in dimension_units(entry['head_dim'], rotary):
                if any(i in removed for i in unit) != all(i in removed for i in unit):
                    raise ValueError('head_dim 删除必须保持双维组和 RoPE 配对完整。')
        if abs(entry.get('attention_scaling', layer.self_attn.scaling) - layer.self_attn.scaling) > 1e-14:
            raise ValueError('结构清单不得改变原 attention scaling。')




def apply_structure(model, plan):
    validate_native_phi(model)
    validate_plan(model, plan)
    for layer, entry in zip(model.model.layers, plan['layers']):
        mlp, attn = layer.mlp, layer.self_attn
        if entry['ffn']['pruned']:
            keep = entry['ffn']['keep']
            mlp.fc1 = subset_linear(mlp.fc1, rows=keep)
            mlp.fc2 = subset_linear(mlp.fc2, columns=keep)
        NativePhiAttentionOTO(layer.self_attn).apply_structure(layer, entry)
    model.config.adamcf_layer_shapes = [
        {'intermediate_size': len(e['ffn']['keep']),
         'num_attention_heads': len(e['attention']['keep']),
         'head_dim': len(e.get('head_dim_keep', list(range(e['head_dim']))))}
        for e in plan['layers']]
    model.config.adamcf_requires_structure_loader = True
    return model


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def save_pruned_package(model, tokenizer, plan, output):
    """导出纯 safetensors + JSON；异构逐层宽度由本文件的 loader 重建。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model.config.save_pretrained(output)
    if tokenizer is not None:
        tokenizer.save_pretrained(output)
    (output / 'architecture.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
    save_file(state, str(output / 'model.safetensors'), metadata={'format': 'pt'})


def load_pruned_package(directory, device='cpu'):
    directory = Path(directory)
    config = PhiConfig.from_pretrained(directory, local_files_only=True)
    config._attn_implementation = 'eager'
    plan = json.loads((directory / 'architecture.json').read_text(encoding='utf-8'))
    # 不反序列化任意 Python 对象，不先分配一份完整稠密模型。
    with torch.device('meta'):
        model = PhiForCausalLM(config)
    apply_structure(model, plan)
    state = load_file(str(directory / 'model.safetensors'), device='cpu')
    model.load_state_dict(state, strict=True, assign=True)
    # 非持久 RoPE buffer 不在 state_dict 中，meta 骨架需显式重建。
    from transformers.models.phi.modeling_phi import PhiRotaryEmbedding
    model.model.rotary_emb = PhiRotaryEmbedding(config=config, device='cpu')
    model.to(device)
    return model


@torch.no_grad()
def validate_zeroed_plan(model, plan):
    """外部 JSON 也必须对应精确零结构；先检查所有层，再允许任何物理修改。"""
    validate_native_phi(model)
    validate_plan(model, plan)
    for layer, entry in zip(model.model.layers, plan['layers']):
        mlp, attn = layer.mlp, layer.self_attn
        slices = [(mlp.fc1.weight, 0, entry['ffn']['pruned']),
                  (mlp.fc1.bias, 0, entry['ffn']['pruned']),
                  (mlp.fc2.weight, 1, entry['ffn']['pruned'])]
        channels = NativePhiAttentionOTO.removed_channels(entry)
        for projection in (attn.q_proj, attn.k_proj, attn.v_proj):
            slices.extend([(projection.weight, 0, channels), (projection.bias, 0, channels)])
        slices.append((attn.dense.weight, 1, channels))
        for parameter, axis, indexes in slices:
            if indexes:
                index = torch.tensor(indexes, device=parameter.device, dtype=torch.long)
                if torch.count_nonzero(parameter.index_select(axis, index)).item():
                    raise RuntimeError('结构清单中被剪参数没有精确归零，拒绝导出。')


class NativePhiAdapter:
    """供 OTO 调度的模型适配层：分组、CRIC 合同、物理导出与安全重载。"""
    name = 'native_phi'

    def __init__(self, model, pruning_mode='joint'):
        if getattr(model.config, 'adamcf_requires_structure_loader', False):
            raise ValueError('已压缩 Phi 的再次结构剪枝尚未适配，请从原始稠密结构开始。')
        self.model = model
        self.pruning_mode = pruning_mode
        self._groups = build_param_groups(model, pruning_mode)
        self.exported = False

    def get_param_groups(self):
        if self.exported:
            raise RuntimeError('该 OTO 已物理导出，重新训练请创建新的模型/OTO。')
        return self._groups

    def collect_pruning_plan(self, optimizer):
        if optimizer is None:
            raise RuntimeError('请先调用 oto.hessocric()，或显式提供已验证的 pruning_plan。')
        if not hasattr(optimizer, 'is_cric_terminated'):
            raise NotImplementedError('自动冻结结构合同当前仅支持 HESSOCRIC；其他优化器需显式零结构合同。')
        owned = {id(p) for group in optimizer.param_groups for p in group['params']}
        if owned != {id(p) for p in self.model.parameters()}:
            raise ValueError('优化器不属于当前模型，拒绝推导结构。')
        optimizer.phi_pruning_mode = self.pruning_mode
        return collect_pruning_plan(self.model, optimizer)

    def construct_subnet(self, *, optimizer, pruning_plan, tokenizer,
                         out_dir, full_group_sparse_model_dir, compressed_model_dir,
                         save_full_group_sparse_model, export_float16, ckpt_format,
                         merge_lora_to_base, unmerge_lora_to_base):
        if self.exported:
            raise RuntimeError('同一 OTO 不能重复进行物理剪枝。')
        if ckpt_format not in ('torch', 'safetensors'):
            raise ValueError('原生 Phi 当前导出安全结构包，ckpt_format 支持 torch/safetensors。')
        if merge_lora_to_base or unmerge_lora_to_base:
            raise ValueError('原生 Phi 适配不接受 LoRA 合并选项；旧图算子分支保持原行为。')
        plan = copy.deepcopy(pruning_plan) if pruning_plan is not None else self.collect_pruning_plan(optimizer)
        validate_zeroed_plan(self.model, plan)
        package = Path(compressed_model_dir or out_dir) / 'compressed_model'
        sparse_path = Path(full_group_sparse_model_dir or out_dir) / 'sparse_model'
        if package.exists() or (save_full_group_sparse_model and sparse_path.exists()):
            raise FileExistsError('导出目录已存在，拒绝覆盖。')
        if save_full_group_sparse_model:
            self.model.save_pretrained(sparse_path, safe_serialization=True)
            if tokenizer is not None:
                tokenizer.save_pretrained(sparse_path)
        apply_structure(self.model, plan)
        self.exported = True
        self._groups = None  # 不保留已替换的稠密 Parameter，释放导出阶段的旧权重引用。
        if export_float16:
            self.model.half()
        plan['parameters_after'] = parameter_count(self.model)
        save_pruned_package(self.model, tokenizer, plan, package)
        return str(package), str(sparse_path) if save_full_group_sparse_model else None, plan
