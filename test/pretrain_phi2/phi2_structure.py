"""原生 Phi 的显式依赖组、物理压缩和可重载结构包。

不修改公共图算子：FFN 连动 fc1 的行/bias 与 fc2 的列；注意力连动
Q/K/V 的完整头与 dense 的对应输入列，保持 residual hidden_size 不变。
"""
import json
from pathlib import Path
import sys

import torch
from torch import nn
from safetensors.torch import load_file, save_file
from transformers import PhiConfig, PhiForCausalLM
from phi2_attention import adapt_head_dimensions, dimension_units

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from only_train_once.transform import TensorTransform, make_index_groups_transform


def validate_native_phi(model):
    config = model.config
    if config.model_type != 'phi' or config.num_attention_heads != config.num_key_value_heads:
        raise ValueError('此适配只支持原生 Phi MHA，不支持 GQA 或其他模型类型。')
    if config.hidden_act not in ('gelu_new', 'gelu', 'relu', 'silu'):
        raise ValueError('FFN 剪枝要求激活函数满足 activation(0)=0。')
    for layer in model.model.layers:
        attn, mlp = layer.self_attn, layer.mlp
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
        heads = attn.q_proj.out_features // attn.head_dim
        if pruning_mode == 'head_dim':
            units = dimension_units(attn.head_dim, attn.rotary_ndims)
            blocks = [[head * attn.head_dim + dim for head in range(heads) for dim in unit]
                      for unit in units]
        else:
            blocks = [list(range(h * attn.head_dim, (h + 1) * attn.head_dim)) for h in range(heads)]
        row_transform = make_index_groups_transform(blocks, axis=0)
        col_transform = make_index_groups_transform(blocks, axis=1)
        params, transforms = [], []
        for projection in (attn.q_proj, attn.k_proj, attn.v_proj):
            params.extend([projection.weight, projection.bias])
            transforms.extend([row_transform, row_transform])
        params.append(attn.dense.weight)
        transforms.append(col_transform)
        if pruning_mode == 'head_dim':
            add(index, 'head_dim', params, transforms, len(units))
            groups[-1]['phi_dimension_units'] = units
        else:
            add(index, 'attention', params, transforms, heads,
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


@torch.no_grad()
def subset_linear(old, rows=None, columns=None):
    """重建更小的 Linear，支持真实参数和用于安全重载的 meta 骨架。"""
    weight = old.weight
    if rows is not None:
        index = torch.tensor(rows, dtype=torch.long, device=weight.device)
        weight = weight.index_select(0, index)
    if columns is not None:
        index = torch.tensor(columns, dtype=torch.long, device=weight.device)
        weight = weight.index_select(1, index)
    with torch.device('meta'):
        result = nn.Linear(weight.shape[1], weight.shape[0], bias=old.bias is not None)
    result.weight = nn.Parameter(weight.contiguous(), requires_grad=old.weight.requires_grad)
    if old.bias is not None:
        bias = old.bias if rows is None else old.bias.index_select(
            0, torch.tensor(rows, dtype=torch.long, device=old.bias.device))
        result.bias = nn.Parameter(bias.contiguous(), requires_grad=old.bias.requires_grad)
    result.train(old.training)
    return result


def apply_structure(model, plan):
    validate_native_phi(model)
    validate_plan(model, plan)
    for layer, entry in zip(model.model.layers, plan['layers']):
        mlp, attn = layer.mlp, layer.self_attn
        if entry['ffn']['pruned']:
            keep = entry['ffn']['keep']
            mlp.fc1 = subset_linear(mlp.fc1, rows=keep)
            mlp.fc2 = subset_linear(mlp.fc2, columns=keep)
        kept_dimensions = entry.get('head_dim_keep', list(range(entry['head_dim'])))
        if entry['attention']['pruned'] or entry.get('head_dim_pruned'):
            physical = [head * entry['head_dim'] + offset
                        for head in entry['attention']['keep'] for offset in kept_dimensions]
            for name in ('q_proj', 'k_proj', 'v_proj'):
                setattr(attn, name, subset_linear(getattr(attn, name), rows=physical))
            attn.dense = subset_linear(attn.dense, columns=physical)
        if entry.get('head_dim_pruned'):
            layer.self_attn = adapt_head_dimensions(
                attn, kept_dimensions, entry['rotary_ndims'], entry['attention_scaling'])
            attn = layer.self_attn
        # 当前原生 Phi 根据投影实际宽度推导头数；RoPE 的 head_dim 保持不变。
        attn.num_key_value_groups = 1
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
