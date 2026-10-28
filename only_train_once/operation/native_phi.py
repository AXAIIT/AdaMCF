"""Phi 每头维度压缩的局部适配；保持原 RoPE 频率与 attention 缩放。"""
import torch
from torch import nn
from transformers.models.phi.modeling_phi import PhiAttention


def dimension_units(head_dim, rotary_ndims):
    """每个逻辑单位含两维，旋转区按 rotate_half 的真实配对组合。

    非旋转区也按相邻双维分组，保持等宽 INDEX_GROUPS 和偶数 head_dim，
    无须修改公共五准则或为变宽分组引入新的评分归一化方式。
    """
    if rotary_ndims % 2 or (head_dim - rotary_ndims) % 2:
        raise ValueError('head_dim 模式要求旋转区和非旋转区均可双维分组。')
    half = rotary_ndims // 2
    return ([[i, i + half] for i in range(half)]
            + [[i, i + 1] for i in range(rotary_ndims, head_dim, 2)])


class HeadDimPhiAttention(PhiAttention):
    """仅选择当前层保留的旋转坐标，其余前向计算沿用原生 Phi。"""

    def forward(self, hidden_states, position_embeddings, attention_mask, **kwargs):
        cos, sin = position_embeddings
        indexes = self.rope_coordinate_indexes
        selected = (cos.index_select(-1, indexes), sin.index_select(-1, indexes))
        return super().forward(hidden_states, position_embeddings=selected,
                               attention_mask=attention_mask, **kwargs)


def adapt_head_dimensions(attention, kept_dimensions, original_rotary_ndims, original_scaling):
    if attention.qk_layernorm:
        raise ValueError('head_dim 剪枝暂不支持 qk_layernorm=True：删维会改变归一化统计。')
    # 在 meta 上建立模块外壳，随后接收已经压缩的投影，避免分配稠密副本。
    with torch.device('meta'):
        result = HeadDimPhiAttention(attention.config, attention.layer_idx)
    for name in ('q_proj', 'k_proj', 'v_proj', 'dense'):
        setattr(result, name, getattr(attention, name))
    result.head_dim = len(kept_dimensions)
    rotary_indexes = [i for i in kept_dimensions if i < original_rotary_ndims]
    result.rotary_ndims = len(rotary_indexes)
    result.scaling = original_scaling
    result.num_key_value_groups = 1
    # JSON 结构清单负责持久化坐标；该非持久 buffer 随模型移动设备，
    # meta 重载时先建立在 CPU，之后与已分配的模型一起移到目标设备。
    parameter_device = attention.q_proj.weight.device
    index_device = 'cpu' if parameter_device.type == 'meta' else parameter_device
    result.register_buffer('rope_coordinate_indexes',
                           torch.tensor(rotary_indexes, dtype=torch.long, device=index_device),
                           persistent=False)
    result.train(attention.training)
    return result


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


class NativePhiAttentionOTO:
    """原生 HF Phi 的结构算子，与旧 PhiMHA/PhiAttentionOTO 独立。

    单个逻辑组同时拥有 Q/K/V 输出行、bias 和 dense 输入列；物理压缩
    使用同一份原始坐标合同，防止优化器分组与导出索引各自解释布局。
    """

    def __init__(self, module):
        self.module = module

    def group_spec(self, pruning_mode):
        from ..transform import make_index_groups_transform
        attn = self.module
        heads = attn.q_proj.out_features // attn.head_dim
        if pruning_mode == 'head_dim':
            units = dimension_units(attn.head_dim, attn.rotary_ndims)
            blocks = [[h * attn.head_dim + d for h in range(heads) for d in unit]
                      for unit in units]
        else:
            units = None
            blocks = [list(range(h * attn.head_dim, (h + 1) * attn.head_dim))
                      for h in range(heads)]
        row = make_index_groups_transform(blocks, axis=0)
        column = make_index_groups_transform(blocks, axis=1)
        params, transforms = [], []
        for projection in (attn.q_proj, attn.k_proj, attn.v_proj):
            params.extend([projection.weight, projection.bias])
            transforms.extend([row, row])
        return (params + [attn.dense.weight], transforms + [column],
                len(blocks), units)

    @staticmethod
    def removed_channels(entry):
        kept_dimensions = entry.get('head_dim_keep', list(range(entry['head_dim'])))
        kept = {h * entry['head_dim'] + d for h in entry['attention']['keep']
                for d in kept_dimensions}
        return sorted(set(range(entry['attention']['original'] * entry['head_dim'])) - kept)

    def apply_structure(self, layer, entry):
        attn = self.module
        dimensions = entry.get('head_dim_keep', list(range(entry['head_dim'])))
        if entry['attention']['pruned'] or entry.get('head_dim_pruned'):
            physical = [h * entry['head_dim'] + d for h in entry['attention']['keep']
                        for d in dimensions]
            for name in ('q_proj', 'k_proj', 'v_proj'):
                setattr(attn, name, subset_linear(getattr(attn, name), rows=physical))
            attn.dense = subset_linear(attn.dense, columns=physical)
        if entry.get('head_dim_pruned'):
            attn = adapt_head_dimensions(attn, dimensions, entry['rotary_ndims'],
                                        entry['attention_scaling'])
            layer.self_attn = attn
        # 当前原生 Phi 根据实际投影宽度推导头数，residual hidden_size 保持不变。
        attn.num_key_value_groups = 1

