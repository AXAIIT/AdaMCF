"""Phi 每头维度压缩的局部适配；保持原 RoPE 频率与 attention 缩放。"""
import torch
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
