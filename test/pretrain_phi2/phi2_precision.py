"""用更高精度审计结构等价性，保留原生 FP32 的真实误差记录。

矩阵宽度改变后 GEMM 的求和顺序会变化。这里对同一对模型执行 FP64
结构对照；只在此诊断作用域中使 attention softmax 也使用 FP64。
训练、保存与正常推理仍使用原生 Phi，不修改安装包或公共算子。
"""
from unittest.mock import patch

import torch
from transformers.models.phi import modeling_phi


def fp64_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key = modeling_phi.repeat_kv(key, module.num_key_value_groups)
    value = modeling_phi.repeat_kv(value, module.num_key_value_groups)
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, :key.shape[-2]]
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, value).transpose(1, 2).contiguous(), probabilities


@torch.no_grad()
def audit_structure_precision(sparse_model, compressed_model, batch, atol=1e-8, rtol=1e-8):
    """严格 FP64 门槛不随 FP32 结果放宽；函数结束时恢复 FP32。"""
    inputs = {k: v for k, v in batch.items() if k != 'labels'}
    sparse_model.eval()
    compressed_model.eval()
    sparse_fp32 = sparse_model(**inputs, use_cache=False).logits
    compact_fp32 = compressed_model(**inputs, use_cache=False).logits
    error = (sparse_fp32 - compact_fp32).abs()
    native = {
        'max_abs_error': float(error.max()), 'mean_abs_error': float(error.mean()),
        'relative_l2_error': float(error.norm() / sparse_fp32.norm().clamp_min(1e-12)),
        'top1_agreement': float((sparse_fp32.argmax(-1) == compact_fp32.argmax(-1)).float().mean()),
        'strict_fp32_allclose_at_2e_4': bool(torch.allclose(sparse_fp32, compact_fp32, atol=2e-4, rtol=2e-4)),
    }
    try:
        sparse_model.double()
        compressed_model.double()
        with patch.object(modeling_phi, 'eager_attention_forward', fp64_attention):
            reference = sparse_model(**inputs, use_cache=False).logits
            actual = compressed_model(**inputs, use_cache=False).logits
        torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
        high_precision = {'passed': True, 'dtype': 'float64', 'softmax_dtype': 'float64',
                          'atol': atol, 'rtol': rtol,
                          'max_abs_error': float((reference - actual).abs().max()),
                          'shared_rope_constants': 'native Phi position embeddings'}
    finally:
        sparse_model.float()
        compressed_model.float()
    return {'native_fp32': native, 'structure_fp64': high_precision}, compact_fp32
