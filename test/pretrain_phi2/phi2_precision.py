"""用更高精度审计结构等价性，保留原生 FP32 的真实误差记录。

矩阵宽度改变后 GEMM 的求和顺序会变化。这里对同一对模型执行 FP64
结构对照；只在此诊断作用域中使 attention softmax 也使用 FP64。
训练、保存与正常推理仍使用原生 Phi，不修改安装包或公共算子。
"""
from unittest.mock import patch
import copy
import gc

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


@torch.no_grad()
def merge_lora_with_precision_audit(model, probes, atol=1e-8, rtol=1e-8):
    """区分 W·x+B·A·x 与 (W+B·A)·x 的舍入差异和错误合并。

    严格 FP64 对照验证合并公式；实际导出仍由原 FP32 模型执行 PEFT
    标准合并。FP32 的旧门槛结果、误差和 argmax 一致率照实保存，
    不把 FP64 通过写成 FP32 逐元素一致。这里只处理 CPU 导出诊断。
    """
    if not probes or any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('合并审计需要非空探针和 CPU FP32 模型。')
    inputs = [{k:v for k,v in batch.items() if k != 'labels'} for batch in probes]
    model.eval()
    references = [model(**batch,use_cache=False).logits for batch in inputs]
    high = copy.deepcopy(model).double().eval()
    try:
        with patch.object(modeling_phi,'eager_attention_forward',fp64_attention):
            expected = [high(**batch,use_cache=False).logits for batch in inputs]
            high = high.merge_and_unload(safe_merge=True).eval()
            errors = []
            for batch,reference in zip(inputs,expected):
                actual = high(**batch,use_cache=False).logits
                torch.testing.assert_close(actual,reference,atol=atol,rtol=rtol)
                errors.append(float((actual-reference).abs().max()))
    finally:
        del high
        gc.collect()
    merged = model.merge_and_unload(safe_merge=True).eval()
    records, outputs = [], []
    for batch,reference in zip(inputs,references):
        actual = merged(**batch,use_cache=False).logits
        if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
            raise FloatingPointError('合并审计出现非有限 FP32 输出。')
        error = (actual-reference).abs()
        records.append({'input_shape':list(batch['input_ids'].shape),
            'max_abs_error':float(error.max()),'mean_abs_error':float(error.mean()),
            'relative_l2_error':float(error.norm()/reference.norm().clamp_min(1e-12)),
            'top1_agreement':float((actual.argmax(-1)==reference.argmax(-1)).double().mean()),
            'strict_fp32_allclose_at_2e_4':bool(torch.allclose(actual,reference,atol=2e-4,rtol=2e-4))})
        outputs.append(actual)
    audit = {'native_fp32':records,'merge_fp64':{'passed':True,'atol':atol,'rtol':rtol,
        'max_abs_error':max(errors),'per_probe_max_abs_error':errors},
        'export_dtype':'float32','fp32_elementwise_equivalence_claimed':all(r['strict_fp32_allclose_at_2e_4'] for r in records)}
    return merged,audit,outputs
