"""从已保存的稀疏检查点独立执行物理压缩及精度审计，无需重新训练。"""
import argparse
import copy
import gc
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, PhiForCausalLM

from phi2_data import file_sha256, to_device
from phi2_precision import audit_structure_precision
from phi2_structure import apply_structure, parameter_count, save_pruned_package


def export_run(run, device):
    run, device = Path(run), torch.device(device)
    if (run / 'compressed_model').exists():
        raise FileExistsError('压缩包已存在，拒绝覆盖。')
    context = json.loads((run / 'export_context.json').read_text(encoding='utf-8'))
    plan = json.loads((run / 'pruning_plan.json').read_text(encoding='utf-8'))
    probe = torch.load(run / 'sparse_probe.pt', map_location='cpu', weights_only=True)
    batch = to_device(probe['batch'], device)
    sparse = PhiForCausalLM.from_pretrained(
        run / 'sparse_model', local_files_only=True, torch_dtype=torch.float32,
        attn_implementation='eager').to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(run / 'sparse_model', local_files_only=True)
    with torch.no_grad():
        reloaded_sparse = sparse(**batch, use_cache=False).logits
    torch.testing.assert_close(reloaded_sparse, probe['sparse_logits'].to(device),
                               atol=probe['atol'], rtol=probe['rtol'])
    compressed = copy.deepcopy(sparse)
    apply_structure(compressed, plan)
    after_count = parameter_count(compressed)
    if not after_count < context['parameters_before']:
        raise AssertionError('实际参数量没有减少。')
    diagnostics, compact_logits = audit_structure_precision(sparse, compressed, batch)
    (run / 'precision_audit.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
    plan['parameters_after'] = after_count
    # 独立重载应对齐同一压缩结构的原生 FP32 参考；稀疏/压缩结构等价
    # 单独由更严格的 FP64 门槛确认，FP32 漂移完整保留为诊断结果。
    torch.save({'batch': probe['batch'], 'sparse_logits': probe['sparse_logits'],
                'compressed_logits': compact_logits.cpu(), 'atol': probe['atol'],
                'rtol': probe['rtol']}, run / 'equivalence_probe.pt')
    del sparse, reloaded_sparse
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    package = run / 'compressed_model'
    save_pruned_package(compressed, tokenizer, plan, package)
    (run / 'pruning_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    report = {**context, 'functional_pruning_passed': True,
              'parameters_after': after_count,
              'parameter_reduction_ratio': 1 - after_count / context['parameters_before'],
              'precision_audit': diagnostics,
              'sparse_compressed_logits_max_abs_error': diagnostics['native_fp32']['max_abs_error'],
              'compressed_package': str(package),
              'compressed_weights_sha256': file_sha256(package / 'model.safetensors')}
    if device.type == 'cuda':
        report.update(peak_allocated_GiB=torch.cuda.max_memory_allocated(device) / 2**30,
                      peak_reserved_GiB=torch.cuda.max_memory_reserved(device) / 2**30)
    for name in ('pruning_result.json', 'status.json'):
        (run / name).write_text(json.dumps({'state': 'completed', **report}, indent=2), encoding='utf-8')
    print('PRUNING_COMPLETE', json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    export_run(args.run_dir, args.device)


if __name__ == '__main__':
    main()
