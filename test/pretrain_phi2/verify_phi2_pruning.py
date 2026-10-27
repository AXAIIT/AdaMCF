"""新进程重载压缩结构包，独立验证输出、生成缓存及压缩模型反向传播。"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from phi2_structure import load_pruned_package, parameter_count
from phi2_data import file_sha256, to_device


def verify(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    run = args.run_dir
    result = json.loads((run / 'pruning_result.json').read_text(encoding='utf-8'))
    package = run / 'compressed_model'
    if file_sha256(package / 'model.safetensors') != result['compressed_weights_sha256']:
        raise ValueError('压缩权重摘要与剪枝结果记录不一致。')
    model = load_pruned_package(package, device)
    if parameter_count(model) != result['parameters_after']:
        raise AssertionError('重载参数量错误。')
    probe = torch.load(run / 'equivalence_probe.pt', map_location='cpu', weights_only=True)
    batch = to_device(probe['batch'], device)
    model.eval()
    with torch.no_grad():
        output = model(**batch, use_cache=False)
    if not torch.isfinite(output.logits).all() or not torch.isfinite(output.loss):
        raise FloatingPointError('重载后 logits/loss 非有限。')
    reference = probe['compressed_logits'].to(device)
    torch.testing.assert_close(output.logits, reference, atol=probe['atol'], rtol=probe['rtol'])
    error = float((output.logits - reference).abs().max())
    tokenizer = AutoTokenizer.from_pretrained(package, local_files_only=True)
    prompt = tokenizer('The capital of France is', return_tensors='pt').to(device)
    with torch.no_grad():
        generated = model.generate(**prompt, max_new_tokens=8, do_sample=False,
                                   use_cache=True, pad_token_id=tokenizer.eos_token_id)
    model.train()
    model.zero_grad(set_to_none=True)
    loss = model(**batch, use_cache=False).loss
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    report = {'passed': True, 'parameters': parameter_count(model),
              'reload_compressed_logits_max_abs_error': error,
              'structure_fp64_passed': result['precision_audit']['structure_fp64']['passed'],
              'reload_loss': float(output.loss), 'backward_loss': float(loss.detach()),
              'gradient_norm': float(norm), 'generated_token_count': generated.shape[1] - prompt.input_ids.shape[1],
              'generation_with_cache': True, 'generated_text': tokenizer.decode(generated[0], skip_special_tokens=True),
              'architecture_layers': len(model.model.layers), 'device': str(device)}
    (run / 'independent_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    verify(parser.parse_args())


if __name__ == '__main__':
    main()
