"""新进程重载压缩结构包，独立验证输出、生成缓存及压缩模型反向传播。"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from phi2_structure import parameter_count
from only_train_once import OTO
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
    model = OTO.load_subnet(package, device)
    if parameter_count(model) != result['parameters_after']:
        raise AssertionError('重载参数量错误。')
    plan = json.loads((package / 'architecture.json').read_text(encoding='utf-8'))
    layer_shapes = []
    for layer, entry in zip(model.model.layers, plan['layers']):
        heads = len(entry['attention']['keep'])
        width = len(entry.get('head_dim_keep', list(range(entry['head_dim']))))
        attn, mlp = layer.self_attn, layer.mlp
        if (attn.head_dim != width or attn.q_proj.out_features != heads * width
                or attn.k_proj.out_features != heads * width
                or attn.v_proj.out_features != heads * width
                or attn.dense.in_features != heads * width
                or mlp.fc1.out_features != len(entry['ffn']['keep'])
                or mlp.fc2.in_features != len(entry['ffn']['keep'])):
            raise AssertionError('重载后的真实矩阵尺寸与结构清单不一致。')
        expected_scaling = entry.get('attention_scaling', entry['head_dim'] ** -0.5)
        if abs(attn.scaling - expected_scaling) > 1e-14:
            raise AssertionError('重载改变了 attention scaling。')
        layer_shapes.append({'heads': heads, 'head_dim': width,
                             'ffn_width': mlp.fc1.out_features, 'scaling': attn.scaling})
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
        cached = model(**prompt, use_cache=True)
        cache_shapes = []
        for i, expected in enumerate(layer_shapes):
            key, value = cached.past_key_values[i]
            if tuple(key.shape[1::2]) != (expected['heads'], expected['head_dim']) or key.shape != value.shape:
                raise AssertionError('实际 KV cache 的头数或每头维度错误。')
            cache_shapes.append(list(key.shape))
        next_token = cached.logits[:, -1:].argmax(dim=-1)
        continuation = model(input_ids=next_token, past_key_values=cached.past_key_values, use_cache=True)
        if not torch.isfinite(continuation.logits).all():
            raise FloatingPointError('缓存续写输出非有限。')
        generated = model.generate(**prompt, max_new_tokens=8, do_sample=False,
                                   use_cache=True, pad_token_id=tokenizer.eos_token_id)
    model.train()
    model.zero_grad(set_to_none=True)
    loss = model(**batch, use_cache=False).loss
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    report = {'passed': True, 'parameters': parameter_count(model),
              'pruning_mode': result.get('pruning_mode', 'joint'),
              'layer_shapes': layer_shapes, 'kv_cache_shapes': cache_shapes,
              'cached_continuation_finite': True,
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
