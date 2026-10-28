"""真实 Phi-2 的 AdaMCF/HESSO-CRIC 结构剪枝入口。

顺序：读取冻结数据 → 加载原权重 → CRIC 全流程 → 精确零组检查 →
物理压缩 → 稀疏/压缩输出对齐 → 保存结构包。数据预处理独立执行。
"""
import argparse
import copy
import tempfile
from dataclasses import asdict, dataclass
from functools import partial
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PhiForCausalLM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from only_train_once import OTO
from phi2_data import PreparedDataset, collate_examples, file_sha256, load_manifest, to_device
from only_train_once.model_adapters.phi import group_summary, parameter_count
from phi2_precision import audit_structure_precision


@dataclass
class PruningConfig:
    pruning_mode: str = 'joint'
    seed: int = 42
    batch_size: int = 1
    learning_rate: float = 0.0001
    optimizer_variant: str = 'sgd'
    first_momentum: float = 0.0
    weight_decay: float = 0.0
    target_group_sparsity: float = 0.1
    warmup_steps: int = 2
    sampling_steps: int = 2
    max_cycle_period: int = 2
    hybrid_training_steps: int = 8
    post_pruning_steps: int = 4
    adaptive_alpha: bool = True
    alpha_eval_batches: int = 1
    grad_clip_norm: float = 1.0
    log_every: int = 10
    equivalence_atol: float = 0.0002
    equivalence_rtol: float = 0.0002


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def memory_stats(device):
    if device.type != 'cuda':
        return {}
    torch.cuda.synchronize(device)
    return {'allocated_GiB': torch.cuda.memory_allocated(device) / 2**30,
            'peak_allocated_GiB': torch.cuda.max_memory_allocated(device) / 2**30,
            'peak_reserved_GiB': torch.cuda.max_memory_reserved(device) / 2**30}


@torch.no_grad()
def evaluate_response_loss(model, loader, device):
    """报告按响应 token 加权的 NLL，不将它称为 WikiText 困惑度或七任务准确率。"""
    model.eval()
    total_loss, tokens = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        count = int((batch['labels'][:, 1:] != -100).sum())
        loss = model(**batch, use_cache=False).loss
        if not torch.isfinite(loss):
            raise FloatingPointError('评估 loss 非有限。')
        total_loss += float(loss) * count
        tokens += count
    if not tokens:
        raise ValueError('评估数据没有监督 token。')
    nll = total_loss / tokens
    return {'response_nll': nll, 'exp_response_nll': math.exp(min(nll, 80)),
            'supervised_tokens': tokens, 'samples': len(loader.dataset)}


def phase_name(optimizer):
    if optimizer.is_cric_terminated:
        end = optimizer.cric_terminated_step + optimizer.hybrid_training_steps
        return 'hybrid' if optimizer.num_steps <= end else 'post_pruning'
    if optimizer.num_steps < optimizer.start_cric_step:
        return 'warmup'
    return 'projection' if optimizer.num_steps < optimizer.start_global_sampling_step else 'sampling'


def train_cric(model, optimizer, train_loader, alpha_loader, config, output, device):
    # alpha 与 train/eval 是互斥的原始行索引，预处理清单已经核对。
    alpha_batches = []
    for batch in alpha_loader:
        alpha_batches.append((to_device(batch, device), None))
        if len(alpha_batches) >= config.alpha_eval_batches:
            break
    maximum_steps = (optimizer.start_global_sampling_step
                     + config.sampling_steps * config.max_cycle_period
                     + config.hybrid_training_steps + config.post_pruning_steps + 2)
    iterator = iter(train_loader)
    started = time.perf_counter()
    with (output / 'training_trace.jsonl').open('w', encoding='utf-8', buffering=1) as trace:
        for _ in range(maximum_steps):
            try:
                cpu_batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                cpu_batch = next(iterator)
            batch = to_device(cpu_batch, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise FloatingPointError('训练 loss 非有限。')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip_norm, error_if_nonfinite=True)
            updated = False
            if optimizer.will_update_alpha_on_next_step():
                updated = optimizer.alpha_scheduler.update(model, None, alpha_batches)
            optimizer.step(loss=loss.detach())
            record = {'step': optimizer.num_steps, 'phase': phase_name(optimizer),
                      'loss': float(loss.detach()), 'gradient_norm_before_clip': float(norm),
                      'cric_cycle': optimizer.curr_cycle_period,
                      'alpha': optimizer.alpha_scheduler.alpha.detach().cpu().tolist(),
                      'alpha_updated': updated,
                      'elapsed_seconds': time.perf_counter() - started}
            trace.write(json.dumps(record) + '\n')
            if optimizer.num_steps == 1 or optimizer.num_steps % config.log_every == 0 or updated:
                record.update(memory_stats(device))
                write_json(output / 'status.json', {'state': 'running', **record})
                print(json.dumps(record), flush=True)
            if (optimizer.is_cric_terminated and optimizer.num_steps >=
                    optimizer.cric_terminated_step + config.hybrid_training_steps + config.post_pruning_steps):
                return record
    raise RuntimeError('达到步数保护上限，CRIC 未完成；不生成成功的压缩结果。')


def run(args):
    config = PruningConfig(**json.loads(args.config.read_text(encoding='utf-8')))
    if (config.batch_size < 1 or config.alpha_eval_batches < 1 or config.log_every < 1
            or config.post_pruning_steps < 1 or config.warmup_steps < 0):
        raise ValueError('批量、日志间隔和剪后步数必须为正，预热步数不能为负。')
    if args.output.exists():
        raise FileExistsError('输出目录已存在；请为新实验使用独立目录。')
    args.output.mkdir(parents=True)
    args.created_output = True
    device = torch.device(args.device)
    torch.set_num_threads(4)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    manifest = load_manifest(args.data_dir)
    for name, expected_sha256 in manifest['tokenizer_files'].items():
        path = args.model_dir / name
        if not path.is_file() or file_sha256(path) != expected_sha256:
            raise ValueError(f'模型目录的 {name} 与冻结数据使用的分词器/配置不一致。')
    collate = partial(collate_examples, pad_token_id=manifest['pad_token_id'])
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {}
    for split in ('train', 'alpha', 'eval'):
        dataset = PreparedDataset(args.data_dir, split, manifest)
        loaders[split] = DataLoader(dataset, batch_size=config.batch_size,
                                    shuffle=split == 'train', generator=generator if split == 'train' else None,
                                    collate_fn=collate, num_workers=0)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = PhiForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation='eager').to(device)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    inputs = {'model_dir': str(args.model_dir), 'data_dir': str(args.data_dir),
              'data_manifest_sha256': file_sha256(args.data_dir / 'manifest.json'),
              'weights': {p.name: file_sha256(p) for p in sorted(args.model_dir.glob('*.safetensors'))},
              'config': asdict(config), 'device': str(device), 'dtype': str(next(model.parameters()).dtype),
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'code_sha256': {p.name: file_sha256(p) for p in HERE.glob('*.py')}}
    framework = HERE.parents[1] / 'only_train_once'
    inputs['framework_code_sha256'] = {
        str(p.relative_to(framework)): file_sha256(p)
        for p in [framework/'__init__.py', *sorted((framework/'model_adapters').glob('*.py')),
                  framework/'operation/native_phi.py', framework/'optimizer/hesso_cric.py',
                  framework/'optimizer/cric_adapter.py', framework/'optimizer/cric_alpha.py']}
    write_json(args.output / 'inputs.json', inputs)
    before_count = parameter_count(model)
    baseline = evaluate_response_loss(model, loaders['eval'], device)
    oto = OTO(model=model, pruning_mode=config.pruning_mode)
    groups = oto.get_param_groups()
    write_json(args.output / 'parameter_groups.json', group_summary(groups))
    optimizer = oto.hessocric(
        variant=config.optimizer_variant, lr=config.learning_rate,
        first_momentum=config.first_momentum, weight_decay=config.weight_decay,
        target_group_sparsity=config.target_group_sparsity,
        start_cric_step=config.warmup_steps + 1,
        sampling_steps=config.sampling_steps, max_cycle_period=config.max_cycle_period,
        hybrid_training_steps=config.hybrid_training_steps,
        adaptive_alpha=config.adaptive_alpha,
        alpha_scheduler_cfg={'warmup_updates': 0, 'eval_batch_limit': config.alpha_eval_batches})
    optimizer.phi_pruning_mode = config.pruning_mode
    print(json.dumps({'stage': 'loaded', 'parameters': before_count,
                      'prunable_parameter_groups': len(optimizer.prunable_param_group_dict),
                      'logical_groups': optimizer.total_num_groups,
                      'projection_steps': optimizer.start_global_sampling_step - optimizer.start_cric_step,
                      'baseline': baseline, **memory_stats(device)}), flush=True)
    if args.profile_only:
        batch = to_device(next(iter(loaders['train'])), device)
        model.train()
        model(**batch, use_cache=False).loss.backward()
        optimizer.preview_grad_variant()
        optimizer.compute_importance_scores()
        profile = {'state': 'profile_only', 'pruning_performed': False, **memory_stats(device)}
        write_json(args.output / 'profile.json', profile)
        print(json.dumps(profile), flush=True)
        return

    final_step = train_cric(model, optimizer, loaders['train'], loaders['alpha'], config, args.output, device)
    plan = oto.collect_pruning_plan()
    plan['parameters_before'] = before_count
    plan['pruned_logical_groups'] = sum(len(entry[kind]['pruned'])
                                       for entry in plan['layers'] for kind in ('ffn', 'attention'))
    plan['pruned_logical_groups'] += sum(len(e['head_dim_pruned_units']) for e in plan['layers'])
    plan['alpha_final'] = optimizer.alpha_scheduler.alpha.detach().cpu().tolist()
    plan['alpha_updates'] = optimizer.alpha_scheduler.update_count
    counts = {
        'pruned_ffn_neurons': sum(len(e['ffn']['pruned']) for e in plan['layers']),
        'pruned_attention_heads': sum(len(e['attention']['pruned']) for e in plan['layers']),
        'pruned_head_dim_units': sum(len(e['head_dim_pruned_units']) for e in plan['layers']),
        'pruned_head_dimensions_per_head_sum': sum(len(e['head_dim_pruned']) for e in plan['layers']),
        'pruned_attention_channels': sum(
            e['attention']['original'] * e['head_dim']
            - len(e['attention']['keep']) * len(e['head_dim_keep']) for e in plan['layers']),
        'candidate_logical_groups': optimizer.total_num_groups,
        'candidate_parameter_groups': len(optimizer.prunable_param_group_dict),
    }
    expected = {'ffn': 'pruned_ffn_neurons', 'num_head': 'pruned_attention_heads',
                'head_dim': 'pruned_head_dim_units'}
    if config.pruning_mode in expected:
        if counts[expected[config.pruning_mode]] <= 0:
            raise AssertionError('指定模式没有产生真实 CRIC 删除结果。')
        for mode, key in expected.items():
            if mode != config.pruning_mode and counts[key] != 0:
                raise AssertionError('指定模式误删了其他类型结构。')
    sparse_eval = evaluate_response_loss(model, loaders['eval'], device)
    probe = to_device(next(iter(loaders['eval'])), device)
    probe = {k: v[:1] for k, v in probe.items()}
    with torch.no_grad():
        sparse_logits = model(**probe, use_cache=False).logits.detach()
    # 在更换 Parameter 之前释放旧优化器的参数引用、缓存和梯度。
    optimizer.zero_grad(set_to_none=True)
    del optimizer, groups, oto
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    # 先持久保存稀疏状态，导出诊断失败时不必再跑 384 步分层试探。
    sparse_dir = args.output / 'sparse_model'
    model.save_pretrained(sparse_dir, safe_serialization=True, max_shard_size='5GB')
    tokenizer.save_pretrained(sparse_dir)
    write_json(args.output / 'pruning_plan.json', plan)
    torch.save({'batch': {k: v.cpu() for k, v in probe.items()},
                'sparse_logits': sparse_logits.cpu(), 'atol': config.equivalence_atol,
                'rtol': config.equivalence_rtol}, args.output / 'sparse_probe.pt')
    context = {'full_benchmark_evaluated': False,
              'pruning_mode': config.pruning_mode,
              'parameters_before': before_count,
              'target_logical_group_sparsity': config.target_group_sparsity,
              'pruned_logical_groups': plan['pruned_logical_groups'],
              **counts,
              'alpha_final': plan['alpha_final'], 'alpha_updates': plan['alpha_updates'],
              'final_training_step': final_step, 'baseline_eval': baseline, 'sparse_eval': sparse_eval,
              'training_memory': memory_stats(device)}
    write_json(args.output / 'export_context.json', context)
    del model, sparse_logits
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    export_run(args.output, device)



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
    # 框架统一调度物理裁剪与结构包保存。先写唯一临时目录，审计通过后
    # 原子发布为 compressed_model；失败不会留下看似可用的最终模型包。
    staging = tempfile.TemporaryDirectory(prefix='.export-', dir=run)
    try:
        oto = OTO(model=compressed, pruning_mode=plan.get('pruning_mode', 'joint'))
        oto.construct_subnet(out_dir=staging.name, ckpt_format='safetensors',
                             save_full_group_sparse_model=False,
                             pruning_plan=plan, tokenizer=tokenizer)
        plan = oto.pruning_plan
        after_count = parameter_count(compressed)
        if not after_count < context['parameters_before']:
            raise AssertionError('实际参数量没有减少。')
        diagnostics, compact_logits = audit_structure_precision(sparse, compressed, batch)
        Path(oto.compressed_model_path).rename(run / 'compressed_model')
    finally:
        staging.cleanup()
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
    parser.add_argument('--config', type=Path, default=HERE / 'phi2_smoke_config.json')
    parser.add_argument('--model-dir', type=Path, default=HERE / 'models/phi-2')
    parser.add_argument('--data-dir', type=Path, help='训练模式必填；仅导出时不需要。')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--profile-only', action='store_true', help='只测全梯度、评分与缓存显存，不执行剪枝。')
    parser.add_argument('--export-only', action='store_true', help='仅从 --output 的已有稀疏检查点调用框架导出，无需重训。')
    args = parser.parse_args()
    if args.export_only:
        if args.profile_only:
            parser.error('--export-only 与 --profile-only 不能同时使用。')
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        export_run(args.output, args.device)
        return
    if args.data_dir is None:
        parser.error('训练模式需要 --data-dir；已有稀疏检查点可使用 --export-only。')
    try:
        run(args)
    except Exception as error:
        if getattr(args, 'created_output', False):
            write_json(args.output / 'failure.json', {'state': 'failed', 'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
