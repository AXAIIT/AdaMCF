"""真实 Phi-2 的 AdaMCF/HESSO-CRIC 结构剪枝入口。

顺序：读取冻结数据 → 加载原权重 → CRIC 全流程 → 精确零组检查 →
物理压缩并重载 → 可选的独立 LoRA 恢复 → 保存结构包。数据预处理独立执行。
"""
import argparse
import copy
import tempfile
from dataclasses import asdict, dataclass, field
from functools import partial
import gc
import json
import logging
from datetime import datetime, timezone, timedelta
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
from phi2_data import PreparedDataset, collate_examples, collate_alpha_examples, file_sha256, load_manifest, to_device
from only_train_once.model_adapters.phi import group_summary, parameter_count
from phi2_precision import audit_structure_precision
from only_train_once.optimizer.alpha_data import AlphaEvaluationPool


@dataclass
class PruningConfig:
    """默认值用于功能验收，不是论文完整训练配方或正式任务评测配置。"""
    pruning_mode: str = 'joint'
    export_format: str = 'safetensors'
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
    alpha_batch_size: int = 0
    alpha_seed: int = 43
    recovery: dict = field(default_factory=dict)
    grad_clip_norm: float = 1.0
    log_every: int = 10
    equivalence_atol: float = 0.0002
    equivalence_rtol: float = 0.0002
    onnx_atol: float = 0.001
    onnx_rtol: float = 0.001


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def run_logger(output):
    """终端和 UTF-8 run.log 同步输出；重复导出追加日志，拒绝覆盖仍在上层检查。"""
    output = Path(output)
    logger = logging.getLogger('adamcf.phi2')
    target = str(output.resolve())
    if getattr(logger, '_run_directory', None) != target:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        class ShanghaiFormatter(logging.Formatter):
            def formatTime(self, record, datefmt=None):
                return datetime.fromtimestamp(record.created, timezone(timedelta(hours=8))).isoformat(timespec='seconds')
        formatter = ShanghaiFormatter('%(asctime)s | %(levelname)s | %(message)s')
        for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(output/'run.log',encoding='utf-8')):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger._run_directory = target
    return logger


def log_stage(output, stage, message, **fields):
    """人类可读的阶段说明与 JSONL 事件共用字段，避免日志和结果口径不同。"""
    timestamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    run_logger(output).info('[%s] %s | %s',stage,message,json.dumps(fields,ensure_ascii=False))
    with (Path(output)/'events.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps({'time':timestamp,'stage':stage,'message':message,**fields},ensure_ascii=False)+'\n')


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
    """全梯度 CRIC；post_pruning 是本次联合训练的尾段，不是 LoRA 恢复。"""
    # alpha 与 train/eval 是互斥的原始行索引，预处理清单已经核对。
    alpha_pool = AlphaEvaluationPool(alpha_loader)
    maximum_steps = (optimizer.start_global_sampling_step
                     + config.sampling_steps * config.max_cycle_period
                     + config.hybrid_training_steps + config.post_pruning_steps + 2)
    iterator = iter(train_loader)
    started = time.perf_counter()
    previous_phase = None
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
            alpha_ids, alpha_tokens, alpha_batch_count = [], 0, 0
            if optimizer.will_update_alpha_on_next_step():
                updated = optimizer.alpha_scheduler.update(model, None, alpha_pool)
                if updated:
                    stats=optimizer.alpha_scheduler.last_eval_stats
                    alpha_ids=stats['sample_ids']
                    alpha_batch_count=stats['microbatches']
                    alpha_tokens=stats['supervised_tokens']
            optimizer.step(loss=loss.detach())
            record = {'step': optimizer.num_steps, 'phase': phase_name(optimizer),
                      'loss': float(loss.detach()), 'gradient_norm_before_clip': float(norm),
                      'cric_cycle': optimizer.curr_cycle_period,
                      'alpha': optimizer.alpha_scheduler.alpha.detach().cpu().tolist(),
                      'alpha_updated': updated,
                      'alpha_row_ids':alpha_ids,'alpha_microbatches':alpha_batch_count,
                      'alpha_supervised_tokens':alpha_tokens,
                      'alpha_unique_samples_seen':len(alpha_pool.used_sample_ids),
                      'learning_rate':optimizer.param_groups[0]['lr'],
                      'pruned_logical_groups':sum(len(g.get('pruned_idxes',[])) for g in optimizer.param_groups if g['is_prunable']),
                      'elapsed_seconds': time.perf_counter() - started}
            trace.write(json.dumps(record) + '\n')
            if optimizer.num_steps == 1 or optimizer.num_steps % config.log_every == 0 or updated or record['phase'] != previous_phase:
                record.update(memory_stats(device))
                write_json(output / 'status.json', {'state': 'running', **record})
                log_stage(output,'训练','CRIC 进度；α 仅在周期边界更新。',**record)
            previous_phase = record['phase']
            if (optimizer.is_cric_terminated and optimizer.num_steps >=
                    optimizer.cric_terminated_step + config.hybrid_training_steps + config.post_pruning_steps):
                return record
    raise RuntimeError('达到步数保护上限，CRIC 未完成；不生成成功的压缩结果。')


def run(args):
    """阶段一全梯度剪枝；压缩重载后才进入独立 LoRA 阶段，不执行七任务评测。"""
    config = PruningConfig(**json.loads(args.config.read_text(encoding='utf-8')))
    if getattr(args,'export_format',None) is not None:
        config.export_format=args.export_format
    if config.export_format not in ('pt','torch','onnx','safetensors'):
        raise ValueError('导出格式必须为 pt/torch、onnx 或 safetensors。')
    if (config.batch_size < 1 or config.alpha_eval_batches < 1 or config.alpha_batch_size < 0 or config.log_every < 1
            or config.post_pruning_steps < 0 or config.warmup_steps < 0):
        raise ValueError('批量和日志间隔必须为正，预热和剪后尾段步数不能为负。')
    from finetune_phi2_lora import RecoveryConfig, require_peft, validate_config
    recovery_config = RecoveryConfig(**config.recovery)
    validate_config(recovery_config)
    if recovery_config.enabled and not args.profile_only:
        require_peft()  # 开始长时间剪枝前发现依赖问题，脚本不自动安装或升级包。
    if args.output.exists():
        raise FileExistsError('输出目录已存在；请为新实验使用独立目录。')
    args.output.mkdir(parents=True)
    args.created_output = True
    log_stage(args.output,'1/6 数据','读取冻结的监督数据；内部 eval NLL 与七项 benchmark 准确率分开记录。',config=asdict(config))
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
    log_stage(args.output,'1/6 数据','三个角色分别用于梯度更新、α 更新和独立 NLL。',
              split_rows={k:manifest['splits'][k]['rows'] for k in ('train','alpha','eval')},
              max_length=manifest['max_length'],loss_on=manifest['loss_on'],
              alpha_microbatch_limit=config.alpha_eval_batches,data_manifest_sha256=file_sha256(args.data_dir/'manifest.json'))
    for name, expected_sha256 in manifest['tokenizer_files'].items():
        path = args.model_dir / name
        if not path.is_file() or file_sha256(path) != expected_sha256:
            raise ValueError(f'模型目录的 {name} 与冻结数据使用的分词器/配置不一致。')
    collate = partial(collate_examples, pad_token_id=manifest['pad_token_id'])
    generator = torch.Generator().manual_seed(config.seed)
    alpha_generator = torch.Generator().manual_seed(config.alpha_seed)
    loaders = {}
    for split in ('train', 'alpha', 'eval'):
        dataset = PreparedDataset(args.data_dir, split, manifest)
        loaders[split] = DataLoader(dataset, batch_size=(config.alpha_batch_size or config.batch_size) if split=='alpha' else config.batch_size,
                                    shuffle=split in ('train','alpha'),
                                    generator=generator if split=='train' else alpha_generator if split=='alpha' else None,
                                    collate_fn=partial(collate_alpha_examples,pad_token_id=manifest['pad_token_id']) if split=='alpha' else collate,
                                    num_workers=0)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = PhiForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation='eager').to(device)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    log_stage(args.output,'2/6 模型','加载 FP32 原生模型，剪枝阶段不附加 LoRA。',
              parameters=parameter_count(model),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
              lora_parameters=sum(p.numel() for n,p in model.named_parameters() if 'lora_' in n.lower()),
              dtype=str(next(model.parameters()).dtype),autocast=False,tf32=False,export_format=config.export_format)
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
    log_stage(args.output,'3/6 剪前诊断','仅报告训练来源内部保留子集的监督 NLL。',**baseline)
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
    log_stage(args.output,'4/6 剪枝','开始完整 CRIC 流程，候选比例不等于参数减少比例。',**{'parameters': before_count,
                      'prunable_parameter_groups': len(optimizer.prunable_param_group_dict),
                      'logical_groups': optimizer.total_num_groups,
                      'projection_steps': optimizer.start_global_sampling_step - optimizer.start_cric_step,
                      'baseline': baseline, **memory_stats(device)})
    if args.profile_only:
        # 用冻结训练池最长样本探测峰值，避免只测到第一个短样本。
        worst=sorted(loaders['train'].dataset.rows,key=lambda row:len(row['input_ids']),reverse=True)[:config.batch_size]
        batch = to_device(collate(worst), device)
        model.train()
        model(**batch, use_cache=False).loss.backward()
        optimizer.preview_grad_variant()
        optimizer.compute_importance_scores()
        if config.adaptive_alpha:
            alpha_size=config.alpha_batch_size or config.batch_size
            alpha_worst=sorted(loaders['alpha'].dataset.rows,key=lambda row:len(row['input_ids']),reverse=True)[:alpha_size]
            model.eval()
            with torch.no_grad():model(**to_device(collate(alpha_worst),device),use_cache=False)
        profile = {'state': 'profile_only', 'pruning_performed': False,
                   'profile_train_length':int(batch['input_ids'].shape[1]),
                   'profile_alpha_batch_size':(config.alpha_batch_size or config.batch_size) if config.adaptive_alpha else 0,
                   **memory_stats(device)}
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
    log_stage(args.output,'5/6 剪后诊断','结构尚未物理删除；这是全梯度稀疏模型的 NLL，不是 LoRA 恢复准确率。',**sparse_eval,**counts)
    probe = to_device(next(iter(loaders['eval'])), device)
    probe = {k: v[:1] for k, v in probe.items()}
    with torch.no_grad():
        sparse_logits = model(**probe, use_cache=False).logits.detach()
    # 在更换 Parameter 之前释放旧优化器的参数引用、缓存和梯度。
    score_criteria_for_report=dict(optimizer.importance_score_criteria)
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
              'method':'adamcf_hesso_cric','importance_score_criteria':score_criteria_for_report,
              'lora_recovery_performed':False,'training_dtype':'float32','export_format':config.export_format,
              'lora_recovery_requested':recovery_config.enabled,
              'onnx_atol':config.onnx_atol,'onnx_rtol':config.onnx_rtol,
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
    # ONNX 不能训练；需要恢复时先保存可训练中间包，最终仍按用户选择导出。
    base_format = 'safetensors' if recovery_config.enabled and config.export_format=='onnx' else config.export_format
    export_run(args.output, device, base_format)
    if recovery_config.enabled:
        from finetune_phi2_lora import run_recovery
        write_json(args.output/'status.json',{'state':'running','stage':'lora_recovery',
                                             'compressed_package':str(args.output/'compressed_model')})
        recovery = run_recovery(args.output/'compressed_model',args.data_dir,
                                args.output/'lora_recovery',recovery_config,device,
                                export_format=config.export_format)
        pruning = json.loads((args.output/'pruning_result.json').read_text(encoding='utf-8'))
        workflow = {'state':'completed','two_stage':True,'pruning':pruning,'recovery':recovery,
                    'full_benchmark_evaluated':False}
        write_json(args.output/'workflow_result.json',workflow)
        write_json(args.output/'status.json',workflow)
        log_stage(args.output,'两阶段完成','CRIC 压缩与独立 LoRA 恢复均完成；任务准确率仍需另行评测。',
                  recovered_package=recovery['recovered_package'])



def export_run(run, device, export_format=None):
    """从稀疏检查点调用统一导出；格式只改变保存/推理接口，不改变剪枝决策。"""
    run, device = Path(run), torch.device(device)
    if (run / 'compressed_model').exists():
        raise FileExistsError('压缩包已存在，拒绝覆盖。')
    context = json.loads((run / 'export_context.json').read_text(encoding='utf-8'))
    selected_format=export_format or context.get('export_format','safetensors')
    if selected_format not in ('pt','torch','onnx','safetensors'):
        raise ValueError('导出格式必须为 pt/torch、onnx 或 safetensors。')
    log_stage(run,'6/6 导出','加载既有稀疏检查点，不重复训练；执行结构审计后发布。',export_format=selected_format)
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
        oto.construct_subnet(out_dir=staging.name, ckpt_format=selected_format,
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
    dynamic_probes=[]
    if selected_format == 'onnx':
        # 独立进程验证不同 batch/长度，防止 ONNX 只有样例 shape 能运行。
        for batch_size,length in ((1,7),(2,5)):
            ids=batch['input_ids'][:1,:length].repeat(batch_size,1)
            mask=torch.ones_like(ids)
            if batch_size==2:mask[1,-1]=0
            with torch.no_grad():logits=compressed(input_ids=ids,attention_mask=mask,use_cache=False).logits
            dynamic_probes.append({'input_ids':ids.cpu(),'attention_mask':mask.cpu(),'logits':logits.cpu()})
    torch.save({'batch': probe['batch'], 'sparse_logits': probe['sparse_logits'],
                'compressed_logits': compact_logits.cpu(), 'atol': probe['atol'],
                'rtol': probe['rtol'],'onnx_dynamic_probes':dynamic_probes}, run / 'equivalence_probe.pt')
    del sparse, reloaded_sparse
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    package = run / 'compressed_model'
    (run / 'pruning_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    report = {**context, 'functional_pruning_passed': True,
              'export_format':selected_format,
              'onnx_atol':context.get('onnx_atol',1e-3),'onnx_rtol':context.get('onnx_rtol',1e-3),
              'parameters_after': after_count,
              'parameter_reduction_ratio': 1 - after_count / context['parameters_before'],
              'precision_audit': diagnostics,
              'sparse_compressed_logits_max_abs_error': diagnostics['native_fp32']['max_abs_error'],
              'compressed_package': str(package)}
    architecture=json.loads((package/'architecture.json').read_text(encoding='utf-8'))
    weights_file=architecture.get('weights_file','model.safetensors')
    report.update(compressed_weights_file=weights_file,
                  compressed_weights_sha256=file_sha256(package/weights_file),
                  compressed_files_sha256={str(p.relative_to(package)):file_sha256(p) for p in sorted(package.rglob('*')) if p.is_file()})
    if device.type == 'cuda':
        report.update(peak_allocated_GiB=torch.cuda.max_memory_allocated(device) / 2**30,
                      peak_reserved_GiB=torch.cuda.max_memory_reserved(device) / 2**30)
    write_json(run/'pruning_result.json',{'state':'completed',**report})
    write_json(run/'status.json',{'state':'pruning_completed' if context.get('lora_recovery_requested') else 'completed',**report})
    log_stage(run,'完成','结构审计与导出完成；正式任务准确率及 LoRA 恢复尚未执行。',
              export_format=selected_format,parameters_after=after_count,
              reduction=report['parameter_reduction_ratio'],structure_fp64=diagnostics['structure_fp64'],
              native_fp32=diagnostics['native_fp32'],package=str(package))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'phi2_two_stage_smoke_config.json')
    parser.add_argument('--model-dir', type=Path, default=HERE / 'models/phi-2')
    parser.add_argument('--data-dir', type=Path, help='训练模式必填；仅导出时不需要。')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--profile-only', action='store_true', help='只测全梯度、评分与缓存显存，不执行剪枝。')
    parser.add_argument('--export-only', action='store_true', help='仅从 --output 的已有稀疏检查点调用框架导出，无需重训。')
    parser.add_argument('--export-format',choices=('pt','torch','onnx','safetensors'),default=None,
                        help='压缩产物格式；覆盖配置，旧配置默认 safetensors。ONNX 仅供推理。')
    args = parser.parse_args()
    if args.export_only:
        if args.profile_only:
            parser.error('--export-only 与 --profile-only 不能同时使用。')
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            export_run(args.output, args.device, args.export_format)
        except Exception:
            if (args.output/'run.log').is_file() and not (args.output/'compressed_model').exists():
                run_logger(args.output).exception('导出失败；稀疏检查点保留，可修正问题后重试。')
            raise
        return
    if args.data_dir is None:
        parser.error('训练模式需要 --data-dir；已有稀疏检查点可使用 --export-only。')
    try:
        run(args)
    except Exception as error:
        if getattr(args, 'created_output', False):
            write_json(args.output / 'failure.json', {'state': 'failed', 'error': repr(error)})
            write_json(args.output / 'status.json', {'state': 'failed', 'error': repr(error),
                       'pruning_result_preserved':(args.output/'pruning_result.json').is_file()})
            run_logger(args.output).exception('运行失败，未把失败记录标记为成功。')
        raise


if __name__ == '__main__':
    main()
