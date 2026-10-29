"""第二阶段：重载物理压缩的 Phi，只训练 LoRA A/B，保存适配器和合并模型。

数据必须先由独立预处理脚本冻结。本文件不剪枝、不更新 α、不训练基座，
也不把内部响应 NLL 当作七任务准确率。可由剪枝入口调用或单独运行。
"""
import argparse
from dataclasses import asdict, dataclass, field
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from only_train_once import OTO
from only_train_once.model_adapters.phi import save_pruned_package
from phi2_data import PreparedDataset, collate_examples, file_sha256, load_manifest, to_device


@dataclass
class RecoveryConfig:
    """训练步数按 optimizer.step 计；默认禁用以兼容既有剪枝配置。"""
    enabled: bool = False
    seed: int = 44
    steps: int = 100
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = .01
    warmup_steps: int = 0
    rank: int = 8
    lora_alpha: int = 16
    dropout: float = .05
    target_modules: list = field(default_factory=lambda: ['q_proj','k_proj','v_proj','dense','fc1','fc2'])
    grad_clip_norm: float = 1.
    eval_every: int = 50
    log_every: int = 10


def require_peft():
    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    except ImportError as error:
        raise ImportError('LoRA 恢复需要 PEFT；已验证版本见 requirements-lora.txt，请安装到个人环境。') from error
    return LoraConfig, PeftModel, TaskType, get_peft_model


def validate_config(config):
    for key in ('steps','batch_size','gradient_accumulation_steps','rank','lora_alpha','eval_every','log_every'):
        value = getattr(config, key)
        if not isinstance(value,int) or isinstance(value,bool) or value < 1:
            raise ValueError(f'LoRA {key} 必须为正整数。')
    if not isinstance(config.warmup_steps,int) or not 0 <= config.warmup_steps < config.steps:
        raise ValueError('LoRA warmup_steps 必须在 [0, steps) 内。')
    for key in ('learning_rate','weight_decay','grad_clip_norm','dropout'):
        if not math.isfinite(getattr(config,key)):
            raise ValueError(f'LoRA {key} 必须为有限数。')
    if config.learning_rate <= 0 or config.weight_decay < 0 or config.grad_clip_norm <= 0 or not 0 <= config.dropout < 1:
        raise ValueError('LoRA 学习率/裁剪阈值必须为正，衰减非负，dropout 在 [0,1) 内。')
    if (not config.target_modules or len(set(config.target_modules)) != len(config.target_modules)
            or not set(config.target_modules) <= {'q_proj','k_proj','v_proj','dense','fc1','fc2'}):
        raise ValueError('LoRA target_modules 必须是 Phi 投影层名称的非空、无重复子集。')


def parameter_fingerprint(named_parameters):
    """逐张量计算摘要，不复制一整份大模型；名字来自包装前的同一组参数引用。"""
    digest = hashlib.sha256()
    for name, parameter in named_parameters:
        digest.update(f'{name}:{tuple(parameter.shape)}:{parameter.dtype}'.encode())
        data = parameter.detach().cpu().contiguous().view(torch.uint8).numpy()
        digest.update(memoryview(data).cast('B'))
    return digest.hexdigest()


def attach_lora(base, config):
    """按压缩后的实际 Linear 尺寸创建 A/B，禁止误解冻基座或其他参数。"""
    LoraConfig, _, TaskType, get_peft_model = require_peft()
    found = {name.rsplit('.',1)[-1] for name, module in base.named_modules() if isinstance(module,nn.Linear)}
    if not set(config.target_modules) <= found:
        raise ValueError('压缩模型缺少指定 LoRA 投影层。')
    base.requires_grad_(False)
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=config.rank, lora_alpha=config.lora_alpha,
        lora_dropout=config.dropout, target_modules=config.target_modules, bias='none'))
    trainable = []
    for name, parameter in model.named_parameters():
        is_ab = any(part in ('lora_A','lora_B') for part in name.split('.'))
        parameter.requires_grad_(is_ab)
        if is_ab:
            trainable.append((name,parameter))
    if not trainable:
        raise RuntimeError('没有可训练的 LoRA A/B。')
    return model, trainable


def package_hashes(base_dir):
    return {p.name:file_sha256(p) for p in sorted(Path(base_dir).iterdir()) if p.is_file()}


def load_recovery(directory, device='cpu', base_dir=None):
    """显式加载 OTO 异构基座后附加 adapter，不能由 AutoModel 猜原始密集结构。"""
    directory = Path(directory)
    manifest = json.loads((directory/'recovery_manifest.json').read_text(encoding='utf-8'))
    base_dir = Path(base_dir or manifest['base_package'])
    for name, expected in manifest['base_files'].items():
        if not (base_dir/name).is_file() or file_sha256(base_dir/name) != expected:
            raise ValueError('LoRA 所依赖的压缩基座摘要不一致。')
    for name, expected in manifest.get('adapter_files',{}).items():
        path=directory/'adapter'/name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError('LoRA adapter 文件摘要不一致。')
    _, PeftModel, _, _ = require_peft()
    base = OTO.load_subnet(base_dir,device=device)
    return PeftModel.from_pretrained(base,directory/'adapter',is_trainable=False,local_files_only=True).eval()


def export_recovered_model(model, tokenizer, architecture, output, export_format, dataset, pad_token_id):
    """只合并与导出，不做梯度更新；供训练尾部和断点导出共用。"""
    from phi2_precision import merge_lora_with_precision_audit
    from prune_phi2_hesso_cric import write_json
    indices = sorted({0,len(dataset)//3,2*len(dataset)//3})
    probes = [collate_examples([dataset[i]],pad_token_id) for i in indices]
    model = model.cpu().eval()
    merged,audit,logits = merge_lora_with_precision_audit(model,probes)
    audit['internal_eval_row_ids'] = [dataset[i]['row_id'] for i in indices]
    write_json(output/'merge_precision_audit.json',audit)
    save_pruned_package(merged,tokenizer,architecture,output/'recovered_model',weights_format=export_format)
    torch.save({'batch':probes[0],'logits':logits[0]},output/'recovery_probe.pt')
    del merged,model;gc.collect()
    loaded=OTO.load_subnet(output/'recovered_model',device='cpu')
    reload_errors=[]
    for probe,reference in zip(probes,logits):
        if export_format=='onnx':
            actual=torch.from_numpy(loaded(probe['input_ids'],probe['attention_mask']))
            torch.testing.assert_close(actual,reference,atol=1e-3,rtol=1e-3)
        else:
            loaded.eval()
            with torch.no_grad():actual=loaded(**probe,use_cache=False).logits
            torch.testing.assert_close(actual,reference,atol=1e-5,rtol=1e-4)
        reload_errors.append(float((actual-reference).abs().max()))
    return {'recovered_package':str(output/'recovered_model'),'merged_reload_verified':True,
            'reload_logits_max_abs_error':max(reload_errors),'merge_precision_audit':audit}


def finalize_saved_recovery(directory, export_format='safetensors'):
    """从已完成训练的 adapter 续做导出，拒绝覆盖，保留原失败记录。

    先校验基座/adapter/数据摘要与完整训练轨迹。旧运行没有独立训练结果
    文件时，只接受有冻结基座检查和完成事件的日志，不把半途训练当完成。
    """
    from prune_phi2_hesso_cric import write_json,log_stage
    directory=Path(directory)
    if (directory/'recovered_model').exists() or (directory/'recovery_result.json').exists():
        raise FileExistsError('恢复模型或完成结果已存在，拒绝重复导出。')
    bindings=json.loads((directory/'recovery_manifest.json').read_text(encoding='utf-8'))
    data=Path(bindings['data_dir']);manifest=load_manifest(data)
    if file_sha256(data/'manifest.json') != bindings['data_manifest_sha256']:
        raise ValueError('LoRA 训练数据清单已改变。')
    trace=[json.loads(line) for line in (directory/'finetuning_trace.jsonl').read_text(encoding='utf-8').splitlines()]
    steps=bindings['config']['steps']
    if [row['step'] for row in trace] != list(range(1,steps+1)) or any(row['base_gradients_present'] for row in trace):
        raise ValueError('LoRA 训练不完整或存在基座梯度，不能继续导出。')
    result_file=directory/'training_result.json'
    if result_file.exists():
        result=json.loads(result_file.read_text(encoding='utf-8'))
        if (result.get('steps') != steps or not result.get('base_parameters_unchanged')
                or not result.get('adapter_parameters_changed')
                or result.get('base_parameter_sha256_before') != bindings['base_parameter_sha256']
                or result.get('base_parameter_sha256_after') != bindings['base_parameter_sha256']):
            raise ValueError('独立训练结果与基座绑定或完整步数不一致。')
    else:
        events=[json.loads(line) for line in (directory/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        start=next(row for row in events if row['stage']=='阶段二 2/4 LoRA')
        end=next(row for row in reversed(events) if row['stage']=='阶段二 3/4 保存')
        if not end['base_parameters_unchanged'] or not end['adapter_parameters_changed']:
            raise ValueError('缺少训练完成的基座冻结/adapter 更新证据。')
        result={'state':'training_completed','lora_recovery_performed':True,'full_benchmark_evaluated':False,
            'base_parameters_unchanged':True,'adapter_parameters_changed':True,
            'base_parameter_sha256_before':bindings['base_parameter_sha256'],
            'base_parameter_sha256_after':bindings['base_parameter_sha256'],
            'steps':steps,'dtype':'float32','internal_eval_before':start['internal_eval_before'],
            'internal_eval_after':end['internal_eval_after'],'adapter_package':str(directory/'adapter'),
            'training_metadata_source':'legacy_trace_and_events'}
    model=load_recovery(directory)
    # PEFT 包装插入 base_layer 层级；去掉包装名后与训练前原始参数顺序比较。
    base_parameters=[(name.replace('.base_layer.','.'),p) for name,p in model.get_base_model().named_parameters()
                     if not any(part in ('lora_A','lora_B') for part in name.split('.'))]
    if parameter_fingerprint(base_parameters) != bindings['base_parameter_sha256']:
        raise ValueError('重载基座与训练前逐参数摘要不一致。')
    tokenizer=AutoTokenizer.from_pretrained(directory/'adapter',local_files_only=True)
    architecture=json.loads((Path(bindings['base_package'])/'architecture.json').read_text(encoding='utf-8'))
    dataset=PreparedDataset(data,'eval',manifest)
    exported=export_recovered_model(model,tokenizer,architecture,directory,export_format,dataset,manifest['pad_token_id'])
    result.update(exported,state='completed',export_format=export_format,finalized_from_saved_adapter=True,
                  adapter_package=str(directory/'adapter'))
    write_json(directory/'recovery_result.json',result);write_json(directory/'status.json',result)
    log_stage(directory,'阶段二完成','已从保存的 adapter 完成合并、导出和独立重载；没有重训。',**result)
    return result


def run_recovery(base_dir, data_dir, output, config, device, export_format='safetensors', tokenizer=None):
    from functools import partial
    from prune_phi2_hesso_cric import evaluate_response_loss, log_stage, memory_stats, write_json
    validate_config(config)
    require_peft()
    base_dir, data_dir, output = Path(base_dir), Path(data_dir), Path(output)
    device = torch.device(device)
    if export_format not in ('torch','pt','safetensors','onnx'):
        raise ValueError('恢复模型格式必须为 pt/torch、safetensors 或 onnx。')
    architecture = json.loads((base_dir/'architecture.json').read_text(encoding='utf-8'))
    if architecture.get('weights_format') == 'onnx':
        raise ValueError('ONNX 基座不能训练；请使用同一压缩结构的 pt/safetensors 中间包。')
    if output.exists():
        raise FileExistsError('LoRA 输出目录已存在，拒绝覆盖。')
    manifest = load_manifest(data_dir)
    output.mkdir(parents=True)
    try:
        random.seed(config.seed);torch.manual_seed(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if device.type == 'cuda':
            torch.cuda.set_device(device);torch.cuda.manual_seed_all(config.seed)
            torch.cuda.reset_peak_memory_stats(device)
        collate = partial(collate_examples,pad_token_id=manifest['pad_token_id'])
        generator = torch.Generator().manual_seed(config.seed)
        train_loader = DataLoader(PreparedDataset(data_dir,'train',manifest),batch_size=config.batch_size,
                                  shuffle=True,generator=generator,collate_fn=collate,num_workers=0)
        eval_loader = DataLoader(PreparedDataset(data_dir,'eval',manifest),batch_size=config.batch_size,
                                 shuffle=False,collate_fn=collate,num_workers=0)
        log_stage(output,'阶段二 1/4 重载','释放旧优化器后，从磁盘重载物理压缩的可训练基座。',
                  base_package=str(base_dir),config=asdict(config),dtype='float32',autocast=False)
        base = OTO.load_subnet(base_dir,device=device).float()
        base.config.use_cache = False
        base_parameters = list(base.named_parameters())
        before_hash = parameter_fingerprint(base_parameters)
        baseline = evaluate_response_loss(base,eval_loader,device)
        model, trainable = attach_lora(base,config)
        before_adapter = parameter_fingerprint(trainable)
        layout = [{'name':name,'in_features':module.in_features,'out_features':module.out_features,
                   'rank':config.rank} for name,module in model.named_modules() if hasattr(module,'lora_A')]
        write_json(output/'lora_layout.json',layout)
        # 新优化器只持有 A/B；绝不复用 CRIC 的状态或剪枝预算。
        optimizer = torch.optim.AdamW([p for _,p in trainable],lr=config.learning_rate,weight_decay=config.weight_decay)
        def lr_factor(step):
            if config.warmup_steps and step < config.warmup_steps:
                return (step+1)/config.warmup_steps
            return max(0.,(config.steps-step)/max(1,config.steps-config.warmup_steps))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
        log_stage(output,'阶段二 2/4 LoRA','压缩基座冻结，仅 A/B 训练；按监督 token 加权累积梯度。',
                  base_parameters=sum(p.numel() for _,p in base_parameters),
                  trainable_parameters=sum(p.numel() for _,p in trainable),trainable_tensors=len(trainable),
                  internal_eval_before=baseline,packages={k:importlib.metadata.version(k) for k in ('peft','torch','transformers')})
        iterator = iter(train_loader)
        started = time.perf_counter()
        with (output/'finetuning_trace.jsonl').open('w',encoding='utf-8',buffering=1) as trace:
            for step in range(1,config.steps+1):
                cpu_batches = []
                for _ in range(config.gradient_accumulation_steps):
                    try:batch=next(iterator)
                    except StopIteration:
                        iterator=iter(train_loader);batch=next(iterator)
                    cpu_batches.append(batch)
                counts = [int((b['labels'][:,1:] != -100).sum()) for b in cpu_batches]
                token_count = sum(counts)
                if not token_count:raise ValueError('恢复批次没有监督 token。')
                model.train();optimizer.zero_grad(set_to_none=True)
                nll = 0.
                for cpu_batch,count in zip(cpu_batches,counts):
                    batch=to_device(cpu_batch,device)
                    loss=model(**batch,use_cache=False).loss
                    if not torch.isfinite(loss):raise FloatingPointError('LoRA loss 非有限。')
                    (loss*(count/token_count)).backward()
                    nll+=float(loss.detach())*count/token_count
                if any(p.grad is not None for _,p in base_parameters):
                    raise AssertionError('恢复阶段出现基座梯度。')
                norm=torch.nn.utils.clip_grad_norm_([p for _,p in trainable],config.grad_clip_norm,error_if_nonfinite=True)
                lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
                record={'step':step,'response_nll':nll,'supervised_tokens':token_count,
                        'learning_rate':lr,'gradient_norm_before_clip':float(norm),
                        'base_gradients_present':False,'elapsed_seconds':time.perf_counter()-started}
                if step%config.eval_every==0 or step==config.steps:
                    record['internal_eval']=evaluate_response_loss(model,eval_loader,device)
                trace.write(json.dumps(record)+'\n')
                if step==1 or step%config.log_every==0 or 'internal_eval' in record:
                    log_stage(output,'阶段二训练','仅 LoRA 更新进度；此处不是七任务准确率。',**record,**memory_stats(device))
                write_json(output/'status.json',{'state':'running',**record})
        after_hash=parameter_fingerprint(base_parameters)
        after_adapter=parameter_fingerprint(trainable)
        if before_hash != after_hash:raise AssertionError('LoRA 恢复改变了冻结基座。')
        if before_adapter == after_adapter:raise AssertionError('LoRA 参数未发生更新。')
        recovered_eval=evaluate_response_loss(model,eval_loader,device)
        bindings={'base_package':str(base_dir.resolve()),'base_files':package_hashes(base_dir),
                  'data_dir':str(data_dir.resolve()),'data_manifest_sha256':file_sha256(data_dir/'manifest.json'),
                  'config':asdict(config),'base_parameter_sha256':before_hash,'precision':'float32'}
        model.peft_config['default'].base_model_name_or_path=str(base_dir.resolve())
        model.save_pretrained(output/'adapter',safe_serialization=True)
        if tokenizer is None:tokenizer=AutoTokenizer.from_pretrained(base_dir,local_files_only=True)
        tokenizer.save_pretrained(output/'adapter')
        bindings['adapter_files']=package_hashes(output/'adapter')
        write_json(output/'recovery_manifest.json',bindings)
        log_stage(output,'阶段二 3/4 保存','A/B 已更新，基座逐参数摘要未变；保存 adapter 与基座绑定。',
                  base_parameters_unchanged=True,adapter_parameters_changed=True,internal_eval_after=recovered_eval)
        # 在 CPU 完成合并/导出等价校验，避免同时在 GPU 保留两份大基座。
        optimizer.zero_grad(set_to_none=True)
        del optimizer,scheduler,base_parameters,trainable,base
        model=model.cpu().eval();gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
        result={'state':'training_completed','lora_recovery_performed':True,'full_benchmark_evaluated':False,
                'base_parameters_unchanged':True,'adapter_parameters_changed':True,
                'base_parameter_sha256_before':before_hash,'base_parameter_sha256_after':after_hash,
                'steps':config.steps,'dtype':'float32','internal_eval_before':baseline,'internal_eval_after':recovered_eval,
                'export_format':export_format,'adapter_package':str(output/'adapter')}
        write_json(output/'training_result.json',result)
        result.update(export_recovered_model(model,tokenizer,architecture,output,export_format,
                      eval_loader.dataset,manifest['pad_token_id']),state='completed')
        write_json(output/'recovery_result.json',result);write_json(output/'status.json',result)
        log_stage(output,'阶段二完成','adapter、合并模型及独立重载校验完成。',**result)
        return result
    except Exception as error:
        write_json(output/'failure.json',{'state':'failed','error':repr(error)})
        write_json(output/'status.json',{'state':'failed','error':repr(error)})
        from prune_phi2_hesso_cric import run_logger
        run_logger(output).exception('LoRA 恢复失败，保留阶段一模型和诊断记录。')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir',type=Path)
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--finalize-only',type=Path,help='已完成训练的 LoRA 目录；只续做合并导出，不训练。')
    parser.add_argument('--config',type=Path,default=HERE/'phi2_two_stage_smoke_config.json')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--export-format',choices=('pt','safetensors','onnx'),default='safetensors')
    args=parser.parse_args()
    if args.finalize_only:
        if args.base_dir or args.data_dir or args.output:parser.error('--finalize-only 不与训练路径同时使用。')
        torch.set_num_threads(4)
        finalize_saved_recovery(args.finalize_only,args.export_format)
        return
    if not all((args.base_dir,args.data_dir,args.output)):parser.error('训练需要 --base-dir/--data-dir/--output。')
    source=json.loads(args.config.read_text(encoding='utf-8'))
    config=RecoveryConfig(**source.get('recovery',source))
    run_recovery(args.base_dir,args.data_dir,args.output,config,args.device,args.export_format)


if __name__=='__main__':
    main()
