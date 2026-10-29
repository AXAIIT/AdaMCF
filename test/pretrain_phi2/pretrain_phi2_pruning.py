"""Phi-2 两阶段总入口：按 BERT 的阅读顺序组织剪枝与微调。

阅读 main() 即可看到：配置 → 冻结数据 → CRIC 剪枝/物理压缩 →
重载压缩基座 → LoRA 微调 → 评估汇总。两阶段训练循环仍由原模块维护：
  prune_phi2_hesso_cric.py: train_cric() / run() / export_run()
  finetune_phi2_lora.py:    run_recovery()

默认使用已完成七任务首轮实验的配置，实际运行会开始完整训练。
原始数据预处理与七项 benchmark 评分各保留一个独立程序。
加 --benchmarks 可在同一总入口调度空闲 GPU、两阶段训练及七项全量评测。
"""

# ==================== 1. 导入与默认路径 ====================
import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import gc
import json
import logging
import os
import re
import subprocess
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from phi2_data import PreparedDataset, file_sha256, load_manifest
from prune_phi2_hesso_cric import (
    PruningConfig,
    run as run_cric_pruning,
    validate_pruning_config,
    write_json,
)
from finetune_phi2_lora import (
    RecoveryConfig,
    require_peft,
    run_recovery,
    validate_config as validate_recovery_config,
)

# 超参数只有一个来源：JSON 中顶层为剪枝参数，recovery 为 LoRA 参数。
# 常改项：target_group_sparsity / hybrid_training_steps / recovery.steps。
DEFAULT_CONFIG = HERE / 'phi2_seven_task_config.json'
DEFAULT_MODEL = HERE / 'models/phi-2'
DEFAULT_DATA = Path(
    '/adminGroup/datasets_source203/llm_pruning/processed/'
    'phi2_seven_task_train_s42_v1'
)
SHANGHAI = timezone(timedelta(hours=8))


# ==================== 2. 配置、日志与冻结数据 ====================
def load_configuration(args):
    """先验证两阶段配置；不静默启用被用户关闭的 LoRA，不修改原 JSON。"""
    pruning = PruningConfig(**json.loads(args.config.read_text(encoding='utf-8')))
    if args.export_format is not None:
        pruning.export_format = args.export_format
    recovery = RecoveryConfig(**pruning.recovery)
    if args.log_every is not None:
        pruning.log_every = recovery.log_every = args.log_every
    validate_pruning_config(pruning)
    validate_recovery_config(recovery)
    if not recovery.enabled:
        raise ValueError('两阶段入口要求 recovery.enabled=true；仅剪枝请使用原剪枝入口。')
    pruning.recovery = asdict(recovery)
    return pruning, recovery


def create_logger(output):
    """总日志只写主流程；两阶段自己的 run.log 和逐步 JSONL 仍各自保存。"""
    class ShanghaiFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            return datetime.fromtimestamp(record.created, SHANGHAI).isoformat(timespec='seconds')

    logger = logging.Logger('phi2.two_stage')
    formatter = ShanghaiFormatter('%(asctime)s | %(levelname)s | %(message)s')
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(output / 'run.log', encoding='utf-8')):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def prepare_phi2_dataset(data_dir, model_dir, logger):
    """读取独立预处理脚本冻结的数据；训练、α、内部验证角色保持不变。"""
    manifest = load_manifest(data_dir)
    for name, expected in manifest['tokenizer_files'].items():
        path = model_dir / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f'冻结数据的分词器/配置与当前模型不一致：{name}')
    for role, description in [('train', '剪枝/LoRA 梯度训练'),
                              ('alpha', '公共优化器 α 更新'),
                              ('eval', '独立内部 NLL 验证')]:
        dataset = PreparedDataset(data_dir, role, manifest)
        logger.info('  %-5s | %5d 条 | %s', role, len(dataset), description)
    logger.info('  序列上限：%d；prompt/padding 不计入监督损失。', manifest['max_length'])
    logger.info('  本入口读取已冻结 JSONL；预处理与最终 benchmark 评测独立执行。')
    return manifest


# ==================== 3. 阶段一：全梯度 CRIC 剪枝 ====================
def prune_model(args, config, output):
    """运行已验证的 CRIC 和 construct_subnet 导出；这里不附加 LoRA。"""
    phase_config = copy.deepcopy(asdict(config))
    phase_config['recovery']['enabled'] = False
    # ONNX 仅供部署推理；阶段二必须从同一压缩结构的可训练包重载。
    if config.export_format == 'onnx':
        phase_config['export_format'] = 'safetensors'
    phase_config_path = args.output / 'pruning_config.json'
    write_json(phase_config_path, phase_config)
    run_cric_pruning(SimpleNamespace(
        config=phase_config_path,
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output=output,
        device=args.device,
        profile_only=False,
        export_format=None,
    ))
    result = json.loads((output / 'pruning_result.json').read_text(encoding='utf-8'))
    if result['state'] != 'completed' or not result['functional_pruning_passed']:
        raise RuntimeError('阶段一未完成结构验证，不能开始微调。')
    return result


# ==================== 4. 阶段二：重载压缩基座并训练 LoRA ====================
def finetune_model(args, config, recovery, pruning, output):
    """从磁盘重新加载异构结构；只训练 A/B，使用独立 AdamW 和调度器。"""
    # 不把旧 model/optimizer 传给微调。run_recovery 内部通过 OTO.load_subnet
    # 重建每层实际形状、冻结基座、附加 LoRA，并检查基座逐参数摘要不变。
    return run_recovery(
        base_dir=Path(pruning['compressed_package']),
        data_dir=args.data_dir,
        output=output,
        config=recovery,
        device=args.device,
        export_format=config.export_format,
    )


# ==================== 5. 最终评估与压缩汇总 ====================
def summarize_experiment(args, pruning, recovery, elapsed_seconds, logger):
    """汇总两阶段实测；内部响应 NLL 与七任务准确率明确区分。"""
    result = {
        'state': 'completed',
        'two_stage': True,
        'entrypoint': Path(__file__).name,
        'pruning': pruning,
        'recovery': recovery,
        'full_benchmark_evaluated': False,
        'elapsed_seconds': elapsed_seconds,
    }
    metrics = [
        ('剪枝前', pruning['baseline_eval']['response_nll']),
        ('物理压缩前稀疏模型', pruning['sparse_eval']['response_nll']),
        ('压缩包重载后', recovery['internal_eval_before']['response_nll']),
        ('LoRA 微调后', recovery['internal_eval_after']['response_nll']),
    ]
    logger.info('参数量：%s → %s，实际减少 %.4f%%',
                f"{pruning['parameters_before']:,}", f"{pruning['parameters_after']:,}",
                pruning['parameter_reduction_ratio'] * 100)
    logger.info('结构删除：FFN 神经元 %d，注意力头 %d，头内维度单元 %d',
                pruning['pruned_ffn_neurons'], pruning['pruned_attention_heads'],
                pruning['pruned_head_dim_units'])
    for name, nll in metrics:
        logger.info('内部响应 NLL | %s：%.6f', name, nll)
    logger.info('LoRA 基座冻结检查：%s；合并模型重载检查：%s',
                recovery['base_parameters_unchanged'], recovery['merged_reload_verified'])
    logger.info('剪后基座：%s', pruning['compressed_package'])
    logger.info('LoRA adapter：%s', recovery['adapter_package'])
    logger.info('最终模型：%s', recovery['recovered_package'])
    logger.info('七任务准确率：本入口未执行；请运行 evaluate_phi2_benchmarks.py。')
    lines = [
        '# Phi-2 两阶段运行结果', '',
        f"参数：{pruning['parameters_before']:,} → {pruning['parameters_after']:,}。",
        f"目标逻辑组稀疏率：{pruning['target_logical_group_sparsity']:.2%}；"
        f"实际参数减少：{pruning['parameter_reduction_ratio']:.4%}。", '',
        '|阶段|内部响应 NLL|', '|---|---:|',
        *[f'|{name}|{nll:.6f}|' for name, nll in metrics], '',
        '上述为冻结 train 来源保留子集的诊断，不是七任务准确率。', '',
        f"- 剪后基座：`{pruning['compressed_package']}`",
        f"- LoRA adapter：`{recovery['adapter_package']}`",
        f"- 最终模型：`{recovery['recovered_package']}`", '',
        '总入口 run.log 记录阶段，两个阶段目录内的 run.log / JSONL 记录训练细节。',
    ]
    (args.output / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(args.output / 'workflow_result.json', result)
    return result


# ==================== 6. 主流程：按下面的步骤顺序阅读 ====================
def main(args=None):
    if args is None:
        args = parse_arguments()
    if getattr(args, 'benchmarks', False):
        return run_benchmark_experiment(args)
    # --- 第1步：读取配置、检查设备与依赖，建立本次独立运行目录 ---
    if args.output.exists():
        raise FileExistsError('输出目录已存在；请为新实验使用独立目录。')
    config, recovery_config = load_configuration(args)
    require_peft()  # 在长时间剪枝前确认 LoRA 依赖，脚本不安装或升级包。
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA 不可用；CPU 功能验证请显式指定 --device cpu。')
        torch.cuda.get_device_properties(device)
    if not (args.model_dir / 'config.json').is_file():
        raise FileNotFoundError('模型目录缺少 config.json。')
    args.output.mkdir(parents=True)
    logger = create_logger(args.output)
    started = time.perf_counter()
    current_stage = 'configuration'

    def step(number, title, stage):
        nonlocal current_stage
        current_stage = stage
        logger.info('\n--- 第%d步：%s ---', number, title)
        write_json(args.output / 'status.json', {'state': 'running', 'stage': stage})

    try:
        step(1, '配置与运行路径', 'configuration')
        write_json(args.output / 'config.json', asdict(config))
        logger.info('模型：%s', args.model_dir)
        logger.info('冻结数据：%s', args.data_dir)
        logger.info('设备：%s；精度：FP32；导出格式：%s', device, config.export_format)
        logger.info('剪枝：模式=%s，逻辑组目标=%.2f%%，lr=%g，混合稀疏步数=%d',
                    config.pruning_mode, config.target_group_sparsity * 100,
                    config.learning_rate, config.hybrid_training_steps)
        logger.info('LoRA：步数=%d，r=%d，alpha=%d，lr=%g，梯度累积=%d',
                    recovery_config.steps, recovery_config.rank, recovery_config.lora_alpha,
                    recovery_config.learning_rate, recovery_config.gradient_accumulation_steps)
        logger.info('总日志：%s', args.output / 'run.log')

        # --- 第2步：读取已预处理的数据，确认训练/α/验证互斥与分词器一致 ---
        step(2, '检查冻结数据与分词器', 'data')
        manifest = prepare_phi2_dataset(args.data_dir, args.model_dir, logger)
        write_json(args.output / 'inputs.json', {
            'config_source': str(args.config), 'config_source_sha256': file_sha256(args.config),
            'model_dir': str(args.model_dir), 'data_dir': str(args.data_dir),
            'data_manifest_sha256': file_sha256(args.data_dir / 'manifest.json'),
            'split_rows': {role: manifest['splits'][role]['rows'] for role in ('train', 'alpha', 'eval')},
            'device': str(device), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'code_sha256': {name: file_sha256(HERE / name) for name in (
                Path(__file__).name, 'prune_phi2_hesso_cric.py', 'finetune_phi2_lora.py')},
        })

        # --- 第3步：阶段一：剪前 NLL → 全梯度 CRIC → construct_subnet 物理压缩 ---
        step(3, '阶段一：预热、CRIC 剪枝与物理压缩', 'pruning')
        pruning = prune_model(args, config, args.output / 'stage1_pruning')

        # --- 第4步：确认结构检查通过，释放阶段一资源，明确微调输入包 ---
        step(4, '检查压缩结果，准备从磁盘重载', 'compression_check')
        logger.info('参数量：%s → %s；可训练压缩包：%s',
                    f"{pruning['parameters_before']:,}", f"{pruning['parameters_after']:,}",
                    pruning['compressed_package'])
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # --- 第5步：阶段二：OTO 重载 → 冻结基座 → 附加 A/B → 新 AdamW 微调 ---
        step(5, '阶段二：重载压缩模型并进行 LoRA 微调', 'finetuning')
        recovered = finetune_model(
            args, config, recovery_config, pruning, args.output / 'stage2_finetuning')

        # --- 第6步：汇总内部评估、参数压缩、冻结检查与所有保存路径 ---
        step(6, '最终评估与压缩汇总', 'summary')
        result = summarize_experiment(args, pruning, recovered, time.perf_counter() - started, logger)
        write_json(args.output / 'status.json', {'state': 'completed', 'stage': 'completed'})
        logger.info('两阶段完成，总耗时 %.1f 秒；结果：%s', result['elapsed_seconds'], args.output / 'summary.md')
        return result
    except Exception as error:
        failure = {'state': 'failed', 'stage': current_stage, 'error': repr(error),
                   'pruning_result_preserved': (args.output / 'stage1_pruning/pruning_result.json').is_file()}
        write_json(args.output / 'failure.json', failure)
        write_json(args.output / 'status.json', failure)
        logger.exception('两阶段流程失败，保留已完成阶段、配置与日志。')
        raise
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()


# ==================== 可选的空闲卡调度与七任务全量评测 ====================
def gpu_snapshot(index):
    output=subprocess.check_output(['nvidia-smi','--id',str(index),
       '--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    physical,uuid,total,used,util=[item.strip() for item in output.strip().split(',')]
    return {'physical_index':int(physical),'uuid':uuid,'total_mib':int(total),'used_mib':int(used),'util_percent':int(util)}

def run_benchmark_experiment(args):
    """可选完整实验调度；子进程复用本文件普通模式，训练流程只有一份。"""
    if not args.source_commit or not re.fullmatch('[0-9a-f]{40}',args.source_commit):
        raise ValueError('--benchmarks 需要 --source-commit 记录已提交的40位源码 SHA。')
    if args.output.exists():raise FileExistsError('实验输出已存在，拒绝覆盖。')
    manifest=load_manifest(args.data_dir)
    if manifest.get('alpaca_used') is not False:raise ValueError('本实验要求七任务 train 数据，不接受 Alpaca。')
    expected={'boolq','piqa','hellaswag','winogrande','arc_easy','arc_challenge','openbookqa'}
    if set(manifest['source_files'])!=expected:raise ValueError('数据必须来自规定七项官方 train。')
    config=json.loads(args.config.read_text(encoding='utf-8'))
    if getattr(args,'export_format',None) is not None:
        config['export_format']=args.export_format
    if getattr(args,'log_every',None) is not None:
        config['log_every']=args.log_every
        config.setdefault('recovery',{})['log_every']=args.log_every
    if config.get('export_format','safetensors') == 'onnx':
        raise ValueError('七任务 GPU 评分需要 pt/safetensors；ONNX 可用普通两阶段模式训练并导出。')
    if not config.get('adaptive_alpha') or not config.get('recovery',{}).get('enabled'):
        raise ValueError('本轮要求本框架自适应 α 剪枝与独立 LoRA 恢复。')
    args.output.mkdir(parents=True)
    effective_config=args.output/'benchmark_config.json'
    write_json(effective_config,config)
    inputs={'source_commit':args.source_commit,'data_dir':str(args.data_dir),'config':config,
            'data_manifest_sha256':file_sha256(args.data_dir/'manifest.json'),'config_sha256':file_sha256(args.config),
            'code_sha256':{p.name:file_sha256(p) for p in sorted(HERE.glob('*.py'))},
            'physical_gpu':args.physical_gpu,'alpaca_used':False,'original_optimizer_used':False,
            'pruning_python':args.pruning_python,'evaluation_python':args.evaluation_python}
    write_json(args.output/'experiment_inputs.json',inputs)
    env=dict(os.environ,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONNOUSERSITE='1')

    def wait_gpu(stage):
        stable=0
        while stable<2:
            snapshot=gpu_snapshot(args.physical_gpu)
            stable=stable+1 if snapshot['used_mib']<=2048 and snapshot['util_percent']<=5 else 0
            write_json(args.output/'controller_status.json',{'state':'waiting_for_idle_gpu','next_stage':stage,'gpu':snapshot,'stable_checks':stable})
            if stable<2:time.sleep(15)
        return snapshot

    def execute(stage,command):
        gpu=wait_gpu(stage);child_env=dict(env,CUDA_VISIBLE_DEVICES=gpu['uuid'])
        write_json(args.output/'controller_status.json',{'state':'running','stage':stage,'gpu':gpu,'command':command})
        print(json.dumps({'stage':stage,'gpu':gpu,'command':command}),flush=True)
        with (args.output/(stage+'.stdout.log')).open('w',encoding='utf-8',buffering=1) as stream:
            process=subprocess.Popen(command,env=child_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                     cwd=HERE.parents[1],text=True,encoding='utf-8',errors='replace',bufsize=1)
            write_json(args.output/'current_child.json',{'stage':stage,'pid':process.pid,'gpu_uuid':gpu['uuid']})
            for line in process.stdout:
                stream.write(line)
                print(line,end='',flush=True)
            rc=process.wait()
        if rc:raise RuntimeError(f'{stage} 失败，exit={rc}；见独立日志。')

    try:
        base=[args.pruning_python,'-u',str(HERE/'prune_phi2_hesso_cric.py'),
              '--data-dir',str(args.data_dir),'--model-dir',str(args.model_dir),
              '--config',str(effective_config),'--device','cuda:0']
        execute('profile',[*base,'--output',str(args.output/'profile'),'--profile-only'])
        # 训练统一走人类可读的两阶段总入口；profile 继续使用底层剪枝资源探测。
        training_command=[args.pruning_python,'-u',str(HERE/'pretrain_phi2_pruning.py'),
                          '--data-dir',str(args.data_dir),'--model-dir',str(args.model_dir),
                          '--config',str(effective_config),'--device','cuda:0',
                          '--output',str(args.output/'training')]
        execute('training',training_command)
        training=args.output/'training'
        for stage,model in [('evaluation_pruned',training/'stage1_pruning/compressed_model'),
                            ('evaluation_recovered',training/'stage2_finetuning/recovered_model')]:
            execute(stage,[args.evaluation_python,'-u',str(HERE/'evaluate_phi2_benchmarks.py'),
                    '--model-dir',str(model),'--output',str(args.output/stage),'--device','cuda:0'])
        pruning=json.loads((training/'stage1_pruning/pruning_result.json').read_text())
        results={stage:json.loads((args.output/stage/'summary.json').read_text())
                 for stage in ('evaluation_pruned','evaluation_recovered')}
        if not all(item['full_seven_task_evaluation'] for item in results.values()):raise AssertionError('七任务评测不完整。')
        summary={'state':'completed','source_commit':args.source_commit,'alpaca_used':False,
                 'method':'adamcf_hesso_cric_with_post_pruning_lora','target_logical_group_sparsity':config['target_group_sparsity'],
                 'actual_parameter_reduction_ratio':pruning['parameter_reduction_ratio'],'evaluations':results,
                 'paper_baseline_is_reference_only':True,'author_exact_recipe_reproduced':False}
        write_json(args.output/'experiment_summary.json',summary)
        paper=results['evaluation_recovered']['paper_reference'];names=paper['task_order']
        lines=['# Phi-2 七任务实验结果','',f'源码：{args.source_commit}',
               f"实际参数减少比例：{pruning['parameter_reduction_ratio']:.4%}；目标逻辑组稀疏率：{config['target_group_sparsity']:.2%}。",
               '训练仅使用七任务官方 train；官方验证/测试仅评测。未使用 Alpaca，未运行原版 CRIC。','',
               '| 任务 | 论文未剪枝参考 | 本框架剪后 acc | LoRA 后 acc | LoRA 后 acc_norm（若定义） |',
               '| --- | ---: | ---: | ---: | ---: |']
        for i,name in enumerate(names):
            a=results['evaluation_pruned']['tasks'][name];b=results['evaluation_recovered']['tasks'][name]
            norm=f"{b['acc_norm']*100:.2f}" if 'acc_norm' in b else '-'
            lines.append(f"| {name} | {paper['dense_phi2_percent'][i]:.2f} | {a['acc']*100:.2f} | {b['acc']*100:.2f} | {norm} |")
        lines+=['','同时保存全 acc 平均与有 acc_norm 时取 acc_norm 的平均；作者精确指标混合口径未核实，不将两者混报。',
                '表格参考值不是本地未剪枝测量。本轮训练配方和数据取样不同，不能声称复现作者的精确实验。']
        (args.output/'experiment_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        write_json(args.output/'controller_status.json',{'state':'completed','summary':str(args.output/'experiment_summary.json')})
        return summary
    except Exception as error:
        write_json(args.output/'controller_status.json',{'state':'failed','error':repr(error)})
        write_json(args.output/'controller_failure.json',{'state':'failed','error':repr(error)})
        raise


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, help='新实验目录；默认使用带时间的 runs 子目录。')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--export-format', choices=('pt', 'safetensors', 'onnx'))
    parser.add_argument('--log-every', type=int, help='同时覆盖两阶段日志间隔；1 表示逐步输出。')
    parser.add_argument('--benchmarks', action='store_true', help='同一入口执行空闲卡调度、资源探测、两阶段训练及七项全量评分。')
    parser.add_argument('--physical-gpu', type=int, default=0, help='--benchmarks 使用的物理 GPU；映射为子进程 cuda:0。')
    parser.add_argument('--source-commit', help='--benchmarks 要求记录源码40位提交 SHA。')
    parser.add_argument('--pruning-python', default=sys.executable)
    parser.add_argument('--evaluation-python', default='/zeng/GY/.runtime/phi2_eval_20261029/bin/python')
    args = parser.parse_args()
    if args.output is None:
        stamp = datetime.now(SHANGHAI).strftime('%Y%m%d_%H%M%S_%f')
        args.output = HERE / 'runs' / (stamp + '_phi2_two_stage')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    return args


if __name__ == '__main__':
    main()
