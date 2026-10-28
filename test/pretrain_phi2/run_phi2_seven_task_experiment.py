"""单次实验控制器：空闲卡检查→资源探测→本框架两阶段→七任务评测。

不创建或覆盖数据。数据预处理独立完成，所有阶段使用固定配置与代码摘要。
文献表格是外部参考，始终与本地测量分开；本程序不运行原版 CRIC 或 Alpaca。
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from phi2_data import file_sha256,load_manifest


def write_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def gpu_snapshot(index):
    output=subprocess.check_output(['nvidia-smi','--id',str(index),
       '--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    physical,uuid,total,used,util=[item.strip() for item in output.strip().split(',')]
    return {'physical_index':int(physical),'uuid':uuid,'total_mib':int(total),'used_mib':int(used),'util_percent':int(util)}


def run(args):
    if not re.fullmatch('[0-9a-f]{40}',args.source_commit):raise ValueError('必须记录已提交的源码 SHA。')
    if args.output.exists():raise FileExistsError('实验输出已存在，拒绝覆盖。')
    manifest=load_manifest(args.data_dir)
    if manifest.get('alpaca_used') is not False:raise ValueError('本实验要求七任务 train 数据，不接受 Alpaca。')
    expected={'boolq','piqa','hellaswag','winogrande','arc_easy','arc_challenge','openbookqa'}
    if set(manifest['source_files'])!=expected:raise ValueError('数据必须来自规定七项官方 train。')
    config=json.loads(args.config.read_text())
    if not config.get('adaptive_alpha') or not config.get('recovery',{}).get('enabled'):
        raise ValueError('本轮要求本框架自适应 α 剪枝与独立 LoRA 恢复。')
    args.output.mkdir(parents=True)
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
        with (args.output/(stage+'.stdout.log')).open('w') as stream:
            process=subprocess.Popen(command,env=child_env,stdout=stream,stderr=subprocess.STDOUT,cwd=HERE.parents[1])
            write_json(args.output/'current_child.json',{'stage':stage,'pid':process.pid,'gpu_uuid':gpu['uuid']})
            rc=process.wait()
        if rc:raise RuntimeError(f'{stage} 失败，exit={rc}；见独立日志。')

    try:
        base=[args.pruning_python,'-u',str(HERE/'prune_phi2_hesso_cric.py'),
              '--data-dir',str(args.data_dir),'--model-dir',str(args.model_dir),
              '--config',str(args.config),'--device','cuda:0']
        execute('profile',[*base,'--output',str(args.output/'profile'),'--profile-only'])
        execute('training',[*base,'--output',str(args.output/'training')])
        training=args.output/'training'
        for stage,model in [('evaluation_pruned',training/'compressed_model'),
                            ('evaluation_recovered',training/'lora_recovery/recovered_model')]:
            execute(stage,[args.evaluation_python,'-u',str(HERE/'evaluate_phi2_benchmarks.py'),
                    '--model-dir',str(model),'--output',str(args.output/stage),'--device','cuda:0'])
        pruning=json.loads((training/'pruning_result.json').read_text())
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
    except Exception as error:
        write_json(args.output/'controller_status.json',{'state':'failed','error':repr(error)})
        write_json(args.output/'controller_failure.json',{'state':'failed','error':repr(error)})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=HERE/'phi2_seven_task_config.json')
    p.add_argument('--model-dir',type=Path,default=HERE/'models/phi-2')
    p.add_argument('--physical-gpu',type=int,default=0);p.add_argument('--source-commit',required=True)
    p.add_argument('--pruning-python',default='/opt/conda/envs/gy_adamcf_rtx6000d/bin/python')
    p.add_argument('--evaluation-python',default='/zeng/GY/.runtime/phi2_eval_20261029/bin/python')
    run(p.parse_args())


if __name__=='__main__':main()
