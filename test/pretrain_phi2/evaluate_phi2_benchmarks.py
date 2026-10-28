"""固定 lm-eval 七任务评测；保存逐例结果，文献表格只作参考，不伪装成本地测量。"""
import argparse
from functools import partial
import hashlib
import json
import logging
import math
from pathlib import Path
import sys
import time
import torch
from transformers import AutoTokenizer,PhiForCausalLM

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from only_train_once import OTO
from phi2_benchmark_tasks import load_protocol,build_task,HARNESS_REVISION
from phi2_data import file_sha256


def json_default(value):
    if isinstance(value,Path):return str(value)
    if isinstance(value,(torch.dtype,torch.device)):return str(value)
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if hasattr(value,'tolist'):return value.tolist()
    if isinstance(value,partial):return {'function':value.func.__module__+'.'+value.func.__name__,'kwargs':value.keywords}
    if callable(value):return value.__module__+'.'+value.__qualname__
    raise TypeError(f'不能序列化 {type(value)}')


def write_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=json_default),encoding='utf-8');temp.replace(path)


def wilson(value,count):
    z=1.959963984540054;den=1+z*z/count
    center=(value+z*z/(2*count))/den
    radius=z*math.sqrt(value*(1-value)/count+z*z/(4*count*count))/den
    return [max(0.,center-radius),min(1.,center+radius)]


def evaluate(args):
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    tasks=load_protocol(args.protocol)
    names=args.tasks or list(tasks)
    if not set(names)<=set(tasks):raise ValueError('仅支持协议中的七任务。')
    if args.output.exists() and not args.resume:raise FileExistsError('评测输出已存在，请指定新目录或 --resume。')
    args.created_output=not args.output.exists()
    args.output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,handlers=[logging.StreamHandler(),logging.FileHandler(args.output/'run.log',encoding='utf-8')],force=True)
    model_files={str(p.relative_to(args.model_dir)):file_sha256(p) for p in sorted(args.model_dir.rglob('*'))
                 if p.is_file() and p.suffix in ('.json','.safetensors','.pt')}
    inputs={'model_dir':str(args.model_dir),'model_kind':args.model_kind,'model_files':model_files,
            'harness_revision':HARNESS_REVISION,'tasks':names,'num_fewshot':0,'apply_chat_template':False,
            'dtype':'float32','batch_size':args.batch_size,'max_length':2048,'limit':args.limit,'seed':42,
            'protocol_sha256':file_sha256(args.protocol),'code_sha256':file_sha256(__file__)}
    if (args.output/'inputs.json').exists():
        if json.loads((args.output/'inputs.json').read_text())!=inputs:raise ValueError('恢复评测的模型/配置/代码不一致。')
    else:write_json(args.output/'inputs.json',inputs)
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    if args.model_kind=='dense':
        model=PhiForCausalLM.from_pretrained(args.model_dir,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').to(device)
    else:model=OTO.load_subnet(args.model_dir,device=device).float()
    model.eval();model.config.use_cache=False
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
    lm=HFLM(pretrained=model,tokenizer=tokenizer,backend='causal',device=str(device),
            batch_size=args.batch_size,max_length=2048,add_bos_token=False,dtype='float32')
    manager=TaskManager(include_defaults=False)
    completed={}
    for name in names:
        result_file=args.output/(name+'.json')
        if args.resume and result_file.exists():
            existing=json.loads(result_file.read_text());completed[name]=existing;continue
        started=time.perf_counter();task,path,hashes=build_task(name,tasks[name])
        expected=min(args.limit,tasks[name]['expected_rows']) if args.limit else tasks[name]['expected_rows']
        logging.info('任务 %s：split=%s rows=%s，零样本，不使用 chat template',name,tasks[name]['official_split'],expected)
        write_json(args.output/'status.json',{'state':'running','current_task':name,'completed_tasks':list(completed)})
        result=simple_evaluate(model=lm,tasks=[task],task_manager=manager,num_fewshot=0,limit=args.limit,
                               bootstrap_iters=0,log_samples=True,apply_chat_template=False,
                               random_seed=42,numpy_random_seed=42,torch_random_seed=42,fewshot_random_seed=42)
        samples=result.pop('samples')[name]
        if len(samples)!=expected:raise AssertionError('实际评测样本数不正确。')
        sample_file=args.output/(name+'_samples.jsonl')
        with sample_file.open('w',encoding='utf-8') as stream:
            for sample in samples:stream.write(json.dumps(sample,ensure_ascii=False,default=json_default)+'\n')
        metrics=result['results'][name]
        # raw 与长度归一化准确率均保存，不擅自断定作者混合指标口径。
        accuracies={key.split(',')[0]:float(value) for key,value in metrics.items() if key in ('acc,none','acc_norm,none')}
        report={'task':name,'official_split':tasks[name]['official_split'],'rows':expected,
                'full_task_evaluated':args.limit is None,'metrics':accuracies,
                'wilson_95pct':{key:wilson(value,expected) for key,value in accuracies.items()},
                'data_file':str(path),'data_sha256':file_sha256(path),'task_definition_sha256':hashes,
                'elapsed_seconds':time.perf_counter()-started,'samples_file':sample_file.name,
                'samples_sha256':file_sha256(sample_file),'harness_result':result}
        write_json(result_file,report);completed[name]=report
        logging.info('任务 %s 完成：%s',name,accuracies)
    full=args.limit is None and set(names)==set(tasks)
    raw_mean=sum(r['metrics']['acc'] for r in completed.values())/len(completed)
    comparison_metric={name:('acc_norm' if 'acc_norm' in completed[name]['metrics'] else 'acc') for name in names}
    norm_style_mean=sum(completed[name]['metrics'][comparison_metric[name]] for name in names)/len(names)
    paper=json.loads((HERE/'paper_phi2_reference.json').read_text(encoding='utf-8'))
    summary={'state':'completed','full_seven_task_evaluation':full,'functional_only':args.limit is not None,
             'model_dir':str(args.model_dir),'tasks':{n:r['metrics'] for n,r in completed.items()},
             'mean_acc':raw_mean,'normalized_style_mean':norm_style_mean,'normalized_style_metric_map':comparison_metric,
             'paper_reference':paper,'paper_numbers_are_not_local_measurements':True,
             'paper_metric_convention_verified':False}
    write_json(args.output/'summary.json',summary);write_json(args.output/'status.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--model-kind',choices=('dense','compressed'),default='compressed')
    p.add_argument('--protocol',type=Path,default=HERE/'evaluation_protocol.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--limit',type=int,default=None)
    p.add_argument('--tasks',nargs='+');p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.batch_size<1 or args.limit is not None and args.limit<1:p.error('batch-size/limit 必须为正。')
    try:evaluate(args)
    except Exception as error:
        if getattr(args,'created_output',False) or args.resume:
            write_json(args.output/'failure.json',{'state':'failed','error':repr(error)})
            logging.exception('评测失败，已完成任务的独立结果保留。')
        raise


if __name__=='__main__':main()
