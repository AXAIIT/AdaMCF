"""独立冻结七个官方 train 划分中的剪枝、α 和内部验证样本；不使用 Alpaca。"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from phi2_data import HARNESS_REVISION, build_task, file_sha256, load_protocol, training_pair

def encode_pair(context,continuation,tokenizer,max_length):
    encoded=tokenizer(context+continuation,add_special_tokens=False,return_offsets_mapping=True,
                      truncation=False,padding=False,return_attention_mask=False)
    ids=list(encoded['input_ids'])
    labels=[token if end>len(context) else -100 for token,(start,end) in zip(ids,encoded['offset_mapping'])]
    if not any(label!=-100 for label in labels):raise ValueError('正确续写没有监督 token。')
    ids.append(tokenizer.eos_token_id);labels.append(tokenizer.eos_token_id)
    if len(ids)>max_length:return None
    return ids,labels


def prepare(args):
    if args.output.exists():raise FileExistsError('数据输出目录已存在，拒绝覆盖。')
    quotas={'train':args.train_per_task,'alpha':args.alpha_per_task,'eval':args.eval_per_task}
    if min(quotas.values())<1 or args.max_length<8:raise ValueError('配额必须为正，长度至少为 8。')
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    if not tokenizer.is_fast or tokenizer.eos_token_id is None:raise ValueError('需要 fast tokenizer 和 EOS。')
    tasks=load_protocol(args.protocol)
    eval_fingerprints=set();evaluation_sources={}
    for name,info in tasks.items():
        task,path,hashes=build_task(name,info)
        for doc in task.eval_docs:
            eval_fingerprints.add(training_pair(task,doc)[3])
        evaluation_sources[name]={'path':str(path),'sha256':file_sha256(path),'rows':len(task.eval_docs),'task_definition_sha256':hashes}
        del task
    rows={split:[] for split in quotas};selected_fingerprints=set();sources={};rejected=Counter()
    for index,(name,info) in enumerate(tasks.items()):
        task,path,hashes=build_task(name,info,'training');docs=task.eval_docs
        order=list(range(len(docs)));random.Random(args.seed+index).shuffle(order)
        needed=sum(quotas.values());selected=[]
        for source_row in order:
            doc=docs[source_row]
            try:context,answer,gold,fingerprint=training_pair(task,doc)
            except (ValueError,TypeError,IndexError):rejected[name+':invalid_label']+=1;continue
            if fingerprint in eval_fingerprints:rejected[name+':official_eval_overlap']+=1;continue
            if fingerprint in selected_fingerprints:rejected[name+':duplicate']+=1;continue
            encoded=encode_pair(context,answer,tokenizer,args.max_length)
            if encoded is None:rejected[name+':overlength']+=1;continue
            ids,labels=encoded
            selected.append({'row_id':f'{name}:train:{source_row}','task':name,'source_split':'train',
                             'source_row_id':source_row,'correct_choice':gold,'input_ids':ids,'labels':labels,
                             'question_sha256':fingerprint})
            selected_fingerprints.add(fingerprint)
            if len(selected)==needed:break
        if len(selected)<needed:raise ValueError(f'{name} 可用训练样本不足 {needed}，仅 {len(selected)}。')
        offset=0
        for split,count in quotas.items():rows[split].extend(selected[offset:offset+count]);offset+=count
        sources[name]={'path':str(path),'sha256':file_sha256(path),'official_split':'train',
                       'source_rows':len(docs),'selected_rows':needed,'task_definition_sha256':hashes}
        print(json.dumps({'task':name,'source_rows':len(docs),'selected':needed}),flush=True)
        del task,docs
    args.output.mkdir(parents=True)
    manifest={'format_version':1,'purpose':'seven_official_train_splits_user_recipe_v1','alpaca_used':False,
              'model_dir':str(args.model_dir),'seed':args.seed,'max_length':args.max_length,
              'loss_on':'correct_continuation_and_eos','pad_token_id':tokenizer.eos_token_id,
              'prompt_truncation':'none_filter_overlength_training_only','label_mask':'joint_encoding_offsets',
              'source':'seven_official_train_splits','source_files':sources,'evaluation_sources':evaluation_sources,
              'source_sha256':hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
              'harness_revision':HARNESS_REVISION,'preprocessor_sha256':file_sha256(__file__),
              'tokenizer_files':{p.name:file_sha256(p) for p in sorted(args.model_dir.iterdir()) if p.is_file() and p.suffix in ('.json','.txt')},
              'official_eval_question_overlap_removed':True,'rejected_candidates':dict(rejected),'splits':{}}
    for split,items in rows.items():
        random.Random(args.seed+100+list(quotas).index(split)).shuffle(items)
        path=args.output/(split+'.jsonl')
        with path.open('w',encoding='utf-8',newline='\n') as stream:
            for row in items:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        manifest['splits'][split]={'file':path.name,'rows':len(items),'row_ids':[r['row_id'] for r in items],
                                   'supervised_tokens':sum(sum(x!=-100 for x in r['labels'][1:]) for r in items),
                                   'task_counts':dict(Counter(r['task'] for r in items)),
                                   'max_actual_length':max(len(r['input_ids']) for r in items),'sha256':file_sha256(path)}
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'prepare.log').write_text(json.dumps({'quotas_per_task':quotas,'rejected_candidates':dict(rejected),
        'sources':sources,'evaluation_sources':evaluation_sources},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(args.output),'counts':{k:len(v) for k,v in rows.items()},'alpaca_used':False}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,default=HERE/'models/phi-2')
    p.add_argument('--protocol',type=Path,default=HERE/'evaluation_protocol.json')
    p.add_argument('--max-length',type=int,default=2048);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--train-per-task',type=int,default=768)
    p.add_argument('--alpha-per-task',type=int,default=64);p.add_argument('--eval-per-task',type=int,default=64)
    prepare(p.parse_args())


if __name__=='__main__':main()
