"""七项任务的本地数据适配；模板/清洗/候选评分定义复用固定版 lm-eval。

这里只替换数据来源和 BoolQ 字段名，不自行重写 WinoGrande 后缀评分或
HellaSwag 清洗。训练只读官方 train，最终评测只读指定 validation/test。
"""
from functools import partial
import copy
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
HARNESS_REVISION='d6de81643928d653435c431bae19945d41d32520'
TASK_FILES={
 'boolq':'super_glue/boolq/default.yaml','piqa':'piqa/piqa.yaml',
 'hellaswag':'hellaswag/hellaswag.yaml','winogrande':'winogrande/default.yaml',
 'arc_easy':'arc/arc_easy.yaml','arc_challenge':'arc/arc_challenge.yaml',
 'openbookqa':'openbookqa/openbookqa.yaml'}


def load_protocol(path=None):
    protocol=json.loads(Path(path or HERE/'evaluation_protocol.json').read_text(encoding='utf-8'))
    tasks={item['task']:item for item in protocol['tasks']}
    if set(tasks)!=set(TASK_FILES):raise ValueError('协议必须包含规定七项任务。')
    if protocol['reference_harness_commit']!=HARNESS_REVISION:raise ValueError('评测器版本与协议不一致。')
    return tasks


def train_file(info):
    path=Path(info['local_file']);split=info['official_split']
    if path.parent.name==split:path=path.parent.parent/'train'/path.name
    else:path=path.with_name(path.name.replace(split+'-','train-',1))
    if not path.is_file():raise FileNotFoundError(f'未找到官方 train：{path}')
    return path


def local_dataset(*,path,split,task,**_):
    from datasets import load_dataset
    dataset=load_dataset('parquet',data_files={split:str(path)})
    if task=='boolq':
        # 本地源用 answer: bool；官方任务 YAML 用 label: int。
        dataset[split]=dataset[split].map(lambda row:{'label':int(row['answer'])})
    return dataset


def task_definition(name):
    import lm_eval
    from lm_eval.tasks._yaml_loader import load_yaml
    path=Path(lm_eval.__file__).parent/'tasks'/TASK_FILES[name]
    config=load_yaml(path)
    files=[path]
    if name=='arc_challenge':files.append(path.parent/'arc_easy.yaml')
    if name=='hellaswag':files.append(path.parent/'utils.py')
    if name=='winogrande':files.append(path.parent/'preprocess_winogrande.py')
    hashes={str(p.relative_to(Path(lm_eval.__file__).parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return config,hashes


def build_task(name,info,role='evaluation'):
    from lm_eval.api.task import ConfigurableTask
    config,hashes=task_definition(name);config=copy.deepcopy(config)
    split='train' if role=='training' else info['official_split']
    path=train_file(info) if role=='training' else Path(info['local_file'])
    config.update(dataset_path='parquet',dataset_name=None,dataset_kwargs={},training_split=None,
                  validation_split=split if split!='test' else None,test_split=split if split=='test' else None,
                  custom_dataset=partial(local_dataset,path=path,split=split,task=name),num_fewshot=0)
    task=ConfigurableTask(config=config)
    if role=='evaluation' and len(task.eval_docs)!=info['expected_rows']:
        raise ValueError(f'{name} 评测行数与协议不一致。')
    return task,path,hashes


def training_pair(task,doc):
    """将官方任务的正确续写变为 causal-LM 监督；不把选项字母当答案文本。"""
    text=task.doc_to_text(doc);target=task.doc_to_target(doc);choices=task.doc_to_choice(doc)
    delimiter=task.config.target_delimiter
    if isinstance(text,int):
        # WinoGrande：比较两种前缀下相同后缀的似然。
        gold=text;context=choices[gold];continuation=delimiter+str(target)
        fingerprint_text=json.dumps([choices,str(target)],ensure_ascii=False)
    else:
        gold=int(target)
        if not 0<=gold<len(choices):raise ValueError('训练标签越界。')
        context=str(text);continuation=delimiter+str(choices[gold])
        fingerprint_text=json.dumps([context,choices],ensure_ascii=False)
    fingerprint=hashlib.sha256(fingerprint_text.encode()).hexdigest()
    return context,continuation,gold,fingerprint
