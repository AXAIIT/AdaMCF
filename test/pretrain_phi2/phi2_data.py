"""共享数据工具：读取冻结 token、构造批次、复用官方七任务模板。

任务定义按需导入 lm-eval，普通剪枝/微调无需加载评测依赖。
α 抽样策略仍由公共优化器管理，本模块只提供数据与元信息。
"""
import copy
from functools import partial
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from only_train_once.optimizer.alpha_data import AlphaBatch


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(directory):
    """核对索引互斥和文件摘要；不把内部 eval 误当作最终任务测试集。"""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format_version') != 1:
        raise ValueError('不支持的数据版本。')
    seen = set()
    for split in ('train', 'alpha', 'eval'):
        info = manifest['splits'][split]
        if seen.intersection(info['row_ids']):
            raise ValueError('train/alpha/eval 出现原始样本重叠。')
        seen.update(info['row_ids'])
        if file_sha256(directory / info['file']) != info['sha256']:
            raise ValueError(f'{split} 文件摘要与预处理清单不一致。')
    return manifest


class PreparedDataset(Dataset):
    def __init__(self, directory, split, manifest):
        info = manifest['splits'][split]
        path = Path(directory) / info['file']
        self.rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if [row['row_id'] for row in self.rows] != info['row_ids']:
            raise ValueError(f'{split} 的样本顺序与清单不一致。')
        if len(self.rows) != info['rows']:
            raise ValueError(f'{split} 行数错误。')
        for row in self.rows:
            ids, labels = row['input_ids'], row['labels']
            if len(ids) != len(labels) or not 2 <= len(ids) <= manifest['max_length']:
                raise ValueError('token 长度或 labels 长度非法。')
            if not any(label != -100 for label in labels[1:]):
                raise ValueError('样本没有可计算的因果损失。')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate_examples(rows, pad_token_id):
    """EOS 可兼作 padding ID，但 padding 的 attention=0、label=-100。

    mask 由真实长度构造，不能简单按 token_id==EOS 屏蔽，否则会误删
    真实响应末尾 EOS 的训练信号。input_ids 与 labels 保持未位移对齐。
    """
    length = max(len(row['input_ids']) for row in rows)
    ids = torch.full((len(rows), length), pad_token_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        n = len(row['input_ids'])
        ids[i, :n] = torch.tensor(row['input_ids'])
        labels[i, :n] = torch.tensor(row['labels'])
        mask[i, :n] = 1
    return {'input_ids': ids, 'attention_mask': mask, 'labels': labels}


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def collate_alpha_examples(rows, pad_token_id):
    """只适配因果 LM 批次与元数据；取批/复用/覆盖由公共 α 调度器负责。"""
    batch=collate_examples(rows,pad_token_id)
    return AlphaBatch(batch,sample_ids=tuple(row['row_id'] for row in rows),
                      supervised_tokens=int((batch['labels'][:,1:] != -100).sum()))


# ==================== 七任务的本地数据适配与官方模板 ====================
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
