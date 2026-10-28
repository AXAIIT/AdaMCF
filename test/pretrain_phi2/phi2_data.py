"""读取已经冻结的 token 数据；不在训练脚本里执行抽样、切分或分词。"""
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


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
