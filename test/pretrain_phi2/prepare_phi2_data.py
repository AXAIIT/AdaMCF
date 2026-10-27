"""独立准备 Phi-2 的 Alpaca 因果语言模型数据，不加载模型或占用 GPU。"""
import argparse
import hashlib
import json
from pathlib import Path
import random

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path('/adminGroup/datasets_source203/llm_pruning')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def encode_example(row, row_id, tokenizer, max_length):
    """保留响应监督；prompt、padding 均不参与损失，不手工移动 labels。"""
    prompt = f"### Instruction:\n{row['instruction'].strip()}\n\n"
    if row['input'].strip():
        prompt += f"### Input:\n{row['input'].strip()}\n\n"
    prompt += '### Response:\n'
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(row['output'].strip(), add_special_tokens=False)
    answer_ids.append(tokenizer.eos_token_id)
    # 分开编码使监督边界明确。长 prompt 保留尾部与 Response 标记，
    # 至少为响应保留一半预算；截断的响应以 EOS 结束。
    prompt_ids = prompt_ids[-min(len(prompt_ids), max_length // 2):]
    answer_ids = answer_ids[:max_length - len(prompt_ids)]
    answer_ids[-1] = tokenizer.eos_token_id
    ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    if len(ids) < 2 or not any(x != -100 for x in labels[1:]):
        raise ValueError(f'样本 {row_id} 没有可监督的下一个 token。')
    return {'row_id': row_id, 'input_ids': ids, 'labels': labels}


def prepare(args):
    if args.max_length < 8 or min(args.train_samples, args.alpha_samples, args.eval_samples) < 1:
        raise ValueError('长度至少为 8，三个划分的样本数必须为正。')
    if args.output.exists():
        raise FileExistsError(f'输出目录已存在，请复用它或指定新目录：{args.output}')
    raw = json.loads(args.source.read_text(encoding='utf-8'))
    total = args.train_samples + args.alpha_samples + args.eval_samples
    if total > len(raw):
        raise ValueError('请求的样本总数超过原数据集。')
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError('分词器必须提供 EOS。')
    order = list(range(len(raw)))
    random.Random(args.seed).shuffle(order)
    selected, offset = {}, 0
    for split, count in [('train', args.train_samples), ('alpha', args.alpha_samples), ('eval', args.eval_samples)]:
        selected[split] = order[offset:offset + count]
        offset += count
    assert len(set(sum(selected.values(), []))) == total
    args.output.mkdir(parents=True)
    manifest = {
        'format_version': 1, 'source': str(args.source),
        'source_sha256': sha256_file(args.source), 'raw_rows': len(raw),
        'model_dir': str(args.model_dir), 'seed': args.seed,
        'max_length': args.max_length, 'loss_on': 'response_and_eos',
        'prompt_truncation': 'left_keep_tail_up_to_half_length',
        'pad_token_id': tokenizer.eos_token_id,
        'tokenizer_files': {p.name: sha256_file(p) for p in sorted(args.model_dir.glob('*'))
                            if p.is_file() and p.suffix in ('.json', '.txt')},
        'preprocessor_sha256': sha256_file(__file__), 'splits': {},
    }
    for split, indexes in selected.items():
        path = args.output / f'{split}.jsonl'
        supervised = 0
        with path.open('w', encoding='utf-8', newline='\n') as stream:
            for row_id in indexes:
                example = encode_example(raw[row_id], row_id, tokenizer, args.max_length)
                supervised += sum(x != -100 for x in example['labels'][1:])
                stream.write(json.dumps(example, ensure_ascii=False) + '\n')
        manifest['splits'][split] = {
            'file': path.name, 'rows': len(indexes), 'row_ids': indexes,
            'supervised_tokens': supervised, 'sha256': sha256_file(path),
        }
    (args.output / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'counts': {k: len(v) for k, v in selected.items()}}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DATA_ROOT / 'alpaca/alpaca_data.json')
    parser.add_argument('--model-dir', type=Path, default=HERE / 'models/phi-2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--train-samples', type=int, default=4096)
    parser.add_argument('--alpha-samples', type=int, default=128)
    parser.add_argument('--eval-samples', type=int, default=128)
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
