import os
import re
import string
import argparse
import collections
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


# ----------------------------
# SQuAD v1.1 metric (offline)
# ----------------------------
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]) -> float:
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def compute_em_f1(predictions: List[Dict], references: List[Dict]) -> Dict[str, float]:
    # predictions: [{"id":..., "prediction_text":...}]
    # references:  [{"id":..., "answers":{"text":[...], "answer_start":[...]}}]
    ref_map = {r["id"]: r["answers"]["text"] for r in references}

    em_sum, f1_sum, total = 0.0, 0.0, 0
    for p in predictions:
        qid = p["id"]
        pred = p["prediction_text"]
        gts = ref_map[qid]
        em_sum += metric_max_over_ground_truths(exact_match_score, pred, gts)
        f1_sum += metric_max_over_ground_truths(f1_score, pred, gts)
        total += 1

    return {
        "exact_match": 100.0 * em_sum / max(total, 1),
        "f1": 100.0 * f1_sum / max(total, 1),
    }


# ----------------------------
# QA post-processing (sliding window)
# ----------------------------
def prepare_features(examples, tokenizer, max_length=384, doc_stride=128):
    questions = [q.lstrip() for q in examples["question"]]
    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    example_ids = []
    new_offset_mapping = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        example_id = examples["id"][sample_idx]
        example_ids.append(example_id)

        # 只保留 context 部分的 offset；非 context token 的 offset 置 None
        sequence_ids = tokenized.sequence_ids(i)
        cleaned_offsets = []
        for k, o in enumerate(offsets):
            if sequence_ids[k] == 1:
                cleaned_offsets.append(o)
            else:
                cleaned_offsets.append(None)
        new_offset_mapping.append(cleaned_offsets)

    tokenized["example_id"] = example_ids
    tokenized["offset_mapping"] = new_offset_mapping
    return tokenized


def postprocess_predictions(
    examples,
    features,
    start_logits_list,
    end_logits_list,
    n_best_size=20,
    max_answer_length=30,
):
    # examples: original dataset (id, context, answers...)
    # features: tokenized features with example_id & offset_mapping
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, eid in enumerate(features["example_id"]):
        features_per_example[eid].append(i)

    predictions = {}
    for eid, feat_indices in features_per_example.items():
        context = examples["context"][example_id_to_index[eid]]
        best_answer = {"text": "", "score": -1e9}

        for fi in feat_indices:
            start_logits = start_logits_list[fi]
            end_logits = end_logits_list[fi]
            offsets = features["offset_mapping"][fi]

            start_indexes = torch.topk(torch.tensor(start_logits), k=n_best_size).indices.tolist()
            end_indexes = torch.topk(torch.tensor(end_logits), k=n_best_size).indices.tolist()

            for s_idx in start_indexes:
                for e_idx in end_indexes:
                    if s_idx >= len(offsets) or e_idx >= len(offsets):
                        continue
                    if offsets[s_idx] is None or offsets[e_idx] is None:
                        continue
                    if e_idx < s_idx:
                        continue
                    length = e_idx - s_idx + 1
                    if length > max_answer_length:
                        continue

                    start_char, _ = offsets[s_idx]
                    _, end_char = offsets[e_idx]
                    text = context[start_char:end_char]
                    score = float(start_logits[s_idx] + end_logits[e_idx])

                    if score > best_answer["score"]:
                        best_answer = {"text": text, "score": score}

        predictions[eid] = best_answer["text"]

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/workspace/OTOv2_v2/test/Bert/models/bert-base-uncased-squad-v1")
    parser.add_argument("--data_dir", type=str, default="/workspace/data/squad")
    parser.add_argument("--max_eval", type=int, default=0)  # 先跑通建议 200；要全量可设很大
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)
    args = parser.parse_args()

    val_path = os.path.join(args.data_dir, "plain_text", "validation-00000-of-00001.parquet")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"找不到验证集 parquet：{val_path}")

    # 读取本地 parquet（离线）
    ds = load_dataset("parquet", data_files={"validation": val_path})["validation"]
    if args.max_eval > 0:
        ds = ds.select(range(min(args.max_eval, len(ds))))

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_dir, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # Tokenize with sliding window
    tokenized = ds.map(
        lambda x: prepare_features(x, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=ds.column_names,
    )

    # DataLoader
    cols = ["input_ids", "attention_mask", "token_type_ids"]
    torch_dataset = tokenized.with_format("torch", columns=cols)

    loader = DataLoader(torch_dataset, batch_size=args.batch_size)

    all_start, all_end = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            all_start.extend(out.start_logits.detach().cpu().tolist())
            all_end.extend(out.end_logits.detach().cpu().tolist())

    pred_map = postprocess_predictions(ds, tokenized, all_start, all_end)

    predictions = [{"id": ex_id, "prediction_text": pred_map[ex_id]} for ex_id in ds["id"]]
    references = [{"id": ex_id, "answers": ans} for ex_id, ans in zip(ds["id"], ds["answers"])]

    # 优先用 evaluate（若环境已有缓存/可用），否则用内置实现
    try:
        import evaluate
        metric = evaluate.load("squad")
        res = metric.compute(predictions=predictions, references=references)
        print("EM/F1:", res)
    except Exception:
        res = compute_em_f1(predictions, references)
        print("EM/F1 (offline):", res)


if __name__ == "__main__":
    main()
