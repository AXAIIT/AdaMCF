import os
import sys

# 【修改点】必须在导入 transformers 或 datasets 之前设置此变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import random
import re
import string
import collections
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from thop import profile

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from only_train_once import OTO
from only_train_once.optimizer.alpha_scheduler import ALPHA_CFG  # 新增导入配置


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
        if qid not in ref_map:
            continue
        gts = ref_map[qid]
        em_sum += metric_max_over_ground_truths(exact_match_score, pred, gts)
        f1_sum += metric_max_over_ground_truths(f1_score, pred, gts)
        total += 1

    return {
        "exact_match": 100.0 * em_sum / max(total, 1),
        "f1": 100.0 * f1_sum / max(total, 1),
    }

def load_local_squad_from_parquet(data_dir: str, max_train: int = 0, max_eval: int = 0):
    plain_text_dir = os.path.join(data_dir, "plain_text")
    train_path = os.path.join(plain_text_dir, "train-00000-of-00001.parquet")
    val_path = os.path.join(plain_text_dir, "validation-00000-of-00001.parquet")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"找不到训练集 parquet：{train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"找不到验证集 parquet：{val_path}")

    ds = load_dataset(
        "parquet",
        data_files={"train": train_path, "validation": val_path},
    )
    train_ds = ds["train"]
    eval_ds = ds["validation"]

    if max_train and max_train > 0:
        train_ds = train_ds.select(range(min(max_train, len(train_ds))))
    if max_eval and max_eval > 0:
        eval_ds = eval_ds.select(range(min(max_eval, len(eval_ds))))

    # 关键字段检查（便于更快定位本地 parquet schema 不一致问题）
    need_cols = {"id", "question", "context", "answers"}
    missing_train = need_cols - set(train_ds.column_names)
    missing_eval = need_cols - set(eval_ds.column_names)
    if missing_train:
        raise ValueError(f"训练集缺少列: {sorted(missing_train)}，现有列: {train_ds.column_names}")
    if missing_eval:
        raise ValueError(f"验证集缺少列: {sorted(missing_eval)}，现有列: {eval_ds.column_names}")

    return train_ds, eval_ds


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_bert_model(model_dir: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    return tokenizer, model


def prepare_train_features(examples, tokenizer, max_length=384, doc_stride=128):
    tokenized = tokenizer(
        examples["question"],
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

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id) if tokenizer.cls_token_id in input_ids else 0

        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # 找到 context 的 token 范围
        token_start_index = 0
        while token_start_index < len(sequence_ids) and sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while token_end_index >= 0 and sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # 答案不在当前窗口
        if token_start_index >= len(offsets) or token_end_index < 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        # 向右找 start
        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        start_positions.append(token_start_index - 1)

        # 向左找 end
        while offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        end_positions.append(token_end_index + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def prepare_validation_features(examples, tokenizer, max_length=384, doc_stride=128):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized["offset_mapping"]

    tokenized["example_id"] = []
    for i in range(len(tokenized["input_ids"])):
        sample_index = sample_mapping[i]
        tokenized["example_id"].append(examples["id"][sample_index])

        sequence_ids = tokenized.sequence_ids(i)
        # 只保留 context 的 offset，其它置 None，方便后处理
        tokenized["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset_mapping[i])
        ]
    return tokenized


def collate_train(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    # token_type_ids 对某些模型可能不存在，这里做兼容
    keys = ["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"]
    batch = {}
    for k in keys:
        if k in features[0]:
            dtype = torch.long
            batch[k] = torch.tensor([f[k] for f in features], dtype=dtype)
    return batch


def collate_eval(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["input_ids", "attention_mask", "token_type_ids"]
    batch: Dict[str, Any] = {}
    for k in keys:
        if k in features[0]:
            batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)
    batch["example_id"] = [f["example_id"] for f in features]
    batch["offset_mapping"] = [f["offset_mapping"] for f in features]
    return batch


def build_dataloaders(
    train_dataset,
    eval_dataset,
    tokenizer,
    batch_size: int,
    max_length: int = 384,
    doc_stride: int = 128,
    num_workers: int = 4,
):
    train_features = train_dataset.map(
        lambda x: prepare_train_features(x, tokenizer, max_length=max_length, doc_stride=doc_stride),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    eval_features = eval_dataset.map(
        lambda x: prepare_validation_features(x, tokenizer, max_length=max_length, doc_stride=doc_stride),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    trainloader = DataLoader(
        train_features,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_train,
        pin_memory=torch.cuda.is_available(),
    )
    evalloader = DataLoader(
        eval_features,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_eval,
        pin_memory=torch.cuda.is_available(),
    )

    return trainloader, evalloader, train_features, eval_features


def postprocess_qa_predictions(
    examples,
    features,
    raw_predictions: Tuple[np.ndarray, np.ndarray],
    n_best_size: int = 20,
    max_answer_length: int = 30,
):
    """
    简化版后处理：对每个 example 汇总其所有 feature 的 start/end logits，选最大得分 span。
    返回 dict[id] = prediction_text
    """
    all_start_logits, all_end_logits = raw_predictions

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example: Dict[int, List[int]] = {i: [] for i in range(len(examples["id"]))}
    for i, f in enumerate(features):
        ex_id = f["example_id"]
        features_per_example[example_id_to_index[ex_id]].append(i)

    predictions = {}

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        context = example["context"]

        best_score = -1e30
        best_text = ""

        for fi in feature_indices:
            start_logits = all_start_logits[fi]
            end_logits = all_end_logits[fi]
            offsets = features[fi]["offset_mapping"]

            start_indexes = np.argsort(start_logits)[-n_best_size:][::-1]
            end_indexes = np.argsort(end_logits)[-n_best_size:][::-1]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offsets) or end_index >= len(offsets):
                        continue
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    score = float(start_logits[start_index] + end_logits[end_index])
                    if score > best_score:
                        best_score = score
                        start_char = offsets[start_index][0]
                        end_char = offsets[end_index][1]
                        best_text = context[start_char:end_char]

        predictions[example["id"]] = best_text

    return predictions


@torch.no_grad()
def evaluate_squad(model, tokenizer, eval_dataset, eval_features, evalloader, device, max_batches: Optional[int] = None):
    model.eval()

    all_start_logits = []
    all_end_logits = []

    seen = 0
    for batch in evalloader:
        inputs = {
            "input_ids": batch["input_ids"].to(device, non_blocking=True),
            "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        }
        if "token_type_ids" in batch:
            inputs["token_type_ids"] = batch["token_type_ids"].to(device, non_blocking=True)

        outputs = model(**inputs)
        all_start_logits.append(outputs.start_logits.detach().cpu().numpy())
        all_end_logits.append(outputs.end_logits.detach().cpu().numpy())

        seen += 1
        if max_batches is not None and seen >= max_batches:
            break

    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    used_feature_count = all_start_logits.shape[0]
    used_eval_features = eval_features.select(range(used_feature_count))

    preds = postprocess_qa_predictions(
        examples=eval_dataset,
        features=used_eval_features,
        raw_predictions=(all_start_logits, all_end_logits),
    )

    processed_example_ids = set(used_eval_features["example_id"])
    
    formatted_preds = [
        {"id": k, "prediction_text": v} 
        for k, v in preds.items() 
        if k in processed_example_ids
    ]
    
    references = [
        {"id": ex["id"], "answers": ex["answers"]} 
        for ex in eval_dataset 
        if ex["id"] in processed_example_ids
    ]

    metrics = compute_em_f1(formatted_preds, references)
    return metrics['exact_match'], metrics['f1']


def prepare_alpha_eval_batches_bert(
    dataset,
    collate_fn,
    batch_size: int,
    device: torch.device,
    seed: int = 42,
):
    """
    完全对齐 alpha_scheduler.py 的策略：
    1. 使用 ALPHA_CFG.n_alpha_val (默认5000) 创建固定随机子集。
    2. 使用 ALPHA_CFG.eval_batch_limit (默认64) 缓存微批。
    3. 适配 BERT 的字典格式数据进行分割。
    """
    from torch.utils.data import Subset, DataLoader

    # 1. 确定采样数量 (Align with n_alpha_val)
    n_alpha = ALPHA_CFG.n_alpha_val
    total_len = len(dataset)
    take = min(n_alpha, total_len)

    # 2. 创建固定随机子集 (Align with build_alpha_loader)
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(total_len, generator=g)[:take].tolist()
    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,  # 子集已经是随机的了
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # 3. 缓存指定数量的微批 (Align with cache_val_batches)
    limit = ALPHA_CFG.eval_batch_limit
    val_batches = []
    it = iter(loader)
    
    for _ in range(limit):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        # 4. 拆分数据为 (inputs, targets) 用于 update
        vx = {
            "input_ids": batch["input_ids"].to(device, non_blocking=True),
            "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        }
        if "token_type_ids" in batch:
            vx["token_type_ids"] = batch["token_type_ids"].to(device, non_blocking=True)

        vy = {
            "start_positions": batch["start_positions"].to(device, non_blocking=True),
            "end_positions": batch["end_positions"].to(device, non_blocking=True),
        }
        val_batches.append((vx, vy))

    return val_batches, len(subset)


def qa_criterion(outputs, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    供 alpha_scheduler.update(...) 调用的 criterion(out, vy)。
    """
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    start_pos = targets["start_positions"]
    end_pos = targets["end_positions"]
    ce = nn.CrossEntropyLoss()
    return 0.5 * (ce(start_logits, start_pos) + ce(end_logits, end_pos))


def compute_flops_with_thop(model, input_size, device="cpu"):
    """
    使用 thop 计算BERT模型 FLOPs 和参数量。
    说明：用 wrapper 适配 HF 模型 forward（thop 更偏好位置参数）。
    """
    import copy

    class _ThopBertQAWrapper(nn.Module):
        def __init__(self, m: nn.Module):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            out = self.m(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            return out.start_logits, out.end_logits

    model_to_copy = model.module if isinstance(model, torch.nn.DataParallel) else model
    model_cpu = copy.deepcopy(model_to_copy).to(device).eval()
    wrapped = _ThopBertQAWrapper(model_cpu)

    # thop 会真的跑一次 forward，BERT embedding 需要 Long/Int
    dummy_input_ids = torch.randint(0, 30522, input_size, device=device, dtype=torch.long)
    dummy_attention_mask = torch.ones(input_size, device=device, dtype=torch.long)
    dummy_token_type_ids = torch.zeros(input_size, device=device, dtype=torch.long)

    with torch.no_grad():
        macs, params = profile(
            wrapped,
            inputs=(dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
            verbose=False,
        )

    flops_m = macs * 2 / 1e6
    params_m = params / 1e6

    del wrapped, model_cpu
    if device == "cuda":
        torch.cuda.empty_cache()

    return flops_m, params_m


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    scheduler_type: str = "linear",
):
    """
    BERT 微调/剪枝常用：warmup + (linear/cosine) decay。
    scheduler_type: "linear" | "cosine"
    """
    num_warmup_steps = int(max(0, num_warmup_steps))
    num_training_steps = int(max(1, num_training_steps))

    if scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    # default: linear
    return get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

def build_adamw_param_groups(model: torch.nn.Module, weight_decay: float):
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (nodecay if any(nd in n for nd in no_decay) else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]


def prune_model(seed: int = 42):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model_dir = "/workspace/OTOv2_v2/test/Bert/models/bert-base-uncased-squad-v1"
    out_dir = (Path(__file__).parent / "cache_bert_base_uncased_squad_pruned_sparsity_0_5")
    out_dir.mkdir(parents=True, exist_ok=True)
    squad_data_dir = "/workspace/data/squad"

    batch_size = 16
    num_workers = 4
    max_length = 384
    doc_stride = 128
    prune_lr = 3e-5
    group_sparsity = 0.5
    
    warmup_epochs = 5
    pruning_epochs = 150
    finetune_epochs = 50
    num_train_epochs = warmup_epochs + pruning_epochs + finetune_epochs
    pruning_periods = pruning_epochs

    # LR scheduler 类型（linear 最常用；可改为 "cosine"）
    lr_scheduler_type = "linear"

    print("\n--- 第 1 步：加载数据与预训练模型 ---")
    train_dataset, eval_dataset = load_local_squad_from_parquet(
        data_dir=squad_data_dir,
        max_train=0,
        max_eval=0,
    )
    tokenizer, model = load_bert_model(model_dir, device=device)
    model.train()
    trainloader, evalloader, train_features, eval_features = build_dataloaders(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        doc_stride=doc_stride,
        num_workers=num_workers,
    )

    print("\n--- 准备 Alpha 更新子集（QA 微批缓存） ---")
    # 使用与 alpha_scheduler 一致的构建策略
    val_batches, n_alpha_count = prepare_alpha_eval_batches_bert(
        dataset=train_features,  # 注意：使用处理后的 features
        collate_fn=collate_train,
        batch_size=batch_size,
        device=device,
        seed=42
    )
    print(f"[α-Subset] 样本池大小={n_alpha_count}, 缓存微批数={len(val_batches)} (Limit={len(val_batches)})")

    print("\n--- 第 2 步：初始化 OTO ---")
    dummy_input = (
        torch.zeros(1, max_length, dtype=torch.long, device=device),
        torch.ones(1, max_length, dtype=torch.long, device=device),
        torch.zeros(1, max_length, dtype=torch.long, device=device),
    )
    oto = OTO(model=model, dummy_input=dummy_input)
    oto.visualize(view=False, out_dir=out_dir)
    nodes_to_keep_unpruned = ['node-208']
    oto.mark_unprunable_by_node_ids(nodes_to_keep_unpruned)
    print(f"标记节点 {nodes_to_keep_unpruned} 为不可剪枝...")

    num_update_steps_per_epoch = len(trainloader)
    warmup_steps = warmup_epochs * num_update_steps_per_epoch
    pruning_steps = pruning_epochs * num_update_steps_per_epoch
    start_pruning_step = warmup_steps
    total_training_steps = num_train_epochs * num_update_steps_per_epoch

    optimizer_prune = oto.hesso(
        variant="adamw",
        lr=prune_lr,
        first_momentum=0.9,
        second_momentum=0.999,
        dampening=0.0,
        weight_decay=0.001,
        target_group_sparsity=group_sparsity,
        start_pruning_step=start_pruning_step,
        pruning_periods=pruning_periods,
        pruning_steps=pruning_steps,
        device=device,
    )

    # LR warmup + decay
    lr_scheduler_prune = build_lr_scheduler(
        optimizer=optimizer_prune,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
        scheduler_type=lr_scheduler_type,
    )
    print(f"[剪枝阶段] LR scheduler={lr_scheduler_type}, warmup_steps={warmup_steps}, total_steps={total_training_steps}")

    print("\n--- 第 3 步：剪枝前评估（EM/F1） ---")
    EM_before, F1_before = evaluate_squad(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        eval_features=eval_features,
        evalloader=evalloader,
        device=device,
        max_batches=None,
    )
    print(f"剪枝前EM: {EM_before:.2f}%, 剪枝前F1: {F1_before:.2f}%")
    flops_before_m, params_before_m = compute_flops_with_thop(
        model=model,
        input_size=(1, 384),
        device="cpu"
    )
    print(f"剪枝前 FLOPs: {flops_before_m:.2f} M")
    print(f"剪枝前参数量: {params_before_m:.2f} M")

    print("\n--- 第 4 步：使用 OTO(HESSO) 进行剪枝（Warmup + Pruning） ---")
    print("\n--- 阶段一：预热与剪枝 ---")
    model.train()
    optimizer_prune.zero_grad(set_to_none=True)

    best_F1 = 0.0
    best_EM = 0.0
    best_epoch = 0
    global_step = 0
    compressed_model_path = None  # 初始化变量防止后续 UnboundLocalError
    
    constructed = False  # 新增：防止 construct 后继续训练导致维度错配

    for epoch in range(num_train_epochs):
        print(f"\n[阶段一] Epoch {epoch+1}/{num_train_epochs} 正在剪枝...")
        epoch_loss = 0.0
        epoch_lr = 0.0
        epoch_steps = 0
        t0 = time.time()

        for batch in trainloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "start_positions": batch["start_positions"],
                "end_positions": batch["end_positions"],
            }
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = batch["token_type_ids"]

            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()

            if optimizer_prune.will_enter_new_period_on_next_step():
                optimizer_prune.alpha_scheduler.update(
                    model=model,
                    criterion=qa_criterion,
                    inputs=val_batches,
                    targets=None,
                    t_in_period=0,
                )

            optimizer_prune.step()
            lr_scheduler_prune.step()
            optimizer_prune.zero_grad(set_to_none=True)

            global_step += 1
            current_lr = optimizer_prune.param_groups[0]["lr"]
            epoch_loss += loss.item()
            epoch_lr += current_lr
            epoch_steps += 1

        dt = max(time.time() - t0, 1e-8)
        avg_loss = epoch_loss / epoch_steps
        avg_lr = epoch_lr / epoch_steps
        it_per_sec = epoch_steps / dt
        eta = (num_train_epochs - epoch - 1) * dt
        eta_h = int(eta // 3600)
        eta_m = int((eta % 3600) // 60)
        print(f"[阶段一] Epoch {epoch+1}/{num_train_epochs} 评估\n"
              f"平均Loss: {avg_loss:.4f}, 平均LR: {avg_lr:.6f}, "
              f"吞吐: {it_per_sec:.2f} it/s, 用时: {dt:.2f}s, "
              f"估计剩余: {eta_h}h:{eta_m}m")

        metrics = optimizer_prune.compute_metrics()
        print(f"Sparsity: {metrics.group_sparsity:.4f}")
        print(dict(zip(optimizer_prune._alpha_keys,
                       optimizer_prune.alpha_scheduler.get_alpha().tolist())))

        model.eval()
        EM_epoch, F1_epoch  = evaluate_squad(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            eval_features=eval_features,
            evalloader=evalloader,
            device=device,
            max_batches=None,
        )
        print(
            f"--- Epoch {epoch + 1}/{num_train_epochs} 测试:"
            f" EM {EM_epoch:.2f}%, F1 {F1_epoch:.2f}%",
            file=sys.stdout,
            flush=True
        )
        model.train()

        if epoch + 1 > warmup_epochs + pruning_epochs:
            EM, F1 = EM_epoch, F1_epoch 
            
            if F1 > best_F1 or (F1 == best_F1 and EM > best_EM):
                best_F1 = F1
                best_EM = EM
                best_epoch = epoch + 1
                print(f"当前最佳F1分数: {best_F1:.2f}% (Epoch {best_epoch})")
                print(f"  - 保存模型到: {out_dir}")

                print("\n--- 构建子网络 ---")
                oto.construct_subnet(out_dir=out_dir)
                compressed_model_path = oto.compressed_model_path
                print(f"压缩模型保存在: {compressed_model_path}")

                constructed = True
                break  # 新增：跳出 epoch 循环（construct 后不要再继续训练）

        if constructed:
            break

    if compressed_model_path is None:
        raise RuntimeError("未生成压缩模型。可能是因为训练轮数不足以进入微调阶段，或未找到更优模型。")

    compressed_model = torch.load(compressed_model_path, map_location=device, weights_only=False)
    compressed_model.to(device).eval()
    print("\n--- 剪枝阶段完成，模型结构已固定 ---")
    print("\n--- 第 5 步：剪枝后评估（EM/F1） ---")
    EM_after, F1_after = evaluate_squad(
            model=compressed_model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            eval_features=eval_features,
            evalloader=evalloader,
            device=device,
            max_batches=None,
        )
    print(f"剪枝后EM: {EM_after:.2f}%, 剪枝后F1: {F1_after:.2f}%")

    flops_after_m, params_after_m = compute_flops_with_thop(
    model=compressed_model, 
    input_size=(1, 384),
    device="cpu"
    )

    eps = 1e-8
    pruned_flops_m = max(flops_before_m - flops_after_m, 0.0)
    pruned_params_m = max(params_before_m - params_after_m, 0.0)
    pruned_flops_percent = pruned_flops_m / (flops_before_m + eps) * 100
    pruned_params_percent = pruned_params_m / (params_before_m + eps) * 100
    flops_compression_ratio = (flops_before_m / (flops_after_m + eps)) if flops_after_m > 0 else float("inf")
    params_compression_ratio = (params_before_m / (params_after_m + eps)) if params_after_m > 0 else float("inf")

    print("\n--- 第 6 步：压缩总结 ---")
    print(f"减少的 FLOPs: {pruned_flops_m:.2f}M ({pruned_flops_percent:.2f}%)")
    print(f"减少的参数量: {pruned_params_m:.2f}M ({pruned_params_percent:.2f}%)")

    if flops_before_m > 0 and flops_after_m > 0:
        print(f"FLOPs 压缩比: {flops_compression_ratio:.2f}x "
              f"({flops_before_m:.2f}M -> {flops_after_m:.2f}M)")
    else:
        print(f"FLOPs 压缩比: N/A (剪枝前: {flops_before_m:.2f}M, 剪枝后: {flops_after_m:.2f}M)")

    if params_before_m > 0 and params_after_m > 0:
        print(f"参数量压缩比: {params_compression_ratio:.2f}x "
              f"({params_before_m:.2f}M -> {params_after_m:.2f}M)")
    else:
        print(f"参数量压缩比: N/A (剪枝前: {params_before_m:.2f}M, 剪枝后: {params_after_m:.2f}M)")

    print(f"\nEM变化: 从 {EM_before:.2f}% 变为 {EM_after:.2f}% (变化: {EM_after - EM_before:+.2f}%)")
    print(f"F1变化: 从 {F1_before:.2f}% 变为 {F1_after:.2f}% (变化: {F1_after - F1_before:+.2f}%)")


def main():
    prune_model()


if __name__ == "__main__":
    main()
