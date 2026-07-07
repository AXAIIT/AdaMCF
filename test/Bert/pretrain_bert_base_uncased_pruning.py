import sys
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    BertForQuestionAnswering,
    BertTokenizerFast, 
    get_scheduler,
    squad_convert_examples_to_features,
    SquadV1Processor,
    squad_convert_examples_to_features
)
from datasets import load_dataset

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from only_train_once import OTO


def prepare_squad_dataset(tokenizer, batch_size, max_length=384, stride=128):
    """
    准备SQuAD数据集
    """
    print("加载SQuAD数据集...")
    try:
        squad_data_dir = os.path.join(project_root, 'data', 'squad')
        train_file = os.path.join(squad_data_dir, 'plain_text', 'train-00000-of-00001.parquet')
        val_file = os.path.join(squad_data_dir, 'plain_text', 'validation-00000-of-00001.parquet')
        
        if os.path.exists(train_file) and os.path.exists(val_file):
            print(f"从本地文件加载SQuAD数据集: {train_file}")
            train_dataset = load_dataset('parquet', data_files={'train': train_file}, split='train')
            val_dataset = load_dataset('parquet', data_files={'validation': val_file}, split='validation')
        else:
            print("本地文件不存在，从Hugging Face加载SQuAD数据集")
            squad_dataset = load_dataset("squad")
            train_dataset = squad_dataset["train"]
            val_dataset = squad_dataset["validation"]
    except Exception as e:
        print(f"加载本地数据失败: {e}，尝试从Hugging Face下载")
        squad_dataset = load_dataset("squad")
        train_dataset = squad_dataset["train"]
        val_dataset = squad_dataset["validation"]

    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        contexts = [c.strip() for c in examples["context"]]
        
        inputs = tokenizer(
            questions,
            contexts,
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        offset_mapping = inputs.pop("offset_mapping")

        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            
            if len(answer["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue
                
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            
            token_start_index = 0
            while token_start_index < len(offset) and offset[token_start_index][0] <= start_char:
                token_start_index += 1
            token_start_index -= 1
            
            token_end_index = len(offset) - 1
            while token_end_index >= 0 and offset[token_end_index][1] >= end_char:
                token_end_index -= 1
            token_end_index += 1
            
            # 检查答案是否在当前切片内，并确保索引不越界
            if not (offset[token_start_index][0] <= start_char and 
                    token_end_index < len(offset) and  # 添加边界检查
                    offset[token_end_index][1] >= end_char):
                start_positions.append(0)
                end_positions.append(0)
            else:
                start_positions.append(token_start_index)
                end_positions.append(min(token_end_index, len(offset) - 1))

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    # 预处理数据集
    print("预处理训练集...")
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=4
    )
    
    print("预处理验证集...")
    val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=val_dataset.column_names,
        num_proc=4
    )

    def custom_collate_fn(batch):
        result = {}
        for key in batch[0].keys():
            try:
                # 尝试堆叠张量
                result[key] = torch.stack([sample[key] for sample in batch])
            except Exception as e:
                # 对于关键字段，强制转换为张量
                if key in ["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"]:
                    try:
                        result[key] = torch.tensor([sample[key] for sample in batch])
                    except Exception as e2:
                        print(f"警告：无法将字段 {key} 转换为张量: {e2}")
                        # 如果无法转换，仍然保留列表，但这可能会在后续处理中导致错误
                        result[key] = [sample[key] for sample in batch]
                else:
                    # 非关键字段可以保留为列表
                    result[key] = [sample[key] for sample in batch]
        return result

    # 修改DataLoader使用新的collate_fn
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn
    )

    return train_loader, val_loader, train_dataset, val_dataset


def compute_metrics(start_logits, end_logits, start_positions, end_positions):
    """
    计算问答任务的指标
    """
    # 转换为CPU上的numpy数组
    start_logits = start_logits.cpu().numpy()
    end_logits = end_logits.cpu().numpy()
    start_positions = start_positions.cpu().numpy()
    end_positions = end_positions.cpu().numpy()
    
    # 计算准确率
    start_pred = np.argmax(start_logits, axis=1)
    end_pred = np.argmax(end_logits, axis=1)
    
    start_acc = (start_pred == start_positions).mean()
    end_acc = (end_pred == end_positions).mean()
    
    # 完全匹配分数
    exact_match = ((start_pred == start_positions) & (end_pred == end_positions)).mean()
    
    return {
        "start_accuracy": float(start_acc),
        "end_accuracy": float(end_acc),
        "exact_match": float(exact_match)
    }


def evaluate_model(model, val_loader, device):
    """
    评估模型在验证集上的性能
    """
    model.eval()
    all_metrics = {
        "start_accuracy": 0.0,
        "end_accuracy": 0.0,
        "exact_match": 0.0
    }
    
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="评估中"):
            # 确保所有模型输入都是张量并移到正确设备
            input_batch = {}
            try:
                for k, v in batch.items():
                    if k in ["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"]:
                        if isinstance(v, torch.Tensor):
                            input_batch[k] = v.to(device)
                        else:
                            # 将列表转换为张量
                            input_batch[k] = torch.tensor(v).to(device)
                
                # 确保所有必要字段都存在
                if not all(k in input_batch for k in ["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"]):
                    print("警告: 批次缺少必要字段，跳过")
                    continue
                    
                outputs = model(
                    input_ids=input_batch["input_ids"],
                    attention_mask=input_batch["attention_mask"],
                    token_type_ids=input_batch["token_type_ids"]
                )
                
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                
                metrics = compute_metrics(
                    start_logits,
                    end_logits,
                    input_batch["start_positions"],
                    input_batch["end_positions"]
                )
                
                batch_size = input_batch["input_ids"].size(0)
                total_samples += batch_size
                
                for key in all_metrics:
                    all_metrics[key] += metrics[key] * batch_size
            
            except Exception as e:
                print(f"处理批次时出错，跳过: {e}")
                continue
    
    # 计算平均值
    if total_samples > 0:
        for key in all_metrics:
            all_metrics[key] /= total_samples
    else:
        print("警告：没有成功处理任何批次，指标可能不准确")
        
    return all_metrics


def main():
    # --- 配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    base_dir = os.path.dirname(__file__)
    output_cache_dir = os.path.join(base_dir, 'cache_bert_squad_pruned')
    os.makedirs(output_cache_dir, exist_ok=True)
    
    model_path = os.path.join(base_dir, "models", "bert-large-uncased-whole-word-masking-finetuned-squad")
    
    batch_size = 16
    max_seq_length = 384
    original_model_size_bytes = 0

    # --- 第1步：加载模型和数据集 ---
    print("\n--- 第1步：加载模型和数据集 ---")
    
    # 加载模型和tokenizer
    print(f"从 {model_path} 加载预训练的BERT模型...")
    try:
        tokenizer = BertTokenizerFast.from_pretrained(model_path, local_files_only=True)
        model = BertForQuestionAnswering.from_pretrained(model_path, local_files_only=True)
        print("成功从本地加载模型和tokenizer")
    except Exception as e:
        print(f"从本地加载失败: {e}，尝试从Hugging Face下载")
        tokenizer = BertTokenizerFast.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        
        # 保存到本地
        os.makedirs(model_path, exist_ok=True)
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        print(f"模型和tokenizer已保存到: {model_path}")
    
    model.to(device)
    model_file = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(model_file):
        original_model_size_bytes = os.path.getsize(model_file)
        print(f"原始模型文件大小: {original_model_size_bytes / (1024**2):.2f} MB")
    
    # 加载SQuAD数据集
    train_loader, val_loader, train_dataset, val_dataset = prepare_squad_dataset(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_seq_length
    )
    
    print(f"数据集加载完成。训练集大小: {len(train_dataset)}，验证集大小: {len(val_dataset)}")

    # 关闭缓存以确保 FLOPs 计算准确
    if hasattr(model, 'config'):
        model.config.use_cache = False
    
    # --- 第2步：初始化OTO ---
    print("\n--- 第2步：初始化OTO ---")
    # 创建一个示例输入
    context = "BERT (Bidirectional Encoder Representations from Transformers) 是谷歌开发的一种基于Transformer的机器学习技术，用于自然语言处理的预训练。"
    question = "谁开发了BERT？"
    dummy_input = tokenizer(question, context, return_tensors='pt').input_ids.to(device)
    
    oto = OTO(model=model, dummy_input=dummy_input, strict_out_nodes=True)
    print("正在生成剪枝前的依赖图...")
    try:
        oto.visualize(view=False, out_dir=output_cache_dir)
        print("依赖图已生成。")
    except Exception as e:
        print(f"无法生成依赖图: {e}")
        print("将继续执行，但不生成可视化依赖图。")
    
    # --- 第3步：剪枝前评估 ---
    print("\n--- 第3步：剪枝前评估 ---")
    metrics_before = evaluate_model(model, val_loader, device)
    print(f"剪枝前评估结果:")
    print(f"  - 起始位置准确率: {metrics_before['start_accuracy']:.4f}")
    print(f"  - 结束位置准确率: {metrics_before['end_accuracy']:.4f}")
    print(f"  - 精确匹配率: {metrics_before['exact_match']:.4f}")
    
    flops_before_m = oto.compute_flops(in_million=True)['total']
    params_before_m = oto.compute_num_params(in_million=True)
    print(f"剪枝前 FLOPs: {flops_before_m:.2f} M, 参数量: {params_before_m:.2f} M")
    
    # --- 第4步：剪枝和微调 ---
    print("\n--- 第4步：剪枝和微调 ---")
    learning_rate_prune = 1e-5      # 阶段一的初始学习率
    finetune_lr = 2e-6              # 阶段二的微调学习率
    num_train_epochs = 120          # 总训练周期
    pruning_periods = 200           # 增加剪枝的频率，使其更平滑
    lr_scheduler_type = "cosine"    # 学习率调度器类型
    group_sparsity = 0.3            # 目标稀疏率
    warmup_epochs = 5               # 热身阶段周期数
    pruning_epochs = 45             # 剪枝阶段周期数
    finetune_epochs = num_train_epochs - warmup_epochs - pruning_epochs     # 微调周期数
    
    num_update_steps_per_epoch = len(train_loader)
    finetune_steps = finetune_epochs * num_update_steps_per_epoch
    
    # --- 阶段一：预热和剪枝 ---
    print("\n--- 阶段一：预热与剪枝 ---")
    pruning_and_warmup_epochs = warmup_epochs + pruning_epochs
    pruning_and_warmup_steps = pruning_and_warmup_epochs * num_update_steps_per_epoch
    
    start_pruning_step = warmup_epochs * num_update_steps_per_epoch
    pruning_steps = pruning_epochs * num_update_steps_per_epoch
    
    print(f"目标组稀疏率: {group_sparsity}")
    print(f"总训练周期: {num_train_epochs} = {warmup_epochs}(热身) + {pruning_epochs}(剪枝) + {finetune_epochs}(微调)")
    print(f"剪枝将在第 {start_pruning_step} 步开始，持续 {pruning_steps} 步。")
    
    optimizer_prune = oto.hesso(
        variant='adamw',
        lr=learning_rate_prune,
        target_group_sparsity=group_sparsity,
        start_pruning_step=start_pruning_step,
        pruning_periods=pruning_periods,
        pruning_steps=pruning_steps,
    )
    
    lr_scheduler_prune = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer_prune,
        num_warmup_steps=0,
        num_training_steps=pruning_and_warmup_steps,
    )
    
    # 阶段一训练主循环
    print("\n--- 开始预热与剪枝训练 ---")
    progress_bar_prune = tqdm(range(pruning_and_warmup_steps), desc="[阶段一] 剪枝中")
    model.train()
    
    for epoch in range(pruning_and_warmup_epochs):
        for batch in train_loader:
            # 确保所有输入都是张量
            processed_batch = {}
            try:
                for k, v in batch.items():
                    if k in ["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"]:
                        if isinstance(v, torch.Tensor):
                            processed_batch[k] = v.to(device)
                        else:
                            processed_batch[k] = torch.tensor(v).to(device)
                            
                outputs = model(
                    input_ids=processed_batch["input_ids"],
                    attention_mask=processed_batch["attention_mask"],
                    token_type_ids=processed_batch["token_type_ids"],
                    start_positions=processed_batch["start_positions"],
                    end_positions=processed_batch["end_positions"]
                )

            except Exception as e:
                print(f"处理批次时出错，跳过: {e}")
                continue
            
            loss = outputs.loss
            loss.backward()
            
            optimizer_prune.step()
            lr_scheduler_prune.step()
            optimizer_prune.zero_grad()
            
            progress_bar_prune.update(1)
            
            metrics = optimizer_prune.compute_metrics()
            current_lr = optimizer_prune.param_groups[0]['lr']
            progress_bar_prune.set_description(
                f"[阶段一] Epoch {epoch+1}, Loss: {loss.item():.4f}, "
                f"Sparsity: {metrics.group_sparsity:.4f}, LR: {current_lr:.6f}"
            )
            
            # 检查是否已完成预定步数
            if progress_bar_prune.n >= pruning_and_warmup_steps:
                break
                
        # 在每个epoch结束后评估
        print(f"\n--- [阶段一] Epoch {epoch+1}/{pruning_and_warmup_epochs} 评估 ---")
        metrics_epoch = evaluate_model(model, val_loader, device)
        print(f"  - 起始位置准确率: {metrics_epoch['start_accuracy']:.4f}")
        print(f"  - 结束位置准确率: {metrics_epoch['end_accuracy']:.4f}")
        print(f"  - 精确匹配率: {metrics_epoch['exact_match']:.4f}")
        model.train()
    
    print("\n--- 剪枝阶段完成，模型结构已固定 ---")
    
    # --- 构建并加载子网络 ---
    print("\n--- 构建并加载子网络用于微调 ---")
    oto.construct_subnet(out_dir=output_cache_dir)
    compressed_model_path = oto.compressed_model_path
    if not os.path.exists(compressed_model_path):
        print(f"错误：压缩模型未在 {compressed_model_path} 找到。无法进行微调。")
        return
    
    print(f"加载压缩后的模型从: {compressed_model_path}")
    model = torch.load(compressed_model_path, map_location=device, weights_only=False)
    model.to(device)
    print("子网络加载完成，将用此模型进行微调。")
    
    # --- 阶段二：微调 ---
    print("\n--- 阶段二：微调 ---")
    print(f"将使用AdamW优化器进行 {finetune_epochs} 个周期的微调。")
    
    optimizer_finetune = torch.optim.AdamW(
        model.parameters(),
        lr=finetune_lr,
        weight_decay=0.01
    )
    
    lr_scheduler_finetune = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer_finetune,
        num_warmup_steps=0,
        num_training_steps=finetune_steps,
    )
    
    progress_bar_finetune = tqdm(range(finetune_steps), desc="[阶段二] 微调中")
    model.train()
    
    for epoch in range(finetune_epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                start_positions=batch["start_positions"],
                end_positions=batch["end_positions"]
            )
            
            loss = outputs.loss
            loss.backward()
            
            optimizer_finetune.step()
            lr_scheduler_finetune.step()
            optimizer_finetune.zero_grad()
            
            progress_bar_finetune.update(1)
            
            current_lr = optimizer_finetune.param_groups[0]['lr']
            progress_bar_finetune.set_description(
                f"[阶段二] Epoch {epoch+1+pruning_and_warmup_epochs}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}"
            )
            
            # 检查是否已完成预定步数
            if progress_bar_finetune.n >= finetune_steps:
                break
        
        # 在每个epoch结束后评估
        print(f"\n--- [阶段二] Epoch {epoch+1+pruning_and_warmup_epochs}/{num_train_epochs} 评估 ---")
        metrics_epoch = evaluate_model(model, val_loader, device)
        print(f"  - 起始位置准确率: {metrics_epoch['start_accuracy']:.4f}")
        print(f"  - 结束位置准确率: {metrics_epoch['end_accuracy']:.4f}")
        print(f"  - 精确匹配率: {metrics_epoch['exact_match']:.4f}")
        model.train()
    
    print("剪枝和微调过程完成。")
    
    print(f"\n--- 保存微调后的模型到: {compressed_model_path} ---")
    torch.save(model, compressed_model_path)
    print("模型保存成功。")
    
    model.eval()
    
    # --- 第5步：最终评估 ---
    print("\n--- 第5步：最终模型评估 ---")
    final_metrics = optimizer_prune.compute_metrics()
    final_group_sparsity = final_metrics.group_sparsity
    print(f"  - 最终组稀疏度: {final_metrics.group_sparsity:.4f}")
    print(f"  - 总可剪枝组数: {final_metrics.num_groups}")
    print(f"  - 零权重组数: {final_metrics.num_zero_groups}")
    print(f"  - 重要组数: {final_metrics.num_important_groups}")
    print(f"  - 冗余组数: {final_metrics.num_redundant_groups}")
    
    print("\n正在评估最终微调后模型的性能...")
    metrics_after = evaluate_model(model, val_loader, device)
    print(f"剪枝后评估结果:")
    print(f"  - 起始位置准确率: {metrics_after['start_accuracy']:.4f}")
    print(f"  - 结束位置准确率: {metrics_after['end_accuracy']:.4f}")
    print(f"  - 精确匹配率: {metrics_after['exact_match']:.4f}")
    
    # 计算剪枝后的FLOPs和参数数量
    flops_after_m = oto.compute_flops(in_million=True)['total']
    params_after_m = oto.compute_num_params(in_million=True)
    print(f"剪枝后 FLOPs: {flops_after_m:.2f} M")
    print(f"剪枝后参数量: {params_after_m:.2f} M")
    
    compressed_model_size_bytes = os.path.getsize(compressed_model_path)
    print(f"压缩模型文件大小: {compressed_model_size_bytes / (1024**2):.2f} MB")
    
    # --- 第6步：压缩总结 ---
    print("\n--- 第6步：压缩总结 ---")
    print(f"最终剪枝率 (组稀疏率): {final_group_sparsity:.2f}")
    
    if flops_before_m > 0 and flops_after_m >= 0:
        flops_compression_ratio = flops_before_m / flops_after_m if flops_after_m > 0 else float('inf')
        print(f"FLOPs 压缩比: {flops_compression_ratio:.2f}x ({flops_before_m:.2f}M -> {flops_after_m:.2f}M)")
    
    if params_before_m > 0 and params_after_m >= 0:
        params_compression_ratio = params_before_m / params_after_m if params_after_m > 0 else float('inf')
        print(f"参数量压缩比: {params_compression_ratio:.2f}x ({params_before_m:.2f}M -> {params_after_m:.2f}M)")
    
    if original_model_size_bytes > 0 and compressed_model_size_bytes > 0:
        model_size_compression_ratio = original_model_size_bytes / compressed_model_size_bytes
        print(f"模型文件大小压缩比: {model_size_compression_ratio:.2f}x ({original_model_size_bytes / (1024**2):.2f}MB -> {compressed_model_size_bytes / (1024**2):.2f}MB)")
    
    print(f"\n性能变化:")
    print(f"  - 起始位置准确率: {metrics_before['start_accuracy']:.4f} -> {metrics_after['start_accuracy']:.4f} (变化: {metrics_after['start_accuracy'] - metrics_before['start_accuracy']:+.4f})")
    print(f"  - 结束位置准确率: {metrics_before['end_accuracy']:.4f} -> {metrics_after['end_accuracy']:.4f} (变化: {metrics_after['end_accuracy'] - metrics_before['end_accuracy']:+.4f})")
    print(f"  - 精确匹配率: {metrics_before['exact_match']:.4f} -> {metrics_after['exact_match']:.4f} (变化: {metrics_after['exact_match'] - metrics_before['exact_match']:+.4f})")
    
    # --- 第7步：导出压缩模型 ---
    print("\n--- 第7步：导出压缩模型 ---")
    print(f"压缩后的模型已保存在: {oto.compressed_model_path}")
    print(f"完整的组稀疏模型 (未移除零权重组) 保存在: {oto.full_group_sparse_model_path}")
    print("\n剪枝和压缩测试完成。")

if __name__ == "__main__":
    main()
    