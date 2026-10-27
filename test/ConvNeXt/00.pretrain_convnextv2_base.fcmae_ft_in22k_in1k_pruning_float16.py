import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
import math
from tqdm.auto import tqdm
from transformers import get_scheduler
from torch.cuda.amp import GradScaler  # 已导入，用于混合精度训练

# 将项目根目录添加到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from only_train_once import OTO
from data.imagenet_dataloader import load_prepared_imagenet_data 

def check_accuracy(model, loader):
    """
    计算模型在给定数据加载器上的 Top-1 和 Top-5 准确率。
    """
    model.eval()
    num_correct = 0
    num_samples = 0
    top5_num_correct = 0
    model_device = next(model.parameters()).device

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(model_device, non_blocking=True)
            y = y.to(model_device, non_blocking=True)
            
            with torch.amp.autocast(device_type=model_device.type):
                scores = model(x)
            
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)

            _, top5_indices = scores.topk(5, dim=1)
            y_reshaped = y.view(-1, 1)
            top5_correct_batch = (top5_indices == y_reshaped).any(dim=1).sum().item()
            top5_num_correct += top5_correct_batch
            
            if (i + 1) % 100 == 0:  # 每处理100个批次打印一次进度
                print(f"  Accuracy check: processed batch {i+1}/{len(loader)}")

    if num_samples == 0:
        print("警告: check_accuracy 接收到的样本数量为0。")
        return 0.0, 0.0

    acc = float(num_correct) / num_samples * 100
    top5_acc = float(top5_num_correct) / num_samples * 100
    
    print(f"  Accuracy check: Total samples: {num_samples}, Top-1 Correct: {num_correct}, Top-5 Correct: {top5_num_correct}")
    return acc, top5_acc

def main():
    # --- 配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    base_dir = os.path.dirname(__file__)
    imagenet_data_path = os.path.join(project_root, 'data', 'imagenet')
    output_cache_dir = os.path.join(base_dir, 'convnextv2_base_imagenet_pruned')
    os.makedirs(output_cache_dir, exist_ok=True)
    
    # 定义模型权重路径
    local_model_dir = os.path.join('/workspace/UniPruning/test_new/ConvNeXt/models/convnextv2_base.fcmae_ft_in22k_in1k')
    local_model_weights_path = os.path.join(local_model_dir, 'pytorch_model.bin')

    batch_size = 64                 # batch_size
    num_workers = 48                # worker数量
    input_size = 224                # ConvNeXtV2 Base的标准输入大小
    original_model_size_bytes = 0   # 初始化原始模型大小

    # --- 第 1 步：加载数据和预训练模型 ---
    print("\n--- 第 1 步：加载数据和预训练模型 ---")

    print("正在加载 ConvNeXtV2 Base 模型...")
    if os.path.exists(local_model_weights_path):
        print(f"从本地路径加载预训练权重: {local_model_weights_path}")
        model = timm.create_model(
            'convnextv2_base.fcmae_ft_in22k_in1k',
            pretrained=False,
            num_classes=1000
        )
        try:
            model.load_state_dict(torch.load(local_model_weights_path, map_location=device, weights_only=True))
            print("成功从本地加载权重。")
            original_model_size_bytes = os.stat(local_model_weights_path).st_size
        except Exception as e:
            print(f"从本地加载权重失败: {e}")
            return
    else:
        print(f"错误：本地权重文件 {local_model_weights_path} 未找到。")
        return

    model.to(device)
    model.eval()
    print("ConvNeXtV2 Base 模型加载完成。")
    if original_model_size_bytes > 0:
        print(f"原始模型文件大小: {original_model_size_bytes / (1024**2):.2f} MB")

    print(f"正在从 {imagenet_data_path} 加载 ImageNet 数据...")
    try:
        train_loader, val_loader, train_dataset_full, val_dataset = load_prepared_imagenet_data(
            base_imagenet_path=imagenet_data_path,
            batch_size=batch_size,
            workers=num_workers,
            image_size=input_size, 
            pin_memory=True if device.type == 'cuda' else False
        )
        print(f"成功加载 ImageNet 完整训练集，包含 {len(train_dataset_full)} 张图片。")
        
        print(f"训练集图片数: {len(train_dataset_full)}, 验证集图片数: {len(val_dataset)}")
        print(f"训练集批次数: {len(train_loader)}, 验证集批次数: {len(val_loader)}")

        if len(val_dataset) == 0:
            print(f"警告: ImageNet 验证集为空。请检查路径 {os.path.join(imagenet_data_path, 'val')}")
            return
    except FileNotFoundError as e:
        print(f"加载 ImageNet 数据集失败: {e}")
        print(f"请确保 ImageNet 数据位于 '{imagenet_data_path}' 目录，")
        print("并且该目录下包含 'train' 和 'val' 子文件夹 (val 文件夹内容需要按类别整理)。")
        return
    except Exception as e:
        print(f"加载 ImageNet 数据集时发生未知错误: {e}")
        return

    # --- 第 2 步：初始化 OTO ---
    print("\n--- 第 2 步：初始化 OTO ---")
    dummy_input = torch.rand(1, 3, input_size, input_size).to(device)
    oto = OTO(model=model, dummy_input=dummy_input)
    print("正在生成剪枝前的依赖图...")
    oto.visualize(view=False, out_dir=output_cache_dir)

    print(f"训练集大小（训练集批次数）: {len(train_loader)}, 验证集大小（验证集批次数）: {len(val_loader)}")

    # --- 第 3 步：剪枝前评估 ---
    print("\n--- 第 3 步：剪枝前评估 ---")
    print("正在评估剪枝前模型的准确率 (这可能需要一些时间)...")
    accuracy1_before, accuracy5_before = check_accuracy(model, val_loader)
    print(f"剪枝前准确率: Top-1: {accuracy1_before:.2f}%, Top-5: {accuracy5_before:.2f}%")

    flops_before_m = oto.compute_flops(in_million=True)['total']
    params_before_m = oto.compute_num_params(in_million=True)
    print(f"剪枝前 FLOPs: {flops_before_m:.2f} M")
    print(f"剪枝前参数量: {params_before_m:.2f} M")

    # --- 第 4 步：使用 OTO 进行剪枝和微调 ---
    print("\n--- 第 4 步：使用 OTO 进行剪枝和微调 ---")

    learning_rate = 2e-3             # 阶段一的初始学习率
    num_train_epochs = 60            # 总训练周期
    group_sparsity = 0.5             # 目标组稀疏率
    lr_scheduler_type = "cosine"     # 学习率调度器类型
    finetune_lr = 2e-4          # 微调时使用较小的学习率
    
    # 阶段划分
    warmup_epochs = 5   # 热身阶段周期数
    pruning_epochs = 15 # 剪枝周期
    pruning_periods = 200            # 增加剪枝的频率，使其更平滑
    finetune_epochs = num_train_epochs - warmup_epochs - pruning_epochs
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

    criterion = nn.CrossEntropyLoss().to(device)

    # 设置 OTO HESSO 优化器 (仅用于阶段一)
    optimizer_prune = oto.hesso(
        variant='adamw',
        lr=learning_rate,
        target_group_sparsity=group_sparsity,
        start_pruning_step=start_pruning_step,
        pruning_periods=pruning_periods,
        pruning_steps=pruning_steps,
        fixed_zero_groups=True,
    )

    # 创建阶段一的学习率调度器
    lr_scheduler_prune = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer_prune,
        num_warmup_steps=0,
        num_training_steps=pruning_and_warmup_steps,
    )

    # 初始化 GradScaler 用于混合精度训练
    scaler_prune = GradScaler()

    # 阶段一训练主循环（启用混合精度）
    print("\n--- 开始预热与剪枝训练 ---")
    progress_bar_prune = tqdm(range(pruning_and_warmup_steps), desc="[阶段一] 剪枝中")
    model.train()
    for epoch in range(pruning_and_warmup_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler_prune.scale(loss).backward()
            scaler_prune.step(optimizer_prune)
            scaler_prune.update()
            lr_scheduler_prune.step()
            optimizer_prune.zero_grad()
            
            progress_bar_prune.update(1)
            
            metrics = optimizer_prune.compute_metrics()
            current_lr = optimizer_prune.param_groups[0]['lr']
            progress_bar_prune.set_description(
                f"[阶段一] Epoch {epoch+1}, Loss: {loss.item():.4f}, "
                f"Sparsity: {metrics.group_sparsity:.4f}, LR: {current_lr:.6f}"
            )

        # 在每个 epoch 结束后进行评估
        print(f"\n--- [阶段一] Epoch {epoch+1}/{pruning_and_warmup_epochs} 评估 ---")
        model.eval()
        accuracy1_epoch, accuracy5_epoch = check_accuracy(model, val_loader)
        print(f"Epoch {epoch+1} 验证准确率: Top-1: {accuracy1_epoch:.2f}%, Top-5: {accuracy5_epoch:.2f}%")
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
    model = torch.load(compressed_model_path, map_location=device)
    model.to(device)
    print("子网络加载完成，将用此模型进行微调。")

    # --- 阶段二：微调 ---
    print("\n--- 阶段二：微调 ---")
    print(f"将使用新的标准AdamW优化器进行 {finetune_epochs} 个周期的微调。")
    print(f"微调学习率: {finetune_lr}")

    optimizer_finetune = torch.optim.AdamW(
        model.parameters(),
        lr=finetune_lr,
        weight_decay=1e-4
    )

    lr_scheduler_finetune = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer_finetune,
        num_warmup_steps=0,
        num_training_steps=finetune_steps,
    )

    # 初始化 GradScaler 用于阶段二的混合精度训练
    scaler_finetune = GradScaler()

    # 阶段二训练主循环（启用混合精度）
    progress_bar_finetune = tqdm(range(finetune_steps), desc="[阶段二] 微调中")
    model.train()
    for epoch in range(finetune_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler_finetune.scale(loss).backward()
            scaler_finetune.step(optimizer_finetune)
            scaler_finetune.update()
            lr_scheduler_finetune.step()
            optimizer_finetune.zero_grad()
            
            progress_bar_finetune.update(1)
            current_lr = optimizer_finetune.param_groups[0]['lr']
            progress_bar_finetune.set_description(
                f"[阶段二] Epoch {epoch+1+pruning_and_warmup_epochs}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}"
            )
        
        print(f"\n--- [阶段二] Epoch {epoch+1+pruning_and_warmup_epochs}/{num_train_epochs} 评估 ---")
        model.eval()
        accuracy1_epoch, accuracy5_epoch = check_accuracy(model, val_loader)
        print(f"Epoch {epoch+1+pruning_and_warmup_epochs} 验证准确率: Top-1: {accuracy1_epoch:.2f}%, Top-5: {accuracy5_epoch:.2f}%")
        model.train()

    print("剪枝和微调过程完成。")
    
    print(f"\n--- 保存微调后的模型到: {compressed_model_path} ---")
    torch.save(model, compressed_model_path)
    print("模型保存成功。")

    model.eval()

    print("\n--- 计算最终模型指标 ---")
    final_metrics = optimizer_prune.compute_metrics()
    final_group_sparsity = final_metrics.group_sparsity
    print(f"  - 最终组稀疏度 (Final Group Sparsity): {final_metrics.group_sparsity:.4f}")
    print(f"  - 总可剪枝组数 (Total Prunable Groups): {final_metrics.num_groups}")
    print(f"  - 零权重组数 (Zero Groups): {final_metrics.num_zero_groups}")
    print(f"  - 重要组数 (Important Groups): {final_metrics.num_important_groups}")
    print(f"  - 冗余组数 (Redundant Groups): {final_metrics.num_redundant_groups}")
    print(f"  - 总参数范数 (Total Param Norm): {final_metrics.norm_params:.2f}")

    # --- 第 5 步：最终模型评估 ---
    print("\n--- 第 5 步：最终模型评估 ---")
    
    compressed_model = model
    print("正在评估最终微调后模型的准确率...")
    accuracy1_after, accuracy5_after = check_accuracy(compressed_model, val_loader)
    print(f"剪枝后准确率: Top-1: {accuracy1_after:.2f}%, Top-5: {accuracy5_after:.2f}%")

    flops_after_m = oto.compute_flops(in_million=True)['total']
    params_after_m = oto.compute_num_params(in_million=True)
    print(f"剪枝后 FLOPs: {flops_after_m:.2f} M")
    print(f"剪枝后参数量: {params_after_m:.2f} M")

    compressed_model_size_bytes = os.stat(oto.compressed_model_path).st_size
    print(f"压缩模型文件大小: {compressed_model_size_bytes / (1024**2):.2f} MB")

    # --- 第 6 步：压缩总结 ---
    print("\n--- 第 6 步：压缩总结 ---")
    print(f"最终剪枝率 (组稀疏率): {final_group_sparsity:.2f}")
    
    if flops_before_m > 0 and flops_after_m >= 0:
        flops_compression_ratio = flops_before_m / flops_after_m if flops_after_m > 0 else float('inf')
        print(f"FLOPs 压缩比: {flops_compression_ratio:.2f}x ({flops_before_m:.2f}M -> {flops_after_m:.2f}M)")
    else:
        print(f"FLOPs 压缩比: N/A (剪枝前: {flops_before_m:.2f}M, 剪枝后: {flops_after_m:.2f}M)")

    if params_before_m > 0 and params_after_m >= 0:
        params_compression_ratio = params_before_m / params_after_m if params_after_m > 0 else float('inf')
        print(f"参数量压缩比: {params_compression_ratio:.2f}x ({params_before_m:.2f}M -> {params_after_m:.2f}M)")
    else:
        print(f"参数量压缩比: N/A (剪枝前: {params_before_m:.2f}M, 剪枝后: {params_after_m:.2f}M)")

    if original_model_size_bytes > 0 and compressed_model_size_bytes > 0:
        model_size_compression_ratio = original_model_size_bytes / compressed_model_size_bytes
        print(f"模型文件大小压缩比: {model_size_compression_ratio:.2f}x ({original_model_size_bytes / (1024**2):.2f}MB -> {compressed_model_size_bytes / (1024**2):.2f}MB)")
    elif original_model_size_bytes > 0:
        print(f"原始模型文件大小: {original_model_size_bytes / (1024**2):.2f} MB (压缩后模型大小: {compressed_model_size_bytes / (1024**2):.2f}MB)")
    elif compressed_model_size_bytes > 0:
        print(f"压缩后模型文件大小: {compressed_model_size_bytes / (1024**2):.2f} MB (原始模型大小未知)")
    else:
        print("模型文件大小压缩比: N/A (无法获取原始或压缩后的模型大小)")
    
    print(f"\n准确率变化: Top-1 从 {accuracy1_before:.2f}% 变为 {accuracy1_after:.2f}% (变化: {accuracy1_after - accuracy1_before:+.2f}%)")
    print(f"准确率变化: Top-5 从 {accuracy5_before:.2f}% 变为 {accuracy5_after:.2f}% (变化: {accuracy5_after - accuracy5_before:+.2f}%)")

    # --- 第 7 步：导出压缩模型 ---
    print("\n--- 第 7 步：导出压缩模型 ---")
    print(f"压缩后的模型已保存在: {oto.compressed_model_path}")
    print(f"完整的组稀疏模型 (未移除零权重组) 保存在: {oto.full_group_sparse_model_path}")
    print("\nConvNeXtV2 Base ImageNet 剪枝和压缩测试完成。")

if __name__ == "__main__":
    main()
