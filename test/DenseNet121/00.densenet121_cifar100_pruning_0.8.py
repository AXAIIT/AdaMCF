import sys
import os
import random
from pathlib import Path

import detectors
# timm 已不再需要，可以移除
# import timm
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import get_scheduler

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 将模型定义文件所在的目录添加到 sys.path
MODEL_DEFINITION_ROOT = CURRENT_FILE.parent / "model"
if str(MODEL_DEFINITION_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_DEFINITION_ROOT))

DATA_MODULE_ROOT = Path("/workspace/data")
if str(DATA_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_MODULE_ROOT))

from cifar100_dataloader import (
    get_cifar100_train_loader,
    get_cifar100_test_loader,
)
from only_train_once import OTO
from densenet import densenet121


def load_pretrained_densenet121(weights_path: str, device: torch.device) -> nn.Module:
    # 使用与训练时相同的模型定义
    model = densenet121()
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    # 直接加载 model_state_dict，不再检查
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    return model.to(device)


def check_accuracy(model, loader):
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            _, top1 = outputs.max(dim=1)
            _, top5 = outputs.topk(5, dim=1)
            top1_correct += (top1 == targets).sum().item()
            top5_correct += (
                top5 == targets.unsqueeze(1)
            ).any(dim=1).sum().item()
            total += targets.size(0)
    if total == 0:
        return 0.0, 0.0
    return top1_correct / total * 100.0, top5_correct / total * 100.0


def main():
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    base_dir = CURRENT_FILE.parent
    output_cache_dir = base_dir / "cache_densenet121_cifar100_pruned"
    output_cache_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 256
    num_workers = 36 
    data_root_dir = "/workspace/data/cifar100"
    weights_path = "/workspace/OTOv2_v2/test/pretrain_densenet121_cifar100/model/best_model_81.pth"

    model = load_pretrained_densenet121(weights_path=weights_path, device=device)

    print("加载 CIFAR-100 数据集...")
    trainloader, trainset = get_cifar100_train_loader(
        data_root_dir=data_root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        download=False,
    )
    testloader, testset = get_cifar100_test_loader(
        data_root_dir=data_root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        download=False,
    )
    print(
        f"训练集: {len(trainset)} 张，测试集: {len(testset)} 张，批量 {batch_size}"
    )

    dummy_input = torch.rand(1, 3, 32, 32, device=device)
    oto = OTO(model=model, dummy_input=dummy_input)
    print("生成剪枝前依赖图...")
    oto.visualize(view=False, out_dir=str(output_cache_dir))

    nodes_to_keep_unpruned = []
    if nodes_to_keep_unpruned:
        oto.mark_unprunable_by_node_ids(nodes_to_keep_unpruned)

    print("\n--- 剪枝前评估 ---")
    accuracy1_before, accuracy5_before = check_accuracy(model, testloader)
    print(
        f"剪枝前准确率: Top-1 {accuracy1_before:.2f}%, Top-5 {accuracy5_before:.2f}%"
    )

    flops_before_m = oto.compute_flops(in_million=True)["total"]
    params_before_m = oto.compute_num_params(in_million=True)
    print(f"剪枝前 FLOPs: {flops_before_m:.2f} M")
    print(f"剪枝前参数量: {params_before_m:.2f} M")

    original_model_size = (
        os.stat(weights_path).st_size / (1024**2)
        if os.path.exists(weights_path)
        else 0
    )
    if original_model_size:
        print(f"原始权重文件大小: {original_model_size:.2f} MB")

    learning_rate = 5e-3
    finetune_lr = 5e-3
    num_train_epochs = 300
    warmup_epochs = 10
    pruning_epochs = 90
    pruning_periods = pruning_epochs
    finetune_epochs = num_train_epochs - warmup_epochs - pruning_epochs
    target_group_sparsity = 0.8
    lr_scheduler_type = "cosine"

    num_update_steps_per_epoch = len(trainloader)
    warmup_steps = warmup_epochs * num_update_steps_per_epoch
    pruning_steps = pruning_epochs * num_update_steps_per_epoch
    pruning_and_warmup_steps = (warmup_epochs + pruning_epochs) * num_update_steps_per_epoch
    finetune_steps = max(finetune_epochs * num_update_steps_per_epoch, 1)
    start_pruning_step = warmup_steps

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer_prune = oto.hesso(
        variant="sgd",
        lr=learning_rate,
        target_group_sparsity=target_group_sparsity,
        start_pruning_step=start_pruning_step,
        pruning_periods=pruning_periods,
        pruning_steps=pruning_steps,
        device=device,
    )
    lr_scheduler_prune = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer_prune,
        num_warmup_steps=warmup_steps,
        num_training_steps=pruning_and_warmup_steps,
    )

    print("\n--- 阶段一：预热与剪枝 ---")
    progress_bar_prune = tqdm(range(pruning_and_warmup_steps), desc="[阶段一] 剪枝中")
    model.train()
    for epoch in range(warmup_epochs + pruning_epochs):
        for inputs, targets in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_prune.step()
            lr_scheduler_prune.step()
            optimizer_prune.zero_grad()
            progress_bar_prune.update(1)
            metrics = optimizer_prune.compute_metrics()
            current_lr = optimizer_prune.param_groups[0]["lr"]
            progress_bar_prune.set_description(
                f"[阶段一] Epoch {epoch + 1}, Loss {loss.item():.4f}, "
                f"Sparsity {metrics.group_sparsity:.4f}, LR {current_lr:.6f}"
            )

        model.eval()
        acc1_epoch, acc5_epoch = check_accuracy(model, testloader)
        print(
            f"--- [阶段一] Epoch {epoch + 1}/{warmup_epochs + pruning_epochs} 测试:"
            f" Top-1 {acc1_epoch:.2f}%, Top-5 {acc5_epoch:.2f}%",
            file=sys.stdout, 
            flush=True
        )
        model.train()

    print("\n--- 剪枝阶段完成，构建子网络 ---")
    oto.construct_subnet(out_dir=str(output_cache_dir))
    compressed_model_path = oto.compressed_model_path
    if not os.path.exists(compressed_model_path):
        raise FileNotFoundError(
            f"压缩模型未找到: {compressed_model_path}"
        )

    print(f"加载压缩模型: {compressed_model_path}")
    model = torch.load(compressed_model_path, map_location=device, weights_only=False)
    model.to(device)

    optimizer_finetune = torch.optim.SGD(
        model.parameters(), lr=finetune_lr, momentum=0.9, weight_decay=5e-4
    )
    lr_scheduler_finetune = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer_finetune,
        num_warmup_steps=0,
        num_training_steps=finetune_steps,
    )

    print("\n--- 阶段二：微调 ---")
    best_accuracy1 = 0.0
    best_epoch = 0
    progress_bar_finetune = tqdm(range(finetune_steps), desc="[阶段二] 微调中")
    model.train()
    for epoch in range(finetune_epochs):
        for inputs, targets in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_finetune.step()
            lr_scheduler_finetune.step()
            optimizer_finetune.zero_grad()
            progress_bar_finetune.update(1)
            current_lr = optimizer_finetune.param_groups[0]["lr"]
            progress_bar_finetune.set_description(
                f"[阶段二] Epoch {epoch + 1 + warmup_epochs + pruning_epochs}, "
                f"Loss {loss.item():.4f}, LR {current_lr:.6f}"
            )

        model.eval()
        acc1_epoch, acc5_epoch = check_accuracy(model, testloader)
        epoch_idx = epoch + 1 + warmup_epochs + pruning_epochs
        print(
            f"\n--- [阶段二] Epoch {epoch_idx}/{num_train_epochs} 测试:"
            f" Top-1 {acc1_epoch:.2f}%, Top-5 {acc5_epoch:.2f}%"
        )
        if acc1_epoch > best_accuracy1:
            best_accuracy1 = acc1_epoch
            best_epoch = epoch_idx
            print(
                f"当前最佳 Top-1 准确率: {best_accuracy1:.2f}% (Epoch {best_epoch})"
            )
            print(f"保存模型至: {compressed_model_path}")
            torch.save(model, compressed_model_path)
        model.train()

    print("\n剪枝与微调完成，载入最佳模型评估...")
    model = torch.load(compressed_model_path, map_location=device, weights_only=False)
    model.to(device)
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

    print("\n--- 第 5 步：最终模型评估 ---")
    compressed_model = model
    accuracy1_after, accuracy5_after = check_accuracy(compressed_model, testloader)
    print(f"剪枝后准确率: Top-1: {accuracy1_after:.2f}%, Top-5: {accuracy5_after:.2f}%")

    flops_after_m = oto.compute_flops(in_million=True)['total']
    params_after_m = oto.compute_num_params(in_million=True)
    print(f"剪枝后 FLOPs: {flops_after_m:.2f} M")
    print(f"剪枝后参数量: {params_after_m:.2f} M")

    compressed_model_size_bytes = os.stat(oto.compressed_model_path).st_size if os.path.exists(oto.compressed_model_path) else 0
    if compressed_model_size_bytes:
        print(f"压缩模型文件大小: {compressed_model_size_bytes / (1024**2):.2f} MB")
    else:
        print("无法获取压缩模型文件大小。")

    final_metrics = optimizer_prune.compute_metrics(
        flops_before=flops_before_m,
        flops_after=flops_after_m,
        params_before=params_before_m,
        params_after=params_after_m,
    )
    final_group_sparsity = final_metrics.group_sparsity

    print("\n--- 计算最终模型指标 ---")
    print(f"  - 最终组稀疏度 (Final Group Sparsity): {final_group_sparsity:.4f} ({final_group_sparsity * 100:.2f}%)")
    print(f"  - 总可剪枝组数 (Total Prunable Groups): {final_metrics.num_groups}")
    print(f"  - 零权重组数 (Zero Groups): {final_metrics.num_zero_groups} ({final_metrics.zero_group_ratio * 100:.2f}%)")
    print(f"  - 已剪枝组数 (Pruned Groups): {final_metrics.num_pruned_groups} ({final_metrics.pruned_group_ratio * 100:.2f}%)")
    print(f"  - 活跃冗余组数 (Active Redundant Groups): {final_metrics.num_active_redundant_groups} ({final_metrics.active_redundant_group_ratio * 100:.2f}%)")
    print(f"  - 冗余组数 (Redundant Groups): {final_metrics.num_redundant_groups} ({final_metrics.redundant_group_ratio * 100:.2f}%)")
    print(f"  - 重要组数 (Important Groups): {final_metrics.num_important_groups} ({final_metrics.important_group_ratio * 100:.2f}%)")
    print(f"  - 总参数范数 (Total Param Norm): {final_metrics.norm_params:.2f}")

    print("\n--- 第 6 步：压缩总结 ---")
    print(f"最终剪枝率 (组稀疏率): {final_group_sparsity:.2f} ({final_group_sparsity * 100:.2f}%)")

    pruned_flops_m = final_metrics.pruned_flops
    pruned_params_m = final_metrics.pruned_params
    pruned_flops_percent = final_metrics.pruned_flops_percent
    pruned_params_percent = final_metrics.pruned_params_percent

    print(f"减少的 FLOPs: {pruned_flops_m:.2f}M ({pruned_flops_percent:.2f}%)")
    print(f"减少的参数量: {pruned_params_m:.2f}M ({pruned_params_percent:.2f}%)")

    if final_metrics.flops_before > 0 and final_metrics.flops_after > 0:
        print(f"FLOPs 压缩比: {final_metrics.flops_compression_ratio:.2f}x "
              f"({final_metrics.flops_before:.2f}M -> {final_metrics.flops_after:.2f}M)")
    else:
        print(f"FLOPs 压缩比: N/A (剪枝前: {flops_before_m:.2f}M, 剪枝后: {flops_after_m:.2f}M)")

    if final_metrics.params_before > 0 and final_metrics.params_after > 0:
        print(f"参数量压缩比: {final_metrics.params_compression_ratio:.2f}x "
              f"({final_metrics.params_before:.2f}M -> {final_metrics.params_after:.2f}M)")
    else:
        print(f"参数量压缩比: N/A (剪枝前: {params_before_m:.2f}M, 剪枝后: {params_after_m:.2f}M)")

    print(f"\n准确率变化: Top-1 从 {accuracy1_before:.2f}% 变为 {accuracy1_after:.2f}% (变化: {accuracy1_after - accuracy1_before:+.2f}%)")
    print(f"准确率变化: Top-5 从 {accuracy5_before:.2f}% 变为 {accuracy5_after:.2f}% (变化: {accuracy5_after - accuracy5_before:+.2f}%)")

    print("\n--- 第 7 步：导出压缩模型 ---")
    print(f"压缩后的模型已保存在: {oto.compressed_model_path}")
    print(f"完整的组稀疏模型 (未移除零权重组) 保存在: {oto.full_group_sparse_model_path}")
    print("\n剪枝和压缩测试完成。")


if __name__ == "__main__":
    main()