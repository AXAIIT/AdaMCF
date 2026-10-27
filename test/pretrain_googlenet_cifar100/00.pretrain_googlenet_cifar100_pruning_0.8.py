import sys
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import get_scheduler
from torch.distributions.beta import Beta

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 添加模型定义目录（你的GoogLeNet代码所在路径）
MODEL_DEFINITION_ROOT = CURRENT_FILE.parent / "model"  # 替换为你的GoogLeNet代码目录
if str(MODEL_DEFINITION_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_DEFINITION_ROOT))

DATA_MODULE_ROOT = Path("/workspace/data")
if str(DATA_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_MODULE_ROOT))

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

model_base_path = os.path.join(project_root, 'model')
if model_base_path not in sys.path:
    sys.path.insert(0, model_base_path)

# 导入CIFAR-100数据加载器（复用你的代码）
from cifar100_dataloader import (
    get_cifar100_train_loader,
    get_cifar100_test_loader,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from only_train_once import OTO
from googlenet import googlenet


def load_pretrained_googlenet(weights_path: str, device: torch.device) -> nn.Module:
    model = googlenet(num_classes=100, aux_logits=True, transform_input=False)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    if hasattr(model, "aux_logits"):
        model.aux_logits = False
    if hasattr(model, "aux1"):
        model.aux1 = None
    if hasattr(model, "aux2"):
        model.aux2 = None
    return model.to(device)


def check_accuracy(model: nn.Module, loader):
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
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, top1 = outputs.max(dim=1)
            _, top5 = outputs.topk(5, dim=1)
            top1_correct += (top1 == targets).sum().item()
            top5_correct += (top5 == targets.unsqueeze(1)).any(dim=1).sum().item()
            total += targets.size(0)
    if total == 0:
        return 0.0, 0.0
    return top1_correct / total * 100.0, top5_correct / total * 100.0


def compute_loss_with_aux(outputs, targets, criterion):
    if isinstance(outputs, tuple):
        logits = outputs[0]
        aux_outputs = outputs[1:]
    else:
        logits = outputs
        aux_outputs = ()
    loss = criterion(logits, targets)
    for aux in aux_outputs:
        if aux is not None:
            loss += 0.3 * criterion(aux, targets)
    return loss


# ---- 从 train.py 复用的 mixup & aux 配置 ----
MIXUP_ALPHA = 0.4
MIXUP_ENABLE_EPOCH = 10          # 预热之后启用
MIXUP_DISABLE_EPOCH = 300        # 300 轮后关闭
AUX_WEIGHT_INIT = 0.3
AUX_DECAY_START = 260            # 260 轮开始衰减
AUX_DECAY_END = 340              # 340 轮衰减到 0


def mixup_data(inputs, targets, alpha=0.4):
    beta_dist = Beta(alpha, alpha)
    lam = beta_dist.sample().item()
    perm = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[perm]
    targets_a, targets_b = targets, targets[perm]
    return mixed_inputs, targets_a, targets_b, lam


def mixup_criterion(criterion, preds, targets_a, targets_b, lam):
    return lam * criterion(preds, targets_a) + (1 - lam) * criterion(preds, targets_b)


def current_mixup_alpha(epoch: int) -> float:
    if epoch < MIXUP_ENABLE_EPOCH or epoch >= MIXUP_DISABLE_EPOCH:
        return 0.0
    remain = MIXUP_DISABLE_EPOCH - epoch
    window = MIXUP_DISABLE_EPOCH - MIXUP_ENABLE_EPOCH
    return MIXUP_ALPHA * remain / max(1, window)


def current_aux_weight(epoch: int) -> float:
    if epoch < AUX_DECAY_START:
        return AUX_WEIGHT_INIT
    if epoch >= AUX_DECAY_END:
        return 0.0
    ratio = (AUX_DECAY_END - epoch) / max(1, AUX_DECAY_END - AUX_DECAY_START)
    return AUX_WEIGHT_INIT * ratio


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
    output_cache_dir = base_dir / "cache_googlenet_cifar100_pruned"
    output_cache_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 256
    num_workers = 36
    data_root_dir = "/workspace/data/cifar100"
    weights_path = "/workspace/OTOv2_v2/test/pretrain_googlenet_cifar100/model/best_model.pth"

    model = load_pretrained_googlenet(weights_path=str(weights_path), device=device)

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
    print(f"训练集: {len(trainset)} 张，测试集: {len(testset)} 张，批量 {batch_size}")

    dummy_input = torch.rand(1, 3, 32, 32, device=device)
    oto = OTO(model=model, dummy_input=dummy_input)
    print("生成剪枝前依赖图...")
    oto.visualize(view=False, out_dir=str(output_cache_dir))

    nodes_to_keep_unpruned = ['node-520', 'node-526', 'node-532', 'node-536']
    oto.mark_unprunable_by_node_ids(nodes_to_keep_unpruned)

    print("\n--- 剪枝前评估 ---")
    accuracy1_before, accuracy5_before = check_accuracy(model, testloader)
    print(f"剪枝前准确率: Top-1 {accuracy1_before:.2f}%, Top-5 {accuracy5_before:.2f}%")

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
    finetune_lr = 5e-2
    num_train_epochs = 410
    warmup_epochs = 10
    pruning_epochs = 100
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

    criterion = nn.CrossEntropyLoss(label_smoothing=0.02).to(device)
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
    global_step = 0  # 用于 mixup / aux 策略按 epoch 控制

    for epoch in range(warmup_epochs + pruning_epochs):
        train_loss = 0.0

        for inputs, targets in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # ---- 与 train.py 相同的 mixup 策略 ----
            mixup_alpha = current_mixup_alpha(epoch)
            use_mixup = mixup_alpha > 0.0
            if use_mixup:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=mixup_alpha)
            else:
                targets_a = targets_b = None
                lam = 1.0

            outputs = model(inputs)
            # GoogLeNet 训练时返回 (logits, aux2, aux1)
            if isinstance(outputs, tuple):
                logits, aux_logits2, aux_logits1 = outputs
            else:
                logits = outputs
                aux_logits2 = aux_logits1 = None

            # ---- 与 train.py 相同的 aux loss 衰减策略 ----
            aux_weight = current_aux_weight(epoch)

            if use_mixup:
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                if aux_logits1 is not None and aux_logits2 is not None and aux_weight > 0:
                    loss += aux_weight * mixup_criterion(criterion, aux_logits1, targets_a, targets_b, lam)
                    loss += aux_weight * mixup_criterion(criterion, aux_logits2, targets_a, targets_b, lam)
            else:
                loss = criterion(logits, targets)
                if aux_logits1 is not None and aux_logits2 is not None and aux_weight > 0:
                    loss += aux_weight * criterion(aux_logits1, targets)
                    loss += aux_weight * criterion(aux_logits2, targets)

            optimizer_prune.zero_grad()
            loss.backward()
            optimizer_prune.step()
            lr_scheduler_prune.step()

            train_loss += loss.item() * inputs.size(0)
            global_step += 1
            progress_bar_prune.update(1)

            metrics = optimizer_prune.compute_metrics()
            group_sparsity = getattr(metrics, "group_sparsity", 0.0)
            current_lr = optimizer_prune.param_groups[0]["lr"]
            progress_bar_prune.set_description(
                f"[阶段一] Epoch {epoch + 1}/{warmup_epochs + pruning_epochs}, "
                f"Loss {loss.item():.4f}, Sparsity {group_sparsity:.4f}, LR {current_lr:.6f}"
            )

        # 每个 epoch 结束后评估一次
        model.eval()
        avg_train_loss = train_loss / len(trainset)
        acc1_epoch, acc5_epoch = check_accuracy(model, testloader)
        print(
            f"--- [阶段一] Epoch {epoch + 1}/{warmup_epochs + pruning_epochs} 测试:",
            f" 训练损失: {avg_train_loss:.4f}, Top-1 {acc1_epoch:.2f}%, Top-5 {acc5_epoch:.2f}%",
            file=sys.stdout,
            flush=True
        )
        model.train()

    print("\n--- 剪枝阶段完成，构建子网络 ---")
    oto.construct_subnet(out_dir=str(output_cache_dir))
    compressed_model_path = oto.compressed_model_path
    if not os.path.exists(compressed_model_path):
        raise FileNotFoundError(f"压缩模型未找到: {compressed_model_path}")

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
    total_training_steps_finetune = finetune_epochs * len(trainloader)
    step_in_finetune = 0

    for epoch in range(finetune_epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # ---- 与 train.py 相同的 mixup 策略 ----
            # 注意：这里 epoch 是全局 epoch（继续按 0~299 的规则算 mixup/aux）
            # 如果你希望继续沿用原来的 300 轮时间表，可以把 global_epoch 换成：
            # global_epoch = epoch + warmup_epochs + pruning_epochs
            global_epoch = epoch + warmup_epochs + pruning_epochs

            mixup_alpha = current_mixup_alpha(global_epoch)
            use_mixup = mixup_alpha > 0.0
            if use_mixup:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=mixup_alpha)
            else:
                targets_a = targets_b = None
                lam = 1.0

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                logits, aux_logits2, aux_logits1 = outputs
            else:
                logits = outputs
                aux_logits2 = aux_logits1 = None

            aux_weight = current_aux_weight(global_epoch)

            if use_mixup:
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                if aux_logits1 is not None and aux_logits2 is not None and aux_weight > 0:
                    loss += aux_weight * mixup_criterion(criterion, aux_logits1, targets_a, targets_b, lam)
                    loss += aux_weight * mixup_criterion(criterion, aux_logits2, targets_a, targets_b, lam)
            else:
                loss = criterion(logits, targets)
                if aux_logits1 is not None and aux_logits2 is not None and aux_weight > 0:
                    loss += aux_weight * criterion(aux_logits1, targets)
                    loss += aux_weight * criterion(aux_logits2, targets)

            optimizer_finetune.zero_grad()
            loss.backward()
            optimizer_finetune.step()
            lr_scheduler_finetune.step()

            train_loss += loss.item() * inputs.size(0)
            step_in_finetune += 1
            progress_bar_finetune.update(1)
            current_lr = optimizer_finetune.param_groups[0]["lr"]
            progress_bar_finetune.set_description(
                f"[阶段二] Epoch {epoch + 1 + warmup_epochs + pruning_epochs}/{num_train_epochs}, "
                f"Loss {loss.item():.4f}, LR {current_lr:.6f}"
            )

        # 每轮结束后评估（与 train.py 相同风格）
        model.eval()
        avg_train_loss = train_loss / len(trainset)
        acc1_epoch, acc5_epoch = check_accuracy(model, testloader)
        epoch_idx = epoch + 1 + warmup_epochs + pruning_epochs
        print(
            f"\n--- [阶段二] Epoch {epoch_idx}/{num_train_epochs} 测试:",
            f" 训练损失: {avg_train_loss:.4f}, Top-1 {acc1_epoch:.2f}%, Top-5 {acc5_epoch:.2f}%",
            file=sys.stdout,
            flush=True
        )

        if acc1_epoch > best_accuracy1:
            best_accuracy1 = acc1_epoch
            best_epoch = epoch_idx
            print(f"当前最佳 Top-1 准确率: {best_accuracy1:.2f}% (Epoch {best_epoch})")
            print(f"保存模型至: {compressed_model_path}")
            torch.save(model, compressed_model_path)

        model.train()

    # 微调结束后的评估部分保持不变
    print("\n剪枝与微调完成，载入最佳模型评估...")
    model = torch.load(compressed_model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    print("\n--- 计算最终模型指标 ---")
    final_metrics = optimizer_prune.compute_metrics()
    final_group_sparsity = getattr(final_metrics, "group_sparsity", 0.0)
    if final_metrics is not None:
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

    flops_after_m = oto.compute_flops(in_million=True)["total"]
    params_after_m = oto.compute_num_params(in_million=True)
    print(f"剪枝后 FLOPs: {flops_after_m:.2f} M")
    print(f"剪枝后参数量: {params_after_m:.2f} M")

    compressed_model_size_bytes = (
        os.stat(oto.compressed_model_path).st_size if os.path.exists(oto.compressed_model_path) else 0
    )
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
    final_group_sparsity = getattr(final_metrics, "group_sparsity", final_group_sparsity)

    if final_metrics is not None:
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
            print(
                f"FLOPs 压缩比: {final_metrics.flops_compression_ratio:.2f}x "
                f"({final_metrics.flops_before:.2f}M -> {final_metrics.flops_after:.2f}M)"
            )
        else:
            print(f"FLOPs 压缩比: N/A (剪枝前: {flops_before_m:.2f}M, 剪枝后: {flops_after_m:.2f}M)")

        if final_metrics.params_before > 0 and final_metrics.params_after > 0:
            print(
                f"参数量压缩比: {final_metrics.params_compression_ratio:.2f}x "
                f"({final_metrics.params_before:.2f}M -> {final_metrics.params_after:.2f}M)"
            )
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