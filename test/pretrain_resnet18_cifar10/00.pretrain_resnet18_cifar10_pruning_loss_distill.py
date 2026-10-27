import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Subset 
from tqdm.auto import tqdm
from transformers import get_scheduler
import numpy as np
import random

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from only_train_once import OTO
from only_train_once.optimizer.alpha_scheduler import ALPHA_CFG


def load_pretrained_resnet18(checkpoint_path, device):
    model = None
    if os.path.exists(checkpoint_path):
        print(f"尝试从 {checkpoint_path} 加载预训练模型...")
        try:
            loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if isinstance(loaded, torch.nn.Module):
                model = loaded
                print("成功加载完整模型。")
            elif isinstance(loaded, dict):
                base = resnet18()
                base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                base.maxpool = nn.Identity()
                base.fc = nn.Linear(base.fc.in_features, 10)
                state_dict = loaded.get('state_dict') or loaded.get('model_state_dict') or loaded
                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
                missing, unexpected = base.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"缺失参数: {missing}")
                if unexpected:
                    print(f"多余参数: {unexpected}")
                model = base
                print("已将权重加载到 torchvision ResNet18。")
            else:
                print("权重文件类型不支持，使用随机初始化模型。")
        except Exception as err:
            print(f"加载失败: {err}，使用随机初始化模型。")
    else:
        print(f"未找到预训练权重文件: {checkpoint_path}，使用随机初始化模型。")

    if model is None:
        from torchvision.models import resnet18 as tv_resnet18
        model = tv_resnet18()
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 10)
        print("已构建随机初始化的 ResNet18。")
    return model


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
            top5_correct += (top5 == targets.unsqueeze(1)).any(dim=1).sum().item()
            total += targets.size(0)

    if total == 0:
        return 0.0, 0.0
    return top1_correct / total * 100.0, top5_correct / total * 100.0


# === 知识蒸馏损失函数 ===
def kd_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    """
    蒸馏损失 = alpha * T*T * KL(soft_student || soft_teacher) + (1-alpha) * CE(hard)
    """
    soft_student = torch.nn.functional.log_softmax(student_logits / T, dim=1)
    soft_teacher = torch.nn.functional.softmax(teacher_logits / T, dim=1)
    distill_loss = torch.nn.functional.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)
    ce_loss = torch.nn.functional.cross_entropy(student_logits, labels)
    return alpha * distill_loss + (1 - alpha) * ce_loss

def main():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    base_dir = os.path.dirname(__file__)
    local_data_path = '/workspace/data/cifar10'
    output_cache_dir = os.path.join(base_dir, 'cache_resnet18_cifar10_pruned')
    os.makedirs(output_cache_dir, exist_ok=True)

    batch_size = 256
    num_workers = 48

    local_model_path = '/workspace/OTOv2/test/pretrain_resnet18_cifar10/model/pytorch_model.bin'
    model = load_pretrained_resnet18(local_model_path, device).to(device)
    model.eval()

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                             std=[0.2673, 0.2564, 0.2761])
    ])

    try:
        testset = CIFAR10(root=local_data_path, train=False, download=False, transform=transform_test)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=device.type == 'cuda')
        print(f"成功加载 CIFAR-10 测试集，包含 {len(testset)} 张图片。")

        trainset = CIFAR10(root=local_data_path, train=True, download=False, transform=transform_test)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=device.type == 'cuda')
        print(f"成功加载 CIFAR-10 训练集，包含 {len(trainset)} 张图片。")
    except Exception as exc:
        print(f"加载 CIFAR-10 数据集失败: {exc}")
        return
    
    # === α-验证子集（来自训练集，避免评估泄漏）===
    # 固定随机子集，保证复现；规模按需调整（建议 5k 左右足够稳定）
    n_alpha_val = 5000
    alpha_batch_size = batch_size      # 也可单独设置更小的 batch 以得到更多微批
    g = torch.Generator().manual_seed(1234)
    take = min(n_alpha_val, len(trainset))
    alpha_indices = torch.randperm(len(trainset), generator=g)[:take].tolist()

    alpha_subset = Subset(trainset, alpha_indices)
    alpha_loader = DataLoader(
        alpha_subset,
        batch_size=alpha_batch_size,
        shuffle=False,                 # 固定顺序，配合调度器里的“轮转指针”均匀覆盖
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )
    print(f"[α-Subset] 来自训练集的 α-更新子集：样本数={len(alpha_subset)}, batch_size={alpha_batch_size}")

    dummy_input = torch.rand(1, 3, 32, 32).to(device)
    oto = OTO(model=model, dummy_input=dummy_input)
    print("正在生成剪枝前的依赖图...")
    oto.visualize(view=False, out_dir=output_cache_dir)

    nodes_to_keep_unpruned = []
    if nodes_to_keep_unpruned:
        oto.mark_unprunable_by_node_ids(nodes_to_keep_unpruned)

    print("\n--- 剪枝前评估 ---")
    accuracy1_before, accuracy5_before = check_accuracy(model, testloader)
    print(f"剪枝前准确率: Top-1: {accuracy1_before:.2f}%, Top-5: {accuracy5_before:.2f}%")

    flops_before_m = oto.compute_flops(in_million=True)['total']
    params_before_m = oto.compute_num_params(in_million=True)
    print(f"剪枝前 FLOPs: {flops_before_m:.2f} M")
    print(f"剪枝前参数量: {params_before_m:.2f} M")

    original_model_size_bytes = os.stat(local_model_path).st_size if os.path.exists(local_model_path) else 0
    if original_model_size_bytes:
        print(f"原始模型文件大小: {original_model_size_bytes / (1024**2):.2f} MB")
    else:
        print("无法确定原始模型文件大小。")

    # === 缓存固定的验证微批（用于 α 更新；来自训练子集 alpha_loader）===
    def cache_val_batches(loader, n, device):
        val_batches = []
        it = iter(loader)
        for _ in range(n):
            try:
                vx, vy = next(it)
            except StopIteration:
                it = iter(loader); vx, vy = next(it)
            # 你当前未采用“自动搬设备”的改动，因此这里直接放到 device 上
            vx = vx.to(device, non_blocking=True)
            vy = vy.to(device, non_blocking=True)
            val_batches.append((vx, vy))
        return val_batches

    # 与调度器的 ALPHA_CFG.eval_batch_limit 保持一致最省心；若取不到则回退 64
    num_val_microbatches = getattr(ALPHA_CFG, "eval_batch_limit", 64)
    val_batches = cache_val_batches(alpha_loader, num_val_microbatches, device)
    print(f"α 更新缓存微批：共 {len(val_batches)} 个（与 ALPHA_CFG.eval_batch_limit 对齐）。")

    learning_rate = 6e-3
    finetune_lr = 5e-3
    num_train_epochs = 250
    warmup_epochs = 10
    pruning_epochs = 90
    pruning_periods = 90
    target_group_sparsity = 0.8
    finetune_epochs = num_train_epochs - warmup_epochs - pruning_epochs
    lr_scheduler_type = "cosine"

    num_update_steps_per_epoch = len(trainloader)
    warmup_steps = warmup_epochs * num_update_steps_per_epoch
    pruning_steps = pruning_epochs * num_update_steps_per_epoch
    pruning_and_warmup_steps = (warmup_epochs + pruning_epochs) * num_update_steps_per_epoch
    finetune_steps = finetune_epochs * num_update_steps_per_epoch
    start_pruning_step = warmup_steps

    criterion = nn.CrossEntropyLoss().to(device)
    distill_temperature = 4.0
    distill_alpha = 0.7

    # === 教师模型（原始预训练模型）===
    teacher_model = load_pretrained_resnet18(local_model_path, device).to(device)
    teacher_model.eval()
    print("教师模型加载完成，将用于阶段一和阶段二的知识蒸馏。")

    optimizer_prune = oto.hesso(
        variant='sgd',
        lr=learning_rate,
        target_group_sparsity=target_group_sparsity,
        start_pruning_step=start_pruning_step,
        pruning_periods=pruning_periods,
        pruning_steps=pruning_steps,
        device=device
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
        alpha_updates = 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
            outputs = model(inputs)
            loss = kd_loss(outputs, teacher_outputs, targets, T=distill_temperature, alpha=distill_alpha)
            loss.backward()
            # 同一批样本上，进入新 period 前更新 α
            if optimizer_prune.will_enter_new_period_on_next_step():
                optimizer_prune.alpha_scheduler.update(
                    model=model,
                    criterion=criterion,
                    inputs=val_batches,  # 注意：这里传入的是 list[(vx, vy), ...]
                    targets=None,        # 无需使用
                    t_in_period=0,      # 刚进入新 period 的第0小步
                )
                alpha_updates += 1
            optimizer_prune.step()
            lr_scheduler_prune.step()
            optimizer_prune.zero_grad()
            progress_bar_prune.update(1)
            current_lr = optimizer_prune.param_groups[0]['lr']
            progress_bar_prune.set_description(
                f"[阶段一] Epoch {epoch+1}, Loss: {loss.item():.4f}, "
                f"LR: {current_lr:.6f}"
            )

        print(f"\n--- [阶段一] Epoch {epoch+1}/{warmup_epochs + pruning_epochs} 评估 ---")
        print(dict(zip(optimizer_prune._alpha_keys, optimizer_prune.alpha_scheduler.get_alpha().tolist())))
        metrics = optimizer_prune.compute_metrics()
        print(f"\n[阶段一] Epoch {epoch+1}, Sparsity: {metrics.group_sparsity:.4f}")
        model.eval()
        accuracy1_epoch, accuracy5_epoch = check_accuracy(model, testloader)
        print(f"Epoch {epoch+1} 测试准确率: Top-1: {accuracy1_epoch:.2f}%, Top-5: {accuracy5_epoch:.2f}%")
        model.train()

    print("\n--- 剪枝阶段完成，模型结构已固定 ---")

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

    optimizer_finetune = torch.optim.SGD(
        model.parameters(),
        lr=finetune_lr,
        momentum=0.9,
        weight_decay=5e-4
    )

    lr_scheduler_finetune = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer_finetune,
        num_warmup_steps=0,
        num_training_steps=finetune_steps,
    )
    
    # === 教师模型（原始预训练模型）===
    teacher_model.eval()
    print("教师模型准备就绪，用于阶段二知识蒸馏。")

    print("\n--- 阶段二：微调 ---")
    best_accuracy1 = 0.0
    best_accuracy5 = 0.0
    best_epoch = 0
    progress_bar_finetune = tqdm(range(finetune_steps), desc="[阶段二] 微调中")
    model.train()
    for epoch in range(finetune_epochs):
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            # 1. 清零梯度
            optimizer_finetune.zero_grad()
            # 教师前向（no_grad）
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
            # 学生前向
            outputs = model(inputs)
            # 蒸馏损失
            # loss = kd_loss(outputs, teacher_outputs, targets, T=20.0, alpha=0.95)
            loss = kd_loss(outputs, teacher_outputs, targets, T=distill_temperature, alpha=distill_alpha)
            # loss = criterion(outputs, targets)
            loss.backward()
            optimizer_finetune.step()
            lr_scheduler_finetune.step()
            progress_bar_finetune.update(1)
            current_lr = optimizer_finetune.param_groups[0]['lr']
            progress_bar_finetune.set_description(
                f"[阶段二] Epoch {epoch+1+warmup_epochs+pruning_epochs}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}"
            )
        print(f"\n--- [阶段二] Epoch {epoch+1+warmup_epochs+pruning_epochs}/{num_train_epochs} 评估 ---")
        model.eval()
        accuracy1_epoch, accuracy5_epoch = check_accuracy(model, testloader)
        print(f"Epoch {epoch+1+warmup_epochs+pruning_epochs} 测试准确率: Top-1: {accuracy1_epoch:.2f}%, Top-5: {accuracy5_epoch:.2f}%")

        if accuracy1_epoch > best_accuracy1:
            best_accuracy1 = accuracy1_epoch
            best_epoch = epoch+1+warmup_epochs+pruning_epochs
            print(f"当前最佳Top-1准确率: {best_accuracy1:.2f}% (Epoch {best_epoch})")
            print(f"  - 保存模型到: {compressed_model_path}")
            torch.save(model, compressed_model_path)
        if accuracy1_epoch == best_accuracy1 and accuracy5_epoch > best_accuracy5:
            best_accuracy5 = accuracy5_epoch
            print(f"当前最佳Top-5准确率: {best_accuracy5:.2f}% (Epoch {best_epoch})")
            torch.save(model, compressed_model_path)
            print(f"当前最佳Top-1准确率: {best_accuracy1:.2f}% (Epoch {best_epoch})")
            print(f"  - 保存模型到: {compressed_model_path}")

        model.train()

    print("剪枝和微调过程完成。")
    print(f"\n--- 保存微调后的模型到: {compressed_model_path} ---")
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

    if original_model_size_bytes > 0 and compressed_model_size_bytes > 0:
        print(f"模型文件大小压缩比: {original_model_size_bytes / compressed_model_size_bytes:.2f}x "
              f"({original_model_size_bytes / (1024**2):.2f}MB -> {compressed_model_size_bytes / (1024**2):.2f}MB)")
    elif original_model_size_bytes > 0:
        print(f"原始模型文件大小: {original_model_size_bytes / (1024**2):.2f} MB (压缩后模型大小未知)")
    elif compressed_model_size_bytes > 0:
        print(f"压缩模型文件大小: {compressed_model_size_bytes / (1024**2):.2f} MB (原始模型大小未知)")
    else:
        print("模型文件大小压缩比: N/A (无法获取原始或压缩后的模型大小)")

    print(f"\n准确率变化: Top-1 从 {accuracy1_before:.2f}% 变为 {accuracy1_after:.2f}% (变化: {accuracy1_after - accuracy1_before:+.2f}%)")
    print(f"准确率变化: Top-5 从 {accuracy5_before:.2f}% 变为 {accuracy5_after:.2f}% (变化: {accuracy5_after - accuracy5_before:+.2f}%)")

    print("\n--- 第 7 步：导出压缩模型 ---")
    print(f"压缩后的模型已保存在: {oto.compressed_model_path}")
    print(f"完整的组稀疏模型 (未移除零权重组) 保存在: {oto.full_group_sparse_model_path}")
    print("\n剪枝和压缩测试完成。")


if __name__ == "__main__":
    main()
