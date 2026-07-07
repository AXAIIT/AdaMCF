import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler
import numpy as np
import random
from thop import profile
from pathlib import Path
import time
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

DATA_MODULE_ROOT = Path("/workspace/data")
if str(DATA_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_MODULE_ROOT))

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from only_train_once import OTO
from model.model import resnet110
from only_train_once.optimizer.alpha_scheduler import ALPHA_CFG, prepare_alpha_eval_batches


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


def compute_flops_with_thop(model, input_size=(1, 3, 32, 32)):
    """
    使用 thop 计算模型 FLOPs 和参数量。
    返回:
        flops_m: MFLOPs
        params_m: M 参数量
    """
    import copy

    model_cpu = copy.deepcopy(model).to("cpu").eval()
    dummy_input = torch.randn(*input_size)

    with torch.no_grad():
        macs, params = profile(model_cpu, inputs=(dummy_input,), verbose=False)

    flops_m = macs * 2 / 1e6        # MACs -> FLOPs
    params_m = params / 1e6         # 参数量 -> M
    return flops_m, params_m


def prune_model(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    base_dir = os.path.dirname(__file__)
    local_data_path = '/workspace/data/cifar10'
    output_cache_dir = os.path.join(base_dir, 'cache_resnet110_cifar10_pruned_0.75')
    os.makedirs(output_cache_dir, exist_ok=True)

    batch_size = 128
    num_workers = 16

    local_model_path = '/workspace/OTOv2_v2/test/pretrain_resnet110_cifar10/model/resnet110-1d1ed7c2.th'
    print(f"加载预训练 ResNet110 权重: {local_model_path}")
    model = resnet110()
    ckpt = torch.load(local_model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

    trainset = CIFAR10(root=local_data_path, train=True, download=False, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=device.type == 'cuda')
    print(f"成功加载 CIFAR-10 训练集，包含 {len(trainset)} 张图片。")

    testset = CIFAR10(root=local_data_path, train=False, download=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == 'cuda')
    print(f"成功加载 CIFAR-10 测试集，包含 {len(testset)} 张图片。")

    # ===== α 子集 =====
    val_batches, alpha_info = prepare_alpha_eval_batches(
        trainset=trainset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print(
        f"[α-Subset] 来自训练集的 α-更新子集：样本数={alpha_info['n_alpha_val']}, "
        f"batch_size={alpha_info['alpha_batch_size']}"
    )
    print(
        f"α 更新缓存微批：共 {alpha_info['num_val_microbatches']} 个"
        f"（与 ALPHA_CFG.eval_batch_limit 对齐）。"
    )

    dummy_input = torch.rand(1, 3, 32, 32).to(device)
    oto = OTO(model=model, dummy_input=dummy_input)
    print("正在生成剪枝前的依赖图...")
    oto.visualize(view=False, out_dir=output_cache_dir)

    nodes_to_keep_unpruned = []
    oto.mark_unprunable_by_node_ids(nodes_to_keep_unpruned)

    print("\n--- 剪枝前评估 ---")
    accuracy1_before, accuracy5_before = check_accuracy(model, testloader)
    print(f"剪枝前准确率: Top-1: {accuracy1_before:.2f}%, Top-5: {accuracy5_before:.2f}%")

    flops_before_m, params_before_m = compute_flops_with_thop(model, input_size=(1, 3, 32, 32))
    print(f"剪枝前 FLOPs: {flops_before_m:.2f} M")
    print(f"剪枝前参数量: {params_before_m:.2f} M")

    learning_rate = 5e-3
    warmup_epochs = 10
    pruning_epochs = 200
    pruning_periods = 100
    target_group_sparsity = 0.75
    lr_scheduler_type = "cosine"

    num_update_steps_per_epoch = len(trainloader)
    warmup_steps = warmup_epochs * num_update_steps_per_epoch
    pruning_steps = pruning_epochs * num_update_steps_per_epoch
    total_epochs = warmup_epochs + pruning_epochs
    pruning_and_warmup_steps = (warmup_epochs + pruning_epochs) * num_update_steps_per_epoch
    start_pruning_step = warmup_steps

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer_prune = oto.hesso(
        variant='sgd',
        lr=learning_rate,
        first_momentum=0.9,
        second_momentum=0.0,
        dampening=0.0,
        weight_decay=5e-4, 
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
    model.train()
    for epoch in range(warmup_epochs + pruning_epochs):
        print(f"\n[阶段一] Epoch {epoch+1}/{total_epochs} 正在剪枝...")
        epoch_loss = 0.0
        epoch_lr = 0.0
        epoch_steps = 0
        t0 = time.time()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if optimizer_prune.will_enter_new_period_on_next_step():
                optimizer_prune.alpha_scheduler.update(
                    model=model,
                    criterion=criterion,
                    inputs=val_batches,
                    targets=None,
                    t_in_period=0,
                )
            optimizer_prune.step()
            lr_scheduler_prune.step()
            optimizer_prune.zero_grad()
            cur_lr = optimizer_prune.param_groups[0]['lr']
            epoch_loss += loss.item()
            epoch_lr += cur_lr
            epoch_steps += 1

        dt = max(time.time() - t0, 1e-8)
        avg_loss = epoch_loss / epoch_steps
        avg_lr = epoch_lr / epoch_steps
        it_per_sec = epoch_steps / dt
        eta = (total_epochs - epoch - 1) * dt
        eta_h = int(eta // 3600)
        eta_m = int((eta % 3600) // 60)
        print(f"[阶段一] Epoch {epoch+1}/{total_epochs} 评估\n"
              f"平均Loss: {avg_loss:.4f}, 平均LR: {avg_lr:.6f}, "
              f"吞吐: {it_per_sec:.2f} it/s, 用时: {dt:.2f}s, "
              f"估计剩余: {eta_h}h:{eta_m}m")

        metrics = optimizer_prune.compute_metrics()
        print(f"Sparsity: {metrics.group_sparsity:.4f}")
        print(dict(zip(optimizer_prune._alpha_keys,
                       optimizer_prune.alpha_scheduler.get_alpha().tolist())))

        model.eval()
        acc1_ep, acc5_ep = check_accuracy(model, testloader)
        print(f"Epoch {epoch+1} 测试准确率: Top-1: {acc1_ep:.2f}%, Top-5: {acc5_ep:.2f}%")
        model.train()

    print("\n--- 构建并加载子网络用于微调 ---")
    oto.construct_subnet(out_dir=output_cache_dir)
    compressed_model_path = oto.compressed_model_path
    compressed_model = torch.load(compressed_model_path, map_location=device, weights_only=False)
    compressed_model.to(device)
    compressed_model.eval()

    accuracy1_after, accuracy5_after = check_accuracy(compressed_model, testloader)
    print(f"剪枝后准确率: Top-1: {accuracy1_after:.2f}%, Top-5: {accuracy5_after:.2f}%")

    flops_after_m, params_after_m = compute_flops_with_thop(compressed_model, input_size=(1, 3, 32, 32))
    print(f"剪枝前 FLOPs(thop): {flops_after_m:.2f} M")
    print(f"剪枝前参数量: {params_after_m:.2f} M")

    final_metrics = optimizer_prune.compute_metrics(
        flops_before=flops_before_m,
        flops_after=flops_after_m,
        params_before=params_before_m,
        params_after=params_after_m,
    )
    final_group_sparsity = final_metrics.group_sparsity
    pruned_flops_m = final_metrics.pruned_flops
    pruned_params_m = final_metrics.pruned_params
    pruned_flops_percent = final_metrics.pruned_flops_percent
    pruned_params_percent = final_metrics.pruned_params_percent

    print("\n--- 第 6 步：压缩总结 ---")
    print(f"最终剪枝率 (组稀疏率): {final_group_sparsity:.2f} ({final_group_sparsity * 100:.2f}%)")
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

    prune_results = {
        "seed": seed,
        "accuracy1_before": accuracy1_before,
        "accuracy5_before": accuracy5_before,
        "accuracy1_after": accuracy1_after,
        "accuracy5_after": accuracy5_after,
        "flops_before_m": flops_before_m,
        "flops_after_m": flops_after_m,
        "params_before_m": params_before_m,
        "params_after_m": params_after_m,
        "final_group_sparsity": final_group_sparsity,
        "pruned_flops_m": pruned_flops_m,
        "pruned_flops_percent": pruned_flops_percent,
        "pruned_params_m": pruned_params_m,
        "pruned_params_percent": pruned_params_percent,
        "flops_compression_ratio": final_metrics.flops_compression_ratio
            if final_metrics.flops_before > 0 and final_metrics.flops_after > 0 else None,
        "params_compression_ratio": final_metrics.params_compression_ratio
            if final_metrics.params_before > 0 and final_metrics.params_after > 0 else None,         
        "compressed_model_path": oto.compressed_model_path,
    }
    return prune_results


def finetune_model(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== 阶段二：微调 ===")
    print(f"使用设备: {device}")

    base_dir = CURRENT_FILE.parent
    local_data_path = "/workspace/data/cifar10"

    pruned_cache_dir = base_dir / "cache_resnet110_cifar10_pruned_0.75"
    compressed_model_path = None
    if pruned_cache_dir.exists():
        for f in os.listdir(pruned_cache_dir):
            if f.endswith("_compressed.pt"):
                compressed_model_path = str(pruned_cache_dir / f)
                break

    if not compressed_model_path or not os.path.exists(compressed_model_path):
        print("错误：未找到压缩模型，请先运行剪枝阶段。")
        return

    print(f"加载压缩模型: {compressed_model_path}")
    model = torch.load(compressed_model_path, map_location=device, weights_only=False)
    model.to(device)

    batch_size = 128
    num_workers = 16

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

    trainset = CIFAR10(root=local_data_path, train=True, download=False, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=device.type == 'cuda')
    print(f"成功加载 CIFAR-10 训练集，包含 {len(trainset)} 张图片。")

    testset = CIFAR10(root=local_data_path, train=False, download=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == 'cuda')
    print(f"成功加载 CIFAR-10 测试集，包含 {len(testset)} 张图片。")

    finetune_lr = 5e-2
    num_train_epochs = 300
    warmup_epochs = 0
    total_epochs = num_train_epochs
    steps_per_epoch = len(trainloader)
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    lr_scheduler_type = "cosine"

    finetuned_model_dir = os.path.join(base_dir, 'finetuned_resnet110_sparsity_0_75')
    os.makedirs(finetuned_model_dir, exist_ok=True)
    finetuned_model_path = os.path.join(finetuned_model_dir, 'resnet110_cifar10_finetuned.pt')

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=finetune_lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss().to(device)

    best_top1 = 0.0
    best_top5 = 0.0
    best_epoch = 0

    model.train()
    for epoch in range(num_train_epochs):
        in_warmup = epoch < warmup_epochs
        phase = "[阶段二-预热]" if in_warmup else "[阶段二-微调]"
        print(f"\n{phase} Epoch {epoch+1}/{total_epochs} 正在训练...")

        epoch_loss = 0.0
        epoch_lr = 0.0
        epoch_steps = 0
        t0 = time.time()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()
            cur_lr = optimizer.param_groups[0]['lr']
            epoch_loss += loss.item()
            epoch_lr += cur_lr
            epoch_steps += 1

        dt = max(time.time() - t0, 1e-8)
        avg_loss = epoch_loss / epoch_steps
        avg_lr = epoch_lr / epoch_steps
        it_per_sec = epoch_steps / dt
        eta = (total_epochs - epoch - 1) * dt
        eta_h = int(eta // 3600)
        eta_m = int((eta % 3600) // 60)
        print(f"{phase} Epoch {epoch+1}/{total_epochs} 评估\n"
              f"平均Loss: {avg_loss:.4f}, 平均LR: {avg_lr:.6f}, "
              f"吞吐: {it_per_sec:.2f} it/s, 用时: {dt:.2f}s, "
              f"估计剩余: {eta_h}h:{eta_m}m")

        model.eval()
        accuracy1_after, accuracy5_after = check_accuracy(model, testloader)
        print(f"Epoch {epoch+1} 测试准确率: Top-1: {accuracy1_after:.2f}%, Top-5: {accuracy5_after:.2f}%")

        if accuracy1_after > best_top1 or (accuracy1_after == best_top1 and accuracy5_after > best_top5):
            best_top1 = accuracy1_after
            best_top5 = accuracy5_after
            best_epoch = epoch + 1
            print(f"当前最佳Top-1准确率: {best_top1:.2f}% (Epoch {best_epoch})")
            print(f"  - 保存模型到: {finetuned_model_path}")
            torch.save(model, finetuned_model_path)

        model.train()

    print("微调过程完成。")
    print(f"最佳模型已保存在: {finetuned_model_path}")

    model = torch.load(finetuned_model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    top1_final, top5_final = check_accuracy(model, testloader)
    print(f"微调后最终准确率: Top-1: {top1_final:.2f}%, Top-5: {top5_final:.2f}%")

    baseline_model_path = '/workspace/OTOv2_v2/test/pretrain_resnet110_cifar10/model/resnet110-1d1ed7c2.th'
    baseline_model = resnet110()
    ckpt = torch.load(baseline_model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    baseline_model.load_state_dict(state_dict, strict=True)

    baseline_model.to(device)
    baseline_model.eval()
    baseline_top1, baseline_top5 = check_accuracy(baseline_model, testloader)
    print(f"未剪枝模型准确率: Top-1: {baseline_top1:.2f}%, Top-5: {baseline_top5:.2f}%")
    print(f"相较未剪枝模型: Top-1 变化 {top1_final - baseline_top1:.2f}%, "f"Top-5 变化 {top5_final - baseline_top5:.2f}%")

    finetune_results = {
        "seed": seed,
        "finetune_best_epoch": best_epoch,
        "finetune_top1_final": accuracy1_after,
        "finetune_top5_final": accuracy5_after,
        "baseline_top1": baseline_top1 if os.path.exists(baseline_model_path) else None,
        "baseline_top5": baseline_top5 if os.path.exists(baseline_model_path) else None,
        "delta_top1_vs_baseline": (accuracy1_after - baseline_top1) if os.path.exists(baseline_model_path) else None,
        "delta_top5_vs_baseline": (accuracy1_after - baseline_top5) if os.path.exists(baseline_model_path) else None,
        "finetuned_model_path": finetuned_model_path,
    }
    return finetune_results


def main():
    prune_model()
    finetune_model()


if __name__ == "__main__":
    main()
