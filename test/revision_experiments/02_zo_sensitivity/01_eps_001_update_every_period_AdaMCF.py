import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import numpy as np
import random
import importlib
from thop import profile
from pathlib import Path
import time


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

DATA_MODULE_ROOT = Path("/workspace/data")
if str(DATA_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_MODULE_ROOT))

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESNET110_MODULE_ROOT = PROJECT_ROOT / 'test' / 'Ablation' / 'resnet110_cifar100'
if str(RESNET110_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(RESNET110_MODULE_ROOT))

from only_train_once import OTO
from only_train_once.optimizer.alpha_scheduler import ALPHA_CFG, prepare_alpha_eval_batches

resnet110 = importlib.import_module("model.model").resnet110


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _setup_file_logger():
    log_path = CURRENT_FILE.with_suffix(".log")
    log_fp = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_fp)
    sys.stderr = _Tee(sys.stderr, log_fp)
    print(f"[Log] writing to: {log_path}")
    return log_fp


EXP_NAME = "01_eps_001_update_every_period_AdaMCF"


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
    import copy
    model_cpu = copy.deepcopy(model).to("cpu").eval()
    dummy_input = torch.randn(*input_size)
    with torch.no_grad():
        macs, params = profile(model_cpu, inputs=(dummy_input,), verbose=False)
    flops_m = macs * 2 / 1e6
    params_m = params / 1e6
    return flops_m, params_m

def prune_model(seed: int = 42):
    base_dir = os.path.dirname(__file__)
    data_root = '/workspace/data/cifar100'
    pretrained_ckpt_path = '/workspace/OTOv2_v2/test/Ablation/resnet110_cifar100/model/best_model.pth'
    output_cache_dir = os.path.join(base_dir, f'cache_resnet110_cifar100_pruned_{EXP_NAME}_0.5')
    os.makedirs(output_cache_dir, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage 1: Pruning ResNet-110 on CIFAR-100 ===")
    print(f"Device: {device}")
    print(f"[Experiment] {EXP_NAME}")

    # --- ZO Config ---
    alpha_scheduler_cfg = {
        "eps": 0.01,
        "update_stride": 1,
    }
    M = 5
    print(
        f"[ZO Config] eps={alpha_scheduler_cfg['eps']}, "
        f"update_stride={alpha_scheduler_cfg['update_stride']}, "
        f"eval_per_update={2 * M}"
    )

    # --- 2. 加载数据 ---
    batch_size = 128
    num_workers = 16

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = CIFAR100(root=data_root, train=True, download=False, transform=transform_train)
    test_set = CIFAR100(root=data_root, train=False, download=False, transform=transform_test)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    val_batches, alpha_info = prepare_alpha_eval_batches(
        trainset=train_set, device=device, batch_size=batch_size, num_workers=num_workers
    )
    print(
        f"[α-Subset] 来自训练集的 α-更新子集：样本数={alpha_info['n_alpha_val']}, "
        f"batch_size={alpha_info['alpha_batch_size']}"
    )
    print(
        f"α 更新缓存微批：共 {alpha_info['num_val_microbatches']} 个"
        f"（与 ALPHA_CFG.eval_batch_limit 对齐）。"
    )

    print(f"Loading pretrained weights from: {pretrained_ckpt_path}")
    model = resnet110(num_classes=100)
    
    ckpt = torch.load(pretrained_ckpt_path, map_location=device)
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    print("\n--- 剪枝前评估 ---")
    accuracy1_before, accuracy5_before = check_accuracy(model, test_loader)
    print(f"剪枝前准确率: Top-1: {accuracy1_before:.2f}%, Top-5: {accuracy5_before:.2f}%")

    flops_before_m, params_before_m = compute_flops_with_thop(model, input_size=(1, 3, 32, 32))
    print(f"剪枝前 FLOPs(thop): {flops_before_m:.2f} M")
    print(f"剪枝前参数量: {params_before_m:.2f} M")

    dummy_input = torch.rand(1, 3, 32, 32).to(device)
    oto = OTO(model=model, dummy_input=dummy_input)
    oto.visualize(view=False, out_dir=output_cache_dir)

    # --- 4. 配置 AdaMCF 优化器 ---
    lr = 0.01
    target_sparsity = 0.5
    
    warmup_epochs = 5
    pruning_epochs = 100
    total_epochs = warmup_epochs + pruning_epochs
    
    steps_per_epoch = len(train_loader)
    total_steps = total_epochs * steps_per_epoch
    start_pruning_step = warmup_epochs * steps_per_epoch
    pruning_steps = pruning_epochs * steps_per_epoch
    schedule_mode = 'cosine_front'

    optimizer_prune = oto.hesso(
        variant='sgd',
        lr=lr,
        first_momentum=0.9,
        second_momentum=0.0,
        dampening=0.0,
        weight_decay=5e-4,
        target_group_sparsity=target_sparsity,
        start_pruning_step=start_pruning_step,
        pruning_periods=pruning_epochs,
        pruning_steps=pruning_steps,
        pruning_quota_mode=schedule_mode,
        alpha_scheduler_cfg=alpha_scheduler_cfg,
        device=device,
    )
    print(f"[Schedule] mode={schedule_mode}")
    print(f"[Schedule] period quotas={optimizer_prune.active_num_redundant_groups}")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_prune, 
        T_max=total_epochs)

    print("\n--- 阶段一：预热与剪枝 ---")
    criterion = nn.CrossEntropyLoss().to(device)
    model.train()
    
    for epoch in range(total_epochs):
        print(f"\n[阶段一] Epoch {epoch+1}/{total_epochs} 正在剪枝...")
        epoch_loss = 0.0
        epoch_lr = 0.0
        epoch_steps = 0
        t0 = time.time()
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer_prune.zero_grad()
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
            cur_lr = optimizer_prune.param_groups[0]['lr']
            epoch_loss += loss.item()
            epoch_lr += cur_lr
            epoch_steps += 1
        
        scheduler.step()
        
        
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
        acc1_ep, acc5_ep = check_accuracy(model, test_loader)
        print(f"Epoch {epoch+1} 测试准确率: Top-1: {acc1_ep:.2f}%, Top-5: {acc5_ep:.2f}%")
        model.train()

    print("\n--- 构建并加载子网络用于微调 ---")
    oto.construct_subnet(out_dir=output_cache_dir)
    compressed_model_path = oto.compressed_model_path
    compressed_model = torch.load(compressed_model_path, map_location=device, weights_only=False)
    compressed_model.to(device)
    compressed_model.eval()

    accuracy1_after, accuracy5_after = check_accuracy(compressed_model, test_loader)
    print(f"剪枝后准确率: Top-1: {accuracy1_after:.2f}%, Top-5: {accuracy5_after:.2f}%")

    flops_after_m, params_after_m = compute_flops_with_thop(compressed_model, input_size=(1, 3, 32, 32))
    print(f"剪枝后 FLOPs(thop): {flops_after_m:.2f} M")
    print(f"剪枝后参数量: {params_after_m:.2f} M")

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


def finetune_model(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== 阶段二：微调 ===")
    print(f"使用设备: {device}")

    base_dir = CURRENT_FILE.parent
    local_data_path = "/workspace/data/cifar100"

    pruned_cache_dir = base_dir / f"cache_resnet110_cifar100_pruned_{EXP_NAME}_0.5"
    compressed_model_path = None

    finetuned_model_dir = os.path.join(base_dir, f"finetuned_{EXP_NAME}_0_5")
    os.makedirs(finetuned_model_dir, exist_ok=True)
    finetuned_model_path = os.path.join(
        finetuned_model_dir,
        f"resnet110_cifar100_finetuned_{EXP_NAME}.pt"
    )

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

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = CIFAR100(root=local_data_path, train=True, download=False, transform=transform_train)
    test_set = CIFAR100(root=local_data_path, train=False, download=False, transform=transform_test)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    epochs = 200
    lr = 0.01
    
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=lr, 
        momentum=0.9, 
        weight_decay=5e-4, 
        nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    best_top1 = 0.0
    best_top5 = 0.0
    best_epoch = 0
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_lr = 0.0
        epoch_steps = 0
        t0 = time.time()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        epoch_loss += loss.item()
        cur_lr = optimizer.param_groups[0]['lr']
        epoch_loss += loss.item()
        epoch_lr += cur_lr
        epoch_steps += 1

        dt = max(time.time() - t0, 1e-8)
        avg_loss = epoch_loss / epoch_steps
        avg_lr = epoch_lr / epoch_steps
        it_per_sec = epoch_steps / dt
        eta = (epochs - epoch - 1) * dt
        eta_h = int(eta // 3600)
        eta_m = int((eta % 3600) // 60)
        print(f"Epoch {epoch+1}/{epochs} 评估\n"
              f"平均Loss: {avg_loss:.4f}, 平均LR: {avg_lr:.6f}, "
              f"吞吐: {it_per_sec:.2f} it/s, 用时: {dt:.2f}s, "
              f"估计剩余: {eta_h}h:{eta_m}m")

        model.eval()
        accuracy1_after, accuracy5_after = check_accuracy(model, test_loader)
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
    top1_final, top5_final = check_accuracy(model, test_loader)
    print(f"微调后最终准确率: Top-1: {top1_final:.2f}%, Top-5: {top5_final:.2f}%")

    baseline_model_path = '/workspace/OTOv2_v2/test/Ablation/resnet110_cifar100/model/best_model.pth'
    baseline_model = resnet110()
    ckpt = torch.load(baseline_model_path, map_location=device)
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    baseline_model.load_state_dict(state_dict, strict=True)
    
    model.to(device)
    baseline_model.eval()
    baseline_top1, baseline_top5 = check_accuracy(baseline_model, test_loader)
    print(f"未剪枝模型准确率: Top-1: {baseline_top1:.2f}%, Top-5: {baseline_top5:.2f}%")
    print(f"相较未剪枝模型: Top-1 变化 {top1_final - baseline_top1:.2f}%, "f"Top-5 变化 {top5_final - baseline_top5:.2f}%")


if __name__ == "__main__":
    _log_fp = _setup_file_logger()
    try:
        prune_model()
        finetune_model()
    finally:
        _log_fp.close()