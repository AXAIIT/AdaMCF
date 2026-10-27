import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import numpy as np
import random
from pathlib import Path
import time


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
REFERENCE_EXPERIMENT_ROOT = (
    PROJECT_ROOT / "test" / "Ablation" / "resnet110_cifar100"
)
for path in (PROJECT_ROOT, REFERENCE_EXPERIMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from only_train_once import OTO
from model.model import resnet110 
from only_train_once.optimizer.alpha_scheduler import ALPHA_CFG, prepare_alpha_eval_batches
from motivation_probes import (
    SpatialPeriodTracker,
    TemporalPeriodTracker,
    run_spatial_probe,
)


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


def evaluate_loss_accuracy(model, loader, criterion):
    """Return deterministic full-loader loss/Top-1/Top-5 and restore mode."""
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss_sum += float(criterion(outputs, targets).item()) * targets.size(0)
            top1 = outputs.argmax(dim=1)
            top5 = outputs.topk(5, dim=1).indices
            top1_correct += int((top1 == targets).sum().item())
            top5_correct += int(
                (top5 == targets.unsqueeze(1)).any(dim=1).sum().item()
            )
            total += int(targets.size(0))
    model.train(was_training)
    denominator = max(total, 1)
    return (
        loss_sum / denominator,
        100.0 * top1_correct / denominator,
        100.0 * top5_correct / denominator,
    )

def compute_flops_with_thop(model, input_size=(1, 3, 32, 32)):
    import copy
    from thop import profile
    model_cpu = copy.deepcopy(model).to("cpu").eval()
    dummy_input = torch.randn(*input_size)
    with torch.no_grad():
        macs, params = profile(model_cpu, inputs=(dummy_input,), verbose=False)
    flops_m = macs * 2 / 1e6
    params_m = params / 1e6
    return flops_m, params_m

def prune_model(
    seed: int = 42,
    track_spatial_periods: bool = False,
    track_temporal_periods: bool = False,
    production_schedule_mode: str | None = None,
):
    seed = int(os.environ.get('MOTIVATION_SEED', str(seed)))
    if production_schedule_mode is None:
        production_schedule_mode = os.environ.get(
            'MOTIVATION_PRODUCTION_SCHEDULE', 'cosine_front'
        )
    schedule_aliases = {
        'cosine': 'cosine_front',
        'cosine_front': 'cosine_front',
        'uniform': 'uniform',
        'balanced_uniform': 'uniform',
    }
    requested_schedule = str(production_schedule_mode).strip().lower()
    if requested_schedule not in schedule_aliases:
        raise ValueError(
            "production_schedule_mode must be cosine_front or uniform"
        )
    production_schedule_mode = schedule_aliases[requested_schedule]
    base_dir = os.path.dirname(__file__)
    data_root = os.environ.get('MOTIVATION_DATA_ROOT', '/workspace/data/cifar100')
    pretrained_ckpt_path = os.environ.get(
        'MOTIVATION_PRETRAINED_CKPT',
        str(REFERENCE_EXPERIMENT_ROOT / 'model' / 'best_model.pth'),
    )
    cache_name = (
        f'cache_resnet110_cifar100_pruned_{production_schedule_mode}_0.5'
        if track_temporal_periods
        else 'cache_resnet110_cifar100_pruned_AdaMCF_Full_0.5'
    )
    output_cache_dir = os.path.join(base_dir, cache_name)
    os.makedirs(output_cache_dir, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage 1: Pruning ResNet-110 on CIFAR-100 ===")
    print(f"Device: {device}")

    # --- 2. 加载数据 ---
    batch_size = int(os.environ.get('MOTIVATION_BATCH_SIZE', '128'))
    num_workers = int(os.environ.get('MOTIVATION_NUM_WORKERS', '16'))

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
    
    warmup_epochs = 20
    pruning_epochs = 100
    total_epochs = warmup_epochs + pruning_epochs
    
    steps_per_epoch = len(train_loader)
    start_pruning_step = warmup_epochs * steps_per_epoch
    pruning_steps = pruning_epochs * steps_per_epoch

    optimizer_prune = oto.hesso(
        variant='sgd',
        lr=lr,
        first_momentum=0.9,
        second_momentum=0.0,
        dampening=0.0,
        weight_decay=1e-4,
        target_group_sparsity=target_sparsity,
        start_pruning_step=start_pruning_step,
        pruning_periods=pruning_epochs,
        pruning_steps=pruning_steps,
        device=device,
        seed=seed,
        pruning_schedule_mode=production_schedule_mode,
    )

    # The formal experiment accepts only a real end-to-end parameter pass-through:
    # entry point -> OTO.hesso -> HESSO.  Validate silently so HESSO remains the
    # single source of schedule output; never overwrite a mismatched backend.
    backend_schedule_mode = str(
        getattr(optimizer_prune, "pruning_schedule_mode", "")
    ).strip().lower()
    if backend_schedule_mode != production_schedule_mode:
        raise RuntimeError(
            "pruning_schedule_mode was not propagated from OTO.hesso to HESSO: "
            f"requested={production_schedule_mode!r}, "
            f"backend={backend_schedule_mode!r}. Sync only_train_once/__init__.py "
            "and only_train_once/optimizer/hesso.py."
        )
    actual_production_quota = [
        int(value) for value in optimizer_prune.active_num_redundant_groups
    ]
    if not (
        len(actual_production_quota) == optimizer_prune.pruning_periods
        and sum(actual_production_quota)
        == optimizer_prune.target_num_redundant_groups
    ):
        raise RuntimeError("HESSO constructed an invalid production schedule.")

    spatial_tracker = None
    if track_spatial_periods:
        spatial_tracker = SpatialPeriodTracker(
            pruning_fraction=float(
                os.environ.get("MOTIVATION_SPATIAL_PERIOD_FRACTION", "0.30")
            ),
            seed=seed,
        )

    temporal_tracker = None
    if track_temporal_periods:
        temporal_tracker = TemporalPeriodTracker(
            data_root=Path(data_root),
            probe_fraction=float(
                os.environ.get("MOTIVATION_TEMPORAL_PERIOD_FRACTION", "0.05")
            ),
            strategy=os.environ.get(
                "MOTIVATION_TEMPORAL_PERIOD_STRATEGY", "static_fusion"
            ),
            eval_subset=int(
                os.environ.get("MOTIVATION_TEMPORAL_EVAL_SUBSET", "10000")
            ),
            batch_size=batch_size,
            num_workers=int(
                os.environ.get("MOTIVATION_TEMPORAL_NUM_WORKERS", "4")
            ),
            eval_seed=int(
                os.environ.get("MOTIVATION_TEMPORAL_EVAL_SEED", "2026")
            ),
            warmup_epochs=warmup_epochs,
            pruning_epochs=pruning_epochs,
            production_schedule_mode=production_schedule_mode,
            training_config={
                "seed": seed,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cuda_runtime_version": torch.version.cuda,
                "device": str(device),
                "cuda_device_name": (
                    torch.cuda.get_device_name(torch.cuda.current_device())
                    if torch.cuda.is_available()
                    else None
                ),
                "data_root": str(data_root),
                "pretrained_checkpoint": str(pretrained_ckpt_path),
                "batch_size": batch_size,
                "main_num_workers": num_workers,
                "learning_rate": lr,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "target_group_sparsity": target_sparsity,
                "production_schedule_mode": production_schedule_mode,
                "production_period_quota": list(actual_production_quota),
                "total_pruning_budget": int(sum(actual_production_quota)),
                "steps_per_epoch": steps_per_epoch,
                "start_pruning_step": start_pruning_step,
                "pruning_steps": pruning_steps,
                "lr_scheduler": "CosineAnnealingLR",
                "lr_scheduler_t_max": total_epochs,
                "alpha_config": dict(vars(ALPHA_CFG)),
                "alpha_subset": dict(alpha_info),
            },
        )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_prune, 
        T_max=total_epochs
        )

    print("\n--- 阶段一：预热与剪枝 ---")
    criterion = nn.CrossEntropyLoss().to(device)
    model.train()
    training_history = []
    
    for epoch in range(total_epochs):
        print(f"\n[阶段一] Epoch {epoch+1}/{total_epochs} 正在剪枝...")
        epoch_loss = 0.0
        epoch_lr = 0.0
        epoch_steps = 0
        epoch_examples = 0
        epoch_top1_correct = 0
        epoch_top5_correct = 0
        epoch_first_lr = None
        epoch_last_lr = None
        t0 = time.time()
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer_prune.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            top1 = outputs.argmax(dim=1)
            top5 = outputs.topk(5, dim=1).indices
            epoch_top1_correct += int((top1 == targets).sum().item())
            epoch_top5_correct += int(
                (top5 == targets.unsqueeze(1)).any(dim=1).sum().item()
            )
            epoch_examples += int(targets.size(0))
            loss.backward()
            entering_period = optimizer_prune.will_enter_new_period_on_next_step()
            is_real_pruning_period = False
            if entering_period:
                optimizer_prune.alpha_scheduler.update(
                    model=model,
                    criterion=criterion,
                    inputs=val_batches,
                    targets=None,
                    t_in_period=0,
                )
                next_step = optimizer_prune.num_steps + 1
                is_real_pruning_period = (
                    next_step >= optimizer_prune.start_pruning_step
                    and optimizer_prune.curr_pruning_period
                    < optimizer_prune.pruning_periods
                )
                if spatial_tracker is not None and is_real_pruning_period:
                    spatial_tracker.capture(
                        optimizer_prune, epoch=epoch + 1, model=model
                    )
                if temporal_tracker is not None and is_real_pruning_period:
                    temporal_tracker.capture(
                        optimizer_prune, model=model, epoch=epoch + 1
                    )
            optimizer_prune.step()
            if temporal_tracker is not None and is_real_pruning_period:
                temporal_tracker.record_production_step(optimizer_prune)
            cur_lr = optimizer_prune.param_groups[0]['lr']
            if epoch_first_lr is None:
                epoch_first_lr = float(cur_lr)
            epoch_last_lr = float(cur_lr)
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
        alpha_payload = dict(zip(
            optimizer_prune._alpha_keys,
            optimizer_prune.alpha_scheduler.get_alpha().tolist(),
        ))
        print(alpha_payload)

        test_loss_ep, acc1_ep, acc5_ep = evaluate_loss_accuracy(
            model, test_loader, criterion
        )
        print(
            f"Epoch {epoch+1} 测试: Loss: {test_loss_ep:.4f}, "
            f"Top-1: {acc1_ep:.2f}%, Top-5: {acc5_ep:.2f}%"
        )

        epoch_number = epoch + 1
        period_index = epoch_number - warmup_epochs - 1
        period_number = period_index + 1 if period_index >= 0 else 0
        period_quota = (
            int(optimizer_prune.active_num_redundant_groups[period_index])
            if 0 <= period_index < pruning_epochs
            else 0
        )
        completed_periods = int(optimizer_prune.curr_pruning_period)
        cumulative_quota = int(sum(
            optimizer_prune.active_num_redundant_groups[:completed_periods]
        ))
        history_row = {
            "production_schedule_mode": production_schedule_mode,
            "epoch": epoch_number,
            "stage": "warmup" if epoch_number <= warmup_epochs else "pruning",
            "pruning_period": period_number,
            "completed_pruning_periods": completed_periods,
            "scheduled_period_quota": period_quota,
            "scheduled_cumulative_quota": cumulative_quota,
            "global_step": int(optimizer_prune.num_steps),
            "lr_first": float(epoch_first_lr or 0.0),
            "lr_last": float(epoch_last_lr or 0.0),
            "lr_mean": float(avg_lr),
            "lr_after_scheduler": float(optimizer_prune.param_groups[0]["lr"]),
            "train_loss": float(avg_loss),
            "train_top1": float(
                100.0 * epoch_top1_correct / max(epoch_examples, 1)
            ),
            "train_top5": float(
                100.0 * epoch_top5_correct / max(epoch_examples, 1)
            ),
            "test_loss": float(test_loss_ep),
            "test_top1": float(acc1_ep),
            "test_top5": float(acc5_ep),
            "group_sparsity": float(metrics.group_sparsity),
            "num_groups": int(metrics.num_groups),
            "num_zero_groups": int(metrics.num_zero_groups),
            "num_pruned_groups": int(metrics.num_pruned_groups),
            "num_unpruned_groups": int(metrics.num_unpruned_groups),
            "selected_production_groups": int(
                len(set(int(index) for index in optimizer_prune.pruned_group_idxes))
            ),
            "epoch_seconds": float(dt),
            "throughput_iterations_per_second": float(it_per_sec),
            **{
                f"alpha_{name}": float(value)
                for name, value in alpha_payload.items()
            },
        }
        training_history.append(history_row)
        if temporal_tracker is not None:
            temporal_tracker.write_training_history(training_history)
        model.train()

    if spatial_tracker is not None:
        spatial_tracker.finalize(expected_periods=pruning_epochs)
    if temporal_tracker is not None:
        temporal_tracker.finalize(expected_periods=pruning_epochs)

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
    if temporal_tracker is not None:
        temporal_tracker.write_final_summary({
            "seed": seed,
            "warmup_epochs": warmup_epochs,
            "pruning_epochs": pruning_epochs,
            "pruning_periods": pruning_epochs,
            "target_group_sparsity": target_sparsity,
            "initial_top1": float(accuracy1_before),
            "initial_top5": float(accuracy5_before),
            "compressed_top1": float(accuracy1_after),
            "compressed_top5": float(accuracy5_after),
            "top1_change": float(accuracy1_after - accuracy1_before),
            "top5_change": float(accuracy5_after - accuracy5_before),
            "group_sparsity": float(final_group_sparsity),
            "flops_before_m": float(flops_before_m),
            "flops_after_m": float(flops_after_m),
            "params_before_m": float(params_before_m),
            "params_after_m": float(params_after_m),
            "pruned_flops_percent": float(pruned_flops_percent),
            "pruned_params_percent": float(pruned_params_percent),
            "compressed_model_path": str(compressed_model_path),
            "complete_training_epochs": len(training_history),
        })


def finetune_model(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== 阶段二：微调 ===")
    print(f"使用设备: {device}")

    base_dir = CURRENT_FILE.parent
    local_data_path = os.environ.get('MOTIVATION_DATA_ROOT', '/workspace/data/cifar100')

    pruned_cache_dir = base_dir / "cache_resnet110_cifar100_pruned_AdaMCF_Full_0.5"
    compressed_model_path = None

    finetuned_model_dir = os.path.join(base_dir, 'finetuned_resnet110_sparsity_AdaMCF_Full_0_5')
    os.makedirs(finetuned_model_dir, exist_ok=True)
    finetuned_model_path = os.path.join(finetuned_model_dir, 'resnet110_cifar100_finetuned.pt')

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

    epochs = 300
    lr = 0.1
    
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
            epoch_loss += loss.item()
            epoch_lr += optimizer.param_groups[0]['lr']
            epoch_steps += 1
        
        scheduler.step()

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

    baseline_model_path = os.environ.get(
        'MOTIVATION_PRETRAINED_CKPT',
        str(REFERENCE_EXPERIMENT_ROOT / 'model' / 'best_model.pth'),
    )
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
    
    baseline_model.to(device)
    baseline_model.eval()
    baseline_top1, baseline_top5 = check_accuracy(baseline_model, test_loader)
    print(f"未剪枝模型准确率: Top-1: {baseline_top1:.2f}%, Top-5: {baseline_top5:.2f}%")
    print(f"相较未剪枝模型: Top-1 变化 {top1_final - baseline_top1:.2f}%, "f"Top-5 变化 {top5_final - baseline_top5:.2f}%")


def main():
    mode = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "help"
    if mode == "spatial":
        run_spatial_probe()
    elif mode == "spatial_periods":
        prune_model(track_spatial_periods=True)
    elif mode in {"temporal", "temporal_periods"}:
        prune_model(track_temporal_periods=True)
    elif mode in {"temporal_cosine", "temporal_cosine_front"}:
        prune_model(
            track_temporal_periods=True,
            production_schedule_mode="cosine_front",
        )
    elif mode in {"temporal_uniform", "temporal_balanced_uniform"}:
        prune_model(
            track_temporal_periods=True,
            production_schedule_mode="uniform",
        )
    elif mode == "motivation_periods":
        prune_model(
            track_spatial_periods=True,
            track_temporal_periods=True,
            production_schedule_mode="cosine_front",
        )
    elif mode == "prune":
        prune_model()
    elif mode == "finetune":
        finetune_model()
    elif mode == "full":
        prune_model()
        finetune_model()
    else:
        print(
            "Usage: python resnet110_cifar100_motivation.py "
            "{spatial|spatial_periods|temporal_cosine|temporal_uniform|"
            "temporal_periods|motivation_periods|prune|finetune|full}"
        )


if __name__ == "__main__":
    main()
