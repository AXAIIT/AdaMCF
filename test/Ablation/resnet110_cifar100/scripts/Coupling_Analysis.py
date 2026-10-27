import sys
import os
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from thop import profile


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

from only_train_once import OTO
from model.model import resnet110
from only_train_once.optimizer.alpha_scheduler import prepare_alpha_eval_batches


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


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


def build_coupling_batch(
    data_root: str,
    batch_size: int = 256,
    seed: int = 42,
):
    """
    构造用于 coupling analysis 的固定 batch：
    - 来自训练集
    - 无随机增强
    - 固定 seed
    """
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_set = CIFAR100(
        root=data_root,
        train=True,
        download=False,
        transform=transform_eval,
    )

    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(eval_set), generator=rng)[:batch_size].tolist()

    subset = torch.utils.data.Subset(eval_set, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    fixed_inputs, fixed_targets = next(iter(loader))
    return fixed_inputs, fixed_targets


def build_coupling_batches(
    data_root: str,
    batch_size: int = 512,
    num_batches: int = 8,
    seed: int = 42,
):
    """
    构造用于 coupling analysis 的多个固定 batch：
    - 来自训练集
    - 无随机增强
    - 固定 seed
    """
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_set = CIFAR100(
        root=data_root,
        train=True,
        download=False,
        transform=transform_eval,
    )

    total_samples = min(len(eval_set), batch_size * num_batches)
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(eval_set), generator=rng)[:total_samples].tolist()

    subset = torch.utils.data.Subset(eval_set, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    fixed_batches = []
    for inputs, targets in loader:
        fixed_batches.append((inputs, targets))
        if len(fixed_batches) >= num_batches:
            break
    return fixed_batches


def calibrate_batchnorm_stats(model, fixed_batches, device, max_batches: int = 8):
    """
    使用固定样本前向更新 BN running stats，减少结构扰动导致的评估噪声。
    """
    if max_batches <= 0:
        return

    was_training = model.training
    model.train()
    with torch.no_grad():
        for idx, (inputs, _) in enumerate(fixed_batches):
            if idx >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            _ = model(inputs)
    if not was_training:
        model.eval()


def compute_spearman_corr(x, y):
    if len(x) < 2:
        return float("nan")
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def compute_trimmed_pearson(x, y, trim_ratio: float = 0.02):
    if len(x) < 2:
        return float("nan")

    if trim_ratio <= 0.0:
        return float(np.corrcoef(x, y)[0, 1])

    low_x, high_x = np.quantile(x, [trim_ratio, 1.0 - trim_ratio])
    low_y, high_y = np.quantile(y, [trim_ratio, 1.0 - trim_ratio])
    mask = (x >= low_x) & (x <= high_x) & (y >= low_y) & (y <= high_y)

    if int(mask.sum()) < 2:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def evaluate_plan_loss(model, criterion, batches, plan, optimizer_prune, ratio, device):
    backups = []
    for group in optimizer_prune.param_groups:
        gid = id(group)
        if gid in plan and len(plan[gid]) > 0:
            local_idxes = plan[gid]
            group_backups = optimizer_prune._apply_ratio_and_backup(
                group, local_idxes, ratio=ratio
            )
            backups.extend(group_backups)

    loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for inputs, targets in batches:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            batch_loss = criterion(outputs, targets).item()
            batch_n = int(targets.size(0))
            loss_sum += batch_loss * batch_n
            sample_count += batch_n

    optimizer_prune._restore_from_backups(backups)

    if sample_count == 0:
        return 0.0
    return loss_sum / sample_count


def build_alpha_candidates(
    M: int,
    n_dirichlet: int = 30000,
    dirichlet_conc: float = 0.7,
    seed: int = 42,
):
    """
    构造更丰富的 alpha 候选池：
    1) one-hot
    2) 两两均分
    3) Dirichlet 随机采样
    """
    rng = np.random.default_rng(seed)
    candidates = []

    # one-hot
    eye = np.eye(M, dtype=np.float32)
    candidates.extend(list(eye))

    # pairwise 0.5/0.5
    for i in range(M):
        for j in range(i + 1, M):
            vec = np.zeros(M, dtype=np.float32)
            vec[i] = 0.5
            vec[j] = 0.5
            candidates.append(vec)

    # Dirichlet
    alpha_param = np.full(M, dirichlet_conc, dtype=np.float32)
    dirichlet_samples = rng.dirichlet(alpha_param, size=n_dirichlet).astype(np.float32)
    candidates.extend(list(dirichlet_samples))

    rng.shuffle(candidates)
    return candidates


def run_coupling_analysis(
    model,
    optimizer_prune,
    criterion,
    data_root,
    device,
    base_dir,
    seed,
    target_sparsity,
    stop_epoch,
):
    """
    Theorem 2 的实证支撑图：
    在固定权重、固定 batch、固定 sparsity 下，
    随机采样多组 alpha，比较 soft proxy 与 hard objective 的耦合关系。
    """
    print("\n\n======== 耦合度分析 (Coupling Analysis) ========")
    model.eval()
    optimizer_prune.zero_grad()

    # 1) 固定 batch：训练集中的多个固定子集，无随机增强
    fixed_batches = build_coupling_batches(
        data_root=data_root,
        batch_size=512,
        num_batches=8,
        seed=seed,
    )
    if len(fixed_batches) == 0:
        print("[错误] 未构建出用于耦合分析的固定 batch。")
        return

    print(f"固定评估批次数={len(fixed_batches)}, 单批大小≈{fixed_batches[0][0].size(0)}")

    # 2) 在固定样本上校准 BN running stats
    calibrate_batchnorm_stats(model=model, fixed_batches=fixed_batches, device=device, max_batches=8)
    model.eval()

    # 3) 先做一次 forward/backward，保证 Fisher / GradWeight 这类 raw scores 可用
    grad_inputs, grad_targets = fixed_batches[0]
    grad_inputs = grad_inputs.to(device, non_blocking=True)
    grad_targets = grad_targets.to(device, non_blocking=True)
    out_for_grad = model(grad_inputs)
    loss_for_grad = criterion(out_for_grad, grad_targets)
    loss_for_grad.backward()

    optimizer_prune.compute_importance_scores()

    prunable_groups = [
        g for g in optimizer_prune.param_groups
        if g.get('is_prunable') and not g.get('is_auxiliary')
    ]
    total_prunable = sum(g['num_groups'] for g in prunable_groups)

    analysis_sparsity = target_sparsity
    target_k = max(1, int(total_prunable * analysis_sparsity))

    # 让 dry-pick 在固定 sparsity 下工作
    optimizer_prune.curr_pruning_period = 0
    optimizer_prune.active_num_redundant_groups[0] = target_k
    optimizer_prune.pruned_group_idxes = []

    M = len(optimizer_prune._alpha_keys)
    print("Proxy keys M =", M, optimizer_prune._alpha_keys)
    print(f"Coupling Analysis: total_prunable={total_prunable}, target_k={target_k}")

    alpha_pool = build_alpha_candidates(
        M=M,
        n_dirichlet=30000,
        dirichlet_conc=0.7,
        seed=seed,
    )

    desired_unique = 300
    soft_ratio = 0.25
    unique_masks = set()
    proxy_losses = []
    hard_losses = []
    collected_alphas = []

    print(
        f"Coupling config: desired_unique={desired_unique}, "
        f"n_dirichlet=30000, dirichlet_conc=0.7, soft_ratio={soft_ratio}"
    )
    print(f"开始搜索 {desired_unique} 组不同的剪枝结构 (unique masks)...")

    for alpha_np in alpha_pool:
        if len(unique_masks) >= desired_unique:
            break

        alpha_vec = torch.tensor(alpha_np, dtype=torch.float32, device=device)

        # dry-run 挑选待剪枝组
        plan = optimizer_prune._dry_pick_active_by_alpha(alpha_vec)

        mask_signature = []
        for gid in sorted(plan.keys()):
            mask_signature.append((gid, tuple(sorted(plan[gid]))))
        mask_signature = tuple(mask_signature)

        if mask_signature in unique_masks:
            continue

        unique_masks.add(mask_signature)
        collected_alphas.append(alpha_np.copy())

        # ===== 软代理损失：多固定批次均值 =====
        loss_soft = evaluate_plan_loss(
            model=model,
            criterion=criterion,
            batches=fixed_batches,
            plan=plan,
            optimizer_prune=optimizer_prune,
            ratio=soft_ratio,
            device=device,
        )
        proxy_losses.append(loss_soft)

        # ===== 真实硬剪枝损失：多固定批次均值 =====
        loss_hard = evaluate_plan_loss(
            model=model,
            criterion=criterion,
            batches=fixed_batches,
            plan=plan,
            optimizer_prune=optimizer_prune,
            ratio=0.0,
            device=device,
        )
        hard_losses.append(loss_hard)

        print(
            f"进度: {len(unique_masks):03d}/{desired_unique} | "
            f"Hard={loss_hard:.4f}, Soft={loss_soft:.4f}"
        )

    if len(hard_losses) < 20:
        print(
            f"[警告] 当前只收集到 {len(hard_losses)} 个 unique masks，"
            f"散点可能偏少。可继续增大 n_dirichlet。"
        )

    # ===== 绘图：散点 + 拟合虚线 + Pearson R =====
    import csv
    import matplotlib.pyplot as plt

    hard_arr = np.array(hard_losses, dtype=np.float64)
    proxy_arr = np.array(proxy_losses, dtype=np.float64)

    # Pearson R / Spearman rho / Trimmed Pearson R
    r = float(np.corrcoef(hard_arr, proxy_arr)[0, 1]) if len(hard_arr) > 1 else float("nan")
    rho = compute_spearman_corr(hard_arr, proxy_arr)
    r_trim = compute_trimmed_pearson(hard_arr, proxy_arr, trim_ratio=0.02)
    print(f"Pearson R = {r:.4f}")
    print(f"Spearman rho = {rho:.4f}")
    print(f"Trimmed Pearson R (2%) = {r_trim:.4f}")

    # 一阶线性拟合
    if len(hard_arr) >= 2 and np.any(np.diff(np.sort(hard_arr)) > 0):
        coef = np.polyfit(hard_arr, proxy_arr, deg=1)
        x_line = np.linspace(hard_arr.min(), hard_arr.max(), 200)
        y_line = coef[0] * x_line + coef[1]
    else:
        coef = [0.0, float(proxy_arr.mean()) if len(proxy_arr) else 0.0]
        x_line = np.array([hard_arr.min(), hard_arr.max()]) if len(hard_arr) else np.array([0.0, 1.0])
        y_line = np.array([coef[1], coef[1]])

    from matplotlib import font_manager

    times_new_roman_candidates = [
        "/root/.local/share/fonts/msttfonts/times.ttf",
        "/root/.local/share/fonts/msttfonts/timesi.ttf",
        "/root/.local/share/fonts/msttfonts/timesbd.ttf",
        "/root/.local/share/fonts/msttfonts/timesbi.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/timesi.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/timesbi.ttf",
    ]
    for font_path in times_new_roman_candidates:
        if os.path.exists(font_path):
            try:
                font_manager.fontManager.addfont(font_path)
            except Exception:
                continue

    plt.rcParams['font.family'] = ['Times New Roman']
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    ax.scatter(
        hard_arr,
        proxy_arr,
        s=36,
        alpha=0.70,
        edgecolors='k',
        linewidths=0.4,
    )
    ax.plot(
        x_line,
        y_line,
        linestyle='--',
        linewidth=2.0,
    )

    ax.set_xlabel(r'$\mathcal{L}_p^H(\alpha)$', fontweight='bold', fontsize=16)
    ax.set_ylabel(r'$\mathcal{L}_p^S(\alpha)$', fontweight='bold', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.5)

    textstr = (
        f'Pearson $R$ = {r:.4f}\n'
        f'Spearman $\\rho$ = {rho:.4f}\n'
        f'Trimmed Pearson $R$ = {r_trim:.4f}'
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
    ax.text(
        0.05, 0.95, textstr,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=props
    )

    plt.tight_layout()

    save_path_pdf = os.path.join(base_dir, "coupling_analysis_tpami.pdf")
    save_path_png = os.path.join(base_dir, "coupling_analysis_tpami.png")
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"耦合分析图已保存到: {save_path_pdf}")

    # ===== 保存原始数据 =====
    csv_path = os.path.join(base_dir, "Coupling_Analysis_data.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["Index", "Hard_Loss", "Soft_Proxy_Loss"] + [f"alpha_{i}" for i in range(M)]
        writer.writerow(header)
        for i in range(len(hard_arr)):
            row = [i, hard_arr[i], proxy_arr[i]] + collected_alphas[i].tolist()
            writer.writerow(row)
    print(f"原始数据（共 {len(hard_arr)} 对独立 mask）已保存到: {csv_path}")


def prune_model(seed: int = 42):
    base_dir = os.path.dirname(__file__)
    data_root = '/workspace/data/cifar100'
    pretrained_ckpt_path = '/workspace/OTOv2_v2/test/Ablation/resnet110_cifar100/model/best_model.pth'
    output_cache_dir = os.path.join(base_dir, 'cache_resnet110_cifar100_pruned_Adaptive_Fusion_0.5')
    os.makedirs(output_cache_dir, exist_ok=True)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage 1: Pruning ResNet-110 on CIFAR-100 ===")
    print(f"Device: {device}")

    # --- 数据 ---
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

    # alpha 更新子集（训练过程仍沿用 OTO 的实现）
    val_batches, alpha_info = prepare_alpha_eval_batches(
        trainset=train_set, device=device, batch_size=batch_size, num_workers=num_workers
    )
    print(
        f"[α-Subset] 来自训练集的 α-更新子集：样本数={alpha_info['n_alpha_val']}, "
        f"batch_size={alpha_info['alpha_batch_size']}"
    )
    print(
        f"α 更新缓存微批：共 {alpha_info['num_val_microbatches']} 个"
        f"（与 alpha scheduler 配置对齐）。"
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

    # --- AdaMCF 优化器 ---
    lr = 0.01
    target_sparsity = 0.5

    warmup_epochs = 5
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
        weight_decay=5e-4,
        target_group_sparsity=target_sparsity,
        start_pruning_step=start_pruning_step,
        pruning_periods=pruning_epochs,
        pruning_steps=pruning_steps,
        device=device
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_prune,
        T_max=total_epochs
    )

    print("\n--- 阶段一：先训练到稳定时刻，再做耦合度分析 ---")
    criterion = nn.CrossEntropyLoss().to(device)
    model.train()

    stop_epoch = 50

    for epoch in range(stop_epoch):
        print(f"\n[阶段一] Epoch {epoch + 1}/{stop_epoch} 正在训练以稳定权重...")
        epoch_loss = 0.0
        epoch_steps = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

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
            epoch_loss += loss.item()
            epoch_steps += 1

        scheduler.step()
        print(f"Epoch {epoch + 1} done. Avg Loss: {epoch_loss / max(epoch_steps, 1):.4f}")

    # ===== 耦合度分析 =====
    run_coupling_analysis(
        model=model,
        optimizer_prune=optimizer_prune,
        criterion=criterion,
        data_root=data_root,
        device=device,
        base_dir=base_dir,
        seed=seed,
        target_sparsity=target_sparsity,
        stop_epoch=stop_epoch,
    )

    return


def finetune_model(seed: int = 42):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== 阶段二：微调 ===")
    print(f"使用设备: {device}")

    base_dir = CURRENT_FILE.parent
    local_data_path = "/workspace/data/cifar100"

    pruned_cache_dir = base_dir / "cache_resnet110_cifar100_pruned_Adaptive_Fusion_0.5"
    compressed_model_path = None

    finetuned_model_dir = os.path.join(base_dir, 'finetuned_resnet110_sparsity_Adaptive_Fusion_0_5')
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

    epochs = 200
    lr = 0.01

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss().to(device)

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
        avg_loss = epoch_loss / max(epoch_steps, 1)
        avg_lr = epoch_lr / max(epoch_steps, 1)
        it_per_sec = epoch_steps / dt
        eta = (epochs - epoch - 1) * dt
        eta_h = int(eta // 3600)
        eta_m = int((eta % 3600) // 60)

        print(
            f"Epoch {epoch + 1}/{epochs} 评估\n"
            f"平均Loss: {avg_loss:.4f}, 平均LR: {avg_lr:.6f}, "
            f"吞吐: {it_per_sec:.2f} it/s, 用时: {dt:.2f}s, "
            f"估计剩余: {eta_h}h:{eta_m}m"
        )

        model.eval()
        accuracy1_after, accuracy5_after = check_accuracy(model, test_loader)
        print(f"Epoch {epoch + 1} 测试准确率: Top-1: {accuracy1_after:.2f}%, Top-5: {accuracy5_after:.2f}%")

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
    baseline_model = resnet110(num_classes=100)

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
    print(
        f"相较未剪枝模型: Top-1 变化 {top1_final - baseline_top1:.2f}%, "
        f"Top-5 变化 {top5_final - baseline_top5:.2f}%"
    )


if __name__ == "__main__":
    prune_model()