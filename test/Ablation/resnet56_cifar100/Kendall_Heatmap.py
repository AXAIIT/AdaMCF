import sys
import os
import json
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from transformers import get_scheduler
import numpy as np
import random
from thop import profile
from pathlib import Path
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

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
from model.model import resnet56
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


def _dedupe_keep_order(items):
    seen = set()
    out = []
    for item in items:
        if item is None:
            continue
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _to_list_if_needed(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _extract_param_names_from_obj(obj):
    if obj is None:
        return []

    candidate_attrs = [
        "param_names",
        "parameter_names",
        "learnable_param_names",
        "p_names",
        "params_name",
        "names",
    ]

    collected = []
    for attr in candidate_attrs:
        if not hasattr(obj, attr):
            continue
        value = getattr(obj, attr)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        for x in _to_list_if_needed(value):
            if isinstance(x, str):
                collected.append(x)

    return _dedupe_keep_order(collected)


def _extract_group_params(group, named_params):
    # 1) 先尝试直接拿张量/参数
    for attr in ("params", "parameters"):
        if not hasattr(group, attr):
            continue
        value = getattr(group, attr)
        if callable(value):
            try:
                value = list(value())
            except TypeError:
                value = None
        values = _to_list_if_needed(value)
        tensors = [x for x in values if isinstance(x, (torch.Tensor, nn.Parameter))]
        if tensors:
            return tensors

    # 2) 再尝试通过参数名映射
    names = []
    names.extend(_extract_param_names_from_obj(group))

    for container_attr in ("nodes", "contained_nodes", "group_nodes", "node_list"):
        if not hasattr(group, container_attr):
            continue
        nodes = _to_list_if_needed(getattr(group, container_attr))
        for node in nodes:
            names.extend(_extract_param_names_from_obj(node))
            op = getattr(node, "op", None)
            if op is not None:
                names.extend(_extract_param_names_from_obj(op))

    names = _dedupe_keep_order(names)
    params = [named_params[name] for name in names if name in named_params]
    return params


def _iter_dependency_groups(oto):
    candidates = [
        getattr(oto, "node_groups", None),
        getattr(getattr(oto, "_graph", None), "node_groups", None),
        getattr(getattr(oto, "graph", None), "node_groups", None),
        getattr(getattr(oto, "dependency_graph", None), "node_groups", None),
    ]

    for groups in candidates:
        if groups is None:
            continue
        if isinstance(groups, dict):
            groups = list(groups.values())
        else:
            groups = list(groups)
        if len(groups) > 0:
            return groups

    raise RuntimeError("未能从 OTO 对象中找到依赖组（node_groups）。")


def _group_identifier(group, idx):
    for attr in ("id", "group_id", "gid", "name"):
        if hasattr(group, attr):
            value = getattr(group, attr)
            if value is not None:
                return str(value)
    return f"group_{idx:04d}"


def collect_dependency_group_scores(model, oto, optimizer_prune=None, prev_score_state=None):
    del model, oto, prev_score_state

    if optimizer_prune is None:
        raise RuntimeError("collect_dependency_group_scores 需要传入有效的 optimizer_prune。")
    if not hasattr(optimizer_prune, "export_kendall_raw_scores"):
        raise RuntimeError(
            "当前 HESSO 版本缺少 `export_kendall_raw_scores()` 接口，无法导出真实原始分数。"
        )

    exported = optimizer_prune.export_kendall_raw_scores(refresh=True)
    score_map = exported["score_map"]
    group_ids = exported["group_ids"]

    if len(group_ids) == 0:
        raise RuntimeError("HESSO 未导出任何结构组原始分数，无法计算 Kendall 分析。")

    return score_map, group_ids


def kendall_tau_b(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if x.size != y.size:
        raise ValueError("Kendall tau 输入长度不一致。")
    if x.size < 2:
        return 0.0

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0

    for i in range(x.size - 1):
        dx = x[i] - x[i + 1:]
        dy = y[i] - y[i + 1:]
        sx = np.sign(dx)
        sy = np.sign(dy)

        both_nonzero = (sx != 0) & (sy != 0)
        concordant += np.count_nonzero(sx[both_nonzero] == sy[both_nonzero])
        discordant += np.count_nonzero(sx[both_nonzero] != sy[both_nonzero])
        ties_x += np.count_nonzero((sx == 0) & (sy != 0))
        ties_y += np.count_nonzero((sx != 0) & (sy == 0))

    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def build_kendall_matrix(score_map):
    names = ["L1", "L2", "Fisher", "GW", "Trend"]
    matrix = np.eye(len(names), dtype=np.float64)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            tau = kendall_tau_b(score_map[names[i]], score_map[names[j]])
            matrix[i, j] = tau
            matrix[j, i] = tau

    return names, matrix


def plot_kendall_heatmap(names, matrix, save_path_png, save_path_pdf, title):
    del title

    label_map = {
        "L1": r"$L_1$",
        "L2": r"$L_2$",
        "Fisher": "Fisher",
        "GW": "GW",
        "Trend": "Trend",
    }
    display_names = [label_map.get(name, name) for name in names]

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
                fm.fontManager.addfont(font_path)
            except Exception:
                continue

    plt.rcParams.update({
        "font.family": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    })

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=300)

    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    masked_matrix = np.ma.array(matrix, mask=mask)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="white", alpha=0.0)

    im = ax.imshow(masked_matrix, cmap=cmap, vmin=-1.0, vmax=1.0, interpolation="none")

    ax.set_xticks(np.arange(len(display_names)))
    ax.set_yticks(np.arange(len(display_names)))
    ax.set_xticklabels(display_names, fontsize=18)
    ax.set_yticklabels(display_names, fontsize=18)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True, length=0)
    plt.setp(ax.get_xticklabels(), rotation=28, ha="right", rotation_mode="anchor")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if j > i:
                continue
            value = matrix[i, j]
            color = "white" if abs(value) >= 0.55 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=14)

    ax.set_xticks(np.arange(-0.5, len(display_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(display_names), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(-0.5, len(display_names) - 0.5)
    ax.set_ylim(len(display_names) - 0.5, -0.5)
    ax.set_aspect("equal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r"Kendall's $\tau$", fontsize=16)
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=14, length=3)
    cbar.outline.set_linewidth(0.6)

    fig.tight_layout(pad=0.4)
    fig.savefig(save_path_png, dpi=300, format="png", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(save_path_pdf, format="pdf", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def save_kendall_analysis(score_map, group_ids, out_dir, prefix, title):
    os.makedirs(out_dir, exist_ok=True)

    numeric_score_map = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in score_map.items()
    }

    # 5 个一维数组
    for name, arr in numeric_score_map.items():
        np.save(os.path.join(out_dir, f"{prefix}_{name}.npy"), arr)

    # 依赖组 ID
    np.save(
        os.path.join(out_dir, f"{prefix}_group_ids.npy"),
        np.asarray(group_ids, dtype=object),
        allow_pickle=True,
    )

    # 打包保存
    np.savez(
        os.path.join(out_dir, f"{prefix}_all_scores.npz"),
        **numeric_score_map,
    )

    # Kendall 矩阵
    names, kendall_matrix = build_kendall_matrix(numeric_score_map)
    np.save(os.path.join(out_dir, f"{prefix}_kendall_tau.npy"), kendall_matrix)

    with open(os.path.join(out_dir, f"{prefix}_kendall_tau.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "criteria": names,
                "group_ids": group_ids,
                "kendall_tau": kendall_matrix.tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 热力图
    heatmap_path_png = os.path.join(out_dir, f"{prefix}_kendall_tau_heatmap.png")
    heatmap_path_pdf = os.path.join(out_dir, f"{prefix}_kendall_tau_heatmap.pdf")
    plot_kendall_heatmap(names, kendall_matrix, heatmap_path_png, heatmap_path_pdf, title)


def prune_model(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    base_dir = os.path.dirname(__file__)
    local_data_path = '/workspace/data/cifar100'
    output_cache_dir = os.path.join(base_dir, 'cache_resnet56_cifar10_pruned_AdaMCF_Full_0.5')
    os.makedirs(output_cache_dir, exist_ok=True)

    kendall_output_dir = os.path.join(base_dir, "kendall_rank_analysis_AdaMCF_Full_0.5")
    os.makedirs(kendall_output_dir, exist_ok=True)

    batch_size = 128
    num_workers = 16

    local_model_path = '/workspace/OTOv2_v2/test/Ablation/resnet56_cifar100/model/best.pth'
    print(f"加载预训练 ResNet56 权重: {local_model_path}")
    model = resnet56()
    ckpt = torch.load(local_model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    
    model.to(device)
    model.eval()

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

    # ===== α 子集 =====
    val_batches, alpha_info = prepare_alpha_eval_batches(
        trainset=train_set,
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
    accuracy1_before, accuracy5_before = check_accuracy(model, test_loader)
    print(f"剪枝前准确率: Top-1: {accuracy1_before:.2f}%, Top-5: {accuracy5_before:.2f}%")

    flops_before_m, params_before_m = compute_flops_with_thop(model, input_size=(1, 3, 32, 32))
    print(f"剪枝前 FLOPs(thop): {flops_before_m:.2f} M")
    print(f"剪枝前参数量: {params_before_m:.2f} M")

    learning_rate = 1e-3
    warmup_epochs = 10
    pruning_epochs = 120
    pruning_periods = pruning_epochs
    target_group_sparsity = 0.5
    lr_scheduler_type = "cosine"

    num_update_steps_per_epoch = len(train_loader)
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
        weight_decay=1e-4,
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

    global_step = 0
    period_analysis_idx = 0
    for epoch in range(warmup_epochs + pruning_epochs):
        print(f"\n[阶段一] Epoch {epoch+1}/{total_epochs} 正在剪枝...")
        epoch_loss = 0.0
        epoch_lr = 0.0
        epoch_steps = 0
        t0 = time.time()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # 在“即将进入下一个 pruning period”时，先暂停做 Kendall 分析。
            # 这里发生在真正 step() 之前，因此不会提前触发本次剪枝。
            if (global_step + 1) >= start_pruning_step and optimizer_prune.will_enter_new_period_on_next_step():
                period_analysis_idx += 1
                print(f"[Kendall] 周期 {period_analysis_idx} 结束，开始统计 5 个准则的依赖组原始分数...")

                score_map, group_ids = collect_dependency_group_scores(
                    model=model,
                    oto=oto,
                    optimizer_prune=optimizer_prune,
                    prev_score_state=None,
                )

                save_kendall_analysis(
                    score_map=score_map,
                    group_ids=group_ids,
                    out_dir=kendall_output_dir,
                    prefix=f"period_{period_analysis_idx:03d}",
                    title=f"Kendall Tau Heatmap - Period {period_analysis_idx}",
                )

                print(f"[Kendall] 已保存到: {kendall_output_dir}")
                print(f"[Kendall] 文件前缀: period_{period_analysis_idx:03d}")

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
            global_step += 1

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

    pruned_cache_dir = base_dir / "cache_resnet56_cifar10_pruned_AdaMCF_Full_0.5"
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

    finetune_lr = 1e-3
    num_train_epochs = 300
    warmup_epochs = 0
    total_epochs = num_train_epochs
    steps_per_epoch = len(train_loader)
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    lr_scheduler_type = "cosine"

    finetuned_model_dir = os.path.join(base_dir, 'finetuned_resnet56_sparsity_AdaMCF_Full_0_5')
    os.makedirs(finetuned_model_dir, exist_ok=True)
    finetuned_model_path = os.path.join(finetuned_model_dir, 'resnet56_cifar10_finetuned.pt')

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=finetune_lr,
        momentum=0.9,
        weight_decay=1e-4,
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
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

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

    baseline_model_path = '/workspace/OTOv2_v2/test/Ablation/resnet56_cifar100/model/best.pth'
    baseline_model = resnet56()
    ckpt = torch.load(baseline_model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    baseline_model.load_state_dict(state_dict, strict=True)  # 修复这一行
        
    baseline_model.to(device)
    baseline_model.eval()
    baseline_top1, baseline_top5 = check_accuracy(baseline_model, test_loader)
    print(f"未剪枝模型准确率: Top-1: {baseline_top1:.2f}%, Top-5: {baseline_top5:.2f}%")
    print(f"相较未剪枝模型: Top-1 变化 {top1_final - baseline_top1:.2f}%, "f"Top-5 变化 {top5_final - baseline_top5:.2f}%")


def main():
    prune_model()
    finetune_model()


if __name__ == "__main__":
    main()
