import sys
import os
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from only_train_once import OTO
from model.model import resnet110


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def build_eval_loader(
    data_root,
    subset_size=2048,
    batch_size=128,
    num_workers=0,
    seed=42,
):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_set = CIFAR100(root=data_root, train=False, download=False, transform=transform)

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(test_set), generator=g)[:subset_size].tolist()
    subset = Subset(test_set, indices)

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_grad_loader(
    data_root,
    subset_size=2048,
    batch_size=128,
    num_workers=0,
    seed=42,
):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = CIFAR100(root=data_root, train=True, download=False, transform=transform)

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_set), generator=g)[:subset_size].tolist()
    subset = Subset(train_set, indices)

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_avg_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)

    return total_loss / max(total, 1)


def load_checkpoint_strict(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def build_strategy_weights(alpha_keys, strategy):
    weights = {k: 0.0 for k in alpha_keys}

    if strategy == 'L1_only':
        hit = None
        for k in alpha_keys:
            if 'l1' in k.lower():
                hit = k
                break
        if hit is None:
            raise KeyError(f"Cannot find L1 proxy in alpha keys: {alpha_keys}")
        weights[hit] = 1.0

    elif strategy == 'Fisher_only':
        hit = None
        for k in alpha_keys:
            if 'fisher' in k.lower():
                hit = k
                break
        if hit is None:
            raise KeyError(f"Cannot find Fisher proxy in alpha keys: {alpha_keys}")
        weights[hit] = 1.0

    elif strategy == 'Uniform_Static':
        val = 1.0 / len(alpha_keys)
        for k in alpha_keys:
            weights[k] = val
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return weights


def accumulate_gradients_for_scores(model, grad_loader, criterion, device, max_batches=12):
    model.eval()
    model.zero_grad(set_to_none=True)

    for bi, (inputs, targets) in enumerate(grad_loader):
        if bi >= max_batches:
            break

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()


def evaluate_one_static_strategy(
    base_model,
    eval_loader,
    grad_loader,
    criterion,
    device,
    target_group_sparsity,
    strategy,
    grad_batches=12,
):
    model_eval = copy.deepcopy(base_model).to(device)
    model_eval.eval()

    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    oto = OTO(model=model_eval, dummy_input=dummy_input)

    optimizer_prune = oto.hesso(
        variant='sgd',
        lr=0.01,
        first_momentum=0.9,
        second_momentum=0.0,
        dampening=0.0,
        weight_decay=5e-4,
        target_group_sparsity=target_group_sparsity,
        start_pruning_step=0,
        pruning_periods=1,
        pruning_steps=1,
        device=device,
    )

    alpha_keys = list(optimizer_prune._alpha_keys)
    static_weights = build_strategy_weights(alpha_keys, strategy)
    optimizer_prune.importance_score_criteria = dict(static_weights)

    accumulate_gradients_for_scores(
        model=model_eval,
        grad_loader=grad_loader,
        criterion=criterion,
        device=device,
        max_batches=grad_batches,
    )

    optimizer_prune.compute_importance_scores()

    prunable_groups = [
        g for g in optimizer_prune.param_groups
        if g.get('is_prunable') and not g.get('is_auxiliary')
    ]
    total_prunable = sum(g['num_groups'] for g in prunable_groups)
    target_k = max(1, int(total_prunable * target_group_sparsity))

    optimizer_prune.curr_pruning_period = 0
    optimizer_prune.pruned_group_idxes = []
    optimizer_prune.active_num_redundant_groups[0] = target_k

    alpha_vec = torch.tensor(
        [static_weights[k] for k in alpha_keys],
        dtype=torch.float32,
        device=device,
    )

    plan = optimizer_prune._dry_pick_active_by_alpha(alpha_vec)

    backups = []
    for group in optimizer_prune.param_groups:
        gid = id(group)
        if gid in plan and len(plan[gid]) > 0:
            local_idxes = plan[gid]
            b = optimizer_prune._apply_ratio_and_backup(group, local_idxes, ratio=0.0)
            backups.extend(b)

    pruned_loss = get_avg_loss(model_eval, eval_loader, criterion, device)
    optimizer_prune._restore_from_backups(backups)

    del model_eval, oto, optimizer_prune
    torch.cuda.empty_cache()

    return pruned_loss


def plot_results(epochs_to_test, results, save_dir):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8.6, 5.8), dpi=300)

    all_vals = []
    for v in results.values():
        all_vals.extend(v)

    use_log = all(x > 0 for x in all_vals)
    if use_log:
        ax.set_yscale('log')

    ax.yaxis.grid(True, linestyle=(0, (1, 3)), color='gray', alpha=0.5, linewidth=0.8)
    ax.xaxis.grid(True, linestyle='--', color='lightgray', alpha=0.7, linewidth=0.8)

    ax.plot(
        epochs_to_test, results['L1_only'],
        label='Pure L1',
        color='royalblue',
        marker='o',
        linestyle='-',
        linewidth=2.2,
        markersize=7,
    )
    ax.plot(
        epochs_to_test, results['Fisher_only'],
        label='Pure Fisher',
        color='crimson',
        marker='^',
        linestyle='-',
        linewidth=2.2,
        markersize=7,
    )
    ax.plot(
        epochs_to_test, results['Uniform_Static'],
        label='Uniform Static',
        color='forestgreen',
        marker='s',
        linestyle='--',
        linewidth=2.2,
        markersize=7,
    )

    ax.set_xlabel('Training Epochs (Checkpoint)', fontsize=22, fontweight='bold', labelpad=4)
    ax.set_ylabel('Instant Loss Spike (Log)' if use_log else 'Instant Loss Spike',
                  fontsize=22, fontweight='bold', labelpad=4)

    ax.tick_params(axis='both', which='major', labelsize=18, width=1.5, length=5, direction='in')
    ax.set_xticks(epochs_to_test)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    ax.legend(
        loc='best',
        frameon=True,
        fontsize=16,
        handlelength=2.0,
        handletextpad=0.5,
        edgecolor='black',
        framealpha=1.0,
    )

    plt.tight_layout()

    save_png = os.path.join(save_dir, 'Motivation_Fig1b_LossSpike.png')
    save_pdf = os.path.join(save_dir, 'Motivation_Fig1b_LossSpike.pdf')
    plt.savefig(save_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_pdf, bbox_inches='tight')
    print(f"Plot saved to: {save_png}")
    print(f"Plot saved to: {save_pdf}")


def run_loss_spike_experiment():
    seed = 42
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Running Temporal Drift & Loss Spike Experiment ===")
    print(f"Device: {device}")

    data_root = '/workspace/data/cifar100'
    save_dir = os.path.dirname(__file__)

    epochs_to_test = [10, 30, 60, 90, 120]
    ckpt_paths = {
        ep: f"/workspace/OTOv2_v2/test/Ablation/resnet110_cifar100/model_snapshots/chkpt_epoch_{ep}.pth"
        for ep in epochs_to_test
    }

    strategies = ['L1_only', 'Fisher_only', 'Uniform_Static']
    results = {s: [] for s in strategies}

    target_group_sparsity = 0.30
    grad_batches = 12
    batch_size = 128

    eval_loader = build_eval_loader(
        data_root=data_root,
        subset_size=2048,
        batch_size=batch_size,
        num_workers=0,
        seed=seed,
    )
    grad_loader = build_grad_loader(
        data_root=data_root,
        subset_size=2048,
        batch_size=128,
        num_workers=0,
        seed=seed,
    )

    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in epochs_to_test:
        print(f"\n--- Testing checkpoint @ Epoch {epoch} ---")

        base_model = resnet110(num_classes=100).to(device)
        load_checkpoint_strict(base_model, ckpt_paths[epoch], device)
        base_model.eval()

        original_loss = get_avg_loss(base_model, eval_loader, criterion, device)
        print(f"Original baseline loss = {original_loss:.6f}")

        for strategy in strategies:
            pruned_loss = evaluate_one_static_strategy(
                base_model=base_model,
                eval_loader=eval_loader,
                grad_loader=grad_loader,
                criterion=criterion,
                device=device,
                target_group_sparsity=target_group_sparsity,
                strategy=strategy,
                grad_batches=grad_batches,
            )

            loss_spike = pruned_loss - original_loss
            results[strategy].append(loss_spike)

            print(
                f"  [{strategy:<14}] "
                f"pruned_loss={pruned_loss:.6f} | "
                f"loss_spike={loss_spike:.6f}"
            )

        torch.cuda.empty_cache()

    print("\n=== Experiment Complete ===")
    for strategy in strategies:
        print(f"{strategy}: {results[strategy]}")

    plot_results(
        epochs_to_test=epochs_to_test,
        results=results,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    run_loss_spike_experiment()
    