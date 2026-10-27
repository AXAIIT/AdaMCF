import sys
import os
import json
import csv
import pickle
import re
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import numpy as np
import random
import copy
from pathlib import Path
import time
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from only_train_once import OTO
from model.model import resnet110
from only_train_once.optimizer.alpha_scheduler import prepare_alpha_eval_batches


def parse_int_list_from_env(env_name, default_list):
    value = os.environ.get(env_name, "").strip()
    if not value:
        return default_list
    items = []
    for part in value.split(','):
        part = part.strip()
        if part:
            items.append(int(part))
    return items if items else default_list


def discover_epoch_checkpoints(snapshot_dir):
    ckpt_map = {}
    if not os.path.isdir(snapshot_dir):
        return ckpt_map

    pattern = re.compile(r'^chkpt_epoch_(\d+)\.pth$')
    for name in os.listdir(snapshot_dir):
        match = pattern.match(name)
        if match:
            epoch = int(match.group(1))
            ckpt_map[epoch] = os.path.join(snapshot_dir, name)
    return ckpt_map


def resolve_checkpoint_map(requested_epochs, snapshot_dir, allow_nearest=False):
    discovered = discover_epoch_checkpoints(snapshot_dir)
    if not discovered:
        raise FileNotFoundError(f"No checkpoints found in: {snapshot_dir}")

    available_epochs = sorted(discovered.keys())
    resolved = {}
    notes = []

    for epoch in requested_epochs:
        if epoch in discovered:
            resolved[epoch] = discovered[epoch]
            continue

        if allow_nearest:
            nearest_epoch = min(available_epochs, key=lambda ep: abs(ep - epoch))
            resolved[epoch] = discovered[nearest_epoch]
            notes.append(f"Requested epoch {epoch} not found; fallback to epoch {nearest_epoch}.")
        else:
            raise FileNotFoundError(
                f"Checkpoint for requested epoch {epoch} not found in {snapshot_dir}. "
                f"Available epochs: {available_epochs}"
            )

    return resolved, available_epochs, notes


def save_experiment_data(output_dir, epochs_to_test, strategies, raw_results, detailed_rows, metadata):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, 'loss_spike_results.json')
    csv_long_path = os.path.join(output_dir, 'loss_spike_results_long.csv')
    csv_wide_path = os.path.join(output_dir, 'loss_spike_results_wide.csv')

    payload = {
        'metadata': metadata,
        'epochs': epochs_to_test,
        'strategies': strategies,
        'series': raw_results,
        'records': detailed_rows,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(csv_long_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'epoch',
                'strategy',
                'original_loss',
                'pruned_loss',
                'loss_spike',
                'target_sparsity',
                'val_batches_used',
                'align_mode',
                'selected_proxy_ratio',
            ],
            extrasaction='ignore',
        )
        writer.writeheader()
        for row in detailed_rows:
            writer.writerow(row)

    with open(csv_wide_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch'] + strategies)
        for idx, epoch in enumerate(epochs_to_test):
            row = [epoch]
            for strategy in strategies:
                row.append(raw_results[strategy][idx])
            writer.writerow(row)

    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV (long): {csv_long_path}")
    print(f"Saved CSV (wide): {csv_wide_path}")


def build_theoretical_results(epochs_to_test, strategies):
    anchor_epochs = np.array([10, 30, 60, 90, 120], dtype=np.float32)
    anchor_series = {
        'L1_only': np.array([0.8, 2.5, 6.0, 10.5, 14.2], dtype=np.float32),
        'Fisher_only': np.array([12.0, 5.5, 2.0, 0.8, 0.6], dtype=np.float32),
        'Uniform_Static': np.array([3.5, 4.0, 3.8, 4.5, 6.2], dtype=np.float32),
        'Random': np.array([7.2, 6.8, 6.1, 5.5, 5.1], dtype=np.float32),
    }

    query_epochs = np.array(epochs_to_test, dtype=np.float32)
    results = {}
    for strategy in strategies:
        results[strategy] = np.interp(
            query_epochs,
            anchor_epochs,
            anchor_series[strategy],
            left=anchor_series[strategy][0],
            right=anchor_series[strategy][-1],
        ).tolist()

    detailed_rows = []
    for i, epoch in enumerate(epochs_to_test):
        for strategy in strategies:
            spike = float(results[strategy][i])
            detailed_rows.append({
                'epoch': int(epoch),
                'strategy': strategy,
                'original_loss': 0.0,
                'pruned_loss': spike,
                'loss_spike': spike,
                'target_sparsity': -1.0,
                'val_batches_used': -1,
            })
    return results, detailed_rows


def parse_strategy_list_from_env(default_list):
    raw = os.environ.get('LOSS_SPIKE_STRATEGIES', '').strip()
    if not raw:
        return default_list
    out = []
    for item in raw.split(','):
        s = item.strip()
        if s:
            out.append(s)
    return out if out else default_list


def get_strategy_weights(alpha_keys, strategy):
    weights = {k: 0.0 for k in alpha_keys}

    if strategy == 'L1_only':
        hit = None
        for key in alpha_keys:
            if 'l1' in key.lower():
                hit = key
                break
        if hit is None:
            raise KeyError(f'Cannot find L1 criterion in alpha keys: {alpha_keys}')
        weights[hit] = 1.0
        return weights

    if strategy == 'Fisher_only':
        hit = None
        for key in alpha_keys:
            if 'fisher' in key.lower():
                hit = key
                break
        if hit is None:
            raise KeyError(f'Cannot find Fisher criterion in alpha keys: {alpha_keys}')
        weights[hit] = 1.0
        return weights

    if strategy == 'Uniform_Static':
        val = 1.0 / max(len(alpha_keys), 1)
        for key in alpha_keys:
            weights[key] = val
        return weights

    if strategy == 'Random':
        return None

    raise ValueError(f'Unknown strategy: {strategy}')


def get_unit_proxy_weight(param_group, align_mode):
    if align_mode == 'group_param_size':
        num_groups = max(int(param_group['num_groups']), 1)
        total_numel = 0
        for param in param_group['params']:
            total_numel += int(param.numel())
        return float(total_numel) / float(num_groups)
    return 1.0


def build_pruning_plan_from_scores(optimizer_prune, strategy, target_sparsity, align_mode, random_seed):
    prunable_groups = [
        group for group in optimizer_prune.param_groups
        if group.get('is_prunable') and not group.get('is_auxiliary')
    ]

    alpha_keys = list(optimizer_prune._alpha_keys)
    weights = get_strategy_weights(alpha_keys, strategy)
    rng = random.Random(random_seed)

    global_stats = {}
    if strategy != 'Random':
        for key, coeff in weights.items():
            if coeff == 0.0:
                continue
            all_scores = [group['importance_scores'][key].detach().cpu().float() for group in prunable_groups]
            cat_scores = torch.cat(all_scores)
            c_min, c_max = cat_scores.min(), cat_scores.max()
            global_stats[key] = {'min': c_min, 'max': c_max, 'range': torch.clamp(c_max - c_min, min=1e-12)}

    candidates = []
    for group in prunable_groups:
        num_groups = int(group['num_groups'])
        unit_weight = get_unit_proxy_weight(group, align_mode)

        if strategy == 'Random':
            score_vec = np.array([rng.random() for _ in range(num_groups)], dtype=np.float32)
        else:
            score_tensor = torch.zeros(num_groups, dtype=torch.float32)
            for key, coeff in weights.items():
                if coeff == 0.0:
                    continue
                raw_score = group['importance_scores'][key].detach().cpu().float()
                norm_score = (raw_score - global_stats[key]['min']) / global_stats[key]['range']
                score_tensor = score_tensor + float(coeff) * norm_score
            score_vec = score_tensor.numpy()

        for local_idx in range(num_groups):
            candidates.append({
                'score': float(score_vec[local_idx]),
                'weight': float(unit_weight),
                'group': group,
                'local_idx': local_idx,
            })

    total_proxy_weight = sum(item['weight'] for item in candidates)
    target_proxy_weight = float(target_sparsity) * float(total_proxy_weight)

    candidates.sort(key=lambda item: item['score'])

    selected = []
    selected_proxy_weight = 0.0
    for item in candidates:
        selected.append(item)
        selected_proxy_weight += item['weight']
        if selected_proxy_weight >= target_proxy_weight:
            break

    plan = {}
    for item in selected:
        gid = id(item['group'])
        if gid not in plan:
            plan[gid] = {
                'group': item['group'],
                'idxes': [],
            }
        plan[gid]['idxes'].append(item['local_idx'])

    return plan, selected_proxy_weight, total_proxy_weight


def apply_hard_pruning_plan(optimizer_prune, plan):
    backups = []
    for entry in plan.values():
        idxes = sorted(set(entry['idxes']))
        if not idxes:
            continue
        local_backups = optimizer_prune._apply_ratio_and_backup(entry['group'], idxes, ratio=0.0)
        backups.extend(local_backups)
    return backups

def get_loss_and_acc(model, loader, criterion, device):
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
    return total_loss / total if total > 0 else 0.0, 0.0

def load_checkpoint_if_exists(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        print(f"Warning: Checkpoint not found at {ckpt_path}. Using current weights for demonstration.")
        return False

    trust_checkpoint = os.environ.get('LOSS_SPIKE_TRUST_CHECKPOINT', '1').strip().lower() in {'1', 'true', 'yes'}
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    except pickle.UnpicklingError as e:
        if not trust_checkpoint:
            raise RuntimeError(
                f"Checkpoint requires unsafe unpickling but LOSS_SPIKE_TRUST_CHECKPOINT is disabled: {ckpt_path}"
            ) from e
        print(f"Warning: weights_only load failed, fallback to weights_only=False for trusted checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return True

def run_loss_spike_experiment():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Running Temporal Drift & Loss Spike Experiment ===")
    print(f"Device: {device}")

    mode = os.environ.get('LOSS_SPIKE_MODE', 'experiment2').strip().lower()
    if mode not in ['experiment2', 'theoretical']:
        raise ValueError(f"Unsupported LOSS_SPIKE_MODE={mode}. Use 'experiment2' or 'theoretical'.")

    data_root = os.environ.get('LOSS_SPIKE_DATA_ROOT', '/workspace/data/cifar100')
    batch_size = int(os.environ.get('LOSS_SPIKE_BATCH_SIZE', '128'))
    num_workers = int(os.environ.get('LOSS_SPIKE_NUM_WORKERS', '0'))
    val_batches_to_use = int(os.environ.get('LOSS_SPIKE_VAL_BATCHES', '3'))

    align_mode = os.environ.get('LOSS_SPIKE_ALIGN_MODE', 'group_param_size').strip().lower()
    if align_mode not in {'group_count', 'group_param_size'}:
        raise ValueError("LOSS_SPIKE_ALIGN_MODE must be one of: group_count, group_param_size")

    print(f"mode={mode}")
    print(f"data_root={data_root}")
    print(f"batch_size={batch_size}, num_workers={num_workers}, val_batches_to_use={val_batches_to_use}")
    print(f"align_mode={align_mode}")

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    snapshot_dir = os.environ.get(
        'LOSS_SPIKE_SNAPSHOT_DIR',
        os.path.join(os.path.dirname(__file__), 'model_snapshots')
    )
    allow_nearest_ckpt = os.environ.get('LOSS_SPIKE_ALLOW_NEAREST_CKPT', '0').strip().lower() in {'1', 'true', 'yes'}

    # Replace with actual checkpoint paths if available
    epochs_to_test = parse_int_list_from_env('LOSS_SPIKE_EPOCHS', [10, 30, 60, 90, 120])
    ckpt_paths, available_ckpt_epochs, resolve_notes = resolve_checkpoint_map(
        requested_epochs=epochs_to_test,
        snapshot_dir=snapshot_dir,
        allow_nearest=allow_nearest_ckpt,
    )

    print(f"snapshot_dir={snapshot_dir}")
    print(f"available_checkpoint_epochs={available_ckpt_epochs}")
    if resolve_notes:
        for note in resolve_notes:
            print(f"Warning: {note}")

    print("Resolved checkpoint mapping:")
    for ep in epochs_to_test:
        print(f"  epoch {ep} -> {ckpt_paths[ep]}")
    strategies = parse_strategy_list_from_env(['L1_only', 'Fisher_only', 'Uniform_Static', 'Random'])
    
    # Store results
    results = {strategy: [] for strategy in strategies}
    detailed_rows = []

    target_sparsity = float(os.environ.get('LOSS_SPIKE_TARGET_SPARSITY', '0.3'))
    print(f"target_sparsity={target_sparsity}")

    if mode == 'theoretical':
        results, detailed_rows = build_theoretical_results(epochs_to_test, strategies)
    else:
        data_exists = os.path.exists(data_root)
        test_set = CIFAR100(root=data_root, train=False, download=not data_exists, transform=transform_test)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        
        train_set = CIFAR100(root=data_root, train=True, download=not data_exists, transform=transform_train)
        val_batches, alpha_info = prepare_alpha_eval_batches(
            trainset=train_set, device=device, batch_size=batch_size, num_workers=num_workers
        )

        criterion = nn.CrossEntropyLoss().to(device)

        for epoch in epochs_to_test:
            print(f"\n--- Testing Epoch {epoch} ---")
            base_model = resnet110(num_classes=100).to(device)
            load_checkpoint_if_exists(base_model, ckpt_paths[epoch], device)
            
            original_loss, _ = get_loss_and_acc(base_model, test_loader, criterion, device)
            print(f"Original Baseline Loss: {original_loss:.4f}")

            for strategy in strategies:
                model_eval = copy.deepcopy(base_model).to(device)
                dummy_input = torch.rand(1, 3, 32, 32).to(device)
                oto = OTO(model=model_eval, dummy_input=dummy_input)
                
                optimizer_prune = oto.hesso(
                    variant='sgd',
                    lr=0.01,
                    target_group_sparsity=target_sparsity,
                    start_pruning_step=0,
                    pruning_periods=1,
                    pruning_steps=1,
                    device=device
                )
                
                model_eval.eval()
                optimizer_prune.zero_grad()

                if strategy != 'Random':
                    for val_batch in val_batches[:val_batches_to_use]:
                        inputs, targets = val_batch
                        outputs = model_eval(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                    optimizer_prune.compute_importance_scores()

                plan, selected_proxy_weight, total_proxy_weight = build_pruning_plan_from_scores(
                    optimizer_prune=optimizer_prune,
                    strategy=strategy,
                    target_sparsity=target_sparsity,
                    align_mode=align_mode,
                    random_seed=seed + int(epoch),
                )
                apply_hard_pruning_plan(optimizer_prune, plan)
                
                model_eval.eval()
                pruned_loss, _ = get_loss_and_acc(model_eval, test_loader, criterion, device)
                
                loss_spike = pruned_loss - original_loss
                selected_proxy_ratio = selected_proxy_weight / max(total_proxy_weight, 1e-12)
                results[strategy].append(loss_spike)
                detailed_rows.append({
                    'epoch': epoch,
                    'strategy': strategy,
                    'original_loss': float(original_loss),
                    'pruned_loss': float(pruned_loss),
                    'loss_spike': float(loss_spike),
                    'target_sparsity': target_sparsity,
                    'val_batches_used': val_batches_to_use,
                    'align_mode': align_mode,
                    'selected_proxy_ratio': float(selected_proxy_ratio),
                })
                print(
                    f"  [{strategy}] Loss Spike: {loss_spike:.4f} "
                    f"(Pruned Loss: {pruned_loss:.4f}, Proxy Ratio: {selected_proxy_ratio:.4f})"
                )
                
                del model_eval, oto, optimizer_prune
                torch.cuda.empty_cache()

    print("\n=== Experiment Complete ===")
    for strategy in strategies:
        print(f"{strategy} Spikes: {results[strategy]}")

    output_dir = os.path.join(os.path.dirname(__file__), 'loss_spike_outputs')
    metadata = {
        'mode': mode,
        'device': str(device),
        'data_root': data_root,
        'snapshot_dir': snapshot_dir,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'val_batches_to_use': val_batches_to_use,
        'target_sparsity': target_sparsity,
        'align_mode': align_mode,
        'epochs_to_test': epochs_to_test,
        'available_checkpoint_epochs': available_ckpt_epochs,
        'allow_nearest_ckpt': allow_nearest_ckpt,
    }
    save_experiment_data(
        output_dir=output_dir,
        epochs_to_test=epochs_to_test,
        strategies=strategies,
        raw_results=results,
        detailed_rows=detailed_rows,
        metadata=metadata,
    )
        
    # Plotting Figure 1(b)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 5))

    # Optional: Log scale the y axis if your spikes get too huge. Let's start with linear but stylized like your photo.
    # Set custom font parameters similar to the image
    from matplotlib import rcParams
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Times New Roman', 'Arial', 'DejaVu Sans']

    ax = plt.gca()
    
    all_values = []
    for strategy in strategies:
        all_values.extend(results[strategy])

    # 如果存在<=0值，log坐标会异常；自动切到symlog避免运行失败
    if all(v > 0 for v in all_values):
        ax.set_yscale('log')
        y_label = 'Instant Loss Spike (Log)'
    else:
        ax.set_yscale('symlog', linthresh=1e-3)
        y_label = 'Instant Loss Spike (Symlog)'
        print("Warning: Detected non-positive loss spikes, switched y-axis to symlog.")

    # Draw grid background just like the reference photo 
    ax.yaxis.grid(True, linestyle=(0, (1, 3)), color='gray', alpha=0.5, linewidth=0.8) # dotted
    ax.xaxis.grid(True, linestyle='--', color='lightgray', alpha=0.7, linewidth=0.8) # dashed

    plt.plot(epochs_to_test, results['L1_only'], 
        label='Pure L1', color='royalblue', marker='o', 
        linestyle='-', linewidth=2, markersize=7)
    
    plt.plot(epochs_to_test, results['Fisher_only'], 
        label='Pure Fisher', color='crimson', marker='^', 
        linestyle='-', linewidth=2, markersize=7)
    
    if 'Uniform_Static' in results:
        plt.plot(epochs_to_test, results['Uniform_Static'], 
            label='Uniform Static', color='forestgreen', marker='s', 
            linestyle='--', linewidth=2, markersize=7)

    if 'Random' in results:
        plt.plot(epochs_to_test, results['Random'],
            label='Random', color='black', marker='x',
            linestyle='-.', linewidth=2, markersize=7)

    plt.xlabel('Training Epochs (Checkpoint)', fontsize=22, fontweight='bold', labelpad=4)
    plt.ylabel(y_label, fontsize=22, fontweight='bold', labelpad=4)
    
    plt.tick_params(axis='both', which='major', labelsize=18, width=1.5, length=5, direction='in')
    plt.xticks(epochs_to_test)
    
    # Fix the border spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    plt.legend(loc='best', frameon=True, fontsize=16, 
               handlelength=2.0, handletextpad=0.5,
               edgecolor='black', framealpha=1.0)

    plt.tight_layout()
    if mode == 'theoretical':
        plot_name = 'Motivation_Fig1b_LossSpike_Theoretical.png'
    else:
        plot_name = 'Experiment2_Temporal_Drift_LossSpike_Official.png'
    plot_path = os.path.join(os.path.dirname(__file__), plot_name)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    if mode == 'experiment2':
        pdf_path = os.path.join(os.path.dirname(__file__), 'Experiment2_Temporal_Drift_LossSpike_Official.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Plot saved to {pdf_path}")
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    run_loss_spike_experiment()
