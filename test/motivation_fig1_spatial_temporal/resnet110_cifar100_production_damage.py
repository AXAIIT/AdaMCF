"""Measure real HESSO production damage with a period-level matched control.

This experiment does not insert the fixed 5% temporary probe used by Fig. 1(b).
At the beginning of every real pruning period it deep-copies the production
model and HESSO optimizer after the period's alpha update and first backward
pass.  The two branches then consume exactly the same augmented minibatches:

* production: execute the requested HESSO quota and gradual group decay;
* control: execute the same optimizer state and training, but set only the
  current period's newly allocated quota to zero.

At period end, the causal production damage is evaluated on the same
deterministic CIFAR-100 test set:

    d_p = L(production_end) - L(control_end)

The production branch is retained for the next period; the control branch is
discarded.  Uniform and cosine-front must be run as independent jobs.
"""

from __future__ import annotations

import copy
import csv
import gc
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100


CURRENT_FILE = Path(__file__).resolve()
CURRENT_DIR = CURRENT_FILE.parent
PROJECT_ROOT = CURRENT_FILE.parents[2]
REFERENCE_EXPERIMENT_ROOT = (
    PROJECT_ROOT / "test" / "Ablation" / "resnet110_cifar100"
)
for path in (PROJECT_ROOT, REFERENCE_EXPERIMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from only_train_once import OTO
from model.model import resnet110
from only_train_once.optimizer.alpha_scheduler import (
    ALPHA_CFG,
    prepare_alpha_eval_batches,
)


OUTPUT_ROOT = CURRENT_DIR / "outputs" / "production_damage"
IMPLEMENTATION_VERSION = "full_hesso_state_clone_v3_scheduler_safe"


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if value is None:
        return ""
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _persist_run(
    run_dir: Path,
    production_records: list[dict[str, Any]],
    training_history: list[dict[str, Any]],
) -> None:
    _json_dump(run_dir / "production_damage.json", production_records)
    _write_csv(run_dir / "production_damage.csv", production_records)
    _json_dump(run_dir / "training_history.json", training_history)
    _write_csv(run_dir / "training_history.csv", training_history)


def _evaluate(model, loader, criterion) -> tuple[float, float, float]:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    loss_sum = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0
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


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _normalize_schedule_mode(value: str) -> str:
    aliases = {
        "uniform": "uniform",
        "balanced_uniform": "uniform",
        "cosine": "cosine_front",
        "cosine_front": "cosine_front",
    }
    key = str(value).strip().lower()
    if key not in aliases:
        raise ValueError("schedule mode must be uniform or cosine_front")
    return aliases[key]


def _load_dense_checkpoint(model, checkpoint_path: Path, device) -> None:
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    state_dict = {
        str(key).replace("module.", ""): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)


def _optimizer_parameter_ids(optimizer) -> set[int]:
    result: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group.get("params", []):
            result.add(id(parameter))
    return result


def _deepcopy_bound_model_optimizer(model, optimizer):
    """Clone the complete model/HESSO state while preserving parameter binding.

    ``torch.optim.Optimizer.__getstate__`` serializes only the standard optimizer
    fields.  Consequently, ``copy.deepcopy((model, optimizer))`` can silently
    drop HESSO-specific runtime attributes such as ``curr_pruning_period`` and
    ``active_num_redundant_groups``.  ``nn.Parameter.__deepcopy__`` also does not
    guarantee that an already-computed ``.grad`` is carried to the clone.

    The control branch is created immediately after backward, so both details
    matter: it needs every HESSO runtime field and the identical first-minibatch
    gradients.  Clone the model first, map every source parameter/buffer to its
    cloned counterpart, then deepcopy the optimizer's complete ``__dict__``
    with that mapping.
    """
    control_model = copy.deepcopy(model)

    source_parameters = list(model.parameters())
    control_parameters = list(control_model.parameters())
    if len(source_parameters) != len(control_parameters):
        raise RuntimeError("Model deepcopy changed the parameter count.")

    control_optimizer = type(optimizer).__new__(type(optimizer))
    memo = {
        id(model): control_model,
        id(optimizer): control_optimizer,
    }
    for source_parameter, control_parameter in zip(
        source_parameters, control_parameters
    ):
        memo[id(source_parameter)] = control_parameter
        if source_parameter.grad is None:
            control_parameter.grad = None
        else:
            control_parameter.grad = source_parameter.grad.detach().clone(
                memory_format=torch.preserve_format
            )

    source_buffers = list(model.buffers())
    control_buffers = list(control_model.buffers())
    if len(source_buffers) != len(control_buffers):
        raise RuntimeError("Model deepcopy changed the buffer count.")
    for source_buffer, control_buffer in zip(source_buffers, control_buffers):
        memo[id(source_buffer)] = control_buffer

    control_optimizer.__dict__ = copy.deepcopy(optimizer.__dict__, memo)
    missing_optimizer_fields = set(optimizer.__dict__) - set(
        control_optimizer.__dict__
    )
    if missing_optimizer_fields:
        raise RuntimeError(
            "HESSO full-state deepcopy dropped fields: "
            + ", ".join(sorted(missing_optimizer_fields))
        )

    # PyTorch LR schedulers patch ``optimizer.step`` on the optimizer instance
    # with a closure that keeps a weak reference to the original optimizer.
    # Functions are not rebound by deepcopy, so retaining that wrapper would
    # make ``control_optimizer.step()`` advance Production again.  Control has
    # no LR scheduler of its own within this one-period branch; remove only the
    # inherited instance wrapper so Python binds HESSO.step to Control.
    control_optimizer.__dict__.pop("step", None)
    if getattr(control_optimizer.step, "__self__", None) is not control_optimizer:
        raise RuntimeError("Control HESSO.step is not bound to control HESSO.")

    if hasattr(control_optimizer, "alpha_scheduler"):
        control_optimizer.alpha_scheduler.attach_backend(control_optimizer)
        if control_optimizer.alpha_scheduler.backend is not control_optimizer:
            raise RuntimeError("Control Alpha scheduler is not bound to control HESSO.")
    return control_model, control_optimizer


def _clone_no_new_pruning_control(model, optimizer):
    """Clone a bound model/optimizer pair and suppress only the next quota."""
    control_model, control_optimizer = _deepcopy_bound_model_optimizer(
        model, optimizer
    )
    production_parameter_ids = {id(parameter) for parameter in model.parameters()}
    control_model_parameter_ids = {
        id(parameter) for parameter in control_model.parameters()
    }
    control_optimizer_parameter_ids = _optimizer_parameter_ids(control_optimizer)
    if control_optimizer_parameter_ids - control_model_parameter_ids:
        raise RuntimeError(
            "Deep-copied HESSO parameters are not bound to the control model."
        )
    if control_optimizer_parameter_ids & production_parameter_ids:
        raise RuntimeError(
            "Control HESSO still references production-model parameters."
        )

    period_index = int(control_optimizer.curr_pruning_period)
    if not 0 <= period_index < int(control_optimizer.pruning_periods):
        raise RuntimeError("Cannot create a control outside the pruning stage.")
    original_quota = int(
        control_optimizer.active_num_redundant_groups[period_index]
    )
    control_optimizer.active_num_redundant_groups[period_index] = 0
    if int(control_optimizer.num_steps) != int(optimizer.num_steps):
        raise RuntimeError(
            "Control clone starts from a different optimizer step: "
            f"production={int(optimizer.num_steps)}, "
            f"control={int(control_optimizer.num_steps)}."
        )
    return control_model, control_optimizer, period_index, original_quota


def _copy_latest_artifacts(run_dir: Path, mode_root: Path) -> None:
    for filename in (
        "run_metadata.json",
        "production_damage.json",
        "production_damage.csv",
        "training_history.json",
        "training_history.csv",
        "final_summary.json",
    ):
        source = run_dir / filename
        if source.exists():
            shutil.copyfile(source, mode_root / filename)
    _json_dump(
        mode_root / "latest_run.json",
        {"run_id": run_dir.name, "run_dir": str(run_dir)},
    )


def run_production_damage(schedule_mode: str) -> None:
    schedule_mode = _normalize_schedule_mode(schedule_mode)
    seed = int(os.environ.get("MOTIVATION_SEED", "42"))
    batch_size = int(os.environ.get("MOTIVATION_BATCH_SIZE", "128"))
    num_workers = int(os.environ.get("MOTIVATION_NUM_WORKERS", "16"))
    eval_num_workers = int(
        os.environ.get("PRODUCTION_DAMAGE_EVAL_NUM_WORKERS", "4")
    )
    data_root = Path(
        os.environ.get("MOTIVATION_DATA_ROOT", "/workspace/data/cifar100")
    )
    pretrained_checkpoint = Path(
        os.environ.get(
            "MOTIVATION_PRETRAINED_CKPT",
            str(REFERENCE_EXPERIMENT_ROOT / "model" / "best_model.pth"),
        )
    )
    warmup_epochs = 20
    pruning_epochs = 100
    total_epochs = warmup_epochs + pruning_epochs
    learning_rate = 0.01
    target_group_sparsity = 0.5

    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mode_root = OUTPUT_ROOT / schedule_mode
    run_dir = mode_root / run_id
    cache_dir = CURRENT_DIR / f"cache_production_damage_{schedule_mode}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    cache_dir.mkdir(parents=True, exist_ok=False)

    print("=== Matched-control production-damage experiment ===")
    print(f"Implementation: {IMPLEMENTATION_VERSION}")
    print(f"Schedule: {schedule_mode}")
    print(f"Device: {device}")
    print(f"Run directory: {run_dir}")

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = CIFAR100(
        root=str(data_root), train=True, download=False, transform=train_transform
    )
    test_set = CIFAR100(
        root=str(data_root), train=False, download=False, transform=test_transform
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=True,
    )
    val_batches, alpha_info = prepare_alpha_eval_batches(
        trainset=train_set,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = resnet110(num_classes=100)
    _load_dense_checkpoint(model, pretrained_checkpoint, device)
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    initial_loss, initial_top1, initial_top5 = _evaluate(
        model, test_loader, criterion
    )
    print(
        f"Initial dense model: loss={initial_loss:.6f}, "
        f"Top-1={initial_top1:.2f}%, Top-5={initial_top5:.2f}%"
    )

    dummy_input = torch.rand(1, 3, 32, 32, device=device)
    oto = OTO(model=model, dummy_input=dummy_input)
    oto.visualize(view=False, out_dir=str(cache_dir))
    steps_per_epoch = len(train_loader)
    start_pruning_step = warmup_epochs * steps_per_epoch
    pruning_steps = pruning_epochs * steps_per_epoch
    optimizer = oto.hesso(
        variant="sgd",
        lr=learning_rate,
        first_momentum=0.9,
        second_momentum=0.0,
        dampening=0.0,
        weight_decay=1e-4,
        target_group_sparsity=target_group_sparsity,
        start_pruning_step=start_pruning_step,
        pruning_periods=pruning_epochs,
        pruning_steps=pruning_steps,
        device=device,
        seed=seed,
        pruning_schedule_mode=schedule_mode,
    )
    backend_mode = str(getattr(optimizer, "pruning_schedule_mode", ""))
    if backend_mode != schedule_mode:
        raise RuntimeError(
            f"HESSO schedule mismatch: requested={schedule_mode}, "
            f"backend={backend_mode}"
        )
    production_quota = [
        int(value) for value in optimizer.active_num_redundant_groups
    ]
    if not (
        len(production_quota) == pruning_epochs
        and sum(production_quota) == optimizer.target_num_redundant_groups
    ):
        raise RuntimeError("HESSO constructed an invalid production schedule.")
    if int(optimizer.pruning_period_duration) != steps_per_epoch:
        raise RuntimeError(
            "Matched-control protocol requires exactly one pruning period per epoch."
        )
    initial_group_count = int(optimizer.total_num_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs
    )
    (
        preflight_model,
        preflight_optimizer,
        preflight_period_index,
        preflight_quota,
    ) = _clone_no_new_pruning_control(model, optimizer)
    if preflight_period_index != 0 or preflight_quota != production_quota[0]:
        raise RuntimeError("Matched-control deepcopy preflight is misaligned.")
    if int(optimizer.active_num_redundant_groups[0]) != production_quota[0]:
        raise RuntimeError("Control preflight mutated the production schedule.")
    del preflight_model, preflight_optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        "[MatchedControl] scheduler-safe full-state clone/binding "
        "preflight passed."
    )

    metadata = {
        "protocol": "period_level_matched_control",
        "implementation_version": IMPLEMENTATION_VERSION,
        "damage_definition": (
            "production_end_loss_minus_no_new_pruning_control_end_loss"
        ),
        "schedule_mode": schedule_mode,
        "seed": seed,
        "device": str(device),
        "data_root": str(data_root),
        "pretrained_checkpoint": str(pretrained_checkpoint),
        "batch_size": batch_size,
        "main_num_workers": num_workers,
        "eval_num_workers": eval_num_workers,
        "warmup_epochs": warmup_epochs,
        "pruning_epochs": pruning_epochs,
        "pruning_periods": pruning_epochs,
        "steps_per_period": steps_per_epoch,
        "learning_rate": learning_rate,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "target_group_sparsity": target_group_sparsity,
        "initial_group_count": initial_group_count,
        "total_pruning_budget": int(sum(production_quota)),
        "production_period_quota": production_quota,
        "alpha_config": dict(vars(ALPHA_CFG)),
        "alpha_subset": dict(alpha_info),
        "control_details": {
            "deepcopy_binding_preflight_passed": True,
            "same_model_and_hesso_state_at_period_start": True,
            "same_augmented_minibatch_tensors": True,
            "same_first_batch_gradients": True,
            "only_current_period_new_quota_is_zeroed": True,
            "production_branch_continues": True,
            "control_branch_discarded_after_period": True,
            "fixed_5pct_probe_used": False,
        },
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "run_id": run_id,
        "run_dir": str(run_dir),
    }
    _json_dump(run_dir / "run_metadata.json", metadata)

    production_records: list[dict[str, Any]] = []
    training_history: list[dict[str, Any]] = []
    last_production_eval = (initial_loss, initial_top1, initial_top5)
    model.train()

    for epoch_index in range(total_epochs):
        epoch_number = epoch_index + 1
        pruning_period_index = epoch_index - warmup_epochs
        is_pruning_epoch = 0 <= pruning_period_index < pruning_epochs
        period_number = pruning_period_index + 1 if is_pruning_epoch else 0
        period_start_eval = last_production_eval if is_pruning_epoch else None
        control_model = None
        control_optimizer = None
        production_selected_before: set[int] | None = None
        control_selected_before: set[int] | None = None
        expected_period_quota = (
            int(production_quota[pruning_period_index])
            if is_pruning_epoch
            else 0
        )
        control_train_loss_sum = 0.0
        control_train_steps = 0
        production_train_loss_sum = 0.0
        production_train_steps = 0
        t0 = time.time()

        print(
            f"\nEpoch {epoch_number}/{total_epochs} "
            f"({'period ' + str(period_number) if is_pruning_epoch else 'warmup'})"
        )
        for batch_index, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            production_outputs = model(inputs)
            production_loss = criterion(production_outputs, targets)
            production_loss.backward()
            production_train_loss_sum += float(production_loss.item())
            production_train_steps += 1

            entering_period = optimizer.will_enter_new_period_on_next_step()
            is_real_period_start = False
            if entering_period:
                optimizer.alpha_scheduler.update(
                    model=model,
                    criterion=criterion,
                    inputs=val_batches,
                    targets=None,
                    t_in_period=0,
                )
                next_step = int(optimizer.num_steps + 1)
                is_real_period_start = (
                    next_step >= int(optimizer.start_pruning_step)
                    and int(optimizer.curr_pruning_period)
                    < int(optimizer.pruning_periods)
                )
                if is_real_period_start:
                    if not is_pruning_epoch or batch_index != 0:
                        raise RuntimeError(
                            "Configured one-period-per-epoch alignment was violated."
                        )
                    if int(optimizer.curr_pruning_period) != pruning_period_index:
                        raise RuntimeError(
                            "HESSO period index is not aligned with the epoch."
                        )
                    production_selected_before = {
                        int(index) for index in optimizer.pruned_group_idxes
                    }
                    (
                        control_model,
                        control_optimizer,
                        control_period_index,
                        cloned_quota,
                    ) = _clone_no_new_pruning_control(model, optimizer)
                    if control_period_index != pruning_period_index:
                        raise RuntimeError("Control period index is misaligned.")
                    if cloned_quota != expected_period_quota:
                        raise RuntimeError(
                            "Control was cloned from an unexpected production quota."
                        )
                    control_selected_before = {
                        int(index) for index in control_optimizer.pruned_group_idxes
                    }
                    print(
                        f"[MatchedControl] period={period_number:03d}, "
                        f"production_quota={expected_period_quota}, "
                        "control_quota=0"
                    )

            optimizer.step()

            if control_model is not None:
                if is_real_period_start:
                    # The clone already contains the identical first-batch
                    # gradients and BatchNorm update from the production branch.
                    control_optimizer.step()
                    control_train_loss_sum += float(production_loss.item())
                    control_train_steps += 1
                else:
                    control_optimizer.zero_grad()
                    control_outputs = control_model(inputs)
                    control_loss = criterion(control_outputs, targets)
                    control_loss.backward()
                    control_optimizer.step()
                    control_train_loss_sum += float(control_loss.item())
                    control_train_steps += 1

                if int(control_optimizer.num_steps) != int(optimizer.num_steps):
                    raise RuntimeError(
                        "Production/control optimizer steps diverged immediately: "
                        f"period={period_number}, batch={batch_index}, "
                        f"production={int(optimizer.num_steps)}, "
                        f"control={int(control_optimizer.num_steps)}."
                    )
                if int(control_optimizer.curr_pruning_period) != int(
                    optimizer.curr_pruning_period
                ):
                    raise RuntimeError(
                        "Production/control period counters diverged immediately: "
                        f"period={period_number}, batch={batch_index}, "
                        f"production={int(optimizer.curr_pruning_period)}, "
                        f"control={int(control_optimizer.curr_pruning_period)}."
                    )

        scheduler.step()

        production_eval = _evaluate(model, test_loader, criterion)
        metrics = optimizer.compute_metrics()
        epoch_seconds = max(time.time() - t0, 1e-8)
        history_row: dict[str, Any] = {
            "schedule_mode": schedule_mode,
            "epoch": epoch_number,
            "stage": "pruning" if is_pruning_epoch else "warmup",
            "pruning_period": period_number,
            "scheduled_period_quota": expected_period_quota,
            "global_step": int(optimizer.num_steps),
            "learning_rate_after_scheduler": float(optimizer.param_groups[0]["lr"]),
            "production_train_loss": float(
                production_train_loss_sum / max(production_train_steps, 1)
            ),
            "production_test_loss": float(production_eval[0]),
            "production_test_top1": float(production_eval[1]),
            "production_test_top5": float(production_eval[2]),
            "group_sparsity": float(metrics.group_sparsity),
            "num_pruned_groups": int(metrics.num_pruned_groups),
            "epoch_seconds": float(epoch_seconds),
        }

        if is_pruning_epoch:
            if (
                control_model is None
                or control_optimizer is None
                or production_selected_before is None
                or control_selected_before is None
                or period_start_eval is None
            ):
                raise RuntimeError("Pruning period completed without a control branch.")
            if control_train_steps != steps_per_epoch:
                raise RuntimeError("Control did not consume every period minibatch.")
            if int(control_optimizer.num_steps) != int(optimizer.num_steps):
                raise RuntimeError(
                    "Production and control optimizer steps diverged: "
                    f"production={int(optimizer.num_steps)}, "
                    f"control={int(control_optimizer.num_steps)}."
                )
            if int(control_optimizer.curr_pruning_period) != int(
                optimizer.curr_pruning_period
            ):
                raise RuntimeError("Production and control period counters diverged.")

            production_selected_after = {
                int(index) for index in optimizer.pruned_group_idxes
            }
            control_selected_after = {
                int(index) for index in control_optimizer.pruned_group_idxes
            }
            newly_selected_production = sorted(
                production_selected_after - production_selected_before
            )
            newly_selected_control = sorted(
                control_selected_after - control_selected_before
            )
            if len(newly_selected_production) != expected_period_quota:
                raise RuntimeError(
                    f"Production selected {len(newly_selected_production)} groups, "
                    f"expected {expected_period_quota}."
                )
            if newly_selected_control:
                raise RuntimeError("No-new-pruning control selected new groups.")

            control_eval = _evaluate(control_model, test_loader, criterion)
            damage_loss = float(production_eval[0] - control_eval[0])
            damage_top1_drop = float(control_eval[1] - production_eval[1])
            damage_top5_drop = float(control_eval[2] - production_eval[2])
            quota_fraction_initial = float(
                expected_period_quota / max(initial_group_count, 1)
            )
            damage_per_1pct = (
                float(damage_loss * 0.01 / quota_fraction_initial)
                if expected_period_quota > 0
                else None
            )
            production_net_change = float(production_eval[0] - period_start_eval[0])
            control_training_change = float(control_eval[0] - period_start_eval[0])
            decomposition_error = float(
                production_net_change
                - (control_training_change + damage_loss)
            )
            record = {
                "schedule_mode": schedule_mode,
                "period": period_number,
                "epoch": epoch_number,
                "period_steps": steps_per_epoch,
                "initial_group_count": initial_group_count,
                "production_quota_groups": expected_period_quota,
                "production_quota_fraction_initial": quota_fraction_initial,
                "production_selected_global_indices": newly_selected_production,
                "period_start_loss": float(period_start_eval[0]),
                "period_start_top1": float(period_start_eval[1]),
                "period_start_top5": float(period_start_eval[2]),
                "production_end_loss": float(production_eval[0]),
                "production_end_top1": float(production_eval[1]),
                "production_end_top5": float(production_eval[2]),
                "control_end_loss": float(control_eval[0]),
                "control_end_top1": float(control_eval[1]),
                "control_end_top5": float(control_eval[2]),
                "production_damage_loss": damage_loss,
                "production_damage_top1_drop": damage_top1_drop,
                "production_damage_top5_drop": damage_top5_drop,
                "production_damage_loss_per_1pct_initial_groups": damage_per_1pct,
                "production_net_loss_change": production_net_change,
                "control_training_loss_change": control_training_change,
                "loss_decomposition_error": decomposition_error,
                "production_train_loss": float(
                    production_train_loss_sum / max(production_train_steps, 1)
                ),
                "control_train_loss": float(
                    control_train_loss_sum / max(control_train_steps, 1)
                ),
                "production_cumulative_selected_groups": len(
                    production_selected_after
                ),
                "control_new_selected_groups": 0,
                "group_sparsity_end": float(metrics.group_sparsity),
            }
            production_records.append(record)
            history_row.update({
                "control_train_loss": record["control_train_loss"],
                "control_test_loss": float(control_eval[0]),
                "control_test_top1": float(control_eval[1]),
                "control_test_top5": float(control_eval[2]),
                "production_damage_loss": damage_loss,
                "production_damage_top1_drop": damage_top1_drop,
            })
            print(
                f"[ProductionDamage] period={period_number:03d}, "
                f"quota={expected_period_quota}, "
                f"L_prod={production_eval[0]:.6f}, "
                f"L_ctrl={control_eval[0]:.6f}, d_p={damage_loss:.6g}, "
                f"Top1 drop={damage_top1_drop:.4f}"
            )
            del control_model, control_optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        training_history.append(history_row)
        last_production_eval = production_eval
        _persist_run(run_dir, production_records, training_history)
        model.train()

    if len(production_records) != pruning_epochs:
        raise RuntimeError(
            f"Expected {pruning_epochs} production-damage records, "
            f"found {len(production_records)}."
        )
    if int(optimizer.curr_pruning_period) != pruning_epochs:
        raise RuntimeError("Production HESSO did not complete every period.")

    oto.construct_subnet(out_dir=str(cache_dir))
    compressed_model_path = Path(oto.compressed_model_path)
    compressed_model = torch.load(
        str(compressed_model_path), map_location=device, weights_only=False
    )
    compressed_model.to(device)
    compressed_eval = _evaluate(compressed_model, test_loader, criterion)
    final_metrics = optimizer.compute_metrics()
    zero_quota_damage = [
        abs(float(row["production_damage_loss"]))
        for row in production_records
        if int(row["production_quota_groups"]) == 0
    ]
    final_summary = {
        "schedule_mode": schedule_mode,
        "run_id": run_id,
        "protocol": "period_level_matched_control",
        "implementation_version": IMPLEMENTATION_VERSION,
        "initial_loss": float(initial_loss),
        "initial_top1": float(initial_top1),
        "initial_top5": float(initial_top5),
        "compressed_loss": float(compressed_eval[0]),
        "compressed_top1": float(compressed_eval[1]),
        "compressed_top5": float(compressed_eval[2]),
        "group_sparsity": float(final_metrics.group_sparsity),
        "initial_group_count": initial_group_count,
        "total_pruning_budget": int(sum(production_quota)),
        "completed_periods": len(production_records),
        "mean_period_production_damage_loss": float(np.mean([
            float(row["production_damage_loss"]) for row in production_records
        ])),
        "sum_period_production_damage_loss": float(np.sum([
            float(row["production_damage_loss"]) for row in production_records
        ])),
        "max_abs_zero_quota_damage_loss": (
            float(max(zero_quota_damage)) if zero_quota_damage else None
        ),
        "compressed_model_path": str(compressed_model_path),
    }
    _json_dump(run_dir / "final_summary.json", final_summary)
    _persist_run(run_dir, production_records, training_history)
    _copy_latest_artifacts(run_dir, mode_root)
    print("\n=== Production-damage experiment complete ===")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python resnet110_cifar100_production_damage.py "
            "{uniform|cosine_front|both}"
        )
    mode = str(sys.argv[1]).strip().lower()
    if mode == "both":
        run_production_damage("uniform")
        run_production_damage("cosine_front")
    else:
        run_production_damage(mode)


if __name__ == "__main__":
    main()
