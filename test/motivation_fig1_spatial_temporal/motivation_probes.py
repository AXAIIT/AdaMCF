"""Empirical probes for the two motivations used by AdaMCF Fig. 1.

This module never changes the production HESSO implementation.  It builds a
temporary HESSO backend around a copied ResNet-110 checkpoint and reads the
same dependency groups and normalized criterion scores used by AdaMCF.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import pickle
import random
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
REFERENCE_EXPERIMENT_ROOT = (
    PROJECT_ROOT / "test" / "Ablation" / "resnet110_cifar100"
)

for path in (PROJECT_ROOT, REFERENCE_EXPERIMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from only_train_once import OTO
from model.model import resnet110


CRITERIA_ORDER = ("L1", "L2", "Fisher", "GW", "Trend")
OUTPUT_ROOT = CURRENT_DIR / "outputs"


def _env_int_list(name: str, default: Sequence[int]) -> List[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _env_str_list(name: str, default: Sequence[str]) -> List[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except (TypeError, RuntimeError, pickle.UnpicklingError):
        # Legacy project checkpoints may contain the trusted OTO optimizer
        # state (including TensorTransform enums) in addition to state_dict.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, Mapping) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def _build_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = resnet110(num_classes=100)
    _load_checkpoint(model, checkpoint_path, device)
    return model.to(device)


def _deterministic_transform() -> transforms.Compose:
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _subset_indices(length: int, size: int, seed: int, offset: int = 0) -> List[int]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(length, generator=generator).tolist()
    if not order:
        return []
    size = min(max(int(size), 1), length)
    offset = int(offset) % length
    if offset + size <= length:
        return order[offset:offset + size]
    wrapped = order[offset:] + order[: offset + size - length]
    return wrapped


def _build_eval_loader(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    subset_size: int,
    seed: int,
) -> DataLoader:
    dataset = CIFAR100(
        root=str(data_root),
        train=False,
        download=False,
        transform=_deterministic_transform(),
    )
    if 0 < subset_size < len(dataset):
        dataset = Subset(dataset, _subset_indices(len(dataset), subset_size, seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def _build_train_probe_loader(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    num_batches: int,
    seed: int,
    offset: int = 0,
) -> DataLoader:
    dataset = CIFAR100(
        root=str(data_root),
        train=True,
        download=False,
        transform=_deterministic_transform(),
    )
    subset_size = max(1, num_batches) * batch_size
    indices = _subset_indices(len(dataset), subset_size, seed, offset=offset)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def _cache_batches(loader: DataLoader, limit: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    batches = []
    for inputs, targets in loader:
        batches.append((inputs.clone(), targets.clone()))
        if len(batches) >= limit:
            break
    return batches


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    loss_sum = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_sum += float(loss.item()) * targets.size(0)
            top1 = outputs.argmax(dim=1)
            top5 = outputs.topk(5, dim=1).indices
            top1_correct += int((top1 == targets).sum().item())
            top5_correct += int((top5 == targets.unsqueeze(1)).any(dim=1).sum().item())
            total += targets.size(0)
    denom = max(total, 1)
    return loss_sum / denom, 100.0 * top1_correct / denom, 100.0 * top5_correct / denom


def _build_probe_backend(
    model: nn.Module,
    device: torch.device,
    target_sparsity: float,
    seed: int,
):
    dummy_input = torch.zeros(1, 3, 32, 32, device=device)
    oto = OTO(model=model, dummy_input=dummy_input)
    backend = oto.hesso(
        variant="sgd",
        lr=0.01,
        first_momentum=0.9,
        second_momentum=0.0,
        dampening=0.0,
        weight_decay=0.0,
        target_group_sparsity=target_sparsity,
        start_pruning_step=0,
        pruning_periods=1,
        pruning_steps=1,
        device=device,
        seed=seed,
    )
    return oto, backend


def _accumulate_score_gradients(
    model: nn.Module,
    backend,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int,
) -> None:
    model.eval()
    backend.zero_grad()
    used = 0
    for inputs, targets in loader:
        if used >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        used += 1
    if used == 0:
        raise RuntimeError("No score batches were consumed.")


def _collect_core_scores(model, backend, score_loader, criterion, device, score_batches):
    _accumulate_score_gradients(
        model=model,
        backend=backend,
        loader=score_loader,
        criterion=criterion,
        device=device,
        max_batches=score_batches,
    )
    if not hasattr(backend, "export_kendall_raw_scores"):
        raise RuntimeError("The current HESSO backend lacks export_kendall_raw_scores().")
    exported = backend.export_kendall_raw_scores(refresh=True)
    score_map = {
        name: np.asarray(exported["score_map"][name], dtype=np.float64)
        for name in CRITERIA_ORDER
    }
    return list(exported["group_ids"]), score_map


def _selection_masks(score_map: Mapping[str, np.ndarray], fraction: float):
    n_groups = len(next(iter(score_map.values())))
    k = min(n_groups, max(1, int(round(fraction * n_groups))))
    masks = []
    selected = {}
    for name in CRITERIA_ORDER:
        idx = np.argsort(score_map[name], kind="stable")[:k]
        mask = np.zeros(n_groups, dtype=np.uint8)
        mask[idx] = 1
        masks.append(mask)
        selected[name] = set(int(i) for i in idx.tolist())
    return k, np.stack(masks, axis=0), selected


def _pairwise_overlap(selected: Mapping[str, set], k: int):
    agreement = np.eye(len(CRITERIA_ORDER), dtype=np.float64)
    jaccard = np.eye(len(CRITERIA_ORDER), dtype=np.float64)
    rows = []
    for i, left in enumerate(CRITERIA_ORDER):
        for j, right in enumerate(CRITERIA_ORDER):
            if j <= i:
                continue
            intersection = len(selected[left] & selected[right])
            union = len(selected[left] | selected[right])
            topk_agreement = intersection / max(k, 1)
            jac = intersection / max(union, 1)
            agreement[i, j] = agreement[j, i] = topk_agreement
            jaccard[i, j] = jaccard[j, i] = jac
            rows.append({
                "criterion_a": left,
                "criterion_b": right,
                "shared_groups": intersection,
                "k": k,
                "topk_agreement": topk_agreement,
                "jaccard": jac,
            })
    return agreement, jaccard, rows


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _json_load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_spatial_probe() -> None:
    """Export group-level criterion masks for Fig. 1(a)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(os.environ.get("MOTIVATION_SEED", "42"))
    _set_seed(seed)

    data_root = Path(os.environ.get("MOTIVATION_DATA_ROOT", "/workspace/data/cifar100"))
    checkpoint = Path(os.environ.get(
        "MOTIVATION_SPATIAL_CKPT",
        str(REFERENCE_EXPERIMENT_ROOT / "model" / "best_model.pth"),
    ))
    batch_size = int(os.environ.get("MOTIVATION_BATCH_SIZE", "128"))
    num_workers = int(os.environ.get("MOTIVATION_NUM_WORKERS", "0"))
    score_batches = int(os.environ.get("MOTIVATION_SCORE_BATCHES", "12"))
    pruning_fraction = float(os.environ.get("MOTIVATION_SPATIAL_FRACTION", "0.30"))

    score_loader = _build_train_probe_loader(
        data_root, batch_size, num_workers, score_batches, seed
    )
    model = _build_model(checkpoint, device)
    criterion = nn.CrossEntropyLoss().to(device)
    oto, backend = _build_probe_backend(model, device, pruning_fraction, seed)
    group_ids, score_map = _collect_core_scores(
        model, backend, score_loader, criterion, device, score_batches
    )

    k, masks, selected = _selection_masks(score_map, pruning_fraction)
    agreement, jaccard, overlap_rows = _pairwise_overlap(selected, k)
    output_dir = OUTPUT_ROOT / "spatial"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "spatial_probe.npz",
        criteria=np.asarray(CRITERIA_ORDER, dtype=object),
        group_ids=np.asarray(group_ids, dtype=object),
        masks=masks,
        topk_agreement=agreement,
        jaccard=jaccard,
        **{f"score_{name}": score_map[name] for name in CRITERIA_ORDER},
    )
    with (output_dir / "spatial_topk_overlap.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overlap_rows[0].keys()))
        writer.writeheader()
        writer.writerows(overlap_rows)

    _json_dump(output_dir / "spatial_probe.json", {
        "metadata": {
            "model": "ResNet-110",
            "dataset": "CIFAR-100",
            "checkpoint": str(checkpoint),
            "seed": seed,
            "score_batches": score_batches,
            "pruning_fraction": pruning_fraction,
            "num_dependency_groups": len(group_ids),
            "k": k,
            "score_source": "HESSO.export_kendall_raw_scores(refresh=True)",
        },
        "criteria": list(CRITERIA_ORDER),
        "overlap": overlap_rows,
    })
    print(f"[Spatial] exported {len(group_ids)} dependency groups to {output_dir}")
    del model, oto, backend
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class SpatialPeriodTracker:
    """Record cross-criterion Top-K decisions at every real pruning period."""

    def __init__(
        self,
        pruning_fraction: float = 0.30,
        seed: int = 42,
        run_id: str | None = None,
    ) -> None:
        if not 0.0 < pruning_fraction < 1.0:
            raise ValueError("pruning_fraction must be in (0, 1)")
        self.pruning_fraction = float(pruning_fraction)
        self.seed = int(seed)
        default_run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_id = run_id or os.environ.get(
            "MOTIVATION_SPATIAL_RUN_ID", default_run_id
        )
        self.root = OUTPUT_ROOT / "spatial_periods"
        self.run_dir = self.root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.records: List[Dict] = []
        self.best_selection_key = None
        _json_dump(self.run_dir / "run_metadata.json", {
            "run_id": self.run_id,
            "model": "ResNet-110",
            "dataset": "CIFAR-100",
            "seed": self.seed,
            "pruning_fraction": self.pruning_fraction,
            "criteria": list(CRITERIA_ORDER),
            "selection_rule": (
                "among periods with actual_period_quota > 0, maximize "
                "balanced_disagreement_fraction (groups receiving 2-vs-3 "
                "criterion votes); ties prefer larger conflict_index then earlier period"
            ),
        })
        print(f"[SpatialPeriods] run directory: {self.run_dir}")

    @staticmethod
    def _global_indices(group_ids: Sequence[str]) -> np.ndarray:
        indices = []
        pattern = re.compile(r"::global_(\d+)$")
        for position, group_id in enumerate(group_ids):
            match = pattern.search(str(group_id))
            indices.append(int(match.group(1)) if match else position)
        if len(set(indices)) != len(indices):
            raise RuntimeError("Exported dependency-group global indices are not unique.")
        return np.asarray(indices, dtype=np.int64)

    def _write_summary(self) -> None:
        if not self.records:
            return
        fieldnames = list(self.records[0].keys())
        with (self.run_dir / "spatial_period_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
        _json_dump(self.run_dir / "spatial_period_summary.json", self.records)

    @torch.no_grad()
    def capture(self, backend, epoch: int, model: nn.Module | None = None) -> Dict:
        """Capture scores already refreshed by AlphaScheduler at a period boundary."""
        period_index = int(backend.curr_pruning_period)
        if period_index >= int(backend.pruning_periods):
            raise RuntimeError("Attempted to capture beyond the configured pruning periods.")
        exported = backend.export_kendall_raw_scores(refresh=False)
        all_group_ids = list(exported["group_ids"])
        global_indices = self._global_indices(all_group_ids)
        pruned = set(int(index) for index in backend.pruned_group_idxes)
        remaining_positions = np.asarray(
            [pos for pos, index in enumerate(global_indices) if int(index) not in pruned],
            dtype=np.int64,
        )
        if len(remaining_positions) < 2:
            raise RuntimeError("Fewer than two unpruned dependency groups remain.")

        group_ids = [all_group_ids[pos] for pos in remaining_positions]
        remaining_global_indices = global_indices[remaining_positions]
        score_map = {
            name: np.asarray(exported["score_map"][name], dtype=np.float64)[
                remaining_positions
            ]
            for name in CRITERIA_ORDER
        }
        k, masks, selected = _selection_masks(score_map, self.pruning_fraction)
        agreement, jaccard, overlap_rows = _pairwise_overlap(selected, k)

        upper = agreement[np.triu_indices(len(CRITERIA_ORDER), k=1)]
        nontrend = agreement[:4, :4][np.triu_indices(4, k=1)]
        trend = agreement[-1, :-1]
        votes = masks.sum(axis=0)
        any_disagreement = (votes > 0) & (votes < len(CRITERIA_ORDER))
        balanced_disagreement = (votes == 2) | (votes == 3)
        alpha = backend.alpha_scheduler.get_alpha().detach().cpu().tolist()
        alpha_payload = {
            key: float(value) for key, value in zip(backend._alpha_keys, alpha)
        }
        current_quota = int(backend.active_num_redundant_groups[period_index])

        period_number = period_index + 1
        period_dir = self.run_dir / f"period_{period_number:03d}"
        period_dir.mkdir(parents=True, exist_ok=False)
        np.savez(
            period_dir / "spatial_probe.npz",
            criteria=np.asarray(CRITERIA_ORDER, dtype=object),
            group_ids=np.asarray(group_ids, dtype=object),
            global_indices=remaining_global_indices,
            masks=masks,
            topk_agreement=agreement,
            jaccard=jaccard,
            period=np.asarray(period_number),
            epoch=np.asarray(int(epoch)),
            **{f"score_{name}": score_map[name] for name in CRITERIA_ORDER},
        )
        with (period_dir / "spatial_topk_overlap.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(overlap_rows[0].keys()))
            writer.writeheader()
            writer.writerows(overlap_rows)

        record = {
            "period": period_number,
            "epoch": int(epoch),
            "global_step_before_period": int(backend.num_steps + 1),
            "already_pruned_groups": len(pruned),
            "remaining_groups": len(group_ids),
            "probe_k": int(k),
            "probe_fraction": self.pruning_fraction,
            "actual_period_quota": current_quota,
            "mean_pairwise_agreement": float(np.mean(upper)),
            "conflict_index": float(1.0 - np.mean(upper)),
            "min_pairwise_agreement": float(np.min(upper)),
            "max_pairwise_agreement": float(np.max(upper)),
            "nontrend_mean_agreement": float(np.mean(nontrend)),
            "trend_mean_agreement": float(np.mean(trend)),
            "any_disagreement_fraction": float(np.mean(any_disagreement)),
            "balanced_disagreement_fraction": float(np.mean(balanced_disagreement)),
            "independent_random_expected_agreement": float(k / len(group_ids)),
            "alpha_json": json.dumps(alpha_payload, ensure_ascii=False, sort_keys=True),
        }
        _json_dump(period_dir / "spatial_probe.json", {
            "summary": record,
            "criteria": list(CRITERIA_ORDER),
            "overlap": overlap_rows,
            "selection_scope": "currently unpruned dependency groups only",
        })
        self.records.append(record)
        self._write_summary()
        selection_key = (
            record["balanced_disagreement_fraction"],
            record["conflict_index"],
            -record["period"],
        )
        # A period whose quota is zero is only a post-target observation: no
        # group will actually be removed there.  Keep it in the full trajectory,
        # but never promote it as the evidence checkpoint for a pruning decision.
        if model is not None and current_quota > 0 and (
            self.best_selection_key is None or selection_key > self.best_selection_key
        ):
            self.best_selection_key = selection_key
            checkpoint_path = self.run_dir / "best_candidate_checkpoint.pth"
            torch.save({
                "state_dict": {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                },
                "period": period_number,
                "epoch": int(epoch),
                "selection_summary": record,
                "run_id": self.run_id,
            }, checkpoint_path)
            _json_dump(self.run_dir / "best_candidate_checkpoint.json", {
                "checkpoint": checkpoint_path.name,
                "period": period_number,
                "epoch": int(epoch),
                "selection_summary": record,
            })
            print(
                f"[SpatialPeriods] updated best candidate checkpoint at "
                f"period {period_number:03d}"
            )
        print(
            f"[SpatialPeriods] period={period_number:03d}, "
            f"remaining={len(group_ids)}, K={k}, "
            f"conflict_index={record['conflict_index']:.4f}, "
            f"balanced={record['balanced_disagreement_fraction']:.4f}"
        )
        return record

    def finalize(self, expected_periods: int | None = None) -> Dict:
        self._write_summary()
        if not self.records:
            raise RuntimeError("No spatial periods were captured.")

        eligible_records = [
            row for row in self.records if int(row["actual_period_quota"]) > 0
        ]
        if not eligible_records:
            raise RuntimeError(
                "No period with actual_period_quota > 0 was captured; "
                "there is no real pruning-decision checkpoint to select."
            )

        selected = max(
            eligible_records,
            key=lambda row: (
                row["balanced_disagreement_fraction"],
                row["conflict_index"],
                -row["period"],
            ),
        )
        conflict_values = np.asarray(
            [row["conflict_index"] for row in self.records], dtype=np.float64
        )
        eligible_conflict_values = np.asarray(
            [row["conflict_index"] for row in eligible_records], dtype=np.float64
        )
        median_value = float(np.median(eligible_conflict_values))
        typical = min(
            eligible_records,
            key=lambda row: (abs(row["conflict_index"] - median_value), row["period"]),
        )
        selected_period = int(selected["period"])
        selected_dir = self.run_dir / f"period_{selected_period:03d}"
        spatial_dir = OUTPUT_ROOT / "spatial"
        spatial_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            selected_dir / "spatial_probe.npz", spatial_dir / "spatial_probe.npz"
        )
        shutil.copyfile(
            selected_dir / "spatial_topk_overlap.csv",
            spatial_dir / "spatial_topk_overlap.csv",
        )
        with (selected_dir / "spatial_probe.json").open("r", encoding="utf-8") as handle:
            selected_payload = json.load(handle)
        selection_payload = {
            "run_id": self.run_id,
            "captured_periods": len(self.records),
            "expected_periods": expected_periods,
            "complete": expected_periods is None or len(self.records) == expected_periods,
            "selection_rule": (
                "among periods with actual_period_quota > 0, maximize "
                "balanced_disagreement_fraction; ties prefer larger "
                "conflict_index then earlier period"
            ),
            "active_pruning_periods": len(eligible_records),
            "post_target_zero_quota_periods": len(self.records) - len(eligible_records),
            "selected_for_fig1a": selected,
            "typical_median_conflict": typical,
            "active_conflict_index_median": median_value,
            "active_conflict_index_mean": float(np.mean(eligible_conflict_values)),
            "active_conflict_index_min": float(np.min(eligible_conflict_values)),
            "active_conflict_index_max": float(np.max(eligible_conflict_values)),
            "conflict_index_median": float(np.median(conflict_values)),
            "conflict_index_mean": float(np.mean(conflict_values)),
            "conflict_index_min": float(np.min(conflict_values)),
            "conflict_index_max": float(np.max(conflict_values)),
            "selected_checkpoint": (
                "best_candidate_checkpoint.pth"
                if (self.run_dir / "best_candidate_checkpoint.pth").exists()
                else None
            ),
        }
        best_metadata_path = self.run_dir / "best_candidate_checkpoint.json"
        if best_metadata_path.exists():
            with best_metadata_path.open("r", encoding="utf-8") as handle:
                best_metadata = json.load(handle)
            if int(best_metadata["period"]) != selected_period:
                raise RuntimeError(
                    "Best-candidate checkpoint period does not match final selection."
                )
        _json_dump(self.run_dir / "selection.json", selection_payload)
        _json_dump(self.root / "latest_run.json", {
            "run_id": self.run_id,
            "selection_file": f"{self.run_id}/selection.json",
        })
        _json_dump(spatial_dir / "spatial_probe.json", {
            "selection": selection_payload,
            **selected_payload,
        })
        print(
            f"[SpatialPeriods] selected period {selected_period} with "
            f"balanced={selected['balanced_disagreement_fraction']:.4f}, "
            f"conflict_index={selected['conflict_index']:.4f}; "
            f"typical period={typical['period']}"
        )
        return selection_payload


def _strategy_score(score_map: Mapping[str, np.ndarray], strategy: str) -> np.ndarray:
    normalized = strategy.strip().lower()
    if normalized in {"static_fusion", "uniform", "uniform_static", "fixed_uniform"}:
        return np.mean(np.stack([score_map[name] for name in CRITERIA_ORDER]), axis=0)
    aliases = {
        "l1": "L1",
        "l2": "L2",
        "fisher": "Fisher",
        "gw": "GW",
        "trend": "Trend",
    }
    if normalized not in aliases:
        raise ValueError(
            "MOTIVATION_TEMPORAL_PERIOD_STRATEGY must be one of: "
            "static_fusion, l1, l2, fisher, gw, trend"
        )
    return score_map[aliases[normalized]].copy()


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks, including deterministic tie handling."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _spearman_or_none(left: np.ndarray, right: np.ndarray):
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        return None
    value = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return value if math.isfinite(value) else None


class TemporalPeriodTracker:
    """Measure fixed-absolute-budget damage along the real pruning trajectory.

    At every real pruning period, including periods whose production quota is
    zero, the tracker temporarily zeroes the same absolute number of currently
    unpruned dependency groups.  It evaluates the instantaneous loss on a
    deterministic subset and restores every modified tensor before the
    production optimizer step.  No extra recovery training is performed.
    """

    def __init__(
        self,
        data_root: Path,
        probe_fraction: float = 0.05,
        strategy: str = "static_fusion",
        eval_subset: int = 10000,
        batch_size: int = 128,
        num_workers: int = 4,
        eval_seed: int = 2026,
        warmup_epochs: int = 20,
        pruning_epochs: int = 100,
        production_schedule_mode: str = "cosine_front",
        training_config: Mapping | None = None,
        run_id: str | None = None,
    ) -> None:
        if not 0.0 < probe_fraction < 1.0:
            raise ValueError("probe_fraction must be in (0, 1)")
        self.probe_fraction = float(probe_fraction)
        self.strategy = strategy.strip().lower()
        # Validate the strategy at construction time rather than halfway through a run.
        _strategy_score(
            {name: np.asarray([0.0], dtype=np.float64) for name in CRITERIA_ORDER},
            self.strategy,
        )
        self.eval_subset = int(eval_subset)
        self.eval_seed = int(eval_seed)
        schedule_aliases = {
            "cosine": "cosine_front",
            "cosine_front": "cosine_front",
            "uniform": "uniform",
            "balanced_uniform": "uniform",
        }
        requested_schedule = str(production_schedule_mode).strip().lower()
        if requested_schedule not in schedule_aliases:
            raise ValueError(
                "Temporal real-trajectory schedule must be cosine_front or uniform."
            )
        self.production_schedule_mode = schedule_aliases[requested_schedule]
        self.eval_loader = _build_eval_loader(
            Path(data_root),
            int(batch_size),
            int(num_workers),
            self.eval_subset,
            self.eval_seed,
        )
        self.criterion = None
        default_run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_id = run_id or os.environ.get(
            "MOTIVATION_TEMPORAL_RUN_ID", default_run_id
        )
        self.root = (
            OUTPUT_ROOT / "temporal_periods" / self.production_schedule_mode
        )
        self.run_dir = self.root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.records: List[Dict] = []
        self.production_quota: List[int] | None = None
        self.uniform_quota: List[int] | None = None
        self.cosine_front_quota: List[int] | None = None
        self._production_selected_seen: set[int] = set()
        _json_dump(self.run_dir / "run_metadata.json", {
            "run_id": self.run_id,
            "model": "ResNet-110",
            "dataset": "CIFAR-100",
            "time_axis": "real production pruning periods",
            "starting_state": "one final dense pretrained checkpoint",
            "production_schedule_mode": self.production_schedule_mode,
            "warmup_epochs": int(warmup_epochs),
            "pruning_epochs": int(pruning_epochs),
            "pruning_periods": int(pruning_epochs),
            "total_stage1_epochs": int(warmup_epochs + pruning_epochs),
            "training_config": dict(training_config or {}),
            "probe_fraction_of_initial_groups": self.probe_fraction,
            "probe_budget_policy": (
                "same absolute group count at every period, rounded once from "
                "the initial dependency-group count"
            ),
            "fixed_probe_strategy": self.strategy,
            "eval_subset": self.eval_subset,
            "eval_seed": self.eval_seed,
            "measurement": "instantaneous loss damage per 1% initial groups",
            "trajectory_policy": (
                "temporary dependency-group zeroing is fully restored before "
                "the production optimizer step"
            ),
        })
        print(f"[TemporalPeriods] run directory: {self.run_dir}")

    def _write_summary(self) -> None:
        if not self.records:
            return
        fieldnames = list(self.records[0].keys())
        with (self.run_dir / "temporal_period_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
        _json_dump(self.run_dir / "temporal_period_summary.json", self.records)

    @staticmethod
    def _build_remaining_plan(backend, fixed_scores: np.ndarray, fraction: float):
        if not np.all(np.isfinite(fixed_scores)):
            raise RuntimeError("Temporal fixed-strategy scores contain NaN or Inf.")
        pruned = set(int(index) for index in backend.pruned_group_idxes)
        all_global = []
        group_by_global = {}
        for group in backend.param_groups:
            if not group.get("is_prunable", False) or group.get("is_auxiliary", False):
                continue
            global_idxes = np.asarray(group["global_idxes"], dtype=np.int64)
            for local_idx, global_idx in enumerate(global_idxes.tolist()):
                all_global.append(int(global_idx))
                group_by_global[int(global_idx)] = (group, int(local_idx))
        if len(all_global) != len(fixed_scores):
            raise RuntimeError(
                f"Mapped {len(all_global)} dependency groups but received "
                f"{len(fixed_scores)} temporal scores."
            )
        if len(set(all_global)) != len(all_global):
            raise RuntimeError("Dependency-group global indices are not unique.")
        score_by_global = {
            int(global_idx): float(fixed_scores[position])
            for position, global_idx in enumerate(all_global)
        }

        remaining = np.asarray(
            [index for index in all_global if index not in pruned], dtype=np.int64
        )
        initial_count = len(all_global)
        if len(remaining) < 2:
            raise RuntimeError("Fewer than two unpruned dependency groups remain.")
        k = max(1, int(round(fraction * initial_count)))
        if len(remaining) < k:
            raise RuntimeError(
                f"Only {len(remaining)} groups remain, fewer than the fixed "
                f"absolute probe budget k={k}."
            )
        remaining_scores = np.asarray(
            [score_by_global[int(index)] for index in remaining], dtype=np.float64
        )
        order = np.argsort(remaining_scores, kind="stable")[:k]
        selected_global = remaining[order]
        selected_scores = remaining_scores[order]

        plan = {}
        for global_idx in selected_global.tolist():
            group, local_idx = group_by_global[int(global_idx)]
            key = id(group)
            plan.setdefault(key, {"group": group, "idxes": []})
            plan[key]["idxes"].append(local_idx)
        return (
            plan,
            selected_global,
            selected_scores,
            int(k),
            int(len(remaining)),
            int(initial_count),
        )

    def _initialize_schedules(self, backend) -> None:
        if self.production_quota is not None:
            return
        backend_schedule = str(
            getattr(backend, "pruning_schedule_mode", "cosine_front")
        ).strip().lower()
        if backend_schedule != self.production_schedule_mode:
            raise RuntimeError(
                f"Temporal tracker expects production schedule "
                f"{self.production_schedule_mode!r}, but HESSO uses "
                f"{backend_schedule!r}."
            )
        periods = int(backend.pruning_periods)
        self.production_quota = [
            int(value) for value in backend.active_num_redundant_groups
        ]
        if len(self.production_quota) != periods:
            raise RuntimeError(
                "Production quota length does not match configured pruning periods."
            )
        total_budget = int(sum(self.production_quota))
        self.uniform_quota = [
            int(value)
            for value in backend._build_period_quota(
                total_budget, periods, mode="uniform"
            )
        ]
        self.cosine_front_quota = [
            int(value)
            for value in backend._build_period_quota(
                total_budget, periods, mode="cosine_front"
            )
        ]
        if not (
            sum(self.uniform_quota)
            == sum(self.cosine_front_quota)
            == total_budget
        ):
            raise RuntimeError("Matched-budget schedule construction failed.")
        expected_production = (
            self.uniform_quota
            if self.production_schedule_mode == "uniform"
            else self.cosine_front_quota
        )
        if self.production_quota != expected_production:
            raise RuntimeError(
                "The HESSO production quota does not match the configured "
                f"{self.production_schedule_mode} schedule."
            )

    @torch.no_grad()
    def capture(self, backend, model: nn.Module, epoch: int) -> Dict:
        period_index = int(backend.curr_pruning_period)
        if period_index >= int(backend.pruning_periods):
            raise RuntimeError("Attempted to capture beyond configured pruning periods.")
        self._initialize_schedules(backend)
        period_number = period_index + 1
        current_quota = int(backend.active_num_redundant_groups[period_index])
        pruned_count = len(set(int(index) for index in backend.pruned_group_idxes))

        common = {
            "period": period_number,
            "epoch": int(epoch),
            "production_schedule_mode": self.production_schedule_mode,
            "global_step_before_period": int(backend.num_steps + 1),
            "scheduled_period_quota": current_quota,
            "already_pruned_groups": pruned_count,
        }
        period_dir = self.run_dir / f"period_{period_number:03d}"
        period_dir.mkdir(parents=True, exist_ok=False)

        # Probe every period, including zero-quota periods.  Uniform pruning
        # still allocates budget late in training, so its matched-budget cost
        # cannot be computed if late-period damage is missing.
        # AlphaScheduler has just refreshed grad variants and importance scores
        # at this exact boundary.  Reusing that cache is important: refresh=True
        # would update SGD momentum buffers a second time and perturb the real
        # production trajectory that this diagnostic is meant to observe.
        exported = backend.export_kendall_raw_scores(refresh=False)
        score_map = {
            name: np.asarray(exported["score_map"][name], dtype=np.float64)
            for name in CRITERIA_ORDER
        }
        fixed_scores = _strategy_score(score_map, self.strategy)
        (
            plan,
            selected_global,
            selected_scores,
            k,
            remaining_count,
            initial_count,
        ) = self._build_remaining_plan(backend, fixed_scores, self.probe_fraction)

        device = next(model.parameters()).device
        if self.criterion is None:
            self.criterion = nn.CrossEntropyLoss().to(device)
        was_training = model.training
        baseline_loss, baseline_top1, _ = _evaluate(
            model, self.eval_loader, self.criterion, device
        )
        backups = _apply_plan(backend, plan)
        try:
            probe_loss, probe_top1, _ = _evaluate(
                model, self.eval_loader, self.criterion, device
            )
        finally:
            backend._restore_from_backups(backups)
            model.train(was_training)
        for kind, parameter, idx_t, original in backups:
            if kind == "param_col":
                restored = parameter.data[:, idx_t, ...]
            else:
                restored = parameter.data[idx_t, ...]
            if not torch.equal(restored, original):
                raise RuntimeError(
                    f"Temporal probe failed to restore a {kind} tensor slice."
                )

        immediate_damage = float(probe_loss - baseline_loss)
        actual_fraction_initial = float(k / initial_count)
        damage_per_1pct = float(
            immediate_damage * 0.01 / actual_fraction_initial
        )
        if not all(math.isfinite(value) for value in (
            baseline_loss,
            probe_loss,
            immediate_damage,
            damage_per_1pct,
        )):
            raise RuntimeError("Temporal loss probe produced NaN or Inf.")
        record = {
            **common,
            "probed": True,
            "initial_groups": initial_count,
            "remaining_groups": remaining_count,
            "selected_probe_groups": k,
            "requested_probe_fraction_of_initial_groups": self.probe_fraction,
            "actual_probe_fraction_of_initial_groups": actual_fraction_initial,
            "fixed_strategy": self.strategy,
            "baseline_loss": float(baseline_loss),
            "probe_pruned_loss": float(probe_loss),
            "immediate_damage": immediate_damage,
            "immediate_damage_per_1pct_initial_groups": damage_per_1pct,
            "baseline_top1": float(baseline_top1),
            "probe_pruned_top1": float(probe_top1),
            "restoration_verified": True,
        }
        _json_dump(period_dir / "temporal_probe.json", {
            "summary": record,
            "selected_global_indices": [int(index) for index in selected_global],
            "selected_fixed_scores": [float(value) for value in selected_scores],
            "selection_scope": (
                "fixed absolute budget selected from currently unpruned "
                "dependency groups"
            ),
        })
        self.records.append(record)
        self._write_summary()
        print(
            f"[TemporalPeriods] period={period_number:03d}, quota={current_quota}, "
            f"remaining={remaining_count}, probe_k={k}, "
            f"damage/1%={damage_per_1pct:.6g}"
        )
        return record

    def record_production_step(self, backend) -> Dict:
        """Record the actual group identities selected by the real HESSO step."""
        period_number = int(backend.curr_pruning_period)
        if period_number <= 0 or period_number > len(self.records):
            raise RuntimeError(
                "Production-step recording is not aligned with temporal capture."
            )
        current_selected = set(
            int(index) for index in backend.pruned_group_idxes
        )
        newly_selected = sorted(current_selected - self._production_selected_seen)
        self._production_selected_seen = current_selected
        record = self.records[period_number - 1]
        expected = int(record["scheduled_period_quota"])
        if len(newly_selected) != expected:
            raise RuntimeError(
                f"Period {period_number} selected {len(newly_selected)} production "
                f"groups, but its scheduled quota is {expected}."
            )
        record.update({
            "actual_production_selected_groups": len(newly_selected),
            "production_cumulative_selected_groups": len(current_selected),
            "production_selection_recorded": True,
        })
        period_dir = self.run_dir / f"period_{period_number:03d}"
        _json_dump(period_dir / "production_pruning_event.json", {
            "period": period_number,
            "production_schedule_mode": self.production_schedule_mode,
            "scheduled_quota": expected,
            "actual_selected_groups": len(newly_selected),
            "selected_global_indices": newly_selected,
            "cumulative_selected_groups": len(current_selected),
        })
        self._write_summary()
        return record

    def write_training_history(self, rows: Sequence[Mapping]) -> None:
        """Persist complete epoch-level state for later trajectory plots.

        The file is rewritten after every epoch so an interrupted 120-epoch run
        still leaves a self-describing partial trajectory.
        """
        if not rows:
            return
        serializable = [dict(row) for row in rows]
        fieldnames = list(serializable[0].keys())
        run_csv = self.run_dir / "training_history.csv"
        with run_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(serializable)
        _json_dump(self.run_dir / "training_history.json", serializable)

        canonical_dir = OUTPUT_ROOT / "temporal" / self.production_schedule_mode
        canonical_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run_csv, canonical_dir / "training_history.csv")
        _json_dump(canonical_dir / "training_history.json", serializable)

    def write_final_summary(self, summary: Mapping) -> None:
        payload = dict(summary)
        payload.setdefault("run_id", self.run_id)
        payload.setdefault(
            "production_schedule_mode", self.production_schedule_mode
        )
        _json_dump(self.run_dir / "final_summary.json", payload)
        canonical_dir = OUTPUT_ROOT / "temporal" / self.production_schedule_mode
        canonical_dir.mkdir(parents=True, exist_ok=True)
        _json_dump(canonical_dir / "final_summary.json", payload)

    def finalize(self, expected_periods: int | None = None) -> Dict:
        self._write_summary()
        if not self.records:
            raise RuntimeError("No temporal period was probed.")
        if not all(bool(row["probed"]) for row in self.records):
            raise RuntimeError("Every scheduled period must receive a damage probe.")
        if not all(
            bool(row.get("production_selection_recorded", False))
            for row in self.records
        ):
            raise RuntimeError(
                "Every temporal period must record its actual production selection."
            )
        if (
            self.production_quota is None
            or self.uniform_quota is None
            or self.cosine_front_quota is None
        ):
            raise RuntimeError("Temporal schedules were not initialized.")

        periods = np.asarray(
            [row["period"] for row in self.records], dtype=np.int64
        )
        damage = np.asarray(
            [
                row["immediate_damage_per_1pct_initial_groups"]
                for row in self.records
            ],
            dtype=np.float64,
        )
        production_quota = np.asarray(self.production_quota, dtype=np.float64)
        uniform_quota = np.asarray(self.uniform_quota, dtype=np.float64)
        cosine_quota = np.asarray(self.cosine_front_quota, dtype=np.float64)
        if not (
            len(periods)
            == len(production_quota)
            == len(uniform_quota)
            == len(cosine_quota)
        ):
            raise RuntimeError("Damage and schedule lengths do not match.")

        uniform_budget = float(np.sum(uniform_quota))
        cosine_budget = float(np.sum(cosine_quota))
        if uniform_budget <= 0.0 or uniform_budget != cosine_budget:
            raise RuntimeError("Uniform and cosine-front budgets are not matched.")
        uniform_cost = float(np.dot(damage, uniform_quota) / uniform_budget)
        cosine_cost = float(np.dot(damage, cosine_quota) / cosine_budget)
        production_cost = float(
            np.dot(damage, production_quota) / float(np.sum(production_quota))
        )
        third = max(1, len(damage) // 3)
        uniform_alignment = _spearman_or_none(uniform_quota, -damage)
        cosine_alignment = _spearman_or_none(cosine_quota, -damage)

        diagnostics = {
            "run_id": self.run_id,
            "production_schedule_mode": self.production_schedule_mode,
            "captured_periods": len(self.records),
            "expected_periods": expected_periods,
            "complete": expected_periods is None or len(self.records) == expected_periods,
            "probed_periods": len(self.records),
            "production_zero_quota_periods": int(np.sum(production_quota == 0)),
            "fixed_probe_strategy": self.strategy,
            "probe_fraction_of_initial_groups": self.probe_fraction,
            "unit_damage_metric": (
                "immediate_damage_per_1pct_initial_groups"
            ),
            "damage_mean": float(np.mean(damage)),
            "damage_std": float(np.std(damage)),
            "damage_min": float(np.min(damage)),
            "damage_max": float(np.max(damage)),
            "damage_range": float(np.max(damage) - np.min(damage)),
            "early_damage_mean": float(np.mean(damage[:third])),
            "late_damage_mean": float(np.mean(damage[-third:])),
            "late_minus_early_damage": float(
                np.mean(damage[-third:]) - np.mean(damage[:third])
            ),
            "spearman_uniform_quota_vs_negative_damage": uniform_alignment,
            "spearman_cosine_front_quota_vs_negative_damage": cosine_alignment,
            "uniform_matched_budget_cost": uniform_cost,
            "cosine_front_matched_budget_cost": cosine_cost,
            "production_weighted_probe_cost": production_cost,
            "matched_budget_cost_gap_uniform_minus_cosine": float(
                uniform_cost - cosine_cost
            ),
            "production_matches_cosine_front": bool(
                np.array_equal(production_quota, cosine_quota)
            ),
            "production_matches_uniform": bool(
                np.array_equal(production_quota, uniform_quota)
            ),
            "diagnostic_caveat": (
                "Both C(b) values hold the measured per-period damage fixed on "
                "the production trajectory. They explain timing alignment but "
                "do not replace end-to-end matched-budget schedule runs."
            ),
        }
        schedule_payload = {
            "run_id": self.run_id,
            "production_schedule_mode": self.production_schedule_mode,
            "periods": [int(value) for value in periods],
            "production_period_quota": [
                int(value) for value in production_quota
            ],
            "uniform_period_quota": [int(value) for value in uniform_quota],
            "cosine_front_period_quota": [
                int(value) for value in cosine_quota
            ],
            "quota_source": "HESSO._build_period_quota",
            "same_total_budget": True,
            "total_pruning_budget": int(uniform_budget),
            "cost_definition": "C(b)=sum_p c_p*b_p/sum_p b_p",
            "diagnostics": diagnostics,
        }
        _json_dump(self.run_dir / "temporal_diagnostics.json", diagnostics)
        _json_dump(self.run_dir / "temporal_schedule.json", schedule_payload)
        _json_dump(self.root / "latest_run.json", {
            "run_id": self.run_id,
            "diagnostics_file": f"{self.run_id}/temporal_diagnostics.json",
        })

        output_dir = OUTPUT_ROOT / "temporal" / self.production_schedule_mode
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            self.run_dir / "temporal_period_summary.csv",
            output_dir / "temporal_probe_records.csv",
        )
        _json_dump(output_dir / "temporal_probe.json", {
            "metadata": _json_load(self.run_dir / "run_metadata.json"),
            "records": self.records,
            "diagnostics": diagnostics,
        })
        _json_dump(output_dir / "temporal_schedule.json", schedule_payload)
        print(
            f"[TemporalPeriods] exported {len(self.records)} probes; "
            f"C_uniform={uniform_cost:.6g}, "
            f"C_cosine_front={cosine_cost:.6g}, "
            f"gap={uniform_cost - cosine_cost:.6g}"
        )
        return diagnostics


def _build_probe_plan(backend, global_scores: np.ndarray, fraction: float, align_mode: str):
    candidates = []
    cursor = 0
    for group in backend.param_groups:
        if not group.get("is_prunable", False) or group.get("is_auxiliary", False):
            continue
        num_groups = int(group["num_groups"])
        if align_mode == "group_param_size":
            total_numel = sum(int(param.numel()) for param in group["params"])
            unit_weight = float(total_numel) / max(num_groups, 1)
        elif align_mode == "group_count":
            unit_weight = 1.0
        else:
            raise ValueError("align_mode must be group_count or group_param_size")
        for local_idx in range(num_groups):
            if cursor >= len(global_scores):
                raise RuntimeError("Core score length does not match dependency-group mapping.")
            candidates.append({
                "score": float(global_scores[cursor]),
                "weight": unit_weight,
                "group": group,
                "local_idx": local_idx,
                "global_idx": cursor,
            })
            cursor += 1
    if cursor != len(global_scores):
        raise RuntimeError(
            f"Mapped {cursor} groups but received {len(global_scores)} core scores."
        )

    candidates.sort(key=lambda item: (item["score"], item["global_idx"]))
    total_weight = sum(item["weight"] for item in candidates)
    target_weight = fraction * total_weight
    picked = []
    picked_weight = 0.0
    for item in candidates:
        picked.append(item)
        picked_weight += item["weight"]
        if picked_weight >= target_weight:
            break

    plan = {}
    for item in picked:
        key = id(item["group"])
        plan.setdefault(key, {"group": item["group"], "idxes": []})
        plan[key]["idxes"].append(item["local_idx"])
    return plan, picked, picked_weight / max(total_weight, 1e-12), len(candidates)


def _apply_plan(backend, plan):
    backups = []
    for entry in plan.values():
        idxes = sorted(set(int(i) for i in entry["idxes"]))
        backups.extend(backend._apply_ratio_and_backup(entry["group"], idxes, ratio=0.0))
    return backups


@torch.no_grad()
def _enforce_zero_mask(backups) -> None:
    for kind, parameter, idx_t, _ in backups:
        if kind == "param_col":
            parameter.data[:, idx_t, ...] = 0
        else:
            parameter.data[idx_t, ...] = 0


def _recover_for_fixed_steps(
    model: nn.Module,
    batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    steps: int,
    lr: float,
    device: torch.device,
    zero_backups=None,
) -> None:
    if steps <= 0:
        return
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    criterion = nn.CrossEntropyLoss().to(device)
    model.train()
    for step in range(steps):
        inputs, targets = batches[step % len(batches)]
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        if zero_backups is not None:
            _enforce_zero_mask(zero_backups)


def _discover_snapshots(snapshot_dir: Path) -> Dict[int, Path]:
    pattern = re.compile(r"^chkpt_epoch_(\d+)\.pth$")
    result = {}
    if snapshot_dir.is_dir():
        for path in snapshot_dir.iterdir():
            match = pattern.match(path.name)
            if match:
                result[int(match.group(1))] = path
    return result


def _aggregate_quota_to_probe_stages(
    period_quota: Sequence[int], probe_epochs: Sequence[int]
) -> Tuple[List[int], List[int]]:
    """Aggregate a full pruning schedule around normalized probe positions.

    Each real pruning period is assigned to the nearest checkpoint on the
    normalized training-progress axis.  This keeps the five plotted bars tied
    to the production schedule instead of rebuilding an artificial 5-period
    schedule.
    """
    if not period_quota or not probe_epochs:
        raise ValueError("period_quota and probe_epochs must be non-empty")
    max_epoch = max(int(epoch) for epoch in probe_epochs)
    if max_epoch <= 0:
        raise ValueError("probe_epochs must contain a positive epoch")

    probe_progress = np.asarray(probe_epochs, dtype=np.float64) / max_epoch
    if np.any(np.diff(probe_progress) <= 0):
        raise ValueError("probe_epochs must be strictly increasing")
    period_progress = (
        np.arange(len(period_quota), dtype=np.float64) + 0.5
    ) / len(period_quota)
    assignments = np.abs(
        period_progress[:, None] - probe_progress[None, :]
    ).argmin(axis=1)
    stage_quota = [
        int(sum(period_quota[index] for index in np.flatnonzero(assignments == stage)))
        for stage in range(len(probe_epochs))
    ]
    stage_period_counts = [
        int(np.sum(assignments == stage)) for stage in range(len(probe_epochs))
    ]
    if sum(stage_quota) != sum(period_quota):
        raise RuntimeError("Stage quota aggregation changed the total pruning budget.")
    return stage_quota, stage_period_counts


def _finite_or_none(value: float):
    return float(value) if math.isfinite(float(value)) else None


@torch.no_grad()
def _structural_weight_zero_fraction(model: nn.Module) -> float:
    """Return exact-zero fraction for matrix/kernel parameters only."""
    zero_count = 0
    total_count = 0
    for parameter in model.parameters():
        if parameter.ndim < 2 or not parameter.is_floating_point():
            continue
        zero_count += int((parameter == 0).sum().item())
        total_count += int(parameter.numel())
    return zero_count / max(total_count, 1)


def run_legacy_dense_snapshot_temporal_probe() -> None:
    """Legacy dense-snapshot probe retained only for result archaeology.

    This is not the final Fig. 1(b) protocol. The final protocol is implemented
    by TemporalPeriodTracker on the real production pruning trajectory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(os.environ.get("MOTIVATION_DATA_ROOT", "/workspace/data/cifar100"))
    snapshot_dir = Path(os.environ.get(
        "MOTIVATION_SNAPSHOT_DIR",
        str(CURRENT_DIR / "model_snapshots"),
    ))
    epochs = _env_int_list("MOTIVATION_TEMPORAL_EPOCHS", [10, 30, 60, 90, 120])
    probe_seeds = _env_int_list("MOTIVATION_PROBE_SEEDS", [42, 43, 44])
    strategy = os.environ.get("MOTIVATION_TEMPORAL_STRATEGY", "static_fusion")
    fraction = float(os.environ.get("MOTIVATION_TEMPORAL_FRACTION", "0.05"))
    total_target_sparsity = float(os.environ.get("MOTIVATION_TOTAL_SPARSITY", "0.50"))
    schedule_periods = int(os.environ.get("MOTIVATION_SCHEDULE_PERIODS", "100"))
    align_mode = os.environ.get("MOTIVATION_ALIGN_MODE", "group_param_size")
    recovery_steps = int(os.environ.get("MOTIVATION_RECOVERY_STEPS", "50"))
    recovery_lr = float(os.environ.get("MOTIVATION_RECOVERY_LR", "0.01"))
    score_batches = int(os.environ.get("MOTIVATION_SCORE_BATCHES", "12"))
    batch_size = int(os.environ.get("MOTIVATION_BATCH_SIZE", "128"))
    num_workers = int(os.environ.get("MOTIVATION_NUM_WORKERS", "0"))
    eval_subset = int(os.environ.get("MOTIVATION_EVAL_SUBSET", "10000"))
    dense_zero_tolerance = float(
        os.environ.get("MOTIVATION_DENSE_ZERO_TOLERANCE", "0.0001")
    )

    available = _discover_snapshots(snapshot_dir)
    missing = [epoch for epoch in epochs if epoch not in available]
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoints {missing} in {snapshot_dir}. "
            "Run prepare_dense_snapshots.py first."
        )

    eval_loader = _build_eval_loader(
        data_root, batch_size, num_workers, eval_subset, seed=2026
    )
    criterion = nn.CrossEntropyLoss().to(device)
    records = []
    schedule_payload = None

    for probe_seed in probe_seeds:
        _set_seed(probe_seed)
        score_loader = _build_train_probe_loader(
            data_root, batch_size, num_workers, score_batches, probe_seed
        )
        recovery_loader = _build_train_probe_loader(
            data_root,
            batch_size,
            num_workers,
            max(recovery_steps, 1),
            probe_seed,
            offset=score_batches * batch_size,
        )
        recovery_batches = _cache_batches(recovery_loader, max(recovery_steps, 1))

        for epoch in epochs:
            checkpoint_path = available[epoch]
            print(f"[Temporal] seed={probe_seed}, epoch={epoch}, ckpt={checkpoint_path}")
            base_model = _build_model(checkpoint_path, device)
            zero_fraction = _structural_weight_zero_fraction(base_model)
            if zero_fraction > dense_zero_tolerance:
                raise RuntimeError(
                    f"Checkpoint {checkpoint_path} has {zero_fraction:.2%} exact zeros "
                    "in matrix/kernel weights and is likely already pruned. Temporal "
                    "stage-cost probes require dense snapshots so that training stage "
                    "is not confounded with cumulative sparsity. Run "
                    "prepare_dense_snapshots.py into a clean MOTIVATION_SNAPSHOT_DIR."
                )
            baseline_loss, baseline_top1, baseline_top5 = _evaluate(
                base_model, eval_loader, criterion, device
            )

            pruned_model = copy.deepcopy(base_model).to(device)
            oto, backend = _build_probe_backend(pruned_model, device, fraction, probe_seed)
            _, score_map = _collect_core_scores(
                pruned_model,
                backend,
                score_loader,
                criterion,
                device,
                score_batches,
            )
            fixed_scores = _strategy_score(score_map, strategy)
            plan, picked, actual_fraction, total_groups = _build_probe_plan(
                backend, fixed_scores, fraction, align_mode
            )
            backups = _apply_plan(backend, plan)
            immediate_loss, immediate_top1, _ = _evaluate(
                pruned_model, eval_loader, criterion, device
            )
            immediate_damage = immediate_loss - baseline_loss

            control_recovery_loss = None
            pruned_recovery_loss = None
            residual_damage = None
            if recovery_steps > 0:
                control_model = copy.deepcopy(base_model).to(device)
                _recover_for_fixed_steps(
                    control_model,
                    recovery_batches,
                    recovery_steps,
                    recovery_lr,
                    device,
                )
                _recover_for_fixed_steps(
                    pruned_model,
                    recovery_batches,
                    recovery_steps,
                    recovery_lr,
                    device,
                    zero_backups=backups,
                )
                control_recovery_loss, _, _ = _evaluate(
                    control_model, eval_loader, criterion, device
                )
                pruned_recovery_loss, _, _ = _evaluate(
                    pruned_model, eval_loader, criterion, device
                )
                residual_damage = pruned_recovery_loss - control_recovery_loss
                del control_model

            per_1pct_scale = 0.01 / max(actual_fraction, 1e-12)
            records.append({
                "probe_seed": probe_seed,
                "epoch": epoch,
                "checkpoint": str(checkpoint_path),
                "structural_weight_zero_fraction": zero_fraction,
                "strategy": strategy,
                "align_mode": align_mode,
                "requested_probe_fraction": fraction,
                "actual_probe_fraction": actual_fraction,
                "selected_groups": len(picked),
                "total_groups": total_groups,
                "baseline_loss": baseline_loss,
                "baseline_top1": baseline_top1,
                "baseline_top5": baseline_top5,
                "immediate_pruned_loss": immediate_loss,
                "immediate_pruned_top1": immediate_top1,
                "immediate_damage": immediate_damage,
                "immediate_damage_per_1pct": immediate_damage * per_1pct_scale,
                "recovery_steps": recovery_steps,
                "recovery_lr": recovery_lr,
                "control_recovery_loss": control_recovery_loss,
                "pruned_recovery_loss": pruned_recovery_loss,
                "residual_damage": residual_damage,
                "residual_damage_per_1pct": (
                    residual_damage * per_1pct_scale if residual_damage is not None else None
                ),
            })

            if schedule_payload is None:
                total_budget = max(1, int(round(total_target_sparsity * total_groups)))
                uniform_period = backend._build_period_quota(
                    total_budget, schedule_periods, mode="uniform"
                )
                cosine_period = backend._build_period_quota(
                    total_budget, schedule_periods, mode="cosine_front"
                )
                uniform, stage_period_counts = _aggregate_quota_to_probe_stages(
                    uniform_period, epochs
                )
                cosine, cosine_stage_period_counts = _aggregate_quota_to_probe_stages(
                    cosine_period, epochs
                )
                if stage_period_counts != cosine_stage_period_counts:
                    raise RuntimeError("Schedule modes produced inconsistent stage bins.")
                schedule_payload = {
                    "epochs": epochs,
                    "total_groups": total_groups,
                    "total_budget": total_budget,
                    "target_group_sparsity": total_target_sparsity,
                    "schedule_periods": schedule_periods,
                    "stage_period_counts": stage_period_counts,
                    "uniform_quota": uniform,
                    "cosine_front_quota": cosine,
                    "uniform_period_quota": uniform_period,
                    "cosine_front_period_quota": cosine_period,
                    "quota_source": "HESSO._build_period_quota",
                    "aggregation": (
                        "Each production pruning period is assigned to the nearest "
                        "probe checkpoint on normalized training progress."
                    ),
                }

            del base_model, pruned_model, oto, backend
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    output_dir = OUTPUT_ROOT / "temporal"
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with (output_dir / "temporal_probe_records.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    json_records = [
        {key: _finite_or_none(value) if isinstance(value, float) else value for key, value in row.items()}
        for row in records
    ]
    _json_dump(output_dir / "temporal_probe.json", {
        "metadata": {
            "model": "ResNet-110",
            "dataset": "CIFAR-100",
            "snapshot_dir": str(snapshot_dir),
            "epochs": epochs,
            "probe_seeds": probe_seeds,
            "fixed_strategy": strategy,
            "schedule_periods": schedule_periods,
            "score_batches": score_batches,
            "eval_subset": eval_subset,
            "dense_zero_tolerance": dense_zero_tolerance,
            "recovery_control": "unpruned control trained on identical fixed batches",
            "mask_policy": "zeroed dependency-group slices are re-zeroed after every recovery step",
        },
        "records": json_records,
        "schedule": schedule_payload,
    })
    _json_dump(output_dir / "temporal_schedule.json", schedule_payload)
    print(f"[Temporal] exported {len(records)} probe records to {output_dir}")
