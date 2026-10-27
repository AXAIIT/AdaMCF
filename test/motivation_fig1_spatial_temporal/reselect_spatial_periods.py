"""Repair/materialize Fig. 1(a) selection for an existing all-period run.

This utility only reads already exported JSON/NPZ/CSV artifacts and never
imports the training stack. It is useful for runs produced before zero-quota
periods were excluded from candidate selection.
"""

from __future__ import annotations

import json
import shutil
import statistics
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = CURRENT_DIR / "outputs"


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _selection_key(row: dict) -> tuple[float, float, int]:
    return (
        float(row["balanced_disagreement_fraction"]),
        float(row["conflict_index"]),
        -int(row["period"]),
    )


def reselect(run_dir: Path) -> dict:
    summary_path = run_dir / "spatial_period_summary.json"
    selection_path = run_dir / "selection.json"
    records = _load_json(summary_path)
    old_selection = _load_json(selection_path)
    active = [row for row in records if int(row["actual_period_quota"]) > 0]
    if not active:
        raise RuntimeError("No period with actual_period_quota > 0 was found.")

    selected = max(active, key=_selection_key)
    active_conflicts = [float(row["conflict_index"]) for row in active]
    all_conflicts = [float(row["conflict_index"]) for row in records]
    active_median = statistics.median(active_conflicts)
    typical = min(
        active,
        key=lambda row: (
            abs(float(row["conflict_index"]) - active_median),
            int(row["period"]),
        ),
    )

    previous_period = int(old_selection["selected_for_fig1a"]["period"])
    backup_path = run_dir / "selection_before_active_quota_filter.json"
    if backup_path.exists():
        original_selection = _load_json(backup_path)
        original_period = int(original_selection["selected_for_fig1a"]["period"])
    else:
        original_period = int(
            old_selection.get("corrected_previous_selected_period", previous_period)
        )
    selected_period = int(selected["period"])
    checkpoint_name = None
    checkpoint_note = None
    checkpoint_metadata_path = run_dir / "best_candidate_checkpoint.json"
    if checkpoint_metadata_path.exists():
        checkpoint_metadata = _load_json(checkpoint_metadata_path)
        checkpoint_period = int(checkpoint_metadata["period"])
        if checkpoint_period == selected_period:
            checkpoint_name = str(checkpoint_metadata["checkpoint"])
        else:
            checkpoint_note = (
                f"No model checkpoint was saved for selected active period "
                f"{selected_period}; the existing legacy candidate checkpoint "
                f"belongs to zero-quota period {checkpoint_period}."
            )

    corrected = {
        **old_selection,
        "selection_rule": (
            "among periods with actual_period_quota > 0, maximize "
            "balanced_disagreement_fraction; ties prefer larger "
            "conflict_index then earlier period"
        ),
        "active_pruning_periods": len(active),
        "post_target_zero_quota_periods": len(records) - len(active),
        "selected_for_fig1a": selected,
        "typical_median_conflict": typical,
        "active_conflict_index_median": active_median,
        "active_conflict_index_mean": statistics.fmean(active_conflicts),
        "active_conflict_index_min": min(active_conflicts),
        "active_conflict_index_max": max(active_conflicts),
        "conflict_index_median": statistics.median(all_conflicts),
        "conflict_index_mean": statistics.fmean(all_conflicts),
        "conflict_index_min": min(all_conflicts),
        "conflict_index_max": max(all_conflicts),
        "selected_checkpoint": checkpoint_name,
        "checkpoint_note": checkpoint_note,
        "corrected_previous_selected_period": original_period,
    }

    if not backup_path.exists():
        shutil.copyfile(selection_path, backup_path)
    _write_json(selection_path, corrected)

    selected_dir = run_dir / f"period_{selected_period:03d}"
    spatial_dir = OUTPUT_ROOT / "spatial"
    spatial_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected_dir / "spatial_probe.npz", spatial_dir / "spatial_probe.npz")
    shutil.copyfile(
        selected_dir / "spatial_topk_overlap.csv",
        spatial_dir / "spatial_topk_overlap.csv",
    )
    selected_payload = _load_json(selected_dir / "spatial_probe.json")
    _write_json(
        spatial_dir / "spatial_probe.json",
        {"selection": corrected, **selected_payload},
    )

    print(
        f"Corrected {run_dir.name}: period {original_period} -> {selected_period}; "
        f"checkpoint={checkpoint_name or 'unavailable'}"
    )
    return corrected


def main() -> None:
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1]).expanduser().resolve()
    else:
        root = OUTPUT_ROOT / "spatial_periods"
        latest = _load_json(root / "latest_run.json")
        run_dir = root / latest["run_id"]
    reselect(run_dir)


if __name__ == "__main__":
    main()
