"""Legacy single-trajectory helper; not a substitute for Algorithm-2 reruns.

The temporal damage measurements are independent of the plotted Uniform
baseline.  This script preserves the 100 measured c_t values, rebuilds the
matched-budget Uniform schedule with cumulative Round differencing.  It cannot
replace a legacy cosine-front production trajectory with an Algorithm-2
trajectory; formal figures must rerun both production schedules.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = CURRENT_DIR / "outputs"


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dump_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _backup_once(path: Path) -> None:
    backup = path.with_name(f"{path.stem}.legacy_uniform_backup{path.suffix}")
    if not backup.exists():
        shutil.copy2(path, backup)


def _balanced_uniform(total_budget: int, periods: int) -> list[int]:
    if periods <= 0:
        raise ValueError("periods must be positive")
    return [
        int(math.floor((index + 1) * total_budget / periods + 0.5))
        - int(math.floor(index * total_budget / periods + 0.5))
        for index in range(periods)
    ]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return ranks


def _spearman_or_none(left: np.ndarray, right: np.ndarray):
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        return None
    value = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return value if math.isfinite(value) else None


def _read_damage(records_path: Path) -> np.ndarray:
    with records_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No temporal records in {records_path}")
    if not all(str(row.get("probed", "")).strip().lower() == "true" for row in rows):
        raise RuntimeError("Every temporal period must be probed before repair.")
    return np.asarray(
        [float(row["immediate_damage_per_1pct_initial_groups"]) for row in rows],
        dtype=np.float64,
    )


def _repair_payload(schedule: dict, damage: np.ndarray) -> tuple[dict, bool]:
    periods = [int(value) for value in schedule["periods"]]
    cosine = np.asarray(schedule["cosine_front_period_quota"], dtype=np.float64)
    production = np.asarray(schedule["production_period_quota"], dtype=np.float64)
    old_uniform = np.asarray(schedule["uniform_period_quota"], dtype=np.float64)
    if not (len(periods) == len(damage) == len(cosine) == len(production)):
        raise RuntimeError("Temporal damage and schedule lengths do not match.")
    if not np.array_equal(production, cosine):
        raise RuntimeError("Production schedule is not cosine_front; refusing repair.")

    total_budget = int(round(float(cosine.sum())))
    uniform = np.asarray(
        _balanced_uniform(total_budget, len(periods)), dtype=np.float64
    )
    if float(uniform.sum()) != float(cosine.sum()):
        raise RuntimeError("Repaired Uniform and cosine_front budgets do not match.")
    changed = not np.array_equal(old_uniform, uniform)

    uniform_cost = float(np.dot(damage, uniform) / uniform.sum())
    cosine_cost = float(np.dot(damage, cosine) / cosine.sum())
    diagnostics = dict(schedule.get("diagnostics", {}))
    repaired_legacy_at_least_once = bool(
        diagnostics.get("legacy_uniform_remainder_spike_repaired", False)
    ) or changed
    diagnostics.update({
        "spearman_uniform_quota_vs_negative_damage": _spearman_or_none(
            uniform, -damage
        ),
        "uniform_matched_budget_cost": uniform_cost,
        "cosine_front_matched_budget_cost": cosine_cost,
        "matched_budget_cost_gap_uniform_minus_cosine": (
            uniform_cost - cosine_cost
        ),
        "uniform_quota_min": int(uniform.min()),
        "uniform_quota_max": int(uniform.max()),
        "uniform_schedule_policy": "Algorithm-2 cumulative-Round balanced",
        "legacy_uniform_remainder_spike_repaired": repaired_legacy_at_least_once,
    })

    repaired = dict(schedule)
    repaired.update({
        "uniform_period_quota": [int(value) for value in uniform],
        "quota_source": (
            "Uniform rebuilt by Algorithm-2 cumulative Round differencing; "
            "cosine_front preserved from the legacy backend production schedule"
        ),
        "same_total_budget": True,
        "total_pruning_budget": total_budget,
        "uniform_schedule_policy": (
            "b_t=Round((t+1)B/T)-Round(tB/T); max quota difference <= 1"
        ),
        "diagnostics": diagnostics,
    })
    return repaired, changed


def main() -> None:
    canonical_dir = OUTPUT_ROOT / "temporal"
    records_path = canonical_dir / "temporal_probe_records.csv"
    canonical_schedule_path = canonical_dir / "temporal_schedule.json"
    canonical_probe_path = canonical_dir / "temporal_probe.json"
    latest_path = OUTPUT_ROOT / "temporal_periods" / "latest_run.json"
    for required in (
        records_path,
        canonical_schedule_path,
        canonical_probe_path,
        latest_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    damage = _read_damage(records_path)
    canonical_schedule = _load_json(canonical_schedule_path)
    repaired, changed = _repair_payload(canonical_schedule, damage)

    latest = _load_json(latest_path)
    run_dir = OUTPUT_ROOT / "temporal_periods" / latest["run_id"]
    run_schedule_path = run_dir / "temporal_schedule.json"
    run_diagnostics_path = run_dir / "temporal_diagnostics.json"
    for required in (run_schedule_path, run_diagnostics_path):
        if not required.exists():
            raise FileNotFoundError(required)

    for path in (
        canonical_schedule_path,
        canonical_probe_path,
        run_schedule_path,
        run_diagnostics_path,
    ):
        _backup_once(path)

    _dump_json(canonical_schedule_path, repaired)
    _dump_json(run_schedule_path, repaired)
    _dump_json(run_diagnostics_path, repaired["diagnostics"])

    canonical_probe = _load_json(canonical_probe_path)
    canonical_probe["diagnostics"] = repaired["diagnostics"]
    _dump_json(canonical_probe_path, canonical_probe)

    diagnostics = repaired["diagnostics"]
    print(
        "Temporal Uniform schedule repaired." if changed
        else "Temporal Uniform schedule was already balanced; diagnostics refreshed."
    )
    print(
        f"quota range={diagnostics['uniform_quota_min']}.."
        f"{diagnostics['uniform_quota_max']}, "
        f"C_uniform={diagnostics['uniform_matched_budget_cost']:.6g}, "
        f"C_cosine={diagnostics['cosine_front_matched_budget_cost']:.6g}, "
        f"gap={diagnostics['matched_budget_cost_gap_uniform_minus_cosine']:.6g}"
    )


if __name__ == "__main__":
    main()
