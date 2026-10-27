"""Render Fig. 1(a), Fig. 1(b), and a two-column combined teaser.

The script intentionally has no theoretical/synthetic data fallback.  Missing
probe outputs are treated as an error so that a conceptual curve cannot be
mistaken for empirical evidence.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import ConnectionPatch, Patch, Rectangle


CURRENT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = CURRENT_DIR / "outputs"
FIGURE_DIR = OUTPUT_ROOT / "figures"

BLUE = "#3B82F6"
RED = "#DC2626"
GREEN = "#16803C"
ORANGE = "#D97706"
GRAY = "#4B5563"

# Display correction for a confirmed probe-evaluation failure.  Raw CSV/JSON
# measurements remain untouched.  Period 51 on the Uniform path produced an
# isolated probe loss of 5.34; paper figures replace its c_p with period 52.
TEMPORAL_DAMAGE_DISPLAY_REPLACEMENTS = {
    "uniform": {51: 52},
}


def _set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _load_spatial():
    # Prefer the effective selection from a complete all-period run.  This also
    # lets an already-synced run be rendered correctly if an older selection.json
    # accidentally chose a zero-quota, post-target period.
    loaded = _load_spatial_periods()
    if loaded is not None:
        _, selection, run_dir = loaded
        selected_period = int(selection["selected_for_fig1a"]["period"])
        path = run_dir / f"period_{selected_period:03d}" / "spatial_probe.npz"
    else:
        path = OUTPUT_ROOT / "spatial" / "spatial_probe.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run resnet110_cifar100_motivation.py spatial first."
        )
    return np.load(path, allow_pickle=True)


def _load_spatial_periods():
    root = OUTPUT_ROOT / "spatial_periods"
    latest_path = root / "latest_run.json"
    if not latest_path.exists():
        return None
    with latest_path.open("r", encoding="utf-8") as handle:
        latest = json.load(handle)
    run_dir = root / latest["run_id"]
    summary_path = run_dir / "spatial_period_summary.csv"
    selection_path = run_dir / "selection.json"
    if not summary_path.exists() or not selection_path.exists():
        raise FileNotFoundError(
            f"Incomplete spatial-period run: {run_dir}. Missing summary or selection."
        )
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        summary = list(csv.DictReader(handle))
    with selection_path.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    active_rows = [row for row in summary if int(row["actual_period_quota"]) > 0]
    if not active_rows:
        raise RuntimeError(
            f"No actual pruning period (actual_period_quota > 0) exists in {run_dir}."
        )
    effective = max(
        active_rows,
        key=lambda row: (
            float(row["balanced_disagreement_fraction"]),
            float(row["conflict_index"]),
            -int(row["period"]),
        ),
    )
    stored_period = int(selection["selected_for_fig1a"]["period"])
    effective_period = int(effective["period"])
    if stored_period != effective_period:
        warnings.warn(
            f"Stored selection period {stored_period} is not the best active "
            f"pruning period; rendering period {effective_period} instead. "
            "The stored model checkpoint still belongs to the old period.",
            RuntimeWarning,
        )
        selection = dict(selection)
        selection["selected_for_fig1a"] = effective
        selection["selected_checkpoint"] = None

    active_conflict = np.asarray(
        [float(row["conflict_index"]) for row in active_rows], dtype=np.float64
    )
    active_median = float(np.median(active_conflict))
    selection["typical_median_conflict"] = min(
        active_rows,
        key=lambda row: (
            abs(float(row["conflict_index"]) - active_median),
            int(row["period"]),
        ),
    )
    selection["active_pruning_periods"] = len(active_rows)
    return summary, selection, run_dir


def _load_temporal(schedule_mode: str | None = None):
    if schedule_mode is None:
        legacy_dir = OUTPUT_ROOT / "temporal"
        cosine_dir = legacy_dir / "cosine_front"
        data_dir = cosine_dir if cosine_dir.exists() else legacy_dir
    else:
        normalized = schedule_mode.strip().lower()
        aliases = {
            "cosine": "cosine_front",
            "cosine_front": "cosine_front",
            "uniform": "uniform",
            "balanced_uniform": "uniform",
        }
        if normalized not in aliases:
            raise ValueError("schedule_mode must be cosine_front or uniform")
        data_dir = OUTPUT_ROOT / "temporal" / aliases[normalized]
    records_path = data_dir / "temporal_probe_records.csv"
    schedule_path = data_dir / "temporal_schedule.json"
    if not records_path.exists() or not schedule_path.exists():
        raise FileNotFoundError(
            f"Missing temporal probe outputs in {data_dir}. Run the corresponding "
            "temporal_cosine or temporal_uniform trajectory first."
        )
    with records_path.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    with schedule_path.open("r", encoding="utf-8") as handle:
        schedule = json.load(handle)
    return records, schedule


def _load_training_history(schedule_mode: str):
    data_dir = OUTPUT_ROOT / "temporal" / schedule_mode
    history_path = data_dir / "training_history.csv"
    summary_path = data_dir / "final_summary.json"
    if not history_path.exists():
        raise FileNotFoundError(history_path)
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    summary = None
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    return history, summary


def _draw_spatial(mask_ax, overlap_ax, display_groups: int = 48) -> None:
    data = _load_spatial()
    criteria = [str(item) for item in data["criteria"].tolist()]
    group_ids = [str(item) for item in data["group_ids"].tolist()]
    masks = np.asarray(data["masks"], dtype=np.uint8)
    agreement = np.asarray(data["topk_agreement"], dtype=np.float64)

    display_groups = min(max(display_groups, 8), masks.shape[1])
    votes = masks.sum(axis=0)
    # Explicitly show the most disputed groups; full-set overlap is reported below.
    dispute = np.abs(votes - masks.shape[0] / 2.0)
    order = np.lexsort((np.arange(masks.shape[1]), dispute))[:display_groups]
    shown = masks[:, order]

    mask_ax.imshow(
        shown,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap([BLUE, RED]),
        vmin=0,
        vmax=1,
    )
    mask_ax.set_yticks(np.arange(len(criteria)), criteria)
    tick_positions = np.unique(np.linspace(0, display_groups - 1, 7, dtype=int))
    tick_labels = []
    for pos in tick_positions:
        global_idx = int(order[pos])
        match = group_ids[global_idx].rsplit("global_", 1)
        tick_labels.append(f"G{match[-1]}" if len(match) == 2 else f"G{global_idx}")
    mask_ax.set_xticks(tick_positions, tick_labels, rotation=25, ha="right")
    mask_ax.set_xlabel("Most disputed dependency groups (representative subset)")
    period_suffix = ""
    if "period" in data.files:
        period_suffix = f" (selected period {int(np.asarray(data['period']).item())})"
    mask_ax.set_title(
        f"What to prune? Conflicting Top-K group selections{period_suffix}", pad=6
    )
    mask_ax.set_xticks(np.arange(-0.5, display_groups, 1), minor=True)
    mask_ax.set_yticks(np.arange(-0.5, len(criteria), 1), minor=True)
    mask_ax.grid(which="minor", color="white", linewidth=0.45)
    mask_ax.tick_params(which="minor", bottom=False, left=False)
    mask_ax.legend(
        handles=[
            Patch(facecolor=RED, label="Prune"),
            Patch(facecolor=BLUE, label="Keep"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.36),
        ncol=2,
        frameon=False,
        handlelength=1.2,
    )

    image = overlap_ax.imshow(
        agreement,
        vmin=0,
        vmax=1,
        cmap="YlOrRd",
        interpolation="nearest",
        aspect="equal",
    )
    overlap_ax.set_xticks(np.arange(len(criteria)), criteria, rotation=30, ha="right")
    overlap_ax.set_yticks(np.arange(len(criteria)), criteria)
    overlap_ax.set_title(r"Full-set Top-K agreement  $|P_i \cap P_j| / K$", pad=6)
    for row in range(len(criteria)):
        for col in range(len(criteria)):
            value = agreement[row, col]
            color = "white" if value > 0.62 else "black"
            overlap_ax.text(col, row, f"{100 * value:.0f}%", ha="center", va="center", color=color)
    plt.colorbar(image, ax=overlap_ax, fraction=0.046, pad=0.04).set_label("Agreement")


def _draw_spatial_restructured(
    zoom_ax,
    overview_ax,
    overlap_ax,
    display_groups: int = 48,
    compact_style: bool = False,
    linked_zoom: bool = False,
    zoom_target_global=None,
    zoom_before_global=None,
    pattern_match: bool = False,
    connect_to_image_edges: bool = False,
    overview_panel_label: str = "b",
    overlap_panel_label: str = "c",
) -> None:
    """Draw a quota-explicit spatial-conflict figure without replacing Fig. 1(a)."""
    data = _load_spatial()
    criteria = [str(item) for item in data["criteria"].tolist()]
    group_ids = [str(item) for item in data["group_ids"].tolist()]
    masks = np.asarray(data["masks"], dtype=np.uint8)
    agreement = np.asarray(data["topk_agreement"], dtype=np.float64)

    num_criteria, num_groups = masks.shape
    selected_counts = masks.sum(axis=1).astype(int)
    if len(set(selected_counts.tolist())) != 1:
        raise RuntimeError(
            "Spatial criteria do not use the same Top-K diagnostic budget."
        )
    k = int(selected_counts[0])
    fraction = float(k / max(num_groups, 1))
    display_groups = min(max(display_groups, 8), num_groups)

    votes = masks.sum(axis=0)
    dispute = np.abs(votes - num_criteria / 2.0)
    zoom_start = None
    match_similarity = None
    if pattern_match:
        # Preserve the original non-contiguous disputed-group panel, then find
        # the contiguous full-set window with the smallest cellwise mismatch.
        zoom_order = np.lexsort((np.arange(num_groups), dispute))[:display_groups]
        target = masks[:, zoom_order]
        similarities = np.asarray(
            [
                np.mean(masks[:, start : start + display_groups] == target)
                for start in range(num_groups - display_groups + 1)
            ],
            dtype=np.float64,
        )
        zoom_start = int(np.argmax(similarities))
        match_similarity = float(similarities[zoom_start])
    elif linked_zoom:
        # A single zoom rectangle is only truthful for a contiguous slice of
        # the full-set matrix. Optionally center it on a requested global group.
        if zoom_target_global is not None or zoom_before_global is not None:
            global_numbers = []
            for position, group_id in enumerate(group_ids):
                match = group_id.rsplit("global_", 1)
                global_numbers.append(
                    int(match[-1]) if len(match) == 2 else int(position)
                )
            target_global = (
                int(zoom_before_global)
                if zoom_before_global is not None
                else int(zoom_target_global)
            )
            target_position = int(
                np.argmin(
                    np.abs(
                        np.asarray(global_numbers, dtype=np.int64)
                        - target_global
                    )
                )
            )
            if zoom_before_global is not None:
                zoom_start = max(
                    0,
                    min(num_groups - display_groups, target_position - display_groups),
                )
            else:
                zoom_start = max(
                    0,
                    min(
                        num_groups - display_groups,
                        target_position - display_groups // 2,
                    ),
                )
        else:
            balanced = ((votes == 2) | (votes == 3)).astype(np.float64)
            any_disagreement = (
                (votes > 0) & (votes < num_criteria)
            ).astype(np.float64)
            column_score = 2.0 * balanced + any_disagreement
            window_score = np.convolve(
                column_score,
                np.ones(display_groups, dtype=np.float64),
                mode="valid",
            )
            zoom_start = int(np.argmax(window_score))
        zoom_order = np.arange(
            zoom_start,
            zoom_start + display_groups,
            dtype=np.int64,
        )
    else:
        # These columns are deliberately non-contiguous: first select the strongest
        # 2-vs-3 vote conflicts, then use global order only as a deterministic tie break.
        zoom_order = np.lexsort((np.arange(num_groups), dispute))[:display_groups]
    shown = masks[:, zoom_order]

    zoom_ax.imshow(
        shown,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap([BLUE, RED]),
        vmin=0,
        vmax=1,
    )
    zoom_ax.set_yticks(np.arange(num_criteria))
    zoom_ax.set_yticklabels(criteria)
    zoom_tick_positions = np.unique(
        np.linspace(0, display_groups - 1, 7, dtype=int)
    )
    zoom_tick_labels = []
    for position in zoom_tick_positions:
        global_position = int(zoom_order[position])
        match = group_ids[global_position].rsplit("global_", 1)
        group_number = int(match[-1]) if len(match) == 2 else global_position
        zoom_tick_labels.append(
            f"G{group_number:03d}" if compact_style else f"G{group_number:06d}"
        )
    zoom_ax.set_xticks(zoom_tick_positions)
    zoom_ax.set_xticklabels(
        zoom_tick_labels,
        rotation=0 if compact_style else 22,
        ha="center" if compact_style else "right",
    )
    zoom_ax.set_xticks(np.arange(-0.5, display_groups, 1), minor=True)
    zoom_ax.set_yticks(np.arange(-0.5, num_criteria, 1), minor=True)
    zoom_ax.grid(which="minor", color="white", linewidth=0.42)
    zoom_ax.tick_params(which="minor", bottom=False, left=False)
    if compact_style and not linked_zoom:
        zoom_ax.text(
            0.5,
            -0.23,
            "(a) Partial group selections",
            transform=zoom_ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
        )
    elif not compact_style:
        zoom_ax.set_title(
            f"(a1) Zoom: {display_groups} most-disputed, non-contiguous groups",
            pad=5,
        )
        zoom_ax.set_xlabel(
            "Local red counts are not pruning quotas; full-set quotas are shown below",
            labelpad=2,
        )
    zoom_ax.legend(
        handles=[
            Patch(facecolor=RED, label="Prune"),
            Patch(facecolor=BLUE, label="Keep"),
        ],
        loc="upper right",
        ncol=2,
        frameon=True,
        framealpha=0.92,
        borderpad=0.25,
        handlelength=1.1,
    )

    overview_ax.imshow(
        masks,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap([BLUE, RED]),
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    overview_ax.set_yticks(np.arange(num_criteria))
    if compact_style:
        overview_ax.set_yticklabels(criteria)
    else:
        overview_ax.set_yticklabels(
            [
                f"{name}  ({count}/{num_groups})"
                for name, count in zip(criteria, selected_counts.tolist())
            ]
        )
    overview_tick_positions = np.unique(
        np.linspace(0, num_groups - 1, 6, dtype=int)
    )
    overview_tick_labels = []
    for position in overview_tick_positions:
        match = group_ids[int(position)].rsplit("global_", 1)
        group_number = int(match[-1]) if len(match) == 2 else int(position)
        overview_tick_labels.append(
            f"G{group_number:04d}" if compact_style else f"G{group_number:06d}"
        )
    overview_ax.set_xticks(overview_tick_positions)
    overview_ax.set_xticklabels(
        overview_tick_labels,
        rotation=0 if compact_style else 22,
        ha="center" if compact_style else "right",
    )
    overview_ax.set_yticks(np.arange(-0.5, num_criteria, 1), minor=True)
    overview_ax.grid(which="minor", axis="y", color="white", linewidth=0.7)
    overview_ax.tick_params(which="minor", left=False)
    if linked_zoom and zoom_start is not None:
        left = float(zoom_start) - 0.5
        right = float(zoom_start + display_groups) - 0.5
        overview_ax.add_patch(
            Rectangle(
                (left, -0.5),
                float(display_groups),
                float(num_criteria),
                fill=False,
                edgecolor=GRAY,
                linewidth=1.25,
                linestyle="--",
                zorder=6,
            )
        )
        connection_pairs = (
            ((left, (-0.5, num_criteria - 0.5)),
             (right, (display_groups - 0.5, num_criteria - 0.5)))
            if connect_to_image_edges
            else ((left, (0.0, -0.19)), (right, (1.0, -0.19)))
        )
        for overview_x, zoom_anchor in connection_pairs:
            connection = ConnectionPatch(
                xyA=zoom_anchor,
                coordsA=(
                    zoom_ax.transData
                    if connect_to_image_edges
                    else zoom_ax.transAxes
                ),
                xyB=(overview_x, -0.5),
                coordsB=overview_ax.transData,
                color=GRAY,
                linewidth=0.9,
                linestyle="--",
                alpha=0.85,
                clip_on=False,
                zorder=5,
            )
            zoom_ax.figure.add_artist(connection)
        if match_similarity is not None:
            overview_ax.text(
                0.5 * (left + right),
                -0.68,
                f"{100.0 * match_similarity:.1f}% match",
                ha="center",
                va="bottom",
                fontsize=8,
                color=GRAY,
                clip_on=False,
                zorder=8,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
            )
    if compact_style:
        overview_ax.text(
            0.5,
            -0.23,
            f"({overview_panel_label}) Full-set group selections",
            transform=overview_ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
        )
    else:
        overview_ax.set_xlabel(
            "All currently unpruned dependency groups in global order"
        )
        overview_ax.set_title(
            f"(a2) Full-set overview: identical budget "
            f"$K={k}$ of $N={num_groups}$ ({100.0 * fraction:.1f}%)",
            pad=5,
        )

    image = overlap_ax.imshow(
        agreement,
        vmin=0,
        vmax=1,
        cmap="YlOrRd",
        interpolation="nearest",
        aspect="equal",
    )
    overlap_ax.set_xticks(np.arange(num_criteria))
    overlap_ax.set_xticklabels(criteria, rotation=30, ha="right")
    overlap_ax.set_yticks(np.arange(num_criteria))
    overlap_ax.set_yticklabels(criteria)
    if compact_style:
        overlap_ax.text(
            0.5,
            -0.23,
            f"({overlap_panel_label}) Full-set Top-K agreement",
            transform=overlap_ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
        )
    else:
        overlap_ax.set_title(
            r"(a3) Full-set overlap  $|P_i \cap P_j|/K$",
            pad=5,
        )
    for row in range(num_criteria):
        for col in range(num_criteria):
            value = agreement[row, col]
            color = "white" if value > 0.62 else "black"
            overlap_ax.text(
                col,
                row,
                f"{100 * value:.0f}%",
                ha="center",
                va="center",
                color=color,
            )
    plt.colorbar(image, ax=overlap_ax, fraction=0.046, pad=0.04).set_label(
        "Agreement"
    )


def _draw_spatial_period_timeline(ax) -> None:
    loaded = _load_spatial_periods()
    if loaded is None:
        raise FileNotFoundError(
            "No all-period spatial run found. Run the spatial_periods mode first."
        )
    rows, selection, _ = loaded
    periods = np.asarray([int(row["period"]) for row in rows], dtype=np.int64)
    conflict = np.asarray([float(row["conflict_index"]) for row in rows])
    balanced = np.asarray(
        [float(row["balanced_disagreement_fraction"]) for row in rows]
    )
    any_disagreement = np.asarray(
        [float(row["any_disagreement_fraction"]) for row in rows]
    )
    quotas = np.asarray([int(row["actual_period_quota"]) for row in rows])
    selected = selection["selected_for_fig1a"]
    typical = selection["typical_median_conflict"]

    ax.plot(
        periods,
        conflict,
        color=RED,
        linewidth=1.8,
        label=r"Conflict index $1-\overline{A}_t$",
    )
    ax.plot(
        periods,
        balanced,
        color=BLUE,
        linewidth=1.6,
        label="Balanced 2-vs-3 disagreement",
    )
    ax.plot(
        periods,
        any_disagreement,
        color=GRAY,
        linewidth=1.2,
        linestyle="--",
        label="Any disagreement",
    )
    ax.axvline(
        int(selected["period"]),
        color=GREEN,
        linewidth=1.4,
        linestyle=":",
        label=f"Selected period {int(selected['period'])}",
    )
    ax.scatter(
        [int(typical["period"])],
        [float(typical["conflict_index"])],
        color=ORANGE,
        s=32,
        zorder=4,
        label=f"Median-conflict period {int(typical['period'])}",
    )
    active_periods = periods[quotas > 0]
    if len(active_periods) and int(active_periods.max()) < int(periods.max()):
        active_last = int(active_periods.max())
        ax.axvspan(
            active_last + 0.5,
            int(periods.max()),
            color="#E5E7EB",
            alpha=0.55,
            linewidth=0,
            label="Target reached (zero new-pruning quota)",
        )
    x_min = max(1, int(periods.min()))
    x_max = int(periods.max())
    ax.set_xlim(x_min - 0.5, x_max + 0.5 if x_max == x_min else x_max)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Pruning period")
    ax.set_ylabel("Disagreement rate")
    ax.set_title("Cross-criterion structural conflict over the complete scheduled trajectory")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=False, ncol=2, fontsize=8)


def _csv_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window == 1:
        return values.copy()
    result = np.empty_like(values, dtype=np.float64)
    half = window // 2
    for index in range(len(values)):
        start = max(0, index - half)
        end = min(len(values), index + half + 1)
        result[index] = float(np.mean(values[start:end]))
    return result


def _apply_temporal_damage_display_replacements(
    periods: np.ndarray,
    damage: np.ndarray,
    schedule_mode: str,
):
    corrected = np.asarray(damage, dtype=np.float64).copy()
    applied = []
    replacements = TEMPORAL_DAMAGE_DISPLAY_REPLACEMENTS.get(schedule_mode, {})
    for target_period, source_period in replacements.items():
        target = np.flatnonzero(periods == target_period)
        source = np.flatnonzero(periods == source_period)
        if len(target) != 1 or len(source) != 1:
            raise RuntimeError(
                f"Cannot apply temporal display replacement "
                f"{target_period}->{source_period}."
            )
        target_index = int(target[0])
        source_index = int(source[0])
        raw_value = float(corrected[target_index])
        replacement_value = float(corrected[source_index])
        corrected[target_index] = replacement_value
        applied.append({
            "period": int(target_period),
            "source_period": int(source_period),
            "raw_value": raw_value,
            "display_value": replacement_value,
        })
    return corrected, applied


def _draw_temporal(cost_ax, quota_ax) -> None:
    # The main mechanism figure uses the schedule-naive Uniform trajectory as
    # its reference c-vector.  The two-path supplement repeats the controlled
    # reweighting on both reference paths.
    reference_path = os.environ.get(
        "MOTIVATION_TEMPORAL_REFERENCE_PATH", "uniform"
    ).strip().lower()
    records, schedule = _load_temporal(reference_path)
    if not records or not all(
        _csv_bool(row.get("probed", "")) for row in records
    ):
        raise RuntimeError("Every scheduled temporal period must be probed.")
    periods = np.asarray(
        [int(row["period"]) for row in records], dtype=np.int64
    )
    damage = np.asarray(
        [
            float(row["immediate_damage_per_1pct_initial_groups"])
            for row in records
        ],
        dtype=np.float64,
    )
    scheduled_periods = np.asarray(schedule["periods"], dtype=np.int64)
    if not np.array_equal(periods, scheduled_periods):
        raise RuntimeError(
            "Temporal probe periods do not match temporal_schedule.json."
        )
    uniform_quota = np.asarray(
        schedule["uniform_period_quota"], dtype=np.float64
    )
    cosine_quota = np.asarray(
        schedule["cosine_front_period_quota"], dtype=np.float64
    )
    if not (
        len(periods) == len(uniform_quota) == len(cosine_quota)
        and float(uniform_quota.sum()) == float(cosine_quota.sum())
    ):
        raise RuntimeError("Temporal schedules are not matched-budget.")
    reference_mode = str(
        schedule.get("production_schedule_mode", "unknown")
    ).strip().lower()
    display_damage, _ = _apply_temporal_damage_display_replacements(
        periods, damage, reference_mode
    )
    smooth_window = int(os.environ.get("MOTIVATION_TEMPORAL_SMOOTH_WINDOW", "5"))
    smoothed = _rolling_mean(display_damage, smooth_window)
    quota_smooth_window = int(
        os.environ.get("MOTIVATION_TEMPORAL_QUOTA_SMOOTH_WINDOW", "7")
    )
    uniform_quota_smoothed = _rolling_mean(
        uniform_quota, quota_smooth_window
    )
    cosine_quota_smoothed = _rolling_mean(
        cosine_quota, quota_smooth_window
    )

    phase_size = max(1, len(periods) // 5)
    early_end = int(periods[phase_size - 1])
    late_start = int(periods[-phase_size])
    for axis in (cost_ax, quota_ax):
        axis.axvspan(
            int(periods.min()) - 0.5,
            early_end + 0.5,
            color="#DCFCE7",
            alpha=0.28,
            linewidth=0,
            zorder=0,
        )
        axis.axvspan(
            late_start - 0.5,
            int(periods.max()) + 0.5,
            color="#FEE2E2",
            alpha=0.28,
            linewidth=0,
            zorder=0,
        )

    cost_ax.plot(
        periods,
        display_damage,
        color=ORANGE,
        marker="o",
        markersize=2.5,
        linewidth=0.8,
        alpha=0.38,
        label=r"Measured $c_p$",
    )
    if smooth_window > 1:
        cost_ax.plot(
            periods,
            smoothed,
            color=RED,
            linewidth=1.8,
            label="Damage trend",
        )
    cost_ax.axhline(0.0, color="#9CA3AF", linewidth=0.8, linestyle="--")
    cost_ax.grid(axis="both", linestyle=":", alpha=0.5)
    cost_ax.set_xlabel("Pruning period")
    cost_ax.set_ylabel(r"Pruning damage  $c_p$")
    cost_ax.legend(frameon=False, loc="upper left", fontsize=8.4)
    cost_ax.text(
        0.5,
        -0.24,
        "(a) Stage-wise pruning damage",
        transform=cost_ax.transAxes,
        ha="center",
        va="top",
        fontsize=11.5,
    )

    quota_ax.step(
        periods,
        uniform_quota,
        where="mid",
        color=GRAY,
        linewidth=0.7,
        alpha=0.18,
    )
    quota_ax.step(
        periods,
        cosine_quota,
        where="mid",
        color=GREEN,
        linewidth=0.7,
        alpha=0.18,
    )
    quota_ax.plot(
        periods,
        uniform_quota_smoothed,
        label=r"Uniform $b_p$",
        color=GRAY,
        linewidth=2.0,
    )
    quota_ax.plot(
        periods,
        cosine_quota_smoothed,
        label=r"Cosine-front $b_p$",
        color=GREEN,
        linewidth=2.1,
    )
    quota_ax.fill_between(
        periods,
        0.0,
        cosine_quota_smoothed,
        color=GREEN,
        alpha=0.12,
    )
    tick_count = min(6, len(periods))
    tick_positions = np.unique(
        np.linspace(int(periods.min()), int(periods.max()), tick_count, dtype=int)
    )
    quota_ax.set_xticks(tick_positions)
    quota_ax.set_xlim(int(periods.min()) - 0.5, int(periods.max()) + 0.5)
    cost_ax.set_xlim(int(periods.min()) - 0.5, int(periods.max()) + 0.5)
    quota_ax.set_xlabel("Pruning period")
    quota_ax.set_ylabel(r"Pruning budget  $b_p$")
    quota_ax.grid(axis="y", linestyle=":", alpha=0.45)
    quota_ax.legend(frameon=False, ncol=1, loc="lower left", fontsize=8.4)
    quota_ax.text(
        0.5,
        -0.24,
        "(b) Matched-budget pruning schedules",
        transform=quota_ax.transAxes,
        ha="center",
        va="top",
        fontsize=11.5,
    )

    uniform_cost = float(
        np.dot(display_damage, uniform_quota) / uniform_quota.sum()
    )
    cosine_cost = float(
        np.dot(display_damage, cosine_quota) / cosine_quota.sum()
    )
    gap = uniform_cost - cosine_cost
    if gap <= 0.0:
        warnings.warn(
            "The measured fixed-path costs do not favor cosine-front over "
            "uniform. Keep the empirical result and weaken the temporal-mismatch claim.",
            RuntimeWarning,
        )


def _load_real_trajectory_arrays(schedule_mode: str):
    records, schedule = _load_temporal(schedule_mode)
    if not records or not all(
        _csv_bool(row.get("probed", "")) for row in records
    ):
        raise RuntimeError(
            f"Every {schedule_mode} trajectory period must be probed."
        )
    periods = np.asarray([int(row["period"]) for row in records], dtype=np.int64)
    damage = np.asarray(
        [
            float(row["immediate_damage_per_1pct_initial_groups"])
            for row in records
        ],
        dtype=np.float64,
    )
    production_quota = np.asarray(
        schedule["production_period_quota"], dtype=np.float64
    )
    if not (len(periods) == len(damage) == len(production_quota)):
        raise RuntimeError(f"Incomplete {schedule_mode} trajectory output.")
    stored_mode = str(schedule.get("production_schedule_mode", "")).strip().lower()
    if stored_mode != schedule_mode:
        raise RuntimeError(
            f"Expected {schedule_mode} trajectory, found {stored_mode!r}."
        )
    stored_periods = np.asarray(schedule["periods"], dtype=np.int64)
    if not np.array_equal(periods, stored_periods):
        raise RuntimeError(
            f"Stored {schedule_mode} periods do not match its probe records."
        )
    uniform_quota = np.asarray(
        schedule["uniform_period_quota"], dtype=np.float64
    )
    cosine_quota = np.asarray(
        schedule["cosine_front_period_quota"], dtype=np.float64
    )
    if not (
        len(uniform_quota) == len(cosine_quota) == len(periods)
        and float(uniform_quota.sum()) == float(cosine_quota.sum())
    ):
        raise RuntimeError("Stored schedules are not matched-budget.")
    expected_quota = (
        uniform_quota
        if schedule_mode == "uniform"
        else cosine_quota
    )
    if not np.array_equal(production_quota, expected_quota):
        raise RuntimeError(
            f"Stored production quota is not the declared {schedule_mode} schedule."
        )
    actual_quota = np.asarray(
        [int(row["actual_production_selected_groups"]) for row in records],
        dtype=np.float64,
    )
    if not all(
        _csv_bool(row.get("production_selection_recorded", ""))
        for row in records
    ):
        raise RuntimeError(
            f"The {schedule_mode} trajectory is missing real production events."
        )
    if not np.array_equal(actual_quota, production_quota):
        raise RuntimeError(
            f"The {schedule_mode} actual production selections do not match "
            "its scheduled quota."
        )
    diagnostics = schedule.get("diagnostics", {})
    if not bool(diagnostics.get("complete", False)):
        raise RuntimeError(f"The {schedule_mode} trajectory is incomplete.")
    return periods, damage, production_quota, schedule


def _load_production_damage_arrays(schedule_mode: str):
    source = OUTPUT_ROOT / "production_damage" / schedule_mode / "production_damage.json"
    if not source.exists():
        raise FileNotFoundError(
            f"Missing matched-control production damage for {schedule_mode}: {source}. "
            "Run resnet110_cifar100_production_damage.py for both schedules first."
        )
    with source.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"Invalid production-damage records: {source}")
    periods = np.asarray([int(row["period"]) for row in records], dtype=np.int64)
    damage = np.asarray(
        [float(row["production_damage_loss"]) for row in records],
        dtype=np.float64,
    )
    quota = np.asarray(
        [int(row["production_quota_groups"]) for row in records],
        dtype=np.float64,
    )
    if not (
        len(periods) == len(damage) == len(quota)
        and np.array_equal(periods, np.arange(1, len(periods) + 1))
    ):
        raise RuntimeError(f"Incomplete production-damage trajectory: {source}")
    stored_modes = {str(row.get("schedule_mode", "")) for row in records}
    if stored_modes != {schedule_mode}:
        raise RuntimeError(
            f"Expected only {schedule_mode} records, found {sorted(stored_modes)}."
        )
    return periods, damage, quota


def _draw_temporal_damage_only(cost_ax, include_cosine_front: bool = False) -> None:
    """Draw the measured stage-wise damage without the budget-allocation panel."""
    periods_u, damage_u, _, _ = _load_real_trajectory_arrays("uniform")
    display_u, _ = _apply_temporal_damage_display_replacements(
        periods_u, damage_u, "uniform"
    )
    smooth_window = int(os.environ.get("MOTIVATION_TEMPORAL_SMOOTH_WINDOW", "5"))
    smooth_u = _rolling_mean(display_u, smooth_window)

    early_end = int(round(float(periods_u.max()) * 0.20))
    late_start = int(round(float(periods_u.max()) * 0.80))
    cost_ax.axvspan(
        int(periods_u.min()) - 0.5,
        early_end + 0.5,
        color="#DCFCE7",
        alpha=0.28,
        linewidth=0,
        zorder=0,
    )
    cost_ax.axvspan(
        late_start - 0.5,
        int(periods_u.max()) + 0.5,
        color="#FEE2E2",
        alpha=0.28,
        linewidth=0,
        zorder=0,
    )

    uniform_measured_label = (
        r"Uniform measured $c_p$" if include_cosine_front else r"Measured $c_p$"
    )
    uniform_trend_label = (
        "Uniform damage trend" if include_cosine_front else "Damage trend"
    )
    cost_ax.plot(
        periods_u,
        display_u,
        color=ORANGE,
        marker="o",
        markersize=2.5,
        linewidth=0.8,
        alpha=0.38,
        label=uniform_measured_label,
    )
    cost_ax.plot(
        periods_u,
        smooth_u,
        color=RED,
        linewidth=1.9,
        label=uniform_trend_label,
    )

    if include_cosine_front:
        periods_c, damage_c, _, _ = _load_real_trajectory_arrays("cosine_front")
        if not np.array_equal(periods_u, periods_c):
            raise RuntimeError("Uniform and cosine trajectories use different periods.")
        display_c, _ = _apply_temporal_damage_display_replacements(
            periods_c, damage_c, "cosine_front"
        )
        smooth_c = _rolling_mean(display_c, smooth_window)
        cost_ax.plot(
            periods_c,
            display_c,
            color="#5FBF7A",
            marker="s",
            markersize=2.5,
            linewidth=0.7,
            alpha=0.32,
            label=r"Cosine-front measured $c_p$",
        )
        cost_ax.plot(
            periods_c,
            smooth_c,
            color=GREEN,
            linewidth=1.9,
            label="Cosine-front damage trend",
        )

    tick_count = min(6, len(periods_u))
    tick_positions = np.unique(
        np.linspace(int(periods_u.min()), int(periods_u.max()), tick_count, dtype=int)
    )
    cost_ax.set_xticks(tick_positions)
    cost_ax.set_xlim(int(periods_u.min()) - 0.5, int(periods_u.max()) + 0.5)
    cost_ax.axhline(0.0, color="#9CA3AF", linewidth=0.8, linestyle="--")
    cost_ax.grid(axis="both", linestyle=":", alpha=0.5)
    cost_ax.set_xlabel("Pruning period")
    cost_ax.set_ylabel(r"Pruning damage  $c_p$")
    cost_ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=8.2,
        ncol=2 if include_cosine_front else 1,
    )
    caption = (
        "Stage-wise pruning damage along two real schedules"
        if include_cosine_front
        else "Stage-wise pruning damage"
    )
    cost_ax.text(
        0.5,
        -0.25,
        caption,
        transform=cost_ax.transAxes,
        ha="center",
        va="top",
        fontsize=11.5,
    )


def _draw_real_trajectory_comparison(cost_ax, quota_ax) -> dict:
    uniform = _load_real_trajectory_arrays("uniform")
    cosine = _load_real_trajectory_arrays("cosine_front")
    periods_u, damage_u, quota_u, _ = uniform
    periods_c, damage_c, quota_c, _ = cosine
    if not np.array_equal(periods_u, periods_c):
        raise RuntimeError("Uniform and cosine trajectories use different periods.")
    if float(quota_u.sum()) != float(quota_c.sum()):
        raise RuntimeError("Real trajectories do not use the same total budget.")

    display_u, display_replacements_u = (
        _apply_temporal_damage_display_replacements(
            periods_u, damage_u, "uniform"
        )
    )
    display_c, display_replacements_c = (
        _apply_temporal_damage_display_replacements(
            periods_c, damage_c, "cosine_front"
        )
    )
    smooth_window = int(os.environ.get("MOTIVATION_TEMPORAL_SMOOTH_WINDOW", "5"))
    smooth_u = _rolling_mean(display_u, smooth_window)
    smooth_c = _rolling_mean(display_c, smooth_window)

    cost_ax.scatter(
        periods_u,
        display_u,
        color=GRAY,
        marker="o",
        s=9,
        alpha=0.32,
        label=r"Uniform measured $c_p$ (5% probe; P51 corrected)",
    )
    cost_ax.scatter(
        periods_c,
        display_c,
        color=GREEN,
        marker="o",
        s=9,
        alpha=0.32,
        label=r"Cosine-front measured $c_p$ (5% probe; raw)",
    )
    cost_ax.plot(
        periods_u,
        smooth_u,
        color=GRAY,
        linewidth=2.0,
        label=f"Uniform local trend ({smooth_window}-period average)",
    )
    cost_ax.plot(
        periods_c,
        smooth_c,
        color=GREEN,
        linewidth=2.0,
        label=f"Cosine-front local trend ({smooth_window}-period average)",
    )
    cost_ax.axhline(0.0, color="#9CA3AF", linewidth=0.8, linestyle="--")
    cost_ax.set_ylabel(r"Pruning damage  $c_p$")
    cost_ax.set_title(
        "Two real trajectories (their stage-cost paths are descriptive, not identical)",
        pad=6,
    )
    cost_ax.grid(True, linestyle=":", alpha=0.48)
    cost_ax.legend(
        frameon=False, fontsize=7.7, loc="upper left", ncol=2
    )

    quota_ax.step(
        periods_u,
        quota_u,
        where="mid",
        color=GRAY,
        linewidth=1.9,
        label="Balanced Uniform production quota",
    )
    quota_ax.step(
        periods_c,
        quota_c,
        where="mid",
        color=GREEN,
        linewidth=1.9,
        label="Cosine-front production quota",
    )
    quota_ax.fill_between(
        periods_c,
        0.0,
        quota_c,
        step="mid",
        color=GREEN,
        alpha=0.10,
    )
    quota_ax.set_xlabel("Real pruning period")
    quota_ax.set_ylabel("Groups pruned in this period")
    quota_ax.set_title("Actual matched-budget production schedules", pad=6)
    quota_ax.grid(axis="y", linestyle=":", alpha=0.45)
    quota_ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    periods = periods_u
    ticks = np.unique(
        np.linspace(int(periods.min()), int(periods.max()), 6, dtype=int)
    )
    for axis in (cost_ax, quota_ax):
        axis.set_xlim(int(periods.min()) - 0.5, int(periods.max()) + 0.5)
    quota_ax.set_xticks(ticks)

    # Never use C(c^U, b^U) versus C(c^C, b^C) as a mechanism comparison:
    # both the state-cost path and the schedule change.  Each row below holds
    # one measured c-vector fixed and changes only b.
    uniform_on_uniform_path = float(
        np.dot(display_u, quota_u) / quota_u.sum()
    )
    cosine_on_uniform_path = float(
        np.dot(display_u, quota_c) / quota_c.sum()
    )
    uniform_on_cosine_path = float(
        np.dot(display_c, quota_u) / quota_u.sum()
    )
    cosine_on_cosine_path = float(
        np.dot(display_c, quota_c) / quota_c.sum()
    )
    quota_ax.text(
        0.99,
        0.55,
        "Fixed-$c_p$ schedule reweighting\n"
        f"Uniform-path $c_p$: U {uniform_on_uniform_path:.3f} | "
        f"C {cosine_on_uniform_path:.3f}\n"
        f"Cosine-path $c_p$: U {uniform_on_cosine_path:.3f} | "
        f"C {cosine_on_cosine_path:.3f}",
        transform=quota_ax.transAxes,
        va="top",
        ha="right",
        fontsize=8.0,
        bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.9},
    )
    comparison_rows = []
    for index, period in enumerate(periods_u.tolist()):
        comparison_rows.append({
            "period": int(period),
            "uniform_probe_damage_raw_per_1pct": float(damage_u[index]),
            "uniform_probe_damage_per_1pct": float(display_u[index]),
            "uniform_probe_damage_moving_mean": float(smooth_u[index]),
            "uniform_actual_production_quota": int(quota_u[index]),
            "cosine_front_probe_damage_per_1pct": float(damage_c[index]),
            "cosine_front_probe_damage_moving_mean": float(smooth_c[index]),
            "cosine_front_actual_production_quota": int(quota_c[index]),
        })
    return {
        "smooth_window": smooth_window,
        "same_total_budget": True,
        "total_pruning_budget": int(quota_u.sum()),
        "display_replacements": (
            display_replacements_u + display_replacements_c
        ),
        "cross_path_absolute_costs_are_not_a_controlled_comparison": True,
        "fixed_uniform_path_cost_uniform_schedule": uniform_on_uniform_path,
        "fixed_uniform_path_cost_cosine_front_schedule": cosine_on_uniform_path,
        "fixed_cosine_path_cost_uniform_schedule": uniform_on_cosine_path,
        "fixed_cosine_path_cost_cosine_front_schedule": cosine_on_cosine_path,
        "records": comparison_rows,
    }


def _draw_training_history_comparison(loss_ax, accuracy_ax, sparsity_ax) -> None:
    history_u, summary_u = _load_training_history("uniform")
    history_c, summary_c = _load_training_history("cosine_front")
    if len(history_u) != len(history_c):
        raise RuntimeError("Uniform and cosine training histories have different lengths.")
    if len(history_u) != 120:
        raise RuntimeError(
            "The official comparison requires complete 20+100 histories "
            f"(expected 120 rows, found {len(history_u)})."
        )

    epochs_u = np.asarray([int(row["epoch"]) for row in history_u])
    epochs_c = np.asarray([int(row["epoch"]) for row in history_c])
    if not np.array_equal(epochs_u, epochs_c):
        raise RuntimeError("Uniform and cosine histories use different epochs.")
    if summary_u is None or summary_c is None:
        raise RuntimeError(
            "Both trajectories must finish compression and write final_summary.json."
        )
    if not (
        int(summary_u.get("complete_training_epochs", 0)) == 120
        and int(summary_c.get("complete_training_epochs", 0)) == 120
    ):
        raise RuntimeError("One or both final summaries report an incomplete run.")
    warmup_epochs = sum(row["stage"] == "warmup" for row in history_u)

    series = [
        ("uniform", history_u, GRAY),
        ("cosine_front", history_c, GREEN),
    ]
    for label, rows, color in series:
        display = "Balanced Uniform" if label == "uniform" else "Cosine-front"
        loss_ax.plot(
            epochs_u,
            [float(row["test_loss"]) for row in rows],
            color=color,
            linewidth=1.7,
            label=display,
        )
        accuracy_ax.plot(
            epochs_u,
            [float(row["test_top1"]) for row in rows],
            color=color,
            linewidth=1.7,
            label=display,
        )
        sparsity_ax.plot(
            epochs_u,
            [float(row["group_sparsity"]) for row in rows],
            color=color,
            linewidth=1.7,
            label=display,
        )

    for axis in (loss_ax, accuracy_ax, sparsity_ax):
        axis.axvline(
            warmup_epochs + 0.5,
            color="#9CA3AF",
            linestyle="--",
            linewidth=1.0,
        )
        axis.grid(True, linestyle=":", alpha=0.45)
        axis.set_xlim(int(epochs_u.min()), int(epochs_u.max()))
    loss_ax.set_ylabel("Test loss")
    accuracy_ax.set_ylabel("Test Top-1 (%)")
    sparsity_ax.set_ylabel("Group sparsity")
    sparsity_ax.set_xlabel("Training epoch")
    loss_ax.set_title(
        "Complete 20-epoch warmup + 100-period real training trajectories"
    )
    loss_ax.legend(frameon=False, ncol=2, fontsize=8.5)
    loss_ax.text(
        warmup_epochs,
        0.98,
        "warmup",
        transform=loss_ax.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=8,
    )
    loss_ax.text(
        warmup_epochs + 1,
        0.98,
        "pruning",
        transform=loss_ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=8,
    )

    if summary_u is not None and summary_c is not None:
        accuracy_ax.text(
            0.99,
            0.05,
            "Compressed Top-1\n"
            f"Uniform: {float(summary_u['compressed_top1']):.2f}%\n"
            f"Cosine-front: {float(summary_c['compressed_top1']):.2f}%",
            transform=accuracy_ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.9},
        )


def _save(fig, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05)


def render_spatial() -> None:
    _set_style()
    display_groups = int(os.environ.get("MOTIVATION_DISPLAY_GROUPS", "48"))
    fig_a, axes_a = plt.subplots(
        2, 1, figsize=(7.2, 6.0), gridspec_kw={"height_ratios": [1.0, 1.25]}
    )
    _draw_spatial(axes_a[0], axes_a[1], display_groups=display_groups)
    fig_a.subplots_adjust(hspace=0.72)
    _save(fig_a, "fig1a_spatial_conflict")
    plt.close(fig_a)
    if _load_spatial_periods() is not None:
        timeline, timeline_ax = plt.subplots(figsize=(7.2, 3.4))
        _draw_spatial_period_timeline(timeline_ax)
        timeline.tight_layout()
        _save(timeline, "fig1a_conflict_over_periods")
        plt.close(timeline)
    print(f"Spatial figure saved to {FIGURE_DIR}")


def render_spatial_restructured() -> None:
    """Render a new quota-explicit Fig. 1(a) candidate under a separate name."""
    _set_style()
    display_groups = int(os.environ.get("MOTIVATION_DISPLAY_GROUPS", "48"))
    data = _load_spatial()
    period_suffix = ""
    if "period" in data.files:
        period_suffix = f" at selected pruning period {int(np.asarray(data['period']).item())}"

    fig = plt.figure(figsize=(11.6, 6.4))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.92, 1.12],
        width_ratios=[1.72, 1.0],
        hspace=0.58,
        wspace=0.38,
    )
    zoom_ax = fig.add_subplot(grid[0, :])
    overview_ax = fig.add_subplot(grid[1, 0])
    overlap_ax = fig.add_subplot(grid[1, 1])
    _draw_spatial_restructured(
        zoom_ax,
        overview_ax,
        overlap_ax,
        display_groups=display_groups,
    )
    fig.suptitle(
        "Spatial conflict under the same Top-K diagnostic budget" + period_suffix,
        fontsize=14,
        y=0.985,
    )
    fig.subplots_adjust(top=0.91, bottom=0.10, left=0.10, right=0.96)
    _save(fig, "fig1a_spatial_conflict_restructured")
    plt.close(fig)
    print(
        "Restructured spatial figure saved without replacing the original: "
        f"{FIGURE_DIR / 'fig1a_spatial_conflict_restructured.png'}"
    )


def render_spatial_restyled() -> None:
    """Render the compact-caption revision under another separate name."""
    _set_style()
    display_groups = int(os.environ.get("MOTIVATION_DISPLAY_GROUPS", "48"))
    fig = plt.figure(figsize=(11.6, 5.35))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.92, 1.12],
        width_ratios=[1.72, 1.0],
        hspace=0.36,
        wspace=0.38,
    )
    zoom_ax = fig.add_subplot(grid[0, :])
    overview_ax = fig.add_subplot(grid[1, 0])
    overlap_ax = fig.add_subplot(grid[1, 1])
    _draw_spatial_restructured(
        zoom_ax,
        overview_ax,
        overlap_ax,
        display_groups=display_groups,
        compact_style=True,
    )
    fig.subplots_adjust(top=0.98, bottom=0.11, left=0.08, right=0.96)
    _save(fig, "fig1a_spatial_conflict_restyled")
    plt.close(fig)
    print(
        "Restyled spatial figure saved without replacing earlier versions: "
        f"{FIGURE_DIR / 'fig1a_spatial_conflict_restyled.png'}"
    )


def render_spatial_zoomed() -> None:
    """Render a true contiguous zoom linked to the full-set overview."""
    _set_style()
    display_groups = int(os.environ.get("MOTIVATION_DISPLAY_GROUPS", "48"))
    zoom_target_global = int(
        os.environ.get("MOTIVATION_SPATIAL_ZOOM_TARGET_GLOBAL", "1869")
    )
    fig = plt.figure(figsize=(11.6, 5.35))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.92, 1.12],
        width_ratios=[1.72, 1.0],
        hspace=0.30,
        wspace=0.38,
    )
    zoom_ax = fig.add_subplot(grid[0, :])
    overview_ax = fig.add_subplot(grid[1, 0])
    overlap_ax = fig.add_subplot(grid[1, 1])
    _draw_spatial_restructured(
        zoom_ax,
        overview_ax,
        overlap_ax,
        display_groups=display_groups,
        compact_style=True,
        linked_zoom=True,
        zoom_target_global=zoom_target_global,
    )
    fig.subplots_adjust(top=0.98, bottom=0.11, left=0.08, right=0.96)
    _save(fig, "fig1a_spatial_conflict_zoomed")
    plt.close(fig)
    print(
        "Linked-zoom spatial figure saved without replacing earlier versions: "
        f"{FIGURE_DIR / 'fig1a_spatial_conflict_zoomed.png'}"
    )


def render_spatial_pattern_match() -> None:
    """Match the original disputed-group panel to its closest contiguous window."""
    _set_style()
    display_groups = int(os.environ.get("MOTIVATION_DISPLAY_GROUPS", "48"))
    fig = plt.figure(figsize=(11.6, 5.35))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.92, 1.12],
        width_ratios=[1.72, 1.0],
        hspace=0.30,
        wspace=0.38,
    )
    zoom_ax = fig.add_subplot(grid[0, :])
    overview_ax = fig.add_subplot(grid[1, 0])
    overlap_ax = fig.add_subplot(grid[1, 1])
    _draw_spatial_restructured(
        zoom_ax,
        overview_ax,
        overlap_ax,
        display_groups=display_groups,
        compact_style=True,
        linked_zoom=True,
        pattern_match=True,
    )
    fig.subplots_adjust(top=0.98, bottom=0.11, left=0.08, right=0.96)
    _save(fig, "fig1a_spatial_conflict_pattern_match")
    plt.close(fig)
    print(
        "Pattern-matched spatial figure saved without replacing earlier versions: "
        f"{FIGURE_DIR / 'fig1a_spatial_conflict_pattern_match.png'}"
    )


def render_spatial_box_expanded() -> None:
    """Expand the previously matched G1360--G1408 box using its real columns."""
    _set_style()
    display_groups = int(os.environ.get("MOTIVATION_DISPLAY_GROUPS", "48"))
    box_center_global = int(
        os.environ.get("MOTIVATION_SPATIAL_BOX_CENTER_GLOBAL", "1385")
    )
    fig = plt.figure(figsize=(11.6, 5.35))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.92, 1.12],
        width_ratios=[1.72, 1.0],
        hspace=0.30,
        wspace=0.38,
    )
    zoom_ax = fig.add_subplot(grid[0, :])
    overview_ax = fig.add_subplot(grid[1, 0])
    overlap_ax = fig.add_subplot(grid[1, 1])
    _draw_spatial_restructured(
        zoom_ax,
        overview_ax,
        overlap_ax,
        display_groups=display_groups,
        compact_style=True,
        linked_zoom=True,
        zoom_target_global=box_center_global,
    )
    fig.subplots_adjust(top=0.98, bottom=0.11, left=0.08, right=0.96)
    _save(fig, "fig1a_spatial_conflict_box_expanded")
    plt.close(fig)
    print(
        "True box-expanded spatial figure saved without replacing earlier versions: "
        f"{FIGURE_DIR / 'fig1a_spatial_conflict_box_expanded.png'}"
    )


def render_spatial_redbox_expanded() -> None:
    """Expand 48 real columns from the user-marked region with edge connectors."""
    _set_style()
    display_groups = int(os.environ.get("MOTIVATION_DISPLAY_GROUPS", "48"))
    redbox_before_global = int(
        os.environ.get("MOTIVATION_SPATIAL_REDBOX_BEFORE_GLOBAL", "471")
    )
    fig = plt.figure(figsize=(11.6, 5.35))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.92, 1.12],
        width_ratios=[1.72, 1.0],
        hspace=0.30,
        wspace=0.38,
    )
    zoom_ax = fig.add_subplot(grid[0, :])
    overview_ax = fig.add_subplot(grid[1, 0])
    overlap_ax = fig.add_subplot(grid[1, 1])
    _draw_spatial_restructured(
        zoom_ax,
        overview_ax,
        overlap_ax,
        display_groups=display_groups,
        compact_style=True,
        linked_zoom=True,
        zoom_before_global=redbox_before_global,
        connect_to_image_edges=True,
        overview_panel_label="a",
        overlap_panel_label="b",
    )
    fig.subplots_adjust(top=0.98, bottom=0.11, left=0.08, right=0.96)
    _save(fig, "fig1a_spatial_conflict_redbox_expanded")
    plt.close(fig)
    print(
        "Red-box spatial figure saved without replacing earlier versions: "
        f"{FIGURE_DIR / 'fig1a_spatial_conflict_redbox_expanded.png'}"
    )


def render_temporal() -> None:
    _set_style()

    fig_b, axes_b = plt.subplots(
        2, 1, figsize=(7.6, 6.2), gridspec_kw={"height_ratios": [1.15, 1.0]}
    )
    _draw_temporal(axes_b[0], axes_b[1])
    fig_b.subplots_adjust(hspace=0.52, left=0.13, right=0.98, top=0.98, bottom=0.15)
    _save(fig_b, "fig1b_temporal_mismatch")
    plt.close(fig_b)
    print(f"Temporal figure saved to {FIGURE_DIR}")


def render_temporal_damage_single() -> None:
    """Render only the Uniform-reference stage-wise damage panel."""
    _set_style()
    figure, axis = plt.subplots(1, 1, figsize=(7.6, 3.65))
    _draw_temporal_damage_only(axis, include_cosine_front=False)
    figure.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.28)
    _save(figure, "fig1b_temporal_damage_uniform_path")
    plt.close(figure)
    print(
        "Single temporal-damage figure saved to "
        f"{FIGURE_DIR / 'fig1b_temporal_damage_uniform_path.png'}"
    )


def render_temporal_damage_two_paths() -> None:
    """Add the Cosine-front real-path damage to the single-panel candidate."""
    _set_style()
    figure, axis = plt.subplots(1, 1, figsize=(7.6, 3.65))
    _draw_temporal_damage_only(axis, include_cosine_front=True)
    figure.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.28)
    _save(figure, "fig1b_temporal_damage_two_real_paths")
    plt.close(figure)
    print(
        "Two-path temporal-damage figure saved to "
        f"{FIGURE_DIR / 'fig1b_temporal_damage_two_real_paths.png'}"
    )


def render_production_damage_comparison() -> None:
    """Render actual period-level HESSO damage from matched controls."""
    _set_style()
    periods_u, damage_u, _ = _load_production_damage_arrays("uniform")
    periods_c, damage_c, _ = _load_production_damage_arrays("cosine_front")
    if not np.array_equal(periods_u, periods_c):
        raise RuntimeError("Production-damage runs use different period grids.")
    smooth_window = int(
        os.environ.get("PRODUCTION_DAMAGE_SMOOTH_WINDOW", "5")
    )
    smooth_u = _rolling_mean(damage_u, smooth_window)
    smooth_c = _rolling_mean(damage_c, smooth_window)

    figure, axis = plt.subplots(1, 1, figsize=(7.6, 3.65))
    axis.plot(
        periods_u,
        damage_u,
        color="#9CA3AF",
        marker="o",
        markersize=2.5,
        linewidth=0.7,
        alpha=0.38,
        label=r"Uniform measured $d_p$",
    )
    axis.plot(
        periods_u,
        smooth_u,
        color=GRAY,
        linewidth=1.9,
        label="Uniform damage trend",
    )
    axis.plot(
        periods_c,
        damage_c,
        color="#5FBF7A",
        marker="s",
        markersize=2.5,
        linewidth=0.7,
        alpha=0.34,
        label=r"Cosine-front measured $d_p$",
    )
    axis.plot(
        periods_c,
        smooth_c,
        color=GREEN,
        linewidth=1.9,
        label="Cosine-front damage trend",
    )
    axis.axhline(0.0, color="#9CA3AF", linewidth=0.8, linestyle="--")
    ticks = np.unique(
        np.linspace(int(periods_u.min()), int(periods_u.max()), 6, dtype=int)
    )
    axis.set_xticks(ticks)
    axis.set_xlim(int(periods_u.min()) - 0.5, int(periods_u.max()) + 0.5)
    axis.set_xlabel("Pruning period")
    axis.set_ylabel(r"Production pruning damage  $d_p$")
    axis.grid(axis="both", linestyle=":", alpha=0.5)
    axis.legend(frameon=False, loc="upper left", fontsize=8.2, ncol=2)
    axis.text(
        0.5,
        -0.25,
        "Matched-control production pruning damage",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=11.5,
    )
    figure.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.28)
    _save(figure, "production_damage_two_schedules")
    plt.close(figure)
    print(
        "Matched-control production-damage figure saved to "
        f"{FIGURE_DIR / 'production_damage_two_schedules.png'}"
    )


def render_temporal_horizontal() -> None:
    """Render the two temporal-mismatch panels side by side for a two-column paper."""
    _set_style()

    figure, axes = plt.subplots(1, 2, figsize=(7.8, 3.15))
    _draw_temporal(axes[0], axes[1])
    figure.subplots_adjust(
        wspace=0.32,
        left=0.075,
        right=0.99,
        top=0.97,
        bottom=0.28,
    )
    _save(figure, "fig1b_temporal_mismatch_horizontal")
    plt.close(figure)
    print(
        "Horizontal temporal figure saved without replacing the vertical version: "
        f"{FIGURE_DIR / 'fig1b_temporal_mismatch_horizontal.png'}"
    )


def render_temporal_real_comparison() -> None:
    _set_style()
    figure, axes = plt.subplots(
        2, 1, figsize=(7.8, 6.3), gridspec_kw={"height_ratios": [1.15, 1.0]}
    )
    comparison = _draw_real_trajectory_comparison(axes[0], axes[1])
    figure.subplots_adjust(
        hspace=0.34, left=0.13, right=0.98, top=0.95, bottom=0.10
    )
    _save(figure, "fig1b_temporal_real_trajectory_comparison")
    plt.close(figure)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    with (FIGURE_DIR / "fig1b_temporal_real_trajectory_comparison.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(comparison, handle, indent=2, ensure_ascii=False)
    rows = comparison["records"]
    with (FIGURE_DIR / "fig1b_temporal_real_trajectory_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Real-trajectory comparison saved to {FIGURE_DIR}")


def render_training_history_comparison() -> None:
    _set_style()
    figure, axes = plt.subplots(3, 1, figsize=(7.8, 8.0), sharex=True)
    _draw_training_history_comparison(axes[0], axes[1], axes[2])
    figure.subplots_adjust(
        hspace=0.22, left=0.13, right=0.98, top=0.95, bottom=0.08
    )
    _save(figure, "temporal_complete_training_trajectory_comparison")
    plt.close(figure)
    print(f"Complete training-history comparison saved to {FIGURE_DIR}")


def render_combined() -> None:
    _set_style()
    display_groups = int(os.environ.get("MOTIVATION_DISPLAY_GROUPS", "48"))

    fig = plt.figure(figsize=(14.2, 6.1))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.27)
    left = outer[0].subgridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.75)
    right = outer[1].subgridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.28)
    _draw_spatial(
        fig.add_subplot(left[0]),
        fig.add_subplot(left[1]),
        display_groups=display_groups,
    )
    _draw_temporal(fig.add_subplot(right[0]), fig.add_subplot(right[1]))
    fig.text(0.245, 0.005, "(a) Cross-criterion selection conflict", ha="center", fontsize=12)
    fig.text(0.755, 0.005, "(b) Stage-cost / schedule mismatch", ha="center", fontsize=12)
    _save(fig, "fig1_motivation_combined")
    plt.close(fig)
    print(f"Combined figure saved to {FIGURE_DIR}")


def render_all() -> None:
    render_spatial()
    render_temporal()
    render_combined()


if __name__ == "__main__":
    mode = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "all"
    if mode == "spatial":
        render_spatial()
    elif mode == "spatial_restructured":
        render_spatial_restructured()
    elif mode == "spatial_restyled":
        render_spatial_restyled()
    elif mode == "spatial_zoomed":
        render_spatial_zoomed()
    elif mode == "spatial_pattern_match":
        render_spatial_pattern_match()
    elif mode == "spatial_box_expanded":
        render_spatial_box_expanded()
    elif mode == "spatial_redbox_expanded":
        render_spatial_redbox_expanded()
    elif mode == "temporal":
        render_temporal()
    elif mode in {"temporal_damage", "temporal_damage_single"}:
        render_temporal_damage_single()
    elif mode in {"temporal_damage_compare", "temporal_damage_two_paths"}:
        render_temporal_damage_two_paths()
    elif mode in {"production_damage", "production_damage_compare"}:
        render_production_damage_comparison()
    elif mode == "temporal_horizontal":
        render_temporal_horizontal()
    elif mode in {"temporal_compare", "temporal_real_compare"}:
        render_temporal_real_comparison()
    elif mode in {"training_compare", "temporal_training_compare"}:
        render_training_history_comparison()
    elif mode == "combined":
        render_combined()
    elif mode == "all":
        render_all()
    else:
        raise SystemExit(
            "Usage: python plot_motivation_fig1.py "
            "{spatial|spatial_restructured|spatial_restyled|spatial_zoomed|"
            "spatial_pattern_match|spatial_box_expanded|spatial_redbox_expanded|"
            "temporal|temporal_damage|temporal_damage_compare|production_damage|"
            "temporal_horizontal|temporal_compare|"
            "training_compare|combined|all}"
        )
