#!/usr/bin/env python
"""YOLO11 的 AdaMCF 结构化剪枝入口。

此脚本沿用论文 YOLOv5 实验的四个关键步骤：

1. OTO 建立结构依赖图；
2. HESSO 以 AdaMCF 多准则融合分数逐步生成 group sparsity；
3. OTO 物理删除冗余通道，构建真实子网络；
4. 通过 Ultralytics 完成子网微调与 ONNX 导出。

YOLO11n 的 C2PSA、Attention、CSP 与 Detect 语义边界在默认保守 profile 中
保持保护；实验 profile 只按已声明的 dependency contract 逐项开放。任何新边界
均必须先完成真实逻辑命中、物理构网、checkpoint 重载和固定输入前向，不能把
“进入候选集”解释为已经支持结构化剪枝。

示例：
  python prune_yolo11_adamcf.py audit --weights <yolo11n.pt> --device cpu
  python prune_yolo11_adamcf.py prune --weights <yolo11n.pt> \
      --epochs 50 --warmup-epochs 5 --prune-ratio 0.10 --device 0 --no-amp
  python prune_yolo11_adamcf.py finetune --weights <structured.pt> \
      --epochs 80 --device 0
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# =========================
# 运行环境与工程路径
# =========================
os.environ.setdefault("YOLO_NO_AUTO_INSTALL", "True")
os.environ.setdefault("ULTRALYTICS_SETTINGS", "0")

CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parent
# CURRENT_FILE = .../OTOv2_v2/test/pretrain_yolo11_coco/prune_yolo11_adamcf.py
OTO_ROOT = CURRENT_FILE.parents[2]
DEFAULT_DATA_YAML = BASE_DIR / "coco2017_yolo11.yaml"

if str(OTO_ROOT) not in sys.path:
    sys.path.insert(0, str(OTO_ROOT))

import torch
import ultralytics
from thop import profile
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

from only_train_once import OTO
from only_train_once.graph.node_group import NodeGroup as OTONodeGroup
from only_train_once.transform import TensorTransform, make_index_groups_transform


# ---------------------------
# 小工具
# ---------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON 顶层必须是对象：{path}")
    return value


def stable_json_sha256(value: Any) -> str:
    """对可 JSON 序列化的合同/报告生成稳定指纹。"""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def export_oto_graphs(
    oto,
    output_dir: Path,
    *,
    formats: tuple[str, ...] = ("svg", "pdf"),
    display_params: bool = False,
    display_flops: bool = False,
) -> dict[str, Any]:
    """调用 OTO 既有 ``build_dot`` 生成节点组图和原始算子图。

    该函数只位于 YOLO11 实验入口，不修改 OTO Graph、依赖生成或剪枝流程。
    DOT 源文件始终保存；系统缺少 Graphviz 可执行程序时只把渲染状态记为失败，
    不影响建图审计本身。
    """

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_formats = tuple(
        dict.fromkeys(str(value).lower() for value in formats if value)
    ) or ("svg", "pdf")
    unsupported = sorted(set(normalized_formats) - {"svg", "pdf", "png"})
    if unsupported:
        raise ValueError(f"OTO 图不支持的格式：{unsupported}")

    records: list[dict[str, Any]] = []
    for variant, by_node_groups in (
        ("node_groups", True),
        ("operators", False),
    ):
        dot = oto._graph.build_dot(
            vertical=True,
            by_node_groups=by_node_groups,
            display_params=bool(display_params),
            display_flops=bool(display_flops),
        )
        dot_path = output_dir / f"yolo11n_oto_{variant}.dot"
        dot_path.write_text(dot.source, encoding="utf-8")
        rendered: list[str] = []
        render_errors: dict[str, str] = {}
        for file_format in normalized_formats:
            output_path = output_dir / f"yolo11n_oto_{variant}.{file_format}"
            try:
                output_path.write_bytes(dot.pipe(format=file_format))
                rendered.append(str(output_path))
            except Exception as exc:  # Graphviz executable/format availability is environment-specific.
                render_errors[file_format] = f"{type(exc).__name__}: {exc}"
        records.append(
            {
                "variant": variant,
                "by_node_groups": by_node_groups,
                "dot": str(dot_path),
                "dot_sha256": sha256_file(dot_path),
                "rendered": rendered,
                "render_errors": render_errors,
            }
        )

    manifest = {
        "status": (
            "rendered"
            if all(record["rendered"] for record in records)
            else "dot_only"
        ),
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "formats": list(normalized_formats),
        "display_params": bool(display_params),
        "display_flops": bool(display_flops),
        "implementation": "yolo11_entry_calls_existing_oto_graph_build_dot_v1",
        "records": records,
    }
    manifest_path = output_dir / "oto_graph_manifest.json"
    json_dump(manifest_path, manifest)
    manifest["manifest"] = str(manifest_path)
    return manifest


def hesso_optimizer_step_schedule(
    *,
    batches_per_epoch: int,
    epochs: int,
    warmup_epochs: float,
    gradient_accumulation: int,
) -> dict[str, int]:
    """按实际 ``optimizer.step()`` 次数生成 HESSO 调度。

    HESSO 的 ``num_steps`` 每执行一次 ``HESSO.step()`` 才加一；Ultralytics
    在小 batch 下会做梯度累积，因此不能直接用 dataloader batch 数作为 HESSO
    step。当前剪枝入口关闭了 Ultralytics 自身的 warmup，故 accumulate 在整个
    训练过程中保持不变，可以据此精确估计调度步数。
    """
    batches_per_epoch = int(batches_per_epoch)
    epochs = int(epochs)
    gradient_accumulation = int(gradient_accumulation)
    if batches_per_epoch <= 0 or epochs <= 0 or gradient_accumulation <= 0:
        raise ValueError("HESSO 调度需要正的 epoch、batch 数和梯度累积步数。")

    total_batches = epochs * batches_per_epoch
    warmup_batches = int(round(float(warmup_epochs) * batches_per_epoch))
    total_optimizer_steps = total_batches // gradient_accumulation
    start_pruning_step = warmup_batches // gradient_accumulation
    pruning_steps = total_optimizer_steps - start_pruning_step
    if total_optimizer_steps <= 0 or pruning_steps <= 0:
        raise ValueError("warmup_epochs 必须小于总 epochs，且训练应至少产生一次 optimizer.step()。")

    return {
        "total_batches": total_batches,
        "warmup_batches": warmup_batches,
        "gradient_accumulation": gradient_accumulation,
        "total_optimizer_steps": total_optimizer_steps,
        "start_pruning_step": start_pruning_step,
        "pruning_steps": pruning_steps,
    }


def scalar_float(value: Any) -> float | None:
    """将 Python 数字或零维 Tensor 规范为 float；不可转换时返回 None。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError, RuntimeError):
            return None
    return None


def numeric_metrics(metrics: Any) -> dict[str, float]:
    """抽取 Ultralytics 版本间保持兼容的数值验证指标。"""
    if hasattr(metrics, "results_dict"):
        metrics = metrics.results_dict
    if not isinstance(metrics, dict):
        return {}
    result: dict[str, float] = {}
    for name, value in metrics.items():
        number = scalar_float(value)
        if number is not None:
            result[str(name)] = number
    return result


def validation_summary(metrics: Any) -> dict[str, Any]:
    """统一提取 precision、recall、mAP50、mAP50-95 与原始指标字典。"""
    values = numeric_metrics(metrics)
    summary: dict[str, Any] = {"metrics": values}
    for name, value in values.items():
        key = name.lower().replace("_", "-")
        if "map50-95" in key or "map50-95" in key.replace("/", ""):
            summary["map50_95"] = value
        elif "map50" in key:
            summary["map50"] = value
        elif "precision" in key:
            summary["precision"] = value
        elif "recall" in key:
            summary["recall"] = value
        elif key == "fitness" or key.endswith("/fitness"):
            summary["fitness"] = value
    return summary


def format_percent(value: float | None) -> str:
    return "?" if value is None else f"{value * 100:.2f}%"


def flatten_record(value: Any, prefix: str = "") -> dict[str, Any]:
    """将 epoch 嵌套记录扁平化，便于写入 CSV。"""
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_record(item, name))
        return flattened
    return {prefix: value}


def hesso_summary(optimizer: Any) -> dict[str, Any]:
    """读取 HESSO 当前稀疏度、组计数和 AdaMCF α 权重。"""
    if not hasattr(optimizer, "compute_metrics"):
        return {}

    result: dict[str, Any] = {}
    # 这些属性由 HESSO 直接维护，不依赖 compute_metrics()，用于确认调度是否真正进入剪枝期。
    for name in (
        "num_steps",
        "start_pruning_step",
        "curr_pruning_period",
        "pruning_periods",
        "pruning_period_duration",
    ):
        value = scalar_float(getattr(optimizer, name, None))
        if value is not None:
            result[name] = int(value)
    try:
        metrics = optimizer.compute_metrics()
        for name in (
            "group_sparsity",
            "zero_group_ratio",
            "pruned_group_ratio",
            "num_groups",
            "num_zero_groups",
            "num_pruned_groups",
            "num_active_redundant_groups",
        ):
            value = scalar_float(getattr(metrics, name, None))
            if value is not None:
                result[name] = int(value) if name.startswith("num_") else value
    except (AttributeError, IndexError, RuntimeError, ValueError) as exc:
        result["metrics_error"] = str(exc)

    scheduler = getattr(optimizer, "alpha_scheduler", None)
    keys = list(getattr(optimizer, "_alpha_keys", []))
    if scheduler is not None and keys and hasattr(scheduler, "get_alpha"):
        try:
            alpha_values = scheduler.get_alpha().detach().cpu().flatten().tolist()
            result["alpha"] = {name: float(value) for name, value in zip(keys, alpha_values)}
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            result["alpha_error"] = str(exc)
    return result


class EpochReporter:
    """以 YOLOv5 参考日志的粒度记录 Ultralytics 每个验证完成后的 epoch。"""

    def __init__(self, phase: str):
        self.phase = phase
        self.records: list[dict[str, Any]] = []
        self.last_epoch: int | None = None
        self.jsonl_path: Path | None = None
        self.json_path: Path | None = None
        self.csv_path: Path | None = None

    def attach(self, trainer) -> None:
        save_dir = Path(trainer.save_dir)
        self.jsonl_path = save_dir / f"{self.phase}_detailed_epoch_metrics.jsonl"
        self.json_path = save_dir / f"{self.phase}_detailed_epoch_metrics.json"
        self.csv_path = save_dir / f"{self.phase}_detailed_epoch_metrics.csv"
        trainer.add_callback("on_fit_epoch_end", self.on_fit_epoch_end)

    def on_fit_epoch_end(self, trainer) -> None:
        if getattr(trainer, "rank", -1) not in {-1, 0}:
            return
        epoch = int(trainer.epoch) + 1
        # Ultralytics 会在 final_eval 后再次触发该回调；同 epoch 不重复记一行。
        if self.last_epoch == epoch:
            return
        self.last_epoch = epoch

        epoch_seconds = float(getattr(trainer, "epoch_time", 0.0) or 0.0)
        elapsed_seconds = max(0.0, time.time() - float(getattr(trainer, "train_time_start", time.time())))
        total_epochs = int(getattr(trainer, "epochs", epoch))

        try:
            losses = trainer.label_loss_items(trainer.tloss)
        except (AttributeError, TypeError, ValueError):
            losses = {}
        losses = {name: value for name, value in numeric_metrics(losses).items()}

        record = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "epoch_seconds": epoch_seconds,
            "elapsed_seconds": elapsed_seconds,
            "gpu_memory": str(trainer._get_memory()) if hasattr(trainer, "_get_memory") else "?",
            "loss": losses,
            "learning_rate": numeric_metrics(getattr(trainer, "lr", {})),
            "validation": validation_summary(getattr(trainer, "metrics", {})),
            "hesso": hesso_summary(getattr(trainer, "optimizer", None)),
        }
        self.records.append(record)
        self._save_records()
        self._print_record(record)

    def _save_records(self) -> None:
        if self.jsonl_path is None or self.json_path is None or self.csv_path is None:
            return
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.records[-1], ensure_ascii=False, sort_keys=True) + "\n")
        json_dump(self.json_path, {"phase": self.phase, "epochs": self.records})

        rows = [flatten_record(record) for record in self.records]
        fields = sorted({name for row in rows for name in row})
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _print_record(self, record: dict[str, Any]) -> None:
        loss_text = ", ".join(f"{name.split('/')[-1]}={value:.4f}" for name, value in record["loss"].items())
        lr_values = list(record["learning_rate"].values())
        lr_text = ", ".join(f"{value:.6g}" for value in lr_values[:3]) or "?"
        validation = record["validation"]
        hesso = record["hesso"]
        print(
            f"\n[{self.phase}] Epoch {record['epoch']}/{record['total_epochs']} 评估\n"
            f"Loss: {loss_text or '?'} | LR: {lr_text} | GPU: {record['gpu_memory']} | "
            f"本周期用时: {record['epoch_seconds']:.1f}s | "
            f"累计总耗时: {record['elapsed_seconds'] / 3600:.2f}h\n"
            f"验证结果: mAP@0.5 {format_percent(validation.get('map50'))}, "
            f"mAP@0.5:0.95 {format_percent(validation.get('map50_95'))}",
            flush=True,
        )
        if hesso:
            print(
                "HESSO: "
                f"steps={hesso.get('num_steps', '?')}/start={hesso.get('start_pruning_step', '?')}, "
                f"period={hesso.get('curr_pruning_period', '?')}/"
                f"{hesso.get('pruning_periods', '?')}, "
                f"group_sparsity={hesso.get('group_sparsity', '?')}, "
                f"pruned_groups={hesso.get('num_pruned_groups', '?')}/"
                f"{hesso.get('num_groups', '?')}",
                flush=True,
            )
            if hesso.get("alpha"):
                print(f"AdaMCF α: {hesso['alpha']}", flush=True)


def compression_summary(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    """计算物理模型前后参数量、FLOPs 的减少量和压缩比。"""
    summary: dict[str, float] = {}
    for metric in ("parameters", "flops"):
        before_value = float(before[metric])
        after_value = float(after[metric])
        reduction = max(before_value - after_value, 0.0)
        summary[f"{metric}_reduction"] = reduction
        summary[f"{metric}_reduction_ratio"] = reduction / before_value if before_value > 0 else 0.0
        summary[f"{metric}_compression_ratio"] = before_value / after_value if after_value > 0 else float("inf")
    return summary


def save_stage_summary(
    path: Path,
    *,
    phase: str,
    started_at: float,
    source_weights: Path,
    final_weights: Path,
    before_stats: dict[str, float],
    after_stats: dict[str, float],
    final_validation: dict[str, Any],
    epoch_reporter: EpochReporter | None,
    hesso: dict[str, Any] | None = None,
    baseline_validation: dict[str, Any] | None = None,
    validation_details: dict[str, Any] | None = None,
    architecture_check: dict[str, Any] | None = None,
    validation_scope: str = "Ultralytics 训练结束后的最终验证结果。",
) -> dict[str, Any]:
    """写出阶段结束摘要，并在控制台打印与 YOLOv5 参考脚本对应的核心信息。"""
    summary = {
        "phase": phase,
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "total_seconds": time.time() - started_at,
        "source_weights": str(source_weights),
        "source_weights_sha256": sha256_file(source_weights),
        "final_weights": str(final_weights),
        "final_weights_sha256": sha256_file(final_weights),
        "before_model_stats": before_stats,
        "after_model_stats": after_stats,
        "compression": compression_summary(before_stats, after_stats),
        "final_validation": final_validation,
        "baseline_validation": baseline_validation or {},
        "validation_details": validation_details or {},
        "architecture_check": architecture_check or {},
        "validation_scope": validation_scope,
        "hesso": hesso or {},
        "epoch_metrics": {
            "count": len(epoch_reporter.records) if epoch_reporter is not None else 0,
            "json": str(epoch_reporter.json_path) if epoch_reporter and epoch_reporter.json_path else None,
            "jsonl": str(epoch_reporter.jsonl_path) if epoch_reporter and epoch_reporter.jsonl_path else None,
            "csv": str(epoch_reporter.csv_path) if epoch_reporter and epoch_reporter.csv_path else None,
        },
    }
    json_dump(path, summary)

    compression = summary["compression"]
    validation = summary["final_validation"]
    print_stage(f"{phase} 阶段详细总结")
    print(
        f"总耗时: {summary['total_seconds'] / 3600:.2f}h\n"
        f"参数量: {before_stats['parameters_m']:.3f}M -> {after_stats['parameters_m']:.3f}M "
        f"(减少 {compression['parameters_reduction_ratio'] * 100:.2f}%, "
        f"压缩比 {compression['parameters_compression_ratio']:.2f}x)\n"
        f"FLOPs: {before_stats['flops_g']:.3f}G -> {after_stats['flops_g']:.3f}G "
        f"(减少 {compression['flops_reduction_ratio'] * 100:.2f}%, "
        f"压缩比 {compression['flops_compression_ratio']:.2f}x)\n"
        f"最终验证: mAP@0.5 {format_percent(validation.get('map50'))}, "
        f"mAP@0.5:0.95 {format_percent(validation.get('map50_95'))}\n"
        f"详细摘要: {path}",
        flush=True,
    )
    return summary


def metric_delta_percent(before: dict[str, Any], after: dict[str, Any], key: str) -> float | None:
    before_value = scalar_float(before.get(key))
    after_value = scalar_float(after.get(key))
    if before_value is None or after_value is None:
        return None
    return (after_value - before_value) * 100.0


def format_delta_percent(value: float | None) -> str:
    return "?" if value is None else f"{value:+.2f}%"


def print_pruning_compression_report(
    *,
    structured_weights: Path,
    before_stats: dict[str, float],
    after_stats: dict[str, float],
    baseline_validation: dict[str, Any],
    structured_validation: dict[str, Any],
    hesso: dict[str, Any],
) -> None:
    """按项目既有模板打印适用于目标检测的剪枝后统计。"""

    compression = compression_summary(before_stats, after_stats)
    group_sparsity = scalar_float(hesso.get("group_sparsity"))
    map50_delta = metric_delta_percent(baseline_validation, structured_validation, "map50")
    map_delta = metric_delta_percent(baseline_validation, structured_validation, "map50_95")

    print("\n--- 构建并加载子网络用于微调 ---", flush=True)
    print(f"结构化子网已保存在: {structured_weights}", flush=True)
    print(
        f"剪枝后准确率: mAP@0.5: {format_percent(structured_validation.get('map50'))}, "
        f"mAP@0.5:0.95: {format_percent(structured_validation.get('map50_95'))}",
        flush=True,
    )
    print(f"剪枝后 FLOPs(thop): {after_stats['flops'] / 1e6:.2f} M", flush=True)
    print(f"剪枝后参数量: {after_stats['parameters_m']:.2f} M", flush=True)

    print("\n--- 第 6 步：压缩总结 ---", flush=True)
    print(
        "最终剪枝率 (组稀疏率): "
        + ("?" if group_sparsity is None else f"{group_sparsity:.2f} ({group_sparsity * 100:.2f}%)"),
        flush=True,
    )
    print(
        f"减少的 FLOPs: {compression['flops_reduction'] / 1e6:.2f}M "
        f"({compression['flops_reduction_ratio'] * 100:.2f}%)",
        flush=True,
    )
    print(
        f"减少的参数量: {compression['parameters_reduction'] / 1e6:.2f}M "
        f"({compression['parameters_reduction_ratio'] * 100:.2f}%)",
        flush=True,
    )
    print(
        f"FLOPs 压缩比: {compression['flops_compression_ratio']:.2f}x "
        f"({before_stats['flops'] / 1e6:.2f}M -> {after_stats['flops'] / 1e6:.2f}M)",
        flush=True,
    )
    print(
        f"参数量压缩比: {compression['parameters_compression_ratio']:.2f}x "
        f"({before_stats['parameters_m']:.2f}M -> {after_stats['parameters_m']:.2f}M)",
        flush=True,
    )
    print(
        f"准确率变化: mAP@0.5 从 {format_percent(baseline_validation.get('map50'))} "
        f"变为 {format_percent(structured_validation.get('map50'))} "
        f"(变化: {format_delta_percent(map50_delta)})",
        flush=True,
    )
    print(
        f"准确率变化: mAP@0.5:0.95 从 {format_percent(baseline_validation.get('map50_95'))} "
        f"变为 {format_percent(structured_validation.get('map50_95'))} "
        f"(变化: {format_delta_percent(map_delta)})",
        flush=True,
    )


def print_finetune_completion_report(
    *,
    final_weights: Path,
    baseline_validation: dict[str, Any],
    final_validation: dict[str, Any],
    architecture_check: dict[str, Any],
) -> None:
    """按项目既有模板打印适用于目标检测的微调完成统计。"""

    map50_delta = metric_delta_percent(baseline_validation, final_validation, "map50")
    map_delta = metric_delta_percent(baseline_validation, final_validation, "map50_95")
    print("\n微调过程完成。", flush=True)
    print(f"最佳模型已保存在: {final_weights}", flush=True)
    print(
        f"微调后最终准确率: mAP@0.5: {format_percent(final_validation.get('map50'))}, "
        f"mAP@0.5:0.95: {format_percent(final_validation.get('map50_95'))}",
        flush=True,
    )
    print(
        f"未剪枝模型准确率: mAP@0.5: {format_percent(baseline_validation.get('map50'))}, "
        f"mAP@0.5:0.95: {format_percent(baseline_validation.get('map50_95'))}",
        flush=True,
    )
    print(
        f"相较未剪枝模型: mAP@0.5 变化 {format_delta_percent(map50_delta)}, "
        f"mAP@0.5:0.95 变化 {format_delta_percent(map_delta)}",
        flush=True,
    )
    print(
        f"剪枝模型结构加载校验: "
        f"{'通过' if architecture_check.get('structure_matched') else '失败'} "
        f"({architecture_check.get('expected_sha256', '?')} -> "
        f"{architecture_check.get('actual_sha256', '?')})",
        flush=True,
    )
    print(
        f"剪枝模型初始权重加载校验: "
        f"{'通过' if architecture_check.get('weights_matched_before_training') else '失败'} "
        f"({architecture_check.get('expected_state_sha256', '?')} -> "
        f"{architecture_check.get('actual_state_sha256_before_training', '?')})",
        flush=True,
    )
    print(
        f"微调后网络结构保持校验: "
        f"{'通过' if architecture_check.get('final_structure_matched') else '失败'} "
        f"({architecture_check.get('expected_sha256', '?')} -> "
        f"{architecture_check.get('final_model_sha256', '?')})",
        flush=True,
    )


def print_stage(title: str) -> None:
    print("\n" + "=" * 60, flush=True)
    print(title, flush=True)
    print("=" * 60, flush=True)


def print_human_summary(
    title: str,
    rows: list[tuple[str, Any]],
    *,
    artifact: Path | str | None = None,
) -> None:
    """在控制台只打印人类可读摘要；完整结构化信息继续写入 JSON。"""

    print_stage(title)
    visible_rows = [(str(label), value) for label, value in rows if value is not None]
    width = max((len(label) for label, _ in visible_rows), default=0)
    for label, value in visible_rows:
        print(f"{label:<{width}} : {value}", flush=True)
    if artifact is not None:
        print(f"详细报告{' ' * max(0, width - 4)} : {artifact}", flush=True)


def compact_family_coverage(dependency_control: dict[str, Any]) -> str:
    families = (
        (dependency_control or {})
        .get("candidate_family_coverage", {})
        .get("families", {})
    )
    if not isinstance(families, dict) or not families:
        return "未记录"
    parts = []
    for family, record in families.items():
        if not isinstance(record, dict):
            continue
        candidate = int(record.get("candidate_group_count", 0) or 0)
        prunable = int(record.get("prunable_group_count_after_protection", 0) or 0)
        parts.append(f"{family} {prunable}/{candidate}")
    return "，".join(parts) if parts else "未记录"


def compact_family_hits(redundancy: dict[str, Any]) -> str:
    families = (redundancy or {}).get("opened_module_family_hits", {})
    if not isinstance(families, dict) or not families:
        return "无"
    parts = []
    for family, record in families.items():
        if not isinstance(record, dict):
            continue
        group_count = int(record.get("dependency_group_count", 0) or 0)
        parts.append(f"{family}({group_count}组)")
    return "，".join(parts) if parts else "无"


def c2psa_split_console_summary(metadata: dict[str, Any]) -> str:
    structure = metadata.get("yolo11_c2psa_paired_split_structure", {})
    records = (
        structure.get("records", [])
        if isinstance(structure, dict)
        and isinstance(structure.get("records", []), list)
        else []
    )
    if not records:
        return "不适用"
    record = records[0]
    return (
        f"paired split：每侧 {record.get('original_c', '?')} → "
        f"{record.get('current_c', '?')}，cv2 输入 "
        f"{record.get('cv2_in_channels', '?')}"
    )


def c3k2_chunk_console_summary(metadata: dict[str, Any]) -> str:
    structure = metadata.get("yolo11_c3k2_paired_chunk_structure", {})
    records = (
        structure.get("records", [])
        if isinstance(structure, dict)
        and isinstance(structure.get("records", []), list)
        else []
    )
    if not records:
        return "不适用"
    passed = sum(record.get("status") == "pass" for record in records)
    reductions = sorted(
        {
            (record.get("original_c"), record.get("current_c"))
            for record in records
        }
    )
    reduction_text = "，".join(f"{before}→{after}" for before, after in reductions)
    return f"{passed}/{len(records)} 个 C3k2 通过；每段宽度 {reduction_text}"


def c3k2_dynamic_console_summary(metadata: dict[str, Any]) -> str:
    structure = metadata.get("yolo11_c3k2_dynamic_concat_structure", {})
    records = (
        structure.get("records", [])
        if isinstance(structure, dict)
        and isinstance(structure.get("records", []), list)
        else []
    )
    if not records:
        return "不适用"
    selected_children = sum(
        int(record.get("selected_child_count", 0) or 0) for record in records
    )
    reductions = []
    for record in records:
        for child in record.get("children", []):
            if child.get("selected"):
                reductions.append(
                    f"{record.get('module_prefix')}.m.{child.get('child_index')}:"
                    f"{child.get('original_width')}→{child.get('current_width')}"
                )
    return (
        f"{selected_children} 个独立 segment 通过；"
        + ("，".join(reductions) if reductions else "未记录宽度变化")
    )


def c3k2_composite_console_summary(metadata: dict[str, Any]) -> str:
    structure = metadata.get("yolo11_c3k2_composite_structure", {})
    records = (
        structure.get("records", [])
        if isinstance(structure, dict)
        and isinstance(structure.get("records", []), list)
        else []
    )
    if not records:
        return "不适用"
    passed = sum(record.get("status") == "pass" for record in records)
    selected = sum(int(record.get("selected_child_count", 0) or 0) for record in records)
    reductions = "，".join(
        f"{record.get('module_prefix')}:{record.get('root_original_c')}→{record.get('current_c')}"
        for record in records
    )
    return (
        f"{passed}/{len(records)} 个父级复合合同通过；"
        f"{selected} 个独立 child-output 命中；根部 {reductions}"
    )


def output_consumer_closure_console_summary(metadata: dict[str, Any]) -> str:
    structure = metadata.get("yolo11_output_consumer_closure_structure", {})
    records = (
        structure.get("records", [])
        if isinstance(structure, dict)
        and isinstance(structure.get("records", []), list)
        else []
    )
    quarantined = int(structure.get("quarantined_contract_count", 0) or 0)
    if not records:
        return "无已解析闭包" + (f"；自动隔离 {quarantined} 个输出轴" if quarantined else "")
    passed = sum(record.get("status") == "pass" for record in records)
    selected = sum(bool(record.get("physical_output_reduced")) for record in records)
    reductions = "，".join(
        f"{record.get('module_prefix')}:{record.get('original_output_dim')}→"
        f"{record.get('current_output_dim')}"
        for record in records
        if record.get("physical_output_reduced")
    )
    return (
        f"{passed}/{len(records)} 个输出—消费者闭包通过；"
        f"{selected} 个输出轴物理收缩"
        + (f"；{reductions}" if reductions else "")
        + (f"；自动隔离 {quarantined} 个不可解析输出轴" if quarantined else "")
    )


def attention_console_summary(metadata: dict[str, Any]) -> str:
    contract = metadata.get("yolo11_attention_redundancy_contract", {})
    structure = metadata.get("yolo11_attention_structure", {})
    if not isinstance(contract, dict):
        contract = {}
    if not isinstance(structure, dict):
        structure = {}
    selected = contract.get("selected_record_count", 0)
    records = structure.get("records", []) if isinstance(structure.get("records", []), list) else []
    external_records = [
        record for record in records if record.get("prune_mode") == "external_width"
    ]
    if external_records:
        record = external_records[0]
        return (
            "内部 num_head/head_dim 固定，外部宽度 "
            f"{record.get('original_output_dim', '?')} → "
            f"{record.get('proj_out_channels', '?')}"
        )
    if int(selected or 0) <= 0:
        return "内部剪枝轴未被本轮选择"
    if not records:
        return "已被选中，但结构记录缺失"
    record = records[0]
    mode = record.get("prune_mode", "unknown")
    if mode == "num_head":
        before = record.get("original_num_heads")
        after = record.get("num_heads", "?")
        return (
            f"num_head：{before} → {after} 个 head"
            if before is not None
            else f"num_head：剩余 {after} 个 head"
        )
    if mode == "head_dim":
        before = record.get("original_head_dim")
        after = record.get("head_dim", "?")
        heads = record.get("num_heads", "?")
        return (
            f"head_dim：{before} → {after} 维/head，head 数保持 {heads}"
            if before is not None
            else f"head_dim：每个 head 剩余 {after} 维，head 数 {heads}"
        )
    return f"已被选中，模式 {mode}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_oto_trainable_parameters(model) -> int:
    """使独立 audit 的参数状态与 Ultralytics 训练器初始化后的状态一致。

    直接 ``YOLO(weights).model`` 载入的预训练权重可能全部是
    ``requires_grad=False``。Ultralytics 的 Trainer 会在训练前恢复这些参数；
    audit 也必须做同样处理，否则 OTO 虽能建图却不会产生任何可训练参数组。
    DFL 维持冻结，和 Ultralytics 默认策略一致。
    """

    enabled = 0
    for name, parameter in model.named_parameters():
        if ".dfl" in name:
            parameter.requires_grad = False
            continue
        parameter.requires_grad = True
        enabled += 1
    return enabled


def device_from_arg(device: str):
    if device.lower() == "cpu":
        return torch.device("cpu")
    if device.startswith("cuda:"):
        return torch.device(device)
    if device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def recursive_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: recursive_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [recursive_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(recursive_to_cpu(item) for item in value)
    return value


def recursive_to_device(value: Any, device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: recursive_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [recursive_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(recursive_to_device(item, device) for item in value)
    return value


C2PSA_PAIRED_SPLIT_PROFILE = "c2psa_paired_split_v1_experimental"
C3K2_PAIRED_CHUNK_PROFILE = "c3k2_paired_chunk_v1_experimental"
C3K2_DYNAMIC_CONCAT_PROFILE = "c3k2_dynamic_concat_v1_experimental"
C3K2_COMPOSITE_PROFILE = "c3k2_paired_chunk_dynamic_concat_v1_experimental"
C3K2_OUTPUT_CLOSURE_PROFILE = "c3k2_output_consumer_closure_v1_experimental"
ALL_CV1_OPEN_PROFILE = "yolo11_all_cv1_open_v1"
# v1 是早期的 Attention-num-head 联合 profile。保留它的候选与恢复语义，
# 不回写任何既有 run；v2 在 v1 的 Attention + C2PSA 输出闭包基础上纳入
# C3k2 composite 所有权。C2PSA paired-split 的“外部宽度”轴会改写同一
# PSABlock 的 Attention/FFN 外部接口，当前仍与 Attention 内部轴互斥，
# 不能作为同一轮联合候选。
ALL_VERIFIED_SAFE_PROFILE = "all_verified_safe_num_head_v1_experimental"
ALL_VERIFIED_SAFE_NUM_HEAD_V2_PROFILE = "all_verified_safe_num_head_v2_experimental"
ALL_VERIFIED_SAFE_HEAD_DIM_V2_PROFILE = "all_verified_safe_head_dim_v2_experimental"
# v4 在 v2-num-head 的联合搜索空间中接入通用“生产者输出—消费者输入”
# 闭包。它只解除 C3k2.cv2 的临时输出 guard；Detect 的语义输出 guard，以及
# Attention-num-head 与 C2PSA paired-split 的互斥关系保持不变。
ALL_VERIFIED_SAFE_NUM_HEAD_V4_OUTPUT_CLOSURE_PROFILE = (
    "all_verified_safe_num_head_v4_output_closure_experimental"
)
ALL_VERIFIED_SAFE_PROFILES = frozenset(
    {
        ALL_VERIFIED_SAFE_PROFILE,
        ALL_VERIFIED_SAFE_NUM_HEAD_V2_PROFILE,
        ALL_VERIFIED_SAFE_HEAD_DIM_V2_PROFILE,
        ALL_VERIFIED_SAFE_NUM_HEAD_V4_OUTPUT_CLOSURE_PROFILE,
    }
)
ALL_VERIFIED_SAFE_V2_PROFILES = frozenset(
    {
        ALL_VERIFIED_SAFE_NUM_HEAD_V2_PROFILE,
        ALL_VERIFIED_SAFE_HEAD_DIM_V2_PROFILE,
    }
)
# v2 的名称属于已归档 profile，不能把新合同静默写回旧 run。因此新建联合集合
# 供 v4 的重建 provenance、复合合同和编排门禁复用。
ALL_VERIFIED_SAFE_JOINT_COMPOSITE_PROFILES = frozenset(
    {
        *ALL_VERIFIED_SAFE_V2_PROFILES,
        ALL_VERIFIED_SAFE_NUM_HEAD_V4_OUTPUT_CLOSURE_PROFILE,
    }
)
C3K2_COMPOSITE_PROFILES = frozenset(
    {
        C3K2_COMPOSITE_PROFILE,
        *ALL_VERIFIED_SAFE_JOINT_COMPOSITE_PROFILES,
    }
)
# v1/v2 依赖这个临时 guard 来阻止 C3k2.cv2 的输出轴被下游共享组带动裁剪；
# v4 改由已验证的 output-consumer closure 接管该轴，故不能复用此 guard。
C3K2_COMPOSITE_FIXED_OUTPUT_GUARD_PROFILES = frozenset(
    {
        C3K2_COMPOSITE_PROFILE,
        *ALL_VERIFIED_SAFE_V2_PROFILES,
    }
)
OUTPUT_CONSUMER_CLOSURE_PROFILES = frozenset(
    {
        C3K2_OUTPUT_CLOSURE_PROFILE,
        ALL_VERIFIED_SAFE_NUM_HEAD_V4_OUTPUT_CLOSURE_PROFILE,
    }
)
ALL_VERIFIED_SAFE_REQUIRED_FAMILIES_V1 = (
    "C3k2",
    "C3k",
    "Bottleneck",
    "SPPF",
    "Detect",
    "standalone_or_boundary_Conv2d",
    "C2PSA_OUTPUT",
    "PSA_FFN",
    "YOLO11_ATTN_NUM_HEAD",
)
ALL_VERIFIED_SAFE_REQUIRED_FAMILIES_BY_PROFILE = {
    ALL_VERIFIED_SAFE_PROFILE: ALL_VERIFIED_SAFE_REQUIRED_FAMILIES_V1,
    ALL_VERIFIED_SAFE_NUM_HEAD_V2_PROFILE: (
        *ALL_VERIFIED_SAFE_REQUIRED_FAMILIES_V1,
        "C3K2_CHUNK",
        "C3K2_DYNAMIC_CONCAT",
    ),
    ALL_VERIFIED_SAFE_HEAD_DIM_V2_PROFILE: (
        *(
            family
            for family in ALL_VERIFIED_SAFE_REQUIRED_FAMILIES_V1
            if family != "YOLO11_ATTN_NUM_HEAD"
        ),
        "YOLO11_ATTN_HEAD_DIM",
        "C3K2_CHUNK",
        "C3K2_DYNAMIC_CONCAT",
    ),
    ALL_VERIFIED_SAFE_NUM_HEAD_V4_OUTPUT_CLOSURE_PROFILE: (
        *ALL_VERIFIED_SAFE_REQUIRED_FAMILIES_V1,
        "C3K2_CHUNK",
        "C3K2_DYNAMIC_CONCAT",
        "C3K2_OUTPUT_CLOSURE",
    ),
}
ATTENTION_ONLY_PROFILES = frozenset(
    {
        "c2psa_attention_num_head_v1_experimental",
        "c2psa_attention_head_dim_v1_experimental",
    }
)
MODULE_COVERAGE_PROFILES = (
    "conservative_v1",
    "bottleneck_hidden_v1_experimental",
    "csp_dense_closure_v1_experimental",
    "csp_psa_ffn_closure_v1_experimental",
    "c2psa_output_closure_v1_experimental",
    C2PSA_PAIRED_SPLIT_PROFILE,
    C3K2_PAIRED_CHUNK_PROFILE,
    C3K2_DYNAMIC_CONCAT_PROFILE,
    C3K2_COMPOSITE_PROFILE,
    C3K2_OUTPUT_CLOSURE_PROFILE,
    ALL_CV1_OPEN_PROFILE,
    "c2psa_attention_num_head_v1_experimental",
    "c2psa_attention_head_dim_v1_experimental",
    ALL_VERIFIED_SAFE_PROFILE,
    ALL_VERIFIED_SAFE_NUM_HEAD_V2_PROFILE,
    ALL_VERIFIED_SAFE_HEAD_DIM_V2_PROFILE,
    ALL_VERIFIED_SAFE_NUM_HEAD_V4_OUTPUT_CLOSURE_PROFILE,
)
ATTENTION_PROFILE_MODES = {
    "c2psa_attention_num_head_v1_experimental": "num_head",
    "c2psa_attention_head_dim_v1_experimental": "head_dim",
    ALL_VERIFIED_SAFE_PROFILE: "num_head",
    ALL_VERIFIED_SAFE_NUM_HEAD_V2_PROFILE: "num_head",
    ALL_VERIFIED_SAFE_HEAD_DIM_V2_PROFILE: "head_dim",
    ALL_VERIFIED_SAFE_NUM_HEAD_V4_OUTPUT_CLOSURE_PROFILE: "num_head",
}
DENSE_CLOSURE_PROFILES = frozenset(
    {
        "csp_dense_closure_v1_experimental",
        "csp_psa_ffn_closure_v1_experimental",
        "c2psa_output_closure_v1_experimental",
        C2PSA_PAIRED_SPLIT_PROFILE,
        C3K2_PAIRED_CHUNK_PROFILE,
        C3K2_DYNAMIC_CONCAT_PROFILE,
        C3K2_COMPOSITE_PROFILE,
        C3K2_OUTPUT_CLOSURE_PROFILE,
        ALL_VERIFIED_SAFE_NUM_HEAD_V4_OUTPUT_CLOSURE_PROFILE,
        ALL_CV1_OPEN_PROFILE,
        *ATTENTION_PROFILE_MODES,
    }
)
PSA_PARTIAL_OPEN_PROFILES = frozenset(
    {
        "csp_psa_ffn_closure_v1_experimental",
        "c2psa_output_closure_v1_experimental",
        C2PSA_PAIRED_SPLIT_PROFILE,
        ALL_CV1_OPEN_PROFILE,
        *ATTENTION_PROFILE_MODES,
    }
)
C2PSA_EXTERNAL_WIDTH_PROFILES = frozenset(
    {
        C2PSA_PAIRED_SPLIT_PROFILE,
        ALL_CV1_OPEN_PROFILE,
    }
)
C3K2_PAIRED_CHUNK_PROFILES = frozenset(
    {
        C3K2_PAIRED_CHUNK_PROFILE,
        C3K2_COMPOSITE_PROFILE,
        ALL_CV1_OPEN_PROFILE,
        ALL_VERIFIED_SAFE_NUM_HEAD_V2_PROFILE,
        ALL_VERIFIED_SAFE_HEAD_DIM_V2_PROFILE,
        ALL_VERIFIED_SAFE_NUM_HEAD_V4_OUTPUT_CLOSURE_PROFILE,
    }
)
C3K2_DYNAMIC_CONCAT_PROFILES = frozenset(
    {
        C3K2_DYNAMIC_CONCAT_PROFILE,
        C3K2_COMPOSITE_PROFILE,
        ALL_VERIFIED_SAFE_NUM_HEAD_V2_PROFILE,
        ALL_VERIFIED_SAFE_HEAD_DIM_V2_PROFILE,
        ALL_VERIFIED_SAFE_NUM_HEAD_V4_OUTPUT_CLOSURE_PROFILE,
    }
)
PROBE_MODULE_FAMILIES = (
    "C3k2",
    "C3k",
    "C3",
    "Bottleneck",
    "SPPF",
    "Detect",
    "C2PSA",
    "C2PSA_OUTPUT",
    "C2PSA_SPLIT",
    "C3K2_CHUNK",
    "C3K2_DYNAMIC_CONCAT",
    "C3K2_OUTPUT_CLOSURE",
    "PSABlock",
    "PSA_FFN",
    "YOLO11_ATTN_NUM_HEAD",
    "YOLO11_ATTN_HEAD_DIM",
    "standalone_or_boundary_Conv2d",
)
KNOWN_CSP_CHUNK_QUARANTINE = "known_csp_chunk_boundary_ratio_2"
CSP_CHUNK_SOURCE_QUARANTINE = "yolo11_c3k2_cv1_chunk_source_boundary"
CSP_DYNAMIC_CONCAT_QUARANTINE = "yolo11_c3k2_dynamic_concat_child_output_boundary"


def is_yolo11_attention_structure(module) -> bool:
    """按 qkv/proj/pe 合同识别官方及 OTO 物理化后的 YOLO11 Attention。"""

    qkv_conv = getattr(getattr(module, "qkv", None), "conv", None)
    proj_conv = getattr(getattr(module, "proj", None), "conv", None)
    pe_conv = getattr(getattr(module, "pe", None), "conv", None)
    return bool(
        isinstance(qkv_conv, torch.nn.Conv2d)
        and isinstance(proj_conv, torch.nn.Conv2d)
        and isinstance(pe_conv, torch.nn.Conv2d)
        and hasattr(module, "num_heads")
        and hasattr(module, "head_dim")
        and hasattr(module, "key_dim")
    )


def yolo11_attention_prune_mode(module) -> str:
    """返回统一对外使用的 YOLO11 Attention ``prune_mode``。"""

    physical_mode = getattr(module, "_oto_attention_physical_mode", None)
    if physical_mode in {"num_head", "head_dim", "external_width"}:
        return str(physical_mode)
    legacy_mode = getattr(module, "_oto_attention_prune_mode", None)
    if legacy_mode in {"num_head", "head_dim", "external_width"}:
        return str(legacy_mode)
    return str(getattr(module, "_oto_attention_active_prune_mode", "disabled"))


def configure_yolo11_attention_pruning(model, module_coverage_profile: str) -> list[dict[str, Any]]:
    """仅由实验 profile 给 YOLO11 Attention 声明剪枝模式。

    该函数不改变 forward，也不硬编码层号；OTO 核心只读取模块上的结构合同。
    非 Attention profile 会显式关闭本轮 active mode，防止同一模型对象重复
    建图时串模式；物理子网的 physical mode provenance 不会被覆盖。
    """
    mode = ATTENTION_PROFILE_MODES.get(module_coverage_profile, "disabled")
    records: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not is_yolo11_attention_structure(module):
            continue
        qkv_conv = getattr(getattr(module, "qkv", None), "conv", None)
        proj_conv = getattr(getattr(module, "proj", None), "conv", None)
        pe_conv = getattr(getattr(module, "pe", None), "conv", None)
        # 兼容旧 v3/v4 custom checkpoint：旧字段记录的是物理来源，但过去也
        # 被误作运行开关。只在动态物理子网上把它迁移到新 physical 字段。
        if (
            getattr(module, "_oto_dynamic_inner_attention", False)
            and not hasattr(module, "_oto_attention_physical_mode")
        ):
            legacy_physical_mode = getattr(module, "_oto_attention_prune_mode", None)
            if legacy_physical_mode in {"num_head", "head_dim"}:
                module._oto_attention_physical_mode = legacy_physical_mode
        module._oto_attention_active_prune_mode = mode
        if module_coverage_profile in ATTENTION_PROFILE_MODES:
            qkv_bn = getattr(module.qkv, "bn", None)
            pe_bn = getattr(module.pe, "bn", None)
            expected_qkv = int(module.num_heads) * (
                int(module.key_dim) * 2 + int(module.head_dim)
            )
            expected_value = int(module.num_heads) * int(module.head_dim)
            physical_issues = []
            if qkv_conv.out_channels != expected_qkv:
                physical_issues.append(
                    f"qkv.out={qkv_conv.out_channels}, expected={expected_qkv}"
                )
            if not isinstance(qkv_bn, torch.nn.BatchNorm2d) or qkv_bn.num_features != expected_qkv:
                physical_issues.append(
                    f"qkv.bn={type(qkv_bn).__name__}/"
                    f"{getattr(qkv_bn, 'num_features', None)}, expected=BatchNorm2d/{expected_qkv}"
                )
            if not (
                pe_conv.in_channels == expected_value
                and pe_conv.out_channels == expected_value
                and pe_conv.groups == expected_value
            ):
                physical_issues.append(
                    f"pe(in/out/groups)={pe_conv.in_channels}/{pe_conv.out_channels}/"
                    f"{pe_conv.groups}, expected={expected_value}/{expected_value}/{expected_value}"
                )
            if not isinstance(pe_bn, torch.nn.BatchNorm2d) or pe_bn.num_features != expected_value:
                physical_issues.append(
                    f"pe.bn={type(pe_bn).__name__}/"
                    f"{getattr(pe_bn, 'num_features', None)}, expected=BatchNorm2d/{expected_value}"
                )
            if proj_conv.in_channels != expected_value:
                physical_issues.append(
                    f"proj.in={proj_conv.in_channels}, expected={expected_value}"
                )
            if physical_issues:
                raise RuntimeError(
                    f"Attention profile 模块 {name} 物理合同预检失败："
                    + "；".join(physical_issues)
                )
        records.append(
            {
                "module_prefix": name,
                "module_type": type(module).__name__,
                "prune_mode": mode,
                "num_heads": int(module.num_heads),
                "key_dim": int(module.key_dim),
                "head_dim": int(module.head_dim),
                "qkv_out_channels": int(qkv_conv.out_channels),
                "value_channels": int(pe_conv.out_channels),
                "external_output_channels": int(proj_conv.out_channels),
                # qkv/pe 是 HESSO 选择逻辑组的内部驱动参数。proj 同一个权重
                # 同时承担“内部 V 消费者输入轴”和“外部固定 C 输出轴”，旧 OTO
                # 参数组只能给同一 Parameter 一个优化器归属，因此不把 proj 的
                # 外部输出节点误列为独立 Attention 候选；物理构建仍由复合算子
                # 按 axis=1 联动裁剪，并由结构门禁验证。
                "hesso_candidate_prefixes": [
                    f"{name}.qkv.conv",
                    f"{name}.pe.conv",
                ],
                "physical_consumer_prefixes": [f"{name}.proj.conv"],
                "external_output_fixed": True,
            }
        )
    if module_coverage_profile in ATTENTION_PROFILE_MODES and not records:
        raise RuntimeError("Attention profile 未找到结构匹配的 YOLO11 qkv/proj/pe Attention。")
    return records


def _profile_index_select_parameter(parameter, preserved: list[int], axis: int):
    """YOLO11 profile 本地物理切片；不修改 OTO 核心 Parameter 工具。"""

    indexes = torch.as_tensor(preserved, dtype=torch.long, device=parameter.device)
    data = torch.index_select(parameter.detach(), axis, indexes).clone()
    result = torch.nn.Parameter(data, requires_grad=parameter.requires_grad)
    if parameter.grad is not None:
        result.grad = torch.index_select(parameter.grad.detach(), axis, indexes).clone()
    return result


def _profile_preserved_indexes(width: int, pruned_idxes: list[int]) -> list[int]:
    pruned = {int(index) for index in pruned_idxes}
    if not pruned:
        return list(range(int(width)))
    if min(pruned) < 0 or max(pruned) >= int(width):
        raise IndexError(
            f"profile 物理切片索引越界：width={width}, pruned={sorted(pruned)}"
        )
    preserved = [index for index in range(int(width)) if index not in pruned]
    if not preserved:
        raise ValueError("profile 物理切片至少必须保留一个通道。")
    return preserved


def _profile_prune_conv_out(conv, pruned_idxes: list[int]) -> None:
    if not isinstance(conv, torch.nn.Conv2d) or int(conv.groups) != 1:
        raise TypeError(
            "C2PSA paired-split 仅支持 groups=1 的 Conv2d 输出裁剪；"
            f"actual={type(conv).__name__}, groups={getattr(conv, 'groups', None)}"
        )
    preserved = _profile_preserved_indexes(conv.out_channels, pruned_idxes)
    conv.weight = _profile_index_select_parameter(conv.weight, preserved, 0)
    if conv.bias is not None:
        conv.bias = _profile_index_select_parameter(conv.bias, preserved, 0)
    conv.out_channels = len(preserved)


def _profile_prune_conv_in(conv, pruned_idxes: list[int]) -> None:
    if not isinstance(conv, torch.nn.Conv2d) or int(conv.groups) != 1:
        raise TypeError(
            "C2PSA paired-split 仅支持 groups=1 的 Conv2d 输入裁剪；"
            f"actual={type(conv).__name__}, groups={getattr(conv, 'groups', None)}"
        )
    preserved = _profile_preserved_indexes(conv.in_channels, pruned_idxes)
    conv.weight = _profile_index_select_parameter(conv.weight, preserved, 1)
    conv.in_channels = len(preserved)


def _profile_prune_batch_norm(bn, pruned_idxes: list[int]) -> None:
    if not isinstance(bn, torch.nn.BatchNorm2d):
        raise TypeError(
            "C2PSA paired-split 需要 BatchNorm2d；"
            f"actual={type(bn).__name__}"
        )
    preserved = _profile_preserved_indexes(bn.num_features, pruned_idxes)
    if bn.affine:
        bn.weight = _profile_index_select_parameter(bn.weight, preserved, 0)
        bn.bias = _profile_index_select_parameter(bn.bias, preserved, 0)
    if bn.track_running_stats:
        indexes = torch.as_tensor(
            preserved,
            dtype=torch.long,
            device=bn.running_mean.device,
        )
        bn.running_mean = torch.index_select(bn.running_mean, 0, indexes).clone()
        bn.running_var = torch.index_select(bn.running_var, 0, indexes).clone()
    bn.num_features = len(preserved)


def _c2psa_external_parameter_prefixes(
    module_prefix: str,
    module,
) -> tuple[set[str], set[str], set[str]]:
    paired_prefixes = {
        f"{module_prefix}.cv1.conv",
        f"{module_prefix}.cv1.bn",
    }
    singleton_prefixes: set[str] = set()
    # 不同 Ultralytics/ONNX trace 版本可能把 Attention 内部参数或外部输入
    # 消费者并入同一个 residual 依赖组。它们必须继续训练，但不能按 paired
    # split 的输出轴参与 HESSO 评分或置零；输入轴由物理合同显式联动。
    nonstructural_prefixes: set[str] = {
        f"{module_prefix}.cv2.conv",
        f"{module_prefix}.cv2.bn",
    }
    blocks = list(getattr(module, "m", ()))
    if not blocks:
        raise RuntimeError(f"C2PSA {module_prefix} 没有 PSABlock 子模块。")
    for block_index, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        ffn = getattr(block, "ffn", None)
        if not is_yolo11_attention_structure(attn) or not isinstance(
            ffn, torch.nn.Sequential
        ) or len(ffn) < 2:
            raise RuntimeError(
                f"C2PSA {module_prefix}.m.{block_index} 不是受支持的 PSABlock 结构。"
            )
        singleton_prefixes.update(
            {
                f"{module_prefix}.m.{block_index}.attn.proj.conv",
                f"{module_prefix}.m.{block_index}.attn.proj.bn",
                f"{module_prefix}.m.{block_index}.ffn.1.conv",
                f"{module_prefix}.m.{block_index}.ffn.1.bn",
            }
        )
        nonstructural_prefixes.update(
            {
                f"{module_prefix}.m.{block_index}.attn.qkv.conv",
                f"{module_prefix}.m.{block_index}.attn.qkv.bn",
                f"{module_prefix}.m.{block_index}.attn.pe.conv",
                f"{module_prefix}.m.{block_index}.attn.pe.bn",
                f"{module_prefix}.m.{block_index}.ffn.0.conv",
                f"{module_prefix}.m.{block_index}.ffn.0.bn",
            }
        )
    return paired_prefixes, singleton_prefixes, nonstructural_prefixes


class _YOLO11C2PSAPairedSplitNodeGroup(OTONodeGroup):
    """YOLO11 profile 本地 C2PSA paired-split 逻辑组。

    该类只包装已经由 OTO 建好的普通 NodeGroup：逻辑组数由 ``2c`` 改为
    ``c``，并用通用 ``INDEX_GROUPS`` 表达 ``[i, c+i]``。它不会注册到 OTO
    ``COMPOSED_MODULES``，也不会改变其他模型或其他节点组的行为。
    """

    @property
    def num_groups(self):
        return int(self._oto_c2psa_split_contract["logical_group_count"])

    def get_param_groups(self):
        param_group = OTONodeGroup.get_param_groups(self)
        contract = self._oto_c2psa_split_contract
        logical_groups = int(contract["logical_group_count"])
        paired_groups = contract["paired_output_index_groups"]
        singleton_groups = contract["singleton_output_index_groups"]
        paired_prefixes = tuple(contract["paired_parameter_prefixes"])
        singleton_prefixes = tuple(contract["singleton_parameter_prefixes"])
        nonstructural_prefixes = tuple(
            contract.get("nonstructural_parameter_prefixes", [])
        )
        transforms = []
        unexpected = []
        for parameter_name in param_group["p_names"]:
            if parameter_name.startswith(paired_prefixes):
                transforms.append(make_index_groups_transform(paired_groups, axis=0))
            elif parameter_name.startswith(singleton_prefixes):
                transforms.append(make_index_groups_transform(singleton_groups, axis=0))
            elif nonstructural_prefixes and parameter_name.startswith(
                nonstructural_prefixes
            ):
                transforms.append(TensorTransform.NO_PRUNE)
            else:
                unexpected.append(parameter_name)
                transforms.append(TensorTransform.NO_PRUNE)
        if unexpected:
            raise RuntimeError(
                "C2PSA paired-split 参数闭包出现未分类参数；拒绝继续："
                f"{unexpected}"
            )
        param_group["num_groups"] = logical_groups
        param_group["p_transform"] = transforms
        param_group["c2psa_contract"] = "yolo11_c2psa_paired_split_v1"
        param_group["c2psa_module_prefix"] = contract["module_prefix"]
        param_group["c2psa_original_segment_width"] = logical_groups
        param_group["parameter_role"] = "c2psa_external_residual_width_owner"
        param_group["parameter_ownership_priority"] = 90
        return param_group

    def prune_out_dim(self, global_skip_modules=set()):
        logical_pruned = sorted(
            {int(index) for index in self.pruning_redundant_idxes}
        )
        if not logical_pruned:
            return
        contract = self._oto_c2psa_split_contract
        module = self._oto_c2psa_module
        graph = self._oto_c2psa_graph
        original_c = int(contract["logical_group_count"])
        if len(logical_pruned) >= original_c:
            raise ValueError("C2PSA paired-split 至少必须保留一个局部通道组。")
        paired_pruned = sorted(
            [index for logical in logical_pruned for index in (logical, original_c + logical)]
        )
        current_c = original_c - len(logical_pruned)

        cv1 = getattr(module, "cv1", None)
        cv2 = getattr(module, "cv2", None)
        if not all(
            hasattr(wrapper, "conv") and hasattr(wrapper, "bn")
            for wrapper in (cv1, cv2)
        ):
            raise RuntimeError(
                f"C2PSA {contract['module_prefix']} cv1/cv2 不是受支持的 Conv 包装器。"
            )
        _profile_prune_conv_out(cv1.conv, paired_pruned)
        _profile_prune_batch_norm(cv1.bn, paired_pruned)
        _profile_prune_conv_in(cv2.conv, paired_pruned)

        output_module_ids = {id(cv1.conv), id(cv1.bn)}
        input_module_ids = {id(cv2.conv)}
        for block_index, block in enumerate(list(module.m)):
            attn = block.attn
            ffn0, ffn1 = block.ffn[0], block.ffn[1]
            if not all(
                hasattr(wrapper, "conv") and hasattr(wrapper, "bn")
                for wrapper in (attn.qkv, attn.proj, ffn0, ffn1)
            ):
                raise RuntimeError(
                    f"C2PSA {contract['module_prefix']}.m.{block_index} 内部 Conv 包装器不完整。"
                )

            if not hasattr(attn, "_oto_attention_original_num_heads"):
                attn._oto_attention_original_num_heads = int(attn.num_heads)
            if not hasattr(attn, "_oto_attention_original_head_dim"):
                attn._oto_attention_original_head_dim = int(attn.head_dim)
            if not hasattr(attn, "_oto_attention_original_key_dim"):
                attn._oto_attention_original_key_dim = int(attn.key_dim)
            if not hasattr(attn, "_oto_attention_original_output_dim"):
                attn._oto_attention_original_output_dim = int(attn.proj.conv.out_channels)

            from only_train_once.operation.yolo11_attention import (
                upgrade_yolo11_attention_module,
            )

            upgrade_yolo11_attention_module(attn, "external_width")
            _profile_prune_conv_in(attn.qkv.conv, logical_pruned)
            _profile_prune_conv_out(attn.proj.conv, logical_pruned)
            _profile_prune_batch_norm(attn.proj.bn, logical_pruned)
            _profile_prune_conv_in(ffn0.conv, logical_pruned)
            _profile_prune_conv_out(ffn1.conv, logical_pruned)
            _profile_prune_batch_norm(ffn1.bn, logical_pruned)

            inner_dim = int(attn.num_heads) * int(attn.head_dim)
            attn._oto_attention_inner_dim = inner_dim
            attn._oto_attention_contract = "yolo11_partitioned_qkv_v1"
            attn._oto_attention_physical_mode = "external_width"
            attn._oto_attention_active_prune_mode = "disabled"
            output_module_ids.update(
                {
                    id(attn),
                    id(attn.proj.conv),
                    id(attn.proj.bn),
                    id(ffn1.conv),
                    id(ffn1.bn),
                }
            )
            input_module_ids.update(
                {
                    id(attn),
                    id(attn.qkv.conv),
                    id(ffn0.conv),
                }
            )

        if not hasattr(module, "_oto_c2psa_original_c"):
            module._oto_c2psa_original_c = original_c
        if not hasattr(module, "_oto_c2psa_original_output_dim"):
            module._oto_c2psa_original_output_dim = int(cv2.conv.out_channels)
        module.c = current_c
        module._oto_c2psa_current_c = current_c
        module._oto_c2psa_contract = "yolo11_c2psa_paired_split_v1"
        module._oto_c2psa_physical_mode = "paired_split"

        for node in graph.nodes.values():
            node_module = getattr(getattr(node, "op", None), "module", None)
            module_id = id(node_module)
            if module_id in output_module_ids:
                node.pruned_status["out_dim"] = True
            if module_id in input_module_ids:
                node.pruned_status["in_dim"] = True
        self._oto_c2psa_physical_pruned = True


# 保持 OTO 中现有 ``type(node_group).__name__ == 'NodeGroup'`` 的兼容判断。
_YOLO11C2PSAPairedSplitNodeGroup.__name__ = "NodeGroup"


def apply_c2psa_paired_split_contract(oto, model, module_coverage_profile: str) -> list[dict[str, Any]]:
    """在 YOLO11 profile 层把 C2PSA 外部残差组改写为 paired-split 合同。"""

    if module_coverage_profile not in C2PSA_EXTERNAL_WIDTH_PROFILES:
        return []
    records: list[dict[str, Any]] = []
    for module_prefix, module in model.named_modules():
        if not module_prefix or type(module).__name__ != "C2PSA":
            continue
        c = int(getattr(module, "c", 0))
        cv1 = getattr(module, "cv1", None)
        cv2 = getattr(module, "cv2", None)
        if c <= 0 or not all(
            hasattr(wrapper, "conv") and hasattr(wrapper, "bn")
            for wrapper in (cv1, cv2)
        ):
            raise RuntimeError(f"C2PSA {module_prefix} 缺少合法 c/cv1/cv2 结构。")
        if int(cv1.conv.out_channels) != 2 * c or int(cv2.conv.in_channels) != 2 * c:
            raise RuntimeError(
                f"C2PSA {module_prefix} split/concat 原始宽度不一致："
                f"c={c}, cv1.out={cv1.conv.out_channels}, cv2.in={cv2.conv.in_channels}"
            )

        cv1_weight_name = f"{module_prefix}.cv1.conv.weight"
        node_groups = [
            node_group
            for node_group in oto._graph.node_groups.values()
            if cv1_weight_name in set(node_group.param_names)
        ]
        if len(node_groups) != 1:
            raise RuntimeError(
                f"C2PSA {module_prefix} cv1 必须唯一落入一个 OTO 节点组，实际={len(node_groups)}。"
            )
        node_group = node_groups[0]
        if type(node_group).__name__ != "NodeGroup" or bool(node_group.is_auxiliary):
            raise RuntimeError(
                f"C2PSA {module_prefix} 外部宽度组类型错误："
                f"type={type(node_group).__name__}, auxiliary={node_group.is_auxiliary}"
            )
        if int(node_group.num_groups) != 2 * c:
            raise RuntimeError(
                f"C2PSA {module_prefix} 原始 OTO 组数应为 2c={2*c}，"
                f"实际={node_group.num_groups}。"
            )

        (
            paired_prefixes,
            singleton_prefixes,
            nonstructural_prefixes,
        ) = _c2psa_external_parameter_prefixes(module_prefix, module)
        actual_state_names = set(node_group.param_names)
        structural_prefixes = tuple(paired_prefixes | singleton_prefixes)
        optional_nonstructural_prefixes = tuple(nonstructural_prefixes)
        expected_parameters = {
            name
            for name, _ in model.named_parameters()
            if name.startswith(structural_prefixes)
        }
        allowed_nonstructural_parameters = {
            name
            for name, _ in model.named_parameters()
            if optional_nonstructural_prefixes
            and name.startswith(optional_nonstructural_prefixes)
        }
        # OTO 图节点的 ``param_names`` 同时可能记录 BatchNorm 的运行统计量。
        # running_mean/running_var 属于 buffer，不进入 HESSO 参数组，但物理裁剪时
        # 仍会由本 profile 的 BatchNorm 合同同步切片；因此应作为合法闭包成员，
        # 不能误报为 unexpected。未被 trace 暴露的 buffer 不要求强制出现。
        all_allowed_prefixes = structural_prefixes + optional_nonstructural_prefixes
        allowed_buffers = {
            name
            for name, _ in model.named_buffers()
            if all_allowed_prefixes and name.startswith(all_allowed_prefixes)
        }
        missing_parameters = sorted(expected_parameters - actual_state_names)
        unexpected_state_names = sorted(
            actual_state_names
            - expected_parameters
            - allowed_nonstructural_parameters
            - allowed_buffers
        )
        if missing_parameters or unexpected_state_names:
            raise RuntimeError(
                f"C2PSA {module_prefix} paired-split 参数闭包不精确："
                f"missing={missing_parameters}, unexpected={unexpected_state_names}"
            )

        contract = {
            "contract": "yolo11_c2psa_paired_split_v1",
            "module_prefix": module_prefix,
            "logical_group_count": c,
            "source_output_width": 2 * c,
            "segment_count": 2,
            "segment_original_widths": [c, c],
            "segment_source_offsets": [0, c],
            "paired_output_index_groups": [[index, c + index] for index in range(c)],
            "singleton_output_index_groups": [[index] for index in range(c)],
            "paired_parameter_prefixes": sorted(paired_prefixes),
            "singleton_parameter_prefixes": sorted(singleton_prefixes),
            "nonstructural_parameter_prefixes": sorted(nonstructural_prefixes),
            "consumer": f"{module_prefix}.cv2.conv",
            "external_output_fixed": True,
            "attention_internal_axes_fixed": True,
        }
        node_group.__class__ = _YOLO11C2PSAPairedSplitNodeGroup
        node_group._oto_c2psa_split_contract = contract
        node_group._oto_c2psa_module = module
        node_group._oto_c2psa_graph = oto._graph
        node_group.is_prunable = True
        records.append(
            {
                **contract,
                "node_group_id": str(node_group.id),
                "parameter_names": sorted(expected_parameters),
                "nonstructural_parameter_names": sorted(
                    actual_state_names & allowed_nonstructural_parameters
                ),
                "buffer_names": sorted(actual_state_names & allowed_buffers),
                "state_names": sorted(actual_state_names),
                "status": "configured",
            }
        )
    if not records:
        raise RuntimeError("C2PSA paired-split profile 未找到任何 C2PSA 模块。")
    return records


def _c3k2_external_contract_layout(module_prefix: str, module) -> dict[str, Any]:
    """返回 C3k2 外部隐藏宽度的统一 paired-chunk 布局。"""

    c = int(getattr(module, "c", 0))
    cv1 = getattr(module, "cv1", None)
    cv2 = getattr(module, "cv2", None)
    children = list(getattr(module, "m", ()))
    if c <= 0 or not children:
        raise RuntimeError(f"C3k2 {module_prefix} 缺少合法 c 或 self.m。")
    if not all(
        hasattr(wrapper, "conv") and hasattr(wrapper, "bn")
        for wrapper in (cv1, cv2)
    ):
        raise RuntimeError(f"C3k2 {module_prefix} cv1/cv2 不是受支持的 Conv 包装器。")
    expected_cv1_out = 2 * c
    expected_cv2_in = (2 + len(children)) * c
    if (
        int(cv1.conv.out_channels) != expected_cv1_out
        or int(cv2.conv.in_channels) != expected_cv2_in
    ):
        raise RuntimeError(
            f"C3k2 {module_prefix} chunk/concat 原始宽度不一致："
            f"c={c}, cv1.out={cv1.conv.out_channels}, "
            f"cv2.in={cv2.conv.in_channels}, children={len(children)}"
        )

    child_records: list[dict[str, Any]] = []
    output_prefixes: set[str] = set()
    for child_index, child in enumerate(children):
        child_prefix = f"{module_prefix}.m.{child_index}"
        child_type = type(child).__name__
        if child_type == "Bottleneck":
            wrappers = (child.cv1, child.cv2)
            if not all(
                hasattr(wrapper, "conv") and hasattr(wrapper, "bn")
                for wrapper in wrappers
            ):
                raise RuntimeError(f"{child_prefix} Bottleneck Conv 包装器不完整。")
            if (
                int(child.cv1.conv.in_channels) != c
                or int(child.cv2.conv.out_channels) != c
            ):
                raise RuntimeError(
                    f"{child_prefix} 外部宽度不等于 C3k2.c={c}："
                    f"cv1.in={child.cv1.conv.in_channels}, "
                    f"cv2.out={child.cv2.conv.out_channels}"
                )
            input_paths = ["cv1"]
            output_path = "cv2"
        elif child_type in {"C3", "C3k"}:
            wrappers = (child.cv1, child.cv2, child.cv3)
            if not all(
                hasattr(wrapper, "conv") and hasattr(wrapper, "bn")
                for wrapper in wrappers
            ):
                raise RuntimeError(f"{child_prefix} {child_type} Conv 包装器不完整。")
            if (
                int(child.cv1.conv.in_channels) != c
                or int(child.cv2.conv.in_channels) != c
                or int(child.cv3.conv.out_channels) != c
            ):
                raise RuntimeError(
                    f"{child_prefix} 外部宽度不等于 C3k2.c={c}："
                    f"cv1.in={child.cv1.conv.in_channels}, "
                    f"cv2.in={child.cv2.conv.in_channels}, "
                    f"cv3.out={child.cv3.conv.out_channels}"
                )
            input_paths = ["cv1", "cv2"]
            output_path = "cv3"
        else:
            raise RuntimeError(
                f"C3k2 {module_prefix} 暂不支持子模块 {child_prefix}: {child_type}。"
            )
        output_prefixes.update(
            {
                f"{child_prefix}.{output_path}.conv",
                f"{child_prefix}.{output_path}.bn",
            }
        )
        child_records.append(
            {
                "index": child_index,
                "prefix": child_prefix,
                "type": child_type,
                "input_paths": input_paths,
                "output_path": output_path,
            }
        )

    return {
        "module_prefix": module_prefix,
        "logical_group_count": c,
        "segment_count": 2 + len(children),
        "source_output_width": 2 * c,
        "concat_input_width": (2 + len(children)) * c,
        "paired_parameter_prefixes": sorted(
            {
                f"{module_prefix}.cv1.conv",
                f"{module_prefix}.cv1.bn",
            }
        ),
        "singleton_parameter_prefixes": sorted(output_prefixes),
        "paired_output_index_groups": [[index, c + index] for index in range(c)],
        "singleton_output_index_groups": [[index] for index in range(c)],
        "concat_input_index_groups": [
            [segment * c + index for segment in range(2 + len(children))]
            for index in range(c)
        ],
        "children": child_records,
    }


class _YOLO11C3k2PairedChunkNodeGroup(OTONodeGroup):
    """YOLO11 profile 本地 C3k2 paired-chunk 逻辑组。"""

    @property
    def num_groups(self):
        return int(self._oto_c3k2_chunk_contract["logical_group_count"])

    def get_param_groups(self):
        param_group = OTONodeGroup.get_param_groups(self)
        contract = self._oto_c3k2_chunk_contract
        paired_prefixes = tuple(contract["paired_parameter_prefixes"])
        singleton_prefixes = tuple(contract["singleton_parameter_prefixes"])

        owner_parameters = getattr(self, "_oto_c3k2_owner_parameters", {})
        # C3/C3k 的 cv3 在复合 profile 中由 DynamicConcat owner 独占。旧 OTO 图
        # 有时会把它们与根部 cv1 放入同一个依赖组；必须先从根部参数组剔除，再由
        # Graph 的参数去重交给 child owner，不能让一个张量轴被两条 transform 同时处理。
        owned_names = set(owner_parameters)
        original_names = list(param_group["p_names"])
        kept_indexes = [
            index for index, parameter_name in enumerate(original_names)
            if parameter_name in owned_names
        ]
        for key in ("p_names", "params", "op_names", "node_ids"):
            values = param_group.get(key)
            if isinstance(values, list) and len(values) == len(original_names):
                param_group[key] = [values[index] for index in kept_indexes]

        # C3k 型 child output 在原始 OTO 图中也可能属于独立节点组。其余根部
        # owner 参数由当前高优先级组原子接管，Graph 的参数去重会从旧组移除它们。
        existing_names = set(param_group["p_names"])
        fallback_node_id = next(iter(self.nodes), "")
        for parameter_name, parameter in owner_parameters.items():
            if parameter_name in existing_names or not parameter.requires_grad:
                continue
            param_group["p_names"].append(parameter_name)
            param_group["params"].append(parameter)
            param_group["op_names"].append("c3k2_paired_chunk")
            param_group.setdefault("node_ids", []).append(fallback_node_id)
            existing_names.add(parameter_name)

        transforms = []
        unexpected = []
        for parameter_name in param_group["p_names"]:
            if parameter_name.startswith(paired_prefixes):
                transforms.append(
                    make_index_groups_transform(
                        contract["paired_output_index_groups"], axis=0
                    )
                )
            elif singleton_prefixes and parameter_name.startswith(singleton_prefixes):
                transforms.append(
                    make_index_groups_transform(
                        contract["singleton_output_index_groups"], axis=0
                    )
                )
            else:
                unexpected.append(parameter_name)
                transforms.append(TensorTransform.NO_PRUNE)
        if unexpected:
            raise RuntimeError(
                "C3k2 paired-chunk 参数闭包出现未分类参数；拒绝继续："
                f"{unexpected}"
            )
        param_group["num_groups"] = self.num_groups
        param_group["p_transform"] = transforms
        param_group["c3k2_contract"] = "yolo11_c3k2_paired_chunk_v1"
        param_group["c3k2_module_prefix"] = contract["module_prefix"]
        param_group["c3k2_original_segment_width"] = self.num_groups
        param_group["parameter_role"] = "c3k2_paired_chunk_external_width_owner"
        param_group["parameter_ownership_priority"] = 90
        return param_group

    def prune_out_dim(self, global_skip_modules=set()):
        composite_runtime = getattr(self, "_oto_c3k2_composite_runtime", None)
        if isinstance(composite_runtime, dict):
            _apply_c3k2_composite_physical_pruning(composite_runtime)
            return
        if getattr(self, "_oto_c3k2_physical_pruned", False):
            return
        logical_pruned = sorted({int(index) for index in self.pruning_redundant_idxes})
        if not logical_pruned:
            return
        contract = self._oto_c3k2_chunk_contract
        module = self._oto_c3k2_module
        graph = self._oto_c3k2_graph
        original_c = int(contract["logical_group_count"])
        if len(logical_pruned) >= original_c:
            raise ValueError("C3k2 paired-chunk 至少必须保留一个局部通道组。")
        current_c = original_c - len(logical_pruned)
        paired_pruned = sorted(
            index
            for logical in logical_pruned
            for index in (logical, original_c + logical)
        )
        concat_pruned = sorted(
            index
            for logical in logical_pruned
            for index in contract["concat_input_index_groups"][logical]
        )

        _profile_prune_conv_out(module.cv1.conv, paired_pruned)
        _profile_prune_batch_norm(module.cv1.bn, paired_pruned)
        _profile_prune_conv_in(module.cv2.conv, concat_pruned)

        output_module_ids = {id(module.cv1.conv), id(module.cv1.bn)}
        input_module_ids = {id(module.cv2.conv)}
        for child_record, child in zip(contract["children"], list(module.m)):
            child_type = child_record["type"]
            if child_type == "Bottleneck":
                _profile_prune_conv_in(child.cv1.conv, logical_pruned)
                _profile_prune_conv_out(child.cv2.conv, logical_pruned)
                _profile_prune_batch_norm(child.cv2.bn, logical_pruned)
                input_module_ids.add(id(child.cv1.conv))
                output_module_ids.update({id(child.cv2.conv), id(child.cv2.bn)})
            elif child_type in {"C3", "C3k"}:
                _profile_prune_conv_in(child.cv1.conv, logical_pruned)
                _profile_prune_conv_in(child.cv2.conv, logical_pruned)
                _profile_prune_conv_out(child.cv3.conv, logical_pruned)
                _profile_prune_batch_norm(child.cv3.bn, logical_pruned)
                input_module_ids.update({id(child.cv1.conv), id(child.cv2.conv)})
                output_module_ids.update({id(child.cv3.conv), id(child.cv3.bn)})
            else:
                raise RuntimeError(
                    f"C3k2 物理构网遇到未支持子模块：{child_type}。"
                )

        if not hasattr(module, "_oto_c3k2_original_c"):
            module._oto_c3k2_original_c = original_c
        if not hasattr(module, "_oto_c3k2_original_output_dim"):
            module._oto_c3k2_original_output_dim = int(module.cv2.conv.out_channels)
        module.c = current_c
        module._oto_c3k2_current_c = current_c
        module._oto_c3k2_contract = "yolo11_c3k2_paired_chunk_v1"
        module._oto_c3k2_physical_mode = "paired_chunk"

        for node in graph.nodes.values():
            node_module = getattr(getattr(node, "op", None), "module", None)
            module_id = id(node_module)
            if module_id in output_module_ids:
                node.pruned_status["out_dim"] = True
            if module_id in input_module_ids:
                node.pruned_status["in_dim"] = True
        self._oto_c3k2_physical_pruned = True


_YOLO11C3k2PairedChunkNodeGroup.__name__ = "NodeGroup"


def apply_c3k2_paired_chunk_contract(
    oto,
    model,
    module_coverage_profile: str,
) -> list[dict[str, Any]]:
    """一次性把全部 C3k2 根部 cv1 改写为 paired-chunk 逻辑组。"""

    if module_coverage_profile not in C3K2_PAIRED_CHUNK_PROFILES:
        return []
    records: list[dict[str, Any]] = []
    for module_prefix, module in model.named_modules():
        if not module_prefix or type(module).__name__ != "C3k2":
            continue
        layout = _c3k2_external_contract_layout(module_prefix, module)
        cv1_weight_name = f"{module_prefix}.cv1.conv.weight"
        node_groups = [
            node_group
            for node_group in oto._graph.node_groups.values()
            if cv1_weight_name in set(node_group.param_names)
        ]
        if len(node_groups) != 1:
            raise RuntimeError(
                f"C3k2 {module_prefix} cv1 必须唯一落入一个 OTO 节点组，"
                f"实际={len(node_groups)}。"
            )
        node_group = node_groups[0]
        if type(node_group).__name__ != "NodeGroup" or bool(node_group.is_auxiliary):
            raise RuntimeError(
                f"C3k2 {module_prefix} chunk 源组类型错误："
                f"type={type(node_group).__name__}, auxiliary={node_group.is_auxiliary}"
            )
        expected_original_groups = 2 * int(layout["logical_group_count"])
        if int(node_group.num_groups) != expected_original_groups:
            raise RuntimeError(
                f"C3k2 {module_prefix} 原始 OTO 组数应为 2c={expected_original_groups}，"
                f"实际={node_group.num_groups}。"
            )

        paired_prefixes = tuple(layout["paired_parameter_prefixes"])
        singleton_prefixes = tuple(layout["singleton_parameter_prefixes"])
        allowed_prefixes = paired_prefixes + singleton_prefixes
        actual_state_names = set(node_group.param_names)
        expected_root_parameters = {
            name
            for name, _ in model.named_parameters()
            if name.startswith(paired_prefixes)
        }
        allowed_parameters = {
            name
            for name, _ in model.named_parameters()
            if name.startswith(allowed_prefixes)
        }
        allowed_buffers = {
            name
            for name, _ in model.named_buffers()
            if name.startswith(allowed_prefixes)
        }
        missing_root_parameters = sorted(expected_root_parameters - actual_state_names)
        unexpected_state_names = sorted(
            actual_state_names - allowed_parameters - allowed_buffers
        )
        if missing_root_parameters or unexpected_state_names:
            raise RuntimeError(
                f"C3k2 {module_prefix} paired-chunk 参数闭包不精确："
                f"missing_root={missing_root_parameters}, "
                f"unexpected={unexpected_state_names}"
            )

        composite_mode = module_coverage_profile in C3K2_COMPOSITE_PROFILES
        if composite_mode:
            # 复合 profile 中，C3/C3k 的 cv3 输出轴由 DynamicConcat owner 独占；
            # 根部 owner 仅持有 cv1 的 paired 输出及共享所有权 Bottleneck.cv2 输出。
            root_singleton_prefixes = sorted(
                {
                    f"{child_record['prefix']}.{child_record['output_path']}.conv"
                    for child_record in layout["children"]
                    if child_record["type"] == "Bottleneck"
                }
                | {
                    f"{child_record['prefix']}.{child_record['output_path']}.bn"
                    for child_record in layout["children"]
                    if child_record["type"] == "Bottleneck"
                }
            )
        else:
            root_singleton_prefixes = list(layout["singleton_parameter_prefixes"])
        root_owner_prefixes = tuple(layout["paired_parameter_prefixes"]) + tuple(
            root_singleton_prefixes
        )
        contract = {
            "contract": "yolo11_c3k2_paired_chunk_v1",
            **layout,
            "singleton_parameter_prefixes": root_singleton_prefixes,
            "external_output_fixed": True,
            "child_outputs_synchronized": True,
            "independent_dynamic_concat_child_pruning": composite_mode,
            "composite_owner_mode": composite_mode,
        }
        node_group.__class__ = _YOLO11C3k2PairedChunkNodeGroup
        node_group._oto_c3k2_chunk_contract = contract
        node_group._oto_c3k2_module = module
        node_group._oto_c3k2_graph = oto._graph
        node_group._oto_c3k2_owner_parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if name in allowed_parameters
            and name.startswith(root_owner_prefixes)
        }
        node_group.is_prunable = True
        records.append(
            {
                **contract,
                "node_group_id": str(node_group.id),
                "parameter_names": sorted(actual_state_names & allowed_parameters),
                "buffer_names": sorted(actual_state_names & allowed_buffers),
                "status": "configured",
            }
        )
    if not records:
        raise RuntimeError("C3k2 paired-chunk profile 未找到任何 C3k2 模块。")
    return records


def _c3k2_dynamic_concat_layout(module_prefix: str, module) -> dict[str, Any]:
    """返回 C3k2 动态 concat 的原始 segment 账本。"""

    c = int(getattr(module, "c", 0))
    cv1 = getattr(module, "cv1", None)
    cv2 = getattr(module, "cv2", None)
    children = list(getattr(module, "m", ()))
    if c <= 0 or not children or not all(
        hasattr(wrapper, "conv") and hasattr(wrapper, "bn")
        for wrapper in (cv1, cv2)
    ):
        raise RuntimeError(f"C3k2 {module_prefix} 缺少合法 c/cv1/cv2/self.m 结构。")
    if int(cv1.conv.out_channels) != 2 * c:
        raise RuntimeError(
            f"C3k2 {module_prefix} dynamic concat 要求 cv1.out=2c，"
            f"实际 c={c}, cv1.out={cv1.conv.out_channels}。"
        )

    segments: list[dict[str, Any]] = [
        {
            "segment_index": 0,
            "source": f"{module_prefix}.cv1.chunk0",
            "source_type": "chunk",
            "original_width": c,
            "original_offset": 0,
            "independent_candidate": False,
        },
        {
            "segment_index": 1,
            "source": f"{module_prefix}.cv1.chunk1",
            "source_type": "chunk",
            "original_width": c,
            "original_offset": c,
            "independent_candidate": False,
        },
    ]
    running_offset = 2 * c
    for child_index, child in enumerate(children):
        child_prefix = f"{module_prefix}.m.{child_index}"
        child_type = type(child).__name__
        if child_type == "Bottleneck" and hasattr(child, "cv2"):
            output_path = "cv2"
            independent_candidate = False
        elif child_type in {"C3", "C3k"} and hasattr(child, "cv3"):
            output_path = "cv3"
            independent_candidate = True
        else:
            raise RuntimeError(
                f"C3k2 {module_prefix} dynamic concat 暂不支持子模块 "
                f"{child_prefix}: {child_type}。"
            )
        output_wrapper = getattr(child, output_path)
        if not hasattr(output_wrapper, "conv") or not hasattr(output_wrapper, "bn"):
            raise RuntimeError(f"{child_prefix}.{output_path} 不是受支持的 Conv 包装器。")
        width = int(output_wrapper.conv.out_channels)
        if width <= 0 or int(output_wrapper.bn.num_features) != width:
            raise RuntimeError(
                f"{child_prefix}.{output_path} 输出/BN 宽度非法："
                f"conv={width}, bn={output_wrapper.bn.num_features}。"
            )
        segments.append(
            {
                "segment_index": 2 + child_index,
                "source": f"{child_prefix}.{output_path}",
                "source_type": child_type,
                "child_index": child_index,
                "child_prefix": child_prefix,
                "output_path": output_path,
                "output_wrapper_prefix": f"{child_prefix}.{output_path}",
                "original_width": width,
                "original_offset": running_offset,
                "independent_candidate": independent_candidate,
            }
        )
        running_offset += width

    if int(cv2.conv.in_channels) != running_offset:
        raise RuntimeError(
            f"C3k2 {module_prefix} dynamic concat 原始账本不闭合："
            f"segments={running_offset}, cv2.in={cv2.conv.in_channels}。"
        )
    return {
        "module_prefix": module_prefix,
        "root_width": c,
        "segment_count": len(segments),
        "concat_input_width": running_offset,
        "consumer": f"{module_prefix}.cv2.conv",
        "segments": segments,
    }


def _c3k2_child_output_wrapper(child):
    child_type = type(child).__name__
    if child_type == "Bottleneck" and hasattr(child, "cv2"):
        return child.cv2
    if child_type in {"C3", "C3k"} and hasattr(child, "cv3"):
        return child.cv3
    raise RuntimeError(f"不支持的 C3k2 child output 类型：{child_type}。")


class _YOLO11C3k2DynamicConcatNodeGroup(OTONodeGroup):
    """C3k2 独立 child-output 到动态 concat 消费者的本地合同。"""

    @property
    def num_groups(self):
        return int(self._oto_c3k2_dynamic_contract["logical_group_count"])

    def get_param_groups(self):
        param_group = OTONodeGroup.get_param_groups(self)
        contract = self._oto_c3k2_dynamic_contract
        output_prefixes = tuple(contract["output_parameter_prefixes"])
        index_groups = contract["output_index_groups"]
        transforms = []
        unexpected = []
        for parameter_name in param_group["p_names"]:
            if parameter_name.startswith(output_prefixes):
                transforms.append(make_index_groups_transform(index_groups, axis=0))
            else:
                unexpected.append(parameter_name)
                transforms.append(TensorTransform.NO_PRUNE)
        if unexpected:
            raise RuntimeError(
                "C3k2 dynamic-concat 参数闭包出现未分类参数；拒绝继续："
                f"{unexpected}"
            )
        param_group["num_groups"] = self.num_groups
        param_group["p_transform"] = transforms
        param_group["c3k2_dynamic_contract"] = (
            "yolo11_c3k2_segmented_dynamic_concat_v1"
        )
        param_group["c3k2_dynamic_parent_prefix"] = contract["module_prefix"]
        param_group["c3k2_dynamic_child_prefix"] = contract["child_prefix"]
        param_group["c3k2_dynamic_segment_index"] = int(
            contract["segment_index"]
        )
        param_group["c3k2_dynamic_original_width"] = self.num_groups
        param_group["parameter_role"] = "c3k2_dynamic_child_output_owner"
        param_group["parameter_ownership_priority"] = 95
        return param_group

    def prune_out_dim(self, global_skip_modules=set()):
        composite_runtime = getattr(self, "_oto_c3k2_composite_runtime", None)
        if isinstance(composite_runtime, dict):
            _apply_c3k2_composite_physical_pruning(composite_runtime)
            return
        if getattr(self, "_oto_c3k2_dynamic_physical_pruned", False):
            return
        logical_pruned = sorted({int(index) for index in self.pruning_redundant_idxes})
        if not logical_pruned:
            return
        contract = self._oto_c3k2_dynamic_contract
        parent = self._oto_c3k2_dynamic_parent
        child = self._oto_c3k2_dynamic_child
        graph = self._oto_c3k2_dynamic_graph
        output_wrapper = _c3k2_child_output_wrapper(child)
        original_width = int(contract["logical_group_count"])
        current_width_before = int(output_wrapper.conv.out_channels)
        if current_width_before != original_width:
            raise RuntimeError(
                "C3k2 dynamic-concat 定向合同要求 child output 仍处于原始宽度；"
                f"{contract['child_prefix']} current={current_width_before}, "
                f"original={original_width}。根部 paired-chunk 与独立 child 轴"
                "的组合需要后续复合所有权合同，不能在本 profile 静默叠加。"
            )
        if len(logical_pruned) >= original_width:
            raise ValueError("C3k2 dynamic child 至少必须保留一个输出通道组。")

        parent_c = int(getattr(parent, "c", 0))
        children = list(getattr(parent, "m", ()))
        child_index = int(contract["child_index"])
        if parent_c <= 0 or child_index >= len(children):
            raise RuntimeError(
                f"C3k2 dynamic-concat parent/child 索引非法："
                f"parent_c={parent_c}, child_index={child_index}。"
            )
        current_offset = 2 * parent_c
        preceding_widths: list[int] = []
        for preceding_child in children[:child_index]:
            preceding_wrapper = _c3k2_child_output_wrapper(preceding_child)
            width = int(preceding_wrapper.conv.out_channels)
            preceding_widths.append(width)
            current_offset += width
        consumer_pruned = sorted(current_offset + index for index in logical_pruned)
        cv2 = getattr(parent, "cv2", None)
        if not hasattr(cv2, "conv"):
            raise RuntimeError(
                f"C3k2 {contract['module_prefix']} 缺少 cv2.conv 消费者。"
            )
        if consumer_pruned and max(consumer_pruned) >= int(cv2.conv.in_channels):
            raise RuntimeError(
                "C3k2 dynamic-concat 消费者索引越界："
                f"offset={current_offset}, local={logical_pruned}, "
                f"cv2.in={cv2.conv.in_channels}。"
            )

        _profile_prune_conv_out(output_wrapper.conv, logical_pruned)
        _profile_prune_batch_norm(output_wrapper.bn, logical_pruned)
        _profile_prune_conv_in(cv2.conv, consumer_pruned)

        current_width = original_width - len(logical_pruned)
        child._oto_c3k2_dynamic_child_contract = (
            "yolo11_c3k2_segmented_dynamic_concat_v1"
        )
        child._oto_c3k2_dynamic_original_width = original_width
        child._oto_c3k2_dynamic_current_width = current_width
        child._oto_c3k2_dynamic_output_path = contract["output_path"]
        child._oto_c3k2_dynamic_segment_index = int(contract["segment_index"])
        if not hasattr(parent, "_oto_c3k2_dynamic_original_c"):
            parent._oto_c3k2_dynamic_original_c = int(
                contract["parent_root_width"]
            )
        if not hasattr(parent, "_oto_c3k2_dynamic_original_cv2_input"):
            parent._oto_c3k2_dynamic_original_cv2_input = int(
                contract["parent_original_concat_width"]
            )
        if not hasattr(parent, "_oto_c3k2_dynamic_original_output_dim"):
            parent._oto_c3k2_dynamic_original_output_dim = int(
                cv2.conv.out_channels
            )
        parent._oto_c3k2_dynamic_parent_contract = (
            "yolo11_c3k2_segmented_dynamic_concat_v1"
        )
        physical_segments = dict(
            getattr(parent, "_oto_c3k2_dynamic_physical_segments", {})
        )
        physical_segments[contract["child_prefix"]] = {
            "child_index": child_index,
            "segment_index": int(contract["segment_index"]),
            "original_offset": int(contract["original_offset"]),
            "current_offset_before_prune": current_offset,
            "preceding_current_widths": preceding_widths,
            "original_width": original_width,
            "current_width": current_width,
            "removed_local_indexes": logical_pruned,
        }
        parent._oto_c3k2_dynamic_physical_segments = physical_segments

        output_module_ids = {id(output_wrapper.conv), id(output_wrapper.bn)}
        input_module_ids = {id(cv2.conv)}
        for node in graph.nodes.values():
            node_module = getattr(getattr(node, "op", None), "module", None)
            module_id = id(node_module)
            if module_id in output_module_ids:
                node.pruned_status["out_dim"] = True
            if module_id in input_module_ids:
                node.pruned_status["in_dim"] = True
        self._oto_c3k2_dynamic_physical_pruned = True


_YOLO11C3k2DynamicConcatNodeGroup.__name__ = "NodeGroup"


def apply_c3k2_dynamic_concat_contract(
    oto,
    model,
    module_coverage_profile: str,
) -> list[dict[str, Any]]:
    """一次性开放全部独立 C3/C3k child-output 动态 concat 轴。"""

    if module_coverage_profile not in C3K2_DYNAMIC_CONCAT_PROFILES:
        return []
    records: list[dict[str, Any]] = []
    for module_prefix, module in model.named_modules():
        if not module_prefix or type(module).__name__ != "C3k2":
            continue
        layout = _c3k2_dynamic_concat_layout(module_prefix, module)
        for segment in layout["segments"]:
            if not segment.get("independent_candidate"):
                continue
            output_wrapper_prefix = str(segment["output_wrapper_prefix"])
            weight_name = f"{output_wrapper_prefix}.conv.weight"
            node_groups = [
                node_group
                for node_group in oto._graph.node_groups.values()
                if weight_name in set(node_group.param_names)
            ]
            if len(node_groups) != 1:
                raise RuntimeError(
                    f"C3k2 dynamic child {output_wrapper_prefix} 必须唯一落入一个 "
                    f"OTO 节点组，实际={len(node_groups)}。"
                )
            node_group = node_groups[0]
            if type(node_group).__name__ != "NodeGroup" or bool(
                node_group.is_auxiliary
            ):
                raise RuntimeError(
                    f"C3k2 dynamic child {output_wrapper_prefix} 组类型错误："
                    f"type={type(node_group).__name__}, "
                    f"auxiliary={node_group.is_auxiliary}。"
                )
            original_width = int(segment["original_width"])
            if int(node_group.num_groups) != original_width:
                raise RuntimeError(
                    f"C3k2 dynamic child {output_wrapper_prefix} 原始组数错误："
                    f"expected={original_width}, actual={node_group.num_groups}。"
                )
            output_prefixes = (
                f"{output_wrapper_prefix}.conv",
                f"{output_wrapper_prefix}.bn",
            )
            actual_state_names = set(node_group.param_names)
            expected_parameters = {
                name
                for name, _ in model.named_parameters()
                if name.startswith(output_prefixes)
            }
            allowed_buffers = {
                name
                for name, _ in model.named_buffers()
                if name.startswith(output_prefixes)
            }
            missing_parameters = sorted(expected_parameters - actual_state_names)
            unexpected_state_names = sorted(
                actual_state_names - expected_parameters - allowed_buffers
            )
            if missing_parameters or unexpected_state_names:
                raise RuntimeError(
                    f"C3k2 dynamic child {output_wrapper_prefix} 参数闭包不精确："
                    f"missing={missing_parameters}, "
                    f"unexpected={unexpected_state_names}。"
                )

            child = list(module.m)[int(segment["child_index"])]
            contract = {
                "contract": "yolo11_c3k2_segmented_dynamic_concat_v1",
                "module_prefix": module_prefix,
                "child_prefix": segment["child_prefix"],
                "child_index": int(segment["child_index"]),
                "child_type": segment["source_type"],
                "output_path": segment["output_path"],
                "output_wrapper_prefix": output_wrapper_prefix,
                "segment_index": int(segment["segment_index"]),
                "logical_group_count": original_width,
                "original_width": original_width,
                "original_offset": int(segment["original_offset"]),
                "parent_root_width": int(layout["root_width"]),
                "parent_original_concat_width": int(
                    layout["concat_input_width"]
                ),
                "parent_original_output_dim": int(module.cv2.conv.out_channels),
                "consumer": layout["consumer"],
                "output_parameter_prefixes": list(output_prefixes),
                "output_index_groups": [[index] for index in range(original_width)],
                "consumer_offset_policy": (
                    "runtime_current_offset_from_preceding_segment_widths"
                ),
                "parent_external_output_fixed": True,
                "root_paired_chunk_axis_fixed": (
                    module_coverage_profile not in C3K2_COMPOSITE_PROFILES
                ),
                "composite_root_paired_chunk_allowed": (
                    module_coverage_profile in C3K2_COMPOSITE_PROFILES
                ),
            }
            node_group.__class__ = _YOLO11C3k2DynamicConcatNodeGroup
            node_group._oto_c3k2_dynamic_contract = contract
            node_group._oto_c3k2_dynamic_parent = module
            node_group._oto_c3k2_dynamic_child = child
            node_group._oto_c3k2_dynamic_graph = oto._graph
            node_group.is_prunable = True
            records.append(
                {
                    **contract,
                    "node_group_id": str(node_group.id),
                    "parameter_names": sorted(
                        actual_state_names & expected_parameters
                    ),
                    "buffer_names": sorted(actual_state_names & allowed_buffers),
                    "status": "configured",
                }
            )
    if not records:
        raise RuntimeError(
            "C3k2 dynamic-concat profile 未找到独立 C3/C3k child-output 组。"
        )
    return records


def _node_group_redundant_indices(node_group) -> list[int]:
    """返回当前物理构网将删除的逻辑索引，统一为有序、去重的整数列表。"""

    return sorted({int(index) for index in node_group.pruning_redundant_idxes})


def _mark_c3k2_composite_node_statuses(
    graph,
    *,
    output_module_ids: set[int],
    input_module_ids: set[int],
) -> None:
    """标记已由父级复合协调器完成的输入/输出维，阻止 OTO 二次裁剪。"""

    for node in graph.nodes.values():
        node_module = getattr(getattr(node, "op", None), "module", None)
        module_id = id(node_module)
        if module_id in output_module_ids:
            node.pruned_status["out_dim"] = True
        if module_id in input_module_ids:
            node.pruned_status["in_dim"] = True


def _apply_c3k2_composite_physical_pruning(runtime: dict[str, Any]) -> None:
    """一次性执行 C3k2 根部 paired-chunk 与独立 child-output 的物理裁剪。

    ``cv2`` 的输入列由根部和 child 两个逻辑轴共同拥有。不能让两个 NodeGroup
    依次对已经变窄的 ``cv2`` 使用原始 offset；本函数在第一次进入任一 owner 时
    收集全部零组，并在原始坐标系中一次性裁剪所有受影响张量，因此构建顺序无关。
    """

    if runtime.get("applied"):
        return

    parent = runtime["parent"]
    graph = runtime["graph"]
    root_node_group = runtime["root_node_group"]
    root_contract = runtime["root_contract"]
    dynamic_nodes = runtime["dynamic_node_groups"]
    dynamic_contracts = runtime["dynamic_contracts"]
    original_c = int(root_contract["logical_group_count"])
    root_pruned = _node_group_redundant_indices(root_node_group)
    dynamic_pruned = {
        child_prefix: _node_group_redundant_indices(node_group)
        for child_prefix, node_group in dynamic_nodes.items()
    }
    if not root_pruned and not any(dynamic_pruned.values()):
        return
    if len(root_pruned) >= original_c:
        raise ValueError("C3k2 composite root 至少必须保留一个局部通道组。")

    cv1 = getattr(parent, "cv1", None)
    cv2 = getattr(parent, "cv2", None)
    children = list(getattr(parent, "m", ()))
    original_cv2_input = int(root_contract["concat_input_width"])
    if (
        int(getattr(parent, "c", 0)) != original_c
        or not all(hasattr(wrapper, "conv") and hasattr(wrapper, "bn") for wrapper in (cv1, cv2))
        or int(cv1.conv.out_channels) != 2 * original_c
        or int(cv2.conv.in_channels) != original_cv2_input
    ):
        raise RuntimeError(
            "C3k2 composite 物理构网必须从原始父模块几何开始；"
            f"parent={root_contract['module_prefix']}, c={getattr(parent, 'c', None)}, "
            f"cv1.out={getattr(getattr(cv1, 'conv', None), 'out_channels', None)}, "
            f"cv2.in={getattr(getattr(cv2, 'conv', None), 'in_channels', None)}。"
        )

    child_records_by_index = {
        int(record["index"]): record for record in root_contract["children"]
    }
    dynamic_contracts_by_child = {
        str(record["child_prefix"]): record for record in dynamic_contracts
    }
    if set(dynamic_nodes) != set(dynamic_contracts_by_child):
        raise RuntimeError(
            "C3k2 composite child owner 与冻结合同不一致："
            f"nodes={sorted(dynamic_nodes)}, contracts={sorted(dynamic_contracts_by_child)}"
        )

    # 首先在原始 cv2 输入坐标系生成所有删除列。根部逻辑轴作用于两个 chunk
    # 与共享 Bottleneck 输出；独立 C3/C3k 输出只由各自 child owner 作用。
    consumer_pruned: list[int] = []
    consumer_pruned.extend(root_pruned)
    consumer_pruned.extend(original_c + index for index in root_pruned)
    running_offset = 2 * original_c
    output_module_ids = {id(cv1.conv), id(cv1.bn)}
    input_module_ids = {id(cv2.conv)}
    physical_segments: dict[str, dict[str, Any]] = {}

    for child_index, child in enumerate(children):
        child_record = child_records_by_index.get(child_index)
        if child_record is None:
            raise RuntimeError(
                f"C3k2 composite 缺少 child {child_index} 的根部合同："
                f"{root_contract['module_prefix']}。"
            )
        child_prefix = str(child_record["prefix"])
        child_type = str(child_record["type"])
        output_wrapper = _c3k2_child_output_wrapper(child)
        original_child_width = int(output_wrapper.conv.out_channels)
        if original_child_width != original_c:
            raise RuntimeError(
                "C3k2 composite 要求构网前 child 输出仍为父根部原始宽度："
                f"{child_prefix}={original_child_width}, root={original_c}。"
            )

        if child_type == "Bottleneck":
            local_pruned = root_pruned
            for input_path in child_record["input_paths"]:
                input_wrapper = getattr(child, input_path)
                if root_pruned:
                    _profile_prune_conv_in(input_wrapper.conv, root_pruned)
                input_module_ids.add(id(input_wrapper.conv))
            if root_pruned:
                _profile_prune_conv_out(output_wrapper.conv, root_pruned)
                _profile_prune_batch_norm(output_wrapper.bn, root_pruned)
                output_module_ids.update({id(output_wrapper.conv), id(output_wrapper.bn)})
            ownership = "root_paired_chunk_owner"
        elif child_type in {"C3", "C3k"}:
            dynamic_contract = dynamic_contracts_by_child.get(child_prefix)
            if dynamic_contract is None:
                raise RuntimeError(
                    f"C3k2 composite 缺少独立 child-output 合同：{child_prefix}。"
                )
            local_pruned = dynamic_pruned[child_prefix]
            expected_width = int(dynamic_contract["original_width"])
            if expected_width != original_child_width:
                raise RuntimeError(
                    "C3k2 composite child 原始宽度与动态合同不一致："
                    f"{child_prefix} contract={expected_width}, actual={original_child_width}。"
                )
            if len(local_pruned) >= original_child_width:
                raise ValueError(
                    f"C3k2 composite child 至少必须保留一个输出通道组：{child_prefix}。"
                )
            for input_path in child_record["input_paths"]:
                input_wrapper = getattr(child, input_path)
                if root_pruned:
                    _profile_prune_conv_in(input_wrapper.conv, root_pruned)
                input_module_ids.add(id(input_wrapper.conv))
            if local_pruned:
                _profile_prune_conv_out(output_wrapper.conv, local_pruned)
                _profile_prune_batch_norm(output_wrapper.bn, local_pruned)
            if local_pruned:
                output_module_ids.update({id(output_wrapper.conv), id(output_wrapper.bn)})
            ownership = "dynamic_child_output_owner"
        else:
            raise RuntimeError(f"C3k2 composite 不支持 child 类型：{child_type}。")

        consumer_pruned.extend(running_offset + index for index in local_pruned)
        current_width = original_child_width - len(local_pruned)
        if child_type in {"C3", "C3k"}:
            child._oto_c3k2_dynamic_child_contract = (
                "yolo11_c3k2_segmented_dynamic_concat_v1"
            )
            child._oto_c3k2_dynamic_original_width = original_child_width
            child._oto_c3k2_dynamic_current_width = current_width
            child._oto_c3k2_dynamic_output_path = child_record["output_path"]
            child._oto_c3k2_dynamic_segment_index = 2 + child_index
            physical_segments[child_prefix] = {
                "child_index": child_index,
                "segment_index": 2 + child_index,
                "original_offset": running_offset,
                "current_offset_before_prune": (
                    2 * (original_c - len(root_pruned))
                    + sum(
                        int(_c3k2_child_output_wrapper(previous).conv.out_channels)
                        for previous in children[:child_index]
                    )
                ),
                "original_width": original_child_width,
                "current_width": current_width,
                "removed_local_indexes": list(local_pruned),
                "ownership": ownership,
            }
        running_offset += original_child_width

    consumer_pruned = sorted(set(consumer_pruned))
    if consumer_pruned and max(consumer_pruned) >= original_cv2_input:
        raise RuntimeError(
            "C3k2 composite cv2 输入索引越界："
            f"max={max(consumer_pruned)}, cv2.in={original_cv2_input}。"
        )
    paired_pruned = sorted(
        index for logical in root_pruned for index in (logical, original_c + logical)
    )
    if paired_pruned:
        _profile_prune_conv_out(cv1.conv, paired_pruned)
        _profile_prune_batch_norm(cv1.bn, paired_pruned)
    _profile_prune_conv_in(cv2.conv, consumer_pruned)

    current_c = original_c - len(root_pruned)
    parent.c = current_c
    parent._oto_c3k2_original_c = original_c
    parent._oto_c3k2_original_output_dim = int(cv2.conv.out_channels)
    parent._oto_c3k2_contract = "yolo11_c3k2_paired_chunk_v1"
    parent._oto_c3k2_composite_contract = (
        "yolo11_c3k2_paired_chunk_dynamic_concat_v1"
    )
    parent._oto_c3k2_current_c = current_c
    parent._oto_c3k2_physical_mode = "paired_chunk_dynamic_concat_composite"
    parent._oto_c3k2_dynamic_original_c = original_c
    parent._oto_c3k2_dynamic_original_cv2_input = original_cv2_input
    parent._oto_c3k2_dynamic_original_output_dim = int(cv2.conv.out_channels)
    parent._oto_c3k2_dynamic_parent_contract = (
        "yolo11_c3k2_segmented_dynamic_concat_v1"
    )
    parent._oto_c3k2_dynamic_physical_segments = physical_segments

    _mark_c3k2_composite_node_statuses(
        graph,
        output_module_ids=output_module_ids,
        input_module_ids=input_module_ids,
    )
    root_node_group._oto_c3k2_physical_pruned = True
    for node_group in dynamic_nodes.values():
        node_group._oto_c3k2_dynamic_physical_pruned = True
    runtime["applied"] = True


def apply_c3k2_composite_ownership_contract(
    oto,
    model,
    module_coverage_profile: str,
    paired_chunk_contracts: list[dict[str, Any]],
    dynamic_concat_contracts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """将同一 C3k2 的根部与独立 child 输出组成唯一的父级物理所有权。"""

    if module_coverage_profile not in C3K2_COMPOSITE_PROFILES:
        return []

    root_groups = {
        str(contract["module_prefix"]): node_group
        for node_group in oto._graph.node_groups.values()
        if isinstance(getattr(node_group, "_oto_c3k2_chunk_contract", None), dict)
        for contract in [node_group._oto_c3k2_chunk_contract]
    }
    dynamic_groups = {
        str(contract["child_prefix"]): node_group
        for node_group in oto._graph.node_groups.values()
        if isinstance(getattr(node_group, "_oto_c3k2_dynamic_contract", None), dict)
        for contract in [node_group._oto_c3k2_dynamic_contract]
    }
    paired_by_parent = {
        str(contract["module_prefix"]): dict(contract)
        for contract in paired_chunk_contracts
    }
    dynamic_by_parent: dict[str, list[dict[str, Any]]] = {}
    for contract in dynamic_concat_contracts:
        dynamic_by_parent.setdefault(str(contract["module_prefix"]), []).append(dict(contract))

    records: list[dict[str, Any]] = []
    named_modules = dict(model.named_modules())
    for parent_prefix, child_contracts in sorted(dynamic_by_parent.items()):
        parent = named_modules.get(parent_prefix)
        root_node_group = root_groups.get(parent_prefix)
        root_contract = paired_by_parent.get(parent_prefix)
        if parent is None or root_node_group is None or root_contract is None:
            raise RuntimeError(
                "C3k2 composite 缺少父级 paired-chunk owner："
                f"parent={parent_prefix}, module={parent is not None}, "
                f"group={root_node_group is not None}, contract={root_contract is not None}。"
            )
        if not bool(root_contract.get("composite_owner_mode")):
            raise RuntimeError(f"C3k2 composite 根部合同未启用复合所有权：{parent_prefix}。")

        child_nodes: dict[str, Any] = {}
        dynamic_owner_parameters: set[str] = set()
        for contract in child_contracts:
            child_prefix = str(contract["child_prefix"])
            node_group = dynamic_groups.get(child_prefix)
            if node_group is None:
                raise RuntimeError(
                    f"C3k2 composite 未找到 child owner：{child_prefix}。"
                )
            child_nodes[child_prefix] = node_group
            dynamic_owner_parameters.update(contract.get("parameter_names", []))

        root_owner_parameters = set(
            getattr(root_node_group, "_oto_c3k2_owner_parameters", {}).keys()
        )
        ownership_overlap = sorted(root_owner_parameters & dynamic_owner_parameters)
        if ownership_overlap:
            raise RuntimeError(
                "C3k2 composite 参数所有权重叠；同一 Parameter 不允许由根部和 child 同时拥有："
                f"{ownership_overlap}"
            )

        record = {
            "contract": "yolo11_c3k2_paired_chunk_dynamic_concat_v1",
            "module_prefix": parent_prefix,
            "root_node_group_id": str(root_node_group.id),
            "root_logical_group_count": int(root_contract["logical_group_count"]),
            "root_candidate_parameter_names": [
                f"{parent_prefix}.cv1.conv.weight"
            ],
            "root_owner_parameter_names": sorted(root_owner_parameters),
            "dynamic_child_contracts": sorted(
                child_contracts,
                key=lambda item: int(item["child_index"]),
            ),
            "dynamic_child_owner_parameter_names": sorted(dynamic_owner_parameters),
            "shared_consumer": f"{parent_prefix}.cv2.conv",
            "ownership_overlap": ownership_overlap,
            "physical_pruning_policy": "parent_coordinator_single_pass_original_cv2_coordinate",
            "status": "configured",
        }
        runtime = {
            "parent": parent,
            "graph": oto._graph,
            "root_node_group": root_node_group,
            "root_contract": root_contract,
            "dynamic_node_groups": child_nodes,
            "dynamic_contracts": record["dynamic_child_contracts"],
            "record": record,
            "applied": False,
        }
        root_node_group._oto_c3k2_composite_runtime = runtime
        for node_group in child_nodes.values():
            node_group._oto_c3k2_composite_runtime = runtime
        records.append(record)

    if not records:
        raise RuntimeError("C3k2 composite profile 未找到任何独立 dynamic child-output 合同。")
    return records


def apply_output_consumer_closure_contract(
    oto,
    model,
    module_coverage_profile: str,
) -> list[dict[str, Any]]:
    """为 C3k2 对外 ``cv2`` 输出自动登记生产者—消费者闭包。

    Graph 的参数组只反映同一逻辑轴的所有权，不能代替真实的输出到输入数据流：
    普通 Conv 消费者位于另一节点组，Detect 的深度卷积消费者则与生产者共享
    axis=0 耦合轴。本合同因此同时使用“参数组所有权查询”和“图拓扑消费者查询”。

    对 channel concat，OTO 核心 ``SegmentedConcatConsumerContract`` 按 Concat
    原始输入顺序保存每段 offset/width，并在物理构网时一次性裁剪普通 Conv 的
    输入轴。残差 merge、split/slice 或不支持的 grouped Conv 仍必须 fail-closed；
    不能把同一输出的另一条分支假定为不存在。
    """

    if module_coverage_profile not in OUTPUT_CONSUMER_CLOSURE_PROFILES:
        return []

    named_modules = dict(model.named_modules())
    source_records = [
        (name, module, f"{name}.cv2.conv.weight")
        for name, module in named_modules.items()
        if name and type(module).__name__ == "C3k2"
    ]
    if not source_records:
        raise RuntimeError("output-consumer closure 未找到 C3k2 模块。")

    ownership_records = oto._graph.output_consumer_closure_records(
        producer_parameter_names=[record[2] for record in source_records]
    )
    topology_records = oto._graph.discover_output_consumer_closures(
        producer_parameter_names=[record[2] for record in source_records]
    )
    segmented_concat_registration = (
        oto._graph.register_segmented_concat_consumer_contracts(
            producer_parameter_names=[record[2] for record in source_records]
        )
    )
    segmented_concat_contracts = list(
        segmented_concat_registration.get("contracts", [])
        if isinstance(segmented_concat_registration, dict)
        else []
    )
    segmented_concat_coverage = (
        segmented_concat_registration.get("source_coverage", {})
        if isinstance(segmented_concat_registration, dict)
        else {}
    )
    topology_by_producer = {
        str(record.get("producer_parameter_name")): dict(record)
        for record in topology_records
        if isinstance(record, dict) and record.get("producer_parameter_name")
    }
    records: list[dict[str, Any]] = []

    def quarantine_source(
        *,
        module_prefix: str,
        source_parameter_name: str,
        original_output_dim: int | None,
        matches: list[dict[str, Any]],
        reason: str,
        topology: dict[str, Any] | None = None,
        concat_coverage: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """精确关闭无法证明安全的生产者所属节点组，并保留完整诊断。"""

        group_records = []
        for match in matches:
            group_id = str(match.get("node_group_id"))
            node_group = oto._graph.node_groups.get(group_id)
            was_prunable = bool(node_group.is_prunable) if node_group is not None else None
            if node_group is not None:
                node_group.is_prunable = False
            group_records.append(
                {
                    "node_group_id": group_id,
                    "is_prunable_before": was_prunable,
                    "is_prunable_after": (
                        bool(node_group.is_prunable) if node_group is not None else None
                    ),
                    "is_auxiliary": bool(node_group.is_auxiliary) if node_group is not None else None,
                    "logical_group_count": int(match.get("num_groups") or 0),
                    "dependency_closure_parameter_names": list(
                        match.get("dependency_closure_parameter_names", [])
                    ),
                }
            )
        return {
            "contract": "output_consumer_closure_v2",
            "producer_module_type": "C3k2",
            "module_prefix": module_prefix,
            "producer_wrapper_prefix": f"{module_prefix}.cv2",
            "producer_parameter_name": source_parameter_name,
            "producer_conv_prefix": source_parameter_name.rsplit(".weight", 1)[0],
            "original_output_dim": original_output_dim,
            "node_group_id": group_records[0]["node_group_id"] if len(group_records) == 1 else None,
            "consumer_input_parameter_names": [],
            "consumer_records": list((topology or {}).get("consumer_records", [])),
            "topology_unresolved_paths": list((topology or {}).get("unresolved_paths", [])),
            "segmented_concat_coverage": list(concat_coverage or []),
            "status": "quarantined_unresolved_consumer_closure",
            "quarantine_reason": reason,
            "quarantined_node_groups": group_records,
        }

    for module_prefix, module, source_parameter_name in source_records:
        cv2 = getattr(module, "cv2", None)
        conv = getattr(cv2, "conv", None)
        bn = getattr(cv2, "bn", None)
        original_output_dim = int(conv.out_channels) if isinstance(conv, torch.nn.Conv2d) else None
        matches = [
            record
            for record in ownership_records
            if source_parameter_name in record.get("producer_parameter_names", [])
        ]
        topology = topology_by_producer.get(source_parameter_name)
        concat_coverage = [
            dict(record)
            for record in segmented_concat_coverage.get(source_parameter_name, [])
            if isinstance(record, dict)
        ]
        if not isinstance(conv, torch.nn.Conv2d) or not isinstance(bn, torch.nn.BatchNorm2d):
            records.append(
                quarantine_source(
                    module_prefix=module_prefix,
                    source_parameter_name=source_parameter_name,
                    original_output_dim=original_output_dim,
                    matches=matches,
                    reason="unsupported_producer_wrapper_requires_Conv2d_and_BatchNorm2d",
                    topology=topology,
                    concat_coverage=concat_coverage,
                )
            )
            continue
        if len(matches) != 1:
            records.append(
                quarantine_source(
                    module_prefix=module_prefix,
                    source_parameter_name=source_parameter_name,
                    original_output_dim=original_output_dim,
                    matches=matches,
                    reason=f"expected_exactly_one_oto_group_but_found_{len(matches)}",
                    topology=topology,
                    concat_coverage=concat_coverage,
                )
            )
            continue
        if not isinstance(topology, dict):
            records.append(
                quarantine_source(
                    module_prefix=module_prefix,
                    source_parameter_name=source_parameter_name,
                    original_output_dim=original_output_dim,
                    matches=matches,
                    reason="producer_topology_record_missing",
                )
            )
            continue
        unresolved_paths = [
            dict(item)
            for item in topology.get("unresolved_paths", [])
            if isinstance(item, dict)
        ]
        concat_boundary_ids = {
            str(item.get("boundary_node_id"))
            for item in unresolved_paths
            if item.get("reason") == "channel_concat_requires_segment_offset_contract"
            and item.get("boundary_node_id")
        }
        non_concat_unresolved_reasons = sorted(
            {
                str(item.get("reason"))
                for item in unresolved_paths
                if item.get("reason")
                != "channel_concat_requires_segment_offset_contract"
                and item.get("reason")
            }
        )
        configured_concat_contracts = [
            record
            for record in concat_coverage
            if record.get("status") == "configured"
        ]
        configured_concat_ids = {
            str(record.get("concat_node_id"))
            for record in configured_concat_contracts
            if record.get("concat_node_id")
        }
        missing_concat_contract_ids = sorted(
            concat_boundary_ids - configured_concat_ids
        )
        unresolved_concat_records = [
            record
            for record in concat_coverage
            if record.get("status") != "configured"
        ]
        if (
            non_concat_unresolved_reasons
            or missing_concat_contract_ids
            or unresolved_concat_records
        ):
            records.append(
                quarantine_source(
                    module_prefix=module_prefix,
                    source_parameter_name=source_parameter_name,
                    original_output_dim=original_output_dim,
                    matches=matches,
                    reason=(
                        "topology_or_segmented_concat_contract_unresolved:"
                        + ",".join(
                            [*non_concat_unresolved_reasons]
                            + [
                                f"missing_concat_contract:{node_id}"
                                for node_id in missing_concat_contract_ids
                            ]
                            + [
                                str(record.get("errors") or record.get("reason"))
                                for record in unresolved_concat_records
                            ]
                            or ["no_direct_or_concat_consumer"]
                        )
                    ),
                    topology=topology,
                    concat_coverage=concat_coverage,
                )
            )
            continue
        ownership_record = dict(matches[0])
        node_group = oto._graph.node_groups.get(ownership_record["node_group_id"])
        if node_group is None:
            records.append(
                quarantine_source(
                    module_prefix=module_prefix,
                    source_parameter_name=source_parameter_name,
                    original_output_dim=original_output_dim,
                    matches=[ownership_record],
                    reason="resolved_oto_group_not_found_in_graph",
                    topology=topology,
                    concat_coverage=concat_coverage,
                )
            )
            continue
        consumer_records = [
            dict(record)
            for record in topology.get("consumer_records", [])
            if isinstance(record, dict)
        ]
        source_concat_contracts = [
            dict(contract)
            for contract in segmented_concat_contracts
            if isinstance(contract, dict)
            and contract.get("status") == "configured"
            and source_parameter_name
            in set(contract.get("active_source_parameter_names", []))
        ]
        for concat_contract in source_concat_contracts:
            for segment in concat_contract.get("segments", []):
                if (
                    isinstance(segment, dict)
                    and segment.get("is_active_owner")
                    and segment.get("producer_parameter_name") == source_parameter_name
                ):
                    consumer_records.append(
                        {
                            "consumer_node_id": concat_contract.get("consumer_node_id"),
                            "consumer_parameter_name": concat_contract.get(
                                "consumer_parameter_name"
                            ),
                            "consumer_kind": (
                                "segmented_concat_standard_conv_input_axis1"
                            ),
                            "concat_contract_id": concat_contract.get("contract_id"),
                            "concat_node_id": concat_contract.get("concat_node_id"),
                            "concat_input_index": segment.get("input_index"),
                            "concat_original_offset": segment.get("original_offset"),
                            "concat_original_width": segment.get("original_width"),
                        }
                    )
        consumer_names = sorted(
            {
                str(record["consumer_parameter_name"])
                for record in consumer_records
                if record.get("consumer_parameter_name")
            }
        )
        if not consumer_names:
            records.append(
                quarantine_source(
                    module_prefix=module_prefix,
                    source_parameter_name=source_parameter_name,
                    original_output_dim=original_output_dim,
                    matches=[ownership_record],
                    reason="resolved_topology_without_conv_weight_consumer",
                    topology=topology,
                    concat_coverage=concat_coverage,
                )
            )
            continue
        contract = {
            "contract": "output_consumer_closure_v2",
            "producer_module_type": "C3k2",
            "module_prefix": module_prefix,
            "producer_wrapper_prefix": f"{module_prefix}.cv2",
            "producer_parameter_name": source_parameter_name,
            "producer_conv_prefix": source_parameter_name.rsplit(".weight", 1)[0],
            "producer_bn_parameter_names": [
                f"{module_prefix}.cv2.bn.weight",
                f"{module_prefix}.cv2.bn.bias",
            ],
            "original_output_dim": int(conv.out_channels),
            "node_group_id": str(ownership_record["node_group_id"]),
            "logical_group_count": int(ownership_record["num_groups"]),
            "consumer_input_parameter_names": consumer_names,
            "consumer_records": consumer_records,
            "topology_unresolved_paths": [],
            "segmented_concat_contracts": source_concat_contracts,
            "dependency_closure_parameter_names": list(
                ownership_record["dependency_closure_parameter_names"]
            ),
            "physical_pruning_policy": (
                "oto_source_axis0_then_graph_input_inference_for_standard_conv;"
                "depthwise_coupled_axis0;"
                "segmented_concat_consumer_original_coordinate_once"
            ),
            "status": "configured_resolved_consumer_closure",
        }
        existing_contracts = list(
            getattr(node_group, "_oto_output_consumer_closure_contracts", [])
        )
        existing_contracts.append(contract)
        node_group._oto_output_consumer_closure_contracts = existing_contracts
        records.append(contract)
    return sorted(records, key=lambda record: record["module_prefix"])


def output_consumer_composite_ownership_report(
    output_consumer_contracts: list[dict[str, Any]],
    composite_contracts: list[dict[str, Any]],
    candidate_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    """审计 C3k2 对外输出轴与内部 composite owner 的物理所有权。

    C3k2 ``cv2`` 的输出轴可由 output-consumer closure 持有；而同一模块的
    ``cv2`` *输入轴* 仍只允许 paired-chunk 根部和 dynamic-concat child 的
    coordinator 改写。二者会触及同一卷积权重的不同维度，不能仅凭参数名相同
    误判冲突，也绝不能让同一 OTO node group 同时承担两个逻辑轴。
    """

    candidates_by_id = {
        str(record.get("node_group_id")): record
        for record in candidate_groups
        if isinstance(record, dict) and record.get("node_group_id") is not None
    }
    composite_owner_group_ids: set[str] = set()
    composite_owner_parameter_names: set[str] = set()
    for composite in composite_contracts:
        if not isinstance(composite, dict):
            continue
        root_group_id = composite.get("root_node_group_id")
        if root_group_id is not None:
            composite_owner_group_ids.add(str(root_group_id))
        composite_owner_parameter_names.update(
            str(name)
            for name in composite.get("root_owner_parameter_names", [])
            if name
        )
        composite_owner_parameter_names.update(
            str(name)
            for name in composite.get("dynamic_child_owner_parameter_names", [])
            if name
        )
        for child in composite.get("dynamic_child_contracts", []):
            if isinstance(child, dict) and child.get("node_group_id") is not None:
                composite_owner_group_ids.add(str(child["node_group_id"]))

    records: list[dict[str, Any]] = []
    for contract in output_consumer_contracts:
        if not isinstance(contract, dict):
            continue
        if contract.get("status") != "configured_resolved_consumer_closure":
            continue
        group_id = str(contract.get("node_group_id"))
        source_parameter_name = str(contract.get("producer_parameter_name") or "")
        candidate = candidates_by_id.get(group_id)
        record = {
            "module_prefix": contract.get("module_prefix"),
            "source_node_group_id": group_id,
            "source_parameter_name": source_parameter_name,
            "source_group_is_prunable_non_auxiliary": bool(
                candidate
                and candidate.get("is_prunable")
                and not candidate.get("is_auxiliary")
            ),
            "source_group_collides_composite_owner": (
                group_id in composite_owner_group_ids
            ),
            "source_parameter_collides_composite_owner": (
                source_parameter_name in composite_owner_parameter_names
            ),
            # consumer_input_parameter_names 允许属于后继 C3k2.cv1；那是输入轴
            # 同步，而非 HESSO 的输出逻辑轴所有权，故不纳入 collision 判断。
            "consumer_input_axes_are_not_owner_collisions": True,
        }
        record["status"] = (
            "pass"
            if (
                record["source_group_is_prunable_non_auxiliary"]
                and not record["source_group_collides_composite_owner"]
                and not record["source_parameter_collides_composite_owner"]
            )
            else "fail"
        )
        records.append(record)
    return {
        "contract": "yolo11_output_consumer_composite_ownership_v1",
        "status": (
            "not_applicable"
            if not output_consumer_contracts and not composite_contracts
            else "pass"
            if records and all(record["status"] == "pass" for record in records)
            else "fail"
        ),
        "resolved_output_source_count": len(records),
        "composite_owner_group_count": len(composite_owner_group_ids),
        "records": records,
    }


def segmented_concat_candidate_owner_coverage_report(
    contracts: list[dict[str, Any]],
    candidate_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    """确认 Concat 的每个可剪输出段都能同步裁消费者 input-axis。

    output-consumer closure 的显式 source 只覆盖发起闭包发现的生产者；联合
    profile 还可能让同一 Concat 的其他 Conv 输出轴被 HESSO 选中。该检查按候选
    参数而非模块路径识别这些被动 segment owner，防止其物理缩窄后遗漏消费者列。
    """

    candidates_by_parameter: dict[str, list[dict[str, Any]]] = {}
    for group in candidate_groups:
        if not isinstance(group, dict):
            continue
        for parameter_name in group.get("candidate_parameter_names", []):
            candidates_by_parameter.setdefault(str(parameter_name), []).append(group)

    records: list[dict[str, Any]] = []
    for contract in contracts:
        if not isinstance(contract, dict) or contract.get("status") != "configured":
            continue
        for segment in contract.get("segments", []):
            if not isinstance(segment, dict):
                continue
            parameter_name = str(segment.get("producer_parameter_name") or "")
            candidate_owners = [
                group
                for group in candidates_by_parameter.get(parameter_name, [])
                if group.get("is_prunable") and not group.get("is_auxiliary")
            ]
            requires_input_sync = bool(candidate_owners)
            registered_owner_id = segment.get("producer_owner_node_group_id")
            registered_as_owner = bool(
                segment.get("is_active_owner")
                or segment.get("is_passive_output_owner")
            )
            candidate_owner_ids = {
                str(group.get("node_group_id")) for group in candidate_owners
            }
            record = {
                "contract_id": contract.get("contract_id"),
                "consumer_parameter_name": contract.get("consumer_parameter_name"),
                "input_index": segment.get("input_index"),
                "producer_parameter_name": parameter_name,
                "requires_input_sync": requires_input_sync,
                "registered_as_owner": registered_as_owner,
                "registered_owner_node_group_id": registered_owner_id,
                "candidate_owner_node_group_ids": sorted(candidate_owner_ids),
                "status": (
                    "pass"
                    if (
                        not requires_input_sync
                        or (
                            registered_as_owner
                            and str(registered_owner_id) in candidate_owner_ids
                        )
                    )
                    else "fail"
                ),
            }
            records.append(record)
    return {
        "contract": "segmented_concat_candidate_owner_coverage_v1",
        "status": (
            "not_applicable"
            if not contracts
            else "pass"
            if all(record["status"] == "pass" for record in records)
            else "fail"
        ),
        "candidate_segment_count": sum(
            bool(record["requires_input_sync"]) for record in records
        ),
        "records": records,
    }


def protected_prefixes(model, module_coverage_profile: str = "conservative_v1") -> list[str]:
    """按覆盖 profile 返回本轮不允许物理删除输出通道的模块前缀。"""

    if module_coverage_profile not in MODULE_COVERAGE_PROFILES:
        raise ValueError(
            f"未知 module coverage profile：{module_coverage_profile}；"
            f"当前支持 {MODULE_COVERAGE_PROFILES}"
        )

    # 其他任务 head 始终整体保护。检测模型的 Detect 在 dense closure profile
    # 中改为只固定最终预测输出组。
    protected_types = {
        "Segment",
        "Pose",
        "OBB",
        "WorldDetect",
        "v10Detect",
    }
    if module_coverage_profile in PSA_PARTIAL_OPEN_PROFILES:
        # 累计 profile 已开放 PSABlock.ffn.0；C2PSA output profile 再开放
        # C2PSA.cv2 输出边界。Attention 仍整体保护；旧 PSA 结构没有纳入本轮
        # 合同，继续整体保护。C2PSA/PSABlock 不按父模块粗粒度保护，避免把
        # 已开放候选依赖组连带封闭。
        protected_types.add("PSA")
        if (
            module_coverage_profile not in ATTENTION_PROFILE_MODES
            and module_coverage_profile not in C2PSA_EXTERNAL_WIDTH_PROFILES
        ):
            protected_types.add("Attention")
    else:
        # C2PSA 内含卷积式 QKV attention；尚未实现 QKV/head/PE 物理合同。
        protected_types.update({"C2PSA", "PSA", "PSABlock", "Attention"})

    if module_coverage_profile not in DENSE_CLOSURE_PROFILES:
        protected_types.update({"Detect", "SPPF"})
    if module_coverage_profile == "conservative_v1":
        # P10/P30 已验证的保守策略：复合 CSP 块整体保护。
        protected_types.update({"C3k2", "C3k", "Bottleneck"})

    prefixes = [name for name, module in model.named_modules() if type(module).__name__ in protected_types]
    if (
        module_coverage_profile not in ATTENTION_PROFILE_MODES
        and module_coverage_profile not in C2PSA_EXTERNAL_WIDTH_PROFILES
    ):
        # 物理子网类型会从 Attention 变为 OTOPrunableYOLO11Attention，不能只
        # 依赖类名保护。C2PSA paired-split 需要联动缩小 Attention 外部输入/输出，
        # 因而只保持内部 qkv/pe 逻辑轴关闭，不整体保护 Attention 模块。
        prefixes.extend(
            name
            for name, module in model.named_modules()
            if name and is_yolo11_attention_structure(module)
        )

    if module_coverage_profile in PSA_PARTIAL_OPEN_PROFILES:
        # C2PSA.cv1 是 split 源，在 PairedSplit Contract 完成前始终固定。
        # PSA FFN profile 也固定 cv2；C2PSA output profile 只解除 cv2 输出保护。
        # PSABlock.ffn.1 输出参与残差，只允许其输入维跟随 ffn.0 隐藏维变化。
        if module_coverage_profile in C2PSA_EXTERNAL_WIDTH_PROFILES:
            # cv1、Attention proj 和 ffn.1 属于 paired-split 外部宽度合同；
            # cv2 只固定输出轴，其输入由合同按两段 concat 索引手动同步裁剪。
            c2psa_boundaries = ("cv2",)
            psablock_boundaries = ()
        else:
            c2psa_boundaries = (
                ("cv1",)
                if module_coverage_profile
                in {"c2psa_output_closure_v1_experimental", *ATTENTION_PROFILE_MODES}
                else ("cv1", "cv2")
            )
            psablock_boundaries = (
                ("ffn.1",)
                if module_coverage_profile in ATTENTION_PROFILE_MODES
                else ("attn", "ffn.1")
            )
        boundary_children = {
            "C2PSA": c2psa_boundaries,
            "PSABlock": psablock_boundaries,
        }
        for name, module in model.named_modules():
            for child_name in boundary_children.get(type(module).__name__, ()):
                prefixes.append(f"{name}.{child_name}" if name else child_name)

    if module_coverage_profile == "bottleneck_hidden_v1_experimental":
        # 第一阶段只允许 Bottleneck.cv1 隐藏通道变化：
        # - C3k2.cv1/cv2 保持 split/concat 边界与模块输出不变；
        # - C3k.cv1/cv2/cv3 保持两支及融合边界不变；
        # - Bottleneck.cv2 输出保持不变，使 shortcut 两端 shape 始终一致。
        # 因此 OTO 只会尝试裁剪 Bottleneck.cv1 输出，并同步其后 cv2 的输入维。
        boundary_children = {
            "C3k2": ("cv1", "cv2"),
            "C3k": ("cv1", "cv2", "cv3"),
            "Bottleneck": ("cv2",),
        }
        for name, module in model.named_modules():
            for child_name in boundary_children.get(type(module).__name__, ()):
                prefixes.append(f"{name}.{child_name}" if name else child_name)

    return sorted(set(prefix for prefix in prefixes if prefix))


def module_prefix_ancestor_families(model, module_prefix: str) -> set[str]:
    """返回候选卷积所属的已知复合模块族；无复合祖先时归为边界 Conv2d。"""

    named_modules = dict(model.named_modules())
    families: set[str] = set()
    parts = module_prefix.split(".")
    for end in range(1, len(parts) + 1):
        module = named_modules.get(".".join(parts[:end]))
        if module is not None and type(module).__name__ in PROBE_MODULE_FAMILIES:
            families.add(type(module).__name__)
        attention_mode = (
            getattr(module, "_oto_attention_active_prune_mode", None)
            if module is not None
            else None
        )
        if attention_mode in {None, "disabled"} and module is not None:
            attention_mode = getattr(module, "_oto_attention_physical_mode", None)
        if attention_mode == "num_head":
            families.add("YOLO11_ATTN_NUM_HEAD")
        elif attention_mode == "head_dim":
            families.add("YOLO11_ATTN_HEAD_DIM")
    if ".ffn.0.conv" in module_prefix and "PSABlock" in families:
        families.add("PSA_FFN")
    if module_prefix.endswith(".cv2.conv") and "C2PSA" in families:
        families.add("C2PSA_OUTPUT")
    if module_prefix.endswith(".cv2.conv") and "C3k2" in families:
        families.add("C3K2_OUTPUT_CLOSURE")
    if module_prefix.endswith(".cv1.conv"):
        owner_prefix = module_prefix[: -len(".cv1.conv")]
        owner_module = named_modules.get(owner_prefix)
        owner_type = type(owner_module).__name__ if owner_module is not None else None
        if owner_type == "C2PSA":
            families.add("C2PSA_SPLIT")
        elif owner_type == "C3k2":
            families.add("C3K2_CHUNK")
    if (
        module_prefix.endswith(".cv3.conv")
        and ".m." in module_prefix
        and "C3k2" in families
    ):
        families.add("C3K2_DYNAMIC_CONCAT")
    if not families:
        families.add("standalone_or_boundary_Conv2d")
    return families


def candidate_family_coverage_report(
    model,
    candidate_groups_before: list[dict[str, Any]],
    candidate_groups_after: list[dict[str, Any]],
) -> dict[str, Any]:
    """汇总候选依赖组在保护前后按模块族的真实可剪覆盖。

    一个候选卷积可以同时位于 Bottleneck/C3k/C3k2 等嵌套祖先中，因此这里
    延续物理命中报告的 ancestor-family 口径；特殊结构同时补充 PSA_FFN、
    C2PSA_OUTPUT 与 YOLO11_ATTN_* 精确族。该报告只判断依赖组是否仍可剪，
    不代表 HESSO 最终一定会选择该组。
    """

    after_by_id = {
        str(record.get("node_group_id")): record
        for record in candidate_groups_after
        if record.get("node_group_id") is not None
    }
    family_records: dict[str, dict[str, Any]] = {}
    group_records: list[dict[str, Any]] = []
    for before in candidate_groups_before:
        group_id = str(before.get("node_group_id"))
        prefixes = sorted(
            {
                name.rsplit(".weight", 1)[0]
                if name.endswith(".weight")
                else name.rsplit(".bias", 1)[0]
                if name.endswith(".bias")
                else name
                for name in before.get("candidate_parameter_names", [])
            }
        )
        # candidate_parameter_names 通常指向 ``*.conv.weight``；去掉参数叶子后，
        # 剩余字符串就是可交给 named_modules 祖先链归类的模块前缀。
        families: set[str] = set()
        for prefix in prefixes:
            families.update(module_prefix_ancestor_families(model, prefix))
        after = after_by_id.get(group_id, {})
        is_auxiliary = bool(after.get("is_auxiliary", before.get("is_auxiliary")))
        prunable_before = bool(before.get("is_prunable")) and not bool(before.get("is_auxiliary"))
        prunable_after = bool(after.get("is_prunable")) and not is_auxiliary
        group_record = {
            "node_group_id": before.get("node_group_id"),
            "candidate_parameter_names": before.get("candidate_parameter_names", []),
            "families": sorted(families),
            "is_prunable_before_protection": prunable_before,
            "is_prunable_after_protection": prunable_after,
            "blocked_by_protection": prunable_before and not prunable_after,
        }
        group_records.append(group_record)
        for family in families:
            record = family_records.setdefault(
                family,
                {
                    "candidate_group_ids": [],
                    "prunable_group_ids_after_protection": [],
                    "blocked_group_ids": [],
                },
            )
            record["candidate_group_ids"].append(before.get("node_group_id"))
            if prunable_after:
                record["prunable_group_ids_after_protection"].append(
                    before.get("node_group_id")
                )
            elif prunable_before:
                record["blocked_group_ids"].append(before.get("node_group_id"))

    for record in family_records.values():
        for key in (
            "candidate_group_ids",
            "prunable_group_ids_after_protection",
            "blocked_group_ids",
        ):
            record[key] = sorted({str(value) for value in record[key] if value is not None})
        record["candidate_group_count"] = len(record["candidate_group_ids"])
        record["prunable_group_count_after_protection"] = len(
            record["prunable_group_ids_after_protection"]
        )
        record["blocked_group_count"] = len(record["blocked_group_ids"])

    return {
        "families": dict(sorted(family_records.items())),
        "groups": group_records,
        "policy": (
            "all_verified_safe profile 的每个已验证模块族在所有保护与 quarantine "
            "生效后必须至少保留一个可剪非 auxiliary 依赖组。"
        ),
    }


def profile_fixed_output_candidate_prefixes(
    model,
    module_coverage_profile: str,
) -> list[str]:
    """返回历史 composite profile 中只允许输入联动的临时 guard 卷积。

    C3k2 ``cv2`` 同时消费 root paired-chunk 与 child dynamic-concat 的输入索引，
    因而其输出轴曾不得成为独立 HESSO 候选。候选发现阶段先排除该轴；随后
    ``fixed_output_module_prefixes()`` 按精确模块对象关闭与下游输入联动的 OTO
    输出组。两层约束缺一不可：仅排除前缀无法阻止共享组由下游 Conv 作为候选，
    从而反向裁掉 C3k2 的外部输出；而复合 coordinator 仍独占 ``cv2`` 的输入列裁剪。
    v4 已以 output-consumer closure 接管该输出轴，故明确排除在本函数之外。
    """

    if module_coverage_profile not in C3K2_COMPOSITE_FIXED_OUTPUT_GUARD_PROFILES:
        return []
    return sorted(
        f"{name}.cv2.conv"
        for name, module in model.named_modules()
        if name and type(module).__name__ == "C3k2"
    )


def module_pruning_candidate_prefixes(
    model,
    module_coverage_profile: str,
    probe_module_families: tuple[str, ...] = (),
) -> list[str]:
    """返回实验 profile 本轮新增开放的精确候选模块前缀。"""

    if module_coverage_profile == "conservative_v1":
        return []
    if module_coverage_profile == "bottleneck_hidden_v1_experimental":
        return sorted(
            f"{name}.cv1"
            for name, module in model.named_modules()
            if name and type(module).__name__ == "Bottleneck"
        )
    if module_coverage_profile == C2PSA_PAIRED_SPLIT_PROFILE:
        prefixes = sorted(
            f"{name}.cv1.conv"
            for name, module in model.named_modules()
            if name and type(module).__name__ == "C2PSA"
        )
        if not prefixes:
            raise RuntimeError("C2PSA paired-split profile 未找到 cv1.conv 候选。")
        requested_families = set(probe_module_families)
        if requested_families and requested_families != {"C2PSA_SPLIT"}:
            raise ValueError(
                "C2PSA paired-split 定向 profile 只接受 --probe-module-family C2PSA_SPLIT。"
            )
        return prefixes
    if module_coverage_profile == C3K2_COMPOSITE_PROFILE:
        root_prefixes = {
            f"{name}.cv1.conv"
            for name, module in model.named_modules()
            if name and type(module).__name__ == "C3k2"
        }
        dynamic_prefixes: set[str] = set()
        for parent_name, parent in model.named_modules():
            if not parent_name or type(parent).__name__ != "C3k2":
                continue
            for child_index, child in enumerate(list(getattr(parent, "m", ()))):
                if type(child).__name__ in {"C3", "C3k"} and hasattr(child, "cv3"):
                    dynamic_prefixes.add(f"{parent_name}.m.{child_index}.cv3.conv")
        prefixes = sorted(root_prefixes | dynamic_prefixes)
        if not root_prefixes or not dynamic_prefixes:
            raise RuntimeError(
                "C3k2 composite profile 必须同时找到根部 cv1 与独立 C3/C3k child-output 候选。"
            )
        requested_families = set(probe_module_families)
        expected_families = {"C3K2_CHUNK", "C3K2_DYNAMIC_CONCAT"}
        if requested_families and requested_families != expected_families:
            raise ValueError(
                "C3k2 composite 定向 profile 必须同时指定 "
                "--probe-module-family C3K2_CHUNK 与 C3K2_DYNAMIC_CONCAT。"
            )
        return prefixes
    if module_coverage_profile == C3K2_PAIRED_CHUNK_PROFILE:
        prefixes = sorted(
            f"{name}.cv1.conv"
            for name, module in model.named_modules()
            if name and type(module).__name__ == "C3k2"
        )
        if not prefixes:
            raise RuntimeError("C3k2 paired-chunk profile 未找到 cv1.conv 候选。")
        requested_families = set(probe_module_families)
        if requested_families and requested_families != {"C3K2_CHUNK"}:
            raise ValueError(
                "C3k2 paired-chunk 定向 profile 只接受 --probe-module-family C3K2_CHUNK。"
            )
        return prefixes
    if module_coverage_profile == C3K2_DYNAMIC_CONCAT_PROFILE:
        prefixes: list[str] = []
        for parent_name, parent in model.named_modules():
            if not parent_name or type(parent).__name__ != "C3k2":
                continue
            for child_index, child in enumerate(list(getattr(parent, "m", ()))):
                if type(child).__name__ not in {"C3", "C3k"} or not hasattr(child, "cv3"):
                    continue
                prefixes.append(f"{parent_name}.m.{child_index}.cv3.conv")
        prefixes = sorted(set(prefixes))
        if not prefixes:
            raise RuntimeError(
                "C3k2 dynamic-concat profile 未找到独立 C3/C3k child-output 候选。"
            )
        requested_families = set(probe_module_families)
        if requested_families and requested_families != {"C3K2_DYNAMIC_CONCAT"}:
            raise ValueError(
                "C3k2 dynamic-concat 定向 profile 只接受 "
                "--probe-module-family C3K2_DYNAMIC_CONCAT。"
            )
        return prefixes
    if module_coverage_profile == C3K2_OUTPUT_CLOSURE_PROFILE:
        prefixes = sorted(
            f"{name}.cv2.conv"
            for name, module in model.named_modules()
            if name and type(module).__name__ == "C3k2"
        )
        if not prefixes:
            raise RuntimeError(
                "C3k2 output-consumer closure profile 未找到任何 C3k2.cv2.conv 候选。"
            )
        requested_families = set(probe_module_families)
        if requested_families and requested_families != {"C3K2_OUTPUT_CLOSURE"}:
            raise ValueError(
                "C3k2 output-consumer closure 定向 profile 只接受 "
                "--probe-module-family C3K2_OUTPUT_CLOSURE。"
            )
        return prefixes
    if module_coverage_profile == ALL_CV1_OPEN_PROFILE:
        hard_protected = protected_prefixes(model, module_coverage_profile)
        fixed_outputs = fixed_output_module_prefixes(model, module_coverage_profile)
        prefixes = sorted(
            name
            for name, module in model.named_modules()
            if name
            and type(module).__name__ == "Conv2d"
            and name.endswith(".cv1.conv")
            and not any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in hard_protected
            )
            and not any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in fixed_outputs
            )
        )
        if not prefixes:
            raise RuntimeError("all-cv1 profile 未找到任何 cv1.conv 候选。")
        return prefixes
    if module_coverage_profile in ATTENTION_PROFILE_MODES:
        expected_mode = ATTENTION_PROFILE_MODES[module_coverage_profile]
        named_modules = dict(model.named_modules())
        attention_prefixes: list[str] = []
        physical_consumer_prefixes: list[str] = []
        for name, module in named_modules.items():
            if not name or getattr(module, "_oto_attention_active_prune_mode", None) != expected_mode:
                continue
            for child_path in ("qkv.conv", "pe.conv"):
                prefix = f"{name}.{child_path}"
                child = named_modules.get(prefix)
                if not isinstance(child, torch.nn.Conv2d):
                    raise RuntimeError(
                        f"Attention profile 内部候选不存在或类型错误：{prefix} -> "
                        f"{type(child).__name__ if child is not None else None}"
                    )
                attention_prefixes.append(prefix)
            physical_consumer_prefixes.append(f"{name}.proj.conv")
        if not attention_prefixes:
            raise RuntimeError(
                f"Attention profile {module_coverage_profile} 没有找到 qkv/pe 内部候选。"
            )
        if module_coverage_profile in ATTENTION_ONLY_PROFILES:
            return sorted(set(attention_prefixes))
        if module_coverage_profile not in ALL_VERIFIED_SAFE_PROFILES:
            raise ValueError(f"未实现 Attention 累计候选规则：{module_coverage_profile}")

        # 联合 profile 复用 dense closure 已验证的全部卷积候选，同时把 qkv+pe
        # 保留为规范 Attention 复合组。proj.conv 只消费被裁剪后的内部 V 轴，
        # 其外部输出轴必须固定，不能作为普通 Conv 候选再次参与 HESSO。
        hard_protected = protected_prefixes(model, module_coverage_profile)
        fixed_outputs = fixed_output_module_prefixes(model, module_coverage_profile)
        contract_fixed_outputs = set(
            profile_fixed_output_candidate_prefixes(model, module_coverage_profile)
        )
        physical_consumers = set(physical_consumer_prefixes)
        requested_families = set(probe_module_families)
        candidates = sorted(
            name
            for name, module in named_modules.items()
            if name
            and type(module).__name__ == "Conv2d"
            and name not in physical_consumers
            and name not in contract_fixed_outputs
            and not any(name == prefix or name.startswith(f"{prefix}.") for prefix in hard_protected)
            and not any(name == prefix or name.startswith(f"{prefix}.") for prefix in fixed_outputs)
            and (
                not requested_families
                or bool(module_prefix_ancestor_families(model, name) & requested_families)
            )
        )
        if not requested_families:
            missing_attention_prefixes = sorted(set(attention_prefixes) - set(candidates))
            if missing_attention_prefixes:
                raise RuntimeError(
                    "全部安全模块 profile 未完整开放 Attention qkv+pe 候选："
                    f"{missing_attention_prefixes}"
                )
        return candidates
    if module_coverage_profile in DENSE_CLOSURE_PROFILES:
        hard_protected = protected_prefixes(model, module_coverage_profile)
        fixed_outputs = fixed_output_module_prefixes(model, module_coverage_profile)
        contract_fixed_outputs = set(
            profile_fixed_output_candidate_prefixes(model, module_coverage_profile)
        )
        requested_families = set(probe_module_families)
        return sorted(
            name
            for name, module in model.named_modules()
            if name
            and type(module).__name__ == "Conv2d"
            and name not in contract_fixed_outputs
            and not any(name == prefix or name.startswith(f"{prefix}.") for prefix in hard_protected)
            and not any(name == prefix or name.startswith(f"{prefix}.") for prefix in fixed_outputs)
            and (
                not requested_families
                or bool(module_prefix_ancestor_families(model, name) & requested_families)
            )
        )
    raise ValueError(f"未实现候选模块规则：{module_coverage_profile}")


def fixed_output_module_prefixes(model, module_coverage_profile: str) -> list[str]:
    """返回输出维必须固定、输入维仍可由受支持 owner 联动的模块前缀。"""

    if module_coverage_profile not in DENSE_CLOSURE_PROFILES:
        return []
    prefixes: list[str] = []
    for name, module in model.named_modules():
        if type(module).__name__ != "Detect":
            continue
        for branch_name in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
            branches = getattr(module, branch_name, None)
            if branches is None:
                continue
            for branch_index, branch in enumerate(branches):
                if len(branch) == 0:
                    continue
                prefixes.append(f"{name}.{branch_name}.{branch_index}.{len(branch) - 1}")
        if hasattr(module, "dfl") and hasattr(module.dfl, "conv"):
            prefixes.append(f"{name}.dfl.conv")
    if module_coverage_profile in C3K2_COMPOSITE_FIXED_OUTPUT_GUARD_PROFILES:
        # C3k2.cv2 的输入列由复合 coordinator 在原始 concat 坐标中一次性裁剪；
        # 输出宽度是父模块对外接口，不能再被下游普通 Conv 的共享 OTO 组独立裁剪。
        prefixes.extend(
            f"{name}.cv2"
            for name, module in model.named_modules()
            if name and type(module).__name__ == "C3k2"
        )
    return sorted(set(prefixes))


def protected_param_names(
    model,
    extra_prefixes: list[str] | None = None,
    module_coverage_profile: str = "conservative_v1",
    module_probe_only: bool = False,
) -> tuple[list[str], list[str]]:
    prefixes = protected_prefixes(model, module_coverage_profile)
    prefixes.extend(extra_prefixes or [])
    prefixes = sorted(set(prefix.rstrip(".") for prefix in prefixes if prefix))
    names = [
        name
        for name, _ in model.named_parameters()
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    ]
    return prefixes, names


def mark_profile_unprunable_by_module_prefixes(
    oto,
    model,
    module_prefixes: list[str],
    *,
    include_stateful_descendants: bool,
    classification: str,
    action: str,
) -> dict[str, Any]:
    """YOLO11 profile 本地策略：按模块对象固定其 OTO 节点组。

    这是实验入口的保护策略，不属于 OTO 通用剪枝框架。参数名交集只作为
    兼容回退和审计证据；无状态共享激活不会因对象复用而误伤外部节点组。
    """

    prefixes = sorted({str(prefix).rstrip(".") for prefix in module_prefixes if prefix})
    named_modules = dict(model.named_modules())
    missing_prefixes = sorted(set(prefixes) - set(named_modules))
    existing_prefixes = [prefix for prefix in prefixes if prefix in named_modules]

    prefixes_by_module_id: dict[int, set[str]] = {}
    for prefix in existing_prefixes:
        root_module = named_modules[prefix]
        modules = [root_module]
        if include_stateful_descendants:
            modules.extend(
                module
                for module in root_module.modules()
                if module is not root_module
                and (
                    any(True for _ in module.parameters(recurse=False))
                    or any(True for _ in module.buffers(recurse=False))
                )
            )
        for module in modules:
            prefixes_by_module_id.setdefault(id(module), set()).add(prefix)

    named_parameter_names = [name for name, _ in model.named_parameters()]
    parameter_names_by_prefix = {
        prefix: {
            name
            for name in named_parameter_names
            if name == prefix or name.startswith(f"{prefix}.")
        }
        for prefix in existing_prefixes
    }
    matched_prefixes: set[str] = set()
    records: list[dict[str, Any]] = []
    for node_group in oto._graph.node_groups.values():
        object_prefixes = {
            prefix
            for module in node_group.get_modules()
            for prefix in prefixes_by_module_id.get(id(module), set())
        }
        group_parameter_names = set(node_group.param_names)
        parameter_prefixes = {
            prefix
            for prefix, parameter_names in parameter_names_by_prefix.items()
            if group_parameter_names & parameter_names
        }
        group_prefixes = sorted(object_prefixes | parameter_prefixes)
        if not group_prefixes:
            continue

        matched_parameter_names = sorted(
            group_parameter_names
            & set().union(*(parameter_names_by_prefix[prefix] for prefix in group_prefixes))
        )
        was_prunable = bool(node_group.is_prunable)
        node_group.is_prunable = False
        matched_prefixes.update(group_prefixes)
        records.append(
            {
                "node_group_id": node_group.id,
                "module_prefixes": group_prefixes,
                "matched_parameter_names": matched_parameter_names,
                "matched_by_module_identity": sorted(object_prefixes),
                "matched_by_parameter_fallback": sorted(parameter_prefixes - object_prefixes),
                "num_groups": int(node_group.num_groups),
                "is_auxiliary": bool(node_group.is_auxiliary),
                "is_prunable_before": was_prunable,
                "is_prunable_after": False,
                "classification": classification,
                "action": action,
            }
        )

    return {
        "module_prefixes": prefixes,
        "groups": records,
        "missing_module_prefixes": missing_prefixes,
        "unmatched_module_prefixes": sorted(set(existing_prefixes) - matched_prefixes),
    }


def candidate_quarantine_scope(
    candidate_group_ids,
    constrained_group_ids,
) -> dict[str, Any]:
    """YOLO11 profile 本地审计：只计算当前候选与已知隔离集合的交集。"""

    candidate_ids = {
        str(group_id)
        for group_id in (candidate_group_ids or [])
        if group_id is not None and str(group_id)
    }
    quarantine_ids = {
        str(group_id)
        for group_id in (constrained_group_ids or [])
        if group_id is not None and str(group_id)
    }
    intersection_ids = candidate_ids & quarantine_ids
    return {
        "candidate_group_ids": sorted(candidate_ids),
        "candidate_group_count": len(candidate_ids),
        "constrained_group_ids": sorted(quarantine_ids),
        "constrained_group_count": len(quarantine_ids),
        "intersection_group_ids": sorted(intersection_ids),
        "intersection_group_count": len(intersection_ids),
        "fraction": len(intersection_ids) / max(1, len(candidate_ids)),
        "scope": "current_candidate_group_intersection",
    }


def quarantine_c3k2_chunk_source_groups(oto, model, module_coverage_profile: str) -> dict[str, Any]:
    """固定 YOLO11 ``C3k2.cv1 -> chunk(2)`` 的输出节点组。

    OTOv2 的旧依赖图能够对这些参数执行 tensor transformation，但没有表达
    ``chunk(2)`` 两半之间的成对索引关系。若直接删除 ``cv1`` 输出通道，真实
    PyTorch chunk 尺寸与 OTO 推导的内部 C3/C3k 输入尺寸可能不同。因此在实现
    paired-chunk transform 前，只固定 chunk 源输出；C3k2 内部其他普通卷积组
    仍可独立参与 HESSO。
    """

    if module_coverage_profile not in DENSE_CLOSURE_PROFILES:
        return {
            "classification": CSP_CHUNK_SOURCE_QUARANTINE,
            "enabled": False,
            "boundary_prefixes": [],
            "boundary_parameter_names": [],
            "groups": [],
            "unmatched_boundary_prefixes": [],
        }

    if module_coverage_profile in C3K2_PAIRED_CHUNK_PROFILES:
        records = []
        boundary_prefixes = []
        boundary_parameter_names = []
        for node_group in oto._graph.node_groups.values():
            contract = getattr(node_group, "_oto_c3k2_chunk_contract", None)
            if not isinstance(contract, dict):
                continue
            prefix = f"{contract['module_prefix']}.cv1"
            boundary_prefixes.append(prefix)
            parameter_names = sorted(
                name
                for name in node_group.param_names
                if name == prefix or name.startswith(f"{prefix}.")
            )
            boundary_parameter_names.extend(parameter_names)
            records.append(
                {
                    "node_group_id": str(node_group.id),
                    "module_prefixes": [prefix],
                    "boundary_prefixes": [prefix],
                    "matched_parameter_names": parameter_names,
                    "num_groups": int(node_group.num_groups),
                    "is_auxiliary": bool(node_group.is_auxiliary),
                    "is_prunable_before": True,
                    "is_prunable_after": bool(node_group.is_prunable),
                    "classification": "yolo11_c3k2_paired_chunk_contract_managed",
                    "action": "managed_by_paired_chunk_contract",
                }
            )
        return {
            "classification": CSP_CHUNK_SOURCE_QUARANTINE,
            "enabled": False,
            "contract_managed": True,
            "boundary_prefixes": sorted(boundary_prefixes),
            "boundary_parameter_names": sorted(set(boundary_parameter_names)),
            "groups": records,
            "unmatched_boundary_prefixes": [],
            "reason": "C3k2.cv1 chunk 源已由 paired-chunk 合同接管，不再 quarantine。",
        }

    boundary_prefixes = sorted(
        f"{name}.cv1"
        for name, module in model.named_modules()
        if name and type(module).__name__ == "C3k2"
    )
    boundary_parameter_names = sorted(
        name
        for name, _ in model.named_parameters()
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in boundary_prefixes)
    )
    protection = mark_profile_unprunable_by_module_prefixes(
        oto,
        model,
        boundary_prefixes,
        include_stateful_descendants=True,
        classification=CSP_CHUNK_SOURCE_QUARANTINE,
        action="quarantine_node_group_output_until_paired_chunk_transform_exists",
    )
    if protection["missing_module_prefixes"]:
        raise RuntimeError(
            "C3k2 chunk 源模块未出现在模型中："
            f"{protection['missing_module_prefixes']}"
        )
    records = protection["groups"]
    for record in records:
        record["boundary_prefixes"] = list(record["module_prefixes"])

    return {
        "classification": CSP_CHUNK_SOURCE_QUARANTINE,
        "enabled": True,
        "boundary_prefixes": boundary_prefixes,
        "boundary_parameter_names": boundary_parameter_names,
        "groups": records,
        "unmatched_boundary_prefixes": protection["unmatched_module_prefixes"],
        "reason": (
            "OTOv2 尚未表达 C3k2.cv1 输出经 chunk(2) 后的成对通道索引；"
            "固定 chunk 源输出，避免内部 C3/C3k 输入与外层 concat 输入发生物理 shape 偏差。"
        ),
    }


def quarantine_c3k2_dynamic_concat_groups(oto, model, module_coverage_profile: str) -> dict[str, Any]:
    """固定 C3k2 ``y.extend(...)`` 写入动态 concat 的直接子模块输出。

    YOLO11 的 C3k2/C2f 并非静态二输入 Concat，而是先把两个 chunk 放入列表，
    再依次追加 ``self.m`` 的输出。旧 OTO 没有把直接子模块输出的删减索引和
    最终 ``cv2`` 输入偏移完整绑定。这里只固定列表元素的接口输出，子模块内部
    隐藏卷积以及 C3k2.cv2 自身输出仍保持开放。
    """

    if module_coverage_profile not in DENSE_CLOSURE_PROFILES:
        return {
            "classification": CSP_DYNAMIC_CONCAT_QUARANTINE,
            "enabled": False,
            "boundary_prefixes": [],
            "boundary_parameter_names": [],
            "groups": [],
            "unsupported_child_modules": [],
            "unmatched_boundary_prefixes": [],
        }

    boundary_prefixes: list[str] = []
    unsupported_child_modules: list[dict[str, str]] = []
    for name, module in model.named_modules():
        if not name or type(module).__name__ != "C3k2":
            continue
        children = getattr(module, "m", None)
        if children is None:
            unsupported_child_modules.append({"module": name, "type": "missing_m"})
            continue
        for child_index, child in enumerate(children):
            child_prefix = f"{name}.m.{child_index}"
            child_type = type(child).__name__
            if child_type in {"C3", "C3k"} and hasattr(child, "cv3"):
                boundary_prefixes.append(f"{child_prefix}.cv3")
            elif child_type == "Bottleneck" and hasattr(child, "cv2"):
                boundary_prefixes.append(f"{child_prefix}.cv2")
            else:
                unsupported_child_modules.append({"module": child_prefix, "type": child_type})

    boundary_prefixes = sorted(set(boundary_prefixes))

    if module_coverage_profile in C3K2_DYNAMIC_CONCAT_PROFILES:
        managed_records: list[dict[str, Any]] = []
        managed_prefixes: set[str] = set()
        for node_group in oto._graph.node_groups.values():
            contract = getattr(node_group, "_oto_c3k2_dynamic_contract", None)
            if not isinstance(contract, dict):
                continue
            output_prefix = str(contract["output_wrapper_prefix"])
            managed_prefixes.add(output_prefix)
            matched_parameter_names = sorted(
                name
                for name in node_group.param_names
                if name == output_prefix or name.startswith(f"{output_prefix}.")
            )
            node_group.is_prunable = True
            managed_records.append(
                {
                    "node_group_id": str(node_group.id),
                    "module_prefixes": [output_prefix],
                    "boundary_prefixes": [output_prefix],
                    "matched_parameter_names": matched_parameter_names,
                    "num_groups": int(node_group.num_groups),
                    "is_auxiliary": bool(node_group.is_auxiliary),
                    "is_prunable_before": True,
                    "is_prunable_after": True,
                    "classification": (
                        "yolo11_c3k2_dynamic_concat_contract_managed"
                    ),
                    "action": "managed_by_segmented_dynamic_concat_contract",
                }
            )
        if module_coverage_profile in C3K2_COMPOSITE_PROFILES:
            # 复合 profile 中，Bottleneck child-output 不是“不支持的动态轴”，而是
            # 根部 paired-chunk 轴的一部分。若仍用 module-prefix protection 固定它，
            # ``mark_*`` 会按整个 OTO 依赖组闭包，反向关闭同时已开放的根部 cv1。
            # 因此显式把它记为由父级 composite coordinator 接管，而不是隔离。
            for node_group in oto._graph.node_groups.values():
                root_contract = getattr(node_group, "_oto_c3k2_chunk_contract", None)
                if not isinstance(root_contract, dict) or not bool(
                    root_contract.get("composite_owner_mode")
                ):
                    continue
                root_prefixes = sorted(
                    {
                        f"{child['prefix']}.{child['output_path']}"
                        for child in root_contract.get("children", [])
                        if child.get("type") == "Bottleneck"
                    }
                )
                if not root_prefixes:
                    continue
                managed_prefixes.update(root_prefixes)
                matched_parameter_names = sorted(
                    name
                    for name in node_group.param_names
                    if any(
                        name == prefix or name.startswith(f"{prefix}.")
                        for prefix in root_prefixes
                    )
                )
                managed_records.append(
                    {
                        "node_group_id": str(node_group.id),
                        "module_prefixes": root_prefixes,
                        "boundary_prefixes": root_prefixes,
                        "matched_parameter_names": matched_parameter_names,
                        "num_groups": int(node_group.num_groups),
                        "is_auxiliary": bool(node_group.is_auxiliary),
                        "is_prunable_before": bool(node_group.is_prunable),
                        "is_prunable_after": bool(node_group.is_prunable),
                        "classification": "yolo11_c3k2_composite_contract_managed",
                        "action": "managed_by_c3k2_composite_owner",
                    }
                )
        if not managed_records:
            raise RuntimeError(
                "C3k2 dynamic-concat profile 未找到已配置的 managed child-output 组。"
            )
        quarantined_prefixes = sorted(set(boundary_prefixes) - managed_prefixes)
        quarantined_parameter_names = sorted(
            name
            for name, _ in model.named_parameters()
            if any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in quarantined_prefixes
            )
        )
        if quarantined_prefixes:
            protection = mark_profile_unprunable_by_module_prefixes(
                oto,
                model,
                quarantined_prefixes,
                include_stateful_descendants=True,
                classification=CSP_DYNAMIC_CONCAT_QUARANTINE,
                action=(
                    "quarantine_nonindependent_child_output_until_composite_axis_exists"
                ),
            )
            if protection["missing_module_prefixes"]:
                raise RuntimeError(
                    "C3k2 dynamic-concat 剩余边界模块未出现在模型中："
                    f"{protection['missing_module_prefixes']}"
                )
            protected_records = protection["groups"]
            for record in protected_records:
                record["boundary_prefixes"] = list(record["module_prefixes"])
            unmatched = protection["unmatched_module_prefixes"]
        else:
            protected_records = []
            unmatched = []
        return {
            "classification": CSP_DYNAMIC_CONCAT_QUARANTINE,
            "enabled": bool(quarantined_prefixes),
            "contract_managed": True,
            "all_boundary_prefixes": boundary_prefixes,
            "managed_boundary_prefixes": sorted(managed_prefixes),
            "boundary_prefixes": quarantined_prefixes,
            "boundary_parameter_names": quarantined_parameter_names,
            "groups": sorted(
                managed_records + protected_records,
                key=lambda record: str(record.get("node_group_id")),
            ),
            "unsupported_child_modules": unsupported_child_modules,
            "unmatched_boundary_prefixes": unmatched,
            "reason": (
                "独立 C3/C3k child-output 由 SegmentedDynamicConcatContract 接管；"
                "Bottleneck child-output 在 composite profile 中由父级协调器接管，"
                "其他 profile 继续保持 fail-closed。"
            ),
        }

    boundary_parameter_names = sorted(
        name
        for name, _ in model.named_parameters()
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in boundary_prefixes)
    )
    protection = mark_profile_unprunable_by_module_prefixes(
        oto,
        model,
        boundary_prefixes,
        include_stateful_descendants=True,
        classification=CSP_DYNAMIC_CONCAT_QUARANTINE,
        action="quarantine_child_output_until_dynamic_concat_offset_mapping_exists",
    )
    if protection["missing_module_prefixes"]:
        raise RuntimeError(
            "C3k2 动态 concat 边界模块未出现在模型中："
            f"{protection['missing_module_prefixes']}"
        )
    records = protection["groups"]
    for record in records:
        record["boundary_prefixes"] = list(record["module_prefixes"])
        if module_coverage_profile in C3K2_PAIRED_CHUNK_PROFILES:
            node_group = oto._graph.node_groups.get(record.get("node_group_id"))
            if node_group is not None and hasattr(
                node_group, "_oto_c3k2_chunk_contract"
            ):
                # Bottleneck 型 C3k2 的 child output 与 cv1 chunk 源位于同一
                # OTO 依赖组。该组已经由 paired-chunk 合同接管，不能再被动态
                # concat quarantine 反向关闭；C3k 型独立 child output 组仍保持隔离。
                node_group.is_prunable = True
                record["is_prunable_after"] = True
                record["classification"] = (
                    "yolo11_c3k2_paired_chunk_contract_managed"
                )
                record["action"] = "managed_by_paired_chunk_contract"

    return {
        "classification": CSP_DYNAMIC_CONCAT_QUARANTINE,
        "enabled": True,
        "boundary_prefixes": boundary_prefixes,
        "boundary_parameter_names": boundary_parameter_names,
        "groups": records,
        "unsupported_child_modules": unsupported_child_modules,
        "unmatched_boundary_prefixes": protection["unmatched_module_prefixes"],
        "reason": (
            "独立 child output 剪枝尚未建立动态 concat segment offset 合同；"
            "paired-chunk profile 只同步缩小全部外部 segment，独立 child output 组继续隔离。"
            if module_coverage_profile in C3K2_PAIRED_CHUNK_PROFILES
            else "OTOv2 尚未把 C3k2/C2f 动态列表 concat 中每个 self.m 子模块输出的"
            "删减索引映射到最终 cv2 输入偏移。"
        ),
    }


def classify_tensor_unsafe_exclusion(
    parameter_name: str,
    parameter_shape: list[int],
    num_groups: int,
    error: Exception | None = None,
) -> str:
    """只识别已由 YOLO11 C3k2 ``chunk(2)`` 证明的 2:1 边界错组。"""

    is_bottleneck_cv2_bn = (
        ".m." in parameter_name
        and ".cv2.bn." in parameter_name
        and parameter_name.endswith((".weight", ".bias"))
    )
    if (
        is_bottleneck_cv2_bn
        and len(parameter_shape) == 1
        and parameter_shape[0] > 0
        and num_groups == 2 * parameter_shape[0]
    ):
        return KNOWN_CSP_CHUNK_QUARANTINE
    return "unclassified_tensor_transformation_failure"


def exclude_oto_tensor_unsafe_groups(oto) -> list[dict[str, Any]]:
    """排除 OTO 无法正确变换的 YOLO11 节点组。

    OTO 的旧版 C3 规则会把部分 C3k2 的 ACCESSORY 参数按错误的 group 数 reshape。
    这些组若交给 HESSO 会在重要性计算阶段失败，或更糟地造成错误的物理裁剪。因此
    在创建 HESSO 前逐组探测其 tensor transformation，并把失败组标为不可剪枝。
    """

    from only_train_once.transform import TensorTransform, tensor_transformation

    def validate_transform(parameter, transform, param_group) -> None:
        if transform == TensorTransform.NO_PRUNE:
            return
        if transform == TensorTransform.MULTIHEAD_HEADDIM:
            tensor_transformation(
                parameter,
                transform,
                param_group["num_groups"],
                param_group["num_heads"],
            )
            return
        if isinstance(transform, list):
            transformed = parameter.data
            for transform_type, config in transform:
                if transform_type in (
                    TensorTransform.MULTIHEAD_NUMHEAD,
                    TensorTransform.MULTIHEAD_NUMHEAD_SPREAD,
                ):
                    transformed = tensor_transformation(
                        transformed,
                        transform_type,
                        config["num_heads"],
                    )
                elif transform_type == TensorTransform.MULTIHEAD_HEADDIM:
                    transformed = tensor_transformation(
                        transformed,
                        transform_type,
                        config["head_dim"],
                        config["num_heads"],
                    )
                else:
                    transformed = tensor_transformation(
                        transformed,
                        transform_type,
                        num_groups=config["num_groups"],
                    )
            return
        tensor_transformation(parameter, transform, param_group["num_groups"])

    excluded: list[dict[str, Any]] = []
    validation_errors = (
        RuntimeError,
        ValueError,
        IndexError,
        KeyError,
        TypeError,
        NotImplementedError,
    )
    for node_group in oto._graph.node_groups.values():
        if not node_group.is_prunable:
            continue
        param_group = node_group.get_param_groups()
        for name, parameter, transform in zip(
            param_group["p_names"],
            param_group["params"],
            param_group["p_transform"],
        ):
            try:
                validate_transform(parameter, transform, param_group)
            except validation_errors as exc:
                node_group.is_prunable = False
                parameter_shape = list(parameter.shape)
                excluded.append(
                    {
                        "node_group_id": param_group["id"],
                        "parameter": name,
                        "parameter_shape": parameter_shape,
                        "transform": str(transform),
                        "num_groups": int(param_group["num_groups"]),
                        "error_type": type(exc).__name__,
                        "reason": str(exc),
                        "classification": classify_tensor_unsafe_exclusion(
                            name,
                            parameter_shape,
                            int(param_group["num_groups"]),
                            exc,
                        ),
                        "action": "quarantine_node_group",
                    }
                )
                break
    return excluded


class _THOPInferenceWrapper:
    """把 YOLO11 的可能复合输出规整为 THOP 可接受的 tensor 输出。"""

    def __init__(self, model):
        self.module = model

    def as_module(self):
        model = self.module

        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model

            def forward(self, images):
                output = self.model(images)
                if isinstance(output, (tuple, list)):
                    return output[0]
                return output

        return Wrapper()


def model_stats(model, imgsz: int, device) -> dict[str, float]:
    """使用 THOP 计算物理模型的参数量与 FLOPs，避免 OTO 对 PSA shape 的估计误差。"""

    model_copy = copy.deepcopy(model).to(device).float().eval()
    wrapped = _THOPInferenceWrapper(model_copy).as_module().eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        macs, params = profile(wrapped, inputs=(dummy,), verbose=False)
    return {
        "parameters": int(params),
        "parameters_m": float(params) / 1e6,
        "flops": float(macs) * 2.0,
        "flops_g": float(macs) * 2.0 / 1e9,
    }


def model_structure_signature(model) -> dict[str, Any]:
    """生成不依赖参数数值的结构指纹，用于阻止微调时静默重建标准 YOLO11。"""

    model = unwrap_model(model)
    tensors: list[dict[str, Any]] = []
    for kind, named_tensors in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, tensor in named_tensors:
            tensors.append(
                {
                    "kind": kind,
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }
            )
    modules = [{"name": name, "type": type(module).__qualname__} for name, module in model.named_modules()]
    payload = {"tensors": tensors, "modules": modules}
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "sha256": digest,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_tensor_count": sum(1 for _ in model.parameters()),
        "buffer_tensor_count": sum(1 for _ in model.buffers()),
        "module_count": sum(1 for _ in model.modules()),
    }


def model_state_signature(model) -> dict[str, Any]:
    """对完整 state_dict 的名称、shape、dtype 和数值生成确定性哈希。"""

    model = unwrap_model(model)
    digest = hashlib.sha256()
    state = model.state_dict()
    value_count = 0
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        metadata = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\0")
        if tensor.numel():
            # 以 uint8 读取底层字节，兼容浮点、整数、布尔和 bfloat16 buffer。
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
        value_count += tensor.numel()
    return {
        "sha256": digest.hexdigest(),
        "tensor_count": len(state),
        "value_count": value_count,
    }


def validate_yolo_checkpoint(
    weights: Path,
    args: argparse.Namespace,
    *,
    project: Path,
    name: str,
) -> dict[str, Any]:
    """对指定 checkpoint 做一次独立 COCO val，并返回可序列化的检测指标。"""

    weights = weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"验证权重不存在：{weights}")
    validation_dir = project.expanduser().resolve() / name
    if args.no_overwrite and validation_dir.exists():
        raise FileExistsError(f"禁止覆盖已有验证目录：{validation_dir}；请更换 --name / RUN_ID。")
    print_stage(f"独立验证：{name}")
    metrics = YOLO(str(weights)).val(
        data=str(Path(args.data).expanduser().resolve()),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project.expanduser().resolve()),
        name=name,
        exist_ok=not args.no_overwrite,
        plots=False,
        save_json=False,
        verbose=True,
    )
    summary = validation_summary(metrics)
    summary["weights"] = str(weights)
    summary["weights_sha256"] = sha256_file(weights)
    return summary


def yolo11_attention_optimizer_coverage(oto, model) -> dict[str, Any]:
    """确认每个可训练 Attention 参数在最终 optimizer groups 中恰好出现一次。"""

    optimizer_groups = list(oto._graph.get_param_groups())
    ownership_counts: dict[int, int] = {}
    for group in optimizer_groups:
        for parameter in group.get("params", []):
            ownership_counts[id(parameter)] = ownership_counts.get(id(parameter), 0) + 1

    model_parameter_names = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    records: list[dict[str, Any]] = []
    for module_name, module in model.named_modules():
        if not module_name or not is_yolo11_attention_structure(module):
            continue
        prefix = f"{module_name}."
        expected = sorted(
            name for name in model_parameter_names.values() if name.startswith(prefix)
        )
        missing = sorted(
            model_parameter_names[parameter_id]
            for parameter_id in model_parameter_names
            if model_parameter_names[parameter_id].startswith(prefix)
            and ownership_counts.get(parameter_id, 0) == 0
        )
        duplicated = sorted(
            model_parameter_names[parameter_id]
            for parameter_id in model_parameter_names
            if model_parameter_names[parameter_id].startswith(prefix)
            and ownership_counts.get(parameter_id, 0) > 1
        )
        records.append(
            {
                "module_prefix": module_name,
                "module_type": type(module).__name__,
                "prune_mode": yolo11_attention_prune_mode(module),
                "expected_trainable_parameter_names": expected,
                "missing_parameter_names": missing,
                "duplicated_parameter_names": duplicated,
                "status": "pass" if expected and not missing and not duplicated else "fail",
            }
        )
    return {
        "status": (
            "not_applicable"
            if not records
            else "pass"
            if all(record["status"] == "pass" for record in records)
            else "fail"
        ),
        "identity": "python_parameter_object_id_within_single_graph_build",
        "expected_ownership_count": 1,
        "records": records,
    }


def build_oto(
    model,
    imgsz: int,
    device,
    extra_protect: list[str],
    onnx_opset_version: int,
    module_coverage_profile: str = "conservative_v1",
    module_probe_only: bool = False,
    probe_module_families: tuple[str, ...] = (),
):
    """建立 OTO 图并施加 YOLO11 保护层策略。"""
    attention_pruning_contracts = configure_yolo11_attention_pruning(
        model, module_coverage_profile
    )
    attention_physical_consumer_prefixes = sorted(
        {
            prefix
            for contract in attention_pruning_contracts
            for prefix in contract.get("physical_consumer_prefixes", [])
        }
    )
    model.eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
    oto = OTO(
        model=model,
        dummy_input=dummy,
        strict_out_nodes=True,
        onnx_opset_version=onnx_opset_version,
    )
    c2psa_paired_split_contracts = apply_c2psa_paired_split_contract(
        oto,
        model,
        module_coverage_profile,
    )
    c3k2_paired_chunk_contracts = apply_c3k2_paired_chunk_contract(
        oto,
        model,
        module_coverage_profile,
    )
    c3k2_dynamic_concat_contracts = apply_c3k2_dynamic_concat_contract(
        oto,
        model,
        module_coverage_profile,
    )
    c3k2_composite_contracts = apply_c3k2_composite_ownership_contract(
        oto,
        model,
        module_coverage_profile,
        c3k2_paired_chunk_contracts,
        c3k2_dynamic_concat_contracts,
    )
    output_consumer_closure_contracts = apply_output_consumer_closure_contract(
        oto,
        model,
        module_coverage_profile,
    )
    output_consumer_segmented_concat_contracts = sorted(
        {
            str(contract.get("contract_id")): dict(contract)
            for closure_contract in output_consumer_closure_contracts
            if isinstance(closure_contract, dict)
            for contract in closure_contract.get("segmented_concat_contracts", [])
            if isinstance(contract, dict) and contract.get("contract_id")
        }.values(),
        key=lambda contract: str(contract.get("contract_id")),
    )
    prefixes, names = protected_param_names(
        model,
        extra_protect,
        module_coverage_profile,
        module_probe_only,
    )
    candidate_prefixes = module_pruning_candidate_prefixes(
        model,
        module_coverage_profile,
        probe_module_families,
    )
    if module_coverage_profile == C3K2_OUTPUT_CLOSURE_PROFILE:
        # 仅把消费者闭包已完整解析的生产者加入 HESSO。不可解析的输出节点组已在
        # apply_output_consumer_closure_contract() 中被精确关闭并记录原因；这里不
        # 使用人工模块保护，也不会把未知边错误开放。
        candidate_prefixes = sorted(
            {
                str(contract["producer_conv_prefix"])
                for contract in output_consumer_closure_contracts
                if contract.get("status") == "configured_resolved_consumer_closure"
                and contract.get("producer_conv_prefix")
            }
        )
    contract_fixed_output_candidate_prefixes = profile_fixed_output_candidate_prefixes(
        model,
        module_coverage_profile,
    )
    candidate_groups_before_protection = module_candidate_group_report(oto, candidate_prefixes)

    # 硬保护仍按完整 OTO 依赖组生效，避免只保护模块自身参数却破坏前后 shape。
    oto.mark_unprunable_by_param_names(names)

    # Detect 语义输出及 C3k2 composite 的对外 cv2 输出维必须固定，但其输入维
    # 仍应由上游/复合 owner 同步。不能把这些参数名前缀交给上面的全局保护，
    # 而是在 OTO 图中按精确模块对象固定其输出节点组。
    fixed_output_prefixes = fixed_output_module_prefixes(model, module_coverage_profile)
    fixed_output_groups = mark_fixed_output_node_groups(oto, model, fixed_output_prefixes)

    # C3k2.cv1 的输出紧接 chunk(2)。旧 OTO 的逐组 tensor transformation 虽然
    # 可以执行，却没有表达两个 chunk 的配对索引；这类“语义不安全”不能只靠
    # 下方的 reshape 异常探测发现，必须在 HESSO 前显式固定 chunk 源输出组。
    c3k2_chunk_source_quarantine = quarantine_c3k2_chunk_source_groups(
        oto,
        model,
        module_coverage_profile,
    )
    c3k2_dynamic_concat_quarantine = quarantine_c3k2_dynamic_concat_groups(
        oto,
        model,
        module_coverage_profile,
    )
    prefixes = sorted(
        set(
            prefixes
            + c3k2_chunk_source_quarantine["boundary_prefixes"]
            + c3k2_dynamic_concat_quarantine["boundary_prefixes"]
        )
    )
    names = sorted(
        set(
            names
            + c3k2_chunk_source_quarantine["boundary_parameter_names"]
            + c3k2_dynamic_concat_quarantine["boundary_parameter_names"]
        )
    )

    # 单独记录 profile guard 对候选依赖组的真实影响。模块前缀“看起来开放”
    # 不代表保护后仍有可剪组；该报告用于发现 Attention/残差/边界保护把候选
    # 连带封闭的情况，并在 audit 阶段 fail closed。
    candidate_groups_after_profile_guards = module_candidate_group_report(
        oto,
        candidate_prefixes,
    )
    guard_group_sources: dict[str, set[str]] = {}
    for source, records in (
        ("fixed_output_semantics", fixed_output_groups),
        (CSP_CHUNK_SOURCE_QUARANTINE, c3k2_chunk_source_quarantine["groups"]),
        (CSP_DYNAMIC_CONCAT_QUARANTINE, c3k2_dynamic_concat_quarantine["groups"]),
        (
            "output_consumer_closure_auto_quarantine",
            [
                record
                for record in output_consumer_closure_contracts
                if record.get("status") != "configured_resolved_consumer_closure"
            ],
        ),
    ):
        for record in records:
            group_id = record.get("node_group_id")
            if group_id is not None:
                guard_group_sources.setdefault(str(group_id), set()).add(source)
    candidate_guard_overlap = candidate_profile_guard_overlap_report(
        candidate_groups_before_protection,
        candidate_groups_after_profile_guards,
        names,
        guard_group_sources,
    )

    probe_isolation = isolate_candidate_dependency_groups(
        oto,
        candidate_groups_before_protection,
        enabled=module_probe_only,
    )
    tensor_unsafe_exclusions = exclude_oto_tensor_unsafe_groups(oto)
    candidate_groups_after_protection = module_candidate_group_report(oto, candidate_prefixes)
    output_consumer_composite_ownership = output_consumer_composite_ownership_report(
        output_consumer_closure_contracts,
        c3k2_composite_contracts,
        candidate_groups_after_protection,
    )
    segmented_concat_candidate_owner_coverage = (
        segmented_concat_candidate_owner_coverage_report(
            output_consumer_segmented_concat_contracts,
            candidate_groups_after_protection,
        )
    )
    candidate_family_coverage = candidate_family_coverage_report(
        model,
        candidate_groups_before_protection,
        candidate_groups_after_protection,
    )
    attention_optimizer_coverage = yolo11_attention_optimizer_coverage(oto, model)
    # 不在建图函数中把完整诊断 JSON 拼进异常。覆盖结果写入 dependency_control，
    # 由统一候选审计生成简短控制台结论并把详细记录保存到 audit/runtime JSON。
    attention_physical_consumer_groups = module_candidate_group_report(
        oto,
        attention_physical_consumer_prefixes,
    ) if attention_physical_consumer_prefixes else []
    known_quarantined_groups = [
        record
        for record in tensor_unsafe_exclusions
        if record["classification"] == KNOWN_CSP_CHUNK_QUARANTINE
    ]
    managed_boundary_actions = {
        "managed_by_paired_chunk_contract",
        "managed_by_segmented_dynamic_concat_contract",
        "managed_by_c3k2_composite_owner",
    }
    known_quarantined_group_ids = {
        record["node_group_id"] for record in known_quarantined_groups
    } | {
        record["node_group_id"]
        for record in c3k2_chunk_source_quarantine["groups"]
        if record.get("action") not in managed_boundary_actions
    } | {
        record["node_group_id"]
        for record in c3k2_dynamic_concat_quarantine["groups"]
        if record.get("action") not in managed_boundary_actions
    }
    unclassified_unsafe_groups = [
        record
        for record in tensor_unsafe_exclusions
        if record["classification"] != KNOWN_CSP_CHUNK_QUARANTINE
    ]
    quarantine_scope = candidate_quarantine_scope(
        candidate_group_ids=[
            record["node_group_id"]
            for record in candidate_groups_after_protection
            if record.get("node_group_id") is not None
        ],
        constrained_group_ids=known_quarantined_group_ids,
    )
    dependency_control = {
        "candidate_group_ids_before_protection": [
            record["node_group_id"] for record in candidate_groups_before_protection
        ],
        "candidate_groups_before_protection": candidate_groups_before_protection,
        "candidate_groups_after_profile_guards": candidate_groups_after_profile_guards,
        "candidate_profile_guard_overlap": candidate_guard_overlap,
        "candidate_groups_after_protection": candidate_groups_after_protection,
        "candidate_family_coverage": candidate_family_coverage,
        "boundary_blocked_candidate_group_ids": sorted(
            record["node_group_id"]
            for record in candidate_groups_after_protection
            if not record["is_prunable"] and not record["is_auxiliary"]
        ),
        "fixed_output_module_prefixes": fixed_output_prefixes,
        "fixed_output_node_groups": fixed_output_groups,
        "c3k2_paired_chunk_contracts": c3k2_paired_chunk_contracts,
        "c3k2_dynamic_concat_contracts": c3k2_dynamic_concat_contracts,
        "c3k2_composite_contracts": c3k2_composite_contracts,
        "yolo11_output_consumer_closure_contracts": output_consumer_closure_contracts,
        "output_consumer_segmented_concat_contracts": (
            output_consumer_segmented_concat_contracts
        ),
        "output_consumer_closure_implementation": (
            "graph_topology_direct_depthwise_and_segmented_concat_v4_passive_owner"
        ),
        "output_consumer_composite_ownership": output_consumer_composite_ownership,
        "segmented_concat_candidate_owner_coverage": (
            segmented_concat_candidate_owner_coverage
        ),
        "c3k2_chunk_source_quarantine": c3k2_chunk_source_quarantine,
        "c3k2_dynamic_concat_quarantine": c3k2_dynamic_concat_quarantine,
        "known_quarantined_csp_chunk_groups": known_quarantined_groups,
        "known_quarantined_csp_chunk_group_ids": sorted(known_quarantined_group_ids),
        "known_quarantined_candidate_group_ids": quarantine_scope[
            "intersection_group_ids"
        ],
        "known_quarantined_candidate_group_count": quarantine_scope[
            "intersection_group_count"
        ],
        "unclassified_tensor_unsafe_groups": unclassified_unsafe_groups,
        "known_quarantine_candidate_group_count": quarantine_scope["candidate_group_count"],
        "known_quarantine_fraction_of_candidate_groups": quarantine_scope["fraction"],
        "known_quarantine_fraction_scope": quarantine_scope["scope"],
        "semantic_boundary_guard_implementation": "yolo11_profile_local_module_identity_v1",
        "tensor_transform_audit_implementation": "yolo11_profile_local_fail_closed_v1",
        "candidate_isolation_implementation": "yolo11_profile_local_node_group_v1",
        "yolo11_attention_pruning_contracts": attention_pruning_contracts,
        "yolo11_attention_physical_consumer_prefixes": attention_physical_consumer_prefixes,
        "yolo11_attention_physical_consumer_groups": attention_physical_consumer_groups,
        "yolo11_attention_optimizer_coverage": attention_optimizer_coverage,
        "yolo11_c2psa_paired_split_contracts": c2psa_paired_split_contracts,
        "yolo11_c3k2_paired_chunk_contracts": c3k2_paired_chunk_contracts,
        "yolo11_c3k2_composite_contracts": c3k2_composite_contracts,
        "yolo11_c3k2_dynamic_concat_contracts": c3k2_dynamic_concat_contracts,
        **probe_isolation,
    }
    contract_descriptor = {
        "schema_version": "yolo11_oto_dependency_contract_v1",
        "module_coverage_profile": module_coverage_profile,
        "module_probe_only": bool(module_probe_only),
        "probe_module_families": sorted(probe_module_families),
        "oto_graph_opset": int(onnx_opset_version),
        "framework_versions": framework_version_summary(),
        "candidate_module_prefixes": candidate_prefixes,
        "contract_fixed_output_candidate_prefixes": (
            contract_fixed_output_candidate_prefixes
        ),
        "protected_module_prefixes": prefixes,
        "fixed_output_module_prefixes": fixed_output_prefixes,
        "chunk_source_boundary_prefixes": c3k2_chunk_source_quarantine["boundary_prefixes"],
        "dynamic_concat_boundary_prefixes": c3k2_dynamic_concat_quarantine["boundary_prefixes"],
        "known_tensor_unsafe_classification": KNOWN_CSP_CHUNK_QUARANTINE,
        "known_quarantine_fraction_scope": quarantine_scope["scope"],
        "semantic_boundary_guard_implementation": "yolo11_profile_local_module_identity_v1",
        "tensor_transform_audit_implementation": "yolo11_profile_local_fail_closed_v1",
        "candidate_isolation_implementation": "yolo11_profile_local_node_group_v1",
        "unclassified_tensor_unsafe_policy": "fail_closed",
        "candidate_isolation_scope": "oto_node_group",
        "preserve_auxiliary_groups": True,
        "yolo11_attention_pruning_contracts": attention_pruning_contracts,
        "yolo11_attention_physical_consumer_prefixes": attention_physical_consumer_prefixes,
        "attention_candidate_role": "qkv_pe_internal_driver",
        "attention_physical_consumer_role": "proj_input_axis_external_output_fixed",
        "yolo11_c2psa_paired_split_contracts": c2psa_paired_split_contracts,
        "c2psa_split_contract_implementation": "yolo11_profile_local_index_groups_v1",
        "yolo11_c3k2_paired_chunk_contracts": c3k2_paired_chunk_contracts,
        "c3k2_chunk_contract_implementation": "yolo11_profile_local_index_groups_v1",
        "yolo11_c3k2_dynamic_concat_contracts": c3k2_dynamic_concat_contracts,
        "c3k2_dynamic_concat_contract_implementation": (
            "yolo11_profile_local_runtime_segment_offset_v1"
        ),
        "yolo11_output_consumer_closure_contracts": output_consumer_closure_contracts,
        "output_consumer_segmented_concat_contracts": (
            output_consumer_segmented_concat_contracts
        ),
        "output_consumer_closure_implementation": (
            "graph_topology_direct_depthwise_and_segmented_concat_v4_passive_owner"
        ),
    }
    dependency_control["contract"] = {
        "sha256": stable_json_sha256(contract_descriptor),
        "descriptor": contract_descriptor,
    }
    return oto, prefixes, names, tensor_unsafe_exclusions, candidate_prefixes, dependency_control


def module_candidate_group_report(oto, candidate_prefixes: list[str]) -> list[dict[str, Any]]:
    """列出实验 profile 候选参数实际落入的 OTO 参数组及其可剪枝状态。"""

    records: list[dict[str, Any]] = []
    node_group_types = {
        str(node_group.id): type(node_group).__name__
        for node_group in oto._graph.node_groups.values()
    }
    for group in oto._graph.get_param_groups():
        parameter_names = list(group.get("p_names", []))
        matched_names = sorted(
            name
            for name in parameter_names
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in candidate_prefixes)
        )
        if not matched_names:
            continue
        records.append(
            {
                "node_group_id": group.get("id"),
                "node_group_type": node_group_types.get(str(group.get("id"))),
                "num_groups": int(group.get("num_groups", 0)),
                "is_prunable": bool(group.get("is_prunable")),
                "is_auxiliary": bool(group.get("is_auxiliary")),
                "operator_names": sorted({str(name) for name in group.get("op_names", [])}),
                "candidate_parameter_names": matched_names,
                # 该字段是候选卷积真实的前后依赖闭包；其中可能包含 BN、后继卷积
                # 输入或 CSP/Concat 传播所绑定的参数，不能只看 candidate 名称。
                "dependency_closure_parameter_names": sorted(parameter_names),
                "external_dependency_parameter_names": sorted(
                    name for name in parameter_names if name not in matched_names
                ),
                "attention_prune_mode": group.get("prune_mode"),
                "attention_contract": group.get("attention_contract"),
                "attention_num_heads": group.get("num_heads"),
                "attention_key_dim": group.get("key_dim"),
                "attention_head_dim": group.get("head_dim"),
                "c2psa_contract": group.get("c2psa_contract"),
                "c2psa_module_prefix": group.get("c2psa_module_prefix"),
                "c2psa_original_segment_width": group.get(
                    "c2psa_original_segment_width"
                ),
                "c3k2_contract": group.get("c3k2_contract"),
                "c3k2_module_prefix": group.get("c3k2_module_prefix"),
                "c3k2_original_segment_width": group.get(
                    "c3k2_original_segment_width"
                ),
                "c3k2_dynamic_contract": group.get("c3k2_dynamic_contract"),
                "c3k2_dynamic_parent_prefix": group.get(
                    "c3k2_dynamic_parent_prefix"
                ),
                "c3k2_dynamic_child_prefix": group.get(
                    "c3k2_dynamic_child_prefix"
                ),
                "c3k2_dynamic_segment_index": group.get(
                    "c3k2_dynamic_segment_index"
                ),
                "c3k2_dynamic_original_width": group.get(
                    "c3k2_dynamic_original_width"
                ),
                "parameter_role": group.get("parameter_role"),
            }
        )
    return records


def candidate_profile_guard_overlap_report(
    candidate_groups_before: list[dict[str, Any]],
    candidate_groups_after: list[dict[str, Any]],
    protected_parameter_names: list[str],
    guard_group_sources: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """解释 profile 保护是否按依赖组连带封闭了候选。

    该审计只属于 YOLO11 实验入口，不改变 OTO 的保护语义。输出同时记录参数
    交集和节点组级 guard 来源，避免把“候选前缀存在”误判为“保护后可剪”。
    """

    protected_names = set(protected_parameter_names)
    after_by_id = {
        str(record["node_group_id"]): record
        for record in candidate_groups_after
        if record.get("node_group_id") is not None
    }
    sources = guard_group_sources or {}
    records: list[dict[str, Any]] = []
    for before in candidate_groups_before:
        group_id = str(before.get("node_group_id"))
        after = after_by_id.get(group_id, {})
        closure_names = set(before.get("dependency_closure_parameter_names", []))
        was_prunable = bool(before.get("is_prunable")) and not bool(before.get("is_auxiliary"))
        remains_prunable = bool(after.get("is_prunable")) and not bool(after.get("is_auxiliary"))
        records.append(
            {
                "node_group_id": before.get("node_group_id"),
                "candidate_parameter_names": before.get("candidate_parameter_names", []),
                "is_prunable_before_profile_guards": was_prunable,
                "is_prunable_after_profile_guards": remains_prunable,
                "blocked_by_profile_guards": was_prunable and not remains_prunable,
                "protected_dependency_parameter_names": sorted(closure_names & protected_names),
                "node_group_guard_sources": sorted(sources.get(group_id, set())),
            }
        )
    blocked = [record for record in records if record["blocked_by_profile_guards"]]
    unexplained = [
        record
        for record in blocked
        if not record["protected_dependency_parameter_names"]
        and not record["node_group_guard_sources"]
    ]
    return {
        "candidate_group_count_before_profile_guards": len(records),
        "candidate_group_count_after_profile_guards": sum(
            record["is_prunable_after_profile_guards"] for record in records
        ),
        "blocked_candidate_group_count": len(blocked),
        "blocked_candidate_group_ids": sorted(
            str(record["node_group_id"]) for record in blocked
        ),
        "unexplained_blocked_candidate_groups": unexplained,
        "records": records,
        "policy": (
            "候选被保护连带封闭必须有参数交集或节点组 guard 证据；"
            "定向 probe 的目标候选不得全部被封闭。"
        ),
    }


def mark_fixed_output_node_groups(
    oto,
    model,
    fixed_output_prefixes: list[str],
) -> list[dict[str, Any]]:
    """精确固定最终预测模块的输出节点组，同时保留其输入维传播能力。"""

    named_modules = dict(model.named_modules())
    protection = mark_profile_unprunable_by_module_prefixes(
        oto,
        model,
        fixed_output_prefixes,
        include_stateful_descendants=False,
        classification="fixed_output_semantics",
        action="fix_output_group_allow_consumer_input_propagation",
    )
    if protection["missing_module_prefixes"]:
        raise RuntimeError(
            "固定输出模块未出现在 named_modules："
            f"{protection['missing_module_prefixes']}"
        )
    records = protection["groups"]
    for record in records:
        record["was_prunable"] = record["is_prunable_before"]
        record["is_prunable"] = record["is_prunable_after"]

    unmatched_prefixes = protection["unmatched_module_prefixes"]
    # dfl.conv 是冻结常量，某些 OTO/ONNX 版本会把它折叠而不单独形成节点组。
    # 仅允许这种无可训练参数的模块不匹配；最终分类/回归卷积必须精确匹配。
    unmatched_trainable = [
        prefix
        for prefix in unmatched_prefixes
        if any(parameter.requires_grad for parameter in named_modules[prefix].parameters())
    ]
    records.extend(
        {
            "node_group_id": None,
            "module_prefixes": [prefix],
            "was_prunable": None,
            "is_prunable": None,
            "fixed_by": "ERROR_trainable_module_not_matched_to_oto_group",
        }
        for prefix in unmatched_trainable
    )
    records.extend(
        {
            "node_group_id": None,
            "module_prefixes": [prefix],
            "was_prunable": False,
            "is_prunable": False,
            "fixed_by": "frozen_constant_not_materialized_as_oto_group",
        }
        for prefix in unmatched_prefixes
        if prefix not in unmatched_trainable
    )
    return records


def isolate_candidate_dependency_groups(
    oto,
    candidate_groups: list[dict[str, Any]],
    *,
    enabled: bool,
) -> dict[str, Any]:
    """probe 模式只保留候选依赖组；Concat 等 auxiliary 组继续承担索引传播。"""

    candidate_group_ids = {
        record["node_group_id"]
        for record in candidate_groups
        if not record["is_auxiliary"] and record.get("node_group_id") is not None
    }
    isolated_group_ids: list[str] = []
    if enabled:
        for node_group in oto._graph.node_groups.values():
            if node_group.is_auxiliary or node_group.id in candidate_group_ids:
                continue
            if node_group.is_prunable:
                node_group.is_prunable = False
                isolated_group_ids.append(node_group.id)
    return {
        "probe_group_isolation_enabled": enabled,
        "probe_allowed_candidate_group_ids": sorted(candidate_group_ids),
        "probe_preserved_auxiliary_group_count": sum(
            bool(group.is_auxiliary) for group in oto._graph.node_groups.values()
        ),
        "probe_isolated_non_candidate_group_ids": sorted(isolated_group_ids),
        "probe_group_isolation_implementation": "yolo11_profile_local_node_group_v1",
    }


def module_candidate_group_issues(
    module_coverage_profile: str,
    candidate_prefixes: list[str],
    candidate_groups: list[dict[str, Any]],
    tensor_unsafe_exclusions: list[dict[str, Any]] | None = None,
    dependency_control: dict[str, Any] | None = None,
) -> list[str]:
    """检查实验 profile 是否确实暴露了预期、可剪且非 auxiliary 的 OTO 组。"""

    if module_coverage_profile == "conservative_v1":
        return []
    issues: list[str] = []
    attention_optimizer_coverage = (dependency_control or {}).get(
        "yolo11_attention_optimizer_coverage",
        {},
    )
    if (
        module_coverage_profile in ATTENTION_PROFILE_MODES
        and isinstance(attention_optimizer_coverage, dict)
        and attention_optimizer_coverage.get("status") != "pass"
    ):
        failed_records = [
            record
            for record in attention_optimizer_coverage.get("records", [])
            if isinstance(record, dict) and record.get("status") != "pass"
        ]
        issues.append(
            "Attention optimizer 参数所有权检查失败："
            f"{len(failed_records)} 个模块未通过；详细参数记录见结构化报告。"
        )
    matched_names = [
        name
        for record in candidate_groups
        for name in record["candidate_parameter_names"]
    ]
    missing_prefixes = [
        prefix
        for prefix in candidate_prefixes
        if not any(name == prefix or name.startswith(f"{prefix}.") for name in matched_names)
    ]
    if missing_prefixes:
        issues.append(f"候选前缀未进入任何 OTO 参数组：{missing_prefixes}")

    contract_fixed_outputs = set(
        (dependency_control or {})
        .get("contract", {})
        .get("descriptor", {})
        .get("contract_fixed_output_candidate_prefixes", [])
    )
    accidentally_opened_fixed_outputs = sorted(
        set(candidate_prefixes) & contract_fixed_outputs
    )
    if accidentally_opened_fixed_outputs:
        issues.append(
            "复合合同要求固定的外部输出轴被误加入 HESSO 候选："
            f"{accidentally_opened_fixed_outputs}"
        )

    prunable_groups = [
        record
        for record in candidate_groups
        if record["is_prunable"] and not record["is_auxiliary"]
    ]
    if not prunable_groups:
        issues.append("实验 profile 没有暴露任何可剪且非 auxiliary 的候选参数组。")

    guard_overlap = (dependency_control or {}).get("candidate_profile_guard_overlap", {})
    unexplained_guard_blocks = guard_overlap.get("unexplained_blocked_candidate_groups", [])
    if unexplained_guard_blocks:
        issues.append(
            "候选依赖组在 profile 保护后被封闭，但没有参数交集或节点组 guard 证据："
            f"{unexplained_guard_blocks}"
        )

    if module_coverage_profile in ALL_VERIFIED_SAFE_PROFILES:
        coverage = (dependency_control or {}).get("candidate_family_coverage", {})
        family_coverage = coverage.get("families", {}) if isinstance(coverage, dict) else {}
        required_families = ALL_VERIFIED_SAFE_REQUIRED_FAMILIES_BY_PROFILE[
            module_coverage_profile
        ]
        missing_candidate_families = [
            family
            for family in required_families
            if family not in family_coverage
        ]
        fully_blocked_families = [
            family
            for family in required_families
            if family in family_coverage
            and int(
                family_coverage[family].get(
                    "prunable_group_count_after_protection",
                    0,
                )
            )
            <= 0
        ]
        if missing_candidate_families:
            issues.append(
                "全部安全模块 profile 未包含已验证模块族候选："
                f"{missing_candidate_families}"
            )
        if fully_blocked_families:
            issues.append(
                "已验证模块族候选与保护边界耦合后被全部封闭："
                f"{fully_blocked_families}；coverage={family_coverage}"
            )

    directed_families = set(
        (dependency_control or {})
        .get("contract", {})
        .get("descriptor", {})
        .get("probe_module_families", [])
    )
    if (
        module_coverage_profile in PSA_PARTIAL_OPEN_PROFILES
        and module_coverage_profile != ALL_CV1_OPEN_PROFILE
    ):
        psa_ffn_groups = [
            record
            for record in candidate_groups
            if any(".ffn.0.conv.weight" in name for name in record["candidate_parameter_names"])
        ]
        if (not directed_families or "PSA_FFN" in directed_families) and not psa_ffn_groups:
            issues.append("PSA FFN profile 未找到任何 PSABlock.ffn.0 候选依赖组。")
        blocked_psa_ffn_groups = [
            record
            for record in psa_ffn_groups
            if not record["is_prunable"] or record["is_auxiliary"]
        ]
        if blocked_psa_ffn_groups:
            issues.append(
                "PSABlock.ffn.0 候选被 Attention/残差/边界保护连带封闭："
                f"{blocked_psa_ffn_groups}"
            )

    if (
        module_coverage_profile
        in {"c2psa_output_closure_v1_experimental", *ALL_VERIFIED_SAFE_PROFILES}
        and (not directed_families or "C2PSA_OUTPUT" in directed_families)
    ):
        # 不能仅用 ``*.cv2.conv.weight`` 后缀识别 C2PSA 输出：C3k2、
        # Bottleneck 等大量普通候选也具有同名 cv2，联合 profile 下会被误判为
        # C2PSA_OUTPUT，并把合法的 chunk/concat 隔离组错误报告为 C2PSA 被封闭。
        # 使用前面按真实模块祖先生成的 family coverage 精确定位 C2PSA.cv2 组。
        family_coverage = (
            (dependency_control or {})
            .get("candidate_family_coverage", {})
            .get("families", {})
        )
        c2psa_output_group_ids = {
            str(group_id)
            for group_id in (
                family_coverage.get("C2PSA_OUTPUT", {})
                .get("candidate_group_ids", [])
            )
            if group_id is not None
        }
        c2psa_output_groups = [
            record
            for record in candidate_groups
            if str(record.get("node_group_id")) in c2psa_output_group_ids
        ]
        if not c2psa_output_groups:
            issues.append("C2PSA output profile 未找到 C2PSA.cv2 候选依赖组。")
        blocked_c2psa_output_groups = [
            record
            for record in c2psa_output_groups
            if not record["is_prunable"] or record["is_auxiliary"]
        ]
        if blocked_c2psa_output_groups:
            issues.append(
                "C2PSA.cv2 输出候选被后继 C3k2/Concat/边界保护连带封闭："
                f"{blocked_c2psa_output_groups}"
            )

    if module_coverage_profile in C2PSA_EXTERNAL_WIDTH_PROFILES:
        split_groups = [
            record
            for record in candidate_groups
            if record.get("c2psa_contract") == "yolo11_c2psa_paired_split_v1"
        ]
        configured_contracts = (dependency_control or {}).get(
            "yolo11_c2psa_paired_split_contracts",
            [],
        )
        if not split_groups:
            issues.append("C2PSA paired-split profile 未找到规范 paired-split 候选组。")
        if len(split_groups) != len(configured_contracts):
            issues.append(
                "C2PSA paired-split 候选组数量与配置合同数量不一致："
                f"groups={len(split_groups)}, contracts={len(configured_contracts)}"
            )
        invalid_split_groups = [
            record
            for record in split_groups
            if record.get("node_group_type") != "NodeGroup"
            or record.get("parameter_role")
            != "c2psa_external_residual_width_owner"
            or int(record.get("num_groups") or 0)
            != int(record.get("c2psa_original_segment_width") or 0)
            or not any(
                name.endswith(".cv1.conv.weight")
                for name in record.get("candidate_parameter_names", [])
            )
            or not bool(record.get("is_prunable"))
            or bool(record.get("is_auxiliary"))
        ]
        if invalid_split_groups:
            issues.append(
                "C2PSA paired-split 候选组类型、逻辑宽度、参数角色或可剪状态错误："
                f"{invalid_split_groups}"
            )

    if module_coverage_profile in C3K2_PAIRED_CHUNK_PROFILES:
        chunk_groups = [
            record
            for record in candidate_groups
            if record.get("c3k2_contract") == "yolo11_c3k2_paired_chunk_v1"
        ]
        configured_contracts = (dependency_control or {}).get(
            "yolo11_c3k2_paired_chunk_contracts",
            [],
        )
        if not chunk_groups:
            issues.append("C3k2 paired-chunk profile 未找到规范 paired-chunk 候选组。")
        if len(chunk_groups) != len(configured_contracts):
            issues.append(
                "C3k2 paired-chunk 候选组数量与配置合同数量不一致："
                f"groups={len(chunk_groups)}, contracts={len(configured_contracts)}"
            )
        invalid_chunk_groups = [
            record
            for record in chunk_groups
            if record.get("node_group_type") != "NodeGroup"
            or record.get("parameter_role")
            != "c3k2_paired_chunk_external_width_owner"
            or int(record.get("num_groups") or 0)
            != int(record.get("c3k2_original_segment_width") or 0)
            or not any(
                name.endswith(".cv1.conv.weight")
                for name in record.get("candidate_parameter_names", [])
            )
            or not bool(record.get("is_prunable"))
            or bool(record.get("is_auxiliary"))
        ]
        if invalid_chunk_groups:
            issues.append(
                "C3k2 paired-chunk 候选组类型、逻辑宽度、参数角色或可剪状态错误："
                f"{invalid_chunk_groups}"
            )

    if module_coverage_profile in C3K2_DYNAMIC_CONCAT_PROFILES:
        dynamic_groups = [
            record
            for record in candidate_groups
            if record.get("c3k2_dynamic_contract")
            == "yolo11_c3k2_segmented_dynamic_concat_v1"
        ]
        configured_contracts = (dependency_control or {}).get(
            "yolo11_c3k2_dynamic_concat_contracts",
            [],
        )
        if not dynamic_groups:
            issues.append(
                "C3k2 dynamic-concat profile 未找到规范 child-output 候选组。"
            )
        if len(dynamic_groups) != len(configured_contracts):
            issues.append(
                "C3k2 dynamic-concat 候选组数量与配置合同数量不一致："
                f"groups={len(dynamic_groups)}, contracts={len(configured_contracts)}"
            )
        invalid_dynamic_groups = [
            record
            for record in dynamic_groups
            if record.get("node_group_type") != "NodeGroup"
            or record.get("parameter_role")
            != "c3k2_dynamic_child_output_owner"
            or int(record.get("num_groups") or 0)
            != int(record.get("c3k2_dynamic_original_width") or 0)
            or not any(
                name.endswith(".cv3.conv.weight")
                for name in record.get("candidate_parameter_names", [])
            )
            or not bool(record.get("is_prunable"))
            or bool(record.get("is_auxiliary"))
        ]
        if invalid_dynamic_groups:
            issues.append(
                "C3k2 dynamic-concat 候选组类型、逻辑宽度、参数角色或可剪状态错误："
                f"{invalid_dynamic_groups}"
            )

    if module_coverage_profile in C3K2_COMPOSITE_FIXED_OUTPUT_GUARD_PROFILES:
        composite_contracts = (dependency_control or {}).get(
            "yolo11_c3k2_composite_contracts",
            [],
        )
        if not composite_contracts:
            issues.append("C3k2 composite profile 未配置父级复合所有权合同。")
        invalid_composite_contracts = []
        for contract in composite_contracts:
            if not isinstance(contract, dict):
                invalid_composite_contracts.append(contract)
                continue
            root_owner = set(contract.get("root_owner_parameter_names", []))
            dynamic_owner = set(
                contract.get("dynamic_child_owner_parameter_names", [])
            )
            dynamic_children = contract.get("dynamic_child_contracts", [])
            if (
                contract.get("contract")
                != "yolo11_c3k2_paired_chunk_dynamic_concat_v1"
                or not root_owner
                or not dynamic_owner
                or not dynamic_children
                or bool(root_owner & dynamic_owner)
                or contract.get("ownership_overlap")
                or contract.get("physical_pruning_policy")
                != "parent_coordinator_single_pass_original_cv2_coordinate"
            ):
                invalid_composite_contracts.append(contract)
        if invalid_composite_contracts:
            issues.append(
                "C3k2 composite 参数所有权或单次 cv2 坐标裁剪合同错误："
                f"{invalid_composite_contracts}"
            )

        # ``cv2`` 对外输出是 C3k2 父模块接口，必须固定；但其输入列仍由父级
        # coordinator 根据 root paired-chunk 和 child dynamic-concat 的零索引裁剪。
        # 仅把 cv2 从候选前缀中删除并不足够：它可能与下游 Conv 的输入轴同处一个
        # OTO 依赖组。此处在训练前同时验证精确输出组 guard 已生效，且没有误伤两类
        # 复合 owner；否则直接 audit fail-closed，禁止进入 HESSO。
        fixed_output_prefixes = set(
            (dependency_control or {}).get("fixed_output_module_prefixes", [])
        )
        fixed_output_groups = (dependency_control or {}).get(
            "fixed_output_node_groups",
            [],
        )
        expected_cv2_outputs = {
            f"{contract.get('module_prefix')}.cv2"
            for contract in composite_contracts
            if isinstance(contract, dict) and contract.get("module_prefix")
        }
        missing_fixed_output_registration = sorted(
            expected_cv2_outputs - fixed_output_prefixes
        )
        if missing_fixed_output_registration:
            issues.append(
                "C3k2 composite 外部 cv2 输出未注册精确 guard："
                f"{missing_fixed_output_registration}"
            )
        fixed_group_prefixes = {
            str(prefix)
            for record in fixed_output_groups
            if isinstance(record, dict)
            and record.get("node_group_id") is not None
            and record.get("is_prunable") is False
            for prefix in record.get("module_prefixes", [])
        }
        missing_fixed_output_groups = sorted(
            expected_cv2_outputs - fixed_group_prefixes
        )
        if missing_fixed_output_groups:
            issues.append(
                "C3k2 composite 外部 cv2 输出组未被实际固定："
                f"{missing_fixed_output_groups}"
            )
        composite_owner_group_ids = {
            str(contract.get("root_node_group_id"))
            for contract in composite_contracts
            if isinstance(contract, dict) and contract.get("root_node_group_id")
        }
        composite_owner_group_ids.update(
            str(child.get("node_group_id"))
            for contract in composite_contracts
            if isinstance(contract, dict)
            for child in contract.get("dynamic_child_contracts", [])
            if isinstance(child, dict) and child.get("node_group_id")
        )
        fixed_cv2_group_ids = {
            str(record.get("node_group_id"))
            for record in fixed_output_groups
            if isinstance(record, dict)
            and record.get("node_group_id") is not None
            and any(
                str(prefix) in expected_cv2_outputs
                for prefix in record.get("module_prefixes", [])
            )
        }
        owner_guard_collisions = sorted(
            fixed_cv2_group_ids & composite_owner_group_ids
        )
        if owner_guard_collisions:
            issues.append(
                "C3k2 composite cv2 输出 guard 误封闭 root/child 复合 owner："
                f"{owner_guard_collisions}"
            )

    if module_coverage_profile in OUTPUT_CONSUMER_CLOSURE_PROFILES:
        closure_contracts = (dependency_control or {}).get(
            "yolo11_output_consumer_closure_contracts",
            [],
        )
        resolved_contracts = [
            contract
            for contract in closure_contracts
            if isinstance(contract, dict)
            and contract.get("status") == "configured_resolved_consumer_closure"
        ]
        quarantined_contracts = [
            contract
            for contract in closure_contracts
            if isinstance(contract, dict)
            and contract.get("status") != "configured_resolved_consumer_closure"
        ]
        if not resolved_contracts:
            issues.append(
                "output-consumer closure profile 未发现任何完整解析、可作为 HESSO 候选的生产者—消费者合同："
                f"quarantined={quarantined_contracts}"
            )
        candidate_by_id = {
            str(record.get("node_group_id")): record for record in candidate_groups
        }
        invalid_closure_contracts = []
        for contract in resolved_contracts:
            if not isinstance(contract, dict):
                invalid_closure_contracts.append(contract)
                continue
            group = candidate_by_id.get(str(contract.get("node_group_id")))
            source_name = contract.get("producer_parameter_name")
            consumers = set(contract.get("consumer_input_parameter_names", []))
            consumer_records = [
                record
                for record in contract.get("consumer_records", [])
                if isinstance(record, dict)
            ]
            consumer_record_names = {
                str(record.get("consumer_parameter_name"))
                for record in consumer_records
                if record.get("consumer_parameter_name")
            }
            consumer_kinds = {
                str(record.get("consumer_kind"))
                for record in consumer_records
            }
            depthwise_consumer_names = {
                str(record.get("consumer_parameter_name"))
                for record in consumer_records
                if record.get("consumer_kind") == "depthwise_coupled_axis0"
                and record.get("consumer_parameter_name")
            }
            segmented_consumer_records = [
                record
                for record in consumer_records
                if record.get("consumer_kind")
                == "segmented_concat_standard_conv_input_axis1"
            ]
            segmented_contracts = {
                str(item.get("contract_id")): item
                for item in contract.get("segmented_concat_contracts", [])
                if isinstance(item, dict) and item.get("contract_id")
            }
            segmented_records_valid = bool(segmented_consumer_records) == bool(
                segmented_contracts
            ) and all(
                str(record.get("concat_contract_id")) in segmented_contracts
                and source_name
                in set(
                    segmented_contracts[
                        str(record.get("concat_contract_id"))
                    ].get("active_source_parameter_names", [])
                )
                and any(
                    isinstance(segment, dict)
                    and segment.get("is_active_owner")
                    and segment.get("producer_parameter_name") == source_name
                    and segment.get("input_index")
                    == record.get("concat_input_index")
                    for segment in segmented_contracts[
                        str(record.get("concat_contract_id"))
                    ].get("segments", [])
                )
                for record in segmented_consumer_records
            )
            group_closure = set(
                group.get("dependency_closure_parameter_names", []) if group else []
            )
            if (
                contract.get("contract") != "output_consumer_closure_v2"
                or not source_name
                or not consumers
                or consumers != consumer_record_names
                or not consumer_records
                or not consumer_kinds <= {
                    "standard_conv_input_axis1",
                    "depthwise_coupled_axis0",
                    "segmented_concat_standard_conv_input_axis1",
                }
                or not segmented_records_valid
                or bool(contract.get("topology_unresolved_paths"))
                or group is None
                or source_name not in group.get("candidate_parameter_names", [])
                # 深度卷积按同一输出轴物理裁剪，必须已经属于源 owner 的参数
                # 闭包；普通 Conv 输入轴属于另一节点组，由第二遍图反向推理裁剪。
                or not depthwise_consumer_names <= group_closure
                or not group.get("is_prunable")
                or group.get("is_auxiliary")
            ):
                invalid_closure_contracts.append(
                    {
                        "contract": contract,
                        "candidate_group": group,
                    }
                )
        if invalid_closure_contracts:
            issues.append(
                "output-consumer closure 候选、消费者闭包或可剪状态错误："
                f"{invalid_closure_contracts}"
            )
        invalid_quarantine = [
            contract
            for contract in quarantined_contracts
            if any(
                group.get("is_prunable_after") is not False
                for group in contract.get("quarantined_node_groups", [])
                if isinstance(group, dict)
            )
        ]
        if invalid_quarantine:
            issues.append(
                "output-consumer closure 不可解析输出轴没有被精确 fail-closed："
                f"{invalid_quarantine}"
            )

        # 联合 v4 同时启用了 C3k2 composite 和对外 cv2 输出闭包。这里必须证明
        # 复合 coordinator 只拥有 cv2 输入轴，而 source closure 只拥有 cv2 输出
        # 轴；任何同一 node group / source Parameter 的双重所有权都 fail-closed。
        if module_coverage_profile in C3K2_COMPOSITE_PROFILES:
            composite_contracts = (dependency_control or {}).get(
                "yolo11_c3k2_composite_contracts", []
            )
            invalid_composite_contracts = [
                contract
                for contract in composite_contracts
                if not isinstance(contract, dict)
                or contract.get("contract")
                != "yolo11_c3k2_paired_chunk_dynamic_concat_v1"
                or not contract.get("root_owner_parameter_names")
                or not contract.get("dynamic_child_owner_parameter_names")
                or not contract.get("dynamic_child_contracts")
                or bool(
                    set(contract.get("root_owner_parameter_names", []))
                    & set(contract.get("dynamic_child_owner_parameter_names", []))
                )
                or bool(contract.get("ownership_overlap"))
                or contract.get("physical_pruning_policy")
                != "parent_coordinator_single_pass_original_cv2_coordinate"
            ]
            if not composite_contracts or invalid_composite_contracts:
                issues.append(
                    "C3k2 composite 参数所有权或单次 cv2 坐标裁剪合同错误："
                    f"invalid={invalid_composite_contracts}"
                )
            ownership = (dependency_control or {}).get(
                "output_consumer_composite_ownership", {}
            )
            if (
                not isinstance(ownership, dict)
                or ownership.get("status") != "pass"
                or int(ownership.get("resolved_output_source_count") or 0)
                != len(resolved_contracts)
            ):
                issues.append(
                    "output-consumer closure 与 C3k2 composite 的输出/输入轴所有权冲突或覆盖不完整："
                    f"{ownership}"
                )
        if module_coverage_profile == ALL_VERIFIED_SAFE_NUM_HEAD_V4_OUTPUT_CLOSURE_PROFILE:
            expected_source_count = len(closure_contracts)
            segmented_contract_ids = {
                str(item.get("contract_id"))
                for contract in resolved_contracts
                if isinstance(contract, dict)
                for item in contract.get("segmented_concat_contracts", [])
                if isinstance(item, dict) and item.get("contract_id")
            }
            if (
                len(resolved_contracts) != expected_source_count
                or quarantined_contracts
                or len(segmented_contract_ids) != 3
            ):
                issues.append(
                    "v4 联合 output-consumer closure 覆盖不足："
                    f"resolved={len(resolved_contracts)}/{expected_source_count}, "
                    f"quarantined={len(quarantined_contracts)}, "
                    f"segmented_concat_contracts={len(segmented_contract_ids)}/3"
                )
            segmented_owner_coverage = (dependency_control or {}).get(
                "segmented_concat_candidate_owner_coverage", {}
            )
            if (
                not isinstance(segmented_owner_coverage, dict)
                or segmented_owner_coverage.get("status") != "pass"
                or int(segmented_owner_coverage.get("candidate_segment_count") or 0)
                <= 0
            ):
                issues.append(
                    "v4 segmented-concat 的可剪被动/显式输出段所有权覆盖不完整："
                    f"{segmented_owner_coverage}"
                )

    if module_coverage_profile in ATTENTION_PROFILE_MODES:
        expected_mode = ATTENTION_PROFILE_MODES[module_coverage_profile]
        expected_family = (
            "YOLO11_ATTN_NUM_HEAD"
            if expected_mode == "num_head"
            else "YOLO11_ATTN_HEAD_DIM"
        )
        attention_groups = [
            record
            for record in candidate_groups
            if "yolo11_attention" in record.get("operator_names", [])
        ]
        if (not directed_families or expected_family in directed_families) and not attention_groups:
            issues.append(
                f"YOLO11 Attention {expected_mode} profile 未找到复合算子候选组。"
            )
        def is_canonical_attention_group(record: dict[str, Any]) -> bool:
            names = set(record.get("candidate_parameter_names", []))
            has_qkv = any(name.endswith(".qkv.conv.weight") for name in names)
            has_pe = any(name.endswith(".pe.conv.weight") for name in names)
            expected_num_groups = (
                int(record.get("attention_num_heads") or 0)
                if expected_mode == "num_head"
                else int(record.get("attention_head_dim") or 0)
            )
            return (
                record.get("node_group_type") == "NodeGroupComposedOp"
                and record.get("attention_prune_mode") == expected_mode
                and record.get("attention_contract") == "yolo11_partitioned_qkv_v1"
                and int(record.get("num_groups") or 0) == expected_num_groups
                and expected_num_groups > 0
                and has_qkv
                and has_pe
            )

        canonical_attention_groups = [
            record for record in attention_groups if is_canonical_attention_group(record)
        ]
        wrong_contract_groups = [
            record
            for record in attention_groups
            if not is_canonical_attention_group(record)
        ]
        if wrong_contract_groups:
            issues.append(
                "YOLO11 Attention 内部候选组类型/模式/逻辑组或索引合同错误："
                f"{wrong_contract_groups}"
            )
        if attention_groups and not canonical_attention_groups:
            issues.append(
                "YOLO11 Attention 未找到 qkv+pe 共同驱动的规范复合候选组。"
            )
        blocked_attention_groups = [
            record
            for record in canonical_attention_groups
            if not record["is_prunable"] or record["is_auxiliary"]
        ]
        if blocked_attention_groups:
            issues.append(
                "YOLO11 Attention 候选被外部保护或依赖规则连带封闭："
                f"{blocked_attention_groups}"
            )
        physical_consumer_prefixes = set(
            (dependency_control or {}).get(
                "yolo11_attention_physical_consumer_prefixes",
                [],
            )
        )
        physical_consumer_groups = (
            dependency_control or {}
        ).get("yolo11_attention_physical_consumer_groups", [])
        matched_physical_consumers = {
            name.rsplit(".weight", 1)[0]
            for record in physical_consumer_groups
            for name in record.get("candidate_parameter_names", [])
            if name.endswith(".weight")
        }
        missing_physical_consumers = sorted(
            physical_consumer_prefixes - matched_physical_consumers
        )
        if missing_physical_consumers:
            issues.append(
                "YOLO11 Attention proj 物理消费者未进入外部输出依赖组："
                f"{missing_physical_consumers}"
            )
        invalid_physical_consumer_groups = [
            record
            for record in physical_consumer_groups
            if record.get("node_group_type") != "NodeGroup"
            or record.get("is_prunable")
            or record.get("is_auxiliary")
        ]
        if invalid_physical_consumer_groups:
            issues.append(
                "YOLO11 Attention proj 外部输出组必须固定且非 auxiliary；"
                "其输入轴只允许由复合算子物理联动："
                f"{invalid_physical_consumer_groups}"
            )

    if module_coverage_profile in DENSE_CLOSURE_PROFILES:
        unmatched_fixed_outputs = [
            record
            for record in (dependency_control or {}).get("fixed_output_node_groups", [])
            if record.get("fixed_by") == "ERROR_trainable_module_not_matched_to_oto_group"
        ]
        if unmatched_fixed_outputs:
            issues.append(f"Detect 可训练最终输出模块未匹配到 OTO 节点组：{unmatched_fixed_outputs}")
        chunk_source_quarantine = (dependency_control or {}).get("c3k2_chunk_source_quarantine", {})
        unmatched_chunk_sources = chunk_source_quarantine.get("unmatched_boundary_prefixes", [])
        if unmatched_chunk_sources:
            issues.append(f"C3k2 chunk 源边界未匹配到 OTO 节点组：{unmatched_chunk_sources}")
        incorrectly_open_chunk_groups = [
            record
            for record in chunk_source_quarantine.get("groups", [])
            if record.get("action") != "managed_by_paired_chunk_contract"
            and (record.get("is_prunable_after") or record.get("is_auxiliary"))
        ]
        if incorrectly_open_chunk_groups:
            issues.append(
                "C3k2 chunk 源输出组未被正确隔离："
                f"{incorrectly_open_chunk_groups}"
            )
        dynamic_concat_quarantine = (dependency_control or {}).get(
            "c3k2_dynamic_concat_quarantine",
            {},
        )
        unsupported_dynamic_children = dynamic_concat_quarantine.get("unsupported_child_modules", [])
        if unsupported_dynamic_children:
            issues.append(
                "C3k2 动态 concat 存在未识别的直接子模块输出类型："
                f"{unsupported_dynamic_children}"
            )
        unmatched_dynamic_boundaries = dynamic_concat_quarantine.get(
            "unmatched_boundary_prefixes",
            [],
        )
        if unmatched_dynamic_boundaries:
            issues.append(
                "C3k2 动态 concat 子模块输出边界未匹配到 OTO 节点组："
                f"{unmatched_dynamic_boundaries}"
            )
        managed_dynamic_actions = {
            "managed_by_paired_chunk_contract",
            "managed_by_segmented_dynamic_concat_contract",
            # 复合 profile 中的共享 Bottleneck.cv2 输出不再由独立
            # DynamicConcat owner 管理，而是保持可剪并交给同一 C3k2
            # 父级 coordinator 与根部 paired-chunk 一次性物理裁剪。
            # 这与前两种已登记的合同管理状态同样是合法开放状态，不能被
            # audit 误报为“未正确隔离”。
            "managed_by_c3k2_composite_owner",
        }
        incorrectly_open_dynamic_groups = [
            record
            for record in dynamic_concat_quarantine.get("groups", [])
            if record.get("action") not in managed_dynamic_actions
            and (record.get("is_prunable_after") or record.get("is_auxiliary"))
        ]
        if incorrectly_open_dynamic_groups:
            issues.append(
                "C3k2 动态 concat 子模块输出组未被正确隔离："
                f"{incorrectly_open_dynamic_groups}"
            )
        candidate_group_by_id = {record["node_group_id"]: record for record in candidate_groups}
        candidate_group_ids = set(candidate_group_by_id)
        unsafe_candidate_groups = [
            record
            for record in (tensor_unsafe_exclusions or [])
            if record.get("node_group_id") in candidate_group_ids
        ]
        unclassified_unsafe_groups: list[dict[str, Any]] = []
        known_quarantined_group_ids: set[str] = set()
        for exclusion in unsafe_candidate_groups:
            group = candidate_group_by_id[exclusion["node_group_id"]]
            parameter_name = exclusion.get("parameter", "")
            csp_prefix = parameter_name.split(".m.", 1)[0]
            paired_conv_name = parameter_name.rsplit(".bn.", 1)[0] + ".conv.weight"
            expected_candidate_names = {
                f"{csp_prefix}.cv1.conv.weight",
                paired_conv_name,
            }
            is_verified_chunk_boundary = (
                exclusion.get("classification") == KNOWN_CSP_CHUNK_QUARANTINE
                and expected_candidate_names.issubset(set(group["candidate_parameter_names"]))
                and not group["is_prunable"]
                and not group["is_auxiliary"]
            )
            if is_verified_chunk_boundary:
                known_quarantined_group_ids.add(exclusion["node_group_id"])
            else:
                unclassified_unsafe_groups.append(exclusion)

        if unclassified_unsafe_groups:
            issues.append(
                "dense closure 候选依赖组存在未识别或未正确隔离的 tensor transformation failure："
                f"{unclassified_unsafe_groups}"
            )
        known_quarantined_group_ids.update(
            record["node_group_id"]
            for record in chunk_source_quarantine.get("groups", [])
            if record.get("action") != "managed_by_paired_chunk_contract"
        )
        known_quarantined_group_ids.update(
            record["node_group_id"]
            for record in dynamic_concat_quarantine.get("groups", [])
            if record.get("action") not in managed_dynamic_actions
        )
        quarantine_scope = candidate_quarantine_scope(
            candidate_group_ids=candidate_group_ids,
            constrained_group_ids=known_quarantined_group_ids,
        )
    return issues


def framework_version_summary() -> dict[str, str]:
    """记录当前模块依赖契约实际绑定的软件版本。"""

    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "ultralytics": str(ultralytics.__version__),
        "oto_revision": "local_OTOv2_v2_yolo11_c3k2_composite_ownership_v15_passive_concat_owner",
    }


def serialize_output_shape(output) -> list[list[int]]:
    if isinstance(output, (tuple, list)):
        return [list(item.shape) for item in output if hasattr(item, "shape")]
    return [list(output.shape)] if hasattr(output, "shape") else []


def primary_prediction_tensor(output: Any) -> torch.Tensor:
    """提取 YOLO eval/export 路径的主预测张量，忽略仅供训练调试的 feature list。"""

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in sorted(output):
            try:
                return primary_prediction_tensor(output[key])
            except TypeError:
                continue
    if isinstance(output, (tuple, list)):
        for item in output:
            try:
                return primary_prediction_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"模型输出中没有 Tensor：{type(output).__name__}")


def fixed_input_image_tensor(image_path: Path, imgsz: int) -> tuple[torch.Tensor, dict[str, Any]]:
    """以冻结的静态 letterbox 规则把一张图片转换成可同时交给 PyTorch/ORT 的输入。"""

    import cv2
    import numpy as np

    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"固定输入图片不存在：{image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV 无法读取固定输入图片：{image_path}")

    original_height, original_width = image.shape[:2]
    ratio = min(imgsz / original_height, imgsz / original_width)
    resized_width = max(1, int(round(original_width * ratio)))
    resized_height = max(1, int(round(original_height * ratio)))
    if (resized_width, resized_height) != (original_width, original_height):
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    pad_width = imgsz - resized_width
    pad_height = imgsz - resized_height
    left = int(round(pad_width / 2 - 0.1))
    right = int(round(pad_width / 2 + 0.1))
    top = int(round(pad_height / 2 - 0.1))
    bottom = int(round(pad_height / 2 + 0.1))
    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    if image.shape[:2] != (imgsz, imgsz):
        raise RuntimeError(
            f"固定输入 letterbox 尺寸错误：{image.shape[:2]}，期望 {(imgsz, imgsz)}"
        )

    rgb_chw = image[:, :, ::-1].transpose(2, 0, 1)
    array = np.ascontiguousarray(rgb_chw, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0)
    metadata = {
        "source_image": str(image_path),
        "source_image_sha256": sha256_file(image_path),
        "original_shape_hw": [original_height, original_width],
        "resized_shape_hw": [resized_height, resized_width],
        "letterbox_padding_lrtb": [left, right, top, bottom],
        "input_shape": list(tensor.shape),
        "input_dtype": str(tensor.dtype),
        "normalization": "RGB float32 / 255",
        "letterbox_value": 114,
        "letterbox_auto": False,
    }
    return tensor, metadata


def tensor_value_summary(tensor: torch.Tensor) -> dict[str, Any]:
    import numpy as np

    array = np.ascontiguousarray(tensor.detach().cpu().float().numpy())
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
        "min": float(array.min()) if array.size else None,
        "max": float(array.max()) if array.size else None,
        "mean": float(array.mean()) if array.size else None,
        "sha256": digest.hexdigest(),
    }


def fixed_input_forward_report(
    weights: Path,
    fixed_input: Path,
    *,
    imgsz: int,
    device,
    output_dir: Path,
    report_name: str,
) -> dict[str, Any]:
    """在物理 checkpoint 上执行冻结样本前向并保存输入张量和可追溯摘要。"""

    import numpy as np

    weights = weights.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_tensor, input_metadata = fixed_input_image_tensor(fixed_input, imgsz)
    input_npy = output_dir / f"{report_name}_input.npy"
    np.save(input_npy, input_tensor.numpy(), allow_pickle=False)

    model = YOLO(str(weights)).model.to(device).float().eval()
    with torch.no_grad():
        prediction = primary_prediction_tensor(model(input_tensor.to(device)))
    prediction_summary = tensor_value_summary(prediction)
    report = {
        "status": "pass" if prediction_summary["finite"] else "fail_non_finite_output",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "model_structure": model_structure_signature(model),
        "model_state": model_state_signature(model),
        "fixed_input": input_metadata,
        "fixed_input_npy": str(input_npy),
        "fixed_input_npy_sha256": sha256_file(input_npy),
        "prediction": prediction_summary,
        "device": str(device),
        "preprocess_contract": "static_letterbox_rgb_float32_v1",
    }
    report_path = output_dir / f"{report_name}.json"
    json_dump(report_path, report)
    if report["status"] != "pass":
        raise RuntimeError(f"固定输入前向出现非有限输出：{report_path}")
    return report


# ---------------------------
# OTO 建图审计
# ---------------------------
def audit_model(args: argparse.Namespace) -> None:
    """检查 OTO 图构建、保护策略和原始 YOLO11 前向。"""
    print_stage("开始执行 YOLO11 OTO 建图审计")
    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"权重不存在：{weights}")
    run_dir = Path(args.output).expanduser().resolve() / args.name
    if args.no_overwrite and run_dir.exists():
        raise FileExistsError(f"禁止覆盖已有审计目录：{run_dir}；请更换 --name / RUN_ID。")

    set_seed(args.seed)
    device = device_from_arg(args.device)
    yolo = YOLO(str(weights))
    model = yolo.model.to(device).float().eval()
    trainable_parameter_tensors = enable_oto_trainable_parameters(model)

    baseline_stats = model_stats(model, args.imgsz, device)
    (
        oto,
        prefixes,
        names,
        tensor_unsafe_exclusions,
        candidate_prefixes,
        dependency_control,
    ) = build_oto(
        model,
        args.imgsz,
        device,
        args.extra_protect_prefix,
        args.oto_opset,
        args.module_coverage_profile,
        args.module_probe_only,
        tuple(args.probe_module_family),
    )
    oto_graph_artifacts = (
        export_oto_graphs(
            oto,
            run_dir / "oto_graph",
            formats=tuple(args.oto_graph_format or ("svg", "pdf")),
            display_params=args.oto_graph_display_params,
            display_flops=args.oto_graph_display_flops,
        )
        if args.visualize_oto_graph
        else {"status": "not_requested", "records": []}
    )

    with torch.no_grad():
        output = model(torch.zeros(1, 3, args.imgsz, args.imgsz, device=device))
    candidate_groups = module_candidate_group_report(oto, candidate_prefixes)
    candidate_issues = module_candidate_group_issues(
        args.module_coverage_profile,
        candidate_prefixes,
        candidate_groups,
        tensor_unsafe_exclusions,
        dependency_control,
    )

    report = {
        "status": "audit_failed" if candidate_issues else "audit_passed",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "device": str(device),
        "imgsz": args.imgsz,
        "baseline_model_stats": baseline_stats,
        "output_shapes": serialize_output_shape(output),
        "oto_root": str(OTO_ROOT),
        "framework_versions": framework_version_summary(),
        "oto_graph_opset": args.oto_opset,
        "oto_graph_artifacts": oto_graph_artifacts,
        "module_coverage_profile": args.module_coverage_profile,
        "module_probe_only": args.module_probe_only,
        "probe_module_families": list(args.probe_module_family),
        "module_pruning_candidate_prefixes": candidate_prefixes,
        "module_pruning_candidate_count": len(candidate_prefixes),
        "module_pruning_candidate_groups": candidate_groups,
        "module_pruning_candidate_issues": candidate_issues,
        "module_dependency_control": dependency_control,
        "module_pruning_prunable_group_count": sum(
            record["is_prunable"] and not record["is_auxiliary"] for record in candidate_groups
        ),
        "oto_node_groups": len(oto._graph.node_groups),
        "oto_parameter_groups": len(list(oto._graph.get_param_groups())),
        "oto_prunable_parameter_groups": sum(
            bool(group["is_prunable"]) and not bool(group["is_auxiliary"])
            for group in oto._graph.get_param_groups()
        ),
        "trainable_parameter_tensors": trainable_parameter_tensors,
        "protected_module_prefixes": prefixes,
        "protected_parameter_count": len(names),
        "oto_tensor_unsafe_exclusions": tensor_unsafe_exclusions,
        "note": (
            "audit 只验证依赖图和原始模型前向；它不是剪枝质量、ONNX 或端侧收益结论。"
        ),
    }
    audit_path = run_dir / "audit.json"
    json_dump(audit_path, report)
    print_human_summary(
        "OTO 建图审计结果",
        [
            ("状态", "通过" if not candidate_issues else "失败"),
            ("剪枝配置", args.module_coverage_profile),
            ("候选前缀", len(candidate_prefixes)),
            (
                "保护后可剪组",
                report["module_pruning_prunable_group_count"],
            ),
            ("模块族覆盖", compact_family_coverage(dependency_control)),
            ("原模型参数量", f"{baseline_stats['parameters_m']:.3f}M"),
            ("原模型 FLOPs", f"{baseline_stats['flops_g']:.3f}G"),
            (
                "OTO 结构图",
                oto_graph_artifacts.get("manifest")
                if args.visualize_oto_graph
                else "未生成",
            ),
            (
                "问题",
                "无" if not candidate_issues else f"{len(candidate_issues)} 项，详见报告",
            ),
        ],
        artifact=audit_path,
    )
    if candidate_issues:
        raise RuntimeError(
            f"复杂模块候选组审计失败，共 {len(candidate_issues)} 项；详见：{audit_path}"
        )


@dataclass(frozen=True)
class AdaMCFTrainingConfig:
    oto_opset: int
    visualize_oto_graph: bool
    oto_graph_formats: tuple[str, ...]
    oto_graph_display_params: bool
    oto_graph_display_flops: bool
    imgsz: int
    prune_ratio: float
    warmup_epochs: float
    pruning_periods: int | None
    group_divisible: int
    module_coverage_profile: str
    module_probe_only: bool
    probe_module_families: tuple[str, ...]
    seed: int
    extra_protect_prefixes: tuple[str, ...]
    alpha_cache_batches: int
    enable_alpha_update: bool
    alpha_lr: float
    alpha_eps: float
    alpha_ema_gamma: float
    alpha_grad_ema_gamma: float
    alpha_max_step_l1: float

    def alpha_scheduler_cfg(self) -> dict[str, Any]:
        return {
            "lr": self.alpha_lr,
            "eps": self.alpha_eps,
            "ema_gamma": self.alpha_ema_gamma,
            "g_ema_gamma": self.alpha_grad_ema_gamma,
            "max_step_l1": self.alpha_max_step_l1,
            "eval_drop_ratio": 0.7,
            "eval_batch_limit": max(1, self.alpha_cache_batches),
            "alpha_seed": self.seed,
        }


# ---------------------------
# AdaMCF / HESSO 稀疏训练
# ---------------------------
def create_adamcf_trainer_class():
    """生成与当前 Ultralytics 版本绑定的 AdaMCF DetectionTrainer。"""

    class AdaMCFDetectionTrainer(DetectionTrainer):
        def __init__(self, *trainer_args, adamcf_config: AdaMCFTrainingConfig, **trainer_kwargs):
            self.adamcf_config = adamcf_config
            self.oto = None
            self.epoch_reporter = EpochReporter("prune")
            self.protected_module_prefixes: list[str] = []
            self.protected_parameter_names: list[str] = []
            self.tensor_unsafe_exclusions: list[dict[str, Any]] = []
            self.module_pruning_candidate_prefixes: list[str] = []
            self.module_pruning_candidate_groups: list[dict[str, Any]] = []
            self.module_dependency_control: dict[str, Any] = {}
            self.oto_graph_artifacts: dict[str, Any] = {
                "status": "not_requested",
                "records": [],
            }
            self.alpha_batches: deque[dict[str, Any]] = deque(maxlen=adamcf_config.alpha_cache_batches)
            self.alpha_update_count = 0
            super().__init__(*trainer_args, **trainer_kwargs)

        def _setup_train(self):
            if self.world_size > 1:
                raise RuntimeError(
                    "AdaMCF/OTO 当前只验证了单进程训练。请设置单卡 device，例如 --device 0，"
                    "不要使用 DDP。"
                )
            if self.args.compile:
                raise RuntimeError("AdaMCF/OTO 图追踪不支持 Ultralytics --compile；请显式设置 compile=False。")

            super()._setup_train()
            raw_model = unwrap_model(self.model)
            device = self.device
            (
                self.oto,
                self.protected_module_prefixes,
                self.protected_parameter_names,
                self.tensor_unsafe_exclusions,
                self.module_pruning_candidate_prefixes,
                self.module_dependency_control,
            ) = build_oto(
                raw_model,
                self.adamcf_config.imgsz,
                device,
                list(self.adamcf_config.extra_protect_prefixes),
                self.adamcf_config.oto_opset,
                self.adamcf_config.module_coverage_profile,
                self.adamcf_config.module_probe_only,
                self.adamcf_config.probe_module_families,
            )
            if self.adamcf_config.visualize_oto_graph:
                self.oto_graph_artifacts = export_oto_graphs(
                    self.oto,
                    Path(self.save_dir) / "oto_graph",
                    formats=self.adamcf_config.oto_graph_formats,
                    display_params=self.adamcf_config.oto_graph_display_params,
                    display_flops=self.adamcf_config.oto_graph_display_flops,
                )
            self.module_pruning_candidate_groups = module_candidate_group_report(
                self.oto,
                self.module_pruning_candidate_prefixes,
            )
            candidate_issues = module_candidate_group_issues(
                self.adamcf_config.module_coverage_profile,
                self.module_pruning_candidate_prefixes,
                self.module_pruning_candidate_groups,
                self.tensor_unsafe_exclusions,
                self.module_dependency_control,
            )
            if candidate_issues:
                diagnostic_path = Path(self.save_dir) / "candidate_group_audit_failure.json"
                json_dump(
                    diagnostic_path,
                    {
                        "status": "fail",
                        "issues": candidate_issues,
                        "module_pruning_candidate_groups": self.module_pruning_candidate_groups,
                        "module_dependency_control": self.module_dependency_control,
                    },
                )
                raise RuntimeError(
                    f"复杂模块候选组审计失败，共 {len(candidate_issues)} 项；"
                    f"详细诊断：{diagnostic_path}"
                )
            self.epoch_reporter.attach(self)

            batches_per_epoch = len(self.train_loader)
            # HESSO.num_steps 由 HESSO.step()（即一次 optimizer.step）驱动；不能把
            # dataloader 的 micro-batch 数直接传入，否则 batch < nbs 时剪枝会晚
            # ``gradient_accumulation`` 倍开始。这里取 Ultralytics 已实际采用的 accumulate。
            gradient_accumulation = int(getattr(self, "accumulate", 1))
            step_schedule = hesso_optimizer_step_schedule(
                batches_per_epoch=batches_per_epoch,
                epochs=self.epochs,
                warmup_epochs=self.adamcf_config.warmup_epochs,
                gradient_accumulation=gradient_accumulation,
            )
            total_steps = step_schedule["total_optimizer_steps"]
            start_pruning_step = step_schedule["start_pruning_step"]
            pruning_steps = step_schedule["pruning_steps"]
            periods = self.adamcf_config.pruning_periods or max(1, int(self.epochs - self.adamcf_config.warmup_epochs))
            periods = min(int(periods), pruning_steps)
            self.optimizer = self.oto.hesso(
                variant="sgd",
                lr=float(self.args.lr0),
                first_momentum=float(self.args.momentum),
                weight_decay=float(self.args.weight_decay),
                target_group_sparsity=self.adamcf_config.prune_ratio,
                start_pruning_step=start_pruning_step,
                pruning_steps=pruning_steps,
                pruning_periods=periods,
                group_divisible=self.adamcf_config.group_divisible,
                device=str(device),
                alpha_scheduler_cfg=self.adamcf_config.alpha_scheduler_cfg(),
                seed=self.adamcf_config.seed,
            )
            # Ultralytics 的 warmup/scheduler 需要 initial_lr；OTO 的自定义参数组默认没有该字段。
            for group in self.optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
            self._setup_scheduler()
            self.scheduler.last_epoch = self.start_epoch - 1

            runtime = {
                "status": "sparse_train_initialized",
                "imgsz": self.adamcf_config.imgsz,
                "batches_per_epoch": batches_per_epoch,
                "batch_size": int(getattr(self, "batch_size", 0) or 0),
                "nominal_batch_size": int(getattr(self.args, "nbs", 0) or 0),
                "hesso_step_unit": "optimizer.step",
                **step_schedule,
                "total_steps": total_steps,
                "start_pruning_step": start_pruning_step,
                "pruning_steps": pruning_steps,
                "pruning_periods": periods,
                "prune_ratio": self.adamcf_config.prune_ratio,
                "group_divisible": self.adamcf_config.group_divisible,
                "module_coverage_profile": self.adamcf_config.module_coverage_profile,
                "module_probe_only": self.adamcf_config.module_probe_only,
                "probe_module_families": list(self.adamcf_config.probe_module_families),
                "module_pruning_candidate_prefixes": self.module_pruning_candidate_prefixes,
                "module_pruning_candidate_count": len(self.module_pruning_candidate_prefixes),
                "module_pruning_candidate_groups": self.module_pruning_candidate_groups,
                "module_dependency_control": self.module_dependency_control,
                "module_pruning_prunable_group_count": sum(
                    record["is_prunable"] and not record["is_auxiliary"]
                    for record in self.module_pruning_candidate_groups
                ),
                "oto_root": str(OTO_ROOT),
                "framework_versions": framework_version_summary(),
                "oto_graph_opset": self.adamcf_config.oto_opset,
                "oto_graph_artifacts": self.oto_graph_artifacts,
                "oto_node_groups": len(self.oto._graph.node_groups),
                "protected_module_prefixes": self.protected_module_prefixes,
                "protected_parameter_count": len(self.protected_parameter_names),
                "oto_tensor_unsafe_exclusions": self.tensor_unsafe_exclusions,
                "alpha_scheduler": self.adamcf_config.alpha_scheduler_cfg(),
            }
            runtime_path = Path(self.save_dir) / "adamcf_runtime.json"
            json_dump(runtime_path, runtime)
            print_human_summary(
                "HESSO 剪枝调度",
                [
                    ("每轮 batch 数", batches_per_epoch),
                    ("梯度累积", gradient_accumulation),
                    ("优化器总步数", total_steps),
                    ("开始剪枝步", start_pruning_step),
                    ("剪枝步数", pruning_steps),
                    ("剪枝周期", periods),
                    ("目标组稀疏率", f"{self.adamcf_config.prune_ratio * 100:.1f}%"),
                ],
                artifact=runtime_path,
            )

        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            if self.adamcf_config.enable_alpha_update and self.adamcf_config.alpha_cache_batches > 0:
                self.alpha_batches.append(recursive_to_cpu(batch))
            return batch

        def _alpha_loss(self, predictions, target_batch):
            raw_model = unwrap_model(self.model)
            loss, _ = raw_model.loss(target_batch, predictions)
            return loss.sum()

        def _maybe_update_alpha(self) -> None:
            if not self.adamcf_config.enable_alpha_update:
                return
            if not self.alpha_batches:
                return
            if not self.optimizer.will_enter_new_period_on_next_step():
                return

            eval_batches = []
            for cached_batch in self.alpha_batches:
                batch = recursive_to_device(cached_batch, self.device)
                eval_batches.append((batch["img"], batch))

            self.optimizer.alpha_scheduler.update(
                model=unwrap_model(self.model),
                criterion=self._alpha_loss,
                inputs=eval_batches,
                targets=None,
                t_in_period=0,
            )
            self.alpha_update_count += 1

        def optimizer_step(self):
            """在 HESSO step 前插入 AdaMCF alpha 更新，余下语义保持 Ultralytics 原实现。"""

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self._maybe_update_alpha()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)

    return AdaMCFDetectionTrainer


def create_finetune_trainer_class(
    structured_model,
    source_weights: Path,
    *,
    admission_probe: bool = False,
):
    """保留物理子网结构进行微调，并附加 epoch 详细记录和结构指纹硬校验。"""

    structured_template = copy.deepcopy(structured_model).cpu().float()
    if hasattr(structured_template, "criterion"):
        structured_template.criterion = None
    expected_signature = model_structure_signature(structured_template)
    expected_state_signature = model_state_signature(structured_template)
    source_file_sha256 = sha256_file(source_weights)

    class DetailedFinetuneTrainer(DetectionTrainer):
        def __init__(self, *trainer_args, **trainer_kwargs):
            self.epoch_reporter = EpochReporter("finetune")
            self.structure_check: dict[str, Any] = {}
            self.loss_backward_check: dict[str, Any] = {}
            super().__init__(*trainer_args, **trainer_kwargs)

        def get_model(self, cfg=None, weights=None, verbose=True):
            """绕过 DetectionTrainer 对 YAML 的标准网络重建，直接复制 OTO 物理子网。"""

            del cfg, weights
            model = copy.deepcopy(structured_template).float()
            # OTO 压缩模型可能携带稀疏训练阶段已实例化的 criterion。v8DetectionLoss
            # 的 DFL proj 是普通 Tensor，不会随 model.to(cuda) 自动迁移；必须清空，
            # 让 DetectionModel.loss() 在首个 CUDA batch 上按当前设备重新创建它。
            if hasattr(model, "criterion"):
                model.criterion = None
            if verbose and hasattr(model, "info"):
                model.info(verbose=True)
            return model

        def _setup_train(self):
            super()._setup_train()
            actual_signature = model_structure_signature(self.model)
            actual_state_signature = model_state_signature(self.model)
            structure_matched = actual_signature["sha256"] == expected_signature["sha256"]
            weights_matched = actual_state_signature["sha256"] == expected_state_signature["sha256"]
            matched = structure_matched and weights_matched
            self.structure_check = {
                "matched": matched,
                "structure_matched": structure_matched,
                "weights_matched_before_training": weights_matched,
                "source_weights": str(source_weights),
                "source_weights_file_sha256": source_file_sha256,
                "expected_sha256": expected_signature["sha256"],
                "actual_sha256": actual_signature["sha256"],
                "expected_state_sha256": expected_state_signature["sha256"],
                "actual_state_sha256_before_training": actual_state_signature["sha256"],
                "expected_parameter_count": expected_signature["parameter_count"],
                "actual_parameter_count": actual_signature["parameter_count"],
                "expected_signature": expected_signature,
                "actual_signature": actual_signature,
                "expected_state_signature": expected_state_signature,
                "actual_state_signature_before_training": actual_state_signature,
            }
            load_check_path = Path(self.save_dir) / "admission" / "finetune_load_check.json"
            json_dump(load_check_path, self.structure_check)
            print_human_summary(
                "微调模型装载检查",
                [
                    ("结构一致", "通过" if structure_matched else "失败"),
                    ("初始权重一致", "通过" if weights_matched else "失败"),
                    (
                        "参数量",
                        f"{expected_signature['parameter_count']} → "
                        f"{actual_signature['parameter_count']}",
                    ),
                    ("输入模型", source_weights),
                ],
                artifact=load_check_path,
            )
            if not matched:
                raise RuntimeError(
                    "微调器中的模型结构或权重数值与输入的 OTO 物理子网不一致，"
                    "已在训练开始前终止，避免把标准 YOLO11 或部分权重迁移误当成结构化子网微调。"
                )
            self.epoch_reporter.attach(self)

        def optimizer_step(self):
            """在首个真实 optimizer step 前审计 detection loss 反向产生的梯度。"""

            first_checked_step = False
            report_path = Path(self.save_dir) / "admission" / "task_loss_backward.json"
            if admission_probe and not self.loss_backward_check:
                first_checked_step = True
                trainable_parameter_count = 0
                parameters_with_grad = 0
                parameters_with_nonzero_grad = 0
                non_finite_parameter_names: list[str] = []
                gradient_parameter_samples: list[str] = []
                for name, parameter in unwrap_model(self.model).named_parameters():
                    if not parameter.requires_grad:
                        continue
                    trainable_parameter_count += 1
                    gradient = parameter.grad
                    if gradient is None:
                        continue
                    parameters_with_grad += 1
                    if len(gradient_parameter_samples) < 20:
                        gradient_parameter_samples.append(name)
                    if not bool(torch.isfinite(gradient).all().item()):
                        non_finite_parameter_names.append(name)
                    if int(torch.count_nonzero(gradient).item()) > 0:
                        parameters_with_nonzero_grad += 1

                loss_value = None
                trainer_loss = getattr(self, "tloss", None)
                if isinstance(trainer_loss, torch.Tensor):
                    loss_value = float(trainer_loss.detach().float().sum().cpu().item())
                passed = (
                    parameters_with_grad > 0
                    and parameters_with_nonzero_grad > 0
                    and not non_finite_parameter_names
                    and loss_value is not None
                    and bool(torch.isfinite(torch.tensor(loss_value)).item())
                )
                self.loss_backward_check = {
                    "status": "pass" if passed else "fail",
                    "time_utc": datetime.now(timezone.utc).isoformat(),
                    "loss_sum": loss_value,
                    "trainable_parameter_count": trainable_parameter_count,
                    "parameters_with_grad": parameters_with_grad,
                    "parameters_with_nonzero_grad": parameters_with_nonzero_grad,
                    "non_finite_parameter_names": non_finite_parameter_names,
                    "gradient_parameter_samples": gradient_parameter_samples,
                    "amp": bool(getattr(self.args, "amp", False)),
                    "optimizer_step_completed": False,
                    "note": "该检查发生在物理子网首个真实 detection loss backward 之后、optimizer.step 之前。",
                }
                json_dump(report_path, self.loss_backward_check)
                print(
                    "[task-loss-backward-check] "
                    f"status={self.loss_backward_check['status']} loss={loss_value} "
                    f"with_grad={parameters_with_grad}/{trainable_parameter_count} "
                    f"nonzero_grad={parameters_with_nonzero_grad} "
                    f"non_finite={len(non_finite_parameter_names)}",
                    flush=True,
                )
                if not passed:
                    raise RuntimeError(f"物理子网 task loss 反向审计失败：{report_path}")

            super().optimizer_step()
            if first_checked_step:
                self.loss_backward_check["optimizer_step_completed"] = True
                json_dump(report_path, self.loss_backward_check)

    return DetailedFinetuneTrainer


def standard_ultralytics_checkpoint(structured_model, trainer, output_path: Path, ultralytics_version: str, torch) -> None:
    """将 OTO 的裸 DetectionModel 封装成 Ultralytics 可加载 checkpoint。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ema_model = copy.deepcopy(structured_model).cpu().half()
    # criterion 属于训练期缓存，不应固化进可迁移 checkpoint；其中包含未注册的
    # device-specific Tensor，重新加载到其他设备后可能导致 CPU/CUDA 混用。
    if hasattr(ema_model, "criterion"):
        ema_model.criterion = None
    checkpoint = {
        "epoch": int(getattr(trainer, "epoch", -1)),
        "best_fitness": getattr(trainer, "best_fitness", None),
        "model": None,
        "ema": ema_model,
        "updates": int(getattr(getattr(trainer, "ema", None), "updates", 0)),
        "optimizer": None,
        "train_args": vars(trainer.args).copy(),
        "date": datetime.now(timezone.utc).isoformat(),
        "version": ultralytics_version,
    }
    torch.save(checkpoint, output_path)


# ---------------------------
# 物理子网构建与保存
# ---------------------------
def c2psa_paired_split_structure_summary(model) -> dict[str, Any]:
    """验证 C2PSA paired-split 物理宽度、残差和最终 concat 合同。"""

    records: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if getattr(module, "_oto_c2psa_contract", None) != "yolo11_c2psa_paired_split_v1":
            continue
        current_c = int(getattr(module, "c", 0))
        original_c = int(getattr(module, "_oto_c2psa_original_c", 0))
        cv1 = getattr(module, "cv1", None)
        cv2 = getattr(module, "cv2", None)
        if current_c <= 0 or original_c <= 0 or not all(
            hasattr(wrapper, "conv") and hasattr(wrapper, "bn")
            for wrapper in (cv1, cv2)
        ):
            records.append(
                {
                    "module_prefix": name,
                    "status": "fail_missing_structure",
                    "current_c": current_c,
                    "original_c": original_c,
                }
            )
            continue
        original_output_dim = int(
            getattr(module, "_oto_c2psa_original_output_dim", cv2.conv.out_channels)
        )
        block_records: list[dict[str, Any]] = []
        for block_index, block in enumerate(list(getattr(module, "m", ()))):
            attn = getattr(block, "attn", None)
            ffn = getattr(block, "ffn", None)
            if not is_yolo11_attention_structure(attn) or not isinstance(
                ffn, torch.nn.Sequential
            ) or len(ffn) < 2:
                block_records.append(
                    {
                        "block_index": block_index,
                        "status": "fail_missing_psablock_structure",
                    }
                )
                continue
            ffn0, ffn1 = ffn[0], ffn[1]
            inner_dim = int(attn.num_heads) * int(attn.head_dim)
            expected_qkv = int(attn.num_heads) * (
                int(attn.key_dim) * 2 + int(attn.head_dim)
            )
            checks = {
                "attention_input_width": int(attn.qkv.conv.in_channels) == current_c,
                "attention_qkv_width": int(attn.qkv.conv.out_channels) == expected_qkv,
                "attention_pe_depthwise": (
                    int(attn.pe.conv.in_channels) == inner_dim
                    and int(attn.pe.conv.out_channels) == inner_dim
                    and int(attn.pe.conv.groups) == inner_dim
                ),
                "attention_proj_input": int(attn.proj.conv.in_channels) == inner_dim,
                "attention_proj_output": int(attn.proj.conv.out_channels) == current_c,
                "attention_proj_bn": int(attn.proj.bn.num_features) == current_c,
                "attention_axis_fixed": (
                    int(attn.num_heads)
                    == int(getattr(attn, "_oto_attention_original_num_heads", -1))
                    and int(attn.key_dim)
                    == int(getattr(attn, "_oto_attention_original_key_dim", -1))
                    and int(attn.head_dim)
                    == int(getattr(attn, "_oto_attention_original_head_dim", -1))
                ),
                "attention_external_mode": yolo11_attention_prune_mode(attn)
                == "external_width",
                "attention_persistent_forward": type(attn).__name__
                == "OTOPrunableYOLO11Attention",
                "ffn_input_width": int(ffn0.conv.in_channels) == current_c,
                "ffn_output_width": int(ffn1.conv.out_channels) == current_c,
                "ffn_output_bn": int(ffn1.bn.num_features) == current_c,
            }
            block_records.append(
                {
                    "block_index": block_index,
                    "status": "pass" if all(checks.values()) else "fail",
                    "num_heads": int(attn.num_heads),
                    "key_dim": int(attn.key_dim),
                    "head_dim": int(attn.head_dim),
                    "inner_dim": inner_dim,
                    "qkv_in_channels": int(attn.qkv.conv.in_channels),
                    "qkv_out_channels": int(attn.qkv.conv.out_channels),
                    "proj_in_channels": int(attn.proj.conv.in_channels),
                    "proj_out_channels": int(attn.proj.conv.out_channels),
                    "ffn0_in_channels": int(ffn0.conv.in_channels),
                    "ffn1_out_channels": int(ffn1.conv.out_channels),
                    "checks": checks,
                }
            )
        checks = {
            "logical_width_reduced": 0 < current_c < original_c,
            "cv1_paired_output": int(cv1.conv.out_channels) == 2 * current_c,
            "cv1_paired_bn": int(cv1.bn.num_features) == 2 * current_c,
            "cv2_segmented_concat_input": int(cv2.conv.in_channels) == 2 * current_c,
            "cv2_external_output_fixed": int(cv2.conv.out_channels)
            == original_output_dim,
            "all_psablocks_valid": bool(block_records)
            and all(record.get("status") == "pass" for record in block_records),
        }
        records.append(
            {
                "module_prefix": name,
                "status": "pass" if all(checks.values()) else "fail",
                "contract": "yolo11_c2psa_paired_split_v1",
                "physical_mode": getattr(module, "_oto_c2psa_physical_mode", None),
                "original_c": original_c,
                "current_c": current_c,
                "cv1_out_channels": int(cv1.conv.out_channels),
                "cv2_in_channels": int(cv2.conv.in_channels),
                "cv2_out_channels": int(cv2.conv.out_channels),
                "original_output_dim": original_output_dim,
                "blocks": block_records,
                "checks": checks,
            }
        )
    return {
        "status": (
            "not_applicable"
            if not records
            else "pass"
            if all(record.get("status") == "pass" for record in records)
            else "fail"
        ),
        "records": records,
    }


def c3k2_paired_chunk_structure_summary(model) -> dict[str, Any]:
    """验证全部 C3k2 paired-chunk 外部宽度和 concat 输入合同。"""

    records: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if getattr(module, "_oto_c3k2_contract", None) != "yolo11_c3k2_paired_chunk_v1":
            continue
        if getattr(module, "_oto_c3k2_composite_contract", None) == (
            "yolo11_c3k2_paired_chunk_dynamic_concat_v1"
        ):
            # 该父模块的 C3/C3k child 输出不再与 root 宽度同步，必须由联合
            # linter 检查；这里仅保留没有 independent child owner 的普通 root。
            continue
        current_c = int(getattr(module, "c", 0))
        original_c = int(getattr(module, "_oto_c3k2_original_c", 0))
        cv1 = getattr(module, "cv1", None)
        cv2 = getattr(module, "cv2", None)
        children = list(getattr(module, "m", ()))
        if current_c <= 0 or original_c <= 0 or not children or not all(
            hasattr(wrapper, "conv") and hasattr(wrapper, "bn")
            for wrapper in (cv1, cv2)
        ):
            records.append(
                {
                    "module_prefix": name,
                    "status": "fail_missing_structure",
                    "current_c": current_c,
                    "original_c": original_c,
                }
            )
            continue
        original_output_dim = int(
            getattr(module, "_oto_c3k2_original_output_dim", cv2.conv.out_channels)
        )
        child_records: list[dict[str, Any]] = []
        for child_index, child in enumerate(children):
            child_type = type(child).__name__
            if child_type == "Bottleneck":
                checks = {
                    "input_width": int(child.cv1.conv.in_channels) == current_c,
                    "output_width": int(child.cv2.conv.out_channels) == current_c,
                    "output_bn": int(child.cv2.bn.num_features) == current_c,
                }
                details = {
                    "cv1_in_channels": int(child.cv1.conv.in_channels),
                    "cv2_out_channels": int(child.cv2.conv.out_channels),
                }
            elif child_type in {"C3", "C3k"}:
                checks = {
                    "branch1_input_width": int(child.cv1.conv.in_channels) == current_c,
                    "branch2_input_width": int(child.cv2.conv.in_channels) == current_c,
                    "output_width": int(child.cv3.conv.out_channels) == current_c,
                    "output_bn": int(child.cv3.bn.num_features) == current_c,
                }
                details = {
                    "cv1_in_channels": int(child.cv1.conv.in_channels),
                    "cv2_in_channels": int(child.cv2.conv.in_channels),
                    "cv3_out_channels": int(child.cv3.conv.out_channels),
                }
            else:
                child_records.append(
                    {
                        "child_index": child_index,
                        "type": child_type,
                        "status": "fail_unsupported_child",
                    }
                )
                continue
            child_records.append(
                {
                    "child_index": child_index,
                    "type": child_type,
                    "status": "pass" if all(checks.values()) else "fail",
                    **details,
                    "checks": checks,
                }
            )
        checks = {
            "logical_width_reduced": 0 < current_c < original_c,
            "cv1_paired_output": int(cv1.conv.out_channels) == 2 * current_c,
            "cv1_paired_bn": int(cv1.bn.num_features) == 2 * current_c,
            "cv2_segmented_concat_input": int(cv2.conv.in_channels)
            == (2 + len(children)) * current_c,
            "cv2_external_output_fixed": int(cv2.conv.out_channels)
            == original_output_dim,
            "all_children_synchronized": bool(child_records)
            and all(record.get("status") == "pass" for record in child_records),
        }
        records.append(
            {
                "module_prefix": name,
                "status": "pass" if all(checks.values()) else "fail",
                "contract": "yolo11_c3k2_paired_chunk_v1",
                "physical_mode": getattr(module, "_oto_c3k2_physical_mode", None),
                "original_c": original_c,
                "current_c": current_c,
                "segment_count": 2 + len(children),
                "cv1_out_channels": int(cv1.conv.out_channels),
                "cv2_in_channels": int(cv2.conv.in_channels),
                "cv2_out_channels": int(cv2.conv.out_channels),
                "original_output_dim": original_output_dim,
                "children": child_records,
                "checks": checks,
            }
        )
    return {
        "status": (
            "not_applicable"
            if not records
            else "pass"
            if all(record.get("status") == "pass" for record in records)
            else "fail"
        ),
        "records": records,
    }


def c3k2_dynamic_concat_structure_summary(
    model,
    configured_contracts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """按冻结合同验证独立 child-output 与 C3k2.cv2 动态 segment。

    结构正确性不能依赖物理构网期间写入的临时 Python 属性：这些属性可能在
    checkpoint 序列化、框架包装或版本迁移后缺失。优先使用 audit 冻结的合同，
    再直接读取物理模型的真实通道宽度。是否要求全部候选真实命中由后置门禁
    单独判断；本函数只判断已选/未选状态下的结构是否自洽。
    """

    named_modules = dict(model.named_modules())
    contracts = [
        dict(record)
        for record in (configured_contracts or [])
        if isinstance(record, dict)
        and record.get("contract")
        == "yolo11_c3k2_segmented_dynamic_concat_v1"
    ]

    # 兼容早期物理 checkpoint：若调用方没有提供 audit 合同，则仅从持久化的
    # physical segment 记录恢复最小合同。正式 prune/rebuild 路径始终传入合同。
    if not contracts:
        for module_prefix, module in named_modules.items():
            physical_segments = getattr(
                module,
                "_oto_c3k2_dynamic_physical_segments",
                {},
            )
            if not isinstance(physical_segments, dict):
                continue
            for child_prefix, segment in physical_segments.items():
                child = named_modules.get(str(child_prefix))
                if child is None or not isinstance(segment, dict):
                    continue
                output_path = getattr(
                    child,
                    "_oto_c3k2_dynamic_output_path",
                    None,
                )
                original_width = int(
                    getattr(child, "_oto_c3k2_dynamic_original_width", 0)
                )
                if not output_path or original_width <= 0:
                    continue
                contracts.append(
                    {
                        "contract": "yolo11_c3k2_segmented_dynamic_concat_v1",
                        "module_prefix": module_prefix,
                        "child_prefix": str(child_prefix),
                        "child_index": int(segment.get("child_index", -1)),
                        "output_path": str(output_path),
                        "segment_index": int(segment.get("segment_index", -1)),
                        "original_width": original_width,
                        "parent_root_width": int(
                            getattr(module, "_oto_c3k2_dynamic_original_c", 0)
                        ),
                        "parent_original_concat_width": int(
                            getattr(
                                module,
                                "_oto_c3k2_dynamic_original_cv2_input",
                                0,
                            )
                        ),
                    }
                )

    if not contracts:
        return {
            "status": "not_applicable",
            "configured_child_count": 0,
            "selected_child_count": 0,
            "configured_child_prefixes": [],
            "selected_child_prefixes": [],
            "records": [],
        }

    contracts_by_parent: dict[str, list[dict[str, Any]]] = {}
    for contract in contracts:
        contracts_by_parent.setdefault(
            str(contract.get("module_prefix", "")),
            [],
        ).append(contract)

    records: list[dict[str, Any]] = []
    configured_child_prefixes: list[str] = []
    selected_child_prefixes: list[str] = []
    for module_prefix, parent_contracts in sorted(contracts_by_parent.items()):
        module = named_modules.get(module_prefix)
        if module is None:
            records.append(
                {
                    "module_prefix": module_prefix,
                    "status": "fail_missing_module",
                    "checks": {"parent_module_exists": False},
                }
            )
            continue

        cv1 = getattr(module, "cv1", None)
        cv2 = getattr(module, "cv2", None)
        children = list(getattr(module, "m", ()))
        current_c = int(getattr(module, "c", 0))
        original_c = int(parent_contracts[0].get("parent_root_width", 0))
        original_cv2_input = int(
            parent_contracts[0].get("parent_original_concat_width", 0)
        )
        if current_c <= 0 or original_c <= 0 or original_cv2_input <= 0 or not children or not all(
            hasattr(wrapper, "conv") and hasattr(wrapper, "bn")
            for wrapper in (cv1, cv2)
        ):
            records.append(
                {
                    "module_prefix": module_prefix,
                    "status": "fail_missing_structure",
                    "current_c": current_c,
                    "original_c": original_c,
                    "original_cv2_in_channels": original_cv2_input,
                }
            )
            continue

        contract_by_child_index = {
            int(contract.get("child_index", -1)): contract
            for contract in parent_contracts
        }
        missing_child_indexes = sorted(
            index
            for index in contract_by_child_index
            if index < 0 or index >= len(children)
        )
        running_offset = 2 * current_c
        expected_removed_from_concat = 0
        child_records: list[dict[str, Any]] = []
        parent_selected_count = 0
        for child_index, child in enumerate(children):
            output_wrapper = _c3k2_child_output_wrapper(child)
            current_width = int(output_wrapper.conv.out_channels)
            current_offset = running_offset
            running_offset += current_width
            contract = contract_by_child_index.get(child_index)
            configured = contract is not None
            child_prefix = (
                str(contract.get("child_prefix"))
                if configured
                else f"{module_prefix}.m.{child_index}"
            )
            if configured:
                configured_child_prefixes.append(child_prefix)
                original_width = int(contract.get("original_width", 0))
                expected_segment_index = int(contract.get("segment_index", -1))
                selected = 0 < current_width < original_width
                if selected:
                    parent_selected_count += 1
                    selected_child_prefixes.append(child_prefix)
                expected_removed_from_concat += max(0, original_width - current_width)

                marker_value = getattr(
                    child,
                    "_oto_c3k2_dynamic_child_contract",
                    getattr(child, "_oto_c3k2_dynamic_concat_contract", None),
                )
                marker_present = marker_value is not None
                marker_consistent = True
                if marker_present:
                    marker_consistent = bool(
                        marker_value == "yolo11_c3k2_segmented_dynamic_concat_v1"
                        and int(
                            getattr(
                                child,
                                "_oto_c3k2_dynamic_original_width",
                                -1,
                            )
                        )
                        == original_width
                        and int(
                            getattr(
                                child,
                                "_oto_c3k2_dynamic_current_width",
                                -1,
                            )
                        )
                        == current_width
                        and int(
                            getattr(
                                child,
                                "_oto_c3k2_dynamic_segment_index",
                                -1,
                            )
                        )
                        == expected_segment_index
                    )
                checks = {
                    "width_within_contract": 0 < current_width <= original_width,
                    "output_bn": int(output_wrapper.bn.num_features)
                    == current_width,
                    "segment_index_matches_child_order": expected_segment_index
                    == 2 + child_index,
                    "optional_physical_marker_consistent": marker_consistent,
                }
            else:
                original_width = current_width
                selected = False
                marker_present = False
                checks = {
                    "unmanaged_width_positive": current_width > 0,
                    "output_bn": int(output_wrapper.bn.num_features)
                    == current_width,
                }
            child_records.append(
                {
                    "child_index": child_index,
                    "child_prefix": child_prefix,
                    "child_type": type(child).__name__,
                    "configured": configured,
                    "selected": selected,
                    "physical_marker_present": marker_present,
                    "status": "pass" if all(checks.values()) else "fail",
                    "original_width": original_width,
                    "current_width": current_width,
                    "current_offset": current_offset,
                    "checks": checks,
                }
            )

        contract_original_output = parent_contracts[0].get(
            "parent_original_output_dim"
        )
        stored_original_output = getattr(
            module,
            "_oto_c3k2_dynamic_original_output_dim",
            None,
        )
        original_output_dim = int(
            contract_original_output
            if contract_original_output is not None
            else stored_original_output
            if stored_original_output is not None
            else cv2.conv.out_channels
        )
        actual_concat_reduction = original_cv2_input - int(cv2.conv.in_channels)
        checks = {
            "root_axis_fixed": current_c == original_c,
            "cv1_output_fixed": int(cv1.conv.out_channels) == 2 * current_c,
            "cv1_bn_fixed": int(cv1.bn.num_features) == 2 * current_c,
            "dynamic_concat_input": int(cv2.conv.in_channels) == running_offset,
            "concat_reduction_matches_children": actual_concat_reduction
            == expected_removed_from_concat,
            "concat_input_not_expanded": 0 < int(cv2.conv.in_channels)
            <= original_cv2_input,
            "external_output_fixed": int(cv2.conv.out_channels)
            == original_output_dim,
            "external_output_bn": int(cv2.bn.num_features)
            == int(cv2.conv.out_channels),
            "all_configured_children_present": not missing_child_indexes,
            "all_children_valid": bool(child_records)
            and all(record.get("status") == "pass" for record in child_records),
        }
        records.append(
            {
                "module_prefix": module_prefix,
                "status": "pass" if all(checks.values()) else "fail",
                "contract": "yolo11_c3k2_segmented_dynamic_concat_v1",
                "original_c": original_c,
                "current_c": current_c,
                "configured_child_count": len(parent_contracts),
                "selected_child_count": parent_selected_count,
                "segment_count": 2 + len(children),
                "cv1_out_channels": int(cv1.conv.out_channels),
                "cv2_in_channels": int(cv2.conv.in_channels),
                "original_cv2_in_channels": original_cv2_input,
                "expected_removed_from_concat": expected_removed_from_concat,
                "actual_removed_from_concat": actual_concat_reduction,
                "cv2_out_channels": int(cv2.conv.out_channels),
                "original_output_dim": original_output_dim,
                "original_output_dim_source": (
                    "configured_contract"
                    if contract_original_output is not None
                    else "physical_marker"
                    if stored_original_output is not None
                    else "current_fixed_output"
                ),
                "missing_child_indexes": missing_child_indexes,
                "children": child_records,
                "physical_segments": dict(
                    getattr(module, "_oto_c3k2_dynamic_physical_segments", {})
                ),
                "checks": checks,
            }
        )

    configured_child_prefixes = sorted(set(configured_child_prefixes))
    selected_child_prefixes = sorted(set(selected_child_prefixes))
    return {
        "status": (
            "pass"
            if records and all(record.get("status") == "pass" for record in records)
            else "fail"
        ),
        "configured_child_count": len(configured_child_prefixes),
        "selected_child_count": len(selected_child_prefixes),
        "configured_child_prefixes": configured_child_prefixes,
        "selected_child_prefixes": selected_child_prefixes,
        "records": records,
    }


def output_consumer_closure_structure_summary(
    model,
    configured_contracts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """验证生产者输出与所有已识别消费者输入的物理闭包。"""

    all_contracts = [
        dict(record)
        for record in (configured_contracts or [])
        if isinstance(record, dict)
    ]
    if not all_contracts:
        return {"status": "not_applicable", "records": []}
    contracts = [
        record
        for record in all_contracts
        if record.get("status") == "configured_resolved_consumer_closure"
    ]
    quarantined_contracts = [
        record
        for record in all_contracts
        if record.get("status") != "configured_resolved_consumer_closure"
    ]
    if not contracts:
        return {
            "status": "fail",
            "configured_contract_count": 0,
            "quarantined_contract_count": len(quarantined_contracts),
            "quarantined_contracts": quarantined_contracts,
            "records": [],
        }

    named_modules = dict(model.named_modules())
    records: list[dict[str, Any]] = []
    for contract in contracts:
        producer_prefix = str(contract.get("producer_wrapper_prefix", ""))
        producer = named_modules.get(producer_prefix)
        conv = getattr(producer, "conv", None)
        bn = getattr(producer, "bn", None)
        original_output_dim = int(contract.get("original_output_dim") or 0)
        consumer_contracts = {
            str(record.get("consumer_parameter_name")): dict(record)
            for record in contract.get("consumer_records", [])
            if isinstance(record, dict) and record.get("consumer_parameter_name")
        }
        segmented_contracts = {
            str(record.get("contract_id")): dict(record)
            for record in contract.get("segmented_concat_contracts", [])
            if isinstance(record, dict) and record.get("contract_id")
        }
        consumer_records = []
        for parameter_name in contract.get("consumer_input_parameter_names", []):
            module_prefix = str(parameter_name).rsplit(".weight", 1)[0]
            consumer = named_modules.get(module_prefix)
            is_conv = isinstance(consumer, torch.nn.Conv2d)
            consumer_contract = consumer_contracts.get(str(parameter_name), {})
            consumer_kind = consumer_contract.get("consumer_kind")
            standard_matches = (
                bool(isinstance(conv, torch.nn.Conv2d))
                and is_conv
                and int(consumer.groups) == 1
                and int(consumer.in_channels) == int(conv.out_channels)
            )
            depthwise_matches = (
                bool(isinstance(conv, torch.nn.Conv2d))
                and is_conv
                and int(consumer.groups) == int(consumer.in_channels)
                == int(consumer.out_channels)
                == int(conv.out_channels)
                and int(consumer.weight.shape[0]) == int(consumer.out_channels)
                and int(consumer.weight.shape[1]) == 1
            )
            segmented_contract = segmented_contracts.get(
                str(consumer_contract.get("concat_contract_id", ""))
            )
            segmented_entries = (
                [
                    dict(segment)
                    for segment in segmented_contract.get("segments", [])
                    if isinstance(segment, dict)
                ]
                if isinstance(segmented_contract, dict)
                else []
            )
            segmented_current_widths = []
            segmented_sources_valid = bool(segmented_entries)
            for segment in segmented_entries:
                segment_prefix = str(
                    segment.get("producer_parameter_name", "")
                ).rsplit(".weight", 1)[0]
                segment_producer = named_modules.get(segment_prefix)
                if not isinstance(segment_producer, torch.nn.Conv2d):
                    segmented_sources_valid = False
                    segmented_current_widths.append(None)
                else:
                    segmented_current_widths.append(
                        int(segment_producer.out_channels)
                    )
            segmented_expected_input = (
                sum(width for width in segmented_current_widths if width is not None)
                if segmented_sources_valid
                else None
            )
            segmented_matches = (
                is_conv
                and int(consumer.groups) == 1
                and segmented_sources_valid
                and segmented_expected_input is not None
                and int(consumer.in_channels) == int(segmented_expected_input)
                and int(consumer.weight.shape[1]) == int(segmented_expected_input)
            )
            consumer_records.append(
                {
                    "parameter_name": parameter_name,
                    "module_prefix": module_prefix,
                    "module_type": type(consumer).__name__ if consumer is not None else None,
                    "consumer_kind": consumer_kind,
                    "in_channels": int(consumer.in_channels) if is_conv else None,
                    "out_channels": int(consumer.out_channels) if is_conv else None,
                    "groups": int(consumer.groups) if is_conv else None,
                    "concat_contract_id": consumer_contract.get("concat_contract_id"),
                    "concat_expected_in_channels": segmented_expected_input,
                    "concat_segment_current_widths": segmented_current_widths,
                    "matches_producer_output": (
                        standard_matches
                        if consumer_kind == "standard_conv_input_axis1"
                        else depthwise_matches
                        if consumer_kind == "depthwise_coupled_axis0"
                        else segmented_matches
                        if consumer_kind
                        == "segmented_concat_standard_conv_input_axis1"
                        else False
                    ),
                }
            )
        checks = {
            "producer_conv": isinstance(conv, torch.nn.Conv2d),
            "producer_bn": isinstance(bn, torch.nn.BatchNorm2d),
            "output_width_valid": (
                isinstance(conv, torch.nn.Conv2d)
                and 0 < int(conv.out_channels) <= original_output_dim
            ),
            "producer_bn_matches": (
                isinstance(conv, torch.nn.Conv2d)
                and isinstance(bn, torch.nn.BatchNorm2d)
                and int(bn.num_features) == int(conv.out_channels)
            ),
            "has_consumers": bool(consumer_records),
            "all_consumers_are_supported_conv": bool(consumer_records)
            and all(
                record["module_type"] == "Conv2d"
                and record["consumer_kind"]
                in {
                    "standard_conv_input_axis1",
                    "depthwise_coupled_axis0",
                    "segmented_concat_standard_conv_input_axis1",
                }
                for record in consumer_records
            ),
            "all_consumer_inputs_match": bool(consumer_records)
            and all(record["matches_producer_output"] for record in consumer_records),
        }
        current_output_dim = int(conv.out_channels) if isinstance(conv, torch.nn.Conv2d) else None
        records.append(
            {
                "contract": contract.get("contract"),
                "module_prefix": contract.get("module_prefix"),
                "producer_wrapper_prefix": producer_prefix,
                "node_group_id": contract.get("node_group_id"),
                "original_output_dim": original_output_dim,
                "current_output_dim": current_output_dim,
                "physical_output_reduced": (
                    current_output_dim is not None and current_output_dim < original_output_dim
                ),
                "consumers": consumer_records,
                "checks": checks,
                "status": "pass" if all(checks.values()) else "fail",
            }
        )
    return {
        "status": "pass" if all(record["status"] == "pass" for record in records) else "fail",
        "contract_count": len(records),
        "configured_contract_count": len(contracts),
        "quarantined_contract_count": len(quarantined_contracts),
        "quarantined_contracts": quarantined_contracts,
        "selected_record_count": sum(
            bool(record["physical_output_reduced"]) for record in records
        ),
        "records": records,
    }


def c3k2_composite_structure_summary(
    model,
    configured_contracts: list[dict[str, Any]] | None = None,
    output_consumer_closure_structure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """验证 C3k2 root paired-chunk 与 child dynamic-concat 的联合物理合同。

    该检查只以冻结的配置合同和当前模型真实通道宽度为准，不能依赖构网阶段的
    临时节点状态。它特别验证 ``cv2`` 的输入列只按“原始坐标一次性”删除：根部
    轴作用于两个 chunk 与 Bottleneck child，独立轴仅作用于 C3/C3k child。
    """

    named_modules = dict(model.named_modules())
    contracts = [
        dict(record)
        for record in (configured_contracts or [])
        if isinstance(record, dict)
        and record.get("contract")
        == "yolo11_c3k2_paired_chunk_dynamic_concat_v1"
    ]
    if not contracts:
        return {
            "status": "not_applicable",
            "configured_child_count": 0,
            "selected_child_count": 0,
            "configured_child_prefixes": [],
            "selected_child_prefixes": [],
            "records": [],
        }

    # 联合 v4 允许 C3k2.cv2 的**输出轴**由通用 output-consumer closure
    # 裁剪；root paired-chunk + dynamic child 仍只拥有同一个 cv2 的**输入轴**。
    # 因此不能继续无条件要求 cv2.out_channels 保持原宽度，也不能仅删除该
    # 检查。只有已经由 closure linter 证明“生产者 BN 与全部消费者输入均闭合”
    # 的输出轴才允许缩窄；未登记/失败/语义不明的旧 profile 仍严格固定输出。
    closure_records = {
        str(record.get("module_prefix")): dict(record)
        for record in ((output_consumer_closure_structure or {}).get("records", []))
        if isinstance(record, dict) and record.get("module_prefix")
    }

    records: list[dict[str, Any]] = []
    configured_child_prefixes: list[str] = []
    selected_child_prefixes: list[str] = []
    for contract in sorted(contracts, key=lambda item: str(item.get("module_prefix", ""))):
        module_prefix = str(contract.get("module_prefix", ""))
        parent = named_modules.get(module_prefix)
        root_original_c = int(contract.get("root_logical_group_count", 0))
        dynamic_contracts = [
            dict(item)
            for item in contract.get("dynamic_child_contracts", [])
            if isinstance(item, dict)
        ]
        dynamic_by_child = {
            str(item.get("child_prefix", "")): item for item in dynamic_contracts
        }
        configured_child_prefixes.extend(dynamic_by_child)
        if parent is None:
            records.append(
                {
                    "module_prefix": module_prefix,
                    "status": "fail_missing_module",
                    "checks": {"parent_module_exists": False},
                }
            )
            continue

        cv1 = getattr(parent, "cv1", None)
        cv2 = getattr(parent, "cv2", None)
        children = list(getattr(parent, "m", ()))
        current_c = int(getattr(parent, "c", 0))
        if (
            root_original_c <= 0
            or current_c <= 0
            or not children
            or not all(
                hasattr(wrapper, "conv") and hasattr(wrapper, "bn")
                for wrapper in (cv1, cv2)
            )
        ):
            records.append(
                {
                    "module_prefix": module_prefix,
                    "status": "fail_missing_structure",
                    "root_original_c": root_original_c,
                    "current_c": current_c,
                }
            )
            continue

        root_removed = root_original_c - current_c
        child_records: list[dict[str, Any]] = []
        expected_removed_from_concat = 2 * root_removed
        running_concat_width = 2 * current_c
        child_prefixes_seen: set[str] = set()
        for child_index, child in enumerate(children):
            child_prefix = f"{module_prefix}.m.{child_index}"
            child_prefixes_seen.add(child_prefix)
            child_type = type(child).__name__
            output_wrapper = _c3k2_child_output_wrapper(child)
            current_width = int(output_wrapper.conv.out_channels)
            dynamic_contract = dynamic_by_child.get(child_prefix)
            if child_type == "Bottleneck":
                checks = {
                    "input_width": int(child.cv1.conv.in_channels) == current_c,
                    "output_width_matches_root": current_width == current_c,
                    "output_bn": int(output_wrapper.bn.num_features) == current_width,
                    "not_independently_owned": dynamic_contract is None,
                }
                original_width = root_original_c
                owner = "root_paired_chunk_owner"
                expected_removed_from_concat += root_removed
                selected = root_removed > 0
            elif child_type in {"C3", "C3k"}:
                original_width = int(
                    dynamic_contract.get("original_width", 0)
                    if dynamic_contract is not None
                    else 0
                )
                dynamic_removed = original_width - current_width
                checks = {
                    "branch1_input_width": int(child.cv1.conv.in_channels) == current_c,
                    "branch2_input_width": int(child.cv2.conv.in_channels) == current_c,
                    "has_independent_contract": dynamic_contract is not None,
                    "width_within_contract": 0 < current_width <= original_width,
                    "output_bn": int(output_wrapper.bn.num_features) == current_width,
                }
                owner = "dynamic_child_output_owner"
                expected_removed_from_concat += dynamic_removed
                selected = dynamic_removed > 0
                if selected:
                    selected_child_prefixes.append(child_prefix)
            else:
                child_records.append(
                    {
                        "child_index": child_index,
                        "child_prefix": child_prefix,
                        "child_type": child_type,
                        "status": "fail_unsupported_child",
                    }
                )
                continue
            running_concat_width += current_width
            child_records.append(
                {
                    "child_index": child_index,
                    "child_prefix": child_prefix,
                    "child_type": child_type,
                    "owner": owner,
                    "selected": selected,
                    "original_width": original_width,
                    "current_width": current_width,
                    "status": "pass" if all(checks.values()) else "fail",
                    "checks": checks,
                }
            )

        missing_dynamic_children = sorted(set(dynamic_by_child) - child_prefixes_seen)
        original_cv2_input = (2 + len(children)) * root_original_c
        actual_removed_from_concat = original_cv2_input - int(cv2.conv.in_channels)
        original_output_dim = int(
            getattr(parent, "_oto_c3k2_original_output_dim", cv2.conv.out_channels)
        )
        closure_record = closure_records.get(module_prefix)
        external_output_fixed = int(cv2.conv.out_channels) == original_output_dim
        if closure_record is None:
            external_output_policy = "fixed_output_required"
            external_output_contract_valid = external_output_fixed
        else:
            closure_current_output = closure_record.get("current_output_dim")
            external_output_policy = "output_consumer_closure"
            external_output_contract_valid = bool(
                closure_record.get("status") == "pass"
                and closure_record.get("producer_wrapper_prefix")
                == f"{module_prefix}.cv2"
                and closure_current_output is not None
                and int(closure_current_output) == int(cv2.conv.out_channels)
                and 0 < int(cv2.conv.out_channels) <= original_output_dim
            )
        checks = {
            "root_width_within_contract": 0 < current_c <= root_original_c,
            "root_cv1_paired_output": int(cv1.conv.out_channels) == 2 * current_c,
            "root_cv1_paired_bn": int(cv1.bn.num_features) == 2 * current_c,
            "cv2_dynamic_concat_input": int(cv2.conv.in_channels) == running_concat_width,
            "concat_reduction_matches_all_owners": actual_removed_from_concat
            == expected_removed_from_concat,
            "cv2_input_not_expanded": 0 < int(cv2.conv.in_channels) <= original_cv2_input,
            "external_output_contract_valid": external_output_contract_valid,
            "external_output_bn": int(cv2.bn.num_features) == int(cv2.conv.out_channels),
            "all_configured_children_present": not missing_dynamic_children,
            "all_children_valid": bool(child_records)
            and all(record.get("status") == "pass" for record in child_records),
        }
        records.append(
            {
                "module_prefix": module_prefix,
                "status": "pass" if all(checks.values()) else "fail",
                "contract": "yolo11_c3k2_paired_chunk_dynamic_concat_v1",
                "root_original_c": root_original_c,
                "current_c": current_c,
                "root_removed_group_count": root_removed,
                "configured_child_count": len(dynamic_by_child),
                "selected_child_count": sum(
                    record.get("owner") == "dynamic_child_output_owner"
                    and bool(record.get("selected"))
                    for record in child_records
                ),
                "cv1_out_channels": int(cv1.conv.out_channels),
                "cv2_in_channels": int(cv2.conv.in_channels),
                "original_cv2_in_channels": original_cv2_input,
                "expected_removed_from_concat": expected_removed_from_concat,
                "actual_removed_from_concat": actual_removed_from_concat,
                "cv2_out_channels": int(cv2.conv.out_channels),
                "original_output_dim": original_output_dim,
                "external_output_fixed": external_output_fixed,
                "external_output_policy": external_output_policy,
                "output_consumer_closure_record": closure_record,
                "missing_dynamic_children": missing_dynamic_children,
                "children": child_records,
                "checks": checks,
            }
        )

    configured_child_prefixes = sorted(set(configured_child_prefixes))
    selected_child_prefixes = sorted(set(selected_child_prefixes))
    return {
        "status": (
            "pass"
            if records and all(record.get("status") == "pass" for record in records)
            else "fail"
        ),
        "configured_child_count": len(configured_child_prefixes),
        "selected_child_count": len(selected_child_prefixes),
        "configured_child_prefixes": configured_child_prefixes,
        "selected_child_prefixes": selected_child_prefixes,
        "records": records,
    }


def yolo11_attention_structure_summary(model) -> dict[str, Any]:
    """审计物理子网 Attention 的内部/外部宽度与持久化 forward 合同。"""
    records: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not getattr(module, "_oto_dynamic_inner_attention", False):
            continue
        qkv_conv = getattr(getattr(module, "qkv", None), "conv", None)
        pe_conv = getattr(getattr(module, "pe", None), "conv", None)
        proj_conv = getattr(getattr(module, "proj", None), "conv", None)
        if not all(isinstance(conv, torch.nn.Conv2d) for conv in (qkv_conv, pe_conv, proj_conv)):
            records.append(
                {
                    "module_prefix": name,
                    "status": "fail_missing_conv_structure",
                    "module_type": type(module).__name__,
                }
            )
            continue
        num_heads = int(module.num_heads)
        key_dim = int(module.key_dim)
        head_dim = int(module.head_dim)
        prune_mode = yolo11_attention_prune_mode(module)
        original_num_heads_value = getattr(
            module, "_oto_attention_original_num_heads", None
        )
        original_head_dim_value = getattr(
            module, "_oto_attention_original_head_dim", None
        )
        original_key_dim_value = getattr(
            module, "_oto_attention_original_key_dim", None
        )
        has_axis_provenance = all(
            value is not None
            for value in (
                original_num_heads_value,
                original_head_dim_value,
                original_key_dim_value,
            )
        )
        original_num_heads = (
            int(original_num_heads_value) if original_num_heads_value is not None else None
        )
        original_head_dim = (
            int(original_head_dim_value) if original_head_dim_value is not None else None
        )
        original_key_dim = (
            int(original_key_dim_value) if original_key_dim_value is not None else None
        )
        if not has_axis_provenance:
            mode_specific_axis_check = True
            key_dim_fixed_check = True
            axis_provenance_status = "legacy_not_recorded"
        elif prune_mode == "num_head":
            mode_specific_axis_check = (
                num_heads < original_num_heads
                and head_dim == original_head_dim
            )
            key_dim_fixed_check = key_dim == original_key_dim
            axis_provenance_status = "verified"
        elif prune_mode == "head_dim":
            mode_specific_axis_check = (
                num_heads == original_num_heads
                and head_dim < original_head_dim
            )
            key_dim_fixed_check = key_dim == original_key_dim
            axis_provenance_status = "verified"
        elif prune_mode == "external_width":
            mode_specific_axis_check = (
                num_heads == original_num_heads
                and head_dim == original_head_dim
            )
            key_dim_fixed_check = key_dim == original_key_dim
            axis_provenance_status = "verified_external_width"
        else:
            mode_specific_axis_check = False
            key_dim_fixed_check = False
            axis_provenance_status = "invalid_mode"
        inner_dim = num_heads * head_dim
        expected_qkv = num_heads * (key_dim * 2 + head_dim)
        original_output_dim = int(
            getattr(module, "_oto_attention_original_output_dim", proj_conv.out_channels)
        )
        external_width_reduced = (
            prune_mode == "external_width"
            and 0 < int(proj_conv.out_channels) < original_output_dim
        )
        checks = {
            "qkv_width": qkv_conv.out_channels == expected_qkv,
            "qkv_external_input": qkv_conv.in_channels == proj_conv.out_channels,
            "pe_depthwise_width": (
                pe_conv.in_channels == inner_dim
                and pe_conv.out_channels == inner_dim
                and pe_conv.groups == inner_dim
            ),
            "proj_internal_input": proj_conv.in_channels == inner_dim,
            "proj_external_output_fixed": (
                proj_conv.out_channels == original_output_dim
                if prune_mode != "external_width"
                else True
            ),
            "proj_external_width_reduced": (
                external_width_reduced
                if prune_mode == "external_width"
                else True
            ),
            "stored_inner_dim": int(
                getattr(module, "_oto_attention_inner_dim", -1)
            ) == inner_dim,
            "persistent_forward_class": type(module).__name__ == "OTOPrunableYOLO11Attention",
            "contract_marker": getattr(module, "_oto_attention_contract", None)
            == "yolo11_partitioned_qkv_v1",
            "mode_specific_axis_change": mode_specific_axis_check,
            "key_dim_fixed": key_dim_fixed_check,
        }
        records.append(
            {
                "module_prefix": name,
                "status": "pass" if all(checks.values()) else "fail",
                "module_type": type(module).__name__,
                "prune_mode": prune_mode,
                "num_heads": num_heads,
                "key_dim": key_dim,
                "head_dim": head_dim,
                "original_num_heads": original_num_heads,
                "original_key_dim": original_key_dim,
                "original_head_dim": original_head_dim,
                "axis_provenance_status": axis_provenance_status,
                "inner_dim": inner_dim,
                "qkv_out_channels": int(qkv_conv.out_channels),
                "pe_in_channels": int(pe_conv.in_channels),
                "pe_out_channels": int(pe_conv.out_channels),
                "pe_groups": int(pe_conv.groups),
                "proj_in_channels": int(proj_conv.in_channels),
                "proj_out_channels": int(proj_conv.out_channels),
                "original_output_dim": original_output_dim,
                "checks": checks,
            }
        )
    return {
        "status": (
            "not_applicable"
            if not records
            else "pass"
            if all(record.get("status") == "pass" for record in records)
            else "fail"
        ),
        "records": records,
    }


def yolo11_attention_redundancy_contract(
    oto,
    *,
    require_physical_redundancy: bool = True,
) -> dict[str, Any]:
    """核对 Attention 的优化器所有权与物理冗余判定是否一致。

    Attention-only probe 必须真实产生冗余 head/维度，因此默认要求
    ``physical_redundancy_detected``。联合全局剪枝只要求 Attention 正确进入
    HESSO 候选池；全局排序可能不选择它，此时不能把“未命中”误判为参数所有权
    错误。调用侧通过 ``require_physical_redundancy=False`` 区分这两种语义。
    """

    optimizer_groups = {
        str(group.get("id")): group for group in oto._graph.get_param_groups()
    }
    oto._graph.set_pruning_redundant_idxes()
    records: list[dict[str, Any]] = []
    for node_group in oto._graph.node_groups.values():
        op = getattr(node_group, "op", None)
        if getattr(op, "op_name", None) != "yolo11_attention":
            continue
        redundancy_group = node_group.get_param_groups()
        optimizer_group = optimizer_groups.get(str(node_group.id), {})
        optimizer_names = list(optimizer_group.get("p_names", []))
        redundancy_names = list(redundancy_group.get("p_names", []))
        redundant_idxes = [
            int(index)
            for index in getattr(node_group, "pruning_redundant_idxes", [])
        ]
        ownership_checks = {
            "composed_node_group": type(node_group).__name__ == "NodeGroupComposedOp",
            "group_is_prunable": bool(node_group.is_prunable),
            "optimizer_and_redundancy_parameter_ownership_match": (
                sorted(optimizer_names) == sorted(redundancy_names)
            ),
            "proj_excluded_from_redundancy_driver": not any(
                name.endswith(".proj.conv.weight") for name in redundancy_names
            ),
        }
        physical_redundancy_detected = bool(redundant_idxes)
        required_checks = dict(ownership_checks)
        if require_physical_redundancy:
            required_checks["physical_redundancy_detected"] = physical_redundancy_detected
        checks = {
            **ownership_checks,
            "physical_redundancy_detected": physical_redundancy_detected,
        }
        records.append(
            {
                "node_group_id": str(node_group.id),
                "status": "pass" if all(required_checks.values()) else "fail",
                "ownership_status": "pass" if all(ownership_checks.values()) else "fail",
                "physical_redundancy_status": (
                    "pass" if physical_redundancy_detected else "not_selected"
                ),
                "prune_mode": getattr(op, "prune_mode", None),
                "num_groups": int(node_group.num_groups),
                "optimizer_parameter_names": optimizer_names,
                "redundancy_parameter_names": redundancy_names,
                "pruning_redundant_idxes": redundant_idxes,
                "checks": checks,
                "required_checks": sorted(required_checks),
            }
        )

    ownership_status = (
        "not_applicable"
        if not records
        else "pass"
        if all(record["ownership_status"] == "pass" for record in records)
        else "fail"
    )
    selected_record_count = sum(
        record["physical_redundancy_status"] == "pass" for record in records
    )
    physical_redundancy_status = (
        "not_applicable"
        if not records
        else "not_selected"
        if selected_record_count == 0
        else "pass"
        if selected_record_count == len(records)
        else "partial"
    )
    status = (
        "not_applicable"
        if not records
        else "pass"
        if ownership_status == "pass"
        and (not require_physical_redundancy or physical_redundancy_status == "pass")
        else "fail"
    )
    return {
        "status": status,
        "ownership_status": ownership_status,
        "physical_redundancy_status": physical_redundancy_status,
        "selected_record_count": int(selected_record_count),
        "attention_record_count": len(records),
        "require_physical_redundancy": bool(require_physical_redundancy),
        "records": records,
    }


def finalize_structured_subnet(trainer, args: argparse.Namespace) -> dict[str, Any]:
    """物理删除 HESSO 已置零的通道，并保存可被 Ultralytics 重新加载的 checkpoint。"""

    if trainer.oto is None:
        raise RuntimeError("训练器未建立 OTO 图，拒绝构建结构化子网。")

    run_dir = Path(trainer.save_dir)
    oto_dir = run_dir / "oto_artifacts"
    raw_model = unwrap_model(trainer.model).float().eval()
    profile = trainer.adamcf_config.module_coverage_profile
    require_attention_hit = profile in ATTENTION_ONLY_PROFILES
    attention_redundancy = yolo11_attention_redundancy_contract(
        trainer.oto,
        require_physical_redundancy=require_attention_hit,
    )
    if (
        profile in ATTENTION_PROFILE_MODES
        and attention_redundancy["status"] != "pass"
    ):
        diagnostic_path = run_dir / "attention_redundancy_contract_failure.json"
        json_dump(diagnostic_path, attention_redundancy)
        raise RuntimeError(
            "Attention 参数所有权或冗余合同失败；"
            f"详细诊断：{diagnostic_path}"
        )
    attention_was_selected = attention_redundancy.get(
        "physical_redundancy_status"
    ) in {"pass", "partial"}
    trainer.oto.construct_subnet(out_dir=str(oto_dir), ckpt_format="torch")

    compressed_raw_path = Path(trainer.oto.compressed_model_path)
    if not compressed_raw_path.is_file():
        raise FileNotFoundError(f"OTO 未产生压缩模型：{compressed_raw_path}")
    structured_model = torch.load(compressed_raw_path, map_location="cpu", weights_only=False).float().eval()

    standard_path = run_dir / "weights" / args.structured_name
    standard_ultralytics_checkpoint(structured_model, trainer, standard_path, ultralytics.__version__, torch)

    # 最小烟雾验证：加载标准 checkpoint 后必须能在固定 shape 上前向。
    reloaded = YOLO(str(standard_path))
    with torch.no_grad():
        output = reloaded.model.float().eval()(torch.zeros(1, 3, args.imgsz, args.imgsz))
    compressed_stats = model_stats(structured_model, args.imgsz, torch.device("cpu"))
    redundancy = physical_redundancy_summary(
        trainer.oto,
        raw_model,
        trainer.module_pruning_candidate_prefixes,
    )
    attention_structure = yolo11_attention_structure_summary(structured_model)
    c2psa_split_structure = c2psa_paired_split_structure_summary(structured_model)
    dependency_control = trainer.module_dependency_control or {}
    output_consumer_closure_structure = output_consumer_closure_structure_summary(
        structured_model,
        configured_contracts=dependency_control.get(
            "yolo11_output_consumer_closure_contracts", []
        ),
    )
    if profile in C3K2_COMPOSITE_PROFILES:
        # 根部和 child 在该 profile 共同修改同一个 cv2 输入轴，只能由联合
        # linter 判断。未参与复合的 C3k2 root 仍可按 PairedChunk 单轴合同
        # 构网，因此保留普通 paired linter；该 linter 会跳过已经标记为 composite
        # 的父模块，避免把独立 child 输出误判为 root 同步输出。
        c3k2_chunk_structure = c3k2_paired_chunk_structure_summary(structured_model)
        c3k2_dynamic_structure = {"status": "not_applicable", "records": []}
        c3k2_composite_structure = c3k2_composite_structure_summary(
            structured_model,
            configured_contracts=dependency_control.get(
                "yolo11_c3k2_composite_contracts", []
            ),
            output_consumer_closure_structure=output_consumer_closure_structure,
        )
    else:
        c3k2_chunk_structure = c3k2_paired_chunk_structure_summary(structured_model)
        c3k2_dynamic_structure = c3k2_dynamic_concat_structure_summary(
            structured_model,
            configured_contracts=dependency_control.get(
                "yolo11_c3k2_dynamic_concat_contracts", []
            ),
        )
        c3k2_composite_structure = {"status": "not_applicable", "records": []}
    if (
        profile in C2PSA_EXTERNAL_WIDTH_PROFILES
        and c2psa_split_structure.get("status") != "pass"
    ):
        diagnostic_path = run_dir / "c2psa_paired_split_structure_failure.json"
        json_dump(diagnostic_path, c2psa_split_structure)
        raise RuntimeError(
            "C2PSA paired-split 物理结构合同失败；"
            f"详细诊断：{diagnostic_path}"
        )
    if (
        profile in C3K2_PAIRED_CHUNK_PROFILES
        and (
            (
                profile in C3K2_COMPOSITE_PROFILES
                and c3k2_chunk_structure.get("status") not in {"not_applicable", "pass"}
            )
            or (
                profile not in C3K2_COMPOSITE_PROFILES
                and c3k2_chunk_structure.get("status") != "pass"
            )
        )
    ):
        diagnostic_path = run_dir / "c3k2_paired_chunk_structure_failure.json"
        json_dump(diagnostic_path, c3k2_chunk_structure)
        raise RuntimeError(
            "C3k2 paired-chunk 物理结构合同失败；"
            f"详细诊断：{diagnostic_path}"
        )
    if (
        profile in C3K2_DYNAMIC_CONCAT_PROFILES
        and profile not in C3K2_COMPOSITE_PROFILES
        and c3k2_dynamic_structure.get("status") != "pass"
    ):
        diagnostic_path = run_dir / "c3k2_dynamic_concat_structure_failure.json"
        json_dump(diagnostic_path, c3k2_dynamic_structure)
        raise RuntimeError(
            "C3k2 segmented dynamic-concat 物理结构合同失败；"
            f"详细诊断：{diagnostic_path}"
        )
    if (
        profile in C3K2_COMPOSITE_PROFILES
        and c3k2_composite_structure.get("status") != "pass"
    ):
        diagnostic_path = run_dir / "c3k2_composite_structure_failure.json"
        json_dump(diagnostic_path, c3k2_composite_structure)
        raise RuntimeError(
            "C3k2 paired-chunk 与 dynamic-concat 复合物理结构合同失败；"
            f"详细诊断：{diagnostic_path}"
        )
    if (
        profile in OUTPUT_CONSUMER_CLOSURE_PROFILES
        and output_consumer_closure_structure.get("status") != "pass"
    ):
        diagnostic_path = run_dir / "output_consumer_closure_structure_failure.json"
        json_dump(diagnostic_path, output_consumer_closure_structure)
        raise RuntimeError(
            "生产者输出—消费者输入物理闭包合同失败；"
            f"详细诊断：{diagnostic_path}"
        )
    attention_structure_status = attention_structure.get("status")
    if profile not in ATTENTION_PROFILE_MODES:
        attention_structure_valid = True
    elif attention_was_selected or require_attention_hit:
        attention_structure_valid = attention_structure_status == "pass"
    else:
        attention_structure_valid = attention_structure_status in {
            "not_applicable",
            "pass",
        }
    if not attention_structure_valid:
        diagnostic = {
            "profile": profile,
            "attention_was_selected": attention_was_selected,
            "structure": attention_structure,
        }
        diagnostic_path = run_dir / "attention_structure_contract_failure.json"
        json_dump(diagnostic_path, diagnostic)
        raise RuntimeError(
            "Attention 物理结构合同失败；"
            f"详细诊断：{diagnostic_path}"
        )
    metadata = {
        "status": "structured_subnet_built",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "sparse_training_run": str(run_dir),
        "oto_full_group_sparse_model": str(trainer.oto.full_group_sparse_model_path),
        "oto_compressed_raw_model": str(compressed_raw_path),
        "structured_ultralytics_checkpoint": str(standard_path),
        "structured_checkpoint_sha256": sha256_file(standard_path),
        "structured_model_stats": compressed_stats,
        "physical_redundancy": redundancy,
        "yolo11_attention_redundancy_contract": attention_redundancy,
        "yolo11_attention_structure": attention_structure,
        "yolo11_c2psa_paired_split_structure": c2psa_split_structure,
        "yolo11_c3k2_paired_chunk_structure": c3k2_chunk_structure,
        "yolo11_c3k2_dynamic_concat_structure": c3k2_dynamic_structure,
        "yolo11_c3k2_composite_structure": c3k2_composite_structure,
        "yolo11_output_consumer_closure_structure": output_consumer_closure_structure,
        "reloaded_output_shapes": serialize_output_shape(output),
        "alpha_update_count": trainer.alpha_update_count,
        "framework_versions": framework_version_summary(),
        "module_coverage_profile": trainer.adamcf_config.module_coverage_profile,
        "module_probe_only": trainer.adamcf_config.module_probe_only,
        "module_pruning_candidate_prefixes": trainer.module_pruning_candidate_prefixes,
        "module_pruning_candidate_groups": trainer.module_pruning_candidate_groups,
        "module_dependency_control": trainer.module_dependency_control,
        "oto_graph_artifacts": trainer.oto_graph_artifacts,
        "protected_module_prefixes": trainer.protected_module_prefixes,
        "oto_tensor_unsafe_exclusions": trainer.tensor_unsafe_exclusions,
        "warning": (
            "此处完成物理子网构建和固定 shape 前向验证；主流程随后会独立执行原模型与"
            "结构化子网 COCO val。仍需继续执行 finetune、ONNX 导出与端侧 benchmark。"
        ),
    }
    json_dump(run_dir / "structured_subnet.json", metadata)
    return metadata


def physical_redundancy_summary(
    oto,
    model=None,
    candidate_prefixes: list[str] | None = None,
) -> dict[str, Any]:
    """汇总实际物理零 group，并区分候选合同是否被 HESSO 真实命中。"""

    total_groups = 0
    redundant_groups = 0
    groups_with_redundancy: list[dict[str, Any]] = []
    candidate_prefixes = list(candidate_prefixes or [])
    unwrapped_model = unwrap_model(model) if model is not None else None
    family_hit_summary: dict[str, dict[str, int]] = {}
    for node_group in oto._graph.node_groups.values():
        if not node_group.is_prunable or node_group.is_auxiliary:
            continue
        num_groups = int(node_group.num_groups)
        redundant_idxes = [int(index) for index in getattr(node_group, "pruning_redundant_idxes", [])]
        total_groups += num_groups
        redundant_groups += len(redundant_idxes)
        if redundant_idxes:
            parameter_names = sorted(node_group.param_names)
            matched_candidate_prefixes = sorted(
                prefix
                for prefix in candidate_prefixes
                if any(name == prefix or name.startswith(f"{prefix}.") for name in parameter_names)
            )
            ancestor_module_families: set[str] = set()
            if unwrapped_model is not None:
                for prefix in matched_candidate_prefixes:
                    ancestor_module_families.update(
                        module_prefix_ancestor_families(unwrapped_model, prefix)
                    )
            if matched_candidate_prefixes and not ancestor_module_families:
                ancestor_module_families.add("standalone_or_boundary_Conv2d")
            for family in ancestor_module_families:
                counters = family_hit_summary.setdefault(
                    family,
                    {"dependency_group_count": 0, "redundant_channel_group_count": 0},
                )
                counters["dependency_group_count"] += 1
                counters["redundant_channel_group_count"] += len(redundant_idxes)
            groups_with_redundancy.append(
                {
                    "node_group_id": node_group.id,
                    "num_groups": num_groups,
                    "redundant_group_count": len(redundant_idxes),
                    "redundant_idxes": redundant_idxes,
                    "parameter_names": parameter_names,
                    "matched_candidate_prefixes": matched_candidate_prefixes,
                    "ancestor_module_families": sorted(ancestor_module_families),
                    "newly_opened_contract_hit": bool(matched_candidate_prefixes),
                }
            )
    actual_candidate_hit_groups = [
        record for record in groups_with_redundancy if record["newly_opened_contract_hit"]
    ]
    return {
        "eligible_group_count": total_groups,
        "physically_redundant_group_count": redundant_groups,
        "physical_group_sparsity": redundant_groups / max(1, total_groups),
        "groups_with_redundancy": groups_with_redundancy,
        "actual_candidate_hit_dependency_group_count": len(actual_candidate_hit_groups),
        "actual_candidate_hit_redundant_channel_group_count": sum(
            record["redundant_group_count"] for record in actual_candidate_hit_groups
        ),
        "actual_candidate_hit_group_ids": [
            record["node_group_id"] for record in actual_candidate_hit_groups
        ],
        "opened_module_family_hits": family_hit_summary,
    }


def c3k2_composite_recovery_expected_candidate_parameter_sets(
    dependency_control: dict[str, Any] | None,
    candidate_parameter_names: list[str] | set[str] | tuple[str, ...],
) -> tuple[set[str], set[str]]:
    """从已登记合同恢复 C3k2 root/child 的语义候选集合。

    recovery 的所有权门禁不能以 ``*.cv1.conv.weight`` 这样的名称后缀猜测
    C3k2 root：YOLO11 的普通 Conv、C2PSA、C3/C3k child 同样存在 ``cv1``。
    必须以 build_oto 已冻结的 PairedChunk / DynamicConcat 合同为唯一来源，
    再与当前 profile 的真实候选参数集合取交集。这样既能拒绝缺失 owner，
    也不会把无关模块错误纳入 source/rebuild 一致性比较。
    """

    dependency_control = dependency_control or {}
    candidates = {str(name) for name in candidate_parameter_names}

    def registered_candidate_parameters(contract_key: str) -> set[str]:
        registered: set[str] = set()
        for contract in dependency_control.get(contract_key, []) or []:
            if not isinstance(contract, dict):
                continue
            registered.update(
                str(name)
                for name in contract.get("parameter_names", []) or []
                if str(name).endswith(".conv.weight")
            )
        return registered & candidates

    return (
        registered_candidate_parameters("yolo11_c3k2_paired_chunk_contracts"),
        registered_candidate_parameters("yolo11_c3k2_dynamic_concat_contracts"),
    )


def rebuild_structured_subnet(args: argparse.Namespace) -> None:
    """从 OTO 全组稀疏裸模型重新建图并构建物理子网，不重复稀疏训练。"""

    print_stage("从全组稀疏模型重新构建 YOLO11 物理子网")
    source_weights = Path(args.weights).expanduser().resolve()
    if not source_weights.is_file():
        raise FileNotFoundError(f"全组稀疏模型不存在：{source_weights}")

    run_dir = Path(args.output).expanduser().resolve() / args.name
    if args.no_overwrite and run_dir.exists():
        raise FileExistsError(f"禁止覆盖已有重建目录：{run_dir}；请更换 --name。")
    run_dir.mkdir(parents=True, exist_ok=True)
    device = device_from_arg(args.device)
    set_seed(args.seed)

    loaded = torch.load(source_weights, map_location=device, weights_only=False)
    if not isinstance(loaded, torch.nn.Module):
        checkpoint_kind = (
            "Ultralytics checkpoint dict"
            if isinstance(loaded, dict)
            else type(loaded).__name__
        )
        raise TypeError(
            "rebuild-subnet 只接受 OTO construct_subnet 保存的裸模型 "
            "DetectionModel_full_group_sparse.pt；"
            f"当前文件类型为 {checkpoint_kind}。"
            "普通 Ultralytics best.pt/last.pt 中的 EMA/model 不能作为物理零组恢复证据，"
            "因为 EMA 可能不保留 HESSO 的精确零组。"
        )
    sparse_model = loaded
    sparse_model = sparse_model.to(device).float().eval()
    enable_oto_trainable_parameters(sparse_model)

    (
        oto,
        protected_prefixes_value,
        protected_names,
        tensor_unsafe_exclusions,
        candidate_prefixes,
        dependency_control,
    ) = build_oto(
        sparse_model,
        args.imgsz,
        device,
        args.extra_protect_prefix,
        args.oto_opset,
        args.module_coverage_profile,
        args.module_probe_only,
        tuple(args.probe_module_family),
    )
    oto_graph_artifacts = (
        export_oto_graphs(
            oto,
            run_dir / "oto_graph",
            formats=tuple(args.oto_graph_format or ("svg", "pdf")),
            display_params=args.oto_graph_display_params,
            display_flops=args.oto_graph_display_flops,
        )
        if args.visualize_oto_graph
        else {"status": "not_requested", "records": []}
    )
    candidate_groups = module_candidate_group_report(oto, candidate_prefixes)
    candidate_issues = module_candidate_group_issues(
        args.module_coverage_profile,
        candidate_prefixes,
        candidate_groups,
        tensor_unsafe_exclusions,
        dependency_control,
    )
    if candidate_issues:
        diagnostic_path = run_dir / "rebuild_candidate_group_audit_failure.json"
        json_dump(
            diagnostic_path,
            {
                "status": "fail",
                "issues": candidate_issues,
                "module_pruning_candidate_groups": candidate_groups,
                "module_dependency_control": dependency_control,
            },
        )
        raise RuntimeError(
            f"重建前候选组审计失败，共 {len(candidate_issues)} 项；"
            f"详细诊断：{diagnostic_path}"
        )

    # recovery 可以复用既有零权重，但不能把不同的 HESSO 搜索空间拼接成同一证据。
    # 从 full_group_sparse.pt 所在 run 读取初始化清单，只比较真正决定稀疏置零的
    # 候选组所有权；允许物理构建实现 revision 更新，但候选组/参数闭包必须相同。
    source_runtime_path = source_weights.parent.parent / "adamcf_runtime.json"
    source_runtime = json_load(source_runtime_path) if source_runtime_path.is_file() else {}

    def nested_value(value, *keys):
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value

    def candidate_ownership_signature(records) -> dict[str, Any]:
        keys = (
            "node_group_type",
            "num_groups",
            "is_prunable",
            "is_auxiliary",
            "candidate_parameter_names",
            "dependency_closure_parameter_names",
            "external_dependency_parameter_names",
            "operator_names",
            "attention_prune_mode",
            "attention_contract",
            "attention_num_heads",
            "attention_key_dim",
            "attention_head_dim",
            "c2psa_contract",
            "c2psa_module_prefix",
            "c2psa_original_segment_width",
            "c3k2_contract",
            "c3k2_module_prefix",
            "c3k2_original_segment_width",
            "c3k2_dynamic_contract",
            "c3k2_dynamic_parent_prefix",
            "c3k2_dynamic_child_prefix",
            "c3k2_dynamic_segment_index",
            "c3k2_dynamic_original_width",
            "parameter_role",
        )
        normalized = []
        diagnostic_trace_node_group_ids = []
        for record in records or []:
            if not isinstance(record, dict):
                continue
            item = {key: record.get(key) for key in keys}
            diagnostic_trace_node_group_ids.append(str(record.get("node_group_id", "")))
            for key in (
                "candidate_parameter_names",
                "dependency_closure_parameter_names",
                "external_dependency_parameter_names",
                "operator_names",
            ):
                item[key] = sorted(item.get(key) or [])
            normalized.append(item)
        normalized.sort(
            key=lambda record: json.dumps(record, ensure_ascii=False, sort_keys=True)
        )
        return {
            "sha256": stable_json_sha256(normalized),
            "records": normalized,
            "diagnostic_trace_node_group_ids": sorted(diagnostic_trace_node_group_ids),
            "identity_policy": "stable_parameter_semantics_excludes_ephemeral_trace_node_ids_v2",
        }

    source_candidate_signature = candidate_ownership_signature(
        source_runtime.get("module_pruning_candidate_groups", [])
    )
    rebuild_candidate_signature = candidate_ownership_signature(candidate_groups)
    source_probe_families = sorted(source_runtime.get("probe_module_families", []) or [])
    rebuild_probe_families = sorted(args.probe_module_family or [])
    expected_candidate_parameter_names = sorted(
        name
        for name, parameter in sparse_model.named_parameters()
        if parameter.requires_grad
        and any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in candidate_prefixes
        )
    )
    expected_attention_candidate_parameter_names = sorted(
        name
        for name in expected_candidate_parameter_names
        if ".attn.qkv." in name or ".attn.pe." in name
    )

    def attention_owner_invariants(signature: dict[str, Any]) -> dict[str, bool]:
        records = [
            record
            for record in signature["records"]
            if "yolo11_attention" in (record.get("operator_names") or [])
        ]
        if len(records) != 1:
            return {
                "exactly_one_candidate_owner": False,
                "candidate_owner_is_composed_prunable_non_auxiliary": False,
                "candidate_parameter_set_exact": False,
                "dependency_closure_contains_candidates": False,
                "attention_role_and_logical_axis_valid": False,
            }
        record = records[0]
        candidates = set(record.get("candidate_parameter_names") or [])
        closure = set(record.get("dependency_closure_parameter_names") or [])
        mode = record.get("attention_prune_mode")
        num_groups = int(record.get("num_groups") or 0)
        expected_groups = (
            int(record.get("attention_num_heads") or 0)
            if mode == "num_head"
            else int(record.get("attention_head_dim") or 0)
            if mode == "head_dim"
            else 0
        )
        return {
            "exactly_one_candidate_owner": True,
            "candidate_owner_is_composed_prunable_non_auxiliary": bool(
                record.get("node_group_type") == "NodeGroupComposedOp"
                and record.get("is_prunable")
                and not record.get("is_auxiliary")
                and "yolo11_attention" in (record.get("operator_names") or [])
            ),
            "candidate_parameter_set_exact": bool(
                sorted(candidates) == expected_attention_candidate_parameter_names
                and candidates
                and not any(".proj." in name for name in candidates)
            ),
            "dependency_closure_contains_candidates": bool(
                candidates and candidates.issubset(closure)
            ),
            "attention_role_and_logical_axis_valid": bool(
                mode == ATTENTION_PROFILE_MODES.get(args.module_coverage_profile)
                and record.get("attention_contract") == "yolo11_partitioned_qkv_v1"
                and num_groups == expected_groups
                and expected_groups > 1
            ),
        }

    def c2psa_split_owner_invariants(signature: dict[str, Any]) -> dict[str, bool]:
        records = [
            record
            for record in signature["records"]
            if record.get("c2psa_contract") == "yolo11_c2psa_paired_split_v1"
        ]
        if len(records) != 1:
            return {
                "exactly_one_candidate_owner": False,
                "candidate_owner_is_prunable_non_auxiliary": False,
                "candidate_parameter_set_exact": False,
                "dependency_closure_contains_candidates": False,
                "paired_split_role_and_logical_axis_valid": False,
            }
        record = records[0]
        candidates = set(record.get("candidate_parameter_names") or [])
        closure = set(record.get("dependency_closure_parameter_names") or [])
        expected_groups = int(record.get("c2psa_original_segment_width") or 0)
        return {
            "exactly_one_candidate_owner": True,
            "candidate_owner_is_prunable_non_auxiliary": bool(
                record.get("node_group_type") == "NodeGroup"
                and record.get("is_prunable")
                and not record.get("is_auxiliary")
            ),
            "candidate_parameter_set_exact": bool(
                sorted(candidates) == expected_candidate_parameter_names
                and candidates
                and all(name.endswith(".cv1.conv.weight") for name in candidates)
            ),
            "dependency_closure_contains_candidates": bool(
                candidates and candidates.issubset(closure)
            ),
            "paired_split_role_and_logical_axis_valid": bool(
                record.get("parameter_role")
                == "c2psa_external_residual_width_owner"
                and int(record.get("num_groups") or 0) == expected_groups
                and expected_groups > 1
            ),
        }

    def c3k2_chunk_owner_invariants(signature: dict[str, Any]) -> dict[str, bool]:
        records = [
            record
            for record in signature["records"]
            if record.get("c3k2_contract") == "yolo11_c3k2_paired_chunk_v1"
        ]
        candidate_sets = [
            set(record.get("candidate_parameter_names") or []) for record in records
        ]
        candidate_union = set().union(*candidate_sets) if candidate_sets else set()
        closure_contains_candidates = all(
            candidates
            and candidates.issubset(
                set(record.get("dependency_closure_parameter_names") or [])
            )
            for record, candidates in zip(records, candidate_sets)
        )
        valid_roles = all(
            record.get("node_group_type") == "NodeGroup"
            and record.get("is_prunable")
            and not record.get("is_auxiliary")
            and record.get("parameter_role")
            == "c3k2_paired_chunk_external_width_owner"
            and int(record.get("num_groups") or 0)
            == int(record.get("c3k2_original_segment_width") or 0)
            and int(record.get("num_groups") or 0) > 1
            for record in records
        )
        return {
            "candidate_owner_count_exact": bool(
                records
                and len(records) == len(expected_candidate_parameter_names)
            ),
            "candidate_owners_prunable_non_auxiliary": bool(records and valid_roles),
            "candidate_parameter_set_exact": bool(
                candidate_union == set(expected_candidate_parameter_names)
                and all(
                    len(candidates) == 1
                    and next(iter(candidates)).endswith(".cv1.conv.weight")
                    for candidates in candidate_sets
                )
            ),
            "dependency_closure_contains_candidates": bool(
                records and closure_contains_candidates
            ),
            "paired_chunk_role_and_logical_axis_valid": bool(
                records and valid_roles
            ),
        }

    def c3k2_dynamic_owner_invariants(signature: dict[str, Any]) -> dict[str, bool]:
        records = [
            record
            for record in signature["records"]
            if record.get("c3k2_dynamic_contract")
            == "yolo11_c3k2_segmented_dynamic_concat_v1"
        ]
        candidate_sets = [
            set(record.get("candidate_parameter_names") or []) for record in records
        ]
        candidate_union = set().union(*candidate_sets) if candidate_sets else set()
        closure_contains_candidates = all(
            candidates
            and candidates.issubset(
                set(record.get("dependency_closure_parameter_names") or [])
            )
            for record, candidates in zip(records, candidate_sets)
        )
        valid_roles = all(
            record.get("node_group_type") == "NodeGroup"
            and record.get("is_prunable")
            and not record.get("is_auxiliary")
            and record.get("parameter_role")
            == "c3k2_dynamic_child_output_owner"
            and int(record.get("num_groups") or 0)
            == int(record.get("c3k2_dynamic_original_width") or 0)
            and int(record.get("num_groups") or 0) > 1
            for record in records
        )
        return {
            "candidate_owner_count_exact": bool(
                records and len(records) == len(expected_candidate_parameter_names)
            ),
            "candidate_owners_prunable_non_auxiliary": bool(records and valid_roles),
            "candidate_parameter_set_exact": bool(
                candidate_union == set(expected_candidate_parameter_names)
                and all(
                    len(candidates) == 1
                    and next(iter(candidates)).endswith(".cv3.conv.weight")
                    for candidates in candidate_sets
                )
            ),
            "dependency_closure_contains_candidates": bool(
                records and closure_contains_candidates
            ),
            "dynamic_concat_role_and_logical_axis_valid": bool(
                records and valid_roles
            ),
        }

    def c3k2_composite_owner_invariants(signature: dict[str, Any]) -> dict[str, bool]:
        """恢复时分别核对根部轴和 child 轴，禁止用单一 owner 覆盖二者。"""

        records = list(signature.get("records", []))
        root_records = [
            record
            for record in records
            if record.get("c3k2_contract") == "yolo11_c3k2_paired_chunk_v1"
        ]
        dynamic_records = [
            record
            for record in records
            if record.get("c3k2_dynamic_contract")
            == "yolo11_c3k2_segmented_dynamic_concat_v1"
        ]
        root_sets = [
            set(record.get("candidate_parameter_names") or [])
            for record in root_records
        ]
        dynamic_sets = [
            set(record.get("candidate_parameter_names") or [])
            for record in dynamic_records
        ]
        root_union = set().union(*root_sets) if root_sets else set()
        dynamic_union = set().union(*dynamic_sets) if dynamic_sets else set()
        # 不能用 ``cv1/cv3`` 名称后缀推测 C3k2 语义；只有冻结的 C3k2
        # PairedChunk / DynamicConcat 合同能区分同名的普通 Conv、C2PSA、
        # C3/C3k 等模块。该集合同时用于 source 和 rebuild，故仍是严格的
        # provenance 门禁而不是放宽比较。
        expected_root, expected_dynamic = (
            c3k2_composite_recovery_expected_candidate_parameter_sets(
                dependency_control,
                expected_candidate_parameter_names,
            )
        )
        expected_composite = expected_root | expected_dynamic

        def valid_root(record: dict[str, Any]) -> bool:
            return bool(
                record.get("node_group_type") == "NodeGroup"
                and record.get("is_prunable")
                and not record.get("is_auxiliary")
                and record.get("parameter_role")
                == "c3k2_paired_chunk_external_width_owner"
                and int(record.get("num_groups") or 0)
                == int(record.get("c3k2_original_segment_width") or 0)
                and int(record.get("num_groups") or 0) > 1
            )

        def valid_dynamic(record: dict[str, Any]) -> bool:
            return bool(
                record.get("node_group_type") == "NodeGroup"
                and record.get("is_prunable")
                and not record.get("is_auxiliary")
                and record.get("parameter_role")
                == "c3k2_dynamic_child_output_owner"
                and int(record.get("num_groups") or 0)
                == int(record.get("c3k2_dynamic_original_width") or 0)
                and int(record.get("num_groups") or 0) > 1
            )

        def closure_contains(record: dict[str, Any], candidates: set[str]) -> bool:
            return bool(candidates) and candidates.issubset(
                set(record.get("dependency_closure_parameter_names") or [])
            )

        return {
            "root_candidate_owners_present": bool(root_records and expected_root),
            "dynamic_candidate_owners_present": bool(dynamic_records and expected_dynamic),
            "root_candidate_parameter_set_exact": root_union == expected_root,
            "dynamic_candidate_parameter_set_exact": dynamic_union == expected_dynamic,
            "candidate_parameter_union_exact": (
                root_union | dynamic_union
            ) == expected_composite,
            "root_dynamic_parameter_ownership_disjoint": not bool(root_union & dynamic_union),
            "root_owner_roles_valid": bool(root_records)
            and all(valid_root(record) for record in root_records),
            "dynamic_owner_roles_valid": bool(dynamic_records)
            and all(valid_dynamic(record) for record in dynamic_records),
            "root_dependency_closure_contains_candidates": bool(root_records)
            and all(
                closure_contains(record, candidates)
                for record, candidates in zip(root_records, root_sets)
            ),
            "dynamic_dependency_closure_contains_candidates": bool(dynamic_records)
            and all(
                closure_contains(record, candidates)
                for record, candidates in zip(dynamic_records, dynamic_sets)
            ),
        }

    def output_consumer_closure_owner_invariants(
        signature: dict[str, Any],
    ) -> dict[str, bool]:
        """恢复时核对 C3k2.cv2 输出轴，禁止误混入其他 ``*.cv2`` 轴。"""

        expected_sources = {
            f"{name}.cv2.conv.weight"
            for name, module in sparse_model.named_modules()
            if name and type(module).__name__ == "C3k2"
        }
        records = [
            record
            for record in signature.get("records", [])
            if set(record.get("candidate_parameter_names") or []) & expected_sources
        ]
        candidate_sets = [
            set(record.get("candidate_parameter_names") or []) for record in records
        ]
        source_union = set().union(*candidate_sets) & expected_sources if candidate_sets else set()
        closure_valid = all(
            candidates
            and candidates.issubset(
                set(record.get("dependency_closure_parameter_names") or [])
            )
            for record, candidates in zip(records, candidate_sets)
        )
        return {
            "output_source_owner_count_exact": len(records) == len(expected_sources),
            "output_source_parameter_set_exact": source_union == expected_sources,
            "output_source_owners_prunable_non_auxiliary": bool(records)
            and all(
                record.get("node_group_type") == "NodeGroup"
                and record.get("is_prunable")
                and not record.get("is_auxiliary")
                for record in records
            ),
            "output_source_dependency_closure_contains_candidates": bool(records)
            and closure_valid,
        }

    provenance_checks = {
        "runtime_manifest_found": bool(source_runtime),
        "module_coverage_profile_matched": (
            source_runtime.get("module_coverage_profile") == args.module_coverage_profile
        ),
        "module_probe_only_matched": (
            bool(source_runtime.get("module_probe_only")) == bool(args.module_probe_only)
        ),
        "probe_module_families_matched": source_probe_families == rebuild_probe_families,
        "candidate_semantic_ownership_matched": (
            source_candidate_signature["sha256"] == rebuild_candidate_signature["sha256"]
        ),
    }
    # v2 全量联合 profile 同时拥有 Attention 与 C3k2 composite 两类语义轴。
    # C2PSA paired-split 与 Attention 内部轴当前仍互斥，不能在这里伪造三轴
    # 联合 provenance。两套 invariant 都必须写入 rebuild provenance，不能因为
    # Attention 分支先命中而漏检 C3k2。旧 profile 保持原有单分支语义，避免
    # 影响已归档 checkpoint 的重建兼容性。
    if args.module_coverage_profile in ALL_VERIFIED_SAFE_JOINT_COMPOSITE_PROFILES:
        source_owner_checks = {
            **{
                f"attention_{name}": value
                for name, value in attention_owner_invariants(
                    source_candidate_signature
                ).items()
            },
            **{
                f"c3k2_composite_{name}": value
                for name, value in c3k2_composite_owner_invariants(
                    source_candidate_signature
                ).items()
            },
            **(
                {
                    f"output_consumer_closure_{name}": value
                    for name, value in output_consumer_closure_owner_invariants(
                        source_candidate_signature
                    ).items()
                }
                if args.module_coverage_profile
                == ALL_VERIFIED_SAFE_NUM_HEAD_V4_OUTPUT_CLOSURE_PROFILE
                else {}
            ),
        }
        rebuild_owner_checks = {
            **{
                f"attention_{name}": value
                for name, value in attention_owner_invariants(
                    rebuild_candidate_signature
                ).items()
            },
            **{
                f"c3k2_composite_{name}": value
                for name, value in c3k2_composite_owner_invariants(
                    rebuild_candidate_signature
                ).items()
            },
            **(
                {
                    f"output_consumer_closure_{name}": value
                    for name, value in output_consumer_closure_owner_invariants(
                        rebuild_candidate_signature
                    ).items()
                }
                if args.module_coverage_profile
                == ALL_VERIFIED_SAFE_NUM_HEAD_V4_OUTPUT_CLOSURE_PROFILE
                else {}
            ),
        }
    elif args.module_coverage_profile in ATTENTION_PROFILE_MODES:
        source_owner_checks = attention_owner_invariants(source_candidate_signature)
        rebuild_owner_checks = attention_owner_invariants(rebuild_candidate_signature)
    elif args.module_coverage_profile in C2PSA_EXTERNAL_WIDTH_PROFILES:
        source_owner_checks = c2psa_split_owner_invariants(source_candidate_signature)
        rebuild_owner_checks = c2psa_split_owner_invariants(rebuild_candidate_signature)
    elif args.module_coverage_profile in C3K2_COMPOSITE_PROFILES:
        source_owner_checks = c3k2_composite_owner_invariants(source_candidate_signature)
        rebuild_owner_checks = c3k2_composite_owner_invariants(rebuild_candidate_signature)
    elif args.module_coverage_profile in C3K2_PAIRED_CHUNK_PROFILES:
        source_owner_checks = c3k2_chunk_owner_invariants(source_candidate_signature)
        rebuild_owner_checks = c3k2_chunk_owner_invariants(rebuild_candidate_signature)
    elif args.module_coverage_profile in C3K2_DYNAMIC_CONCAT_PROFILES:
        source_owner_checks = c3k2_dynamic_owner_invariants(source_candidate_signature)
        rebuild_owner_checks = c3k2_dynamic_owner_invariants(rebuild_candidate_signature)

    if (
        args.module_coverage_profile in ATTENTION_PROFILE_MODES
        or args.module_coverage_profile in C2PSA_EXTERNAL_WIDTH_PROFILES
        or args.module_coverage_profile in C3K2_PAIRED_CHUNK_PROFILES
        or args.module_coverage_profile in C3K2_DYNAMIC_CONCAT_PROFILES
    ):
        provenance_checks.update(
            {f"source_{name}": value for name, value in source_owner_checks.items()}
        )
        provenance_checks.update(
            {f"rebuild_{name}": value for name, value in rebuild_owner_checks.items()}
        )
        source_prune_ratio = source_runtime.get("prune_ratio")
        source_group_divisible = source_runtime.get("group_divisible")
        provenance_checks.update(
            {
                "declared_prune_ratio_matched": bool(
                    source_prune_ratio is not None
                    and abs(float(source_prune_ratio) - float(args.prune_ratio)) <= 1e-12
                ),
                "declared_group_divisible_matched": bool(
                    source_group_divisible is not None
                    and int(source_group_divisible) == int(args.group_divisible)
                ),
            }
        )
    source_sparse_training_provenance = {
        "status": "pass" if all(provenance_checks.values()) else "fail",
        "runtime_manifest": str(source_runtime_path),
        "runtime_manifest_found": bool(source_runtime),
        "runtime_manifest_sha256": (
            sha256_file(source_runtime_path) if source_runtime_path.is_file() else None
        ),
        "source_oto_revision": nested_value(
            source_runtime, "framework_versions", "oto_revision"
        ),
        "rebuild_oto_revision": framework_version_summary()["oto_revision"],
        "source_contract_sha256": nested_value(
            source_runtime, "module_dependency_control", "contract", "sha256"
        ),
        "rebuild_contract_sha256": nested_value(
            dependency_control, "contract", "sha256"
        ),
        "revision_migration": {
            "source_and_rebuild_revision_equal": nested_value(
                source_runtime, "framework_versions", "oto_revision"
            )
            == framework_version_summary()["oto_revision"],
            "policy": (
                "revision/contract 哈希允许因实现升级而变化；恢复放行只依据候选参数、"
                "依赖闭包、逻辑轴和 Attention 结构属性的稳定语义完全相等。"
            ),
        },
        "source_sparse_training_settings": {
            "prune_ratio": source_runtime.get("prune_ratio"),
            "group_divisible": source_runtime.get("group_divisible"),
        },
        "rebuild_declared_training_settings": {
            "prune_ratio": float(args.prune_ratio),
            "group_divisible": int(args.group_divisible),
        },
        "source_candidate_ownership": source_candidate_signature,
        "rebuild_candidate_ownership": rebuild_candidate_signature,
        "checks": provenance_checks,
        "trace_node_id_policy": (
            "node-* 是每次 ONNX trace 的临时序号，只保留诊断，不参与跨重建身份哈希；"
            "候选参数名、依赖闭包、算子类型、逻辑组数和复合结构属性仍严格匹配。"
        ),
    }
    if (
        (
            args.module_coverage_profile in ATTENTION_PROFILE_MODES
            or args.module_coverage_profile in C2PSA_EXTERNAL_WIDTH_PROFILES
            or args.module_coverage_profile in C3K2_PAIRED_CHUNK_PROFILES
            or args.module_coverage_profile in C3K2_DYNAMIC_CONCAT_PROFILES
        )
        and source_sparse_training_provenance["status"] != "pass"
    ):
        diagnostic_path = run_dir / "rebuild_source_provenance_failure.json"
        json_dump(diagnostic_path, source_sparse_training_provenance)
        failed_checks = [
            name
            for name, passed in provenance_checks.items()
            if not passed
        ]
        raise RuntimeError(
            "恢复源与当前定向候选合同不一致："
            f"{len(failed_checks)} 项检查失败；详细诊断：{diagnostic_path}"
        )

    with torch.no_grad():
        sparse_output = sparse_model(torch.zeros(1, 3, args.imgsz, args.imgsz, device=device))
    sparse_stats = model_stats(sparse_model, args.imgsz, device)

    oto_dir = run_dir / "oto_artifacts_rebuild"
    require_attention_hit = args.module_coverage_profile in ATTENTION_ONLY_PROFILES
    attention_redundancy = yolo11_attention_redundancy_contract(
        oto,
        require_physical_redundancy=require_attention_hit,
    )
    if (
        args.module_coverage_profile in ATTENTION_PROFILE_MODES
        and attention_redundancy["status"] != "pass"
    ):
        diagnostic_path = run_dir / "rebuild_attention_redundancy_failure.json"
        json_dump(diagnostic_path, attention_redundancy)
        raise RuntimeError(
            "Attention 重建参数所有权或冗余合同失败；"
            f"详细诊断：{diagnostic_path}"
        )
    attention_was_selected = attention_redundancy.get(
        "physical_redundancy_status"
    ) in {"pass", "partial"}
    oto.construct_subnet(out_dir=str(oto_dir), ckpt_format="torch")
    compressed_raw_path = Path(oto.compressed_model_path).resolve()
    if not compressed_raw_path.is_file():
        raise FileNotFoundError(f"OTO 重建未产生压缩模型：{compressed_raw_path}")
    redundancy = physical_redundancy_summary(oto, sparse_model, candidate_prefixes)
    structured_model = torch.load(
        compressed_raw_path,
        map_location="cpu",
        weights_only=False,
    ).float().eval()

    standard_path = run_dir / "weights" / args.structured_name
    trainer_stub = SimpleNamespace(
        epoch=-1,
        best_fitness=None,
        ema=None,
        args=args,
    )
    standard_ultralytics_checkpoint(
        structured_model,
        trainer_stub,
        standard_path,
        ultralytics.__version__,
        torch,
    )

    # 这是恢复流程的硬门槛：重载后的真实结构化 checkpoint 必须通过固定 shape 前向。
    reloaded = YOLO(str(standard_path))
    with torch.no_grad():
        structured_output = reloaded.model.float().eval()(
            torch.zeros(1, 3, args.imgsz, args.imgsz)
        )
    compressed_stats = model_stats(structured_model, args.imgsz, torch.device("cpu"))
    attention_structure = yolo11_attention_structure_summary(structured_model)
    c2psa_split_structure = c2psa_paired_split_structure_summary(structured_model)
    output_consumer_closure_structure = output_consumer_closure_structure_summary(
        structured_model,
        configured_contracts=(dependency_control or {}).get(
            "yolo11_output_consumer_closure_contracts", []
        ),
    )
    if args.module_coverage_profile in C3K2_COMPOSITE_PROFILES:
        c3k2_chunk_structure = c3k2_paired_chunk_structure_summary(structured_model)
        c3k2_dynamic_structure = {"status": "not_applicable", "records": []}
        c3k2_composite_structure = c3k2_composite_structure_summary(
            structured_model,
            configured_contracts=(dependency_control or {}).get(
                "yolo11_c3k2_composite_contracts",
                [],
            ),
            output_consumer_closure_structure=output_consumer_closure_structure,
        )
    else:
        c3k2_chunk_structure = c3k2_paired_chunk_structure_summary(structured_model)
        c3k2_dynamic_structure = c3k2_dynamic_concat_structure_summary(
            structured_model,
            configured_contracts=(dependency_control or {}).get(
                "yolo11_c3k2_dynamic_concat_contracts",
                [],
            ),
        )
        c3k2_composite_structure = {"status": "not_applicable", "records": []}
    if (
        args.module_coverage_profile in C2PSA_EXTERNAL_WIDTH_PROFILES
        and c2psa_split_structure.get("status") != "pass"
    ):
        diagnostic_path = run_dir / "rebuild_c2psa_paired_split_structure_failure.json"
        json_dump(diagnostic_path, c2psa_split_structure)
        raise RuntimeError(
            "C2PSA paired-split 重建物理结构合同失败；"
            f"详细诊断：{diagnostic_path}"
        )
    if (
        args.module_coverage_profile in C3K2_PAIRED_CHUNK_PROFILES
        and (
            (
                args.module_coverage_profile in C3K2_COMPOSITE_PROFILES
                and c3k2_chunk_structure.get("status") not in {"not_applicable", "pass"}
            )
            or (
                args.module_coverage_profile not in C3K2_COMPOSITE_PROFILES
                and c3k2_chunk_structure.get("status") != "pass"
            )
        )
    ):
        diagnostic_path = run_dir / "rebuild_c3k2_paired_chunk_structure_failure.json"
        json_dump(diagnostic_path, c3k2_chunk_structure)
        raise RuntimeError(
            "C3k2 paired-chunk 重建物理结构合同失败；"
            f"详细诊断：{diagnostic_path}"
        )
    if (
        args.module_coverage_profile in C3K2_DYNAMIC_CONCAT_PROFILES
        and args.module_coverage_profile not in C3K2_COMPOSITE_PROFILES
        and c3k2_dynamic_structure.get("status") != "pass"
    ):
        diagnostic_path = (
            run_dir / "rebuild_c3k2_dynamic_concat_structure_failure.json"
        )
        json_dump(diagnostic_path, c3k2_dynamic_structure)
        raise RuntimeError(
            "C3k2 segmented dynamic-concat 重建物理结构合同失败；"
            f"详细诊断：{diagnostic_path}"
        )
    if (
        args.module_coverage_profile in C3K2_COMPOSITE_PROFILES
        and c3k2_composite_structure.get("status") != "pass"
    ):
        diagnostic_path = run_dir / "rebuild_c3k2_composite_structure_failure.json"
        json_dump(diagnostic_path, c3k2_composite_structure)
        raise RuntimeError(
            "C3k2 paired-chunk 与 dynamic-concat 复合重建物理结构合同失败；"
            f"详细诊断：{diagnostic_path}"
        )
    if (
        args.module_coverage_profile in OUTPUT_CONSUMER_CLOSURE_PROFILES
        and output_consumer_closure_structure.get("status") != "pass"
    ):
        diagnostic_path = run_dir / "rebuild_output_consumer_closure_structure_failure.json"
        json_dump(diagnostic_path, output_consumer_closure_structure)
        raise RuntimeError(
            "生产者输出—消费者输入物理闭包重建合同失败；"
            f"详细诊断：{diagnostic_path}"
        )
    attention_structure_status = attention_structure.get("status")
    if attention_was_selected or require_attention_hit:
        attention_structure_valid = attention_structure_status == "pass"
    else:
        attention_structure_valid = attention_structure_status in {
            "not_applicable",
            "pass",
        }
    if (
        args.module_coverage_profile in ATTENTION_PROFILE_MODES
        and not attention_structure_valid
    ):
        diagnostic = {
            "profile": args.module_coverage_profile,
            "attention_was_selected": attention_was_selected,
            "structure": attention_structure,
        }
        diagnostic_path = run_dir / "rebuild_attention_structure_failure.json"
        json_dump(diagnostic_path, diagnostic)
        raise RuntimeError(
            "Attention 重建物理结构合同失败；"
            f"详细诊断：{diagnostic_path}"
        )
    metadata = {
        "status": "structured_subnet_rebuilt_and_forward_verified",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "source_full_group_sparse_model": str(source_weights),
        "source_full_group_sparse_sha256": sha256_file(source_weights),
        "source_sparse_training_provenance": source_sparse_training_provenance,
        "oto_compressed_raw_model": str(compressed_raw_path),
        "structured_ultralytics_checkpoint": str(standard_path),
        "structured_checkpoint_sha256": sha256_file(standard_path),
        "source_sparse_output_shapes": serialize_output_shape(sparse_output),
        "source_sparse_model_stats": sparse_stats,
        "structured_output_shapes": serialize_output_shape(structured_output),
        "structured_model_stats": compressed_stats,
        "physical_redundancy": redundancy,
        "yolo11_attention_redundancy_contract": attention_redundancy,
        "yolo11_attention_structure": attention_structure,
        "yolo11_c2psa_paired_split_structure": c2psa_split_structure,
        "yolo11_c3k2_paired_chunk_structure": c3k2_chunk_structure,
        "yolo11_c3k2_dynamic_concat_structure": c3k2_dynamic_structure,
        "yolo11_c3k2_composite_structure": c3k2_composite_structure,
        "yolo11_output_consumer_closure_structure": output_consumer_closure_structure,
        "framework_versions": framework_version_summary(),
        "module_coverage_profile": args.module_coverage_profile,
        "module_probe_only": args.module_probe_only,
        "probe_module_families": list(args.probe_module_family),
        "module_pruning_candidate_prefixes": candidate_prefixes,
        "module_pruning_candidate_groups": candidate_groups,
        "module_dependency_control": dependency_control,
        "oto_graph_artifacts": oto_graph_artifacts,
        "protected_module_prefixes": protected_prefixes_value,
        "protected_parameter_count": len(protected_names),
        "oto_tensor_unsafe_exclusions": tensor_unsafe_exclusions,
        "warning": (
            "该子网由既有全组稀疏模型重新建图生成，没有重复稀疏训练。"
            "当前 profile 的结构边界以本次 dependency contract 为准；"
            "最终物理压缩率应以本报告和后续验证为准。"
        ),
    }
    json_dump(run_dir / "rebuild_structured_subnet.json", metadata)
    recovery_hesso = {
        "evidence_source": "reconstructed_from_full_group_sparse_tensor_zeros",
        "num_pruned_groups": int(
            redundancy["actual_candidate_hit_redundant_channel_group_count"]
        ),
        "num_total_groups": int(redundancy["eligible_group_count"]),
        "group_sparsity": float(redundancy["physical_group_sparsity"]),
    }
    recovery_summary = {
        "phase": "剪枝物理子网恢复构建",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "source_weights": str(source_weights),
        "source_weights_sha256": sha256_file(source_weights),
        "final_weights": str(standard_path),
        "final_weights_sha256": sha256_file(standard_path),
        "before_model_stats": sparse_stats,
        "after_model_stats": compressed_stats,
        "compression": compression_summary(sparse_stats, compressed_stats),
        "final_validation": {},
        "baseline_validation": {},
        "validation_details": {},
        "hesso": recovery_hesso,
        "recovery_provenance": {
            "status": "reused_existing_full_group_sparse_model",
            "sparse_training_not_repeated": True,
            "physical_redundancy_evidence": redundancy,
            "attention_redundancy_contract": attention_redundancy,
        },
    }
    json_dump(run_dir / "prune_summary.json", recovery_summary)
    metadata["recovery_prune_summary"] = recovery_summary
    json_dump(run_dir / "rebuild_structured_subnet.json", metadata)
    compression = recovery_summary["compression"]
    print_human_summary(
        "物理子网恢复结果",
        [
            ("状态", "通过：模型已重载并完成固定尺寸前向"),
            (
                "物理零组",
                f"{redundancy['physically_redundant_group_count']}/"
                f"{redundancy['eligible_group_count']} "
                f"({redundancy['physical_group_sparsity'] * 100:.2f}%)",
            ),
            ("实际命中模块", compact_family_hits(redundancy)),
            (
                "参数量",
                f"{sparse_stats['parameters_m']:.3f}M → "
                f"{compressed_stats['parameters_m']:.3f}M "
                f"(-{compression['parameters_reduction_ratio'] * 100:.2f}%)",
            ),
            (
                "FLOPs",
                f"{sparse_stats['flops_g']:.3f}G → {compressed_stats['flops_g']:.3f}G "
                f"(-{compression['flops_reduction_ratio'] * 100:.2f}%)",
            ),
            ("C2PSA split", c2psa_split_console_summary(metadata)),
            ("C3k2 chunk", c3k2_chunk_console_summary(metadata)),
            ("C3k2 dynamic concat", c3k2_dynamic_console_summary(metadata)),
            ("C3k2 composite", c3k2_composite_console_summary(metadata)),
            ("输出—消费者闭包", output_consumer_closure_console_summary(metadata)),
            ("Attention", attention_console_summary(metadata)),
            ("结构化模型", standard_path),
        ],
        artifact=run_dir / "rebuild_structured_subnet.json",
    )


def ultralytics_train_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "data": str(Path(args.data).expanduser().resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": str(Path(args.output).expanduser().resolve()),
        "name": args.name,
        "exist_ok": args.exist_ok,
        "seed": args.seed,
        "optimizer": "SGD",
        "lr0": args.lr0,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": 0.0,
        "close_mosaic": args.close_mosaic,
        "amp": args.amp,
        "compile": False,
        "pretrained": True,
        "plots": args.plots,
    }


# ---------------------------
# 剪枝、微调与导出主流程
# ---------------------------
def require_data_arg(args: argparse.Namespace) -> None:
    if not args.data:
        raise ValueError(f"{args.mode} 需要 --data <COCO data yaml>。")
    if not Path(args.data).expanduser().is_file():
        raise FileNotFoundError(f"数据配置不存在：{args.data}")


def prune_model(args: argparse.Namespace) -> None:
    """执行 YOLO11 AdaMCF 稀疏训练，并构建物理结构化子网。"""
    print_stage("开始执行 YOLO11 AdaMCF 结构化剪枝")
    require_data_arg(args)
    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"权重不存在：{weights}")
    requested_run_dir = Path(args.output).expanduser().resolve() / args.name
    if args.no_overwrite and requested_run_dir.exists():
        raise FileExistsError(f"禁止覆盖已有剪枝目录：{requested_run_dir}；请更换 --name / RUN_ID。")
    if not 0.0 < args.prune_ratio < 1.0:
        raise ValueError("--prune-ratio 必须在 (0, 1) 之间。")
    if args.group_divisible < 1:
        raise ValueError("--group-divisible 必须 >= 1。")

    print_human_summary(
        "剪枝运行配置",
        [
            ("模型", weights),
            ("数据集", Path(args.data).expanduser().resolve()),
            ("剪枝模式", args.module_coverage_profile),
            ("定向模块", "全局" if not args.probe_module_family else "，".join(args.probe_module_family)),
            ("目标组稀疏率", f"{args.prune_ratio * 100:.1f}%"),
            ("训练轮数", args.epochs),
            ("输入尺寸", args.imgsz),
            ("Batch / Workers", f"{args.batch} / {args.workers}"),
            ("设备", args.device),
            ("AMP", "开启" if args.amp else "关闭"),
            ("OTO 结构图", "生成" if args.visualize_oto_graph else "不生成"),
        ],
    )

    stage_started = time.time()
    set_seed(args.seed)
    config = AdaMCFTrainingConfig(
        oto_opset=args.oto_opset,
        visualize_oto_graph=bool(args.visualize_oto_graph),
        oto_graph_formats=tuple(args.oto_graph_format or ("svg", "pdf")),
        oto_graph_display_params=bool(args.oto_graph_display_params),
        oto_graph_display_flops=bool(args.oto_graph_display_flops),
        imgsz=args.imgsz,
        prune_ratio=args.prune_ratio,
        warmup_epochs=args.warmup_epochs,
        pruning_periods=args.pruning_periods,
        group_divisible=args.group_divisible,
        module_coverage_profile=args.module_coverage_profile,
        module_probe_only=args.module_probe_only,
        probe_module_families=tuple(args.probe_module_family),
        seed=args.seed,
        extra_protect_prefixes=tuple(args.extra_protect_prefix),
        alpha_cache_batches=args.alpha_cache_batches,
        enable_alpha_update=args.enable_alpha_update,
        alpha_lr=args.alpha_lr,
        alpha_eps=args.alpha_eps,
        alpha_ema_gamma=args.alpha_ema_gamma,
        alpha_grad_ema_gamma=args.alpha_grad_ema_gamma,
        alpha_max_step_l1=args.alpha_max_step_l1,
    )
    trainer_class = create_adamcf_trainer_class()
    yolo = YOLO(str(weights))
    baseline_stats = model_stats(yolo.model, args.imgsz, torch.device("cpu"))
    kwargs = ultralytics_train_kwargs(args)
    kwargs["trainer"] = partial(trainer_class, adamcf_config=config)
    yolo.train(**kwargs)

    final_hesso = hesso_summary(yolo.trainer.optimizer)
    sparse_training_validation = validation_summary(yolo.trainer.metrics)
    metadata = finalize_structured_subnet(yolo.trainer, args)
    metadata["source_weights"] = str(weights)
    metadata["source_weights_sha256"] = sha256_file(weights)
    config_dict = asdict(config)
    config_dict["oto_root"] = str(OTO_ROOT)
    metadata["adamcf_config"] = config_dict
    run_dir = Path(yolo.trainer.save_dir)
    validation_project = run_dir / "validation"
    baseline_validation = validate_yolo_checkpoint(
        weights,
        args,
        project=validation_project,
        name="baseline_unpruned",
    )
    structured_weights = Path(metadata["structured_ultralytics_checkpoint"])
    structured_validation = validate_yolo_checkpoint(
        structured_weights,
        args,
        project=validation_project,
        name="structured_post_prune",
    )
    metadata["baseline_validation"] = baseline_validation
    metadata["sparse_training_validation"] = sparse_training_validation
    metadata["structured_validation"] = structured_validation
    json_dump(run_dir / "structured_subnet.json", metadata)
    prune_summary = save_stage_summary(
        run_dir / "prune_summary.json",
        phase="剪枝",
        started_at=stage_started,
        source_weights=weights,
        final_weights=structured_weights,
        before_stats=baseline_stats,
        after_stats=metadata["structured_model_stats"],
        final_validation=structured_validation,
        epoch_reporter=yolo.trainer.epoch_reporter,
        hesso=final_hesso,
        baseline_validation=baseline_validation,
        validation_details={"sparse_training": sparse_training_validation, "structured_subnet": structured_validation},
        validation_scope="此 mAP 是重新加载物理结构化子网 checkpoint 后执行的独立 COCO val 结果。",
    )
    print_pruning_compression_report(
        structured_weights=structured_weights,
        before_stats=baseline_stats,
        after_stats=metadata["structured_model_stats"],
        baseline_validation=baseline_validation,
        structured_validation=structured_validation,
        hesso=final_hesso,
    )
    metadata["prune_summary"] = prune_summary
    metadata_path = run_dir / "adamcf_run_metadata.json"
    json_dump(metadata_path, metadata)
    print_human_summary(
        "剪枝结构验证",
        [
            ("物理构网", "通过" if metadata.get("status") == "structured_subnet_built" else metadata.get("status")),
            ("模型重载前向", "通过" if metadata.get("reloaded_output_shapes") else "失败"),
            (
                "实际命中模块",
                compact_family_hits(metadata.get("physical_redundancy", {})),
            ),
            ("C2PSA split", c2psa_split_console_summary(metadata)),
            ("C3k2 chunk", c3k2_chunk_console_summary(metadata)),
            ("C3k2 dynamic concat", c3k2_dynamic_console_summary(metadata)),
            ("C3k2 composite", c3k2_composite_console_summary(metadata)),
            ("输出—消费者闭包", output_consumer_closure_console_summary(metadata)),
            ("Attention", attention_console_summary(metadata)),
            ("OTO 结构图", metadata.get("oto_graph_artifacts", {}).get("manifest", "未生成")),
            ("结构化模型", structured_weights),
        ],
        artifact=metadata_path,
    )


def finetune_model(args: argparse.Namespace) -> None:
    """对 OTO 物理结构化子网执行常规 Ultralytics 微调。"""
    print_stage("开始执行 YOLO11 结构化子网微调")
    require_data_arg(args)
    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"结构化 checkpoint 不存在：{weights}")
    requested_run_dir = Path(args.output).expanduser().resolve() / args.name
    if args.no_overwrite and requested_run_dir.exists():
        raise FileExistsError(f"禁止覆盖已有微调目录：{requested_run_dir}；请更换 --name / RUN_ID。")
    if args.admission_probe:
        if args.epochs < 1:
            raise ValueError("--admission-probe 要求 --epochs 至少为 1。")
        if not args.fixed_input:
            raise ValueError("--admission-probe 要求 --fixed-input <冻结样本图片>。")
        if args.amp:
            raise ValueError("--admission-probe 当前要求 --no-amp，避免缩放梯度掩盖首步反向审计。")

    stage_started = time.time()
    set_seed(args.seed)
    prune_summary_path = weights.parent.parent / "prune_summary.json"
    prune_summary = json_load(prune_summary_path) if prune_summary_path.is_file() else {}
    baseline_weights_value = args.baseline_weights or prune_summary.get("source_weights")
    if not baseline_weights_value:
        raise ValueError(
            "微调总结需要未剪枝模型作为基线；请传 --baseline-weights <原始 yolo11n.pt>。"
        )
    baseline_weights = Path(str(baseline_weights_value)).expanduser().resolve()
    if not baseline_weights.is_file():
        raise FileNotFoundError(
            f"未剪枝基线权重不存在：{baseline_weights}；请用 --baseline-weights 指定服务器上的正确路径。"
        )

    fixed_forward: dict[str, Any] = {}
    preflight_dir: Path | None = None
    if args.fixed_input:
        preflight_dir = (
            Path(args.output).expanduser().resolve()
            / "_admission_preflight"
            / args.name
        )
        fixed_forward = fixed_input_forward_report(
            weights,
            Path(args.fixed_input),
            imgsz=args.imgsz,
            device=device_from_arg(args.device),
            output_dir=preflight_dir,
            report_name="structured_fixed_input_forward",
        )
        print(
            "[fixed-input-forward] "
            f"status={fixed_forward['status']} input_sha256="
            f"{fixed_forward['fixed_input']['source_image_sha256']} "
            f"output_shape={fixed_forward['prediction']['shape']}",
            flush=True,
        )

    validation_project = Path(args.output).expanduser().resolve() / "_validation"
    baseline_candidate = prune_summary.get("baseline_validation")
    validation_details = prune_summary.get("validation_details")
    structured_candidate = (
        validation_details.get("structured_subnet") if isinstance(validation_details, dict) else None
    )
    baseline_sha256 = sha256_file(baseline_weights)
    structured_sha256 = sha256_file(weights)
    if (
        isinstance(baseline_candidate, dict)
        and baseline_candidate.get("weights_sha256") == baseline_sha256
        and scalar_float(baseline_candidate.get("map50")) is not None
        and scalar_float(baseline_candidate.get("map50_95")) is not None
    ):
        baseline_validation = baseline_candidate
        print(f"[validation-cache] 复用剪枝阶段未剪枝基线验证：{prune_summary_path}", flush=True)
    else:
        baseline_validation = validate_yolo_checkpoint(
            baseline_weights,
            args,
            project=validation_project,
            name=f"{args.name}_baseline_unpruned",
        )
    if (
        isinstance(structured_candidate, dict)
        and structured_candidate.get("weights_sha256") == structured_sha256
        and scalar_float(structured_candidate.get("map50")) is not None
        and scalar_float(structured_candidate.get("map50_95")) is not None
    ):
        structured_validation = structured_candidate
        print(f"[validation-cache] 复用剪枝阶段物理子网验证：{prune_summary_path}", flush=True)
    else:
        structured_validation = validate_yolo_checkpoint(
            weights,
            args,
            project=validation_project,
            name=f"{args.name}_structured_pre_finetune",
        )

    yolo = YOLO(str(weights))
    structured_template = copy.deepcopy(yolo.model).cpu().float()
    before_stats = model_stats(structured_template, args.imgsz, torch.device("cpu"))

    kwargs = ultralytics_train_kwargs(args)
    # 微调阶段采用正常 SGD；AdaMCF/HESSO 仅用于上一阶段的稀疏化和子网选择。
    # 必须把 OTO 物理子网直接交给 trainer；按 YAML 重建会退回标准 YOLO11 结构。
    kwargs["pretrained"] = False
    kwargs["trainer"] = create_finetune_trainer_class(
        structured_template,
        weights,
        admission_probe=args.admission_probe,
    )
    yolo.train(**kwargs)
    run_dir = Path(yolo.trainer.save_dir)
    if preflight_dir is not None and preflight_dir.is_dir():
        shutil.copytree(preflight_dir, run_dir / "admission", dirs_exist_ok=True)
    best_checkpoint = run_dir / "weights" / "best.pt"
    last_checkpoint = run_dir / "weights" / "last.pt"
    if args.admission_probe and not best_checkpoint.is_file():
        raise FileNotFoundError(
            f"复杂模块九段准入要求独立重载 best.pt，但微调后未找到：{best_checkpoint}"
        )
    final_checkpoint = best_checkpoint if best_checkpoint.is_file() else last_checkpoint
    if not final_checkpoint.is_file():
        raise FileNotFoundError(f"微调结束后未找到 best.pt 或 last.pt：{run_dir}")
    final_yolo = YOLO(str(final_checkpoint))
    after_stats = model_stats(final_yolo.model, args.imgsz, torch.device("cpu"))
    expected_signature = model_structure_signature(structured_template)
    final_signature = model_structure_signature(final_yolo.model)
    architecture_check = dict(yolo.trainer.structure_check)
    architecture_check["final_checkpoint"] = str(final_checkpoint)
    architecture_check["final_checkpoint_sha256"] = sha256_file(final_checkpoint)
    architecture_check["final_model_sha256"] = final_signature["sha256"]
    architecture_check["final_parameter_count"] = final_signature["parameter_count"]
    architecture_check["final_structure_matched"] = (
        final_signature["sha256"] == expected_signature["sha256"]
    )
    architecture_check["matched"] = bool(architecture_check.get("matched")) and architecture_check[
        "final_structure_matched"
    ]
    if not architecture_check["matched"]:
        raise RuntimeError(
            "微调保存的 checkpoint 已不再是输入的物理结构化子网，拒绝生成成功摘要。"
        )
    loss_backward_check = dict(getattr(yolo.trainer, "loss_backward_check", {}))
    if args.admission_probe and (
        loss_backward_check.get("status") != "pass"
        or not loss_backward_check.get("optimizer_step_completed")
    ):
        raise RuntimeError("1 epoch 微调结束，但未取得通过的 task loss 反向/optimizer step 证据。")

    final_validation = validate_yolo_checkpoint(
        final_checkpoint,
        args,
        project=run_dir / "validation",
        name="finetuned_best",
    )
    finetune_summary = save_stage_summary(
        run_dir / "finetune_summary.json",
        phase="微调",
        started_at=stage_started,
        source_weights=weights,
        final_weights=final_checkpoint,
        before_stats=before_stats,
        after_stats=after_stats,
        final_validation=final_validation,
        epoch_reporter=yolo.trainer.epoch_reporter,
        baseline_validation=baseline_validation,
        validation_details={
            "structured_pre_finetune": structured_validation,
            "trainer_final_eval": validation_summary(yolo.trainer.metrics),
            "finetuned_best_independent": final_validation,
        },
        architecture_check=architecture_check,
        validation_scope="此 mAP 是重新加载微调阶段最佳 checkpoint 后执行的独立 COCO val 结果。",
    )
    module_admission = {
        "enabled": bool(args.admission_probe),
        "fixed_input_forward": fixed_forward,
        "task_loss_backward": loss_backward_check,
        "one_epoch_finetune": {
            "status": (
                "pass"
                if args.epochs >= 1 and len(yolo.trainer.epoch_reporter.records) >= 1
                else "not_applicable" if not args.admission_probe else "fail"
            ),
            "configured_epochs": int(args.epochs),
            "completed_epoch_records": len(yolo.trainer.epoch_reporter.records),
        },
        "best_checkpoint_independent_reload": {
            "status": "pass" if architecture_check.get("final_structure_matched") else "fail",
            "checkpoint": str(final_checkpoint),
            "checkpoint_role": "best" if final_checkpoint == best_checkpoint else "last_fallback",
            "checkpoint_sha256": sha256_file(final_checkpoint),
            "structure_sha256": final_signature["sha256"],
            "parameter_count": final_signature["parameter_count"],
            "validation": final_validation,
        },
    }
    finetune_summary["module_admission"] = module_admission
    json_dump(run_dir / "finetune_summary.json", finetune_summary)
    print_finetune_completion_report(
        final_weights=final_checkpoint,
        baseline_validation=baseline_validation,
        final_validation=final_validation,
        architecture_check=architecture_check,
    )
    metadata = {
        "status": "finetune_completed",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "input_structured_weights": str(weights),
        "input_structured_weights_sha256": sha256_file(weights),
        "baseline_weights": str(baseline_weights),
        "baseline_weights_sha256": sha256_file(baseline_weights),
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
        "architecture_check": architecture_check,
        "module_admission": module_admission,
        "finetune_summary": finetune_summary,
        "note": "微调完成后仍需用项目一的 official COCO eval 脚本验证质量。",
    }
    finetune_metadata_path = run_dir / "finetune_metadata.json"
    json_dump(finetune_metadata_path, metadata)
    print_human_summary(
        "微调验收结果",
        [
            ("状态", "完成"),
            (
                "固定输入前向",
                module_admission.get("fixed_input_forward", {}).get("status", "未执行"),
            ),
            (
                "任务损失反向",
                module_admission.get("task_loss_backward", {}).get("status", "未执行"),
            ),
            (
                "独立重载",
                module_admission.get("best_checkpoint_independent_reload", {}).get("status", "未执行"),
            ),
            ("最终模型", final_checkpoint),
        ],
        artifact=finetune_metadata_path,
    )


def check_onnx_model(onnx_path: Path) -> dict[str, Any]:
    """运行 onnx.checker；不在代码中自动安装或替换依赖版本。"""

    try:
        import onnx

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        return {
            "status": "pass",
            "onnx_version": str(onnx.__version__),
            "graph_input_names": [value.name for value in model.graph.input],
            "graph_output_names": [value.name for value in model.graph.output],
            "node_count": len(model.graph.node),
        }
    except Exception as exc:
        return {
            "status": "fail",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def onnxruntime_alignment_report(
    *,
    weights: Path,
    onnx_path: Path,
    fixed_input: Path,
    imgsz: int,
    device,
    atol: float,
    rtol: float,
    output_dir: Path,
) -> dict[str, Any]:
    """使用同一冻结 tensor 对齐 PyTorch 物理子网与 ONNX Runtime 输出。"""

    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    input_tensor, input_metadata = fixed_input_image_tensor(fixed_input, imgsz)
    input_array = np.ascontiguousarray(input_tensor.numpy(), dtype=np.float32)
    input_npy = output_dir / "onnxruntime_alignment_input.npy"
    np.save(input_npy, input_array, allow_pickle=False)

    try:
        import onnxruntime as ort

        model = YOLO(str(weights)).model.to(device).float().eval()
        # 与 Ultralytics ONNX 导出路径保持一致，只取 Detect 的最终预测 tensor。
        for module in model.modules():
            if type(module).__name__ == "Detect":
                if hasattr(module, "export"):
                    module.export = True
                if hasattr(module, "format"):
                    module.format = "onnx"
                if hasattr(module, "dynamic"):
                    module.dynamic = False
        with torch.no_grad():
            pytorch_prediction = primary_prediction_tensor(model(input_tensor.to(device)))
        pytorch_array = np.ascontiguousarray(
            pytorch_prediction.detach().cpu().float().numpy(),
            dtype=np.float32,
        )

        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        session_inputs = session.get_inputs()
        session_outputs = session.get_outputs()
        if len(session_inputs) != 1:
            raise RuntimeError(f"期望 ONNX 只有一个输入，实际为 {len(session_inputs)}")
        ort_values = session.run(None, {session_inputs[0].name: input_array})
        if len(ort_values) != 1:
            raise RuntimeError(f"期望 ONNX 只有一个主输出，实际为 {len(ort_values)}")
        ort_array = np.asarray(ort_values[0], dtype=np.float32)

        shape_matched = list(pytorch_array.shape) == list(ort_array.shape)
        pytorch_finite = bool(np.isfinite(pytorch_array).all())
        ort_finite = bool(np.isfinite(ort_array).all())
        if shape_matched:
            absolute_error = np.abs(pytorch_array - ort_array)
            max_abs_error = float(absolute_error.max()) if absolute_error.size else 0.0
            mean_abs_error = float(absolute_error.mean()) if absolute_error.size else 0.0
            relative_error = absolute_error / np.maximum(np.abs(pytorch_array), atol)
            max_relative_error = float(relative_error.max()) if relative_error.size else 0.0
            allclose = bool(np.allclose(pytorch_array, ort_array, atol=atol, rtol=rtol))
        else:
            max_abs_error = None
            mean_abs_error = None
            max_relative_error = None
            allclose = False
        passed = shape_matched and pytorch_finite and ort_finite and allclose
        report = {
            "status": "pass" if passed else "fail",
            "onnxruntime_version": str(ort.__version__),
            "execution_providers": session.get_providers(),
            "input_name": session_inputs[0].name,
            "output_names": [value.name for value in session_outputs],
            "pytorch_output": tensor_value_summary(torch.from_numpy(pytorch_array)),
            "onnxruntime_output": tensor_value_summary(torch.from_numpy(ort_array)),
            "shape_matched": shape_matched,
            "pytorch_finite": pytorch_finite,
            "onnxruntime_finite": ort_finite,
            "atol": atol,
            "rtol": rtol,
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "max_relative_error": max_relative_error,
            "allclose": allclose,
        }
    except Exception as exc:
        report = {
            "status": "fail",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "atol": atol,
            "rtol": rtol,
        }

    report.update(
        {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "weights": str(weights),
            "weights_sha256": sha256_file(weights),
            "onnx": str(onnx_path),
            "onnx_sha256": sha256_file(onnx_path),
            "fixed_input": input_metadata,
            "fixed_input_npy": str(input_npy),
            "fixed_input_npy_sha256": sha256_file(input_npy),
            "preprocess_contract": "static_letterbox_rgb_float32_v1",
        }
    )
    json_dump(output_dir / "onnxruntime_alignment.json", report)
    return report


def export_model(args: argparse.Namespace) -> None:
    """导出已训练或微调的 YOLO11 checkpoint 为 ONNX。"""
    print_stage("开始导出 YOLO11 ONNX")
    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"权重不存在：{weights}")
    output = Path(args.export_output).expanduser().resolve()
    output_metadata = output.with_suffix(output.suffix + ".metadata.json")
    default_export = weights.with_suffix(".onnx")
    if args.no_overwrite:
        existing = [path for path in {output, output_metadata, default_export} if path.exists()]
        if existing:
            raise FileExistsError(
                "禁止覆盖已有导出产物：" + ", ".join(str(path) for path in sorted(existing, key=str))
                + "；请更换 --export-output / RUN_ID。"
            )
    output.parent.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(str(weights))
    exported = Path(
        yolo.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            dynamic=args.dynamic,
            simplify=args.simplify,
            nms=False,
            device=args.device,
        )
    ).resolve()
    if not exported.is_file():
        raise FileNotFoundError(f"Ultralytics 未产生 ONNX：{exported}")
    if exported != output:
        shutil.copy2(exported, output)

    onnx_checker = (
        check_onnx_model(output)
        if args.onnx_checker
        else {"status": "not_executed", "enabled": False}
    )
    onnxruntime_alignment = (
        onnxruntime_alignment_report(
            weights=weights,
            onnx_path=output,
            fixed_input=Path(args.fixed_input),
            imgsz=args.imgsz,
            device=device_from_arg(args.device),
            atol=args.alignment_atol,
            rtol=args.alignment_rtol,
            output_dir=output.parent / "admission" / "onnxruntime",
        )
        if args.onnxruntime_align
        else {"status": "not_executed", "enabled": False}
    )
    verification_passed = (
        (not args.onnx_checker or onnx_checker.get("status") == "pass")
        and (not args.onnxruntime_align or onnxruntime_alignment.get("status") == "pass")
    )

    metadata = {
        "status": "onnx_export_verified" if verification_passed else "onnx_export_verification_failed",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "onnx": str(output),
        "onnx_sha256": sha256_file(output),
        "onnx_size_mb": output.stat().st_size / (1024 * 1024),
        "imgsz": args.imgsz,
        "opset": args.opset,
        "dynamic": args.dynamic,
        "simplify": args.simplify,
        "onnx_checker": onnx_checker,
        "onnxruntime_alignment": onnxruntime_alignment,
    }
    json_dump(output_metadata, metadata)
    print_human_summary(
        "ONNX 导出结果",
        [
            ("状态", "通过" if verification_passed else "失败"),
            ("ONNX", output),
            ("文件大小", f"{metadata['onnx_size_mb']:.2f} MB"),
            ("ONNX checker", onnx_checker.get("status")),
            ("PyTorch/ORT 对齐", onnxruntime_alignment.get("status")),
        ],
        artifact=output_metadata,
    )
    if not verification_passed:
        raise RuntimeError(f"ONNX checker/ORT 数值对齐失败：{output_metadata}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO11 AdaMCF structured pruning")
    parser.add_argument(
        "mode",
        choices=("audit", "prune", "sparse-train", "rebuild-subnet", "finetune", "export"),
        help=(
            "prune 为主剪枝入口；sparse-train 保留为兼容旧命令的别名；"
            "rebuild-subnet 从 OTO 全组稀疏裸模型恢复构建，不重复训练。"
        ),
    )
    parser.add_argument("--weights", required=True, help="源模型或结构化模型的 .pt 路径")
    parser.add_argument(
        "--structured-name",
        default="yolo11_adamcf_structured.pt",
        help="prune 模式保存的物理结构化 checkpoint 文件名；必须是不含目录的 .pt 文件名。",
    )
    parser.add_argument(
        "--baseline-weights",
        help="finetune 模式用于统计对比的未剪枝原始权重；默认尝试从同级 prune_summary.json 推断。",
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_YAML),
        help="Ultralytics COCO data YAML；默认使用本目录 coco2017_yolo11.yaml，可按需覆盖。",
    )
    parser.add_argument(
        "--oto-opset",
        type=int,
        default=16,
        help="仅供 OTO 图追踪使用的 ONNX opset；与 export 模式的 --opset 独立",
    )
    parser.add_argument(
        "--visualize-oto-graph",
        action="store_true",
        help="建图后生成 OTO 节点组图与原始算子图；DOT 始终保存，默认尝试 SVG/PDF。",
    )
    parser.add_argument(
        "--oto-graph-format",
        action="append",
        choices=("svg", "pdf", "png"),
        default=[],
        help="OTO 图渲染格式，可重复指定；未指定时生成 SVG 和 PDF。",
    )
    parser.add_argument(
        "--oto-graph-display-params",
        action="store_true",
        help="在 OTO 图标签中显示参数信息；整网图可能明显变大。",
    )
    parser.add_argument(
        "--oto-graph-display-flops",
        action="store_true",
        help="在 OTO 图标签中显示 FLOPs 信息。",
    )
    parser.add_argument("--output", default=str(BASE_DIR / "runs"), help="训练/audit 输出根目录")
    parser.add_argument("--name", default="yolo11_adamcf", help="本次实验名称")
    parser.add_argument("--exist-ok", action="store_true", help="允许 Ultralytics 复用同名输出目录")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="任一阶段发现同名目录、验证结果或导出文件时立即终止；30% 专用入口自动启用。",
    )
    parser.add_argument("--device", default="0", help="例如 0、cuda:0 或 cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="默认关闭，避免 Ultralytics AMP 自检下载官方 yolo11n.pt；验证后可显式传 --amp 启用。",
    )
    parser.add_argument("--plots", action="store_true")
    parser.add_argument(
        "--fixed-input",
        help=(
            "复杂模块准入使用的冻结样本图片。finetune --admission-probe 用它执行物理子网固定输入前向；"
            "export --onnxruntime-align 用同一图片生成完全相同的 PyTorch/ORT 输入 tensor。"
        ),
    )
    parser.add_argument(
        "--admission-probe",
        action="store_true",
        help=(
            "finetune 模式启用九段门槛：在首个 epoch 审计固定输入、真实 task loss 梯度和首次 "
            "optimizer step；可用于 1 epoch 准入，也可用于连续正式微调。"
        ),
    )

    parser.add_argument("--prune-ratio", type=float, default=0.10, help="目标结构化 group sparsity")
    parser.add_argument("--warmup-epochs", type=float, default=5.0, help="HESSO 开始物理置零前的训练 epoch")
    parser.add_argument("--pruning-periods", type=int, help="HESSO 剪枝周期数；默认按剪枝 epoch 数自动设置")
    parser.add_argument("--group-divisible", type=int, default=8, help="保留通道数的可整除约束")
    parser.add_argument(
        "--module-coverage-profile",
        choices=MODULE_COVERAGE_PROFILES,
        default="conservative_v1",
        help=(
            "模块剪枝覆盖策略。conservative_v1 保持 P10/P30 原行为；"
            "bottleneck_hidden_v1_experimental 仅开放 Bottleneck.cv1 隐藏通道；"
            "csp_dense_closure_v1_experimental 开放 CSP/SPPF、前后常规 Conv 与 Detect 隐藏/输入支路的"
            "卷积依赖闭包，同时固定 Attention、Detect 最终预测输出维及尚无 paired transform 的"
            "C3k2.cv1 chunk 源/self.m 动态 concat 输出；"
            "csp_psa_ffn_closure_v1_experimental 在此前基础上只新增开放 PSABlock.ffn.0 隐藏维，"
            "Attention、C2PSA split/concat 边界和 ffn.1 残差输出继续精确固定；"
            "c2psa_output_closure_v1_experimental 再解除 C2PSA.cv2 输出保护，但 cv1 split 源、"
            "Attention 与 ffn.1 残差输出仍固定；"
            "c2psa_paired_split_v1_experimental 仅开放 C2PSA.cv1 的成对 split 逻辑组，"
            "同步缩小旁路/PSA 外部宽度和 cv2 concat 输入，Attention 内部 head/head_dim 固定；"
            "c3k2_paired_chunk_v1_experimental 一次性开放全部 C3k2.cv1 paired-chunk 逻辑组，"
            "同步缩小两个 chunk、所有 self.m 外部宽度和最终 cv2 concat 输入；"
            "独立 child output 动态 concat 轴继续隔离；"
            "c3k2_dynamic_concat_v1_experimental 一次性开放 3 个独立 C3/C3k "
            "child-output，通过前序 segment 当前宽度计算最终 cv2 输入 offset；"
            "根部 paired-chunk 轴及共享所有权的 Bottleneck child-output 保持固定；"
            "c3k2_output_consumer_closure_v1_experimental 自动以每个 C3k2.cv2 输出为候选，"
            "按图拓扑识别直连普通 Conv 输入轴、深度卷积耦合轴及 channel Concat→普通 Conv；"
            "Concat 使用通用 segmented-concat 原始 offset 合同一次性裁剪消费者输入轴，"
            "残差/切片或未解析路径仍 fail-closed，并验证物理裁剪后所有消费者闭合；"
            "yolo11_all_cv1_open_v1 将模型中全部 cv1.conv 一次性加入候选池，"
            "C2PSA.cv1 与 C3k2.cv1 分别复用 paired-split / paired-chunk 合同；"
            "Attention 内部 num_head/head_dim、Detect 最终语义输出和独立动态 concat 轴保持固定；"
            "c2psa_attention_num_head_v1_experimental / c2psa_attention_head_dim_v1_experimental "
            "分别开放 YOLO11 Attention 完整 head 和所有 heads 的同一 V/head 维，"
            "Attention 外部输出宽度仍固定；"
            "all_verified_safe_num_head_v1_experimental 联合开放已通过准入的 dense closure、"
            "SPPF、PSA FFN、C2PSA 输出及 Attention 完整 head，同时继续固定 CSP chunk/concat、"
            "C2PSA split、PSA residual、Detect 最终语义输出和 Attention proj 外部输出边界。"
            "all_verified_safe_num_head_v2_experimental / "
            "all_verified_safe_head_dim_v2_experimental 在 v1 的基础上分别纳入 Attention "
            "完整 head / head_dim 和 C3k2 paired-chunk + child dynamic-concat 的已验证"
            "复合所有权；C2PSA paired-split 外部宽度仍与 Attention 内部轴互斥，两种"
            "Attention 轴也必须分开运行。"
            "all_verified_safe_num_head_v4_output_closure_experimental 在 num-head v2 "
            "的基础上以通用生产者输出—消费者输入闭包替代 C3k2.cv2 临时输出 guard；"
            "audit 强制检查 8 个输出闭包、3 个 segmented-concat 合同以及与 C3k2 "
            "复合输入轴 owner 的无重叠关系。"
        ),
    )
    parser.add_argument(
        "--module-probe-only",
        action="store_true",
        help=(
            "在 OTO 组级仅让实验 profile 的候选依赖组参与 HESSO；Concat 等 auxiliary 组仍保留索引传播，"
            "用于模块级短程物理剪枝 probe。"
        ),
    )
    parser.add_argument(
        "--probe-module-family",
        action="append",
        choices=PROBE_MODULE_FAMILIES,
        default=[],
        help=(
            "仅在 dense closure 实验 profile + --module-probe-only 下使用；"
            "把候选卷积进一步限制到指定复合模块族，可重复传入，用于 SPPF/Detect 等定向真实命中。"
        ),
    )
    parser.add_argument(
        "--extra-protect-prefix",
        action="append",
        default=[],
        help="额外保护的 model 参数前缀；可重复指定，例如 model.22",
    )

    parser.add_argument("--alpha-cache-batches", type=int, default=2, help="AdaMCF alpha 更新缓存的微批数")
    parser.add_argument("--enable-alpha-update", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--alpha-lr", type=float, default=0.02)
    parser.add_argument("--alpha-eps", type=float, default=0.03)
    parser.add_argument("--alpha-ema-gamma", type=float, default=0.6)
    parser.add_argument("--alpha-grad-ema-gamma", type=float, default=0.5)
    parser.add_argument("--alpha-max-step-l1", type=float, default=0.18)

    parser.add_argument("--export-output", help="export 模式下的 ONNX 输出路径")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--onnx-checker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="export 后运行 onnx.checker；默认启用。",
    )
    parser.add_argument(
        "--onnxruntime-align",
        action="store_true",
        help="export 后使用 --fixed-input 做 PyTorch/ONNX Runtime 同输入数值对齐。",
    )
    parser.add_argument("--alignment-atol", type=float, default=1e-4)
    parser.add_argument("--alignment-rtol", type=float, default=1e-3)
    args = parser.parse_args()

    if args.mode == "export" and not args.export_output:
        parser.error("export 模式需要 --export-output <model.onnx>")
    if args.admission_probe and args.mode != "finetune":
        parser.error("--admission-probe 只能用于 finetune 模式")
    if args.onnxruntime_align and args.mode != "export":
        parser.error("--onnxruntime-align 只能用于 export 模式")
    if args.onnxruntime_align and not args.fixed_input:
        parser.error("--onnxruntime-align 需要 --fixed-input <冻结样本图片>")
    if args.fixed_input and not Path(args.fixed_input).expanduser().is_file():
        parser.error(f"--fixed-input 不存在：{args.fixed_input}")
    if args.alignment_atol < 0 or args.alignment_rtol < 0:
        parser.error("--alignment-atol/--alignment-rtol 必须 >= 0")
    if args.no_overwrite and args.exist_ok:
        parser.error("--no-overwrite 与 --exist-ok 互斥")
    if args.module_probe_only and args.module_coverage_profile == "conservative_v1":
        parser.error("--module-probe-only 必须搭配实验性 --module-coverage-profile")
    if args.probe_module_family and (
        args.module_coverage_profile not in DENSE_CLOSURE_PROFILES
        or not args.module_probe_only
    ):
        parser.error(
            "--probe-module-family 必须搭配 dense closure 实验 profile 和 --module-probe-only"
        )
    if (
        not args.structured_name.endswith(".pt")
        or "/" in args.structured_name
        or "\\" in args.structured_name
        or args.structured_name in {".pt", "..pt"}
    ):
        parser.error("--structured-name 必须是不含目录的有效 .pt 文件名")
    if args.oto_opset <= 0:
        parser.error("--oto-opset 必须为正整数")
    return args


def main() -> None:
    print("=" * 60, flush=True)
    print("程序启动 - YOLO11 AdaMCF 剪枝、微调与导出", flush=True)
    print(f"OTO 根目录: {OTO_ROOT}", flush=True)
    print("=" * 60, flush=True)

    args = parse_args()
    started = time.time()
    if args.mode == "audit":
        audit_model(args)
    elif args.mode in {"prune", "sparse-train"}:
        prune_model(args)
    elif args.mode == "rebuild-subnet":
        rebuild_structured_subnet(args)
    elif args.mode == "finetune":
        finetune_model(args)
    elif args.mode == "export":
        export_model(args)
    else:  # pragma: no cover - argparse 已限制 choices
        raise ValueError(f"未知模式：{args.mode}")
    print(f"[done] mode={args.mode}, elapsed_seconds={time.time() - started:.1f}")


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if os.environ.get("OTO_DEBUG_TRACEBACK") == "1":
            raise
        print_human_summary(
            "程序执行失败",
            [
                ("错误类型", type(exc).__name__),
                ("原因", str(exc)),
                ("调试方式", "设置 OTO_DEBUG_TRACEBACK=1 可显示完整 traceback"),
            ],
        )
        raise SystemExit(1)
