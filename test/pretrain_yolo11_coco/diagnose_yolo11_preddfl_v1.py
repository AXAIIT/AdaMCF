#!/usr/bin/env python
"""Y11-PredDFL-KD-v1 的只读诊断门禁。

本脚本用于回答 A1 蒸馏增益较小是否主要由选点语义或 T/损失标度造成。它不调用
Ultralytics Trainer、不创建 optimizer、不调用 backward()，也不保存模型 checkpoint。
学生与教师均处于 eval 模式；对学生只用 autograd.grad() 读取 task/KD 梯度范数，
并在结束时比对完整 state_dict 指纹，fail-closed 地证明模型状态未变化。

诊断数据使用 COCO train split 的固定顺序、验证式预处理（无 mosaic/随机增强）。这
保证 GT 几何覆盖可复现；它不是训练或准确率实验，不能用其结果替代 matched KD/control。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_NO_AUTO_INSTALL", "True")
os.environ.setdefault("ULTRALYTICS_SETTINGS", "0")

CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.tal import dist2bbox, make_anchors
from ultralytics.utils.torch_utils import unwrap_model
from ultralytics.utils import DEFAULT_CFG

import distill_yolo11_preddfl as preddfl


DIAGNOSTIC_ID = "Y11-PredDFL-v1-readonly-diagnostic-v1"
IMPLEMENTATION = "ultralytics_8_3_246_raw_detect_dfl_readonly_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(path_value: str, label: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在：{path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_fingerprint(model: nn.Module) -> str:
    """覆盖参数和 buffer 的内容指纹，用于证明只读诊断没有改变模型状态。"""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        contiguous = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def resolve_device(value: str) -> torch.device:
    normalized = str(value).strip()
    if normalized.lower() == "cpu":
        return torch.device("cpu")
    if normalized.isdigit():
        normalized = f"cuda:{normalized}"
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("请求 CUDA 诊断，但 torch.cuda.is_available() 为 False。")
        device = torch.device(normalized)
        torch.cuda.set_device(device)
        return device
    raise ValueError(f"--device 仅支持 cpu、N 或 cuda:N，实际为：{value}")


def restore_ultralytics_trainability(model: nn.Module) -> dict[str, Any]:
    """复现 v8.3.246 无 --freeze 时的可训练参数规则，但不创建 optimizer。

    从 `best.pt` 加载的推理模型会把所有参数置为 requires_grad=False；真实 Trainer
    会重新解冻浮点参数，并始终固定 `.dfl`。诊断需要同一梯度所有权，因此只恢复这个
    元数据，不改变任何参数值、buffer 或训练模式。
    """

    trainable: list[str] = []
    frozen: list[str] = []
    skipped_nonfloating: list[str] = []
    for name, parameter in model.named_parameters():
        if ".dfl" in name:
            parameter.requires_grad_(False)
            frozen.append(name)
        elif parameter.dtype.is_floating_point:
            parameter.requires_grad_(True)
            trainable.append(name)
        else:
            parameter.requires_grad_(False)
            skipped_nonfloating.append(name)
    return {
        "policy": "ultralytics_8_3_246_no_freeze_plus_always_freeze_.dfl",
        "trainable_parameter_count": len(trainable),
        "always_frozen_dfl_parameter_names": frozen,
        "nonfloating_parameter_names": skipped_nonfloating,
    }


def quantile_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p99": None}
    tensor = torch.tensor(values, dtype=torch.float64)
    q = torch.quantile(tensor, torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64))
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "p50": float(q[0].item()),
        "p90": float(q[1].item()),
        "p99": float(q[2].item()),
    }


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """返回 xyxy 两组 box 的成对 IoU；不依赖 torchvision。"""

    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (bottom_right - top_left).clamp_min(0)
    intersection = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0))
    return intersection / (area1[:, None] + area2[None, :] - intersection).clamp_min(1e-9)


def xywhn_to_xyxy(boxes: torch.Tensor, image_height: int, image_width: int) -> torch.Tensor:
    scale = boxes.new_tensor([image_width, image_height, image_width, image_height])
    xywh = boxes * scale
    center, wh = xywh[..., :2], xywh[..., 2:]
    return torch.cat((center - wh / 2, center + wh / 2), dim=-1)


def flatten_masks(masks: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([mask.squeeze(1).flatten(1) for mask in masks], dim=1)


def level_offsets(raw_layers: list[torch.Tensor]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start = 0
    for layer in raw_layers:
        end = start + int(layer.shape[-2] * layer.shape[-1])
        result.append((start, end))
        start = end
    return result


def decode_teacher_boxes(
    teacher_raw: list[torch.Tensor],
    *,
    reg_max: int,
    strides: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """严格按 v8DetectionLoss 的 DFL decode 合同得到像素级 xyxy box 和 anchor 中心。"""

    batch_size = int(teacher_raw[0].shape[0])
    distributions = torch.cat([layer.view(batch_size, 4 * reg_max + (layer.shape[1] - 4 * reg_max), -1) for layer in teacher_raw], 2)
    pred_dist = distributions[:, : 4 * reg_max].permute(0, 2, 1).contiguous()
    anchor_points, stride_tensor = make_anchors(teacher_raw, strides, 0.5)
    projection = torch.arange(reg_max, dtype=pred_dist.dtype, device=pred_dist.device)
    distances = pred_dist.view(batch_size, -1, 4, reg_max).softmax(3).matmul(projection)
    boxes_grid = dist2bbox(distances, anchor_points, xywh=False)
    return boxes_grid * stride_tensor, anchor_points * stride_tensor, stride_tensor


def kd_components(
    student_raw: list[torch.Tensor],
    teacher_raw: list[torch.Tensor],
    masks: list[torch.Tensor],
    *,
    reg_max: int,
    nc: int,
    temperature: float,
    cls_weight: float,
    dfl_weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """与 A1 完全同式的 KD 项；warmup 固定为 1，仅供诊断梯度标度。"""

    template = student_raw[0]
    cls_numerator = template.new_zeros(())
    cls_denominator = template.new_zeros(())
    dfl_numerator = template.new_zeros(())
    dfl_denominator = template.new_zeros(())
    level_records: list[dict[str, Any]] = []
    for level, (student, teacher, mask) in enumerate(zip(student_raw, teacher_raw, masks), start=3):
        selected = mask.squeeze(1).flatten(1).reshape(-1)
        valid_positions = selected.sum()
        if int(valid_positions.item()) == 0:
            level_records.append({"level": f"P{level}", "valid_positions": 0, "cls_raw": None, "dfl_raw": None})
            continue
        student_dfl = student[:, : 4 * reg_max].reshape(student.shape[0], 4, reg_max, *student.shape[-2:])
        teacher_dfl = teacher[:, : 4 * reg_max].reshape(teacher.shape[0], 4, reg_max, *teacher.shape[-2:])
        student_cls = student[:, 4 * reg_max : 4 * reg_max + nc]
        teacher_cls = teacher[:, 4 * reg_max : 4 * reg_max + nc]
        student_cls_selected = student_cls.permute(0, 2, 3, 1).reshape(-1, nc)[selected]
        teacher_cls_selected = teacher_cls.permute(0, 2, 3, 1).reshape(-1, nc)[selected]
        cls_sum = F.binary_cross_entropy_with_logits(
            student_cls_selected / temperature,
            (teacher_cls_selected / temperature).sigmoid(),
            reduction="none",
        ).sum()
        cls_norm = valid_positions.to(dtype=student.dtype) * nc
        student_dfl_selected = student_dfl.permute(0, 3, 4, 1, 2).reshape(-1, 4, reg_max)[selected]
        teacher_dfl_selected = teacher_dfl.permute(0, 3, 4, 1, 2).reshape(-1, 4, reg_max)[selected]
        dfl_sum = F.kl_div(
            F.log_softmax(student_dfl_selected / temperature, dim=2),
            F.softmax(teacher_dfl_selected / temperature, dim=2),
            reduction="none",
        ).sum(dim=2).sum()
        dfl_norm = valid_positions.to(dtype=student.dtype) * 4
        cls_numerator += cls_sum
        cls_denominator += cls_norm
        dfl_numerator += dfl_sum
        dfl_denominator += dfl_norm
        level_records.append(
            {
                "level": f"P{level}",
                "valid_positions": int(valid_positions.item()),
                "cls_raw": float((cls_sum / cls_norm.clamp_min(1.0)).detach().float().cpu().item()),
                "dfl_raw": float((dfl_sum / dfl_norm.clamp_min(1.0)).detach().float().cpu().item()),
            }
        )
    cls_loss = temperature**2 * cls_numerator / cls_denominator.clamp_min(1.0)
    dfl_loss = temperature**2 * dfl_numerator / dfl_denominator.clamp_min(1.0)
    kd_core = cls_weight * cls_loss + dfl_weight * dfl_loss
    kd_term = kd_core * student_raw[0].shape[0]
    return kd_term, {
        "temperature": float(temperature),
        "cls_kd_loss": float(cls_loss.detach().float().cpu().item()),
        "dfl_kd_loss": float(dfl_loss.detach().float().cpu().item()),
        "kd_term": float(kd_term.detach().float().cpu().item()),
        "level_losses": level_records,
    }


def gradient_norm(loss: torch.Tensor, parameters: list[nn.Parameter]) -> dict[str, Any]:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    squared_sum = loss.new_zeros((), dtype=torch.float64)
    nonzero_count = 0
    nonfinite_count = 0
    for gradient in gradients:
        if gradient is None:
            continue
        detached = gradient.detach()
        if not bool(torch.isfinite(detached).all().item()):
            nonfinite_count += 1
            continue
        if int(torch.count_nonzero(detached).item()) > 0:
            nonzero_count += 1
        squared_sum += detached.double().square().sum()
    return {
        "l2_norm": float(squared_sum.sqrt().cpu().item()),
        "nonzero_parameter_count": int(nonzero_count),
        "nonfinite_parameter_count": int(nonfinite_count),
        "parameter_count": int(len(parameters)),
    }


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in batch.items():
        output[key] = value.to(device, non_blocking=device.type == "cuda") if isinstance(value, torch.Tensor) else value
    output["img"] = output["img"].float() / 255.0
    return output


def build_readonly_loader(args: argparse.Namespace, stride: int):
    data = check_det_dataset(str(require_file(args.data, "数据 YAML")))
    cfg = get_cfg(
        DEFAULT_CFG,
        {
            "data": str(args.data),
            "imgsz": int(args.imgsz),
            "batch": int(args.batch),
            "workers": int(args.workers),
            "task": "detect",
            "rect": False,
            "cache": False,
            "fraction": 1.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "multi_scale": False,
        },
    )
    dataset = build_yolo_dataset(cfg, data["train"], int(args.batch), data, mode="val", rect=False, stride=stride)
    loader = build_dataloader(dataset, batch=int(args.batch), workers=int(args.workers), shuffle=False, rank=-1)
    return data, loader


def analyze_geometry(
    masks: list[torch.Tensor],
    teacher_boxes: torch.Tensor,
    anchor_centers: torch.Tensor,
    raw_layers: list[torch.Tensor],
    batch: dict[str, Any],
) -> dict[str, Any]:
    selected = flatten_masks(masks)
    offsets = level_offsets(raw_layers)
    image_height, image_width = int(batch["img"].shape[2]), int(batch["img"].shape[3])
    batch_indices = batch["batch_idx"].view(-1).long()
    gt_boxes_all = xywhn_to_xyxy(batch["bboxes"], image_height, image_width)

    gt_total = 0
    gt_zero_selected = 0
    selected_total = 0
    selected_outside_total = 0
    gt_covered_per_level = [0, 0, 0]
    selected_per_level = [0, 0, 0]
    gt_candidate_per_level = [0, 0, 0]
    best_iou_per_gt: list[float] = []
    selected_iou_values: list[float] = []
    per_image: list[dict[str, Any]] = []

    for image_index in range(int(selected.shape[0])):
        image_selected = selected[image_index]
        image_gt = gt_boxes_all[batch_indices == image_index]
        selected_count = int(image_selected.sum().item())
        selected_total += selected_count
        level_selected = [int(image_selected[start:end].sum().item()) for start, end in offsets]
        for level, value in enumerate(level_selected):
            selected_per_level[level] += value
        if image_gt.numel() == 0:
            selected_outside_total += selected_count
            per_image.append({"image_index": image_index, "gt_count": 0, "selected_count": selected_count, "zero_covered_gt": 0})
            continue

        gt_total += int(image_gt.shape[0])
        centers = anchor_centers
        inside_any = torch.zeros_like(image_selected, dtype=torch.bool)
        zero_covered = 0
        for gt in image_gt:
            inside = (
                (centers[:, 0] >= gt[0])
                & (centers[:, 0] <= gt[2])
                & (centers[:, 1] >= gt[1])
                & (centers[:, 1] <= gt[3])
            )
            inside_any |= inside
            selected_inside = image_selected & inside
            if not bool(selected_inside.any().item()):
                zero_covered += 1
                gt_zero_selected += 1
            else:
                selected_boxes = teacher_boxes[image_index, selected_inside]
                ious = box_iou(selected_boxes, gt.unsqueeze(0)).squeeze(1)
                best_iou_per_gt.append(float(ious.max().detach().cpu().item()))
                selected_iou_values.extend(float(value) for value in ious.detach().cpu().tolist())
            for level, (start, end) in enumerate(offsets):
                candidate_count = int(inside[start:end].sum().item())
                gt_candidate_per_level[level] += candidate_count
                if candidate_count > 0 and bool(selected_inside[start:end].any().item()):
                    gt_covered_per_level[level] += 1
        selected_outside_total += int((image_selected & ~inside_any).sum().item())
        per_image.append(
            {
                "image_index": image_index,
                "gt_count": int(image_gt.shape[0]),
                "selected_count": selected_count,
                "zero_covered_gt": zero_covered,
            }
        )

    return {
        "gt_count": gt_total,
        "zero_covered_gt_count": gt_zero_selected,
        "zero_covered_gt_ratio": None if gt_total == 0 else gt_zero_selected / gt_total,
        "selected_position_count": selected_total,
        "selected_outside_gt_count": selected_outside_total,
        "selected_outside_gt_ratio": None if selected_total == 0 else selected_outside_total / selected_total,
        "levels": [
            {
                "level": f"P{level + 3}",
                "selected_position_count": selected_per_level[level],
                "gt_candidate_count": gt_candidate_per_level[level],
                "gt_with_selected_candidate_count": gt_covered_per_level[level],
            }
            for level in range(3)
        ],
        "teacher_best_iou_per_covered_gt": quantile_summary(best_iou_per_gt),
        "teacher_iou_over_all_selected_gt_inside_positions": quantile_summary(selected_iou_values),
        "per_image": per_image,
    }


def summarize_batch(
    batch_index: int,
    student: nn.Module,
    teacher: nn.Module,
    criterion,
    batch: dict[str, Any],
    contract: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    student_output = student(batch["img"])
    student_raw = preddfl.raw_detect_layers(student_output, label="diagnostic student")
    with torch.no_grad():
        teacher_output = teacher(batch["img"])
        teacher_raw = [value.detach() for value in preddfl.raw_detect_layers(teacher_output, label="diagnostic teacher")]
    preddfl.validate_runtime_raw_contract(student_raw, teacher_raw, contract)
    reg_max = int(contract["detect"]["reg_max"])
    nc = int(contract["detect"]["nc"])
    masks, mask_stats = preddfl.build_teacher_quality_masks(
        teacher_raw,
        reg_max=reg_max,
        nc=nc,
        confidence_threshold=float(args.confidence_threshold),
        topk_per_image=int(args.topk_per_image),
    )
    teacher_boxes, anchor_centers, _stride_tensor = decode_teacher_boxes(
        teacher_raw,
        reg_max=reg_max,
        strides=torch.as_tensor(contract["detect"]["stride"], device=teacher_raw[0].device),
    )
    geometry = analyze_geometry(masks, teacher_boxes, anchor_centers, teacher_raw, batch)
    task_loss, task_items = criterion(student_output, batch)
    task_total = task_loss.sum()
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    task_gradient = gradient_norm(task_total, parameters)
    temperature_records: dict[str, Any] = {}
    for temperature in (1.0, 2.0):
        kd_term, kd_record = kd_components(
            student_raw,
            teacher_raw,
            masks,
            reg_max=reg_max,
            nc=nc,
            temperature=temperature,
            cls_weight=float(args.cls_weight),
            dfl_weight=float(args.dfl_weight),
        )
        kd_gradient = gradient_norm(kd_term, parameters)
        kd_record["kd_to_task_loss_ratio"] = float(kd_term.detach().float().cpu().item()) / max(
            abs(float(task_total.detach().float().cpu().item())), 1e-12
        )
        kd_record["kd_to_task_gradient_norm_ratio"] = kd_gradient["l2_norm"] / max(task_gradient["l2_norm"], 1e-12)
        kd_record["gradient"] = kd_gradient
        temperature_records[f"T{int(temperature)}"] = kd_record
    return {
        "batch_index": batch_index,
        "batch_size": int(batch["img"].shape[0]),
        "task_loss": float(task_total.detach().float().cpu().item()),
        "task_loss_items": [float(value) for value in task_items.detach().float().cpu().tolist()],
        "task_gradient": task_gradient,
        "mask": mask_stats,
        "geometry": geometry,
        "temperatures": temperature_records,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise RuntimeError("诊断没有产生任何 batch 记录。")
    gt_count = sum(int(record["geometry"]["gt_count"]) for record in records)
    zero_gt = sum(int(record["geometry"]["zero_covered_gt_count"]) for record in records)
    selected = sum(int(record["geometry"]["selected_position_count"]) for record in records)
    outside = sum(int(record["geometry"]["selected_outside_gt_count"]) for record in records)
    levels: list[dict[str, Any]] = []
    for level_index in range(3):
        source = [record["geometry"]["levels"][level_index] for record in records]
        levels.append(
            {
                "level": f"P{level_index + 3}",
                "selected_position_count": sum(int(item["selected_position_count"]) for item in source),
                "gt_candidate_count": sum(int(item["gt_candidate_count"]) for item in source),
                "gt_with_selected_candidate_count": sum(int(item["gt_with_selected_candidate_count"]) for item in source),
            }
        )
    return {
        "batch_count": len(records),
        "image_count": sum(int(record["batch_size"]) for record in records),
        "gt_count": gt_count,
        "zero_covered_gt_count": zero_gt,
        "zero_covered_gt_ratio": None if gt_count == 0 else zero_gt / gt_count,
        "selected_position_count": selected,
        "selected_outside_gt_count": outside,
        "selected_outside_gt_ratio": None if selected == 0 else outside / selected,
        "levels": levels,
        "mean_task_loss": sum(float(record["task_loss"]) for record in records) / len(records),
        "mean_T1_kd_to_task_loss_ratio": sum(float(record["temperatures"]["T1"]["kd_to_task_loss_ratio"]) for record in records) / len(records),
        "mean_T2_kd_to_task_loss_ratio": sum(float(record["temperatures"]["T2"]["kd_to_task_loss_ratio"]) for record in records) / len(records),
        "mean_T1_kd_to_task_gradient_ratio": sum(float(record["temperatures"]["T1"]["kd_to_task_gradient_norm_ratio"]) for record in records) / len(records),
        "mean_T2_kd_to_task_gradient_ratio": sum(float(record["temperatures"]["T2"]["kd_to_task_gradient_norm_ratio"]) for record in records) / len(records),
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "batch_index",
        "batch_size",
        "task_loss",
        "gt_count",
        "zero_covered_gt_count",
        "zero_covered_gt_ratio",
        "selected_position_count",
        "selected_outside_gt_count",
        "selected_outside_gt_ratio",
        "t1_kd_to_task_loss_ratio",
        "t2_kd_to_task_loss_ratio",
        "t1_kd_to_task_gradient_ratio",
        "t2_kd_to_task_gradient_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            geometry = record["geometry"]
            writer.writerow(
                {
                    "batch_index": record["batch_index"],
                    "batch_size": record["batch_size"],
                    "task_loss": record["task_loss"],
                    "gt_count": geometry["gt_count"],
                    "zero_covered_gt_count": geometry["zero_covered_gt_count"],
                    "zero_covered_gt_ratio": geometry["zero_covered_gt_ratio"],
                    "selected_position_count": geometry["selected_position_count"],
                    "selected_outside_gt_count": geometry["selected_outside_gt_count"],
                    "selected_outside_gt_ratio": geometry["selected_outside_gt_ratio"],
                    "t1_kd_to_task_loss_ratio": record["temperatures"]["T1"]["kd_to_task_loss_ratio"],
                    "t2_kd_to_task_loss_ratio": record["temperatures"]["T2"]["kd_to_task_loss_ratio"],
                    "t1_kd_to_task_gradient_ratio": record["temperatures"]["T1"]["kd_to_task_gradient_norm_ratio"],
                    "t2_kd_to_task_gradient_ratio": record["temperatures"]["T2"]["kd_to_task_gradient_norm_ratio"],
                }
            )


def run(args: argparse.Namespace) -> None:
    teacher_weights = require_file(args.teacher, "教师权重")
    student_weights = require_file(args.student, "学生权重")
    require_file(args.data, "数据 YAML")
    output = Path(args.output).expanduser().resolve()
    csv_output = output.with_suffix(".csv")
    if args.no_overwrite and (output.exists() or csv_output.exists()):
        raise FileExistsError(f"诊断产物已存在，禁止覆盖：{output} / {csv_output}")
    device = resolve_device(args.device)
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    student = YOLO(str(student_weights)).model.to(device).float().eval()
    teacher = YOLO(str(teacher_weights)).model.to(device).float().eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    trainability = restore_ultralytics_trainability(student)
    runtime_cfg = get_cfg(DEFAULT_CFG, {"data": str(args.data), "imgsz": int(args.imgsz), "task": "detect"})
    student.args = runtime_cfg
    teacher.args = runtime_cfg
    if hasattr(student, "criterion"):
        student.criterion = None
    if hasattr(teacher, "criterion"):
        teacher.criterion = None
    contract = preddfl.build_prediction_contract(student, teacher, imgsz=int(args.imgsz))
    _index, detect, _f = preddfl._detect_metadata(student, label="diagnostic student")
    stride = max(int(detect.stride.max().item()), 32)
    data, loader = build_readonly_loader(args, stride)
    criterion = unwrap_model(student).init_criterion()
    if not any(parameter.requires_grad for parameter in student.parameters()):
        raise RuntimeError("学生没有任何原始可训练参数，无法进行 KD/任务梯度标度诊断。")
    student_before = state_fingerprint(student)
    teacher_before = state_fingerprint(teacher)
    records: list[dict[str, Any]] = []
    for batch_index, source_batch in enumerate(loader):
        if batch_index >= int(args.max_batches):
            break
        batch = move_batch_to_device(source_batch, device)
        records.append(summarize_batch(batch_index, student, teacher, criterion, batch, contract, args))
        print(
            "[diagnose] "
            f"batch={batch_index + 1}/{args.max_batches}, "
            f"gt_zero={records[-1]['geometry']['zero_covered_gt_ratio']}, "
            f"outside={records[-1]['geometry']['selected_outside_gt_ratio']}, "
            f"T2_grad_ratio={records[-1]['temperatures']['T2']['kd_to_task_gradient_norm_ratio']:.4f}",
            flush=True,
        )
    student_after = state_fingerprint(student)
    teacher_after = state_fingerprint(teacher)
    state_check = {
        "status": "pass" if student_before == student_after and teacher_before == teacher_after else "fail",
        "student_before": student_before,
        "student_after": student_after,
        "teacher_before": teacher_before,
        "teacher_after": teacher_after,
    }
    report = {
        "status": "pass" if state_check["status"] == "pass" else "fail",
        "time_utc": utc_now(),
        "diagnostic_id": DIAGNOSTIC_ID,
        "implementation": IMPLEMENTATION,
        "read_only_contract": {
            "trainer_created": False,
            "optimizer_created": False,
            "backward_called": False,
            "parameter_update": False,
            "checkpoint_saved": False,
            "student_eval_mode": True,
            "teacher_eval_mode": True,
            "gradient_method": "torch.autograd.grad_only",
            "student_trainability_policy": trainability["policy"],
            "data_scope": "COCO train split, fixed order, validation-style preprocessing without mosaic/random augmentation",
        },
        "inputs": {
            "teacher": {"path": str(teacher_weights), "sha256": sha256_file(teacher_weights)},
            "student": {"path": str(student_weights), "sha256": sha256_file(student_weights)},
            "data_yaml": str(Path(args.data).resolve()),
            "dataset_train_path": str(data["train"]),
            "imgsz": int(args.imgsz),
            "batch": int(args.batch),
            "max_batches": int(args.max_batches),
            "workers": int(args.workers),
            "device": str(device),
            "seed": int(args.seed),
            "confidence_threshold": float(args.confidence_threshold),
            "topk_per_image": int(args.topk_per_image),
            "cls_weight": float(args.cls_weight),
            "dfl_weight": float(args.dfl_weight),
        },
        "prediction_contract": contract,
        "student_trainability": trainability,
        "state_immutability": state_check,
        "aggregate": aggregate(records),
        "batches": records,
        "interpretation": {
            "not_a_training_result": True,
            "required_follow_up": "依据 GT 覆盖、GT 外选点、教师 IoU 与 T/梯度比例决定 A2 或 A1 T/lambda 小消融；任何训练仍需新的 matched control。",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(csv_output, records)
    if report["status"] != "pass":
        raise RuntimeError(f"只读状态合同失败：{output}")
    print("=" * 72, flush=True)
    print("Y11-PredDFL-v1 只读诊断完成", flush=True)
    print(f"状态: {report['status']}", flush=True)
    print(f"JSON: {output}", flush=True)
    print(f"CSV : {csv_output}", flush=True)
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True, help="只读诊断 JSON 路径；同目录生成 CSV。")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cls-weight", type=float, default=0.5)
    parser.add_argument("--dfl-weight", type=float, default=0.25)
    parser.add_argument("--confidence-threshold", type=float, default=0.10)
    parser.add_argument("--topk-per-image", type=int, default=100)
    parser.add_argument("--no-overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_batches <= 0 or args.batch <= 0:
        raise ValueError("--batch 和 --max-batches 必须为正整数。")
    run(args)


if __name__ == "__main__":
    main()
