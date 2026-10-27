#!/usr/bin/env python
"""YOLO11 物理剪枝学生的预测级 KD / 特征 KD，以及同入口 no-KD 对照。

本脚本固定面向服务器 ``ultralytics==8.3.246``，不修改 OTO/HESSO。
``control`` 与 ``train`` 使用同一学生起点、训练器、保存逻辑和验收链路：

* ``control``：只保留原始 YOLO 检测损失，作为 matched no-KD continuation；
* ``train``：可独立启用预测级分类 + DFL、Neck-SL2 或完整 FGD 特征蒸馏；
  首轮完整 FGD 与其他 KD 分支互斥，保证 matched 实验的因果可解释性。

两种模式必须以相同的 seed、数据、训练参数和 student checkpoint 成对运行；唯一
允许的训练目标差异是 KD 项。最终 checkpoint 永远只包含物理学生，不能包含教师、
criterion 或训练期辅助模块。
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
import sys
import time
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
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

import distill_yolo11_necksl2 as neck
import yolo11_fgd as fgd
from yolo11_qualityfg_mask import (
    DEFAULT_QUALITY_TOPK_PER_GT,
    QUALITYFG_MASK_CONTRACT,
    build_qualityfg_teacher_masks,
)


DISTILLATION_ID = "Y11-PredDFL-KD-v1"
QUALITYFG_DISTILLATION_ID = "Y11-QualityFG-PredDFL-KD-v2"
IMPLEMENTATION = "legacy_8_3_246_preddfl_paired_control_v1"
QUALITYFG_IMPLEMENTATION = "legacy_8_3_246_preddfl_paired_control_v2_optional_qualityfg"
CONTROL_ID = "Y11-noKD-continuation-control-v1"
QUALITYFG_CONTROL_ID = "Y11-noKD-continuation-control-qualityfg-v2"
HYBRID_DISTILLATION_ID = "Y11-Hybrid-QualityFG-PredDFL-NeckSL2-KD-v1"
HYBRID_IMPLEMENTATION = "legacy_8_3_246_multibranch_qualityfg_preddfl_necksl2_v1"
FGD_DISTILLATION_ID = "Y11-FGD-Full-KD-v1"
FGD_COMPONENT_ABLATION_ID = "Y11-FGD-ComponentAblation-v1"
FGD_IMPLEMENTATION = "legacy_8_3_246_official_fgd_featureloss_area_mask_attention_gc_relation_v3_component_switches"


def resolved_fgd_component_flags(args: argparse.Namespace) -> dict[str, bool]:
    """读取四项 FGD 子损失开关；默认全开以保持历史完整 FGD 路径。"""

    return fgd.fgd_component_flags(
        enable_foreground=args.fgd_enable_foreground,
        enable_background=args.fgd_enable_background,
        enable_mask=args.fgd_enable_mask,
        enable_relation=args.fgd_enable_relation,
    )


def fgd_loss_contract(component_flags: dict[str, bool]) -> str:
    """把当前实际参与反向的 FGD 项写成可追溯合同。"""

    pieces = []
    if component_flags["foreground"]:
        pieces.append("alpha*L_fg")
    if component_flags["background"]:
        pieces.append("beta*L_bg")
    if component_flags["mask"]:
        pieces.append("gamma*L_mask")
    if component_flags["relation"]:
        pieces.append("lambda*L_relation")
    if not pieces:
        raise ValueError("FGD 子损失开关不能全部关闭。")
    return "L_total=L_det + fgd_warmup*lambda_fgd*(" + " + ".join(pieces) + ")*batch_size"


def fgd_contract_name(component_flags: dict[str, bool]) -> str:
    """完整官方配置保留原合同名，部分配置使用显式消融合同名。"""

    return fgd.FGD_FULL_CONTRACT if all(component_flags.values()) else fgd.FGD_COMPONENT_ABLATION_CONTRACT


def scalar_tensors_to_floats(values: dict[str, torch.Tensor | float]) -> dict[str, float]:
    """把训练期标量合并为一次 D2H 同步，避免逐指标阻塞 CUDA 流。"""

    keys = list(values)
    first_tensor = next((value for value in values.values() if isinstance(value, torch.Tensor)), None)
    if first_tensor is None:
        return {key: float(value) for key, value in values.items()}
    packed = torch.stack(
        [
            value.detach().float().reshape(())
            if isinstance(value, torch.Tensor)
            else first_tensor.new_tensor(float(value), dtype=torch.float32)
            for value in values.values()
        ]
    )
    numbers = packed.cpu().tolist()
    return {key: float(number) for key, number in zip(keys, numbers)}


def raw_detect_layers(value: Any, *, label: str) -> list[torch.Tensor]:
    """严格取得 Detect 的三尺度原始输出，不允许使用 decoded 或 NMS 结果。"""

    raw = neck.extract_raw_detect_outputs(value)
    if len(raw) != 3:
        raise RuntimeError(f"{label}原始 Detect 输出必须恰有三个尺度，实际为 {len(raw)}。")
    if not all(tensor.ndim == 4 for tensor in raw):
        raise RuntimeError(f"{label}原始 Detect 输出必须全部为四维 Tensor。")
    return raw


def _detect_metadata(model: nn.Module, *, label: str) -> tuple[int, nn.Module, list[int]]:
    raw = unwrap_model(model)
    index, detect = neck.find_detect_module(raw)
    layers = getattr(raw, "model", None)
    if layers is None:
        raise RuntimeError(f"{label}模型缺少 model 层序列。")
    detect_f = [neck.normalize_layer_index(int(value), len(layers)) for value in list(detect.f)]
    if len(detect_f) != 3:
        raise RuntimeError(f"{label} Detect.f 必须含三个尺度，实际为 {detect_f}。")
    return index, detect, detect_f


def build_prediction_contract(
    student_template: nn.Module,
    teacher_template: nn.Module,
    *,
    imgsz: int,
    experiment_id: str = DISTILLATION_ID,
) -> dict[str, Any]:
    """冻结 raw Detect 输出的语义、尺度和空间网格合同。"""

    student = copy.deepcopy(unwrap_model(student_template)).cpu().float().eval()
    teacher = copy.deepcopy(unwrap_model(teacher_template)).cpu().float().eval()
    student_index, student_detect, student_f = _detect_metadata(student, label="student")
    teacher_index, teacher_detect, teacher_f = _detect_metadata(teacher, label="teacher")

    student_nc, teacher_nc = int(student_detect.nc), int(teacher_detect.nc)
    student_reg_max, teacher_reg_max = int(student_detect.reg_max), int(teacher_detect.reg_max)
    if student_nc != teacher_nc or student_reg_max != teacher_reg_max:
        raise RuntimeError(
            "师生 Detect 语义不一致："
            f"student(nc={student_nc},reg_max={student_reg_max})，"
            f"teacher(nc={teacher_nc},reg_max={teacher_reg_max})。"
        )
    if student_f != teacher_f:
        raise RuntimeError(f"师生 Detect.f 不一致：student={student_f}，teacher={teacher_f}。")
    student_stride = [float(value) for value in student_detect.stride.detach().cpu().flatten().tolist()]
    teacher_stride = [float(value) for value in teacher_detect.stride.detach().cpu().flatten().tolist()]
    if student_stride != teacher_stride:
        raise RuntimeError(f"师生 Detect stride 不一致：student={student_stride}，teacher={teacher_stride}。")

    with torch.no_grad():
        image = torch.zeros(1, 3, imgsz, imgsz)
        student_raw = raw_detect_layers(student(image), label="student contract")
        teacher_raw = raw_detect_layers(teacher(image), label="teacher contract")

    no = 4 * teacher_reg_max + teacher_nc
    levels: list[dict[str, Any]] = []
    for level, (student_tensor, teacher_tensor) in enumerate(zip(student_raw, teacher_raw), start=3):
        if student_tensor.shape != teacher_tensor.shape:
            raise RuntimeError(
                f"P{level} raw Detect shape 不一致：student={neck.tensor_shape(student_tensor)}，"
                f"teacher={neck.tensor_shape(teacher_tensor)}。"
            )
        if student_tensor.shape[1] != no:
            raise RuntimeError(
                f"P{level} raw Detect 语义宽度错误：{neck.tensor_shape(student_tensor)}，期望 no={no}。"
            )
        levels.append(
            {
                "level": f"P{level}",
                "student_raw_shape": neck.tensor_shape(student_tensor),
                "teacher_raw_shape": neck.tensor_shape(teacher_tensor),
                "spatial_shape": [int(student_tensor.shape[-2]), int(student_tensor.shape[-1])],
                "stride": student_stride[level - 3],
            }
        )

    return {
        "status": "pass",
        "distillation_id": experiment_id,
        "detect": {
            "student_index": student_index,
            "teacher_index": teacher_index,
            "student_f": student_f,
            "teacher_f": teacher_f,
            "nc": teacher_nc,
            "reg_max": teacher_reg_max,
            "no": no,
            "stride": student_stride,
        },
        "raw_levels": levels,
        "input_shape": [1, 3, imgsz, imgsz],
        "contract": "three_level_raw_detect_logits_same_semantics_same_grid_v1",
    }


def validate_runtime_raw_contract(
    student_raw: list[torch.Tensor],
    teacher_raw: list[torch.Tensor] | None,
    contract: dict[str, Any],
) -> None:
    """运行期 fail-closed：禁止 reshape、插值或广播来掩盖 Detect 不匹配。"""

    records = list(contract["raw_levels"])
    if len(student_raw) != len(records):
        raise RuntimeError(f"student raw 输出尺度数变化：{len(student_raw)}，期望 {len(records)}。")
    if teacher_raw is not None and len(teacher_raw) != len(records):
        raise RuntimeError(f"teacher raw 输出尺度数变化：{len(teacher_raw)}，期望 {len(records)}。")
    expected_no = int(contract["detect"]["no"])
    for index, record in enumerate(records):
        student_tensor = student_raw[index]
        expected_spatial = tuple(record["spatial_shape"])
        if student_tensor.ndim != 4 or student_tensor.shape[1] != expected_no or tuple(student_tensor.shape[-2:]) != expected_spatial:
            raise RuntimeError(
                f"{record['level']} student raw 合同失败：实际={neck.tensor_shape(student_tensor)}，"
                f"期望 channels={expected_no}, spatial={expected_spatial}。"
            )
        if teacher_raw is not None:
            teacher_tensor = teacher_raw[index]
            if teacher_tensor.shape != student_tensor.shape:
                raise RuntimeError(
                    f"{record['level']} 师生 raw 运行期 shape 不一致："
                    f"student={neck.tensor_shape(student_tensor)}，teacher={neck.tensor_shape(teacher_tensor)}。"
                )


def optimizer_ownership_report(optimizer, student: nn.Module, teacher: nn.Module) -> dict[str, Any]:
    """验证所有可训练 student 参数仅出现一次，教师参数不出现。"""

    ownership: dict[int, int] = {}
    for group in optimizer.param_groups:
        for parameter in group.get("params", []):
            ownership[id(parameter)] = ownership.get(id(parameter), 0) + 1

    student_parameters = [(name, parameter) for name, parameter in student.named_parameters() if parameter.requires_grad]
    teacher_parameters = [(name, parameter) for name, parameter in teacher.named_parameters()]
    student_missing_or_duplicated = [
        name for name, parameter in student_parameters if ownership.get(id(parameter), 0) != 1
    ]
    teacher_in_optimizer = [name for name, parameter in teacher_parameters if ownership.get(id(parameter), 0) != 0]
    return {
        "status": "pass" if not student_missing_or_duplicated and not teacher_in_optimizer else "fail",
        "student_trainable_parameter_count": len(student_parameters),
        "student_parameters_missing_or_duplicated": student_missing_or_duplicated,
        "teacher_parameter_count": len(teacher_parameters),
        "teacher_parameters_in_optimizer": teacher_in_optimizer,
        "expected_student_ownership": 1,
        "expected_teacher_ownership": 0,
    }


def _quantile_summary(scores: torch.Tensor) -> dict[str, float]:
    flat = scores.detach().float().flatten()
    if flat.numel() == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p99": float("nan")}
    quantiles = torch.quantile(flat, torch.tensor([0.50, 0.90, 0.99], device=flat.device))
    return {
        "mean": float(flat.mean().cpu().item()),
        "p50": float(quantiles[0].cpu().item()),
        "p90": float(quantiles[1].cpu().item()),
        "p99": float(quantiles[2].cpu().item()),
    }


def build_teacher_quality_masks(
    teacher_raw: list[torch.Tensor],
    *,
    reg_max: int,
    nc: int,
    confidence_threshold: float,
    topk_per_image: int,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """按每图跨尺度 threshold OR Top-K 建立无梯度有效位置掩码。"""

    score_maps = [
        raw[:, 4 * reg_max : 4 * reg_max + nc].detach().sigmoid().amax(dim=1)
        for raw in teacher_raw
    ]
    flattened = [score.flatten(1) for score in score_maps]
    all_scores = torch.cat(flattened, dim=1)
    batch_size, positions_per_image = all_scores.shape
    if positions_per_image <= 0:
        raise RuntimeError("教师 score map 没有任何位置，无法构造 KD 掩码。")
    topk = min(int(topk_per_image), int(positions_per_image))
    threshold_mask = all_scores >= float(confidence_threshold)
    topk_mask = torch.zeros_like(threshold_mask)
    if topk > 0:
        topk_indices = all_scores.topk(topk, dim=1, largest=True, sorted=False).indices
        topk_mask.scatter_(1, topk_indices, True)
    selected = threshold_mask | topk_mask

    masks: list[torch.Tensor] = []
    level_offset = 0
    selected_per_level: list[float] = []
    for score, flat in zip(score_maps, flattened):
        count = flat.shape[1]
        mask = selected[:, level_offset : level_offset + count].reshape_as(score).unsqueeze(1)
        masks.append(mask)
        selected_per_level.append(float(mask.sum().detach().cpu().item()))
        level_offset += count
    if level_offset != positions_per_image:
        raise RuntimeError("PredDFL Top-K 掩码分段长度不守恒。")

    score_summary = _quantile_summary(all_scores)
    threshold_count = threshold_mask.sum().detach().float()
    selected_count = selected.sum().detach().float()
    topk_added = (selected & ~threshold_mask).sum().detach().float()
    return masks, {
        "teacher_score_mean": score_summary["mean"],
        "teacher_score_p50": score_summary["p50"],
        "teacher_score_p90": score_summary["p90"],
        "teacher_score_p99": score_summary["p99"],
        "mask_positions_per_image": float(positions_per_image),
        "mask_threshold_selected_per_image": float(threshold_count.cpu().item()) / batch_size,
        "mask_topk_added_per_image": float(topk_added.cpu().item()) / batch_size,
        "mask_selected_per_image": float(selected_count.cpu().item()) / batch_size,
        "mask_selected_ratio": float(selected_count.cpu().item()) / float(batch_size * positions_per_image),
        "mask_topk_per_image": float(topk),
        "mask_empty_images": float((selected.sum(dim=1) == 0).sum().detach().cpu().item()),
        "mask_p3_selected_per_batch": selected_per_level[0],
        "mask_p4_selected_per_batch": selected_per_level[1],
        "mask_p5_selected_per_batch": selected_per_level[2],
    }


class PredDFLCriterion:
    """保留原检测损失，选择性增加分类软标签和 DFL 分布蒸馏。"""

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        contract: dict[str, Any],
        *,
        kd_enabled: bool,
        cls_weight: float,
        dfl_weight: float,
        cls_temperature: float,
        dfl_temperature: float,
        confidence_threshold: float,
        topk_per_image: int,
        selection_mode: str,
        quality_topk_per_gt: int,
        warmup_epochs: int,
        trainer,
        admission_probe: bool,
        record_metrics: bool = True,
    ) -> None:
        self.student = unwrap_model(student)
        self.teacher = teacher.eval()
        self.contract = contract
        self.kd_enabled = bool(kd_enabled)
        self.cls_weight = float(cls_weight)
        self.dfl_weight = float(dfl_weight)
        self.cls_temperature = float(cls_temperature)
        self.dfl_temperature = float(dfl_temperature)
        self.confidence_threshold = float(confidence_threshold)
        self.topk_per_image = int(topk_per_image)
        self.selection_mode = str(selection_mode)
        self.quality_topk_per_gt = int(quality_topk_per_gt)
        self.warmup_epochs = int(warmup_epochs)
        self.trainer = trainer
        self.admission_probe = bool(admission_probe)
        self.record_metrics = bool(record_metrics)
        self.base_criterion = self.student.init_criterion()
        self.first_kd_gradient: dict[str, Any] = {"status": "not_applicable" if not kd_enabled else "pending"}
        self.last_metrics: dict[str, float] = {}
        self.last_teacher_raw: list[torch.Tensor] | None = None
        if self.selection_mode not in {"global_score_v1", "quality_fg_v2"}:
            raise ValueError(f"未知 PredDFL selection_mode：{self.selection_mode}")

    def close(self) -> None:
        """和 Neck-SL2 接口保持一致；本路线没有 forward hook。"""

        self.last_teacher_raw = None

    def _publish_metrics(self, values: dict[str, float]) -> None:
        self.last_metrics = dict(values)
        if self.record_metrics:
            self.trainer.kd_meter.update(values)

    def warmup_factor(self) -> float:
        if not self.kd_enabled or self.warmup_epochs <= 0:
            return 1.0 if self.kd_enabled else 0.0
        return min(1.0, float(int(self.trainer.epoch) + 1) / float(self.warmup_epochs))

    def _capture_first_kd_gradient(self, kd_term: torch.Tensor) -> None:
        if not self.admission_probe or self.first_kd_gradient.get("status") != "pending":
            return
        # QualityFG 的纯背景 batch 合法地没有任何 GT 选择点；等待首个实际 KD batch，
        # 不能把零 KD 项误报为梯度链路失败。
        if float(kd_term.detach().abs().cpu().item()) == 0.0:
            return
        trainable = [(name, parameter) for name, parameter in self.student.named_parameters() if parameter.requires_grad]
        gradients = torch.autograd.grad(
            kd_term,
            [parameter for _name, parameter in trainable],
            retain_graph=True,
            allow_unused=True,
        )
        nonzero_names: list[str] = []
        non_finite_names: list[str] = []
        for (name, _parameter), gradient in zip(trainable, gradients):
            if gradient is None:
                continue
            if not bool(torch.isfinite(gradient).all().item()):
                non_finite_names.append(name)
            elif torch.count_nonzero(gradient).item() > 0:
                nonzero_names.append(name)
        self.first_kd_gradient = {
            "status": "pass" if nonzero_names and not non_finite_names else "fail",
            "trainable_student_parameter_count": len(trainable),
            "student_parameters_with_nonzero_kd_gradient": len(nonzero_names),
            "nonzero_kd_gradient_parameter_names": nonzero_names,
            "non_finite_kd_gradient_parameter_names": non_finite_names,
        }

    def __call__(self, preds, batch):
        task_loss, task_items = self.base_criterion(preds, batch)
        # v8DetectionLoss 返回按 batch_size 缩放的 [box, cls, dfl] 向量；Trainer
        # 默认对其求和后反向，因此 KD 项也必须叠加到总 task scalar 上。
        task_loss_total = task_loss.sum()
        student_raw = raw_detect_layers(preds, label="student train")
        validate_runtime_raw_contract(student_raw, None, self.contract)

        if not self.kd_enabled:
            latest = {
                "task_loss": float(task_loss_total.detach().float().cpu().item()),
                "cls_kd_loss": 0.0,
                "dfl_kd_loss": 0.0,
                "kd_term": 0.0,
                "total_loss": float(task_loss_total.detach().float().cpu().item()),
                "kd_enabled": 0.0,
                "warmup_factor": 0.0,
            }
            self._publish_metrics(latest)
            return task_loss_total, task_items

        with torch.no_grad():
            teacher_output = self.teacher(batch["img"])
            teacher_raw = [tensor.detach() for tensor in raw_detect_layers(teacher_output, label="teacher train")]
        self.last_teacher_raw = teacher_raw
        validate_runtime_raw_contract(student_raw, teacher_raw, self.contract)

        reg_max = int(self.contract["detect"]["reg_max"])
        nc = int(self.contract["detect"]["nc"])
        if self.selection_mode == "global_score_v1":
            # A1 行为必须保持逐项等价：既不改变选点，也不改变分母和损失求和顺序。
            masks, mask_stats = build_teacher_quality_masks(
                teacher_raw,
                reg_max=reg_max,
                nc=nc,
                confidence_threshold=self.confidence_threshold,
                topk_per_image=self.topk_per_image,
            )
            position_weights = None
        else:
            masks, position_weights, mask_stats = build_qualityfg_teacher_masks(
                teacher_raw,
                batch,
                reg_max=reg_max,
                nc=nc,
                strides=list(self.contract["detect"]["stride"]),
                topk_per_gt=self.quality_topk_per_gt,
            )
        cls_numerator = task_loss_total.new_zeros(())
        cls_denominator = task_loss_total.new_zeros(())
        dfl_numerator = task_loss_total.new_zeros(())
        dfl_denominator = task_loss_total.new_zeros(())
        level_cls_losses: list[torch.Tensor] = []
        level_dfl_losses: list[torch.Tensor] = []
        for level_index, (student_tensor, teacher_tensor, mask) in enumerate(zip(student_raw, teacher_raw, masks)):
            selected = mask.squeeze(1).flatten(1).reshape(-1)
            valid_positions = selected.sum()
            if self.selection_mode == "global_score_v1" and valid_positions.item() <= 0:
                raise RuntimeError("PredDFL 掩码出现空位置；Top-K 合同应保证每图至少保留一个位置。")
            student_dfl = student_tensor[:, : 4 * reg_max].reshape(
                student_tensor.shape[0], 4, reg_max, *student_tensor.shape[-2:]
            )
            teacher_dfl = teacher_tensor[:, : 4 * reg_max].reshape(
                teacher_tensor.shape[0], 4, reg_max, *teacher_tensor.shape[-2:]
            )
            student_cls = student_tensor[:, 4 * reg_max : 4 * reg_max + nc]
            teacher_cls = teacher_tensor[:, 4 * reg_max : 4 * reg_max + nc]

            # 先选位置，再构造分类/DFL 概率张量。这样既不会让背景位置进入 KD，
            # 也避免为全部 8400 个网格保留不必要的 softmax/BCE 中间张量。
            student_cls_selected = student_cls.permute(0, 2, 3, 1).reshape(-1, nc)[selected]
            teacher_cls_selected = teacher_cls.permute(0, 2, 3, 1).reshape(-1, nc)[selected]
            cls_per_element = F.binary_cross_entropy_with_logits(
                student_cls_selected / self.cls_temperature,
                (teacher_cls_selected / self.cls_temperature).sigmoid(),
                reduction="none",
            )
            if position_weights is None:
                cls_sum = cls_per_element.sum()
                cls_norm = valid_positions.to(dtype=student_tensor.dtype) * nc
            else:
                selected_weights = position_weights[level_index].squeeze(1).flatten(1).reshape(-1)[selected]
                selected_weights = selected_weights.to(dtype=student_tensor.dtype)
                cls_sum = (cls_per_element.sum(dim=1) * selected_weights).sum()
                cls_norm = selected_weights.sum() * nc
            cls_numerator = cls_numerator + cls_sum
            cls_denominator = cls_denominator + cls_norm

            student_dfl_selected = student_dfl.permute(0, 3, 4, 1, 2).reshape(-1, 4, reg_max)[selected]
            teacher_dfl_selected = teacher_dfl.permute(0, 3, 4, 1, 2).reshape(-1, 4, reg_max)[selected]
            dfl_per_bin = F.kl_div(
                F.log_softmax(student_dfl_selected / self.dfl_temperature, dim=2),
                F.softmax(teacher_dfl_selected / self.dfl_temperature, dim=2),
                reduction="none",
            )
            if position_weights is None:
                dfl_sum = dfl_per_bin.sum(dim=2).sum()
                dfl_norm = valid_positions.to(dtype=student_tensor.dtype) * 4
            else:
                dfl_per_position = dfl_per_bin.sum(dim=2).sum(dim=1)
                dfl_sum = (dfl_per_position * selected_weights).sum()
                dfl_norm = selected_weights.sum() * 4
            dfl_numerator = dfl_numerator + dfl_sum
            dfl_denominator = dfl_denominator + dfl_norm
            level_cls_losses.append(cls_sum / cls_norm.clamp_min(1.0))
            level_dfl_losses.append(dfl_sum / dfl_norm.clamp_min(1.0))

        cls_kd_loss = self.cls_temperature**2 * cls_numerator / cls_denominator.clamp_min(1.0)
        dfl_kd_loss = self.dfl_temperature**2 * dfl_numerator / dfl_denominator.clamp_min(1.0)
        warmup = self.warmup_factor()
        kd_core = self.cls_weight * cls_kd_loss + self.dfl_weight * dfl_kd_loss
        kd_term = warmup * kd_core * batch["img"].shape[0]
        self._capture_first_kd_gradient(kd_term)
        total_loss = task_loss_total + kd_term
        latest = scalar_tensors_to_floats(
            {
                "task_loss": task_loss_total,
                "cls_kd_loss": cls_kd_loss,
                "dfl_kd_loss": dfl_kd_loss,
                "kd_term": kd_term,
                "total_loss": total_loss,
                "kd_enabled": 1.0,
                "warmup_factor": warmup,
                "effective_cls_weight": warmup * self.cls_weight,
                "effective_dfl_weight": warmup * self.dfl_weight,
                "p3_cls_raw_loss": level_cls_losses[0],
                "p4_cls_raw_loss": level_cls_losses[1],
                "p5_cls_raw_loss": level_cls_losses[2],
                "p3_dfl_raw_loss": level_dfl_losses[0],
                "p4_dfl_raw_loss": level_dfl_losses[1],
                "p5_dfl_raw_loss": level_dfl_losses[2],
            }
        )
        latest.update(mask_stats)
        self._publish_metrics(latest)
        return total_loss, task_items


class PredDFLEpochReporter(neck.core.EpochReporter):
    """在既有每 epoch 摘要中加入 control / PredDFL 分项。"""

    def __init__(self, phase: str) -> None:
        super().__init__(phase)
        self._pending: dict[str, Any] | None = None

    def on_fit_epoch_end(self, trainer) -> None:
        if getattr(trainer, "rank", -1) in {-1, 0} and self.last_epoch != int(trainer.epoch) + 1:
            self._pending = trainer.kd_meter.consume_epoch()
        try:
            super().on_fit_epoch_end(trainer)
        finally:
            self._pending = None

    def _save_records(self) -> None:
        if self.records and self._pending is not None:
            self.records[-1]["distillation"] = self._pending
        super()._save_records()

    def _print_record(self, record: dict[str, Any]) -> None:
        super()._print_record(record)
        kd = record.get("distillation", {})
        if not kd:
            return
        if kd.get("kd_enabled", 0.0) < 0.5 and kd.get("fgd_enabled", 0.0) < 0.5:
            print(
                f"no-KD control: task={kd.get('task_loss', float('nan')):.6f}, "
                f"total={kd.get('total_loss', float('nan')):.6f}",
                flush=True,
            )
            return
        if kd.get("prediction_kd_enabled", 1.0) < 0.5:
            return
        mode_label = "QualityFG-PredDFL-KD" if "qualityfg_gt_total" in kd else "PredDFL-KD"
        selection_detail = (
            f", gt={kd.get('qualityfg_gt_total', float('nan')):.0f}, "
            f"fallback={kd.get('qualityfg_gt_fallback_nearest_p3', float('nan')):.0f}, "
            f"q_iou_p50={kd.get('qualityfg_score_iou_p50', float('nan')):.6f}"
            if "qualityfg_gt_total" in kd
            else f", score_p90={kd.get('teacher_score_p90', float('nan')):.6f}"
        )
        print(
            f"{mode_label}: "
            f"task={kd.get('task_loss', float('nan')):.6f}, "
            f"cls={kd.get('cls_kd_loss', float('nan')):.6f}, "
            f"dfl={kd.get('dfl_kd_loss', float('nan')):.6f}, "
            f"term={kd.get('kd_term', float('nan')):.6f}, "
            f"w=({kd.get('effective_cls_weight', float('nan')):.4f},"
            f"{kd.get('effective_dfl_weight', float('nan')):.4f}), "
            f"mask={kd.get('mask_selected_ratio', float('nan')):.4%}"
            f"{selection_detail}",
            flush=True,
        )


class MultiBranchCriterion:
    """统一承载预测级 KD、Neck-SL2 或完整 FGD；教师每 batch 最多前向一次。"""

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        prediction_contract: dict[str, Any],
        *,
        prediction_kd_enabled: bool,
        neck_sl2_enabled: bool,
        fgd_enabled: bool,
        neck_contract: dict[str, Any] | None,
        cls_weight: float,
        dfl_weight: float,
        cls_temperature: float,
        dfl_temperature: float,
        confidence_threshold: float,
        topk_per_image: int,
        selection_mode: str,
        quality_topk_per_gt: int,
        kd_warmup_epochs: int,
        feature_weight: float,
        feature_warmup_epochs: int,
        fgd_weight: float,
        fgd_warmup_epochs: int,
        fgd_temperature: float,
        fgd_foreground_weight: float,
        fgd_background_weight: float,
        fgd_mask_weight: float,
        fgd_relation_weight: float,
        fgd_component_flags: dict[str, bool],
        trainer,
        admission_probe: bool,
    ) -> None:
        if not prediction_kd_enabled and not neck_sl2_enabled and not fgd_enabled:
            raise ValueError("MultiBranchCriterion 至少需要启用一个蒸馏分支。")
        self.student = unwrap_model(student)
        self.teacher = teacher.eval()
        self.trainer = trainer
        self.prediction_kd_enabled = bool(prediction_kd_enabled)
        self.neck_sl2_enabled = bool(neck_sl2_enabled)
        self.fgd_enabled = bool(fgd_enabled)
        self.neck_contract = neck_contract
        self.feature_weight = float(feature_weight)
        self.feature_warmup_epochs = int(feature_warmup_epochs)
        self.fgd_weight = float(fgd_weight)
        self.fgd_warmup_epochs = int(fgd_warmup_epochs)
        self.fgd_temperature = float(fgd_temperature)
        self.fgd_foreground_weight = float(fgd_foreground_weight)
        self.fgd_background_weight = float(fgd_background_weight)
        self.fgd_mask_weight = float(fgd_mask_weight)
        self.fgd_relation_weight = float(fgd_relation_weight)
        self.fgd_component_flags = dict(fgd_component_flags)
        if self.fgd_enabled and not any(self.fgd_component_flags.values()):
            raise ValueError("已启用 FGD，但四项 FGD 子损失均被关闭。")
        self.prediction = PredDFLCriterion(
            self.student,
            self.teacher,
            prediction_contract,
            kd_enabled=self.prediction_kd_enabled,
            cls_weight=cls_weight,
            dfl_weight=dfl_weight,
            cls_temperature=cls_temperature,
            dfl_temperature=dfl_temperature,
            confidence_threshold=confidence_threshold,
            topk_per_image=topk_per_image,
            selection_mode=selection_mode,
            quality_topk_per_gt=quality_topk_per_gt,
            warmup_epochs=kd_warmup_epochs,
            trainer=trainer,
            admission_probe=admission_probe,
            record_metrics=False,
        )
        self.student_capture: neck.FeatureCapture | None = None
        self.teacher_capture: neck.FeatureCapture | None = None
        self.detect: nn.Module | None = None
        if self.neck_sl2_enabled or self.fgd_enabled:
            if self.neck_contract is None:
                raise RuntimeError("已启用特征级蒸馏但没有 neck contract。")
            if self.neck_sl2_enabled:
                projectors = getattr(self.student, "_distill_projectors", None)
                if not isinstance(projectors, nn.ModuleList) or len(projectors) != 3:
                    raise RuntimeError("已启用 Neck-SL2 但训练期 projector 未正确注册。")
            if self.fgd_enabled:
                modules = getattr(self.student, "_distill_fgd_modules", None)
                if not isinstance(modules, nn.ModuleList) or len(modules) != 3:
                    raise RuntimeError("已启用官方完整 FGD 但训练期辅助模块未正确注册。")
            layer_indices = list(self.neck_contract["detect"]["student_f"])
            self.student_capture = neck.FeatureCapture(self.student, layer_indices, "student_multibranch_train")
            self.teacher_capture = neck.FeatureCapture(self.teacher, layer_indices, "teacher_multibranch_train")
            self.detect = neck.find_detect_module(self.teacher)[1]

    def close(self) -> None:
        self.prediction.close()
        if self.student_capture is not None:
            self.student_capture.close()
        if self.teacher_capture is not None:
            self.teacher_capture.close()

    def _feature_warmup_factor(self) -> float:
        if self.feature_warmup_epochs <= 0:
            return 1.0
        return min(1.0, float(int(self.trainer.epoch) + 1) / float(self.feature_warmup_epochs))

    def _fgd_warmup_factor(self) -> float:
        if self.fgd_warmup_epochs <= 0:
            return 1.0
        return min(1.0, float(int(self.trainer.epoch) + 1) / float(self.fgd_warmup_epochs))

    def _neck_feature_term(
        self,
        teacher_raw: list[torch.Tensor],
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if not self.neck_sl2_enabled:
            zero = batch["img"].new_zeros(())
            return zero, {
                "feature_kd_loss": 0.0,
                "feature_term": 0.0,
                "feature_weight": 0.0,
                "effective_feature_weight": 0.0,
                "feature_warmup_factor": 0.0,
                "teacher_score_mean": 0.0,
            }
        assert self.neck_contract is not None
        assert self.student_capture is not None and self.teacher_capture is not None and self.detect is not None
        student_features = self.student_capture.ordered()
        teacher_features = [feature.detach() for feature in self.teacher_capture.ordered()]
        projectors = self.student._distill_projectors
        if not (len(student_features) == len(teacher_features) == len(teacher_raw) == len(projectors) == 3):
            raise RuntimeError("运行时 Neck 特征、教师 raw 输出或 projector 数量不一致。")
        expected_no = int(self.neck_contract["detect"]["no"])
        reg_max = int(self.detect.reg_max)
        nc = int(self.detect.nc)
        level_losses: list[torch.Tensor] = []
        level_scores: list[torch.Tensor] = []
        for record, projector, student_feature, teacher_feature, raw in zip(
            self.neck_contract["feature_levels"], projectors, student_features, teacher_features, teacher_raw
        ):
            expected_spatial = tuple(record["spatial_shape"])
            if (
                student_feature.shape[0] != teacher_feature.shape[0]
                or tuple(student_feature.shape[-2:]) != expected_spatial
                or tuple(teacher_feature.shape[-2:]) != expected_spatial
                or raw.ndim != 4
                or raw.shape[1] != expected_no
                or tuple(raw.shape[-2:]) != expected_spatial
            ):
                raise RuntimeError(
                    "运行时 Neck/Detect 合同失败："
                    f"level={record['level']} student={neck.tensor_shape(student_feature)} "
                    f"teacher={neck.tensor_shape(teacher_feature)} raw={neck.tensor_shape(raw)}。"
                )
            projected = projector(student_feature)
            if projected.shape != teacher_feature.shape:
                raise RuntimeError(
                    f"{record['level']} projector 输出形状错误："
                    f"projected={neck.tensor_shape(projected)} teacher={neck.tensor_shape(teacher_feature)}。"
                )
            score_map = raw[:, 4 * reg_max : 4 * reg_max + nc].sigmoid().amax(dim=1, keepdim=True)
            numerator = ((projected - teacher_feature).pow(2) * score_map).sum()
            denominator = score_map.sum() * teacher_feature.shape[1] + 1e-12
            level_losses.append(numerator / denominator)
            level_scores.append(score_map.detach().mean())
        feature_loss = torch.stack(level_losses).mean()
        warmup = self._feature_warmup_factor()
        feature_term = self.feature_weight * warmup * feature_loss * batch["img"].shape[0]
        metrics = scalar_tensors_to_floats(
            {
                "feature_kd_loss": feature_loss,
                "feature_term": feature_term,
                "feature_weight": self.feature_weight,
                "effective_feature_weight": self.feature_weight * warmup,
                "feature_warmup_factor": warmup,
                "teacher_score_mean": torch.stack(level_scores).mean(),
            }
        )
        return feature_term, metrics

    def _fgd_feature_term(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        """官方 FGD：面积掩码、注意力对齐和 GC 全局关系。"""

        if not self.fgd_enabled:
            zero = batch["img"].new_zeros(())
            return zero, {
                "fgd_enabled": 0.0,
                "fgd_feature_loss": 0.0,
                "fgd_term": 0.0,
                "fgd_weight": 0.0,
                "fgd_effective_weight": 0.0,
                "fgd_warmup_factor": 0.0,
                "fgd_foreground_loss": 0.0,
                "fgd_background_loss": 0.0,
                "fgd_mask_loss": 0.0,
                "fgd_relation_loss": 0.0,
                "fgd_foreground_enabled": 0.0,
                "fgd_background_enabled": 0.0,
                "fgd_mask_enabled": 0.0,
                "fgd_relation_enabled": 0.0,
            }
        assert self.neck_contract is not None
        assert self.student_capture is not None and self.teacher_capture is not None
        student_features = self.student_capture.ordered()
        teacher_features = [feature.detach() for feature in self.teacher_capture.ordered()]
        modules = self.student._distill_fgd_modules
        if not (len(student_features) == len(teacher_features) == len(modules) == 3):
            raise RuntimeError("官方完整 FGD 的 Neck 特征或训练期辅助模块数量不一致。")
        for record, student_feature, teacher_feature in zip(
            self.neck_contract["feature_levels"], student_features, teacher_features
        ):
            expected_spatial = tuple(record["spatial_shape"])
            if (
                student_feature.shape[0] != teacher_feature.shape[0]
                or tuple(student_feature.shape[-2:]) != expected_spatial
                or tuple(teacher_feature.shape[-2:]) != expected_spatial
                or int(student_feature.shape[1]) != int(record["student_channels"])
                or int(teacher_feature.shape[1]) != int(record["teacher_channels"])
            ):
                raise RuntimeError(
                    "运行时官方完整 FGD Neck 合同失败："
                    f"level={record['level']} student={neck.tensor_shape(student_feature)} "
                    f"teacher={neck.tensor_shape(teacher_feature)}。"
                )
        fgd_loss, fgd_metrics = fgd.full_fgd_feature_loss(
            student_features,
            teacher_features,
            modules,
            batch,
            foreground_weight=self.fgd_foreground_weight,
            background_weight=self.fgd_background_weight,
            mask_weight=self.fgd_mask_weight,
            relation_weight=self.fgd_relation_weight,
            enable_foreground=self.fgd_component_flags["foreground"],
            enable_background=self.fgd_component_flags["background"],
            enable_mask=self.fgd_component_flags["mask"],
            enable_relation=self.fgd_component_flags["relation"],
        )
        warmup = self._fgd_warmup_factor()
        fgd_term = self.fgd_weight * warmup * fgd_loss * batch["img"].shape[0]
        metrics = scalar_tensors_to_floats(
            {
                **fgd_metrics,
                "fgd_enabled": 1.0,
                "fgd_term": fgd_term,
                "fgd_weight": self.fgd_weight,
                "fgd_effective_weight": self.fgd_weight * warmup,
                "fgd_warmup_factor": warmup,
                "fgd_temperature": self.fgd_temperature,
                "fgd_foreground_weight": self.fgd_foreground_weight,
                "fgd_background_weight": self.fgd_background_weight,
                "fgd_mask_weight": self.fgd_mask_weight,
                "fgd_relation_weight": self.fgd_relation_weight,
            }
        )
        return fgd_term, metrics

    def __call__(self, preds, batch):
        prediction_total, task_items = self.prediction(preds, batch)
        prediction_metrics = dict(self.prediction.last_metrics)
        teacher_raw = self.prediction.last_teacher_raw
        if (self.neck_sl2_enabled or self.fgd_enabled) and teacher_raw is None:
            with torch.no_grad():
                teacher_output = self.teacher(batch["img"])
                teacher_raw = (
                    [tensor.detach() for tensor in raw_detect_layers(teacher_output, label="teacher Neck-SL2")]
                    if self.neck_sl2_enabled
                    else None
                )
        if self.neck_sl2_enabled and teacher_raw is None:
            raise RuntimeError("Neck-SL2 训练未获得教师 raw Detect 输出。")
        feature_term, feature_metrics = self._neck_feature_term(teacher_raw or [], batch)
        fgd_term, fgd_metrics = self._fgd_feature_term(batch)
        total_loss = prediction_total + feature_term + fgd_term
        prediction_term = float(prediction_metrics.get("kd_term", 0.0))
        latest = dict(prediction_metrics)
        latest.update(feature_metrics)
        latest.update(fgd_metrics)
        latest.update(
            scalar_tensors_to_floats(
                {
                    "total_loss": total_loss,
                    "combined_kd_term": prediction_total.new_tensor(prediction_term) + feature_term + fgd_term,
                    "prediction_kd_enabled": float(self.prediction_kd_enabled),
                    "neck_sl2_enabled": float(self.neck_sl2_enabled),
                    "fgd_enabled": float(self.fgd_enabled),
                }
            )
        )
        self.trainer.kd_meter.update(latest)
        return total_loss, task_items


class MultiBranchEpochReporter(PredDFLEpochReporter):
    """在预测级摘要外附加 Neck-SL2 / 完整 FGD 分项，保留逐 epoch 单行输出。"""

    def _print_record(self, record: dict[str, Any]) -> None:
        super()._print_record(record)
        kd = record.get("distillation", {})
        if kd.get("neck_sl2_enabled", 0.0) >= 0.5:
            print(
                "NeckSL2: "
                f"feature={kd.get('feature_kd_loss', float('nan')):.6f}, "
                f"term={kd.get('feature_term', float('nan')):.6f}, "
                f"w={kd.get('effective_feature_weight', float('nan')):.4f}, "
                f"score_mean={kd.get('teacher_score_mean', float('nan')):.6f}",
                flush=True,
            )
        if kd.get("fgd_enabled", 0.0) >= 0.5:
            components = "+".join(
                name
                for name, key in (
                    ("fg", "fgd_foreground_enabled"),
                    ("bg", "fgd_background_enabled"),
                    ("mask", "fgd_mask_enabled"),
                    ("relation", "fgd_relation_enabled"),
                )
                if kd.get(key, 0.0) >= 0.5
            )
            print(
                "OfficialFGD: "
                f"components={components or 'none'}, "
                f"feature={kd.get('fgd_feature_loss', float('nan')):.6f}, "
                f"term={kd.get('fgd_term', float('nan')):.6f}, "
                f"w={kd.get('fgd_effective_weight', float('nan')):.4f}, "
                f"fg={kd.get('fgd_foreground_loss', float('nan')):.3f}, "
                f"bg={kd.get('fgd_background_loss', float('nan')):.3f}, "
                f"mask={kd.get('fgd_mask_loss', float('nan')):.3f}, "
                f"relation={kd.get('fgd_relation_loss', float('nan')):.3f}",
                flush=True,
            )


def create_preddfl_trainer_class(
    student_template: nn.Module,
    teacher_template: nn.Module,
    student_weights: Path,
    teacher_weights: Path,
    contract: dict[str, Any],
    *,
    kd_enabled: bool,
    cls_weight: float,
    dfl_weight: float,
    cls_temperature: float,
    dfl_temperature: float,
    confidence_threshold: float,
    topk_per_image: int,
    selection_mode: str,
    quality_topk_per_gt: int,
    kd_warmup_epochs: int,
    admission_probe: bool,
):
    """返回只训练物理 student 的单卡 DetectionTrainer。"""

    expected_physical = neck.physical_student_copy(student_template)
    expected_structure = neck.core.model_structure_signature(expected_physical)
    expected_state = neck.core.model_state_signature(expected_physical)
    student_sha256 = neck.core.sha256_file(student_weights)
    teacher_sha256 = neck.core.sha256_file(teacher_weights)
    phase = "qualityfg_distillation" if kd_enabled and selection_mode == "quality_fg_v2" else (
        "distillation" if kd_enabled else "no_kd_control"
    )
    distillation_id = (
        QUALITYFG_DISTILLATION_ID if selection_mode == "quality_fg_v2" else DISTILLATION_ID
    ) if kd_enabled else (
        QUALITYFG_CONTROL_ID if selection_mode == "quality_fg_v2" else CONTROL_ID
    )
    implementation = QUALITYFG_IMPLEMENTATION if selection_mode == "quality_fg_v2" else IMPLEMENTATION

    class PredDFLTrainer(DetectionTrainer):
        def __init__(self, *trainer_args, **trainer_kwargs):
            self.epoch_reporter = PredDFLEpochReporter(phase)
            self.kd_meter = neck.KDMeter()
            self.structure_check: dict[str, Any] = {}
            self.preflight: dict[str, Any] = {}
            self.distill_criterion: PredDFLCriterion | None = None
            super().__init__(*trainer_args, **trainer_kwargs)

        def get_model(self, cfg=None, weights=None, verbose=True):
            del cfg, weights
            model = neck.physical_student_copy(student_template).float()
            if verbose and hasattr(model, "info"):
                model.info(verbose=True)
            return model

        def _setup_train(self):
            super()._setup_train()
            if int(getattr(self, "world_size", 1)) != 1:
                raise RuntimeError("PredDFL-KD 当前仅允许单进程单卡训练，禁止 DDP 静默运行。")

            physical_before = neck.physical_student_copy(self.model)
            actual_structure = neck.core.model_structure_signature(physical_before)
            actual_state = neck.core.model_state_signature(physical_before)
            structure_matched = actual_structure["sha256"] == expected_structure["sha256"]
            state_matched = actual_state["sha256"] == expected_state["sha256"]
            self.structure_check = {
                "status": "pass" if structure_matched and state_matched else "fail",
                "source_student_weights": str(student_weights),
                "source_student_sha256": student_sha256,
                "expected_structure": expected_structure,
                "actual_structure_before_training": actual_structure,
                "expected_state": expected_state,
                "actual_state_before_training": actual_state,
                "structure_matched": structure_matched,
                "state_matched": state_matched,
            }
            if not (structure_matched and state_matched):
                raise RuntimeError("训练器未直接加载输入物理学生，禁止继续。")

            teacher = copy.deepcopy(unwrap_model(teacher_template)).to(self.device).float().eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            self.distill_criterion = PredDFLCriterion(
                self.model,
                teacher,
                contract,
                kd_enabled=kd_enabled,
                cls_weight=cls_weight,
                dfl_weight=dfl_weight,
                cls_temperature=cls_temperature,
                dfl_temperature=dfl_temperature,
                confidence_threshold=confidence_threshold,
                topk_per_image=topk_per_image,
                selection_mode=selection_mode,
                quality_topk_per_gt=quality_topk_per_gt,
                warmup_epochs=kd_warmup_epochs,
                trainer=self,
                admission_probe=admission_probe,
            )
            unwrap_model(self.model).criterion = self.distill_criterion
            ownership = optimizer_ownership_report(self.optimizer, unwrap_model(self.model), teacher)
            frozen = neck.frozen_parameter_report(teacher)
            self.preflight = {
                "status": "pass" if ownership["status"] == "pass" and frozen["requires_grad_count"] == 0 else "fail",
                "time_utc": neck.utc_now(),
                "implementation": implementation,
                "distillation_id": distillation_id,
                "mode": "train" if kd_enabled else "control",
                "teacher": {"weights": str(teacher_weights), "sha256": teacher_sha256, "frozen": frozen},
                "student": {"weights": str(student_weights), "sha256": student_sha256, "physical_structure": expected_structure},
                "prediction_contract": contract,
                "optimizer_ownership": ownership,
                "loss_contract": (
                    "L_total=L_det + warmup*(lambda_cls*L_clsKD + lambda_dfl*L_dflKD)*batch_size"
                    if kd_enabled
                    else "L_total=L_det (matched no-KD continuation control)"
                ),
                "checkpoint_contract": "checkpoint_only_contains_physical_student_no_teacher_no_criterion",
            }
            report_path = Path(self.save_dir) / "admission" / "distillation_preflight.json"
            neck.core.json_dump(report_path, self.preflight)
            if self.preflight["status"] != "pass":
                raise RuntimeError(f"PredDFL/control 训练所有权或教师冻结检查失败：{report_path}")
            self.epoch_reporter.attach(self)

        def optimizer_step(self):
            # QualityFG 在纯背景 batch 合法地没有 GT 选择点。首步真正的 KD 梯度
            # 只能在首次非零 KD 项后验收，不能把背景 batch 当成梯度失败。
            kd_pending = (
                kd_enabled
                and self.distill_criterion is not None
                and self.distill_criterion.first_kd_gradient.get("status") == "pending"
            )
            first_check = admission_probe and "backward" not in self.preflight and not kd_pending
            report_path = Path(self.save_dir) / "admission" / "distillation_preflight.json"
            if first_check:
                assert self.distill_criterion is not None
                raw_student = unwrap_model(self.model)
                trainable = [parameter for parameter in raw_student.parameters() if parameter.requires_grad]
                nonzero_student = sum(
                    int(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0)
                    for parameter in trainable
                )
                non_finite = [
                    name for name, parameter in raw_student.named_parameters()
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
                ]
                latest = dict(self.kd_meter.latest)
                finite_losses = bool(latest) and all(math.isfinite(float(value)) for value in latest.values())
                teacher_frozen = neck.frozen_parameter_report(self.distill_criterion.teacher)
                kd_gradient = dict(self.distill_criterion.first_kd_gradient)
                kd_gradient_passed = (
                    kd_gradient.get("status") == "pass" if kd_enabled else kd_gradient.get("status") == "not_applicable"
                )
                passed = (
                    finite_losses
                    and nonzero_student > 0
                    and not non_finite
                    and teacher_frozen["requires_grad_count"] == 0
                    and teacher_frozen["gradient_present_count"] == 0
                    and kd_gradient_passed
                )
                self.preflight["backward"] = {
                    "status": "pass" if passed else "fail",
                    "latest_loss": latest,
                    "trainable_student_parameter_count": len(trainable),
                    "student_parameters_with_nonzero_gradient": nonzero_student,
                    "non_finite_gradient_parameter_names": non_finite,
                    "teacher_after_backward": teacher_frozen,
                    "kd_gradient": kd_gradient,
                    "optimizer_step_completed": False,
                }
                neck.core.json_dump(report_path, self.preflight)
                if not passed:
                    raise RuntimeError(f"PredDFL/control 首次反向传播准入失败：{report_path}")
            super().optimizer_step()
            if first_check:
                self.preflight["backward"]["optimizer_step_completed"] = True
                neck.core.json_dump(report_path, self.preflight)

        def save_model(self):
            """和 Neck-SL2 相同：只保存物理 student 的 EMA 副本。"""

            ema_source = getattr(getattr(self, "ema", None), "ema", None)
            source = ema_source if ema_source is not None else unwrap_model(self.model)
            student_only = neck.physical_student_copy(source).half()
            checkpoint = {
                "epoch": int(getattr(self, "epoch", -1)),
                "best_fitness": getattr(self, "best_fitness", None),
                "model": None,
                "ema": student_only,
                "updates": int(getattr(getattr(self, "ema", None), "updates", 0)),
                "optimizer": None,
                "train_args": vars(self.args).copy(),
                "date": neck.utc_now(),
                "version": str(getattr(neck.core.ultralytics, "__version__", "unknown")),
                "distillation_checkpoint_contract": "physical_student_only_v1",
            }
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            serialized = buffer.getvalue()
            self.last.write_bytes(serialized)
            if getattr(self, "best_fitness", None) == getattr(self, "fitness", None):
                self.best.write_bytes(serialized)
            save_period = int(getattr(self, "save_period", getattr(self.args, "save_period", -1)))
            if save_period > 0 and self.epoch > 0 and self.epoch % save_period == 0:
                (Path(self.wdir) / f"epoch{self.epoch}.pt").write_bytes(serialized)

    return PredDFLTrainer


def create_multibranch_trainer_class(
    student_template: nn.Module,
    teacher_template: nn.Module,
    student_weights: Path,
    teacher_weights: Path,
    prediction_contract: dict[str, Any],
    *,
    prediction_kd_enabled: bool,
    neck_sl2_enabled: bool,
    fgd_enabled: bool,
    neck_contract: dict[str, Any] | None,
    cls_weight: float,
    dfl_weight: float,
    cls_temperature: float,
    dfl_temperature: float,
    confidence_threshold: float,
    topk_per_image: int,
    selection_mode: str,
    quality_topk_per_gt: int,
    kd_warmup_epochs: int,
    feature_weight: float,
    feature_warmup_epochs: int,
    fgd_weight: float,
    fgd_warmup_epochs: int,
    fgd_temperature: float,
    fgd_foreground_weight: float,
    fgd_background_weight: float,
    fgd_mask_weight: float,
    fgd_relation_weight: float,
    fgd_component_flags: dict[str, bool],
    fgd_run_label: str,
    fgd_implementation: str,
    admission_probe: bool,
):
    """带显式分支开关的统一训练器；特征分支仅注册训练期 projector。"""

    if not prediction_kd_enabled and not neck_sl2_enabled and not fgd_enabled:
        raise ValueError("multi-branch trainer 至少需要启用一个蒸馏分支。")
    if (neck_sl2_enabled or fgd_enabled) and neck_contract is None:
        raise ValueError("已启用特征级蒸馏但未提供 neck contract。")
    expected_physical = neck.physical_student_copy(student_template)
    expected_structure = neck.core.model_structure_signature(expected_physical)
    expected_state = neck.core.model_state_signature(expected_physical)
    student_sha256 = neck.core.sha256_file(student_weights)
    teacher_sha256 = neck.core.sha256_file(teacher_weights)
    phase = (
        "full_fgd_distillation"
        if fgd_enabled and all(fgd_component_flags.values())
        else "fgd_component_ablation"
        if fgd_enabled
        else "hybrid_qualityfg_necksl2_distillation"
        if prediction_kd_enabled and selection_mode == "quality_fg_v2"
        else "multibranch_distillation"
    )

    class MultiBranchTrainer(DetectionTrainer):
        def __init__(self, *trainer_args, **trainer_kwargs):
            self.epoch_reporter = MultiBranchEpochReporter(phase)
            self.kd_meter = neck.KDMeter()
            self.structure_check: dict[str, Any] = {}
            self.preflight: dict[str, Any] = {}
            self.distill_criterion: MultiBranchCriterion | None = None
            super().__init__(*trainer_args, **trainer_kwargs)

        def get_model(self, cfg=None, weights=None, verbose=True):
            del cfg, weights
            model = neck.physical_student_copy(student_template).float()
            if neck_sl2_enabled:
                assert neck_contract is not None
                model._distill_projectors = neck.make_projectors(neck_contract)
            if fgd_enabled:
                assert neck_contract is not None
                model._distill_fgd_modules = fgd.make_official_fgd_modules(
                    neck_contract,
                    temperature=fgd_temperature,
                )
            if verbose and hasattr(model, "info"):
                model.info(verbose=True)
            return model

        def _setup_train(self):
            super()._setup_train()
            if int(getattr(self, "world_size", 1)) != 1:
                raise RuntimeError("multi-branch KD 当前仅允许单进程单卡训练，禁止 DDP 静默运行。")

            physical_before = neck.physical_student_copy(self.model)
            actual_structure = neck.core.model_structure_signature(physical_before)
            actual_state = neck.core.model_state_signature(physical_before)
            structure_matched = actual_structure["sha256"] == expected_structure["sha256"]
            state_matched = actual_state["sha256"] == expected_state["sha256"]
            self.structure_check = {
                "status": "pass" if structure_matched and state_matched else "fail",
                "source_student_weights": str(student_weights),
                "source_student_sha256": student_sha256,
                "expected_structure": expected_structure,
                "actual_structure_before_training": actual_structure,
                "expected_state": expected_state,
                "actual_state_before_training": actual_state,
                "structure_matched": structure_matched,
                "state_matched": state_matched,
            }
            if not (structure_matched and state_matched):
                raise RuntimeError("训练器未直接加载输入物理学生，禁止继续。")

            teacher = copy.deepcopy(unwrap_model(teacher_template)).to(self.device).float().eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            self.distill_criterion = MultiBranchCriterion(
                self.model,
                teacher,
                prediction_contract,
                prediction_kd_enabled=prediction_kd_enabled,
                neck_sl2_enabled=neck_sl2_enabled,
                fgd_enabled=fgd_enabled,
                neck_contract=neck_contract,
                cls_weight=cls_weight,
                dfl_weight=dfl_weight,
                cls_temperature=cls_temperature,
                dfl_temperature=dfl_temperature,
                confidence_threshold=confidence_threshold,
                topk_per_image=topk_per_image,
                selection_mode=selection_mode,
                quality_topk_per_gt=quality_topk_per_gt,
                kd_warmup_epochs=kd_warmup_epochs,
                feature_weight=feature_weight,
                feature_warmup_epochs=feature_warmup_epochs,
                fgd_weight=fgd_weight,
                fgd_warmup_epochs=fgd_warmup_epochs,
                fgd_temperature=fgd_temperature,
                fgd_foreground_weight=fgd_foreground_weight,
                fgd_background_weight=fgd_background_weight,
                fgd_mask_weight=fgd_mask_weight,
                fgd_relation_weight=fgd_relation_weight,
                fgd_component_flags=fgd_component_flags,
                trainer=self,
                admission_probe=admission_probe,
            )
            raw_student = unwrap_model(self.model)
            raw_student.criterion = self.distill_criterion
            ownership = optimizer_ownership_report(self.optimizer, raw_student, teacher)
            projector_ownership = (
                neck.optimizer_ownership_report(self.optimizer, raw_student._distill_projectors, teacher)
                if neck_sl2_enabled
                else {"status": "not_applicable", "projector_parameter_count": 0}
            )
            fgd_auxiliary_ownership = (
                neck.optimizer_ownership_report(self.optimizer, raw_student._distill_fgd_modules, teacher)
                if fgd_enabled
                else {"status": "not_applicable", "projector_parameter_count": 0}
            )
            frozen = neck.frozen_parameter_report(teacher)
            preflight_passed = (
                ownership["status"] == "pass"
                and projector_ownership["status"] in {"pass", "not_applicable"}
                and fgd_auxiliary_ownership["status"] in {"pass", "not_applicable"}
                and frozen["requires_grad_count"] == 0
            )
            self.preflight = {
                "status": "pass" if preflight_passed else "fail",
                "time_utc": neck.utc_now(),
                "implementation": fgd_implementation if fgd_enabled else HYBRID_IMPLEMENTATION,
                "distillation_id": fgd_run_label if fgd_enabled else HYBRID_DISTILLATION_ID,
                "mode": "train",
                "branches": {
                    "prediction_kd_enabled": prediction_kd_enabled,
                    "neck_sl2_enabled": neck_sl2_enabled,
                    "fgd_enabled": fgd_enabled,
                    "fgd_components": dict(fgd_component_flags) if fgd_enabled else None,
                },
                "teacher": {"weights": str(teacher_weights), "sha256": teacher_sha256, "frozen": frozen},
                "student": {"weights": str(student_weights), "sha256": student_sha256, "physical_structure": expected_structure},
                "prediction_contract": prediction_contract,
                "neck_contract": neck_contract if (neck_sl2_enabled or fgd_enabled) else None,
                "fgd_contract": fgd_contract_name(fgd_component_flags) if fgd_enabled else None,
                "optimizer_ownership": ownership,
                "projector_optimizer_ownership": projector_ownership,
                "fgd_auxiliary_optimizer_ownership": fgd_auxiliary_ownership,
                "loss_contract": (
                    "L_total=L_det"
                    + (
                        " + pred_warmup*(lambda_cls*L_clsKD + lambda_dfl*L_dflKD)*batch_size"
                        if prediction_kd_enabled
                        else ""
                    )
                    + (
                        " + feat_warmup*lambda_feat*mean_P3P4P5(L_SL2)*batch_size"
                        if neck_sl2_enabled
                        else ""
                    )
                    + (" + " + fgd_loss_contract(fgd_component_flags).removeprefix("L_total=L_det + ") if fgd_enabled else "")
                ),
                "checkpoint_contract": "checkpoint_only_contains_physical_student_no_teacher_no_training_auxiliary_no_criterion",
            }
            report_path = Path(self.save_dir) / "admission" / "distillation_preflight.json"
            neck.core.json_dump(report_path, self.preflight)
            if self.preflight["status"] != "pass":
                raise RuntimeError(f"multi-branch KD 所有权或教师冻结检查失败：{report_path}")
            self.epoch_reporter.attach(self)

        def optimizer_step(self):
            prediction_pending = (
                prediction_kd_enabled
                and self.distill_criterion is not None
                and self.distill_criterion.prediction.first_kd_gradient.get("status") == "pending"
            )
            first_check = admission_probe and "backward" not in self.preflight and not prediction_pending
            report_path = Path(self.save_dir) / "admission" / "distillation_preflight.json"
            if first_check:
                assert self.distill_criterion is not None
                raw_student = unwrap_model(self.model)
                named_trainable = [(name, parameter) for name, parameter in raw_student.named_parameters() if parameter.requires_grad]
                physical_trainable = [
                    (name, parameter)
                    for name, parameter in named_trainable
                    if not name.startswith("_distill_projectors.") and not name.startswith("_distill_fgd_modules.")
                ]
                projector_parameters = (
                    list(raw_student._distill_projectors.parameters()) if neck_sl2_enabled else []
                )
                fgd_auxiliary_parameters = (
                    list(raw_student._distill_fgd_modules.parameters()) if fgd_enabled else []
                )
                nonzero_physical = sum(
                    int(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0)
                    for _name, parameter in physical_trainable
                )
                nonzero_projector = sum(
                    int(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0)
                    for parameter in projector_parameters
                )
                nonzero_fgd_auxiliary = sum(
                    int(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0)
                    for parameter in fgd_auxiliary_parameters
                )
                non_finite = [
                    name
                    for name, parameter in named_trainable
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
                ]
                latest = dict(self.kd_meter.latest)
                finite_losses = bool(latest) and all(math.isfinite(float(value)) for value in latest.values())
                teacher_frozen = neck.frozen_parameter_report(self.distill_criterion.teacher)
                prediction_gradient = dict(self.distill_criterion.prediction.first_kd_gradient)
                prediction_gradient_passed = (
                    prediction_gradient.get("status") == "pass"
                    if prediction_kd_enabled
                    else prediction_gradient.get("status") == "not_applicable"
                )
                projector_gradient_passed = not neck_sl2_enabled or nonzero_projector > 0
                fgd_auxiliary_gradient_passed = not fgd_enabled or nonzero_fgd_auxiliary > 0
                passed = (
                    finite_losses
                    and nonzero_physical > 0
                    and projector_gradient_passed
                    and fgd_auxiliary_gradient_passed
                    and not non_finite
                    and teacher_frozen["requires_grad_count"] == 0
                    and teacher_frozen["gradient_present_count"] == 0
                    and prediction_gradient_passed
                )
                self.preflight["backward"] = {
                    "status": "pass" if passed else "fail",
                    "latest_loss": latest,
                    "trainable_physical_student_parameter_count": len(physical_trainable),
                    "physical_student_parameters_with_nonzero_gradient": nonzero_physical,
                    "projector_parameter_count": len(projector_parameters),
                    "projector_parameters_with_nonzero_gradient": nonzero_projector,
                    "fgd_auxiliary_parameter_count": len(fgd_auxiliary_parameters),
                    "fgd_auxiliary_parameters_with_nonzero_gradient": nonzero_fgd_auxiliary,
                    "non_finite_gradient_parameter_names": non_finite,
                    "teacher_after_backward": teacher_frozen,
                    "prediction_kd_gradient": prediction_gradient,
                    "optimizer_step_completed": False,
                }
                neck.core.json_dump(report_path, self.preflight)
                if not passed:
                    raise RuntimeError(f"multi-branch KD 首次反向传播准入失败：{report_path}")
            super().optimizer_step()
            if first_check:
                self.preflight["backward"]["optimizer_step_completed"] = True
                neck.core.json_dump(report_path, self.preflight)

        def save_model(self):
            ema_source = getattr(getattr(self, "ema", None), "ema", None)
            source = ema_source if ema_source is not None else unwrap_model(self.model)
            student_only = neck.physical_student_copy(source).half()
            checkpoint = {
                "epoch": int(getattr(self, "epoch", -1)),
                "best_fitness": getattr(self, "best_fitness", None),
                "model": None,
                "ema": student_only,
                "updates": int(getattr(getattr(self, "ema", None), "updates", 0)),
                "optimizer": None,
                "train_args": vars(self.args).copy(),
                "date": neck.utc_now(),
                "version": str(getattr(neck.core.ultralytics, "__version__", "unknown")),
                "distillation_checkpoint_contract": "physical_student_only_v1",
            }
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            serialized = buffer.getvalue()
            self.last.write_bytes(serialized)
            if getattr(self, "best_fitness", None) == getattr(self, "fitness", None):
                self.best.write_bytes(serialized)
            save_period = int(getattr(self, "save_period", getattr(self.args, "save_period", -1)))
            if save_period > 0 and self.epoch > 0 and self.epoch % save_period == 0:
                (Path(self.wdir) / f"epoch{self.epoch}.pt").write_bytes(serialized)

    return MultiBranchTrainer


def _common_training_contract(args: argparse.Namespace, student_weights: Path, teacher_weights: Path) -> dict[str, Any]:
    return {
        "runner_script_sha256": neck.core.sha256_file(CURRENT_FILE),
        "fgd_module_sha256": neck.core.sha256_file(Path(fgd.__file__).resolve()),
        "student_source_sha256": neck.core.sha256_file(student_weights),
        "teacher_sha256": neck.core.sha256_file(teacher_weights),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "workers": int(args.workers),
        "optimizer": "SGD",
        "lr0": float(args.lr0),
        "momentum": float(args.momentum),
        "weight_decay": float(args.weight_decay),
        "warmup_epochs": 0.0,
        "close_mosaic": int(args.close_mosaic),
        "amp": False,
        "cls_weight": float(args.cls_weight),
        "dfl_weight": float(args.dfl_weight),
        "cls_temperature": float(args.cls_temperature),
        "dfl_temperature": float(args.dfl_temperature),
        "confidence_threshold": float(args.confidence_threshold),
        "topk_per_image": int(args.topk_per_image),
        "kd_warmup_epochs": int(args.kd_warmup_epochs),
        "selection_mode": str(args.selection_mode),
        "quality_topk_per_gt": int(args.quality_topk_per_gt),
        # 是否开启 KD 及 FGD 子项选择是 control/KD 唯一允许不同的目标字段，不能写入比较合同。
        # 所有 FGD 数值超参数必须保留，使共享 control 与各 FGD 分项 KD 的训练口径可被严格校验。
        "feature_weight": float(args.feature_weight),
        "feature_warmup_epochs": int(args.feature_warmup_epochs),
        "fgd_weight": float(args.fgd_weight),
        "fgd_warmup_epochs": int(args.fgd_warmup_epochs),
        "fgd_temperature": float(args.fgd_temperature),
        "fgd_foreground_weight": float(args.fgd_foreground_weight),
        "fgd_background_weight": float(args.fgd_background_weight),
        "fgd_mask_weight": float(args.fgd_mask_weight),
        "fgd_relation_weight": float(args.fgd_relation_weight),
    }


def run_train(args: argparse.Namespace) -> None:
    teacher_weights = neck.require_file(args.teacher, "教师权重")
    student_weights = neck.require_file(args.student, "学生权重")
    data_yaml = neck.require_file(args.data, "数据 YAML")
    fixed_input = neck.require_file(args.fixed_input, "固定输入图像")
    output_root = Path(args.output).expanduser().resolve()
    requested_run_dir = output_root / args.name
    requested_prediction = args.enable_prediction_kd
    prediction_kd_enabled = args.mode == "train" if requested_prediction is None else bool(requested_prediction)
    neck_sl2_enabled = bool(args.enable_neck_sl2)
    fgd_enabled = bool(args.enable_fgd)
    fgd_component_flags = resolved_fgd_component_flags(args)
    if args.mode == "control":
        if prediction_kd_enabled or neck_sl2_enabled or fgd_enabled:
            raise ValueError("control 模式必须关闭 --enable-prediction-kd、--enable-neck-sl2 和 --enable-fgd。")
        prediction_kd_enabled = False
        neck_sl2_enabled = False
        fgd_enabled = False
    elif not prediction_kd_enabled and not neck_sl2_enabled and not fgd_enabled:
        raise ValueError("train 模式至少启用一个蒸馏分支。")
    if fgd_enabled and (prediction_kd_enabled or neck_sl2_enabled):
        raise ValueError("完整 FGD 当前必须作为独立路线评估；禁止与 Prediction KD 或 Neck-SL2 同 run 混合。")
    args.resolved_prediction_kd_enabled = prediction_kd_enabled
    args.resolved_neck_sl2_enabled = neck_sl2_enabled
    args.resolved_fgd_enabled = fgd_enabled
    args.resolved_fgd_component_flags = dict(fgd_component_flags)
    kd_enabled = prediction_kd_enabled or neck_sl2_enabled or fgd_enabled
    if args.no_overwrite and requested_run_dir.exists():
        raise FileExistsError(f"禁止覆盖已有输出目录：{requested_run_dir}；请更换 --name。")
    if args.epochs < 1:
        raise ValueError("--epochs 必须 >= 1。")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold 必须位于 [0, 1]。")
    if args.selection_mode == "global_score_v1" and args.topk_per_image < 1:
        raise ValueError("--topk-per-image 必须 >= 1。")
    if args.selection_mode == "quality_fg_v2" and args.quality_topk_per_gt < 1:
        raise ValueError("--quality-topk-per-gt 必须 >= 1。")
    if args.cls_temperature <= 0 or args.dfl_temperature <= 0:
        raise ValueError("分类与 DFL temperature 必须 > 0。")
    if args.cls_weight < 0 or args.dfl_weight < 0:
        raise ValueError("分类与 DFL 权重必须 >= 0。")
    if prediction_kd_enabled and args.cls_weight == 0 and args.dfl_weight == 0:
        raise ValueError("train 模式至少一个 KD 权重必须 > 0。")
    if args.kd_warmup_epochs < 0:
        raise ValueError("--kd-warmup-epochs 必须 >= 0。")
    if neck_sl2_enabled and args.feature_weight <= 0:
        raise ValueError("启用 Neck-SL2 时 --feature-weight 必须 > 0。")
    if args.feature_warmup_epochs < 0:
        raise ValueError("--feature-warmup-epochs 必须 >= 0。")
    if fgd_enabled:
        if args.fgd_weight <= 0:
            raise ValueError("启用完整 FGD 时 --fgd-weight 必须 > 0。")
        if args.fgd_warmup_epochs < 0:
            raise ValueError("--fgd-warmup-epochs 必须 >= 0。")
        if not math.isfinite(float(args.fgd_temperature)) or args.fgd_temperature <= 0:
            raise ValueError("--fgd-temperature 必须为有限正数。")
        official_weights = (
            args.fgd_foreground_weight,
            args.fgd_background_weight,
            args.fgd_mask_weight,
            args.fgd_relation_weight,
        )
        if not all(math.isfinite(float(value)) and value >= 0 for value in official_weights):
            raise ValueError("FGD alpha/beta/gamma/lambda 权重必须为有限非负数。")
        effective_component_weights = {
            "foreground": args.fgd_foreground_weight if fgd_component_flags["foreground"] else 0.0,
            "background": args.fgd_background_weight if fgd_component_flags["background"] else 0.0,
            "mask": args.fgd_mask_weight if fgd_component_flags["mask"] else 0.0,
            "relation": args.fgd_relation_weight if fgd_component_flags["relation"] else 0.0,
        }
        if sum(effective_component_weights.values()) <= 0:
            raise ValueError("至少一个已启用 FGD 子损失的权重必须 > 0。")

    neck.core.set_seed(args.seed)
    device = neck.resolve_device(args.device)
    started_at = time.time()
    if not kd_enabled:
        run_label = QUALITYFG_CONTROL_ID if args.selection_mode == "quality_fg_v2" else CONTROL_ID
        implementation = QUALITYFG_IMPLEMENTATION if args.selection_mode == "quality_fg_v2" else IMPLEMENTATION
    elif fgd_enabled:
        run_label = FGD_DISTILLATION_ID if all(fgd_component_flags.values()) else FGD_COMPONENT_ABLATION_ID
        implementation = FGD_IMPLEMENTATION
    elif prediction_kd_enabled and neck_sl2_enabled:
        run_label = HYBRID_DISTILLATION_ID
        implementation = HYBRID_IMPLEMENTATION
    elif neck_sl2_enabled:
        run_label = "Y11-NeckSL2-KD-v1"
        implementation = HYBRID_IMPLEMENTATION
    else:
        run_label = QUALITYFG_DISTILLATION_ID if args.selection_mode == "quality_fg_v2" else DISTILLATION_ID
        implementation = QUALITYFG_IMPLEMENTATION if args.selection_mode == "quality_fg_v2" else IMPLEMENTATION
    print("=" * 60, flush=True)
    print(f"开始 {run_label}", flush=True)
    print(f"教师: {teacher_weights}", flush=True)
    print(f"学生: {student_weights}", flush=True)
    print(f"输出: {requested_run_dir}", flush=True)
    print("=" * 60, flush=True)

    student_template = YOLO(str(student_weights)).model.cpu().float()
    teacher_template = YOLO(str(teacher_weights)).model.cpu().float()
    if hasattr(student_template, "criterion"):
        student_template.criterion = None
    if hasattr(teacher_template, "criterion"):
        teacher_template.criterion = None
    before_physical = neck.physical_student_copy(student_template)
    before_structure = neck.core.model_structure_signature(before_physical)
    before_state = neck.core.model_state_signature(before_physical)
    before_stats = neck.core.model_stats(before_physical, args.imgsz, device)
    contract = build_prediction_contract(student_template, teacher_template, imgsz=args.imgsz, experiment_id=run_label)
    neck_contract = neck.build_neck_contract(student_template, teacher_template, imgsz=args.imgsz) if (neck_sl2_enabled or fgd_enabled) else None
    if neck_sl2_enabled or fgd_enabled:
        trainer_class = create_multibranch_trainer_class(
            student_template,
            teacher_template,
            student_weights,
            teacher_weights,
            contract,
            prediction_kd_enabled=prediction_kd_enabled,
            neck_sl2_enabled=neck_sl2_enabled,
            fgd_enabled=fgd_enabled,
            neck_contract=neck_contract,
            cls_weight=args.cls_weight,
            dfl_weight=args.dfl_weight,
            cls_temperature=args.cls_temperature,
            dfl_temperature=args.dfl_temperature,
            confidence_threshold=args.confidence_threshold,
            topk_per_image=args.topk_per_image,
            selection_mode=args.selection_mode,
            quality_topk_per_gt=args.quality_topk_per_gt,
            kd_warmup_epochs=args.kd_warmup_epochs,
            feature_weight=args.feature_weight,
            feature_warmup_epochs=args.feature_warmup_epochs,
            fgd_weight=args.fgd_weight,
            fgd_warmup_epochs=args.fgd_warmup_epochs,
            fgd_temperature=args.fgd_temperature,
            fgd_foreground_weight=args.fgd_foreground_weight,
            fgd_background_weight=args.fgd_background_weight,
            fgd_mask_weight=args.fgd_mask_weight,
            fgd_relation_weight=args.fgd_relation_weight,
            fgd_component_flags=fgd_component_flags,
            fgd_run_label=run_label,
            fgd_implementation=implementation,
            admission_probe=args.admission_probe,
        )
    else:
        trainer_class = create_preddfl_trainer_class(
            student_template,
            teacher_template,
            student_weights,
            teacher_weights,
            contract,
            kd_enabled=prediction_kd_enabled,
            cls_weight=args.cls_weight,
            dfl_weight=args.dfl_weight,
            cls_temperature=args.cls_temperature,
            dfl_temperature=args.dfl_temperature,
            confidence_threshold=args.confidence_threshold,
            topk_per_image=args.topk_per_image,
            selection_mode=args.selection_mode,
            quality_topk_per_gt=args.quality_topk_per_gt,
            kd_warmup_epochs=args.kd_warmup_epochs,
            admission_probe=args.admission_probe,
        )
    yolo = YOLO(str(student_weights))
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": str(output_root),
        "name": args.name,
        "exist_ok": not args.no_overwrite,
        "seed": args.seed,
        "optimizer": "SGD",
        "lr0": args.lr0,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": 0.0,
        "close_mosaic": args.close_mosaic,
        "amp": False,
        "compile": False,
        "pretrained": True,
        "plots": args.plots,
        "verbose": True,
    }
    yolo.train(trainer=trainer_class, **train_kwargs)
    trainer = yolo.trainer
    if getattr(trainer, "distill_criterion", None) is not None:
        trainer.distill_criterion.close()
    if args.admission_probe and "backward" not in getattr(trainer, "preflight", {}):
        report_path = Path(trainer.save_dir) / "admission" / "distillation_preflight.json"
        raise RuntimeError(f"训练未出现可验证的首个 KD optimizer step，准入不完整：{report_path}")
    run_dir = Path(trainer.save_dir).resolve()
    best_weights = Path(trainer.best).resolve()
    if not best_weights.is_file():
        raise RuntimeError(f"训练完成后缺少 best.pt：{best_weights}")

    checkpoint_report = neck.pure_checkpoint_report(best_weights)
    final_model = YOLO(str(best_weights)).model.cpu().float().eval()
    final_structure = neck.core.model_structure_signature(final_model)
    final_state = neck.core.model_state_signature(final_model)
    structure_matched = final_structure["sha256"] == before_structure["sha256"]
    parameter_matched = final_structure["parameter_count"] == before_structure["parameter_count"]
    if checkpoint_report["status"] != "pass" or not structure_matched or not parameter_matched:
        failure = {
            "status": "fail",
            "checkpoint": checkpoint_report,
            "before_structure": before_structure,
            "after_structure": final_structure,
            "structure_matched": structure_matched,
            "parameter_matched": parameter_matched,
        }
        neck.write_json(run_dir / "distillation_checkpoint_failure.json", failure)
        raise RuntimeError("最佳 checkpoint 不是可独立加载的原物理学生结构。")

    after_stats = neck.core.model_stats(final_model, args.imgsz, device)
    final_validation = neck.core.validate_yolo_checkpoint(
        best_weights,
        args,
        project=run_dir / "independent_validation",
        name="best",
    )
    fixed_forward = neck.core.fixed_input_forward_report(
        best_weights,
        fixed_input,
        imgsz=args.imgsz,
        device=device,
        output_dir=run_dir / "fixed_input",
        report_name="distillation_best_fixed_input",
    )
    architecture_check = {
        "status": "pass",
        "student_source_structure": before_structure,
        "student_source_state": before_state,
        "best_structure": final_structure,
        "best_state": final_state,
        "structure_matched": structure_matched,
        "parameter_matched": parameter_matched,
        "before_stats": before_stats,
        "after_stats": after_stats,
        "checkpoint": checkpoint_report,
        "fixed_input": fixed_forward,
        "teacher_weights": str(teacher_weights),
        "teacher_sha256": neck.core.sha256_file(teacher_weights),
        "student_source_weights": str(student_weights),
        "student_source_sha256": neck.core.sha256_file(student_weights),
        "best_weights": str(best_weights),
        "best_sha256": neck.core.sha256_file(best_weights),
    }
    baseline_validation = {
        "map50": args.student_baseline_map50,
        "map50_95": args.student_baseline_map50_95,
        "source": args.student_baseline_source,
        "weights": str(student_weights),
        "weights_sha256": neck.core.sha256_file(student_weights),
    }
    phase = "蒸馏" if kd_enabled else "无KD对照"
    summary = neck.core.save_stage_summary(
        run_dir / "distillation_summary.json",
        phase=phase,
        started_at=started_at,
        source_weights=student_weights,
        final_weights=best_weights,
        before_stats=before_stats,
        after_stats=after_stats,
        final_validation=final_validation,
        epoch_reporter=trainer.epoch_reporter,
        baseline_validation=baseline_validation,
        architecture_check=architecture_check,
        validation_scope="独立 YOLO(best.pt) COCO val；训练期 teacher 与 criterion 已从 checkpoint 剥离。",
    )
    config = {
        "time_utc": neck.utc_now(),
        "distillation_id": run_label,
        "implementation": implementation,
        "mode": "train" if kd_enabled else "control",
        "command_arguments": vars(args),
        "comparison_contract": _common_training_contract(args, student_weights, teacher_weights),
        "teacher": {"path": str(teacher_weights), "sha256": neck.core.sha256_file(teacher_weights)},
        "student": {
            "path": str(student_weights),
            "sha256": neck.core.sha256_file(student_weights),
            "structure": before_structure,
            "stats": before_stats,
        },
        "prediction_contract": contract,
        "neck_contract": neck_contract,
        "fgd_contract": fgd_contract_name(fgd_component_flags) if fgd_enabled else None,
        "branches": {
            "prediction_kd_enabled": prediction_kd_enabled,
            "neck_sl2_enabled": neck_sl2_enabled,
            "fgd_enabled": fgd_enabled,
            "fgd_components": dict(fgd_component_flags) if fgd_enabled else None,
        },
        "loss": {
            "kd_enabled": kd_enabled,
            "prediction_kd_enabled": prediction_kd_enabled,
            "neck_sl2_enabled": neck_sl2_enabled,
            "cls_weight": args.cls_weight if prediction_kd_enabled else 0.0,
            "dfl_weight": args.dfl_weight if prediction_kd_enabled else 0.0,
            "cls_temperature": args.cls_temperature,
            "dfl_temperature": args.dfl_temperature,
            "confidence_threshold": args.confidence_threshold,
            "topk_per_image": args.topk_per_image,
            "selection_mode": args.selection_mode,
            "quality_topk_per_gt": args.quality_topk_per_gt,
            "kd_warmup_epochs": args.kd_warmup_epochs if prediction_kd_enabled else 0,
            "feature_weight": args.feature_weight if neck_sl2_enabled else 0.0,
            "feature_warmup_epochs": args.feature_warmup_epochs if neck_sl2_enabled else 0,
            "training_only_projector": (
                neck_contract["projector_architecture"] if neck_sl2_enabled and neck_contract is not None else None
            ),
            "fgd_weight": args.fgd_weight if fgd_enabled else 0.0,
            "fgd_warmup_epochs": args.fgd_warmup_epochs if fgd_enabled else 0,
            "fgd_temperature": args.fgd_temperature if fgd_enabled else 0.0,
            "fgd_foreground_weight": args.fgd_foreground_weight if fgd_enabled else 0.0,
            "fgd_background_weight": args.fgd_background_weight if fgd_enabled else 0.0,
            "fgd_mask_weight": args.fgd_mask_weight if fgd_enabled else 0.0,
            "fgd_relation_weight": args.fgd_relation_weight if fgd_enabled else 0.0,
            "fgd_components": dict(fgd_component_flags) if fgd_enabled else None,
            "fgd_active_components": list(fgd.active_fgd_component_names(fgd_component_flags)) if fgd_enabled else [],
            "fgd_loss_contract": fgd_loss_contract(fgd_component_flags) if fgd_enabled else None,
            "fgd_component_execution": (
                "selective_loss_forward_backward_with_registered_training_auxiliary_v1" if fgd_enabled else None
            ),
            "fgd_training_only_auxiliary": (
                fgd.OFFICIAL_FGD_AUXILIARY_ARCHITECTURE if fgd_enabled and neck_contract is not None else None
            ),
            "mask_policy": (
                QUALITYFG_MASK_CONTRACT
                if prediction_kd_enabled and args.selection_mode == "quality_fg_v2"
                else "per_image_global_threshold_or_topk_over_p3_p4_p5_v1"
                if prediction_kd_enabled
                else "not_applicable"
            ),
        },
    }
    neck.write_json(run_dir / "distillation_config.json", config)
    neck.write_json(run_dir / "distillation_metadata.json", {"summary": summary, "architecture_check": architecture_check})
    print("=" * 60, flush=True)
    print("蒸馏阶段完成" if kd_enabled else "无KD对照阶段完成", flush=True)
    print(f"最佳纯学生模型: {best_weights}", flush=True)
    print(
        f"独立验证: mAP@0.5 {neck.core.format_percent(final_validation.get('map50'))}, "
        f"mAP@0.5:0.95 {neck.core.format_percent(final_validation.get('map50_95'))}",
        flush=True,
    )
    print(f"详细摘要: {run_dir / 'distillation_summary.json'}", flush=True)
    print("=" * 60, flush=True)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--teacher", required=True, help="未剪枝教师 checkpoint；control 也必须填写以冻结比较合同")
    parser.add_argument(
        "--student",
        required=True,
        help="物理学生 checkpoint；可为剪枝后结构化子网或已完成普通恢复的 best.pt。",
    )
    parser.add_argument("--data", required=True, help="COCO 数据集 YAML")
    parser.add_argument("--fixed-input", required=True, help="固定输入图像，用于最终前向验收")
    parser.add_argument("--output", default="runs", help="Ultralytics 输出根目录")
    parser.add_argument("--name", required=True, help="唯一 run 名称")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default="0")
    parser.add_argument("--lr0", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--close-mosaic", type=int, default=20)
    parser.add_argument(
        "--enable-prediction-kd",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="预测级分类+DFL KD 开关；train 默认开启，control 默认关闭。",
    )
    parser.add_argument(
        "--enable-neck-sl2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="训练期 P3/P4/P5 score-weighted L2 特征 KD；启用时自动注册并在保存前剥离 projector。",
    )
    parser.add_argument(
        "--enable-fgd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="FGD 特征蒸馏总开关；默认四项子损失全开，训练期辅助模块在保存前剥离。",
    )
    parser.add_argument(
        "--fgd-enable-foreground",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="FGD 前景特征损失开关；默认开启。仅在 --enable-fgd 时生效。",
    )
    parser.add_argument(
        "--fgd-enable-background",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="FGD 背景特征损失开关；默认开启。仅在 --enable-fgd 时生效。",
    )
    parser.add_argument(
        "--fgd-enable-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="FGD 空间/通道注意力 mask 损失开关；默认开启。仅在 --enable-fgd 时生效。",
    )
    parser.add_argument(
        "--fgd-enable-relation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="FGD 全局关系损失开关；默认开启。仅在 --enable-fgd 时生效。",
    )
    parser.add_argument("--cls-weight", type=float, default=0.5)
    parser.add_argument("--dfl-weight", type=float, default=0.25)
    parser.add_argument("--cls-temperature", type=float, default=2.0)
    parser.add_argument("--dfl-temperature", type=float, default=2.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.10)
    parser.add_argument("--topk-per-image", type=int, default=100)
    parser.add_argument(
        "--selection-mode",
        choices=["global_score_v1", "quality_fg_v2"],
        default="global_score_v1",
        help="global_score_v1 保持 A1 行为；quality_fg_v2 使用每 GT 跨尺度质量 Top-K。",
    )
    parser.add_argument(
        "--quality-topk-per-gt",
        type=int,
        default=DEFAULT_QUALITY_TOPK_PER_GT,
        help="QualityFG 每 GT 的跨尺度 Top-K；默认 15 约匹配 A1 诊断中的平均选点预算。",
    )
    parser.add_argument("--kd-warmup-epochs", type=int, default=5)
    parser.add_argument("--feature-weight", type=float, default=3.0, help="Neck-SL2 分支系数；仅启用该分支时生效。")
    parser.add_argument(
        "--feature-warmup-epochs",
        type=int,
        default=5,
        help="Neck-SL2 分支线性升温周期；仅启用该分支时生效。",
    )
    parser.add_argument("--fgd-weight", type=float, default=1.0, help="官方 FGD 四项子损失之外的总体系数；只在 --enable-fgd 时生效。")
    parser.add_argument("--fgd-warmup-epochs", type=int, default=5, help="完整 FGD 线性升温周期。")
    parser.add_argument("--fgd-temperature", type=float, default=fgd.DEFAULT_TEMPERATURE, help="官方空间/通道 softmax 注意力温度。")
    parser.add_argument("--fgd-foreground-weight", type=float, default=fgd.DEFAULT_ALPHA_FGD, help="官方 FGD alpha：前景特征损失权重。")
    parser.add_argument("--fgd-background-weight", type=float, default=fgd.DEFAULT_BETA_FGD, help="官方 FGD beta：背景特征损失权重。")
    parser.add_argument("--fgd-mask-weight", type=float, default=fgd.DEFAULT_GAMMA_FGD, help="官方 FGD gamma：师生注意力 mask 损失权重。")
    parser.add_argument("--fgd-relation-weight", type=float, default=fgd.DEFAULT_LAMBDA_FGD, help="官方 FGD lambda：GC 全局关系损失权重。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--student-baseline-map50", type=float, default=0.529458)
    parser.add_argument("--student-baseline-map50-95", type=float, default=0.371384)
    parser.add_argument(
        "--student-baseline-source",
        default="P30_S0_finetuned_physical_student",
        help="写入摘要的学生起点语义；KD 与 control 必须完全一致。",
    )
    parser.add_argument("--admission-probe", action="store_true", help="首个真实 optimizer step 验证梯度与纯 checkpoint 合同")
    parser.add_argument("--plots", action="store_true", help="保留 Ultralytics plots；默认关闭")
    parser.add_argument("--no-overwrite", action="store_true", help="目录存在即失败，保护历史产物")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    control = subparsers.add_parser("control", help="同入口 matched no-KD continuation control")
    train = subparsers.add_parser("train", help="执行已启用的预测级、Neck-SL2 或官方 FGD 蒸馏路线")
    add_common_arguments(control)
    add_common_arguments(train)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_train(args)


if __name__ == "__main__":
    main()
