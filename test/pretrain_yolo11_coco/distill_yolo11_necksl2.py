#!/usr/bin/env python
"""YOLO11 物理剪枝子网的独立 Neck score-weighted L2 蒸馏入口。

本脚本面向服务器上固定的 ``ultralytics==8.3.246``。它不升级 Ultralytics，
也不修改 OTO/HESSO；仅复用 ``prune_yolo11_adamcf.py`` 已验证的物理子网直接
装载、统计、验证与固定输入前向工具。

实现参考当前 Ultralytics 主线 ``DistillationModel`` 的 Detect 输入 Neck 特征
和 score-weighted L2（SL2）思想，但不复制其 wrapper：训练时的 projector、
教师与 criterion 均不得进入最终部署 checkpoint。

部署前请把本文件同步到：
  /workspace/OTOv2_v2/test/pretrain_yolo11_coco/distill_yolo11_necksl2.py

示例：
  python -u distill_yolo11_necksl2.py train --teacher <teacher.pt> \
    --student <physical_student_best.pt> --data <coco.yaml> \
    --fixed-input <image.jpg> --name <unique_run_id> --epochs 1
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
from torch import nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

import prune_yolo11_adamcf as core


DISTILLATION_ID = "Y11-NeckSL2-KD-v1"
DISTILLATION_IMPLEMENTATION = "legacy_8_3_246_minimal_necksl2_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在：{path}")
    return path


def resolve_device(value: str):
    device = core.device_from_arg(value)
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def find_detect_module(model: nn.Module) -> tuple[int, nn.Module]:
    """返回最终 Detect 的模块序号和对象；不依赖 YOLO11 的固定层号。"""

    raw = unwrap_model(model)
    layers = getattr(raw, "model", None)
    if layers is None:
        raise RuntimeError("模型不含 model 层序列，无法定位 Detect。")
    detected: list[tuple[int, nn.Module]] = []
    for index, module in enumerate(layers):
        if type(module).__name__ == "Detect" or (
            hasattr(module, "f") and hasattr(module, "nc") and hasattr(module, "reg_max")
        ):
            detected.append((index, module))
    if len(detected) != 1:
        raise RuntimeError(f"必须且只能找到一个 Detect，实际找到 {len(detected)} 个。")
    return detected[0]


def normalize_layer_index(index: int, layer_count: int) -> int:
    normalized = index if index >= 0 else layer_count + index
    if not 0 <= normalized < layer_count:
        raise RuntimeError(f"Detect.f 包含越界层索引：{index}（层数 {layer_count}）。")
    return normalized


class FeatureCapture:
    """捕获多个指定层的单次前向输出，且不改变图与梯度流。"""

    def __init__(self, model: nn.Module, layer_indices: list[int], label: str):
        raw = unwrap_model(model)
        layers = getattr(raw, "model", None)
        if layers is None:
            raise RuntimeError(f"{label}模型缺少 model 层序列。")
        self.label = label
        self.layer_indices = list(layer_indices)
        self.features: dict[int, torch.Tensor] = {}
        self.handles = []
        for layer_index in self.layer_indices:
            module = layers[layer_index]
            self.handles.append(module.register_forward_hook(self._make_hook(layer_index)))

    def _make_hook(self, layer_index: int):
        def hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                if len(output) != 1 or not isinstance(output[0], torch.Tensor):
                    raise RuntimeError(
                        f"{self.label}层 {layer_index} 输出不是单一 Tensor，"
                        "不能作为 Neck 蒸馏特征。"
                    )
                output = output[0]
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                raise RuntimeError(
                    f"{self.label}层 {layer_index} 输出必须为四维 Tensor，"
                    f"实际为 {type(output).__name__}/{getattr(output, 'shape', None)}。"
                )
            self.features[layer_index] = output

        return hook

    def ordered(self) -> list[torch.Tensor]:
        missing = [index for index in self.layer_indices if index not in self.features]
        if missing:
            raise RuntimeError(f"{self.label}未捕获到 Detect 输入层特征：{missing}。")
        return [self.features[index] for index in self.layer_indices]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def extract_raw_detect_outputs(output: Any) -> list[torch.Tensor]:
    """从 8.3.246 的 eval Detect 输出中严格取出三尺度原始 logits。"""

    def as_raw_layers(value: Any) -> list[torch.Tensor] | None:
        if isinstance(value, (tuple, list)) and value and all(
            isinstance(item, torch.Tensor) and item.ndim == 4 for item in value
        ):
            return list(value)
        return None

    direct = as_raw_layers(output)
    if direct is not None:
        return direct
    if isinstance(output, (tuple, list)):
        # Ultralytics Detect.eval() 通常返回 (decoded, raw_layers)。
        for item in output[1:]:
            nested = as_raw_layers(item)
            if nested is not None:
                return nested
    raise RuntimeError(
        "无法从教师 eval 输出中提取三尺度原始 Detect logits；"
        "禁止使用 decoded/NMS 输出代替 score map。"
    )


def frozen_parameter_report(model: nn.Module) -> dict[str, Any]:
    parameters = list(model.parameters())
    return {
        "parameter_count": len(parameters),
        "parameter_numel": sum(parameter.numel() for parameter in parameters),
        "requires_grad_count": sum(int(parameter.requires_grad) for parameter in parameters),
        "gradient_present_count": sum(int(parameter.grad is not None) for parameter in parameters),
    }


def optimizer_ownership_report(optimizer, projectors: nn.ModuleList, teacher: nn.Module) -> dict[str, Any]:
    ownership: dict[int, int] = {}
    for group in optimizer.param_groups:
        for parameter in group.get("params", []):
            ownership[id(parameter)] = ownership.get(id(parameter), 0) + 1

    projector_parameters = list(projectors.parameters())
    teacher_parameters = list(teacher.parameters())
    missing_projectors = [
        index for index, parameter in enumerate(projector_parameters) if ownership.get(id(parameter), 0) != 1
    ]
    teacher_in_optimizer = [
        index for index, parameter in enumerate(teacher_parameters) if ownership.get(id(parameter), 0) != 0
    ]
    return {
        "status": "pass" if not missing_projectors and not teacher_in_optimizer else "fail",
        "projector_parameter_count": len(projector_parameters),
        "projector_parameters_missing_or_duplicated": missing_projectors,
        "teacher_parameter_count": len(teacher_parameters),
        "teacher_parameters_in_optimizer": teacher_in_optimizer,
        "expected_projector_ownership": 1,
        "expected_teacher_ownership": 0,
    }


def physical_student_copy(model: nn.Module) -> nn.Module:
    """复制纯部署学生，临时训练对象不会泄露到保存副本中。"""

    raw = unwrap_model(model)
    had_criterion = hasattr(raw, "criterion")
    original_criterion = getattr(raw, "criterion", None)
    if had_criterion:
        raw.criterion = None
    try:
        physical = copy.deepcopy(raw).cpu().float()
    finally:
        if had_criterion:
            raw.criterion = original_criterion
    if hasattr(physical, "_distill_projectors"):
        delattr(physical, "_distill_projectors")
    # 旧版 Full FGD projector 名称保留剥离逻辑，确保历史临时模型也能安全导出。
    if hasattr(physical, "_distill_fgd_projectors"):
        delattr(physical, "_distill_fgd_projectors")
    # 官方 FGD 的 align、attention mask 与 GC relation 均是训练期辅助模块；
    # 必须整体从部署学生中剥离，不能泄露到 checkpoint 或影响历史模型。
    if hasattr(physical, "_distill_fgd_modules"):
        delattr(physical, "_distill_fgd_modules")
    # CrossHeadAdapter-v2 与 Neck-SL2 projector 一样都只是训练期辅助模块；
    # 明确按属性名剥离，不影响任何历史物理学生或既有 OTO/HESSO 结构。
    if hasattr(physical, "_distill_cross_head_adapters"):
        delattr(physical, "_distill_cross_head_adapters")
    if hasattr(physical, "criterion"):
        physical.criterion = None
    return physical


def pure_checkpoint_report(weights: Path) -> dict[str, Any]:
    """只依赖普通 YOLO 加载器检查最佳 checkpoint 不含训练期对象。"""

    reloaded = YOLO(str(weights)).model.cpu().float().eval()
    forbidden = [
        name
        for name, _module in reloaded.named_modules()
        if "distill" in name.lower() or "projector" in name.lower() or "teacher" in name.lower()
    ]
    return {
        "status": "pass" if not forbidden else "fail",
        "weights": str(weights),
        "weights_sha256": core.sha256_file(weights),
        "structure": core.model_structure_signature(reloaded),
        "state": core.model_state_signature(reloaded),
        "forbidden_training_module_names": forbidden,
        "criterion_is_none": getattr(reloaded, "criterion", None) is None,
    }


def tensor_shape(tensor: torch.Tensor) -> list[int]:
    return [int(value) for value in tensor.shape]


def build_neck_contract(
    student_template: nn.Module,
    teacher_template: nn.Module,
    *,
    imgsz: int,
) -> dict[str, Any]:
    """在训练开始前冻结三个 Detect 输入特征与教师 raw logits 合同。"""

    student = copy.deepcopy(unwrap_model(student_template)).cpu().float().eval()
    teacher = copy.deepcopy(unwrap_model(teacher_template)).cpu().float().eval()
    student_detect_index, student_detect = find_detect_module(student)
    teacher_detect_index, teacher_detect = find_detect_module(teacher)
    student_layers = getattr(student, "model")
    teacher_layers = getattr(teacher, "model")
    student_f = [normalize_layer_index(int(index), len(student_layers)) for index in list(student_detect.f)]
    teacher_f = [normalize_layer_index(int(index), len(teacher_layers)) for index in list(teacher_detect.f)]
    if len(student_f) != 3 or len(teacher_f) != 3:
        raise RuntimeError(f"Detect 输入必须为三尺度，student={student_f}，teacher={teacher_f}。")
    if student_f != teacher_f:
        raise RuntimeError(f"师生 Detect.f 不一致：student={student_f}，teacher={teacher_f}。")
    if int(student_detect.nc) != int(teacher_detect.nc) or int(student_detect.reg_max) != int(teacher_detect.reg_max):
        raise RuntimeError(
            "师生 Detect 语义不一致："
            f"student(nc={student_detect.nc}, reg_max={student_detect.reg_max})，"
            f"teacher(nc={teacher_detect.nc}, reg_max={teacher_detect.reg_max})。"
        )

    student_capture = FeatureCapture(student, student_f, "student_contract")
    teacher_capture = FeatureCapture(teacher, teacher_f, "teacher_contract")
    try:
        image = torch.zeros(1, 3, imgsz, imgsz)
        with torch.no_grad():
            _ = student(image)
            teacher_output = teacher(image)
        student_features = student_capture.ordered()
        teacher_features = teacher_capture.ordered()
        teacher_raw = extract_raw_detect_outputs(teacher_output)
    finally:
        student_capture.close()
        teacher_capture.close()

    if len(teacher_raw) != 3:
        raise RuntimeError(f"教师原始 Detect 输出必须为三尺度，实际为 {len(teacher_raw)}。")
    no = 4 * int(teacher_detect.reg_max) + int(teacher_detect.nc)
    records: list[dict[str, Any]] = []
    for level, (student_feature, teacher_feature, raw) in enumerate(
        zip(student_features, teacher_features, teacher_raw), start=3
    ):
        if student_feature.shape[0] != teacher_feature.shape[0] or student_feature.shape[-2:] != teacher_feature.shape[-2:]:
            raise RuntimeError(
                f"P{level} Neck 空间合同失败：student={tensor_shape(student_feature)}，"
                f"teacher={tensor_shape(teacher_feature)}。"
            )
        if raw.ndim != 4 or raw.shape[0] != teacher_feature.shape[0] or raw.shape[1] != no:
            raise RuntimeError(
                f"P{level} 教师 raw logits 合同失败：raw={tensor_shape(raw)}，预期 no={no}。"
            )
        if raw.shape[-2:] != teacher_feature.shape[-2:]:
            raise RuntimeError(
                f"P{level} score map 与 Neck 特征空间不一致：raw={tensor_shape(raw)}，"
                f"feature={tensor_shape(teacher_feature)}。"
            )
        records.append(
            {
                "level": f"P{level}",
                "layer_index": student_f[level - 3],
                "student_feature_shape": tensor_shape(student_feature),
                "teacher_feature_shape": tensor_shape(teacher_feature),
                "teacher_raw_shape": tensor_shape(raw),
                "student_channels": int(student_feature.shape[1]),
                "teacher_channels": int(teacher_feature.shape[1]),
                "spatial_shape": [int(student_feature.shape[-2]), int(student_feature.shape[-1])],
            }
        )

    return {
        "status": "pass",
        "distillation_id": DISTILLATION_ID,
        "detect": {
            "student_index": student_detect_index,
            "teacher_index": teacher_detect_index,
            "student_f": student_f,
            "teacher_f": teacher_f,
            "nc": int(teacher_detect.nc),
            "reg_max": int(teacher_detect.reg_max),
            "no": no,
        },
        "feature_levels": records,
        "projector_architecture": "Conv1x1(C_student,C_teacher) -> ReLU -> Conv1x1(C_teacher,C_teacher)",
        "input_shape": [1, 3, imgsz, imgsz],
    }


def make_projectors(contract: dict[str, Any]) -> nn.ModuleList:
    return nn.ModuleList(
        [
            nn.Sequential(
                nn.Conv2d(record["student_channels"], record["teacher_channels"], kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(record["teacher_channels"], record["teacher_channels"], kernel_size=1, bias=True),
            )
            for record in contract["feature_levels"]
        ]
    )


class KDMeter:
    def __init__(self) -> None:
        self._values: dict[str, float] = {}
        self._count = 0
        self.latest: dict[str, Any] = {}

    def update(self, values: dict[str, float]) -> None:
        self._count += 1
        for key, value in values.items():
            self._values[key] = self._values.get(key, 0.0) + float(value)
        self.latest = dict(values)

    def consume_epoch(self) -> dict[str, Any]:
        if self._count <= 0:
            return {"batches": 0}
        output = {"batches": self._count}
        output.update({key: value / self._count for key, value in self._values.items()})
        self._values = {}
        self._count = 0
        return output


class NeckSL2Criterion:
    """保留原 Ultralytics task loss，仅额外加入训练期三尺度 SL2。"""

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        contract: dict[str, Any],
        *,
        feature_weight: float,
        warmup_epochs: int,
        trainer,
    ) -> None:
        self.student = unwrap_model(student)
        self.teacher = teacher.eval()
        self.contract = contract
        self.feature_weight = float(feature_weight)
        self.warmup_epochs = int(warmup_epochs)
        self.trainer = trainer
        self.base_criterion = self.student.init_criterion()
        layer_indices = list(contract["detect"]["student_f"])
        self.student_capture = FeatureCapture(self.student, layer_indices, "student_train")
        self.teacher_capture = FeatureCapture(self.teacher, layer_indices, "teacher_train")
        self.detect = find_detect_module(self.teacher)[1]

    def close(self) -> None:
        self.student_capture.close()
        self.teacher_capture.close()

    def warmup_factor(self) -> float:
        if self.warmup_epochs <= 0:
            return 1.0
        return min(1.0, float(int(self.trainer.epoch) + 1) / float(self.warmup_epochs))

    def __call__(self, preds, batch):
        task_loss, task_items = self.base_criterion(preds, batch)
        # Ultralytics 8.3.246 的 v8DetectionLoss 返回已按 batch_size 缩放的
        # 三分量向量 [box, cls, dfl]，而 Trainer 会在 backward 前自行 sum()。
        # 蒸馏项必须与其总 task loss 相加，不能把这个向量直接当作 Python scalar。
        task_loss_total = task_loss.sum()
        student_features = self.student_capture.ordered()
        with torch.no_grad():
            teacher_output = self.teacher(batch["img"])
            teacher_features = [feature.detach() for feature in self.teacher_capture.ordered()]
            teacher_raw = [tensor.detach() for tensor in extract_raw_detect_outputs(teacher_output)]

        if not (len(student_features) == len(teacher_features) == len(teacher_raw) == len(self.student._distill_projectors)):
            raise RuntimeError("运行时 Neck 特征、教师 raw 输出或 projector 数量不一致。")

        expected_no = int(self.contract["detect"]["no"])
        reg_max = int(self.detect.reg_max)
        nc = int(self.detect.nc)
        level_losses: list[torch.Tensor] = []
        level_scores: list[float] = []
        for record, projector, student_feature, teacher_feature, raw in zip(
            self.contract["feature_levels"],
            self.student._distill_projectors,
            student_features,
            teacher_features,
            teacher_raw,
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
                    f"level={record['level']} student={tensor_shape(student_feature)} "
                    f"teacher={tensor_shape(teacher_feature)} raw={tensor_shape(raw)}。"
                )
            projected = projector(student_feature)
            if projected.shape != teacher_feature.shape:
                raise RuntimeError(
                    f"{record['level']} projector 输出形状错误："
                    f"projected={tensor_shape(projected)} teacher={tensor_shape(teacher_feature)}。"
                )
            class_logits = raw[:, 4 * reg_max : 4 * reg_max + nc]
            score_map = class_logits.sigmoid().amax(dim=1, keepdim=True)
            numerator = ((projected - teacher_feature).pow(2) * score_map).sum()
            denominator = score_map.sum() * teacher_feature.shape[1] + 1e-12
            level_losses.append(numerator / denominator)
            level_scores.append(float(score_map.detach().mean().cpu().item()))

        feature_loss = torch.stack(level_losses).mean()
        warmup = self.warmup_factor()
        feature_term = float(self.feature_weight) * warmup * feature_loss * batch["img"].shape[0]
        total_loss = task_loss_total + feature_term
        latest = {
            "task_loss": float(task_loss_total.detach().float().cpu().item()),
            "feature_kd_loss": float(feature_loss.detach().float().cpu().item()),
            "feature_term": float(feature_term.detach().float().cpu().item()),
            "total_loss": float(total_loss.detach().float().cpu().item()),
            "feature_weight": float(self.feature_weight),
            "warmup_factor": warmup,
            "effective_feature_weight": float(self.feature_weight) * warmup,
            "teacher_score_mean": sum(level_scores) / len(level_scores),
        }
        self.trainer.kd_meter.update(latest)
        return total_loss, task_items


class NeckSL2EpochReporter(core.EpochReporter):
    """沿用现有 epoch 摘要，并在同一条 epoch 记录中加入蒸馏分项。"""

    def __init__(self) -> None:
        super().__init__("distillation")
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
        if kd:
            print(
                "NeckSL2: "
                f"task={kd.get('task_loss', float('nan')):.6f}, "
                f"feature={kd.get('feature_kd_loss', float('nan')):.6f}, "
                f"term={kd.get('feature_term', float('nan')):.6f}, "
                f"w={kd.get('effective_feature_weight', float('nan')):.4f}, "
                f"score_mean={kd.get('teacher_score_mean', float('nan')):.6f}",
                flush=True,
            )


def create_necksl2_trainer_class(
    student_template: nn.Module,
    teacher_template: nn.Module,
    student_weights: Path,
    teacher_weights: Path,
    contract: dict[str, Any],
    *,
    feature_weight: float,
    feature_warmup_epochs: int,
    admission_probe: bool,
):
    """返回只训练物理学生 + projector 的 8.3.246 DetectionTrainer。"""

    expected_physical = physical_student_copy(student_template)
    expected_structure = core.model_structure_signature(expected_physical)
    expected_state = core.model_state_signature(expected_physical)
    student_sha256 = core.sha256_file(student_weights)
    teacher_sha256 = core.sha256_file(teacher_weights)

    class NeckSL2Trainer(DetectionTrainer):
        def __init__(self, *trainer_args, **trainer_kwargs):
            self.epoch_reporter = NeckSL2EpochReporter()
            self.kd_meter = KDMeter()
            self.structure_check: dict[str, Any] = {}
            self.preflight: dict[str, Any] = {}
            self.distill_criterion: NeckSL2Criterion | None = None
            super().__init__(*trainer_args, **trainer_kwargs)

        def get_model(self, cfg=None, weights=None, verbose=True):
            del cfg, weights
            model = physical_student_copy(student_template).float()
            model._distill_projectors = make_projectors(contract)
            if verbose and hasattr(model, "info"):
                model.info(verbose=True)
            return model

        def _setup_train(self):
            super()._setup_train()
            if int(getattr(self, "world_size", 1)) != 1:
                raise RuntimeError("Y11-NeckSL2-KD-v1 当前只允许单进程单卡训练，禁止 DDP 静默运行。")

            physical_before = physical_student_copy(self.model)
            actual_structure = core.model_structure_signature(physical_before)
            actual_state = core.model_state_signature(physical_before)
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
                raise RuntimeError("训练器未直接装载输入物理学生，禁止继续蒸馏。")

            teacher = copy.deepcopy(unwrap_model(teacher_template)).to(self.device).float().eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            self.distill_criterion = NeckSL2Criterion(
                self.model,
                teacher,
                contract,
                feature_weight=feature_weight,
                warmup_epochs=feature_warmup_epochs,
                trainer=self,
            )
            unwrap_model(self.model).criterion = self.distill_criterion
            ownership = optimizer_ownership_report(self.optimizer, unwrap_model(self.model)._distill_projectors, teacher)
            frozen = frozen_parameter_report(teacher)
            self.preflight = {
                "status": "pass" if ownership["status"] == "pass" and frozen["requires_grad_count"] == 0 else "fail",
                "time_utc": utc_now(),
                "implementation": DISTILLATION_IMPLEMENTATION,
                "distillation_id": DISTILLATION_ID,
                "teacher": {
                    "weights": str(teacher_weights),
                    "sha256": teacher_sha256,
                    "frozen": frozen,
                },
                "student": {
                    "weights": str(student_weights),
                    "sha256": student_sha256,
                    "physical_structure": expected_structure,
                },
                "neck_contract": contract,
                "optimizer_ownership": ownership,
                "feature_weight": feature_weight,
                "feature_warmup_epochs": feature_warmup_epochs,
                "loss_contract": "L_total=L_det + lambda_feat * warmup * mean_P3P4P5(L_SL2) * batch_size",
                "checkpoint_contract": "checkpoint_only_contains_physical_student_no_teacher_no_projector_no_criterion",
            }
            report_path = Path(self.save_dir) / "admission" / "distillation_preflight.json"
            core.json_dump(report_path, self.preflight)
            if self.preflight["status"] != "pass":
                raise RuntimeError(f"NeckSL2 训练所有权或教师冻结检查失败：{report_path}")
            self.epoch_reporter.attach(self)

        def optimizer_step(self):
            first_check = admission_probe and "backward" not in self.preflight
            report_path = Path(self.save_dir) / "admission" / "distillation_preflight.json"
            if first_check:
                assert self.distill_criterion is not None
                raw_student = unwrap_model(self.model)
                projector_parameters = list(raw_student._distill_projectors.parameters())
                trainable = [parameter for parameter in raw_student.parameters() if parameter.requires_grad]
                nonzero_student = sum(
                    int(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0)
                    for parameter in trainable
                )
                nonzero_projector = sum(
                    int(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0)
                    for parameter in projector_parameters
                )
                non_finite = [
                    name
                    for name, parameter in raw_student.named_parameters()
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
                ]
                latest = dict(self.kd_meter.latest)
                finite_losses = bool(latest) and all(math.isfinite(float(value)) for value in latest.values())
                teacher_frozen = frozen_parameter_report(self.distill_criterion.teacher)
                passed = (
                    finite_losses
                    and nonzero_student > 0
                    and nonzero_projector > 0
                    and not non_finite
                    and teacher_frozen["requires_grad_count"] == 0
                    and teacher_frozen["gradient_present_count"] == 0
                )
                self.preflight["backward"] = {
                    "status": "pass" if passed else "fail",
                    "latest_loss": latest,
                    "trainable_student_and_projector_parameter_count": len(trainable),
                    "student_parameters_with_nonzero_gradient": nonzero_student,
                    "projector_parameters_with_nonzero_gradient": nonzero_projector,
                    "non_finite_gradient_parameter_names": non_finite,
                    "teacher_after_backward": teacher_frozen,
                    "optimizer_step_completed": False,
                }
                core.json_dump(report_path, self.preflight)
                if not passed:
                    raise RuntimeError(f"NeckSL2 首次反向传播准入失败：{report_path}")
            super().optimizer_step()
            if first_check:
                self.preflight["backward"]["optimizer_step_completed"] = True
                core.json_dump(report_path, self.preflight)

        def save_model(self):
            """显式保存纯学生，避免 default checkpoint 固化教师或 projector。"""

            ema_source = getattr(getattr(self, "ema", None), "ema", None)
            source = ema_source if ema_source is not None else unwrap_model(self.model)
            student_only = physical_student_copy(source).half()
            checkpoint = {
                "epoch": int(getattr(self, "epoch", -1)),
                "best_fitness": getattr(self, "best_fitness", None),
                "model": None,
                "ema": student_only,
                "updates": int(getattr(getattr(self, "ema", None), "updates", 0)),
                "optimizer": None,
                "train_args": vars(self.args).copy(),
                "date": utc_now(),
                "version": str(getattr(core.ultralytics, "__version__", "unknown")),
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

    return NeckSL2Trainer


def write_json(path: Path, value: dict[str, Any]) -> None:
    core.json_dump(path, value)


def run_train(args: argparse.Namespace) -> None:
    teacher_weights = require_file(args.teacher, "教师权重")
    student_weights = require_file(args.student, "学生权重")
    data_yaml = require_file(args.data, "数据 YAML")
    fixed_input = require_file(args.fixed_input, "固定输入图像")
    output_root = Path(args.output).expanduser().resolve()
    requested_run_dir = output_root / args.name
    if args.no_overwrite and requested_run_dir.exists():
        raise FileExistsError(f"禁止覆盖已有蒸馏目录：{requested_run_dir}；请更换 --name。")
    if not 0 < args.feature_weight:
        raise ValueError("--feature-weight 必须大于 0。")
    if args.feature_warmup_epochs < 0:
        raise ValueError("--feature-warmup-epochs 必须 >= 0。")
    if args.epochs < 1:
        raise ValueError("--epochs 必须 >= 1。")

    core.set_seed(args.seed)
    device = resolve_device(args.device)
    started_at = time.time()
    print("=" * 60, flush=True)
    print(f"开始 {DISTILLATION_ID} 独立蒸馏", flush=True)
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
    before_physical = physical_student_copy(student_template)
    before_structure = core.model_structure_signature(before_physical)
    before_state = core.model_state_signature(before_physical)
    before_stats = core.model_stats(before_physical, args.imgsz, device)
    contract = build_neck_contract(student_template, teacher_template, imgsz=args.imgsz)

    trainer_class = create_necksl2_trainer_class(
        student_template,
        teacher_template,
        student_weights,
        teacher_weights,
        contract,
        feature_weight=args.feature_weight,
        feature_warmup_epochs=args.feature_warmup_epochs,
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
    run_dir = Path(trainer.save_dir).resolve()
    best_weights = Path(trainer.best).resolve()
    if not best_weights.is_file():
        raise RuntimeError(f"蒸馏完成后缺少 best.pt：{best_weights}")

    checkpoint_report = pure_checkpoint_report(best_weights)
    final_model = YOLO(str(best_weights)).model.cpu().float().eval()
    final_structure = core.model_structure_signature(final_model)
    final_state = core.model_state_signature(final_model)
    structure_matched = final_structure["sha256"] == before_structure["sha256"]
    parameter_matched = final_structure["parameter_count"] == before_structure["parameter_count"]
    if not checkpoint_report["status"] == "pass" or not structure_matched or not parameter_matched:
        failure = {
            "status": "fail",
            "checkpoint": checkpoint_report,
            "before_structure": before_structure,
            "after_structure": final_structure,
            "structure_matched": structure_matched,
            "parameter_matched": parameter_matched,
        }
        write_json(run_dir / "distillation_checkpoint_failure.json", failure)
        raise RuntimeError("蒸馏 best.pt 不是可独立加载的原物理学生结构。")

    after_stats = core.model_stats(final_model, args.imgsz, device)
    final_validation = core.validate_yolo_checkpoint(
        best_weights,
        args,
        project=run_dir / "independent_validation",
        name="best",
    )
    fixed_forward = core.fixed_input_forward_report(
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
        "teacher_sha256": core.sha256_file(teacher_weights),
        "student_source_weights": str(student_weights),
        "student_source_sha256": core.sha256_file(student_weights),
        "best_weights": str(best_weights),
        "best_sha256": core.sha256_file(best_weights),
    }
    baseline_validation = {
        "map50": args.student_baseline_map50,
        "map50_95": args.student_baseline_map50_95,
        "source": "frozen_from_06B_1",
        "weights": str(student_weights),
        "weights_sha256": core.sha256_file(student_weights),
    }
    summary = core.save_stage_summary(
        run_dir / "distillation_summary.json",
        phase="蒸馏",
        started_at=started_at,
        source_weights=student_weights,
        final_weights=best_weights,
        before_stats=before_stats,
        after_stats=after_stats,
        final_validation=final_validation,
        epoch_reporter=trainer.epoch_reporter,
        baseline_validation=baseline_validation,
        architecture_check=architecture_check,
        validation_scope="独立 YOLO(best.pt) COCO val；训练期 projector、教师与 criterion 已从 checkpoint 剥离。",
    )
    config = {
        "time_utc": utc_now(),
        "distillation_id": DISTILLATION_ID,
        "implementation": DISTILLATION_IMPLEMENTATION,
        "command_arguments": vars(args),
        "teacher": {"path": str(teacher_weights), "sha256": core.sha256_file(teacher_weights)},
        "student": {
            "path": str(student_weights),
            "sha256": core.sha256_file(student_weights),
            "structure": before_structure,
            "stats": before_stats,
        },
        "neck_contract": contract,
        "loss": {
            "feature_weight": args.feature_weight,
            "feature_warmup_epochs": args.feature_warmup_epochs,
            "reduction": "mean_of_three_score_weighted_l2_levels",
            "training_only_projector": contract["projector_architecture"],
        },
    }
    write_json(run_dir / "distillation_config.json", config)
    write_json(run_dir / "distillation_metadata.json", {"summary": summary, "architecture_check": architecture_check})
    print("=" * 60, flush=True)
    print("蒸馏阶段完成", flush=True)
    print(f"最佳纯学生模型: {best_weights}", flush=True)
    print(
        f"独立验证: mAP@0.5 {core.format_percent(final_validation.get('map50'))}, "
        f"mAP@0.5:0.95 {core.format_percent(final_validation.get('map50_95'))}",
        flush=True,
    )
    print(f"详细摘要: {run_dir / 'distillation_summary.json'}", flush=True)
    print("=" * 60, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    train = subparsers.add_parser("train", help="执行独立 Neck score-weighted L2 蒸馏")
    train.add_argument("--teacher", required=True, help="未剪枝教师 checkpoint")
    train.add_argument("--student", required=True, help="P30 物理子网普通微调后的 best.pt")
    train.add_argument("--data", required=True, help="COCO 数据集 YAML")
    train.add_argument("--fixed-input", required=True, help="固定输入图像，用于最终前向验收")
    train.add_argument("--output", default="runs", help="Ultralytics 输出根目录")
    train.add_argument("--name", required=True, help="唯一 run 名称")
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=64)
    train.add_argument("--workers", type=int, default=12)
    train.add_argument("--device", default="0")
    train.add_argument("--lr0", type=float, default=5e-4)
    train.add_argument("--momentum", type=float, default=0.937)
    train.add_argument("--weight-decay", type=float, default=5e-4)
    train.add_argument("--close-mosaic", type=int, default=20)
    train.add_argument("--feature-weight", type=float, default=3.0)
    train.add_argument("--feature-warmup-epochs", type=int, default=5)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--student-baseline-map50", type=float, default=0.5295)
    train.add_argument("--student-baseline-map50-95", type=float, default=0.3714)
    train.add_argument("--admission-probe", action="store_true", help="首个真实 optimizer step 执行教师/梯度准入检查")
    train.add_argument("--plots", action="store_true", help="保留 Ultralytics plots；默认关闭")
    train.add_argument("--no-overwrite", action="store_true", help="目录已存在即失败，保护历史产物")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "train":
        run_train(args)
    else:  # pragma: no cover - argparse 已限制 mode
        raise ValueError(f"未知模式：{args.mode}")


if __name__ == "__main__":
    main()
