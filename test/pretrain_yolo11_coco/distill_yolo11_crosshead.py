#!/usr/bin/env python
"""YOLO11 物理剪枝学生的检测头跨头预测蒸馏及 matched no-KD 对照。

该实现参考 CrossKD 的核心思想，但严格适配本项目固定的
``ultralytics==8.3.246`` YOLO11 Detect：

* 正常学生 Detect 输出只接受原始标注的检测损失；
* 捕获学生 ``cv2/cv3`` 第一层后的三尺度中间特征；
* 将该特征送入**冻结教师 Detect 对应分支的后半段**，得到训练期 cross-head
  预测；
* 只让 cross-head 预测在 QualityFG 位置模仿教师原始预测。

因此，它不会把整张异宽 Neck 特征图直接拟合到教师空间，也不会把教师、冻结教师
头后段、hook 或 criterion 写入最终 ``best.pt``。``control`` 与 ``train`` 共用
学生起点、优化器、训练参数和保存/验收流程；唯一训练目标差异为 cross-head KD。

部署前同步本文件及其依赖到：
``/workspace/OTOv2_v2/test/pretrain_yolo11_coco/``。
"""

from __future__ import annotations

import argparse
import copy
import io
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
import distill_yolo11_preddfl as preddfl


CROSS_HEAD_DISTILLATION_ID = "Y11-CrossHeadAdapter-QualityFG-PredDFL-KD-v2"
CROSS_HEAD_CONTROL_ID = "Y11-CrossHeadAdapter-noKD-continuation-control-v2"
CROSS_HEAD_IMPLEMENTATION = "legacy_8_3_246_yolo11_detect_cross_head_proxy_adapter_v2"
CROSS_HEAD_ADAPTER_ATTRIBUTE = "_distill_cross_head_adapters"


def _sequential_children(module: nn.Module, *, label: str) -> list[nn.Module]:
    """仅接受当前 YOLO11 Detect 的可分解 Sequential 分支。"""

    children = list(module.children())
    if not children:
        raise RuntimeError(f"{label} 不是可分解的 Detect Sequential 分支。")
    return children


def _conv_input_channels(module: nn.Module, *, label: str) -> int:
    """读取复合 Detect 子块的实际首个 Conv2d 输入宽度。

    YOLO11 ``cv3`` 的前两层是 ``Sequential(DWConv, Conv)``，不能假设外层
    模块自身带有 ``in_channels``。遍历顺序与模块 forward 顺序一致，因此首个
    Conv2d 的输入即该子块的输入宽度。
    """

    convolutions = [child for child in module.modules() if isinstance(child, nn.Conv2d)]
    if not convolutions:
        raise RuntimeError(f"{label} 无法读取卷积输入通道：未找到内部 Conv2d，module={type(module).__name__}。")
    return int(convolutions[0].in_channels)


def _conv_output_channels(module: nn.Module, *, label: str) -> int:
    """读取复合 Detect 子块的实际末个 Conv2d 输出宽度。"""

    convolutions = [child for child in module.modules() if isinstance(child, nn.Conv2d)]
    if not convolutions:
        raise RuntimeError(f"{label} 无法读取卷积输出通道：未找到内部 Conv2d，module={type(module).__name__}。")
    return int(convolutions[-1].out_channels)


def _adapter_key(branch_name: str, level_index: int) -> str:
    """为每条 Detect 分支的每个尺度生成稳定、可序列化的适配器键。"""

    return f"{branch_name}_p{int(level_index) + 3}"


class CrossHeadAdapters(nn.Module):
    """训练期跨头通道适配器；同宽时显式使用 Identity，异宽时使用 1×1 Conv。

    原始 CrossKD 的典型设置保持学生/教师 head 的中间宽度一致。P30 物理学生
    的 Detect 中间宽度已被结构化裁剪，故必须通过这个显式模块把学生特征映射到
    冻结教师尾部的输入空间。它只挂载在训练模型上，并由
    ``neck.physical_student_copy`` 在保存前移除。
    """

    def __init__(self, contract: dict[str, Any]) -> None:
        super().__init__()
        self.adapters = nn.ModuleDict()
        self.records: list[dict[str, Any]] = []
        for branch in contract["branches"]:
            branch_name = str(branch["name"])
            for level in branch["levels"]:
                level_index = int(level["level_index"])
                key = _adapter_key(branch_name, level_index)
                input_channels = int(level["student_intermediate_channels"])
                output_channels = int(level["teacher_tail_input_channels"])
                requires_adapter = input_channels != output_channels
                if requires_adapter:
                    # 重叠通道以恒等映射开始，新增教师通道以零开始；避免随机投影在
                    # 第一个 KD step 制造无意义的大扰动，同时仍保留完整可学习映射。
                    module: nn.Module = nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False)
                    with torch.no_grad():
                        module.weight.zero_()
                        diagonal = min(input_channels, output_channels)
                        indices = torch.arange(diagonal)
                        module.weight[indices, indices, 0, 0] = 1.0
                    module_type = "Conv2d_1x1_overlap_identity_zero_fill_v1"
                else:
                    module = nn.Identity()
                    module_type = "Identity"
                self.adapters[key] = module
                self.records.append(
                    {
                        "key": key,
                        "branch": branch_name,
                        "level": str(level["level"]),
                        "level_index": level_index,
                        "input_channels": input_channels,
                        "output_channels": output_channels,
                        "requires_trainable_adapter": requires_adapter,
                        "module_type": module_type,
                    }
                )

    def forward(self, branch_name: str, level_index: int, feature: torch.Tensor) -> torch.Tensor:
        key = _adapter_key(branch_name, level_index)
        if key not in self.adapters:
            raise RuntimeError(f"cross-head 适配器缺少 {key}。")
        return self.adapters[key](feature)

    def report(self) -> dict[str, Any]:
        parameters = list(self.parameters())
        return {
            "status": "pass",
            "attribute": CROSS_HEAD_ADAPTER_ATTRIBUTE,
            "policy": "explicit_training_only_1x1_channel_adapter_for_mismatch_identity_for_equal_width",
            "records": self.records,
            "trainable_adapter_count": sum(int(record["requires_trainable_adapter"]) for record in self.records),
            "adapter_parameter_count": len(parameters),
            "adapter_parameter_numel": sum(parameter.numel() for parameter in parameters),
        }


def make_cross_head_adapters(contract: dict[str, Any]) -> CrossHeadAdapters:
    """由已冻结的 shape 合同创建适配器，禁止按运行期 shape 临时猜测。"""

    adapters = CrossHeadAdapters(contract)
    expected_count = int(contract.get("adapter", {}).get("trainable_adapter_count", -1))
    actual_count = sum(int(record["requires_trainable_adapter"]) for record in adapters.records)
    if expected_count != actual_count:
        raise RuntimeError(
            f"cross-head 适配器合同不一致：contract={expected_count}，actual={actual_count}。"
        )
    return adapters


class HeadIntermediateCapture:
    """捕获学生 Detect 两条分支在固定切点后的特征，保持梯度图不变。"""

    def __init__(self, model: nn.Module, contract: dict[str, Any], label: str):
        raw = unwrap_model(model)
        detect_index, detect = neck.find_detect_module(raw)
        expected_index = int(contract["detect"]["student_index"])
        if detect_index != expected_index:
            raise RuntimeError(
                f"{label} Detect 序号与冻结合同不一致：实际={detect_index}，期望={expected_index}。"
            )
        self.label = label
        self.outputs: dict[tuple[str, int], torch.Tensor] = {}
        self.handles: list[Any] = []
        start_layer = int(contract["cross_head_start_layer"])
        for branch in contract["branches"]:
            branch_name = str(branch["name"])
            attribute = str(branch["attribute"])
            branches = getattr(detect, attribute, None)
            if not isinstance(branches, nn.ModuleList):
                raise RuntimeError(f"{label} Detect.{attribute} 不是 ModuleList。")
            for record in branch["levels"]:
                level_index = int(record["level_index"])
                modules = _sequential_children(branches[level_index], label=f"{label}.{attribute}[{level_index}]")
                capture_index = start_layer - 1
                if capture_index >= len(modules):
                    raise RuntimeError(f"{label}.{attribute}[{level_index}] 不包含切点层 {start_layer}。")
                self.handles.append(
                    modules[capture_index].register_forward_hook(self._make_hook(branch_name, level_index))
                )

    def _make_hook(self, branch_name: str, level_index: int):
        def hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                if len(output) != 1 or not isinstance(output[0], torch.Tensor):
                    raise RuntimeError(f"{self.label} {branch_name}[{level_index}] 输出不是单一 Tensor。")
                output = output[0]
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                raise RuntimeError(
                    f"{self.label} {branch_name}[{level_index}] 输出必须为 NCHW Tensor，"
                    f"实际={type(output).__name__}/{getattr(output, 'shape', None)}。"
                )
            self.outputs[(branch_name, level_index)] = output

        return hook

    def get(self, branch_name: str, level_index: int) -> torch.Tensor:
        key = (branch_name, level_index)
        if key not in self.outputs:
            raise RuntimeError(f"{self.label} 未捕获 Detect {branch_name}[{level_index}] 中间特征。")
        return self.outputs[key]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.outputs.clear()


def _forward_teacher_tail(
    teacher: nn.Module,
    capture: HeadIntermediateCapture,
    contract: dict[str, Any],
    adapters: CrossHeadAdapters,
) -> list[torch.Tensor]:
    """经显式适配后，把学生头中间特征送入冻结教师尾部，形成代理 raw logits。"""

    raw_teacher = unwrap_model(teacher)
    detect_index, detect = neck.find_detect_module(raw_teacher)
    expected_index = int(contract["detect"]["teacher_index"])
    if detect_index != expected_index:
        raise RuntimeError(
            f"教师 Detect 序号与冻结合同不一致：实际={detect_index}，期望={expected_index}。"
        )
    start_layer = int(contract["cross_head_start_layer"])
    branch_map = {str(record["name"]): record for record in contract["branches"]}
    regression = branch_map.get("regression")
    classification = branch_map.get("classification")
    if regression is None or classification is None:
        raise RuntimeError("cross-head 合同缺少 regression/classification 两条分支。")

    outputs: list[torch.Tensor] = []
    for level_index in range(int(contract["detect"]["num_levels"])):
        branch_outputs: list[torch.Tensor] = []
        for branch in (regression, classification):
            attribute = str(branch["attribute"])
            modules = _sequential_children(
                getattr(detect, attribute)[level_index],
                label=f"teacher.{attribute}[{level_index}]",
            )
            feature = capture.get(str(branch["name"]), level_index)
            expected_channels = int(branch["levels"][level_index]["student_intermediate_channels"])
            if feature.shape[1] != expected_channels:
                raise RuntimeError(
                    f"cross-head {branch['name']} P{level_index + 3} 中间通道异常："
                    f"实际={feature.shape[1]}，期望={expected_channels}。"
                )
            feature = adapters(str(branch["name"]), level_index, feature)
            teacher_tail_input_channels = int(branch["levels"][level_index]["teacher_tail_input_channels"])
            if feature.shape[1] != teacher_tail_input_channels:
                raise RuntimeError(
                    f"cross-head {branch['name']} P{level_index + 3} 适配器输出通道异常："
                    f"实际={feature.shape[1]}，期望={teacher_tail_input_channels}。"
                )
            for module in modules[start_layer:]:
                feature = module(feature)
            branch_outputs.append(feature)
        raw = torch.cat(branch_outputs, dim=1)
        outputs.append(raw)
    return outputs


def build_cross_head_contract(
    student_template: nn.Module,
    teacher_template: nn.Module,
    *,
    imgsz: int,
    start_layer: int,
) -> dict[str, Any]:
    """冻结 YOLO11 Detect 的可用 cross-head 切点与三尺度语义合同。"""

    if start_layer < 1:
        raise ValueError("--cross-head-start-layer 必须 >= 1。")
    student = copy.deepcopy(unwrap_model(student_template)).cpu().float().eval()
    teacher = copy.deepcopy(unwrap_model(teacher_template)).cpu().float().eval()
    prediction_contract = preddfl.build_prediction_contract(
        student, teacher, imgsz=imgsz, experiment_id=CROSS_HEAD_DISTILLATION_ID
    )
    student_index, student_detect = neck.find_detect_module(student)
    teacher_index, teacher_detect = neck.find_detect_module(teacher)
    if int(student_detect.nl) != 3 or int(teacher_detect.nl) != 3:
        raise RuntimeError(
            f"当前 cross-head 仅接受三尺度 Detect，student={student_detect.nl}，teacher={teacher_detect.nl}。"
        )

    branch_specs = (
        ("regression", "cv2", 4 * int(teacher_detect.reg_max)),
        ("classification", "cv3", int(teacher_detect.nc)),
    )
    branch_records: list[dict[str, Any]] = []
    for branch_name, attribute, expected_output_channels in branch_specs:
        student_branches = getattr(student_detect, attribute, None)
        teacher_branches = getattr(teacher_detect, attribute, None)
        if not isinstance(student_branches, nn.ModuleList) or not isinstance(teacher_branches, nn.ModuleList):
            raise RuntimeError(f"Detect.{attribute} 必须在师生两端均为 ModuleList。")
        if len(student_branches) != 3 or len(teacher_branches) != 3:
            raise RuntimeError(
                f"Detect.{attribute} 必须有三条尺度分支，student={len(student_branches)}，teacher={len(teacher_branches)}。"
            )
        levels: list[dict[str, Any]] = []
        for level_index in range(3):
            student_modules = _sequential_children(
                student_branches[level_index], label=f"student.{attribute}[{level_index}]"
            )
            teacher_modules = _sequential_children(
                teacher_branches[level_index], label=f"teacher.{attribute}[{level_index}]"
            )
            if start_layer >= len(student_modules) or start_layer >= len(teacher_modules):
                raise RuntimeError(
                    f"{branch_name} P{level_index + 3} 不支持切点 {start_layer}："
                    f"student_layers={len(student_modules)}，teacher_layers={len(teacher_modules)}。"
                )
            student_intermediate_channels = _conv_output_channels(
                student_modules[start_layer - 1], label=f"student.{attribute}[{level_index}][{start_layer - 1}]"
            )
            teacher_tail_input_channels = _conv_input_channels(
                teacher_modules[start_layer], label=f"teacher.{attribute}[{level_index}][{start_layer}]"
            )
            teacher_tail_output_channels = _conv_output_channels(
                teacher_modules[-1], label=f"teacher.{attribute}[{level_index}][-1]"
            )
            if teacher_tail_output_channels != expected_output_channels:
                raise RuntimeError(
                    f"{branch_name} P{level_index + 3} 教师尾输出语义异常："
                    f"实际={teacher_tail_output_channels}，期望={expected_output_channels}。"
                )
            levels.append(
                {
                    "level": f"P{level_index + 3}",
                    "level_index": level_index,
                    "student_intermediate_channels": student_intermediate_channels,
                    "teacher_tail_input_channels": teacher_tail_input_channels,
                    "teacher_tail_output_channels": teacher_tail_output_channels,
                    "student_branch_layer_count": len(student_modules),
                    "teacher_branch_layer_count": len(teacher_modules),
                    "teacher_tail_layers": len(teacher_modules) - start_layer,
                    "adapter_key": _adapter_key(branch_name, level_index),
                    "requires_trainable_adapter": student_intermediate_channels != teacher_tail_input_channels,
                    "adapter_type": (
                        "Conv2d_1x1_overlap_identity_zero_fill_v1"
                        if student_intermediate_channels != teacher_tail_input_channels
                        else "Identity"
                    ),
                }
            )
        branch_records.append({"name": branch_name, "attribute": attribute, "levels": levels})

    contract = {
        "status": "pass",
        "distillation_id": CROSS_HEAD_DISTILLATION_ID,
        "cross_head_start_layer": int(start_layer),
        "cross_head_definition": (
            "student Detect(cv2/cv3) prefix output -> explicit training-only channel adapter -> "
            "frozen teacher matching Detect tail -> teacher raw logits"
        ),
        "detect": {
            "student_index": student_index,
            "teacher_index": teacher_index,
            "num_levels": 3,
            "nc": int(teacher_detect.nc),
            "reg_max": int(teacher_detect.reg_max),
            "no": int(prediction_contract["detect"]["no"]),
        },
        "branches": branch_records,
        "adapter": {
            "attribute": CROSS_HEAD_ADAPTER_ATTRIBUTE,
            "policy": "explicit_training_only_1x1_channel_adapter_for_mismatch_identity_for_equal_width",
            "trainable_adapter_count": sum(
                int(level["requires_trainable_adapter"])
                for branch in branch_records
                for level in branch["levels"]
            ),
            "checkpoint_policy": "must_be_removed_from_deployment_checkpoint",
        },
        "prediction_contract": prediction_contract,
        "input_shape": [1, 3, imgsz, imgsz],
    }

    adapters = make_cross_head_adapters(contract).cpu().float().eval()
    capture = HeadIntermediateCapture(student, contract, "cross_head_contract_student")
    try:
        image = torch.zeros(1, 3, imgsz, imgsz)
        with torch.no_grad():
            _ = student(image)
            teacher_output = teacher(image)
            cross_raw = _forward_teacher_tail(teacher, capture, contract, adapters)
            teacher_raw = [
                tensor.detach() for tensor in preddfl.raw_detect_layers(teacher_output, label="cross_head_contract_teacher")
            ]
        preddfl.validate_runtime_raw_contract(cross_raw, teacher_raw, prediction_contract)
    finally:
        capture.close()
    return contract


def _selected_prediction_losses(
    cross_raw: list[torch.Tensor],
    teacher_raw: list[torch.Tensor],
    batch: dict[str, Any],
    contract: dict[str, Any],
    *,
    cls_temperature: float,
    dfl_temperature: float,
    confidence_threshold: float,
    topk_per_image: int,
    selection_mode: str,
    quality_topk_per_gt: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float], list[torch.Tensor], list[torch.Tensor]]:
    """复用已有 PredDFL 的位置选择语义，只替换被蒸馏的学生预测为 cross-head logits。"""

    preddfl.validate_runtime_raw_contract(cross_raw, teacher_raw, contract)
    reg_max = int(contract["detect"]["reg_max"])
    nc = int(contract["detect"]["nc"])
    if selection_mode == "global_score_v1":
        masks, mask_stats = preddfl.build_teacher_quality_masks(
            teacher_raw,
            reg_max=reg_max,
            nc=nc,
            confidence_threshold=confidence_threshold,
            topk_per_image=topk_per_image,
        )
        position_weights = None
    elif selection_mode == "quality_fg_v2":
        masks, position_weights, mask_stats = preddfl.build_qualityfg_teacher_masks(
            teacher_raw,
            batch,
            reg_max=reg_max,
            nc=nc,
            strides=list(contract["detect"]["stride"]),
            topk_per_gt=quality_topk_per_gt,
        )
    else:
        raise ValueError(f"未知 selection_mode：{selection_mode}")

    reference = cross_raw[0]
    cls_numerator = reference.new_zeros(())
    cls_denominator = reference.new_zeros(())
    dfl_numerator = reference.new_zeros(())
    dfl_denominator = reference.new_zeros(())
    level_cls_losses: list[torch.Tensor] = []
    level_dfl_losses: list[torch.Tensor] = []
    for level_index, (cross_tensor, teacher_tensor, mask) in enumerate(zip(cross_raw, teacher_raw, masks)):
        selected = mask.squeeze(1).flatten(1).reshape(-1)
        valid_positions = selected.sum()
        if selection_mode == "global_score_v1" and valid_positions.item() <= 0:
            raise RuntimeError("global-score cross-head KD 掩码为空，Top-K 合同被破坏。")
        cross_dfl = cross_tensor[:, : 4 * reg_max].reshape(
            cross_tensor.shape[0], 4, reg_max, *cross_tensor.shape[-2:]
        )
        teacher_dfl = teacher_tensor[:, : 4 * reg_max].reshape(
            teacher_tensor.shape[0], 4, reg_max, *teacher_tensor.shape[-2:]
        )
        cross_cls = cross_tensor[:, 4 * reg_max : 4 * reg_max + nc]
        teacher_cls = teacher_tensor[:, 4 * reg_max : 4 * reg_max + nc]
        cross_cls_selected = cross_cls.permute(0, 2, 3, 1).reshape(-1, nc)[selected]
        teacher_cls_selected = teacher_cls.permute(0, 2, 3, 1).reshape(-1, nc)[selected]
        cls_per_element = F.binary_cross_entropy_with_logits(
            cross_cls_selected / cls_temperature,
            (teacher_cls_selected / cls_temperature).sigmoid(),
            reduction="none",
        )
        cross_dfl_selected = cross_dfl.permute(0, 3, 4, 1, 2).reshape(-1, 4, reg_max)[selected]
        teacher_dfl_selected = teacher_dfl.permute(0, 3, 4, 1, 2).reshape(-1, 4, reg_max)[selected]
        dfl_per_bin = F.kl_div(
            F.log_softmax(cross_dfl_selected / dfl_temperature, dim=2),
            F.softmax(teacher_dfl_selected / dfl_temperature, dim=2),
            reduction="none",
        )
        if position_weights is None:
            cls_sum = cls_per_element.sum()
            cls_norm = valid_positions.to(dtype=cross_tensor.dtype) * nc
            dfl_sum = dfl_per_bin.sum(dim=2).sum()
            dfl_norm = valid_positions.to(dtype=cross_tensor.dtype) * 4
        else:
            selected_weights = position_weights[level_index].squeeze(1).flatten(1).reshape(-1)[selected]
            selected_weights = selected_weights.to(dtype=cross_tensor.dtype)
            cls_sum = (cls_per_element.sum(dim=1) * selected_weights).sum()
            cls_norm = selected_weights.sum() * nc
            dfl_per_position = dfl_per_bin.sum(dim=2).sum(dim=1)
            dfl_sum = (dfl_per_position * selected_weights).sum()
            dfl_norm = selected_weights.sum() * 4
        cls_numerator = cls_numerator + cls_sum
        cls_denominator = cls_denominator + cls_norm
        dfl_numerator = dfl_numerator + dfl_sum
        dfl_denominator = dfl_denominator + dfl_norm
        level_cls_losses.append(cls_sum / cls_norm.clamp_min(1.0))
        level_dfl_losses.append(dfl_sum / dfl_norm.clamp_min(1.0))
    cls_loss = cls_temperature**2 * cls_numerator / cls_denominator.clamp_min(1.0)
    dfl_loss = dfl_temperature**2 * dfl_numerator / dfl_denominator.clamp_min(1.0)
    return cls_loss, dfl_loss, mask_stats, level_cls_losses, level_dfl_losses


class CrossHeadCriterion:
    """标注监督学生原预测，教师监督 cross-head 预测，避免两者直接竞争同一末端输出。"""

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        prediction_contract: dict[str, Any],
        cross_head_contract: dict[str, Any],
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
    ) -> None:
        self.student = unwrap_model(student)
        self.teacher = teacher.eval()
        self.prediction_contract = prediction_contract
        self.cross_head_contract = cross_head_contract
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
        self.base_criterion = self.student.init_criterion()
        adapters = getattr(self.student, CROSS_HEAD_ADAPTER_ATTRIBUTE, None)
        if not isinstance(adapters, CrossHeadAdapters):
            raise RuntimeError("训练模型未在 optimizer 创建前注册 CrossHead 训练期通道适配器。")
        self.adapters = adapters
        self.capture = HeadIntermediateCapture(self.student, cross_head_contract, "cross_head_train_student")
        self.first_kd_gradient: dict[str, Any] = {"status": "pending" if kd_enabled else "not_applicable"}
        self.last_metrics: dict[str, float] = {}

    def close(self) -> None:
        self.capture.close()

    def warmup_factor(self) -> float:
        if not self.kd_enabled:
            return 0.0
        if self.warmup_epochs <= 0:
            return 1.0
        return min(1.0, float(int(self.trainer.epoch) + 1) / float(self.warmup_epochs))

    def _capture_first_kd_gradient(self, kd_term: torch.Tensor) -> None:
        if not self.admission_probe or self.first_kd_gradient.get("status") != "pending":
            return
        if float(kd_term.detach().abs().cpu().item()) == 0.0:
            return
        trainable = [(name, parameter) for name, parameter in self.student.named_parameters() if parameter.requires_grad]
        adapter_prefix = f"{CROSS_HEAD_ADAPTER_ATTRIBUTE}."
        physical_trainable = [(name, parameter) for name, parameter in trainable if not name.startswith(adapter_prefix)]
        adapter_trainable = [(name, parameter) for name, parameter in trainable if name.startswith(adapter_prefix)]
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
        nonzero_adapter_names = [name for name in nonzero_names if name.startswith(adapter_prefix)]
        nonzero_physical_names = [name for name in nonzero_names if not name.startswith(adapter_prefix)]
        adapter_gradient_required = bool(adapter_trainable)
        self.first_kd_gradient = {
            "status": (
                "pass"
                if nonzero_physical_names
                and (not adapter_gradient_required or nonzero_adapter_names)
                and not non_finite_names
                else "fail"
            ),
            "trainable_student_parameter_count": len(trainable),
            "trainable_physical_student_parameter_count": len(physical_trainable),
            "physical_student_parameters_with_nonzero_cross_head_kd_gradient": len(nonzero_physical_names),
            "trainable_cross_head_adapter_parameter_count": len(adapter_trainable),
            "cross_head_adapter_parameters_with_nonzero_gradient": len(nonzero_adapter_names),
            "cross_head_adapter_gradient_required": adapter_gradient_required,
            "student_parameters_with_nonzero_cross_head_kd_gradient": len(nonzero_names),
            "nonzero_cross_head_kd_gradient_parameter_names": nonzero_names,
            "nonzero_cross_head_kd_gradient_physical_parameter_names": nonzero_physical_names,
            "nonzero_cross_head_kd_gradient_adapter_parameter_names": nonzero_adapter_names,
            "non_finite_cross_head_kd_gradient_parameter_names": non_finite_names,
        }

    def __call__(self, preds, batch):
        task_loss, task_items = self.base_criterion(preds, batch)
        task_loss_total = task_loss.sum()
        student_raw = preddfl.raw_detect_layers(preds, label="cross_head_student_train")
        preddfl.validate_runtime_raw_contract(student_raw, None, self.prediction_contract)
        if not self.kd_enabled:
            metrics = {
                "task_loss": float(task_loss_total.detach().float().cpu().item()),
                "cls_kd_loss": 0.0,
                "dfl_kd_loss": 0.0,
                "kd_term": 0.0,
                "cross_head_kd_enabled": 0.0,
                "total_loss": float(task_loss_total.detach().float().cpu().item()),
                "warmup_factor": 0.0,
            }
            self.last_metrics = metrics
            self.trainer.kd_meter.update(metrics)
            return task_loss_total, task_items

        with torch.no_grad():
            teacher_output = self.teacher(batch["img"])
            teacher_raw = [
                tensor.detach()
                for tensor in preddfl.raw_detect_layers(teacher_output, label="cross_head_teacher_train")
            ]
        cross_raw = _forward_teacher_tail(self.teacher, self.capture, self.cross_head_contract, self.adapters)
        cls_kd_loss, dfl_kd_loss, mask_stats, level_cls, level_dfl = _selected_prediction_losses(
            cross_raw,
            teacher_raw,
            batch,
            self.prediction_contract,
            cls_temperature=self.cls_temperature,
            dfl_temperature=self.dfl_temperature,
            confidence_threshold=self.confidence_threshold,
            topk_per_image=self.topk_per_image,
            selection_mode=self.selection_mode,
            quality_topk_per_gt=self.quality_topk_per_gt,
        )
        warmup = self.warmup_factor()
        kd_core = self.cls_weight * cls_kd_loss + self.dfl_weight * dfl_kd_loss
        kd_term = warmup * kd_core * batch["img"].shape[0]
        self._capture_first_kd_gradient(kd_term)
        total_loss = task_loss_total + kd_term
        metrics = preddfl.scalar_tensors_to_floats(
            {
                "task_loss": task_loss_total,
                "cls_kd_loss": cls_kd_loss,
                "dfl_kd_loss": dfl_kd_loss,
                "kd_term": kd_term,
                "cross_head_kd_term": kd_term,
                "total_loss": total_loss,
                "cross_head_kd_enabled": 1.0,
                "warmup_factor": warmup,
                "effective_cls_weight": warmup * self.cls_weight,
                "effective_dfl_weight": warmup * self.dfl_weight,
                "p3_cls_raw_loss": level_cls[0],
                "p4_cls_raw_loss": level_cls[1],
                "p5_cls_raw_loss": level_cls[2],
                "p3_dfl_raw_loss": level_dfl[0],
                "p4_dfl_raw_loss": level_dfl[1],
                "p5_dfl_raw_loss": level_dfl[2],
            }
        )
        metrics.update(mask_stats)
        self.last_metrics = metrics
        self.trainer.kd_meter.update(metrics)
        return total_loss, task_items


class CrossHeadEpochReporter(preddfl.PredDFLEpochReporter):
    """沿用每 epoch 汇总与文件记录，不输出 batch 级蒸馏日志。"""

    def _print_record(self, record: dict[str, Any]) -> None:
        super()._print_record(record)
        kd = record.get("distillation", {})
        if kd.get("cross_head_kd_enabled", 0.0) >= 0.5:
            print(
                "CrossHeadKD: "
                f"term={kd.get('cross_head_kd_term', float('nan')):.6f}, "
                f"w=({kd.get('effective_cls_weight', float('nan')):.4f},"
                f"{kd.get('effective_dfl_weight', float('nan')):.4f}), "
                f"mask={kd.get('mask_selected_ratio', float('nan')):.4%}",
                flush=True,
            )


def create_cross_head_trainer_class(
    student_template: nn.Module,
    teacher_template: nn.Module,
    student_weights: Path,
    teacher_weights: Path,
    prediction_contract: dict[str, Any],
    cross_head_contract: dict[str, Any],
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
    """返回只持久化物理学生的单卡 CrossHead/Control 训练器。"""

    expected_physical = neck.physical_student_copy(student_template)
    expected_structure = neck.core.model_structure_signature(expected_physical)
    expected_state = neck.core.model_state_signature(expected_physical)
    student_sha256 = neck.core.sha256_file(student_weights)
    teacher_sha256 = neck.core.sha256_file(teacher_weights)
    phase = "cross_head_qualityfg_distillation" if kd_enabled else "cross_head_no_kd_control"
    distillation_id = CROSS_HEAD_DISTILLATION_ID if kd_enabled else CROSS_HEAD_CONTROL_ID

    class CrossHeadTrainer(DetectionTrainer):
        def __init__(self, *trainer_args, **trainer_kwargs):
            self.epoch_reporter = CrossHeadEpochReporter(phase)
            self.kd_meter = neck.KDMeter()
            self.structure_check: dict[str, Any] = {}
            self.preflight: dict[str, Any] = {}
            self.distill_criterion: CrossHeadCriterion | None = None
            super().__init__(*trainer_args, **trainer_kwargs)

        def get_model(self, cfg=None, weights=None, verbose=True):
            del cfg, weights
            model = neck.physical_student_copy(student_template).float()
            # 先报告真实部署学生规模，避免把训练期 adapter 的参数误展示为部署参数。
            if verbose and hasattr(model, "info"):
                model.info(verbose=True)
            # 必须在 super()._setup_train() 之前注册，确保 adapter 与物理学生均进入
            # 同一个 optimizer；control 也保留该参数组以保证 paired contract 一致。
            setattr(model, CROSS_HEAD_ADAPTER_ATTRIBUTE, make_cross_head_adapters(cross_head_contract))
            return model

        def _setup_train(self):
            super()._setup_train()
            if int(getattr(self, "world_size", 1)) != 1:
                raise RuntimeError("CrossHead-KD 当前仅允许单进程单卡训练，禁止 DDP 静默运行。")
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
            self.distill_criterion = CrossHeadCriterion(
                self.model,
                teacher,
                prediction_contract,
                cross_head_contract,
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
            raw_student = unwrap_model(self.model)
            raw_student.criterion = self.distill_criterion
            ownership = preddfl.optimizer_ownership_report(self.optimizer, raw_student, teacher)
            adapter_ownership = neck.optimizer_ownership_report(
                self.optimizer,
                getattr(raw_student, CROSS_HEAD_ADAPTER_ATTRIBUTE),
                teacher,
            )
            frozen = neck.frozen_parameter_report(teacher)
            passed = (
                ownership["status"] == "pass"
                and adapter_ownership["status"] == "pass"
                and frozen["requires_grad_count"] == 0
            )
            self.preflight = {
                "status": "pass" if passed else "fail",
                "time_utc": neck.utc_now(),
                "implementation": CROSS_HEAD_IMPLEMENTATION,
                "distillation_id": distillation_id,
                "mode": "train" if kd_enabled else "control",
                "teacher": {"weights": str(teacher_weights), "sha256": teacher_sha256, "frozen": frozen},
                "student": {"weights": str(student_weights), "sha256": student_sha256, "physical_structure": expected_structure},
                "prediction_contract": prediction_contract,
                "cross_head_contract": cross_head_contract,
                "optimizer_ownership": ownership,
                "cross_head_adapter_optimizer_ownership": adapter_ownership,
                "loss_contract": (
                    "L_total=L_det + warmup*(lambda_cls*L_cross_clsKD + lambda_dfl*L_cross_dflKD)*batch_size"
                    if kd_enabled
                    else "L_total=L_det (matched no-KD continuation control)"
                ),
                "checkpoint_contract": "checkpoint_only_contains_physical_student_no_teacher_no_cross_head_adapter_no_criterion",
            }
            report_path = Path(self.save_dir) / "admission" / "distillation_preflight.json"
            neck.core.json_dump(report_path, self.preflight)
            if self.preflight["status"] != "pass":
                raise RuntimeError(f"CrossHead/Control 训练所有权或教师冻结检查失败：{report_path}")
            self.epoch_reporter.attach(self)

        def optimizer_step(self):
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
                trainable = [(name, parameter) for name, parameter in raw_student.named_parameters() if parameter.requires_grad]
                adapter_prefix = f"{CROSS_HEAD_ADAPTER_ATTRIBUTE}."
                physical_trainable = [
                    (name, parameter) for name, parameter in trainable if not name.startswith(adapter_prefix)
                ]
                adapter_parameters = list(getattr(raw_student, CROSS_HEAD_ADAPTER_ATTRIBUTE).parameters())
                nonzero_student = sum(
                    int(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0)
                    for _name, parameter in physical_trainable
                )
                nonzero_adapter = sum(
                    int(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0)
                    for parameter in adapter_parameters
                )
                non_finite = [
                    name
                    for name, parameter in trainable
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
                    and (not kd_enabled or not adapter_parameters or nonzero_adapter > 0)
                    and not non_finite
                    and teacher_frozen["requires_grad_count"] == 0
                    and teacher_frozen["gradient_present_count"] == 0
                    and kd_gradient_passed
                )
                self.preflight["backward"] = {
                    "status": "pass" if passed else "fail",
                    "latest_loss": latest,
                    "trainable_physical_student_parameter_count": len(physical_trainable),
                    "physical_student_parameters_with_nonzero_gradient": nonzero_student,
                    "cross_head_adapter_parameter_count": len(adapter_parameters),
                    "cross_head_adapter_parameters_with_nonzero_gradient": nonzero_adapter,
                    "non_finite_gradient_parameter_names": non_finite,
                    "teacher_after_backward": teacher_frozen,
                    "cross_head_kd_gradient": kd_gradient,
                    "optimizer_step_completed": False,
                }
                neck.core.json_dump(report_path, self.preflight)
                if not passed:
                    raise RuntimeError(f"CrossHead/Control 首次反向传播准入失败：{report_path}")
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

    return CrossHeadTrainer


def _comparison_contract(args: argparse.Namespace, student_weights: Path, teacher_weights: Path) -> dict[str, Any]:
    return {
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
        "selection_mode": str(args.selection_mode),
        "quality_topk_per_gt": int(args.quality_topk_per_gt),
        "cross_head_start_layer": int(args.cross_head_start_layer),
        "cls_weight": float(args.cls_weight),
        "dfl_weight": float(args.dfl_weight),
        "cls_temperature": float(args.cls_temperature),
        "dfl_temperature": float(args.dfl_temperature),
        "kd_warmup_epochs": int(args.kd_warmup_epochs),
    }


def run_train(args: argparse.Namespace) -> None:
    teacher_weights = neck.require_file(args.teacher, "教师权重")
    student_weights = neck.require_file(args.student, "学生权重")
    data_yaml = neck.require_file(args.data, "数据 YAML")
    fixed_input = neck.require_file(args.fixed_input, "固定输入图像")
    output_root = Path(args.output).expanduser().resolve()
    requested_run_dir = output_root / args.name
    kd_enabled = args.mode == "train"
    if args.no_overwrite and requested_run_dir.exists():
        raise FileExistsError(f"禁止覆盖已有输出目录：{requested_run_dir}；请更换 --name。")
    if args.epochs < 1:
        raise ValueError("--epochs 必须 >= 1。")
    if args.cross_head_start_layer < 1:
        raise ValueError("--cross-head-start-layer 必须 >= 1。")
    if args.cls_temperature <= 0 or args.dfl_temperature <= 0:
        raise ValueError("分类与 DFL temperature 必须 > 0。")
    if args.cls_weight < 0 or args.dfl_weight < 0:
        raise ValueError("分类与 DFL 权重必须 >= 0。")
    if kd_enabled and args.cls_weight == 0 and args.dfl_weight == 0:
        raise ValueError("CrossHead train 至少一个 KD 权重必须 > 0。")
    if args.kd_warmup_epochs < 0:
        raise ValueError("--kd-warmup-epochs 必须 >= 0。")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold 必须位于 [0, 1]。")
    if args.selection_mode == "global_score_v1" and args.topk_per_image < 1:
        raise ValueError("--topk-per-image 必须 >= 1。")
    if args.selection_mode == "quality_fg_v2" and args.quality_topk_per_gt < 1:
        raise ValueError("--quality-topk-per-gt 必须 >= 1。")

    neck.core.set_seed(args.seed)
    device = neck.resolve_device(args.device)
    started_at = time.time()
    run_label = CROSS_HEAD_DISTILLATION_ID if kd_enabled else CROSS_HEAD_CONTROL_ID
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
    prediction_contract = preddfl.build_prediction_contract(
        student_template, teacher_template, imgsz=args.imgsz, experiment_id=run_label
    )
    cross_head_contract = build_cross_head_contract(
        student_template,
        teacher_template,
        imgsz=args.imgsz,
        start_layer=args.cross_head_start_layer,
    )
    trainer_class = create_cross_head_trainer_class(
        student_template,
        teacher_template,
        student_weights,
        teacher_weights,
        prediction_contract,
        cross_head_contract,
        kd_enabled=kd_enabled,
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
        best_weights, args, project=run_dir / "independent_validation", name="best"
    )
    fixed_forward = neck.core.fixed_input_forward_report(
        best_weights,
        fixed_input,
        imgsz=args.imgsz,
        device=device,
        output_dir=run_dir / "fixed_input",
        report_name="cross_head_distillation_best_fixed_input",
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
    summary = neck.core.save_stage_summary(
        run_dir / "distillation_summary.json",
        phase="蒸馏" if kd_enabled else "无KD对照",
        started_at=started_at,
        source_weights=student_weights,
        final_weights=best_weights,
        before_stats=before_stats,
        after_stats=after_stats,
        final_validation=final_validation,
        epoch_reporter=trainer.epoch_reporter,
        baseline_validation=baseline_validation,
        architecture_check=architecture_check,
        validation_scope=(
            "独立 YOLO(best.pt) COCO val；训练期 teacher、冻结 teacher tail、hook、"
            "cross-head channel adapter 与 criterion 均从 checkpoint 剥离。"
        ),
    )
    config = {
        "time_utc": neck.utc_now(),
        "distillation_id": run_label,
        "implementation": CROSS_HEAD_IMPLEMENTATION,
        "mode": "train" if kd_enabled else "control",
        "command_arguments": vars(args),
        "comparison_contract": _comparison_contract(args, student_weights, teacher_weights),
        "teacher": {"path": str(teacher_weights), "sha256": neck.core.sha256_file(teacher_weights)},
        "student": {
            "path": str(student_weights),
            "sha256": neck.core.sha256_file(student_weights),
            "structure": before_structure,
            "stats": before_stats,
        },
        "prediction_contract": prediction_contract,
        "cross_head_contract": cross_head_contract,
        "branches": {
            "cross_head_prediction_kd_enabled": kd_enabled,
            "cross_head_channel_adapter_enabled": True,
            "neck_sl2_enabled": False,
        },
        "loss": {
            "kd_enabled": kd_enabled,
            "cross_head_prediction_kd_enabled": kd_enabled,
            "selection_mode": args.selection_mode,
            "quality_topk_per_gt": args.quality_topk_per_gt,
            "confidence_threshold": args.confidence_threshold,
            "topk_per_image": args.topk_per_image,
            "cls_weight": args.cls_weight if kd_enabled else 0.0,
            "dfl_weight": args.dfl_weight if kd_enabled else 0.0,
            "cls_temperature": args.cls_temperature,
            "dfl_temperature": args.dfl_temperature,
            "kd_warmup_epochs": args.kd_warmup_epochs if kd_enabled else 0,
            "cross_head_start_layer": args.cross_head_start_layer,
            "training_only_components": (
                "explicit_cross_head_channel_adapters + frozen_teacher_detect_tail + forward_hooks; "
                "none_is_persisted"
            ),
            "mask_policy": (
                preddfl.QUALITYFG_MASK_CONTRACT
                if args.selection_mode == "quality_fg_v2"
                else "per_image_global_threshold_or_topk_over_p3_p4_p5_v1"
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
    parser.add_argument("--student", required=True, help="物理学生 checkpoint；KD 与 control 必须为同一文件。")
    parser.add_argument("--data", required=True, help="COCO 数据集 YAML")
    parser.add_argument("--fixed-input", required=True, help="固定输入图像，用于最终前向验收")
    parser.add_argument("--output", default="runs", help="Ultralytics 输出根目录")
    parser.add_argument("--name", required=True, help="唯一 run 名称")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default="0")
    parser.add_argument("--lr0", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--close-mosaic", type=int, default=20)
    parser.add_argument("--cls-weight", type=float, default=0.5)
    parser.add_argument("--dfl-weight", type=float, default=0.25)
    parser.add_argument("--cls-temperature", type=float, default=2.0)
    parser.add_argument("--dfl-temperature", type=float, default=2.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.10)
    parser.add_argument("--topk-per-image", type=int, default=100)
    parser.add_argument(
        "--selection-mode",
        choices=["global_score_v1", "quality_fg_v2"],
        default="quality_fg_v2",
        help="默认 quality_fg_v2：每个 GT 跨尺度质量 Top-K；可显式选择 global_score_v1。",
    )
    parser.add_argument("--quality-topk-per-gt", type=int, default=15)
    parser.add_argument("--kd-warmup-epochs", type=int, default=5)
    parser.add_argument(
        "--cross-head-start-layer",
        type=int,
        default=1,
        help="学生 Detect 分支在第几层后交给教师尾部；当前 YOLO11 适配固定为 1。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--student-baseline-map50", type=float, default=0.529458)
    parser.add_argument("--student-baseline-map50-95", type=float, default=0.371384)
    parser.add_argument("--student-baseline-source", default="P30_S0_finetuned_physical_student")
    parser.add_argument("--admission-probe", action="store_true", help="首个真实 optimizer step 验证梯度与纯 checkpoint 合同")
    parser.add_argument("--plots", action="store_true", help="保留 Ultralytics plots；默认关闭")
    parser.add_argument("--no-overwrite", action="store_true", help="目录存在即失败，保护历史产物")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    control = subparsers.add_parser("control", help="同入口 matched no-KD continuation control")
    train = subparsers.add_parser("train", help="执行检测头 cross-head QualityFG PredDFL KD")
    add_common_arguments(control)
    add_common_arguments(train)
    return parser


def main() -> None:
    run_train(build_parser().parse_args())


if __name__ == "__main__":
    main()
