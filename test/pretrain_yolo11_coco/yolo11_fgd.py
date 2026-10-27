"""YOLO11 检测蒸馏的官方 FGD FeatureLoss 适配组件。

本文件按作者公开仓库 ``yzd-v/FGD`` 的 ``FeatureLoss`` 组织方式实现：

* 可选的 student-to-teacher 1×1 通道对齐；
* 面积归一化的 GT 前景掩码与按图归一化的背景掩码；
* 师生各自的空间/通道 softmax 注意力及 ``L_mask``；
* 基于 ``conv_mask``、全局上下文池化和 ``channel_add_conv`` 的官方关系项。

它只管理训练期辅助模块与张量损失。调用方必须把 ``OfficialFGDLevel``
挂在 student 上、交给 optimizer，并在保存部署 checkpoint 前剥离。教师检测网络本身
始终冻结；这里只允许 FGD 的 teacher-side 辅助分支参与训练，这与官方 FeatureLoss 一致。
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn


FGD_FULL_CONTRACT = "official_fgd_featureloss_gt_area_mask_attention_mask_gc_relation_v1"
FGD_COMPONENT_ABLATION_CONTRACT = "official_fgd_featureloss_component_switches_v1"
FGD_COMPONENT_NAMES = ("foreground", "background", "mask", "relation")
DEFAULT_TEMPERATURE = 0.5
DEFAULT_ALPHA_FGD = 0.001
DEFAULT_BETA_FGD = 0.0005
DEFAULT_GAMMA_FGD = 0.0005
DEFAULT_LAMBDA_FGD = 0.000005
OFFICIAL_FGD_AUXILIARY_ARCHITECTURE = (
    "per_level_optional_Conv1x1(C_student,C_teacher)+conv_mask_s/t+"
    "GC_context_channel_add_s/t"
)


def fgd_component_flags(
    *,
    enable_foreground: bool = True,
    enable_background: bool = True,
    enable_mask: bool = True,
    enable_relation: bool = True,
) -> dict[str, bool]:
    """规范化 FGD 四项子损失的显式开关。"""

    return {
        "foreground": bool(enable_foreground),
        "background": bool(enable_background),
        "mask": bool(enable_mask),
        "relation": bool(enable_relation),
    }


def active_fgd_component_names(component_flags: dict[str, bool]) -> tuple[str, ...]:
    """返回按官方损失顺序启用的子项名称，用于合同和实验记录。"""

    unknown = set(component_flags) - set(FGD_COMPONENT_NAMES)
    if unknown:
        raise ValueError(f"FGD 子损失配置含未知项：{sorted(unknown)}。")
    return tuple(name for name in FGD_COMPONENT_NAMES if bool(component_flags.get(name, False)))


def _as_spatial_shapes(values: Iterable[tuple[int, int] | list[int]]) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for value in values:
        if len(value) != 2:
            raise ValueError(f"FGD feature spatial shape 必须为 (H,W)，实际为 {value}。")
        height, width = int(value[0]), int(value[1])
        if height < 1 or width < 1:
            raise ValueError(f"FGD feature spatial shape 必须为正数，实际为 {(height, width)}。")
        shapes.append((height, width))
    if len(shapes) != 3:
        raise ValueError(f"FGD 必须收到 P3/P4/P5 三尺度 shape，实际为 {len(shapes)}。")
    return shapes


def _normalized_xyxy(batch: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """读取 Ultralytics 归一化 ``xywh`` 标注并转换成 ``xyxy``。"""

    if "bboxes" not in batch or "batch_idx" not in batch:
        raise KeyError("FGD batch 缺少 bboxes 或 batch_idx。")
    boxes = batch["bboxes"].to(device=device, dtype=dtype)
    indices = batch["batch_idx"].to(device=device).reshape(-1).long()
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise RuntimeError(f"FGD bboxes 必须为 [M,4] 归一化 xywh，实际={tuple(boxes.shape)}。")
    if indices.numel() != boxes.shape[0]:
        raise RuntimeError(f"FGD batch_idx/bboxes 数量不一致：{indices.numel()} vs {boxes.shape[0]}。")
    if boxes.numel() == 0:
        return indices, boxes.new_zeros((0, 4))
    center = boxes[:, :2]
    half = boxes[:, 2:] / 2
    return indices, torch.cat((center - half, center + half), dim=1).clamp_(0.0, 1.0)


def build_fgd_foreground_background_masks(
    batch: dict[str, Any],
    feature_shapes: Iterable[tuple[int, int] | list[int]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, torch.Tensor]]:
    """构造作者 FeatureLoss 同语义的 P3/P4/P5 前景、背景 mask。

    对每个 GT，先将其映射到 feature grid 的 ``floor/ceil`` 闭区间，并将框内每格
    赋值为该框离散面积的倒数；重叠区域取最大值。这样每个未重叠 GT 的前景质量为 1。
    背景为前景以外区域，并按每张图的背景格数归一化为质量 1。
    """

    if batch_size < 1:
        raise ValueError(f"FGD batch_size 必须 >= 1，实际为 {batch_size}。")
    shapes = _as_spatial_shapes(feature_shapes)
    indices, boxes = _normalized_xyxy(batch, device=device, dtype=dtype)
    if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= batch_size):
        raise RuntimeError(f"FGD batch_idx 超出 batch 范围：min={int(indices.min())}, max={int(indices.max())}, B={batch_size}。")
    if not hasattr(torch.Tensor, "scatter_reduce_"):
        raise RuntimeError("FGD mask 构造需要 torch.Tensor.scatter_reduce_（服务器 torch>=1.12）。")

    foreground_masks: list[torch.Tensor] = []
    background_masks: list[torch.Tensor] = []
    foreground_mass: list[torch.Tensor] = []
    background_mass: list[torch.Tensor] = []
    foreground_cells: list[torch.Tensor] = []

    for height, width in shapes:
        foreground_flat = torch.zeros((batch_size, height * width), device=device, dtype=dtype)
        if boxes.shape[0] > 0:
            x_min = torch.floor(boxes[:, 0] * float(width)).long().clamp_(0, width - 1)
            x_max = torch.ceil(boxes[:, 2] * float(width)).long().clamp_(0, width - 1)
            y_min = torch.floor(boxes[:, 1] * float(height)).long().clamp_(0, height - 1)
            y_max = torch.ceil(boxes[:, 3] * float(height)).long().clamp_(0, height - 1)
            # clamp 后仍显式保证闭区间非空，匹配官方 floor/ceil + inclusive slice 语义。
            x_max = torch.maximum(x_max, x_min)
            y_max = torch.maximum(y_max, y_min)
            area = ((x_max - x_min + 1) * (y_max - y_min + 1)).to(dtype=dtype).reciprocal()
            grid_y = torch.arange(height, device=device).view(1, height, 1)
            grid_x = torch.arange(width, device=device).view(1, 1, width)
            inside = (
                (grid_x >= x_min.view(-1, 1, 1))
                & (grid_x <= x_max.view(-1, 1, 1))
                & (grid_y >= y_min.view(-1, 1, 1))
                & (grid_y <= y_max.view(-1, 1, 1))
            )
            weighted_boxes = inside.to(dtype=dtype) * area.view(-1, 1, 1)
            foreground_flat.scatter_reduce_(
                0,
                indices[:, None].expand(-1, height * width),
                weighted_boxes.reshape(boxes.shape[0], -1),
                reduce="amax",
                include_self=True,
            )

        foreground = foreground_flat.reshape(batch_size, height, width)
        background = (foreground == 0).to(dtype=dtype)
        background_sum = background.sum(dim=(1, 2), keepdim=True)
        background = torch.where(
            background_sum > 0,
            background / background_sum.clamp_min(torch.finfo(dtype).eps),
            background,
        )
        foreground_masks.append(foreground)
        background_masks.append(background)
        foreground_mass.append(foreground.sum())
        background_mass.append(background.sum())
        foreground_cells.append((foreground > 0).sum().to(dtype=dtype))

    total_cells = torch.as_tensor(
        float(sum(batch_size * height * width for height, width in shapes)), device=device, dtype=dtype
    )
    statistics = {
        "fgd_gt_total": torch.as_tensor(float(boxes.shape[0]), device=device, dtype=dtype),
        "fgd_foreground_mass_p3": foreground_mass[0],
        "fgd_foreground_mass_p4": foreground_mass[1],
        "fgd_foreground_mass_p5": foreground_mass[2],
        "fgd_background_mass_p3": background_mass[0],
        "fgd_background_mass_p4": background_mass[1],
        "fgd_background_mass_p5": background_mass[2],
        "fgd_foreground_cell_ratio": torch.stack(foreground_cells).sum() / total_cells.clamp_min(1.0),
    }
    return foreground_masks, background_masks, statistics


class OfficialFGDLevel(nn.Module):
    """一条尺度特征的官方 FGD ``FeatureLoss`` 训练期辅助模块。"""

    def __init__(self, student_channels: int, teacher_channels: int, *, temperature: float = DEFAULT_TEMPERATURE) -> None:
        super().__init__()
        if student_channels < 1 or teacher_channels < 1:
            raise ValueError("FGD student_channels 与 teacher_channels 必须为正数。")
        if temperature <= 0 or not math.isfinite(float(temperature)):
            raise ValueError("FGD temperature 必须为有限正数。")
        self.temperature = float(temperature)
        self.align = (
            nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
            if student_channels != teacher_channels
            else None
        )
        self.conv_mask_s = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        hidden_channels = max(teacher_channels // 2, 1)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(teacher_channels, hidden_channels, kernel_size=1),
            nn.LayerNorm([hidden_channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, teacher_channels, kernel_size=1),
        )
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(teacher_channels, hidden_channels, kernel_size=1),
            nn.LayerNorm([hidden_channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, teacher_channels, kernel_size=1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """匹配官方实现：mask conv Kaiming，channel-add 最后一层零初始化。"""

        for module in (self.conv_mask_s, self.conv_mask_t):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        for module in (self.channel_add_conv_s, self.channel_add_conv_t):
            final = module[-1]
            assert isinstance(final, nn.Conv2d)
            nn.init.zeros_(final.weight)
            if final.bias is not None:
                nn.init.zeros_(final.bias)

    @staticmethod
    def _require_4d(feature: torch.Tensor, label: str) -> None:
        if feature.ndim != 4:
            raise RuntimeError(f"{label} 必须为 [B,C,H,W]，实际={tuple(feature.shape)}。")

    def align_student(self, student_feature: torch.Tensor) -> torch.Tensor:
        self._require_4d(student_feature, "FGD student feature")
        return self.align(student_feature) if self.align is not None else student_feature

    def get_attention(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """与官方 ``get_attention`` 同公式，返回 spatial [B,H,W]、channel [B,C]。"""

        self._require_4d(feature, "FGD attention feature")
        batch_size, channels, height, width = feature.shape
        value = feature.abs()
        feature_map = value.mean(dim=1, keepdim=True)
        spatial_attention = (
            height
            * width
            * F.softmax((feature_map / self.temperature).reshape(batch_size, -1), dim=1)
        ).reshape(batch_size, height, width)
        channel_map = value.mean(dim=(2, 3))
        channel_attention = channels * F.softmax(channel_map / self.temperature, dim=1)
        return spatial_attention, channel_attention

    def _spatial_pool(self, feature: torch.Tensor, *, student_branch: bool) -> torch.Tensor:
        batch_size, channels, height, width = feature.shape
        input_x = feature.reshape(batch_size, channels, height * width).unsqueeze(1)
        mask_conv = self.conv_mask_s if student_branch else self.conv_mask_t
        context_mask = F.softmax(mask_conv(feature).reshape(batch_size, 1, height * width), dim=2).unsqueeze(-1)
        return torch.matmul(input_x, context_mask).reshape(batch_size, channels, 1, 1)

    def _relation_loss(self, student_feature: torch.Tensor, teacher_feature: torch.Tensor) -> torch.Tensor:
        context_student = self._spatial_pool(student_feature, student_branch=True)
        context_teacher = self._spatial_pool(teacher_feature, student_branch=False)
        output_student = student_feature + self.channel_add_conv_s(context_student)
        output_teacher = teacher_feature + self.channel_add_conv_t(context_teacher)
        return F.mse_loss(output_student, output_teacher, reduction="sum") / student_feature.shape[0]

    def forward(
        self,
        student_feature: torch.Tensor,
        teacher_feature: torch.Tensor,
        foreground_mask: torch.Tensor | None,
        background_mask: torch.Tensor | None,
        *,
        compute_foreground: bool,
        compute_background: bool,
        compute_mask: bool,
        compute_relation: bool,
    ) -> dict[str, torch.Tensor]:
        """返回按显式开关计算的 FGD 原始子损失。

        默认四项全开时与官方 ``FeatureLoss`` 的既有路径完全一致。关闭某项时，
        该项不参与前向损失计算和反向传播；训练期辅助模块仍统一注册在学生模型上，
        以保持保存/所有权/旧全量 FGD 的结构兼容。
        """

        teacher_feature = teacher_feature.detach()
        projected_student = self.align_student(student_feature)
        self._require_4d(teacher_feature, "FGD teacher feature")
        if projected_student.shape != teacher_feature.shape:
            raise RuntimeError(
                "FGD 对齐后师生特征合同失败："
                f"student={tuple(projected_student.shape)}, teacher={tuple(teacher_feature.shape)}。"
            )
        batch_size, _channels, height, width = projected_student.shape
        expected_mask_shape = (batch_size, height, width)
        use_focal_masks = bool(compute_foreground or compute_background)
        if use_focal_masks:
            if foreground_mask is None or background_mask is None:
                raise RuntimeError("FGD 已启用前景或背景项，但没有收到对应 mask。")
            if foreground_mask.shape != expected_mask_shape or background_mask.shape != expected_mask_shape:
                raise RuntimeError(
                    "FGD mask 特征合同失败："
                    f"fg={tuple(foreground_mask.shape)}, bg={tuple(background_mask.shape)}, expected={expected_mask_shape}。"
                )
            if foreground_mask.dtype != projected_student.dtype or background_mask.dtype != projected_student.dtype:
                raise RuntimeError("FGD foreground/background mask dtype 必须与特征一致。")

        zero = projected_student.new_zeros(())
        need_attention = bool(compute_foreground or compute_background or compute_mask)
        if need_attention:
            spatial_teacher, channel_teacher = self.get_attention(teacher_feature)
            spatial_student, channel_student = self.get_attention(projected_student)
        else:
            spatial_teacher = zero
            channel_teacher = zero
            spatial_student = zero
            channel_student = zero

        foreground_loss = zero
        background_loss = zero
        mask_loss = zero
        relation_loss = zero
        if compute_foreground or compute_background:
            assert foreground_mask is not None and background_mask is not None
            spatial_weight = spatial_teacher.unsqueeze(1).sqrt()
            channel_weight = channel_teacher.unsqueeze(-1).unsqueeze(-1).sqrt()
            weighted_teacher = teacher_feature * spatial_weight * channel_weight
            weighted_student = projected_student * spatial_weight * channel_weight
            if compute_foreground:
                foreground_weight = foreground_mask.unsqueeze(1).sqrt()
                foreground_loss = F.mse_loss(
                    weighted_student * foreground_weight,
                    weighted_teacher * foreground_weight,
                    reduction="sum",
                ) / batch_size
            if compute_background:
                background_weight = background_mask.unsqueeze(1).sqrt()
                background_loss = F.mse_loss(
                    weighted_student * background_weight,
                    weighted_teacher * background_weight,
                    reduction="sum",
                ) / batch_size
        if compute_mask:
            mask_loss = (
                (channel_student - channel_teacher).abs().sum() / batch_size
                + (spatial_student - spatial_teacher).abs().sum() / batch_size
            )
        if compute_relation:
            relation_loss = self._relation_loss(projected_student, teacher_feature)
        return {
            "foreground_loss": foreground_loss,
            "background_loss": background_loss,
            "mask_loss": mask_loss,
            "relation_loss": relation_loss,
            "spatial_attention_teacher_mean": spatial_teacher.detach().mean(),
            "channel_attention_teacher_mean": channel_teacher.detach().mean(),
            "spatial_attention_student_mean": spatial_student.detach().mean(),
            "channel_attention_student_mean": channel_student.detach().mean(),
        }


def make_official_fgd_modules(contract: dict[str, Any], *, temperature: float) -> nn.ModuleList:
    """按运行时 P3/P4/P5 通道合同创建三条官方 FGD 辅助模块。"""

    records = contract.get("feature_levels")
    if not isinstance(records, list) or len(records) != 3:
        raise RuntimeError("FGD 创建辅助模块需要包含三条 feature_levels 的 neck contract。")
    return nn.ModuleList(
        [
            OfficialFGDLevel(
                int(record["student_channels"]),
                int(record["teacher_channels"]),
                temperature=temperature,
            )
            for record in records
        ]
    )


def full_fgd_feature_loss(
    student_features: list[torch.Tensor],
    teacher_features: list[torch.Tensor],
    modules: nn.ModuleList,
    batch: dict[str, Any],
    *,
    foreground_weight: float,
    background_weight: float,
    mask_weight: float,
    relation_weight: float,
    enable_foreground: bool = True,
    enable_background: bool = True,
    enable_mask: bool = True,
    enable_relation: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """计算三尺度 FGD 子项的加权和及 GPU 标量统计。

    四个 ``enable_*`` 默认值均为 ``True``，因此旧调用保持完整官方 FGD 语义。
    子项关闭后不参与该次前向/反向，允许在不改动物理学生结构的前提下做受控消融。
    """

    if not (len(student_features) == len(teacher_features) == len(modules) == 3):
        raise RuntimeError(
            "完整 FGD 必须收到 P3/P4/P5 三尺度特征和模块："
            f"student={len(student_features)}, teacher={len(teacher_features)}, modules={len(modules)}。"
        )
    weights = (foreground_weight, background_weight, mask_weight, relation_weight)
    if not all(math.isfinite(float(value)) and value >= 0 for value in weights):
        raise ValueError("FGD foreground/background/mask/relation 权重必须为有限非负数。")
    components = fgd_component_flags(
        enable_foreground=enable_foreground,
        enable_background=enable_background,
        enable_mask=enable_mask,
        enable_relation=enable_relation,
    )
    effective_weights = {
        "foreground": float(foreground_weight) if components["foreground"] else 0.0,
        "background": float(background_weight) if components["background"] else 0.0,
        "mask": float(mask_weight) if components["mask"] else 0.0,
        "relation": float(relation_weight) if components["relation"] else 0.0,
    }
    if sum(effective_weights.values()) <= 0:
        raise ValueError("FGD 至少一个已启用子损失的权重必须 > 0。")
    reference = student_features[0]
    if reference.ndim != 4:
        raise RuntimeError(f"FGD 特征必须为四维 Tensor，实际={tuple(reference.shape)}。")
    batch_size = int(reference.shape[0])
    if any(int(feature.shape[0]) != batch_size for feature in student_features):
        raise RuntimeError("FGD P3/P4/P5 student batch size 不一致。")
    use_focal_masks = components["foreground"] or components["background"]
    if use_focal_masks:
        feature_shapes = [tuple(feature.shape[-2:]) for feature in student_features]
        foreground_masks, background_masks, mask_stats = build_fgd_foreground_background_masks(
            batch,
            feature_shapes,
            device=reference.device,
            dtype=reference.dtype,
            batch_size=batch_size,
        )
    else:
        foreground_masks = [None, None, None]
        background_masks = [None, None, None]
        zero = reference.new_zeros(())
        mask_stats = {
            "fgd_gt_total": zero,
            "fgd_foreground_mass_p3": zero,
            "fgd_foreground_mass_p4": zero,
            "fgd_foreground_mass_p5": zero,
            "fgd_background_mass_p3": zero,
            "fgd_background_mass_p4": zero,
            "fgd_background_mass_p5": zero,
            "fgd_foreground_cell_ratio": zero,
        }

    foreground_losses: list[torch.Tensor] = []
    background_losses: list[torch.Tensor] = []
    mask_losses: list[torch.Tensor] = []
    relation_losses: list[torch.Tensor] = []
    teacher_spatial_means: list[torch.Tensor] = []
    teacher_channel_means: list[torch.Tensor] = []
    student_spatial_means: list[torch.Tensor] = []
    student_channel_means: list[torch.Tensor] = []
    for level, (module, student_feature, teacher_feature, foreground, background) in enumerate(
        zip(modules, student_features, teacher_features, foreground_masks, background_masks), start=3
    ):
        if not isinstance(module, OfficialFGDLevel):
            raise TypeError(f"FGD P{level} 辅助模块类型错误：{type(module).__name__}。")
        terms = module(
            student_feature,
            teacher_feature,
            foreground,
            background,
            compute_foreground=components["foreground"],
            compute_background=components["background"],
            compute_mask=components["mask"],
            compute_relation=components["relation"],
        )
        foreground_losses.append(terms["foreground_loss"])
        background_losses.append(terms["background_loss"])
        mask_losses.append(terms["mask_loss"])
        relation_losses.append(terms["relation_loss"])
        teacher_spatial_means.append(terms["spatial_attention_teacher_mean"])
        teacher_channel_means.append(terms["channel_attention_teacher_mean"])
        student_spatial_means.append(terms["spatial_attention_student_mean"])
        student_channel_means.append(terms["channel_attention_student_mean"])

    # 官方配置把每个 FPN level 的 FeatureLoss 直接相加；此处保持同一聚合语义。
    foreground_loss = torch.stack(foreground_losses).sum()
    background_loss = torch.stack(background_losses).sum()
    mask_loss = torch.stack(mask_losses).sum()
    relation_loss = torch.stack(relation_losses).sum()
    total_loss = (
        effective_weights["foreground"] * foreground_loss
        + effective_weights["background"] * background_loss
        + effective_weights["mask"] * mask_loss
        + effective_weights["relation"] * relation_loss
    )
    metrics = {
        **mask_stats,
        "fgd_foreground_loss": foreground_loss.detach(),
        "fgd_background_loss": background_loss.detach(),
        "fgd_mask_loss": mask_loss.detach(),
        "fgd_relation_loss": relation_loss.detach(),
        "fgd_feature_loss": total_loss.detach(),
        "fgd_teacher_spatial_attention_mean": torch.stack(teacher_spatial_means).mean(),
        "fgd_teacher_channel_attention_mean": torch.stack(teacher_channel_means).mean(),
        "fgd_student_spatial_attention_mean": torch.stack(student_spatial_means).mean(),
        "fgd_student_channel_attention_mean": torch.stack(student_channel_means).mean(),
        "fgd_level_count": reference.new_tensor(3.0),
        "fgd_foreground_enabled": reference.new_tensor(float(components["foreground"])),
        "fgd_background_enabled": reference.new_tensor(float(components["background"])),
        "fgd_mask_enabled": reference.new_tensor(float(components["mask"])),
        "fgd_relation_enabled": reference.new_tensor(float(components["relation"])),
        "fgd_foreground_effective_weight": reference.new_tensor(effective_weights["foreground"]),
        "fgd_background_effective_weight": reference.new_tensor(effective_weights["background"]),
        "fgd_mask_effective_weight": reference.new_tensor(effective_weights["mask"]),
        "fgd_relation_effective_weight": reference.new_tensor(effective_weights["relation"]),
    }
    for level, values in enumerate(
        zip(foreground_losses, background_losses, mask_losses, relation_losses), start=3
    ):
        foreground_level, background_level, mask_level, relation_level = values
        metrics[f"fgd_foreground_loss_p{level}"] = foreground_level.detach()
        metrics[f"fgd_background_loss_p{level}"] = background_level.detach()
        metrics[f"fgd_mask_loss_p{level}"] = mask_level.detach()
        metrics[f"fgd_relation_loss_p{level}"] = relation_level.detach()
    return total_loss, metrics
