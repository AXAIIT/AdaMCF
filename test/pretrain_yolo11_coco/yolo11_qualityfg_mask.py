"""YOLO11 预测级蒸馏的 GT×教师质量位置选择合同。

该模块只构造无梯度位置掩码和相对质量权重；不包含 Trainer、optimizer、
checkpoint 或任何 OTO/HESSO 逻辑。教师 raw Detect 输出必须保持 P3/P4/P5、
``no=4*reg_max+nc`` 的原始语义。
"""

from __future__ import annotations

from typing import Any

import torch
from ultralytics.utils.tal import dist2bbox, make_anchors


QUALITYFG_MASK_CONTRACT = "gt_inside_teacher_cls_times_decoded_dfl_iou_per_gt_cross_scale_topk_v2"
DEFAULT_QUALITY_TOPK_PER_GT = 15


def _quantiles_or_zero(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().float().flatten()
    if flat.numel() == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    quantiles = torch.quantile(flat, torch.tensor([0.5, 0.9, 0.99], device=flat.device))
    return {
        "mean": float(flat.mean().cpu().item()),
        "p50": float(quantiles[0].cpu().item()),
        "p90": float(quantiles[1].cpu().item()),
        "p99": float(quantiles[2].cpu().item()),
    }


def _xywhn_to_xyxy(boxes: torch.Tensor, image_height: int, image_width: int) -> torch.Tensor:
    scale = boxes.new_tensor([image_width, image_height, image_width, image_height])
    xywh = boxes * scale
    center, wh = xywh[..., :2], xywh[..., 2:]
    return torch.cat((center - wh / 2, center + wh / 2), dim=-1)


def _iou_to_one(boxes: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(boxes[:, :2], target[:2])
    bottom_right = torch.minimum(boxes[:, 2:], target[2:])
    intersection_wh = (bottom_right - top_left).clamp_min(0)
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
    boxes_area = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    target_area = (target[2] - target[0]).clamp_min(0) * (target[3] - target[1]).clamp_min(0)
    return intersection / (boxes_area + target_area - intersection).clamp_min(1e-9)


def _level_offsets(raw_layers: list[torch.Tensor]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    start = 0
    for raw in raw_layers:
        end = start + int(raw.shape[-2] * raw.shape[-1])
        offsets.append((start, end))
        start = end
    return offsets


def _decode_dfl_boxes(
    raw_layers: list[torch.Tensor], *, reg_max: int, strides: list[float] | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """与 v8DetectionLoss 相同的 DFL decode，返回像素 xyxy box 与 anchor center。"""

    batch_size = int(raw_layers[0].shape[0])
    distributions = torch.cat([raw.view(batch_size, raw.shape[1], -1) for raw in raw_layers], dim=2)
    pred_dist = distributions[:, : 4 * reg_max].permute(0, 2, 1).contiguous()
    stride_tensor = torch.as_tensor(strides, device=pred_dist.device, dtype=pred_dist.dtype)
    anchor_points, anchors_stride = make_anchors(raw_layers, stride_tensor, 0.5)
    projection = torch.arange(reg_max, dtype=pred_dist.dtype, device=pred_dist.device)
    distances = pred_dist.view(batch_size, -1, 4, reg_max).softmax(3).matmul(projection)
    boxes = dist2bbox(distances, anchor_points, xywh=False) * anchors_stride
    centers = anchor_points * anchors_stride
    return boxes, centers, _level_offsets(raw_layers)


def _build_qualityfg_teacher_masks_reference(
    teacher_raw: list[torch.Tensor],
    batch: dict[str, Any],
    *,
    reg_max: int,
    nc: int,
    strides: list[float] | torch.Tensor,
    topk_per_gt: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, float]]:
    """按 GT 内 anchor、教师分类×定位质量、每 GT 跨尺度 Top-K 选择 KD 位置。

    每个 GT 的候选质量 ``q=max_c(sigmoid(cls))*IoU(decoded_DFL_box, GT)`` 在各尺度
    合并后取 Top-K。所选 q 按 GT 归一化为总权重 K，因而既保留相对质量，又不会让大
    目标天然拥有更大的总损失质量。重叠 GT 的同一位置取最大相对权重。

    极小 GT 若没有任何 grid center 落在框内，才回退到距离 GT 中心最近的 P3 center；
    回退数量会显式写入统计，不能静默掩盖。
    """

    if len(teacher_raw) != 3:
        raise RuntimeError(f"QualityFG 必须收到 P3/P4/P5 三尺度 raw 输出，实际为 {len(teacher_raw)}。")
    if topk_per_gt < 1:
        raise ValueError("QualityFG --quality-topk-per-gt 必须 >= 1。")
    batch_size = int(teacher_raw[0].shape[0])
    if any(int(raw.shape[0]) != batch_size for raw in teacher_raw):
        raise RuntimeError("QualityFG 三尺度 batch size 不一致。")
    score_layers = [raw[:, 4 * reg_max : 4 * reg_max + nc].detach().sigmoid().amax(dim=1) for raw in teacher_raw]
    scores = torch.cat([score.flatten(1) for score in score_layers], dim=1)
    boxes, centers, offsets = _decode_dfl_boxes(teacher_raw, reg_max=reg_max, strides=strides)
    weight_flat = torch.zeros_like(scores)
    batch_indices = batch["batch_idx"].view(-1).long()
    image_height, image_width = int(batch["img"].shape[2]), int(batch["img"].shape[3])
    gt_boxes_all = _xywhn_to_xyxy(batch["bboxes"], image_height, image_width)
    p3_start, p3_end = offsets[0]

    gt_total = 0
    gt_with_inside_candidate = 0
    gt_with_fallback = 0
    gt_selected = 0
    per_level_candidate_gt = [0, 0, 0]
    per_level_selected_gt = [0, 0, 0]
    selected_quality_values: list[torch.Tensor] = []
    selected_iou_values: list[torch.Tensor] = []

    for image_index in range(batch_size):
        image_gt = gt_boxes_all[batch_indices == image_index]
        for gt in image_gt:
            gt_total += 1
            inside = (
                (centers[:, 0] >= gt[0])
                & (centers[:, 0] <= gt[2])
                & (centers[:, 1] >= gt[1])
                & (centers[:, 1] <= gt[3])
            )
            candidates = torch.nonzero(inside, as_tuple=False).flatten()
            if candidates.numel() == 0:
                gt_center = (gt[:2] + gt[2:]) / 2
                p3_distances = (centers[p3_start:p3_end] - gt_center).square().sum(dim=1)
                candidates = p3_distances.argmin().reshape(1) + p3_start
                gt_with_fallback += 1
            else:
                gt_with_inside_candidate += 1

            for level_index, (start, end) in enumerate(offsets):
                candidate_on_level = candidates[(candidates >= start) & (candidates < end)]
                if candidate_on_level.numel() > 0:
                    per_level_candidate_gt[level_index] += 1
            ious = _iou_to_one(boxes[image_index, candidates], gt)
            qualities = scores[image_index, candidates] * ious
            k = min(int(topk_per_gt), int(candidates.numel()))
            top_quality, top_relative = torch.topk(qualities, k=k, largest=True, sorted=False)
            chosen = candidates[top_relative]
            chosen_iou = ious[top_relative]
            quality_sum = top_quality.sum()
            if float(quality_sum.detach().cpu().item()) > 0.0:
                normalized_weight = top_quality / quality_sum * float(k)
            else:
                normalized_weight = torch.ones_like(top_quality)
            previous = weight_flat[image_index, chosen]
            weight_flat[image_index, chosen] = torch.maximum(previous, normalized_weight)
            selected_quality_values.append(top_quality.detach())
            selected_iou_values.append(chosen_iou.detach())
            gt_selected += 1
            for level_index, (start, end) in enumerate(offsets):
                if bool(((chosen >= start) & (chosen < end)).any().item()):
                    per_level_selected_gt[level_index] += 1

    masks: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    selected_per_level: list[float] = []
    weight_sum_per_level: list[float] = []
    for raw, (start, end) in zip(teacher_raw, offsets):
        height, width = raw.shape[-2:]
        level_weights = weight_flat[:, start:end].reshape(batch_size, 1, height, width)
        level_mask = level_weights > 0
        masks.append(level_mask)
        weights.append(level_weights)
        selected_per_level.append(float(level_mask.sum().detach().cpu().item()))
        weight_sum_per_level.append(float(level_weights.sum().detach().cpu().item()))

    selected_total = (weight_flat > 0).sum().detach().float()
    empty_images = ((weight_flat > 0).sum(dim=1) == 0).sum().detach().float()
    quality_tensor = torch.cat(selected_quality_values) if selected_quality_values else scores.new_zeros((0,))
    iou_tensor = torch.cat(selected_iou_values) if selected_iou_values else scores.new_zeros((0,))
    quality_stats = _quantiles_or_zero(quality_tensor)
    iou_stats = _quantiles_or_zero(iou_tensor)
    return masks, weights, {
        "mask_policy_id": 2.0,
        "mask_selected_per_image": float(selected_total.cpu().item()) / max(batch_size, 1),
        "mask_selected_ratio": float(selected_total.cpu().item()) / float(weight_flat.numel()),
        "mask_empty_images": float(empty_images.cpu().item()),
        "mask_p3_selected_per_batch": selected_per_level[0],
        "mask_p4_selected_per_batch": selected_per_level[1],
        "mask_p5_selected_per_batch": selected_per_level[2],
        "qualityfg_topk_per_gt": float(topk_per_gt),
        "qualityfg_gt_total": float(gt_total),
        "qualityfg_gt_with_inside_candidate": float(gt_with_inside_candidate),
        "qualityfg_gt_fallback_nearest_p3": float(gt_with_fallback),
        "qualityfg_gt_selected": float(gt_selected),
        "qualityfg_gt_zero_selected": float(gt_total - gt_selected),
        "qualityfg_p3_gt_with_candidate": float(per_level_candidate_gt[0]),
        "qualityfg_p4_gt_with_candidate": float(per_level_candidate_gt[1]),
        "qualityfg_p5_gt_with_candidate": float(per_level_candidate_gt[2]),
        "qualityfg_p3_gt_selected": float(per_level_selected_gt[0]),
        "qualityfg_p4_gt_selected": float(per_level_selected_gt[1]),
        "qualityfg_p5_gt_selected": float(per_level_selected_gt[2]),
        "qualityfg_weight_sum": float(weight_flat.sum().detach().cpu().item()),
        "qualityfg_p3_weight_sum": weight_sum_per_level[0],
        "qualityfg_p4_weight_sum": weight_sum_per_level[1],
        "qualityfg_p5_weight_sum": weight_sum_per_level[2],
        "qualityfg_score_iou_mean": quality_stats["mean"],
        "qualityfg_score_iou_p50": quality_stats["p50"],
        "qualityfg_score_iou_p90": quality_stats["p90"],
        "qualityfg_score_iou_p99": quality_stats["p99"],
        "qualityfg_teacher_iou_mean": iou_stats["mean"],
        "qualityfg_teacher_iou_p50": iou_stats["p50"],
        "qualityfg_teacher_iou_p90": iou_stats["p90"],
        "qualityfg_teacher_iou_p99": iou_stats["p99"],
    }


def _quantile_tensors_or_zero(values: torch.Tensor) -> dict[str, torch.Tensor]:
    """返回 GPU 标量；仅在整批统计打包时同步一次。"""

    flat = values.detach().float().flatten()
    if flat.numel() == 0:
        zero = values.new_zeros((), dtype=torch.float32)
        return {"mean": zero, "p50": zero, "p90": zero, "p99": zero}
    quantiles = torch.quantile(flat, torch.tensor([0.5, 0.9, 0.99], device=flat.device))
    return {"mean": flat.mean(), "p50": quantiles[0], "p90": quantiles[1], "p99": quantiles[2]}


def _scalar_statistics_to_floats(values: dict[str, torch.Tensor | float | int]) -> dict[str, float]:
    """将所有统计标量合并为一次 D2H 同步，避免逐指标阻塞 CUDA 流。"""

    if not values:
        return {}
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


def _pairwise_iou(boxes: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """返回 [GT, anchor] IoU；仅用于单张图内的向量化 QualityFG 选择。"""

    top_left = torch.maximum(boxes.unsqueeze(0)[..., :2], targets.unsqueeze(1)[..., :2])
    bottom_right = torch.minimum(boxes.unsqueeze(0)[..., 2:], targets.unsqueeze(1)[..., 2:])
    intersection_wh = (bottom_right - top_left).clamp_min(0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    boxes_area = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    target_area = (targets[:, 2] - targets[:, 0]).clamp_min(0) * (targets[:, 3] - targets[:, 1]).clamp_min(0)
    return intersection / (target_area.unsqueeze(1) + boxes_area.unsqueeze(0) - intersection).clamp_min(1e-9)


def build_qualityfg_teacher_masks(
    teacher_raw: list[torch.Tensor],
    batch: dict[str, Any],
    *,
    reg_max: int,
    nc: int,
    strides: list[float] | torch.Tensor,
    topk_per_gt: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, float]]:
    """按图向量化的 QualityFG 掩码，语义与 reference 实现严格相同。

    旧实现在每个 GT 上发起多个 CUDA 操作并调用 ``.item()``，会使 GPU 与 Python
    高频同步。这里将同一图的全部 GT 拼为 ``[G, A]`` 张量，只保留 batch 级的短循环；
    不改变候选定义、极小 GT 回退、Top-K、质量归一化或重叠位置 max 所有权。
    """

    if len(teacher_raw) != 3:
        raise RuntimeError(f"QualityFG 必须收到 P3/P4/P5 三尺度 raw 输出，实际为 {len(teacher_raw)}。")
    if topk_per_gt < 1:
        raise ValueError("QualityFG --quality-topk-per-gt 必须 >= 1。")
    if not hasattr(torch.Tensor, "scatter_reduce_"):
        raise RuntimeError("QualityFG 向量化路径需要 torch.Tensor.scatter_reduce_（服务器 torch>=1.12）。")
    batch_size = int(teacher_raw[0].shape[0])
    if any(int(raw.shape[0]) != batch_size for raw in teacher_raw):
        raise RuntimeError("QualityFG 三尺度 batch size 不一致。")

    score_layers = [raw[:, 4 * reg_max : 4 * reg_max + nc].detach().sigmoid().amax(dim=1) for raw in teacher_raw]
    scores = torch.cat([score.flatten(1) for score in score_layers], dim=1)
    boxes, centers, offsets = _decode_dfl_boxes(teacher_raw, reg_max=reg_max, strides=strides)
    weight_flat = torch.zeros_like(scores)
    batch_indices = batch["batch_idx"].view(-1).long()
    image_height, image_width = int(batch["img"].shape[2]), int(batch["img"].shape[3])
    gt_boxes_all = _xywhn_to_xyxy(batch["bboxes"], image_height, image_width)
    p3_start, p3_end = offsets[0]
    zero = scores.new_zeros(())
    gt_total = zero.clone()
    gt_with_inside_candidate = zero.clone()
    gt_with_fallback = zero.clone()
    per_level_candidate_gt = [zero.clone() for _ in offsets]
    per_level_selected_gt = [zero.clone() for _ in offsets]
    selected_quality_blocks: list[torch.Tensor] = []
    selected_iou_blocks: list[torch.Tensor] = []
    valid_blocks: list[torch.Tensor] = []
    max_topk = min(int(topk_per_gt), int(scores.shape[1]))

    for image_index in range(batch_size):
        image_gt = gt_boxes_all[batch_indices == image_index]
        gt_count = int(image_gt.shape[0])
        if gt_count == 0:
            continue
        gt_total = gt_total + gt_count
        gt_center = (image_gt[:, :2] + image_gt[:, 2:]) / 2
        inside = (
            (centers.unsqueeze(0)[..., 0] >= image_gt[:, None, 0])
            & (centers.unsqueeze(0)[..., 0] <= image_gt[:, None, 2])
            & (centers.unsqueeze(0)[..., 1] >= image_gt[:, None, 1])
            & (centers.unsqueeze(0)[..., 1] <= image_gt[:, None, 3])
        )
        has_inside = inside.any(dim=1)
        fallback = ~has_inside
        p3_nearest = (centers[p3_start:p3_end].unsqueeze(0) - gt_center.unsqueeze(1)).square().sum(dim=2).argmin(dim=1)
        p3_nearest = p3_nearest + p3_start
        fallback_one_hot = torch.zeros_like(inside)
        fallback_one_hot.scatter_(1, p3_nearest.unsqueeze(1), True)
        candidates = inside | (fallback.unsqueeze(1) & fallback_one_hot)
        gt_with_inside_candidate = gt_with_inside_candidate + has_inside.sum()
        gt_with_fallback = gt_with_fallback + fallback.sum()

        for level_index, (start, end) in enumerate(offsets):
            per_level_candidate_gt[level_index] = per_level_candidate_gt[level_index] + candidates[:, start:end].any(dim=1).sum()

        ious = _pairwise_iou(boxes[image_index], image_gt)
        qualities = scores[image_index].unsqueeze(0) * ious
        top_quality, chosen = qualities.masked_fill(~candidates, float("-inf")).topk(max_topk, dim=1, largest=True, sorted=False)
        valid = torch.isfinite(top_quality)
        quality_for_norm = top_quality.masked_fill(~valid, 0.0)
        valid_count = valid.sum(dim=1, keepdim=True).to(dtype=qualities.dtype)
        quality_sum = quality_for_norm.sum(dim=1, keepdim=True)
        normalized_weight = torch.where(
            quality_sum > 0,
            quality_for_norm / quality_sum.clamp_min(torch.finfo(qualities.dtype).tiny) * valid_count,
            valid.to(dtype=qualities.dtype),
        )
        normalized_weight = normalized_weight * valid.to(dtype=qualities.dtype)
        weight_flat[image_index].scatter_reduce_(
            0,
            chosen.reshape(-1),
            normalized_weight.reshape(-1),
            reduce="amax",
            include_self=True,
        )
        chosen_iou = ious.gather(1, chosen)
        selected_quality_blocks.append(quality_for_norm.detach())
        selected_iou_blocks.append(chosen_iou.detach())
        valid_blocks.append(valid)
        for level_index, (start, end) in enumerate(offsets):
            per_level_selected_gt[level_index] = per_level_selected_gt[level_index] + (
                ((chosen >= start) & (chosen < end) & valid).any(dim=1).sum()
            )

    masks: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    selected_per_level: list[torch.Tensor] = []
    weight_sum_per_level: list[torch.Tensor] = []
    for raw, (start, end) in zip(teacher_raw, offsets):
        height, width = raw.shape[-2:]
        level_weights = weight_flat[:, start:end].reshape(batch_size, 1, height, width)
        level_mask = level_weights > 0
        masks.append(level_mask)
        weights.append(level_weights)
        selected_per_level.append(level_mask.sum())
        weight_sum_per_level.append(level_weights.sum())

    if selected_quality_blocks:
        quality_values = torch.cat(selected_quality_blocks).flatten()
        iou_values = torch.cat(selected_iou_blocks).flatten()
        valid_values = torch.cat(valid_blocks).flatten()
        quality_values = quality_values[valid_values]
        iou_values = iou_values[valid_values]
    else:
        quality_values = scores.new_zeros((0,))
        iou_values = scores.new_zeros((0,))
    quality_stats = _quantile_tensors_or_zero(quality_values)
    iou_stats = _quantile_tensors_or_zero(iou_values)
    selected_total = (weight_flat > 0).sum()
    empty_images = ((weight_flat > 0).sum(dim=1) == 0).sum()
    statistics: dict[str, torch.Tensor | float | int] = {
        "mask_policy_id": 2.0,
        "mask_selected_per_image": selected_total / max(batch_size, 1),
        "mask_selected_ratio": selected_total / weight_flat.numel(),
        "mask_empty_images": empty_images,
        "mask_p3_selected_per_batch": selected_per_level[0],
        "mask_p4_selected_per_batch": selected_per_level[1],
        "mask_p5_selected_per_batch": selected_per_level[2],
        "qualityfg_topk_per_gt": float(topk_per_gt),
        "qualityfg_gt_total": gt_total,
        "qualityfg_gt_with_inside_candidate": gt_with_inside_candidate,
        "qualityfg_gt_fallback_nearest_p3": gt_with_fallback,
        "qualityfg_gt_selected": gt_total,
        "qualityfg_gt_zero_selected": 0.0,
        "qualityfg_p3_gt_with_candidate": per_level_candidate_gt[0],
        "qualityfg_p4_gt_with_candidate": per_level_candidate_gt[1],
        "qualityfg_p5_gt_with_candidate": per_level_candidate_gt[2],
        "qualityfg_p3_gt_selected": per_level_selected_gt[0],
        "qualityfg_p4_gt_selected": per_level_selected_gt[1],
        "qualityfg_p5_gt_selected": per_level_selected_gt[2],
        "qualityfg_weight_sum": weight_flat.sum(),
        "qualityfg_p3_weight_sum": weight_sum_per_level[0],
        "qualityfg_p4_weight_sum": weight_sum_per_level[1],
        "qualityfg_p5_weight_sum": weight_sum_per_level[2],
        "qualityfg_score_iou_mean": quality_stats["mean"],
        "qualityfg_score_iou_p50": quality_stats["p50"],
        "qualityfg_score_iou_p90": quality_stats["p90"],
        "qualityfg_score_iou_p99": quality_stats["p99"],
        "qualityfg_teacher_iou_mean": iou_stats["mean"],
        "qualityfg_teacher_iou_p50": iou_stats["p50"],
        "qualityfg_teacher_iou_p90": iou_stats["p90"],
        "qualityfg_teacher_iou_p99": iou_stats["p99"],
    }
    return masks, weights, _scalar_statistics_to_floats(statistics)
