#!/usr/bin/env python
"""Y11-QualityFG-PredDFL-KD-v2 的 GT 覆盖、质量权重与回退合同单元测试。"""

from __future__ import annotations

import torch

from yolo11_qualityfg_mask import _build_qualityfg_teacher_masks_reference, build_qualityfg_teacher_masks


def assert_vectorized_matches_reference(layers: list[torch.Tensor], batch: dict[str, torch.Tensor], *, topk_per_gt: int) -> None:
    reference = _build_qualityfg_teacher_masks_reference(
        layers,
        batch,
        reg_max=4,
        nc=2,
        strides=[8.0, 16.0, 32.0],
        topk_per_gt=topk_per_gt,
    )
    vectorized = build_qualityfg_teacher_masks(
        layers,
        batch,
        reg_max=4,
        nc=2,
        strides=[8.0, 16.0, 32.0],
        topk_per_gt=topk_per_gt,
    )
    reference_masks, reference_weights, reference_stats = reference
    vectorized_masks, vectorized_weights, vectorized_stats = vectorized
    assert all(torch.equal(actual, expected) for actual, expected in zip(vectorized_masks, reference_masks))
    assert all(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6) for actual, expected in zip(vectorized_weights, reference_weights))
    assert vectorized_stats.keys() == reference_stats.keys()
    for key, expected in reference_stats.items():
        assert abs(vectorized_stats[key] - expected) < 1e-5, (key, vectorized_stats[key], expected)


def raw_layer(batch: int, height: int, width: int, reg_max: int = 4, nc: int = 2) -> torch.Tensor:
    raw = torch.zeros(batch, 4 * reg_max + nc, height, width)
    # 保持正分数但避免 Top-K 平局，以便严格比对 reference 与向量化实现。
    raw[:, 4 * reg_max, :, :] = 1.0 + torch.arange(height * width, dtype=torch.float32).reshape(height, width) / 1000.0
    return raw


def main() -> None:
    # 64×64 输入，P3/P4/P5 网格分别为 8×8、4×4、2×2；两个 GT 都应有内部候选。
    layers = [raw_layer(1, 8, 8), raw_layer(1, 4, 4), raw_layer(1, 2, 2)]
    batch = {
        "img": torch.zeros(1, 3, 64, 64),
        "batch_idx": torch.tensor([0.0, 0.0]),
        "bboxes": torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.75, 0.75, 0.25, 0.25]]),
    }
    masks, weights, stats = build_qualityfg_teacher_masks(
        layers,
        batch,
        reg_max=4,
        nc=2,
        strides=[8.0, 16.0, 32.0],
        topk_per_gt=2,
    )
    assert [tuple(value.shape) for value in masks] == [(1, 1, 8, 8), (1, 1, 4, 4), (1, 1, 2, 2)]
    assert [tuple(value.shape) for value in weights] == [(1, 1, 8, 8), (1, 1, 4, 4), (1, 1, 2, 2)]
    assert stats["qualityfg_gt_total"] == 2.0
    assert stats["qualityfg_gt_zero_selected"] == 0.0
    assert stats["qualityfg_gt_fallback_nearest_p3"] == 0.0
    assert stats["qualityfg_gt_selected"] == 2.0
    assert stats["qualityfg_score_iou_p50"] >= 0.0
    assert all(torch.isfinite(value).all() for value in weights)
    assert sum(int(mask.sum().item()) for mask in masks) >= 2
    # 两个远离的 GT 没有重叠，质量归一化后每个 GT 的总权重应为 K=2。
    assert abs(sum(float(weight.sum().item()) for weight in weights) - 4.0) < 1e-5
    assert_vectorized_matches_reference(layers, batch, topk_per_gt=2)

    # 小于 P3 网格的 GT 没有内部 center 时，必须显式走 P3 最近点回退，不能静默丢失。
    tiny_batch = {
        "img": torch.zeros(1, 3, 64, 64),
        "batch_idx": torch.tensor([0.0]),
        "bboxes": torch.tensor([[0.015625, 0.015625, 0.01, 0.01]]),
    }
    tiny_masks, tiny_weights, tiny_stats = build_qualityfg_teacher_masks(
        layers,
        tiny_batch,
        reg_max=4,
        nc=2,
        strides=[8.0, 16.0, 32.0],
        topk_per_gt=2,
    )
    assert tiny_stats["qualityfg_gt_total"] == 1.0
    assert tiny_stats["qualityfg_gt_fallback_nearest_p3"] == 1.0
    assert tiny_stats["qualityfg_gt_zero_selected"] == 0.0
    assert sum(int(mask.sum().item()) for mask in tiny_masks) >= 1
    assert abs(sum(float(weight.sum().item()) for weight in tiny_weights) - 1.0) < 1e-5
    assert_vectorized_matches_reference(layers, tiny_batch, topk_per_gt=2)

    # 纯背景 batch 合法地没有 QualityFG 点；训练器必须等待后续非零 KD batch 做首步准入。
    background_batch = {
        "img": torch.zeros(1, 3, 64, 64),
        "batch_idx": torch.zeros(0),
        "bboxes": torch.zeros(0, 4),
    }
    background_masks, background_weights, background_stats = build_qualityfg_teacher_masks(
        layers,
        background_batch,
        reg_max=4,
        nc=2,
        strides=[8.0, 16.0, 32.0],
        topk_per_gt=2,
    )
    assert background_stats["qualityfg_gt_total"] == 0.0
    assert background_stats["mask_empty_images"] == 1.0
    assert sum(int(mask.sum().item()) for mask in background_masks) == 0
    assert sum(float(weight.sum().item()) for weight in background_weights) == 0.0
    assert_vectorized_matches_reference(layers, background_batch, topk_per_gt=2)
    print("PASS: QualityFG GT coverage, quality weighting and nearest-P3 fallback contracts", flush=True)


if __name__ == "__main__":
    main()
