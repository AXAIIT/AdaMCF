#!/usr/bin/env python
"""PredDFL-KD 位置选择与概率分布张量合同的轻量单元测试。

不读取 COCO 或 checkpoint；完整模型的 raw 输出、教师冻结和真实 KD 梯度仍由
``distill_yolo11_preddfl.py --admission-probe`` 的 1 epoch 准入验证。
"""

from __future__ import annotations

import torch

from distill_yolo11_preddfl import build_teacher_quality_masks


def make_raw(*, batch: int, height: int, width: int, nc: int = 2, reg_max: int = 4) -> torch.Tensor:
    return torch.zeros(batch, 4 * reg_max + nc, height, width)


def main() -> None:
    # P3/P4/P5 共 4+1+1=6 个位置；threshold=0.8 只选一个位置，Top-K=2 必须逐图补足到至少两个。
    raw = [make_raw(batch=2, height=2, width=2), make_raw(batch=2, height=1, width=1), make_raw(batch=2, height=1, width=1)]
    raw[0][0, 16, 0, 0] = 4.0  # image 0 的显式 threshold positive
    raw[1][1, 17, 0, 0] = 3.0  # image 1 的显式 threshold positive
    masks, stats = build_teacher_quality_masks(
        raw,
        reg_max=4,
        nc=2,
        confidence_threshold=0.8,
        topk_per_image=2,
    )
    assert [tuple(mask.shape) for mask in masks] == [(2, 1, 2, 2), (2, 1, 1, 1), (2, 1, 1, 1)]
    selected_per_image = torch.cat([mask.flatten(1) for mask in masks], dim=1).sum(dim=1)
    assert torch.equal(selected_per_image, torch.tensor([2, 2]))
    assert stats["mask_empty_images"] == 0.0
    assert stats["mask_topk_per_image"] == 2.0
    assert 0.0 < stats["mask_selected_ratio"] <= 1.0

    # DFL 每一条边必须沿 reg_max 轴形成概率分布；本测试防止实现把 channel 轴误当作 DFL 轴。
    student = torch.randn(2, 4, 4, 3, 5, requires_grad=True)
    teacher = torch.randn_like(student)
    kd = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student / 2.0, dim=2),
        torch.nn.functional.softmax(teacher / 2.0, dim=2),
        reduction="none",
    ).sum(dim=2).mean() * 4.0
    kd.backward()
    assert student.grad is not None and torch.count_nonzero(student.grad).item() > 0
    print("PASS: PredDFL mask/top-k and DFL-axis contracts", flush=True)


if __name__ == "__main__":
    main()
