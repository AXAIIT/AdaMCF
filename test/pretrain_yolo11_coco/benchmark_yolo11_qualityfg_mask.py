#!/usr/bin/env python
"""比较 QualityFG reference 与按图向量化选点的 GPU 微基准。

使用与 YOLO11 640 Detect 输出、batch=64、约 7 GT/图相同的张量规模。它只测量
QualityFG 掩码构造，不训练、不修改权重、不写 checkpoint；用于确认优化未改变掩码
结果并量化消除 GT 级 GPU/Python 同步后的收益。
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from yolo11_qualityfg_mask import _build_qualityfg_teacher_masks_reference, build_qualityfg_teacher_masks


def make_batch(batch_size: int, gt_per_image: int, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(20260728)
    count = batch_size * gt_per_image
    centers = torch.rand(count, 2, generator=generator, device=device) * 0.8 + 0.1
    sizes = torch.rand(count, 2, generator=generator, device=device) * 0.25 + 0.02
    return {
        "img": torch.empty(batch_size, 3, 640, 640, device=device),
        "batch_idx": torch.arange(batch_size, device=device).repeat_interleave(gt_per_image),
        "bboxes": torch.cat((centers, sizes), dim=1),
    }


def run_once(function, raw, batch, topk: int):
    return function(raw, batch, reg_max=16, nc=80, strides=[8.0, 16.0, 32.0], topk_per_gt=topk)


def elapsed_seconds(function, raw, batch, topk: int, iterations: int) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        run_once(function, raw, batch, topk)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--gt-per-image", type=int, default=7)
    parser.add_argument("--topk-per-gt", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("该微基准必须在可用 CUDA 设备上运行。")
    if args.batch < 1 or args.gt_per_image < 1 or args.topk_per_gt < 1 or args.iterations < 1:
        raise ValueError("batch、gt-per-image、topk-per-gt 与 iterations 均必须 >= 1。")

    torch.manual_seed(20260728)
    raw = [
        torch.randn(args.batch, 144, 80, 80, device=device),
        torch.randn(args.batch, 144, 40, 40, device=device),
        torch.randn(args.batch, 144, 20, 20, device=device),
    ]
    batch = make_batch(args.batch, args.gt_per_image, device)
    reference = run_once(_build_qualityfg_teacher_masks_reference, raw, batch, args.topk_per_gt)
    vectorized = run_once(build_qualityfg_teacher_masks, raw, batch, args.topk_per_gt)
    masks_equal = all(torch.equal(actual, expected) for actual, expected in zip(vectorized[0], reference[0]))
    weights_close = all(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6) for actual, expected in zip(vectorized[1], reference[1]))
    stats_close = all(abs(vectorized[2][key] - value) < 1e-5 for key, value in reference[2].items())
    if not (masks_equal and weights_close and stats_close):
        raise AssertionError("向量化 QualityFG 与 reference 结果不一致，禁止用于训练。")

    # 先预热，随后 reference 测一次、vectorized 测多次，避免把初始化成本记入结论。
    run_once(build_qualityfg_teacher_masks, raw, batch, args.topk_per_gt)
    reference_seconds = elapsed_seconds(_build_qualityfg_teacher_masks_reference, raw, batch, args.topk_per_gt, 1)
    vectorized_seconds = elapsed_seconds(build_qualityfg_teacher_masks, raw, batch, args.topk_per_gt, args.iterations)
    report = {
        "status": "pass",
        "device": str(device),
        "batch": args.batch,
        "gt_per_image": args.gt_per_image,
        "topk_per_gt": args.topk_per_gt,
        "reference_seconds_per_call": reference_seconds,
        "vectorized_seconds_per_call": vectorized_seconds,
        "speedup": reference_seconds / vectorized_seconds if vectorized_seconds > 0 else None,
        "masks_equal": masks_equal,
        "weights_close": weights_close,
        "statistics_close": stats_close,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
