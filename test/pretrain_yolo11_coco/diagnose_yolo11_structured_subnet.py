#!/usr/bin/env python3
"""定位 OTO 物理子网中的首个卷积输入通道不一致问题。

该脚本只读取已生成的结构化 checkpoint，并在一次固定 shape 前向中记录
每个模块的输入/输出 shape。它不会修改或重新保存模型权重。
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from ultralytics import YOLO


class ChannelMismatchDetected(RuntimeError):
    """由 Conv2d 前置钩子抛出，使报错包含精确模块路径。"""


def tensor_shapes(value: Any) -> Any:
    """把嵌套输出转换为可写入 JSON 的 shape。"""

    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [tensor_shapes(item) for item in value]
    if isinstance(value, dict):
        return {str(key): tensor_shapes(item) for key, item in value.items()}
    return type(value).__name__


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="诊断 YOLO11 OTO 物理子网通道不一致")
    parser.add_argument("--weights", required=True, help="已生成的 Ultralytics 结构化 .pt checkpoint")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="默认 cpu；也可指定 cuda:0")
    parser.add_argument(
        "--output",
        help="诊断 JSON；默认写到权重同目录，文件名追加 .channel_diagnostic.json",
    )
    parser.add_argument("--recent-events", type=int, default=80, help="保留故障前最近多少个模块输出事件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"结构化权重不存在：{weights}")
    if args.imgsz <= 0:
        raise ValueError("--imgsz 必须为正整数")
    if args.recent_events <= 0:
        raise ValueError("--recent-events 必须为正整数")

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else weights.with_suffix(weights.suffix + ".channel_diagnostic.json")
    )
    device = torch.device(args.device)
    model = YOLO(str(weights)).model.to(device).float().eval()
    module_names = {id(module): name for name, module in model.named_modules()}
    recent: deque[dict[str, Any]] = deque(maxlen=args.recent_events)
    mismatch: dict[str, Any] | None = None
    handles = []

    def completed_module_hook(module: nn.Module, inputs: tuple[Any, ...], result: Any) -> None:
        name = module_names.get(id(module), "<unknown>")
        if not name:
            return
        recent.append(
            {
                "module": name,
                "type": type(module).__name__,
                "input_shapes": tensor_shapes(inputs),
                "output_shapes": tensor_shapes(result),
            }
        )

    def conv_input_guard(module: nn.Conv2d, inputs: tuple[Any, ...]) -> None:
        nonlocal mismatch
        if not inputs or not isinstance(inputs[0], torch.Tensor) or inputs[0].ndim < 2:
            return
        actual = int(inputs[0].shape[1])
        expected = int(module.in_channels)
        if actual == expected:
            return
        mismatch = {
            "module": module_names.get(id(module), "<unknown>"),
            "type": type(module).__name__,
            "expected_input_channels": expected,
            "actual_input_channels": actual,
            "channel_difference": actual - expected,
            "input_shape": list(inputs[0].shape),
            "weight_shape": list(module.weight.shape),
            "out_channels": int(module.out_channels),
            "groups": int(module.groups),
        }
        raise ChannelMismatchDetected(
            f"{mismatch['module']}: expected {expected} input channels, got {actual}; "
            f"weight_shape={mismatch['weight_shape']}"
        )

    for module in model.modules():
        handles.append(module.register_forward_hook(completed_module_hook))
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_pre_hook(conv_input_guard))

    report: dict[str, Any] = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "weights": str(weights),
        "imgsz": args.imgsz,
        "device": str(device),
    }
    exit_code = 0
    try:
        with torch.no_grad():
            result = model(torch.zeros(1, 3, args.imgsz, args.imgsz, device=device))
        report.update(
            {
                "status": "forward_passed",
                "output_shapes": tensor_shapes(result),
                "recent_completed_modules": list(recent),
            }
        )
        print("[physical-subnet-diagnostic] 固定 shape 前向通过。", flush=True)
    except ChannelMismatchDetected as exc:
        report.update(
            {
                "status": "channel_mismatch",
                "error": str(exc),
                "first_mismatch": mismatch,
                "recent_completed_modules": list(recent),
            }
        )
        print("[physical-subnet-diagnostic] 检测到首个通道不一致：", flush=True)
        print(json.dumps(mismatch, ensure_ascii=False, indent=2), flush=True)
        exit_code = 2
    except Exception as exc:
        report.update(
            {
                "status": "other_forward_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "recent_completed_modules": list(recent),
            }
        )
        print(f"[physical-subnet-diagnostic] 前向发生其他错误：{type(exc).__name__}: {exc}", flush=True)
        exit_code = 3
    finally:
        for handle in handles:
            handle.remove()
        write_report(output, report)

    print(f"诊断报告已保存：{output}", flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
