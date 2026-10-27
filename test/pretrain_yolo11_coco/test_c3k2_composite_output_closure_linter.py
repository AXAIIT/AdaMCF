#!/usr/bin/env python3
"""回归 C3k2 复合合同与 output-consumer closure 的联合结构验证。

该测试直接重放一次已完成 HESSO 稀疏训练并已构造出的 YOLO11 物理子网，
不重新训练、不重新选择冗余组。其目标是防止 C3k2 复合 linter 沿用旧的
``cv2 输出必须不变`` 假设：只要 cv2 输出轴具备已配置、已验证的
output-consumer closure，输出缩小是合法的；没有该闭包合同的 C3k2 仍必须
保持输出宽度不变。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from prune_yolo11_adamcf import (
    c3k2_composite_structure_summary,
    output_consumer_closure_structure_summary,
)


def _load_module(path: Path) -> torch.nn.Module:
    """兼容新旧 PyTorch 的原始 OTO 物理子网加载。"""

    try:
        model = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # 兼容旧 PyTorch。
        model = torch.load(path, map_location="cpu")
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"压缩产物不是 torch.nn.Module：{type(model).__name__}")
    return model.float().eval()


def _tensor_shapes(value: Any) -> list[list[int]]:
    """仅记录主路径输出张量形状，避免把整个特征列表写入报告。"""

    if isinstance(value, torch.Tensor):
        return [list(value.shape)]
    if isinstance(value, (list, tuple)):
        shapes: list[list[int]] = []
        for item in value:
            shapes.extend(_tensor_shapes(item))
        return shapes
    if isinstance(value, dict):
        shapes = []
        for item in value.values():
            shapes.extend(_tensor_shapes(item))
        return shapes
    return []


def run(compressed: Path, runtime: Path, imgsz: int, output: Path) -> None:
    if not compressed.is_file():
        raise FileNotFoundError(f"找不到 OTO 压缩物理子网：{compressed}")
    if not runtime.is_file():
        raise FileNotFoundError(f"找不到稀疏训练运行时清单：{runtime}")

    runtime_manifest = json.loads(runtime.read_text(encoding="utf-8"))
    dependency = runtime_manifest.get("module_dependency_control", {})
    composite_contracts = dependency.get("yolo11_c3k2_composite_contracts", [])
    closure_contracts = dependency.get("yolo11_output_consumer_closure_contracts", [])
    if not composite_contracts or not closure_contracts:
        raise AssertionError(
            "运行时清单不含 C3k2 复合合同或 output-consumer closure 合同，"
            "不能用于该回归。"
        )

    model = _load_module(compressed)
    with torch.no_grad():
        forward_output = model(torch.zeros(1, 3, imgsz, imgsz))
    forward_shapes = _tensor_shapes(forward_output)
    if not forward_shapes:
        raise AssertionError("物理子网前向未产生任何 Tensor 输出，回归无效。")

    closure_summary = output_consumer_closure_structure_summary(
        model,
        configured_contracts=closure_contracts,
    )
    composite_summary = c3k2_composite_structure_summary(
        model,
        configured_contracts=composite_contracts,
        output_consumer_closure_structure=closure_summary,
    )
    if closure_summary.get("status") != "pass":
        raise AssertionError(
            "output-consumer closure 物理结构未闭合："
            f"{json.dumps(closure_summary, ensure_ascii=False, sort_keys=True)}"
        )
    if composite_summary.get("status") != "pass":
        raise AssertionError(
            "C3k2 paired-chunk + dynamic-concat 复合结构仍不成立："
            f"{json.dumps(composite_summary, ensure_ascii=False, sort_keys=True)}"
        )

    closure_by_module = {
        str(record.get("module_prefix")): record
        for record in closure_summary.get("records", [])
        if isinstance(record, dict)
    }
    reduced_composite_records = [
        record
        for record in composite_summary.get("records", [])
        if isinstance(record, dict) and not bool(record.get("external_output_fixed"))
    ]
    if not reduced_composite_records:
        raise AssertionError("回归产物没有命中 C3k2 的合法外部输出收缩，测试无效。")
    for record in reduced_composite_records:
        module_prefix = str(record.get("module_prefix"))
        closure_record = closure_by_module.get(module_prefix)
        if (
            record.get("external_output_policy") != "output_consumer_closure"
            or not record.get("checks", {}).get("external_output_contract_valid")
            or not isinstance(closure_record, dict)
            or closure_record.get("status") != "pass"
            or not closure_record.get("physical_output_reduced")
        ):
            raise AssertionError(
                "C3k2 输出收缩没有被闭包合同逐项证明："
                f"module={module_prefix}, composite={record}, closure={closure_record}"
            )

    report = {
        "status": "pass",
        "compressed_model": str(compressed.resolve()),
        "runtime_manifest": str(runtime.resolve()),
        "forward_shapes": forward_shapes,
        "closure_summary": closure_summary,
        "composite_summary": composite_summary,
        "reduced_composite_modules": [
            str(record.get("module_prefix")) for record in reduced_composite_records
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[pass] C3k2 复合 + output-consumer closure linter 回归通过：{output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compressed", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/c3k2_composite_output_closure_linter/report.json"),
    )
    args = parser.parse_args()
    run(args.compressed, args.runtime, args.imgsz, args.output)


if __name__ == "__main__":
    main()
