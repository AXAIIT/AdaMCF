#!/usr/bin/env python3
"""回归 C3k2 composite recovery 的语义候选所有权集合。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prune_yolo11_adamcf import (
    c3k2_composite_recovery_expected_candidate_parameter_sets,
)


def _signature_union(runtime: dict, *, root: bool) -> set[str]:
    records = runtime.get("module_pruning_candidate_groups", []) or []
    selected = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if root:
            matched = record.get("c3k2_contract") == "yolo11_c3k2_paired_chunk_v1"
        else:
            matched = (
                record.get("c3k2_dynamic_contract")
                == "yolo11_c3k2_segmented_dynamic_concat_v1"
            )
        if matched:
            selected.extend(record.get("candidate_parameter_names", []) or [])
    return {str(name) for name in selected}


def run(runtime_path: Path, output: Path) -> None:
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    dependency = runtime.get("module_dependency_control", {})
    candidate_names = {
        str(name)
        for record in runtime.get("module_pruning_candidate_groups", []) or []
        if isinstance(record, dict)
        for name in record.get("candidate_parameter_names", []) or []
    }
    expected_root, expected_dynamic = (
        c3k2_composite_recovery_expected_candidate_parameter_sets(
            dependency,
            candidate_names,
        )
    )
    actual_root = _signature_union(runtime, root=True)
    actual_dynamic = _signature_union(runtime, root=False)
    if not expected_root or not expected_dynamic:
        raise AssertionError(
            "运行时清单未解析出 C3k2 root 或 DynamicConcat child 候选，测试无效。"
        )
    if expected_root != actual_root or expected_dynamic != actual_dynamic:
        raise AssertionError(
            "C3k2 recovery 语义候选集合与实际所有权签名不一致："
            f"expected_root={sorted(expected_root)}, actual_root={sorted(actual_root)}, "
            f"expected_dynamic={sorted(expected_dynamic)}, actual_dynamic={sorted(actual_dynamic)}"
        )

    report = {
        "status": "pass",
        "runtime": str(runtime_path.resolve()),
        "root_candidate_parameter_names": sorted(expected_root),
        "dynamic_candidate_parameter_names": sorted(expected_dynamic),
        "root_count": len(expected_root),
        "dynamic_count": len(expected_dynamic),
        "selection_policy": (
            "registered_yolo11_c3k2_paired_chunk_and_dynamic_concat_contracts_intersect_real_candidates_v1"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[pass] C3k2 composite recovery 所有权回归通过：{output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/c3k2_composite_recovery_ownership_v1/report.json"),
    )
    args = parser.parse_args()
    if not args.runtime.is_file():
        parser.error(f"--runtime 不存在：{args.runtime}")
    run(args.runtime, args.output)


if __name__ == "__main__":
    main()
