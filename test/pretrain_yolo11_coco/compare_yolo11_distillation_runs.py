#!/usr/bin/env python
"""比较一组 matched no-KD control 与 KD run，拒绝混入不一致的实验。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少 JSON：{path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON 根节点必须为对象：{path}")
    return value


def metric(summary: dict[str, Any], name: str) -> float:
    value = summary.get("final_validation", {}).get(name)
    if not isinstance(value, (float, int)):
        raise RuntimeError(f"最终验证缺少 {name}：{value}")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", required=True)
    parser.add_argument("--kd-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true", help="仅在明确指定时覆盖已有比较报告")
    args = parser.parse_args()

    control_dir = Path(args.control_dir).expanduser().resolve()
    kd_dir = Path(args.kd_dir).expanduser().resolve()
    control_config = read_json(control_dir / "distillation_config.json")
    kd_config = read_json(kd_dir / "distillation_config.json")
    control_summary = read_json(control_dir / "distillation_summary.json")
    kd_summary = read_json(kd_dir / "distillation_summary.json")

    issues: list[str] = []
    if control_config.get("mode") != "control":
        issues.append(f"control mode 错误：{control_config.get('mode')}")
    if kd_config.get("mode") != "train":
        issues.append(f"KD mode 错误：{kd_config.get('mode')}")
    control_contract = control_config.get("comparison_contract")
    kd_contract = kd_config.get("comparison_contract")
    if control_contract != kd_contract:
        issues.append("comparison_contract 不一致；禁止把不同起点或不同训练口径的 run 相比。")
    if control_config.get("student", {}).get("sha256") != kd_config.get("student", {}).get("sha256"):
        issues.append("student source SHA256 不一致。")
    if control_config.get("teacher", {}).get("sha256") != kd_config.get("teacher", {}).get("sha256"):
        issues.append("teacher SHA256 不一致。")
    if control_summary.get("architecture_check", {}).get("status") != "pass":
        issues.append("control 物理学生结构验收未通过。")
    if kd_summary.get("architecture_check", {}).get("status") != "pass":
        issues.append("KD 物理学生结构验收未通过。")

    control_map50 = metric(control_summary, "map50")
    control_map5095 = metric(control_summary, "map50_95")
    kd_map50 = metric(kd_summary, "map50")
    kd_map5095 = metric(kd_summary, "map50_95")
    report = {
        "status": "paired_comparison_pass" if not issues else "paired_comparison_fail",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "control_dir": str(control_dir),
        "kd_dir": str(kd_dir),
        "issues": issues,
        "comparison_contract": control_contract,
        "metrics": {
            "control": {"map50": control_map50, "map50_95": control_map5095},
            "kd": {"map50": kd_map50, "map50_95": kd_map5095},
            "kd_minus_control": {
                "map50": kd_map50 - control_map50,
                "map50_95": kd_map5095 - control_map5095,
            },
        },
        "quality_gate": {
            "primary_metric": "map50_95",
            "kd_noninferior_to_control": kd_map5095 >= control_map5095,
            "kd_improves_primary_metric": kd_map5095 > control_map5095,
        },
    }
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"比较报告已存在，禁止覆盖：{output}；请更换路径或显式传 --overwrite。")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if issues:
        raise RuntimeError(f"配对蒸馏比较失败：{output}")


if __name__ == "__main__":
    main()
