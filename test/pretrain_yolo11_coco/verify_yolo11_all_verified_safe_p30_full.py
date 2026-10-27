#!/usr/bin/env python
"""全开放 YOLO11 联合 profile 的 P30 正式九段流程入口。

这是早期 P10/P30 完整流程的当前版本：复用九段编排器，而非复制 OTO/HESSO
实现。它固定当前已通过 L3 的联合 profile、P30 和 90/10/80 调度，并保留
编排器的 post-prune gate，避免结构合同未通过时继续微调。
"""

from __future__ import annotations

import math
import sys

from verify_yolo11_csp_dense_closure import main


PROFILE = "all_verified_safe_num_head_v4_output_closure_experimental"
P30_RATIO = 0.30
FIXED_OPTIONS = {
    "--module-coverage-profile": PROFILE,
    "--group-divisible": "1",
    "--prune-epochs": "90",
    "--warmup-epochs": "10",
    "--pruning-periods": "80",
    "--finetune-epochs": "80",
    "--close-mosaic": "20",
}


def option_values(argv: list[str], option: str) -> list[str]:
    values: list[str] = []
    for index, token in enumerate(argv[1:], start=1):
        if token == option:
            if index + 1 >= len(argv):
                raise SystemExit(f"{option} 缺少数值。")
            values.append(argv[index + 1])
        elif token.startswith(f"{option}="):
            values.append(token.split("=", 1)[1])
    return values


def require_exact_option(argv: list[str], option: str, expected: str) -> None:
    values = option_values(argv, option)
    if len(values) > 1:
        raise SystemExit(f"{option} 只能指定一次，当前为 {values}。")
    if not values:
        argv.extend([option, expected])
    elif values[0] != expected:
        raise SystemExit(f"全开放 P30 正式入口的 {option} 必须为 {expected}，当前为 {values[0]}。")


def configure_full_run(argv: list[str]) -> None:
    for option, expected in FIXED_OPTIONS.items():
        require_exact_option(argv, option, expected)

    ratio_values = option_values(argv, "--prune-ratio")
    if len(ratio_values) > 1:
        raise SystemExit(f"--prune-ratio 只能指定一次，当前为 {ratio_values}。")
    if not ratio_values:
        argv.extend(["--prune-ratio", str(P30_RATIO)])
    elif not math.isclose(float(ratio_values[0]), P30_RATIO, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(
            f"全开放 P30 正式入口的 --prune-ratio 必须为 {P30_RATIO:.2f}，当前为 {ratio_values[0]}。"
        )


if __name__ == "__main__":
    configure_full_run(sys.argv)
    main()
