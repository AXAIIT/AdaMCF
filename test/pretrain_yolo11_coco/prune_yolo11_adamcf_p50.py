#!/usr/bin/env python
"""YOLO11 AdaMCF 已验证安全模块联合 50% 组稀疏率专用入口。

本文件不复制主流程实现，而是固定 ``audit`` / ``prune`` / ``sparse-train`` /
``rebuild-subnet`` 使用 ``all_verified_safe_num_head_v1_experimental`` 累计
coverage profile；并固定剪枝比例为 0.50、物理子网文件名为
``yolo11n_adamcf_p50_all_verified_structured.pt``，随后委托主入口执行。

联合开放范围包括已经完成物理闭包验证的 dense CSP/常规 Conv/Detect 隐藏支路、
SPPF、PSA FFN、C2PSA 输出以及 YOLO11 Attention 完整 head。CSP chunk/动态
concat、C2PSA split、PSA residual、Detect 最终语义输出和 Attention proj
外部输出边界继续由主入口精确保护。
"""

from __future__ import annotations

import math
import sys

from prune_yolo11_adamcf import ALL_VERIFIED_SAFE_PROFILE, main


P50_RATIO = 0.50
P50_PROFILE = ALL_VERIFIED_SAFE_PROFILE
P50_STRUCTURED_NAME = "yolo11n_adamcf_p50_all_verified_structured.pt"
SUPPORTED_MODES = {"audit", "prune", "sparse-train", "rebuild-subnet"}


def option_value(argv: list[str], option: str) -> str | None:
    """读取 ``--key value`` 或 ``--key=value`` 形式的选项。"""

    for index, token in enumerate(argv[2:], start=2):
        if token == option:
            if index + 1 >= len(argv):
                raise SystemExit(f"{option} 缺少数值。")
            return argv[index + 1]
        if token.startswith(f"{option}="):
            return token.split("=", 1)[1]
    return None


def enforce_option(argv: list[str], option: str, expected: str, description: str) -> None:
    """注入固定选项，并拒绝与专用入口合同不一致的显式配置。"""

    current = option_value(argv, option)
    if current is None:
        argv.extend([option, expected])
    elif current != expected:
        raise SystemExit(
            f"{__file__} 是{description}专用入口，{option} 必须为 {expected}，当前为 {current}。"
        )


def enforce_p50(argv: list[str]) -> None:
    """固定联合安全 profile、50% 比例和结构化 checkpoint 名称。"""

    if len(argv) < 2 or argv[1] not in SUPPORTED_MODES:
        return

    enforce_option(
        argv,
        "--module-coverage-profile",
        P50_PROFILE,
        "已验证安全模块联合剪枝",
    )

    # 联合 profile 的候选前缀已经覆盖全部已验证安全模块；仍启用组级隔离，
    # 将未进入该安全集合的任何 OTO 可剪组全部关闭，避免未知模块混入 HESSO。
    if "--module-probe-only" not in argv:
        argv.append("--module-probe-only")
    if any(
        token == "--probe-module-family" or token.startswith("--probe-module-family=")
        for token in argv[2:]
    ):
        raise SystemExit(
            f"{__file__} 用于完整联合剪枝，禁止用 --probe-module-family 缩小候选范围。"
        )

    if argv[1] not in {"prune", "sparse-train", "rebuild-subnet"}:
        return

    ratio_value = option_value(argv, "--prune-ratio")
    ratio = float(ratio_value) if ratio_value is not None else None
    if ratio is None:
        argv.extend(["--prune-ratio", str(P50_RATIO)])
    elif not math.isclose(ratio, P50_RATIO, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(
            f"{__file__} 是 50% 专用入口，--prune-ratio 必须为 {P50_RATIO:.2f}，当前为 {ratio}。"
        )

    enforce_option(
        argv,
        "--structured-name",
        P50_STRUCTURED_NAME,
        "50% 已验证安全模块联合剪枝",
    )


if __name__ == "__main__":
    enforce_p50(sys.argv)
    main()
