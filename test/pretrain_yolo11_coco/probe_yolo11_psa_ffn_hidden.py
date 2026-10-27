#!/usr/bin/env python
"""YOLO11 PSABlock FFN 隐藏维结构化剪枝定向 probe。

该薄封装复用 ``prune_yolo11_adamcf.py`` 的全部实现，只固定：

- ``csp_psa_ffn_closure_v1_experimental`` 累计 coverage profile；
- ``--module-probe-only``，让 HESSO 只处理本轮候选依赖组；
- 独立结构化 checkpoint 名称，避免覆盖 CSP/SPPF 与保守实验产物。

本阶段只新增开放 ``PSABlock.ffn.0`` 隐藏通道。C2PSA split/concat 边界、
Attention qkv/proj/pe、``ffn.1`` 残差输出和 Detect 最终语义输出仍由调用侧
profile 精确固定；这些保护不进入 OTO 核心。
"""

from __future__ import annotations

import sys

from prune_yolo11_adamcf import main


PROFILE = "csp_psa_ffn_closure_v1_experimental"
STRUCTURED_NAME = "yolo11n_psa_ffn_hidden_probe_structured.pt"


def option_value(argv: list[str], option: str) -> str | None:
    for index, token in enumerate(argv[2:], start=2):
        if token == option:
            if index + 1 >= len(argv):
                raise SystemExit(f"{option} 缺少数值。")
            return argv[index + 1]
        if token.startswith(f"{option}="):
            return token.split("=", 1)[1]
    return None


def configure_probe(argv: list[str]) -> None:
    if len(argv) < 2:
        return
    if argv[1] not in {"audit", "prune", "sparse-train", "rebuild-subnet"}:
        raise SystemExit(
            "本入口只用于 audit/prune/rebuild-subnet；微调与导出请使用 prune_yolo11_adamcf.py。"
        )

    profile = option_value(argv, "--module-coverage-profile")
    if profile is None:
        argv.extend(["--module-coverage-profile", PROFILE])
    elif profile != PROFILE:
        raise SystemExit(f"--module-coverage-profile 必须为 {PROFILE}，当前为 {profile}。")

    if "--module-probe-only" not in argv:
        argv.append("--module-probe-only")

    if argv[1] in {"prune", "sparse-train", "rebuild-subnet"}:
        structured_name = option_value(argv, "--structured-name")
        if structured_name is None:
            argv.extend(["--structured-name", STRUCTURED_NAME])
        elif structured_name != STRUCTURED_NAME:
            raise SystemExit(f"--structured-name 必须为 {STRUCTURED_NAME}，当前为 {structured_name}。")


if __name__ == "__main__":
    configure_probe(sys.argv)
    main()
