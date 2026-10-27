#!/usr/bin/env python
"""YOLO11 Bottleneck 隐藏通道物理剪枝 probe 入口。

该薄封装不复制训练逻辑，只为 ``audit`` / ``prune`` 固定：

- ``bottleneck_hidden_v1_experimental`` 覆盖 profile；
- ``--module-probe-only``，使 HESSO 只在 ``Bottleneck.cv1`` 候选组中选择；
- 独立的结构化 checkpoint 名称，避免和 P10/P30 产物混淆。

完整训练仍应回到主入口，并移除 ``--module-probe-only`` 与保守 profile 做对照。
"""

from __future__ import annotations

import sys

from prune_yolo11_adamcf import main


PROFILE = "bottleneck_hidden_v1_experimental"
STRUCTURED_NAME = "yolo11n_bottleneck_hidden_probe_structured.pt"


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
    if argv[1] not in {"audit", "prune", "sparse-train"}:
        raise SystemExit("本入口只用于 audit/prune；微调与导出请使用 prune_yolo11_adamcf.py。")

    profile = option_value(argv, "--module-coverage-profile")
    if profile is None:
        argv.extend(["--module-coverage-profile", PROFILE])
    elif profile != PROFILE:
        raise SystemExit(f"--module-coverage-profile 必须为 {PROFILE}，当前为 {profile}。")

    if "--module-probe-only" not in argv:
        argv.append("--module-probe-only")

    if argv[1] in {"prune", "sparse-train"}:
        structured_name = option_value(argv, "--structured-name")
        if structured_name is None:
            argv.extend(["--structured-name", STRUCTURED_NAME])
        elif structured_name != STRUCTURED_NAME:
            raise SystemExit(f"--structured-name 必须为 {STRUCTURED_NAME}，当前为 {structured_name}。")


if __name__ == "__main__":
    configure_probe(sys.argv)
    main()
