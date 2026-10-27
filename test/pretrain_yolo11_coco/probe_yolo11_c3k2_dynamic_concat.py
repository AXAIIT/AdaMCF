#!/usr/bin/env python
"""YOLO11 C3k2 独立 child-output 动态 concat 定向 L3 probe。

该入口一次性开放全部具备独立 OTO 输出组的 C3/C3k child-output：

- ``model.6.m.0.cv3``；
- ``model.8.m.0.cv3``；
- ``model.22.m.0.cv3``；
- 根部 ``C3k2.cv1 -> chunk(2)`` 保持固定；
- 根据前序 segment 的当前宽度计算 ``cv2`` 输入删除 offset；
- 自动生成 OTO 节点组图和原始算子图；
- 当前阶段只用于 audit/prune/rebuild-subnet，不执行微调或导出。
"""

from __future__ import annotations

import sys

from prune_yolo11_adamcf import main


PROFILE = "c3k2_dynamic_concat_v1_experimental"
FAMILY = "C3K2_DYNAMIC_CONCAT"
STRUCTURED_NAME = "yolo11n_c3k2_dynamic_concat_probe_structured.pt"


def option_values(argv: list[str], option: str) -> list[str]:
    values: list[str] = []
    for index, token in enumerate(argv[2:], start=2):
        if token == option:
            if index + 1 >= len(argv):
                raise SystemExit(f"{option} 缺少数值。")
            values.append(argv[index + 1])
        elif token.startswith(f"{option}="):
            values.append(token.split("=", 1)[1])
    return values


def option_value(argv: list[str], option: str) -> str | None:
    values = option_values(argv, option)
    if len(values) > 1:
        raise SystemExit(f"{option} 只能指定一次，当前为 {values}。")
    return values[0] if values else None


def configure_probe(argv: list[str]) -> None:
    if len(argv) < 2:
        return
    if argv[1] not in {"audit", "prune", "sparse-train", "rebuild-subnet"}:
        raise SystemExit(
            "本入口只用于 audit/prune/sparse-train/rebuild-subnet；"
            "微调和导出不属于当前逐步开放阶段。"
        )

    profile = option_value(argv, "--module-coverage-profile")
    if profile is None:
        argv.extend(["--module-coverage-profile", PROFILE])
    elif profile != PROFILE:
        raise SystemExit(
            f"--module-coverage-profile 必须为 {PROFILE}，当前为 {profile}。"
        )

    families = option_values(argv, "--probe-module-family")
    if not families:
        argv.extend(["--probe-module-family", FAMILY])
    elif families != [FAMILY]:
        raise SystemExit(
            f"--probe-module-family 必须且只能为 {FAMILY}，当前为 {families}。"
        )

    if "--module-probe-only" not in argv:
        argv.append("--module-probe-only")
    if "--visualize-oto-graph" not in argv:
        argv.append("--visualize-oto-graph")

    if argv[1] in {"prune", "sparse-train", "rebuild-subnet"}:
        structured_name = option_value(argv, "--structured-name")
        if structured_name is None:
            argv.extend(["--structured-name", STRUCTURED_NAME])
        elif structured_name != STRUCTURED_NAME:
            raise SystemExit(
                f"--structured-name 必须为 {STRUCTURED_NAME}，当前为 {structured_name}。"
            )


if __name__ == "__main__":
    configure_probe(sys.argv)
    main()
