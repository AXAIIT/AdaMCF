#!/usr/bin/env python
"""YOLO11 全部 ``cv1`` 通道轴累计 L3 probe。

该入口只固定 YOLO11 profile 和验证范围，不复制 OTO/HESSO 核心实现：

- 一次性把模型中全部 ``*.cv1.conv`` 加入候选池；
- C2PSA ``cv1 -> split(2)`` 复用 paired-split 合同；
- C3k2 ``cv1 -> chunk(2)`` 复用 paired-chunk 合同；
- 普通 Bottleneck/C3k 等 ``cv1`` 继续使用 OTO 原生依赖传播；
- Attention 内部 ``num_head/head_dim/key_dim``、Detect 最终语义输出和
  C3k2 child-output 独立动态 concat 轴保持固定；
- 自动生成 OTO 节点组图和原始算子图；
- 当前阶段只用于 audit/prune/rebuild-subnet，不执行微调或导出。
"""

from __future__ import annotations

import sys

from prune_yolo11_adamcf import main


PROFILE = "yolo11_all_cv1_open_v1"
STRUCTURED_NAME = "yolo11n_all_cv1_open_structured.pt"


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
    if families:
        raise SystemExit(
            "all-cv1 profile 已覆盖全部 cv1，不接受 --probe-module-family 缩小范围。"
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
