#!/usr/bin/env python
"""YOLO11 Attention 的 V/head_dim 结构化剪枝定向 probe。

该模式保持 num_heads 与 Q/K 的 key_dim 不变，只删除所有 heads 中相同位置的
V 维度，并同步裁剪 PE depthwise 通道与 proj 输入通道。
"""

from __future__ import annotations

import sys

from prune_yolo11_adamcf import main


PROFILE = "c2psa_attention_head_dim_v1_experimental"
FAMILY = "YOLO11_ATTN_HEAD_DIM"
STRUCTURED_NAME = "yolo11n_attention_head_dim_probe_structured.pt"


def option_value(argv: list[str], option: str) -> str | None:
    for index, token in enumerate(argv[2:], start=2):
        if token == option:
            if index + 1 >= len(argv):
                raise SystemExit(f"{option} 缺少数值。")
            return argv[index + 1]
        if token.startswith(f"{option}="):
            return token.split("=", 1)[1]
    return None


def ensure_option(argv: list[str], option: str, value: str) -> None:
    current = option_value(argv, option)
    if current is None:
        argv.extend([option, value])
    elif current != value:
        raise SystemExit(f"{option} 必须为 {value}，当前为 {current}。")


def configure_probe(argv: list[str]) -> None:
    if len(argv) < 2:
        return
    if argv[1] not in {"audit", "prune", "sparse-train", "rebuild-subnet"}:
        raise SystemExit("本入口只用于 audit/prune/sparse-train/rebuild-subnet。")
    ensure_option(argv, "--module-coverage-profile", PROFILE)
    if "--module-probe-only" not in argv:
        argv.append("--module-probe-only")
    families = [
        argv[index + 1]
        for index, token in enumerate(argv[:-1])
        if token == "--probe-module-family"
    ]
    if not families:
        argv.extend(["--probe-module-family", FAMILY])
    elif families != [FAMILY]:
        raise SystemExit(f"--probe-module-family 必须且只能为 {FAMILY}，当前为 {families}。")
    if argv[1] in {"prune", "sparse-train", "rebuild-subnet"}:
        ensure_option(argv, "--structured-name", STRUCTURED_NAME)


if __name__ == "__main__":
    configure_probe(sys.argv)
    main()
