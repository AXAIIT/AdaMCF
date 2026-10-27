#!/usr/bin/env python
"""YOLO11 C3k2 root paired-chunk + child DynamicConcat 复合 L3 probe。

本入口同时开放两条互不重叠的逻辑轴：

- C3k2.cv1 的 paired chunk 根部宽度；
- C3/C3k child 的 cv3 独立输出宽度。

二者共同消费同一 C3k2.cv2 输入轴，因此主脚本中的父级协调器会先收集全部
零组，再按原始坐标一次性裁剪。该入口只做 audit/prune/rebuild-subnet；微调、
导出与 ORT 对齐由统一九阶段编排器按单独授权执行，不替换已有实验产物。
"""

from __future__ import annotations

import sys

from prune_yolo11_adamcf import main


PROFILE = "c3k2_paired_chunk_dynamic_concat_v1_experimental"
FAMILIES = ("C3K2_CHUNK", "C3K2_DYNAMIC_CONCAT")
STRUCTURED_NAME = "yolo11n_c3k2_composite_probe_structured.pt"


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
        for family in FAMILIES:
            argv.extend(["--probe-module-family", family])
    elif set(families) != set(FAMILIES) or len(families) != len(FAMILIES):
        raise SystemExit(
            "--probe-module-family 必须各指定一次 "
            f"{', '.join(FAMILIES)}，当前为 {families}。"
        )

    if "--module-probe-only" not in argv:
        argv.append("--module-probe-only")
    if "--visualize-oto-graph" not in argv:
        argv.append("--visualize-oto-graph")
    if "--oto-graph-display-params" not in argv:
        argv.append("--oto-graph-display-params")
    if "--oto-graph-display-flops" not in argv:
        argv.append("--oto-graph-display-flops")

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
