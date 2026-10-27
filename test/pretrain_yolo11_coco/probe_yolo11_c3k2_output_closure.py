#!/usr/bin/env python
"""C3k2 ``cv2`` 对外输出—下游消费者自动闭包的定向 L3 probe。

该入口验证的不是手工保护名单，而是 OTO 图的真实生产者—消费者数据流：普通
Conv 走输出 axis=0 到下游输入 axis=1，深度卷积走 input/output/groups 同步的
耦合轴。YOLO11 profile 只按 C3k2 模块特征自动发现生产者；含 channel Concat 的
路径由 OTO 核心的 ``segmented_concat_consumer_v1`` 合同处理：合同冻结每个输入段
的原始顺序、offset 和宽度，物理构网时合并所有活跃生产者的原始坐标，令消费者
输入轴只裁剪一次。残差、split/slice、非标准 grouped Conv 或无法唯一恢复输入段
的路径仍会明确 fail-closed，绝不猜测坐标。

当前入口仅用于 audit/prune/rebuild-subnet；不执行微调、导出或 ORT。
"""

from __future__ import annotations

import sys

from prune_yolo11_adamcf import main


PROFILE = "c3k2_output_consumer_closure_v1_experimental"
FAMILY = "C3K2_OUTPUT_CLOSURE"
STRUCTURED_NAME = "yolo11n_c3k2_output_closure_probe_structured.pt"


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
            "微调和导出不属于当前结构适配阶段。"
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
    for option in (
        "--visualize-oto-graph",
        "--oto-graph-display-params",
        "--oto-graph-display-flops",
    ):
        if option not in argv:
            argv.append(option)

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
