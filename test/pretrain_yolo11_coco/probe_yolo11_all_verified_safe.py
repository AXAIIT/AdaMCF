#!/usr/bin/env python
"""YOLO11 已验证安全集合的联合剪枝-only L3 入口。

该入口只封装运行合同，不复制 OTO/HESSO 或物理构网实现。它将此前独立通过
准入的 dense closure、C2PSA 输出闭包、C3k2 composite 与一种 Attention 内部
轴合并为一次全量搜索；Attention ``num_head`` 与 ``head_dim`` 仍保持两条互斥
profile。C2PSA paired-split 的外部宽度轴会改写同一 Attention/FFN 接口，当前
仍保留为独立 profile，不能与 Attention 内部轴在同一轮混用。

仅支持 audit/prune/sparse-train/rebuild-subnet。微调、导出和 ONNX Runtime
对齐不属于本次全量联合剪枝-only 验证。
"""

from __future__ import annotations

import sys

from prune_yolo11_adamcf import main


DEFAULT_PROFILE = "all_verified_safe_num_head_v2_experimental"
STRUCTURED_NAMES = {
    "all_verified_safe_num_head_v2_experimental": (
        "yolo11n_all_verified_safe_num_head_v2_structured.pt"
    ),
    "all_verified_safe_head_dim_v2_experimental": (
        "yolo11n_all_verified_safe_head_dim_v2_structured.pt"
    ),
    "all_verified_safe_num_head_v4_output_closure_experimental": (
        "yolo11n_all_verified_safe_num_head_v4_output_closure_structured.pt"
    ),
}
SUPPORTED_MODES = {"audit", "prune", "sparse-train", "rebuild-subnet"}


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
    if argv[1] not in SUPPORTED_MODES:
        raise SystemExit(
            "本入口只用于 audit/prune/sparse-train/rebuild-subnet；"
            "本轮不允许启动微调、导出或 ORT 对齐。"
        )

    profile = option_value(argv, "--module-coverage-profile")
    if profile is None:
        profile = DEFAULT_PROFILE
        argv.extend(["--module-coverage-profile", profile])
    elif profile not in STRUCTURED_NAMES:
        raise SystemExit(
            "--module-coverage-profile 必须为全量联合 Attention-num-head 或 "
            f"Attention-head-dim profile，当前为 {profile}。"
        )

    families = option_values(argv, "--probe-module-family")
    if families:
        raise SystemExit(
            "全量联合 profile 禁止 --probe-module-family；各模块族由同一轮 HESSO "
            "重要性排序共同选择，audit 会分别检查候选覆盖。"
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
        expected_name = STRUCTURED_NAMES[profile]
        if structured_name is None:
            argv.extend(["--structured-name", expected_name])
        elif structured_name != expected_name:
            raise SystemExit(
                f"{profile} 的 --structured-name 必须为 {expected_name}，"
                f"当前为 {structured_name}。"
            )


if __name__ == "__main__":
    configure_probe(sys.argv)
    main()
