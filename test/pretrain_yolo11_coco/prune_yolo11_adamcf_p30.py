#!/usr/bin/env python
"""YOLO11 AdaMCF 30% 组稀疏率专用入口。

本文件不复制主流程实现，而是固定 ``prune`` / ``sparse-train`` 模式的
``--prune-ratio`` 为 0.30，并把物理子网固定保存为
``yolo11n_adamcf_p30_structured.pt``，然后委托主入口执行。
因此物理子网构建、独立验证、微调时的结构/权重哈希校验和总结格式均与
已经验证的主入口保持完全一致。
"""

from __future__ import annotations

import math
import sys

from prune_yolo11_adamcf import main


P30_RATIO = 0.30
P30_STRUCTURED_NAME = "yolo11n_adamcf_p30_structured.pt"


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


def enforce_p30(argv: list[str]) -> None:
    """为剪枝模式注入30%比例和专用文件名，并拒绝不一致的显式配置。"""

    if len(argv) < 2:
        return
    if argv[1] not in {"prune", "sparse-train"}:
        return

    ratio_value = option_value(argv, "--prune-ratio")
    ratio = float(ratio_value) if ratio_value is not None else None

    if ratio is None:
        argv.extend(["--prune-ratio", str(P30_RATIO)])
    elif not math.isclose(ratio, P30_RATIO, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(
            f"{__file__} 是 30% 专用入口，--prune-ratio 必须为 {P30_RATIO:.2f}，当前为 {ratio}。"
        )

    structured_name = option_value(argv, "--structured-name")
    if structured_name is None:
        argv.extend(["--structured-name", P30_STRUCTURED_NAME])
    elif structured_name != P30_STRUCTURED_NAME:
        raise SystemExit(
            f"30% 专用入口的 --structured-name 必须为 {P30_STRUCTURED_NAME}，"
            f"当前为 {structured_name}。"
        )


if __name__ == "__main__":
    enforce_p30(sys.argv)
    main()
