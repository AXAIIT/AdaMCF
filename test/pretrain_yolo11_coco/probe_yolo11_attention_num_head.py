#!/usr/bin/env python
"""YOLO11 Attention 完整 head 结构化剪枝定向 probe。"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from prune_yolo11_adamcf import (
    compression_summary,
    device_from_arg,
    json_dump,
    main,
    model_stats,
    model_structure_signature,
    primary_prediction_tensor,
    sha256_file,
    yolo11_attention_structure_summary,
)


PROFILE = "c2psa_attention_num_head_v1_experimental"
FAMILY = "YOLO11_ATTN_NUM_HEAD"
STRUCTURED_NAME = "yolo11n_attention_num_head_probe_structured.pt"
PRUNE_RATIO = 0.50  # YOLO11n 的 C2PSA Attention 为 2 heads，必须删除 1/2 才能真实命中。


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
    if argv[1] in {"prune", "sparse-train"}:
        ratio = option_value(argv, "--prune-ratio")
        if ratio is None:
            argv.extend(["--prune-ratio", str(PRUNE_RATIO)])
        elif not math.isclose(float(ratio), PRUNE_RATIO, rel_tol=0.0, abs_tol=1e-12):
            raise SystemExit(f"完整 head probe 的 --prune-ratio 必须为 {PRUNE_RATIO}。")
    if argv[1] in {"prune", "sparse-train", "rebuild-subnet"}:
        ensure_option(argv, "--structured-name", STRUCTURED_NAME)


def attention_snapshot(model) -> list[dict[str, Any]]:
    """读取 YOLO11 Attention 的真实物理属性，用于剪枝前后对比。"""

    records: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not all(
            hasattr(module, attr)
            for attr in ("qkv", "proj", "pe", "num_heads", "key_dim", "head_dim")
        ):
            continue
        qkv = getattr(getattr(module, "qkv", None), "conv", None)
        proj = getattr(getattr(module, "proj", None), "conv", None)
        pe = getattr(getattr(module, "pe", None), "conv", None)
        if not all(isinstance(conv, torch.nn.Conv2d) for conv in (qkv, proj, pe)):
            continue
        records.append(
            {
                "module_prefix": name,
                "module_type": type(module).__name__,
                "prune_mode": getattr(module, "_oto_attention_prune_mode", None),
                "num_heads": int(module.num_heads),
                "key_dim": int(module.key_dim),
                "head_dim": int(module.head_dim),
                "inner_dim": int(module.num_heads) * int(module.head_dim),
                "qkv_in_channels": int(qkv.in_channels),
                "qkv_out_channels": int(qkv.out_channels),
                "pe_in_channels": int(pe.in_channels),
                "pe_out_channels": int(pe.out_channels),
                "pe_groups": int(pe.groups),
                "proj_in_channels": int(proj.in_channels),
                "proj_out_channels": int(proj.out_channels),
            }
        )
    return records


def forward_report(model, imgsz: int, device) -> dict[str, Any]:
    """执行真实固定 shape 前向，确认输出 shape 和有限性。"""

    model = model.to(device).float().eval()
    inputs = torch.zeros(1, 3, imgsz, imgsz, device=device)
    with torch.inference_mode():
        prediction = primary_prediction_tensor(model(inputs))
    report = {
        "input_shape": list(inputs.shape),
        "output_shape": list(prediction.shape),
        "output_dtype": str(prediction.dtype),
        "finite": bool(torch.isfinite(prediction).all().item()),
        "min": float(prediction.min().item()),
        "max": float(prediction.max().item()),
        "mean": float(prediction.float().mean().item()),
    }
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def print_loaded_model_comparison(report: dict[str, Any], report_path: Path) -> None:
    """按 P30 主流程风格打印面向人的剪枝前后对比总结。"""

    before_stats = report["baseline_model_stats"]
    after_stats = report["structured_model_stats"]
    compression = report["compression"]
    before_forward = report["baseline_forward"]
    after_forward = report["structured_forward"]

    print("\n" + "=" * 72, flush=True)
    print("YOLO11 Attention 物理剪枝真实加载对比", flush=True)
    print("=" * 72, flush=True)
    print(f"未剪枝模型: {report['baseline_weights']}", flush=True)
    print(f"剪枝后模型: {report['structured_weights']}", flush=True)
    print(f"测试输入: 1×3×{report['imgsz']}×{report['imgsz']} | 设备: {report['device']}", flush=True)

    print("\n--- 1. 全模型参数量与计算量 ---", flush=True)
    print(
        f"参数量: {int(before_stats['parameters']):,} -> "
        f"{int(after_stats['parameters']):,} | "
        f"减少 {int(compression['parameters_reduction']):,} "
        f"({compression['parameters_reduction_ratio'] * 100:.4f}%) | "
        f"压缩比 {compression['parameters_compression_ratio']:.6f}x",
        flush=True,
    )
    print(
        f"THOP FLOPs: {before_stats['flops_g']:.6f} G -> "
        f"{after_stats['flops_g']:.6f} G | "
        f"减少 {compression['flops_reduction'] / 1e6:.3f} M "
        f"({compression['flops_reduction_ratio'] * 100:.4f}%) | "
        f"压缩比 {compression['flops_compression_ratio']:.6f}x",
        flush=True,
    )

    print("\n--- 2. Attention 物理结构 ---", flush=True)
    for before, after in zip(report["baseline_attention"], report["structured_attention"]):
        print(f"模块: {before['module_prefix']}", flush=True)
        print(
            f"类型: {before['module_type']} -> {after['module_type']}",
            flush=True,
        )
        print(
            f"剪枝模式: 未剪枝 -> {after['prune_mode']}",
            flush=True,
        )
        print(
            f"Attention heads: {before['num_heads']} -> {after['num_heads']} "
            f"(减少 {(before['num_heads'] - after['num_heads']) / before['num_heads'] * 100:.2f}%)",
            flush=True,
        )
        print(
            f"内部宽度: {before['inner_dim']} -> {after['inner_dim']}",
            flush=True,
        )
        print(
            f"QKV 输出通道: {before['qkv_out_channels']} -> {after['qkv_out_channels']}",
            flush=True,
        )
        print(
            f"PE 输入/输出/groups: "
            f"{before['pe_in_channels']}/{before['pe_out_channels']}/{before['pe_groups']} -> "
            f"{after['pe_in_channels']}/{after['pe_out_channels']}/{after['pe_groups']}",
            flush=True,
        )
        print(
            f"proj 输入通道: {before['proj_in_channels']} -> {after['proj_in_channels']}",
            flush=True,
        )
        print(
            f"proj 输出通道: {before['proj_out_channels']} -> {after['proj_out_channels']} "
            "(外部输出维保持不变)",
            flush=True,
        )

    print("\n--- 3. 真实前向可用性 ---", flush=True)
    print(
        f"未剪枝输出: shape={before_forward['output_shape']} | "
        f"dtype={before_forward['output_dtype']} | finite={before_forward['finite']}",
        flush=True,
    )
    print(
        f"剪枝后输出: shape={after_forward['output_shape']} | "
        f"dtype={after_forward['output_dtype']} | finite={after_forward['finite']}",
        flush=True,
    )
    print(
        f"结构指纹是否变化: {report['structure_changed']} | "
        f"Attention 结构合同: {report['structured_attention_contract'].get('status')}",
        flush=True,
    )

    print("\n--- 4. 验证结论 ---", flush=True)
    if report["status"] == "pass":
        print("[PASS] 参数量真实下降", flush=True)
        print("[PASS] THOP FLOPs 真实下降", flush=True)
        print("[PASS] Attention head 已发生物理删除", flush=True)
        print("[PASS] 剪枝前后结构指纹不同", flush=True)
        print("[PASS] 剪枝模型可独立加载并完成前向", flush=True)
        print("[PASS] 输出 shape 保持一致且无 NaN/Inf", flush=True)
        print("最终结论: Attention 物理结构化剪枝有效，剪枝模型可用。", flush=True)
    else:
        print("[FAIL] Attention 剪枝模型对比未通过：", flush=True)
        for issue in report["issues"]:
            print(f"  - {issue}", flush=True)

    print(f"\n机器可读完整报告: {report_path}", flush=True)
    print("=" * 72, flush=True)


def compare_loaded_models_after_rebuild(argv: list[str]) -> None:
    """rebuild-subnet 后重新加载原模型和物理子网，生成真实对比报告。"""

    if len(argv) < 2 or argv[1] != "rebuild-subnet":
        return

    base_dir = Path(__file__).resolve().parent
    output = Path(option_value(argv, "--output") or (base_dir / "runs")).expanduser().resolve()
    name = option_value(argv, "--name") or "yolo11_adamcf"
    run_dir = output / name
    structured_weights = run_dir / "weights" / STRUCTURED_NAME
    baseline_weights = Path(
        option_value(argv, "--baseline-weights") or (base_dir / "models" / "yolo11n.pt")
    ).expanduser().resolve()
    imgsz = int(option_value(argv, "--imgsz") or 640)
    device = device_from_arg(option_value(argv, "--device") or "0")

    if not baseline_weights.is_file():
        raise FileNotFoundError(f"未剪枝模型不存在：{baseline_weights}")
    if not structured_weights.is_file():
        raise FileNotFoundError(f"Attention 物理子网不存在：{structured_weights}")

    print("\n========== 加载未剪枝模型与 Attention 剪枝模型重新统计 ==========", flush=True)
    baseline_model = YOLO(str(baseline_weights)).model.cpu().float().eval()
    structured_model = YOLO(str(structured_weights)).model.cpu().float().eval()

    baseline_stats = model_stats(baseline_model, imgsz, device)
    structured_stats = model_stats(structured_model, imgsz, device)
    compression = compression_summary(baseline_stats, structured_stats)
    baseline_signature = model_structure_signature(baseline_model)
    structured_signature = model_structure_signature(structured_model)
    baseline_attention = attention_snapshot(baseline_model)
    structured_attention = attention_snapshot(structured_model)
    structured_contract = yolo11_attention_structure_summary(structured_model)
    baseline_forward = forward_report(baseline_model, imgsz, device)
    structured_forward = forward_report(structured_model, imgsz, device)

    issues: list[str] = []
    if structured_stats["parameters"] >= baseline_stats["parameters"]:
        issues.append("剪枝模型参数量未下降")
    if structured_stats["flops"] >= baseline_stats["flops"]:
        issues.append("剪枝模型 THOP FLOPs 未下降")
    if baseline_signature["sha256"] == structured_signature["sha256"]:
        issues.append("剪枝前后结构指纹相同")
    if structured_contract.get("status") != "pass":
        issues.append(f"Attention 物理结构合同失败：{structured_contract}")
    if not baseline_forward["finite"] or not structured_forward["finite"]:
        issues.append("剪枝前后至少一个模型输出包含 NaN/Inf")
    if baseline_forward["output_shape"] != structured_forward["output_shape"]:
        issues.append(
            "剪枝前后检测输出 shape 不一致："
            f"{baseline_forward['output_shape']} != {structured_forward['output_shape']}"
        )
    if len(baseline_attention) != len(structured_attention) or not structured_attention:
        issues.append(
            "剪枝前后 Attention 模块数量异常："
            f"{len(baseline_attention)} != {len(structured_attention)}"
        )
    else:
        for before, after in zip(baseline_attention, structured_attention):
            if after["prune_mode"] != "num_head":
                issues.append(f"prune_mode 错误：{after['prune_mode']} != num_head")
            if after["num_heads"] >= before["num_heads"]:
                issues.append(
                    "Attention head 未减少："
                    f"{before['num_heads']} -> {after['num_heads']}"
                )

    report = {
        "status": "pass" if not issues else "fail",
        "baseline_weights": str(baseline_weights),
        "baseline_weights_sha256": sha256_file(baseline_weights),
        "structured_weights": str(structured_weights),
        "structured_weights_sha256": sha256_file(structured_weights),
        "imgsz": imgsz,
        "device": str(device),
        "baseline_model_stats": baseline_stats,
        "structured_model_stats": structured_stats,
        "compression": compression,
        "baseline_structure_signature": baseline_signature,
        "structured_structure_signature": structured_signature,
        "structure_changed": baseline_signature["sha256"] != structured_signature["sha256"],
        "baseline_attention": baseline_attention,
        "structured_attention": structured_attention,
        "structured_attention_contract": structured_contract,
        "baseline_forward": baseline_forward,
        "structured_forward": structured_forward,
        "issues": issues,
    }
    report_path = run_dir / "loaded_model_comparison.json"
    json_dump(report_path, report)

    summary_path = run_dir / "prune_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise TypeError(f"prune_summary.json 顶层必须是对象：{summary_path}")
        summary["loaded_model_comparison"] = report
        json_dump(summary_path, summary)

    print_loaded_model_comparison(report, report_path)
    if issues:
        raise RuntimeError("Attention 剪枝模型真实加载对比失败：" + "；".join(issues))


if __name__ == "__main__":
    configure_probe(sys.argv)
    main()
    compare_loaded_models_after_rebuild(sys.argv)
