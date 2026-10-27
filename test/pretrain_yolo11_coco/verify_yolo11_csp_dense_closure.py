#!/usr/bin/env python
"""按同一冻结 v3 合同执行 YOLO11 复杂模块九段准入闭环。

该文件只负责编排和汇总证据；OTO/HESSO、物理子网、微调和导出实现仍复用
``prune_yolo11_adamcf.py``，避免出现第二份剪枝逻辑。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
CORE_SCRIPT = BASE_DIR / "prune_yolo11_adamcf.py"
DEFAULT_PROFILE = "csp_dense_closure_v1_experimental"
C3K2_DYNAMIC_CONCAT_PROFILE = "c3k2_dynamic_concat_v1_experimental"
C3K2_COMPOSITE_PROFILE = "c3k2_paired_chunk_dynamic_concat_v1_experimental"
C3K2_OUTPUT_CLOSURE_PROFILE = "c3k2_output_consumer_closure_v1_experimental"
ALL_CV1_OPEN_PROFILE = "yolo11_all_cv1_open_v1"
ALL_VERIFIED_SAFE_PROFILE = "all_verified_safe_num_head_v1_experimental"
ALL_VERIFIED_SAFE_NUM_HEAD_V2_PROFILE = "all_verified_safe_num_head_v2_experimental"
ALL_VERIFIED_SAFE_HEAD_DIM_V2_PROFILE = "all_verified_safe_head_dim_v2_experimental"
ALL_VERIFIED_SAFE_NUM_HEAD_V4_OUTPUT_CLOSURE_PROFILE = (
    "all_verified_safe_num_head_v4_output_closure_experimental"
)
ALL_VERIFIED_SAFE_V2_PROFILES = frozenset(
    {
        ALL_VERIFIED_SAFE_NUM_HEAD_V2_PROFILE,
        ALL_VERIFIED_SAFE_HEAD_DIM_V2_PROFILE,
    }
)
ALL_VERIFIED_SAFE_JOINT_COMPOSITE_PROFILES = frozenset(
    {
        *ALL_VERIFIED_SAFE_V2_PROFILES,
        ALL_VERIFIED_SAFE_NUM_HEAD_V4_OUTPUT_CLOSURE_PROFILE,
    }
)
OUTPUT_CONSUMER_CLOSURE_PROFILES = frozenset(
    {
        C3K2_OUTPUT_CLOSURE_PROFILE,
        ALL_VERIFIED_SAFE_NUM_HEAD_V4_OUTPUT_CLOSURE_PROFILE,
    }
)
PROBE_CONFIGS = {
    DEFAULT_PROFILE: {
        "script": BASE_DIR / "probe_yolo11_csp_dense_closure.py",
        "structured_name": "yolo11n_csp_dense_closure_probe_structured.pt",
    },
    "csp_psa_ffn_closure_v1_experimental": {
        "script": BASE_DIR / "probe_yolo11_psa_ffn_hidden.py",
        "structured_name": "yolo11n_psa_ffn_hidden_probe_structured.pt",
    },
    "c2psa_output_closure_v1_experimental": {
        "script": BASE_DIR / "probe_yolo11_c2psa_output.py",
        "structured_name": "yolo11n_c2psa_output_probe_structured.pt",
    },
    "c2psa_paired_split_v1_experimental": {
        "script": BASE_DIR / "probe_yolo11_c2psa_paired_split.py",
        "structured_name": "yolo11n_c2psa_paired_split_probe_structured.pt",
    },
    "c3k2_paired_chunk_v1_experimental": {
        "script": BASE_DIR / "probe_yolo11_c3k2_paired_chunk.py",
        "structured_name": "yolo11n_c3k2_paired_chunk_probe_structured.pt",
    },
    C3K2_DYNAMIC_CONCAT_PROFILE: {
        "script": BASE_DIR / "probe_yolo11_c3k2_dynamic_concat.py",
        "structured_name": "yolo11n_c3k2_dynamic_concat_probe_structured.pt",
    },
    C3K2_COMPOSITE_PROFILE: {
        "script": BASE_DIR / "probe_yolo11_c3k2_composite.py",
        "structured_name": "yolo11n_c3k2_composite_probe_structured.pt",
    },
    C3K2_OUTPUT_CLOSURE_PROFILE: {
        "script": BASE_DIR / "probe_yolo11_c3k2_output_closure.py",
        "structured_name": "yolo11n_c3k2_output_closure_probe_structured.pt",
    },
    ALL_CV1_OPEN_PROFILE: {
        "script": BASE_DIR / "probe_yolo11_all_cv1_open.py",
        "structured_name": "yolo11n_all_cv1_open_structured.pt",
    },
    "c2psa_attention_num_head_v1_experimental": {
        "script": BASE_DIR / "probe_yolo11_attention_num_head.py",
        "structured_name": "yolo11n_attention_num_head_probe_structured.pt",
    },
    "c2psa_attention_head_dim_v1_experimental": {
        "script": BASE_DIR / "probe_yolo11_attention_head_dim.py",
        "structured_name": "yolo11n_attention_head_dim_probe_structured.pt",
    },
    ALL_VERIFIED_SAFE_PROFILE: {
        "script": BASE_DIR / "prune_yolo11_adamcf_p50.py",
        "structured_name": "yolo11n_adamcf_p50_all_verified_structured.pt",
    },
    ALL_VERIFIED_SAFE_NUM_HEAD_V2_PROFILE: {
        "script": BASE_DIR / "probe_yolo11_all_verified_safe.py",
        "structured_name": "yolo11n_all_verified_safe_num_head_v2_structured.pt",
    },
    ALL_VERIFIED_SAFE_HEAD_DIM_V2_PROFILE: {
        "script": BASE_DIR / "probe_yolo11_all_verified_safe.py",
        "structured_name": "yolo11n_all_verified_safe_head_dim_v2_structured.pt",
    },
    ALL_VERIFIED_SAFE_NUM_HEAD_V4_OUTPUT_CLOSURE_PROFILE: {
        "script": BASE_DIR / "probe_yolo11_all_verified_safe.py",
        "structured_name": (
            "yolo11n_all_verified_safe_num_head_v4_output_closure_structured.pt"
        ),
    },
}
DEFAULT_REQUIRED_FAMILIES = ("C3k2", "C3k", "Bottleneck", "SPPF", "Detect")
ATTENTION_ONLY_PROFILES = frozenset(
    {
        "c2psa_attention_num_head_v1_experimental",
        "c2psa_attention_head_dim_v1_experimental",
    }
)
PROFILE_REQUIRED_FAMILIES: dict[str, tuple[str, ...]] = {
    "c2psa_paired_split_v1_experimental": (
        "C2PSA_SPLIT",
    ),
    "c3k2_paired_chunk_v1_experimental": (
        "C3K2_CHUNK",
    ),
    C3K2_DYNAMIC_CONCAT_PROFILE: (
        "C3K2_DYNAMIC_CONCAT",
    ),
    C3K2_COMPOSITE_PROFILE: (
        "C3K2_CHUNK",
        "C3K2_DYNAMIC_CONCAT",
    ),
    C3K2_OUTPUT_CLOSURE_PROFILE: (
        "C3K2_OUTPUT_CLOSURE",
    ),
    ALL_CV1_OPEN_PROFILE: (
        "C2PSA_SPLIT",
        "C3K2_CHUNK",
    ),
    "c2psa_attention_num_head_v1_experimental": (
        "YOLO11_ATTN_NUM_HEAD",
    ),
    "c2psa_attention_head_dim_v1_experimental": (
        "YOLO11_ATTN_HEAD_DIM",
    ),
}
STAGES = ("audit", "prune", "finetune", "export", "report")

STAGE_LABELS = {
    "audit": "依赖图审计",
    "prune": "稀疏训练与物理构网",
    "finetune": "结构化子网微调",
    "export": "ONNX 导出与对齐",
    "report": "准入汇总",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def nested(value: dict[str, Any], *keys: str, default=None):
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def scalar(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def print_human_summary(
    title: str,
    rows: list[tuple[str, Any]],
    *,
    artifact: Path | str | None = None,
) -> None:
    """控制台展示精简结论，完整证据保存在 JSON 和阶段日志中。"""

    print("\n" + "=" * 72, flush=True)
    print(title, flush=True)
    print("=" * 72, flush=True)
    visible_rows = [(str(label), value) for label, value in rows if value is not None]
    width = max((len(label) for label, _ in visible_rows), default=0)
    for label, value in visible_rows:
        print(f"{label:<{width}} : {value}", flush=True)
    if artifact is not None:
        print(f"详细报告{' ' * max(0, width - 4)} : {artifact}", flush=True)


def compact_family_hits(family_hits: dict[str, Any]) -> str:
    if not isinstance(family_hits, dict) or not family_hits:
        return "无"
    parts = []
    for family, record in family_hits.items():
        if isinstance(record, dict):
            parts.append(
                f"{family}({int(record.get('dependency_group_count', 0) or 0)}组)"
            )
    return "，".join(parts) if parts else "无"


def resolved_required_families(args: argparse.Namespace) -> list[str]:
    """返回当前运行必须取得真实物理命中的模块族。

    定向 probe 必须命中其目标族；联合全局 profile 只要求所有已验证族在 audit
    后仍进入可剪候选池，不强制全局重要性排序在同一次运行中每族至少删除一个组。
    """

    if args.required_hit_family:
        return list(args.required_hit_family)
    if args.probe_module_family:
        return list(args.probe_module_family)
    if args.module_coverage_profile in {
        ALL_VERIFIED_SAFE_PROFILE,
        *ALL_VERIFIED_SAFE_JOINT_COMPOSITE_PROFILES,
    }:
        return []
    return list(
        PROFILE_REQUIRED_FAMILIES.get(
            args.module_coverage_profile,
            DEFAULT_REQUIRED_FAMILIES,
        )
    )


def write_console_bytes(chunk: bytes) -> None:
    """原样转发子进程 stdout，保留 Ultralytics 进度条的 ``\r`` 覆盖语义。"""

    binary_stdout = getattr(sys.stdout, "buffer", None)
    if binary_stdout is not None:
        binary_stdout.write(chunk)
        binary_stdout.flush()
    else:  # pragma: no cover - 兼容测试宿主替换 sys.stdout 的情形
        sys.stdout.write(chunk.decode("utf-8", errors="replace"))
        sys.stdout.flush()


def run_stage(stage: str, command: list[str], log_path: Path) -> float:
    """执行一个阶段，并返回启动时刻供实际 Ultralytics run-dir 解析使用。"""

    started_at = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = subprocess.list2cmdline(command)
    stage_label = STAGE_LABELS.get(stage, stage)
    print("\n" + "=" * 80, flush=True)
    print(f"阶段开始：{stage_label}", flush=True)
    print(f"执行命令：{printable}", flush=True)
    print("=" * 80, flush=True)
    with log_path.open("wb") as log:
        log.write((printable + "\n").encode("utf-8"))
        process = subprocess.Popen(
            command,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # 不能启用 text/universal-newline：它会把 ``\r`` 转成 ``\n``，令
            # Ultralytics 的单行 batch 进度条变成每 batch 一行。
            bufsize=0,
        )
        assert process.stdout is not None
        last_byte: bytes | None = None
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            log.write(chunk)
            log.flush()
            write_console_bytes(chunk)
            last_byte = chunk[-1:]
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"阶段 {stage} 失败，exit_code={return_code}，日志：{log_path}")
    if last_byte not in {None, b"\n"}:
        print(flush=True)
    print(f"阶段完成：{stage_label}；日志：{log_path}", flush=True)
    return started_at


def replace_option_value(command: list[str], option: str, value: str) -> None:
    """原地替换 ``--key value`` 命令项，避免后续阶段引用静态预期目录。"""

    for index, token in enumerate(command):
        if token == option:
            if index + 1 >= len(command):
                raise ValueError(f"命令项 {option} 缺少值：{command}")
            command[index + 1] = value
            return
    raise ValueError(f"命令中未找到 {option}：{command}")


def update_prune_paths(
    paths: dict[str, Path],
    *,
    prune_dir: Path,
    structured_name: str,
    rebuild_from_full_sparse: bool,
) -> None:
    """将所有后续消费方绑定到实际完成的剪枝目录。"""

    paths["prune_metadata"] = prune_dir / (
        "rebuild_structured_subnet.json"
        if rebuild_from_full_sparse
        else "adamcf_run_metadata.json"
    )
    paths["prune_summary"] = prune_dir / "prune_summary.json"
    paths["structured_weights"] = prune_dir / "weights" / structured_name


def update_finetune_paths(paths: dict[str, Path], *, finetune_dir: Path, run_id: str) -> None:
    """将导出与九段报告绑定到实际完成的微调目录。"""

    paths["finetune_summary"] = finetune_dir / "finetune_summary.json"
    paths["onnx"] = finetune_dir / f"{run_id}.onnx"
    paths["export_metadata"] = paths["onnx"].with_suffix(paths["onnx"].suffix + ".metadata.json")


def resolve_actual_stage_run_dir(
    *,
    output: Path,
    requested_name: str,
    completion_artifact: str,
    stage_started_at: float | None,
) -> Path:
    """解析 Ultralytics 的实际 run-dir，兼容其自动追加 ``2``、``3`` 等后缀。

    不使用“取最新同前缀目录”的不安全猜测：新启动的阶段必须在启动时刻之后
    写出完成标记；恢复既有阶段则由调用方显式传入 ``--prune-run-dir``。
    """

    name_pattern = re.compile(rf"^{re.escape(requested_name)}(?:\d+)?$")
    candidates = []
    for directory in output.iterdir() if output.is_dir() else ():
        if not directory.is_dir() or not name_pattern.fullmatch(directory.name):
            continue
        artifact = directory / completion_artifact
        if artifact.is_file():
            candidates.append((artifact.stat().st_mtime, directory))
    if not candidates:
        raise FileNotFoundError(
            f"阶段完成后未找到 {completion_artifact}；请求目录前缀={requested_name}，输出目录={output}"
        )

    if stage_started_at is not None:
        # 文件系统时间存在亚秒/秒级舍入，保留 2 秒容差；训练/微调本身远大于此窗口。
        recent = [item for item in candidates if item[0] >= stage_started_at - 2.0]
        if len(recent) != 1:
            details = [
                {"directory": str(directory), "artifact_mtime": modified}
                for modified, directory in sorted(candidates, key=lambda item: item[0], reverse=True)
            ]
            raise RuntimeError(
                "无法唯一确定本次阶段的实际 Ultralytics 运行目录；"
                f"requested_name={requested_name}, completion_artifact={completion_artifact}, "
                f"recent_count={len(recent)}, candidates={details}。"
            )
        actual_dir = recent[0][1]
    else:
        # 仅用于用户显式指定的恢复路径之外的只读诊断，不允许静默选择旧目录。
        raise RuntimeError("恢复既有剪枝阶段必须显式提供 --prune-run-dir，禁止自动猜测旧目录。")

    print(
        f"[run-dir] 请求目录={output / requested_name}；实际目录={actual_dir}",
        flush=True,
    )
    return actual_dir


def build_admission_report(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    audit = read_json(paths["audit_json"])
    prune_metadata = read_json(paths["prune_metadata"])
    prune_summary = read_json(paths["prune_summary"])
    finetune_summary = read_json(paths["finetune_summary"])
    export_metadata = read_json(paths["export_metadata"])

    audit_contract = nested(audit, "module_dependency_control", "contract", "sha256")
    prune_contract = nested(prune_metadata, "module_dependency_control", "contract", "sha256")
    contract_matched = bool(audit_contract) and audit_contract == prune_contract
    candidate_family_coverage = nested(
        audit,
        "module_dependency_control",
        "candidate_family_coverage",
        default={},
    ) or {}

    hesso_pruned_groups = scalar(nested(prune_summary, "hesso", "num_pruned_groups")) or 0.0
    redundancy = nested(prune_metadata, "physical_redundancy", default={}) or {}
    candidate_hit_groups = scalar(redundancy.get("actual_candidate_hit_dependency_group_count")) or 0.0
    raw_family_hits = redundancy.get("opened_module_family_hits", {})
    if not isinstance(raw_family_hits, dict):
        raw_family_hits = {}
    family_hits = {
        str(family): dict(counters)
        for family, counters in raw_family_hits.items()
        if isinstance(counters, dict)
    }
    family_hit_classification: dict[str, Any] = {
        "status": "metadata_used_as_is",
        "raw_opened_module_family_hits": raw_family_hits,
        "normalized_opened_module_family_hits": family_hits,
    }

    # 旧 physical_redundancy_summary 的模块族集合没有包含 PSA_FFN，导致已精确
    # 命中的 PSABlock.ffn.0 被误标为 standalone Conv。对于单一模块族定向 probe，
    # audit 候选组和物理命中组在相同 contract 下可提供无歧义重分类证据，允许只
    # 重建 admission report，不改写历史训练 metadata，也不重复训练。
    directed_families = list(args.probe_module_family or [])
    audit_candidate_group_ids = {
        str(record.get("node_group_id"))
        for record in audit.get("module_pruning_candidate_groups", [])
        if isinstance(record, dict) and record.get("node_group_id") is not None
    }
    audit_candidate_prefixes = {
        str(prefix)
        for prefix in audit.get("module_pruning_candidate_prefixes", [])
        if prefix
    }
    actual_candidate_group_ids = {
        str(group_id)
        for group_id in redundancy.get("actual_candidate_hit_group_ids", [])
        if group_id is not None
    }
    actual_hit_records = [
        record
        for record in redundancy.get("groups_with_redundancy", [])
        if isinstance(record, dict) and record.get("newly_opened_contract_hit")
    ]
    actual_hit_prefixes = {
        str(prefix)
        for record in actual_hit_records
        for prefix in record.get("matched_candidate_prefixes", [])
        if prefix
    }
    exact_directed_reclassification = (
        contract_matched
        and len(directed_families) == 1
        and bool(actual_candidate_group_ids)
        # node-* 是每次 ONNX trace 临时生成的序号，不能跨 audit/rebuild
        # 比较。跨进程身份使用稳定的候选模块前缀；组 ID 只在同一份 prune
        # metadata 内核对 hit 记录自洽性。
        and bool(actual_hit_prefixes)
        and actual_hit_prefixes <= audit_candidate_prefixes
        and {
            str(record.get("node_group_id")) for record in actual_hit_records
        }
        == actual_candidate_group_ids
        and all(
            str(record.get("node_group_id")) in actual_candidate_group_ids
            and bool(record.get("matched_candidate_prefixes"))
            for record in actual_hit_records
        )
    )
    if exact_directed_reclassification:
        directed_family = directed_families[0]
        normalized_counters = {
            "dependency_group_count": len(actual_candidate_group_ids),
            "redundant_channel_group_count": int(
                scalar(redundancy.get("actual_candidate_hit_redundant_channel_group_count")) or 0
            ),
        }
        if family_hits.get(directed_family) != normalized_counters:
            family_hits = {directed_family: normalized_counters}
            family_hit_classification = {
                "status": "exact_directed_probe_reclassified",
                "reason": (
                    "单一 directed family、audit/prune contract 一致，且所有物理命中组均为"
                    "稳定 audit 候选前缀并携带 matched_candidate_prefixes；临时 node-*"
                    " 仅作同一 metadata 内诊断。"
                ),
                "directed_family": directed_family,
                "audit_candidate_group_ids": sorted(audit_candidate_group_ids),
                "audit_candidate_prefixes": sorted(audit_candidate_prefixes),
                "actual_candidate_hit_group_ids": sorted(actual_candidate_group_ids),
                "actual_candidate_hit_prefixes": sorted(actual_hit_prefixes),
                "cross_process_identity_policy": "stable_candidate_prefix_not_trace_node_id",
                "raw_opened_module_family_hits": raw_family_hits,
                "normalized_opened_module_family_hits": family_hits,
            }
    required_families = resolved_required_families(args)
    missing_required_families = [family for family in required_families if family not in family_hits]
    family_gate_passed = args.allow_partial_family_hit or not missing_required_families

    compression_ratio = scalar(nested(prune_summary, "compression", "parameters_reduction_ratio")) or 0.0
    fixed_forward_status = nested(
        finetune_summary,
        "module_admission",
        "fixed_input_forward",
        "status",
    )
    backward_status = nested(
        finetune_summary,
        "module_admission",
        "task_loss_backward",
        "status",
    )
    optimizer_step_completed = bool(
        nested(
            finetune_summary,
            "module_admission",
            "task_loss_backward",
            "optimizer_step_completed",
            default=False,
        )
    )
    one_epoch_status = nested(
        finetune_summary,
        "module_admission",
        "one_epoch_finetune",
        "status",
    )
    best_reload_status = nested(
        finetune_summary,
        "module_admission",
        "best_checkpoint_independent_reload",
        "status",
    )
    checker_status = nested(export_metadata, "onnx_checker", "status")
    ort_status = nested(export_metadata, "onnxruntime_alignment", "status")
    attention_profile_modes = {
        "c2psa_attention_num_head_v1_experimental": "num_head",
        "c2psa_attention_head_dim_v1_experimental": "head_dim",
        ALL_VERIFIED_SAFE_PROFILE: "num_head",
        ALL_VERIFIED_SAFE_NUM_HEAD_V2_PROFILE: "num_head",
        ALL_VERIFIED_SAFE_HEAD_DIM_V2_PROFILE: "head_dim",
        ALL_VERIFIED_SAFE_NUM_HEAD_V4_OUTPUT_CLOSURE_PROFILE: "num_head",
    }
    expected_attention_mode = attention_profile_modes.get(args.module_coverage_profile)
    expected_attention_family = (
        "YOLO11_ATTN_NUM_HEAD"
        if expected_attention_mode == "num_head"
        else "YOLO11_ATTN_HEAD_DIM"
        if expected_attention_mode == "head_dim"
        else None
    )
    attention_hit_count = int(
        scalar(
            (family_hits.get(expected_attention_family, {}) or {}).get(
                "redundant_channel_group_count"
            )
        )
        or 0
    ) if expected_attention_family else 0
    attention_hit_required = args.module_coverage_profile in ATTENTION_ONLY_PROFILES
    attention_was_selected = attention_hit_count > 0
    attention_structure = prune_metadata.get("yolo11_attention_structure", {})
    attention_records = (
        attention_structure.get("records", [])
        if isinstance(attention_structure, dict)
        else []
    )
    c2psa_split_structure = prune_metadata.get(
        "yolo11_c2psa_paired_split_structure",
        {},
    )
    c2psa_split_structure_gate = (
        True
        if args.module_coverage_profile
        not in {"c2psa_paired_split_v1_experimental", ALL_CV1_OPEN_PROFILE}
        else c2psa_split_structure.get("status") == "pass"
        and bool(c2psa_split_structure.get("records", []))
    )
    c3k2_chunk_structure = prune_metadata.get(
        "yolo11_c3k2_paired_chunk_structure",
        {},
    )
    c3k2_chunk_structure_gate = (
        c3k2_chunk_structure.get("status") in {"pass", "not_applicable"}
        if args.module_coverage_profile in ALL_VERIFIED_SAFE_JOINT_COMPOSITE_PROFILES
        else (
            True
            if args.module_coverage_profile
            not in {"c3k2_paired_chunk_v1_experimental", ALL_CV1_OPEN_PROFILE}
            else c3k2_chunk_structure.get("status") == "pass"
            and bool(c3k2_chunk_structure.get("records", []))
        )
    )
    c3k2_dynamic_structure = prune_metadata.get(
        "yolo11_c3k2_dynamic_concat_structure",
        {},
    )
    c3k2_dynamic_structure_gate = (
        c3k2_dynamic_structure.get("status") in {"pass", "not_applicable"}
        if args.module_coverage_profile in ALL_VERIFIED_SAFE_JOINT_COMPOSITE_PROFILES
        else (
            True
            if args.module_coverage_profile != C3K2_DYNAMIC_CONCAT_PROFILE
            else c3k2_dynamic_structure.get("status") == "pass"
            and bool(c3k2_dynamic_structure.get("records", []))
        )
    )
    c3k2_composite_structure = prune_metadata.get(
        "yolo11_c3k2_composite_structure",
        {},
    )
    c3k2_composite_structure_gate = (
        c3k2_composite_structure.get("status") in {"pass", "not_applicable"}
        if args.module_coverage_profile in ALL_VERIFIED_SAFE_JOINT_COMPOSITE_PROFILES
        else (
            True
            if args.module_coverage_profile != C3K2_COMPOSITE_PROFILE
            else c3k2_composite_structure.get("status") == "pass"
            and bool(c3k2_composite_structure.get("records", []))
        )
    )
    output_consumer_closure_structure = prune_metadata.get(
        "yolo11_output_consumer_closure_structure",
        {},
    )
    output_consumer_closure_structure_gate = (
        True
        if args.module_coverage_profile not in OUTPUT_CONSUMER_CLOSURE_PROFILES
        else output_consumer_closure_structure.get("status") == "pass"
        and bool(output_consumer_closure_structure.get("records", []))
    )

    if expected_attention_mode is None:
        attention_structure_gate = True
    elif attention_was_selected or attention_hit_required:
        attention_structure_gate = (
            attention_structure.get("status") == "pass"
            and bool(attention_records)
            and all(
                record.get("prune_mode") == expected_attention_mode
                for record in attention_records
            )
        )
    else:
        attention_structure_gate = (
            attention_structure.get("status") in {"not_applicable", "pass"}
            and (
                not attention_records
                or all(
                    record.get("prune_mode") == expected_attention_mode
                    for record in attention_records
                )
            )
        )

    stage_results = [
        {
            "stage": 1,
            "name": "oto_graph",
            "status": "pass" if audit.get("status") == "audit_passed" and audit_contract else "fail",
            "evidence": str(paths["audit_json"]),
            "contract_sha256": audit_contract,
            "candidate_family_coverage": candidate_family_coverage,
        },
        {
            "stage": 2,
            "name": "single_pruning_period_real_zeroing",
            "status": (
                "pass"
                if hesso_pruned_groups > 0 and candidate_hit_groups > 0 and family_gate_passed
                else "fail"
            ),
            "hesso_pruned_groups": int(hesso_pruned_groups),
            "actual_candidate_hit_dependency_groups": int(candidate_hit_groups),
            "opened_module_family_hits": family_hits,
            "family_hit_classification": family_hit_classification,
            "required_hit_families": required_families,
            "missing_required_hit_families": missing_required_families,
            "allow_partial_family_hit": bool(args.allow_partial_family_hit),
            "evidence": str(paths["prune_metadata"]),
        },
        {
            "stage": 3,
            "name": "physical_subnet_build",
            "status": (
                "pass"
                if prune_metadata.get("status") in {
                    "structured_subnet_built",
                    "structured_subnet_rebuilt_and_forward_verified",
                }
                and paths["structured_weights"].is_file()
                and compression_ratio > 0
                and contract_matched
                and attention_structure_gate
                and c2psa_split_structure_gate
                and c3k2_chunk_structure_gate
                and c3k2_dynamic_structure_gate
                and c3k2_composite_structure_gate
                and output_consumer_closure_structure_gate
                else "fail"
            ),
            "contract_matched": contract_matched,
            "audit_contract_sha256": audit_contract,
            "prune_contract_sha256": prune_contract,
            "parameter_reduction_ratio": compression_ratio,
            "c2psa_paired_split_structure_gate": c2psa_split_structure_gate,
            "yolo11_c2psa_paired_split_structure": c2psa_split_structure,
            "c3k2_paired_chunk_structure_gate": c3k2_chunk_structure_gate,
            "yolo11_c3k2_paired_chunk_structure": c3k2_chunk_structure,
            "c3k2_dynamic_concat_structure_gate": c3k2_dynamic_structure_gate,
            "yolo11_c3k2_dynamic_concat_structure": c3k2_dynamic_structure,
            "c3k2_composite_structure_gate": c3k2_composite_structure_gate,
            "yolo11_c3k2_composite_structure": c3k2_composite_structure,
            "output_consumer_closure_structure_gate": output_consumer_closure_structure_gate,
            "yolo11_output_consumer_closure_structure": output_consumer_closure_structure,
            "attention_structure_gate": attention_structure_gate,
            "yolo11_attention_structure": attention_structure,
            "evidence": str(paths["prune_metadata"]),
        },
        {
            "stage": 4,
            "name": "frozen_manifest_input_forward",
            "status": "pass" if fixed_forward_status == "pass" else "fail",
            "evidence": str(paths["finetune_summary"]),
        },
        {
            "stage": 5,
            "name": "task_loss_backward",
            "status": "pass" if backward_status == "pass" and optimizer_step_completed else "fail",
            "optimizer_step_completed": optimizer_step_completed,
            "evidence": str(paths["finetune_summary"]),
        },
        {
            "stage": 6,
            "name": "one_epoch_finetune",
            "status": "pass" if one_epoch_status == "pass" else "fail",
            "evidence": str(paths["finetune_summary"]),
        },
        {
            "stage": 7,
            "name": "best_checkpoint_independent_reload",
            "status": "pass" if best_reload_status == "pass" else "fail",
            "evidence": str(paths["finetune_summary"]),
        },
        {
            "stage": 8,
            "name": "onnx_export_and_checker",
            "status": (
                "pass"
                if paths["onnx"].is_file() and checker_status == "pass"
                else "fail"
            ),
            "evidence": str(paths["export_metadata"]),
        },
        {
            "stage": 9,
            "name": "onnxruntime_same_input_alignment",
            "status": "pass" if ort_status == "pass" else "fail",
            "evidence": str(paths["export_metadata"]),
        },
    ]
    all_passed = all(result["status"] == "pass" for result in stage_results)
    return {
        "status": "pass" if all_passed else "fail",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "profile": args.module_coverage_profile,
        "probe_module_families": list(args.probe_module_family or []),
        "sparse_training_provenance": (
            {
                "mode": "reused_existing_full_group_sparse_model",
                "source": str(args.rebuild_from_full_sparse),
                "sparse_training_not_repeated": True,
            }
            if args.rebuild_from_full_sparse
            else {"mode": "executed_in_current_run", "sparse_training_not_repeated": False}
        ),
        "contract_rule": "audit_and_sparse_training_must_share_exact_contract_sha256",
        "contract_sha256": audit_contract if contract_matched else None,
        "candidate_family_coverage_after_protection": candidate_family_coverage,
        "family_hit_classification": family_hit_classification,
        "stage_results": stage_results,
        "paths": {name: str(path) for name, path in paths.items()},
        "note": (
            "九项必须全部通过。定向 probe 还要求目标模块族真实物理命中；"
            "联合全局 profile 只要求所有已验证模块族在 audit 后仍有可剪候选，"
            "HESSO 可按全局重要性只选择其中一部分，实际命中情况完整记录但不强制每族至少一组。"
        ),
    }


def enforce_post_prune_gate(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    """进入微调/导出前验证全局物理命中、定向命中合同和子网结构。"""

    audit = read_json(paths["audit_json"])
    prune_metadata = read_json(paths["prune_metadata"])
    prune_summary = read_json(paths["prune_summary"])
    audit_contract = nested(audit, "module_dependency_control", "contract", "sha256")
    prune_contract = nested(
        prune_metadata,
        "module_dependency_control",
        "contract",
        "sha256",
    )
    redundancy = prune_metadata.get("physical_redundancy", {})
    if not isinstance(redundancy, dict):
        redundancy = {}
    family_hits = redundancy.get("opened_module_family_hits", {})
    if not isinstance(family_hits, dict):
        family_hits = {}
    required_families = resolved_required_families(args)
    missing_families = [family for family in required_families if family not in family_hits]
    hesso_pruned_groups = int(
        scalar(nested(prune_summary, "hesso", "num_pruned_groups")) or 0
    )
    candidate_hit_groups = int(
        scalar(redundancy.get("actual_candidate_hit_dependency_group_count")) or 0
    )
    issues: list[str] = []
    if not audit_contract or audit_contract != prune_contract:
        issues.append(
            f"audit/prune contract 不一致：audit={audit_contract}, prune={prune_contract}"
        )
    if prune_metadata.get("status") not in {
        "structured_subnet_built",
        "structured_subnet_rebuilt_and_forward_verified",
    }:
        issues.append(f"物理子网状态异常：{prune_metadata.get('status')}")
    if not paths["structured_weights"].is_file():
        issues.append(f"物理子网 checkpoint 不存在：{paths['structured_weights']}")
    if hesso_pruned_groups <= 0 or candidate_hit_groups <= 0:
        issues.append(
            "没有真实候选物理命中："
            f"hesso_pruned_groups={hesso_pruned_groups}, candidate_hit_groups={candidate_hit_groups}"
        )
    if missing_families and not args.allow_partial_family_hit:
        issues.append(
            f"必需模块族未被物理命中：{missing_families}；actual={sorted(family_hits)}"
        )
    attention_profiles = {
        "c2psa_attention_num_head_v1_experimental": (
            "num_head",
            "YOLO11_ATTN_NUM_HEAD",
        ),
        "c2psa_attention_head_dim_v1_experimental": (
            "head_dim",
            "YOLO11_ATTN_HEAD_DIM",
        ),
        ALL_VERIFIED_SAFE_PROFILE: (
            "num_head",
            "YOLO11_ATTN_NUM_HEAD",
        ),
        ALL_VERIFIED_SAFE_NUM_HEAD_V2_PROFILE: (
            "num_head",
            "YOLO11_ATTN_NUM_HEAD",
        ),
        ALL_VERIFIED_SAFE_HEAD_DIM_V2_PROFILE: (
            "head_dim",
            "YOLO11_ATTN_HEAD_DIM",
        ),
    }
    c2psa_split_evidence: dict[str, Any] = {}
    if args.module_coverage_profile in {
        "c2psa_paired_split_v1_experimental",
        ALL_CV1_OPEN_PROFILE,
    }:
        c2psa_split_structure = prune_metadata.get(
            "yolo11_c2psa_paired_split_structure",
            {},
        )
        c2psa_split_records = (
            c2psa_split_structure.get("records", [])
            if isinstance(c2psa_split_structure, dict)
            else []
        )
        if c2psa_split_structure.get("status") != "pass" or not c2psa_split_records:
            issues.append(
                "C2PSA paired-split 物理结构合同未通过："
                f"{c2psa_split_structure}"
            )
        c2psa_split_evidence = {
            "expected_contract": "yolo11_c2psa_paired_split_v1",
            "physical_structure": c2psa_split_structure,
        }

    c3k2_chunk_evidence: dict[str, Any] = {}
    if args.module_coverage_profile in {
        "c3k2_paired_chunk_v1_experimental",
        ALL_CV1_OPEN_PROFILE,
        *ALL_VERIFIED_SAFE_JOINT_COMPOSITE_PROFILES,
    }:
        c3k2_chunk_structure = prune_metadata.get(
            "yolo11_c3k2_paired_chunk_structure",
            {},
        )
        c3k2_chunk_records = (
            c3k2_chunk_structure.get("records", [])
            if isinstance(c3k2_chunk_structure, dict)
            else []
        )
        if (
            args.module_coverage_profile not in ALL_VERIFIED_SAFE_JOINT_COMPOSITE_PROFILES
            and (
                c3k2_chunk_structure.get("status") != "pass"
                or not c3k2_chunk_records
            )
        ):
            issues.append(
                "C3k2 paired-chunk 物理结构合同未通过："
                f"{c3k2_chunk_structure}"
            )
        original_contracts = nested(
            audit,
            "module_dependency_control",
            "yolo11_c3k2_paired_chunk_contracts",
            default=[],
        ) or []
        original_prefixes = {
            str(record.get("module_prefix"))
            for record in original_contracts
            if isinstance(record, dict) and record.get("module_prefix")
        }
        physical_prefixes = {
            str(record.get("module_prefix"))
            for record in c3k2_chunk_records
            if isinstance(record, dict) and record.get("module_prefix")
        }
        unknown_physical_prefixes = sorted(physical_prefixes - original_prefixes)
        if unknown_physical_prefixes:
            issues.append(
                "C3k2 paired-chunk 物理记录不属于 audit 合同："
                f"{unknown_physical_prefixes}"
            )
        c3k2_chunk_evidence = {
            "expected_contract": "yolo11_c3k2_paired_chunk_v1",
            "configured_module_count": len(original_contracts),
            "physically_selected_module_count": len(c3k2_chunk_records),
            "unknown_physical_prefixes": unknown_physical_prefixes,
            "original_contracts": original_contracts,
            "physical_structure": c3k2_chunk_structure,
        }

    c3k2_dynamic_evidence: dict[str, Any] = {}
    if args.module_coverage_profile == C3K2_DYNAMIC_CONCAT_PROFILE:
        dynamic_structure = prune_metadata.get(
            "yolo11_c3k2_dynamic_concat_structure",
            {},
        )
        dynamic_records = (
            dynamic_structure.get("records", [])
            if isinstance(dynamic_structure, dict)
            else []
        )
        if dynamic_structure.get("status") != "pass" or not dynamic_records:
            issues.append(
                "C3k2 segmented dynamic-concat 物理结构合同未通过："
                f"{dynamic_structure}"
            )
        original_contracts = nested(
            audit,
            "module_dependency_control",
            "yolo11_c3k2_dynamic_concat_contracts",
            default=[],
        ) or []
        original_child_prefixes = {
            str(record.get("child_prefix"))
            for record in original_contracts
            if isinstance(record, dict) and record.get("child_prefix")
        }
        selected_child_prefixes = {
            str(prefix)
            for prefix in dynamic_structure.get("selected_child_prefixes", [])
            if prefix
        }
        if not selected_child_prefixes:
            selected_child_prefixes = {
                str(child.get("child_prefix"))
                for record in dynamic_records
                if isinstance(record, dict)
                for child in record.get("children", [])
                if isinstance(child, dict)
                and child.get("selected")
                and child.get("child_prefix")
            }
        unknown_selected_prefixes = sorted(
            selected_child_prefixes - original_child_prefixes
        )
        missing_selected_prefixes = sorted(
            original_child_prefixes - selected_child_prefixes
        )
        if unknown_selected_prefixes:
            issues.append(
                "C3k2 dynamic-concat 物理记录不属于 audit 合同："
                f"{unknown_selected_prefixes}"
            )
        if missing_selected_prefixes:
            issues.append(
                "C3k2 dynamic-concat 定向准入要求 3 个独立 child-output 全部真实命中；"
                f"未命中={missing_selected_prefixes}"
            )
        c3k2_dynamic_evidence = {
            "expected_contract": "yolo11_c3k2_segmented_dynamic_concat_v1",
            "configured_child_count": len(original_contracts),
            "physically_selected_child_count": len(selected_child_prefixes),
            "selected_child_prefixes": sorted(selected_child_prefixes),
            "missing_selected_prefixes": missing_selected_prefixes,
            "unknown_selected_prefixes": unknown_selected_prefixes,
            "all_configured_children_selected": not missing_selected_prefixes,
            "original_contracts": original_contracts,
            "physical_structure": dynamic_structure,
        }

    c3k2_composite_evidence: dict[str, Any] = {}
    if args.module_coverage_profile in {
        C3K2_COMPOSITE_PROFILE,
        *ALL_VERIFIED_SAFE_JOINT_COMPOSITE_PROFILES,
    }:
        composite_structure = prune_metadata.get(
            "yolo11_c3k2_composite_structure",
            {},
        )
        composite_records = (
            composite_structure.get("records", [])
            if isinstance(composite_structure, dict)
            else []
        )
        original_contracts = nested(
            audit,
            "module_dependency_control",
            "yolo11_c3k2_composite_contracts",
            default=[],
        ) or []
        selected_child_prefixes = {
            str(prefix)
            for prefix in composite_structure.get("selected_child_prefixes", [])
            if prefix
        }
        root_hit = any(
            int(record.get("root_removed_group_count", 0) or 0) > 0
            for record in composite_records
            if isinstance(record, dict)
        )
        if composite_structure.get("status") not in {"pass", "not_applicable"}:
            issues.append(
                "C3k2 composite 物理结构合同未通过："
                f"{composite_structure}"
            )
        if (
            args.module_coverage_profile == C3K2_COMPOSITE_PROFILE
            and not root_hit
        ):
            issues.append("C3k2 composite 未产生根部 paired-chunk 的真实物理收缩。")
        if (
            args.module_coverage_profile == C3K2_COMPOSITE_PROFILE
            and not selected_child_prefixes
        ):
            issues.append(
                "C3k2 composite 未产生独立 child-output 的真实物理收缩。"
            )
        malformed_contracts = [
            contract
            for contract in original_contracts
            if not isinstance(contract, dict)
            or contract.get("contract")
            != "yolo11_c3k2_paired_chunk_dynamic_concat_v1"
            or contract.get("ownership_overlap")
        ]
        if malformed_contracts:
            issues.append(
                "C3k2 composite audit 合同缺少互斥所有权证据："
                f"{malformed_contracts}"
            )
        c3k2_composite_evidence = {
            "expected_contract": "yolo11_c3k2_paired_chunk_dynamic_concat_v1",
            "configured_parent_count": len(original_contracts),
            "joint_global_selection_not_required": (
                args.module_coverage_profile in ALL_VERIFIED_SAFE_JOINT_COMPOSITE_PROFILES
            ),
            "root_axis_physically_selected": root_hit,
            "physically_selected_dynamic_child_count": len(selected_child_prefixes),
            "selected_dynamic_child_prefixes": sorted(selected_child_prefixes),
            "audit_contracts": original_contracts,
            "physical_structure": composite_structure,
        }

    attention_evidence: dict[str, Any] = {}
    if args.module_coverage_profile in attention_profiles:
        expected_mode, expected_family = attention_profiles[args.module_coverage_profile]
        attention_hit_required = args.module_coverage_profile in ATTENTION_ONLY_PROFILES
        attention_structure = prune_metadata.get("yolo11_attention_structure", {})
        attention_records = (
            attention_structure.get("records", [])
            if isinstance(attention_structure, dict)
            else []
        )
        original_contracts = nested(
            audit,
            "module_dependency_control",
            "yolo11_attention_pruning_contracts",
            default=[],
        ) or []
        hit_count = int(
            scalar(
                (family_hits.get(expected_family, {}) or {}).get(
                    "redundant_channel_group_count"
                )
            )
            or 0
        )
        attention_was_selected = hit_count > 0
        if attention_hit_required and not attention_was_selected:
            issues.append(
                f"Attention-only probe 未产生真实物理命中：{expected_family}"
            )
        if attention_was_selected or attention_hit_required:
            if attention_structure.get("status") != "pass" or not attention_records:
                issues.append(f"Attention 物理结构合同未通过：{attention_structure}")
            if len(original_contracts) != len(attention_records):
                issues.append(
                    "Attention 原始/物理模块数量不一致："
                    f"{len(original_contracts)} != {len(attention_records)}"
                )
            for original, physical in zip(original_contracts, attention_records):
                prune_mode = physical.get("prune_mode")
                if prune_mode != expected_mode:
                    issues.append(
                        f"Attention prune_mode 错误：{prune_mode} != {expected_mode}"
                    )
                expected_heads = int(original.get("num_heads", 0))
                expected_head_dim = int(original.get("head_dim", 0))
                if expected_mode == "num_head":
                    expected_heads -= hit_count
                else:
                    expected_head_dim -= hit_count
                if (
                    int(physical.get("num_heads", -1)) != expected_heads
                    or int(physical.get("head_dim", -1)) != expected_head_dim
                    or int(physical.get("key_dim", -1)) != int(original.get("key_dim", -2))
                ):
                    issues.append(
                        "Attention 物理属性与真实命中数不一致："
                        f"original={original}, physical={physical}, hit_count={hit_count}"
                    )
        elif attention_structure.get("status") not in {"not_applicable", "pass"}:
            issues.append(
                "联合全局剪枝未选择 Attention，但未剪 Attention 的结构状态异常："
                f"{attention_structure}"
            )
        attention_evidence = {
            "expected_mode": expected_mode,
            "expected_family": expected_family,
            "attention_hit_required": attention_hit_required,
            "attention_was_selected": attention_was_selected,
            "redundant_logical_group_count": hit_count,
            "original_contracts": original_contracts,
            "physical_structure": attention_structure,
        }
    evidence = {
        "status": "pass" if not issues else "fail",
        "audit_contract_sha256": audit_contract,
        "prune_contract_sha256": prune_contract,
        "hesso_pruned_groups": hesso_pruned_groups,
        "actual_candidate_hit_dependency_groups": candidate_hit_groups,
        "opened_module_family_hits": family_hits,
        "required_hit_families": required_families,
        "missing_required_hit_families": missing_families,
        "c2psa_paired_split_structure_gate": c2psa_split_evidence,
        "c3k2_paired_chunk_structure_gate": c3k2_chunk_evidence,
        "c3k2_dynamic_concat_structure_gate": c3k2_dynamic_evidence,
        "c3k2_composite_structure_gate": c3k2_composite_evidence,
        "attention_structure_gate": attention_evidence,
        "issues": issues,
    }
    gate_path = paths["prune_metadata"].parent / "post_prune_gate.json"
    write_json(gate_path, evidence)
    attention_text = "不适用"
    if attention_evidence:
        attention_text = (
            f"{attention_evidence.get('expected_mode')}，"
            f"{'已剪' if attention_evidence.get('attention_was_selected') else '未被本轮选中'}，"
            f"冗余逻辑组 {attention_evidence.get('redundant_logical_group_count', 0)}"
        )
    print_human_summary(
        "剪枝后早停门禁",
        [
            ("状态", "通过" if not issues else "失败"),
            ("HESSO 实际剪枝组", hesso_pruned_groups),
            ("候选依赖组命中", candidate_hit_groups),
            ("实际命中模块", compact_family_hits(family_hits)),
            (
                "定向必需模块",
                "无（全局重要性选择）" if not required_families else "，".join(required_families),
            ),
            ("未命中必需模块", "无" if not missing_families else "，".join(missing_families)),
            (
                "C2PSA split",
                "通过" if c2psa_split_evidence and not any(
                    "C2PSA paired-split" in issue for issue in issues
                ) else "不适用" if not c2psa_split_evidence else "失败",
            ),
            (
                "C3k2 chunk",
                "通过" if c3k2_chunk_evidence and not any(
                    "C3k2 paired-chunk" in issue for issue in issues
                ) else "不适用" if not c3k2_chunk_evidence else "失败",
            ),
            (
                "C3k2 dynamic concat",
                "通过" if c3k2_dynamic_evidence and not any(
                    "dynamic-concat" in issue for issue in issues
                ) else "不适用" if not c3k2_dynamic_evidence else "失败",
            ),
            ("Attention", attention_text),
            ("问题", "无" if not issues else f"{len(issues)} 项"),
        ],
        artifact=gate_path,
    )
    if issues:
        for index, issue in enumerate(issues, start=1):
            print(f"  {index}. {issue}", flush=True)
        raise RuntimeError(
            f"剪枝后早停门禁失败，共 {len(issues)} 项；详见：{gate_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO11 dense closure 九段准入编排")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--fixed-input", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--rebuild-from-full-sparse",
        help=(
            "恢复模式：stage 2/3 不重复稀疏训练，改为从既有 "
            "DetectionModel_full_group_sparse.pt 重建物理子网；其余微调、重载、"
            "ONNX 与 ORT 门槛不变。"
        ),
    )
    parser.add_argument(
        "--prune-run-dir",
        help=(
            "恢复微调/导出时显式指定已完成剪枝的实际运行目录（可为 Ultralytics 自动追加 "
            "数字后缀的 *_prune2 / *_prune5）。该目录必须包含 prune_summary.json、"
            "对应 metadata 和结构化权重。"
        ),
    )
    parser.add_argument(
        "--module-coverage-profile",
        choices=tuple(PROBE_CONFIGS),
        default=DEFAULT_PROFILE,
        help="选择累计 coverage profile；默认保持已验证的 CSP dense profile。",
    )
    parser.add_argument("--output", default=str(BASE_DIR / "runs"))
    parser.add_argument("--logs", default=str(BASE_DIR / "logs"))
    parser.add_argument(
        "--report-output",
        help=(
            "可选的 admission JSON 输出路径；用于基于不可变历史产物生成修正报告，"
            "默认仍写 runs/<run-id>/nine_stage_admission.json。"
        ),
    )
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--export-device",
        default="cpu",
        help="ONNX 导出与 PyTorch/ORT 对齐设备；默认 cpu，避免 Ultralytics 安装不匹配的 onnxruntime-gpu。",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prune-ratio", type=float, default=0.10)
    parser.add_argument("--group-divisible", type=int, default=1)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument(
        "--prune-epochs",
        type=int,
        default=1,
        help="稀疏训练总 epoch 数；默认 1 保持既有 L3 准入行为。",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=0.0,
        help="HESSO 开始物理置零前的预热 epoch 数。",
    )
    parser.add_argument(
        "--pruning-periods",
        type=int,
        default=1,
        help="HESSO 稀疏调度周期数；默认 1 保持既有 L3 准入行为。",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=1,
        help="物理子网微调 epoch 数；首个 epoch 始终执行准入门禁。",
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=0,
        help="微调阶段关闭 mosaic 的尾部 epoch 数。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oto-opset", type=int, default=16)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--alignment-atol", type=float, default=1e-4)
    parser.add_argument("--alignment-rtol", type=float, default=1e-3)
    parser.add_argument("--start-stage", choices=STAGES, default="audit")
    parser.add_argument("--stop-after", choices=STAGES, default="report")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--required-hit-family", action="append")
    parser.add_argument(
        "--probe-module-family",
        action="append",
        choices=(
            "C3k2",
            "C3k",
            "C3",
            "Bottleneck",
            "SPPF",
            "Detect",
            "C2PSA",
            "C2PSA_OUTPUT",
            "C2PSA_SPLIT",
            "C3K2_CHUNK",
            "C3K2_DYNAMIC_CONCAT",
            "C3K2_OUTPUT_CLOSURE",
            "PSABlock",
            "PSA_FFN",
            "YOLO11_ATTN_NUM_HEAD",
            "YOLO11_ATTN_HEAD_DIM",
            "standalone_or_boundary_Conv2d",
        ),
        help="定向限制 HESSO 候选模块族；可重复传入。未指定时测试完整 dense profile。",
    )
    parser.add_argument(
        "--allow-partial-family-hit",
        action="store_true",
        help="只用于诊断；允许 profile 有真实命中但部分模块族未命中。正式验收不要使用。",
    )
    args = parser.parse_args()
    for name in ("weights", "data", "fixed_input"):
        path = Path(getattr(args, name)).expanduser()
        if not path.is_file():
            parser.error(f"--{name.replace('_', '-')} 不存在：{path}")
        setattr(args, name, str(path.resolve()))
    if args.rebuild_from_full_sparse:
        full_sparse = Path(args.rebuild_from_full_sparse).expanduser()
        if not full_sparse.is_file():
            parser.error(f"--rebuild-from-full-sparse 不存在：{full_sparse}")
        args.rebuild_from_full_sparse = str(full_sparse.resolve())
        if args.start_stage != "audit":
            parser.error(
                "--rebuild-from-full-sparse 必须从 --start-stage audit 开始；"
                "audit 会用当前代码重新生成依赖合同，禁止复用修改前的旧 audit.json。"
            )
    if args.prune_run_dir:
        prune_run_dir = Path(args.prune_run_dir).expanduser()
        if not prune_run_dir.is_dir():
            parser.error(f"--prune-run-dir 不存在或不是目录：{prune_run_dir}")
        args.prune_run_dir = str(prune_run_dir.resolve())
        if args.start_stage in {"audit", "prune"}:
            parser.error("--prune-run-dir 仅用于从 finetune、export 或 report 恢复。")
    if STAGES.index(args.start_stage) > STAGES.index(args.stop_after):
        parser.error("--start-stage 不能晚于 --stop-after")
    if args.prune_epochs < 1:
        parser.error("--prune-epochs 必须 >= 1")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.prune_epochs:
        parser.error("--warmup-epochs 必须满足 0 <= warmup < prune-epochs")
    if args.pruning_periods < 1:
        parser.error("--pruning-periods 必须 >= 1")
    if args.finetune_epochs < 1:
        parser.error("--finetune-epochs 必须 >= 1")
    if args.close_mosaic < 0 or args.close_mosaic > args.finetune_epochs:
        parser.error("--close-mosaic 必须满足 0 <= close-mosaic <= finetune-epochs")
    if STAGES.index(args.start_stage) > STAGES.index("prune") and not args.prune_run_dir:
        parser.error(
            "从 finetune/export/report 恢复必须显式提供 --prune-run-dir，"
            "避免将自动递增目录（例如 *_prune5）误解析为静态 *_prune。"
        )
    return args


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    logs = Path(args.logs).expanduser().resolve()
    probe_config = PROBE_CONFIGS[args.module_coverage_profile]
    probe_script = Path(probe_config["script"])
    structured_name = str(probe_config["structured_name"])
    audit_name = f"{args.run_id}_audit"
    prune_name = f"{args.run_id}_prune"
    finetune_name = f"{args.run_id}_finetune1"
    audit_dir = output / audit_name
    prune_dir = (
        Path(args.prune_run_dir).expanduser().resolve()
        if args.prune_run_dir
        else output / prune_name
    )
    finetune_dir = output / finetune_name
    structured_weights = prune_dir / "weights" / structured_name
    finetune_summary_path = finetune_dir / "finetune_summary.json"
    onnx_path = finetune_dir / f"{args.run_id}.onnx"
    admission_report_path = (
        Path(args.report_output).expanduser().resolve()
        if args.report_output
        else output / args.run_id / "nine_stage_admission.json"
    )
    paths = {
        "audit_json": audit_dir / "audit.json",
        "prune_metadata": prune_dir / (
            "rebuild_structured_subnet.json"
            if args.rebuild_from_full_sparse
            else "adamcf_run_metadata.json"
        ),
        "prune_summary": prune_dir / "prune_summary.json",
        "structured_weights": structured_weights,
        "finetune_summary": finetune_summary_path,
        "onnx": onnx_path,
        "export_metadata": onnx_path.with_suffix(onnx_path.suffix + ".metadata.json"),
        "admission_report": admission_report_path,
    }
    update_prune_paths(
        paths,
        prune_dir=prune_dir,
        structured_name=structured_name,
        rebuild_from_full_sparse=bool(args.rebuild_from_full_sparse),
    )
    common = [
        "--imgsz", str(args.imgsz),
        "--device", args.device,
        "--output", str(output),
    ]
    train_common = [
        "--data", args.data,
        "--batch", str(args.batch),
        "--workers", str(args.workers),
        "--lr0", str(args.lr0),
        "--seed", str(args.seed),
        "--no-amp",
    ]
    if args.exist_ok:
        common.append("--exist-ok")

    commands = {
        "audit": [
            sys.executable, "-u", str(probe_script), "audit",
            "--weights", args.weights,
            "--oto-opset", str(args.oto_opset),
            *common,
            "--name", audit_name,
        ],
        "prune": [
            sys.executable, "-u", str(probe_script), "prune",
            "--weights", args.weights,
            *train_common,
            "--epochs", str(args.prune_epochs),
            "--warmup-epochs", str(args.warmup_epochs),
            "--pruning-periods", str(args.pruning_periods),
            "--prune-ratio", str(args.prune_ratio),
            "--group-divisible", str(args.group_divisible),
            "--oto-opset", str(args.oto_opset),
            *common,
            "--name", prune_name,
        ],
        "finetune": [
            sys.executable, "-u", str(CORE_SCRIPT), "finetune",
            "--weights", str(structured_weights),
            "--baseline-weights", args.weights,
            *train_common,
            "--epochs", str(args.finetune_epochs),
            "--close-mosaic", str(args.close_mosaic),
            "--fixed-input", args.fixed_input,
            "--admission-probe",
            *common,
            "--name", finetune_name,
        ],
        "export": [],
    }
    if args.rebuild_from_full_sparse:
        commands["prune"] = [
            sys.executable, "-u", str(probe_script), "rebuild-subnet",
            "--weights", args.rebuild_from_full_sparse,
            "--oto-opset", str(args.oto_opset),
            # recovery 不会重新训练，但仍必须显式声明源训练搜索空间，供核心
            # provenance 门禁核对，不能悄悄使用 parser 默认值。
            "--prune-ratio", str(args.prune_ratio),
            "--group-divisible", str(args.group_divisible),
            "--seed", str(args.seed),
            *common,
            "--name", prune_name,
        ]
    family_args = [
        token
        for family in (args.probe_module_family or [])
        for token in ("--probe-module-family", family)
    ]
    # v2 使用同一个轻量入口承载两种 Attention 轴，必须由编排器把当前 profile
    # 显式传给 probe；其余历史入口仍自行固定 profile，保持旧命令不变。
    profile_args = (
        ["--module-coverage-profile", args.module_coverage_profile]
        if args.module_coverage_profile in ALL_VERIFIED_SAFE_JOINT_COMPOSITE_PROFILES
        else []
    )
    commands["audit"].extend(profile_args)
    commands["prune"].extend(profile_args)
    commands["audit"].extend(family_args)
    commands["prune"].extend(family_args)

    start_index = STAGES.index(args.start_stage)
    stop_index = STAGES.index(args.stop_after)
    try:
        if start_index > STAGES.index("prune"):
            # 恢复模式不重跑 HESSO，但必须先以显式实际 prune dir 执行同一物理门禁；
            # 否则微调可能引用一个存在却不属于当前 audit 合同的结构化子网。
            enforce_post_prune_gate(args, paths)
        for stage in ("audit", "prune", "finetune", "export"):
            stage_index = STAGES.index(stage)
            if stage_index < start_index or stage_index > stop_index:
                continue
            if stage == "export":
                finetune_summary = read_json(paths["finetune_summary"])
                final_weights_value = finetune_summary.get("final_weights")
                if not final_weights_value:
                    raise FileNotFoundError(f"微调摘要缺少 final_weights：{paths['finetune_summary']}")
                commands["export"] = [
                    sys.executable, "-u", str(CORE_SCRIPT), "export",
                    "--weights", str(final_weights_value),
                    "--imgsz", str(args.imgsz),
                    "--device", args.export_device,
                    "--opset", str(args.opset),
                    "--export-output", str(paths["onnx"]),
                    "--fixed-input", args.fixed_input,
                    "--onnx-checker",
                    "--onnxruntime-align",
                    "--alignment-atol", str(args.alignment_atol),
                    "--alignment-rtol", str(args.alignment_rtol),
                ]
            stage_started_at = run_stage(stage, commands[stage], logs / f"{args.run_id}_{stage}.log")
            if stage == "prune":
                actual_prune_dir = resolve_actual_stage_run_dir(
                    output=output,
                    requested_name=prune_name,
                    completion_artifact=(
                        "rebuild_structured_subnet.json"
                        if args.rebuild_from_full_sparse
                        else "adamcf_run_metadata.json"
                    ),
                    stage_started_at=stage_started_at,
                )
                update_prune_paths(
                    paths,
                    prune_dir=actual_prune_dir,
                    structured_name=structured_name,
                    rebuild_from_full_sparse=bool(args.rebuild_from_full_sparse),
                )
                replace_option_value(commands["finetune"], "--weights", str(paths["structured_weights"]))
                # 即使本轮以 --stop-after prune 结束，也必须执行物理子网早停门禁；
                # 否则只能证明命令运行完，不能证明全部必需模块族真实命中且结构安全。
                enforce_post_prune_gate(args, paths)
            elif stage == "finetune":
                actual_finetune_dir = resolve_actual_stage_run_dir(
                    output=output,
                    requested_name=finetune_name,
                    completion_artifact="finetune_summary.json",
                    stage_started_at=stage_started_at,
                )
                update_finetune_paths(paths, finetune_dir=actual_finetune_dir, run_id=args.run_id)
    except Exception as exc:
        failure = {
            "status": "execution_failed",
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": args.run_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "paths": {name: str(path) for name, path in paths.items()},
        }
        write_json(paths["admission_report"], failure)
        print_human_summary(
            "准入流程失败",
            [
                ("运行编号", args.run_id),
                ("错误类型", type(exc).__name__),
                ("原因", str(exc)),
                ("调试方式", "设置 OTO_DEBUG_TRACEBACK=1 可显示完整 traceback"),
            ],
            artifact=paths["admission_report"],
        )
        if os.environ.get("OTO_DEBUG_TRACEBACK") == "1":
            raise
        raise SystemExit(1)

    if stop_index >= STAGES.index("report") or args.start_stage == "report":
        report = build_admission_report(args, paths)
        write_json(paths["admission_report"], report)
        stage_results = report.get("stage_results", [])
        passed = sum(
            1
            for stage in stage_results
            if isinstance(stage, dict) and stage.get("status") == "pass"
        )
        failed_names = [
            str(stage.get("name"))
            for stage in stage_results
            if isinstance(stage, dict) and stage.get("status") != "pass"
        ]
        print_human_summary(
            "九阶段准入结果",
            [
                ("状态", "通过" if report.get("status") == "pass" else "失败"),
                ("通过阶段", f"{passed}/{len(stage_results)}"),
                ("失败阶段", "无" if not failed_names else "，".join(failed_names)),
                ("运行配置", args.module_coverage_profile),
                ("结构化模型", paths["structured_weights"]),
                ("ONNX", paths["onnx"] if paths["onnx"].is_file() else "未生成"),
            ],
            artifact=paths["admission_report"],
        )
        if report["status"] != "pass":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
