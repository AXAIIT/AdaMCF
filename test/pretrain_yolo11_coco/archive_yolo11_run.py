#!/usr/bin/env python
"""将远程同步回本机的 YOLO11 run 归档为项目六的轻量、可追溯证据包。

默认迁入 JSON/YAML/CSV/文本日志和 checkpoint 清单（名称、大小、SHA256），不会
复制大体积权重。若某个 run 已被选为 06G selected run，显式传入
``--include-checkpoints`` 才会把 checkpoint 实体一并归档。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


EVIDENCE_SUFFIXES = {".json", ".jsonl", ".yaml", ".yml", ".csv", ".txt", ".log"}
CHECKPOINT_SUFFIXES = {".pt", ".onnx", ".engine"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_evidence(
    source: Path,
    destination: Path,
    *,
    stage_name: str | None,
) -> list[dict[str, str | int]]:
    copied: list[dict[str, str | int]] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EVIDENCE_SUFFIXES:
            continue
        relative = path.relative_to(source)
        target = destination / "outputs"
        if stage_name:
            target = target / stage_name
        target = target / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(
            {
                "source": str(path),
                "archived": str(target),
                "size_bytes": path.stat().st_size,
            }
        )
    return copied


def checkpoint_manifest(
    source: Path,
    destination: Path,
    include_checkpoints: bool,
    *,
    stage_name: str | None,
) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in CHECKPOINT_SUFFIXES:
            continue
        relative = path.relative_to(source)
        record: dict[str, str | int] = {
            "source": str(path),
            "source_stage": stage_name or source.name,
            "relative_path": str(relative).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "copied": include_checkpoints,
        }
        if include_checkpoints:
            target = destination / "checkpoints"
            if stage_name:
                target = target / stage_name
            target = target / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            record["archived"] = str(target)
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        required=True,
        action="append",
        help="已同步到本机的 runs/<run_id> 阶段目录；完整九段 run 可重复传入 audit/prune/finetune/report 目录。",
    )
    parser.add_argument("--destination", required=True, help="项目六 runs/<run_id> 目标目录")
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        help="同一 run 的外部日志；可重复传入。",
    )
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="显式复制 checkpoint 实体；默认仅保存清单和 SHA256。",
    )
    args = parser.parse_args()

    sources = [Path(value).expanduser().resolve() for value in args.source_run]
    destination = Path(args.destination).expanduser().resolve()
    invalid_sources = [str(source) for source in sources if not source.is_dir()]
    if invalid_sources:
        parser.error(f"--source-run 不存在或不是目录：{invalid_sources}")
    if len({str(source) for source in sources}) != len(sources):
        parser.error("同一 --source-run 不能重复传入。")
    if destination.exists():
        parser.error(f"目标目录已存在，拒绝覆盖：{destination}")

    destination.mkdir(parents=True)
    multiple_sources = len(sources) > 1
    copied_evidence: list[dict[str, str | int]] = []
    checkpoints: list[dict[str, str | int]] = []
    for source in sources:
        stage_name = source.name if multiple_sources else None
        copied_evidence.extend(
            copy_evidence(source, destination, stage_name=stage_name)
        )
        checkpoints.extend(
            checkpoint_manifest(
                source,
                destination,
                args.include_checkpoints,
                stage_name=stage_name,
            )
        )
    copied_logs: list[dict[str, str | int]] = []
    for raw_path in args.log:
        log_path = Path(raw_path).expanduser().resolve()
        if not log_path.is_file():
            parser.error(f"--log 不存在：{log_path}")
        target = destination / "logs" / log_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log_path, target)
        copied_logs.append(
            {
                "source": str(log_path),
                "archived": str(target),
                "size_bytes": log_path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": "project6_yolo11_run_archive_v2",
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_runs": [str(source) for source in sources],
        "destination": str(destination),
        "checkpoint_payloads_included": bool(args.include_checkpoints),
        "evidence_files": copied_evidence,
        "external_logs": copied_logs,
        "checkpoint_manifest": checkpoints,
        "missing_items_policy": (
            "仅归档 source-run 和显式 --log 中真实存在的文件；缺失的阶段产物不会被推断为通过。"
        ),
    }
    (destination / "archive_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "archived", "destination": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
