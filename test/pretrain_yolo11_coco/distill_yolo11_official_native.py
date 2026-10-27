#!/usr/bin/env python
"""YOLO11 官方原生 Neck-SL2 蒸馏的物理子网桥接入口。

官方 ``yolo train model=<physical_student.pt> distill_model=<teacher.pt>`` 在
8.4.77 中会读取学生 checkpoint 的原始 YAML，再由 ``DetectionTrainer`` 重建
满宽模型。这对 OTO 物理子网是不安全的：权重虽来自剪枝模型，网络却退回未剪
枝结构。本入口只改变 Trainer 桥接层，不修改 OTO/HESSO：

* ``get_model`` 忽略 cfg/weights，始终复制输入的物理学生；
* 让官方 ``BaseTrainer`` 正常包装 ``DistillationModel``，保留官方 projector、
  teacher 冻结和 score-weighted Neck-SL2 loss；
* ``save_model`` 显式只保存 ``DistillationModel.student_model`` 的物理副本；
* 训练前后均用结构/状态签名确认没有回退为满宽 YOLO11。

部署位置：
  /workspace/OTOv2_v2/test/pretrain_yolo11_coco/distill_yolo11_official_native.py

该脚本需要运行在独立的 ``yolo11_native_kd`` 环境（Ultralytics 8.4.77）。
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_NO_AUTO_INSTALL", "True")
os.environ.setdefault("ULTRALYTICS_SETTINGS", "0")

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.distill_model import DistillationModel
from ultralytics.utils.torch_utils import unwrap_model

import distill_yolo11_necksl2 as neck


def require_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在：{path}")
    return path


def physical_from_training_model(model: torch.nn.Module) -> torch.nn.Module:
    """从官方 wrapper/EMA 中取出并复制纯物理 student。"""

    raw = unwrap_model(model)
    if isinstance(raw, DistillationModel):
        raw = raw.student_model
    return neck.physical_student_copy(raw)


def optimizer_contract(optimizer, wrapper: DistillationModel) -> dict[str, Any]:
    """确认 teacher 未进入 optimizer，student/projector 各自恰好一次。"""

    ownership: dict[int, int] = {}
    for group in optimizer.param_groups:
        for parameter in group.get("params", []):
            ownership[id(parameter)] = ownership.get(id(parameter), 0) + 1
    student = wrapper.student_model
    teacher = wrapper.teacher_model
    projector = wrapper.projector
    student_parameters = list(student.parameters())
    teacher_parameters = list(teacher.parameters()) if teacher is not None else []
    projector_parameters = list(projector.parameters())
    student_missing = [i for i, p in enumerate(student_parameters) if ownership.get(id(p), 0) != 1]
    projector_missing = [i for i, p in enumerate(projector_parameters) if ownership.get(id(p), 0) != 1]
    teacher_owned = [i for i, p in enumerate(teacher_parameters) if ownership.get(id(p), 0) != 0]
    # Ultralytics 8.4.77 的通用 build_optimizer 会把冻结 teacher 的参数也放入
    # optimizer param_groups；它们 requires_grad=False，因此不会产生梯度或更新。
    # 这里不能把该官方行为误判为结构错误，真正的硬门禁是 student/projector
    # 各恰好一次、teacher 全部冻结且无梯度。
    teacher_frozen = all(not parameter.requires_grad for parameter in teacher_parameters)
    return {
        "status": "pass" if not student_missing and not projector_missing and teacher_frozen else "fail",
        "student_parameter_count": len(student_parameters),
        "student_missing_or_duplicated": student_missing,
        "projector_parameter_count": len(projector_parameters),
        "projector_missing_or_duplicated": projector_missing,
        "teacher_parameter_count": len(teacher_parameters),
        "teacher_in_optimizer": teacher_owned,
        "teacher_parameters_all_frozen": teacher_frozen,
        "teacher_optimizer_membership_allowed": True,
    }


def create_bridge_trainer(student_template: torch.nn.Module, expected_structure: dict[str, Any], expected_state: dict[str, Any]):
    """返回官方 DetectionTrainer 的物理学生桥接子类。"""

    class OfficialPhysicalKDTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            # 这是防止 YAML 回退的关键。Model.train() 会主动调用 get_model(cfg, weights)，
            # 所以仅设置 trainer.model 或仅覆盖 setup_model 都不够。
            del cfg, weights
            model = neck.physical_student_copy(student_template).float()
            if verbose and hasattr(model, "info"):
                model.info(verbose=True)
            return model

        def _setup_train(self):
            super()._setup_train()
            wrapper = unwrap_model(self.model)
            if not isinstance(wrapper, DistillationModel):
                raise RuntimeError("官方 Trainer 未构建 DistillationModel，蒸馏准入失败。")
            physical = neck.physical_student_copy(wrapper.student_model)
            actual_structure = neck.core.model_structure_signature(physical)
            actual_state = neck.core.model_state_signature(physical)
            structure_matched = actual_structure["sha256"] == expected_structure["sha256"]
            state_matched = actual_state["sha256"] == expected_state["sha256"]
            ownership = optimizer_contract(self.optimizer, wrapper)
            teacher_frozen = neck.frozen_parameter_report(wrapper.teacher_model)
            self.bridge_preflight = {
                "status": "pass" if structure_matched and state_matched and ownership["status"] == "pass" and teacher_frozen["requires_grad_count"] == 0 else "fail",
                "implementation": "ultralytics_8_4_77_official_distillationmodel_physical_student_bridge_v1",
                "expected_structure": expected_structure,
                "actual_student_structure": actual_structure,
                "expected_state": expected_state,
                "actual_student_state": actual_state,
                "structure_matched": structure_matched,
                "state_matched": state_matched,
                "student_parameter_count": sum(p.numel() for p in wrapper.student_model.parameters()),
                "teacher_parameter_count": sum(p.numel() for p in wrapper.teacher_model.parameters()),
                "projector_parameter_count": sum(p.numel() for p in wrapper.projector.parameters()),
                "distill_layers": list(wrapper.feats_idx),
                "dis": float(wrapper.dis),
                "optimizer_contract": ownership,
                "teacher_frozen": teacher_frozen,
            }
            report_path = Path(self.save_dir) / "admission" / "official_native_kd_preflight.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            neck.core.json_dump(report_path, self.bridge_preflight)
            if self.bridge_preflight["status"] != "pass":
                raise RuntimeError(f"官方原生 KD 物理学生准入失败：{report_path}")

        def save_model(self):
            """覆盖官方 wrapper/EMA 保存，部署 checkpoint 只保留物理 student。"""

            ema_source = getattr(getattr(self, "ema", None), "ema", None)
            source = ema_source if ema_source is not None else self.model
            student_only = physical_from_training_model(source).half()
            checkpoint = {
                "epoch": int(getattr(self, "epoch", -1)),
                "best_fitness": getattr(self, "best_fitness", None),
                "model": student_only,
                "ema": None,
                "updates": int(getattr(getattr(self, "ema", None), "updates", 0)),
                "optimizer": None,
                "scaler": None,
                "train_args": vars(self.args).copy(),
                "train_metrics": {**getattr(self, "metrics", {}), "fitness": getattr(self, "fitness", None)},
                "date": neck.utc_now(),
                "version": str(getattr(neck.core.ultralytics, "__version__", "unknown")),
                "distillation_checkpoint_contract": "official_distillation_physical_student_only_v1",
            }
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            serialized = buffer.getvalue()
            self.last.write_bytes(serialized)
            if getattr(self, "best_fitness", None) == getattr(self, "fitness", None):
                self.best.write_bytes(serialized)
            save_period = int(getattr(self, "save_period", getattr(self.args, "save_period", -1)))
            if save_period > 0 and self.epoch > 0 and self.epoch % save_period == 0:
                (Path(self.wdir) / f"epoch{self.epoch}.pt").write_bytes(serialized)

    return OfficialPhysicalKDTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO11 官方原生 Neck-SL2 KD 物理学生桥接")
    sub = parser.add_subparsers(dest="mode", required=True)
    train = sub.add_parser("train")
    train.add_argument("--teacher", required=True)
    train.add_argument("--student", required=True)
    train.add_argument("--data", required=True)
    train.add_argument("--fixed-input", required=True)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--workers", type=int, default=8)
    train.add_argument("--device", default="0")
    train.add_argument("--dis", type=float, default=6.0)
    train.add_argument("--optimizer", default="auto", help="官方默认 auto；需要严格匹配旧实验时可传 SGD")
    train.add_argument("--warmup-epochs", type=float, default=3.0)
    train.add_argument("--lr0", type=float, default=0.0005)
    train.add_argument("--momentum", type=float, default=0.937)
    train.add_argument("--weight-decay", type=float, default=0.0005)
    train.add_argument("--close-mosaic", type=int, default=20)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--output", default="runs")
    train.add_argument("--name", required=True)
    train.add_argument("--no-overwrite", action="store_true")
    train.add_argument("--validation-batch", type=int, default=None)
    return parser.parse_args()


def run_train(args: argparse.Namespace) -> None:
    teacher_weights = require_file(args.teacher, "teacher 权重")
    student_weights = require_file(args.student, "physical student 权重")
    data_yaml = require_file(args.data, "数据集 YAML")
    fixed_input = require_file(args.fixed_input, "固定输入")
    run_root = Path(args.output).expanduser().resolve()
    requested_run_dir = run_root / args.name
    if args.no_overwrite and requested_run_dir.exists():
        raise FileExistsError(f"禁止覆盖已有运行目录：{requested_run_dir}")

    started_at = time.time()
    student_template = YOLO(str(student_weights)).model.cpu().float()
    teacher_template = YOLO(str(teacher_weights)).model.cpu().float()
    student_physical = neck.physical_student_copy(student_template)
    before_structure = neck.core.model_structure_signature(student_physical)
    before_state = neck.core.model_state_signature(student_physical)
    before_stats = neck.core.model_stats(student_physical, args.imgsz, torch.device("cpu"))
    print("=" * 60, flush=True)
    print("开始 YOLO11 官方原生 KD（物理学生桥接）", flush=True)
    print(f"teacher: {teacher_weights}")
    print(f"student: {student_weights}")
    print(f"student 参数量: {before_structure['parameter_count']:,}")
    print(f"ultralytics: {getattr(neck.core.ultralytics, '__version__', 'unknown')}")
    print("=" * 60, flush=True)

    trainer_class = create_bridge_trainer(student_template, before_structure, before_state)
    yolo = YOLO(str(student_weights))
    yolo.train(
        trainer=trainer_class,
        data=str(data_yaml),
        distill_model=str(teacher_weights),
        dis=args.dis,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(run_root),
        name=args.name,
        exist_ok=not args.no_overwrite,
        pretrained=False,
        optimizer=args.optimizer,
        lr0=args.lr0,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        close_mosaic=args.close_mosaic,
        seed=args.seed,
        amp=False,
        compile=False,
        plots=False,
        verbose=True,
    )
    trainer = yolo.trainer
    run_dir = Path(trainer.save_dir).resolve()
    best_weights = Path(trainer.best if trainer.best.is_file() else trainer.last).resolve()
    if not best_weights.is_file():
        raise RuntimeError(f"训练完成后缺少 best/last checkpoint：{best_weights}")

    checkpoint_report = neck.pure_checkpoint_report(best_weights)
    final_model = YOLO(str(best_weights)).model.cpu().float().eval()
    final_structure = neck.core.model_structure_signature(final_model)
    final_state = neck.core.model_state_signature(final_model)
    structure_matched = final_structure["sha256"] == before_structure["sha256"]
    parameter_matched = final_structure["parameter_count"] == before_structure["parameter_count"]
    if checkpoint_report["status"] != "pass" or not structure_matched or not parameter_matched:
        report = {
            "status": "fail",
            "checkpoint": checkpoint_report,
            "before_structure": before_structure,
            "after_structure": final_structure,
            "structure_matched": structure_matched,
            "parameter_matched": parameter_matched,
        }
        neck.core.json_dump(run_dir / "official_native_kd_structure_failure.json", report)
        raise RuntimeError(f"官方原生 KD 最佳 checkpoint 不是输入物理学生结构：{run_dir / 'official_native_kd_structure_failure.json'}")

    validation_args = argparse.Namespace(
        data=str(data_yaml), imgsz=args.imgsz, batch=args.validation_batch or args.batch,
        device=args.device, workers=args.workers, no_overwrite=args.no_overwrite,
    )
    final_validation = neck.core.validate_yolo_checkpoint(
        best_weights, validation_args,
        project=run_dir / "independent_validation", name="best",
    )
    fixed_forward = neck.core.fixed_input_forward_report(
        best_weights, fixed_input, imgsz=args.imgsz,
        device=neck.resolve_device(args.device), output_dir=run_dir / "fixed_input",
        report_name="official_native_kd_best_fixed_input",
    )
    after_stats = neck.core.model_stats(final_model, args.imgsz, neck.resolve_device(args.device))
    bridge_report = {
        "status": "pass",
        "teacher_weights": str(teacher_weights),
        "teacher_sha256": neck.core.sha256_file(teacher_weights),
        "student_source_weights": str(student_weights),
        "student_source_sha256": neck.core.sha256_file(student_weights),
        "best_weights": str(best_weights),
        "best_sha256": neck.core.sha256_file(best_weights),
        "before_structure": before_structure,
        "after_structure": final_structure,
        "before_state": before_state,
        "after_state": final_state,
        "structure_matched": structure_matched,
        "parameter_matched": parameter_matched,
        "checkpoint": checkpoint_report,
        "trainer_preflight": getattr(trainer, "bridge_preflight", {}),
        "validation": final_validation,
        "fixed_input": fixed_forward,
    }
    neck.core.json_dump(run_dir / "official_native_kd_bridge_report.json", bridge_report)
    neck.core.save_stage_summary(
        run_dir / "distillation_summary.json", phase="官方原生蒸馏", started_at=started_at,
        source_weights=student_weights, final_weights=best_weights,
        before_stats=before_stats, after_stats=after_stats,
        final_validation=final_validation, epoch_reporter=None,
        architecture_check=bridge_report,
        validation_scope="Ultralytics 8.4.77 官方 DistillationModel + 物理学生桥接后的独立验证。",
    )
    print(f"[done] 官方原生 KD bridge 通过，checkpoint: {best_weights}", flush=True)


def main() -> None:
    args = parse_args()
    if args.mode != "train":
        raise RuntimeError(f"未知模式：{args.mode}")
    run_train(args)


if __name__ == "__main__":
    main()
