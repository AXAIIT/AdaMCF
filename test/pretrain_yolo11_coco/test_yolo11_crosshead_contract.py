#!/usr/bin/env python
"""YOLO11 CrossHead 蒸馏的无训练结构合同单测。

该测试只验证三尺度 Detect 中间特征经过显式训练期通道适配器后，能够进入冻结
教师尾部并产生与教师原始 ``[N, 144, H, W]`` logits 对齐的 cross-head 输出；不会
创建 optimizer、不会反传、不会写 checkpoint。通过后仍必须执行
``--admission-probe`` 的真实 1 epoch 准入。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

import distill_yolo11_crosshead as crosshead
import distill_yolo11_necksl2 as neck
import distill_yolo11_preddfl as preddfl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--cross-head-start-layer", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    teacher_weights = neck.require_file(args.teacher, "教师权重")
    student_weights = neck.require_file(args.student, "学生权重")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有合同报告：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    device = neck.resolve_device(args.device)
    student = YOLO(str(student_weights)).model.to(device).float().eval()
    teacher = YOLO(str(teacher_weights)).model.to(device).float().eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if hasattr(student, "criterion"):
        student.criterion = None
    if hasattr(teacher, "criterion"):
        teacher.criterion = None

    contract = crosshead.build_cross_head_contract(
        student.cpu(), teacher.cpu(), imgsz=args.imgsz, start_layer=args.cross_head_start_layer
    )
    student = student.to(device)
    teacher = teacher.to(device)
    adapters = crosshead.make_cross_head_adapters(contract).to(device).float().eval()
    capture = crosshead.HeadIntermediateCapture(student, contract, "cross_head_unit_student")
    try:
        image = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
        with torch.no_grad():
            _ = student(image)
            teacher_output = teacher(image)
            teacher_raw = preddfl.raw_detect_layers(teacher_output, label="cross_head_unit_teacher")
            cross_raw = crosshead._forward_teacher_tail(teacher, capture, contract, adapters)
        preddfl.validate_runtime_raw_contract(cross_raw, teacher_raw, contract["prediction_contract"])
        finite = all(bool(torch.isfinite(value).all().item()) for value in cross_raw)
        if not finite:
            raise AssertionError("cross-head 输出包含 NaN/Inf。")
    finally:
        capture.close()
    report = {
        "status": "pass",
        "teacher": str(teacher_weights),
        "student": str(student_weights),
        "device": str(device),
        "cross_head_contract": contract,
        "cross_head_adapter": adapters.report(),
        "teacher_raw_shapes": [list(value.shape) for value in teacher_raw],
        "cross_head_raw_shapes": [list(value.shape) for value in cross_raw],
        "teacher_frozen": neck.frozen_parameter_report(teacher),
        "contract_test": "three_scale_cross_head_adapter_raw_shape_and_finiteness_pass",
    }
    neck.write_json(output, report)
    print(f"PASS: YOLO11 CrossHead Detect 合同；报告：{output}")


if __name__ == "__main__":
    main()
