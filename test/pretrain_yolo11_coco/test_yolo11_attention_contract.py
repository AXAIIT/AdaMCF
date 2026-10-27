#!/usr/bin/env python
"""YOLO11 Attention 两种结构化剪枝合同的快速运行测试。

该测试不训练 COCO，只从给定 YOLO11 权重中复制一个 Attention 模块，分别验证：

1. 逻辑组 -> QKV/PE/proj 物理索引变换；
2. 完整 head 与所有 heads 中同一 V/head 维的真实物理删除；
3. 固定输入前向、反向传播；
4. 自定义物理子网类的 torch.save/torch.load 可持久化。

完整项目验收仍须运行两个 profile 各自的九阶段闭环。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
OTO_ROOT = CURRENT_FILE.parents[2]
if str(OTO_ROOT) not in sys.path:
    sys.path.insert(0, str(OTO_ROOT))

import torch
from ultralytics import YOLO

from only_train_once.graph.node_group import NodeGroupComposedOp
from only_train_once.operation.operator import YOLO11AttentionOTO, is_yolo11_conv_attention
from only_train_once.operation.yolo11_attention import upgrade_yolo11_attention_module
from only_train_once.transform import (
    TensorTransform,
    index_transformation_param_group,
    pruning_axis,
    tensor_transformation,
)


def find_attention(model):
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if is_yolo11_conv_attention(module)
    ]
    if not matches:
        raise RuntimeError("权重中没有找到 YOLO11 qkv/proj/pe Attention。")
    return matches[0]


def parameter_count(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def output_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from output_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from output_tensors(item)


def verify_forward_adapter_equivalence(source_module, device: torch.device) -> dict:
    """在未剪枝宽度上证明动态 forward 与官方 forward 数值等价。"""
    official = copy.deepcopy(source_module).to(device).float().eval()
    adapted = copy.deepcopy(source_module).to(device).float().eval()
    upgrade_yolo11_attention_module(adapted, "head_dim")
    torch.manual_seed(42)
    input_tensor = torch.randn(
        2, official.qkv.conv.in_channels, 8, 8, device=device
    )
    with torch.no_grad():
        expected = official(input_tensor)
        actual = adapted(input_tensor)
    max_abs = float((expected - actual).abs().max().cpu().item())
    allclose = bool(torch.allclose(expected, actual, atol=1e-6, rtol=1e-5))
    if not allclose:
        raise AssertionError(f"动态 Attention forward 与官方实现不等价：max_abs={max_abs}")
    return {
        "status": "pass",
        "allclose": allclose,
        "max_abs_error": max_abs,
        "shape": list(actual.shape),
    }


def verify_disabled_mode_trainability(source_module, device: torch.device) -> dict:
    """验证“不可结构剪枝”不会把 Attention 参数从 optimizer 所有权中删除。"""

    module = copy.deepcopy(source_module).to(device).float().train()
    module.requires_grad_(True)
    module._oto_attention_active_prune_mode = "disabled"
    op = YOLO11AttentionOTO(
        id="attention_disabled_under_test",
        _type="Attention",
        module=module,
        model=None,
    )
    group = op.get_param_groups()
    expected_names = sorted(
        name for name, parameter in op.name_to_param.items() if parameter.requires_grad
    )
    actual_names = sorted(group["p_names"])
    if op.is_stem or op.num_groups != 1:
        raise AssertionError(
            "disabled Attention 不应成为结构剪枝 stem："
            f"is_stem={op.is_stem}, num_groups={op.num_groups}"
        )
    if not expected_names or actual_names != expected_names:
        raise AssertionError(
            "disabled Attention 可训练参数所有权不完整："
            f"expected={expected_names}, actual={actual_names}"
        )
    if len({id(parameter) for parameter in group["params"]}) != len(expected_names):
        raise AssertionError("disabled Attention 参数所有权存在重复 Parameter。")
    if any(
        transform != TensorTransform.NO_PRUNE
        for transform in group["p_transform"]
    ):
        raise AssertionError("disabled Attention 参数必须统一使用 NO_PRUNE。")
    return {
        "status": "pass",
        "structurally_prunable": False,
        "trainable_parameter_count": len(expected_names),
        "optimizer_owned_parameter_names": actual_names,
        "transforms": "NO_PRUNE",
    }


def run_mode(source_module, mode: str, device: torch.device) -> dict:
    module = copy.deepcopy(source_module).to(device).float().train()
    # ``YOLO(weights)`` loads an inference checkpoint whose parameters may all be
    # frozen.  The real prune/audit entry point mirrors Ultralytics Trainer and
    # enables trainable parameters before OTO graph construction.  This focused
    # contract test must reproduce the same lifecycle; otherwise OTO correctly
    # filters every frozen parameter and the synthetic parameter group is empty.
    frozen_parameters_on_load = sum(
        1 for parameter in module.parameters() if not parameter.requires_grad
    )
    module.requires_grad_(True)
    trainable_parameters = sum(
        1 for parameter in module.parameters() if parameter.requires_grad
    )
    if trainable_parameters == 0:
        raise AssertionError("Attention 合同准备失败：没有恢复任何可训练参数。")
    module._oto_attention_active_prune_mode = mode
    op = YOLO11AttentionOTO(
        id="attention_under_test",
        _type="Attention",
        module=module,
        model=None,
    )
    hesso_group = op.get_param_groups(skip_output_node=True)
    physical_group = op.get_param_groups(skip_output_node=False)
    # 正式 OTO 路径由 NodeGroupComposedOp.get_param_groups() 在算子返回值外层
    # 注入 num_groups。本测试直接调用复合算子，因此在这里构造同样的最小
    # node-group 参数合同，避免把测试夹具差异误判成算子合同错误。
    logical_groups = int(op.num_groups)
    for group_name, group in (
        ("hesso_internal", hesso_group),
        ("physical_contract", physical_group),
    ):
        if "num_groups" in group and int(group["num_groups"]) != logical_groups:
            raise AssertionError(
                "Attention 算子/节点组逻辑组数不一致："
                f"role={group_name}, op={logical_groups}, group={group['num_groups']}。"
            )
        group["num_groups"] = logical_groups
    if logical_groups <= 0:
        raise AssertionError(
            f"Attention 逻辑组数无效：mode={mode}, num_groups={logical_groups}。"
        )
    if hesso_group["prune_mode"] != mode or physical_group["prune_mode"] != mode:
        raise AssertionError(
            "模式未进入参数组："
            f"hesso={hesso_group['prune_mode']}, physical={physical_group['prune_mode']}, "
            f"expected={mode}"
        )
    if not hesso_group["params"]:
        raise AssertionError("Attention 合同没有收集任何参数。")
    if (
        hesso_group.get("parameter_role")
        != "qkv_pe_internal_structural_owner"
        or int(hesso_group.get("parameter_ownership_priority", 0)) != 100
    ):
        raise AssertionError(
            "HESSO 内部组没有声明最高优先级的结构所有权："
            f"role={hesso_group.get('parameter_role')}, "
            f"priority={hesso_group.get('parameter_ownership_priority')}"
        )
    if any(
        name.endswith(".proj.conv.weight") or ".proj.bn." in name
        for name in hesso_group["p_names"]
    ):
        raise AssertionError("HESSO 内部冗余组错误包含了 proj 固定输出参数。")
    if not any(name.endswith(".qkv.conv.weight") for name in hesso_group["p_names"]):
        raise AssertionError("HESSO 内部冗余组缺少 qkv 驱动参数。")
    if not any(name.endswith(".pe.conv.weight") for name in hesso_group["p_names"]):
        raise AssertionError("HESSO 内部冗余组缺少 pe 驱动参数。")
    if not any(name.endswith(".proj.conv.weight") for name in physical_group["p_names"]):
        raise AssertionError("Attention 物理合同缺少 proj axis=1 消费者。")

    # 显式参数查询模拟 proj 所在的外部固定 BasicNodeGroup。eval/ONNX trace
    # 通常只暴露 proj.conv.weight，但该查询必须原子接管同一包装器的 BN
    # weight/bias；三者都只正常训练，不能把外部输出索引用于内部 V 维裁剪。
    proj_trace_names = [
        name for name in physical_group["p_names"] if name.endswith(".proj.conv.weight")
    ]
    external_proj_group = op.get_param_groups(param_names=proj_trace_names)
    expected_external_proj_names = sorted(
        name
        for name, parameter in op.name_to_param.items()
        if parameter.requires_grad
        and (name.endswith(".proj.conv.weight") or ".proj.bn." in name)
    )
    actual_external_proj_names = sorted(external_proj_group["p_names"])
    if (
        actual_external_proj_names != expected_external_proj_names
        or any(
            transform != TensorTransform.NO_PRUNE
            for transform in external_proj_group["p_transform"]
        )
        or external_proj_group.get("parameter_role")
        != "fixed_proj_output_conv_bn_owner"
        or int(external_proj_group.get("parameter_ownership_priority", 0)) != 0
    ):
        raise AssertionError(
            "proj 外部固定组没有原子接管 Conv+BN 的 NO_PRUNE 所有权："
            f"expected={expected_external_proj_names}, actual={actual_external_proj_names}, "
            f"transforms={external_proj_group['p_transform']}"
        )

    # 模拟最终 optimizer 的两类所有者：内部 composed group + 外部固定 proj
    # group。每个可训练 Attention Parameter 必须恰好出现一次，直接覆盖本次
    # proj.bn.weight/bias 丢失回归。
    expected_trainable_names = sorted(
        name for name, parameter in op.name_to_param.items() if parameter.requires_grad
    )
    ownership_counts = {}
    ownership_names = {}
    for owner_name, owner_group in (
        ("hesso_internal", hesso_group),
        ("fixed_proj_output", external_proj_group),
    ):
        for name, parameter in zip(owner_group["p_names"], owner_group["params"]):
            parameter_id = id(parameter)
            ownership_counts[parameter_id] = ownership_counts.get(parameter_id, 0) + 1
            ownership_names.setdefault(parameter_id, []).append((owner_name, name))
    missing_owners = sorted(
        name
        for name, parameter in op.name_to_param.items()
        if parameter.requires_grad and ownership_counts.get(id(parameter), 0) == 0
    )
    duplicate_owners = sorted(
        name
        for name, parameter in op.name_to_param.items()
        if parameter.requires_grad and ownership_counts.get(id(parameter), 0) > 1
    )
    if missing_owners or duplicate_owners:
        raise AssertionError(
            "Attention optimizer 参数所有权不唯一："
            f"missing={missing_owners}, duplicated={duplicate_owners}, "
            f"owners={ownership_names}"
        )
    actual_owned_names = sorted(
        name
        for owner_group in (hesso_group, external_proj_group)
        for name in owner_group["p_names"]
    )
    if actual_owned_names != expected_trainable_names:
        raise AssertionError(
            "Attention optimizer 参数所有权集合不完整："
            f"expected={expected_trainable_names}, actual={actual_owned_names}"
        )
    transformed_shapes = {}
    expanded_indexes = {}
    for name, parameter, transform in zip(
        physical_group["p_names"], physical_group["params"], physical_group["p_transform"]
    ):
        transformed = tensor_transformation(parameter, transform, physical_group["num_groups"])
        if transformed.shape[0] != physical_group["num_groups"]:
            raise AssertionError(
                f"{name} 变换后逻辑组错误：{tuple(transformed.shape)}"
            )
        physical = index_transformation_param_group([0], transform, physical_group)
        axis = pruning_axis(transform)
        if not physical or max(physical) >= parameter.shape[axis]:
            raise AssertionError(
                f"{name} 物理索引越界：axis={axis}, indexes={physical[-8:]}"
            )
        transformed_shapes[name] = list(transformed.shape)
        expanded_indexes[name] = {
            "axis": axis,
            "logical_group_0_physical_count": len(physical),
            "min": min(physical),
            "max": max(physical),
        }

    before = {
        "num_heads": int(module.num_heads),
        "key_dim": int(module.key_dim),
        "head_dim": int(module.head_dim),
        "qkv_out": int(module.qkv.conv.out_channels),
        "pe_channels": int(module.pe.conv.out_channels),
        "proj_in": int(module.proj.conv.in_channels),
        "proj_out": int(module.proj.conv.out_channels),
        "parameters": parameter_count(module),
    }
    if mode == "num_head" and before["num_heads"] < 2:
        raise AssertionError("num_head 测试至少需要 2 个 Attention heads。")
    if mode == "head_dim" and before["head_dim"] < 2:
        raise AssertionError("head_dim 测试至少需要 2 个 head dimensions。")

    # 不直接手工调用 ``op.prune_out_dim([0])``。真实复现 HESSO 的逻辑组置零，
    # 再让 NodeGroupComposedOp 走 construct_subnet 使用的冗余重算路径；本回归
    # 专门防止非零 proj 消费者重新混入范数后把零组掩盖。
    with torch.no_grad():
        for parameter, transform in zip(
            hesso_group["params"], hesso_group["p_transform"]
        ):
            physical = index_transformation_param_group([0], transform, hesso_group)
            indexes = torch.as_tensor(physical, dtype=torch.long, device=parameter.device)
            parameter.index_fill_(pruning_axis(transform), indexes, 0.0)
    composed_group = NodeGroupComposedOp(is_prunable=True, op=op)
    composed_group.set_pruning_redundant_idxes()
    reconstructed_redundancy = [
        int(index) for index in composed_group.pruning_redundant_idxes
    ]
    if reconstructed_redundancy != [0]:
        raise AssertionError(
            "Attention composed group 未能从 HESSO 零权重恢复逻辑冗余索引："
            f"{reconstructed_redundancy}"
        )
    composed_group.prune_out_dim()
    original_axis_provenance = {
        "num_heads": getattr(module, "_oto_attention_original_num_heads", None),
        "key_dim": getattr(module, "_oto_attention_original_key_dim", None),
        "head_dim": getattr(module, "_oto_attention_original_head_dim", None),
    }
    expected_axis_provenance = {
        "num_heads": before["num_heads"],
        "key_dim": before["key_dim"],
        "head_dim": before["head_dim"],
    }
    if original_axis_provenance != expected_axis_provenance:
        raise AssertionError(
            "Attention 物理子网没有正确保存剪枝前轴属性："
            f"expected={expected_axis_provenance}, actual={original_axis_provenance}"
        )
    if getattr(module, "_oto_attention_physical_mode", None) != mode:
        raise AssertionError("物理子网没有持久记录实际 Attention 剪枝模式。")
    if getattr(module, "_oto_attention_active_prune_mode", None) != "disabled":
        raise AssertionError("物理子网保存前没有关闭 active mode，存在意外二次剪枝风险。")
    input_tensor = torch.randn(
        2,
        module.qkv.conv.in_channels,
        8,
        8,
        device=device,
        requires_grad=True,
    )
    output = module(input_tensor)
    if output.shape != (2, before["proj_out"], 8, 8):
        raise AssertionError(f"物理剪枝后输出形状错误：{tuple(output.shape)}")
    output.square().mean().backward()
    if input_tensor.grad is None or not torch.isfinite(input_tensor.grad).all():
        raise AssertionError("物理剪枝后反向传播未产生有限输入梯度。")

    after = {
        "num_heads": int(module.num_heads),
        "key_dim": int(module.key_dim),
        "head_dim": int(module.head_dim),
        "qkv_out": int(module.qkv.conv.out_channels),
        "qkv_bn": int(module.qkv.bn.num_features),
        "pe_channels": int(module.pe.conv.out_channels),
        "pe_in": int(module.pe.conv.in_channels),
        "pe_groups": int(module.pe.conv.groups),
        "pe_bn": int(module.pe.bn.num_features),
        "proj_in": int(module.proj.conv.in_channels),
        "proj_out": int(module.proj.conv.out_channels),
        "parameters": parameter_count(module),
        "output_shape": list(output.shape),
        "output_finite": bool(torch.isfinite(output).all()),
    }
    if after["parameters"] >= before["parameters"]:
        raise AssertionError("Attention 物理剪枝后参数量没有下降。")
    expected_inner = after["num_heads"] * after["head_dim"]
    expected_qkv = after["num_heads"] * (
        after["key_dim"] * 2 + after["head_dim"]
    )
    if mode == "num_head" and not (
        after["num_heads"] == before["num_heads"] - 1
        and after["head_dim"] == before["head_dim"]
    ):
        raise AssertionError(f"num_head 属性变化不符合合同：before={before}, after={after}")
    if mode == "head_dim" and not (
        after["num_heads"] == before["num_heads"]
        and after["head_dim"] == before["head_dim"] - 1
    ):
        raise AssertionError(f"head_dim 属性变化不符合合同：before={before}, after={after}")
    if after["key_dim"] != before["key_dim"]:
        raise AssertionError("两种 Attention 模式都不应改变 key_dim。")
    if not (
        after["qkv_out"] == after["qkv_bn"] == expected_qkv
        and after["pe_in"]
        == after["pe_channels"]
        == after["pe_groups"]
        == after["pe_bn"]
        == after["proj_in"]
        == expected_inner
    ):
        raise AssertionError(f"QKV/PE/proj 内部宽度合同不一致：{after}")
    if after["proj_out"] != before["proj_out"]:
        raise AssertionError("Attention 外部输出宽度不应改变。")

    with tempfile.TemporaryDirectory(prefix=f"oto_yolo11_attention_{mode}_") as tmp:
        checkpoint = Path(tmp) / "attention.pt"
        torch.save(module.eval(), checkpoint)
        reloaded = torch.load(checkpoint, map_location=device, weights_only=False)
        if getattr(reloaded, "_oto_attention_physical_mode", None) != mode:
            raise AssertionError("Attention 重载后丢失 physical mode provenance。")
        if getattr(reloaded, "_oto_attention_active_prune_mode", None) != "disabled":
            raise AssertionError("Attention 重载后 active mode 未保持关闭。")
        with torch.no_grad():
            reloaded_output = reloaded(input_tensor.detach())
        if reloaded_output.shape != output.shape or not torch.isfinite(reloaded_output).all():
            raise AssertionError("Attention 物理模块保存/重载后前向失败。")

    return {
        "status": "pass",
        "mode": mode,
        "parameter_state_preparation": {
            "frozen_tensors_on_checkpoint_load": frozen_parameters_on_load,
            "trainable_tensors_before_oto_grouping": trainable_parameters,
        },
        "logical_groups": int(hesso_group["num_groups"]),
        "contract": hesso_group["attention_contract"],
        "parameter_ownership": {
            "hesso_internal_parameter_names": list(hesso_group["p_names"]),
            "physical_contract_parameter_names": list(physical_group["p_names"]),
            "external_proj_parameter_names": list(external_proj_group["p_names"]),
            "external_proj_transform": "NO_PRUNE",
            "all_trainable_parameters_owned_exactly_once": True,
        },
        "composed_redundancy_reconstruction": {
            "status": "pass",
            "zeroed_logical_groups": [0],
            "reconstructed_redundant_groups": reconstructed_redundancy,
        },
        "before": before,
        "after": after,
        "original_axis_provenance": original_axis_provenance,
        "transformed_shapes": transformed_shapes,
        "expanded_indexes": expanded_indexes,
        "forward_backward": "pass",
        "torch_save_reload": "pass",
    }


def verify_compatible_yolo11_checkpoint(weights: Path, device: torch.device) -> dict:
    """加载既有保守剪枝/微调权重，验证前向及 disabled 参数所有权。"""

    yolo = YOLO(str(weights))
    model = yolo.model.to(device).float().eval()
    attention_modules = [
        (name, module)
        for name, module in model.named_modules()
        if is_yolo11_conv_attention(module)
    ]
    if not attention_modules:
        raise AssertionError(f"兼容权重没有 YOLO11 Attention：{weights}")
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 64, 64, device=device))
    tensors = list(output_tensors(output))
    if not tensors or any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise AssertionError(f"兼容权重固定输入前向无有效有限 Tensor：{weights}")
    ownership = {
        name: verify_disabled_mode_trainability(module, device)
        for name, module in attention_modules
    }
    return {
        "status": "pass",
        "weights": str(weights),
        "model_type": type(model).__name__,
        "attention_module_types": {
            name: type(module).__name__ for name, module in attention_modules
        },
        "output_shapes": [list(tensor.shape) for tensor in tensors],
        "output_finite": True,
        "disabled_attention_optimizer_ownership": ownership,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO11 Attention OTO 合同快速测试")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    parser.add_argument(
        "--compat-weights",
        action="append",
        default=[],
        help="可重复传入既有 YOLO11 保守剪枝/微调 checkpoint 做加载兼容回归。",
    )
    args = parser.parse_args()
    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file():
        parser.error(f"--weights 不存在：{weights}")
    device = torch.device(args.device)
    compatibility_weights = [
        Path(value).expanduser().resolve() for value in args.compat_weights
    ]
    missing_compatibility_weights = [
        str(path) for path in compatibility_weights if not path.is_file()
    ]
    if missing_compatibility_weights:
        parser.error(f"--compat-weights 不存在：{missing_compatibility_weights}")
    yolo = YOLO(str(weights))
    attention_name, attention = find_attention(yolo.model)
    report = {
        "status": "pass",
        "weights": str(weights),
        "attention_module": attention_name,
        "device": str(device),
        "unpruned_forward_adapter_equivalence": verify_forward_adapter_equivalence(
            attention, device
        ),
        "disabled_mode_trainability": verify_disabled_mode_trainability(
            attention, device
        ),
        "modes": {
            mode: run_mode(attention, mode, device)
            for mode in ("num_head", "head_dim")
        },
        "legacy_checkpoint_compatibility": [
            verify_compatible_yolo11_checkpoint(path, device)
            for path in compatibility_weights
        ],
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    output = None
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")

    num_head = report["modes"]["num_head"]
    head_dim = report["modes"]["head_dim"]
    print("\n" + "=" * 72, flush=True)
    print("YOLO11 Attention 两种物理剪枝合同", flush=True)
    print("=" * 72, flush=True)
    print(f"状态             : 通过", flush=True)
    print(f"Attention 模块   : {attention_name}", flush=True)
    print(
        "num_head 模式    : "
        f"{num_head['before']['num_heads']} → {num_head['after']['num_heads']} 个 head，"
        f"前向/反向与重载通过",
        flush=True,
    )
    print(
        "head_dim 模式    : "
        f"{head_dim['before']['head_dim']} → {head_dim['after']['head_dim']} 维/head，"
        f"head 数保持 {head_dim['after']['num_heads']}，前向/反向与重载通过",
        flush=True,
    )
    print(
        "Q/K 维度         : 两种模式均保持 key_dim="
        f"{head_dim['after']['key_dim']}",
        flush=True,
    )
    if output is not None:
        print(f"详细报告         : {output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if os.environ.get("OTO_DEBUG_TRACEBACK") == "1":
            raise
        print("\n" + "=" * 72, flush=True)
        print("YOLO11 Attention 合同测试失败", flush=True)
        print("=" * 72, flush=True)
        print(f"错误类型         : {type(exc).__name__}", flush=True)
        print(f"原因             : {exc}", flush=True)
        print("调试方式         : 设置 OTO_DEBUG_TRACEBACK=1 可显示完整 traceback", flush=True)
        raise SystemExit(1)
