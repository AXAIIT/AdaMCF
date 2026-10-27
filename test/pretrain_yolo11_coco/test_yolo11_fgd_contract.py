#!/usr/bin/env python
"""官方 FGD FeatureLoss 的 mask、分项开关、数值和梯度合同单测。"""

from __future__ import annotations

import torch

from yolo11_fgd import (
    build_fgd_foreground_background_masks,
    full_fgd_feature_loss,
    make_official_fgd_modules,
)


WEIGHTS = {
    "foreground": 0.001,
    "background": 0.0005,
    "mask": 0.0005,
    "relation": 0.000005,
}


def _synchronize_official_branches(modules) -> None:
    """让师生辅助分支初值一致，以验证相同特征时官方四项损失为零。"""

    with torch.no_grad():
        for module in modules:
            module.conv_mask_s.load_state_dict(module.conv_mask_t.state_dict())
            module.channel_add_conv_s.load_state_dict(module.channel_add_conv_t.state_dict())


def _reset_gradients(modules) -> None:
    for parameter in modules.parameters():
        parameter.grad = None


def _loss(
    student,
    teacher,
    modules,
    batch,
    *,
    foreground: bool = True,
    background: bool = True,
    mask: bool = True,
    relation: bool = True,
):
    return full_fgd_feature_loss(
        student,
        teacher,
        modules,
        batch,
        foreground_weight=WEIGHTS["foreground"],
        background_weight=WEIGHTS["background"],
        mask_weight=WEIGHTS["mask"],
        relation_weight=WEIGHTS["relation"],
        enable_foreground=foreground,
        enable_background=background,
        enable_mask=mask,
        enable_relation=relation,
    )


def _expected_loss(metrics, flags: dict[str, bool]) -> torch.Tensor:
    return sum(
        (WEIGHTS[name] if flags[name] else 0.0) * metrics[f"fgd_{name}_loss"]
        for name in ("foreground", "background", "mask", "relation")
    )


def _assert_component_metrics(metrics, flags: dict[str, bool]) -> None:
    for name, enabled in flags.items():
        assert float(metrics[f"fgd_{name}_enabled"].item()) == float(enabled)
        if not enabled:
            assert float(metrics[f"fgd_{name}_loss"].item()) == 0.0
            assert float(metrics[f"fgd_{name}_effective_weight"].item()) == 0.0


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch = {
        "img": torch.zeros(2, 3, 64, 64, device=device),
        "batch_idx": torch.tensor([0, 0], device=device),
        # 第二个极小框在官方 floor/ceil 离散 mask 下依旧至少覆盖一个 feature cell。
        "bboxes": torch.tensor(
            [[0.50, 0.50, 0.25, 0.25], [0.13, 0.27, 0.0001, 0.0001]],
            device=device,
        ),
    }
    shapes = [(8, 8), (4, 4), (2, 2)]
    foreground, background, statistics = build_fgd_foreground_background_masks(
        batch,
        shapes,
        device=device,
        dtype=torch.float32,
        batch_size=2,
    )
    assert len(foreground) == len(background) == 3
    assert all(mask.shape == (2, *shape) and mask.dtype == torch.float32 for mask, shape in zip(foreground, shapes))
    assert all(torch.all(mask >= 0) for mask in foreground + background)
    for mask in background:
        background_mass = mask.sum(dim=(1, 2))
        # 官方实现若某尺度被前景完全覆盖，保留全零背景 mask；否则背景质量归一化为 1。
        assert bool(
            torch.all(
                torch.isclose(background_mass, torch.zeros_like(background_mass))
                | torch.isclose(background_mass, torch.ones_like(background_mass))
            ).item()
        )
    assert sum(int((mask[0] > 0).sum().item()) for mask in foreground) > 0
    assert sum(int((mask[1] > 0).sum().item()) for mask in foreground) == 0
    assert float(statistics["fgd_gt_total"].item()) == 2.0
    assert float(statistics["fgd_foreground_cell_ratio"].item()) > 0.0

    contract = {
        "feature_levels": [
            {"student_channels": 8, "teacher_channels": 8},
            {"student_channels": 8, "teacher_channels": 8},
            {"student_channels": 8, "teacher_channels": 8},
        ]
    }
    modules = make_official_fgd_modules(contract, temperature=0.5).to(device)
    _synchronize_official_branches(modules)
    teacher = [torch.randn(2, 8, height, width, device=device) for height, width in shapes]
    identical = [value.detach().clone().requires_grad_(True) for value in teacher]
    zero_loss, zero_metrics = _loss(identical, teacher, modules, batch)
    assert float(zero_loss.detach().item()) < 1e-7
    assert float(zero_metrics["fgd_mask_loss"].item()) < 1e-7
    assert float(zero_metrics["fgd_relation_loss"].item()) < 1e-7

    # 1) 默认全开必须和显式四项全开完全等价，避免破坏已验证的完整 FGD 路径。
    base_student = [value.detach() + 0.1 * torch.randn_like(value) for value in teacher]
    default_student = [value.detach().clone().requires_grad_(True) for value in base_student]
    explicit_student = [value.detach().clone().requires_grad_(True) for value in base_student]
    default_loss, default_metrics = _loss(default_student, teacher, modules, batch)
    explicit_loss, explicit_metrics = _loss(
        explicit_student,
        teacher,
        modules,
        batch,
        foreground=True,
        background=True,
        mask=True,
        relation=True,
    )
    torch.testing.assert_close(default_loss, explicit_loss, rtol=0.0, atol=0.0)
    for name in ("foreground", "background", "mask", "relation", "feature"):
        torch.testing.assert_close(
            default_metrics[f"fgd_{name}_loss"],
            explicit_metrics[f"fgd_{name}_loss"],
            rtol=0.0,
            atol=0.0,
        )
    _assert_component_metrics(default_metrics, {name: True for name in WEIGHTS})
    assert bool(torch.isfinite(default_loss).item()) and float(default_loss.detach().item()) > 0.0
    default_loss.backward()
    assert all(
        feature.grad is not None
        and bool(torch.isfinite(feature.grad).all().item())
        and int(torch.count_nonzero(feature.grad).item()) > 0
        for feature in default_student
    )
    auxiliary_nonzero_gradients = sum(
        int(parameter.grad is not None and int(torch.count_nonzero(parameter.grad).item()) > 0)
        for parameter in modules.parameters()
    )
    assert auxiliary_nonzero_gradients > 0

    # 2) 累加式消融：F1=foreground，F2=foreground+background，F3=foreground+background+mask。
    # 每个组合都直接按启用项的官方权重求和；未启用项既不进入损失，也不应激活关系分支。
    component_plans = {
        "F1_foreground": {"foreground": True, "background": False, "mask": False, "relation": False},
        "F2_foreground_background": {"foreground": True, "background": True, "mask": False, "relation": False},
        "F3_foreground_background_mask": {"foreground": True, "background": True, "mask": True, "relation": False},
    }
    for plan_name, flags in component_plans.items():
        _reset_gradients(modules)
        student = [value.detach().clone().requires_grad_(True) for value in base_student]
        loss, metrics = _loss(student, teacher, modules, batch, **flags)
        torch.testing.assert_close(loss, _expected_loss(metrics, flags), rtol=1e-6, atol=1e-8)
        _assert_component_metrics(metrics, flags)
        loss.backward()
        assert all(
            feature.grad is not None
            and bool(torch.isfinite(feature.grad).all().item())
            and int(torch.count_nonzero(feature.grad).item()) > 0
            for feature in student
        ), plan_name
        if plan_name == "F1_foreground":
            relation_parameters = [
                *[parameter for module in modules for parameter in module.conv_mask_s.parameters()],
                *[parameter for module in modules for parameter in module.conv_mask_t.parameters()],
                *[parameter for module in modules for parameter in module.channel_add_conv_s.parameters()],
                *[parameter for module in modules for parameter in module.channel_add_conv_t.parameters()],
            ]
            assert all(parameter.grad is None for parameter in relation_parameters)

    print("PASS: official FGD full-path parity, component switches and gradient contracts", flush=True)


if __name__ == "__main__":
    main()
