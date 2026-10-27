#!/usr/bin/env python3
"""通用 segmented-Concat 物理裁剪合同的最小端到端回归测试。

该测试刻意不依赖 YOLO 或任何层路径：两个独立 Conv 输出按 channel 维拼接，
再由一个普通 Conv 消费。合同只由 branch_a 显式 source 触发，但 branch_b 仍是
可剪输出轴；它验证 Graph 能冻结 Concat 的输入顺序和原始 offset，并在物理构网
时将显式与被动生产者的局部冗余索引一次性映射到同一个消费者 input-axis。若该
测试失败，YOLO11 C3k2 的 output-consumer profile 不应进入 audit/prune。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from only_train_once import OTO


class TwoBranchChannelConcat(nn.Module):
    """两个独立生产者 -> channel concat -> 普通 Conv 消费者。"""

    def __init__(self):
        super().__init__()
        self.branch_a = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.branch_b = nn.Conv2d(3, 5, kernel_size=1, bias=False)
        self.consumer = nn.Conv2d(9, 6, kernel_size=1, bias=False)

    def forward(self, x):
        return self.consumer(torch.cat((self.branch_a(x), self.branch_b(x)), dim=1))


def _single_owner(graph, parameter_name: str):
    owners = [
        group
        for group in graph.node_groups.values()
        if parameter_name in group.param_names
    ]
    if len(owners) != 1:
        raise AssertionError(
            f"参数 {parameter_name} 必须有唯一 OTO 逻辑所有者，实际为 {len(owners)}。"
        )
    return owners[0]


def run(output: Path):
    torch.manual_seed(7)
    model = TwoBranchChannelConcat().train()
    dummy_input = torch.randn(2, 3, 8, 8)
    oto = OTO(model=model, dummy_input=dummy_input)

    registration = oto._graph.register_segmented_concat_consumer_contracts(
        producer_parameter_names=("branch_a.weight",),
    )
    contracts = registration.get("contracts", [])
    if len(contracts) != 1 or contracts[0].get("status") != "configured":
        raise AssertionError(
            "segmented Concat 合同必须配置成功："
            f"{json.dumps(registration, ensure_ascii=False, sort_keys=True)}"
        )
    contract = contracts[0]
    segments = contract.get("segments", [])
    expected_ledger = [
        ("branch_a.weight", 0, 4),
        ("branch_b.weight", 4, 5),
    ]
    actual_ledger = [
        (
            segment.get("producer_parameter_name"),
            segment.get("original_offset"),
            segment.get("original_width"),
        )
        for segment in segments
    ]
    if actual_ledger != expected_ledger:
        raise AssertionError(
            f"Concat 段账本与真实 torch.cat 输入顺序不一致：{actual_ledger}"
        )
    if (
        not segments[0].get("is_active_owner")
        or segments[0].get("is_passive_output_owner")
        or segments[1].get("is_active_owner")
        or not segments[1].get("is_passive_output_owner")
    ):
        raise AssertionError(
            "Concat 合同没有区分显式 source 与可剪被动输出段："
            f"{segments}"
        )

    # 同时设置显式 source a 与被动 b 的输出通道为零：a 删除 {1, 3}，b 删除
    # {0, 4}。consumer 的输入轴必须删除原始坐标 {1, 3, 4, 8}，并且只能裁一次。
    with torch.no_grad():
        # ``weight[[...]].zero_()`` 是 advanced-indexing 的临时副本，不能真正
        # 改写原参数；必须显式沿输出轴 index_fill_。
        model.branch_a.weight.index_fill_(
            0,
            torch.tensor([1, 3], dtype=torch.long, device=model.branch_a.weight.device),
            0.0,
        )
        model.branch_b.weight.index_fill_(
            0,
            torch.tensor([0, 4], dtype=torch.long, device=model.branch_b.weight.device),
            0.0,
        )
    if not torch.count_nonzero(model.branch_a.weight[[1, 3]]).item() == 0:
        raise AssertionError("branch_a 的目标输出通道没有被真实置零。")
    if not torch.count_nonzero(model.branch_b.weight[[0, 4]]).item() == 0:
        raise AssertionError("branch_b 的目标输出通道没有被真实置零。")

    owner_a = _single_owner(oto._graph, "branch_a.weight")
    owner_b = _single_owner(oto._graph, "branch_b.weight")
    if not owner_a.is_prunable or not owner_b.is_prunable:
        raise AssertionError("测试生产者必须处于可剪状态。")

    output.mkdir(parents=True, exist_ok=True)
    oto.construct_subnet(
        out_dir=str(output),
        save_full_group_sparse_model=False,
        ckpt_format="torch",
    )

    if model.branch_a.out_channels != 2 or model.branch_b.out_channels != 3:
        raise AssertionError(
            "生产者输出通道未按零组裁剪："
            f"a={model.branch_a.out_channels}, b={model.branch_b.out_channels}"
        )
    if model.consumer.in_channels != 5 or model.consumer.weight.shape[1] != 5:
        raise AssertionError(
            "Concat 消费者 input-axis 未按两个 segment 的联合冗余索引裁剪："
            f"in_channels={model.consumer.in_channels}, weight={tuple(model.consumer.weight.shape)}"
        )

    runtime = list(getattr(oto._graph, "segmented_concat_contract_runtime", []))
    if len(runtime) != 1 or runtime[0].get("status") != "pass":
        raise AssertionError(f"segmented Concat 物理构网执行记录异常：{runtime}")
    if runtime[0].get("pruned_input_indices_original_coordinate") != [1, 3, 4, 8]:
        raise AssertionError(
            "消费者输入轴没有使用原始 Concat 坐标："
            f"{runtime[0].get('pruned_input_indices_original_coordinate')}"
        )

    # 保存后的物理子网须能独立重载并前向，避免只验证内存对象。
    try:
        reloaded = torch.load(
            oto.compressed_model_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:  # 兼容尚未支持 weights_only 参数的旧 PyTorch。
        reloaded = torch.load(oto.compressed_model_path, map_location="cpu")
    reloaded.eval()
    with torch.no_grad():
        reloaded_output = reloaded(torch.randn(2, 3, 8, 8))
    if tuple(reloaded_output.shape) != (2, 6, 8, 8):
        raise AssertionError(
            f"重载子网前向输出形状异常：{tuple(reloaded_output.shape)}"
        )

    # 子网须能真实前向、反向和一次优化器更新，避免只有形状账本正确。
    model.train()
    output_tensor = model(torch.randn(2, 3, 8, 8))
    loss = output_tensor.square().mean()
    loss.backward()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.step()

    report = {
        "status": "pass",
        "contract": contract,
        "runtime": runtime,
        "post_prune_shapes": {
            "branch_a_weight": list(model.branch_a.weight.shape),
            "branch_b_weight": list(model.branch_b.weight.shape),
            "consumer_weight": list(model.consumer.weight.shape),
        },
        "forward_shape": list(output_tensor.shape),
        "reloaded_forward_shape": list(reloaded_output.shape),
        "compressed_model_path": str(oto.compressed_model_path),
    }
    report_path = output / "segmented_concat_contract_unit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[pass] 通用 segmented-Concat 合同测试通过：{report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/segmented_concat_contract_unit"),
        help="测试报告和 OTO 子网文件的输出目录。",
    )
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
