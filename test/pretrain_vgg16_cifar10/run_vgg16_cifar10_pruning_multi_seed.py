import os
import json
import numpy as np

from vgg16_cifar10_pruning_0_8_loss import prune_model, finetune_model

def run_experiments(seeds=(42, 43, 44)):
    all_prune_results = []
    all_finetune_results = []

    for i, seed in enumerate(seeds):
        print("\n" + "=" * 80)
        print(f"开始第 {i+1}/{len(seeds)} 次实验，seed = {seed}")
        print("=" * 80)

        prune_res = prune_model(seed=seed)
        finetune_res = finetune_model(seed=seed)

        all_prune_results.append(prune_res)
        all_finetune_results.append(finetune_res)

    return all_prune_results, all_finetune_results


def _mean_std(values):
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def summarize_results(all_prune_results, all_finetune_results, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "#" * 80)
    print("三次实验结果汇总（用于论文统计）")
    print("#" * 80)

    # ---------- 剪枝阶段汇总 ----------
    print("\n[剪枝阶段统计]")

    keys_for_prune = [
        "accuracy1_before", "accuracy5_before",
        "accuracy1_after",  "accuracy5_after",
        "flops_before_m",   "flops_after_m",
        "params_before_m",  "params_after_m",
        "final_group_sparsity",
        "pruned_flops_m", "pruned_flops_percent",
        "pruned_params_m", "pruned_params_percent",
        "flops_compression_ratio", "params_compression_ratio",
        "latency_before", "latency_after", "speedup"
    ]

    prune_summary = {}
    for key in keys_for_prune:
        vals = [r[key] for r in all_prune_results if r.get(key) is not None]
        if not vals:
            continue
        mean, std = _mean_std(vals)
        prune_summary[key] = {"mean": mean, "std": std}
        print(f"{key}: mean={mean:.4f}, std={std:.4f}")

    # ---------- 微调阶段汇总 ----------
    print("\n[微调阶段统计]")

    keys_for_ft = [
        "finetune_top1_final", "finetune_top5_final",
        "baseline_top1", "baseline_top5",
        "delta_top1_vs_baseline", "delta_top5_vs_baseline",
    ]

    finetune_summary = {}
    for key in keys_for_ft:
        vals = [r[key] for r in all_finetune_results if r.get(key) is not None]
        if not vals:
            continue
        mean, std = _mean_std(vals)
        finetune_summary[key] = {"mean": mean, "std": std}
        print(f"{key}: mean={mean:.4f}, std={std:.4f}")

    # ---------- 保存到 JSON，便于论文表格整理 ----------
    summary = {
        "num_runs": len(all_prune_results),
        "prune_results": all_prune_results,
        "finetune_results": all_finetune_results,
        "prune_summary": prune_summary,
        "finetune_summary": finetune_summary,
    }

    out_path = os.path.join(save_dir, "vgg16_cifar10_pruning_sparsity_0.8_multi_seed_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n完整统计结果已保存到: {out_path}")


def main():
    # 你可以在这里换成其他种子列表
    seeds = [42, 0, 1]
    all_prune_results, all_finetune_results = run_experiments(seeds=seeds)

    save_dir = os.path.join(
        os.path.dirname(__file__),
        "multi_seed_stats_sparsity_0_8"
    )
    summarize_results(all_prune_results, all_finetune_results, save_dir)


if __name__ == "__main__":
    main()