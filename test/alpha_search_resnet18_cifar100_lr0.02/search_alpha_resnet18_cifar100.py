'''import os
import json
import time
import csv
import itertools
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = "/workspace/OTOv2_v2/test/pretrain_resnet18_cifar100/00.resnet18_cifar100_pruning_loss_search_alpha.py"

def run_one(alpha_cfg, work_dir: Path):
    """
    alpha_cfg: dict, 如 {"lr": 0.08, "eps": 0.015, "max_step_l1": 0.15, "g_ema_gamma": 0.5}
    """
    tag = "_".join(f"{k}{v}" for k, v in alpha_cfg.items())
    exp_name = f"alpha_{tag}"
    exp_dir = work_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    result_path = exp_dir / "result.json"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    # 把要搜索的超参与环境变量绑定
    if "lr" in alpha_cfg:
        env["OTO_ALPHA_LR"] = str(alpha_cfg["lr"])
    if "eps" in alpha_cfg:
        env["OTO_ALPHA_EPS"] = str(alpha_cfg["eps"])
    if "max_step_l1" in alpha_cfg:
        env["OTO_ALPHA_MAX_STEP_L1"] = str(alpha_cfg["max_step_l1"])
    if "g_ema_gamma" in alpha_cfg:
        env["OTO_ALPHA_G_EMA_GAMMA"] = str(alpha_cfg["g_ema_gamma"])

    # 告诉训练脚本把结果写到这里
    env["OTO_RESULT_PATH"] = str(result_path)

    cmd = ["python", str(TRAIN_SCRIPT)]
    print(f"\n[RUN] {exp_name}")
    print("CMD:", " ".join(cmd))
    print("ALPHA_CFG:", alpha_cfg)

    t0 = time.time()
    subprocess.run(cmd, check=True, env=env)
    elapsed = time.time() - t0

    if not result_path.is_file():
        raise RuntimeError(f"Expected result file not found: {result_path}")

    with open(result_path, "r") as f:
        metrics = json.load(f)

    metrics["exp_name"] = exp_name
    metrics["elapsed_sec"] = elapsed
    metrics["alpha_cfg"] = alpha_cfg
    return metrics

def main():
    work_dir = Path(__file__).resolve().parent
    work_dir.mkdir(parents=True, exist_ok=True)

    # === 搜索空间（可按需调整） ===
    lr_list = [0.001, 0.003, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.30, 0.90]
    eps_list = [0.010, 0.0125, 0.015, 0.0175, 0.020]
    max_step_l1_list = [0.10, 0.15, 0.20]
    g_ema_list       = [0.0, 0.3, 0.5]

    all_rows = []

    # 第一阶段：只调 lr
    results_stage1 = []
    for lr in lr_list:
        cfg = {
            "lr": lr,
            "eps": 0.015,
            "max_step_l1": 0.15,
            "g_ema_gamma": 0.5,
        }
        res = run_one(cfg, work_dir)
        results_stage1.append(res)
        with open(work_dir / "summary_stage1.json", "w") as f:
            json.dump(results_stage1, f, indent=2)

        all_rows.append({
            "stage": "stage1",
            "lr": cfg["lr"],
            "eps": cfg["eps"],
            "max_step_l1": cfg["max_step_l1"],
            "g_ema_gamma": cfg["g_ema_gamma"],
            # 精度
            "acc1_before": res.get("acc1_before", 0.0),
            "acc5_before": res.get("acc5_before", 0.0),
            "acc1_after": res.get("acc1_after", 0.0),
            "acc5_after": res.get("acc5_after", 0.0),
            # 稀疏度与组统计
            "final_group_sparsity": res.get("final_group_sparsity", 0.0),
            "num_groups": res.get("num_groups", 0),
            "num_zero_groups": res.get("num_zero_groups", 0),
            "zero_group_ratio": res.get("zero_group_ratio", 0.0),
            "num_pruned_groups": res.get("num_pruned_groups", 0),
            "pruned_group_ratio": res.get("pruned_group_ratio", 0.0),
            "num_active_redundant_groups": res.get("num_active_redundant_groups", 0),
            "active_redundant_group_ratio": res.get("active_redundant_group_ratio", 0.0),
            "num_redundant_groups": res.get("num_redundant_groups", 0),
            "redundant_group_ratio": res.get("redundant_group_ratio", 0.0),
            "num_important_groups": res.get("num_important_groups", 0),
            "important_group_ratio": res.get("important_group_ratio", 0.0),
            "norm_params": res.get("norm_params", 0.0),
            # FLOPs / Params 与压缩
            "flops_before_m": res.get("flops_before_m", 0.0),
            "flops_after_m": res.get("flops_after_m", 0.0),
            "pruned_flops_m": res.get("pruned_flops_m", 0.0),
            "pruned_flops_percent": res.get("pruned_flops_percent", 0.0),
            "flops_compression_ratio": res.get("flops_compression_ratio", 0.0),
            "params_before_m": res.get("params_before_m", 0.0),
            "params_after_m": res.get("params_after_m", 0.0),
            "pruned_params_m": res.get("pruned_params_m", 0.0),
            "pruned_params_percent": res.get("pruned_params_percent", 0.0),
            "params_compression_ratio": res.get("params_compression_ratio", 0.0),
            # 时间
            "elapsed_sec": res.get("elapsed_sec", 0.0),
        })

    best1 = max(results_stage1, key=lambda x: x.get("acc1_after", 0.0))
    print("\n[Stage1] Best:", best1["alpha_cfg"], "acc1_after=", best1.get("acc1_after"))

    # 第二阶段：围绕 best1 的 lr 细调其他超参
    base_lr = best1["alpha_cfg"]["lr"]
    results_stage2 = []
    for eps, ms, g_ema in itertools.product(eps_list, max_step_l1_list, g_ema_list):
        cfg = {
            "lr": base_lr,
            "eps": eps,
            "max_step_l1": ms,
            "g_ema_gamma": g_ema,
        }
        res = run_one(cfg, work_dir)
        results_stage2.append(res)
        with open(work_dir / "summary_stage2.json", "w") as f:
            json.dump(results_stage2, f, indent=2)

        all_rows.append({
            "stage": "stage2",
            "lr": cfg["lr"],
            "eps": cfg["eps"],
            "max_step_l1": cfg["max_step_l1"],
            "g_ema_gamma": cfg["g_ema_gamma"],
            "acc1_before": res.get("acc1_before", 0.0),
            "acc5_before": res.get("acc5_before", 0.0),
            "acc1_after": res.get("acc1_after", 0.0),
            "acc5_after": res.get("acc5_after", 0.0),
            "final_group_sparsity": res.get("final_group_sparsity", 0.0),
            "num_groups": res.get("num_groups", 0),
            "num_zero_groups": res.get("num_zero_groups", 0),
            "zero_group_ratio": res.get("zero_group_ratio", 0.0),
            "num_pruned_groups": res.get("num_pruned_groups", 0),
            "pruned_group_ratio": res.get("pruned_group_ratio", 0.0),
            "num_active_redundant_groups": res.get("num_active_redundant_groups", 0),
            "active_redundant_group_ratio": res.get("active_redundant_group_ratio", 0.0),
            "num_redundant_groups": res.get("num_redundant_groups", 0),
            "redundant_group_ratio": res.get("redundant_group_ratio", 0.0),
            "num_important_groups": res.get("num_important_groups", 0),
            "important_group_ratio": res.get("important_group_ratio", 0.0),
            "norm_params": res.get("norm_params", 0.0),
            "flops_before_m": res.get("flops_before_m", 0.0),
            "flops_after_m": res.get("flops_after_m", 0.0),
            "pruned_flops_m": res.get("pruned_flops_m", 0.0),
            "pruned_flops_percent": res.get("pruned_flops_percent", 0.0),
            "flops_compression_ratio": res.get("flops_compression_ratio", 0.0),
            "params_before_m": res.get("params_before_m", 0.0),
            "params_after_m": res.get("params_after_m", 0.0),
            "pruned_params_m": res.get("pruned_params_m", 0.0),
            "pruned_params_percent": res.get("pruned_params_percent", 0.0),
            "params_compression_ratio": res.get("params_compression_ratio", 0.0),
            "elapsed_sec": res.get("elapsed_sec", 0.0),
        })

    best2 = max(results_stage2, key=lambda x: x.get("acc1_after", 0.0))
    print("\n[Stage2] Best:", best2["alpha_cfg"], "acc1_after=", best2.get("acc1_after"))

    final = {
        "stage1_best": {
            "alpha_cfg": best1["alpha_cfg"],
            "acc1_after": best1.get("acc1_after"),
            "elapsed_sec": best1.get("elapsed_sec"),
        },
        "stage2_best": {
            "alpha_cfg": best2["alpha_cfg"],
            "acc1_after": best2.get("acc1_after"),
            "elapsed_sec": best2.get("elapsed_sec"),
        },
    }
    with open(work_dir / "final_best.json", "w") as f:
        json.dump(final, f, indent=2)

    # === 导出 CSV ===
    csv_path = work_dir / "alpha_search_results.csv"
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"\n[Table] CSV saved to: {csv_path}")

    # === 导出 Markdown 表（方便直接贴文档） ===
    md_path = work_dir / "alpha_search_results.md"
    with open(md_path, "w") as f:
        # 简化一下，只展示最重要的列
        f.write("| stage | lr | eps | max_step_l1 | g_ema_gamma | acc1_before | acc1_after | flops_before(M) | flops_after(M) | pruned_flops% | params_before(M) | params_after(M) | pruned_params% | sparsity |\n")
        f.write("|-------|----|-----|-------------|-------------|-------------|------------|-----------------|----------------|---------------|------------------|-----------------|---------------|----------|\n")
        for r in all_rows:
            f.write(
                f"| {r['stage']} | {r['lr']:.3f} | {r['eps']:.3f} | {r['max_step_l1']:.2f} | {r['g_ema_gamma']:.2f} | "
                f"{r['acc1_before']:.2f} | {r['acc1_after']:.2f} | "
                f"{r['flops_before_m']:.2f} | {r['flops_after_m']:.2f} | {r['pruned_flops_percent']:.2f} | "
                f"{r['params_before_m']:.2f} | {r['params_after_m']:.2f} | {r['pruned_params_percent']:.2f} | "
                f"{r['final_group_sparsity']:.3f} |\n"
            )
    print(f"[Table] Markdown saved to: {md_path}")

    print("\n[Final] Best alpha cfg:")
    print(json.dumps(final, indent=2))

if __name__ == "__main__":
    main()'''


import os
import json
import time
import csv
import itertools
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = "/workspace/OTOv2_v2/test/pretrain_resnet18_cifar100/00.resnet18_cifar100_pruning_loss_search_alpha.py"

def run_one(alpha_cfg, work_dir: Path):
    tag = "_".join(f"{k}{v}" for k, v in alpha_cfg.items())
    exp_name = f"alpha_{tag}"
    exp_dir = work_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    result_path = exp_dir / "result.json"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    if "lr" in alpha_cfg:
        env["OTO_ALPHA_LR"] = str(alpha_cfg["lr"])
    if "eps" in alpha_cfg:
        env["OTO_ALPHA_EPS"] = str(alpha_cfg["eps"])
    if "max_step_l1" in alpha_cfg:
        env["OTO_ALPHA_MAX_STEP_L1"] = str(alpha_cfg["max_step_l1"])
    if "g_ema_gamma" in alpha_cfg:
        env["OTO_ALPHA_G_EMA_GAMMA"] = str(alpha_cfg["g_ema_gamma"])

    env["OTO_RESULT_PATH"] = str(result_path)

    cmd = ["python", str(TRAIN_SCRIPT)]
    print(f"\n[RUN] {exp_name}")
    print("CMD:", " ".join(cmd))
    print("ALPHA_CFG:", alpha_cfg)

    t0 = time.time()
    subprocess.run(cmd, check=True, env=env)
    elapsed = time.time() - t0

    if not result_path.is_file():
        raise RuntimeError(f"Expected result file not found: {result_path}")

    with open(result_path, "r") as f:
        metrics = json.load(f)

    metrics["exp_name"] = exp_name
    metrics["elapsed_sec"] = elapsed
    metrics["alpha_cfg"] = alpha_cfg
    return metrics

def main():
    work_dir = PROJECT_ROOT / "alpha_search_resnet18_cifar100_lr0.02_gema0.5_ms0.18"
    work_dir.mkdir(parents=True, exist_ok=True)

    # 固定 lr=0.02, g_ema_gamma=0.5, max_step_l1=0.18，只搜索 eps
    fixed_lr = 0.02
    fixed_g_ema = 0.5
    fixed_ms = 0.18

    eps_list = [0.010, 0.0125, 0.015, 0.0175, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.060, 0.070, 0.080]

    all_rows = []
    results = []

    for eps in eps_list:
        cfg = {
            "lr": fixed_lr,
            "eps": eps,
            "max_step_l1": fixed_ms,
            "g_ema_gamma": fixed_g_ema,
        }
        res = run_one(cfg, work_dir)
        results.append(res)

        all_rows.append({
            "stage": "eps_only",
            "lr": cfg["lr"],
            "eps": cfg["eps"],
            "max_step_l1": cfg["max_step_l1"],
            "g_ema_gamma": cfg["g_ema_gamma"],
            "acc1_before": res.get("acc1_before", 0.0),
            "acc5_before": res.get("acc5_before", 0.0),
            "acc1_after": res.get("acc1_after", 0.0),
            "acc5_after": res.get("acc5_after", 0.0),
            "final_group_sparsity": res.get("final_group_sparsity", 0.0),
            "flops_before_m": res.get("flops_before_m", 0.0),
            "flops_after_m": res.get("flops_after_m", 0.0),
            "pruned_flops_m": res.get("pruned_flops_m", 0.0),
            "pruned_flops_percent": res.get("pruned_flops_percent", 0.0),
            "flops_compression_ratio": res.get("flops_compression_ratio", 0.0),
            "params_before_m": res.get("params_before_m", 0.0),
            "params_after_m": res.get("params_after_m", 0.0),
            "pruned_params_m": res.get("pruned_params_m", 0.0),
            "pruned_params_percent": res.get("pruned_params_percent", 0.0),
            "params_compression_ratio": res.get("params_compression_ratio", 0.0),
            "elapsed_sec": res.get("elapsed_sec", 0.0),
        })

        with open(work_dir / "summary_eps.json", "w") as f:
            json.dump(results, f, indent=2)

    best = max(results, key=lambda x: x.get("acc1_after", 0.0))
    print("\n[Search] Best (lr=0.02, g_ema_gamma=0.5, max_step_l1=0.18):",
          best["alpha_cfg"], "acc1_after=", best.get("acc1_after"))

    final = {
        "best_cfg": best["alpha_cfg"],
        "acc1_after": best.get("acc1_after"),
        "elapsed_sec": best.get("elapsed_sec"),
    }
    with open(work_dir / "final_best.json", "w") as f:
        json.dump(final, f, indent=2)

    csv_path = work_dir / "alpha_search_results_lr0.02_gema0.5_ms0.18.csv"
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"\n[Table] CSV saved to: {csv_path}")

    md_path = work_dir / "alpha_search_results_lr0.02_gema0.5_ms0.18.md"
    with open(md_path, "w") as f:
        f.write("| lr | eps | max_step_l1 | g_ema_gamma | acc1_after | flops_after(M) | pruned_flops% |\n")
        f.write("|----|-----|-------------|-------------|------------|----------------|---------------|\n")
        for r in all_rows:
            f.write(
                f"| {r['lr']:.3f} | {r['eps']:.4f} | {r['max_step_l1']:.2f} | {r['g_ema_gamma']:.2f} | "
                f"{r['acc1_after']:.2f} | {r['flops_after_m']:.2f} | {r['pruned_flops_percent']:.2f} |\n"
            )
    print(f"[Table] Markdown saved to: {md_path}")

    print("\n[Final] Best alpha cfg (lr=0.02, g_ema_gamma=0.5, max_step_l1=0.18):")
    print(json.dumps(final, indent=2))

if __name__ == "__main__":
    main()

