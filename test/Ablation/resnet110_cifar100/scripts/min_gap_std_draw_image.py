import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt


DEFAULT_SELECTED = [0.05, 0.10, 0.15, 0.25, 0.50, 1.00]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = SCRIPT_DIR / "ablation_min_gap_std.txt"
DEFAULT_OUT_PATH = SCRIPT_DIR / "ablation_min_gap_std_selected.png"


def read_ablation_log(log_path: Path) -> dict[float, float]:
    """
    读取 min_gap_std.py 生成的日志文件 ablation_min_gap_std.txt
    格式：
      min_gap_std,final_accuracy
      0.05,xx.xx
      ...
    返回：{min_gap_std(float): final_accuracy(float)}
    """
    if not log_path.exists():
        raise FileNotFoundError(f"未找到日志文件: {log_path}")

    results: dict[float, float] = {}
    with log_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "min_gap_std" not in reader.fieldnames or "final_accuracy" not in reader.fieldnames:
            raise ValueError(
                f"日志表头不匹配，期望包含 min_gap_std 与 final_accuracy，实际为: {reader.fieldnames}"
            )

        for row in reader:
            if not row:
                continue
            k = row.get("min_gap_std", "").strip()
            v = row.get("final_accuracy", "").strip()
            if k == "" or v == "":
                continue
            try:
                results[float(k)] = float(v)
            except ValueError:
                # 跳过无法解析的行
                continue

    return results


def plot_selected_points(
    results_map: dict[float, float],
    selected_x: list[float],
    out_path: Path,
    title: str = "",
):
    xs = []
    ys = []
    missing = []

    for x in selected_x:
        if x in results_map:
            xs.append(x)
            ys.append(results_map[x])
        else:
            missing.append(x)

    if not xs:
        raise RuntimeError(
            "在日志中未找到任何选定的 min_gap_std 点，请检查日志内容或 selected 列表。"
        )

    if missing:
        print(f"[警告] 日志中缺少这些点，将跳过：{missing}")

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, marker="o", linestyle="-", color="b")
    plt.title(title)
    plt.xlabel("min_gap_std")
    plt.ylabel("Final Accuracy (%)")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.xticks(xs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"已保存图片: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--log",
        type=str,
        default=str(DEFAULT_LOG_PATH),
        help=f"ablation 日志路径（默认：{DEFAULT_LOG_PATH}）",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT_PATH),
        help=f"输出图片路径（默认：{DEFAULT_OUT_PATH}）",
    )
    p.add_argument(
        "--selected",
        type=float,
        nargs="*",
        default=DEFAULT_SELECTED,
        help="需要绘制的 min_gap_std 点（默认：0.05 0.10 0.15 0.25 0.50 1.00）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    log_path = Path(args.log).resolve()
    out_path = Path(args.out).resolve()

    results_map = read_ablation_log(log_path)

    selected = list(args.selected)
    selected.sort()

    plot_selected_points(
        results_map=results_map,
        selected_x=selected,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()