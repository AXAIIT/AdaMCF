import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _is_close(a: float, b: float, eps: float = 1e-9) -> bool:
    return abs(a - b) <= eps


def main():
    script_dir = Path(__file__).resolve().parent
    txt_path = script_dir / "ablation_eval_drop_ratio.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"找不到日志文件: {txt_path}")

    drop_points = {0.4, 0.6}

    xs, ys = [], []
    with txt_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row["eval_drop_ratio"])
            y = float(row["final_accuracy"])
            if any(_is_close(x, dp) for dp in drop_points):
                continue
            xs.append(x)
            ys.append(y)

    # 按 x 排序，保证折线连接顺序正确
    pairs = sorted(zip(xs, ys), key=lambda t: t[0])
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, marker="o", linestyle="-", color="b")
    # 不需要图标题：不调用 plt.title(...)
    plt.xlabel("eval_drop_ratio")
    plt.ylabel("Final Accuracy (%)")
    plt.grid(True)
    plt.xticks(xs)

    out_path = script_dir / "ablation_eval_drop_ratio.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()