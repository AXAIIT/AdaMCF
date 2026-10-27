import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import rcParams
from matplotlib import font_manager as fm
from matplotlib.patches import Patch

def _set_style():
    times_new_roman_candidates = [
        "/root/.local/share/fonts/msttfonts/times.ttf",
        "/root/.local/share/fonts/msttfonts/timesi.ttf",
        "/root/.local/share/fonts/msttfonts/timesbd.ttf",
        "/root/.local/share/fonts/msttfonts/timesbi.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/timesi.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/timesbi.ttf",
    ]
    for font_path in times_new_roman_candidates:
        if os.path.exists(font_path):
            try:
                fm.fontManager.addfont(font_path)
            except Exception:
                continue

    rcParams["font.family"] = ["Times New Roman"]
    rcParams["axes.labelsize"] = 20
    rcParams["xtick.labelsize"] = 20
    rcParams["ytick.labelsize"] = 20
    rcParams["axes.unicode_minus"] = False

_set_style()

# =========================
# 1) 实验名 & 结果
# =========================
experiments = [
    "Uniform_Weights", "Weights_00", "Weights_01", "L1_Biased",
    "L2_Biased", "Fisher_Biased", "GradWeight_Biased", "Cosine_Biased",
    "AdaMCF"
]
x_labels = ["Uniform", "W0", "W1", r"$L_1$", r"$L_2$", "Fisher", "GW", "Cosine", "AdaMCF"]

base_accuracies = [70.85, 71.01, 70.92, 67.49, 68.83, 70.93, 70.72, 70.23, 71.53]

colors = [
    "#000000", "#FF0000", "#006400", "#FF8C00", "#8B0000", "#0000FF", "#00FFFF", "#800080",
    "#C1DE17"  # AdaMCF：灰色（可按需换成更醒目的颜色）
]

# =========================
# 2) 融合比例（顺序：L1/L2/Cos/Fisher/GW）
# =========================
criteria_weights = {
    "Uniform_Weights":     {"l1": 0.2, "l2": 0.2, "cos": 0.2, "fisher": 0.2, "gw": 0.2},
    "Weights_00":          {"l1": 0.1, "l2": 0.1, "cos": 0.2, "fisher": 0.2, "gw": 0.4},
    "Weights_01":          {"l1": 0.0, "l2": 0.1, "cos": 0.3, "fisher": 0.3, "gw": 0.3},
    "L1_Biased":           {"l1": 0.5, "l2": 0.1, "cos": 0.1, "fisher": 0.1, "gw": 0.2},
    "L2_Biased":           {"l1": 0.1, "l2": 0.5, "cos": 0.1, "fisher": 0.1, "gw": 0.2},
    "Fisher_Biased":       {"l1": 0.1, "l2": 0.1, "cos": 0.1, "fisher": 0.6, "gw": 0.1},
    "GradWeight_Biased":   {"l1": 0.1, "l2": 0.1, "cos": 0.1, "fisher": 0.1, "gw": 0.6},
    "Cosine_Biased":       {"l1": 0.1, "l2": 0.1, "cos": 0.6, "fisher": 0.1, "gw": 0.1},
    "AdaMCF":              None,  # 自适应/学习得到，不固定比例
}

def format_ratio(exp_name: str) -> str:
    """更紧凑、保证单行：L1/L2/C/F/GW；AdaMCF 显示为 adaptive。"""
    w = criteria_weights.get(exp_name)
    if w is None:
        return r"$\alpha$=adaptive"
    return f"$\\alpha$={w['l1']:.1f}/{w['l2']:.1f}/{w['cos']:.1f}/{w['fisher']:.1f}/{w['gw']:.1f}"

# =========================
# 3) 作图：模仿示例风格
# =========================
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("#EEF0F6")

x = np.arange(len(experiments))
bars = ax.bar(
    x, base_accuracies,
    width=0.55,
    color=colors,
    edgecolor="black",
    linewidth=1.2,
    zorder=3
)

ax.set_ylabel("Top-1 Accuracy(%)", fontsize=16, fontweight="normal")
ax.set_xlabel("")
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=18, fontweight="normal")

# 减少两侧留白
ax.margins(x=0.01)

# y轴范围自动根据数据给点边距，避免顶端拥挤
ymin = np.floor(min(base_accuracies) * 2) / 2 - 0.5
ymax = np.ceil(max(base_accuracies) * 2) / 2 + 0.5
ax.set_ylim(ymin, ymax)
ax.set_yticks(np.arange(ymin, ymax + 1e-9, 0.5))
ax.tick_params(axis="y", labelsize=18)

ax.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.45, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 顶部数值标注
for bar, acc in zip(bars, base_accuracies):
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.08,
        f"{acc:.2f}",
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        color="#222222",
        zorder=5
    )

# =========================
# 4) 图例：单行标签 + 横向展开
#    9项 -> 3列更匀称，减少左右留白
# =========================
legend_labels = [f"{short} ({format_ratio(exp)})" for exp, short in zip(experiments, x_labels)]
legend_handles = [Patch(facecolor=c, edgecolor="black", label=lab) for c, lab in zip(colors, legend_labels)]

ax.legend(
    handles=legend_handles,
    loc="upper left",
    bbox_to_anchor=(0.0, -0.20, 1.0, 0.1),
    mode="expand",
    ncol=3,
    frameon=False,
    fontsize=18,
    handlelength=1.0,
    handletextpad=0.5,
    columnspacing=0.8,
    borderaxespad=0.0,
    labelspacing=0.8
)

plt.tight_layout(rect=[0, 0.13, 1, 1])

output_path = "/workspace/OTOv2_v2/test/Ablation/resnet110_cifar100/ablation_results/pruning_accuracy_comparison.png"
pdf_output_path = "/workspace/OTOv2_v2/test/Ablation/resnet110_cifar100/ablation_results/pruning_accuracy_comparison.pdf"
plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0, format="png")
plt.savefig(pdf_output_path, bbox_inches="tight", pad_inches=0, format="pdf")
plt.show()

print(f"图表已保存至: {output_path}")
print(f"图表已保存至: {pdf_output_path}")

best_idx = int(np.argmax(base_accuracies))
worst_idx = int(np.argmin(base_accuracies))
print(f"\n最佳配置: {experiments[best_idx]} (准确率: {base_accuracies[best_idx]:.2f}%)")
print(f"最差配置: {experiments[worst_idx]} (准确率: {base_accuracies[worst_idx]:.2f}%)")