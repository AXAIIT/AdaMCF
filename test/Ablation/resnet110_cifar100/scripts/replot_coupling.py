import os
import re
import numpy as np
import matplotlib.pyplot as plt

filepath = '/workspace/OTOv2_v2/test/Ablation/resnet110_cifar100/Coupling_Analysis_data.txt'
base_dir = os.path.dirname(filepath)

hard_losses = []
proxy_losses = []

with open(filepath, 'r') as f:
    for line in f:
        match = re.search(r'Hard=([\d\.]+), Soft=([\d\.]+)', line)
        if match:
            hard_losses.append(float(match.group(1)))
            proxy_losses.append(float(match.group(2)))

hard_arr = np.array(hard_losses, dtype=np.float64)
proxy_arr = np.array(proxy_losses, dtype=np.float64)

r = float(np.corrcoef(hard_arr, proxy_arr)[0, 1]) if len(hard_arr) > 1 else float("nan")

coef = np.polyfit(hard_arr, proxy_arr, deg=1)
x_line = np.linspace(hard_arr.min(), hard_arr.max(), 200)
y_line = coef[0] * x_line + coef[1]

plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

ax.scatter(
    hard_arr,
    proxy_arr,
    s=40,
    alpha=0.6,
    edgecolors='k',
    linewidths=0.5,
)
ax.plot(
    x_line,
    y_line,
    linestyle='--',
    linewidth=2.5,
)

ax.set_xlabel(r'$\mathcal{L}_p^H(\alpha)$', fontweight='bold', fontsize=16)
ax.set_ylabel(r'$\mathcal{L}_p^S(\alpha)$', fontweight='bold', fontsize=16)
ax.grid(True, linestyle='--', alpha=0.5)

textstr = f'Pearson $R$ = {r:.4f}'
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
ax.text(
    0.05, 0.95, textstr,
    transform=ax.transAxes,
    fontsize=14,
    verticalalignment='top',
    bbox=props
)

plt.tight_layout()

save_path_pdf = os.path.join(base_dir, "coupling_analysis_tpami.pdf")
save_path_png = os.path.join(base_dir, "coupling_analysis_tpami.png")
plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
print(f"Saved {save_path_pdf} and {save_path_png}")
