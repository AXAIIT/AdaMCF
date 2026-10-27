import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
from matplotlib import font_manager as fm


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

    rcParams['font.family'] = ['Times New Roman']
    rcParams['font.weight'] = 'normal'
    rcParams['axes.labelweight'] = 'normal'
    rcParams['axes.titleweight'] = 'normal'
    rcParams['axes.unicode_minus'] = False


_set_style()

# Estimated data
epochs = [10, 30, 60, 90, 120]
l1 = [0.8, 2.5, 6.0, 10.5, 14.2]
fisher = [12.0, 5.5, 2.0, 0.8, 0.6]
uniform_static = [3.5, 4.0, 3.8, 4.5, 6.2]
random = [7.2, 6.8, 6.1, 5.5, 5.1]

# Style settings suitable for academic papers
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')

fig, ax = plt.subplots(figsize=(6.2, 5.2))

# Plot lines
ax.plot(epochs, l1, label=r'$L_1$', color='royalblue', marker='o', linestyle='-', linewidth=2.5, markersize=8)
ax.plot(epochs, fisher, label='Fisher', color='crimson', marker='^', linestyle='-', linewidth=2.5, markersize=8)
ax.plot(epochs, uniform_static, label='Uniform Static', color='forestgreen', marker='s', linestyle='--', linewidth=2.5, markersize=8)
ax.plot(epochs, random, label='Random', color='black', marker='x', linestyle='-.', linewidth=2.5, markersize=8)

# Log scale for y-axis
ax.set_yscale('log')

# Labels and ticks
ax.set_xlabel('Epochs', fontsize=18, fontweight='normal')
ax.set_ylabel('Loss', fontsize=18, fontweight='normal')
ax.set_xticks(epochs)
ax.tick_params(axis='both', which='major', labelsize=14)

# Grid lines for better readability
ax.yaxis.grid(True, linestyle=':', color='gray', alpha=0.7)
ax.xaxis.grid(True, linestyle='--', color='lightgray', alpha=0.7)

# Adjust y-limit to make room
ax.set_ylim(bottom=0.4, top=20) 

# Legend setup - moving slightly right based on user feedback
legend = ax.legend(loc='center right', bbox_to_anchor=(1.00, 0.40), frameon=True, fontsize=16, edgecolor='gray', framealpha=1.0)
for text in legend.get_texts():
    text.set_weight('bold')

# Bounding box borders
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

plt.tight_layout()

# Save the plot
output_dir = '/workspace/OTOv2_v2/test/Ablation/resnet110_cifar100'
pdf_path = os.path.join(output_dir, 'recreated_loss_spike.pdf')
png_path = os.path.join(output_dir, 'recreated_loss_spike.png')

plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.12, dpi=300, format='pdf')
plt.savefig(png_path, bbox_inches='tight', pad_inches=0.12, dpi=300, format='png')

print(f"Academic plots generated:\n -> {pdf_path}\n -> {png_path}")
