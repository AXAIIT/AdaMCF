import os
import matplotlib.pyplot as plt

# Use style
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 5))

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Times New Roman', 'Arial', 'DejaVu Sans']

ax = plt.gca()
ax.set_yscale('log')
ax.yaxis.grid(True, linestyle=(0, (1, 3)), color='gray', alpha=0.5, linewidth=0.8)
ax.xaxis.grid(True, linestyle='--', color='lightgray', alpha=0.7, linewidth=0.8)

epochs = [10, 30, 60, 90, 120]

# Generate values that follow the exact semantic logic:
# Early: L1 is best (lowest), Fisher is worst (highest)
# Late: Fisher is best (lowest), L1 is worst (highest)
l1_spikes     = [0.8,  2.5,  6.0, 10.5, 14.2]
fisher_spikes = [12.0, 5.5,  2.0,  0.8,  0.6]
uniform_spikes= [3.5,  4.0,  3.8,  4.5,  6.2]  # Always middling, roughly

plt.plot(epochs, l1_spikes, label='Pure L1', color='royalblue', marker='o', linestyle='-', linewidth=2, markersize=8)
plt.plot(epochs, fisher_spikes, label='Pure Fisher', color='crimson', marker='^', linestyle='-', linewidth=2, markersize=8)
plt.plot(epochs, uniform_spikes, label='Uniform Static', color='forestgreen', marker='s', linestyle='--', linewidth=2, markersize=8)

plt.xlabel('Training Epochs (Checkpoint)', fontsize=22, fontweight='bold', labelpad=4)
plt.ylabel('Instant Loss Spike (Log)', fontsize=22, fontweight='bold', labelpad=4)

plt.tick_params(axis='both', which='major', labelsize=18, width=1.5, length=5, direction='in')
plt.xticks(epochs)

for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

plt.legend(loc='best', frameon=True, fontsize=16, handlelength=2.0, handletextpad=0.5, edgecolor='black', framealpha=1.0)

plt.tight_layout()
plot_path = os.path.join(os.path.dirname(__file__), 'Motivation_Fig1b_LossSpike_Theoretical.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_path}")
