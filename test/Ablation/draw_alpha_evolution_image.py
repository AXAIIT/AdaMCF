import os
import sys
import ast
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
try:
    from scipy.interpolate import make_interp_spline
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

matplotlib.use("Agg")
from matplotlib import rcParams
from matplotlib import font_manager as fm
import os


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
    rcParams["legend.fontsize"] = 18
    rcParams["axes.unicode_minus"] = False


_set_style()

def parse_log_file(file_path):
    """
    读取日志文件并解析包含 alpha 值的字典行。
    """
    alpha_records = []
    
    if not os.path.exists(file_path):
        print(f"错误: 文件未找到 - {file_path}")
        return []

    print(f"正在读取文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 检查行是否以 '{' 开头并且包含特定的键名，这是 alpha 字典的特征
            if line.startswith("{") and "'l1_avg_magnitude'" in line:
                try:
                    # 使用 ast.literal_eval 安全地将字符串转换为字典
                    data_dict = ast.literal_eval(line)
                    alpha_records.append(data_dict)
                except (ValueError, SyntaxError):
                    continue
    
    print(f"共解析到 {len(alpha_records)} 条 alpha 记录。")
    return alpha_records

def draw_alpha_evolution(txt_path, warmup_epochs=5, pruning_epochs=100):
    """
    绘制 alpha 演化曲线。
    
    Args:
        txt_path: 日志文件路径 (.txt)
        warmup_epochs: 预热周期数 (跳过前 N 个数据点)
        pruning_epochs: 剪枝周期数 (绘制接下来的 N 个数据点)
    """
    # 1. 解析数据
    all_records = parse_log_file(txt_path)
    
    if not all_records:
        print("没有数据可绘图。")
        return

    # 2. 根据周期筛选数据
    start_idx = warmup_epochs
    end_idx = start_idx + pruning_epochs
    
    # 确保索引不越界
    if start_idx >= len(all_records):
        print(f"警告: 预热周期 ({warmup_epochs}) 大于或等于记录总数 ({len(all_records)})。")
        return
        
    sliced_records = all_records[start_idx:end_idx]
    
    if not sliced_records:
        print("筛选后没有数据。")
        return

    print(f"绘图数据范围: Epoch {start_idx + 1} 到 {start_idx + len(sliced_records)}")

    # 3. 整理绘图数据
    # 获取所有键 (例如: l1_avg_magnitude, l2_avg_magnitude...)
    keys = list(sliced_records[0].keys())
    
    # 将数据转换为 {key: [values...]} 的形式
    plot_data = {key: [] for key in keys}
    for record in sliced_records:
        for key in keys:
            if key in record:
                plot_data[key].append(record[key])
    
    # 4. 绘图设置
    try:
        plt.style.use("seaborn-v0_8-colorblind")
    except OSError:
        plt.style.use("ggplot") # 如果找不到样式则回退

    plt.figure(figsize=(10, 6), dpi=200)
    
    # X 轴坐标 (从 0 开始，表示剪枝周期的相对时间)
    epochs_axis = np.arange(0, len(sliced_records))
    
    # 获取颜色映射
    palette = plt.get_cmap("tab10")
    
    # 图例标签映射（只替换常用键，其他保持原名）
    legend_label_map = {
        'l1_avg_magnitude': 'L1',
        'l2_avg_magnitude': 'L2',
        'cosine_similarity': 'Trend',
        'fisher': 'Fisher',
        'grad_weight_product': 'GW'
    }

    # 绘制每条曲线
    for i, key in enumerate(keys):
        y_values = np.array(plot_data[key])
        
        # --- 平滑处理 ---
        if _HAS_SCIPY and len(epochs_axis) > 3: # 只有在 scipy 可用且数据点足够多时才进行平滑
            # 创建更密集的 X 轴用于插值
            x_smooth = np.linspace(epochs_axis.min(), epochs_axis.max(), 300)
            # 使用 B-Spline 进行插值 (k=3 为三次样条)
            spl = make_interp_spline(epochs_axis, y_values, k=3)
            y_smooth = spl(x_smooth)
            
            # 绘制平滑后的曲线
            plt.plot(x_smooth, y_smooth, 
                     label=legend_label_map.get(key, key), 
                     color=palette(i % 10), 
                     linewidth=2.0,  # 线条加粗
                     alpha=0.8)
        else:
            # 数据点太少，直接绘制原数据
            plt.plot(epochs_axis, y_values, 
                     label=legend_label_map.get(key, key), 
                     color=palette(i % 10), 
                     linewidth=2.0,  # 线条加粗
                     alpha=0.8)

    # 统一标签字号并应用 Times 字体样式
    plt.xlabel("Pruning Epoch", fontsize=24, fontweight='normal')
    plt.ylabel(r"Importance Score Contribution ($\alpha$)", fontsize=24, fontweight='normal')

    # 设置刻度字号（若需要额外控制）
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    # 图例始终放在图内：使用 matplotlib 的 'best' 自动位置，
    # 并根据图例数量自动设置列数以减少遮挡（最多 3 列）。
    ncol = min(3, max(1, len(keys)))
    ax.legend(frameon=True, loc='best', fontsize=16, ncol=ncol, framealpha=0.85)
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.3)


    output_dir = os.path.dirname(txt_path)
    file_name = os.path.splitext(os.path.basename(txt_path))[0]
    output_path = os.path.join(output_dir, f"{file_name}_alpha_evolution.png")
    pdf_output_path = os.path.join(output_dir, f"{file_name}_alpha_evolution.pdf")
    
    # 增大 pad_inches 以避免坐标轴标签在 tight 模式下被裁切
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.12, format='png')
    plt.savefig(pdf_output_path, bbox_inches='tight', pad_inches=0.12, format='pdf')
    plt.close()
    print(f"图像已保存至: {output_path}")
    print(f"图像已保存至: {pdf_output_path}")

if __name__ == "__main__":
    target_txt_path = "/workspace/OTOv2_v2/test/pretrain_vgg16_cifar10/vgg16_cifar10_pruning_0_8_loss.txt"
    
    WARMUP_EPOCHS = 0
    PRUNING_EPOCHS = 160
    
    if os.path.exists(target_txt_path):
        draw_alpha_evolution(target_txt_path, warmup_epochs=WARMUP_EPOCHS, pruning_epochs=PRUNING_EPOCHS)
    else:
        print(f"请检查文件路径是否存在: {target_txt_path}")