import argparse
import importlib.util
import os
import sys
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision.models as tv_models


matplotlib.use("Agg")


MODEL_PATHS: Dict[str, Dict[str, str]] = {
    "vgg": {
        "original_dir": "/workspace/OTOv2_v2/test/pretrain_vgg16_cifar10/models",
        "pruned_path": "/workspace/OTOv2_v2/test/pretrain_vgg16_cifar10/finetuned_models_sparsity_0.8/vgg16_cifar10_finetuned.pt",
    },
    "resnet18": {
        "original_dir": "/workspace/OTOv2_v2/test/pretrain_resnet18_cifar100/model",
        "pruned_path": "/workspace/OTOv2_v2/test/pretrain_resnet18_cifar100/finetuned_resnet18_cifar100_sparsity_0_5/resnet18_cifar100_finetuned.pt",
    },
    "resnet50": {
        "original_dir": "/workspace/OTOv2_v2/test/pretain_resnet50_imagenet/model",
        "pruned_path": "/workspace/OTOv2_v2/test/pretain_resnet50_imagenet/finetuned_resnet50_sparsity_0_5/resnet50_imagenet_finetuned.pt",
    },
}

UNIFIED_FIGSIZE = (7.8, 5.2)


def _prepare_import_paths() -> None:
    extra_paths = [
        "/workspace/OTOv2_v2/test/pretrain_vgg16_cifar10",
        "/workspace/OTOv2_v2/test/pretrain_resnet18_cifar100",
        "/workspace/OTOv2_v2/test/pretrain_vgg16_cifar10/models",
        "/workspace/OTOv2_v2/test/pretrain_resnet18_cifar100/model",
    ]
    for path in extra_paths:
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_model_object(path: str) -> torch.nn.Module:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到模型文件: {path}")
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, torch.nn.Module):
        raise TypeError(f"{path} 不是完整模型对象，类型为: {type(loaded)}")
    loaded.eval()
    return loaded


def _load_original_model(model_name: str, original_dir: str) -> torch.nn.Module:
    if model_name == "vgg":
        model_path = os.path.join(original_dir, "VGG16_cifar10.pt")
        return _load_model_object(model_path)

    if model_name == "resnet18":
        state_path = os.path.join(original_dir, "best_model.pth")
        definition_path = os.path.join(original_dir, "resnet18.py")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"未找到 ResNet18 原始权重: {state_path}")
        if not os.path.exists(definition_path):
            raise FileNotFoundError(f"未找到 ResNet18 模型定义: {definition_path}")

        spec = importlib.util.spec_from_file_location("resnet18_definition", definition_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载模块定义: {definition_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        resnet18_ctor = getattr(module, "ResNet18")

        state_dict = torch.load(state_path, map_location="cpu", weights_only=False)
        model = resnet18_ctor()
        model.load_state_dict(state_dict)
        model.eval()
        return model

    if model_name == "resnet50":
        state_path = os.path.join(original_dir, "resnet50_imagenet_torchvision.pth")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"未找到 ResNet50 原始权重: {state_path}")
        state_dict = torch.load(state_path, map_location="cpu", weights_only=False)
        model = tv_models.resnet50(weights=None)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    raise ValueError(f"不支持的模型: {model_name}")


def _extract_conv_channels(model: torch.nn.Module) -> List[int]:
    channels = [
        module.out_channels
        for module in model.modules()
        if isinstance(module, torch.nn.Conv2d)
    ]
    if not channels:
        raise ValueError("未在模型中找到 Conv2d 层，无法绘图。")
    return channels


def _extract_conv_channels_by_kernel(model: torch.nn.Module, kernel_size: int) -> List[int]:
    channels = [
        module.out_channels
        for module in model.modules()
        if isinstance(module, torch.nn.Conv2d)
        and module.kernel_size == (kernel_size, kernel_size)
    ]
    if not channels:
        raise ValueError(f"未找到 kernel={kernel_size}x{kernel_size} 的 Conv2d 层。")
    return channels


def build_model_series(model_name: str, split_resnet50_final_by_kernel: bool) -> Dict[str, List[int]]:
    spec = MODEL_PATHS[model_name]
    original_model = _load_original_model(model_name, spec["original_dir"])
    pruned_model = _load_model_object(spec["pruned_path"])
    original_channels = _extract_conv_channels(original_model)
    pruned_channels = _extract_conv_channels(pruned_model)

    if model_name == "resnet50" and split_resnet50_final_by_kernel:
        original_1x1 = _extract_conv_channels_by_kernel(original_model, kernel_size=1)
        original_3x3 = _extract_conv_channels_by_kernel(original_model, kernel_size=3)
        final_1x1 = _extract_conv_channels_by_kernel(pruned_model, kernel_size=1)
        final_3x3 = _extract_conv_channels_by_kernel(pruned_model, kernel_size=3)
        return {
            "Original 1x1": original_1x1,
            "Original 3x3": original_3x3,
            "Final 1x1": final_1x1,
            "Final 3x3": final_3x3,
        }

    return {
        "Original": original_channels,
        "Final": pruned_channels,
    }


def _set_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")
    from matplotlib import font_manager

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
                font_manager.fontManager.addfont(font_path)
            except Exception:
                continue

    plt.rcParams["font.family"] = ["Times New Roman"]
    # Ensure PDFs embed fonts in a common, editable format (Type 42)
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    # Print resolved font used by matplotlib for logging/verification
    try:
        from matplotlib.font_manager import FontProperties, findfont

        preferred = plt.rcParams.get("font.family")
        pref = preferred[0] if isinstance(preferred, (list, tuple)) and preferred else preferred
        resolved_path = findfont(FontProperties(family=pref), fallback_to_default=True)
        resolved_name = font_manager.FontProperties(fname=resolved_path).get_name()
        print(f"[font] Preferred family: {pref} -> Resolved font: {resolved_name} ({resolved_path})")
    except Exception:
        pass
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18
    plt.rcParams["legend.fontsize"] = 15


def draw_single_model(model_name: str, series_map: Dict[str, List[int]], output_dir: str, dpi: int = 300) -> List[str]:
    max_len = max(len(values) for values in series_map.values())

    fig, ax = plt.subplots(figsize=UNIFIED_FIGSIZE, dpi=dpi)

    line_styles = {
        "Original": {"color": "royalblue", "marker": "o", "linewidth": 1.8, "markersize": 3.0},
        "Final": {"color": "orange", "marker": "^", "linewidth": 1.6, "markersize": 2.9},
        "Final 1x1": {"color": "crimson", "marker": "^", "linewidth": 1.6, "markersize": 2.9},
        "Final 3x3": {"color": "crimson", "marker": "s", "linewidth": 1.6, "markersize": 2.8},
    }

    for label, values in series_map.items():
        x = list(range(1, len(values) + 1))
        style = line_styles.get(label, {"linewidth": 1.6, "markersize": 2.9})
        ax.plot(x, values, label=label, **style)

    all_values = [value for values in series_map.values() for value in values]
    unique_values = sorted(set(all_values))
    if len(unique_values) <= 8:
        ax.set_yticks(unique_values)

    ax.set_xlabel("Block Index", fontweight="normal")
    ax.set_ylabel("Number of Filters", fontweight="normal")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", frameon=True)
    tick_step = max(1, max_len // 12)
    ax.set_xticks(list(range(1, max_len + 1, tick_step)))
    ax.set_xlim(0.7, max_len + 0.3)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, f"{model_name}_filter_curve.png")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0)
    # Also save as vector PDF for publication-quality output
    out_pdf = os.path.join(output_dir, f"{model_name}_filter_curve.pdf")
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return [out_png, out_pdf]


def draw_resnet50_kernel_split(series_map: Dict[str, List[int]], output_dir: str, dpi: int = 300) -> List[str]:
    required = ["Original 1x1", "Final 1x1", "Original 3x3", "Final 3x3"]
    for name in required:
        if name not in series_map:
            raise ValueError(f"resnet50 分组绘图缺少序列: {name}")

    os.makedirs(output_dir, exist_ok=True)

    figure_specs = [
        {
            "key_original": "Original 1x1",
            "key_final": "Final 1x1",
            "style_original": {"color": "royalblue", "marker": "o", "linewidth": 1.8, "markersize": 3.0},
            "style_final": {"color": "crimson", "marker": "^", "linewidth": 1.7, "markersize": 2.9},
            "xlabel": "Block Index (1x1 Conv)",
            "filename": "resnet50_filter_curve_1x1.png",
        },
        {
            "key_original": "Original 3x3",
            "key_final": "Final 3x3",
            "style_original": {"color": "seagreen", "marker": "s", "linewidth": 1.8, "markersize": 3.0},
            "style_final": {"color": "darkorange", "marker": "d", "linewidth": 1.7, "markersize": 2.9},
            "xlabel": "Block Index (3x3 Conv)",
            "filename": "resnet50_filter_curve_3x3.png",
        },
    ]

    output_paths: List[str] = []
    for spec in figure_specs:
        original_vals = series_map[spec["key_original"]]
        final_vals = series_map[spec["key_final"]]
        x_original = list(range(1, len(original_vals) + 1))
        x_final = list(range(1, len(final_vals) + 1))
        max_len = max(len(original_vals), len(final_vals))

        fig, ax = plt.subplots(figsize=UNIFIED_FIGSIZE, dpi=dpi)
        ax.plot(x_original, original_vals, label=spec["key_original"], **spec["style_original"])
        ax.plot(x_final, final_vals, label=spec["key_final"], **spec["style_final"])

        ax.set_ylabel("Number of Filters", fontweight="normal")
        ax.set_xlabel(spec["xlabel"], fontweight="normal")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper left", frameon=True)
        tick_step = max(1, max_len // 12)
        ax.set_xticks(list(range(1, max_len + 1, tick_step)))
        ax.set_xlim(0.7, max_len + 0.3)

        plt.tight_layout()
        out_png = os.path.join(output_dir, spec["filename"])
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0)
        out_pdf = os.path.join(output_dir, spec["filename"].replace('.png', '.pdf'))
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        output_paths.extend([out_png, out_pdf])

    return output_paths


def draw_resnet50_kernel_combined(series_map: Dict[str, List[int]], output_dir: str, dpi: int = 300) -> List[str]:
    required = ["Original 1x1", "Final 1x1", "Original 3x3", "Final 3x3"]
    for name in required:
        if name not in series_map:
            raise ValueError(f"resnet50 同图绘制缺少序列: {name}")

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=UNIFIED_FIGSIZE, dpi=dpi)
    style_map = {
        "Original 1x1": {"color": "royalblue", "marker": "o", "linewidth": 1.8, "markersize": 3.0, "linestyle": "-"},
        "Final 1x1": {"color": "crimson", "marker": "^", "linewidth": 1.7, "markersize": 2.9, "linestyle": "-"},
        "Original 3x3": {"color": "seagreen", "marker": "s", "linewidth": 1.8, "markersize": 3.0, "linestyle": "--"},
        "Final 3x3": {"color": "darkorange", "marker": "d", "linewidth": 1.7, "markersize": 2.9, "linestyle": "--"},
    }

    max_len = max(len(values) for values in series_map.values())
    plot_order = ["Original 1x1", "Final 1x1", "Original 3x3", "Final 3x3"]
    for name in plot_order:
        values = series_map[name]
        x = list(range(1, len(values) + 1))
        ax.plot(x, values, label=name, **style_map[name])

    ax.set_ylabel("Number of Filters", fontweight="normal")
    ax.set_xlabel("Block Index", fontweight="normal")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", frameon=True)
    tick_step = max(1, max_len // 12)
    ax.set_xticks(list(range(1, max_len + 1, tick_step)))
    ax.set_xlim(0.7, max_len + 0.3)

    plt.tight_layout()
    out_png = os.path.join(output_dir, "resnet50_filter_curve_kernel_split.png")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0)
    out_pdf = os.path.join(output_dir, "resnet50_filter_curve_kernel_split.pdf")
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return [out_png, out_pdf]


def build_data_from_args(
    include_vgg: bool,
    include_resnet18: bool,
    include_resnet50: bool,
) -> List[str]:
    chosen = {}
    if include_vgg:
        chosen["vgg"] = True
    if include_resnet18:
        chosen["resnet18"] = True
    if include_resnet50:
        chosen["resnet50"] = True
    return list(chosen.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从原始模型与剪枝模型读取通道数并绘制分图")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/OTOv2_v2/test/Ablation/figures",
        help="输出目录（默认: /workspace/OTOv2_v2/test/Ablation/figures）",
    )
    parser.add_argument("--dpi", type=int, default=300, help="输出分辨率")
    parser.add_argument("--vgg", action="store_true", help="仅/包含绘制 VGG")
    parser.add_argument("--resnet18", action="store_true", help="仅/包含绘制 ResNet-18")
    parser.add_argument("--resnet50", action="store_true", help="仅/包含绘制 ResNet-50")
    parser.add_argument(
        "--no-resnet50-kernel-split",
        action="store_true",
        help="关闭 ResNet-50 的 Final 1x1/Final 3x3 拆分（默认开启）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _prepare_import_paths()
    _set_style()

    has_selection = args.vgg or args.resnet18 or args.resnet50
    model_names = build_data_from_args(
        include_vgg=args.vgg or not has_selection,
        include_resnet18=args.resnet18 or not has_selection,
        include_resnet50=args.resnet50 or not has_selection,
    )

    if not model_names:
        raise ValueError("未选择任何模型，请使用 --vgg/--resnet18/--resnet50。")

    split_resnet50_final_by_kernel = not args.no_resnet50_kernel_split

    for model_name in model_names:
        series_map = build_model_series(
            model_name=model_name,
            split_resnet50_final_by_kernel=split_resnet50_final_by_kernel,
        )
        if model_name == "resnet50" and split_resnet50_final_by_kernel:
            out_paths = draw_resnet50_kernel_split(
                series_map=series_map,
                output_dir=args.output_dir,
                dpi=args.dpi,
            )
            combined_out_paths = draw_resnet50_kernel_combined(
                series_map=series_map,
                output_dir=args.output_dir,
                dpi=args.dpi,
            )
            # combine lists of output paths
            out_paths.extend(combined_out_paths)
        else:
            out_paths = draw_single_model(
                model_name=model_name,
                series_map=series_map,
                output_dir=args.output_dir,
                dpi=args.dpi,
            )
        summary = ", ".join([f"{name}={len(values)}" for name, values in series_map.items()])
        print(f"{model_name}: {summary}")
        for out_path in out_paths:
            print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
