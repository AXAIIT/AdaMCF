# 三模型分图绘制说明（自动读取模型）

脚本位置：`/workspace/OTOv2_v2/test/Ablation/draw_model_filter_curves.py`

该脚本会自动读取以下对象并提取每个 `Conv2d` 的 `out_channels`：

- `vgg`
	- 原始：`/workspace/OTOv2_v2/test/pretrain_vgg16_cifar10/models/VGG16_cifar10.pt`
	- 剪枝后：`/workspace/OTOv2_v2/test/pretrain_vgg16_cifar10/finetuned_models_sparsity_0.8/vgg16_cifar10_finetuned.pt`
- `resnet18`
	- 原始：`/workspace/OTOv2_v2/test/pretrain_resnet18_cifar100/model/best_model.pth`
	- 剪枝后：`/workspace/OTOv2_v2/test/pretrain_resnet18_cifar100/finetuned_resnet18_cifar100_sparsity_0_5/resnet18_cifar100_finetuned.pt`
- `resnet50`
	- 原始：`/workspace/OTOv2_v2/test/pretain_resnet50_imagenet/model/resnet50_imagenet_torchvision.pth`
	- 剪枝后：`/workspace/OTOv2_v2/test/pretain_resnet50_imagenet/finetuned_resnet50_sparsity_0_5/resnet50_imagenet_finetuned.pt`

默认行为：`resnet50` 会把剪枝后曲线拆成 `Final 1x1` 和 `Final 3x3` 两条线。

## 1) 一次生成三张图

```bash
python /workspace/OTOv2_v2/test/Ablation/draw_model_filter_curves.py
```

默认输出目录：`/workspace/OTOv2_v2/test/Ablation/figures`

- `vgg_filter_curve.png`
- `resnet18_filter_curve.png`
- `resnet50_filter_curve_1x1.png`（默认 split：仅 `1x1 conv` 对比图）
- `resnet50_filter_curve_3x3.png`（默认 split：仅 `3x3 conv` 对比图）
- `resnet50_filter_curve_kernel_split.png`（默认 split：`1x1` 与 `3x3` 同图四曲线对比）

## 2) 只画某个模型

```bash
python /workspace/OTOv2_v2/test/Ablation/draw_model_filter_curves.py --vgg
python /workspace/OTOv2_v2/test/Ablation/draw_model_filter_curves.py --resnet18
python /workspace/OTOv2_v2/test/Ablation/draw_model_filter_curves.py --resnet50
```

## 3) 自定义输出目录和分辨率

```bash
python /workspace/OTOv2_v2/test/Ablation/draw_model_filter_curves.py --output-dir /workspace/OTOv2_v2/test/Ablation/my_figures --dpi 400
```

## 4) 关闭 ResNet-50 的 1x1/3x3 拆分

```bash
python /workspace/OTOv2_v2/test/Ablation/draw_model_filter_curves.py --resnet50 --no-resnet50-kernel-split
```

关闭后输出文件为：`resnet50_filter_curve.png`

说明：默认 split 模式会同时输出“分开两张图 + 同图一张图”。同图中 `1x1` 与 `3x3` 使用两组不同对比色：

- `1x1`: 蓝色（Original）/红色（Final）
- `3x3`: 绿色（Original）/橙色（Final）

## 5) 修改模型路径

如果后续你要换实验结果，只需要修改脚本顶部的 `MODEL_PATHS`。
