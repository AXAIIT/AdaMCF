# 02 ResNet-50 / ImageNet-1K Component Profiling Optional

## 实验目的

在更接近正文 ImageNet 设置的模型上补充 AdaMCF 组件级开销。该实验为可选项，用于增强 appendix profiling 的代表性。

## 做什么

对 ResNet-50 / ImageNet-1K 的 AdaMCF 剪枝阶段进行 component-wise wall-clock profiling，测量与 ResNet-110 / CIFAR-100 相同的组件。

## 怎么做

基于现有 ResNet-50 / ImageNet 实验脚本添加计时，不改变原始剪枝策略、训练轮数和数据处理方式。GPU 计时必须使用 `torch.cuda.synchronize()`。

## 预期效果

预期较大模型上的绝对耗时更高，但主要 overhead 来源仍应是 ZO proxy-loss evaluation。

## 需要记录

`component_timing.csv`、`component_timing_summary.json`、模型与数据集配置、GPU 信息、batch size、各组件 mean/std 时间。