# 03 Jetson / ResNet-110 / Batch Size 1

## 实验目的

测量 ResNet-110 dense 与 AdaMCF pruned model 在代表性边缘设备 Jetson 上的 batch size 1 推理延迟。

## 做什么

在 Jetson 上使用相同 backend、precision 和 input size 比较 dense 与 pruned model。优先使用 FP16，如果使用 FP32 或其他 backend，必须明确记录。

## 怎么做

将 dense 和 pruned checkpoint 同步到 Jetson，固定 batch size 1，进行 warm-up 后重复计时。不要把模型加载、数据读取或首次 CUDA 初始化计入正式 latency。

## 预期效果

边缘设备上可能更能体现结构化剪枝的部署价值，但实际 speedup 仍取决于 backend、算子实现和内存访问。

## 需要记录

Dense latency、Pruned latency、Speedup、precision、backend、Jetson 型号、功耗/频率模式、warm-up / timing iteration 数量。