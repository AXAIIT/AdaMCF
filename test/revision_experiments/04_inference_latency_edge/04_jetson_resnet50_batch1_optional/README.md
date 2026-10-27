# 04 Jetson / ResNet-50 / Batch Size 1 Optional

## 实验目的

在 Jetson 上补充 ResNet-50 dense 与 AdaMCF pruned model 的 batch size 1 latency。该实验为可选项，用于增强 edge-device measurement 的完整性。

## 做什么

比较 ResNet-50 dense model 和 AdaMCF pruned model 在 Jetson 上的单样本推理延迟。

## 怎么做

使用相同 backend、precision、input size、warm-up iterations 和 timing iterations。若 Jetson 内存不足或 backend 不支持，应记录失败原因，不要删除失败日志。

## 预期效果

ResNet-50 的实际 speedup 可能不明显，尤其在 batch size 1 下。该结果可用于说明 FLOPs reduction 与真实 latency speedup 不一定线性对应。

## 需要记录

Dense latency、Pruned latency、Speedup、precision、backend、Jetson 型号、内存占用、是否 OOM、warm-up / timing iteration 数量。