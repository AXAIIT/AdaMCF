# 03 Component-Wise Overhead Profiling

## 实验目的

本类实验用于回应审稿意见中关于 AdaMCF 复杂度和训练开销的问题。目标是测量各核心组件的 wall-clock overhead，而不是只报告端到端训练时间。

## 实验数量

本类共 2 个实验：

```text
01_resnet110_cifar100
02_resnet50_imagenet_optional
```

其中 ResNet-110 / CIFAR-100 为必做，ResNet-50 / ImageNet-1K 为可选补充。

## 需要测量的组件

```text
Dependency-group construction
Multi-criteria scoring
Adaptive fusion and ranking
Soft-mask proxy construction
ZO proxy-loss evaluation
Projection / momentum / alpha update
Cosine-front scheduling
```

## 怎么做

在不改变剪枝逻辑的前提下，对关键函数调用前后加入计时。GPU 计时必须使用 `torch.cuda.synchronize()`，避免 CUDA 异步执行导致计时不准。

## 预期效果

预期 ZO proxy-loss evaluation 是主要额外训练成本；cosine-front schedule 本身耗时应很小。结果用于补充 appendix 中的 component-wise profiling table。