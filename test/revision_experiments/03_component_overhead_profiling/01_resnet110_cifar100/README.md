# 01 ResNet-110 / CIFAR-100 Component Profiling

## 实验目的

测量 AdaMCF 在 ResNet-110 / CIFAR-100 上的组件级训练开销。这是 component-wise overhead profiling 的必做实验。

## 做什么

在默认 AdaMCF Full 设置下，对剪枝阶段的关键组件分别计时，包括 scoring、fusion/ranking、soft-mask proxy、ZO proxy-loss evaluation 和 alpha update。

## 怎么做

在对应操作前后加入同步计时：

```python
torch.cuda.synchronize()
t0 = time.perf_counter()
# measured operation
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
```

计时不应包含 data loading、checkpoint saving、完整 validation 和绘图。

## 预期效果

预期 ZO proxy-loss evaluation 占主要额外成本；schedule 读取和 projection 等操作成本很小。

## 需要记录

`component_timing.csv`、`component_timing_summary.json`、GPU 信息、batch size、pruning periods、每个组件的 mean/std 时间。