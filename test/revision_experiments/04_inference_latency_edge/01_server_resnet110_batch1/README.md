# 01 Server RTX 3090 / ResNet-110 / Batch Size 1

## 实验目的

测量 ResNet-110 dense 与 AdaMCF pruned model 在服务器 RTX 3090 上的 batch size 1 推理延迟。

## 做什么

同一环境下分别运行 dense model 和 pruned model，记录单样本 latency、throughput 和 speedup。

## 怎么做

使用相同 input size、precision、backend、warm-up iterations 和 timing iterations。正式计时前进行 warm-up，GPU 计时前后执行 `torch.cuda.synchronize()`。

## 预期效果

ResNet-110 是 CIFAR 小模型，结构化剪枝后应能体现一定 batch size 1 latency speedup，但 speedup 不要求线性等于 FLOPs reduction。

## 运行脚本

统一使用上级目录中的 server batch=1 脚本：

```bash
cd /workspace/OTOv2_v2
MODELS="resnet110" PRECISION=fp32 GPU=0 \
  bash test/revision_experiments/04_inference_latency_edge/run_server_batch1_latency.sh
```

默认 dense model：

```text
/workspace/OTOv2_v2/test/Ablation/resnet110_cifar100/model/best_model.pth
```

默认 pruned model：

```text
/workspace/OTOv2_v2/test/Ablation/resnet110_cifar100/cache_resnet110_cifar100_pruned_Adaptive-α_0.5/ResNet_compressed.pt
```

## 需要记录

Dense latency、Pruned latency、Speedup、FLOPs reduction、Params reduction、precision、backend、GPU 信息、warm-up / timing iteration 数量。