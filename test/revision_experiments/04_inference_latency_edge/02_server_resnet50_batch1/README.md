# 02 Server RTX 3090 / ResNet-50 / Batch Size 1

## 实验目的

测量 ResNet-50 dense 与 AdaMCF pruned model 在服务器 RTX 3090 上的 batch size 1 推理延迟。

## 做什么

在相同 server GPU、backend、precision 和 input size 下比较 dense 与 pruned model 的 latency。

## 怎么做

加载对应 ResNet-50 dense checkpoint 和 AdaMCF pruned checkpoint，使用 batch size 1 dummy input 或验证集样本进行 warm-up 和 repeated timing。GPU 计时必须同步。

## 预期效果

ResNet-50 的 FLOPs reduction 不一定等比例转化为 batch size 1 latency speedup，可能受到 kernel launch、memory access 和 operator implementation 影响。

## 运行脚本

统一使用上级目录中的 server batch=1 脚本：

```bash
cd /workspace/OTOv2_v2
MODELS="resnet50" PRECISION=fp32 GPU=0 \
  bash test/revision_experiments/04_inference_latency_edge/run_server_batch1_latency.sh
```

默认 dense model：

```text
/workspace/OTOv2_v2/test/Ablation/pretain_resnet50_imagenet/model/resnet50_imagenet_torchvision.pth
```

默认 pruned model：

```text
/workspace/OTOv2_v2/test/Ablation/pretain_resnet50_imagenet/cache_resnet50_imagenet_pruned_Adaptive-α_0_5/ResNet_compressed.pt
```

## 需要记录

Dense latency、Pruned latency、Speedup、FLOPs reduction、Params reduction、precision、backend、GPU 信息和 timing 配置。