# 04 Inference Latency and Edge Measurement

## 实验目的

本目录用于回应审稿意见中关于 **FLOPs reduction 是否能够转化为真实 inference latency speedup** 的问题，并补充以下两类证据：

1. 服务器 RTX 3090 上的 batch size 1 单样本推理延迟；
2. 代表性边缘设备 Jetson Xavier NX 8GB 上的真实 TensorRT 推理延迟。

实验不预设“剪枝后一定更快”。结构化剪枝可能减少 FLOPs 和参数量，但真实延迟还受到以下因素影响：

```text
kernel launch overhead
operator implementation
memory access
TensorRT tactic selection
GPU utilization
input resolution
batch size
precision
power mode and thermal throttling
```

因此，本实验的成功标准是得到 **可复现、可追溯、公平对比** 的结果，而不是强制得到大于 1 的 speedup。

---

## 当前结论与实施路线

板端仓库：

```text
E:\Edge_Compression_Deploy\edge-inference-deploy-lab
```

其中已验证的 Jetson 环境主线为：

| 项目 | 当前板端环境 |
|---|---|
| Board | Jetson Xavier NX 8GB |
| OS | Ubuntu 20.04 LTS, aarch64 |
| JetPack / L4T | JetPack 5.1.2 / L4T 35.6.4 |
| CUDA | 11.4.315 |
| cuDNN | 8.6.0.166 |
| TensorRT | 8.5.2.2 |
| Python | 3.8.10 |
| Power mode | `MODE_15W_6CORE` |
| Cooling | active fan |
| Monitor | `tegrastats` |
| Board workspace | `/home/jetson/edge-inference-deploy-lab` |

本实验采用以下明确分工：

```text
训练服务器 / OTOv2 环境
    ├── 加载 dense checkpoint
    ├── 加载 AdaMCF compressed model
    ├── 导出静态 batch=1 ONNX
    ├── 执行 ONNX checker 与 ONNX Runtime 输出对齐
    └── 将 ONNX 和配置同步到 Jetson

Jetson Xavier NX
    ├── 使用板端 TensorRT 8.5.2.2 构建 FP16 engine
    ├── 验证 dense / pruned engine 输出与源模型对齐
    ├── 使用相同输入、precision、warmup、repeat 运行 benchmark
    ├── 使用 tegrastats 记录内存、温度、功耗和 GPU 利用率
    └── 保存逐次 raw result、聚合结果、日志和模型 hash
```

**不在 Jetson 上重新训练或重新剪枝。**

**不把直接加载 `ResNet_compressed.pt` 的板端 PyTorch 计时作为正式端侧结论。** 原因是该文件通常是完整 Python pickle 模型，依赖原始模块路径、PyTorch 版本和 OTOv2 运行环境，不适合作为稳定的部署产物。正式端侧链路应为：

```text
PyTorch model -> ONNX -> TensorRT FP16 engine -> Jetson benchmark
```

---

## 当前第一优先任务：Dense / Pruned 模型统一转换为 Jetson FP16 格式

在编写板端 latency benchmark 之前，先完成未剪枝模型和 AdaMCF 已剪枝模型的统一转换。最终板端运行格式统一为：

```text
TensorRT serialized engine (.engine)
precision = FP16
batch size = 1
execution provider = TensorRT-GPU
```

ONNX 仅作为服务器与 Jetson 之间的中间交换和输出校验格式，不作为最终板端实验格式。

### 待转换模型

| Model | Variant | Source artifact | Static input | Intermediate format | Jetson format | Precision | Priority |
|---|---|---|---|---|---|---|---|
| ResNet-110 / CIFAR-100 | Dense | `03_jetson_resnet110_batch1/model/best_model.pth` + `model/model.py` | `1x3x32x32` | ONNX opset 12 | TensorRT `.engine` | FP16 | 必做 |
| ResNet-110 / CIFAR-100 | AdaMCF Pruned | `03_jetson_resnet110_batch1/ResNet_compressed.pt` | `1x3x32x32` | ONNX opset 12 | TensorRT `.engine` | FP16 | 必做 |
| ResNet-50 / ImageNet-1K | Dense | `04_jetson_resnet50_batch1_optional/model/resnet50_imagenet_torchvision.pth` | `1x3x224x224` | ONNX opset 12 | TensorRT `.engine` | FP16 | 可选 |
| ResNet-50 / ImageNet-1K | AdaMCF Pruned | `04_jetson_resnet50_batch1_optional/ResNet_compressed.pt` | `1x3x224x224` | ONNX opset 12 | TensorRT `.engine` | FP16 | 可选 |

### 转换顺序

```text
1. 冻结源模型文件并计算 SHA256
2. 在 OTOv2 原始环境中分别加载 dense / pruned 模型
3. model.eval()，使用固定 batch=1 输入保存 PyTorch reference logits
4. 导出静态 ONNX：input name=input，output name=logits，opset=12
5. 执行 onnx.checker 和 ONNX Runtime 输出对齐
6. 将 ONNX、固定输入、配置和 hash manifest 同步到 Jetson
7. 在 Jetson Xavier NX 上使用 TensorRT 8.5.2.2 和 --fp16 构建 engine
8. 使用相同固定输入验证 TensorRT logits、Top-1 和 Top-5
9. 对齐通过后才允许进入 latency benchmark
```

Dense 与 pruned 必须使用相同的：

```text
ONNX opset
static batch and input shape
TensorRT version
builder flags
workspace setting
precision policy
tactic source
timing cache policy
power mode
```

禁止比较 Dense FP32 engine 与 Pruned FP16 engine，也禁止一个模型使用 PyTorch、另一个模型使用 TensorRT。

### 计划产物

服务器侧中间产物：

```text
04_inference_latency_edge/artifacts/
├── resnet110/dense/resnet110_dense_b1_opset12.onnx
├── resnet110/adamcf_pruned/resnet110_adamcf_pruned_b1_opset12.onnx
├── resnet50/dense/resnet50_dense_b1_opset12.onnx
└── resnet50/adamcf_pruned/resnet50_adamcf_pruned_b1_opset12.onnx
```

Jetson 最终运行产物：

```text
models/adamcf_latency/tensorrt/
├── resnet110_dense_b1_fp16.engine
├── resnet110_adamcf_pruned_b1_fp16.engine
├── resnet50_dense_b1_fp16.engine
└── resnet50_adamcf_pruned_b1_fp16.engine
```

ResNet-110 的两个 FP16 engine 是主实验最低前置条件；ResNet-50 两个 engine 属于可选增强项。

### 转换阶段验收条件

- dense / pruned 源模型均能独立加载；
- ONNX 均通过 checker；
- PyTorch 与 ONNX Runtime 的 Top-1 一致；
- Jetson 上两个 TensorRT FP16 engine 均构建成功；
- TensorRT 输出无 NaN / Inf；
- TensorRT 与 PyTorch/ORT 的 Top-1、Top-5 一致或差异满足预设容差；
- 每个源模型、ONNX、engine 和配置都有 SHA256；
- 构建日志能够证明 dense / pruned 使用同一套 FP16 转换参数。

---

## 实验矩阵与优先级

本类共 4 个实验：

| ID | 目录 | 平台 | 模型 | Backend | Precision | Batch | 优先级 |
|---:|---|---|---|---|---|---:|---|
| 01 | `01_server_resnet110_batch1` | RTX 3090 server | ResNet-110 / CIFAR-100 | PyTorch CUDA | FP32 | 1 | 必做 |
| 02 | `02_server_resnet50_batch1` | RTX 3090 server | ResNet-50 / ImageNet-1K | PyTorch CUDA | FP32 | 1 | 必做 |
| 03 | `03_jetson_resnet110_batch1` | Jetson Xavier NX 8GB | ResNet-110 / CIFAR-100 | TensorRT GPU | FP16 | 1 | 必做，板端主实验 |
| 04 | `04_jetson_resnet50_batch1_optional` | Jetson Xavier NX 8GB | ResNet-50 / ImageNet-1K | TensorRT GPU | FP16 | 1 | 可选增强项 |

推荐执行顺序：

```text
01_server_resnet110_batch1
02_server_resnet50_batch1
03_jetson_resnet110_batch1
04_jetson_resnet50_batch1_optional
```

必须先完成服务器结果和 ONNX 输出对齐，再进行板端 TensorRT benchmark。

---

## 当前模型产物

### ResNet-110 / CIFAR-100

当前目录已有：

```text
03_jetson_resnet110_batch1/model/best_model.pth
03_jetson_resnet110_batch1/model/model.py
03_jetson_resnet110_batch1/ResNet_compressed.pt
```

含义：

| 产物 | 角色 |
|---|---|
| `model/best_model.pth` | dense baseline state dict / checkpoint |
| `model/model.py` | dense ResNet-110 architecture definition |
| `ResNet_compressed.pt` | AdaMCF pruned model，完整序列化模型 |

### ResNet-50 / ImageNet-1K

当前目录已有：

```text
04_jetson_resnet50_batch1_optional/model/resnet50_imagenet_torchvision.pth
04_jetson_resnet50_batch1_optional/ResNet_compressed.pt
```

含义：

| 产物 | 角色 |
|---|---|
| `model/resnet50_imagenet_torchvision.pth` | dense torchvision ResNet-50 state dict |
| `ResNet_compressed.pt` | AdaMCF pruned model，完整序列化模型 |

正式实验前必须计算并保存所有源模型、ONNX 和 TensorRT engine 的 SHA256。若文件 hash 发生变化，必须新建 run，不能覆盖旧结果。

---

## 当前代码状态与需要修正的问题

### 1. ResNet-110 Jetson 脚本

当前文件：

```text
03_jetson_resnet110_batch1/Adaptive_Fusion_Inference_latency.py
```

该脚本目前混合了：

```text
数据加载
AdaMCF 剪枝
微调
准确率评估
FLOPs 统计
服务器 PyTorch latency benchmark
```

并且存在以下不适合板端正式实验的问题：

1. 依赖 `only_train_once`、OTO 和 alpha scheduler；
2. 硬编码 `/workspace/OTOv2_v2` 与 `/workspace/data`；
3. 在脚本主入口中重新执行 Fixed-alpha 和 Adaptive-alpha 剪枝；
4. 只保存平均值和标准差，没有逐次 latency、p50/p90/p95/p99；
5. 未记录模型 hash、环境基线、功耗、温度、内存和 GPU 利用率；
6. 板端正式结果仍是 PyTorch CUDA，而不是已验证的 TensorRT 部署链路。

因此该文件只能作为历史训练与服务器测量参考，不能直接作为 Jetson 正式入口。

### 2. ResNet-50 Jetson 脚本

当前文件：

```text
04_jetson_resnet50_batch1_optional/resnet50_imagenet_inference_latency.py
```

当前主要问题：

1. 模型路径硬编码为 `/workspace/OTOv2_v2/...`；
2. `use_amp=True`，但日志写成 FP32，precision 口径不一致；
3. 同时 benchmark batch=1 和 batch=128，与该目录的 batch=1 实验定位混杂；
4. 输出只有平均值和标准差，没有逐次 raw result 和 percentile；
5. 没有环境基线、模型 hash、资源监控和质量对齐；
6. 仍是 PyTorch benchmark，不是 Jetson TensorRT 主线。

该文件后续应拆分为“服务器导出/验证”和“Jetson TensorRT benchmark”两个阶段。

### 3. Server 入口缺失

两个 server 子目录 README 当前引用：

```text
run_server_batch1_latency.sh
```

但本目录当前没有该脚本，也没有统一的：

```text
server_batch1_latency_benchmark.py
```

后续需要补充统一 server benchmark 入口，不能继续让 README 引用不存在的文件。

---

## 统一控制变量

同一组 dense / pruned 对比中，下面配置必须完全一致：

| 控制项 | 要求 |
|---|---|
| Platform | 同一台服务器或同一块 Jetson |
| Environment baseline | 相同 `environment_baseline_id` |
| Backend/runtime | PyTorch CUDA 对 PyTorch CUDA；TensorRT 对 TensorRT |
| Execution provider | 同为 GPU；不能一个使用 GPU、另一个 CPU fallback |
| Precision | 同为 FP32 或同为 FP16 |
| Batch size | 固定为 1 |
| Input shape | ResNet-110: `1x3x32x32`; ResNet-50: `1x3x224x224` |
| Input data | 相同固定输入 tensor 或相同固定输入集 |
| Warmup | 相同次数 |
| Repeat | 相同次数 |
| CUDA synchronization | PyTorch GPU 计时前后必须同步 |
| TensorRT context | engine 加载和 context 创建不计入正式 latency |
| Host-to-device copy | 主 inference-only 指标中不计入；pipeline 指标单独报告 |
| Power mode | Jetson dense / pruned 完全一致 |
| Cooling / clock | 完全一致，并写入环境基线 |
| Measurement order | 多轮交替执行，降低温升和顺序偏差 |

禁止以下不公平比较：

```text
Dense FP32 vs Pruned FP16
Dense PyTorch vs Pruned TensorRT
Dense batch=1 vs Pruned batch>1
Dense engine-only vs Pruned end-to-end pipeline
Dense 未锁定输入 shape vs Pruned 静态 shape
Dense 在低温首轮运行、Pruned 在温度稳定后运行且不做交替
```

---

## Benchmark 口径

本实验区分两个 measurement scope。

### 1. `inference_only`：论文主表口径

只测量模型前向推理：

```text
input tensor already prepared
model / engine already loaded
execution context already created
no dataset loading
no image decode
no model loading
no ONNX / engine build
no first CUDA initialization
```

该口径用于论文中的 single-sample inference latency 表格。

### 2. `pipeline`：部署补充口径

可选补充真实输入下的：

```text
preprocess_ms
inference_ms
postprocess_ms
latency_ms = preprocess + inference + postprocess
```

对于分类模型，postprocess 通常只包含 softmax / argmax 或 Top-K 解码。`pipeline` 结果不能覆盖或替代 `inference_only` 结果，两者必须分别保存和标注。

---

## 固定 Benchmark 参数

正式实验默认采用：

| 参数 | 设置 |
|---|---|
| Batch size | 1 |
| Warmup | 50 iterations |
| Repeat | 500 measured iterations |
| Independent rounds | 5 rounds |
| ResNet-110 input | `1x3x32x32` |
| ResNet-50 input | `1x3x224x224` |
| Server precision | FP32 |
| Jetson precision | TensorRT FP16 |
| Jetson execution provider | TensorRT-GPU |
| Random seed | 42 |
| Timing unit | ms |
| Raw granularity | one record per measured iteration |

上述参数高于板端仓库规定的最低要求：

```text
warmup >= 20
repeat >= 200
```

如果设备状态或运行时间限制只能执行 200 次 repeat，必须在 run 记录中将状态标记为 `degraded`，并写明原因；dense 和 pruned 仍必须使用相同 repeat。

### 多轮执行顺序

为了降低温度漂移和先后顺序偏差，建议使用交替顺序：

```text
Round 1: Dense -> Pruned
Round 2: Pruned -> Dense
Round 3: Dense -> Pruned
Round 4: Pruned -> Dense
Round 5: Dense -> Pruned
```

每轮开始前分别完成 warmup。聚合时保留：

```text
per-iteration raw latency
per-round summary
overall summary
```

---

## 延迟统计与 Speedup 定义

每个模型版本必须报告：

```text
mean latency
standard deviation
p50 latency
p90 latency
p95 latency
p99 latency
throughput / FPS
```

论文当前表格使用平均延迟，因此主 speedup 定义保持为：

```text
Speedup_mean = Dense mean latency / Pruned mean latency
```

同时补充更稳健的：

```text
Speedup_p50 = Dense p50 latency / Pruned p50 latency
Speedup_p95 = Dense p95 latency / Pruned p95 latency
```

不能只报告最小延迟，也不能删除异常较慢的合法迭代。只有明确属于初始化、engine 构建、模型加载、OOM 或系统中断的无效样本才能排除，并必须记录排除规则。

---

## 服务器 PyTorch Benchmark 设计

服务器实验用于与论文当前 batch=1 latency 表格对齐。

### 统一接口建议

后续新增：

```text
04_inference_latency_edge/scripts/benchmark_pytorch_batch1.py
```

建议参数：

```bash
python scripts/benchmark_pytorch_batch1.py \
  --model resnet110 \
  --dense-checkpoint <dense_checkpoint> \
  --pruned-model <pruned_model> \
  --device cuda:0 \
  --precision fp32 \
  --batch-size 1 \
  --warmup 50 \
  --repeat 500 \
  --rounds 5 \
  --raw-output <raw_jsonl> \
  --summary-output <summary_json>
```

PyTorch CUDA 计时必须执行：

```python
torch.cuda.synchronize()
start = time.perf_counter_ns()
_ = model(x)
torch.cuda.synchronize()
end = time.perf_counter_ns()
```

也可以使用 CUDA Event，但 dense 和 pruned 必须使用同一种计时实现。需要固定：

```python
torch.backends.cudnn.benchmark = True
model.eval()
torch.inference_mode()
```

如果启用 TF32，dense 和 pruned 必须同时启用，并在结果中记录；默认建议关闭，以保持 FP32 口径明确。

---

## ONNX 导出规范

### 为什么在服务器导出

`ResNet_compressed.pt` 是完整序列化模型，应该在原始 OTOv2 / PyTorch 环境中加载。Jetson 仅接收稳定的 ONNX 和 TensorRT engine，不承担 OTOv2 Python 依赖。

### 建议新增脚本

```text
04_inference_latency_edge/scripts/export_latency_models_onnx.py
04_inference_latency_edge/scripts/validate_latency_models_onnx.py
```

### 固定导出参数

| 参数 | ResNet-110 | ResNet-50 |
|---|---|---|
| Input shape | `1x3x32x32` | `1x3x224x224` |
| Batch | static 1 | static 1 |
| Opset | 12 | 12 |
| Input name | `input` | `input` |
| Output name | `logits` | `logits` |
| Dynamic axes | disabled | disabled |
| Constant folding | enabled | enabled |

导出示意：

```python
torch.onnx.export(
    model.eval(),
    dummy_input,
    output_path,
    opset_version=12,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes=None,
    do_constant_folding=True,
)
```

加载本项目可信的完整模型时可能需要：

```python
torch.load(path, map_location="cpu", weights_only=False)
```

该选项只允许用于本项目生成且 hash 已确认的可信模型文件，不用于来源不明的 pickle 文件。

### 导出后必须验证

每个 ONNX 必须通过：

1. `onnx.checker.check_model`；
2. ONNX Runtime 可执行；
3. PyTorch 与 ONNX Runtime 在固定输入上的 logits 对齐；
4. 输出 shape 一致；
5. dense / pruned 的类别数一致；
6. 保存 ONNX SHA256 和导出日志。

建议至少记录：

```text
max_abs_error
mean_abs_error
cosine_similarity
top1_agreement
```

---

## Jetson TensorRT Engine 构建

TensorRT engine 必须在目标 Jetson 上构建，避免跨 GPU 架构和跨 TensorRT 版本复用 engine。

### ResNet-110 dense 示例

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=artifacts/onnx/resnet110_dense_b1.onnx \
  --saveEngine=artifacts/tensorrt/resnet110_dense_b1_fp16.engine \
  --fp16 \
  --shapes=input:1x3x32x32 \
  --workspace=1024 \
  --dumpProfile \
  --separateProfileRun \
  --exportProfile=logs/runtime/resnet110_dense_b1_fp16_profile.json \
  > logs/conversion/resnet110_dense_b1_fp16_build.log 2>&1
```

### ResNet-110 pruned 示例

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=artifacts/onnx/resnet110_pruned_b1.onnx \
  --saveEngine=artifacts/tensorrt/resnet110_pruned_b1_fp16.engine \
  --fp16 \
  --shapes=input:1x3x32x32 \
  --workspace=1024 \
  --dumpProfile \
  --separateProfileRun \
  --exportProfile=logs/runtime/resnet110_pruned_b1_fp16_profile.json \
  > logs/conversion/resnet110_pruned_b1_fp16_build.log 2>&1
```

ResNet-50 将 shape 改为：

```text
input:1x3x224x224
```

构建完成后必须执行：

```bash
sha256sum artifacts/tensorrt/*.engine
```

Dense 和 pruned 必须使用相同：

```text
TensorRT version
FP16 mode
workspace limit
static batch=1 shape
GPU execution provider
power mode
clock policy
```

DLA 不作为本实验主线。只有 dense 和 pruned 都能完整使用相同 DLA 配置时，才允许新建独立探索实验；不能一个走 DLA、另一个走 GPU。

---

## Jetson Benchmark 脚本设计

板端仓库已有 YOLO11n TensorRT benchmark 和 `TRTInference` 封装，但其预处理、后处理和 raw schema 面向目标检测，不能直接复制为 ResNet 分类脚本。可以复用以下工程能力：

```text
TensorRT Python API loading
CUDA buffer allocation
tegrastats monitor parsing
SHA256 recording
JSONL raw result
p50/p90/p95/p99 aggregation
MLPerf-style Summary
```

后续建议新增一个分类专用入口：

```text
04_inference_latency_edge/scripts/benchmark_tensorrt_classification.py
```

或同步到板端仓库：

```text
projects/06_model_compression/scripts/benchmark/benchmark_tensorrt_classification.py
```

建议接口：

```bash
python benchmark_tensorrt_classification.py \
  --dense-engine artifacts/tensorrt/resnet110_dense_b1_fp16.engine \
  --pruned-engine artifacts/tensorrt/resnet110_pruned_b1_fp16.engine \
  --model resnet110 \
  --input-shape 1,3,32,32 \
  --input-set configs/inputs/resnet110_fixed_inputs.npz \
  --warmup 50 \
  --repeat 500 \
  --rounds 5 \
  --measurement-scope inference_only \
  --env-baseline-id <environment_baseline_id> \
  --monitor-log logs/monitor/<run_id>_tegrastats.log \
  --raw-output results/raw/<run_id>.jsonl \
  --summary-output results/processed/<run_id>_summary.json
```

正式计时必须满足：

1. engine 和 execution context 只初始化一次；
2. 输入 tensor 预先准备；
3. warmup 不写入正式统计；
4. 每次 iteration 保存独立 latency；
5. 使用 CUDA Event 或等价的 GPU 同步计时；
6. 不把 engine deserialize、内存分配和首次 CUDA 初始化计入正式 latency；
7. 记录实际 engine binding shape 和 dtype；
8. 确认 `cpu_fallback=false`；
9. 输出 dense / pruned 独立 raw record；
10. 生成 overall 和 per-round summary。

---

## 固定输入集

### 性能输入

性能 benchmark 推荐使用预生成且落盘的固定 tensor 集，而不是每次随机生成：

```text
ResNet-110: normalized CIFAR-100 tensors, shape N x 3 x 32 x 32
ResNet-50: normalized ImageNet tensors, shape N x 3 x 224 x 224
```

固定输入集必须记录：

```text
input_set_id
source dataset
sample indices
normalization mean/std
file path
file SHA256
random seed
```

`inference_only` 可循环使用同一批固定 tensor，但不能 dense 使用随机输入、pruned 使用另一组随机输入。

### 质量输入

质量验证分为两层：

#### 层 1：部署产物对齐

分别验证：

```text
Dense PyTorch -> Dense ONNX -> Dense TensorRT
Pruned PyTorch -> Pruned ONNX -> Pruned TensorRT
```

这一步判断格式转换和 FP16 TensorRT 是否改变模型行为，不比较 dense 与 pruned 本身是否输出一致。

#### 层 2：任务级质量

分别在 CIFAR-100 / ImageNet-1K 验证集上记录 dense 与 pruned 的 Top-1 / Top-5。剪枝模型允许存在论文中已经报告的准确率变化，但 TensorRT engine 应尽量保持其对应源模型的任务精度。

只完成固定输入 logits 对齐、没有完成验证集评估时：

```text
task_level_quality_status = not_verified
```

不能写成质量 `pass`。

---

## 质量门槛

### 格式转换门槛

建议默认：

| 对齐项 | 建议门槛 |
|---|---:|
| PyTorch vs ONNX Runtime Top-1 agreement | `>= 99.9%` fixed inputs |
| PyTorch vs TensorRT FP16 Top-1 agreement | `>= 99.0%` fixed inputs |
| Output shape | exact match |
| Class dimension | exact match |
| NaN / Inf | 0 |

logits 误差阈值应根据 FP16 数值范围和实际输出分布记录，不建议只用一个绝对误差阈值判断所有样本。

### 任务级门槛

TensorRT engine 相对其对应源模型的验证集 Top-1 下降建议不超过：

```text
0.20 percentage points
```

若超过门槛，结果仍需保存，但状态标记为 `degraded` 或 `fail`，并检查：

```text
normalization mismatch
NCHW / NHWC mismatch
FP16 overflow
TensorRT tactic or unsupported layer
wrong class dimension
incorrect checkpoint
```

### 性能门槛

不设置强制 speedup 下限：

```text
Speedup > 1: 真实加速
Speedup ≈ 1: 延迟基本不变
Speedup < 1: 剪枝模型在该 backend / hardware 上更慢
```

三种情况均可作为有效结果。只有 measurement protocol 不完整、环境不一致或结果不可追溯时，实验才不合格。

---

## Jetson 电源、温度与监控

正式 benchmark 前必须确认：

```bash
nvpmodel -q
cat /etc/nv_tegra_release
```

当前已有环境基线使用：

```text
MODE_15W_6CORE
jetson_clocks = false
active fan
```

如果后续执行：

```bash
sudo jetson_clocks
```

则 clock policy 已发生变化，应建立新的环境基线，不能把锁频结果与未锁频结果放在同一主表中直接比较。

监控示例：

```bash
mkdir -p logs/monitor
tegrastats --interval 1000 \
  --logfile logs/monitor/<run_id>_tegrastats.log &
echo $! > logs/monitor/<run_id>_tegrastats.pid

python benchmark_tensorrt_classification.py ...

kill "$(cat logs/monitor/<run_id>_tegrastats.pid)"
```

至少记录：

```text
memory peak
swap usage
temperature max
VDD_IN power if available
GPU utilization
CPU utilization
power mode
clock policy
```

若某字段不可观测，写明：

```text
not_observable: reason
```

不能直接删除字段。

---

## 建议目录结构

在保留现有四个实验子目录的基础上，建议补充共享脚本、配置、产物和结果目录：

```text
04_inference_latency_edge/
├── README.md
├── scripts/
│   ├── benchmark_pytorch_batch1.py
│   ├── export_latency_models_onnx.py
│   ├── validate_latency_models_onnx.py
│   ├── benchmark_tensorrt_classification.py
│   ├── aggregate_latency_results.py
│   └── build_jetson_engines.sh
├── configs/
│   ├── server_rtx3090_fp32.yaml
│   ├── jetson_xavier_nx_trt_fp16.yaml
│   ├── resnet110_cifar100.yaml
│   └── resnet50_imagenet.yaml
├── artifacts/
│   ├── onnx/
│   └── tensorrt/
├── inputs/
│   ├── resnet110_fixed_inputs.npz
│   └── resnet50_fixed_inputs.npz
├── results/
│   ├── raw/
│   └── processed/
├── logs/
│   ├── export/
│   ├── conversion/
│   ├── benchmark/
│   ├── runtime/
│   └── monitor/
├── 01_server_resnet110_batch1/
├── 02_server_resnet50_batch1/
├── 03_jetson_resnet110_batch1/
└── 04_jetson_resnet50_batch1_optional/
```

如果正式运行完全纳入 `edge-inference-deploy-lab`，则板端 raw result 和日志应同时按项目六规范归档到：

```text
benchmark/raw/06_model_compression/edge/
benchmark/processed/06_model_compression/
logs/benchmark/06_model_compression/edge/
logs/monitor/06_model_compression/edge/
projects/06_model_compression/runs/<run_id>/
projects/06_model_compression/reports/edge_speedup_report.md
```

revision 目录保留论文实验入口、配置快照和最终汇总；板端仓库保留真实部署证据。两边通过 `run_id`、模型 hash 和结果路径建立对应关系。

---

## Raw Result 字段

如果板端结果归档到 `edge-inference-deploy-lab`，必须遵守：

```text
benchmark/schemas/model_compression_raw_schema.json
```

该 schema 设置了 `additionalProperties=false`，因此不能随意新增字段。每次 measured iteration 至少使用以下正式字段：

```text
schema_version = 1
run_id
timestamp
project = 06_model_compression
experiment_type = edge_eval
measurement_granularity = edge_iteration
model_role = student_baseline | compressed
model_name = resnet110 | resnet50
model_variant = resnet110_dense | resnet110_adamcf_pruned | resnet50_dense | resnet50_adamcf_pruned
model_file_hash
compression_method = none | pruning
precision_or_quantization = fp16
dataset_id
input_set_id
environment_baseline_id
config_path
config_sha256
metric_name
current_metric
quality_gate_id
quality_gate
quality_status
board = jetson_8gb
backend_runtime = tensorrt
execution_provider = TensorRT-GPU
backend_artifact_path
backend_artifact_sha256
cpu_fallback = false
fallback_reason
warmup
repeat
input_id
iteration_index
is_warmup
preprocess_ms
inference_ms
postprocess_ms
latency_ms
memory_mb
memory_peak_mb
temperature_c
power_w
power_mode
cpu_gpu_npu_bpu_utilization
contribution_group = student_baseline | pruned
status
raw_log_path
monitor_log_path
related_project1_baseline
related_troubleshooting_id
notes
```

当前项目六 schema 没有独立的：

```text
batch
round_index
measurement_scope
```

因此在不修改 schema 的前提下，这三项必须写入 config snapshot，并在 `notes` 中记录，例如：

```text
batch=1; round_index=2; measurement_scope=inference_only
```

5 轮实验的 `iteration_index` 使用全局连续编号：

```text
global_iteration_index = round_index * repeat + local_iteration_index
```

对于 `inference_only` 记录，项目六正式 edge schema 要求三段耗时字段存在，因此使用：

```text
preprocess_ms = 0.0
postprocess_ms = 0.0
latency_ms = inference_ms
```

并在 `notes` 中明确说明预处理和后处理不属于本次测量范围，不能把 `0.0` 解释为真实 pipeline 耗时。

Warmup 可以写入独立 debug 日志，但不进入正式聚合；如果写入 raw JSONL，必须设置：

```text
is_warmup = true
```

并在聚合时明确排除。

---

## 聚合结果字段

`results/processed/` 至少生成：

```text
run_id
model
platform
backend
precision
batch
warmup
repeat
rounds
mean_latency_ms
std_latency_ms
p50_latency_ms
p90_latency_ms
p95_latency_ms
p99_latency_ms
throughput_fps
memory_peak_mb
temperature_max_c
power_max_w
quality_status
status
```

Dense / pruned 配对后额外生成：

```text
speedup_mean
speedup_p50
speedup_p95
flops_reduction_percent
params_reduction_percent
model_size_reduction_percent
```

FLOPs 和 Params 来自模型结构分析，不从 latency 反推。

---

## Run 记录要求

每个正式实验建立独立 run，建议命名：

```text
YYYYMMDD_server_resnet110_pytorch_fp32_b1
YYYYMMDD_server_resnet50_pytorch_fp32_b1
YYYYMMDD_jetson_resnet110_tensorrt_fp16_b1
YYYYMMDD_jetson_resnet50_tensorrt_fp16_b1
```

每个 run 至少保存：

```text
run.md
config snapshot
source model path and SHA256
ONNX path and SHA256
TensorRT engine path and SHA256
exact command
export log
conversion log
benchmark log
monitor log
raw JSONL
processed JSON / CSV
quality alignment result
failure / degradation note
```

失败实验也必须保留：

```text
status = fail
related_troubleshooting_id = <non-empty id>
```

不能删除失败日志后只保留成功结果。

---

## 当前论文结果对齐

当前 Appendix 中已有的 server batch=1 参考值为：

| Model | Version | Batch | Latency (ms) | Speedup |
|---|---|---:|---:|---:|
| ResNet-50 | Dense Baseline | 1 | 4.87 | 1.00 |
| ResNet-50 | AdaMCF | 1 | 4.86 | 1.00 |
| ResNet-110 | Dense Baseline | 1 | 10.17 | 1.00 |
| ResNet-110 | AdaMCF | 1 | 7.80 | 1.30 |

这些数值仅作为当前论文对齐参考。正式修改代码后，应根据保存的 raw latency 和当前环境重新核对：

```text
model artifact 是否相同
precision 是否相同
warmup / repeat 是否相同
计时实现是否相同
是否排除了初始化
结果是否能复现
```

如果新实测值与论文值存在差异，应优先排查测量口径和模型版本；不能为了匹配论文手工修改板端或服务器实测值。

Jetson 结果当前尚未形成可追溯的正式表格，必须完成 TensorRT engine、逐次 raw result、质量门槛和资源监控后再填入论文。

---

## 最终结果表模板

所有转换、质量校验和板端 latency 结果最终必须汇总成一张可直接用于论文表格重建的主表。逐次 raw JSONL、构建日志和资源监控是证据层；下面的表格是论文呈现层，不能直接手工填写，必须由 processed result 自动生成或核对。

### 论文重建主表

| Model | Platform | Variant | Backend | Precision | Input | Params Red. (%) | FLOPs Red. (%) | Engine Size (MB) | Top-1 (%) | Mean Latency (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Speedup (×) | Memory Peak (MB) | Temp Max (°C) | Quality Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ResNet-110 | Jetson Xavier NX | Dense | TensorRT-GPU | FP16 | `1x3x32x32` | 0.00 | 0.00 |  |  |  |  |  |  | 1.00 |  |  |  |
| ResNet-110 | Jetson Xavier NX | AdaMCF | TensorRT-GPU | FP16 | `1x3x32x32` |  |  |  |  |  |  |  |  |  |  |  |  |
| ResNet-50 | Jetson Xavier NX | Dense | TensorRT-GPU | FP16 | `1x3x224x224` | 0.00 | 0.00 |  |  |  |  |  |  | 1.00 |  |  |  |
| ResNet-50 | Jetson Xavier NX | AdaMCF | TensorRT-GPU | FP16 | `1x3x224x224` |  |  |  |  |  |  |  |  |  |  |  |  |

论文主表的核心字段建议保留：

```text
Model
Variant
Platform
Backend / Precision
Latency
Speedup
```

若版面允许，再加入：

```text
FLOPs reduction
Params reduction
Engine size
Top-1
Memory peak
```

p50/p95/p99、温度和详细资源信息可以放入 Appendix 或补充材料，但必须保留在 processed result 中，保证论文表格后续可重新生成。

Speedup 统一计算为：

```text
Speedup = Dense mean latency / AdaMCF mean latency
```

同时在 processed 文件中保留：

```text
speedup_mean
speedup_p50
speedup_p95
```

论文默认采用 `speedup_mean`，不得在不同模型间选择不同口径以获得更好结果。

### Server batch=1

| Model | Variant | Backend | Precision | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Speedup mean | Top-1 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ResNet-110 | Dense | PyTorch CUDA | FP32 |  |  |  |  | 1.00 |  |
| ResNet-110 | AdaMCF | PyTorch CUDA | FP32 |  |  |  |  |  |  |
| ResNet-50 | Dense | PyTorch CUDA | FP32 |  |  |  |  | 1.00 |  |
| ResNet-50 | AdaMCF | PyTorch CUDA | FP32 |  |  |  |  |  |  |

### Jetson batch=1

| Model | Variant | Backend | Precision | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Speedup mean | Memory peak | Temp max | Quality status |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ResNet-110 | Dense | TensorRT-GPU | FP16 |  |  |  |  | 1.00 |  |  |  |
| ResNet-110 | AdaMCF | TensorRT-GPU | FP16 |  |  |  |  |  |  |  |  |
| ResNet-50 | Dense | TensorRT-GPU | FP16 |  |  |  |  | 1.00 |  |  |  |
| ResNet-50 | AdaMCF | TensorRT-GPU | FP16 |  |  |  |  |  |  |  |  |

---

## 论文写法建议

实验设置可以写成：

```text
We measure single-sample inference latency with batch size 1 on both an RTX 3090 server GPU and a representative Jetson edge device. For each platform, the dense and AdaMCF-pruned models are evaluated using the same backend, precision, static input shape, warm-up iterations, timing iterations, power mode, and measurement procedure. Model loading, engine construction, data loading, and first-run initialization are excluded from the inference-only latency. In addition to mean latency and throughput, we report p50, p90, p95, and p99 latency, together with memory and thermal measurements on the Jetson platform.
```

结果分析应根据实测情况选择：

```text
The pruned ResNet-110 achieves a measurable single-sample latency reduction, indicating that its structured FLOPs reduction can be translated into practical acceleration on the evaluated backend.
```

或：

```text
The latency improvement of ResNet-50 is marginal despite substantial FLOPs reduction. This result confirms that FLOPs alone do not fully characterize deployment efficiency, which is also affected by operator implementation, kernel-launch overhead, memory access, hardware utilization, and backend tactic selection.
```

如果 Jetson speedup 小于 1，也应如实写：

```text
The pruned model does not outperform the dense baseline on the evaluated Jetson backend. The result suggests that the resulting channel configuration is not necessarily hardware-friendly for this specific device and TensorRT tactic set, despite reducing theoretical computation.
```

---

## 实施阶段

### 阶段 A：文档与口径冻结

- [x] 明确 server 与 Jetson 实验矩阵；
- [x] 明确 Jetson Xavier NX / TensorRT FP16 主线；
- [x] 明确 inference-only 与 pipeline 分离；
- [x] 明确 warmup、repeat、rounds、percentile 和资源字段；
- [x] 明确质量门槛和模型 hash 要求。

### 阶段 B：模型转换与校验脚本

- [ ] 新增统一 dense / pruned 模型加载器；
- [ ] 新增 dense / pruned 静态 ONNX 导出脚本；
- [ ] 新增 PyTorch / ONNX Runtime 输出对齐脚本；
- [ ] 新增 Jetson TensorRT FP16 engine 构建脚本；
- [ ] 新增 TensorRT logits、Top-1、Top-5 对齐脚本；
- [ ] 新增转换配置 YAML、固定输入和 hash manifest；
- [ ] 先完成 ResNet-110 dense / pruned 两个 FP16 engine，再开始 latency 代码。

### 阶段 C：Benchmark 与论文表格生成脚本

- [ ] 新增统一 server PyTorch batch=1 benchmark；
- [ ] 新增 Jetson TensorRT 分类 benchmark；
- [ ] 新增逐次 raw JSONL 聚合脚本；
- [ ] 新增论文主表 CSV / Markdown 自动生成脚本；
- [ ] 确保论文表格字段可从 processed result 重建，不手工转录数值。

### 阶段 D：ResNet-110 主实验

- [ ] 复测 RTX 3090 batch=1；
- [ ] 导出 dense / pruned ONNX；
- [ ] 完成 ORT 对齐；
- [ ] 在 Jetson 构建两个 FP16 engine；
- [ ] 完成 TensorRT 固定输入对齐；
- [ ] 完成 5 轮 latency benchmark；
- [ ] 完成 tegrastats 资源记录；
- [ ] 汇总论文表格。

### 阶段 E：ResNet-50 可选增强

- [ ] 复测 RTX 3090 batch=1；
- [ ] 导出 dense / pruned ONNX；
- [ ] 完成 ORT 对齐；
- [ ] 在 Jetson 构建两个 FP16 engine；
- [ ] 完成 latency、资源和质量记录；
- [ ] 判断是否纳入 Appendix。

---

## 验收标准

本类实验完成时必须满足：

1. 服务器 ResNet-110 和 ResNet-50 均有真实 batch=1 raw latency；
2. Jetson ResNet-110 dense / pruned 均使用 TensorRT FP16 engine；
3. dense / pruned 在同一环境基线、backend、precision、input、warmup 和 repeat 下比较；
4. 每个正式结果具有模型、ONNX、engine 和配置 SHA256；
5. 具有逐次 raw result，而不是只有控制台平均值；
6. 报告 mean、std、p50、p90、p95、p99 和 throughput；
7. Jetson 具有内存、温度、功耗或电源模式、GPU 利用率记录；
8. 具有 PyTorch -> ONNX -> TensorRT 的部署产物对齐；
9. 具有任务级 Top-1 / Top-5 质量状态，或明确标记 `not_verified`；
10. 失败、无加速、CPU fallback、质量下降和环境差异均保留证据；
11. 论文中的每个 latency 和 speedup 数字都能追溯到 raw result、run 和日志；
12. ResNet-50 Jetson 即使不执行，也必须明确标记 `status=not_executed`，并在 `notes` 中注明其为 optional experiment，不能留下含糊状态。
