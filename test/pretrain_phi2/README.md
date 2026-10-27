# Phi-2 的 AdaMCF / HESSO-CRIC 结构剪枝

目录位置：`/zeng/GY/OTOv2_v2/test/pretrain_phi2`。

沿用 BERT/VGG 的模型和剪枝产物布局。数据预处理、结构适配、训练与独立验收分别实现，使用已有环境，不需要安装新包。

```text
pretrain_phi2/
├── models/
│   └── phi-2/               # 下载的完整原始模型目录
├── cache_phi2_pruned/       # 剪枝缓存、稀疏模型和结构导出
├── finetuned_models/        # 剪后微调模型
├── logs/                   # 文本日志
├── runs/                   # 按实验名称保存配置快照、指标和结果
├── .gitignore
└── README.md
```

## 下载模型放在哪里

将完整模型文件放到：

```text
/zeng/GY/OTOv2_v2/test/pretrain_phi2/models/phi-2/
```

请保留原下载目录中的配置、分词器相关文件和全部权重；分片权重还需保留配套索引文件。`config.json` 应直接位于上述 phi-2 目录内，避免额外套一层同名文件夹。`.gitkeep` 是空目录占位文件，不影响模型加载。

## 代码职责

| 文件 | 职责 |
| --- | --- |
| `prepare_phi2_data.py` | 固定互斥 train/alpha/eval 原始行索引，分词，保存 JSONL 与摘要 |
| `phi2_data.py` | 校验和读取冻结数据、padding、生成模型输入 |
| `phi2_structure.py` | 原生 Phi 参数组、物理压缩、异构逐层宽度的保存与重建 |
| `phi2_precision.py` | 原生 FP32 漂移记录和严格 FP64 结构对照，诊断后恢复原生实现 |
| `export_phi2_pruned.py` | 从已保存的稀疏检查点独立物理导出，无需重跑训练 |
| `prune_phi2_hesso_cric.py` | 加载真实权重，运行完整 CRIC，检查零组，导出与对齐输出 |
| `verify_phi2_pruning.py` | 新进程重载，检查参数量、logits、缓存生成和反向传播 |
| `test_phi2_contract.py` | 随机小模型结构、数据隔离和输出保护测试 |
| `phi2_smoke_config.json` | 可读的功能验收配置，可另存为独立实验配置 |

## 已准备的共享数据（2026-10-27）

共享数据根目录：`/adminGroup/datasets_source203/llm_pruning`。

- Alpaca：`alpaca/alpaca_data.json`，52,002 条，指令训练/恢复候选。
- 评测：BoolQ、PIQA、HellaSwag、WinoGrande、ARC-Easy、ARC-Challenge、OpenBookQA。
- 困惑度：`wikitext2/wikitext-2-raw-v1`。
- 现有通用文本：`/adminGroup/datasets_source203/wikitext-103-v1`。

来源与固定版本见共享目录的 README.md、download_manifest.json、validation_result.json。七项最终评测集不参与选组或 α 更新。派生 token 数据仍放共享目录；原始权重及运行产物已由 .gitignore 排除。

## 结构剪枝范围

- FFN：联动 `fc1.weight` 的行、`fc1.bias` 与 `fc2.weight` 的列。
- 注意力：联动 Q/K/V 的完整头行、对应 bias 和 `dense.weight` 的输入列；保留每头宽度和 RoPE 配对关系。
- residual hidden size、embedding、LayerNorm、词表维度不做结构剪枝；这些参数与输出 bias 仍可训练。
- 所有参数恰好归属一次。拒绝 GQA、量化 Linear、空保留组及非法结构索引。

本地适配器按原生模块和实际宽度构造显式依赖合同，再调用已有 HESSOCRIC；它不使用面向旧 PhiMHA 的公共图算子。因此本次不代表通用图追踪器已支持任意 Phi 实现。公共 HESSO、CRIC 与既有模型算子没有为此修改。

五准则、自适应 α、周期内冻结和最终历史重融合使用此前已适配的 CRIC。32 层 Phi-2 有 64 个可剪枝参数组，初始投影长度为 `6 × 64 = 384` 步；投影试探步不等于普通梯度更新步。

## 独立预处理

监督响应与 EOS；prompt 和 padding 标签为 -100，模型内部执行因果位移。prompt/响应分别编码，长 prompt 从左侧截断并保留尾部，最多占一半长度；长响应从右侧截断，以 EOS 结束。manifest.json 冻结原始数据、分词器、种子、长度、原始行号与各 JSONL 摘要。

在本目录执行，所有输出使用不存在的新目录：

```bash
P=/opt/conda/envs/gy_adamcf_rtx6000d/bin/python
DATA=/adminGroup/datasets_source203/llm_pruning/processed/phi2_alpaca_seq128_seed42_my_run
$P prepare_phi2_data.py --output "$DATA" --max-length 128 \
  --train-samples 4096 --alpha-samples 128 --eval-samples 128
```

## 显存探测、剪枝与重载验收

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4

$P prune_phi2_hesso_cric.py --data-dir "$DATA" \
  --output runs/my_resource_profile --profile-only
$P prune_phi2_hesso_cric.py --data-dir "$DATA" \
  --config phi2_smoke_config.json --output runs/my_phi2_run
$P verify_phi2_pruning.py --run-dir runs/my_phi2_run
```

训练结束后先保存稀疏检查点。若导出诊断失败且尚未产生 compressed_model，可独立重试：

```bash
$P export_phi2_pruned.py --run-dir runs/my_phi2_run
```

profile-only 加载真实模型与 CRIC 缓存，执行全梯度和五准则评分，不执行剪枝。长任务应以后台进程启动，重定向 stdout/stderr；训练状态另外写入 JSON 和逐步日志。

验收配置为 FP32、无动量 SGD、batch 1、目标逻辑组稀疏率 10%，开启五准则和 α。无动量 SGD 用于控制全梯度与 CRIC 缓存的显存，属于已有 HESSO-CRIC 支持的基础更新方式。可修改配置选择 Adam/AdamW，但本轮结果不覆盖其大模型资源和精度表现。

2 步预热、2 步采样周期、最多 2 个 CRIC 周期、8 步混合稀疏化、4 步剪后训练是功能验收配置，不是正式精度恢复配方。目标比例针对逻辑组，实际参数减少比例从结果读取；逐层最小保留约束可能影响实际组数。

## 产物与成功条件

每次运行保存在指定的 runs/实验名：

| 文件 | 内容 |
| --- | --- |
| inputs.json | 权重、数据和代码摘要，配置与环境 |
| parameter_groups.json | 全部参数归属与可剪枝组 |
| training_trace.jsonl | 每步损失、梯度范数、CRIC 周期、α 和更新记录 |
| status.json / failure.json | 运行状态或异常；拒绝覆盖既有实验目录 |
| pruning_plan.json | 逐层 FFN/注意力头的保留与删除索引 |
| sparse_model/ | 稀疏原尺寸检查点；保留试探结果，避免导出失败后重跑训练 |
| sparse_probe.pt / equivalence_probe.pt | 固定输入、稀疏及压缩参考 logits，纯 Tensor 检查数据 |
| precision_audit.json | 原生 FP32 误差、相对 L2 与 token top-1 一致率，严格 FP64 对照结果 |
| compressed_model/ | model.safetensors、architecture.json、模型配置与分词器 |
| pruning_result.json | 实际压缩率、误差、显存、响应损失 |
| independent_verification.json | 新进程输出对齐、缓存生成与反向传播结果 |

成功条件：CRIC 和稀疏化结束；最终剪枝组精确归零；真实矩阵及参数量缩小；FP64 结构对照在 atol/rtol=1e-8 下通过；新进程重载与压缩模型的原生 FP32 参考对齐，并通过缓存生成和反向传播。

真实深层 Phi 的矩阵尺寸改变会改变 FP32 GEMM 求和顺序，原生 FP32 稀疏/压缩 logits 可能不满足逐元素 2e-4 门槛；该失败标志和实测误差保留，不改写为通过。为判别结构错误与浮点漂移，另外执行严格 FP64 对照，诊断作用域内将 attention softmax 也提升为 FP64，RoPE 使用双方相同的原生位置常数。作用域退出即恢复 FP32，安装包、训练和正常推理使用原生实现。独立重载检查仍使用原有 FP32 容差，对齐相同压缩结构的参考输出。

逐层剪后宽度可以不同，原始全局 PhiConfig 无法单独重建这种结构。**通过 phi2_structure.load_pruned_package() 加载压缩包**，不要直接用普通 AutoModelForCausalLM.from_pretrained() 加载本包。加载器用 meta 骨架和 JSON 清单重建，严格匹配 safetensors 键与形状，不反序列化任意 Python 模型对象。

## 指标边界与测试

response_nll 和 exp_response_nll 来自独立 Alpaca 响应检查样本，后者不是 WikiText 困惑度。七项基准准确率、充分恢复训练、吞吐和推理加速需要另行实验，不能从结构缩小或短验收直接推断。本入口目前不提供中途断点恢复；失败运行保留日志，修复后使用新目录从原始权重重新执行。

```bash
$P test_phi2_contract.py
```
