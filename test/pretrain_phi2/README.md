# Phi-2 的 AdaMCF / HESSO-CRIC 结构剪枝

当前实验按用户 2026-10-29 最新要求：仅使用本框架与表格七个数据集，train 用于剪枝/恢复，官方 validation/test 仅评测；不用原版优化器，不用 Alpaca。文献表格只作外部参考。
当前运行入口与数据配方见 `docs/phi2-seven-task-run-20261029.md`。下方旧 Alpaca 命令保留为历史功能验证说明，不是本轮实验命令。

目录位置：`/zeng/GY/OTOv2_v2/test/pretrain_phi2`。

沿用 BERT/VGG 的模型和剪枝产物布局。数据预处理、结构适配、训练与独立验收分别实现；剪枝使用既有个人环境，七任务预处理/评测使用隔离 venv 的固定版官方 lm-eval。

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
| `phi2_precision.py` | 原生 FP32 漂移记录和严格 FP64 结构对照，诊断后恢复原生实现 |
| `prune_phi2_hesso_cric.py` | 运行完整 CRIC 并调用框架导出；--export-only 从已有稀疏检查点独立导出与审计 |
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
- num_head：联动 Q/K/V 的完整头行、对应 bias 和 dense 输入列，保留每头宽度。
- head_dim：同步压缩每个头的相同维度；每个逻辑单位为两维，旋转区按真实 RoPE 配对，非旋转区按相邻两维配对。保留原位置频率与 attention scaling，不按新 head_dim 重算。
- residual hidden size、embedding、LayerNorm、词表维度不做结构剪枝；这些参数与输出 bias 仍可训练。
- 所有参数恰好归属一次。拒绝 GQA、量化 Linear、空保留组及非法结构索引。

原生 Phi 的结构实现已迁入框架 `only_train_once/model_adapters/phi.py` 和 `operation/native_phi.py`，由 `OTO` 自动分派。训练使用 `oto.hessocric()`，物理裁剪和结构包保存使用 `oto.construct_subnet()`，重载使用 `OTO.load_subnet()`。实验代码直接导入框架结构实现，独立兼容文件已清理。旧 `PhiMHA`、CNN、BERT、YOLO 等继续使用原有 Graph/ONNX 路径；此次新增原生模型结构适配不代表 ONNX 追踪器已支持任意 Phi 实现。

五准则、自适应 α、周期内冻结和最终历史重融合使用此前已适配的 CRIC。joint 有 64 个可剪枝参数组，投影长度 384 步；三种独立模式各有 32 个可剪枝参数组，投影长度 192 步。投影试探步不等于普通梯度更新步。

## 独立剪枝模式

| pruning_mode | 候选结构 | 不做结构剪枝的部分 | Phi-2 逻辑候选总数 |
| --- | --- | --- | --- |
| joint（兼容原默认） | FFN 神经元 + 注意力整头 | head_dim | 328704 |
| ffn | FFN 神经元 | 整头、head_dim | 327680 |
| num_head | 注意力整头 | FFN、head_dim | 1024 |
| head_dim | 每头双维组，共 40 组/层 | FFN、整头 | 1280 |

非候选参数保留训练资格，候选范围决定 CRIC 的选组预算，不手工指定最终删除索引。目标稀疏率针对各自逻辑候选，不代表三种模式有相同参数压缩率。原有 joint 调用保持可用，旧版结构包继续支持加载。

head_dim 需要 qk_layernorm=False；当前原始 Phi-2 满足该条件。旋转区原本为每头 32 维，按 (i,i+16) 配对。压缩后局部注意力模块从原生 RoPE 常数中选择保留坐标，保留原来的 1/sqrt(80) 缩放。正常推理与生成使用原生计算流程，仅增加坐标选择；不修改安装包。

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

分别验收三种模式时，使用对应配置和不同输出目录：

```bash
$P prune_phi2_hesso_cric.py --data-dir "$DATA" --config phi2_ffn_smoke_config.json --output runs/my_ffn_run
$P prune_phi2_hesso_cric.py --data-dir "$DATA" --config phi2_num_head_smoke_config.json --output runs/my_num_head_run
$P prune_phi2_hesso_cric.py --data-dir "$DATA" --config phi2_head_dim_smoke_config.json --output runs/my_head_dim_run
```

每个目录完成后再执行 verify_phi2_pruning.py。验收会核对逐层 Q/K/V、dense、FFN 的实际尺寸及 KV cache 的实际头数和 head_dim，并检查缓存续写、生成和反向传播。

训练结束后先保存稀疏检查点。若导出诊断失败且尚未产生 compressed_model，可独立重试：

```bash
$P prune_phi2_hesso_cric.py --export-only --output runs/my_phi2_run
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
| compressed_model/ | 按格式保存 model.pt / model.safetensors / model.onnx（含外置权重），以及结构、配置和分词器 |
| pruning_result.json | 实际压缩率、误差、显存、响应损失 |
| independent_verification.json | 新进程输出对齐、缓存生成与反向传播结果 |

成功条件：CRIC 和稀疏化结束；最终剪枝组精确归零；真实矩阵及参数量缩小；FP64 结构对照在 atol/rtol=1e-8 下通过；新进程重载与压缩模型的原生 FP32 参考对齐，并通过缓存生成和反向传播。

真实深层 Phi 的矩阵尺寸改变会改变 FP32 GEMM 求和顺序，原生 FP32 稀疏/压缩 logits 可能不满足逐元素 2e-4 门槛；该失败标志和实测误差保留，不改写为通过。为判别结构错误与浮点漂移，另外执行严格 FP64 对照，诊断作用域内将 attention softmax 也提升为 FP64，RoPE 使用双方相同的原生位置常数。作用域退出即恢复 FP32，安装包、训练和正常推理使用原生实现。独立重载检查仍使用原有 FP32 容差，对齐相同压缩结构的参考输出。

逐层剪后宽度可以不同，原始全局 PhiConfig 无法单独重建这种结构。**通过 OTO.load_subnet() 加载压缩包**，不要直接用普通 AutoModelForCausalLM.from_pretrained() 加载本包。加载器用 meta 骨架和 JSON 清单重建，严格匹配 pt/safetensors 的键与形状；ONNX 则加载为 CPU Runtime 推理接口，不反序列化任意 Python 模型对象。

## 指标边界与测试

response_nll 和 exp_response_nll 来自独立 Alpaca 响应检查样本，后者不是 WikiText 困惑度。七项基准准确率、充分恢复训练、吞吐和推理加速需要另行实验，不能从结构缩小或短验收直接推断。本入口目前不提供中途断点恢复；失败运行保留日志，修复后使用新目录从原始权重重新执行。

```bash
$P test_phi2_contract.py
```

## 框架统一接口

```python
from only_train_once import OTO

# 自动识别当前 Transformers 原生 PhiForCausalLM，无需 ONNX dummy_input。
oto = OTO(model=model, pruning_mode="head_dim")  # joint / ffn / num_head / head_dim
optimizer = oto.hessocric(
    lr=1e-4, first_momentum=0., target_group_sparsity=.1,
    start_cric_step=3, sampling_steps=2, max_cycle_period=2,
    hybrid_training_steps=8,
)
# 训练及周期边界 α 更新沿用 prune_phi2_hesso_cric.py 的循环。
# CRIC 完成、混合稀疏化结束且被剪结构精确归零后：
oto.construct_subnet(out_dir="runs/example", tokenizer=tokenizer,
                     ckpt_format="safetensors")
compressed = OTO.load_subnet(oto.compressed_model_path, device="cuda:0")
```

该入口会原地改变模型结构；导出后旧优化器的 Parameter 引用不能继续使用。
原生 Phi 输出 `out_dir/compressed_model/`，默认同时保存 `out_dir/sparse_model/`。
已有稀疏检查点可通过 `pruning_plan=plan` 导出，框架会先检查所有删除位置精确为零。
原生压缩产物可用 `ckpt_format="pt"`、`"safetensors"`、`"onnx"` 显式选择；
框架默认 `"torch"` 等同于 `"pt"`，不再隐式输出 safetensors。pt 和 safetensors
保存可训练权重与结构 JSON；ONNX 为无 KV cache 的 logits 推理图，不能直接做 LoRA
微调。ONNX 大权重文件必须随 .onnx 主文件一起复制。旧 safetensors 包继续支持加载。
旧图模型的文件命名、torch/ONNX/HF 保存分支和旧算子注册保持原行为。
详细兼容边界、回归测试和模块职责见 `docs/native-phi-adapter.md`。

## 格式选择与日志

训练模式：在现有命令中加入 `--export-format pt` / `onnx` / `safetensors`。
仅导出：`$P prune_phi2_hesso_cric.py --export-only --output runs/EXISTING_SPARSE_RUN --export-format onnx`。
每个输出目录只发布一种格式；已有 compressed_model 时拒绝覆盖。
CLI 旧配置默认 safetensors，API 默认 torch/pt；日志与结果会记录实际格式。
ONNX：`session = OTO.load_subnet(package, device="cpu")`，`logits = session(input_ids, attention_mask)`。
其余两种格式返回 PyTorch 模型。格式与训练精度分开，当前剪枝与独立 LoRA 恢复均固定 FP32。

终端与 run.log 同步输出阶段信息，events.jsonl 保存结构事件，training_trace.jsonl 保存每步记录。
新数据预处理过滤空响应并保存 prepare.log；不会重写已有 proof_v1 数据。
evaluation_protocol.json 是七任务评测的审查提案，不是已经可执行的评测器，也不代表任务实验已开始。
正式实验前需冻结长度/截断、训练与 α 池、LoRA 配方、评测实现和指标口径；见框架 docs 的协议审查文档。

后续来源审查与只读长度统计见 `docs/phi2-data-lora-reference-review-20261028.md`：
官方 Alpaca 模板下，当前有效数据最长 1044 tokens，2048 上限可完整保留；
旧 128-token 预处理和配置仍是功能验证版本，没有被静默替换为正式协议。
LoRA 候选保护由 `HESSOCRIC` 统一执行，覆盖评分、预算、置零、辅助传播和恢复检查；
无需逐模型复制参数名判断。独立恢复入口已接入，见下文；旧 post_pruning_steps 仍是全梯度尾段。

## 两阶段剪枝与 LoRA 恢复

默认配置现为 `phi2_two_stage_smoke_config.json`，组织参考 BERT 的剪枝/压缩重载/新优化器微调流程：
阶段一全梯度 CRIC，不附加 LoRA；阶段二从磁盘重载压缩基座、冻结基座、按实际矩阵尺寸附加 A/B，使用独立 AdamW。
新 smoke 配置的 post_pruning_steps=0、恢复仅 4 步，是功能验证配方；旧显式配置继续只剪枝，不会被自动开启 LoRA。

```bash
# DATA 指向已经冻结的数据目录。正式配方尚未确定，不应直接把 smoke 当正式实验。
$P prune_phi2_hesso_cric.py --data-dir "$DATA" --config phi2_two_stage_smoke_config.json \
  --output runs/my_two_stage --export-format safetensors

# 已有可训练压缩包时，可单独运行阶段二，不必重新剪枝。
$P finetune_phi2_lora.py --base-dir runs/my_phi2_run/compressed_model \
  --data-dir "$DATA" --config phi2_two_stage_smoke_config.json \
  --output runs/my_recovery --export-format pt
```

阶段二依赖 PEFT，已在个人环境验证 `requirements-lora.txt` 的 0.17.1；不修改公共 Torch/CUDA。
恢复使用 train，内部验证使用 eval；alpha 不参与恢复梯度，七项 benchmark 也不作为此处训练/选模数据。
梯度累积按真实监督 token 加权，日志保存损失、token 数、学习率、梯度范数与内部 NLL。
逐参数 SHA256 检查确认训练期间基座不变；只允许 A/B 出现梯度和更新。

α loader 由独立种子 alpha_seed 打乱；取批、渐增数量、评估内复用、成功覆盖计数由 optimizer/alpha_data.py 与两个公共 α 调度器统一负责。Phi 只提供 DataLoader 和 AlphaBatch 数据/行号适配，不再有独立 phi2_alpha.py 策略。
HESSO 与 CRIC 共用 advance_alpha 数值内核和边界置信统计；两者各自的预算/时间轴与 CRIC 最终历史重融合保持原流程。旧 HESSO 固定缓存列表调用继续可用，也可传入通用 AlphaEvaluationPool；数据源类型和配置相同才保证取样一致。
一次 α 差分更新内所有扰动复用同一批样本；training_trace.jsonl 记录实际行号、微批、监督 token 与累计独立覆盖量。gap 等条件跳过的更新不计入覆盖。

产物：`compressed_model/` 是阶段一基座，`lora_recovery/adapter/` 保存 A/B，`recovery_manifest.json` 绑定基座和 adapter 摘要；
`lora_recovery/recovered_model/` 保存合并后的完整结构包，可选 pt/safetensors/onnx。ONNX 模式先保留 safetensors 中间基座供训练，最终恢复产物再导出 ONNX。
合并只发生在恢复完成后，训练期间的冻结检查与合并产物中的 W+BA 不是同一数值状态。

适配器必须用框架结构 loader 重建基座，禁止 AutoModel 根据原始全局宽度猜测异构结构：

```python
from finetune_phi2_lora import load_recovery
model = load_recovery("runs/my_two_stage/lora_recovery", device="cpu")
```

workflow_result.json 记录两阶段结果；单独剪枝成功不再使等待恢复的总体任务标记为完成。阶段二失败时保留阶段一结果及独立失败日志。
本次仅在小型 Phi 上完成四类剪枝模式、三格式和真实 LoRA 的 CPU 回归；未启动真实 Phi-2 正式恢复或七任务评测。

## 正式实验的数据一致性

Alpaca-only 仍只用于功能验证，不能称为论文实验数据已对齐。公开证据链为 HESSO-CRIC 参考 LoRAShear，后者列出了 OpenWebText、2022 Wikipedia、Gutenberg、BookCorpus，以及指令恢复用 Alpaca。
当前未找到作者的 Phi-2 精确语料子集/混合比例/样本索引；四类预训练语料也未在当前共享数据清单中发现。WikiText 不能替代 2022 Wikipedia，BookCorpusOpen 不能静默替代 BookCorpus。
corpus_alignment.json 明确记录已确认来源、当前文件摘要和待核实项，formal_alignment_complete=false；正式预处理/训练继续等待数据对齐，已有 proof 与任务 split 不改。
详细公共策略与数据审查见 `docs/shared-alpha-and-corpus-alignment-20261028.md`。
