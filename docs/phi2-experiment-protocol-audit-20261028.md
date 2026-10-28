# Phi-2 数据、LoRA、精度与导出协议审查

审查日期：2026-10-28。未启动七项正式任务实验，也未进行新一轮剪枝训练或 LoRA 恢复。

## 1. 当前代码实际完成什么

当前是原生 Phi-2 的 **FP32 全梯度 HESSO-CRIC 剪枝功能验证**，已有物理压缩、
严格结构诊断、可训练格式重载、缓存生成和反向传播证据。`post_pruning_steps`
是联合训练流程的尾段，仍使用原优化器，不是独立的 LoRA 微调。

当前入口没有 get_peft_model、LoraConfig 或 LoRA 恢复训练循环。个人环境实测未安装
标准 peft、lm_eval 分发包；仓库的旧 LoRA 代码没有被本次原生 Phi 管线调用。
已下载七项数据不等于已经接入或完成七项任务评测。`evaluation_protocol.json`
明确标为审查提案，runner_implemented=false，不是可直接执行的评测器。

## 2. 数据是否合理，哪些属于标准做法

### 2.1 官方评测划分

| 任务 | 当前保留划分 | 样本数 | 标签/输入注意事项 |
| --- | --- | ---: | --- |
| BoolQ | validation | 3,270 | 本地 answer 为 bool；映射 False=0、True=1，对应 no/yes |
| PIQA | validation | 1,838 | label 为 0/1，两个候选 sol1/sol2 |
| HellaSwag | validation | 10,042 | label 是数字字符串；保留官方大小写、括号等清洗 |
| WinoGrande XL | validation | 1,267 | answer 为 1/2；按官方前缀/后缀似然协议打分 |
| ARC-Easy | test | 2,376 | 按 choices.label 查找 answerKey，不能只按字母减 A |
| ARC-Challenge | test | 1,172 | 与 ARC-Easy 使用同一协议、不同配置 |
| OpenBookQA main | test | 500 | 使用 main 的 question_stem，不擅自加入额外事实或检索结果 |

共 20,465 条。划分选择与所审阅的 lm-evaluation-harness 任务定义一致；不重新
随机拆分这些评测集，不用它们训练、选剪枝结构、更新 α 或逐轮挑最佳权重。
这与 BERT 的任务特定监督训练场景不同，不能直接照搬每轮验证/选权重流程。

数据源版本、原文件摘要已冻结。本地 BoolQ 来自 google/boolq，所审阅 harness 使用
aps/super_glue/boolq；本地 PIQA 来自 ybisk/piqa 的固定转换版本，harness 使用
baber/piqa。字段与公开行数相容，但未逐条证明不同发布源完全一致，不能宣称已经
原封不动运行官方默认下载路径。后续本地 task 适配需保留官方模板、清洗、候选评分
与指标实现，仅替换数据来源和必要字段映射，并记录其版本与摘要。

参考实现固定在 [lm-evaluation-harness d6de816](https://github.com/EleutherAI/lm-evaluation-harness/tree/d6de81643928d653435c431bae19945d41d32520)。
具体 task YAML 和辅助预处理文件的链接、标签映射与路径见 evaluation_protocol.json。
BoolQ/WinoGrande 主定义为 acc，其余任务同时提供 acc、acc_norm；主指标、few-shot 数、
是否使用 chat template 必须在运行前固定。当前提案是 zero-shot、不加 chat template、
七项 acc 的等权平均，另完整保存 acc_norm。它是我们的可复现提案，不能冒称论文
已经明确公布的同一配置；也不能看完结果再为每项挑较高指标求平均。

### 2.2 Alpaca 的内部角色与处理

Alpaca 原数据没有官方 train/alpha/eval 三分法。从训练来源按固定种子划出互斥角色
是合理做法；当前已核对原始行索引、实际 JSONL 顺序、文件摘要和 token 长度。
标签保持与 input_ids 对齐，causal shift 由模型内部完成；prompt/padding 标为 -100，
attention_mask 按真实长度构造，真实 EOS 不会因为兼作 pad ID 被误屏蔽。
这属于常见的 response-only SFT 处理，但不是唯一公认模板，更不是七项分类任务
应采用的标签格式。七项任务应按选项条件对数似然评分，不应套 Alpaca 的响应 NLL。

当前 proof_v1 为 train=256、alpha=2、eval=8、max_length=128。α 实际每次仅使用
一个 microbatch；它只验证自适应 α 机制可运行，不能作为稳定的正式验证池。
内部 eval 的 NLL 来自 225 个监督 token，exp(NLL) 也不是 WikiText 困惑度。

只读审查实测：

- 原始 Alpaca 52,002 条，按 strip 后字段核对，完整记录和 instruction/input 键均未发现重复；空响应 28 条。
- 当前 256/2/8 三个池按 instruction/input 键无交叉，三个已选池均未包含空响应。
- train 有 6/256 条 prompt、49/256 条响应被截断；alpha 有 1/2 条响应、eval 有 1/8 条响应被截断。
- 当前规则将 prompt 限到最多半个窗口，截断响应末尾强制 EOS。这是 proof 为控制成本采用的规则，不能直接标成标准完整训练配方。

新预处理代码先过滤空响应再固定种子抽样，保留原始 row_id，并记录过滤数与 prepare.log。
原始数据、既有 proof_v1 目录和其清单没有重写。正式实验还需统计长序列覆盖率，冻结
总长度与截断/EOS 策略、足够规模的 α/模型选择池及训练语料；长度调整必须结合显存
验证，不能直接把 Phi-2 上下文上限当成 FP32 全梯度训练一定可行的长度。

本次没有执行七项任务与训练语料的全面污染排查；内部互斥不等于已证明所有预训练
和下游样本无污染。正式运行前还需固定相应检查与报告范围。

## 3. LoRA 恢复与禁止误剪

[HESSO-CRIC §4.6](https://arxiv.org/html/2409.09085v2#S4.SS6) 明确说 HESSO/HESSO-CRIC
使用全梯度，并将几种 LoRA 方法作为比较对象。“设置参考 LoRAShear”不能推导出
HESSO-CRIC 本身一定采用 LoRA 剪枝或 LoRA 恢复。[LoRAShear §3.5、§4.1](https://arxiv.org/html/2310.18356v2)
另有 LoRA 知识恢复及多种训练语料的设计。我们不能混淆两种算法或把论文未披露的
Phi-2 训练比例、采样数和微调超参数自行补成官方事实。

遵守“LoRA 矩阵不参与剪枝”的方案是：

1. 对无 LoRA 的原生基座进行全梯度 HESSO-CRIC，完成物理裁剪并保存结构。
2. 加载压缩后的 pt/safetensors 可训练模型，在已经固定的实际投影维度上附加新 LoRA。
3. 冻结压缩基座，只把 LoRA A/B 放入普通恢复优化器；A/B 需要训练，但不进入稀疏化优化器，不裁剪其 rank。
4. 使用独立内部验证决定训练计划/模型选择，最后对原始、剪后、LoRA 恢复后三个阶段分别跑同一套七任务。

后续审查已将 LoRA 候选排除统一移至 HESSOCRIC，移除了 Phi 中重复的参数名拒绝检查；
覆盖评分、预算、试探/最终置零、辅助传播和检查点恢复。详见
`phi2-data-lora-reference-review-20261028.md`。这不代表物理导出已兼容 PEFT 包装层。
若在裁剪前已挂 LoRA，仅设置“不评分”也不足以保证维度安全：基座行列缩小后，A/B
的输入/输出尺寸仍需同步处理。因此本方案选择剪后再挂 LoRA，不在现有稀疏组中混入
适配器。后续恢复脚本、兼容版本 PEFT 依赖和保存/合并规则尚未实现或执行，不能声称
已有 LoRA 恢复精度结果。这会是我们的恢复方案，应在论文实验设置中明确标注。

## 4. FP32 与论文证据边界

| 环节 | 当前实测/实现 | 结论范围 |
| --- | --- | --- |
| 原始下载权重 | FP16 存储 | 不决定加载后的训练精度 |
| CRIC 剪枝及尾段训练 | 显式 torch.float32，无 autocast，关闭 TF32 | 基座权重、梯度按当前 FP32 路径计算 |
| LoRA 恢复训练 | 尚未接入 | 不能说已经用 FP32 完成微调；后续需固定基座、A/B、优化器状态的 dtype |
| FP64 结构审计 | 临时诊断、随后恢复 FP32 | 不是 FP64 训练，也不表示原生 FP32 零舍入误差 |
| pt / safetensors / ONNX | 三种序列化/执行格式 | 格式后缀不决定 FP32/FP16 |

HESSO-CRIC 文中“full gradient”描述哪些参数参与求导，不等于“FP32”。所审阅论文
和公开 Phi sanity 示例不足以确定正式 Phi 实验的训练、推理和优化器状态精度。
官方示例的 export_float16=False 只是该示例的导出选项，不是论文训练精度的证据。
因此目前只能如实写“我们使用 FP32”，不能写“论文也同样使用 FP32”。

## 5. 注释、日志与后续每任务结果

借鉴 BERT 现有“加载—剪前—剪枝—剪后—压缩总结—恢复”分阶段表达，Phi 新增：

- run.log：UTF-8、UTC+8 时间、终端同步的阶段说明。
- events.jsonl：同源结构化事件，记录配置、数据角色/数量/摘要、dtype、可训练/LoRA 参数量、候选预算、导出格式与结果。
- training_trace.jsonl：每步 loss、学习率、梯度、CRIC 周期、α、更新标记、已剪逻辑组与耗时；阶段边界和定期日志记录显存。
- prepare.log：空响应过滤、原始/可用数量、划分规模与摘要。
- failure.json 和 traceback：失败与完成分开，拒绝覆盖既有压缩结果。

现有代码只报告内部响应 NLL 和结构验证，不输出虚构的 F1、七任务准确率或 LoRA
恢复收益。后续每个任务、每个阶段应单独保存结果、逐样本选项似然/预测/标签、
样本数、模板/数据/模型/分词器版本与摘要、few-shot/chat/dtype、运行耗时及失败记录。
记录 acc/acc_norm 和预先选定的主指标，最后等权汇总。不能仅保留一个平均数。

## 6. 三种格式的真实语义

旧框架通用图路径原本有 torch(.pt)、onnx，以及 export_huggingface_format 分支，
所以“原来绝对只有 pt/onnx”不完整；HF 的 save_pretrained 也可保存 safetensors。
新原生分支此前把 torch 参数也输出为 safetensors，存在 API 语义不一致，本次已修正。

| 选择 | 文件 | 使用 |
| --- | --- | --- |
| pt / torch | model.pt + architecture.json + config.json | tensor state_dict，通过 OTO.load_subnet 重建，可继续微调 |
| safetensors | model.safetensors + 同样的结构和配置 | 同样是可训练权重，保持旧包加载兼容 |
| onnx | model.onnx + 必需外置权重 + 结构和配置 | 无 KV cache 的动态 batch/sequence logits 推理图 |

API 使用 construct_subnet(ckpt_format=...)；剪枝/独立导出 CLI 使用 --export-format。
CLI 旧配置默认 safetensors，框架 API 默认 torch/pt，实际格式写入日志和结果。
ONNX 用 CPUExecutionProvider；不宣称可直接训练、LoRA 微调、缓存生成或速度最优。
超过 2GB 的 ONNX 权重不能只拷贝主文件，必须把包内外置数据一并保留。验证记录包括
整个结构包的文件摘要。原生 pt 为纯权重字典；旧图模型原有整模型 pt 行为不变。

真实 Phi-2 初次 ORT 默认高级优化验证中出现少量超出预设门槛的 logits，失败日志保留。
本次固定为关闭 ORT 图融合、确定性计算，按相同门槛重新验证，不放宽 atol/rtol。
跨后端比较为 atol=rtol=1e-3，即 |误差| <= atol + rtol*|参考|；同一 PyTorch 结构重载
与严格 FP64 结构门槛仍分别记录。格式测试仅证明导出/加载行为，不证明任务准确率。

## 7. 正式实验前必须固定

训练语料及样本、序列截断和 EOS、α 与模型选择池、LoRA rank/alpha/target_modules/
学习率/轮数、PEFT 与评测器版本、seven-task 主指标、种子重复、资源预算。
还需区分“逻辑组稀疏率”和“全模型参数减少比例”：当前 10% 候选组只对应约 3%–6%
的参数减少，不能直接标成论文 20%/25%/30%。某些单一剪枝模式也可能达不到同一
全模型目标，需要预先说明候选范围或选用 joint，而非事后混用压缩率口径。

参考来源：

- [HESSO-CRIC 论文](https://arxiv.org/html/2409.09085v2#S4.SS6)。
- [LoRAShear 论文](https://arxiv.org/html/2310.18356v2#S4.SS1)。
- [官方 Phi sanity 示例](https://github.com/microsoft/only_train_once/blob/4517648c55f09e8743850427031d559767dec908/sanity_check/test_phi2.py)。
- [Transformers 4.53 模型保存文档](https://huggingface.co/docs/transformers/v4.53.0/en/main_classes/model#transformers.PreTrainedModel.save_pretrained)。
- [PEFT LoRA 文档](https://huggingface.co/docs/peft/en/developer_guides/lora)。
- [PyTorch 2.7 ONNX 导出](https://docs.pytorch.org/docs/2.7/onnx_torchscript.html)、[ONNX external data](https://onnx.ai/onnx/repo-docs/ExternalData.html)。
