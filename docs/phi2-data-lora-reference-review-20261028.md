# Phi-2 数据处理、两阶段恢复与 LoRA 统一保护

审查日期：2026-10-28。基线源码 `824ac19a2f0b6ba61e16b4b15d9b5b191f57007f`。
本次进行了官方源码审查、只读 token 长度统计和 CPU 单元回归；未开始正式剪枝、LoRA 恢复或七任务实验。

## 1. 上游资料提供了什么

| 来源与固定版本 | 已核实的实现 | 可以借鉴的范围 |
| --- | --- | --- |
| OTO `4517648c55f09e8743850427031d559767dec908` | Phi 示例使用随机模型和人工重复文本进行结构/导出验证；LoRA 示例同样不是完整语料训练 | 框架用法，不能据此补写 Alpaca 划分、长度、α 池或正式训练参数 |
| Stanford Alpaca `761dc5bfbdeeffa89b8bff5d038781a4055f796a` | 提示模板、整段编码、回答追加 EOS、提示标签 -100、动态右填充；默认长度 512 | 回答监督的 SFT 数据处理；其原始训练配方针对 LLaMA，不能称为 Phi-2 原始配方 |
| 微软 SliceGPT `183cd92d8469bf3f1e979585a08dc382692bfbed` | 显式 Phi-2 适配；Alpaca 80/10/10 划分；全文 CLM 数据加载；剪后 PEFT LoRA 微调 | 与 Phi-2 压缩直接相关的可执行参考；不是 HESSO-CRIC 官方实验代码 |
| Transformers `v4.53.0` | CLM 拼接、分块、labels=input_ids，模型内部完成因果错位 | 通用语言模型训练参考，不能直接替代回答监督标签 |

OTO 当前审查提交的 Phi 示例没有 Alpaca 加载/训练流程。该结论限定于已核实的提交和相关文件，不能推及所有历史版本。

源码链接：

- [OTO Phi 结构验证](https://github.com/microsoft/only_train_once/blob/4517648c55f09e8743850427031d559767dec908/sanity_check/test_phi2.py)
- [OTO 旧 LoRA 示例](https://github.com/microsoft/only_train_once/blob/4517648c55f09e8743850427031d559767dec908/sanity_check/test_llamav1_lora.py)
- [Alpaca 官方预处理与 collator](https://github.com/tatsu-lab/stanford_alpaca/blob/761dc5bfbdeeffa89b8bff5d038781a4055f796a/train.py)
- [SliceGPT 的 Phi-2 适配](https://github.com/microsoft/TransformerCompression/blob/183cd92d8469bf3f1e979585a08dc382692bfbed/src/slicegpt/adapters/phi2_adapter.py)
- [SliceGPT 数据加载与划分](https://github.com/microsoft/TransformerCompression/blob/183cd92d8469bf3f1e979585a08dc382692bfbed/src/slicegpt/data_utils.py)
- [SliceGPT 剪后 LoRA 恢复](https://github.com/microsoft/TransformerCompression/blob/183cd92d8469bf3f1e979585a08dc382692bfbed/experiments/run_finetuning.py)
- [Transformers CLM 示例](https://github.com/huggingface/transformers/blob/v4.53.0/examples/pytorch/language-modeling/run_clm.py)

## 2. 128-token 截断已有明确解决方案

在实际下载的 Phi-2 tokenizer 上，按上述 Alpaca 官方模板，整体编码“提示+回答+EOS”，不截断、不填充。对 52002 条原始样本先按 `.strip()` 排除 28 条空回答，审查剩余 51974 条。只使用 CPU/tokenizer，没有加载模型。

| 指标 | 实测结果 |
| --- | ---: |
| 有效样本 | 51974 |
| 完整序列中位数 / P95 / P99 | 96 / 210 / 316 tokens |
| 最长完整序列 | 1044 tokens |
| 长度大于 128 | 15112 条 |
| 长度大于 512 | 51 条 |
| 长度大于 1024 | 1 条 |
| 长度大于 2048 | 0 条 |

上述统计使用官方完整模板，和旧简化模板下“49/256 回答被截断”的口径不同，不能将两个计数直接相减。旧 proof 训练池相同 256 个 row_id 在官方模板下最长 398 tokens。

建议正式 SFT 预处理采用 `max_length=2048`，保留完整提示/回答与 EOS，取消“提示最多占一半”和截断后强行用 EOS 覆盖末 token 的策略。动态右填充到当前批次最长样本，不是把每个样本强制补到 2048。当前源数据不需要截断、丢弃长回答或滑窗；未来换数据仍须检查超长样本并明确处理策略。2048 是官方上下文容量上限，不是本次已经验证过的长序列训练显存保证。

原始数据、proof_v1 manifest 和三个冻结 split 的 SHA256 在统计前后逐一一致；旧预处理、旧冻结数据没有被改写。

## 3. 两种公开的训练目标要明确选择

### 回答监督 SFT：Alpaca 官方路径

1. 将 instruction 和可选 input 按官方模板组成提示，回答追加真实 EOS。
2. 对完整字符串编码，得到 input_ids；不要先任意切掉提示头部。
3. labels 与 input_ids 同长，提示部分设 -100，回答和真实 EOS 保留 token id。
4. 动态右填充：补齐 input_ids，padding 的 labels 设 -100；attention_mask 根据真实长度构造。
5. tokenizer 复用下载模型对应的版本；不为照搬 LLaMA 示例而擅自增加 Phi 词表或换模板。

若 pad 与 EOS 共用 id，不能用 `input_ids != eos_id` 生成 mask，否则会屏蔽真实 EOS。现有按长度生成的 mask 应保留。
整段编码与单独提示编码在子词边界可能不同；正式实现应检查提示/回答边界，不能假设两次 tokenization 直接拼接必定与整段编码相同。

### 全文 CLM：SliceGPT 的 Phi-2 相关恢复路径

其 Alpaca 加载使用已格式化的 text，训练加载器可拼接样本到指定长度；labels 从 input_ids 复制，即对提示与回答都做 next-token 预测。我们若移植此路径，仍应将 padding 的 labels 设为 -100；不能原样照抄在不同填充设置下的细节。

全文 CLM 更贴近基座语言模型的知识恢复目标；回答监督 SFT 更贴近指令回答目标。二者都可以用于后续恢复，但损失含义不同，不能合并成一个未经说明的“官方默认”。当前管线是回答监督验证版，不能声称已按 SliceGPT 全文目标训练。

建议先沿用回答监督目标，把长度与两个阶段落地；若研究重点改为最大程度贴近 SliceGPT 的 Phi-2 恢复协议，再单列全文 CLM 对照。这是建议，尚未更换训练目标、重新切分数据或生成正式训练集。

## 4. α 池、模型选择与正式评测

Alpaca 官方 SFT 没有自适应 α 池，也没有唯一公认的 α 样本数。不能把我们新增的 α 集包装成上游标准。
SliceGPT 明确使用 seed=42，先分出 20%，再把这一部分对半分为内部 validation/test，可直接借鉴其 80/10/10 思路。

可执行的后续建议：先对有效源样本固定 row_id 与去重规则，再按上述 80/10/10 拆分；仅从 80% 训练部分划出独立 α 池（如 1024 条，属于本方法新增设置），验证部分用于模型选择，内部 test 仅作最终内部检验。若实际训练只抽子集，也要保存被选 row_id 和 token 总量，不能只记录“样本数”。所有阶段先划分再分块/打包，同一原始 row_id 的所有片段归属同一池。

当前代码一次缓存前 `alpha_eval_batches` 个批次，反复复用；默认 batch=1、limit=1，因此“α 池有 1024 条”也可能实际只看 1 条。正式配置必须让有效评估批次数和池覆盖率匹配，记录真实独立样本数及有效监督 tokens；若采用轮换子集，需要另行明确轮换规则。先保留已确认的 CRIC 周期边界更新和最终 α 重融合，不改时间流程。

BoolQ、PIQA、HellaSwag、WinoGrande、ARC-e、ARC-c、OBQA 保持既定官方有标签 split，使用各自标准任务模板/候选项评分；不将七任务验证/测试答案混入剪枝训练、α 或恢复训练。Alpaca 内部损失不等于这七项任务准确率。若将来要逐任务监督微调，需另立训练/验证/测试协议，不能与零样本评测混报。

## 5. BERT 的两阶段代码可以借鉴组织，不能照抄任务损失

已核对 `/zeng/GY/OTOv2_v2/test/Bert/pretrain_bert_base_uncased_pruning.py`：

- 阶段一用 `oto.hesso()`；阶段结束调用 `construct_subnet()` 并重新加载压缩模型。
- 阶段二重新创建 AdamW 和学习率调度器，训练压缩模型；它没有附加 LoRA，是普通微调。
- BERT 使用 SQuAD 起止位置标签；Phi 使用因果语言模型 labels，不能复制 start/end loss 或对应 EM/F1。

Phi 的目标组织应为：

```text
原模型评估
→ 全梯度 CRIC 剪枝与周期边界 α 更新
→ construct_subnet 物理压缩，保存并重载可训练结构包
→ 剪后评估
→ 在实际压缩维度上附加 LoRA，冻结基座，仅训练 A/B
→ 新建恢复优化器与调度器
→ 恢复后评估、保存 adapter 与基座版本对应关系
```

模型的物理压缩会更换 Parameter，恢复阶段必须重新建立优化器。pt/safetensors 可训练包用于恢复；ONNX 仍为推理产物。
现有 post_pruning_steps 是同一 CRIC 优化器的全梯度尾段，本次未将其重命名为 LoRA，也未声称恢复脚本已经接入。

## 6. LoRA 统一保护的实际修改与边界

保护放在 `only_train_once/optimizer/hesso_cric.py` 的公共 HESSOCRIC 类，按标准路径段 `lora_*` 识别 A/B、embedding LoRA 等张量。模型算子必须提供真实、完整的 p_names；非标准适配器命名需显式扩展统一规则，不能凭张量形状猜测。

1. LoRA 参数标为 NO_PRUNE，已有 NO_UPDATE 不被降级；纯 LoRA 组不计入剪枝预算。
2. 在五准则统计视图中排除 LoRA；原评分函数并非都能正确跳过 NO_PRUNE，因此仅设标记不足以保证排除。
3. 试探投影、α 掩码、稀疏化、最终置零和辅助依赖传播统一过滤被保护的张量身份。
4. 保留 A/B 的正常梯度更新；统计临时视图在 finally 中恢复完整参数列表。
5. 加载检查点前验证 LoRA 保护和组预算；不安全旧检查点明确拒绝，不静默改写历史预算。
6. 移除 Phi 中重复的参数名拒绝检查，保留其原生 nn.Linear、MHA、RoPE/布局等结构检查。

未改 hesso.py、旧图算子、其他模型脚本、三格式实现或 CRIC 时间流程。排除 LoRA 候选不等于任意包装层都能物理裁剪；基座缩小会改变 A/B 的输入输出维度，本次仍推荐压缩后再挂 LoRA。

CPU 回归：原公共 CRIC 18 项 + 新 LoRA 策略 5 项、Phi 合同 14 项、框架集成 17 项，共 54 项通过。新增测试覆盖不兼容 rank 的混合组评分、普通梯度、纯 LoRA 预算、辅助传播、试探与最终置零、异常恢复、检查点兼容性。集成测试覆盖统一入口；原三模式、三格式、小模型与旧算子测试继续通过。

## 7. Phi-2 原始预训练精度的证据边界

[Phi-2 官方模型卡](https://huggingface.co/microsoft/phi-2) 与[微软发布说明](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)
公开了 next-token 目标、2048 上下文、训练 token 数、A100 和软件栈；在本次查阅资料中没有明确披露原始预训练的 FP32/FP16/BF16 计算与优化器状态组合。
[官方数据说明卡](https://huggingface.co/microsoft/phi-2/blob/main/data_summary_card.md) 介绍数据来源，也不是可直接运行的预训练数据管线。

| 可核实事实 | 不能据此推导 |
| --- | --- |
| 发布权重为 FP16；推理示例使用 auto dtype | 原始预训练全程使用 FP16 |
| Alpaca 官方示例使用 BF16 参数 | Phi-2 原始预训练使用 BF16 |
| SliceGPT 恢复脚本 dtype 默认 fp16，可选 fp32 | HESSO-CRIC 或 Phi-2 原始预训练使用同一精度 |
| Azure Phi-2 微调规格 precision=16 | Phi-2 原始预训练的完整混合精度策略 |

[Azure 微调规格来源](https://github.com/Azure/azureml-assets/blob/main/assets/models/system/microsoft-phi-2/spec.yaml)。Phi-2 模型卡还明确提醒 FP16 attention 溢出问题。

因此现在不能宣称“照原始训练精度复现”。保留已验证 FP32 剪枝基线合理；如后续选择 BF16 autocast 加 FP32 重要性统计，需要验证 loss/梯度有限性、评分排序、α 稳定性和结构导出误差。它是我们的混合精度方案，不是已证实的原作者设置；本次未调整精度或任何公共 CUDA/包版本。

## 8. 交付与后续实施范围

服务器运行证据：`/zeng/GY/phi2_data_lora_review_20261028/`，包含三组 CPU 日志、status.json、只读 length_audit.json、阶段副本及部署前逐文件备份。旧运行证据和原始数据不作为源码提交。
代码目录：`/zeng/GY/OTOv2_v2`；本报告位于其 `docs/`。

已完成：来源核对、长度问题定量解决方案、BERT 两阶段对照、LoRA 保护集中化和 CPU 回归。
尚未实施：正式 2048-token 预处理版本、正式划分与 α 覆盖策略、独立 LoRA 恢复入口、混合精度方案及七项任务实验。后续实施必须保留当前验证数据，并为新配方生成独立目录、配置、manifest 和完整日志。
