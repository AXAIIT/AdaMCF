# Phi-2 两阶段剪枝与 LoRA 恢复部署说明

后续变更：α 策略已从 Phi 辅助文件移入公共优化器层，并与 HESSO 共用数值内核；正式语料一致性尚未完成。当前结论见 `shared-alpha-and-corpus-alignment-20261028.md`。下文保留本次两阶段实现时的记录。

日期：2026-10-28。基线：`a97efbec5ee7150fe59534cf06e199581744dab0`。
修改仅涉及 Phi 实验入口、辅助模块、配置、测试和文档；未修改 HESSO、CRIC 时间流程、图算子、其他模型脚本或三格式实现。

## 1. α 更新的实际取样机制

已核对现有 `optimizer/alpha_scheduler.py`：build_alpha_loader 用固定种子的 randperm 从训练数据选出 α 子集，shuffle=False；prepare_alpha_eval_batches 只缓存最多 eval_batch_limit 个微批；update 再按游标轮换缓存，并逐步增加本次微批数。
因此“α 池有随机来源”成立，“每次更新都重新从完整池随机抽样”不符合该实现。hesso.py 中其他 random.shuffle 也不能作为 α 每次重新抽样的证据。

旧 Phi 入口只缓存前 alpha_eval_batches 个微批，默认 limit=1，因此实际始终只有第一个批次。
现在 Phi alpha loader 使用独立 generator（alpha_seed），shuffle=True；AlphaBatchStream 持续遍历，遍历完后重新随机排列，不把整个池常驻显存。固定的 train/alpha/eval row_id 划分仍不变。

每个 CRIC 更新边界按原先渐增微批规则取数，所有 α 正/负有限差分共用同一列表，防止把抽样变化误当成代理权重的梯度。边界之间仍冻结 α，最终仍按最终 α 重融合历史评分。
是否真正更新仍由原来的 gap/stride 等规则决定；仅成功更新所用行号计入 coverage。跨完整遍历可以再次遇到同一样本，不保证整个训练期间永不重复。

training_trace.jsonl 新增 alpha_row_ids、alpha_microbatches、alpha_supervised_tokens、alpha_unique_samples_seen。alpha_eval_batches 是上限，并不代表第一次更新立即使用这么多批；默认渐增规则仍从一个微批开始。
小 α 池仍只是功能验证，取样修正不等于自动扩大或重新划分了冻结数据。

## 2. Alpaca 与表格指标

Alpaca 是本管线选择的剪枝/恢复训练语料，不对应论文表格的一列，也不声称是 HESSO-CRIC 未公开细节中的精确训练配方。

| 表格列 | 对应评测 |
| --- | --- |
| BoolQ | 布尔问答 |
| PIQA | 物理常识选择 |
| HellaSwag | 常识续写选择 |
| WinoGrande | 指代消解选择 |
| ARC-e / ARC-c | 科学问答 easy / challenge |
| OBQA | OpenBookQA |
| Average | 七任务汇总，具体 acc/acc_norm 口径仍须冻结 |

Alpaca 内部响应 NLL 衡量已监督回答 token 的预测损失，不是上述任一任务准确率，也不是 WikiText 困惑度。
七任务的最终有标签评测 split 不用于剪枝、α、LoRA 恢复或反复挑权重。评测器仍未接入，未生成七任务分数。
论文表格来源：[HESSO-CRIC §4.6](https://arxiv.org/html/2409.09085v2#S4.SS6)。

## 3. 参考 BERT 的两阶段组织

已对照 `/zeng/GY/OTOv2_v2/test/Bert/pretrain_bert_base_uncased_pruning.py` 的剪枝、construct_subnet、加载压缩模型、新建 AdamW 微调顺序。BERT 第二阶段是普通全参微调，Phi 此处实现为冻结基座的 LoRA。

```text
原模型内部 NLL
→ 阶段一：全梯度 CRIC，无 LoRA
→ 保存稀疏检查点，construct_subnet 物理压缩和等价审计
→ 释放旧优化器和 Parameter 引用
→ 从磁盘用 OTO.load_subnet 重载压缩基座
→ 剪后内部 NLL
→ 阶段二：冻结基座，按实际压缩尺寸创建 LoRA A/B
→ 新建仅包含 A/B 的 AdamW 与独立学习率调度器
→ 恢复后内部 NLL、adapter 保存、合并模型与重载校验
```

新默认配置是 `phi2_two_stage_smoke_config.json`：剪枝尾段 post_pruning_steps=0，独立 LoRA 恢复 4 次 optimizer.step、rank=4、alpha=8、dropout=0。仅用于功能验证，不是正式收敛配方。
旧配置不含 recovery 时默认关闭恢复，既有单阶段命令和显式模式配置继续可用；post_pruning_steps 保留原来的全梯度尾段含义。
恢复批量、梯度累积、lr、warmup、rank/alpha/dropout、目标投影、eval/log 间隔均独立配置。梯度累积按有效监督 token 加权，不平均不同长度微批的均值损失。

文件职责：

- prune_phi2_hesso_cric.py：阶段一与两阶段调度，复用统一框架导出。
- phi2_alpha.py：α 池持续随机排列取样及行号覆盖记录。
- finetune_phi2_lora.py：独立恢复、冻结检查、保存和 adapter 重载，可单独执行。
- prepare_phi2_data.py / phi2_data.py：仍独立负责预处理及冻结数据读取，不混入训练脚本。

## 4. LoRA 和剪枝是否冲突

[LoRA 原论文](https://arxiv.org/abs/2106.09685) 的标准做法是冻结基座 W，仅学习低秩增量 BA。A/B 的更新正是其训练过程；同时全梯度更新 W 就不再是该标准参数高效微调设置。
如果物理裁剪 W 的行/列，已有 A/B 的依赖维度和优化器状态可能失配；仅禁止裁剪 A/B 的 rank 不能解决整个问题。专门的 LoRA 剪枝方法可联合设计这些操作，但不能直接将普通 LoRA 挂入当前原生 Phi 裁剪流程。

此次明确分开：剪枝时没有 LoRA；压缩后结构固定才新建 A/B。恢复不调用稀疏优化器、不再剪结构，不改变 rank。
既有 HESSOCRIC 中的统一 LoRA 候选保护继续保留，不增加逐模型重复规则。
恢复期间每次检查基座没有梯度，优化器只持有 A/B，训练前后比较所有基座参数 SHA256，同时确认 adapter 摘要发生变化。
恢复后为推理生成 W+BA 的合并产物是预期行为，不能把合并后的数值变化误报为“训练期间基座被更新”。

## 5. 保存、重载和日志

| 产物 | 内容 |
| --- | --- |
| compressed_model/ | 阶段一可训练压缩基座 |
| pruning_result.json | 阶段一结果，不冒充整个两阶段任务完成 |
| lora_recovery/adapter/ | PEFT 标准 adapter 与 tokenizer |
| recovery_manifest.json | 基座/adapter 文件摘要、数据摘要、LoRA 配方与精度 |
| lora_layout.json | 各投影实际 in/out 宽度和 LoRA rank |
| finetuning_trace.jsonl | 逐 optimizer.step 损失、监督 tokens、lr、梯度和内部验证 |
| recovery_result.json | 冻结/更新检查、恢复内部 NLL、合并重载验证 |
| workflow_result.json | 两阶段最终结果；没有七任务实验结果 |

表中恢复文件均位于 lora_recovery/。两阶段分别保存 UTF-8 run.log、events.jsonl、status.json；失败保存 traceback/failure.json，并保留阶段一压缩结果。
load_recovery 显式通过 OTO loader 重建逐层异构结构，校验基座和 adapter 摘要后加载 A/B；不能直接让 AutoModel 按原始全局 config 宽度重建。

恢复合并模型沿用框架 model_adapters/phi.py 的 save_pruned_package，不重新实现裁剪，也不再次调用 construct_subnet。最终 pt/safetensors/onnx 均可选择。
选择 ONNX 时阶段一先生成 safetensors 可训练中间包，恢复完成后再输出 ONNX；adapter 采用 PEFT 标准 safetensors 独立保存。
ONNX 仍是无 KV 的 logits 推理格式。导出格式不决定训练精度，本次剪枝与恢复均为 FP32、无 autocast、关闭 TF32。

## 6. 依赖变更和验收

仅在 `/opt/conda/envs/gy_adamcf_rtx6000d` 对应的个人环境新增 peft==0.17.1，安装使用 --no-deps --no-index；安装前检查 wheel 依赖均由现有包满足。
所有既有分发包版本逐一比较不变，未升级 Torch/CUDA/Transformers 或他人环境。wheel SHA256：`3d129d64def3d74779c32a080d2567e5f7b674e77d546e3585138216d903f99e`。
[PEFT 固定版本源码](https://github.com/huggingface/peft/tree/v0.17.1)。

CPU 测试 59 项全部通过：原 CRIC 23、Phi 合同 14、框架 17、新两阶段/α 5。新增集成用例内部实际完成六组小型 Phi 运行：joint×pt/safetensors/onnx、num_head×pt、head_dim×safetensors、ffn×pt。
每组均执行真实 CRIC→物理压缩→磁盘重载→LoRA 训练，验证基座不变、A/B 更新、合并模型重载和独立 adapter 重载；另验证 α 全池覆盖、相同种子复现且不污染全局 RNG、实际行号追踪、非法配方/ONNX 基座拒绝、输出拒绝覆盖与基座绑定。

证据根目录：`/zeng/GY/phi2_two_stage_20261028/`。包含 dependency_result.json、安装前后包清单、四组 CPU 日志和 status.json；evidence/ 下保存六个用例的结果/行号/LoRA布局/逐步日志。小模型权重和合成样本是临时单元夹具，测试后已回收，记录中的临时目录不是正式模型交付目录。

本次未运行真实 Phi-2 的正式 LoRA 恢复，未改原始权重、旧冻结 Alpaca 数据、既有剪枝实验或共享数据集。正式预处理 2048、池规模、训练步数和七任务评测协议仍需单独冻结；不能据此宣称任务准确率已恢复。
