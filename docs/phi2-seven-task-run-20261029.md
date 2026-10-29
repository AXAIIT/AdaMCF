# Phi-2 七任务流程实验

日期：2026-10-29；源码起点 0765a462bd501890f2dc7ad5b323404347ee13d0。

## 本轮范围

依用户最新要求：仅用我们已适配的 HESSO-CRIC 框架，保留五准则、自适应 α、CRIC 周期边界更新和最终历史重融合。不接入或运行原版优化器；此前本地准备的上游入口已撤回，未部署、未训练。
只使用表格中的 BoolQ、PIQA、HellaSwag、WinoGrande、ARC-e、ARC-c、OBQA，不使用 Alpaca，也不下载其他训练语料。作者没有明确披露用这七项 train 剪枝，本轮按用户批准的备用方案执行，不冒充论文精确配方。
表格数字保存于 paper_phi2_reference.json，是文献参考，不是本地未剪枝测量。

## 数据与划分

数据目录：`/adminGroup/datasets_source203/llm_pruning/processed/phi2_seven_task_train_s42_v1`。

| 角色 | 每个任务 | 七任务合计 | 来源 |
| --- | ---: | ---: | --- |
| 剪枝/LoRA 训练 | 768 | 5376 | 官方 train |
| 独立 α | 64 | 448 | 官方 train |
| 内部 NLL 验证 | 64 | 448 | 官方 train |

固定 seed=42，每个任务使用确定的独立随机排列，按配额划分后再打乱各角色；row_id 含任务名、官方 split、原始行号。
三个角色按行号和问题内容去重互斥；候选训练问题与完整最终评测问题比对，排除重复。实际发现并排除 1 条 ARC-e 重复问题，比较指纹不依赖评测 gold label。
训练/α/内部验证最长实际长度分别为 775/287/526；上限 2048，完整问题与正确续写整段编码，offset 确认监督边界，真实 EOS 参与损失，提示/padding 不参与。仅过滤超长训练候选，不裁掉答案，不过滤最终评测样本。
训练采用正确候选续写的 causal-LM SFT，不增加分类头。WinoGrande 使用正确前缀及公共后缀；该训练配方是我们的公开实现，不宣称为作者原设置。

官方最终评测：BoolQ validation 3270、PIQA validation 1838、HellaSwag validation 10042、WinoGrande XL validation 1267、ARC-e test 2376、ARC-c test 1172、OBQA main test 500，共 20465 条。
评测原始文件未重写，其内容/版本与任务定义摘要保存在 manifest/evaluation results 中。

## 代码职责

新的两阶段阅读入口为 `pretrain_phi2_pruning.py`；七任务控制器训练部分也经由该入口执行。新输出使用 `training/stage1_pruning` 与 `training/stage2_finetuning`，详细阅读顺序和命令见 `phi2-two-stage-entrypoint-20261029.md`。本文后续的旧运行路径描述对应此前已完成实验，未移动其产物。

- phi2_benchmark_tasks.py：固定官方模板/清洗定义与本地 parquet 适配。BoolQ 将 answer:boolean 映射到 label:int；HellaSwag 清洗、WinoGrande 后缀评分直接使用官方实现。
- prepare_phi2_benchmark_data.py：独立 CPU 预处理、分割、去重与摘要；不加载模型或占用 GPU。
- prune_phi2_hesso_cric.py：本框架 CRIC，物理压缩后重载，执行独立 LoRA 恢复。原单模型入口仍可用。
- evaluate_phi2_benchmarks.py：固定官方 lm-eval，按任务保存结果、逐例选择似然/预测和置信区间，可恢复完成到一半的评测。
- run_phi2_seven_task_experiment.py：检查 GPU、资源探测、训练、剪后七任务、LoRA 后七任务和最终报告；数据预处理保持独立。

## 固定配置

phi2_seven_task_config.json：joint 模式，目标逻辑组稀疏率 20%，FP32，无 autocast，关闭 TF32；SGD lr=1e-4，无动量/权重衰减，64 步预热，采样周期64、最多10周期，混合稀疏1024步。最终实际参数减少比例从导出结果读取，不把逻辑组比例称为参数比例。
α 仅在 CRIC 边界更新，独立批量8、微批上限64，按公共策略渐增并记录实际行号/覆盖。
物理压缩后冻结基座，LoRA rank8、alpha16、dropout0.05；独立 AdamW lr=1e-4，4次微批按监督 token 加权累积，512次 optimizer.step、32次 warmup。基座逐参数摘要不变检查继续执行。
该配置是首轮流程配置，不保证恢复到表格精度；不在官方最终评测集上调权重或选模型。

## 评测与基线口径

固定 harness 提交 d6de81643928d653435c431bae19945d41d32520；零样本，不应用 chat template，FP32。完整评测不得传 --limit。CPU 单元验证使用 limit=2，结果明确标为 functional_only，不能当作实验分数。
按任务保存 acc、若定义则保存 acc_norm、逐例输出、数据/模板 SHA、实际样本数、耗时及 Wilson 95% 区间。WinoGrande 不改成答案字符串分类。
同时输出全 acc 平均和“有 acc_norm 则使用 acc_norm”的平均及其映射；作者精确混合口径未核实，不将某一个平均擅自标成作者口径。固定版 SliceGPT 的评测脚本也使用部分任务 acc_norm，但不能据此证明 HESSO-CRIC 完全相同。
paper_phi2_reference.json 保留表中未剪枝 71.09 平均，以及 CRIC 20/25/30 行；均标为 reported_paper_numbers_not_local_measurements。

## 环境与执行

剪枝环境仍为 `/opt/conda/envs/gy_adamcf_rtx6000d`，没有更改它的包。评测 venv `/zeng/GY/.runtime/phi2_eval_20261029` 继承个人环境，在其自身范围安装固定 harness 和所需依赖，安装前后原环境包清单一致。
源码归档 SHA256：c60d18c7dcedb03fdd95d43cd2f0bc4a575862a64eb8c9b390e96f5129e442a4。
用户指定物理 GPU 0；控制器用 UUID 映射到逻辑 cuda:0，不使用 GPU 1，也不停止其他进程。每个阶段开始前确认 GPU0 连续低负载，资源探测选择最长冻结训练样本并叠加最大 α 微批。

运行目录计划：`test/pretrain_phi2/runs/20261029_seven_task_joint_s20_v1/`。profile、training、evaluation_pruned、evaluation_recovered 分别保留日志与结果；顶层 controller_status.json 表示真实状态，最终 experiment_summary.json/experiment_report.md 只在全部阶段成功后生成。
失败保留已完成阶段，不把失败或部分任务写成全量成功。评测恢复会校验模型、协议、源码和参数一致性；拒绝覆盖已有完整结果。

## 验证与证据边界

CPU 回归覆盖原框架 65 项及新增七任务 4 项，共69项；新增测试含七项真实任务定义+本地数据在小型 Phi 上逐任务运行、完整计数、监督 mask、gold 无关指纹、输出保护。GPU完整模型状态另见运行目录，不能由 CPU 测试推断精度。
最初评测日志序列化 torch.dtype 失败已保存，修复后通过；未丢弃该失败证据。
本轮不需要作者精确样本清单，但保留所有配方偏离与来源记录。

## 已完成训练与导出续接

源码 912880f15b777a8523cfa81570c5689f6a9b12d5 已运行完整 Phi-2：1537 步 CRIC 后物理压缩，独立 LoRA 恢复512步。2,779,683,840 参数压至2,443,029,300，实际减少12.1113%；目标20%是逻辑组比例。本次选中65740个 FFN 神经元，没有选中注意力整头/头内维度，不能把本次结果写成注意力剪枝结果。

原流程在 LoRA 合并 FP32 逐元素2e-4校验处停止，adapter 和冻结基座证据已完整保存。三条内部样本单独诊断：FP32 最大绝对差约0.006、argmax一致；相同 adapter 的 FP64 合并差低于1e-11，支持浮点求和顺序差异。此前 failure.json 与原日志保留。

新增严格 FP64 合并审计，单独保存原生 FP32 的最大/平均/相对L2误差、argmax及原门槛结果；不声称严格FP32等价。导出仍使用 PEFT 标准FP32合并，训练/权重不改为FP64。恢复入口 `finetune_phi2_lora.py --finalize-only TRAIN/lora_recovery` 核对基座/adapter/数据摘要和完整训练轨迹，只续接导出，不重训；已有导出时拒绝覆盖。

本次两阶段训练已完成，但在七项全量评测结束前不能称为精度实验完成。续接状态独立记入 continuation_status.json，原控制器失败证据不删除；最终 experiment_summary.json 才表示端到端完成。
更新后的6项训练回归覆盖非零adapter、故意破坏合并必须失败、四种剪枝/三格式、新式与旧式保存断点、拒绝不完整轨迹和重复导出。
