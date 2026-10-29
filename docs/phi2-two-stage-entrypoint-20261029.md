# Phi-2 两阶段总入口

入口：`/zeng/GY/OTOv2_v2/test/pretrain_phi2/pretrain_phi2_pruning.py`。

## 阅读顺序

参照 `test/Bert/pretrain_bert_base_uncased_pruning.py` 的组织顺序，先看新入口的 `main()`：

|顺序|代码职责|Phi-2 的具体行为|
|---|---|---|
|1|配置、设备与路径|顶层配置管理 CRIC，recovery 管理 LoRA；不修改源 JSON|
|2|prepare_phi2_dataset|读取冻结 JSONL，核对 train/alpha/eval、摘要与分词器|
|3|prune_model|原生 FP32 全梯度 CRIC，训练循环复用 train_cric，框架 construct_subnet 物理压缩|
|4|压缩检查|确认结构审计成功，报告实际参数变化，释放阶段一资源|
|5|finetune_model|从磁盘 OTO.load_subnet，冻结压缩基座，附加 A/B，新 AdamW 独立训练|
|6|summarize_experiment|输出剪前/剪后/微调后内部 NLL、实际结构删除和模型保存路径|

数据预处理继续独立执行；α 抽样与更新仍在公共优化器层。新入口只编排两个已验证阶段，不复制训练循环，不修改优化器策略。
与 BERT 的任务差异保留：Phi 是 causal-LM 续写损失，不使用问答 span 标签；第二阶段只更新 LoRA A/B，不全量更新基座。

## 运行

默认配置为已经完成首轮实验的 `phi2_seven_task_config.json`，会执行完整训练。调整配置时将 JSON 另存为新文件，再用 `--config` 指定。

```bash
cd /zeng/GY/OTOv2_v2/test/pretrain_phi2
P=/opt/conda/envs/gy_adamcf_rtx6000d/bin/python
$P -u pretrain_phi2_pruning.py \
  --config phi2_seven_task_config.json \
  --model-dir models/phi-2 \
  --data-dir /adminGroup/datasets_source203/llm_pruning/processed/phi2_seven_task_train_s42_v1 \
  --output runs/my_phi2_two_stage \
  --device cuda:0 --export-format safetensors --log-every 10
```

`--export-format` 可选 pt、safetensors、onnx；ONNX 时阶段一保留可训练 safetensors，阶段二最终模型导出 ONNX。`--log-every 1` 可逐步输出；未指定则保留配置中各阶段的日志间隔。
GPU 选择遵守 `CUDA_VISIBLE_DEVICES` 与 `--device`；需要自动等空闲卡并运行七项全量评测时，使用原 `run_phi2_seven_task_experiment.py`，它已改为调用新总入口训练。

## 输出与日志

```text
runs/my_phi2_two_stage/
├── config.json                    # 生效的完整配置
├── pruning_config.json            # 阶段一独立配置，明确关闭恢复阶段
├── inputs.json                    # 源配置/数据/入口源码摘要与设备
├── run.log                        # 按六个步骤输出的人类可读总日志
├── status.json                     # 当前阶段/完成/失败
├── workflow_result.json           # 仅在两阶段和汇总成功后写入
├── summary.md                     # 内部评估、压缩率、产物路径
├── stage1_pruning/
│   ├── run.log / events.jsonl / training_trace.jsonl
│   ├── pruning_result.json
│   ├── sparse_model/
│   └── compressed_model/          # 阶段二实际重载的异构结构包
└── stage2_finetuning/
    ├── run.log / events.jsonl / finetuning_trace.jsonl
    ├── recovery_result.json / merge_precision_audit.json
    ├── adapter/                   # 仅 A/B，绑定阶段一基座摘要
    └── recovered_model/           # 合并后的最终模型
```

终端及时输出步骤、训练 loss、CRIC 周期与 α、学习率、内部验证和保存进度。总日志记录主流程；每步训练详情保存在各阶段日志和 JSONL。新入口拒绝覆盖已有目录；失败记录当前阶段，并保留已完成剪枝结果。

内部响应 NLL 不是七项准确率。七任务自动控制器会分别评测 `training/stage1_pruning/compressed_model` 和 `training/stage2_finetuning/recovered_model`，并从新位置读取剪枝结果。此前完成的实验目录与旧入口继续保留。

## 验证范围

使用真实小型 Phi 覆盖 joint/num_head/head_dim/ffn 和 pt/safetensors/onnx；相同输入、种子与配方下，比对旧入口与新入口最终模型的全部 state_dict 张量及 α。检查阶段一不会再次启动 LoRA、配置不被改写、失败状态/已有目录保护，并验证七任务控制器的脚本与模型路径。
该改动是主流程组织与共享参数检查，不改变训练算法或此前实验权重；本次不重新运行完整 Phi-2 GPU 训练。
