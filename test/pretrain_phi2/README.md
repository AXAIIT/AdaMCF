# Phi-2 剪枝与 LoRA 微调

生产代码保留七个文件，每个职责只保留一个实现。

| 文件 | 职责 |
| --- | --- |
| `pretrain_phi2_pruning.py` | 总入口：两阶段训练；加 `--benchmarks` 调度空闲卡、资源探测及七项全量评测 |
| `prune_phi2_hesso_cric.py` | 全梯度 CRIC 剪枝、物理导出、`--verify-only` 独立验收 |
| `finetune_phi2_lora.py` | 重载压缩基座、冻结基座、LoRA 微调、保存与合并导出 |
| `prepare_phi2_benchmark_data.py` | 七任务官方 train 的独立预处理与角色划分 |
| `evaluate_phi2_benchmarks.py` | 固定官方 lm-eval 模板和评分，保存逐任务及逐例结果 |
| `phi2_data.py` | 共享冻结数据读取、collate、摘要检查和七任务本地适配 |
| `phi2_precision.py` | 共享 FP32 误差记录、FP64 结构与 LoRA 合并审计 |

从 `pretrain_phi2_pruning.py` 的 `main()` 顺序阅读：配置 → 数据 → CRIC/物理压缩 → 压缩检查 → LoRA → 汇总。
数据预处理与评测各保留一个脚本；α 策略继续由公共优化器管理。

## 输入与配置

- 原始模型：`models/phi-2/`，包含完整权重、config 和 tokenizer。
- 冻结数据：`/adminGroup/datasets_source203/llm_pruning/processed/phi2_seven_task_train_s42_v1`。
- `phi2_seven_task_config.json`：已用于七任务首轮实验的完整配方。
- `phi2_two_stage_smoke_config.json`：短流程验证配置。四种剪枝模式由同一配置中的 pruning_mode 选择，不再保留四份同类配置。
- `evaluation_protocol.json`：七任务数据路径、官方划分、样本数与模板版本；`paper_phi2_reference.json`：文献参考值。
- `corpus_alignment.json`：来源审查历史；当前实验按用户决定使用七任务官方 train，不使用 Alpaca。

只预处理官方 train，冻结互斥 train/alpha/内部 eval；完整官方 validation/test 留给最终评测。预处理脚本拒绝覆盖已有数据。

## 两阶段训练

```bash
cd /zeng/GY/OTOv2_v2/test/pretrain_phi2
P=/opt/conda/envs/gy_adamcf_rtx6000d/bin/python
$P -u pretrain_phi2_pruning.py --config phi2_seven_task_config.json \
  --output runs/my_phi2_run --device cuda:0 --export-format safetensors --log-every 10
```

默认模型与冻结数据路径如上，可用 --model-dir/--data-dir 覆盖。默认配置会执行完整训练；短流程另选 smoke 配置。
阶段一目录 `stage1_pruning/`；阶段二目录 `stage2_finetuning/`。终端及时输出，根目录 run.log 记录步骤，各阶段 run.log/JSONL 保存逐步记录。
失败保存当前阶段，已有输出目录拒绝覆盖。内部响应 NLL 与七任务准确率分别记录。

## 独立导出、验收与微调

```bash
# 复用已保存稀疏检查点，不重训；已有 compressed_model 时拒绝覆盖。
$P prune_phi2_hesso_cric.py --export-only --output runs/EXISTING_STAGE1 --export-format pt

# 同一剪枝入口完成独立重载、结构、输出、缓存生成与反向传播检查。
$P prune_phi2_hesso_cric.py --verify-only --output runs/EXISTING_STAGE1 --device cpu

# 仅做第二阶段；必须指向可训练的压缩结构包。
$P finetune_phi2_lora.py --base-dir runs/EXISTING_STAGE1/compressed_model \
  --data-dir /adminGroup/datasets_source203/llm_pruning/processed/phi2_seven_task_train_s42_v1 \
  --config phi2_seven_task_config.json --output runs/my_recovery --device cuda:0

# LoRA 已训练完成但导出失败时，只续做合并与导出。
$P finetune_phi2_lora.py --finalize-only runs/EXISTING_RECOVERY --export-format safetensors
```

pt/safetensors/onnx 均可由总入口选择。ONNX 时阶段一保存可训练的 safetensors，阶段二导出 ONNX；ONNX 不用于梯度微调或 KV cache 生成。
物理压缩由框架 construct_subnet 处理，异构结构由 OTO.load_subnet 重载。LoRA 只在压缩之后附加，基座逐参数摘要检查与合并审计继续执行。

## 自动七任务实验

```bash
# SOURCE_SHA 为本次已提交源码的40位提交号。
$P -u pretrain_phi2_pruning.py --benchmarks --source-commit "$SOURCE_SHA" \
  --config phi2_seven_task_config.json --physical-gpu 0 --output runs/my_seven_task
```

该模式复用本文件的普通训练入口，在每个 GPU 阶段前确认指定物理卡空闲，并实时转发子进程输出。
完整模式的训练结果位于 `training/stage1_pruning`、`training/stage2_finetuning`；两套模型各跑七项完整评测，逐例结果保存在 evaluation_pruned/evaluation_recovered。
七任务 GPU 评分目前需要 pt/safetensors；ONNX 可在普通两阶段模式训练并导出。文献表格仅作外部参考。

## 回归测试

测试统一在仓库 `test/phi2/`，生产目录不再混放测试文件。合成数据夹具也只在测试目录内，不再保留旧 Alpaca 预处理入口。

```bash
cd /zeng/GY/OTOv2_v2
CUDA_VISIBLE_DEVICES='' PHI2_EVAL_PYTHON=/zeng/GY/.runtime/phi2_eval_20261029/bin/python \
  /opt/conda/envs/gy_adamcf_rtx6000d/bin/python -m unittest discover -s test/phi2 -p 'test_phi2_*.py' -v
```

旧脚本及配置的历史版本已保留在 Git 检查点；模型权重、冻结数据、已有 runs/ 和日志不参与此次清理。
