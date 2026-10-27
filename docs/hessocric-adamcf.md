# AdaMCF 的 HESSO-CRIC 接入说明

本版将现有 HESSO 的五准则和损失引导自适应 α 接入 HESSO-CRIC。保留 CRIC 的投影、全局采样、终止判断、冗余组确定和混合稀疏化顺序，不引入 HESSO 的余弦预算调度。基线为 `pre-hessocric-20261027`（`5ba91b0eb391fa2b30e980fb44a94e9ae8034504`）。

## 方法与实现

- 默认准则为 `l1_avg_magnitude`、`l2_avg_magnitude`、`cosine_similarity`、`fisher`、`grad_weight_product`，初始 α 各为 0.2。评分及全局 L2 归一化直接复用现有 HESSO 方法。
- α 评估发生在 CRIC 开始及全局采样周期边界；周期内冻结。`warmup_updates` 可另外启用 CRIC 开始前的 α 预热，示例中关闭。
- α 使用现有调度器的中心差分、梯度 EMA、动量、单纯形归一化和 L1 单步幅度约束。排序间隔门限可能跳过某次更新，因此“边界可评估”不等于“必然更新”。
- 每周期只累计各准则**未乘 α 的归一化评分之和与样本数**，存放在 CPU。终止时先求全历史均值，再统一乘最终 α；不平均已经乘过旧 α 的 `overall`。
- 原 CRIC 的 loss 校正单独保存、单独加入最终分数，权重默认 1.0，不参与 α 的单纯形归一化。
- 临时剪枝评估后恢复权重、各子模块训练状态、模型 buffers 和随机数状态；预览梯度变体不推进 Adam/SGD 动量，真实训练步只更新一次动量。
- CRIC 的试投影、采样、稀疏化及最终固定为零统一使用逻辑组到物理索引的映射，支持 BASIC、TRANSPOSE、多头以及 INDEX_GROUPS，保留 NO_PRUNE/NO_UPDATE；辅助节点使用逻辑组索引加 offset。

最终历史评分为：

```text
score(g) = Σ_j α_final[j] × (Σ_t normalized_raw[j,g,t] / N)
           + 原 CRIC 的 loss 校正项
```

改动分散在独立 CRIC 兼容层中：`hesso_cric.py` 保留原阶段实现，通过 `AdaptiveCRICMixin` 覆盖所需行为；`cric_adapter.py` 实现评分历史、索引和检查点；`cric_alpha.py` 实现周期边界 α 更新。公共 HESSO、原 α 调度器、公共基类、图构建与模型算子没有修改。

## 调用方式

`oto` 应已使用正确的模型和输入构图，参数组的剪枝合同须先验证。下面展示 Hugging Face 风格的训练循环；`alpha_batches` 是固定评估微批列表，所有 Tensor 已放在模型设备上，每项为 `(inputs_dict, None)`，字典内包含 `labels`。

```python
optimizer = oto.hessocric(
    variant='adamw', lr=2e-5, target_group_sparsity=0.2,
    start_cric_step=100, max_cycle_period=10,
    sampling_steps=50, hybrid_training_steps=1000,
    alpha_scheduler_cfg={'warmup_updates': 0},
)

for batch in train_loader:
    # batch 为已经搬到模型设备的字典，包含 input_ids、labels 等。
    optimizer.zero_grad()
    loss = model(**batch).loss
    loss.backward()
    if optimizer.will_update_alpha_on_next_step():
        optimizer.alpha_scheduler.update(model, None, alpha_batches)
    optimizer.step(loss=loss.detach())
```

也可直接从 `only_train_once.optimizer` 导入 `HESSOCRIC`，传入现有框架生成的参数组。普通分类模型传 `criterion` 和 `[(inputs, targets), ...]`；单个微批可使用 `update(model, criterion, inputs, targets)`。多输入模型使用字典或把多输入 tuple 放在微批的 `inputs` 位置。

`sampling_steps`、`hybrid_training_steps` 必须显式设置为正整数；`start_cric_step=0` 解释为第一步开始。CRIC 投影阶段必须向 `step` 传入有限标量 loss；建议整个训练过程统一传入。当前只接受参数位于同一设备的配置。

CRIC 的投影检查长度沿用原实现，为 `6 × 可剪枝非辅助参数组数`。`max_cycle_period` 的计数方式及终止判断也沿用原实现，不能将其直接理解为 HESSO 的预算周期数。最终分组还受原有 `group_divisible` 约束，实际稀疏度需从结果核验。

`adaptive_alpha=False` 关闭 α 更新，保留固定加权五准则与 CRIC 流程。它不是官方旧评分基线；如需旧准则，可显式传入 `importance_score_criteria`。旧别名 `magnitude`、`avg_magnitude` 分别映射到 L2 magnitude 和 L2 average magnitude。

## 检查点

```python
# 只加载自己信任的检查点；训练器另行保存 RNG、数据顺序、epoch 和其他调度器状态。
torch.save({'model': model.state_dict(),
            'optimizer': optimizer.state_dict()}, checkpoint_path)

saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(saved['model'])
optimizer.load_state_dict(saved['optimizer'])
```

优化器检查点包含 CRIC 阶段、参考权重、违反组集合、未加权历史评分、loss 历史、α/动量/EMA 和基础优化器的一二阶动量。恢复后历史评分强制回到 CPU float32。缺少完整 CRIC 状态或评分项/参数组布局不一致的检查点明确拒绝作为续训检查点加载；模型权重仍可单独加载。

## 验证与边界

2026-10-27 在 Python 3.11.11、PyTorch 2.7.1+cu128、RTX 6000D 环境验证：

| 检查 | 结果及范围 |
| --- | --- |
| 新增 CRIC 测试 | CPU 18/18，CUDA 18/18 |
| 四阶段断点续训 | 第 2/8/10/17 步保存恢复，模型、α、步数及剪枝集合与连续训练逐值一致 |
| 自适应 α | 在受控排序边界响应评估损失；同一边界只更新一次，周期内不更新 |
| 小型随机 Phi | 原生 PhiForCausalLM，2 层、hidden 32、vocab 128，手工正确参数组；36 步 CRIC+α，剪除 34 个逻辑组，loss/logits 有限 |
| 既有 VGG/HESSO | 512/128 样本，2 轮剪枝与 1 轮微调，物理压缩、保存重载通过 |
| 既有 YOLO11 Attention | num_head/head_dim 两种合同，前后向及保存重载通过 |
| 既有拼接合同 | 分段 concat 物理剪枝、保存重载及训练一步通过 |

小型 Phi 测试验证的是优化器与原生 causal loss 的连接，参数组为测试中手工建立。它未验证预训练 Phi-2 2.7B 的自动构图、物理结构导出及真实语料精度，也没有改动 LoRA 优化器。VGG 的短流程用于执行回归，剪后准确率低，不能作为精度恢复或论文指标。

在仓库根目录运行新增测试：

```bash
PYTHONPATH="$PWD" PYTHONNOUSERSITE=1 \
  /opt/conda/envs/gy_adamcf_rtx6000d/bin/python test/hesso_cric/test_adamcf_cric.py
CUDA_VISIBLE_DEVICES=0 CRIC_TEST_DEVICE=cuda:0 PYTHONPATH="$PWD" \
  PYTHONNOUSERSITE=1 /opt/conda/envs/gy_adamcf_rtx6000d/bin/python \
  test/hesso_cric/test_adamcf_cric.py
```

完整日志与清单位于部署侧 `/zeng/GY/hessocric_adaptation_20261027/`，不随代码提交权重、数据或运行产物。
