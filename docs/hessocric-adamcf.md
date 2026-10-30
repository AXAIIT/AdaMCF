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

`oto` 应已使用正确的模型和输入构图，参数组的剪枝合同须先验证。
2026-10-30 起，CRIC 默认使用以下与模型无关的样本百分比策略。

```python
optimizer = oto.hessocric(
    variant='adamw', lr=2e-5, target_group_sparsity=0.2,
    start_cric_step=100, max_cycle_period=10,
    sampling_steps=50, hybrid_training_steps=1000,
    alpha_scheduler_cfg={'warmup_updates': 0, 'pool_ratio': 0.1, 'update_ratio': 0.5},
)

# 显式池优先；传 None 时从训练数据构建回退池。
# groups 是调用方提供的任务/类别标签；单任务可以不提供。
gradient_data, alpha_pool, pool_info = optimizer.prepare_alpha_data(
    training_dataset,
    alpha_dataset=independent_alpha_dataset,  # 没有独立 α 数据时传 None
    batch_size=8,
    collate_fn=alpha_collate,                 # 返回 AlphaBatch 或 (inputs, targets)
    train_groups=training_group_labels,       # 可选，长度必须与 dataset 一致
    alpha_groups=alpha_group_labels,          # 可选，显式 α dataset 的分层
    seed=43,
    holdout=True,                            # 回退池从梯度训练数据中移除
)
train_loader = DataLoader(gradient_data, batch_size=train_batch_size, collate_fn=train_collate)

for batch in train_loader:
    # batch 为已经搬到模型设备的字典，包含 input_ids、labels 等。
    optimizer.zero_grad()
    loss = model(**batch).loss
    loss.backward()
    if optimizer.will_update_alpha_on_next_step():
        optimizer.alpha_scheduler.update(model, None)  # 使用已绑定的 alpha_pool
    optimizer.step(loss=loss.detach())
```

`alpha_collate` 对字典模型可返回 `AlphaBatch(inputs_dict, sample_ids=...)`；普通分类模型
返回 `(inputs, targets)`，调用 update 时传入分类 criterion。None criterion 使用模型输出 loss。
不同模型仅提供数据、collate 和分层元信息，取样决策不写在模型训练脚本内。

也可直接导入 `prepare_cric_alpha_data` 在创建优化器前准备数据；返回相同三元组。
`holdout=False` 显式选择共享训练样本模式，不将该池称为独立验证数据。
显式 α 数据不再按 pool_ratio 缩减，调用方负责保证与最终测试数据隔离。

### 百分比、取整和兼容

- 默认 `sampling_policy='ratio'`、`pool_ratio=0.1`、`update_ratio=0.5`、
  `align_power_of_two=True`。按每个分层计算百分比，先向上取整，再按绝对距离
  就近对齐到 2 的次幂（等距向上）；不能超出可用样本。对齐后实际比例可能变化。
- `min_pool_samples/max_pool_samples/min_update_samples/max_update_samples` 是**每层**
  样本上下限，默认最小 1、最大 None，不用隐含上限覆盖用户比例；可显式限制大数据成本。
  独立划出模式每层至少保留一条梯度训练样本，小到无法同时分配的层明确报错。
- 无分层标签自然退化为单任务；不同任务数据量不等时按各层自己的比例抽取，
  不自动改成任务等权目标。七任务各 64 条只是示例，不是框架约束。
- 单次评估内无重复，跨边界先遍历未见样本、耗尽后重新打乱；所有正负差分共享同一批数据。
  局部 RNG 不扰动训练 RNG。失败会恢复池游标/RNG，成功后才提交覆盖记录。
- GPU `batch_size` 与样本总量独立；ratio 模式忽略旧 `eval_batch_limit` 的渐增截断。
  旧批次缓存仍可使用，ratio 此时只能按**批次数**比例回退，并在统计中明确单位，
  无法承诺精确独立样本比例。新代码应使用 `CRICAlphaPool`。
- `sampling_policy='legacy'` 保留既有渐增微批规则，配合旧缓存/AlphaEvaluationPool；
  显式传入旧 AlphaScheduleConfig 对象、恢复不含新字段的旧优化器检查点也采用 legacy。
  HESSO 的配置、数值内核、旧取批入口不变。CRIC 的更新时间、置信门限、
  损失汇总、历史最终重融合也不在本次改动中调整。

例：七层各 64 条的显式池，每次 50% 是各 32 条，总计 224；微批 8 时共 28 批。
两个实际成功评估覆盖完整 448 条；排序间隔门限可能跳过边界，日志需要区别
“可更新边界”“执行差分”和“α 数值实际改变”。不保证增加数据就提高最终精度。

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

绑定 CRICAlphaPool 后，检查点还保存池内排列、游标、局部 RNG、已覆盖样本和
百分比配置。恢复时以相同冻结 dataset、索引及微批大小重新 prepare_alpha_data；
待恢复的池状态会在绑定/下一次评估前加载。框架校验索引分层和微批大小，
数据内容摘要、训练 loader 的顺序/RNG 仍由训练器一并冻结，不能只保存模型权重。

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
