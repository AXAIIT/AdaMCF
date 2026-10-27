# 02 Zeroth-Order Update Sensitivity

## 实验目的

本实验用于回应审稿意见中关于 **zeroth-order fusion-weight update** 稳定性与敏感性的质疑。重点分析 AdaMCF 的自适应融合权重 `alpha` 更新机制对以下因素的敏感性：

1. finite-difference perturbation radius，即扰动尺度 `eps`；
2. alpha update interval，即更新频率 `update_stride`；
3. total zeroth-order proxy-loss evaluation budget，即整个训练过程中的总函数评估预算；
4. training noise，即 mini-batch noise 对有限差分估计的影响。

本实验不改变剪枝准则、不改变 cosine-front schedule、不改变目标稀疏率、不关闭 AdaMCF。所有设置均使用论文主方法：

```text
Full AdaMCF + cosine-front schedule + matched pruning budget
```

唯一变化变量是：

```python
alpha_scheduler_cfg = {
    "eps": ...,
    "update_stride": ...,
}
```

并通过：

```python
oto.hesso(..., alpha_scheduler_cfg=alpha_scheduler_cfg)
```

传入 HESSO / AdaMCF optimizer。

---

## 实验设计概述

原始方案只包含 5 个代表性设置。为了让 sensitivity analysis 更完整、更直观，修订后采用规则网格实验：

```text
eps ∈ {0.01, 0.02, 0.03, 0.04, 0.05}
update_stride ∈ {1, 2, 4}
```

因此共有：

```text
5 × 3 = 15 组实验
```

其中默认配置为：

```text
eps = 0.03
update_stride = 1
```

该默认点可复用论文当前 AdaMCF Full 的 ResNet-110 / CIFAR-100 结果；如果为了日志完整性和实验环境一致，也可以重新运行一次。

---

## 关于 function evaluations 的说明

AdaMCF 采用 coordinate-wise two-point centered finite-difference estimator 来更新融合权重 `alpha`。对于第 `k` 个 criterion 维度，梯度估计为：

```text
g_hat_k = [L_s(alpha + eps * e_k) - L_s(alpha - eps * e_k)] / (2 * eps)
```

因此每个 criterion 维度需要 2 次 proxy-loss evaluation。默认共有 `M=5` 个 pruning criteria，因此每次 alpha update 的函数评估次数为：

```text
Eval. / Update = 2M = 10
```

本实验保持 `Eval. / Update = 2M` 不变，避免引入 one-sided estimator、random-direction estimator 或 coordinate-subset estimator 带来的 estimator bias / variance 混杂。函数评估预算通过 `update_stride` 控制：

| update_stride | Alpha update frequency | Eval. / Update | Approx. total eval. budget |
|---:|---|---:|---:|
| 1 | every period | 2M | 1.00× |
| 2 | every 2 periods | 2M | 0.50× |
| 4 | every 4 periods | 2M | 0.25× |

因此，`update_stride` 同时反映 alpha 更新频率和整个训练过程中的总 zeroth-order proxy-loss evaluation budget。

---

## 固定设置

| 项目 | 设置 |
|---|---|
| Model | ResNet-110 |
| Dataset | CIFAR-100 |
| Method | Full AdaMCF |
| Schedule | cosine-front |
| Target group sparsity | 0.5 |
| Warmup epochs | 5 |
| Pruning epochs | 100 |
| Fine-tuning epochs | 200 / 按当前脚本设置为准 |
| Seed | 42 |
| Adaptive fusion | Enabled |
| Alpha update | Enabled |
| Eval. / Update | 2M = 10 |
| 变化变量 | `eps`, `update_stride` |

---

## 建议文件结构

建议将当前 5 个脚本扩展为 15 个脚本。命名规则如下：

```text
{index}_eps_{eps_tag}_update_every_{stride}_periods_AdaMCF.py
```

其中：

```text
eps=0.01 -> eps_001
eps=0.02 -> eps_002
eps=0.03 -> eps_003
eps=0.04 -> eps_004
eps=0.05 -> eps_005
update_stride=1 -> update_every_period
update_stride=2 -> update_every_2_periods
update_stride=4 -> update_every_4_periods
```

建议 15 个脚本如下：

```text
01_eps_001_update_every_period_AdaMCF.py
02_eps_002_update_every_period_AdaMCF.py
03_eps_003_update_every_period_AdaMCF.py
04_eps_004_update_every_period_AdaMCF.py
05_eps_005_update_every_period_AdaMCF.py

06_eps_001_update_every_2_periods_AdaMCF.py
07_eps_002_update_every_2_periods_AdaMCF.py
08_eps_003_update_every_2_periods_AdaMCF.py
09_eps_004_update_every_2_periods_AdaMCF.py
10_eps_005_update_every_2_periods_AdaMCF.py

11_eps_001_update_every_4_periods_AdaMCF.py
12_eps_002_update_every_4_periods_AdaMCF.py
13_eps_003_update_every_4_periods_AdaMCF.py
14_eps_004_update_every_4_periods_AdaMCF.py
15_eps_005_update_every_4_periods_AdaMCF.py
```

如果不新增脚本，也可以只保留一个模板脚本，通过命令行参数传入 `eps` 和 `update_stride`。但为了复现审稿实验，15 个固定脚本更直观，也更便于保存日志。

---

## 每个脚本需要统一修改的位置

### 1. 增加 `alpha_scheduler_cfg`

建议放在创建 HESSO optimizer 前，例如 `pruning_steps` 计算之后：

```python
alpha_scheduler_cfg = {
    "eps": 0.03,
    "update_stride": 1,
}
```

不同脚本只修改 `eps` 和 `update_stride`。

### 2. 传入 `oto.hesso(...)`

在 HESSO optimizer 创建处加入：

```python
optimizer_prune = oto.hesso(
    variant='sgd',
    lr=lr,
    first_momentum=0.9,
    second_momentum=0.0,
    dampening=0.0,
    weight_decay=5e-4,
    target_group_sparsity=target_sparsity,
    start_pruning_step=start_pruning_step,
    pruning_periods=pruning_epochs,
    pruning_steps=pruning_steps,
    device=device,
    alpha_scheduler_cfg=alpha_scheduler_cfg,
)
```

### 3. 修改输出缓存目录，避免互相覆盖

每个实验必须使用独立目录。例如：

```python
exp_name = "01_eps_001_update_every_period_AdaMCF"

output_cache_dir = os.path.join(
    base_dir,
    f"cache_resnet110_cifar100_pruned_{exp_name}_0.5"
)

pruned_cache_dir = base_dir / f"cache_resnet110_cifar100_pruned_{exp_name}_0.5"
```

### 4. 修改微调输出目录，避免互相覆盖

建议每个实验使用独立微调目录和模型文件名：

```python
finetuned_model_dir = os.path.join(base_dir, f"finetuned_{exp_name}_0_5")
finetuned_model_path = os.path.join(
    finetuned_model_dir,
    f"resnet110_cifar100_finetuned_{exp_name}.pt"
)
```

### 5. 打印当前 ZO 配置

为方便日志核查，建议在剪枝开始前打印：

```python
print(
    f"[ZO Config] eps={alpha_scheduler_cfg['eps']}, "
    f"update_stride={alpha_scheduler_cfg['update_stride']}, "
    f"eval_per_update={2 * 5}"
)
```

如果代码中 `M` 可直接读取，则改为：

```python
print(f"[ZO Config] eps={alpha_scheduler_cfg['eps']}, update_stride={alpha_scheduler_cfg['update_stride']}, eval_per_update={2 * M}")
```

---

## 15 组实验配置表

| ID | Script | eps | update_stride | Alpha update frequency | Eval. / Update | Approx. total eval. budget | 备注 |
|---:|---|---:|---:|---|---:|---:|---|
| 01 | `01_eps_001_update_every_period_AdaMCF.py` | 0.01 | 1 | every period | 2M | 1.00× | small perturbation |
| 02 | `02_eps_002_update_every_period_AdaMCF.py` | 0.02 | 1 | every period | 2M | 1.00× | small-to-default perturbation |
| 03 | `03_eps_003_update_every_period_AdaMCF.py` | 0.03 | 1 | every period | 2M | 1.00× | default |
| 04 | `04_eps_004_update_every_period_AdaMCF.py` | 0.04 | 1 | every period | 2M | 1.00× | default-to-large perturbation |
| 05 | `05_eps_005_update_every_period_AdaMCF.py` | 0.05 | 1 | every period | 2M | 1.00× | large perturbation |
| 06 | `06_eps_001_update_every_2_periods_AdaMCF.py` | 0.01 | 2 | every 2 periods | 2M | 0.50× | lower update budget |
| 07 | `07_eps_002_update_every_2_periods_AdaMCF.py` | 0.02 | 2 | every 2 periods | 2M | 0.50× | lower update budget |
| 08 | `08_eps_003_update_every_2_periods_AdaMCF.py` | 0.03 | 2 | every 2 periods | 2M | 0.50× | default eps, lower update budget |
| 09 | `09_eps_004_update_every_2_periods_AdaMCF.py` | 0.04 | 2 | every 2 periods | 2M | 0.50× | lower update budget |
| 10 | `10_eps_005_update_every_2_periods_AdaMCF.py` | 0.05 | 2 | every 2 periods | 2M | 0.50× | lower update budget |
| 11 | `11_eps_001_update_every_4_periods_AdaMCF.py` | 0.01 | 4 | every 4 periods | 2M | 0.25× | sparse update budget |
| 12 | `12_eps_002_update_every_4_periods_AdaMCF.py` | 0.02 | 4 | every 4 periods | 2M | 0.25× | sparse update budget |
| 13 | `13_eps_003_update_every_4_periods_AdaMCF.py` | 0.03 | 4 | every 4 periods | 2M | 0.25× | default eps, sparse update budget |
| 14 | `14_eps_004_update_every_4_periods_AdaMCF.py` | 0.04 | 4 | every 4 periods | 2M | 0.25× | sparse update budget |
| 15 | `15_eps_005_update_every_4_periods_AdaMCF.py` | 0.05 | 4 | every 4 periods | 2M | 0.25× | sparse update budget |

---

## 结果记录表

| ID | eps | update_stride | Total eval. budget | Top-1 | Top-5 | FLOPs Red. | Params Red. | Update Count | Training Time | Comment |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 01 | 0.01 | 1 | 1.00× |  |  |  |  |  |  |  |
| 02 | 0.02 | 1 | 1.00× |  |  |  |  |  |  |  |
| 03 | 0.03 | 1 | 1.00× | 71.53 / 复用或重跑 |  | 66.02 | 42.90 |  |  | default |
| 04 | 0.04 | 1 | 1.00× |  |  |  |  |  |  |  |
| 05 | 0.05 | 1 | 1.00× |  |  |  |  |  |  |  |
| 06 | 0.01 | 2 | 0.50× |  |  |  |  |  |  |  |
| 07 | 0.02 | 2 | 0.50× |  |  |  |  |  |  |  |
| 08 | 0.03 | 2 | 0.50× |  |  |  |  |  |  |  |
| 09 | 0.04 | 2 | 0.50× |  |  |  |  |  |  |  |
| 10 | 0.05 | 2 | 0.50× |  |  |  |  |  |  |  |
| 11 | 0.01 | 4 | 0.25× |  |  |  |  |  |  |  |
| 12 | 0.02 | 4 | 0.25× |  |  |  |  |  |  |  |
| 13 | 0.03 | 4 | 0.25× |  |  |  |  |  |  |  |
| 14 | 0.04 | 4 | 0.25× |  |  |  |  |  |  |  |
| 15 | 0.05 | 4 | 0.25× |  |  |  |  |  |  |  |

---

## 建议可视化方式

建议将 15 组实验画成离散三维柱状图，而不是三维曲面图。原因是本实验变量是离散网格，不是连续函数。

推荐图设置：

```text
x-axis: perturbation radius eps
 y-axis: alpha update interval / update_stride
 z-axis: Top-1 accuracy 或 Top-1 drop from default
```

更推荐使用：

```text
z-axis: Top-1 drop from default
```

原因是 Top-1 accuracy 的数值范围通常较窄，直接从 0 开始画柱状图不容易看出差异；用相对默认设置的 accuracy drop 能更清楚地展示 sensitivity。

图注建议写成：

```text
Sensitivity visualization of zeroth-order adaptive fusion on ResNet-110 with CIFAR-100. The perturbation radius eps and the update interval of alpha are varied under the same pruning budget and training protocol. Each update uses 2M proxy-loss evaluations, while the total evaluation budget decreases as the update interval increases. The bar height denotes the Top-1 accuracy drop relative to the default setting.
```

---

## 论文正文写法建议

可以写成：

```text
We evaluate the sensitivity of AdaMCF to the zeroth-order update hyperparameters by varying the finite-difference perturbation radius and the update interval of alpha under the same pruning schedule, target sparsity, adaptive fusion module, optimizer, and training protocol. Each alpha update uses a coordinate-wise two-point centered finite-difference estimator with 2M proxy-loss evaluations. Since M=5 in the default setting, this corresponds to 10 proxy-loss evaluations per update. The per-update estimator is kept fixed to avoid confounding the analysis with estimator bias, while the total zeroth-order evaluation budget is controlled by the update interval.
```

如果结果差异较小，可以写：

```text
The sensitivity landscape shows that AdaMCF remains stable around the default configuration. Moderate changes of eps and update interval cause only small accuracy variations, indicating that the adaptive fusion mechanism is not overly sensitive to these hyperparameters.
```

如果 `update_stride=4` 明显下降，可以写：

```text
Very sparse alpha updates can degrade performance, suggesting that sufficiently frequent fusion-weight updates are important for tracking the evolving pruning dynamics.
```

---

## 运行后需要保存的材料

每组实验建议保存：

1. 最终 Top-1 / Top-5；
2. FLOPs reduction；
3. Params reduction；
4. alpha update count；
5. alpha trajectory；
6. pruning log；
7. finetuning log；
8. training time；
9. final checkpoint；
10. 对应脚本版本。

这些材料可用于：

1. Table X 或 Appendix sensitivity table；
2. 三维 sensitivity visualization；
3. Response letter 中说明 perturbation scale、update frequency、function evaluations 和 training noise 的处理。