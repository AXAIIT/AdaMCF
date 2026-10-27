# 01 Schedule Comparison

## 当前约定

本目录用于实现 revision 中的 schedule comparison。当前约定如下：

1. 不在 `*_AdaMCF.py` 中硬编码不同 schedule。
2. `*_AdaMCF.py` 保持与原始 AdaMCF / Adaptive Fusion 脚本一致。
3. schedule 通过已有接口或配置项切换，例如 `quota_mode`、`schedule_mode` 或 `pruning_quota_mode`。
4. README 只定义每一种模式的论文来源、实验含义和配额分配公式，供后续实现接口时对齐。

当前 5 个 Python 文件只作为实验入口或日志命名参考：

```text
01_uniform_AdaMCF.py
02_linear_AdaMCF.py
03_cubic_AdaMCF.py
04_exponential_AdaMCF.py
05_cosine_front_AdaMCF.py
```

如果直接运行这些脚本而不通过接口切换 schedule，它们的行为会相同。正式实验必须通过接口设置对应的 `schedule_mode`。

## 固定实验设置

所有 schedule 对比都应使用相同 AdaMCF 主方法设置，只改变 temporal pruning-budget allocation。

| 项目 | 设置 |
|---|---|
| Model | ResNet-110 |
| Dataset | CIFAR-100 |
| Dense checkpoint | `/workspace/OTOv2_v2/test/Ablation/resnet110_cifar100/model/best_model.pth` |
| Method | Full AdaMCF / adaptive multi-criteria fusion |
| Adaptive fusion | Enabled |
| α update | Enabled |
| Target group sparsity | 0.5 |
| Warmup epochs | 5 |
| Pruning epochs | 100 |
| Fine-tuning epochs | 200 |
| Batch size | 128 |
| Optimizer during pruning | HESSO / SGD variant |
| Seed | 42 |

核心控制变量只有一个：

```python
optimizer_prune.active_num_redundant_groups
```

即每个 pruning period 删除多少 dependency groups。

## 论文引用与 schedule 模式对应关系

| 模式 | 建议接口名 | 类型 | 对应引用 | 是否建议纳入主实验 | 说明 |
|---|---|---|---|---|---|
| Uniform | `uniform` | Static baseline | 无特定引用，基础对照 | 是 | 每个 period 分配近似相同预算。 |
| Linear progressive | `linear_progressive` / `linear` | Static progressive | 论文正文把 linear schedules 放在 progressive pruning 讨论中，并引用 [33] | 可选 | 温和后期递增 baseline。 |
| Polynomial / Cubic | `polynomial_cubic` / `cubic` | Static progressive | [33] Zhu & Gupta, gradual pruning / cubic sparsity schedule | 是 | 更贴近 [33] 的经典 gradual pruning schedule。 |
| Exponential | `exponential` | Static progressive | [34] Guo et al., Dynamic Network Surgery，论文正文称为 exponential schedules | 是 | 更激进的非均匀 progressive baseline。 |
| Smooth sparsity evolution | `smooth_evolution` | Motivation only | [18] You et al., Early-Bird Tickets | 不建议单独实现 | 更适合作为 early-stage structure/mask stability 的动机，不是明确 period-level quota baseline。 |
| Loss-adaptive | `loss_adaptive` | Adaptive baseline | 当前表格中列出，但源码无明确实现来源 | 不建议在方案 B 中实现 | 会引入 loss 信号、触发规则和额外超参数。 |
| Cosine-front | `cosine_front` | Static front-loaded, ours | 本文方法 | 是 | 前期/中期释放更多剪枝预算，后期减少结构扰动。 |

引用条目：

```text
[18] H. You et al., Drawing Early-Bird Tickets: Towards More Efficient Training of Deep Networks, 2019.
[31] S. Han et al., Learning both weights and connections for efficient neural network, 2015.
[32] J. Frankle and M. Carbin, The Lottery Ticket Hypothesis, 2018.
[33] M. Zhu and S. Gupta, To prune, or not to prune: exploring the efficacy of pruning for model compression, 2017.
[34] Y. Guo, A. Yao, and Y. Chen, Dynamic Network Surgery for Efficient DNNs, 2016.
```

其中 [31]、[32] 是 one-shot pruning / lottery-ticket 背景引用，不建议作为本实验的 schedule baseline。第一个实验重点比较 [33]、[34] 所代表的 progressive schedule 形式与本文 cosine-front。

## 通用整数配额构造方式

设：

```text
G = total redundant groups to be removed
P = number of pruning periods
p = 0, 1, ..., P-1
τ_p = (p + 1) / P
q_p = quota assigned to period p
```

所有模式都必须满足：

```text
q_p >= 0
sum(q_p for p in range(P)) = G
```

推荐采用 cumulative budget differencing，避免逐期 rounding 造成总预算不一致：

```python
def allocate_by_cumulative(total_groups, periods, Phi):
    quota = []
    prev = 0
    for p in range(periods):
        tau = (p + 1) / periods
        cur = round(total_groups * Phi(tau))
        cur = max(prev, min(total_groups, cur))
        quota.append(cur - prev)
        prev = cur
    quota[-1] += total_groups - sum(quota)
    return quota
```

如果采用 period-wise weights，也必须在最后修正总预算：

```python
def allocate_by_weights(total_groups, weights):
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    weights = weights / weights.sum()
    raw = weights * total_groups
    quota = np.floor(raw).astype(int)
    rest = total_groups - int(quota.sum())
    order = np.argsort(-(raw - quota))
    for i in range(rest):
        quota[order[i % len(quota)]] += 1
    return quota.tolist()
```

为了和论文解释一致，优先推荐 cumulative 形式，因为它直接表示“到第 p 个 period 已完成多少剪枝预算”。

## 各 schedule 的详细定义

### 1. Uniform baseline

建议接口名：

```text
uniform
```

引用关系：无特定论文引用，作为最基础 matched-budget baseline。

累计预算函数：

```text
Φ_uniform(τ) = τ
```

实现：

```python
def Phi_uniform(tau):
    return tau

quota = allocate_by_cumulative(G, P, Phi_uniform)
```

等价简单实现：

```python
quota = [G // P] * P
quota[-1] += G - sum(quota)
```

注意：如果采用简单实现，余数全部加到最后一个 period；如果采用 cumulative rounding，余数会更均匀地分散。正式实验应统一一种实现。

### 2. Linear progressive schedule

建议接口名：

```text
linear_progressive
```

引用关系：论文正文将 linear schedules 作为 progressive pruning 的代表，并引用 [33]。

含义：period-level quota 随训练推进线性增大。前期剪枝少，后期剪枝多。

推荐累计预算函数：

```text
Φ_linear(τ) = τ^2
```

因为其导数为：

```text
dΦ/dτ = 2τ
```

即单期预算随时间近似线性增加。

实现：

```python
def Phi_linear_progressive(tau):
    return tau ** 2

quota = allocate_by_cumulative(G, P, Phi_linear_progressive)
```

如果用 period-wise weights：

```python
x = np.linspace(0, 1, P, endpoint=True)
weights = x + 1e-8
quota = allocate_by_weights(G, weights)
```

注意：`linear_progressive` 和 `uniform` 不应混淆。Uniform 是累计预算线性、period quota 近似恒定；Linear progressive 是 period quota 线性递增、累计预算呈二次曲线。

### 3. Polynomial / Cubic gradual schedule

建议接口名：

```text
polynomial_cubic
```

或：

```text
cubic
```

引用关系：

```text
[33] M. Zhu and S. Gupta, To prune, or not to prune: exploring the efficacy of pruning for model compression, 2017.
```

含义：对应 gradual pruning 中常见的 cubic sparsity evolution。它不是简单的 `weights = τ^3` 后期集中剪枝。更贴近 [33] 的写法是累计稀疏率按三次曲线从初始稀疏率平滑过渡到目标稀疏率。

当初始稀疏率为 0、最终稀疏率为 1 个归一化预算时：

```text
Φ_cubic(τ) = 1 - (1 - τ)^3
```

它的性质是：前期增长较快，后期增长放缓。

实现：

```python
def Phi_polynomial_cubic(tau):
    return 1.0 - (1.0 - tau) ** 3

quota = allocate_by_cumulative(G, P, Phi_polynomial_cubic)
```

如果严格写成 Zhu & Gupta 的目标稀疏率形式：

```text
s_t = s_f + (s_i - s_f) * (1 - (t - t_0) / (n * Δt))^3
```

在本实验中可令：

```text
s_i = 0
s_f = target group sparsity
```

再把每个 period 的目标累计删除组数转成 quota。

### 4. Exponential progressive schedule

建议接口名：

```text
exponential
```

引用关系：

```text
[34] Y. Guo, A. Yao, and Y. Chen, Dynamic Network Surgery for Efficient DNNs, 2016.
```

含义：论文正文把 exponential schedules 归入已有 progressive sparsification 方法。作为 baseline 时，不复现 Dynamic Network Surgery 的完整 connection splicing 机制，只抽取指数式 budget evolution 作为同框架下的 schedule baseline。

推荐累计预算函数：

```text
Φ_exp(τ) = (exp(kτ) - 1) / (exp(k) - 1)
```

建议默认：

```text
k = 4
```

实现：

```python
def Phi_exponential(tau, k=4.0):
    return (np.exp(k * tau) - 1.0) / (np.exp(k) - 1.0)

quota = allocate_by_cumulative(G, P, Phi_exponential)
```

性质：前期预算较少，后期预算快速增加，是 aggressive late-progressive baseline。

### 5. Loss-adaptive schedule

建议接口名：

```text
loss_adaptive
```

引用关系：当前论文表格中列出 Loss-adaptive，但当前源码没有明确可复用的 loss-adaptive quota 实现，也没有和 [33]、[34]、[18] 一一对应的公式。

方案 B 中不建议实现该模式，原因是：

```text
1. 需要设计额外 loss signal，例如 training loss、validation loss 或 proxy loss；
2. 需要设计触发规则和缩放系数；
3. 会引入额外超参数；
4. 对比变量不再只是 schedule curve。
```

如果后续必须实现，只能作为内部 adaptive baseline，而不是声称完整复现某篇引用论文。建议接口先预留，但主实验表格可以不填。

### 6. Cosine-front schedule, ours

建议接口名：

```text
cosine_front
```

引用关系：本文方法，不是外部引用方法。

论文中的累计预算函数：

```text
Φ_cosine_front(τ) = τ + sin(πτ) / π
```

性质：

```text
Φ_cosine_front(0) = 0
Φ_cosine_front(1) = 1
Φ_cosine_front(τ) >= τ, τ ∈ [0, 1]
Φ_cosine_front(0.5) = 0.5 + 1/π ≈ 0.818
```

即训练前半段释放约 81.8% 的 pruning budget。

实现：

```python
def Phi_cosine_front(tau):
    return tau + np.sin(np.pi * tau) / np.pi

quota = allocate_by_cumulative(G, P, Phi_cosine_front)
```

与当前 `hesso.py` 中的 density 形式也可以对应。因为：

```text
dΦ/dτ = 1 + cos(πτ)
```

period-wise weight 可以写成：

```python
x = np.linspace(0, 1, P, endpoint=True)
weights = 1.0 + np.cos(np.pi * x)
quota = allocate_by_weights(G, weights)
```

当前 `hesso.py` 中已有等价写法：

```python
w = 1 - np.cos(np.pi * (1 - x))
```

因为：

```text
1 - cos(π(1 - x)) = 1 + cos(πx)
```

为了和论文公式最一致，推荐接口实现时使用 cumulative 形式：

```text
Φ(τ) = τ + sin(πτ) / π
```

## 推荐主实验组合

如果只做最干净的方案 B，建议保留 4 个模式：

```text
uniform
polynomial_cubic
exponential
cosine_front
```

如果希望覆盖表格中更多代表性曲线，可以保留 5 个模式：

```text
uniform
linear_progressive
polynomial_cubic
exponential
cosine_front
```

不建议把 `loss_adaptive` 放进主实验，除非后续明确给出其引用来源、loss 信号、缩放规则和额外超参数。

## 实验输出要求

每个模式至少保存：

```text
schedule_mode
quota list
schedule_quota.csv
Top-1 before pruning
Top-1 after pruning
Top-1 after fine-tuning
Top-5 after fine-tuning
FLOPs before / after
Params before / after
group sparsity
full log
```

`schedule_quota.csv` 建议格式：

```text
period,quota,cumulative,cumulative_ratio
1,...,...,...
2,...,...,...
...
```

## 论文写法建议

实验说明应写成：

```text
We instantiate representative sparsification schedules inspired by prior work under the same AdaMCF framework and matched pruning budgets. All compared schedules use the same adaptive fusion module, target sparsity, pruning epochs, optimizer configuration, and data augmentation policy; only the temporal allocation of the pruning budget is changed.
```

避免写成：

```text
We reproduce Zhu & Gupta / Dynamic Network Surgery.
```

因为本实验只比较 schedule-level budget allocation，不复现外部论文的完整剪枝方法。
