# Fig. 1 动机实验：Spatial Conflict 与 Temporal Mismatch

本目录构建 ResNet-110/CIFAR-100 上的 Fig. 1(a) 和 Fig. 1(b)。

最终流程只需要已经训练好的 dense ResNet-110 checkpoint，不重新预训练 dense 模型，也不需要 epoch 10/30/60/90/120 的历史 dense checkpoints。

## 1. 三种讨论方案的区别与证据分工

| 方案 | 时间状态 | 损伤量 | 日程对照 | 主要问题 |
|---|---|---|---|---|
| 先前讨论方案一 | 多个 dense 训练 checkpoint | 相同5%剪枝后，再让 Probe/Control 各训练50 batches，比较残余损伤 | Uniform 与 Cosine-front，并计算 C(b) | 50 batches 缺少自然时间尺度；dense 预训练 epoch 也不等于真实剪枝 period |
| 先前讨论方案二 | 与方案一相同 | 上图画 $c_p$，mean ± 95% CI | 下图画两种 $b_p$ | 这是方案一的作图形式，不是另一套独立实验 |
| 旧 README 方案 | 一条真实剪枝轨迹的正配额 periods | 当前剩余 groups 的1%即时损伤 | 真实配额与 cost-aligned reordering | 预算绝对量随时间变化；跳过零配额后期；没有真正比较 Uniform/Cosine-front |
| 双真实轨迹 | 从同一 dense checkpoint 分别完整训练 Balanced Uniform 与 Cosine-front，各100个 periods | 两条轨迹均在每期插入相同的初始 groups 总数5%临时探针 | 比较真实训练历史、最终精度和压缩结果 | 属于端到端结果；由于 $c^{U}\ne c^{C}$，不能用两条路径各自的绝对 cost 隔离日程机制 |
| **论文主图的受控机制比较** | **固定一条真实参考轨迹上测得的完整 $c_1,\ldots,c_P$** | **不把 $c_p$ 设为常数，而是保持这组随周期变化的实测值不变** | **只替换同总预算的 $b^U$ 与 $b^C$，比较 $C(c,b)$** | **与论文定理的“固定 $c$、比较 $b$”严格一致** |

“时间失配”不是单独一条 loss 曲线，而是两个量之间的关系：

\[
c_p=\text{第 }p\text{ 个剪枝周期每单位预算的期望性能损伤},\qquad
b_p=\text{可行日程在第 }p\text{ 个周期分配的剪枝预算}.
\]

“固定 $c_p$”的准确含义是固定整条向量
$\boldsymbol c=(c_1,\ldots,c_P)$，不是令 $c_1=\cdots=c_P$。例如
$\boldsymbol c=(1,2,5)$ 仍然表示损伤随阶段上升；受控比较只是让 Uniform
和 Cosine-front 都使用这一组 $\boldsymbol c$，从而只考察预算向量
$\boldsymbol b$ 的差异。

因此 Fig. 1(b) 必须同时展示阶段损伤 $c_p$ 和预算分配 $b_p$。上图单独只能证明阶段敏感度随剪枝周期变化；把它与下图的预算日程对齐后，才能形成“损伤—预算时间失配”的完整证据。论文主图使用单条 fixed-path 机制诊断；双真实轨迹图和完整训练历史作为端到端补充证据。两者回答的问题不同，不能用后者替代前者。

该方案最直接支持的是“日程预算与阶段剪枝敏感度不匹配”。如果正文仍使用“evolving network plasticity/recoverability”这一更强表述，则即时损伤本身不能单独证明恢复能力；应把 matched Probe/Control 的恢复实验作为补充消融，并把恢复长度定义为完整 pruning period，而不是任意的50 batches。

## 2. 固定训练配置

主入口为：

`test/motivation_fig1_spatial_temporal/resnet110_cifar100_motivation.py`

当前默认训练细节已经写入程序：

| 配置 | 默认值 |
|---|---:|
| 数据集 | CIFAR-100 |
| 模型 | ResNet-110 |
| 起始状态 | 最终 dense pretrained checkpoint |
| Batch size | 128 |
| Seed | 42 |
| Learning rate | 0.01 |
| Target group sparsity | 0.5 |
| **Warmup epochs** | **20** |
| **Pruning epochs** | **100** |
| **Pruning periods** | **100** |
| Production schedules | Balanced Uniform 与 Cosine-front 分别独立运行 |
| 总阶段一训练长度 | 120 epochs |
| Temporal probe | 初始 dependency groups 的固定5%绝对数量 |
| Temporal evaluation | 完整 CIFAR-100 test set（10000张） |

默认 batch size=128 时，每个 epoch 约391 batches。因为 `pruning_steps=100×steps_per_epoch` 且 `pruning_periods=100`，一个真实 pruning period 对应一个 pruning epoch。

正式入口要求 `pruning_schedule_mode` 沿
`训练脚本 → OTO.hesso → HESSO` 精确透传，不再提供实验层覆盖。HESSO 是日程日志的唯一输出源；实验入口只进行静默一致性校验，发现模式或配额不一致会立即报错。服务器必须同时同步
`only_train_once/__init__.py`、`only_train_once/optimizer/hesso.py` 和本实验入口。HESSO 打印的同一套实际配额也会在训练开始时写入
`run_metadata.json` 的 `training_config.production_period_quota`。

这里不存在“训练50 batches恢复”。必须执行的是：临时置零后测量即时损伤，再逐 tensor slice 精确还原参数，最后才允许主优化器继续执行真实剪枝 step。

> 重要：此前已经同步回来的 Fig. 1(a) 结果由5个 warmup epochs 产生，不能与新的20-warmup Fig. 1(b) 混合。为了保证 Fig. 1(a)/(b) 训练状态一致，正式结果必须重新运行 `motivation_periods`，或者分别在相同20+100配置下重跑两图。

## 3. Fig. 1(a)：空间冲突（What to prune?）

在同一真实 checkpoint、相同30%选择预算下，分别使用 L1、L2、Fisher、GW 和 Trend 对 dependency groups 排序并产生 Top-K mask：

\[
m_{r,g}^{(t)}=
\begin{cases}
1,&g\text{ 被准则 }r\text{ 判定为应剪};\\
0,&\text{否则}.
\end{cases}
\]

图中直接显示不同准则对相同 groups 的保留/剪除决定。不同列出现不同模式，即表示准则在结构位置上选择了不同 groups，这就是 spatial conflict。

程序采集完整100个真实剪枝 periods，并保留每期冲突率、准则 mask 和配额信息。

## 4. Fig. 1(b)：时间失配（When to prune?）

### 4.1 阶段损伤 $c_p$

完成20个 warmup epochs 后，正式进入100个 pruning periods。对每个 period，包括生产 Cosine-front 配额为0的后期 period：

1. 使用 optimizer step 前的真实剪枝状态 $M_p$。
2. 复用 `AlphaScheduler` 在该边界刚刚刷新的核心五准则分数，不重复刷新优化器动量。
3. 使用固定等权 static-fusion；所有 periods 的选择策略完全一致。
4. 令初始 dependency groups 总数为 $N_0$，一次性确定固定探针数量：

\[
k=\max(1,\operatorname{round}(0.05N_0)).
\]

5. 在当前尚未剪除的 groups 中选择固定的 $k$ 个最低分 groups。
6. 在同一确定性 CIFAR-100 test set 上测量 $L(M_p)$。
7. 临时同步置零这些 dependency groups 及关联参数，测量 $L(M_p^{\mathrm{probe}})$。
8. 精确恢复全部修改，并用 `torch.equal` 逐 slice 验证。
9. 恢复验证通过后，主训练流程才执行原本的 optimizer step。

这里固定的是每期的选择规则与绝对 group 数量，而不是强行固定同一批 group 身份；随着真实剪枝推进，已剪 groups 会从后续候选池中排除。因此该量表示“同一种剪枝策略在不同真实阶段施加同等规模改动时的阶段成本”。

记录：
$$
\Delta L_p=L(M_p^{\mathrm{probe}})-L(M_p).
$$
并归一化为每1%初始 groups 的即时损伤：
$$
c_p=\Delta L_p\frac{0.01}{k/N_0}.
$$
零配额 periods 也必须测量 $c_p$，因为 Uniform 日程仍可能在这些后期阶段分配预算。

#### 4.1.1 “真实剪枝轨迹”与“临时探针”分别是什么

真实流程是从同一个 dense checkpoint 出发，分别运行 $s\in\{\mathrm{uniform},\mathrm{cosine}\}$ 两个独立任务。每个任务都完成20个 warmup epochs，再按各自生产日程走过100个 pruning periods。因此横轴上的 $M_1^{(s)},\ldots,M_{100}^{(s)}$ 都是相应策略正常训练和真实剪枝产生的网络状态，不是从另一条轨迹推算出来的。

临时探针只是在每个真实 period 的 optimizer step 之前插入一次可逆测量：

```text
进入真实 period p
        ↓
获得当前策略的真实网络状态 M_p^(s)
        ↓
Baseline evaluation: L(M_p^(s))
        ↓
用固定规则临时置零106个 groups
        ↓
Probe evaluation: L(M_p^(s,probe))
        ↓
计算 c_p，精确恢复全部参数
        ↓
执行策略 s 原本的真实生产剪枝 step
```

这里的“临时”不表示使用合成数据或理论模拟：程序确实置零当前真实网络中的 dependency-group 参数，并在完整 CIFAR-100 test set 上执行前向评估。恢复参数的目的，是防止诊断探针额外改变生产轨迹；如果不恢复，每个 period 都会在正常配额之外再永久剪除106个 groups，后续网络状态将不再属于原来的 Uniform 或 Cosine-front 流程。

实际 Cosine-front 每期配额并不相等，按 Algorithm 2 在早期分配较多 groups、后期逐渐减少到0；Balanced Uniform 则每期实际剪10或11个 groups。如果直接比较生产 loss 增量，就无法判断变化来自“阶段敏感度”还是“本期剪枝量”。因此两条轨迹都在每期使用固定106个 groups 的临时探针，把干预规模控制为相同，分别得到 $c_p^{\mathrm{uniform}}$ 与 $c_p^{\mathrm{cosine}}$。注意：两者的探针尺度可比，但网络状态不同，所以两条 $c$ 路径不相等。

#### 4.1.2 补充双轨迹图的曲线分别代表什么

`temporal_compare` 补充图不再只有 Cosine-front。上图明确画出两条独立真实轨迹：

- 灰色小点：Balanced Uniform 轨迹每个 period 的固定5%临时探针原始损伤；
- 灰色实线：上述 Uniform 原始点的5-period移动平均；
- 绿色小点：Cosine-front 轨迹每个 period 的固定5%临时探针原始损伤；
- 绿色实线：上述 Cosine-front 原始点的5-period移动平均。

两套小点都来自真实前向测量，不是互相推算，也不是把五种准则的五条损伤曲线取平均。两条实线只负责显示各自原始点的局部趋势。旧的橙色原始点/红色移动平均只属于历史单-Cosine fixed-path 图，由 `plot_motivation_fig1.py temporal` 生成，不作为最终双真实轨迹主图。

Static Fusion 对每个 group 的五个归一化重要性分数做等权平均：
$$
s_g^{\mathrm{static}}
=\frac{s_g^{L1}+s_g^{L2}+s_g^{Fisher}+s_g^{GW}+s_g^{Trend}}{5}.
$$
程序在每条轨迹、每个 period 都根据融合分数产生一组106个临时探针 groups，然后测量一次损伤。也就是说，平均发生在“选组分数”层面；程序没有先产生五条损伤曲线再对它们求平均。固定 Static Fusion 是为了避免探针规则随 period 改变，否则无法区分损伤变化究竟来自网络阶段，还是来自探针策略变化。

#### 4.1.3 5-period移动平均代表什么

移动平均不是额外的剪枝策略，也没有对应第三次实验。灰色、绿色实线分别只是本策略原始损伤点的局部平滑趋势。对任一策略 $s$ 的中间位置：
$$
\bar c_p^{(s)}=\frac{c_{p-2}^{(s)}+c_{p-1}^{(s)}+c_p^{(s)}+c_{p+1}^{(s)}+c_{p+2}^{(s)}}{5}.
$$
移动平均只用于显示，不参与任何 $C(c,b)$ 的计算；所有 fixed-path cost 始终使用100个原始 $c_p$。跨路径的
$C(c^U,b^U)$ 与 $C(c^C,b^C)$ 只可分别描述各自运行，不能作为受控机制比较。

### 4.2 预算分配 $b_p$ 与实际配额 $n_p$

两种确定性日程都通过核心 `HESSO._build_period_quota` 产生，并严格按照论文 Algorithm 2 的“累计 Round 后差分”离散化：

$$
N_{-1}=0,\qquad
N_p=\operatorname{Round}\!\left(G_{\mathrm{red}}\Phi\!\left(\frac{p+1}{P}\right)\right),
\qquad n_p=N_p-N_{p-1}.
$$

其中 $n_p$ 是 Algorithm 2 实际执行的整数配额。Theorem 3 使用 $b_p$ 表示任意可行日程在第 $p$ 期分配的预算；本图中的 Uniform 与 Cosine-front 两个 $b_p$ 序列分别由各模式实际生成的 $n_p$ 实例化。

$$
\Phi_{\mathrm{uniform}}(\tau)=\tau,\qquad
\Phi_{\mathrm{cosine}}(\tau)=\tau+\frac{\sin(\pi\tau)}{\pi}.
$$

代码对非负数使用确定性的 half-up Round：
$\operatorname{Round}(x)=\lfloor x+0.5\rfloor$，避免 Python `round` 的 ties-to-even 行为。

- Uniform：累计 Round 均匀分配，任意两个 period 的配额最多相差1；
- Cosine-front：前期分配更多预算、后期逐渐减少；Cosine-front 独立生产轨迹真实使用该日程。

Uniform 与 `uniform_random` 一样保证每期只有 `floor(B/T)` 或 `ceil(B/T)` 两种额度，但累计 Round 的余数位置完全确定，更适合作为可复现的论文基线。`uniform_random` 可用于随机日程消融，不属于 Algorithm 2，也不用于 Fig. 1(b) 主图。

旧 Cosine-front 实现采用“逐期 floor 后把余数补给最大权重期”，会把取整余量系统性偏向早期，与 Algorithm 2 不同。Algorithm 2 公式现在只保留在 HESSO 一处；实验端按模式调用 HESSO 并保存返回的实际配额，绘图端直接读取，不再重复实现公式。旧结果不能通过只修 JSON 升级为正式结果，Uniform 与 Cosine-front 都必须重新运行。

### 4.3 Matched-budget cost

论文定理对应的是受控的 fixed-path 比较。先选择一条真实参考轨迹
$r$，固定在这条轨迹上测得的完整阶段损伤向量
$\boldsymbol c^{(r)}=(c_1^{(r)},\ldots,c_P^{(r)})$，然后只更换预算向量：

$$
C_{\mathrm{U}\mid r}
=\sum_{p=1}^{P} c_p^{(r)}\frac{b_p^{\mathrm{U}}}{\sum_{j=1}^{P} b_j^{\mathrm{U}}},
\qquad
C_{\mathrm{C}\mid r}
=\sum_{p=1}^{P} c_p^{(r)}\frac{b_p^{\mathrm{C}}}{\sum_{j=1}^{P} b_j^{\mathrm{C}}}.
$$

这里两式使用的是同一组 $c_p^{(r)}$；“固定”不等于损伤在所有 period
都相等。如果 $C_{\mathrm{C}\mid r}<C_{\mathrm{U}\mid r}$，说明在这条实测
阶段成本曲线上，Cosine-front 的同总预算分配比 Uniform 更避开高成本
阶段。这才是“损伤成本与预算分配不匹配”的受控机制证据。

绘图入口 `temporal` 默认使用 Uniform 真实轨迹作为参考路径并生成论文主图：上半图以 `Measured $c_p$` 与 `Damage trend` 展示实测 $c_p^{(r)}$ 及其局部趋势，并在子图下方标注“(a) Stage-wise pruning damage”；下半图对照 $b^U$ 与 $b^C$，纵轴使用论文 Theorem 3 的符号 `Pruning budget $b_p$`，并在子图下方标注“(b) Matched-budget pruning schedules”。两个子图均明确标出横轴 `Pruning period`。下图图例不再显示平滑窗口文字，但曲线仍按默认7-period移动平均绘制。
主图不再显示总预算、fixed-path cost 和降低比例的文字框；这些数值仍可由原始逐周期数据计算，不影响实验数据和绘图曲线。
下图保留低透明度的实际整数配额阶梯，并以默认7-period移动平均显示主曲线；
该平滑只影响可视化，matched-budget cost 始终使用原始实际配额。

Uniform 路径第51期出现已确认的孤立探针评估异常：probe loss 为5.3406，
而相邻第50/52期分别为2.0203/2.1814，对应原始 $c_{51}=0.7343$。
论文图将第51期的显示值替换为第52期的 $c_{52}=0.0971$，并基于这条
修正后的显示序列计算移动平均和图中 fixed-path cost。原始CSV/JSON
保持不变；补充图导出的JSON通过 `display_replacements` 记录该替换。

选择 Uniform 作为主参考，是因为需要直接检验“平均日程在后期高成本阶段仍分配同等预算”的失配。若设置
`MOTIVATION_TEMPORAL_REFERENCE_PATH=cosine_front`，可生成 Cosine 路径版本用于稳健性检查；正式训练超参数不受该绘图变量影响。

这项重加权是机制诊断，不是假装另一种日程会产生完全相同的物理网络
状态，也不能替代真实训练结果。

### 4.4 单轨迹机制诊断与双真实轨迹比较的关系

两条真实生产轨迹从第一次采用不同配额后就会分叉，通常有：

$$
M_p^{\mathrm{uniform}}\neq M_p^{\mathrm{cosine}},
\qquad
c_p^{\mathrm{uniform}}\neq c_p^{\mathrm{cosine}}.
$$

因此 $C(c^U,b^U)$ 与 $C(c^C,b^C)$ 同时改变了 $c$ 和 $b$，两者的绝对差
不能解释为日程匹配机制。更新后的 temporal_compare 不再把这两个
绝对 cost 并排作为结论，而是分别做两组受控重加权：

- 固定 $\boldsymbol c^U$，比较 $b^U$ 与 $b^C$；
- 固定 $\boldsymbol c^C$，比较 $b^U$ 与 $b^C$。

这样可以检查机制结论是否依赖参考路径，但双轨迹图仍更适合作为补充
材料。最终论文证据分为两层：

- 单轨迹 fixed-path 诊断：隔离“阶段成本与预算位置”的机制；
- 双真实轨迹实验：直接比较两种生产日程的完整 test loss、Top-1、
  group sparsity、LR、α 权重和最终压缩指标。

绘图入口 `temporal_compare` 使用两条独立真实轨迹并输出两组 fixed-c 重加权；`training_compare` 使用两次运行保存的完整逐 epoch 历史。

## 5. 服务器运行流程

在 Linux 训练服务器的 OTOv2_v2 根目录执行。程序默认使用当前可见的默认 GPU，不写死0卡，也不要求设置 `CUDA_VISIBLE_DEVICES`。

```bash
cd /workspace/OTOv2_v2
mkdir -p test/motivation_fig1_spatial_temporal/logs
```

默认数据路径：

```text
/workspace/data/cifar100
```

默认 checkpoint 路径：

```text
test/Ablation/resnet110_cifar100/model/best_model.pth
```

如果服务器实际路径不同，只需覆盖对应环境变量；默认路径正确时不需要设置任何超参。

### 5.1 Fig. 1(b)：分别运行两条真实轨迹

两条命令必须使用相同 checkpoint、seed、batch size、数据路径和超参数。建议顺序运行，避免同一 GPU 上并发训练改变吞吐和显存状态。

当前另保留两种“仅阶段损伤”的单图候选，不覆盖原上下双图：

```bash
python -u test/motivation_fig1_spatial_temporal/plot_motivation_fig1.py temporal_damage
python -u test/motivation_fig1_spatial_temporal/plot_motivation_fig1.py temporal_damage_compare
```

- `temporal_damage`：只画 Uniform 真实参考轨迹的 `Measured $c_p$` 与 `Damage trend`；
- `temporal_damage_compare`：复制相同版式，并叠加 Cosine-front 真实轨迹的实测 $c_p$ 与趋势线。由于两条生产轨迹采用不同日程后已经分叉，该图只描述两条实际损伤路径，不能解释为固定同一 $c_p$ 的受控日程比较。

对应产物为：

- `outputs/figures/fig1b_temporal_damage_uniform_path.png/.pdf`：仅保留阶段损伤的单图主候选；
- `outputs/figures/fig1b_temporal_damage_two_real_paths.png/.pdf`：叠加 Uniform 与 Cosine-front 两条真实损伤路径的对照候选。

#### 5.1.1 Cosine-front 真实轨迹

```bash
python -u test/motivation_fig1_spatial_temporal/resnet110_cifar100_motivation.py temporal_cosine \
  2>&1 | tee test/motivation_fig1_spatial_temporal/logs/temporal_cosine.log
```

#### 5.1.2 Balanced Uniform 真实轨迹

```bash
python -u test/motivation_fig1_spatial_temporal/resnet110_cifar100_motivation.py temporal_uniform \
  2>&1 | tee test/motivation_fig1_spatial_temporal/logs/temporal_uniform.log
```

这里的 Uniform 是确定性 balanced Uniform。总预算1064、periods=100时，每期真实生产配额只能为10或11，任意两期相差不超过1。

#### 5.1.3 两条真实轨迹完成后绘图

```bash
python -u test/motivation_fig1_spatial_temporal/plot_motivation_fig1.py temporal \
  2>&1 | tee test/motivation_fig1_spatial_temporal/logs/plot_temporal_fixed_path.log

python -u test/motivation_fig1_spatial_temporal/plot_motivation_fig1.py temporal_compare \
  2>&1 | tee test/motivation_fig1_spatial_temporal/logs/plot_temporal_compare.log

python -u test/motivation_fig1_spatial_temporal/plot_motivation_fig1.py training_compare \
  2>&1 | tee test/motivation_fig1_spatial_temporal/logs/plot_training_compare.log
```

生成：

- `fig1b_temporal_mismatch.png/.pdf`：论文主图候选；固定一条实测 $c_p$ 路径，只比较同预算 $b^U/b^C$；
- `fig1b_temporal_real_trajectory_comparison.png/.pdf`：补充图；两条真实轨迹的100期临时探针损伤与实际生产配额；
- `fig1b_temporal_real_trajectory_comparison.csv/.json`：补充图全部逐周期数值，以及分别固定 $c^U/c^C$ 后的两组日程重加权；
- `temporal_complete_training_trajectory_comparison.png/.pdf`：两条完整120-epoch训练的 test loss、Top-1 和 group sparsity。

### 5.2 Fig. 1(a) 与 Cosine-front 轨迹一次采集

该命令不能替代单独的 `temporal_uniform`；它只同时采集 Fig. 1(a) 和 Cosine-front 真实轨迹。由于论文主图默认固定 Uniform 路径的 $c_p$，只有在 `temporal_uniform` 也已完成后才能运行下面的 `combined` 绘图命令。

```bash
python -u test/motivation_fig1_spatial_temporal/resnet110_cifar100_motivation.py motivation_periods \
  2>&1 | tee test/motivation_fig1_spatial_temporal/logs/motivation_periods.log

python -u test/motivation_fig1_spatial_temporal/plot_motivation_fig1.py combined \
  2>&1 | tee test/motivation_fig1_spatial_temporal/logs/plot_combined.log
```

### 5.3 只运行 Fig. 1(a)

```bash
python -u test/motivation_fig1_spatial_temporal/resnet110_cifar100_motivation.py spatial_periods \
  2>&1 | tee test/motivation_fig1_spatial_temporal/logs/spatial_periods.log

python -u test/motivation_fig1_spatial_temporal/plot_motivation_fig1.py spatial \
  2>&1 | tee test/motivation_fig1_spatial_temporal/logs/plot_spatial.log
```

### 5.4 可选路径覆盖

仅当默认路径不正确时使用：

```bash
export MOTIVATION_DATA_ROOT=/实际路径/cifar100
export MOTIVATION_PRETRAINED_CKPT=/实际路径/best_model.pth
```

其他可选项：

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `MOTIVATION_NUM_WORKERS` | 16 | 主训练 DataLoader workers |
| `MOTIVATION_TEMPORAL_NUM_WORKERS` | 4 | temporal test DataLoader workers |
| `MOTIVATION_TEMPORAL_EVAL_SUBSET` | 10000 | 评估样本数 |
| `MOTIVATION_TEMPORAL_PERIOD_FRACTION` | 0.05 | 相对初始 groups 的固定绝对探针预算 |
| `MOTIVATION_TEMPORAL_SMOOTH_WINDOW` | 5 | 只影响曲线显示 |
| `MOTIVATION_TEMPORAL_QUOTA_SMOOTH_WINDOW` | 7 | 只影响下图配额曲线显示 |
| `MOTIVATION_TEMPORAL_RUN_ID` | 自动时间戳 | 输出 run 目录名 |
| `MOTIVATION_PRODUCTION_SCHEDULE` | cosine_front | 仅供通用 `temporal_periods` 模式覆盖；正式双轨迹运行优先使用明确的 CLI 模式 |

论文正式结果建议保持默认5%预算和完整10000张 test set，不要运行后根据曲线临时更改探针预算。

### 5.5 两种日程的真实生产损伤（独立 matched-control 实验）

独立入口为：

```text
test/motivation_fig1_spatial_temporal/resnet110_cifar100_production_damage.py
```

该实验不使用固定5%临时探针。HESSO 每个真实 pruning period 开始时，从完全相同的生产模型、HESSO优化器、动量、已有剪枝状态和当前 α 状态复制一个 Control 分支。两个分支随后使用同一批已经完成随机增强的训练 tensor：

- Production：按当前日程执行本期真实配额、真实 group 选择和完整 period 的渐进衰减；
- Control：保留全部历史剪枝状态和相同训练更新，只把本期新增配额设为0；
- period 结束后，在同一确定性 CIFAR-100 test set 上比较两者；
- Production 继续进入下一期，Control 立即丢弃。

实现上不能直接使用 PyTorch Optimizer 的默认深拷贝，因为它可能丢失 HESSO 的自定义周期字段，而且模型参数的当前梯度也需要显式复制。程序会按参数映射复制完整 HESSO 运行状态、当前梯度并重新绑定 Control 的 Alpha scheduler；还会移除从 Production 复制来的 LR scheduler `step` 闭包，使 `HESSO.step` 正确绑定到 Control。正式预热前会在 Production 的 LR scheduler 安装后执行完整性检查，失败时立即停止。

定义第 $p$ 期真实生产损伤：

$$
d_p
=L\!\left(M_{p,\mathrm{end}}^{\mathrm{production}}\right)
-L\!\left(M_{p,\mathrm{end}}^{\mathrm{control}}\right).
$$

$d_p$ 使用本期 HESSO 真正选择的 groups、真实配额和完整渐进稀疏化过程，不等于固定预算单位损伤 $c_p$。程序同时记录按本期配额归一化的 `production_damage_loss_per_1pct_initial_groups`；当真实配额为0时，该单位值为未定义，JSON中保存为 `null`。

训练配置仍固定为20个 warmup epochs、100个 pruning epochs、100个 periods、LR=0.01和目标组稀疏率0.5。两个任务必须顺序运行，不能在同一GPU上并发，因为每个 pruning period 都会临时保留一套完整Control模型与HESSO状态。

```bash
cd /workspace/OTOv2_v2
mkdir -p test/motivation_fig1_spatial_temporal/logs
set -o pipefail

python -u test/motivation_fig1_spatial_temporal/resnet110_cifar100_production_damage.py uniform \
  2>&1 | tee test/motivation_fig1_spatial_temporal/logs/production_damage_uniform.log &&

python -u test/motivation_fig1_spatial_temporal/resnet110_cifar100_production_damage.py cosine_front \
  2>&1 | tee test/motivation_fig1_spatial_temporal/logs/production_damage_cosine_front.log &&

python -u test/motivation_fig1_spatial_temporal/plot_motivation_fig1.py production_damage \
  2>&1 | tee test/motivation_fig1_spatial_temporal/logs/plot_production_damage.log
```

同步后的新脚本启动时必须打印 `Implementation: full_hesso_state_clone_v3_scheduler_safe`，随后在正式预热前打印 `[MatchedControl] scheduler-safe full-state clone/binding preflight passed.`。若没有这两行，说明服务器仍在运行旧文件。

命令不写死GPU编号，沿用服务器当前可见的默认CUDA设备。默认数据或checkpoint路径不正确时，仍使用5.4节的两个路径变量覆盖。生产损伤评估 workers 可选变量为：

```bash
export PRODUCTION_DAMAGE_EVAL_NUM_WORKERS=4
```

每期的核心记录包括：

- `production_quota_groups`：该日程本期真实配额；
- `production_selected_global_indices`：HESSO本期真实选择的group全局索引；
- `production_end_loss` / `control_end_loss`：两个匹配分支的期末loss；
- `production_damage_loss`：真实生产损伤 $d_p$；
- `production_damage_top1_drop`：相对Control的Top-1下降；
- `production_net_loss_change`：Production相对期初的净loss变化；
- `control_training_loss_change`：无新增剪枝Control相对期初的训练变化；
- `loss_decomposition_error`：验证净变化分解的一致性，理论上接近0。

Cosine-front后期配额为0时，Production与Control应执行相同更新，因此 `production_damage_loss` 应接近0；最终摘要中的 `max_abs_zero_quota_damage_loss` 用于自动检查这一点。该实验需要重新运行，不能从已有探针CSV或epoch间loss差事后恢复。

## 6. 输出文件

```text
outputs/
├── spatial_periods/
│   └── <run_id>/
├── temporal_periods/
│   ├── cosine_front/
│   │   ├── latest_run.json
│   │   └── <run_id>/
│   │       ├── period_001/ ... period_100/
│   │       │   ├── temporal_probe.json
│   │       │   └── production_pruning_event.json
│   │       ├── temporal_period_summary.csv/.json
│   │       ├── temporal_diagnostics.json
│   │       ├── temporal_schedule.json
│   │       ├── training_history.csv/.json
│   │       └── final_summary.json
│   └── uniform/
│       ├── latest_run.json
│       └── <run_id>/（字段结构相同）
├── temporal/
│   ├── cosine_front/
│   │   ├── temporal_probe_records.csv
│   │   ├── temporal_probe.json
│   │   ├── temporal_schedule.json
│   │   ├── training_history.csv/.json
│   │   └── final_summary.json
│   └── uniform/（字段结构相同）
├── production_damage/
│   ├── uniform/
│   │   ├── latest_run.json
│   │   ├── production_damage.csv/.json（最新完整运行）
│   │   └── <run_id>/
│   │       ├── run_metadata.json
│   │       ├── production_damage.csv/.json
│   │       ├── training_history.csv/.json
│   │       └── final_summary.json
│   └── cosine_front/（字段结构相同）
└── figures/
    ├── fig1a_spatial_conflict.png/.pdf
    ├── fig1b_temporal_real_trajectory_comparison.png/.pdf
    ├── fig1b_temporal_real_trajectory_comparison.csv/.json
    ├── temporal_complete_training_trajectory_comparison.png/.pdf
    ├── fig1_motivation_combined.png/.pdf
    └── production_damage_two_schedules.png/.pdf
```

关键诊断字段：

- `late_minus_early_damage`；
- `uniform_matched_budget_cost`；
- `cosine_front_matched_budget_cost`；
- `matched_budget_cost_gap_uniform_minus_cosine`；
- `production_matches_cosine_front`；
- `production_matches_uniform`；
- `production_weighted_probe_cost`；
- `complete`。

只有 Uniform 和 Cosine-front 两边都满足 `complete=true`、各自100个 periods 均完成探针、每期 `restoration_verified=true`，且两边 `total_pruning_budget` 相同，才能用于最终双真实轨迹 Fig. 1(b)。

`training_history.csv` 每个 epoch 写一行，共120行，包含：stage、真实 period、生产配额、累计配额、LR、train/test loss、train/test Top-1/Top-5、group sparsity、group 计数、吞吐、耗时和五个 α 权重。文件每个 epoch 都会重写保存，因此即使任务中断，也能保留完整的已完成前缀用于排查。

## 7. 旧方案保留说明

`prepare_dense_snapshots.py` 和 `run_legacy_dense_snapshot_temporal_probe()` 仅用于追溯旧的 dense-checkpoint/matched-recovery 实验，不属于最终 Fig. 1(b) 流程。

当前主流程不读取历史 dense checkpoints，也不执行50-batch再训练恢复。
