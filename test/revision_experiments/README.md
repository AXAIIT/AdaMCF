# TPAMI Revision Experiments for AdaMCF

本目录用于管理 AdaMCF 论文修改阶段的补充实验。当前采用“四类实验”的标准结构，每一类实验对应一个顶层目录；每个具体实验再单独建立子目录并保留 README.md，确保实验目的、变量、做法、预期结果和最终记录可追踪。

## 标准目录结构

```text
test/revision_experiments/
├── 01_schedule_comparison/
│   ├── 01_uniform/
│   ├── 02_linear/
│   ├── 03_cubic/
│   ├── 04_exponential/
│   └── 05_cosine_front/
├── 02_zo_sensitivity/
│   ├── 01_eps_001_update_every_period/
│   ├── 02_eps_003_default/
│   ├── 03_eps_005_update_every_period/
│   ├── 04_eps_003_update_every_2_periods/
│   └── 05_eps_003_update_every_4_periods/
├── 03_component_overhead_profiling/
│   ├── 01_resnet110_cifar100/
│   └── 02_resnet50_imagenet_optional/
└── 04_inference_latency_edge/
    ├── 01_server_resnet110_batch1/
    ├── 02_server_resnet50_batch1/
    ├── 03_jetson_resnet110_batch1/
    └── 04_jetson_resnet50_batch1_optional/
```

## 四类实验与数量

| 类别 | 具体实验数 | 主要回应的问题 | 优先级 |
|---|---:|---|---|
| 01_schedule_comparison | 5 | cosine-front schedule 是否缺少 matched-budget 对比 | 最高 |
| 02_zo_sensitivity | 5 | ZO alpha update 对 eps 和 update frequency 是否敏感 | 最高 |
| 03_component_overhead_profiling | 2 | AdaMCF 各组件训练开销是多少 | 中 |
| 04_inference_latency_edge | 4 | FLOPs reduction 是否转化为实际 latency speedup | 中 |

## 执行原则

1. 不修改 `only_train_once/` 下的底层 optimizer 源码。
2. 不在脚本中强制切换 conda 环境，使用当前 shell 已激活的环境。
3. 每个实验独立保存 `config.json`、日志、结果和 checkpoint。
4. 结果必须来自实际运行日志或结果文件，不能为了符合预期手动调整。
5. 旧目录或旧脚本如果仍存在，只作为临时历史文件；本 README 中列出的四类目录是后续标准结构。
