# YOLO11 AdaMCF 结构化剪枝

本目录将既有 YOLOv5 的 AdaMCF 流程迁移到 YOLO11：OTO 建图、HESSO 多准则组稀疏训练、物理子网构建、微调和 ONNX 导出。

YOLO11 适配已直接落在 OTO 源码中：`only_train_once/torch_compat.py` 兼容 PyTorch 2.11+ 移除的 ONNX 私有全局对象；`only_train_once/operation/operator.py` 根据 `qkv` 的内部类型分派同名 `Attention`。ViT 的 `qkv` 是 `Linear`，仍使用原来的 `VitAttentionOTO`；YOLO11 的 `qkv/proj/pe` 是 `Conv2d` 包装器，使用 `YOLO11AttentionOTO`。入口脚本会从自身位置推导 `OTOv2_v2` 根目录并直接导入 `only_train_once`，不再需要 `yolo11_oto_compat.py`。

## 已验证的兼容边界

- `conservative_v1` 下 YOLO11n 已完成 OTO 建图、50 epoch HESSO、物理子网构建和独立 COCO val。
- P10 实际剪除 `68/688=9.8837%` 组，物理子网参数量下降 `3.5465%`、FLOPs 下降 `1.2300%`；80 epoch 微调后独立验证为 `mAP@0.5=53.8012%`、`mAP@0.5:0.95=38.3219%`。
- P30 实际剪除 `206/688=29.9419%` 组，物理子网参数量下降 `9.9959%`、FLOPs 下降 `4.2051%`；物理子网独立验证为 `52.6992% / 37.2077%`，微调完成证据尚未同步完整。
- `C3k2`、`C3k`、`Bottleneck`、`SPPF`、`C2PSA` / `PSA` / `Attention` 与 `Detect` 在 `conservative_v1` 中默认受保护。P10/P30 的成功不能解释为这些模块已支持剪枝。
- ONNX、固定输入对齐和端侧性能必须按各自产物与日志单独验收，不能由 PyTorch 剪枝成功外推。

YOLO11 的 C2PSA/Attention 已不再整体固定：当前已经分别完成 Attention `num_head`、Attention `head_dim` 和 C2PSA `cv1 -> split(2)` 外部宽度的定向物理结构准入。卷积式 Attention 的 Q/K/V 布局为 `key_dim + key_dim + head_dim`，不同于 ViT 的 `3 × embed_dim`；因此两个内部轴继续由 YOLO11 专用合同管理，不能退化为普通 Conv 通道剪枝。Detect 仍只固定最终回归、分类与 DFL 语义输出，隐藏支路和输入联动按 profile 开放。

## 当前复杂模块主线：CSP 卷积依赖闭包

`csp_dense_closure_v1_experimental` 不再只开放单个 `Bottleneck.cv1`。它开放 C3k2/C3k/Bottleneck、SPPF、其前后常规 Conv，以及 YOLO11 Detect 的隐藏层和输入适配支路，让 HESSO 在完整卷积依赖图中探索。C2PSA/PSA/Attention 仍整体保护；其他任务 head 仍整体保护。

Detect 不能按整个模块保护，否则 backbone/neck 输出会因为后继约束失去可剪性。本 profile 只按 OTO 节点组精确固定 Detect 各尺度最终回归输出 `4 × reg_max`、最终分类输出 `nc` 和 DFL 常量卷积的输出维；最终卷积的输入通道仍允许随上游依赖组同步物理删除。Concat 等 auxiliary 组不交给 HESSO 独立选择，但继续传播通道索引。

专用薄封装为 `probe_yolo11_csp_dense_closure.py`。它自动启用 `--module-probe-only`，按 OTO 组隔离非候选组，不再用“保护所有非候选参数名”的方式误伤同一依赖组。`audit.json` 的 `module_pruning_candidate_groups` 会同时记录候选参数、完整依赖闭包参数和闭包中的外部参数；`module_dependency_control` 会记录固定输出组、允许组、被隔离组和保留的 auxiliary 组。

服务器上先执行审计：

```bash
python -u probe_yolo11_csp_dense_closure.py audit \
  --weights /workspace/OTOv2_v2/test/pretrain_yolo11_coco/models/yolo11n.pt \
  --imgsz 640 --device 0 \
  --output runs --name 20260715_yolo11n_csp_dense_closure_v1_audit
```

审计必须满足候选非空、存在可剪非 auxiliary 组且 Detect 固定输出组均已匹配。YOLO11 C3k2 的 `chunk(2)` 会使少量“CSP `cv1` 完整输出 + Bottleneck `cv2` 半宽输出”被旧 OTO 错合为同组；只有同时满足 `cv2.bn`、`num_groups = 2 × BN channels`、成对卷积参数存在且组已设为不可剪的已知边界，才允许以 `known_csp_chunk_boundary_ratio_2` 隔离。隔离比例上限为候选组的 25%；任何其他 tensor transformation failure 仍使审计失败。随后才执行 1 epoch 物理剪枝 probe：

```bash
python -u probe_yolo11_csp_dense_closure.py prune \
  --weights /workspace/OTOv2_v2/test/pretrain_yolo11_coco/models/yolo11n.pt \
  --data /workspace/OTOv2_v2/test/pretrain_yolo11_coco/coco2017_yolo11.yaml \
  --epochs 1 --warmup-epochs 0 --pruning-periods 1 \
  --prune-ratio 0.10 --group-divisible 1 --lr0 0.001 \
  --imgsz 640 --batch 8 --workers 8 --device 0 --no-amp \
  --output runs --name 20260715_yolo11n_csp_dense_closure_v1_probe
```

这里的 10% 是隔离已知不安全边界后的候选依赖闭包目标组稀疏率。审计报告必须记录 `known_quarantined_csp_chunk_groups`、`c3k2_chunk_source_quarantine` 和剩余可剪组数；不能把隔离组写成已支持。只有物理子网能够重新加载、固定 shape 前向、独立 COCO val，且结构统计确实变化，才进入短微调和更长训练。Attention 专用剪枝仍是后一阶段，不因本 profile 开放。

2026-07-15 的首轮 dense closure probe 证明：仅探测 tensor transformation 是否抛出 reshape 异常仍不充分。`model.22.cv1` 从 256 裁为 227 后，PyTorch `chunk(2)` 真实产生 `114+113` 通道，而旧 OTO 把内部 `model.22.m.0.cv1/cv2` 输入裁为 111，并把外层融合输入裁为 338（真实为 352）。第二次恢复构建进一步证明，即使固定 chunk 源，内部 C3 输出从 128 裁为 125 后，旧 OTO 仍未把 `y.extend(...)` 的动态 concat 输入从 384 同步为真实的 `128+128+125=381`。因此旧 dense profile 同时固定所有 `C3k2.cv1` 输出和 `C3k2.m.*` 直接子模块最终输出；子模块内部隐藏卷积与 C3k2 融合卷积输出继续开放。这是复合模块接口边界的语义隔离，不是 BatchNorm 类型兼容分支。

### C3k2 根部 `cv1` 批量开放：PairedChunkContract

`c3k2_paired_chunk_v1_experimental` 不再逐个处理 C3k2，而是一次性接管 YOLO11n 的 8 个根部 `cv1`：`model.2/4/6/8/13/16/19/22.cv1`。每个模块把原始 `2c` 输出压缩为 `c` 个逻辑组，逻辑位置 `i` 同时映射到两个物理位置 `[i, c+i]`；两个 chunk、每个 `self.m` 子模块的外部输出以及最终 `cv2` concat 输入使用同一局部删除索引。`cv2` 外部输出保持固定，避免改变模块对下游的接口宽度。

当前实现只位于 YOLO11 profile/验证层，复用 OTO 已有 `INDEX_GROUPS`、HESSO 和物理构网流程，没有修改 Graph、Operator、TensorTransform 或通用子网构建代码。跨版本本地 CPU audit 已确认 8 个合同全部配置、8 个候选组保护后均可剪，覆盖为 `C3K2_CHUNK 8/8`，无候选审计问题。随后执行纯内存物理回归，对 8 个模块各删除 1 个逻辑组，所有 C3k2 结构 linter 通过，320 输入整网输出 `[1,84,2100]`。这些结果证明建图合同和物理切片函数成立，但没有经过 HESSO 真实置零、checkpoint 保存重载和服务器 640 前向，仍需单独完成 L3 验证。

本阶段还没有开放 `C3k2.m.*` 最终输出的独立异步剪枝轴。当前 paired-chunk 合同会在根部 `cv1` 被选择时同步缩小所有外部 segment；3 个 C3k 型独立 child-output 组继续 quarantine。下一阶段是通用 `SegmentedDynamicConcatContract`：为 `[chunk0, chunk1, child0, ...]` 记录每段原始宽度、当前宽度、offset 和局部删除索引，使各 child output 可以独立剪枝并准确映射到最终 `cv2` 输入。

C3k2 paired-chunk 定向服务器验证命令如下；当前阶段只运行到 `prune`，不执行微调、ONNX、ORT 或端侧流程：

```bash
cd /workspace/OTOv2_v2/test/pretrain_yolo11_coco
set -o pipefail
export PYTHONPATH=/workspace/OTOv2_v2${PYTHONPATH:+:$PYTHONPATH}

RUN_ID=20260721_yolo11n_c3k2_paired_chunk_v1_q10
WEIGHTS=/workspace/OTOv2_v2/test/pretrain_yolo11_coco/models/yolo11n.pt
DATA=/workspace/OTOv2_v2/test/pretrain_yolo11_coco/coco2017_yolo11.yaml
FIXED_INPUT=/workspace/data/coco2017/images/val2017/000000000139.jpg

CUDA_VISIBLE_DEVICES=2 python -u verify_yolo11_csp_dense_closure.py \
  --weights "$WEIGHTS" --data "$DATA" --fixed-input "$FIXED_INPUT" \
  --run-id "$RUN_ID" \
  --module-coverage-profile c3k2_paired_chunk_v1_experimental \
  --probe-module-family C3K2_CHUNK \
  --prune-ratio 0.10 --group-divisible 1 --lr0 0.001 \
  --imgsz 640 --batch 64 --workers 12 --device 0 --oto-opset 16 \
  --output runs --logs logs --start-stage audit --stop-after prune
```

薄封装会自动打开 OTO 图生成，图和 manifest 分别写入 audit/prune 目录下的 `oto_graph/`。

### 全部 `cv1` 一次性开放

`yolo11_all_cv1_open_v1` 不再按模块逐个测试，而是把 YOLO11n 中所有名称为 `*.cv1.conv` 的卷积同时加入候选池。普通 Bottleneck/C3k 等 `cv1` 使用 OTO 原生依赖传播；C2PSA 根部 `cv1` 由 `PairedSplitContract` 接管；8 个 C3k2 根部 `cv1` 由 `PairedChunkContract` 接管。入口使用 `--module-probe-only` 只隔离非 `cv1` 节点组，不缩小 `cv1` 的模块覆盖范围。

当前 profile 固定 Attention 内部 `num_head/head_dim/key_dim`、Detect 最终分类/回归与 DFL 语义输出，以及 C3k2 child-output 的独立动态 concat 轴。C2PSA 和 C3k2 的外部接口输出保持固定；被选中的根部宽度只在模块内部成对缩小。跨版本本地动态 audit 已确认 24 个候选前缀全部落入 24 个可剪非 auxiliary 依赖组，精确特殊族为 `C2PSA_SPLIT 1/1`、`C3K2_CHUNK 8/8`，Attention 参数所有权通过，候选问题为 0。早停门禁要求两个特殊族都产生真实物理命中，同时验证所有被命中模块的结构合同、checkpoint 重载和 640 固定 shape 前向。

服务器 P50 已完成真实 HESSO、物理构网和重载前向：组稀疏率 `49.78%`，参数量 `2.624M→1.908M`（`-27.29%`），FLOPs `6.614G→5.594G`（`-15.43%`）。C2PSA 每侧 `128→123`；4 个实际命中的 C3k2 根部宽度分别为 `64→1`、`64→52`、`128→2`、`128→127`，全部结构 linter 通过。未微调 mAP 降至 `11.28%/4.89%`，主要原因是全局 P50 在没有保留宽度下限时把部分高敏感模块压到 `1/2` 个通道。这属于容量分配失败，不是 paired-split/paired-chunk 结构失败；后续综合质量实验必须引入按模块族配置的最小保留宽度。

本阶段只运行到 `prune`，不执行微调、ONNX、ORT 或端侧流程：

```bash
cd /workspace/OTOv2_v2/test/pretrain_yolo11_coco
set -o pipefail
export PYTHONPATH=/workspace/OTOv2_v2${PYTHONPATH:+:$PYTHONPATH}

RUN_ID=20260721_yolo11n_all_cv1_open_v1_p50
WEIGHTS=/workspace/OTOv2_v2/test/pretrain_yolo11_coco/models/yolo11n.pt
DATA=/workspace/OTOv2_v2/test/pretrain_yolo11_coco/coco2017_yolo11.yaml
FIXED_INPUT=/workspace/data/coco2017/images/val2017/000000000139.jpg

CUDA_VISIBLE_DEVICES=2 python -u verify_yolo11_csp_dense_closure.py \
  --weights "$WEIGHTS" --data "$DATA" --fixed-input "$FIXED_INPUT" \
  --run-id "$RUN_ID" \
  --module-coverage-profile yolo11_all_cv1_open_v1 \
  --prune-ratio 0.50 --group-divisible 1 --lr0 0.001 \
  --imgsz 640 --batch 64 --workers 12 --device 0 --oto-opset 16 \
  --output runs --logs logs --start-stage audit --stop-after prune
```

### C3k2 独立 child-output：SegmentedDynamicConcatContract

`c3k2_dynamic_concat_v1_experimental` 一次性开放 3 个真正独立的 C3/C3k 输出组：`model.6.m.0.cv3`、`model.8.m.0.cv3`、`model.22.m.0.cv3`。其余 5 个 Bottleneck child-output 与根部 paired-chunk 组共享参数所有权，继续保持 fail-closed；本 profile 也固定全部 C3k2 根部 `cv1`，避免在首次验证中叠加两个逻辑轴。

每个独立 child-output 使用自己的局部删除索引。物理构网时不再使用固定原始 offset，而是在执行当前 group 时根据两个 chunk 和所有前序 child 的**当前宽度**计算本 segment 的实时 offset，再删除外层 `C3k2.cv2` 的对应输入列。该方式支持多个 child segment 按任意构网顺序依次缩小，同时保证：

```text
cv2.in_channels = 2 * current_root_c + sum(current_child_output_widths)
```

当前阶段仍只运行到 `prune`，不执行微调、ONNX、ORT 或端侧流程：

```bash
cd /workspace/OTOv2_v2/test/pretrain_yolo11_coco
set -o pipefail
export PYTHONPATH=/workspace/OTOv2_v2${PYTHONPATH:+:$PYTHONPATH}

RUN_ID=20260721_yolo11n_c3k2_dynamic_concat_v1_p85
WEIGHTS=/workspace/OTOv2_v2/test/pretrain_yolo11_coco/models/yolo11n.pt
DATA=/workspace/OTOv2_v2/test/pretrain_yolo11_coco/coco2017_yolo11.yaml
FIXED_INPUT=/workspace/data/coco2017/images/val2017/000000000139.jpg

CUDA_VISIBLE_DEVICES=2 python -u verify_yolo11_csp_dense_closure.py \
  --weights "$WEIGHTS" --data "$DATA" --fixed-input "$FIXED_INPUT" \
  --run-id "$RUN_ID" \
  --module-coverage-profile c3k2_dynamic_concat_v1_experimental \
  --probe-module-family C3K2_DYNAMIC_CONCAT \
  --prune-ratio 0.85 --group-divisible 1 --lr0 0.001 \
  --imgsz 640 --batch 64 --workers 12 --device 0 --oto-opset 16 \
  --output runs --logs logs --start-stage audit --stop-after prune
```

本定向 profile 共有 `64+128+128=320` 个逻辑组。使用 P85 时理论目标约删除 272 组；即使参考上一轮离散调度少约 3 组，仍约为 269 组，高于任意两个候选轴最多可删除的 254 组，因此早停门禁要求 3 个独立 child-output 全部真实命中。该定向合同通过后，历史 8 个根部 chunk 边界与 3 个独立 child-output 边界均具备结构证据。下一步才是设计根部 paired-chunk 与独立 child-output 的复合所有权，使两类轴能在同一个综合 profile 中同时参与 HESSO。

失败 probe 已保存的 `oto_artifacts/DetectionModel_full_group_sparse.pt` 可以通过 `rebuild-subnet` 重新建图和构建物理子网，无需重复稀疏训练。恢复模式会重新执行候选组审计、保存到独立目录，并以 Ultralytics checkpoint 重载后的 640 固定 shape 前向作为硬门槛；报告写入 `rebuild_structured_subnet.json`。被新策略固定的既有零通道继续保留为结构参数，所以恢复子网的实际物理组稀疏率可能低于原 HESSO 日志中的目标值。

`dynamic_concat_parent_child_marker_v13` 使用 audit 冻结的 3 个 child 合同直接读取物理模型宽度，不再通过同名临时属性枚举父模块。父 C3k2 使用 parent marker，child C3k 使用独立 child marker；旧 checkpoint 的 legacy child marker只作为兼容证据读取。未被 HESSO 选择的候选允许保持原宽，3/3 是否真实命中仍由 `post_prune_gate.json` 单独判定。P50 linter 失败产物可先执行无训练恢复：

```bash
cd /workspace/OTOv2_v2/test/pretrain_yolo11_coco
set -o pipefail
export PYTHONPATH=/workspace/OTOv2_v2${PYTHONPATH:+:$PYTHONPATH}

SOURCE_RUN=20260721_yolo11n_c3k2_dynamic_concat_v1_p50_prune
RECOVERY_RUN=20260721_yolo11n_c3k2_dynamic_concat_v1_p50_rebuild_marker_v13
FULL_SPARSE="runs/${SOURCE_RUN}/oto_artifacts/DetectionModel_full_group_sparse.pt"

test -f "$FULL_SPARSE" && \
CUDA_VISIBLE_DEVICES=2 python -u probe_yolo11_c3k2_dynamic_concat.py rebuild-subnet \
  --weights "$FULL_SPARSE" \
  --prune-ratio 0.50 --group-divisible 1 \
  --imgsz 640 --device 0 --oto-opset 16 \
  --output runs --name "$RECOVERY_RUN" \
  2>&1 | tee "logs/${RECOVERY_RUN}.log"
```

### dense v3 九段正式准入

旧稀疏训练产物与 v3 recovery 的合同不同，只能用作故障恢复证据，不能拼接成正式通过。`verify_yolo11_csp_dense_closure.py` 会在当前代码上重新执行同合同闭环：

1. OTO 建图并冻结 dependency contract SHA256；
2. 1 个 pruning period 真实置零，并统计实际命中的候选依赖组/模块族；
3. 构建物理子网，核对合同 hash 与真实参数下降；
4. 对冻结 COCO 图片执行静态 letterbox 固定输入前向；
5. 在物理子网首个 detection task loss backward 后检查梯度 finite/non-zero；
6. 完成 1 epoch 微调与真实 optimizer step；
7. 独立重载 `best.pt`，校验结构和 COCO val；
8. 导出 ONNX 并执行 `onnx.checker`；
9. 使用第 4 步同一图片和同一预处理 tensor 完成 PyTorch/ONNX Runtime 数值对齐。

服务器一键命令：

```bash
cd /workspace/OTOv2_v2/test/pretrain_yolo11_coco

python -c "import torch, ultralytics, onnx, onnxruntime, cv2, numpy; print({'torch': torch.__version__, 'ultralytics': ultralytics.__version__, 'onnx': onnx.__version__, 'onnxruntime': onnxruntime.__version__, 'opencv': cv2.__version__, 'numpy': numpy.__version__})"

python -u verify_yolo11_csp_dense_closure.py \
  --weights /workspace/OTOv2_v2/test/pretrain_yolo11_coco/models/yolo11n.pt \
  --data /workspace/OTOv2_v2/test/pretrain_yolo11_coco/coco2017_yolo11.yaml \
  --fixed-input /workspace/data/coco2017/images/val2017/000000000139.jpg \
  --run-id 20260716_yolo11n_csp_dense_v3_admission_q10 \
  --prune-ratio 0.10 --group-divisible 1 --lr0 0.001 \
  --imgsz 640 --batch 8 --workers 8 --device 0 \
  --export-device cpu \
  --oto-opset 16 --opset 12 \
  --output runs --logs logs
```

每个阶段有独立日志，最终报告为 `runs/<run-id>/nine_stage_admission.json`。正式模式默认要求 `C3k2/C3k/Bottleneck/SPPF/Detect` 都出现真实物理命中；任一模块族只进入候选而未被 HESSO 选中，最终报告会失败并列出 `missing_required_hit_families`，随后应执行该模块族的定向 probe。`--allow-partial-family-hit` 只用于诊断，不得用于验收。

训练和微调仍使用 `--device 0`；导出与 ORT 对齐默认使用 `--export-device cpu`。这样 Ultralytics 检查的是 CPU `onnxruntime`，不会因为 GPU 导出再次自动安装依赖 CUDA 13 的 `onnxruntime-gpu`。

例如完整 profile 再次没有命中 SPPF 时，使用新的 run-id 定向限制候选组；依赖闭包和 auxiliary 传播仍按 OTO 节点组保留：

```bash
python -u verify_yolo11_csp_dense_closure.py \
  --weights /workspace/OTOv2_v2/test/pretrain_yolo11_coco/models/yolo11n.pt \
  --data /workspace/OTOv2_v2/test/pretrain_yolo11_coco/coco2017_yolo11.yaml \
  --fixed-input /workspace/data/coco2017/images/val2017/000000000139.jpg \
  --run-id 20260718_yolo11n_sppf_v3_admission_q10_scopefix \
  --probe-module-family SPPF --required-hit-family SPPF \
  --prune-ratio 0.10 --group-divisible 1 --lr0 0.001 \
  --imgsz 640 --batch 8 --workers 8 --device 0 \
  --export-device cpu \
  --output runs --logs logs
```

## 保守诊断回退：Bottleneck 隐藏通道

若 dense closure 审计暴露具体算子转换问题，可回退到 `bottleneck_hidden_v1_experimental` 定位。它保持 `C3k2.cv1/cv2`、`C3k.cv1/cv2/cv3` 和 `Bottleneck.cv2` 输出不变，只允许 `Bottleneck.cv1` 输出与后续 `cv2` 输入成对物理删除。这样不改变 split/concat 分区、CSP 输出或 shortcut 两端 shape。

专用薄封装为 `probe_yolo11_bottleneck_hidden.py`。它自动启用 `--module-probe-only`，HESSO 只在本阶段候选组中选择，适合先做模块级短程回归。审计报告会列出每个候选前缀对应的 OTO 参数组、`num_groups` 和可剪状态；任一候选未进入 OTO 图，或没有可剪的非 auxiliary 组，程序会在训练前终止。

服务器上先执行审计：

```bash
python -u probe_yolo11_bottleneck_hidden.py audit \
  --weights /workspace/OTOv2_v2/test/pretrain_yolo11_coco/models/yolo11n.pt \
  --imgsz 640 --device 0 \
  --output runs --name 20260715_yolo11n_bottleneck_hidden_v1_audit
```

审计通过后才执行 1 epoch 物理剪枝 probe：

```bash
python -u probe_yolo11_bottleneck_hidden.py prune \
  --weights /workspace/OTOv2_v2/test/pretrain_yolo11_coco/models/yolo11n.pt \
  --data /workspace/OTOv2_v2/test/pretrain_yolo11_coco/coco2017_yolo11.yaml \
  --epochs 1 --warmup-epochs 0 --pruning-periods 1 \
  --prune-ratio 0.10 --group-divisible 1 --lr0 0.001 \
  --imgsz 640 --batch 8 --workers 8 --device 0 --no-amp \
  --output runs --name 20260715_yolo11n_bottleneck_hidden_v1_probe
```

该 probe 的 10% 是“候选 Bottleneck 隐藏组中的目标组稀疏率”，不能与全模型 P10 直接比较。只有 `audit.json`、`prune_summary.json`、物理子网重载、独立 COCO val 和 tensor unsafe exclusion 全部通过，才进入短微调；随后移除 `--module-probe-only` 做 `conservative_v1` 与新 profile 的完整对照。

## 执行顺序

在远程训练环境中执行。请把 `<...>` 换成该环境的实际路径；不要覆盖既有 YOLOv5 输出。

默认数据集配置为本目录的 `coco2017_yolo11.yaml`，其数据根目录与 YOLOv5 脚本保持一致：`/workspace/data/coco2017`。若服务器路径不同，只修改该 YAML 的 `path`；也可用 `--data <其他yaml>` 覆盖。

```bash
python prune_yolo11_adamcf.py audit \
  --weights <path-to-yolo11n.pt> \
  --oto-opset 16 --output runs --name 20260710_yolo11n_audit --device 0
```

审计通过后，执行 AdaMCF 稀疏训练和物理子网构建：

```bash
python prune_yolo11_adamcf.py prune \
  --weights <path-to-yolo11n.pt> \
  --epochs 50 --warmup-epochs 5 --prune-ratio 0.10 \
  --group-divisible 1 --batch 8 --imgsz 640 --oto-opset 16 --device 0 --no-amp \
  --output runs --name 20260710_yolo11n_adamcf_p30
```

`sparse-train` 仍保留为旧命令别名；新实验请使用 `prune`。

30% 组稀疏率实验使用专用薄封装 `prune_yolo11_adamcf_p30.py`。它会自动注入并强制校验 `--prune-ratio 0.30`，并将物理子网固定保存为 `weights/yolo11n_adamcf_p30_structured.pt`，避免与已有10%权重混淆；其余实现全部复用主入口，避免两份剪枝代码发生修复分叉。防覆盖检查不自动启用；确有需要时仍可在主入口或专用入口中显式传入 `--no-overwrite`。完整审计、剪枝、微调和导出命令见 `运行命令.txt` 的“30%组稀疏率完整流程”。

稀疏训练默认关闭 AMP。Ultralytics 在首次开启 AMP 时会为自检查查找并下载官方 `yolo11n.pt`，这不是本脚本传入的实验权重；完成无 AMP 流程验证后，才按需显式传入 `--amp`。

该步骤会生成：

- `runs/<name>/oto_artifacts/DetectionModel_compressed.pt`：OTO 原始结构子网；
- `runs/<name>/weights/yolo11_adamcf_structured.pt`：可由 Ultralytics 重新加载的结构子网；
- `adamcf_runtime.json`、`structured_subnet.json`、`adamcf_run_metadata.json`：配置、保护层和子网烟雾验证证据；其中 HESSO 调度明确以 `optimizer.step` 为单位，并记录实际梯度累积步数。
- `prune_detailed_epoch_metrics.jsonl`、`prune_detailed_epoch_metrics.json`、`prune_detailed_epoch_metrics.csv`：每个验证完成 epoch 的 loss、学习率、GPU 显存、累计总耗时、mAP、HESSO 当前步数/剪枝周期/稀疏度及 AdaMCF α 权重；
- `prune_summary.json`：剪枝阶段总耗时、前后参数/FLOPs、压缩率，以及原模型、稀疏训练模型和重新加载后的物理结构化子网验证指标。控制台同时按项目既有模板打印剪枝后压缩总结。

随后只对结构子网执行常规微调：

```bash
python prune_yolo11_adamcf.py finetune \
  --weights runs/20260710_yolo11n_adamcf_p30/weights/yolo11_adamcf_structured.pt \
  --baseline-weights <path-to-yolo11n.pt> \
  --epochs 80 --close-mosaic 20 --lr0 0.001 --batch 8 --imgsz 640 --device 0 \
  --output runs --name 20260710_yolo11n_adamcf_p30_finetune
```

微调器会直接复制 OTO 物理子网，不再按 YAML 重建标准 YOLO11；训练开始前同时比对网络结构指纹与完整 `state_dict` 数值哈希，`best.pt` 保存后再次验证网络结构，不一致时直接终止。剪枝压缩总结只在 `prune` 阶段打印，`finetune` 不再重复；微调结束后单独打印最佳模型路径、最终 mAP、未剪枝基线 mAP、精度变化及三项加载校验。微调目录会对应生成 `finetune_detailed_epoch_metrics.{jsonl,json,csv}` 与 `finetune_summary.json`，其中包含未剪枝基线、微调前结构化子网、微调后最佳模型的独立验证结果和结构/权重校验证据。现有 Ultralytics 进度条和其 ETA 保持不变；本脚本只额外记录每个 epoch 的累计总耗时与阶段总结。

最后导出 ONNX。当前入口默认执行 `onnx.checker`；正式复杂模块准入还必须提供固定输入并启用 ORT 对齐：

```bash
python prune_yolo11_adamcf.py export \
  --weights runs/20260710_yolo11n_adamcf_p30_finetune/weights/best.pt \
  --imgsz 640 --opset 12 --device 0 \
  --export-output runs/20260710_yolo11n_adamcf_p30_finetune/yolo11n_adamcf_p30.onnx \
  --fixed-input /workspace/data/coco2017/images/val2017/000000000139.jpg \
  --onnx-checker --onnxruntime-align
```

`--oto-opset` 仅控制 OTO 建图时使用的 ONNX 内部追踪 opset，默认 16；`export` 的 `--opset` 仅控制最终 ONNX 文件，默认 12。两者必须分别记录和按目标运行环境设置。

## 与项目 6 的衔接

这里产出的文件仅是算法测试产物。进入项目 6 前，必须另行建立 `runs/<run_id>/run.md`、输入 manifest、配置 hash、模型 hash、`train_epoch` / `eval_summary` raw result，并继承项目一的 COCO official eval、前后处理、固定输入集与端侧 benchmark。没有这些证据，不能将任何压缩率或速度写为项目 6 的正式结论。
