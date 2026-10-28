# 原生 Phi 的框架模型适配

## 接口与兼容策略

`OTO(model, pruning_mode="joint")` 自动识别 Transformers 原生 `PhiForCausalLM`。
新增参数追加在旧构造参数之后；`model_adapter="graph"` 可显式选择原来的
Graph/ONNX 路径。其他模型默认仍进入原图路径。匹配不使用模糊类名，也不
把原生 `PhiAttention` 注册成旧 `PhiMHA`，因此旧算子的属性和分组语义保持独立。
框架导入和非 Phi 使用不要求安装 Transformers；原生分支延迟导入依赖。

原生分支的依赖发现使用显式模型结构合同，不执行 ONNX 追踪；选择 ONNX 输出时才对固定子网执行导出追踪。输入分词、数据集、训练循环
和精度审计仍由实验代码负责。这一适配经过 Transformers 4.53.0、PyTorch
2.7.1+cu128 验证，不意味着自动支持其他 Transformers 版本或任意 Phi 变体。

## 三项职责的归属

| 职责 | 框架位置 | OTO 入口 |
| --- | --- | --- |
| 参数依赖与逻辑分组 | model_adapters/phi.py、operation/native_phi.py | get_param_groups / hessocric |
| 物理裁剪与安全结构包 | 同上 | construct_subnet |
| 异构逐层结构重建 | model_adapters/phi.py | load_subnet |

`NativePhiAttentionOTO` 同时生成优化器索引和解释导出布局。FFN 联动 fc1 行、bias
和 fc2 列；注意力联动 Q/K/V 行、bias 和 dense 列，残差 hidden_size 固定。
支持 joint、ffn、num_head、head_dim，候选之外的参数仍保留训练资格。
head_dim 采用等宽双维逻辑组，旋转区维度按 rotate_half 配对；保留原始 RoPE
频率、坐标与 attention scaling，不按压缩后的 head_dim 重新定义位置编码。

五准则、α 更新、历史重融合和 CRIC 时间流程继续使用公共优化器。本次不修改
optimizer、graph、dependency_graph、旧 operator.py 或通用 pruning_compression.py。
测试与训练代码直接导入框架模块，不再保留独立结构兼容文件。

## 导出与加载

```python
from only_train_once import OTO

oto = OTO(model=model, pruning_mode="num_head")
optimizer = oto.hessocric(lr=1e-4, first_momentum=0.,
    target_group_sparsity=.1, start_cric_step=3,
    sampling_steps=2, max_cycle_period=2, hybrid_training_steps=8)
# 按训练代码更新 loss/梯度及周期边界 α，完成全部稀疏化。
plan = oto.collect_pruning_plan()
oto.construct_subnet(out_dir="runs/example", tokenizer=tokenizer,
                     ckpt_format="safetensors")
model = OTO.load_subnet(oto.compressed_model_path, device="cuda:0")
```

原生输出固定子目录 compressed_model、sparse_model；显式 compressed_model_dir 和
full_group_sparse_model_dir 指定这两个子目录各自的父目录。默认保存稀疏模型，
已有检查点时可设置 save_full_group_sparse_model=False。目录存在时拒绝覆盖。
可以显式传 pruning_plan 从已保存稀疏检查点导出，不需重跑训练；框架先验证全部
层的索引和删除位置精确归零，检查失败不会开始物理裁剪。无显式 plan 时仅支持
从已完成的 HESSOCRIC 自动冻结合同，并核对优化器的参数归属。

原生格式显式选择：

| ckpt_format | 压缩产物 | 加载与用途 |
| --- | --- | --- |
| torch / pt（框架默认） | model.pt + config.json + architecture.json | 纯 state_dict；OTO.load_subnet 重建可训练模型 |
| safetensors | model.safetensors + config.json + architecture.json | OTO.load_subnet 重建可训练模型 |
| onnx | model.onnx + 必需的外置权重 + config.json + architecture.json | OTO.load_subnet(device="cpu") 返回 ONNX Runtime 推理接口 |

.pt 使用 weights_only=True 加载，不 pickle 整个模型对象。旧版未声明格式的
safetensors 结构包继续兼容。tokenizer 可随包保存。ONNX 支持动态 batch/sequence、
input_ids 与 attention_mask 输入和 logits 输出，不提供 KV cache、反向传播或 LoRA 微调。
超过 protobuf 2GB 限制时必须一并保存外置权重文件，不可只复制 .onnx 主文件。
当前 ONNX 加载器禁用 ORT 图融合并开启确定性计算；默认高级融合曾在真实 Phi-2
上超过预设数值门槛，失败记录保留。它是数值验证基线，不宣称推理速度最优。
权重文件格式不决定 FP32/FP16；当前训练使用 FP32，序列化格式分别选择。
普通 AutoModel.from_pretrained 不会解释逐层异构结构清单，训练格式仍需 OTO.load_subnet。
原图分支的 torch/ONNX/HF 行为保持不变，pt 是 torch 的额外别名；旧算子未替换。

结构包格式版本 1、2 均可加载。loader 在 meta 上建立模型、应用结构合同、严格加载
权重并重建非持久 RoPE buffer，随后移到目标设备。旧实验的安全结构包无需迁移。
导出在原模型上更换 Parameter，旧优化器不再适用；同一 OTO 重复物理导出会报错。

剪枝入口 prune_phi2_hesso_cric.py 的 export_run 负责调用 construct_subnet 并在临时目录保存结果，
通过精度审计后发布最终 compressed_model。物理索引操作已统一由框架执行。已有检查点可用 `--export-only --output RUN_DIR` 重试导出，无需数据集参数和重新训练。
原生 FP32 稀疏/压缩误差和严格 FP64 结构等价门槛分开记录，独立重载再对齐同一
压缩结构 FP32 参考。功能通过不表示下游准确率或吞吐已达标。

## 明确边界

- 当前支持原生 Phi MHA、普通带 bias 的 Linear、eager attention；不支持 GQA、量化投影和 LoRA 合并。
- head_dim 要求 qk_layernorm=False；保留完整双维组且每层至少保留一组。
- 已压缩模型可推理、生成和微调；再次结构剪枝尚未维护复合原始坐标，创建新 OTO 时明确拒绝。
- 可在创建优化器之前按参数名保护完整依赖组。图节点 ID、ONNX 可视化和图 FLOPs 接口不用于原生适配，调用时明确报错。
- 不修改全局 CUDA、驱动、安装包或他人的 Conda 环境。

## 回归命令

```bash
CUDA_VISIBLE_DEVICES='' python test/pretrain_phi2/test_phi2_contract.py
CUDA_VISIBLE_DEVICES='' python test/pretrain_phi2/test_phi2_framework.py
CUDA_VISIBLE_DEVICES='' python test/hesso_cric/test_adamcf_cric.py
CUDA_VISIBLE_DEVICES='' python test/pretrain_yolo11_coco/test_segmented_concat_contract.py --output /path/to/new/evidence
```

另在实际 Phi-2 权重上验证三种模式的 OTO 工厂训练、construct_subnet、严格结构
等价、独立进程 load_subnet、实际 KV cache、缓存续写、生成和反向传播。结果及
运行路径记录在部署验收报告，不将测试运行产物提交到源代码仓库。

## 注释、日志与实验协议

剪枝入口会同步输出终端、UTF-8 run.log 和 events.jsonl；每步统计另存 training_trace.jsonl。
阶段日志包括数据角色/数量、token 长度、数据和源码摘要、FP32 与可训练参数量、LoRA 参数量、
候选预算、CRIC 阶段、学习率、梯度、α、剪除组数、显存、NLL、结构审计与保存路径。
失败保存 traceback 和 failure.json；这些信息不替代尚未实施的七任务评测或 LoRA 恢复。
预处理单独保存 prepare.log，先过滤空响应再固定种子选样，原始 row_id 与旧冻结数据保留。
详细数据/论文审查见 phi2-experiment-protocol-audit-20261028.md。
