# HESSO/CRIC 公共 α 策略与 Phi-2 数据一致性审查

> 本文为 2026-10-28 的历史审查快照。2026-10-30 起 CRIC 默认改为样本百分比池，
> HESSO 保留原渐增取批；下文“两者相同取批策略”只适用于 CRIC 的 legacy 模式。
> 当前用法见 [公共 CRIC 说明](hessocric-adamcf.md)。

日期：2026-10-28；修改前基线 ec66994be0616000af96de8179130d83a835703e。
本次没有启动正式 Phi-2 剪枝/恢复/七任务实验，没有下载或替换正式训练语料。

## 1. 职责修正

上次将通用 AlphaBatchStream 放在 Phi 目录，且脚本重复计算渐增微批规则，职责划分不合理。现已删除 phi2_alpha.py：

- optimizer/alpha_data.py：AlphaBatch、AlphaEvaluationPool、取批数量、列表游标/loader 遍历、设备传输与覆盖记录。
- optimizer/alpha_scheduler.py：HESSO 调度器、共用 advance_alpha 数值内核、共用边界置信统计。
- optimizer/cric_alpha.py：CRIC 特有的触发、预算与安全评估协议，调用上述共同内核。
- Phi 的 phi2_data.py：仅将任务样本 collate 为 AlphaBatch，提供原始行号和监督 token 数。
- Phi 剪枝循环：仅传入 pool、调用 scheduler.update、读取 last_eval_stats，不计算 α 取批策略。

任意模型都可以使用同一 API，无需引用 Phi 文件：

```python
from only_train_once.optimizer import AlphaEvaluationPool
pool = AlphaEvaluationPool(alpha_loader)
if optimizer.will_enter_new_period_on_next_step():
    optimizer.alpha_scheduler.update(model, criterion, pool)
```

模型负责提供适当的 loader/collate/criterion、明确数据角色和切分，这是任务数据接口，不属于 α 算法。Hugging Face 输出带 loss 时可用 criterion=None。

## 2. “相同策略”的准确含义

| 部分 | 当前关系 |
| --- | --- |
| 五准则融合后的 α 数值更新 | 共用同一 advance_alpha 实现：有限差分梯度后的 EMA、动量、L1 信赖域、非负单纯形归一化 |
| 微批增长、缓存轮换、loader 持续遍历 | 共用同一选择函数，正/负扰动固定同一批次，成功更新后记录覆盖 |
| 边界置信门限 | 共用 HESSO 原来的样本标准差定义；此前 CRIC 用总体标准差，现已消除该差异 |
| α 更新时间 | HESSO 按剪枝预算周期；CRIC 按原 CRIC 周期边界，周期内冻结。保留用户已确认的时间流程 |
| 评估候选预算 | HESSO 使用当前剪枝预算；CRIC 使用其冗余识别预算，属于算法本身差异 |
| 历史评分 | CRIC 保存分项历史，最终按最终 α 重融合；不替换成 HESSO 的分期预算流程 |
| 评估兼容与异常恢复 | CRIC 保留其原有 RNG/模式/buffer/参数回滚协议；没有为了代码合并删去旧模型支持 |

因此可以说两者共用 α 更新和取批策略，不能说整个 HESSO 与 CRIC 训练过程相同，也不能说既有所有实验的数据使用配置相同。
旧 HESSO 的 prepare_alpha_eval_batches 先随机选子集，再缓存前 eval_batch_limit 个微批；旧脚本继续传固定列表，没有被自动改成流式数据源。新接口支持两个优化器都使用 AlphaEvaluationPool；相同 source/seed/配置/评估轮次下才保证相同样本与数值。
本次保留 HESSO 的原算术顺序。CRIC 去掉重复数值实现，使用相同累加顺序及标准差定义；该修正可能影响极接近置信阈值的 CRIC 更新判定，但不改变触发时间或预算流程。

## 3. 实证验证与兼容边界

65 项 CPU 测试通过：公共 CRIC/LoRA/新 α 测试 29、Phi 两阶段 5、结构 14、集成 17。
新公共测试不导入 Phi：相同标量模型、数据池和配置下，HESSO/CRIC 的 α、动量、梯度 EMA、样本记录逐位一致；覆盖 EMA 关闭、动量启用、信赖域限幅、渐增微批、随机遍历、跳过不消耗池、同批差分及 CRIC 异常回滚。

此外保存了修改前 alpha_scheduler.py 的逐字节快照，作为独立 oracle：

- 三组 α 配置各运行 5 次更新，新旧 HESSO 的 α/m/g_ema 与缓存游标逐位一致。
- 同一小型网络分别运行 SGD、AdamW 各 16 步，新旧 HESSO 的损失序列、模型权重、α 和最终剪枝索引逐位一致。
- Phi 原六个小型完整管线用例继续通过，覆盖 FFN、整头、头内维度、联合模式和三种保存格式。

这证明上述有限测试范围的兼容性，不宣称全部模型与全部配置都已完成新实验。hesso.py、HESSOCRIC 主类、其他模型脚本和算子均未改写；共享调度器的修改已通过旧快照对照。
旧优化器 checkpoint 字段未重命名，CRIC checkpoint 恢复测试继续通过。外部 DataLoader 的随机状态/位置仍属于调用方训练检查点范畴，不声称新增了完整数据流断点恢复。

## 4. 数据一致性必须先解决

HESSO-CRIC §4.6 直接说明其 Phi-2 实验设置参考 LoRAShear，但没有公开 Phi-2 的完整语料/样本清单。[HESSO-CRIC](https://arxiv.org/html/2409.09085v2#S4.SS6)
LoRAShear v2 §4.1 明确列出四种英文预训练语料，以及 Alpaca 指令数据；§3.5 的知识恢复先使用预训练语料，再使用指令数据。[LoRAShear](https://arxiv.org/html/2310.18356v2#S4.SS1)

| 来源 | 论文明确的信息 | 当前本地状态及缺口 |
| --- | --- | --- |
| OpenWebText | 作为英文 CommonCrawl/C4 替代 | 当前共享清单未发现；原作者实际版本/样本未知 |
| Wikipedia | 2022 年处理后的 dump | 当前未发现；具体月份、清洗版本未知，WikiText 不能替代 |
| Gutenberg | Gerlach & Font-Clos 标准语料 | 当前未发现；归档版本/书目/样本未知 |
| BookCorpus | Zhu et al. 2015 | 当前未发现；不能直接换成 BookCorpusOpen |
| Alpaca | 约 52K 指令及回答 | 当前有原始 52002 条；功能预处理排除 28 条空响应，作者是否同样过滤未知 |

重要证据边界：LoRAShear v2 报告的是 LLaMAv1 实验，不能把其语料全集、处理或动态抽样直接宣称为 HESSO-CRIC 的 Phi-2 精确样本。
公开预处理只明确提到清理异常字符、保留超过 64 tokens 的段落；没有完整清洗实现和具体 tokenizer，因此不臆造这些细节。
OTO 官方 README 仍把完整 LoRAShear 管线列为待发布事项；microsoft/lorashear 仓库本次访问返回 404。[固定版本 README](https://github.com/microsoft/only_train_once/blob/4517648c55f09e8743850427031d559767dec908/README.md)

当前 Alpaca-only 是功能验证，不能作为“实验数据已经一致”的证据。七项评测数据齐全也不代表训练和恢复语料已对齐。
test/pretrain_phi2/corpus_alignment.json 已记录正式数据未对齐、来源证据、现有文件摘要和缺失信息；evaluation_protocol.json 同步标为 formal_corpus_alignment_complete=false。

## 5. 接下来应如何冻结样本

先获得/核实作者 Phi-2 的数据清单或补充配置，确认上述语料在剪枝和恢复中的真实用途，再确定各源版本、样本量、混合比例、清洗/去重/段落规则与 tokenizer。
随后生成不可变的源文档/段落 ID、内容 SHA256、训练/α/内部验证划分和采样顺序清单。两套优化器与各剪枝模式共用这份清单，只改变方法变量。
若作者没有公布样本索引，最多能做到公开数据来源/版本级对齐并冻结我们自己的样本；论文中应明示，不能声称逐条样本完全相同。不能因暂时拿不到某源就静默用相似语料替代。

已通过异步问题请求作者配置/样本清单链接或本地路径。缺少证据不影响本次公共 α 代码修正，但正式数据构建与正式实验不依赖猜测继续推进。
本次没有动旧 proof、原始数据、权重、已完成运行或任何 Conda 依赖。

## 6. 交付证据

服务器审查根目录 `/zeng/GY/shared_alpha_data_audit_20261029/`：core/oracle/training/contract/framework 日志、status.json、hesso_legacy_parity.json、六个 Phi 用例日志、部署前备份和源码摘要。
