# RepairScope 设计与实现对应表

以 [proposal](REPAIRSCOPE_PROPOSAL.md) 为方法定义，以实际代码、配置和执行收据为工程依据。首版借鉴 JEPA 的预测式学习思想，直接监督结构化状态，不增加 proposal 未要求的 EMA teacher。

当前阶段仅授权 `run_primary_experiment.sh` 的 A1/A3、种子100主对照。下表说明完整实现能力，其余实验入口保留待后续决定，不自动运行。

| 设计部分 | 实现位置 | 已完成的本地工程验证 |
| --- | --- | --- |
| WMA 冻结策略、候选聚合、预测后重排 | `agent/wma_core.py`、`agent/repairscope_agent.py`、`repairscope/actor.py` | 原 WMA 与新入口共用函数；Top-k、重复候选、稳定排序、桥接返回契约 |
| 公开字段、关系、未知值、类型与复制槽 | `repairscope/schema.py`、`tensorizer.py` | 类型校验、未来字段隔离、部分审计未知掩码、nullable 与创建/删除 |
| Dynamic Scope 与 B-lite | `repairscope/model.py` | 关系块、字段时序块、学习边、范围预算、风险/不确定性扩展及结构消融 |
| 类型化 State Patch 与 Goal 四态 | `repairscope/model.py`、`losses.py` | 数值、类别、布尔、引用/文本复制、存在性输出；00/01/10/11 与 no-goal 行为 |
| Step + H4 自由递推 | `repairscope/training.py`、`losses.py` | 软状态递推梯度、有效窗口、因果历史、有限损失与参数更新 |
| 条件 Boundary、Escape、成对 effect/support | `repairscope/losses.py`、`collection.py` | 同前缀与有效条件检查；两分支/四格配对、范围外变更与审计掩码 |
| 预防、合法检查、修复与 noop | `repairscope/control.py` | 共享预测器、公开目标保护、只读 API 约束、失败计费与硬预算 |
| 主动反事实与持续更新 | `repairscope/probing.py` | 同候选池四种策略、MC 分歧、合法四格、经验回放、冻结查询、分支成本上限 |
| 冻结评测与指标 | `evaluation.py`、`prediction_eval.py`、`metrics.py` | H=1/2/4 只用预测中间状态；官方 tracker 规则接口与不完整覆盖掩码 |
| 一次 dev 阈值选择 | `repairscope/calibration.py` | 有效审计域、固定检查率上限、checkpoint hash 绑定及测试标签拒绝 |
| 保存、精确续训与来源记录 | `training.py`、`provenance.py` | 模型/优化器/学习率/RNG/数据/config 保存和恢复；来源与划分检查 |
| 主 A、主 B 和各项消融 | `configs/repairscope/`、`scripts/repairscope/` | 实际脚本 dry-run、空数组/空格路径、多个种子、共同起点与独立结果目录 |
| AppWorld 接入与隔离 | `environment.py`、`appworld_worker.py` | worker 协议与隔离逻辑测试；真实 AppWorld 执行留待服务器 |

## 明确的实现边界

- 预测输入限于公开观测、Schema、合法检索和既有历史。隐藏数据库仅作为训练侧离线审计，不进入 Actor 或预测器输入。
- 可微 rollout 使用起点公开字段与可用复制槽；候选检索、硬 Top-k、硬 JSON 解码及无绑定的新字符串/实体创建不宣称可微。未知值保留 unknown。
- AppWorld 的公开 API 返回到字段 Schema、公开目标谓词及合法条件计划需要服务器真实数据配置；仓库提供格式、加载器和示例，不把 fixture 映射冒充真实环境配置。
- 正式模型默认 BF16；本地只验证 CPU FP32。CUDA/BF16、显存、模型服务、AppWorld 原生执行和真实提点尚未测量。
- 本地验证汇总见 [LOCAL_VERIFICATION.md](LOCAL_VERIFICATION.md)。不把 synthetic 指标用于算法效果结论。
