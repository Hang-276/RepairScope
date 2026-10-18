# RepairScope on WMA

本项目直接在 WMA 官方仓库 `dd894640204895b52f050c66cc0b58f08e00f9b8` 上修改。研究与实现定义见 [proposal](REPAIRSCOPE_PROPOSAL.md)。默认路径使用 AppWorld 的公开观测，不安装或启动 WebArena、Playwright、浏览器 Docker。

具体设计与代码的对应关系见 [IMPLEMENTATION_COVERAGE.md](IMPLEMENTATION_COVERAGE.md)，当前本地验证记录见 [LOCAL_VERIFICATION.md](LOCAL_VERIFICATION.md)。

## WMA 中实际复用的代码

| 上游部分 | 当前处理 |
| --- | --- |
| `agent/world_model_agent.py` 的候选解析、频数去重、Top-k、概率归一化 | 抽成 `agent/wma_core.py`，原 WMA 入口、RepairScope Actor 和桥接器共用同一实现 |
| 候选 → 预测 → 评分 → 稳定重排 | 共享 core 的 `predict_and_rank`、`stable_rank`，RepairScope 更换预测与评分回调 |
| WMA 的 `policy_llm.generate`、prompt template、`extract_action` | `agent/repairscope_agent.py` 直接调用，保留 `next_action` 五元组返回契约 |
| 网页的 HTML、URL、浏览器动作和 LLM 状态差分预测 | 从 RepairScope 默认链路移除，由结构化状态、类型化 Patch 与 AppWorld worker 替代 |

原 WebArena 文件保留为上游参照，默认安装和测试不导入其依赖。`agent.__init__` 使用延迟导入。新的模型、训练目标和 AppWorld 数据接口是方法新增部分，不能由上游网页模型直接替代。

## 完整运行链路

```text
公开 API 返回 / schema / 既有历史
  → 类型化字段、实体关系、未知值与创建槽
  → 静态关系 + 学习作用分数 + 不确定性/风险的范围选择
  → 动作条件 Relation Attention
  → 字段级 Causal Temporal Attention
  → Scope、Patch、Escape、Goal 四态预测头
  → WMA 共享候选重排
  → 正常执行 / 合法只读检查 / 修复候选（含不修复）
  → 真实结果与成本记录
```

训练使用 Step、自由递推 Rollout、条件 Boundary、Scope Escape。成对 effect/support 对照使用同一批分支数据。每步只使用起点公开信息和模型递推，不读取真实未来状态；离线数据库差分与目标标签单独存储。只有具有有效审计或四格结果的样本参与对应监督。

数值回归、类别/布尔分布、引用和文本复制、存在性和新增/删除均有实际输出。冻结的确定性文本特征由 `tensorizer.py` 生成，无需本地下载权重；字段与动作投影及 Predictor 参与训练。复制槽之外的新字符串/不可枚举实体保持 unknown；相关梯度边界写入 checkpoint manifest，不声称硬 JSON 执行可微。

默认配置：384维、6头、4层关系块、2层时序块、历史8步、候选256、Scope128、rollout4。AdamW、30 epochs、micro-batch4、累积4、BF16；前10%更新只预热 Step/Rollout。每个配置实际修改模型或损失计算，不只是改变输出名称。

## MacBook 本地

只运行显式 `synthetic` 的工程 fixture，检查数据、前后向、checkpoint/resume、H4硬递推、控制器、反事实和脚本。没有 AppWorld、CUDA、模型服务或外部 API 调用。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
bash scripts/repairscope/local_smoke.sh
.venv/bin/python -m pytest tests/repairscope -q
```

一次 smoke 完成后直接去服务器进行主实验。此处的 fixture 指标不能作为提点；synthetic checkpoint 被禁止用于正式 AppWorld 评测。

## 服务器主实验

完整指令、输入映射及数据准备见 [SERVER_RUNBOOK.md](SERVER_RUNBOOK.md)。主实验不设置额外 pilot 或分阶段有效性 gate。`doctor` 只是可选依赖诊断，不触发采集或训练。

所有脚本支持 `DRY_RUN=1`，只解析配置和命令。`SEEDS="100 101 102"` 为默认三种子；`RUN_ROOT`、`CONFIG`、`TRAIN_DATA`、`EVAL_DATA`、`TASK_FILE`、`REPAIRSCOPE_PYTHON` 可配置。路径支持空格。

| 实验 | 独立启动脚本 |
| --- | --- |
| A0 冻结 Actor | `scripts/repairscope/a0_actor_only.sh` |
| A1 普通 dynamics | `scripts/repairscope/a1_dynamics.sh` |
| A2 成对 effect/support | `scripts/repairscope/a2_paired_effect.sh` |
| A3 完整 RepairScope | `scripts/repairscope/a3_full.sh` |
| A4 去 Boundary | `scripts/repairscope/a4_no_boundary.sh` |
| A5 去 Escape | `scripts/repairscope/a5_no_escape.sh` |
| 固定静态 Scope | `scripts/repairscope/ablate_static_scope.sh` |
| 去学习作用边 | `scripts/repairscope/ablate_no_learned_edges.sh` |
| 去时序块 | `scripts/repairscope/ablate_no_temporal.sh` |
| 统一 Transformer | `scripts/repairscope/ablate_unified.sh` |
| 去多步训练 | `scripts/repairscope/ablate_no_rollout.sh` |
| 去目标辅助损失 | `scripts/repairscope/ablate_no_goal.sh` |
| 不增加反事实探索 | `scripts/repairscope/ablate_no_probing.sh` |
| B 随机探索 | `scripts/repairscope/probe_random.sh` |
| B 普通不确定性 | `scripts/repairscope/probe_uncertainty.sh` |
| B 新颖性/可学习性 | `scripts/repairscope/probe_novelty_learnability.sh` |
| B 风险引导探索 | `scripts/repairscope/probe_risk.sh` |

`run_main_a.sh` 执行 A0–A5；`run_main_b.sh` 执行四种探索；`run_all_ablations.sh` 执行模型/损失消融。探索消融单独使用 `ablate_no_probing.sh`。

主 A 共享同一数据、Actor、候选数、选择器及检查/修复预算。主 B 共享初始 checkpoint、合法候选实验池、损失与真实执行预算；模型前向与时延另列。每次分支包括完整前缀、条件、验证查询和失败调用的真实成本，预算耗尽即停止。

默认 Actor 候选缓存按公开输入、模型与采样配置共同索引，跨方法复用相同输入的候选结果；锁和原子替换避免独立消融脚本并行写坏缓存。缓存命中与否都保留统一的策略查询预算口径。

## 数据与结果

`repairscope/fixtures/` 包含 schema、公开观测映射、目标谓词和两/四分支计划示例。真实映射根据服务器安装的 AppWorld API 返回配置一次，不把隐藏数据库读成在线观测。`collection.extract_prefix()` 从采集记录提取动作、输出与指纹，避免手工伪造反事实前缀。

训练保存 resolved config、数据 hash、schema、优化器、学习率状态、RNG 和模型；同一数据/config 支持精确续训。CLI 的 `--resume PATH` 从已有 checkpoint 接续，不静默重置 optimizer。正式测试不更新 Predictor，不回滚环境搜索。

TGC/SGC 仅由官方 tracker 原生规则与完整任务覆盖产生；fixture 保持 null。其他输出包括范围预算覆盖、越界率、Brier/ECE、四态 F1、类型化字段误差、受保护目标损害、修复收益/新损害、成本、分组 bootstrap，以及 H=1/2/4 固定起点硬递推指标。服务器真实运行结果与本地工程测试分开保存。
