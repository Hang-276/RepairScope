# RepairScope v2 服务器工作约定

当前任务是 **Goal-Sufficient Dynamic Scope Learning**。用户已授权 v2 架构、决策监督、Goal-Relevant Escape 和新版实验协议；旧版仅 A1 vs A3 的阶段安排已被本次任务替代。

首先阅读 [完整服务器交接](docs/CLAUDE_GOAL_SUFFICIENT_EXPERIMENT.md)。方法与 Eval 的完整版本在 [v2 Proposal](docs/RepairScope_v2_Proposal.md)。默认 `main` 为新版方法；旧版独立流程、文档和上游浏览器模板已从当前工作树移除，历史可由 Git 追溯。A0–A3 保留为新版 Evaluation Plan 明确定义的对照，不能当作默认方法。

当前主要执行入口是 `scripts/repairscope/run_goal_sufficient_main.sh`：A0–A4 加四项必要消融，初轮 seed100、8个训练配置各30 epochs、同一共享数据/Actor/模型容量/Scope预算128/H4/执行预算/官方任务集。B 使用独立 `run_goal_sufficient_b.sh`，每策略2000、合计8000约束的是探索采集执行；冻结查询在当前代码中另计，须事先纳入已批准的全局费用上限，不能解释为免费或自动扩额。额外随机种子和跨环境部署仅为后续计划。

代码、proposal 与交接提示词均由 MacBook 本地维护、提交并推送到默认 `main`（交付分支 `feat/goal-sufficient-scope` 同步更新）。服务器 Claude 只从该 GitHub 分支安全 clone 到独立目录，复用原环境，不在服务器提交/推送代码、不安装依赖、不擅自启动/停止模型服务或运行未满足准入条件的训练。已有原仓库及数据/checkpoint 必须保留；实际接手时核对版本和进程所有权，不能影响其他任务。

## 已知输入缺口

服务器修复 `b3b7df93b18ca73ac9f29e4bf69ca989027b89b0` 已纳入新版历史，保留复合目标表达式、compact audit、计费和前缀修复。2026-10-18 16:10 CST 报告中，目标提取器 train/dev 的 full business goal 为0。`complete` 或无正确答案约束的 `answer` 不能代替业务目标。先修复公开 instruction→目标语义→合法观测绑定；可训练决策监督仍为空时不得开始正式v2训练。一次真实业务字段写入已证明Scope/Patch工程链路可通，不能以此声称方法有效。

## 执行边界

- Actor 参数冻结；单一 B-lite 和现有关系/时序/Patch/H4主干保留。不增加 RSI、Memory Agent、额外大模型或 RL Explorer。
- 训练只用官方train；普通dev和显式冻结dev反事实仅用于评估/一次校准。dev评估分支必须设置 `collection.counterfactual_evaluation_only=true`，计划非自适应且来源明确；禁止test分支或测试期更新。
- Predictor/Actor/合法检查只用公开当前状态/schema、任务要求、当前动作和历史。隐藏数据库、未来结果及官方checker只供允许阶段的离线标签/原生最终评分。不得人工逐题修补test目标。
- Unknown从不作negative。新增Goal-Sufficiency必须对Scope选择器有训练信号；无有效可比较候选、关键字段不可检索或没有预算竞争时报告覆盖缺口。复杂目标无可证明依赖或当前evaluator未实现语义时mask，不能擅自放宽定义。
- 同候选/数据/预算比较。不能用少做动作、提前结束、丢弃失败任务或额外免费重试降低损害率。TGC/SGC使用完整官方test_normal 168及原生scenario聚合，目标损害与任务质量共同报告。
- 只做一次必要新版真实工程Smoke（1–2 train任务、2–5更新、合法CF、H4、梯度/显存/成本）。CPU synthetic测试不是真实AppWorld/GPU证据；不加效果Pilot、超参搜索或多轮小样本筛选。
- 安装/接口等义bug可自行修；新增方法、标签语义、测试协议或预算超出本次已授权范围时停止受影响链路，提交具体问题。不得把本次明确批准的v2改造再当成需要审批的未授权变更。
- 运行和修复日志继续放 `/root/autodl-tmp/RepairScope-server-notes` 与独立run目录，保留真实SHA、配置、数据/模型hash、失败成本和恢复边界。无凭据、模型权重、大数据或无关文件进Git。正式运行只允许同配置/同数据exact resume，v1不冒充v2续训。

完整步骤、实际路径、命令、指标口径和当前真实状态以新版交接任务为准。方法是否有效必须由正式结果回答，工程测试不提供效果保证。
