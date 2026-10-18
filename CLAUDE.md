# RepairScope v2 服务器工作约定

当前方法是 **Goal-Sufficient Dynamic Scope Learning**，当前阶段仅推进 **Ours/A4**。A0–A3、所有消融和B不运行。用户要求在明确边界内自主完成工作，并将每项修改即时记录到Markdown，以便追踪问题。

首先阅读 [续接后的结论与当前交付任务](docs/V2_POST_CONTINUATION_DECISION.md)，再查 [Ours-only自主续接说明](docs/CLAUDE_OURS_ONLY_CONTINUATION.md)中的持续约束。一次离线续接已结束，当前只整理实际补丁、完整正式数据计划和收据，不重复诊断或启动正式实验。方法与完整Eval见 [v2 Proposal](docs/RepairScope_v2_Proposal.md)，旧 [服务器交接](docs/CLAUDE_GOAL_SUFFICIENT_EXPERIMENT.md) 仅补充接口、路径与历史背景。默认 `main` 是新版方法；历史由Git追溯。

正式Ours唯一入口是 `scripts/repairscope/a4_goal_sufficient.sh`，准入、正式数据与明确预算满足后才运行。禁止运行 `run_goal_sufficient_main.sh`（会自动运行其他组）及 `run_goal_sufficient_b.sh`。派生字段修复与一次离线监督判定已完成；前轮3任务上限已到，不自动重置300单位预算，也不追加相同Smoke。当前新增环境/Actor/优化步均为0。

代码与文档由MacBook提交并推送到默认 `main`（交付分支同步）。服务器保留已有无.git快照和修复，不为读取说明覆盖源码；复用环境，不提交/推送、不重装依赖或影响其他任务。服务器修复输出源码patch和hash，由MacBook后续核验合入。

## 已知输入缺口

用户最新服务器报告：基于0ce975d快照已有F1–F3修复、95项测试与H4硬递推；follow等义标签已恢复，follow 6组与phone 4组在当前权重下oracle GS饱和，真实Predictor仍有错误。正式train/dev及test目标映射未冻结，正式预算未明确。主结论为FORMAL_INPUTS_NOT_READY；不能从10组外推全局无信号，也不能把Smoke权重的预测错误当作正式训练失效。详细报告见当前交付任务，实际补丁尚未合入MacBook。

## 执行边界

- Actor 参数冻结；单一 B-lite 和现有关系/时序/Patch/H4主干保留。不增加 RSI、Memory Agent、额外大模型或 RL Explorer。
- 训练只用官方train；普通dev和显式冻结dev反事实仅用于评估/一次校准。dev评估分支必须设置 `collection.counterfactual_evaluation_only=true`，计划非自适应且来源明确；禁止test分支或测试期更新。
- Predictor/Actor/合法检查只用公开当前状态/schema、任务要求、当前动作和历史。隐藏数据库、未来结果及官方checker只供允许阶段的离线标签/原生最终评分。不得人工逐题修补test目标。
- Unknown从不作negative。新增Goal-Sufficiency必须对Scope选择器有训练信号；无有效可比较候选、关键字段不可检索或没有预算竞争时报告覆盖缺口。复杂目标无可证明依赖或当前evaluator未实现语义时mask，不能擅自放宽定义。
- 同候选/数据/预算比较。不能用少做动作、提前结束、丢弃失败任务或额外免费重试降低损害率。TGC/SGC使用完整官方test_normal 168及原生scenario聚合，目标损害与任务质量共同报告。
- 已有Smoke、H4和测试收据复用；只对实际新修复做必要局部回归。无有效数据或预算时不追加任务扫描、GPU小试或效果Pilot；交付续接说明定义的终局结论和可审阅计划。
- 每项代码、配置、目标映射、审计或数据转换修改都即时追加notes下 `CHANGELOG_V2.md`，按续接说明第7节记录原因、文件/函数、前后hash、patch、验证和回退。MacBook每次相关提交同步更新 [项目变更记录](docs/CHANGELOG.md)。
- 安装/接口等义bug可自行修；新增方法、标签语义、测试协议或预算超出本次已授权范围时停止受影响链路，提交具体问题。不得把本次明确批准的v2改造再当成需要审批的未授权变更。
- 运行和修复日志继续放 `/root/autodl-tmp/RepairScope-server-notes` 与独立run目录，保留真实SHA、配置、数据/模型hash、失败成本和恢复边界。无凭据、模型权重、大数据或无关文件进Git。正式运行只允许同配置/同数据exact resume，v1不冒充v2续训。

当前任务以续接后的结论与交付说明为准，Ours-only续接说明中的持续约束保留。完整Proposal描述研究设计，不自动授权全部实验；工程测试不提供方法效果保证。
