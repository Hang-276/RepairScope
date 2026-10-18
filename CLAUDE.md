# RepairScope v2 服务器工作约定

当前方法是 **Goal-Sufficient Dynamic Scope Learning**，当前阶段仅推进 **Ours/A4**。A0–A3、所有消融和B不运行。用户要求在明确边界内自主完成工作，并将每项修改即时记录到Markdown，以便追踪问题。

首先阅读 [Ours/SCPS首轮训练记录](docs/OURS_SCPS_STAGE1_RUN.md)。后续用户授权了预测充分性修订与6小时GPU预算内的一次训练；服务器报告30 epochs/253 updates已完成，但dev全invalid_action、业务指标不可测。旧 [Stage1终局](docs/STAGE1_RESULT.md) 和 [GS审查](docs/GS_OBJECTIVE_REVIEW.md) 保留为历史，不能据旧文档再次声称Ours从未启动。当前优先实际源码交接和dev根因修复，不原样重复训练122条。

Ours-only入口仍是`scripts/repairscope/a4_goal_sufficient.sh`；全套main、A0–A3、消融和B不运行。历史Stage1新增环境578、累计918；本次训练报告253优化步、约4分20秒训练时间，该时间不代表全部GPU成本。原6小时GPU总上限不得重置；剩余862环境单位未因本次训练自动获准使用。当前仅整理文档，新增真实费用0。

代码与文档由MacBook提交并推送到默认 `main`（交付分支同步）。服务器保留已有无.git快照和修复，不为读取说明覆盖源码；复用环境，不提交/推送、不重装依赖或影响其他任务。服务器修复输出源码patch和hash，由MacBook后续核验合入。

## 已知输入缺口

服务器SCPS实际源码patch、完整配置和原始运行产物尚未取得，MacBook当前代码不冒充其复现版本。SCPS支持6/8组；phone copy值须核查公开指令/action来源，不从future标签倒填。支持组13–57字段小于B128，不能宣称预算竞争。dev invalid_action须区分真实执行失败与转换bug；没有合法监督的指标保留null。

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

当前以最新运行记录和实际用户授权为准，历史预算、数据、模型、ledger和日志保留，不重写hash。局部可修复问题应自主解决，不把每个失败当整项任务结束；科学指标与权限边界不能跳过。新增采集需要明确总预算；完整Proposal不自动授权新调用。服务器交付实际可转移patch与精简结果，由MacBook核对合入。
