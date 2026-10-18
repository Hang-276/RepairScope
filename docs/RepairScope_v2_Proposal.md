# RepairScope v2：Goal-Sufficient Dynamic Scope Learning for Reliable Long-Horizon Agents

版本：2.0（2026-10-18）
状态：实施与实验协议；研究效果尚无真实 AppWorld 结果。

## 摘要

长程 Agent 的一次动作可能推进当前目标，也可能破坏此前已满足且仍需保持的目标。完整预测全部状态变化成本高，而仅按变化概率分配预算会忽略低频但高损害的字段。RepairScope v2 研究：在固定预测与检查预算下，Agent 能否学习一组足以支持可靠目标保护决策的最小状态字段，并减少相对于充分离线信息参考决策的损失。

v2 保留冻结 Actor、B-lite Predictor、结构化状态、动作条件关系块、字段级时序注意力、类型化 State Patch、多步递推、合法反事实数据和 Goal Evaluator。核心改动是让 Dynamic Scope 直接受 Goal-Sufficiency 决策蒸馏监督。训练使用合法同前缀反事实结果、完整离线差分与经核验的目标满足标签；推理只用公开当前观测、公开目标、当前动作和历史。固定 Top-k 前向保持 Scope Budget，预算保持的直通松弛梯度使新增损失进入实际 Scope 选择器。

本方案将条件化 Boundary、State Escape、Goal-Relevant Escape 和决策蒸馏分别定义。未知、未审计、非法执行、没有有效业务目标或无法可靠归因的样本均按明确规则 Mask。实验以 AppWorld 为主；其他 GUI/API benchmark 先作为候选，只有在状态恢复、反事实执行和目标损害真值均通过预检后才进入扩展评估。

## 1. 研究问题与可检验主张

### 1.1 研究问题

**Can an agent learn the minimal action-effect scope sufficient for reliable goal-preservation decisions under a fixed scope and inspection budget?**

给定合法候选字段集合 \(C_t\)、动作 \(a_t\)、公开目标 \(G_t\) 和固定预算 \(B\)，学习作用范围 \(M_t\subseteq C_t\)，使受限信息下的动作选择接近使用合法离线充分信息的参考选择，同时保持任务完成质量。

旧版重点是降低 Scope 外真实变化的漏检。v2 的主优化对象是目标保护决策 Regret。状态变化覆盖仍保留为辅助任务和报告指标，不能替代新的决策目标。

### 1.2 假设与所需证据

| 假设 | 可证伪预测 | 必须报告的证据 |
|---|---|---|
| H1：决策充分性 | 在相同 \(B\) 下，A4 相较 A1–A3 降低 held-out Decision Regret。 | 可比较反事实组数、regret 分母与覆盖率、置信区间。 |
| H2：目标相关范围 | 在相同 \(B\) 下，A4 提高 Goal-Relevant Recall@B，且不靠扩大范围或减少动作获益。 | 固定预算、Scope 大小、目标损害、执行/检查成本。 |
| H3：边界与逃逸机制互补 | 去掉 Boundary 或 Goal-Relevant Escape 会在对应条件化或目标相关指标上产生可测变化。 | 预先定义的消融差值与有效标签覆盖。 |
| H4：组成泛化 | 在预先冻结的未见条件/动作组合上，A4 的决策质量优于相同容量对照。 | 训练/评测组合 Manifest、无重叠证明、覆盖率。 |

以上均为待检验假设，不是现有结果。局部工程 smoke、合成数据、离线拟合或单个任务不能支持方法有效的结论。

## 2. 任务定义与信息边界

令 \(s_t\) 表示合法公开观测、公开 API/schema 和历史所构造的结构化状态；未知字段保留 unknown。冻结 Actor 给出相同的一组合法候选动作。Predictor 根据 \((s_t,a_t,h_t,G_t)\) 选择范围并预测类型化补丁：

\[
F_\theta(s_t,a_t,h_t,G_t)\to (M_t,\widehat{\Delta s}_{t+1},\hat r_t),
\qquad |M_t|\le B.
\]

Scope 决定字段级预测和检查所用的可见预算。目标通过公开谓词或预先冻结的合法目标解析器评价。若目标不可判定、保护区间未知、目标依赖不能绑定到可审计字段，样本不得被解释为 Negative。

离线训练标签可以使用合法反事实分支、完整状态差分及经验证的目标满足结果。隐藏数据库和官方 checker 只能用于允许的隔离审计或最终官方评分，不得进入在线状态、Scope 输入、候选筛选或测试期更新。训练、dev、test 的源任务与全部兄弟分支必须按任务/组合整体分组。

## 3. 方法

### 3.1 保留的预测器与 Dynamic Scope

保留原 B-lite 主干：结构化字段 token、静态关系先验、学习得到的动作条件关系、Temporal Attention、typed Patch 和共享参数硬 H=4 递推。Actor 冻结。Predictor 不引入新大型模型、额外 Memory Agent、独立 RL Explorer 或 RSI 组件。

候选字段由当前公开状态、schema、静态关系和合法检索产生。Scope 排序使用静态先验、学习到的决策相关打分、不确定性、已知性和遗漏风险；Goal-Sufficiency 模式直接学习 Scope 排序分数，不再把变化概率简单乘以 Goal Weight。静态关系是先验，不是封闭硬边界。固定预算内保留代码定义的覆盖预留；所有策略使用同一 Scope Budget 和候选上限。

**硬前向与训练梯度。** 前向严格执行 Top-k 和固定覆盖预留；不以软阈值扩大实际范围。反向对非预留可选字段使用总质量固定为 \(k\) 的 sigmoid 松弛，并以直通估计器把梯度传回同一 Scope gate。该估计有偏，提供可训练信号但不构成离散最优性保证。Padding、预留字段和预算内全部选中等无竞争情形按实现屏蔽梯度。

### 3.2 目标效用与合法反事实决策组

每个动作分支 \(a\) 的充分信息离线效用为：

\[
U(a)=\sum_{g\in G} w_g\left[\alpha\,(y_g^{after}-y_g^{before})
-\beta\,\mathbf 1(y_g^{before}=1\land g\text{ 当前受保护}\land y_g^{after}=0)\right]
-\gamma\,c(a).
\]

其中 \(w_g\) 是公开目标权重，\(y_g\in\{0,1,\text{unknown}\}\)，\(c(a)>0\) 是记录的动作成本；\(\alpha,\beta,\gamma\) 与部署决策器使用相同的已冻结配置。只有目标前后均能从合法完整审计和公开目标语义重现、执行明确有效、动作成本有效的分支才参与监督。unknown 不填零。

候选来自同一合法前缀、同一条件单元中的 action/reference 配对。只有效用偏好可识别且不为平局的组进入训练；缺失分支、重复单元、前缀不匹配、目标不匹配、非法动作、无法绑定依赖或标签冲突均 Mask。一个组必须提供“信息范围不同会改变动作偏好”的可学习机会；预审统计候选可检索性、预算竞争、固定预留和组级非零监督。

### 3.3 Goal-Sufficiency Loss

充分信息教师分布为 \(q(a)=\mathrm{softmax}(U(a)/\tau)\)。每个动作的受限效用 \(\widehat U(a;M)\) 只根据 Scope 内字段可见的预测补丁构造。蒸馏损失为：

\[
\mathcal L_{GS}=\mathrm{KL}\left(q(a)\;\middle\|\;
\mathrm{softmax}(\widehat U(a;M_\theta)/\tau)\right).
\]

该损失直接作用于 Scope gate。原子谓词可按字段贡献精确计算；复合谓词在“Scope 内采用离线真实补丁、Scope 外保留前态”的混合状态上调用现有 Goal Evaluator，精确得到受限效用。代码以当前硬 Scope 的精确效用为前向值，以单字段有限差分构造 gate 的直通梯度。若受限复合目标无法判定、有限差分未知或 evaluator 未执行相应语义，整个候选组 Mask，并记录原因。

这一定义把“改变动作选择的最小信息范围”作为训练目标。它不把单独辅助头的准确率当作 Scope 学习，也不声称模型恢复了真实因果图。贡献归属由本方案中实际基线、消融与 held-out 决策指标决定。

### 3.4 Boundary、State Escape 与 Goal-Relevant Escape

**Boundary Loss** 学习动作影响边界随条件如何变化。合法四格数据固定同一前缀：\((c_0,a_r),(c_0,a),(c_1,a_r),(c_1,a)\)。逐字段目标 \(y_j^{c}\) 是同一条件下 action 与 reference 后状态的语义差分；损失比较模型和数据中 \(c_1-c_0\) 的支持变化。只有四分支可对齐、前缀/条件合法且字段审计完整时计算。Boundary 不判断变化是否影响目标决策。

**State Escape Loss** 保留旧版监督：Scope 外出现真实字段变化时为正；只有审计完整覆盖声明字段域时，无变化才能作为负例。它学一般状态覆盖，不把未观察字段视为不变。

**Goal-Relevant Escape Loss** 监督 Scope 外遗漏且有可靠证据会改变目标保护判断的信息。当前实现基于满足中的受保护业务目标在合法执行后变为不满足，并将该变化可靠归因到完整审计字段。一般状态变化不自动成为目标损害；joint attribution、目标 unknown、缺失审计域和非业务提交完成标签都要 Mask。若受保护判断没有变化，才可在标签完整的有效样本中作为负例。

三者联合时：Boundary 学条件化作用支持；State Escape 学任何可信范围外变化；Goal-Relevant Escape 学有证据的目标保护遗漏；Goal-Sufficiency Loss 学固定预算下哪些信息改变候选动作偏好。不得互相替代。

### 3.5 总损失、递推和探索

\[
\mathcal L=\mathcal L_{step}+\lambda_r\mathcal L_{roll}
+\lambda_b\mathcal L_{boundary}+\lambda_e\mathcal L_{escape}
+\lambda_g\mathcal L_{GS}+\lambda_{ge}\mathcal L_{goalEscape}.
\]

Step/Patch 预测真实单步状态差分，Rollout 从合法真实起点出发使用模型自身预测状态递推至 H=4。训练标签只用于 loss，不进入推理状态。硬预测以同一 Scope Budget、字段索引和动作候选运行；记录软训练与硬递推的边界。

沿用现有 Counterfactual Exploration 入口。采样分数把当前 Scope 的目标保护风险、候选证据降低决策错误的潜力和完整反事实成本纳入轻量排序，并保留固定比例随机探索。执行成本包含前缀重放、条件动作、只读证明、目标动作、失败调用、查询和续跑。不得增加 Explorer 模型或独立 RL 流程；已有 novelty-learnability 仅称启发式，不宣称复现 OnlineWM。

## 4. 数据与训练协议

1. 训练仅使用固定 AppWorld `train`。先保存自然轨迹和合法反事实计划，再进行一次共享采集；A0–A4 与消融复用同一数据、候选、Actor 输出和有效审计标签。
2. AppWorld `dev` 只作冻结预测评估、校准和显式 evaluation-only 反事实评估。dev 分支不得进入训练或回放；测试期禁止反事实搜索、更新或阈值回选。
3. 公开目标提取器只在 train/dev 开发并冻结。目标语义保留指令依据、实体、量词、数值/时间限制和保护生效区间；实体绑定只用公开 API 查询与返回。不能将 `complete` 或无正确目标值的 `answer` 当作业务成功。
4. 开始正式训练前输出 `goal_supervision_audit.json`：decision eligible/informative/trainable groups、目标与标签 Mask 原因、关键字段可检索率、固定预留率、预算竞争率。若无可训练组，阻止 A4 正式训练并报告数据/语义缺口；不得调低 Scope Budget、扩大字段候选、放宽 Mask 或强制生成写入来制造信号。
5. 冻结 Actor、候选动作、模型宽度/容量、H=4、Scope Budget、训练 epoch/优化计划、随机种子、执行预算、环境与数据哈希。首轮 seed=100；主脚本执行 A0–A4 和下列四项消融，共八个训练配置（A0 不训练）。训练按当前配置 30 epochs、有效 batch 16、BF16、AdamW；任何显存适配须对所有训练组等量调整并记录。
6. 新 Smoke 只作工程验收：1–2 个真实 train 任务、一个合法同前缀反事实决策组、2–5 次更新、H=4 硬递推。必须证明有效 decision group 实际进入 batch、Goal-Sufficiency Loss 非零、Scope gate/logit 梯度非零、参数有更新、计费和显存有收据。若标签不存在，说明阻塞原因，不把合成 smoke 或预审计改称通过。

## 5. 主要实验、基线与消融

所有方法使用同一 Frozen Actor、任务指令、候选动作、数据、模型容量、Scope/inspection/执行预算和测试任务。目标解析、公开工具与检查协议也必须共享。允许方法配置差异严格限于下表所示机制。

| 组 | 配置 | 作用 |
|---|---|---|
| A0 | Frozen Actor | 无 Predictor 的系统质量参考。 |
| A1 | 普通 Step + H=4 Rollout Predictor | 通用状态预测基线。 |
| A2 | Do-JEPA-style Paired Effect Learning | 在同一候选数据上增加成对效果/作用支持监督；属于适配基线，不是完整复现。 |
| A3 | RepairScope v1.1：Dynamic Scope + Boundary + State Escape | 原方法基线。 |
| A4 | Goal-Sufficient Dynamic Scope | v2 完整方法。 |
| 消融 1 | A4 去掉 Goal-Sufficiency Loss | 检验决策监督对实际 Scope 学习的增量。 |
| 消融 2 | A4 去掉 Boundary Loss | 检验条件化边界监督。 |
| 消融 3 | A4 去掉 Goal-Relevant Escape | 检验目标相关越界监督。 |
| 消融 4 | Static Scope + Uncertainty Expansion | 固定范围先验及不确定性扩展对照。 |

v2 新增小头在可比配置中实例化以控制模型容量；Frozen Actor 不更新。任何配置若未包含有效目标决策监督，必须按真实数据/Mask 状态报告，不能以普通状态标签替代。

主实验以 AppWorld `test_normal` 完整官方清单为主要正式结果。按当前代码/数据 pin 核验任务数与 hash；测试官方 checker 只在隔离状态评分。A0–A4 与四消融的测试任务、随机数、重试、调用上限完全配对。测试失败、缺失或超时按预注册规则计入，不能事后删除。

主 A 结束后才运行预算化探索 B：从同一 A4 checkpoint、相同冻结 train-only 候选计划池起步，Random、Uncertainty、Novelty-Learnability、Risk 四策略每个最多 2000 个真实执行成本单位，总上限 8000，\(\epsilon=0.1\)。按累计 25/50/75/100% 记录性能。探索训练数据不进入冻结查询集；不因效果不佳延长预算。

## 6. Evaluation Plan

### 6.1 主要指标及口径

| 指标 | 计算口径 | 适用范围与报告要求 |
|---|---|---|
| AppWorld TGC / SGC | 使用固定 AppWorld 原生聚合：TGC 为 task success 均值；SGC 为各 scenario 内成功率最小值再平均。 | 只接收原生 checker 完整覆盖；报告任务数/Scenario 数、失败数和 native coverage。 |
| Task Success Rate | 成功任务数 / 预先冻结的计划任务数。 | 与 TGC、SGC 并列呈现；未完成任务保留在分母。 |
| Goal Degradation Rate | 已满足且当前受保护目标中，执行后从 true 变 false 的数量 / 可判定受保护目标步数。 | Unknown 不作负例；报告可判定分母、目标标签覆盖率。 |
| Repair Success Rate | 修复开始前受损目标恢复且无新增受保护目标损害的修复事件 / 可判定修复事件。 | 区分恢复、未恢复、新损害；报告事件覆盖。 |
| Scope Recall@B | 预算 \(B\) 内 Scope 命中的真实审计变化字段数 / 完整审计的真实变化字段数。 | 真实变化只按冻结审计口径；\(B\) 和有效样本数必须报告。 |
| State Escape Rate | 至少存在一个 Scope 外真实变化的完整审计步骤数 / 可完整评价步骤数。 | 部分审计只提供可见正例，不能提供无逃逸负例；分别报告覆盖率。 |
| Goal-Relevant Recall@B | 已确认与受保护目标退化相关的字段被 Scope 覆盖数 / 有可靠字段归因的目标相关字段数。 | joint attribution、未知依赖、未执行谓词等 Mask；此实现的范围标签针对可归因保护损害，不把所有变化称作目标相关。 |
| Goal-Relevant Escape | 有可靠目标相关字段遗漏的步骤数 / 可验证目标相关逃逸步骤数。 | 由保护状态从真到假、合法目标标签及离线审计字段共同构造；报告 Mask 覆盖和校准。 |
| Decision Regret | 对每个合法可比较候选组，充分离线效用最优候选效用减去部署 Controller 实际所选候选效用；按组平均。 | 包含 Predictor、风险、Scope、动作成本等实际决策因素，是系统总 regret，不是 Scope 单独因果效应。报告有效组覆盖率、分子/分母和排除原因。 |
| Scope-Isolation Oracle Diagnostic | 在相同离线候选中，仅替换 Scope 信息范围，用 oracle patch 诊断遗漏导致的决策损失。 | 单列机制诊断，不能替代真实 Predictor 的 Decision Regret 或任务成功。 |
| Goal Preservation–Cost Curve | 对同一 checkpoint、同一候选和冻结数据，扫预先指定的 Scope Budget；同时报告目标保护指标与真实执行/检查成本。 | 离线字段预算曲线与系统任务成功—实际成本曲线分开绘制；不重新训练、不挑测试最优预算。 |
| Unseen Composition Generalization | 在训练前冻结的未见条件、动作或目标组合分组上计算相同目标指标。 | 需要组合 Manifest 与无重叠审计；若当前自动标记/覆盖不足则 N/A 并报告覆盖，不事后挑组合。 |
| Execution / Inspection Cost | 执行动作、合法检查、Actor 调用、反事实前缀/条件/查询/失败/续跑的实际记录成本。 | 分项和总量均报；Scope 较小造成的成本节省不能以漏掉必要动作换取。 |

对所有带标签指标报告原始分子、分母、标签覆盖率和按任务/Scenario 聚类的置信区间。无法可靠计算时记 N/A，不把未观测或未知变成 0。判断任务完成质量必须与损害率、动作执行数及成本并列，防止“少做动作”伪装为安全。

### 6.2 Scope 与预算公平性

固定 Scope Budget、候选字段上限、覆盖预留规则、inspection cap、Actor 候选数、环境调用上限和重试规则。主对照共享同一训练/评测分支与候选缓存。不同 Scope Budget 曲线只在同一冻结 checkpoint 上离线重算，不重新训练；真实系统成本曲线使用相同任务顺序、重试和候选集合。

Decision Regret 只在 action/reference 语义和审计充分的同组候选上计算。若目标 unknown、动作非法、候选不匹配、有效用相同或没有充分信息标签，该组排除并进入 coverage/exclusion 表。不要把训练反事实组上的 regret 当作泛化结果；主要比较使用冻结 dev 反事实组，dev 来源明确且不参与更新。AppWorld 测试集不给反事实搜索或隐藏状态检索。

### 6.3 分层 benchmark 候选池与接入准入

AppWorld 是主训练和核心评测。以下 Tier 2/3 是候选，不要求首轮适配或运行。进入正式方法级泛化前，需先完成只读可用性核查，并证明合法初始状态恢复、同前缀分支执行、目标损害真值覆盖、任务许可和固定成本均满足；否则只报告原生终局成绩或 N/A，不将其包装成 Scope 证据。

| 层级 | Benchmark | 可能提供的证据 | 恢复/反事实/真值边界 | 接入判断 |
|---|---|---|---|---|
| Tier 1 | AppWorld | 结构化 API 状态、目标 checker、可控分支；与本方法最匹配。 | 固定 AppWorld 版本；在线只用公开 API/schema 和返回。独立快照进程做离线 checker，不能在活动 world 直接调用会污染状态的 evaluator。 | 主环境；先跑 `test_normal`。 |
| Tier 2 | WorkArena / WorkArena++ | ServiceNow 企业工作流、组合任务和 browser DOM/AXTree。 | managed instance pool 的训练、持久性及恢复权限需逐条确认；原生任务 validator 不等于全域逐步差分。 | 需可许可自有/冻结实例、同起点 branch 和独立业务审计后才做方法评测。 |
| Tier 2 | AndroidWorld | Android emulator、多应用 GUI、参数化任务和程序化任务 validator。 | setup/teardown 可复位任务；任意中途 snapshot 与跨 app 全状态审计需另验。原生成功检查不自动给出完整副作用差分。 | GUI 方法扩展候选；审计和分支恢复未验证前不计算完整 Escape/Goal Recall。 |
| Tier 2 | WindowsWorld | Windows 上跨应用专业流程。 | 论文报告 181 个任务/17 个应用；原生截图或任务评分不是逐字段目标损害真值，VM 发布与真实复现成本需验证。 | 需 VM/应用可用、合法 snapshot 与状态 validator 通过后再决定。 |
| Tier 3 | OSWorld / OSWorld-Verified | 通用桌面 app 与多步操作。 | 版本需分开固定；provider 的 snapshot 能力不同，原生终局 checker 不覆盖完整逐步副作用。 | 可作为桌面 GUI 候选，需联合恢复 VM、文件与应用数据。 |
| Tier 3 | OSWorld 2.0 | 更长桌面工作流、动态环境和跨源任务。 | 独立基准版本；网站、容器和 VM 需联合恢复；资产部署成本高，原生安全诊断不等于逐步完整状态真值。 | 后续长程扩展，不进入首轮。 |
| Tier 3 | LegacyWorld | 受监测 Windows legacy workflow 的 atomicity 和失败残留风险。 | 论文提供 28 个 GUI workflow 与任务 validator；原子性只覆盖其监测字段，不自动等同完整状态差分或逐步反事实。镜像、许可和平台适配需先验证。 | 适合目标安全补充，必须逐 validator 审核覆盖。 |
| Tier 3 | WebArena | 自托管多站点 web 长程任务。 | 容器/DB/浏览器状态需共同 reset；URL/答案/HTML checker 不等于所有业务字段的 goal-protection truth。 | 作为补充；缺少审计真值时只报原生任务指标。 |

环境来源：AppWorld [ACL 2024 论文](https://aclanthology.org/2024.acl-long.850/)；WorkArena [论文](https://arxiv.org/abs/2403.07718) 与 WorkArena++ [NeurIPS 2024 论文](https://arxiv.org/abs/2407.05291)；AndroidWorld [论文](https://arxiv.org/abs/2405.14573)；WindowsWorld [论文](https://arxiv.org/abs/2604.27776)；OSWorld [论文](https://arxiv.org/abs/2404.07972) 与 [OSWorld 2.0](https://arxiv.org/abs/2606.29537)；LegacyWorld [论文](https://arxiv.org/abs/2608.14131)；WebArena [论文](https://arxiv.org/abs/2307.13854)。正式运行前记录具体 release/tag、代码 commit、清单 ID/hash、环境镜像、恢复接口和 checker 覆盖。

## 7. 与相关工作的边界

| 工作 | 已有贡献 | RepairScope v2 的可区分问题 | 不应声称 |
|---|---|---|---|
| WMA / Web Agents with World Models | 在 Web 导航中预测环境动态并用于候选动作/工作流决策。 | 固定字段预算下学习“哪些状态信息足以保护已完成目标”，并以同前缀反事实决策 regret 监督 Scope。 | 不声称发明 Agent world model、预测式控制或候选重排。 |
| Do-JEPA | 用真实动作干预和参考动作学习效果、作用支持、传播与不变性。 | 以任务公开目标和目标保护偏好定义 Scope sufficiency；核心监督是受限 Scope 与充分离线信息下动作决策的差异。 | A2 是本代码上的公平适配基线，不声称复现其潜空间结构或物理实验。 |
| OnlineWM | 主动在线采样并用反事实对比持续改进世界模型。 | 把探索证据价值与当前 Goal-Preservation 风险、潜在决策错误减少和执行成本关联。 | 当前启发式不等于 OnlineWM 方法复现；不声称一般主动学习首创。 |

参考：[WMA, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a00548031e4647b13042c97c922fadf1-Abstract-Conference.html)；[Do-JEPA, 2026 preprint](https://arxiv.org/abs/2609.37378)；[OnlineWM, 2026 preprint](https://arxiv.org/abs/2609.23753)。创新定位是待实验验证的增量问题，需结合更广泛的目标条件规划、风险敏感决策和信息瓶颈文献继续做先例筛查；不作“首次”主张。

## 8. 贡献边界、局限与停止条件

若结果支持假设，拟报告的贡献是：

1. 定义固定 Scope Budget 下的 Goal-Sufficient Decision Regret 学习目标，并把合法离线决策监督直接接入动态范围选择器。
2. 在同一方法中区分一般状态逃逸、目标相关逃逸和条件化作用边界，并用合法反事实和显式 Mask 学习。
3. 在 AppWorld 上以共享数据、冻结 Actor、等容量/等预算对照评估目标保护收益、Scope 质量和成本；泛化 benchmark 仅按其真值能力使用。

主要局限包括：监督质量依赖公开目标到状态字段的正确绑定；离线目标效用反映设定权重而非普适效用；Top-k 直通梯度有偏；Compound goal 只覆盖 Goal Evaluator 实际支持且可审计的子集；目标相关逃逸归因会对复杂联合损害保守 Mask；单一 AppWorld 主环境不能证明 GUI、机器人或开放世界泛化。真实 AppWorld、4090、held-out Decision Regret、任务成功提升及跨域结果目前都没有完成。

若公开业务目标绑定为零、没有 informative 且 budget-competitive 的合法候选组、Scope 梯度没有进入实际 selector、dev/test 信息边界被破坏或总预算未冻结，则停止受影响实验并报告缺口。不得弱化监督条件、放宽预算、按测试结果改目标或用局部合成结果声称成功。
