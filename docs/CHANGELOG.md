# RepairScope v2 变更记录

本文件记录MacBook端提交的研究代码、配置、文档及版本边界。每次相关提交必须同步追加条目；已有条目不覆盖，发现错误追加更正。精确提交号和逐文件差异由Git历史核验。

服务器逐项操作另记 `/root/autodl-tmp/RepairScope-server-notes/CHANGELOG_V2.md`，字段模板见 [自主续接说明第7节](CLAUDE_OURS_ONLY_CONTINUATION.md#7-每项修改必须即时写入markdown变更日志)。服务器日志、原始数据和权重不因这一要求自动加入Git。

## 2026-10-18：采集硬预算落地（F5）

- 起点：`f8c8a90`。用户转交服务器应用收据：F3/F4及两个回归文件after hash与本地一致，报告33项CPU测试通过；不重复旧Smoke/H4/GS诊断。修订计划仍指出采集预算未在调用前强制执行。
- 修改：新增独立采集持久账本，natural/CF/train/dev共享固定run_id、文件与额度；执行前保存pending预留，成功或已知失败按真实费用结算，未知成本或遗留pending封存停止，不自动重置/重试。真实AppWorld采集必须显式配置预算，fixture无预算保留原工程兼容路径。
- `collection.py`：natural调用前设置per-step/全局剩余额；CF向branch传per-cell/全局剩余额。先结算cost再解析record，失败文件同时保存known_cost、unknown状态及账本快照；不混用controller reset/Actor/prediction单位。
- `actor.py`：每次真实HTTP尝试前调用可选预算hook，失败和格式重试均计数；generation和HTTP分开，cache/scripted不伪造HTTP费用。hook不加入策略输入或候选cache key。
- 验证状态：正在进行纯fixture/mock回归。无真实环境/Actor请求、正式优化或GPU实验；配置接口不代表预算批准。
- 回退：由起点提交恢复相关模块与配置；已有账本不得删除、重置或用新目录绕过封存。恢复代码不恢复已花额度，旧记录与收据保留原版本。

### F5最终验证与服务器补丁

- 完整本地CPU工程回归：`PYTHONPATH=. ../WMA-Agents/.venv/bin/python -m pytest -q tests/repairscope`，**313 passed in 5.86s**。新增45项账本、15项入口/mock HTTP/崩溃测试；已有合成优化单列，不记真实训练。
- 只读独立终审无阻塞。环境pending预写、失败费用、未知封存、跨阶段不可重置、HTTP调用前hook及恢复、目录fsync与锁均有回归覆盖；未执行真实AppWorld/Actor/正式GPU。
- `docs/handoff/2026-10-18-budget/`记录本轮用户转交收据、实际核对项和F5 patch。旧服务器actor仍可能使用`agent.wma_core`，本轮补丁保持该import；同一patch在legacy与规范起点均dry-run/应用通过，after hash匹配各自receipt。
- patch SHA256=`102c79170ac7ec8af06a18b40de3e940891682ecc3d8933e12257a6e5babb3be`；相关源码与新增测试前后hash见`BUDGET_UPDATE_RECEIPT.json`。本轮不改模型、损失、数据或现有checkpoint。
- 当前文档`docs/V2_ACQUISITION_BUDGET.md`说明预算接口、固定15任务候选的静态整理和后续边界；费用数值仍是申请，没有因代码实现自动获批。入口不再要求重复F3/F4或导出。
- 最终校验：13份Markdown的本地链接/围栏、F5源码与patch SHA复核通过；源码与新编文档的Git空白检查通过。用户原样收据中的空白及unified patch上下文按字节留存，排除这两份来源文件的格式空白，不改其hash。

## 2026-10-18：实际交付核验、工程修复与计划审查

- 起点：`7517df3`。收到用户粘贴的实际补丁、Manifest、完整计划与账本；从文本重建的六个文件逐字节SHA256均与报告一致，三份源码的修改前/后hash也可复现。完整tar和其余payload尚未取得，不能宣称22项在MacBook复核通过。
- 来源：`docs/handoff/2026-10-18/`保存原样文本、历史补丁及PROVENANCE；原文件不随审查意见修改。
- F1：`Predictor._goal_context`在scatter前匹配dtype，保持原公开输入与可微投影；F2：五处概率BCE移到autocast外fp32计算。A4入口使用既有协议包装且只运行Ours四阶段。
- F3复核：服务器的`padding[:,0]=False`把缺失history开放成有效上下文；缺失embedding并非零值，且unified分支仍有全mask NaN。拒绝原样应用历史model.patch中的F3。
- F3替代：`causal_padding_mask`只为全遮挡query开放自身；其他有效query仍屏蔽缺失key。应用BLite与unified，不改变真实时间顺序或公开输入。新增局部测试覆盖有限前向/反向与无效history payload不变性。
- 计划审查：采集平均成本不能标硬上限；CF bundle与尾窗改变实际训练调度；test共享ledger不等于环境API数，候选数不等于Actor HTTP数。正式计划待按实际代码修订，未批准新增采集或正式训练。
- 当前验证：正在进行局部CPU工程回归；最终结果、完整文件hash和调度修复单列补充条目。无服务器/真实环境/Actor调用或正式GPU实验。
- 回退：用起点提交恢复相关源码、入口及配置；不覆盖服务器既有数据/checkpoint。若回退F3，原全mask NaN风险恢复。F3修改后的预测须使用新源码hash，不能把既有H4收据重新标成新版复验。

### 同轮补充：完整epoch调度与协议测试

- F4：`training._training_schedule`以独立seed RNG预生成每个epoch的CF bundle顺序与实际装箱，精确计算warmup/cosine总步数。旧未打散估计曾将`max_steps=null`变成过小硬上限，可能在30epochs前停止；现在null完成全部epoch，显式max_steps只截断该完整日程。
- 恢复边界：新checkpoint保存冻结training_schedule；同数据/日程可exact resume，无日程的旧checkpoint仍可推理但不冒充新版exact resume。不会继续既有Smoke checkpoint为正式训练。
- 相关文件：`repairscope/training.py`、`test_training_schedule.py`。局部31项测试通过，含30epochs完成与dropout下逐位一致恢复。源码before=`ff4e9adc08610c4dabab1a52af34a1c374b6d2941ba8c50234200027b13eb326`，after=`2856fe3e14349fe0d8106eec92e5c4ec7312653423a762849087dbfb061804fb`。
- 首次整体工程回归246通过、1失败：旧CLI测试仍允许A4使用两个seed和旧两阶段，已与新的协议入口冲突。将A4移到专用四阶段/seed100/固定参数及覆盖拒绝测试，保留原A1–A3测试；未放宽入口。
- 本地合成优化仅用于上述回归；真实环境、Actor、正式训练和服务器新增成本均为0。最终整体结果在验证收据中记录。

### 同轮最终验证与交付

- 最终整体回归：`PYTHONPATH=. ../WMA-Agents/.venv/bin/python -m pytest -q tests/repairscope`，**253 passed in 5.38s**；Python3.12.14/PyTorch2.10.0/CPU，不安装依赖。
- 源码与文档的Git补丁空白检查通过（原样保存的`.patch`排除格式所需的空白上下文行）；Markdown本地链接与围栏、六份交付文本hash及当前源码hash复核通过。
- 为服务器现有model=42705db9…a73a5、training=ff4e9adc…326生成`server_followup.patch`；在本地重建该起点，dry-run与实际应用均通过，两个源码和两个新增回归文件的after hash全部一致。patch完整SHA256=`e619c7cf9ed53abfc2bc9bbc16d1ed37831a682df4b97190e5d2581eab7d4632`。
- 精确源码hash见`docs/handoff/2026-10-18/LOCAL_INTEGRATION_RECEIPT.json`；服务器应用前后hash见同目录`SERVER_UPDATE_RECEIPT.json`。两份收据没有把服务器CUDA验证、旧H4重验或正式Ours记成完成。
- `F3_MASK_AUDIT.json`保存精确服务器model hash42705db9…a73a5与当前源码的静态CPU对照：服务器unified no_grad两项非有限，缺失history扰动最大约0.254960；修复后均有限且扰动0。收据SHA256=`34c93fcbdf0bdbf865015e1a5ff71d713563e4f4a3df01b18aef3a9dc38a194f`，无环境/Actor/优化步。
- 当前说明为`docs/V2_HANDOFF_INTEGRATION.md`；入口不再要求重复导出。原计划、Manifest和账本原样保留，费用/步数修订意见单独记录，没有擅增预算或更改GS/prior/Scope。
- 恢复方式：可由本条起点commit恢复原MacBook源码；服务器须按新说明备份其不同起点再应用增量patch。旧checkpoint仍可推理，但新日程exact resume要求对应schedule，旧收据保留原hash。

## 2026-10-18：归档离线续接结论与补丁移交

- 起点：`dac39cbcbf1ab5976218255267f4d9b2b27474f0`。
- 原因：用户转交已完成的派生标签、一次离线GS诊断、正式数据计划及日志；需要关闭已完成任务，避免再次执行同一诊断。
- 修改：新增 `docs/V2_POST_CONTINUATION_DECISION.md`；更新CLAUDE、README和续接说明入口，记录服务器报告与本地代码核对的区别；明确下一步只交付实际补丁、计划全文、完整hash和已有收据。
- 解释：当前10组oracle代理饱和不证明所有任务零信号；static prior不是硬纳入规则；Smoke Predictor错误不构成正式训练失效。正式数据与可检验GS增量是独立缺口。
- 版本边界：本次未取得或合入F1–F3补丁，未访问服务器；无源码/方法/配置变更，无新增环境/Actor/优化步。
- 验证：本地核对GS与selector定义，独立只读复核；Markdown本地链接、围栏与补丁空白检查。未重跑工程测试或训练。
- 回退：恢复本条提交前的文档版本；旧入口含已完成任务，回退时需保留已完成状态，不能据此重跑。

## 2026-10-18：Ours-only自主续接说明

- 起点：主分支 `3f6cd617c4b6b9d136b8eaec160e903f69b7bc8f`。
- 原因：用户报告服务器已完成多项工程修复及H4，但GS仍零、派生计数目标unknown、正式数据未冻结；要求Claude自主续接且每项修改有Markdown记录。
- 修改：新增 `docs/CLAUDE_OURS_ONLY_CONTINUATION.md`；更新 `CLAUDE.md`、README、原交接和Proposal的当前阶段说明；建立本变更记录。
- 行为：当前只推进Ours。先对已有记录做一次派生字段等义修复和监督判定，不重置前轮取证预算、不自动追加采集/Smoke。真正具备数据、准入和明确预算时可直接启动Ours；其他组不运行。
- 科学边界：保持GS、先验和字段预算；明确oracle输出覆盖不等于预测条件充分。派生标量不展开为大量token制造竞争。
- 记录要求：每项服务器代码/配置/映射/数据转换修改须即时记录前后hash、diff、验证、费用与回退，F1–F3纳入索引。
- 验证：Markdown链接、目录锚点、围栏和Git补丁空白检查；本条只改文档，不重新运行训练或声称完成新科学验证。
- 回退：可按本条所在提交恢复上一版本文档；回退后旧说明含全套实验安排，当前用户的Ours-only约束仍须单独保留。

## 2026-10-18：独立v2仓库与默认入口

- 提交：`3f6cd617c4b6b9d136b8eaec160e903f69b7bc8f`。
- 原因：用户要求GitHub默认仓库直接承载新方法，不再保留旧版独立项目。
- 修改：移除旧WMA/WebArena/Mind2Web应用、数据、提示、无关脚本与旧文档；候选聚合/排序迁到 `repairscope/candidates.py`，保留来源。
- 默认：`load_config()`、base YAML及CLI选择goal_sufficient；显式full仍是A3，旧checkpoint兼容不重新解释为v2。修复variant覆盖仅改标签的问题。
- 打包：`repairscope 2.0.0`，只包含repairscope包；README和依赖对应当前方法。默认main与交付分支同步。
- 验证：214项本地测试通过；独立wheel构建与仓库外导入通过。使用既有环境PyTorch2.10.0/pytest9.1.1，无真实AppWorld实验。
- 回退：引用该提交的父提交，源码和配置必须一起恢复；不能只回退base YAML却保留v2运行标签。

## 2026-10-18：完整Proposal与Evaluation Plan

- 提交：`30e17abc4f59f4b0b4f2e84ada5c8987112b833e`。
- 修改：14章完整研究方案、26个公式、A0–A4/四消融、精确定义的指标、统计与分层benchmark计划；同步B采集与查询的预算区别。
- 验证：数学编号、括号、Markdown链接和补丁检查。未启动实验。

## 2026-10-18：初版服务器交接与GS梯度实现

- 交接提交：`0ce975d933ab6a4650346609d1558e5e13e3da4c`。
- 核心提交：`62fd96e54891b799cf0819ae5e0c74253363f040`。
- 机制：oracle混合状态的动作偏好KL、直接Scope gate路径、固定质量直通梯度、可训练组预审计及实际梯度诊断。复合目标使用精确硬前向和局部有限差分代理。
- 边界：预审计可训练不保证当前参数梯度非零；合成Smoke不代表真实目标学习；数据和方法效果需服务器独立收据。

## 服务器已报告、尚未据补丁合入本地的修改

后续状态说明：本节保留最初未取得补丁时的历史边界。现已核验并合入F1/F2/A4，F3由本地安全mask替代，另修复F4调度；精确状态以上方“实际交付核验”条目及验证收据为准。服务器尚需应用增量patch，不意味着本地仍缺F1/F2。

以下仅转述用户提供的运行报告，不表示本仓库已经包含对应修复：

| 标识 | 问题 | 服务器报告状态 |
|---|---|---|
| F1 | BF16 goal-context scatter dtype不一致 | 已修复，见BUGFIX_LOG_V2.md及model patch。 |
| F2 | autocast下概率BCE不安全 | 已修复，见BUGFIX_LOG_V2.md及losses patch。 |
| F3 | 空history下全mask因果时序attention产生NaN | 已修复，并报告H4硬递推收据。 |

报告的基础版本为0ce975d，源码树hash为 `10ee6e5cd1e0ac9cdb59923c3698a85e9a04d3491b41fc7165ba9589d8880390`。实际合入前需取得源码patch，核验文件hash与最小回归，再在本文件追加真正的MacBook提交与验证记录。服务器路径或95项测试的转述不能替代这一同步证据。
