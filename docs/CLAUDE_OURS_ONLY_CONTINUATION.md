# RepairScope v2：Ours-only 自主续接任务

**状态更新（2026-10-18）：一次离线续接、文件交付及F3/F4服务器更新已完成，结论仍为FORMAL_INPUTS_NOT_READY，并有NO_IDENTIFIED_SCOPE_LEARNING_GAP。当前执行[采集硬额度与F5任务](V2_ACQUISITION_BUDGET.md)，不要重复下述已完成步骤。本文的Ours-only、预算和变更记录约束继续有效。**

本说明保留已完成离线续接的指令与持续执行边界，当前新增任务以顶部链接的结论与交付说明为准。用户要求第一阶段只跑 Ours，并让服务器 Claude 在明确边界内自主推进。完整 Proposal 中的 A0–A3、消融、B 和跨环境评估是后续研究设计，当前均不启动。

## 1. 接手方式与当前事实

先读本文件，再读同目录的完整 Proposal。原服务器交接用于查接口、路径和既有约定，不重新执行其中已完成的步骤。

以下是用户转交的服务器报告，接手时从已有文件核验；不是 MacBook 对服务器的实时探测：

- 代码目录：`/root/autodl-tmp/RepairScope-v2`，无 `.git`，基础版本 `0ce975d933ab6a4650346609d1558e5e13e3da4c`。
- 复用 Python：`/root/autodl-tmp/RepairScope/.venv/bin/python`；不安装或降级依赖，不重新下载权重。
- 源码树 hash：`10ee6e5cd1e0ac9cdb59923c3698a85e9a04d3491b41fc7165ba9589d8880390`。后续编辑必须产生新 hash，不能继续沿用旧值。
- 已有 F1：BF16 goal-context scatter dtype；F2：autocast 下概率 BCE；F3：空 history 下全 mask temporal attention 的 NaN 修复。补丁和依据见 notes 中 `BUGFIX_LOG_V2.md`。
- 指定测试已报告95项通过；已有真实CF、Boundary相关运行和H=4硬状态递推。先使用这些收据，不重跑全面测试或相同Smoke。
- H4结果见 `v2_supplement/runs/h4_hard_rollout/`。checkpoint位于 `v2_supplement/runs/training_boundary3/checkpoint.pt`，报告的SHA256为 `7a71d286909bde378d4d8369731cffbfa2d0b2989923c61603d18ef11935a0ea`。
- 当前GS仍未取得有效学习信号；phone组KL=0、梯度约8.9e-9；follow的派生follower_count没有可靠审计证据而被Mask。
- 前轮补充取证的3任务上限已到。正式TRAIN_DATA/EVAL_DATA没有冻结；Ours、其他对照、消融和B均未正式启动。

现有代码含服务器修复，不用 `git pull/reset` 或解压新源码覆盖它。只需要获取本说明时，将这个Markdown单独保存到已有notes目录。源码提交与推送继续由MacBook负责；服务器只保存补丁和收据。

## 2. 自主执行范围与明确的方法决定

按下列范围完成工作，不为路径、既有字段重映射、局部回归和日志整理逐步请求确认：

1. 核验并归档现有F1–F3修复。
2. 使用已有公开API说明、合法返回、原始记录和审计，修复可证明等义的派生目标表示或审计投影。
3. 在已有数据上完成一次离线标签重建与GS信号判定，不执行新环境调用、不调用Actor、不做优化更新。
4. 整理正式数据采集Manifest和费用明细，并给出能否按原方法推进的终局结论。
5. 仅当已有正式数据、训练准入和明确批准的剩余总预算全部满足时，直接启动一次Ours；不再等待逐步批准。

本轮保持GS定义、static prior、B=128、candidate_limit=256和目标语义。不得通过降低预算、删除目标、扩大候选、修改权重、软前向读取全部字段、随机丢弃关键字段等办法制造非零GS。

GS=0应分情况解释：unknown/Mask/未入batch/未启用辅助项/没有预算竞争，与“当前oracle Scope已经复现教师偏好”是不同问题。后一种情况零KL是正常结果；不要求每个batch或训练后期始终有正KL。

同时检查代理边界：oracle patch直接提供被选输出字段的真值，不证明模型已能预测它。例如目标输出字段已入Scope，但预测它依赖的条件字段仍遗漏，oracle GS仍可能为0。因此必须分别报告oracle范围诊断与真实Predictor的候选决策表现，不把前者当作预测充分性。

## 3. 派生字段修复：只修测量，不偷换问题

优先核验已有follow记录，按实际公开API和底层关系定义确定follower_count的计算式。不能先假设“关系行数”等于API计数：实体、方向、过滤、去重、时间及创建/删除语义都需一致。

允许两种等义路径：

- 原公开输入已有一个follower_count标量时，保持同一个字段/token。只有现有完整审计依赖足够时，才由底层关系重建其离线前后真值、变化和来源证书。
- 原公开表示本来就暴露对应关系字段时，才考虑用已有count/aggregate DSL绑定相同目标；须证明依赖域完整和谓词实际执行。

**语义等价不自动意味着预算表示等价。** 不能把一个可公开读取的计数标量展开成大量关系token，人为制造N>B；也不能隐藏原可用标量，强迫模型覆盖底层行。

仅有changed rows通常不足以证明总数；`audit.complete`标记不替代实际依赖证据。当前未执行的where不能用来宣称过滤已生效。空集合、删除项、分页未取全、去重或动态依赖无法证明时，继续unknown。

隐藏底层数据只用于允许的离线标签，不补入在线state/history/schema。审计能算出一个值还不够，在线字段必须有合法公开来源。不能借官方checker反推目标或在线依赖。

保留原始记录和原始unknown；另写版本化derived-label文件和转换Manifest。所有未来比较组必须共享相同测量修复，不只修Ours。

## 4. 一次离线判定，避免第三轮相同Smoke

使用现有输入、实际权重和冻结seed，不修改排序分数或删除关键字段。优先复用已有结果，仅计算修复导致变化的部分：

- 逐组列明公开目标、原始/修复标签、候选数、可检索依赖、Scope、固定预留和预算竞争。
- 分列eligible、informative、结构可训练、实际loss组及实际GS梯度；不得以预审计trainable计数代替实际梯度。
- 对修复后的现有记录，必要时在CPU副本启用autograd检查GS到selector的路径；无optimizer.step，记录精度和模式差异，不把CPU诊断冒充BF16训练验收。
- 零信号归因只能依据证据：标签不足、Mask、候选不可达、预算全覆盖、当前偏好已匹配、局部差分平台或实现错误。
- Boundary区分“没有有效四格”“有效标签为零”和“非零条件边界”；H4已有工程收据保留，不重复环境执行。

可自行修复有明确最小复现的等义实现错误，并只跑相关回归。若需要改变GS监督对象、训练分布、Scope策略或目标含义，交付具体机制缺口和最小修改设计，不以工程bug名义直接实施。

若已有数据不足以恢复派生标签，或者所有可靠动作对仍被oracle Scope充分覆盖，本轮得出“现有数据无法检验B128下GS学习增量”的结论。它既不证明方法有效，也不自动反驳整个研究方向；不要再采到有正例为止。

## 5. 预算与正式数据

前轮300执行单位和3任务是补充取证上限，不会因新对话、重命名目录或生成新计划而重新获得。当前本轮离线续接的新增环境调用、Actor调用和优化步预算均为0；先从已有ledger汇总实际已用额，不将上限写成实际消耗。

正式采集必须有另行明确批准且可核验的总预算。此文件不把Ours-only授权解释为任意规模采集、GPU时长或调用数的自动授权；其他组和B的预算也不能借用。若现有文件没有正式预算，完成下面的可审阅产物后结束本轮，不写一句“等待批准”就停工：

- 拟用官方train/dev任务和Scenario清单及hash；
- 公开业务目标类型、映射状态、可审计依赖与缺失项；
- 每个自然/CF计划的合法前缀、条件、参考、续跑、查询及失败/重放成本上限；
- 预计输出记录、独立动作对、四格和H4窗口数，分别标确定值与估计；
- train、dev自然、dev evaluation-only CF、正式系统评测的独立费用及全局总额；
- 由窗口与CF bundles计算的优化步数、GPU时间估计与停止上限。

不要为了凑500/1000/2000条transition复制记录。条数不等于独立监督组；优化步数由实际windows/bundles/minibatches决定。5步Smoke的15.7秒/update仅是粗估，不能直接当作完整运行承诺。

## 6. 条件满足后只运行Ours

若本次离线修复确实解决监督问题，且正式数据和预算本来已经具备，可直接按下列条件推进。当前报告中的Smoke数据不满足“正式数据已冻结”，不能将其改名满足条件。

运行前核验：

- 真实train-only数据、独立dev数据及其源任务/兄弟分支隔离和hash；
- GS真实标签、预算竞争和Scope独立梯度的既定验收，不用浮点底噪通过；
- Boundary等开启项有对应有效证据，未知保持Mask；
- 固定配置、实际代码hash、预算与独立输出目录。

唯一入口是 `scripts/repairscope/a4_goal_sufficient.sh`。它按train→dev predict-eval→一次calibrate→evaluate执行；该单独脚本不锁全部参数，因此必须在最终CONFIG中显式固定并核验：

| 项目 | 本阶段值 |
|---|---|
| variant | goal_sufficient |
| seed/epochs | 100 / 30 |
| max_steps/warmup | null / 0.1，清除Smoke的5步截断 |
| scope/candidates/horizon | 128 / 256 / 4 |
| precision/batch | BF16，micro4、累积4 |
| optimizer | AdamW，3e-4，WD .01，clip1 |
| losses | scope、patch、goal、rollout、Boundary、State Escape、GS、Goal Escape保持完整配置；paired=0 |
| data | 正式TRAIN_DATA/EVAL_DATA，allow_synthetic=false |
| test | test_normal完整官方清单；task_ids=[]，清除TASK_FILE和Smoke子集 |

先DRY_RUN验证只有Ours四个阶段，再在tmux启动。正式训练从固定初始化开始，不能把不同日程的Smoke checkpoint直接续成正式30epochs。相同正式协议中断可exact resume；已完成阶段不覆盖重跑。

dev只预测/校准，test不分支搜索、不更新模型或目标。没有可靠dev校准标签时按事实记录，不借其他checkpoint阈值。Ours完成后停止，不自动运行对照、消融、探索或额外seed；单独Ours结果不宣称相对提点。

## 7. 每项修改必须即时写入Markdown变更日志

**强制要求：所有代码、配置、目标映射、审计投影和数据转换修改，都必须在修改发生时追加记录，不能等出问题后补写。**

主日志使用 `/root/autodl-tmp/RepairScope-server-notes/CHANGELOG_V2.md`。保留已有 `BUGFIX_LOG_V2.md`，在主日志中索引F1–F3及其实际证据，避免重复或相互矛盾。日志只追加；纠错用新条目标注被更正条目。

每条记录至少包含：

~~~text
变更ID、时间与时区：
分类：工程修复 / 等义目标映射 / 配置 / 数据转换 / 仅诊断
修改前版本：基础commit、源码树hash、相关文件hash
触发问题：具体报错或失败指标、最小复现、日志绝对路径
原因与影响：实际根因；涉及哪些方法、数据、checkpoint和指标
修改内容：文件绝对路径、函数/配置键/目标ID、可用时的行号
语义边界：为何等义；是否改变输入、标签、Scope预算或研究机制
前后差异：修改前/后文件hash、仓库外patch/备份路径及SHA256
验证：实际命令、解释器/设备/精度、退出码、结果与覆盖范围
新增成本：环境/Actor调用、GPU时间；无新增则写0
回退：具体恢复源与操作，哪些产物需要标记失效
状态：已验证 / 未验证 / 已回退；未解决问题与后续依赖
~~~

数据变更还需记录输入hash、转换脚本版本、输出hash、受影响行/组数、unknown变化及来源证书；不能原地覆盖原始数据。checkpoint记录实际SHA256及适用源码/配置，不把旧checkpoint标为经过新修复重新训练。

无.git时使用真实基础版本、文件/树hash和最小diff，不伪造commit。修改前先备份所需源码/配置到notes下独立目录，patch只含本次相关源码与配置。凭据、完整原始数据和权重不进入Markdown或patch；不要整份打印环境变量。

服务器不提交/推送。交付可供MacBook复核提交的源码patch、说明和hash；仅有服务器绝对路径不等于补丁已经同步到MacBook。

## 8. 自主推进与本轮结束条件

继续完成所有不受阻工作，不因为一处依赖缺失而停止日志、补丁、数据计划和版本整理。无需再次确认本文已明确允许的等义修复、局部验证和离线整理。

最终交付 `/root/autodl-tmp/RepairScope-server-notes/OURS_CONTINUATION_REPORT.md`，包含事实、改动、证据、成本、未完成项及可审阅的数据计划。只选一个与证据相符的当前结论：

- `OURS_RUNNING`：实际正式Ours已启动，提供命令、tmux、日志和checkpoint路径。
- `OURS_COMPLETED`：正式Ours完成，提供完整评测和成本，其他组仍未启动。
- `SUPERVISION_UNAVAILABLE`：已有证据不足以合法恢复业务监督，列具体缺失依赖，结束本轮离线续接。
- `NO_IDENTIFIED_SCOPE_LEARNING_GAP`：有效组已被当前oracle Scope充分覆盖，当前数据无法检验GS增量；单独报告实际Predictor误差。
- `FORMAL_INPUTS_NOT_READY`：监督问题已澄清，但正式数据或明确预算不存在；交付冻结候选计划和完整费用申请，不启动无界采集。
- `METHOD_REVISION_REQUIRED`：存在可复现的代理目标或梯度机制缺口，交付最小设计及影响，不暗改机制。

若多个缺口同时存在，在报告列出全部，以上标签取当前最主要原因。结束本轮不等于宣称整个研究方向被否定。不要再循环“增加任务→Smoke→等批准”，也不要把文件准备完成写成正式训练中。
