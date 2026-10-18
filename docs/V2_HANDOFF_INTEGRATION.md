# RepairScope v2：交付核验、必要修复与服务器更新

日期：2026-10-18，Asia/Shanghai。本文件接替上一轮的“移交文件”任务；交付已收到，不再要求重新打包、重复GS诊断或重新跑Smoke。当前只推进Ours/A4，正式实验仍未启动。

## 1. 已完成的实际核验

从用户粘贴文本重建的三份patch、FORMAL_DATA_PLAN、COST_LEDGER和HANDOFF_MANIFEST，六份文件的字节数及完整SHA256均与报告完全相同。由三份patch重建的源码前后hash也全部匹配。原样留存在 [交付目录](handoff/2026-10-18/README.md)，不是只保存服务器路径。

完整tar及其余payload未实际传回，原始pytest日志、部分失败执行收据也缺失；不将服务器自检升级为本地全量验证。累计340 units的算术143+197正确，197低于前轮300补充上限；缺原始凭证的费用仍按报告标注。

## 2. 接受与替换的修复

- **F1接受：** goal projection在scatter前匹配state dtype，保持投影可微。
- **F2接受：** 五处概率BCE在autocast外用fp32计算，原logits BCE保持原状。
- **A4入口接受：** 使用既有协议包装，只运行train → predict-eval → calibrate → evaluate，固定seed100、30epochs、H4、B128/N256；不运行其他组。
- **F3替换：** 服务器`padding[:,0]=False`使所有后续query读取缺失history；该embedding含字段编码、bias与时间编码，不是零值。局部静态记录复现中，仅修改mask=false的payload即可改变预测；unified的no_grad仍有NaN。当前改为合并causal/key-padding的逐query mask，只为全遮挡query开放自身；该无效query仍不能作为有效query的key。BLite与unified均覆盖。
- **F4新增工程修复：** CF quartet不可拆分，打散后microbatch装箱数量会变化。旧代码用未打散数量估计total steps，并把它当作`max_steps=null`的硬截断。现在以独立seed RNG预生成全部epoch顺序，按实际装箱计算完整warmup/cosine；null完成全部epochs，显式max_steps只截断同一完整日程。

F4在新checkpoint和`training_schedule.json`保存冻结日程。同数据、配置和日程可exact resume；旧checkpoint仍可推理，但无冻结日程时明确拒绝冒充新版exact resume。既有Smoke checkpoint不续成正式训练。

本地CPU/Python3.12.14/PyTorch2.10.0：**253 passed in 5.38s**。覆盖BF16 dtype/概率BCE、两backbone空/部分/完整history、mask不变性、有限反向、30epochs完成、dropout下逐位一致恢复和A4-only协议。完整命令、hash见 [本地验证收据](handoff/2026-10-18/LOCAL_INTEGRATION_RECEIPT.json)。这是合成工程回归，不是服务器CUDA2.14验证或方法效果证据。

## 3. 原正式计划需要的具体修正

### 预算上限与估计分开

原表数据1,095+20%=1,314，基于natural每任务约10步、2.5 units/步，CF每cell12。但逐plan允许50步×4，CF每cell14。若15个train/dev任务和60个cells全部按该schema执行，所列动作成本最多为：

`15 × 50 × 4 + 60 × 14 = 3,840`；另加20%为4,608。

若明确冻结每任务最多10步，则对应为`15 × 10 × 4 + 60 × 14 = 1,440`。这些是按所列动作计算的方案值；不能遗漏重置、观测、前缀、失败或额外查询后直接称为已执行的全局预算。较小总上限也可以作为预注册停止点，但必须承认可能只完成partial，不能承诺全部完成。

此外，现有`collection.collect`未在每次execute前传入剩余预算；CF入口尚未向已有branch max_cost接口传入每cell/全局上限，异常charged_cost也需接入账本。**纸面每步4/cell14目前不等于代码已经强制执行。** 在采集协议与计费单位确定后，应做相应工程封装与静态/fixture回归；不通过先运行真实任务来测预算是否生效。

### Actor、环境与controller ledger分别计数

一次Actor generation返回多个候选，不是每个候选一次HTTP。当前最多3次HTTP尝试，另有execute和repair阶段。候选数4不能直接乘成33,600 Actor calls。

`168 × 100 = 16,800`对应当前controller共享ledger额度，里面包含reset、Actor、prediction、environment等费用；它不是独立环境API额度，不能与另一项Actor额度简单相加。分别列出generations、HTTP attempts、code blocks/API执行单位、预测/MC前向、共享ledger和GPU时间，明确失败与重试计入哪一项。

### 训练数量按实际window和bundle计数

代码保留每条record的起点，包括H<4尾窗。假设120 natural records加12个完整四格CF，则共有168起点window、132 bundles；其中约84个完整natural H4窗口，不能以84+48计算全部训练起点。

在上述假设、既定bundle排列、seed100/micro4/accum4/30epochs下，冻结调度为**364 updates**，不是270；以既有15.7 s/update仅作粗估约1.59小时。真实冻结数据改变时，必须重新由调度计算，不能把364当已执行结果或普适上限。4自然记录+1四格的局部回归中，旧硬上限60而完整30epochs需78，已由测试捕获。

### 子集与完整覆盖的声明分开

预先按公开语义与可测性冻结的官方train/dev子集可以构成正式阶段，不强制先绑定全部90/57任务。12任务子集不能声称90任务全覆盖，不能按GS是否为正或预测是否成功选样。完整test_normal及既定目标评分要求继续保留，不能借未绑定目标逐题人工修test。

这些修正不自动批准新采集、正式GPU时长或评估费用；正式TRAIN_DATA/EVAL_DATA和公开goal map仍未冻结。当前可靠10组的oracle代理饱和与正式输入缺口仍需分别报告，不改GS/prior/B128来制造信号。

## 4. Claude现在执行的任务

先从GitHub只下载本说明，以及交付目录中的 `server_followup.patch`、`SERVER_UPDATE_RECEIPT.json`，存入既有notes。不要clone或覆盖整个服务器源码树；不要再次打包移交已收到的文件。

1. 核对patch完整SHA256为 `e619c7cf9ed53abfc2bc9bbc16d1ed37831a682df4b97190e5d2581eab7d4632`，再按receipt核对每个目标文件的before hash；新增测试目标原本应不存在。模型起点应为服务器已报告的42705db9…a73a5，training应为ff4e9adc…326。若不符，仅定位差异，不能强行覆盖。
2. 将受影响源码备份到独立notes目录，先`patch -p1 --dry-run`，通过后应用同一补丁。核对所有after hash。losses与A4已经与交付一致，不再次重复应用原始patch。
3. 复用既有Python，在当前源码根目录运行两个新增测试文件；测试是小型CPU工程回归，允许其合成优化步骤，单列记录，不计作真实数据训练。本任务无环境/Actor调用或正式训练。若有接口差异，只修可复现的等义兼容问题并记录diff，不扩大成新一轮Smoke。
4. 按第3节修正已有正式计划的计数、单位、硬上限实现缺口和subset声明，保留原版与修订版的hash。可以静态整理既定15任务候选的公开goal/合法CF计划和缺项，不能新执行环境、增加任务或采到非零GS为止；仍未冻结的目标如实标记。
5. 每项修改即时追加 `CHANGELOG_V2.md`，记录before/after hash、原因、函数、patch、实际验证命令/结果、恢复方法和影响的checkpoint/旧收据。F3替换与F4独立编号，不能重写旧条目使其看似一直正确。

输出一次更新收据及修订费用表即可，不重新输出旧报告全文；服务器不提交/推送。原H4与GS报告保留旧代码hash，不能把本次新mask的单元测试当作它们的重验。

当前任务结束于必要工程更新与可执行计划修订。正式采集和Ours仍需已冻结输入、科学准入与明确总预算；不运行全套main、消融或B。
