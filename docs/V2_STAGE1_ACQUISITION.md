# RepairScope v2：一次固定预算的Stage1采集决策

日期：2026-10-18，Asia/Shanghai。**状态：待用户批准本次新增采集预算，尚未启动。** 本文件是完整可审阅的单次执行范围；[机器清单](STAGE1_ACQUISITION.proposed.json)中的数值不等于预算已经获批。

## 1. 已完成，停止重复检查

用户转交F5应用收据：patch与receipt SHA、五个目标文件after hash均符合预先验证的服务器legacy变体，报告60项fixture/mock/故障注入测试通过。MacBook核对记录及原样报告见 [F5确认](handoff/2026-10-18-stage1/F5_ACKNOWLEDGEMENT.json)。它不等于对服务器实时探测或真实backend限额实测。

F1–F5、A4-only入口、既有Boundary/H4与离线GS判定已有对应工程收据，不再重跑旧33/60/95项、同一Smoke、同一GS诊断或重新打包。无需再增加一轮准备性效果实验。

当前15任务中4个已有精确绑定，10个有待自然公开观测实例化的规则，1个有多义目标。采集前必须冻结选择规则；不要求在合法公开读取发生前就知道所有实体ID或CF指纹。

## 2. 本次申请的唯一总额

| 项目 | 本次新增上限 |
| --- | ---: |
| 环境执行单位 | **1,440** |
| Actor generate调用（含缓存） | **150** |
| Actor HTTP尝试（含失败/格式重试） | **450** |
| 每任务natural尝试 | 10步 |
| 每natural步 | 4环境单位 |
| 每任务CF | 至多1个四格pair |
| 每CF cell（全部前缀/条件/查询/目标） | 14环境单位 |
| 环境动作/CF重跑、替换任务、额外pair | 0 |
| 正式训练优化步/GPU额度 | 本次不申请 |
| test_normal/controller额度 | 本次不申请 |

固定15×10×4+15×4×14=1,440；不另加20%余量，不请求1,728或4,608。失败、重放、公开读取均在同一个额度内。Actor沿用每次generate最多3次格式HTTP尝试，并计入450，不能增加额外generate重试。已有340环境单位另列，若本次用满，按既有报告累计至多1,780；部分历史费用缺原始summary的限制继续保留。

这是上限，不是承诺产出150条natural、60个有效cells或364次训练更新。真实window/bundle数只能采后计算。当前窄子集可能仍没有可检验的GS增量，不能据此补任务或改方法。

## 3. 冻结任务与公开目标规则

顺序：train的302c169、07b42fd、b7a9ee9、aa8502b各`_1/_2/_3`，再dev的d4e9306各`_1/_2/_3`。不得跨split、替换或按结果重排。选择依据为已声明的公开业务语义和可测性，不按GS正负或Predictor正确率。

批准后、首次执行前，将现有`STAGE1_TARGET_BINDING.md`、修订计划与Manifest的实际完整hash记录到本轮run manifest。除预算、细化规则和明确的unknown外，不静默更改原任务语义。机器清单记录了转交的文件hash；MacBook没有声称已收到这些完整原文件。

- 冻结instruction→实体选择谓词、初始时点、集合完整性/分页/去重、需要保持的条件及每个合法动作/参考/条件规则。具体ID仅从预算内公开读取机械解析；目标字段必须有合法公开来源。
- `302c169_2`若仍无法用公开指令和观测区分两个Wake Up，标unknown/read-only，不任选、不写该任务的alarm、不执行其CF、不用checker反推。占原任务槽，未用额度不转成新任务。
- 所有目标集合依赖初始公开业务状态。分页不完整、身份不明、引用或集合补集不能证明时，相关目标unknown；不把未读对象当不存在。
- `follower_count±1`不是无条件等同于follow/unfollow。须证明当前用户与artist、初始关注状态、关系唯一性、API幂等及无其他关系变动。已关注对象可能已满足业务目标，不应强行要求再+1；完整目标集合与保持条件不能由总人数代替。不能证明等义时只保留动作效果诊断或mask该监督，不偷偷把人数代理改成任务成功。
- 不在动作后滚动重设“before+1”目标，不把隐藏审计字段补到在线state/history。训练/dev的隐藏审计只用于已允许的离线标签。

## 4. 获批准后的一次执行

**本节只有在用户明确批准第2节新增预算后生效。** 批准前只保存文件，不创建正式账本、world或Actor调用。

1. 复用当前源码、Python、已有AppWorld和已选FrozenActor配置，记录版本/hash。不得切换默认云模型、provider、重装环境或下载新模型。若实际Actor配置/接口缺失，报告缺项停止，不临时换模型或另开连通性采样。
2. 使用机器清单中的固定run_id与同一持久ledger路径，配置环境1,440、步4、cell14、generate150、HTTP450。所有train/dev/natural/CF命令共享它，不因新目录获得额度。各任务独立输出，失败也保留。
3. 按固定顺序采train natural、dev natural。合法的读取、分页和失败尝试都占最多10步与generate槽。具体目标由已冻结规则解析；遇到未能唯一绑定的目标保留unknown，不因预期GS信号选择动作。首次正常计费调用中核对F5传入额度和实际cost，不另开Smoke；普通调用收据不冒充刻意越界测试。
4. 从natural中按预冻结规则选择首个具备完整公开目标、合法动作/参考和可负担前缀的决策点。以**动作前可见信息与已发生前缀成本**选择，不看未来效果、GS、预测误差或是否成功。没有此点就保留partial和原因，不换prefix采到可用。
5. 精确CF prefix的action/output/fingerprint来自实际合法natural记录。c0/c1干预与只读查询必须在任务规则中预先定义；每cell14包含全部重放、条件、查询和target，不免费充值。一次pair调用失败则保留已完成cells，未完成格mask；不重跑整个pair。已知费用失败可按既定顺序进入后续尚未尝试的任务槽，unknown成本或账本封存立即停止整个阶段。
6. 先train CF，后dev evaluation-only CF。dev规则和模板在执行前冻结，精确prefix机械实例化后记录plans SHA；`counterfactual_evaluation_only=true`、`frozen_query=true`、`used_for_training=false`、split/source_split均dev。dev不参与训练或自适应探索。
7. 一轮结束后只做一次新数据离线审计和Manifest冻结：分别列unknown、eligible、informative、结构trainable、实际GS及独立Scope梯度；使用何种checkpoint/初始化与当前代码hash明确标出。无optimizer更新。旧Smoke和supplement可作历史证据，不能重命名为正式数据。
8. 若可靠组仍oracle饱和，输出`NO_IDENTIFIED_SCOPE_LEARNING_GAP`；若标签不足输出对应覆盖缺口。保留全体任务分母和失败成本，不改prior/B128/GS或追加任务。即使数据可用，本阶段也不自动启动训练、calibrate、test、对照、消融或B。

## 5. 日志与终局交付

每项代码/配置/目标映射/数据转换修改即时追加notes下`CHANGELOG_V2.md`，记录原因、文件/函数、前后hash、实际验证、成本和回退；保持原数据与unknown，派生输出单独版本化。预算账本不能随代码回退删除或重置，恢复不自动重放外部调用。

一次交付：固定任务与规则hash、批准记录、共享账本、实际各类费用、train/dev及各兄弟分支隔离、数据/转换/checkpoint/代码hash、有效四格与H4窗口数、一次监督审计及能否进入后续Ours训练的结论。部分完成明确标partial；没有正式训练就不报告Ours已启动。

本次不要求再打包历史补丁或重新输出此前报告。源码提交/推送仍由MacBook完成，服务器只记录配置、产物与必要patch。
