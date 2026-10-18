# RepairScope v2：采集硬额度落地与当前任务

日期：2026-10-18，Asia/Shanghai。此前F3/F4服务器更新已完成，不重复应用、重新打包或重跑Smoke/H4/GS。当前修复独立采集额度的工程缺口，仍只推进Ours/A4。

## 1. 收据与剩余问题

用户转交的服务器更新中，model、training与两个新增回归文件的完整hash均与本地相符；此前patch和receipt的SHA256也匹配。服务器报告33项CPU合成测试通过。该报告原样保存在 [本轮记录](handoff/2026-10-18-budget/SERVER_INTEGRATION_REPORT.reported.md)，[核对说明](handoff/2026-10-18-budget/SERVER_ACKNOWLEDGEMENT.json)区分了匹配项与尚未取得原文的文件。未宣称服务器整个旧快照等同于已清理的GitHub全树。

纸面4 units/自然步、14 units/CF cell没有被采集入口强制执行，是可直接修复的工程问题。此次F5不改GS、static prior、B128、目标语义或采样分布，不批准任何新的真实费用。

## 2. F5实际行为

- `CollectionBudget`使用独立JSON持久账本。natural、CF、train/dev各命令共享同一绝对路径、run_id和固定额度；重开命令不重置已用额。非阻塞文件锁覆盖整条采集命令，路径别名归一。
- 每个环境调用前落盘pending和本次可用上限，native worker仍在实际API执行前落实限额。自然步为`min(natural_step_limit,环境剩余额)`；CF cell为`min(cf_cell_limit,环境剩余额)`，包含该cell的前缀重放、条件、查询和目标动作。
- 返回后先结算真实cost，再构建或验证transition。因此记录转换失败也不丢已发生费用。已知失败按charged_cost计一次；unknown优先于任何0值，保留已知下界与pending，封存停止。重启发现pending也停止，不能自动重试或删除账本。
- Actor的generate调用次数与HTTP尝试分开计数。每次urllib请求前预记一次尝试，失败和格式重试也受上限；cache/scripted不产生HTTP费用。generate计数含缓存调用，不冒充实际LLM生成次数。临时hook组合并恢复已有hook，不改变prompt、模型参数或cache key。
- AppWorld采集缺少显式预算时，在创建环境或调用Actor前拒绝；fixture未配置预算时保留旧工程兼容路径。无可用环境额度时，先于生成Actor候选停止。
- 采集单位为1 code block + 实际API请求。现有reset只建立world/metadata/初始公开状态，observe复制缓存，不执行world.execute，因此采集账本记0执行单位；它们仍有墙钟/CPU开销。不要把controller混合ledger的reset/prediction/Actor费率套入采集。

本地 **313 passed in 5.86s**，包括45项账本测试和15项入口/HTTP模拟集成测试。覆盖跨阶段额度、部分失败、转换错误、unknown、真实进程退出后的pending恢复、锁、Actor重试与缓存。没有真实AppWorld或Actor请求；完整套件中的fixture和合成优化不算正式实验。收据见 [BUDGET_UPDATE_RECEIPT.json](handoff/2026-10-18-budget/BUDGET_UPDATE_RECEIPT.json)。服务器真实后端仍需按实际收据报告，不能由这些模拟测试宣称科学验证。

## 3. 配置接口

`collection.budget`要求以下字段全部显式提供；额度必须有限、非负，两个Actor上限必须是整数。没有默认的正式预算。下面null表示尚未批准/确定，程序会拒绝执行，不会生成一个可继续使用的零额度正式账本：

~~~yaml
collection:
  budget:
    ledger_path: /root/autodl-tmp/RepairScope-server-notes/stage1/acquisition_ledger.json
    run_id: repairscope-v2-stage1-seed100
    environment_limit: null
    natural_step_limit: 4
    cf_cell_limit: 14
    actor_generation_limit: null
    actor_http_limit: null
~~~

批准后才能填入真实额度并建立正式账本。同一采集阶段所有train/dev/natural/CF命令必须引用同一文件。已有账本的run_id或任一上限变化会拒绝；不能换路径、改run_id或删除文件重新获得额度。新的预算授权与旧账本的恢复/核实须明确记录，不让服务器自行解除unknown封存。

预算耗尽时保留已完成transitions、失败信息、已知费用和账本快照并抛出明确错误，属于预注册partial结果；不能把失败任务丢掉后补采到成功。采集预算不包含正式训练GPU或test controller预算，三者须分开批准。

## 4. 当前给Claude的任务

1. 只下载本说明与 `docs/handoff/2026-10-18-budget/` 下的 `server_budget.patch`、`BUDGET_UPDATE_RECEIPT.json`。按receipt核对patch SHA和每个文件的before hash，备份后先dry-run，再应用一次。新增模块和两个测试原本应不存在。
2. Actor存在两种已核验import形式：规范仓库的`.candidates`，以及服务器0ce975d快照的`agent.wma_core`。同一patch不改import；receipt给出两组before/after hash。保持服务器实际对应形式，不为应用F5迁移整个源码树。其他hash不符时只定位冲突，不能强行覆盖。
3. 复用已有Python运行 `tests/repairscope/test_acquisition_ledger.py` 与 `tests/repairscope/test_acquisition_budget.py`，均为fixture/mock/故障注入，无需真实环境或Actor。旧33项和Smoke/H4/GS不重跑；不要创建正式采集world来验证限额。
4. 根据已有任务说明、公开API/schema和原始合法收据，静态整理既定12 train/3 dev候选的公开goal与计划规则。清单为train的302c169、07b42fd、b7a9ee9、aa8502b各`_1/_2/_3`，dev的d4e9306各`_1/_2/_3`。不新增任务；未能绑定的项保留unknown，不能从隐藏checker反推公开目标。
5. 固定每任务natural最多10步、每任务至多一个四格CF、单步4/cell14的候选规则，明确失败是否占槽、无合法prefix时如何保留partial。CF的精确output/fingerprint只能在合法natural发生后由实际记录实例化；现在冻结选择规则，不能伪造尚未采集的prefix收据，也不能按GS正负选前缀或换任务。
6. 修订费用申请时：15×10×4+60×14=1,440是所列执行槽的最大合计。若仍请求1,728，额外288不是自动重试授权，必须说明它覆盖哪些预定操作；没有额外操作时只是未分配余量。Actor最多150个generate槽、至多450次HTTP尝试是对应150自然步的保守上界，CF固定计划为0 Actor。全部已知失败与重放计入同一额度，不能再加免费20%。这些数字仍是请求值，不是本文件批准的预算。
7. 每项修改及时追加`CHANGELOG_V2.md`，将F5独立编号，记录文件/函数、前后hash、验证、已知/未知费用与回退。只回报F5应用收据、局部测试结果、候选绑定完成/缺失表和一张修订预算表，不重新打包或重复旧报告。

当前新增真实环境调用、Actor HTTP、正式优化均为0。正式数据、科学准入和总预算仍未冻结；上述整理完成后不得自动开始采集、Ours、其他对照、消融或B。下一次实验决定依据冻结输入与一个明确总额作出，不再追加相同工程诊断。
