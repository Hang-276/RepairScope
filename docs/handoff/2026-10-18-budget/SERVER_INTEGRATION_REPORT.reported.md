更新收据
   
  - 更新说明：V2_HANDOFF_INTEGRATION.md 第 4 节
  - GitHub 增量补丁：continuation/handoff_2026-10-18/server_followup.patch
    - SHA256 e619c7cf9ed53abfc2bc9bbc16d1ed37831a682df4b97190e5d2581eab7d4632（与 receipt 一致）
  - GitHub receipt：continuation/handoff_2026-10-18/SERVER_UPDATE_RECEIPT.json
    - SHA256 b62694200fb2d0f6afbd3018a0bbf07dfc26e4236512fb48526f8a2b6277279f
  - 备份：backups/2026-10-18_server_followup/{model.py,training.py}
  - patch -p1 --dry-run 通过后应用一次。目标文件 after hash 全部与 receipt 匹配：

  ┌─────────────────────────────────────┬──────────────────┬──────────────────────────────────────────────────────────────────┐
  │                文件                 │      before      │                              after                               │
  ├─────────────────────────────────────┼──────────────────┼──────────────────────────────────────────────────────────────────┤
  │ repairscope/model.py                │ 42705db9...a73a5 │ 82368a34f1c691060c804ba8e6735a408796f3f84d12d1a828f2435f493b0eb5 │
  ├─────────────────────────────────────┼──────────────────┼──────────────────────────────────────────────────────────────────┤
  │ repairscope/training.py             │ ff4e9adc...eb326 │ 2856fe3e14349fe0d8106eec92e5c4ec7312653423a762849087dbfb061804fb │
  ├─────────────────────────────────────┼──────────────────┼──────────────────────────────────────────────────────────────────┤
  │ tests/.../test_server_fixes.py      │ 不存在           │ 9b71566a053aa333c3b42b2c84779c5997e8817a36f175a70d22ae4c9cee86f7 │
  ├─────────────────────────────────────┼──────────────────┼──────────────────────────────────────────────────────────────────┤
  │ tests/.../test_training_schedule.py │ 不存在           │ 26178da472bff69bb1742ae03f930e1cc1513309c47bf7d92458e9a788281213 │
  └─────────────────────────────────────┴──────────────────┴──────────────────────────────────────────────────────────────────┘

  - 未重复应用 losses.patch / A4 patch：
    - losses.py 仍为 633227eb...4699e
    - a4_goal_sufficient.sh 仍为 96931720...9628e
  - 当前源码树（排除 .git/runs/__pycache__/.pytest_cache）：
    f1e9bb58f0642de8af7c21e0438fab833b325cd24f449117e9628b7bfda77f65
  - 更新收据：V2_HANDOFF_INTEGRATION_RECEIPT.md
    - bytes 5711
    - SHA256 37c351b6ada1f2ea19bc20d7c076fafc5097da14eb0a2ad67a021d1965aec51e

  测试结果

  PYTHONPATH=/root/autodl-tmp/RepairScope-v2 \
  /root/autodl-tmp/RepairScope/.venv/bin/python -m pytest -q \
    tests/repairscope/test_server_fixes.py \
    tests/repairscope/test_training_schedule.py

  33 passed in 7.62s

  - CPU 合成工程回归；测试中的合成 optimizer step 单列，不计真实数据训练。
  - 按说明未重跑旧 95 tests、Smoke、H4 或 GS 诊断。
  - 旧 H4/GS 收据仍对应旧 model.py 42705... / training.py ff4e...，不能用本次单元测试视为重验。
  - 旧 checkpoint 仅可推理，不能冒充新版 exact resume。

  修订费用表

  实际已发生（既有 ledger）

  ┌─────────────────────────────────────────────────────────────────────┬─────────────────────────────────┐
  │                                阶段                                 │           环境 units            │
  ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
  │ smoke（natural + failed CF + superseded 2-cell + 4-pair）           │                             143 │
  ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
  │ bounded supplement（follow + old c1 + c1 intervention + failed c1） │                             197 │
  ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
  │ 累计实际                                                            │                             340 │
  ├─────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
  │ 本次集成                                                            │ 0（env/Actor/formal optimizer） │
  └─────────────────────────────────────────────────────────────────────┴─────────────────────────────────┘

  未保留原始 summary 的失败/被替代项：failed CF 10、superseded 2-cell 27、failed c1 9；仍按 charged cost 标注。

  修订后的计划上限（请求值，不是批准预算）

  15 个 train/dev 任务、60 个 CF cells：

  ┌───────────────────┬─────────────────┬─────────────┬───────┬────────────────┬───────┐
  │       方案        │     natural     │     CF      │ 小计  │ +20% 失败/重放 │ 上限  │
  ├───────────────────┼─────────────────┼─────────────┼───────┼────────────────┼───────┤
  │ 冻结 10 步/task   │   15×10×4 = 600 │ 60×14 = 840 │ 1,440 │            288 │ 1,728 │
  ├───────────────────┼─────────────────┼─────────────┼───────┼────────────────┼───────┤
  │ schema 50 步/task │ 15×50×4 = 3,000 │ 60×14 = 840 │ 3,840 │            768 │ 4,608 │
  └───────────────────┴─────────────────┴─────────────┴───────┴────────────────┴───────┘

  - 共享 controller ledger：16,800（168×100；含 reset、Actor、prediction、environment，不是独立环境 API 额度）。
  - Actor generations / HTTP attempts：未测；一次 generation 可返回多个候选，不能把候选数 4 当成 HTTP 次数；每次最多 3 次 HTTP 尝试。
  - 训练日程：候选假设 168 起点 windows / 132 bundles / seed100 / micro4 / accum4 / 30 epochs → 364 updates（不是 270），按 15.7 s/update 粗估 1.59 h。
  - 测试评估：共享 ledger 上限 16,800，估计 2–6 h。
  - 12 train / 3 dev 是 stage-1 subset，不冒充 90/57 全量覆盖；选择不得按 GS 结果。
  - 硬上限实现缺口仍存在：collect 未在 execute 前传剩余预算，CF 未把 per-cell/global max_cost 传入 branch，异常 charged_cost 未完整接入统一 ledger；纸面 4/14
    上限尚未由代码强制执行。

  状态

  - 修订计划：FORMAL_DATA_PLAN.md SHA256 74d846f9...f288
  - 修订 Manifest：FORMAL_DATA_MANIFEST.json SHA256 60cf0ea7...9748
  - 原版备份：backups/2026-10-18_plan_pre_integration/
  - 正式 Ours：未启动
  - A0–A3、消融、B：未启动
  - 服务器：未提交/未推送 Git
  - 未重新打包导出包；旧 v2_handoff_export 保持原样。