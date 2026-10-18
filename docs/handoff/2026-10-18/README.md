# 2026-10-18 服务器交付与MacBook核验

用户通过粘贴文本交付；本目录从文本重建三份补丁、完整计划、账本与Manifest。**这六个文件的bytes/SHA256均与报告完全一致**，三份修改后源码也可重建到报告的完整hash。见 [PROVENANCE.json](PROVENANCE.json)。未取得tar原文件及其余payload，不能把服务器的22项自检当作MacBook已独立复核。

| 文件 | 用途与状态 |
| --- | --- |
| [model.patch](model.patch) | 历史服务器F1/F3原补丁。F1接受；F3存在缺失history可见性问题，**不要直接应用该历史补丁**。 |
| [losses.patch](losses.patch) | F2原补丁，已核验合入。 |
| [a4_entry.patch](a4_entry.patch) | A4-only协议入口原补丁，已核验合入。 |
| [HANDOFF_MANIFEST.server.json](HANDOFF_MANIFEST.server.json) | 原交付Manifest；指向的其他文件并非都已传回。 |
| [FORMAL_DATA_PLAN.server.md](FORMAL_DATA_PLAN.server.md) | 原计划原样留存，有预算单位和步数问题，不能直接执行。 |
| [COST_LEDGER.server.md](COST_LEDGER.server.md) | 原账本原样留存，340=143+197算术正确；缺原始收据项仍保留限制。 |
| [LOCAL_INTEGRATION_RECEIPT.json](LOCAL_INTEGRATION_RECEIPT.json) | 当前MacBook源码hash及253项本地CPU工程回归结果。 |
| [F3_MASK_AUDIT.json](F3_MASK_AUDIT.json) | 精确重建服务器源码与新mask的静态CPU复现对照；不是AppWorld任务数据。 |
| [server_followup.patch](server_followup.patch) | 从服务器F1–F3快照更新至安全F3与冻结epoch调度，只修改两个源码文件并增加两个回归文件。 |
| [SERVER_UPDATE_RECEIPT.json](SERVER_UPDATE_RECEIPT.json) | 后续补丁hash及逐文件前后hash；应用前必须核对。 |

当前说明见 [交付审查与服务器更新](../../V2_HANDOFF_INTEGRATION.md)。历史原件不原地修订；审查意见、代码修复和后续验证分别记录。
