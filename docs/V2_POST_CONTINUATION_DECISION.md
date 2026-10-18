# RepairScope v2：离线续接后的结论与下一步

日期：2026-10-18（Asia/Shanghai）。本文件更新当前执行状态；[Ours-only续接说明](CLAUDE_OURS_ONLY_CONTINUATION.md)中的预算、信息边界和逐项Markdown记录要求继续有效。该次离线续接已经结束，不重新执行其中的派生标签修复、GS诊断或Smoke。

## 1. 当前决定

正式Ours仍未启动，下一步先交付实际补丁与正式数据计划，供MacBook同步和审查。暂不增加环境调用、Actor调用、优化步或GPU训练，不启动其他实验组。现有材料不足以确定正式采集总额，不能把12–18小时估计转换为采集授权。

当前主标签为 `FORMAL_INPUTS_NOT_READY`，另有 `NO_IDENTIFIED_SCOPE_LEARNING_GAP`：正式输入未冻结；已有可靠组在当前权重下的oracle GS代理已饱和。这是两个独立缺口，仅补齐文件和预算不代表GS增量已经可检验。

## 2. 用户转交的服务器收据

以下来自用户提供的Claude报告，尚未读取服务器原始文件，不能视为MacBook复现或已合入源码：

- 基础提交仍为 `0ce975d933ab6a4650346609d1558e5e13e3da4c`；报告的源码树hash为 `10ee6e5cd1e0ac9cdb59923c3698a85e9a04d3491b41fc7165ba9589d8880390`。F1–F3与A4入口补丁已报告在原始提交副本上通过dry-run。
- follower_count已按公开语义核验为对应artist的user_artist_followings行数，三个任务22/23/21与公开值一致；12条记录、6个组恢复合法标签，原记录未改。
- 一次CPU fp32离线诊断：follow 6组与phone 4组的目标字段均在hard Scope内，oracle restricted与full一致。follow GS=0、梯度约2.12e-9；phone GS约1.49e-8、梯度0。不能把数值底噪当新增学习信号。
- oracle偏好action为10/10；实际Predictor中follow 6/6 tie、phone 4/4偏好reference。它们是既有Smoke/补充checkpoint上的诊断，不能当作正式训练后的科学结果。
- 已有Boundary非零标签及损失、H4硬状态递推与A4-only DRY_RUN收据；不重复执行。
- 正式清单报告train 90任务/30 scenarios、dev 57/19、test_normal 168/56；78个train任务、54个dev任务仍缺可审计业务目标，test_normal公开目标映射未冻结。`FORMAL_DATA_PLAN.md`与Manifest已交付，但MacBook尚未取得全文。
- 报告累计环境执行340 units，本轮离线新增环境/Actor/优化步均0。340是报告中的累计值，不据此推断单独300单位补充预算是否超支；需要分阶段ledger核对。
- 最终收据在notes的 `OURS_CONTINUATION_REPORT.md`、`CHANGELOG_V2.md`、`FORMAL_DATA_PLAN.md`与`FORMAL_DATA_MANIFEST.json`。摘要中的缩略hash不是可验证的完整SHA256，交付时补齐。

## 3. 本地代码核对与科学解释

`repairscope/decision.py`中的GS显式使用oracle patch，不通过真实预测Patch/Goal heads；其KL衡量完整oracle偏好与Scope内oracle偏好的差异。目标输出字段已经入选时，GS为0与Predictor预测错误可以同时成立，符合当前定义。

`repairscope/model.py`的实际排名将static prior与learned logits等相加后Top-k；prior=1不是强制纳入规则。因此当前10组全覆盖不证明所有参数、所有真实任务都无法产生GS信号。结构可训练预审计也不保证当前梯度非零。

需要保留的潜在缺口是：oracle提供选中输出字段的真值，因此不检验预测该值所需的输入条件是否充分。但现有报告尚未把“动力学仍未学会”和“预测条件被Scope遗漏”分开，不能直接断言此盲点导致当前错误，更不能据此宣称整个方法失效。

正式训练可以选择预先冻结、来源合法的官方train子集，不必为凑齐90个train任务反复采样；需明确覆盖范围、样本量依据与dev隔离。完整test_normal评测及既定目标评分要求保持。无论用子集或全量，都不能采到有GS信号为止。若未来改变GS监督对象为预测充分性，须明确记录为方法修订，不能包装成等义工程修复。

## 4. 给服务器Claude的当前任务：一次可移交交付

在已有notes目录建立独立 `v2_handoff_export/`，整理下列现有文件，不重跑已完成的实验：

1. `OURS_CONTINUATION_REPORT.md`、`CHANGELOG_V2.md`、`BUGFIX_LOG_V2.md`、完整`FORMAL_DATA_PLAN.md`与`FORMAL_DATA_MANIFEST.json`。
2. `v2_supplement/model.patch`、`losses.patch`、`a4_entry.patch`，保留实际适用基础版本、F1–F3对应函数和已有dry-run收据。若patch还包含无关变动，分开说明，不能标成全部已合入GitHub。
3. `derive_follow_labels.py`、公式核验和GS诊断的紧凑说明，以及现有相关测试、Boundary、H4、DRY_RUN收据索引。只包含必要汇总；完整原始记录、SQLite、checkpoint和权重不打包。若诊断JSON含完整原始状态，只导出汇总并记录原文件hash。
4. `HANDOFF_MANIFEST.json`：每个交付文件的相对路径、字节数、完整SHA256、用途；基础commit、实际源码树hash、修改文件的完整hash及复现解释器。保留服务器原始文件。
5. 计划中的费用按自然采集、CF各分支/续跑/查询/失败重放、train/dev、正式系统评测分别列出上限与全局合计；标明Actor/GPU计费单位。已有事实与估计分开，缺失费用明确列出，不能补造数字。若原计划已经包含，只原样交付并给出位置，不重新做一套。

生成一个小型可移交归档并报告实际绝对路径、大小和完整SHA256。服务器路径本身无法让MacBook取得文件：如当前客户端支持附件，直接提供附件；否则在回复中提供三份源码patch的完整内容、正式计划全文、Manifest与ledger汇总，过长可分段。不要仅重复摘要和服务器路径，不公开上传或在服务器提交/推送Git。

整理中的每项新增或修改仍立即追加到 `CHANGELOG_V2.md`，包括归档范围、输入来源、校验、费用0和恢复方法。归档的自校验文件避免循环hash：Manifest列出载荷文件，归档SHA在归档外单独记录。

## 5. 下一阶段的决策依据

MacBook取得真实文件后，先核验并合入必要工程修复，再依据实际计划确定一次明确的正式数据/训练预算及可检验假设。未来正式采样的任务分布、预算、覆盖及停止条件须先冻结，并允许最终仍无GS增量。

在这些条件形成之前，不执行正式Ours，不再让服务器重复同一轮“修派生标签→诊断GS→提交缺口”的工作。交付包完成即结束本次服务器任务；它不是正式实验或方法有效性的证明。
