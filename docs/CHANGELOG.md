# RepairScope v2 变更记录

本文件记录MacBook端提交的研究代码、配置、文档及版本边界。每次相关提交必须同步追加条目；已有条目不覆盖，发现错误追加更正。精确提交号和逐文件差异由Git历史核验。

服务器逐项操作另记 `/root/autodl-tmp/RepairScope-server-notes/CHANGELOG_V2.md`，字段模板见 [自主续接说明第7节](CLAUDE_OURS_ONLY_CONTINUATION.md#7-每项修改必须即时写入markdown变更日志)。服务器日志、原始数据和权重不因这一要求自动加入Git。

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

以下仅转述用户提供的运行报告，不表示本仓库已经包含对应修复：

| 标识 | 问题 | 服务器报告状态 |
|---|---|---|
| F1 | BF16 goal-context scatter dtype不一致 | 已修复，见BUGFIX_LOG_V2.md及model patch。 |
| F2 | autocast下概率BCE不安全 | 已修复，见BUGFIX_LOG_V2.md及losses patch。 |
| F3 | 空history下全mask因果时序attention产生NaN | 已修复，并报告H4硬递推收据。 |

报告的基础版本为0ce975d，源码树hash为 `10ee6e5cd1e0ac9cdb59923c3698a85e9a04d3491b41fc7165ba9589d8880390`。实际合入前需取得源码patch，核验文件hash与最小回归，再在本文件追加真正的MacBook提交与验证记录。服务器路径或95项测试的转述不能替代这一同步证据。
