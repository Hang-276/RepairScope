# RepairScope 服务器工作约定

当前任务是把已经确定的 RepairScope 实现接入真实服务器，完成既定主实验和消融。研究 idea 已冻结。请先完整阅读：

1. `README.md`
2. `docs/REPAIRSCOPE_PROPOSAL.md`
3. `docs/REPAIRSCOPE.md`
4. `docs/SERVER_RUNBOOK.md`
5. `docs/IMPLEMENTATION_COVERAGE.md`
6. `docs/LOCAL_VERIFICATION.md`
7. `docs/CLAUDE_SERVER_HANDOFF.md`

执行细节以服务器交接文档为准；其限制同样适用于子代理、自动修复和续接会话。

## 已有证据与执行范围

- MacBook 本地 **84 项工程测试已通过**。已有最小 CPU FP32 smoke、H=1/2/4 接口、阈值入口和反事实更新闭环，证据全部为 synthetic fixture。
- 尚未执行真实 AppWorld、Actor 模型服务、CUDA/BF16 或正式 GPU 主实验。不要将本地工程指标写成真实提点、TGC/SGC 或服务器就绪证明。
- 在服务器安装并对齐必要环境后，只做确有必要的最小接入 smoke，随后直接执行既定主实验及消融。不得追加 pilot、网格搜索、多轮小规模有效性验证，或因没有提点自行改 idea。

## 冻结的方法

Frozen Actor 与 WMA 共享候选聚合/稳定重排；单一 B-lite Predictor 包含动作条件关系注意力、字段级因果时序注意力、Dynamic Scope、类型化 State Patch、共享参数 H4 rollout、Boundary 和 Escape。预防、合法只读检查及含 noop 的修复共用该预测器。

不得自行改变研究机制、架构、损失及权重语义、采样公式、标签、目标谓词含义、指标及原生聚合口径、数据划分、控制决策语义或实验公平性。不得引入额外 planner/teacher、训练 Actor、以隐藏状态补充合法观测、把 unknown 当不变或假值，或以 fixture 替代真实结果。

主 A 的 A0–A5 使用冻结输入与统一资源口径；A1–A5 共享一次采集的自然轨迹及反事实数据。主 B 的四种策略使用对应种子同一 A3 checkpoint、同一合法候选池、相同损失和真实执行预算。前缀重放、条件构造、查询、失败及续跑均计费。官方测试禁止更新及分支搜索；冻结查询结果不得回流训练。

## 工程修复与机制问题的界线

可以自主处理路径、安装、依赖、服务连接、公开 API/schema 的等义适配，以及能证明不改变上述语义的工程 bug。修复后只运行定位该问题所必需的最小回归，并使所有相关实验保持同一修复版本和预算规则。

**如果发现机制 bug，或修复可能改变任何冻结项：停止受影响实验，不得写补丁、改配置、删样本、降标准或换目标来绕过。** 将事实、最小复现、拟议修复、语义和公平性影响、已消耗预算写入仓库外 Markdown，等待用户明确审批。不能确定是否影响机制时，按机制问题处理。继续与该问题无关的只读分析和证据整理即可。

公开映射缺失时，只能依据真实公开 API/schema 和公开任务要求做语义对齐；不得从隐藏 checker 反推在线目标，不得伪造目标或结果。无法明确等义映射时停止相关链路并询问用户。

## 仓库外修复日志

默认位置为 **`repo.resolve().parent / "RepairScope-server-notes" / "BUGFIX_LOG.md"`**。创建前必须解析真实绝对路径，同时检查目录和日志文件经符号链接解析后均不在仓库内。机制问题写在同一仓库外目录的 `MECHANISM_ISSUES.md`，采用同样检查。

每次工程修复都记录：初始 commit/配置与数据标识、问题和分类依据、修复前证据、修改文件、仓库外 patch 位置及 diff hash、修复后最小验证、执行命令、修复 commit（如已提交）、已消耗预算及受影响结果。日志和 patch 不得 `git add`，不得写入凭据。详细模板见交接文档。

保留失败收据和已有结果；使用独立输出目录与运行 namespace，不能删除或覆盖旧结果使其看似成功。不要擅自 push、merge、重置或覆盖远端；本次仓库发布授权不能解释成服务器后续修复的持续发布授权。
