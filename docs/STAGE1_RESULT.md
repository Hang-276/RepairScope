# RepairScope v2 Stage1终局记录

> 历史阶段说明：本文记录oracle GS采集阶段的停止。后续用户另行授权预测充分性方法修订与训练，最新 [Ours/SCPS运行记录](OURS_SCPS_STAGE1_RUN.md) 已报告训练完成；本文原始结论和费用不回写。

日期：2026-10-18，Asia/Shanghai。**结论：NO_IDENTIFIED_SCOPE_LEARNING_GAP。Stage1已结束；正式Ours、dev校准和test_normal均未启动。**

本记录转述用户提供的服务器报告。MacBook核对了报告中源码hash与已交付版本的对应关系及费用算术，未实时访问服务器、读取原始数据、账本或审计文件。完整转录字段与报告hash见 [结构化记录](results/STAGE1_RESULT.reported.json)。不再要求重新打包历史文件。

## 实际产出与科学分母

| 项目 | 报告值 |
| --- | ---: |
| train natural | 11任务，106记录 |
| dev natural | 3任务，30记录 |
| train CF | 4任务，16 cells |
| dev CF | 0 |
| train/dev合并记录 | 122 / 30 |
| H4窗口 | 73（转交报告未单列split） |
| eligible/informative可靠决策组 | 8，来自4个CF任务 |
| oracle restricted=full | 8/8 |
| GS及独立Scope梯度 | 0或浮点残差 / 0 |
| Boundary有效标签 / 非零标签 | 112 / 0 |

四个CF任务为302c169_1及07b42fd_1/2/3。302c169_2因公开Wake Up目标多义而unknown/read-only；alarm自然尝试失败、部分follow提前失败、缺少CF目标的任务均保留，未用新任务替换。证据覆盖应报告8个决策组、4个CF任务及其scenario聚类；组间独立性未建立，136条natural、16条CF和73个H4窗口不能视为额外独立样本。

fresh seed100、CPU Predictor对phone为tie，对follow偏好reference。它尚未正式训练，因此该错误不证明模型容量或最终训练失效，也不证明Scope缺少预测依赖。oracle输出覆盖与预测充分性须分开。

## 费用与停止边界

| 项目 | 实际 / 上限 |
| --- | ---: |
| natural环境 | 402 |
| CF环境 | 176 |
| Stage1新增环境 | **578 / 1,440** |
| Actor generate | 138 / 150 |
| Actor HTTP尝试 | 133 / 450 |
| 既有历史环境 | 340 |
| 合并累计环境 | **918** |
| 正式优化 / Ours GPU / test controller | 0 / 0 / 0 |

报告称账本未封存、无pending、无unknown费用；服务/tmux已停止、GPU回到0 MiB。账本状态是费用结清状态，不是继续实验的许可。**剩余862环境单位不用于追加任务、重开采集或制造GS信号。** 条件训练/评测授权因科学准入未通过而未触发，不能仅因GPU额度未使用就开跑。

保留服务器stage1数据、ledger、配置、派生脚本、Manifest和终局报告；不删除失败记录，不改旧hash，不把现有数据改名为已完成正式训练。现有数值依旧是报告值；本地不会声称已复核原始文件。

## 研究决定

当前协议在实际权重和本轮可判定组上，没有识别到可用于检验GS学习增量的oracle Scope缺口。按事先停止条件，当前v2正式实验结束在准入前；不再扩采同类任务，不降低B128、削弱prior或修改目标来追正结果。

这不是对全部AppWorld、所有参数状态或整个研究方向的证伪。其他损失在后续训练中可能改变Scope，使GS重新出现；本轮没有测试这种正则化效应。不得据此断言GS永远为零，也不能把它当作已观察到的收益。

下一步只做 [GS监督对象与方法主张核对](GS_OBJECTIVE_REVIEW.md)。若要实施新的预测充分性目标，须明确记录为方法修订，先完成数学对象、梯度路径和可证伪条件，不以工程bug名义直接替换loss。代码与实验资产保留，不回到原版方案。
