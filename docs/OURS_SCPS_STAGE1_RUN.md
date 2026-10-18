# Ours/SCPS Stage1首轮训练记录

> 定位更正：用户要求的是完整官方任务划分上的正式主实验。本次122条运行仅作为小样本工程与拟合记录，不作为该主实验的完成。先前为尽快启动而直接指定Stage1数据的推进范围不足，不能靠增加epochs弥补。正式路径仍是官方train90任务采集与训练、dev57任务有效评估、test_normal168任务最终评测，具体数量按pin版本清单核对。

日期：2026-10-18。来源：用户转交的服务器运行报告；MacBook未取得本轮实际源码增量、配置、checkpoint或原始日志，以下均为报告值，不是独立复验结果。

**当前状态：Stage1数据上的Ours-only训练及dev predict-eval已报告完成；没有可用的dev业务效果指标，也没有B128预算竞争下的学习证据。** 旧[Stage1终局](STAGE1_RESULT.md)描述oracle GS采集阶段，后续用户授权了预测充分性方法修订与此次训练，两者不能混为同一次实验。

## 运行与配置

| 项目 | 报告值 |
| --- | --- |
| run | `ours_scps_stage1_seed100` |
| train / dev记录 | 122 / 30 |
| epochs / optimizer updates | 30 / 253 |
| 包含SCPS terms的updates | 84 |
| 模型 / horizon | B-lite 384 / H4 |
| Scope / candidate | 128 / 256 |
| microbatch / accumulation | 4 / 4 |
| seed / precision / GPU | 100 / BF16 / RTX 4090 |
| 训练时间 | 约4分20秒；非全部调试、dev和累计GPU费用 |
| variant | `goal_sufficient` |
| `model.goal_sufficient` / `model.scps_key_gate` | true / true |
| `loss.goal_sufficiency` / `loss.prediction_sufficiency` | 0.0 / 1.0 |
| `train.max_steps` | null |

服务器报告训练无NaN。CF batch的SCPS约0.0245降至0.00034，selector梯度最大约8.4e-3、末期约1e-5。单个CF batch至多1组。30 epochs或253次更新不等于253个独立科学样本。

## 支持覆盖与训练集拟合

16条CF记录形成8个decision groups：6个follower_count numeric eq组支持，2个phone time组因`unrepresentable_goal_value`被mask。当前报告称`22:30`不在公开copy表；这仍需核对其在原始公开指令/当前action中的来源，不能将“未入表”直接等同于“只能从未来取值”。若值来自真实next-state或监督标签则禁止倒填；合法公开来源可采用统一规则处理，不能逐组硬编码。

训练checkpoint在同一train CF上的SCPS约4.13e-5，梯度约6.14e-6；预测follower_count约23.015、24.017、22.028。这是训练集拟合，不是held-out效果。

**这些组的字段数仅13–57，小于B128，全部可入选。** 字段入选和损失下降不能证明在预算竞争下学会选择依赖；不能通过人工添字段、删字段或降低B来追认本次证据。

## dev：执行完成，指标不可测

报告H1/H2/H4窗口分别30/27/21，共168个prediction rows；本次predict-eval环境调用及优化步均为0。这168行预测不是官方test_normal的168任务。

`goal_labeled_transitions=0`、`typed_patch.n_fields=0`，goal相关mask原因为`invalid_action`。业务目标、preservation、scope覆盖、typed patch accuracy及numeric MAE均不可测；未做有效校准。**不可测保留null，不写成0、无提升或安全。**

下一步须逐条核对原始action、执行返回、before/after与转换记录，区分实际执行失败、解析/映射错误和公开目标缺失。状态Patch监督与业务目标监督分别核查；不能仅用缺少goal标签解释typed patch为零。原始失败记录保留，不将`action_valid=false`无证据改成true。

## 产物与版本边界

报告仅给出路径后缀及部分hash：

- `runs/ours_scps_stage1_seed100/training/checkpoint.pt`：`ae400ac2...`。
- `training/result.json`：`68faf2fa...`。
- `prediction/command_summary.json`：`5e7be46d...`。
- `training/metrics.jsonl`、`train.log`。

上述路径的服务器绝对根目录及完整SHA256未提供，不能重建或宣称验证。MacBook当前树尚未取得、合入本轮SCPS生产源码；不能声称仅clone当前GitHub即可复现此checkpoint。需要一次实际可转移的源码patch、配置、支持覆盖和精简结果包，不能以服务器路径或hash列表替代文件内容。大权重、凭据及原始大数据不进Git。

## 当前下一步

先并行完成源码交接与dev根因定位，核对phone公开值的合法表示；不原样重训122条记录。复用已有原始证据能等义恢复的标签另存派生版本，保留来源与hash；确实缺失的合法执行不能伪造。

下一次效果实验需要冻结有效train/dev及支持覆盖，并报告实际候选数、B128是否约束和task/scenario聚类。缺少预算竞争时仍可报告预测表现，但不能宣称动态Scope选择收益。新增真实采集必须有明确总预算；此前6小时GPU上限不自动重置，剩余环境额度不因本次训练自动获准使用。
