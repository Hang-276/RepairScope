# RepairScope v2 Ours-only 正式数据与预算计划（待批准）

路径：`/root/autodl-tmp/RepairScope-server-notes/FORMAL_DATA_PLAN.md`
时间：2026-10-18 19:00 CST
状态：**未执行采集；无正式预算；正式 TRAIN_DATA/EVAL_DATA 未冻结。**

本文件是续接说明要求的可审阅计划，不是采集授权，也不是“已等待批准”。服务器
本轮没有新增环境/Actor/优化步预算；以下数字为按已有实测成本推导的计划上限。

## 1. 官方 split 事实

| split | 任务数 | scenario 前缀数 | 官方清单 sha256 |
|---|---:|---:|---|
| train | 90 | 30 | `93d9fe71e7a2e3b7529803d4a20b604f4ebf5ae806f321081140238068189d37` |
| dev | 57 | 19 | `9fa976589300ea8905708257144d801d1604b06d85fb0181e381df8a3ba85001` |
| test_normal | 168 | 56 | `c3af41497b6f2f0860a2ff8c09b335dca527e2cf48e59b4aabdb301b6b68db8f` |

文件路径：
`/root/autodl-tmp/RepairScope/vendor/appworld/data/datasets/{train,dev,test_normal}.txt`
（按条目解析；不用 `wc -l` 代替任务数）。任务 ID 的 `_1/_2/_3` 变体共享 scenario
前缀；dev 与 test 的变体不跨 split。

## 2. 已验证/候选目标范围

### 已验证（有真实记录或离线投影）

| 目标族 | train 候选 | dev 候选 | 公开目标 | 状态 |
|---|---|---|---|---|
| Phone alarm move/disable | `302c169_1` | 无 | `phone.alarms[id].time/enabled` | `_1` CF 已验；`_2/_3` 待绑定 |
| Follow artists (genre/follower threshold) | `07b42fd_1/2/3` | 无同族 | 公开 `follower_count` before+1 | 3 任务已采；D3 离线投影已验证公式 |
| Follow artists (liked songs/albums) | `b7a9ee9_1/2/3`、`aa8502b_1/2/3` | `d4e9306_1/2/3` | 公开 `follower_count` before+1 或关系 | 目标未绑定；同公式候选 |

### 缺失（阻挡正式 Ours）

- 其余 train 任务/目标族：约 26 个 scenario、78 个任务没有可审计的公开业务目标
  映射（尤其 Spotify rating/review、歌曲集合成员、Venmo request status）。
- dev：18 个 scenario、54 个任务未绑定；`d4e9306` 仅有候选公式验证，尚无 dev
  natural/CF 记录或 frozen evaluation-only CF。
- test_normal：168 个任务没有冻结的公开 goal map；正式控制器评估不能在没有
  合法公开目标的情况下运行。
- 因以上缺口，**不能把 4 个已验证/候选任务当作完整正式训练集**。

## 3. 候选阶段（仅作预算申请示例）

以下为最小可审阅候选：12 个 train 任务、3 个 dev 任务、4 个 train scenario；
**不是完整 90 任务覆盖**，只能作为阶段 1；若目标是完整正式 Ours，需要另外
补齐 26+18 个 scenario 的目标绑定。

### 3.1 计划计数（确定值来自已验证 smoke；其余为估计）

| 项目 | 值 | 标注 |
|---|---:|---|
| train 任务 | 12 | 估计/候选 |
| dev 任务 | 3 | 估计/候选 |
| train natural records | 约 120 | 估计（每任务约 10 步） |
| train CF cells | 48 | 确定式：12 tasks ×1 pair ×4 cells |
| train independent pairs | 12 | 1/task |
| train four-cell groups | 12 | 1/task |
| train H4 windows | 约 84 | 估计（10 步轨迹→7 窗） |
| dev natural records | 约 30 | 估计 |
| dev evaluation-only CF cells | 12 | 3 tasks ×1 pair ×4 |
| dev H4 windows | 约 21 | 估计 |

### 3.2 费用（环境执行单位；1 code block + 1 API = 2）

按实测：
- Phone CF：12 units/cell（4 condition reads + query + target）
- Follow CF：10–12 units/cell
- Natural：约 2 units/step（固定公开读；Actor 生成的动作 code block 可能更高）

| 阶段 | 环境单位上限 | Actor 调用上限 | 说明 |
|---|---:|---:|---|
| train natural | 300 | 120 | 12×10 步，按 2.5/步 |
| train CF | 576 | 0 | 12×4×12 |
| dev natural | 75 | 30 | 3×10 步 |
| dev CF evaluation-only | 144 | 0 | 3×4×12 |
| **数据小计** | **1,095** | **150** | 不含失败/续跑余量 |
| 失败/续跑 20% 余量 | 219 | — | 上限 1,314 |
| test_normal 评估 | 16,800 | 33,600 | 168×100 calls；Actor 168×50×4 理论上限 |
| **候选全局上限** | **18,114** | **33,750** | 实际通常低于上限 |

**这不是已批准预算。** 完整 90 任务数据采集约需上述数据阶段的 7–8 倍
（环境约 6,000–9,000，Actor 约 800–1,200）；test 评估仍为 16,800/33,600
上限。任何正式采集必须先批准独立总预算。

### 3.3 优化步数与 GPU 时间

按实际 windows/bundles 计算，不按 transition 条数硬凑：
- 候选 train：约 84 natural windows + 48 CF windows = 132 bundles；
  micro4/accum4 → 约 33 minibatches → 约 9 optimizer updates/epoch；
  30 epochs → 约 270 updates。
- 实测 15.7 s/update（15 条 smoke，4090/BF16）：
  **候选训练约 1.2 h**；这是粗估，不承诺完整运行。
- 完整 90 任务（估计 7.5×）：约 2,000–2,400 updates → **约 9–10.5 h**。
- dev predict-eval + 一次 calibrate：约 10–30 min。
- test_normal 168 评估：约 2–6 h（上限 100 calls/task 时可能更长）。
- **候选 Ours（数据已冻结）约 4–8 h；完整覆盖约 12–18 h**；不含数据采集。

### 3.4 停止上限（请求值，不是自动授权）

- 数据环境：≤1,314 units（候选）；完整覆盖另行批准。
- 正式 test 环境：≤16,800 units；Actor：≤33,600 calls。
- 训练：≤2,400 updates；GPU wall ≤12 h；test wall ≤8 h。
- 任何超限立即停止；不自动续跑/不加任务；失败/重试计入同一上限。

### 3.5 每个计划的合法性与成本上限（schema）

- Natural plan（每个 train/dev 任务）：
  - 只用公开 API 读取/合法 Actor 动作；记录完整审计、指纹、成本；
  - 每步成本上限 4 units（code block + 最多 3 个 API），失败/重放不静默零成本；
  - 结束/失败/超时都保留 partial 与 charged cost；最多 50 步。
- CF plan（每个 task/pair）：
  - `prefix` 必须是 natural 已验证的 action/output/fingerprint；同条件 action/reference
    共享前缀；
  - c0/c1 条件只用公开读或可验证条件动作；查询必须为公开读且 exact output 匹配；
  - target action/reference 均为公开合法动作；四格全部保留；
  - 每 cell 成本上限 14 units；失败 cell 计入；不重试到成功；最多 2 条件；
  - 已有可用样例：`v2_supplement/cf_plan_supplement_v2.json`（follow）和
    `v2_smoke/cf_plan.json` / `v2_supplement/phone_c1_intervention_plan.json`（phone）。
- dev evaluation-only CF：
  - 计划先冻结并记录 plans_sha256；`counterfactual_evaluation_only=true`，
  - `frozen_query=true`，`used_for_training=false`，不得进入训练/回放/探索。
- 未绑定任务（b7a9ee9、aa8502b、d4e9306 及剩余 scenario）：
  **尚无合法计划**；必须先完成公开 instruction→目标→公开检索绑定，不能先采后编目标。

## 4. 正式数据隔离与计划要求

- train：只使用官方 train；源任务及 CF 兄弟分支同组；`used_for_training=true`。
- dev natural：官方 dev；仅预测/校准；不进入训练。
- dev CF：`counterfactual_evaluation_only=true`、`frozen_query=true`、
  `used_for_training=false`、`split=dev`、`source_split=dev`；计划先冻结 hash。
- test：不执行反事实、不分支搜索、不更新目标；只做完整 `test_normal` 官方评估。
- 所有失败/前缀/条件/查询/续跑成本保留；未知保持 Mask。
- 正式 goal map、observation map、数据转换都必须版本化并记录 hash。
- 原 `v2_smoke`/`v2_supplement` 数据不冒充正式数据。

## 5. 结论

现有文件**没有正式采集预算**，正式 train/dev/test 目标映射也存在大幅覆盖缺口，
因此本轮不能启动正式 Ours。本计划提供候选阶段的可审阅规模、费用与停止上限；
需要用户/项目明确批准后，才能进入正式数据采集与 Ours-only。
