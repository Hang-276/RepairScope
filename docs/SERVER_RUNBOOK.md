# 服务器运行：一次必要 smoke 后直接主实验

**当前只运行一个实验：A1 普通预测器 vs A3 完整 RepairScope，固定种子100。** 两组共用一次采集的数据、同一评测集和预算。完成主对照汇总后停止等待用户决定；完整主 A、主 B、其它种子和全部消融暂不启动。该阶段安排优先于下方保留的完整研究规划，不改变方法机制。

## 1. 安装与固定输入

使用 Linux、Python 3.11+。在仓库根目录执行：

```bash
bash scripts/repairscope/setup_server.sh
source .venv/bin/activate
appworld install
export APPWORLD_ROOT=/path/to/appworld
appworld download data
export ACTOR_MODEL=your_frozen_served_model
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
```

Actor 使用 OpenAI 兼容服务；认证从 `OPENAI_API_KEY` 读取。模型参数不会更新。可将 Actor 采集与 Predictor 训练分时运行，避免两模型常驻显存。安装脚本只安装环境，不启动模型服务、采集或训练。

复制 `configs/repairscope/base.yaml` 为服务器配置文件，设置以下实际数据输入：

```yaml
base: base.yaml
environment:
  root: /path/to/appworld
  observation_map: /path/to/public_observation_map.json
data:
  train: /path/to/shared_train.jsonl
  dev: /path/to/recorded_dev.jsonl
  goal_map: /path/to/public_goal_map.json
actor:
  model: your_frozen_served_model
```

`base` 相对于配置文件所在目录解析；放在 `configs/repairscope/` 下可直接继承 `base.yaml`。观测/目标映射格式见 `repairscope/fixtures/README.md`。映射是公开字段与训练标签的对齐数据，不需要修改 Predictor 代码。保留官方任务划分，所有条件变体与兄弟分支跟随源任务。

如果本地已完成工程 smoke，服务器不再开展小规模提点试验。环境安装问题按报错修复，之后直接使用固定配置开始主实验。

## 2. 一次采集并冻结共享训练数据

```bash
export CONFIG=configs/repairscope/server.yaml
export RUN_ROOT=/path/to/experiment-runs
bash scripts/repairscope/collect_train.sh
export PROBE_PLANS=/path/to/legal_train_branch_plans.json
bash scripts/repairscope/collect_counterfactual.sh
python scripts/repairscope/build_training_data.py \
  --inputs "$RUN_ROOT/natural/transitions.jsonl" "$RUN_ROOT/counterfactual/transitions.jsonl" \
  --output /path/to/shared_train.jsonl
```

自然轨迹来自冻结 Actor；反事实计划包含相同起点前缀、合法 a/reference，以及可执行并可查询确认的条件。计划准备示例可通过 `collection.extract_prefix(records, stop_step)` 读取真实收据。不要将测试任务、四格输出真值或隐藏 checker 加入 Actor 输入。

dev 普通冻结轨迹仅供主评测的 H=1/2/4 预测指标与一次阈值校准，不回流训练；禁止在 dev/test 做反事实探索。它是主结果的一部分，不是新的前置有效性试验。

```bash
bash scripts/repairscope/collect_dev.sh
export EVAL_DATA="$RUN_ROOT/dev-natural/transitions.jsonl"
```

把服务器配置的 `data.dev` 也指向该文件，以便训练入口检查 train/dev 来源隔离；启动脚本通过 `EVAL_DATA` 接入各配置的 H=1/2/4 主预测指标评测和一次阈值选择。

## 3. 当前唯一主对照：A1 vs A3

```bash
export TRAIN_DATA=/path/to/shared_train.jsonl
export EVAL_DATA=/path/to/recorded_dev.jsonl
export SEEDS=100
bash scripts/repairscope/run_primary_experiment.sh
```

仅训练/评测 A1 与 A3，在同一数据上运行相同训练计划，随后使用同一正式任务列表和冻结 checkpoint 评测。当前正式 eval 固定为 **AppWorld 官方 `test_normal` 全集**，从 `$APPWORLD_ROOT/data/datasets/test_normal.txt` 读取，不设置自选 `TASK_FILE` 子集；两组保留相同任务列表及其 hash。训练仅用 `train`；`dev` 仅供 H=1/2/4 和一次阈值选择；本阶段不运行 `test_challenge`。正式任务不使用额外分支搜索。结果保存为 `a1_dynamics/seed_100` 与 `a3_full/seed_100`。

设置 `EVAL_DATA` 后，每个训练配置先产生主结果的 H=1/2/4 预测指标，再仅对 H=1 的 dev 审计记录选择一次风险阈值。默认检查率上限25%，在该上限内最大化范围越界召回；结果存为 `calibration/calibration.json`，绑定 checkpoint hash。正式任务只读取这份阈值，不使用测试结果调参。未提供 dev 记录时使用首轮固定阈值0.5，并且不宣称已校准。

两组完成后汇总 TGC/SGC、受保护目标损害、修复损害、H=1/2/4 与成本，报告 A3 相对 A1 的绝对变化。标明单种子限制，不解释成跨种子稳定性。**此时停止，不自动追加任何实验。**

## 4. 保留的后续实验入口：本阶段不运行

以下独立配置与消融脚本完整保留，仅在用户后续决定后使用：

```bash
bash scripts/repairscope/a3_full.sh
bash scripts/repairscope/a4_no_boundary.sh
bash scripts/repairscope/a5_no_escape.sh
```

完整主 A 和消融入口（当前不启动）：

```bash
bash scripts/repairscope/run_main_a.sh
bash scripts/repairscope/run_all_ablations.sh
```

当前入口的纯命令连通性检查可使用 `DRY_RUN=1 SEEDS=100 bash scripts/repairscope/run_primary_experiment.sh`；这不会加载数据、模型、环境或 Actor，也不生成任何实验结果。

### 后续主实验 B：相同环境执行预算的探索比较（暂缓）

```bash
export PROBE_PLANS=/path/to/same_legal_train_candidate_pool.json
bash scripts/repairscope/run_main_b.sh
```

默认每种策略从对应种子 A3 的同一 checkpoint 出发；`CHECKPOINT` 可显式固定共同起点。`probing.budget_calls` 固定完整真实执行成本，四格包括四个独立分支。Actor 与 Predictor 计算单独记录，不能通过隐去续跑、失败调用或合法检索扩大预算。

若后续获准运行主 B，保持主 A 的 `TRAIN_DATA` 设置，以同一数据初始化经验回放。`ablate_no_probing.sh` 使用相同起点、零探索预算，结果单独保存在 `ablate_no_probing/seed_<种子>/`。主 B 与不探索消融当前均不执行。

默认起点会自动继承 A3 对应种子的 dev 阈值收据；自定义起点可用 `CALIBRATION_FILE` 指定对应收据。B 持续更新过程中阈值固定，保留初始来源，不对查询任务重新拟合。

设置 `probing.query_task_ids` 指定不参与采样及更新的冻结查询任务。累计预算25%、50%、75%、100%处记录模型/目标损害曲线及 checkpoint。查询结果不能进入回放池。`update_every` 控制真实新记录积累后的周期更新；旧样本由受限经验回放池保留。

## 5. 产物与续训

训练产物位于 `<RUN_ROOT>/<配置>/seed_<种子>/training/`，正式任务结果位于同级 `evaluation/`；若提供 `EVAL_DATA`，固定起点预测结果位于 `prediction/`。探索结果位于 `probe_<策略>/seed_<种子>/`。

续训使用完全相同的配置和数据：

```bash
python -m repairscope train --config "$CONFIG" --variant full --seed 100 \
  --data "$TRAIN_DATA" --resume /path/to/training/checkpoint.pt \
  --output-dir /path/to/training
```

数据/config不一致会拒绝精确续训，避免把重新训练误写成续训。当前仅报告 A3 相对 A1 的绝对提点、受保护目标损害及成本，保留无效/失败执行收据。无论有无提点，汇总这一组主对照后停止等待用户决定，不自动追加其它实验。
