# RepairScope v2：服务器 Claude 实验执行任务

本文件是完整、自包含的服务器交接任务。用户已决定放弃原版 A1 vs A3 方案，直接执行 RepairScope v2。代码由 MacBook 提交并推送到 GitHub；服务器只 clone 指定分支、核对代码版本并执行实验，不在服务器提交或推送代码。不要启动旧版入口，也不要覆盖原仓库、环境、数据、checkpoint 或其他任务。

## 1. 代码与已有状态

- 仓库：`https://github.com/Hang-276/RepairScope.git`
- 分支：`feat/goal-sufficient-scope`
- v2 Goal-Sufficiency 核心代码提交：`62fd96e54891b799cf0819ae5e0c74253363f040`（必须是服务器 clone 后分支 HEAD 的祖先）；同时记录 clone 时实际 branch、完整 HEAD SHA 和工作区状态。随该分支发布的 proposal 和本提示词也须存在。
- 来源分支：`feat/goal-sufficient-scope`。只有指定分支及上述核心代码提交已出现在 GitHub 后才开始接入；若不可见或祖先检查失败，报告代码同步阻塞，不用旧版代码启动。
- 原服务器仓库：`/root/autodl-tmp/RepairScope`，原分支 `rs/primary-a1-vs-a3`。
- 已取回的服务器修复提交：`b3b7df93b18ca73ac9f29e4bf69ca989027b89b0`，包含复合目标表达式、compact audit、Actor 计费、匹配前缀与监督诊断修复。
- 新版准备目录：优先 `/root/autodl-tmp/RepairScope-v2`。若该路径已存在，先只读检查其内容；不得覆盖、清空或重置。选一个未占用的明确 sibling 目录（例如 `RepairScope-v2-<branch>-<date>`），并在后续命令中一致使用该绝对路径。
- 原记录目录：`/root/autodl-tmp/RepairScope-server-notes`。继续沿用该目录，不把凭据、原始数据、模型权重或运行日志提交 Git。
- 已核对的现有 Python：`/root/autodl-tmp/RepairScope/.venv/bin/python`；Actor 环境位于原仓库 `.venv-actor/`，不复制。
- 已核对的现有配置：`/root/autodl-tmp/RepairScope-server-notes/server.yaml`；AppWorld 根目录：`/root/autodl-tmp/RepairScope/vendor/appworld`；公开观测 map：`/root/autodl-tmp/RepairScope-server-notes/data/observation_map.json`，审计模式 compact。原配置的 base 指向旧仓库，派生新版配置时必须改为新版 base。

先读本文件、`docs/RepairScope_v2_Proposal.md`、`repairscope/decision.py`、`repairscope/model.py`、`repairscope/config.py`、`repairscope/prediction_eval.py` 和 `scripts/repairscope/run_goal_sufficient_main.sh`。旧版 A1 vs A3 执行限制已由用户明确替换；数据隔离、checker 隔离和真实计费规则继续适用。

2026-10-18 服务器先前报告显示没有正式采集、训练、dev 标定或 test 评测；用户已明确表示旧版实验已经停下。实际接手时仍只读核对当前进程、GPU 和 tmux 状态，记录是否与报告一致。不要结束任何其他任务或进程；原仓库未跟踪的 `.venv-actor/`、`vendor/` 是现有环境，不是需要清理的内容。

## 2. 第一个实际阻塞：业务目标绑定

服务器最新报告不是“目标已可训练”：`goal-extractor-v2-2026-10-18c` 在 train/dev 得到 **0 个 full business goal**；train 19 任务/30 个 partial、dev 3 任务/3 个 partial。具体实体 ID、liked/yesterday/coworker 等 qualifier 尚未绑定。`complete` 表示声明完成，未带可验证正确答案的 `answer` 表示提交非空，均不能当作业务成功。

阅读现有 `PROGRESS.md`、`MODIFICATIONS_SUMMARY.md`、`GOAL_MAP_PROVENANCE.md`、`SUPERVISION_TRACE_REPORT.md` 和 `prep/` 中的目标报告。只集中修正公开目标提取与合法字段绑定，不重复已完成的存储/计费/审计全面加固。

目标只能依据公开 instruction、API 文档、合法 API 返回和既有历史构造。实体 ID 在运行时通过公开查询绑定，禁止从隐藏数据库/checker 回填在线目标。保留 instruction 依据、对象、量词、数值/时间限定与保护生效条件；不完整/不可判定的目标继续 unknown。通用提取器只在 train/dev 开发并冻结，test 自动应用，不逐题查看或修补 test。

不能靠启用旧 conservative map、增加 complete/answer 数量、强制 Actor 写入或修改 mask 让监督非零。修复使用已批准的表达式能力；如必须改变任务语义或引入新机制，报告具体缺口并停止受影响链路。

已有一次业务写入 Smoke 已证明 `spotify.playlists[344].title` 能进入 audit_mask/target_changed/Patch；不再重复此检查。旧 Smoke 的 Boundary 两个非零标签来自 supervisor，实际公开候选交集的业务 Boundary 非零为 0，不能冒充已具备业务条件化监督。

## 3. 方法必须按代码实际实现

Actor 参数冻结，B-lite 关系块、时序注意力、类型化 Patch 和 H=4 递推继续复用。v2 学习固定字段预算下足够支持目标保护决策的 Scope，不引入 RSI、Memory Agent、额外大模型或 RL Explorer。

完整离线目标效用采用与控制器统一的 progress、受保护目标损害与动作成本权重。合法同前缀 action/reference 分支提供参考动作偏好；受限 Scope 的 oracle-patch 决策与充分信息参考之间做蒸馏。该损失直接经过 Scope gate 更新实际选择器，不能替换成仅训练辅助头。

原子谓词可用精确可加字段效用；复合谓词必须由现有 Goal Evaluator 对 Scope 混合状态实际求值，不能把非线性量词/聚合目标当作独立字段效用相加。复杂依赖、部分审计无法证明闭世界覆盖、当前 evaluator 未执行的 where 等情况应明确 mask/报告，不能扩大语义来制造监督。

硬前向始终遵守 Top-k 和固定覆盖预留；反向使用固定总质量 sigmoid 松弛的直通估计。它是有偏梯度估计，不是离散最优化证明。训练同时保留状态变化覆盖辅助监督。Boundary 学条件化作用支持，State Escape 学遗漏变化，Goal-Relevant Escape 只用有证据的保护判断变化；三者不能互相代替。

检查 `goal_supervision_audit.json` 中 eligible、informative、trainable group、可检索关键字段和预算竞争覆盖。所有有效组关键字段不可检索、预算覆盖全部候选、标签未知或决策无差异时，不得声称 Goal-Sufficiency 已训练。代码在没有可训练组时拒绝正式 v2 训练。

原始审计、未来状态和 goal_after 只能作为离线标签。推理只接收公开当前状态/schema、当前动作、公开目标和既有历史。不得调用隐藏测试 checker 做在线决策；AppWorld 原生评分在独立进程/隔离状态中完成，不污染活跃世界。

## 4. 复用环境与安全同步

复用已有 Python 环境、AppWorld 数据、Actor 权重、依赖和缓存。已有环境未损坏时不重装依赖，不重新下载数据/模型，不升级系统包。不改变全局 Claude、网络或模型服务配置。

AppWorld 已记录 revision `42b5bcf3cd334fee33f0c37c02070a9f5807add5`、数据 0.2.0。任务清单实际为 train 90 / dev 57 / test_normal 168；使用按条目解析后的官方清单及其 hash，不能用无尾换行文件的 `wc -l` 代替条目数。Actor 已记录 Qwen3-8B-AWQ、vLLM 本地 8000 端口、32k 上下文、非 thinking 模板；核对实际启动命令和模型 revision，复用原有服务配置。

### 从 GitHub 安全 clone

只有 MacBook 已将 `feat/goal-sufficient-scope` 推送至 GitHub 后才执行。先检查目标路径是否存在；只有路径不存在或为空目录时才 clone，不复用原服务器 checkout：

```bash
if [ -e /root/autodl-tmp/RepairScope-v2 ]; then
  echo '目标目录已存在；先检查并选择未占用的 sibling 目录，禁止覆盖。'
  exit 2
fi
git clone --branch feat/goal-sufficient-scope --single-branch \
  https://github.com/Hang-276/RepairScope.git \
  /root/autodl-tmp/RepairScope-v2
```

如果目录存在但为空，可在确认其确为空后选择另一个不存在的目标路径并替换上述目标参数；不要删除已有目录来腾位置。Clone 后在新版仓库先运行只读核对：

```bash
cd /root/autodl-tmp/RepairScope-v2
git status --short
git branch --show-current
git rev-parse HEAD
git log -5 --oneline
```

并确认 MacBook 核心提交包含在当前分支历史中：

```bash
git merge-base --is-ancestor 62fd96e54891b799cf0819ae5e0c74253363f040 HEAD
test -f docs/RepairScope_v2_Proposal.md
test -f docs/CLAUDE_GOAL_SUFFICIENT_EXPERIMENT.md
```

上述任一检查失败，报告同步阻塞并停止；不得从原服务器仓库复制源码补齐。

不要 `reset --hard`、覆盖原仓库、删除分支、复制整套环境/权重。新版必须包含上述服务器修复；若原服务器还有更新，只将相关源码修复按 commit 合并，保留来源并记录最终 HEAD，不从运行目录复制数据或凭据。

从原仓库现有配置确认实际环境 Python、APPWORLD_ROOT、模型、观测 map、goal map、Actor 参数及各个数据路径。为新版创建一个明确命名的服务器配置文件（位于新版目录或已有 notes 中），复用合法绝对路径。配置的 `base` 必须指向真实存在的 `configs/repairscope/base.yaml`；不把旧配置中的 A1/A3 方法覆盖传给所有 variant。密钥沿用现有安全环境变量，不打印其值。

使用原有 Python 时显式设置新版模块路径，避免其 editable 安装仍导入旧仓库：

```bash
export REPAIRSCOPE_PYTHON=/root/autodl-tmp/RepairScope/.venv/bin/python
export PYTHONPATH=/root/autodl-tmp/RepairScope-v2
export APPWORLD_ROOT=/root/autodl-tmp/RepairScope/vendor/appworld
"$REPAIRSCOPE_PYTHON" -c 'import repairscope; print(repairscope.__file__)'
```

输出必须位于新版目录；不用重新 `pip install` 修补这个路径问题。

## 5. 唯一一次必要工程 Smoke

只对新版尚未验证的真实链路做一次 Smoke：1–2 个 train 任务、至少一组合法同前缀反事实、2–5 次优化更新、一次 H=4 硬递推。复用既有有效记录优先；不为凑损害正例扫描任务。

`python -m repairscope smoke` 是 **CPU synthetic fixture**，不能当作真实 AppWorld/GPU Smoke。真实 Smoke 用已有真实 train 记录的固定小子集，和实际服务器 Python 执行现有 `train` 入口，例如：

```bash
"$REPAIRSCOPE_PYTHON" -m repairscope train --config "$CONFIG" \
  --variant goal_sufficient --data "$SMOKE_TRAIN_DATA" --seed 100 \
  --set train.epochs=1 --set train.max_steps=5 --set train.warmup_fraction=0 \
  --output-dir "$SMOKE_OUTPUT"
```

上面的路径变量须由实际产物填入；Scope Budget 保持正式 128，candidate_limit=256。若这 1–2 个任务可检索候选数少于预算、没有有效业务目标或合法决策对，明确报告无法验证新增梯度，不缩小真实 Smoke 预算来冒充正式条件。MacBook 上当前 Python 缺少 PyTorch，相关测试无法在该环境收集；Python 编译和 shell 语法检查已通过。服务器使用原有项目 Python 环境，一次执行下列目标学习、目标决策、协议和 dev 反事实评估测试；不重装依赖或重复运行多轮试验：

```bash
"$REPAIRSCOPE_PYTHON" -m pytest -q \
  tests/repairscope/test_goal_decision.py \
  tests/repairscope/test_v2_learning.py \
  tests/repairscope/test_v2_protocol.py \
  tests/repairscope/test_v2_evaluation.py \
  tests/repairscope/test_v2_cf_evaluation.py
```

检查正常 Patch/Scope、Goal-Sufficiency、Boundary、State/Goal-Relevant Escape 的有效标签数、有限损失、Scope 参数实际梯度和更新、显存、计费及日志。训练日志中必须看到至少一个有效、informative、budget-competitive 决策组进入实际 batch，`actual_label_loss_groups > 0`、Goal-Sufficiency loss 非零且 Scope selector/logit 梯度非零；仅有全数据预审计非零不够。Smoke 的固定小数据应包含一条 H=4 可递推自然轨迹和一组合法、可训练的同前缀 action/reference 分支，并确保该组在 2–5 次更新内实际参与训练。H=4 使用模型自己的预测状态递推，不用真实中间状态。Smoke 仅验证工程，不能据效果筛选方法。发生具体工程错误，只修对应失败链路；不追加效果 Pilot 或多轮小对照。

本地旧合成工程 smoke `runs/v2-engineering-smoke` 只能作工程上下文：它使用 synthetic fixture，训练 5 次更新，但训练步记录的 `decision_eligible_groups=0`、`goal_sufficiency=0`；单独保存的预审计则报告 2 个 trainable groups。该差异尚未在真实环境验证。不要声称旧 smoke 已验证决策目标；服务器这一次 smoke 必须核清实际 batch 与审计收据的对应关系。

## 6. 共享数据与可比真值

自然轨迹与合法反事实分支只采一次，由所有方法共用。复用已有记录时保留原 Scope/model_version、完整审计、前缀证明和执行成本；不能把历史 Scope 标签当作 v2 新范围。训练数据只含官方 train，源任务及全部兄弟分支同组。

入口均存在于当前仓库：

```bash
bash scripts/repairscope/collect_train.sh
bash scripts/repairscope/collect_counterfactual.sh
"$REPAIRSCOPE_PYTHON" scripts/repairscope/build_training_data.py \
  --inputs "$NATURAL_TRANSITIONS" "$CF_TRANSITIONS" --output "$TRAIN_DATA"
bash scripts/repairscope/collect_dev.sh
```

先根据实际配置设置 `CONFIG`、`RUN_ROOT`、`PROBE_PLANS`、输出路径变量。不要把示例路径、fixture 计划或假 ID 用到真实采集。现有前缀、条件动作、只读证明、失败调用和续跑全部计费。

普通 dev 自然轨迹没有同前缀候选结果，不能凭空计算 Decision Regret。冻结的 dev 反事实评估接口只作非自适应、独立评估数据，记录 evaluation_only/frozen_query，绝不进入训练、回放或探索；禁止 test 反事实执行。计划必须显式写 `split=dev`、`source_split=dev`，完整计划在采集前保存 hash，不根据结果追加候选：

```bash
"$REPAIRSCOPE_PYTHON" -m repairscope counterfactual --config "$CONFIG" \
  --plans "$DEV_CF_PLANS" --set collection.split=dev \
  --set collection.counterfactual_evaluation_only=true \
  --output-dir "$DEV_CF_OUTPUT"
```

将合法 dev 自然轨迹与 dev CF 评估记录合并为独立的 `EVAL_DATA`，保持 dev/frozen_query 来源；不得使用仅接受 train 的 `build_training_data.py`。完整任务 TGC/SGC 与这些离线决策标签分开报告。缺少可比较真值的指标为 N/A，并报告分母和覆盖率，不能把 train 拟合结果改称 held-out 泛化。

冻结共享 train/dev 文件与 hash、公开目标提取器、Actor 候选缓存、任务/组合分组 Manifest、代码版本和总成本上限，再启动正式训练。目标未知样本仍保留用于可以可靠计算的其他任务，缺失标签不变成 negative。

## 7. 正式 A0–A4 与必要消融

一次 Smoke 通过且存在真实可训练业务决策监督后，直接开始正式实验。完整协议已在 `docs/RepairScope_v2_Proposal.md` 的 Evaluation Plan 中冻结；Proposal 文档已随代码一起提供，不需要等待额外文档工作。

| 标签 | variant | 定义 |
|---|---|---|
| A0 | actor_only | Frozen Actor |
| A1 | dynamics | 普通 Step + Rollout |
| A2 | paired_effect | Do-JEPA-style 适配效果学习 |
| A3 | full | 原 v1.1 Dynamic Scope + Boundary + State Escape |
| A4 | goal_sufficient | v2 Goal-Sufficient Scope |
| 消融 | no_goal_sufficiency | v2 去掉新增决策损失 |
| 消融 | v2_no_boundary | v2 去掉 Boundary |
| 消融 | no_goal_relevant_escape | v2 去掉 Goal-Relevant Escape |
| 消融 | v2_static_scope | 静态 Scope + 不确定性扩展 |

所有组同一 Actor/提示/候选动作/训练数据/模型容量/预算/任务清单；v2 新小头在所有配置均实例化以匹配容量。A0 不训练。初轮固定 **seed100，8 个训练配置各 30 epochs**，有效 batch=16（micro4、累积4）、BF16、AdamW 3e-4、weight decay .01、clip1，H4、候选上限256、Scope128、候选动作4、控制器总calls100/任务、步数50。只有显存故障允许两组一致地减 microbatch并增累积，保持有效batch及兄弟分支完整。

使用全新的结果目录名，必须包含 v2/goal_sufficient；入口拒绝覆盖已有目录。配置、训练输入和 dev 输入用实际绝对路径。

```bash
export SEEDS=100
export RUN_ROOT=/root/autodl-tmp/RepairScope-runs-v2-main
export TRAIN_DATA=/实际/冻结共享train.jsonl
export EVAL_DATA=/实际/冻结dev.jsonl
DRY_RUN=1 bash scripts/repairscope/run_goal_sufficient_main.sh
```

Dry-run 只解析命令，不运行环境。确认实际配置后，在 tmux 中启动同一入口，保存完整日志。可写一个只含上述已核实变量与以下命令的启动脚本，放在已有 notes 下并记录其 hash：

```bash
bash scripts/repairscope/run_goal_sufficient_main.sh
```

用 `tmux new-session -d -s rs_v2_main` 启动长任务，并让会话执行这个实际脚本。不要依赖未导出的交互变量；日志使用 `tee` 时开启 `set -o pipefail`。Actor 服务与 Predictor 训练分时使用单卡 RTX4090，复用原 Actor tmux 启动方式；无需服务时按该组任务的进程所有权安全暂停，禁止广泛 `pkill`。

主脚本先 A0–A4，后四项必要消融。各 checkpoint 在 dev 上仅一次阈值选择：v2 对 goal_escape 的可靠标签校准，v1 对 state escape，无 Escape 对 uncertainty。没有可靠 dev 标签时报告阻塞/未校准，不能借用其他 checkpoint 的阈值。正式评测用完整官方 test_normal 168；不改提示、阈值或数据，不用测试结果回流训练。

## 8. 指标、探索与停止条件

同时报告 TGC/SGC、任务成功、目标退化、修复恢复与新损害、Scope Recall@B、State Escape、Goal-Relevant Recall@B、Goal-Relevant Escape、Decision Regret、执行与检查成本。

Decision Regret 是部署控制器在相同合法可比较候选组上所选动作，相对充分离线目标效用最优动作的效用差，含预测/风险/不确定性等选择影响；不是 Scope 遗漏的纯因果归因。`scope_isolation_oracle_patch_diagnostic` 只考察范围遗漏，单独列出。不得把 Oracle-patch 诊断当作实际 Predictor 的有效性。

预算曲线显式配置 `evaluation.scope_budgets`；同一 checkpoint、同一公开候选上限、同一冻结 dev 数据，不重新训练、不挑最佳测试预算。它是离线字段预算—目标保护曲线，不等于真实 AppWorld 任务成功—总调用成本曲线。未见条件/动作/目标组合按提前冻结的 Manifest 分组；未实现自动组合标签时保留 N/A，不事后挑选有利组。

主 A 完成后执行独立 B：同一起始 A4 checkpoint、同一 train-only 候选计划池，随机/一般不确定性/novelty-learnability/risk 四策略各 **2000** 真实环境执行成本单位，ε=.1，总上限 **8000**，失败/前缀/条件/查询/续跑全部记账。不能因样本少或结果差增加预算。当前 novelty-learnability 是轻量启发式，不能称完整复现 OnlineWM。记录累计25/50/75/100%预算点，查询任务冻结且不进入更新。

```bash
export CHECKPOINT=/实际/A4/seed_100/training/checkpoint.pt
export PROBE_PLANS=/实际/冻结train-only计划.json
export RUN_ROOT=/root/autodl-tmp/RepairScope-runs-v2-exploration
bash scripts/repairscope/run_goal_sufficient_b.sh
```

共享采集与 dev 评估分支的总调用上限必须从已有批准预算/冻结计划按真实 replay 成本核算并写 Manifest，不能把 B 的 8000 预算借给采集或自动扩张。若原预算尚未明确，先完成固定计划的费用核算，报告尚缺的明确预算值，不开始无界采集。

普通工程错误可在当前范围等义修复并留源码 diff、必要回归、真实版本与成本。正式 checkpoint 仅在完全同数据/模型/loss/效用/日程时 exact resume；v1 checkpoint 不直接续训成 v2。版本变化影响对照时保持全组一致。没有真实业务目标、没有可训练 Scope 决策组、真值/成本无法核实、或必须改变机制/预算时停止受影响链路并报告；不重复全面审计，不用无意义 Pilot 填补数据缺口。

## 9. 交付与运行状态

保持正式计划推进，不因暂时负面指标改方法或重启实验。给用户交付：实际 Git SHA/分支、环境和模型版本、Smoke收据、正式启动命令、tmux session、日志与 checkpoint 绝对路径、完整配置/Manifest、数据hash、已耗环境调用与GPU时长/峰值显存，以及**当前实际执行状态**。未启动/被阻塞必须明确写出，不能把准备完毕写成正在训练。

运行日志继续写既有 notes 目录与专用 run目录；所有实验独立输出。不得提交密钥、权重、大型训练数据或无关文件。保留负面结果与失败成本，不以工程测试通过冒充科学验证。
