# 服务器执行手册

本仓库在本地验证的是依赖隔离、数据边界、假环境分叉、候选选择和 CPU
fixture 的软件流程。**真实 AppWorld 运行、4090 峰值显存、真实标签质量和方法
收益均待服务器验证。** `fixture-smoke` 成功不能写成主实验已成功。

以下命令均从本仓库根目录执行。完整研究约束见
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)，标签审计见
[LABEL_AUDIT.md](LABEL_AUDIT.md)。

## 1. 建立独立工作区

将仓库 clone 到一个新的、明确属于本次实验的路径。不要覆盖或迁移原 B2
目录，不要复用原 B2 的虚拟环境，不要修改系统 Python、代理、路由、SSH 配置、
其他服务或他人的运行进程。模型缓存可以在权限允许且不改变原配置时只读复用。

```bash
bash scripts/bootstrap_server.sh
.venv/bin/python scripts/prepare_models.py --download
```

该脚本准备独立 `.venv`、独立上游代码目录和 AppWorld 数据。Actor 服务使用
另一套 `.venv-actor`，避免 vLLM 的依赖要求改动训练环境。若已有目录或资源与
脚本预期冲突，应保存当前信息并换一个新的本次实验路径，不要清理旧目录解决。

核验以下内容并保留记录：

- 上游 AppWorld commit：`42b5bcf3cd334fee33f0c37c02070a9f5807add5`。
- 数据版本：`0.2.0`。划分大小为 train90、dev57、test_normal168、
  test_challenge417，四个划分的 task/scenario 不交叉。
- `configs/appworld.toml` 是模板；模型准备脚本生成 `runs/configs/appworld.json`。检查后者的 `root`、`output_dir` 是本次服务器的真实绝对路径。
- Actor 和 predictor 的模型标识解析到实际下载的 snapshot/revision。保存实际
  revision、配置、包版本、GPU 型号和可用显存，不以可变化的 `main` 代替记录。
- 不在仓库、日志和提示词中写入密钥。需要密钥时通过指定环境变量传入。

```bash
.venv/bin/python -m b2_appworld doctor --config runs/configs/appworld.json --output runs/doctor.json
.venv/bin/python -m b2_appworld manifest --config runs/configs/appworld.json --output runs/dataset_manifest.json
.venv/bin/python -m pytest -q
.venv/bin/python -m b2_appworld fixture-smoke --output-dir runs/local_contract_fixture
```

`doctor.live_ready` 只报告依赖/数据/GPU 的静态必要条件；它不测试HTTP服务、真实任务、NF4训练或显存共存，不能视为服务器验收通过。
不要为了修复测试错误，改为未固定的上游最新版。

## 2. 启动冻结 Actor，测量单卡共存

在一个独立终端启动：

```bash
bash scripts/serve_actor.sh
```

主配置使用冻结 `Qwen/Qwen3-8B-AWQ`、关闭 thinking、temperature0、
8192 context、最多3000输出 token、并发1。vLLM 初始
`gpu_memory_utilization=0.35`，约对应24GB卡上的8GB预算量级。
4B NF4 predictor 另预留约4GB量级作为初始估算。**这两个估算不是显存承诺**：
量化权重、KV cache、上下文、CUDA graph、激活与碎片都会改变峰值。

先用真实的最长观察历史和完整四候选进行短试跑，同时记录整卡占用与进程占用。
训练日志的 PyTorch peak 只反映当前训练进程，不能代表 vLLM 与 predictor
并存时的整卡峰值。保留 OOM、峰值和调整记录。

若共存不足，先检查服务是否由本次实验启动、是否意外并发，再统一降低本次
Actor 的显存预算或统一缩短上下文；任何调整都应用到全部比较方法并重跑相关
基线。不要只让 B2 使用更短历史，或为了腾显存停止未知进程。当前代码不提供
Actor/predictor 自动卸载与换入，不能把这种策略当作已实现路径。

## 3. 接口 pilot：先3个任务

```bash
bash scripts/run_stage.sh pilot
```

此阶段的少量 train/dev 任务用于确认服务通信、上游 agent、任务执行、代码
保存、checkpoint 和环境关闭。3个 task 可能仅来自1个 scenario，因此它不是
研究有效性实验，也不足以验证泛化。

对应 CLI 示例：

```bash
.venv/bin/python -m b2_appworld run --config runs/configs/appworld.json --split train --method baseline --experiment pilot_baseline_train --limit 3
.venv/bin/python -m b2_appworld run --config runs/configs/appworld.json --split train --method capture --experiment pilot_capture_train --limit 3
```

每个 experiment 名称必须全新；代码会拒绝覆盖已有结果。检查
`runs/EXPERIMENT/run.json`、`receipt.json` 和
`tasks/TASK/trajectory.json` 的实际存在、结束状态与执行块。检查准备宣告完成时
不会被反复无效果修复阻塞；若存在该问题，先修复统一策略再继续。

`capture` 会产生 goal/candidate 的辅助推理调用，但执行原 Actor 提议。其用途
是采集候选后果，不作为原生 ReAct 性能基线，也不与其他方法混算成本。

## 4. 采集真实 train/dev 轨迹与分叉标签

接口通过后先做扩展的 train/dev 小样本，按下面的同一流程审计标签。确认标签
可用再执行完整 capture；不要先投入全部训练时间。

```bash
bash scripts/run_stage.sh capture
bash scripts/run_stage.sh labels
```

主阶段使用 train 全90个任务、dev 全57个任务。也可逐阶段执行明确的 CLI：

```bash
.venv/bin/python -m b2_appworld run --config runs/configs/appworld.json --split train --method capture --experiment capture_train
.venv/bin/python -m b2_appworld run --config runs/configs/appworld.json --split dev --method capture --experiment capture_dev
.venv/bin/python -m b2_appworld map-goals --config runs/configs/appworld.json --trajectories runs/capture_train --output runs/labels/train.mapping.json
.venv/bin/python -m b2_appworld map-goals --config runs/configs/appworld.json --trajectories runs/capture_dev --output runs/labels/dev.mapping.json
```

`map-goals` 输出是仅供 labeler 使用的私有映射。按标签审计文档审查后，再采标签：

```bash
.venv/bin/python -m b2_appworld collect --config runs/configs/appworld.json --trajectories runs/capture_train --mapping runs/labels/train.mapping.json --output runs/labels/train.jsonl
.venv/bin/python -m b2_appworld collect --config runs/configs/appworld.json --trajectories runs/capture_dev --mapping runs/labels/dev.mapping.json --output runs/labels/dev.jsonl
.venv/bin/python -m b2_appworld quality --data runs/labels/train.jsonl --output runs/labels/train_quality.json
.venv/bin/python -m b2_appworld quality --data runs/labels/dev.jsonl --output runs/labels/dev_quality.json
```

`labels` 便利阶段会先生成自动映射，不能代替人的语义审计；若审计修订了映射，
必须重新执行相应 `collect` 并保留旧版本，不能仅改报告。

采集器为每个 reference/candidate 新建进程和环境，完整重放前缀，比较相同的
数据库变更与观察，不在在线任务中免费分叉或回滚。无可干预 checkpoint 的任务
不会采到训练标签，但继续属于所有完整任务评测的分母。

检查 `train.jsonl.receipt.json` 与 `train.jsonl.states.jsonl`。若
`status=no_accepted_examples`、已接受样本为0、类别仅剩00/11，或没有真实候选
差异，应停止训练并诊断；不要用 fixture、伪造01/10或测试集补足。

## 5. 冻结标签和协议，释放自己启动的 Actor

按实验协议完成至少30个 goal 的人工审计及数据门控。保存输入 JSONL 和映射
哈希、scenario 分组、训练配置、目标/候选提示词、效用权重与成本统计方案。

在启动 Actor 的终端使用 Ctrl-C 结束**本次启动的服务**。若采用后台启动，先
核对自己保存的 PID、进程命令和工作目录，再停止该 PID。禁止使用不加核验的
`pkill`、停止占端口的未知进程或终止整卡上所有 Python 进程。

训练阶段不需要 Actor 服务。完成一类/一种子后退出进程，再训练另一类/种子，
避免两个4B训练器并发占卡。

## 6. 训练后果模型与同容量标量对照

```bash
bash scripts/run_stage.sh folds
bash scripts/run_stage.sh train
```

对应单次训练 CLI：

```bash
.venv/bin/python -m b2_appworld train --config runs/configs/train_effect.json --kind effect --seed 100 --output-dir runs/checkpoints/effect_s100
.venv/bin/python -m b2_appworld train --config runs/configs/train_scalar.json --kind scalar --seed 100 --output-dir runs/checkpoints/scalar_s100
```

主配方是 Qwen3-4B、NF4、LoRA rank16/alpha32/dropout0.05，length4096、
batch1、累积16、最多3 epochs。以配置文件为真实执行来源，保存 resolved
model revision。主比较让两类模型使用同一 fully-labeled cohort；effect 的
部分标签训练只能作为单独消融，不能暗中获得更多监督。

先完成一个短真实训练并核查有限 loss、optimizer step、best checkpoint
重载、dev/scenario holdout 预测和峰值显存，再执行三个种子100/101/102。
不要将 tiny backend 的成功当成4B NF4已经跑通。

```bash
.venv/bin/python -m b2_appworld evaluate-predictor --checkpoint runs/checkpoints/effect_s100/best --data runs/labels/dev.jsonl --output runs/checkpoints/effect_s100/dev_diagnostics.json
.venv/bin/python -m b2_appworld evaluate-predictor --checkpoint runs/checkpoints/scalar_s100/best --data runs/labels/dev.jsonl --output runs/checkpoints/scalar_s100/dev_diagnostics.json
```

官方 dev 可用于 checkpoint/global protocol 选择。train-side scenario folds
是另一个机制检验：同一 scenario 的全部任务、状态和候选必须一起划入同一侧。
不要把训练侧 fold 的标签混入官方 dev 或 test。

## 7. Dev 闭环，再冻结完整测试

重新启动自己的 Actor 服务；先测 learned predictor 与 Actor 并存的整卡峰值。

```bash
bash scripts/run_stage.sh dev
```

四个主方法是 `baseline`、`direct`、`scalar`、`effect`。`rule` 是可选的
零样本逐目标后果清单，不是已实现的符号依赖闭包方法。每个方法共用同一个
Actor、环境、步骤/输出预算和检查点规则；辅助调用全部记账。对于 scalar/effect，
检查训练数据、骨干和输入保持一致，而不只是参数数量接近。

配置冻结后，才执行 `test` 阶段；该阶段耗时取决于实际服务速度，按服务器
测量安排。`test` 不使用 `--limit`，也不做 fine-grained capture。

```bash
bash scripts/run_stage.sh test
```

明确 CLI 示例：

```bash
.venv/bin/python -m b2_appworld run --config runs/configs/appworld.json --split test_normal --method baseline --experiment final_baseline_normal --final
.venv/bin/python -m b2_appworld evaluate-native --config runs/configs/appworld.json --experiment final_baseline_normal
.venv/bin/python -m b2_appworld run --config runs/configs/appworld.json --split test_normal --method effect --checkpoint runs/checkpoints/effect_s100/best --experiment final_effect100_normal --final
.venv/bin/python -m b2_appworld evaluate-native --config runs/configs/appworld.json --experiment final_effect100_normal
```

同样完整运行 test_challenge，并运行 direct、scalar 及其冻结 checkpoint。
原生结果读取 `native_aggregate.json`。TGC 是完整任务成功率；SGC 是同一
scenario 的3个任务全部成功的比例。不要把逐目标正确率当作 SGC，不要丢弃
ineligible、预算耗尽或方法失败的任务。

## 8. 交付与失败处理

保存本次 repository commit、upstream commit、配置/数据/模型哈希、包版本、
数据门控、映射审计、训练 summary、峰值显存、run receipts 和 native aggregate。
分别报告训练、离线采标签及在线执行成本。日志中不得包含密钥。

若出现错误，保留最小错误记录并先在 train/dev 重现。不要针对 test 单例
改提示词、阈值、候选或模型；若发现与方法无关的实现错误，记录修复理由，
重新冻结配置并对全部受影响方法完整重跑，不选择性补跑失败任务。

服务器最终报告应明确列出：已真实完成的阶段、实际命令与提交、产物路径、
实验结果、资源测量和未完成事项。没有真实 native aggregate 时，不能宣称
主实验已完成或方法有效。
