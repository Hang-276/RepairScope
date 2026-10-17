# RepairScope-Evo

**Self-Improving Repair Knowledge through Joint Counterfactual Execution**

冻结模型参数，从真实训练任务的执行经验中发现、验证和修订条件化修复知识：什么条件下需要补偿操作，何时补偿多余，以及它是否破坏已经完成的目标。

本仓库按 [Phased Proposal v1.0](docs/PROPOSAL_v1.0.md) 重建，当前代码范围为 **P0 资产与协议冻结、P1 有预算的可行性验证**。旧四态预测、4B 训练和旧 GATE 2 工作流已经废弃；旧版本仅保留在 Git 历史中。P2 持续学习及后续正式实验，须在真实 P1 证据满足条件后另行推进。

> 当前没有连接服务器、运行新的 AppWorld 实验或证明方法有效。合成夹具、格式检查、资产 receipt 与真实研究结果分别记录。详见 [实现状态](docs/IMPLEMENTATION_STATUS.md)。

## 获取与安装

GitHub 仓库保持公开，服务器可以直接克隆：

```sh
git clone https://github.com/Hang-276/RepairScope.git
cd RepairScope
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
```

`repairscope` 包使用 Python 3.11 及以上版本，运行时只依赖标准库。它不会自动下载任务或模型、安装 AppWorld、启动或重配模型服务。真实执行使用服务器已经部署的 AppWorld 环境和冻结 Qwen3-8B-AWQ 服务；轻量编排环境不能代替既有 AppWorld 执行环境。

`vendor/appworld` 继续固定到 `42b5bcf3cd334fee33f0c37c02070a9f5807add5`。需要查看该版本源码时可运行 `git submodule update --init vendor/appworld`；离线测试不需要拉取子模块。

## 本地检查

```sh
python -m pytest
repairscope fixture-smoke --output-dir runs/fixture
python -m build
```

`fixture-smoke` 使用明确标记为 `synthetic_fixture` 的小型状态环境，检查编排和证据路径。它不使用 AppWorld 数据、不调用真实模型，也不能成为 P1 的成功见证。

## 服务器阶段入口

先按 [服务器操作说明](docs/SERVER_RUNBOOK.md) 收集现有资产，准备资产说明文件；保留服务器未提交的 C2/C5 等修订。资产路径由服务器操作者提供，代码不猜测服务器目录，也不自动下载缺失资产。

```sh
repairscope p0 --assets /path/to/ASSETS.json --output-dir runs/evo_v1
repairscope inspect --p0-dir runs/evo_v1
repairscope p1 --p0-dir runs/evo_v1 --events /path/to/EVENTS.json \
  --output-dir runs/p1 --base-url http://127.0.0.1:8000/v1
```

也可以通过 `python scripts/run_evo_stage.py` 调用相同命令。缺少可核验的服务器资产、冻结目标或合格的重放证据时，P0 应报告 `blocked`；没有真实输入不能将其判为通过。

P0 固定资产身份、数据边界、目标、任务次序和预算。P1 锁定既有原任务成绩，再在隔离训练副本中核验 1–4 个实际已有候选；系统自动提议“操作变化 × 条件变化”的四格实验，依据执行证据修订知识，并检查旧固定程序和新知识在旧情形中重新生成的行为。留出行为评估单独记账，留出状态不能参与对应规则的归纳或回归门控。

服务器操作者需要按 [输入契约](docs/SERVER_RUNBOOK.md) 导出完整既有 train 轨迹、候选、目标和 receipts。当前包提供离线分支及知识验证编排，尚不驱动原生在线 Actor 或 capture 流程，不会补造缺失资产。`configs/assets.example.json` 提供占位路径；`configs/events.example.json` 是故意留空、不能通过真实运行校验的输入外壳。冻结预算示例为 `configs/evo_v1.yaml`，使用 JSON 语法。

## 研究边界

- 学习来源只允许 train；dev 仅用于后续冻结评估，test 在正式阶段前封存。固定模型参数，不训练 4B、LoRA、RL 或 `K_probe`。
- 首批计划 3–5 个 train 任务、10–15 个状态；每状态最多四个初始候选，每事件最多 12 个新分支，首批最多 180 个分支。只允许一次预设扩展，最多再增加 15 个状态和 180 个分支，不能更换种子反复采到通过。
- 每任务最多两次实际知识学习事件；额外自然状态继续保留全部候选见证尝试，不再启动联合学习。每学习事件最多六次模型调用、合计 8000 输出 token；留出行为评估最多三个案例、每案例三次新旧配对，共最多 18 次行为运行，单列成本。
- 条件干预必须通过合法 API；只有经审计、在两种状态中语义不变的 checker 才能支持机制结论。无法判断记 `unknown`，不能修改 evaluator 使样本通过。
- 逐任务人工任务、规则和干预编辑必须为零。原任务成绩不可被影子分支覆盖，同一任务中知识固定，知识更新只影响后续任务。
- 公共仓库只发布通用代码、配置和可公开文档。任务内容、轨迹、目标锁文件、数据库、模型权重、派生知识及完整运行证据不应提交。

当前记录完整分支身份与 `cache_key`，尚未实现重放缓存；所有实际分支均消耗预算，不存在免费缓存命中。

所有运行及实验过程日志使用清晰英文，控制台与运行目录的 `run.log` 同步即时输出；原始 JSON/JSONL 证据独立保存。实验只在服务器执行，本地用于代码实现和离线软件检查。日志格式及内容边界见 [English run logs](docs/LOGGING.md)。

完整预算、P1 A/B/C 验收和停止条件见 [协议](docs/PROTOCOL.md)。操作入口见 [服务器说明](docs/SERVER_RUNBOOK.md)，旧项目处理见 [迁移记录](docs/MIGRATION.md)。

## 目录

```text
src/repairscope/             P0/P1 实现与显式合成夹具
configs/                    冻结协议配置
scripts/run_evo_stage.py    阶段入口
tests/                      离线单元与集成检查
docs/PROPOSAL_v1.0.md       原始研究提案
docs/PROTOCOL.md            当前范围与验收条件
docs/SERVER_RUNBOOK.md      服务器资产准备和运行步骤
docs/IMPLEMENTATION_STATUS.md 实现与真实验证的边界
docs/MIGRATION.md           旧方案替换与服务保护记录
vendor/appworld/            固定的上游子模块
```

## 许可

本项目采用 Apache-2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。AppWorld 上游源码及受保护资源继续遵守其原有许可和发布限制。公开仓库不包含受保护任务数据或模型权重。
