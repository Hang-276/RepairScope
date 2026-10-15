# RepairScope

**Predicting Goal-Level Repair Effects for Scoped Agent Recovery**

给定当前可观察任务状态和候选修复，预测它会恢复哪些目标、破坏哪些目标，并选择合理的恢复范围。当前实验载体是 **AppWorld API/code agents**；这是工具代理研究，不宣称已经验证 GUI 或具身迁移。

本仓库包含可运行的采集、训练、选择和评测入口。**尚无真实 AppWorld 主实验结果，也未在 4090 实测峰值显存。** 本地合成检查只证明实现链路，不能证明研究假设、创新性或 CCF-A 录用前景。

## 基线与方法

直接继承官方 `SimplifiedReActCodeAgent`，固定 AppWorld 源码到 [`42b5bcf`](https://github.com/StonyBrookNLP/appworld/tree/42b5bcf3cd334fee33f0c37c02070a9f5807add5)，通过 Git submodule 引入。保留官方提示词、Python 代码提取和 ReAct 对话更新；所有方法统一使用有预算记录的本地 HTTP 传输、8192 上下文限制和 50 个执行代码块上限。这是明确记录的适配版基线，不能直接与上游论文成绩作公平性等同。

1. 冻结 `Qwen3-8B-AWQ` actor，从原始任务提取有逐字证据的目标。
2. 仅当至少两个目标且均为最终持久状态目标时启用干预；其他任务继续原始 actor，仍计入原生评测分母。
3. 在可观察执行错误、每 5 个已执行块、宣告完成前检查。保留原始动作，另生成 local / composite / broad 三个当前状态修复候选。
4. 离线 train/dev 通过独立进程完整重放前缀，获得每个目标的 `00 / 01 / 10 / 11` 标签；无法可靠映射的目标为 unknown。
5. `Qwen3-4B` NF4 + LoRA 编码完整目标与候选，用共享四分类头预测每个目标的前后状态。一个候选一次 forward 同时输出全部目标。
6. 固定选择效用：`mean P(01) - 2 mean P(10) - .05 normalized_cost`。只执行最终候选，把实际代码写入后续上下文。

不执行未选中的候选、不在线访问 evaluator、不免费回滚。不引入 RL、图学习或额外排序损失。低成本合成后端会强制标记 `fixture_backend=true`，并被真实实验入口拒绝。

主比较为 `baseline / direct / scalar / effect`；`rule` 是可选的零样本四状态比较。Scalar 与 effect 使用同编码器、同输入、同完整标签样本和同候选成本归一化。未知标签的 partial effect 训练只作另行声明的诊断。

## 本地代码检查

Python 3.11 或 3.12：

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test,ml]'
.venv/bin/python -m pytest -q
.venv/bin/repairscope fixture-smoke --output-dir runs/my_fixture
```

这些命令不下载 AppWorld 数据或大模型。`fixture-smoke` 真实执行 CPU 反向传播、保存、加载与预测，但数据和小模型均为合成夹具。

## 服务器执行

```bash
git clone --recurse-submodules https://github.com/Hang-276/RepairScope.git
cd RepairScope
PYTHON_BIN=python3.11 bash scripts/bootstrap_server.sh
.venv/bin/python scripts/prepare_models.py --download
bash scripts/serve_actor.sh
```

Actor 服务前台运行；在另一个终端按阶段执行：

```bash
bash scripts/run_stage.sh pilot
bash scripts/run_stage.sh capture
bash scripts/run_stage.sh labels
# 按 docs/LABEL_AUDIT.md 审计真实映射并记录；不能虚构审计通过。
.venv/bin/python scripts/check_collection.py
# 训练时释放本项目 actor 的显存；只停止自己启动的服务。
bash scripts/run_stage.sh folds
bash scripts/run_stage.sh train
# 重新启动 actor，测量与4B分类器共存峰值，再跑dev。
bash scripts/run_stage.sh dev
.venv/bin/python scripts/freeze_protocol.py create
bash scripts/run_stage.sh test
```

每个阶段保留独立结果，已有实验目录会拒绝覆盖。`test` 需要冻结代码、配置、标签和全部最终检查点；只输出完整 split 的原生 aggregate TGC/SGC。若初步数据缺少真实损害、恢复或有差异的候选，应报告问题尚未得到支持，不能靠合成故障填充主结果。

4090 的建议是采集/评测用 8B-AWQ actor，训练时释放 actor 后训练 4B NF4 predictor；共存推理从 actor `gpu-memory-utilization=.35` 起试。此数值是待服务器实测的初始设置。vLLM 使用独立环境，避免覆盖训练环境的 Torch。

## 文档

- [固定实现与计划差异](docs/IMPLEMENTATION_NOTES.md)：最终可执行协议优先于早期提案。
- [方法与实验设计](docs/METHOD_DESIGN.md)
- [研究方向与4090可行性调研](docs/RESEARCH_REVIEW.md)
- [实验协议](docs/EXPERIMENT_PROTOCOL.md)
- [标签语义审计](docs/LABEL_AUDIT.md)
- [服务器运行手册](docs/SERVER_RUNBOOK.md)
- [服务器 Agent 提示词](docs/SERVER_AGENT_PROMPT.md)
- [本地验证记录](docs/LOCAL_VALIDATION.md)

源码为 Apache-2.0。AppWorld 受保护数据、解包源码、任务轨迹、模型权重、凭据和派生标签不提交到本仓库。通过官方工具重建数据与环境。

## 相关依据与限制

AppWorld 原生 SGC 是 **Scenario Goal Completion**：同一场景所有实例均完成；它不是子目标完成率。版本 0.2.0 源码对应 train/dev/test_normal/test_challenge 为 90/57/168/417，实际下载后必须通过 manifest 验证。

[AppWorld 官方仓库](https://github.com/StonyBrookNLP/appworld)提供任务环境与评测。[Qwen3-8B-AWQ 模型卡](https://huggingface.co/Qwen/Qwen3-8B-AWQ)说明 AWQ 和 non-thinking 切换；本实现显式发送 `enable_thinking=false`。[vLLM 0.11.0 文档](https://docs.vllm.ai/en/v0.11.0/models/supported_models.html)是所固定推理版本的兼容性依据，实际服务器安装仍需验收。

[Three Roles, One Model](https://arxiv.org/abs/2604.11465)研究小模型在单张 24GB GPU 上的推理角色编排，是硬件与代理脚手架的相关工作；不能将其结果转作 RepairScope 的实测证据。本研究的核心待检验点仍是：目标后果监督是否比同容量 scalar 和直接选择带来稳定收益。
