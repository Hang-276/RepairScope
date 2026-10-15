# B2：AppWorld 代码基线、方法、workflow 与实验设置

日期：2026-10-15。本文是待实现的具体方案。本轮已获取并静态核对官方源码，没有安装任务环境、下载数据或模型、运行训练或评测。文件中的默认超参数属于建议，明确标为“继承”的项目才是核验过的上游设置。

## 1. 确定采用的仓库基线

**单一上游仓库：[StonyBrookNLP/appworld](https://github.com/StonyBrookNLP/appworld)。具体继承官方 `SimplifiedReActCodeAgent`，不是只使用 AppWorld 数据集。**

固定源码提交：

```text
42b5bcf3cd334fee33f0c37c02070a9f5807add5
```

采用这个版本的 `appworld` 环境与同仓库 `experiments/` 中的 `appworld-agents`。不将环境安装成一个随机 PyPI 版本、agent 却使用另一版源码；实际安装后保存依赖锁文件、数据版本与模型 revision。

官方已经将旧 `legacy` 实现标为 deprecated，推荐 `simplified`：它没有额外 agent 框架，包含模型接口、Python 交互执行、日志、预算与评测，适合直接修改。[上游说明](https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/README.md#L1279-L1291)

继承的具体实现：

| 部分 | 固定提交下的文件 | 我们的处理 |
| --- | --- | --- |
| ReAct actor | [`react_code_agent.py`](https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/experiments/code/simplified/react_code_agent.py) | 保留原类，新增 B2 子类，原类用于同配置重跑 |
| 执行循环 | [`agent.py`](https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/experiments/code/simplified/agent.py) | 沿用生成代码、执行、接收观察、停止的主循环 |
| 模型接口 | [`language_model.py`](https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/experiments/code/simplified/language_model.py) | 保留本地 OpenAI-compatible 服务接入 |
| 基础提示 | [`instructions.txt`](https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/experiments/prompts/react_code_agent/instructions.txt) | 保留通用 onboarding；B2 提示单独存放 |
| 实验配置 | [`simplified_react_code_agent.jsonnet.j2`](https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/experiments/configs/_generator/templates/simplified_react_code_agent.jsonnet.j2) | 从此生成/派生配置，保存最终展开值 |
| 环境 | [`environment.py`](https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/src/appworld/environment.py) | 不改任务语义、API 和原生执行/判分规则 |
| 官方评测 | [`evaluator.py`](https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/src/appworld/evaluator.py) | 保留 TGC/SGC；B2 指标另设分析入口 |

AppWorld 没有现成的 B2 后果预测训练流程和对应权重，需要新增采数、预测器训练与选择模块。它提供的是我们真正能够扩展的 agent 与环境基础。不要把其环境 checkpoint 称为模型训练 checkpoint。

## 2. 建议的方法：逐目标后果预测驱动的恢复选择

第一版只包含一个学习模块：**目标条件的修复后果预测器**；配一个固定的候选选择规则。actor、目标抽取器和候选生成器均冻结，不训练完整 agent，不增加依赖图学习、多智能体、RL 或辅助 ranking loss。

### 2.1 输入和输出

输入：

```text
x = 当前允许观察到的交互历史 + 已检索 API 文档
G = 从原始用户指令抽出的目标与明确保留约束
r = 一个完整、可执行、受预算约束的候选修复代码块
```

输出：对每个目标 `g_i`，预测四态联合概率：

| 类别 | 执行前 → 修复代码块结束后 | 意义 |
| --- | --- | --- |
| 00 | 未满足 → 未满足 | 未恢复 |
| 01 | 未满足 → 满足 | 恢复 |
| 10 | 满足 → 未满足 | 破坏 |
| 11 | 满足 → 满足 | 保留 |

写成 `f_theta(x, G, r) -> M × 4`。当前目标的真实满足状态只用于离线标签，不提前提供给在线模型。通过 `P(10)+P(11)` 可得到当前满足概率，需单独检查当前状态估计误差，避免把误判当前状态当成正确预测修复。

修复候选是一段完整代码，可能含多个 API 调用与根据返回值的条件分支。其结果在**整个代码块结束时**判断；中途暂时撤销、结束前恢复的目标，不计为最终破坏。异常代码可能已产生部分写入，结束标签按实际状态计算，不能假设自动回滚。

若后续发现修复必须跨多个代码块，再扩展有明确终止条件的宏。第一版不根据若干单动作预测直接拼出任意多步恢复闭包。

### 2.2 目标如何取得

同一个冻结的 actor 模型从原始指令提取目标，一次输出：

```json
{
  "goals": [
    {"id": "g1", "type": "state_write", "text": "原指令要求的终态条件", "evidence_span": "对应原指令片段"}
  ]
}
```

这里只给格式，不代表已创建的真实任务。保留的目标必须有指令证据；“找到对象”“已经登录”这类中间步骤不自动变成永久维护目标。

训练/开发任务上，将高层目标映射到独立 checker 或一组官方断言。多个断言可能属于同一个目标；`no_op_pass/no_op_fail` 也不是独立目标类别。不能把隐藏测试的 requirement、正确答案或失败报告当作目标描述送给模型。

第一轮四态监督只覆盖可在中间数据库状态判定的终态写入目标。需要最终回答、整段轨迹才能判定的条件不强行贴中间状态标签。官方主测试仍包含所有任务，不因无法细粒度标注而删任务。

为使训练与在线范围一致，抽取器只依据指令将目标分为 `state_write / final_answer / trajectory / unknown`。**第一版仅在全部目标均为可操作的 state_write、且至少有两个独立目标时启用 B2；混合目标、回答型、未知类型或抽取失败的任务回退原生 ReAct。** B/C/D 共用这个条件，原生主表仍计算全部任务；单列启用覆盖率。训练侧可以利用单目标写入样本增加监督，但不能据测试 checker 是否可判定来决定线上启用资格。第一版不声称覆盖所有类型的目标。

### 2.3 候选池与恢复范围

每次决策使用 `K=4` 个候选：

1. actor 原本准备执行的下一段代码，代表继续当前方案。
2. 只针对最可能失败目标的局部修复。
3. 针对相关目标子集的组合修复。
4. 从当前真实状态出发、覆盖更广目标的重新规划。

后三个候选由同一个冻结 actor 通过一次通用提示生成，输出 `target_goal_ids + code`，加上原提议组成候选池。不得为具体 AppWorld API 或故障手写修复程序，也不得依据真实 checker 挑修复目标。上游禁止把具体 API 调用硬编码为 agent 技巧；通用提示让 actor 自行探索 API 则可沿用。[开发规则](https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/README.md#L937-L953)

具体长度预算：目标抽取最多 512 个输出 token；三个新候选共用一次最多 2,400 token 的生成，各候选代码最多 600 token，其余留给 JSON 与元信息。超长/截断候选记为无效，不执行被截断的代码。原始 actor 提议仍沿用上游 3,000 token 上限。所有选择器使用完全相同的候选池，另报告相近长度候选的匹配分析。

“广范围重新规划”不是恢复初始数据库，也没有免费历史回滚。需要撤销错误时，只能通过正常 API 完成补偿或修改。

候选的 `target_goal_ids` 是提议意图，预测/实际后果才是它的影响范围。不能用代码长度、API 次数或模型写出的 scope 标签，直接充当范围正确性。

### 2.4 候选选择

第一版采用：

```text
benefit(r) = mean_i P_i(01)
harm(r)    = mean_i P_i(10)
score(r)   = benefit(r) - lambda * harm(r) - beta * estimated_cost(r)
```

初始默认 `lambda=2`、`beta=0.05`。候选提议时附上根据可见代码和已知循环边界估计的 API 调用数 `estimated_api_calls`；用候选池内最大估计值将其归一化到 0-1，作为统一的 `estimated_cost`。它是可见信息下的代价代理，不是真实执行成本；所有方法共享这一估计，另报告估计误差和实际 API/时间代价。无需再学习成本模型。

相同得分先选估计代价较低者，再按固定候选顺序打破平局。额外做 `beta=0` 消融，区分后果预测和代价偏好。第一版只声称在候选池中做恢复/保留/代价权衡；不能仅凭这个代理代价宣称恢复实体范围全局最小。范围结论仍需实际目标与写入实体的评测支持。

若原指令有明确的保留优先级，可转为固定权重；第一版默认等权，不从隐藏评测中推导重要性。上述概率加权是经验选择规则，不构成“绝无目标破坏”的保证。

只执行获选代码块，然后返回原生 ReAct 循环。候选全部格式无效时回退原提议，并统计回退率。返回 actor 的历史必须记录**实际执行的代码和观察**，不能仍记被替换、实际未执行的草稿。

### 2.5 预测器训练

默认选择文本模型 `Qwen/Qwen3-4B`，4-bit NF4 QLoRA，加共享四分类头：将完整输入放在前缀，再附每个 goal 的查询位置，在对应 hidden state 上预测四个类别。目标列表已在前缀提供；真实转移标签不作为输入。一个候选一次前向得到所有 goal 输出，避免每个 goal 都重新编码整段历史。

只用四分类交叉熵，按候选内 goal 平均、再按候选平均。未知或无法判定的 checker 标签 mask，不强行设为 00。第一版不写长解释、不增加奖励模型或额外因果损失。

同容量标量基线使用同一个 4B 骨干、量化、LoRA 和输入，直接预测由真实转移算出的效用。只有当逐目标结构在同等条件下改善组合泛化或决策，才支持方法贡献。

## 3. 在线 workflow

固定恢复检查时点，先不学习另一个“何时干预”模型：

- 刚发生代码/API 异常后；
- 每完成 5 个环境交互后；
- actor 准备调用完成任务 API 前。

重复事件合并为一个检查点，只对已启用 B2 的任务进行。间隔 5 是我们的初始工程设置，不是上游默认，也不作为论文贡献。除原生 ReAct 外，所有候选选择对照共用相同检查时点。

```text
读取任务指令
    ↓
抽取并固定目标 G
    ↓
官方 ReAct actor 根据可见历史提出下一段代码
    ↓
是否到固定检查点？
  否 → 按原生流程执行
  是 → 根据同一历史生成 4 个候选
          ↓
        预测每个候选对所有目标的 00/01/10/11 概率
          ↓
        固定分数选择一个候选
          ↓
        只执行获选代码；记录实际结果
    ↓
继续 ReAct，直到任务宣告完成或原生预算耗尽
    ↓
独立调用官方 evaluator，得到完整任务 TGC/SGC
```

在线不调用 evaluator 检查“到底失败了哪些目标”，不试执行未选候选，不调用 `load_state` 撤销结果。模拟器分支用于离线标签和单列的 oracle 对照。

每次执行的修复代码块占用一次正常环境交互；目标抽取、候选生成、预测全部计入总推理和时间。50 次交互外不赠送额外恢复执行。

## 4. 离线采集与开发 workflow

### 阶段 A：冻结上游并重跑原版

1. 从固定提交建立后续开发分支，保留官方原类。
2. 安装与此提交匹配的环境、agent、Git LFS bundle 和数据；数据使用 `minimal` test ground-truth 模式，不申请维护者私有测试答案。
3. 保存展开配置和运行环境版本；运行官方环境校验及少量 train/dev 基线任务，获取真实耗时、显存和错误。
4. 所有主表数字重跑得到，不直接搬论文中的旧模型分数。

这里不预先创建远程仓库、不覆盖旧 B2 仓库。正式实现可以把本方案新增文件放入从 AppWorld 派生的 B2 仓库，并保留 upstream remote。

### 阶段 B：从 train 采集候选后果

1. 在官方 train 上运行冻结 actor，记录完整代码和观察前缀。
2. 在固定检查点采样；每条轨迹上限先设 10 个状态，既保留出错状态，也保留未明显出错状态。不要仅挑 oracle 已知可修的状态。
3. 固定原始前缀、目标抽取结果和候选池。
4. 每个候选启动独立子进程，从初始任务重放**完全相同的执行代码前缀**，包括失败但可能部分写入的那段代码。
5. 核对重放数据库状态、可观察输出和执行上下文一致性，再执行候选，建立逐目标前后状态标签。
6. 将故障状态来源、候选覆盖、无法重放、无法贴标签和格式无效样本全部记账。

按当前源码预期 90 个 train tasks，10 个状态、4 个候选计算，自然轨迹首批上限约 3,600 个状态—候选对。实际有效量取决于轨迹长度、可判目标覆盖和重放一致性；不会承诺原生 train 自动产生几万条独立样本。

需要补充 10 类或复杂组合时，可从 train 做明确标识的故障增强或继续修复产生新状态；自然/合成来源分别记录。主测试不插故障。数据有效规模同时报告 scenario、task、state、candidate 数，不把 goal 行数或重复前缀当成新的独立任务。

### 阶段 C：训练预测器和同容量对照

- 先跑一版四态模型与标量模型，不加其他损失。
- 超参数与错误分析使用官方 dev；不用 dev 的反事实标签反向训练参数。
- 细粒度研究评估采用 train 内按 scenario 分组的 5-fold：同 scenario 的三个任务及全部状态、候选在同一 fold，固定超参数后才汇总留出表现。
- 5-fold 首轮用训练 seed 100；配置冻结后，最终用于原生完整任务的四态/标量模型在全部 train 上各训 seed 100/101/102。

这是区分“未见修复状态上的机制评估”和“原生完整任务效果”的具体办法。5-fold 模型与最终全 train 模型分开存储，不能拿后者在训练过的研究状态上当留出结果。

### 阶段 D：原生任务评测与机制实验

完成配置冻结后，对完整 `test_normal`、`test_challenge` 运行所有主对照，只读取聚合评分。训练/开发来源的 recovery diagnostic 用于逐目标效果、scope 和失败案例分析。

官方测试不用于调整候选提示、手工补 checker 或筛出“适合我们”的任务。若需要新目标组合的可公开细粒度研究测试，应从训练侧生成独立任务并明示为 B2 扩展集，不冒充官方测试。

## 5. 哪些实验设置直接继承

参考当前 simplified 配置模板与模型注册配方，而不是混用 ACL 2024 历史论文、基类默认值和 CI 配置。[配置模板](https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/experiments/configs/_generator/templates/simplified_react_code_agent.jsonnet.j2)；[Qwen 本地模型配方](https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/experiments/configs/_generator/models/alibaba.py#L196-L236)

| 设置 | 上游实际设置 | 本方案 |
| --- | --- | --- |
| Agent | Simplified ReAct Code | 继承 |
| 接口 | Python REPL + `apis.app.api(...)` | 继承 |
| actor 基础系列 | 已有 Qwen3-8B without reasoning 配方 | 继承系列与模式，改官方 AWQ 权重适配单卡 |
| temperature | 0 | 继承，actor/目标抽取/候选提议均先用 0 |
| enable_thinking | false（此模型配方） | 继承 |
| 模型/环境 seed | 100 | actor 与环境固定 100；另列预测器训练 seeds |
| 每次 actor 生成 | 最大 3,000 completion tokens | 继承 |
| max_steps | 模板 50；基类默认 40 | 使用模板 50 个实际环境交互 |
| ignore_multiple_calls | true | 继承：执行提取到的第一个代码块 |
| 每任务输出 token 总上限 | 100,000 | 继承上限，扩展计入所有新增生成调用 |
| 单代码块 timeout | Linux 默认 100 秒 | 继承 |
| 环境 max_interactions | 1,000 | 保留底层默认；agent 50 步先约束 |
| 单交互最大 API calls | 1,000 | 保留，不误写为每任务 1,000 API 总上限 |
| 模板字符截断 | max_prompt_length/max_output_length 为 null | 因单卡上下文新增统一 token 裁剪，属于修改项 |
| 官方主指标 | TGC/SGC | 原样保留 |
| 原始 task instruction/initial DB/判分 | 原生任务定义 | 主测试全部保留，不插故障 |

官方本地模型的美元单价为零，美元预算不能有效约束本地计算。另报告总推理 token、候选前向次数、GPU/墙钟时间和 API 调用；不能用“$0”表示无成本。

### 5.1 数据版本和当前源码预期规模

固定提交的 `DATA_VERSION/DB_VERSION` 为 `0.2.0`。采用同版数据，不混装论文初版数据。[版本常量](https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/src/appworld/common/constants.py)；[规模测试](https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/tests/package/test_load_task_ids.py#L9-L26)

| split | 当前源码预期 scenarios | 当前源码预期 tasks | 使用方式 |
| --- | ---: | ---: | --- |
| train | 30 | 90 | 采数、训练；内部按 scenario 做机制留出 |
| dev | 19 | 57 | 超参数、提示冻结、误差分析 |
| test_normal | 56 | 168 | 最终完整任务聚合评测 |
| test_challenge | 139 | 417 | 最终完整任务聚合评测 |
| 总计 | 244 | 732 | 每 scenario 有 3 个任务 |

**这些数量是源码断言的预期，不是本轮已下载并计数的数据结果。** 首次准备数据时核验版本、split manifest 和哈希；若不匹配先解决版本一致性，不改数据清单凑数。ACL 2024 论文的 train/dev 为 105/60、总量 750，属于历史版本，不能与本方案混报。

Test-C 的历史设计包含未见应用挑战；不能直接将 Test-C 分数命名为“未见修复组合泛化”。后者需我们的独立机制切分证明。

### 5.2 TGC 和 SGC 的准确含义

- **TGC（Task Goal Completion）**：全部任务测试通过的任务比例，严格成功，不是每个任务通过了多少条断言。
- **SGC（Scenario Goal Completion）**：按 scenario 分组，该组所有任务都成功才记 1。它不是 Subgoal Completion。

`world.task_completed()` 只是 supervisor 的完成宣告/状态，不等于 evaluator 判定全部测试通过。[评分实现](https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/src/appworld/evaluator.py#L321-L366)

## 6. 4090 的确定起始配置

### 6.1 actor 和预测器

| 部件 | 起始配置 | 来源/地位 |
| --- | --- | --- |
| actor / 目标抽取 / 候选提议 | `Qwen/Qwen3-8B-AWQ`，非 thinking | 上游 Qwen3-8B 配方的量化适配；官方 AWQ 权重确实存在 |
| actor serving | vLLM、max_num_seqs=1、max_model_len=8192 | 我们的单卡配置；上游示例是 5 并发、32,000 context |
| actor 显存预留 | gpu_memory_utilization 先设 0.50 | 初始规划值，给预测器留空间；需显存测量确认 |
| predictor | `Qwen/Qwen3-4B`，NF4 + double quant，BF16 compute | 我们新增，文本任务不需视觉 encoder |
| predictor 输入 | 最长 4,096 tokens | 固定指令、目标、候选；历史按统一规则裁剪 |
| classifier | 每个目标共享 4 类 linear head | 我们新增；输出四态概率 |
| predictor 推理 batch | 1，顺序评估 K 个候选 | 初始保守配置 |

模型资料：[Qwen3-8B-AWQ](https://huggingface.co/Qwen/Qwen3-8B-AWQ)、[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)、[PEFT 量化训练](https://huggingface.co/docs/peft/main/en/developer_guides/quantization)。

训练预测器时停止 actor 服务，完整 24GB 给训练；推理时才同时放入量化 actor 与量化 predictor，并限制 vLLM 预分配。单卡可行性以短运行峰值实测确认，不以权重文件大小代替显存验收。本方案不依赖付费强模型生成候选。

统一上下文策略：actor 总窗口 8,192，输出最多 3,000，因此输入预算最多 5,192 tokens；固定任务说明与通用指令优先保留，先删除旧 observation 正文，再删最旧完整交互对，保留最新上下文。predictor 在 4,096 token 中优先保留 G、r 和最新观察。超出固定部分可容纳长度时记为显式 overflow，不悄悄截掉目标或候选。所有对照使用相同规则。

这项调整不是原库的 null 字符截断，也不是原论文的 20,000 字符历史限制。我们的分数必须称为“同版本、同模型和硬件配置下重跑”，不能声称与原论文数字直接复现一致。

### 6.2 训练默认值

下列训练值全是我们的起始设置，上游仓库没有可照搬的 B2 训练超参数。

| 项目 | 默认值 |
| --- | --- |
| 方法 | 4-bit QLoRA + 4 类分类头 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| target modules | q_proj、k_proj、v_proj、o_proj、gate_proj、up_proj、down_proj |
| microbatch / gradient accumulation | 1 / 16 |
| max sequence length | 4,096 |
| epochs | 3，按 dev 决策表现保存最佳 checkpoint |
| optimizer | AdamW，LoRA lr=1e-4、分类头 lr=1e-3，weight_decay=0.01 |
| schedule | cosine，warmup_ratio=0.03 |
| gradient checkpointing | 开启 |
| precision | BF16 compute；基础权重 NF4 |
| loss | 仅四分类交叉熵，mask 无法判定目标 |
| 训练 seeds | 100、101、102 |
| candidate K / lambda / beta | 4 / 2 / 0.05 |

开发阶段如确需调参，预先限制小范围，例如 LoRA lr `{5e-5, 1e-4}`、lambda `{1, 2, 4}`；只用 dev，不根据最终研究留出或官方 test 结果追加搜索。不要把这组默认值写成已实测最优值。

## 7. 实验矩阵

### 7.1 原生完整任务主表

| 方法 | 目的 |
| --- | --- |
| A. 官方 Simplified ReAct，同模型/量化/预算重跑 | 代码与环境 baseline |
| B. ReAct + 同候选直接选择 | 判断是否只靠多生成了几个修复方案 |
| C. ReAct + 同容量条件标量效用预测器 | 直接检验逐目标表示是否必要 |
| D. ReAct + B2 四态后果预测器 | 我们的方法 |

所有方法保留官方完整任务与 evaluator。B/C/D 使用相同抽取器、检查时点和候选生成规则；C/D 使用相同 4B 骨干、数据、LoRA 和训练预算。A 可另加目标清单版本，排除仅靠任务拆分改善的因素。

主表同时列 TGC、SGC、输出/输入 tokens、环境交互、API calls 和实际时间。除相同环境步上限外，再给相同总推理预算的比较；给 A/B 使用等量额外反思/采样的机会，避免把额外计算当结构化预测优势。4B 分类器前向虽不生成 token，也要计算处理 token 与 GPU 时间，不能混入“0 输出 token所以免费”。

### 7.2 机制诊断

固定可见状态与候选池，比较：

- 同骨干直接提示预测后果；
- 实体/读写相关性规则与局部修复选择；
- 同容量条件标量；
- B2 四态模型；
- 真实候选后果 oracle，仅离线使用；
- 可用沙箱试完候选的执行基线，单列权限和实测代价。

研究指标：01/10 macro-F1、有害影响漏检率、校准、当前状态估计误差；恢复后保留原已满足目标的成功率、回退数、候选池 oracle regret。恢复范围用目标/实体集合和实际修改衡量，另做相近 API 次数、相近候选长度的匹配比较。

置信区间按 scenario/task 分组配对计算；不能把同一受损状态的 K 个候选视为 K 次独立实验。对没有发生恢复或只有一个可判目标的样本分别报告覆盖，不从原生主表分母移除。

### 7.3 主实验次数和资源预算

官方测试按当前源码预期共 585 tasks。A/B 在固定 actor/environment seed=100 下各跑一轮；C/D 对三个训练 seed 各跑一轮，共约 8×585=4,680 条完整任务轨迹。开发诊断先用一版 seed，最终稳定性主要看 C/D，不机械重复确定性的原版三次。

以每轨迹平均 20 次代码交互估算，主表约 93,600 次环境交互；若平均每步含 actor 推理为 3/10 秒，仅基础部分约 78/260 小时，尚未计候选提议、预测、重放、训练与失败重试。因此整个项目应按数百 GPU 小时规划，而非声称单卡几天必然完成。平均步数与速度都是示例假设，首次 train/dev 基线运行后用实测值更新预算。

## 8. 计划新增的代码边界

以下是拟新增位置，不表示已经实现：

```text
experiments/code/b2/
  agent.py                  # 原生 ReAct 子类；检查点、选择和真实历史回写
  goals.py                  # 指令目标与 evidence_span
  candidates.py             # 冻结提议器，生成通用候选，不硬编码具体 API
  effect_predictor.py       # 4B QLoRA + 四分类头
  selector.py               # 固定 benefit/harm/cost 规则
  collect.py                # 独立进程重放前缀、配对标签
  train.py                  # 四态与条件标量共用训练基础
  evaluate.py               # B2 机制指标与覆盖率
experiments/configs/b2/
  baseline_4090.jsonnet
  effect_4090.jsonnet
  scalar_4090.jsonnet
```

原生 evaluator、task files、数据库定义和 API 实现不为 B2 修改。为新增 agent 注册类型、导入模块和运行入口做必要接入即可。日志记录已选候选、其他候选、概率、选择得分和真实执行；未选分支的 oracle 标签只出现在离线采集目录。

Python 3.11 起步。官方环境与 actor 沿用仓库依赖约束；预测器训练环境单独锁定 Torch/Transformers/PEFT/bitsandbytes，推理服务保留独立依赖环境，避免为了训练库升级破坏官方执行环境。各环境通过本地模型服务或文件通信，不安装不必要的 smolagents/openai-agents extras。

## 9. 已发现、实现时要处理的具体问题

1. **状态保存不完整。** 当前 `save_state()` 不传 ID 时返回 `None`，不是一个可加载的新 checkpoint ID；命名保存需 `save_state("id")`。它只保存 DB，不恢复任意 Python 变量、actor messages 和计数。第一版离线采数因此采用新进程重放前缀。[源码](https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/src/appworld/environment.py#L1279-L1355)
2. **多 world 不能简单共享线程。** unified 模式每进程同时只持一个 world，各分支使用唯一 experiment 名，避免数据库、时间冻结与输出互相污染。[官方并行指南](https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/guides/parallelizing_worlds.md)
3. **本地 Qwen 配方存在于 config generator。** 不要求仓库中已预生成同名每个 split 的 jsonnet；使用当前模型注册和模板生成并保存展开配置，不将过时 CI 配置当默认。
4. **训练目标尚需映射。** 官方测试给出执行真值，但不是现成的 B2 逐目标数据；goal/checker 映射是本项目新增工作，需要覆盖统计与审计。
5. **保留官方测试边界。** test 只用于最终自动运行和聚合结果，不查看逐任务报告、不据其改提示或训练。逐目标案例分析使用训练侧留出或开发任务。
6. **保护内容发布方式。** 公开方法代码可继承 Apache-2.0；受保护 bundle 及其衍生数据的公开再分发需按上游加密要求处理，优先发布重建脚本和 manifest。[许可说明](https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/README.md#L1604-L1618)

## 10. 本版决策

**采用 AppWorld 官方 Simplified ReAct 做唯一代码上游；用 Qwen3-8B-AWQ 固定 actor，用 Qwen3-4B QLoRA 学逐目标修复转移；原生 TGC/SGC 做主结果，训练侧 scenario 留出做机制验证。**

可照搬的是 agent 主循环、API/环境语义、原生 split、主评测和大部分运行参数。必须新增的是后果标签、预测器与候选选择；必须调整的是单卡量化、上下文和并发。范围选择的有效性由同候选、同容量标量、规则和 oracle 比较检验，不先假定它会胜出。
