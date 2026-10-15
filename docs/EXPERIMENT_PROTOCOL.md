# RepairScope 实验协议 v0.1

## 研究问题与当前证据

给定任务的可见历史、固定目标集合和一个候选修复代码块，模型能否预测各目标
在执行前后的满足状态，并用该预测选择合适的恢复范围？第一版研究有限候选池
中的下一代码块选择，不声称求出全局最小修复计划，也不假设多个动作效果可相加。

本地已有的是软件实现和 synthetic fixture 检查。真实 AppWorld/4090 兼容性、
可识别的目标标签、自然副作用频率和端到端收益都需要实测。本协议提供执行与
证据标准，不预先宣布研究问题或新颖性已经成立。

## 固定基线与任务划分

代码基线为 AppWorld 官方 `SimplifiedReActCodeAgent`，commit
`42b5bcf3cd334fee33f0c37c02070a9f5807add5`。继承其提示词、代码解析和任务
执行框架；通过独立 HTTP transport 记录 Actor 及辅助推理成本。

数据版本固定0.2.0。train90任务/30scenario；dev57/19；test_normal168/56；
test_challenge417/139。四个划分的 task ID 与 scenario ID 均不交叉。manifest
只检查 test ID 清单，不为了训练或调参读取 test 目标标签、checker 或私有数据。

训练仅使用 train。官方 dev 用于统一设置与 checkpoint 选择；机制泛化另做
train-side scenario holdout，全部同 scenario 的任务、状态、候选一起切分。
测试开始前冻结配置。两个 test split 分别完整报告，主测试分母分别为168和417。

## 可见输入、目标与候选

每个任务先从原始 instruction 抽取固定目标，保存 exact evidence span。
只有至少2个目标且所有目标都是可判定 `state_write` 的任务启用干预。
回答型、轨迹型、抽取失败或其他不满足条件的任务继续执行原生 ReAct，仍计入
主评测完整分母。报告 eligible/task 比例与实际干预次数。

模型输入只含任务指令、目标及已执行的可见历史、候选代码和公开静态成本估计。
不得包含 evaluator、ground truth、checker requirement、其他候选的执行结果、
未来观察或真实目标满足状态。当前状态不可见时模型必须从观察推断，不能补给
隐藏状态当作已知事实。

固定触发器是：可见执行错误、每5个已执行代码块、原 Actor 准备宣告完成。
同一状态生成四候选：

| ID | 含义 |
|---|---|
| `original` | 原 Actor 下一代码块，保持字节完整 |
| `local` | 局部修复 |
| `composite` | 处理相关目标组合 |
| `broad` | 从当前状态进行较广重规划 |

每个候选是完整可执行的下一代码块，不是无限长计划。三个新候选各限600
token，统一生成预算2400输出 token；原代码保持 Actor 的3000上限。语法/
预算/生成失败时显式回退并记账，不截断代码伪装为成功候选。

同一干预点先生成候选再选择，只执行选中代码。在线不免费试跑其他候选、不
恢复数据库快照、不重复撤销同一个实际操作。恢复范围由选中候选涉及的目标/
实体体现；不将代码长度等同于范围。

## 方法与主对照

后果模型对每个目标输出四态分布，顺序固定为 `00,01,10,11`：

- 00：未满足→未满足。
- 01：未满足→满足，即恢复。
- 10：满足→未满足，即破坏。
- 11：满足→满足，即保留。

候选效用为：

\[
U(r)=\frac{1}{|G|}\sum_g\left[p_{01}(g,r)-2p_{10}(g,r)\right]
-0.05\frac{\widehat c(r)}{\max(1,\max_{r'\in R}\widehat c(r'))}.
\]

`estimated_api_calls` 是静态代码调用数估计，不是真实执行成本。真实 API
调用数由执行日志另报。相同效用先选更低估计成本，再按原候选顺序破同分。
主实验统一使用破坏权重2、成本权重0.05，不能训练用一组、在线选择又用另一组。

| 方法 | 行为与用途 |
|---|---|
| baseline | 原生 ReAct；没有 goal/candidate 辅助调用 |
| direct | 相同目标/候选/观察，冻结 Actor 直接选择候选 |
| scalar | 4B 同骨干/同输入/同监督，直接回归该效用 |
| effect | 4B 逐目标四态预测，再用固定效用选候选 |
| rule，可选 | 冻结 Actor 零样本逐目标后果清单，不能称为符号依赖图 |
| capture，仅采数 | 产生候选但继续执行 original；成本单列，不作 native 主对照 |

单步目标效用也可能低估必要的读取/信息获取操作，或在所有状态目标已满足时偏好无效果候选而延迟完成。候选提示词禁止故意noop，但这不是语义保证；pilot必须检查真实完成行为与信息操作。如出现系统性问题，先在train/dev修复统一策略、重新冻结并完整重跑相关对照，不能在test阶段临时改动。

本版本不含运行时 oracle、RL、额外多智能体或依赖图。离线 paired-candidate
oracle 只用于说明候选池内可改善空间和 regret，不可作为在线可部署方法。

baseline 与干预方法的辅助计算开销不同，主表必须同时报告效果与实际成本。
direct/scalar/effect 使用同一候选生成器与触发规则；在线轨迹分化后，状态和
候选内容自然会不同，不能声称它们在线始终拥有完全相同的候选池。离线同状态
机制诊断才使用完全配对的四候选池。

## 标签来源、审计与便宜 pilot

从同一自然受损状态分别执行各候选获得标签，全部限定 train/dev。每个分叉
在新进程、新 world 和独立 experiment 中完整重放历史，保留失败代码的部分
数据库写入与 Python 局部状态。接受候选前检查 reference 与 branch 的观察、
语义 DB 变更指纹和执行前 checker 状态一致；不一致的样本拒绝并保留 receipt。

先固定 instruction-level goals，再把 evaluator requirement 分组为其 checker。
不能把每条 assertion 自动等同于一个 goal。联合/全局/不完备/非状态 checker
保持 unknown；共享 checker 不能为多个独立目标提供已知标签。未知标签用
`-100`，从不当成00或负例。详见 [LABEL_AUDIT.md](LABEL_AUDIT.md)。

先完成3-task接口 smoke，然后扩展真实 train/dev 小采数，审计至少30个 goal，
尽可能覆盖至少5个独立 scenario 及不同 checker 结构。检查 evidence、目标
独立性、checker充分必要性和部分恢复情况。30个审计不是标签准确率的统计证明；
所有已发现的 known-label 错误都先修正并重采。

建议将以下数值作为**工程推进门槛**，在投入完整训练前记录是否达到：

| 项目 | 起始工程门槛建议 |
|---|---|
| 分叉一致性 | 已尝试候选接受率≥95%，拒绝原因已检查 |
| 语义审计 | ≥30个goal、尽量≥5scenario；纳入known的已发现错误已修复 |
| 完整候选池 | ≥50个真实、四候选、全目标可标注状态 |
| 01恢复机会 | ≥20个状态，跨至少3scenario |
| 10破坏机会 | ≥10个状态，跨至少3scenario |
| 候选确有差异 | ≥20状态的候选恢复/破坏计数或效用不同 |
| 未知标签比例 | 建议≤30%；完整标签公共训练 cohort 足够，排除原因可解释 |

这些值用于发现“没有训练信号”“checker不可分解”“只记住一个模板”等工程
问题，不构成效应量、统计功效或顶会贡献证明。未达标时先分析自然失败分布、
候选质量和标签可识别性，再决定是否增加 train/dev 采数；不能悄悄降低门槛，
或把人为故障/fixture混入自然主实验来填数。若自然10极少，应如实缩小副作用
结论，甚至调整研究问题，而非制造论文所需现象。

便利脚本的最低失败检查可采用更低的起点：至少有完整候选池，train真实01
状态、10状态、候选差异状态各至少10。脚本通过只表示排除了最明显的空数据
问题，不表示已经达到上表的覆盖建议，更不表示足以投稿。将实际门槛、检查
结果和继续/停止决定保存到本次实验记录，不把建议值偷偷改写成结果。

## 4090 训练与推理配置

| 项目 | 固定主配方/待测边界 |
|---|---|
| Actor | Qwen3-8B-AWQ，冻结，thinking关闭，temperature0，seed100 |
| Actor context/输出 | 8192/3000 token，全部方法一致 |
| Actor服务 | 独立vLLM环境，并发1，初始显存比例0.35 |
| Predictor | Qwen3-4B，NF4＋LoRA；保留resolved revision |
| LoRA | rank16，alpha32，dropout0.05 |
| Predictor输入长度 | 4096，超长显式排除/回退，禁止截断goals/code |
| 优化 | batch1、累积16、最多3epochs、AdamW、骨干lr1e-4、head lr1e-3、cosine、warmup0.03 |
| 训练种子 | 100、101、102；scalar/effect配对使用 |
| validation | 主配方官方dev，checkpoint按相同验证原则选择 |
| 执行预算 | 每任务50代码块，100000模型输出token；辅助输出计入 |
| 单卡调度 | 采数/在线时Actor，训练时仅停止自己启动的Actor；训练器串行 |

effect 用四分类交叉熵，scalar 用效用 MSE。两者采用同一 fully-labeled
cohort，输入均含完整目标集合、候选、可见历史与效用条件。effect 使用部分
unknown数据可作额外消融，并明确说明监督量不同。类别加权若做必须单列，
仅从 train 统计并对选择协议冻结，不能只给主方法额外调参预算。

同卡 actor约8GB、NF4 predictor约4GB仅是初始资源估算。先测最长真实输入下
实际整卡峰值和吞吐，再启动完整在线任务；训练停止actor后另测峰值。batch1
也可能因模型依赖、上下文或激活导致OOM。包版本、精度、GPUutil及上下文调整
都要记录，所有受影响方法统一使用同样硬件设置。

## 评估与判定

第一层是 train/dev 后果诊断：四态 macro-F1、每类 support、10 recall/漏检率、
ECE/Brier/NLL；并报告完整候选池内 regret、oracle与原候选的空间、效用排序和
真实恢复/破坏计数。对 unknown 与缺类情况报告 null/coverage，不填0假装准确。

第二层是完整任务闭环：官方 TGC、SGC、eligible覆盖、干预次数、终止原因、
实际执行块/API调用、Actor及辅助模型token/秒、predictor处理token/秒、总墙钟。
离线采集成本、训练GPU小时和在线成本分别列出。

TGC衡量完整任务成功；SGC要求一个scenario下3个任务都成功。它们均不等于
逐目标准确率。ineligible、模型失败、候选回退和预算耗尽任务不得从主分母删除。
不能只在多目标写入子集上报主TGC；该子集可作为有明确分母的辅助分析。

三个 predictor seed 分别保存结果，报告离散程度。seed重复不产生新的任务
样本；若计算置信区间或配对检验，应以scenario为聚类单位，保留同一scenario
中的3任务关系。当前代码的机制bootstrap不自动证明完整native结果显著；不要
将有限sample的oracle或一次最好seed当作普遍结论。

进入完整测试前，要求真实短训练、checkpoint重载、scalar/effect同数据输入、
dev闭环、预算记账和自然01/10诊断均通过。测试期间只接收聚合原生评分，不做
逐测试任务调参或依据test错误挑选方法版本。发现实现错误时记录并统一重跑
全部受影响方法。

支持方向成立的证据应同时包含：存在真实候选恢复空间；后果预测在未见
scenario/目标组合上有效；相较同容量scalar和direct，effect改善完整任务效果
或明确的保留/成本权衡。仅训练loss下降、四态准确率提高、或优于局部重试不足
以支持完整方法贡献。与scalar持平、收益只在熟悉模板出现、或副作用没有可
识别标签，同样是需要保留并据此调整方向的结果。

## 复现产物

保留Git及上游commit、实际依赖与模型revision、dataset manifest、审计映射
哈希、采集状态receipt、训练/验证scenario清单、训练配置与最佳checkpoint、
GPU峰值/吞吐、全部run receipt和native aggregate。原始私有数据和密钥不push。
`synthetic_fixture` 与 `synthetic_fault` 不能混入自然主实验；当前正式HF训练
入口要求自然数据，tiny backend只接受fixture。任何真实实验完成声明都必须
能落到上述对应产物，而非本地fixture测试或计划表。
