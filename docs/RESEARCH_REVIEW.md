# B2：修复后果预测与恢复范围

研究领域选择 · 最新近邻 · 单 RTX 4090 主实验 · CCF-A 会议路线

> 建议主线：多目标工具型智能体的可靠恢复。以 AppWorld 做机制主实验，以 Agent-Diff 做独立工具域检验；Web GUI 作为优先迁移方向。具身若参与，选家庭任务层，冻结底层技能。

判断是“值得做廉价证伪”，尚不足以直接承诺 CCF-A。动作后果、反事实干预、最小修复和保留已有进展都有直接前作。潜在贡献在于：面对未知、状态相关的修复后果，预测哪些目标恢复、哪些目标回退，并证明这类结构化预测对恢复范围有不可替代的决策价值。

检索截止为 2026-10-15（北京时间）。本报告将论文事实、作者报告结果与研究建议分开。未复现文献数字；未登录远端、下载模型、训练或开展主实验。4090 结论是资源与工作流可行性判断，不是该服务器的运行验收。

## 目录

- [01 研究问题与可检验边界](#definition)
- [02 领域热度与选择](#domains)
- [03 最直接近邻：数字智能体](#neighbors)
- [04 经典规划、具身与软件工程](#crossdomain)
- [05 环境选择与数据可用性](#environments)
- [06 最小方法与实验设计](#design)
- [07 先证明值得学：去留门槛](#falsify)
- [08 单 4090 的真实预算](#compute)
- [09 CCF-A 会议路线与结论](#venue)
- [参考文献与官方资料](#references)

<a id="definition"></a>

## 01 研究问题与可检验边界

最新 B2 的研究单位是“一次候选修复对多个目标的影响”，旧版预算验证调度只提供背景。应将“给定一个修复操作的预测任务”与“在多个修复方案中选择恢复范围的决策任务”分别定义。

### 预测什么

输入：当前允许观测到的状态及历史 x、原始目标集合 G、一个候选修复 r。r 可以是一条 API 操作，也可以是具有明确终止条件的短程序或技能。输出每个目标在修复完成后变为满足或不满足的概率，据此得到恢复集合、回退集合与不变集合；信息不足时允许输出不确定。

“恢复”要求目标修复前未满足、修复后满足；“回退”要求目标此前满足、修复后失效。不变应区分持续满足与持续未满足。只报告整体成功概率，不能完整回答用户提出的目标级问题。

### 范围如何成为真正的决策变量

需要候选集合 C(x)，每个候选包含操作序列、修改的实体/目标集合及代价。可选择局部操作、跨分支组合修复、重新规划或合法重建。若只给定唯一且不可改变的操作，影响范围可以预测，但恢复范围没有可选择性。

> 建议定义：在候选池和预算内，选能恢复必要目标、满足保留约束且代价较低的修复方案。最小性只相对于该候选池成立；不宣称求得全局最小修复。

### 时间与信息边界

主标签先取宏操作结束后的稳定状态，再用固定续跑策略评估延迟影响。取走盘子可能暂时破坏“盘子在桌上”，若技能随后放回，不能混作永久损害。多步修复可能存在协同，单动作向量不能直接推出完整恢复闭包。

完整数据库/图状态可用于离线判定真值；主部署模型只能读获准的观察、工具返回和历史。目标描述由任务指令形成，隐藏测试代码、答案、未来状态和执行后的 oracle 不进入输入。当前状态版本、故障状态版本和候选修复后的版本需要严格对应。

### 最有价值的研究假设

在未见目标组合、修复组合或用户保留优先级下，目标级后果表示比同容量的条件标量效用模型、完整下一状态预测和静态依赖规则，取得更好的保留率与恢复代价权衡。这是待检验假设，不是向量表示天然优越的定理。

<a id="domains"></a>

## 02 领域热度与选择

热度根据持续的一手工作、正式会议成果、基准与开源生态判断；没有统一口径的全领域论文计数，因此不提供伪精确增长率、引用排名或录用概率。下表的优先级是针对 B2、单 4090 和 CCF-A 目标的判断。

| 领域 | 活跃性与竞争证据 | 适配性与建议 |
| --- | --- | --- |
| 跨应用工具 / workflow | AppWorld 为 ACL 2024 正式资源；2026 有 Agent-Diff、VoS、DARC、Sentry 等独立路线。[1](#ref-1)[2](#ref-2)[5](#ref-5)[12](#ref-12)[13](#ref-13) | 热度高，目标真值与反事实分支清楚。主域首选；必须面对免费试运行和规则基线。 |
| Web GUI / computer use | FCPAgent、SafePred、ProRe 及近期 latent world action model；恢复、预测、验证均拥挤。[3](#ref-3)[4](#ref-4)[8](#ref-8)[11](#ref-11) | 问题直观且有生态价值。第二优先；适合检验由工具状态转向截图/界面观察的迁移。 |
| 全桌面 GUI | OSWorld 提供真实应用任务；状态涉及窗口、文件与异步保存。[26](#ref-26) | 受 VM、KVM、重置与执行噪声限制。可用既有 B2 经验做小规模外部验证，不先承担主采数。 |
| 具身家庭任务层 | EAI（NeurIPS 2024）、ENACT（ICLR 2026）、AHA、FAER、MAGMA-GEN 等。[16](#ref-16)-[20](#ref-20) | 大方向高热。语义自然，但已知规则下强符号基线几乎直接给答案。可作替代主域或后续验证域。 |
| 低层机器人 / VLA | 恢复、失败解释与世界模型持续活跃；多模态与物理控制投入高。 | 当前优先级低。单 4090 可运行部分仿真，难同时承担策略训练、复杂仿真和完整消融。 |
| 软件修复 / 变更影响 | ICSE 2026 RIPPLE 预测 change impact set；LLM4Reg 研究回归检测。[23](#ref-23)[24](#ref-24) | 热度高但成熟问题强重叠。仅有明确代码影响分析或测试贡献时走 ICSE/FSE/ASE/ISSTA。 |
| 经典规划 / 安全副作用 | plan repair、plan stability、goal preservation、AUP 已成体系。[14](#ref-14)[15](#ref-15)[21](#ref-21)[22](#ref-22) | 提供必要理论与强基线；不能把“少改动、别破坏其他目标”再当全新原则。 |

建议的论文定位是：可靠智能体、目标条件的修复后果学习与受约束决策。环境服务于这个问题。如果额外引入视觉，应证明视觉观察确实改变预测难度，而不是只给符号任务加截图。

<a id="neighbors"></a>

## 03 最直接近邻：数字智能体

| 工作 / 状态 | 已经解决的部分 | B2 必须越过的门槛 |
| --- | --- | --- |
| FCPAgent [3](#ref-3)<br>2026-07 arXiv | 以可证伪承诺表示子目标；动作前后检测；按执行、技能、规划层级修订最小充分部分。 | “确定合理恢复范围”已撞题。需检验跨目标后果、非前后缀的稀疏修复，而非再做层级分类。 |
| SafePred [4](#ref-4)<br>2026-02 arXiv | 基于状态、候选动作与策略预测短长期风险，再做步骤干预和任务重规划。 | “给 state+action 预测后果再决策”不新。要预测具体目标的恢复与回退，并超过风险/效用基线。 |
| VoS [5](#ref-5)<br>2026-10-06 arXiv | 用约 82K 反事实续跑学习干预收益，以 harm budget 控制对成功轨迹的打扰；已用 AppWorld。 | 最强近邻之一。多目标标签必须带来组合泛化或保留收益；只是标量改多输出头，贡献不足。 |
| WebWorld [6](#ref-6)<br>2026-08 arXiv | web-code 修复后用浏览器检查目标进展及已验证能力的保留，再接纳修复。 | 修一处坏一处、progress+preservation 都已有。区分事前后果预测与事后全量验证。 |
| CHILL-Harness [7](#ref-7)<br>2026-07 arXiv | 用反事实干预学习 harness orchestration 的收益，约束成功保持。 | 反事实干预、收益学习、选择性验证不能单独算新点。区分环境修复与 harness 调整。 |
| ProRe [8](#ref-8)<br>ICLR 2026 正式论文 | reasoner 安排探查任务，actor 主动观察后评价 GUI 完成情况。 | 要区分无法看清当前目标和无法预测动作后果；额外观察是必须比较的替代方案。 |
| RIR / Sentry [9](#ref-9)[13](#ref-13)<br>2026-09 / 10 arXiv | 前者回滚环境并保留反思；后者选择干预、软/硬修复与故障条件经验调用。 | 选 checkpoint、保留有效前缀与选择恢复时机均已有。B2 应控制语义目标/实体范围。 |

### 一般 world model 的竞争也已很强

WebWorld 的 2026-02 论文使用百万级开放网页交互；Qwen-AgentWorld 报告千万级跨域轨迹。[10](#ref-10)[35](#ref-35) 最新 Latent World Action Model（2026-10-09）已把预测从完整屏幕转为动作相关的潜在变化。[11](#ref-11) 因此“不生成完整图片、只预测有用变化”也不足以单独成为创新。

注意两篇 WebWorld 不同：[6](#ref-6) 是网页代码修复与保留验证，[10](#ref-10) 是 Web agent 训练的环境世界模型。本文将新 arXiv 当竞争证据，不将其自动写成顶会录用或已独立验证结果。

<a id="crossdomain"></a>

## 04 经典规划、具身与软件工程

### 经典规划给出的概念边界

AAAI 2022 的 Planning to Avoid Side Effects 已研究保护其他目标/策略；NeurIPS 2020 的 AUP 用其他目标能力变化惩罚副作用。[14](#ref-14)[15](#ref-15) ICAPS 的 plan repair / plan stability，以及 C-TFLAP 的承诺保持，说明小恢复范围、保留进展和避免连锁损害都不是新概念。[21](#ref-21)[22](#ref-22)

如果已给完整状态、准确动作模型与目标谓词，直接前向模拟和规划通常能计算影响。B2 的学习必要性应来自有限观测、未知或条件性后果、组合泛化，以及受限观测/执行预算，不能通过故意不给规则基线必要信息来制造优势。

### 具身最近邻

EAI 已将 transition modeling 作为评测模块，预测前置条件与动作 effects；ENACT 已评第一视角交互的正向/逆向世界建模。[16](#ref-16)[17](#ref-17) AHA 研究合成失败与解释，FAER 扩展到长程失败纠正；MAGMA-GEN 已从同状态重执行候选局部修复并验证后续进展。[18](#ref-18)-[20](#ref-20)

因此具身路线应保留 B2 的核心问题：多目标修复的正负后果、保留约束和合理范围。若采用 VirtualHome/EAI，只冻结技能、研究任务层。若必须学习抓取/移动控制器，难以解释改善究竟来自恢复决策还是执行技能。

### 软件工程最近邻

ICSE 2026 RIPPLE 的输入包含变更意图和种子编辑，输出影响方法集合；LLM4Reg 面向回归 bug 检测。[23](#ref-23)[24](#ref-24) SWE-bench 的既有失败转成功与通过转失败测试思想也与“恢复且保留”天然接近。仅将代码方法换成目标标签不足以建立贡献。

软件路线只有在研究对象确实是代码编辑、影响分析、补丁正确性或测试选择时才值得主攻。届时需处理仓库依赖、测试不完备与潜在污染，不能把单个旧基准视为无误真值。[25](#ref-25) 这是一条可能更贴合软件工程 A 会、但并不更省事的路线。

### 其他可选但暂不优先的落点

τ²-bench 的故障排除和双控制任务适合补充工具恢复，但用户模拟和交互策略又增加变量。[36](#ref-36) AIOps/微服务修复的 blast radius 很自然，不过有效主实验往往需要真实负载、故障注入、系统依赖与运行环境。缺少这类资源时不建议只为题目新鲜而迁入。

> 所有领域共同的审稿问题：为什么需要“学后果”，它比已知模型计算、直接执行验证、或直接学最终效用更好在哪里？换环境不能替代回答这个问题。

<a id="environments"></a>

## 05 环境选择与数据可用性

| 环境 | 可用条件与匹配度 | 主要限制 |
| --- | --- | --- |
| AppWorld [1](#ref-1) | 跨应用 API 环境，有数据库状态测试并检查 collateral damage；适合配对执行后建立目标效果真值。 | 原生断言不等于独立语义目标；需任务分组、恢复候选和状态分支。不能把已有 benchmark 标签改名当新数据集。 |
| Agent-Diff [2](#ref-2) | 224 个企业 API 任务，容器化 replica、state-diff contract；适合独立工具域。 | 需另行做目标分解与修复数据；CRUD 结果不代表视觉或物理泛化。 |
| WebArena / BrowserGym [27](#ref-27) | 自托管网站与功能性评测，可检验界面观察下的恢复。 | 完整状态还含 DB、登录/session 与 browser；恢复不能只重载截图。 |
| OSWorld [26](#ref-26) | Office、文件、系统任务中副作用直观；可继承旧 B2 经验。 | KVM/VM、CPU/RAM/SSD、异步保存与 grounder 都可能成为瓶颈；不是有 4090 就能并发跑。 |
| VirtualHome / EAI [16](#ref-16)[28](#ref-28) | 纯图执行便于复制和谓词检查；具身任务层的低成本首选。 | 完整动作规则构成极强基线；图引擎与真实 Unity/物理语义不能直接等同。 |
| BEHAVIOR [29](#ref-29) | 家庭目标与共享物体/资源丰富，可保存和加载仿真状态。 | 官方最低 32GB RAM/8GB VRAM 只代表启动门槛，复杂场景和 VLM 会竞争资源。 |
| ManiSkill3 [30](#ref-30) | 官方有 4090 性能基准和状态保存接口，可做物理分支。 | 现成单目标控制任务不足以支持 B2；多目标语义任务、稳定技能和额外状态仍需开发。 |

### 推荐的实验组合

资源最稳：AppWorld 主实验 + Agent-Diff 第二工具域。若机制成立且论文要强调 computer use，再加一个自托管 Web GUI 子集。不要同时承诺完整桌面 GUI 与具身；二者都会把问题变成多个系统项目。

### 采数与发布的现实条件

只在训练/开发来源生成训练分支；测试任务及其相邻状态不进入训练。自然 agent 错误与人工故障分别统计。目标清单来自指令和人工核验，不能直接暴露隐藏测试。候选来自冻结提议器或公开可实现的规则，不读取测试答案。

AppWorld 的保护部分及其衍生内容有加密再分发要求。[1](#ref-1) 若发布恢复数据，应提供合规打包或生成脚本；新领域扩展不能直接假定可明文上传。标签执行的可复原性需要先做配对一致性检查。

<a id="design"></a>

## 06 最小方法与实验设计

### 方法保持简单

一个共享的小模型骨干，输入观察/历史、目标列表和候选修复，输出目标级后果概率。决策器按恢复目标、保留约束和真实代价选择候选；信息不足时选择进一步观察或更保守的方案。首先采用监督学习与校准，不引入在线 RL、多智能体协商、视频生成或复杂动态图。

使用明确的候选池可隔离预测能力与修复生成能力。先固定提议器和执行器，所有方法共享候选及动作预算。之后才评开放提议，避免把更强的修复生成误归因于后果预测。

| 需要比较的基线 | 回答什么问题 |
| --- | --- |
| 局部重试 / 全量重规划 / 合法重建 | 简单策略是否已足够，收益是否只是更大恢复预算。 |
| API schema + 实体读写集 + 依赖闭包 | 状态相关的学习是否真超过现成规则与运行时追踪。 |
| 同容量条件标量效用 / VoS 风格模型 | 显式目标后果是否优于直接预测代价、伤害与成功；标量模型必须看到相同目标和优先级。 |
| 完整下一状态 / state-diff，再投影到 goals | 紧凑后果表示相对于一般 dynamics 预测有何数据、延迟或泛化优势。 |
| 基模型直接提示 + FCP/SafePred 风格恢复 | 训练是否必要，与已有范围修订、风险预测相比有何额外贡献。 |
| 允许的沙箱试运行 / oracle effect 上界 | 预测有没有现实价值；上界与部署可用信息必须分别标记。 |

### 关键切分与机制对照

按原始任务家族、损坏状态来源、实体关系和操作组合切分，不按分支随机切分。测试未见目标共现、修复组合、状态条件和保留优先级。同一动作在不同状态产生相反后果，是排除记工具名称的必要检查。

重点案例：需要重修 A 与 C、保留 B 与 D 的非后缀范围；恢复收益相近但影响不同保护目标的候选；局部看似成功、续跑后再次失效的候选。强标量模型若也能处理这些情况，应如实承认，不能预设其不可区分。

### 指标分两层

预测层：恢复/回退 macro F1、效果集合 exact match、有害效果漏检率与校准。决策层：必要目标恢复且原有目标保留的比例、回归数、修改实体/目标数、总动作/观察/推理代价和相对候选池 oracle 的遗憾。对任务家族做配对置信区间；共享状态下的多个分支不能视为独立样本。

<a id="falsify"></a>

## 07 先证明值得学：去留门槛

> 目前的建议是条件性推进：先做约 100 个受损状态的廉价诊断，再决定训练。以下规模与阈值是建议的研究门槛，不是已经观察到的结果。

### 第一关：环境里真有范围权衡吗

每个状态准备 3-5 个候选，包含局部修复、跨分支修复与更宽重建。检查是否频繁出现“能恢复，但会破坏其他目标”，以及是否存在更小且足够的替代方案。若几乎没有副作用，或全量重建同样便宜且更稳，这个环境不适合主论文。不要为了制造结果专挑手工反例。

### 第二关：oracle 上界值得追吗

从同状态独立执行候选并全量验证目标，得到候选池内的可行上界。若 oracle 相比最强简单策略没有稳定且有实际意义的优势，说明不是预测器差，而是任务中没有值得学习的决策空间。可预先把约 10 个百分点的保留约束恢复率上界差距作为继续研究的参考值；它是项目门槛，不是统计定律。

### 第三关：规则与条件标量会不会直接解决

先跑实体读写/依赖规则、基模型提示和同容量标量效用基线。对完整模型可得的设置，加入准确符号模拟。若它们已接近 oracle，或目标向量仅增加解释性、不改善决策或泛化，方法论文的核心贡献不够，应改问题或停止。

### 第四关：试运行是否比预测更划算

公开两种真实协议：可分叉且可可靠检查的沙箱；部署时无法克隆、操作不可逆或检查不完备的环境。在第一种协议里实测试完所有候选与预测筛选的成本；模拟器执行很快时，预测完全可能输。第二种协议只能依据真实能力边界成立，不能人为加延迟或禁止本来可用的免费回退。

### 第五关：信息和执行没有偷渡优势

模型不能读测试、未来状态或 oracle 依赖；范围由候选操作实际修改的对象决定。比较无干预续跑，排除单纯重采样收益。先测快照一致性、候选有效性和预测时点，再看结果。自然故障与注入故障、精确状态与部分观测分别报告。

### 进入主实验的条件

存在非平凡上界差距；简单规则不够；基线间有可靠的恢复/保留权衡；配对真值稳定。满足后再投入 5k-20k 级状态-候选样本与小模型训练。若不满足，应给出明确否定结论，不靠堆模块、扩大模型或换一个更复杂叙事补救。

<a id="compute"></a>

## 08 单 4090 的真实预算

结论：24GB RTX 4090 足以支撑“离线采数 + 4B 级参数高效训练 + 8B/9B 量化复核 + 受限在线恢复”的主实验。它不支持本项目从头训练大 world model，也不适合先做完整 VLA/GUI policy 的高开销 RL。模型能装入显存与整套实验能做完是两件事。

| 配置 | 资源判断：均非本机实测 |
| --- | --- |
| 主预测器 | 4B 级模型，优先文本/结构化观察；可选已公开 Qwen3.5-4B 或成熟 Qwen3 系列。冻结视觉部分，仅训 adapter/head。[31](#ref-31)[32](#ref-32) |
| 较强复核模型 | 8B/9B 采用 4-bit 推理或 QLoRA；从短上下文、batch 1 起。两个模型按阶段顺序加载，避免同时常驻。 |
| 显存账 | 4B 的 BF16 参数约 8GB；8B 约 16GB。4-bit 参数理论约 2/4GB，但量化元数据、未量化模块、缓存和激活另计。参数大小不是峰值显存。 |
| 训练边界 | 4B 全参数 Adam 混合精度通常仅模型/梯度/优化器就约 48-64GB，尚不含激活。24GB 应优先 LoRA/QLoRA、梯度检查点和短序列。[33](#ref-33) |
| 主机资源 | 工具环境建议 8-16 CPU 核、32-64GB RAM、100-200GB 可用 SSD；完整 Web/VM 通常需要更多。这是规划建议，不是各环境官方最低配置。 |
| 环境与训练分时 | AppWorld/图仿真采数尽量走 CPU；停止仿真后训练；视觉验证再开启单实例。复杂 BEHAVIOR 不与训练同占 24GB。 |

### 数据规模与时长的透明算法

先诊断 100 states × 4 candidates，约 400 个配对分支。正式预测集可从 2,000-5,000 个独立状态、每状态 4 个候选开始，得到约 8k-20k 状态-候选对。独立任务模板数比复制状态数量更重要；天然多目标比例需要先实测。

时长按 T = 样本数 × epoch × 平均 token / 有效处理速度估算。例：10k 对、2 epoch、1.5k token，共 30M token；若实际训练速度为 100/300 token/s，则约 83/28 GPU 小时。三种子约 250/83 小时，尚未含对照、推理和评测。速度是假设场景，不是 4090 测速结果。

闭环例：200 个状态 × 5 种策略 × 3 次重复 × 平均 10 步 = 30k 步。若每步 3/10 秒，单此项约 25/83 小时，还需重置。先用同一离线分支结果比较选择策略，再对代表性的 2-3 个对照做真实闭环，避免重复生成相同标签。

正式预算前应测 50-100 个分支和短训练吞吐，记录峰值显存、长度分布、失败重置率及每条分支耗时。以上说明“有单卡路线”，没有承诺几天完成，更没有完成服务器实测。

<a id="venue"></a>

## 09 CCF-A 会议路线与结论

### 最新分级核验

2026-03-31 发布的第七版 CCF 目录，其正式 PDF 列 ICLR 为 A、IJCAI 为 B；AAAI、NeurIPS、ACL、CVPR、ICCV、ICML 为 A。[34](#ref-34) 核验采用丽水学院科研处转载的正式原始 PDF，人工智能表在 PDF 第 57-58 页，软件工程表在第 27 页。CCF 发布页有检索证据，直接访问遇到网站验证；未用旧静态子页覆盖新 PDF。[37](#ref-37)

| 投稿路线 | 当前匹配度与必须形成的贡献 |
| --- | --- |
| AAAI | 当前问题形态最自然：明确的可靠决策问题、跨两类环境、强规划与学习基线。不能只有若干 prompts 与流程拼接。 |
| ACL 主会 | 工具使用、目标/后果语言表示、可复核数据及组合泛化更契合。只多报一个恢复标签集仍不够。Findings 不满足本次 CCF-A 主会目标。 |
| ICLR / NeurIPS / ICML | 若结构化后果学习具有显著数据效率、OOD 组合泛化、校准或约束决策优势，可优先考虑。现阶段应视为需要证据支撑的路线，不能仅凭接 world model 命名。 |
| ICSE / FSE / ASE / ISSTA | 正式 PDF 列为软件工程 A 类。只有当问题落在软件操作、变更影响、回归与恢复验证，并提出明确 SE 贡献时适配；泛化工具 benchmark 本身不自动构成 SE 论文。 |
| CVPR / ICCV / CHI / UIST | 前两者需要视觉能力不可替代；后两者通常需实质交互贡献与合适用户证据。当前不应为迁就 venue 增加视觉或用户研究。 |

### 我的最终建议

> 研究领域：多目标工具型智能体的可靠恢复。主环境：AppWorld；独立验证：Agent-Diff；优先迁移：Web GUI。先做廉价去留诊断，证明后果预测的必要性，再决定是否启动 4090 主实验。

更有力的论文命题是：“学习状态相关、可组合的修复后果，使智能体在恢复失败目标时保留其他目标，并在新目标组合下选择较小且足够的恢复方案。” 这保留了最新 B2 的问题，也提供了清楚的反例与否定条件。

若你必须在“具身 / GUI”二选一，我选 Web GUI，并使用可验证工具环境建立机制证据。若优先目标是单卡做出有说服力的 CCF-A 工作，我不建议现在锁死纯 GUI，更不建议迁入低层具身控制。

所有方法优势、规模和门槛仍属研究建议。文献调研能排除明显重复，却不能证明无人研究过或保证录用。当前最值得投入的是必要性诊断；最不值得的是先训练一个大 agent，再期待恢复模块自然形成论文贡献。

<a id="references"></a>

## 参考文献与官方资料

所有链接均为论文原文、会议/机构页面或作者官方项目。arXiv 是竞争工作证据，不自动代表同行评审结论。数字若出现在正文，均保持原始研究的报告边界。

<a id="ref-1"></a>

### [1] AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents

ACL 2024 正式论文；环境、目标与 collateral 判定。

[原文 / 官方资料](https://aclanthology.org/2024.acl-long.850/) · [补充链接](https://github.com/StonyBrookNLP/appworld)

<a id="ref-2"></a>

### [2] Agent-Diff: Enterprise API Tasks with State-Diff-Based Evaluation

2026-02 arXiv；独立企业 API 环境，非已核实 KDD 录用。

[原文 / 官方资料](https://arxiv.org/abs/2602.11224)

<a id="ref-3"></a>

### [3] Falsifiable Commitment Planning for Self-Correcting Web Agents

2026-07-27 arXiv；scope-aware repair 的最直接近邻。

[原文 / 官方资料](https://arxiv.org/abs/2607.24167)

<a id="ref-4"></a>

### [4] SafePred: Predictive Guardrail for Computer-Using Agents via World Models

2026-02-02 arXiv；动作后果、风险与重规划。

[原文 / 官方资料](https://arxiv.org/abs/2602.01725)

<a id="ref-5"></a>

### [5] From Uncertainty to Action: Learning to Steer LLM Agents

2026-10-06 arXiv；反事实收益学习、VoS 与 harm budget。

[原文 / 官方资料](https://arxiv.org/abs/2610.09115)

<a id="ref-6"></a>

### [6] WebWorld: The Browser as a World Model for Self-Improving Web Code

2026-08-31 arXiv；浏览器执行验证 progress 和 preservation。

[原文 / 官方资料](https://arxiv.org/abs/2608.30530)

<a id="ref-7"></a>

### [7] CHILL-Harness: Counterfactual Harness Learning for Long-Horizon Agents

2026-07 arXiv；反事实 harness 干预收益。

[原文 / 官方资料](https://arxiv.org/abs/2607.25825)

<a id="ref-8"></a>

### [8] ProRe: Proactive Reward via Reasoner-Actor Collaboration

ICLR 2026 正式版；主动探查与 GUI 评估。

[原文 / 官方资料](https://openreview.net/pdf?id=xtysskccFc) · [补充链接](https://arxiv.org/abs/2509.21823)

<a id="ref-9"></a>

### [9] Rollback the World, Keep the Reflection: RIR

2026-09 arXiv；回滚深度、保留有效前缀与反思。

[原文 / 官方资料](https://arxiv.org/abs/2609.18304)

<a id="ref-10"></a>

### [10] WebWorld: A Large-Scale World Model for Web Agent Training

2026-02-16 arXiv；开放 Web transition world model。

[原文 / 官方资料](https://arxiv.org/abs/2602.14721)

<a id="ref-11"></a>

### [11] Latent World Action Model for GUI Agents

2026-10-09 arXiv；预测动作相关潜在信息，避免完整屏幕重建。

[原文 / 官方资料](https://arxiv.org/abs/2610.14228)

<a id="ref-12"></a>

### [12] Diagnosis Before Recovery: Selective Self-Correction / DARC

2026-08-12 arXiv；诊断、干预剪枝与成本策略。

[原文 / 官方资料](https://arxiv.org/abs/2608.11772)

<a id="ref-13"></a>

### [13] Sentry: Learning to Recover from LLM Agent Failures at Test Time

2026-10 arXiv；故障条件经验与运行时恢复。

[原文 / 官方资料](https://arxiv.org/abs/2610.02994)

<a id="ref-14"></a>

### [14] Planning to Avoid Side Effects

AAAI 2022 正式论文；其他目标与策略的保护。

[原文 / 官方资料](https://ojs.aaai.org/index.php/AAAI/article/view/21219)

<a id="ref-15"></a>

### [15] Avoiding Side Effects in Complex Environments

NeurIPS 2020 正式论文；AUP 与目标能力保持。

[原文 / 官方资料](https://papers.nips.cc/paper_files/paper/2020/hash/f50a6c02a3fc5a3a5d4d9391f05f3efc-Abstract.html)

<a id="ref-16"></a>

### [16] Embodied Agent Interface

NeurIPS 2024 D&B；目标解释、规划与 transition modeling。

[原文 / 官方资料](https://arxiv.org/abs/2410.07166) · [补充链接](https://github.com/embodied-agent-interface/embodied-agent-interface)

<a id="ref-17"></a>

### [17] ENACT: Embodied Cognition with World Modeling of Egocentric Interaction

ICLR 2026 正式论文；部分观测交互的世界建模。

[原文 / 官方资料](https://www.jiajunwu.com/papers/enact_iclr.pdf) · [补充链接](https://arxiv.org/abs/2511.20937)

<a id="ref-18"></a>

### [18] AHA: A Vision-Language Model for Detecting and Reasoning over Failures

ICLR 2025；失败合成、检测与解释。

[原文 / 官方资料](https://proceedings.iclr.cc/paper_files/paper/2025/hash/70a06501001e1820fd1eb9ee821302d2-Abstract-Conference.html) · [补充链接](https://github.com/NVlabs/AHA)

<a id="ref-19"></a>

### [19] FAER

Findings ACL 2026；长程具身失败与纠正。此状态不同于 ACL 主会。

[原文 / 官方资料](https://aclanthology.org/2026.findings-acl.948/)

<a id="ref-20"></a>

### [20] MAGMA-GEN

2026-09 arXiv；同状态配对重执行候选恢复并验证进展。

[原文 / 官方资料](https://arxiv.org/abs/2609.20056)

<a id="ref-21"></a>

### [21] Plan Repair / Plan Stability

ICAPS 2005/2006；修复与重规划、控制改动范围的经典前作。

[原文 / 官方资料](https://cdn.aaai.org/ICAPS/2005/ICAPS05-017.pdf) · [补充链接](https://strathprints.strath.ac.uk/2776/1/strathprints002776.pdf)

<a id="ref-22"></a>

### [22] Plan commitment: Replanning versus plan repair

Engineering Applications of AI 2023；C-TFLAP 与连锁影响。

[原文 / 官方资料](https://doi.org/10.1016/j.engappai.2023.106275)

<a id="ref-23"></a>

### [23] RIPPLE: From Seed to Scope: Reasoning to Identify Change Impact Sets

ICSE 2026；变更意图与种子编辑到影响方法集合。

[原文 / 官方资料](https://conf.researchr.org/details/icse-2026/icse-2026-research-track/223/From-Seed-to-Scope-Reasoning-to-Identify-Change-Impact-Sets)

<a id="ref-24"></a>

### [24] Characterizing Regression Bug-Inducing Changes and Improving LLM-Based Regression Bug Detection

ICSE 2026；LLM4Reg，软件回归检测的直接竞争方向。

[原文 / 官方资料](https://chenbihuan.github.io/paper/icse26-song-regression-bug.pdf) · [补充链接](https://github.com/FudanSELab/LLM4Reg)

<a id="ref-25"></a>

### [25] Why we no longer evaluate SWE-bench Verified

OpenAI 2026-02-23 官方分析；旧基准测试与污染的证据边界。

[原文 / 官方资料](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)

<a id="ref-26"></a>

### [26] OSWorld

官方仓库；真实桌面环境、VM/KVM 与配置要求。

[原文 / 官方资料](https://github.com/xlang-ai/OSWorld)

<a id="ref-27"></a>

### [27] WebArena / WebArena-Verified

官方环境资料；自托管网站、容器与环境资源。

[原文 / 官方资料](https://github.com/web-arena-x/webarena) · [补充链接](https://servicenow.github.io/webarena-verified/getting_started/environments/)

<a id="ref-28"></a>

### [28] VirtualHome

官方仓库；Evolving Graph 与 Unity 两条执行路径。

[原文 / 官方资料](https://github.com/xavierpuigf/virtualhome)

<a id="ref-29"></a>

### [29] BEHAVIOR / OmniGibson

官方安装与快照文档；最低资源及恢复语义。

[原文 / 官方资料](https://behavior.stanford.edu/getting_started/installation.html) · [补充链接](https://behavior.stanford.edu/tutorials/save_load.html)

<a id="ref-30"></a>

### [30] ManiSkill3

官方性能文档；4090 示例不等于 B2 预期吞吐。

[原文 / 官方资料](https://maniskill.readthedocs.io/en/latest/user_guide/additional_resources/performance_benchmarking.html) · [补充链接](https://github.com/mani-skill/ManiSkill)

<a id="ref-31"></a>

### [31] Qwen3.5-4B

官方模型卡；可用的当代 4B 级骨干，不代表本服务器已安装。

[原文 / 官方资料](https://huggingface.co/Qwen/Qwen3.5-4B)

<a id="ref-32"></a>

### [32] Qwen3-VL

官方模型与微调仓库；小型视觉骨干、LoRA 与分辨率控制。

[原文 / 官方资料](https://github.com/QwenLM/Qwen3-VL) · [补充链接](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune)

<a id="ref-33"></a>

### [33] Hugging Face TRL: Reducing Memory Usage

官方训练文档；PEFT、梯度检查点及长度/打包优化。

[原文 / 官方资料](https://huggingface.co/docs/trl/main/en/reducing_memory_usage)

<a id="ref-34"></a>

### [34] 第七版 CCF 推荐国际学术会议和期刊目录（正式版）

CCF 原始文档，丽水学院科研处镜像；PDF 第 57-58、27 页。

[原文 / 官方资料](https://kyc.lsu.edu.cn/_upload/article/files/20/77/2cbaa3754eb9aff9ed74cafed8ff/23a8de02-594c-445f-b084-69b0193c05b3.pdf) · [补充链接](https://kyc.lsu.edu.cn/2026/0518/c1772a367789/page.htm)

<a id="ref-35"></a>

### [35] Qwen-AgentWorld: Language World Models for General Agents

2026-06-23 arXiv；千万级跨域交互的 world model。

[原文 / 官方资料](https://arxiv.org/abs/2606.24597)

<a id="ref-36"></a>

### [36] τ²-bench

官方项目；故障排除、工具使用与双控制环境。

[原文 / 官方资料](https://github.com/sierra-research/tau2-bench)

<a id="ref-37"></a>

### [37] 第七版 CCF 目录正式发布公告

CCF 官方公告，2026-03-31；主站直接读取遇到网站验证。

[原文 / 官方资料](https://www.ccf.org.cn/Academic_Evaluation/By_category/)

<a id="ref-38"></a>

### [38] SABER: Small Actions, Big Errors

2025 arXiv；mutation-gated verification、定向反思。

[原文 / 官方资料](https://arxiv.org/abs/2512.07850)

<a id="ref-39"></a>

### [39] Sherlock

2025 arXiv；DAG 脆弱点、选择性验证与回滚。

[原文 / 官方资料](https://arxiv.org/abs/2511.00330)

<a id="ref-40"></a>

### [40] Strategic Verification for Long-Running LLM Agents / BAVAR

2026 preprints.org；表中数字明确为示意、非实证。不可用作复现结果。

[原文 / 官方资料](https://www.preprints.org/manuscript/202608.2057)

### 附加证据边界

BAVAR [40](#ref-40) 的表格是 illustrative，不能把表中数字当已完成实验。Sherlock [39](#ref-39) 与 SABER [38](#ref-38) 可为更广的强对照库。候选基线应按问题重叠与可复现性筛选，不需要把每个近邻都机械实现。

关于旧 B2 的记忆仅用于辨认方向调整；本报告没有把旧工程测试、环境整理或尚未完成的实机轨迹升级为本研究的有效性证据。
