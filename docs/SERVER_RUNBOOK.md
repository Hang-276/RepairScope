# 服务器 P0/P1 操作说明

这份说明供已经部署 AppWorld 与 Qwen3-8B-AWQ 的服务器使用。本次仓库重建没有连接该服务器，也没有核验它的当前运行状态。不要根据新仓库的默认配置重新安装环境或重启服务。

实验仅在服务器执行。所有运行及实验过程状态日志使用清晰英文，控制台和运行目录的 UTF-8 `run.log` 同步即时输出；JSON/JSONL 证据单独保存，不能把完整任务或模型内容写入状态日志。格式见 [English run logs](LOGGING.md)。

## 1. 保存服务器现状

在原服务器工作目录先检查 Git HEAD、分支、未提交改动、模型锁文件、环境 manifest 与 C1–C7 receipts。特别保留原报告中尚未 push 的 `rs/protocol-v1` 和 F1 目标冻结工作。用户已废弃旧研究路线，但这不意味着服务器上未提交的服务修订可以直接覆盖。

公共远端为 `https://github.com/Hang-276/RepairScope.git`。建议将新入口克隆到独立目录，读取现有部署资产；不要在含服务器改动的目录中强制 reset。旧代码可以从 Git 历史查阅，新工作树以新方案为准。

安装轻量编排包时使用 Python 3.11 以上版本。AppWorld 执行应使用现有可工作的服务器解释器与环境。不要为运行标准库编排器把旧环境依赖升级一遍。

```sh
git clone https://github.com/Hang-276/RepairScope.git RepairScope-Evo
cd RepairScope-Evo
python -m pip install -e .
repairscope --help
```

## 2. 准备真实资产

资产说明和事件清单由服务器操作者依据实际文件编写，不由仓库自动下载或猜测目录。下面给出当前输入契约。示意中的占位值必须由真实文件、完整 hash 和既有 receipt 替换；它们不是可以直接通过验收的测试数据。

仓库提供 `configs/assets.example.json` 的占位路径模板、`configs/events.example.json` 的空事件外壳，以及使用 JSON 语法的 `configs/evo_v1.yaml`。空事件外壳会被故意阻止运行，不能作为真实数据。当前包尚不驱动原生在线 Actor/capture，必须先从既有服务器产物导出完整 train 轨迹、候选和计分结果；缺失资产需要报告缺失，不能在导出时合成补齐。

P0 至少需要可核验的以下信息：

- 服务器仓库实际完整 HEAD、dirty 状态与服务配置/验收证据；已有有效配置不得被旧 main 的配方覆盖。
- AppWorld 版本与固定源码提交、模型 ID 与完整 revision、实际服务上下文和锁文件。`32000` 是待核验的上下文要求；短 hash 不够。
- 真实 split manifest、train 轨迹/前缀索引与来源身份。dev/test 仅有必要的 manifest 元数据，不能把其任务内容交给学习器。
- 既有 D1 抽取文件、目标映射和文件 hash。使用既有抽取结果冻结，不逐任务人工补目标。
- 至少一个可信既有固定前缀重放 receipt，能够关联任务、前缀、环境、程序和目标映射，并覆盖 Python 上下文及异常前部分写入。
- 冻结任务顺序、状态选择、预算与唯一扩展计划；所有输入应可追踪实际来源，不能使用 fixture 代替。

绝对路径应指向服务器本地已有文件。不要把受保护数据库、任务文本、目标、轨迹、答案、密钥或模型内容提交 GitHub。运行证据保存在被忽略的 `runs/` 或仓库外部私有目录。

### P0 资产说明格式

`ASSETS.json` 的文件路径相对于该说明文件所在目录解析；也可以提供绝对路径。每个 `sha256` 是对应文件原始字节的 SHA256，而不是 JSON 对象 hash。

```json
{
  "repository": "/absolute/path/to/existing/server/repository",
  "model_lock": {"path": "model.lock.json", "sha256": "<file_sha256>"},
  "environment_manifest": {"path": "environment.json", "sha256": "<file_sha256>"},
  "train_manifest": {"path": "train_manifest.json", "sha256": "<file_sha256>"},
  "goals": {"path": "D1_goals.json", "sha256": "<file_sha256>"},
  "replay_receipt": {"path": "prefix_replay.json", "sha256": "<file_sha256>"},
  "service_receipt": {"path": "service_receipt.json", "sha256": "<file_sha256>"}
}
```

所引用文件均为 JSON，至少满足以下字段：

| 文件 | 当前校验字段 |
|---|---|
| `model_lock` | `name = "Qwen/Qwen3-8B-AWQ"`，`revision` 为真实 40 位十六进制完整提交 |
| `environment_manifest` | `appworld_version = "0.2.0"`，`appworld_commit` 为固定完整 SHA |
| `train_manifest` | 非空 `tasks` 数组；每项含 `task_id`、`scenario_id`、`split = "train"`、来源 `source`、完整 `task_sha256` 和 `trajectory_sha256` |
| `goals` | `source = "existing_D1"`、`extractor_frozen = true`、`tasks` 为以 task ID 为键的对象，且与 train allowlist 完全一致 |
| 单任务目标记录 | `split = "train"`、原始 `instruction` 和 `goals` 数组；每项目标含唯一 `id`、来自原指令原文的 `evidence_span`、`kind`；至少两个 `kind` 为 `state_write` 或 `preservation_constraint` |
| `service_receipt` | `accepted = true`、`context_length = 32000`、`c2_truncation_verified = true`、`c5_verified = true`、`model_revision` 与模型完整 revision 一致 |
| `replay_receipt` | `ok = true`、`evidence_origin = "AppWorld"`、`prefix_verified = true`、`split = "train"`、自然事件 `source`；`task_id` 属于 allowlist；`identity.task_sha256` 与 train manifest 一致，`identity.goals_sha256` 为本任务冻结目标记录的对象 hash，`identity.environment_sha256` 为环境 JSON 对象 hash |

布尔字段只能是 JSON 布尔值；不能因为程序要求 `true` 就填写未发生的验收。既有文件字段不一致时，可以通过通用转换程序生成带来源信息的输入文件，但不能虚构验收、重新手写任务知识或改写历史结果。P0 只核验声明与输入身份，不替代对原始 receipts 的真实性审查。

## 3. 运行并检查 P0

```sh
repairscope p0 --assets /path/to/ASSETS.json --output-dir runs/evo_v1
repairscope inspect --p0-dir runs/evo_v1
```

检查 `P0_report.md` 与冻结文件，确认版本、hash、train 边界、完整模型 revision、目标和重放证据。产物包括 `assets.json`、`protocol.yaml`、`protocol_migration.md`、`data_manifest.json`、`goals.lock.json`；`protocol.yaml` 是 JSON 子集 YAML。本仓库完整迁移说明保存在 `docs/MIGRATION.md`。

P0 因资产缺失或不一致而 `blocked` 是有效结果。补齐真实资产后重新冻结，保留旧输出目录，不用修改报告状态绕过检查。真实 P0 不接受合成资产作为科学证据。P0 通过也不表示已重新验证模型服务或方法效果。

## 4. 准备 P1 事件与隔离执行

事件清单只能引用冻结 train 清单中的原生任务。冻结 scenario 轮转次序、3–5 个任务及计划 10–15 个初始状态，每阶段最多 15 个状态，记录自然事件来源。每个状态接收 1–4 个实际已有初始候选，不能只导出成功候选，也不能为凑四个而补造缺失候选。每任务最多两次实际知识学习事件，额外状态仍核验并存档全部候选见证，不再启动联合学习。

原任务必须先完成计分并保存原始结果；任务内知识固定。影子环境使用隔离副本，实际修复从当前状态出发。重放不能丢失 Python 变量或异常前的部分写入，影子分支不能写回原任务数据库。

对每个目标/checker 映射做通用审计，确认合法条件干预前后仍判断同一语义。审计可以标记无效和 unknown，不能手写正确规则、参考修复或逐任务干预帮助方法过关。记录 `human_task_edits`、`human_rule_edits`、`human_intervention_edits` 与审计时间；主机制证据前三项必须为零。

留出行为状态单独声明，在对应修订提交前不能用于规则生成、实验选择或旧证据门控。条件或程序提议由冻结服务自动产生；具体事件的标准答案不得由操作者填写。

### P1 事件格式

`EVENTS.json` 顶层只接受 `schema_version`、`events`、`transfer_cases`、`allow_extension`。`allow_extension` 必须在首次执行前声明；`events` 中 initial 在前，extension 在后，任务顺序服从 P0 冻结的 scenario 轮转次序。

| 事件字段 | 内容 |
|---|---|
| `event_id`、`phase` | 唯一事件 ID；`phase` 为 `initial` 或预设的 `extension` |
| `request` | 下表所述真实 train 分支请求 |
| `public_context` | 只接受 `instruction`、`history`、`api_docs`、`observations`；原指令须与 D1 完全一致，observations 为可见键值对象 |
| `candidates` | 既有初始候选数组，每项 `id` 与 `code`；最多四个，保留失败候选，不补采直到成功 |
| `original_score` | `scored = true`、一致的 `task_id`、`split = "train"`、原任务 `result` 和其 JSON 对象 hash `result_sha256` |
| `forbidden_literals` | 仅供审计的 task ID、实体、答案字面值排除表，至少包含本事件 task ID；不得注入可部署规则 |
| 人工记录 | 三个 `human_*_edits` 必须显式为整数 0，可附 `audit_minutes` |

分支 `request` 字段：

| 字段 | 内容 |
|---|---|
| `task_id`、`scenario_id`、`split`、`source` | 冻结清单中的身份，split 为 train，起点为自然事件 |
| `root` | 既有 AppWorld 数据根目录，须含 `data/datasets/train.txt` |
| `steps` | 精确前缀数组，每项含 `code` 和实际 `output`；失败输出及部分写入不得删去 |
| `program` | 合法 Python 修复代码字段，执行时由固定候选或模型提议替换 |
| `g_fix`、`g_keep` | 预先声明的非空、互不重叠目标 ID 数组，不能在结果后修改 |
| `goal_mapping` | 覆盖全部冻结目标 ID；每项目标含 `complete` 布尔值和唯一的 `requirements` 字符串数组 |
| `identity` | 五个完整 SHA256，计算规则见下文 |
| `checker_spec`、`checker_audit_sha256` | 经通用审计的 checker，格式 `module:function`，及真实审计文件 hash；条件干预世界缺少有效 checker 时结果为 unknown |
| `intervention` | 原自然事件和留出请求必须为 `null` 或省略；由系统在影子探测时提出 |

真实 P1 不允许 `factory_spec`。该接口只供显式合成夹具使用。`checker_spec` 所指函数应可由现有解释器导入，签名为 `(world, mapping)`，返回恰好包含全部 goal ID、值为 `true/false/null` 的对象。名称或 hash 不会自动证明 checker 语义正确，审计须实际完成。

对象 hash 使用 `repairscope.protocol.digest`：键排序、UTF-8、紧凑 JSON、保留 Unicode、禁止 NaN。`identity` 五项须精确对应：`protocol_sha256 = digest(P0 protocol)`、`goals_sha256 = digest(本任务冻结目标记录)`、`environment_sha256 = P0 assets.environment_sha256`、`task_sha256 = train_manifest 中本任务 hash`、`mapping_sha256 = digest(goal_mapping)`。原始文件 hash 使用 `repairscope.protocol.file_digest`，二者不能混用。

`transfer_cases` 每项含唯一 `case_id`、`request`、`public_context`、`repeats`（1–3）及三个为 0 的人工编辑字段。当前实现采用较严格的 task 级留出：transfer task 不得出现在任何学习事件中；最多三个案例。生成和更新结束后才执行这些状态，不把结果反馈更新器。

## 5. 运行 P1

确认 P0 通过并且现有 HTTP 服务身份与锁文件一致，再使用同一个服务顺序执行提议、归纳和修订：

```sh
repairscope p1 --p0-dir runs/evo_v1 --events /path/to/EVENTS.json \
  --output-dir runs/p1 --base-url http://127.0.0.1:8000/v1
```

服务地址示例仅指向服务器本地现有端点。代码不启动或重新配置服务。实际模型完整 revision、上下文与 C2/C5 行为必须由服务器的锁和 receipts 对应确认，不能仅凭 `/models` 返回的名称认定一致。

每事件最多 12 个新分支、六次学习模型调用、8000 输出 token；首批总分支最多 180。实际初始候选、四格探测、拟议规则重新生成行为验证以及旧证据回归均计入事件预算。旧证据回归同时执行旧固定程序和新知识在旧上下文中重新生成的程序，全部成本累计。预算不足记录 unresolved，不自动增加候选或分支。

当前仅记录完整分支身份和 `cache_key`，没有重放缓存，缓存命中数为 0；每次实际分支执行都计费。不要将重复运行当作免费缓存。

最多三个留出案例、每案例三次新旧知识配对，最多 18 次行为评估，另计成本。只允许使用一次预设覆盖扩展：再增加最多 15 个状态、180 个分支；不能重新开目录、换种子或循环同一清单绕过阶段总账。

留出改善还须核验：真实 AppWorld 来源、新旧生成程序有变化、变更规则实际注入新提示、知识提示 hash 确实变化。全部配对结果存档，不能只报告有利的一次输出。

## 6. 审查产物并停止

审查以下产物是否覆盖所有输入、失败、unknown、拒绝与成本：

```text
P1_report.md
witnesses.jsonl
joint_probes.jsonl
knowledge_versions.jsonl
transfer_cases.jsonl
coverage_and_cost.json
```

按 [协议](PROTOCOL.md) 判断 A/B/C：至少三个真实成功修复跨两个 scenario、两类条件机制、一次有旧证据检查及独立迁移行为证据的自动实质修订。条件不满足则给出停止原因；满足后提交继续建议，由后续研究决定是否进入 P2。本入口不自动启动 P2、训练、全量 test、第二环境或任务生成。

本地 `fixture-smoke` 只用于接口排错。不要把合成报告、可信 receipt 引用、程序合法或四格文件齐全写成真实 P1 已通过。结果报告必须区分实测、已有报告引用和未核验信息。
