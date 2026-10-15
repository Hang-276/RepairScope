# 本地验证记录

验证日期：2026-10-15。代码为首版研究实现，结果边界如下。

| 检查 | 实际结果 |
|---|---|
| 完整测试 | **69 passed，0 skipped** |
| 固定官方 ReAct 源码联调 | 读取 submodule 中的真实官方 class；周边世界和 HTTP 使用 mock |
| 分支采集 | 合成环境中的独立子进程前缀重放、失败部分写入/变量保留、污染与指纹不匹配拒绝 |
| CPU 数值链路 | effect/scalar 均真实反向传播、保存、加载、预测和留出诊断 |
| HF/PEFT 接口 | 本地随机初始化微型 Qwen3 的 LoRA 反传和 adapter/head 保存重载通过 |
| Python 包 | sdist/wheel 构建成功；全新无 ML 依赖的虚拟环境从 wheel 安装并运行 CLI、导入核心模块 |
| 包内容 | wheel 中全部 Python 源文件与工作树逐字节一致 |
| Shell 与源码 | 三个 Shell 脚本语法检查、Python compileall 通过 |
| 官方源码 | Git submodule 固定 `42b5bcf3cd334fee33f0c37c02070a9f5807add5`；本地对象完整性检查通过 |
| 真实 AppWorld 任务 | **0；本机未下载任务数据，也未安装解包后的完整环境** |
| 预训练 4B/NF4/CUDA | **未验证** |
| RTX 4090 峰值显存/速度 | **未验证** |
| 原生 TGC/SGC 主结果 | **尚无结果** |

本机为 macOS arm64、Python3.12.14，数值检查使用 Torch2.5.1、Transformers4.57.6、PEFT0.18.1。真实服务器环境须另行锁定；此列表不是 Linux CUDA 依赖锁。

合成 smoke 收据显式包含 `evidence_origin=synthetic_fixture`、`fixture_backend=true`、`research_validation=false`、`real_appworld_tasks_run=0`、`real_models_trained=0`。其数值不用于科学效果比较。随机微型 Qwen3 的测试也只验证 API 契约，不能替代预训练 Qwen3-4B 或量化训练实测。

`doctor.live_ready=false` 是本机的预期结果。服务器必须完成真实 AppWorld smoke、训练/重载、数据语义审计和 GPU 共存测量，才能宣称实验就绪。若预检发现任务不具备足够恢复/损害差异，应保留负面结果，不能把软件检查改写成研究验证。
