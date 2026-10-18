# 本地工程验证记录

日期：2026-10-18。上游 WMA：`dd894640204895b52f050c66cc0b58f08e00f9b8`。工作分支：`codex/repairscope`。

环境：MacBook / macOS 26.6.2 / arm64，Python 3.12.14，PyTorch 2.10.0；仅 CPU FP32。

## 已通过

| 检查 | 结果与证据 |
| --- | --- |
| 工程测试 | **84 passed**；`runs/local_checks/junit.xml` |
| 一次最小模型 smoke | 2 个合成训练任务、18 条 transition、5 次优化更新、checkpoint 重载、H4 硬递推和执行/检查/修复控制器；`runs/final_smoke_20261018/smoke_report.json` |
| 同一 smoke checkpoint 的预测评测入口 | H=1/2/4 共 42 行预测，窗口数分别 10/8/4；不执行环境、不更新参数；`runs/final_smoke_20261018/prediction/summary.json` |
| dev 阈值入口 | 合成 dev 的 H1 有效审计记录选择一次阈值，绑定 checkpoint hash；`runs/final_smoke_20261018/calibration/calibration.json` |
| 主动采样闭环入口 | 同一 smoke checkpoint；合法四格产生 4 条新记录，2 次优化更新；分支与准备成本 18/20，按剩余预算不足停止；查询结果不进入训练；`runs/final_smoke_20261018/probe/summary.json` |
| 主实验与消融脚本 | 单种子 dry-run：主 A 21 条命令（含 H1/2/4 与阈值选择）、主 B 4 条、模型/损失消融 16 条、不探索 1 条；均不启动模型、API 或环境 |
| 打包 | wheel 构建、独立目录安装后导入、fixture 资源完整；没有导入旧 WebArena 运行依赖 |
| 静态检查 | Bash 语法、Python 编译、`git diff --check` 通过 |

新增集成检查沿用同一微型 checkpoint，目的是检查此前未连接的命令接口。没有比较方法提点，没有为算法筛选重新训练，也没有进行反复的小规模有效性试验。脚本复核修复了主 B 训练数据透传和 no-probing 输出目录冲突，并增加相应回归检查。

## 证据边界

这些记录全部为 synthetic fixture 或工程检查；TGC/SGC 保持 null。不存在真实 AppWorld 任务成功率、提点、泛化或 GPU 性能结论。

未在本地安装/执行 AppWorld，未访问 Actor 模型服务，未运行 CUDA/BF16。服务器需提供实际公开观测/目标映射、合法训练分支计划及数据，然后按 [SERVER_RUNBOOK.md](SERVER_RUNBOOK.md) 直接进行主实验。真实环境和 GPU 上的正确性与效果仍需由这些正式运行确认。

`runs/` 和 `.venv/` 只保存在本地，不进入源码交付包。wheel 位于 `dist/repairscope_wma-0.1.0-py3-none-any.whl`；包含 WMA 上游与全部新增源码、配置、测试、脚本和文档的交付包为 `dist/repairscope-wma-source.tar.gz`。
