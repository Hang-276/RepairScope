# 服务器 Agent 执行提示词

你要在一台 RTX 4090 服务器上复现并运行 RepairScope。用户已授权构建环境、clone 新仓库、运行本项目实验；请实际执行到有可核验结果，不要只返回计划。

仓库：https://github.com/Hang-276/RepairScope （私有，使用服务器已有 GitHub 身份；不要输出或提交 token）。新建独立工作目录，例如 `/root/autodl-tmp/RepairScope`，已有同名目录时先核验来源，不能覆盖。原 `/root/autodl-tmp/B2` 及其环境、模型、进程全部保留。

先记录 Git HEAD、GPU 型号/空闲显存/其他进程、磁盘、驱动与 Python。阅读仓库 README、docs/IMPLEMENTATION_NOTES.md、EXPERIMENT_PROTOCOL.md、LABEL_AUDIT.md、SERVER_RUNBOOK.md。保留现有代理和网络配置；只建立本项目独立 `.venv` / `.venv-actor`。不得终止他人任务，不要修改全局路由。

1. `git clone --recurse-submodules https://github.com/Hang-276/RepairScope.git`；检查 submodule 为 `42b5bcf3cd334fee33f0c37c02070a9f5807add5`。运行 `PYTHON_BIN=python3.11 bash scripts/bootstrap_server.sh`（Python 3.12 也可）。保存依赖锁定清单和 doctor 输出。基础环境失败可在独立环境修复并记录，不能把失败写成通过。
2. 执行 `.venv/bin/python scripts/prepare_models.py --download`，保存两个模型的不可变 revision；正式命令使用 `runs/configs/` 下配置。启动 `bash scripts/serve_actor.sh`，只监听本机，记录本项目服务 PID 与日志。确认 non-thinking、模型名、8192 context、3000 actor 输出上限和真实 HTTP usage。先完成真实环境 API/code smoke，不要只看 import。
3. 跑 `bash scripts/run_stage.sh pilot`。检查至少一条真实 trajectory 的执行和声明完成语义、出错后部分写入、实际候选选择与后续消息一致、预算守卫、四候选生成成功率、目标覆盖和峰值显存。三任务只证明接口，不是方法效果。
4. 跑 `capture` 和 `labels` 阶段。所有候选标签只能来自 train/dev 的独立进程完整前缀重放；运行中的代理不可访问 evaluator。检查副本一致性、标签 unknown、候选差异与真实 01/10。按 LABEL_AUDIT.md 对至少30个目标进行有依据的语义审计，记录发现、修正及 `runs/labels/mapping_audit.json`，不要仅因 JSON 合法便宣布标签正确。修改映射后重采受影响标签。执行 `scripts/check_collection.py`。
5. 只有真实数据通过质量检查才花 GPU 训练。若没有足够损害/恢复或有效候选差异，扩大自然 train/dev 采集并诊断；如果任务不能支持问题，明确报告证伪/不足，不得用合成故障补主数据，不得访问 test 来调整方法。候选 noop 导致拒绝完成、one-step utility 不重视信息操作等行为必须在 train/dev 检查；若改代码或协议，保存理由并重跑所有受影响比较。
6. 训练前只停止自己启动的 actor 服务，释放显存。跑 `folds` 的5折场景隔离机制检验，再跑 `train` 的 effect/scalar 三个训练种子100/101/102。两者须同4B模型、同候选输入、同完整标签样本；保存 adapter、head、metadata、split/data hash、loss、训练时间和峰值显存。adapter量化训练必须实跑反向传播及保存/重载预测，不允许以 tiny backend 替代。
7. 重新启动 actor，先验证8B actor与4B NF4 predictor共存的真实峰值。然后跑 `dev` 原生比较：baseline/direct各一组固定actor种子，effect/scalar各3个训练检查点，环境与actor仍固定100。统计完整任务TGC/SGC、干预覆盖率、实际API次数、全部辅助输入/输出token、分类器处理token与时间；不要把静态调用估计当真实API开销。显存不足可在train/dev统一调整actor预留比例或上下文，并对全部方法同改同重跑；不能悄悄换大GPU或更大actor。
8. 冻结最终 Git commit、配置、模型revision、标签哈希和检查点：`.venv/bin/python scripts/freeze_protocol.py create`。只有协议固定且已有真实dev结果时执行 `bash scripts/run_stage.sh test`。测试必须保留完整天然任务与原生分母，只查看aggregate TGC/SGC；不要逐任务测试集调试、搜索答案或回看test误例改方法。SGC是Scenario Goal Completion。

所有长阶段保留日志、退出码与明确完成标记。不要因某一步启动成功便称主实验完成；不覆盖旧结果。若中断，说明哪些任务实际结束、哪些仍在跑、可复现续跑的路径和阻碍。将真实结果汇总为 `runs/REPORT.md`，包括仓库commit、机器/环境、数据版本与规模、标签审计、各比较数字与不确定性、训练/推理耗时和显存、失败/负面结果。保留原始私有日志与权重；不要把受保护的任务内容、派生标签、凭据推上 GitHub。只在证据支持时写结论，不许将本地fixture测试或其他论文的成绩冒充本方法结果。
