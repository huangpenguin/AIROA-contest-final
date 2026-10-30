# 2026-10-30 家目录整理与 PARC 收尾交接

## 当前事实

PARC 比赛已经结束。用户明确授权删除尝试模型、训练/评测数据和大体积生成物，并要求保留 SHA、配置及实验记录以便复现；之后又要求先归档全部实验分支，再删除分支/实验工作区，仅保留主代码。本轮只向既有 origin 推送 main，不发布实验分支，不重启训练或评测。

模型/数据清理已完成：删除 156,615 个文件（逻辑 1332.14GiB），/home 可用空间净增 1142.35GiB，最终约 2.0T 可用。217 个顶层和嵌套 Git checkout 的 HEAD、refs、登记和 staged/unstaged binary diff 与清理前基线一致；其中 3 个原有 unborn 仓库的 HEAD 明确为 null。21 个独立大权重 inode 实际计算了 SHA256，覆盖 37 个删除路径；原始数据、旧 Orbax 大块等未全部逐文件重新 hash，记录明确标注。

源码、未提交/未跟踪修改、Git 对象和实验配置/报告/日志原先均保留。少量精确归一化参数、纯代码源包和 6 个共 18,032B 的冻结测试证据作为复现材料保留。已安装环境及共享 uv/HF/conda 缓存未清理，/mnt/data 和非 PARC 项目未动。不得按旧的约 1.2TiB 完整数据迁移方案重复复制或尝试恢复已删除的模型。

## 接续入口

主计划：`/home/huang/.cursor/plans/home_code_cleanup_d89db511.plan.md`。

完整收尾档案：`/mnt/home/huang/cst_ai/archive/PARC2026_closeout_20261030/`：

- `README.md`、`REPORT.md`、`INDEX.json`：复现入口、删除范围及记录索引。
- `git/INDEX.json`：217 个 checkout 的版本、refs、状态和补丁；清理后校验在 `git/verification-after.json`。
- `records/`：关键实验配置/命令/seed/指标/manifest、508 份数据模型来源定义、压缩包内独有源码和小型统计参数。原位保留的普通记录不等于全部已经复制到此目录。
- `environments/index.json`：五个当前 Python 环境的 freeze、Python 和 uv 版本；这些是当前环境快照，不是已验证的历史重建结果。
- `actual-payload-sha256.jsonl`、`deletions.jsonl`、`cleanup-result.json`：实际权重校验值和逐文件删除记录。
- `branch-archive/`：实验分支及工作区的完整恢复材料、归档验证和清理报告。使用前先读取此目录的恢复说明，不要仅凭 SHA 假定未推送提交可从 GitHub 获取。

Git SHA 只标识代码，不能从哈希反推出已删除权重。复现需要按版本/补丁恢复对应代码和配置，使用环境快照与来源 revision 重新获取模型/数据，再按保留的命令重跑；不保证随机训练逐字节生成相同权重。

## 重启 Codex 后的后续清理

1. 退出持有 `/home/huang/code/PARC2026_final` 或 `BasicSR.bak` cwd 的旧 Codex/终端/服务；从仓库外启动新会话。仅把某条工具命令的 cwd 设置为 `/` 不会改变已运行服务的 cwd。
2. 读取本交接、主计划最新章节和 branch-archive 的实际结果。不要把旧训练计划的 next_plan 当成新的执行授权。
3. 确认 `/mnt/home` 仍实际挂载为可写 NFS，目标是 `/mnt/home/huang/code`。逐项复查 cwd/fd/maps/容器 bind/mount，不能因 PID 过时或终端断开推定已解除占用，不结束用户未授权结束的进程。
4. PARC 主代码目前仍在本地原路径；若进行最后的代码迁移，保留 main 当前工作区和所有嵌套源码修改、Git 对象依赖与环境规格。根据实际剩余 worktree/分支和归档核验结果操作，不盲跑旧 `followup-parc-copy.py`，不在待搬目录内执行迁移。
5. BasicSR.bak 已在 `/mnt/home/huang/cst_ai/archive/code-cleanup-20261030/BasicSR.bak` 归档（排除可重建旧 .venv、旧 .cache/uv），内容和 Git 元数据已核验；解除其 tmux/zsh 占用后，重新同步验证，再删除本地旧副本。正式 NFS BasicSR 和本地 basicsr 环境不动。
6. 其余 9 个独立仓库及 pic_gen/data 已在 NFS，旧路径为 symlink。它们各自原有未提交内容仍保留，不属于本次 PARC main 的提交或推送范围。未提交源码和未推送提交没有可验证恢复副本时，不得因“已有 SHA”直接删除。

本文件随 main 保存；较大的本地复现档案不加入 Git 仓库。
