# PARC2026 迁移与执行报告

日期：2026-08-24  
官方 HEAD：`2059989571d8a7da3526b994756eb77b748d37af`

## 已完成

- 克隆正赛仓库，确认分支 `main`、remote 和初始 clean 状态；未初始化 submodule/LFS 权重。
- 完成官方 README、Docker/setup、训练示例、Policy server、pipeline、scorer、validator、Omnicampus 文档和附件 PDF/TXT 的审计。
- 新增 `MIGRATION_PLAN.md`、`MIGRATION_AUDIT.md`、`contracts/official_policy_contract.yaml`。
- 新增 `configs/pi05_lora.yaml` 与 `tools/run_experiment.py`：配置校验、白名单 override、dry-run、preflight、训练/smoke 调用、官方本地 eval 调用、静态/动态 validator 调用、打包、archive 和 Omni submit dry-run 命令。
- 新增只读 `notebooks/00_eda_and_contract_check.ipynb`。
- 按正赛事实重写通用经验与初学者流程：`docs/EXPERIENCES_AND_PITFALLS.md`、`docs/BEGINNER_WORKFLOW.md`。
- 新增中文手动操作清单 `MANUAL.md`，包含训练启动、候选合并、双终端评测、Docker E2E、Omnicampus CLI/浏览器提交和每日额度注意事项。
- 修改训练 helper，使 `WANDB_MODE` 显式支持 `online/offline/disabled`，并默认关闭 model artifact；API key 只从环境读取。

## 已执行的验证

| 检查 | 结果 |
|---|---|
| YAML dry-run 与路径展开 | PASS |
| 允许 override（tag/steps/seed/mode/gpu） | PASS |
| 未知 override 拒绝 | PASS |
| 非法 W&B mode 拒绝 | PASS |
| dry-run 无目录/文件写入 | PASS |
| Python compile / notebook JSON / shell syntax / git diff check | PASS |
| 官方 `submission_template` 静态 validator | PASS |
| runner 打包官方模板并再次静态 validator | PASS |
| Omni submit dry-run 命令生成 | PASS；本机无 `omni`，仅打印命令 |
| 仓库 pytest | 未运行：本机没有 `pytest` 命令 |
| 动态 server smoke | 未执行：本机没有 `numpy`/评测依赖 |

## 未完成且必须在 Omnicampus GPU 环境执行

1. 从 `~/dataset` 解包数据到 `~/data`、执行 EDA 并填写实际 dataset manifest/revision/hash。
2. `setup_train.sh` 建立训练 venv、HF gated weight 访问确认和 10–100 step smoke。
3. 正式 LoRA 训练、checkpoint merge、tokenizer 离线同包。
4. 单张 L4 上的 `/health`、`/reset`、`/act` latency smoke 和官方 Track1/2/3 本地 pipeline。
5. 官方 Docker 的 zip E2E 验证、依赖安装时间、显存峰值和结果归档。
6. 运行 `omni submit --dry-run` 后由用户确认课题/Track/次数，再执行真实提交。

当前开发机检测到 Docker 命令但 NVIDIA 驱动不可用，且没有 `omni`/`pytest`/numpy，因此以上步骤不能安全地在本机冒充完成。

## 人工确认项

- 最新 Slack/Omnicampus 通知是否改变 assignment ID、截止时间或提交规则。
- 每个 Track 当天是否尚有一次提交额度。
- 候选模型是否确实包含独立训练参数并符合“参加者实施的学习要素”规则。
- submit dry-run 的路径、文件大小、课题名和 Track 是否正确。
- 本地 example 结果只能作为工程门禁；不把它当作正式 leaderboard 分数。
