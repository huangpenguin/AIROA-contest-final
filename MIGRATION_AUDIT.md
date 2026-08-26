# PARC2026 正赛仓库审计

审计日期：2026-08-24  
仓库：`matsuolab/PARC2026_final`  
HEAD：`2059989571d8a7da3526b994756eb77b748d37af`  
分支：`main`  
初始状态：clean，未初始化 submodule，使用 `GIT_LFS_SKIP_SMUDGE=1` 克隆。

## 官方契约表

| 项目 | 正赛仓库证据 | 结论 |
|---|---|---|
| Track | `README.md:112-121`、`compe/t1/T1_TASKS.csv`、`compe/t2/T2_TASKS.csv`、`compe/t3/` | `track1/2/3` 分别由 `libero_t1/2/3` 注册；本地只有 example 任务，不能当作私有 leaderboard 分数。 |
| Policy server | `README.md:135-144`、`submission_template/policy_server.py:25-135` | `GET /health`、`POST /reset`、`POST /act`；action 必须是 float32 `(7,)`，顺序 `[dx,dy,dz,droll,dpitch,dyaw,gripper]`。 |
| Observation | `submission_template/policy_server.py:34-56` | 两路 `(128,128,3) uint8` 图像与 joint/eef/gripper 字段；server 只接收这六类 observation。 |
| Reset/latency | `README.md:284-301`、`PARC2026開発コンペティション_本選.txt:288-301` | `/reset`、`/act` 每个请求必须低于 10 秒；server 启动本地默认 120 秒，正赛 1200 秒。 |
| Runtime | `README.md:146-181`、`Dockerfile:14-19` | CUDA 13.0、PyTorch 2.11.0+cu130、Python 3.10、NVIDIA L4、MUJOCO EGL；LIBERO-plus/LIBERO/asset revision 固定。 |
| Local evaluation | `README.md:90-121`、`evaluate.py:154-230`、`pipeline/cli.py` | 官方 pipeline 负责本地 example rollout；`evaluate.py` 负责 zip 解包、依赖安装、server 启动和 E2E。 |
| Static/dynamic validation | `README.md:123-131`、`validate_submission.py:1-40`、`submission_template/validate_submission.py` | 提交前检查必需文件、zip 安全、server 启动、/health→/reset→/act、action shape/dtype/finite 与 latency。 |
| Training entry | `examples/pi05_libero_finetune/scripts/setup_train.sh:69-154`、`full_pi05_lora.sh:60-90` | 学习使用单独 venv、lerobot v0.4.4、pi0.5 LoRA；官方核心命令保留，runner 只组装与门禁。 |
| Checkpoint merge | `examples/pi05_libero_finetune/scripts/merge_lora.py:1-113` | LoRA 必须 merge；tokenizer 必须离线同包；训练专用 config key 会被剥离。 |
| Submission | `submission_template/PARC2026 演習環境からの提出手順（omni submit）.md`、附件 TXT p.23-25 | 未压缩目录可交给 `omni submit --path`，CLI 会压缩；也可浏览器上传 zip；真实上传前必须 dry-run。 |
| Daily quota | 附件 TXT p.22-25 | 每 Track 每日一次；上传完成即消耗；错误也消耗；三 Track 独立。 |
| Scoring | `pipeline/scorer.py`、`pipeline/total_score.py`、附件 TXT p.12-13 | 本地输出成功率与轨迹原始指标；官方 leaderboard 权重/归一化非公开，不能将本地代理分数称为正式分数。 |

## 仓库已有能力

- 官方 server template、pi0.5 推理示例与 `model_weights/` 约定。
- 官方 Track 注册、rollout、成功/碰撞判定和本地原始指标输出。
- `validate_submission.py` 静态检查与动态 smoke。
- Dockerfile/setup.sh，能够复现评测环境；训练 setup 与 LoRA merge 脚本独立于评测 venv。

## 缺口与本次补齐

| 缺口 | 处理 |
|---|---|
| 没有机器可读 contract | 新增 `contracts/official_policy_contract.yaml`，仅记录官方已确认字段。 |
| 没有完整实验 YAML/resolved manifest | 新增 `configs/pi05_lora.yaml`、`tools/run_experiment.py`。 |
| 没有统一 dry-run/smoke/eval/package/archive 编排 | 新增 `tools/run_experiment.py`，调用官方脚本，不重写 evaluator。 |
| 没有 EDA 入口 | 新增只读 `notebooks/00_eda_and_contract_check.ipynb`；缺少数据时显式失败，不回退旧路径。 |
| 没有中文手动操作手册 | 新增 `MANUAL.md`，记录训练、评测、validator、Omnicampus 与手动确认点。 |
| 通用经验缺少正赛上下文 | 新增/改写 `docs/EXPERIENCES_AND_PITFALLS.md`、`docs/BEGINNER_WORKFLOW.md`。 |

## 不迁移项

- 不迁移旧仓库 evaluator、scoring、manifest、policy vendor、Track 专用脚本。
- 不迁移旧数据 revision、历史分数、旧绝对路径或旧依赖版本。
- 不把预选赛 state/camera 假设写入正赛 contract；只使用正赛 `submission_template/policy_server.py` 的字段。

## 风险与人工确认

1. 附件 PDF/TXT 是 Version 1.0（2026-08-24）；Slack/Omnicampus 的后续通知优先，截止日需人工确认。
2. 当前仓库公开的是 example 任务，公开结果不能预测私有 leaderboard。
3. 训练需要 `HF_TOKEN`，W&B/API key 不能写进配置、代码或日志。
4. 训练机器可用不代表单 L4 可评测；提交前必须在官方 Docker 中验证。
5. `omni submit` 是外部有副作用操作，本次不会自动执行真实提交。
