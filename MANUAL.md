# PARC2026 本选手动操作清单

本文是给操作者的中文 checklist。自动 runner 只负责配置校验、调用官方脚本、记录 provenance 和打印命令；涉及 GPU、账号许可、候选激活和真实 Omni 评分的动作必须人工确认。

## A. 开始前的人工准备

- [ ] 已接受 GitHub 邀请并能访问 `matsuolab/PARC2026_final`。
- [ ] 在 Omnicampus GPU 环境确认有 NVIDIA GPU、Docker/nvidia-container-toolkit，且 `nvidia-smi` 正常。
- [ ] 仅在 Hugging Face 页面完成 `google/paligemma-3b-pt-224` 的许可后，在当前 shell 设置 `HF_TOKEN`。
- [ ] 如果要用 W&B，设置 `WANDB_API_KEY`；不要写入 YAML、代码、Git 或日志。
- [ ] 确认比赛 Slack/Omnicampus 的最新通知。附件 PDF/TXT 是 2026-08-24 Version 1.0，后续通知优先。

## B. 评测环境（单独终端）

```bash
cd /workspace/PARC2026_final
bash setup.sh                 # 首次执行；会安装 CUDA 13 评测依赖并取得 LIBERO assets
source env.sh                 # 每个评测终端都执行
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

官方目标环境是 Python 3.10、CUDA 13.0、PyTorch 2.11.0+cu130、单张 NVIDIA L4 24GB、`MUJOCO_GL=EGL`。训练 venv 不要拿来跑 evaluator。

## C. 数据和 EDA

1. 按 Omnicampus 的数据说明，从只读 `~/dataset/` 选择需要的 tar。
2. 只把要用的 tar 解压到 `~/data/`，不要改写 `~/dataset/`。
3. `~/data` 需要持久化时手动执行 `parc-home-sync data-push`；新 session 恢复执行 `parc-home-sync data-pull`。
4. 打开 `notebooks/00_eda_and_contract_check.ipynb`，在仓库根目录运行。
5. 在正式训练前记录 dataset revision、源 tar SHA-256、manifest hash、task/episode/frame 计数和 train/validation episode 是否重叠。长度异常和 action spike 只做人工复核。

## D. 训练环境与启动

训练必须使用单独的 venv：

```bash
cd /workspace/PARC2026_final/examples/pi05_libero_finetune
bash scripts/setup_train.sh
source env_train.sh
```

首次 setup 可能需要安装 ffmpeg、clone lerobot v0.4.4、应用 patch 和安装依赖。确认输出中的 lerobot、torch、CUDA 和 patch 检查均通过。

### 1. 只读 dry-run（推荐先做）

```bash
cd /workspace/PARC2026_final
python tools/run_experiment.py --action dry-run \
  --config configs/pi05_lora.yaml \
  --set run.tag=pi05_smoke \
  --set train.steps=30 \
  --set wandb.mode=disabled
```

可覆盖的 CLI 字段只有：`run.tag`、`run.output_dir`、`run.seed`、`train.steps`、`wandb.mode`、`hardware.gpu_ids`。其它参数必须改 YAML；未知字段会直接失败。dry-run 不建目录、不训练、不启动 server、不上传。

### 2. smoke

若工作树有本次开发改动，可在 YAML 中明确设置 `run.smoke_allow_dirty: true`，然后：

```bash
python tools/run_experiment.py --action smoke \
  --config configs/pi05_lora.yaml \
  --set run.tag=pi05_smoke \
  --set train.steps=30 \
  --set wandb.mode=disabled
```

smoke 仍要求 dataset root 存在，且 smoke 输出目录不存在。检查 loss/gradient finite、checkpoint 生成、LoRA merge、tokenizer 同包和峰值显存。

### 3. 正式训练

确认 git clean、数据存在、`HF_TOKEN` 有效、输出目录不存在后：

```bash
python tools/run_experiment.py --action train --config configs/pi05_lora.yaml
```

它调用官方 `examples/pi05_libero_finetune/scripts/full_pi05_lora.sh`，默认 LoRA rank 16、batch 16、gradient accumulation 8、20,000 steps、action chunk 10。训练日志和 checkpoint 放在 YAML 的 output 目录；`resolved_config.yaml` 与 `run_manifest.json` 会在 run 记录和成功输出中保存。

若要手动断点续训，先检查 checkpoint 和 `train_config.json`，再按官方脚本的 resume 语义执行；不要让新实验覆盖已有 output。

## E. 合并和候选激活（必须人工决定）

训练输出不是提交包。先合并到独立 candidate：

```bash
python examples/pi05_libero_finetune/scripts/merge_lora.py \
  --adapter <run-output>/checkpoints/last/pretrained_model \
  --out <candidate>/model_weights
cp examples/pi05_libero_finetune/submission/policy_server.py <candidate>/policy_server.py
cp examples/pi05_libero_finetune/submission/requirements.txt <candidate>/requirements.txt
```

人工确认以下项目后，才把 candidate 整理成 `submission_candidate`：

- `model_weights/`、tokenizer、processor/statistics、config 文件齐全；
- policy server 只改 `MyPolicy` 的模型加载、reset、推理部分；
- action 是有限的 float32 `(7,)`，顺序为 `[dx,dy,dz,droll,dpitch,dyaw,gripper]`；
- 不依赖评测时的外网下载；
- candidate 的代码、模型和依赖与实际训练记录一致。

## F. 本地 server 和 evaluator

终端 A 启动 policy server：

```bash
cd <candidate>
python policy_server.py --port 8000
```

终端 B：

```bash
cd /workspace/PARC2026_final
source env.sh
python validate_submission.py <candidate> --static
python validate_submission.py <candidate> --pip-dry-run
python -m pipeline --server-url http://localhost:8000 \
  --track track1 --n-episodes 2 --max-steps 300 --timeout 10
```

或者用 runner：

```bash
python tools/run_experiment.py --action eval \
  --config configs/pi05_lora.yaml --server-url http://localhost:8000
```

先跑 1–2 episode，再扩展到固定协议的正式本地评测。遇到 `/act`/`/reset` 超过 10 秒、action shape/dtype/NaN、server crash 或 collision，停止后续提交流程。公开 example 的本地结果不是 Omni 正式分数。

## G. 打包与 Docker 最终确认

```bash
python tools/run_experiment.py --action pack \
  --config configs/pi05_lora.yaml --submission-dir <candidate>
python tools/run_experiment.py --action validate \
  --target artifacts/parc2026_submission.zip
```

提交 zip 上限 20GB；禁止嵌套 zip、zip bomb、符号/硬链接、路径穿越和不必要的混淆。最终必须在官方 Docker 内执行：

```bash
python evaluate.py /data/parc2026_submission.zip --n-episodes 2
```

Docker E2E 会解压、安装提交依赖、启动 server 并跑官方 pipeline；这一步最接近真实评分环境。

## H. Omnicampus 提交方式

本选 1 Version 1.0 的 assignment ID：

| Track | ID |
|---|---:|
| Track 1 | 1007 |
| Track 2 | 1008 |
| Track 3 | 1009 |

### 方法 1：Omnicampus GPU 环境中的 CLI

`--path` 传未压缩的提交目录，CLI 会负责压缩：

```bash
omni submit --assignment-id 1007 \
  --path /absolute/path/to/submission_candidate --dry-run
```

人工核对 dry-run 的 assignment/Track、目录、大小、文件清单和剩余次数后，才执行真实提交：

```bash
omni submit --assignment-id 1007 \
  --path /absolute/path/to/submission_candidate
```

runner 也可打印同一命令：

```bash
python tools/run_experiment.py --action submit-dry-run \
  --config configs/pi05_lora.yaml --assignment-id 1007 \
  --submission-dir /absolute/path/to/submission_candidate
```

### 方法 2：浏览器

进入课程首页 → “宿題” → 选择对应 Track → 选择 `.zip` 文件 → 点击提交。上传完成时算一次提交；每个 Track 每日只能一次，错误也消耗次数。不要在截止时间或午夜附近才上传大 zip。

### 提交后

等待页面从“採点リクエスト中/採点中”变为“採点完了”，记录时间、score 和 log。不要连续点提交或刷新。若上传/评分出错，先保存日志，不要当天再次尝试消耗剩余次数；按规则通过官方问询渠道确认。

## I. 归档和关机

- [ ] 评测完成后可运行 `python tools/run_experiment.py --action archive --config configs/pi05_lora.yaml`，它会复制可用的 config、manifest、zip 和 evaluation JSON，并为每个文件记录 SHA-256。
- [ ] 归档 resolved config、run manifest、git commit/dirty diff hash、dataset/model/mix hash。
- [ ] 归档 server log、validator JSON、local evaluator JSON、zip SHA-256、W&B URL。
- [ ] 不保存 token、完整 observation payload 或不必要的大视频。
- [ ] 训练完成后停止 notebook/kernel；关闭浏览器标签不一定会停止 GPU。
- [ ] `~/data` 的重要数据执行 `parc-home-sync data-push`。
