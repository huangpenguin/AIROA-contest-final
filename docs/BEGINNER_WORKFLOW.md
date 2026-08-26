# PARC2026 正赛执行流程

## 0. 一次性环境准备

评测环境（推荐在官方 GPU Docker 内）：

```bash
cd /workspace/PARC2026_final
bash setup.sh                 # 第一次执行；会下载评测依赖、LIBERO 和 assets
source env.sh                 # 每个评测终端都要执行
```

训练环境（与评测环境分开）：

```bash
cd /workspace/PARC2026_final/examples/pi05_libero_finetune
bash scripts/setup_train.sh
source env_train.sh
```

pi0.5 需要先在 Hugging Face 完成 gated model 的使用许可，并在当前 shell 设置 `HF_TOKEN`。如果启用 W&B，再设置 `WANDB_API_KEY`；不要把它们写入 YAML。

## 1. 配置与只读 dry-run

```bash
cd /workspace/PARC2026_final
python tools/run_experiment.py --action dry-run \
  --config configs/pi05_lora.yaml \
  --set run.tag=pi05_smoke
```

CLI 只允许覆盖：`run.tag`、`run.output_dir`、`run.seed`、`train.steps`、`wandb.mode`、`hardware.gpu_ids`。其它值修改 YAML。dry-run 不建目录、不写文件、不训练、不上传。

先做门禁检查：

```bash
python tools/run_experiment.py --action preflight --config configs/pi05_lora.yaml
```

## 2. EDA 与数据

打开 `notebooks/00_eda_and_contract_check.ipynb` 并在正赛仓库根目录运行。数据缺失时 notebook 应明确失败；不要回退到旧仓库路径。将 Omnicampus 数据解包到 `~/data` 或 YAML 指定的位置，并确认 train/validation episode 不重叠。

## 3. 10–100 step smoke

```bash
python tools/run_experiment.py --action smoke \
  --config configs/pi05_lora.yaml \
  --set run.tag=pi05_smoke \
  --set train.steps=30 \
  --set wandb.mode=disabled
```

smoke 允许由 YAML 显式设置 `run.smoke_allow_dirty: true`，但输出目录仍必须不存在；它会调用官方 `scripts/smoke_pi05.sh`。检查 loss/gradient 是否 finite、checkpoint 是否生成、merge 后 tokenizer 是否同包、GPU 显存和速度是否合理。

## 4. 正式训练

确认 git clean、dataset root 存在、`HF_TOKEN` 有效、输出目录不存在后：

```bash
python tools/run_experiment.py --action train --config configs/pi05_lora.yaml
```

runner 会调用官方 `full_pi05_lora.sh`，并在 run 记录中保存 resolved config、git commit/diff hash、命令、seed、dataset/model revision 字段和环境信息。候选输出不自动进入提交目录。断点续训请按官方脚本手动确认目标 checkpoint 后执行，不要让 runner 覆盖正式 run。

## 5. 合并 LoRA 并准备候选

```bash
python examples/pi05_libero_finetune/scripts/merge_lora.py \
  --adapter <run-output>/checkpoints/last/pretrained_model \
  --out <candidate>/model_weights
cp examples/pi05_libero_finetune/submission/policy_server.py <candidate>/policy_server.py
cp examples/pi05_libero_finetune/submission/requirements.txt <candidate>/requirements.txt
```

确认 `model_weights/tokenizer/`、预处理统计和 config 都存在。只有明确决定提交后，才把 candidate 复制/整理为 `submission_candidate`；不要直接改官方 server 的不可修改部分。

## 6. 本地 server smoke 与评测

终端 A：

```bash
cd <candidate>
python policy_server.py --port 8000
```

终端 B：

```bash
cd /workspace/PARC2026_final
source env.sh
python tools/run_experiment.py --action eval \
  --config configs/pi05_lora.yaml \
  --server-url http://localhost:8000
```

也可以直接使用官方命令：

```bash
python -m pipeline --server-url http://localhost:8000 \
  --track track1 --n-episodes 2 --max-steps 300
```

官方本地结果包含成功率和轨迹原始指标，不包含 leaderboard 的非公开归一化分数。先用 1–2 episode 验证 server，再扩大 episode 数；任何 `/act` timeout、NaN、错误 shape 或 collision 都停止后续流程。

## 7. validator、Docker E2E 与打包

```bash
python tools/run_experiment.py --action validate --target <candidate> --static
python tools/run_experiment.py --action validate --target <candidate> --install --pip-dry-run
python tools/run_experiment.py --action pack --submission-dir <candidate>
python tools/run_experiment.py --action validate \
  --target artifacts/parc2026_submission.zip
```

最终应在官方 Docker 中执行：

```bash
python evaluate.py /data/parc2026_submission.zip --n-episodes 2
```

评测环境会屏蔽外部网络；tokenizer/权重如果依赖运行时下载，提交会失败。

## 8. Omnicampus 手动提交

本选 1 的 assignment ID（附件 Version 1.0）：Track 1=`1007`、Track 2=`1008`、Track 3=`1009`。提交目录应是未压缩目录，`omni` 会负责压缩；也可以浏览器上传 `.zip`。

```bash
omni submit --assignment-id 1007 \
  --path /path/to/submission_candidate --dry-run
```

确认 dry-run 的课题名、Track、提交路径、文件大小和剩余次数均正确后，再手动执行：

```bash
omni submit --assignment-id 1007 \
  --path /path/to/submission_candidate
```

每个 Track 每日最多一次，上传完成时计数，提交错误也会消耗次数。临近午夜和截止时间不要提交大包；上传时间可能晚于命令启动时间。提交后等待 Omnicampus 显示“採点完了”，不要连续点击提交或刷新。

## 9. 结果归档与停止 GPU

归档 resolved config、run manifest、git commit/diff hash、dataset/model/mix hash、server log、评测 JSON、validator JSON、zip SHA-256、W&B URL 和人工确认记录。不要归档 token、完整 observation 或无必要的视频。训练/Notebook 完成后关闭 kernel；Omnicampus GPU notebook 即使关闭浏览器标签也可能继续运行。
