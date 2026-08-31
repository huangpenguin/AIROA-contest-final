# AXIS Pi0.5 R6：Track 1 官方 Pre-check 与人工提交计划

> 制定日期：2026-08-31（Asia/Tokyo）
> 审核输入 commit：`8938bcada74df00341ba020bf7c94f40eebd5178`
> 当前状态：`r6_track1_official_precheck_ready`
> 目标：对唯一冻结 ZIP 做官方配布 Blackwell pre-check；通过后停在人工 Track 1 提交确认点，并完整保存官方日志/结果。
> 禁止：修改/重建 ZIP、训练、JAX/E1/C20、Track 2/3、SLOW fallback、公开上传、自动 `omni submit` 或重复提交。

## 0. 唯一 artifact

```text
local path
  /home/huang/code/PARC2026_final/artifacts/submission_axis_r6_step500_fast_positive_score_20260831.zip

bytes
  9383571677

sha256
  95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a

Track / assignment
  Track 1 / 1007
```

任何 hash/bytes 不同立即停止。不得用旧 base、F2/M1 FAST、重新压缩包或同名覆盖包替代。

## 1. O0：启动官方 Notebook 前的人工准备

这一步需要用户明确决定消耗配布 GPU 计时与外部传输；执行者不得自动启动 Notebook或上传。

启动前确认：

- 当天 Track 1 是否仍有一次提交机会；上传完成也会消耗次数；
- ZIP 已通过私有、可授权的方式可传入官方环境；不得公开 HF/GitHub；
- 本地 ZIP、`.sha256`、review record 与 commit `8938bca` 已保留；
- 预留至少 60 分钟完成 9.38GB 传输、约 6 分钟 cold start、pre-check和日志回传；
- 临近午夜若无法保证上传完成，不启动正式提交。

若使用官方 `~/data` 持久盘，ZIP 与小报告放 `~/data/submissions/r6_track1/`，完成后执行 `parc-home-sync data-push`。解压/容器临时文件放 `/opt/dlami/nvme`，不要挤占 40GB home loop。

## 2. O1：官方环境只读验收

Notebook 启动后立即记录：

```text
date -Is
nvidia-smi
df -h "$HOME" "$HOME/data" /opt/dlami/nvme
```

将 ZIP 放入官方环境后先执行：

```text
sha256sum <official_zip_path>
stat -c '%s %n' <official_zip_path>
unzip -t <official_zip_path>
```

必须 exact 匹配第 0 节。随后用仓库官方 validator 运行：

```text
python validate_submission.py --static --json <official_zip_path>
```

静态 validator、CRC、hash、size 任一失败立即停止，不解压启动模型、不提交。

## 3. O2：配布 pre-check Docker

严格按官方 slides p.35 以后提供的 Docker 启动方法执行，不自制环境替代。容器内保持网络关闭/离线；不得设置 HF token作为运行依赖。

要求：

- Python/CUDA/PyTorch 与配布 pre-check image一致；
- health startup timeout 使用 official contract `1200s`，记录实际 cold-start；
- 日志必须出现 `FAST_REQUIRED_AND_VERIFIED`、`fast=True`、`chunk=10`、`n_action_steps=8`、`action_ema=0.0`、`rtc=False`；
- 不得出现 SLOW/remap/vision/missing/unexpected-key warning；
- `/health`、带 seed `/reset`、至少 3 个真实格式 `/act` 全部成功，action `(7,)`、float32、finite，每次 `<10s`；
- 记录 `nvidia-smi` before/load/act/after、target PID、峰值显存；Blackwell峰值仅作兼容证据，不宣称等价于 L4 24GB；
- server退出后确认无遗留进程。

如果官方提供的 `evaluate.py` 可用，执行：

```text
python evaluate.py <official_zip_path> --n-episodes 2
```

保存 stdout/stderr、每 episode outcome、server log与退出码。该结果仍是 pre-check，不是 leaderboard分数。

O2 通过状态：

```text
r6_track1_official_precheck_passed_pending_human_submit
```

任何 OOM、startup `>=1200s`、`/act >=10s`、HTTP错误、SLOW/warning、nonfinite、shape或offline依赖失败：写 `official_precheck_failed`并停止，不提交。

## 4. O3：回传证据并关闭 GPU

保存到官方持久盘：

```text
~/data/submissions/r6_track1/precheck_<timestamp>/
  environment.txt
  zip_identity.txt
  static_validator.json
  server.log
  request_metrics.json
  evaluate.log
  summary.json
```

执行 `parc-home-sync data-push`，确认完成后主动 shutdown Notebook/kernel；关闭浏览器标签不等于停止计时。

把小型报告同步回本项目的新 namespace并形成 clean commit。不得把 token、完整 observation或 9.38GB ZIP提交到 Git。

## 5. O4：人工 Track 1 提交确认

只有以下条件同时满足才向用户建议提交：

```text
ZIP hash/bytes                         exact
local release review                   approved
official pre-check                     passed
FAST markers / no forbidden warning    passed
official /act <10s                     passed
remaining Track 1 daily attempt        confirmed
assignment                             1007 / Track 1
```

执行者必须先停止并把 approved ZIP path/hash、pre-check summary和当日剩余次数展示给用户。只有用户再次明确确认“提交”后，才能由用户手动浏览器上传 ZIP，或对解压目录先运行：

```text
omni submit --assignment-id 1007 --path <unpacked_submission_dir> --dry-run
```

核对课题名、Track、路径、大小、剩余次数后，用户手动去掉 `--dry-run`。本计划不授权执行正式提交命令。

## 6. O5：官方分数归档

提交后不要重复点击或重新提交。等待 Omnicampus 显示计分完成，保存：

- submission ID、assignment、开始/完成时间；
- 完整官方 stdout/stderr/server log；
- track、episodes、success/score；
- 实际 FAST/SLOW marker、startup、每次 `/act` latency；
- ZIP SHA-256 与结果对应关系。

若分数仍为 0，不删除产物，不立即训练；先比较官方日志中的 FAST marker、vision/key、reset/chunk/action与本地证据。若得到 `>0`，冻结此包为首个 deployment baseline，再恢复 standard LIBERO full-FT研究。

## 7. Hard stop

本计划完成 O0–O3 后必须停止等待人工提交决定。不得顺便启动训练、JAX 340、C20、Track 2/3或第二个 ZIP。
