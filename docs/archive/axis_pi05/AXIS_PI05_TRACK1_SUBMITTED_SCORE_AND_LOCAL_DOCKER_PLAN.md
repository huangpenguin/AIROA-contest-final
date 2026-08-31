# AXIS Pi0.5 R6：Track 1 已提交后的本地 Docker 与计分归档计划

> 归档日期：2026-08-31（Asia/Tokyo）
> 归档状态：`superseded_by_official_score_0_47`
> 用户已报告 Track 1 正式得分 `0.47`；真实 scorer 结果已经提供比本地 Docker E2E 更强的部署证据。
> T0–T2 未执行，本文件不再提供 Docker、GPU或结果归档任务授权。

> 制定日期：2026-08-31（Asia/Tokyo）
> 输入 commit：`43aef30`
> 当前状态：`r6_track1_user_submitted_score_pending`
> 唯一目标：不使用配布 GPU、不重复提交，在自有 GPU1 完成真实 scorer-dependency Docker E2E，并在官方计分完成后把结果归因到部署或策略能力。
> 禁止：训练、修改/重建 ZIP、官方 Blackwell Notebook、GPU0、Track 2/3、自动上传或再次提交。

## 0. 冻结输入

```text
ZIP
  /home/huang/code/PARC2026_final/artifacts/submission_axis_r6_step500_fast_positive_score_20260831.zip

bytes
  9383571677

sha256
  95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a

assignment
  1007 / Track 1

local image
  parc2026:latest
  sha256:2481fd635a465c6b9dd38b6101c8307ce97d0bf7a3c2ffb57e487103cec9ccec
```

任一 ZIP hash/bytes 不一致立即停止。不得重压缩、换包或把后续本地结果解释为已经提交了另一个 artifact。

## 1. T0：登记人工提交事实

先建立新的小型记录：

```text
data/reports/axis_libero_30k/post_r6_lora/r6_track1_submission/
  submission_record.json
  THREAD_REPORT_ZH.md
```

在没有 Omnicampus 证据时只能填写：

```text
status                user_reported_submitted_score_pending
assignment            1007
artifact/hash/bytes    exact
submission_id         null
submitted_at          null
score                  null
official_log           null
official_result        false
```

若用户提供页面截图、submission ID或日志，再追加原始值与证据路径；不得自行猜测时间和分数。禁止调用 `omni submit`。

## 2. T1：锁定本地 scorer-dependency image

只读复核：

```bash
nvidia-smi
docker --version
docker inspect parc2026:latest
sha256sum Dockerfile setup.sh evaluate.py validate_submission.py
docker run --rm parc2026:latest bash -lc \
  'sha256sum Dockerfile setup.sh evaluate.py validate_submission.py; python -c "import sys,torch; print(sys.version); print(torch.__version__, torch.version.cuda)"'
```

门禁：

- image ID 必须与第 0 节一致；
- Dockerfile/setup/evaluator/validator 的 host/container hash exact；
- Python `3.10.12`、PyTorch `2.11.0+cu130`、CUDA `13.0`；
- nvidia runtime 可用；
- GPU1 free，GPU0外部 PID不变。

核心 hash exact 时禁止为了文档 README 变化重建 40.8GB image。

## 3. T2：自有 GPU1 真实 Docker E2E

创建新输出目录，不覆盖历史 `official_docker_e2e.json`：

```bash
export R6_ZIP="/home/huang/code/PARC2026_final/artifacts/submission_axis_r6_step500_fast_positive_score_20260831.zip"
export R6_DOCKER_OUT="/home/huang/code/PARC2026_final/data/reports/axis_libero_30k/post_r6_lora/r6_track1_submission/local_docker_e2e"
mkdir -p "$R6_DOCKER_OUT/results"

sha256sum "$R6_ZIP" | tee "$R6_DOCKER_OUT/zip_identity.txt"
nvidia-smi | tee "$R6_DOCKER_OUT/nvidia_before.txt"

docker run --rm \
  --gpus 'device=1' \
  --network none \
  -e SERVER_TIMEOUT=1200 \
  -v "$R6_ZIP:/data/submission.zip:ro" \
  -v "$R6_DOCKER_OUT/results:/workspace/results" \
  parc2026:latest \
  bash -lc 'python evaluate.py /data/submission.zip --n-episodes 2' \
  2>&1 | tee "$R6_DOCKER_OUT/evaluate.log"

nvidia-smi | tee "$R6_DOCKER_OUT/nvidia_after.txt"
```

执行 Agent 应另启只读 GPU1 memory monitor，保存 target container/PID、load与推理峰值；不得影响 GPU0。若 `--network none` 发现提交包试图联网，判为 offline failure，不得开放网络重试来伪造通过。

通过门禁：

```text
container exit                         0
ZIP identity                           exact
FAST_REQUIRED_AND_VERIFIED / fast=True present
SLOW/remap/vision/missing warning      absent
health/reset                           success
each /act and /reset                   <10s
action                                 float32 (7,), finite
episodes attempted                     2
server/process cleanup                 complete
```

成功率可以是 0/2；这不等于 deployment failure。只要 server、时延、finite、shape和FAST门禁通过，就将其记作 `local_scorer_dependency_e2e_passed_behavior_observed`，并如实记录 episode outcomes。

若 V100 上因硬件特性失败，必须区分容器/依赖失败与 V100≠L4 的硬件差异；不得宣称正式 scorer必然失败。

## 4. T3：官方结果归档

官方页面显示“採点完了”后，由用户提供或放入仓库外的原始结果。记录：

- submission ID、assignment、submitted/completed time；
- ZIP hash；
- Track、score、success episodes/total；
- server startup、FAST/SLOW、key/vision warning、每次 `/act` latency；
- 完整官方 log 的本地私有路径与小型脱敏 summary。

未拿到这些证据时保持 `score_pending`，不要轮询、刷新或重复提交。

## 5. T4：根据官方结果选择下一研究计划

### A. score `>0`

冻结该 ZIP 为首个 official deployment baseline。下一审核重新激活 `AXIS_PI05_STANDARD_LIBERO_30K_RESEARCH_PLAN.md` 的 full-FT 主线，但先只在自有 GPU1–3 做 Phase 2 数值/restore/FSDP预飞行；配布 Blackwell继续保留给 recipe锁定后的长训练冲刺。

### B. score `=0` 且日志有 server/error/timeout/SLOW/key warning

下一计划只做最小 deployment remediation。不得用更多训练掩盖部署错误。

### C. score `=0`，但 FAST正确加载、所有 episode完整跑满且无 server错误

判为策略行为能力不足的主要证据。下一计划进入 canonical standard LIBERO full-FT，不再围绕同一 R6 500-step LoRA反复调阈值。

### D. score 尚未完成

T2完成后 hard stop，等待用户提供结果。

## 6. 配布 GPU 使用策略

本计划不启动配布 GPU。后续只有以下条件同时满足时才能制定 official training sprint：

1. 自有 V100 完成 full-FT 1/10/50/100-step finite、restore和global batch语义验证；
2. 选定 full-FT/LoRA recipe、数据 revision、norm、precision、checkpoint cadence；
3. 训练可在 12小时单次上限内安全续跑，checkpoint能同步到持久盘；
4. 估算 60小时内的 optimizer steps和至少一次恢复余量；
5. 用户再次明确授权启动 Notebook。

## 7. Hard stop

执行 T0–T2后停止。若官方结果已经由用户提供，可完成T3并输出 A/B/C 分支建议，但不得直接训练、启动官方 GPU或生成第二个 ZIP。
