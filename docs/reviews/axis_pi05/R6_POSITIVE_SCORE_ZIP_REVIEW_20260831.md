# AXIS Pi0.5 R6 正分优先 ZIP 独立审核记录

> 审核时间：2026-08-31（Asia/Tokyo）
> 被审核 commit：`8938bcada74df00341ba020bf7c94f40eebd5178`
> 审核结论：`approved`
> 批准范围：进入官方配布 Blackwell pre-check 与 Track 1 人工提交决策；不代表官方 scorer/L4 或分数已验证。

## 1. 冻结提交物

```text
path      artifacts/submission_axis_r6_step500_fast_positive_score_20260831.zip
bytes     9383571677
sha256    95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a
entries   3401
```

独立重算 ZIP SHA-256 与 sidecar/report 一致；`unzip -t` 全量 CRC 通过。官方仓库静态 validator 复跑：

```text
.venv_axis_pi05/bin/python validate_submission.py --static --json <ZIP>
ok=true, errors=[], warnings=[]
```

包内存在唯一 `model_weights/model_fast.safetensors`，不存在 `model_weights/model.safetensors`；root layout、大小、duplicate/symlink/nested archive 门禁通过。

## 2. 模型与转换

```text
candidate                       r6_dual_l2_fresh500_r1
topology / step                 L2_action_lora_heads / 500
normalization                   ACTION/STATE QUANTILES
folded source sha256            5c73e1e8217dbb75a934c2b0165208e851f69e66cfe2ea0430590a29901e6d5b
FAST sha256                     e55bf7dcca78b2195993dd7a12b96a4ea13e94761d711c99a82b7411c3f8ceb6
FAST bytes / keys               9354050544 / 813
FAST dtype tensors              F16 255 / F32 558
missing/unexpected/shape        0/0/0
strict reload                   passed
vision patch weight+bias        present, mapped, shape exact
historical FAST / training      false / false
```

manifest 中 candidate、topology、step、QUANTILES、source/processor/FAST hashes 与执行证据一致；`submission_status=NOT_SUBMITTED`、`official_result=false`、`auto_submit=false`。

## 3. 行为门禁

### 3.1 Source↔FAST parity

```text
fixtures × seeds                8 × 2 = 16
executed steps                  8 of chunk 10
finite rate                     1.0
median abs error                0.0027961135
p95 abs error                   0.0362169139
max abs error                   0.1235293150
mean cosine                     0.9991701320
gripper sign agreement          1.0
status                          passed
```

预注册门禁为 median `<=0.03`、p95 `<=0.10`、execute8 gripper sign flip `0`；全部通过。max 超过 p95 门限不是 failure，因为 max 被要求记录但未注册为 hard threshold。

### 3.2 Server 与 C5

```text
/act                            12/12
p50 / p95 / max seconds        0.0483 / 1.0131 / 1.4912
FAST/chunk/execute/EMA/RTC      correct
SLOW/remap/vision/key warning  absent
C5 coverage                     5/5
source-success reproduced       3/3
HTTP/nonfinite/stall            0/0/0
```

C5 的三个 source-success cases 在 FAST 下全部复现；另一个旧 timeout diagnostic case也在 FAST 下成功。该结果是部署回归证据，不是隐藏测试分数估计。

## 4. 独立测试与资源边界

```text
tests                           122 passed, 7 skipped, 21 warnings
root/nested at audit start      clean
nested OpenPI                   dafa5a27266996eecf28920cacc32f3629776d91
GPU0 touched                    false
upload/submission               not performed
```

## 5. 剩余风险与解释

1. `official_docker_e2e.json` 的 execution 明确为 `local_offline_staged_server_equivalent`，不是官方 Docker；文件名不能被解释成官方验证。
2. 本地 server cold startup 约 `362s`，低于 contract 的 official `1200s`，但高于 local默认 `120s`；必须在官方 pre-check 用 `1200s` health timeout重验。
3. `server_latency.json.health_error` 保存的是 health 成功前最后一次 connection-refused poll；最终 `health=true`、reset成功、12/12 actions成功。它是报告语义瑕疵，不是实际 health failure；不得改写历史报告或因此重建 ZIP。
4. 本轮未记录可映射到 L4 24GB 的正式 peak VRAM；Blackwell 可验证运行时兼容性，但其 96GB 显存不能替代 L4 容量证明。
5. 本地 B80/C5 与 parity 均支持“比旧 SLOW 0/6 包更可信”，但不能保证官方一定 `>0`。

结论：当前 ZIP 是唯一获本次审核批准进入官方 pre-check 的 Track 1 artifact。pre-check 前不得修改或重建；任一 byte 变化都必须使用新文件名/hash并重新审核。
