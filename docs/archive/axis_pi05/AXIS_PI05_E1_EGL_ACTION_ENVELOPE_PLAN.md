# AXIS Pi0.5：E1 全量 EGL Action-Response Envelope 执行计划

> 归档日期：2026-08-31（Asia/Tokyo）
> 归档状态：`stopped_at_e1_3_jax_full_microbatch4_oom`
> 已完成：E1.0 input lock、E1.1 frozen inputs/noise、E1.2 JAX 4-observation sentinel。
> 未完成：E1.3 JAX 340-observation full、E1.4 PyTorch、E1.5 comparator。
> 本文件不再提供执行授权；后续仅以计划索引中的唯一活动计划为准。

> 制定日期：2026-08-31（Asia/Tokyo）
> 审核输入 commit：`327316f2ffdca0df3375748bcdd65b5ce263f8f8`
> 输入状态：`e0_closure_complete_pending_review`
> 批准状态：`e0_closure_approved_e1_ready`
> 目标：在冻结的 170 个 content-unique EGL pair 上，分别测量 R6 folded JAX 与真实 on-disk LeRobot folded PyTorch 的 action response。
> 执行边界：允许单张空闲本地 GPU；只执行 E1.0–E1.6，完成后 hard stop。
> 禁止：closed-loop、E2 protocol/rollout、FAST、ZIP、训练、比赛 GPU、push、上传或比赛提交。

## 0. 一句话任务

先在不读取任何新 action 的前提下冻结 170-pair runnable inputs 和固定 noise；形成 lock commit 后，在同一物理 GPU 上顺序运行 JAX/PyTorch 两个 source backend，生成完整 sensitivity envelope。任何 hard gate 失败立即停止，不进入 E2。

```text
E1.0 release/input lock
  → E1.1 runnable manifest + frozen inputs/noise（CPU，先 commit）
  → E1.2 backend/runtime preflight
  → E1.3 JAX source inference（170 pairs / 340 observations）
  → E1.4 LeRobot PyTorch source inference（同一 GPU、同一输入/noise）
  → E1.5 envelope comparator + hard gates
  → E1.6 reports/tests/clean commit
  → STOP，等待人工审核
```

## 1. 不可变输入

### 1.1 E0 与旧 failure

```text
E0 closure summary
  data/reports/axis_libero_30k/post_r6_lora/r6_egl_statistical_release/e0_closure/summary.json
  sha256 088376936c2ecba61258316bcafe48c70c04c7c115c62917222f7add2aba3c43

E0 content-unique manifest
  data/reports/axis_libero_30k/post_r6_lora/r6_egl_statistical_release/e0_closure/content_unique_pair_manifest.json
  sha256 b4ea8d3bed7f9cfc60f2f7f53613471e222d9857a8d6857f39b854720905c7bf

old S1.3 status                        s1_real_egl_pairs_failed
old strict threshold                   1e-6
old S1.3 summary sha256                96f6c55f2b42ea55d3e103820e52912a4fcfad79794b686411e0e1659ca1a952
old pair manifest sha256               dacb81bcc6216c0b86be9ff350d731cdc199612b5e8f6f662560598cd588e972
raw-LoRA Q2 status                     n1_q2_policy_sensitivity_failed_raw_lora_fresh_replay
```

E1 通过不能回写为旧 `1e-6` 门禁通过；它只能证明完整 EGL envelope 落入此前预注册的人工统计审核区间。

### 1.2 R6 source artifacts

```text
run id                                 r6_dual_l2_fresh500_r1
raw Orbax tree sha256                  d203635f1eb0b006ec27841db6ce5171f74a37035eca4291c8107c05838c276f
folded model.safetensors sha256        5c73e1e8217dbb75a934c2b0165208e851f69e66cfe2ea0430590a29901e6d5b
R6 fresh norm sha256                   4c803bf59c52677c36b6b420c614d8a7742e0847708443863e13e67cb80b18f5
normalization                          ACTION/STATE QUANTILES
external/internal action               7D / 32D
chunk / executed steps                 10 / 8
```

唯一 PyTorch source 是上述 on-disk folded model 与其 LeRobot pre/post processors。禁止旧 F2/M1 model、旧 norm、旧 processor、FAST 或 SLOW/remap fallback。

## 2. E1.0：release 与环境锁定

执行前记录：

- HEAD、branch、root/nested clean；`327316f` 必须是 HEAD ancestor；
- nested OpenPI commit；
- 第 1 节全部文件/hash；
- Python/JAX/PyTorch/CUDA/cuDNN、driver、GPU UUID；
- `CUDA_VISIBLE_DEVICES` 到物理 GPU 的映射；
- GPU0 外部 PID 与 GPU1–3 占用快照。

主运行只选择一张明确空闲的 GPU，优先物理 GPU1；若 GPU1 不空闲，可选择 GPU2/3，但两个 backend 必须在同一物理 GPU 上顺序 fresh-process 运行。GPU0 及其外部进程不得触碰。

V100 不使用 BF16；保持项目已验证的 FP32/runtime。任何 artifact/hash/processor/norm 漂移立即写 `e1_input_or_runtime_drifted` 并停止。

非阻断机械清理仅允许：删除 `_candidate_common` 中值相同的重复 key，并新增一个不改变输出的测试。CLI 固定使用：

```text
.venv_axis_pi05/bin/python -m tools.axis_pi05.r6_s1_egl ...
```

不得改变 semantic key、pair order、image transform、state/action contract 或 E0 manifest。

## 3. E1.1：先冻结 runnable inputs 与 noise

本阶段 CPU only，且必须在模型 inference 前完成并形成独立 clean lock commit。

### 3.1 Runnable pair manifest

从 E0 manifest 按既有顺序生成：

```text
data/reports/axis_libero_30k/post_r6_lora/r6_egl_statistical_release/e1_action_envelope/
  e1_input_lock.json
  e1_frozen_pair_manifest.json
```

必须保持：

```text
pair count                              170
snapshot / trajectory                   13 / 157
observation sides                       340
semantic key duplicates                 0
source occurrence hash mismatch         0
prompt/state/image hash mismatch        0
other-camera mismatch                   0
not evaluable                           0
```

每个 runnable pair 明确保存：

- stable `pair_id`、E0 content hash 与 semantic key；
- case/task/instruction、prompt hash；
- source kind、phase/mode/frame index、camera；
- left/right selected-camera data hash 与可重读位置；
- exact other-camera data hash 与可重读位置；
- 8D policy state 值或不可歧义引用及 SHA-256；
- 两个 side 的完整 policy input hash；
- source occurrence 和 Q1 comparison provenance。

prompt 必须由冻结 C20/task manifest 恢复并与 E0 prompt hash 相符；不得按 hash 猜测或手写。

### 3.2 Frozen noise

为每个 pair 生成稳定 pair-local noise：

- seed 从固定 global seed + content hash 以明确算法导出；
- left/right 使用完全相同 noise；
- JAX/PyTorch 使用完全相同数值 noise，而不只是相同整数 seed；
- dtype/shape/value SHA-256 写入 manifest；
- 不使用 runtime 默认 RNG。

建议生成一个 ignored 的 `e1_frozen_inputs_and_noise.npz`，在 lock 中记录 bytes/SHA-256。大型 NPZ 不进 Git；lock/manifest 必须进 Git。

### 3.3 Lock-before-action 门禁

在任何 backend load/inference 前：

- 校验 170/340、hash 与唯一性；
- 写明 `actions_observed=false`；
- `git diff --check`、JSON 校验、focused tests；
- 形成 clean lock commit。

lock 失败则停止，不得运行模型。运行 action 后不得修改 pair/noise/阈值；若发现 manifest 错误，整个 E1 作废并回到新审核计划。

## 4. E1.2：backend 与 runtime preflight

分别以 fresh process 加载：

1. R6 folded JAX source；
2. on-disk LeRobot folded PyTorch source +真实 processors。

先用冻结 manifest 的 4 个 sentinel pair 验证：

- 实际 model/norm/processor hash；
- token/mask/state/image/action shape/dtype；
- input/noise SHA 与 lock exact；
- finite；
- external action 7D、internal action 32D、chunk 10；
- PyTorch fresh-process sentinel replay bitwise exact；
- 无 SLOW/remap/vision missing-key warning。

sentinel 失败状态为 `e1_backend_or_contract_preflight_failed`，不得继续 170-pair run。

## 5. E1.3–E1.4：两 backend 全量 inference

按 lock 中相同 pair order 运行 340 observations。两个 backend 使用同一物理 GPU、同一 frozen inputs/noise，顺序 fresh process；第一个进程完全退出并保存 GPU monitor 后才能启动第二个。

每个 backend 保存 ignored raw artifact：

```text
internal action       [340, 10, 32]
external action       [340, 10, 7]
tokens / masks
input / noise hashes
per-batch latency
model/runtime provenance
```

raw NPZ 不进 Git，但必须记录 SHA-256、bytes、shape/dtype、finite count。不得缓存另一 backend 的 action、复制 trace 或用 JAX 冒充 PyTorch。

任一 OOM、non-finite、coverage gap、hash/contract drift 或 GPU0 触碰立即停止。

## 6. E1.5：完整 action-response comparator

对 170 pairs 分别计算：

### 6.1 EGL-induced response

对每个 backend 比较同 pair left/right：

- internal/external full-chunk 与 execute8 max/mean；
- per horizon、per dimension；
- camera、snapshot/trajectory、phase/mode 分组；
- p50/p95/p99/max；
- gripper sign flip；
- worst pair/content hash 和完整 provenance。

### 6.2 Backend correspondence

对 340 个相同 side 输入比较 folded JAX 与 PyTorch source：

- internal/external full/execute8；
- token/mask/state/noise mismatch；
- per-dim/per-horizon p50/p95/p99/max；
- gripper sign agreement。

### 6.3 Hard gates

进入下一审核点的必要条件：

```text
pair/input coverage                         170/170, 340/340 per backend
all actions finite                          true
shape/dtype/model/norm/processor contract   exact
input/state/prompt/noise hash mismatch      0
token/mask mismatch                         0
PyTorch sentinel fresh replay               bitwise exact
任一 backend EGL-induced external max       <= 1e-4
EGL-induced execute8 gripper sign flip      0
folded JAX ↔ PyTorch external max           <= 1e-3
backend gripper sign agreement               100% on execute8
```

`1e-4` 是进入后续统计 closed-loop 设计的人工审核上界，不是旧 strict `1e-6` 门禁的新阈值。不得把通过写成模型显著稳定、性能提升或正式 winner。

## 7. E1.6：产物、测试与停止

新 namespace：

```text
data/reports/axis_libero_30k/post_r6_lora/r6_egl_statistical_release/e1_action_envelope/
  e1_input_lock.json
  e1_frozen_pair_manifest.json
  summary.json
  pair_metrics.jsonl
  E1_ACTION_ENVELOPE_REPORT_ZH.md
  THREAD_REPORT_ZH.md
  jax_source/<raw ignored artifacts>
  lerobot_pytorch_source/<raw ignored artifacts>
```

small JSON/JSONL/Markdown、工具、测试和 handoff 显式进入 Git；action NPZ、模型与大输入 artifact 不进 Git，只保存 hash。

验证：

```text
.venv_axis_pi05/bin/python -m py_compile tools/axis_pi05/*.py pipeline/*.py
.venv_axis_pi05/bin/pytest -q tests
jq/jsonl schema and count checks
git diff --check
```

全部 hard gate 通过时状态只能是：

```text
egl_action_response_bounded_pending_statistical_closed_loop
```

任一门禁失败则写准确 first failure 和未执行阶段，状态使用 `e1_*_failed`，不得删除失败 pair 或改阈值。

形成 clean commit 后停止，等待人工审核。不得在同一 section 创建或执行 E2 protocol、closed-loop、FAST 或 ZIP。

## 8. 下一审核点

审核者将：

- 独立核对 lock-before-action commit、340 input/action coverage、raw artifact hash、worst pairs 与全部 hard gate；
- 通过则归档本计划，并落盘独立 E2 统计协议预注册计划；
- 失败则归档本计划，并落盘最小 E1 remediation；
- 两种情况均更新 plan index、review record、Markdown/JSON handoff 并形成 clean commit。
