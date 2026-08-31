# AXIS Pi0.5：E1 JAX Microbatch-1 可恢复全量推理计划

> 制定日期：2026-08-31（Asia/Tokyo）
> 审核输入 commit：`48206a708439b4a08af0ea9f8809eab9950572a5`
> 审核结论：`remediation_required`
> 当前状态：`e1_jax_mb1_canary_failed`
> 唯一目标：使用既有 immutable lock，以 folded JAX source、固定 `inference microbatch=1` 完成 340/340 observations，并生成可恢复、可审核的 full artifact。
> 执行边界：允许一张空闲本地 GPU，优先物理 GPU1；完成或失败后 hard stop。
> 禁止：LeRobot/PyTorch inference、JAX↔PyTorch comparator、environment、closed-loop、E2、FAST、ZIP、训练、比赛 GPU、push、上传或比赛提交。

本轮执行已在 R2 canary hard gate 停止：`[0,16)` 覆盖、finite、provenance 与 GPU boundary 通过，但前 4 个 action 未与旧 sentinel bitwise exact；未启动 R3 full。详见 `data/reports/axis_libero_30k/post_r6_lora/r6_egl_statistical_release/e1_jax_mb1_recovery/summary.json`。

## 0. 一句话任务

不要重建 E1 输入，也不要重跑训练。先给 JAX full inference 增加 fail-closed 的原子分片与 resume 机制；用 `microbatch=1` 跑 16-observation canary，确认其前 4 个 action 与既有 sentinel bitwise exact；随后在同一 fresh process 中完成 340 observations，并在每 16 个 observation 后落盘可校验分片。聚合完成后停止，等待下一次审核。

```text
R0 immutable release audit
  → R1 CPU-only resumable shard implementation/tests
  → R2 GPU1 16-observation microbatch-1 canary
  → R3 GPU1 340-observation microbatch-1 full + atomic shards
  → R4 aggregate/report/tests/clean commit
  → STOP（不进入 PyTorch）
```

## 1. R0：immutable release audit

执行前必须记录 HEAD、branch、root/nested clean、nested commit、GPU UUID/占用与 GPU0 外部 PID。`48206a7` 必须是 HEAD ancestor。

下列输入及 hash 必须 exact：

```text
input lock
  data/reports/axis_libero_30k/post_r6_lora/r6_egl_statistical_release/e1_action_envelope/e1_input_lock.json
  sha256 b96acf22e01f81d89c0bebf8adfd3dfd729e485cdc29a1b87c6fc748f67534c4

pair manifest
  data/reports/axis_libero_30k/post_r6_lora/r6_egl_statistical_release/e1_action_envelope/e1_frozen_pair_manifest.json
  sha256 09c9e000135c192b6a3154b82c06f754a047d8006122a888f6c933852887f66e

frozen inputs/noise
  data/reports/axis_libero_30k/post_r6_lora/r6_egl_statistical_release/e1_action_envelope/e1_frozen_inputs_and_noise.npz
  sha256 263a7723d1f866eb73eee9423eda7480d96e1678b4a401bd4d86460b7c37c7fe

prior sentinel
  data/reports/axis_libero_30k/post_r6_lora/r6_egl_statistical_release/e1_action_envelope/jax_source/sentinel/actions.npz
  actions sha256 f3977cebecbd186c37ee5f4a5de53e7ce43bcef8141ed93586cd16e335422e65
```

同时验证 170 pairs、340 observations、snapshot/trajectory `13/157`、input/noise/model/norm/raw-tree provenance 与原 E1 summary 一致。禁止修改、重生成或重排序上述 lock/manifest/input/noise/sentinel；原 `e1_action_envelope/summary.json` 的 OOM failure 必须保留为历史证据，不得覆盖为成功。

任一 hash/count/provenance 不同：写 `e1_jax_mb1_recovery_input_drifted` 并停止，不运行 GPU。

## 2. R1：CPU-only resumable shard implementation

在既有 `tools.axis_pi05.e1_action_envelope` 基础上实现最小 recovery runner。要求：

- inference microbatch 固定为 `1`；CLI 若给其他值必须 fail closed；
- 默认一个 fresh process 完成全量，模型只加载一次；不要为每个 shard 重启并重复编译；
- 每处理 16 observations 原子保存一个 shard：先写同目录临时文件，完成 flush/hash 后再 rename 为最终文件；
- 每个 shard 包含连续 `[start, stop)`、internal/external actions、shape/dtype/finite、action SHA-256、lock/manifest/input/noise/model/norm/raw-tree hashes、microbatch、物理 GPU 与 runtime provenance；
- shard 名和 aggregate order 必须由 observation index 唯一确定，禁止依赖文件系统枚举顺序；
- resume 前逐个重算 shard 内容/metadata hash，只接受从 index 0 开始连续且全部 provenance exact 的完整 shard；第一个缺失 shard之后的孤立 shard不得跳过使用；
- corrupt、overlap、gap、wrong-order、wrong-hash、partial shard 一律 fail closed；
- ignored action NPZ 不进 Git，小型 JSON/Markdown summary、工具和测试进 Git；
- 在 CPU 单元测试中覆盖 atomic partial rejection、resume gap/overlap/corruption/provenance drift、aggregate order、340 exact coverage 与 immutable source lock。

新的独立 namespace：

```text
data/reports/axis_libero_30k/post_r6_lora/r6_egl_statistical_release/
  e1_jax_mb1_recovery/
    input_audit.json
    canary/report.json
    full/shards/<start>_<stop>.npz       # ignored
    full/shards/<start>_<stop>.json
    full/actions.npz                     # ignored aggregate
    full/report.json
    summary.json
    E1_JAX_MB1_RECOVERY_REPORT_ZH.md
    THREAD_REPORT_ZH.md
```

实现、focused tests、`py_compile` 与 `git diff --check` 通过后才可运行 GPU。此阶段不得加载模型或观察新 action。

## 3. R2：16-observation canary

使用一个 fresh process、同一 folded JAX source、物理 GPU1（若不空闲可选 GPU2/3）、FP32、`microbatch=1`，按 lock 顺序运行 observations `[0,16)`。

硬门禁：

```text
coverage                              16/16
microbatch                            1
internal/external shape               [16,10,32] / [16,10,7]
all finite                            true
input/noise/model/norm/raw-tree hash  exact
first 4 internal actions              bitwise exact vs prior sentinel
first 4 external actions              bitwise exact vs prior sentinel
GPU0 touched                          false
allocator/dtype/model changes         none
```

必须保存 latency、peak GPU memory、target PID/UUID 与 GPU0 evidence。任一 mismatch、OOM、non-finite 或 GPU0 触碰时写准确 first failure 并停止；不得尝试 allocator preallocation/memory fraction、BF16、alternate GPU model、删减输入或放宽 action equality。

## 4. R3：340-observation full recovery

R2 全部通过后，启动新的 fresh process进行 full run。固定：

```text
backend                      r6_folded_jax_source
observation order            frozen lock index 0..339
inference microbatch         1
logical shard size           16（末 shard 4）
internal/external action     [340,10,32] / [340,10,7]
runtime/dtype                与通过的 sentinel/canary exact
```

正常情况在同一进程顺序完成并每 16 observations 原子落盘；若进程非模型原因中断，可按 R1 的连续 shard 校验恢复，但恢复时仍需 fresh process、相同 GPU/runtime，且必须记录 resume boundary/reason。不得把多次运行中同一 index 的不同 action择优拼接。

按既有 sentinel 的 steady-state latency 约 `12s/observation` 估算，340 observations 约需 `68` 分钟，另加首次编译、分片校验与聚合，预留 `80–100` 分钟。长运行期间至少按 shard 更新机器可读进度；不得为了缩短时间提高 microbatch。

full hard gates：

- 340/340 连续、无 duplicate/gap/reorder；
- 所有 internal/external action finite；
- input/noise/model/norm/raw-tree/processor/source-config hashes exact；
- aggregate 与所有 shard 的逐元素拼接 bitwise exact；
- aggregate 前 4 actions 与 prior sentinel bitwise exact；
- report 明确记录 `microbatch=1`、每 observation latency、peak memory、resume count 和 GPU provenance；
- GPU0 未触碰；没有 PyTorch、comparator 或其他禁止阶段启动。

任一门禁失败立即停止，保留已完成 shard 与 failure report；禁止在本计划内换 dtype、allocator、模型、batch 或阈值重试。

## 5. R4：报告、测试与 hard stop

成功状态只能写：

```text
e1_jax_full_recovered_pending_review
```

失败使用准确状态，例如：

```text
e1_jax_mb1_canary_failed
e1_jax_mb1_full_failed
e1_jax_mb1_recovery_provenance_failed
```

summary 必须同时指出：这只恢复了 E1 JAX source full artifact，不代表跨 backend parity 或 EGL envelope 已完成，不批准候选，也不批准发布。

最终验证：

```text
.venv_axis_pi05/bin/python -m py_compile tools/axis_pi05/*.py pipeline/*.py
.venv_axis_pi05/bin/pytest -q tests
JSON/shard count/hash/schema checks
git diff --check
root/nested clean commit
```

形成 clean commit 后 hard stop，等待人工审核。不得自动进入 LeRobot/PyTorch、comparator、closed-loop、E2、FAST、ZIP、训练、比赛 GPU、push、上传或比赛提交。

## 6. 下一审核点

审核者将独立核对：

- immutable hashes 与旧 OOM failure 是否保留；
- canary 前 4 actions 与 prior sentinel 是否 bitwise exact；
- 340/340 shard/aggregate coverage、顺序、finite、hash 与 GPU evidence；
- resume 是否 fail closed、测试和 Git 是否 clean。

只有审核通过后，才会归档本计划并另建一个仅运行 on-disk LeRobot/PyTorch full source 的计划；PyTorch 不在本计划授权范围内。
