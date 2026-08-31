# AXIS Pi0.5 R6：当日 FAST 诊断 ZIP 并行执行计划

> 制定时间：2026-08-31 17:40（Asia/Tokyo）
> 审核输入 commit：`222e5942d8274a10322ebf4679aaeb6d4f989420`
> 当前状态：`r6_same_day_diagnostic_zip_ready_to_execute`
> 目标：今天优先产出经过 R6 source identity、FAST parity、server smoke 与静态校验的本地诊断 ZIP；同时在独立 GPU 上补 JAX 340/340 数值证据。
> 资源：GPU0 禁用；GPU1 用 JAX evidence lane，GPU2 用 FAST parity/server，GPU3 只作备用或独立 source reference。
> 禁止：重新训练、修改 R6 model/norm、复用旧 F2/M1 FAST、SLOW fallback、比赛 GPU、push、上传或比赛提交。

## 0. 交付级别与时间预算

本计划允许连续执行 Z0–Z6，critical hard gate 失败才停止，不在每个成功小阶段等待人工审核。

```text
Z0 CPU release lock/code audit             20–40 min
Z1 R6 source staging + FAST export         30–70 min
Z2 source↔FAST fixed parity                30–60 min
Z3 FAST server + C5 smoke                  30–70 min
Z4 ZIP build/hash/static/offline           20–45 min
JAX 340/340 lane（与 Z1–Z3 并行）          80–100 min
```

若无新失败，约 2.5–4.5 小时可得到诊断 ZIP。交付状态分三档：

1. `no_zip_gate_failed`：FAST/export/parity/server critical gate 失败，不生成 ZIP。
2. `r6_diagnostic_zip_local_ready`：Z0–Z4 通过，但 full envelope/C20/official pre-check 未完成；ZIP 存在但 `submission_ready=false`。
3. `r6_local_release_candidate_pending_official_precheck`：Z0–Z6 全通过；仍需人工授权官方环境，不自动上传/提交。

## 1. Z0：release lock 与工具去历史硬编码

执行前记录 HEAD、branch、root/nested clean、GPU UUID/占用、GPU0 外部 PID；`222e594` 必须是 HEAD ancestor。冻结：

```text
candidate                    r6_dual_l2_fresh500_r1 / step 500
folded source model          outputs/axis_pi05/axis_libero_lora_folded/r6_dual_l2_fresh500_r1_step500/model.safetensors
source model sha256          5c73e1e8217dbb75a934c2b0165208e851f69e66cfe2ea0430590a29901e6d5b
raw Orbax tree sha256        d203635f1eb0b006ec27841db6ce5171f74a37035eca4291c8107c05838c276f
norm sha256                  4c803bf59c52677c36b6b420c614d8a7742e0847708443863e13e67cb80b18f5
normalization                ACTION/STATE QUANTILES
chunk / execute              10 / 8
action_ema / rtc             0.0 / false
```

先修正并测试已有 FAST 工具中的历史候选硬编码：

- `audit_fast_submission.py` 不得再写死 `F2/M1`、旧 norm 或旧 Orbax checkpoint；candidate/scope/norm/source 必须从本计划的 R6 release lock 显式传入并校验 hash。
- fixed parity 的 authoritative source 必须是真实 on-disk R6 folded PyTorch + R6 processors；不得以旧 F2 Orbax server 或 JAX trace冒充。
- exporter 可增加 fail-closed `--source-model-dir` 模式以避免构造/解压 14GB 中间 source ZIP；无论使用 source dir 还是 source ZIP，都必须生成逐文件 inventory/hash，拒绝 symlink、旧 FAST 和额外 norm。
- packager manifest 必须写 `candidate=r6_dual_l2_fresh500_r1`、`training_steps=500`、`topology=L2_action_lora_heads`、`normalization=QUANTILES`、`formal_winner=false`；不得出现 F2/M1 身份。
- 增加回归测试，显式断言旧 F2 model/hash/norm、`PI05Policy.from_pretrained` SLOW path 和 `strict=False` 容错加载不能进入 R6 final package。

CPU-only `py_compile`、focused tests、source hash audit 与 `git diff --check` 通过后形成 clean lock commit。任何 source/norm/processor hash 漂移立即停止。

## 2. 并行 Lane A：JAX 340/340 数值 evidence（GPU1）

本 lane 不读取或修改 FAST staging。新 namespace：

```text
data/reports/axis_libero_30k/post_r6_lora/r6_egl_statistical_release/e1_jax_numeric_recovery/
```

要求：

- 固定 `microbatch=1`、FP32、同一 frozen 340 inputs/noise/model/norm；
- 不复用或改写旧 `usable_as_full_prefix=false` shard，从 observation 0 新建原子 16-observation shards；
- 同一 fresh process 算出前 4 observations 后，立即比较旧 sentinel，执行预注册 replay 门禁：internal/external max `<=2.384185791015625e-7`、全 action sign mismatch `0`、execute8 gripper sign flip `0`、finite/provenance exact；
- 前四门禁通过后才继续 340/340；覆盖、顺序、shard/aggregate bitwise 拼接、hash、GPU0 evidence 必须完整；
- 旧 `e1_jax_mb1_canary_failed` 与旧 bitwise fields 原样保留，新报告只能写 `numeric_replay_passed_bitwise_failed_preserved`。

若 Lane A 发生 provenance/non-finite/sign failure，立即停止所有 release 工作。若仅发生 GPU1 OOM/基础设施中断，FAST critical lane可以完成诊断 ZIP，但最终状态不得高于 `diagnostic_only_not_submission_ready`。

## 3. Critical Lane B：R6 staging 与 FAST export

新建全新 staging/report namespace，禁止 `--reuse-staging` 指向旧 F2：

```text
outputs/axis_pi05/r6_same_day_fast/staging/
data/reports/axis_libero_30k/post_r6_lora/r6_same_day_fast/
```

只复制已锁定的 R6 folded model、config、pre/post processors、当前通用 policy server/requirements 与离线 vendor/tokenizer。inventory 要求无旧 F2 hash、无第二份 norm、无预存 `model_fast.safetensors`。

FAST export 必须 atomic，并通过：

```text
source model hash                      exact R6 hash
missing/unexpected/shape mismatch      0/0/0
duplicate target mapping               0
strict reload                          passed
vision patch embedding weight+bias     present, mapped, shape exact
FAST key count/dtype/bytes/hash         recorded
historical FAST asset used             false
training performed                     false
```

任何 SLOW/remap/vision warning、旧 candidate identity 或 non-strict reload 立即停止，不进入 parity/ZIP。

## 4. Z2：真实 folded PyTorch ↔ FAST parity（GPU2）

用同一批 32 个冻结 fixtures、同一 prompt/state/images、processors 与固定 noise，分别真实加载：

1. on-disk R6 folded PyTorch source；
2. 新导出的 FAST policy。

必须比较 internal 32D、external 7D、full chunk 与 execute8，并保存 per-dim/per-horizon/worst fixture。硬门禁：

```text
coverage                              32/32
token/mask/state/input mismatch       0
shape/dtype                           exact
all actions finite                    true
median external abs error             <= 0.03
p95 external abs error                <= 0.10
full-chunk gripper sign agreement     >= 0.95
execute8 gripper sign flip            0
```

旧 F2/M1 的 median `0.0526`、p95 `0.538`、gripper agreement `0.70` 必须被拒绝。禁止复制 JAX action 或只检查 key 不检查行为。

## 5. Z3：FAST-only server 与最小 closed-loop smoke

实际从 final staging 启动 policy server，日志必须明确：

```text
FAST_REQUIRED_AND_VERIFIED
fast=True
chunk=10
n_action_steps=8
action_ema=0.0
rtc=False
```

不得出现 SLOW/remap/vision/missing-key warning。执行：

- `/health`、带 seed 的 `/reset`；
- 至少 24 个冻结 `/act` 请求，shape `(7,)`、finite、nonzero，每请求 `<10s`；记录 cold/p50/p95/max 与 peak VRAM；
- C5 FAST 两个 fresh-process replay，coverage 完整、stall=0、无 HTTP timeout/server error；
- C5 与已锁定 folded source reference 不得出现超过 `1/5` success drop、collision/timeout 各自不得恶化超过 `1/5`。

V100 不是 L4。若 peak VRAM 超过 `23 GiB` 或任何 `/act >=10s`，可保留报告，但不生成 ZIP。

## 6. Z4：生成本地诊断 ZIP

只有 Z0、Lane B、Z2、Z3 全部 critical gates 通过才允许打包：

```text
artifacts/submission_axis_r6_step500_fast_diagnostic_<timestamp>.zip
artifacts/submission_axis_r6_step500_fast_diagnostic_<timestamp>.zip.sha256
artifacts/submission_axis_r6_step500_fast_diagnostic_<timestamp>.manifest.json
```

门禁：

- ZIP `<20GB`，CRC/testzip、root layout、无 symlink/hardlink/duplicate/nested archive；
- 包内唯一权重为 `model_weights/model_fast.safetensors`，严禁 `model.safetensors`；
- vision weight+bias、processor/norm/config hash 与 R6 release lock 一致；
- requirements/static validator/offline import 通过，运行时不得访问 HF/网络；
- manifest 写明 `diagnostic_only=true`、`submission_ready=false`、`official_result=false`、`formal_winner=false`、`auto_upload=false`；
- 记录 ZIP SHA-256、bytes、entry inventory、FAST hash、source hash、Git/nested commit。

ZIP 生成后不得自动上传或提交。

## 7. Z5–Z6：有余时才提升 release 等级

诊断 ZIP 已安全落盘后，若仍有时间：

1. 用 GPU3 对相同 340 frozen observations 运行真实 on-disk folded PyTorch source；
2. 完成 JAX↔PyTorch 与 170-pair EGL response comparator，沿用原预注册 `1e-4` EGL review upper bound、source parity `1e-3`、execute8 gripper sign flip `0`；
3. 运行 C20 FAST 与 folded source paired non-regression：FAST success drop不超过 `1/20`，collision/timeout各不恶化超过 `2/20`，stall/server error为0；
4. 全部通过才把状态提升为 `r6_local_release_candidate_pending_official_precheck`，但不得修改已生成 ZIP 内的 `submission_ready=false`；如需 release-candidate ZIP，必须重新构建为新文件并保留两个 hash。

官方 Blackwell Docker、L4 24GB 与上传仍需用户另行授权，不在本计划执行。

## 8. 最终验证与停止

```text
.venv_axis_pi05/bin/python -m py_compile tools/axis_pi05/*.py pipeline/*.py
.venv_axis_pi05/bin/pytest -q tests
all JSON/JSONL/hash/count checks
git diff --check
root/nested clean commit
```

报告必须逐项区分 `observed`、`inference`、`not_run`，明确今天得到的是哪一档交付。完成或任一 critical gate 失败后 hard stop，等待审核。
