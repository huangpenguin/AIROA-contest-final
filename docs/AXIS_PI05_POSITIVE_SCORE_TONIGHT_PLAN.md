# AXIS Pi0.5：今晚正分优先 FAST/ZIP 最短执行计划

> 制定时间：2026-08-31 18:11（Asia/Tokyo）
> 输入 commit：`58d3fcc46cfa1eaffd5d057be59edf97b03f02eb`
> 当前状态：`r6_positive_score_fast_path_ready`
> 唯一目标：尽快生成一个真正走 R6 FAST、无视觉 key 错误、通过最小行为回归的 Track 1 ZIP，供用户人工提交争取 `>0`。
> 时间目标：本地 ZIP 尽量在 21:30–22:30 前落盘；失败时输出准确 blocker，不转去训练或旧包。
> 禁止：JAX 340、完整 E1 envelope、C20/V160/H160、任何训练、旧 F2 FAST、SLOW fallback、比赛 GPU、自动 push/upload/submit。

## 0. 今晚只保留这一条链

```text
P0 R6 release lock + 去 F2/M1 硬编码
  → P1 R6 folded PyTorch → model_fast.safetensors
  → P2 8 fixtures × 2 seeds 的 source↔FAST parity
  → P3 FAST-only server 12 acts + locked C5
  → P4 ZIP/static/offline checks
  → STOP，交给用户人工确认/提交
```

不要执行或等待 JAX numeric 340；它移回赛后/次日的完整 release 审核。今晚也不运行额外 full-FT。

## 1. P0：15–30 分钟 release lock

执行前记录 HEAD、root/nested clean、GPU0 外部 PID、GPU1–3 状态。只允许使用 GPU1/2；GPU0 不得触碰。

冻结唯一候选：

```text
candidate                   r6_dual_l2_fresh500_r1
training/topology           LoRA L2_action_lora_heads, step 500
folded model                outputs/axis_pi05/axis_libero_lora_folded/r6_dual_l2_fresh500_r1_step500/model.safetensors
model sha256                5c73e1e8217dbb75a934c2b0165208e851f69e66cfe2ea0430590a29901e6d5b
raw tree sha256             d203635f1eb0b006ec27841db6ce5171f74a37035eca4291c8107c05838c276f
norm sha256                 4c803bf59c52677c36b6b420c614d8a7742e0847708443863e13e67cb80b18f5
normalization               ACTION/STATE QUANTILES
chunk / execute             10 / 8
action_ema / rtc            0.0 / false
local B80                   49/80, not statistically significant
```

最小修改已有 exporter/auditor：

- 删除/参数化 `F2`、`M1`、旧 norm、旧 checkpoint 硬编码；
- authoritative source 必须是上述 on-disk R6 folded PyTorch + 其 pre/post processors；
- manifest 必须写 R6 identity、QUANTILES、step500、`formal_winner=false`；
- final server 禁止 `PI05Policy.from_pretrained`、SLOW/remap fallback；
- 增加 targeted tests，证明旧 F2 hash/norm 不能进入 R6 staging。

P0 只跑 targeted tests、`py_compile`、JSON/hash 与 `git diff --check`，通过后立即进入 P1，不等待人工审核。

## 2. P1：30–60 分钟生成 R6 FAST

使用新目录，禁止复用旧 staging：

```text
outputs/axis_pi05/r6_positive_score_fast/staging/
data/reports/axis_libero_30k/post_r6_lora/r6_positive_score_fast/
```

从 locked R6 source 生成 atomic `model_fast.safetensors`。硬门禁：

```text
source model/norm/processor hash             exact
missing/unexpected/shape mismatch            0/0/0
duplicate target mapping                     0
strict reload                                passed
vision patch embedding weight + bias         present/mapped/shape exact
historical FAST used                         false
training performed                           false
FAST hash/key count/dtype/bytes               recorded
```

禁止为了赶时间忽略 warning、使用 `strict=False` 结果、复制旧 9.3GB FAST 或保留 SLOW 权重。P1 失败则今晚 hard stop：转换链本身未解决，短训练没有意义。

## 3. P2：20–35 分钟最小行为 parity

固定 8 个已有、未针对官方隐藏题选择的 Track 1 fixtures，每个使用 2 个固定 policy seeds，共 16 comparisons。真实加载：

1. on-disk R6 folded PyTorch source；
2. 新 R6 FAST。

比较 internal 32D、external 7D、full chunk/execute8。今晚压缩门禁：

```text
coverage                              16/16
input/token/mask/state mismatch       0
all action finite                     true
median external abs error             <= 0.03
p95 external abs error                <= 0.10
execute8 gripper sign flip            0
```

另存 max、p99、per-dim/per-horizon，不能只写 pass。旧 F2 的 `median 0.0526 / p95 0.538 / gripper agreement 0.70` 必须失败。P2 失败不打包。

## 4. P3：30–60 分钟 FAST server 与正分代理

实际从 final staging 启动 server。日志必须同时包含：

```text
FAST_REQUIRED_AND_VERIFIED
fast=True
chunk=10
n_action_steps=8
action_ema=0.0
rtc=False
```

不得出现 SLOW/remap/vision/missing-key warning。

### 4.1 请求门禁

- `/health` 和带 seed 的 `/reset` 通过；
- 12 个 frozen `/act` 请求，输出 `(7,)`、finite、nonzero；
- 每个请求 `<10s`，记录 cold/p50/p95/max 和 peak VRAM；
- GPU0 untouched，server 退出后显存释放。

任何 `/act >=10s`、HTTP 500、NaN、错误 shape 或 server crash 立即停止。

### 4.2 Locked C5

在读取 FAST rollout 前，从既有 seeded B80 报告锁定 5 cases：

- 3 个不同 task 的 R6 source-success regression cases；
- 2 个不同 failure type 的 diagnostic cases；
- 将 case IDs、task/instruction、init/env/policy seed、旧 source outcome 与 manifest SHA 在 FAST rollout 前写入 clean lock commit。

这不是新的性能估计，而是部署回归检查。FAST 必须：

```text
coverage                              5/5
source-success cases reproduced      >=2/3
HTTP/server/non-finite/stall          0
```

若 3 个 source-success case 全部失败，禁止打包；这最接近上一版 0/6 灾难回归。

## 5. P4：20–40 分钟打包

只有 P0–P3 全部通过才生成：

```text
artifacts/submission_axis_r6_step500_fast_positive_score_<timestamp>.zip
artifacts/submission_axis_r6_step500_fast_positive_score_<timestamp>.zip.sha256
artifacts/submission_axis_r6_step500_fast_positive_score_<timestamp>.manifest.json
```

要求：

- `<20GB`、ZIP CRC/testzip、root layout、无 duplicate/symlink/nested archive；
- 包内只有唯一 `model_weights/model_fast.safetensors`，不得含 `model.safetensors`；
- policy server/FAST/config/processors/norm hash 与 P0–P3 exact；
- static validator、offline import、一次 unpacked `/health/reset/act` 通过；
- manifest 明确 `track=track1`、`fast=True`、R6 identity、`official_result=false`、`auto_submit=false`；
- 写出 SHA-256、bytes、entry inventory 与人工提交检查清单。

本计划最多给出：

```text
r6_positive_score_zip_ready_for_human_decision
```

它表示“比上一份 0 分包更可信，值得人工决定是否使用当天 Track 1 次数”，不保证官方一定 `>0`。

## 6. 人工提交边界

执行者不得上传或调用 `omni submit`。完成后必须把以下内容集中在最终报告首页：

- ZIP 绝对路径、SHA-256、bytes；
- `fast=True` server 日志证据；
- vision weight/bias 与 strict key gate；
- 16-case parity；
- 12-request latency；
- locked C5 的逐 case outcome；
- 当天提交前仍需用户确认的风险。

用户确认后再单独执行官方上传/提交；官方结果返回后保留完整日志，不根据分数删除失败产物。

## 7. Full-FT 后续，不在今晚执行

今晚 ZIP 完成或明确失败后，另建训练计划：

```text
Axis init
  → standard LIBERO 10 Hz full-FT 1K smoke
  → 3K gate
  → 10K
  → 30K reproduction target
  → LIBERO-plus / Track 3 20 Hz branch + new norm
```

比较 full-FT、action-expert dense 与高覆盖 LoRA；每个 checkpoint 都必须走同一 FAST/server/ZIP 门禁。不得用今晚的紧急打包结果替代长期训练研究。

## 8. 最终停止条件

成功或失败均执行：

```text
.venv_axis_pi05/bin/python -m py_compile tools/axis_pi05/*.py pipeline/*.py
.venv_axis_pi05/bin/pytest -q tests
JSON/hash/count checks
git diff --check
root/nested clean commit
```

形成 clean commit 后 hard stop，等待人工审核与提交决定。
