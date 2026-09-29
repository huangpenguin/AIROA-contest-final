# AXIS π0.5：T1 canary 收口后今晚官方 Track3 测试训练计划（2026-09-11）

> 创建日期：2026-09-11（Asia/Tokyo）
> 授权来源：用户确认 T1-P20 canary 已出现真实 optimizer step，并要求在同一官方 Blackwell session 上联合测试 Track3；Track3 已通过 Hugging Face 同步，尚未开训、未占 GPU。
> 计划性质：日常轻量迭代执行合同。不是 formal audit，不改 plan index 的唯一活动采集计划。
> 并行边界：`docs/AXIS_PI05_TRACK3_FAST_TELEOP_400_AND_SHORT_TRAIN_PLAN.md` 仍是仓库唯一活动采集计划。本文件不改其 cursor / manifest / 采集队列。
> 当前阶段：`JT0d_serial_rebuild_then_t1_t3_t2`
> 2026-09-11 11:28 清点：data-pull 只找回 27 MiB 旧 probe，没有 T1 checkpoint。`~/dataset` tar 仍在。禁止三赛道同时训。

本文不把自身提升为仓库唯一活动计划。T1/T2 合同仍是 `docs/AXIS_PI05_TRACK1_TRACK2_OFFICIAL_BLACKWELL_LORA_PLAN_20260910.md`。Track3 官方 recipe 仍以 `docs/AXIS_PI05_TRACK3_POST_Q5_SEMANTIC_SIGNOFF_AND_OFFICIAL_STEP100_PLAN_20260910.md` 的 N4 为准。

## 0. 对用户日志的结论

**已经真正跑起来了。** 17:19 的片段不是编译卡死，而是过了 restore、过了第一步 backward，并进入连续 optimizer update：

- PID 约 `27043`，tqdm `5.00it/100it`，约 `13.1s/it`。
- `optimizer_step=5` 随后 `optimizer_step=6`，`examples_seen=64`，`micro_steps=16`。
- `loss≈0.83–0.88`，`nonfinite_*_count=0`，`post_clip_grad_norm=0.1000`。

这就是 T1 canary 的成功点火标准。当时剩余约 21 分钟即可到 step 100。本文件写于 2026-09-11 02:20 JST，距该日志约 9 小时，因此 **今晚第一件事是现场确认 T1 结局，而不是假设它仍在 step 5**。

官方机只有 1 张 RTX PRO 6000 Blackwell。**“联合测试”=同一 session 串行，不是两套 `train.py` 同时占 GPU。**

## 1. 冻结身份

### 1.1 Hugging Face Track3 payload（已上传，禁止重传）

```text
repo           Huangpenguin/parc2026-track3-q7-transfer   (private dataset)
commit         669789a2aee4647bae9f8030e2a62b7fcd64774a
payload        253 files / 14,165,000 bytes
roundtrip      253/253
Q7 manifest    42cb9194819aef424afe0f374d9d0c44fb92920dface9ff472f9cb99781c05dc
receipt        data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q7_gpu3_frozen_eval/q7_hf_upload_receipt.json
Q7 status      ready_for_official_transfer_after_track12_release
official_training_started  false
E2 paired      passed / 48 cases
```

### 1.2 不进这份 payload、继续用现有副本

```text
R6 params      Huangpenguin/parc2026-r6-l2-params-step500 @ e788d92131c984f4b4f7914e0b90b244ad014e4c
R6 tree SHA    d9b7e0d15781060fccfd61dc1890ad98993f0224e4aebba1dd6658d7c011931f
官方现有路径   /opt/dlami/nvme/ckpts/r6/params
libero_plus    lerobot/libero_plus @ f3f49f426d75030177b18778374005bc12ccd588
官方现有路径   /opt/dlami/nvme/dataset/lerobot/libero_plus
Q5 step-10     诊断产物；未上传；禁止作官方/今晚初始化
```

### 1.3 今晚 T3 测试训练 recipe

与 Q5 / N4 相同的训练合同，只把 Blackwell 运行时改成 T1 已验证栈：

```text
config                     axis_pi05_track3_q5
init                       R6 step500 params-only，fresh AdamW
topology                   L2_action_lora_heads
precision                  FP32
peak LR                    1e-6
warmup                     100
clip                       0.1
EMA                        off
global batch               64
microbatch                 4
accumulation               16
OPENPI_KEEP_STEPS          0,1,10,25,50,100
fail on nonfinite          true
run dir                    /opt/dlami/nvme/runs/t3_four_gold_canary100
train log                  /opt/dlami/nvme/runs/t3_four_gold_canary100.train.log
mix                        four_gold_80_20_v1，forward 0.8 / reverse 0.2
norm SHA                   faf3f20f04756c8f85eeea7b72b893cd6a712d37d91cb5c0144004323cc994ab
mix SHA                    cc6c7206458277512a4f8a33a7e43a49044d8d845cea796f70f1fc292b1d8244
sampling SHA               2276a6bf2fa5b5741cb11f69c3076a865b7aa4b6ff7800baffa40897f8bd1cb9
```

明确偏离：

| 项 | Q5 原文 / 官方 runner | 今晚测试 | 原因 |
|---|---|---|---|
| microbatch | 1 / acc 64 | **4 / 16** | T1 已在同一卡证明 mb=4 稳定；mb=1 会把 100 步拉到约 80 分钟 |
| keep steps | 官方 runner `0,1,100` | **0,1,10,25,50,100** | 12h session，必须可从中间步恢复 |
| 启动方式 | `run_track3_official_blackwell_lora.py` 直接 mkdir+train | **T1 同款 notebook `Popen`，禁止预先 `mkdir RUN`** | 空 RUN 会 `FileExistsError` |
| T12 门 | 完整 T1+T2 release | **T1 canary 100 正常结束即允许今晚 T3 测试** | 完整 T2 100 + C20 今晚放不下 |
| OpenPI 源 | Q7 payload 的 `train.py`/`data_loader.py` | **保留 T1 官方机已打补丁的 live tree** | Q7 那三份文件没有 JAX 0.11 Auto-mesh / v3 IPEC / unroll / gather 修复 |

OOM 时只把 microbatch 降到 2 或 1，并用 accumulation 钉住 global 64。不改 LR、不切 BF16、不缩 LoRA、不换 mix。

## 2. 今晚 GPU 与时间门

官方实例 1 GPU、Notebook 最长约 12 小时、idle 1 小时停机。不要点延長，不要 Stop My Server。

```text
现在允许          CPU snapshot_download 到 staging；读 T1 日志；写本计划证据
现在禁止          第二套 train.py；覆盖 $CODE；rm -rf T1 RUN；覆盖 R6 / libero_plus
T3 启动 GPU 门    无 train.py / 无 JAX 占卡；T1 RUN 已自然结束或已确认 crash 且不再恢复今晚 T1
T3 启动时间门     剩余 wall-clock ≥ 90 分钟（compile + step1 + 100 步按 13s/it ≈ 25 分钟 + restore + data-push）
不够 90 分钟      只做 staging 校验与落位准备，不开 T3
T2 今晚           默认不做。仅当 T3 step100 已通过且剩余 ≥ 4 小时才考虑 T2-P30 canary
评测今晚          默认不做 C20 / reverse holdout。若 T3 100 后仍剩 ≥ 3 小时，最多 reverse 4-case；C20 下一 session
```

今晚所谓 “Track1/2 release proof” **只用于释放 T3 测试训练**，不是 T12 ZIP / 提交证明：

```json
{
  "schema_version": "t1_p20_canary100_gpu_release_proof_v1",
  "status": "passed",
  "active": false,
  "training_active": false,
  "run_dir": "/opt/dlami/nvme/runs/t1_p20_canary100",
  "final_step": 100,
  "note": "canary-complete GPU release for Track3 test train; not a Track1/2 ZIP release"
}
```

该 JSON 写入 `/opt/dlami/nvme/runs/t1_p20_canary100.release_proof.json`。不要把它当成 T12_A6 行为评测通过。

## 3. 执行阶段

### JT0 — 现场确认 T1，并允许 CPU staging

在官方 notebook 只读检查，顺序固定：

1. `nvidia-smi`：是否仍有 PID `27043` 或其它 `train.py` / python 占卡。
2. 读 `/opt/dlami/nvme/runs/t1_p20_canary100.train.log` 尾部：是否 `100/100`、exit 0、还是 traceback。
3. 列 `/opt/dlami/nvme/runs/t1_p20_canary100/`：是否有 checkpoint `0,1,10,25,50,100`。
4. 若存在 `training_metrics.jsonl`，确认 100 行 finite、无 NaN。
5. 看 Jupyter 状态栏剩余时间。

分支：

- **T1 仍在跑**：不要杀。只允许 JT0 的 staging 下载。禁止 overlay，禁止第二训练。
- **T1 已 100 且进程已退出**：进入 JT1。
- **T1 crash / 未到 100**：保留 RUN 与 log，报告最后完整 checkpoint。今晚优先从 last kept step 把 T1 跑完，再开 T3。不要用半截 T1 当 release proof。

**现在就可做、且不占 GPU 的 staging：**

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Huangpenguin/parc2026-track3-q7-transfer",
    repo_type="dataset",
    revision="669789a2aee4647bae9f8030e2a62b7fcd64774a",
    local_dir="/opt/dlami/nvme/staging/track3_q7",
)
```

下载后只校验，不 copy 进 `$CODE`：

```text
文件数              payload/ 下 253
总字节              14,165,000
Q7 manifest SHA     42cb9194819aef424afe0f374d9d0c44fb92920dface9ff472f9cb99781c05dc
禁止覆盖            $CODE/third_party/axis_openpi/**
禁止覆盖            /opt/dlami/nvme/ckpts/r6/params
禁止覆盖            /opt/dlami/nvme/dataset/lerobot/libero_plus
禁止覆盖            /opt/dlami/nvme/runs/t1_p20_canary100
```

若 T1 仍在写盘，staging 下载应错开高峰；14 MB 通常可接受，但不要 `parc-home-sync` 大目录。

### JT1 — T1 canary closeout（GPU 释放证明）

仅当 trainer 已退出：

1. 不要 `rm -rf` T1 RUN。
2. `parc-home-sync data-push`：至少 push T1 log、metrics、checkpoint 清单、本 release proof。不要等 ZIP。
3. 写上一节的 `t1_p20_canary100.release_proof.json`。
4. 再查一次 `nvidia-smi`：compute apps 必须空。
5. 确认 T1 官方机上的 JAX 0.11 补丁仍在 live `$CODE`：

```text
sharding.py        make_mesh 使用 AxisType.Auto
train.py           device_count()==1 时 data_sharding = replicated
gemma.py           embedding gather 使用 NamedSharding
model.py           augmax unroll；StepMetadata unwrap
array_typing.py    跳过 jax._src* 与 flax.nnx.transforms*
data_loader.py     IPEC v3 AxisPlusV3Dataset 仍在（T1 数据依赖它；T3 正向也走 libero_plus v3）
```

这些补丁是 T3 能 compile 的前提。JT2 落位时 **不得** 用 Q7 payload 里的三份 OpenPI 文件覆盖它们。

### JT2 — 选择性落位，禁止整包 overlay

T1 释放 GPU 后，把 staging 里的 **数据与合同文件** 拷到约定路径，不要 `cp -a staging/track3_q7/payload/ $CODE/`。

允许落位：

```text
data/processed/axis_pi05_track3_reverse_gold_v1/
data/manifests_track3/four_gold_80_20_v1/
configs/axis_pi05/track3_four_gold_mix_v1_20260910.yaml
data/eval_manifests/axis_pi05_eval_v1/public_t3_holdout.jsonl
data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/   （证据/JSON，不覆盖 T1 RUN）
tools/axis_pi05/run_track3_official_blackwell_lora.py            （若官方机还没有）
```

禁止落位或覆盖：

```text
third_party/axis_openpi/scripts/train.py
third_party/axis_openpi/src/openpi/training/config.py
third_party/axis_openpi/src/openpi/training/data_loader.py
r6_step500_params/
outputs/axis_pi05/track3_four_gold_mix_v1/q5_local_gpu3/         （即使 payload 里没有完整 ckpt 也不当 init）
data/manifests_track12/                                         （T1/T2 mix）
```

落位后立刻重算：

```text
norm_stats.json              == faf3f20f04756c8f85eeea7b72b893cd6a712d37d91cb5c0144004323cc994ab
sampling_weights.parquet      == 2276a6bf2fa5b5741cb11f69c3076a865b7aa4b6ff7800baffa40897f8bd1cb9
manifest.json                == cc6c7206458277512a4f8a33a7e43a49044d8d845cea796f70f1fc292b1d8244
Q7 manifest + sidecar        == 42cb9194819aef424afe0f374d9d0c44fb92920dface9ff472f9cb99781c05dc
R6 params tree               == d9b7e0d15781060fccfd61dc1890ad98993f0224e4aebba1dd6658d7c011931f
libero_plus revision         核 meta / git revision，缺则再下；已有则禁止覆盖
```

`lerobot/libero_plus` 合同仍写“官方机单独下载”，但 T1 已经在用 NVMe 副本。JT2 只核 revision/hash，不盲目 `snapshot_download` 覆盖。

### JT3 — Track3 preflight（仍不占训练图）

用官方 Python：

```text
/opt/dlami/nvme/work/trainvenv/bin/python
CODE=/opt/dlami/nvme/code/PARC2026_final
PYTHONPATH=$CODE/third_party/axis_openpi/src:$CODE
```

检查：

1. GPU 空、UUID 仍是同一张 Blackwell。
2. 正向根：`/opt/dlami/nvme/dataset/lerobot/libero_plus`
3. 逆向根：`$CODE/data/processed/axis_pi05_track3_reverse_gold_v1`（4 eps / 1060 frames）
4. `axis_pi05_track3_q5` 可 `get_config`
5. 不要预建 `/opt/dlami/nvme/runs/t3_four_gold_canary100`
6. 日志路径必须在 RUN **外**
7. 官方 runner 默认 `PYTHON=.venv_axis_pi05`、默认 `params` 在 `$CODE/outputs/...`。今晚 **不要直接跑它的 `step1`**，除非先改：`AXIS_PI05_PYTHON`、`--params /opt/dlami/nvme/ckpts/r6/params`、禁止 mkdir 空 RUN、且 hasher 能处理 ROOT 外的 params。缺这些补丁时，走 JT4 的 T1 同款启动。

可选：runner `preflight` 只读核对 hash。它若因 ROOT-relative R6 hasher 失败，记下原因，改用手动 SHA 检查，不要为此重下 12 GiB R6。

### JT4 — Track3 step-1 点火，同一 run 拉到 100

环境骨架（T1 已验证项必须保留）：

```text
CUDA_VISIBLE_DEVICES=0
JAX_PLATFORMS=cuda
XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_MEM_FRACTION=.90
HF_HUB_OFFLINE=1
PYTHONUNBUFFERED=1
AXIS_PI05_DISABLE_ACTIVATION_SHARDING_CONSTRAINT=1
AXIS_PI05_LORA_ARM=L2_action_lora_heads
AXIS_PI05_TRACK3_PARAMS=/opt/dlami/nvme/ckpts/r6/params
AXIS_PI05_TRACK3_NORM_STATS=$CODE/data/manifests_track3/four_gold_80_20_v1/norm_stats.json
AXIS_PI05_TRACK3_SAMPLING_MANIFEST=$CODE/data/manifests_track3/four_gold_80_20_v1/sampling_weights.parquet
AXIS_PI05_TRACK3_LOCAL_REPO_IDS=lerobot/libero_plus,axis-pi05/track3_reverse_gold_v1
AXIS_PI05_TRACK3_LOCAL_REPO_ROOTS=/opt/dlami/nvme/dataset/lerobot/libero_plus:$CODE/data/processed/axis_pi05_track3_reverse_gold_v1
AXIS_PI05_TRACK3_SOURCE_WEIGHTS=0.8,0.2
AXIS_PI05_TRACK3_GLOBAL_BATCH=64
AXIS_PI05_TRACK3_MICROBATCH=4
AXIS_PI05_TRACK3_ACCUMULATION=16
AXIS_PI05_TRACK3_PEAK_LR=1e-6
AXIS_PI05_TRACK3_WARMUP_STEPS=100
AXIS_PI05_TRACK3_RUN_DIR=/opt/dlami/nvme/runs/t3_four_gold_canary100
OPENPI_KEEP_STEPS=0,1,10,25,50,100
AXIS_PI05_FAIL_ON_NONFINITE=1
```

启动方式与 T1 相同：notebook 里 `os.environ` + `Popen(..., start_new_session=True)`，stdout 接到 RUN 外 log。官方 xterm 不要贴超长一行。Cursor 远程点 Run 不可靠，用户本地 Shift+Enter。

成功点火标准（对标 T1 step 5/6）：

- 出现 `optimizer_step=1` 且 `loss` finite
- `examples_seen=64`，`micro_steps=16`
- `post_clip_grad_norm=0.1000`
- `nonfinite_*_count=0`
- 不要在 step1 另开新 RUN；确认 step 0/1 保存后，**同一进程或同一 RUN resume** 到 100

100 步通过标准：

- `100/100`，trainer 退出码 0
- checkpoint `0,1,10,25,50,100` 完整，无 partial
- metrics 连续 100 行 finite
- 新进程能 restore step 100
- 未改 LR / mix / norm / seed / global batch

中途 12h 将尽：`parc-home-sync data-push` 后停在 last kept step，下一 session resume **同一 RUN**。禁止第二实例。

### JT5 — 今晚收口，不自动评测、不自动 T2

T3 100 通过后立即：

1. `parc-home-sync data-push` T3 log、metrics、checkpoint 清单。
2. 写 `/opt/dlami/nvme/runs/t3_four_gold_canary100.summary.json`：status、final_step、SHA、microbatch、LR。
3. 明确标记 `official_result=false`。这是官方机测试训练，不是提交。
4. T2-P30、T1 C20、T3 reverse/C20 N5 **默认下一 session**。
5. 不要 Stop My Server，除非用户明确要求结束计费。

## 4. 明确禁止

- 第二套 `train.py` / 第二张逻辑 GPU。
- 用 Q5 step-10 或任何 Q5 Orbax 当 init。
- 用官方 `pi05_step005000_submission_py310.zip` 冒充 R6。
- 整包 overlay Q7 payload，尤其覆盖 T1 已修复的 OpenPI。
- 预 `mkdir` 空的 T3 RUN；log 写进 RUN 内。
- `uv pip install -e third_party/axis_openpi`（会把 JAX 打回 0.5.3/cuda12）。
- 改 Fast400 cursor/manifest；碰本地 GPU0 YOLO。
- 自动 fold / ZIP / `omni submit`。
- 把 T1 的 `3e-7` / warmup 10 误用到 T3；也把 T3 的 `1e-6` / warmup 100 改回 T1。
- 把今晚 T3 100 直接宣布为 N5 通过。

## 5. 失败与恢复

| 现象 | 动作 |
|---|---|
| T1 仍在跑 | 观察；只做 staging |
| T1 crash 且未到 100 | 先修/续 T1，不开 T3 |
| staging hash 不匹配 | 停止 overlay，对照 commit `669789a2...` 重下，不手工补文件 |
| T3 compile 再爆 sharding | 先确认 Auto mesh / 1-GPU replicated / unroll 仍在；不要回退 Q7 data_loader |
| T3 OOM | mb 4→2→1，acc 16→32→64；换新 RUN ID 或同一 RUN 仅当尚未写出 step1 |
| NaN/Inf | 停在最近完整 checkpoint；不改 LR 续同一失败图 |
| session 不足 90 分钟 | 只落位，不开训 |
| 官方缺 simulator | 评测拉回本地 GPU3；永不碰 GPU0 |

## 6. 下一位执行者的最短路线

今晚已完成并冻结：

- T1 canary 100 已同步保存；进程已退出；GPU `0 MiB / 0%`。
- checkpoint 现存 `10, 25, 50, 100`（各约 12.6 GiB）。`0`/`1` 缺失，不阻塞明天 T3 测试。
- staging：`/opt/dlami/nvme/staging/track3_q7`，259 文件 / 14,411,814 bytes；payload **253 / 14,165,000**；`has_q7_manifest True`。
- 未 overlay、未开 T3、未 data-push 那 4 个大 checkpoint。

明天从 **JT2 选择性落位** 开始，不要重做 T1，不要整包 overlay。

1. 确认 GPU 空、无 `train.py`、剩余时间 ≥ 90 分钟。
2. 若 staging 仍在且 payload 仍是 253/14165000：直接 JT2。若 NVMe 被回收：按同一 commit 重下 14 MB。
3. 选择性落位（禁止覆盖 `third_party/axis_openpi`、R6、libero_plus、T1 RUN）。
4. SHA 核对后用 T1 同款 Popen 启动 T3 100-step；log 在 RUN 外。
5. 100 通过后 data-push 并 hard stop；评测与 T2 另开。

当前不应执行：T2 开训、N5 评测、ZIP、提交、覆盖 Fast400、整包 overlay、预先 `mkdir` T3 RUN。

重启后先跑第 8 节 inventory，再决定是 resume 还是重建 NVMe。禁止在未清点前开第二套训练。

## 8. 2026-09-11 11:22 重启后的盘符事实

官方资料（`docs/COMPETITION_GPU_GUIDE_SUMMARY_ZH.md`）本来就把盘写死了：

| 路径 | 容量 | 停机后 |
|---|---|---|
| `$HOME` | 40 GiB | 持久。每 10 分钟及实例结束时保存 |
| `~/data` | 100 GiB | **必须** `parc-home-sync data-push` 才保存；新 session `data-pull` |
| `~/dataset` | 只读 | 配布 tar / baseline，一直在 |
| `/opt/dlami/nvme` | ~1.7 TiB | **临时。停止/结束实例会丢** |

昨晚实际落点：

```text
T1 RUN / log / ckpt     /opt/dlami/nvme/runs/t1_p20_canary100*     临时，未 push
T3 staging              /opt/dlami/nvme/staging/track3_q7          临时，未 push
R6 params               /opt/dlami/nvme/ckpts/r6/params          临时
libero_plus / IPEC      /opt/dlami/nvme/dataset/...                临时
trainvenv / 代码补丁     /opt/dlami/nvme/work 与 /opt/dlami/nvme/code 临时
```

T1 训练**做过了**（step 100 同步保存，日志在聊天里），但权重很可能已随 NVMe 消失。T3 payload 仍在 Hugging Face，14 MB 可重下。R6 仍在私有仓库。T1 100 步约 13 分钟，可重跑；贵的是 venv、JAX 补丁、R6、正向数据解压。

重启后硬规则：任何完整 canary 的 **log + metrics + 选定 step 的 checkpoint** 必须立刻拷到 `~/data` 并 `parc-home-sync data-push`。不要把 4×12 GiB 都推进 100 GiB 盘；T1 只留 step 100（必要时再留 50）。

## 7. 2026-09-11 02:43 暂停点

用户决定明天继续。官方 Jupyter 上的命令仍需用户 Shift+Enter；agent 不能可靠地点 Run。

离开前建议：

- **不要**点延長。
- **不要**开始 T3 overlay 或训练。
- 若要结束计费：先确认你能接受 NVMe（T1 RUN、staging、可能还有 R6/libero_plus）被回收；staging 14 MB 可重下，T1 100 步约 13 分钟可重跑，R6/libero_plus 重下更贵。
- 若 session 还要挂着：注意 idle 约 1 小时可能停机；不要假设进程还在值守。
- 不要 `rm -rf /opt/dlami/nvme/runs/t1_p20_canary100`。
