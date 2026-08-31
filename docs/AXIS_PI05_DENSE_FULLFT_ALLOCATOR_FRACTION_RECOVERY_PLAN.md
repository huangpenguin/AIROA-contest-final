# AXIS Pi0.5 Dense Full-FT Allocator Fraction 无损复测计划

> 制定日期：2026-09-01（Asia/Tokyo）
> 审核输入：root `6239704752dc2966d23b0f6c9d49744efdb152ef`，nested OpenPI `92604d6d07d5bbf636827ee903f758538c8ba015`
> 当前状态：`dense_fullft_allocator_fraction_recovery_ready`
> 唯一目标：保持训练数学配方完全不变，先排除 JAX 默认 75% BFC pool 对 4×V100 first update 的人为容量限制，最多验证到连续 10 optimizer steps。

## 0. 不可变合同

| 字段 | 固定值 |
|---|---|
| initialization | Axis released params revision `c342608c489515efa9ba8281400dc648bf3970e1` |
| data | `physical-intelligence/libero` native 10Hz，1693 episodes |
| norm | fresh per-dimension QUANTILES，SHA-256 `4c803bf5…b18f5` |
| topology | dense full model，`nnx.Nothing`，LoRA leaf `0` |
| precision | FP32；BF16/FP16 禁止 |
| optimizer | AdamW β1=.9、β2=.95、eps=1e-8、wd=1e-10，global clip=1.0 |
| EMA | `0.999` |
| per-device microbatch | `1` |
| devices / accumulation | 4 / `16` |
| effective global batch | exact `64` |
| model remat/sharding | 保持 nested OpenPI `92604d6` 实现，本轮不改 |

唯一自变量是 JAX preallocated pool fraction。不得同时修改 dtype、batch、EMA、optimizer、freeze scope、remat、image/token shape、data/norm 或 sharding。

本轮不启动比赛 GPU，不跑 C20/50/100/1K，不 export FAST/生成 ZIP/上传/提交。

## A0. 证据与 step-0 checkpoint 锁（CPU only）

在新 namespace 建立：

```text
data/reports/axis_libero_30k/dense_fullft_allocator_fraction_recovery/
outputs/axis_pi05/dense_fullft_allocator_fraction_recovery/
```

1. 保留旧 `dense_fullft_sharded_init_recovery/` 报告/日志/hash immutable；
2. 锁定 JAX/JAXLIB `0.5.3`、4×V100 总显存 `32768 MiB/card`、R4 allocator env unset、NVML peak `24897 MiB/card`、OOM request `143327232 bytes/card`；
3. 计算并写入 `24897/32768`、75% pool `24576 MiB`、pool 外显存 `7871 MiB`和推定 context/driver overhead；
4. 锁定现有 step-0 checkpoint `outputs/axis_pi05/dense_fullft_sharded_init_recovery/train_10/0`的 tree/metadata hash、字节数、internal step、params/Adam/EMA shape/dtype；
5. 只有 checkpoint 完整、step `0`、recipe/hash 与本计划一致才允许 restore；否则 hard stop，不自动 fresh-init 绕过。

旧 summary 中 `dense_fullft_first_update_true_compute_oom_pending_official_gpu_review` 作为历史字段保留，不原地修改；新审核将它重分类为 `default_75pct_allocator_pool_oom_requires_fraction_retest`。

## A1. Allocator 可观测性（CPU/code only）

在不改 train-step math/HLO 的前提下，对启动/恢复/首个 microbatch/累积/update/checkpoint 记录：

- `XLA_CLIENT_MEM_FRACTION`、`XLA_PYTHON_CLIENT_MEM_FRACTION`、`XLA_PYTHON_CLIENT_PREALLOCATE`、allocator 类型与全部相关 XLA flags；
- `jax.devices()[i].memory_stats()` 所有可用 keys，至少提取 `bytes_in_use`、`peak_bytes_in_use`、`pool_bytes`/`bytes_limit`等当前版本实际提供的字段；
- 编译后 executable `memory_analysis()`（若 JAX `0.5.3` 可用）的 argument/output/temp/alias bytes；
- NVML 每 `<=0.25s` 采样的 process/device memory、GPU utilization 和外部 PID；
- OOM 异常时仍 flush 完成的 allocator/NVML/log/phase evidence。

新增单测要求：不得同时设新/旧 fraction 变量；fraction 必须是预注册的 `0.90` 或 `0.95`；preallocation 必须启用；禁止 `platform`、`vmm`、`address`、`cuda_malloc_async`；启动命令和 summary 的 fraction 必须一致。

运行定向测试、`py_compile`、全量 `.venv_axis_pi05/bin/pytest -q tests`、JSON parse 和 `git diff --check`。

## A2. 0.90 唯一 first-update canary

只有 4 张 GPU 都自然空闲、无外部 PID 且可独占时才运行。明确设置：

```text
CUDA_VISIBLE_DEVICES=0,1,2,3
XLA_CLIENT_MEM_FRACTION=0.90
XLA_PYTHON_CLIENT_PREALLOCATE=true
```

不设置 deprecated `XLA_PYTHON_CLIENT_MEM_FRACTION`，不设置其他 allocator。从 A0 锁定的 step-0 checkpoint fresh-process restore，验证 step、params sentinel、optimizer、EMA、RNG、scheduler 全部一致且 finite，然后最多完成一次 global-batch-64 update。

通过条件：

- 16 个 microbatch loss/grad finite，只在第 16 个后 clip/update/EMA/scheduler 各前进一次；
- optimizer step `1`、examples seen `64`，全 trainable scope 有有限 grad/update；
- checkpoint step `1` atomic save，fresh-process restore step `1` exact/finite；
- internal allocator peak、pool limit、largest-free/fragmentation 可用证据与 NVML peak 完整；
- FP32/dense/EMA/global64 与 input hashes 不变。

若 `0.90` 通过，直接进入 A4，不运行 `0.95`。若在 step1 前 OOM，只有无外部 PID、GPU 退出后正常释放、旧 checkpoint 未修改且证据完整时才进入 A3；其他错误立即 hard stop。

## A3. 条件唯一 0.95 canary

仅 A2 是 allocator/BFC OOM 才授权；nonfinite、restore/hash/step 错误、外部 PID、代码异常不得进入。新 fresh process 明确设置：

```text
CUDA_VISIBLE_DEVICES=0,1,2,3
XLA_CLIENT_MEM_FRACTION=0.95
XLA_PYTHON_CLIENT_PREALLOCATE=true
```

重新从 immutable step-0 restore，只执行一次 global64 update，门禁与 A2 相同。不授权 `>0.95`、preallocate false 或第三种 allocator。

`0.95` 仍 OOM 则状态为 `dense_fullft_local_95pct_pool_true_capacity_oom_pending_official_gpu_review`，hard stop；该证据才可用于下一轮是否启动 Blackwell 的人工决策。

## A4. 通过 fraction 的连续 10-step 验证

只使用完成 A2/A3 的最低 passing fraction。从已恢复验证的 step-1 checkpoint 续跑到 step `10`，不重新从 step0 重复第一步；配方和 allocator env 必须完全一致。

- steps `2…10` 的 loss/grad/update/params/Adam/EMA 全部 finite；
- optimizer step、examples seen、accumulation、LR warmup、EMA 次数一致；
- step `10` atomic checkpoint 与 fresh-process restore 通过；
- 报告每步 allocator/NVML peak、去编译后 throughput、checkpoint/restore 时间；
- 只估算 `50/100/1K/5K/10K/30K` 本地与 Blackwell 60h 时间，不执行它们。

A4 完成后 hard stop，等待人工审核；不跑 C20。

## 5. 最终状态与禁止项

最终状态只能是：

```text
dense_fullft_allocator_fraction_recovery_failed
dense_fullft_local_95pct_pool_true_capacity_oom_pending_official_gpu_review
dense_fullft_local_10step_allocator_passed_pending_review
```

禁止：比赛 GPU、fraction `>0.95`、preallocate false、其他 allocator、改 precision/batch/EMA/optimizer/remat/sharding、C20/50/100/1K、FAST/ZIP/upload/submit。

交付 `summary.json`、中文报告、A0 lock、step0/1/10 restore 证据、allocator internal stats、NVML trace、XLA memory analysis、测试、root/nested clean commit，然后停止。
