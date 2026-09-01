# Dense full-FT accumulation memory-lifetime 执行记录

日期：2026-09-01（Asia/Tokyo）
执行输入：root `1ac116f`，nested OpenPI `e31c2bf`
计划：`docs/AXIS_PI05_DENSE_FULLFT_ACCUMULATION_MEMORY_LIFETIME_RECOVERY_PLAN.md`

## 结论

执行状态为 `dense_fullft_accumulation_lifetime_remediation_failed`。这不是模型能力失败结论，而是本地 `0.95` JAX BFC pool 在第二份梯度归并时仍然无法容纳额外临时 buffer；按计划未启动 M3。

## 门禁结果

| 阶段 | 结果 |
|---|---|
| M0 标签纠偏 | 通过；旧 raw logs/checkpoints 未修改 |
| M1 CPU 生命周期修复 | 通过；全量 `137 passed, 7 skipped, 23 warnings` |
| step-0 immutable copy | 通过；`24,879,588,870` bytes，atomic rename，source hash 未变 |
| step-0 restore/sentinel | 通过；params、optimizer、EMA、scheduler/RNG finite/exact |
| M2 四卡 canary | 失败；见下文 |
| M3 step 10 | 未启动 |

M2 使用四张自然空闲 Tesla V100 32GB，`microbatch=1`、`accumulation=16`、global batch `64`、FP32、dense、AdamW、EMA `0.999` 均保持不变；allocator 为 `XLA_CLIENT_MEM_FRACTION=0.95`、`XLA_PYTHON_CLIENT_PREALLOCATE=true`。GPU0 没有外部 PID，训练退出后四卡均回到约 `5 MiB`。

## 失败定位

真实训练完成了 microbatch compute index `[0, 1]`。index `0` 已记录 `accumulation_complete` 与 `temporaries_released`；index `1` 的计算完成，但在：

```text
_add_gradients(grad_sum, grads)
```

申请额外 `72.00 MiB` 时发生 BFC OOM。退出码为 `1`，非 timeout；optimizer update 为 `0`，metric steps 为空，只有 step `0` checkpoint。M2 plateau gate 因缺少 index `1…15` 的 release rows 而失败，不能声称 16/16 accumulation 或 step1 通过。

原始证据：

- summary：`data/reports/axis_libero_30k/dense_fullft_accumulation_memory_lifetime_recovery/summary.json`
- M2 report：`data/reports/axis_libero_30k/dense_fullft_accumulation_memory_lifetime_recovery/m2_step1.json`
- allocator raw trace：`data/reports/axis_libero_30k/dense_fullft_accumulation_memory_lifetime_recovery/m2_lifetime_canary_allocator.jsonl`
- 原始训练 log：`data/reports/axis_libero_30k/dense_fullft_accumulation_memory_lifetime_recovery/m2_lifetime_canary.log`

禁止事项核对：官方/比赛 GPU、Blackwell、M3、100/1K/长训、C20、FAST、ZIP、push、上传和提交均未启动。当前停止等待审核；不得从本记录自动推导下一项修复或扩大资源授权。
