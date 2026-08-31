# Dense full-FT allocator fraction 执行审核记录

日期：2026-09-01（Asia/Tokyo）
计划：`docs/AXIS_PI05_DENSE_FULLFT_ALLOCATOR_FRACTION_RECOVERY_PLAN.md`
执行结果：`remediation_required`，最终状态为 `dense_fullft_local_95pct_pool_true_capacity_oom_pending_official_gpu_review`

## 结论

A0 CPU lock、step-0 checkpoint lock 与 allocator contract 通过。按计划保留 FP32、dense、AdamW、EMA `0.999`、每卡 microbatch `1`、4 卡 accumulation `16`、global batch `64`，只改变预注册的 JAX preallocated pool fraction。

`0.90` 在恢复 step `0` 后进入真实首个 microbatch，记录到 allocator/BFC compute OOM；因此按条件执行唯一的 `0.95`。`0.95` 同样在真实首个 microbatch 计算阶段 OOM。两次都没有 optimizer update，不能生成 step `1` 或 step `10`，A4 按 hard gate 不执行。

这证明当前 4×V100 在保持本训练配方和 preallocation 的 `0.95` pool 边界下无法完成 first update；它不是模型质量失败，也不自动授权比赛 Blackwell。是否转向官方 GPU 必须另行人工审核。

## 证据

| 项目 | `0.90` | `0.95` |
|---|---:|---:|
| classification | `allocator_bfc_oom_during_compute` | `allocator_bfc_oom_during_compute` |
| OOM allocation / card | `1207959552 bytes` | `75497472 bytes` |
| restore validation | step `0`, passed | step `0`, passed |
| checkpoint steps | `[0]` | `[0]` |
| metric steps | `[]` | `[]` |
| optimizer updates | `0` | `0` |
| allocator samples | `6` | `6` |
| NVML samples | `212` | `208` |
| external PIDs | none | none |
| GPU0 external process | untouched | untouched |

step-0 source tree SHA-256 为 `9380fd770ced87368c82a05a4dbb8c00fd0f0e82c4e569879aacd127a01038f5`，两次运行前后的 source checkpoint 均未改变。allocator internal stats 与 NVML traces 已保留；JAX `0.5.3` 的 `memory_analysis()` 探针已记录后端 `TypeCheckError`，不把不可用的 analysis 冒充为计算峰值结论。

## 测试与边界

- `.venv_axis_pi05/bin/pytest -q tests`：`132 passed, 7 skipped, 21 warnings`。
- 0.90/0.95 运行前后四张 V100 均约 `5 MiB`；没有外部 PID。
- 未启动 A4/10-step、C20、50/100/1K/长程训练、官方/比赛 GPU、FAST、ZIP、上传、push 或提交。
- 当前只等待人工审核；在人工授权前不得创建或执行新的训练阶段。

主要文件：

- `data/reports/axis_libero_30k/dense_fullft_allocator_fraction_recovery/summary.json`
- `data/reports/axis_libero_30k/dense_fullft_allocator_fraction_recovery/RECOVERY_REPORT_ZH.md`
- `data/reports/axis_libero_30k/dense_fullft_allocator_fraction_recovery/cpu_lock.json`
- `data/reports/axis_libero_30k/dense_fullft_allocator_fraction_recovery/root_cause_lock.json`
- `data/reports/axis_libero_30k/dense_fullft_allocator_fraction_recovery/step0_lock.json`
- `data/reports/axis_libero_30k/dense_fullft_allocator_fraction_recovery/allocator_contract.json`
