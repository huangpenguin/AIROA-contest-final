# Track 3 Tonight TZ0–TZ9 执行收尾（2026-09-06）

本文件是执行收尾记录，不替代此前的独立审核报告；旧失败标签和旧 namespace 均未改写。

## 结果

- TZ0/TZ1：`passed`。R6 FAST 来源和本地 fallback 都完成全量 SHA-256、ZIP CRC、entry/layout、FAST-only 与 C5 source-success 检查。两份 ZIP 均为 `9,383,571,677` bytes、SHA-256 `95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a`，entry count `3401`，`model_fast` 唯一、无 slow 权重；C5 source-success 为 `3/3`。交付标签为 `fallback_no_track3_improvement`。
- TZ2：实际完成 `7/7` 个 CPU/no-renderer fresh child，未超出冻结上限，也未重跑旧 `object_frame_nominal`/`object_frame_conservative`。support 为 3 个 identity（2 个正常 `family_failure`，首个因执行器遗漏延迟 numpy import 记录为 `infrastructure_failure`）；containment `2/2 family_failure`；articulation `2/2 family_failure`。该执行器 bug 在停止其余 lane 后修复，已消耗的首个 identity 保持原证据、没有 retry；其余六个未执行 identity 才继续运行。
- 三族成功 profile 均为 `0`。失败轨迹均为诊断证据，`training_eligible=false`，不进入数据集。
- TZ3–TZ8：全部 `not_evaluated`，因为没有成功 family；不是推断的失败或成功。GPU3 camera、generation、fresh replay、accepted reverse、LoRA training、candidate ZIP 分别为 `0/0/0/0/false/false`。
- TZ9：最终 fallback 已再次 CRC/SHA 复验并交付本地路径 `artifacts/submission_axis_r6_fallback_revalidated_20260906.zip`。没有上传、比赛提交、Teleop、GPU0、official GPU、dense full-FT。

## 证据与边界

机器证据目录：`data/reports/axis_libero_30k/track3_tonight_zip_and_crossfamily_recovery/`。

- `tz0_summary.json`：fallback revalidation 与 C5 `3/3`。
- `tz1_cli_and_profile_lock.json` / `profile_lock.json`：TZ0–TZ9 CLI 合同与 `3+2+2=7` identity freeze。
- `tz2_cpu_matrix.json`：7/7 matrix、family isolation、首个 infrastructure failure 和后续六个 family failure。
- `tz3_summary.json`–`tz8_summary.json`：逐阶段 `not_evaluated` 原因。
- `final_manifest.json`、`phase_ledger.csv`、`tz9_validation.json`：fallback 选择、LF CSV、strict JSON namespace `23/23`（含 machine handoff `24/24`）、py_compile/diff-check、无 partial、GPU0–3 空闲快照。

最终机器 handoff 状态为 `track3_tonight_zip_and_crossfamily_recovery_completed_fallback_hard_stop_pending_review`，下一权限为 `new_reviewed_plan_required`。root/OpenPI 在最终提交后保持 clean；`LIBERO-plus` 的既有 dirty 未触碰。

最终回归：目标测试 `50 passed`；项目全量 `293 passed, 7 skipped, 23 warnings`。Tonight namespace strict JSON 为 `23/23`（含 machine handoff `24/24`），phase CSV 为 LF，py_compile、diff-check、无 partial 和 GPU0–3 空闲均通过。
