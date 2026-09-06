# Track3 EG0–EG4 独立审核（2026-09-06）

## 审核范围与输入

本审核只检查新的 EG evidence namespace 和其生命周期记录，不重跑 physics、旧 EF/ER/TZ identity、GPU、generation/replay、training、candidate ZIP、upload 或 submission。审核输入为 execution source `0799142b92ad3e799dd2d1b78ca964f07dd87587`、已提交 evidence `5d20e25d3890e937c0278ea2656236449b82898f`，以及冻结的历史审核输入 root `b1195dd59830f7c44d831bd0007e61229446d9b5`。本报告在上述 evidence commit 之后生成，未与 execution/evidence 同提交。

## Provenance gap

`ef4_first_attempt_failure.json` 明确承认 shared `final_validation.json` 被 corrected attempt 替换。EG0 将首次失败原始 validation、完整测试 stdout/stderr 与 partial inventory 均记录为 `unavailable`；只读检查 Git history、临时目录和 unreachable objects 未找到可恢复副本。派生记录保持 derived，未被重构为 raw，EF/ER/TZ 历史 namespace 与用户给定的 `074ccdf` 比对未发生变化。

## Attempt-specific finalizer

无 physics fixture 的 `attempt_0001→0002` fail→retry 回归通过，首个 attempt 的每个文件 bytes/hash 保持不变，两个 attempt 均可 strict parse、manifest/outer/closed-heartbeat/closure 校验。closeout 中 `attempt_0003` 诚实记录全量测试失败，`attempt_0004` 诚实记录 root aggregate 已存在而拒绝覆盖；二者的 validation、测试输出、manifest/outer/heartbeat/closure 均保留。不可复用的 `attempt_0005` 最终通过，`latest_attempt.json` 仅指向目标 hash。

## Identity 与生命周期

EG2 clean-source fixture 的 parent→child identity gate、CPU/no-renderer profile、root/tool/OpenPI/command/porcelain digest 通过；root dirty 与 command drift 均 fail-close。EG3 独立确认顶层 `latest_audited_execution_commit`、`latest_audit_input_commit`、`latest_execution_root_commit` 都是实际冻结的 `b1195dd...`；活动计划已归档且不存在，index、Markdown handoff、machine JSON 与测试均指向 `independent_audit_required`。execution source、evidence commit 和本审核提交分离记录。

## EG4 闭包与资源边界

`attempt_0005` final validation、attempt closure 与 root attempt closure 均通过；root manifest `73` entries 的 bytes/SHA exact，outer hash、inactive cursor、closed heartbeat exact。证据提交后独立重算得到 strict JSON `94/94`、typed NPZ `1/1` no-pickle/finite、CSV 全部 LF、无 `.partial`。既有 fallback 只读校验为 `9383571677` bytes、SHA `95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a`、CRC passed、3401 entries、FAST-only。

EG4 定向测试为 `7 passed`，全量测试为 `332 passed, 7 skipped, 23 warnings`；`py_compile` 与 `git diff --check` 通过。最终只读进程扫描无 Track3 worker，GPU0–3 无 compute app；root/OpenPI clean，`LIBERO-plus` 原有 dirty 未触碰。新 physics identity、旧 identity 重跑、GPU compute、generation/replay、training、新 candidate ZIP、upload 和 submission 均为零/false。

## 审核结论

`approved`。该结论仅表示 EF 首次失败 provenance gap、attempt-specific 不可覆盖 finalizer、clean-source identity 与 EG evidence/lifecycle 闭包已满足；不释放任何 physics、GPU、数据生成、训练、ZIP、上传或比赛提交权限。当前 hard stop，下一权限为 `new_reviewed_plan_required`。

机器摘要：`data/reports/axis_libero_30k/track3_ef_failure_provenance_and_lifecycle_audit_20260906/summary.json`
