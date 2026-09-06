# Track3 EG0–EG4 独立复核与下一计划（2026-09-06）

审核结论：`approved`

审核输入为当前 root `4bbfe7eb7fdb404085f39712aae94572c9d0fbce`、execution source `0799142b92ad3e799dd2d1b78ca964f07dd87587`、evidence commit `5d20e25d3890e937c0278ea2656236449b82898f`、既有独立审核提交 `ad4f34e81ccaa39397f5ac550dd41f21734f47cb` 与 nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`。本审核重新计算原始 bytes/hash/closure，不以既有 `approved` 字段作为通过依据。

## 1. 独立验证结果

- EG namespace 的 `110/110` tracked files 与 evidence commit `5d20e25:path` 原字节一致；EG tool 与 finalizer source `0799142:path` 一致。EF/ER/TZ 历史 namespace 相对 `074ccdf` 无变化。
- `attempt_0001–0005` 的 producer manifest 均 exact，outer manifest/entries hash、validation/manifest/outer/heartbeat closure links 与 inactive cursor 均通过。状态保留为：`0001=synthetic failed/closure passed`、`0002=passed`、`0003=validation failed/closure failed`、`0004=passed`、`0005=passed`；attempt_0001 自首次入 Git 后字节未变化。
- `latest_attempt.json` 为 pointer-only，目标 validation/closure hash 精确指向 attempt_0005。共享 root cursor/heartbeat 保留 attempt_0003 失败是旧 aggregate 的不可覆盖证据；当前成功闭包由 `root_attempt_0005_*` 表达。其 manifest `73/73` exact，outer/heartbeat/cursor/final-validation links 均通过。
- 首次 EF4 失败的 validation、完整测试 stdout/stderr 与 partial inventory 继续诚实标为 `unavailable`；`/tmp` 与 unreachable Git blobs 未发现该 schema 的失败 raw，未重构或把 derived 说明冒充 raw。
- 当前 checkout 的 EG、既有 EG audit 与 machine handoff 共 `96/96` JSON 通过 duplicate-key rejection；既有 `94/94` 是 evidence-commit 时点口径，不是当前文件总数。typed NPZ `1/1` 可 `allow_pickle=False` 读取且 finite；两份 CSV 均 LF-only，无 `.partial`。
- fallback 独立全包复核为 `9383571677` bytes、SHA `95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a`、CRC passed、3401 entries、无 duplicate/unsafe path、唯一 FAST 且无 slow model。
- 审核前 targeted 为 `14 passed`，项目全量为 `334 passed, 7 skipped, 23 warnings`。root/OpenPI clean；`LIBERO-plus` 既有 dirty 未触碰；无 live Track3 worker或 GPU compute。

## 2. 裁决与生命周期说明

EG0–EG4 的目标是诚实冻结不可恢复 gap，并证明未来 attempt 不覆盖、clean-source identity 与独立审核边界；这些目标均满足。不存在新 physics identity、旧 identity 重跑、GPU compute、generation/replay、accepted data、training、新 candidate ZIP、upload 或 submission。

本轮同步修正两个不影响 EG 技术结论的交接口径：计划索引顶部仍写“等待独立审核”，MEMORY current snapshot 仍停在 EG 执行前；顶层 `latest_*` commit 也更新为实际 EG execution/evidence 边界。修正进入新的 audit/lifecycle namespace，不改写 EG 原始 evidence 或旧审核。

旧 EG 计划已归档。唯一下一活动计划为 `docs/AXIS_PI05_TRACK3_CROSSFAMILY_ACQUISITION_RECOVERY_CANARY_PLAN.md`：先只读诊断和 code/fixture gate，再最多运行 support、containment、articulation 各一条全新 CPU/no-renderer identity。成功也只到 `pending_replay_review`；不授权 GPU、generation/replay、accepted data、training、FAST/ZIP、upload 或 submission。

机器摘要：`data/reports/axis_libero_30k/track3_eg0_eg4_reaudit_20260906/summary.json`。

## 3. 生命周期更新后校验

最终当前集合 strict JSON `97/97`；targeted `19 passed`；项目全量 `339 passed, 7 skipped, 23 warnings`；py_compile、`git diff --check`、无 `.partial`、无 live Track3 worker 和 GPU compute 均通过。首轮 lifecycle targeted 曾因旧测试仍要求 EG authorization 指向已归档前的根目录路径而得到 `18 passed, 1 failed`；仅同步该陈旧路径断言后重跑为 `19 passed`，未修改历史执行证据。

提交前 root porcelain 只包含本轮 audit/lifecycle/plan/test 变更；nested OpenPI clean，`LIBERO-plus` 的既有 dirty 集合与审核开始时一致。提交后再次只读核对 root clean、nested 边界和进程/GPU 状态。
