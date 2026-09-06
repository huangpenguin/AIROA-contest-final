# Track3 EF0–EF4 独立复核（2026-09-06）

审核结论：`remediation_required`

审核输入为 root `b1195dd59830f7c44d831bd0007e61229446d9b5`、EF0 anchor `38635fbf08e4e8df8c4877ddcb6b2b2ec8b9b118`、nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`。本审核不接受与 EF execution/evidence 同在 `b1195dd` 提交中的既有 `approved` 报告作为独立证据，并在新的 audit/derived namespace 给出修正，不改写旧报告。

## 1. 通过的独立复核

- ER anchor 的 29 个 tracked files 均与 `38635fb:path` 原字节一致；EF namespace 的 22 个 tracked files 均与 `b1195dd:path` 一致。
- EF producer manifest 13/13 bytes/SHA exact；manifest SHA `5649d2c09add79d9ab7213bada6e0454672358e5c25ae0f127bbb0aac11d795d`，outer file SHA `05f4511946a8a3513ddffac15832730ae47b7e2f5a42596197eac6c2fbf61d84`，entries canonical hash和 checkout closure 四项 hash 均通过。18 个 JSON（含 machine handoff）通过 duplicate-key rejection；typed NPZ 可 `allow_pickle=False` 读取且 finite，CSV 为 LF-only，无 `.partial`。
- EF1 identity/resume drift、EF2 singleton/heartbeat/TERM→KILL/no-descendant/sticky stop、EF3 telemetry 与 required-contact 三态分离的代码和 fixture 结果成立。EF0–EF3 的技术结论不因本次 remediation verdict 失效。
- fallback 独立全包复核为 `9383571677` bytes、SHA `95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a`、CRC passed、3401 entries、无 duplicate/unsafe path、唯一 FAST 且无 slow model。它仍是 `fallback_no_track3_improvement`，不是 Track3 改进候选。
- 审核开始时 root/OpenPI clean，`LIBERO-plus` 的既有 dirty 未触碰；无 live Track3 worker，GPU0–3 无 compute process。未发现新 physics、旧 identity 重跑、generation、replay、accepted data、training、candidate ZIP、upload 或 submission。
- 生命周期更新前独立 targeted 为 `12 passed`，项目全量为 `320 passed, 7 skipped, 23 warnings`。

## 2. 必须修复的问题

### P1：首次 EF4 失败 raw 被覆盖

`ef4_first_attempt_failure.json` 明确记录 `historical_report_replaced_by_corrected_attempt=true`。当前 `final_validation.json` 只保存 corrected pass；首次失败的完整 validation、测试 stdout/stderr 和当时 partial inventory 没有 attempt-specific 原始副本。仓库现存文件、`/tmp` 和不可达 Git blobs 的只读搜索均未找到该 raw。

这违反“历史报告和 failure label 必须保留、纠正写入新 namespace、不得覆盖旧证据使其通过”的强制规则。派生失败说明诚实且应保留，但不能等价替代已丢失的 raw。旧审核报告声称“未覆盖”与其引用的机器字段直接矛盾，因此其 `approved` 结论被本审核 supersede。

### P1：审核独立性与交接指针错误

EF tool、执行 evidence、生命周期文档和声称独立的审核报告全部进入同一提交 `b1195dd`，不能证明独立 post-execution review。machine handoff 顶层 `latest_audited_execution_commit` / `latest_audit_input_commit` 仍为旧 ER `c2d9f23...`，`latest_execution_root_commit` 仍为 `04eef...`，与当前 EF execution/audit 对象不一致。

## 3. 裁决与下一权限

唯一审核分类为 `remediation_required`。EF0–EF3、corrected EF4 checkout closure和 fallback 机械属性可以保留使用；首次失败 raw 不可补造，必须永久标为 `unavailable`。旧 EF 计划已经归档，不再恢复。

唯一新活动计划为 `docs/AXIS_PI05_TRACK3_EF_FAILURE_PROVENANCE_AND_LIFECYCLE_REMEDIATION_PLAN.md`，只允许 EG0–EG4 的 evidence-governance、code 与无 physics fixture 修复；不授权任何 physics/GPU/生成/replay/训练/ZIP/上传/提交。完成后必须再次 hard stop 并接受新的独立审核。

机器摘要：`data/reports/axis_libero_30k/track3_ef0_ef4_independent_audit_20260906/summary.json`。

## 4. 生命周期更新后校验

最终独立 targeted 为 `17 passed`，项目全量为 `325 passed, 7 skipped, 23 warnings`；EF、新审核与 machine handoff 共 `19/19` JSON 通过 duplicate-key rejection。`py_compile`、`git diff --check`、无 `.partial`、无 live Track3 worker 和 GPU0–3 compute process 为空均通过。root 仅包含本次审核/lifecycle 变更，nested OpenPI clean，`LIBERO-plus` 既有 dirty 未触碰；这些变更形成 clean commit 后才完成本次交付。
