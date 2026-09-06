# Track3 AC0–AC6 独立审核（2026-09-07）

审核结论：`remediation_required`

审核输入为 root/evidence commit `720733047a687de5cb456830467151c0102e4171`、execution source `afabffdb2124a870dd16bd9e2fcf3ba9df246327`、tool SHA `3062ca7861dcf7941832562e5a3c914a94aa02a6497fb802ffda4684a4882c3d` 与 nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`。本审核只读重算 raw bytes、trace 与 gate，不接受 AC 自报的 `passed` 作为依据。

## 1. 经独立确认的有效事实

- AC namespace 的 `38/38` Git-tracked files 与 `7207330:path` 字节一致；AC0 的 22 项 EG/ER/HDF5/BDDL/snapshot lock、profile canonical hash 和 source tool hash 均 exact。
- producer manifest `28/28` 与 outer hash exact；final validation、manifest、outer、closed heartbeat 的 closure links，以及 `latest_attempt.json` 的 final/closure links均 exact。cursor inactive、heartbeat closed，记录 PID/PGID 当前不存在。
- 当前 AC 完整集合 `24/24` JSON 通过 duplicate-key rejection；finalizer 内部 `20/20` 是 final/closure/pointer 生成前的 scan-time 口径。4/4 NPZ 可 `allow_pickle=False` 读取且 finite，CSV LF-only，当前无 `.partial`。
- 三条 child 确实按 support→containment→articulation 各运行一次，profile/seed/command 均不同；parent/child identity passed、return code 0、无 timeout或后代，CPU/no-renderer 环境成立。raw trace 分别为 `155/149/180` steps，controller clip 为 true，saturation 为 `10/5/10`，source 全程 true、goal 全程 false，因此三条均可保留为诊断性 `family_failure`，不能入训。
- 未发现 retry、旧 identity 重跑、GPU compute、generation/replay、accepted data、training、norm、FAST/export、新 ZIP、upload 或 submission。root/OpenPI clean；`LIBERO-plus` 既有五项 dirty 未改变。
- AC3 partial observation 诚实记录路径、190 bytes 与 SHA `85980fcfb45dde3f948b341080c14323665639ac7155b23f2b585348a80031a3`；raw partial 已清除且没有重构，不能独立验证其历史 bytes，只能保留该 derived provenance 事实。

## 2. 必须修复的合同缺口

1. 冻结 AC3 计划明确规定任何 partial 问题都是 shared infrastructure failure 并禁止启动 AC4。实际 observation 自承 timeout fixture 留下 partial 且实现没有自动处理，但 `ac3_clean_source_supervisor_fixture.json` 仍写 `status=passed`，随后三条 AC4 在 14:04–14:05 UTC 被启动。清除 residue 不能追溯性地把 AC3 变成 pass，也不能解除 sticky stop。
2. stage 三态不合规。三条 raw 只进入 approach 或 approach+descend，但 `source_released`、`goal_continuous_true` 等未到阶段的 gate 被写成 `failed`，不是计划要求的 `not_evaluated`；support 还在未到 grasp 时以全 trace 的后续 contact window 写成 contact passed。
3. `_runtime_gate` 使用 `source[-1] is False`；NumPy `bool_` 与 Python `False` 不同一，合成的合法 true→false trace仍得到 source-release failed，因此 success path 被确定性封死。required-contact 又没有限制在 grasp/pull acquisition barrier 前，barrier 后四帧 contact 也会错误通过。
4. 所谓 immutable attempt 实际固定 `ATTEMPT_ID=1`，phase/result writer 可覆盖既有路径，fixture 甚至先 `unlink()`；没有单调 allocator 或 existing-attempt fail-close。当前 bytes 因已入 Git 而可审计，但运行时不可覆盖合同未实现。
5. manifest 按 basename 排除所有 `heartbeat.json`，导致两个 nested fixture heartbeat 未进入 28-entry producer closure。summary/CSV 的 AC6 状态仍为 `running`；finalizer strict scan 不覆盖之后生成的 final validation/closure/pointer，必须区分 scan-time 与完整当前口径。

因此不能批准“AC0–AC6 合规完成”或据此进入 replay/training。三条真实物理结果继续作为诊断性 family failure 保留；缺失/未进入阶段仍为 `not_evaluated`。旧 AC evidence 不修改。

## 3. 下一计划

只读审核计划已归档。唯一活动计划为 `docs/AXIS_PI05_TRACK3_ACQUISITION_GATE_AND_ATTEMPT_LIFECYCLE_REMEDIATION_PLAN.md`，按 BR0–BR11 连续推进：先完成 evidence-derived closure、不可覆盖 attempt、partial/sticky、完整 manifest、stage 三态、source/contact/progress 和全矩阵 fixture；全部通过并形成 clean source 后，才允许三族各一条全新 CPU/no-renderer identity，总上限 3、无 retry，随后 hard stop。

GPU0–3、official GPU、Teleop、generation/replay、accepted data、training/norm、FAST/export、candidate ZIP、upload 与 submission 仍未授权。

机器摘要：`data/reports/axis_libero_30k/track3_ac0_ac6_independent_audit_20260907/summary.json`。

## 4. 生命周期更新后校验

AC 完整集合、本轮审核 summary 与 machine handoff 共 `26/26` JSON 通过 duplicate-key rejection；定向回归 `14 passed`；项目规范全量 `353 passed, 7 skipped, 23 warnings`。AC tool 与新增 lifecycle test 的 `py_compile`、`git diff --check`、无 `.partial`、无 live Track3 worker、GPU0–3 compute empty 均通过。nested OpenPI clean，`LIBERO-plus` 原有五项 dirty 保持不变；root 仅包含本轮预期 lifecycle 变更，并在提交后复核 clean。
