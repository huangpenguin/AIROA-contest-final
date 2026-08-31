# AXIS Pi0.5 E0 Closure 审核记录

> 审核日期：2026-08-31（Asia/Tokyo）
> 审核 commit：`327316f2ffdca0df3375748bcdd65b5ce263f8f8`
> 审核结论：`approved`
> 批准状态：`e0_closure_approved_e1_ready`
> E1：允许按新的独立计划执行，不允许扩展至 E2。

## 1. 独立复核结果

- root/nested worktree 在审核时 clean；nested OpenPI 仍为 `dafa5a27266996eecf28920cacc32f3629776d91`。
- 临时目录重新生成 closure 成功；170 个 pair 的 content hash、semantic key、source kind、frame index 和 source-occurrence hashes 与已提交 manifest 逐项一致。
- coverage expected/observed sets exact；coverage issue、not evaluable、dynamics mismatch 均为 0。
- 数量复核：snapshot groups 23、trajectory groups 64、trajectory occurrences 302、snapshot/trajectory unique 13/157、总 unique 170、duplicates 155。
- closure summary SHA-256：`088376936c2ecba61258316bcafe48c70c04c7c115c62917222f7add2aba3c43`。
- content-unique manifest SHA-256：`b4ea8d3bed7f9cfc60f2f7f53613471e222d9857a8d6857f39b854720905c7bf`。
- 旧 S1.3 summary/pair-manifest SHA-256 仍为 `96f6c55...` / `dacb81b...`，旧 failure 未改写。
- 全量测试独立复跑：`112 passed, 7 skipped, 21 warnings`。
- 四个 closure 小型证据文件均已进入 Git；E1/GPU/model/closed-loop/FAST/ZIP 未启动。

## 2. 非阻断备注

- CLI 应使用 `python -m tools.axis_pi05.r6_s1_egl`；直接把文件路径交给 Python 会因 package-relative import 失败。
- `_candidate_common` 中存在一个值完全相同的重复字典 key，不影响当前生成结果；允许在 E1 CPU preflight 中机械清理并补测试，但不得借此改变 manifest 语义或 hash。

## 3. 审核决策

E0 Closure 正式完成，原活动计划归档。下一阶段不是 closed-loop，也不是 FAST/ZIP；只允许在冻结的 170 个 EGL content-unique pair 上运行 R6 folded JAX 与真实 on-disk LeRobot folded PyTorch，测量 action-response envelope。

当前唯一活动计划：`docs/AXIS_PI05_E1_EGL_ACTION_ENVELOPE_PLAN.md`。
