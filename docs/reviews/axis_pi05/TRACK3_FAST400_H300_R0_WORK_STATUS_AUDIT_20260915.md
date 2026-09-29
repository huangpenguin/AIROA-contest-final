# Track3 Fast400 H300 R0 工作状态审核（2026-09-15 11:04 JST）

> 类型：轻量状态审核，不是 release/ZIP 正式审核
> 唯一结论：`remediation_required`
> 当前阶段：`R0_not_started`
> E1：`not_started_not_authorized`

## 1. 工作状态

上一轮 E0 范围审核之后没有新的 R0 实现。adapter 与 focused tests 的字节、大小和 mtime 均未变化；当前仍是 E0 主体实现存在、但 R6 哈希合同未闭合的状态。

阶段账本：

- 已完成：H300 本地 300 optimizer-step 训练及 fresh step300 restore；E0 adapter 主体与 12 个 focused tests；
- 未完成：R0 双哈希合同、real R6 preflight、resume PID/heartbeat、systemwide-zero 语义测试；
- `not_started`：R6/step300 behavior workers、step100 diagnostic、merge；
- `not_evaluated`：matched reverse/C20 行为分数与 T3 release gate；
- 未执行：fold、FAST、ZIP、上传、比赛提交、dense/30k continuation。

## 2. 独立复核证据

- root HEAD：`36e8f269ecd30d4a2a6e555e67bd7315011e5dbf`；nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`；两处 worktree 均已有大量其他历史/用户修改，本审核未清理、回退或覆盖；
- adapter SHA-256：`6e2b485ad92949fc1dcafb8ea4348a2681f59a8cdb46fc93b0e4916b69b3e8b1`；mtime `2026-09-15 01:08:13 +09:00`；
- tests SHA-256：`bfe7bd0e9e16ad9a06a4e32c28ddb1c060e8724c74c927034e0a9456c2fd3c3c`；mtime `2026-09-15 01:05:17 +09:00`；
- `py_compile`：passed；focused tests：`12 passed in 0.08s`；
- R6 只读 preflight：仍 fail-closed，`e98d2151... != d9b7e0d1...`；
- step300 只读 preflight：passed，params tree `6182a8ddabf6602b76ef9c18fc1ee487f192ee306135d0775d954b25204e9f4e`；
- E1 evidence/output root 均不存在，无 cursor、heartbeat、attempt lock、case、video 或 release decision；
- GPU0–3 均为 5 MiB、0% utilization，无项目 trainer/evaluator/policy-server compute process。

## 3. 仍然有效的阻断项

1. `P1`：adapter-native params-relative NUL tree SHA 被拿来与历史 Q0 repo-relative SHA 比较。历史 Q0 `d9b7...` 与 native `e98d...` 是同一文件树的两种摘要，不是权重漂移。
2. `P1`：12 个测试没有覆盖真实 R6 hash/norm fail-closed，测试通过不能关闭 E0。
3. `P2`：resume 后 cursor/heartbeat 可能保留旧 PID/GPU，heartbeat 运行中不周期刷新。
4. `P2`：名为 `no_systemwide_all_zero_actions` 的 gate 实际要求每条 episode trace 都非零，与计划文字不一致。

## 4. 指针差异

仓库顶层 plan index、Markdown handoff 和全局 `post_r6_lora/PROJECT_HANDOFF_20260831.json` 仍停在历史 Fast400 FT0；它们不能据此重跑采集或训练。本条 Fast400 H300 训后支线的当前指针是：

`data/reports/axis_libero_30k/t3_fast400_r6_h300_4gpu_r1_local4/NEXT_AGENT_HANDOFF_20260915.json`

本轮不改写三份跨项目旧 handoff，以免覆盖其并行历史内容；下一执行者必须以 scoped pointer 和本审核的 live checks 为准。

## 5. 审核结论

R0 没有被执行，E1 也没有误启动；资源和历史产物安全。下一唯一动作是代码级 R0 remediation，见 `docs/AXIS_PI05_T3_FAST400_H300_R0_REMEDIATION_EXECUTION_PLAN_20260915.md`。R0 结束后必须 hard stop 并落盘结果，不自动连跑 E1。
