# AXIS Pi0.5 Track 3：CR0–CR14 独立审核计划

> 创建日期：2026-09-07（Asia/Tokyo）
> 唯一活动计划：否；2026-09-07 已完成并归档
> 审核对象：CR0–CR14 执行闭包及其 derived CR14 finalizer correction
> 审核结论：`remediation_required`；详见 `docs/reviews/axis_pi05/TRACK3_CR0_CR14_INDEPENDENT_AUDIT_20260907.md`

本计划是只读审核计划。CR0–CR14 已按授权执行并 hard stop；旧 identity 一律不得重跑。本计划不授权新 physics、GPU replay、Teleop、generation、accepted data、training、norm、FAST/export、新 ZIP、upload 或 submission。BR、AC、ER、EG、TZ 以及 CR 历史 attempt 均保持不可变。

## A0：冻结执行边界

- 以 CR clean-source execution commit、CR0–CR8 passed attempt、CR9–CR10 canary attempt、CR11–CR14 closeout 和 CR14 derived correction 的实际 hash 为准。
- 核对 root/OpenPI HEAD、root/OpenPI porcelain、`LIBERO-plus` 原有 dirty 集合、process/GPU 边界及旧 namespace 未修改。
- 核对 attempt allocator 单调性、6 个新 profile ID、primary/alternative 上限、无 retry 和 family/shared failure 分类。

## A1：逐项复算 CR0–CR8

- 独立解析 CR0 输入锁，确认历史 43-byte raw partial 与 39-byte synthetic observation provenance 不互换。
- 复算 dirty expected/observed identity fail-close、telemetry unknown→`infrastructure_failure`、未进入 stage→`not_evaluated`、barrier 后 contact 排除和 normalized action 直接限幅。
- 复算 CR2 TERM→grace→KILL、无 descendants、normal fixture heartbeat/result，以及 CR3 manifest 在全部 producer 后生成。
- 检查 profile lock 的 canonical command/profile hash 与旧 AC/BR/ER ID/trace 集合无重用。

## A2：逐条审核 CR9–CR10

- 只读核对三族 primary 和三族 alternative 的 child identity、command、CPU/no-renderer 环境、return code、typed trace、action/trace hash。
- 确认三条 primary 和三条 alternative 均为独立正常 `family_failure`；若发现 infra/timeout/partial/telemetry unknown，应按共享基础设施失败解释，不能 promotion。
- 复算每族最多一次 alternative、总 physics=6、无 retry、策略结构差异和与 AC/BR 及本轮 primary 的 byte/action identity 去重。
- 复算 stage、contact、source、goal、clip、finite 和 typed no-pickle gate，不把未进入阶段的 gate 改写成 failed 或 success。

## A3：审核 CR11–CR14 与 finalizer

- 确认 success=0、GPU3 replay=`not_evaluated`、`pending_replay_review=0`，没有 GPU compute、Teleop、generation、accepted data、training 或 ZIP。
- 确认逐族 human handoff 是准备态，`unattended_teleop_started=false`；CR13 仅为 machine-ready dry-run。
- 独立重算 producer manifest、outer hash、strict JSON、NPZ typed/no-pickle/finite、CSV LF、无 partial、process/GPU 和完整 current set。
- 保留 attempt_0004 原始 finalizer `failed` 证据；单独审核 CR14 derived correction，不将 correction 伪装成原始 evidence。

## A4：生命周期结论

- 复核 execution closeout、machine JSON、计划索引、Markdown handoff、测试合同和归档路径是否同步。
- 审核只能返回 `approved`、`remediation_required` 或 `blocked` 之一；未执行或未能独立证明的事项必须是 `not_evaluated`。
- 无论审核结果如何，本计划结束后不得自动释放训练、打包、上传或比赛提交权限；需要新的 reviewed plan 才能改变授权。
