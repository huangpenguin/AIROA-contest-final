# Track3 O1.2 Over-Budget 独立审核（2026-09-09）

## 唯一审核结论

`approved`

该结论只批准 O1.2 的事实分类与 hard-stop 闭包：这一条 red-mug episode 是真实执行的物理成功轨迹，但 `314 > 300`，所以保持 `gold_eligible=false`、`replay=not_evaluated`，不能进入 accepted/training。`approved` 不等于批准该 episode 为 gold，也不追认 recorder metadata 内部的 `candidate_gate=passed` 为计划 gate。

## 审核边界

- 审核输入 HEAD：`2b544b110189324b3db71070dcfadbf2634eccd7`
- O1.2 执行闭包提交：`d5ee290`
- O1.2 provenance anchor：`2b544b1`
- nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
- episode：`data/reports/axis_libero_30k/track3_reverse_pilot_v2/pending/official_t3_red_mug_1788921471`
- 洡്生资格：`data/reports/axis_libero_30k/track3_operations_o1_2_red_mug_user_takeover_20260909/derived_eligibility/summary.json`

## 独立核验结果

1. provenance 与结构成立。metadata SHA 为 `7828256a84e286058d6d602196166732e39b535e6842f2944056a612b70d5a33`，trajectory SHA 为 `91ddeeddb98f424c88f9c87edaa190a0cdad2f7af466d390d7cd859e77f3e6d8`；front/wrist 视频均为 314 帧。NPZ 以 `allow_pickle=false` 打开，26 个数组均为 typed array，无 object dtype，actions 为 `float32[314,7]`，states 为 `float64[314,84]`，所有数值数组 finite；contact offsets 为 315 项且最终 offset 与 `sum(contact_n)=16232` 一致。
2. 物理门成立。start 为 source=true/goal=false；gripper↔subject required contact 在 116–119 连续 4 帧成立，首次 source release 后保持 false，goal/check_success 在 308–313 连续 6 帧成立；unrelated contact 全程 false。由此只可称 machine-observed physical success，尚未经过 fresh replay 或人工视觉 acceptance。
3. 预算门正确失败。frame_count 为 314、硬上限为 300、over budget 为 14；derived evidence 没有用 recorder 的内部 gate 覆盖计划硬门，且没有启动 replay、accepted promotion、训练、FAST/ZIP、上传或比赛提交。
4. timing 诊断成立但命名需在下一计划修复。墙钟帧间隔中位数约 `0.499955 s`、均值约 `0.499915 s`，与 `operator_hz=2.0` 一致；`deadline_miss=314/314` 是现有 5 ms 判据相对 operator tick 的统计口径，不代表 314 帧都是物理无效帧。它也不证明 TigerVNC 是超预算原因。
5. 轨迹可作为新执行的 retiming seed，但不能离线变成 gold。动作序列只有 10 种离散行，`284/313` 对相邻动作完全相同；其中有 15 个零运动 action，但 required-contact 与 goal 连续窗口占用其中若干。后续可从本轨迹派生 keypose/压缩方案，并从冻结 start state 重新进行连续物理执行；不得删帧、改 timestamp、插值视频或改 metadata 后直接 promotion。
6. 当前 root、OpenPI、LIBERO clean；LIBERO-plus 保留既有 5 项 dirty。无 Track3/Teleop/MuJoCo/本项目训练进程。GPU0–2 由外部训练占用，GPU3 为 `5 MiB / 0%`，未终止或修改外部进程。

## 审核解释

O1.2 的目标不是“录到成功就一定入训”，而是验证一条可审计的 canonical gold 链。执行者在物理成功后仍按 300-step 计划门拒绝 promotion，并把 replay/O2 保持为 `not_evaluated`，因此闭包是诚实且合规的。主要瓶颈已经从“抓取失败”转为“路径与 formal-recorder 预算控制”；这允许下一计划扩大到多题目采集与短续训，但不能降低每条 accepted episode 的门。

## 下一授权

唯一活动计划切换为 `docs/AXIS_PI05_TRACK3_MULTITASK_GOLD_AND_R6_SHORT_CONTINUATION_PLAN.md`。先执行 MG0 的 recorder budget/timing 修复与 no-save gate；再用 314-step 成功轨迹做一次实际重新执行式 retiming，并按 red-mug → alphabet-soup → yellow-book 顺序采集。每条 machine success 立即 fresh replay `3/3` 与人工审核；达到训练门后才启动 R6 continuation。TigerVNC :3 保持首选操作面，HTTP 只保留 HUD/备用输入，不因本次超预算切换远程桌面栈。

## 最终工程验证

最终验证以稳定的 Git tracked 集合计为 JSON `853/853` duplicate-key strict parse；1 个 tracked `malformed` fixture 保持故意不可解析；定向 `8 passed`；项目全量 `440 passed, 7 skipped, 26 warnings`；py_compile 与 `git diff --check` 通过。任何未执行的 MG 阶段继续为 `not_evaluated`。
