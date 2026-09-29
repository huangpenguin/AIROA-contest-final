# Track3 MG0/MG1 快速采集独立审核（2026-09-09）

## 唯一审核结论

`remediation_required`

MG0/MG1 的行为结果与 raw 记录可信，但执行闭包没有满足活动计划自己要求的 cursor/heartbeat/outer hash 与最终全量测试证据，且 plan index 顶部仍停在 MG0，落后于 machine handoff 的 MG1 状态。用户已明确改变采集策略，因此旧计划必须归档并由简化的 400-step 快速采集计划取代。

## 独立核验

- 审核输入 root：`36e8f269ecd30d4a2a6e555e67bd7315011e5dbf`；nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`。
- MG0 namespace strict JSON `15/15`，manifest `13/13` exact；2 Hz 与 4 Hz no-save 结果保留，未产生正式 episode。
- MG1 两个 namespace strict JSON `14/14`；retry manifest `7/7` bytes/SHA exact，closure 对 manifest SHA exact。
- assisted 轨迹为 77 帧完整 telemetry 的 acquisition family failure；manual episode `official_t3_red_mug_1788934804` 为 300 帧，source 末帧 true、goal/success false。
- manual NPZ 为 typed/no-pickle：27 arrays，actions `float32[300,7]`，states `float64[300,84]`，数值 finite；contact offsets `(301,)` 且末 offset 与 `sum(contact_n)=15820` 一致。
- manual front/wrist 视频均为 300 帧、20 fps；六个 episode artifact 的 bytes/SHA 与 `manual_fallback_result.json` 全部一致。
- 历史 O1.2 成功 episode `official_t3_red_mug_1788921471` 保持 314 帧、旧 300-policy 非 gold；其 raw SHA 不改。
- 无 Track3/Teleop/MuJoCo/项目训练进程。GPU0–2 是外部作业，GPU3 空闲。未 kill/抢占外部进程，未运行 replay、accepted promotion、训练、新 ZIP、上传或提交。
- root 在修改前 clean；OpenPI clean。`LIBERO` 新出现的两个 asset XML dirty 与 `LIBERO-plus` 既有 dirty 均未触碰、未纳入本提交。

## 需要修复的点

1. MG1 没有活动计划要求的 outer hash、cursor/heartbeat 和该执行阶段自己的最终 full-test evidence；因此只能批准 raw/分类事实，不能批准完整 lifecycle closure。
2. plan index 顶部仍写 MG0 ready，但 machine/Markdown handoff 已到 MG1 exhausted，current pointer 不一致。
3. 300-step 作为进程 hard stop 与用户最后一周的采集目标冲突；一次 Save gate 失败也不应写 rejected 并结束人工 session。

## 本次 remediation

- 旧多题目计划原样归档，不重写 MG0/MG1 历史 evidence。
- 新录制器使用 400 作为 direct-gold 上限而非退出条件；默认 8 Hz，加入 0.5×–2.0× 速度滑杆；Reset 不限次数；失败 Save 只暂停提示；>400 成功 raw 进入 `pending_postprocess`。
- 历史 314-step episode 只新增派生 replay eligibility，不改旧 metadata、不直接 promotion。
- 下一唯一活动计划为 `docs/AXIS_PI05_TRACK3_FAST_TELEOP_400_AND_SHORT_TRAIN_PLAN.md`。训练门简化为 2 条 accepted / 2 task 后立即 R6 100-step。

## 验证说明

本次代码/lifecycle 修改按仓库强制规则执行 CPU self-test、strict duplicate-key JSON、targeted/full tests、py_compile 与 `git diff --check`。具体最终计数记录在 machine evidence 的 `validation.json`；这一次验证不扩展到 physics、replay 或 GPU 训练。
