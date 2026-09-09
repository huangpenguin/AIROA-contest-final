# Track3 O1.2 user Teleop red-mug execution closeout

日期：2026-09-09（Asia/Tokyo）

## 结论

本轮唯一获授权的 O1.2 formal recording 已完成，物理目标成功，但未达到 canonical gold 的计划资格：314 帧超过 O1 的 300-step 硬上限 14 帧。因此它保留为 pending historical raw，gold eligible=false，replay=not_evaluated；当前 O1 授权在此 hard stop，不能直接开始第二条。

## 运行事实

- 用户已明确接管 Teleop；task 为 official_t3_red_mug，20 Hz recorder，起点 source=true、goal=false、夹爪 OPEN。
- recorder 落盘 status=pending_replay，内置 candidate_gate=passed；但该 gate 没有覆盖本计划的 300-step 硬上限，派生 plan gate 为 failed_over_budget。
- frame_count=314，over_budget_by=14；end_success=true，goal 连续窗口通过，source release 通过，required demonstrator↔subject contact 通过，unrelated/unknown contact=false，actions finite。
- trajectory.npz 使用 allow_pickle=false 可读，无 object array，contact_offsets 与 contact_n 一致。保存的实际时间间隔约 0.5 秒（operator_hz=2），314 帧全部被现有 timing check 标记 deadline_miss；“加快时间流速”需要新的 reviewed plan 和 recorder 参数/逻辑变更，不能在本 attempt 后追加录制。
- 视频最终以 mp4v 写入；OpenCV 的 H264 encoder warning 已随原始运行输出保留，未将其伪装成 H264 成功。
- 没有 replay、训练、accepted promotion 或 O2；录制结束后无 Track3/MuJoCo/Teleop 进程，GPU3 回到 5 MiB。

## 证据治理

历史 metadata 与新 episode artifact 未修改。派生资格报告位于：

data/reports/axis_libero_30k/track3_operations_o1_2_red_mug_user_takeover_20260909/derived_eligibility/summary.json

data/reports/axis_libero_30k/track3_operations_o1_2_red_mug_user_takeover_20260909/derived_eligibility/validation.json

当前 O1/O2 计划只释放最多一条 canonical formal recording；这条已消耗该额度且超出硬预算。下一步不是自动补录或直接提速重录，而是等待新的 reviewed plan，明确新的 timing/waypoint 方案及重新授权。O2 继续冻结。
