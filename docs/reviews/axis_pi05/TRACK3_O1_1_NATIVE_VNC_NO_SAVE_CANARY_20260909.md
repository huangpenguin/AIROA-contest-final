# Track3 O1.1 native MuJoCo/TurboVNC no-save canary

日期：2026-09-09（Asia/Tokyo）

## 结论

O1.1 通过，且已在正式人工录制前 hard stop。服务器现有 TurboVNC display :3 可用，native MuJoCo 在物理 GPU3 上启动；本次只执行 reset、控制接口、预览、单步 writer contract 和 fail-closed 检查，没有调用正式录制入口。

使用的是 O1.0 已冻结的 388-step red-mug preferred seed 的 start_state。原始 BDDL reset 本身不是 reverse 起点（source=false、goal=false），因此 canary 先验证 BDDL reset，再加载冻结 start state；加载后 source=true、goal=false、success=false，连续两次 reset 的 state 相等。

## 已通过的检查

- X11 display :3 可用，preview root/state/front/wrist 全部 HTTP 200。
- task official_t3_red_mug，20 Hz；close/open action 均有效，abort fail-closed=true。
- 物理 GPU3 显存由 5 MiB 增至 853 MiB，低于 1024 MiB canary 上限；GPU0–2 未使用。
- 临时 v4 writer 只写入 /tmp；NPZ 无 object array，contact_offsets 与 contact_n 一致，样本 contact_n=51。
- candidate inventory 前后完全不变（episode metadata 12 条）。
- save_event_seen=false、record_episode_called=false、candidate_created=false、pending_created=false。
- replay=not_evaluated，formal_recording=not_started；O2 仍冻结。

## 证据治理

O1.1 共保留 5 个 attempt。attempt 0001–0004 的失败分别是缺少 libero import、原始 BDDL reset 不是 reverse 起点、不支持的 gripper keyup 事件、摘要读取 contact_n 的派生 bug；它们没有被覆盖。attempt 0005 是物理 canary 通过的原始摘要，也保持不变。

attempt 0005 的 reset.goal_false 仅是摘要字段写反；其运行时 guard 已验证 goal0=false，且 first_reset_goal_false=true。修正写入独立 derived correction namespace，未重跑物理、未修改历史摘要：

- correction summary：data/reports/axis_libero_30k/track3_operations_o1_1_native_vnc_canary_20260909_derived_correction_20260909/summary.json
- attempt index：data/reports/axis_libero_30k/track3_operations_o1_1_native_vnc_canary_20260909_derived_correction_20260909/attempt_index.json

## 停止点

当前状态为 track3_operations_o1_1_no_save_canary_passed_ready_for_explicit_user_takeover。下一步 O1.2 只有在用户明确接管 Teleop 后，才允许最多一条 canonical red-mug formal recording；连接 VNC 本身不等于接管。O2 只有在 O1 获得 accepted 且 fresh replay=3/3 后才可启动。


## 验证闭包

项目规范测试套件为 436 passed、7 skipped、26 warnings；O1.1 定向测试为 9 passed。strict duplicate-key JSON（10 个 O1.1 相关 JSON）、canary py_compile 与 git diff --check 通过；O1.1 namespace 没有 partial 文件，candidate inventory 的实际文件 bytes/SHA 与成功 attempt 完全一致，display :3 可用，结束时无 Track3/MuJoCo/Teleop 进程。

仓库根目录的无范围 pytest 还会递归到缓存的 libero-infinity 和 third_party OpenPI 测试，因环境缺少 scenic/pynvml 在 collection 阶段失败；它不是项目规范全量门。既有的 2026-09-07 BR fixture partial 保留原位，未由 O1.1 产生、删除或移动。

Execution commit: `36457d5`.
