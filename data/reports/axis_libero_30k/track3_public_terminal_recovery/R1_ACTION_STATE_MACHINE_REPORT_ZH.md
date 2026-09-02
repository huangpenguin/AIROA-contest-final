# Track 3 R1 action/control state machine 报告

状态：`passed`

- `ActionStateMachine` 是 terminal 与 browser 的共同输入路径；no-key、pause、disconnect、abort 都只输出前六维为零且 gripper 为锁定的 `-1/+1`。
- `g` 在当前 tick 更新 gripper；运动键只改变对应 OSC_POSE 维度；终端按键是单 tick edge，browser 使用 keydown/keyup 控制 hold。
- `z/reset` 重新载入 P1 task snapshot、清空 candidate frames 并递增 `reset_count`；`c` 切换 front/wrist；Enter 只请求 candidate 保存。
- 非法 shape、NaN、gripper `0` 均 fail closed；异常/中断写入 `diagnostic/` 或 `rejected/`，不写 accepted。

CPU self-test 实际覆盖连续 50 次 no-key、pause/unpause、两次 gripper toggle、camera、reset、save、abort 与非法 action。对应测试和工具：`tests/test_track3_public_terminal_recovery.py`、`tools/axis_pi05/track3_public_teleop.py`。
