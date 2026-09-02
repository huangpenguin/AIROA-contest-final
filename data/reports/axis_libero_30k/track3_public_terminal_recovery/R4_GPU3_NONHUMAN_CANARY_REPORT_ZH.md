# Track 3 R4 GPU3 非人工 canary 报告

状态：`passed`

执行工具：`tools/axis_pi05/track3_public_teleop_canary.py --run`。只读 GPU3 guard 在执行前后均为 `5 MiB / 32768 MiB`、utilization `0`；子进程使用 `CUDA_VISIBLE_DEVICES=3` 与 `MUJOCO_EGL_DEVICE_ID=3`。任务顺序严格为 `task19 → task10 → task20`。

每个任务均通过：

- locked P1 full-state reset 后 reverse goal 为 false；
- 20 个 no-key tick，无崩溃，gripper 集合为 `[1.0]`，action 全部合法；
- localhost preview 首页、front、wrist 首帧均 HTTP `200`；
- task19 额外通过运动键、pause/unpause、camera、reset snapshot byte-equal、abort fail-closed；task10/20 完成 allowlist idle gate；
- 未写入 candidate/accepted，accepted episode count 为 `0`。

结构化原始结果：`R4_GPU3_NONHUMAN_CANARY.json`。review asset 的实际 codec/hash、20 帧 `raw_rgb_npz` 和 contact sheet 也由该 JSON 指向。该 canary 没有人工输入，不能替代 human teleop 或 human review。
