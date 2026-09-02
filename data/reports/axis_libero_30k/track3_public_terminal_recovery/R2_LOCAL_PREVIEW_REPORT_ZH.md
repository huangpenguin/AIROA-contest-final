# Track 3 R2 localhost preview 报告

状态：`passed`

预览服务只绑定 `127.0.0.1`，并排提供 front/wrist JPEG、task index、reverse instruction、goal predicate、step、20Hz/lag、gripper、pause、reset count、success。页面的 keydown/keyup、Pause、Reset、Front/Wrist、Save Candidate、Abort 全部进入 R1 state machine；失焦或断连清空 motion keys，保持 gripper。

录制路径直接使用锁定 full state 的 `set_init_state` observation 发布安全首帧，不先推进一次 physics；本地 HTTP client 未访问首页及 front/wrist 首帧前，不进入 record loop，因此保存的 replay 起点与首个录制 action 对齐。GPU3 canary 对三个 allowlist task 均获得 root/front/wrist `200`。

GPU3 canary 期间为 task10/19/20 各保存 20 帧、20Hz 的真实 settle review asset：OpenCV 实际 codec、路径和 SHA-256 以 `R4_GPU3_NONHUMAN_CANARY.json` 为准；同时保存 `raw_rgb_npz` 精确时序与 contact sheet。未将 PNG hash 冒充 H.264。

人工作业固定端口 URL：`http://127.0.0.1:8765/`。远程查看可执行：

```bash
ssh -N -L 8765:127.0.0.1:8765 huang@gpu01
```

这是 localhost-only forwarding，不开放公网。
