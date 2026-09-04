# Track 3 Background Predicate Convergence K4 收尾

- 最终执行状态：`track3_background_convergence_gpu_failed_task19_canary`
- K0：15 份 raw worker report 已显式纳入 Git，总计 `16,072,261` bytes；独立 fresh checkout 验证 `15/15` hash 一致。
- K1：合同已冻结为 critical source 严格 `true`、reverse goal 严格 `false`、background initializer 仅允许单调 `false→true`，禁止 `true→false` 与回摆。
- K2：只读复算通过，reverse `9/9`、forward `6/6`；旧 J2 exact-vector `8/9` 与 task19 run1 失败永久保留。新 ledger 中只有原先记录的 1 次允许 `false→true`，没有 forbidden transition。

## K3 结果

按计划只运行了 1 条 task19 canary，随后立即 hard stop，三族 matrix 为 `not_evaluated`、运行数为 `0`。worker exception 为 `0`，进程返回码合同通过。

canary 的 physical/contact/participant/robot 与 reverse goal gate 通过，但整体未通过：

- background predicate `(On akita_black_bowl_1 main_table_akita_black_bowl_region)` 在 step19→20 为 `true→false`；
- renderer schema 报告 `use_camera_obs=false`、`has_offscreen_renderer=false`，虽实际取得两路各 `21/21` camera frames；
- 两路 H.264 因临时输出文件扩展名无法被 ffmpeg 识别而失败；
- direct state sensitivity 的 before/destroyed/restored 观测为 `true/false/true`，但当前 gate 状态仍为 `failed`，按原始证据保留，不重标。

因此不修复、不 retry、不换 seed、不运行 task10/task20/task19 matrix。三族动态验证、SE(3)、抓取/demo、Teleop、批量生成、训练均为 `not_evaluated` 或禁止项。

## K4 验证与边界

- targeted tests：`12 passed`
- full tests：`201 passed, 7 skipped, 26 warnings`
- `py_compile`、JSON parse、`git diff --check`：通过
- GPU3 仅用于本次 task19 canary；GPU0/1/2 未触碰，GPU0 外部 PID `1133706` 未触碰。
- reverse demo counts 仍为 `exact/pending/accepted = 0/0/0`；旧 I2 hash 保持不变；未执行 SE(3)、demo、Teleop 或训练。

本轮已 hard stop，等待下一次审核决定是否需要专门修复 camera/renderer telemetry、H.264 编码和 K3 predicate 语义；在新授权前不得重跑 GPU 或进入 SE(3)。
