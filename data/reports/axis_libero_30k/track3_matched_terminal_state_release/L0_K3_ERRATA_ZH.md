# Track 3 Matched Terminal-State L0：K3 Errata

本文件是新 namespace 的只读 errata，不修改旧 K3/K4 raw evidence。

- 旧 task19 native reverse K3 的 step5 有 39 个 `akita_black_bowl_1_main` ↔ `wooden_cabinet_1_cabinet_middle` geom contacts，最深 penetration 为约 `-6.797624546751698e-3 m`；该 contact 不能被 step5 baseline grandfather。
- K3 renderer 的 flags 只读了 wrapper 层；实际 camera frames 仍为 agentview `21/21`、wrist `21/21`。
- H.264 失败来自 `*.mp4.partial` 临时扩展名，ffmpeg 无法选择 muxer；新工具使用 `*.partial.mp4` 后再 atomic rename。
- task19 sensitivity 的 raw mutation 从 `qpos=-0.146432...` 继续向负方向移动，仍满足 public `WoodenCabinet.is_open`；因此 raw truth 是 `Open true → true → true`，不是旧中文报告中的 `true → false → true`。新实现从 public close range 的绝对 qpos 中点验证并 exact restore。
- 旧 K3 没有显式 seed；历史 `no_seed_change` 只作为历史字段保留，不能冒充 seed identity。

新 L0 固定使用公开 HDF5 中同一个 demo 的 `model_file` 与 terminal full-state；predicate oracle 只在加载该 XML 的 runtime env 中执行。
