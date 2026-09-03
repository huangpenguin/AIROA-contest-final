# S0 公开来源复核

状态：`s0_primary_sources_verified`

已复核 LIBERO-Infinity 的真实 reverse BDDL 源码、Behavior Prompting/LIBERO-Gen 的 motion-planning 生成器与 20 Hz/OSC_POSE 配置；原始 LIBERO 的 full-state HDF5 证据沿用已锁定 P0。LIBERO-Gen 两个 HDF5 只读 HF API 元数据，未下载 159/378 GB payload。`lerobot/libero_plus` 在该 revision 的 primary card 未声明 license，因此仅作既有本地只读参考。

S0 未启动 simulator/CUDA、未创建训练 episode、未触碰人工 Teleop。
