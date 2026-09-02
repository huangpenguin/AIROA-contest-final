# T0 CPU-only 输入与资源锁

状态：`track3_reverse_t0_cpu_locked`

本阶段只读取已锁定资产、公开元数据和 GPU 状态；没有创建 CUDA context、启动 simulator、生成 demonstration 或启动训练。

- root commit：`61d4a9f0e70853854aa3c724859e0af0c3c58b25`；nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
- GPU1–3 idle gate：`passed`；GPU0 policy：`never_use`
- evaluation-only denylist overlap：`0`
- Track1/2/3 分数 provenance：均为 `user_reported`，没有伪造官方 receipt
- FAST/ZIP：仅复用既有审核 hash，未修改或重新压缩 ZIP

机器锁：[t0_cpu_lock.json](/home/huang/code/PARC2026_final/data/reports/axis_libero_30k/track3_reverse_lora/t0_cpu_lock.json)
