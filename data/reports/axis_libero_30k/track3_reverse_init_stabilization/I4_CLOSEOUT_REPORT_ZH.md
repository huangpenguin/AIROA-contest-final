# Track 3 Init Stabilization I0–I4 收尾

状态：`track3_reverse_init_stabilization_failed_named_scene_stability`

## 结论

I0 输入锁定、I1 语义门禁和 I2 的 12 个 CPU/no-renderer fresh process 均完成。三族 reverse operator-critical source relation 在固定 step5 均为 true，并在 step5→20 连续保持；三族 reverse goal relation 同一窗口连续为 false。task10 forward 对照还观察到 background initializer truth failure，证明 background truth 必须单独报告，不能自动等价为 reverse operator 失败。

但 I2 的 named scene stability 未通过，因此按计划不运行 I3。task10 reverse 最大 scene rotation drift 为 `0.0564043246` rad；task20 reverse 最大 object translation drift 为 `0.021504916` m；task19 亦超过既有 named threshold。此结果不是 t=0 R2 的重标，也不是 BDDL、SE(3) 或抓取结论。

## 证据与边界

- I2 fresh process：`12/12`；schema、finite、detail 和进程 return-code：`12/12`。
- 固定 gate：step `5`；`10/20` 仅为诊断/连续性证据，未用于 hindsight 选点。
- `settle` 与 `controlled_destruction` 属于未执行的 I3 项，均为 `not_evaluated`，不是 false。
- exact/pending/accepted reverse demo：`0/0/0`。
- 本轮仅 CPU/no-renderer；未触碰 GPU3/GPU0，未执行 SE(3)、demo、Teleop、批量生成、训练、norm、FAST/ZIP、官方 GPU、上传或提交。
- 旧 R2/R4 evidence 仍 immutable，`LIBERO-plus` 原有 dirty 状态未触碰。

机器明细：`i0_release_lock.json`、`i1_semantics.json`、`i2_stabilization_time_series.json`、`runs/`、`summary.json`。

结论后 hard stop，等待下一次审核；不得自行进入 I3、SE(3)、数据生成或训练。
