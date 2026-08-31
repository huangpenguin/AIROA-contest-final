# E1 JAX microbatch-1 recovery 执行记录

日期：2026-08-31（Asia/Tokyo）

## 结论

本轮状态为 `e1_jax_mb1_canary_failed`。CPU-only 输入审计、原子分片实现与 focused tests 已通过；GPU1 上的 16-observation canary 覆盖和 finite 检查通过，但旧 sentinel 要求的前 4 个 action bitwise exact 未通过。因此按计划 hard stop，未运行 JAX 340/340 full。

## Canary 证据

- `microbatch=1`，物理 GPU1，`[0,16)`，internal `[16,10,32]`，external `[16,10,7]`，全部 finite。
- input lock、manifest、frozen inputs/noise、folded model、norm、raw parameter tree provenance 均 exact。
- internal 前 4 个 action：281/1280 元素 exact；最大绝对差 `1.1920928955078125e-7`。
- external 前 4 个 action：226/280 元素 exact；最大绝对差 `1.1920928955078125e-7`。
- `gpu0_touched=false`；GPU1 目标 PID `856004`，峰值约 `24706 MiB`；GPU0 原外部 PID `237527` 未触碰。
- `[0,16)` 已作为原子 shard 写入并保留，但因 canary gate 失败，明确不可作为 full prefix 使用。

详细机器记录见 `canary/report.json`，汇总见 `summary.json`。

## 边界

本轮只执行了 folded JAX microbatch-one canary。没有启动 JAX full、LeRobot/PyTorch、comparator、closed-loop、E2、FAST、ZIP、训练、比赛 GPU、push 或上传。等待审核或新的最小 remediation 计划。
