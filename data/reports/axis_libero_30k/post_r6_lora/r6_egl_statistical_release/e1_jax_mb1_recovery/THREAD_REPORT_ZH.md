# E1 JAX microbatch-1 recovery thread report

本执行链在 immutable input audit 和 CPU-only recovery tests 后，于物理 GPU1 完成 `[0,16)` canary。覆盖、finite、provenance 和 GPU boundary 通过；前 4 个 action 与既有 sentinel 未 bitwise exact，故状态为 `e1_jax_mb1_canary_failed` 并停止。

JAX 340/340、PyTorch、comparator、closed-loop、E2、FAST、ZIP、训练、比赛 GPU、push 和上传均未启动。failure summary 与 canary report 保留在同一 recovery namespace，等待审核。
