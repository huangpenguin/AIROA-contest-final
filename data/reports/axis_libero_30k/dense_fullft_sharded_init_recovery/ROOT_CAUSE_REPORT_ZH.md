# Dense full-FT sharded init CPU 定位报告

- 状态：`dense_fullft_sharded_init_recovery_cpu_locked`。
- 旧两卡/四卡证据均保持 immutable；traceback 终点是 `init_train_state`，optimizer updates 为 `0`。
- 旧四卡日志记录 input/output arguments `26840528788` bytes、rematerialization floor `13433323028` bytes、OOM allocation `56623104` bytes。
- 旧实现将约 `12.5 GiB` host Axis params 以 replicated `PartitionSpec()` 作为 init 输入；新实现直接使用 `state_sharding.params` leaf sharding 并在 JIT 前 `device_put`。
- 方案保持 FP32、dense、EMA `0.999`、AdamW、per-device microbatch `1`、global batch `64`；不以 donation、降精度或改 batch 声称节省显存。
- 下一步仅允许四卡 init-only；通过后才允许唯一 first update，最多连续到 10 steps。

旧证据 hash 见 `root_cause_lock.json`，数值/静态门禁见 `numerical_gate.json`。
