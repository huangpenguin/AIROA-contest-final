# Dense full-FT allocator fraction CPU 锁

- 状态：`dense_fullft_allocator_fraction_cpu_locked`。
- 旧 R4 NVML 峰值：`24897/32768 MiB`；默认 75% pool：`24576 MiB`；峰值后物理剩余：`7871 MiB`。
- 旧单次 OOM allocation：`143327232 bytes/card`；旧证据 hash 见 `root_cause_lock.json`。
- step-0 source tree：`9380fd770ced87368c82a05a4dbb8c00fd0f0e82c4e569879aacd127a01038f5`，internal step `0`；源目录只读锁定，不会 fresh-init 绕过。
- 固定 FP32、dense、AdamW、EMA `0.999`、microbatch `1`、4 卡 accumulation `16`、global batch `64`。
- 唯一 allocator 自变量：`XLA_CLIENT_MEM_FRACTION=0.90`；只有 allocator/BFC OOM 才允许一次 `0.95`。
- 禁止比赛 GPU、preallocate false、其他 allocator、C20/50/100/1K、FAST、ZIP、上传和提交。
