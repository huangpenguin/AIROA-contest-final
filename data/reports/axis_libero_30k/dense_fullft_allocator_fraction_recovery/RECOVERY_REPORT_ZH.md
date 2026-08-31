# Dense full-FT allocator fraction recovery

- 最终状态：`dense_fullft_local_95pct_pool_true_capacity_oom_pending_official_gpu_review`。
- 0.90：`allocator_bfc_oom_during_compute`，return code `1`；首个 microbatch 后申请 `1207959552 bytes/card` 失败。
- 0.95：`allocator_bfc_oom_during_compute`；首个 microbatch 后申请 `75497472 bytes/card` 失败。
- 两次 restore validation 都通过 step `0`，checkpoint steps 都只有 `[0]`，metric steps 都为空，optimizer updates 都为 `0`。
- 通过 fraction：`none`。
- 训练配方保持 FP32/dense/AdamW/EMA 0.999/microbatch 1/global batch 64；没有为适配显存而改 recipe。
- 本轮没有启动比赛 GPU、C20、长程训练、FAST、ZIP、上传或提交。
- allocator internal stats、XLA memory analysis（后端探针不可用原因已记录）和 0.25s NVML trace 由同名 phase artifacts 提供；有效运行分别为 allocator `6` samples/NVML `212` samples 与 allocator `6` samples/NVML `208` samples。
- CPU-only 全量测试：`.venv_axis_pi05/bin/pytest -q tests`，`132 passed, 7 skipped, 21 warnings`。
- 现在停止，等待人工审核。
