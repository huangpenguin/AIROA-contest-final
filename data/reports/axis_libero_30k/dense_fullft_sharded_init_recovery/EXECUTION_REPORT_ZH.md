# Dense full-FT sharded init recovery 执行报告

## 结论

状态为 `dense_fullft_first_update_true_compute_oom_pending_official_gpu_review`。sharded init 已通过，随后真实训练计算在首个 microbatch OOM；这证明当前 4×V100 的失败已从初始化复制问题推进到训练计算峰值，但不授权立即启用比赛 GPU。

## 执行证据

- R0/R1：CPU 根因锁、旧证据 immutable hash、direct parameter sharding 修复、toy/静态门禁和全量项目测试完成；`.venv_axis_pi05/bin/pytest -q tests` 为 `127 passed, 7 skipped, 21 warnings`。
- R3：四卡 init-only 通过，51 leaves、3,353,433,872 FP32 params，large replicated leaf `0`，3 个 sentinel 全部 exact，params/Adam/EMA finite，train-state step `0`；未启动 data loader/forward/backward。详情见 `r3_init_only.json`。
- R4：同一 fresh run 完成 sharded init 和 step-0 atomic checkpoint，随后进入 `train_step_accumulated_host` 的真实首个 microbatch；每卡申请 `136.69 MiB` 时 OOM。optimizer update、step1 metric、step1 checkpoint 均未完成。详情见 `r4_r5_train_10.json` 与 `r4_r5_train_10.log`。
- R4 GPU 监控峰值约 `24,897 MiB/card`；allocator 使用 JAX 默认设置，未用 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 伪装容量。退出后四卡均约 `5 MiB`。

## 边界

本轮保持 FP32、dense、EMA `0.999`、AdamW、per-device microbatch `1`、四卡 accumulation `16`、global batch `64`。未运行 R5 10 steps、restore、C20、50/100/1K/30K、比赛 GPU、FAST、ZIP、push、上传或提交。GPU0 外部进程没有被触碰。

代码 commit：root `28f47792b35f2f9c494df6e2c37f073fd884999d`，nested OpenPI `92604d6d07d5bbf636827ee903f758538c8ba015`。
