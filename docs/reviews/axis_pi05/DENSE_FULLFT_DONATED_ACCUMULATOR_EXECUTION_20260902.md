# Dense full-FT Donated Accumulator 执行记录

日期：2026-09-02（Asia/Tokyo）
执行输入：root `8e9f3cf`，nested OpenPI `5c29b0e`
执行结论：`remediation_required`

## 1. 授权范围

本轮只执行唯一活动计划 `AXIS_PI05_DENSE_FULLFT_DONATED_ACCUMULATOR_RECOVERY_PLAN.md` 的 D0–D4。FP32、dense full-FT、AdamW、EMA `0.999`、microbatch `1`、ordered accumulation `16` 和 effective global batch `64` 均保持不变。

## 2. 门禁结果

- D0/D1：CPU/code gate 通过。donated tree add 保持严格 FP32、左结合和原 microbatch 顺序；移除会使 donated optimizer state 失效的 compile observer，不改变实际 optimizer update；全量测试为 `141 passed, 7 skipped, 23 warnings`。
- D2：通过两份真实 microbatch merge-only canary。compute/accumulation/release indexes 均为 `[0,1]`；merge 后每卡 live bytes 为 `19916301568–19916302080`，最大漂移 `512` bytes，donated alias 为 `6479127152` bytes，optimizer updates 为 `0`。
- D3：最终 fresh run 通过 `16/16` compute/merge/release，完成 step `1`、examples seen `64`，finite gate 通过并成功原子写入 step1 checkpoint。此前两次 observer/logging 生命周期失败的输出目录均保留，最终 run 使用独立 report label。
- D4：从 step1 续跑时，step2 microbatch `0` 完成并释放；microbatch `1` 在真实 forward/backward 阶段触发 BFC allocator OOM，申请 `143327232` bytes（`136.69MiB`），四卡均为 `RESOURCE_EXHAUSTED`。continuation return code 为 `1`，未生成 step10 checkpoint，分类为 `allocator_bfc_oom_during_compute`。

## 3. 资源与边界

旧 runner 在 continuation 失败后短暂启动了 restore-only 子进程，随后立即终止；无完成的 step10 restore，该日志和 NVML trace 仅作为不完整证据保留。没有启动 Blackwell/官方 GPU、C20、100/1K/长训、FAST/ZIP、上传、push 或提交；四张本地 GPU 退出后均约 `5MiB`。

## 4. 证据

- `data/reports/axis_libero_30k/dense_fullft_donated_accumulator_recovery/d2_merge_canary.json`
- `data/reports/axis_libero_30k/dense_fullft_donated_accumulator_recovery/d3_step1.json`
- `data/reports/axis_libero_30k/dense_fullft_donated_accumulator_recovery/d4_donated_continue_10.json`
- `data/reports/axis_libero_30k/dense_fullft_donated_accumulator_recovery/d4_step10.json`
- `data/reports/axis_libero_30k/dense_fullft_donated_accumulator_recovery/summary.json`

## 5. 停止

D4 的真实计算 OOM 证明当前 donated accumulator 方案尚未达到 step10 门禁，但不单独证明必须使用 Blackwell，也不授权改变配方或继续训练。当前状态为 `dense_fullft_donated_accumulator_remediation_failed`，停止等待下一次人工审核。
