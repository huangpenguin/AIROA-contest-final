# Dense full-FT D4 跨 optimizer step 生命周期审核

日期：2026-09-02（Asia/Tokyo）
审核输入：root `c8b0a42`，nested OpenPI `5c29b0e`
审核结论：`remediation_required`

## 1. 审核结论

D0–D3 的结果有效：donated gradient add 已证明约 `6,479,127,152` bytes 的 alias，2-microbatch merge 回落和完整 `16/16` accumulation 均通过；D3 从 step 0 完成一次 update 到 internal step 1，并保存了可恢复 checkpoint。

D4 的 hard stop 也正确，但现有文字需按原始 trace 纠正：D4 从 internal step 1 恢复后，先完整完成了 `16/16` accumulation 和一次 optimizer update，`train_state.step` 从 `1` 前进到 `2`；随后进入 `optimizer_step_before=2` 的下一轮，microbatch `0` 完成，microbatch `1` 才在真实 compute 中因申请 `143,327,232` bytes（`136.69MiB`）OOM。因此它不是“第二个 optimizer update 未完成”，而是“D4 进程已完成 step 1→2 的 update，但第三次 update 的累积在 microbatch 1 前失败”。step 2 没有原子 checkpoint/restore，所以只按 metrics、state step 和 allocator trace认定为完成 update，不升级为 checkpoint 门禁通过。

审核状态为 `remediation_required`。本轮不批准直接启用 Blackwell/比赛 GPU，也不批准修改精度、batch、optimizer、EMA、数据或数学配方。下一最小问题是：为什么 update 完成后仍有一份约 `6.5GB/card` 的 full-gradient-equivalent device tree 跨步存活。

## 2. 原始证据重建

同一 D4 fresh process 的关键每卡 `bytes_in_use`（GPU1–3数值近似一致）为：

| 边界 | internal step | bytes/card |
|---|---:|---:|
| step1 checkpoint restore / next loop start | 1 | `13,435,318,528` |
| step1 microbatch 0 release（持有 accumulator） | 1 | `19,941,729,536` |
| step1 microbatch 15 release | 1 | `19,941,729,792` |
| optimizer update complete | 2 | `19,941,608,704` |
| next loop start | 2 | `19,943,419,136` |
| step2 microbatch 0 release | 2 | `26,510,640,384` |

从 restore baseline 到第一份 accumulator 的增长约 `6.506GB/card`；下一轮 microbatch 0 后又增长约 `6.567GB/card`。该量级与 donated accumulator alias size `6,479,127,152` bytes 一致，说明跨 optimizer boundary 至少仍保留一份 full-tree-equivalent device allocation。证据不足以在本审核中唯一归因：候选包括未 donation 的 `grads` 输入、由 device `info`/norm outputs 维持的异步依赖、旧 state/update 引用或 host logging/checkpoint observer 生命周期。不得把推断写成已证实根因。

原始日志、allocator/NVML trace和 checkpoint必须保持 immutable。语义修正只进入新审核、派生 summary和handoff，不回写或删除旧 failure 证据。

## 3. 已通过项与边界

- D2：compute/merge/release `2/2`，post-merge drift最大 `512` bytes，optimizer update `0`；
- D3：compute/merge/release `16/16`，internal step `1`，examples seen `64`，finite，step1 checkpoint/restore通过；
- D4：internal step `1→2` update完成且metrics finite，随后在 `optimizer_step_before=2` 的microbatch `1` compute OOM；
- FP32、dense full model、AdamW、EMA `0.999`、microbatch/device `1`、ordered accumulation `16`、effective global batch `64`未改；
- 全量测试由执行方报告为 `141 passed, 7 skipped, 23 warnings`；
- 未启动Blackwell/比赛GPU、C20、100/1K/长训、FAST/ZIP、上传或比赛提交；四卡退出后约 `5MiB`；
- 审核开始时root/nested worktree clean，提交分别为 `c8b0a42` / `5c29b0e`。

## 4. 决策

旧 donated accumulator 计划归档。唯一活动计划改为 `docs/AXIS_PI05_DENSE_FULLFT_INTERSTEP_STATE_LIFETIME_RECOVERY_PLAN.md`：先对 optimizer output、gradient input、device metrics与Python引用建立可归因的跨步生命周期证据，再做 step1→2→下一microbatch 0 的最小 transition canary；只有显存回到稳定 plateau 才允许 step3/step10。本计划仍仅授权本地4×V100、最多internal step10。
