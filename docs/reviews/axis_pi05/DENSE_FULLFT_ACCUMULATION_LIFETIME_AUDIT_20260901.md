# Dense full-FT 梯度累积内存生命周期审核

日期：2026-09-01（Asia/Tokyo）
审核输入：root `c353ff0`，nested OpenPI `81d8220`
审核结论：`remediation_required`

## 关键发现

1. `0.90` 和 `0.95` 的 raw allocator trace 都包含 `microbatch_complete` index `0`、`1`；失败发生在 index `2` 完成之前。现有“首个 microbatch OOM”描述不准确。
2. `0.95` 下每卡 live bytes 为 `13,435,195,392 → 19,941,608,448 → 26,420,739,584`。第二个 completed microbatch 增加约 `6.48GB/card`，与保留一份完整 FP32 gradients 相符。
3. host loop 在累加后继续持有 `grads`；审计分支的 `values` 字典也持有全部 gradient leaves。下一 microbatch 的 RHS 在这些局部变量重新绑定前开始，存在明确 observer/lifetime 风险。
4. 因此 `dense_fullft_local_95pct_pool_true_capacity_oom_pending_official_gpu_review` 过度归因。证据只证明当前实现/审计生命周期在0.95 pool内失败，不证明数学配方的单-microbatch或稳定累积最小容量超过32GB。
5. 机器 handoff 顶层仍为旧 `dense_fullft_allocator_fraction_recovery_ready`、旧测试 `127 passed` 和旧 commits，而 execution子树已是132 tests与新提交；当前交接不是单一真相。

## 保留的通过项

- direct-sharded init、step0 checkpoint、params/Adam/EMA restore和sentinel证据有效；
- FP32/dense/global64合同在两次canary中未被改动；
- optimizer update为0，未伪造step1/10；
- 未启动官方GPU、C20、长训、FAST/ZIP/upload/submit；
- root与nested worktree在审核开始时clean。

## 决策

不批准直接转向Blackwell，也不批准长训。先执行新的 accumulation memory-lifetime remediation：修复临时梯度引用与审计observer effect，要求16份microbatch后显存平台化，再完成本地step1/step10。若修复后稳定平台仍在真正optimizer-update阶段OOM，才重新评估更大显存。

审核后的全量测试：`.venv_axis_pi05/bin/pytest -q tests` 为 `132 passed, 7 skipped, 21 warnings`。审核过程未启动GPU、训练、rollout、FAST、ZIP、上传或提交。
