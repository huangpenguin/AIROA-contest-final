# Post-R6 LoRA N0 审计固化报告

## 结论

N0 已通过，可以进入 N1。此阶段只完成代码审计、协议硬化、测试和资源快照，没有启动新的 GPU training、rollout 或导出。

## 已固化的门禁

- paired report 现在强制 candidate/baseline 的逐 case `official_task_index`、task、language、BDDL hash、init index/hash、env seed、policy seed、max steps 一致；manifest 顺序、覆盖率、视频存在性和 summary/CSV count 也必须一致。
- stall 使用 rollout 与 paired report 共用的 `is_stall_episode()` 判定：失败且 `action_norm_mean < 1e-3`。
- paired report 输出 episode bootstrap、task-cluster bootstrap、McNemar exact test，并明确区分 `point_noninferiority` 与 `ci_noninferiority`。
- episode reset 强制整数 `policy_seed`，HTTP 日志包含 `reset_kind`、`episode_seed`、`chunk_index`、`reseed`；execute8 的 `chunk_discard` 不再 reseed。
- 新增单元测试覆盖 seed/init mismatch、summary/CSV count mismatch、chunk discard 不 reseed 及 collision/timeout delta 方向。

## 起点资源与 provenance

- root commit：`8ca78616664a638d1aaafc879c97dd7f398085ec`，clean。
- nested OpenPI：`dafa5a27266996eecf28920cacc32f3629776d91`，clean。
- canonical data：`physical-intelligence/libero@a4336d589d589045d1c56423ffdf3b88a0e19b1f`，native 10 Hz。
- fresh norm：`4c803bf59c52677c36b6b420c614d8a7742e0847708443863e13e67cb80b18f5`，ACTION/STATE 均为 `QUANTILES`。
- B80 manifest：`6209fa920dd5e7c6f225069592c08f4ee6fb14c31427414d61d2c98c04f1b581`。
- `/home` 可用空间约 914 GB，高于计划规定的 500 GB。
- GPU1/2/3 当时各约 5 MiB；GPU0 有外部 PID `237527`，未绑定、未终止、未重置，`gpu0_touched=false`。

## 验证

```text
.venv_axis_pi05/bin/python -m pytest -q tests
82 passed, 7 skipped, 21 warnings
```

`py_compile` 与 `git diff --check` 均通过。机器锁定信息见同目录的 `N0_AUDIT_LOCK.json`。
