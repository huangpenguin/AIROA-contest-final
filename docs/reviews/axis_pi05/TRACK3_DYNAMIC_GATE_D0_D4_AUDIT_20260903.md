# Track 3 Dynamic Gate D0–D4 独立审核

> 审核日期：2026-09-03（Asia/Tokyo）
> 审核输入：root `84c2655f610bb957b0ad1ec2a952776bf023821e`，parent `f0034b9976e6de81644d1535d60328006bbe83cf`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`
> 新状态：`track3_native_reverse_validator_recovery_planned`

## 1. 获批事项

- D0 lock 通过：28 项旧 evidence、3 份官方 HDF5 与 3 份 P1 snapshot 均被锁定；三份 HDF5 的 `states[0] == init_state`，time/qpos offset=`0/1`，exact/pending/accepted reverse demo 仍为 `0`。
- 新 validator 不再把 HDF5 state 用作跨 model predicate oracle；reverse native reset 和逐 predicate `_eval_predicate` 路线正确。
- controller reset 顺序、20+20 warm-up/measurement 分窗、7D OSC_POSE 零 gripper incremental command、named/dimensioned robot/scene metrics 与 controlled destruction 均有 CPU 测试。
- D2 只启动了一次 task10 fresh process；发生异常后没有重跑或进入 D3。未执行 SE(3)、demo、训练、Teleop、FAST/ZIP、官方 GPU、上传或比赛提交。
- root/OpenPI clean，`LIBERO-plus` 既有 dirty 状态未触碰；全量测试记录为 `180 passed, 7 skipped, 23 warnings`。

hard stop 与范围控制完全正确。

## 2. D2 的正确解释

D2 run 的唯一结果是：

```text
status             worker_exception
error              AttributeError: 'TargetZone' object has no attribute 'get_geom_state'
worker_returncode  0
```

异常发生在 predicate detail/geometry telemetry 层，早于可持久化的 native predicate、settle 与 controlled-destruction 证据。因此当前 `track3_native_reverse_dynamic_canary_failed_support` 只是历史运行标签，不能解释为 task10 predicate、控制器、动力学或 reverse semantics 失败；D3 `0/9` 也不是三族失败样本。

## 3. 修复审核

commit 中的后置修复方向与 LIBERO 源码一致：

- `object_sites_dict[region]` 是原始 `TargetZone` / `SiteObject`；
- `object_states_dict[region]` 是 `SiteObjectState` wrapper，才提供 `get_geom_state()`；
- 当前 `_object_or_site_pose()` 已改为从 `object_states_dict` 获取 site wrapper；新增 unit test 覆盖了该路由。

但仅有 fake object unit test 不足以批准 D3。执行前仍需补三个约束：

1. 用真实无 renderer 的 task10/task20/task19 native env 做 runtime schema probe，覆盖所有 predicate argument 类型，而不是只测 fake wrapper；
2. 将 authoritative predicate truth 与 advisory detail extraction 分离。detail 缺失应记录为明确 schema failure，不能让已经求出的 predicate truth 消失或把 worker 变成无诊断异常；
3. worker exception 必须同时产生非零 process return code，parent 必须拒绝 `returncode != 0` 与 `status=passed` 的矛盾组合。现有 `worker_exception + returncode 0` 不符合进程级 fail-closed provenance。

完成上述 CPU/runtime 门禁后，原 D2 因 validator 异常而没有消耗一次“有效行为 canary”。允许使用相同 task/spec/threshold 做且只做一次恢复复测是合理的，不属于失败后调参。

## 4. 审核裁决

审核状态为 `remediation_required`：

- 批准 D0 evidence 与 hard stop；
- 保留原 D2 JSON immutable，不重标、不覆盖；
- 不批准直接进入 D3；
- 批准新计划中的 runtime schema closure 和一次 task10 recovery canary；
- recovery canary 全通过后，可在代码/hash/threshold 不变的条件下执行原定三族 `3/3`，随后必须停止等待审核。

## 5. 不授权事项

本审核不授权旧 SE(3) 提取、抓取执行器、真实 reverse demo、人工 Teleop、批量生成、训练、norm、FAST/ZIP、官方 GPU、上传或比赛提交。

## 6. 审核验证

- 定向测试：`20 passed, 2 warnings`
- 全量测试：`180 passed, 7 skipped, 23 warnings`
- machine JSON parse、`py_compile` 与 `git diff --check`：通过
