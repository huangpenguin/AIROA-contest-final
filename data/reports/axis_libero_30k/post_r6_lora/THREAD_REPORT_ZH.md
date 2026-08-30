# Post-R6 LoRA 执行线程报告

## 当前状态

当前状态为：

```text
r6_evaluator_and_fold_reproducibility_failed
```

N0 已通过；N1.1 已完成 action trace 落盘；N1.2 同 GPU deterministic replay 失败。按照计划硬停止，N1.3 parity grid、N2 V160/H160、N3 长程训练及后续阶段均未启动。

## N0

- paired report 已加入逐 case task/init/BDDL/env seed/policy seed/max-step 硬门禁。
- summary/CSV 的 case、success、collision、timeout、stall 计数必须一致。
- episode reset 强制使用 manifest 的整数 `policy_seed`；`chunk_discard` 只清空服务端 action buffer，不 reseed。
- N0 测试结果：`82 passed, 7 skipped, 21 warnings`；`py_compile` 和 `git diff --check` 通过。

## N1.2 重放结果

| 模型 | GPU | coverage | trace bytes mismatch | outcome mismatch | 最大数值 action 差异 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| Axis step-0 | 1 | 20/20 | 20/20 | 15/20 | 2.0025522708892822 | failed |
| R6 step-500 | 2 | 20/20 | 20/20 | 16/20 | 2.0012078285217285 | failed |

两组 replay 的 checkpoint、root/nested commit、数据 revision、R6 fresh norm hash、QUANTILES 模式和 policy-seed contract 均相同；`stall=0`，`gpu0_touched=false`。第一处数值 trace 差异分别为 Axis `C20-task02-init0 [0,0]`、R6 `C20-task02-init0 [0,3]`。

这不是“R6 性能提升未显著”的统计结论，而是 evaluator 可重复性门禁失败。当前不能把已有 B80、raw/folded fixed parity 或 E2E processor parity 当作闭环 evaluator 可重复性的替代证据，也不能批准候选。

## Action range 审计

用户提出的现象方向正确：7D raw action 是不同 controller command units，gripper 为 `{-1,+1}`，旋转维度通常比平移小；但 canonical persisted action 的前三维不是简单的 `±0.1` 米，fresh norm 的 q01/q99 约为 `x [-0.704,0.938]`、`y [-0.801,0.868]`、`z [-0.938,0.938]`，旋转约为 `[-0.115,0.132]`、`[-0.164,0.193]`、`[-0.224,0.335]`。因此当前正确做法是逐维 `QUANTILES`，不是统一 scalar range、mean/std 或 `IDENTITY`；该 range 异构本身不是 NaN 证据。

## 机器产物

- 同卡 Axis 报告：`evaluator_reproducibility/axis_step0_same_gpu_summary.json`
- 同卡 R6 报告：`evaluator_reproducibility/r6_step500_same_gpu_summary.json`
- 四次 C20 replay 的 summary、CSV、action trace、视频和日志均在同目录下。
- 汇总：`evaluator_reproducibility/summary.json`

## 停止与恢复条件

不得放宽 action trace 阈值，也不得直接重跑训练或扩大评价面板。下一步只能是独立定位 evaluator/environment/JAX/runtime 的 nondeterminism；修复后以相同 C20、相同 GPU、相同 seed contract 重新通过 N1.2，才可考虑进入 N1.3。
