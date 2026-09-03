# Track 3 S0–S5 自动 Canary 执行报告

## 结论

当前状态：`track3_scripted_executor_failed`。

S0–S3 门禁通过后，S4/S5 按预注册顺序只尝试了 1 条 support canary。它在 `descend_object` 阶段 120 步内未达到预注册的 EEF 误差阈值，随后按计划停止；没有 retry，也没有继续 containment 或 articulation。失败资产没有裁剪，保存在：

`data/reports/axis_libero_30k/track3_public_autogen/diagnostics/support_remove_task_10/`

该目录包含 132 帧完整 `trajectory.npz`、`start_state.npy`/`end_state.npy`、policy state、predicate/subgoal/contact/deadline/action 记录和 front/wrist H.264。失败原因是 `stage_timeout:descend_object`，最终 predicate 为 false；`pending_autogen` 数量为 `0`。

## 阶段结果

| 阶段 | 结果 | 关键证据 |
|---|---|---|
| S0 | passed | primary source、revision/tree/license、R6/HDF5/P1/R4 hash lock；未下载大型数据 |
| S1 | passed | 21,961 episodes；exact `0`、reverse-relevant `19,704`、generator-reference `2,452`；denylist overlap `0` |
| S2 | passed | task10/task20 full-state replay；IPEC/Plus schema/video sample；新增下载 `0` bytes |
| S3 | passed | 三族 reverse BDDL adapter，predicate nontrivial，official Track3 claim=false |
| S4/S5 | stopped | support 单次失败；GPU3 postflight `6 MiB` |

S2 的 HDF5 replay 对齐为 `states[0]` restore 后执行 `actions[1:]`，报告中已明确记录该事实。IPEC/Plus 仅作为 schema/reference，未越权升级身份。

## 边界与后续

- 所有自动生成候选都保持 `generator_reference_only`，未进入训练候选池。
- 没有创建 `accepted/`，没有 human review sidecar 或 promotion。
- GPU0/1/2 未被本轮执行触碰；GPU3 只用于已授权 simulator/replay，收尾仍通过只读门禁。
- 没有训练、norm 替换、官方 GPU、FAST、ZIP、上传或比赛提交。
- 本报告不授权 retry；下一步只能由审核决定最小 remediation 或是否另开人工兜底计划。
