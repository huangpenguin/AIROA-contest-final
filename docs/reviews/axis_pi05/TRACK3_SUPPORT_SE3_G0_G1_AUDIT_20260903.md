# Track 3 Support SE(3) G0–G1 独立审核

> 审核日期：2026-09-03（Asia/Tokyo）
> 审核输入：root `179045a6d7a7998bf1a54452ef79739c9383c09f`，parent `8e9883cb748ecf15e5ff1fb988e2707e02dd153e`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`
> 新状态：`track3_dynamic_gate_model_state_reconciliation_planned`

## 1. 获批结果

- G0 输入锁与 hard stop 均执行正确；没有进入 G2/G3/G4，没有生成 pending/accepted demo，没有训练、FAST/ZIP、官方 GPU、上传或比赛提交。
- S1 v2 通过：task rows=`195`、episode rows=`21,961`，两层 identity sum 分别严格相等；exact reverse task/episode 均为 `0`，denylist overlap=`0`。
- S3 v2 typed language 通过，三条 instruction 不再含内部 identifier，并保留 object/source/destination 语义。
- 三族各 `3/3` fresh process 确实运行，共 9 次；HDF5 time/qpos offset 修正为 time scalar 位于 index 0、qpos 从 index 1 开始是正确修复。
- root 与 nested OpenPI clean；`LIBERO-plus` 的既有 dirty 状态未被修改；GPU0 未触碰，GPU3 已释放。测试为 `170 passed, 7 skipped, 23 warnings`。

因此，G0、S1 v2、typed language 与执行边界均可作为后续 immutable evidence；不需要重做。

## 2. G1.3 结论为何不能批准

### 2.1 HDF5 state 被跨 model/XML 使用

`dynamic_worker()` 从官方 HDF5 读取 `data/demo_0/states[0]`，随后直接送入由 BDDL 新建的 reverse environment。MuJoCo state 只携带 time/qpos/qvel；HDF5 的 exact model 则另存在 demo 的 `model_file` attribute 中。三份 HDF5 的 `model_file` SHA-256 分别为：

```text
task10  e9a508e1e98a7bd2631796a4c41476a449fd9821fb45837a27d2ebefd87c8830
task19  ccf958ea89aa0e84e32cb90b99e31faf66f3cd5e33082f7d2a1d4444c4660257
task20  4e0497a77437ac5846c84c6bea8e08a9af66b11829f8edbe8bc692616a1be723
```

它们均不等于 G1 新建 reverse env 的 XML hash。task10/task19 的三个 fresh run 甚至各自产生三个不同 XML hash，说明随机化后的 body/site/fixture model 不能由相同 qpos 向量完整重建。故 task10/task19 的 `states[0]` reverse goal=false 不能证明 reverse predicate 错误，只能证明“跨 model state 不是合法 goal oracle”。task20 恰好为 true 也不能使该方法有效。

此外，`reverse_initial_at_p1` 实际赋值为 forward env 的 `forward_terminal_success`，并未在 reverse env 中逐条求值 reverse initial predicates；字段名与验证内容不一致。

### 2.2 robot settle 是异量纲且被 action 主动扰动

旧 gate 对整个 `robot0_proprio-state` 取单一最大绝对差并与 `0.05` 比较。该向量包含 joint sin/cos、joint velocity、EEF position/quaternion、gripper qpos/qvel，不能共用一个物理阈值。代码还在 reset controller goal 后再次调用 `set_init_state()`，然后连续 20 步发送 xyz/rotation 全零但 `gripper=-1` 的 action；这会主动打开/改变夹爪，并使 controller goal 与 restored state 的顺序存在歧义。

实际 scene 证据与“环境整体失稳”不符：

```text
family                 object drift (m)   fixture drift       raw proprio max
support/task10         1.39e-7            1.55e-8             0.1307
containment/task20     1.20e-11           5.27e-12            0.1389
articulation/task19    8.71e-10           1.75e-10            0.5144
```

目标与非目标场景元素稳定到远低于旧门限的量级。现有失败更符合 validator 定义错误，而不是三族 reverse BDDL 同时发生动力学失稳。

## 3. 审核裁决

1. 保留 `g1_s3_dynamic_gates.json` 为历史失败证据，不改写、不放宽旧阈值，也不把它改标为 pass。
2. 撤销“`track3_reverse_semantic_adapter_v2_failed_support_remove` 已证明 reverse semantics 失败”的解释；正确解释是 `dynamic validator inconclusive/invalid for semantic release`。
3. 不批准原计划 G2–G4；先执行新的 model/state reconciliation 计划。
4. 暂不回退到人工 Teleop，也不开始 LoRA。当前最短路径是修复验证器并直接使用 reverse BDDL 自身的 native reset；若该路径通过，robot 初态也比移植 forward terminal 更适合生成训练轨迹。

## 4. 不授权事项

本审核不授权 HDF5 SE(3) 提取、旧失败重建、执行器修改、support retry、containment/articulation canary、人工 Teleop、批量生成、训练、norm、FAST/ZIP、官方 GPU、上传或比赛提交。授权边界仅见新的唯一活动计划。

## 5. 审核验证

- 定向测试：`11 passed, 2 warnings`
- 全量测试：`171 passed, 7 skipped, 23 warnings`
- machine JSON parse 与 `git diff --check`：通过
