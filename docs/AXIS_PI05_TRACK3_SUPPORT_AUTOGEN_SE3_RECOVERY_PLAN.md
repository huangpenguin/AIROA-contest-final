# AXIS Pi0.5 Track 3：Support 自动生成 SE(3) 最小恢复计划

> 制定日期：2026-09-03（Asia/Tokyo）
> 输入执行 commit：root `73f675477a41a80241a7549f12f991187ecaaa25`，nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`
> 当前状态：`track3_support_autogen_se3_recovery_planned`
> 唯一授权：执行 G0–G4，最多重新执行 1 条 `support_remove_task_10` canary；随后只做 G5 收尾并 hard stop
> 不授权：containment/articulation、人工 Teleop、批量生成、训练、norm、FAST/ZIP、官方 GPU、上传或比赛提交

## 0. 目标

本计划不从 Axis init 重训，也不开始 Track3 LoRA。当前瓶颈在可训练 reverse demonstration 为零，而且旧 S3 与自动执行器门禁不完整。先完成三件最小事项：

```text
统计/语义证据闭合
  → 从公开成功 HDF5 提取真实 object→EEF 6D 抓取位姿
  → 一次 support-only 自动 canary + 3/3 replay
```

若这条 support canary 仍失败，下一次审核再决定使用 Behavior Prompting 的 planner、恢复单条人工 reference，或停止自动生成。不得在本计划内边失败边调参。

## 1. 固定事实与旧证据

- 正式用户报告：R6 FAST Track1=`0.47`、Track2=`0.33`、Track3=`0`。
- 当前 exact reverse / pending / accepted episodes 均为 `0`。
- S0 固定的 LIBERO-Infinity、Behavior Prompting、LIBERO、LIBERO-Gen revisions 与 license/schema 证据保留；本轮不联网下载新数据。
- S2 可保留为 selective replay/schema evidence，但 HDF5 state/action alignment 在用于位姿挖掘前需新增逐步数值核验。
- 旧 `s1_semantic_mining.json`、`coverage_matrix.json`、`s3_reverse_adapter.json`、三个 v1 BDDL 和 support 失败资产全部 immutable，不原地改写。
- support 失败已表现为 descend error plateau 与接触数增加；仅增加 timeout 不获授权。

## 2. G0：不可变证据锁（CPU only）

输出：

```text
data/reports/axis_libero_30k/track3_support_autogen_se3_recovery/g0_release_lock.json
```

必须锁定：

1. root/nested commit 与 clean 状态；`LIBERO-plus` 既有 dirty porcelain 只读快照；
2. 旧 S0–S5 小型证据的 SHA-256；
3. support 失败的 metadata、trajectory、start/end state、front/wrist H.264 hash；
4. public denylist hash、P1 task10 snapshot/hash、原始 task10 HDF5/hash；
5. `pending_autogen=0`、`accepted=0`、无训练/FAST/ZIP/官方 GPU。

任一旧证据 hash 不匹配、root/nested dirty，或既有 pending/accepted 非零，状态置为 `track3_support_recovery_input_drift` 并停止。不得 reset/stash/修改 `LIBERO-plus`。

## 3. G1：S1/S3 证据闭合

### G1.1 S1 v2 分层计数

修复 mining 工具，但输出到新 v2 namespace：

```text
data/manifests/track3_public_primitives_v2/task_catalog.jsonl
data/manifests/track3_public_primitives_v2/episode_catalog.jsonl
data/manifests/track3_public_primitives_v2/coverage_matrix.json
data/reports/axis_libero_30k/track3_support_autogen_se3_recovery/g1_s1_accounting_v2.json
```

必须分别报告：

- `episode_identity_counts`，和必须严格等于 `episode_count=21,961`；
- `task_identity_counts`，和必须严格等于 `task_row_count=195`；
- episode/task 的 family counts 分开；
- 每一行恰好一个 identity，未知/缺失/重复 identity fail closed；
- exact reverse episode 仍为 `0`，不得通过文本关键词升级；
- deterministic rerun 的 catalog/hash exact，denylist overlap=`0`。

不得再使用“只检查三个 key 是否存在”的门禁。

### G1.2 S3 v2 自然语言

新增 typed language renderer，不直接把 BDDL identifier `_1`、`_region`、`contain`、`target_object` 拼进句子。三条 canary 的最低语义形式为：

```text
support:      remove the akita black bowl from the plate and place it back at its original position on the table
containment:  take the orange juice out of the basket and place it back at its original position on the floor
articulation: close the middle drawer of the wooden cabinet
```

具体冠词/大小写可规范化，但 object、source relation、destination relation 必须与 BDDL predicates 一致。输出 v2 BDDL/spec；保留 v1 immutable。静态测试至少拒绝 internal token、错误自指代、空 destination 和对象/区域类型错配。

### G1.3 S3 v2 动态门禁（GPU3 only）

对 support/containment/articulation 三个公开 spec 各运行 3 个 fresh process：

1. reverse initial predicates at reset 为真；reverse goal 为假；
2. 静置 20 simulator steps，目标/非目标 object、fixture joint 与 robot state 不发生超阈值漂移；
3. 用同一公开任务的 forward initial full-state 验证 reverse goal=true；
4. 用 forward terminal/P1 snapshot 验证 reverse goal=false at reset 且 reverse initial=true；
5. 对受控破坏状态验证 predicate 能从 true 变 false，不是恒真；
6. 每次记录 BDDL/XML/controller/state hash，不得只报告 boolean 汇总。

任何一族不是 `3/3`，或受控 true/false 不成立，状态为 `track3_reverse_semantic_adapter_v2_failed_<family>` 并停止；不得进入 G2。

## 4. G2：support 失败根因与公开抓取位姿挖掘

仅使用本地 task10 原始成功 HDF5 和已保存失败 state；不执行新的 reverse canary。

### G2.1 HDF5 alignment

至少抽取 task10 的 3 条公开成功 demos。fresh process 恢复 `states[0]`，分别检验候选 offset 约定，逐步比较 replay 后 simulator state 与记录 state；报告 qpos/qvel/object pose 最大/分位误差及最终 predicate。只有一个 alignment 约定通过预注册数值门限并保持 3/3 success，才可进入位姿挖掘。

不得仅以“最终也成功”证明 `states[0] + actions[1:]` 是 exact alignment。

### G2.2 失败轨迹重建

逐帧 restore 已保存 state，不施加新动作，导出：

- EEF position/quaternion、gripper site/geom position；
- bowl 与 plate 的 body pose、AABB/关键 site；
- target position/quaternion、position/orientation error；
- contact pair 的 geom/body 名称、penetration distance/force（可获得时）；
- bowl/plate/non-target drift；
- stage、action、saturation 与 simulator time。

必须证明或否定：`object center + 0.025 m` 是否把 EEF 目标放入不可达/碰撞几何。只看总 contact count 不足以宣布根因。

### G2.3 成功 demo 的相对 SE(3) 模板

用 gripper transition、object-follow 与 lift 共同检测每条 demo 的 grasp interval，提取：

```text
T_object_to_eef at pregrasp / grasp / lift
EEF orientation and approach axis
gripper close timing
clearance and contact pairs
```

至少 3 条 demo 得到一致模板；以稳健中位数和离散度报告，禁止人工盲填 z offset。若模板跨 demo 不一致、alignment 不通过或 contact 根因仍不清楚，状态为 `track3_support_grasp_geometry_unresolved` 并停止。

## 5. G3：SE(3) 执行器最小修复

在不改变环境/controller/norm 的前提下：

1. 目标由 object pose 与公开 demo 的 `T_object_to_eef` 组合得到；pregrasp 沿参考 approach axis 留 clearance；
2. 控制完整 6D OSC_POSE：xyz + shortest-path axis-angle orientation error；所有维度 finite、clip，并记录 saturation；
3. 抓取后用“夹爪闭合 + bowl 相对 EEF 保持 + lift”验证 grasp，不能仅靠固定步数；
4. 接触采用名称级 whitelist/abort；与 plate/桌面/非目标物的异常接触立即失败并保留资产；
5. trajectory 增加 current/target EEF pose、object/target pose、orientation error、contact pair、sim time；
6. `control_freq=20` 与 wall-clock throughput 分栏。离线生成可以慢于实时，不能把视频编码时间当控制频率；`/act <10s` 是部署门禁，不适用于本阶段；
7. 所有写入 atomic，failure 不进入 pending，success 也只进入 `pending_autogen_v2/`。

CPU tests 必须覆盖 S1 和数、typed language、quaternion/axis-angle、frame transform、grasp detector、contact abort、timeout、atomic failure write、denylist 与旧 evidence immutability。随后 GPU3 做一次无接触小幅 ±xyz/±rotation controller sign/gain canary；任一轴方向错误或状态漂移即停止，不得尝试 support。

## 6. G4：唯一一次 support canary（GPU3 only）

固定 `support_remove_task_10`、固定 P1 start-state、固定 v2 BDDL/spec、固定由 G2 得到的 SE(3) 模板。只允许一次真实执行，不允许参数网格、失败后 retry 或临时改阈值。

成功门禁：

1. reset goal=false，最终 goal 连续稳定满足；
2. 20 Hz simulator control、7D finite actions、无 gripper convention 错误；
3. grasp/lift/transport/place/release 各 subgoal 有数值证据；
4. contact whitelist、非目标对象漂移、collision、drop、saturation 通过预注册阈值；
5. 保存完整 state/action/telemetry 与 front/wrist H.264；
6. 从实际产物目录 fresh-process action replay `3/3`；
7. denylist overlap=`0`，identity 只能是 `exact_reverse_demo_pending_review`；
8. 进入 `pending_autogen_v2/`，不得 promotion 到 accepted。

通过状态：

```text
track3_support_autogen_v2_pending_human_video_review
```

失败状态：

```text
track3_support_autogen_v2_failed_with_geometry_diagnostics
```

两种结果都必须立即 hard stop；不得继续 containment/articulation 或训练。

## 7. G5：收尾（仅记录，不扩展授权）

执行者更新 plan index、handoff Markdown 与 machine JSON；写中文执行报告，明确 G0–G4 哪一步停止。小型 code/config/report/test 纳入 Git，视频/NPZ/state 保持 ignored；运行：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. .venv_axis_pi05/bin/pytest -q tests
git diff --check
```

同时检查 JSON/YAML、`py_compile`、root/nested clean，并形成 clean commit。不得 push、上传或比赛提交。

## 8. 本计划之后（不授权）

- support pending 通过人工视频审核后，才可把同一 SE(3) 方案迁移到 containment；articulation 单独设计 fixture joint/contact controller。
- 三族各至少一条 verified pending/accepted 后，另建 30–60 条扩充与 template-level holdout 计划。
- 只有 exact reverse 数据、20 Hz mixture manifest 与 fresh QUANTILES 通过，才从 R6 step500 分叉 25→100→250→500-step LoRA；不从 Axis init 重来。
- Track1/2 retention 与 unseen reverse template 必须联合选优；Track3 hidden score不得被当作逐 checkpoint 调参集。
