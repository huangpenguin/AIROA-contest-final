# AXIS Pi0.5 Track 3：Matched Terminal-State Release 计划

> 制定日期：2026-09-04（Asia/Tokyo）
> 输入 commit：root `8490f8dde1148aec0aec4d3d105082b44cfbb286`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`
> 当前状态：`track3_matched_terminal_state_release_planned`
> 唯一授权：L0–L5；随后 hard stop
> 不授权：修改官方 HDF5、生成 action/demo、Teleop、SE(3)/grasp、批量合成、训练/norm、FAST/ZIP、官方 GPU、上传或比赛提交

## 0. 目标

停止随机重跑已发生 drawer↔bowl 穿透的本地 reverse-BDDL native init。修复 K3 观测工具后，使用公开 LIBERO HDF5 中同一 demo 的 `model_file`、full state 和成功后缀构造 exact-model matched reverse-start 候选，完成三族可复现、可渲染、碰撞可审计的初态 release。

新产物只能写入：

```text
data/reports/axis_libero_30k/track3_matched_terminal_state_release/
```

旧 K0–K4、P1 snapshots、HDF5 与 reverse BDDL 均只读。

## 1. L0：输入锁与 K3 errata（CPU/filesystem only）

锁定 `8490f8d`、`cb8e898`、K0 manifest、K1 contract、K2 summary、K3 raw canary、K4 summary、三份公开 HDF5及其 SHA-256。输出 `l0_input_lock.json` 与 `L0_K3_ERRATA_ZH.md`，必须明确：

- K3 contact 的 step/body pair/penetration 与 background transition；
- renderer 为 wrapper telemetry 误判，但 camera frames 确实为 `21/21`；
- H.264 是临时扩展名错误；
- sensitivity raw 为 `Open true→true→true`，mutation sign/target错误；
- K3 没有显式 seed，旧 `no_seed_change` 只保留为历史字段；
- 旧报告不原地修改或重标。

同时预注册 HDF5 selection：task10/task20/task19 各按 `demo_0,demo_1,demo_2` 固定顺序；每个 demo 选择 reward 首次进入并持续成功后缀的第一份 state，若 schema 不支持则 fail-closed，不得换 demo 补位。

## 2. L1：观测与 contact contract 修复（CPU/code gate）

新增独立工具和测试，不修改旧 K3 raw evidence。

1. renderer schema从 wrapper 与 underlying env 双层解析，并同时要求 observation 中两路 uint8 camera key存在；构造参数、实际 flags和实际 frames分栏记录。
2. H.264 使用 `name.partial.mp4` + atomic rename；ffprobe 必须确认 codec=`h264`、20 FPS、预期帧数、非零尺寸。
3. task19 sensitivity从 public WoodenCabinet articulation metadata读取 close range，设置 qpos 为合法 close range 中点（绝对值，不是相对负 delta）；要求 `Open true / Close false → Open false / Close true → exact restore Open true / Close false`，全程不发 action。
4. contact contract升为 v2：source↔receiver 合法接触保留；operator target/subject 与无关 background 的接触无论是否已存在于 step5 baseline都必须单列。penetration、body/geom pair和持续步数不能被 baseline grandfather 隐藏。
5. tri-state、failure reason、seed/model/state identity均 fail-closed；passed gate不得携带失败 reason。

单测必须复现旧三个工具缺陷，并证明修复后可在合成小帧、公开 articulation range 与 synthetic contact matrix上通过/拒绝正确。L1不得启动 simulator/GPU。

## 3. L2：exact-model terminal-state CPU release

仅使用 L0 预注册的 9 个 `(task, demo)`；每次 fresh process：

1. 从该 demo attr读取 `model_file` 并记录 SHA-256；state必须来自同 demo，禁止跨 XML restore。
2. 在 public forward env 中 `reset_from_xml_string(model_file)` 后恢复候选 state，确认 forward goal成功；再在对应 reverse-goal env加载同一 XML/state。
3. 运行 5 warm-up + step5–20 的 7D zero hold，gripper只保持 terminal sign；记录 critical source每步 true、reverse goal每步 false、robot/participant、逐 body/geom contact、background truth与位姿。
4. hard clean gate：schema/finite/model/state identity exact、critical/goal通过、无 operator target/subject↔无关 background contact、无新 background contact、无超过 `2 mm` translation 的 background predicate退化。
5. native reverse-BDDL K3 collision只作旧诊断，不能作为 matched source oracle。

必须三族各 `3/3`；少一条即状态 `track3_matched_terminal_state_cpu_failed_<family>`，直接进入 L5。不得改选择规则或换 demo。

## 4. L3：GPU3 task19 exact-model camera canary

仅在 L0–L2 全通过、root/OpenPI clean、GPU3 自然空闲时执行；GPU0/1/2禁止。使用已预注册的 task19/demo_0 model XML + state，真实 EGL、20 Hz、21 frames：

- 完整运行 L1 renderer/contact/predicate/sensitivity gates；
- 两路 H.264 + ffprobe通过；
- 输出逐帧 JSON、视频 hash、GPU UUID/PID/peak；
- 不发 manipulation action，不 retry、不换 demo/seed。

失败即进入 L5。

## 5. L4：三族最小 camera release

L3通过后，仅额外运行 task10/demo_0 与 task20/demo_0；task19沿用L3，共三族各1条。要求三条全部满足 L2/L3 gates、视频完整、worker exception=0。不得运行其余 6 条 GPU副本，也不得生成 demo。

唯一成功状态：

```text
track3_matched_terminal_state_dynamic_validated_pending_executor_review
```

该状态只说明初态源可用于下一轮设计，不授权自动执行、Teleop或训练。

## 6. L5：收尾与 hard stop

无论成功或失败：

- 输出 `summary.json`、逐 demo manifest、contact ledger、`L5_CLOSEOUT_REPORT_ZH.md`；未执行项写 `not_evaluated`；
- 保留旧 K3 mixed failure 与新 matched-source结果并列，不覆盖历史；
- 重验 HDF5/K0–K4 hashes、reverse-demo `0/0/0`、root/OpenPI clean、`LIBERO-plus`未触碰；
- 运行 targeted/full tests、JSON parse、`py_compile`、`git diff --check`；
- 更新计划索引及两份交接，形成 clean commit并 hard stop。

## 7. 禁止项

- 不得通过换 demo、扩大 demo 集、改阈值、延长 settle 或忽略 contact 来补失败；
- 不得原地修改旧 K3/K4、官方 HDF5、public BDDL 或已有 snapshots；
- 不得创建 exact/pending/accepted reverse demo，不得运行 action/SE(3)/grasp/Teleop；
- 不得训练 LoRA/dense、计算新 norm、导出 FAST/ZIP、使用官方 GPU、上传或提交。
