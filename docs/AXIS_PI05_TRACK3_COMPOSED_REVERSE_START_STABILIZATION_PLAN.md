# AXIS Pi0.5 Track 3：Composed Reverse-Start Stabilization 计划

> 制定日期：2026-09-04（Asia/Tokyo）
> 审核输入：root closeout `0dc402e2d70e6576c5861a674e0aac04e4427ef0`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`
> 当前状态：`track3_composed_reverse_start_stabilization_planned`
> 唯一授权：M0–M5；随后 hard stop
> 不授权：修改公开 HDF5/BDDL、生成 manipulation action 或 reverse demo、SE(3)/grasp、Teleop、批量合成、训练/norm、FAST/ZIP、官方 GPU、上传或比赛提交

## 0. 目标与核心假设

旧 L2 已证明 9/9 exact terminal states 的 reverse source 成立、reverse goal 未成立，但 task20 的物体仍在成功后沉降，task19 的 forward terminal robot 尚在回稳。所有 demo 的 positive reward suffix 只有一帧，公开 HDF5 没有可直接选取的 post-success dwell。

本计划只测试一个可证伪假设：reverse scene 应继承同 demo 成功终态中的 object/fixture configuration，而 robot/gripper 应恢复为同 demo 初态的默认姿态；这比把 forward 操作结束时的 robot residual state 当作新任务初态更接近标准 LIBERO reset。该假设不声称揭示 hidden Track 3 的精确初态分布。

新产物只能写入：

```text
data/reports/axis_libero_30k/track3_composed_reverse_start_stabilization/
```

旧 HDF5、L0–L5 raw、K0–K4、reverse BDDL 与所有历史失败证据只读。

## 1. M0：审核闭包与不可变输入锁（CPU/filesystem only）

1. 锁定 root `0dc402e`、execution `19cb1ea`、nested OpenPI、归档 L0–L5 plan、L0 manifest、L2 summary、9 个 raw report 与 contact ledger 的 bytes/SHA-256。
2. 重新验证 9/9 same-demo identity、forward goal、reverse critical source与reverse goal；输出逐 run 的失败 gate 分解，不得只保留首个失败 family。
3. 锁定三份 HDF5 中每个固定 demo 的 `model_file`、`init_state`、terminal state、state/action/reward shape与hash；确认 `states[0] == init_state`，reward positive suffix仍为一帧。
4. 输出 `m0_release_lock.json` 与 `M0_L2_DIAGNOSIS_ZH.md`。本阶段不得 import simulator、不得使用 GPU。

任一历史 hash、固定 demo identity 或计数不一致即状态 `track3_composed_reverse_start_m0_input_drift`，直接进入 M5。

## 2. M1：named state-composition 与 contact-v3 contract（CPU/code only）

新增独立工具与测试，不修改旧 `track3_matched_terminal_state_release.py` 或旧报告。

### 2.1 MuJoCo state ownership ledger

实现一个接收MuJoCo model metadata的纯函数，通过 `model.jnt_qposadr`、joint type、`model.jnt_dofadr`、`nq/nv/na` 解码 flattened state；不能只依赖硬编码 offset。M1用synthetic metadata测试，真实同demo XML/model ledger只在M2 fresh worker内生成。输出每个 joint 的：

- joint id/name/type；
- qpos half-open range；
- dof/qvel half-open range；
- ownership：`robot_arm`、`robot_gripper`、`operator_scene`、`receiver_scene` 或 `background_scene`；
- source：`demo_init`、`demo_terminal` 或 `zero`。

所有 qpos/qvel index 必须恰好被分类一次；overlap、gap、未知 robot/gripper joint、shape变化或 opaque trailing state 未说明均 fail-closed。

### 2.2 固定 composition contract

对每个固定 demo 构造状态，不得优化或按结果选择：

- `time` 与 actuator/opaque trailing fields：取同 demo `init_state`；
- robot arm 与 gripper qpos：取同 demo `init_state`；
- 所有 scene/object/fixture qpos：取同 demo terminal success state；
- 所有 qvel：写为 exact zero；
- XML：仍为同 demo `model_file`；
- controller：在 restore+`sim.forward()` 后按固定 `update(force=True) → reset_goal()` 顺序初始化；
- hold action：固定 `float32[7]` 全零，包括 neutral gripper `0`，禁止继续发送旧 terminal sign `-1`。

必须保存 component source ledger、逐 index source bitmap、input/output hashes 和 round-trip reconstruction diff。composition 不是“exact terminal full-state”，报告中必须使用 `composed_state`，禁止复用旧 identity 标签。

### 2.3 contact-v3

将 contact 明确分为：

1. operator↔intended receiver/support：按 task contract允许且记录；
2. operator↔robot/gripper：`demonstrator_contact`，不是 background；
3. operator↔unrelated scene background：永远 hard fail，不得 baseline grandfather；
4. background↔background：记录新增 pair 与 penetration。

composed t0 与正式 measurement window 中第 2 类必须为零；第 3 类任何一步都必须为零。task10 的 bowl↔plate 是 required support contact；task20 的 orange juice↔basket contact只允许并记录，`In` predicate才是containment source oracle；task19 不要求 operator receiver contact。

单测至少覆盖：joint range overlap/gap/unknown fail-closed、free joint 7-qpos/6-dof mapping、robot/init与scene/terminal逐 index provenance、qvel exact zero、neutral 7D hold、gripper contact不再误标 background、真实 unrelated background contact仍拒绝、passed gate无失败 reason。M1 不得启动 simulator/GPU。

## 3. M2：task20/demo0 单条 CPU canary

只有 M0/M1 全通过才运行一次 fresh CPU/no-renderer process。固定 task20/demo0，不得 retry、换 demo、seed 或阈值。

1. 同 XML 分别验证 raw terminal state 的 public forward goal为真，以及 composed state 的 public forward goal仍为真。
2. 在 reverse env restore composed state；t0 必须 critical source=true、reverse goal=false、operator↔robot/gripper contact=0、operator↔unrelated background contact=0。
3. controller reset 后运行固定 20-step burn-in，再运行固定 20-step measurement；40步均用全零 7D hold。不得延长 burn-in 或“等到通过”。
4. 全程记录 full predicate truth/detail、named object/fixture/robot pose、qpos/qvel、EEF、逐 geom/body contact和finite/schema。
5. burn-in 全程 source=true、goal=false；measurement hard gate沿用旧 L2 已预注册阈值：operator participant translation `<=2 mm`、robot arm qpos `<=0.002 rad`、qvel `<=0.02 rad/s`、EEF translation `<=2 mm`、EEF rotation `<=0.02 rad`。operator subject自由旋转只记录，不以统一 rotation阈值否决。
6. measurement 内 contact-v3 第2/3类均为0，background不得出现 `true→false` 或回摆；intended source relation必须持续成立。
7. step20 settled composed state可写入本报告 namespace作为 `start_state_candidate`，但不得写入 dataset、`pending/` 或 `accepted/`，不得计为 reverse demo。

失败即状态 `track3_composed_reverse_start_cpu_canary_failed_task20_demo0`，直接进入 M5。通过才允许 M3。

## 4. M3：固定 9 条 CPU matrix

复用 M2 的 task20/demo0结果，不重跑；其余 8 条按 `task10→task20→task19`、每族 `demo_0→1→2` 顺序 fresh process运行。composition、20+20窗口和所有 gate 与 M2 完全相同。

必须三族各 `3/3`，并额外满足：

- 9/9 input/output/provenance hash闭合；
- 9/9 public forward goal在 raw terminal及composed state均为真；
- 9/9 reverse source全程为真、goal全程为假；
- 9/9 measurement contact-v3与稳定性通过；
- worker exception为非零return code，completed pass/fail gate为零return code；
- exact/pending/accepted reverse demo仍为`0/0/0`。

任一失败即状态 `track3_composed_reverse_start_cpu_matrix_failed_<family>_<demo>`，不得换样本或放宽门限，直接进入 M5。

## 5. M4：GPU3 三族 camera canary

仅在 M3 `9/9`、root/OpenPI clean且GPU3自然空闲时执行；GPU0/1/2禁止。固定 task10/demo0、task20/demo0、task19/demo0，各最多一次 fresh EGL process；顺序固定 task10→task20→task19。

- 复现同一 composition及20+20 dynamics gates；
- 显式记录 seed、GPU UUID/PID/peak、wrapper与underlying renderer flags；
- agentview与wrist各41帧 uint8，写 `name.partial.mp4` 后 atomic rename；
- ffprobe要求H.264、20 FPS、41帧、非零尺寸；
- JSON/video/composed-state hash闭合，无 `.partial` 残留；
- 不发 manipulation action，不 retry、不换 demo/seed。

任一失败立即停止剩余 canary并进入 M5。三条均通过时状态为 `track3_composed_reverse_start_validated_pending_executor_review`，仍不得进入 executor/demo/训练。

## 6. M5：收尾与人工审核 hard stop

无论在哪个门禁停止，都必须：

1. 生成 `summary.json`、逐 run index、contact ledger与中文 closeout；`passed/failed/not_evaluated`严格区分。
2. 复核 M0 immutable hashes、固定 demo、composition source bitmap、start-state candidate count，以及 reverse demo `exact/pending/accepted=0/0/0`。
3. 记录 root/OpenPI/`LIBERO-plus` porcelain、GPU资源、执行/未执行阶段、所有禁止项。
4. 运行 targeted tests、`PYTHONPATH=. JAX_PLATFORMS=cpu .venv_axis_pi05/bin/pytest -q tests`、JSON parse、`py_compile`、`git diff --check`。
5. 更新 plan index、Markdown/JSON handoff并 clean commit；随后 hard stop等待人工审核。

即使 M4 全通过，下一步也只能由新的审核计划授权一个 operator executor canary；不得从本计划直接生成 demo、启动 LoRA/full-FT 或制作 ZIP。
