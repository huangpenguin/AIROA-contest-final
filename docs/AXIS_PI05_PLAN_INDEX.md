# AXIS Pi0.5 计划索引与审核生命周期

> 更新日期：2026-09-04（Asia/Tokyo）
> 当前唯一活动计划：`docs/AXIS_PI05_TRACK3_MATCHED_TERMINAL_STATE_RELEASE_PLAN.md`
> 当前状态：`track3_matched_terminal_state_cpu_failed_containment_remove`
> 最新审核：`docs/reviews/axis_pi05/TRACK3_BACKGROUND_CONVERGENCE_K0_K4_AUDIT_20260904.md`

## 1. 本轮执行停止点

Init Stabilization I0–I4 已完成并停止。三族 reverse operator-critical source relation在固定 step5 成立并连续到 step20，reverse goal 同窗口连续为 false；I3 因旧 full-scene max gate 未通过而未运行。独立审核确认这个 failure 不能解释为 reverse init 语义失败：task20/task19 的超限值来自无关背景物，task10 是 bowl 在 plate 上接触落稳的自由旋转，9 个 reverse run 的完整 init/goal truth vector 均为零跃迁。

Operator-Scoped Stability Release的J0/J1/J2/J4已完成并获执行审核；J2新矩阵为reverse`8/9`、forward`6/6`，J3未执行。唯一跃迁已定位为task19 run1的background initializer `(On bowl region)`在step20由false变true；critical `Open`始终true、goal `Close`始终false，物理/contact门禁均通过。这是单调收敛，不是task19语义回退。

旧I2和J2 failure保持immutable。K0已将15个raw worker reports及manifest闭包进Git并完成fresh-checkout hash验证；K1/K2以“critical/goal严格、background init单调非递减”的新合同只读复算通过（reverse`9/9`、forward`6/6`）。K3仅运行一次GPU3 task19 canary即hard stop。独立审核确认这是一条混合失败：native reverse init 中 bowl↔middle drawer 从step5起已有约`-6.80 mm`穿透且background `On`在step20退化；renderer wrapper telemetry、H.264临时扩展名和task19 sensitivity mutation另有三个实现缺陷。三族matrix未执行，SE(3)、demo、Teleop和训练仍未授权。

G0/G1 已按失败门禁停止，S1 v2 与 typed language 获批准，三族 `3/3` fresh-process 运行及 hard stop 也真实完成；但 G1.3 的失败结论不能证明 reverse semantics 无效。旧 validator 把 HDF5 `states[0]` 恢复到不同/randomized XML 的 reverse env，以 forward success 冒充 reverse initial predicate 检查，并对包含 joint velocity、EEF quaternion、gripper qpos/qvel 的整段 proprio 使用一个 `0.05` 阈值；20 步 action 还固定发送 `gripper=-1`。相反，三族 object/fixture drift 仅约 `1e-7～1e-11`。

最新审核仍为 `remediation_required`，但 R2 已取得第一条有效结论：task10 reset 瞬间 7 条 reverse init 中 3 条 false，goal 正确 false；bowl 与 plate XY 完全一致而 Z 相差 `0.11 m`，符合 sampler 已放置但尚未形成 runtime contact。三族 R1 都至少有一个 background init atom=false；本地 LIBERO evaluator 会先执行 5 个零动作物理步。因此当前先校准 initializer-vs-runtime 和 step5 stabilization 语义，不能直接改 BDDL 或宣布 reverse start 无效。证据：`docs/reviews/axis_pi05/TRACK3_NATIVE_VALIDATOR_R0_R4_AUDIT_20260903.md`。

## 2. 当前计划与停止点

- `AXIS_PI05_TRACK3_MATCHED_TERMINAL_STATE_RELEASE_PLAN.md`
  - L0锁定K0–K4和三份公开HDF5，并以errata保留K3真实raw语义；
  - L1只修复wrapper renderer telemetry、H.264 atomic扩展名、task19 close-range sensitivity和contact contract v2；
  - L2固定使用task10/task20/task19各`demo_0/1/2`的同demo `model_file + terminal full-state`，CPU fresh-process要求三族各`3/3`；
  - L2通过后L3只在GPU3运行task19/demo_0，L4最多补task10/demo_0与task20/demo_0，共三族各一条camera release；
  - L5无论成败hard stop；不生成action/demo，不执行SE(3)、Teleop、训练、norm、FAST/ZIP、官方GPU、上传或提交。

任何执行者看到多个计划时，以本索引的“唯一活动计划”为准，不得从归档计划的未执行尾部自行恢复任务。

正式Track1 `0.47`与Track2 `0.33`已证明R6 FAST在官方20Hz闭环有效；Track3 `0`说明纯正向LIBERO LoRA并未自然获得逆向能力。当前优先级不是立即训练，而是先用公开完整simulator state建立合法、通用、可回放的reverse supervision；dense full-FT继续冻结为独立研究线。

## 3. 已归档的 R6 执行链

以下计划已完成其获授权阶段或已按 hard gate 停止，并被后续计划取代。文件保存在 `docs/archive/axis_pi05/`，只用于审计：

1. `AXIS_PI05_LORA_NAN_LOCALIZATION_AND_RECOVERY_PLAN.md`：R0–R6 完成，产生 R6 step-500。
2. `AXIS_PI05_POST_R6_LORA_SCALING_AND_SELECTION_PLAN.md`：执行至 N1，按 reproducibility 门禁停止。
3. `AXIS_PI05_N1_EVALUATOR_NONDETERMINISM_RECOVERY_PLAN.md`：执行至 D1，转入 policy-boundary 专项。
4. `AXIS_PI05_N1_POLICY_BOUNDARY_AND_DIAGNOSTIC_ZIP_PLAN.md`：执行至 Q2，raw-LoRA strict `1e-6` 失败。
5. `AXIS_PI05_R6_FOLDED_DEPLOYMENT_AND_DIAGNOSTIC_SUBMISSION_PLAN.md`：执行至 S1.3，folded real-EGL strict `1e-6` 失败。
6. `AXIS_PI05_R6_EGL_STATISTICAL_RELEASE_PLAN.md`：E0 已完成首次审核；E1–E7 从未获本轮执行授权。
7. `AXIS_PI05_E0_CLOSURE_BEFORE_E1_PLAN.md`：C0–C5 完成并通过独立审核，正式释放 E1。
8. `AXIS_PI05_E1_EGL_ACTION_ENVELOPE_PLAN.md`：完成 lock 与 JAX sentinel；E1.3 `microbatch=4` OOM 后按 hard gate 停止，E1.4/E1.5 未执行。
9. `AXIS_PI05_E1_JAX_MB1_RECOVERY_PLAN.md`：完成 atomic shard/resume 与 16/16 canary；旧 bitwise gate失败，340/340未启动。
10. `AXIS_PI05_R6_SAME_DAY_DIAGNOSTIC_ZIP_PLAN.md`：在执行前被更短的当晚正分路径取代；其中 JAX 340、完整 envelope/C20 不进入今晚任务。
11. `AXIS_PI05_POSITIVE_SCORE_TONIGHT_PLAN.md`：P0–P4完成，生成唯一获批进入官方 pre-check 的 R6 Track 1 FAST ZIP；未上传/提交。
12. `AXIS_PI05_TRACK1_OFFICIAL_PRECHECK_AND_SUBMISSION_PLAN.md`：因用户已人工提交并决定保留配布GPU给训练而在执行前取代；O0–O3均未执行。
13. `AXIS_PI05_TRACK1_SUBMITTED_SCORE_AND_LOCAL_DOCKER_PLAN.md`：用户随后报告正式Track1=`0.47`，真实scorer证据取代本地Docker任务；T0–T2未执行。
14. `AXIS_PI05_POST_SCORE_DENSE_FULLFT_PREFLIGHT_PLAN.md`：F0/F1完成；2卡和4卡均在`init_train_state` OOM，未进入optimizer update；原计划已按hard gate结束并转入sharded-init修复。
15. `AXIS_PI05_DENSE_FULLFT_SHARDED_INIT_RECOVERY_PLAN.md`：R3 direct-sharded init通过；R4在默认75% JAX allocator pool内的首轮累积计算OOM，未完成update/10-step；转入allocator-fraction最小复测。
16. `AXIS_PI05_DENSE_FULLFT_ALLOCATOR_FRACTION_RECOVERY_PLAN.md`：完成0.90/0.95 canary且均无optimizer update；后续审核发现两次均已完成microbatch 0/1，失败与完整gradient引用生命周期强相关，转入最小accumulation remediation。
17. `AXIS_PI05_DENSE_FULLFT_ACCUMULATION_MEMORY_LIFETIME_RECOVERY_PLAN.md`：M0/M1完成；M2在index1 compute完成后的普通gradient merge申请72MiB失败，未完成update/M3；转入donated accumulator最小修复。
18. `AXIS_PI05_DENSE_FULLFT_DONATED_ACCUMULATOR_RECOVERY_PLAN.md`：D2 donated merge和D3 step1通过；D4从step1完成update到step2后，在下一轮microbatch 1 compute OOM；转入跨optimizer step生命周期修复。
19. `AXIS_PI05_DENSE_FULLFT_INTERSTEP_STATE_LIFETIME_RECOVERY_PLAN.md`：L0语义/hash锁与L1生命周期CPU gate通过；GPU0被外部YOLO占用，L2–L4未启动；冻结成果并改道Track3正分优先。
20. `AXIS_PI05_TRACK3_REVERSE_LORA_POSITIVE_SCORE_PLAN.md`：T0资产锁通过；T1编译40个公开候选但真实20Hz pilot为0条，按门禁停止；LoRA及后续阶段均未执行，转入公开完整终态与teleop/planner恢复。
21. `AXIS_PI05_TRACK3_PUBLIC_TERMINAL_TELEOP_PLANNER_RECOVERY_PLAN.md`：P0锁定40个官方公开HDF5，P1三个family full-state canary通过，P2 recorder原型完成；独立审核发现recorder尚不可见/不可安全空闲/replay promotion不合格，转入最小修复。
22. `AXIS_PI05_TRACK3_TELEOP_RECORDER_REMEDIATION_PLAN.md`：R0–R4完成，修复action/reset、localhost preview与pending/replay/review promotion语义；GPU3 task19→10→20非人工canary通过，转入单条task19人工端到端canary。
23. `AXIS_PI05_TRACK3_TASK19_HUMAN_CANARY_PLAN.md`：在H0执行前被公开primitive挖掘与自动生成路线取代；没有人工candidate。recorder/replay实现保留为自动生成失败后的显式人工兜底，不再是活动授权。
24. `AXIS_PI05_TRACK3_PUBLIC_PRIMITIVE_MINING_AND_AUTOGEN_PLAN.md`：S0–S5 已执行；首条 support canary 在 descend 超时后正确停止。独立审核发现 S1 统计和 S3 动态/语言门禁未闭合，转入 support-only SE(3) 最小恢复。
25. `AXIS_PI05_TRACK3_SUPPORT_AUTOGEN_SE3_RECOVERY_PLAN.md`：G0、S1 v2、typed language 与三族动态运行完成；G1.3 按旧门禁停止。独立审核确认跨 model HDF5 state、proxy predicate 与异量纲 raw proprio settle 使结论无效，G2–G4 未授权，转入 model/state 对齐修复。
26. `AXIS_PI05_TRACK3_DYNAMIC_GATE_MODEL_STATE_RECONCILIATION_PLAN.md`：D0/D1 完成；D2 唯一 task10 process 因 `TargetZone` telemetry schema 异常而无有效动态结论，D3 未运行，D4 正确停止；转入 validator runtime/进程契约最小恢复。
27. `AXIS_PI05_TRACK3_NATIVE_REVERSE_VALIDATOR_RECOVERY_PLAN.md`：R1 三族 runtime schema 通过；R2 有效发现 reset 瞬间 task10 full init conjunction 非全真并停止，R3 未运行。后续先校准官方式 5-step stabilization 与 critical/background predicate 语义。
28. `AXIS_PI05_TRACK3_REVERSE_INIT_STABILIZATION_SEMANTICS_PLAN.md`：I0–I4完成；step5 critical relation三族通过，但旧 full-scene max stability gate失败，I3未运行。独立审核后转入 operator-scoped release contract。
29. `AXIS_PI05_TRACK3_OPERATOR_SCOPED_STABILITY_RELEASE_PLAN.md`：J0/J1/J2/J4完成；新reverse`8/9`，唯一失败为task19 background initializer在step20由false变true；J3未运行。独立审核后转入evidence closure与monotonic convergence contract。
30. `AXIS_PI05_TRACK3_BACKGROUND_PREDICATE_CONVERGENCE_RELEASE_PLAN.md`：K0–K2通过；K3单条task19 GPU3 canary发现native reverse init的drawer↔bowl穿透，同时暴露renderer/H.264/sensitivity工具缺陷；三族matrix未运行，转入exact-model matched terminal-state release。

归档不表示旧计划所有尾部阶段都完成；只表示该文件不再是活动授权入口。旧 failure、阈值、hash 和未执行阶段必须继续保留。

当前只执行 init stabilization 语义校准、step5 critical-relation 门禁，以及通过后最多 `1+9` 次 GPU3 非抓取动态验证；不得修改 BDDL、恢复旧 SE(3)/G2–G4、人工 task19、真实 support demo、批量生成、旧 reverse LoRA/interstep 训练或直接转向 Blackwell。

## 3.1 G0–G1 执行审核与门禁重构

G0 输入锁、S1 v2 分层统计和 S3 typed language 已完成并获批准。S3 动态门禁运行三族各 `3/3` fresh process，共 9 次，并正确按旧门禁停止；但 task10/task19 的 HDF5 `states[0]` 来自与 fresh reverse env 不同的 recorded model，不能作为 goal oracle。旧 robot drift 又混合异量纲字段，且固定 `gripper=-1` 主动改变状态；三族 object/fixture 实际稳定。

审核结论为 `remediation_required`，当前状态改为 `track3_reverse_init_stabilization_semantics_planned`。R1 validator/runtime 门禁已批准；R2 的 t=0 full-conjunction failure 真实有效，但只证明 reset 瞬间 contact/predicate 未全成立。新计划必须先按本地官方 evaluator 的 5 个 zero-action 语义验证 critical source relation，并修正早停时把未执行 gate 错写为 false 的报告问题。

## 3.2 Init Stabilization I0–I4 执行收尾（2026-09-04）

I0 锁定 root `ed0ec3bff61c09a213300181959491dd107bf778`、旧 R2/R4、三族 BDDL/source 与 zero reverse-demo counts；I1 的 operator-critical/background 分层、truth/detail 分层、`passed/failed/not_evaluated` 与固定 7D zero-action contract 通过。I0 immutable inputs 在 I4 重验全部保持不变。

I2 完成 `12/12` fresh CPU/no-renderer processes：task10 forward `3/3`、task10 reverse `3/3`、task20 reverse `3/3`、task19 reverse `3/3`。所有 schema、finite、detail 与 process return-code 通过；task10 forward 对照观察到 background initializer false。三族 reverse critical source 在固定 step5 成立并连续至 step20，reverse goal 同窗口连续为 false。

I2 named scene stability 未通过：support reverse 最大 rotation drift 约 `0.0564 rad`，containment reverse 最大 object translation drift 约 `0.0215 m`，articulation reverse 也出现超过既有阈值的 rotation/translation drift。因此 I3（GPU3 task10 canary 与三族 `3/3`）为 `0` 次，I4 已 fail-closed hard stop。`settle` 与 `controlled_destruction` 均记录为 `not_evaluated`，不是 false；exact/pending/accepted reverse demo 仍为 `0/0/0`。等待下一次审核，禁止自行进入 I3、SE(3)、demo、Teleop、批量生成或训练。

本轮机器证据：`data/reports/axis_libero_30k/track3_reverse_init_stabilization/` 下的 `i0_release_lock.json`、`i1_semantics.json`、`i2_stabilization_time_series.json`、`runs/`、`summary.json` 与 `I4_CLOSEOUT_REPORT_ZH.md`。

## 3.3 I0–I4 独立审核与 Operator-Scoped Release 授权（2026-09-04）

审核输入 root `0dec0b1930a3d2ad9f196febe1aeaabb0c89445d`、nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`。I0–I4 执行、12/12 fresh CPU processes、I3 `0`、资源边界与 hard stop 获批准；审核结论为 `remediation_required`，原因是旧 gate 的作用域，而不是 reverse semantics。

独立逐对象复算显示：support 的 `0.0564 rad` 来自 bowl 在 plate 上继续接触落稳，translation仍小于 `1 mm`且 `On` 连续；containment 的 `21.5 mm` 来自背景 salad dressing，orange juice仅约 `5.54e-7 m`；articulation 的超限来自背景 bowl，目标 drawer site/joint约 `1e-7`。9个 reverse runs 的完整 init/goal truth vector在 step5→20 均无跃迁。

旧 I2 failure 和全场诊断永久保留。唯一活动计划改为 `docs/AXIS_PI05_TRACK3_OPERATOR_SCOPED_STABILITY_RELEASE_PLAN.md`：冻结按算子角色划分的稳定性 contract，CPU复算既有证据并执行9条带完整contact telemetry的reverse、6条forward controls；通过后最多在GPU3执行`1+9`次production-like动态验证。仍不授权SE(3)、demo、Teleop、训练或比赛GPU。完整审核见 `docs/reviews/axis_pi05/TRACK3_INIT_STABILIZATION_I0_I4_AUDIT_20260904.md`。

## 3.4 J0–J4 独立审核与 Background Convergence 授权（2026-09-04）

审核输入execution `bc2419e62ff05a9584d1d6033d22b0e55d03a240`、record `f969393f3e16dc94985675083d09daa712cc7c04`、nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`。J0/J1/J2/J4执行、15个CPU process、J3`not_evaluated`、资源边界与hard stop获批准；结论为`remediation_required`。

15个raw reports的逐atom复算只有一次transition：task19 reverse run1的background initializer `(On akita_black_bowl_1 main_table_akita_black_bowl_region)`在step5–19为false、step20为true。critical `Open`连续true、goal `Close`连续false，drawer/robot/contact全部通过；没有任何critical/goal变化、background `true→false`或回摆。因此旧exact-vector `8/9`有效保留，但不能解释为reverse语义失败。

J2 summaries已经入Git，但15个必要raw reports被ignore，`git ls-files .../runs/*=0`；机器handoff current pointers也仍停在上一轮。唯一活动计划改为`docs/AXIS_PI05_TRACK3_BACKGROUND_PREDICATE_CONVERGENCE_RELEASE_PLAN.md`：K0先完成raw evidence closure，K1冻结background monotonic contract，K2只读复算；通过后K3最多GPU3 `1+9`次，K4 hard stop。完整审核见`docs/reviews/axis_pi05/TRACK3_OPERATOR_SCOPED_J0_J4_AUDIT_20260904.md`。

## 3.5 K0–K4 独立审核与 Matched Terminal-State 授权（2026-09-04）

审核输入root `8490f8dde1148aec0aec4d3d105082b44cfbb286`、K2 frozen `cb8e898640a66e9baf5a587f21d5f44a84494a8e`、nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`。K0 raw closure、K1/K2只读复算、K3单次运行后的hard stop和资源边界获批准；审核结论为`remediation_required`。

K3 raw显示task19 native reverse init在step5已有39个bowl↔middle-drawer geom contacts，最深penetration约`-6.80 mm`，到step20仍有39个且background `On(table region)`由true退为false。旧contact合同把step5已有target↔background pair当作diagnostic，所以physical gate虽显示passed，初态仍不适合直接作为清洗后的训练源。

另外三项是确定的工具缺陷：renderer helper只读wrapper而误报flags；`*.mp4.partial`使ffmpeg无法选择muxer；task19 destruction把qpos由`-0.1464`继续减到`-0.2664`，按public WoodenCabinet阈值仍是Open，raw实际为`true/true/true`。旧K4中文报告写成`true/false/true`不准确，已在新审核报告中以errata纠正但不修改旧证据。

旧计划已归档。新唯一活动计划`docs/AXIS_PI05_TRACK3_MATCHED_TERMINAL_STATE_RELEASE_PLAN.md`先修复观测工具，再固定读取三份公开HDF5中`demo_0/1/2`各自的`model_file + terminal full-state`，做三族各`3/3` exact-model CPU release；通过后GPU3最多运行task19、task10、task20各一条camera canary。即使全通过也只停在`pending_executor_review`，不生成action/demo，不进入SE(3)、Teleop或训练。完整审核见`docs/reviews/axis_pi05/TRACK3_BACKGROUND_CONVERGENCE_K0_K4_AUDIT_20260904.md`。

## 3.4 Operator-Scoped J0–J4 执行收尾（2026-09-04）

J0 锁定旧 I2 namespace、BDDL/source、root ancestry 与 nested OpenPI；J1 固定 operator-scoped contract，三族 contact contract、step5 baseline、分维度阈值和 tri-state 语义均通过。J2 复算旧 I2 的 9 条 reverse 与 3 条 task10 forward：旧字段 operator-scoped `9/9` 通过，但历史 contact matrix 明确为 `not_evaluated`，不能单独释放 J3。

J2.2 已真实完成 15 个 CPU/no-renderer fresh processes：9 条 reverse、6 条 forward。修正 step5 baseline 和 goal-false expectation 语义后，reverse 为 `8/9`、forward 为 `6/6`，contact matrix 为 `15/15` 完整。唯一失败是 task19 run1 在 step5→20 的完整 init truth vector 出现 background predicate 跃迁；该结果按固定 gate 记录为失败，没有改阈值、删背景或从 forward 对照反推。

因此 J3 GPU3 task10 canary 与三族 `3/3` 均为 `not_evaluated`，J4 已 hard stop。J4 targeted/full tests、JSON、py_compile、diff check 全部通过；旧 I2 hash 与 reverse-demo `0/0/0` 保持不变。证据位于 `data/reports/axis_libero_30k/track3_operator_scoped_stability/`，等待下一次审核；不得自行进入 SE(3)、demo、Teleop、批量生成或训练。

## 3.6 Matched Terminal-State L0–L5 收尾（2026-09-04）

L0/L1 已完成：锁定旧 K0–K4、三份公开 HDF5 及固定的 task10/task20/task19 `demo_0/1/2` 选择；每条选择 reward 持续成功后缀的第一份 state。新 validator contract 覆盖 wrapper/underlying renderer flags、`name.partial.mp4` H.264 atomic publish、public WoodenCabinet absolute close midpoint，以及无 baseline grandfather 的 contact v2。L0/L1 没有启动 simulator 或 GPU。

L2 完成 9 个 fresh CPU/no-renderer exact-model processes：task10 `3/3`，task20 `0/3`，task19 `0/3`。task10 三条通过；task20 的 fixed clean gate 因目标物 participant drift 和严格 operator-background contact 未通过，task19 因 robot arm/EEF drift 和严格 operator-background contact 未通过。L3 task19 camera 与 L4 task10/task20 camera 均为 `not_evaluated`，没有触碰 GPU3。

L5 已按失败门禁 hard stop：`exact/pending/accepted reverse demo = 0/0/0`；没有生成 action/demo，没有 SE(3)、grasp、Teleop、批量生成、训练/norm、FAST/ZIP、官方 GPU、上传或提交。最终测试 `211 passed, 7 skipped, 26 warnings`，JSON parse、py_compile、diff check 通过；`LIBERO-plus` 原有 dirty 状态未触碰。证据位于 `data/reports/axis_libero_30k/track3_matched_terminal_state_release/`，下一步仅等待审核。

## 4. 保留但非活动的研究线

- `AXIS_PI05_STANDARD_LIBERO_30K_RESEARCH_PLAN.md`：长期标准LIBERO 30K/full-FT研究的事实与路线来源；当前执行授权由donated accumulator计划收窄到最多10 steps。
- `TRACK3_REVERSE_DATA_SYNTHESIS_PLAN.md`：长期完整逆向数据研究的事实来源；当前执行授权由正分优先计划收窄到pilot、500-step LoRA和候选ZIP。
- `AXIS_PI05_TWO_GPU_LORA_TODAY_PLAN.md`：历史 provenance 路径被配置/工具引用，因此保留原位置；不再是活动计划。
- 其余早期 post-training、低分恢复、urgent FAST、比赛 GPU 计划均不能覆盖本索引的当前活动计划。

## 5. 今后每一步审核的固定动作

用户要求从本次开始，每一次阶段审核都执行完整闭环：

1. 只读核对 commit、输入 hash、产物、门禁、测试、GPU/外部动作边界。
2. 给出唯一审核状态：`approved`、`remediation_required` 或 `blocked`。
3. 已完成/已被取代的活动计划移入 `docs/archive/axis_pi05/`，不得删除历史。
4. 在 `docs/` 根目录新建且只保留一个下一阶段活动计划；不得继续向旧计划追加新授权。
5. 同步更新：
   - `docs/AXIS_PI05_PLAN_INDEX.md`；
   - `docs/AXIS_PI05_PROJECT_HANDOFF_20260831.md`；
   - `data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json`。
6. 对忽略目录中的小型 summary/report 使用显式 `git add -f`；模型、视频、NPZ、大型 rollout 不进入 Git。
7. 运行 JSON、Markdown、测试与 `git diff --check` 校验，形成 clean commit。
8. 下一执行者只运行新计划明确授权的阶段，结束后再次 hard stop 等待审核。

若本轮审核失败，下一计划应是最小 remediation，而不是跳到原定后续阶段。
