# AXIS Pi0.5 计划索引与审核生命周期

> 更新日期：2026-09-06（Asia/Tokyo）
> 跨session入口：先完整读取根目录`AGENTS.md`与`MEMORY.md`，再读取本索引、handoff、machine JSON、唯一活动计划和最新审核；先查live PID/cursor，禁止重复启动。
> 当前唯一活动计划：无（EG0–EG4 已执行并 hard stop，等待独立审核）
> 当前状态：`track3_ef_failure_provenance_and_lifecycle_remediation_completed_hard_stop_pending_independent_audit`
> 最新审核：`docs/reviews/axis_pi05/TRACK3_EF0_EF4_INDEPENDENT_AUDIT_20260906.md`

## 1. 本轮执行停止点

Z0–Z4的runtime/identity/phase-ledger修复成立；Z5只运行一次task10/demo0、seed3121 nominal CPU canary。28步均在approach，冻结4-step window由normalized error `0.531→1.077`触发`stage_stall`；close/lift/transport/place/release未进入，相关gate为`not_evaluated`。worker已原子发布`failed_gate`，但随后因缺少completed-status常量return 1，历史parent按`infrastructure`停止且没有重跑。

独立审核可把物理层派生为有效nominal `family_failure`，同时保留历史parent标签。Z14 manifest内`28/28` exact，但namespace有36个tracked文件；两个Z5 log及六个Z4/Z14 finalizer artifacts没有完整hash闭包，其中stderr是NameError原始证据。旧postmortem还从错误层级读取gate而产生null；Z4 targeted原文为`9 passed`，不是交接中的8。

45项输入、26个JSON、NPY/NPZ/CSV、Z5 identity与资源边界均独立通过；生命周期更新前全量为`281 passed, 7 skipped, 23 warnings`，更新后项目全量为`282 passed, 7 skipped, 23 warnings`。Z6–Z13=`not_evaluated`，`exact/pending/accepted=0/0/0`，没有GPU、generation/replay、训练或ZIP。

当前执行已完成AA0–AA5并进入AA14 hard stop：旧Z namespace已完成`36/36` artifact closure，派生分类、真实worker/finalizer回归和AA4 clean identity均通过；AA5唯一一次冻结参数的`object_frame_conservative` CPU/no-renderer canary为`family_failure`。审核确认AA5在approach/descend/close走了`49/21/6`步，EEF到位且无controller clip，但连续接触和source release未成立。

审核同时发现旧计划的family isolation没有实现：工具只提供AA0–AA5/AA14入口，AA6–AA13没有CLI；AA14又把support family failure扩散为全部尾部`not_evaluated`，与“support失败不阻止task20/task19”的计划冲突。当前T3数据、训练和新ZIP仍为0；旧R6 FAST ZIP已复核SHA exact，但其已知Track3结果为0，只能作为明确标注的fallback。

审核生命周期更新后targeted为`45 passed`，项目全量为`288 passed, 7 skipped, 23 warnings`；strict JSON、引用、py_compile与diff check通过。

Tonight执行的fallback本体、七个child attempt、首个infrastructure failure、后六条诊断性family failure、TZ3–TZ8未执行和所有零计数成立；本次独立审核结论为`remediation_required`。TZ1实际只有CLI flags，TZ4–TZ8仍是stub且没有真实singleton/周期heartbeat/deadline/identity/finalizer；首个shared infrastructure failure又被`343c5a5`的resume分支跨提交绕过，最终matrix顶层把它清成null。冻结seed未应用，三对同族trace分别byte-identical；6个NPZ含object dtype且不能全array `allow_pickle=False`读取，clip/contact字段亦为硬编码。旧failure和raw均保留，不能把六条诊断结果称为合规TZ2 completion。

ER审核确认三条新CPU/no-renderer canary均为有效诊断性`family_failure`，typed trace和fallback也成立；但ER namespace `0/29`文件进入Git，resume未复验完整identity，contact-completeness只表示telemetry枚举成功，ER3要求的真实supervisor回归缺失，ER5没有按声明顺序实际运行tests且final pass不覆盖ZIP/process/GPU/tests。唯一结论继续为`remediation_required`；三条ER4 identity已消耗，下一计划只做证据锚定和code/fixture修复。

## 2. 当前计划与停止点

- `AXIS_PI05_TRACK3_ER_EVIDENCE_ANCHOR_AND_FINALIZER_VERIFICATION_PLAN.md` 已归档；EF0–EF4 完成并 hard stop。EF0–EF3 技术合同、corrected EF4 checkout closure 与 fallback 机械属性经独立复核成立。
- 既有 EF `approved` 审核与 execution/evidence 同在 `b1195dd`，且其引用的 `ef4_first_attempt_failure.json` 明确承认首次失败 report 被 corrected attempt 覆盖。首次失败 raw 当前不可恢复，不能用派生说明替代；本轮唯一结论修正为 `remediation_required`。
- `AXIS_PI05_TRACK3_EF_FAILURE_PROVENANCE_AND_LIFECYCLE_REMEDIATION_PLAN.md` 已归档；EG0–EG4 已执行并 hard stop，新的独立审核尚未完成。当前没有活动计划，`next_phase_only=independent_audit_required`。
- 三条 ER4、旧七条 TZ identity 和既有 EF fixture 均禁止作为历史结果重跑。任何新 physics、GPU0–3、官方 GPU、Teleop、generation/replay、accepted 数据、训练/norm、新 candidate ZIP、上传/提交均不授权。

任何执行者看到多个计划时，以本索引的“唯一活动计划”为准；不得从归档计划的未执行尾部自行恢复任务。

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
31. `AXIS_PI05_TRACK3_MATCHED_TERMINAL_STATE_RELEASE_PLAN.md`：L0/L1通过；L2 exact-model CPU为task10 `3/3`、task20 `0/3`、task19 `0/3`，L3/L4未执行并hard stop。审核确认source/goal 9/9成立、失败来自success末帧动力学和contact-v2角色误标，转入composed reverse-start stabilization。
32. `AXIS_PI05_TRACK3_COMPOSED_REVERSE_START_STABILIZATION_PLAN.md`：M2 task20 canary、M3 CPU `9/9`与M4 camera `3/3`通过并hard stop。核心state语义/动力学获批准，但M4 provenance、full-window gate和可restore candidate artifact未闭合，转入start bundle与单条自动support executor canary。
33. `AXIS_PI05_TRACK3_START_BUNDLE_AND_SUPPORT_EXECUTOR_CANARY_PLAN.md`：P0/P1与P2 `9/9`通过；P3首条task10因stdout管道阻塞在result publication失败并hard stop，P4/P5未执行。审核发现4个manifest artifact未入Git，转入I/O与artifact closure后恢复P3。
34. `AXIS_PI05_TRACK3_P3_IO_RECOVERY_AND_SUPPORT_EXECUTOR_CANARY_PLAN.md`：R0/R1与artifact closure通过；R2 task10 attempt2 child通过，但GPU source commit缺`gpu_provenance_gate`定义导致parent NameError并hard stop，R3–R5未执行。转入parent纯函数回归和overnight autogen/LoRA流水线。
35. `AXIS_PI05_TRACK3_OVERNIGHT_AUTOGEN_LORA_AND_ZIP_PLAN.md`：O0/O1通过；O2 task10 attempt3 child与相机通过，parent仅因同步GPU query耗时148.46ms超过误设的100ms阈值而停止。O3–O9未执行；旧O10 manifest另有CSV CRLF/LF hash不稳定。转入telemetry v2、只读attempt3升格和通宵续跑。
36. `AXIS_PI05_TRACK3_PROVENANCE_JITTER_RECOVERY_AND_OVERNIGHT_CONTINUATION_PLAN.md`：J0–J3核心释放成立；J4 executor合同与J5 quaternion约定不完整，首条task10 nominal另遇MP4原子扩展名错误并在J5 hard stop。J6–J10未执行；转入support-only executor/atomic-video最小恢复。
37. `AXIS_PI05_TRACK3_J5_EXECUTOR_CONTRACT_AND_ATOMIC_VIDEO_RECOVERY_PLAN.md`：V0–V3工程闭包与V4单次运行/V5 hard stop成立；审核发现OSC rotation物理scale、release-side HDF5 anchor、stage-dependent三态和实际supervisor仍未闭合，转入CPU-first最小恢复。
38. `AXIS_PI05_TRACK3_SUPPORT_RELEASE_ANCHOR_AND_OSC_SCALE_RECOVERY_PLAN.md`：W0–W3内部检查与W4单次CPU family failure/W6 hard stop成立；审核发现destination object→EEF组合、acquisition barrier、progress stall、active cursor和final validation顺序仍未闭合，转入一次性bounded autogen/LoRA条件链。
39. `AXIS_PI05_TRACK3_OBJECT_FRAME_EXECUTOR_AND_BOUNDED_AUTOGEN_LORA_PLAN.md`：X0–X3内部报告与X13 closure成立；X4在0 physics step时因不存在的runtime helper发生infrastructure hard stop。审核另发现完整input lock、stall执行、阶段门禁、周期heartbeat与finalizer自哈希仍未闭合，转入runtime preflight后再恢复bounded continuation。
40. `AXIS_PI05_TRACK3_RUNTIME_PREFLIGHT_AND_BOUNDED_CONTINUATION_PLAN.md`：Y0全量input lock与Y2 artifact closure成立；Y3未执行。审核发现child argv无法通过顶层parser、最终hard gate丢失阶段作用域、destination缺数值验证且source identity自指，转入launch/phase-ledger remediation。
41. `AXIS_PI05_TRACK3_Y3_LAUNCH_PHASE_LEDGER_REMEDIATION_PLAN.md`：Z0–Z4工程合同成立；Z5 nominal在28步approach stall，worker发布结果后finalizer NameError使parent按infrastructure hard stop。审核派生物理family failure，并发现28/36 manifest coverage、自哈希与postmortem gate-path缺口，转入derived closure和单次conservative continuation。
42. `AXIS_PI05_TRACK3_Z5_DERIVED_RELEASE_AND_CONSERVATIVE_CONTINUATION_PLAN.md`：AA0–AA4闭环成立；AA5 support conservative在76步acquisition barrier正常family failure。执行器却未实现AA6–AA13且错误停止独立family，转入Tonight ZIP与cross-family一次性恢复。
43. `AXIS_PI05_TRACK3_TONIGHT_ZIP_AND_CROSSFAMILY_RECOVERY_PLAN.md`：fallback SHA/CRC/布局和七个CPU child attempt成立；但完整调度器、sticky infrastructure stop、identity/seed、computed gates、no-pickle trace与finalizer合同未闭合，转入ER0–ER5最小修复。
44. `AXIS_PI05_TRACK3_TZ_EVIDENCE_AND_EXECUTOR_CONTRACT_REMEDIATION_PLAN.md`：ER0旧证据派生、fallback和三条新CPU诊断family failure成立；但29项ER证据均未入Git，resume identity、真实supervisor回归、physical contact-completeness与finalizer验证顺序未闭合，转入EF0–EF4只读锚定/code-only修复。

归档不表示旧计划所有尾部阶段都完成；只表示该文件不再是活动授权入口。旧 failure、阈值、hash 和未执行阶段必须继续保留。

ER0–ER5修复链已执行并 hard stop；不得修改历史证据/公开HDF5/BDDL、重跑旧七条TZ identity、把失败轨迹入训，或使用任何GPU/Teleop/generation/replay/训练/上传/提交。ER4实际三族各一条全新CPU/no-renderer identity且均为正常family failure；当前下一权限为`new_reviewed_plan_required`。

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

L5 已按失败门禁 hard stop（execution commit `19cb1ea`）：`exact/pending/accepted reverse demo = 0/0/0`；没有生成 action/demo，没有 SE(3)、grasp、Teleop、批量生成、训练/norm、FAST/ZIP、官方 GPU、上传或提交。最终测试 `211 passed, 7 skipped, 26 warnings`，JSON parse、py_compile、diff check 通过；`LIBERO-plus` 原有 dirty 状态未触碰。证据位于 `data/reports/axis_libero_30k/track3_matched_terminal_state_release/`，下一步仅等待审核。

## 3.7 L0–L5 独立审核与 Composed Reverse-Start 授权（2026-09-04）

审核输入closeout `0dc402e2d70e6576c5861a674e0aac04e4427ef0`、execution `19cb1eab9556b6e9e0f8b3f9d4b49fb3893b81a0`。9份raw hash、9个fresh process、L3/L4未执行与hard stop获批准；结论为`remediation_required`。

9/9 reverse source成立且goal为false。task20失败主要是success末帧后目标物沉降，task19失败主要是forward terminal robot回稳；9条reward positive suffix均只有一帧。contact-v2把gripper/robot↔operator误标为unrelated background，旧失败保留但不能沿用此角色解释。

旧计划已归档。新唯一活动计划`docs/AXIS_PI05_TRACK3_COMPOSED_REVERSE_START_STABILIZATION_PLAN.md`固定构造same-demo terminal scene + init robot/gripper + zero-qvel state，先task20/demo0 CPU canary，再补齐固定9条CPU matrix；全部通过后GPU3最多三族各一条camera canary。M5 hard stop，不授权action/demo、SE(3)、Teleop、训练或官方GPU。完整审核见`docs/reviews/axis_pi05/TRACK3_MATCHED_TERMINAL_L0_L5_AUDIT_20260904.md`。

## 3.8 Composed Reverse-Start M0–M5 执行收尾（2026-09-04）

本轮按唯一活动计划 `docs/AXIS_PI05_TRACK3_COMPOSED_REVERSE_START_STABILIZATION_PLAN.md` 完成 M0–M5。M0 锁定并复核上一轮 L0–L5、9 条固定 demo 及三份公开 HDF5；M1 新增 named state-composition、逐 index provenance、全零 qvel、neutral 7D hold 与 contact-v3 纯逻辑合约，均通过。

M2 的唯一 task20/demo0 CPU canary 通过；M3 固定九条 fresh CPU matrix 为 `9/9`，三族各 `3/3`。所有 run 均保持 reverse source=true、reverse goal=false，raw terminal 与 composed state 的 public forward goal 均为 true，严格稳定性与 contact-v3 门禁均通过。composition 固定为同 demo terminal scene qpos + 同 demo init robot/gripper qpos + zero qvel，不改 HDF5/BDDL。

M4 仅使用 GPU3，按 task10→task20→task19 各执行一次 camera canary；三条均通过，agentview/wrist 各为 `41` 帧 H.264、20 FPS、原子发布且无 `.partial` 残留。M5 full tests 为 `223 passed, 7 skipped, 23 warnings`，JSON、py_compile、diff check 和 M0 immutable hash revalidation 全部通过。

当前状态为 `track3_composed_reverse_start_validated_pending_executor_review`，已 hard stop。`start_state_candidate` 仅有 `1` 条 report-only 记录，`exact/pending/accepted reverse demo = 0/0/0`。没有生成 manipulation action/demo，没有 SE(3)、Teleop、批量生成、训练、norm、FAST/ZIP、官方 GPU、上传或提交；后续必须等待新的审核计划。

机器证据：`data/reports/axis_libero_30k/track3_composed_reverse_start_stabilization/summary.json`、`m0_release_lock.json`、`m1_validator_contract.json`、`m2_summary.json`、`m3_summary.json`、`m4_summary.json`、`run_index.json`、`contact_ledger.json` 与 `M5_CLOSEOUT_REPORT_ZH.md`。最终amended execution commit为`eef683b`；旧handoff中的`020d291`是reflog可见的amend前orphan。nested OpenPI clean；`LIBERO-plus`既有dirty状态仅记录未触碰。

## 3.9 M0–M5 独立审核与 Start Bundle/Support Executor 授权（2026-09-04）

审核输入root `eef683b0db9629a3d917ea91ebeb46540a34fcb9`。M2/M3的9个CPU start、M4的3个camera canary、6段视频及禁止项获核心批准；12份raw required gates全通过，restore最大误差`2.22e-16`。只读补查的burn-in contact违规为0。

总体仍为`remediation_required`：M4 raw缺seed/PID/GPU peak；contact required gate漏step1–20；report-only candidate只有hash没有state vector；`round_trip...`字段误名、run index缺file hash且handoff使用orphan `020d291`。用户人工操作为`not_required`。

旧计划已归档；唯一活动计划`docs/AXIS_PI05_TRACK3_START_BUNDLE_AND_SUPPORT_EXECUTOR_CANARY_PLAN.md`先闭合上述机器合同并物化9个state，通过后只运行task10/demo0自动SE(3) canary与最多3次replay。成功也只进入`pending_autogen_v3`，不进入accepted或训练。完整审核见`docs/reviews/axis_pi05/TRACK3_COMPOSED_REVERSE_START_M0_M5_AUDIT_20260904.md`。

## 3.10 Start Bundle / Support Executor P0–P6 收尾（2026-09-04）

P0/P1 通过；P2 物化 9 个 no-pickle start state，fresh CPU restore 为 `9/9`，task10/demo0 固定 HDF5 抓取几何提取通过。P3 按固定顺序只启动 task10/demo0 一条 GPU3 camera canary；41 帧与两路 H.264 已封存，但 worker 因完整 checkpoint JSON 填满 stdout 管道而在结果发布阶段阻塞，按 hard stop 终止。修复后的文件重定向/压缩输出工具未在本轮重启 canary。

P3 失败后 task20/task19、P4、P5 均未启动；没有 action、SE(3)、demo、pending、Teleop 或训练。P6 完成 artifact manifest、CSV、strict JSON/NPZ/video/py_compile/diff check；targeted `8 passed`，全量 `231 passed, 7 skipped, 23 warnings`。最终状态 `track3_support_executor_canary_failed_worker_result_publication`，`validated_start_state=9`、`exact/pending/accepted=0/0/0`，已 hard stop 等待新审核计划。证据位于 `data/reports/axis_libero_30k/track3_start_bundle_and_support_executor_canary/`。

## 3.11 P0–P6 独立审核与 P3 I/O Recovery 授权（2026-09-05）

审核输入root `ad2d4ef7d6603e7b0e79a8519155de9cfdefbde2`。P0/P1、P2的9-state no-pickle bundle、fresh CPU restore `9/9`、grasp geometry、P3首错即停与全部禁止项获批准。最大restore误差`2.22e-16`；P4/P5真实未执行，reverse demo仍为`0/0/0`。

审核结论为`remediation_required`：旧manifest 17项在本机全部hash exact，但4个NPZ/MP4未被Git跟踪；旧P6的summary exclusion字段错误；P3失败时的pre-fix dirty-worktree工具没有commit，而当前post-fix实现仍缺独立result file与超时终止。P3没有有效worker result，因此不能把失败归因于环境/state。

旧计划已归档；唯一活动计划`docs/AXIS_PI05_TRACK3_P3_IO_RECOVERY_AND_SUPPORT_EXECUTOR_CANARY_PLAN.md`先完成17/17 Git closure与无管道result/timeout压力测试，再只授权一次task10 attempt 2；通过后补齐task20/task19，才允许task10-only executor与一次generation/3次replay。用户人工操作仍为`not_required`。完整审核见`docs/reviews/axis_pi05/TRACK3_START_BUNDLE_P0_P6_AUDIT_20260905.md`。

## 3.12 R0–R6 独立审核与 Overnight Autogen/LoRA 授权（2026-09-05）

审核输入root `309ccafe9e6e2cddd0c9f16db111fc054478eb8b`、GPU source `ad0a428c76f2c597109855f4d67f00df78184514`。旧17/17、新33/33 artifact closure、R1 I/O压力合同、R2仅一次attempt2、child全部物理/camera gate、parent fail-closed与R3–R5未执行获批准。reverse demo仍为`0/0/0`。

失败是确定性parent符号缺失：`ad0a428`中调用`gpu_provenance_gate`但没有定义，`309ccaf`才补入。因start snapshot/UUID/peak不可追溯，attempt2不能升格为P3通过；但无需重做state、R1或回到AXIS init。

旧计划已归档。唯一活动计划`docs/AXIS_PI05_TRACK3_OVERNIGHT_AUTOGEN_LORA_AND_ZIP_PLAN.md`授权最长8–12小时的条件流水线：先修parent并运行attempt3，随后按family独立pilot/generation/replay；有严格machine-accepted数据后才用20Hz LIBERO-plus做`80/20`混合、fresh QUANTILES、R6 L2 LoRA 100/500/1000、本地holdout/retention和FAST候选ZIP。GPU0/官方GPU、dense full-FT与自动提交继续禁止。完整审核见`docs/reviews/axis_pi05/TRACK3_P3_IO_RECOVERY_R0_R6_AUDIT_20260905.md`。

## 3.13 Overnight O0–O2 独立审核与 Jitter Recovery 授权（2026-09-05）

审核输入root `8c0d296f1dfedb0c777040a733b6f9e352234111`。O0/O1、task10 attempt3单次执行、child四类hard gate、两路41帧20 FPS H.264、parent PID/UUID/start/end/peak/return/timeout证据及O2 hard stop获批准。独立复跑为`244 passed, 7 skipped, 23 warnings`。

旧O2失败是监控器语义错误：`parent_poll_interval_max_ms=148.4566`测的是同步`nvidia-smi`query latency而不是heartbeat gap；113 samples、25.5秒起止跨度、return code 0与无timeout表明provenance主体完整。另发现O10 manifest中CSV写入时CRLF hash与Git检出LF hash不一致；旧证据保留为failed，不能直接声称closure全部hash exact。

旧计划已归档。唯一活动计划`docs/AXIS_PI05_TRACK3_PROVENANCE_JITTER_RECOVERY_AND_OVERNIGHT_CONTINUATION_PLAN.md`先做telemetry v2、LF closure和attempt3只读升格，禁止task10 attempt4；通过后补task20/task19并按family独立执行executor、最多12条generation/36次replay、20Hz 80/20 mixture、R6 L2 LoRA 100/500/1000及条件式本地FAST ZIP。GPU0、官方GPU、dense full-FT与自动上传/提交继续禁止。完整审核见`docs/reviews/axis_pi05/TRACK3_OVERNIGHT_O0_O2_AUDIT_20260905.md`。

## 3.14 J0–J5 执行停止点（2026-09-05）

J0–J2完成input/telemetry/derived release，J3在GPU3分别运行task20、task19一次且均通过；J4报告释放三族executor。J5只运行task10/demo0 nominal一次：child产生180帧finite轨迹但`approach_grasp`未收敛，parent视频编码因`agentview.mp4.partial`失败，J5按infrastructure hard stop。J6–J10均为`not_evaluated`，没有training/Teleop/demo/FAST/ZIP，计数`0/0/0`。

执行提交`7ae488b4475169ac1cc3b586158454639e6a2be9`；J5 clean source为`3a5641956770fa74faaed20f12832489d7f257b7`。全量测试记录及独立复跑均为`250 passed, 7 skipped, 23 warnings`。GPU3已释放，GPU0外部PID保持，root/OpenPI clean，`LIBERO-plus`既有dirty未改。

## 3.15 J0–J5 独立审核与 Executor/Atomic Video 恢复授权（2026-09-05）

审核结论为`remediation_required`。J0 immutable hashes、J2只读升格、J3两条P3、J5单次执行与hard stop、J6–J10未执行及资源边界获批准；但J1 launcher、J4 intended executor与J11/manifest closure未获批准。

主要根因是Robosuite `xyzw`值被按`wxyz`处理：正确object-relative EEF重构误差`2.276e-7 rad`，旧helper偏差`2.21036 rad`，旧轨迹180/180 action saturated且最终rotation error `3.12605 rad`。另有`*.mp4.partial`、valid `failed_gate` return code 1、J11 `no_family_release`误映射、无周期heartbeat/启动identity校验，以及旧J3/J11 manifest分别`5/21`、`4/37`漂移。

旧计划已归档；唯一活动计划`docs/AXIS_PI05_TRACK3_J5_EXECUTOR_CONTRACT_AND_ATOMIC_VIDEO_RECOVERY_PLAN.md`先CPU修复并离线derived closure，clean commit后只允许一次corrected task10 nominal GPU3 canary，随后hard stop。task20/task19 manipulation、conservative、generation/replay、训练和ZIP仍不授权。完整审核见`docs/reviews/axis_pi05/TRACK3_JITTER_J0_J5_AUDIT_20260905.md`。

## 3.16 Executor Recovery V0–V5 执行收尾（2026-09-05）

V0 immutable input为`18/18` hash exact；V1修正Robosuite `xyzw`单点重构；V2从旧180帧原子重编码两路H.264；V3在`0b17f66` source上形成`1b70440` clean release。V4只在GPU3执行task10/demo0 nominal、seed3111一次：worker PID一致、return code0、telemetry/camera通过，但150帧后停在`approach_grasp`，parent分类为`family_failure`。

V5传播family failure并hard stop。最终manifest `19/19`均Git-tracked/hash exact；全量测试为`256 passed, 7 skipped, 23 warnings`。没有retry、conservative、其他family、demo/generation、Teleop、训练/norm、FAST/ZIP或提交；`exact/pending/accepted=0/0/0`。最终execution commit为`1148362`。

## 3.17 V0–V5 独立审核与 Release-Anchor/OSC-Scale 授权（2026-09-05）

审核批准V0/V2/V3核心工程证据、V4单次运行的`family_failure`分类、V5 hard stop、19/19 manifest和禁止项；唯一结论仍为`remediation_required`。

`xyzw`已修正，但action builder把声明的`0.12 rad` rotation clip归一到`±1`，实际OSC config的`±1`对应`±0.5 rad`，旋转命令约放大`4.1667×`。V1只使用close-side index32，没有闭合HDF5 release-side object-follow/contact；release前index75与index32的relative pose相差`0.0387545 m / 0.600317 rad`。V4又在approach停止后把contact completion与same-demo destination误写为failed；真实launcher也没有调用identity/heartbeat gate。

旧计划已归档。唯一活动计划`docs/AXIS_PI05_TRACK3_SUPPORT_RELEASE_ANCHOR_AND_OSC_SCALE_RECOVERY_PLAN.md`先完成CPU anchor/scale/三态/supervisor闭包，再只运行一次CPU/no-renderer完整canary；通过才允许一次GPU3 camera canary。完整审核见`docs/reviews/axis_pi05/TRACK3_J5_EXECUTOR_RECOVERY_V0_V5_AUDIT_20260905.md`。

## 3.18 Support Release-Anchor W0–W6 执行收尾（2026-09-05）

W0的15项immutable hash、旧V5 `19/19`、root/OpenPI/GPU边界通过；W1独立锁定task10/demo0 close/release=`32/76`、state75 contact=`3/1/0`与release-side relative pose；W2按实际controller output scale把`0.12 rad`映射为normalized `0.24`；W3形成clean release source。

W4只运行一次CPU/no-renderer、seed3121。worker正常return 0并发布`failed_gate`，parent分类`family_failure`。trajectory为252步，stage为approach41、descend12、close4、lift15、transport180；transport未达标，source全程true、goal全程false。W5未启动GPU3并记录`not_evaluated`。W6 `17/17` tracked/hash exact，strict JSON/NPZ/NPY/CSV/no-partial通过，记录`257 passed, 7 skipped, 23 warnings`后hard stop。最终commit为`2997c6f`，计数`0/0/0`。

## 3.19 W0–W6 独立审核与 Object-Frame/Bounded Autogen 授权（2026-09-05）

审核批准W0/W1/OSC scale、W4真实运行的family分类、W5未执行、W6 artifact closure、资源与禁止项；唯一结论为`remediation_required`。W4 destination侧没有组合`T_world^object_des·T_object^eef_anchor`，当前place target与正确EEF target相差`0.0817122 m / 1.816425 rad`。同时source为`253/253 true`、contact completion failed，却仍越过acquisition barrier进入transport；progress-window stall、运行中active cursor和post-finalization验证也未闭合。

旧计划已归档。唯一活动计划`docs/AXIS_PI05_TRACK3_OBJECT_FRAME_EXECUTOR_AND_BOUNDED_AUTOGEN_LORA_PLAN.md`在X0–X3修复上述合同，随后单计划条件式授权三族profile/pilot、最多12条generation/36次replay、20Hz mixture/fresh QUANTILES、R6 L2 LoRA 100/500/1000、holdout/C20与本地FAST ZIP。family failure隔离，共享错误全局停止，X13统一hard stop；完整审核见`docs/reviews/axis_pi05/TRACK3_SUPPORT_RELEASE_ANCHOR_W0_W6_AUDIT_20260905.md`。

## 3.20 AA0–AA14 执行收尾（2026-09-06）

AA0–AA4已完成：旧Z namespace的36个Git-tracked artifacts均已在`aa1_old_z_artifact_manifest.json`中exact闭合；AA2保留历史parent=`infrastructure`，并从raw child `hard_gate.gates`派生nominal `family_failure`（`stage_stall:approach_grasp`），不改写旧postmortem；AA3真实worker/finalizer回归覆盖`passed`、`failed_gate`、`worker_exception`、return code、TERM→KILL、16 MiB stdout、LF CSV、inactive cursor和PID祖先过滤；AA4冻结`object_frame_conservative` clean identity。

AA5严格只运行一次task10/demo0、seed3121、CPU/no-renderer、无参数变更canary。worker正常return 0并发布`failed_gate`，parent分类为`family_failure`：共76个physics steps，进入approach/descend/close但acquisition barrier未通过（contact continuous/source released均为false）；trajectory finite为`76×7` actions / `76×79` states，未进入same-demo destination，不能入训。没有retry。

AA14已执行hard stop：AA6–AA13均为`not_evaluated`，`exact/pending/accepted=0/0/0`，没有GPU3、generation、replay、training、ZIP、Teleop、上传或提交。执行证据位于`data/reports/axis_libero_30k/track3_z5_derived_release_and_conservative_continuation/`，最终closeout、manifest、validation和outer hash均exact；执行root为`cf27e7b1fe9276eaf0d4285128d56586e049d28f`，nested OpenPI仍为`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`。AA4记录的定向/全量测试为`5/287 passed`（全量`7 skipped, 23 warnings`）；文档同步后的最终测试另行复核。

当前唯一活动计划仍是`docs/AXIS_PI05_TRACK3_Z5_DERIVED_RELEASE_AND_CONSERVATIVE_CONTINUATION_PLAN.md`，但执行授权已消耗并硬停止；下一步唯一权限是`new_reviewed_plan_required`。LIBERO-plus既有dirty保持原样，root/OpenPI clean，GPU0–3无计算进程。

## 3.21 Tonight TZ0–TZ9 执行收尾（2026-09-06）

TZ0完成R6 FAST来源与本地fallback的全量SHA/CRC/布局复验：3401 entries，`model_fast`唯一、无slow权重，来源与物化ZIP均为`9383571677` bytes、SHA `95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a`，并明确标注`fallback_no_track3_improvement`。C5 source-success为`3/3`，没有上传或提交。

TZ1完成全部TZ0–TZ9与worker CLI的静态入口、原子cursor/heartbeat/结果协议、TERM→KILL合同和7个新identity冻结。TZ2实际完成7/7 CPU/no-renderer fresh children：support `3`（其中1个执行器自身import bug被保留为infrastructure、2个正常family failure）、containment `2/2 family_failure`、articulation `2/2 family_failure`；修复import bug后未重跑已消耗的首个identity，故仍严格不超过7次。三族成功均为`0`，且旧nominal/conservative没有重跑。

由于没有成功family，TZ3 GPU3 camera、TZ4 generation/replay、TZ5 mixture/QUANTILES、TZ6 LoRA、TZ7 holdout/C20与TZ8 candidate均为`not_evaluated`，不是失败或成功。TZ9选择并再次CRC/SHA复验fallback，accepted reverse为`0/0/0`、candidate为false、training为false，写入`data/reports/axis_libero_30k/track3_tonight_zip_and_crossfamily_recovery/`并hard stop；下一权限为`new_reviewed_plan_required`。

最终验证：目标测试`50 passed`；项目全量`293 passed, 7 skipped, 23 warnings`；项目外部缓存/nested测试树未纳入项目口径且未修改。Tonight namespace strict JSON为`23/23`（含machine handoff为`24/24`），phase CSV为LF，py_compile、diff-check、无partial和GPU0–3空闲均通过。

## 3.22 Tonight TZ0–TZ9 独立审核与 ER 修复授权（2026-09-06）

审核输入root `2ae08e974d6be9ac59541236ce7ddb1d7f288dd6`，首个/后六条worker source分别为`1066774918941a574db3f02c7060f40aeb0c9c0a`与`343c5a554b0bfa05839b145f48a8c785d38edd19`。fallback独立复核为`9383571677` bytes、SHA `95415a...`、CRC通过、3401 entries、无duplicate/unsafe path、FAST-only；C5及其8份引用报告hash exact。七个child attempt、首个NameError infrastructure failure、后六条诊断性family failure和TZ3–TZ8 `not_evaluated`成立。

审核结论为`remediation_required`：TZ1把CLI flag误当完整实现，实际缺singleton/周期heartbeat/deadline/身份和完整尾部；首个shared infrastructure failure后又跨提交继续并在aggregate清空failure；seed未应用、三对trace相同；clip/contact为硬编码，6个NPZ不能全array no-pickle读取，TZ9无完整manifest/outer hash且validation顺序不闭合。

旧TZ计划已归档。唯一活动计划`docs/AXIS_PI05_TRACK3_TZ_EVIDENCE_AND_EXECUTOR_CONTRACT_REMEDIATION_PLAN.md`先做不可变derived closure和code-only修复，再最多运行三族各一条全新CPU/no-renderer canary并hard stop。不授权任何GPU、generation/replay、accepted数据、训练或新candidate ZIP。完整审核见`docs/reviews/axis_pi05/TRACK3_TONIGHT_TZ0_TZ9_AUDIT_20260906.md`。

生命周期更新后的targeted为`53 passed, 2 warnings`，项目全量为`296 passed, 7 skipped, 23 warnings`；strict JSON、引用/归档与py_compile通过，最终树仍须通过diff-check和clean commit复核。

## 3.23 ER0–ER5 Executor Contract 修复执行收尾（2026-09-06）

ER0 在新 namespace `data/reports/axis_libero_30k/track3_tz_evidence_and_executor_contract_remediation/` 完成旧 Tonight 证据派生闭包：23/23 JSON strict、6/6 旧 NPZ 的 object-array/no-pickle 失败原样保留、1 个 phase CSV、两个 fallback ZIP 均完成只读 bytes/SHA/CRC/布局复核；首条 `support_contact_repair_a` infrastructure failure、6 条诊断性 family failure、顶层 aggregate null 矛盾和三组同族 identical trace 均保留。

ER1–ER3 通过真实文件锁、PID/PGID、固定 argv/runtime identity、atomic heartbeat、TERM→KILL/no-descendant 合同、sticky stop 逻辑、有效 seed 应用、三态 computed gate、typed no-pickle trace、离线阶段诊断和每族一条新 identity freeze。ER4 按 support→containment→articulation 只执行三条全新 CPU/no-renderer child；三条 identity gate 均通过，seed 均实际应用，结果分别为正常 `family_failure`，support/containment 在 `descend_object`、articulation 在 `approach_drawer_region` 停止，成功族为 `0/3`，没有 retry、GPU 或旧 identity 重跑。

ER5 以 fallback 作为唯一交付对象并 hard stop：`delivered_zip_kind=r6_fallback`、`fallback_no_track3_improvement`、`9383571677` bytes、SHA `95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a`；accepted reverse、training、new candidate 均为 `0/false`。artifact manifest、outer hash、inactive cursor、final validation 均已生成；3/3 typed NPZ no-pickle/finite、18 个 JSON strict、CSV LF、py_compile、diff-check 和 GPU0–3 empty 通过。ER4 后 targeted 为`15 passed`，项目全量为`308 passed, 7 skipped, 23 warnings`；canary source root `04eef24` clean、nested OpenPI `b58aff0` clean，LIBERO-plus 原有 dirty 未触碰。完整执行证据见 `data/reports/axis_libero_30k/track3_tz_evidence_and_executor_contract_remediation/summary.json`；下一权限为`new_reviewed_plan_required`，不上传、不提交。

## 3.24 ER0–ER5 独立审核与 EF0–EF4 授权（2026-09-06）

审核输入root `c2d9f23bb0bbe13a0860aeb4fecf3a75c932ae0d`、worker source `04eef24161bfe62adc9a51378235f73b34e6ac51`、nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`。旧证据30/30 current hash、fallback全包、ER manifest 26/26/outer hash、三条raw/NPZ、运行身份和禁止项均独立复核。三条canary保持有效诊断性family failure：support/containment/articulation分别155/149/180步，source仍true、goal false，action saturation分别10/5/10；成功族0，不能入训。

唯一审核结论为`remediation_required`。ER namespace 29个文件全部ignored且未入任何commit；resume只验result hash/profile ID，child又以当前dirty状态回填expected；pre-physics没有真实timeout/heartbeat/sticky/finalizer fixture回归；physical required-contact gate未实现；ER5未实际运行targeted/full tests，final pass也不包含expected ZIP、process/GPU和tests，heartbeat仍停在`finalizing`。

旧ER计划已归档。唯一活动计划`docs/AXIS_PI05_TRACK3_ER_EVIDENCE_ANCHOR_AND_FINALIZER_VERIFICATION_PLAN.md`只授权现存ER bytes原样锚定、完整identity/resume、无physics supervisor fixture、required-contact code gate和真实finalizer。三条ER4及旧七条TZ identity禁止重跑；任何新physics、GPU、generation/replay、accepted数据、训练或新ZIP均不授权。完整审核见`docs/reviews/axis_pi05/TRACK3_TZ_ER0_ER5_AUDIT_20260906.md`。

## 3.25 EF0–EF4 Evidence Anchor / Finalizer 执行与独立审核（2026-09-06）

EF0 已将 ER namespace 的 29 个现存文件以原字节 force-add 入 `38635fbf08e4e8df8c4877ddcb6b2b2ec8b9b118`，并逐项通过 commit read/hash 复核；旧 namespace 的 24 个 Git-persisted records 与 6 个 local NPZ 的差异状态保留。EF1 的 parent snapshot 显式冻结 root/tool/OpenPI/profile/command、三份 porcelain 及 digest 和 CPU/no-renderer 环境；真实 drift probes 与合法 resume 通过，trace bytes/SHA drift fail-close。

EF2 仅运行无 physics fixture child：singleton lock 被第二入口拒绝；正常 family-failure return `0`；周期 heartbeat 观察到至少四次 atomic 更新；timeout 真实经历 TERM→grace→KILL，结束后无后代；sticky stop 在启动 child 前拒绝第二入口。EF3 将 telemetry schema 与 required physical contact 分为独立三态 gate，support/containment 使用 demonstrator↔object 连续窗口，articulation 使用 demonstrator↔handle/region；零相关接触为 failed，未进入阶段为 `not_evaluated`。

EF4 在新 namespace 中按 producer close/fsync→inactive cursor→manifest/outer hash→strict JSON/NPZ/CSV 与 fallback ZIP→真实 process/GPU scan→targeted/full tests→final validation→closed heartbeat 完成；fallback 仅作既有机械 fallback，不创建新 candidate ZIP。最终独立审核结论为 `approved`，无新的 physics identity、GPU compute、生成、replay、训练、上传或提交；计划已归档，下一权限为 `new_reviewed_plan_required`。完整审核见 `docs/reviews/axis_pi05/TRACK3_ER_EVIDENCE_ANCHOR_AND_FINALIZER_VERIFICATION_AUDIT_20260906.md`。

EF4 finalizer targeted 为`7 passed`、项目全量为`320 passed, 7 skipped, 23 warnings`；审核后生命周期 targeted 为`12 passed`、项目全量仍为`320 passed, 7 skipped, 23 warnings`；strict JSON、引用/归档、py_compile、diff-check与资源边界均通过。

## 3.26 EF0–EF4 真正独立复核与 EG 修复授权（2026-09-06）

审核输入 root `b1195dd59830f7c44d831bd0007e61229446d9b5`、EF0 anchor `38635fbf08e4e8df8c4877ddcb6b2b2ec8b9b118`、nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`。ER `29/29`、EF checkout `22/22`、producer manifest `13/13`、outer/closure hashes、18 个 strict JSON、typed NPZ、LF CSV 和 fallback 全包复核通过；EF0–EF3 技术合同与 corrected EF4 closure 保留有效。审核前 targeted/full 为 `12 passed` 与 `320 passed, 7 skipped, 23 warnings`；无 live worker 或 GPU compute，禁止项均未发生。

唯一结论为 `remediation_required`：`ef4_first_attempt_failure.json` 明确写有 `historical_report_replaced_by_corrected_attempt=true`，首次失败的原始 validation、测试 stdout/stderr 和 partial inventory 没有保留；仓库、临时目录和不可达 Git blob 中也未找到副本。既有 `approved` 报告与 execution/evidence 同在一个提交，并错误声称没有覆盖；machine handoff 顶层三个 latest commit 指针还停在旧 ER 阶段。

旧 EF 计划保持归档。唯一活动计划为 `docs/AXIS_PI05_TRACK3_EF_FAILURE_PROVENANCE_AND_LIFECYCLE_REMEDIATION_PLAN.md`：EG0–EG4 只做不可恢复 gap 的诚实冻结、attempt-specific immutable finalizer、clean-source 无 physics fixture、独立性/lifecycle 回归与最终闭包。不得伪造首次失败 raw，也不授权任何 physics、GPU、generation/replay、accepted data、training、candidate ZIP、upload 或 submission。完整审核见 `docs/reviews/axis_pi05/TRACK3_EF0_EF4_INDEPENDENT_AUDIT_20260906.md`。

生命周期更新后 targeted 为 `17 passed`，项目全量为 `325 passed, 7 skipped, 23 warnings`；19/19 strict JSON、py_compile、diff-check、无 partial、process/GPU 边界通过。

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
