# AXIS Pi0.5 计划索引与审核生命周期

> 更新日期：2026-09-02（Asia/Tokyo）
> 当前唯一活动计划：`docs/AXIS_PI05_TRACK3_PUBLIC_TERMINAL_TELEOP_PLANNER_RECOVERY_PLAN.md`
> 当前状态：`track3_public_terminal_recovery_ready`
> 最新审核：`docs/reviews/axis_pi05/TRACK3_T1_TERMINAL_SOURCE_AUDIT_20260902.md`

## 1. 唯一活动计划

- `AXIS_PI05_TRACK3_PUBLIC_TERMINAL_TELEOP_PLANNER_RECOVERY_PLAN.md`
  - 独立审核确认T1 hard stop正确：40个CPU reverse specs只是候选，GPU3真实20Hz pilot为`0 accepted episodes`；旧`.pruned_init`/8D state路径不能恢复完整成功终态；
  - P0先下载并锁定官方公开LIBERO原始HDF5，检查`data/demo_i/states`、task/BDDL/model provenance；下载不使用GPU；
  - P1只用完整官方HDF5 state或当前环境中R6公开正向成功的live full-state capture，完成restore、settle、predicate与fresh-process 3/3门禁；
  - P2实现keyboard必备/SpaceMouse可选的20Hz teleop recorder并取得3 families canary；P3才允许从成功teleop提取闭环keypose planner；
  - P4逐级扩展到30–60条、3 families、6 templates、2 scenes、逐条3/3 replay与H.264人工审核；P5只做schema/manifest/loader forward smoke；
  - GPU0不触碰，GPU3只用于simulator/capture，GPU1/2不训练；本计划不授权LoRA、dense、FAST、ZIP、C20、E2、比赛GPU、上传或提交。

任何执行者看到多个计划时，以本索引的“唯一活动计划”为准，不得从归档计划的未执行尾部自行恢复任务。

正式Track1 `0.47`与Track2 `0.33`已证明R6 FAST在官方20Hz闭环有效；Track3 `0`说明纯正向LIBERO LoRA并未自然获得逆向能力。当前优先级不是立即训练，而是先用公开完整simulator state建立合法、通用、可回放的reverse supervision；dense full-FT继续冻结为独立研究线。

## 2. 已归档的 R6 执行链

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

归档不表示旧计划所有尾部阶段都完成；只表示该文件不再是活动授权入口。旧 failure、阈值、hash 和未执行阶段必须继续保留。

当前只执行Track3 terminal recovery P0–P5；不得从归档reverse LoRA/interstep计划自动恢复训练，也不得直接转向Blackwell。

## 3. 保留但非活动的研究线

- `AXIS_PI05_STANDARD_LIBERO_30K_RESEARCH_PLAN.md`：长期标准LIBERO 30K/full-FT研究的事实与路线来源；当前执行授权由donated accumulator计划收窄到最多10 steps。
- `TRACK3_REVERSE_DATA_SYNTHESIS_PLAN.md`：长期完整逆向数据研究的事实来源；当前执行授权由正分优先计划收窄到pilot、500-step LoRA和候选ZIP。
- `AXIS_PI05_TWO_GPU_LORA_TODAY_PLAN.md`：历史 provenance 路径被配置/工具引用，因此保留原位置；不再是活动计划。
- 其余早期 post-training、低分恢复、urgent FAST、比赛 GPU 计划均不能覆盖本索引的当前活动计划。

## 4. 今后每一步审核的固定动作

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
