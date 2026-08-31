# AXIS Pi0.5 计划索引与审核生命周期

> 更新日期：2026-08-31（Asia/Tokyo）
> 当前唯一活动计划：`docs/AXIS_PI05_DENSE_FULLFT_SHARDED_INIT_RECOVERY_PLAN.md`
> 当前状态：`dense_fullft_first_update_true_compute_oom_pending_official_gpu_review`
> 最新审核：`docs/reviews/axis_pi05/DENSE_FULLFT_INIT_OOM_REVIEW_20260831.md`

## 1. 唯一活动计划

- `AXIS_PI05_DENSE_FULLFT_SHARDED_INIT_RECOVERY_PLAN.md`
  - 已审核确认两卡/四卡 OOM 都在 `init_train_state`，没有进入 batch/forward/backward/update；
  - 根因候选是约 `12.5 GiB` `partial_params` 被 `PartitionSpec()` 整树 replicated，然后才生成 FSDP-sharded train state；
  - 本轮只修复 sharded init，保持 FP32、dense、EMA `0.999`、AdamW、per-device microbatch `1`、accumulation `16`、global batch `64`；
  - CPU 定位和 4 卡 sharded init-only 已通过；唯一 first update 在真实计算阶段 OOM，未进入 10-step；不授权比赛 GPU、50/100/1K、C20、FAST/ZIP。

任何执行者看到多个计划时，以本索引的“唯一活动计划”为准，不得从归档计划的未执行尾部自行恢复任务。

正式`0.47`已经比本地Docker E2E更直接地证明R6 FAST部署链有效。下一问题不再是“包能否运行”，而是“如何用AXIS原团队式标准LIBERO dense full-FT提高策略能力”。两/四卡失败首先暴露的是 replicated initialization transient，而不是真实训练 batch 峰值。修复前不消耗比赛GPU，也不通过改精度、关EMA、改batch或冻结层偷换配方。

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

归档不表示旧计划所有尾部阶段都完成；只表示该文件不再是活动授权入口。旧 failure、阈值、hash 和未执行阶段必须继续保留。

## 3. 保留但非活动的研究线

- `AXIS_PI05_STANDARD_LIBERO_30K_RESEARCH_PLAN.md`：长期标准LIBERO 30K/full-FT研究的事实与路线来源；当前执行授权由sharded-init修复计划收窄到最多10 steps。
- `TRACK3_REVERSE_DATA_SYNTHESIS_PLAN.md`：Track 3 逆向任务研究，当前不授权生成或训练。
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
