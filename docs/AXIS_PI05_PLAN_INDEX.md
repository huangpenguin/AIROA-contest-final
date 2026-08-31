# AXIS Pi0.5 计划索引与审核生命周期

> 更新日期：2026-08-31（Asia/Tokyo）
> 当前唯一活动计划：`docs/AXIS_PI05_E1_EGL_ACTION_ENVELOPE_PLAN.md`
> 当前状态：`e1_jax_full_inference_failed`
> 最新审核：`docs/reviews/axis_pi05/E0_CLOSURE_REVIEW_20260831.md`

## 1. 唯一活动计划

- `AXIS_PI05_E1_EGL_ACTION_ENVELOPE_PLAN.md`
  - 先用 CPU 冻结 170 pair / 340 observation 与固定 noise，并形成 lock commit；
  - 随后允许在同一张空闲本地 GPU 上顺序运行 folded JAX 与真实 LeRobot PyTorch source；
  - 只测 frozen-observation action-response envelope，不运行 environment 或 closed-loop；
  - 完成后必须停止审核，不授权 E2、FAST、ZIP、训练、上传或比赛提交。

任何执行者看到多个计划时，以本索引的“唯一活动计划”为准，不得从归档计划的未执行尾部自行恢复任务。

E0 closure 已通过人工审核。E1 是当前唯一授权阶段；不得跳到归档 EGL 计划中的 E2–E7。

本轮 E1 已完成 CPU lock 与 JAX 4-observation sentinel；JAX 340-observation full run 在 V100 GPU1 的 microbatch=4 触发 CUDA OOM，按 E1 hard gate 停止。LeRobot/PyTorch full inference、E1 comparator、closed-loop 与 E2 均未执行。当前仍只保留本 E1 计划，等待人工审核决定最小 remediation；不得自行重试或跳阶段。

## 2. 已归档的 R6 执行链

以下计划已完成其获授权阶段或已按 hard gate 停止，并被后续计划取代。文件保存在 `docs/archive/axis_pi05/`，只用于审计：

1. `AXIS_PI05_LORA_NAN_LOCALIZATION_AND_RECOVERY_PLAN.md`：R0–R6 完成，产生 R6 step-500。
2. `AXIS_PI05_POST_R6_LORA_SCALING_AND_SELECTION_PLAN.md`：执行至 N1，按 reproducibility 门禁停止。
3. `AXIS_PI05_N1_EVALUATOR_NONDETERMINISM_RECOVERY_PLAN.md`：执行至 D1，转入 policy-boundary 专项。
4. `AXIS_PI05_N1_POLICY_BOUNDARY_AND_DIAGNOSTIC_ZIP_PLAN.md`：执行至 Q2，raw-LoRA strict `1e-6` 失败。
5. `AXIS_PI05_R6_FOLDED_DEPLOYMENT_AND_DIAGNOSTIC_SUBMISSION_PLAN.md`：执行至 S1.3，folded real-EGL strict `1e-6` 失败。
6. `AXIS_PI05_R6_EGL_STATISTICAL_RELEASE_PLAN.md`：E0 已完成首次审核；E1–E7 从未获本轮执行授权。
7. `AXIS_PI05_E0_CLOSURE_BEFORE_E1_PLAN.md`：C0–C5 完成并通过独立审核，正式释放 E1。

归档不表示旧计划所有尾部阶段都完成；只表示该文件不再是活动授权入口。旧 failure、阈值、hash 和未执行阶段必须继续保留。

## 3. 保留但非活动的研究线

- `AXIS_PI05_STANDARD_LIBERO_30K_RESEARCH_PLAN.md`：长期标准 LIBERO 30K/full-FT 研究，当前 deferred。
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
