# R6 folded deployment diagnostic handoff

## 2026-08-31 审计修正与续执行入口

旧 S1.3 `s1_real_egl_pairs_failed` 保持不变，但审计确认当前报告有三项需要在新 namespace 修正：

1. `axis_reference_jax` 实际是 R6 folded JAX，不是 Axis step-0；
2. agentview worst/representative 的图像内容 SHA 相同，实质为重复 pair；
3. selector 未覆盖计划要求的 zero64/fixed64 trajectory。

这些缺陷不推翻 `9.66e-6～5.13e-5 > 1e-6` 的 failure；它们意味着当前四 pair 不能称为完整 EGL envelope。正确 venv 的全量测试已复核为 `99 passed, 7 skipped`，`pynvml` 不是 collection blocker。

后续不得直接恢复旧 S2。新 section 接手文件：

- `docs/AXIS_PI05_PROJECT_HANDOFF_20260831.md`
- `docs/AXIS_PI05_R6_EGL_STATISTICAL_RELEASE_PLAN.md`
- `data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json`

下一轮只执行新计划 E0，修正标签、content dedupe、trajectory coverage 和只读 reanalysis，clean commit 后停止审核；不得同时启动 GPU、closed-loop、FAST 或 ZIP。

截至 2026-08-30，本任务按 `AXIS_PI05_R6_FOLDED_DEPLOYMENT_AND_DIAGNOSTIC_SUBMISSION_PLAN.md` 停止在 S1.3；没有启动训练、FAST 导出、server/closed-loop、ZIP、外部上传或官方 Notebook。

## 已完成

- S0 release lock：R6 folded source、R6 fresh norm、processors、raw checkpoint、C20 manifest 和 worktree provenance 已锁定。
- S1.2 frozen grid：真实磁盘 `model.safetensors` + LeRobot pre/post processors，8 task × 4 seed = 32/32。
  - same-process PyTorch internal/external：bitwise exact；
  - fresh-process PyTorch external：max abs `0`；
  - JAX folded ↔ folded PyTorch internal max `1.4007091522216797e-6`，external max `1.043081283569336e-6`；
  - token/mask mismatch `0`，finite，gripper sign flip `0`；
  - ACTION/STATE normalization：`QUANTILES`。
- S1.3 已从 Q1 无损数组锁定每个 camera 的最差 transformed pair 和代表性非零 pair，并实际通过 folded PyTorch/LeRobot 与 JAX reference 运行。

## 硬停止

状态：`s1_real_egl_pairs_failed`

首个失败：`pair00_agentview_image_worst_transformed_pytorch_full_chunk_le_1e-6`

- agentview pair full/execute8 max：`9.655952453613281e-6`；
- wrist worst pair full max：`2.8252601623535156e-5`；
- wrist representative pair full max：`5.125999450683594e-5`；
- 四个 pair 均 finite、state/processor 通过、token/mask mismatch `0`、gripper sign flip `0`；
- Axis JAX reference 也呈现同方向的约 `9.8e-6`–`5.13e-5` pair drift，因此不能把现象归因于 PyTorch load 错误，也不能把 `1e-6` 门禁改宽。

该结果只说明 Q1 真实 EGL 图像微扰在当前严格 action sensitivity 门禁下未通过；不覆盖、不改写 raw-LoRA Q2 研究失败：
`n1_q2_policy_sensitivity_failed_raw_lora_fresh_replay`，max abs `1.0728836059570312e-6`。

## 未启动阶段

`S2, S3, S4, S5, S6, S7`

受影响产物与机器 summary：

- [S1.3 summary.json](./summary.json)
- [S1.3 pair manifest](./pair_manifest.json)
- [S1.3 E2E report](./e2e_gpu1/e2e_action_parity_report.json)
- [S1.3 E2E artifact](./e2e_gpu1/e2e_action_parity.npz)

GPU 证据完整：目标进程只出现在物理 GPU1；GPU0 仅有既有外部 PID `237527`，目标进程未触碰 GPU0。当前不得批准候选、不得生成 ZIP ready 状态，除非人工另行批准对 S1.3 失败进行新的诊断计划。
