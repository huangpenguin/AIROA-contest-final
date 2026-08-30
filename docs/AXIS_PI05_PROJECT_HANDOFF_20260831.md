# AXIS Pi0.5 项目交接快照（2026-08-31）

> 用途：新 Codex section 从这里接手；不要从旧聊天重新猜测状态。
> 审计输入 HEAD：`be56e8edca77377b57a2941d0483564ce4112f79`
> branch：`codex/axis-pi05-audit-remediation`（当前无 upstream）
> nested OpenPI：`dafa5a27266996eecf28920cacc32f3629776d91`
> 当前总状态：`r6_egl_statistical_protocol_pending`
> 下一计划：`docs/AXIS_PI05_R6_EGL_STATISTICAL_RELEASE_PLAN.md`

## 1. 一句话结论

canonical LIBERO 10 Hz 数据、R6 两卡 LoRA 500-step、fold、LeRobot/PyTorch frozen parity 和 full-C20 environment dynamics 都已完成；R6 本地 B80 只比 Axis init 多 `1/80`，统计不明确。当前 blocker 不是训练 NaN 或模型加载，而是 EGL 的稀疏像素抖动会造成约 `1e-5～5e-5` action drift，使旧 strict `1e-6` sensitivity gate 失败。现在没有 R6 FAST、server、ZIP 或官方结果；下一 section 先修正 S1.3 报告，再预注册统计 closed-loop 协议。

## 2. 数据与复现基础

canonical source：`physical-intelligence/libero` 原生 10 Hz，不插帧、不重采样。

```text
dataset dir                     data/raw/hf/physical-intelligence/libero
size                            约 33G
episode parquet                 1693
.incomplete                     0
official payload files          1699
payload manifest sha256         8ede620026a9196a9d453508dfe2d4820e684d98cda1a5bc706cef68d1583cc3
payload bytes                   34938927454
meta/info.json sha256           77c0da93c67960bac803a8ea25dabf380d752c26c1d91c54151121dd10e7d82f
```

主要证据：

- `data/reports/axis_libero_30k/phase0_summary.json`
- `data/reports/axis_libero_30k/canonical_payload_manifest.json`
- `configs/axis_pi05/standard_libero_30k.yaml`
- `configs/axis_pi05/libero_plus_track3_20hz.yaml`

10 Hz canonical reproduction 与比赛 20 Hz adaptation 是两条独立线。20 Hz 分支必须从完成的 checkpoint 分叉、使用真实 20 Hz 数据并重新计算 norm；禁止插帧冒充 20 Hz。

## 3. 训练与 R6 checkpoint

当前研究候选：

```text
run id          r6_dual_l2_fresh500_r1
topology        L2_action_lora_heads / dual LoRA + action heads/time MLP
optimizer steps 500
GPU             local GPU1+GPU2，GPU0 未触碰
norm            R6 fresh per-dimension QUANTILES
norm sha256      4c803bf59c52677c36b6b420c614d8a7742e0847708443863e13e67cb80b18f5
```

raw Orbax：

```text
outputs/axis_pi05/axis_libero_lora_screen/
  L2_action_lora_heads/r6_dual_l2_fresh500_r1/500
tree sha256 d203635f1eb0b006ec27841db6ce5171f74a37035eca4291c8107c05838c276f
```

folded LeRobot/PyTorch：

```text
outputs/axis_pi05/axis_libero_lora_folded/
  r6_dual_l2_fresh500_r1_step500
model.safetensors bytes  14467165872
model sha256              5c73e1e8217dbb75a934c2b0165208e851f69e66cfe2ea0430590a29901e6d5b
```

ACTION/STATE normalization 为 `QUANTILES`，external action 7D、internal action 32D、chunk 10、execute 8。不要把 observation EEF 物理量范围与 OSC_POSE command action 混淆；persisted action 前三维约 `±0.94`，gripper 为 `{-1,+1}`。

R6 只是 500-step LoRA anchor，不是 Axis 原团队 30K full-FT 复现，也不是正式 winner。长程 `1K→3K→10K→30K` 尚未启动。

## 4. 本地性能证据

seeded B80 execute8：

```text
Axis step-0 success/collision/timeout/stall    48/80, 23, 19, 0
R6 step-500 success/collision/timeout/stall    49/80, 20, 15, 0
paired wins/losses/ties                         9/8/63
success delta                                  +0.0125
episode bootstrap 95% CI                       [-0.0875, 0.1125]
task-cluster bootstrap 95% CI                  [-0.0875, 0.125]
McNemar two-sided p                             1.0
```

结论：本地技术门禁通过、性能结果 `review_required`；不能称为显著提升。证据：

`data/reports/axis_libero_30k/phase1/rollout/r6_dual_l2_fresh500_r1_B80_seeded_paired_comparison/summary.json`

## 5. Evaluator / EGL 确定性进展

### 5.1 Q1 environment-only

C20 × 3 fresh-process replica：model/controller/simulator/qpos/qvel/8D policy state、post-init、warm-up step10、zero64/fixed64 数值动力学全部 exact。

renderer：1680 comparisons 中 87 个 non-exact；raw uint8 max diff 2，transformed float32 max diff `0.0156862736`，差异稀疏。状态：

```text
n1_environment_dynamics_locked_renderer_jitter_measured
```

证据：`data/reports/axis_libero_30k/post_r6_lora/n1_policy_boundary/q1_environment_full_c20/q1_environment_full_c20_summary.json`

### 5.2 Q2 frozen policy

- Axis step-0 fresh max `5.9604645e-7`：通过旧 `1e-6`。
- JAX folded dense fresh max `3.5762787e-7`：通过。
- raw-LoRA fresh max `1.0728836e-6`：只有一个元素超线，无 sign flip，但按原 hard gate failed，永久保留。

### 5.3 S1 real deployment

S1.2 实际读取 on-disk LeRobot model/processors，32/32；PyTorch same/fresh-process internal/external bitwise exact；JAX folded↔PyTorch external max `1.0430813e-6`、internal `1.4007092e-6`，均在 `1e-3` fold parity 门禁内，无 sign flip。

S1.3 四个 real EGL pair 的 PyTorch action drift：

```text
agentview              9.6559525e-6
wrist worst            2.8252602e-5
wrist representative   5.1259995e-5
```

全部 finite、token/mask/state contract 正常、gripper sign flip 0；R6 folded JAX 同量级。因此不像 PyTorch load/norm 错误，但 strict `1e-6` sensitivity gate failed。

## 6. S1.3 审计修正（新 section 必须先处理）

当前 S1.3 报告有三项语义/coverage 缺陷，不推翻 failure：

1. `axis_reference_jax` 实际是 R6 folded JAX，不是 Axis step-0；
2. agentview worst/representative 的内容 SHA 相同，实质重复 pair；
3. selector 只遍历 E1/E2/E3 snapshot，没有覆盖计划中的 zero64/fixed64 trajectory。

旧产物不得覆盖。新 section 按新计划 E0 生成 reanalysis，然后停止审核。

## 7. 测试、Git 与资源状态

正确测试入口的最新复核：

```text
.venv_axis_pi05/bin/pytest -q tests
99 passed, 7 skipped, 21 warnings
```

此前“缺少 pynvml 导致 collection 中断”不是仓库指定 venv 的真实 blocker；正确 venv 可以收集 106 tests。交接时不要为此随意安装依赖。

Git：

```text
branch               codex/axis-pi05-audit-remediation
audited input HEAD   be56e8edca77377b57a2941d0483564ce4112f79
nested OpenPI        dafa5a27266996eecf28920cacc32f3629776d91
root/nested          clean（生成本交接文档前）
upstream             当前 branch 未设置 upstream
```

`outputs/` 与 `artifacts/` 被 `.gitignore` 排除。换 Codex section 但仍使用同一 workspace 时可见；换服务器/clone 时不会随 Git 出现，必须单独传输并校验 hash。

本地服务器：4× V100 32GB。GPU0 有外部 PID `237527`、约 14GB，占用不得触碰；GPU1–3 当前空闲。磁盘约 913GB 可用。

官方边界：配布开发 GPU 是 RTX PRO 6000 Blackwell，启动 Notebook 即计入本选1的 60 小时；正式 scorer 是单张 L4 24GB、CUDA13/PyTorch2.11，`/act` 与 `/reset` 每请求必须 `<10s`。Blackwell 能运行不能替代 L4 显存/延迟门禁。

## 8. 当前没有什么

- 没有 R6 `model_fast.safetensors`；
- 没有 R6 FAST server/closed-loop；
- 没有 R6 submission ZIP；
- 没有官方 pre-check/hidden score；
- 没有 V160/H160；
- 没有 1K/3K/10K/30K 长程 LoRA；
- 没有正式 Track 3 20 Hz checkpoint。

旧 `submission_axis_f2_m1_500_fast_*.zip` readiness 为 `NOT_READY`，行为 parity 明显失败，禁止作为 R6 fallback。旧 base ZIP 没有 `model_fast.safetensors`，存在 SLOW 路径风险。

## 9. Track 3 项目记忆

官方因 `check_no_leaks.py` 误发而向全体公开：Track 3 核心是现有 LIBERO task 的逆顺序/逆操作任务；具体任务和数量仍未公开。本战1保持该设计。Track 3 合成/20 Hz 适配是独立后续线，必须从完成的 10 Hz checkpoint 分叉并重新计算 20 Hz norm，不能污染当前 R6 10 Hz release 证据。

## 10. 新 section 的接手顺序

1. 阅读本文件。
2. 阅读 `docs/AXIS_PI05_R6_EGL_STATISTICAL_RELEASE_PLAN.md`。
3. 只执行 Phase E0。
4. 运行正确 venv 的全量测试。
5. 更新 `PROJECT_HANDOFF_20260831.json` 与当前线程报告，clean commit 后停止审核。
6. 未经人工授权，不启动 E1 GPU run、closed-loop、FAST、官方 Notebook、HF upload 或比赛提交。

不要重新训练 R6，也不要恢复旧 F2 FAST 包。当前最短可信路径是：修正 S1.3 证据 → 锁定统计协议 → source closed-loop → FAST → ZIP。
