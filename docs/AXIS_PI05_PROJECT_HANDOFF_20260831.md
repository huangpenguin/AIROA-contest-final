# AXIS Pi0.5 项目交接快照（2026-08-31）

> 用途：新 Codex section 从这里接手；不要从旧聊天重新猜测状态。
> 最新已审核执行 HEAD：`8938bcada74df00341ba020bf7c94f40eebd5178`
> branch：`codex/axis-pi05-audit-remediation`（当前无 upstream）
> nested OpenPI：`dafa5a27266996eecf28920cacc32f3629776d91`
> 当前总状态：`r6_track1_official_precheck_ready`
> 唯一活动计划：`docs/AXIS_PI05_TRACK1_OFFICIAL_PRECHECK_AND_SUBMISSION_PLAN.md`
> 计划索引：`docs/AXIS_PI05_PLAN_INDEX.md`
> closure execution parent：`be62f678c0927ebd6a0b2d29a63450c2edc74177`
> closure input：`78109647e95d79690511c922845141ff492a395f`
> closure execution commit：`327316f2ffdca0df3375748bcdd65b5ce263f8f8`

## 1. 一句话结论

canonical LIBERO 10 Hz、R6 LoRA 500-step、folded PyTorch、本地 B80 `49/80` 与 FAST-only Track 1 ZIP 都已完成。唯一冻结 ZIP 为 `9383571677` bytes、SHA-256 `95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a`；本地独立审核已批准进入官方 pre-check。当前不再训练或改包，下一步是用户授权后在配布 Blackwell 官方 Docker 验证 hash/offline startup/FAST markers/`/act <10s`，随后停在 assignment `1007` 的人工提交确认点。

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

## 6. E0 完成情况与审核结论

commit `7810964` 已完成第一次 E0，只读结果为：

```text
comparison groups                         1680
non-exact groups                            87
snapshot / trajectory groups              23 / 64
trajectory affected-frame occurrences       302
snapshot / trajectory content-unique      13 / 157
content-unique total                         170
content duplicates                           155
dynamics mismatch                              0
```

旧四 pair 只有 3 个 content-unique；误标的 Axis reference 已确认实际为 R6 folded JAX；旧 S1.3 failure/hash 保持不变。

第一次 E0 的 formal 问题已由后续 closure 修正；旧 E0 与旧 S1.3 产物均保留。

## 6.1 E0 Closure 结果

正式 closure 产物：

`data/reports/axis_libero_30k/post_r6_lora/r6_egl_statistical_release/e0_closure/`

```text
snapshot non-exact groups                 23
trajectory non-exact groups               64
snapshot candidate occurrences             23
trajectory affected-frame occurrences    302
snapshot content-unique                    13
trajectory content-unique                157
content-unique total                      170
content duplicate occurrences             155
renderer-only evaluable                   170
not evaluable                               0
dynamics mismatch                           0
coverage issues                             0
```

expected/observed snapshot 与 trajectory group sets 完全相等。`302` 是 affected-frame occurrence，`157` 是 trajectory content-unique pair，二者不能互换。closure summary SHA-256 为 `088376936c2ecba61258316bcafe48c70c04c7c115c62917222f7add2aba3c43`，content-unique manifest SHA-256 为 `b4ea8d3bed7f9cfc60f2f7f53613471e222d9857a8d6857f39b854720905c7bf`。

原 formal-closure 问题已由 E0 解决并保留为历史记录：

1. missing 数据现在显式为 not exact / not evaluable；
2. 缺失、shape/dtype mismatch 和无 changed frame 现在形成 coverage issue；
3. expected/observed 集合相等是 coverage complete 的必要条件；
4. 旧 S1.3 status、summary hash、manifest hash 有 immutability 回归测试；
5. 四个小型 closure 证据文件已显式纳入 Git。

独立审核还完成了：

- 在临时目录重新生成 closure；170 个 pair 的 semantic/content/source-occurrence hashes 与提交产物逐项一致；
- 旧 S1.3 summary/pair-manifest hash 复核不变；
- 全量测试独立复跑通过；
- root/nested clean，E1/GPU/model/closed-loop 均未启动。

审核结论：`approved`。E0 closure 计划已归档，允许按唯一活动计划执行 E1；不授权 E2。

## 6.2 E1 执行结果与 hard stop

E1 CPU lock 已独立提交，commit `7d545ef`；后续只修正 raw tree provenance 记录并提交 `bdadd67`。lock 保持 immutable：`actions_observed=false`、`170 pairs / 340 observations`、snapshot/trajectory `13/157`、source occurrence hash mismatch `0`、other-camera mismatch `0`、not-evaluable `0`。

JAX sentinel 在物理 GPU1 fresh process 通过 `4 observations`，model/norm/raw-tree provenance 与输入/noise hash 均准确，GPU0 未触碰。随后 JAX full run 使用 microbatch `4`，在 V100 CUDA OOM（申请 `216.00 MiB` 失败）后按计划 hard gate 停止；没有重试，没有启动 LeRobot/PyTorch，没有运行 comparator。

E1 状态：`e1_jax_full_inference_failed`。完整失败记录：`data/reports/axis_libero_30k/post_r6_lora/r6_egl_statistical_release/e1_action_envelope/summary.json`。已执行/未执行细节、GPU evidence 与 sentinel provenance 见该 namespace 的 `E1_ACTION_ENVELOPE_REPORT_ZH.md`、`THREAD_REPORT_ZH.md` 和 `jax_source/sentinel/report.json`。

本轮没有生成 170-pair action-response envelope；不得把 4-observation sentinel 当成全量结果。独立审核结论为 `remediation_required`：lock 与 sentinel 通过，JAX full 失败只证明 `microbatch=4` 不适合当前 V100 显存边界，不证明模型或输入损坏，也不批准 E1。

sentinel 实际按单 observation 推理，峰值约 `24717 MiB / 32768 MiB`；full run 在 `microbatch=4` 额外申请 `216 MiB` 时失败。下一活动计划固定使用 `microbatch=1`，先做 16-observation canary（前 4 actions 必须与旧 sentinel bitwise exact）。本轮 canary 已完成 16/16，但前 4 action 未 bitwise exact，状态为 `e1_jax_mb1_canary_failed`，因此未启动 340/340；已按 hard gate 停止并等待审核。成功后仍须 hard stop，另行审核后才可能授权 PyTorch source。

本轮 recovery 证据位于 `data/reports/axis_libero_30k/post_r6_lora/r6_egl_statistical_release/e1_jax_mb1_recovery/`：canary internal/external 前四比较分别为 `281/1280`、`226/280` exact，最大绝对差均为 `1.1920928955078125e-7`；GPU1 峰值约 `24706 MiB`，GPU0 未触碰。`[0,16)` 原子 shard 已保留但不得视作 full prefix。JAX full、PyTorch、comparator、closed-loop、E2、FAST、ZIP、训练、比赛 GPU、push、上传均未启动。

独立审核进一步核对：internal/external `>1e-6` 元素均为 0，全 action sign mismatch 与 external gripper sign mismatch 均为 0；全量测试为 `121 passed, 7 skipped`。旧 `e1_jax_mb1_canary_failed` 保留。新 future-run replay 门禁预注册为 internal/external max `<=2.384185791015625e-7`、sign mismatch `0`、finite/provenance exact；它不替代旧 EGL `1e-6` failure，也不代表模型性能通过。

此前的并行 JAX+FAST 方案已在执行前被更短的正分优先路径取代。今晚不再跑 JAX 340：唯一候选为 R6 step-500，唯一 critical path 为 R6-only FAST staging、真实 folded-PyTorch↔FAST 16-case parity、12-request FAST server、locked C5 和 ZIP 静态/离线门禁。locked C5 在看到 FAST rollout 前从既有 B80 锁定 3 个 source-success + 2 个 diagnostic cases，FAST 至少复现 `2/3` source successes才允许打包。

Axis init/原团队式 standard LIBERO full-FT 是后续正确研究方向，但不是今晚正分路径。短 full-FT 仍需重新 fold/FAST/server/ZIP，并可能引入 optimizer、NaN、norm 或灾难回归；今晚 R6 已有 `49/80` direct-policy 证据，先修部署链更快。ZIP 后另建 `1K→3K→10K→30K` full-FT 计划。

正分优先决策：`docs/reviews/axis_pi05/POSITIVE_SCORE_PIVOT_20260831.md`。

## 7. 测试、Git 与资源状态

正确测试入口的最新独立复核：

```text
.venv_axis_pi05/bin/pytest -q tests
122 passed, 7 skipped, 21 warnings（包含 R6 FAST release tests）。
```

此前“缺少 pynvml 导致 collection 中断”不是仓库指定 venv 的真实 blocker；正确 venv 当前可以收集 129 tests（122 passed + 7 skipped）。交接时不要为此随意安装依赖。

Git：

```text
branch               codex/axis-pi05-audit-remediation
latest audited HEAD   8938bcada74df00341ba020bf7c94f40eebd5178
nested OpenPI        dafa5a27266996eecf28920cacc32f3629776d91
root/nested          clean（审核 8938bca 开始时）
upstream             当前 branch 未设置 upstream
```

`outputs/` 与 `artifacts/` 被 `.gitignore` 排除。换 Codex section 但仍使用同一 workspace 时可见；换服务器/clone 时不会随 Git 出现，必须单独传输并校验 hash。

本地服务器：4× V100 32GB。GPU0 有外部 PID `237527`、约 14GB，占用不得触碰；GPU1–3 当前空闲。磁盘约 913GB 可用。

官方边界：配布开发 GPU 是 RTX PRO 6000 Blackwell，启动 Notebook 即计入本选1的 60 小时；正式 scorer 是单张 L4 24GB、CUDA13/PyTorch2.11，`/act` 与 `/reset` 每请求必须 `<10s`。Blackwell 能运行不能替代 L4 显存/延迟门禁。

## 8. 当前仍没有什么

- 没有官方 pre-check/hidden score；
- 没有人工提交或官方分数；
- 没有 V160/H160；
- 没有 1K/3K/10K/30K 长程 LoRA；
- 没有正式 Track 3 20 Hz checkpoint；ZIP 只存在本地 artifacts，未上传。

旧 `submission_axis_f2_m1_500_fast_*.zip` readiness 为 `NOT_READY`，行为 parity 明显失败，禁止作为 R6 fallback。旧 base ZIP 没有 `model_fast.safetensors`，存在 SLOW 路径风险。

## 9. Track 3 项目记忆

官方因 `check_no_leaks.py` 误发而向全体公开：Track 3 核心是现有 LIBERO task 的逆顺序/逆操作任务；具体任务和数量仍未公开。本战1保持该设计。Track 3 合成/20 Hz 适配是独立后续线，必须从完成的 10 Hz checkpoint 分叉并重新计算 20 Hz norm，不能污染当前 R6 10 Hz release 证据。

## 10. 新 section 的接手顺序

1. 阅读本文件与 `docs/AXIS_PI05_PLAN_INDEX.md`。
2. 阅读并只执行 `docs/AXIS_PI05_TRACK1_OFFICIAL_PRECHECK_AND_SUBMISSION_PLAN.md`。
3. 唯一允许进入 pre-check 的 ZIP hash 为 `95415a…c5f5a`；不得修改、重建、重新压缩或换旧包。
4. 启动配布 Notebook/私有传输前必须由用户确认消耗计时与当天 Track 1 次数；执行者不得自动上传。
5. 官方 pre-check 通过后先回传报告、关闭 Notebook，再停在 assignment `1007` 的人工提交确认点。
6. 不得重新训练、运行JAX/C20、使用 SLOW/旧 F2 FAST、自动 push/upload/submit或处理Track 2/3。

当前路径是：R6 FAST ZIP（本地审核 approved）→ 官方 pre-check → 人工 Track 1 提交 → 官方分数归档 → 后续 standard LIBERO full-FT 1K/3K/10K/30K。

## 11. 今晚正分路径执行结果：ZIP 等待人工决定

本轮只执行了今晚计划允许的 R6 step-500 FAST 路径，最终状态为
`r6_positive_score_zip_ready_for_human_decision`。真实 folded-PyTorch source、FAST
导出、视觉 key/weight/bias、strict reload、16-case parity、12 次 server `/act`、
锁定 C5 和 ZIP 静态/离线门禁均已通过。

- R6 release lock：`data/reports/axis_libero_30k/post_r6_lora/r6_positive_score_fast/R6_FOLDED_RELEASE_LOCK.json`；candidate 为 `r6_dual_l2_fresh500_r1`，500 steps，`L2_action_lora_heads`，QUANTILES。
- 8 个冻结 T1 fixture 及 manifest：`outputs/axis_pi05/r6_positive_score_fast/fixtures/`；8 cases × 2 policy seeds 形成 16 个 parity comparison。
- parity：finite `1.0`、gripper sign `1.0`、median abs `0.002796`、p95 abs `0.036217`、mean cosine `0.999170`，status `passed`。
- server latency：`12/12` 请求完成，最大 `1.491s`、p95 `1.013s`，全部低于 10s；FAST/chunk/execute/EMA/RTC markers 正确，SLOW/remap/missing-key/vision-key markers 均不存在。
- C5：5/5 coverage，3/3 source-success regression 复现；两个 diagnostic case 均无 HTTP、nonfinite 或 stall，`task07` 保留 collision 诊断。
- 证据报告目录：`data/reports/axis_libero_30k/post_r6_lora/r6_positive_score_fast/`，包括 `source_inventory.json`、`export_report.json`、`fast_submission_audit.json`、`key_mapping.json`、`fixed_input_parity.json`、`server_latency.json`、`track1_calibration/summary.json`、`official_docker_e2e.json` 和 `package_validation.json`。
- Track 1 ZIP：[submission_axis_r6_step500_fast_positive_score_20260831.zip](/home/huang/code/PARC2026_final/artifacts/submission_axis_r6_step500_fast_positive_score_20260831.zip)，大小 `9383571677` bytes，SHA-256 `95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a`；sidecars 为同名 `.sha256` 与 `.manifest.json`。
- ZIP manifest 明确为 `NOT_SUBMITTED`，`official_result=false`，`auto_submit=false`；本地离线等价报告也明确 `official_result=false`，不能当作官方分数。

本轮未执行 JAX 340、完整统计 envelope、comparator、E2、C20、FAST 之外的 SLOW fallback、训练、比赛 GPU、push、上传或提交。运行时只使用物理 GPU1；GPU0 仅存在外部 PID `237527`，本轮未触碰。下一步仅等待人工审核/提交决定。

## 12. R6 FAST ZIP 独立审核

commit `8938bca` 已独立审核为 `approved`，批准范围为官方 pre-check与人工提交决策。复核包括：ZIP SHA/bytes、全量 `unzip -t`、仓库官方 `validate_submission.py --static --json`、source/FAST/processor identity、813-key/vision/strict reload、16-case parity、server latency、C5与全量测试。

保留风险：`official_docker_e2e.json` 实际是本地离线等价检查；本地 cold startup约 `362s`，须使用 official `1200s` timeout重验；没有L4 24GB正式 peak-memory证据。`server_latency.json.health_error` 是 health成功前最后一次轮询错误，最终 health/reset/12 acts均成功，不需要重建ZIP。

审核记录：`docs/reviews/axis_pi05/R6_POSITIVE_SCORE_ZIP_REVIEW_20260831.md`。
