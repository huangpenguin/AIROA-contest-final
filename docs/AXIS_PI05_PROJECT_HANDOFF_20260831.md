# AXIS Pi0.5 项目交接快照（2026-08-31）

> 用途：新 Codex section 从这里接手；不要从旧聊天重新猜测状态。
> 新session先读：根目录`AGENTS.md`、`MEMORY.md`，然后按其中顺序核对本handoff、machine JSON、活动计划、最新审核与live cursor/PID。
> 最新审核状态：`remediation_required`
> interstep L0/L1 execution HEAD（已冻结）：`88796b2`
> branch：`codex/axis-pi05-audit-remediation`（当前无 upstream）
> nested OpenPI：`b58aff0`（interstep lifetime implementation）
> 当前总状态：`track3_runtime_y2_clean_release_ready`
> 唯一活动计划：`docs/AXIS_PI05_TRACK3_RUNTIME_PREFLIGHT_AND_BOUNDED_CONTINUATION_PLAN.md`
> 计划索引：`docs/AXIS_PI05_PLAN_INDEX.md`
> closure execution parent：`be62f678c0927ebd6a0b2d29a63450c2edc74177`
> closure input：`78109647e95d79690511c922845141ff492a395f`
> closure execution commit：`327316f2ffdca0df3375748bcdd65b5ce263f8f8`

## 1. 一句话结论

canonical LIBERO、R6 LoRA500-step、folded PyTorch与FAST-only ZIP均已完成；用户报告正式Track1=`0.47`、Track2=`0.33`、Track3=`0`。Track3 object-frame数学与hard-stop闭包已有进展，但X4 worker在0 physics step时因缺失runtime helper停止，且审计发现stall、close/lift/source-release顺序、周期heartbeat与完整input lock尚未闭合。新计划先完成runtime preflight和执行合同，再在同一条件链内继续三族bounded generation、LoRA、holdout与本地ZIP；用户无需Teleop或中途逐阶段审核。

Y0–Y2工程补救已完成并形成clean-release候选：完整输入bytes/SHA锁、canonical runtime helper smoke、正确阶段合同、真实stall早停、周期heartbeat、supervisor identity与`failed_gate`/`worker_exception`返回码分层均已写入新的Y运行时模块；finalizer自哈希、严格artifact检查、定向/全量CPU测试均通过。GPU0未触碰；复核时历史外部PID已退出并如实记录。Y3 nominal CPU canary尚未运行，Track3 reverse数据、训练与ZIP仍未开始。

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

`physical-intelligence/libero` metadata为10 FPS；IPEC标准20 FPS转换版的同一episode并非两倍帧：episode 0同为214帧，8D state与前6维action exact，timestamp从0.1改0.05秒，gripper从`{-1,+1}`改为`{0,1}`。训练loader按frame index取action chunk且不输入timestamp。比赛环境默认control_freq=20，历史R6 C5强制10只能作对照；正式Track1/2分数是20Hz闭环证据。LIBERO-plus仍是独立20Hz适配数据，混合前必须统一gripper并重算norm。

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
146 passed, 7 skipped, 23 warnings（interstep lifetime L0/L1 执行记录）。
```

此前“缺少 pynvml 导致 collection 中断”不是仓库指定 venv 的真实 blocker；最新执行记录为 134 tests（127 passed + 7 skipped）。交接时不要为此随意安装依赖。

Git：

```text
branch               codex/axis-pi05-audit-remediation
latest execution HEAD 35e67b6
nested OpenPI        b58aff0
root/nested          clean（审核 6239704 开始时）
upstream             当前 branch 未设置 upstream
```

`outputs/` 与 `artifacts/` 被 `.gitignore` 排除。换 Codex section 但仍使用同一 workspace 时可见；换服务器/clone 时不会随 Git 出现，必须单独传输并校验 hash。

本地服务器：4× V100 32GB。dense preflight 执行结束时四卡均约 `5 MiB`；每次新运行仍必须重新检查 PID/显存，不得凭历史状态假设空闲。当前磁盘预飞行记录可用 `894.73 GiB`。

官方边界：配布开发 GPU 是 RTX PRO 6000 Blackwell，启动 Notebook 即计入本选1的 60 小时；正式 scorer 是单张 L4 24GB、CUDA13/PyTorch2.11，`/act` 与 `/reset` 每请求必须 `<10s`。Blackwell 能运行不能替代 L4 显存/延迟门禁。

## 8. 当前仍没有什么

- 没有完整官方日志、submission ID 或 episode 级 hidden 结果；用户已报告 Track1 正式分数 `0.47`、Track2 正式分数 `0.33`；
- 没有 V160/H160；
- 没有 1K/3K/10K/30K 长程 LoRA；
- 没有正式 Track 3 20 Hz checkpoint；ZIP 只存在本地 artifacts，未上传。

旧 `submission_axis_f2_m1_500_fast_*.zip` readiness 为 `NOT_READY`，行为 parity 明显失败，禁止作为 R6 fallback。旧 base ZIP 没有 `model_fast.safetensors`，存在 SLOW 路径风险。

## 9. Track 3 项目记忆

官方因 `check_no_leaks.py` 误发而向全体公开：Track 3 核心是现有 LIBERO task 的逆顺序/逆操作任务；具体任务和数量仍未公开。本战1保持该设计。Track 3 合成/20 Hz 适配是独立后续线，必须从完成的 10 Hz checkpoint 分叉并重新计算 20 Hz norm，不能污染当前 R6 10 Hz release 证据。

## 10. 新 section 的接手顺序

1. 阅读本文件与 `docs/AXIS_PI05_PLAN_INDEX.md`。
2. 阅读并只执行`docs/AXIS_PI05_TRACK3_MATCHED_TERMINAL_STATE_RELEASE_PLAN.md`的L0–L5，之后hard stop。
3. 将Track1=`0.47`、Track2=`0.33`、Track3=`0`及ZIP `95415a…c5f5a`锁为当前official-score baseline；无上传回执的字段保持conditional provenance。
4. L0锁定K0–K4、三份公开HDF5和errata；旧raw evidence保持immutable。
5. L1只修复renderer/H.264/sensitivity/contact contract工具，不启动simulator或GPU。
6. L2固定task10/task20/task19各`demo_0/1/2`，每条必须使用同demo的`model_file + terminal full-state`，CPU三族各`3/3`。
7. L2通过后GPU3最多运行task19/demo_0、task10/demo_0、task20/demo_0三条camera release；成功也只停在`pending_executor_review`。

当前路径是：R6 FAST ZIP Track1=`0.47`/Track2=`0.33`/Track3=`0`（冻结）→ 公开full-state与20Hz数据（已在本地）→ step5 critical semantics与raw evidence closure（已通过）→ exact-model matched terminal-state清理与camera release → 新审核 → reverse executor/Teleop选择 → 30–60 exact reverse扩充与R6分叉LoRA。Dense 30K为冻结的独立研究线。

## 19. Dense full-FT allocator fraction recovery 执行结果

本轮按唯一活动计划执行 A0–A3。CPU lock、step-0 checkpoint tree/hash、fixed recipe 与 allocator contract 均通过；step-0 source tree 为 `9380fd770ced87368c82a05a4dbb8c00fd0f0e82c4e569879aacd127a01038f5`，源 checkpoint 未被修改。全量测试为 `.venv_axis_pi05/bin/pytest -q tests`：`132 passed, 7 skipped, 21 warnings`。

- `0.90`：`allocator_bfc_oom_during_compute`；restore step `0` 通过，allocator `6` samples、NVML `212` samples；microbatch index `0/1` 已完成，index `2` 未完成并在申请 `1207959552 bytes` 时失败，checkpoint/metric steps 仅 `[0]`/`[]`。
- `0.95`：在 A2 allocator OOM 后按条件执行；restore step `0` 通过，allocator `6` samples、NVML `208` samples；microbatch index `0/1` 已完成，index `2` 未完成并在申请 `75497472 bytes` 时失败，checkpoint/metric steps 仅 `[0]`/`[]`。
- 两次均为 FP32 dense、AdamW、EMA `0.999`、microbatch `1`、global batch `64`；无 optimizer update，A4/10-step、C20、长训均未执行。
- 4×V100 运行前后均无外部 PID，GPU0 未被外部进程触碰；四卡退出后均约 `5 MiB`。比赛 GPU、官方 Blackwell、FAST、ZIP、上传、push、提交均未启动。

证据目录为 `data/reports/axis_libero_30k/dense_fullft_allocator_fraction_recovery/`；当时的执行审核为 `docs/reviews/axis_pi05/DENSE_FULLFT_ALLOCATOR_FRACTION_EXECUTION_REVIEW_20260901.md`。该 hard stop 已由第20节的独立审核接管。

## 11. ZIP 生成阶段执行结果（历史快照）

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
- ZIP manifest 在生成时明确为 `NOT_SUBMITTED`，`official_result=false`，`auto_submit=false`；这些字段描述 package-generation 时点，不覆盖第13节所记用户后续人工提交。本地离线等价报告仍不能当作官方分数。

截至执行 commit `8938bca`，未执行 JAX 340、完整统计 envelope、comparator、E2、C20、FAST 之外的 SLOW fallback、训练、比赛 GPU、push、上传或提交；运行时只使用物理 GPU1，GPU0未触碰。之后的用户人工提交是仓库外动作，当前按第13/14节记录为正式Track1 score=`0.47`、provenance=`user_reported`。

## 12. R6 FAST ZIP 独立审核

commit `8938bca` 已独立审核为 `approved`，批准范围为官方 pre-check与人工提交决策。复核包括：ZIP SHA/bytes、全量 `unzip -t`、仓库官方 `validate_submission.py --static --json`、source/FAST/processor identity、813-key/vision/strict reload、16-case parity、server latency、C5与全量测试。

保留风险：`official_docker_e2e.json` 实际是本地离线等价检查；本地 cold startup约 `362s`，须使用 official `1200s` timeout重验；没有L4 24GB正式 peak-memory证据。`server_latency.json.health_error` 是 health成功前最后一次轮询错误，最终 health/reset/12 acts均成功，不需要重建ZIP。

审核记录：`docs/reviews/axis_pi05/R6_POSITIVE_SCORE_ZIP_REVIEW_20260831.md`。

## 13. 人工提交与本地 scorer 环境决策

用户已报告完成Track1人工提交并取得正式score=`0.47`；仓库仍没有submission ID、提交时间、截图或完整官方日志，所以provenance为`user_reported`，不得补造episode或隐藏任务细节。

README的“同一环境”仍指依赖与评测流程同构；但真实`0.47`已经取代补跑本地Docker的必要性。无需消耗配布Blackwell pre-check，也无需重建image。

新审核记录：`docs/reviews/axis_pi05/R6_LOCAL_SCORER_AND_SUBMISSION_DECISION_REVIEW_20260831.md`。

## 14. `0.47` 基线与 dense full-FT 决策

R6正式Track1=`0.47`确认FAST-only、vision/key映射与server链在真实scorer有效。冻结该ZIP与部署参数，不再从R6 raw LoRA续训：其500-step schedule已到horizon、无EMA、valid7 objective和LoRA topology均不是AXIS论文dense 30K配方。

新主线从Axis params、canonical LIBERO10Hz、fresh QUANTILES norm开始dense full-FT。约3.353B模型的FP32 params+Adam+EMA+grads下限约67GB；2×V100每卡下限约33.5GB且未含activation，预计只适合一次allocation canary。3卡不能精确构成uniform global64；首选GPU0自然释放后的4卡，或未来同一次Blackwell长训开头完成preflight。

审核记录：`docs/reviews/axis_pi05/R6_TRACK1_SCORE_047_REVIEW_20260831.md`。当时的活动计划已归档为`docs/archive/axis_pi05/AXIS_PI05_POST_SCORE_DENSE_FULLFT_PREFLIGHT_PLAN.md`。

## 15. Dense full-FT preflight 执行结果

本次从 root `3bbee76`、nested OpenPI `d7e41eb` 的 clean lock 开始，完成了 CPU-only F0/F1、一次 GPU1+GPU2 first-update canary，以及 GPU0 自然释放后的唯一一次 4×V100 条件预飞行。两卡在申请约 `1.12 GiB` 时 OOM；四卡在 XLA 将计算降到约 `25.00 GiB` 后，每卡申请 `54 MiB` 时 OOM。两次均为初始化/首个计算失败，没有 optimizer update、checkpoint 或 metric step。

因此当前总状态为 `dense_fullft_preflight_failed`。由于 F2 没有可用拓扑，checkpoint restore、C20、吞吐报告和 60 小时预算均未执行；不能据此判定模型能力失败，也不能修改 recipe 继续硬跑。GPU0 外部进程未触碰，四卡在进程退出后均约 `5 MiB`，没有启动官方 GPU、1K以上训练、FAST、ZIP、push 或上传。

完整证据位于 `data/reports/axis_libero_30k/dense_fullft_preflight/`，尤其是 `input_lock.json`、`memory_budget.json`、`two_gpu_canary.json`、`four_gpu_preflight.json` 及对应日志。

## 16. Dense full-FT OOM 审核与无损恢复决策

独立审核结论为 `remediation_required`。两卡/4卡均在 `init_train_state` 失败，还没有进入 batch/forward/backward。现有代码将 Orbax 恢复的约 `12.5 GiB` partial params 通过统一 `PartitionSpec()` replicated 到每卡，输出才是 FSDP state；四卡日志同时记录大量 donation-unusable warning 与 `26,840,528,788` bytes input/output arguments。

microbatch 是每卡每个 microstep 的样本数，当前 `1` 已是最小；global batch `64` 在4卡上由 `1×4×16`梯度累积实现。降 global batch 不会修复 batch 之前的初始化 OOM。

新活动计划只修复 sharded parameter initialization，保持 FP32、dense、AdamW、EMA `0.999`、global64。四卡先 init-only，然后唯一 first update，最多连续10 steps。本轮不启动比赛Blackwell；只有init修复通过而真实first update仍OOM，才建议把下次preflight与长训练合并进同一个60小时session。

审核记录：`docs/reviews/axis_pi05/DENSE_FULLFT_INIT_OOM_REVIEW_20260831.md`。当时的活动计划已归档为 `docs/archive/axis_pi05/AXIS_PI05_DENSE_FULLFT_SHARDED_INIT_RECOVERY_PLAN.md`。

## 17. Sharded init recovery 执行结果

CPU lock 从 root `28f4779`、nested OpenPI `92604d6` 的 clean 状态开始，旧 preflight OOM 证据 hash 保持 immutable；CPU 静态/toy sharding 门禁通过，全量项目测试为 `127 passed, 7 skipped, 21 warnings`。R3 四卡 init-only 通过：51 leaves、3,353,433,872 FP32 params、large replicated leaf `0`、sentinel exact、params/optimizer/EMA finite，且没有 data loader/forward/backward。

同一 fresh run 随后完成 step-0 atomic checkpoint，进入真实 first microbatch 计算；`train_step_accumulated_host` 在每卡申请 `136.69 MiB` 时 OOM，optimizer step 仍为 `0`，step1/10 未生成。GPU 监控峰值约 `24,897 MiB/card`，进程退出后四卡均约 `5 MiB`。该 OOM 的阶段确实是真实训练计算，但后续审核发现它只耗尽了 JAX 默认 75% BFC pool，不足以证明 32GB 物理容量用尽。历史状态 `dense_fullft_first_update_true_compute_oom_pending_official_gpu_review` 保留在原 summary，但不再是当前资源决策。

R5、restore、C20、60 小时预算、比赛 GPU、50/100/1K、FAST、ZIP、push、上传和提交均未执行。证据目录：`data/reports/axis_libero_30k/dense_fullft_sharded_init_recovery/`，主报告为 `summary.json` 和 `ROOT_CAUSE_REPORT_ZH.md`。

## 18. R4 allocator 独立审核

R4 的 allocator 环境变量全部 unset；JAX/JAXLIB 为 `0.5.3`。官方 JAX 文档说明默认在首次 GPU 操作时预分配 75% 显存。本地四卡峰值均为 `24897/32768 MiB=75.98%`；其中纯 75% pool 为 `24576 MiB`，约 `321 MiB` 差值可由 CUDA context/libraries/NCCL/allocator 外开销解释。因此 NVML 中看起来剩余的 `7871 MiB/card` 不在当前 BFC pool 的可用连续块内；这就是“才显示24.9GB也OOM”的原因。

审核结论为 `remediation_required`：先保持 preallocation，只设 `XLA_CLIENT_MEM_FRACTION=0.90`复测；仅 allocator OOM 才允许唯一 `0.95`。这不改 FP32、dense、loss、batch64、AdamW 或 EMA。不使用 preallocate false/其他 allocator，不启动比赛 GPU。

审核记录：`docs/reviews/axis_pi05/DENSE_FULLFT_R4_ALLOCATOR_REVIEW_20260901.md`。当时的活动计划现已归档为 `docs/archive/axis_pi05/AXIS_PI05_DENSE_FULLFT_ALLOCATOR_FRACTION_RECOVERY_PLAN.md`。

## 20. Allocator fraction 后续独立审核与重分类

root `c353ff0`、nested OpenPI `81d8220` 的0.90/0.95执行边界可信：step0 restore通过、配方未改、optimizer update为0、官方GPU及后续阶段未启动。但失败阶段标签和资源决策需要修复：两份raw allocator trace都记录了`microbatch_complete` index `0`与`1`，index `2`才未完成，并非首份microbatch失败。

`0.95`每卡`bytes_in_use`为`13,435,195,392 → 19,941,608,448 → 26,420,739,584`；第二份completed microbatch带来约`6.48GB/card`常驻增长。代码中`grads`在累积后继续存活，审计`values`字典也持有全部gradient leaves，足以形成下一份RHS计算时的完整梯度重复驻留。该证据证明当前实现/observer在0.95 pool失败，但不能证明数学配方的稳定累积需要超过32GB/card。

审核状态为`remediation_required`。旧计划已归档；当前唯一计划为`docs/AXIS_PI05_DENSE_FULLFT_ACCUMULATION_MEMORY_LIFETIME_RECOVERY_PLAN.md`。只允许释放临时梯度引用、建立per-micro memory plateau证据并在本地4×V100验证step1/step10；不批准直接启动Blackwell、C20、100/1K、长训或提交。

审核记录：`docs/reviews/axis_pi05/DENSE_FULLFT_ACCUMULATION_LIFETIME_AUDIT_20260901.md`。

## 21. Accumulation memory-lifetime 执行结果

本轮按唯一活动计划完成 M0/M1，并在四张自然空闲的本地 V100 上执行 M2；没有启动官方 GPU、Blackwell、M3、100/1K/长训、C20、FAST、ZIP、上传、push 或提交。M0 只修正旧 summary 标签并保留 raw logs/checkpoints；M1 只加入 gradient temporary lifetime boundary、释放引用和 CPU 测试，不改变 FP32、dense、AdamW、EMA `0.999`、microbatch `1`、accumulation `16`、global batch `64` 或顺序。

CPU 全量回归为 `137 passed, 7 skipped, 23 warnings`。M2 的 immutable step-0 copy 为 `24,879,588,870` bytes，atomic rename 与 source hash 通过；restore sentinel、scheduler/optimizer/EMA presence 与 finite、四卡自然空闲和 GPU0 外部 PID 门禁均通过。训练使用 `XLA_CLIENT_MEM_FRACTION=0.95`、preallocate=true，实际在 microbatch index `1` 的 `_add_gradients(grad_sum, grads)` 申请 `72 MiB` 时 BFC OOM；compute indexes 为 `[0,1]`，`temporaries_released` 仅 `[0]`，optimizer update=`0`，没有 step1 checkpoint 或 metric。

最终执行状态为 `dense_fullft_accumulation_lifetime_remediation_failed`，不是模型能力失败结论。M3 未启动；四卡退出后均约 `5 MiB`，source step-0 hash 保持 `9380fd770ced87368c82a05a4dbb8c00fd0f0e82c4e569879aacd127a01038f5`。完整小型结果见 `data/reports/axis_libero_30k/dense_fullft_accumulation_memory_lifetime_recovery/summary.json` 与 `m2_step1.json`，执行记录见 `docs/reviews/axis_pi05/DENSE_FULLFT_ACCUMULATION_MEMORY_LIFETIME_EXECUTION_20260901.md`。当前停止等待审核，不自动进入下一修复或比赛资源。

## 22. M2 Gradient Merge 独立审核

独立审核确认M2 hard stop正确，但失败不能归因于index1 forward/backward：raw trace的`microbatch_compute_complete`为`[0,1]`，`accumulation_complete/temporaries_released`只有`[0]`。index1 compute完成时每卡约`26.42GB` live，随后普通逐叶`_add_gradients`在创建新output时申请`72MiB`失败。

nested OpenPI `e31c2bf`已移除审计`values`长期引用并显式block/delete，但`_add_gradients`仍是非JIT、非donated的`x+y`。它必须同时保留旧accumulator、current grads和新output；因此当前结果只证明普通merge不适合0.95 pool，不证明稳定donated accumulation或单microbatch需要更大GPU。

审核状态为`remediation_required`。旧memory-lifetime计划已归档；当前唯一活动计划为`docs/AXIS_PI05_DENSE_FULLFT_DONATED_ACCUMULATOR_RECOVERY_PLAN.md`。D2必须先证明两份真实microbatch donated merge后live bytes回到baseline；失败即停止，不批准Blackwell、降精度/降batch或继续完整step。

审核记录：`docs/reviews/axis_pi05/DENSE_FULLFT_M2_DONATION_AUDIT_20260902.md`。

## 23. Donated accumulator D0–D4 执行结果

本轮执行输入为 root `8e9f3cf`、nested OpenPI `5c29b0e`，只运行 D0–D4。D0/D1 的 CPU/code gate 通过：严格 FP32、左结合、按 microbatch 顺序的 donated tree add 已落地；optimizer observer 因会使 donated state 失效而移除，实际 update 配方不变，NVML/allocator runtime trace 保留。全量测试为 `141 passed, 7 skipped, 23 warnings`。

D2 两份真实 merge-only canary 通过：compute/accumulation/release indexes 均为 `[0,1]`；merge 后每卡 live bytes `19916301568–19916302080`，最大漂移 `512` bytes（门限 `256MiB`）；donated alias `6479127152` bytes；optimizer updates 为 `0`。

D3 最终 fresh run 通过 `16/16` compute/merge/release，完成一次 optimizer update（step `1`、examples seen `64`），finite gate 通过并成功原子写入 step1 checkpoint。此前两次 D3 运行的 observer/logging 生命周期故障已保留在带后缀的输出目录，最终独立 report 为通过结果。

D4 从internal step1续跑，先完成`16/16` accumulation和一次optimizer update，`train_state.step`由`1`前进到`2`；随后在下一轮`optimizer_step_before=2`中，microbatch `0`完成并释放，microbatch `1`在真实forward/backward阶段因BFC allocator申请`143327232` bytes（`136.69MiB`）失败，四卡均为`RESOURCE_EXHAUSTED`。continuation return code为`1`，未生成step10 checkpoint；step2也没有atomic checkpoint/restore，不能升级为checkpoint门禁通过。旧runner启动的restore-only子进程在确认continuation失败后立即终止，保留为不完整restore证据，不视作通过。

最终执行状态为`dense_fullft_donated_accumulator_remediation_failed`。没有启动Blackwell/官方GPU、C20、100/1K/长训、FAST/ZIP、上传、push或提交；本地四张GPU均已释放至约`5MiB`。主证据为`data/reports/axis_libero_30k/dense_fullft_donated_accumulator_recovery/{d2_merge_canary.json,d3_step1.json,d4_step10.json,summary.json}`。

## 24. D4 独立审核与跨 step 生命周期决策

审核输入为root `c8b0a42`、nested OpenPI `5c29b0e`，结论为`remediation_required`。D2 donated alias/plateau、D3 step1 checkpoint、D4 hard stop均可信；但原始allocator trace把失败定位得更精确：step1 checkpoint恢复基线约`13.435GB/card`，第一轮持有accumulator时约`19.942GB/card`，update `1→2`完成后仍约`19.942GB/card`，下一轮microbatch 0 release升到约`26.511GB/card`。新增约`6.57GB/card`与一份gradient tree量级一致。

这证明当前阻塞已从“gradient merge无法复用output”推进到“optimizer update之后full-tree-equivalent device allocation跨step存活”。可能涉及未donate的gradient输入、device metrics异步依赖、旧state/update引用或observer生命周期；现有证据尚不能唯一归因，禁止把推断写成根因。

旧donated计划已归档。唯一活动计划为`docs/AXIS_PI05_DENSE_FULLFT_INTERSTEP_STATE_LIFETIME_RECOVERY_PLAN.md`：L0闭合step语义，L1只修复host/device对象生命周期并证明数学exact，L2从step1完成update到step2后只探测下一microbatch 0的plateau；通过才允许step3和step10。仍不授权比赛GPU、Blackwell、降精度/降batch、C20或长训。审核记录为`docs/reviews/axis_pi05/DENSE_FULLFT_D4_INTERSTEP_LIFETIME_AUDIT_20260902.md`。
## 25. Interstep L0/L1 执行阻塞

本轮执行输入为 root `35e67b6`、nested OpenPI `b58aff0`。L0 已锁定 D4 raw evidence、step1 checkpoint tree hash `9aeeb1a9b1fdf428486b476930ca9117c28d822dd108b4efaeebf911ca0fafef` 及正确的 step 语义；L1 仅加入 gradient/state donation、host metrics materialization 和 device temporary release 边界，CPU 全量测试为 `146 passed, 7 skipped, 23 warnings`。

启动 L2 前的只读资源检查发现 GPU0 被外部 YOLO 训练 PID `1133706` 占用约 `14.3GiB`，GPU utilization 非零；GPU1–3 各约 `5MiB`。L0 因此按 fail-closed 规则记录为 `dense_fullft_interstep_l0_blocked_external_gpu`。没有触碰或终止外部 PID，也没有启动 L2/L3/L4、Blackwell/官方 GPU、C20、长训、FAST/ZIP、上传或提交。

该次执行在此停止等待审核；证据为`data/reports/axis_libero_30k/dense_fullft_interstep_state_lifetime_recovery/l0_cpu_lock.json`及`docs/reviews/axis_pi05/DENSE_FULLFT_INTERSTEP_L0_BLOCKED_EXECUTION_20260902.md`。后续资源决策以第26节为准。

## 26. Interstep 阻塞审核与 Track 3 正分改道

独立审核输入root `88796b2`、nested OpenPI `b58aff0`。L0 semantic/hash lock和L1生命周期实现通过；全量测试为`146 passed, 7 skipped, 23 warnings`。GPU0外部PID `1133706`未被触碰，L2–L4未运行。审核状态为`blocked`，但只阻塞锁定的4卡dense实验，不阻塞GPU1–3上的LoRA/rollout。

三卡不能无损复现当前4卡dense配方：microbatch/device `1`和integer accumulation不能得到global batch `64`，3-device sharding/checkpoint topology也不同。它只能成为新recipe，当前不以此绕过门禁。interstep计划已归档，L0/L1代码与step1 tree hash `9aeeb1a9b1fdf428486b476930ca9117c28d822dd108b4efaeebf911ca0fafef`保留；即使GPU0释放，也要另开恢复审核，不能自动运行L2。

用户提供的官方公开规则确认Track3核心是现有LIBERO任务的逆向操作。现有R6在Track1/2有正分而Track3为0，故继续纯正向LIBERO LoRA缺乏直接supervision。唯一活动计划改为`docs/AXIS_PI05_TRACK3_REVERSE_LORA_POSITIVE_SCORE_PLAN.md`：以公开LIBERO构造generic reverse init/goal/instruction，真实环境20Hz执行并3/3 replay验证30–60条pilot，再从R6 raw step500以GPU1+2训练最多两个保守LoRA arm、GPU3做reverse holdout和retention；通过后最多生成独立Track3 FAST ZIP并停止。禁止使用泄漏脚本中的family/hidden task/数量/命名等内部细节。

审核记录：`docs/reviews/axis_pi05/INTERSTEP_BLOCKED_TRACK3_REDIRECTION_AUDIT_20260902.md`。

## 27. Track 3 T0 CPU-only 执行记录（待审核）

T0 在 root `61d4a9f`、nested OpenPI `b58aff0` 上完成。R6 raw/folded/FAST/正分 ZIP、canonical 10Hz、LIBERO-plus 20Hz metadata、action/state/gripper contract 已锁定；R6 资产和正分 ZIP 未修改。GPU1–3 idle gate 通过；GPU0 外部 PID 只做只读观察且未触碰。evaluation-only denylist 与 canonical training manifest exact overlap 为 `0`，没有输出 denylist identifiers。T0 没有启动 CUDA context、simulator、数据生成或训练。

证据：`data/reports/axis_libero_30k/track3_reverse_lora/t0_cpu_lock.json`、`T0_CPU_LOCK_REPORT_ZH.md`。下一步只允许 T1 generic reverse compiler/pilot；在真实 20Hz、predicate、3/3 replay 的 30–60 条 accepted episode 门禁前不得启动 LoRA。

## 28. Track 3 T1 compiler / pilot 执行结果

T1 CPU compiler 从公开 canonical LIBERO catalog 与 public BDDL/init 生成40个候选 reverse spec，`32 train / 8 template-holdout`，但没有把它们当作 demonstration。GPU3 的真实20Hz探测未能获得可审计 terminal snapshot：task10 reverse object-on-plate 初始状态失稳，task17 motion primitive 未抓取；公开 episode0 的 forward init 对齐/replay 也未产生可验证 terminal success。accepted episode 为 `0`，没有达到30–60 episode、3 families、6 templates、2 scenes、每 family两 template及3/3 replay门禁。

当前状态：`track3_reverse_rules_or_pilot_blocked`。T2、两卡 LoRA、T3、FAST、ZIP、closed-loop、E2、C20、训练、比赛 GPU、上传和提交均未启动；GPU0 未触碰，R6资产和正分 ZIP 未修改。候选 manifest 与报告：`data/reports/axis_libero_30k/track3_reverse_lora/t1_reverse_task_manifest.jsonl`、`T1_CPU_COMPILER_REPORT_ZH.md`、`T1_PILOT_BLOCKED_REPORT_ZH.md`。继续工作需要新计划明确提供可审计的公开 terminal-state 获取与 teleop/planner 路径。

## 29. T1 独立审核与公开终态恢复决策

独立审核输入为root `1e71ec5`、nested OpenPI `b58aff0`。T1边界、测试和hard stop可信；40/40只表示公开reverse spec编译成功，不表示物理initial state或demonstration成功。`0 accepted episode`定位到旧terminal source/action proposal路径：`.pruned_init`不是成功终态，LeRobot 8D observation state不是完整MuJoCo state，predicate swap也不会产生稳定物理状态。约`2.29mm`最近init差异与3/3 replay门禁均不得放宽。

官方公开LIBERO原始HDF5包含完整`data/demo_i/states`；官方检查代码会把demonstration末状态送入`env.set_init_state`。新唯一活动计划为`docs/AXIS_PI05_TRACK3_PUBLIC_TERMINAL_TELEOP_PLANNER_RECOVERY_PLAN.md`：P0下载并锁定公开原始HDF5；P1验证official full state，版本不兼容时改用当前环境R6公开正向成功live full-state capture；P2实现20Hz teleop并取得3 families canary；P3从成功teleop抽取闭环keypose planner；P4扩展到30–60条、6 templates、2 scenes且逐条3/3 replay；P5只做schema/manifest/loader forward smoke。完成后状态只能到`track3_reverse_pilot_ready_pending_review`，仍不得训练。

GPU0继续禁止触碰；GPU3仅用于simulator/capture，GPU1/2不启动训练；比赛GPU、LoRA/dense、FAST、ZIP、C20、E2、上传与提交均未授权。独立`LIBERO-plus`目录的既有dirty状态原样保留。审核记录：`docs/reviews/axis_pi05/TRACK3_T1_TERMINAL_SOURCE_AUDIT_20260902.md`。

## 30. Track 3 P0–P2 执行停止点

本轮执行输入为 root `5c0e92b`、nested OpenPI `b58aff0`。P0 已下载并锁定官方公开四 suite 各 10 个 HDF5，共 `40` 个、`33784856577` bytes；40/40 具备 `data/demo_0/states/actions/rewards/dones`，task/BDDL 映射通过，reverse BDDL 由官方 BDDL 与 `.pruned_init` CPU 重新编译。P0 证据和 hash 见 `data/reports/axis_libero_30k/track3_reverse_lora/{p0_lock.json,p0_download_manifest.json,p0_hdf5_schema_inventory.json,p0_task_mapping.json}`。

P1 使用 GPU3 fresh process 完成 task10 support、task20 containment、task19 articulation 三个公开 family 的 `3/3` restore/settle/reverse-not-success canary，共 3 个 snapshot；activation task16 的官方 state 与 live public forward capture 均在 no-op settle 后失稳，独立摘要保留在 `data/reports/axis_libero_30k/track3_public_terminal_recovery/p1_activation_task16_failed_summary.json`，不计入成功。

P2 keyboard-first recorder 原型已实现并通过其CPU self-test、py_compile；支持20Hz、7D OSC_POSE、完整per-frame state、front/wrist、timestamp/deadline miss、start/end state与视频写出。执行提交将状态记为`track3_teleop_input_pending_human`，没有accepted episode；P3/P4/P5未启动。执行报告声称全量测试为`155 passed, 7 skipped, 23 warnings`，但该claim已被第31节的clean-tree复跑否定。

GPU0 外部任务未触碰，GPU1/2 未训练，官方 GPU、LoRA/dense、FAST、ZIP、C20、E2、上传和提交均未启动。`LIBERO-plus` 既有 dirty 状态未修改。执行时曾给出的GPU3人工命令已由第31节审核撤销，不得继续使用。

## 31. P0–P2 独立审核与 Recorder 修复决策

独立审核输入root `9dac308`、nested OpenPI `b58aff0`，结论为`remediation_required`。P0/P1核心证据保留：本地实际有40个HDF5、总计`33784856577` bytes、无partial；task10/19/20各自fresh-process restore/settle/reverse-not-success为3/3；activation task16正确排除。但P1仍标`pending_video_review`，accepted demonstration仍为0。

P2不能交给人工：真实record loop在no-key时把7D action全置0，gripper=0会被自身校验拒绝；当前`DISPLAY`/`WAYLAND_DISPLAY`为空且代码没有实时preview，camera toggle没有可见消费；reset字段未实现按键与state reload；predicate成功会在3/3 replay和human review前直接写入`accepted/`。报告中的`--operator <operator-name>`也不能原样粘贴。

证据闭合还有两处错误：`p0_lock.json`实际SHA-256为`be12795b3d18d46f774022d817789a14d64a5e16c8e27bee58e256c29acb4ab5`，不是报告/handoff中的`06c2d8...`；在clean `9dac308`上复跑两个相关测试文件得到`1 failed, 6 passed`，旧测试仍断言P0状态，因此提交记录的全量155通过不可复现。

第30节最后一句授权旧GPU3命令已被本节撤销。旧计划已归档；新唯一活动计划`docs/AXIS_PI05_TRACK3_TELEOP_RECORDER_REMEDIATION_PLAN.md`只允许：闭合hash/tests，修复action/reset，增加localhost headless实时画面，实现pending→fresh-process 3/3 replay→human review promotion，再做GPU3 idle/reset/preview diagnostic。完成后hard stop于`track3_teleop_ready_for_human_after_remediation`；在此之前不要运行旧命令。审核记录：`docs/reviews/axis_pi05/TRACK3_P0_P2_RECORDER_READINESS_AUDIT_20260902.md`。

## 32. Track 3 R0–R4 Recorder remediation 完成停止点

本轮实际执行输入为 root `07de201`（计划旧标签 `9dac308` 的直接后继），nested OpenPI `b58aff0`；P0 lock、P1 summary 与 activation exclusion 的原始 evidence 未改写。R0 correction/hash closure 见 `data/reports/axis_libero_30k/track3_public_terminal_recovery/R0_CORRECTION_SUMMARY.json`，其中锁定实际 P0 lock SHA-256 `be12795b3d18d46f774022d817789a14d64a5e16c8e27bee58e256c29acb4ab5` 与 P1 summary SHA-256 `62db697a1ff6afb5d157c9fc786c77eb17a0bdf1054edb060fea50b3f3fe211e`。

R1–R3 已完成：共同 action state machine 使 no-key 永远保持 `-1/+1` gripper，支持 pause、camera、reset、abort、断连安全控制；localhost-only front/wrist preview 要求首帧被取到后才开始记录；candidate 先 atomic 写入 `pending/`，fresh-process replay `3/3` 与完整 human-review sidecar 通过后才允许 atomic promotion 到 `accepted/`。R3 只实现门禁，本轮没有 candidate、replay、human review 或 promotion，accepted demonstration 仍为 `0`。

R4 仅使用 GPU3，任务顺序 `19 → 10 → 20`。三项 reverse goal reset false、20 个 no-key tick、preview root/front/wrist HTTP 200 均通过；task19 额外通过 motion/pause/camera/reset byte-equal/abort，task10 与 task20 完成 idle gate。GPU3 执行前后均 `5 MiB / 32768 MiB`，GPU0 未触碰，GPU1/2 未训练。真实 review asset 的 codec/hash 见 `data/reports/axis_libero_30k/track3_public_terminal_recovery/R4_GPU3_NONHUMAN_CANARY.json`；没有把 PNG hash 冒充 H.264。

当前唯一状态为 `track3_teleop_ready_for_human_after_remediation`。人工命令固定使用 task19、operator `operator-local` 与 localhost `8765`；人工顺序只能是 task19→task10→20。不得执行旧 teleop 命令，也不得进入 P3/P4/P5、LoRA/dense、FAST、ZIP、C20、E2、官方GPU、上传或提交。`LIBERO-plus` 既有 dirty 状态保持原样。

## 33. R0–R4 独立审核与 Task19 单条释放

独立审核输入root `15791f9`、nested OpenPI `b58aff0`，结论为`approved`，但只批准task19单条人工pending canary及其fresh-process 3/3 replay。审核确认R1 no-key/gripper/reset状态机、R2 loopback first-frame preview、R3 pending/replay/review promotion门禁与R4 GPU3 task19→10→20非人工canary均符合计划；独立全量测试为`158 passed, 7 skipped, 23 warnings`。

task19 front contact sheet已目视确认中间抽屉打开、robot/cabinet可见且settle无明显漂移；wrist信息有限但front足够作为主画面。实际视频codec为`mp4v`，另有exact raw RGB NPZ，记录真实且不冒充H.264。R4没有调用完整人工record/replay路径，R3 promotion也只有fixture测试，因此task19仍必须作为第一个端到端canary。

旧remediation计划已归档。新唯一活动计划为`docs/AXIS_PI05_TRACK3_TASK19_HUMAN_CANARY_PLAN.md`：H0检查GPU3/端口/空目录，H1只人工关闭中间抽屉并写一条pending，H2只对返回的实际目录replay 3/3，H3停在front/wrist视频人审前。不得创建human_review、promote到accepted、录task10/20或进入planner/训练。审核记录：`docs/reviews/axis_pi05/TRACK3_TELEOP_R0_R4_READINESS_AUDIT_20260903.md`。

## 34. 公开 Primitive / 自动生成路线规划决策

用户账户内可访问的研究对话“Track3逆向训练方案”没有发现一个可直接使用的成熟Track3 reverse expert dataset，但给出了需独立复核的公开组合：LIBERO-Infinity负责reverse init/goal/language，原始LIBERO/LIBERO-plus/LIBERO-Gen提供成功reverse-relevant primitives，Behavior Prompting/LIBERO-Gen提供20Hz waypoint/scripted collection。研究对话属于线索，不是项目provenance；所有source/commit/license/schema必须在新计划S0从primary source重新验证。

本地已确认可供S1挖掘的20Hz原料为IPEC五套`5,614` episodes与`lerobot/libero_plus` `14,347` episodes，合计`19,961 episodes / 3,080,750 frames`；另有40个原始LIBERO full-state HDF5及task10/19/20三族restore canary。`data/eval_manifests/axis_pi05_eval_v1/public_t3_holdout.jsonl`的40 cases继续作为evaluation-only denylist。`exact_reverse_demo`、`reverse_relevant_primitive`与`generator_reference_only`必须分开，不得把forward task或`inverse_view`重标成Track3 supervision。

task19人工计划在H0前被取代并归档，没有产生candidate；recorder/replay实现不删除。新唯一活动计划为`docs/AXIS_PI05_TRACK3_PUBLIC_PRIMITIVE_MINING_AND_AUTOGEN_PLAN.md`，只授权S0–S5：primary-source lock、CPU semantic mining、每族少量选择性schema/replay、reverse semantic adapter、scripted executor及support/containment/articulation每族最多1条non-human pending canary。新增外部数据上限20GiB，GPU仅允许GPU3，官方GPU和训练均禁止。

任一hard gate失败即停止；尤其close-drawer自动primitive失败时不得自动转人工，而应进入`track3_autogen_close_drawer_gap_pending_review`。三族都通过也只能进入`track3_autogen_three_family_canary_pending_human_video_review`，等待审核后才可另建30–60条exact reverse扩充、auxiliary primitive bank和从R6 step500分叉的小步LoRA计划。

本次只落盘计划与状态指针，未执行S0或任何实验。计划切换后全量CPU回归为`158 passed, 7 skipped, 23 warnings`；root/nested在计划修改前clean，`LIBERO-plus`既有dirty内容未触碰。提交前还需完成JSON/Markdown路径、`git diff --check`与最终worktree校验。

## 35. Track 3 S0–S5 自动执行停止点

本轮以 root `b0078dc6ffc54036006c732c890f02824ca593a0`、nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1` 为 S0 lock 输入。S0 primary source/revision/license 复核、S1 本地语义挖掘、S2 选择性 schema/replay 与 S3 reverse semantic adapter 均通过。S1 共索引 `21,961` episodes / `3,080,750` 本地20Hz frames：`exact_reverse_demo=0`、`reverse_relevant_primitive=19,704`、`generator_reference_only=2,452`，denylist overlap=`0`。S2 的 full-state replay 明确采用 `states[0]` restore 后执行 `actions[1:]`，没有新增下载；S3 生成三族静态 reverse BDDL，`official_track3_claim=false`。

S4/S5 严格按 `support_remove → containment_remove → articulation_close` 只尝试一次 support canary。task10 在 `descend_object` 阶段 120 步超时，最终 predicate 为 false；保留了 132 帧未裁剪失败资产（完整 simulator states、policy states、predicate/subgoal/contact/deadline/action、front/wrist H.264）于 `data/reports/axis_libero_30k/track3_public_autogen/diagnostics/support_remove_task_10/`。按计划停止，没有 retry、containment/articulation 继续、人工 Teleop、human review sidecar、accepted promotion 或训练；`pending_autogen=0`。

GPU3 仅作已授权 simulator/replay，postflight 为 `6 MiB`；GPU0/1/2 未被本轮执行触碰。没有使用官方 GPU、FAST、ZIP、上传或比赛提交。当前唯一状态为 `track3_scripted_executor_failed`，后续只能由新的审核决定最小 remediation 或是否另开人工兜底计划；本交接不授权恢复旧 task19 命令。

## 36. S0–S5 独立审核与 support SE(3) 恢复授权

独立审核输入 root `73f675477a41a80241a7549f12f991187ecaaa25`、parent `b0078dc6ffc54036006c732c890f02824ca593a0`、nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`。commit/hash、root/OpenPI clean、`LIBERO-plus` dirty-preservation、单次 support 尝试与 hard stop 均核对无误；S0 source lock 和 S2 selective replay 保留。

审核结论为 `remediation_required`，原因有三项。第一，S1 identity/family counts 来自 `task_rows + episode_rows`，所以三类总和 `22,156` 不等于 `21,961` episodes，旧 `identity_classes_separate` 仅验证 key set。第二，S3 `passed` 只有静态 predicate-list 不等检查，缺少计划要求的三族 fresh-process `3/3`、20-step settle 和受控 goal true/false；三条 instruction 还含 `its akita black bowl`、`floor target object`、`wooden cabinet 1 middle` 等错误文本。第三，support 执行器只发 xyz，旋转恒零，抓取目标为 object center + `0.025 m`。

失败轨迹的 approach 在 12 steps 达到 `0.003496823 m`，descend 的最小误差为 `0.027499298 m`，最终为 `0.032033743 m`；接触数由 20 上升至最高 36。结合 front/wrist 抽帧，当前最强假设是目标位姿/几何接触错误，而非 120-step timeout 太短。旧 telemetry 没有 EEF quaternion、object pose 与 contact pair name，因此下一计划必须先重建根因，不能直接增加 timeout。

旧计划已归档；新唯一活动计划 `docs/AXIS_PI05_TRACK3_SUPPORT_AUTOGEN_SE3_RECOVERY_PLAN.md` 只授权 G0–G4 实验与 G5 收尾：v2 分层统计、typed instruction、三族动态 predicate 门禁、task10 公开成功 HDF5 alignment/相对 SE(3) 抓取模板、6D OSC_POSE 和一次 support-only canary。通过也只能写入 `pending_autogen_v2/` 并 replay `3/3` 后停；不授权 containment/articulation、人工、批量数据、训练、norm、FAST/ZIP、官方 GPU、上传或提交。完整审核见 `docs/reviews/axis_pi05/TRACK3_PUBLIC_AUTOGEN_S0_S5_AUDIT_20260903.md`。

## 37. G0–G1 恢复执行停止（2026-09-03）

本次 G0 lock 有效，旧 evidence 未改写；S1 v2 将 `195` 条 task rows 与 `21,961` 条 episodes 分开计数，identity sums 分别为 `195` 和 `21,961`，exact reverse 仍为 `0`。三条 S3 typed language 通过内部标识检查。

S3 动态门禁使用 GPU3 跑三族各 `3/3` fresh process，共 9 次。task10/task19 的 forward HDF5 `states[0]` reverse goal 为 false；三族 P1 reverse initial/goal、受控 destruction 和 object/fixture settle 检查通过，但 robot policy-state 20-step 漂移分别为 `0.1306809517`、`0.1388652157`、`0.5143735875`，均超过 `0.05` 门限。因此状态为 `track3_reverse_semantic_adapter_v2_failed_support_remove`，不得进入 G2。

执行期间修正了受控破坏的 HDF5 state offset（time scalar 在 index 0，qpos offset 为 1），并复跑完整 9 次；这只修正测试实现，不改变 hard stop。没有提取 task10 成功抓取位姿、没有重建旧失败轨迹、没有实现 SE(3)、没有 retry support，没有创建 pending，未触碰训练/FAST/ZIP/官方 GPU/上传。详细中文报告：`docs/reports/AXIS_PI05_TRACK3_SUPPORT_AUTOGEN_SE3_RECOVERY_G1_CLOSEOUT_20260903.md`。

## 38. G0–G1 独立审核与 dynamic validator 对齐授权（2026-09-03）

审核输入 root `179045a6d7a7998bf1a54452ef79739c9383c09f`、parent `8e9883cb748ecf15e5ff1fb988e2707e02dd153e`、nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`。root/OpenPI clean、`LIBERO-plus` 既有 dirty 未触碰、GPU 边界与 hard stop 均核对通过；S1 v2、typed language 与 HDF5 qpos offset 修复获批准。

审核发现旧 G1.3 的两个结论性缺陷。第一，三份 HDF5 的 exact MuJoCo XML 存在 `model_file` attribute，hash 分别为 task10 `e9a508e...8830`、task19 `ccf958e...0257`、task20 `4e0497a...723`，均不等于 fresh reverse env XML；task10/task19 的 fresh XML 每次还不同。MuJoCo state 不携带完整 model body/site placement，故跨 model 的 `states[0]` predicate false 不是合法反向语义证据。第二，旧 settle 把 joint sin/cos/velocity、EEF pose/quaternion 与 gripper qpos/qvel 合成一个 max，并连续发送 `gripper=-1`；object/fixture 漂移实际仅 `1e-7～1e-11`，不支持场景失稳结论。

因此审核为 `remediation_required`，但不是退回 Teleop 或宣布 reverse BDDL 无效。旧 G1 report 保持 immutable，旧计划归档；唯一活动计划改为 `docs/AXIS_PI05_TRACK3_DYNAMIC_GATE_MODEL_STATE_RECONCILIATION_PLAN.md`。它只授权：HDF5/model compatibility fail-closed、reverse BDDL native reset 的逐 predicate 检查、按单位拆分的 controller/gripper settle，以及 GPU3 上 task10 单次 canary 后三族各 `3/3` 非抓取动态 release。即使全通过也必须停在 `track3_reverse_semantics_v2_dynamic_validated_pending_se3_review`，等待新审核才可恢复 SE(3) 与真实 support canary。完整审核见 `docs/reviews/axis_pi05/TRACK3_SUPPORT_SE3_G0_G1_AUDIT_20260903.md`。

审核后全量测试为 `171 passed, 7 skipped, 23 warnings`。

## 39. Dynamic gate reconciliation D0–D4 收尾（2026-09-03）

D0 在新 namespace `data/reports/axis_libero_30k/track3_dynamic_gate_reconciliation/` 完成输入锁定：旧 evidence 28 项、三份官方 HDF5、三份 P1 attempt-1 snapshot 均保留 hash；HDF5 `states[0] == init_state` 均为真，time/qpos offset 为 `0/1`，exact/pending/accepted reverse demo 仍为 `0/0/0`。`LIBERO-plus` 只读取并保持原有 dirty 状态。

D1 的 validator/schema CPU 门禁修复后为 `15 passed`：reverse initial 逐条调用 reverse env predicate，native reset 后按 `sim.forward → controller.update → controller.reset_goal`，7D OSC_POSE 使用零 gripper command 保持当前 target；四元数、named field、单位阈值、warm-up/measurement 分窗和 fail-closed schema 均有回归测试。旧 `g1_s3_dynamic_gates.json` hash 未变化。

D2 仅在 GPU3 启动一次 task10 fresh process，未抓取、未写 demo。首次运行在 validator site detail 将原始 `TargetZone` 错当 state wrapper、调用不存在的 `get_geom_state`，故没有形成有效 predicate/settle 证据；状态为 `track3_native_reverse_dynamic_canary_failed_support`。随后只修复 validator 并补 CPU 回归，D2 不重跑。

因此 D3 未运行（`0/9`），D4 已 hard stop。没有 SE(3)、demo/pending、训练、Teleop、FAST/ZIP、官方 GPU、上传或提交；本轮结果不解释为 reverse semantics 成功或失败。证据：`d0_release_lock.json`、`d2_task10_canary.json`、`runs/d2_task10_canary_run_1.json`、`summary.json`、`D4_CLOSEOUT_REPORT_ZH.md`。下一步需要新的审核和最小 validator remediation 计划。

## 40. D0–D4 独立审核与 Native Validator 恢复授权（2026-09-03）

审核输入 root `84c2655f610bb957b0ad1ec2a952776bf023821e`、parent `f0034b9976e6de81644d1535d60328006bbe83cf`、nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`。D0 evidence、单次 D2、D3 `0/9`、范围边界、root/OpenPI clean 与 `LIBERO-plus` 既有 dirty preservation 均核对通过。审核结论为 `remediation_required`，不是 reverse semantics failure。

D2 唯一持久化结果为 `worker_exception: TargetZone has no get_geom_state`，且 `worker_returncode=0`；异常发生在 predicate detail telemetry，未形成 native predicate、settle 或 controlled-destruction 证据。后置修复从 `object_states_dict` 获取 `SiteObjectState` wrapper，与 LIBERO 源码一致；但 fake unit test 尚不能代替三个真实 native env 的 runtime schema。新计划还要求 predicate truth 与 advisory detail 分层，以及 exception 的非零 OS return code。

旧计划已归档；唯一活动计划为 `docs/AXIS_PI05_TRACK3_NATIVE_REVERSE_VALIDATOR_RECOVERY_PLAN.md`。R0/R1 只做 evidence/runtime/进程契约；通过后 R2 只允许一次 task10 recovery canary。R2 全通过可在不改代码/spec/threshold 的前提下执行 R3 三族各 `3/3`，随后 R4 hard stop。即使 `9/9` 通过也不授权 SE(3)、抓取/demo、Teleop、批量生成、训练、norm、FAST/ZIP、官方 GPU、上传或比赛提交。审核记录：`docs/reviews/axis_pi05/TRACK3_DYNAMIC_GATE_D0_D4_AUDIT_20260903.md`。

审核后全量测试为 `180 passed, 7 skipped, 23 warnings`。

## 41. Native Reverse Validator Recovery R0–R4 收尾（2026-09-03）

本轮按唯一活动计划 `docs/AXIS_PI05_TRACK3_NATIVE_REVERSE_VALIDATOR_RECOVERY_PLAN.md` 完成并停止。R0 新 namespace 锁定了 root 输入 `84c2655f...`、观察到的审核 descendant `3748c2eb...`、nested OpenPI `b58aff0...`、旧 D2/D4 与三族 v2 specs；旧 D2 的 `TargetZone.get_geom_state`、worker returncode=`0`、有效 predicate/settle evidence=`0` 保持不变。`LIBERO-plus` 仅读取，原 dirty 状态未触碰。

R1 使用三个独立 fresh process 和真实 `ControlEnv(has_renderer=false, has_offscreen_renderer=false, use_camera_obs=false)` 完成 schema probe，task10/task20/task19 三族 `3/3 schema_passed`。逐条 native `_eval_predicate` truth 与 geometry/joint/contact detail 已分层；对象使用 `ObjectState`、区域使用 `SiteObjectState`；7D OSC_POSE zero hold、named robot/scene snapshot、target resolution、finite/schema 和 worker return-code 契约均通过，HDF5 未作为 predicate oracle。

R2 在 GPU3 仅执行一次 task10 `recovery_1`。worker returncode=`0`、schema_complete=`true`、reverse goal 全 false，但 native reverse reset 的 reverse initial predicates 未全部为 true，故最终状态为 `track3_native_reverse_dynamic_failed_support_reverse_initial_or_goal`。这不是旧 D2 exception，也不是 settle/SE(3) 结论；settle 与 controlled destruction 因 native initial gate fail closed 未执行。R3 按授权未运行（`0/9`），R4 已停止。

本轮没有 exact/pending/accepted reverse demo，没有 SE(3) 抓取、人工 Teleop、批量生成、训练/norm、FAST/ZIP、官方 GPU、上传或比赛提交。机器证据：`data/reports/axis_libero_30k/track3_native_reverse_validator_recovery/r0_release_lock.json`、`r1_runtime_schema_probe.json`、`r2_task10_recovery_canary.json`、`runs/r2_task10_recovery_run_1.json`、`summary.json`、`R4_CLOSEOUT_REPORT_ZH.md`。等待下一次审核。

## 42. R0–R4 独立审核与 Init Stabilization 授权（2026-09-03）

审核输入 root `ed0ec3bff61c09a213300181959491dd107bf778`、parent `3748c2ebc2a9e5fc66ca90657cf8a4639488f7d7`、nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`。R0/R1、有效单次 R2、R3 `0/9`、hard stop、root/OpenPI clean 与 `LIBERO-plus` dirty preservation 均核对通过；审核状态为 `remediation_required`。

R2 的 7 条 task10 init atoms 中 plate→table region、cream-cheese→table region、bowl→plate 为 false；goal 为 false。bowl/plate XY 完全相同，Z 为 `1.08/0.97 m`，说明 `ObjectBasedSampler` 已放置 bowl，但 reset 瞬间还没有满足运行时 `On` 的 contact 条件。R1 的三族 native reset 都至少有一个 background site relation false。LIBERO 的 `:init` 用于构建 placement samplers，而运行时 `On` 使用另一套 contact/geometry predicate；本地官方 lifelong evaluator 在 set-init 后固定执行 5 个 zero-action physics steps，README 示例执行 10 个。

R2 外层 summary 还把未执行 settle/destruction 错写成 false，并把明确“无抓取”写成 false；下一计划须改成 passed/failed/not-evaluated 三态。旧 R2 保持 immutable，不直接改 BDDL。唯一活动计划 `docs/AXIS_PI05_TRACK3_REVERSE_INIT_STABILIZATION_SEMANTICS_PLAN.md` 先对 original-forward/reverse 做 `t=0/1/5/10/20` 对照，并要求三族 operator-critical source relation 在固定 step5 成立且稳定；通过才允许 GPU3 上一次 settled task10 canary及三族 `3/3`。仍不授权 BDDL 修改、SE(3)、demo、Teleop、训练、FAST/ZIP、官方 GPU、上传或提交。完整审核：`docs/reviews/axis_pi05/TRACK3_NATIVE_VALIDATOR_R0_R4_AUDIT_20260903.md`。

审核后全量测试为 `182 passed, 7 skipped, 23 warnings`。

## 43. Init Stabilization I0–I4 执行收尾（2026-09-04）

本轮按唯一活动计划 `docs/AXIS_PI05_TRACK3_REVERSE_INIT_STABILIZATION_SEMANTICS_PLAN.md` 执行。I0 锁定 root `ed0ec3bff61c09a213300181959491dd107bf778`、旧 R2/R4、三族 v2 BDDL/source、nested OpenPI 与 zero reverse-demo counts；I0 immutable input hash 在 I4 重验全部保持，`LIBERO-plus` 仅记录既有 dirty 状态且未触碰。

I1 通过：三族 operator-critical source 固定为 task10 `(On akita_black_bowl_1 plate_1)`、task20 `(In orange_juice_1 basket_1_contain_region)`、task19 `(Open wooden_cabinet_1_middle_region)`；reverse goal relation 单独登记并要求 false。background initializer 逐条保留 truth/detail，不在无 source contract 时升级为 behavior gate；未执行项使用 `not_evaluated`，不再写成 false。

I2 完成 `12/12` 个独立 CPU/no-renderer fresh processes（task10 forward `3/3`、task10 reverse `3/3`、task20 reverse `3/3`、task19 reverse `3/3`）。schema、finite、detail、OS return-code 均通过。三族 reverse critical source 在固定 step5 为 true，step5→20 连续为 true；reverse goals 同窗口连续为 false。task10 forward 对照观察到 background initializer false，支持分层解释。

I2 的 named scene stability 未通过：support reverse 最大 object rotation drift `0.0564 rad`，containment reverse 最大 object translation drift `0.0215 m`，articulation reverse 也有 rotation/translation 超过既有阈值。因此 I3 的 GPU3 task10 canary 与三族 `3/3` 均为 `0` 次；没有 SE(3)、抓取/demo、Teleop、批量生成、训练/norm、FAST/ZIP、官方 GPU、上传或提交。`settle`、`controlled_destruction` 为 `not_evaluated`，不是失败；exact/pending/accepted reverse demo 仍为 `0/0/0`。

机器证据：`data/reports/axis_libero_30k/track3_reverse_init_stabilization/` 下 `i0_release_lock.json`、`i1_semantics.json`、`i2_stabilization_time_series.json`、`runs/`、`summary.json`、`I4_CLOSEOUT_REPORT_ZH.md`。当前状态：`track3_reverse_init_stabilization_failed_named_scene_stability`；已按 I4 hard stop，等待下一次审核。

最终校验：定向测试 `24 passed`；全量测试 `189 passed, 7 skipped, 23 warnings`；JSON/Markdown/`py_compile`/`git diff --check` 均通过。

## 44. I0–I4 独立审核与 Operator-Scoped Release 计划（2026-09-04）

独立审核批准 `0dec0b1` 对 I0–I4 的执行与 hard stop，但结论为 `remediation_required`。旧 I2 对所有命名 object/site 取 full-scene max：task20 的 `21.5 mm` 来自背景 `salad_dressing_1`，关键 orange juice仅约 `5.54e-7 m`；task19 超限来自背景 bowl，目标 drawer site/joint约 `1e-7`；task10 bowl虽旋转 `0.0564 rad`，但translation小于 `1 mm`、`On`从step5到20连续为真、bowl↔plate contact持续存在。9条reverse run的完整init/goal truth vector均为零跃迁。

因此旧 `track3_reverse_init_stabilization_failed_named_scene_stability`永久保留为旧合同结果，但不能重标为reverse init失败。旧计划已归档；唯一活动计划为 `docs/AXIS_PI05_TRACK3_OPERATOR_SCOPED_STABILITY_RELEASE_PLAN.md`。J0–J2只做input lock、预注册contract、既有时序复算、9条带完整contact telemetry的新reverse和task20/task19 forward对照；全部通过后J3最多只在GPU3运行一次task10 canary及三族各`3/3` production-like动态验证。J4必须hard stop。

本审核仍不授权BDDL修改、SE(3)、抓取/demo、Teleop、批量生成、LoRA/dense训练、norm、FAST/ZIP、官方GPU、上传或比赛提交。审核记录：`docs/reviews/axis_pi05/TRACK3_INIT_STABILIZATION_I0_I4_AUDIT_20260904.md`。

## 45. Operator-Scoped Stability Release J0–J4 收尾（2026-09-04）

J0/J1 已通过并锁定旧证据与 operator contract。J2.1 对旧 I2 的 9 条 reverse 复算为已保存字段 `9/9` 通过，但历史 contact matrix 为 `not_evaluated`，不能单独释放 J3。J2.2 已真实完成 15 个 CPU/no-renderer fresh processes：reverse `9`、forward `6`。以 step5 为 participant/contact baseline 并修正 goal-false expectation 后，reverse operator gate 为 `8/9`、forward control 为 `6/6`、新 contact matrix 为 `15/15` 完整。

唯一失败为 task19 run1：step5→20 的完整 init predicate truth vector 在 step20 出现 background predicate 跃迁。该结果按固定 gate 记录为失败，没有删背景、改阈值或用 forward 对照反推。J3 因 J2 未达 `9/9` 而为 `not_evaluated`，GPU3 task10 canary、三族 `3/3`、H.264 与 sensitivity probe 均未执行。

J4 已生成 `data/reports/axis_libero_30k/track3_operator_scoped_stability/summary.json` 与 `J4_CLOSEOUT_REPORT_ZH.md`；targeted/full tests、JSON parse、py_compile、diff check 全通过（`194 passed, 7 skipped, 26 warnings`）。旧 I2 hash、旧 full-scene failure、LIBERO-plus 既有 dirty 与 reverse-demo `0/0/0` 保持不变。最终状态：`track3_operator_scoped_cpu_release_failed_truth_vector_transition_task19_run1`；execution commit 为 `bc2419e62ff05a9584d1d6033d22b0e55d03a240`；已 hard stop，等待审核，禁止自行重跑 J2、进入 J3、SE(3)、demo、Teleop、批量生成或训练。

## 46. J0–J4 独立审核与 Background Predicate Convergence 计划（2026-09-04）

独立审核批准`bc2419e`/`f969393`对J0–J4的执行与hard stop，但结论为`remediation_required`。逐atom复算15个raw reports只发现一次transition：task19 reverse run1的background initializer `(On akita_black_bowl_1 main_table_akita_black_bowl_region)`在step5–19为false、step20为true。critical `Open`连续true、goal `Close`连续false，drawer site/joint约`1e-7`量级，robot、participant和contact全部通过。其余8个reverse与6个forward没有transition。

该变化是background initializer朝BDDL预期状态单调收敛；旧J1要求完整truth vector绝对不变，因此旧J2`8/9`和hard stop仍然有效且永久保留，但不能解释为task19或reverse start失败。新合同只允许background init最多一次`false→true`，继续禁止`true→false`、回摆、critical source变化和goal变化。

审核还确认：`j2_summary.json`引用的15个raw worker reports（共`16,072,261 bytes`、每条21 steps）和15个corrected reports都被`.gitignore`排除，Git中为0条；summary不能替代逐帧/contact raw证据。机器handoff顶层current commit/tests/next-plan也仍为上一轮值。唯一活动计划`docs/AXIS_PI05_TRACK3_BACKGROUND_PREDICATE_CONVERGENCE_RELEASE_PLAN.md`先做raw evidence closure与fresh-checkout hash验证，再只读复算；通过才允许GPU3 task19 canary和三族`3/3`。仍不授权SE(3)、demo、Teleop、训练或比赛GPU。审核报告：`docs/reviews/axis_pi05/TRACK3_OPERATOR_SCOPED_J0_J4_AUDIT_20260904.md`。

## 47. Background Predicate Convergence K0–K4 收尾（2026-09-04）

K0 已将审核锁定的 15 份 J2 raw worker reports 显式纳入 Git，合计 `16,072,261 bytes`；manifest canonical hash 为 `83a23cb06c2e0d882c70b38bde8104cefc78f4728b67de1b36955fc12f5de2b8`，独立 fresh checkout 验证 `15/15` 文件 bytes/hash 一致。K1 冻结了 critical source 严格 true、reverse goal 严格 false、background initializer 只允许单调 `false→true` 的合同，并保留未知角色、缺失 atom/step、truth/detail 未执行和非 bool 的 fail-closed 规则。

K2 只读使用 manifest 内的 raw telemetry，未启动 simulator，结果为 reverse `9/9`、forward `6/6`；critical/goal transition 均为 `0`，background 允许的 `false→true` 为 `1`，forbidden `true→false` 与回摆均为 `0`。旧 J2 exact-vector `8/9` 与 task19 run1 的原始失败保持 immutable。

K3 按授权仅在 GPU3 执行 1 条 task19 canary；因为 initial background predicate 在 step19→20 出现 `true→false`，renderer schema、两路 H.264 及 direct sensitivity gates 未整体通过，立即停止。worker exception 为 `0`，三族 matrix 为 `not_evaluated`（`0` 次），没有 retry、换 seed 或修复后重跑。K4 已生成 `data/reports/axis_libero_30k/track3_background_predicate_convergence/k4_summary.json` 与 `K4_CLOSEOUT_REPORT_ZH.md`，状态为 `track3_background_convergence_gpu_failed_task19_canary` 并 hard stop。

本轮完整测试为 `201 passed, 7 skipped, 26 warnings`；targeted tests `12 passed`，py_compile、JSON parse、diff check 通过。GPU0/1/2 未触碰，GPU0 外部 PID `1133706` 未触碰；SE(3)、grasp/demo、Teleop、批量生成、训练/norm、FAST/ZIP、官方 GPU、上传/提交均未执行。reverse demo counts 仍为 `exact/pending/accepted = 0/0/0`。等待下一次审核决定后续最小 remediation，当前不得重跑 GPU 或进入 SE(3)。

## 48. K0–K4 独立审核与 Matched Terminal-State 计划（2026-09-04）

独立审核输入root `8490f8dde1148aec0aec4d3d105082b44cfbb286`、K2 frozen `cb8e898640a66e9baf5a587f21d5f44a84494a8e`、nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`。K0的15份raw closure与fresh-checkout、K1/K2只读复算、K3失败后停止和K4资源边界获批准；审核状态为`remediation_required`。

K3 task19 raw canary显示，step5已有39个bowl↔middle-drawer geom contacts，最深penetration约`-0.00680 m`；到step20仍为39个、约`-0.00550 m`，同时bowl的background `On(table region)`由true退为false。旧contact rule把step5已经存在的body pair当作diagnostic，因而physical gate显示passed；本地native reverse-BDDL随机初态不能直接晋升为清洗后的训练源。

三个独立工具缺陷也已确认：renderer flags实际位于wrapper内部，尽管两路camera均为`21/21`仍被误报；ffmpeg目标为`*.mp4.partial`而无法选择muxer；task19 destruction把WoodenCabinet qpos从`-0.1464`继续减到`-0.2664`，根据公开阈值仍为Open，raw sensitivity实际是`true/true/true`。旧K4中文报告中的`true/false/true`不准确，但旧证据不原地修改。

旧计划已归档。新唯一活动计划为`docs/AXIS_PI05_TRACK3_MATCHED_TERMINAL_STATE_RELEASE_PLAN.md`：L0/L1锁定errata并修复工具；L2固定使用三份公开HDF5中task10/task20/task19各`demo_0/1/2`的同demo `model_file + terminal full-state`，要求CPU三族各`3/3`；通过后GPU3最多做三族各一条camera release。L5必须hard stop。当前不授权action/demo、SE(3)、Teleop、训练、FAST/ZIP、官方GPU、上传或比赛提交。审核报告：`docs/reviews/axis_pi05/TRACK3_BACKGROUND_CONVERGENCE_K0_K4_AUDIT_20260904.md`。

## 49. Matched Terminal-State L0–L5 收尾（2026-09-04）

L0/L1 已完成：锁定旧 K0–K4、三份公开 HDF5 及固定的 task10/task20/task19 `demo_0/1/2` 选择；每条选择 reward 持续成功后缀的第一份 state。新 validator contract 覆盖 wrapper/underlying renderer flags、`name.partial.mp4` H.264 atomic publish、public WoodenCabinet absolute close midpoint，以及无 baseline grandfather 的 contact v2。L0/L1 没有启动 simulator 或 GPU。

L2 完成 9 个 fresh CPU/no-renderer exact-model processes：task10 `3/3`，task20 `0/3`，task19 `0/3`。task10 三条通过；task20 的 fixed clean gate 因目标物 participant drift 和严格 operator-background contact 未通过，task19 因 robot arm/EEF drift 和严格 operator-background contact 未通过。L3 task19 camera 与 L4 task10/task20 camera 均为 `not_evaluated`，没有触碰 GPU3。

L5 已按失败门禁 hard stop（execution commit `19cb1ea`）：`exact/pending/accepted reverse demo = 0/0/0`；没有生成 action/demo，没有 SE(3)、grasp、Teleop、批量生成、训练/norm、FAST/ZIP、官方 GPU、上传或提交。最终测试 `211 passed, 7 skipped, 26 warnings`，JSON parse、py_compile、diff check 通过；`LIBERO-plus` 原有 dirty 状态未触碰。证据：`data/reports/axis_libero_30k/track3_matched_terminal_state_release/summary.json`、`l2_summary.json`、`contact_ledger.json`、`L5_CLOSEOUT_REPORT_ZH.md`。下一步仅等待审核。

## 50. L0–L5 独立审核与 Composed Reverse-Start 授权（2026-09-04）

审核输入closeout `0dc402e2d70e6576c5861a674e0aac04e4427ef0`、execution `19cb1eab9556b6e9e0f8b3f9d4b49fb3893b81a0`、nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`。9份raw report hash、9个fresh process、L3/L4未执行、禁止项与hard stop获批准；审核结论为`remediation_required`。

9/9均通过same-demo identity、public forward goal、reverse critical source和reverse goal-false，因此task20/task19的`0/3`不是reverse语义失败。task20的orange juice在step5→20继续平移`2.088–7.390 mm`；task19主要是forward terminal robot回稳。全部9条demo的reward positive suffix都只有最后一帧，无法从HDF5选择post-success dwell。

`contact_contract_v2`还把gripper/robot↔operator contact错误归入“无关background”：task20/demo1为约`11.2 µm`的finger↔orange-juice接触，task19/demo2为最大约`0.263 µm`的right-finger↔drawer接触。旧raw与failure保持不变，新contact-v3必须按receiver、robot/gripper、unrelated background分层。

旧计划已归档；唯一活动计划为`docs/AXIS_PI05_TRACK3_COMPOSED_REVERSE_START_STABILIZATION_PLAN.md`。它固定使用同demo terminal scene qpos、同demo init robot/gripper qpos与全零qvel，先运行task20/demo0 CPU canary，通过后补齐9条CPU matrix，全部通过后GPU3最多三族各一条camera canary。M5必须hard stop；只允许report namespace中的start-state candidate，不授权action/demo、SE(3)、Teleop、训练、FAST/ZIP、官方GPU或提交。完整审核见`docs/reviews/axis_pi05/TRACK3_MATCHED_TERMINAL_L0_L5_AUDIT_20260904.md`。

## 51. Composed Reverse-Start M0–M5 执行收尾（2026-09-04）

本轮按唯一活动计划完成 M0–M5。M0 锁定上一轮归档 L0–L5、9 个 raw worker reports、contact ledger、三份公开 HDF5 及固定 demo identity；9/9 的 HDF5 `states[0] == init_state`、model/terminal hash、单帧 positive suffix 与旧失败 gate 分解复核通过。M1 新增纯逻辑 named state-composition、free joint 7-qpos/6-dof 映射、逐 index source bitmap、zero-qvel、neutral hold 和 contact-v3 contract；12 项 targeted tests 通过。

M2 唯一 task20/demo0 fresh CPU canary 通过；M3 固定九条 CPU matrix 为 `9/9`，三族各 `3/3`。每条均使用同 demo terminal scene qpos + 同 demo init robot/gripper qpos + 全零 qvel，reverse source 全程 true、goal 全程 false，raw terminal 与 composed state 的 public forward goal 均为 true，contact-v3 与严格 stability gates 均通过。`start_state_candidate` 只写入 report namespace，未写入 dataset/pending/accepted。

M4 仅在 GPU3 执行 task10/demo0、task20/demo0、task19/demo0 各一次 camera canary；GPU3 preflight 为 5 MiB、无 compute process，GPU0/1/2 未触碰。三条均通过，agentview/wrist 每路 `41` 帧、H.264、20 FPS、128×128，均无 `.partial` 残留。M5 full tests 为 `223 passed, 7 skipped, 23 warnings`，JSON parse、py_compile、diff check 和 M0 immutable hash revalidation 全部通过。

当前状态：`track3_composed_reverse_start_validated_pending_executor_review`。最终amended execution commit为`eef683b`；machine handoff中的`020d291`是amend前orphan。已 hard stop，等待新的审核计划；不得自行进入 operator executor、SE(3)、Teleop、批量生成、训练、norm、FAST/ZIP、官方 GPU、上传或提交。最终证据：`data/reports/axis_libero_30k/track3_composed_reverse_start_stabilization/summary.json`、`run_index.json`、`contact_ledger.json`、`M5_CLOSEOUT_REPORT_ZH.md`；root/OpenPI clean，`LIBERO-plus` 既有 dirty 状态未触碰。

## 52. M0–M5 独立审核与自动 Support Executor 授权（2026-09-04）

审核输入root `eef683b0db9629a3d917ea91ebeb46540a34fcb9`、nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`。CPU `9/9`、GPU3 camera `3/3`、6段H.264及hard stop的核心结果获批准；restore最大误差`2.22e-16`。只读复核还确认12个run的burn-in step1–20中demonstrator/unrelated contact均为0。

总体审核为`remediation_required`而非blocked：M4没有记录计划要求的seed/PID/GPU peak，contact required gate漏了burn-in；`start_state_candidate=1`只有t0 hash、没有可恢复vector；`round_trip...`字段实际是composed-vs-init差异，run index缺file hash，machine handoff还引用amend前orphan `020d291`。

这些都是agent可自动修复的工程闭包，不需要用户人工审核视频、Teleop或选择轨迹。旧计划已归档；唯一活动计划为`docs/AXIS_PI05_TRACK3_START_BUNDLE_AND_SUPPORT_EXECUTOR_CANARY_PLAN.md`：P0–P3补齐证据与9-state bundle，P4实现task10/demo0 HDF5-grounded SE(3) executor，P5最多一次generation并在成功后fresh replay `3/3`。成功也只产生最多1条`pending_autogen_v3`，accepted仍为0，不授权task20/task19、训练或官方GPU。完整审核：`docs/reviews/axis_pi05/TRACK3_COMPOSED_REVERSE_START_M0_M5_AUDIT_20260904.md`。
## 53. Start Bundle / Support Executor P0–P6 执行收尾（2026-09-04）

P0/P1 通过；P2 已物化 9 个 `float64` no-pickle start states，并以 fresh CPU/no-renderer 完成 restore `9/9`；task10/demo0 的固定首次 `-1→+1` gripper transition、相邻 state 与真实 `T_object^-1·T_eef` 几何提取通过。P3 只按固定顺序启动 task10/demo0 一条 GPU3 camera canary，41 帧与两路 H.264/20 FPS/ffprobe 证据已保存。

P3 worker 在写完整 checkpoint JSON 时填满父子 stdout 管道，阻塞于结果发布；该次 canary 已终止并记为 `worker_result_publication` 工具失败。没有重试 task10，也没有启动 task20/task19、P4 或 P5。P6 已完成 manifest、CSV、strict JSON/NPZ/video/py_compile/diff check；targeted `8 passed`，全量 `231 passed, 7 skipped, 23 warnings`。

最终状态：`track3_support_executor_canary_failed_worker_result_publication`。`validated_start_state=9`、`exact/pending/accepted=0/0/0`；没有 action、SE(3) executor、demo、Teleop、训练、norm、FAST/ZIP、官方 GPU、上传或提交。P6 已 hard stop，等待新审核计划；`LIBERO-plus` 既有 dirty 状态未触碰。完整证据位于`data/reports/axis_libero_30k/track3_start_bundle_and_support_executor_canary/`。

## 54. P0–P6 独立审核与 P3 I/O Recovery 计划（2026-09-05）

独立审核输入root `ad2d4ef7d6603e7b0e79a8519155de9cfdefbde2`、nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`。P0/P1与P2的9个`float64[79]`state获批准：NPZ no-pickle、shape/dtype/finite/hash为`9/9`，fresh CPU restore与truth/contact gates均为`9/9`，最大误差`2.22e-16`。P3只启动task10一次并在错误后停止；P4/P5没有执行，禁止项与`0/0/0`计数成立。

审核结论为`remediation_required`。旧artifact manifest的17项在本机全部存在且hash/bytes exact，但`start_states.npz`、task10 camera NPZ和两路MP4共4项受ignore且未入Git；P6 summary还错误声称排除自身。P3 stdout deadlock解释可信，但失败时的pre-fix dirty-worktree工具没有commit provenance，当前post-fix代码也尚未通过大输出/timeout验证，所以task10没有有效的P3 worker result。

旧计划已归档。唯一活动计划为`docs/AXIS_PI05_TRACK3_P3_IO_RECOVERY_AND_SUPPORT_EXECUTOR_CANARY_PLAN.md`：R0/R1完成17/17 Git closure、独立原子result JSON、16MiB stdout压力与300秒超时终止；形成clean commit后R2只允许task10 attempt 2，通过才补齐task20/task19。P3新证据`3/3`后才实现task10-only SE(3)，并最多运行一次generation与冻结trace replay `3/3`。全过程不需要用户Teleop、视频选择或人工轨迹审核，也不授权训练、官方GPU或提交。

审核报告：`docs/reviews/axis_pi05/TRACK3_START_BUNDLE_P0_P6_AUDIT_20260905.md`。

## 55. P3 I/O Recovery R0–R6 审核与 Overnight 计划（2026-09-05）

审核输入root `309ccafe9e6e2cddd0c9f16db111fc054478eb8b`、GPU阶段clean source `ad0a428c76f2c597109855f4d67f00df78184514`。R0旧17/17与新33/33 artifacts全部存在、hash/bytes exact且Git-tracked；R1 atomic result、16MiB stdout、TERM/KILL与异常fail-closed合同通过。R2只启动task10 attempt2一次，child的restore/source/goal/contact/camera全部passed，最大restore误差`2.22e-16`，两路41帧H.264闭合。

parent失败原因已精确定位：`ad0a428`中的`_run_gpu_canary()`调用`gpu_provenance_gate()`，但该source commit没有定义该函数；最终`309ccaf`加入定义并保留child结果。缺失的start snapshot、physical UUID与peak不能追溯补写，因此attempt2仍为failed_closed，但这不是机器人/state/reverse语义失败。R3–R5未执行，`exact/pending/accepted=0/0/0`，完整测试报告为`239 passed, 7 skipped, 23 warnings`。

旧计划已归档；唯一活动计划为`docs/AXIS_PI05_TRACK3_OVERNIGHT_AUTOGEN_LORA_AND_ZIP_PLAN.md`。O0/O1先纯函数化parent assembly并做NameError/watchdog回归，O2只给task10 attempt3一次机会；随后三族executor按独立lane最多两个冻结profile。严格generation与`3/3` replay产生数据后，使用本地20Hz LIBERO-plus按forward/reverse=`80/20`重算QUANTILES，从R6 step500继续两卡L2 LoRA，先两个100-step LR screen，再按reverse holdout与C20遗忘门禁续到500/1000，最后条件式fold/FAST/本地ZIP。

该计划预计上限8–12小时，允许无人值守和原子resume；共同基础设施错误全局停止，单family物理失败只关闭该lane。GPU0、官方GPU、dense full-FT、失败轨迹入训、自动上传/提交均不授权。审核报告：`docs/reviews/axis_pi05/TRACK3_P3_IO_RECOVERY_R0_R6_AUDIT_20260905.md`。

## 56. Overnight O0–O2 审核与 Provenance Jitter 续跑计划（2026-09-05）

审核输入root `8c0d296f1dfedb0c777040a733b6f9e352234111`。O0/O1通过；O2只运行task10/demo0 attempt3一次。child restore/truth/contact-v3/camera均passed，restore误差`2.22e-16`；两路相机均41帧、20 FPS H.264。parent完整记录seed3100、PID一致、GPU3 UUID、start/end snapshot、峰值459 MiB、113次采样、return code0与无timeout。

旧parent仍按合同failed_closed，因为`parent_poll_interval_max_ms=148.45660293940455`超过100ms。代码审计确认该字段计时包围同步`gpu_snapshot/nvidia-smi`调用，实际是query latency，不是heartbeat interval；因此它只能成为performance warning，不能否定完整provenance或task10物理结果。历史parent/O2/O10不得改写，下一轮用derived reanalysis升格。

同时发现旧O10 manifest 10项中CSV一项hash不稳定：manifest记录CRLF SHA `ad4f84c9...`，Git检出LF SHA `c39b4d30...`。下一轮固定CSV LF并在新namespace生成checkout-stable closure；旧manifest保留。

独立复跑全量测试为`244 passed, 7 skipped, 23 warnings`。O3–O9未执行，reverse demo仍为`0/0/0`，没有训练/FAST/ZIP/官方GPU/上传。root/OpenPI clean，GPU0外部进程保留，GPU1–3空闲。

旧计划已归档。唯一活动计划为`docs/AXIS_PI05_TRACK3_PROVENANCE_JITTER_RECOVERY_AND_OVERNIGHT_CONTINUATION_PLAN.md`：J0/J1修telemetry v2与LF manifest，J2只读释放attempt3且禁止attempt4；之后J3–J10连续执行task20/task19、family executor、最多12条generation/36次replay、20Hz 80/20 mixture、R6 L2 LoRA 100/500/1000、reverse/C20评估与条件式本地FAST ZIP，J11 hard stop。用户无需人工中途审核；自动上传/比赛提交不授权。

审核报告：`docs/reviews/axis_pi05/TRACK3_OVERNIGHT_O0_O2_AUDIT_20260905.md`。

## 57. Provenance Jitter J0–J5 执行停止点（2026-09-05）

执行HEAD为`7ae488b4475169ac1cc3b586158454639e6a2be9`，J5 clean source为`3a5641956770fa74faaed20f12832489d7f257b7`。J0/J1/J2生成input lock、telemetry v2和task10 attempt3 derived release；J3只在GPU3依次运行task20/demo0、task19/demo0各一次，均通过restore/truth/contact-v3/camera、PID/UUID、return/timeout和两路41帧20 FPS H.264。J4报告释放三族executor。

J5只启动task10/demo0 nominal一次。child真实产生180帧双相机和finite trajectory，但`approach_grasp`没有收敛，最后translation/rotation error为`0.015306 m / 3.126052 rad`，180帧均action saturated，goal始终false。parent随后因ffmpeg无法识别`agentview.mp4.partial`而camera失败；child return code 1又使telemetry记录`process_returncode`错误，J5 summary正确写为infrastructure hard stop。J11错误改写成`no_family_release`。

J6–J10均为`not_evaluated`，`exact/pending/accepted=0/0/0`。没有training、norm、Teleop、demo、FAST、ZIP、官方GPU、上传或提交。全量测试为`250 passed, 7 skipped, 23 warnings`；GPU3已释放，GPU0外部PID保持，root/OpenPI clean，`LIBERO-plus`既有dirty未改。

## 58. J0–J5 独立审核与最小恢复计划（2026-09-05）

审核唯一结论为`remediation_required`。J0 hashes、J2/J3核心release、J5单次执行/hard stop、未执行边界与资源边界获批准。J4 intended executor不获批准：Robosuite返回`xyzw`，工具按`wxyz`做compose/log；正确HDF5 EEF重构误差为`2.276e-7 rad`，现helper偏差`2.21036 rad`。旧nominal的历史`failed_gate`保留，但对修正后executor的profile结论只能是`not_evaluated`。

审核还确认：J4未证明same-demo init target、object-follow、task19 joint direction或实际contact/stall/unrelated gates；J1 launcher没有实现宣称的dirty/commit/tool校验和周期heartbeat；旧J3/J11 manifests分别有`5/21`与`4/37`项因cursor/log/PID finalization顺序而漂移，J3 core entries为`14/14`、J11 core execution entries为`25/25` exact。

旧计划已归档。新唯一活动计划`docs/AXIS_PI05_TRACK3_J5_EXECUTOR_CONTRACT_AND_ATOMIC_VIDEO_RECOVERY_PLAN.md`执行V0–V5：CPU统一`xyzw`、support HDF5与hard-gate合同、原子MP4、进程分类、J11传播、supervisor和manifest；离线重编码旧180帧并写derived errata；形成clean commit后仅GPU3运行一次corrected task10/demo0 nominal canary，然后hard stop。task20/task19 manipulation、conservative、J6 generation/replay、训练与ZIP均未授权。审核报告：`docs/reviews/axis_pi05/TRACK3_JITTER_J0_J5_AUDIT_20260905.md`。

## 59. J5 Executor Recovery V0–V5 执行停止点（2026-09-05）

执行HEAD为`11483623b7710430a197231b8ed0662c4813c12d`，V4 clean release commit为`1b704405fa918b5237a50a1219d4a26b61c293de`。V0锁定18项immutable input；V1完成explicit `xyzw`单点pose重构；V2离线原子编码旧nominal两路180帧H.264；V3形成clean commit。

V4只在GPU3执行task10/demo0 nominal、seed3111一次。worker PID `1450749`正常发布`failed_gate`并return 0；parent正确分类`family_failure`。GPU3 UUID、159次telemetry、459 MiB峰值、无timeout及两路150帧H.264均通过。轨迹停在`approach_grasp`，没有进入后续抓取、搬运或放置阶段。

V5 hard stop；最终manifest为`19/19` Git-tracked/hash exact。全量测试`256 passed, 7 skipped, 23 warnings`；GPU1–3释放，GPU0外部PID保留，root/OpenPI clean，`LIBERO-plus`既有dirty未改。没有retry、conservative、其他family、demo/generation、Teleop、训练/norm、FAST/ZIP、上传或提交；`exact/pending/accepted=0/0/0`。

## 60. V0–V5 独立审核与 Release-Anchor/OSC-Scale 计划（2026-09-05）

审核唯一结论为`remediation_required`。V0/V2/V3核心、V4单次运行的observed `family_failure`分类、V5 hard stop、manifest和禁止项获批准；但“已执行正确nominal profile”不获批准。

V1的`xyzw`修正确实有效，但action builder把`rotation_clip=0.12 rad`直接归一到`±1`，本地OSC runtime则将`±1`映射到`±0.5 rad`，实际旋转增益约`4.1667×`。V1只使用forward close transition index32；独立exact-model restore确认release transition为76，release前index75有`3/1/0` demonstrator/receiver/unrelated contacts，而index32→75 object-relative pose漂移为`0.0387545 m / 0.600317 rad`。

V4在approach 150帧后translation error仍为`0.104612 m`，后续contact/place阶段没有执行，所以contact completion与same-demo destination必须修正为`not_evaluated`。`launch_identity_gate`/`heartbeat_due`也没有进入真实V4 launcher，parent缺root/tool identity。

旧计划已归档。唯一活动计划`docs/AXIS_PI05_TRACK3_SUPPORT_RELEASE_ANCHOR_AND_OSC_SCALE_RECOVERY_PLAN.md`执行W0–W6：先CPU闭合release-side HDF5 anchor、真实OSC scale、stage-dependent三态和supervisor；clean commit后仅一次CPU/no-renderer完整task10 canary，通过才允许一次GPU3 camera canary，然后hard stop。conservative、task20/task19、demo/generation、训练和ZIP均不授权。审核报告：`docs/reviews/axis_pi05/TRACK3_J5_EXECUTOR_RECOVERY_V0_V5_AUDIT_20260905.md`。

## 61. Support Release-Anchor W0–W6 执行收尾（2026-09-05）

最终execution commit为`2997c6f5d8c5b05345d74e0e0c4b96698def727e`，W4 clean source为`1ec308a26e2fad306b221197ab9041d1b7633346`。W0复核15项immutable hash、旧V5 `19/19`、root/OpenPI与GPU边界；W1锁定task10/demo0 close/release=`32/76`、release-side state75 contact=`3/1/0`；W2确认controller output=`0.05 m / 0.5 rad`和`0.12 rad→0.24`normalized mapping；W3形成clean release。

W4仅运行一次CPU/no-renderer、seed3121。worker正常发布`failed_gate`且return 0，parent分类`family_failure`，无timeout。trajectory为252步：approach41、descend12、close4、lift15、transport180；transport耗尽预算，末帧误差`0.0191271 m / 0.442809 rad`。source为`253/253 true`，goal为`253/253 false`，contact completion failed。W5=`not_evaluated`，没有启动GPU3。

W6最终manifest为17项，全部Git-tracked且bytes/SHA exact；14个final JSON严格duplicate-key parse通过，NPZ/NPY no-pickle finite、CSV LF、无partial。执行记录的全量测试为`257 passed, 7 skipped, 23 warnings`。GPU0外部PID保留，GPU1–3释放；没有generation/replay、Teleop、训练/norm、FAST/ZIP、官方GPU、上传或提交；计数`exact/pending/accepted=0/0/0`。已hard stop等待审核。

## 62. W0–W6 独立审核与 Object-Frame/Bounded Autogen 计划（2026-09-05）

审核唯一结论为`remediation_required`。W0/W1/OSC scale、W4真实family failure、W5未执行、W6 artifact closure与资源/禁止项获批准；但intended object-frame nominal profile仍为`not_evaluated`。

W4在grasp侧正确使用`T_world^object·T_object^eef`，到transport/place却把state0 bowl pose直接当EEF target。按冻结anchor复算，current place target与正确组合相差`0.0817122 m / 1.816425 rad`。同时source全程true且contact completion failed，执行器仍越过acquisition barrier进入transport；旧transport timeout保留，但不能当作唯一根因。progress-window stall未实现，运行中全局cursor未置active，W6验证又早于最终closeout/manifest写入。

旧计划已归档。唯一活动计划为`docs/AXIS_PI05_TRACK3_OBJECT_FRAME_EXECUTOR_AND_BOUNDED_AUTOGEN_LORA_PLAN.md`：X0–X3修复object→EEF、phase prerequisite/contact、stall、active cursor和finalizer；X4–X7按family隔离执行support/task20/task19 bounded profiles；X8最多12条generation/36次replay；严格accepted数据通过后，X9–X12才进入20Hz `80/20` mixture/fresh QUANTILES、R6 L2 LoRA 100/500/1000、reverse/C20与本地FAST ZIP。任何分支X13 hard stop，不使用GPU0/官方GPU，不Teleop、不自动上传或提交。审核报告：`docs/reviews/axis_pi05/TRACK3_SUPPORT_RELEASE_ANCHOR_W0_W6_AUDIT_20260905.md`。

## 63. Object-Frame X0–X13 执行收尾（2026-09-05）

最终execution commit为`64a0d31c3b2864562887044bad7ef8b522a213db`，X4 clean source为`ce86e724dddf300406465d6986577199d39cc19b`。X0–X3写出object→EEF、tri-state、progress fixture、supervisor identity与derived W4 replay报告，并在进入X4前形成clean commit。

X4只启动task10/demo0、seed3121、`object_frame_nominal`的一个CPU/no-renderer worker。identity gate通过；worker在`ControlEnv`导入/构造和任何`env.step`之前调用不存在的`track3_reverse_init_stabilization._configure_public_env`，发布`worker_exception`并return 1。parent正确分类`infrastructure`并全局停止；conservative、GPU3、task20/task19、generation/replay、mixture/norm、LoRA、评估与ZIP均未运行，X5–X12=`not_evaluated`。

X13 manifest为13项，全部Git-tracked且bytes/SHA exact；14个JSON strict、CSV LF、无partial，执行记录与独立复跑均为`260 passed, 7 skipped, 23 warnings`。`exact/pending/accepted=0/0/0`。GPU0外部PID保留，GPU1–3释放；root/OpenPI clean，LIBERO-plus既有dirty未触碰。已hard stop，不自动恢复。

## 64. X0–X13 独立审核与 Runtime-Preflight Continuation（2026-09-05）

审核唯一结论为`remediation_required`。X4确定性infrastructure分类、0 physics step、X5–X12未执行、X13 artifact closure、资源边界和禁止项成立；object-frame数学与三态fixture可作为后续输入保留。

但X0没有记录计划要求的W4 trajectory、state bundle、三份public HDF5、R6 checkpoint/norm完整immutable inventory。X2的worker只记录progress window而不在failure时产生stall/早停；又在close后、lift前错误要求source已释放，和冻结的close→lift contact、lift→transport source-release阶段合同相反。运行期symbol smoke与长run周期heartbeat也缺失。X13未记录其承诺的manifest自身/final-validation checkout-stable hash。

旧X计划已归档。唯一活动计划为`docs/AXIS_PI05_TRACK3_RUNTIME_PREFLIGHT_AND_BOUNDED_CONTINUATION_PLAN.md`：Y0–Y2先完成全量input lock、runtime symbol/阶段/stall/heartbeat/finalizer补救与clean release；通过后Y3才以新identity执行一次同task/demo/seed nominal物理canary。后续Y4–Y12仍是family隔离、最多12条generation/36次replay、20Hz `80/20` mixture/fresh QUANTILES、R6 L2 LoRA 100/500/1000、独立复验和条件式本地FAST ZIP；Y13在任何分支hard stop。生命周期更新后的最终全量测试为`261 passed, 7 skipped, 23 warnings`。审核报告：`docs/reviews/axis_pi05/TRACK3_OBJECT_FRAME_X0_X13_AUDIT_20260905.md`。

## 65. Runtime Preflight Y0–Y2 执行收尾（2026-09-06）

Y0首次运行因进程扫描误计当前launcher且GPU0历史外部PID已退出而保留为失败证据；修正后Y0复核通过。新的锁定文件记录X/W历史证据、三份公开HDF5、start-state bundle、R6 raw/folded checkpoint/norm的逐文件bytes/SHA，并记录复核时GPU0无compute app、未被本任务触碰，GPU1–3空闲。代码clean release commit为`b16211bc7e48ec96ae8b89b9edf88462e73d3078`，最终交接闭包提交为`d643e105830ac93622169a53198e2f3373f7b658`。

Y1通过canonical `track3_composed_reverse_start_stabilization._configure_public_env` fresh subprocess smoke；阶段合同固定为close→lift连续demonstrator contact、lift后object-follow/source-release、transport、place、release；progress window在首个失败完整窗口产生`stage_stall:<stage>`并停止；atomic heartbeat、真实launcher identity与`failed_gate` return 0 / `worker_exception` return 1分类均通过。Y3 worker/supervisor已实现但未执行。

Y2 finalizer写入inactive cursor、LF phase CSV、manifest与checkout-stable manifest/final-validation SHA；strict duplicate-key JSON、NPZ no-pickle/finite、CSV LF、py_compile、定向测试`10 passed`、全量测试`271 passed, 7 skipped, 23 warnings`及`git diff --check`均通过。当前`exact/pending/accepted=0/0/0`；Y3–Y13为`not_evaluated`，没有启动physics、GPU、generation、replay、training、norm、FAST或ZIP。证据：`data/reports/axis_libero_30k/track3_runtime_preflight_and_bounded_continuation/`。
