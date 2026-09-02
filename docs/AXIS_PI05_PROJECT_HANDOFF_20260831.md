# AXIS Pi0.5 项目交接快照（2026-08-31）

> 用途：新 Codex section 从这里接手；不要从旧聊天重新猜测状态。
> 最新已审核执行 HEAD：`88796b2`
> interstep L0/L1 execution HEAD：`88796b2`
> branch：`codex/axis-pi05-audit-remediation`（当前无 upstream）
> nested OpenPI：`b58aff0`（interstep lifetime implementation）
> 当前总状态：`track3_reverse_lora_plan_ready`
> 唯一活动计划：`docs/AXIS_PI05_TRACK3_REVERSE_LORA_POSITIVE_SCORE_PLAN.md`
> 计划索引：`docs/AXIS_PI05_PLAN_INDEX.md`
> closure execution parent：`be62f678c0927ebd6a0b2d29a63450c2edc74177`
> closure input：`78109647e95d79690511c922845141ff492a395f`
> closure execution commit：`327316f2ffdca0df3375748bcdd65b5ce263f8f8`

## 1. 一句话结论

canonical LIBERO、R6 LoRA500-step、folded PyTorch与FAST-only ZIP均已完成；用户报告正式Track1=`0.47`、Track2=`0.33`、Track3=`0`。Dense interstep的L0/L1已完成，但L2因GPU0外部YOLO阻塞；成果已冻结。当前用GPU1+2/GPU3改道Track3：只依据官方公开的“现有LIBERO逆向操作”高层规则，先做真实20Hz reverse pilot，再从R6分叉保守LoRA，不使用泄漏的隐藏任务细节。

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
2. 阅读并只执行 `docs/AXIS_PI05_DENSE_FULLFT_DONATED_ACCUMULATOR_RECOVERY_PLAN.md`。
3. 将Track1=`0.47`、Track2=`0.33`及ZIP `95415a…c5f5a`锁为当前官方baseline；Track2包身份在无上传回执/hash时保持conditional provenance。
4. D0先闭合summary/handoff的stale字段，并区分compute complete、merge failure和下一未尝试compute；raw logs/checkpoints immutable。
5. D1只把严格左结合FP32 tree add改为donated/aliased旧accumulator，不改EMA/global64、RNG、loss或data/norm。
6. D2先做两份真实microbatch merge-only canary；通过后D3/D4最多本地step1/step10。不得启动官方GPU、Blackwell、C20、100/1K、长训、FAST/ZIP或提交。

当前路径是：R6 FAST ZIP Track1=`0.47`/Track2=`0.33`（冻结）→ sharded-init已通过 → donated merge D2 → 本地step1/10 → 人工审核显存/restore/吞吐 → 100-step+C20（20Hz主协议）→ 标准LIBERO同run里程碑长训 → paired选模 → 复用R6 FAST部署链 → 目标正式分数`>0.47`。

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
