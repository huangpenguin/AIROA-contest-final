# AXIS Pi0.5 T3 Fast400 H300 训后评测与候选发布交接

> 创建时间：2026-09-15（Asia/Tokyo）
> 状态：`ready_for_next_agent`
> 唯一下一阶段：`E0_implement_fast400_h300_paired_eval_adapter`
> 本文件是针对已完成本地 H300 的窄范围执行交接；全局 handoff 仍滞后在 Fast400 采集阶段，不能据此重跑采集或训练。

## 1. 唯一目标

对 `t3_fast400_r6_h300_4gpu_r1` 完成可归属的训后闭环：

1. 用 Fast400 新 norm 在同一冻结 24-case 面板上配对评测 R6 params-only baseline 与 H300 step300；
2. 严格合并 reverse-dev 4 + C20 20；
3. 只有 `reverse delta >= +1/4` 且 `C20 delta >= -2/20`、覆盖完整且无基础设施错误时，才 fold step300、生成 FAST ZIP 并完成 release audit；
4. 不上传、不自动比赛提交，最终 ZIP 交给用户。

step100 仅为诊断对照，不是主要候选，不得延迟 step300 必需评测。

## 2. 已完成且不得重跑的训练

- run ID：`t3_fast400_r6_h300_4gpu_r1`
- run dir：`outputs/axis_pi05/track3_fast400_h300/t3_fast400_r6_h300_4gpu_r1`
- evidence：`data/reports/axis_libero_30k/t3_fast400_r6_h300_4gpu_r1_local4`
- 完成时间：`2026-09-15 00:28:19 JST`
- summary SHA-256：`d9f81e6b1fde66a26e4d659647674785c0815d2cfd489f745bfc9f0df2912bed`
- metrics：300/300 连续 optimizer steps，SHA-256 `f36a4cfacb5eba7112dd0106b74be78a0d3fca5d795e9b608b85b2c7385f648a`
- batch accounting：每步 `micro_steps=16`、`examples_seen=64`
- checkpoints：step1、step100、step300 均有 `_CHECKPOINT_METADATA`、`params/`、`train_state/`
- fresh restore：step300、4 devices、`restored_step=300`
- 当前行为结果：`not_evaluated`
- 当前 GPU：0–3 全部空闲；当前无 trainer/evaluator/server 进程

冻结训练输入：

- R6 params q0 SHA-256：`d9b7e0d15781060fccfd61dc1890ad98993f0224e4aebba1dd6658d7c011931f`
- Fast400 manifest：`ba940bbcaff15395803c5a942f004e491bb5a6fce69be8b9b55bb4f759fac97e`
- Fast400 norm：`7e4e24dacb24d6333d32cb982d301761c2f00e8f5daca418109bea226268f333`
- Fast400 sampling：`aed3380f646bf4c2635090b2b6e994b60a492ee064cda6067cbdc4071e6c90a1`
- Fast400 source roots：`2082c27207bf3b73a2139803a7923a21e8837b02b97c01961469fc1dc7115bdc`
- Fast400 reverse tree：`966303c7d128ffd169cb7a10f5fbb017bda24a9181f64f05bf1eeb39bfd66e50`

## 3. 现有工具不可原样运行

必须先做一个窄范围 adapter；不要直接调用旧 runner：

- `run_track3_q5_step10_frozen_eval.py` 硬编码旧 `four_gold_80_20_v1` norm、旧 checkpoint root，并只接受 step0/10；
- `run_track3_n5_official_paired_eval.py` 硬编码旧 norm/namespace、candidate step100 语义和旧 GPU0 禁令；
- `export_candidate_fast_zip.py` 的既有 candidate registry 面向旧 canary100，不含 Fast400 H300 身份。

允许复用上述模块的 case normalization、Execute8 policy、HTTP server、rollout row、resume case-file 和 merge 逻辑，但新 runner 必须拥有新的常量、身份和输出 namespace，不得 monkey-patch 后写回旧 Q7/N5 目录。

建议新增：

- runner：`tools/axis_pi05/run_track3_fast400_h300_paired_eval.py`
- focused test：`tests/test_track3_fast400_h300_paired_eval.py`
- evidence root：`data/reports/axis_libero_30k/t3_fast400_r6_h300_paired_eval_20260915`
- rollout root：`outputs/axis_pi05/t3_fast400_r6_h300_paired_eval_20260915`

## 4. E0：专用 adapter 合同

冻结评测输入：

- evaluation freeze：`data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/evaluation_freeze.json`
- freeze SHA-256：`2c8d2eca6b5138ddc59d9d00837c95e774f4a2e14687cd656662ee6e9dd77b77`
- reverse manifest：`data/eval_manifests/axis_pi05_eval_v1/public_t3_holdout.jsonl`
- reverse SHA-256：`64a1124ef17a866e259ba7e8bce96351cd183dff1ebf76c1afbd5694dfe0ec95`
- C20 manifest：`data/reports/axis_libero_30k/phase1/rollout/C20.jsonl`
- C20 SHA-256：`1dad183301e255e4287348470d962b9016420c2365f0ac8a8c2eef7ed0052231`
- norm：`data/manifests_track3/fast400_gold_80_20_20260913/norm_stats.json`
- norm SHA-256：`7e4e24dacb24d6333d32cb982d301761c2f00e8f5daca418109bea226268f333`

必须比较：

- baseline：`outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1/500/params`
- candidate：`outputs/axis_pi05/track3_fast400_h300/t3_fast400_r6_h300_4gpu_r1/300`
- diagnostic：同 run 的 step100，可选

baseline 与 candidate 必须都使用 Fast400 新 norm。旧 four-gold norm 下的 R6 `0/4 + 14/20` 只能作历史背景，不能直接作为本次 matched gate 的 baseline。

共同协议：`axis_pi05_track3_q5`、20 Hz、chunk10、execute8、max300、相同 case IDs/init states/env seeds/policy seeds、QUANTILES。所有 JSON/JSONL 严格拒绝 duplicate keys。

adapter 至少提供：

- `preflight`：验证 hashes、checkpoint 完整性、无 partial、GPU/进程边界；只读；
- `worker --role r6|step300|step100 --gpu N`：每个 role 使用独立 port、cursor、heartbeat、case files 和视频目录，可 resume；
- `merge`：要求 baseline/step300 各 24 个完全相同 case IDs，生成 paired summary 与 release decision；
- `status`：只读汇总 worker 进度和 PID liveness。

focused tests 要覆盖：错误 norm/hash fail closed、step300 身份、24-case exact coverage、case-ID mismatch 拒绝、gate 边界、旧 namespace 拒绝、GPU0 本轮授权生效、resume 不重跑已完成 case。

## 5. E1：并行评测顺序

用户已明确授权 GPU0，本轮四卡均可用。建议：

1. GPU0：R6 baseline + Fast400 norm，24 cases；
2. GPU1：H300 step300 + Fast400 norm，24 cases；
3. GPU2：可选 H300 step100 诊断，24 cases；只有 GPU0/GPU1 健康启动后再开；
4. GPU3：保留给失败重试、fold/parity/FAST；必要时可接 step100。

不要让多个 worker 共享 evidence/output/cursor/port。MuJoCo/EGL 使用各自物理 GPU ID。训练已结束，评测可立即开始；全链本地离线，不依赖 VPN/HF。

## 6. E2：唯一候选门

step300 required pairs：24/24；reverse=4；C20=20。

通过条件全部为 true：

- 无 model/server/nonfinite/asset/init-state 基础设施错误；
- case fingerprints 和 seeds 完全一致；
- reverse candidate success - matched baseline success `>= 1`；
- C20 candidate success - matched baseline success `>= -2`；
- action traces finite，非系统性全零，execute8 协议成立。

若未通过：写 `release_decision.status=behavior_gate_failed_no_zip`，保留 checkpoint/视频/action traces；不 fold、不 ZIP、不把 training loss 当收益证据。

## 7. E3：仅 gate 通过后 fold / FAST / ZIP

1. 本地 fold：

   `.venv_axis_pi05/bin/python -m tools.axis_pi05.convert_for_lerobot --checkpoint outputs/axis_pi05/track3_fast400_h300/t3_fast400_r6_h300_4gpu_r1/300 --output outputs/axis_pi05/track3_fast400_h300/t3_fast400_r6_h300_4gpu_r1_folded`

2. 用 `prepare_lerobot_assets.py` 写入 Fast400 norm processors；legacy template 可用已验证 R6 folded 目录，但 identity/provenance 必须写 H300 step300 与新 norm。
3. 跑 raw-LoRA ↔ folded JAX、folded JAX ↔ PyTorch/processor parity；JAX checkpoint 必须显式指向本 run step300，不能回落到默认 R6。
4. FAST export、FAST-vs-folded parity、server latency/smoke。
5. 生成 T3 ZIP；运行完整机械发布审核：ZIP entry/CRC、`unzip -t`、`validate_submission.py --static --json`、SHA-256、bytes、source model/norm/checkpoint identity。
6. manifest 必须写：candidate=`t3_fast400_r6_h300_4gpu_r1`、training_steps=300、topology=`L2_action_lora_heads`、normalization=`QUANTILES`、official_result=false、auto_submit=false。

建议产物：

- folded：`outputs/axis_pi05/track3_fast400_h300/t3_fast400_r6_h300_4gpu_r1_folded`
- release evidence：`data/reports/axis_libero_30k/t3_fast400_r6_h300_release_20260915`
- ZIP：`artifacts/submission_axis_t3_fast400_r6_h300_r1_fast_<timestamp>.zip`

ZIP/提交边界属于 release boundary，执行正式机械审核；不自动上传或提交。

## 8. 禁止事项与终止条件

- 不再启动 dense、30k、第二条 H300 或 train-state continuation；
- 不重写 H300 training summary 中的历史事实；评测写新 evidence；
- 不覆盖旧 Q7/N5、R6、T2 H10K ZIP；
- 不因旧 handoff 滞后而回到 FT0/采集；
- 不把 step100 诊断拖成主路径；
- 未通过行为 gate 时不制作“可提交 winner” ZIP；
- 不自动上传、不自动比赛提交。

交付完成定义：要么得到 `behavior_gate_failed_no_zip` 的完整 24-pair 证据；要么得到 gate-passed、机械审核通过、SHA/bytes 已锁定且仍未提交的 T3 ZIP。
