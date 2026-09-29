# AXIS Pi0.5 T3 Post-H300 Step100 Salvage and R6-Style Recovery Plan

> 创建时间：2026-09-15T19:20:46+09:00  
> 状态：`ready_for_S100_salvage_implementation_and_preflight`  
> 当前阶段：`S0_closeout_and_step100_comparator`  
> 当前 release：`none`  
> 自动上传/提交：`false`

## 1. 目标与总路线

当前 H300 step300 已被行为门拒绝。下一路线分两层：

1. deadline salvage：只评测已经存在的 H300 step100；不重新训练。若严格 paired gate 通过且剩余时间足够，再制作 step100 release；
2. 赛后 recovery：复制 R6 的“多臂短筛选 -> fresh formal run -> 多 checkpoint paired gate”方法，但先修正 T3 数据覆盖、采样和 continuation LR。

关键路径：

`S0 comparator/tests -> S1 step100 fresh worker -> S2 paired gate -> conditional S3 release | fail-stop -> R0-R4 long-term recovery`

## 2. 冻结事实与禁止项

只读冻结：

- closed step300 evidence：`data/reports/axis_libero_30k/t3_fast400_r6_h300_paired_eval_20260915_attempt_0002`；
- closed step300 output：`outputs/axis_pi05/t3_fast400_r6_h300_paired_eval_20260915_attempt_0002`；
- R6 baseline role：上述 evidence/output 的 `r6/`；
- paired summary SHA：`e74a112aae506a368ff67ed6d7b860a84d1651872a93856f96ad21370a6ec1b4`；
- release decision SHA：`f7d6cc4fb8c72b17a0db8efb27774a749b2448863235c40fc49cdf7a72ea166b`；
- paired adapter SHA：`9c736c0ac1e3809df0ee35710781ffe8011929220794c58a75cf756e5b8a6c97`；
- Fast400 norm SHA：`7e4e24dacb24d6333d32cb982d301761c2f00e8f5daca418109bea226268f333`；
- step100 params native SHA：`870022fc61f4e00d283adcc10371ac4ecf53e15c93c871b415a126bdf0bce314`；
- step100 params Q0 SHA：`0a8ee72f7bb58490ecade86ca2dad351273ba0938116ee25d5c300a34f8657ad`；
- step100 metadata SHA：`d6b64050fe3c82dc7e78236cc00eb6bf5409dfaffab1043d3e242fa7c04b5bde`。

禁止：改写 attempt_0002；重跑其 R6/step300；从 step300 打 ZIP；启动新训练、dense/30k、step300 continuation；自动上传或提交。

## 3. S0：step100 comparator 与 fail-closed preflight

允许新增：

- `tools/axis_pi05/run_track3_fast400_h300_step100_salvage.py`；
- `tests/test_track3_fast400_h300_step100_salvage.py`。

不修改 closed paired evidence。comparator 必须：

1. 只读加载 attempt_0002 的 R6 24-case evidence/action traces；
2. 要求 R6 lock completed、adapter SHA=`9c736...c97`、Fast400 norm、case fingerprints/seeds/protocol exact；
3. 加载全新 step100 role evidence，并验证 checkpoint 三个冻结 hash；
4. 对两个 role 的 48 条 action trace重新 no-pickle/shape/finite/nonzero验证；
5. 写入新 salvage namespace 的 paired summary 与 release decision，不得修改原 summary；
6. 保留 step100 原身份 `diagnostic_only=true`。只有新 paired gate 通过时，release decision 才可显式写 `step100_salvage_promoted_for_release=true`。

至少测试 source/hash mismatch、case/seed mismatch、partial、nonfinite、all-zero、旧 root write refusal、重复 merge refusal、gate pass/fail。

新 roots：

- evidence：`data/reports/axis_libero_30k/t3_fast400_h300_step100_salvage_20260915_attempt_0001`；
- output：`outputs/axis_pi05/t3_fast400_h300_step100_salvage_20260915_attempt_0001`。

先运行 focused tests、py_compile、strict JSON、`git diff --check`，再执行 step100/GPU0 只读 preflight。任何失败均停止。

## 4. S1：一个 fresh step100 worker

使用现有 paired-eval runner，GPU0 fresh 运行 `role=step100`，指定上述全新 roots；禁止 `--resume`。GPU1–3 保持空闲，避免为了约十几分钟收益引入 shard/comparator 复杂度。

必须完成：

- exact `24/24`；
- checkpoint=`H300/100`，training_steps=`100`；
- active Fast400 norm health/log SHA exact；
- execute8、20 Hz、chunk10、max300；
- no infrastructure/model/nonfinite error；
- cursor completed、heartbeat stopped、controlled SIGTERM、无 orphan/partial。

预估 S0+S1+S2 为 `25–45` 分钟。

## 5. S2：复用 R6 的 step100 paired gate

comparator 只读复用 attempt_0002 R6，生成 `R6 vs H300-step100`：

- exact paired coverage `24/24`；
- reverse delta `>= +1/4`；
- C20 delta `>= -2/20`；
- errors=`0`；
- fingerprints/seeds/protocol exact；
- trace finite，两个 role 均非 systemwide all-zero。

任一失败：写 `step100_behavior_gate_failed_no_zip` 并 hard stop。不得因 step100 loss 更低或单条视频看似更好而放宽门。

## 6. S3：仅 step100 gate 通过后的 release

只有同时满足以下条件才进入：

- S2 gate passed；
- 新 decision 显式 promotion；
- 距离比赛截止至少 `75` 分钟。

随后使用专用 step100 release root完成 raw-LoRA/folded、JAX/PyTorch/processor、FAST parity、ZIP CRC、`unzip -t`、static validation、SHA/bytes/provenance。ZIP 和 manifest必须写 `training_steps=100`，不得冒充 step300或30k。仍不自动上传/提交。

若当前剩余时间不足约 `120` 分钟，不承诺完成“评测+发布”；可以只完成 S2 作为研究证据。若不足 `75` 分钟，不启动 S3。

## 7. R0–R4：赛后 R6-style T3 recovery

### R0：失败轨迹归因

对四条 reverse 的 R6/step100/step300 action、视频和阶段进展做只读对齐；单独分析 C20-task13 的 `baseline success -> step300 failure`。先区分未接近、抓取失败、抓取后未释放、collision/timeout，不用 success-only 信号盲调训练。

### R1：重做数据合同

当前只有 8 条 reverse episode、4 tasks，却占 20% source mass；单条 reverse sampling probability 是典型 forward 的约 `238–1190x`。下一数据版本要求：

- 每个目标 task 至少 3 条独立 accepted demonstration；
- 至少 2 个独立 recording lineage；
- train/dev 按 lineage 隔离；
- 先比较 `95/5`、`90/10`，不默认复用 `80/20`；
- norm、sampling probability、每条 effective exposure 全部落盘；
- 旧 Fast400 保持 immutable。

### R2：四臂 25-step screen

四卡各跑一条单卡、global-batch-32 screen，全部 fresh optimizer：

1. R6 params + L2，peak LR `3e-7`；
2. R6 params + L1 action-expert-only，peak LR `3e-7`；
3. released Axis + zero-delta L2，peak LR `1e-6`；
4. R6 params + small dual LoRA，action rank32/PaliGemma rank16，peak LR `1e-7`。

这些数值是待验证 screen arms，不是预先认定 winner。每步记录 pre/post clip；若某 arm连续高 clip、nonfinite、scope drift 或 restore失败，淘汰。最多两个 arm进入100-step。

### R3：100/300/500 checkpoint selection

不要默认最后一步最好。候选必须在100、300、500分别与同一冻结 R6 做 paired behavior gate；只要 reverse未增益或 C20跌超过2就停止该 arm。formal run必须从冻结 init重新 fresh 启动、effective global batch64，不能从单卡 screen checkpoint续跑。

### R4：何时考虑官方30k

只有满足以下条件才值得上比赛 GPU 做更长训练：

- 至少一个 500-step candidate 两个独立 T3 dev panel均通过；
- reverse improvement不只来自训练 lineage；
- C20 retention成立；
- clip、norm、restore、fold/FAST parity稳定；
- 数据规模与多样性足够支持长训练。

30,000 steps是后续 canonical scale，不是当前默认动作；先用500/1k验证方向，再决定是否投入30k。

## 8. 当前权限

- S0 comparator/tests：authorized；
- step100 preflight + 单 fresh worker：authorized；
- S2 merge/gate：authorized；
- step100 release：仅 S2 pass 且剩余时间 >=75分钟时 authorized；
- R0只读失败分析：authorized；
- R1数据准备：需后续明确执行；
- R2–R4训练：当前不授权；
- upload/submission：不授权。
