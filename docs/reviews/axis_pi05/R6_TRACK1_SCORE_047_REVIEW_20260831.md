# R6 Track 1 官方分数 `0.47` 审核记录

> 审核日期：2026-08-31（Asia/Tokyo）
> 输入 commit：`97f3a35`
> 审核结论：`approved_as_user_reported_official_baseline`

## 1. 结果

用户报告冻结 R6 FAST ZIP 的 Track 1 正式得分为：

```text
track           Track 1
assignment      1007
score           0.47
artifact        artifacts/submission_axis_r6_step500_fast_positive_score_20260831.zip
sha256          95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a
candidate       r6_dual_l2_fresh500_r1
training        L2 action-expert LoRA + action projections/time MLP, 500 optimizer steps
deployment      FAST-only, chunk10/execute8, EMA0, RTC off
```

当前仓库仍没有 submission ID、计分时间、截图或官方完整 log。分数作为用户提供的正式结果进入项目记忆，provenance 标记为 `user_reported`；不得补造 episode数、server latency或隐藏任务详情。

## 2. 能得出的结论

1. R6 FAST-only 包在真实 scorer 中完成了有效评测；先前 SLOW/remap/vision-key 导致的 `0/6` 灾难路径已被避开。
2. `0.47` 证明当前 Axis→LoRA→fold→FAST→server 链可以作为后续所有 checkpoint 的冻结部署基线。
3. 用户报告它显著优于此前纯 Pi0.5。若历史对照确为 `0.395`，则绝对提升 `0.075`、相对约 `19%`；在没有对应 submission记录前，该计算只作历史对照说明。
4. 本地 R6 vs Axis step-0 B80 仅为 `49/80` vs `48/80`、CI跨0，所以 `0.47` 不能单独把增益严格归因于500-step LoRA；它验证的是整条候选链的正式表现。

## 3. 研究决策

- 冻结 R6 ZIP、FAST hash、server参数和导出门禁；不再为该包运行本地Docker E2E或官方Blackwell pre-check。
- 不从 R6 step-500 raw LoRA直接“续到30K”：其学习率在500-step horizon已衰减，LoRA topology/无EMA/7D objective也不是AXIS论文的dense 30K配方。
- 主研究重新从 `axisrobotics/Pi0.5-Axis` params 初始化，使用 canonical `physical-intelligence/libero` 10Hz、fresh QUANTILES norm、dense full-FT、global batch64、EMA0.999和30K目标。
- 当前只批准把full-FT运行时做到100-step preflight ready；配布Blackwell启动仍需用户单独授权。

## 4. 硬件判断

模型约 `3.353B` 参数。已审计估算：FP32 params约13.4GB、Adam state约26.8GB、EMA约13.4GB；再加grads约13.4GB，总计至少约67GB，尚未包含activation与编译峰值。

- 2×V100 FSDP下每卡下限约33.5GB，超过32GB后还没有activation空间，预计无法承载完整dense训练；只值得做有时限的allocation/first-update canary。
- 3×V100虽有更多总显存，但当前uniform microbatch/accumulation无法精确形成global batch64，不能静默改成63或66冒充canonical配方。
- 4×V100是本地canonical preflight的首选；GPU0仍被外部PID占用时必须等待，不能触碰。
- 单张96GB Blackwell理论上更适合承载完整状态，但必须以真实first-update peak和throughput为准；60小时是否可到30K未知。

下一活动计划因此同时解决：dense配置/门禁、两卡快速可行性证明、四卡或Blackwell的可恢复preflight，以及60小时训练预算，而不是立即盲跑30K。
