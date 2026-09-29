# AXIS Pi0.5 T3 Post-Step100 Failure: Data-First R6-Style Screen Plan

> 创建时间：2026-09-16T01:24:27+09:00  
> 状态：`ready_for_D0_read_only_failure_and_data_audit`  
> 当前阶段：`D0_failure_taxonomy_and_lineage_inventory`  
> 当前 release：`none`  
> 自动上传/提交：`false`

## 1. 结论与目标

H300 step100 与 step300 都没有在冻结的四条 reverse-dev case 上超过 R6：两者均为 `0/4 -> 0/4`。step100 的 C20 为 `15/20 -> 14/20`，step300 同样为 `15/20 -> 14/20`，但丢失的是不同 case。现有 H300 方向没有 release 候选，也没有继续到更多 optimizer step 的行为依据。

下一路线不再继续 H300。复制 R6 的方法论，但先补齐 R6 当时具备而 Fast400 reverse 分支缺失的条件：数据覆盖、独立 lineage、可控 exposure、fresh norm、短臂筛选和 checkpoint 行为门。

关键路径：

`D0只读失败/数据审计 -> D1评测finalizer加固 -> D2新增reverse数据门 -> D3两套mix/norm -> D4四卡25-step短臂 -> D5最多两臂fresh 100-step -> D6条件式300/500`

## 2. 冻结事实与不可变输入

以下路径全部只读：

- step100 evidence：`data/reports/axis_libero_30k/t3_fast400_h300_step100_salvage_20260915_attempt_0001`；
- step100 output：`outputs/axis_pi05/t3_fast400_h300_step100_salvage_20260915_attempt_0001`；
- step300 closed evidence：`data/reports/axis_libero_30k/t3_fast400_r6_h300_paired_eval_20260915_attempt_0002`；
- step300 closed output：`outputs/axis_pi05/t3_fast400_r6_h300_paired_eval_20260915_attempt_0002`；
- R6 checkpoint 与上述两个闭环引用的 R6 role。

冻结 SHA：

- step100 paired summary：`3e3ff78775d9157cd8d767150225b71ec54dd515375c8d40a24cd088167c7f9b`；
- step100 release decision：`459adb758cc70b51d8155a953a5d0364504fb740f8e481e966e81d6c7f78fd71`；
- step300 paired summary：`e74a112aae506a368ff67ed6d7b860a84d1651872a93856f96ad21370a6ec1b4`；
- step300 release decision：`f7d6cc4fb8c72b17a0db8efb27774a749b2448863235c40fc49cdf7a72ea166b`；
- base paired adapter：`9c736c0ac1e3809df0ee35710781ffe8011929220794c58a75cf756e5b8a6c97`；
- step100 comparator：`b9187f2603075c64e0a95b543139c052430feed2bff665dc018b7d891bbb412d`；
- step100 focused test：`3c005ef45280685c8177a0fef8b859222b02d2f69ab80aee90c6313efaa53ffa`；
- Fast400 norm：`7e4e24dacb24d6333d32cb982d301761c2f00e8f5daca418109bea226268f333`；
- R6 params native/Q0：`e98d21511d74b36c1daa17c4aebacfe508960db2cc556f92223c561a4bc2fddc` / `d9b7e0d15781060fccfd61dc1890ad98993f0224e4aebba1dd6658d7c011931f`；
- step100 params native/Q0/metadata：`870022fc61f4e00d283adcc10371ac4ecf53e15c93c871b415a126bdf0bce314` / `0a8ee72f7bb58490ecade86ca2dad351273ba0938116ee25d5c300a34f8657ad` / `d6b64050fe3c82dc7e78236cc00eb6bf5409dfaffab1043d3e242fa7c04b5bde`。

禁止改写或 resume 旧 attempt、重新 merge、fold/FAST/ZIP、上传或提交。

## 3. D0：只读失败归因与数据 lineage 审计

允许新增：

- `tools/axis_pi05/analyze_track3_fast400_failure_trajectories.py`；
- `tests/test_track3_fast400_failure_trajectories.py`；
- 新 evidence root：`data/reports/axis_libero_30k/t3_post_salvage_failure_analysis_20260916_attempt_0001`。

工具必须只读对齐 R6、step100、step300 的同 case/seed/protocol：

1. 输出四条 reverse 的 success/collision/timeout、步数、action shape、aligned RMSE/mean-L2；
2. 输出所有 C20 success flip，当前已知为 task13 `R6/100/300=true/true/false`、task18=`true/false/true`；
3. 不从 action norm 推断语义成功；视频阶段标签若由人工给出，必须标记 `human_annotation`；
4. 对 Fast400 reverse episode 统计 task、source episode、recording lineage、payload SHA、重复/近重复、train/dev 泄漏；
5. 复算每条 episode sampling probability、相对 forward median exposure、每个 task/source 的预期 25/100/300-step exposure；
6. 所有输入 hash、分析脚本 hash、strict JSON 和 no-write namespace fingerprint 写入 summary。

D0 不启动 simulator、policy server 或 GPU。

## 4. D1：评测 finalizer 的最小加固

在未来新行为评测前修复三项非致命 provenance/lifecycle 缺口：

1. final paired summary 与 release decision 同时记录 comparator SHA、worker adapter SHA、当前活动计划 SHA，字段名区分 `worker_plan_sha256` 与 `comparison_plan_sha256`；
2. video gate 使用真实 `ffprobe`，不能只检查文件非空；测试不得再用任意字节串冒充可读 MP4；
3. paired summary 与 release decision 使用 attempt-scoped finalizer/manifest，支持首次在两文件之间中断后的同源码 recovery，同时继续拒绝覆盖已完整关闭的 attempt。

D1 仅 code/tests/fixture；不得借修复重跑 step100、step300 或 R6。

## 5. D2：新增 reverse 数据硬门

进入任何新训练前，每个目标 T3 task 必须满足：

- 至少 3 条 machine-valid accepted demonstration；
- 至少 2 个独立 recording lineage；
- train/dev 按 lineage 隔离，dev 不得是 train 的裁剪、重定时或派生复制；
- typed/no-pickle、finite、20 Hz、action/state/norm 语义一致；
- task/source/lineage/payload SHA 可追溯；
- 失败、pending、over-budget 或仅诊断轨迹不得进入 train。

当前已知只有 8 条 reverse / 4 tasks，不能仅靠提高采样权重补足覆盖。D2 数据采集/晋级需用户另行授权；若任一目标 task 不满足门，D3–D6 均不得启动。

## 6. D3：两套候选 mix 与 fresh norm

仅在 D2 通过后构建不可变的 `95/5` 与 `90/10` forward/reverse source-mass 候选：

- 每套 mix 都独立生成 manifest、sampling table 与 fresh QUANTILES norm；
- 单条 reverse episode 相对典型 forward episode exposure 必须显式报告，最大值不得超过 `128x`；超过时增加独立数据或降低 reverse source mass，不通过复制同一轨迹解决；
- task/lineage exposure 不能由单条样本主导；
- 两套 mix 的 dev panel、R6 baseline、case seed 和评测 norm 合同冻结后不得中途更改。

## 7. D4：四卡并行 25-step R6-style screen

仅 D0–D3 全通过后，GPU0–3 各运行一个独立单卡 arm，fresh optimizer、global batch 32、checkpoint `0/10/25`：

1. GPU0：R6 params + L2，peak LR `3e-7`；
2. GPU1：R6 params + L1 action-expert-only，peak LR `3e-7`；
3. GPU2：released Axis + zero-delta L2，peak LR `1e-6`；
4. GPU3：R6 params + small dual LoRA（action rank32/PaliGemma rank16），peak LR `1e-7`。

每步记录 pre/post clip、finite、scope hash 和 restore。四臂独立 output/evidence/cursor/heartbeat；任何一臂失败只停该臂，共享数据/norm/runner缺陷才全局停止。训练完成后使用同一 lineage-disjoint reverse panel与 C20 gate，最多两臂进入 D5。

## 8. D5–D6：短程选择，不追 30k

D5 的最多两条 100-step formal run 必须从各自冻结 init 重新 fresh 启动，不能续接25-step screen checkpoint。门仍为 reverse 至少 `+1` 且 C20 delta `>=-2/20`；未通过即淘汰。

只有 100-step 通过的候选才可在新授权下评测 300/500。至少两个 lineage-disjoint T3 dev panel都保持增益且 C20 retention 成立，才讨论 1k；30k、dense、比赛 GPU、fold/ZIP、上传和提交均不在本计划授权内。

## 9. 当前权限与下一停止点

- D0 只读分析与 focused tests：authorized；
- D1 code/tests/fixture remediation：authorized；
- D2 数据采集或 accepted promotion：需用户明确授权；
- D3 mix/norm：仅 D2 通过后另行 review；
- D4–D6 GPU 训练：当前不授权；
- fold/FAST/ZIP/upload/submission：不授权。

下一执行者只做 D0，然后落盘结果并硬停在 D1 实现前供复核。
