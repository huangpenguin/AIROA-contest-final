# T3 Fast400 H300 交接补充：T2 H10K 官方分 0.13

> 创建时间：2026-09-15（Asia/Tokyo）
> 证据等级：`user_reported_not_independently_verified`
> 适用主计划：`docs/AXIS_PI05_T3_FAST400_H300_POSTTRAIN_EVAL_RELEASE_HANDOFF_20260915.md`

用户报告最新提交的 `t2_p30_r6_h10k_r1` 官方 T2 分数为 `0.13`。当前未提供截图或机器可读比赛回执，因此不得标成独立验证的官方事实；但候选决策按用户回报执行。

- T2 H10K：`0.13`
- T2 R6 incumbent：`0.33`
- delta：`-0.20`
- 本地 C20：H10K `13/20`，R6 `16/20`，delta=`-3`
- 决策：H10K 淘汰；T2 最终继续使用 R6 `0.33`
- 禁止：再次提交 H10K、把 H10K 续成 30k、用训练 loss 为 H10K 翻案

该结果不取消 T3 H300 的 matched behavior evaluation，因为 T3 使用 Fast400 reverse gold、新 norm 和不同训练目标；但它强化主计划的 fail-closed 发布门：

1. 必须用 Fast400 norm 配对评测 R6 baseline 与 H300 step300；
2. 必须覆盖 reverse-dev 4 + C20 20；
3. 只有 reverse delta `>= +1` 且 C20 delta `>= -2`、无基础设施错误时，才 fold/FAST/ZIP；
4. 未过门不得仅凭训练 loss 或 T3 与 T2 不同而提交；
5. 不自动上传或比赛提交。

机器证据：`data/reports/axis_libero_30k/t2_p30_r6_h10k_r1_local_gpu3_c20_20260914/official_score_user_reported_20260915.json`。
