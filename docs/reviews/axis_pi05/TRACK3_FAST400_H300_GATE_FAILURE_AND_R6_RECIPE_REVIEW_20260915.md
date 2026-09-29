# Track3 Fast400 H300 Gate Failure and R6 Recipe Review

> 审核时间：2026-09-15T19:20:46+09:00  
> 状态：`h300_step300_rejected_r6_recipe_recovered`  
> 当前最优口径：`R6_is_best_existing_cross_track_fallback_not_a_proven_T3_winner`  
> 下一阶段：`S100_existing_checkpoint_salvage_then_r6_style_recovery`

## 1. 结论

R6 的确是当前已有 Axis LoRA 中证据最完整、跨 Track1/Track2 最好的 fallback，但它不是已证明的 T3 winner。用户记录的历史官方结果是 T1=`0.47`、T2=`0.33`、T3=`0`；最新 T2 H10K 为用户报告的 `0.13`。本轮 Fast400 H300 step300 又在严格 paired panel 上得到 reverse `0/4 -> 0/4`、C20 `15/20 -> 14/20`，因此 H300 step300 必须保持 `behavior_gate_failed_no_zip`。

R6 值得复制的是方法：宽而稳定的数据、fresh zero-delta LoRA、小 trainable scope、保守 FP32 优化、先短臂筛选再 fresh 500-step、多个 checkpoint 和冻结 seed 的 paired behavior gate。不能复制的是“继续用同一 `1e-6` 在 8 条 reverse episode 上训练更多步”。

当前已有 H300 step100 checkpoint，运行它的完整 24-case paired candidate 只需要一个额外 worker，预期远低于重新训练的成本。因此 deadline 路径先做一次 step100 salvage；失败即停止。赛后才进入真正的 R6-style T3 数据/arm 筛选。

## 2. R6 实际是什么

`r6_dual_l2_fresh500_r1` 名字中的 `dual` 指双 GPU formal run，不是 dual-LoRA topology。实际 topology 是 `L2_action_lora_heads`：

- 初始化：`axisrobotics/Pi0.5-Axis@c342608c...` released dense params + exact zero-delta LoRA；
- action expert attention/FFN LoRA rank 32；
- 同时训练 action in/out projection 与 time MLP；
- PaliGemma/vision/base 全冻结；
- trainable `18` leaves、`24,284,192` parameters，占总参数 `0.7194%`；
- objective：`valid7_temporal`，action horizon 10，只优化有效 7D action；
- 数据：`physical-intelligence/libero@a4336d...`，1,693 episodes、40 tasks、273,465 frames、native 10 Hz、无重采样；
- fresh QUANTILES norm SHA：`4c803bf59c52677c36b6b420c614d8a7742e0847708443863e13e67cb80b18f5`；
- FP32 AdamW，beta1/beta2=`0.9/0.95`，epsilon=`1e-8`，weight decay=`1e-10`，clip=`0.1`，peak LR=`1e-6`，warmup 100，EMA off；
- GPU1+GPU2，microbatch 1/GPU，accumulation 32，effective global batch 64；
- fresh 500 optimizer updates，checkpoint `0/1/10/25/50/100/250/500`。

它先在 L1–L4 四臂筛选中由 L2 胜出，再从 released Axis dense params 重新 fresh 训练 500 steps；不是从 screen checkpoint 续跑。

## 3. R6 为什么成为当前 fallback

最终 seeded B80 execute8 使用相同 case/init/policy seed：

- Axis step0：`48/80`；
- R6 step500：`49/80`；
- paired wins/losses/ties：`9/8/63`；
- collision：`23 -> 20`；timeout：`19 -> 15`；
- bootstrap 95% CI：约 `[-0.0875, 0.1125]`。

所以 R6 的本地提升只有 `+1/80`，置信区间跨 0，不能把它解释成确定的架构优势。它之所以仍是当前最优 fallback，是因为：训练、restore、fold、processor、FAST 链路完整，官方 T1/T2 用户报告分数优于后续 T2 H10K，而且没有其他新候选通过 paired behavior gate。

## 4. H300 与 R6 的关键差异

H300 不是 R6 配方的简单复刻：

- 初始化从 R6 step500 params-only + fresh optimizer，而不是 released Axis + zero-delta LoRA；
- 数据换成 12,832 forward train episodes + 8 reverse episodes，source mass=`80/20`；
- reverse episode probability 为 `0.013333–0.066667`，forward median 为约 `0.000056`，单条 reverse 的采样概率约为典型 forward 的 `238–1190x`；
- norm 换成 Fast400 SHA `7e4e24...f333`；
- 仍用 L2 topology、peak LR `1e-6`、warmup 100、global batch 64；
- 四卡 accumulation 16，训练 300 steps。

H300 训练全程 finite，但 300/300 optimizer steps 的 pre-clip gradient norm 都大于 `0.1`；首/中/尾窗口均为 100% clip，pre-clip mean 约 `0.61–0.65`。R6 对应窗口约 `0.10–0.11`，clip fraction `36–68%`。不同数据/norm 下 loss 不能直接横比，但持续全裁剪说明不能把 R6 的 `1e-6` 无条件移植到该窄 reverse mixture。

这些事实与“8 条 reverse 被高倍率重复、step300 reverse 仍 0、C20 丢 1”一致，支持优先检查数据覆盖、采样质量和 continuation LR；它们尚不足以唯一证明某一个因果根因。

## 5. 已核验的 H300 关闭状态

- DL0 controlled shutdown、Fast400 active norm attestation：passed；
- focused tests：独立复现 `36 passed in 59.79s`；
- R6/step300：各 `24/24` completed；
- 62 个 JSON strict duplicate-key parse通过；
- 48 个 action trace no-pickle/finite/`N x 7`，48 个视频非空；
- 无 `.partial`、failure artifact、live worker、policy server 或 trainer；
- GPU0–3：`5 MiB / 0%`；
- step300 release：`fold_authorized=false`、`zip_created=false`。

## 6. 下一步

执行 `docs/AXIS_PI05_T3_POST_H300_STEP100_SALVAGE_AND_R6_STYLE_RECOVERY_PLAN_20260915.md`。不得改写已关闭的 attempt_0002；step100 必须写入全新 namespace。step100 gate 不通过则本比赛窗口停止所有新 T3 candidate 工作。
