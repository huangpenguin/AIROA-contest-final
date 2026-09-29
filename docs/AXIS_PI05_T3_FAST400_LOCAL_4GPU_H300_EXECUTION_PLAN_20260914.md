# AXIS Pi0.5 T3 Fast400 本地四卡 H300 执行计划

> 冻结时间：2026-09-14（Asia/Tokyo）
> 状态：authorized_for_immediate_local_execution
> 用户授权：GPU0/1/2/3 全部可用于本轮；允许接管当前训练决策，但不覆盖已有产物

## 1. 唯一训练身份

- run ID：`t3_fast400_r6_h300_4gpu_r1`
- init：`r6_dual_l2_fresh500_r1/500/params`，params-only + fresh optimizer
- topology：`L2_action_lora_heads`
- data：Fast400 accepted 8 episodes / 4 tasks，80% forward + 20% reverse
- manifest SHA-256：`ba940bbcaff15395803c5a942f004e491bb5a6fce69be8b9b55bb4f759fac97e`
- norm SHA-256：`7e4e24dacb24d6333d32cb982d301761c2f00e8f5daca418109bea226268f333`
- sampling SHA-256：`aed3380f646bf4c2635090b2b6e994b60a492ee064cda6067cbdc4071e6c90a1`

## 2. 四卡配方

- local GPUs：`CUDA_VISIBLE_DEVICES=0,1,2,3`
- JAX mesh：data parallel `batch=4, fsdp=1`
- precision：FP32
- effective global batch：64
- per-device microbatch：1
- gradient accumulation：16
- optimizer steps：300
- warmup：100
- peak LR：`1e-6`
- cosine decay LR：`1e-7`
- EMA：false
- checkpoints：1、100、300；不保存冗余 step0

四卡仅改变一次 optimizer step 的物理执行方式：`1 sample/device × 4 devices × 16 microsteps = 64`。数据、总 batch、optimizer、schedule 与单卡合同保持不变。

## 3. 时间盒与停止规则

1. 0–15 分钟：输入 hash、磁盘、四卡空闲与 JAX mesh preflight；创建隔离 source manifest/cursor/heartbeat。
2. 15 分钟–3 小时：连续跑到 step300。step1/100 是安全检查点，不因尚无收益评测而停。
3. 3–5 小时：fresh restore；优先并行跑 step300 reverse dev 与 C20 retention，step100 仅作诊断对照。
4. 5–6 小时：若 reverse dev 至少 `+1` 且 C20 相对 R6 下降不超过 `2/20`，fold/FAST/ZIP；否则保留 checkpoint 和证据，不宣称新 winner。

训练只因以下技术 hard stop 中止：OOM、NaN/Inf、输入 hash 漂移、checkpoint partial/restore 失败、磁盘不足。四卡吞吐未达到预期但 ETA 仍可接受时不重开身份。

## 4. 输出边界

- checkpoint：`outputs/axis_pi05/track3_fast400_h300/t3_fast400_r6_h300_4gpu_r1`
- evidence：`data/reports/axis_libero_30k/t3_fast400_r6_h300_4gpu_r1_local4`
- 本地结果不是官方分；未完成行为评测前保持 `not_evaluated`。
- 本轮不上传、不自动提交，不触碰 T2 H10K ZIP。
- 官方 30k 只作为论文/基准背景；本 run 的冻结 horizon 是 H300，不能续写为“30k 的前 300 步”。
