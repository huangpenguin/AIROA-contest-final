# AXIS Pi0.5 比赛最后四天训练执行计划

> 创建日期：2026-09-12（Asia/Tokyo）
> 时间预算：96 小时
> 状态：t2_p30_r6_h10k_r1_user_reported_training_started_20260913
> 赛后问题底稿：`docs/AXIS_PI05_OFFICIAL_BLACKWELL_T2_H10K_LAUNCH_ISSUES_20260913.md`
> 训后移交（下一任审核入口）：`docs/AXIS_PI05_T2_H10K_POST_TRAIN_AGENT_HANDOFF_20260913.md`
> 官方截止：2026-09-17 23:59 JST（用户 2026-09-12 确认）
> H0：2026-09-12 18:53 JST；内部停新训：2026-09-15 18:53 JST；官方冻结：2026-09-16 23:59 JST
> 机器决策：data/reports/axis_libero_30k/t1_t2_dedicated_panel_and_official_250_horizon_20260911/final_4day_execution_decision_20260912.json
> 长期研究背景：docs/AXIS_PI05_FINAL_DAYS_LONG_HORIZON_AND_TRACK3_RECOVERY_PLAN_20260912.md
> Track3 数据计划：docs/AXIS_PI05_TRACK3_FAST_TELEOP_400_AND_SHORT_TRAIN_PLAN.md
> 本文件是四天窗口内训练顺序、horizon 和停止点的最高优先级说明。它不恢复旧 canary，不改 Fast400 cursor，也不自动上传或提交。

## 1. 最终选择

比赛内第一条且默认唯一的长训练是 T2-P30 LoRA，从 R6 step500 params-only + fresh optimizer 开始。

不选择：

- 现有 T1/T2/T3 canary100 的 train_state continuation；
- 三条 Track 合并训练；
- 比赛期间 dense full-FT；
- 直接承诺 30k；
- 为利用空闲 GPU 同时开多条未经评分的 run。

选择 T2 的原因：

- T2 canary100 为 0.31，相对 R6 0.33 只低 0.02；
- T1 canary100 为 0.30，相对 R6 0.47 低 0.17；
- T2 是用最少一次长周期实验验证 horizon 假设的低风险入口；
- T3 canary100 仍为 0，必须先改变 reverse 数据，而不是只增加旧数据的训练步数。

## 2. 四天时间盒

从执行者接手并确认官方截止时间开始记 H0：

| 时间 | 主任务 |
|---|---|
| H0–H6 | runner 修改、focused tests、数据/norm/params/磁盘检查、Blackwell 吞吐基准 |
| H6–H48 | T2 主训练，完成 3k；若选择 10k horizon，则保存 3k checkpoint 并继续 |
| H36–H60 | T2@3k 行为评测、fold/FAST/ZIP 与首次比赛分；可与后续训练交错，但不能破坏同一 optimizer |
| H48–H72 | 完成可达的 T2 终点；只有满足条件才插入 T1 或 T3 短训练 |
| H72–H96 | 禁止新训练 identity；只做评测、打包、上传、比赛回执、最终候选与回退 |

H72 是比赛内部冻结线。实际官方截止若早于 H96，所有阶段按同样比例前移，最后 24 小时保留不变。

## 3. 开跑前只允许一次 horizon 选择

30k 不作为本次比赛 horizon。四天内应完成一个完整 cosine，而不是预声明 30k 后只停在高学习率早期。

执行者先在 Blackwell 做不超过 100 optimizer steps 的吞吐测量，但测量 run 与正式 run 身份分开。使用稳定段计算 seconds_per_optimizer_step，并加入：

- 15% 训练波动余量；
- checkpoint 保存与 data-push 时间；
- 至少 6 小时的 T2 评测、fold、FAST、ZIP 时间；
- 最后 24 小时完全不计入训练。

在正式 step1 前二选一：

### 方案 A：H10K，优先

当预计 10,000 steps、checkpoint 和一次恢复验证能在 H60 前完成时，使用：

- run ID：t2_p30_r6_h10k_r1
- cosine horizon：10,000
- 实际训练终点：10,000
- checkpoints：1、10、100、1000、3000、10000

### 方案 B：H3K，时间不足时

当 10k 不可完成，但 3k 能在 H48 前完成时，使用：

- run ID：t2_p30_r6_h3k_r1
- cosine horizon：3,000
- 实际训练终点：3,000
- checkpoints：1、10、100、1000、3000

方案一旦在 step1 前冻结，不得把 H3K train_state 改成 H10K，也不得把 H10K 改成 30k。若两者都不能留足发布时间，则不开新长训，T1/T2 保持 R6。

## 4. T2 冻结配方

| 项 | 值 |
|---|---|
| init | r6_dual_l2_fresh500_r1 step500 params-only |
| optimizer / scheduler | fresh |
| topology | L2_action_lora_heads |
| data | T2-P30 |
| manifest SHA | 85700f9d85311cdadbacadcecf2b7c9b6617b77e03f51de42a81681ffc770af4 |
| norm SHA | 39ff8a7cc2ff2a0be7d43659e708f9ec260686e876cf603c0460972b49746b41 |
| mix | 70% standard + 10% camera_pose + 10% sensor_noise + 10% coarse env |
| precision | FP32 |
| effective global batch | 64 |
| peak LR | 3e-7 |
| warmup | 10 |
| decay LR | 3e-8 |
| EMA | false |
| action | valid7_temporal，chunk10 / execute8 / 20 Hz |

本轮只检验更长完整 horizon，不同时调整 mix、norm、LR、rank、冻结层、EMA 或 action objective。

## 5. Runner 修改

保留现有 fresh-canary launcher 的不超过 100 限制。新增正式 runner，明确分离：

- schedule_horizon_steps；
- stop_after_step；
- save_steps；
- new 与 resume；
- frozen identity hash。

Runner 必须做到：

1. 只从 R6 params 加载初始参数，拒绝 canary100 train_state。
2. 同一正式 run resume 时保持 optimizer、scheduler、horizon、manifest、norm 和代码身份不变。
3. 同名输出目录存在时默认拒绝，必须显式 new 或 resume。
4. step1、step10、step100 均能 fresh restore。
5. 任一 horizon 或输入 hash 漂移 fail closed。
6. checkpoint metadata 同时记录 global optimizer step 和冻结 horizon。

H0–H6 只允许实现、focused tests 和 preflight。正式训练是否启动仍由用户或接手执行者的明确执行指令决定。

## 6. T2 连续训练与门

### step1 / step10

只检查：

- params、optimizer state、loss、grad、update finite；
- 实际训练 leaf 与 L2 声明一致；
- fresh restore 成功；
- GPU/进程/磁盘/checkpoint 无基础设施错误。

### step100

step100 是安全点，不是收益门。不得因为比赛 canary100 曾低分而在此停止新的 H3K/H10K run。

继续条件：

- 数值和恢复门通过；
- resolved config、manifest、norm、R6 params exact；
- 无数据污染、学习率跳变或 checkpoint 部分写入。

不在 step100 生成比赛 ZIP。

### step1000

- 保存完整 checkpoint；
- 做一次有限动作分布、collision/stall 与 C20 smoke；
- 只有 C20 下降超过 2/20、明显动作崩塌或技术 hard stop 才中止；
- 未超过 0.33 不构成停止理由。

### step3000

无论选择 H3K 或 H10K，step3000 都是第一个行为与外部分数点：

1. fresh restore；
2. 冻结 C20；
3. fold/JAX→PyTorch→FAST parity；
4. policy server smoke、CRC、路径与大小验证；
5. 生成 T2@3000 ZIP；
6. 用户提交并落盘比赛分。

如果选择 H10K，默认允许同一 optimizer 继续到 10k，不等待 3k 分数，除非：

- 技术 hard stop；
- C20 下降超过 2/20；
- 3k 比赛分已经回传且低于 0.25；
- ETA 漂移会侵入最后 24 小时。

### step10000

仅 H10K：

- 完成 cosine 终点；
- 跑同一 C20 与部署链；
- 生成 T2@10000 ZIP 并取得最终 T2 分数；
- 10k 不再续 30k。

候选选择只看可归属比赛分和技术硬门：

- 高于 0.33：新 T2 winner；
- 等于或低于 0.33：最终使用 R6；
- 回执无法归属：不覆盖 R6。

## 7. T1 条件队列

四天内 T1 默认不训练，R6 0.47 是强 fallback。

只有以下条件全部满足，才在 H48 前创建一条 T1 run：

1. T2@3000 已严格高于 0.33；
2. T2 不再需要 Blackwell 长训练；
3. T1 H3K 完整 horizon、评测、打包和一次提交可在 H72 前完成；
4. 从 R6 params-only + fresh optimizer 开始；
5. 使用 T1-P20 原 manifest/norm，不恢复 T1 canary100。

T1 只允许 H3K，不在四天窗口内再开 T1 H10K。若 T2 没证明 horizon 有效，或时间不足，跳过 T1，使用 R6 0.47。

## 8. T3 并行路线

T3=0 已落盘。four-gold canary100 不续训。

Fast400 使用本地 GPU3，可与官方 T2 Blackwell 训练并行：

1. FT0 完成 recorder；
2. 历史 314-step red-mug fresh replay 3/3；
3. 用户 accept 后才 promotion；
4. 再采 alphabet-soup、yellow-book；
5. 在 H48 前达到至少 2 条新增 accepted、覆盖 2 个 task ID。

只有达到数据门，才允许：

- 重建 reverse manifest；
- 合并合格旧 gold 与新 gold；
- 重算 80/20 mix 和 QUANTILES；
- 从 R6 params-only + fresh optimizer 新开 T3 H300；
- step100 作安全点，稳定后走到 step300；
- reverse dev 至少 +1 且 C20 下降不超过 2/20 后生成 ZIP。

若 H48 仍不满足 2 accepted / 2 task，本届比赛不再开 T3 训练。保留 raw，比赛后继续，不用旧 four-gold 数据多训来替代新信息。

## 9. 并行与 GPU

- 官方 Blackwell：T2 主训练；条件满足后才轮到 T1 或 T3 短训练。
- 本地 GPU3：Fast400 replay/Teleop/eval。
- 本地 GPU1：runner smoke 或应急 restore，不作为四天主训练。
- GPU0/GPU2：外部 YOLO，禁止触碰。

官方与本地可以并行，但禁止两个写者共享同一 run ID、checkpoint、cursor、manifest 或 norm 目录。

## 10. Dense、P2 与不做事项

比赛四天内：

- dense full-FT 不启动；
- T1 P2 不单独占官方配额；
- 不恢复三条 canary100；
- 不做 T1/T2/T3 merged mix；
- 不自动删除现有 winner、R6、canary ZIP、Fast400 raw 或最近恢复点；
- 不自动上传或提交。

Dense 30k、完整 T1 H10K、T2 H30K 和更大 T3 数据计划全部进入赛后研究 backlog。

## 11. 最后 24 小时

H72 后：

1. 不创建新训练 run。
2. 不修改 horizon、mix、norm 或 optimizer 身份。
3. 完成已有 checkpoint 的 fold、FAST、ZIP、CRC、server smoke。
4. 固定每条 ZIP 的 path、bytes、SHA、checkpoint tree SHA、norm SHA。
5. T1 无新正收益使用 R6 0.47。
6. T2 无新正收益使用 R6 0.33。
7. T3 只使用机械通过且 provenance 清楚的最佳 ZIP；若所有分数仍为 0，诚实保留 0。
8. 预留至少一次上传或部署失败重试。
9. 比赛提交仍由用户完成，除非用户另行明确授权 agent。

## 12. 当前停止点

已完成：

- T1/T2/T3 canary100 机械 ZIP；
- 用户报告比赛分 0.30/0.31/0；
- T3=0 回执；
- 四天 horizon 选择与时间盒计划；
- 正式 long-horizon runner：`tools/axis_pi05/run_track12_long_horizon_lora.py`；
- OpenPI 已拆开 `schedule_horizon_steps` 与 `stop_after_step`；fresh-canary launcher 仍限 100 步；
- focused tests `20 passed`；T2 manifest/norm/R6 params 本地 preflight 通过；
- 冻结 T2 norm 已放入 mix 目录，SHA `39ff8a7cc2ff2a0be7d43659e708f9ec260686e876cf603c0460972b49746b41`；
- 吞吐 probe 已跑完并锁定 **H10K**；
- 用户 2026-09-13 约 16:42 JST 确认官方 `t2_p30_r6_h10k_r1` 已成功启动。启动过程中的失败分类见
  `docs/AXIS_PI05_OFFICIAL_BLACKWELL_T2_H10K_LAUNCH_ISSUES_20260913.md`。

尚未执行 / 仍禁止：

- 本文件不记录当前 optimizer step；以官方 log / checkpoint 为准；
- Fast400 FT0；
- 评测、fold、FAST、ZIP、上传或提交；
- 第二份 `train.py`、Stop My Server、resume canary100、改 mix/norm/LR。

### 官方 Notebook 隔夜运行与防丢合同（2026-09-13 修订）

官方资料 v1.1 p.15 的精确规则是：关闭浏览器标签不会立即停止 Notebook；只有连续 1 小时既无
cell 执行也无处理的完全 idle 才自动停止；非 idle 也受启动后 12 小时自动停止约束。浏览器标签、
Mac 锁屏或本地网络连接本身不是官方定义的活动判据，因此不再要求“留标签、Mac 不休眠”。

参与者经验库记录过一例隔夜训练消失及后续讨论，但没有进程、OOM 或退出日志，不能据此证明
“后台活跃训练会在 1 小时被强杀”。本项目对任何类似事件按以下顺序分类：

1. 先查训练日志末尾、W&B 最后时间/step、PID、GPU、完整 checkpoint 与实例启动时间；
2. 只有训练仍有处理证据且恰在 idle 窗口被停止，才把平台 idle 判定列为候选根因；
3. 若训练先报错/OOM/退出，随后一小时关机，根因是训练失败后进入 idle，而非 idle 杀死活跃训练；
4. 无上述证据时标记 `cause_unresolved`，不得把猜测写成事实。

正式隔夜运行优先让训练 launcher 在 Notebook cell 中前台等待（不使用 `--detach`）；浏览器可关闭。
若已经用 `--detach` 启动，不重启第二训练实例，可另开一个前台监督 cell，持续检查真实 PID、日志推进、
GPU/step 和 checkpoint；监督发现训练退出后立即结束并报告，不做无实际工作的 keepalive。

持久化要求：不要等到 step1000 才首次保存隔夜恢复点。下一次尚未启动的 long-horizon run 应增加
不改变 optimizer/scheduler 的时间型或更密集 step checkpoint（目标最大可丢训练时间不超过 30 分钟），
每个完整 checkpoint 关闭/fsync 后同步到 `~/data` 并执行 `parc-home-sync data-push`。当前已经启动的
`t2_p30_r6_h10k_r1` 不热改 save schedule、identity 或运行代码；只能使用其冻结保存点，并在可用保存点
完成后尽快同步。`/opt/dlami/nvme` 仅作临时工作区，不能作为唯一恢复来源。

训后官方卡只保留 data-push、JAX 0.11 fold、persist 核对；C20/ZIP/提交见
`docs/AXIS_PI05_T2_H10K_POST_TRAIN_AGENT_HANDOFF_20260913.md`。
证据：`data/reports/axis_libero_30k/t1_t2_dedicated_panel_and_official_250_horizon_20260911/h0_h6_preflight_20260912.json`；
启动问题：`data/reports/axis_libero_30k/t2_p30_r6_h10k_r1_official_launch_issues_20260913/summary.json`；
训后移交：`data/reports/axis_libero_30k/t2_p30_r6_h10k_r1_post_train_handoff_20260913/summary.json`
