# AXIS Pi0.5 比赛最后阶段长周期训练与 Track3 恢复计划

> 创建日期：2026-09-12（Asia/Tokyo）
> 状态：plan_ready_no_execution_started
> 机器决策：data/reports/axis_libero_30k/t1_t2_dedicated_panel_and_official_250_horizon_20260911/final_days_long_horizon_decision_20260912.json
> Track3 数据采集计划保持独立：docs/AXIS_PI05_TRACK3_FAST_TELEOP_400_AND_SHORT_TRAIN_PLAN.md
> 本计划取代 docs/AXIS_PI05_T1_T2_DEDICATED_PANEL_AND_OFFICIAL_250_HORIZON_PLAN_20260911.md 对未来训练的决策权限；旧文件、原始评测和回执保持历史不变。
> 本计划只落盘路线与执行合同。当前没有启动训练、官方服务器、评测、上传或提交。

## 1. 高层结论

比赛最后阶段不再把 100 steps 当作模型收益的终局判断，但三条已经跑完的 canary100 仍全部冻结，不恢复其 train_state：

- T1-P20：比赛分 0.30，低于 R6 的 0.47。
- T2-P30：比赛分 0.31，低于 R6 的 0.33。
- T3 four-gold 80/20：比赛分 0，等于 R6 的 0；本地 reverse dev 也是 0→0/4。

选择的新训练主线是：

1. 第一优先：T2-P30，从 R6 step500 params-only 新开长 horizon LoRA。
2. 第二优先：T1-P20，只在 T2 证明长 horizon 机制有竞争力后排队。
3. Track3：继续 Fast400 收集新 reverse gold；不续现有 four-gold canary。
4. Dense full-FT：比赛期间继续 parked。
5. 不合并 T1/T2/T3 的 mix，不因 GPU 空闲而占卡。

## 2. 为什么 100-step 分数不能回答长周期问题

冻结训练集规模：

| Track | 训练 episodes | 训练 frames | 100 steps × batch64 | 约占 frame mass |
|---|---:|---:|---:|---:|
| T1-P20 | 10,149 | 1,552,977 | 6,400 | 0.41% |
| T2-P30 | 12,696 | 1,948,579 | 6,400 | 0.33% |

旧 canary 又把 cosine horizon 设为 100，step100 已降到约 3e-8。因此 0.30/0.31 测到的是极低数据暴露加短周期学习率终点，不等价于预声明长 horizon 的 step100。

30k 对 batch64 是 1,920,000 个 frame-anchor 样本，约等于 T1 1.24 次、T2 0.99 次 frame-mass。这个数量解释了为什么 30k 值得作为日程身份，但不能把官方 dense 30k 的效果直接外推为 LoRA 必然有效。

## 3. 现有候选与最终回退

| Track | 当前提交默认 | canary100 状态 | 近期动作 |
|---|---|---|---|
| T1 | R6，0.47 | 0.30，冻结 | 等 T2 长 horizon 结论 |
| T2 | R6，0.33 | 0.31，冻结 | 新开 T2 长 horizon |
| T3 | R6 与 four-gold canary 均为 0 | 无收益，冻结 | Fast400 新数据 |

三条 canary ZIP 的机械通过状态继续保留，但 T1/T2 不重复提交，T3 旧 ZIP 不再作为有收益候选。比赛截止前如果没有新模型超过上述基线，T1/T2 使用 R6 fallback。

## 4. 新 T2 长 horizon 身份

Run ID：t2_p30_r6_lh30k_r1

冻结合同：

| 项 | 值 |
|---|---|
| 初始化 | r6_dual_l2_fresh500_r1 step500 params-only |
| optimizer / scheduler | fresh，不加载 canary train_state |
| topology | L2_action_lora_heads |
| 数据 | T2-P30：70% standard + 10% camera_pose + 10% sensor_noise + 10% coarse env |
| manifest SHA | 85700f9d85311cdadbacadcecf2b7c9b6617b77e03f51de42a81681ffc770af4 |
| norm SHA | 39ff8a7cc2ff2a0be7d43659e708f9ec260686e876cf603c0460972b49746b41 |
| precision | FP32；禁止 BF16 |
| effective global batch | 64 |
| peak LR | 3e-7 |
| warmup | 10 optimizer steps |
| decay LR | 3e-8 |
| schedule horizon | 30,000 |
| EMA | false |
| action protocol | chunk10 / execute8 / 20 Hz / FAST-only |

本轮只改变 horizon 语义，不同时改变 mix、norm、rank、LR、warmup、optimizer 或 action objective。这样才能判断短 canary 是否主要受 horizon 影响。

## 5. Runner 必须先改

现有 fresh-canary launcher 的不超过 100 语义保持不变，不扩大它的权限。新增独立 long-horizon runner，至少提供：

- schedule_horizon_steps：固定为 30000，并写入每个 checkpoint metadata。
- stop_after_step：控制本次实际停止点，不改变 scheduler 终点。
- resume：只允许同一 run ID、同一 optimizer state、同一 schedule horizon 和全部输入 hash exact。
- save_steps：0、1、10、100、500、1000、3000、10000、30000。
- dry_run / preflight：检查 resolved config、R6 params、manifest、norm、磁盘、HF/persist 目标与 GPU 身份。

禁止：

- 用现有 canary100 train_state 接长 cosine；
- 把 AXIS_PI05_TRACK12_STEPS 同时作为实际停止点和 scheduler horizon；
- 在 resume 时重置 optimizer、warmup 或 step；
- 修改 fresh-canary launcher 的 100-step 安全边界来绕过新 runner；
- 在 trainvenv 中安装 LIBERO 评测栈。

Runner focused tests 至少覆盖：

1. 30k schedule 在 stop100 后保存，fresh restore 后 LR/step 连续。
2. horizon、manifest、norm、R6 params 或 optimizer identity 任一漂移均 fail closed。
3. stop_after_step 不改变 schedule_horizon_steps。
4. 旧 canary100 checkpoint 被明确拒绝作为 resume source。
5. 同名目录存在时必须显式 new/resume，不允许默认覆盖。

## 6. Blackwell 吞吐与截止时间门

执行 agent 必须从比赛官方页面重新确认准确截止时间，不从聊天猜。定义：

- T_deadline：官方截止时间。
- T_freeze：T_deadline 减 24 小时。
- 24 小时保留给 fold、FAST、ZIP、上传、比赛排队、失败重试和最终回退。

在新 T2 run 的 10→100 段记录真实 seconds_per_optimizer_step，并计算：

- ETA_3000；
- ETA_10000；
- ETA_30000；
- 剩余官方 GPU 配额；
- 至少 6 小时评测/打包余量和 20% 配额余量。

决策：

1. 若 3000 无法在 T_freeze 前完成，不启动长训，直接保留 R6。
2. 若 3000 可完成而 10000 不可完成，首轮上限为 3000。
3. 若 10000 可在余量内完成，仍先停在 3000 取得一次比赛分，再决定续跑。
4. 不在没有实测吞吐和明确截止时间的情况下承诺跑满 30000。

## 7. T2 执行阶梯

### L0：preflight 与 0→10

- 只读检查官方 persist/HF/NFS 的 R6、T2 manifest 和 norm。
- 新建 run ID，不复用 t2_p30_canary100。
- step1、step10 必须 finite、可 fresh restore。
- 记录 GPU、代码与 resolved config 身份。

### L1：10→100

step100 是安全检查点，不是收益终点。满足以下条件即可继续：

- loss、grad、update、params、optimizer state finite；
- fresh restore 成功；
- manifest/norm/config exact；
- 无 dataloader、checkpoint、GPU 或磁盘异常；
- 冻结 C20 相对 R6 下降不超过 2/20。

不要求 step100 已超过比赛 0.33，也不提交 step100。

### L2：100→500→1000

- 保持同一 optimizer/scheduler。
- step500 和 step1000 保存完整恢复点。
- 只做 finite、restore、C20 与明显 collision/stall 安全检查。
- 不因未达到本地 plus +2 自动停止，也不消耗比赛提交额度。

### L3：1000→3000

- step3000 重新跑冻结 C20。
- 通过 fold/JAX→PyTorch→FAST parity、norm 身份、server smoke、CRC/布局/大小门。
- 生成一个新的 T2-3000 ZIP，由用户提交一次。
- 没有回执前不自动继续。

step3000 比赛分决策：

| T2@3000 分数 | 决策 |
|---:|---|
| 大于 0.33 | 成为新 T2 winner；若时间/配额允许，续同一 run 到 10000 |
| 0.31–0.33 | 不是 winner；仅当 C20 安全且 10000 能在 T_freeze 前完成时，允许一次 3000→10000 长周期验证 |
| 小于 0.31 | 停止该 run；保留 R6，不开 T1 同配方 |
| 无法归属或部署失败 | 标为 not_evaluated，先修部署，不推断模型能力 |

### L4：3000→10000

这一段只在 L3 符合上表时执行。step10000 再取一次比赛分：

- 高于当时最佳 T2 分数：保留为 winner。
- 不高于 0.33：停止；不因 30k 日程身份继续烧配额。
- 高于 0.33 且曲线仍改善、T1/T3 不被挤占、可在 T_freeze 前完成：才另行决定是否继续到 30000。

没有自动 10000→30000。30k 是可连续恢复的长期终点，不是最后几天必须消耗完的配额。

## 8. T1 排队规则

T1 不与 T2 并行，不先做独立 P2。现有 4-case T1 面板只作为以后 checkpoint 的 smoke/诊断，不足以覆盖 0.17 的比赛回归。

只有以下条件同时满足，才创建 t1_p20_r6_lh30k_r1：

1. T2@3000 或 T2@10000 严格超过 0.33，证明长 horizon 机制至少在一个 Track 有效；
2. T1 仍有在 T_freeze 前完成至少 3000 steps、评测、打包和一次提交的时间；
3. T1-P20 manifest、norm 与 coarse-env 标签保持冻结；
4. 从 R6 params-only + fresh optimizer 开始，不恢复 T1 canary100。

T1 使用与 T2 相同的 30k schedule-horizon / stop-after 分离、100/500/1000 安全点和 3000 首次收益点。若条件不满足，最终继续使用 R6 0.47。

## 9. Track3 路线

T3 four-gold canary100 的比赛分为 0，当前结论是数据信号不足，不是继续原 train_state：

1. 保持 Fast400 cursor、manifest、raw、pending、accepted 不变，先完成 FT0 recorder。
2. 优先 fresh replay 历史 314-step physical-success red-mug；3/3 + 用户 accept 后才 promotion。
3. 再采 alphabet-soup 与 yellow-book，失败只影响该 episode。
4. 至少新增 2 条 accepted、覆盖 2 个 task ID 后，冻结新 reverse manifest。
5. 合并既有可验证 gold 与新 gold，重新计算 80/20 mix 和 QUANTILES。
6. 新 T3 run 必须从 R6 params-only + fresh optimizer 开始；不恢复 four-gold canary100。
7. T3 仍走数据受限短适配：预声明 300-step horizon，step100 只作安全/早期 reverse 观测，稳定后走到 step300。
8. step300 必须 reverse dev 至少 +1、C20 下降不超过 2/20，才 fold/FAST/ZIP；比赛提交由用户决定。

如果 T_freeze 前仍达不到 2 accepted / 2 task，不再为 T3 启动训练。保留 raw，比赛后继续采集。

## 10. Dense 与 P2

比赛期间 dense full-FT 继续 parked：

- 本地 4×V100 已有跨 optimizer-step OOM；
- 官方 30k full-FT 是 Axis init + 标准 10Hz LIBERO 的独立研究身份；
- 它不能替代 R6 + 20Hz Track-specific LoRA，也不应与最后几天候选竞争配额。

Dense 只在比赛后或以下条件全部满足时另立计划：长 horizon LoRA 已得到可归属分数、数据/norm/部署问题已排除、官方 Blackwell step1/3/10 memory canary 有明确配额、用户再次授权。

T1 P2 暂缓。它最多证明 4-case 面板可评，不能把 T1 canary100 从 0.30 变成可续候选。以后只在真正的 T1 长 horizon checkpoint 已存在时顺手评。

## 11. 资源并行边界

- 官方 Blackwell：T2/T1 长 horizon 训练、checkpoint 即时评测、fold/ZIP。
- 本地 GPU3：Fast400 replay/Teleop/评测，一次一个 simulator。
- 本地 GPU1：仅在官方不可用时做 runner smoke 或 tiny restore；不把单卡 V100 当最后阶段主训练。
- GPU0/GPU2：当前为外部 YOLO，禁止触碰。
- Track3 Fast400 可与官方 T2 训练并行；两者不得写同一 cursor、run ID、manifest 或 checkpoint 目录。

所有长跑先检查 PID、command、cursor、heartbeat、GPU UUID、磁盘和 dirty writer。SSH 断线不代表训练停止；不得启动第二实例。

## 12. Checkpoint、HF 与清理

- 活动 checkpoint 先写官方 NVMe，0/1000/3000/10000 的完整恢复点同步到 persist。
- 每个决策 checkpoint data-push 到固定 HF revision，并记录 tree SHA、metadata SHA、norm SHA 和 resolved config SHA。
- NFS 继续作归档，不直接承载未经验证的活动 Orbax 写入。
- 不删除 R6 step500、R6 fallback ZIP、三条 canary ZIP/回执、Fast400 raw/manifest、当前 winner 和最近一个可恢复 optimizer state。
- 清理只处理已有 deletion_eligibility 中明确列出的路径；本计划不扩大删除范围。

## 13. 最终 24 小时

到 T_freeze 后：

1. 停止启动新训练 identity。
2. 完成正在生成的 winner checkpoint 的 fold/FAST/ZIP。
3. 逐 Track 固定 ZIP path、bytes、SHA、checkpoint tree SHA、norm SHA 与比赛分。
4. T1/T2 若无更高分，回退 R6 0.47/0.33。
5. T3 只提交有机械通过且来源清楚的最佳候选；如果仍全部 0，不伪称改进。
6. 至少留一次上传/部署失败重试窗口。
7. agent 不自动点击比赛提交，除非用户另行明确授权。

## 14. 当前停止点与下一执行者

当前状态：

- T3=0 回执已落盘；
- 新长 horizon runner 尚未实现；
- 新 T2 run 尚未创建；
- 训练、评测、上传、提交均未启动；
- Fast400 FT0 仍是 Track3 数据侧下一步。

下一执行者首先只做 runner 实现、focused tests、官方截止时间/配额/吞吐 preflight，并把结果回报用户。用户明确要求开始后，第一条占用训练资源的命令只能是 t2_p30_r6_lh30k_r1 的 L0，不得启动 T1、T3 four-gold continuation 或 dense。
