# AXIS Pi0.5：`0.47` 基线后的 Dense Full-FT 预飞行计划

> 制定日期：2026-08-31（Asia/Tokyo）
> 输入 commit：`97f3a35`
> 当前状态：`r6_score_047_locked_dense_fullft_preflight_ready`
> 本轮唯一目标：把AXIS标准LIBERO dense full-FT实现、显存、数值、checkpoint/restore和训练时长预算验证到可安全进入长训练冲刺；本轮不启动配布GPU、不跑1K/5K/30K、不生成新ZIP。

## 0. 最终优化目标与基线

竞争目标：在保持已验证FAST部署链的前提下，训练出Track 1正式得分高于`0.47`的checkpoint。

研究目标：尽量复现AXIS原团队的标准LIBERO post-training主干：

```text
axisrobotics/Pi0.5-Axis params
  → discard Axis-domain norm/optimizer/step/EMA
  → physical-intelligence/libero native 10 Hz
  → fresh canonical QUANTILES norm
  → dense full-model fine-tuning, global batch 64
  → 1k/5k/10k/20k/30k same-run ladder
  → paired local selection
  → proven R6 FAST exporter/server
  → human Track 1 submission
```

冻结正式基线：

```text
score                         0.47 (user_reported official Track 1)
ZIP SHA-256                   95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a
R6 local B80 execute8         49/80
Axis step-0 local B80         48/80
deployment                    FAST-only, chunk10, execute8, action_ema=0, rtc=false
```

`0.47`是新checkpoint的提交基线，不是训练loss门限；隐藏分数不得用于逐step调参。

## 1. 不进入本轮的事项

- 不从R6 step-500 LoRA resume、改学习率或重置optimizer后伪称“连续训练”；
- 不使用LIBERO-plus、20Hz、Track3逆向合成、LIBERO-90或M0/M1/M2 mixture；
- 不用BF16训练V100，不把global batch改成63/66；
- 不触碰GPU0外部PID，不等待时擅自kill/reset；
- 不启动官方Blackwell Notebook，不消耗60小时；
- 不跑1K以上、不export FAST、不生成或提交第二个ZIP。

## 2. F0：正式分数与输入锁（CPU only）

建立：

```text
data/reports/axis_libero_30k/dense_fullft_preflight/
  official_baseline.json
  input_lock.json
  memory_budget.json
  disk_budget.json
  THREAD_REPORT_ZH.md
```

`official_baseline.json`只记录用户已提供的Track1=`0.47`、ZIP identity和未知字段；submission ID/log/time保持null。

`input_lock.json`必须锁定：

- Axis revision `c342608c489515efa9ba8281400dc648bf3970e1` 与params tree；
- LIBERO revision `a4336d589d589045d1c56423ffdf3b88a0e19b1f`；
- payload manifest `8ede6200…3cc3`，1693 episodes/40 tasks/native10Hz；
- sampling manifest `41715b29…45b1`；
- fresh norm `4c803bf5…b18f5`，Axis released norm禁止加载；
- step-0 C20/B80 manifest、execute8 protocol和policy seed；
- root/nested commit与dirty status。

重新计算磁盘。不得沿用8月29日约824.5GiB headroom结论：当前可用空间已经下降。至少预留一个完整state临时写入余量、当前有效checkpoint/ZIP和200GiB安全边界；空间不足时先列出候选清理对象，禁止自动删除。

## 3. F1：实现唯一 canonical dense config（CPU only）

在locked OpenPI新增独立配置，例如`axis_pi05_libero_fullft`，不得复用LoRA screen配置。resolved config必须为：

| 字段 | 值 |
|---|---|
| model | pi05 dense `gemma_2b` + dense `gemma_300m` |
| init | Axis released params only |
| state/action | 8D / external7D / internal32D |
| horizon | 10 |
| loss | canonical padded32 temporal |
| freeze filter | `nnx.Nothing`；所有params trainable |
| optimizer | AdamW β1=.9 β2=.95 eps=1e-8 wd=1e-10 |
| grad clip | global norm 1.0 |
| LR | warmup10K to5e-5，之后保持5e-5至30K |
| EMA | 0.999 |
| precision | FP32 on V100；BF16 fail-closed |
| effective global batch | exact64 |
| optimizer steps | 30K target；preflight override最多100 |
| reset | optimizer/scheduler/step/EMA all fresh |

实现并测试：

1. resolved config与`standard_libero_30k.yaml`逐字段一致；
2. dense topology中不存在LoRA leaf，trainable numel与全部参数numel一致；
3. Axis norm、R6 raw/folded checkpoint和mixture manifest均不能进入加载路径；
4. accumulation后只clip/optimizer/EMA/scheduler一次；
5. optimizer step、micro step、examples seen分别记录；
6. full params、grads、Adam、EMA均按FSDP sharding，不能每卡复制完整state；
7. checkpoint支持同步atomic save、fresh-process restore和same-run continuation；
8. `OPENPI_KEEP_STEPS`、save cadence和磁盘retention显式化。

F1只允许单元/fixture、shape evaluation和train-state abstract memory analysis，不分配完整GPU optimizer state。

## 4. F2：硬件拓扑可行性门禁

### F2.1 两卡有时限 canary（GPU1+GPU2）

目的只是回答“2×32GB能否容纳dense first update”，不是开始训练。

```text
CUDA_VISIBLE_DEVICES=1,2
fsdp_devices=2
microbatch_per_device=1
accumulation=32
effective_global_batch=64
precision=float32
max optimizer steps=1
```

记录init、params/optimizer/EMA、forward、backward、update各阶段峰值与OOM位置。编译+运行最多尝试一次；失败后释放GPU并保留完整日志，不用关闭EMA、缩batch语义、冻结层或改FP16来“救活”。

预期内存下限已经接近/超过每卡32GB，因此OOM本身不是代码失败。只有first update finite、每卡至少保留2GiB余量、checkpoint/restore通过，才允许把两卡列为preflight topology。

### F2.2 四卡条件路径

若GPU0自然释放且四卡均可独占，再使用：

```text
CUDA_VISIBLE_DEVICES=0,1,2,3
fsdp_devices=4
microbatch_per_device=1
accumulation=16
effective_global_batch=64
```

用户/外部进程未释放GPU0时状态写`waiting_for_4gpu_or_official_sprint`，不得使用三卡batch63/66替代。

### F2.3 Blackwell路径边界

本计划只生成single-Blackwell候选配置与命令，不启动Notebook。候选从microbatch1/accum64开始，根据实测余量提高microbatch并等比例降低accum，始终保持global64。first-update preflight将作为未来同一次长训练sprint的开头，不单独浪费一次官方会话。

## 5. F3：1/10/50/100-step 数值预飞行

只有F2存在通过的拓扑才执行，并从同一个fresh run连续到100，不得每个点重新初始化。

检查点：`0,1,10,50,100`。每个点要求：

- loss、所有trainable grad/update/param、optimizer state、EMA finite；
- trainable leaf覆盖vision、language、action expert、action projections和time MLP；
- step/examples/scheduler/EMA只在完整global64 update后前进一次；
- clip前后norm、LR、throughput、peak VRAM与host RAM记录；
- step1、50、100 fresh-process restore；step50 restore后续训到51与原run语义一致；
- checkpoint internal step/目录/manifest一致，无半完成目录。

step100只跑冻结C20 execute8，使用与Axis step-0相同env/init/policy seeds。行为门禁仅防灾难回归：

```text
episode errors / NaN / stall       0
success                            >= 8/20
timeout+collision                  不得比step0增加超过4/20
```

由于warmup为10K，step100不要求超过R6；不能用短程success决定dense配方无效。

## 6. F4：吞吐与60小时冲刺预算

用去除首次编译后的step时间生成：

```text
throughput_report.json
official_sprint_budget.json
```

分别估算1K/5K/10K/20K/30K所需训练时间、checkpoint写入、restore、评估和FAST export。官方60小时计划至少保留：

- 4小时环境/数据/input lock与first-update；
- 4小时checkpoint同步、restore、export和异常恢复；
- 至少一次12小时自动停止前的持久化恢复演练；
- 其余最多52小时用于optimizer updates。

若30K预测超过预算，不能把较短checkpoint叫“30K复现”；锁定预算内最高里程碑，优先保证完整可恢复的1K/5K/10K，而不是在截止时留下不可恢复的半run。

## 7. 下一阶段路线图（本轮不授权）

预飞行审核通过后，另行落盘长训练计划：

1. 同一run到1K：C20 + B80、raw vs EMA、restore；
2. 到5K：完整标准LIBERO面板，第一次正式选模；
3. 根据吞吐到10K/20K/30K；
4. winner相对step0/R6做paired evaluation；
5. dense JAX→folded PyTorch→FAST，复用R6已经验证的813-key/vision/strict reload/server/C5/ZIP门禁；
6. 只有人工审核后才提交，目标高于Track1 `0.47`。

Track3 20Hz/reverse数据只从完成的canonical checkpoint分叉，不进入本轮。

## 8. Hard stop 与交付

执行者最多完成F0–F4。以下任一发生立即停止：

- input/hash/norm不一致；
- dense scope缺叶或出现LoRA；
- 非finite、step/EMA/accumulation语义错误；
- 两卡OOM且四卡不可用；
- checkpoint restore失败；
- C20灾难回归；
- 磁盘无法保留atomic full state与200GiB安全边界。

最终状态只能是：

```text
dense_fullft_preflight_passed_ready_for_sprint_review
waiting_for_4gpu_or_official_sprint
dense_fullft_preflight_failed
```

不得在本计划内启动1K、官方Notebook、FAST export、ZIP或比赛提交。
