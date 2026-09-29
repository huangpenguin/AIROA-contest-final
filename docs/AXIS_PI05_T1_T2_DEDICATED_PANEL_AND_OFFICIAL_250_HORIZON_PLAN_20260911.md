# T1/T2 专用面板冻结与 T1 正式 250-horizon 计划（2026-09-11）

> 创建日期：2026-09-11（Asia/Tokyo）
> 权威评测交接：`docs/AXIS_PI05_OFFICIAL_CANARY100_LOCAL_GPU3_EVAL_HANDOFF_20260911.md`
> 机器 JSON：`data/reports/axis_libero_30k/official_canary100_local_gpu3_eval_20260911/EVAL_HANDOFF.json`
> Track3 采集唯一活动计划（不得改其 cursor/manifest）：`docs/AXIS_PI05_TRACK3_FAST_TELEOP_400_AND_SHORT_TRAIN_PLAN.md`
> 本文件是 **official canary 延续** 的唯一活动计划
> 2026-09-11 修订：增加 canary100 诊断 ZIP、比赛分数回流、dense 去留与 checkpoint 迁移/清理；本次修订不执行其中任何阶段
> 当前状态：`canary100_external_regression_t1_0.30_t2_0.31`；用户已报告 T1/T2 canary100 比赛分，不再要求重复提交这两条 ZIP；T1 专用面板 P0/P1 已 `frozen_constructible`；S2=`passed`；S3 已删 11 条资格路径、`/home` 约 453 GiB；自动上传/提交与 250/500/30k 仍为 false
> 下一阶段：不重复提交 T1/T2 canary100；T3 ZIP 已生成但无比赛分、默认不催交；可选官方 Blackwell P2；不启 250
> 长训授权：T1=`false`，T2=`false`，T3=`false`；三条都不自动进入 step500
> 本地诊断 ZIP 生成授权：T1=`true`，T2=`true`，T3=`true`；自动上传/比赛提交=`false`
> Dense full-FT：`parked_not_current_mainline`；不在本计划中启动

前置事实：本地 GPU3 已完成 R6 vs 官方 canary100 的 C20/T3 配对评测。T1 C20 17=17，T2 C20 16→15（−1），T3 reverse 0/4、C20 14=14。C20 掉幅门都过了。当时不能续 250/300，是因为专用 Plus 面板未冻，以及 T3 reverse 没有 +1。现有 `launch_official_blackwell.py` 仍拒绝 fresh `--steps` > 100。

用户 20:41 问过能否因「40 题都见过」改门；随后又明确要求生成可提交 ZIP，以比赛实际分数参与比较。本计划不取消本地技术/retention 门，但把“专用面板 +2”从唯一收益证据改为两条可选收益路径：冻结专用面板，或有完整 provenance 的同赛道比赛分数严格超过 R6 baseline。ZIP 生成不等于长训授权，上传/提交仍由用户操作。

不要合并三条训练。数据分布、QUANTILES、收益目标都不同。合并会破坏归因，并可能把 T2 的 C20 退化和 T3 reverse 分布带进 T1。GPU 空闲是正确门控，不要为占卡消耗官方配额。

## 0. 双轨权威

| 轨 | 计划 | 允许做什么 | 禁止 |
|---|---|---|---|
| Track3 采集 | Fast400 | reverse gold、recorder、replay | 不要改 Fast400 cursor；不要用当前 T3-100 开长训 |
| Official canary 延续 | 本文件 | 存储迁移/清理；T1/T2/T3 canary100 诊断 ZIP；比赛分数回流；T1/T2 专用面板；条件式 250-horizon | 未获得收益证据不开长训；T2 不自动跟；T3 训练仍由 Fast400 数据条件约束；不开 500 |

执行前必须再读 live GPU/PID。本地只用 GPU3。GPU0/1/2 是外部 YOLO，不要碰。

## 1. 延续门（修订为本地 + 比赛外部观测双路径）

相对 **同 mix 的 R6 step0**（R6 params + 该 mix 检查点 assets norm）：

所有路径都必须先满足：checkpoint/restore finite、fold/FAST parity、norm 身份准确、无基础设施异常，以及已冻结 C20 下降不超过 **2/20**。C20 沿用已完成数字，不重跑来凑门，除非协议被明确修改。

收益门满足以下任一路径即可进入“等待用户批准正式长训”，但不自动启动：

1. **本地面板路径**：专用面板合计至少 **+2 paired successes**，且每个扰动类别无净退化。
2. **比赛外部观测路径**：同一 Track、同一提交协议下，canary100 ZIP 的比赛分数严格超过已记录的 R6 baseline，并且 submission ID、ZIP SHA、checkpoint/norm SHA 和结果时间已落盘。

若比赛分数与 R6 相等，记为 `external_neutral`，不能单独授权 250；若更低，记为 `external_regression` 并冻结该 canary。比赛分数优先用于候选排序，但不能覆盖 non-finite、norm 错配、部署 parity 失败或 C20 掉幅超过 2/20 等技术硬门。

不用 offline loss、不用 validation parquet 成功率、不用「训练见过 40 题」代替收益门。

C20 已有结果：T1 Δ0，T2 Δ−1。本轮专用面板评测不得改写这两份 `paired_summary.json`。

## 2. 专用面板：必须先冻结，再评 R6 与 step100

### 2.1 类别

- T1：`lighting` 与 coarse `env`（mix 标签是 `recovered-coarse-not-background-texture`；公开 example CSV 把 coarse env 写成 `Background Textures` / `_table_*`）。
- T2：`camera_pose`、`sensor_noise` 与 coarse `env`（mix 标签是 `recovered-coarse-not-object-layout-or-robot-init`；example CSV 拆成 Objects Layout 与 Robot Initial States，评测必须写明 coarse env 映射，不得假装细粒度）。

### 2.2 冻结合同

每个 case 在第一次 rollout 之前写入 freeze JSON，之后禁止按结果增删：

- `task_id` / BDDL 路径 / BDDL SHA
- `init_state_index` 与 init tensor SHA
- `env_seed`、`policy_seed`
- `perturbation_family` 与可复现参数（或 Plus BDDL 身份）
- `max_steps=300`，20 Hz，chunk10 / execute8
- `case_id` = 上述字段规范化哈希

冻结种子与选择规则必须事先写死（例如 example suite 顺序 + 固定 init 0 + `env_seed=1000+i`）。**不能**看 R6/step100 谁会成功再挑 case。

`validation_episodes.parquet` 只是训练划分。Plus 没有与闭环 BDDL/init 对齐的 provenance，**不能**把 val episode 直接当成可复现闭环面板。

### 2.3 构造性，不许伪造

当前 `pipeline.config.TRACK_PERTURBATIONS` 对 T1/T2 都是空配置。空扰动评测不得冒充 lighting/camera。

公开 example 身份在：

```text
compe/t1/T1_TASKS.csv   # 4 行：Light Conditions×2，Background Textures×2
compe/t2/T2_TASKS.csv   # 8 行：Sensor Noise×2，Camera Viewpoints×2，Objects Layout×2，Robot Initial States×2
```

`libero_t1` / `libero_t2` 会注册这些 task；init 加载会剥掉 `_light_*` / `_table_*` 后缀去找标准 LIBERO pruned_init。P0 必须用 effect test 证明：同一 base 与 perturbed 下，灯光、纹理、相机或噪声字段的 simulator/observation 哈希按预期变化，且同 seed 可复现。

若某一 family 不能在本地 LIBERO-plus 可靠重建：

- freeze 里写 `status: panel_not_constructible`
- `reason`、失败的 effect test、缺的资产路径
- **不要**写假 success/delta
- 该 family 计为未构造，不能用 0/0 假装评过

只有 **全部声明 family 都 constructible 且已冻结**，才进入配对评测。部分 family 失败则整面板不能授权本地 +2 路径；可以等待第 9–10 节的比赛外部观测，不得用剩下的 family 凑 +2。

### 2.4 评测放哪：比赛 GPU 为主，本地不当搬运工

权重已经在官方 persist/NVMe 上，**不要**每次 12 GiB 拷回本地再评。比赛 GPU 最适合「刚写出的 checkpoint、立刻 20 Hz 闭环」。本地 GPU3 只做 freeze/effect test、评测代码开发、官方 Stop 之后的补洞，以及 Fast400 replay。

官方能做的闭环评测（有独立 `evalvenv` 时）：

- 冻结的 C20（标准 LIBERO）
- 冻结的 T1/T2 专用面板（`libero_t1` / `libero_t2` example + 通过 effect test 的 Plus BDDL）
- 正式 250-horizon 的 step100 / step250 门禁评测（R6 vs 该步，同一 freeze）

官方做不到、也不该假装能做的：

- 隐藏正式题（只有提交 scorer）
- Fast400 Teleop / TigerVNC
- 把 LIBERO 装进 `trainvenv`（会打 JAX）

架构必须 **两个 venv、HTTP 切开**：

```text
trainvenv  JAX + OpenPI     -> checkpoint_http_server（只加载 params）
evalvenv   LIBERO + EGL     -> 本仓库 pipeline / freeze JSON 客户端
freeze JSON / C20.jsonl     几 KB，从 git 拉到 NVMe，不要随权重走
结果 JSON/视频               写 ~/data/parc20260911/eval/ 再 data-push
```

`evalvenv` 优先装在 persist（`~/data/.../evalvenv`）或用 lockfile 每次 Start 后在 NVMe 重建。Stop 仍会清 NVMe 解压数据，但评测身份是 freeze 文件。

占用规则：训完 `stop_after_step=100` 后 **同一 session、同一张卡** 立刻评，再决定是否 resume。不要为「以后可能要评」开着空实例烧配额。官方空闲时 Stop；评测代码在本地准备好再 Start。

### 2.5 本轮 T1 面板执行

- P0/P1（constructibility + freeze）在本地 CPU 做完，freeze JSON 进 git。
- P2 优先在官方 Blackwell：权重已在 persist 的 T1-100 与 R6，不必再下载到本机。
- 若官方当时 Stop、或 evalvenv 还没就绪：才退回本地 GPU3，且只拉这次要用的那一套权重。
- T1 面板用 assets norm `f49d37da…`；T2 用 `39ff8a7c…`。
- 新 evidence namespace，例如 `data/reports/axis_libero_30k/track12_t1_dedicated_panel_20260911/`。
- 禁止写入 Q7 `q7_gpu3_frozen_eval`。
- 排队：**先 T1 专用面板**。T1 未出结论前不要开 T2 面板。

## 3. 面板无法构造时

先走第 9–10 节诊断 ZIP 与比赛外部观测。若比赛分数也没有提供正收益，才由用户再发一句明确授权改门。即使改门：

- 最多先批准 **T1 一条探索性正式训练**（仍须第 4 节的 250-horizon runner，不是把旧 canary resume 成 250）
- T2、T3 仍不自动放行
- 探索性训练结果不得称为 official competition result；只有实际比赛回传才可记为 user-reported competition result

未收到比赛正收益或用户明确改门授权前，`panel_not_constructible` = 停止长训，不影响生成本地诊断 ZIP。

## 4. T1 正式训练：禁止把 canary step100 resume 成 250

### 4.1 学习率跳变（必须修）

旧 canary：peak `3e-7`，warmup 10，decay `3e-8`，**horizon=100**。step100 时 cosine 已到约 `3.0e-8`。

若同一 optimizer 却把 horizon 改成 250，step100 的 cosine 位置变成 `(100-10)/(250-10)=0.375`，LR ≈ `2.17e-7`，相对原 step100 约 **7.2× 跳变**。禁止 `--resume --steps 250` 接到现有 canary100 上。

也禁止继续用 `tools/axis_pi05/launch_official_blackwell.py` 做这次正式训练（`MAX_FRESH_CANARY_STEPS = 100`）。

### 4.2 新正式 runner 合同

新建独立入口（建议 `tools/axis_pi05/run_track1_official_250_horizon.py`），不要改 fresh-canary launcher 的 ≤100 语义。

- 从 **R6 params-only + fresh optimizer/scheduler** 开始，不加载 canary `train_state`
- `num_train_steps` / cosine horizon **一开始就固定 250**
- peak `3e-7`，warmup 10，decay `3e-8`（与 canary 配方相同，只改 horizon）
- 独立 `stop_after_step=100`：跑到 100 保存完整 checkpoint（params + train_state + metadata + assets/norm）后退出，不改 scheduler 终点
- 正式 step100 **就在官方机** 用同一冻结 T1 专用面板 + 既有 C20 复评；回传 `paired_summary.json`，不要先拷 12 GiB
- 过门后用 **同一 optimizer state、同一 250-step schedule** 从 100 resume 到 250
- 250 后再在官方机复评；不过门则停
- 不自动 500
- 官方机：不把 LIBERO 装进 trainvenv；另建 evalvenv；不叠 Q7 OpenPI overlay

现有 canary100 权重只作对照，不作这条正式 run 的 optimizer 起点。

## 5. 排队

1. **S0** 只读盘点 checkpoint/ZIP/HF/NFS/官方 persist，生成迁移和删除资格清单。
2. **S1–S3** 先复制、校验、fresh restore，再定点清理；本地可用空间恢复到至少 `550 GiB`。不得边复制边删除。
3. **Z0–Z3** 一次只处理一个候选：先 T1 canary100 折叠、FAST parity、机械验证并生成诊断 ZIP。
4. 用户手动提交 T1；等待比赛结果时可并行做 **P0/P1** T1 专用面板 constructibility/effect test 与 freeze。
5. 按比赛提交额度依次生成/提交 T2，再决定是否提交 T3。R6 fallback 已有旧分数，不重复提交来占额度。
6. **R0** 把比赛 submission ID、ZIP/checkpoint/norm SHA 和分数落盘；按第 10 节更新收益门，不改写本地原始评测。
7. T1 由“本地面板 +2”或“比赛分数严格高于 R6”任一路径提供正收益后，才进入等待用户批准的正式 250-horizon runner 与官方 `0→100→评→250→评`。
8. 再决定 T2 是否走同一流程。T2 canary C20 已 −1；若比赛分数未提高，不得仅凭覆盖率继续。
9. **T3 不因本计划开训**。诊断 ZIP 只提供外部分数；采集仍只走 Fast400。补新 gold、重建 mix/QUANTILES 后才允许新的 T3 canary。
10. Dense 与三条 step500 均不自动进入。

## 6. T3 动作差异（不训练）

只读已有：

```text
outputs/axis_pi05/track3_four_gold_mix_v1/n5_official_t3_step100_paired_eval/full/step_{0,100}/
```

四个 reverse case 均为双方失败。下一份 T3 诊断应报告：是否几乎零动作、是否撞、是否与训练 reverse 指令/start-state 错配。这不能授权 T3-300。新 canary 必须等 Fast400 新 gold + 新 mix + 新 QUANTILES。

## 7. `not_evaluated` / 禁止

未完成前不得写成已跑：S0–S3、Z0–Z3、比赛提交/分数回流、P0–P8、T2 面板、正式 T1 0/100/250、step500、改门探索训。当前三条 canary100 新 ZIP 均不存在；只有其本地生成获得本计划授权。

禁止：GPU0/1/2；合并三条数据；伪造面板；用 val parquet 冒充闭环；fresh launcher >100；把 canary100 resume 成 250；未授权长训；自动上传/比赛提交；启动 dense；动 Fast400 cursor/manifest。

## 8. 阶段状态

| 阶段 | 状态 |
|---|---|
| S0 storage/release inventory | 通过 |
| S1 NFS/HF/persist archive verification | 通过 |
| S2 fresh restore / hash verification | 通过（11 条 `deletion_eligible`） |
| S3 enumerated cleanup | 完成清单内 11 条；`incomplete_below_target`（453 < 550 GiB）；不再扩删 |
| Z0–Z3 T1 canary100 diagnostic ZIP | 通过；用户已报告比赛分 0.30，不重复提交 |
| T2/T3 canary100 diagnostic ZIP | T2 用户已报告 0.31；T3 ZIP 已生成、无分数、不催交 |
| 比赛提交与分数回流 | 用户报告 T1=0.30 / T2=0.31，无 submission ID；T3 未报 |
| P0 T1 constructibility | `passed`（2026-09-12 GPU3 `--create-env`） |
| P1 T1 freeze | `frozen_constructible`；`p2_authorized=true`；250 仍 false |
| P2 T1 dedicated eval | 面板可评，优先官方 Blackwell；本地未跑；不能解冻 canary100 长训 |
| P3 T1 official 250-horizon runner | 未开始 |
| P4 T1 official 0→100 | 未授权 |
| P5 官方机复评正式 100 | 未授权 |
| P6 T1 100→250 | 未授权 |
| T2 面板与正式训 | 未授权 |
| T3 动作差异 | 未开始 |
| Fast400 reverse gold | 仍由其自己的计划授权 |
| Dense full-FT | parked；本计划不启动 |
| step500 | 禁止自动进入 |

## 9. 现有 ZIP 与 canary100 诊断 ZIP

### 9.1 现在已经存在的机械可提交 ZIP

已有 R6 FAST fallback：

```text
artifacts/submission_axis_r6_fallback_revalidated_20260906.zip
bytes   9,383,571,677
sha256  95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a
entries 3401
layout  FAST-only
CRC     passed
```

它与 `submission_axis_r6_step500_fast_positive_score_20260831.zip` 是同 inode 的两个名字，不是两个模型。该 ZIP 已对应用户报告的历史比赛分数 T1=`0.47`、T2=`0.33`、T3=`0`；缺少 submission ID/官方回执时只称 `user_reported_official_score`。它可以保底提交，但重复提交不能提供新的模型对比信息。

### 9.2 三条 canary100 当前可打包、但尚无 ZIP

| Track | 完整 checkpoint | norm | 当前 ZIP |
|---|---|---|---|
| T1 | `outputs/axis_pi05/official_canary100/t1_p20_canary100/100` | assets `T1-P20`，评测锁 `f49d37da…` | `not_generated` |
| T2 | `outputs/axis_pi05/official_canary100/t2_p30_canary100/100` | assets `T2-P30`，评测锁 `39ff8a7c…` | `not_generated` |
| T3 | `outputs/axis_pi05/official_canary100/t3_four_gold_canary100/100` | assets `four_gold_80_20_v1`，评测锁 `faf3f20f…` | `not_generated` |

三者各约 12 GiB，均包含 `params`、`train_state`、`_CHECKPOINT_METADATA` 和候选自己的 `assets/.../norm_stats.json`。HF 记录 commit 分别为 T1 `42e9464…`、T2 `6282753…`、T3 `c1825ad…`；执行时必须 refetch/校验，不能只相信 upload 时记录。

### 9.3 候选专用 fold → FAST → ZIP 合同

现有 `tools/axis_pi05/export_fast_submission_weights.py` 锁死 R6 candidate/model/norm hash，不能通过替换路径直接复用。新增候选专用入口或先安全参数化，并保留 R6 锁定模式的原语义：

1. 为一个 candidate 写 input lock：Track、Orbax tree SHA、checkpoint metadata SHA、norm SHA、R6 base SHA、代码 HEAD、nested OpenPI HEAD。
2. 从 raw LoRA 恢复，折叠全部 LoRA，断言无残留 LoRA leaf，再转换成候选专用 LeRobot/PyTorch 目录；processor 必须来自该 Track norm。
3. 固定 fixture 比较 raw JAX LoRA → JAX folded → PyTorch folded → FAST：全部 finite、shape/token/mask 一致、gripper sign flip=0，既有部署 parity 门 `max_abs <= 1e-3`。
4. 生成 FAST-only staging；运行 policy server import/smoke、静态 `validate_submission.py`、CRC、重复/绝对/穿越路径、entry/size、chunk10/execute8 与 norm 身份检查。
5. ZIP 边界按仓库规则运行 targeted tests、全量 tests 和 `git diff --check`，写 release lock、manifest、SHA256 与失败闭包。
6. 任一门失败只冻结该 ZIP，不改 checkpoint/norm，不回退到旧 M0/M1/M2 processor。

命名建议：

```text
submission_axis_t1_p20_canary100_fast_diagnostic_<timestamp>.zip
submission_axis_t2_p30_canary100_fast_diagnostic_<timestamp>.zip
submission_axis_t3_four_gold_canary100_fast_diagnostic_<timestamp>.zip
```

通过机械门后只写 `diagnostic_submission_candidate=true`、`official_result=false`。本计划允许生成本地 ZIP，不允许 agent 自动上传或比赛提交。

### 9.4 生成与提交顺序

1. T1：C20 17=17，优先级最高。
2. T2：C20 16→15，第二。
3. T3：reverse 0=0；只有比赛额度不稀缺时才提交，但生成 ZIP 本身仍允许。
4. R6 fallback 只保留作回滚，不重复占用提交额度。

## 10. 比赛实际分数回流与决策

### 10.1 回执格式

用户完成提交后，执行 agent 只做结果落盘，不代替用户点击提交。每次提交独立写入：

```text
data/reports/axis_libero_30k/official_score_feedback_20260911/
  <track>_<candidate>/
    submission_receipt.json
    score_evidence.<png|json|html>   # 用户实际提供什么就保存什么；没有则不伪造
```

`submission_receipt.json` 至少记录：

- Track、candidate/run ID、checkpoint step
- ZIP 绝对路径、bytes、SHA256、release-lock SHA
- raw checkpoint tree SHA、folded model SHA、FAST model SHA、norm SHA
- 比赛 submission ID、提交与完成时间、状态、原始分数
- baseline 来源与数值、`score_delta`
- 证据是 API/页面导出、截图还是仅用户口述
- `official_result=true` 仅在有比赛回执证据时使用；只有口述则写 `user_reported_official_score=true`

不得把比赛结果回写进第 1 节的本地 `paired_summary.json`；外部与本地证据并列保存。

### 10.2 同赛道 baseline

| Track | R6 baseline | 当前来源 |
|---|---:|---|
| T1 | 0.47 | 用户报告 |
| T2 | 0.33 | 用户报告 |
| T3 | 0.00 | 用户报告 |

若后续拿到原 submission ID/页面回执，追加 provenance，不覆盖“此前仅用户报告”的历史语义。

### 10.3 分数决策矩阵

| 结果 | 标签 | 后续 |
|---|---|---|
| candidate score > 同 Track R6 | `external_gain_gate_passed` | 与技术/C20 门合并后进入等待用户批准的下一训练或候选选择 |
| candidate score = R6 | `external_neutral` | 不单独授权 250；继续专用面板或停在当前候选 |
| candidate score < R6 | `external_regression` | 冻结该 candidate，不续其长训 |
| 提交失败、超时、无可归属回执 | `not_evaluated` | 修部署/回执，不推断模型能力 |

T1 若严格高于 0.47，可替代“专用面板 +2”作为收益证据，但正式 250-horizon 仍等待用户明确开训。T2 同理以严格高于 0.33 为外部正收益，且仍保留 C20 `−1` 风险标记。T3 若高于 0，只能说明 T3-100 ZIP 值得保留；它不修改 Fast400 cursor，也不自动授权 T3-300。

## 11. Checkpoint 迁移与数据清理

### 11.1 当前容量事实与目标

2026-09-11 只读盘点：

```text
/home available                                      ~455 GiB
/mnt/home NFS available                              ~14 TiB
official_canary100 T1/T2/T3                          ~36 GiB
track3 q5_local_gpu3 step0/1/10                      ~36 GiB
R6 r6_dual_l2_fresh500_r1 全 checkpoint 历史         ~95 GiB
单个 canary100 checkpoint                            ~12 GiB
```

现有本地训练 runner 的安全门是 `500 GiB`。清理目标设为 **至少 550 GiB 可用**，留出一个候选的 fold/FAST/ZIP staging、checkpoint 临时文件与失败残留空间。NFS 是 `soft` mount，默认只作归档；活动 JAX checkpoint 仍写本地盘，除非独立验证 NFS Orbax 原子语义。

### 11.2 S0：只读 inventory，不移动

生成机器清单，逐个记录绝对路径、用途、bytes、file count、稳定 tree SHA、关键 metadata SHA、link count/inode、HF repo+commit、官方 persist 身份和当前是否被进程打开。至少覆盖：

- R6 原始 step500 及 0/1/10/25/50/100/250 历史步
- 三条 `official_canary100/*/100`
- Track3 Q5 step0/1/10
- dense full-FT 大 checkpoint 与全部证据目录
- `artifacts/` 中旧 F2/M1 ZIP、R6 fallback 与任何 staging
- T1/T2/T3 manifests、QUANTILES/norm、Fast400 raw/pending/accepted 数据

清单给每个对象赋予且仅赋予一个状态：`keep_local`、`archive_then_delete_source`、`archive_only`、`immutable_keep`、`unknown_do_not_touch`。检查到 live PID、打开文件、临时 Orbax 目录或另一 dirty writer 时，该对象自动变为 `unknown_do_not_touch`。

### 11.3 S1：复制到 NFS/HF，不删除源

归档根建议：

```text
/mnt/home/huang/PARC2026_final_archive/20260911/
  checkpoints/
  dense_history/
  artifacts/
  manifests/
  receipts/
```

使用显式路径列表与保留 hardlink/mtime 的复制方式；不得使用未解析 glob、`~`、仓库根或 `/home` 作为递归目标。每个复制完成后写 source/destination 清单和 hash。优先级：

1. 旧 F2/M1 ZIP 与 staging 移入 NFS quarantine；不让它们参与新候选。
2. dense full-FT 大 checkpoint 归档；小型 JSON/日志/审核继续留在仓库。
3. Track3 Q5 step0/1/10 归档。
4. 三条 official canary100 在各自诊断 ZIP 完成前至少保留一份本地完整 copy；ZIP 完成后可归档。
5. R6 中间 checkpoint 归档；R6 step500 anchor 与机械有效 fallback ZIP 默认保留本地，直到新候选和回滚链闭合。

HF canary commit 目前只是 upload 时记录、未 refetch。S1 必须实际按固定 revision refetch 到临时位置并校验，或用 NFS 完整副本 + fresh restore 作为第二恢复证据，不能把“有 repo 名”当成备份成功。

### 11.4 S2：恢复验证

只有同时满足下列条件，源目录才能进入删除资格清单：

1. source/destination bytes、file count、稳定 tree SHA 一致；
2. `_CHECKPOINT_METADATA`、params、train_state、assets/norm 全部存在；
3. 从 NFS copy 或 refetched HF copy 在 fresh process 成功 restore 到声明 step，params/train_state finite；
4. norm SHA 与权威评测交接一致；
5. 至少两份独立可恢复副本存在，例如 NFS + HF，或 NFS + official persist；
6. 生成 `archive_verification.json` 和逐路径 `deletion_eligibility.json`。

R6 文档历史 tree SHA 与当前 hasher SHA 算法不同；S2 必须对 source/destination 使用同一 hasher 比较，不能因两个算法值不同误删或误判。

### 11.5 S3：定点清理

清理只处理 `deletion_eligibility.json` 中 `archive_verified=true` 的显式路径。禁止 broad `rm -rf`、禁止删除 workspace/root、禁止跟随未解析 symlink。每批清理后记录 before/after `df`、删除路径、bytes、归档位置和恢复命令。

永不作为本轮清理目标：

- `data/reports/` 的原始/派生证据与评测 summary
- `data/manifests_track12/`、`data/manifests_track3/`、候选 norm/QUANTILES
- Fast400 cursor/manifest、raw、pending replay 与 accepted gold
- 当前唯一 R6 fallback ZIP
- 还没有完成 ZIP/release lock 的候选 checkpoint
- 任何状态为 `unknown_do_not_touch` 的对象

达到 `/home available >= 550 GiB` 即停止清理，不为追求更多空闲继续删除。

### 11.6 官方 Blackwell NVMe

官方机另做清单：先确认三条 canary 在 `~/data/parc20260911/`、HF 固定 revision 和本地/NFS 至少各有需要的恢复证据；再归档 `/opt/dlami/nvme/runs` 日志、resolved config 和补丁。只有这些证明齐全后，解压数据副本、失败 run 和可重建 venv 才能进入显式清理清单。不得删除 persist 的 `~/data/parc20260911/`；是否 Stop server 是用户操作，不由清理脚本代做。

## 12. Dense full-FT 去留

### 12.1 当前结论：不需要作为近期主线

Dense 不是“模型能力已失败”，但现在没有足够收益证据支持继续投入。已有本地 4×V100 FP32 dense 线最终只证明：

- donated accumulation 完成过一个可恢复 optimizer step；
- D4 又完成 internal step `1→2`，随后在下一轮 microbatch 1 申请约 `136.69 MiB` 时 OOM；
- 没有稳定连续 10 steps、没有 C20、没有比赛 Blackwell、没有 dense ZIP 或比赛分数；
- dense step0 约 24.9 GB，带完整 train_state 的 step1 约 44.7 GB，存储和调试成本显著高于当前 LoRA。

与此同时，R6 LoRA 已有非零 T1/T2 比赛分数，而三条 track-specific canary100 尚未经过比赛 scorer。当前信息价值最高的动作是先生成诊断 ZIP 并取回真实分数，不是先重启 dense 工程线。

因此本计划固定：

```text
dense_status = parked_not_current_mainline
dense_training_authorized = false
dense_artifacts = archive_preserve_evidence
```

### 12.2 何时才复议

只有同时满足以下条件，才创建一份独立 dense 计划：

1. T1/T2/T3 canary100 中计划提交的候选均已取得可归属的比赛分数；
2. 至少最高优先级 Track 已完成一次正确 250-horizon LoRA，或已有明确证据证明该路径不可行；
3. 失败已排除 ZIP/FAST/norm/控制频率/数据映射问题，剩余假设才是 LoRA 容量不足；
4. 官方 Blackwell 剩余配额、checkpoint 空间和恢复路径已盘点；
5. 用户再次明确授权 dense。

任一 canary/250-horizon LoRA 在对应 Track 严格超过 R6 baseline，就不为该 Track 启动 dense。T3 的问题当前首先是 reverse 数据信号，不把 dense 当成绕过 Fast400 的手段。

### 12.3 若未来激活 dense

未来 dense 必须是独立实验，不并入本计划 cursor，也不复用 T1/T2/T3 LoRA run ID：

1. 优先在官方大显存 Blackwell 做 `restore/preflight → step1 → fresh restore → step3 → step10`，不再在本地 4×V100 重复旧 OOM 路径。
2. 冻结 precision、optimizer、EMA、batch、dataset/norm 与 sharding；一次只改变一个已审计的 memory-lifetime 因素。
3. step10 连续稳定、finite、可恢复后，才讨论 50/100；不直接跳 1K/30K。
4. dense 候选仍需相同 C20、部署 parity、FAST/ZIP 和比赛分数闭环。
5. 任何 dense 正式训练、上传或提交都需要新的明确授权。

## 13. 步数门 vs 官方 Axis 30k（2026-09-12 用户提出，待高层审核）

用户指出：官方 Axis 路线大约是 **30000 optimizer steps** 且效果好，而本仓库用 100/250/禁止自动 500 的门把续训拦住了，怀疑门设计有问题。

这是 **open design review**，不是本轮开训授权。机器摘要：`data/reports/axis_libero_30k/t1_t2_dedicated_panel_and_official_250_horizon_20260911/step_horizon_gate_design_review_20260912.json`。

必须分开的事实：

1. `axisrobotics/Pi0.5-Axis` 是 continual-pretraining 初始化，**不是**已经做完 LIBERO 30k post-train 的模型。
2. 仓库里本来就有 `docs/AXIS_PI05_STANDARD_LIBERO_30K_RESEARCH_PLAN.md`：标准 LIBERO **dense full-FT 连续 30k**，1k/5k/10k/20k 只是同一 run 的观测点。它没跑完是因为本地 4×V100 FP32 OOM，不是因为 30k 被判定为错误目标。
3. 当前 LoRA canary 的 100-step 上限、禁止把 canary100 resume 成 250、以及不自动 500，是为了避免学习率终点跳变、配额空转和不可归因的 mix 变更。它们 **没有**证明「100 步已经足够」。
4. 用户报告的 canary100 比赛分 T1=`0.30`、T2=`0.31` 低于 R6 `0.47`/`0.33`，只能冻结 **这两条 100-step canary 的续训**。它既不能证明「再训到 30k 会更好」，也不能证明「步数门是唯一根因」。

后续高层审核应单独回答：

- canary 的 `stop_after_step=100` 应保持「失败即停」，还是改成「预声明长 horizon 里的检查点」？
- 本地面板/外部分数门应阻止 **宣称收益**，还是连 **同一 optimizer 继续往 1k/30k 走** 也一起禁止？
- 若重开 30k，身份是 parked dense full-FT、从 R6 重新开的长 horizon LoRA，还是两者分开？不得把 step 数从一种 topology 抄到另一种。

在新计划明确授权前：不启 250/500/1k/30k，不把 canary100 `train_state` 接到更长 cosine，不合并三条 mix。

