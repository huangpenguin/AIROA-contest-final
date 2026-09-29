# AXIS Pi0.5 Track1 / Track2 官方 Blackwell LoRA 训练计划（设计稿）

> 创建日期：2026-09-10（Asia/Tokyo）
> 修订：2026-09-10 按仓库内 R6 / LoRA screen / 续训合同重写为可执行设计
> 授权来源：用户要求在已启动的官方 JupyterHub GPU 上执行 Track1/Track2，并按本仓库文档设计经验、参数与目标
> 并行边界：本地 Track3 Fast400 仍是 plan index 的唯一活动采集计划。本文件不启动 Track3 训练，不修改其 cursor、manifest 或现有工作目录。
> 当前阶段：`T12_A0_official_login_and_env_audit`（部分完成；训练未启动）
> 配置锁：`configs/axis_pi05/track12_official_lora.yaml`

本文是 Track1/Track2 官方续训的执行合同，不是把本文件提升为仓库唯一活动计划。Track3 采集继续按 Fast400 计划独立推进。

## 0. 设计目标与经验来源

### 0.1 要解决的问题

正式提交的 R6 FAST ZIP 已经证明 **20 Hz 闭环有效**：用户报告 Track1 `0.47`、Track2 `0.33`、Track3 `0`。本地 seeded B80 只有 `49/80` vs Axis `48/80`（paired +1，CI 穿过 0），所以 R6 不是显著赢家，只是当前合法续训锚点。

本计划不重做 Axis 30K full-FT，也不重做旧 M0/M1/M2 的 50% Plus 混合。目标是：

1. 从 `r6_dual_l2_fresh500_r1` step 500 **只加载参数**，用 **赛道专用、低配比 Plus** 混合做短续训。
2. 用 **行为评测** 而不是 offline loss 决定是否从 100 步续到 250/500。
3. 保住标准 LIBERO C20，同时让 Track1 / Track2 专用面板至少有可复核的 paired 收益。
4. 各自产出 FAST ZIP，**不自动上传或提交**。

### 0.2 经验从哪里来

| 来源 | 采用什么 | 明确不采用什么 |
|---|---|---|
| `docs/AXIS_PI05_TWO_GPU_LORA_TODAY_PLAN.md` | `L2_action_lora_heads`、`valid7_temporal`、AdamW、clip 0.1、EMA off、global batch 64、execute8 | L3/L4 的 PaliGemma LoRA、`padded32_temporal`、L2 初筛用的 `1e-6` 作为本轮主臂 |
| R6 执行链 / handoff | 续训初始化固定 R6 step500 params；fresh QUANTILES；FAST chunk10/execute8 | 回到 Axis init；复用 R6 optimizer/EMA；复用 R6 QUANTILES |
| 本地 B80 / C20 门 | C20 最多掉 `2/20`；paired comparison；分开记 success/collision/timeout | 用 loss 或非配对 rollout 晋级 |
| Track3 续训合同（过夜/四金计划，只借配方） | params-only restore、100-step screen、再决定 250/500、12h 可恢复 | 80/20 正逆向混合；Track3 reverse gold；Q5 的 `1e-6` 作为本轮主 LR |
| 官方 GPU policy / 四金 Q7 | NVMe 工作盘、`~/data` 持久化、`parc-home-sync`、12h 停机可 resume；不因 Blackwell 显存改 recipe | dense full-FT、BF16、改 global batch、unsupported Docker |
| 用户当前指令 | T1-P20 / T2-P30 质量配比、`3e-7`、先 T1 后 T2、Track3 完全不动 | 同时跑大网格；用 episode index 猜细粒度标签 |

### 0.3 本轮明确不做

- Track3 训练、cursor、manifest、既有工作目录。
- dense full fine-tuning（V100 已多次卡在 interstep OOM，且已停车）。
- 旧 M0/M1/M2 的 50% Plus。
- 本地 4×V100 上的正式 T1/T2 训练。GPU0 外部 YOLO 永不触碰；GPU1/2 现亦被外部占用；GPU3 现有本仓库 Python 进程，本计划不抢。
- 关闭浏览器当停机。计费停止必须 Hub Control Panel → **Stop My Server**，且只在本阶段工作结束后由用户确认。
- 自动 `omni submit` / 上传 ZIP。

---

## 1. 共同基线（冻结）

这些字段在整个筛选期间禁止改动。显存只允许改 **microbatch**，并用 accumulation 把 effective global batch 钉在 64。

```text
init                       r6_dual_l2_fresh500_r1 / 500 / params only
optimizer / scheduler / EMA fresh（不恢复 R6 train_state）
topology                   L2_action_lora_heads   rank 32
trainable                  action-expert attn/FFN LoRA
                           + action_in_proj / action_out_proj
                           + time MLP
frozen                     dense vision / language / expert dense weights
objective                  valid7_temporal
action horizon             10
deploy                     FAST only, chunk 10 / execute 8
norm                       每个精确 mix 单独重算 per-dimension QUANTILES
optimizer                   AdamW  b1=0.9  b2=0.95  eps=1e-8  wd=1e-10
peak LR                    3e-7
warmup (100-step canary)   10
clip                       0.1
EMA                        off
precision                  FP32（本轮禁止 BF16）
effective global batch     64
microbatch                 Blackwell 上按显存选择；accumulation = 64 / microbatch
fail on nonfinite          true
checkpoints                0, 1, 10, 25, 50, 100, 250, 500
first round                100 optimizer steps only
```

### 1.1 为什么是 L2 而不是 L1/L3

- L1 只训 action-expert LoRA，R6 没有选它。
- L2 在 L1 上打开 action in/out 与 time MLP，是已提交正分 ZIP 的拓扑。
- L3 额外打开 PaliGemma LoRA，初筛 LR 才降到 `3e-7`；本轮要冻 vison/language 主干，避免用更大容量去拟合 coarse Plus 标签。
- `valid7_temporal` 只计外部真实 7D，并排除尾部 temporal padding。禁止切到 `padded32_temporal`。

### 1.2 为什么主臂是 `3e-7` 而不是历史 L2 的 `1e-6`

L2 从 Axis 冷启动时的 screen LR 是 `1e-6`，R6 500 步就是这条臂。本轮是 **已训练 LoRA 换 mix 的续训**：新 QUANTILES、新采样质量、同一套 action head。仓库里所有 R6 续训合同都把 `[3e-7, 1e-6]` 写成 screen，并要求先跑保守臂。

本轮第一轮 **只跑 `3e-7`**。`1e-6` 不是并列网格，只在下面两个条件同时成立时才允许开第二个新 run（fresh optimizer，不得从 `3e-7` 的 train_state 改 LR）：

- C20 相对 step 0 掉分 ≤ 2/20，专用面板无净退化；
- 专用面板 paired success 增益 < 2，即“无回归但提升不足”。

不得因为 offline loss 更低而切到 `1e-6`。

### 1.3 warmup 与 12 小时恢复

- 100-step canary 的 warmup 固定 10。若沿用 R6 的 warmup 100，整个 canary 都停在升温区，评测没有意义。
- 若 100 步过门后续到 250/500：从 **step 100 的 optimizer checkpoint** 继续，不重新 warmup，不换 mix/norm/LR/seed/global batch。decay horizon 与最终授权步数对齐（250 或 500），写进该次续跑配置，不得 silently 沿用 canary 的 100-step schedule。
- 单次 Notebook 最长 12 小时、idle 1 小时停机。训练必须按 `OPENPI_KEEP_STEPS` 可恢复。恢复前核对 PID/cursor、GPU UUID、manifest/norm SHA 与 run dir，禁止开第二实例。

### 1.4 初始化与 norm 隔离

1. 只加载 R6 `500/params`。`train_state/` 可以存在于磁盘，但训练入口不得 restore optimizer/EMA。
2. 官方 `pi05_step005000_submission_py310.zip` 是配布 baseline，**不是** R6，禁止当初始化。
3. 每个 mix 一份 QUANTILES。训练、OpenPI rollout、fold、LeRobot processors、FAST 必须引用同一 `norm_stats.json` SHA。
4. 禁止混用 Axis release norm、旧 R6 norm、旧 M0/M1/M2 norm。
5. 换 mix 后必须先做 **step-0、params=R6、仅换新 norm** 的 paired eval，排除“只换 norm 就退化”。若 step-0 已净退化，停止，不开始 100-step。

---

## 2. 数据合同

### 2.1 官方端已观测事实（A0，2026-09-10）

官方 JupyterLab 已登录，GPU 为 RTX PRO 6000 Blackwell Server Edition，CUDA nvcc 13.2。本选配额当时约剩 58 小时 55 分。

`~/dataset` → `/opt/parc/dataset`。**没有** 解压后的 `libero_combined_20hz` 目录，只有：

- `libero_combined_20hz.tar`
- `libero_combined_20hz_v2.tar`
- `lerobot/libero_combined_20hz.tar`
- `IPEC-COMMUNITY/` 五个 suite 也是 tar
- `pi05_step005000_submission_py310.zip`

禁止把这些 tar 解压进 `$HOME`（40 GiB）。只解压到 `/opt/dlami/nvme`。先 `tar -tf` 审计身份与标签，再决定解压哪些。

### 2.2 采样质量（禁止再变成 50% Plus）

**T1-P20**

| 层 | 质量 |
|---|---:|
| 标准 LIBERO 20 Hz | 0.80 |
| Plus `lighting` | 0.10 |
| Plus `env` | 0.10 |

`env` 只有 coarse label，不得写成 Background Texture。Track1 公布示例对应的开发组合尽量 holdout，不用评测结果反向入训。当前 `configs/axis_pi05/track1_example_holdout.json` 的 `episodes` 仍为空，A1 必须对照官方公开示例填实或显式记录“官方未给出可锁定 episode id”。

**T2-P30**

| 层 | 质量 |
|---|---:|
| 标准 LIBERO 20 Hz | 0.70 |
| Plus `camera_pose` | 0.10 |
| Plus `sensor_noise` | 0.10 |
| Plus `env` | 0.10 |

`env` 覆盖 Objects Layout / Robot Initial States，但 coarse metadata 不能可靠拆开。禁止用 episode index 猜标签。只有官方 `libero_combined_20hz`（优先 v2）**真有** 这两个细粒度字段时，才把最后 10% 改成 5% object layout + 5% robot init，并重算 QUANTILES 与新 mix id（例如 `T2-P30-split`）。没有字段就保持 `T2-P30`。

### 2.3 采样与切分算法

与 `build_mix_v2.py` / `tools/axis_pi05/build_track12_mix.py` 一致：

1. split seed `1000`，validation fraction `0.10`。
2. split unit = `source_repo / revision / source_episode_index`。
3. 训练/验证不共享 episode。
4. 在 suite × task × `perturb_category` 内 episode-balanced，得到 episode 质量 `p`。
5. `frame_weight = p / observed_frame_count`，避免长轨迹额外加权。
6. `sum(p) = 1`，`sum(frame_weight * frames) = 1`，容差 `1e-10`。

本地对照 mix（官方数据 identity 匹配前不得直接开训）：

- T1-P20：train 10149 / val 1217，观测质量 ≈ 0.80 / 0.10 lighting / 0.10 env
- T2-P30：train 12696 / val 1517，观测质量 ≈ 0.70 / 0.10 / 0.10 / 0.10

若官方 combined 与本地 IPEC+Plus revision/fps/state8/action7/gripper 任一不一致：废弃对照 QUANTILES，在官方数据上重建 manifest 与 norm，换 mix id。

Gripper 合同保持 `{-1,+1}`。IPEC 部分导出是 `{0,1}`，loader 必须在入 mix 前统一，并写进 sampler audit。

---

## 3. 评测与晋级门禁

部署口径整个筛选期间锁定 FAST、chunk10 / execute8。主协议 execute8；full chunk 只作诊断。

### 3.1 共同保留门

相对 **同 mix 的 step 0**（R6 params + 该 mix 新 QUANTILES）：

- 标准 C20 success 最多下降 `2/20`。
- 专用面板不得净退化，并且至少 **+2 paired successes** 才允许续 250。
- 不因为 offline loss 更低而晋级。
- 非有限 loss/grad/update/params、restore 失败、train/val 污染：hard stop。

### 3.2 必须分开报告的计数

Track1 / Track2 都要按 episode 报告，禁止把 collision 成功算进主 success：

- goal reached
- collision
- goal reached but collision
- timeout

Track2 额外按阶段记账，失败停在哪一层就记哪一层，未进入的阶段为 `not_evaluated`：

- approach
- contact
- grasp
- transport
- place/release
- collision/timeout

### 3.3 评测点

| 点 | 作用 |
|---|---|
| step 0 | 新 norm 对照；也是所有 delta 的基线 |
| step 50 | 早期预警，不单独晋级 |
| step 100 | 第一轮晋级门 |
| step 250 | 仅 100 过门后；再决定 500 |
| step 500 | 仅 250 仍有稳定收益时 |

配对要求：同一 eval identity、同一 policy/noise seed、fresh process restore。step 50/100/250/500 都与 step 0 配对，不与 Axis 冷启动重新配对作为晋级门。

### 3.4 评测放在哪

优先在官方机做 T1/T2 专用面板与 C20，因为正式分数是官方 20 Hz 闭环。若官方环境缺 simulator/assets：把 selected checkpoint + 同一 norm SHA 拉回本地，等 GPU3 空闲后 fresh restore 评测。不得用 GPU0/1/2。不得在 YOLO 进程上插队。

---

## 4. 官方机执行阶段

工作文件：`/opt/dlami/nvme`。最终 checkpoint、manifest、日志、ZIP 入 `~/data`，关键阶段 `parc-home-sync data-push`。不要先把大数据复制进 home。

建议墙钟（合计约 60 小时，对齐官方 60h 预算口径；本选当时约剩 59 小时）：

| 阶段 | 预算 |
|---|---:|
| A0–A2 环境、数据、代码、R6 | 3 h |
| A3–A6 T1 QUANTILES + 100-step + 评测 | 18 h |
| B1–B5 T2 审计 + 100-step + 评测 | 22 h |
| C fold / parity / FAST / 双 ZIP | 8 h |
| 故障与复验 | 9 h |

单次 session 12 小时，因此 T1 与 T2 几乎必然跨 session。每次恢复先做 step-1 级 restore canary，再续跑。

### T12_A0 登录与环境审计 — 部分完成

已完成：JupyterLab 可达、GPU/CUDA/路径记录、确认 combined 仍是 tar、未解压、未开训、未 Stop Server。

未完成：驱动/JAX 版本、磁盘余量精确值、代码 commit 在官方机上的记录、`parc-home-sync` 试推。

### T12_A1 官方 20 Hz 数据审计与 NVMe 解压

1. `tar -tf` 比较 v1 / v2 / `lerobot/` 副本，记录顶层布局与是否含细粒度 metadata。
2. 选择 **一个** 身份作为训练源（优先 v2，若其 schema 完整）。
3. 解压到 `/opt/dlami/nvme/dataset/...`，写 bytes/SHA/episode 数/fps/state/action/gripper 审计。
4. 判定 T2 最后 10% 能否拆成 object layout / robot init。
5. 与本地 catalog revision 对照。匹配则本地 mix 可作为对照；否则官方重建 mix。

### T12_A2 代码、R6 params、JAX 环境

官方机还没有 R6 params（约 12 GiB）和本仓库 T12 训练入口。必须由用户指定通道（Jupyter 上传、私有 HF、或其他），分片传输并逐片验 SHA。不得用官方 baseline ZIP 代替。

环境：用官方 CUDA 13.2 / Blackwell 构建 JAX+OpenPI venv，不假设本地 V100 栈可复制。记录 `nvidia-smi`、driver、`jax.devices()`、磁盘、root/OpenPI commit（含 dirty bytes manifest，因为根仓当前 dirty）。

A2 结束前做一次 **step 1 compile/update canary**（可先用极小 manifest 或不采样真实 100 步）：验证 restore params-only、global batch 64、finite、无 OOM。由此锁定 microbatch。OOM 时只降 microbatch、提高 accumulation，**不**改 global 64、不切 BF16、不缩 LoRA。

### T12_A3 T1-P20 QUANTILES

在 **最终有效 T1 manifest** 上重算。输出 `norm_stats.json` + provenance + sha256。同一 SHA 锁进训练/评测/fold/FAST。

### T12_A4 T1 step-0 行为

R6 params + 新 T1 QUANTILES，不训练。C20 与 Track1 面板。若相对“R6+旧 R6 norm”出现不可解释净退化，停并报告，不把退化算进 100-step 头上。

### T12_A5 T1 100-step canary

`3e-7`、warmup 10、保存 0/1/10/25/50/100。每步 finite。12h 到点则 data-push 后 Stop Server，下一 session 从 last kept step 恢复。

### T12_A6 T1 paired eval（step 50/100 vs 0）

过门才进入 A7。不过门：保留原 R6 作为 Track1 候选，不堆步数，不改 LR 重跑同一 mix。

### T12_A7 T1 续 250，再决定 500

同一 run 的 optimizer 续跑。250 评测后再决定 500。500 不是默认。

### T12_B1–B5 Track2

A6 有结论后再启动 T2，避免两套 12h 训练抢同一实例。顺序：细粒度标签结论 → T2 QUANTILES → step-0 → 100-step → 同门禁。仅当 P30 无回归但提升不足时，才开 **一个** `T2-P20` 保守臂（例如 80/10/10，具体质量在 B1 冻结），不同时跑网格。

### T12_C 发布前机械门（过行为门的赛道才做）

1. LoRA fold
2. raw / folded parity（沿用现有 `1e-3` fold 门，不把已失败的 strict `1e-6` EGL 门重新当成发布 blocker）
3. QUANTILES hash 全链路一致
4. FAST server precheck、`/act < 10s`、L4 24 GiB 口径若官方文档仍要求则执行
5. Track1、Track2 **分别** ZIP；不自动上传或提交
6. 每个关键阶段 `parc-home-sync data-push`

---

## 5. 停止规则

全局停止：live trainer/notebook 冲突；R6/code/manifest/norm hash 漂移；NaN/Inf；train/val 污染；GPU0 被本计划触碰；Track3 目录被修改；临时盘产物未持久化就停机；12h 恢复时开了第二实例。

局部失败：

- 单条 tar/identity 不匹配：隔离，不拿另一份 tar 的 QUANTILES 凑合。
- step-0 新 norm 已退化：不开始 100-step。
- 100-step 无稳定收益：保留 R6，该赛道本计划结束。
- C20 掉 > 2/20：不续 250，即使专用面板上涨。
- 官方缺细粒度标签：T2 保持 coarse `env` 10%，不猜。

---

## 6. Track3 隔离清单

禁止：启动 Track3 训练；改 Track3 cursor/heartbeat；改 Track3 manifest；改 `data/reports/.../track3_*` 现有工作目录；占用可能的 recorder/simulator；把四条 reverse gold 混进 T1/T2。

本地 Fast400 / 四金 80/20 计划可以继续由其他 agent 推进，但本官方实例本轮只跑 T1/T2。

---

## 7. 当前阻塞（设计已锁，执行未开始）

1. 官方 20 Hz 仍是 tar，未解压到 NVMe。
2. R6 step500 params 与训练代码不在官方机。
3. 官方 JAX/OpenPI 环境未建。
4. T1/T2 QUANTILES 尚未在 **官方最终 mix** 上计算。
5. 根仓 dirty；传输包必须按实际 bytes 建 inventory，不能只写 HEAD。

下一步执行入口：`T12_A1`。在用户指定 R6 传输通道之前，A1 可以先做 tar 列表与标签审计，因为那不依赖 12 GiB 上传。
