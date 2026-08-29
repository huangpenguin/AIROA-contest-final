# AXIS Pi0.5 → 标准 LIBERO 30k 研究计划

> 状态：Phase 1 正在执行；本文不代表已启动训练、已获得新 checkpoint 或已完成评分。Phase 2 之前必须通过 Phase 1 全部门禁并形成 clean commit。
> 日期：2026-08-28（Asia/Tokyo）
> 硬件边界：只使用自有 GPU，暂不使用比赛 GPU
> 权重边界：只从 `axisrobotics/Pi0.5-Axis@c342608c489515efa9ba8281400dc648bf3970e1` 的 `params` 开始，不恢复旧仓库 checkpoint
> 数据边界：主干 30k 只使用标准 LIBERO，不混入 LIBERO-plus 或 Track 3 合成数据

## 0. 决策摘要

本计划取代“Axis init 直接在 LIBERO + LIBERO-plus mixture 上做 500/1000/3000-step LoRA”作为主研究路线。新的主线是：

```text
Pi0.5-Axis continual-pretraining params
  → 丢弃 Axis 训练域 normalization
  → 重置 optimizer / scheduler / step / EMA
  → 标准 LIBERO 四套件、全量参数 post-training 30,000 optimizer steps
  → 标准 LIBERO / LIBERO-plus / reverse proxy 分开评估
  → 再决定是否进入 Track 3 合成数据专项适配
```

锁定以下决策：

1. 当前 F0/M0、F0/M1、F0/M2@1000 和 F0/M1@3000 都标记为 `diagnostic_regressed`；保留作失败对照，不从它们续训。
2. 论文复现主臂使用 **full fine-tuning**，不冻结 vision、PaliGemma、action expert、`action_in_proj`、`action_out_proj` 或 `time_mlp`。
3. LoRA、expert-only 和 projection/time 解冻方案保留为第二层消融，不得冒充原 AXIS 30k 配方。
4. 30k 阶段只使用 LIBERO-Spatial、Object、Goal 和 LIBERO-10，合计 1,693 episodes；LIBERO-90 作为长尾评估/后续消融，不进入主复现训练集。
5. LIBERO-plus 只用于评估，不在 30k 期间更新 normalization 或参数。
6. 主训练从同一 run 连续到 30k；1k/5k/10k/20k 是中间观测点，不是每次从 Axis 重新训。
7. “AXIS 优于普通 π0.5”只能在匹配初始权重以外所有条件的 control 完成后宣称；否则只报告 Axis 路线本身的结果。

## 1. 研究问题

### Q1：`Pi0.5-Axis` 处在什么训练阶段？

HF 仓库没有 model card，因此执行上将它当作“AXIS continual-pretraining 后、标准 LIBERO post-training 前”的初始化。不因它附带 LIBERO-like `norm_stats.json` 就推断参数已经 LIBERO 30k。

在 Phase 0 固化以下证据：

- HF revision、文件清单、params tree hash 和附带 norm hash；
- AXIS 论文中 continual pretraining 与 LIBERO fine-tuning 的分阶段描述；
- 本地 step-0 参数加载、固定输入前向和 rollout 基线。

### Q2：补上标准 LIBERO 30k 能否恢复/建立 LIBERO 能力？

主观测量不是 training loss，而是标准四套件的 simulator success、最差 task、collision、stall 和与 step-0 的 paired delta。

### Q3：标准 LIBERO 能力的建立会如何影响 Plus 和逆向任务泛化？

在 0/1k/5k/10k/20k/30k 记录三条分开的曲线：

- 标准 LIBERO 能力；
- LIBERO-plus 扰动泛化；
- Track 3 逆向 proxy 零样本泛化。

逆向 proxy 不决定是否完成主干 30k，避免为追逆向分数打断标准 LIBERO 复现。

### Q4：资源受限时，哪种参数作用域最有性价比？

只在 canonical full-FT 主臂稳定后比较 LoRA-only、LoRA + projections/time 和 expert-only dense FT。这是工程消融，不是原方法复现的替代定义。

## 2. 数据与 normalization 契约

### 2.1 Canonical 30k 数据集

原 OpenPI `pi05_libero` 配置指向 `physical-intelligence/libero`。本仓库已有四个锁定的 no-noops LeRobot 源，episode 数相加为 1,693：

| Suite | Episodes | 当前锁定源 |
|---|---:|---|
| LIBERO-Spatial | 432 | `IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot` |
| LIBERO-Object | 454 | `IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot` |
| LIBERO-Goal | 428 | `IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot` |
| LIBERO-10 | 379 | `IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot` |

不能仅因 episode 数相等就称两者等价。Phase 0 必须二选一：

1. 优先下载并锁定 `physical-intelligence/libero` 的完整 revision；或
2. 对当前四个源与官方集做 task、episode、frame、instruction、state/action 与图像抽样 hash 等价审计。

若无法证明等价，主复现使用 `physical-intelligence/libero`；现有 IPEC 源只用于数据管道消融。

### 2.2 排除项

Canonical 30k 不包含：

- `libero_90` 的 3,921 episodes；
- `lerobot/libero_plus` 的 14,347 episodes；
- Plus category 重采样；
- Track 3 逆向合成/人工数据；
- 根据本地评分追加的 task quota。

### 2.3 抽样语义

- 复现臂使用官方 loader 的原始 frame/episode 抽样语义，不使用 M0/M1/M2 分层 sampler。
- 不因 task 难度、本地成功率或 episode 长度重采样。
- 若为监测建立 offline probe set，它不从 canonical training exposure 中删除，也不用于宣称泛化。
- 真正选模依据是冻结的 simulator rollout manifest。

### 2.4 Normalization

1. 只从 canonical 标准 LIBERO 30k 训练数据生成新 norm asset。
2. 使用官方 OpenPI 同口径统计，不使用当前 mixture weighted norm 代替。
3. Axis 附带 norm 只记录 hash，不被 canonical trainer 加载。
4. 产出 `norm_stats.json`、输入 manifest hash、统计代码 revision、state/action 每维 quantiles 和 finite report。
5. Step-0 同时保存 `Axis params + official LIBERO norm` 的 rollout 基线。它是后续 paired comparison 的真正基线。

### 2.5 两条实验线的采样契约

- Axis 原团队 LIBERO 30k 复现线使用 `physical-intelligence/libero` 原生 **10 Hz**，不插帧、不重复 action、不重采样；对应配置为 `configs/axis_pi05/standard_libero_30k.yaml`。
- 比赛适配线单独使用 20 Hz 的 LIBERO-plus/Track 3 数据契约，对应配置为 `configs/axis_pi05/libero_plus_track3_20hz.yaml`；该线必须重新计算 norm stats，并从完成的 canonical 30k checkpoint 分叉，不能把 20 Hz 要求反向施加到 canonical 10 Hz 数据。

## 3. Canonical 30k 训练配方

### 3.1 来源冲突处理

当前锁定 OpenPI 代码中 `pi05_libero` 为 batch 256；AXIS 论文的 LIBERO 下游阶段报告 global batch 64。不得默默混用两者。

执行前生成 `reference_recipe_audit.json`，逐项记录论文、锁定 OpenPI config 和实际 resolved config。主研究臂选择 **AXIS-paper profile**；当前 OpenPI 只是实现基础。

### 3.2 锁定主配方

| 项目 | Canonical 设置 |
|---|---|
| Init | `Pi0.5-Axis` params only |
| Data | 标准 LIBERO 1,693 episodes，四套件 |
| Steps | 30,000 optimizer updates |
| Global batch | 64 |
| Parameters | full fine-tuning，`freeze_filter=None` |
| Optimizer | AdamW |
| Gradient clip | global norm 1.0 |
| LR | peak `5e-5` |
| Warmup | 10,000 optimizer steps |
| EMA | `0.999` |
| Action horizon | 10 |
| Internal action dim | 32；对外 LIBERO action dim 7 |
| Checkpoint init | reset optimizer/scheduler/step/EMA |
| Plus mass | 0 |
| Synthetic reverse mass | 0 |

关于 32D loss：复现臂首先保留 AXIS/OpenPI 的 padded-32D 目标，不把“只训前 7D”混入 canonical run。7D mask 只能是后续单独消融。

### 3.3 V100 资源等价方案

原实验硬件与当前 4×V100 32GB 不同，因此目标是优化语义等价，不宣称 bitwise reproduction。

1. 先 profile 每卡 microbatch 1 和 2，选择稳定且不 OOM 的最大值。
2. 用 FSDP/参数与 optimizer-state sharding 承载 full FT；不在每卡复制完整 AdamW state。
3. 用 gradient accumulation 将 effective global batch 精确补到 64。例如 4 卡、每卡 microbatch 1 时 accumulation=16。
4. Scheduler step、optimizer step、gradient clipping 和 EMA 只在 accumulation 完成后更新一次。
5. 报告同时记录 `micro_step`、`optimizer_step`、`examples_seen`；“30k”只指 optimizer steps。
6. V100 不使用已经出现数值问题的 BF16 路径。先比较 FP32 与 FP16 activation + FP32 master/optimizer 的 100-step finite/parity，再锁定精度。
7. 若代码尚不支持语义正确的 accumulation，先实现并用小模型或固定 batch 做等价测试，不用缩小 global batch 偷换复现配方。

### 3.4 Checkpoint 与磁盘策略

开跑前估算 params-only 和 full train-state 实际字节数，预留至少 1.5× 全程需求，避免 Orbax 在中途因空间不足留下半完成 checkpoint。

- Resumable full state：1k、5k、10k、20k、30k。
- Params + EMA snapshot：0、10、50、100、250、500、2.5k，以及所有 full-state 点。
- 每个 checkpoint 先写临时目录，restore/forward 通过后原子化标记 `VALID`。
- 不自动删除旧失败实验；需要清理时另行生成明确目标清单。

## 4. 分阶段执行

### Phase 0：来源与配方审计

产出：

```text
data/reports/axis_libero_30k/source_provenance.json
data/reports/axis_libero_30k/reference_recipe_audit.json
data/reports/axis_libero_30k/dataset_equivalence.json
data/reports/axis_libero_30k/resource_preflight.json
data/reports/axis_libero_30k/canonical_payload_manifest.json
```

门禁：

- Axis params revision/tree hash 可重建；
- 官方 LIBERO 数据 revision 或等价证明完整；
- 实际 resolved config 不含 Plus、LIBERO-90 或 M0/M1/M2 sampler；
- 磁盘、GPU、RAM 和预计时间足够；
- 不读取 Axis 旧 norm。

### Phase 1：数据、norm 与 step-0 基线

1. 建立 canonical catalog 和 official-sampling manifest。
2. 按 OpenPI 口径计算标准 LIBERO norm。
3. 审计 8D state、7D action、**10 Hz**、front/wrist、episode boundary、action padding 和 instruction。
4. 用 Axis params + 新 LIBERO norm 完成 fixed-input forward 与冻结 rollout manifest。
5. 使用 Direct-Orbax 建立 C20/B80 冻结面板，分别记录 `chunk=10` 执行 8 步和完整 10 步；保存 step-0 H.264 成功/失败样例和每 task 摘要。

当前 Phase 1 产物位于 `data/reports/axis_libero_30k/phase1/`，包括 canonical catalog、官方逐帧 sampling manifest、fresh norm、finite report、loader audit、fixed-forward fixture 和 rollout lock。IPEC 不等价仅作为历史消融记录，不阻塞 canonical 主线。

**停止条件：** norm/transform roundtrip、state/action 契约或 fixed forward 任一失败时不启动训练。

### Phase 2：full-FT 数值与恢复能力预飞行

逐级执行 1、10、50、100 optimizer steps：

- 确认所有应训 leaf 都有 finite gradient，不应训 leaf 为零；
- 确认 accumulated gradient 与等价 global batch 的小型测试一致；
- 确认 clip 前/后 gradient norm、optimizer state、EMA 和 scheduler 正确；
- 从 50/100 点在新进程中 restore，验证续训后的 step 和 loss 连续；
- 用 20-case calibration rollout 检查策略是否已出现明显 stall/collision 回归。

**停止条件：** NaN/Inf、EMA 不可恢复、step 语义错误、参数作用域不符或 step-100 行为灾难性退化。

### Phase 3：1k 工程验证

从同一 canonical run 继续到 1,000，不重新初始化。

必须完成：

- 0/100/250/500/1000 的 training/offline drift 曲线；
- step-100、500、1000 calibration rollout；
- step-1000 标准四套件 mini panel；
- raw params 与 EMA params 的固定输入对比；
- checkpoint restore、optimizer state、norm hash 和 params tree hash。

如果 step-1000 相对 step-0 标准 LIBERO paired success 下降超过 0.10，或 timeout/stall 上升超过 0.10，暂停长训并优先检查数据、norm、precision、动作表示和 EMA；不用“还没训够”解释明显崩塌。

### Phase 4：5k 与 10k 中期研究

- 5k：完整标准四套件主面板 + 小型 Plus/reverse diagnostic。
- 10k：完整标准 LIBERO、LIBERO-90 长尾、Plus 分扰动、reverse proxy。
- 所有面板与 step-0 使用相同 task、init state、env seed、policy seed、max steps 和 action execution protocol。
- 继续保存主 run；评分低的 checkpoint 不删除，也不会被打包为 winner。

### Phase 5：20k 与 30k 完整曲线

20k 和 30k 都执行：

1. 全标准 LIBERO panel；
2. 冻结 Plus panel，按 base/env/lighting/language/camera_pose/sensor_noise 报告；
3. 冻结 reverse proxy；
4. 每 task 成功率、suite macro、micro、worst-task、collision、stall、步数；
5. step-0/1k/5k/10k/20k/30k 成对 bootstrap 和 learning curve；
6. 失败 H.264、task/instruction/init-state/扰动/动作曲线的 gallery。

30k 是必须完整保留的研究终点，但最佳行为 checkpoint 可能是 10k 或 20k。不因“训到 30k”就自动将 30k 设为 winner。

## 5. 评估面板

### 5.1 标准 LIBERO 主面板

- Spatial、Object、Goal、LIBERO-10 分开报告；
- 每 task 使用固定 init states，保证 checkpoint 间成对；
- suite macro 为主指标，micro 仅作补充；
- 不用 training demonstration replay success 代替 simulator rollout。

### 5.2 LIBERO-90 长尾面板

用于检查不在 30k 训练集中的任务泛化，始终标记 `diagnostic_not_training_source`。

### 5.3 LIBERO-plus 面板

整个 30k 阶段不根据 Plus 结果调整 sampler。Plus 只记录 Axis→LIBERO 学习进程对扰动泛化的自然影响。

### 5.4 Reverse proxy 面板

- 只使用官方已公开的“逆向任务”高层信息设计；
- 不使用 `check_no_leaks.py` 中未公开的具体推测；
- 不把本地 proxy 写成官方 Track 3 manifest 或官方分数；
- 30k 主干的 reverse 结果是 zero-shot diagnostic，不反向影响主干训练。

### 5.5 推理协议锁定

```yaml
model_action_horizon: 10
n_action_steps: 8
action_ema: 0.0
rtc: false
policy_seed: fixed_per_case
max_steps: 300
```

Direct-Orbax 是研究主评估路径。FAST 导出只对通过本地行为门禁的 checkpoint 执行，并要通过 key、fixed-action 和 rollout parity；不得用 SLOW remap 的结果判断模型学习质量。

## 6. 科学对照与参数作用域消融

### 6.1 最小对照

| ID | Init | LIBERO training | 用途 |
|---|---|---|---|
| `A0` | Axis | 0 | step-0 基线 |
| `A30` | Axis | full FT 30k | 主研究臂 |
| `P0` | 官方 π0.5 base | 0 | 初始化对照 |
| `P30` | 官方 π0.5 base | 同配方 full FT 30k | 用于因果归因 AXIS 优势 |

`P30` 不是“恢复旧仓库模型”，而是匹配训练条件的科学 control。如果算力不允许完成 P30，可报告 A30 的绝对曲线并与公开结果定性比较，但不宣称已证明 AXIS 因果增益。

### 6.2 作用域消融

只在 A30 主臂完成或至少通过 5k 稳定性门禁后启动。每个消融都从原 Axis params 开始，使用相同 canonical 数据、norm、global batch 和 rollout cases。

| Arm | Trainable | Frozen | 首轮预算 |
|---|---|---|---:|
| `F_full` | 全部 | 无 | canonical 30k |
| `F_lora` | action expert LoRA | vision、PaliGemma、projections、time MLP | 5k screen |
| `F_lora_proj` | action expert LoRA + `action_in/out_proj` | vision、PaliGemma、time MLP | 5k screen |
| `F_lora_proj_time` | action expert LoRA + `action_in/out_proj` + `time_mlp_in/out` | vision、PaliGemma | 5k screen |
| `F_expert_dense` | action expert + projections + time MLP | vision + PaliGemma | 5k screen |

要求：

- LoRA 必须是 zero-delta 初始化，step-0 与 dense Axis 等价；
- 每个 arm 输出 exact trainable leaf 列表、数量和百分比；
- 冻结规则用参数 tree 断言，不靠宽泛 regex 猜测；
- 先在 100/500/1k 检查崩塌，最多两个非 canonical arm 进入 5k；
- 只有在 5k 对标准 LIBERO 非劣且显存/时间优势清晰的 arm 才可进入 30k。

## 7. 日志、报告与可视化

每个 run 必须固化：

```text
resolved_config.yaml
source_manifest.json
norm_manifest.json
params_scope.json
checkpoint_manifest.json
training_metrics.jsonl
restore_report.json
eval_protocol.json
summary.json
tasks.csv
episodes.csv
failure_gallery/index.html
```

Training metrics 至少包含：

- optimizer/micro step、examples seen、LR、wall time、throughput；
- total loss 和分 action-dimension loss；
- pre/post clip gradient norm、parameter norm、update norm；
- EMA/raw params drift；
- GPU 显存、finite flag、checkpoint duration、data wait time。

Failure gallery 必须能并排 step-0/1k/5k/10k/20k/30k，并显示：

- H.264 front+wrist 视频；
- suite/task/instruction/init-state/env seed/policy seed；
- checkpoint/norm/data/protocol hash；
- goal progress、EEF 路径、action norm、gripper、collision/stall；
- failure tag：错误对象、错误目标、未抓取、掉落、放置失败、语言误解、碰撞、stall。

## 8. 晋级、停止与结论边界

### 8.1 强制停止

- 任何 NaN/Inf；
- checkpoint 不能在新进程 restore；
- optimizer/EMA/step 语义与 resolved config 不符；
- 数据中意外出现 Plus、LIBERO-90 或 synthetic source；
- step-0 参数或 transform 与 Axis 基线不等价；
- 行为在 100/500/1000 点重现当前 95% stall 一类崩塌。

### 8.2 不得提前宣称

- loss 下降不等于 LIBERO 能力提升；
- 30k 完成不等于 30k checkpoint 自动最佳；
- 本地 reverse proxy 不等于官方 Track 3；
- 没有 P30 matched control 时不宣称 AXIS 对普通 π0.5 的因果增益；
- 未完成 FAST parity 时不宣称 checkpoint 已可提交。

### 8.3 完成定义

主研究臂完成需要同时满足：

- Axis params 、标准 LIBERO 数据、norm 和 resolved config 可重建；
- canonical full-FT 连续达到 30,000 optimizer steps；
- 0/1k/5k/10k/20k/30k 均有可恢复、可对照的证据；
- 标准 LIBERO、Plus 和 reverse proxy 都有冻结 manifest 结果；
- 每个失败有统一详情与 H.264 审阅入口；
- 结论明确分开 `observed`、`inference`、`unverified_hypothesis`。

## 9. 与 Track 3 数据研究的接口

Track 3 数据不进入本计划的 canonical 30k。20 Hz LIBERO-plus/Track 3 适配使用独立配置、独立 norm 和独立 checkpoint 命名空间。只有以下条件全部满足后，才可从某个 Axis→LIBERO checkpoint 分叉专项适配：

1. 至少 5k checkpoint 已显示标准 LIBERO 正向能力稳定提升；
2. 逆向合成数据通过规则、环境、回放和去泄漏门禁；
3. 逆向训练有独立 norm/sampling/config/checkpoint 命名空间；
4. 任何 reverse 增量都同时回归标准 LIBERO 和 Plus，禁止用 Track 3 收益换取全局灾难性遗忘。

专项见 `docs/TRACK3_REVERSE_DATA_SYNTHESIS_PLAN.md`。

## 10. 实施分工顺序（供后续执行 agent 使用）

1. `data`: 官方 LIBERO revision/equivalence、canonical catalog、norm、step-0 fixtures。
2. `train-runtime`: full-FT scope、FSDP、gradient accumulation、EMA、precision 和 restore 语义。
3. `eval`: 标准四套件/90/Plus/reverse 冻结面板和 paired statistics。
4. `experiment`: 100-step preflight → 1k → 5k → 10k → 20k → 30k，每阶段根据门禁继续。
5. `ablation`: canonical 主臂稳定后执行参数作用域与 π0.5-base matched control。
6. `review`: 独立复核数据渗入、step 语义、checkpoint 恢复、结果 coverage 和结论边界。

后续 agent 不得同时修改当前其他 agent 正在处理的评测/提交文件；新实现优先使用 `axis_libero_30k` 独立 config、output 和 report namespace。

## 11. 参考基线

- AXIS paper: <https://arxiv.org/abs/2607.21588>
- OpenPI LIBERO example: <https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero>
- 本地锁定 OpenPI `pi05_libero` config: `third_party/axis_openpi/src/openpi/training/config.py`
- 数据与模型锁：`configs/axis_pi05/data_sources.lock.yaml`
- 当前灾难性退化证据：`docs/AXIS_PI05_LOW_SCORE_RECOVERY_PLAN.md`
