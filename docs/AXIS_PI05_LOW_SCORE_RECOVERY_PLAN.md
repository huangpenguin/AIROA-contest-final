# AXIS Pi0.5 低分恢复、保守后训练与 Track 3 逆向任务计划

> 状态：待执行计划；本文不代表已完成修复、训练、评分或提交
> 日期：2026-08-28（Asia/Tokyo）
> 基线：2026-08-27/28 本地固定评测、T3 diagnostic@3000、用户提供的 Track 1 官方日志及 Track 3 官方披露
> 当前边界：使用自有 GPU；暂不使用比赛 GPU；不合成新的训练 demonstrations
> 核心目标：先恢复可靠 Axis 基线，再定位 post-training 退化，最后只晋升经基线非劣门禁验证的微调模型

## 0. 执行摘要与新决策

当前不能继续沿用“从三个低分微调模型中选相对最好者，再训练到 3000 steps”的路径。最新证据表明有两条相互独立的问题链：

1. **post-training 本身发生了灾难性退化。** 同一套本地 direct-Orbax evaluator 下，Axis init 明显优于所有微调 checkpoint，因此这不是 FAST/SLOW 转换能够单独解释的问题。
2. **官方提交路径还有 FAST/SLOW、key remap 和 action execution horizon 的一致性问题。** 即使训练模型是好的，错误打包或不一致的执行协议仍可让官方 Track 1 回到 0 分。

本计划锁定以下高层决策：

- 立即冻结 F0/M0、F0/M1、F0/M2@1000 和 F0/M1@3000 的“提交候选”资格；保留原始产物用于诊断，不删除、不覆盖。
- Axis init 必须作为真正可胜出的候选进入 selector，而不只是对照基线。若没有微调模型通过非劣门禁，最终 winner 就是 Axis init。
- 第一优先级是生成并验证 **Axis-init FAST rollback package**，目标是恢复此前约 0.395 的官方能力，而不是先追加训练。
- 本地评测协议与官方提交协议统一为 `chunk=10`、默认只执行前 `n_action_steps=8` 后重新规划；`10` 仅作为 A/B 诊断，不再隐式使用。
- 新训练从短步数、低学习率、零增量 LoRA 和更小作用域重新开始；禁止直接再跑一轮 3000 steps。
- Track 1、Track 2、Track 3 分别选候选，不再假设一个 checkpoint 必须同时是三个 Track 的 winner。
- Track 3 使用官方已公开的“现有 LIBERO 任务的逆向操作”设计 `reverse_task_generalization` 本地 proxy；仍不推测隐藏任务数量，也不把公开 example task 用作训练集。
- 本计划不生成逆向 demonstrations。若以后需要反向轨迹、逆向 action 或环境内采集数据，必须由项目负责人另行授权并另开数据计划。

## 1. 当前结果审计

### 1.1 本地 direct-Orbax 结果

所有结果使用相同 82-case `select_core` manifest，coverage 均完整，且没有 runtime/model/infrastructure error：

| 候选 | micro success | task macro | Track 1 | Track 2 | Track 3 | timeout/stall | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| Axis init + M1 norm | 38/82 = 0.4634 | 0.4700 | 0.925 | 0.500 | 0.000 | 0.5000 | 当前唯一可靠能力基线 |
| F0/M0@1000 | 3/82 = 0.0366 | 0.0300 | 0.075 | 0.000 | 0.000 | 0.9634 | 严重退化 |
| F0/M1@1000 | 3/82 = 0.0366 | 0.0300 | 0.075 | 0.000 | 0.000 | 0.9512 | 严重退化 |
| F0/M2@1000 | 1/82 = 0.0122 | 0.0100 | 0.025 | 0.000 | 0.000 | 0.9634 | 严重退化 |
| F0/M1@3000 diagnostic | 2/82 = 0.0244 | 0.0200 | 0.050 | 0.000 | 0.000 | 0.9512 | 追加训练没有恢复 |

F0/M1@1000 相对 Axis init 的同 case success delta 为 `-0.4268`，95% bootstrap CI 为 `[-0.5366, -0.3171]`。差异远大于抽样噪声，必须按真实能力回退处理。

T3@3000 的 82 cases 中只有 2 个成功；78 个达到 timeout/stall，collision rate 为 0.4268。策略能返回 finite action，server 和 checkpoint restore 正常，但行为能力基本丢失。

### 1.2 旧 winner 决策为什么失效

旧 selector 只允许 M0/M1/M2 三个微调模型相互竞争，Axis init 不能成为 winner。因此它在 1.2%–3.7% 的退化候选中通过 collision tie-break 选出了 M1；这是“相对排序正确、绝对选型错误”。

新 selector 必须先执行基线非劣门禁，再进行候选间排序。没有任何候选通过时应明确输出：

```json
{
  "winner": "axis_init",
  "decision": "baseline_fallback",
  "reason": "all_posttrained_candidates_failed_noninferiority"
}
```

### 1.3 官方 Track 1 0 分与本地低分不能混为一谈

用户提供的官方日志显示 Track 1 为 0/6，每个 episode 满 300 steps；server 走 `SLOW from_pretrained`、`fast=False`，并出现视觉 patch embedding key warning。该问题属于提交/转换链。

本地上述微调低分直接加载 Orbax checkpoint，没有经过 LeRobot `model_fast.safetensors` 转换，所以本地灾难性退化不能归因于 SLOW remap。必须分别修复：

- A 线：Axis/微调模型本体行为是否正确；
- B 线：Orbax → LeRobot/FAST → submission server 是否保持行为等价。

### 1.4 新发现的评测协议不一致

当前本地 `checkpoint_http_server.py` 会消费完整 10-action chunk 后才重新推理；用户提供的历史高分官方配置是 `chunk=10, n_action_steps=8`。这会改变闭环纠错频率，使本地分数不能严格预测官方行为。

新 runner 必须显式记录并控制：

```text
model_action_horizon = 10
executed_action_steps = 8
replan_after = 8
action_ema = 0.0
rtc = false
```

## 2. 高风险根因假设

以下项目是需要实验验证的假设，不得提前写成 `verified_cause`。

### H1：LoRA 作用域过大

当前 F0 虽称 `lora_only`，实际约有 49,987,584 个 trainable parameters，覆盖 2B PaliGemma 和 300M action expert 的全部 18 层 attention/MLP LoRA。它并不是只训练小型 action adapter。对 batch=3 的短域数据，这可能快速破坏 Axis 已有视觉—语言—动作对齐。

### H2：LoRA 初始化不是严格 no-op

released Axis params 不包含本地新增 LoRA leaves；当前 loader 从模板补齐这些 leaves，而 LoRA A/B 默认都使用非零 normal 初始化。Axis-init 对照因此不一定是严格的原始 dense Axis 函数。

新实现应使用标准零增量初始化（例如 A normal、B zero），并证明：

```text
exact_dense_axis(observation, seed)
≈ zero_delta_lora_axis(observation, seed)
```

### H3：32 维内部 action head 的训练 loss 被 padding 主导

数据 action 为 7 维，Axis Pi0.5 内部 action head 为 32 维。当前 transform 将 action zero-pad 到 32 维，而 `compute_loss` 在全部 32 维上求均值，25 个 padding 维度也参与 flow loss。这可能使 78.1% 的 action-dimension loss 用于人工零维度，改变 action expert 表示并压低真实 7 维动作能力。

必须增加可配置 valid-action mask，至少比较：

- 原始 32 维 loss；
- 仅前 7 维 loss，但保留 32 维 diffusion noise；
- 前 7 维 loss且 inactive dims 采用固定 Axis 兼容处理。

在确定最佳方案前，不能直接改变 32 维模型结构或裁切 Axis action head。

### H4：episode 尾部 `action_is_pad` 未进入 loss

loader 能产生 `action_is_pad`，但当前训练 iterator 只向模型传 observation 与 actions；越过 episode 末尾的 action 被 clamp/repeat 后仍参与 loss。需要实现 temporal valid mask，并单独统计各数据源的 padded-horizon mass。

### H5：50% LIBERO-plus 暴露和 1e-5 LR 造成快速遗忘

M0/M1/M2 的 LIBERO-plus mass 都是 0.5，三者只改变标准 LIBERO 内部比例；因此当前消融没有真正测试 Plus 比例。所有 mixture 都退化，不能据此证明 M1 数据配比合理。

新的 sampling arm 至少加入：

- `P10`：Plus 10%，标准 LIBERO 90%；
- `P20`：Plus 20%，标准 LIBERO 80%；
- `P50`：Plus 50%，仅作现有配置对照。

每种 mixture 重新生成 norm，并在训练前先跑 Axis-init + corresponding norm control。若 norm 本身让 Axis 基线明显下降，该 mixture 不进入训练。

### H6：优化目标只追 imitation loss，没有行为保持约束

现有训练 loss 从约 0.18 降到 0.05，但 rollout success 从 0.46 降到 0.04，说明训练 loss 与实际能力严重脱钩。需要加入 frozen Axis teacher 的行为保持目标，并以 rollout early stop 为最终门禁。

## 3. Workstream A：先恢复可提交的 Axis FAST 基线

此工作流优先级最高，可与训练修复解耦。

### A0. 冻结和追溯已知基线

建立四个明确候选：

| ID | 权重路径 | 用途 |
|---|---|---|
| `axis_dense_orbax` | 原始 Axis dense params | 函数权威基线 |
| `axis_zero_lora_orbax` | dense Axis + zero-delta LoRA | 新训练 step-0 基线 |
| `axis_fast_lerobot` | 严格导出的 `model_fast.safetensors` | 提交候选 |
| `axis_slow_lerobot` | `model.safetensors` 动态 remap | 只作故障复现，不作为默认提交 |

每个候选记录 Axis revision、params tree hash、norm hash、converter commit、state-dict key report 和模型文件 SHA-256。

### A1. FAST 转换门禁

1. 包内必须存在 `model_fast.safetensors`，启动日志必须确认 `FAST load model_fast.safetensors`。
2. missing、unexpected、shape mismatch 均为零；视觉 patch embedding weight/bias 单独列出。
3. LoRA fold 前后在固定模型输入上的输出一致；attention 和 FFN LoRA scaling 都要纳入测试。
4. FAST/Orbax 使用同一 M1 norm、图像旋转/resize、state quaternion→axis-angle、instruction 和 action unnormalize。
5. 对 32→7 输出只允许既定 LIBERO slicing；不能在 converter 中静默换 action convention。

### A2. 行为等价门禁

使用固定 observation fixtures 和相同 policy seed 比较 `axis_dense_orbax`、`axis_zero_lora_orbax`、`axis_fast_lerobot`：

- 输入 tensor 哈希、dtype、range 一致；
- action chunk shape 为 `[10, 7]`；
- finite rate 100%；
- 每维 MAE、max error、cosine similarity、gripper sign agreement 有预声明阈值；
- `n_action_steps=8` 的同 case 20-episode calibration rollout 不出现统计显著回退；
- FAST 通过后再跑完整 82-case `select_core`。

若 FAST 不能复现 Orbax 行为，停止打包并修 converter；不得使用 SLOW 路径绕过。

### A3. 紧急 rollback 决策

若提交期限早于训练修复完成：

- Track 1/2 使用通过 FAST 等价门禁的 Axis init；
- 保持 `n_action_steps=8`、`action_ema=0.0`；
- Track 3 也默认使用最安全 Axis 基线，除非另一个候选在冻结 reverse proxy 上有真实成功且不破坏 server 门禁；
- 报告中明确这是 `baseline_rollback`，不声称 post-training 改进。

## 4. Workstream B：修复本地评测与 selector

### B0. 统一执行协议

给 direct-Orbax server 和 FAST server 增加相同的显式参数：

```yaml
action_chunk: 10
n_action_steps: 8
action_ema: 0.0
rtc: false
```

server 只执行 chunk 的前 8 步，第 9、10 步丢弃后重新推理。summary、episode row 和 protocol hash 必须包含这些字段。

### B1. 三个 Track 分别报告

保留共享 manifest 行，但分别生成：

```text
track1_summary.json
track2_summary.json
track3_reverse_summary.json
overall_diagnostic_summary.json
```

Track-specific winner 只使用对应 Track 的冻结 primary panel；overall 只用于工程观察。

### B2. baseline-inclusive selector

对每个 Track 先执行基线非劣门禁：

1. coverage、finite、runtime 和 schema 通过；
2. 与 Axis init 同 case paired success delta 的 95% CI 下界 `>= -0.05`；
3. worst-task success 不得出现超过 0.10 的绝对回退；
4. collision 与 timeout rate 相对基线的上升均不得超过 0.05；
5. 通过后才按 Track primary success、worst-task、collision、timeout 排序。

对于较小 panel，同时报告精确 paired counts；不得只因 CI 较宽就宣称非劣。若没有微调候选通过，Axis init 自动胜出。

旧 `winner_decision.json` 保留为历史，不覆盖；新协议输出 `winner_decision_v2.json`。

## 5. Workstream C：定位退化发生在哪一步

在任何新长训练之前执行短程 collapse localization。

### C0. step-0 等价性

- 原始 dense Axis；
- 当前随机补齐 LoRA 的 Axis baseline；
- 新 zero-delta LoRA Axis baseline；
- 三者使用同一 norm、noise seed 和输入 fixtures。

若 zero-delta LoRA 不能匹配 dense Axis，先修模型/loader，不进入训练。

### C1. 时间轴 checkpoint

新 probe 保存：`0, 1, 5, 10, 25, 50, 100, 250, 500`。已有 T0@10、T1@500、T2@1000 和 T3@3000 只作对照，不代替新零增量 probe。

每个点执行：

- 固定 offline validation loss，按 standard/Plus/类别分层；
- teacher action/velocity drift；
- LoRA delta norm和各层 activation drift；
- 20-case paired calibration rollout；
- action magnitude、gripper sign、EEF displacement、stall/collision；
- FAST 导出只对通过 local gate 的点执行。

一旦 paired success 相对 step-0 下降超过 0.05，立即停止该 arm，不等到 500 或 3000 steps。

### C2. action/loss contract audit

为以下事实增加可执行报告：

- raw action 7 维 → normalize → pad 32 → loss → unpad 7 → unnormalize 的 roundtrip；
- 每维 loss，区分有效 7 维与 padding 25 维；
- `action_is_pad` 的 frame/source/category mass；
- gripper action 的取值、符号和反归一化；
- train/eval 两路 image 旋转、camera key、resize 完全一致；
- 相同 demonstration sample 在 Axis teacher 与 student 下的 action/velocity 差异。

只有通过这一阶段，才把 H1–H6 中某项标记为 `verified_cause` 或 `ruled_out`。

## 6. Workstream D：新的保守 post-training 方案

### D0. 训练共同设置

- 从原始 Axis dense params + zero-delta LoRA 初始化；
- 保持 vision tower、PaliGemma 主干、action projections 和 time MLP 冻结作为第一轮默认；
- effective batch 目标至少 24，优先用 gradient accumulation，不因 V100 数量改变优化语义；
- 优先 LR 为 `3e-7`，第二 arm 为 `1e-6`；在这两者通过短程门禁前不再使用 `1e-5`；
- gradient clip 0.1，记录 pre/post clip norm；
- 无 EMA，固定 seed 至少 2 个；
- checkpoint 密集保存，rollout early stop；
- 每个 arm 独立 norm、sampling、scope 和 objective hash。

### D1. 更小且可解释的参数作用域

按顺序逐级开放，上一层级没有候选通过非劣门禁时，不进入下一层级：

| Scope | Trainable | Frozen | 目的 |
|---|---|---|---|
| `S0` | 无 | 全部 | Axis baseline |
| `S1` | action expert 最后 4 层 LoRA | vision、2B PaliGemma、action expert 前 14 层、projections/time | 最保守适配 |
| `S2` | action expert 全 18 层 LoRA | vision、2B PaliGemma、projections/time | 扩大动作适配 |
| `S3` | `S1` + action in/out projections + time MLP | vision、2B PaliGemma | 对应原 F2 的受控版本 |
| `S4` | `S1` + 2B PaliGemma 最后 2 层 LoRA，LR 为 action expert 的 0.1× | vision及其余主干 | 仅在语言理解确有证据不足时使用 |

参数路径不能用模糊字符串猜测。当前联合 PaliGemma tree 中 base expert 与 action expert 常以无后缀/`_1` 分支区分，实现者必须先生成逐 leaf 映射和数量断言。

### D2. masked imitation + teacher preservation

推荐目标：

```text
L = L_demo_valid7_temporal_mask
  + lambda_teacher * L_axis_teacher_anchor
  + lambda_delta * ||LoRA_delta||²
```

- `L_demo_valid7_temporal_mask` 只在真实 7 action dims 和非 padding horizon 上计算；
- teacher 与 student 使用同 observation、timestep 和 diffusion noise，比较有效动作维的 velocity；
- teacher anchor 使用现有标准 LIBERO 真实数据，不生成新 demonstration；
- 第一轮 `lambda_teacher ∈ {0.25, 1.0}`，teacher-free arm 仅作消融；
- Plus 样本优化 demonstration loss，标准 LIBERO 样本同时承担 preservation；
- 记录 imitation、teacher、delta 三项独立曲线，不能只报告总 loss。

### D3. 数据 mixture

第一轮只比较 `P20` 与现有 `P50`，避免无界网格：

- `P20`：Plus 20%，标准 80%；标准内部优先保证 LIBERO-10 和 LIBERO-90，同时保留 Spatial/Object/Goal 最小质量；
- `P50`：现有 M1，作为退化复现对照。

每个 mixture 重新计算 norm。先用 Axis-init + 该 norm 跑同 case baseline；训练后的 paired delta 必须相对“同 norm 的 Axis baseline”计算。

### D4. 最小实验矩阵与 successive halving

第一轮 6 个短 probe：

| Arm | Scope | Mix | LR | Teacher λ | Loss mask |
|---|---|---|---:|---:|---|
| R0 | S0 | P20 | 0 | 0 | control |
| R1 | S1 | P20 | 3e-7 | 1.0 | 7D + temporal |
| R2 | S1 | P20 | 1e-6 | 1.0 | 7D + temporal |
| R3 | S2 | P20 | 3e-7 | 1.0 | 7D + temporal |
| R4 | S1 | P20 | 3e-7 | 0.0 | 7D + temporal |
| R5 | S1 | P50 | 3e-7 | 1.0 | 7D + temporal |

筛选节奏：

- 25 steps：step-0 等价、finite、teacher drift；淘汰明显异常 arm；
- 100 steps：20-case rollout；只保留不劣于基线的最多 3 个；
- 250 steps：完整 Track 1 mini + Track 2 mini；只保留最多 2 个；
- 500 steps：完整 `select_core`；只有通过 baseline noninferiority 的 arm 才可进入 1000；
- 1000 steps：标准泛化和 reverse proxy；没有明确收益则停止，不启动 3000；
- 3000 steps：只允许一个已在 1000-step 证明不劣且仍持续改善的 arm。

## 7. Track 3 逆向任务专项，但不合成训练数据

### T3.1 本地 proxy 设计

建立 `reverse_task_generalization_v1`：

- 只使用仓库中公开可运行的 T3/example tasks；
- 正向/逆向按 base semantic pair 组织；
- 每个 case 记录真实 BDDL、initial state、goal predicate、instruction 和 hash；
- 至少 10 个固定 init states/task；
- 成功判定来自实际 environment goal，不允许只替换 instruction 文本；
- 结果始终标记 `local_public_reverse_proxy`、`official_result=false`。

公开 example tasks 只用于 evaluation。不得把它们加入 train split，也不得根据 `check_no_leaks.py` 猜测完整隐藏清单。

### T3.2 诊断指标

- reverse success；
- 正向成功但逆向失败比例；
- 抓取成功、离开原容器、到达目标区域、最终放置四阶段进度；
- 多对象任务的子目标完成率；
- 语言对象/容器指代错误；
- collision、stall 和错误恢复；
- 与 Axis baseline 的 paired delta。

### T3.3 当前数据约束下的现实预期

Axis init 和现有所有微调模型在本地 T3 均为 0。由于当前不合成或采集逆向 demonstrations，本阶段不能承诺通过训练显著解决 Track 3。新方案的目标是：

1. 不再为追求未知 T3 增益而破坏 Track 1/2；
2. 用 proxy 识别是否存在自然泛化收益；
3. 若所有候选仍为 0，明确报告 `no_reverse_capability_gain`，而不是继续盲目延长训练；
4. 若后续允许合法构造/采集逆向数据，再单独设计数据来源、去泄漏和训练计划。

## 8. Track-specific 选型与提交策略

### Track 1

主指标为 Track 1 paired success。优先恢复 Axis FAST baseline；任何微调模型必须先通过相对 Axis 的 -0.05 非劣 margin，再比较 collision、timeout 和动作效率。

### Track 2

同样以 Axis baseline 为门禁。由于当前所有微调候选为 0、Axis 为 0.5，旧模型全部禁止晋升。

### Track 3

以 reverse proxy success 为主要诊断；候选还必须满足 server/FAST 和基础动作门禁。若全部为 0，使用最稳定、已通过 FAST 等价的 Axis baseline，不根据训练 loss指定 3000-step 模型。

每个 Track 输出独立：

```text
candidate_manifest.json
fast_export_report.json
local_track_summary.json
paired_to_axis.json
submission_package_manifest.json
decision.json
```

## 9. Failure review 改进

在现有 gallery 上增加“基线成功、微调失败”的 35 个 paired regression case 作为首页集合。每个 case 并排显示 Axis、step10、step100/500/1000/3000：

- H264 视频；
- first action chunk 与每次 replan 的 chunk；
- action norm、gripper sign、EEF displacement、reward；
- LoRA delta/teacher drift；
- task、instruction、Track、init state、seed；
- failure class：定位错误、未抓取、抓取后掉落、错误对象、错误目标、collision、stall。

当前轨迹初步显示微调失败 case 的动作仍 finite、范围也相近，但平均动作幅度通常更小；这只能作为 stall/策略漂移线索，不能单独确定根因。

## 10. 实施顺序与停止条件

### Phase 0：冻结退化候选

- 更新 candidate registry，将旧 T2/T3 标为 `diagnostic_regressed`；
- 保留 checkpoint 和报告；
- 阻止自动 submission picker 选择它们。

### Phase 1：Axis FAST rollback

- exact dense/zero-LoRA 等价；
- FAST state-dict/key/vision gate；
- `n_action_steps=8` 固定评测；
- 生成可提交 rollback package。

**停止条件：** FAST 无法复现 Orbax 时只修转换链，不进入新训练。

### Phase 2：loss/LoRA/data contract audit

- 32/7D loss、temporal pad、zero-delta LoRA、scope mapping；
- teacher drift 和逐层 delta 报告。

**停止条件：** step-0 不等价或数据 roundtrip 不通过时禁止训练。

### Phase 3：6-arm 短 probe

- 25/100/250/500 successive halving；
- 每个 checkpoint 小面板 paired rollout。

**停止条件：** 没有任何 arm 通过 baseline noninferiority，则结束 post-training，保留 Axis baseline。

### Phase 4：1000-step 验证

- 最多 2 个候选；
- Track 1/2、standard generalization、reverse proxy；
- FAST export parity。

**停止条件：** 不能同时证明无灾难回退和至少一个目标面板有收益，则不进入 3000。

### Phase 5：可选 3000-step

- 最多 1 个候选；
- 500/1000/3000 全曲线；
- 自动 early stop；
- Track-specific selector。

### Phase 6：官方环境门禁（继续 deferred）

只有项目负责人重新授权使用官方环境后才执行。此前所有分数均为 local/non-official。

## 11. 推荐提交边界

1. `docs: record low-score evidence and recovery protocol`
2. `eval: align action execution horizon and include Axis in selector`
3. `export: add strict Axis FAST parity and vision-key gates`
4. `model: add exact zero-delta LoRA initialization and scope audit`
5. `train: add valid-action and temporal loss masks`
6. `train: add frozen-Axis preservation objective`
7. `data: add P20 mixture and matching normalization control`
8. `eval: add reverse-task public proxy and phase-progress metrics`
9. `experiments: run short collapse-localization probes`
10. `reports: choose track-specific candidates or baseline fallback`

每个提交只包含对应职责；当前 worktree 有其他 agent 的未提交修改，执行者不得覆盖、回退或顺带提交无关文件。

## 12. 验收清单

- [ ] Axis dense 与 zero-delta LoRA 在固定输入上等价；
- [ ] Axis FAST 与 direct-Orbax 通过 key、action 和 rollout parity；
- [ ] `n_action_steps=8` 进入本地 protocol hash；
- [ ] 旧退化 checkpoint 不再可被自动选择；
- [ ] selector 允许 Axis init 直接胜出；
- [ ] 7/32D loss contribution 和 temporal padding mass 可见；
- [ ] action expert 与 2B PaliGemma LoRA leaf 映射准确；
- [ ] 所有新 arm 有 step-0、短 checkpoint 和 teacher drift；
- [ ] 至少两个 seed，且不是只看训练 loss；
- [ ] 每种 mixture 都有同 norm Axis control；
- [ ] Track 1/2 候选通过 paired baseline noninferiority；
- [ ] reverse proxy 未混入训练且标记非官方；
- [ ] 没有候选通过时自动回退 Axis baseline；
- [ ] FAST 包实际加载 `model_fast.safetensors`，视觉 key 完整；
- [ ] 本地结果与官方结果在报告中严格区分。

## 13. 本计划不授权的事项

- 不再次盲跑当前 F0/M1 到 3000 steps；
- 不把现有 T3@3000 作为提交 winner；
- 不仅凭 FAST 文件存在或 server 200 判断模型正确；
- 不把 SLOW warning 直接写成唯一根因；
- 不使用训练 loss 选模型；
- 不把 Axis 仅作为不可胜出的参考线；
- 不训练公开 T3 example/evaluation tasks；
- 不合成反向 demonstrations、反转 actions 或自动生成隐藏任务近似数据；
- 不推测未公开的 Track 3 具体任务或数量；
- 不使用比赛 GPU，除非项目负责人后续明确授权。
