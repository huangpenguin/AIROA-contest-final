# Track 3 逆向任务数据合成与适配研究计划

> 状态：调查/设计计划；本文不授权立即生成数据或启动训练
> 日期：2026-08-28（Asia/Tokyo）
> 公平性边界：只依据官方已公开的“Track 3 是现有 LIBERO 任务的逆向操作”设计；不使用 `check_no_leaks.py` 推测未公开的具体 task 或数量
> 模型边界：数据专项从稳定的 Axis→标准 LIBERO checkpoint 分叉，不从当前退化的 1k/3k mixture checkpoint 续训
> 硬件边界：只使用自有设备，暂不使用比赛 GPU

## 0. 目标与非目标

目标是建立一个可审计的“正向任务 → 逆向 task specification → 真实环境执行 → 高质量 demonstration → held-out reverse evaluation”管线，用来研究 Axis→LIBERO 模型对 Track 3 逆向操作的适应能力。

本计划不做以下事情：

- 不猜测官方隐藏 task 清单、数量、对象或 init states；
- 不把公开 Track 3/example 评测 task 直接加入训练数据；
- 不把正向 demonstration 的 frames/actions 简单倒放当成逆向 demonstration；
- 不只改 instruction 文本而不改环境初始状态和 goal predicate；
- 不在 Axis→标准 LIBERO canonical 30k 主干中混入合成数据；
- 不用 synthetic validation 上的高分代替本地 public proxy 或官方评分。

## 1. 项目负责人首先调查清单

在任何代码或数据写入前，请优先调查以下问题。每项应留下链接、截图/邮件或本地代码位置，不只记录口头结论。

### 1.1 比赛规则与数据权利

1. 比赛是否允许使用自建 simulator demonstrations、人工遥操数据和规则生成数据？
2. 是否允许使用 simulator 的 privileged state/object pose 生成训练数据？
3. 对公开 LIBERO BDDL、assets、demonstrations 的衍生数据是否有许可证、归属和分发限制？
4. 可否使用主办方公开 example tasks 做 evaluation？若可以，它们是否必须严格禁止进入 train/tuning？
5. 请主办方确认：仅基于已公开“逆向操作”类别构造通用 reverse tasks，是否符合当前公平性规则。

若上述任一项没有明确答案，可继续设计 local reverse evaluator，但不生成或训练 demonstrations。

### 1.2 LIBERO 环境能力

1. 自定义 BDDL/task 的注册入口在哪里？是否需要重建 benchmark registry？
2. 现有 predicate 是否完整支持 `inside/on/open/close/turn_on/turn_off` 及对应否定/逆向语义？
3. 能否安全序列化/恢复 MuJoCo simulator state、object poses、joint state 和 controller state？
4. 环境是否可从正向 successful terminal state 建立新 init state，且 reset 后稳定复现？
5. 能否读取 object/receptacle pose、EEF pose、contact 和非目标对象位移，用于 oracle 和质量门禁？
6. 环境成功判定是 goal predicate 即时满足，还是需要连续稳定多帧？
7. 哪些公开任务在物理上可逆，哪些包含不可逆状态变化或缺少可定义的“原位”？

### 1.3 Demonstration 采集能力

1. 是否已有 SpaceMouse/键盘遥操工具，能否记录符合 20 Hz、8D state、7D delta action、front+wrist 契约的轨迹？
2. 是否有可复用的 pick/place/open/close/toggle 运动原语或 privileged oracle？
3. 单个 accepted episode 的人工成本、采集成功率和预期每日产量是多少？
4. 可否对同一 init state 自动回放 action 三次，并验证三次都达成环境 goal？
5. 是否能对每个失败保存 H.264、完整 task spec、seed、失败阶段和 contact/action 曲线？

### 1.4 调查产出

调查完成后建议落盘：

```text
docs/TRACK3_SYNTHESIS_INVESTIGATION.md
data/reports/track3_synthesis_v1/rules_evidence.json
data/reports/track3_synthesis_v1/environment_capabilities.json
data/reports/track3_synthesis_v1/teleop_oracle_benchmark.json
```

`rules_evidence.json` 的结论必须是 `allowed / prohibited / unresolved`；`unresolved` 不能默认当作 allowed。

## 2. 逆向任务的形式化定义

对一个正向 task，分开记录：

```text
ForwardTask = (scene, initial predicates I_f, goal predicates G_f, instruction L_f)
ReverseTask = (same scene family, reverse initial predicates I_r,
               reverse goal predicates G_r, instruction L_r)
```

最基本的约束是：

- `I_r` 应满足正向的 `G_f`，但不应在 reset 后已满足 `G_r`；
- `G_r` 应恢复正向初始状态的**语义条件**，而不是要求连续位姿逐 bit 相等；
- 非目标对象、容器、家具和关节的必要不变式必须显式记录；
- 多步任务不是只把句子词序颠倒，而是对 task graph 按逆序应用各 operator 的 inverse；
- 若无法稳定定义 `G_r`，该 task 标记 `not_safely_reversible`，不为追求数量强行纳入。

### 2.1 通用 reverse operator 分类

| Operator family | 正向示例 | 逆向语义 | 主要风险 |
|---|---|---|---|
| `containment_remove` | 把物体放入容器 | 从容器取出并放到定义的来源区 | 来源区需有语义定义 |
| `support_remove_restore` | 把物体放到支撑面/物体上 | 取下并恢复到来源区 | 原始位置不能是隐式连续坐标 |
| `open_close` | 打开抽屉/柜门 | 关闭到阈值范围 | 关节阈值、手指撤离 |
| `close_open` | 关闭抽屉/柜门 | 打开到阈值范围 | 初始闭合稳定性 |
| `toggle_off` | 开启炉具/设备 | 关闭 | actuator/contact 方向 |
| `toggle_on` | 关闭设备 | 开启 | 指示状态是否可观察 |
| `spatial_restore` | 将物体移至目标区 | 移回另一语义区域 | region 定义与容差 |
| `sequence_inverse` | A 后 B 的多步操作 | inverse(B) 后 inverse(A) | 中间状态必须可达 |

这是通用 operator taxonomy，不是对官方隐藏 Track 3 的任务清单猜测。

### 2.2 不能简单倒放 action

LIBERO 的 7D action 包含局部增量位移/旋转和夹爪命令。正向 action 倒序加取负通常会失败，原因包括：

- 物体与夹爪的接触和摩擦动力学不可逆；
- 夹爪在正向放下后已经失去物体约束；
- controller 内部状态、数值积分和碰撞不会随 frames 倒放自动恢复；
- 视频倒放只是外观上连续，不代表 action 在新 init state 可执行。

倒放轨迹最多用于生成人工遥操的 waypoint 参考或视觉审阅，不能直接进入训练集。

## 3. 数据合成管线

### Stage R0：公开正向 task 盘点

从标准 LIBERO 公开 tasks 提取：

- task name、BDDL hash、scene/template、objects、receptacles、fixtures；
- `I_f`、`G_f`、instruction 和 goal predicate graph；
- 是否有语义明确的来源 region；
- 所需 reverse operator、预计步数、运动原语和失败模式；
- `reversible / reversible_with_added_source_region / not_safely_reversible` 结论。

盘点时不读取内部检查脚本来挑选任务。先按公开标准 LIBERO 全体 task 做通用分类，再冻结 train/holdout template split。

### Stage R1：reverse task compiler

为每个可逆 task 生成显式 `reverse_task_spec.json`：

```json
{
  "base_task_id": "public_standard_task_id",
  "direction": "reverse",
  "reverse_operator": ["containment_remove", "spatial_restore"],
  "forward_bddl_sha256": "...",
  "reverse_initial_predicates": [],
  "reverse_goal_predicates": [],
  "invariants": [],
  "instruction_canonical": "...",
  "source_region_definition": {},
  "generator_eligibility": "pilot",
  "official_track3_claim": false
}
```

Compiler 必须检查：

- reverse reset 后 `I_r=true` 且 `G_r=false`；
- 静置若干步后物体不飘移/掉落；
- goal predicate 对真实物理状态生效；
- instruction 中对象、容器、方向与 BDDL 完全一致；
- 不以精确 XYZ 复位作为唯一成功条件，除非环境本身有公开语义 region。

### Stage R2：reverse initial state 生成

按可靠性优先级采用两种方法：

1. **Forward-terminal snapshot**：在公开正向 demonstration 成功终态上稳定运行数帧，保存完整 simulator/controller state，作为 reverse init。
2. **Predicate-constrained placement**：在允许 privileged state 的前提下，按 `I_r` 放置物体/关节，然后经过物理 settle 和连续 predicate 验证。

不能只保存 RGB frames 或部分 qpos就声称恢复了终态。需要记录 MuJoCo state、robot/controller state、object/joint state、asset/scene revision 和 settle procedure。

### Stage R3：demonstration 生成

按优先级选择：

1. **人工遥操 gold set**：用于验证 task 可解、建立高质量少量样本和 oracle 参考。
2. **Privileged motion-primitive oracle**：使用 object/EEF pose 执行 approach、grasp、lift、transport、place、retract、open/close/toggle；只将实际 simulator 执行过的 7D action 记入数据。
3. **Policy-assisted collection**：用 Axis→LIBERO policy 建议 action/waypoint，人类或 oracle 接管失败步骤。必须显式标记 takeover 区间。
4. **Search/retry augmentation**：在同 task spec 中扰动 grasp pose、waypoint 和速度生成多样成功轨迹，不允许更改 goal 来将失败变成成功。

每条轨迹必须标记 `human / oracle / assisted / retry_variant`，后续抽样时可分层和消融。

### Stage R4：质量门禁

只有同时通过以下检查的 episode 才能进入 `accepted` manifest：

1. Reset 后 reverse goal 未满足，reverse initial predicates 满足。
2. 轨迹结束后 goal 连续满足指定稳定帧数，机器人手指已撤离需要撤离的对象/关节。
3. 从相同 init-state snapshot 完整 action replay 至少 3/3 成功。
4. 所有 state/action 为 finite，shape、dtype、20 Hz、front/wrist 对齐和 episode boundary 符合 unified schema。
5. 非目标对象位移、碰撞、夹爪挤压和 action saturation 进入报告；超阈值的 episode 不自动删除，先进 manual review。
6. 轨迹起点/终点 predicate、instruction 和 reverse task spec 三者一致。
7. 视频可用 ffprobe 验证为 H.264/yuv420p/20 FPS；所有失败保存视频，每个 stratum 至少保存成功参考。
8. 没有 train/validation/test 的 base task/template/init-state 泄漏。

质量状态只使用：

```text
generated → replay_verified → manually_reviewed → accepted
                                  ↳ rejected_with_reason
```

禁止将“oracle 返回了 finite action”或“最后一帧看起来对”当作 accepted。

### Stage R5：LeRobot/OpenPI unified schema

每条 episode 除当前 front/wrist/state/action/task 字段外，追加：

```text
dataset_version
episode_id
base_task_id
base_task_template_id
direction=reverse
reverse_operator[]
forward_bddl_sha256
reverse_spec_sha256
initial_state_sha256
goal_predicates
invariants
instruction_canonical
instruction_variant_id
generator_type
generator_version
human_takeover_ranges
environment_revision
asset_revision
env_seed
generator_seed
replay_successes / replay_attempts
quality_flags[]
license_and_rule_evidence_id
```

大文件建议路径：

```text
data/synthetic/track3_reverse_v1/raw/
data/synthetic/track3_reverse_v1/lerobot/
data/manifests/track3_reverse_v1/
data/reports/track3_synthesis_v1/
artifacts/track3_synthesis_v1/failure_media/
```

原始视频、frames、sim states 和 LeRobot shards 不进 Git；Git 只保留 spec、config、manifest、hash、报告和小型 gallery 索引。

## 4. 去泄漏和分层分割

### 4.1 先分 template，后生成轨迹

不允许先生成所有轨迹再随机按 episode 分割，因为同一 base task 的正/逆轨迹会泄漏几乎相同的 scene 和 goal。

固定分割层级：

```text
base task template
  → object/receptacle composition
    → reverse operator sequence
      → init-state family
        → episode
```

同一 base task template 的所有正向/逆向轨迹只能在同一 split。

### 4.2 建议三类 holdout

1. `composition_holdout`：训练见过 operator，但没见过该 object×receptacle 组合。
2. `template_holdout`：同 operator，整个 base task template 未见。
3. `sequence_holdout`：见过单步 inverse operators，未见过该多步组合/顺序。

最终 Track 3 适配选模不能使用 train-template random episode validation；必须以至少 `template_holdout` 为主指标。

### 4.3 公开评测 task 隔离

- 现有 `libero_t3`/public Track 3/example cases 如已用于评估，就全部加入 evaluation-only denylist。
- 与 denylist task 共用完全相同 BDDL/template 的 synthetic reverse task 默认也排除。
- 可使用相同的通用 inverse operator，但不使用具体对象、容器、scene 与 goal 组合进行近似复制。
- 所有 denylist 条目保存来源与加入时间，不在看到分数后删除或替换。

## 5. 规模与采集节奏

### 5.1 Pilot：验证“能生成”，不训长模型

目标：30–60 条 accepted episodes，覆盖至少 3 个 reverse operator families、2 种 scene 类型、每 family 至少 2 个 base templates。

Pilot 验收：

- task reset/goal semantics 100% 通过；
- accepted episode 3/3 action replay 成功；
- 人工检查视频与 predicate 一致；
- 不存在 split/denylist 泄漏；
- schema 可被现有 OpenPI loader 读取，但仅做 1-step/10-step loader smoke，不因 pilot 数量少就启动 3k 训练。

### 5.2 Dataset v1：广覆盖优先于单 task 堆量

建议目标：300–600 条 accepted episodes。

- 至少 5 个 reverse operator families；
- 任一 operator family 不超过 accepted episodes 的 25%；
- 任一 base task template 不超过 10%；
- 每个 template 覆盖多个 init states 和至少两种成功轨迹风格/路径；
- 人工 gold set 不必数量最多，但每个 operator family 至少有可视审核的 gold 参考；
- generator failure 也保留到 failure corpus，用于修工具和定义难度，不进 imitation train。

不设定“必须凑到 600”的硬指标。如果只有 180 条但覆盖广、回放稳定、template holdout 严格，比 600 条同一 task 重复轨迹更有研究价值。

## 6. 分层抽样与 normalization

### 6.1 逆向数据内部分层

抽样层级：

```text
reverse_operator_family
  → base_task_template
    → object×receptacle/fixture composition
      → generator_type
        → trajectory-length bin
          → episode
```

优先平衡 operator 和 template，不按 frame 数让长轨迹自然占据大部分 batch。同时记录真实的 frame exposure，避免 sampler 报告只在 episode 层看起来平衡。

### 6.2 与标准 LIBERO replay 的 mixture

Track 3 专项从稳定 Axis→LIBERO checkpoint 分叉，第一轮只比较：

| Mix | Synthetic reverse | Standard LIBERO replay | 用途 |
|---|---:|---:|---|
| `R0` | 0% | 100% | 不适配 control |
| `R05` | 5% | 95% | 保守适配 |
| `R10` | 10% | 90% | 主候选 |
| `R20` | 20% | 80% | 强适配/遗忘对照 |

不首轮测试 50% 或 synthetic-only；当前 50% Plus mixture 已经证明有严重遗忘风险。

### 6.3 Norm 决策门禁

逆向轨迹的 action distribution 可能与正向显著不同，但换 norm 本身也会改变 step-0 策略。因此对每个 mix：

1. 计算 candidate mixture norm，但不立即启用；
2. 用冻结 base checkpoint 分别加载 standard-LIBERO norm 与 mixture norm，跑固定 standard/reverse calibration cases；
3. 如 mixture norm 在训练前已造成标准 LIBERO 明显回归，第一轮锁定 standard norm；
4. 若 mixture norm 通过 step-0 非劣门禁，将 norm 作为单独消融，不和 sampling/scope/LR 同时改变。

## 7. 与 Axis 模型的适配实验

### 7.1 启动前提

不在 Axis→LIBERO 主干还处于崩塌时启动 synthetic adaptation。最少需要：

- canonical run 至少到 5k，标准 LIBERO 能力对 step-0 有稳定正向改善；
- 选定 checkpoint 的 raw/EMA params、norm、evaluator 和 FAST 语义清楚；
- reverse dataset pilot/v1 通过 rules、schema、replay、split 和 manual review 门禁；
- 冻结 standard retention、Plus 和 reverse template-holdout 评估 manifest。

### 7.2 参数作用域

合成数据量远小于标准 LIBERO，因此首轮不全量解冻 vision/PaliGemma。分阶段比较：

| Scope | Trainable | Frozen | 解答的问题 |
|---|---|---|---|
| `S0` | 无 | 全部 | reverse zero-shot control |
| `S1` | action expert LoRA | vision、PaliGemma、projections/time | 最保守动作适配 |
| `S2` | `S1` + action in/out projections | vision、PaliGemma、time MLP | 是否需重新对齐逆向 action distribution |
| `S3` | `S2` + time MLP | vision、PaliGemma | 时间/流匹配适配作用 |
| `S4` | action expert dense + projections + time MLP | vision + PaliGemma | 非 LoRA expert-only control |
| `S5` | `S3` + PaliGemma 最后少量 LoRA | vision + 其余 PaliGemma | 只在证据指向逆向语言理解不足时 |

LoRA 为 zero-delta 初始化；每个 scope 先做 step-0 exact-action parity。`S5` 不能因为 reverse success 低就自动启动：必须先从 failure review 证明是对象/目标/方向语言误解，而不是抓取控制失败。

### 7.3 保守训练预算

第一轮不从 3k 起跑，而是 successive halving：

```text
0 → 25 → 100 → 250 → 500 → 1,000 → optional 3,000
```

- 首轮 LR 建议候选 `3e-7` 和 `1e-6`；`3e-6` 仅在前两者无学习且不崩塌时追加。
- Effective global batch 至少 24，并对 sampling source mass 做 actual-loader audit。
- Gradient clip 0.1/1.0 作为单独消融，记录 pre/post clip norm。
- 标准 LIBERO replay 承担能力保留；可加入冻结 base teacher action/velocity anchor，但它是独立 objective ablation。
- 一旦标准 LIBERO paired success 下降超过 0.05，或 timeout/collision 上升超过 0.05，停止该 arm。
- 只有 1k 上 reverse template-holdout 有稳定收益且 standard/Plus 通过非劣的一个 arm 可进入 3k。

### 7.4 最小实验矩阵

不做 scope×mix×LR×norm 全组合网格。第一轮只做：

| Arm | Scope | Mix | LR | Norm | 用途 |
|---|---|---|---:|---|---|
| `T0` | S0 | R0 | 0 | standard | base control |
| `T1` | S1 | R05 | 3e-7 | standard | 最保守 |
| `T2` | S1 | R10 | 3e-7 | standard | 主候选 |
| `T3` | S2 | R10 | 3e-7 | standard | projection 作用 |
| `T4` | S3 | R10 | 3e-7 | standard | time MLP 作用 |
| `T5` | S1 | R20 | 3e-7 | standard | 强适配/遗忘对照 |
| `T6` | S1 | R10 | 1e-6 | standard | LR 消融 |

在 100 steps 最多保留 4 个，250 steps 最多 3 个，500 steps 最多 2 个，1k 最多 1 个进入 3k。如果 T0 在 reverse holdout 上已优于所有适配 arm，结论是“当前合成数据/适配无收益”，不强制选一个已训模型。

## 8. 评估与选模

### 8.1 四个独立面板

1. `reverse_template_holdout`：合成逆向泛化主面板。
2. `reverse_public_proxy`：本地公开 reverse task proxy，始终标记 non-official。
3. `standard_libero_retention`：四套件正向能力。
4. `libero_plus_generalization`：扰动下的能力保留。

合成 validation 与 public proxy 必须分开。在 synthetic validation 上选超参，在 public proxy 上只做冻结的最终审计，避免反复看 public task 过拟合。

### 8.2 Reverse 阶段指标

不只看 binary success，记录：

- instruction 对象/目标解析是否正确；
- 是否靠近目标对象；
- 是否完成抓取/接触；
- 是否从正向目标容器/支撑面成功移出；
- 是否达到 reverse target region；
- 是否稳定放置/关闭/开启并撤离；
- 多步 inverse sequence 完成了几个 subgoals；
- collision、stall、drop、wrong-object、wrong-target 和 timeout。

这些阶段必须来自环境 predicate/状态或一致的人工复核，不靠视频文件名猜测。

### 8.3 晋级规则

候选只有同时满足才可晋级：

- schema/runtime/coverage/finite 100% 通过；
- reverse template-holdout 相对 base checkpoint 有正向 paired gain，且置信区间与样本量一起报告；
- standard LIBERO paired delta 下界不低于预声明非劣 margin；
- Plus 最差扰动类别没有灾难性回归；
- collision、stall 和 timeout 均不超过基线容差；
- 收益不只来自一个 train-like template 或一种 operator。

人工 override 必须保存自动结论、覆盖理由和风险，不得覆盖原 decision file。

## 9. Failure review 产品

每个失败 episode 生成组合页：

- 可播放 H.264，front/wrist 并排；
- base forward task 与 reverse task spec 对照；
- canonical instruction 和 paraphrase；
- `I_r`、`G_r`、不变式与每个 subgoal 的时间线；
- object/EEF 轨迹、gripper、action norm、contact、collision；
- operator family、generator type、scene、object/receptacle、init-state 和 seed；
- generator failure 还是 policy failure；
- 失败 tag 与人工复核意见；
- 与 base checkpoint/当前候选的 paired 视频。

Gallery 首页按以下分组：

```text
base success → adapted failure
base failure → adapted success
both fail, different stage
operator-specific failures
template-holdout failures
collision / stall / wrong-object / wrong-target
```

这是后续扩数据或改 task compiler 的主依据，不按 training loss 盲目加量。

## 10. 实施阶段和停止条件

### Phase A：规则/环境调查

- 完成第 1 节四份调查产出；
- 确认合成数据可用性和权利边界；
- 选出可安全逆向化的 operator families。

**停止：** 规则未解或环境无法恢复/验证 reverse init/goal。

### Phase B：task compiler + 单任务 gold demo

- 每个 operator 至少一个 task spec；
- 人工遥操一条 gold trajectory；
- 3/3 replay 和静置稳定性通过；
- 产出失败视频与 task 详情页。

**停止：** 只能通过修改 goal 或宽松 predicate 才让 demo 成功。

### Phase C：30–60 episode pilot

- 冻结 template split/denylist；
- 至少 3 operator families；
- schema、replay、manual review 和 OpenPI loader smoke 通过；
- 用 Axis→LIBERO checkpoint 做 zero-shot reverse baseline，不做长训练。

**停止：** replay 稳定性低、数据高度单一或 holdout 已泄漏。

### Phase D：300–600 episode v1

- 广覆盖 operator/template/composition；
- 生成 accepted/rejected/manually-reviewed manifests；
- 冻结 R0/R05/R10/R20 sampling manifests 和 norm controls；
- 先执行 0/25/100-step smoke。

**停止：** 为凑数量重复单一 task，或 synthetic validation 与 public proxy 无法分离。

### Phase E：保守适配

- 100/250/500/1k successive halving；
- standard/Plus/reverse 三类非劣与增益门禁；
- 最多一个候选进入 3k；
- 全程保留 failure gallery 和 paired curves。

**停止：** 逆向收益无法跨 template 复现，或标准 LIBERO/Plus 回归超过容差。

## 11. 验收清单

- [ ] 规则允许性有可引用证据，未解项不被默认为允许；
- [ ] 不使用泄漏脚本推测具体 hidden task/数量；
- [ ] reverse task 同时改变 init state、goal predicate 和 instruction；
- [ ] 没有直接倒放 actions 进入 accepted data；
- [ ] 每个 task reset 时 goal=false，轨迹结束稳定 goal=true；
- [ ] accepted episode 均通过 3/3 replay；
- [ ] 20 Hz、8D state、7D action、front/wrist 与 episode boundary 通过；
- [ ] base template 级 train/validation/test 零泄漏；
- [ ] public Track 3/example task 在 denylist 且仅评估；
- [ ] 合成数据源、generator、task spec、sim state 和 license/rules provenance 可追溯；
- [ ] 每个 mixture 有 actual-loader source mass audit 和 step-0 norm control；
- [ ] 适配从稳定 Axis→LIBERO checkpoint 分叉，不污染 canonical 30k；
- [ ] 逆向收益与标准 LIBERO/Plus 能力保留同时报告；
- [ ] 失败 H.264 与 task/instruction/init-state/predicate/operator 详情可组合审阅；
- [ ] 所有本地结果标记 `official_result=false`。

## 12. 本计划不授权的事项

- 不立即生成 demonstrations 或启动 Track 3 训练；
- 不将公开评估 task 加入训练；
- 不根据 `check_no_leaks.py` 构造具体 hidden-task clone；
- 不将倒放视频/动作当作真实逆向轨迹；
- 不在规则未确认时将合成数据用于比赛模型；
- 不从当前退化 F0/M1@3000 续训；
- 不在 canonical Axis→LIBERO 30k 中混入 reverse data；
- 不使用比赛 GPU，除非项目负责人之后另行明确授权。

## 13. 与主干研究的关系

Axis 主干计划见 `docs/AXIS_PI05_STANDARD_LIBERO_30K_RESEARCH_PLAN.md`。推荐执行顺序：

```text
调查 Track 3 规则/环境能力（可与 Axis 30k 并行）
  → reverse task compiler + pilot data
  → Axis→LIBERO 主干至少通过 5k 行为门禁
  → 冻结 reverse holdout/public proxy
  → 从稳定 checkpoint 分叉保守适配
  → 标准 LIBERO + Plus + reverse 联合审核
```

调查和 task compiler 可以在 30k 训练期间并行；真正模型适配必须等待稳定 Axis→LIBERO checkpoint。
