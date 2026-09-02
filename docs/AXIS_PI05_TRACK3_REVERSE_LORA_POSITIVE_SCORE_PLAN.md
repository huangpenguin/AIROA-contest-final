# AXIS Pi0.5 Track 3 Reverse LoRA 正分优先计划

> 制定日期：2026-09-02（Asia/Tokyo）
> 执行输入：root `88796b2`，nested OpenPI `b58aff0`
> 当前状态：`track3_reverse_t0_cpu_locked`
> 资源边界：GPU1+2用于已验证的两卡LoRA；GPU3用于rollout/FAST诊断；GPU0及外部PID绝对不触碰
> 目标：基于官方已公开的“现有LIBERO任务的逆向操作”规则，为Track3构建通用reverse supervision，用最小LoRA适配获得可提交的Track3正分候选，同时保留Track1/2能力。

## 0. 当前事实与优先级

- 用户报告R6 FAST正式分数：Track1 `0.47`、Track2 `0.33`、Track3 `0`。
- 唯一已验证部署锚点是R6 raw step500及其folded/FAST链；不得改写或覆盖原checkpoint和正分ZIP。
- R6配置为action-expert LoRA rank `32`，PaliGemma LoRA关闭，action in/out projections与time MLP随L2 arm训练；FP32、valid7 temporal、LR `1e-6`、clip `0.1`、两卡global batch `64`。
- `lerobot/libero_plus`已本地落盘，metadata为20 FPS、14,347 episodes；它可作20Hz retention/扰动评估，但不是reverse supervision，不能代替逆向数据。
- dense interstep L0/L1成果已冻结；本计划不启动或修改dense full-FT。

## T0 执行记录（待审核）

T0 已在 root `61d4a9f`、nested OpenPI `b58aff0` 上完成 CPU-only 锁定。GPU1–3 idle gate 通过，GPU0 仍由外部 PID 占用但只做只读观察；没有启动 CUDA context、simulator、训练或 reverse 数据生成。

- 证据：[t0_cpu_lock.json](../data/reports/axis_libero_30k/track3_reverse_lora/t0_cpu_lock.json)；报告：[T0_CPU_LOCK_REPORT_ZH.md](../data/reports/axis_libero_30k/track3_reverse_lora/T0_CPU_LOCK_REPORT_ZH.md)
- R6 raw/folded/FAST/正分 ZIP 与 fresh QUANTILES norm hash 已锁定；正分 ZIP 未修改。
- evaluation-only denylist 与 canonical training manifest 的 exact overlap 为 `0`；不输出其中的 task identifiers。
- 下一步仅为 T1 generic reverse compiler/pilot；未达到 30–60 条真实 20Hz、3/3 replay accepted episode 门禁前，不启动 LoRA。

## 1. 公平性与数据红线

只允许使用：

1. 主办方公开的高层事实：Track3是现有LIBERO操作的逆向任务；
2. 公开LIBERO/LIBERO-plus任务、环境、demonstrations和通用机器人学知识；
3. 本计划自行构造、执行和验证的generic reverse tasks。

禁止：

- 从`check_no_leaks.py`或其他泄漏内容提取内部family、hidden task、数量、命名、phase或生成机制；
- 以官方隐藏Track3 BDDL/task作为训练、选模或反复调参数据；
- 把公开example/evaluation task加入训练；它们只进入带来源时间戳的denylist和一次性最终审计；
- 把正向frames倒序、动作取反后未经环境执行验证就称为reverse demonstration；
- 猜测官方具体任务列表或宣称本地proxy等同官方Track3。

若比赛规则明确禁止合成/自采数据，本计划在T0立即停止，只保留local reverse evaluator。

## T0. 资产、分数和资源锁（CPU only）

1. 锁定R6 raw step500 tree hash、fresh QUANTILES norm hash、folded model hash、FAST权重hash及现有正分ZIP SHA-256；
2. Track1/2/3分数均标记`user_reported`，没有官方receipt时不得补造submission ID或episode细节；
3. 锁定canonical 10Hz与LIBERO-plus 20Hz数据revision、payload manifest、gripper/action contract；不得篡改source FPS；
4. 生成规则声明`public_reverse_rule_only=true`和evaluation-only denylist；扫描训练manifest，denylist命中必须为0；
5. 只读确认GPU1–3自然空闲。GPU0无论是否随后释放，本计划都不得使用；
6. root/nested先形成clean implementation commit。T0不启动GPU或生成训练数据。

## T1. Generic reverse task compiler 与 pilot

### T1.1 通用operator families

从公开LIBERO任务中只选择goal/init都能被环境predicate明确验证的通用逆操作：

- `containment_remove_restore`：从容器/抽屉/微波炉中取出并放回明确source region；
- `support_remove_restore`：从plate/stove/rack/cabinet top移走并放回明确source support；
- `articulation_inverse`：open↔close；
- `activation_inverse`：turn_on↔turn_off；
- `sequence_reverse`：仅由已经单独验证的inverse operators组成的两步序列。

“把A放入B”的逆向是改变initial state与goal后把A从B取回/恢复，不是交换A/B文字角色。source region或inverse predicate无法稳定表达的模板直接排除。

### T1.2 Reverse task contract

每个task生成：base public task、reverse instruction、reverse initial predicates、reverse goal predicates、invariants、operator family、scene/template/composition ID、seed和provenance hash。必须满足：

- reset后reverse initial为真、reverse goal为假；
- 成功终态reverse goal为真且invariants成立；
- instruction、predicate与视频人工检查一致；
- 同一base template只属于train或holdout一侧，禁止跨split；
- evaluation denylist与训练template/composition零重合。

### T1.3 真实20Hz reverse demonstration

reverse init优先来自公开正向成功终态的稳定simulator snapshot。动作可由teleop、脚本、motion primitive或正向轨迹的逆序proposal辅助生成；但accepted数据必须是从reverse init在`control_freq=20`真实执行后重新采集的front/wrist/state/action序列。不能复用倒放图像。

任何逆序/取反action只允许作为proposal，并且必须：

1. 在环境中实际执行到reverse goal；
2. action finite、7D OSC_POSE与gripper contract正确；
3. 从保存的reverse init进行3/3 replay成功；
4. 无goal-at-reset、teleport、state overwrite、严重collision或人工修剪隐藏失败。

Pilot最低门禁：`30–60` accepted episodes、至少3个operator families、6个base templates、2种scene、每family至少2个template。未达到就停止，不训练LoRA。

## T2. 数据schema、split、sampling与norm

1. 转成独立`track3_reverse_pilot_v1` LeRobot/OpenPI unified schema；保留真实20 FPS，gripper统一为`{-1,+1}`，action为7D OSC_POSE；
2. split优先按base template：train/validation/template-holdout；至少20% templates完全holdout；
3. source sampling先锁为`reverse 10% + canonical LIBERO replay 90%`，按source→operator→template→episode分层，报告实际episode/frame exposure；canonical仍标记10Hz，不伪造成20Hz；
4. 主实验继续使用R6 fresh standard QUANTILES norm，确保step0策略不因换norm而改变；同时计算mixture norm但只作N1消融；
5. mixture norm只有在R6冻结step0的standard C5、reverse calibration和action parity均非劣时才可进入独立arm，不能与scope/mix/LR同时改变；
6. LIBERO-plus 20Hz只用于retention/扰动评估；除非另行审核，不混入首轮训练。

## T3. 冻结基线与失败分段

使用GPU3对未训练R6运行：

- reverse template-holdout：至少20 episodes，覆盖全部pilot operator families；
- standard retention C5/C20的既有固定manifest；
- 可用的公开reverse proxy只运行一次并标记`official_result=false`。

除binary success外记录：接近对象、正确对象接触、抓取、从source取出、到达restore region、稳定释放、inverse fixture state、多步subgoal数，以及wrong-object/wrong-target/drop/collision/stall/timeout。该结果决定后续是否需要语言LoRA；不得因为总success为0就直接解冻PaliGemma。

## T4. 两卡LoRA successive-halving

### T4.1 初始化与共同配方

- 所有arm从R6 raw step500 params分叉，保留其已学LIBERO能力；不从Axis init重来；
- fresh optimizer/scheduler，internal adaptation step从0计；原R6 optimizer state不续用；
- GPU1+2、FP32、microbatch/device `1`、accumulation `32`、effective global batch `64`；
- valid7 temporal、AdamW原参数、clip `0.1`、EMA关闭；
- primary norm为R6 fresh standard QUANTILES；
- checkpoints `0/1/25/100/250/500`，每阶段fresh-process restore、finite和scope audit。

### T4.2 首轮最多两个arm

| Arm | 从R6继续训练的参数 | Reverse mass | LR | 目的 |
|---|---|---:|---:|---|
| `TR1` | action-expert LoRA only；heads/time冻结在R6值 | 10% | `3e-7` | 最保守reverse行为适配 |
| `TR2` | action-expert LoRA + action in/out + time MLP | 10% | `3e-7` | 检查action distribution/flow重新对齐 |

两arm按`25→100`运行。只有finite、restore和sampling exposure全部通过才评估。GPU3可在一个训练arm结束后做rollout；不得让训练/evaluator同时争用同一GPU或写同一输出目录。

100-step晋级要求：

- reverse template-holdout至少2个成功episode且来自至少2个held-out templates，或相对R6在预注册subgoal score有明确paired提升；
- standard C5成功数不低于R6的`2/3` source-success复现门禁，collision/stall/timeout无灾难性上升；
- action finite、gripper sign正常、无runtime/infrastructure failure；
- paired结果不能只来自一个train-like template。

最多一个arm进入`250→500`。如果TR1/TR2都稳定但完全不学习，只允许一次单变量补充：优先把winner LR改为`1e-6`；只有证据显示reverse样本曝光不足时才改为20% reverse mass。不能同时改LR与mass。

### T4.3 条件语言LoRA

仅当失败分段显示大多数episode在接触前就持续wrong-object/wrong-target/direction错误，而抓取控制本身正常，才允许从R6加入zero-delta PaliGemma末层LoRA的小型`TRL`诊断。必须另报rank/scope和step0 exact parity；不得解冻vision或全PaliGemma。若失败主要是抓取/放置/碰撞，则不启动语言LoRA。

## T5. 选模门禁

500-step候选必须同时满足：

1. reverse template-holdout相对R6成功率和subgoal score均提升，成功覆盖至少2个operator families；
2. standard C20 `execute8` paired delta不低于`-0.05`，timeout/collision/stall各不恶化超过`0.05`；
3. LIBERO-plus小型分扰动面板无灾难性回归；
4. raw restore、folded dense、LeRobot processor和FAST action parity通过既有部署容差；
5. 结果不是train template memorization，所有validation/public proxy标签清晰；
6. failure gallery包含H.264、task/reverse spec、operator、扰动、seed、subgoal timeline、failure tag及R6/candidate paired对照。

如果所有arm仍为0且subgoal无改善，输出`track3_reverse_lora_no_capability_gain`并停止；不得盲目跑3K。若有局部成功但样本不足，先扩充跨template reverse数据，再决定1K。

## T6. FAST与Track3候选ZIP

只有T5通过才允许：

1. fold winner并导出`model_fast.safetensors`；ZIP不得包含slow `model.safetensors`；
2. 固定16-case source↔FAST parity、`/act`至少12请求且每次`<10s`；
3. 本地20Hz reverse positive panel至少复现既有成功，standard C5至少复现`2/3`；
4. 静态layout、CRC、SHA-256、依赖、health/reset/act和ZIP E2E门禁通过；
5. 生成独立Track3候选ZIP与package validation，不覆盖Track1/2正分ZIP。

完成后hard stop，等待人工审核和用户手动提交；不得自动上传或提交比赛。

## 2. 资源替代与恢复原则

- GPU0一直被占：继续本计划，不影响GPU1+2训练/GPU3评估。
- GPU0自然释放：本计划期间仍不自动切回dense；先完成当前Track3 hard gate，之后另开interstep恢复计划。
- 三卡dense：必须作为新global-batch/sharding实验另审，不属于无损fallback。
- 比赛96GB Blackwell：保留给通过本地短门禁后的长训练；不为尚未验证的数据生成或LoRA arm提前计时。

## 3. 最终状态

只允许以下最终状态：

- `track3_reverse_rules_or_pilot_blocked`
- `track3_reverse_pilot_ready_pending_review`
- `track3_reverse_lora_stopped_at_100`
- `track3_reverse_lora_no_capability_gain`
- `track3_reverse_lora_candidate_500_pending_review`
- `track3_reverse_fast_zip_pending_manual_submission`

每个阶段结束都必须更新summary、计划索引、项目handoff、测试和clean commit，并hard stop等待审核。
