# Track 3：运营策略输入复核与下一阶段执行草案

> 日期：2026-09-08（Asia/Tokyo）  
> 状态：`reviewed_draft_not_active`  
> 当前权威活动计划仍为：`docs/AXIS_PI05_TRACK3_SR_CLOSURE_AND_TASK10_USER_TELEOP_GOLD_CANARY_PLAN.md`  
> 本文不授权启动 physics、replay、训练、ZIP、上传或提交。当前 dirty/raw 批次完成独立闭包并同步生命周期后，才可把本文拆成唯一活动计划。  
> 目标：把运营方评分信号和上位者经验转成“可信 reverse 数据 → 非零 Track3 policy success → 可提交候选”的短关键路径。

## 1. 新输入与 provenance

本轮读取两份用户提供的 Slack/PDF 材料：

| 文件 | SHA-256 | 性质 |
|---|---|---|
| `slack/上位発表者Track1_yoshinari1209様.pdf` | `251f7698238d7c9388e7059a92bed5a4625b32334a4616803012acca133f3d6a` | 上位参与者本地验证与后续研究设想，不是官方规则 |
| `slack/松尾研PAIコンペ2026_キックオフ全体スライド.pdf` | `dfd187350f492002a70c4989776973d50090a92123dae4561903e9c382e0e77b` | 运营方技术说明、赛程与参与者 lightning talks；其中参与者经验不等于官方保证 |

运营方材料确认本选 1 截止时间为 **2026-09-17 23:59 JST**。若后续 Slack 公告修改日期，以最新官方公告为准，执行前必须重新核验。

## 2. 从材料中可以采用的结论

### 2.1 官方评分信号

1. 任务成功与 collision 是 gate；time、jerk、SPARC、trajectory length、EEF rotation 等连续指标在 gate 后参与加权归一化乘积。权重和归一化公式不公开。
2. Track 分数相加；未提交的 Track 按 0 计。运营 baseline 参考为 Track1=`0.286`、Track2=`0.165`、Track3=`0.000`。
3. 因为 collision 可让已经完成目标的 episode 整体失败，不能只优化最终 predicate；成功、无碰撞、步数和轨迹质量必须同一条 episode 上成立。
4. 运营材料列出的 Track2 变化包括 blur、camera、distractor、position 与 robot reset；单一固定场景成功不足以证明鲁棒性。

### 2.2 上位参与者证据的正确用法

1. 某参与者在其本地 Track1 协议中测得 OpenVLA-OFT+ `91/120`、Pi0.5 `86/120` clean success；这支持“先建设可信本地评测”，不能外推为本项目或 Track3 的模型排名。
2. 同一参与者对 Pi0.5 使用 LIBERO-Plus imitation data 继续 LoRA 训练 3000 steps，未见明确改善。该结果支持停止盲目追加通用 LIBERO-Plus LoRA；它不证明 reverse-specific data 无效。
3. 另一参与者在 9 月 7 日认为 Track3“一条成功可能显著改变排名”、数据制作是关键、LB 是唯一真实本番条件试跑。这里的排名判断是时间点相关经验，不是运营保证；应采用其“投资源到零成功任务、记录每次试跑”的决策原则。
4. “每日自动提交”是参与者工作流，不构成当前仓库的上传授权。我们采用每日候选冻结/打包节奏，正式上传和提交仍由用户完成。

## 3. 当前项目状态对新目标的约束

2026-09-08 本轮只读现场核验：

- root HEAD：`bb0793f5f8558115d2fc20b07044c252d2c50efa`；branch 相对 upstream ahead 7。
- root dirty：Teleop 工具约 900 行修改、两份测试、SR runtime pointer/log/PID、`MUJOCO_LOG.TXT`、`slack/`。这些内容不能被新计划静默提交或覆盖。
- machine handoff 仍为 `track3_task10_user_teleop_ready_for_user_calibration`，已落后于实际录制状态。
- 没有 live Track3/Teleop/MuJoCo worker。用户的 localhost-only TurboVNC `:3` 会话仍存在。
- GPU0–2 正由外部进程占用；GPU3 为 5 MiB/0% 且无 compute process。新计划不得把外部作业当作可回收资源。
- raw 目录已有两条 `official_t3_red_mug` pending episode，以及多条 red-mug/yellow-book rejected/diagnostic episode；没有 accepted、训练或新 Track3 ZIP。
- 较短 pending red-mug 仍有 388 steps，目标约在 step 365 后成立，超过 300-step evaluation budget；它是 raw seed，不是可直接训练的 gold。
- 当前 venv 没有安装 `scenic`。项目只把 LIBERO-Infinity reversal 当作 `generator_reference_only`；Infinity Scenic 与 LIBERO-PRO 尚未进入实际数据生产。

因此，当前真正的下一步不是继续录更多 raw，也不是立刻安装三个新框架，而是先把已录 episode、dirty source 和 lifecycle 变成可审计输入。

## 4. 改进后的目标树

### 4.1 比旧目标新增的结果目标

1. **第一结果目标：** 至少一条 `accepted`、collision-free、fresh replay `3/3`、在 300-step budget 内完成的 reverse gold。
2. **第二结果目标：** R6 continuation policy 在 disjoint reverse dev 上比 step0 至少多 1 个成功，且 C20 retention 损失不超过 `2/20`。
3. **第三结果目标：** 形成至少一个机械验证、FAST/offline 验证通过的 Track3 candidate ZIP，交由用户决定正式提交。
4. **节奏目标：** 从产生首个有效 checkpoint 起，每天固定一次 candidate freeze；只有模型/数据/协议有实质变化且机械门通过才生成新 ZIP，不用相同 fallback 占用每日机会。

### 4.2 前置过程 KPI

- `accepted reverse episodes / family / unique source group`，不是 raw frame 数；
- candidate replay yield、collision-free yield 与 first blocking stage；
- goal-first step、total steps、EEF path length、rotation、jerk/SPARC proxy、action saturation 与 stop-go ratio；
- reverse dev paired wins/losses、C20 retention；
- 当日是否有 materially-new candidate，以及是否由用户完成提交/回执记录。

### 4.3 停止继续优化的错误目标

- 不再把 pytest 数量、manifest 数或生成 task 数当作 Track3 进展；
- 不再等待 30–50 条数据才做第一次学习性实验；首条 accepted 后即可 tiny-overfit，3 条以上再做短续训；
- 不再要求三族同时打通才训练一个已成功 family；
- 不把 task reversal、Scenic scene 或 PRO perturbation 的“配置生成成功”记作 demo 生成成功；
- 不盲目增加通用 LIBERO-Plus 训练步数。

## 5. 待激活执行阶段

### O0：dirty/raw 现场闭包与生命周期恢复

本阶段只读或 code/evidence-only，不启动 simulator/GPU。

1. 冻结当前 HEAD、完整 porcelain、Teleop diff、测试 diff、SR pointer/log/PID、两份 PDF SHA 和所有新 episode 的路径/bytes/SHA/mtime。
2. 对 pending/rejected/diagnostic metadata 做 duplicate-key rejection；对 NPZ 做 `allow_pickle=False`、dtype/shape/finite、frame/action/state/contact 容量与双相机长度检查。
3. 识别 Teleop 修改和 raw 的作者/运行 identity；历史 raw 不重命名、不覆盖、不把 rejected 改成 pending。
4. 审核旧 UT 授权偏离：旧计划仅允许 task10 最多一条 recorded candidate，但当前存在两条 pending task10 和后续 yellow-book attempts。将超出授权标记为 `out_of_plan_diagnostic`，是否可派生使用由新审核决定，不能自动 accepted。
5. 运行相关定向测试、项目规范全量、strict JSON、py_compile、`git diff --check`；输出唯一 verdict：`approved`、`remediation_required` 或 `blocked`。
6. 若 provenance 可闭合，单独锚定原始字节和 dirty source，再归档旧 UT 计划、同步 index/Markdown handoff/machine JSON/lifecycle tests并形成 clean commit。若不能闭合，保持本草案非活动，不改 cursor/next_plan。

### O1：red-mug 两条 pending 的资格判定与重定时

前置：O0 approved 或完成指定 remediation；GPU3 空闲且单 simulator。

1. 两条 pending 都保留原字节。先比较 goal-first step、contact/lift/source-release、碰撞、action 重复、轨迹长度和视频；选一个作为 keypose seed，另一个保留为 diagnostic/reference。
2. 原始 388/468-step action 不因 machine success 直接入训；超过 300-step budget 的 raw 只能作为 `seed_for_reexecution`。
3. 从真实 EEF/state 提取 `approach → contact → close/dwell → lift → transport → place → open → retreat` keyposes。
4. 只在 free-space 段使用 minimum-jerk/五次曲线或受限 spline；contact/gripper/source-release/goal 段保留离散状态机。禁止直接删重复 action、反转 action 或跨接触段插值。
5. 目标为单物体任务 `goal_first_step <= 240`、总长 `<=260`，给评测留出稳定窗口；派生轨迹必须重新在 MuJoCo 连续执行并保存新的 state/action/contact，不能只修改 metadata。
6. 新 candidate 通过 machine gate 后做 GPU3 三个 fresh-process exact-action replay、双视角视频和用户视觉审核；3/3 前不得 accepted。
7. 派生执行失败时只允许一个预注册修复假设；再失败则转 O2/O3 由用户重录，不循环平滑参数。

### O2：服务器原生 MuJoCo + TurboVNC 录制 canary

目标是保留服务器权威 physics，同时降低当前浏览器 JPEG/HTTP 控制的延迟和重复 action。

1. 复用现有 localhost-only `Xvnc :3`，不另启同号 VNC；执行前复核 PID、socket、所有权和 GPU 映射。
2. 用户 Mac 建立隧道：

   ```bash
   ssh -N -L 5903:127.0.0.1:5903 huang@gpu01
   ```

   TigerVNC Viewer 连接 `127.0.0.1:5903`。不要把 5903 暴露到公网。
3. 先跑 30–60 秒 `no-save` canary：native viewer、20 Hz authoritative control、front/wrist recording、独立 event/step/render timestamps；不生成 candidate。
4. 通过标准：GPU3-only guard；input→next-action p95 `<=150 ms`；view median `>=15 FPS`；20 Hz simulation-step contract；keyup/disconnect 清空 held input；无动作漂移；reset 后 buffer 为零。
5. 若 VirtualGL/GLX 映射到 GPU0–2 或扰动外部作业，立即停止。不能凭 `vglrun` 命令名推断 GPU3。
6. 若 VNC 未达标，fallback 为修复当前 HTTP recorder：control/input/render 三线程解耦、20 Hz monotonic step loop、timestamped input、低频 preview；Mac 本地 MuJoCo只用于练习，不生成正式 gold。

### O3：最小人工 anchor 集

1. 优先顺序：official red mug/support → official alphabet soup/containment → 一个公开 LIBERO articulation reverse；yellow book 用于第二种 support 几何，两 mugs 长程任务在单物体稳定后再做。
2. 每次先 calibration/reset，不保存；每个 task 每轮最多一条 recorded candidate。失败保留明确标签，正常 family failure 不阻断其他 family。
3. 官方公开示例可用于训练，但 dev/holdout 必须来自 disjoint public tasks/source groups；不得把已用于训练的官方 example 再称 sealed holdout。
4. 每条 gold 同时满足 predicate、contact/source/goal continuity、collision/unknown-contact、300-step、finite/action、fresh replay和视觉审核。
5. 轨迹质量从第一条数据开始记录。由于官方归一化/权重未知，先做 Pareto 排序，不凭空设公开分数阈值；明显抖动、碰撞、长时间空 action 或超过预算的 candidate 不入 gold。
6. 第一条 accepted 后立即释放 O6 tiny-overfit；不等待三族齐全。累计至少 3 条、至少 2 个独立 source group 后才释放短 continuation。

### O4：LIBERO-Infinity reversal 与 Scenic position pilot

1. 锁定具体 upstream revision、license 和本地 patch；在独立 Python 3.11 环境安装，不直接追随 `main`，也不污染训练 venv。
2. 先做 compiler-only：公开 forward BDDL → reverse init/goal/language；逐条语义审核，instruction 直接描述动作，禁止 `Reverse the public LIBERO operation:` 前缀。
3. reversal 通过后只开启 `position` axis，每个已打通 anchor 先采样 5 个 constraint-valid scenes，冻结 seed、pose、clearance、rejection count、BDDL/source hash。
4. Scenic 只生成场景。成功轨迹由 O3 keypose transport、已训练 policy 或人工控制真实 rollout 产生；未执行动作的 scene 计 `scene_only`，不能计 raw/accepted demo。
5. position-only yield 稳定后再逐轴评估 object、camera、lighting、robot reset；首轮禁止 `combined/full`，防止无法定位失败来源。

### O5：LIBERO-PRO 风格的隔离鲁棒性评测

1. 官方 PRO suite 初期只作 held-out evaluation/reference，不直接全部混入训练。
2. 轴顺序：semantic paraphrase → position → object appearance；task/target 与 environment 最后，因为它们最容易改变任务语义或物理资产。
3. 训练增强使用 disjoint public sources 上的 PRO-style variants；官方评测集合和训练后代不得共享 source/init lineage。
4. 每轴先 5 个条件做 paired smoke。只有暴露稳定失败模式后才决定加入相应训练变体，不执行“所有扰动全部加上”的组合实验。

### O6：数据链与学习性验证

1. 首条 accepted：用 1 条/最多 3 条做 tiny-overfit，验证 `obs_t → action_t → obs_t+1`、camera/state/gripper、checkpoint、有限 loss/grad/update和训练起点闭环；tiny-overfit 模型不直接发布。
2. `accepted >=3` 且 `unique source groups >=2`：冻结 80/20 forward/reverse mixture 的真实 chunk exposure，重新计算 per-dimension QUANTILES，并做 R6 original-norm、new-norm-step0 与 mixed-finetune 对照。
3. 第一臂只跑 100/300 optimizer steps，保持 R6 continuation、L2 action LoRA heads、execute8 和其余协议；不同时换模型、rank、renderer、horizon。
4. GPU1+2 只有在外部进程完全退出且新计划重新授权后才能训练；不得迁移到 GPU0。等待 GPU 时继续 O0–O5 CPU/code/data 工作。

### O7：policy gate 与 Track3 candidate

1. 评估集分为 training-start diagnostic、reverse dev、sealed reverse holdout、C20 retention；exact-action demo replay 不是 policy score。
2. 最低研究门：相对 step0 至少 `+1` reverse dev success，C20 success loss不超过 `2/20`，且新增成功 collision-free。
3. 候选排序先按 collision-free success，再按官方连续指标的本地 proxy；任何 success=0 的候选不因轨迹更平滑被宣称 Track3 改善。
4. 通过模型、norm、fold/parity、FAST/offline、latency、ZIP CRC/layout/size 后才产生 candidate ZIP；旧 R6 fallback 保持原标签和 hash。

### O8：每日候选与人工提交节奏

从首个有效 checkpoint 开始，每日 JST：

| 时间 | 动作 |
|---|---|
| 20:30 | 冻结当日数据/checkpoint/评估协议，不再临时改架构 |
| 21:30 | 完成 candidate 或明确 `no_material_candidate` |
| 22:00 | 机械校验、SHA、大小、diff 和 submission note |
| 22:30 前 | 如有新合格候选，提醒用户人工上传/提交并保存回执 |

上传/比赛提交默认仍是用户动作。相同权重、相同 norm、相同 ZIP 或仅改文件名不算新候选。每日机会用于回答明确实验问题，不为“保持连续”提交已知相同 fallback。

### O9：截止日前排程

| 日期（JST） | 内部目标 |
|---|---|
| 9 月 8 日晚 | O0：冻结 dirty/raw；停止新增录制 |
| 9 月 9 日 | O1/O2：red-mug adjudication、VNC no-save canary、争取首条 accepted |
| 9 月 10 日 | O3：support/containment/articulation 最小 anchor；首条 accepted 立即 tiny-overfit |
| 9 月 11 日 | O4 position-only 5-scene pilot；形成 3+ accepted/source diversity |
| 9 月 12 日 | O6 100/300-step R6 continuation；paired dev + C20 |
| 9 月 13 日 | 第一份有 reverse policy 证据的 Track3 candidate，或明确 blocker |
| 9 月 14–15 日 | Scenic/PRO 单轴补强、最多每日一个 materially-new candidate |
| 9 月 16 日 | 冻结最终训练；只做 holdout、FAST、ZIP 和提交演练 |
| 9 月 17 日 | 上午确定最终候选；目标 22:30 前由用户完成提交，保留缓冲 |

日期是内部停损点，不是成功承诺；若官方截止更新则重排。

## 6. 失败分流与效率规则

| 失败类型 | 处理 |
|---|---|
| identity/hash/partial/telemetry/GPU/supervisor unknown | 全局 sticky hard stop；先修证据/资源边界 |
| 正常 family behavior failure | 只停该 identity；运行一个预注册结构不同 alternative，随后转人工/keypose；不阻断其他 family/code-only 工作 |
| VNC 延迟或 GPU 映射失败 | 停 VNC canary，回到解耦 HTTP recorder；不改用 Mac 产正式 gold |
| red raw 超 300 steps | 只作为 keypose seed，重新执行重定时轨迹；不抽帧伪造 20 Hz |
| Scenic compiler/sampling 失败 | 停 Infinity pilot，继续固定场景人工/模板数据；不让新依赖阻断首个 gold/训练 |
| accepted=0 | 禁止 norm/正式训练；继续 O1–O3 |
| tiny-overfit 失败 | 先查时间对齐、gripper、norm、adapter；不扩训练步数 |
| reverse 改善但 retention 退化 | 审 norm-only 与 forward exposure，保留候选为 research-only |
| 9 月 13 日仍无 policy success | 集中一个已 accepted family和官方示例，停止 combined perturbation/新模型研究 |
| 9 月 16 日仍无新合格候选 | 保留诚实 fallback；不把失败数据或 scene-only 产物包装成 Track3 improvement |

## 7. 用户需要参与的动作

1. O0 完成前不继续录 demo、不修改或删除现有 raw。
2. VNC canary 被明确释放后，在 Mac 运行 SSH tunnel 并用 TigerVNC 连接 `127.0.0.1:5903`；不用自行启动新的 `vncserver`。
3. 正式录制只在 HUD 显示 correct task/start/gripper/GPU/recording identity 后进行；每条完成后立即等 machine gate，不连续堆 raw。
4. 对 replay 3/3 的审核卡选择：

   ```text
   Episode ID:
   Task / family:
   Start-state SHA / action SHA:
   Goal-first / total steps:
   Collision / contact / source / goal gates:
   Exact replay: 3/3
   Front video:
   Wrist video:
   人工选择: approve_visual | reject_visual | needs_rerecord
   原因: wrong_object | collision | bad_grasp | unsafe_motion | unstable_goal | excessive_jitter | other
   ```

5. 每日收到候选 bundle 后由用户决定是否上传；回传 submission ID/时间/文件 SHA/leaderboard 结果，供下一日归因。

## 8. 激活条件

下一执行者先运行 O0，不得从 O1/O2 开始。只有以下全部成立才把本文拆为唯一 active execution plan：

- 当前 dirty/raw 归属和 provenance 已闭合；
- 旧活动计划完成独立审核并归档；
- root/OpenPI clean；`LIBERO-plus` 原 dirty 保持不变；
- machine handoff、Markdown handoff、plan index、lifecycle tests 同步；
- 无 live Track3 worker，GPU边界重新记录；
- 新计划明确 writable paths、identity budgets、用户输入点、GPU3和GPU1+2条件权限。

在此之前，本文是已经落盘的下一步设计，不是执行授权。
