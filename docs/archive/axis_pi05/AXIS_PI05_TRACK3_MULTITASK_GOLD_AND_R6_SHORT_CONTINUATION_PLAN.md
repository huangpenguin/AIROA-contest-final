# AXIS Pi0.5 Track3：多题目 Gold 采集与 R6 短续训计划

> 创建日期：2026-09-09（Asia/Tokyo）
> 唯一活动计划：是
> 前置审核：`docs/reviews/axis_pi05/TRACK3_O1_2_OVER_BUDGET_INDEPENDENT_AUDIT_20260909.md`
> 前置审核结论：`approved`
> 审核输入：`2b544b110189324b3db71070dcfadbf2634eccd7`
> 第一执行阶段：`MG0_evidence_lock_and_recorder_budget_remediation`
> MG0 execution status: passed_hard_stop_ready_for_user_calibration; next stage: MG2 after MG1 family failure
> MG1 execution status: assisted and manual attempts exhausted as family failures; next stop: MG2_alphabet_soup_requires_explicit_user_takeover
> MG0 执行报告：`docs/reviews/axis_pi05/TRACK3_MG0_RECORDER_BUDGET_TIMING_20260909.md`

本计划把主线从单条 red-mug canary 扩展到“多题目 accepted gold → 数据冻结 → tiny-overfit → R6 100/300-step 短续训 → 当日候选 ZIP”。目标是在不牺牲 provenance、物理门和正向 retention 的前提下提高最后一周的有效吞吐。

## 1. 冻结事实与总边界

- O1.2 episode `official_t3_red_mug_1788921471` 是 `314` 步的 machine-observed physical success，但永久保持 `gold_eligible=false`、`replay=not_evaluated`。它只可作为 read-only retiming seed。
- 禁止通过删帧、样条插值动作/视频、改 timestamp/metadata 或倒序/取负 actions 将失败/超预算 raw 伪造成 accepted。所有 retiming/waypoint 方案必须从冻结 start state 重新连续执行，并保存新 state/action/contact/video。
- accepted 生命周期固定为 `raw → pending_replay → fresh replay 3/3 → human reviewed → accepted → training_eligible`。任何 failed/rejected/diagnostic/over-budget episode 均不得进入模仿学习正样本。
- natural instruction 不加 `Reverse the public LIBERO operation:` 前缀；使用冻结的官方自然指令和官方 BDDL/predicate。
- TigerVNC display `:3` 是已通过 no-save 与一次成功物理录制验证的首选界面。HTTP 仅作 localhost HUD/备用控制；是否使用 HTTP 不是本轮 gold 资格变量。
- GPU0 永不触碰。GPU1/2 仅在外部作业自然退出且重新通过 idle guard 后用于训练；GPU3 仅用于 MuJoCo、fresh replay 与无 simulator 并发时的小型评测。不得 kill/抢占外部进程。
- 不启动 dense full-FT，不改变模型家族、LoRA topology 或 normalization 类型。训练主线固定从 `r6_dual_l2_fresh500_r1`、`L2_action_lora_heads`、per-dimension `QUANTILES` 继续。
- 本计划允许本地 FAST/fold/candidate ZIP，但不允许自动上传或比赛提交。正式提交仍由用户决定和操作。

## 2. Stop policy：缩小正常失败的停止范围

- identity/hash 不一致、raw/metadata/path/partial 不确定、contact telemetry unknown、supervisor/PID 冲突、GPU 边界不明、NaN/Inf 或 writer 覆盖风险：`infrastructure_failure`，全计划 sticky hard stop。
- 已知、完整 telemetry 下的抓取失败、碰撞、掉落、goal 不成立、replay 不一致或超 300 步：该 episode `family_failure`/`rejected`，只停止当前 attempt。若该 task 尚有冻结额度，可做一次针对性修正；否则转到下一 task，不冻结全晚。
- 代码签名/CLI 等可证明发生在 env 创建前、零 physics 的 deterministic bug：封存失败 attempt；本计划预授权一次 code-only 修复、全测试与新 clean-source commit，随后恢复当前阶段。不得在同一 dirty identity 自动 retry。
- machine success 后立即 replay/review，不先积累一批未经验证的 raw。replay 失败只淘汰该 episode；剩余额度允许时再录一次。

## 3. MG0：证据锁、recorder budget 与 timing 修复

MG0 先完成，不录 formal episode：

1. 锁定 314-step seed 六个文件的 bytes/SHA、26 个 NPZ array contract、start/end state、关键 contact/source/goal 窗口；历史 raw/metadata 只读。
2. formal recorder 将 `300` 设为真实 hard stop，而不是只在录完后派生拒绝：step 220 预警、260 目标、290 红色 hard warning；到 300 未满足 gate 时原子写 rejected/diagnostic，不进入 pending。
3. `pause` 时不得调用 `env.step`；断线自动 pause；keyup 立即停止 motion pulse。加入 coarse/fine 两档但默认 action cap 不高于当前 `0.30`，先减少绕路和 overshoot，不以盲目增大动作换步数。
4. metadata 分离 `sim_control_hz=20`、`operator_tick_hz`、wall timestamps 与视频编码 fps；deadline 以 operator tick budget 统计。不得仅改 metadata 声称 20 Hz 实时控制。
5. 在 TigerVNC :3 做 no-save A/B calibration：2 Hz 与 4 Hz 各短测一次，记录实际 input-to-frame latency、keyup 尾巴、reset、pause-zero-step 与方向/夹爪；只选择用户能稳定控制且无 dropped-release 的档位。HTTP 不作为必需主界面。
6. strict JSON、targeted/full tests、py_compile、diff-check、clean-source commit 全部通过后才进入 MG1。

MG0 hard gate：formal writer 无法在 300 阶段 fail-close、pause 会消费 step、或 timing 字段仍相互冒充时停止，不录下一条。

## 4. MG1：red-mug 成功轨迹的重新执行式压缩

1. 从 314-step seed 派生新的 retiming blueprint。可删除 blueprint 中非关键零运动段、缩短重复轴向段、使用实际 EEF/keypose 做最短安全路径；必须保留 grasp/source-release/goal 各连续 4 帧硬窗。目标总长 `<=260`，绝对硬上限 `300`。
2. 优先执行一次 deterministic assisted re-execution：从冻结 start state 逐步执行新 schedule，重新记录全部 action/state/contact/front/wrist。它是新 physics attempt，不是 replay，也不继承旧 episode 的 success。
3. assisted attempt 若为完整 telemetry 下的行为失败，允许一次用户 TigerVNC manual retime attempt；用户先 no-save calibration，再明确回复 `接管 Teleop：official_t3_red_mug`。若 infrastructure failure 则不消费 manual quota，按全局 hard stop 处理。
4. 任一 attempt 达到 machine success、`<=300`、collision/unrelated=0 后，立即用三个 fresh process 做 exact-action replay `3/3`，生成 front/wrist 审核卡。用户按 `审核 <episode_id>：accept` 或 `reject：<原因>` 明确确认；未确认时状态保持 `pending_gold_review`。

red-mug formal quota：assisted 最多 1 条，manual 最多 1 条；一条 accepted 后停止该 task 的继续录制。

## 5. MG2–MG3：多题目录制与即时验收

### 5.1 顺序与额度

在 red-mug accepted 或 red-mug 两个额度均耗尽后继续，不因正常行为失败停掉其他题目：

1. `official_t3_alphabet_soup`：containment 单物体，最多 2 条 formal attempt；
2. `official_t3_yellow_book`：support/shelf 单物体，最多 2 条 formal attempt；
3. `official_t3_two_mugs`：双物体长程，只有前三个单物体任务累计至少 3 条 accepted，且 no-save blueprint 预计 `<=260` 时才允许最多 1 条；否则 `not_evaluated`，不拖慢首轮训练。

多题目总 formal 上限为 7（MG1 的 2 条 + MG2/MG3 的 5 条）。每个 task 首条失败后，第二条只能针对已记录的首个失败阶段调整；不得无假设重复录制。

### 5.2 每条标准流程

1. preflight：root/OpenPI clean、无 live recorder、GPU3 idle、官方 BDDL/hash、source=true/goal=false、自然指令、gripper OPEN。
2. no-save calibration：用户确认画面、移动轴、夹爪、goal 区和阶段预算；reset 后才开始 formal。
3. formal：目标 `<=260`，hard max 300；contact/raw、source/goal、collision/unrelated、start/end state 与双视频逐帧保存。
4. machine success：立即 fresh replay `3/3`；通过后生成 6 帧审查卡（start、precontact、grasp、source-clear、place/release、final）和阶段表。
5. human review：确认正确物体、正确 source/goal、无明显碰撞/穿模/无关物体扰动、动作可接受。只有明确 accept 才 promotion。

建议人工节奏不是统一硬门：approach 约 80、grasp 95、source clear 120、transport 180、place 225、release 240、goal settle 250。每个新几何先在 no-save calibration 中冻结自己的 milestone；step 220 尚未 source-clear 时主动 abort/reset，避免把必超预算轨迹写入 pending。

## 6. MG4：数据冻结、source-group 隔离与 QUANTILES

1. tiny-overfit 最低门：至少 1 条 accepted；正式 R6 短续训最低门：至少 3 条 accepted、至少 2 个 task ID、至少 3 个独立 source/init lineage；覆盖 2 个 family 是优先目标，不是首轮训练硬门。
2. 按 `(task, BDDL/source, init lineage, parent seed)` 分组。相同 seed 的 re-execution/position variant 不得跨 train/dev；accepted train、reverse dev、C20 retention 三者互斥。
3. disjoint reverse dev 从未进入 accepted train 的公开 BDDL/source group 冻结；training demo 的 start state 只用于 tiny-overfit sanity，不作为泛化成功证据。
4. 校验 `obs_t → action_t → obs_(t+1)`、camera/color/resize/state/action/gripper、episode padding/mask 与 chunk 不跨 episode。只有 accepted manifest 进入 loader。
5. 从冻结 train mixture 重新计算 per-dimension QUANTILES；holdout 不进入 norm。记录 episode/frame/chunk 三个 exposure 口径，并以 family→source→episode/chunk 分层避免长轨迹淹没小任务。
6. forward/reverse 首臂固定有效 chunk `80/20`。同一个 reverse episode 可以在 sampler 中重采样，但不得复制文件冒充独立数据。

## 7. MG5：tiny-overfit 与资源门

1. 训练前再次检查 GPU1/2。它们若仍有外部进程，只把 MG5 标为 `waiting_for_gpu1_gpu2`；继续允许 MG2/MG3 的 GPU3 采集，不终止他人进程，也不把 GPU3 临时改成正式双卡训练。
2. 用 1–3 条 accepted 做 tiny-overfit，验证 loss/grad/params finite、方向与 gripper 时序、checkpoint read/write 和训练起点闭环。loss 降低本身不等于 policy success。
3. 同一 R6 weights 先跑 old-norm reference 与 new-norm step0。若 norm-only 造成 C20 退化超过 2/20 或动作显著漂移，停止正式短训，审核 adapter/norm exposure。
4. tiny-overfit 至少在一个训练起点复现任务，且 schema/adapter/norm hash 全闭合，才释放 MG6。

## 8. MG6–MG7：R6 100/300-step 短续训与评测

1. 从 R6 step-500 continuation 初始化，L2 LoRA、FP32、既有 optimizer recipe；不回 Axis init、不做 dense full-FT。
2. 首臂为 80/20，保存 step 0/100/300。step 100 立刻评估，不等待 300 才第一次看闭环；若出现 NaN/Inf、adapter mismatch、全局动作饱和或明显 C20 崩塌则 hard stop。
3. 每个 checkpoint 运行：训练起点 sanity、disjoint reverse dev、固定 C20 正向 retention。发布门为 disjoint reverse 至少增加 1 个成功且 C20 相对 R6 最多损失 2/20。
4. 若 80/20 到 step 300 仍无 reverse improvement，但 tiny-overfit 成功、norm-only 通过且 C20 稳定，预授权唯一一条 recovery arm：只把 reverse chunk 占比改为 40%，其余完全冻结，最多 100 steps；不得同时换 LR/rank/模型/norm。
5. failed checkpoint 保留为 diagnostic，不 fold、不打 ZIP。满足门的最早 checkpoint立即进入 MG8，不必等所有可选数据或可选 arm。

## 9. MG8：当日 candidate 与提交节奏

1. 对最早通过门的 checkpoint 做 raw-LoRA/folded parity、FAST export、离线 startup、latency、layout、CRC、size 和 manifest/SHA 验证。
2. 只有 model/data/norm bytes 实质变化才生成新 ZIP；旧 R6 fallback SHA 不变，不把它重命名为 Track3 improvement。
3. 本地候选可在 step 100 或 step 300 产生。报告明确 `track3_improvement_candidate` 或 `fallback`，并给用户 SHA、大小、评测差异和是否建议提交。
4. 本计划不自动 upload/submission。用户决定提交后再执行外部动作；每天有合格新候选就当天提交，不因等待可选扩增浪费机会。

## 10. MG9：非阻塞扩增支线

- 等待用户或 GPU 时可做 Infinity reversal compiler 与 Scenic position-only 配置/fixture；缺少依赖时先报告，不擅自联网安装。
- 只有 canonical pipeline 已有至少 3 条 accepted 后，才允许每个成功 task 最多 1 个 Scenic position variant，并必须重新连续执行/replay/review。它与 parent 同 lineage，不进入 disjoint dev。
- LIBERO-PRO perturbation 只作 held-out robustness evaluation，第一轮不混入训练。
- Infinity/Scenic/LIBERO-PRO 均不得阻塞 MG5/MG6 的首个短训和候选 ZIP。

## 11. 今晚与最后一周的实际顺序

- 9 月 9 日今晚：MG0 code/no-save → MG1 red assisted re-execution；用户在线则完成 red manual fallback 和 alphabet-soup calibration/首录。
- 9 月 10 日：alphabet-soup、yellow-book 录制/replay/review，争取达到 3 accepted / 2 family；同时冻结 loader/dev/norm。
- 9 月 10–11 日：GPU1/2 可用即 tiny-overfit、100/300-step；最早通过门的 checkpoint 当天出本地 ZIP。
- 9 月 12–14 日：补第二条同 task 或 Scenic position variant、必要时唯一 40% recovery arm；每天只保留最优有实质变化候选。
- 9 月 15–16 日：冻结最终数据、评测和 ZIP；不再扩大基础设施范围。
- 9 月 17 日：只做最终机械复验与用户提交，不临时换模型/训练 recipe。

## 12. 当前需要用户做什么

1. 暂时保留 TigerVNC :3，不需要改回 Mac 本地 MuJoCo，也不要自行启动旧 recorder argv。
2. 执行器完成 MG0 并报告 `ready_for_user_calibration` 后，保持约 20–30 分钟在线，回复 `接管 Teleop：<task_id>`；每条 formal 前都会先 no-save 校准。
3. 收到审核卡后，用 `审核 <episode_id>：accept` 或 `reject：<原因>` 回复。没有这一步不会写 accepted。
4. 在准备进入 MG5 前，告知 GPU1/2 哪个时间窗可由 Track3 独占；不要手工杀掉当前外部 YOLO 进程。若本周无法释放 GPU1/2，需要另行明确授权比赛 Brazos GPU，本计划不自动扩大权限。

## 13. 收尾

MG0–MG9 任一执行批次结束后生成 immutable attempt evidence、cursor/heartbeat/manifest/outer/closure，运行 strict duplicate-key JSON、targeted/full tests、py_compile 与 `git diff --check`，形成 clean commit并 hard stop到独立审核。所有未进入阶段标 `not_evaluated`，不得以 false/pass/zero-risk 代替。
