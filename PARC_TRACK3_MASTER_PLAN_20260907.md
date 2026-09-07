# PARC Track 3：数据优先的总体路线与 Agent 工作包

> 日期：2026-09-07（Asia/Tokyo）
> 文档性质：战略提案，待独立审核；不是活动执行计划，不授权新运行。
> 主仓库：https://github.com/huangpenguin/AIROA-contest-final
> 本次证据锚：`760dc65adc74193112692ab003e19a74d588ea9f`
> 当前唯一活动计划仍为 CR0–CR14 独立审核。本文件不替换它、不修改活动指针。
> 核心目标：第一条可信成功逆向示范 → 可学习的小数据闭环 → R6 续训收益 → 保留正向能力的部署候选。

## 1. 总设计结论

**当前瓶颈是 accepted reverse supervision 为零，不是缺少更大的模型、更多训练步数或又一轮通用基础设施重构。**

主线：CR 独立审核收口 → 六条失败的阶段诊断 → 第一条 gold demo → 小规模 accepted 数据 → tiny-overfit 学习链路验证 → R6 小步续训 → 公共 reverse holdout + 正向 retention → FAST 部署检查。

这里的 gold demo 指真实连续执行成功、fresh-process exact-action replay 合格、经视觉复核和现有机器门验收的示范，不是仅人工标记成功的视频。

六项架构决策：

1. **成功示范优先，自动化次之。** 自动脚本、真人 Teleop、人工 keypose 都是取得有效轨迹的手段。自动生成器没有成功时，不无限完善生成规模。
2. **一族先打通。** 不要求三族同时成功才验证学习；另两族继续保持独立失败/未验证状态，不能混入 accepted。
3. **复用现有资产。** recorder、state restore、replay、typed traces、supervisor、R6 LoRA/folded/FAST 不另起炉灶。
4. **模型主线保持 R6 continuation。** 使用 `r6_dual_l2_fresh500_r1 / L2_action_lora_heads`；不回 Axis init、不启动 dense full-FT、不迁移模型家族。
5. **一次回答一个问题。** 第一轮验证“有效 reverse 数据能否改善 R6”，不同时改变模型、rank、LR、norm 类型、renderer、horizon。
6. **工程可信、示范成功、模型成功、部署成功分别验收。** 测试 passed 数量不是逆向操作能力，能生成 ZIP 不是 Track3 改善。

以下数量、预算、时间窗口均为建议，须由 reviewer 冻结后才能实施；不是当前运行授权或成功保证。

## 2. 本次读取了什么，尚未证明什么

### 2.1 已找到并读取 Drive 知识库

文件：`PAI_PARC_Slack_Knowledge.md`

链接：https://drive.google.com/file/d/1lxWHHzQX36CEpL1t58EiaZFW_mdbKPQU/view

本次读完正文。表头写 2026-09-02 07:49 JST，正文变更日志及 Drive 修改记录到 2026-09-03；不能视为 9 月 7 日最新官方规则。

对本路线有用的约束：

- 冻结评估协议、renderer、flow steps、执行 horizon、task/init/seed；
- task ID 数量不等于操作语义多样性；
- episode-balanced 不等于 frame/chunk-balanced；
- evaluation seeds 与 training seeds 分别度量；
- 按 approach/contact/grasp/lift/transport/place 等阶段分析失败；
- 社区 LoRA 或 H10/H12 结果有特定模型与协议范围，不能直接变成本项目 Pi0.5 的实测结论；
- 自然语言“桌面”与公开 BDDL 目标区域可能不同，参与者的区域尺寸估计不是官方通用规则。

### 2.2 已读仓库证据

已读取 AGENTS、完整 MEMORY、plan index 与 Markdown handoff 的当前字段、machine handoff 当前字段、CR 执行闭包、CR summary、当前 CR 独立审核计划。

| 项目 | 仓库记录 | 含义 |
|---|---|---|
| CR9–CR10 | 3 族 × primary/alternative，共 6 条 CPU/no-renderer；均 family_failure | 存在新尝试，尚无成功监督 |
| CR11 | camera replay=not_evaluated | 不存在已完成回放的成功示范 |
| CR12 | human_input_required handoff | 人工路径准备态，不是人工示范完成 |
| CR13 | machine-ready dry-run | 不等于数据生产或训练 |
| 数据计数 | accepted=0，training_eligible=0 | 当前不能启动声称基于有效 reverse 数据的正式训练 |
| finalizer | 原始失败保留，derived correction 声称通过 | 仍待独立 reviewer 复核 |
| 工程验证 | 报告声称 380 passed / 7 skipped / 26 warnings | 不是物理操作成功证据 |

summary 的 targeted=17 与闭包报告的 lifecycle targeted=110 须确认测试集合和边界；不能直接合并，也不能未经核查就判作矛盾。

MEMORY 某些历史快照与 index/handoff 顶部当前字段不同。按仓库规则优先读取 live + immutable artifacts/current tree，其次 machine current pointers、plan index、Markdown handoff，不能从历史段落恢复已失效授权。

### 2.3 本次没有执行的事项

没有连接执行服务器，没有现场检查当前 git/PID/GPU，没有运行项目测试或复算全部原始 trace/manifest/closure，没有对完整 machine handoff 执行 strict duplicate-key JSON parsing，没有完成 CR 独立审计。

没有启动 physics、replay、Teleop、数据 promotion、训练、norm、FAST/ZIP 或比赛提交。

GitHub 新建分支和创建 Issue 实际返回 `403 Resource not accessible by integration`，仓库未写入。元数据的 push/admin 字段不代表当前连接器具备对应写入 scope。

因此本文是基于已读记录的战略提案，不是独立审核结论；不声称用户服务器目前空闲。

## 3. 关键路径与阶段门

| 阶段 | 要解决的问题 | 产物 | 通过标准 |
|---|---|---|---|
| G0 审核收口 | CR 结果与失败分类可信吗？ | 单一 verdict、blocker 表、failure_matrix | 当前独立审核计划要求完成，现场与证据一致 |
| G1 成功示范 | 能否真实完成一个逆向操作？ | gold demo、fresh replay、双视角复核 | 起点/动作/接触/目标/回放/验收合格 |
| G2 小数据闭环 | 数据能学习且无泄漏吗？ | source-group split、accepted manifest、转换 QA | train 只含 accepted，holdout 与 norm 隔离 |
| G3 学习性验证 | R6 能从这些监督学到动作吗？ | tiny-overfit、短续训、step0 对照 | 对齐正确、有限值、闭环证据、checkpoint 可用 |
| G4 能力验证 | 能否跨起点改善并保留正向能力？ | reverse dev/holdout、C20、paired 结果 | 有收益且 retention 合格，结论不超出证据范围 |
| G5 部署验证 | 改善能否进入提交路径？ | fold/parity/norm/FAST/offline/ZIP 验证 | 行为与机械门合格，由用户决定正式提交 |

G0 完成不自动释放 G1；G1 完成不自动释放规模化生成/训练。集成负责人依现有 lifecycle 形成唯一 reviewed execution plan。各 agent 可并行只读设计，实施需等明确授权。

## 4. G0：一次收口审核，不再无边界治理

必须审的实质风险：clean-source launch、原始 partial/synthetic observation 区分、late-producer manifest、typed/no-pickle/finite、source/goal/contact gate、normalized action 单位、family/infrastructure 分类、进程/GPU 边界、原始失败与 correction 关系。

这些不能因赶进度降级。缺失证据标记 unavailable/not_evaluated，不制造“重构出的原始证据”。

审核同时产出六条 attempt 的 failure_matrix：

`identity / family / strategy / last_entered_stage / first_contact / contact_role / grasp_or_joint_progress / source_before_after / goal_before_after / clip / stall / timeout / stop_reason / evidence_path`

不要从 success=0 推断三族都卡在 grasp。未进入阶段用 not_evaluated；历史 trace 缺接触对就明确缺失，不推造接触事实。

影响授权、数据资格、物理正确性和资源安全的缺陷必须阻断。纯文字、冗余叙述、非关键索引问题应集中修正，不扩展成下一整套治理工程。是否影响执行由独立 reviewer 判断，不由执行者自降标准。

最终仍按现有计划输出 approved/remediation_required/blocked 之一，作者与审核者分离。

## 5. G1：第一条成功示范

### 5.1 三族分别诊断

| family | 优先检查 | 容易混淆之处 |
|---|---|---|
| support_remove | 夹爪相对位姿、稳定夹持、离开原支撑、合法目标放置 | 离开支撑不等于最终目标成功 |
| containment_remove | 抓取可达性、容器边缘、提取通道/高度、对象接触 | 触到不等于抓住，抓住不等于清出容器 |
| articulation_close | joint axis/sign/range、把手或允许接触面、持续推进 | EEF 到位不等于关节到合法关闭状态 |

只读合法公开 BDDL/predicate/目标几何。三族是工程分类，不等于已知隐藏任务的完整分布，不使用泄漏脚本定位隐藏任务。

如果任务要求放置/释放，不用“仍夹持”冒充完成；若仅要求关节关闭，不随意新增无关动作门。

### 5.2 三条获取路径

**A：明确物理假设的自动修正。** 根据 failure_matrix 选择最接近成功的一族。首批建议最多 2 条全新 identity，事先写清假设、修改变量、预期阶段进展和 stop 条件。新 seed 或新 ID 不等于新策略。

**B：真人 Teleop。** 自动路径无实质进展时，请人用已有 recorder 完成一条真实成功轨迹。先确认观察画面与控制方向；无人操作就标记 human_input_required，不能让 agent 冒充人工输入。

**C：人工 keypose + 控制器闭环执行。** 人定义接触点/姿态/关键帧，控制器生成连续实际动作；记录真实 state/action，不把理想 waypoint 当成已经执行的 trajectory。

现阶段不把“接入全新 MPPI/IK/planner 框架”作为首条示范依赖。确需新增模块，由 reviewer 说明现有 controller 的具体能力缺口、最小改动与预算。

### 5.3 禁止伪逆向数据

不能倒序正向 actions、全部取负、倒放视频、只改 instruction，不能在轨迹执行中瞬移对象或强制改 state 来制造成功。起点可以按批准的 restore 合同构建，但其后的动作必须真实连续执行。

数据采集 controller 的 simulator 信息与部署 policy 的可用 observation 分开审；不得把训练辅助的 privileged state 偷渡进提交 policy 输入。

### 5.4 数据里程碑

第一目标：**一族一条 gold demo**。

第二目标：约每族 3 条，尽量来自不同 source/demo/init lineage，总目标约 9 条；只是记录→回放→验收→样本链路验证，不宣称泛化。

第三目标：先可行族约 30–50 条 accepted；提前隔离 dev/test source group 并单独留量。新增量必须带来独立 source/初态/接触覆盖，不靠重复 gold demo 凑数。

扩量依据是 accepted yield、独立起点覆盖、单位采集成本与闭环收益。一个 family 先通过就推进该族小实验，不等三族齐备，也不降低另外两族验收。

## 6. G2：数据契约、回放与隔离

### 6.1 最低 provenance

每条 episode 保存：source URL/revision/license、public task/demo、inverse operator/instruction、parent/source lineage、env/model/assets/controller/wrapper commit/hash、实际 control frequency、完整可恢复状态包/hash、RNG/init identity、实际 7D actions、双相机与 proprio、stage/contact/source/goal 日志、stop reason、record/replay hash、reviewer 与 acceptance 状态。

动作遵守项目 LIBERO OSC_POSE adapter 契约；不能将归一化 command 当成米/弧度再除一次物理尺度。gripper 编码以当前训练/部署 adapter 实测为准。

### 6.2 验收

复用现有状态机的语义：

`raw → pending replay → replay passed → human reviewed → accepted → training eligible`

CPU terminal success 先成为候选；视觉复核也不能绕过机器门。failed/partial/diagnostic/holdout 保留但不进入正样本 imitation training。

新候选 demo 的 fresh-process exact-action replay 是候选验证，不是重跑历史 CR 失败 identity；新计划须写清验证 namespace、资源和预算。

完整恢复不能只检查 qpos/qvel 存在，需按既有实现核对模型及必要的 actuator/mocap/controller/wrapper 状态。物理 replay 一致性与 renderer 像素差异分开判定；既不放松真实动作证据，也不武断对所有 EGL 图像强制 bitwise 相同。

goal 需按预先冻结的检查窗口验证，防止单帧瞬时成立。接触/碰撞门要区分任务所需接触与禁止接触；不能把所有接触都当碰撞，也不能忽略阻碍任务的碰撞。

### 6.3 分组先于扩增

至少按 `(source task, source demo, composed init lineage)` 分组。相同 source 的重放、camera 变体、语言改写、轻微 seed/姿态扰动后代不得跨 train/dev/test，不随机切帧制造 holdout。

封存 holdout 不进训练、norm、checkpoint selection 或行为修复设计；反复用于决策的数据属于 dev。

公共任务可能已在基座预训练中，只能准确称“本次 reverse 微调未见 source/起点/任务”，不能无证据称基座从未见过。

### 6.4 转换 QA

核对 `obs_t → action_t → obs_(t+1)`；chunk 不跨 episode，结尾 padding/mask 正确；camera 顺序、颜色/resize、state 维度、gripper、instruction 与 R6 adapter 一致。

20 Hz 来自实际控制步频，不是仅改 metadata 或复制帧。独立 source 数、episode 数、frame/chunk exposure 分别记录。

reverse 按 family→source→episode/chunk 分层；长轨迹不能淹没其他任务。冻结训练 manifest 后再计算 norm 和启动训练。

## 7. G3：R6 学习性验证与短续训

### 7.1 normalization 是单独风险

继续使用项目的 per-dimension QUANTILES；只从最终 train manifest 的有效分布计算，记录统计权重/方法，holdout 不进入统计。训练、OpenPI、folded PyTorch、LeRobot processors、FAST 引用一致 norm hash。

**换 norm 即使完全不更新权重，也可能改变 R6 行为。** 因此保留以下条件：

| 条件 | 权重 | norm | 目的 |
|---|---|---|---|
| R6-reference | 原 R6 | 原 norm | 原部署行为参考 |
| new-norm-step0 | 同一 R6 | 本轮新 norm | 测 norm-only 差异、真实续训起点 |
| mixed-finetune | R6 续训 | 本轮新 norm | reverse 监督候选 |
| forward-only-control（预算允许） | R6 续训 | 本轮新 norm | 对照“多训练了几步”的影响 |

仅换 norm 已明显退化时，先审数据分位数、adapter、输入输出变换与采样权重，不误判成微调失败，也不偷偷改用其他 norm 类型绕过契约。

### 7.2 tiny-overfit 先验证链路

用 train split 中 1–3 条 accepted demo，检查 loss/grad/params 有限值、动作方向、gripper 时序、checkpoint 读写与明确标注的训练起点 rollout。

loss 降低不是闭环成功；训练起点成功只是学习性证据，不发布 tiny-overfit 模型。只有少量示范时，norm 代表性不足应明确标记，不能据此扩大正式训练。

### 7.3 第一臂续训

从 R6 continuation 初始化，保持 L2_action_lora_heads。沿用当前 forward/reverse=80/20 设计，但先核对现有实现的采样口径；建议明确为有效 chunk 抽样占比，不暗改 episode/frame 语义。

建议先看 100/300/500 optimizer step 检查点。LR、warmup、可训练模块、optimizer 是否重置、effective batch 与 accumulation 由训练 reviewer 从 R6 可复现 recipe 派生，先做小规模有限值/memory canary；本文不凭空指定新 LR。

forward-only 对照尽量匹配新 norm、总 updates、effective batch、种子和评估协议。固定总算力时，它会见到更多 forward，须报告 exposure；严格匹配 forward exposure 的因果对照可后置。

首臂通过后再独立测试 60/40、更多独立示范或更长训练。不要同时换 rank、模型、renderer、horizon。

非有限 loss/grad/update/params/EMA 按冻结作用域停机；有限值本身不证明学习有效。

## 8. G4：评估和候选选择

### 8.1 固定本项目协议

沿用 execute8 primary，不直接复制 SmolVLA 的 H10/H12。冻结 backend/commit、checkpoint、norm、环境、20 Hz、renderer、预处理、flow steps、chunk/execution horizon、task/init/seed、timeout 与成功检查。

协议改变后重跑基线，不能新模型 EGL、旧模型 CPU 混比。RNG seed 与明确的 init-state identity 分别记录。

### 8.2 四种集合

- gold exact-action replay：验证示范与重放，不是 policy score；
- reverse dev：可反复用于诊断和选 checkpoint；
- sealed reverse holdout：候选冻结后才用；
- C20 retention：沿用固定正向清单，最终候选再扩代表性更好的 forward 检查。

建议先每个可行 family 6–10 个独立 dev source/init 条件做配对 smoke；有收益再扩优胜者，例如每族约 20 个独立条件，取决于合法资产和批准预算。这是试验预算，不是统计充分性承诺。

未见 init、未见 source demo、未见 public task 分层报告，不混成一个泛化率。source 数不足时承认证据弱，不能用大量重复 seed 冒充独立性。

### 8.3 研究门不等于发布门

既有“比 step0 至少多 1 个 reverse success，C20 最多损失 2/20”保留为最低继续研究门，不是统计显著性，也不自动允许发布。

报告分族成功、paired wins/losses、source-group 数、阶段失败、collision/stall/timeout。同 source 有相关性时按 source group 聚类；不把帧或重复 episode 当独立样本。

最终优胜者再做独立 training seed 检查；多个 evaluation seed 不等于多个 training seed。预算不够就降低结论强度，不伪装稳定性。

holdout 一旦用于挑 checkpoint 就不再封存。单族收益不等于全部 Track3 解决，公共 proxy 不保证隐藏官方分数。

## 9. G5：资源和部署

GPU0 禁用并保护外部作业，即使看似空闲也不主动占用。按仓库边界：GPU1+2 获授权后做 FP32 LoRA；GPU3 做获授权 simulator/replay/generation/eval，一次一个 simulator。比赛 GPU 与剩余预算另行确认授权。

长作业单 supervisor、identity lock、atomic cursor/heartbeat，先检查 live PID/GPU，不从 SSH 断线或旧仓库记录推断作业停止。

候选检查：manifest/norm hash、raw-to-folded parity、camera/state/action adapter、offline startup、无本机绝对路径/外网下载依赖、FAST latency/吞吐、资源上限、ZIP CRC/layout/size、reset 行为。

旧 R6 fallback 保持原 hash，不重新命名为 Track3 改进。正式 upload/submission 由用户决定，除非后续明确授权。

## 10. Agent 工作包

| Owner | 立即可做的只读工作 | 交付物 | 禁止事项 |
|---|---|---|---|
| A 独立审核 | 按当前唯一计划审 CR、检查服务器 live 状态 | 单一 verdict、blocker 分类、六条 failure_matrix、未验证清单 | 不启动新 physics、不改历史证据 |
| B 行为/示范 | 分析 A 的阶段证据和 CR12 handoff、现有 controller/recorder | 优先 family、具体控制假设、人工/keypose 路径、身份/资源预算 | 不盲扫参数、不用泄漏任务、不冒充真人 |
| C 数据/训练 | 审 recorder→acceptance→loader、split/norm/adapter | schema gap、source-group split、norm-only 对照、tiny-overfit/短训方案 | accepted=0 不启动正式 reverse 训练，不混失败/holdout |
| D 评估/发布 | 审 execute8/C20/public reverse/FAST、核实官方时限 | frozen protocol、paired 表、研究/发布门、提交约束 | 不混协议、不自动正式提交 |
| 集成负责人 | 汇总 A–D、处理冲突和授权 | 单一 reviewed execution plan、input hash、可写路径、GPU、预算、stop/rollback | 不争写 active pointers、不重复启动 |

A–D 可以并行只读设计；所有写入、运行进入新的唯一活动计划。作者和审核者分离，阶段完成后集中同步 index、Markdown handoff、machine JSON、lifecycle tests。

权限恢复后建议将本文提交为：

`docs/strategy/TRACK3_DATA_FIRST_MASTER_PLAN_20260907.md`

它是战略文档，不直接设为 next_plan。短期执行计划应由审核后拆解的工作包生成。

## 11. 内部排程与停损

知识库曾记录截止 **2026-09-17 23:59 JST**。本次公开网页查询没有找到可确认这一日期的官方结果，也没有成功读取最新官方 Slack 公告；以下仅以该日期仍有效为前提。D 必须先确认最新截止和提交方式。

| 建议窗口（JST） | 目标 | 未通过时 |
|---|---|---|
| 9 月 7 日 | CR 审核收口、A–D 并行只读设计 | 只修真实 blocker |
| 9 月 8 日 | 第一族第一条 gold demo | 自动方案无进展则安排真人输入 |
| 9 月 9–10 日 | 可行族 mini accepted、split/loader/norm QA、tiny-overfit | 查链路，不扩训练 |
| 9 月 11–13 日 | R6 短续训与 paired dev、补独立 source | 停低价值臂，不重复同质帧 |
| 9 月 14–15 日 | 候选冻结、sealed holdout、扩展 retention、FAST/offline | 不达标则保留旧 fallback |
| 9 月 16 日 | 格式和上传准备缓冲 | 不临时切架构 |
| 经复核的正式截止前 | 用户决定提交，检查上传结果/回执 | 不把计划完成等同提交完成 |

这些是团队内部目标，不代表本助手后台运行或未来自动交付。

停损规则：

- 无可信原始证据：停在 G0；
- 首批新诊断无阶段进展：停该自动方案，转真人/keypose或新结构假设；
- 无人类操作者：human_input_required；
- 仅一族可行：单族推进，其余族保留真实状态；
- tiny-overfit 失败：先查对齐、gripper、norm、adapter；
- 训练起点成功、独立起点失败：补 source/初态/接触多样性；
- reverse 改善、C20 退化：先审 norm-only 差异和 forward exposure；
- 截止前无达标候选：明确无新 Track3 改善，不用失败数据、fallback 或 ZIP 充当成果。

## 12. 最小进度看板

```text
evidence_commit:
active_plan:
audit_verdict:
new_gold_demos_by_family:
new_accepted_by_family:
unique_source_groups_train_dev_test:
attempted_and_accepted_yield:
first_blocking_stage_by_family:
training_step_and_checkpoint:
reverse_dev_paired_wins_losses:
sealed_holdout_status:
C20_reference_step0_candidate:
GPU_hours_and_resource_violations:
next_decision:
```

真正 KPI：独立成功示范数 → replay/accepted 数 → 独立 source 覆盖 → policy 闭环收益 → retention → 可部署候选。

tests passed、文档数、attempt 数和训练步数是活动指标，不是最终能力指标。

## 13. 来源与维护

所有仓库证据固定在本次锚点，方便下级 reviewer 复查：

- S1 CR 闭包：https://github.com/huangpenguin/AIROA-contest-final/blob/760dc65adc74193112692ab003e19a74d588ea9f/docs/reviews/axis_pi05/TRACK3_CR0_CR14_EXECUTION_CLOSEOUT_20260907.md
- S2 CR summary：https://github.com/huangpenguin/AIROA-contest-final/blob/760dc65adc74193112692ab003e19a74d588ea9f/data/reports/axis_libero_30k/track3_cr0_cr14_execution_20260907/summary.json
- S3 当前审核计划：https://github.com/huangpenguin/AIROA-contest-final/blob/760dc65adc74193112692ab003e19a74d588ea9f/docs/AXIS_PI05_TRACK3_CLEAN_LAUNCH_AND_DIVERSIFIED_CANARY_RECOVERY_INDEPENDENT_AUDIT_PLAN.md
- S4 AGENTS：https://github.com/huangpenguin/AIROA-contest-final/blob/760dc65adc74193112692ab003e19a74d588ea9f/AGENTS.md
- S5 MEMORY：https://github.com/huangpenguin/AIROA-contest-final/blob/760dc65adc74193112692ab003e19a74d588ea9f/MEMORY.md
- S6 计划索引：https://github.com/huangpenguin/AIROA-contest-final/blob/760dc65adc74193112692ab003e19a74d588ea9f/docs/AXIS_PI05_PLAN_INDEX.md
- S7 Markdown handoff：https://github.com/huangpenguin/AIROA-contest-final/blob/760dc65adc74193112692ab003e19a74d588ea9f/docs/AXIS_PI05_PROJECT_HANDOFF_20260831.md
- S8 machine handoff：https://github.com/huangpenguin/AIROA-contest-final/blob/760dc65adc74193112692ab003e19a74d588ea9f/data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json
- S9 Drive 知识库：https://drive.google.com/file/d/1lxWHHzQX36CEpL1t58EiaZFW_mdbKPQU/view

当前状态来自 S1–S3；现有模型/数据/资源边界来自 S4–S8；社区观察范围来自 S9。其余数量、预算、对照、优先级与排程是本文设计建议，不是已执行事实。

后续每次会话推进本项目，先重读仓库 next_plan/状态与知识库修改时间/最新变更段，明确报告找到与否、是否更新、哪些决策受影响。未查最新公告时明确规则新鲜度边界。没有设置后台持续监控。

主仓库统一为 `huangpenguin/AIROA-contest-final`；聊天用于决策、仓库用于版本化计划/证据、Drive 用于社区知识与本提案备份，三者不互相替代。
