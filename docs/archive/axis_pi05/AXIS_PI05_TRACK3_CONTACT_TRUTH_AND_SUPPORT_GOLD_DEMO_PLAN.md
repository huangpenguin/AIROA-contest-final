# AXIS Pi0.5 Track 3：Contact Truth 与 Support Gold Demo 获取计划

> 创建日期：2026-09-07（Asia/Tokyo）
> 唯一活动计划：否（SG0–SG6 已执行并按 infrastructure failure hard stop；独立审核为 `remediation_required`）
> 审核输入：`0ce605cf33d33a0d670d190ecf7013d122d8a1a0`
> 策略参考：`PARC_TRACK3_MASTER_PLAN_20260907.md`，SHA-256 `8c2f87ebdb4c3bfccfe1a2e80294ad3d23e300f76da869f8d7db13b9e52715f5`
> 前置审核：`docs/reviews/axis_pi05/TRACK3_CR0_CR14_INDEPENDENT_AUDIT_20260907.md`（`remediation_required`）
> 后续审核：`docs/reviews/axis_pi05/TRACK3_SG0_SG6_INDEPENDENT_AUDIT_20260907.md`

本计划落实总体方案的“数据优先、一族先打通”。当前 KPI 不是再产生更多失败 trace，也不是立刻训练或再打一个 fallback ZIP，而是取得第一条真实、可重放、可视觉审核的 support reverse gold candidate。CR 六条历史 trace/identity 全部只读，不重跑、不改标签。

## 授权与硬边界

- 仅 support/task10；containment/task20 与 articulation/task19 不启动新 physics。
- 自动/keypose 路径最多 `2` 条全新 CPU/no-renderer identity，无同策略 retry。第2条必须在第1条为正常 family failure、且有可量化阶段进展时才允许，并在运行前冻结结构差异。
- 人工 Teleop 只能在用户在线明确接管后启动，最多 `1` 条 recorded candidate；禁止 agent 模拟键盘、无人值守运行或把理想 keypose 当实际 trajectory。
- 只有 CPU 或人工 candidate 满足真实 success gate，才允许 GPU3 exact-action replay，最多 `3` 次；GPU0–2、official GPU 均禁止。
- 本计划不授权 accepted promotion、规模化 generation、norm、LoRA/训练、FAST/export、新 ZIP、upload 或 submission。合格结果最多到 `pending_gold_review`，之后 hard stop 等独立审核。
- 任何 infrastructure failure、telemetry unknown、dirty source、identity/hash drift、partial residue、失联 supervisor 或 unrelated contact 立即 sticky global hard stop；不得用下一 identity 绕过。

## SG0：冻结输入与 failure matrix

1. 锁定 CR source/evidence/correction commit、六个 child/trace/action hash、43-byte raw partial、当前 root/OpenPI/LIBERO-plus 状态及总体方案 hash。
2. 在新 namespace 写 `failure_matrix.json`，逐条记录 earliest failed stage、steps、末端误差、saturation、source/goal、有效/无效遥测字段。CR required-contact 与 unrelated-contact 列必须标成 `unavailable_invalid_runtime_derivation`，不得依据旧全 true/全 false 数组推断物理接触。
3. 冻结 priority family=`support_remove`。依据仅是 support alternative 的 descend 末端误差 `0.0133621389 m` 最接近当前阈值；这不是 success，也不授权调阈值。

失败处置：输入/hash 不一致则 `blocked_input_drift` 并 hard stop；不修历史文件。

## SG1：真实 contact-v3 运行时合同

1. 从 MuJoCo contact pair、geom/body 名和既有 contact-v3 participant contract 派生：`demonstrator_contact`、`intended_receiver_contact`、`unrelated_background_contact`。
2. `contact_telemetry_complete` 只表示 schema/枚举/role resolution 完整；`required_contact` 必须来自 robot/gripper↔support object 的真实 pair；`unrelated_contact_runtime` 必须来自实际分类，三者禁止互相复制。
3. required contact 只统计 acquisition barrier 前的连续窗口；未进入 barrier 为 `not_evaluated`，零相关接触为 `failed`，unknown schema 为 `infrastructure_failure`。
4. 增加真实/no-physics fixtures：schema complete 但零接触必须失败；目标接触窗口通过；晚接触不能倒填；unrelated contact 必须失败；全 true surrogate 必须被拒绝。

失败处置：任一 role 无法解析或 fixture 不闭合，不启动 physics。

## SG2：实际 waypoint/microstep 与姿态合同

1. support controller 显式实现 `precontact_xy → precontact_z → orientation_align → contact_descent → grasp_dwell → lift → transport → place → release → retreat` 状态机。
2. XY/Z 分轴；每 tick 单独归一化 translation/rotation，冻结 component cap、microstep、dwell、stage budget、stall window 和 orientation tolerance。禁止只换静态 target 坐标却沿用旧策略名。
3. 记录每步实际 normalized action、physical delta、EEF pose/error、phase、source/goal/contact；rotation 路径必须在需要对齐时非零且 quaternion shortest-path 约定通过 fixture。
4. keypose 只定义目标，trajectory 必须是 simulator 真正执行的连续 state/action；不得存理想 waypoint 代替执行数据。

失败处置：action unit、rotation convention、stage transition 或 recorder schema 任一失败，回到 code/fixture；不消耗 identity。

## SG3：真实 failure-injection 与 finalizer 回归

必须实际执行而非写常量：after-seal mutation、missing nested log、bad outer hash、active cursor、leftover partial、dirty source、tool/command/profile drift、timeout TERM→KILL/no descendants、stale pointer。每个坏输入都必须被 finalizer/launcher fail-close，并保留独立 fixture result。

失败处置：任一注入未被拒绝，不进入 SG4。

## SG4：Teleop 与 replay machine-ready 修复

1. 冻结可执行 recorder 命令：

   ```text
   .venv_axis_pi05/bin/python -m tools.axis_pi05.track3_public_teleop --record --task-index 10 --operator operator-local --preview-port 8765
   ```

2. 在不启动人工录制的前提下执行 `--help`、`--self-test`、loopback-only preview、键位/无键 gripper、reset/abort/save、pending-only publication、端口占用和 GPU3 guard 回归。
3. 生成 machine handoff，含 exact argv、cwd、environment、task/start-state/model hash、浏览器地址、键位、目标 predicate、contact/success/abort 条件和输出目录；任何缺项不得标 `machine_ready`。
4. replay 命令与 acceptance state machine 也要 dry-run 校验；本阶段不录制、不 replay、不 promotion。

失败处置：修到 command 在目标环境可解析、自测通过；否则 hard stop，不能把问题留给人工操作时发现。

## SG5：clean-source release

1. 完成 SG0–SG4 后先形成 clean commit；锁定 root commit/tool bytes、OpenPI commit、profile/command hash、root/OpenPI porcelain 和 `LIBERO-plus` 原 dirty digest。
2. pre-spawn 和 child 内双重实时 clean-source/identity gate；attempt 单调分配且 immutable，周期 heartbeat、cursor、PID/PGID、timeout teardown 与 atomic producer 合同通过。
3. 定向、项目全量、strict JSON、py_compile、`git diff --check`、no-partial、process/GPU boundary 全通过才释放 SG6。

失败处置：不允许在 dirty expected/observed 一致时通过；修复必须新 attempt，不覆盖失败 evidence。

## SG6：support 自动/keypose identity 1

- 运行一条 fresh CPU/no-renderer support identity，使用 SG2 的真实分轴 waypoint、姿态与 contact-v3；固定 start-state、seed、command/profile hash和预算。
- success 要求：initial source true/goal false；所有阶段真实进入；required contact 连续通过；source 释放；goal 连续 true；无 unrelated contact/clip/unknown；finite typed no-pickle trace；process/identity 正常。
- 任何 gate 未进入保持 `not_evaluated`。进程正常但行为门失败为 `family_failure`；基础设施异常按全局 hard stop。

## SG7：有条件的 support identity 2

仅当 SG6 是正常 `family_failure`，且相对 CR support alternative 至少有一个冻结进展指标改善（更晚 earliest-failure stage、进入真实 contact、进入 grasp/lift、末端 EEF error 下降之一）时运行。

- 第2策略必须在读取 SG6 输出前已冻结，且在 waypoint topology、orientation/gripper sequence 或 cap 中至少一项有结构差异；不可只改 seed、ID、dwell 或 tolerance。
- trace/action 与所有历史及 SG6 去重；相同则 `no_effect_variant` 并停止。
- SG6 无进展时跳过 SG7，直接进入 SG8；不得盲跑第2条。

## SG8：自动失败后的人工接管分支

- 若 SG6/SG7 均无 success，发布 `human_input_required` machine handoff 并等待用户在线。等待本身不启动 simulator/GPU。
- 用户明确接管后，先做不保存 candidate 的画面/方向/抓手/abort 校准，再允许 `1` 条 task10 Teleop recording。agent 只维护 recorder、监控门和证据，不生成按键输入。
- 人工轨迹仍必须满足同一 source/contact/goal/action/telemetry 合同；人工标记不能覆盖 machine failure。
- 若用户当时不在线，SG8 标 `human_input_required`，继续完成 SG11 的 code-only readiness 后 hard stop。

## SG9：candidate 的 GPU3 exact-action replay

仅对 SG6、SG7 或 SG8 的真实 machine success candidate 执行：

1. GPU3 guard 与 identity/provenance 通过后，从 fresh process 恢复同一 start state，以原 action bytes 运行 `3/3` exact replay；不重规划、不改动作。
2. 保存 front+wrist 视频、per-step source/goal/contact-v3、action/state hash、帧数/频率和 replay outcome。
3. 三次都需 source→required contact→source release→goal 连续成立，无 unrelated contact、unknown、partial 或 identity drift。

失败处置：replay 非 `3/3` 则 candidate 保持 diagnostic/rejected，不进入 pending gold；不得挑最好一次。

## SG10：gold candidate 与视觉复核包

- 机器门全通过后只发布 `pending_gold_review`，包含原轨迹、3次 replay、双视角视频、provenance、source-group ID、所有 hash 与 known limitations。
- 用户/独立 reviewer 检查动作方向、真实抓取/接触、目标完成、异常碰撞、视频与 state/action 对齐。未复核或有疑点保持 pending/rejected，不能进入 accepted。
- 本计划中的 `accepted` 与 `training_eligible` 均保持 `not_evaluated`。

## SG11：G2/G3 code-only readiness（与人工等待可并行）

即使 SG8 等待人工，也继续完成不使用新数据的工程准备：

1. 定义 `raw → pending replay → pending gold review → accepted → training eligible` 的不可跳级 schema 和 atomic pointer；失败/diagnostic 轨迹永不进入 accepted manifest。
2. 定义 source-group split、去重、holdout 隔离、norm 只读 train split 的 validators；没有 accepted demo 时必须 fail-close，不生成伪 norm。
3. 对 synthetic fixture 做 recorder→loader shape/dtype/action/gripper/20Hz 对齐和 tiny-overfit CLI dry-run；不加载 R6 权重、不启动训练、不写 checkpoint。
4. 冻结下一阶段所需的 accepted 数量门、训练预算、step0/R6 对照、reverse dev/holdout 与正向 C20 retention，但数值需下一次独立审核后才可授权。

## SG12：统一闭包并 hard stop

- 汇总所有 attempt、未执行项、human status、identity 预算、success/pending/rejected 计数；缺失工作一律 `not_evaluated`。
- producer 全部 close/fsync 后生成 manifest/outer；随后 strict JSON、typed NPZ、CSV LF、tests、py_compile、diff、process/GPU、complete-set；再写 final validation、closed heartbeat、closure 和 pointer。
- 归档本计划并 hard stop，等待独立审核。不得自动 accepted、训练、打包、上传或提交。
