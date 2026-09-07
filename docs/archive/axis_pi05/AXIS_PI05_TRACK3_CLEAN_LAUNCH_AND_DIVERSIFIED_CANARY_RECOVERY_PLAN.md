# AXIS Pi0.5 Track 3：Clean Launch 与 Diversified Canary 今夜恢复计划

> 创建日期：2026-09-07（Asia/Tokyo）
> 唯一活动计划：否（CR0–CR14 已执行并 hard stop）
> 审核输入：root `474d36d7e36aa3f428c8fff3fbf38b8d5db03f05`
> 审核结论：`remediation_required`
> 执行状态：CR0–CR14 已执行；0 success；等待独立只读审核

本计划不改写 BR attempts `0001–0006`、AC/ER/EG/TZ 等历史 evidence 或 failure label。目标是先修复真正阻止可信 acquisition canary 的五个合同缺口，再利用夜间窗口运行有界 primary/alternative 策略，而不是继续只改 seed/profile ID。

旧 identity 一律不得重跑。GPU0–2、official GPU、无人值守 Teleop、generation、accepted data、training、norm、FAST/export、新 candidate ZIP、upload 和 submission 均禁止。GPU3 仅在 CPU trace 已独立满足全部 success gate 后用于固定 action 的 camera replay。

## CR0：冻结审核输入和消耗账本

- 锁定 `474d36d`、`cba0b12`、BR `88/88` tracked bytes、attempts `0001–0006`、43-byte raw partial、39-byte synthetic observation、三条 AC/BR 同 hash 轨迹和本轮 audit summary。
- 汇总所有已消耗 AC/BR/ER/TZ/Z5/AA5 identity、command hash 与 trace hash；任何复用或历史 trace exact duplicate 都 fail-close。
- 建立新的 CR namespace、单调 attempt allocator、singleton lock、active cursor 与周期 heartbeat；BR namespace只读。

## CR1：先写会失败的合同回归

- 回归必须证明：dirty root/OpenPI 在 `run_*canary` allocate/spawn 前被拒绝；不能通过把 dirty 状态同时写进 expected/observed 来过门。
- 回归必须证明：actual timeout partial 的 exact bytes/SHA 被保留；固定 sequence=`999` 的 synthetic 文件不能冒充 raw。
- 回归必须覆盖 manifest 生成时序、telemetry unknown→infrastructure、normalized action 不 double-normalize、旧/相同 trace 不计新策略。
- 回归先在旧实现上复现问题，再在新 CR 实现上通过；不得修改旧 BR tool 或旧 evidence 使测试变绿。

## CR2：真实 Partial 与 Sticky Stop 修复

- atomic timeout 若留下 `.partial`，立即 fsync并记录原路径/bytes/SHA，然后原子改名到本 attempt 的 immutable `raw_partial.bin`；不得重构内容或写固定替代 payload。
- 该 attempt 永久关闭为 `infrastructure_failure`。清理 active `.partial` 只用于防止残留，不得把同一 attempt 改为 pass，也不得释放 physics。
- 另开全新 attempt 验证正常 fixture；TERM→grace→KILL、no descendants、singleton 和 heartbeat sequence 必须为真实观测。

## CR3：两层 Manifest / Finalizer 闭包

- 全部 producer、raw、fixture log、test stdout/stderr、phase ledger、inactive cursor 和 closing heartbeat 生成后，才建立 producer manifest；明确列出仅因自引用而排除的 manifest/outer/final/closure/pointer。
- final validation 必须重算 producer manifest、outer、strict JSON、NPZ/CSV、tests、diff、process/GPU 和 active partial；closed heartbeat 与 closure 再 hash final/manifest/outer/cursor。
- pointer 最后原子发布并 hash closure。current complete-set scan 包含 final/closure/pointer，并明确报告 self-referential exclusions；禁止写未经集合比较的 `complete_set_rechecked=true`。
- 对每个 manifest 阶段做 after-seal mutation、missing log、bad outer、bad pointer、active cursor 和 leftover partial 失败注入。

## CR4：Normalized Action 单位合同

- `actions[:, :6]` 定义为实际传入 LIBERO controller 的 normalized input；component limit 直接检查 `abs(action)<=1`，不得再次除以 `0.05/0.25`。
- 若要审计物理 translation/rotation delta，必须由 executor 单独记录 `_m` / `_rad` arrays，再各自除冻结 controller output scale；不允许从 normalized input 猜回 raw preclip 值。
- controller clip/saturation 使用 preclip raw 或显式 boolean，不可仅凭 clip 后 action=`1` 推断；增加 axis-combined L2 与 component limit 的独立 fixture。

## CR5：Telemetry 与 Gate 分类

- contact telemetry schema 缺字段、长度不齐、known-mask unknown、role mapping unknown 或 `contact_telemetry_complete=false` 均为 `infrastructure_failure`，并在 child→parent 保持 sticky global stop。
- 正常、完整 telemetry 下未满足 required contact 才是 acquisition `family_failure`。未进入后续 stage 继续写 `not_evaluated`。
- source release、goal continuity、unrelated contact 与 required contact 分离；不能用 bool coercion 把 unknown 变 false。

## CR6：Launch-time Clean Source 与 Identity Fail-close

- 在 attempt allocation 前、每个 child spawn 前各执行一次 live gate：root porcelain为空、OpenPI porcelain为空、当前 tool bytes等于 `HEAD:path`、root/OpenPI commit、profile/command hashes和 `LIBERO-plus` 既有 dirty digest均精确。
- 任一 drift 在零 child spawn 时关闭为 shared `infrastructure_failure`；不得把 observed dirty 状态复制进 expected 后判 passed。
- child 内再次验证 parent-frozen clean identity；resume 必须复验 result/trace bytes与全部 identity，不仅检查 status。

## CR7：Clean-source Release

- 运行 CR1–CR6 的 property/fixture、旧 trace read-only recompute、strict JSON、py_compile、`git diff --check`、相关 targeted 和项目全量测试。
- 任一失败先在新 CR code/attempt 内修复并重新验证；禁止 physics。全部通过后形成单独 clean source commit，并在 commit 后再次验证 root/OpenPI clean和 tool `HEAD:path` exact。
- clean commit 之前的所有 physics 状态为 `not_evaluated`。

## CR8：冻结真正不同的两套策略

- 一次性冻结三族 primary：`object_frame_contact_aligned_v1`。目标来自可接触 geom/site 的 object frame，冻结 approach normal、contact dwell、grip/pull时序、normalized component cap、stage budget与停止阈值。
- 同时冻结三族 alternative：`waypoint_guarded_microstep_v1`。它必须改变 waypoint topology 与 action sequence：XY/Z分轴 microstep、pre-contact staging、较低 cap；articulation 还须改变 handle alignment/pull axis。只改 seed、ID、dwell 或阈值不算 alternative。
- 在看任何新 physics 输出前锁定六个 profile/command/seed/hash；离线证明 primary 与 alternative command/profile canonical hashes不同且不与历史集合重合。

## CR9：Primary Cross-family CPU Canary

- 仅在 CR7 clean release 与 CR8 freeze 均通过后，按 support→containment→articulation 各运行一次 primary；CPU/no-renderer，`CUDA_VISIBLE_DEVICES=""`、`JAX_PLATFORMS=cpu`、`MUJOCO_GL=disable`。
- 每族上限1、总上限3、无 retry。正常 `family_failure` 继续下一个冻结 family；任何 dirty/identity/timeout/partial/telemetry/process/unknown failure立即 sticky global stop。
- success 必须满足完整 stage、barrier 前连续 required contact、source release、goal continuous true、无 controller clip/不相关碰撞、finite typed trace 与完整 child closure；仅标 `pending_replay_review`。

## CR10：Family-local Failure 诊断与 Alternative

- 只对 CR9 的正常 `family_failure` 读取其最早决定性 gate，选择已冻结 alternative；不得现场改阈值、seed或第三套策略。
- approach/stall：使用分轴 waypoint/microstep；contact未建立：使用 contact-aligned staging；已接触但source未release：使用冻结 post-grasp lift/pull路径。策略差异必须反映在 command/profile/action序列。
- 每个失败族最多一条 alternative、总上限3；primary success 的族不运行 alternative。任何新 trace 与历史 AC/BR 或本轮 primary byte-identical/action-exact，分类为 `no_effect_variant` 的 family failure 并停止该族。

## CR11：Success 分支的 GPU3 Camera Replay

- 仅对 CPU gate success 且 child/parent/manifest全通过的 trace，在 GPU3 fresh process 做固定 start state + exact actions 的 camera replay；每条最多3次，首次失败即停止该 trace。
- replay `3/3` 后仍只到 `pending_human_review`，不得自动写 accepted。camera/renderer/identity异常为 shared infrastructure failure并停止所有后续 GPU3。
- GPU0、GPU1、GPU2 和 official GPU始终不可见；不得借本阶段生成新动作。

## CR12：无 Success / 人工配合分支

- 若 primary+alternative 仍零 success，生成逐族 human handoff：公开 task、start-state hash、相机、键位、动作界限、目标 predicate、required contact、stop/abort条件和 recorder命令。
- 无人值守时只准备并验证命令，不启动 Teleop physics；状态写 `human_input_required`。用户在线明确接管后，每族最多1条人工录制，且必须另经新审核授权。
- 不得把 scripted family failure、dry-run或未回放人工轨迹称为 reverse data。

## CR13：下一阶段 Machine-ready Dry-run Bundle

- 不运行 generation/training/ZIP，仅生成并校验条件式流水线配置：GPU3 replay→human review→accepted manifest→bounded LoRA→FAST candidate→ZIP CRC/hash→人工上传。
- 所有未获执行的计数写 `not_evaluated`；配置中锁定输入hash、GPU边界、最大样本/step、失败停止点和恢复命令，使下一次批准后无需重新设计。
- 旧 R6 fallback 只记录现有 hash/大小，不重打包、不冒充 Track3 improvement。

## CR14：派生汇总、完整闭包与 Hard Stop

- 汇总每族 primary/alternative 的最早 failure、行为差异、实际 identity配额、success/pending replay/human input和所有未执行项；不从零 success 推断 evaluator或模型原因。
- 完成两层 manifest/outer/final/closure/pointer、strict JSON、typed NPZ、CSV LF、targeted/full tests、py_compile、diff-check、PID/PGID/process/GPU与nested Git复核。
- 无论结果如何 hard stop，归档本计划并等待独立审核；不得自动进入 accepted data、generation、training、新 ZIP、upload 或 submission。
