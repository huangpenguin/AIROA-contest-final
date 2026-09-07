# Track 3 SR0–SR13 独立审核（2026-09-07）

## 唯一结论

`remediation_required`

SR0–SR7 的代码修复、clean-source 身份和两条 support CPU/no-renderer 诊断轨迹可以保留；两条结果都是正常进程内的 acquisition `family_failure`，不是成功数据，也没有进入训练。但 SR13 在 producer 尚未结束时提前封口，SR7 并非所声明的结构不同策略，现有 Teleop recorder 又不能满足 SR9 声明的逐帧机器门。不得直接启动当前 handoff 中的 Teleop 命令，须先按新计划完成派生 evidence closure 与 recorder readiness。

## 审核边界

- audit input / execution record：`64f278c7f59dc70f17770e1980522d0313ccda27`
- clean-source / physics source：`5c5c74641ba39e41709e2c51ce576e7c065060fc`
- implementation/fix commits：`63b7261d4d321240323c3b862a5756d5a3f392c6`、`a064d52e8264a00340a883ebaf50cc74f20b56ef`、`5c5c74641ba39e41709e2c51ce576e7c065060fc`
- nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
- branch：`codex/axis-pi05-sg0-sg12-support-gold-20260907`，审核开始时相对 upstream ahead 5
- advisory strategy：`PARC_TRACK3_MASTER_PLAN_20260907.md`，SHA-256 `8c2f87ebdb4c3bfccfe1a2e80294ad3d23e300f76da869f8d7db13b9e52715f5`

## 独立成立的事实

- root 与 nested OpenPI clean；`LIBERO-plus` 既有五项 dirty 集合保持不变。SR cursor inactive、heartbeat final、supervisor PID 已退出，无 live Track3/Teleop/replay worker或 `.partial`。
- SR namespace `27/27` JSON 通过 duplicate-key rejection。attempt `0001` 的 SR0–SR4 通过；SR5 的 attempt `0001` 为 tool-SHA fail、attempt `0002` 为 current-pointer dirty fail，均发生在 env/physics 前并保留；attempt `0003` 在 clean root `5c5c746` 通过，tool blob SHA `c2795e847ca4f2a5c7aa4f40927c7c7194967da058010d0107f13900ebd1b775` 与该 commit 精确一致。
- attempt `0004` 有两个 fresh child，return code 均为 0，identity gate 全通过，分别执行 96/112 个 CPU/no-renderer physics step。两个 start state 都是 finite `float64[79]`，SHA 相同且符合固定起点；两个 trajectory SHA 与 child/manifest 一致、typed/no-pickle/finite，动作与 trace 不同。
- primary 与 alternative 都进入 `grasp_dwell`。各自 dwell 最后四帧 `contact_telemetry_complete=true`、`required_contact=false`、`demonstrator_contact=false`、source=true、goal=false；lift/place 未进入，保持 `not_evaluated`。动作 saturation 分别为 66/83，成功、pending、accepted、training eligible 均为 0。
- 这足以支持“冻结 runtime 在 acquisition barrier 未通过”的诊断性 `family_failure`。由于轨迹只保存派生 contact booleans、未保存原始 contact pair 名称，审核不把它扩大解释为对所有可能接触分类的可重算证明；失败轨迹不得训练。
- SR8 正确记录 `human_input_required` 且 recorder 未启动；SR9–SR11 实际未执行。没有 camera/replay 视频或待人工审核 episode。SR12 未启动 loader/norm/tiny-overfit/training；GPU replay、generation、ZIP、upload、submission 均未发生。
- 独立测试在生命周期编辑前为 SR targeted `10 passed`；为保护 GPU0 上的外部 PID，项目规范全量使用 `JAX_PLATFORMS=cpu .venv_axis_pi05/bin/pytest -q tests`，得到 `417 passed, 7 skipped, 23 warnings`。

## 必须修复的事项

### 1. attempt `0004` 的 closure 已过时

`run_sr6_sr7()` 在 SR8、SR12、SR13 之前调用 finalizer。producer/final manifest 因此只有 15 项；当前排除 finalizer 后实际有 18 项，多出的正是 `sr8_route.json`、`sr12_readiness/sr12_code_only_readiness.json` 与 `sr13_closeout.json`。旧 `closure.json` 的 `exact=true` 只代表提前封口时点，不能代表最终 attempt complete set。旧文件保留，不原地改写；新计划在派生 namespace 重算 closure，并修正 finalizer 的唯一时序。

### 2. SR13 的三态账本错误

`sr13_closeout.json.not_evaluated=[]` 与实际 SR9–SR11 未执行矛盾。顶层 summary 和人工 closeout 已诚实写成 `["SR9", "SR10", "SR11"]`，但不能用文字覆盖机器账本。旧 SR13 保留原样，在派生 correction 中显式冻结三项 `not_evaluated`。

### 3. SR7 不是结构不同 alternative

primary 与所谓 `lateral_first_v2` 的实际阶段顺序完全相同，都是 `precontact_xy → precontact_z → orientation_align → contact_descent → grasp_dwell`；后者只改变步长、clip、budget/tolerance。`topology` 字符串宣称 orientation 在 Z 之前，但 `prepare_support_identity()` 没有实现该顺序。该身份已消耗，不重跑、不返还；下一步停止自动策略，转人工可视控制。

### 4. 当前 Teleop 命令尚不具备 gold 资格

`track3_public_teleop --self-test` 虽通过 no-simulator 输入状态机，但当前 recorder：

- 默认 no-key gripper 为 `+1`，与 task10 forward terminal/release 后应保持 open 的起点语义不一致，且网页只有 toggle，操作者无法一眼确认 open/close；
- trajectory 不保存逐帧 raw contact pairs、required/demonstrator/source/goal/unrelated 三态，无法执行 SR9 所称“与 SR6 相同”的机器门；
- 没有本轮 attempt allocator、singleton supervisor、root/tool/OpenPI/command/CUDA identity、cursor/heartbeat 与完整 late-producer finalizer。

因此用户现在无需也不应自行运行旧 argv。先完成 UT0–UT3 code-only remediation；通过 clean-source release 后，才请求用户在线校准和一条 recorded candidate。

### 5. 忽略目录中的小型 raw 尚未进入 Git

attempt manifest 内两个 child 的 stdout/stderr、start-state 与 trajectory 共 8 项在 audit input commit 中未被 Git 跟踪，根级 supervisor stdout/stderr/PID 也未跟踪。它们在审核开始时均存在且 hash 与 manifest/child 一致。本审核不改字节，以 force-add 和新 `raw_evidence_anchor.json` 固定这 11 项，避免换 clone 后丢失；旧 manifest/closure 仍保持原样。

## 当前用户动作

现在没有 episode 需要人工视觉审核，也暂时不要启动 Teleop。下一执行任务先完成 UT0–UT3；看到机器状态 `ready_for_user_calibration` 后，用户只需保持约 15–30 分钟在线并回复“接管 Teleop”。浏览器、SSH 隧道、按键校准和保存条件见新活动计划的“用户接管步骤”。

## 生命周期处置

旧独立审核计划归档。唯一新活动计划为 `docs/AXIS_PI05_TRACK3_SR_CLOSURE_AND_TASK10_USER_TELEOP_GOLD_CANARY_PLAN.md`：先派生修正 SR closure、修复 finalizer 与 Teleop 逐帧门禁/显式夹爪 UI，形成 clean release；再由用户做不保存校准，最多录一条 task10 candidate。只有 machine success 才允许 GPU3 fresh-process exact-action replay `3/3` 与双视角审核包。accepted promotion、规模生成、训练/norm、FAST/ZIP、上传和比赛提交继续禁止。

生命周期最终验证：SR strict JSON `27/27`、audit namespace `2/2`、machine handoff `1/1`；targeted `18 passed`、项目规范全量 `425 passed, 7 skipped, 23 warnings`；py_compile、diff-check、无 partial、无 live Track3 worker、project GPU compute=0、public LIBERO/nested OpenPI clean，`LIBERO-plus` 原 dirty 保持不变。GPU0 的外部训练 PID 只读保留，GPU1–3 idle。
