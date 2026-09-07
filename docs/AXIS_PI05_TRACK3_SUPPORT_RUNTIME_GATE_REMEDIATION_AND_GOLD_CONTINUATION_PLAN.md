# AXIS Pi0.5 Track 3：Support Runtime Gate 修复与 Gold Continuation 计划

> 创建日期：2026-09-07（Asia/Tokyo）
> 唯一活动计划：是
> 审核输入：`55acd39c17d82c89b4f8e809b8f84d9623c3eb81`
> SG6 执行 source：`d5c319b055431d2ec46681775438c91ad64a2be0`
> 已有签名修复：`3e156dd6177e435316b721a79ff7a0da6d0359f8`
> 前置审核：`docs/reviews/axis_pi05/TRACK3_SG0_SG6_INDEPENDENT_AUDIT_20260907.md`（`remediation_required`）
> 策略参考：`PARC_TRACK3_MASTER_PLAN_20260907.md`，仅作为 advisory input

本计划继续“先取得一条可信 support reverse gold demo”的主线，但先修复独立审核发现的 success-path 不可达问题。旧 SG attempt `0001–0004`、failure label、日志、manifest 和 closure 全部只读；禁止把已有 TypeError attempt 当成物理 family failure，也禁止原地重跑或覆盖。

## 当前事实与人工队列

- SG6 attempt `0004` 启动了一个 fresh child，但在 `ControlEnv` 创建前因 Python 参数签名异常结束；`launch_attempt_count=1`，`environment_created_count=0`，`physics_identity_count=0`。
- 当前没有 SG trajectory、camera frame、replay video、Teleop recording、`pending_gold_review` 或 accepted episode；因此现在没有需要用户人工审核的 episode。
- 只有产生 machine-success candidate 且 GPU3 exact-action replay `3/3` 后，才发布视觉审核包。用户人工 Teleop 是失败后的采集路径，不是对现有 SG6 的审核。

## 授权与边界

- 仅 support/task10。containment/task20、articulation/task19、旧 SG/CR/BR/AC/TZ identity 均不得运行。
- 修复后最多执行 `1` 条 corrected primary CPU/no-renderer physics identity；仅在其为正常 `family_failure` 且达到冻结进展条件时，才可再执行 `1` 条预冻结结构不同 alternative。不得同策略 retry。
- 人工 Teleop 最多 `1` 条 recorded candidate，且必须由用户在线明确接管；agent 不产生键盘输入，不无人值守录制。
- 真实 CPU/Teleop success 才允许 GPU3 exact-action replay，固定 `3/3`。GPU0–2、official GPU 禁止。
- 本计划不授权 accepted promotion、规模生成、norm、LoRA/训练、FAST/export、新 ZIP、upload 或 submission。最远状态为 `pending_gold_review`，随后 hard stop。

## 分级停止策略

“立即停”保留为每次运行的安全动作，但不再把所有错误都等价成整条夜间流水线永久停摆：

1. `global_hard_stop`：dirty source、identity/hash drift、unknown telemetry、GPU 越界、partial/manifest 不闭合、失联 supervisor/PID、无法判定是否进入 physics。立即停全部运行，只允许独立审核。
2. `runtime_lane_stop`：确定发生在 env 创建前、零 physics step 的 deterministic code/setup exception。立即停该 child、原子封存 evidence；本计划预授权一次 code-only 修复与测试，但不自动重跑。必须形成新的 clean-source release 才能重新进入 physics。
3. `family_outcome`：进程与遥测正常的行为门失败是 `family_failure`，按 SR7/SR8 路由，不升级为 infrastructure。

任何错误如果无法证明属于第2类，按第1类处理。attempt 计数与 physics identity 计数必须分别记录，不能删除失败 attempt 来“返还”历史。

## SR0：输入、历史与 live 边界锁

1. 锁定 `710183a`、`d5c319b`、`3e156dd`、`55acd39`、nested OpenPI、旧 SG `37/37` JSON、attempt_0004 `8/8` manifest、outer/closure 与 child/log hash。
2. 复核 root/OpenPI clean、LIBERO-plus 原 dirty digest、无 live SG/Teleop/replay worker、GPU compute empty、cursor inactive、无 `.partial`。
3. 新 namespace 和 attempt allocator 单调追加；旧 SG 文件不可修改。任一漂移为 `global_hard_stop`。

## SR1：contact role 语义修复

1. 从冻结 BDDL 显式建立四类角色：subject=`akita_black_bowl_1`、source receiver=`plate_1`、goal receiver=`main_table`、demonstrator=robot/gripper。
2. 分开记录 `demonstrator_contact`、`source_receiver_contact`、`goal_receiver_contact`、`unrelated_background_contact`；合法的 bowl↔plate 和 bowl↔main_table 接触不得标成 unrelated。
3. schema incomplete、body/geom role unknown 仍是 infrastructure；telemetry complete 不能推出任何物理接触。
4. 用实际 runtime 函数执行正负 fixture，而非报告常量。

## SR2：阶段时序与连续窗口修复

1. acquisition barrier 只看 `grasp_dwell` 末尾冻结的连续 `stable_ticks=4` 窗口；precontact 的零 demonstrator contact 不得让 barrier 永久失败，barrier 后的接触不得回填。
2. `source_released` 只能在 lift 已进入后评估；grasp_dwell 时 source 仍 true 是合法状态，不能在 lift 前强制 false。
3. `goal_continuous_true` 必须基于 place/release/retreat 后冻结连续窗口及 `env.check_success()`，不能只取一个末帧。
4. 未进入的阶段和门一律 `not_evaluated`；unknown 是 infrastructure；正常未达门是 family failure。
5. 构造一条不启动 simulator 的阶段序列 fixture：precontact 无夹爪接触 → dwell 连续夹爪接触且 source true → lift 后 source false → main_table 合法接触且 goal 连续 true。该序列必须可达 success；另做 unrelated、unknown、late-contact 负例。

## SR3：pre-env 集成预检

1. 将 SG6 的 asset load、model XML resolution、state bundle、anchor 读取、object→EEF/transport/destination 组合和 stage 构造提取为可单独调用的 preparation 函数。
2. 使用真实锁定资产运行 `prepare-only/no-env` 集成检查，覆盖此前漏掉的参数签名和数组/键/schema；明确断言未 import/create `ControlEnv`、未运行 physics。
3. 检查每个 stage target 有限、预算之和与 profile 一致、transport offset 实际被 stage target 使用、姿态误差非零路径可执行。
4. 任何 preparation error 只能 code-only 修复，不消耗 physics identity。

## SR4：身份、计数与停止合同

1. child result 无论成功或异常都写出 `environment_created`、`physics_steps`、`physics_identity_consumed`、`launch_attempt_id`、`failure_stage`。
2. parent 在 spawn 前与 child 内双重校验 clean commit/tool/OpenPI/command/profile/CUDA/LIBERO-plus digest；release pointer 解析必须精确。
3. 对三类停止策略做真实 process/fixture 回归：pre-env exception 允许 code-only remediation；identity/telemetry/partial unknown sticky global stop；正常 family failure 进入条件路由。
4. timeout 必须 TERM→KILL、无 descendant、关闭 cursor/heartbeat，并将 partial residue 作为失败证据保留或显式登记后清理。

## SR5：clean-source release

1. SR0–SR4 的 strict JSON、targeted、项目全量、py_compile、`git diff --check`、manifest/outer/closure、无 partial、process/GPU 边界全部通过。
2. 形成新的 clean commit，重新计算 tool SHA 和 exact command/profile hash；不得复用 `d5c319b` 的 SG5 release。
3. release 前再次检查 live PID/cursor/GPU；任何漂移 `global_hard_stop`。只有本阶段通过才释放 SR6。

## SR6：corrected primary support identity

- 运行一条 fresh CPU/no-renderer task10/demo0 primary identity。固定 start state、seed、command/profile、budget 与 clean source。
- 成功需同时满足：初始 source true/goal false；阶段顺序真实进入；dwell 末尾 required demonstrator contact 连续通过；lift 后 source 连续 false；place 后 goal 连续 true；目标 receiver 接触合法；无 unrelated/unknown/clip；finite typed no-pickle trace；identity/process 正常。
- 只允许原子发布实际 simulator post-step state/action。keypose 不是 trajectory，失败 trace 不可训练。

## SR7：有条件 alternative

仅当 SR6 为正常 `family_failure`，且相对 CR/SG 历史至少改善一项冻结指标（更晚 failure stage、进入 acquisition、进入 lift/transport/place、末端 EEF error 下降）时，运行一条预冻结的结构不同 alternative。相同 action/trace hash、只改 seed/dwell/tolerance 或无进展均停止自动路径。SR6 为 infrastructure 时不得进入 SR7。

## SR8：自动路径路由

- 任一 CPU candidate 真实 success：进入 SR10，不自动执行 Teleop。
- 两条自动结果均无 success，或 SR6 无资格运行 SR7：发布 `human_input_required` handoff；若用户不在线，继续 SR12 的 code-only readiness 后闭包。
- infrastructure/global stop：不进入后续运行阶段；只完成安全证据闭包。

## SR9：用户在线 Teleop（条件执行）

1. 用户明确接管后，先进行不保存 candidate 的画面、方向、夹爪、reset/abort 校准。
2. 最多记录 `1` 条 task10 candidate 到 pending，agent 只维护 recorder、证据和停止门，不发控制键。
3. Teleop 轨迹必须通过与 SR6 相同的 source/contact/goal/telemetry/action 合同；人工判断不能覆盖 machine failure。

## SR10：GPU3 exact-action replay

仅对 machine-success 的 SR6/SR7/SR9 candidate：GPU3 guard 通过后，从同一 start state 在三个 fresh process 中按原 action bytes exact replay `3/3`；不重规划、不改动作。三次都必须复现 source→contact→source release→goal 连续成立，且无 unrelated/unknown/identity/partial drift。失败即拒绝 candidate，不自动补录。

## SR11：视觉人工审核包

只有 SR10 `3/3` 通过才生成每条 episode 的双视角审核包，并以以下固定格式交给用户：

```text
Episode ID:
Source / start-state / action SHA:
Machine success gates:
Exact replay: 3/3
Front video: <absolute local link>
Wrist video: <absolute local link>
请人工选择: approve_visual | reject_visual | needs_rerecord
原因标签: collision | wrong_object | bad_grasp | unsafe_motion | goal_not_visually_stable | other
备注:
```

用户视觉通过后仍只到 `pending_gold_review`，不在本计划内 promotion 为 accepted。

## SR12：数据链 code-only readiness

可在等待用户在线时执行，但不得使用失败轨迹或开始训练：验证 raw→pending_replay→pending_gold_review→accepted→training_eligible 状态机、source-group split/dedup、obs-action-next_obs 对齐、loader dry-run、norm/tiny-overfit CLI fail-close。没有 accepted demo 时 norm/training 必须拒绝启动。

## SR13：统一闭包并 hard stop

- 汇总历史和新 attempt、`launch_attempt_count`、`physics_identity_count`、success/replay/pending/accepted/training 计数及所有 `not_evaluated` 项。
- producer 关闭后生成完整 manifest，再生成 outer hash 与 closure；strict JSON、targeted、项目全量、py_compile、diff-check、无 partial、process/GPU、root/OpenPI/LIBERO-plus 边界全通过。
- 归档本计划并 hard stop，等待独立审核。不得自动 accepted、训练、ZIP、上传或比赛提交。

## 夜间最短关键路径

```text
SR0→SR5 code-only 修复与 clean release
  → SR6 corrected primary
      ├─ success → SR10 replay → SR11 人工视觉包
      ├─有进展 family_failure → SR7 alternative → SR8
      └─无进展 family_failure → SR8 human handoff
  → 等用户时并行完成 SR12 code-only readiness
  → SR13 hard stop
```

任何需要用户输入的节点都不得由 agent 代答；但用户离线不阻止已授权的 SR12 code-only 工作和最终证据闭包。
