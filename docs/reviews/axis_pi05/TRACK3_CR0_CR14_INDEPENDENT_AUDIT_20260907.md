# Track 3 CR0–CR14 独立审核（2026-09-07）

## 唯一结论

`remediation_required`

CR 的文件完整性、clean-source identity、六次 CPU/no-renderer 执行及其 `family_failure` 分层结论可以保留；但运行时把“接触遥测结构完整”写成“发生 required contact”，alternative 又没有实现冻结计划要求的分轴 waypoint/microstep/较低 cap，CR12 handoff 命令也不可直接执行。六条轨迹只能作为失败诊断，不能作为 contact、gold demo、accepted 或 training supervision。

## 审核边界

- audit input root：`0ce605cf33d33a0d670d190ecf7013d122d8a1a0`
- execution source：`02bbac24857168826557a0aa23b8698111e1be92`
- CR14 derived correction：`ec1f3f4f99aa8d74ddc45ac454914a54b99f1c4a`
- execution evidence/closeout commit：`760dc65adc74193112692ab003e19a74d588ea9f`
- nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
- 用户提供的总体方案：`PARC_TRACK3_MASTER_PLAN_20260907.md`，`24,677` bytes，SHA-256 `8c2f87ebdb4c3bfccfe1a2e80294ad3d23e300f76da869f8d7db13b9e52715f5`

总体方案只作为 strategy input。其“accepted reverse supervision 为零”“一族先打通”“gold demo 必须真实成功并经过 exact replay/视觉/机器门”的主线与仓库证据一致；其中外部知识库、网页查询和截止日期没有在本审核中独立复核，不提升为事实。

## 已独立通过的部分

- CR namespace `69/69` JSON 通过 duplicate-key rejection；attempt `0001–0004` 与 derived correction 均可读，原 attempt_0004 final validation 的 `failed` 原文未被覆盖。
- 原 attempt manifest `34/34`、correction manifest `2/2` 的 bytes/SHA 精确；两个 outer hash、closure 引用和 `latest_attempt.json` 指针精确。原 complete set `38/38`、correction complete set `6/6`。
- attempt_0003 的 actual raw partial 是 `43` bytes、SHA `e77dcf6b8603c76b6636bfc9b0d5b499dbf68e2e566d7745c30284b4d329bd3b`；39-byte sequence=999 synthetic observation 是不同文件。无活动 `.partial`。
- 六条 child 都是 fresh CPU/no-renderer process，identity gate、return code、child self-hash、NPZ hash 和 action hash通过；六个 trace/action hash彼此不同，也不与冻结 AC/BR 六条历史 hash 冲突。
- 六条均在 acquisition barrier 之前由 approach/descend 失败，source 最终仍 true、goal 始终 false。因此 `family_failure` 作为阶段失败分类仍成立；required-contact、source-release、goal-continuity 的后续门保持 `not_evaluated`。
- success/pending replay/exact/accepted/training/new ZIP 均为 `0`；CR11 GPU3 replay 未执行。旧 AC/BR/ER/EG/TZ namespace 未修改；无 live Track3 worker，无 GPU compute。
- 项目规范测试命令 `.venv_axis_pi05/bin/python -m pytest -q tests` 独立复现 `380 passed, 7 skipped, 23 warnings`。CR+recorder 定向为 `14 passed`。

## 必须修复的问题

### 1. required contact 数据不是真值

`track3_clean_launch_and_diversified_canary_recovery.py` 的 contact wrapper 只读取 `matrix.complete`，然后把同一个值同时写入 `required_contact`、`contact_telemetry_complete` 和 `telemetry_known`，并把 `unrelated_contact_runtime` 固定为 false。六条轨迹因此出现不可能的模式：从第一步起到最后一步，required contact 全为 true（合计 `966/966`），恰好等于 telemetry-complete 样本，而 unrelated contact 永远为 false。

这不影响 acquisition barrier 之前的阶段失败，但它使所有物理 contact/unrelated-contact 证据不可用。后续必须从 MuJoCo contact pair 和 contact-v3 role category 真实派生 demonstrator↔object/handle、intended receiver 与 unrelated background，不能从 schema completeness 推断接触。

### 2. alternative 只有目标偏移，不符合冻结策略

冻结计划要求 `waypoint_guarded_microstep_v1` 具有 XY/Z 分轴 microstep、pre-contact staging、较低 cap，并对 articulation 改变 handle alignment/pull axis。实际 patch 只对 `_stage_target` 返回不同静态坐标，没有 waypoint 状态机、microstep 调度或单独较低 cap；六条 action 的 rotation L2 最大值又全部为 `0.0`。hash 不同只能证明字节/动作不同，不能证明声明的结构策略已执行。

### 3. CR12/CR13 不是可运行 handoff

CR12 写出的命令是 `python -m tools.axis_pi05.track3_public_teleop --task <id>`。项目 recorder 实际要求虚拟环境、`--record` 与 `--task-index`；当前命令在默认 Python 缺 NumPy，即便进入虚拟环境也会因未知参数 `--task` 失败。CR13 只有阶段名列表，没有冻结命令、输入 hash、预算和 fail-close 条件，不能称为 machine-ready。

### 4. 若干回归声明只是常量

CR3 的 after-seal mutation、missing log、bad outer、active cursor、leftover partial 五项，以及部分 CR6 child recheck，是直接写入 `true` 的报告字段，不是执行后的 failure-injection 结果。下一阶段必须把这些关键门变成会实际制造坏输入并断言 fail-close 的测试/fixture。

## 六条失败矩阵与优先族

| family / strategy | steps | 最早失败 | 末步 EEF error | saturation | source/goal | 审核处置 |
|---|---:|---|---:|---:|---|---|
| support primary | 153 | descend | 0.034770 m | 8 | true / false | diagnostic only |
| support alternative | 154 | descend | 0.013362 m | 7 | 最终 true / false；中间 false 3步 | 第一优先，但非成功 |
| containment primary | 149 | descend | 0.039906 m | 5 | true / false | diagnostic only |
| containment alternative | 150 | descend | 0.044763 m | 4 | true / false | diagnostic only |
| articulation primary | 180 | approach | 0.075132 m | 12 | true / false | diagnostic only |
| articulation alternative | 180 | approach | 0.080258 m | 180 | true / false | 停该自动结构 |

support alternative 离 descend tolerance 最近，因此按总体方案选择 support 一族进入第一条 gold demo 获取；不把中间 3 步 source false 当作 acquisition，因为没有进入 grasp/contact barrier，且 source 最终重新变 true。

## 生命周期处置

旧 CR 独立审核计划归档，不修改 CR 原始 evidence。唯一新活动计划为 `docs/AXIS_PI05_TRACK3_CONTACT_TRUTH_AND_SUPPORT_GOLD_DEMO_PLAN.md`：先修 contact truth、真实 failure-injection、alternative/Teleop 合同，再最多运行两条 fresh support-only CPU/no-renderer identity。自动路径仍无成功时转 `human_input_required`；只有真实候选 success 才允许 GPU3 exact-action replay。训练、规模化生成、新 ZIP、上传和比赛提交继续禁止。

生命周期更新后的最终验证：CR strict JSON `69/69`、audit summary `1/1`、machine handoff `1/1`、定向 `22 passed`、项目规范全量 `388 passed, 7 skipped, 23 warnings`；`git diff --check`、无 live Track3 worker、GPU compute=0、nested OpenPI clean 与 `LIBERO-plus` 原 dirty 集合保持均通过。
