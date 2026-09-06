# Track3 BR0–BR11 独立审核（2026-09-07）

审核结论：`remediation_required`

审核输入为 root/evidence commit `474d36d7e36aa3f428c8fff3fbf38b8d5db03f05`、BR execution root `80d98a8471509581971335fb627d7b6faa4544c1`、后续 clean-source commit `cba0b12c578412d741b6e071c09b19caf3cbe405`、当前 tool SHA `5f244b943d90d66b9722006991d9cb101820f82b6f31e34e164299e6095b4081` 与 nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`。审核从 raw、Git object、source code 和当前资源边界独立重算，不接受 BR 自报 `passed` 作为充分依据。

## 1. 经独立确认的有效事实

- BR namespace 的 Git-tracked files `88/88` 与 `474d36d:path` 字节一致；当前 tool 与 `cba0b12:path` 字节一致。BR namespace 当前 `71/71` JSON、连同 machine handoff `72/72` 均通过 duplicate-key rejection。
- attempts `0001–0006` 均保留，没有覆盖。attempt `0005` 的 BR0–BR7 与 attempt `0006` 的 BR8 是 code/fixture-only clean-source 记录，没有再次消耗 physics identity；attempt `0006` 记录 root/OpenPI clean。
- attempt `0004` 的三个 parent/child process return code、identity comparison、raw NPZ/CSV、pointer 指向的 final validation/closure hash 均可复核。三条运行均应继续保留为诊断性 `family_failure`，`training_eligible=false`；不批准 success、exact、accepted 或 training data。
- 当前 root 与 nested OpenPI clean；`LIBERO-plus` 原有五项 dirty 保持不变。没有 live Track3/pytest/generation/replay/training worker，GPU compute process 为空。未发现旧 identity 重跑、GPU0–3 compute、Teleop、generation、replay、accepted data、training、norm、FAST/export、新 candidate ZIP、upload 或 submission。
- 独立复跑 BR 相关测试为 `16 passed`；项目全量为 `362 passed, 7 skipped, 23 warnings`。这证明当前工程回归为绿，不证明下列未被测试覆盖的执行合同成立。

## 2. 必须修复的合同缺口

1. `run_br9_to_br11()` 只读取历史 BR8 `status=passed`，在 allocate/spawn 前没有再次调用 `clean_source_gate()`。`_run_br_canary()` 又把当时 dirty checkout 同时写入 expected 与 observed identity，故 `clean_root=false` 仍能得到 identity passed。提交信息声称“enforce clean source before BR canaries”，但实际 launch path 没有实现该门。
2. 六个 attempt 的 manifest 均不是其声称的完整 producer set。attempts `0001/0002/0005` 在 BR5–BR7 之前封口，分别漏 4/5/4 个文件；`0003/0006` 漏 final validation；`0004` 在 heartbeat/cursor/final/closure 前封口并漏 4 个文件。`manifest_current_complete_set_rechecked=true` 因而不是从实际完整集合得出的结论。
3. attempt `0002` 实际保留的 timeout raw partial 是 `43` bytes、SHA `e77dcf6b8603c76b6636bfc9b0d5b499dbf68e2e566d7745c30284b4d329bd3b`，内容 sequence=`13`；它因 `*.partial` 排除规则不在 manifest。另一个被称为 `raw_partial_preserved_as_observation` 的文件是代码固定写入的 `39` bytes、sequence=`999`、SHA `4c6ec1...`，不是该次实际 raw partial。历史两者均保留，但不能把后者当作前者的 raw provenance。
4. telemetry schema/known-mask 不完整会使 gate `failed`，随后 `_br_cpu_worker()` 把任何 gate failure 一律归为 `family_failure`。这与 BR6 声称的“unknown schema is infrastructure failure”矛盾，会允许后续 family 在共享 telemetry 故障后继续。
5. trace 中的 `actions` 已是传给 LIBERO 控制器的 normalized OSC input（legacy executor 直接保存 `closed_loop_action()` 的 `[-1,1]` 输出）。BR7 又把这些值除以 `0.05/0.25`，形成 double normalization，三条均报告 `normalized_action_abs_max=20.0`。component limit 应直接在 normalized actions 上检查；物理位移/旋转必须使用单独数组和单位。
6. BR 三条 NPZ 与 AC 对应三条逐字节相同：support `3537623d...`、containment `186019af...`、articulation `9a1db8e5...`，steps 仍为 `155/149/180`。尽管 profile ID 改名，这三次没有产生新的行为轨迹或新的 reverse 成功数据，并且发生在 `clean_root=false` 的 attempt `0004`，不能批准为合规 BR9–BR10。

因此，上个任务只能确认“代码/夹具修复有部分有效成果，执行已 hard stop”，不能确认“BR0–BR11 合规完成”。历史 failure labels、attempt bytes、dirty-source correction 和 raw partial 均不得改写；未执行的 replay、accepted data、训练与新 ZIP 继续是 `not_evaluated`，不是 false success。

## 3. 下一计划

原只读审核计划已归档。唯一活动计划为 `docs/AXIS_PI05_TRACK3_CLEAN_LAUNCH_AND_DIVERSIFIED_CANARY_RECOVERY_PLAN.md`，按 CR0–CR14 执行。

今晚先用 regression 修复 launch-time clean gate、真实 partial provenance、两层 finalizer、telemetry infrastructure 分类和 normalized-action 单位；形成 clean source commit 后，最多对三族各运行一个 primary 与一个结构上不同的 alternative CPU/no-renderer canary，总上限 6。seed/profile 改名或 byte-identical trace 不算 alternative。正常 `family_failure` 只隔离本族；dirty source、timeout、partial、telemetry unknown、identity drift 或进程异常为 sticky global infrastructure stop。

若产生 success，才允许 GPU3 做相同 action 的 camera replay，每条最多 `3/3`；通过后仍仅为 `pending_human_review`，不自动 accepted。若无 success，则生成可供用户人工配合的 task/动作/相机/停止条件 handoff，但不在无人值守时伪造 Teleop。今晚不授权 GPU0–2、official GPU、generation、accepted data、training/norm、新 candidate ZIP、upload 或 submission；另行生成下一阶段 replay→data→LoRA→ZIP 的 machine-ready dry-run bundle，以缩短下一次审核后的启动时间。

机器摘要：`data/reports/axis_libero_30k/track3_br0_br11_independent_audit_20260907/summary.json`。

## 4. 生命周期更新后校验

BR namespace、本轮审核 summary 与 machine handoff 共 `73/73` JSON 通过 duplicate-key rejection；定向回归 `24 passed`；项目规范全量 `370 passed, 7 skipped, 23 warnings`。BR tool 与新增 lifecycle test 的 `py_compile`、`git diff --check`、引用检查、process/GPU 边界均通过。nested OpenPI clean，`LIBERO-plus` 原有五项 dirty 保持不变；root 仅包含本轮预期 lifecycle 变更，并在提交后复核 clean。
