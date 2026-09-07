# Track 3 SG0–SG6 独立审核（2026-09-07）

## 唯一结论

`remediation_required`

SG0–SG5 的 code/fixture、clean-source release 和 SG6 的 infrastructure fail-close 均可保留；attempt `0004` 的 TypeError、日志、manifest 与 hard stop 真实且闭合。但 `3e156dd` 只修复了首先暴露的参数签名，当前 runtime 还有三项会让 gold success 路径逻辑上不可达的阶段/contact 缺陷。因此不得直接重新 release/re-run SG6，须先按新计划完成 code-only remediation。

## 审核边界

- audit input / execution record：`55acd39c17d82c89b4f8e809b8f84d9623c3eb81`
- SG implementation：`710183a7983e300759c85e3c0a97e1afdbdb67f0`
- SG5/SG6 execution source：`d5c319b055431d2ec46681775438c91ad64a2be0`
- post-failure signature fix：`3e156dd6177e435316b721a79ff7a0da6d0359f8`
- nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
- branch/upstream：`codex/axis-pi05-sg0-sg12-support-gold-20260907`，审核时本地与 origin 均指向 `55acd39`

## 已独立通过的事实

- root 与 nested OpenPI clean；`LIBERO-plus` 的既有五项 dirty 状态未变。无 SG/Teleop/replay/MuJoCo/pytest worker；GPU0–3 无 compute process。
- SG namespace `37/37` JSON 通过 duplicate-key rejection；attempt 目录严格为 `0001–0004`，当前 pointer 为 inactive/hard_stop attempt `0004`，cursor inactive、heartbeat final、无 `.partial`。
- attempt `0003` 的 SG0–SG5 均为 passed；SG5 锁定 root `d5c319b`，其 tool SHA `b85694f6646e129e09bdf1366d09aecaf18ed31c92453030334e01c768e8c31c` 与该 commit blob 独立一致。
- attempt `0004` 的 producer/final manifest 均为 `8/8` bytes/SHA 精确，complete set 精确，outer hash 与 closure 通过。child 正常发布 `infrastructure_failure`，error 为 `compose_object_to_eef() got an unexpected keyword argument 'object_position_offset_m'`。
- 源码执行顺序证明异常发生在 `_configure_public_env()`、`ControlEnv` import/creation 与 `env.reset()` 之前。该 attempt 有 `1` 次 child launch，但 `0` 次 env creation、`0` physics step、`0` trajectory、`0` camera/replay/Teleop episode。
- `3e156dd` 为 `compose_object_to_eef` 增加 offset 参数并加入单元回归；该最小修复本身通过。定向测试独立复现 `11 passed`，项目全量独立复现 `399 passed, 7 skipped, 23 warnings`，py_compile 与 `git diff --check` 通过。
- 未发现 GPU、replay、Teleop、generation、accepted promotion、training、norm、新 ZIP、upload 或 submission；SG7–SG12 均未执行，应保持 `not_evaluated`。

## 必须修复的 success-path 缺陷

### 1. Acquisition contact window 使用了整段前缀

`required_contact_gate()` 当前检查 `samples[:barrier_index + 1]` 并要求每个样本都有 demonstrator contact。正常 precontact 阶段本就不应接触物体，因此任何从无接触接近到抓取的合法轨迹都会被早期 false 永久否决。计划要求的是 grasp barrier 结束前的冻结连续窗口，而不是从 episode 起点开始全 true。

### 2. 在 lift 之前错误要求 source 已释放

worker 在 `grasp_dwell` 结束时同时要求最近样本的 source atom 全为 false，失败就中止；但阶段顺序是 grasp_dwell 之后才 lift。物体完成夹持但仍在 plate 上是合法中间态，当前逻辑会在真正抬起前阻断 source release。

### 3. 合法 goal receiver 被标成 unrelated

任务目标是 bowl 放到 `main_table_akita_black_bowl_region`，实际 receiver body 是 `main_table`。当前 `OPERATOR` 只列 `plate_1` 且 target 为空，因此 bowl↔main_table 会落入 `unrelated_background_contact`。即使执行器完成合法放置，最终也会被 unrelated gate 拒绝。

### 4. “连续 goal”实际只有末帧布尔值

`goal_continuous_true` 只检查 `goal_final` 与 `env.check_success()`，没有冻结连续窗口。它不满足计划声明的 goal 稳定性合同，必须与 place/release/retreat 的阶段范围绑定。

这四项没有被现有 11 个测试覆盖。直接重跑最多得到另一条诊断 trace，不能得到合格 gold candidate，因而当前修复不足以释放 physics。

## “失败立即 stop”效率判断

SG6 当场停止是正确且高效的：它防止用已知坏代码继续消耗第2条 launch identity，也避免污染 evidence。低效的是把所有 infrastructure error 都做成同等级、必须新一轮人工生命周期才能进行任何 code-only 工作。

后续采用分级策略：dirty/identity/telemetry/GPU/partial/supervisor unknown 仍是 sticky global hard stop；可证明发生在 env 创建前且零 physics 的 deterministic code/setup exception 先立即停止并封存，但允许活动计划预授权的一次 code-only remediation 与 pre-env integration test，随后必须新 clean release，绝不自动重跑；正常行为失败走 family 分支。这样保留安全边界，也减少夜间空转。

## 当前人工 episode 队列

| 队列 | 数量 | 当前动作 |
|---|---:|---|
| SG6 trajectory | 0 | 异常发生在 env 创建前，没有轨迹 |
| camera / replay video | 0 | 没有 success candidate，未获 replay 资格 |
| Teleop recorded candidate | 0 | recorder 未启动 |
| pending gold review | 0 | 现在无需用户审核 episode |
| accepted / training eligible | 0 / 0 | 禁止训练 |

用户后续可能需要两种参与：自动路径仍失败时，在线接管一次 task10 Teleop；或 exact replay `3/3` 后，按 front/wrist 双视角卡片审核 candidate。当前两者都尚未到达。

## 生命周期处置

旧 Contact Truth / Support Gold Demo 计划已归档。唯一新活动计划为 `docs/AXIS_PI05_TRACK3_SUPPORT_RUNTIME_GATE_REMEDIATION_AND_GOLD_CONTINUATION_PLAN.md`：先修 contact roles、阶段窗口、pre-env 集成预检和分级停止合同，再形成 clean release；之后运行一条 corrected primary，只有正常且有进展的 family failure 才允许一条 alternative。真实 success 才允许 GPU3 exact replay `3/3` 与视觉审核包；自动失败才进入用户在线 Teleop。accepted、训练、新 ZIP、上传和提交继续禁止。

生命周期更新后的最终验证：SG JSON `37/37`、audit summary `1/1`、machine handoff `1/1`，定向 `19 passed`，项目全量 `407 passed, 7 skipped, 23 warnings`；py_compile、diff-check、无 live worker/GPU compute、nested OpenPI clean与 `LIBERO-plus` 原 dirty保持均通过。
