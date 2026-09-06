# AXIS Pi0.5 Track 3：Acquisition Gate 与 Attempt Lifecycle 今夜修复计划

> 创建日期：2026-09-07（Asia/Tokyo）
> 唯一活动计划：是
> 审核输入：root `720733047a687de5cb456830467151c0102e4171`
> 执行源：`afabffdb2124a870dd16bd9e2fcf3ba9df246327`
> 审核结论：`remediation_required`
> 授权范围：BR0–BR11；先 code/fixture，全部通过后最多三条全新 CPU/no-renderer identity；之后 hard stop

本计划保留 AC 三条真实 `family_failure` 作为诊断证据，但不把 AC3–AC6 称为合规闭包。今晚允许连续推进较多工程阶段，目标是一次修完 partial sticky-stop、attempt 不可覆盖、完整 manifest、stage 三态、source/contact/progress gate，并在 clean source 上最多运行三族各一条全新 CPU/no-renderer canary。任一 shared infrastructure failure 立即停止后续阶段。

旧 AC/ER/TZ/Z5/AA5 identity 一律不得重跑。禁止 GPU0–3、official GPU、Teleop、generation、replay、accepted data、training、norm、FAST/export、candidate ZIP、upload 和 submission。

## BR0：冻结审计输入与历史结果

- 锁定 `7207330`、source `afabffd`、AC tracked files `38/38`、三条 raw/NPZ、partial observation、manifest/outer/final/heartbeat/cursor/closure/pointers 和本轮审核 summary。
- AC 原 namespace 永久只读；三条旧 AC canary 只保留为诊断性 `family_failure`，不得升级为合规 AC4 或 training data。
- 在新 BR namespace 记录派生修正；历史 AC3=`passed`、AC6 ledger=`running` 和旧 gate states 保持原样。

## BR1：把四类缺陷固化为回归

- 无 physics fixture 必须复现并锁定：partial 后错误继续、固定 attempt 1 可覆盖、NumPy `bool_ is False` 造成 source-release 恒失败、contact window 未限制在 acquisition barrier 前。
- 覆盖“stage 未进入→所有后续 gate=`not_evaluated`”，不得把缺失阶段写成 false/failed。
- 回归先失败、修复后通过；不得通过修改旧 raw 或放宽冻结阈值来过测试。

## BR2：单调 Attempt 分配与不可覆盖发布

- 每次顶层运行原子分配单调、不可复用的新 attempt ID；任何已存在 attempt 目录都必须 fail-close，禁止 truncate/unlink/overwrite。
- `current_attempt.json` 只在 active 期间指向运行 attempt；`latest_attempt.json` 只在闭包后以目标 hash 更新。
- 真实 fixture 验证 failed attempt→new attempt 后首个 attempt 的全部 bytes/SHA 不变；不得复用 AC attempt_0001。

## BR3：Partial、timeout 与 sticky infrastructure barrier

- atomic writer/timeout fixture 若留下 partial，先在本 attempt 内原样保留 raw bytes、SHA、来源和 failure classification，再关闭为 `infrastructure_failure`；不得清除后继续 physics。
- TERM→grace→KILL、no-descendant、singleton、周期 heartbeat 和 sticky stop 全部真实回归；sticky 状态必须在 child spawn 前阻止 BR9。
- cleanup 只能发生在后续全新 attempt 的临时 fixture namespace，不能让原失败 attempt 变成 pass。

## BR4：完整 Manifest 与最终状态机

- manifest 使用精确相对路径排除，不得因 basename 误排除 nested fixture heartbeat；所有 producer、fixture raw/log/heartbeat、phase ledger 都必须入 hash closure。
- final artifacts 按 producer→inactive cursor→manifest/outer→strict scans→tests→final validation→closed heartbeat→closure→pointer 顺序生成并交叉 hash。
- summary/CSV/final validation 的 BR11 状态必须最终写成 `passed_hard_stop` 或 `failed_hard_stop`，不得停留 `running`；strict JSON 同时报告 scan-time 与 current complete-set 口径。

## BR5：Stage-aware 三态 Gate

- 为三族建立显式 stage dependency graph；未进入 stage 的 contact/source-release/goal/retreat gate 均为 `not_evaluated`。
- family failure 由最早决定性已执行 gate 触发，不能把未执行尾部扩散成失败；报告必须区分 `failed`、`not_evaluated` 和 `unknown`。
- support/containment 与 articulation 的阶段序列分别在纯函数 fixture 中覆盖全 pass、每阶段 failure 和早停。

## BR6：Source、Goal 与 Acquisition Contact 修复

- 使用值比较而非 Python identity 判断 NumPy boolean；source-release 必须在 acquisition 成立后从 true 变 false并保持，goal 必须在目标阶段进入后连续 true。
- required contact window 只允许在 grasp/pull acquisition barrier 之前或其冻结窗口内累计；barrier 后才出现的 contact 不能倒填 acquisition pass。
- contact telemetry completeness、required contact、unrelated collision 分离计算；任一 schema/known-mask 缺失为 infrastructure failure。

## BR7：Progress、Clip 与 Saturation Runtime 复算

- 从 typed trace 逐步复算 EEF translation/rotation error、normalized action、controller clip、连续 saturation、progress window 和 stall；不得只信 child `stage_results` 或硬编码 metrics。
- 只读复算三条旧 AC trace，保留 support/containment/articulation 的 `155/149/180` steps、saturation `10/5/10`、source 全程 true、goal 全程 false。
- 阈值和 target 只能来自公开 model/state 与 BR profile lock；在看到新 physics 结果后不得修改。

## BR8：全矩阵 Fixture 与 Clean-source Release

- property/fixture 覆盖三族、stage 三态、late-contact rejection、source/goal连续性、clip/saturation/stall、typed no-pickle、identity/resume drift 和全部 finalizer 失败注入。
- targeted/full tests、strict JSON、py_compile、diff-check 通过后形成单独 clean source commit；记录 root/tool/OpenPI、command/profile 与三套 porcelain digest。
- 任一测试、partial、manifest、identity、process/GPU 问题均 hard stop，不进入 BR9。

## BR9：冻结三条全新 CPU Identity

- 仅在 BR0–BR8 全部通过后，一次性冻结 `br_support_remove_20260907_0001`、`br_containment_remove_20260907_0001`、`br_articulation_close_20260907_0001` 的 task/demo/seed/command/threshold/max-steps。
- 总上限 3、每族上限 1、固定 support→containment→articulation、无 retry；CPU/no-renderer，`CUDA_VISIBLE_DEVICES=""`、`JAX_PLATFORMS=cpu`、`MUJOCO_GL=disable`。
- 任何 identity 与历史集合重合、source drift 或 dirty root 均为 shared infrastructure failure。

## BR10：一次性 Cross-family Canary Matrix

- 每族最多启动一次；正常 `family_failure` 不阻止后续冻结 family，shared infrastructure failure/partial/timeout/unknown category 立即 sticky global stop。
- success 必须同时满足全部 stage、正确时序 contact barrier、无 clip/saturation/unrelated collision、source release、goal continuous true、finite typed trace、return 0 和完整 child closure。
- 失败轨迹永久 `training_eligible=false`；成功也仅为 `pending_replay_review`，不得 replay/promotion。

## BR11：派生诊断、完整闭包与 Hard Stop

- 独立按 family 汇总 success/family_failure/infrastructure_failure/not_evaluated，并对旧 AC 与新 BR trace 做只读差异诊断；不得从无 success 推断 evaluator/model 失败。
- 记录实际 identity、配额、所有禁止项、scan-time/current strict JSON、完整 manifest/outer/pointer/closure、tests、process/GPU 和 nested Git 边界。
- 无论结果如何均 hard stop，等待新的独立审核；不得自动进入 GPU camera、generation/replay、accepted data、training、FAST/ZIP、upload 或 submission。
