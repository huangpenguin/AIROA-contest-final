# AXIS Pi0.5 Track 3：Cross-Family Acquisition Recovery Canary 计划

> 创建日期：2026-09-06（Asia/Tokyo）
> 唯一活动计划：否（已完成并归档）
> 审核输入：root `4bbfe7eb7fdb404085f39712aae94572c9d0fbce`
> 已审核 EG evidence：`5d20e25d3890e937c0278ea2656236449b82898f`
> 审核结论：`approved`
> 授权范围：AC0–AC6；最多三条全新 CPU/no-renderer physics identity；之后 hard stop
> 最终独立审核：`remediation_required`；诊断性 family failure 有效，但 lifecycle/gate 合同需修复

本计划在 EG attempt-specific evidence lifecycle 获批后，恢复最小物理进展：只针对 support、containment、articulation 三族此前停在 acquisition 前后的失败，先做旧 trace 的只读诊断与 code/fixture 修复，再按固定顺序每族至多运行一条全新 CPU/no-renderer canary。成功只进入 `pending_replay_review`，不等于 accepted data。

禁止重跑任何旧 TZ/ER identity、Z5 nominal、AA5 conservative 或既有 EF/EG fixture 并冒充新结果。禁止 GPU0–3、official GPU、Teleop、generation、replay、accepted data、training、norm、FAST/export、candidate ZIP、upload 和 submission。

## AC0：输入、身份与配额锁

- 锁定 `4bbfe7e`、EG audit/attempt_0005/root closure、三条 ER4 raw JSON/typed NPZ、对应 public HDF5/model/snapshot/BDDL hashes、nested OpenPI 与 `LIBERO-plus` baseline。
- 三族新 profile、task/demo、seed、命令、最大 steps 与顺序在任何新 physics 前一次性写入 immutable profile lock；总 CPU physics identity 上限为 `3`，每族最多 `1`，不允许 retry。
- 实现代码与测试形成新的 clean source commit；parent 必须在 child 启动前冻结 root/tool/OpenPI/profile/command、三个 porcelain digest 与 CPU/no-renderer 环境。任何 drift 为 shared infrastructure hard stop。

## AC1：旧 ER trace 只读 acquisition 诊断

- 对 support/containment/articulation 的既有 `155/149/180` step typed traces逐步复算 phase、EEF translation/rotation error、normalized action、连续 saturation window、contact pairs、source/goal truth 与停止原因。
- 分离 telemetry enumeration、required physical contact、controller clipping/saturation、stage progress 与 source release；不得继续使用硬编码 `action_saturation=passed` 或仅以 `contact_complete` 代表 demonstrator↔object/handle 接触。
- 诊断必须提出一个预先冻结、跨 family 可解释的 acquisition contract；若只能靠查看新 canary 后调 threshold/target，AC2 前 hard stop。

## AC2：Acquisition contract 与 code/fixture gate

- support/containment 使用 demonstrator↔object 连续接触窗口；articulation 使用 demonstrator↔handle/region 连续接触窗口。stage 未进入时所有后续 gate 为 `not_evaluated`。
- approach/descend target、OSC normalized translation/rotation scale、progress/stall window、gripper command、acquisition barrier 与 source-release gate 必须来自 public model/state 和冻结 profile，不得引用隐藏 Track3 task 信息。
- `controller_clip`、`action_saturation`、required/unrelated contact 全部从 runtime telemetry 计算；finite、typed no-pickle trace、source=true/goal=false 初态为强制门禁。
- 纯函数、临时模型和无 physics fixture 覆盖三族 pass/fail/not_evaluated、dirty/command drift、normal family failure return 0、singleton/heartbeat/timeout/TERM→KILL/no-descendant 与 sticky infrastructure stop。

## AC3：Clean-source preflight 与 supervisor release

- 从 committed clean source 运行 CLI/worker/finalizer 回归，验证唯一 PID/PGID、周期 atomic heartbeat、attempt-specific immutable result、完整 resume identity 与 family/shared failure 分类。
- 任何 import、CLI、identity、manifest、timeout、partial、process/GPU scan 或测试问题均为 shared infrastructure failure，禁止启动 AC4。
- AC3 只做 code/fixture，不启动 ControlEnv，不消耗三条 physics identity。

## AC4：三族各一条 fresh CPU/no-renderer canary

- 固定顺序 support→containment→articulation；每族恰好零或一条新 identity，最多三条。正常 `family_failure` 不阻止其他未运行 family；shared infrastructure failure 立即 sticky global hard stop。
- 参数、threshold、target、seed 与代码在首条启动后全部冻结；不得依据中间结果修改后继续剩余 family。
- canary 通过必须同时满足：全部已定义 stages 到达、required contact 连续窗口、无 controller clip/saturation、无 unrelated collision、source release、goal 连续 true、finite typed trace、正常 child return 0 和完整 attempt closure。
- 失败/部分轨迹永久 `training_eligible=false`；成功轨迹也只标为 `pending_replay_review`，不得写成 exact/accepted。

## AC5：Cross-family classification

- 按 family 独立汇总 `success`、`family_failure`、`infrastructure_failure` 与未运行的 `not_evaluated`；不得把一个 family 的正常失败扩散为其他 family 失败。
- 只有至少一个正常 success 才可在报告中建议后续 replay 计划，但本计划仍不执行 replay、generation、GPU camera 或数据 promotion。
- 全部 family failure 也不是 evaluator/模型失败结论；只表示冻结 executor/profile 未通过 acquisition-to-goal 合同。

## AC6：Attempt-specific finalizer 与 hard stop

- 复用已获批 EG attempt-specific immutable finalizer语义：producer fsync→inactive cursor→manifest/outer→strict JSON/NPZ/CSV→process/GPU scan→targeted/full tests→final validation→closed heartbeat→checkout-stable closure。
- 记录实际 physics identities、每族状态、`exact/pending/accepted` 计数和全部禁止项；缺失/未运行项必须为 `not_evaluated`。
- 无论结果如何均 hard stop，等待新的独立审核；不得自动进入 replay、generation、training、FAST/ZIP、upload 或 submission。
