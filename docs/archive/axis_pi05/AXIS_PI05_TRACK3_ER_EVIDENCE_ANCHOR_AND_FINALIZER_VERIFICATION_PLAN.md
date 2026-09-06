# AXIS Pi0.5 Track 3：ER 证据锚定与 Finalizer 验证计划

> 归档日期：2026-09-06（Asia/Tokyo）
> 归档状态：EF0–EF4 completed; independently audited `approved`; hard stop
> 原计划输入：root `c2d9f23bb0bbe13a0860aeb4fecf3a75c932ae0d`
> EF0 anchor commit：`38635fbf08e4e8df8c4877ddcb6b2b2ec8b9b118`
> 下一权限：`new_reviewed_plan_required`

本计划只修复 ER0–ER5 审核发现的证据持久化、resume identity、contact gate 和 finalizer 验证缺口。三条 ER4 identity 已消耗，原始 raw/NPZ/status、旧 TZ 证据和 R6 fallback 均只读；不得重新运行或改写。
不授权任何新physics identity或旧identity重跑。
不授权 GPU0–3、official GPU、Teleop、generation、replay、training、candidate ZIP、upload 或 submission。

## 0. EF0：ER 原始证据逐字节锚定

- 先锁定本次审核 summary、ER namespace 的 29 个现存文件、26-entry manifest、outer hash、3 个 raw/NPZ、旧 TZ closure 与 fallback/source ZIP。
- 对 29 个 ER 文件逐项复算 bytes/SHA；把现存字节原样 force-add 到 Git，禁止重生成或规范化旧 JSON/CSV/log/NPZ。
- 新建 derived verification，记录旧 namespace 中 24 个已跟踪文件和 6 个本地 NPZ 的不同持久化状态；不得把未跟踪误写为已跟踪。
- EF0 commit 后重新 checkout/read/hash，任何差异立即 hard stop。

## 1. EF1：完整 identity 与 resume fail-close（code/test only）

- parent 必须把冻结的 root/tool/OpenPI/profile/command、root clean 状态、OpenPI porcelain 和 `LIBERO-plus` baseline digest 显式传给 child；child 不得用当前观测值回填 expected 值。
- resume 除自哈希/profile ID 外，必须复验完整 identity、command hash、trace bytes/SHA 和 terminal category；任一 drift latch 为 shared infrastructure stop。
- 真实临时 Git repo/fresh subprocess 回归覆盖 root dirty、OpenPI drift、`LIBERO-plus` baseline drift、command drift、trace drift及合法 resume。

## 2. EF2：Supervisor 真实路径回归（无physics）

- 用专用 fixture child 实测 file lock、唯一 PID/PGID、周期 atomic heartbeat 至少两次更新、正常 family-failure return 0、timeout 的 TERM→grace→KILL、无后代和 sticky stop。
- 第二次入口必须读取 sticky state 并在启动 child 前停止；禁止用已有结果路径清空 failure。
- 回归产物只进新的 EF namespace，不覆盖 ER4 cursor、heartbeat、state 或 raw。

## 3. EF3：物理 gate 语义补全（离线/code only）

- 把 telemetry/schema 完整性与 physical required-contact completeness 分成两个三态 gate；零相关接触不能通过 required-contact gate。
- support/containment 显式检查 demonstrator↔object 连续接触窗口；articulation 检查 demonstrator↔handle/region 接触。未进入相应阶段必须为 `not_evaluated`。
- controller saturation、unrelated contact、known mask 与 typed no-pickle trace 保持 runtime-derived；fixture 覆盖 pass/fail/not_evaluated，不启动 ControlEnv。

## 4. EF4：真实 finalizer 与 hard stop

- 顺序固定为 producers close/fsync→inactive cursor→producer manifest/outer hash→strict JSON/NPZ/CSV 与 fallback SHA/CRC/layout→真实 process/GPU scan→targeted/full tests→final validation→closed heartbeat。
- `final_validation.status` 必须纳入 ZIP expected bytes/SHA/CRC/layout、process scan、GPU boundary、targeted/full return code、manifest/outer hash和无 partial；禁止硬编码 live PID 空列表。
- 将所有小型 evidence、测试结果和最终 validation 跟踪入 Git；commit 后再次 strict parse/hash/diff-check，并写 checkout-stable 派生 closure。
- 无论通过或失败均 hard stop，下一权限为 `new_reviewed_plan_required`。不运行任何 ER/TZ identity，不产生训练数据或新 ZIP。

## 5. 实际收尾

- EF0–EF4 已按顺序完成；新 EF namespace 记录 code-only identity/contact fixture、fallback inventory、真实 process/GPU scan、targeted/full tests、final validation 和 closed heartbeat。
- 独立审核结论为 `approved`；旧 ER/TZ identity 未重跑，GPU compute、physics、generation、replay、training、candidate ZIP、upload 和 submission 均未执行。
