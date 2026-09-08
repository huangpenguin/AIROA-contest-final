# AXIS Pi0.5 Track 3：SR Closure 修复与 Task10 用户 Teleop Gold Canary 计划

> 创建日期：2026-09-07（Asia/Tokyo）
> 唯一活动计划：是
> 审核输入：`64f278c7f59dc70f17770e1980522d0313ccda27`
> SR physics source：`5c5c74641ba39e41709e2c51ce576e7c065060fc`
> 前置审核：`docs/reviews/axis_pi05/TRACK3_SR0_SR13_INDEPENDENT_AUDIT_20260907.md`（`remediation_required`）
> 战略参考：`PARC_TRACK3_MASTER_PLAN_20260907.md`，仅作 advisory input

本计划只解决当前最短关键路径：先把 SR late-producer/Teleop 合同修到可审，再由用户在线取得最多一条 task10/support reverse candidate。旧 SR/SG/CR/BR/AC/TZ physics identity、failure label、raw、manifest 与 closure 全部只读；不得重跑或覆盖。

## 授权与总边界

- 仅 task10 / support-remove；不运行 task20 containment、task19 articulation 或新的自动 SE(3) identity。
- UT0–UT3 为 code/fixture/证据工作，不启动 simulator/GPU。只有 UT0–UT3 全通过并形成新的 clean-source release，才可进入用户在线 UT4/UT5。
- 用户是唯一控制输入源；agent 不发送、模拟或回放键盘/浏览器控制事件。校准不保存 candidate；正式 recorded candidate 最多 1 条。
- 只有 UT5 machine success 才允许 GPU3 做 UT6 exact-action camera replay，固定 3 个 fresh process、3/3；GPU0–2 与 official GPU 禁止。
- 最远状态为 `pending_gold_review`。不授权 accepted promotion、规模生成、norm、LoRA/训练、FAST/export、新 ZIP、upload 或 submission。
- GPU0 有外部作业时只读保护；不得杀进程、抢卡或把 replay 改到 GPU0–2。每次运行仍重新检查 GPU3。

## UT0：输入、raw anchor 与 live 边界

1. 锁定 audit input `64f278c`、physics source `5c5c746`、nested OpenPI、SR `27/27` strict JSON、两条 child/NPZ/start-state、旧 15-entry manifest/outer/closure、SR8/SR12/SR13 late producers及本审核的 raw anchor。
2. 核对 root/OpenPI clean、LIBERO-plus 既有 dirty digest、无 live SR/Teleop/replay/MuJoCo worker、旧 supervisor PID inactive、cursor inactive、heartbeat final、无 `.partial`。
3. 新建独立 namespace 与单调 attempt allocator；不得写旧 SR namespace。任一 hash/身份/partial/进程不确定为 global hard stop。

## UT1：派生 SR closure correction

1. 不修改旧 `attempt_0004` 的 manifest/outer/closure/SR13，生成新的 correction namespace。
2. 对旧 attempt 当前 producer set 重算 expected/observed complete set，必须覆盖 18 项，明确列出旧 15 项与 3 个 late producers；验证 11 项小型 raw anchor bytes/SHA/Git persistence。
3. 派生三态账本固定 SR9、SR10、SR11=`not_evaluated`；success/pending/accepted/training=`0/0/0/0`。
4. producer close 后依次生成 correction manifest、outer hash、closure；任何 late artifact、缺失 raw 或 hash drift 均停止。

## UT2：finalizer 与 lifecycle 回归

1. 修复 SR 工具：只允许所有 producer phase 完成后 seal；manifest 必须递归包含嵌套文件，finalizer 文件用相对路径精确排除，禁止 basename 误排。
2. `not_evaluated` 必须从实际 phase ledger 派生，禁止硬编码空列表；未运行阶段不得写 false/pass/zero-risk。
3. 真实 fixture 覆盖：seal 后 late producer 被拒绝；先写全部 producer 再 seal 为 exact；重复 finalization fail-close；旧 SR attempt 只读。
4. 形成 code-only clean commit与 exact tool/command hash；本阶段不释放旧自动 physics。

## UT3：Teleop gold-readiness 修复

在不启动 simulator 的 fixture/self-test 中完成：

1. 新 supervisor/attempt identity：root/tool/OpenPI/command/task/snapshot/BDDL/CUDA/GPU3 policy、singleton PID、atomic cursor/heartbeat、TERM→KILL/no-descendant、partial sticky stop与两层 finalizer。
2. task10 起点夹爪由公开 demo 的 release transition 复算。默认必须显式为 open；网页显示 `OPEN`/`CLOSED`，提供独立 Open/Close 控件，禁止只靠 toggle 猜状态。
3. 网页同时显示 front/wrist、当前控制轴与符号、paused/connected、step/Hz/lag、source/goal、required/demonstrator/source-receiver/goal-receiver/unrelated contact、candidate gate；断连立即清空 held keys并安全暂停。
4. 每个 post-step 持久化 actual action/state、raw contact body/geom pair（或等价可重算表）、contact role、source/goal 三态、EEF/gripper、front/wrist、timestamp；typed NPZ `allow_pickle=False`，字符串用固定 Unicode dtype。
5. success 必须满足：初始 source=true/goal=false；dwell 连续 required contact；lift 后 source 连续 false；goal 连续 true且 `env.check_success()`；无 unrelated/unknown/action violation；所有阶段有真实 post-step evidence。仅 final success boolean 不足。
6. 校准/正式录制分离：校准 reset 后必须清空 action/video/gate buffers；Save 只有在 machine gate 显示 candidate success 时可产生 `pending_replay`。predicate fail/abort/disconnect/timeout 写 rejected/diagnostic，不进入 pending。
7. `--self-test`、fixture、strict JSON、targeted、全量、py_compile、diff-check 与 clean release 全通过后，状态才可写 `ready_for_user_calibration`。

## UT4：用户在线、不保存校准

只有用户在本任务明确回复“接管 Teleop”后执行。启动器先报告 exact argv、GPU3 guard、preview URL 和当前 attempt ID；用户未确认已看到两个实时画面前不记录正式 candidate。

校准流程：

1. 若用户与服务器不是同一台机器，在用户本地新终端建立隧道：`ssh -N -L 8765:127.0.0.1:8765 <user>@<server>`；端口被占用时由执行器给出新端口，不自行猜测。
2. 浏览器打开 `http://127.0.0.1:8765/`，确认 FRONT 与 WRIST 都是实时 task10 场景，状态显示 `connected=true`、`paused=true`、gripper=`OPEN`。
3. 执行器解除校准暂停后，用户每次只短按一个方向：`W/S = ±X`、`A/D = ±Y`、`R/F = ±Z`；`U/O = ±Rx`、`I/K = ±Ry`、`J/L = ±Rz`。以画面和 HUD 的 axis/sign 双重确认，不凭记忆连续长按。
4. 测试显式 Open/Close 后恢复 Open；测试 `P` pause、`Z` reset。方向、夹爪或画面任一不一致，立即 `P`/`X`，记录 calibration failure，不进入正式录制。
5. 校准通过后按 `Z` 回到 exact start，机器必须确认 frame/action buffers 为 0、source=true、goal=false、gripper=OPEN。该过程不计 recorded candidate。

## UT5：唯一一条用户控制 Task10 candidate

1. 用户从 bowl 上方接近，WRIST 用于对准、FRONT 用于高度与碰撞判断；先保持 Open。
2. 缓慢下降到 bowl 两侧，HUD 出现 demonstrator↔bowl required contact 后再 Close，并保持至少冻结连续窗口；无接触不得抬升。
3. 闭合稳定后垂直抬升，直到 source 从 plate 连续释放；若 bowl 未随夹爪移动，立即 abort，不用视觉愿望覆盖机器门。
4. 搬运到 main-table 合法目标区域上方，先降至稳定支撑，再 Open release，最后向上 retreat；避免碰撞 plate、其他物体或桌面穿透。
5. 只有 HUD 同时显示 goal 连续 true、`env.check_success=true`、unrelated/unknown=false、candidate gate=passed 时才按 Enter/Save。否则按 X abort；不得为了保留机会把失败保存成 pending。
6. 无论 success、行为失败、断连或 timeout，最多消耗这一条 recorded candidate；原子保存 raw 后 hard stop进入相应分支。

## UT6：GPU3 exact-action replay（仅 UT5 success）

- 同一 start-state bytes、同一 action bytes、同一 task/model/controller，在 GPU3 三个 fresh process 中逐条 exact replay；不重规划、不补动作、不改 renderer/controller。
- 三次都必须通过 source→contact→lift/source-release→goal continuous、unrelated/unknown、finite/action、identity/partial门；生成 front/wrist H.264 与逐帧机器账本。
- 任一失败即 candidate=`rejected_replay`，不得自动补录或重放第四次；hard stop等待审核。

## UT7：用户视觉审核包（仅 UT6 3/3）

以固定格式交给用户：

```text
Episode ID:
Task / family: task10 / support_remove
Start-state SHA / action SHA:
Machine success gates:
Exact replay: 3/3
Front video: <absolute local link>
Wrist video: <absolute local link>
请人工选择: approve_visual | reject_visual | needs_rerecord
原因标签: collision | wrong_object | bad_grasp | unsafe_motion | goal_not_visually_stable | other
备注:
```

`approve_visual` 只把状态推进到 `pending_gold_review`，不在本计划内 accepted。用户未回复时保持 `human_review_required`，不得代选。

## UT8：等待人工期间的数据链 code-only bundle

- 校验 raw→pending_replay→pending_gold_review→accepted→training_eligible 的真实实现、obs_t/action_t/obs_t+1 对齐、source-group dedup/split、20 Hz、camera/state/action/gripper adapter。
- 对 accepted=0 的当前状态，loader/norm/tiny-overfit 命令必须实际 fail-close；可生成命令与 schema dry-run，不计算 norm、不启动训练。
- 准备后续 G2/G3 的 machine-readable checklist，但不把失败或 pending 数据列入训练 manifest。

## UT9：统一闭包并 hard stop

汇总所有 attempt、用户是否接管、calibration/recorded candidate 数、success/replay/human-review/pending/accepted/training 及每个未执行阶段；producer 全部关闭后再生成 manifest/outer/closure。重跑 strict JSON、targeted、项目规范全量（GPU0 外部占用时 `JAX_PLATFORMS=cpu`）、py_compile、diff-check、无 partial、PID/GPU/root/OpenPI/LIBERO-plus 边界；形成 clean commit并切换为独立审核计划。

## 失败分支与效率规则

- UT0–UT3 infrastructure 失败：立即停运行；若明确是 env 前、零 physics 的 deterministic code bug，本计划允许一次 code-only 修复与新 clean release，但不自动进入 UT4。
- UT4 校准失败：不消耗 recorded candidate；保留诊断、修 UI/映射后重新 clean release，等待用户再次明确接管。
- UT5 行为失败：不 retry；下一审核再决定人工 keypose+controller 还是另一个 source，不在本计划盲试。
- UT5 recorder infrastructure 且无法确定已发多少 physics step：按 global hard stop，不把机会“返还”。
- UT6 任一 replay 失败：拒绝 candidate，不重录、不训练。
- 用户离线时只完成 UT0–UT3/UT8 code-only；不得冒充人工输入。这样夜间可继续推进工程准备，又不会浪费唯一真人轨迹。
