# AXIS Pi0.5 Track 3 Teleop Recorder 修复与人工就绪计划

> 制定日期：2026-09-02（Asia/Tokyo）
> 执行输入：root `9dac308`，nested OpenPI `b58aff0`
> 当前状态：`track3_teleop_recorder_remediation_ready`
> 目标：把现有P2 recorder修成在当前headless环境可见、可控、可回放且fail-closed的人工采集工具；结束时停止在人工输入前。
> 不授权：真实人工accepted demo、P3 planner、P4/P5、LoRA/dense训练、FAST、ZIP、C20、E2、比赛GPU、上传或提交。

## R0. 证据与状态闭合（CPU only）

1. 锁定root `9dac308`、nested `b58aff0`、P0/P1 raw summaries与snapshot hashes；不改写历史raw evidence。
2. 新建correction summary，记录：
   - `p0_lock.json`实际SHA-256 `be12795b3d18d46f774022d817789a14d64a5e16c8e27bee58e256c29acb4ab5`；
   - 历史报告值`06c2d8...`为stale/incorrect；
   - P1 summary SHA-256 `62db697a1ff6afb5d157c9fc786c77eb17a0bdf1054edb060fea50b3f3fe211e`保持不变；
   - P1状态仍为`pending_video_review`；accepted demonstration=`0`。
3. 修正machine handoff与生命周期测试，先让clean tree上的全量CPU测试可复现；不得通过删除旧测试绕过。
4. 只读确认GPU3状态；R0不启动CUDA。

## R1. Recorder action/control state machine（CPU）

### R1.1 Action contract

将所有输入统一经过一个可单测的`next_action(key, keyboard_state)`或等价状态机：

- no-key、pause、camera/reset/save等control key的前六维为0，但gripper始终保持当前`-1`或`+1`；
- translation/rotation按键只改变对应一维，并保留当前gripper；
- `g`切换应在当前tick或明确记录的下一tick生效，行为必须有测试；
- 任何路径都不能产生gripper `0`、NaN或shape错误；
- deadline等待和键盘事件不能让同一个key重复执行不可控次数。

测试至少覆盖连续50个no-key ticks、pause/unpause、gripper两次toggle、camera控制、save、abort、reset及非法输入。

### R1.2 真正的reset与故障保存

- 绑定不与运动键冲突的reset键/按钮；reset必须重新加载锁定P1 full start state，清空当前candidate frames并记录reset count；
-异常、Ctrl+C、preview断开或predicate fail必须atomic写入`rejected/`或`diagnostic/`，不能丢失原因；
- CLI只允许P1 allowlist task `10/19/20`，并在启动前验证snapshot/BDDL/hash；
- GPU3存在外部PID或显存占用超过预注册门限时fail closed，不终止外部进程。

## R2. Headless实时预览与安全控制

当前环境没有X/Wayland，因此terminal-only模式不得宣称human-ready。实现localhost-only的低延迟web preview，或证明等价的headless可见方案：

- 默认只绑定`127.0.0.1`，不得对公网开放；显示front和wrist（并排或真正可切换）；
- 显示task index、明确reverse instruction、goal predicate、step、20Hz/lag、gripper、pause、reset count、当前success；
- browser keydown/keyup或按钮必须进入R1同一状态机；页面失焦/连接断开立即零前六维并保持gripper；
- UI提供Pause、Reset、Abort、Save Candidate，且Save只在predicate success时允许；
- 输出实际访问URL；远程场景给出明确SSH tunnel命令，本机不得使用shell尖括号placeholder；
- 若preview未连接或首帧未送达，simulator不得开始记录动作。

为P1 task10/19/20各生成20-frame settle H.264或等价时序review asset及contact sheet，记录实际codec/hash；不把PNG hash冒充video review。

## R3. Candidate、Replay与Promotion语义

1. predicate成功的人工轨迹先写`pending/`，绝不能直接进入`accepted/`。
2. 实现fresh-process replay verifier：从保存的start full state逐帧执行原7D actions，3次均要求reset goal false、end goal true、action finite、无异常；输出逐次summary和H.264。
3. 人工review sidecar至少记录operator、task/instruction匹配、对象/目标正确、无teleport/failure trimming、front/wrist视频确认与时间戳。
4. 只有`replay=3/3`且`human_review=approved`时，manifest builder才把episode标为accepted；promotion应为atomic且保留pending provenance。
5. tests必须证明`predicate success + replay pending`、`3/3 + human pending`、`2/3`都不能进入accepted manifest。

## R4. GPU3非人工canary与停止点

在GPU3自然空闲时只运行diagnostic，不产生accepted episode：

1. task19启动后连续20个no-key ticks：无崩溃、gripper全为`±1`、20Hz timing与preview首帧通过；
2. 做一次运动键、pause、camera/front-wrist显示、reset，验证reset后的start-state hash与锁定snapshot一致；
3. abort后轨迹只能进入`diagnostic/rejected`；fresh process再次启动无残留；
4. task10/20各做5个idle ticks，验证allowlist、BDDL/snapshot、preview与reverse goal false；
5. 运行全量CPU测试、py_compile、JSON、`git diff --check`，root/nested clean；GPU0不触碰，GPU1/2不训练。

完成后生成唯一、可直接复制的task19启动命令、localhost URL/SSH tunnel与键位表；operator使用安全默认值或普通字符串，命令中不得含`<operator-name>`。状态写为`track3_teleop_ready_for_human_after_remediation`并hard stop，等待人工审核。

## 后续人工顺序（本计划不执行）

修复获审后按难度逐条操作：task19 articulation → task10 support → task20 containment。每次只录一条candidate，完成后先3/3 replay和视频审核，再决定是否进行下一个task。三族全部通过后才能另建P3 planner计划。

## 允许的最终状态

- `track3_teleop_recorder_remediation_failed`
- `track3_teleop_preview_blocked`
- `track3_teleop_ready_for_human_after_remediation`
