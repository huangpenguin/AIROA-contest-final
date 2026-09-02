# AXIS Pi0.5 Track 3 公开终态、Teleop 与 Planner 恢复计划

> 制定日期：2026-09-02（Asia/Tokyo）
> 执行输入：root `1e71ec5`，nested OpenPI `b58aff0`
> 当前状态：`track3_teleop_input_pending_human`
> P0/P1 已完成；P1 canary 采用 support、containment、articulation 三个公开 family。activation task16 的双来源稳定性失败已单独保留；当前停在 P2 人工 keyboard 输入前。
> 唯一目标：用公开、可追溯的完整 simulator state 建立真实 20 Hz reverse demonstrations；达到 pilot 门禁后停止等待审核。
> 本计划不授权：LoRA/dense训练、FAST、ZIP、C20、E2、比赛 GPU、上传或提交。

## 0. 执行原则

1. 保留并只读引用旧 T0/T1 40-task candidate manifest；不得把 candidate spec 计作 accepted episode。
2. 只使用主办方已公开的 Track 3 高层规则、公开 LIBERO 任务/数据及通用机器人控制方法；不得读取或利用泄漏脚本的内部 task 细节。
3. accepted episode 必须是 `control_freq=20` 环境真实执行得到的前视/腕视/state/action，不得倒放 frames 或直接取反动作。
4. 任何人工 state edit、goal-at-reset、teleport、失败后裁剪、未经验证的 qpos transplant 都不能进入训练集。
5. GPU0与外部任务不触碰。P0仅CPU/网络；P1–P4最多使用GPU3做 simulator/render/R6公开正向 rollout；GPU1/2保持给其他实验但本计划不得启动训练。
6. 独立目录 `LIBERO-plus` 的既有 dirty 文件不可修改、stash、reset或提交；新实现只进入 root clean worktree。

## P0. 官方原始 HDF5 与 provenance lock（CPU/网络）

### P0.1 输入冻结

锁定并写入 machine-readable summary：

- root/nested commit与dirty状态；
- R6 raw/folded/FAST/QUANTILES hashes及正分ZIP hash；
- 旧 T1 candidate manifest/hash、32/8 split、accepted=`0`；
- canonical LeRobot 10 Hz revision与LIBERO环境/package/MuJoCo/robosuite versions；
- 当前独立 `LIBERO-plus` dirty porcelain列表的只读快照。

### P0.2 官方数据下载

用锁定 commit 的官方 `Lifelong-Robot-Learning/LIBERO` 下载脚本或其公开数据链接，把原始 HDF5 放到新目录：

`data/raw/public/libero_official_hdf5/`

要求：

- 只下载覆盖40个公开 candidate base tasks所需的公开 suite；不得下载/探测 hidden Track 3；
- 临时文件用 `.partial`，完成后 atomic rename；支持断点续传；
- 记录来源URL、official repository commit、下载命令、UTC时间、bytes、逐文件SHA-256；
- 校验 HDF5 可读、`data/demo_i/states`、`actions`、`rewards/dones`以及 attrs 中实际存在的 task/BDDL/model/init元数据；不存在的字段必须记 `missing`，不得补造；
- 使用官方 benchmark API 将 demo file 映射到公开 suite/task/BDDL；task、BDDL或model不一致即 fail closed。

P0 输出：download manifest、schema inventory、task mapping、source license/provenance report。下载不使用GPU。

### P0 hard gate

至少3个base templates、3个operator families各有一份可读取的完整 `states` trajectory及明确task映射，才进入P1。否则状态为 `track3_public_terminal_source_blocked`并停止。

## P1. 完整终态资格验证（GPU3 simulator only）

### P1.1 两种公开来源

逐条使用且分开统计：

- A `official_hdf5_terminal_state`：从公开demo的成功后缀选择最早稳定forward-success state，`states[-1]`仅为首个canary；
- B `live_public_forward_success_capture`：若A因XML/package/controller版本不匹配失败，用当前锁定环境运行R6公开正向任务，在环境首次`check_success()==true`时立即保存完整`sim.get_state().flatten()`、model XML hash、controller state与观测。

不得将A与B悄悄混合。每个snapshot包含source、suite/task/demo/seed、BDDL/init/model XML、environment versions与所有hash。

### P1.2 Restore 与稳定性门禁

对每个candidate：

1. fresh-process创建对应公开环境并加载正确XML/BDDL；
2. restore完整state，`sim.forward()`并重新生成observation；
3. forward goal必须为真，拟议reverse goal必须为假；
4. 以20 Hz执行20帧gripper-hold/no-op settle；object/fixture不得坠落、穿模或明显漂移，forward goal仍为真；
5. 保存前后完整state/predicate、qpos/qvel和两相机H.264；
6. fresh-process重复3次，状态恢复、predicate和稳定性必须3/3通过。

若restore后robot处于容器内、碰撞或不适合作为reverse policy起点，执行安全的`retreat-to-neutral`闭环：保持forward object/fixture predicate不变，让末端离开物体并回到公开工作空间安全位，随后重新保存完整state并再做3/3验证。不得只替换robot或object qpos。

### P1 梯度门禁

- canary：3个stable snapshots，覆盖3 families；
- expansion：至少6 templates、2 scenes；
- 任一来源连续3次版本/restore mismatch：停止该来源并写根因，不靠tolerance绕过。

通过canary后状态为 `track3_terminal_snapshot_canary_passed_pending_review`；执行者可在同一计划内继续P2，但不得越过P2 hard gate。

## P2. 20 Hz Teleop recorder 与三族 canary

### P2.1 Recorder功能

在root仓库实现公开LIBERO `OSC_POSE` recorder：

- keyboard为必备输入，SpaceMouse仅为可选适配；
- 20 Hz单调时钟循环，记录deadline miss与实际timestamp；
- 平移、旋转分别可调步长；gripper toggle；camera切换；pause、abort、reset、save；
- reset必须来自P1 accepted完整snapshot；
- 每step记录7D action、完整/策略state、front/wrist RGB、predicates、reward/done/success与subgoal；
- 保存start/end完整sim state、task/reverse spec/hash、操作者、input device与软件版本；
- abort/失败episode进入rejected目录，不进入训练manifest。

### P2.2 人工canary

在3个operator families各录至少1条成功reverse demo。每条必须：

- reset时reverse goal为假；结束时environment predicate success为真；
- action finite且符合7D OSC_POSE，gripper为`{-1,+1}`；
- 从同一start full state fresh-process action replay 3/3成功；
- H.264人工确认instruction、对象、目标、抓取、释放和最终predicate一致；
- 无teleport、隐藏state mutation、failure trimming或goal-at-reset。

若当前agent无法进行人工输入，只实现/测试recorder并以 `track3_teleop_input_pending_human` 停止，向用户给出唯一启动命令；不得用随机primitive代替teleop成功。

## P3. 从Teleop抽取闭环 keypose planner

仅在P2至少3条3/3成功demo后执行。构建generic闭环阶段机：

`pregrasp → approach → grasp → lift/remove → transit → place/fixture inverse → release → retreat`

要求：

- object/target pose、predicate和接触信息只来自当前公开simulator/reverse spec；
- teleop successful trajectory只用于提取relative keyposes、阈值与phase切换，不直接倒放；
- 每步基于最新observation反馈计算并clip 7D OSC_POSE；记录timeout/collision/drop/wrong-object；
- containment/support先做；activation/articulation只有在专用contact primitive能3/3复现时才纳入，否则明确排除；
- planner episode仍须真实20Hz执行、predicate success及fresh-process 3/3 replay。

先对P2的3个canary做planner复现；全部失败即停止，不盲目扩数据。

## P4. 分级扩展到30–60条accepted pilot

严格按以下门禁逐级扩展：

1. `3 episodes / 3 families`：teleop或planner均须3/3 replay；
2. `6 templates / 2 scenes`：每个template至少1条，逐条H.264审核；
3. `30–60 episodes`：至少3 families、6 base templates、2 scenes、每family至少2 templates。

split按base template而非episode；原T1的32/8 split可沿用但必须验证同一base template不跨train/holdout。公开evaluation denylist与train template/composition overlap必须为0。

accepted与rejected分开保存；每个rejected保留failure reason（unstable reset、restore mismatch、wrong object、no grasp、drop、collision、timeout、predicate fail、replay fail等），不得为了凑数删除失败记录或降低3/3标准。

## P5. Schema、manifest 与 loader smoke（仍不训练）

把P4 accepted数据写成独立`track3_reverse_pilot_v2`：

- 真实20 FPS；front/wrist/state/7D OSC_POSE；gripper `{-1,+1}`；
- source snapshot与demo provenance、base template/family/scene/split、所有hash；
- accepted/rejected/replay/human-review manifests；
- action/state分维统计和fresh per-dimension QUANTILES仅作报告，不替换R6 norm；
- loader做1/10-step CPU/GPU3 forward-only smoke，验证shape、temporal mask、finite、action contract和不跨episode；optimizer update必须为0。

最终门禁：30–60 accepted、coverage达标、逐条3/3 replay、H.264审核、denylist overlap 0、loader smoke通过、root/nested clean、全量CPU测试通过。状态写为 `track3_reverse_pilot_ready_pending_review`并hard stop。

## 1. 明确禁止与下一审核

本计划完成后也不得自动恢复旧T2/T4 LoRA。下一次人工审核要先回答：

1. full-state source与当前环境是否版本一致；
2. accepted episodes是否真的由20Hz执行而非state manipulation；
3. reverse成功是否跨family/template而非单一模板记忆；
4. 30–60条是否足以开始从R6 step500分叉LoRA，或需先扩到更均衡的数据量。

只有新审核明确批准，才另建LoRA训练计划。所有阶段结束均更新plan index、handoff、machine JSON、测试与clean commit。

## 2. 允许的最终状态

- `track3_public_terminal_source_blocked`
- `track3_terminal_snapshot_canary_passed_pending_review`
- `track3_teleop_input_pending_human`
- `track3_teleop_planner_canary_passed_pending_review`
- `track3_reverse_pilot_ready_pending_review`
