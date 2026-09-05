# AXIS Pi0.5 Track 3：Object-Frame Executor 与 Bounded Autogen/LoRA 计划

> 制定日期：2026-09-05（Asia/Tokyo）
> 审核输入：root `2997c6f5d8c5b05345d74e0e0c4b96698def727e`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 当前状态：`track3_object_frame_executor_and_bounded_autogen_lora_planned`
> 唯一授权链：`X0_to_X13_then_hard_stop`
> 用户人工操作：`not_required`
> 执行上限：18小时；允许同一supervisor原子resume，但禁止第二实例、无限retry与结果后在线调参
> 不授权：修改历史raw evidence或公开HDF5/BDDL、Teleop、GPU0、官方GPU、dense full-FT、失败/partial/holdout轨迹入训、超过1000 optimizer steps、自动上传或比赛提交

## 0. 目标、故障隔离与资源

本计划先修复W4暴露的object-frame target与stage prerequisite，再在同一份授权中继续三族bounded autogen、LoRA和条件式本地ZIP，避免每个小canary后重新制定计划。

- 共享source/provenance、simulator wrapper、数据污染、数值非finite、GPU边界或manifest错误：全局跳到X13。
- 正常发布的family物理失败：只关闭该family lane，其他已通过前置门禁的lane可继续。
- GPU3只用于单实例simulator/generation/replay/evaluation；GPU1+2只用于FP32 LoRA且不得与GPU3 workload并发；GPU0始终禁止。
- 每个run identity最多执行一次。profile、seed、start identity和阈值均须在看到对应run结果前冻结。

## 1. X0：输入与历史锁（CPU/filesystem only）

- 锁定`2997c6f`、W0–W6全部17-entry manifest内容、W4 parent/child/trajectory、state75 anchor/HDF5、9-state bundle、三份public HDF5、R6 step500 raw/folded checkpoint与norm。
- 复算本审核的destination transform、`253/253` source、252-frame stage分解、W5 `not_evaluated`、`0/0/0`和资源边界；旧W4/V4/J5 failure label永久不改。
- 记录root/OpenPI、`LIBERO-plus` dirty snapshot、live PID/cursor/heartbeat、GPU0–3和磁盘。不得import simulator/CUDA。
- 任一immutable drift、活动实例、未知dirty source或资源冲突全局停止。

## 2. X1：derived errata 与 finalizer合同（CPU only）

- 在新namespace写W4 derived analysis：历史停止点仍为transport timeout，但earliest semantic failure为acquisition/contact未成立；不得回写W4 parent/child。
- 定义trajectory alignment：`state/action/post_step`为N，predicate trace若含t0必须为`N+1`并带`alignment=t0_then_post_step`；所有consumer显式校验。
- finalizer顺序固定为producer close/fsync → phase summary/CSV → inactive cursor/supervisor closeout → manifest → strict/hash/NPZ/video/CSV检查 → targeted/full tests → final validation report → clean commit。manifest排除自身与final validation report，但两者另有checkout-stable hash记录。

## 3. X2：object-frame、acquisition、stall 与 supervisor修复（CPU/code only）

### X2.1 SE(3) target

- 全部目标先定义为desired object/fixture pose，再计算`T_world^eef_des = T_world^object_des · T_object^eef_anchor`；transport/lift offset施加到desired object pose后再组合，禁止把object quaternion直接发送给EEF controller。
- quaternion边界显式`xyzw`并接受`q/-q`；translation/rotation physical clip继续为`0.05 m / 0.12 rad`，除以实际controller output scale`0.05 m / 0.5 rad`。
- 回归必须锁定W1 state0 exact destination的expected EEF，并证明旧target存在`0.0817122 m / 1.816425 rad`差异。

### X2.2 phase prerequisite/contact

- `approach→descend`要求pose稳定；`close→lift`要求连续gripper↔object demonstrator contact；`lift→transport`要求object-follow、source receiver contact解除且reverse source由true变false。
- initial、acquire、lift/transit、destination/release分别建立contact ledger。t0的plate receiver contact不能与后续任意finger contact拼成completion；destination要求goal receiver与reverse goal在release后连续成立。
- 未进入窗口的gate=`not_evaluated`；一旦某stage prerequisite失败立即停止，不执行下游stage。

### X2.3 progress-window stall与single instance

- 每个stage在运行前冻结progress window、最小translation/rotation improvement和grace frames；连续窗口无足够进展即`stall`，stage budget耗尽单独为`timeout`。
- supervisor在启动worker前原子写全局cursor `active=true`和PID/command/commit/tool/OpenPI/GPU policy；周期heartbeat、TERM→KILL、final inactive closeout均可注入短间隔测试。旧PID、PID复用、command/hash不符、dirty source或active cursor拒绝启动。
- parent first_error按infrastructure → earliest stage prerequisite → stall/timeout → downstream gate顺序传播。

## 4. X3：静态/离线门禁与 clean release commit

- 对SE(3) compose/inverse、controller scaling、phase-scoped contact、source transition、object-follow、stall/timeout、tri-state、N/N+1 trace、single-instance cursor、large stdout、atomic result和finalizer做CPU tests。
- 从W4 raw做只读derived replay，证明新逻辑会在acquisition barrier停止，不伪装成新simulator run。
- 运行strict duplicate-key JSON、NPZ/NPY no-pickle、CSV LF、py_compile、targeted/full CPU tests与`git diff --check`，形成clean release commit；此后不得改profile/threshold/anchor。

任一失败直接X13，不得启动X4以后阶段。

## 5. X4：support CPU profile ladder（最多2次）

固定task10/demo0、seed3121、CPU/no-renderer：

1. `object_frame_nominal`：W4现有clip/tolerance/stable ticks/budget，只修正X2语义；
2. 仅当第1次为正常family failure时运行`object_frame_conservative`：translation/rotation clip=`0.025 m / 0.06 rad`、tolerance=`0.012 m / 0.03 rad`、stable ticks=`6`、逐stage budgets=`200/160/50/200/260/160/50/120`。

两个预注册profile各最多运行一次，用于独立资格与数据多样性；基础设施失败全局停止，正常物理失败只关闭该profile。至少一个成功则冻结对应support generation profile；两个均失败才关闭support lane。

## 6. X5：support GPU3 camera release（最多2次）

对X4每个CPU成功profile，以同task/demo/seed各运行一次GPU3 camera canary，最多2次。要求全部family/provenance gate、双相机NPZ、H.264/20 FPS与GPU3 release通过；失败后不重跑。每个profile独立释放，至少一个成功才释放support generation lane。

## 7. X6：containment/articulation HDF5合同（CPU/code only）

- task20 containment：按同demo release-side gripper/object-follow/contact窗口冻结`T_object^-1·T_eef`，desired object pose来自same-demo state0。
- task19 articulation：按同demo handle-relative EEF、drawer joint方向/终点、接触窗口构造close executor，禁止套用object relocation逻辑。
- 每族demo0、nominal/conservative两个profile在运行前冻结；task10修复必须抽象为通用SE(3)/stage组件，不得包含hidden-task信息。
- 合同无法从公开HDF5证明时关闭该lane，不影响其他family。

## 8. X7：其余family pilots（CPU最多4次，GPU3最多4次）

- 固定顺序task20→task19；每族nominal/conservative各最多运行一次CPU pilot。每个profile独立通过或关闭，两个均失败才关闭family lane。
- 每个CPU成功profile随后只运行一次同配置GPU3 camera canary，最多共4次。基础设施错误全局停止，正常物理失败只关闭该profile。
- 至少一个family取得GPU3 release才进入X8；否则跳X13且不生成、不训练。

## 9. X8：bounded generation 与3/3 fresh replay（GPU3）

- train start仅demo0/demo1；demo2永久holdout。每个released family×train start×已冻结成功profile最多一条；若同family两个profile在CPU/GPU都独立成功，可各生成，因此总generation上限12、replay上限36。
- `exact`=冻结identity下producer全部gate通过的唯一trace；`pending`=exact但3/3 replay尚未闭合；`accepted`=三次fresh replay全部通过且hash/manifest闭合。失败/pilot/partial全部进入diagnostics+denylist。
- 每条accepted必须有20Hz state、双相机、7D OSC_POSE、gripper、source/model/start/action hash、语言/family/profile、predicate/contact/collision/stall ledger与H.264；identity=`start_state + action_trace + instruction`去重。
- 至少一个family有`>=2`条accepted train traces才允许X9。

## 10. X9：20Hz mixture与fresh QUANTILES（CPU）

- 锁定本地`lerobot/libero_plus`的20Hz/14347 episodes/2238036 frames/40 tasks及当前hash；reverse只取X8 accepted，holdout/diagnostics/failed/partial denylist。
- 实际sampling manifest固定forward/reverse=`80/20`；forward按task/perturbation分层，reverse按family→start/profile等权。单family模式也保持20%并显式标风险。
- 从有效manifest重算per-dimension `QUANTILES`；training/OpenPI/folded/LeRobot/FAST必须共享同一norm hash。fps、finite、action/gripper、frame alignment、episode边界、train/holdout隔离和权重任一失败不得训练。

## 11. X10：R6 L2 LoRA 100-step screens（GPU1+2）

- 初始化固定R6 raw step500 params；optimizer/scheduler/EMA fresh。拓扑`L2_action_lora_heads`，FP32，microbatch1/device，accumulation32，effective global batch64，clip norm1.0；dense vision/language/expert保持冻结。
- 顺序运行peak LR `3e-7`和`1e-6`两个100-step arm，保存0/1/10/100。每步loss/grad/update/params/EMA finite，checkpoint/optimizer restore必须通过。
- 单arm数值/restore失败只关闭该arm；双失败全局停止。禁止GPU0、BF16、dense full-FT。

## 12. X11：holdout/C20选择与500/1000 continuation（GPU3↔GPU1+2串行）

- 对step0与两个step100运行released-family demo2固定seed execute8 holdout和固定C20 retention；executor reference不算policy成功。
- winner门禁：reverse相对step0至少`+1` success，C20最多下降`2/20`；按reverse success、C20、collision/timeout、较早step排序。未通过则保留autogen但不续训。
- 通过后只续winner到500并评估；500仍过同一门禁才续到1000并评估。保存250/500/750/1000，禁止超过1000。

## 13. X12：条件式fold、FAST与本地ZIP

仅对X11最终winner：raw/folded parity `<=1e-3`；OpenPI→folded PyTorch→processor token/mask exact且action误差`<=1e-5`；norm hash一致；FAST 16-case envelope、`/act 12/12 <10s`、C5 coverage/source-success通过后，验证ZIP CRC/布局/大小/SHA和slow权重排除。

只有reverse holdout至少1个成功且C20合格才可生成本地ZIP。不得上传或提交。

## 14. X13：统一收尾与 hard stop

- 无论成功、lane关闭或提前失败，先停worker并释放GPU，再按X1 finalizer顺序写phase/family/episode/checkpoint summary、CSV、cursor、manifest和final validation。
- 所有未运行工作=`not_evaluated`；不得用false/pass/0-risk代替缺失证据。输出exact/pending/accepted及denylist重叠。
- 重验strict JSON、NPZ/NPY/parquet、CSV LF、ffprobe、checkout hashes、py_compile、targeted/full tests、`git diff --check`；root/OpenPI clean commit，`LIBERO-plus`既有dirty不改。
- 允许最终状态为candidate ZIP、LoRA without ZIP、autogen without training或`stopped_X*`；全部分支hard stop等待审核，不自动恢复、上传或提交。
