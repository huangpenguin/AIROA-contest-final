# AXIS Pi0.5 Track 3：Y3 Launch 与 Phase-Ledger Remediation 计划

> 制定日期：2026-09-06（Asia/Tokyo）
> 审核输入：root `882958c8b863585b13f10f22445d4aa4a42b2d56`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 当前状态：`track3_z4_prephysics_release_ready_pending_clean_commit`
> 唯一授权链：`Z0_to_Z14_then_hard_stop`
> 用户人工操作：`not_required`
> 执行上限：18小时；单一supervisor可原子resume，禁止第二实例、无限retry和结果后调参
> 不授权：改写历史raw evidence或公开HDF5/BDDL、Teleop、GPU0、官方GPU、dense full-FT、失败/partial/holdout入训、超过1000 optimizer steps、自动上传或比赛提交

## 0. 目标与总门禁

本计划保留Y0完整inventory与Y2 artifact closure，修复阻止Y3真实启动和正确判定的三个发布缺口，再有界推进T3 reverse数据、LoRA和条件式本地ZIP。

- Z0–Z4仅CPU/code/filesystem，不创建`ControlEnv`、不调用`env.step`、不初始化CUDA。
- 任一CLI dispatch、source identity、stage ledger、immutable、simulator wrapper、数据污染、nonfinite、GPU边界或finalizer错误：全局跳Z14。
- 正常物理family failure只关闭对应family/profile lane；不得改阈值、换seed或按结果重跑。
- GPU3只用于simulator/generation/replay/evaluation；GPU1+2只用于FP32 LoRA，且不得与GPU3 workload并发；GPU0始终禁止。
- 历史Y0/Y2 failure、X/W报告及所有hash只读；新执行使用新namespace，不覆盖旧证据。

## 1. Z0：审核锁与资源重验（CPU/filesystem only）

- 锁定`882958c`、本次审核summary、Y0两份inventory、Y2十项tracked evidence、三项self hash、45项上游asset bytes/SHA和`exact/pending/accepted=0/0/0`。
- 重验root/OpenPI、LIBERO-plus既有dirty snapshot、PID/cursor/heartbeat、GPU0–3、磁盘与旧Y namespace无drift。
- 任一live Track3 instance、未知dirty或hash drift立即Z14；不得借Z0修补历史报告。

## 2. Z1：真实CLI与child dispatch合同（CPU/code only）

- 顶层parser必须完整接受唯一worker命令`--y3-worker --result-output <absolute-path>`，且禁止缺失/重复/未知worker参数。
- 加入exact-argv fresh-subprocess smoke：使用注入式no-physics worker写atomic sentinel，证明顶层dispatch确实到达worker边界、result path准确、return code与stdout/stderr合同闭合；smoke不得导入或创建simulator。
- 覆盖worker结果缺失、duplicate JSON、return 2、large stdout、TERM→KILL、PID复用、active cursor拒绝与atomic publication。

## 3. Z2：冻结source identity与clean-tree门禁（CPU/code only）

- Z4形成clean source release后，单独冻结其root commit、tool SHA、OpenPI commit、完整child command、CUDA/renderer policy与LIBERO-plus dirty snapshot。
- launcher只能把冻结记录作为expected值；禁止用启动时当前HEAD或worker当前hash自我背书。root HEAD必须等于冻结commit，工具hash必须exact，OpenPI必须exact。
- source dirty扫描只允许忽略本次原子evidence输出namespace；工具、测试、plan、index、handoff、MEMORY与machine JSON不得进入永久白名单。用真实临时dirty注入证明每一类漂移都会在创建run目录、cursor或worker之前fail-closed。

## 4. Z3：阶段作用域ledger与数值门禁（CPU/code only）

- 每个stage持久化`frame_start/frame_end`、stable-window index、目标pose、末端误差、progress windows、进入/完成状态与failure reason；stage结束即冻结，不得由后续样本重算。
- initial source/goal取pre-step snapshot；close contact只取close stable window；object-follow和source-release只取lift stable window；transport只验证其自身收敛与安全；place验证object和EEF对same-demo destination的平移/旋转误差；release/retreat验证goal连续成立与预期解除contact。
- acquisition barrier、stage stall或timeout发生即停止后续动作。所有未进入gate为`not_evaluated`；不得把missing写成false/pass/zero-risk。
- synthetic trajectory至少覆盖完整成功、close contact失败、lift follow失败、source未释放、transport stall、place destination偏差、release goal失败，以及“release后contact解除不会反向否定close”的回归。

## 5. Z4：pre-physics release与finalizer闭包（CPU only）

- fresh subprocess执行canonical helper、三份HDF5/model/state resolution、exact child argv与stage-ledger回归；明确记录simulator未创建、physics step为0、CUDA未启动。
- finalizer顺序固定为producer close/fsync → phase/ledger summary与LF CSV → inactive cursor → manifest → strict/hash/NPZ/NPY检查 → targeted/full tests → final validation → clean commit → 冻结source release identity。
- strict duplicate-key JSON、NPZ/NPY no-pickle/finite、CSV LF、py_compile、targeted/full CPU tests和`git diff --check`全部通过才释放Z5；否则Z14。

## 6. Z5：support nominal CPU物理canary（最多1次）

固定task10/demo0、seed3121、CPU/no-renderer与历史`object_frame_nominal` profile；启动前重验Z4冻结commit/tool/command与资源边界。基础设施错误全局停止；正常family failure关闭nominal并允许Z6 conservative；成功则冻结nominal generation profile。

## 7. Z6：support conservative与GPU3 release

- 仅当Z5为正常family failure，CPU运行一次预注册`object_frame_conservative`；Z5成功时不运行conservative。
- 对每个CPU成功profile仅运行一次同identity GPU3双相机canary；要求physics/provenance gates、H.264 20Hz、stage ledger一致与GPU3释放。至少一个profile通过才释放support lane。

## 8. Z7：containment/articulation公开合同（CPU/code only）

- task20使用同demo release-side object→EEF anchor与same-demo state0 destination，沿用同一phase ledger。
- task19使用handle-relative EEF、drawer joint方向/终点与阶段contact，不复用relocation语义。
- demo0 nominal/conservative identity在任何结果前冻结；无法从公开HDF5证明的lane直接关闭。

## 9. Z8：其余family pilots（CPU最多4次，GPU3最多4次）

固定task20→task19；每族每个预注册profile最多一次CPU，CPU成功profile随后最多一次GPU3 camera canary。family failure隔离；共享基础设施错误全局停止。至少一个family取得GPU3 release才进入Z9。

## 10. Z9：bounded generation与3/3 fresh replay（GPU3）

- train start仅demo0/demo1；demo2永久holdout。released family×start×profile最多一条，总generation≤12、fresh replay≤36。
- `exact`要求producer全部gate；`pending`是exact但replay未闭合；`accepted`只允许三次fresh replay全部通过且hash/manifest闭合。pilot/failed/partial进入diagnostics+denylist。
- 每条accepted包含20Hz state、双相机、7D OSC_POSE、gripper、source/model/start/action hash、instruction/family/profile与完整phase/contact/collision/stall ledger。
- 至少一个family有≥2条accepted train traces才进入Z10。

## 11. Z10：20Hz mixture与fresh QUANTILES（CPU）

锁定本地forward dataset；reverse只取Z9 accepted，holdout/diagnostics/failed/partial进入denylist。sampling manifest固定forward/reverse=`80/20`，从有效manifest重算per-dimension `QUANTILES`；training/OpenPI/folded/LeRobot/FAST共享同一norm hash。

## 12. Z11：R6 L2 LoRA 100-step screens（GPU1+2）

初始化R6 raw step500 params，optimizer/scheduler/EMA fresh；`L2_action_lora_heads`、FP32、microbatch1/device、accumulation32、global batch64。顺序运行peak LR `3e-7`与`1e-6`各100 steps，保存0/1/10/100；finite和restore失败关闭该arm，双失败全局停止。

## 13. Z12：holdout/C20选择与500/1000 continuation

对step0和两个step100运行released-family demo2 execute8 holdout与固定C20。winner须reverse相对step0至少`+1` success、C20最多下降`2/20`；通过才续到500，500复验仍通过才续到1000。保存250/500/750/1000，禁止超过1000。

## 14. Z13：独立复验、fold、FAST与条件式本地ZIP

- fresh process重验winner checkpoint/norm/data manifest、reverse holdout、C20、collision/timeout/stall与checkpoint restore；train/holdout overlap、身份漂移或不可复现立即停止。
- 仅对通过者执行raw/folded parity≤`1e-3`、processor token/mask exact、action误差≤`1e-5`、FAST 16-case envelope、`/act 12/12 <10s`、C5 coverage/source-success及ZIP CRC/布局/大小/SHA/slow权重排除。
- 只有reverse holdout≥1 success且C20合格才可生成本地ZIP；不得上传或提交。

## 15. Z14：统一收尾与hard stop

- 任何分支先停worker/释放GPU，再写phase/family/episode/checkpoint summary、CSV、inactive cursor、manifest、自哈希与final validation。
- 未运行工作=`not_evaluated`；报告exact/pending/accepted和denylist overlap，不得把缺失证据改写为结论。
- 重验strict JSON、NPZ/NPY/parquet、CSV LF、ffprobe、checkout hashes、py_compile、targeted/full tests与`git diff --check`，root/OpenPI clean commit；LIBERO-plus既有dirty不改。
- 所有结果最终hard stop等待审核；不自动resume、上传或提交。
