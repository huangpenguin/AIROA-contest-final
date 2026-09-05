# AXIS Pi0.5 Track 3：Runtime Preflight 与 Bounded Continuation 计划

> 制定日期：2026-09-05（Asia/Tokyo）
> 审核输入：root `64a0d31c3b2864562887044bad7ef8b522a213db`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 当前状态：`track3_runtime_preflight_and_bounded_continuation_planned`
> 唯一授权链：`Y0_to_Y13_then_hard_stop`
> 用户人工操作：`not_required`
> 执行上限：18小时；单一supervisor可原子resume，禁止第二实例、无限retry和结果后调参
> 不授权：改写历史raw evidence或公开HDF5/BDDL、Teleop、GPU0、官方GPU、dense full-FT、失败/partial/holdout入训、超过1000 optimizer steps、自动上传或比赛提交

## 0. 目标与总门禁

本计划保留X0–X13已成立的derived/object-frame与hard-stop成果，先闭合阻止首次物理canary的运行合同，再在同一授权中有界推进T3 reverse数据、LoRA和条件式本地ZIP。

- shared runtime/symbol、immutable、source identity、simulator wrapper、数据污染、nonfinite、GPU边界或finalizer错误：全局跳Y13。
- 正常发布的物理family failure只关闭对应family/profile lane；不得改阈值后重跑。
- GPU3只用于simulator/generation/replay/evaluation；GPU1+2只用于FP32 LoRA，且不得与GPU3 workload并发；GPU0始终禁止。
- X4 startup worker永久计入历史process attempt，但因0 physics step不占新的物理canary配额；新run使用新namespace/identity，不覆盖X4。

## 1. Y0：X审计与完整输入锁（CPU/filesystem only）

- 锁定`64a0d31`、X13全部13-entry manifest、manifest/final-validation独立hash、X4 parent/child/log、旧W4 trajectory、state bundle、三份public HDF5、R6 raw/folded checkpoint和norm；逐项记录path/bytes/SHA。
- 重验root/OpenPI、LIBERO-plus dirty snapshot、PID/cursor/heartbeat、GPU0–3、磁盘以及`exact/pending/accepted=0/0/0`。
- 旧X/W/V/J报告只读；任一drift、live instance或未知dirty全局停止。

## 2. Y1：运行依赖与阶段合同补救（CPU/code only）

- 建立worker引用的跨模块symbol inventory；用fresh subprocess执行import、callability、public env configuration、HDF5/model/state resolution smoke，必须在创建`ControlEnv`前捕获缺失symbol。
- 使用唯一canonical env helper，禁止依赖不存在的私有属性；CPU mode固定`CUDA_VISIBLE_DEVICES=""`、`JAX_PLATFORMS=cpu`和no-renderer。
- 阶段重构为：approach pose → descend pose → close连续demonstrator contact → lift期间object-follow且source由true释放为false → transport → place → release后goal连续成立。未进入的gate严格为`not_evaluated`。
- 每个stage预注册progress window/grace/min improvement；window失败必须立即产生`stage_stall:<stage>`，与`stage_timeout`分离并阻止下游动作。
- parent在worker存活期间按冻结间隔周期写heartbeat；测试需覆盖可注入短间隔、TERM→KILL、large stdout、atomic result、PID复用与active cursor拒绝。

## 3. Y2：finalizer与测试闭包（CPU only）

- finalizer顺序固定为producer close/fsync → phase summary/CSV → inactive cursor/closeout → manifest → strict/hash/NPZ/video/CSV检查 → targeted/full tests → final validation → clean commit。
- manifest排除自身与final validation，但closeout必须记录两者checkout-stable hash；日志若用于根因必须纳入tracked evidence或把必要内容完整复制进tracked JSON。
- targeted tests至少覆盖symbol smoke、close/lift/source-release阶段顺序、stall真实早停、timeout区分、三态、N/N+1、periodic heartbeat和完整immutable inventory。
- strict duplicate-key JSON、NPZ/NPY no-pickle/finite、CSV LF、py_compile、全量CPU tests和`git diff --check`全部通过后形成clean release；否则Y13。

## 4. Y3：support nominal CPU物理canary（最多1次）

固定task10/demo0、seed3121、CPU/no-renderer与旧`object_frame_nominal` profile；使用新attempt identity且运行前验证Y2 clean commit/tool hash。基础设施错误全局停止；正常family failure关闭nominal profile并允许Y4 conservative；成功则冻结nominal generation profile。

## 5. Y4：support conservative与GPU3 release

- 仅当Y3为正常family failure，CPU运行一次预注册`object_frame_conservative`；Y3成功时不运行conservative。
- 对每个CPU成功profile仅运行一次同identity GPU3双相机canary；要求physics/provenance gates、H.264 20Hz和GPU3释放。至少一个profile通过才释放support lane。

## 6. Y5：containment/articulation公开合同（CPU/code only）

- task20使用同demo release-side object→EEF anchor与same-demo state0 destination。
- task19使用handle-relative EEF、drawer joint方向/终点与阶段contact，不复用relocation语义。
- demo0的nominal/conservative identity在任何结果前冻结；无法从公开HDF5证明的lane直接关闭。

## 7. Y6：其余family pilots（CPU最多4次，GPU3最多4次）

固定task20→task19；每族每个预注册profile最多一次CPU，CPU成功profile随后最多一次GPU3 camera canary。family failure隔离；共享基础设施错误全局停止。至少一个family取得GPU3 release才进入Y7。

## 8. Y7：bounded generation与3/3 fresh replay（GPU3）

- train start仅demo0/demo1；demo2永久holdout。released family×start×profile最多一条，总generation≤12、fresh replay≤36。
- `exact`要求producer全部gate；`pending`是exact但replay未闭合；`accepted`只允许三次fresh replay全部通过且hash/manifest闭合。pilot/failed/partial进入diagnostics+denylist。
- 每条accepted包含20Hz state、双相机、7D OSC_POSE、gripper、source/model/start/action hash、instruction/family/profile与完整predicate/contact/collision/stall ledger。
- 至少一个family有≥2条accepted train traces才进入Y8。

## 9. Y8：20Hz mixture与fresh QUANTILES（CPU）

锁定本地forward dataset；reverse只取Y7 accepted，holdout/diagnostics/failed/partial进入denylist。sampling manifest固定forward/reverse=`80/20`，从有效manifest重算per-dimension `QUANTILES`；training/OpenPI/folded/LeRobot/FAST共享同一norm hash。

## 10. Y9：R6 L2 LoRA 100-step screens（GPU1+2）

初始化R6 raw step500 params，optimizer/scheduler/EMA fresh；`L2_action_lora_heads`、FP32、microbatch1/device、accumulation32、global batch64。顺序运行peak LR `3e-7`与`1e-6`各100 steps，保存0/1/10/100；finite和restore失败关闭该arm，双失败全局停止。

## 11. Y10：holdout/C20选择与500/1000 continuation

对step0和两个step100运行released-family demo2 execute8 holdout与固定C20。winner须reverse相对step0至少`+1` success、C20最多下降`2/20`；通过才续到500，500复验仍通过才续到1000。保存250/500/750/1000，禁止超过1000。

## 12. Y11：独立复验与候选冻结

在fresh process重验winner checkpoint/norm/data manifest、reverse holdout、C20、collision/timeout/stall与checkpoint restore。任何train/holdout overlap、身份漂移或不可复现都停止，不进入fold/ZIP。

## 13. Y12：条件式fold、FAST与本地ZIP

仅对Y11 winner：raw/folded parity≤`1e-3`；processor token/mask exact且action误差≤`1e-5`；FAST 16-case envelope、`/act 12/12 <10s`、C5 coverage/source-success、ZIP CRC/布局/大小/SHA和slow权重排除全部通过。只有reverse holdout≥1 success且C20合格才可生成本地ZIP；不得上传或提交。

## 14. Y13：统一收尾与hard stop

- 任何分支先停worker/释放GPU，再写phase/family/episode/checkpoint summary、CSV、inactive cursor、manifest、自哈希与final validation。
- 未运行工作=`not_evaluated`；不得把缺失证据写成false/pass/zero-risk。报告exact/pending/accepted和denylist overlap。
- 重验strict JSON、NPZ/NPY/parquet、CSV LF、ffprobe、checkout hashes、py_compile、targeted/full tests与`git diff --check`，root/OpenPI clean commit；LIBERO-plus既有dirty不改。
- 所有结果最终hard stop等待审核；不自动resume、上传或提交。
