# AXIS Pi0.5 Track 3：Z5 Derived Release 与 Conservative Continuation 计划

> 制定日期：2026-09-06（Asia/Tokyo）
> 审核输入：root `5a270a912362376cbfa13a7f9e9fd66c8b5d6237`
> Z5 frozen source：`933f4d3afb70ca31c95d55881d404b97cb3687d5`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 当前状态：`track3_z5_derived_release_and_conservative_continuation_planned`
> 唯一授权链：`AA0_to_AA14_then_hard_stop`
> 用户人工操作：`not_required`
> 执行上限：18小时；单一supervisor可原子resume，禁止第二实例、无限retry和结果后调参
> 不授权：改写历史raw evidence/公开HDF5/BDDL、重跑nominal、把Z5失败轨迹入训、Teleop、GPU0、官方GPU、dense full-FT、超过1000 optimizer steps、自动上传或比赛提交

## 0. 总目标与门禁

先把Z5已完成的物理观测与尾部return-code缺陷解耦，并闭合全部36个tracked artifacts；然后只给结果前已冻结的conservative profile一次CPU机会。其余family与数据/LoRA链按独立门禁有界继续。

- AA0–AA4仅CPU/code/filesystem，不创建simulator、不执行physics、不初始化CUDA。
- historical parent仍是`infrastructure`；derived physical分类单独记录为nominal `family_failure`。不得覆盖旧parent/child/postmortem/manifest。
- shared runtime/source identity/finalizer/data contamination/nonfinite/GPU边界错误全局跳AA14；正常family failure只关闭对应lane。
- GPU3只用于camera/generation/replay/evaluation；GPU1+2只用于FP32 LoRA且不得与GPU3并发；GPU0始终禁止。

## 1. AA0：Z审核锁与资源重验（CPU/filesystem only）

- 锁定`5a270a9`、本次audit summary、Z5 Git note/frozen source、45项上游inventory、Z4 11-entry、Z14 28-entry、26个JSON、NPY/NPZ/CSV、raw parent/child/ledger/trajectory/log和后续source-fix commits。
- 重验root/OpenPI/LIBERO-plus snapshot、PID/cursor/heartbeat、GPU0–3、磁盘与`exact/pending/accepted=0/0/0`。
- 任一drift、live instance或未知dirty立即AA14。

## 2. AA1：36/36 artifact closure与errata（CPU/filesystem only）

- 在新namespace生成只读audit manifest，覆盖旧Z namespace全部36个Git-tracked文件，包括两个log和六个旧finalizer exclusions；逐项记录path/bytes/SHA/Git tracking。
- 单独记录audit manifest、derived classification、final validation与closeout的checkout-stable hashes；不要求文件自含自身hash，但最终closeout必须通过外层hash record闭合所有排除项。
- 从raw `child.hard_gate.gates`重建gate scope；记录Z4当时targeted=`9 passed`、full=`280 passed`，当前修复后targeted=`10 passed`、full=`281 passed`，保留旧笔误来源。

## 3. AA2：Z5 derived physical classification（CPU/read-only）

- 证明child result/trajectory/ledger在NameError前已原子发布，parent/child PID、identity、task/demo/seed/profile、28-step数组与stall window exact。
- 历史parent=`infrastructure`不变；derived result固定为nominal `family_failure`、reason=`stage_stall:approach_grasp`、nominal quota consumed、training eligible=false。
- close/lift/transport/place/release及其gate保持`not_evaluated`；禁止从未进入阶段推断抓取或放置可行性。

## 4. AA3：worker/finalizer真实路径回归（CPU/code only）

- exact Z5-style worker dispatch使用注入式no-physics fixture分别发布`passed`、`failed_gate`、`worker_exception`；前两者必须return 0，异常/缺result/argparse必须非零且parent分类一致。
- 覆盖producer close/fsync、stderr根因纳入manifest、phase summary、inactive cursor、manifest exclusions显式化、外层self-hash record、final validation、TERM→KILL、large stdout、PID复用与process ancestor过滤。
- source identity只能读取冻结release record；工具、测试、plan/index/handoff/MEMORY/machine JSON不得进入dirty白名单。

## 5. AA4：conservative clean release与身份冻结（CPU only）

- 固定历史X2的`object_frame_conservative`：translation/rotation clip=`0.025 m / 0.06 rad`，tolerance=`0.012 m / 0.03 rad`，stable ticks=`6`，stage budgets=`[200,160,50,200,260,160,50,120]`，step budget=`1250`；progress window/grace/min-improvement=`4/4/0.02`。
- canonical HDF5/model/state、object→EEF anchor、phase ledger和destination numeric smoke均为0 physics；strict JSON、NPY/NPZ finite/no-pickle、CSV LF、py_compile、targeted/full tests与`git diff --check`通过。
- clean commit后冻结root/tool/OpenPI/command/CUDA/LIBERO-plus snapshot；不允许从AA5结果再改profile、阈值、seed或预算。

## 6. AA5：support conservative CPU物理canary（最多1次）

固定task10/demo0、seed3121、CPU/no-renderer和AA4 frozen identity。禁止nominal重跑。基础设施错误全局停止；正常family failure关闭support lane；成功则冻结conservative generation profile并进入AA6。

## 7. AA6：support GPU3 camera release（最多1次）

仅AA5成功时运行同identity双相机canary；要求physics/phase/provenance一致、两路H.264 20Hz、无partial且GPU3释放。失败只按合同区分shared infrastructure与support family/camera lane。

## 8. AA7：containment/articulation公开合同（CPU/code only）

- task20使用同demo release-side object→EEF anchor、same-demo state0 destination和同一阶段ledger。
- task19使用handle-relative EEF、drawer joint方向/终点与阶段contact，不复用relocation语义。
- demo0 nominal/conservative identity在结果前冻结；无法从公开HDF5证明的lane直接关闭。support正常family failure不阻止两条独立family合同。

## 9. AA8：其余family pilots（CPU最多4次，GPU3最多4次）

固定task20→task19；每族每个预注册profile最多一次CPU，CPU成功profile随后最多一次GPU3 camera canary。family failure隔离，共享基础设施错误全局停止。至少一个family取得GPU3 release才进入AA9。

## 10. AA9：bounded generation与3/3 fresh replay（GPU3）

- train start仅demo0/demo1；demo2永久holdout。released family×start×profile最多一条，总generation≤12、fresh replay≤36。
- `exact`要求producer全部gate；`pending`是exact但replay未闭合；`accepted`只允许3次fresh replay全部通过且hash/manifest闭合。pilot/failed/partial/Z5 trajectory进入diagnostics+denylist。
- 至少一个family有≥2条accepted train traces才进入AA10。

## 11. AA10：20Hz mixture与fresh QUANTILES（CPU）

锁定本地forward dataset；reverse只取AA9 accepted，holdout/diagnostics/failed/partial进入denylist。sampling manifest固定forward/reverse=`80/20`，重算per-dimension `QUANTILES`；training/OpenPI/folded/LeRobot/FAST共享同一norm hash。

## 12. AA11：R6 L2 LoRA 100-step screens（GPU1+2）

初始化R6 raw step500 params，optimizer/scheduler/EMA fresh；`L2_action_lora_heads`、FP32、microbatch1/device、accumulation32、global batch64。顺序运行peak LR `3e-7`与`1e-6`各100 steps，保存0/1/10/100；finite和restore失败关闭该arm，双失败全局停止。

## 13. AA12：holdout/C20选择与500/1000 continuation

对step0和两个step100运行released-family demo2 execute8 holdout与固定C20。winner须reverse相对step0至少`+1` success、C20最多下降`2/20`；通过才续到500，500复验仍通过才续到1000。保存250/500/750/1000，禁止超过1000。

## 14. AA13：独立复验、fold、FAST与条件式本地ZIP

- fresh process重验winner checkpoint/norm/data manifest、reverse holdout、C20、collision/timeout/stall与restore；train/holdout overlap、身份漂移或不可复现立即停止。
- 仅对通过者执行raw/folded parity≤`1e-3`、processor action误差≤`1e-5`、FAST envelope/latency、C5 coverage/source-success及ZIP CRC/布局/大小/SHA/slow权重排除。
- 只有reverse holdout≥1 success且C20合格才生成本地ZIP；不得上传或提交。

## 15. AA14：统一收尾与hard stop

- 任何分支先停worker/释放GPU，再写phase/family/episode/checkpoint summary、LF CSV、inactive cursor、complete manifest、外层self-hash record与final validation。
- 未运行工作=`not_evaluated`；报告exact/pending/accepted及denylist overlap，不得把缺失证据改写为结论。
- 重验strict JSON、NPY/NPZ/parquet、CSV LF、ffprobe、checkout hashes、py_compile、targeted/full tests和`git diff --check`；root/OpenPI clean commit，LIBERO-plus既有dirty不改。
- 所有结果最终hard stop等待审核；不自动resume、上传或提交。
