# AXIS Pi0.5 Track 3：P3 I/O Recovery 与 Support Executor Canary 计划

> 制定日期：2026-09-05（Asia/Tokyo）
> 审核输入：root `ad2d4ef7d6603e7b0e79a8519155de9cfdefbde2`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 当前状态：`track3_p3_io_recovery_and_support_executor_canary_planned`
> 用户人工操作：`not_required`
> 唯一授权：R0–R6；随后hard stop
> 不授权：修改旧P0–P6/HDF5/BDDL、人工Teleop、task20/task19 manipulation executor、批量生成、accepted晋升、训练/norm、FAST/ZIP、官方GPU、上传或比赛提交

## 0. 目标与产物边界

保留P2 `9/9`，先闭合未入Git的state/media与worker I/O可复现性，再恢复P3固定三条provenance canary。只有P3新证据`3/3`后，才实现并尝试一次task10/demo0 support-remove executor；成功后最多产生一条`pending_autogen_v3`。

新报告只写入：

```text
data/reports/axis_libero_30k/track3_p3_io_recovery_and_support_executor_canary/
data/pending_autogen_v3/task10_demo0/   # 仅R5 generation与3/3 replay全通过后
```

旧`track3_start_bundle_and_support_executor_canary/`内容与hash保持immutable。允许仅改变其中4个既有文件的Git tracking状态，不得改bytes。

## 1. R0：输入锁与Git artifact closure（CPU/filesystem only）

1. 锁定`ad2d4ef`、旧P0/P1/P2/P3/P6报告、17-entry manifest、9-state bundle、camera NPZ、两路MP4、三份HDF5与nested OpenPI的bytes/SHA-256。
2. 独立验证P2：no-pickle、9 keys、float64/shape/finite/hash、9/9 fresh restore报告、最大误差、truth/contact与固定grasp geometry。禁止重跑P2 simulator。
3. 以`git add -f`纳入旧manifest中4个当前untracked但hash-exact的文件；不得重新编码视频或重写NPZ。生成新`r0_artifact_closure.json`，要求17/17存在、hash exact且`git ls-files`可见。
4. 写errata：旧P6的`entries_exclude_self_and_p6_summary`不准确；旧`source_commit`只表示parent而非pre-fix tool version。旧JSON不原地修改。
5. 新报告使用compact references，不复制P0/P2的逐checkpoint大对象。

任何bytes/hash变化、state key问题或Git closure非17/17，状态`track3_p3_recovery_r0_artifact_closure_failed`并直接R6。

## 2. R1：无管道result协议与超时合同（CPU/code only）

在现有工具之上新增独立recovery入口，不修改旧报告：

- worker必须接受`--result-output`，把最终compact JSON原子写入独立文件；stdout/stderr只写重定向日志，父进程不得从stdout解析协议。
- parent在spawn前登记seed与目标GPU；运行中每`≤100 ms`采样PID/GPU UUID/显存。result必须包含worker PID且与parent观测一致。
- 固定wall-clock deadline预注册为`300 s`。超时先TERM并等待最多`10 s`，仍未退出再KILL；日志、return code、timeout stage和partial cleanup原子保存。
- CPU压力测试的child至少写`16 MiB` stdout后写result，必须无阻塞结束；另测hang、missing/malformed result、nonzero exit与worker exception全部fail-closed。
- 修复manifest generator的P6-summary inclusion字段，并新增“manifest entries必须Git tracked”的测试。
- P3 telemetry只保留41步compact truth/contact counts/hash，不再向顶层summary嵌入raw scene矩阵。

R1不得import simulator或使用GPU。完成代码/测试/4个artifact tracking后先创建clean intermediate commit；记录commit与tool SHA-256，GPU阶段只能从该clean commit运行。

## 3. R2：task10/demo0 P3 recovery attempt 2

仅在R0/R1通过、root/OpenPI clean且GPU3自然空闲时，运行一条fresh EGL task10/demo0。明确记录为`attempt_number=2`，不可覆盖旧attempt 1。

- 固定seed `3100`与P2 `task10_demo0` state；不重算state、不换demo。
- 要求result-output、stdout/stderr log、PID一致、GPU3 UUID、start/end snapshot、数值peak、poll interval与deadline全部闭合。
- restore、source=true、goal=false、step0–40 full-window contact-v3、camera schema必须全部通过。
- 两路41帧H.264、20FPS、frame NPZ及所有hash闭合；不要求与旧失败视频像素bitwise一致。

任一失败立即R6；不得重试或进入R3。通过状态`track3_p3_recovery_task10_passed`。

## 4. R3：补齐task20/task19 provenance

只有R2通过才按task20/demo0 seed`3200`→task19/demo0 seed`3190`各运行一次，合同与R2完全相同。任一失败立即停止剩余运行并R6，不换seed/demo、不调阈值。

三条新有效结果全部通过才记为`track3_start_bundle_gpu_provenance_closed_3_of_3`。旧attempt 1永久计为tool failure，不混入`3/3`分母，也不得删除。

## 5. R4：task10-only executor静态合同

只实现task10/demo0，R4不得启动simulator/GPU：

1. 用公开forward action/state证明首次`-1→+1`后object与EEF共同运动、后续`+1→-1`对应release；若不能证明gripper sign语义，fail-closed，不执行R5。
2. 固定读取P2的`T_object^-1·T_eef`、pregrasp与same-demo init object pose，不按canary结果搜索或调整抓取点。
3. 阶段固定为`home→pregrasp→SE(3) descend→close→lift→transport→descend→open→retreat`；位置与quaternion-log姿态闭环。
4. 在R4预注册每阶段step budget、position/orientation tolerance、stable ticks、translation/rotation clip、gripper sign与合法contact矩阵。R5运行前禁止修改。
5. 单测覆盖SE(3)方向、clip、stage timeout、gripper语义证据、trace immutability、atomic failure assets与禁止task20/task19分支。

输出`r4_executor_contract.json`；只有全部static gates通过才允许R5。

## 6. R5：一次generation与冻结trace 3/3 replay

仅在R0–R4全部通过、root/OpenPI clean且GPU3空闲时，运行task10/demo0唯一一次generation；不得在线调参、retry或换seed/demo。

- 从已锁定state bundle精确restore，20Hz，action必须finite `float32[7]` OSC_POSE。
- 逐阶段记录SE(3)误差、clip/saturation、truth/contact、其他物体位移、stall/timeout及首个success frame。
- 只允许gripper↔bowl及bowl↔plate/table的阶段性接触；unrelated background、其他物体移动`>2 mm`、timeout/stall或goal false均失败。
- 记录result protocol、seed/PID/GPU UUID/peak、两路H.264、冻结action trace bytes/SHA-256和完整失败信息。

generation失败则保存diagnostic并R6，不创建pending。成功后只允许从同start state运行冻结trace的3个fresh replay；`3/3`通过相同门禁后才能原子写入`data/pending_autogen_v3/task10_demo0/`。计数上限：generation=1、replay=3、pending=1、accepted=0。

## 7. R6：收尾hard stop

1. 输出summary、逐phase/task/attempt CSV、新artifact manifest、success/failure gallery和中文closeout；所有manifest文件必须bytes/hash exact且Git tracked，或明确列为外部大数据并有独立不可变manifest。本轮产物均小，应全部跟踪。
2. 复核旧17-entry manifest、4个tracking closure、P2 `9/9`、P3 attempt计数、P4/R5执行边界、pending/accepted及所有禁止项。
3. 运行targeted/full tests、strict JSON、NPZ no-pickle、py_compile、ffprobe、`git diff --check`；root/OpenPI clean commit。
4. 无论成功或失败都hard stop。成功状态只能是`track3_support_pending_autogen_v3_ready_for_audit`；失败必须包含准确phase/stage。

本计划全过程不需要用户Teleop、逐帧视频审核或手工选择轨迹；也不授权直接训练、扩展三族数据或制作提交ZIP。
