# AXIS Pi0.5 Track 3：J5 Executor 合同与原子视频恢复计划

> 制定日期：2026-09-05（Asia/Tokyo）
> 审核输入：root `7ae488b4475169ac1cc3b586158454639e6a2be9`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 当前状态：`track3_j5_executor_contract_and_atomic_video_recovery_planned`
> 唯一授权链：`V0_to_V5_then_hard_stop`
> 用户人工操作：`not_required`
> 不授权：修改历史J0–J11证据、task10 P3 attempt4、task20/task19 manipulation、J6 generation/replay、accepted demo、Teleop、训练/norm、GPU0、官方GPU、dense full-FT、FAST/ZIP、上传或比赛提交

## 0. 目标与边界

本计划不延续归档计划的未执行尾部。它只修复J5首次pilot暴露且由独立审核确认的合同错误，并用一个GPU3 canary验证修复：

1. Robosuite quaternion统一为explicit `xyzw`；
2. `*.partial.mp4`原子视频输出；
3. 正常发布的family failure与worker/infrastructure failure分离；
4. J11原因保持与上游首错一致；
5. supervisor定期heartbeat与启动identity真实实现；
6. manifest仅覆盖最终关闭且不可变的文件。

旧task10 nominal运行保留`failed_gate`和parent `failed_closed`，但对修正后executor的物理profile结论为`not_evaluated`。旧run不进入training/pending/accepted，也不消耗新namespace的唯一corrected canary。

## 1. V0：输入与历史失败锁（CPU only）

- 锁定`7ae488b`、J0–J5/J11全部小型报告、J5 frames/trajectory/start/end、旧tool SHA `984210fb...`、三份HDF5、9-state bundle、R6/norm hashes与本次审核summary。
- 独立记录旧J3 manifest `21项/5 mismatch`、旧J11 manifest `37项/4 mismatch`；不得修补或重写旧manifest。
- 记录旧J5双重事实：child 180帧后`stage_timeout:approach_grasp`，parent同时发生ffmpeg infrastructure failure；不得把任一层覆盖成成功。
- 检查root/OpenPI clean、`LIBERO-plus` dirty snapshot、PID/cursor/heartbeat、GPU0外部进程和GPU1–3空闲；V0不得import simulator/CUDA。

任一核心J3/J5 artifact或R6/HDF5 hash发生未知漂移即hard stop。

## 2. V1：纯CPU合同修复

### V1.1 Quaternion与support HDF5证据

- 内部统一Robosuite `xyzw`，所有字段名携带convention；禁止未标注的4-vector跨边界。
- quaternion compose、inverse、shortest-log误差必须用同一约定；`q`与`-q`等价。
- 对task10 demo0从same-model HDF5重建object-relative EEF，geodesic误差必须`<=1e-5 rad`；旧reference的正确复算值约`2.28e-7 rad`。
- 从HDF5锁定gripper transition、object-follow/contact窗口和same-demo初始bowl pose。reverse transport/place目标必须是该初始pose，不得用region中心替代。
- 只释放`support_remove`合同；task20/task19 manipulation保持`not_evaluated`，后续审核另行决定。

### V1.2 Pilot hard gates

冻结task10 nominal的原translation/rotation clips、tolerance与stage budgets，不依据旧GPU结果调参。修正total step budget语义，明确每stage与全局上限。

success必须同时满足：reverse goal true、finite、source/goal语义、contact-v3角色矩阵、无collision/stall/timeout、无unrelated contact、unrelated movement `<=0.002 m`、H.264双视频与完整telemetry。任何未执行gate为`not_evaluated`，不得写false或passed。

### V1.3 I/O、进程与状态映射

- 原子视频使用`name.partial.mp4`或显式`-f mp4`，验证后`os.replace`到`name.mp4`；异常清除partial并保留错误。
- worker成功发布结构正确的`passed`或`failed_gate`结果时进程return code均为0；只有exception、无法发布、timeout/TERM/KILL或schema损坏为非零/infrastructure。
- parent分类表必须独立覆盖`success`、`family_failure`、`infrastructure`，camera failure始终infrastructure。
- J11/closeout直接传播上游phase/status/category/first-error；`infrastructure_failure`不得改写为`no_family_release`。
- supervisor在启动前拒绝dirty source、commit/tool hash不符、活PID/PID复用或旧active cursor；运行中按不超过15分钟间隔原子更新heartbeat。

### V1.4 Finalization与manifest

先关闭phase artifacts、CSV、cursor和supervisor closeout，再生成manifest。排除manifest自身、仍在写的stdout/stderr log和ephemeral PID；所有被列入的文件必须Git tracked、bytes/SHA exact，并在clean checkout复验。旧manifest只作为输入，不进入“全部exact”统计。

## 3. V2：离线derived closure（CPU only）

- 在新`track3_j5_executor_recovery/`namespace从旧`frames.npz`重编码agentview/wrist各180帧，验证H.264、20 FPS、128×128、原子publish和无partial。
- 生成`derived_old_nominal_errata.json`，同时记录旧child physical trace、错误quaternion contract、旧parent ffmpeg failure、return-code耦合与J11错误映射；所有旧文件hash保持不变。
- 纯函数回归至少覆盖：正确/错误quaternion convention、`q/-q`、HDF5 pose reconstruction、MP4 suffix、valid failed_gate return 0、exception return nonzero、三类parent mapping、J11 reason propagation、periodic heartbeat、manifest finalization和duplicate-key rejection。
- V2不得启动simulator或GPU，不得把离线视频视为pilot成功。

## 4. V3：GPU前clean release commit

- 运行strict JSON、NPZ `allow_pickle=False`、ffprobe、CSV LF、manifest/hash、py_compile、targeted/full CPU tests和`git diff --check`。
- 创建包含工具、测试、V0–V2 derived evidence的clean intermediate commit；记录root commit、tool hash、OpenPI HEAD和`LIBERO-plus` dirty snapshot。
- GPU前再次检查live PID/command/cursor/heartbeat与nvidia-smi；GPU3必须无compute process。GPU0外部进程不得触碰。

任一检查失败即V5 hard stop，不得运行V4。

## 5. V4：唯一corrected task10 nominal canary（GPU3）

只允许一次新namespace运行：

```text
family       support_remove
task/demo    task10/demo0
profile      nominal（V1修正convention，其他冻结参数不调）
seed         3111
max runs     1
GPU          physical GPU3 only
```

该运行不是task10 P3 attempt4；P3使用J2只读release，继续禁止任何P3重跑。V4必须在fresh process恢复同一start bundle，记录PID/UUID/start/end/peak、query/heartbeat、完整gate tri-state、actions/states、双相机和两路H.264。

- 全部success gates通过：状态为`track3_j5_corrected_support_nominal_passed_pending_expansion_review`。
- child完整发布但物理gate失败：状态为`track3_j5_corrected_support_nominal_failed_gate_pending_review`，不得自动运行conservative。
- 任一shared infrastructure/manifest/provenance失败：状态为`track3_j5_executor_recovery_stopped_V4_<reason>`。

三种分支都立即进入V5；不运行task20/task19 manipulation，不进入generation/replay。

## 6. V5：统一收尾与hard stop

- 生成phase status、derived errata、corrected canary summary、最终cursor、supervisor closeout和checkout-stable manifest；未运行项显式`not_evaluated`。
- 再次运行strict JSON/NPZ/video/CSV/hash、py_compile、targeted/full tests与`git diff --check`，形成clean commit。
- 明确`exact/pending/accepted`计数；本计划最大仍为`0/0/0`。
- 验证GPU3释放、GPU0外部进程保留、root/OpenPI clean、`LIBERO-plus`既有dirty未改。
- hard stop，下一步必须重新审核；不自动启动conservative、其他family、J6、训练或ZIP。
