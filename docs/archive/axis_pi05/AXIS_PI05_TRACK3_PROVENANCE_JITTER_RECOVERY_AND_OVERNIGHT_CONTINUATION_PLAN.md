# AXIS Pi0.5 Track 3：Provenance Jitter 修复与通宵续跑计划

> 制定日期：2026-09-05（Asia/Tokyo）
> 审核输入：root `8c0d296f1dfedb0c777040a733b6f9e352234111`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 当前状态：`track3_provenance_jitter_recovery_and_overnight_continuation_planned`
> 用户人工操作：`not_required`
> 执行上限：12小时；J0–J11连续授权后hard stop
> 不授权：task10 attempt4、修改历史证据、修改公开HDF5/BDDL、Teleop、GPU0、官方GPU、dense full-FT、无限retry、失败轨迹入训、自动上传或比赛提交

## 0. 目标和原则

从已经通过child物理/相机门禁的task10 attempt3继续，不重跑Axis init、R6、start bundle或task10 GPU canary。先修复两个观测基础设施问题：

1. `parent_poll_interval_max_ms`实际测量的是同步`nvidia-smi`查询延迟；不能用单样本`>100 ms`否定完整provenance。
2. O10 CSV在Git行尾规范化后hash变化；必须产出checkout-stable closure。

修复通过后，继续task20/task19 P3、三族executor、严格可回放数据、20Hz mixture、R6 LoRA、holdout/retention和条件式FAST ZIP。单个family物理失败关闭该lane但不阻塞其他family；只有共享基础设施、污染或数值失败才全局停止。

资源固定：

- GPU3：camera canary、executor pilot/generation/replay与rollout，同一时刻最多一个simulator process。
- GPU1+GPU2：LoRA训练；GPU3 simulator空闲后再启动，避免CPU/内存争抢。
- GPU0：始终禁止；不得终止或干扰外部PID。
- 每个阶段原子写summary、heartbeat和cursor；正式续跑通过`setsid`后台supervisor启动，写PID file、主日志、启动commit/tool hash并拒绝第二实例。SSH断开后后台进程必须继续，重连只读cursor，不重复成功identity。

## 1. J0：输入与不可变证据锁（CPU only）

1. 锁定`8c0d296`、O0/O1/O2/O10、attempt3 child/parent、NPZ、两路H.264、旧O10 manifest和R6 checkpoint/norm。
2. 独立验证child四类hard gate、PID/UUID、return code、113 samples、峰值459 MiB、无timeout、41帧/20 FPS和无partial。
3. 明确保留旧parent/O2/O10失败状态，不覆盖、不“修复历史”。
4. 记录旧CSV CRLF manifest hash与当前LF checkout hash，解释差异来源。
5. 检查root/OpenPI clean、`LIBERO-plus` dirty preservation、磁盘与GPU；不得import simulator/CUDA。

任一核心artifact缺失或非已知的hash漂移即全局停止。

## 2. J1：Telemetry v2 与 checkout-stable closure（CPU/code only）

### J1.1 字段语义修复

新增版本化telemetry v2，不修改旧schema：

- `gpu_query_latency_ms`: count/min/p50/p95/p99/max；只用于性能诊断。
- `heartbeat_gap_ms`: count/min/p50/p95/p99/max；测相邻采样完成时间。
- `supervisor_elapsed_ms`、`child_elapsed_ms`、`sampling_coverage_count`。
- start/end snapshot return code、GPU UUID、CUDA mapping、parent/child/result PID、peak memory、deadline、TERM/KILL状态继续保留。

新的hard gate：

- 所有required fields存在、finite且类型正确；
- PID一致，GPU3 UUID与CUDA mapping一致；
- start/end query均成功，process return code为0、未timeout；
- 对持续时间超过5秒的child至少5次采样，且平均覆盖不低于每2秒1次；
- `heartbeat_gap_max_ms <= 2000`。单次query latency不设pass/fail阈值，但`p99>500 ms`写warning。

阈值在任何新GPU运行前冻结，不得根据task20/task19结果改动。

### J1.2 CSV与manifest修复

- 所有CSV writer固定`lineterminator="\n"`。
- manifest生成前flush/fsync/close；manifest不包含自身。
- 新建`track3_overnight_jitter_recovery/`namespace，旧O10文件保持不变。
- 测试LF字节、SHA稳定、Git index blob与clean checkout字节一致。

### J1.3 回归与中间提交

覆盖query latency 50/148/600 ms均不导致provenance失败；heartbeat gap 1999 ms通过、2001 ms失败；缺字段、PID/UUID mismatch、query error、稀疏覆盖、timeout、非零return code均fail-closed。完成strict JSON、CSV、manifest、py_compile、targeted/full CPU tests并创建clean intermediate commit。

同时增加单实例overnight launcher：`setsid`启动、PID存活/command/commit核对、15分钟heartbeat、SIGTERM后原子closeout；旧PID、PID复用、dirty source或同namespace已有活动实例均拒绝启动。launcher只允许顺序进入J2–J11，不得绕过阶段门禁。

## 3. J2：attempt3只读升格（CPU only，不重跑task10）

对不可变attempt3 parent/child生成`derived_attempt3_release.json`：

- 原始旧gate仍记录failed，不改写。
- 把旧113个snapshot可获得信息、起止时间与process record映射到telemetry v2；不能从旧证据证明的字段标`not_recoverable`，不得伪造。
- 因旧文件只保存query latency max而未保存每次完成时间，heartbeat详细分位数可标`not_evaluated_historical`；使用113 samples、25.528秒起止跨度、正常process完成作为historical coverage gate。
- 只有child/camera、PID/UUID、start/end、coverage、return/timeout、artifact hash全部通过才释放`support_remove`。

J2通过时明确规定`task10_attempt4=forbidden`。J2无法从现有证据闭合时全局停止并等待人工审核；不得自行重跑task10。

## 4. J3：补齐task20/task19 P3（GPU3）

依次各运行一次：

- task20/demo0/seed3200/attempt1；
- task19/demo0/seed3190/attempt1。

使用telemetry v2与独立atomic result；300秒deadline、10秒TERM grace后KILL。基础设施错误全局停止；child物理/predicate失败只关闭对应family。task10沿用J2 release，不再运行。

至少一个family release才允许J4；否则直接J11。每次运行完成立即落盘并commit evidence，SSH中断不得丢失或重复执行。

## 5. J4：HDF5-grounded executor冻结（CPU/code only）

为已release family分别实现，不要求三族齐全：

- `support_remove`：bowl从plate抓起，回到same-demo init pose并释放。
- `containment_remove`：目标物从container/cabinet抓出，回到same-demo init pose并释放。
- `articulation_close`：从forward轨迹提取handle-relative EEF与drawer joint方向，执行reverse close。

从固定forward HDF5证明gripper sign、object-follow/contact或fixture joint方向；不能证明则关闭lane。每族预注册`nominal`和`conservative`两套profile，冻结SE(3) clip、tolerance、stable ticks、step budget、contact矩阵、stall/timeout与unrelated movement阈值。不得在GPU结果后在线改waypoint。

## 6. J5：family pilot（GPU3，最多6次）

顺序task10→task20→task19；每个released family对demo0最多运行两个冻结profile：

- 首个成功profile冻结为generation profile；
- 两个均失败只关闭该family，保留完整失败视频/阶段/接触/误差；
- infrastructure error全局停止；
- success要求reverse goal true、finite、无collision/stall/timeout、无unrelated contact/movement。

至少一个family成功才进入J6；否则J11，不训练。

## 7. J6：bounded reverse generation/replay（GPU3）

- train starts：每族demo0/demo1；holdout：demo2永久隔离。
- 每个train start只使用J5成功profile；每族最多4条generation，总计最多12条。
- 每条success冻结action trace并做3/3 fresh-process replay，总计最多36次。
- 3/3均通过才标`machine_accepted`；失败/pilot/partial只进diagnostics。
- 每条accepted必须包含20Hz双相机、state、7D OSC_POSE、gripper、sim/source hash、语言/family/profile、predicate/contact/collision ledger与H.264。
- 去重identity=`start_state + action_trace + instruction`；demo2、diagnostics、失败trace进入denylist。

至少一个family有两条machine-accepted train traces才进入J7。

## 8. J7：20Hz mixture与fresh QUANTILES（CPU）

锁定本地`data/raw/hf/lerobot/libero_plus`：20Hz、14347 episodes、2238036 frames、40 tasks。reverse accepted转换到同一schema，forward/reverse目标采样`80/20`：forward按task/扰动分层，reverse按family→start/profile等权。单family仍维持reverse 20%并显式标记family-only。

从实际训练manifest重新计算逐维`QUANTILES`，训练/OpenPI/folded/FAST共享同一hash。检查fps、finite、action/gripper范围、frame/action alignment、episode边界、train/holdout隔离和抽样权重。失败不得训练。

## 9. J8：两卡R6 LoRA screen与continuation（GPU1+2）

初始化固定R6 raw step500 params；optimizer/scheduler/EMA重新初始化；拓扑`L2_action_lora_heads`，dense vision/language/expert冻结。FP32、microbatch 1/device、accumulation 32、global batch64、clip norm1.0。

先顺序运行两个100-step arm：peak LR `3e-7`与`1e-6`，保存0/1/10/100并验证finite、restore、optimizer state。单arm失败关闭该arm；双失败全局停止。

J9选出step100 winner后才续到500；500仍满足门禁才续到1000，保存250/500/750/1000。禁止无评估超过1000 steps。

## 10. J9：reverse holdout与C20 retention（GPU3）

对step0、两臂step100、winner step500/1000运行：

- reverse holdout：各released family固定demo2和policy seeds，execute8；executor reference不算policy成功。
- C20 retention：固定标准LIBERO manifest/seed/execute8。
- 输出paired success、collision/timeout/stall、wins/losses/ties、tasks/episodes CSV和失败H.264。

继续训练/最终选择门禁：reverse相对step0至少增加1个成功，C20最多下降2/20。排序依次为reverse success、C20 retention、collision/timeout，并选择最早达到最佳值的checkpoint。

## 11. J10：条件式fold、FAST与本地ZIP

仅对J9 winner：

1. LoRA fold，raw/folded fixed parity最大绝对误差`<=1e-3`。
2. OpenPI→folded PyTorch→LeRobot processor token/mask exact、action误差`<=1e-5`、norm hash一致。
3. 导出`model_fast.safetensors`；16-case source/FAST envelope、`/act 12/12 <10s`和C5 source-success通过。
4. ZIP CRC、布局、大小、FAST优先、slow权重排除和SHA-256通过。

只有reverse holdout至少1个成功且retention合格才生成ZIP。仅本地落盘，不上传、不提交。

## 12. J11：统一收尾与下一次审核

无论成功、单family关闭还是提前停止，生成phase/family/task/episode/checkpoint summary、CSV、gallery、manifest/norm/checkpoint/FAST/ZIP hashes、GPU provenance和resume cursor。所有未运行阶段必须为`not_evaluated`。

执行strict JSON/NPZ/parquet/CSV LF/manifest checkout hash、ffprobe、py_compile、targeted/full tests、`git diff --check`；root/OpenPI clean commit，`LIBERO-plus`既有dirty不改。

允许最终状态：

- `track3_jitter_recovery_candidate_zip_ready_for_review`
- `track3_jitter_recovery_lora_ready_without_zip`
- `track3_jitter_recovery_autogen_ready_without_training`
- `track3_jitter_recovery_stopped_<phase>_<reason>`

全部分支在J11 hard stop，不自动上传/提交。用户无需保持SSH、无需Teleop或中途逐阶段审核。
