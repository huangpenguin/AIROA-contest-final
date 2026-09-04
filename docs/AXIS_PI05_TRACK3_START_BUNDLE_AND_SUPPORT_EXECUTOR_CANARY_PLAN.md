# AXIS Pi0.5 Track 3：Start-State Bundle 与自动 Support Executor Canary 计划

> 制定日期：2026-09-04（Asia/Tokyo）
> 审核输入：root `eef683b0db9629a3d917ea91ebeb46540a34fcb9`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 当前状态：`track3_support_executor_canary_failed_worker_result_publication`
> 用户人工操作：`not_required`
> 唯一授权：P0–P6；随后hard stop
> 不授权：修改公开HDF5/BDDL、人工Teleop、批量生成、containment/articulation executor、accepted晋升、训练/norm、FAST/ZIP、官方GPU、上传或比赛提交

## 0. 目标

先把已验证但尚不可restore的9个composed reverse start闭合为可执行artifact，并补齐M4缺失的provenance；通过后只运行一条task10/demo0自动support-remove executor canary。全过程不需要用户操作或逐帧审批，失败资产自动保存用于下一次代码审核。

新产物只能写入：

```text
data/reports/axis_libero_30k/track3_start_bundle_and_support_executor_canary/
data/pending_autogen_v3/                         # 仅P5成功且3/3 replay后最多一条
```

旧M0–M5、HDF5、BDDL、视频和历史失败证据只读。

## 1. P0：输入锁与审核errata（CPU/filesystem only）

1. 锁定`eef683b`、nested OpenPI、归档M0–M5计划、审核报告、M0/M1/M2/M3/M4/M5 summaries、12份raw、6段视频与三份HDF5的bytes/SHA-256。
2. 明确保留：M4 seed/PID/peak缺失、contact gate漏burn-in、`round_trip...`误名、candidate只有hash、handoff orphan commit和run-index缺hash；禁止原地改旧报告。
3. 对12份raw只读复算step0–40 contact-v3、truth、stability和restore；输出逐run SHA-256及完整failed/not-evaluated分解。
4. 输出`p0_release_lock.json`与`P0_M0_M5_ERRATA_ZH.md`。不得import simulator或使用GPU。

任一旧hash、9个unique composed hash、6个video hash或zero reverse-demo count漂移，状态`track3_start_bundle_p0_input_drift`并直接进入P6。

## 2. P1：闭合工具合同（CPU/code only）

新增独立工具与测试，不修改旧`track3_composed_reverse_start_stabilization.py`：

- 新字段使用`composed_vs_init_max_abs_diff`；round-trip专指restore后与输入candidate的误差。
- contact-v3 hard gate覆盖step0–40全部41步；demonstrator与unrelated-background在全窗口均必须为零，task10 intended bowl↔plate contact按既有contract检查。
- artifact manifest每个JSON/video/state均记录relative path、bytes、SHA-256、schema和source commit；路径不得只用绝对本机路径。
- 实现no-pickle NPZ验证、dtype/shape/finite、canonical hash、state-source bitmap和same-demo model XML identity检查。
- 实现GPU worker monitor：parent预注册seed，记录child PID、physical GPU UUID、`CUDA_VISIBLE_DEVICES`、开始/结束快照及轮询峰值；PID/peak缺任一项即fail-closed。
- executor action必须为finite `float32[7]` OSC_POSE，translation/rotation分别按controller output max归一化并clip；gripper只能精确`-1/+1`，neutral hold为`0`。

单测必须覆盖burn-in违规被拒绝、state bundle篡改、hash/path mismatch、GPU provenance缺字段、SE(3) quaternion-log方向/clip、worker exception return code和atomic failure asset。P1不得启动simulator/GPU。

## 3. P2：物化并恢复9个start-state candidates

仅在P0/P1通过后：

1. 从同demo HDF5的init/terminal state及已锁定named ownership ledger确定性重算9个t0 composed states；禁止从旧hash反推或换demo。
2. 写单一`start_states.npz`，key固定为`task{10,20,19}_demo{0,1,2}`，`allow_pickle=False`；每个数组必须float64、原始state shape、finite。
3. 生成`start_state_manifest.json`：HDF5/model/init/terminal/composed hash、instruction、source/goal atom、20Hz、state bitmap canonical hash。9个composed hash必须逐项匹配旧M3 raw。
4. 以9个fresh CPU/no-renderer process从NPZ restore；运行固定step0–40全零hold，并通过P1 full-window contract。不得借此重选或修改state。
5. 在task10/demo0的same-model CPU worker中，按forward HDF5固定gripper首次`-1→+1`索引及相邻state提取`T_object^-1·T_eef`、pregrasp geometry和原始object init pose，写`task10_demo0_grasp_geometry.json`；只读restore，不发action、不按结果搜索index。
6. 成功状态为`track3_start_state_bundle_validated_9_of_9`；计数为`validated_start_state=9`、`exact/pending/accepted reverse demo=0/0/0`。

任一失败直接P6；不得进入GPU或executor。

## 4. P3：GPU3 provenance closure

只有P2 `9/9`、root/OpenPI clean且GPU3自然空闲才执行。固定task10/demo0→task20/demo0→task19/demo0，各一个fresh EGL process，最多3次；不覆盖旧M4。

- seed固定为`3100/3200/3190`，在parent、Python、NumPy和env constructor层记录；无可用seed API必须明确记录而不是伪造。
- parent记录真实child PID并以不超过100ms间隔轮询GPU3显存，输出GPU UUID与peak；禁止触碰GPU0/1/2。
- 从P2 NPZ restore，重复41帧zero-hold及full-window gates；两路H.264/20FPS/41帧、frame/video hash与ffprobe闭合。
- 不要求像素bitwise跨进程一致；dynamics/state hash与predicate/contact contract必须一致。

任何一条失败立即停止后续条目并进入P6。三条通过状态`track3_start_bundle_gpu_provenance_closed_3_of_3`。

## 8. 本轮执行收尾（2026-09-04）

P0、P1、P2 已通过：旧输入未漂移，9 个 `float64` no-pickle start state 已物化并 fresh CPU restore `9/9`，task10/demo0 的固定 HDF5 抓取几何提取通过。P3 只启动了固定顺序第一条 task10/demo0 GPU3 canary；41 帧 camera 与 H.264 失败资产已保存。

P3 worker 在输出完整 checkpoint JSON 时填满父子 stdout 管道，阻塞在结果发布阶段。该次 canary 已终止并记录为 `worker_result_publication` 工具失败；没有重试 task10，也没有启动 task20/task19。P4 与 P5 按 hard stop 为 `not_evaluated`，没有 SE(3) action、demo 或 pending candidate。P6 已完成测试与 artifact closure，当前必须等待新的审核计划。

## 5. P4：task10 HDF5-grounded SE(3) executor（CPU/code gate）

只实现support-remove task10/demo0，不得写task20/task19分支，不得运行simulator。

### 5.1 固定抓取与放置几何

- 从P2已锁定的`task10_demo0_grasp_geometry.json`读取公开task10/demo0的gripper首次`-1→+1`索引、`T_object^-1·T_eef` grasp transform、pregrasp world-z offset和原始object init pose；P4不得再次运行simulator或重选index。
- grasp transform必须来自固定demo0，不能根据canary结果优化；记录source frame/index/hash。
- reverse目的地固定为同demo init object pose与公开reverse goal region，不以reverse success结果搜索目标。

### 5.2 固定阶段

```text
home → pregrasp above current bowl → SE(3) descend → close
→ lift → transport above original pose → descend → open → retreat
```

- 位置与姿态均闭环；orientation误差用quaternion log，不能继续使用旧position-only descend。
- 每阶段max steps、tolerance、clip、stable ticks在P4报告中预注册；P5前不得修改。
- 旧`stage_timeout:descend_object`资产及其132帧hash锁定，不能覆盖或称为成功。
- 训练资格、hidden Track3等价性均为false；该executor只验证公开support reverse operator。

输出`p4_executor_contract.json`，targeted tests与static gates全通过才允许P5。

## 6. P5：唯一自动support canary与3/3 replay

仅在P0–P4全部通过、root/OpenPI clean且GPU3空闲时，运行task10/demo0一次generation canary；不retry、不换seed/demo、不在线调参。

### 6.1 Generation hard gates

- 从P2 bundle精确restore；start source=true、reverse goal=false。
- action/observation/state全程finite；20Hz；逐阶段SE(3)误差、clip/saturation、contact-v3、background movement与collision完整记录。
- 只允许gripper↔bowl及bowl↔plate/table的阶段性合法接触；任何operator↔unrelated background、其他物体移动`>2 mm`、stall、timeout或reverse goal false均失败。
- 必须记录seed/PID/GPU UUID/peak、两路H.264、task详情、阶段、失败原因和第一成功帧。

失败时原子写入`diagnostics/task10_demo0/`并状态`track3_support_executor_canary_failed_<stage>`，不得创建pending，直接P6。

### 6.2 Success与replay

generation成功才原子写一个候选；随后从完全相同start state与冻结action trace运行3个fresh-process replay。3/3均通过相同predicate/collision/state gate后，才移动到`data/pending_autogen_v3/task10_demo0/`。

`pending_autogen`最多1、`accepted=0`、human review字段=`not_required_for_this_phase`。任一replay失败则资产保留diagnostic，不得promote。

## 7. P6：收尾 hard stop

1. 输出summary、artifact manifest、stage/task/episode CSV、success或failure H.264 gallery和中文closeout。
2. 复核旧M0–M5 hash、9个state bundle、3个provenance canary、executor generation/replay计数及所有禁止项。
3. 运行targeted tests、`PYTHONPATH=. JAX_PLATFORMS=cpu .venv_axis_pi05/bin/pytest -q tests`、strict JSON duplicate-key parse、NPZ no-pickle restore、py_compile、ffprobe、`git diff --check`。
4. 更新plan index与Markdown/JSON handoff，clean commit后hard stop。

成功停止状态只能是`track3_support_pending_autogen_v3_ready_for_audit`；失败状态保留具体stage。无论结果如何都不需要用户Teleop或逐帧审批，也不得从本计划直接扩展三族数据、训练LoRA/full-FT或制作ZIP。
