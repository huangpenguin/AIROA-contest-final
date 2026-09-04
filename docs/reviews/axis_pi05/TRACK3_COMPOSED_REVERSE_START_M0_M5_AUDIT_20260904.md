# Track 3 Composed Reverse-Start M0–M5 独立审核

> 审核日期：2026-09-04（Asia/Tokyo）
> 审核输入：root `eef683b0db9629a3d917ea91ebeb46540a34fcb9`
> 执行前审计：`2a4fdad95603ea77b31d17d87e4606b3ea1350e7`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`
> 用户人工操作：`not_required`

## 1. 获批准的核心结果

- M0历史证据锁与三份公开HDF5闭包成立；旧matched-terminal evidence未改写。
- M2 task20/demo0 canary通过；M3固定CPU matrix确为`9/9`，三族各`3/3`。9条composed-state hash互不相同，均使用same-demo terminal scene qpos + init robot/gripper qpos + zero qvel。
- 9/9 raw terminal与composed state的public forward goal均为true；reverse source在step0–40始终true，reverse goal始终false。所有required dynamics/stability gate通过，restore最大误差为`2.220446049250313e-16`。
- M4确实只运行GPU3上的task10/demo0、task20/demo0、task19/demo0各一次。6段视频的文件hash与summary一致；ffprobe均为H.264、20 FPS、41帧、128×128，无`.partial`残留。
- 12份raw run、6段视频和收尾证据均被Git跟踪。L2/M3/M4后没有训练、Teleop、SE(3)、批量生成、demo晋升、官方GPU、FAST/ZIP或上传。
- 独立复算了旧hard gate漏看的burn-in step1–20：12个run合计demonstrator contact=`0`、unrelated background contact=`0`。因此下面的gate实现缺口没有推翻本轮物理结论。

## 2. 尚未闭合的机器合同

1. M4计划要求显式记录每个fresh process的seed、PID与GPU peak；三个raw report均没有这些字段。`m4_summary.json`只有运行前GPU3 UUID/5MiB快照，不能追溯填补进程内provenance。
2. `contact_window_gate()`只检查t0和step21–40，没有把burn-in step1–20纳入required gate。raw恰好证明这20步无违规，但代码合同仍需修复。
3. `start_state_candidate_report_only=1`只是M2 t0 `composed_state_sha256`；没有`.npy/.npz`状态向量，也没有step20 settled state。它不是可供executor restore的artifact。M3实际有9个已验证的不同composed-state hash，不能用“1个候选”概括可执行输入。
4. `round_trip_reconstruction_max_abs_diff`实际计算的是`max(abs(composed-init))`，数值`0.15–0.99`是有意scene替换量，不是round-trip误差。真实restore误差在`model_state_identity.restore_max_abs_diff`且已通过。
5. machine handoff仍把execution commit写为amend前的orphan `020d291`；最终可复现输入是`eef683b`。
6. `run_index.json`没有run file bytes/SHA-256。raw已经入Git，但下一闭包必须生成独立artifact manifest，不能只依赖路径。

## 3. 审核判断

composed reverse-start的物理与语义验证获批准；M4 provenance和可执行start-state artifact尚未按计划闭合，所以总体不是无条件`approved`，而是`remediation_required`。这不要求用户逐帧看视频、操作Teleop、选择demo或手动批准轨迹。

本次`pending_executor_review`中的“review”就是本审核。下一步可以由agent自动执行：先补齐full-window contact、artifact hashes、9个可恢复state bundle和GPU provenance，再只尝试一条task10/demo0的自动SE(3) support-remove canary；成功后自动fresh replay `3/3`，仍只进入`pending_autogen`，不进入accepted或训练。

## 4. 独立复核

```text
tracked report artifacts     28
tracked raw run reports      12
tracked H.264 videos          6
required gates               all passed
burn-in forbidden contacts    0
unique CPU composed states    9
materialized state vectors    0
targeted tests               26 passed, 2 warnings
```

审核时GPU0仍为外部PID `1133706`；GPU1–3约`5 MiB`。root与nested OpenPI clean，`LIBERO-plus`既有dirty状态保持不变。
