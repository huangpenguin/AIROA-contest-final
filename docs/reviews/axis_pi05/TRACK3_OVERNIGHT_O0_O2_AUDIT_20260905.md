# Track 3 Overnight O0–O2 独立审核

> 审核日期：2026-09-05（Asia/Tokyo）
> 审核输入：root `8c0d296f1dfedb0c777040a733b6f9e352234111`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`
> 用户人工操作：`not_required`

## 1. 获批准的事实

- O0 release lock与O1 parent/result/watchdog合同通过。
- task10/demo0 attempt3只启动一次；没有attempt4。
- child `status=passed`，restore、truth、contact-v3全窗口、camera schema四类hard gate均通过。
- restore最大绝对误差仍为`2.220446049250313e-16`；source在完整窗口为true，goal为false。
- agentview与wrist均为41帧、20 FPS H.264；NPZ使用`allow_pickle=False`，无`.partial`残留。
- parent正确记录seed `3100`、parent PID `1416912`、child/result PID `1416920`、物理GPU3 UUID、起止快照、峰值`459 MiB`、return code `0`、`timed_out=false`与`113`次GPU采样。
- O2失败后没有运行task20/task19，没有进入O3–O9；reverse `exact/pending/accepted=0/0/0`，没有训练、FAST、ZIP、官方GPU、上传或提交。
- 独立复跑全量CPU测试：`244 passed, 7 skipped, 23 warnings`。
- root与OpenPI clean；GPU0外部进程未触碰，GPU1–3已释放；`LIBERO-plus`既有dirty状态保持。

## 2. Finding A：poll字段语义和hard gate错误

旧代码在每次同步执行`gpu_snapshot(3)`前后计时，并把该值写为`parent_poll_interval_max_ms`。因此`148.45660293940455 ms`实际是单次`nvidia-smi`查询延迟，不是两次heartbeat之间的间隔，也不是子进程失联时间。

该运行同时满足：113次采样、完整start/end snapshot、正确PID/UUID、峰值显存、正常return code、无超时以及child全部hard gate。把一次查询延迟超过`100 ms`作为provenance完整性失败没有安全依据，属于监控器误分类；不能解释为task10物理或逆向语义失败。

审核要求：

1. 历史`parent_result.json`保持`failed_closed`不改写。
2. 新合同把query latency与heartbeat gap拆成不同字段，记录count/min/p50/p95/p99/max。
3. query latency只作performance warning；hard gate只覆盖字段缺失、PID/UUID/CUDA映射不一致、snapshot失败、采样覆盖不足、heartbeat长时间失联、timeout或非零return code。
4. 对attempt3创建只读derived reanalysis；通过后可释放task10，不需要GPU attempt4。

## 3. Finding B：O10 CSV manifest不是checkout-stable

`o10_artifact_manifest.json`共10项，其中9项当前SHA-256一致；`o10_phase_results.csv`不一致：

- manifest记录：`ad4f84c9d715d6a6f22a0e37ed56b613d8d771134fae45c327ad8dc45dfeccc5`
- 当前检出文件：`c39b4d3069cd9431b26c0b01572a03cb264db6538a33a7461cac3570309b8af8`

前者对应CRLF，后者对应Git规范化后的LF。旧证据不得静默改写。下一轮必须固定CSV `lineterminator="\n"`，在新namespace生成derived closure manifest，并验证写入前、`git add`后、clean checkout等价字节hash。

## 4. 当前阶段判定

```text
reverse start/state bundle       complete
worker I/O and child camera      complete
task10 physical P3 evidence      complete, pending derived release
parent telemetry semantics       remediation required
task20/task19 P3                 not evaluated
manipulation executor            not started
accepted reverse demonstrations  0
Track3 LoRA / FAST / ZIP         not started
```

因此当前仍不在训练阶段，但不需要回到state、HDF5、Axis init或task10物理canary重做。最短正确路径是CPU修复telemetry/manifest合同，升格现有attempt3，然后从task20/task19继续。

## 5. 审核决定

旧O0–O10计划完成其获执行分支并归档。批准唯一后续计划：

`docs/AXIS_PI05_TRACK3_PROVENANCE_JITTER_RECOVERY_AND_OVERNIGHT_CONTINUATION_PLAN.md`

它可以在修复J0–J2后连续运行其余family、executor、数据生成、LoRA筛选和条件式本地ZIP；不需要用户Teleop或逐阶段人工确认。GPU0、官方GPU、dense full-FT、失败轨迹入训与自动上传/提交继续禁止。
