# Track3 Step100 Salvage Closeout Review

> 审核时间：2026-09-16T01:24:27+09:00  
> 执行结论：`approved_fail_closed_execution`  
> 候选结论：`step100_rejected_no_zip`  
> 下一状态：`data_first_recovery_planned`

## 1. 总结

step100 salvage 的执行和拒绝结论成立。GPU0 fresh worker 完成 exact `24/24`，cursor=`completed`、heartbeat=`stopped`；reverse `0/4 -> 0/4` 未达到 `+1`，C20 `15/20 -> 14/20` 满足 retention 但不能单独释放候选。`release_decision.json` 正确写入 `step100_behavior_gate_failed_no_zip`，fold/FAST/ZIP/upload/submission 均未执行。

该审核批准的是“执行闭环与 fail-close”，不是模型。step100 与 step300 均未展示 reverse 增益，因此旧 H300 路线到此结束。

## 2. 独立复核结果

- paired summary SHA：`3e3ff78775d9157cd8d767150225b71ec54dd515375c8d40a24cd088167c7f9b`；
- release decision SHA：`459adb758cc70b51d8155a953a5d0364504fb740f8e481e966e81d6c7f78fd71`；
- comparator SHA：`b9187f2603075c64e0a95b543139c052430feed2bff665dc018b7d891bbb412d`；
- focused test SHA：`3c005ef45280685c8177a0fef8b859222b02d2f69ab80aee90c6313efaa53ffa`；
- focused tests：独立复现 `45 passed in 59.96s`；
- strict JSON：R6 role + step100 namespace 共 `62` 个；
- action trace：`48/48` 为 no-pickle、`N x 7`、finite、nonzero；
- video：`48/48` 非空且独立 `ffprobe` 可读；
- `.partial=0`；旧 evidence fingerprint=`11b059f7...5370`、output fingerprint=`9aecab69...8240`，与 paired summary 一致；
- 旧 step300 paired/release SHA 仍为 `e74a112a...c1b4` / `f7d6cc4f...166b`；
- GPU0–3 均 `5 MiB / 0%`，无 trainer/eval/policy-server/pytest 残留；
- `py_compile` 与 `git diff --check` 通过。

## 3. 行为归因

四条 reverse 在 R6、step100、step300 下全部运行 300 steps并 timeout。三条在三个模型下均 collision+timeout；第四条均无 collision但 timeout。step100/step300 对 R6 的 aligned action RMSE 并非全为零，说明 checkpoint 确实改变了策略，但没有形成任何可计数的 reverse success。

C20 的回退不稳定地落在不同 case：task13 为 `R6/step100/step300=true/true/false`，task18 为 `true/false/true`。这更支持先修数据覆盖与采样，而不是把 step100 或 step300 当作接近成功的 checkpoint继续训练。

## 4. 非阻断性实现发现

这些问题不推翻本次失败结果，因为本轮没有 release，且审核已独立补查；但未来新评测必须修复：

1. `paired_summary.input_hashes.plan_sha256=7361398d...` 是底层 worker 计划 SHA；salvage 计划 `d034d455...` 只在 comparator常量/preflight中锁定，最终 paired/release没有独立命名记录 comparison plan 与 comparator SHA。字段语义可解释，但最终 provenance不够自描述。
2. comparator 的 video gate只检查存在且非空；focused fixture使用任意 `b"video"` 也能通过。实际48段视频经本审核 `ffprobe` 全通过，但 future gate必须内置真实媒体解析。
3. comparator先原子写 paired summary、再写 release decision，同时重复 merge拒绝任一文件已存在；若首次运行恰在两次写入之间中断，会留下不能同 attempt recovery的半关闭状态。

分类均为低风险 lifecycle/provenance remediation，不要求重跑已拒绝的 step100。

## 5. 下一步

唯一下一计划为 `docs/AXIS_PI05_T3_POST_STEP100_FAILURE_DATA_FIRST_R6_SCREEN_PLAN_20260916.md`。先执行 D0 的只读失败/lineage/sampling审计；D1再加固 future finalizer。没有新增独立 reverse数据前，不启动四卡训练、300/500/30k、dense或任何候选发布。
