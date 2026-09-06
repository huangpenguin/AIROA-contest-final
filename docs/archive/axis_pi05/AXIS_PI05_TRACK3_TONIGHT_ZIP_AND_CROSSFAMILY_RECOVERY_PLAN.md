# AXIS Pi0.5 Track 3：Tonight ZIP 与 Cross-Family Recovery 计划

> 制定日期：2026-09-06（Asia/Tokyo）
> 审核输入：root `0e7e495d6a8c4dd9abb32525d117eda8166d1a8c`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 当前状态：`track3_tonight_zip_and_crossfamily_recovery_completed_and_audited_remediation_required`
> 唯一授权链：`TZ0_to_TZ9_then_hard_stop`
> 当晚截止：2026-09-06 23:00 JST 前必须留下至少一个经过CRC/SHA/布局复验的本地ZIP
> 用户人工操作：`not_required`
> 不授权：GPU0、官方GPU、Teleop、dense full-FT、隐藏任务信息、历史证据改写、自动上传或比赛提交

`TZ`是“Tonight ZIP”的范围代号，不是继续按字母累计进度。计划同时管理两个不同诚实度的交付物：通过reverse holdout门禁的新T3 candidate；若物理/数据/训练链未及时通过，则保留明确标注`fallback_no_track3_improvement`的R6基线ZIP。二者不得混淆。

## 0. 总目标与硬截止

- 先在TZ0把既有`submission_axis_r6_step500_fast_positive_score_20260831.zip`重新计算SHA、CRC、布局和FAST-only约束；以hard-link优先、reflink/copy兜底，物化`artifacts/submission_axis_r6_fallback_revalidated_20260906.zip`及sidecars。它只保证今晚有一个可读、机械有效的本地ZIP，不宣称Track3改进。
- 同一条计划继续修复AA错误耦合，批量验证support/containment/articulation，不再因单个正常`family_failure`提前停止其他family。
- 只有machine-accepted reverse数据、holdout提升、C20 retention、fold/parity和FAST门禁全部通过，才生成`artifacts/submission_axis_track3_tonight_candidate_20260906.zip`。
- 22:30 JST停止启动新的长训练；23:00前完成candidate或fallback的最终manifest/SHA。任何ZIP均只在本地，不上传、不提交。

## 1. TZ0：审核锁与 fallback ZIP（CPU/filesystem，优先完成）

- 锁定AA审核、`0e7e495`、R6 raw/folded/norm/processor、旧正式正分ZIP及其sidecars；重验root/OpenPI/LIBERO-plus、PID/cursor/heartbeat、GPU和530 GiB级可用磁盘。
- 对旧ZIP执行完整SHA-256、`ZipFile.testzip()`/CRC、entry路径、大小、`model_fast.safetensors`存在、slow权重排除与C5 source-success检查。
- 物化新的fallback路径，不重新压缩权重；记录与源ZIP相同的bytes/SHA和`readiness=fallback_no_track3_improvement`。若基线ZIP复验失败，不能伪造fallback，但后续T3 recovery仍可继续。

## 2. TZ1：完整调度器、family隔离与离线抓取诊断（CPU/code only）

- 在任何physics前实现并测试TZ0–TZ9所有入口、single-instance supervisor、atomic cursor/heartbeat/result、TERM→KILL、返回码、manifest/finalizer和截止时间降级路径；禁止再次出现“文档有后续阶段、CLI没有入口”。
- 从AA5 trajectory与公开同demo HDF5离线复算gripper符号、夹爪joint闭合时间、release-anchor双指接触几何、approach/descend末态和接触窗口。AA5已证明49步approach、21步descend、6步close、零controller clip且EEF到位，但未形成连续接触/释放source。
- 在首个新physics前一次性冻结至多7个新identity：support contact-repair至多3个、containment至多2个、articulation至多2个。只能改变由离线证据支持的grasp offset/close dwell/contact阶段参数；旧nominal/conservative identity不得重跑，结果后不得调参。

## 3. TZ2：一次性CPU pilot矩阵（最多7次）

- 固定顺序task10 support repair→task20 containment→task19 articulation；每个冻结identity最多一次、CPU/no-renderer、fresh process。
- 正常`family_failure`只关闭该identity，绝不阻止剩余family；只有source/model/runtime污染、非finite、进程/身份漂移等shared infrastructure错误才全局停止。
- 至少一个family通过完整acquisition、transport、destination、release和goal gate才进入TZ3；失败轨迹全部denylist。

## 4. TZ3：GPU3 camera release（每个成功family最多1次）

- 只对TZ2每族最先通过的profile运行一次双相机20 Hz canary，GPU3单进程，要求physics/phase/provenance一致、H.264、无partial并释放GPU。
- camera failure只关闭对应lane；其他family继续。至少一个family通过才进入TZ4。

## 5. TZ4：最小reverse数据与3/3 fresh replay（GPU3）

- 每个released family只生成demo0/demo1各一条train trace；demo2永久holdout。总generation≤6、fresh replay≤18。
- `accepted`必须producer gates全通过且3/3 fresh replay全通过；pilot、failed、partial、AA5与holdout轨迹禁止入训。
- 至少一个family得到2条accepted train traces才进入TZ5；否则直接保留fallback并转TZ9。

## 6. TZ5：20 Hz mixture与fresh QUANTILES（CPU）

- forward只用锁定的本地LIBERO-plus，reverse只用TZ4 accepted；固定sampling=`80/20`，按family/start分层。
- 重算per-dimension `QUANTILES`；training、raw rollout、folded、LeRobot processors和FAST必须引用同一manifest/norm SHA。

## 7. TZ6：单臂R6 L2 LoRA 100-step screen（GPU1+2）

- 从`r6_dual_l2_fresh500_r1` step500 params初始化，optimizer/scheduler/EMA fresh；`L2_action_lora_heads`、FP32、peak LR=`1e-6`、microbatch1/device、accumulation32、global batch64。
- 保存0/1/10/100；restore、loss/grad/update/params/EMA必须finite。失败直接使用fallback，不切dense full-FT或第二训练臂。

## 8. TZ7：即时holdout/C20与条件500-step continuation

- 比较step0与step100：released-family demo2 execute8 reverse至少`+1` success，固定C20最多下降`2/20`。
- 通过且预计可在22:30前完成时续到500并保存250/500；否则step100只作为诊断，不得打包为candidate。
- 最终选择仍须reverse holdout≥1 success且C20合格；缺失结果=`not_evaluated`，不是通过。

## 9. TZ8：candidate/fallback ZIP定版

- 合格winner执行fresh-process复验、raw/folded parity≤`1e-3`、processor action误差≤`1e-5`、FAST envelope/latency、C5 coverage/source-success、ZIP CRC/布局/大小/SHA和slow权重排除，然后生成T3 candidate ZIP。
- candidate未通过时，不生成含糊命名的“新模型ZIP”；交付TZ0的revalidated fallback及其`fallback_no_track3_improvement`sidecar。
- 最终summary必须明确`delivered_zip_kind=track3_candidate|r6_fallback`、绝对路径、bytes、SHA、是否推荐T3提交。不得上传或提交。

## 10. TZ9：统一收尾与hard stop

- 停worker并释放GPU，写phase/family/episode/checkpoint/ZIP summary、inactive cursor、LF CSV、complete manifest、outer hash和final validation。
- 重验strict JSON、NPY/NPZ/parquet、CSV LF、ffprobe、ZIP CRC、checkout hashes、py_compile、targeted/full tests、`git diff --check`；root/OpenPI clean commit，LIBERO-plus既有dirty不改。
- 不论交付candidate还是fallback，23:00 JST前hard stop等待审核；不自动上传或比赛提交。

## 11. 本轮执行结果（2026-09-06，TZ9 hard stop）

- TZ0/TZ1通过；fallback已物化并完成全包CRC/SHA/布局复验，交付标签为`fallback_no_track3_improvement`。来源和物化ZIP均为`9383571677` bytes、SHA `95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a`，entry count `3401`，C5 source-success `3/3`。
- TZ2执行上限内的7/7 CPU children。support为1条执行器infrastructure failure（numpy延迟导入bug，已修复且不重跑该identity）加2条正常family failure；containment `2/2`、articulation `2/2`均为正常family failure。没有family success，family isolation保留；旧nominal/conservative identity未重跑。
- 因无成功family，TZ3–TZ8全部为`not_evaluated`；GPU3、generation、replay、accepted reverse data、training和T3 candidate均为0/false。TZ9交付本地R6 fallback并hard stop，下一权限为`new_reviewed_plan_required`。
- 机器证据：`data/reports/axis_libero_30k/track3_tonight_zip_and_crossfamily_recovery/summary.json`；本轮没有上传或提交，`LIBERO-plus`既有dirty未触碰。
- 最终测试为目标`50 passed`、项目全量`293 passed, 7 skipped, 23 warnings`；Tonight namespace strict JSON `23/23`（含machine handoff为`24/24`）、LF CSV、py_compile、diff-check、无partial和GPU0–3空闲均通过。
