# Track 3 Tonight TZ0–TZ9 独立审核

> 审核日期：2026-09-06（Asia/Tokyo）
> 审核输入 HEAD：`2ae08e974d6be9ac59541236ce7ddb1d7f288dd6`
> 首个 worker source：`1066774918941a574db3f02c7060f40aeb0c9c0a`
> 后六条 worker source：`343c5a554b0bfa05839b145f48a8c785d38edd19`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`

## 1. 获得批准的事实

- 审核开始时root在`2ae08e9` clean，OpenPI在`b58aff0` clean；`LIBERO-plus`仍是原有1项modified和4项untracked。TZ supervisor/两个parent/六个已记录child PID均不存活，cursor inactive、heartbeat final，GPU0–3各5 MiB且无compute app。
- fallback交付成立。source与物化文件是同一inode的hardlink，均为`9,383,571,677` bytes、SHA-256 `95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a`。独立全包`ZipFile.testzip()`返回null，3401 entries，无duplicate/unsafe path；`model_fast.safetensors`恰好一项，无slow权重。内嵌manifest可strict解析，引用的8份validation report在本机均hash exact；C5 source-success为`3/3`。该包只批准为`fallback_no_track3_improvement`，不代表Track3改进，也不建议据此声称新T3能力。
- 七个CPU child attempt确实被消费：首个support在进入env后因`np`未导入而`worker_exception`/`infrastructure_failure`；后六个有唯一child PID、return 0、无timeout，均为`failed_gate`。三族HDF5 model-file、resolved XML与snapshot hash独立复算一致，旧nominal/conservative没有重跑。
- 后六条物理分类可保留为诊断性`family_failure`：support两条各155步，approach通过后descend失败；containment两条各149步，approach通过后descend失败；articulation两条各180步，approach失败。六条均未释放source且goal为false，因此即使不采信工具内硬编码的contact/clip字段，也不可能升格为success或training-eligible。
- 没有成功family，故TZ3–TZ8=`not_evaluated`成立；GPU3/generation/replay/accepted reverse/training/candidate均为0/false。当前文件树和进程边界没有上传、提交、Teleop、dense full-FT或比赛GPU证据。审核生命周期更新前独立复跑targeted为`50 passed, 2 warnings`，项目全量为`293 passed, 7 skipped, 23 warnings`。

## 2. 必须修复的合同缺口

### 2.1 TZ1“完整调度器通过”不成立

旧计划要求首个physics前实现并测试完整链、真实single-instance/heartbeat/identity/finalizer和deadline。实际`TZ5`–`TZ8`是无条件`not_evaluated` stub，`TZ4`即使有GPU pass也固定拒绝promotion；没有训练或candidate实现。`cli_contract()`只是返回`single_instance_supervisor=true`等常量，代码没有文件锁、周期heartbeat、deadline或clean-source身份门禁。AA5“离线复算”也只是无source路径的硬编码摘要。因此TZ1不能批准为`passed`，这正是旧AA审核要求避免的“有flag但无可执行尾部”。

### 2.2 shared infrastructure global stop被跨提交绕过

原计划第3节明确规定worker/finalizer infrastructure failure全局停止。首个profile在`1066774`报NameError后，`343c5a5`修改执行器并在同一计划内继续剩余六条；resume分支对已有`infrastructure_failure`直接`continue`，没有恢复sticky stop。最终`tz2_cpu_matrix`虽然profile行保留首错，却把顶层`infrastructure_failure`写成null并把状态写为`completed`。六条raw失败可保留作诊断，但不能据此批准“TZ2按冻结授权严格完成7/7”。

### 2.3 identity与profile独立性未闭合

profile锁记录了七个不同seed，但worker从未把seed应用到Python/NumPy/env；raw也没有root commit、tool SHA、完整command或profile-lock SHA。后六条运行可按时间定位到`343c5a5`，但这是审核派生，不是执行时identity gate。三对同族NPZ分别byte-identical；原因是每对唯一差异为尚未进入阶段的close dwell，而所谓不同seed没有作用。它们是六个fresh child attempt，不是六个经过有效参数差异验证的独立物理profile。

### 2.4 门禁与TZ9闭包不足

worker丢弃`closed_loop_action`返回的`saturated`，并把`controller_clip=false`、`no_unrelated_contact=true`硬编码；contact completeness未参与success。六个NPZ的`source`/`goal`为object dtype，在`allow_pickle=False`下均抛`ValueError`。TZ9工具没有NPZ/parquet/ffprobe/checkout hash、targeted/full tests、complete artifact manifest或outer hash；它在写最终validation/final/summary前统计旧JSON，后续`2ae08e9`又手改计数。最终summary也缺计划要求的`delivered_zip_kind`、绝对路径和T3提交建议字段。

这些缺口没有把任何失败误升格为成功，也不损害已独立CRC/SHA验证的fallback；但它们阻止批准TZ1/TZ2/TZ9证据合同。

## 3. 当前真实状态

```text
R6 fallback ZIP        approved as honest mechanical fallback
TZ1 scheduler contract remediation_required
CPU child attempts     7 consumed; 1 infrastructure + 6 diagnostic family failures
successful family      0
TZ3–TZ8                not_evaluated
accepted/train/candidate 0 / false / false
```

## 4. 后续授权

旧TZ计划归档。唯一新计划为`docs/AXIS_PI05_TRACK3_TZ_EVIDENCE_AND_EXECUTOR_CONTRACT_REMEDIATION_PLAN.md`：先在新namespace闭合旧证据，再修复真实supervisor/identity/seed/sticky-stop、computed gates、no-pickle trace和finalizer；离线诊断后最多冻结并执行三族各一条全新CPU/no-renderer identity。无论结果如何均hard stop。

新计划不授权GPU0–3、官方GPU、Teleop、generation/replay、accepted数据、训练/norm、新candidate ZIP、上传或比赛提交。fallback保持只读；旧七条identity永久不重跑。

## 5. 审核生命周期验证

生命周期文件更新后，strict duplicate-key JSON、引用/归档检查和py_compile通过；targeted为`53 passed, 2 warnings`，项目全量为`296 passed, 7 skipped, 23 warnings`。最终提交前再次执行`git diff --check`并确认root/OpenPI clean、LIBERO-plus原dirty不变、无live Track3/GPU compute process。
