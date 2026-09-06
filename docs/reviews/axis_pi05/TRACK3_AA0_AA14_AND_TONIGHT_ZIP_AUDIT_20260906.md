# Track 3 AA0–AA14 与 Tonight ZIP 独立审核

> 审核日期：2026-09-06（Asia/Tokyo）
> execution HEAD：`0e7e495d6a8c4dd9abb32525d117eda8166d1a8c`
> AA5 frozen source：`cf27e7b1fe9276eaf0d4285128d56586e049d28f`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`

## 1. 已独立确认的事实

- root在`0e7e495` clean，AA5 source是其ancestor；从`cf27e7b`读取执行工具重算SHA为`3484ffe39d5a80277575ca006f3ea0d8f067b510779584955b833818e917cb79`，与freeze一致。OpenPI clean；LIBERO-plus保持原有5项dirty；AA14/AA5/worker PID均不存活，cursor inactive，GPU0–3各5 MiB且无compute app。
- AA0–AA4核心结果成立。旧Z namespace的36个Git-tracked artifacts逐项重算bytes/SHA为`36/36` exact；AA3真实fixture与finalizer合同成立；AA4 clean identity成立。
- AA5只执行一次task10/demo0、seed3121、新冻结的`object_frame_conservative`，CPU/no-renderer。worker return 0、PID匹配、无timeout，parent=`family_failure`、child=`failed_gate`。
- AA5 trajectory为finite `76×7` actions/`76×79` states，阶段分布是approach/descend/close=`49/21/6`。末端EEF translation error约`0.000576 m`、rotation error约`0.00184 rad`，controller clip为`0/76`；source为`77/77 true`、goal为`0/77 true`。失败集中在acquisition：`object_follow=true`，但`contact_continuous=false`、`source_released=false`，未进入transport/destination。
- AA namespace实际有35个文件；AA14 manifest内31项均bytes/SHA exact，另外4项被显式排除，outer record对manifest/validation/closeout三项重算exact且只排除自身。26个namespace JSON加machine handoff共27个均可strict解析；NPY/NPZ no-pickle/finite、CSV LF、无partial。
- 独立复跑用户所述targeted组合为`44 passed`，项目全量为`287 passed, 7 skipped, 23 warnings`。
- AA6–AA13确实没有执行，`exact/pending/accepted=0/0/0`；没有GPU3、generation、replay、training、Teleop、ZIP、上传或提交。

## 2. 必须修复：family isolation没有落实到执行器

旧计划明确规定正常`family_failure`只关闭对应lane，并特别写明support失败不阻止task20/task19合同。但实际工具只实现并暴露AA0–AA5和AA14；AA6–AA13没有CLI入口。`_aa14()`又把任何AA5非success直接映射成AA6–AA13全部`not_evaluated`。因此本次hard stop本身安全，但它错误地把support acquisition failure扩散成全链停止，未执行的containment/articulation没有失败证据。

这也是一周没有新ZIP的直接流程原因：工程检查和support单点canary不断增长，而独立family、accepted数据、训练和包装从未真正获得可执行入口。后续不能再用新的单点计划逐次审核；必须在首个physics前实现完整的有界调度器，并让正常family failure隔离。

## 3. 当前真实进度

```text
运行时/证据闭环       已完成
support自动执行器      到达抓取位，但未抓住
containment/articulation 未评估
accepted reverse数据   0
T3 LoRA                 未评估
新T3 ZIP                0
旧R6 FAST fallback      已存在且SHA复验一致
```

旧fallback为`artifacts/submission_axis_r6_step500_fast_positive_score_20260831.zip`，大小`9383571677` bytes，SHA-256=`95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a`。它已产生Track1/2正分，但用户报告Track3=`0`；重新物化只能保证今晚有机械有效ZIP，不能算T3模型进步。

## 4. 下一步

旧AA计划归档。唯一新计划改用范围代号`TZ0–TZ9`，先复验并物化明确标注的fallback ZIP，再一次性完成family隔离、最多7条CPU pilot、最多3条GPU3 camera、最多6条generation/18次replay、一条100-step LoRA和条件500-step continuation。23:00 JST前必须定版为合格T3 candidate或诚实标注的R6 fallback；不上传、不提交。

完整计划：`docs/AXIS_PI05_TRACK3_TONIGHT_ZIP_AND_CROSSFAMILY_RECOVERY_PLAN.md`。

## 5. 审核生命周期验证

plan/index/Markdown与machine handoff、归档和生命周期测试更新后，strict JSON、引用存在性、py_compile及`git diff --check`通过；targeted为`45 passed`，项目全量为`288 passed, 7 skipped, 23 warnings`。最终树提交后仍须保持root/OpenPI clean、LIBERO-plus原dirty不变且无live Track3/GPU compute process。
