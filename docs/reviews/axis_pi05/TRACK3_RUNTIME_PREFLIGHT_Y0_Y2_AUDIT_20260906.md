# Track 3 Runtime Preflight Y0–Y2 独立审核

> 审核日期：2026-09-06（Asia/Tokyo）
> execution HEAD：`882958c8b863585b13f10f22445d4aa4a42b2d56`
> Y2 clean source：`b16211bc7e48ec96ae8b89b9edf88462e73d3078`
> Y2 evidence closure：`d643e105830ac93622169a53198e2f3373f7b658`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`

## 1. 已独立确认的执行事实

- `y0_recheck_input_lock.json`的45项inventory已逐文件重算bytes/SHA，全部exact，且与首次Y0 inventory一致。首次Y0因把当前launcher计为live process并继续期待已退出的历史GPU0外部PID而失败，原始失败报告与修正理由均完整保留。
- Y2 namespace的10个文件全部Git-tracked；9个JSON均通过duplicate-key rejection。6-entry manifest逐项bytes/SHA exact、无遗漏的非排除tracked artifact；closeout中manifest、final validation与release summary三项自哈希均可复算。
- `y2_attempt1_release_failure.json`保留`source_not_dirty_outside_y=false`的首次失败；最终cursor为inactive，phase CSV把Y3–Y13保持为`not_evaluated`，无partial和Y3 run目录。
- 独立复跑py_compile与定向测试为`10 passed`，全量为`271 passed, 7 skipped, 23 warnings`。审核输入树的root与OpenPI clean；LIBERO-plus仅保留原有dirty。没有live Track3 supervisor/worker；复核时GPU0–3均无compute app。
- `exact/pending/accepted=0/0/0`成立。没有physics、GPU canary、generation、replay、training、norm、FAST或ZIP执行。

## 2. 阻止Y3释放的合同缺口

### 2.1 父进程构造的worker命令无法通过顶层CLI解析

`_run_y3_cpu()`构造`--y3-worker --result-output <path>`，但顶层`main()`只注册`--y3-worker`，并在进入`_y3_worker()`前调用严格`parse_args()`。独立执行`--help`也确认顶层没有`--result-output`。因此当前Y3子进程会在argparse阶段return 2，无法发布child result，也不会进入physics。Y1 smoke只检查helper可调用和三份HDF5路径存在，没有执行真实child argv。

### 2.2 最终hard gate丢失阶段作用域

worker在阶段内对close和lift使用了局部recent window，但循环结束后又用整条轨迹的最后一个window重算`contact_completion`、`object_follow`和`source_release`。成功release/retreat末端本应解除gripper contact，因此会反向否定已通过的close gate；lift gate也被末阶段样本覆盖。`same_demo_destination`只检查place阶段曾进入且存在末端样本，没有验证object/EEF对same-demo destination的数值位置与姿态误差。

这会让合法成功被错判，也可能让不对应阶段的末端状态替代真实阶段证据。每个hard gate必须引用冻结的stage frame span与stable window，后续阶段不得重写先前门禁；未进入的后续阶段仍必须为`not_evaluated`。

### 2.3 source identity仍是自指而非fail-closed

`source_porcelain()`永久排除了worker工具、相关测试、plan index、handoff、MEMORY和machine JSON，因此这些源码或生命周期文件存在未提交修改时仍可能报告clean。Y3又把启动时当前HEAD和worker自己观察到的tool hash直接当作expected值，不能证明它们与审核后的clean-release commit/tool SHA一致。Y1 identity fixture只手工切换`root_dirty`，没有覆盖真实porcelain过滤和冻结身份漂移。

## 3. 阶段定位与影响

当前仍处于“首次有效T3物理canary之前的运行时释放闸门”。Y0完整输入锁和Y2 artifact closure可复用，但Y1的“已可启动Y3”结论不获释放。没有reverse trajectory或训练样本产生，因而对R6 checkpoint、现有forward数据、norm和比赛ZIP没有修改；影响是T3数据生产仍未获得启动资格。

```text
上游资产锁定         Y3 启动资格                    T3 数据链
Y0 45/45 exact  ->  CLI dispatch 修复          ->  单条 CPU canary
Y2 closure exact     frozen source identity         GPU3 camera release
                      phase-scoped ledger           generation / 3x replay
                            ^
                            当前所在位置
```

## 4. 后续授权

旧Y计划按hard stop归档。新的唯一活动计划为`docs/AXIS_PI05_TRACK3_Y3_LAUNCH_PHASE_LEDGER_REMEDIATION_PLAN.md`：先在CPU/code-only阶段闭合真实child argv、冻结源码身份、阶段窗口ledger与destination数值门禁；全部通过并形成新的clean source release后，才允许一次固定task10/demo0/seed3121 nominal CPU canary。后续仍按family隔离、有界generation/replay、数据门禁、LoRA、复验与条件式本地ZIP推进，任何分支最终统一hard stop。

不授权改写历史Y/X evidence、GPU0或官方GPU、Teleop、dense full-FT、失败/partial/holdout入训、自动上传或比赛提交。

## 5. 审计生命周期验证

审核更新plan index、Markdown/machine handoff、归档与生命周期测试后，machine/audit/Y namespace strict JSON、引用存在性、py_compile与`git diff --check`通过；生命周期定向测试为`28 passed, 2 warnings`，最终全量为`271 passed, 7 skipped, 23 warnings`。审核树以clean commit闭包。
