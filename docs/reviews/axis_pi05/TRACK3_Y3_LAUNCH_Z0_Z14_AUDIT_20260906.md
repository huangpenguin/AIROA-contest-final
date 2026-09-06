# Track 3 Y3-Launch Z0–Z14 独立审核

> 审核日期：2026-09-06（Asia/Tokyo）
> execution/final handoff HEAD：`5a270a912362376cbfa13a7f9e9fd66c8b5d6237`
> Z5 frozen source：`933f4d3afb70ca31c95d55881d404b97cb3687d5`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`

## 1. 已独立确认的执行事实

- Z0 inventory的45项上游输入已逐文件重算bytes/SHA，全部exact。Z4 manifest `11/11`、Z14 manifest内的`28/28`均逐项bytes/SHA exact；namespace内26个JSON均通过duplicate-key rejection，3个NPY/2个NPZ可在`allow_pickle=False`下读取且所有数值array finite，3个CSV均为LF，无partial。
- Z1 exact argv确实进入no-physics worker，缺失/未知/重复参数return 2；Z2真实dirty/commit/tool/OpenPI/command/CUDA身份fixture fail-closed。Z5的Git note与`identity_preflight.json`共同锁定root `933f4d3`、tool SHA `bf29061a...`、OpenPI、完整命令和LIBERO-plus dirty snapshot；独立从该commit取tool内容重算SHA一致。
- Z5只运行task10/demo0、seed3121、`object_frame_nominal`一次。parent/child PID一致、无timeout；worker在20.58秒内完成28个CPU physics steps并原子写出`failed_gate`。trajectory是finite `28×7` actions与`28×79` states，所有phase均为`approach_grasp`；source trace `29/29=true`，goal trace `29/29=false`。
- 归一化pose error从约`22.86`持续下降到`0.531`，随后回摆到`1.077`；冻结4-step window的改善为`-0.5459 < 0.02`，故按预注册合同触发`stage_stall:approach_grasp`。close/lift/transport/place/release均未进入，对应contact、follow、source-release、destination与goal gate均为`not_evaluated`。
- worker在写完child result、trajectory和ledger后才因release source缺少`WORKER_COMPLETED_STATUSES`发生NameError并return 1；历史parent据此标为`infrastructure`并正确全局停止。`614220d`只补回`{"passed", "failed_gate"}`常量，没有重跑canary。
- Z6–Z13未执行，`exact/pending/accepted=0/0/0`；没有GPU、conservative、其他family、generation、replay、training、norm、FAST或ZIP。审核时root/OpenPI clean，LIBERO-plus原dirty未改，无live Track3 process，GPU0–3均无compute app。
- 生命周期更新前独立复跑targeted为`10 passed`，全量为`281 passed, 7 skipped, 23 warnings`。

## 2. 审核解释与必须修复项

### 2.1 历史基础设施标签保留，但物理层可派生为nominal family failure

历史parent的分类与return code合同一致，不修改。另一方面，child `failed_gate`、完整trajectory、phase ledger和原子publication均发生在finalizer NameError之前；该异常只污染process return code，没有改变28步physics或stall判定。因此新审核可以派生：`object_frame_nominal`物理结果为有效family failure，nominal配额已消耗且禁止重跑；原X2中结果前冻结的`object_frame_conservative`仍可在新身份下获得最多一次CPU机会。

### 2.2 Z14的`28/28 exact`不是完整artifact closure

namespace共有36个Git-tracked文件。Z14 manifest明确排除Z4/Z14的manifest、validation、closeout六项，又通过代码隐式排除两个Z5日志；后两项没有列入`entries_excluded`。其中`stderr.log`正是finalizer NameError的原始根因证据。Z14 closeout没有记录这八项的checkout-stable hashes，也没有计划要求的manifest/final-validation/closeout自哈希。因此28项内部正确，但不能表达36项完整覆盖。

独立重算的八项SHA如下，必须在新audit/derived namespace闭包，旧Z文件保持不变：

- Z4 manifest/final-validation/closeout：`ab81adb1...` / `34c27d40...` / `b838136e...`
- Z14 manifest/final-validation/closeout：`145da7ca...` / `9eea1571...` / `cd871955...`
- Z5 stderr/stdout：`9b760bb2...` / `e3b0c442...`

### 2.3 postmortem与交接有两项可纠正的派生记录错误

- `z5_infrastructure_postmortem.json`从`hard_gate`顶层读取`progress_stall/contact_completion/object_follow/goal_continuity`，而raw child把这些放在`hard_gate.gates`，所以postmortem中的`approach_stall`及多项scope为null；destination因child另有顶层别名才显示`not_evaluated`。新derived report必须从raw gate map重建，不修改旧postmortem。
- Z4 final validation原文是targeted `9 passed`、full `280 passed`；plan index/handoff曾写`8 passed`，machine后续又写当前targeted `10 passed`。三者分别属于Z4当时、旧交接笔误、当前修复后测试，不应混为一个计数。
- Markdown handoff顶层状态仍停在`track3_z4_prephysics_release_ready_for_z5`，与plan index、machine JSON和Z14 closeout不一致；本次审核生命周期统一修正当前指针，历史§67–§68不删除。

## 3. 阶段定位与影响

当前已越过“首次physics是否能启动”的门槛，但仍未到T3数据生成：nominal在approach阶段按冻结门禁关闭，任何抓取、搬运与放置能力仍未被评价。下一步先闭合artifact/derived evidence，再执行最多一次既有conservative CPU profile；不能把本次28步轨迹当作训练数据。

```text
Z0–Z4 runtime通过 -> Z5 nominal: approach stall -> artifact/derived闭包
                                                  |
                                                  v
                                  1次 conservative CPU -> GPU3 release -> 数据生成
```

## 4. 后续授权

旧Z计划归档。新的唯一活动计划为`docs/AXIS_PI05_TRACK3_Z5_DERIVED_RELEASE_AND_CONSERVATIVE_CONTINUATION_PLAN.md`：AA0–AA4先完成36/36 artifact closure、派生family分类、真实failed-gate return-code/finalizer回归与clean-source freeze；AA5禁止nominal重跑，只允许一次结果前已冻结的support conservative CPU canary。共享基础设施失败全局停止；正常family failure只关闭support lane，并可继续公开task20/task19合同与独立family pilots。严格数据门禁通过后才允许bounded generation/replay、mixture/norm、R6 LoRA与条件式本地ZIP。

不授权改写历史Z evidence、把Z5轨迹入训、GPU0/官方GPU、Teleop、dense full-FT、自动上传或比赛提交。

## 5. 审计生命周期验证

更新plan index、Markdown/machine handoff、MEMORY、归档和生命周期测试后，strict JSON与引用检查通过，py_compile通过，targeted为`11 passed`，项目全量为`282 passed, 7 skipped, 23 warnings`；直接从仓库根运行无范围参数的pytest会误收集缓存第三方源码和nested OpenPI，并因未安装可选`scenic`/`pynvml`在collection退出，未运行任何physics。最终仍须通过`git diff --check`和资源边界复核，再以clean commit闭包。
