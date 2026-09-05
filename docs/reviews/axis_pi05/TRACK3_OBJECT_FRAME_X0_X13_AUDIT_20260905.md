# Track 3 Object-Frame X0–X13 独立审核

> 审核日期：2026-09-05（Asia/Tokyo）
> execution commit：`64a0d31c3b2864562887044bad7ef8b522a213db`
> X4 clean source commit：`ce86e724dddf300406465d6986577199d39cc19b`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`

## 1. 成立的执行事实

- X0–X3 的报告、object→EEF exact destination 数值、三态 fixture、identity fixture与W4 derived acquisition barrier均可复算；历史W/V/J证据未被改写。
- X4只创建一个task10/demo0、seed3121、`object_frame_nominal` CPU worker。identity gate通过，child PID与parent记录一致；child因调用不存在的`track3_reverse_init_stabilization._configure_public_env`返回`worker_exception`，parent以return code 1正确分类为`infrastructure`并全局停止。
- 异常发生在`ControlEnv`导入、构造、reset和任何`env.step`之前。依赖模块导入产生了robosuite warning，但没有物理轨迹、start/end state或可训练数据，因此本轮不能形成任何executor物理结论。
- conservative profile、GPU3、task20/task19、generation/replay、mixture/norm、LoRA、评估与ZIP均未执行；X5–X12保持`not_evaluated`，`exact/pending/accepted=0/0/0`。
- X13的13个manifest entry全部Git-tracked且bytes/SHA exact；namespace内14个JSON通过duplicate-key rejection，CSV为LF，无partial。独立复跑模块测试为`3 passed`，全量为`260 passed, 7 skipped, 23 warnings`。
- 审核时root与OpenPI clean；LIBERO-plus只保留既有dirty。X4/X13 PID均不存在；GPU0外部PID 1133706保留，GPU1–3空闲。

## 2. 必须补救的合同缺口

### 2.1 X0并未完成计划要求的完整输入锁

`x0_input_lock.json`复核了旧W6 manifest、W4分类、commit ancestry和资源边界，但没有逐项记录/重验计划指定的W4 raw trajectory、state bundle、三份public HDF5、R6 raw/folded checkpoint与norm hash。报告内部`passed`不能替代缺失的immutable inventory。

### 2.2 X2的运行路径没有实现其声称的门禁

- `base._configure_public_env`在被调模块中不存在；X3的py_compile和三个单元测试不会发现运行期属性错误。
- worker每帧写入`progress_window_gate`，但从未在window=`failed`时设置stall或停止；真实执行仍只能耗尽stage budget并标为timeout。
- 计划要求close→lift只验证连续gripper↔object contact，lift→transport再验证object-follow与source release。当前代码却在`close_gripper`之后、lift之前要求source已经false，阶段顺序错误。
- supervisor只在worker起止各写一次heartbeat；长run没有按冻结间隔周期更新。

这些问题意味着X0–X3的内部报告可以作为工程中间产物保留，但不能释放新的物理canary或后续数据链。

### 2.3 X13未完整兑现finalizer自哈希约定

13-entry manifest本身正确覆盖除manifest/final-validation外的全部tracked artifacts；但X1承诺manifest自身与final-validation另有checkout-stable hash记录，最终证据中没有这两个字段。独立审核得到：

- `x13_artifact_manifest.json`：`ce4a815669c691015855c51a6d4c9627f7094e557e3b1c2fed881cd8f274a3a4`
- `x13_final_validation.json`：`e7512442d4c88e69fb215ffa27168c60d90df08a0d564fb2c59fb7afec984bd8`

## 3. 阶段定位与影响

当前位于“执行器工程合同完成了一部分、首次有效物理canary尚未开始”的边界。T3 reverse数据生产对应旧X8，本轮停在X4 worker bootstrap，距离数据生成仍需经过：运行合同补救 → support CPU物理canary → GPU3 camera release → 其余family资格。现有`0/0/0`不会影响R6 checkpoint、forward数据、norm或比赛ZIP；影响只是T3数据链没有获得启动资格，不能据此判断executor物理可行或不可行。

## 4. 后续授权

旧X计划按hard stop归档。新的唯一活动计划为`docs/AXIS_PI05_TRACK3_RUNTIME_PREFLIGHT_AND_BOUNDED_CONTINUATION_PLAN.md`：先锁全量输入、修复运行期symbol/phase/stall/heartbeat/finalizer并形成clean release；只有这些门禁通过，才允许在新namespace中执行一次同identity nominal物理canary。后续仍按family隔离、有界generation/replay、数据门禁、LoRA与条件式本地ZIP连续推进；任何共享基础设施错误立即统一收尾。

不授权修改历史X evidence、GPU0/官方GPU、Teleop、dense full-FT、失败/partial/holdout入训、自动上传或比赛提交。

## 5. 审计生命周期验证

更新plan index、Markdown/machine handoff、归档与生命周期测试后，定向测试为`18 passed, 2 warnings`，最终全量为`261 passed, 7 skipped, 23 warnings`；strict JSON、引用存在性与`git diff --check`通过。
