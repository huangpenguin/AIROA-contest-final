# AXIS Pi0.5 Track 3：Operator-Scoped Stability Release 执行计划

> 制定日期：2026-09-04（Asia/Tokyo）
> 输入执行 commit：root `0dec0b1930a3d2ad9f196febe1aeaabb0c89445d`，nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`
> 当前状态：`track3_operator_scoped_stability_release_planned`
> 唯一授权：执行 J0–J4，随后 hard stop
> 不授权：修改 BDDL/region/predicate、SE(3)、抓取/demo、Teleop、批量生成、训练/norm、FAST/ZIP、官方 GPU、上传或比赛提交

## 0. 目标与原则

本计划不放宽旧 I2 的全场 `2 mm / 0.02 rad` 阈值，也不覆盖其 failure。目标是新增一个预注册、按 operator role 划分的 release contract：判断 reverse start 的任务拓扑与实际操作对象是否稳定，同时完整保留背景物自然沉降的诊断信息。

新产物仅写入：

```text
data/reports/axis_libero_30k/track3_operator_scoped_stability/
```

旧 `track3_reverse_init_stabilization/` namespace 全部只读。

## 1. J0：输入锁与实现边界（CPU only）

输出 `j0_release_lock.json`，锁定：

1. root `0dec0b1...`、parent `1a0b387...`、nested OpenPI `b58aff0...`；root/OpenPI clean；只读记录且不触碰 `LIBERO-plus` 既有 dirty porcelain；
2. I0/I1/I2/I4、12 个 per-run JSON、三族 forward/reverse BDDL、LIBERO predicate/evaluator source、审核报告的 hash/bytes；
3. I2 summary SHA-256 `e72ae24d27b6edbba37b043a88042191f8a1c7d00c298b812653a1a2ecc17dab`；
4. exact/pending/accepted=`0/0/0`、I3=`0`、无 SE(3)/demo/训练/Teleop/ZIP/上传；
5. GPU0 外部 PID 仅记录，不触碰或终止；J0–J2 不使用 GPU。

任一 immutable input 漂移、root/OpenPI 预存 dirty 或计数非零，状态置为 `track3_operator_scoped_input_drift` 并停止。

## 2. J1：冻结 operator-scoped contract（CPU/code gate）

实现新的只读分析器和测试；不得修改旧 I2 工具/JSON或 BDDL。contract 必须在运行 J2/J3 前写入 `j1_contract.json` 并 hash 锁定。

### 2.1 全族共同 hard gates

固定 release window 为 step5–20，20 Hz，7D 全零 OSC_POSE，gripper=`0.0`。每个 reverse run 都必须满足：

1. schema、finite、truth/detail、OS return code 完整；
2. operator-critical source 在每个 step 为 true；reverse goal 在每个 step 为 false；
3. step5 的完整 init predicate truth vector 与 goal predicate truth vector在 step6–20 零跃迁；
4. robot 各字段继续使用 I1 已锁的分量/单位阈值；
5. 所有 object/site pose 与 fixture joint finite，名称集合不变；
6. contact telemetry 必须完整，并按下述 family-specific contract 区分预期 receiver contact 与非预期 background contact。

### 2.2 family-specific hard gates

- `support_remove`：`On(bowl, plate)` 连续为 true；bowl↔plate contact 每步存在且不得新增 bowl↔非plate背景物 contact；bowl COM 相对 plate 的 runtime support 几何持续有效；bowl translation 使用原 `2 mm` 阈值，plate translation/rotation使用原 `2 mm / 0.02 rad`。bowl 绕/贴合 support 的 rotation 只记录，不作为该算子的独立失败条件。
- `containment_remove`：`In(orange_juice, basket_region)` 连续为 true；orange juice 与 basket/contain region的 translation/rotation分别使用原 `2 mm / 0.02 rad`；允许orange juice↔basket预期contact，不允许新增orange juice↔非basket背景物contact；背景 sauce/ketchup 的 motion 不能替代关键 participant gate。
- `articulation_close`：`Open(middle_region)` 连续为 true；目标 fixture joint与 middle site使用原 `2 mm / 0.02 rad`；目标drawer不得与松散背景物新增contact，背景 bowl/plate motion不能替代目标 joint/site gate。

### 2.3 背景诊断必须保留

逐 object/site 报告 step5→20 max translation/rotation、最大值出现 step、predicate truth transitions、critical-background contacts。旧全场 max 与旧统一门禁结果原样引用，不能删除、改名为 pass 或从报告中隐藏。

J1 单测至少覆盖：participant mapping、bowl rotation 非独立 hard gate、background max 不冒充 critical max、predicate transition fail-closed、missing contact/schema/pose fail-closed、旧 I2 immutability。

## 3. J2：CPU reanalysis 与原生 forward 对照

### J2.1 只读复算现有证据

对 I2 的 9 个 reverse runs 与3个 task10 forward runs复算，不重新执行 simulator。输出 `j2_existing_i2_reanalysis.json`：

- 逐 run 写出完整 predicate transition、critical/background attribution、family-specific gate、旧 full-scene gate；
- 对历史文件中已保存的字段必须得到 9/9 reverse operator-scoped pass；历史未记录的完整contact matrix必须明确为`not_evaluated`，不得伪造为pass；
- 任一已声明字段的 missing/shape/hash/schema 问题均 fail-closed，不可从 summary 数值反推补齐。J2.1不能单独释放J3。

### J2.2 补齐 forward natural-settling 对照

使用与 I2 相同的 native fresh-process、step0–20、zero-action contract，并新增完整contact matrix telemetry。执行：

```text
task10 reverse           3 runs
task20 reverse           3 runs
task19 reverse           3 runs
task20 original forward  3 runs
task19 original forward  3 runs
```

共15个new CPU fresh processes；连同旧 task10 forward 3 runs形成三族各3条forward对照。对照只用于确认官方原生任务也允许“predicate topology不变但自由背景物发生自然沉降”，不能用forward观测值事后生成或提高reverse阈值。

J2 输出 `j2_reverse_release.json`、`j2_forward_controls.json` 与 `j2_summary.json`。若9/9新reverse operator-scoped gate（含contact）、6/6新forward process完整、所有immutable hashes重验不同时成立，状态置为`track3_operator_scoped_cpu_release_failed_<reason>`并执行J4；不得进入GPU。

## 4. J3：GPU3 production-like 动态 release

仅在 J2 全通过、GPU3 自然空闲且 GPU0 外部进程未被触碰时执行。只允许 GPU3；禁止 GPU0/1/2。使用 frozen J1 contract、同一代码 commit/hash、20 Hz、真实 camera/EGL observation 路径。

### J3.1 task10 单次 canary

执行一次 fresh task10：reset → 5 zero actions → step5–20 measurement。必须通过全部 common/support hard gates，并生成逐帧 JSON与诊断 H.264。随后仅用直接 simulator-state perturbation验证：source predicate可被置 false、restore 后可恢复 true；该 sensitivity probe 不是 action trajectory，不得写入 pending/accepted/demo。

失败即状态 `track3_operator_scoped_gpu_failed_support_<reason>`，执行 J4 停止。

### J3.2 三族 release matrix

J3.1 通过后，在不修改 code/contract/spec/threshold 的前提下运行：

```text
task10 support_remove       3 fresh processes
task20 containment_remove   3 fresh processes
task19 articulation_close   3 fresh processes
```

最大 `9` 次。要求 9/9 operator-scoped pass、sensitivity truth/restore 9/9、视频/JSON完整、exact/pending/accepted仍为 `0/0/0`。任何一条失败立即停止剩余运行并写精确 family/reason，禁止 retry 或调阈值。

全部通过的唯一成功状态：

```text
track3_operator_scoped_dynamic_validated_pending_se3_review
```

此状态只批准 reverse init/validator，不批准生成训练数据。

## 5. J4：收尾、测试与 hard stop

无论成功或失败均：

1. 输出 `summary.json`、`J4_CLOSEOUT_REPORT_ZH.md`，使用 passed/failed/not_evaluated 三态；
2. 记录每阶段执行次数、GPU UUID/PID/峰值、输入/输出 hash、old-vs-new gate并列结果；
3. 重验旧 I2 namespace hash、reverse-demo `0/0/0`、root/OpenPI clean、`LIBERO-plus` 未触碰；
4. 运行 targeted tests、`JAX_PLATFORMS=cpu .venv_axis_pi05/bin/pytest -q tests`、JSON parse、`py_compile`、`git diff --check`；
5. 更新计划索引、Markdown/机器交接并生成一个 clean commit；
6. hard stop 等待人工审核。

## 6. 明确禁止

- 不得修改或“修正”旧 I2 failure、统一阈值或报告；
- 不得修改 BDDL、region、predicate 或使用非公开 hidden task 信息；
- 不得运行 SE(3)、grasp/action demo、Teleop、pending/accepted promotion或批量生成；
- 不得启动 LoRA/dense/30K、norm stats、FAST/ZIP、官方 GPU、上传或比赛提交；
- 不得触碰/终止 GPU0 外部任务，也不得因 GPU3 忙而改用别的 GPU。

J4 后只能由下一次审核决定是否授权一条 support-only SE(3) canary。
