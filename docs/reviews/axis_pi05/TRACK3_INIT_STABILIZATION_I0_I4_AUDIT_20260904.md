# Track 3 Init Stabilization I0–I4 独立审核

> 审核日期：2026-09-04（Asia/Tokyo）
> 审核输入：root `0dec0b1930a3d2ad9f196febe1aeaabb0c89445d`，parent `1a0b387`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`
> 新状态：`track3_operator_scoped_stability_release_planned`

## 1. 获批事项

- I0 的 28 项 immutable input、旧 R2/R4 与 reverse-demo `0/0/0` 计数已锁定；I4 重验通过。
- I1 正确固定三族 operator-critical source/goal、7D zero-action、20 Hz、step5 release checkpoint，以及 `passed/failed/not_evaluated` 三态。
- I2 真实完成 12 个 fresh CPU/no-renderer process；schema、finite、predicate detail 与 OS return-code 契约均通过。
- 三族 reverse critical source 在 step5 为 true，并逐步连续保持到 step20；reverse goal 在同一窗口逐步连续为 false。
- I2 失败后未执行 I3；GPU0/GPU3、BDDL、SE(3)、抓取/demo、Teleop、训练、FAST/ZIP、官方 GPU、上传与提交边界均保持。
- 执行记录全量测试为 `189 passed, 7 skipped, 23 warnings`；root/OpenPI clean，`LIBERO-plus` 既有 dirty 状态未触碰。

因此，本轮执行与 fail-closed 停止本身获批准。

## 2. 当前失败不能解释为 reverse init 语义失败

现有 gate 在 `tools/axis_pi05/track3_reverse_init_stabilization.py` 中，对每个 run 的所有命名 object/site 取最大 translation/rotation，再统一与 `2 mm / 0.02 rad` 比较；随后要求 12 个 run 全部通过。这个规则没有区分 operator-critical participant、receiver 与无关背景物。

独立按对象复算 step5→20 后得到：

| family | 全场 max 的来源 | operator-critical 事实 | 解释 |
|---|---|---|---|
| support_remove | bowl rotation `0.0564043 rad` | bowl translation `0.0009971 m`；`On(bowl, plate)` 16/16；bowl↔plate contact 从 step5 的 2 个增至 step20 的 20 个 | bowl 在 plate 上接触落稳；自由旋转不破坏 support 关系 |
| containment_remove | 背景 `salad_dressing_1` translation `0.0215049 m`，其次背景 ketchup `0.0189571 m` | orange juice translation `5.54e-7 m`；basket region translation `3.13e-7 m`；`In` 16/16 | 失败完全由非关键背景物贡献 |
| articulation_close | 背景 bowl translation/rotation，最坏 `0.0037985 m / 0.0810595 rad` | drawer site translation最坏约 `1.12e-7 m`，fixture joint drift最坏约 `1.09e-7`；`Open` 16/16 | 目标抽屉稳定，失败来自无关背景物 |

9 个 reverse runs 的完整 init predicate truth vector 与 goal predicate truth vector从 step5 到 step20 都没有一次布尔跃迁。task10 的三次结果完全复现；task20 的背景物沉降也三次完全复现。当前数据更支持“可重复的物理落稳 + 任务拓扑保持”，不支持“场景随机失稳”。

## 3. 审核发现的报告问题

机器 handoff 顶层 `tests` 仍保留上一轮审核的 `182 passed`，而本轮 execution 子项记录为 `189 passed`。这不影响 I2 结论，但顶层 current-state 字段必须由本次审核刷新，避免接手者误读。

旧 I2 failure、全场 max、阈值和 JSON 必须 immutable 保留；不能把它们改写为 pass。修复方式是新增 operator-scoped release contract，而不是事后提高统一阈值。

## 4. 审核裁决

结论为 `remediation_required`：

1. 批准 step5 reverse critical source/goal 语义以及 I0–I4 hard stop。
2. 不批准把 `track3_reverse_init_stabilization_failed_named_scene_stability`解释为 reverse BDDL 或 reverse start 失败。
3. 不批准直接跳到 SE(3) 或训练；先冻结按 operator role 划分的稳定性 contract。
4. 新 contract 必须同时要求 critical relation 连续、完整 predicate truth vector 不跃迁、robot/关键 participant 使用原有单位阈值，并把背景物漂移保留为诊断；不得仅删除背景物记录。
5. CPU 复核须包含9条带完整contact telemetry的新reverse与6条forward controls；通过后才允许GPU3上一次task10 canary与同版本三族`3/3` production-like动态验证。

## 5. 不授权事项

本审核不授权修改 BDDL/region/predicate、SE(3)、抓取或 reverse demo、人工 Teleop、批量生成、训练/norm、FAST/ZIP、官方 GPU、上传或比赛提交。

## 6. 审核验证

- `0dec0b1` commit diff 与 `git show --check`：通过。
- I2 汇总 SHA-256：`e72ae24d27b6edbba37b043a88042191f8a1c7d00c298b812653a1a2ecc17dab`。
- 9 个 reverse run 的 step5→20 init/goal truth-vector transition：均为 `0`。
- GPU 审核时：GPU0 为外部 YOLO PID `1133706` 占用，未触碰；GPU1–3 空闲。
- 定向测试：`18 passed, 2 warnings`。
- 全量测试：`189 passed, 7 skipped, 23 warnings`。
- machine JSON parse、`py_compile` 与 `git diff --check`：通过。
