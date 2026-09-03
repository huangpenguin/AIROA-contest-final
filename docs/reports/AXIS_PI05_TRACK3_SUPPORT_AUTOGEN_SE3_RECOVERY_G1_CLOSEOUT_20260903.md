# Track 3 Support SE(3) 恢复：G0–G1 执行报告

日期：2026-09-03（Asia/Tokyo）

## 结论

本轮在 G1 的 S3 动态门禁处 hard stop，状态为：

```text
track3_reverse_semantic_adapter_v2_failed_support_remove
```

没有进入 G2 的 HDF5 alignment/抓取位姿提取、G3 的 SE(3) 执行器、G4 的 support canary，也没有进入 containment/articulation、人工 Teleop、批量生成或训练。

## 已完成

- G0 输入锁有效；旧 S0–S5 evidence、support 失败资产、denylist、P1 task10 snapshot 和 task10 原始 HDF5 未发生 hash 漂移。
- S1 v2 已将 task rows 与 episode rows 分层统计：task=`195`，episode=`21,961`；两者 identity sums 分别严格等于 `195` 与 `21,961`。
- `exact_reverse_demo` 仍为 task=`0`、episode=`0`，没有通过文本关键词升级任何候选。
- S3 typed language 通过，无 `_1`、`_region`、`contain`、`target_object` 等内部标识泄漏。
- S3 动态门禁运行了三族各 `3` 个 fresh process，共 `9` 次，并记录每次 BDDL/XML/controller/state hash。

## 动态门禁结果

| family | fresh runs | forward initial reverse goal | P1 reverse initial / goal | controlled destruction | 20-step settle |
|---|---:|---|---|---|---|
| support_remove task10 | 3/3 | false | true / false | true | robot state fail |
| containment_remove task20 | 3/3 | true | true / false | true | robot state fail |
| articulation_close task19 | 3/3 | false | true / false | true | robot state fail |

object 与 fixture 的 20-step 漂移均在门限内；robot policy-state 最大绝对漂移分别为 `0.1306809517`、`0.1388652157`、`0.5143735875`，均未通过当前 `0.05` 门限。task10 与 task19 的 forward HDF5 `states[0]` 送入 reverse 后 goal 也不为真，因此不能声称三族 reverse initial/goal 语义闭合。

本轮还修正并复核了受控破坏的 state layout：HDF5 `states[0]` 的第 `0` 项为 time scalar，qpos 从索引 `1` 开始。修正后 controlled destruction 三族均为 true；这排除了此前错误 qpos offset 造成的假失败，但没有改变总体 hard stop。

## 证据

- G0 lock：`data/reports/axis_libero_30k/track3_support_autogen_se3_recovery/g0_release_lock.json`
- S1 v2：`data/reports/axis_libero_30k/track3_support_autogen_se3_recovery/g1_s1_accounting_v2.json`
- S3 language：`data/reports/axis_libero_30k/track3_support_autogen_se3_recovery/g1_s3_language_v2.json`
- S3 dynamic：`data/reports/axis_libero_30k/track3_support_autogen_se3_recovery/g1_s3_dynamic_gates.json`
- v2 catalogs：`data/manifests/track3_public_primitives_v2/`

本轮没有创建 `pending_autogen_v2`，当前 exact/pending/accepted reverse demo 仍全部为 `0`。后续若要继续，必须先针对 dynamic gate failure 另行审核和制定 remediation；不得从本报告自动恢复 G2 或 support retry。
