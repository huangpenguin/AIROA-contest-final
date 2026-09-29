# AXIS Pi0.5 T3 D1 Evaluation Finalizer v2 Remediation Plan

> 创建时间：2026-09-16T11:55:54+09:00  
> 状态：`ready_for_D1A_code_test_fixture_only`  
> 当前阶段：`D1A_versioned_finalizer_v2_implementation`  
> D2 gate：`false`  
> GPU / simulator / training：`not_authorized`

## 1. 目标与边界

D0 已获保守批准，但确认 reverse 数据只有 `2/2/1/3`，可证明独立 lineage 为 `0`，reverse dev 不存在，单条 exposure 远超未来 `128x` 门。当前不得进入数据晋级、mix/norm 或训练。

D1 只修未来评测闭包的 provenance、真实视频验证和可恢复原子发布。不得修改或重新运行旧 R6、step100、step300 comparator/evidence/output，也不得把本计划解释为行为候选重新放行。

## 2. 冻结输入

- D0 evidence：`data/reports/axis_libero_30k/t3_post_salvage_failure_analysis_20260916_attempt_0001`；
- D0 analyzer SHA：`bbdc8fd60dba7ddb01afb03271586490ec05ba44629799166c82f7d7326ffb72`；
- D0 focused test SHA：`a84183e98a481431cc28f9a5d2c7f1f350364cce7ca80b5abe71722faa8cb3ee`；
- step100 comparator SHA：`b9187f2603075c64e0a95b543139c052430feed2bff665dc018b7d891bbb412d`；
- paired adapter SHA：`9c736c0ac1e3809df0ee35710781ffe8011929220794c58a75cf756e5b8a6c97`；
- 旧 D0、R6、step100、step300 namespace：只读，禁止 resume、覆盖或重写。

旧源码 SHA 已被历史 evidence 引用，因此 D1 必须新增 versioned 模块，不能原地修改旧 comparator 以伪装同一身份。

## 3. D1A：versioned finalizer v2

允许新增且仅允许新增：

- `tools/axis_pi05/track3_paired_eval_finalizer_v2.py`；
- `tests/test_track3_paired_eval_finalizer_v2.py`；
- 测试所需的最小真实可读 MP4 fixture，或在临时目录中由系统 `ffmpeg` 生成的确定性 fixture；
- attempt-scoped code/test evidence root，不得写入旧评测 namespace。

finalizer v2 必须满足：

1. identity 中分别记录 `comparator_source_sha256`、`worker_adapter_sha256`、`worker_plan_sha256`、`comparison_plan_sha256`，不得用模糊的单一 `plan_sha256` 代替；
2. 对每个 trace 检查 shape、finite、nonzero/允许零值语义与 case identity；
3. 对每个 MP4 实际调用 `ffprobe`，要求 return code、video stream、frame/duration 基本字段有效；任意非空字节不能冒充视频；
4. 在 attempt 专属 staging/partial 目录先生成 `paired_summary.json`、`release_decision.json` 与 complete-set manifest，manifest 对两个 payload 做 hash；
5. 只有 complete-set、hash 与 identity 全部通过才以目录级原子操作发布完成态；
6. 若首次在两个 payload 之间中断，只允许在同 attempt、同源码、同输入 identity、尚未 completed 的条件下恢复；
7. completed attempt 永远拒绝覆盖；identity 不同的 recovery 永远拒绝；
8. 任一行为门、coverage、视频、trace、identity 或 complete-set gate 为 false 时，release 必须为 false；
9. 不写入历史 namespace，不隐式启动 worker、policy server、simulator 或 GPU。

## 4. D1B：focused regression 与失败注入

focused tests 至少覆盖：

- comparator / worker adapter / 两类 plan SHA 各自缺失或不匹配时 fail-closed；
- 真实最小 MP4 通过、任意字节与截断 MP4 失败；
- 模拟第一个 payload 写完后的 crash，再以 exact identity 恢复并得到唯一 complete publication；
- changed source/input identity 的 recovery 被拒绝；
- completed attempt overwrite 被拒绝；
- manifest 漏项、payload hash 错误、额外 late artifact 被拒绝；
- 任一行为/retention gate 失败时 `release=false`；
- 历史 R6/step100/step300/D0 namespace fingerprint 前后不变。

执行 `py_compile`、focused pytest、duplicate-key reject JSON、`git diff --check`。不要求项目全量测试；若修改范围扩张则停止并重新审查。

## 5. D1C：闭包与硬停

D1 code/tests 通过后生成新的 machine summary，记录源码/测试/fixture SHA、测试结果、历史 namespace fingerprint、process/GPU zero、未运行项目清单。随后硬停在 `ready_for_D1_integration_review`。

D1 不把 finalizer 接到现有 behavior runner，不运行 R6/step100/step300，不 fold、不产 ZIP。是否将 v2 接入未来 runner 必须另行审核。

## 6. D2 及以后权限

- D1A–D1C code/tests/fixture：authorized；
- D2 新采集或 accepted promotion：false；
- D3 mix / fresh norm：false；
- D4–D6 GPU 训练：false；
- dense / 30k / 比赛 GPU：false；
- fold / FAST / ZIP / upload / submission：false；
- automatic submission：false。

下一执行者只完成 D1A–D1C 并硬停，不得将“D1 测试通过”解释为 D2 或 GPU 授权。

