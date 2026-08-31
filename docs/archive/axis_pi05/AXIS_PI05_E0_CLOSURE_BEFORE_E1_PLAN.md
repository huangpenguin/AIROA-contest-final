# AXIS Pi0.5：E0 Fail-Closed Closure 与 E1 释放门禁计划

> 归档状态（2026-08-31）：C0–C5 已由 commit `327316f` 完成并通过独立审核；E0 正式批准，后续由独立 E1 action-envelope 计划接管。

> 制定日期：2026-08-31（Asia/Tokyo）
> 审核输入 commit：`78109647e95d79690511c922845141ff492a395f`
> 输入状态：`s1_original_failure_preserved_reporting_corrected`
> 审核结论：E0 当前数值可信，但 formal closure 未完成；状态为 `e0_results_numerically_verified_formal_closure_required`
> 执行边界：CPU only；只执行 C0–C5；结束后再次 hard stop。
> 禁止：E1 policy run、GPU、closed-loop、FAST、ZIP、训练、push、上传、官方 Notebook 或比赛提交。

## 0. 给执行 section 的唯一任务

修复 E0 工具和证据链的 fail-open 问题，重新生成一个不覆盖旧 E0 的 closure artifact，完成测试和 clean commit 后停止。

```text
C0 输入与旧证据锁定
  → C1 coverage/dynamics fail-closed
  → C2 计数语义与完整性证明
  → C3 immutability/缺失数据测试
  → C4 closure 报告、handoff 与 Git 证据入库
  → C5 全量校验
  → STOP，等待人工审核
```

本计划不授权“顺便”执行 E1。即使 C0–C5 全部通过，也只能写 `e0_closure_complete_pending_review`。

## 1. 已审核通过、不得反复重做的事实

当前 E0 数据的独立复核结果：

```text
Q1 comparison groups                    1680
non-exact groups                         87
snapshot non-exact groups                23
trajectory non-exact groups              64
snapshot + trajectory occurrences       325
content duplicates                      155
content-unique pairs                    170
snapshot content-unique                  13
trajectory content-unique               157
trajectory affected-frame occurrences   302
dynamics mismatch                         0
old four-pair content-unique               3
tests                                    103 passed, 7 skipped
```

注意：`302` 是跨 case、replica-pair、mode、camera 的 affected-frame **occurrence** 数量，不是 302 个 content-unique frame。E1 的 content-unique 输入总数是 170，其中 trajectory 157、snapshot 13。

旧证据继续锁定：

```text
old S1.3 status             s1_real_egl_pairs_failed
old S1.3 threshold          1e-6
old S1.3 summary sha256     96f6c55f2b42ea55d3e103820e52912a4fcfad79794b686411e0e1659ca1a952
old pair manifest sha256    dacb81bcc6216c0b86be9ff350d731cdc199612b5e8f6f662560598cd588e972
raw-LoRA Q2 status          n1_q2_policy_sensitivity_failed_raw_lora_fresh_replay
```

## 2. Phase C0：输入与执行边界锁定

执行前保存：

- root HEAD、branch、root/nested worktree；
- `7810964` 必须是 HEAD 的 ancestor；其后的差异只允许为本次计划/交接归档提交；
- nested OpenPI commit 必须仍为 `dafa5a27266996eecf28920cacc32f3629776d91`；
- 旧 S1.3 summary、manifest 和 E0 reanalysis 的 SHA-256；
- Python/pytest 路径；
- `nvidia-smi` 只作为未使用 GPU 的边界快照，不初始化 CUDA context。

任一输入 hash 漂移或 worktree 非预期 dirty，状态写为 `e0_closure_input_drifted` 并停止。

## 3. Phase C1：coverage 与 dynamics 改为 fail-closed

修改 `tools/axis_pi05/r6_s1_egl.py`。

### C1.1 缺失数据不得视为 exact

`_exact_array_pair` 对任意一侧为 `None` 必须返回：

```text
present_left / present_right     显式记录
exact                            false
evaluable                        false
reason                           missing_left / missing_right / missing_both
```

只有两侧数组均存在、shape/dtype 一致且 `np.array_equal` 时才能 `exact=true`。

### C1.2 禁止静默 continue

snapshot/trajectory 枚举遇到下列情况必须写入 `coverage_issues`，不能静默丢弃：

- expected snapshot/phase 缺失；
- comparison group 缺失或 schema 错误；
- image descriptor/path/array 缺失；
- image shape/dtype 不匹配；
- trajectory 长度不是 manifest/descriptor 声明长度；
- policy state 或 simulator/qpos/qvel/ctrl 任一证据缺失；
- other camera 缺失；
- source comparison 声称 non-exact，但没有可定位的 changed frame。

有任一 issue 时：

```text
coverage_complete       false
renderer_only_evaluable false（对应 pair）
status                  e0_closure_coverage_incomplete
```

不得为了保留 170 的数字而忽略异常。

### C1.3 expected/observed set equality

从三个 Q1 comparison JSON 动态构造 expected non-exact semantic group set，不硬编码 23/64/87。分别保存：

```text
expected_snapshot_group_keys
observed_snapshot_group_keys
missing_snapshot_group_keys
extra_snapshot_group_keys
expected_trajectory_group_keys
observed_trajectory_group_keys
missing_trajectory_group_keys
extra_trajectory_group_keys
```

只有 expected/observed 完全相等且 issue count 为 0，才能计算 `coverage_complete=true`。

## 4. Phase C2：计数语义与 manifest 预备

closure summary 必须分别记录：

```text
snapshot_non_exact_group_count
trajectory_non_exact_group_count
snapshot_candidate_occurrence_count
trajectory_affected_frame_occurrence_count
snapshot_content_unique_pair_count
trajectory_content_unique_pair_count
content_unique_pair_count
content_duplicate_occurrence_count
renderer_only_evaluable_count
not_evaluable_count
dynamics_mismatch_count
```

当前源证据预期为 23、64、23、302、13、157、170、155、170、0、0；这些数值是验收期望，不得写死为生成逻辑。

为后续 E1 生成一个只读 `content_unique_pair_manifest.json`，但本阶段不得加载模型或产生 action。每项必须包含 case/phase-or-frame/camera/prompt/state/image/content/source-occurrence hashes；trajectory 项必须包含 `frame_index`。

## 5. Phase C3：测试与旧证据 immutability

至少新增以下测试：

1. `(None, None)`、单侧 missing 均为 not evaluable，不得 exact；
2. snapshot missing/trajectory missing/shape mismatch 必须形成 coverage issue；
3. source 标记 non-exact 但无 changed frame 时 coverage 失败；
4. expected/observed group set 缺一项或多一项时 `coverage_complete=false`；
5. `302 occurrence` 与 `157 trajectory unique` 字段不能混用；
6. 运行 reanalysis 前后，旧 S1.3 status、summary hash、manifest hash 完全不变；
7. agentview legacy duplicate 仍只能得到一个 content-unique pair；
8. corrected artifact 中不得出现误导字段 `axis_reference_jax_*`。

immutability 测试应复制或在临时目录引用 fixture，不得向旧 S1.3 namespace 写文件。

## 6. Phase C4：新 namespace、handoff 与 Git 证据

禁止覆盖初次 E0 文件。新产物写入：

```text
data/reports/axis_libero_30k/post_r6_lora/r6_egl_statistical_release/e0_closure/
  summary.json
  content_unique_pair_manifest.json
  E0_CLOSURE_REPORT_ZH.md
  THREAD_REPORT_ZH.md
```

更新 machine/project handoff：

- 状态为 `e0_closure_complete_pending_review` 或准确失败状态；
- 保存执行 commit 的 parent/input、旧/新 artifact hash、完整计数、测试结果；
- 原 `s1_3_audit_corrections_required` 改为历史 identified 列表，并新增 `resolved_by_e0`；
- 明确 E1 尚未启动；
- `next_phase_only` 只能是 `E1_pending_manual_review`，不能写 approved。

由于 `data/reports/axis_libero_30k/` 默认被忽略，四个小型 closure JSON/Markdown 和更新后的 handoff 必须显式纳入 Git。不要纳入 NPY/NPZ、模型、视频或大型 rollout。

## 7. Phase C5：验证与停止

使用正确环境：

```text
.venv_axis_pi05/bin/python -m py_compile tools/axis_pi05/*.py pipeline/*.py
.venv_axis_pi05/bin/pytest -q tests
jq . <closure summary and manifest>
git diff --check
```

额外核对：

- old S1.3 两个 hash 与 C0 完全一致；
- expected/observed group sets exact；
- coverage issue、not evaluable、dynamics mismatch 均为 0；
- 170 semantic keys 唯一；
- occurrence 合计满足 `325 = 170 + 155`；
- root/nested clean；
- 无 CUDA/GPU/model/server/HTTP execution artifact。

全部通过时只允许状态：

```text
e0_closure_complete_pending_review
```

形成 clean commit 后停止，等待人工审核。不得在同一 section 执行 E1。

## 8. Hard-stop 条件

- 旧 S1.3 status/hash 或 raw-LoRA Q2 failure 被改写；
- 缺失/shape mismatch 被当作 exact；
- coverage complete 仍为常量或无法由集合相等证明；
- 计数通过硬编码、筛选或删除异常维持；
- closure 覆盖初次 E0 或旧 S1.3 artifact；
- 170 个 pair 中有重复 semantic key；
- 测试失败、worktree 不 clean；
- 启动 E1、GPU、模型、closed-loop、FAST、ZIP 或任何外部动作。

停止时仍需写准确失败 summary，不得伪造 pass。

## 9. 下一次审核约定

审核者按 `docs/AXIS_PI05_PLAN_INDEX.md` 的固定流程：

- 若 closure 通过：归档本计划，落盘独立 E1 manifest-freeze/action-envelope 计划；
- 若 closure 失败：归档本计划，落盘最小 remediation 计划；
- 两种情况都更新 handoff/index 并形成 clean commit。
