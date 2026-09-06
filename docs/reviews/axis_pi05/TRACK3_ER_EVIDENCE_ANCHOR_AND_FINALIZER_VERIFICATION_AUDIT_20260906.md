# Track3 EF0–EF4 独立审核（2026-09-06）

审核结论：`approved`

本审核独立复核唯一活动计划 EF0–EF4 的执行证据、代码合同、原始字节、失败保留、最终门禁和 hard stop。审核输入为 EF0 anchor commit `38635fbf08e4e8df8c4877ddcb6b2b2ec8b9b118`，nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`。历史 `TRACK3_TZ_ER0_ER5_AUDIT_20260906.md` 保持不改写。

## 1. EF0：原字节锚定

`data/reports/axis_libero_30k/track3_tz_evidence_and_executor_contract_remediation/` 的现存 ER 证据已在 `38635fb` 中 force-add；独立 `git ls-files` 计数为 `29`，逐文件工作树 SHA 与 `git show HEAD:path` 均 exact。旧 ER0 manifest 中 `24` 个记录已 Git-persisted、`6` 个本地 NPZ 的不同状态仍被 EF0 报告明确记录。历史 ER/TZ namespace 未被重写。

## 2. EF1–EF3：代码与 fixture 证据

- EF1 的 parent 在 child 启动前冻结并传递 root/tool/OpenPI/profile/command、root clean 状态、三个 porcelain/digest 和 CPU/no-renderer 环境；child 不回填 expected。完整 identity、合法 resume 均通过；root dirty、OpenPI、`LIBERO-plus`、command 与 trace bytes/SHA 漂移均按 fail-closed 失败。
- EF2 仅运行 fixture child。第二入口被 file lock 拒绝；正常 `family_failure` return code 为 `0`；周期 heartbeat 观测序列为 `1,2,3,4`；timeout 顺序为 `SIGTERM → grace → SIGKILL`，结束后无 descendants；sticky stop 在启动 child 前生效。
- EF3 为离线 gate。telemetry/schema completeness 与 required physical contact completeness 是独立三态结果；support/containment 使用 demonstrator↔object，articulation 使用 demonstrator↔handle/region；零相关接触为 `failed`，未进入阶段为 `not_evaluated`，typed NPZ 为 finite、无 object dtype 且 `allow_pickle=False` 可读。

## 3. EF4：finalizer 闭包

成功的 corrected finalizer 记录在 `data/reports/axis_libero_30k/track3_er_evidence_anchor_and_finalizer_verification/`：

- phase ledger 为 `EF0=passed, EF1=passed, EF2=passed, EF3=passed, EF4=passed_hard_stop`；`execution_summary.status=hard_stop`；cursor inactive；heartbeat `state=closed`。
- `final_validation.status=passed`，包含 strict JSON、NPZ、LF CSV、既有 fallback ZIP、process scan、GPU boundary、targeted/full tests、manifest/outer hash 和无 partial residue。manifest 为 `13` entries，manifest SHA 为 `5649d2c09add79d9ab7213bada6e0454672358e5c25ae0f127bbb0aac11d795d`，outer hash SHA 为 `05f4511946a8a3513ddffac15832730ae47b7e2f5a42596197eac6c2fbf61d84`。
- fallback 仍是既有 `r6_fallback`，`9383571677` bytes、SHA `95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a`、CRC passed、3401 entries、FAST-only；没有创建 candidate ZIP。
- finalizer 自身 targeted 为 `7 passed`，项目 full 为 `320 passed, 7 skipped, 23 warnings`，return code 均为 `0`；process scan 无 live worker；GPU0–3 final scan 的 compute process 均为空；py_compile、`git diff --check`、strict JSON 和 manifest exact 均通过。

第一次 EF4 尝试的失败原因（生命周期测试断言仍指向活动计划、timeout partial 清理缺口）以 `ef4_first_attempt_failure.json` 作为 derived failure 保留；未覆盖或篡改历史证据，corrected attempt 才作为通过结果。

## 4. 资源边界与生命周期

独立复核确认：新 physics identities `0`；三条 ER4 与旧七条 TZ identity 未重跑；未执行 physics、GPU compute、generation、replay、accepted data、training、new candidate ZIP、upload 或 submission。`LIBERO-plus` 的既有 dirty 内容未触碰；nested OpenPI 保持 clean。EF 计划已归档，当前无 active plan，下一权限唯一为 `new_reviewed_plan_required`。

机器摘要见 `data/reports/axis_libero_30k/track3_er_evidence_anchor_and_finalizer_verification/audit_summary.json`。

## 5. 审核后最终树复验

生命周期断言与审核摘要加入后，独立 targeted 回归为 `12 passed`，项目 full 回归仍为 `320 passed, 7 skipped, 23 warnings`，return code 均为 `0`。新增审核摘要及全部 EF JSON 通过 duplicate-key rejection；`py_compile` 与 `git diff --check` 通过。最终提交前再次检查 root/OpenPI、进程和 GPU 边界；`LIBERO-plus` 既有 dirty 仍保持未触碰。
