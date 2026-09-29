# Track3 D1 Finalizer v2 Integration Audit

> 审核时间：2026-09-16T13:35:58+09:00  
> 结论：`remediation_required_before_D1_integration`  
> D1 execution boundary：`approved_hard_stop_compliant`  
> D2 / GPU / training：仍未授权

## 1. 结论

D1 按计划完成 code/tests/fixture 并硬停，没有启动 GPU、simulator、policy server、行为评测或训练。独立复跑 `8 passed in 10.18s`；源码、测试和真实 MP4 SHA 与 summary 一致；`py_compile`、strict JSON、`ffprobe`、`git diff --check` 通过。五个冻结 namespace 的 file count、bytes、fingerprint 也全部与 D1 summary 相同。

真实 `ffprobe`、trace shape/finite 检查、attempt-scoped staging、目录级 rename、首次 payload 中断恢复和 completed overwrite 拒绝均成立。因此 D1 不是失败执行，历史 summary 保持只读。

但是当前实现不能直接接入正式 comparator/release。独立 `/tmp` 对抗测试证明五个 fail-open：

1. manifest 的 `payloads` 重复 baseline 条目仍被 `validate_published_attempt()` 接受；
2. `release_decision.paired_summary_sha256` 可被改成全零，只要同步更新 manifest 对 release 文件的 hash，validator 仍接受；
3. paired summary 可记录 behavior gate=false/release=false，而 release decision 仍为全 true/release=true；同步更新 paired file hash 后 validator 返回 release=true；
4. baseline/candidate payload 是任意 opaque bytes，声明的五个 identity SHA 只校验格式，不与真实源码、计划或 worker result bytes 绑定，仍可 release=true；
5. 每条 trace 的 `allow_zero` 由调用方直接提供，调用方可让全零 baseline/candidate traces 通过并 release=true。

因此 D1 summary 中 `manifest_exact_set_and_hashes` 与完整 `separate_identity_fields` 应解释为“生成路径存在这些字段”，不能解释为“published attempt 已能抵抗交叉篡改或声明身份伪造”。

## 2. 代码发现

### P1：published complete-set 未验证唯一性和跨 artifact 一致性

`validate_published_attempt()` 只比较 payload name 的 set，没有要求 list 长度等于 2，因此重复项通过。它也没有校验：

- `release_decision.paired_summary_sha256 == actual paired_summary SHA`；
- paired/release 的 gates 与 release 值一致；
- manifest artifact checks 与 paired summary artifacts 一致；
- completed lock 的 request fingerprint 可由已发布内容重算。

这会使 validator 把内部矛盾的 completed attempt 判为有效 release。

### P1：identity、worker payload 与 zero-trace waiver 都是 caller assertion

`validate_identity()` 只校验 64 位小写 hex。`finalize()` 将 opaque bytes 写成 `.bin`，不 strict-parse worker role/cases/status/identity，也把 behavior/retention/coverage gate 和逐行 `allow_zero` 直接交给调用者。当前 `identity_gate=True` 与 `complete_set_gate=True` 是构造时常量，不是从真实源码/输入 bytes 与已发布闭包派生。

这套 API 可作为低层原子写入 primitive，但不能直接成为比赛候选 release authority。

### P2：失败注入覆盖不足

focused tests 只覆盖第一个 binary payload 后中断，没有覆盖 paired summary 后、release decision 后、manifest 后、rename 后返回前等原始问题对应的恢复点；也没有 duplicate entry、cross-link mismatch、paired/release disagreement、fake source SHA 或 caller-waived zero trace 的 regression。

## 3. 保留成立的证据

- finalizer SHA：`3d126e5ef47fe3ecbfda17fe5bc6f3a2921b7ed0115460019198fe74b758ab6d`；
- tests SHA：`3cae8affb50b97983df27eeaf1e4650d750b8e6766935452d087493d60f787cf`；
- fixture SHA：`133b3ccc3377dfcb2f7d6b17bb3a7a4e3a3cb38e80aec6b094b2e1a734da04cc`；
- independent tests：`8 passed in 10.18s`；
- frozen namespace fingerprints：R6 evidence/output、step100 evidence/output、D0 `5/5` exact；
- GPU0–3：`5 MiB / 0%`；相关 live process：0。

## 4. 一天截止下的决策

先用不超过 90 分钟修复上述 release-authority 缺口，然后用冻结的 R6/step100 artifacts 做 finalizer-only integration，必须得到 reject。与此同时只读核查现有 8 条 accepted 是否能由原始 recorder/session evidence 证明 lineage；单靠 episode timestamp、唯一 payload 或唯一 start state 不算 lineage。

若 T+3h 仍不能证明已有样本的独立 lineage，则一天内重新取得每 task 三条、两 lineage 共 12 条合格数据不可控，不应再用旧 8 条开启新的四卡训练。若能够证明已有 lineage，则只补 `1/1/2/0` 四条 accepted，构建 95/5 capped mix，再启动四卡 25-step screen。所有分支、时限和止损点写入新的唯一活动计划。

