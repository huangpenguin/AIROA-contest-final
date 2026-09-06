# AXIS Pi0.5 Track 3：TZ 证据与 Executor Contract 修复计划

> 制定日期：2026-09-06（Asia/Tokyo）
> 审核输入：root `2ae08e974d6be9ac59541236ce7ddb1d7f288dd6`
> worker source commits：首个失败 `1066774918941a574db3f02c7060f40aeb0c9c0a`；后六条 `343c5a554b0bfa05839b145f48a8c785d38edd19`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 当前状态：`track3_tz_evidence_and_executor_contract_remediation_planned`
> 唯一授权链：`ER0_to_ER5_then_hard_stop`
> 用户人工操作：`not_required`
> 不授权：GPU0–3、官方GPU、Teleop、generation/replay、accepted数据、训练/norm、FAST重导出、新candidate ZIP、上传或比赛提交

本计划只修复 Tonight TZ 执行暴露的身份、调度、门禁和证据合同。已交付的 R6 fallback 保持只读，不重新压缩、不改标签；旧七条 identity 均已消耗，不重跑。旧 TZ raw/summary/validation 是历史证据，不得改写。

## 0. ER0：不可变输入与派生证据闭包（只读）

- 锁定 `2ae08e9`、`1066774`、`343c5a5`、`0392710`、nested OpenPI、归档 TZ 计划、23个旧 namespace JSON、phase CSV、6个NPZ、fallback/source ZIP和sidecar。
- 在新的 ER namespace 写逐文件 bytes/SHA、Git tracking、strict duplicate-key JSON、CSV LF、ZIP CRC/entry/path/duplicate/FAST-only和NPZ dtype/loadability结果；旧文件只读。
- 派生记录必须同时保留：首条 `infrastructure_failure`、后六条诊断性 `family_failure`、旧 `tz2_cpu_matrix.infrastructure_failure=null` 的聚合矛盾、三对同族 byte-identical trace，以及6个NPZ的 object-array/no-pickle失败。

## 1. ER1：真实 supervisor 与身份合同（CPU/code only）

- 实现真实 single-instance 文件锁、PID/PGID、固定argv、root/tool/OpenPI/profile-lock SHA、clean-root和CUDA空可见性校验；不能再由常量布尔值自报通过。
- child运行期间周期性原子更新heartbeat；timeout必须TERM→等待→KILL并验证无后代。resume必须验证已有结果hash和完整身份。
- shared infrastructure failure在同一授权链内保持sticky global stop；不得通过“已有文件continue”清空历史failure。若需要修代码，先hard stop、clean commit，再由新审核授权继续。
- 冻结seed必须真正作用于所有相关RNG/env；若执行完全确定性，则从identity中移除伪seed并以确定性合同替代。

## 2. ER2：物理门禁、trace与finalizer合同（CPU/code only）

- `controller_clip`、action saturation、contact completeness/unrelated contact必须从runtime telemetry计算并使用三态；禁止硬编码安全门禁。
- trace只能保存 `allow_pickle=False` 可读的typed arrays；布尔/unknown使用显式值与mask，不保存object dtype。回归覆盖逐array读取与finite检查。
- finalizer顺序固定为producer close/fsync→inactive cursor→artifact manifest→outer hash→strict JSON/NPZ/CSV/ZIP检查→targeted/full tests→final validation。最终报告必须在所有被声明文件写完后再验证。
- summary显式写`delivered_zip_kind`、绝对路径、bytes/SHA、`recommended_for_track3_submission=false`；fallback仍标注`fallback_no_track3_improvement`。

## 3. ER3：离线诊断与三族全新 identity freeze（无physics）

- 从六条旧trace、raw stage results和公开same-demo资产解释support/containment的descend不收敛、articulation的approach不收敛；不得把未进入的close-dwell差异当成已测试。
- 只允许由离线数值证据支持的target/anchor/tolerance修复。冻结每族最多一个全新identity，共最多3个；必须与七个旧profile ID、参数hash和结果路径无重叠。
- 在任何physics前，真实回归必须覆盖identity drift、seed/determinism、sticky infrastructure stop、periodic heartbeat、no-pickle trace、computed gates、manifest/outer hash和inactive closeout。

## 4. ER4：三族一次性 CPU/no-renderer canary（最多3次）

- 固定顺序support→containment→articulation；每个新identity最多一次、fresh child、CPU/no-renderer，记录完整root/tool/profile/command/PID/PGID/CUDA和输入hash。
- 正常`family_failure`只关闭当前identity并继续其他family；任一shared infrastructure/identity/nonfinite/evidence污染立即全局hard stop。
- 无论成功或失败，轨迹都只是diagnostic且`training_eligible=false`；本计划不进入GPU、generation、replay、accepted或训练。

## 5. ER5：统一收尾与hard stop

- 写派生summary、LF ledger、完整manifest/outer hash、inactive cursor和final validation；复核root/OpenPI clean、LIBERO-plus原dirty未改、无live worker、GPU0–3无compute app。
- 运行strict JSON、NPZ no-pickle/finite、CSV LF、py_compile、targeted/full tests与`git diff --check`，形成clean commit。
- hard stop等待下一次独立审核；旧七条identity不得在后续计划中重跑，不上传或提交fallback。
