# AXIS Pi0.5 Track 3：BR0–BR11 独立审核计划

> 创建日期：2026-09-07（Asia/Tokyo）
> 唯一活动计划：是
> 审核对象：`data/reports/axis_libero_30k/track3_acquisition_gate_attempt_lifecycle_remediation/`
> 前置执行：BR0–BR8 clean-source 修复与 BR9–BR11 旧执行证据已 hard stop

本计划只授权对 BR remediation 执行证据做独立只读审核，不授权新的 physics、旧或 BR identity 重跑、GPU、generation、replay、accepted data、training、norm、FAST/export、candidate ZIP、upload 或 submission。

独立审核必须：

1. 重算 BR0 派生 AC closure、attempt `0001` 的 partial failure 历史、attempt `0002` 的修复、clean-source attempt `0006` 和 BR11 attempt `0004` 的 manifest/outer/closure/pointer。
2. 检查 `br9_dirty_source_correction.json`：BR9–BR10 原执行发生在 `clean_root=false` 的 root 下，三条 trace 可以保留为诊断性 `family_failure`，但不能批准为合规 BR9/BR10；不得重跑已消耗 identity。
3. 严格解析当前 BR namespace 与 machine handoff 的全部 JSON，检查 typed NPZ `allow_pickle=False`、CSV LF、无 partial、process/GPU 边界，以及 nested OpenPI clean 与 `LIBERO-plus` 原有 dirty 集合未变。
4. 给出唯一审核分类：`approved`、`remediation_required` 或 `blocked`。如发现 dirty-source 生命周期缺口，保留历史 execution 与 failure label，在新 audit namespace 给出修正，不覆盖 attempt `0004`。

审核完成后，按 AGENTS.md 生命周期规则归档本计划并创建下一份唯一活动计划；在独立审核前不得继续 Track 3 物理、数据或提交流程。
