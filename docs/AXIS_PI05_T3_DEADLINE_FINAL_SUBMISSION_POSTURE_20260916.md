# AXIS Pi0.5 Track3 截止提交姿态（无活动执行）

> 日期：2026-09-16（Asia/Tokyo）  
> 状态：`final_deadline_submission_posture_no_active_track3_execution`  
> 性质：最终姿态与权限撤销；不是训练/评测执行计划

## 1. 决策

R0/R1/G0 已完成，D2 不可达。停止 Track3 当日研发链，不再从已归档的一天计划尾部
恢复任何阶段。当前没有可在截止前可信完成并通过 behavior gate 的“快速提分”路径。

## 2. 提交选择

| track | 当前选择 | 已知正式分/状态 |
|---|---|---:|
| T1 | R6 FAST | `0.47` |
| T2 | R6 FAST | `0.33` |
| T2 H10K | 拒绝 | 用户报告 `0.13` |
| T3 H300 step100/step300 | 不发布 | reverse 均 `0/4 -> 0/4` |
| T3 R6 | 默认不发布 | 已知 `0.0`；仅强制表单下由用户显式选择 fallback |

本文件不授权上传、重复提交或替用户操作比赛页面。

## 3. 权限矩阵

以下权限全部为 `false`：

- 数据 promotion/补写 lineage、mix、norm；
- GPU0–3 训练、25/100/300/500/1k/30k-step、dense full-FT、旧 H300 resume；
- 新 simulator/policy-server/behavior eval；
- fold、FAST export、ZIP；
- 自动上传、自动提交或手工提交代理。

现有 raw、accepted、R6、step100、step300、R0/R1/G0 证据均保持不可变。R1 的拒绝
attempt 不得被重新解释为正式 release provenance。

## 4. 赛后/下一周期入口

需要显式新计划和用户授权，且顺序固定：

1. recorder 在写 episode 时原生记录 session UUID、recorder PID/start time、event-log hash
   与不可变 source identity；不允许从时间戳、episode ID 或相似轨迹事后推断 lineage。
2. 每个 reverse task 采到至少 3 条 accepted，覆盖至少 2 个独立 lineage；另建按 lineage
   隔离的 reverse dev。
3. 在新 attempt 中修复 R1：绑定实际 comparator/adapter bytes，并从 candidate payload
   或正确 frozen pairs 重算逐 case 结果。
4. 只有数据门通过后，才允许重新设计 capped mix/exposure、fresh norm 和短步数 GPU screen。
5. 只有 reverse 至少 `+1` 且 C20 loss `<=2/20`，才讨论 100-step、fold 与 ZIP。

最新审核：
`docs/reviews/axis_pi05/TRACK3_R0_R1_G0_FINAL_INDEPENDENT_AUDIT_20260916.md`。

