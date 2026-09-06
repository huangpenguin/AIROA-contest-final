# AXIS Pi0.5 Track 3：EF 失败证据不可变性与生命周期修复计划

> 创建日期：2026-09-06（Asia/Tokyo）
> 唯一活动计划：是
> 审核输入：root `b1195dd59830f7c44d831bd0007e61229446d9b5`
> 审核结论：`remediation_required`
> 下一停止点：EG0–EG4 完成后 hard stop，等待新的独立审核

本计划只修复 EF4 首次失败原始报告被后续成功结果覆盖、同提交“自审批准”以及 machine handoff 顶层指针漂移所暴露的证据治理缺口。EF0–EF3 的已验证技术结果、现存 EF/ER/TZ bytes 和 R6 fallback 全部只读；无法恢复的首次失败 raw 必须诚实标为 `unavailable`，不得根据派生描述伪造或重建。

不授权任何新 physics identity 或旧 EF/ER/TZ identity 重跑。
不授权 GPU0–3、official GPU、Teleop、generation、replay、accepted data、training、norm、candidate ZIP、upload 或 submission。

## 0. EG0：冻结审核输入与 provenance gap

- 锁定 `b1195dd...`、nested OpenPI `b58aff...`、EF namespace 22 个 tracked files、ER anchor 29 个 files、现存 manifest/outer/closure 与本轮独立审核 summary。
- 对首次 EF4 失败证据记录不可逆事实：原 `final_validation.json` 已被 corrected attempt 覆盖；当前仅有 `ef4_first_attempt_failure.json` 派生说明，原始失败 validation、完整测试 stdout/stderr 与 partial inventory 为 `unavailable`。
- 允许只读检查 Git objects、reflog、临时目录和现存日志；找到原始字节时只能复制到新的 EG namespace 并记录来源/hash，找不到时不得把 derived 说明升级为 raw。
- 禁止改写 EF、ER、TZ 历史 namespace 中任何文件。

## 1. EG1：attempt 不可变 finalizer 合同（code/test only）

- finalizer 每次执行必须先分配单调、不可复用的 attempt ID，并将完整 validation、测试 stdout/stderr、process/GPU scan、partial inventory、cursor 与 heartbeat 写入 `attempts/<attempt_id>/`。
- attempt 终态文件一旦原子发布即不可覆盖；retry 只能创建新 attempt。`latest_attempt.json` 仅可作为带目标 hash 的指针，不能复制或替换历史内容。
- 失败 attempt 必须先闭合自身 manifest/outer hash/closed heartbeat，再允许 caller 决定是否进行新 attempt；任何失败都不能被 aggregate 清空。
- 使用临时目录和无 physics fixture 回归验证 fail→retry 后首个 attempt bytes/hash 不变、两个 attempt 都可独立 strict parse 与审计。

## 2. EG2：独立性与 clean-source 回归（code/fixture only）

- 审核报告不得与其声称独立审核的 execution/evidence commit 同提交生成；机器字段必须分别记录 execution commit、audit input commit 与后续 audit lifecycle commit。
- 从已提交的 clean source 运行无 physics fixture，冻结 root/tool/OpenPI/command/profile/三个 porcelain digest；dirty 或 source drift 必须 fail-close。
- 回归只允许临时文件或新的 EG evidence namespace，不得重跑旧 EF2 fixture 并冒充历史 attempt。

## 3. EG3：machine handoff 与 lifecycle 一致性

- 增加回归检查：顶层 `latest_audited_execution_commit`、`latest_audit_input_commit`、`latest_execution_root_commit` 必须指向实际被审核的 `b1195dd...`；最新 audit、唯一活动计划和 `next_phase_only` 必须在 index、Markdown handoff、machine JSON 与测试中一致。
- 保留旧 EF same-commit `approved` 报告和 summary 作为历史错误证据；新的修正只能写入独立 audit/derived namespace，并显式标记其已被本轮审核 supersede。
- 任何未知或未执行字段保持 `not_evaluated`/`unavailable`，不得写成 pass、false、0-risk 或推断成功。

## 4. EG4：验证、闭包与 hard stop

- 顺序固定为 producers close/fsync→inactive cursor→attempt-specific manifest/outer hash→strict JSON/NPZ/CSV→只读 fallback bytes/SHA/CRC/layout→真实 process/GPU scan→targeted/full tests→final validation→closed heartbeat→checkout-stable closure。
- 最终要求 duplicate-key rejection、targeted tests、全量 tests、`py_compile`、`git diff --check`、无 `.partial`、root/OpenPI clean；`LIBERO-plus` 既有 dirty 只记录且不得触碰。
- 无论通过或失败均 hard stop；通过也只表示证据治理缺口关闭，不释放 physics、GPU、数据生成、训练、ZIP、上传或比赛提交。后续必须由新的独立审核决定。
