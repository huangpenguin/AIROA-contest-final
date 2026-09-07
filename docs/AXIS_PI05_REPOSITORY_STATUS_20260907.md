# PARC2026_final 仓库与项目状态快照

> 快照日期：2026-09-07（Asia/Tokyo）
> 快照基线：root `760dc65adc74193112692ab003e19a74d588ea9f`；nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 说明：本文是便于 GitHub/网页 Chat 使用的状态汇总，不是新的独立审核，不改写任何历史报告、attempt、failure label 或原始证据。

## 1. 当前结论

唯一保留的审核结论是 `remediation_required`。BR0–BR11 的独立审核发现了生命周期和证据合同缺口；随后 CR0–CR14 已按修复计划执行并 hard stop，但其独立审核尚未完成，不能把 CR 执行闭包提升为 `approved`。

当前唯一活动计划是：

`docs/AXIS_PI05_TRACK3_CLEAN_LAUNCH_AND_DIVERSIFIED_CANARY_RECOVERY_INDEPENDENT_AUDIT_PLAN.md`

该计划只授权对 CR0–CR14 做独立只读审核。当前没有新 physics、GPU replay、Teleop、generation、accepted data、training、norm、FAST/export、新 candidate ZIP、upload 或 submission 的授权。

## 2. Git 与本地仓库状态

- 本状态快照从原审计分支派生到新分支：`codex/axis-pi05-repository-status-20260907`。
- GitHub 远端为 `origin = git@github.com:matsuolab/PARC2026_final.git`。
- 推送前，远端只有 `origin/main`；本状态分支此前没有 upstream。本次应只将状态快照分支推送并设置 upstream，不改写 `main`。
- root 工作树在快照前干净；nested OpenPI 为 clean；`LIBERO-plus` 的既有 dirty 状态属于项目原有内容，未清理、未覆盖、未提交。
- 当前没有 Track3、pytest、generation、replay、training 或 supervisor 活动进程；GPU0–3 当前无 compute，显存约各 `5 MiB`。GPU0 仍是禁止使用的外部进程保留卡。

### 2.1 容量分层

| 范围 | 当前大小/规模 | 是否通常随本次文档 push 上传 |
|---|---:|---|
| 整个工作目录 | 约 `1.5 TB` | 否 |
| `outputs/` | 约 `1.2 TB` | 否，已被 ignore |
| `data/` | 约 `99 GB` | 原始数据和多数运行产物被 ignore |
| `.git/` | 约 `50 GB` | 否；其中约 `49.34 GiB` 是临时 pack garbage |
| `docs/` 工作目录 | 约 `3.5 MB` | 计划/报告类已跟踪内容会上传 |
| HEAD 的 Git-tracked tree | `1460` 文件，约 `217.52 MiB` | 已在 Git 历史中 |
| `docs/` 的 Git-tracked tree | `122` 文件，约 `3.17 MiB` | 已在 Git 历史中 |

`outputs/`、原始数据、checkpoint、视频和大多数实验运行目录不会因为 `git push` 自动上传。需要注意的是，部分历史报告和较大的 JSON/NPZ 已经被 Git 跟踪，它们仍属于仓库历史；不能把“当前被 ignore”误解为“历史上从未进入 Git”。`.git` 中的临时 pack garbage 是本地磁盘问题，不是 GitHub 内容，本快照不删除它。

### 2.2 首次推送分支的规模

相对 `origin/main`，原审计分支有约 `285` 个提交、`1402` 个变更文件；当前分支相对 `origin/main` 的独有 blob payload 约 `240 MiB`（这是未压缩 Git object 大小，不是准确网络传输量）。因此：

- 第一次把当前项目线分支推到 GitHub 会比普通文档增量 push 重，可能传输数百 MiB 级别的 Git pack。
- 完成首次分支 push 后，只修改计划、handoff 或小型 summary 的后续提交通常很小。
- 不应把每个 heartbeat、临时 stdout、NPZ、视频、模型权重都提交；应提交小型 manifest、summary、hash 和阶段性 Markdown。

## 3. 上一轮 BR0–BR11 任务进度

BR 独立审核结论为 `remediation_required`，历史证据保持不变。

- BR0–BR8 的部分代码/fixture 修复有效，clean-source、attempt allocator、partial/sticky、manifest、stage-aware gate、source/contact/progress 等问题进入了修复链。
- BR9–BR10 的旧 attempt `0004` 实际运行了 support、containment、articulation 三条 CPU/no-renderer canary，各一次、无 retry；三条可保留为诊断性 `family_failure`，但它们在 `clean_root=false` 下启动，不能升级为合规执行。
- 六个 attempt 的 manifest 漏掉后生成文件；实际 timeout partial 为 `43` bytes，另一个 `39` bytes 的固定 synthetic 文件不能冒充 raw partial provenance。
- telemetry unknown 会错误落入 `family_failure`；已归一化 action 又被二次归一化，产生错误的 `normalized_action_abs_max=20.0`。
- BR 三条 NPZ 与上一轮 AC 对应轨迹逐字节相同，没有形成新的行为轨迹或 reverse 成功数据。
- BR 阶段没有成功 reverse demo，`success/accepted/training/new ZIP` 仍为 `0/0/false/false`。

证据：

- `docs/reviews/axis_pi05/TRACK3_BR0_BR11_INDEPENDENT_AUDIT_20260907.md`
- `data/reports/axis_libero_30k/track3_br0_br11_independent_audit_20260907/summary.json`

## 4. 最近一轮 CR0–CR14 执行进度

完整执行闭包见 `docs/reviews/axis_pi05/TRACK3_CR0_CR14_EXECUTION_CLOSEOUT_20260907.md`，机器摘要见 `data/reports/axis_libero_30k/track3_cr0_cr14_execution_20260907/summary.json`。

### CR0–CR8：代码/fixture 修复

- attempt `0001`：normal fixture startup timeout，physics 未启动，失败证据保留。
- attempt `0002`：stale latest pointer 被拒绝，physics 未启动，失败证据保留。
- attempt `0003`：通过 clean-source release；验证了启动前 root/OpenPI clean gate、dirty identity fail-close、真实 `43`-byte partial、`39`-byte synthetic provenance、telemetry unknown 分类、未进入 stage 的 `not_evaluated`、acquisition barrier 前 contact、direct normalized action limit、typed trace 和两套结构不同策略。
- CR0–CR8 没有消耗 physics identity。

### CR9–CR10：六条新 CPU canary

- 三族各一条 primary，再对每族正常 `family_failure` 各运行一条预冻结、结构不同的 alternative。
- 共 `6/6` 个新 CPU/no-renderer physics identity，`retry=0`。
- support、containment、articulation 的 primary/alternative 全部为正常诊断性 `family_failure`。
- 没有 timeout、partial、telemetry unknown、identity/process 或 GPU failure；六条 trace/action hash 与 AC、BR 历史及本轮 primary 均不同。
- 各族均已达到 alternative 上限，没有成功 reverse demo。

### CR11–CR14：后续状态

- CR11：因 success 为 `0`，GPU3 exact-action camera replay 为 `not_evaluated`，没有启动 GPU3 replay。
- CR12：只生成三族 `human_input_required` handoff；没有无人值守 Teleop。
- CR13：只生成 replay→accepted data→LoRA→ZIP 的 machine-ready dry-run 配置；没有 accepted data、训练或新 ZIP。
- CR14：原 attempt `0004` finalizer 的 late-artifact failure 保持为历史 failed evidence；随后在 derived correction namespace 中，不重跑 physics、不消耗 identity，重算完整 current set、manifest、strict JSON、process/GPU 和 closure，并通过。

CR 执行后的机器计数为：

```text
success=0
pending_replay_review=0
exact=0
accepted=0
training_eligible=0
new_candidate_zip=0
```

历史 attempt `0001/0002/0004` 和原 finalizer failure 不能被覆盖或改名；CR 执行计划已经归档，当前只剩独立审核计划。

## 5. 总体项目进度

### 已完成或可复用

- R6 `r6_dual_l2_fresh500_r1`、`L2_action_lora_heads`、500 optimizer steps 的模型和 folded 部署链已存在。
- public HDF5/任务/BDDL mapping、三类 reverse operator family、same-demo composed reverse start、no-pickle state、CPU restore/static gate、GPU3 camera child pipeline 和 atomic child result protocol 已建立。
- R6 本地 B80：Axis `48/80`，candidate `49/80`；差异很小，当前解释仍是 `review_required_not_significant`，不能当作 Track3 reverse 成功证明。
- 用户报告的官方结果：Track 1 `0.47`、Track 2 `0.33`、Track 3 `0`。Track 3 的零分目前说明正向 LIBERO LoRA 没有自然获得 reverse competence，不单独证明 action head 损坏。

### 尚未完成

- 没有 machine-accepted reverse demo，当前 accepted reverse data 为 `0`。
- 没有基于新 reverse data 的训练、fresh norm、FAST export 或新的 candidate ZIP。
- 没有因 CR11 而产生的 GPU3 exact-action camera replay；没有 generation、Teleop 或人工接管结果。
- 旧 R6 fallback ZIP 虽然机械属性可用，但已知 Track3 为 `0`，没有重新打包或再次提交。

## 6. 当前正确的下一步

1. 先完成 CR0–CR14 的独立只读审核；无论审核结果如何，当前 hard stop 不自动释放 physics、GPU、训练或提交权限。
2. 如果审核批准并形成新的 reviewed plan，才可按新计划处理后续 reverse supervision；旧 AC/BR/ER/TZ/CR identity 不得重跑。
3. 只有 CPU canary 真正 success，才可能进入 GPU3 exact-action replay；即使 replay `3/3`，也只到 `pending_human_review`，不会自动 accepted。
4. 如果仍然零 success，只能使用用户在线接管的 human Teleop handoff，不做无人值守伪造。
5. 只有 machine-accepted reverse data、复算后的 QUANTILES norm、训练和 FAST/ZIP 全部通过相应审核后，才讨论新 ZIP；当前不上传、不提交。

## 7. 建议的 GitHub / 网页 Chat 协作方式

本快照适合放在单独状态分支。建议以后采用“阶段性 checkpoint push”，而不是每条聊天或每个 heartbeat 都 push：

1. 计划确定：提交计划 Markdown、machine handoff 变更和必要的小型 JSON/CSV。
2. 阶段结束：提交 summary、manifest、hash、测试结果和状态更新。
3. 网页 Chat 规划时引用最新 branch、commit SHA、active plan、latest audit 和本状态文件。
4. 每次实验前后只推送可审计的闭包证据；原始大数据、权重、视频和临时日志留在本地，使用 manifest/hash 指向它们。
5. GitHub push 只同步 Git 对象，不会自动同步整个 `1.5 TB` 工作目录；当前新分支首次 push 完成后，后续小型文档更新是现实可行的。

本文件不改变当前 active plan 的授权，也不替代独立审核报告。
