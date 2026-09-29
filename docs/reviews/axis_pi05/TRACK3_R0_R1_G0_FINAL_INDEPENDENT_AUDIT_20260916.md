# Track3 R0/R1/G0 最终独立审核

> 日期：2026-09-16（Asia/Tokyo）  
> 范围：R0 finalizer v2、R1 冻结产物集成、G0 lineage 可行性与截止提交姿态  
> 唯一结论：`approved_final_hard_stop_with_nonblocking_r1_provenance_corrections`

## 结论

批准执行链在 `D2=false` 后硬停。R0 的 fail-closed 修复有效；R1 的发布决定仍应为
`false`；G0 足以证明一天窗口内不能进入 D2。没有可信依据继续 H300、提高 reverse
权重、启动 25/100-step GPU screen，或生成新的 Track3 ZIP。

本批准只覆盖“拒绝候选并停止”的结论，不批准把本次 R1 namespace 当作未来候选的
正式 release provenance。R1 有两项非阻断来源缺陷，需要在未来新计划中修复后重新生成
全新 attempt；不得改写当前已发布的拒绝证据。

## 独立复核结果

### R0 finalizer v2

- 独立复跑 focused tests：`16 passed in 34.41s`。
- `py_compile`、strict JSON、`git diff --check` 通过。
- 当前 finalizer SHA256：
  `3e538a8f9289945f7fc11f842ceeb15a2ced98269e5b0f26a46609ada56aa7d7`。
- 已机械覆盖：manifest 精确成员、strict worker JSON、真实 source-file SHA binding、
  cross-artifact/gate 一致性、request fingerprint 重算、全零 trace 不可由 caller 豁免，
  以及 payload/summary/release/manifest/rename 后的中断恢复。
- 已发布 R1 attempt 能由当前 validator 重放，结果仍为 `release=false`；除 behavior 外
  其余门均为 true。

### R1 冻结结果

独立核对后，总体行为结论保持：reverse `0/4 -> 0/4`，C20
`15/20 -> 14/20`，behavior=false、retention=true、release=false。R6 与 step100 的
冻结 namespace 在执行前后 fingerprint 不变，R1 completed namespace 无 partial。

但 R1 adapter 存在两项来源绑定缺陷：

1. `run()` 在读取 step100 pairs 后，实际传给 `_comparison()` 的仍是 attempt_0002
   R6/step300 的 `pairs`，而不是 `step100_pairs`。task13 与 task18 的逐 case candidate
   success 因而与 step100 不一致；两处 flip 相互抵消，所以聚合 reverse/C20 和
   `release=false` 恰好不变。
2. 已发布 identity 绑定了通用 `compare.py` 与旧 paired-eval runner，却没有绑定实际完成
   payload 转换和 comparison 构造的 R1 adapter 自身。finalizer 的 source binding 机制
   正常，但本次 caller 选择了语义不对应的 source paths。

因此这两项是对“拒绝结果”非阻断、对“未来发布证明”阻断的 provenance corrections。
如未来复用此流程，必须绑定实际 adapter/comparator bytes，并从 candidate worker payload
或正确的 step100 pairs 独立重算逐 case 与聚合 gate。

### G0 / D2

八条 accepted reverse 的 task 分布是 `2/2/1/3`，G0 可证明的独立 lineage 是
`0/0/0/0`，且没有 lineage-disjoint reverse dev。40/40 个声明 artifact hash 与磁盘
一致。

G0 工具自身只扫描 accepted episode root 与已声明 pending 路径，扫描范围偏窄；本次又对
整个 `data/reports/axis_libero_30k` 与仓库执行了八个 episode ID 的只读检索。检索到了
accepted metadata、review cards、manifests、training/source 引用与 seed 引用，但仍未找到
episode-bound recorder/session/process/event log。因此 `lineage=0` 的保守结论有更广范围
证据支持。

即便日后找到部分 lineage，D2 现在仍独立失败：alphabet-soup、red-mug、two-mugs 少于
每 task 3 条 accepted，且四个 task 都没有 lineage-disjoint reverse dev。故 D2=false
不依赖单一推断。

## 冻结与资源边界

- R6 evidence：62 files / 168093 bytes / fingerprint
  `11b059f7ae01497488975b8e7618f08438f75a8f147d9fc4d95915ea366b5370`。
- R6 output：98 files / 7036509 bytes / fingerprint
  `9aecab69e2cae329bf20953740b22c84b958cfe935b750a1da0b3ffdee6a8240`。
- step100 evidence：32 files / 95585 bytes / fingerprint
  `a0fac619397050ae094073b432be04b1b70cb53abdfa974da47038369f87ca59`。
- step100 output：49 files / 3496629 bytes / fingerprint
  `68c877e52770b76d86df9c2b05e54e3cd10d14f54d0f9ed7d3ea62696460206f`。
- GPU0–3 均为约 `5 MiB / 0%`，没有训练、评测或 policy-server 进程。
- 未执行 D2/D3/D4–D6、训练、fold、FAST、ZIP、上传或提交。

## 截止提交姿态与下一步

- Track1：保留已知 R6 正式分 `0.47`。
- Track2：保留已知 R6 正式分 `0.33`；拒绝 H10K 用户报告分 `0.13`。
- Track3：不发布 H300 step100/step300；已知 R6 为 `0.0`，仅在提交表强制要求
  Track3 artifact 且用户明确选择时，才把机械可用的 R6 ZIP 当作“已知零分 fallback”。
- 当前没有 Track3 执行性下一阶段。旧一天计划的条件 D3/D4/D5/fold 权限全部失效。
- 赛后或下一周期必须新建计划，从 recorder 原生写入不可伪造的
  session/process/event lineage 开始；每 task 至少 3 条 accepted、至少 2 个独立 lineage，
  并保留 lineage-disjoint reverse dev。禁止事后推断或回填 lineage。

