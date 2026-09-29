# Track3 D0 Failure / Lineage / Exposure 独立审核

> 审核时间：2026-09-16T11:55:54+09:00  
> 结论：`approved_conservative_d0_hard_stop_with_nonblocking_audit_tool_gaps`  
> D2 数据门：`false`  
> GPU / simulator / policy server：未启动

## 1. 审核结论

D0 的只读执行、主要事实结论与硬停边界成立。R6、step100、step300 在同一四条 reverse case 上均为 `0/4`；每轮均是三条 `collision+timeout` 和一条 `timeout`。C20 为 `15/20 -> 14/20 -> 14/20`，变化集中在 task13 与 task18。D0 没有修改 11 个输入 namespace，没有启动 D1、训练、评测、policy server、simulator 或 GPU，也没有生成 fold、FAST、ZIP、上传或提交。

当前八条 reverse accepted episode 的 task 覆盖为 `2/2/1/3`。八条 payload SHA 与八个实际 `start_state.npy` SHA 均互不重复；但是 metadata 没有可审计的 recording/capture/session lineage 字段，因此可证明的独立 lineage 数仍是 `0`，且没有 lineage-disjoint reverse dev。D2 必须保持 `false`，不能用采样权重代替数据覆盖与 lineage。

## 2. 独立复算结果

- R6 / step100 / step300 reverse：全部 `0/4`；每轮三条 `collision+timeout`、一条 `timeout`。
- C20：R6 `15/20`、step100 `14/20`、step300 `14/20`；task13 为 `true/true/false`，task18 为 `true/false/true`。
- accepted 计数：alphabet-soup `2`、red-mug `2`、two-mugs `1`、yellow-book `3`。
- 八条记录的 `metadata.status=accepted`、人工审核 approved、replay `3/3_passed`、promotion accepted、`training_holdout=false`、official public example 均成立。
- 八个 raw trajectory 与 metadata 的实际 bytes SHA 均与记录值一致；八个 `start_state.npy` 的实际 bytes SHA 均与 metadata 一致且互不重复。
- 未发现 raw payload、start state 或 same-task resampled-action duplicate。两条不同 task 的 episode 共享 `human_review_sha256=0d868a1c07ecb66be04bba76e7d523f397d5cbf646e2723e0efb8d423afd8c1d`；这是审核来源耦合，不是 payload 重复，未来 lineage/审核记录必须显式解释。
- 11 个输入 namespace 的 file count、bytes 和 fingerprint 与 D0 inventory 全部一致；136 个输入 JSON duplicate-key reject parse 通过。
- 五个 D0 payload 与 output manifest 的 SHA 全部一致；六个输出 JSON strict parse 通过；没有 `.partial`。
- focused tests 为 `6 passed in 0.09s`；`py_compile`、`git diff --check` 通过。
- 审核结束时 GPU0–3 均为 `5 MiB / 0%`，无相关训练、评测或 policy-server 进程。

## 3. Exposure 口径判定

12,840 条 sampling rows 由 12,832 条 forward 与 8 条 reverse 构成；概率和为 `1.0`，source mass 为 forward `0.8` / reverse `0.2`。因此实际采样分布本身是归一化的，D0 不能据此宣称 sampler 已损坏。

风险在于数据稀少导致单条 reverse episode 的概率相对 forward median 约为 `236.7x–1183.3x`。现有 `effective_exposures` 对 reverse 行使用 `probability * 8`，与代码中 `len(reverse_rows)` 的实现一致，却不等于 12,840-row nominal epoch 下的 `probability * 12840`。所以该字段是口径/诊断语义不一致，不是 80/20 source-mass 归一化错误。D3 前必须版本化并统一 exposure 定义，且单条 reverse exposure 不得超过计划中的 `128x` 门。

## 4. 不阻断 D0、但 D1/D2 前要修的工具缺口

D0 结论是保守 fail-closed，以下缺口没有造成错误放行，因此本次不改写或重跑冻结 D0 evidence：

1. analyzer 将 `reverse_train_episodes.parquet` 的每行直接计为 accepted，没有在代码中 fail-close 核验 metadata、人工审核、replay 与 promotion 状态；当前八条由本次独立复核补足。
2. analyzer 生成 trajectory/metadata 的 computed SHA，但没有机械要求其等于 stored SHA；当前八条由本次独立复核确认相等。
3. `computed_start_state_sha256` 当前复制 metadata 值，没有读取并哈希 `start_state.npy`；本次已独立读取复算并确认八条相等且唯一。
4. `direct_identity_leakage_detected` 与 exposure inconsistency 结论部分为常量，而不是从 overlap/formula 机械派生；当前 overlap 为零、reverse dev 不存在，D2 仍为 false。
5. step100 closeout 已发现的 evaluator provenance、真实 MP4 检查和双 artifact 中断恢复缺口仍未修复。

## 5. 权限与唯一下一步

唯一下一步切换到 `docs/AXIS_PI05_T3_D1_EVALUATION_FINALIZER_REMEDIATION_PLAN_20260916.md` 的 D1A：新增 versioned finalizer v2，只做 code/tests/valid fixture。旧 comparator、旧 attempt 与 D0 evidence 全部只读。

D1 不启动行为评测或 GPU；D2 数据采集/accepted promotion、D3 mix/norm、D4–D6 训练、fold/FAST/ZIP、上传/提交全部未授权。D1 完成后必须硬停复核，不能自动进入 D2。

