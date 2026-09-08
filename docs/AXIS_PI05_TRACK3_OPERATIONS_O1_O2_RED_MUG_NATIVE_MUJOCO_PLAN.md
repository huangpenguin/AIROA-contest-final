# AXIS Pi0.5 Track 3：O1/O2 Red-Mug Canonical Gold 与 Native MuJoCo 计划

> 创建日期：2026-09-08（Asia/Tokyo）
> 唯一活动计划：是
> 前置审核：docs/reviews/axis_pi05/TRACK3_O0R_REMEDIATION_20260908.md
> 前置审核结论：approved
> O0R remediation commit：20662dd
> 证据 namespace：data/reports/axis_libero_30k/track3_operations_o0r_remediation_20260908/

本计划只释放 O1/O2，不追溯改写历史 raw，也不把两条 pending historical raw 直接升格。首要结果是第一条真正可复核的 reverse canonical gold；后续 tiny-overfit 和短续训都必须建立在该证据之上。

## 总边界

- 历史 episode、metadata、old candidate_gate 与旧 partial 路径只读。两条 pending 仍是 pending_historical_raw、replay=not_evaluated、gold_eligible=false。
- 388-step red-mug 只作为 derived keypose/re-timing 首选 seed；468-step 只作参考。不得把二者直接 replay 或 promotion。
- O1/O2 之前重新检查 root dirty/HEAD、nested OpenPI、active PID/cursor/heartbeat、GPU0–3 和 TurboVNC :3。GPU0–2 外部训练不得触碰；GPU3 才是条件式 replay/短实验设备。
- 不自动操作 Teleop。正式 gold 录制必须有用户在线、明确 takeover；no-save canary 不产生 candidate。
- Infinity reversal、Scenic position-only augmentation、LIBERO-PRO mixing、规模生成、自动提交、上传和比赛 submission 不在本计划授权内。
- 任何 identity/hash/telemetry/partial/supervisor/GPU 边界不确定都 global hard stop；不自动 retry，不消费历史 identity。

## O1：red-mug canonical gold

### O1.0 只读 seed 与重定时

1. 锁定 O0R path resolution、388 seed、相关 start/end state 与 source/goal predicate 证据。
2. 只生成新的 derived keypose/re-timing namespace；不得写回 pending episode 或其 metadata。
3. 预注册不超过一条 canonical gold attempt；失败时保留 raw、日志和 failure label，不把失败数据放入训练。
4. 目标约束：最终 canonical trajectory 不超过 300 steps，contact telemetry 为动态 typed table，每帧 saved_contact_n 等于真实 contact_n，acquisition barrier 位于 close/grasp_dwell 且在首次 source release 之前。

### O1.1 native MuJoCo / TurboVNC no-save canary

1. 使用服务器原生 MuJoCo 与 localhost-only TurboVNC :3 做 no-save canary，确认 front/wrist、start-state、reset、Open/Close、action/state/contact writer 和 candidate reset。
2. canary 不创建 candidate、不进入 pending、不启动 replay；只验证新 schema v4 与相对路径/atomic move 语义。
3. canary 前后不得有新增 partial 文件、active duplicate supervisor 或未知 descendant；GPU3 若非必要保持空闲。

### O1.2 用户 gold anchor 与 fresh replay

1. 只有用户明确在线接管后，才允许最多一条 red-mug canonical gold 录制；先做 no-save calibration，reset 后再正式录制。
2. 录制必须满足 source=true、goal=false 起点，gripper 显式 OPEN；接触只由 demonstrator↔subject 证明，goal-table contact 不算 acquisition。
3. machine gate 通过且无 collision 后，才可在 GPU3 启动三个 fresh replay process，逐 action、逐 start-state、逐 identity exact replay，要求 3/3。
4. 三次 fresh replay、300-step budget、无碰撞和所有硬门均通过，状态才可标记 pending_gold_review；只有完成规定人工审核后才可写 accepted。失败分支不得第四次 replay 或自动补录。

O1 success gate：

- first canonical gold：1 条；
- fresh replay：3/3；
- collision：0；
- frame_count：不超过 300；
- historical metadata mutation：false；
- source-group：至少 1；
- candidate ZIP：本计划不生成最终 submission ZIP。

## O2：held-out robustness 与 tiny-overfit

O2 只有在 O1 产生 accepted canonical gold 后才可执行；否则所有 O2 execution fields 必须为 not_evaluated，不得写成 false/pass。

1. 在 disjoint reverse dev 上先做 LIBERO-PRO 风格隔离评测，保持 held-out，不把全部 dev 混入训练。
2. 首条 accepted 后立即进行 tiny-overfit，用最小可审计数据链验证 loader、normalization、action/contact schema 和 checkpoint 冻结；不得等待几十条数据。
3. tiny-overfit 的 policy 必须在 disjoint reverse dev 至少增加 1 个成功，同时 C20 最多损失 2/20。未达门槛则停止，不启动 R6。
4. 只有累计至少 3 条 canonical accepted gold、且来自至少 2 个独立 source group，才允许一次 R6 100/300-step 短续训；不得盲目追加通用 LIBERO-Plus steps。
5. 从首个 checkpoint 开始每日一次 candidate freeze。只有实质 policy/数据变化才生成新 candidate manifest；同一 bytes 不重复生成候选。正式上传仍必须由用户操作，本计划不自动提交。

O2 输出：

- held-out reverse dev report；
- tiny-overfit checkpoint 与 exact input/output manifest；
- C20 regression report；
- source-group split 与 provenance；
- candidate freeze manifest；
- 若条件不足，显式记录 not_evaluated 与下一 hard-stop 原因。

## 停止与交接

- O1 canonical gold 未达 1/3 fresh replay：停止在 O1，保留 rejected/pending evidence，不进入 O2。
- O1 成功但 O2 held-out/tiny-overfit 失败：保留 O1 accepted 证据，停止短训，不自动进入 R6。
- 任一阶段发生 collision、contact raw incomplete、acquisition gate invalid、path safety violation、GPU/identity/partial 不确定：global hard stop，创建新 derived audit，不覆盖历史。
- O1/O2 完成后必须生成新的 execution closeout，运行 strict duplicate-key JSON、targeted/full tests、py_compile、git diff --check，并切换到下一份独立审核计划；不得把旧 UT 计划重新激活。
