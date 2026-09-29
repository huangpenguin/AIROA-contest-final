# AXIS Pi0.5 Track3 One-Day Finalizer R2, Data Gate and GPU Screen Plan

> 创建时间：2026-09-16T13:35:58+09:00  
> 状态：`ready_for_R0_finalizer_remediation_and_G0_lineage_feasibility`  
> 总时间盒：从执行 T0 起最多 24 小时  
> 当前 GPU：0–3 空闲，但训练仍受 D2 条件门约束  
> 自动上传/提交：`false`

## 1. 一句话策略

今天只做能改变最终决策的工作：最多 90 分钟修复 finalizer release-authority，T+3h 前判断现有数据是否可能满足 lineage 门；只有已有 8 条可由原始证据证明 lineage，且 T+7h 前补齐四条 accepted，才进入 95/5 mix 和四卡 25-step screen。任何门未按时通过立即止损，不再尝试 30k、dense 或旧 H300 续训。

## 2. 冻结事实

- R6 / step100 / step300 reverse 均为 `0/4`；H300 无 release 候选；
- C20 为 `15/20 -> 14/20 -> 14/20`；
- 当前 accepted reverse 为 `2/2/1/3`，显式可证明 lineage 为 `0`，reverse dev 不存在；
- 当前 80/20 sampling 概率归一化，但单条 reverse 相对 forward median 为约 `236.7x–1183.3x`；
- T2 H10K 官方分 `0.13`，低于 R6 T2 `0.33`，不能作为 fallback；
- 当前 finalizer v2 SHA `3d126e5e...ab6d` 保留为历史 D1 input，禁止把后续 remediation 冒充同一源码。

## 3. 并行工作流 R：finalizer R2（T+0h 至 T+2h）

### R0：最小修复，目标 90 分钟

允许修改现有 v2 模块和 focused tests，但必须在新的 remediation evidence root 记录新 SHA。至少修复：

1. manifest payload list 必须 `len == 2` 且 baseline/candidate 各恰好一次；
2. validator 必须交叉校验 release→paired SHA、paired/release gates、release 值、identity、cases、artifact checks；
3. completed request fingerprint 必须能从已发布内容重算，不能只信 lock；
4. comparator、worker adapter、worker plan、comparison plan SHA 必须与显式 source path 的实际 bytes SHA 相等；
5. baseline/candidate payload 改为 strict JSON worker result，验证 role、case/fingerprint exact-set、worker identity、status、无 infrastructure error；opaque bytes 不得 release；
6. coverage 从两个 worker payload 派生；behavior/retention 来自带 comparator SHA 的比较结果，不接受裸 caller boolean；
7. T3 正式 behavior trace 禁止逐行 caller `allow_zero`。任何 zero policy 必须在 comparison plan 中冻结并进入 identity；本轮 reverse/C20 一律 nonzero；
8. 增加 summary 后、release 后、manifest 后的 crash injection/recovery，以及五个本次 adversarial regression。

R0 focused tests 目标不少于 13 个；`py_compile`、strict JSON、diff-check 必须通过。超过 90 分钟仍未闭合则停止 integration，不继续扩写框架。

### R1：冻结 artifact 的 finalizer-only integration（T+1.5h 至 T+2h）

只读取旧 R6/step100 结果，在全新 namespace 运行 adapter+finalizer，不启动 simulator/policy server/GPU。预期结果必须是 reverse `0/4 -> 0/4`、release=false。若得到 true 或 identity/coverage 无法机械派生，hard stop。

## 4. 并行工作流 G：D2 可达性（T+0h 至 T+7h）

### G0：lineage provenance feasibility，T+3h 硬截止

只读查找原始 recorder/supervisor/session evidence。独立 lineage 必须由 capture/session/process/event 证据证明；不同 episode ID、timestamp、payload SHA、start state 或 human-review 文件本身不足以证明。

输出每 task 的 accepted episode→lineage 表。判定：

- 若四个 task 均能从现有数据证明至少两个独立 lineage：进入 G1；
- 若任一 task 不能证明：`D2_unreachable_within_one_day`，取消 G1、D3、D4–D6，不训练；
- 禁止为赶时间回填未经原始证据支持的 lineage 字段。

### G1：只补最小缺口，T+7h 硬截止

仅在 G0 通过且用户可进行采集/审核时，补齐：

- alphabet-soup：1 条；
- red-mug：1 条；
- two-mugs：2 条；
- yellow-book：0 条。

新 episode 必须在写入时记录显式 lineage、20 Hz、typed/no-pickle、finite、payload/start/action SHA；每条 fresh replay `3/3` 并人工 accept 后才计数。GPU0–3 可用于不同 episode 的 fresh replay，但采集本身不得因 GPU 空闲而并发复制同一 lineage。

T+7h 未达到每 task `>=3 accepted`、`>=2 lineages` 且 lineage-disjoint dev，即停止训练。

## 5. D3：唯一 95/5 mix（T+7h 至 T+9h）

只在 D2 机械通过后：

- 建立一个 95% forward / 5% reverse mix，不再同时做 90/10；
- reverse 按 task 与 lineage 平衡；每个 task 留至少一条 lineage-disjoint dev；
- 单条 episode 相对 forward median exposure 必须 `<=128x`；
- fresh QUANTILES norm、manifest、sampling table 与 source hashes 一次冻结；
- 若 5% 仍超过 `128x`，降低 reverse mass，不复制 episode。

## 6. D4：四卡 25-step screen（T+9h 至 T+14h）

只有 R1 与 D3 都通过才启动。GPU0–3 各一个独立 fresh arm：

1. GPU0：R6 params + L2，peak LR `3e-7`；
2. GPU1：R6 params + L1 action-expert-only，peak LR `3e-7`；
3. GPU2：released Axis + zero-delta L2，peak LR `1e-6`；
4. GPU3：R6 params + small dual LoRA，peak LR `1e-7`。

共享数据/norm/input 缺陷全停；单 arm 故障只停该 arm。禁止从旧 H300 step100/300 resume。

## 7. D5：并行行为门与至多一条 100-step（T+14h 至 T+21h）

使用同一 R6 baseline、lineage-disjoint reverse dev 和 C20。四卡并行调度候选 worker；finalizer R2 统一闭包。

25-step 入选门：reverse 至少 `+1`，C20 delta `>=-2/20`，零 infrastructure error。无 arm 通过则停止，不把“最不差”当候选。

最多选择一条 arm，从其冻结 init fresh 启动 100-step，不续接 25-step checkpoint。100-step 重新通过同一行为门后才是 release candidate。

## 8. D6：发布缓冲（T+21h 至 T+24h）

只有 100-step 行为门通过才允许 fold、FAST、static validate、CRC 和本地 ZIP；预留至少 3 小时。自动上传/提交始终 false，由用户人工提交。

若没有新候选：

- T1 保留已知 R6 `0.47`；
- T2 保留已知 R6 `0.33`，不使用 H10K `0.13`；
- T3 不发布已失败的 H300 step100/step300。现有 R6 T3 fallback 机械可用但已知官方为 `0`，仅在规则要求必须有提交时由用户决定。

## 9. 权限与硬停止点

- R0/R1 code、tests、冻结 artifact finalizer-only integration：authorized；
- G0 只读 lineage audit：authorized；
- G1 新采集/replay/人工 accept：仅在 G0 通过且用户在线明确接管后执行；
- D3：仅在 D2 机械通过后条件授权；
- D4/D5 GPU：仅在 R1、D2、D3 全通过且仍早于对应截止点时条件授权；
- D6 fold/ZIP：仅在 100-step 行为门通过后条件授权；
- upload/submission：不授权；
- dense、30k、旧 H300 resume：禁止。

下一执行者优先并行推进 R0 与 G0；任一结果只写新 namespace。计划中的时间点是 hard cutoff，不因已投入时间而顺延。

