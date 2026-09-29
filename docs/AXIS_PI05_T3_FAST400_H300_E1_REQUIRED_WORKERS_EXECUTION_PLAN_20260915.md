# AXIS Pi0.5 T3 Fast400 H300 E1 Required Workers Execution Plan

> 创建时间：2026-09-15T12:53:16+09:00  
> 状态：`ready_for_execution`  
> 唯一下一阶段：`E1_run_required_r6_and_step300_workers`  
> 结束点：`hard_stop_before_merge`

## 1. 唯一目标

并行完成冻结 24-case 面板上的两个必需 behavior worker：

- GPU0：R6 params-only baseline，使用 Fast400 norm；
- GPU1：H300 step300 candidate，使用同一 Fast400 norm。

本阶段不启动 step100、不 merge、不 fold、不导出 FAST、不生成 ZIP、不上传、不提交，也不启动任何新训练或 30k continuation。

## 2. 冻结身份

- adapter：`tools/axis_pi05/run_track3_fast400_h300_paired_eval.py`
- adapter SHA-256：`fbf07adae1a5895bbdd15b996346d3563cae4ad613c167b795d1a9688187334b`
- review：`docs/reviews/axis_pi05/TRACK3_FAST400_H300_E1P0_WORKER_LAUNCH_REVIEW_20260915.md`
- evidence root：`data/reports/axis_libero_30k/t3_fast400_r6_h300_paired_eval_20260915`
- output root：`outputs/axis_pi05/t3_fast400_r6_h300_paired_eval_20260915`
- panel：reverse-dev 4 + C20 20，exact coverage 24
- protocol：20 Hz、chunk10、execute8、max300、QUANTILES
- norm SHA-256：`7e4e24dacb24d6333d32cb982d301761c2f00e8f5daca418109bea226268f333`
- R6 native params SHA-256：`e98d21511d74b36c1daa17c4aebacfe508960db2cc556f92223c561a4bc2fddc`
- step300 native params SHA-256：`6182a8ddabf6602b76ef9c18fc1ee487f192ee306135d0775d954b25204e9f4e`

## 3. Launch 前只读门

下一执行者先按仓库 `AGENTS.md` 重读 authoritative handoff/plan，并只读确认：

1. root/nested HEAD 与本计划审核时一致，或差异已明确归属且不触及本 adapter/checkpoint/eval 输入；
2. adapter SHA 仍等于冻结值；若改变，停止并重新 review；
3. 两个 E1 根目录仍不存在；若已存在，不得按 fresh launch 继续，改为 `status`/PID/heartbeat/lock 取证；
4. 无训练、同 adapter、历史 evaluator 或 orphan policy-server 进程；
5. GPU0/GPU1 均空闲，GPU UUID 仍分别为 `GPU-0a004a52-e72d-9b46-0391-9488e841db41` 与 `GPU-dbf876c3-dcaf-054b-ac9d-035e7198ceda`；
6. 依次再跑一次 R6/GPU0 与 step300/GPU1 `preflight`，都必须通过且不得创建 E1 roots。

任一项不满足即硬停，不要创建第二 namespace，也不要重写已有 cursor/lock。

## 4. 启动顺序

### E1A：GPU0 R6

在持久会话中启动：

```bash
.venv_axis_pi05/bin/python -m tools.axis_pi05.run_track3_fast400_h300_paired_eval worker --role r6 --gpu 0
```

允许 checkpoint/hash preflight 先运行约 1 分钟。确认以下健康信号后才进入 E1B：

- `r6/attempt.lock.json`、`cursor.json`、`heartbeat.json` 出现且严格 JSON 可读；
- lock/cursor/heartbeat 的 role=`r6`、GPU=0、PID 为当前存活 worker；
- heartbeat 在 30 秒周期内推进且 age 不超过 90 秒；
- GPU0 出现该 role 的 parent/policy 资源，GPU1 未被该 role 占用；
- policy server 没有在启动窗口内退出。

### E1B：GPU1 step300

R6 健康后，在另一个持久会话中启动：

```bash
.venv_axis_pi05/bin/python -m tools.axis_pi05.run_track3_fast400_h300_paired_eval worker --role step300 --gpu 1
```

用相同标准确认 role=`step300`、GPU=1、独立 PID/port/role root/output root/heartbeat。随后允许 GPU0/GPU1 并行跑完。

GPU2/GPU3 保持空闲。step100 只属可选诊断，不得在本阶段启动，也不得拖慢两个 required roles。

## 5. 监控与时间预期

只读状态命令：

```bash
.venv_axis_pi05/bin/python -m tools.axis_pi05.run_track3_fast400_h300_paired_eval status
```

每 60–120 秒观察 cursor completed IDs、heartbeat/PID、GPU 与 policy-server 进程；状态无变化但 heartbeat 新鲜时不要误判挂死。历史同协议 case 大约 16–56 秒，单个 24-case role 的保守墙钟预计约 35–70 分钟（含 server load）；两 role 并行的主路径通常应在约 45–75 分钟完成，首次 JAX 编译或长 timeout 可推高到约 90 分钟。这是估算，不是完成承诺。

本地链路使用离线 checkpoint/norm/assets 和 loopback policy HTTP，不依赖 VPN；只需保持机器供电、文件系统和执行会话存活。

## 6. 失败与恢复规则

- episode 内 timeout/collision 是真实行为结果；保存 row/video/action trace 后继续队列，不把它当基础设施错误重跑。
- adapter 捕获的 model/server/nonfinite/asset/init-state exception 必须按原样进入 case evidence；不得手工改成成功。
- 若 worker 进程崩溃、PID 消失、heartbeat 超过 90 秒或出现 `.partial`：只停止受影响 role，另一健康 role 可继续；先保全 `policy_server.log`、`server_shutdown.json`、`failure.json`、partial、cursor、heartbeat 和进程/GPU snapshots。
- process-level failure 后禁止未经审查直接 `--resume`：固定名 server 日志会被写模式覆盖，且 partial 会使 runner fail closed。先落盘失败审查，再决定是否修复 attempt-specific logging/partial policy 后恢复。
- 不删除、不覆盖已完成 case，不创建第二个同 role namespace，不重跑已完成 role。
- 若两个 role 争用同一卡、parent/child GPU identity 不一致或出现非目标卡资源，立即停止受影响 role并保留证据。

## 7. 完成门与硬停

两个 role 各自必须同时满足：

- `summary.status=completed`；
- `case_count=expected_case_count=24` 且 `coverage_complete=true`；
- 24 个 case IDs/fingerprints 与冻结顺序一致；
- case JSON/action trace finite，role-level 非系统性全零；
- cursor/lock 为 completed，heartbeat terminal，PID 不再存活；
- 无 `.partial`、orphan policy server，GPU0/GPU1 已释放。

完成后写 E1 closeout，记录每个 role 的 success count、exception count、summary/case/action/video hashes、起止时间、PID/GPU UUID 和最终 runtime snapshot。随后硬停：

- 不运行 `merge`；
- 不根据肉眼 success count 宣布 gate；
- 不 fold/FAST/ZIP；
- 不上传或提交。

下一阶段只能是独立的 `E1_required_workers_closeout_and_merge_review`。只有该 review 核验 24+24 后，才可授权 merge；只有 merge 的冻结 gate 通过后，才可讨论发布。

## 8. 口径

- H300 必须写作“本地 300 optimizer steps”；官方 canonical 训练口径是 30,000 steps，但本 run 没有完成 30k，不能用“官方 30k”描述本模型。
- 用户报告 T2 H10K 官方分数为 `0.13`，本地未验证。该信息不改变本计划的 paired gate，只说明不得跳过行为评测直接制作候选 ZIP。
- 本阶段所有结果均为本地、`official_result=false`。
