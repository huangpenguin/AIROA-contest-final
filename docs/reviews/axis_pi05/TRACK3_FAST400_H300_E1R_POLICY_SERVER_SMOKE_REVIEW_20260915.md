# Track3 Fast400 H300 E1R Policy-Server Smoke Review

> 审核时间：2026-09-15T17:31:16+09:00  
> 分类：`core_remediation_approved_prelaunch_contract_required`  
> 当前 behavior launch：`not_yet_authorized`  
> 唯一下一阶段：`DL0_signal_close_and_active_norm_attestation`

## 1. 结论

Free-port remediation 的核心实现通过审核：`socket.SOCK_STREAM` 修复正确，33 个 focused tests 通过，adapter source provenance、旧失败 namespace guard 和新 attempt_0002 defaults 均已落地。GPU0 smoke 的 policy server 实际完成了 checkpoint restore、HTTP ready、目标 GPU 资源绑定与受控关闭；原始 `status=failed` 是 close predicate 对主动 SIGTERM 返回码 `-15` 的确定性假阴性，不是 server readiness failure。

但在正式 behavior launch 前仍需一个窄范围 L0 合同修复：

1. close 结果必须记录“由父进程主动 SIGTERM、未升级 SIGKILL、已退出、无 orphan”，predicate 只在这些字段同时成立时接受 `-signal.SIGTERM`；
2. policy server 必须通过 health/log 明确证明最终生效的是 Fast400 norm path/SHA，不能只依赖外层预期值。

L0 通过后，不再要求另一个中间 review；按 deadline recovery 长程计划可连续进入 attempt_0002 双 worker、merge 和条件发布。

## 2. 独立证据核验

- adapter SHA-256：`75e3b0edb956580468d623f962b37ca22510e8d3d4a0b16b9477518f521548a1`
- tests SHA-256：`1e6d5c3204a0022e0296290d40e97d1d97a99fd62743b68959cc3514e1446594`
- 独立 focused tests：`33 passed in 59.64s`
- py_compile：passed
- `git diff --check`：passed
- E1R evidence tree SHA-256：`0d69cc20063945f5ed93a42bfa3829a39662c36517901920480a81114f85c7f3`
- smoke artifact SHA-256：`aa4506b6819e81cda8575ee29893e170df3db78764acb373a2ad5359180571e9`
- server log SHA-256：`e94798481c29442356479ed8ed1a96825b0b66ed3924e793671487c4978295e8`
- progress report SHA-256：`6d1c8ea8a0890d51637c82555791c864e1c9a375b86402e25faff83eb6b46ef1`

原始 smoke 的非返回码 gates 全部为 true：fresh child parsed、server started/ready、GPU0 resource seen、非选定 GPU 无 PID、clean shutdown、无 orphan、无 simulator/case/action/physics、attempt_0002 roots absent、旧失败根 fingerprint 不变。总耗时约 `100.5s`，其中 checkpoint restore 约 `14.3s`。

## 3. `-15` 的语义

`PolicyServer.close()` 在 server 仍存活时调用 `process.terminate()`；POSIX 下 Python `Popen.returncode=-15` 表示子进程被父进程请求的 SIGTERM 终止。现有 child/outer predicate 错误地只接受 `0`，因此 ready server 被标记为 failed。

不能简单写成 `returncode in {0, -15}`。close 结果还必须记录并验证：

- `termination_requested=true`；
- `termination_signal=SIGTERM`；
- `kill_required=false`；
- `clean_shutdown=true`；
- PID 不再存活、无 GPU orphan；
- start/health 无错误。

任何外部 SIGTERM、SIGKILL、timeout、未知负返回码或 orphan 继续 fail closed。

## 4. Fast400 norm 审核

server log 中的 `Loaded norm stats from ...four_gold_80_20_v1...` 来自 `train_config.data.create()` 的 config-side load。随后 `checkpoint_http_server.CheckpointPolicy` 读取 `AXIS_PI05_NORM_STATS`，把显式 stats 传给 `create_trained_policy(..., norm_stats=...)`；实际 Normalize/Unnormalize 使用显式参数，所以源码路径表明本次 policy 使用的是 Fast400 override，而不是旧 norm。

不过当前日志/health 只显示 config-side 旧路径，没有输出最终 active override 的 path/SHA；smoke artifact 也只记录 parent renderer keys，没有保存两个 norm env。该 provenance 歧义必须在 L0 消除：config-side 与 explicit policy-side 都指向同一个 Fast400 文件，并由 `/health` 返回 active SHA `7e4e24dacb24d6333d32cb982d301761c2f00e8f5daca418109bea226268f333`。

## 5. Runtime 边界

- GPU0–3：均 `5 MiB / 0%`；compute apps 为空。
- 无 worker、policy server 或 trainer。
- 旧失败 R6 evidence tree仍为 `98bed72dfcebdc562de4bf1217256ad028819808a4f309c0039936e16a8da849`。
- 新 attempt_0002 evidence/output roots不存在。
- 未执行 preflight、behavior worker、step100、merge、fold、ZIP、上传或提交。
- 磁盘可用约 `388 GiB`，足够双 worker evidence 与单个 release 候选；不授权额外训练或并行候选。

## 6. 下一步

执行 `docs/AXIS_PI05_T3_FAST400_H300_DEADLINE_RECOVERY_LONG_RANGE_PLAN_20260915.md`。该计划是条件式连续授权：当前 E1 仍为 false；只有 DL0 全部通过，才自动解锁 attempt_0002 的 R6/GPU0 与 step300/GPU1。行为 gate 未通过则无 ZIP；通过后才进入 release 边界正式机械审核。
