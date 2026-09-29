# AXIS Pi0.5 T3 Fast400 H300 Deadline-Recovery Long-Range Plan

> 创建时间：2026-09-15T17:31:16+09:00  
> 状态：`ready_for_conditional_execution`  
> 当前阶段：`DL0_signal_close_and_active_norm_attestation`  
> 最终目标：`behavior_gate_decision_then_conditional_release`  
> 自动上传/提交：`false`

## 1. 总策略与时间预算

不再训练、不开 dense/30k、不跑 step100。把剩余时间全部放在一条关键路径：

`DL0 合同修复 -> GPU0/GPU1 required workers -> merge -> behavior gate -> 仅通过时 release audit/ZIP`

估算关键路径：

- DL0 代码、focused tests、一次 fresh no-action smoke、两项 preflight：约 10–25 分钟；
- R6/step300 并行 24+24：约 45–90 分钟；
- merge 与行为门：约 5–10 分钟；
- gate 通过后的 fold/parity/FAST/ZIP/机械审核：约 60–120 分钟；
- 总计典型约 2–3.5 小时，首次编译或压缩慢时更长。

令 `D` 为实际比赛提交截止时间：

- `D-now >= 210 分钟`：执行完整链路；
- `120 <= D-now < 210 分钟`：先完成行为门，只有 gate 通过且 `D-now >= 75 分钟` 才进入 release；
- `D-now < 120 分钟`：不存在可信完成全套新 T3 ZIP 的时间保证。可继续本地行为闭环用于证据，但不得为了赶时间跳过 gate/parity/CRC/static validation；是否改交既有候选由用户决定，本计划不自动提交。

## 2. DL0：signal-aware close 与 active norm attestation

允许修改：

- `tools/axis_pi05/run_track3_fast400_h300_paired_eval.py`
- `tools/axis_pi05/checkpoint_http_server.py`
- `tests/test_track3_fast400_h300_paired_eval.py`

禁止修改 checkpoint、norm 文件、manifests、训练 summary、旧 smoke/失败 evidence 和 release gate。

### DL0.1 close contract

1. `PolicyServer.close()` 返回结构化字段：是否由本进程请求 terminate、signal、是否升级 kill、最终 returncode、PID 是否退出。
2. 抽出纯函数判定 controlled shutdown；接受 `0`，或仅在 `termination_requested=true`、`termination_signal=SIGTERM`、`kill_required=false`、无 orphan 时接受 `-signal.SIGTERM`。
3. child 与 outer smoke 复用同一判定；拒绝 `-SIGKILL`、外部 signal、timeout、未 ready、start error 或 orphan。
4. 原 `policy_server_smoke.json` 保持 immutable；新增 derived replay evidence，证明旧 artifact 除 false predicate 外全门通过，不改写原 status。

### DL0.2 active norm contract

1. child 环境同时设置：
   - `AXIS_PI05_NORM_STATS=<Fast400 norm>`；
   - `AXIS_PI05_TRACK3_NORM_STATS=<同一 Fast400 norm>`；
   - `AXIS_PI05_NORM_STATS_SHA256=7e4e24dacb24d6333d32cb982d301761c2f00e8f5daca418109bea226268f333`。
2. `CheckpointPolicy` 必须要求 norm path存在，读取后计算 SHA，并与 expected env SHA 比较；缺失/不一致 fail closed。
3. `/health` 返回严格 JSON：status、active norm resolved path/SHA、checkpoint resolved path、config name。不得暴露无关环境或秘密。
4. `PolicyServer.start()` 在 `wait_for_server` 后读取 `/health` 并验证 Fast400 SHA、checkpoint 与 config；不一致时在任何 reset/action 前关闭并失败。
5. server log明确写 `ACTIVE_POLICY_NORM` path/SHA。config-side 也指向 Fast400，使日志不再出现旧 four-gold norm作为本 run 的 active 候选。
6. worker preflight/lock/summary 与 server shutdown evidence记录同一个 health attestation。

### DL0.3 tests 与门

新增回归至少覆盖 controlled `-SIGTERM` pass、`-SIGKILL` fail、外部 signal fail、orphan fail、health norm SHA match/pass、wrong/missing norm fail、config-side/explicit path一致、worker 在 action 前拒绝错误 health。

运行 focused suite、py_compile、strict JSON、未限定 socket 搜索和 `git diff --check`。不得扩大到全仓测试；预计 focused tests 数量应大于当前 33。

随后在新的 remediation-only namespace 重跑一次 GPU0 fresh no-action policy-server smoke。必须满足：server ready、health Fast400 SHA exact、受控 SIGTERM classified pass、无 simulator/action/physics、无 orphan、旧根不变、attempt_0002 roots仍不存在。该 smoke 失败则停止，不启动 behavior。

若通过，立即执行 R6/GPU0 和 step300/GPU1 两项只读 preflight；二者通过后同一执行者可直接进入 DL1，无需再等待另一轮人工 review。

## 3. DL1：attempt_0002 双 required workers

唯一 roots：

- evidence：`data/reports/axis_libero_30k/t3_fast400_r6_h300_paired_eval_20260915_attempt_0002`
- output：`outputs/axis_pi05/t3_fast400_r6_h300_paired_eval_20260915_attempt_0002`

旧失败 attempt 和 E1R smoke roots全部只读。

### DL1.1 launch

1. GPU0 fresh 启动 R6，不使用 `--resume`。
2. 确认 lock/cursor/heartbeat 的 adapter SHA、role/GPU/PID，server `/health` 的 active Fast400 norm SHA，以及 GPU0 resource均正确。
3. R6 policy ready且首 case进入运行后，立即在 GPU1 fresh 启动 step300。
4. GPU0/GPU1 并行跑完 24 cases；GPU2/GPU3 保留。不得启动 step100。

### DL1.2 monitor

每 60–120 秒只读观察 status、PID、heartbeat age、completed case IDs、GPU 与 server log。heartbeat新鲜时不要因 case 最长约一分钟而误判挂死。

episode timeout/collision 是真实行为结果，保存后继续。process/server/asset/nonfinite error 是基础设施/模型异常，不得重标为行为失败。若某 role process-level failure，另一健康 role可继续；失败 role保全证据并停止，不在 deadline 路径中临时改源码后 resume。

### DL1.3 completion

每个 role 必须：summary completed、24/24 exact coverage、无 partial、source/norm/checkpoint/seed/protocol一致、action trace finite、PID/server退出、GPU释放。任一 required role不完整则不 merge。

## 4. DL2：立即 merge 与行为门

两个 role完成后，先只读核验 24+24 与同一 adapter/norm，再运行一次 `merge`；不得覆盖已有 merge artifact。

全部 gate：

- exact paired coverage `24/24`；
- infrastructure/model/nonfinite exception `0`；
- reverse-dev delta `>= +1/4`；
- C20 delta `>= -2/20`；
- action traces finite；
- R6/step300 均非 role-level systemwide all-zero；
- case IDs/fingerprints/seeds/protocol exact。

若失败，写 `behavior_gate_failed_no_zip` 并结束。不得用 loss、单个成功视频或 T2 的 `0.13` 推翻该门。

## 5. DL3：仅 gate 通过后的 release

Gate 通过且距离 deadline 至少 75 分钟时，进入 release boundary；此处恢复正式机械审核。

1. fold H300 step300 到新的专用 folded root；不得覆盖 R6/T2 或其他 candidate。
2. 写入 Fast400 norm processors/assets；manifest锁定 candidate、300 optimizer steps、QUANTILES、norm SHA、raw/folded identity。
3. 依次执行 raw-LoRA↔folded JAX、folded JAX↔PyTorch/processor、FAST↔folded parity；显式指定 step300，禁止回落默认 R6。
4. GPU2用于主要 parity，GPU3保留给 FAST server/latency或失败诊断；不要并发写同一 release root。
5. 生成唯一 T3 ZIP，执行 entry/CRC、`unzip -t`、`validate_submission.py --static --json`、SHA-256、bytes、source checkpoint/norm/protocol identity。
6. 388 GiB 当前可用空间足够一条 release；不并行生成多个 8–10 GiB ZIP。
7. `official_result=false`、`auto_upload=false`、`auto_submit=false`。最终只把 ZIP 路径/SHA/bytes交给用户。

任何 parity、CRC、static validation 或 provenance gate失败：保留失败证据，不制作“可提交 winner”。

## 6. DL4：赛后长程改进

无论是否赶上 deadline，都保留以下后续工作，不与本次关键路径并行：

1. 将 policy-server lifecycle/health/norm identity 做成共享 contract，替换各 runner 的重复实现；
2. 将 controlled shutdown、port allocation、health attestation加入真实 subprocess integration tests；
3. 将 policy logs改为 attempt-specific immutable filenames，支持不覆盖的 resume；
4. 为本地 proxy panel与官方分数建立校准表，特别记录 T2 本地指标与官方 `0.13` 的偏差；
5. 重新评估 300-step Fast400 对 Axis 底座是否有统计收益，再决定未来是否值得 30k；在没有行为/官方证据前不宣称 Axis 底座提升。

## 7. 沟通口径

- 本模型是“本地 300 optimizer steps”，不是官方 30,000-step run；30000只作为官方 canonical 对照，不写成已完成事实。
- 本地 DL0–DL3 不依赖 VPN；仅最终人工访问比赛 portal上传/提交时可能需要网络/VPN。
- T2 `0.13` 是用户报告的官方分，未在本地独立验证；它说明 proxy 与官方可能偏离，不能用来缩短 T3 的行为/发布硬门。
