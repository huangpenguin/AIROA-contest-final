# Track3 Fast400 H300 R0 / E1 Launch Review（2026-09-15 12:03 JST）

> 审核类型：轻量 prelaunch review，不是 release/ZIP 正式审核
> R0 核心结论：`approved`
> E1 launch 结论：`remediation_required`
> E1 状态：`not_started_not_authorized`

## 1. 结论

R0 计划要求的双哈希、resume/heartbeat、role-level systemwide-zero 与 focused validation 均已实现，机器报告及引用 SHA 闭合，R0 核心修复获批准。

但当前 adapter 只为 policy-server 子进程构造 GPU/EGL 环境，真正创建 LIBERO `EnvironmentManager/OffScreenRenderEnv` 的 worker 父进程没有绑定 `MUJOCO_GL=egl`、`MUJOCO_EGL_DEVICE_ID=<gpu>`、`AXIS_PI05_EVAL_GPU=<gpu>` 或 `LIBERO_CONTROL_FREQ=20`。当前启动 shell 也没有这些变量。按现有 GPU0/GPU1 并行命令启动，两个 simulator 的 EGL 后端与物理 GPU 隔离尚未成立，违反既有 E1 合同。

因此不得直接启动 E1。下一步只修 parent runtime binding 并做无 policy、无 action、无 E1 namespace 的 GPU0/GPU1 fresh-process EGL smoke。

## 2. 已独立批准的 R0 证据

- R0 summary 与 closeout strict duplicate-key JSON/引用 SHA 全部匹配；
- adapter SHA：`dec3768d8470afb6d3b783817e79a6c09a5642c261964a3c46764ddcfc7c833f`；
- tests SHA：`c015f1754cd57ef54dca3ceed01e8ce54baac6edb65a2293d10e82494194de2c`；
- focused tests 独立复现：`21 passed in 59.70s`；py_compile passed；
- R6 real preflight：passed，Q0 `d9b7...`、native `e98d...`、16 files、12,505,927,866 bytes；
- step300 real preflight：passed，native `6182a8dd...`、16 files、12,512,823,285 bytes；
- heartbeat 30 秒周期、resume current PID/GPU/status、role-level systemwide-zero 均有实现与 focused coverage；
- E1 evidence/output root 不存在；无 behavior worker、policy server 或 trainer；GPU0–3 均 5 MiB、0%。

## 3. 审核发现

### F1 — P1：simulator 父进程未绑定 per-worker EGL GPU

`server_environment(gpu)` 在 adapter 第 594–617 行返回完整环境，但只在 `PolicyServer.start()` 中作为 `subprocess.Popen(..., env=...)` 传给模型服务子进程。`run_worker()` 第 989 行开始后没有 `os.environ.update(...)` 或等价的 parent binding；第 1048–1053 行直接导入并创建 LIBERO environment。

模型服务的 CUDA 隔离不能替代 simulator 父进程的 EGL 隔离。当前 preflight 只验证 GPU idle/checkpoint/hash，不创建 renderer，所以两个 preflight passed 不能证明 E1 并行 renderer 安全。

### F2 — P2：scoped handoff 同时保留相互冲突的新旧状态块

handoff 顶层与 `r0_execution` 已指向 R0 completed，但旧 `audit` 仍写 R6 blocked/12 tests，`status_audit` 又指向“R0 not started”的旧文档同时写入 R0 completed。历史记录可以保留，但必须明确收纳为 history，并增加唯一 current launch-review pointer，避免下一 agent 误读。

## 4. 边界

本审核没有启动 renderer smoke、behavior worker、step100、merge、fold、FAST、ZIP、训练、上传或提交。R0 历史 evidence 不修改。唯一下一计划为 `docs/AXIS_PI05_T3_FAST400_H300_E1_PARENT_EGL_BINDING_REMEDIATION_PLAN_20260915.md`。
