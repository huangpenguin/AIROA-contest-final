# Track3 Fast400 H300 E1P0 Worker Launch Review

> 审核时间：2026-09-15T12:53:16+09:00  
> 审核分类：`approved_with_operational_constraints`  
> 唯一获准下一阶段：`E1_run_required_r6_and_step300_workers`  
> 本审核未启动 behavior worker、policy server、训练、merge、fold、FAST、ZIP、上传或提交。

## 1. 结论

E1P0 父进程 EGL/GPU 绑定修复通过 launch review。允许下一执行者按专用计划启动：

- GPU0：R6 params-only baseline + Fast400 norm；
- GPU1：H300 step300 + Fast400 norm；
- GPU2/GPU3：本阶段保留，不启动 step100。

两个必需 worker 完成后必须硬停在 merge 前。当前仍没有行为分数，H300 behavior eval 仍是 `not_evaluated`。

## 2. 独立核验结果

### 2.1 代码与绑定顺序

- adapter SHA-256：`fbf07adae1a5895bbdd15b996346d3563cae4ad613c167b795d1a9688187334b`
- tests SHA-256：`ce99d395fdcfd2c76704db805067faf56c5cb476442d2c2fb7da02335848ee21`
- `run_worker()` 在 `preflight()`、E1 role root 计算/创建以及 LIBERO/MuJoCo import 前调用 `bind_parent_renderer_environment(gpu)`。
- parent 强制覆盖并验证 `CUDA_VISIBLE_DEVICES`、`MUJOCO_GL=egl`、`MUJOCO_EGL_DEVICE_ID`、`AXIS_PI05_EVAL_GPU`、20 Hz、冻结 config path 与 asset bridge。
- policy child 在启动前再次验证 parent runtime，并使用同一 GPU index 的环境；不一致会 fail closed。
- module 顶层只加载标准库、NumPy 与不导入 simulator/JAX 的本地 common helper；没有发现早于绑定的 LIBERO/MuJoCo import。

### 2.2 E1P0 smoke 证据

- 首次失败原样保留：`-m __main__` 在 renderer import 前返回 1；GPU0/GPU1 均未出现显存增量、policy、action 或 E1 namespace。
- 修正 attempt_0002 使用相同冻结 reverse case，顺序为 GPU0 后 GPU1，均为 fresh process。
- GPU0：PID `208821`，UUID `GPU-0a004a52-e72d-9b46-0391-9488e841db41`；显存 `5 -> 563 MiB`，其他 GPU 保持 `5 MiB`。
- GPU1：PID `209291`，UUID `GPU-dbf876c3-dcaf-054b-ac9d-035e7198ceda`；显存 `5 -> 563 MiB`，其他 GPU 保持 `5 MiB`。
- 两次均创建、reset/恢复冻结 init state、生成两路 `uint8[128,128,3]` finite frame 并关闭；`policy_server_started=false`、`action_calls=0`、`physics_step_calls=0`。
- graphics-only EGL PID 没有出现在 `--query-compute-apps` 中；GPU 归属由 fresh-process 存活窗口内仅目标卡出现的显存/利用率增量证明。未把 compute query 的缺失伪装成 PID 证据。
- attempt_0002 summary SHA-256：`e37eea66db6ad102d28a0c35e060ef02085ff6604d7484a8ae28ee181fa669ea`
- GPU0 smoke SHA-256：`bbf88932a513a4fe8a659f0683c65b7f90e66b66aae268bcea8e8597031f1078`
- GPU1 smoke SHA-256：`fcce7fc7749bd00bf972295672fe44481a50a56201baa63b1adf137e078287f4`
- 7 个 E1P0 JSON 全部通过 duplicate-key rejecting parse；报告声明的文件 SHA 与现场计算一致。

### 2.3 独立复验

- focused tests：`26 passed in 60.05s`
- R6/GPU0 preflight：`passed`
  - Q0 hash：`d9b7e0d15781060fccfd61dc1890ad98993f0224e4aebba1dd6658d7c011931f`
  - native hash：`e98d21511d74b36c1daa17c4aebacfe508960db2cc556f92223c561a4bc2fddc`
  - 16 files / 12,505,927,866 bytes
- step300/GPU1 preflight：`passed`
  - native hash：`6182a8ddabf6602b76ef9c18fc1ee487f192ee306135d0775d954b25204e9f4e`
  - 16 files / 12,512,823,285 bytes
- 两项 preflight 均锁定 4 reverse + 20 C20、Fast400 norm `7e4e24da...`、20 Hz/chunk10/execute8/max300，且为只读执行。

### 2.4 现场边界

- E1 evidence root：不存在。
- E1 output root：不存在。
- GPU0–3：均 `5 MiB / 0%`。
- 无训练、behavior evaluator 或 policy-server 进程。
- root HEAD：`36e8f269ecd30d4a2a6e555e67bd7315011e5dbf`
- axis_openpi HEAD：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
- LIBERO-plus HEAD：`4976dc30028e805ff8094b55501d532c48fec182`
- LIBERO HEAD：`8f1084e3132a39270c3a13ebe37270a43ece2a01`
- 既有 dirty worktree 未清理、未回退、未提交。

## 3. 发现与约束

### O1：smoke 执行时 source SHA 未直接写入 attempt artifact（非阻断）

最终 E1P0 summary 透明记录：成功 smoke 后只收紧了 parent-smoke 的通过谓词，当前 source SHA 因而晚于 smoke。attempt artifact 本身没有执行时 source SHA，不能单独证明该次进程的精确源码快照。

该缺口不阻止本次 worker 启动：smoke artifact 直接保存了命令、环境、frame schema、GPU 全时序 snapshots 和关闭状态；当前源码的绑定路径又通过独立 tests 与两项 preflight。此 smoke 只作为 E1P0 launch gate，不得冒充最终 release/parity 证据。

### O2：process-level resume 可能覆盖固定名 server 日志（运行约束）

`PolicyServer.start()` 当前以写模式打开固定的 `policy_server.log`，`server_shutdown.json` 也是固定路径。首次正常启动不受影响，但进程级失败后直接 `--resume` 可能覆盖原始 server 证据；若存在 `.partial`，当前 runner 也会 fail closed。

因此下一阶段允许首次 launch，不允许在 process-level failure 后未经审查直接 `--resume`。必须先保全日志/partial/failure/cursor/heartbeat，再另行决定恢复或补丁。

## 4. 未改变的事实

- H300 是本地 `300` optimizer steps，不是官方 canonical `30,000` steps；不得宣称完成 30k。
- T2 H10K 官方 `0.13` 是用户报告值，未在本次审核中独立验证；它强化“先行为评测、后 ZIP”的必要性，但不改变冻结 gate。
- 本地 worker 完全离线：`HF_HUB_OFFLINE=1`、本地 checkpoint/norm/asset、loopback HTTP；运行不依赖 VPN。
- 未通过配对行为 gate 前不得 fold、FAST export 或生成 T3 ZIP；任何上传/比赛提交仍禁止自动执行。

## 5. 下一步

严格执行 `docs/AXIS_PI05_T3_FAST400_H300_E1_REQUIRED_WORKERS_EXECUTION_PLAN_20260915.md`。完成两个必需 worker 后落盘 closeout 并硬停在 merge review 前。
