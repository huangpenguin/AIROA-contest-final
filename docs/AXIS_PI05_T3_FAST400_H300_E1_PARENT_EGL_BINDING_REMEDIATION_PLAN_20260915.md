# AXIS Pi0.5 T3 Fast400 H300 E1 Parent EGL Binding Remediation

> 创建时间：2026-09-15 12:03 JST
> 状态：`ready_for_execution`
> 唯一下一阶段：`E1P0_bind_and_smoke_worker_parent_runtime`
> 结束点：`hard_stop_before_behavior_workers`

## 1. 目标

在不启动 policy server、行为 rollout 或 E1 worker namespace 的前提下，修复并证明 GPU0 R6 worker 与 GPU1 step300 worker 的 simulator 父进程各自绑定正确的 MuJoCo EGL 设备。

R0 双哈希、heartbeat/resume 和 zero-action 结论保持不变；不得借本计划修改训练、checkpoint、norm、case panel 或 release gate。

## 2. 允许范围

允许修改：

- `tools/axis_pi05/run_track3_fast400_h300_paired_eval.py`
- `tests/test_track3_fast400_h300_paired_eval.py`
- 新建 code/smoke evidence：`data/reports/axis_libero_30k/t3_fast400_r6_h300_e1_parent_egl_remediation_20260915/`
- 完成后更新 scoped handoff。

禁止创建或写入：

- `data/reports/axis_libero_30k/t3_fast400_r6_h300_paired_eval_20260915`
- `outputs/axis_pi05/t3_fast400_r6_h300_paired_eval_20260915`
- 历史 Q7/N5、R6、H300、R0 evidence。

## 3. E1P0：父进程 runtime contract

1. 新增显式、可测试的 parent worker runtime binding，在任何 LIBERO/MuJoCo import、environment 创建和 E1 namespace 创建之前执行。
2. 对物理 GPU `N` 强制设置并记录：
   - `CUDA_VISIBLE_DEVICES=N`
   - `MUJOCO_GL=egl`
   - `MUJOCO_EGL_DEVICE_ID=N`
   - `AXIS_PI05_EVAL_GPU=N`
   - `LIBERO_CONTROL_FREQ=20`
   - `LIBERO_CONFIG_PATH=<repo>/.eval_libero`
   - `AXIS_PI05_LIBERO_ASSET_BRIDGE=1`
3. policy-server 子进程继续使用同一个物理 GPU/NORM/checkpoint contract；parent 与 child 的 GPU identity 必须一致并写入未来 worker preflight/attempt lock。
4. inherited 冲突值必须被显式覆盖或 fail-closed，不能依赖当前 shell 恰好已设置。
5. 绑定/验证失败时不得创建 E1 evidence/output 目录。

## 4. E1P1：focused tests

至少覆盖：

- GPU0 与 GPU1 parent environment 的全部必需键；
- inherited 错误 GPU/EGL/control-frequency 值会被正确处理；
- parent binding 发生在 EnvironmentManager/MuJoCo import 与 namespace 写入前；
- parent/child GPU identity 不一致 fail-closed；
- 既有 21 个双哈希、heartbeat/resume、zero gate、namespace 与 merge tests 继续通过。

运行 py_compile、focused pytest、strict JSON 与 diff-check。测试不得启动真实 behavior worker。

## 5. E1P2：fresh-process EGL smoke

在 GPU0 和 GPU1 串行各运行一次独立子进程 smoke：

1. 应用对应 parent runtime contract；
2. 导入 LIBERO/MuJoCo；
3. 从 frozen panel 选择同一个只读 case，创建 environment、reset 到冻结 init state、获取/渲染初始 observation；
4. 不启动 policy server，不调用 action，不推进行为 episode，不写 E1 namespace；
5. 记录 PID、物理 GPU UUID、运行时环境、frame shape/dtype/finite、创建/关闭状态、运行中与结束后 `nvidia-smi` snapshot；
6. GPU0 smoke 完成并释放后再运行 GPU1，避免把 smoke 本身变成并发诊断。

通过条件：两次 fresh process 均正常创建/渲染/关闭；运行中所选 GPU 有对应 renderer 资源且非所选 GPU 没有该 PID；结束后无 orphan、四卡恢复空闲。

## 6. 终检与停止

完成 summary 必须记录：source/test SHA、focused test 结果、两个 smoke 的 immutable paths/SHA、GPU UUID/PID/环境、E1 roots absent、behavior worker count=0、`e1_started=false`。

全部通过后 scoped handoff 改为 `ready_for_E1_worker_launch_review`，但仍 hard stop，不自动启动 R6/step300 workers。任一 smoke 或 identity gate 失败则写真实 failure，保持 E1 未授权。

后续 E1 仍是 GPU0 R6 + GPU1 step300；GPU2 step100 仅可选，GPU3 保留。300 optimizer steps必须如实记录，不能宣称完成官方 canonical 30k。
