# AXIS Pi0.5 T3 Fast400 H300 E1 R6 Free-Port Remediation Plan

> 创建时间：2026-09-15T16:26:52+09:00  
> 状态：`ready_for_execution`  
> 唯一下一阶段：`E1R_fix_free_port_and_attempt_provenance`  
> 结束点：`hard_stop_before_behavior_workers`

## 1. 目标

修复 GPU0 R6 worker 在 policy server 创建前触发的 `_free_port()` NameError，补齐运行时 regression 与 adapter source provenance，并为后续 R6/step300 建立新的 attempt_0002 namespace。完成后只做 no-action policy-server readiness smoke 和只读 preflight，硬停等待 launch review。

## 2. 允许范围

允许修改：

- `tools/axis_pi05/run_track3_fast400_h300_paired_eval.py`
- `tests/test_track3_fast400_h300_paired_eval.py`
- 新 remediation evidence：`data/reports/axis_libero_30k/t3_fast400_r6_h300_e1_r6_free_port_remediation_20260915/`
- scoped handoff 和本计划的完成报告

旧失败根禁止任何写入：

- `data/reports/axis_libero_30k/t3_fast400_r6_h300_paired_eval_20260915/`
- `outputs/axis_pi05/t3_fast400_r6_h300_paired_eval_20260915/`

禁止修改 checkpoint、norm、evaluation freeze、manifests、训练结果、E1P0 smoke 或历史 Q7/N5 evidence。

## 3. R1：最小运行时修复

将 `_free_port()` 中唯一错误的 `SOCK_STREAM` 改为 `socket.SOCK_STREAM`。不要改端口分配算法、host、policy protocol 或 server timeout。

全仓搜索必须证明本 adapter 不再存在未限定的 `SOCK_*` 名称。

## 4. R2：attempt/source provenance

在 fresh process 中计算当前 adapter 文件 SHA-256，并至少写入：

- `preflight.json`
- `attempt.lock.json`
- worker `summary.json`
- process-level `failure.json`

resume 必须要求 lock 中的 adapter path/SHA 与当前文件完全一致；缺字段或 SHA 不一致一律 fail closed。merge 必须要求 R6/step300 两个 lock 使用同一个当前 adapter SHA。

旧 failed lock 没有 source SHA，因此修复后对旧根执行 `--resume` 必须明确失败，且失败前不能重写旧 failure/cursor/heartbeat。

将 adapter 的默认 E1 roots更新为新的 sibling namespaces：

- evidence：`data/reports/axis_libero_30k/t3_fast400_r6_h300_paired_eval_20260915_attempt_0002`
- output：`outputs/axis_pi05/t3_fast400_r6_h300_paired_eval_20260915_attempt_0002`

将旧失败 evidence/output roots加入 immutable namespace guard；任何等于旧根或落在旧根下的 worker/preflight/merge 写目标都必须拒绝。只读 status 可以显式查看旧根，但不得更新它。

## 5. R3：focused regression

新增测试至少覆盖：

1. `_free_port()` 可实际调用，返回 `1..65535` 的整数；函数返回后能用新的 `socket.AF_INET/socket.SOCK_STREAM` socket 绑定该 port，证明临时 socket 已关闭；
2. `PolicyServer.__init__()` 在不启动 subprocess 的条件下可取得合法 port；
3. 默认 evidence/output roots 是 attempt_0002，且旧失败根被写入 guard 拒绝；
4. fresh lock/preflight/summary/failure 包含当前 adapter source SHA；
5. adapter SHA 缺失或不一致的 resume 在修改任何旧 evidence 前 fail closed；
6. merge 拒绝两个 role lock 的 adapter SHA 不一致；
7. 既有 parent/child GPU binding、dual hash、heartbeat、zero-action、coverage、gate 与 namespace tests 继续通过。

运行：focused pytest、py_compile、duplicate-key rejecting JSON parse、`git diff --check`。测试不得启动真实 behavior worker。

## 6. R4：no-action policy-server readiness smoke

代码与 tests 通过后，只允许 GPU0 在 fresh process、独立 remediation output 下执行一次 R6 policy-server readiness smoke：

1. 应用已经审核的 GPU0 parent/child runtime contract；
2. 构造 `PolicyServer`，确认合法端口；
3. 启动 R6 checkpoint + Fast400 norm server并等待 ready；
4. 不创建 LIBERO environment，不发送 reset/action 请求，不运行 case；
5. 立即 clean close；记录 PID、port、GPU UUID、runtime、server log SHA、return code、GPU before/during/after；
6. 四卡最终回到空闲，无 orphan；
7. 不创建或写入旧失败根与新的 behavior attempt_0002 roots。

如 readiness smoke 失败，保留真实证据并硬停，不重试、不启动 behavior worker。

## 7. R5：只读 preflight 与停止

对新的默认 attempt_0002 roots 依次执行：

- R6/GPU0 preflight；
- step300/GPU1 preflight。

两项必须通过且新的 behavior roots仍不存在。完成 summary 记录旧失败 tree SHA、新 adapter/test SHA、focused test 数量、source-provenance regression、readiness smoke artifacts、两项 preflight 和最终 GPU/process snapshot。

随后 scoped handoff 改为 `ready_for_E1_attempt_0002_launch_review`，但 `e1_authorized=false`。必须硬停等待独立 review；不得顺手启动 R6 或 step300。

## 8. 后续 launch 形态（本计划不执行）

新的 launch review 通过后，R6/GPU0 与 step300/GPU1 都应 fresh 启动到 attempt_0002；先确认 R6 healthy，再启动 step300。不得 resume 或 merge 旧失败 attempt。GPU2/GPU3 继续保留，step100 仍不授权。

H300 仍只能称为本地 300 optimizer steps；本阶段 `official_result=false`，不需要 VPN，不进行 ZIP/上传/提交。
