# Track3 Fast400 H300 E1 R6 Free-Port Failure Review

> 审核时间：2026-09-15T16:26:52+09:00  
> 分类：`remediation_required`  
> 唯一下一阶段：`E1R_fix_free_port_and_attempt_provenance`  
> 当前 E1 launch authority：`false`

## 1. 结论

GPU0 R6 的首次正式 worker 已在 0/24 cases、policy server 启动前发生确定性 process-level failure。根因是 `_free_port()` 使用了未限定名称 `SOCK_STREAM`，而文件只导入了 `socket` 模块；正确调用应为 `socket.SOCK_STREAM`。

当前不能继续 GPU1 step300，不能直接 `--resume`，也不能复用或覆盖失败的 R6 namespace。先执行窄范围 remediation，完成新 adapter SHA、focused regression、GPU0 no-action policy-server readiness smoke 和新的 launch review。

## 2. 独立复现与测试缺口

现场源码：

```python
def _free_port() -> int:
    with socket.socket(socket.AF_INET, SOCK_STREAM) as sock:
```

独立调用 `_free_port()` 稳定复现：

```text
NameError: name 'SOCK_STREAM' is not defined
```

同时 `py_compile` 通过。这是合法语法中的运行时名称解析错误，静态编译不会发现。现有 26 个 focused tests 没有调用 `_free_port()` 或 `PolicyServer.__init__()`，因此该缺陷未被覆盖。

仓库内其他同类实现都使用 `socket.SOCK_STREAM`；本 adapter 是搜索范围内唯一未限定的 `SOCK_STREAM`。

## 3. 失败边界核验

- adapter SHA-256 未改变：`fbf07adae1a5895bbdd15b996346d3563cae4ad613c167b795d1a9688187334b`
- worker PID：`233982`；GPU0；failure timestamp：`1789456398.8140616`
- R6：0/24 completed；无 `.partial`
- GPU1 step300：未启动
- policy server：未启动；无 `policy_server.log`、`server_shutdown.json`
- action calls：0；physics steps：0；behavior result：`not_evaluated`
- GPU0–3：均 `5 MiB / 0%`
- 无 live worker、policy-server 或 trainer
- root/nested HEAD 与 E1P0 launch review 一致；既有 dirty worktree 未清理或覆盖

失败 evidence 均通过 duplicate-key rejecting JSON parse。冻结哈希：

- R6 evidence tree：`98bed72dfcebdc562de4bf1217256ad028819808a4f309c0039936e16a8da849`
- R6 output tree（无文件）：`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- progress report：`c2b644941f935c8317a62effa12a0c4adc23665ac923aaf18e48e1d028313ea6`
- failure.json：`864e7ab4dfad3f46ca1eb78030e514f4de2e2e59f47ddbd5fd4dacb9aec28cde`
- launcher stderr：`2a54a66b9ad59c07ae2647185b630acd4c41c0207885064137fc79b9e0301ede`

## 4. 为什么不能在旧根 resume

旧 `attempt.lock.json` 锁定 checkpoint、输入 hash、role 和 GPU，但没有锁定 adapter source SHA。修复必然改变源码；直接 `--resume` 会把旧源码创建的 failed lock 与新源码的后续运行混入同一 attempt，并覆盖 failure/cursor/heartbeat 当前状态。虽然尚无 case 或 server log，provenance 仍不完整。

因此旧根必须保持 immutable。修复后的 R6 与 step300 都应在新的 sibling `attempt_0002` evidence/output roots fresh 启动，不能只把 R6 搬到新根而让 step300 使用旧默认根。

## 5. Remediation 要求

执行 `docs/AXIS_PI05_T3_FAST400_H300_E1_R6_FREE_PORT_REMEDIATION_PLAN_20260915.md`，范围只包括：

1. 修复 `socket.SOCK_STREAM`；
2. 给 `_free_port()` 增加真实运行 regression；
3. 将 adapter source SHA 写入 preflight/attempt lock/summary/failure，并在 resume/merge 时 fail closed；
4. 将默认 E1 roots 切换到新的 sibling attempt_0002，并显式拒绝写入本次失败根；
5. 在独立 remediation namespace 做 GPU0 no-action policy-server readiness/close smoke；
6. focused tests、py_compile、strict JSON、diff-check、R6/step300 preflight 全部通过后硬停，等待新的 launch review。

## 6. 未授权事项

- 不启动 R6 或 step300 behavior worker；
- 不使用 `--resume`；
- 不启动 step100；
- 不 merge、fold、FAST、ZIP、上传或提交；
- 不修改 checkpoint、norm、case panel、训练 summary 或 release gate；
- 不把本次失败计为模型行为失败或官方结果。
