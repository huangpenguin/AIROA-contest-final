# AXIS Pi0.5 T3 Fast400 H300 R0-only 修复执行计划

> 创建时间：2026-09-15 11:04 JST
> 状态：`ready_for_execution`
> 唯一下一阶段：`R0_repair_and_verify_paired_eval_adapter`
> 本计划结束点：`hard_stop_before_E1`

## 1. 唯一目标

只修复并验证 `run_track3_fast400_h300_paired_eval.py` 的 checkpoint hash、resume/heartbeat 与 zero-action gate 合同，使 R6 和 H300 step300 的只读 preflight 都能在保持 fail-closed 的前提下通过。

不启动任何 behavior worker，不创建 E1 case/video/cursor，不重跑训练，不 fold/FAST/ZIP，不上传或提交。

## 2. 允许修改的文件

- `tools/axis_pi05/run_track3_fast400_h300_paired_eval.py`
- `tests/test_track3_fast400_h300_paired_eval.py`
- 新建 code-only R0 evidence：`data/reports/axis_libero_30k/t3_fast400_r6_h300_r0_remediation_20260915/`
- 完成后更新 scoped handoff：`data/reports/axis_libero_30k/t3_fast400_r6_h300_4gpu_r1_local4/NEXT_AGENT_HANDOFF_20260915.json`

不得修改历史 E0/H300/R6/T2 evidence；不得写入旧 Q7/N5 或 E1 rollout namespace。

## 3. R0.1 双哈希合同

1. 实现并显式命名版本化算法：
   - `q0_repo_relative_sha256_v1`：repo-root relative `sha256sum` 文本流；
   - `params_relative_nul_sha256_v1`：params-root relative path + NUL + file SHA。
2. R6 必须同时锁定：
   - Q0=`d9b7e0d15781060fccfd61dc1890ad98993f0224e4aebba1dd6658d7c011931f`；
   - native=`e98d21511d74b36c1daa17c4aebacfe508960db2cc556f92223c561a4bc2fddc`；
   - files=`16`；bytes=`12505927866`。
3. identity/input lock 必须记录算法名与对应值；禁止仅替换常量、删除校验或把两个值继续复用同一个无算法字段。
4. H300 checkpoint 继续使用 native tree，但也要记录算法名、文件数和字节数。

## 4. R0.2 resume 与 heartbeat

1. 新 worker 与 `--resume` 都原子写入当前 PID/GPU/status；resume 增加 `resumed_at_unix`，不得沿用旧 PID 作为 live identity。
2. worker 运行时至少每 30 秒原子刷新 heartbeat；正常、异常和中断结束均停止 updater 并写最终状态。
3. `status` 同时报告 PID liveness 与 heartbeat age/freshness；旧 PID 不得让实际存活的 resumed worker显示为 dead。
4. 不改变同 role/同 GPU 冲突保护，也不允许两个 worker 共用 namespace/port。

## 5. R0.3 zero-action gate

按既有计划文字固定为 role-level systemwide 语义：

- baseline 24 条 action traces 不能全部为零；
- candidate 24 条 action traces 不能全部为零；
- 单个 episode 全零不单独触发名为 systemwide 的 gate；其 success/failure 和其他 integrity gate 仍照常计入。

若执行者认为必须采用 every-episode nonzero，先修改计划/字段名并落盘理由，不得静默改变语义。

## 6. R0.4 focused validation

至少新增以下测试：

- synthetic tree 对两种算法给出稳定且不同的摘要；
- 错误 Q0/native hash、文件数、字节数、norm hash 均 fail-closed；
- real R6 checkpoint validation/preflight 通过；
- real step300 preflight 通过；
- resume 刷新当前 PID/GPU/status；
- heartbeat freshness 与停止语义；
- 单 episode 零 action 不触发 systemwide gate、整个 role 全零必须失败；
- 既有 namespace、execute8、paired coverage、no-ZIP 测试继续通过。

执行：

```bash
.venv_axis_pi05/bin/python -m py_compile tools/axis_pi05/run_track3_fast400_h300_paired_eval.py tests/test_track3_fast400_h300_paired_eval.py
.venv_axis_pi05/bin/python -m pytest -q tests/test_track3_fast400_h300_paired_eval.py
.venv_axis_pi05/bin/python -m tools.axis_pi05.run_track3_fast400_h300_paired_eval preflight --role r6 --gpu 0
.venv_axis_pi05/bin/python -m tools.axis_pi05.run_track3_fast400_h300_paired_eval preflight --role step300 --gpu 1
```

preflight 必须保持只读，不得创建：

- `data/reports/axis_libero_30k/t3_fast400_r6_h300_paired_eval_20260915`
- `outputs/axis_pi05/t3_fast400_r6_h300_paired_eval_20260915`

## 7. 完成与停止条件

R0 只有在以下全部成立时才算完成：

- focused tests、py_compile、strict duplicate-key JSON、diff check 全通过；
- R6 与 step300 两个 real preflight 都通过；
- 四卡无冲突进程；E1 两个 namespace 仍不存在；
- R0 summary 写明源码/tests SHA、双哈希算法和值、测试结果、preflight 结果、GPU/process snapshot、`e1_started=false`；
- scoped handoff 更新为 `ready_for_E1_launch_review`，但不启动 E1。

任一门失败：summary 写 `remediation_failed` 和真实错误，保持 E1 `not_started`，hard stop。

300 optimizer steps是本候选的真实身份；30,000 只能表述为未运行的 canonical 参考，禁止宣称本候选完成 30k。
