# AXIS Pi0.5 T3 Fast400 H300：E0 修复与 E1 启动计划

> 创建时间：2026-09-15 07:44 JST  
> 当前状态：`ready_for_remediation`  
> 唯一下一阶段：`R0_repair_hash_contract_and_monitoring`  
> E1：`blocked_until_R0_passes`

## 1. 目标与事实边界

不重跑已完成的 `t3_fast400_r6_h300_4gpu_r1` 训练。先修复 paired-eval adapter 的哈希合同、测试空洞和运行监控语义；只有 R6 与 H300 step300 的只读 preflight 同时通过后，才在本地启动 matched Fast400-norm E1。

本候选真实训练长度是 300 optimizer steps，是短程诊断/候选，不是官方 canonical 30,000-step 复现。后续 evidence、fold manifest 和 ZIP provenance 必须写实际 `training_steps=300`；可以提 30k 仅用于说明“未运行、未宣称复现”，不得把 300 包装成 30k。

冻结行为 gate 不变：reverse-dev delta `>= +1/4`、C20 delta `>= -2/20`、24/24 exact paired coverage、零基础设施/model/nonfinite 错误。T2 H10K 的用户报告官方分 `0.13` 不参与 T3 数学 gate，但禁止绕过本地行为门直接制作/提交 T3 ZIP。

## 2. R0：修复 adapter，禁止启动评测

责任文件：

- `tools/axis_pi05/run_track3_fast400_h300_paired_eval.py`
- `tests/test_track3_fast400_h300_paired_eval.py`

必须完成：

1. 把目录摘要拆成两个带版本的显式算法：
   - `q0_repo_relative_sha256_v1`：文件路径相对 repo root，精确复现历史 Q0 `sha256sum` 文本流；R6 expected=`d9b7e0d15781060fccfd61dc1890ad98993f0224e4aebba1dd6658d7c011931f`；
   - `params_relative_nul_sha256_v1`：文件路径相对 params root，NUL 分隔；当前 R6 observed=`e98d21511d74b36c1daa17c4aebacfe508960db2cc556f92223c561a4bc2fddc`。
2. R6 identity 同时记录算法名、摘要、`file_count=16`、`total_bytes=12505927866`；历史 Q0 校验必须保持 fail-closed。H300 可继续用 native tree 算法，但字段必须带算法标识，不能再把不同口径都叫无定义的 `params_tree_sha256`。
3. 不允许仅把常量从 `d9b7...` 改成 `e98d...`，也不允许删除 R6 校验。
4. `--resume` 每次都把 attempt lock、cursor、heartbeat 的 PID/GPU/status 更新到当前进程，并记录 `resumed_at_unix`；运行中 heartbeat 至少每 30 秒原子更新，正常/异常结束均停止。
5. 对 action-zero gate 做一次明确决策。默认按既有计划文字采用 role-level systemwide 语义：baseline 的 24 条不能全部为零，candidate 的 24 条不能全部为零；如果选择 every-episode nonzero，必须同步改 gate 名称与本计划并说明理由。
6. 继续保持新 namespace、strict JSON、Fast400 norm、role/GPU 冲突保护、无自动 fold/ZIP/upload/submit。

R0 验证门：

- focused tests 覆盖正确/错误双哈希、错误 norm、real R6 preflight、step300 preflight、resume 当前 PID、heartbeat freshness、systemwide-zero 边界；
- focused tests 全通过；
- `preflight --role r6 --gpu 0` 通过；
- `preflight --role step300 --gpu 1` 通过；
- preflight 不创建 E1 evidence/output；
- GPU0–3 仍无冲突进程。

R0 失败时：保留失败输出为诊断，不创建 worker namespace，不启动 E1。

## 3. E1：R0 通过后的并行 matched eval

先启动两个必需 worker：

1. GPU0：R6 params-only baseline + Fast400 norm，frozen 24 cases；
2. GPU1：H300 step300 candidate + Fast400 norm，同一 frozen 24 cases；
3. 两者 server 健康、PID/heartbeat 正常、至少完成首个 case 后，GPU2 才可启动可选 step100 diagnostic；
4. GPU3 保留给失败恢复或后续 gate-passed 的 fold/parity/FAST，不为“占满四卡”复制同一 role。

当前 adapter 是每个 role 一个 worker，第四张卡不会自动把单个 24-case role 加速。若确需四卡切 panel，必须先设计无重叠 shard、独立 cursor/port/output 和严格 merge；本轮时间优先级下不扩展该范围。

推荐命令（仅在 R0 全部通过后）：

```bash
.venv_axis_pi05/bin/python -m tools.axis_pi05.run_track3_fast400_h300_paired_eval worker --role r6 --gpu 0
.venv_axis_pi05/bin/python -m tools.axis_pi05.run_track3_fast400_h300_paired_eval worker --role step300 --gpu 1
```

可选 diagnostic：

```bash
.venv_axis_pi05/bin/python -m tools.axis_pi05.run_track3_fast400_h300_paired_eval worker --role step100 --gpu 2
```

评测链路设置 `HF_HUB_OFFLINE=1`，使用本地 checkpoint、norm 和 LIBERO assets，不依赖 VPN；只有用户自己访问比赛网页、上传或查看官方结果时才需要其网络/VPN。

## 4. E2：merge 与唯一 release 决策

R6 和 step300 各 24/24 完成、无 active partial、input lock/identity 完全一致后执行只读校验并 merge。step100 不得阻塞 required merge。

- gate 失败：写 `behavior_gate_failed_no_zip`；不 fold、不 ZIP、不提交；
- gate 通过：写 `behavior_gate_passed_fold_authorized`，才进入 E3；
- 任一基础设施/model/nonfinite 错误：本次 merge fail-closed，先按新 attempt/明确恢复流程处理，不能把异常 episode 当普通失败混入 winner gate。

## 5. E3：仅 gate 通过后

按既有 `docs/AXIS_PI05_T3_FAST400_H300_POSTTRAIN_EVAL_RELEASE_HANDOFF_20260915.md` 执行 step300 fold、raw-LoRA↔folded JAX↔PyTorch/FAST parity、latency/smoke、ZIP 和正式机械发布审核。仍然禁止自动上传或比赛提交。

## 6. 终止条件

本计划完成条件二选一：

- 得到完整 matched evidence 与 `behavior_gate_failed_no_zip`；或
- gate 通过后得到机械审核通过、SHA/bytes 锁定、实际 `training_steps=300` provenance 清楚且尚未提交的 T3 ZIP。

禁止项：重跑训练、dense/30k continuation、改写历史 R6/T2/H300 evidence、复用旧 Q7/N5 namespace、在行为 gate 前 fold/ZIP、自动上传/提交。
