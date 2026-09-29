# Track3 Fast400 H300 E0 paired-eval adapter 范围审核（2026-09-15）

> 审核时间：2026-09-15 07:44 JST  
> 审核类型：轻量范围审核，不是 release/ZIP 正式审核  
> 唯一结论：`remediation_required`  
> E1 状态：`blocked_before_launch`

## 1. 结论

E0 已完成 adapter 和 focused tests 的主体实现，且没有重跑训练、启动行为评测、fold、生成 ZIP、上传或提交。step300 的只读 preflight 可通过；但 R6 preflight 当前必然 fail-closed，原因是 adapter 把两种不同定义的目录 SHA 当成同一个值比较。

同一份 R6 `params/` 的独立复算结果为：

- 文件数：16；总字节：12,505,927,866；
- 历史 Q0 `sha256(sort -z file-list | xargs -0 sha256sum)` 等价算法，路径相对仓库根目录：`d9b7e0d15781060fccfd61dc1890ad98993f0224e4aebba1dd6658d7c011931f`；
- adapter-native tree v1，路径相对 `params/`、NUL 分隔：`e98d21511d74b36c1daa17c4aebacfe508960db2cc556f92223c561a4bc2fddc`。

历史冻结值仍被原算法精确复现，因此本次没有 R6 权重漂移证据。不能只把常量从 `d9b7...` 替换成 `e98d...`；必须显式区分并锁定算法及路径域。

## 2. 独立核验

- focused tests：`.venv_axis_pi05/bin/python -m pytest -q tests/test_track3_fast400_h300_paired_eval.py`，`12 passed in 0.09s`；
- step300 只读 preflight：`passed`；checkpoint step=300、training_steps=300、train_state present、params tree=`6182a8ddabf6602b76ef9c18fc1ee487f192ee306135d0775d954b25204e9f4e`；
- R6 只读 preflight：按设计 fail-closed，实际错误为 `e98d... != d9b7...`；
- Fast400 norm SHA：`7e4e24dacb24d6333d32cb982d301761c2f00e8f5daca418109bea226268f333`；
- 评测面板仍为 frozen reverse 4 + C20 20，R6 与 H300 step300 matched pairing；
- GPU0–3 各 5 MiB、0% utilization；没有实际 trainer/evaluator/policy-server 进程；
- E1 evidence/output namespace 尚未生成；未 fold、未生成 ZIP、未上传/提交；
- T2 H10K 官方分 `0.13` 仅为用户报告、未独立验证；它不改变 T3 gate，但强化了“行为证据通过前不 release”的必要性。

## 3. 审核发现

### F1 — P1：R6 哈希算法与冻结值口径错配，错误阻断 E1

adapter 的 `sha256_tree()` 使用相对 `params/` 的路径、NUL 和文件 SHA 组成树摘要；`validate_checkpoint()` 却把结果直接与明确标注为 `R6 params q0 SHA` 的 `d9b7...` 比较。因此 `e98d...` 和 `d9b7...` 的差异是算法域差异，不是文件内容差异。

修复必须保留 fail-closed：加入有明确名称和版本的 `q0_repo_relative_sha256_v1` 与 `params_relative_nul_sha256_v1`；R6 至少校验历史 Q0 值，并将 native tree 值连同算法标识、文件数和字节数写入 identity。禁止只改 expected 常量或放宽校验。

### F2 — P1：12 个 tests 通过，但没有覆盖计划要求的错误 norm/hash fail-closed

现有 norm 测试只断言常量和路径；resume 测试只验证完成 case 的加载。测试没有调用真实 R6 checkpoint 校验，也没有证明错误 Q0/native hash 会拒绝、正确双哈希会通过，所以无法捕获 F1。

修复后必须新增：双算法 synthetic fixture、错误 hash fail-closed、R6 real-path preflight、step300 preflight，以及错误 norm hash fail-closed。两个角色只读 preflight 都通过才可关闭 E0。

### F3 — P2：resume 后的 PID/heartbeat 可能继续保留旧进程身份

`--resume` 读取旧 cursor/heartbeat 后没有把 `pid`、`gpu`、`status` 和 resume 时间更新为当前 worker；运行中 heartbeat 也只在开头和结束写一次。`status` 优先读取 cursor PID，因此 resumed worker 可能实际存活却报告 `pid_alive=false`，长 case 卡住时 heartbeat 也无法区分“在跑”与“已失联”。

E1 前应让每次启动/恢复原子刷新当前 PID/GPU，并用独立心跳或至多 30 秒周期更新时间；增加 resumed PID 与 heartbeat freshness 测试。

### F4 — P2：`no_systemwide_all_zero_actions` 的实现比名称/计划更严格

当前实现要求 baseline 和 candidate 的每一个 case 都至少出现一次非零 action；这不是“整套系统性全零”，而是“所有 48 条 episode trace 均非全零”。这会产生额外的 fail-closed 假阴性。

E1 前必须二选一并写测试锁定：按计划语义实现为每个 role 的整套 24-case 不能全部为零；或把 gate/计划显式改名为 every-episode nonzero。不得让名称和执行语义继续不一致。

## 4. 已批准部分与边界

以下 E0 设计可保留：新 namespace、Fast400 norm 显式注入、GPU0 授权、不同 role/GPU 的并发冲突检查、20 Hz/chunk10/execute8/max300、strict duplicate-key JSON/JSONL、case fingerprint/seeds 配对、merge 不 fold/不 ZIP，以及 gate 失败时 `behavior_gate_failed_no_zip`。

本审核不批准 E1 启动，也不授权修改历史 evidence。下一唯一阶段见 `docs/AXIS_PI05_T3_FAST400_H300_E0_REMEDIATION_AND_E1_PLAN_20260915.md`。
