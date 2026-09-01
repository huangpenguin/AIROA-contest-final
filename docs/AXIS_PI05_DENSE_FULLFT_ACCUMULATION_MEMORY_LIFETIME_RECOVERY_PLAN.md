# AXIS Pi0.5 Dense Full-FT 梯度累积内存生命周期修复计划

> 制定日期：2026-09-01（Asia/Tokyo）
> 审核输入：root `1ac116f`，nested OpenPI `e31c2bf`
> 当前状态：`dense_fullft_accumulation_lifetime_remediation_failed`
> 唯一目标：保持训练数学配方不变，消除 host gradient accumulation 与审计代码对上一 microbatch 完整梯度树的非必要保留，在本地 4×V100 上完成 1-step 与最多 10-step 数值预飞行。

## 0. 审核结论与问题锁

上一轮不能解释为“单份 microbatch 或模型本身需要超过 32GB/card”，也不能直接据此启动比赛 Blackwell：

- `0.90` 与 `0.95` 都记录了 `microbatch_complete` index `0`、`1`，随后才在 index `2` 的计算中 OOM；
- `0.95` 下每卡 `bytes_in_use` 从 optimizer-step start 的 `13,435,195,392`，增长到 microbatch 0 后的 `19,941,608,448`，再到 microbatch 1 后的 `26,420,739,584`；第二次增量约 `6.48GB/card`，与一份完整 FP32 gradient tree 同量级；
- 当前 host loop 在 `grad_sum = _add_gradients(grad_sum, grads)` 后仍保留局部变量 `grads`；非有限审计的 `values` 字典也持有全部 gradient leaves，直到下一次 RHS 计算完成前不会释放；
- 因而当前首要假设是 Python/JAX async 生命周期与审计 observer effect，而不是配方的真实最小容量。必须用内存平台化证据验证，不得只凭代码阅读宣布根因已解决。

旧 OOM、日志和 checkpoint 不删除、不改写。旧状态 `dense_fullft_local_95pct_pool_true_capacity_oom_pending_official_gpu_review` 只作为历史输出保留；本轮审核状态为 `remediation_required`。

## 1. 不可变训练合同

| 字段 | 固定值 |
|---|---|
| initialization | Axis released params revision `c342608c489515efa9ba8281400dc648bf3970e1` |
| data | `physical-intelligence/libero@a4336d…`，原始 OpenPI metadata 10 FPS，1693 episodes；不插帧/重采样 |
| norm | fresh per-dimension QUANTILES，SHA-256 `4c803bf5…b18f5` |
| topology | dense full model，LoRA leaf `0` |
| precision | FP32 |
| optimizer | AdamW β1=.9、β2=.95、eps=1e-8、wd=1e-10，global clip=1.0 |
| EMA | `0.999` |
| devices | 本地 4×V100 32GB；每卡 microbatch `1` |
| accumulation/global batch | `16` / exact `64` |
| allocator | preallocate=true，fraction `0.95` 仅用于恢复 canary |

不得修改 dtype、模型、batch、accumulation 数、EMA、optimizer、loss reduction、RNG、dataset/norm、image/token/action shape、remat 或 sharding来换取通过。不得启动官方 GPU、C20、100/1K/长训、FAST、ZIP、上传或提交。

## M0. CPU 证据修正与 fail-closed summary

1. 从两份 allocator JSONL 自动提取 completed microbatch indexes、每阶段 `bytes_in_use/peak/pool`、OOM micro index 和增量，不手抄推断；
2. 将旧报告中的“首个 microbatch OOM”改为“完成 index 0/1，index 2 未完成”；
3. 旧 summary 的 A2 stale status、A3 null status、顶层缺失 `optimizer_updates/tests/next_action` 必须补齐；
4. machine handoff 顶层 stale `status/next_plan/tests/commit` 必须与执行子树一致；
5. 新增单测，若 `microbatch_complete` 已存在却报告 first-microbatch failure，或 machine/top-level 与 execution 子树冲突，必须失败。

M0 只修证据标签，不改旧 raw logs/checkpoints。

## M1. 内存生命周期最小修复（CPU/code only）

在 `train_step_accumulated_host` 内只修改对象生命周期和等价累积实现：

1. 对非有限审计临时容器使用明确作用域/`finally`，检查结束立即释放 `values` 等对完整 gradient leaves 的引用；
2. 累加后明确释放不再需要的 `grads`、旧 accumulator 和大临时对象；异步加法必须在释放依赖前有可验证的 completion boundary；
3. 优先使用 donated、单独 JIT 的 FP32 tree add，或经测试证明等价的内存复用实现；不得改变求和顺序，若求和顺序改变则不属于本计划授权；
4. 保留逐 microbatch finite 检查，但实现不得把整棵 gradient tree 复制进长期存活的 Python dict/list；
5. allocator trace 增加 `microbatch_compute_complete`、`accumulation_complete`、`temporaries_released` 三个边界，并记录 micro index；异常报告必须写明未完成的 index。

CPU 测试至少覆盖：两叶/嵌套大树 exact 累积、16份 FP32 顺序等价、nonfinite fail-closed、异常路径释放、RNG/optimizer/loss reduction 不变、旧 checkpoint restore 合同不变。

## M2. GPU memory-plateau canary

仅在四张 V100 自然空闲且无外部 PID 时，从 immutable step-0 fresh process restore，使用：

```text
CUDA_VISIBLE_DEVICES=0,1,2,3
XLA_CLIENT_MEM_FRACTION=0.95
XLA_PYTHON_CLIENT_PREALLOCATE=true
```

只尝试一个 optimizer update。硬门禁：

- 16/16 microbatches 均 finite；completed indexes 必须精确为 `0…15`；
- microbatch `1…15` 的 `temporaries_released` live bytes 不得随 index 线性增加；相对 index 1 的漂移门限预注册为 `<=256MiB/card`；
- step `1`、examples seen `64`、scheduler/optimizer/EMA 各只前进一次；
- checkpoint step1 atomic save，fresh-process restore exact/finite；
- source step0 tree hash 不变，GPU0无外部进程，退出后四卡释放；
- 若 compute 能完成而 optimizer update OOM，必须单独分类为 update-phase capacity，不得宣称 accumulation 已整体通过。

若在 index `2` 前再次出现与旧 run 同型线性增长，立即 hard stop。不得尝试 `>0.95`、preallocate=false 或 Blackwell绕过。

## M3. 连续 10-step 数值预飞行

只有 M2 全部通过才从已验证 step1 续跑到 step10：

- 每步 16/16 microbatches、loss/grad/update/params/Adam/EMA finite；
- 每步 memory plateau 门禁继续成立，不得只检查 step1；
- step10 atomic checkpoint、fresh restore和step/RNG/scheduler一致；
- 报告编译后吞吐、每步峰值、checkpoint/restore耗时，只估算100/1K/30K，不执行；
- 完成后 hard stop，等待是否在本地/Blackwell进行100-step的下一次人工审核。

## 4. 20 Hz 边界（本计划不执行 rollout）

标准 LIBERO 的 `physical-intelligence` 10 FPS 与 IPEC 20 FPS 转换版不是简单的两倍帧数据。已比对 episode 0：帧数同为214，8D state与前6维 action exact；差异为 timestamp `0.1→0.05s` 与 gripper `{-1,+1}→{0,1}`。训练 loader按 frame index构造 action horizon，不把 timestamp 输入模型。

比赛 evaluator 的 `pipeline/environment.py` 默认 `control_freq=20`，而历史 R6 C5 审计显式强制10；正式 Track1 `0.47`、Track2 `0.33` 是当前20Hz closed-loop的更直接证据。未来 rollout 必须以20Hz为主协议、10Hz只作对照；混合20Hz数据前必须统一gripper并重算norm。上述事项不改变本轮M0–M3训练授权。

## 5. 交付与最终状态

必须交付新 summary、修复前后 per-micro memory table、1/10-step checkpoint/restore（若门禁允许）、测试、root/nested clean commit。最终状态只能是：

```text
dense_fullft_accumulation_lifetime_remediation_failed
dense_fullft_accumulation_plateau_passed_step1_pending_review
dense_fullft_accumulation_10step_passed_pending_review
```

禁止自动进入官方 GPU、100-step、C20、长训或提交。

## 6. 本轮执行停止点

M0 标签纠偏、M1 CPU 生命周期测试和完整 CPU 回归已通过；完整回归为 `137 passed, 7 skipped, 23 warnings`。M2 从 immutable step-0 fresh restore 启动，恢复 sentinel、FP32/dense/EMA/global batch 64 合同和四卡空闲门禁均通过，但在 `0.95` BFC pool 下 microbatch index `1` 的第二份梯度归并申请 `72 MiB` 时 OOM。已完成的 compute index 为 `[0, 1]`，已完成的 `temporaries_released` 仅为 `[0]`，optimizer update 为 `0`，step1 checkpoint 未生成；因此 memory plateau 未通过，M3 未启动。

证据：`data/reports/axis_libero_30k/dense_fullft_accumulation_memory_lifetime_recovery/m2_step1.json` 与同名 allocator/NVML 原始记录。source step-0 tree hash `9380fd770ced87368c82a05a4dbb8c00fd0f0e82c4e569879aacd127a01038f5` 未改变，四卡退出后均回到约 `5 MiB`，GPU0 无外部进程。该状态等待下一次审核；不得据此启动 Blackwell 或扩大训练范围。
