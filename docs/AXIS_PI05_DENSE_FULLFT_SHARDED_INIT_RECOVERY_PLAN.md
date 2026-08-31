# AXIS Pi0.5 Dense Full-FT Sharded Init 无损恢复计划

> 制定日期：2026-08-31（Asia/Tokyo）
> 审核输入 commit：`dbd4f497af4a8fcc4d7f4957ac03dc004b508487`
> 当前状态：`dense_fullft_first_update_true_compute_oom_pending_official_gpu_review`
> 本轮唯一目标：在不改 FP32、dense 全参数、EMA、optimizer、global batch 64 和数据/norm 的前提下，消除 Axis params 初始化时的 replicated 峰值，并将本地 4×V100 验证到最多 10 optimizer steps。

## 0. 冻结基线与执行边界

执行前记录 root/nested commit、dirty status、GPU PID/显存和旧证据 hash。不得覆盖：

```text
data/reports/axis_libero_30k/dense_fullft_preflight/
outputs/axis_pi05/dense_fullft_preflight/
```

新代码/运行使用独立 namespace：

```text
data/reports/axis_libero_30k/dense_fullft_sharded_init_recovery/
outputs/axis_pi05/dense_fullft_sharded_init_recovery/
```

整轮保持：

| 合同 | 固定值 |
|---|---|
| init | Axis released params revision `c342608c489515efa9ba8281400dc648bf3970e1` only |
| data | `physical-intelligence/libero` native 10Hz，1693 episodes |
| norm | fresh per-dimension `QUANTILES`，SHA-256 `4c803bf5…b18f5` |
| topology | dense，`nnx.Nothing`，no LoRA |
| precision | FP32；V100 禁止 BF16/FP16 |
| optimizer | AdamW β1=.9、β2=.95、eps=1e-8、wd=1e-10，clip=1.0 |
| EMA | `0.999` |
| per-device microbatch | `1` |
| 4-GPU accumulation | `16` |
| effective global batch | exact `64` |

梯度累积保持同一 global-batch 数学目标和 FP32 dtype，但不预诺不同 reduction 次序之间 bitwise exact；数值差异必须在 R2/R4 明确报告并与预注册容差比较，不能用“无损”掩盖未测量 drift。

本轮不启动比赛 GPU，不跑 50/100/1K/30K，不跑 C20，不 export FAST/生成 ZIP/上传/提交。

## R0. 失败定位与可重现合同（CPU only）

生成 `root_cause_lock.json` 和 `ROOT_CAUSE_REPORT_ZH.md`，至少锁定：

1. 2 GPU 与 4 GPU 日志/hash/返回码；
2. traceback 均停在 `init_train_state`，optimizer update `0`；
3. loader 恢复的 params 约 `12.5 GiB`；
4. 当前 `partial_params` 的 input sharding 是 replicated `PartitionSpec()`；
5. 4 GPU 的 input/output argument bytes `26840528788`、remat floor `13433323028`、OOM allocation `56623104`；
6. 原失败目录和日志 immutable，复核前后 hash 一致。

R0 不运行大模型 GPU 计算。

## R1. 实现 sharded parameter initialization（CPU + fixture）

修正 `third_party/axis_openpi/scripts/train.py` 的初始化边界：

1. 从 `train_state_shape.params.to_pure_dict()` 派生与 `state_sharding.params` 语义一致的 partial-params sharding tree；
2. RNG 可 replicated，但 partial params 不得再用单一 `PartitionSpec()` 整树 replicated；
3. Orbax/host params 必须在进入 `init` 前按 leaf 直接 `device_put`/恢复到目标 FSDP sharding，不得先在每卡建立整份 model；
4. `jax.jit(init)` 显式使用 `(rng_replicated_sharding, partial_params_sharding_tree)` 输入，`state_sharding` 输出；
5. 不得依赖 buffer donation 来声称节省成功；若保留 donation，必须证明不可 donation 的 full-parameter warning 已消失，否则移除该无效声明；
6. resume/checkpoint 路径不能回退到 replicated full state。

实现必须是通用 FSDP 路径，不能写死 Axis 参数名称或特定 4 GPU shape。

## R2. 精度与 sharding 单测（CPU only）

新增定向测试，必须包括：

- 强制 4 个 CPU device 的 toy pytree：大矩阵沿 FSDP axis 分片，小量/scalar 按现有规则 replicated；
- 防回归：如果 partial params 又被统一 `PartitionSpec()` replicated，测试失败；
- 输入 tree、shape、dtype、leaf count 与 train-state params 完全一致；
- 从 sharded arrays 重建 global arrays，与 host loader 对应 leaf exact，过程不 cast FP32；
- GPU 前写入 `numerical_gate.json`：init/reconstruction 要求 bitwise exact；toy FP32 的 accumulated loss/gradient/update 与非分片 reference 要求 finite、max-abs `<=1e-6`、max-rel `<=1e-5`，且对 `abs(reference)>1e-6` 的元素 sign mismatch 为 `0`；
- dense trainable numel = total numel，LoRA leaf `0`，EMA/optimizer/global-batch 合同不变；
- resume/Orbax 的 abstract sharding tree 与 fresh init 相同。

运行定向测试、`py_compile`、全量 `.venv_axis_pi05/bin/pytest -q tests` 和 `git diff --check`。

## R3. 4×V100 init-only canary

仅当 4 张 GPU 都自然空闲、可独占时启动；不得 kill/reset 他人进程。不再重跑已明确无价值的 2 GPU 路径。

```text
CUDA_VISIBLE_DEVICES=0,1,2,3
per-device microbatch=1
accumulation=16
effective global batch=64
precision=float32
stop immediately after train-state init
```

要求：

- Axis leaf shape/dtype/count 与 R0 lock 一致，参数 sentinel 的 reconstructed values/hash exact；
- 实测每卡 peak/结束 VRAM、host RAM、compile time，保存 XLA memory analysis 和 sharding inventory；
- 大 params leaf 的 addressable shard 只包含对应 local slice，不得每卡 full shape；
- 旧 `26840528788` input/output argument 峰值必须下降；全量 params donation-unusable warning 不得再出现；
- init 后所有 params、Adam state、EMA finite；没有 batch、forward/backward/update。

R3 OOM 则 hard stop，不改 precision/EMA/batch，不转比赛 GPU；报告实际仍 replicated 的 leaf 与峰值再审核。

## R4. 唯一 first-update canary

只有 R3 通过才从全新 run 执行唯一次真实 optimizer update，配方与 R3 完全相同。

阶段化记录 init、data transfer、16 个 microstep 的 forward/backward、accumulation、clip、optimizer、EMA 和 checkpoint 峰值。要求：

- microstep `1…16` 的 loss/grad finite；只在第 16 个 microstep 后发生一次 clip/optimizer/scheduler/EMA update；
- optimizer step `1`、examples seen `64`，LR/warmup 语义正确；
- checkpoint `0/1` atomic save，step 1 fresh-process restore 后 params/optimizer/EMA/step exact 且 finite；
- 不使用 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 伪装配方变化；allocator 环境变量必须记录在 provenance。

若 R4 在 init 之后的真实 forward/backward OOM，记录 HLO/VRAM 阶段峰值并 hard stop。这才是可用来判断 4×V100 训练容量不足、考虑比赛 Blackwell 的证据。

## R5. 连续 10-step 数值预飞行

只有 R4 全部通过才将同一 fresh run 连续到 step `10`，不得从 R4 checkpoint 新建不同配方的 run。

- checkpoint `1/10`、fresh restore `1/10`、step1 restore后续跑的 step2 语义验证；
- 记录每步 loss、grad/clip/update norm、LR、EMA、peak VRAM、host RAM和去首次编译后的 throughput；
- 全部 finite，训练 leaf 覆盖 vision/language/action expert/action projections/time MLP；
- 生成 50/100/1K/5K/10K/30K 和单卡 Blackwell 60h 的时间预算，但不执行它们。

R5 完成后 hard stop；不在本计划内跑 C20 或长训练。

## R6. 本次执行结果（2026-09-01）

R0/R1 CPU 定位、direct-sharded init 修复与单测完成。R3 四卡 init-only 通过：51 个参数 leaf、3,353,433,872 个 FP32 参数，large replicated leaf 为 `0`，sentinel 重建 exact，params/Adam/EMA finite，step 为 `0`，未启动 data loader 或 forward/backward。旧 `26,840,528,788` bytes init argument 峰值不再出现。

R4 在同一 fresh run 完成 sharded init 和 step-0 atomic checkpoint 后，进入真实 `microbatch=1` 训练计算；首个 microbatch 的 `train_step_accumulated_host` 在每卡申请 `136.69 MiB` 时 OOM。没有产生 optimizer update、step1 metric 或 step1 checkpoint；R5 连续 10 steps、restore、C20 和 60 小时预算均未执行。

四卡 GPU 监控峰值约 `24,897 MiB/card`（JAX allocator 默认环境，未使用 `XLA_PYTHON_CLIENT_PREALLOCATE=false`），进程退出后均回到约 `5 MiB`。该结果证明当前本地 V100 的失败发生在真实训练计算峰值，而不是初始化复制；按计划状态为 `dense_fullft_first_update_true_compute_oom_pending_official_gpu_review`。不得在本计划内启用比赛 GPU、改变 recipe、运行 50/100/1K/C20、导出 FAST 或生成 ZIP。

证据：`data/reports/axis_libero_30k/dense_fullft_sharded_init_recovery/summary.json`、`ROOT_CAUSE_REPORT_ZH.md`、`r3_init_only.json`、`r4_r5_train_10.json` 及对应日志/显存监控。

## 6. 比赛 GPU 决策门禁

| 结果 | 下一决策 |
|---|---|
| R3 仍在 init OOM | 修复仍 replicated/临时 leaf，不启动比赛 GPU |
| R3 过、R4 在真实 compute OOM | 人工审核后，新计划把 Blackwell preflight 与长训练合并到同一 60h session |
| R4/R5 本地过 | 先用本地扩到 100-step + C20，再决定是否用 Blackwell 冲刺 |

官方 GPU 不用于运行已有 ZIP 或重复 scorer 环境；只保留给真实训练容量/吞吐确认和后续长训练。

## 7. Hard stop 与交付

任一发生立即停止：

- input/norm/revision/topology/FP32/EMA/global64 不一致；
- 任何 GPU 非自然空闲或外部 PID 被触碰；
- R3 init OOM 或 reconstructed sentinel 不 exact；
- R4/R5 非finite、step/accumulation/EMA 语义错误、checkpoint/restore 失败；
- 试图通过改 dtype、关 EMA、冻结层、改 global batch 或启动比赛 GPU 绕过门禁。

最终状态只能是：

```text
dense_fullft_sharded_init_recovery_failed
dense_fullft_first_update_true_compute_oom_pending_official_gpu_review
dense_fullft_local_10step_passed_pending_review
```

交付 `summary.json`、中文执行报告、代码/配置 diff、旧证据 immutability hashes、GPU 证据、测试结果和 clean commit；然后等待人工审核。
