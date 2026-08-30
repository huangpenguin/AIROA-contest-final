# AXIS Pi0.5 LoRA 两卡 NaN 定位、单卡兜底与 500-Step 恢复执行计划

> 状态：已执行至 R6，已获得 fresh 500-step checkpoint；最终状态为 `lora_candidate_500_pending_review`，详见专项目录报告。
> 日期基线：2026-08-29（Asia/Tokyo）。  
> 上游计划：`docs/AXIS_PI05_TWO_GPU_LORA_TODAY_PLAN.md`。  
> 当前事实状态：历史失败保留为 `lora_two_gpu_blocked_nonfinite_before_first_update`；修复后 R6 数值与 B80 主行为门禁通过，当前候选为 `lora_candidate_500_pending_review`。旧状态名 `lora_screen_stopped_at_500_regression` 仅作为历史记录保留，不再作为技术结论。
> GPU 边界：只允许使用明确空闲且获准使用的 GPU1/GPU2；不得绑定、终止、重置或影响 GPU0 的外部进程。  
> 数据边界：只使用 canonical `physical-intelligence/libero` 原生 10 Hz 数据和 Phase 1 fresh norm；本计划不训练 LIBERO-plus、20 Hz、Track 3 或合成数据。  
> 执行边界：先定位并修复运行时问题，再恢复 500-step；禁止用反复盲跑代替诊断。

## 0. 决策摘要

当前结果应拆成两个互不混淆的结论：

1. `L2_action_lora_heads@100` 是有效的单卡初筛候选：C20 `execute8` 为 15/20，相对 Axis step-0 的 10/20 为 paired `+0.25`；该 checkpoint 继续保留。
2. 两卡正式候选没有完成任何有效 optimizer update。retry2 在第一个累计 step 的 `micro_index=1`、optimizer update 前出现 non-finite `loss_sum`；这不是“500-step 行为回归”。

三次相关运行的代码快照不同：

| run | root commit | OpenPI commit | 结果 |
|---|---|---|---|
| first formal | `208aa4aede242a24ec32bf2da9740eea8782e088` | `8b439402bb588edd8579d2dcaa003eefacbf1710` | 首个累计 update non-finite |
| one-step diagnostic | `da031ae9d129b4082deada31cf0d183b03cb4e27` | `c21f6fe97e98880e8a141d6b3049fe1c8feac258` | 32 microbatches 与 step 1 finite |
| retry2 formal | `f5bd9236882ff43692d834db20acf7a28038118a` | `fe205f14eeaa986fb505636f15b2a5aac9b28785` | `micro_index=1` 的 `loss_sum` non-finite |

因此本计划不预设根因。需要依次排除：

- 特定数据样本或 transform；
- 特定 diffusion time/noise RNG；
- GPU1/GPU2 单卡硬件或 kernel 差异；
- 双卡 data-parallel/sharding/XLA 路径；
- dataloader 顺序与 worker nondeterminism；
- step-0 异步 Orbax 写盘与编译/训练并发；
- 当前尚未观察到的模型数值问题。

本计划有两条后续出口：

```text
精确重放与双卡门禁恢复
  → fresh Axis + zero-delta L2，两卡 global batch 64 → 500 steps

双卡门禁仍失败、但单卡门禁稳定
  → fresh Axis + zero-delta L2，单卡 accumulation 64 → global batch 64 → 500 steps
```

两条出口都不得从现有单卡 100-step checkpoint 接续正式 500-step；L2@100 只用于扩大评估面板和筛选结论确认。

## 1. 固定输入、权重与协议

### 1.1 权重

- 正式训练初始化：`axisrobotics/Pi0.5-Axis@c342608c489515efa9ba8281400dc648bf3970e1` dense params。
- LoRA 初始化：A-random/B-zero，step-0 exact zero-delta。
- 参数作用域：`L2_action_lora_heads`，即 action-expert LoRA + `action_in_proj` + `action_out_proj` + time MLP。
- 不加载旧 M0/M1/M2、F0/F1/F2、mixture 或 LIBERO-plus checkpoint。
- 不使用旧 Axis normalization。

### 1.2 数据与 norm

```text
source repo/revision       physical-intelligence/libero@a4336d589d589045d1c56423ffdf3b88a0e19b1f
payload manifest sha256    8ede620026a9196a9d453508dfe2d4820e684d98cda1a5bc706cef68d1583cc3
sampling manifest sha256   41715b29010fc7067c8fff8fd6fb0939735966bdc9afcf35ee1b3a0fc11b45b1
norm stats sha256          4c803bf59c52677c36b6b420c614d8a7742e0847708443863e13e67cb80b18f5
source fps                 native 10 Hz
resampling                 none
objective                  valid7_temporal
```

不得因为定位 NaN 而重新计算 norm、换数据源、跳过特殊样本、降低 action horizon 或更改 loss mask。

### 1.3 训练语义

```text
precision                  FP32；V100 禁止 BF16
optimizer                  AdamW
peak LR                    1e-6
gradient clip              0.1，accumulation 完成后只执行一次
EMA                        off
effective global batch     64
two-GPU layout             microbatch 1/GPU × accumulation 32
single-GPU fallback        microbatch 1 × accumulation 64
primary eval               execute8
diagnostic eval            full_chunk，不参与选优
```

禁止为绕过 non-finite 临时改变 LR、batch、norm、precision、objective 或样本顺序后仍使用同一个 run id。

## 2. Git、并行工作与安全门禁

执行 Agent 不是仓库中唯一工作者，必须遵守：

- 不得 `git reset`、`git stash`、`git checkout --`、删除或覆盖其他人的修改；
- 不得使用 `git add -A`；只能显式暂存本任务审核过的文件；
- 不得向官方 upstream remote push；只有在用户已经配置并授权私有 remote 时才能 push；
- checkpoint、原始/重放 batch、H.264、训练日志、cache 和 token 不得进入 Git；
- 修改 nested `third_party/axis_openpi` 时，先形成 nested clean commit，再更新 root 中的 lock/patch/tree provenance；
- 每个 GPU 实验必须记录当时的 root commit、nested OpenPI commit/tree、两边 dirty status；任一 dirty 状态都禁止启动 optimizer step；
- 运行期间不得继续修改同一 checkout。若必须改代码，停止所有相关进程，形成新 commit 和新 run id 后重启；
- GPU 启动前记录 UUID、driver、CUDA/JAX/XLA 版本和当前进程；GPU0 只读检查，不得设置到 `CUDA_VISIBLE_DEVICES`。

若当前没有可用私有 remote，clean immutable commit 足以执行本计划，但必须额外生成不进 Git 的本地 commit/patch 备份清单。不得因备份问题把代码推到官方仓库。

## 3. Phase R0：纠正状态与冻结证据

### R0.1 保留历史证据

不得删除或覆盖以下目录：

```text
data/reports/axis_libero_30k/lora_screen/L2_action_lora_heads/
  l2_action_lora_heads_20260829_2gpu500/
  l2_action_lora_heads_20260829_2gpu_diag/
  l2_action_lora_heads_20260829_2gpu500_retry2/

outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/
  对应三个 run 目录
```

### R0.2 报告语义修正

更新全局和中文报告，但不得改写原始 metrics/log：

- 全局状态改为 `lora_two_gpu_blocked_nonfinite_before_first_update`；
- `reason` 改为“第一个累计 optimizer step 内发生 non-finite，未完成有效 optimizer update”；
- 将 `L2@100` 标记为 `preliminary_screen_winner`，不得置为无效；
- 明确 first formal、diagnostic、retry2 使用不同 root/OpenPI commits，不能据此宣称同一运行时下随机复现；
- `norm_manifest.json` 明确标为 norm artifact 生成 provenance；`source_manifest.json` 才是本次 run runtime provenance；
- 当前定向测试基线记录为实际重新运行得到的结果，不硬编码旧的 `17 passed`。

机器可读状态至少包含：

```json
{
  "optimizer_updates_completed": 0,
  "failure_stage": "first_accumulated_step_before_valid_optimizer_update",
  "failed_micro_index": 1,
  "root_commit": "...",
  "openpi_commit": "...",
  "same_runtime_reproduction_established": false
}
```

完成状态：

```text
lora_nan_evidence_frozen
```

## 4. Phase R1：建立可精确重放的 microbatch 审计

这是再次使用 GPU 训练前的强制实现阶段。

### R1.1 样本身份

扩展 canonical dataset/dataloader，使训练 batch 携带仅供审计使用的 metadata，至少包括：

- source repo/revision；
- sampling manifest row；
- task、episode、frame index；
- action window 的起止位置与 temporal padding；
- dataloader epoch、global draw index、worker id；
- rank/process/device assignment；
- transform/config hash。

metadata 不得进入模型参数树或改变训练 tensor。正常训练中可关闭详细输出，但 finite audit 开启时必须保存。

### R1.2 RNG 身份

对每个 optimizer step 和 microbatch 保存：

- `state.step`；
- base RNG、step RNG、fold-in micro index 后 RNG 的稳定十六进制表示或 hash；
- diffusion time/noise RNG；
- dataloader/sampler seed；
- JAX process/device count 与 device order。

restore 后必须证明下一 microbatch 的 sample IDs 和 RNG 与无中断路径一致。

为保证单卡 `1 × 64` 和双卡 `1/GPU × 32` 可以比较，replay-only 路径必须支持按 sample ID 固定的 diffusion time/noise fixture，或接受已经捕获的 time/noise tensor override。该 override 只用于诊断，不得改变正常训练的随机采样实现。不能只给不同 batch 分组传入同一个 base RNG 后声称数值输入相同。

### R1.3 输入与梯度 finite 审计

当任意 loss/component/gradient non-finite 时，必须在抛错前写出：

- 第一个 non-finite leaf 的完整 parameter path；
- loss、valid denominator、time/noise 的 finite/min/max/mean；
- image、token、mask、position、state、action、temporal mask 的 shape/dtype/finite/min/max；
- 每个物理设备对应的 sample IDs；
- microbatch trace 与 resolved config/source hashes；
- 可重放的 transformed batch tree。

建议产物：

```text
data/reports/axis_libero_30k/lora_nan_replay/<run_id>/
  environment.json
  source_manifest.json
  resolved_config.yaml
  microbatch_trace.jsonl
  failure.json
  input_stats.json
  replay_manifest.json

outputs/axis_pi05/axis_libero_lora_nan_replay/<run_id>/
  replay_batch.npz 或等价 tensor artifact
```

大体积 replay tensor 只放 outputs，不进入 Git。`replay_manifest.json` 保存各 tensor 的 shape、dtype 和 SHA-256。

### R1.4 checkpoint 并发开关

为诊断路径提供显式配置：

```text
save_step0
checkpoint_async
wait_for_step0_save_before_training
```

默认诊断设置：

```text
save_step0=false
checkpoint_async=false
wait_for_step0_save_before_training=true
```

正式需要 step-0 checkpoint 时，必须等待写盘和 finalize 完成，再开始 compilation/optimizer step。不得留下 `.orbax-checkpoint-tmp-*` 后仍把 checkpoint 记为成功。

完成状态：

```text
lora_nan_replay_instrumented
```

## 5. Phase R2：CPU/轻量 fixture 与静态门禁

新增或扩展测试，至少覆盖：

1. 同 seed 两次 loader 产生相同 sample-ID 顺序；
2. replay artifact 重建后的 tensor tree hash 完全一致；
3. sample metadata 不改变模型输入或 gradient；
4. micro RNG 在相同 step/index 下完全一致；
5. restore 后 sample/RNG continuity；
6. non-finite fixture 能在抛错前生成完整 failure/replay manifest；
7. 单卡/双卡 metadata 中 device assignment 正确；
8. async/sync checkpoint 开关可审计且不会静默回退；
9. norm artifact provenance 与 runtime provenance 字段不会混用；
10. 现有 zero-delta、scope、loss mask、accumulation parity 和 V100 BF16 rejection 测试继续通过。

静态门禁：

```text
targeted pytest             all pass
py_compile                  pass
OpenPI lock/patch/tree      pass
git diff --check            pass
root status                 clean
nested OpenPI status        clean
```

任何一项失败都不得进入 Phase R3。

## 6. Phase R3：固定 batch/RNG 的 GPU 判因矩阵

### R3.1 基本原则

- 所有矩阵单元使用同一组 64 个样本、相同 resolved config、相同初始 params 和可对应的 noise/time seeds；
- 全部运行来自同一 root/OpenPI clean commits；
- 每个矩阵单元使用独立 run id 和 fresh process；
- 先只做 forward/backward accumulation，不执行 optimizer update；通过后才允许做一次 update；
- 禁止在矩阵中训练超过 10 optimizer steps；
- 默认关闭 step-0 checkpoint 和 async checkpoint。

### R3.2 矩阵

| ID | 设备 | microbatch/accumulation | checkpoint | 目的 |
|---|---|---|---|---|
| `S1_FB` | GPU1 | 1 × 64 | off | 单卡 GPU1 fixed-batch forward/backward |
| `S2_FB` | GPU2 | 1 × 64 | off | 单卡 GPU2 fixed-batch forward/backward |
| `D12_FB` | GPU1+2 | 1/GPU × 32 | off | 双卡 fixed-batch forward/backward |
| `S1_U1` | GPU1 | 1 × 64 | off | 单卡一次 optimizer update |
| `S2_U1` | GPU2 | 1 × 64 | off | 单卡一次 optimizer update |
| `D12_U1` | GPU1+2 | 1/GPU × 32 | off | 双卡一次 optimizer update |
| `D12_SYNC_U1` | GPU1+2 | 1/GPU × 32 | sync step-0 | 排除同步 step-0 save 影响 |
| `D12_ASYNC_WAIT_U1` | GPU1+2 | 1/GPU × 32 | async 且显式 wait | 测试异步 API 但不与训练并发 |
| `D12_ASYNC_OVERLAP_U1` | GPU1+2 | 1/GPU × 32 | async，不 wait | 最后受控重现旧写盘/编译并发，只作判因 |

除 `D12_ASYNC_OVERLAP_U1` 外，每个单元至少重复三次；overlap 单元只在此前全部 finite、磁盘空间充足且临时目录可隔离时运行，先执行一次，复现异常后立即停止。所有重复必须复用同一 replay manifest 和按样本固定的 time/noise tensor，不得重新 shuffle。记录：

- loss sum/denominator/loss；
- 每个 microbatch 的 finite 状态；
- pre/post clip grad norm；
- update/parameter norm；
- 首个 non-finite path；
- tensor/gradient/update tree hash 或数值容差报告；
- GPU UUID、峰值显存、XLA/JAX 版本；
- checkpoint finalize 与临时目录状态。

### R3.3 判因规则

| 观察 | 归类 | 下一步 |
|---|---|---|
| 同一 sample/RNG 在 S1/S2/D12 都失败 | 数据、transform、RNG 或模型数值问题 | 从输入到第一处 non-finite activation 分层定位 |
| 只在某一物理 GPU 失败 | GPU/kernel/硬件相关 | 记录 UUID，交换设备顺序并做硬件健康检查 |
| 单卡都通过、双卡失败 | data-parallel/sharding/XLA 问题 | 比较 per-device batch、replication、collective 与 reduction |
| sync 与 async-wait 通过、async-overlap 失败 | Orbax/资源并发问题 | 正式训练禁用 overlap，step-0 必须 finalize 后再训练 |
| fixed replay 全通过、fresh loader 才失败 | loader 顺序或未记录随机源 | 扩大 sample/RNG trace，修复 sampler/worker 确定性 |
| 相同 commits/replay 下结果不稳定 | 运行时 nondeterminism | 保存编译 flags/HLO 环境，做有界 10 次重复后停止 |

不得在没有对应证据时宣称是 XLA、Orbax 或硬件问题。

### R3.4 退出门禁

双卡恢复需要同时满足：

- 同一 fixed replay 的 D12 forward/backward 三次全部 finite；
- D12 optimizer step 三次全部 finite；
- sync step-0 checkpoint 后 optimizer step 三次全部 finite；
- 单卡和双卡的 loss/gradient/update 差异符合预先写入报告的 FP32 容差；
- 没有 `.orbax-checkpoint-tmp-*` 冒充完整 checkpoint；
- replay artifact 可在 fresh process 精确恢复。

通过后状态：

```text
lora_two_gpu_runtime_replay_gate_passed
```

失败且已定位根因时，修复后必须用新 commit/new run id 重新执行整个 R2/R3；不得只重跑失败单元。

## 7. Phase R4：10/100-Step 新鲜训练稳定性门禁

只有 R3 通过后才可恢复 fresh loader 和 shuffle。

### R4.1 双卡优先路径

从 released Axis params + zero-delta L2 fresh 初始化：

```text
GPU                        1,2
microbatch                 1/GPU
accumulation               32
effective global batch     64
checkpoint                 synchronous or fully awaited
steps                      0,1,10,100
```

先做三个独立 fresh 10-step run。三次必须：

- loss/gradient/update/action 全 finite；
- 完成 10 个真实 optimizer updates；
- step 1/10 checkpoint finalize、restore 和 fixed-forward 通过；
- root/OpenPI commits 完全一致；
- 每次前 32 个 microbatch 的 sample-ID/RNG trace 完整；
- frozen scope、zero-delta、norm/data/config hashes 正确。

三次 10-step 全部通过后，才运行一次 fresh 100-step；不得从 10-step 诊断 run 接续。100-step 通过后状态：

```text
lora_two_gpu_runtime_recovered
```

任一 10-step fresh run non-finite：立即停止双卡训练，不继续 100/500。

### R4.2 单卡兜底门禁

若 R3 证明双卡路径仍不可靠，而 S1 或 S2 稳定，则允许在稳定的单卡上验证等效 global batch 64：

```text
microbatch                 1
accumulation               64
effective global batch     64
precision                  FP32
steps                      0,1,10,100
```

单卡必须运行三个 fresh 10-step 和一个 fresh 100-step，门禁与双卡相同。不得把单卡 accumulation 32 的旧 100-step screen 当成 global-batch-64 正式门禁。

通过后状态：

```text
lora_single_gpu_batch64_runtime_recovered
```

若单卡也失败，则停止全部 L2 正式训练，状态写为：

```text
lora_nan_localization_blocked_no_stable_runtime
```

## 8. Phase R5：现有 L2@100 的 B80 泛化复核

该阶段只评估已存在的单卡 screen checkpoint，不更新参数，可在 R2 完成并形成 clean commit 后执行；不得把其结果冒充正式 global-batch-64 500-step 结果。

固定：

```text
checkpoint                 L2_action_lora_heads 单卡 step 100
panel                      Phase 1 锁定 B80 manifest
primary                    execute8
baseline                   Axis step-0 B80 execute8 = 50/80
diagnostic                 full_chunk，仅在资源允许时运行
coverage                   80/80 CSV + H.264 + failure tags
```

报告 paired wins/losses/ties、success delta、collision、timeout、stall 和 task-level 分布。不得只报告总体成功率。

解释规则：

- `execute8 >= 46/80`：L2@100 可继续保留为非劣 preliminary candidate；
- `execute8 < 46/80`：标记 `screen_c20_not_confirmed_on_b80`，但该结果不证明 LoRA 运行时 NaN 的原因；
- full_chunk 不参与上述判断。

R5 与 R3/R4 的训练判断相互独立。B80 好分不能清除数值门禁，B80 低分也不能解释两卡首步 non-finite。

## 9. Phase R6：恢复 fresh 500-Step 正式候选

只有以下二者之一成立才允许进入：

```text
lora_two_gpu_runtime_recovered
或
lora_single_gpu_batch64_runtime_recovered
```

### R6.1 路径选择

1. 双卡门禁通过：优先 GPU1+2、microbatch 1/GPU、accumulation 32。
2. 双卡门禁失败但单卡门禁通过：使用稳定单卡、microbatch 1、accumulation 64。
3. 两条都通过时只运行双卡正式臂，单卡保留为诊断，不重复消耗资源。

正式 run 必须从 released Axis dense params 和 zero-delta L2 fresh 初始化，不得从 L2@100 或任意 R3/R4 checkpoint 续训。

### R6.2 checkpoint 与观测

```text
target optimizer steps     500
checkpoints                0,1,10,25,50,100,250,500
restore checks             1,10,100,250,500
C20 execute8               100,250
B80 execute8               500
B80 full_chunk             500 diagnostic only
```

step-0 checkpoint 必须先完整 finalize，再开始 optimizer。所有 checkpoint 必须记录 params、optimizer state、internal step、config/data/norm/source hashes，并在 fresh process restore。

训练过程中每个 optimizer step至少记录：

- loss sum、valid denominator、normalized loss；
- pre/post clip gradient norm；
- LoRA、projection、time MLP 的 gradient/update norm；
- parameter/update/delta norm；
- examples seen、micro steps、optimizer step；
- data wait、throughput、峰值显存；
- non-finite leaf count 和首个 path；
- sample/RNG trace 的可追溯索引。

任一 non-finite、scope 污染、restore 失败、manifest/hash 变化或 checkpoint 未 finalize，立即停止且不得自动修改超参数重试。

### R6.3 500-step 行为门禁

主门禁仍是 B80 `execute8`：

```text
Axis step-0 baseline       50/80
minimum accepted success  46/80
paired delta lower bound  -0.05
```

同时要求：

- 80/80 episode 有 CSV、H.264 和 failure tag；
- stall/timeout 相对同协议基线上升不超过 0.05；
- action、loss、gradient、update 全 finite；
- step 500 restore 与 raw-LoRA/folded-dense fixed-forward parity 通过；
- runtime provenance 明确指出单卡或双卡路径，不混写。

通过：

```text
lora_candidate_500_pending_review
```

数值稳定但行为未达门禁：

```text
lora_candidate_500_behavior_noninferiority_failed
```

数值中止：

```text
lora_candidate_500_runtime_nonfinite
```

完成后必须停止，等待人工审核；不得自动进入 1K、5K、30K、Plus、Track 3 或 FAST export。

## 9.4 本次执行结果

本次已完成 R0-R6 的数值主线与 R6 主行为门禁：

- R3 fixed replay `9/9` finite；U1 矩阵 `16/16` finite；sync/async checkpoint 变体均 finalized。
- R4 三个独立 fresh-10 与一个独立 fresh-100 双卡 run 全部通过。
- R6 `r6_dual_l2_fresh500_r1` 完成 `500/500` optimizer updates，metrics `500`、microbatch trace `16000`，checkpoint `0,1,10,25,50,100,250,500` 完整；restore `1,10,100,250,500` 通过。
- B80 `execute8` 完成 `80/80`，step-500 成功 `48/80`，step-0 基线 `50/80`；paired wins/losses `9/11`，delta `-0.025`，满足 `46/80` 与 `-0.05` 门禁。
- dual V100 fixed-forward 保留 repeatability warning：finite，但重复输出未达到 bitwise/记录的 FP32 tolerance；该项交由人工 review，不静默改成 deterministic pass。
- B80 `full_chunk` 500 未执行（diagnostic-only）；不得将其写成已通过。

机器可读与中文最终报告：

```text
data/reports/axis_libero_30k/lora_nan_replay/summary.json
data/reports/axis_libero_30k/lora_nan_replay/NAN_LOCALIZATION_REPORT_ZH.md
```

因此执行状态为 `lora_candidate_500_pending_review`，到此停止，不自动进入后续训练、导出或提交。

## 10. 产物目录

新增专项目录：

```text
data/reports/axis_libero_30k/lora_nan_replay/
  summary.json
  NAN_LOCALIZATION_REPORT_ZH.md
  <run_id>/...

outputs/axis_pi05/axis_libero_lora_nan_replay/
  <run_id>/replay tensor artifacts
```

正式训练继续使用独立命名空间：

```text
outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/<new_run_id>/
data/reports/axis_libero_30k/lora_screen/L2_action_lora_heads/<new_run_id>/
```

全局汇总需要更新：

```text
data/reports/axis_libero_30k/lora_screen/summary.json
data/reports/axis_libero_30k/lora_screen/LORA_SCREEN_REPORT_ZH.md
data/reports/axis_libero_30k/THREAD_REPORT_ZH.md
```

每份报告都必须区分：

- `observed`：实际运行、数值、checkpoint、rollout；
- `inference`：基于矩阵得到的受限解释；
- `unverified_hypothesis`：尚未被同样本/同 RNG 对照验证的猜测。

## 11. 全局停止条件与禁止事项

出现以下任一情况立即停止当前 phase：

- GPU0 被纳入 visible devices 或外部 PID 受到影响；
- root/nested OpenPI dirty 或运行中代码提交发生变化；
- 数据、norm、sampling、config hash 不一致；
- 同 run id 下超参数、设备拓扑或代码发生变化；
- replay manifest 不足以重建失败输入；
- non-finite 后代码自动跳过样本、降低 batch、改变 LR/norm/precision 继续；
- Orbax 临时 checkpoint 被记录为正式 checkpoint；
- 测试、lock、restore 或 fixed-forward 任一失败。

本计划明确禁止：

- 盲目再次启动两卡 500-step；
- 将首步 non-finite 称为 500-step regression；
- 删除/覆盖旧失败证据；
- 使用 BF16 on V100；
- 触碰 GPU0 或终止外部进程；
- 换成 Plus/Track 3/20 Hz/旧 mixture 数据来绕过问题；
- 跳过失败 sample；
- 扩大 LoRA rank、提高 LR 或临时解冻更多参数；
- 从 L2@100 接续正式 global-batch-64 500-step；
- 自动进入 1K/5K/30K、比赛 GPU、FAST 导出或提交。

## 12. 给执行 Agent 的完整任务指令

```text
请完整执行：
/home/huang/code/PARC2026_final/docs/AXIS_PI05_LORA_NAN_LOCALIZATION_AND_RECOVERY_PLAN.md

你不是仓库中唯一的工作者。不得 reset、stash、checkout、删除、覆盖或回退其他人的修改；不得使用 git add -A。只允许使用明确空闲且获准的 GPU1/GPU2，不得触碰 GPU0 的外部进程。不得向官方 upstream remote push。

先完成 R0-R2：纠正“500 regression”的状态语义，保留 L2@100，建立 sample ID、RNG、device assignment、输入统计和 replay tensor 的精确 microbatch 重放能力，并通过全部测试、lock、clean-commit 门禁。然后执行 R3 固定 64 样本/固定 RNG 的单卡 GPU1、单卡 GPU2、双卡、sync/async checkpoint 判因矩阵。所有矩阵运行必须来自完全相同的 root/OpenPI commits，并使用独立 run id。

只有双卡 replay、三次 fresh 10-step 和 fresh 100-step 全部 finite/restore-passed，才允许从 released Axis params + zero-delta L2 重新启动两卡 global-batch-64 的 500-step。若双卡仍失败但单卡 accumulation-64 的三次 10-step 与 fresh 100-step 门禁通过，允许走单卡 global-batch-64 500-step 兜底。正式 500-step 不得从旧 L2@100 续训。

现有 L2@100 可独立运行锁定 B80 execute8 paired 复核，但结果不得清除数值门禁或冒充 500-step。任意 non-finite、provenance/hash/restore/lock 失败都立即停止；不得修改 LR、batch、norm、precision、objective 或跳过样本后继续。最终最多产出 lora_candidate_500_pending_review，并停止等待人工审核；不得自动进入 1K、5K、30K、LIBERO-plus、Track 3、FAST export、官方比赛 GPU 或提交。
```
