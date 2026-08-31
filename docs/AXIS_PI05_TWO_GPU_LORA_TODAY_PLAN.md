# AXIS Pi0.5 两卡 LoRA 当日筛选与 500-Step 验证计划

> 状态：历史计划，已执行并由 R6 recovery 链取代；因配置/工具 provenance 仍引用本路径而保留原位置，不再作为活动授权。
> 日期基线：2026-08-29（Asia/Tokyo）。  
> 主线边界：canonical dense full fine-tuning 暂停为 `DEFERRED_WAITING_FOR_4GPU_OR_96GB`；不得修改、覆盖或伪装完成 `AXIS_PI05_STANDARD_LIBERO_30K_RESEARCH_PLAN.md`。  
> GPU 边界：仅使用两张明确空闲且获准使用的 GPU，默认 GPU1、GPU2；不得绑定、终止或影响 GPU0 的外部进程。  
> 数据边界：本计划只使用锁定的 canonical 10 Hz 标准 LIBERO 和 Phase 1 fresh norm；不使用 LIBERO-plus、Track 3、IPEC mixture、旧 M0/M1/M2 norm 或旧退化 checkpoint。  
> 权重边界：所有 arm 都从 `axisrobotics/Pi0.5-Axis@c342608c489515efa9ba8281400dc648bf3970e1` 的 dense params 重新开始。

## 1. 目的和成功定义

今天不追求论文级 30K 结论。目标是用两张 V100 回答三个更小、但决定后续路线的问题：

1. zero-delta LoRA、loss mask、gradient accumulation、precision 和 checkpoint restore 是否已经形成可信训练链；
2. action-expert LoRA 是否能在学习 canonical LIBERO 的同时保留 Axis step-0 能力；
3. 解冻 projections/time 或加入 PaliGemma LoRA，是否能提高容量而不重现旧 F0/F1/F2 的灾难性退化。

当日最高完成状态为：

```text
lora_candidate_500_pending_review
```

不得在同一任务中自动继续到 1K、3K、5K，不得启动 20 Hz/Track 3 适配，也不得导出或提交比赛 ZIP。

## 2. 已知基线与不得复用的失败结果

Phase 1 锁定基线：

| Panel/protocol | Axis step-0 |
|---|---:|
| C20 `execute8` | 10/20 |
| C20 `full_chunk` | 14/20 |
| B80 `execute8` | 50/80 |
| B80 `full_chunk` | 44/80 |

`execute8` 是比赛对齐主协议；`full_chunk` 只作协议敏感性诊断。不得跨协议选高分或合并结果。

以下旧结果只作失败证据，不得续训或作为新 arm 初始化：

- F0/M0、F0/M1、F0/M2@1000；
- F0/M1@3000 diagnostic；
- 旧 F1/F2 BF16/FP32 retry；
- 任何旧 LIBERO + LIBERO-plus mixture checkpoint 或其 norm。

旧实现至少存在四项已知混杂因素：

1. LoRA A/B 都使用非零 normal 初始化，step-0 不是严格 no-op；
2. frozen params 默认被转成 BF16，而 V100 禁止使用该路径；
3. `train.py` 没有计划要求的真正 gradient accumulation；
4. temporal/action-dimension mask 的分母语义不足以独立比较 valid-7D 和 padded-32D objective。

旧 dual-LoRA 约有 5000 万 trainable parameters。容量不足不是当前第一假设；在证明初始化、objective 和优化语义正确前，不扩大 rank。

## 3. 协作、Git 与安全约束

执行 Agent 不是仓库中唯一工作者。必须遵守：

- 不得 `git reset`、`git stash`、`git checkout --`、删除或覆盖其他人的改动；
- 不得使用 `git add -A`；只显式暂存本任务审核过的代码、配置、测试和小型报告；
- checkpoint、模型权重、H.264、原始数据、cache、运行日志和 token 不得进入 Git；
- 修改 `third_party/axis_openpi` 时同时维护 upstream base、effective commit/tree hash 和 root patch/lock；
- optimizer step 前，root 与 `third_party/axis_openpi` 的本任务实现必须形成可重建的 clean commit；
- 如果发现并行修改，先适配，不回退；无法安全合并时停止并报告具体文件冲突。

建议新增或维护：

```text
configs/axis_pi05/two_gpu_lora_screen.yaml
tests/test_axis_pi05_lora_screen.py
data/reports/axis_libero_30k/lora_screen/
outputs/axis_pi05/axis_libero_lora_screen/
```

不得改写 `configs/axis_pi05/standard_libero_30k.yaml` 的 dense full-FT 契约。

## 4. Phase L0：训练运行时修复与审计

### L0.1 Exact zero-delta LoRA

当前 `LoRAConfig.init_fn` 同时初始化 A/B，不满足标准 no-op LoRA。改为可独立控制 A/B：

```text
A = normal(stddev=0.01)
B = zeros
```

必须覆盖 attention Einsum 和 FFN LoRA 的所有 A/B 参数，并验证：

- dense Axis 与 zero-LoRA 在相同 observation/noise seed 下 fixed-forward 等价；
- step-0 LoRA delta norm 为零；
- step-1 时 B 至少有一个预期 leaf 获得非零 finite gradient；
- 不允许 A/B 都为零造成零梯度死锁；
- fold 前后 fixed-forward 输出在 FP32 容差内一致。

### L0.2 精确参数作用域

不得依赖宽泛 regex 猜测 action expert 与 PaliGemma。基于实际 parameter tree 建立逐 leaf 映射并断言：

- action expert：联合 Gemma tree 的 `_1` attention/MLP 分支；
- PaliGemma language backbone：无 `_1` 后缀的 attention/MLP 分支；
- vision tower 始终冻结；
- `action_in_proj`、`action_out_proj`、`time_mlp_in`、`time_mlp_out` 按 arm 显式开关；
- 每个 arm 输出 trainable/frozen leaf、numel、bytes 和占总参数比例；
- 任意未声明 trainable leaf 都是硬失败。

### L0.3 Gradient accumulation

为 TrainConfig/runner 增加显式配置：

```text
microbatch_size
gradient_accumulation_steps
effective_global_batch
```

实现要求：

1. microbatch gradient 使用 FP32 accumulator；
2. 对真实有效元素的 loss/gradient 做加权求和，最后按有效 denominator 归一化；
3. gradient clip 只在 accumulation 完成后执行一次；
4. optimizer、scheduler、internal step 只更新一次；
5. EMA 本计划关闭，但代码语义必须保证未来 EMA 也只更新一次；
6. metrics 同时记录 micro step、optimizer step、examples seen、valid action count；
7. RNG 对每个 microbatch 可重现，不因 restore 改变后续数据/noise 序列；
8. 小模型/fixture 上 accumulated update 与等价未拆分 batch 的 update 在 FP32 容差内一致。

不得通过把 `batch_size=32/64` 整批送入模型来假装 accumulation。

### L0.4 Loss mask

实现两个互斥、可审计的 objective：

```text
valid7_temporal
padded32_temporal
```

- `valid7_temporal`：只统计外部真实 7 action dims，且排除 episode 尾部 temporal padding；
- `padded32_temporal`：保留 32D Axis/OpenPI objective，但同样排除 temporal padding；
- 分母必须是实际有效的 action elements，不能对置零后的固定张量直接 `mean`；
- 分别记录 7D、25 个人工 padding dims、temporal padding 的 loss mass；
- 全 padding fixture 必须安全返回受控结果或显式拒绝，不能产生 NaN。

### L0.5 V100 precision

- V100 路径硬拒绝 BF16 frozen/trainable params；
- 第一轮使用 frozen base FP32、LoRA/heads FP32、FP32 optimizer；
- 若单卡显存探针 OOM，允许新增独立的 `frozen_fp16_trainable_fp32` 候选，但必须先完成 fixed-forward、loss、gradient-direction parity；
- precision 变化必须产生新的 config/run id，不能覆盖 FP32 arm；
- 不得为挤进显存静默关闭 loss mask、减少 action horizon 或冻结计划外参数。

### L0.6 L0 测试与门禁

至少增加：

- A-random/B-zero 初始化测试；
- step-0 dense/zero-LoRA parity；
- step-1 非零 finite gradient；
- attention/FFN fold parity；
- scope/count assertions；
- valid7/padded32/temporal denominator 测试；
- accumulation parity 与一次 optimizer-step 语义；
- restore 后 RNG/step continuity；
- V100 BF16 rejection。

随后复跑现有 Axis 定向测试、`py_compile`、lock 校验和 `git diff --check`。

只有全部通过且实现可重建，才能写入：

```text
lora_runtime_ready
```

## 5. Phase L1：四个 25-Step 单卡短探针

两张 GPU 分两轮运行，每个 GPU 同时只运行一个 arm。每个 arm 都从 released dense Axis params 重新初始化，并使用完全相同的 canonical sampling manifest、fresh norm、data order、noise fixtures 和 seed。

### L1.1 实验矩阵

| Arm | Trainable | Objective | Peak LR |
|---|---|---|---:|
| `L1_action_lora` | action expert 全 18 层 attention+FFN LoRA，rank 32 | `valid7_temporal` | `1e-6` |
| `L2_action_lora_heads` | L1 + action in/out projections + time MLP | `valid7_temporal` | `1e-6` |
| `L3_dual_lora_heads` | L2 + PaliGemma 全 18 层 attention+FFN LoRA，rank 16 | `valid7_temporal` | `3e-7` |
| `L4_dual_lora_heads_32d` | 与 L3 相同 | `padded32_temporal` | `3e-7` |

第一轮不增加 rank、不使用 LoRA dropout、不加 teacher loss、不使用 Plus mixture。teacher 只作为固定 fixture drift 指标，避免双前向使当日单卡探针复杂化。

### L1.2 公共训练配置

```text
data                       canonical standard LIBERO 10 Hz only
norm                       Phase 1 fresh canonical norm
optimizer                  AdamW
adam beta1/beta2/epsilon   0.9 / 0.95 / 1e-8
weight decay               1e-10
gradient clip              0.1
microbatch                 1
accumulation               32
effective global batch     32
warmup                     100 optimizer steps
EMA                        off
precision                  FP32 primary
checkpoints                0, 1, 10, 25
primary eval protocol      execute8
```

建议调度：

```text
wave 1: GPU1=L1, GPU2=L2
wave 2: GPU1=L3, GPU2=L4
```

启动前重新检查 GPU UUID、显存和进程；若默认 GPU1/2 不空闲，停止并报告，不自动占用 GPU0 或其他未获授权 GPU。

### L1.3 25-step 记录和停止条件

每个 optimizer step 至少记录：

- total/7D/25D/temporal loss；
- pre/post clip gradient norm；
- LoRA、PaliGemma LoRA、action-expert LoRA、projection、time MLP 分组 gradient/update norm；
- parameter/update/delta norm；
- examples seen、LR、throughput、data wait、峰值显存；
- nonfinite leaf count 和第一个 nonfinite path。

step 25 必须在新进程 restore，并重跑 fixed-forward。以下任一情况立即淘汰该 arm：

- NaN/Inf；
- step-0 不等价；
- frozen leaf 发生更新；
- trainable leaf 与声明不符；
- checkpoint、optimizer 或 RNG continuity 失败；
- action non-finite；
- 显存 OOM 后通过未经审计的隐式配置变化继续。

完成后写入：

```text
lora_screen_25_complete
```

## 6. Phase L2：最多两个候选运行到 100 Steps

最多保留两个 arm。选择顺序不是 training loss 最低，而是：

1. 所有数值与 restore 门禁通过；
2. step-0 exact；
3. frozen scope 无污染；
4. fixed teacher/action drift 较小；
5. 有效 action loss 有下降趋势；
6. 吞吐和显存可持续。

候选从自己的 step 25 连续到 100，保存 step 50/100，并在 step 100 运行 C20 `execute8` paired rollout。不得用 `full_chunk` 的 14/20 基线替换主协议。

晋级门禁：

- C20 `execute8` success 至少 9/20；
- 相对 Axis 同协议基线的 paired success delta 不低于 -0.05；
- stall 不得出现系统性增加；
- action、loss、gradient、update 全 finite；
- step 100 新进程 restore 通过；
- 生成 20/20 H.264、episodes.csv、tasks.csv 和 failure tags。

如果所有 arm 都失败：

```text
lora_screen_stopped_no_noninferior_arm
```

此时停止；不得扩大 rank、提高 LR 或盲目继续到 500。

如果两个 arm 分数相同，优先 trainable 参数更少、teacher drift 更小的 arm。完成后写入：

```text
lora_screen_100_complete
```

## 7. Phase L3：最佳配置两卡重新训练到 500 Steps

最佳配置必须从 released Axis dense params 和 zero-delta LoRA **重新初始化**。不得从 single-GPU global-batch-32 screen checkpoint 接着训练，因为正式验证需要固定 global batch 64。

### L3.1 两卡配置

```text
visible GPUs                两张已授权空闲卡，默认 1,2
parallelism                 data parallel；LoRA 不使用 FSDP
microbatch                  1 per GPU
accumulation                32
effective global batch      64
precision                   L1/L2 通过的锁定 precision
EMA                         off
checkpoints                 0, 25, 100, 250, 500
```

必须先做两卡 step-1 与单卡等效 global batch fixture 的 update parity，再启动正式 500-step run。

### L3.2 训练与评估

- step 25/100/250/500：finite、scope、gradient/update/delta、显存和吞吐报告；
- step 100/250：C20 `execute8`；
- step 500：B80 `execute8` 主评估；
- step 500：B80 `full_chunk` 只作同协议 step-0 diagnostic；
- step 100/250/500：新进程 restore；
- step 500：raw LoRA 与 folded dense fixed-forward parity；
- 生成 step-0/100/250/500 paired H.264 failure gallery。

不得根据 C20/B80 临时修改 seed、init state、max steps、action execution protocol 或 norm。

### L3.3 500-step 门禁

主门禁基于 B80 `execute8`：

```text
Axis step-0 baseline       50/80
minimum accepted success  46/80
paired delta lower bound  -0.05
```

同时要求：

- stall/timeout 相对同协议 step-0 不上升超过 0.05；
- 没有 action non-finite、runtime error 或 infrastructure error；
- checkpoint、optimizer state、norm hash、data hash 和 protocol hash 完整；
- LoRA fold parity 通过；
- 全部 80 个 episode 有 CSV 和 H.264 coverage。

通过后只写入：

```text
lora_candidate_500_pending_review
```

失败则写入：

```text
lora_screen_stopped_at_500_regression
```

无论通过或失败，都不得自动进入 1K。

## 8. 产物与报告

每个 arm 使用独立命名空间：

```text
outputs/axis_pi05/axis_libero_lora_screen/<arm>/<run_id>/
data/reports/axis_libero_30k/lora_screen/<arm>/<run_id>/
```

最低产物：

```text
resolved_config.yaml
source_manifest.json
norm_manifest.json
precision_report.json
params_scope.json
zero_delta_report.json
loss_mask_report.json
accumulation_parity_report.json
training_metrics.jsonl
checkpoint_manifest.json
restore_report.json
fixed_forward_report.json
summary.json
tasks.csv
episodes.csv
failure_gallery/index.html
```

全局汇总：

```text
data/reports/axis_libero_30k/lora_screen/summary.json
data/reports/axis_libero_30k/lora_screen/LORA_SCREEN_REPORT_ZH.md
```

汇总必须明确区分：

- `observed`：实际 loss、显存、restore、rollout；
- `inference`：对 scope/objective 的解释；
- `unverified_hypothesis`：是否可能追平 full-FT。

在没有 matched full-FT checkpoint 前，不得宣称 LoRA 已追平 full-FT。只能报告它是否保持/改善 Axis step-0，以及相对计算和显存的效率。

## 9. 当日禁止事项

- 不启动 canonical dense full-FT optimizer step；
- 不使用 GPU0，不终止任何外部进程；
- 不使用 BF16 on V100；
- 不加载旧 Axis norm、mixture norm 或退化 checkpoint；
- 不训练 LIBERO-plus、Track 3 或 synthetic reverse 数据；
- 不扩大 rank、不升到旧 `1e-5/5e-5` LR；
- 不自动进入 1K/3K/5K；
- 不导出 FAST submission，不运行官方提交，不消耗比赛 GPU；
- 不因 loss 下降就宣称行为提升；
- 不删除失败 checkpoint、日志或 H.264 证据。

## 10. 给执行 Agent 的完整任务指令

```text
请完整执行：
/home/huang/code/PARC2026_final/docs/AXIS_PI05_TWO_GPU_LORA_TODAY_PLAN.md

你不是仓库中唯一的工作者，不得 reset、stash、checkout、删除、覆盖或回退其他人的修改。canonical dense 30K full-FT 已暂时搁置；不得修改其配置或启动 optimizer step。只使用两张明确获准且空闲的 GPU，默认 GPU1/2；不得触碰 GPU0。

先完成 Phase L0：A-random/B-zero 的 exact zero-delta LoRA、精确 scope audit、真正的 gradient accumulation、valid7/padded32 temporal loss denominator、V100 FP32 precision 和 restore/parity 测试。实现未形成可重建 clean commit 前不得训练。

随后按 L1-L4 四臂矩阵运行 25-step probes，最多两个进入 100-step C20 execute8 paired gate。只有非劣 arm 才能从 Axis step-0 重新初始化，并用两卡 effective global batch 64 运行到 500；step 500 完成 B80 execute8/full_chunk、restore、fold parity、CSV、H.264 和 failure gallery 后必须停止，输出 lora_candidate_500_pending_review 或明确的 stopped 状态。不得自动进入 1K、Plus、Track 3、FAST export 或官方提交。
```
