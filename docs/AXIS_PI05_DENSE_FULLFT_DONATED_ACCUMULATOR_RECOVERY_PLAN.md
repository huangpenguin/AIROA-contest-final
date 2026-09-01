# AXIS Pi0.5 Dense Full-FT Donated Gradient Accumulator 修复计划

> 制定日期：2026-09-02（Asia/Tokyo）
> 审核输入：root `ecff24f`，nested OpenPI `e31c2bf`
> 当前状态：`dense_fullft_donated_accumulator_remediation_ready`
> 唯一目标：保持FP32 dense full-FT、顺序梯度求和与global batch 64完全不变，让第二份及后续梯度归并的输出真实复用旧accumulator buffer，并在本地4×V100完成step1/step10门禁。

## 0. 审核锁与精确失败边界

上一轮通过项继续有效：M0 raw evidence修正、M1移除长期存活的gradient审计字典、step0 restore/sentinel、CPU回归、GPU/外部动作边界。M2按hard gate停止，没有optimizer update或伪造step1。

M2精确阶段为：

```text
microbatch compute index 0 complete
accumulation/release index 0 complete
microbatch compute index 1 complete
plain _add_gradients(old_accumulator, current_grads)
→ new output buffer allocation 72MiB shortfall / OOM
```

`_add_gradients`当前只是逐叶`x+y`，没有JIT donation/output alias。index1 compute完成时每卡约`26.42GB` live；普通加法要同时保留旧accumulator、current grads和新accumulator output，无法在`0.95` pool中完成。这仍不是单份forward/backward装不下的证据，也不授权Blackwell。

## 1. 不可变合同

| 字段 | 固定值 |
|---|---|
| initialization | Axis released params revision `c342608c489515efa9ba8281400dc648bf3970e1` |
| data/norm | `physical-intelligence/libero@a4336d…` + fresh QUANTILES `4c803bf5…b18f5` |
| topology/precision | dense full model / FP32 |
| optimizer/EMA | AdamW fixed recipe / `0.999` |
| microbatch/devices | `1` per device / local 4×V100 |
| accumulation/global batch | ordered `16` / exact `64` |
| allocator | preallocate=true，fraction `0.95` |
| sharding/remat/RNG/loss | nested OpenPI `e31c2bf` semantics保持不变 |

不得改变求和顺序、dtype、loss reduction、batch、EMA、optimizer、freeze scope、data/norm、action horizon、remat或sharding来换取通过。不得启动官方GPU、Blackwell、C20、100/1K/长训、FAST、ZIP、上传或提交。

## D0. 证据闭合（CPU only）

1. summary顶层补齐`optimizer_updates=0`、`next_action`、M1结果、git/test provenance；
2. machine handoff顶层更新为`ecff24f/e31c2bf`，不得只更新execution子树；
3. 将M2字段拆成：`compute_completed=[0,1]`、`accumulation_completed=[0]`、`failed_accumulation_index=1`、`first_unattempted_compute_index=2`；禁止用单个`first_uncompleted_microbatch`混淆compute与merge；
4. failure classification改为`allocator_bfc_oom_during_gradient_merge`，同时保留raw历史值；
5. 新增summary/handoff一致性和阶段语义单测。raw log/checkpoint immutable。

## D1. Donated accumulator（CPU/code only）

只改gradient merge的buffer语义：

1. 将tree add包装为单一JIT边界，第一参数为donated accumulator，例如等价的`jax.jit(_add_gradients, donate_argnums=(0,))`；实现需适配现有global-array sharding；
2. 每个leaf仍执行严格FP32、左结合、按microbatch index `0→15`的`acc = acc + grad_i`；不得改成tree reduce、pairwise sum、低精度或host accumulation；
3. 编译/运行证据必须证明output alias旧accumulator：保存compiled memory analysis的alias bytes；若当前JAX不能报告，则通过donated input失效、post-merge live-bytes回落和无donation warning三项共同证明；
4. merge完成后block，释放current grads和非必要临时对象，再记录`accumulation_complete`与`temporaries_released`；
5. donation未被接受、出现donated-buffer reuse错误、或任何leaf fallback到普通非alias加法时hard stop，不允许静默继续。

CPU测试必须覆盖：16份嵌套FP32 tree与旧顺序实现逐元素exact、nonfinite fail-closed、donated输入失效、异常路径、RNG/loss/optimizer不变，以及sharded toy tree的alias证据。

## D2. 两份真实microbatch merge canary

只有CPU门禁通过、四卡自然空闲且无外部PID时，从immutable step0 fresh restore。只执行真实microbatch compute `0/1`和一次donated merge，不执行optimizer update。

通过条件：

- compute indexes精确`[0,1]`、merge/release indexes精确`[0,1]`；
- index1 compute前允许`old accumulator + current grads`峰值；merge后每卡live bytes必须回到index0 release基线`±256MiB`，不得停留在约`26.42GB`；
- donated alias证据通过，所有loss/grads/accumulator finite；
- step仍为0，source checkpoint未变，退出后GPU释放。

若D2失败，立即停止；不得直接尝试完整16份或Blackwell。

## D3. 完整16份与step1

只有D2通过才用同一代码、allocator和数学配方fresh restore，完成一次global-batch-64 update：

- compute/merge/release indexes均为`0…15`；
- release阶段相对index1 baseline最大漂移`<=256MiB/card`，无随index线性增长；
- optimizer、scheduler、EMA各前进一次，examples seen=64；
- step1 checkpoint atomic save并fresh-process restore exact/finite；
- 若16份累积通过而optimizer update阶段OOM，单独分类为`optimizer_update_capacity_oom`，停止并进入下一次资源审核。

## D4. 连续step10

仅D3全部通过才从step1续跑到step10。每步重复16份plateau门禁，保存step10 atomic checkpoint与fresh restore；报告编译后吞吐、显存峰值和100/1K/30K时间估算但不执行。完成后hard stop。

## 5. 最终状态

最终状态只能是：

- `dense_fullft_donated_accumulator_remediation_failed`
- `dense_fullft_donated_two_microbatch_merge_passed_pending_review`
- `dense_fullft_donated_step1_passed_pending_review`
- `dense_fullft_donated_step10_passed_pending_review`
- `dense_fullft_optimizer_update_capacity_oom_pending_resource_review`

交付D0 summary、alias证据、per-phase memory table、step1/10 checkpoint与restore（若门禁允许）、全量测试和root/nested clean commit。禁止自动进入任何后续训练或官方资源。
