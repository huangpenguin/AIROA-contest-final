# AXIS Pi0.5 Dense Full-FT 跨 optimizer step State 生命周期修复计划

> 制定日期：2026-09-02（Asia/Tokyo）
> 审核输入：root `c8b0a42`，nested OpenPI `5c29b0e`
> 当前状态：`dense_fullft_interstep_lifetime_remediation_pending`
> 唯一目标：不改变训练数学与global batch 64，消除 optimizer update 后跨步残留的一份约 `6.5GB/card` full-tree-equivalent allocation，并在本地4×V100证明连续到internal step10的稳定显存 plateau。

## 0. 已锁定事实

1. donated accumulator 本身有效：alias `6,479,127,152` bytes，2份和16份ordered FP32 merge均通过；不得回退为普通tree add。
2. D4从internal step1恢复，完整完成一次update到step2；失败发生在下一轮 `optimizer_step_before=2` 的microbatch 1 compute。
3. restore baseline约`13.435GB/card`；持有一份accumulator时约`19.942GB/card`；update后下一轮microbatch 0变为约`26.511GB/card`。跨步残留约一份full gradient tree，但具体引用来源尚未证明。
4. 当前问题是runtime/device-object lifetime，不是模型得分结论，也尚不足以证明必须使用96GB Blackwell。

## 1. 不可变合同与禁止项

| 字段 | 固定值 |
|---|---|
| initialization | Axis released params revision `c342608c489515efa9ba8281400dc648bf3970e1` |
| data/norm | `physical-intelligence/libero@a4336d…`、原生10Hz、fresh per-dimension QUANTILES `4c803bf5…b18f5` |
| topology/precision | dense full model / FP32 |
| optimizer/EMA | 既有AdamW recipe / `0.999` |
| devices/microbatch | 本地4×V100 / `1` per device |
| accumulation/global batch | 严格左结合ordered `16` / exact `64` |
| allocator | preallocate=true、fraction `0.95` |
| checkpoint source | D3 immutable internal step1 checkpoint，先锁tree hash再使用 |

不得降低精度或batch，不得改变accumulation顺序、loss、optimizer、EMA、freeze scope、data/norm、RNG、remat或sharding。不得用`jax.clear_caches()`每步清缓存、每步重启进程或checkpoint-per-step作为正式修复；这些最多只能作为明确标记的诊断，不得成为通过路径。不得启动官方/比赛GPU、Blackwell、C20、50/100/1K/30K、FAST/ZIP、上传或提交。

## L0. 证据语义闭合（CPU only）

1. 锁定D2/D3/D4 raw log、allocator trace、metrics与checkpoint hashes；不修改原文件；
2. 新派生summary必须区分：`restored_internal_step=1`、`completed_update=1→2`、`failed_optimizer_step_before=2`、`completed_microbatches=[0]`、`failed_compute_microbatch=1`；
3. 生成按边界和GPU列出的bytes表，至少包含restore、step start、每个release、optimizer start/complete、metrics host materialization、next-step start；
4. 新增解析测试，禁止再把loop index、pre-update internal step与post-update checkpoint step混为一谈；
5. 校验root/nested clean、四卡自然空闲且没有外部PID。L0不启动GPU。

## L1. 最小生命周期实现与CPU门禁

只允许处理引用/输出/donation语义，不允许改训练值：

1. 在host accumulation路径增加明确边界：`optimizer_returned`、`new_state_ready`、`metrics_host_materialized`、`device_metrics_released`、`next_step_start`；allocator observer不得lower/compile或读取已donate对象；
2. optimizer返回的所有metrics先同步为host scalar，再写JSON/累积日志；`infos`不得跨step持有device arrays。完成finite检查后显式释放device `info`、`grads`、`updates`可达引用和旧局部对象；
3. 对`papply_optimizer_update`记录compiled alias/donation证据。由于optimizer update后不再使用mean gradient，允许在CPU/toy门禁通过后把gradient参数与state一起donate；不得静默忽略donation warning或fallback；
4. 对同一fixed toy state/batch比较修复前后：new params、Adam state、EMA、RNG、step、所有metrics必须在既有dtype下exact；若仅因host scalar container造成类型变化，数值和序列化文本必须exact；
5. 测试Python弱引用/已删除buffer、异常路径、nonfinite fail-closed、metrics logging和checkpoint调用，证明没有旧state/gradient/device info跨step可达；
6. 若无法判定是哪类引用，允许CPU/code-only的`no-device-metrics`与`donate-gradient`拆分探针，但正式GPU候选必须只保留一个实现，禁止在GPU上无界枚举方案。

L1失败即停止。不得以`gc.collect()`、sleep或降低allocator压力伪造通过。

## L2. step1→2→next-micro0 transition canary

只有L0/L1及全量CPU测试通过才允许一次本地4×V100 fresh process：

1. 从immutable internal step1 checkpoint恢复，确认params/optimizer/EMA/RNG finite与sentinel exact；
2. 完成严格16份microbatch及update `1→2`，metrics finite、examples seen `64`；
3. 不保存step2 checkpoint，不进入第二份microbatch；只计算下一轮 `optimizer_step_before=2` 的microbatch `0`，完成donated accumulator建立与release，然后按专用boundary-stop正常退出；
4. update后`next_step_start`必须回到不含gradient的baseline；下一microbatch 0 release必须与上一轮任一post-merge release的每卡live bytes相差`<=256MiB`，目标约`19.94GB/card`，不得再次约`26.51GB/card`；
5. donation/alias证据、finite、RNG/step、GPU PID和退出释放全部通过。任何OOM、donation warning、跨步增长或数值不一致都hard stop。

L2只验证transition memory，不把boundary-stop产物称为训练checkpoint。

## L3. 两次连续update与step3 checkpoint

只有L2通过才fresh restore internal step1，连续完成update `1→2` 与 `2→3`：

- 每轮compute/merge/release精确`16/16`；
- 每个step的无gradient baseline与持有accumulator plateau相对L2分别漂移`<=256MiB/card`，不得随step单调增加；
- optimizer/scheduler/EMA/RNG各前进两次，cumulative examples seen `128`，所有metrics finite；
- internal step3 checkpoint原子保存，fresh-process restore exact/finite；
- checkpoint observer前后单独记录显存，保存完成后不得保留旧state snapshot。

L3失败立即停止，不允许转Blackwell。

## L4. 连续到internal step10

只有L3全部通过才从step3续跑到step10。每步重复plateau和finite门禁；保存step10 atomic checkpoint并fresh-process restore。报告：

- 每step compute/merge/update时长及编译后吞吐；
- baseline/accumulator/update/checkpoint峰值表；
- 100/1K/10K/30K wall-clock估算及60小时可达step估算；
- 本地V100与比赛Blackwell只按显存/吞吐做资源建议，不启动后者。

完成L4后hard stop等待人工审核；不得自动开始C20、长训、FAST或提交。

## 2. 最终状态与交付

最终状态只能是：

- `dense_fullft_interstep_lifetime_remediation_failed`
- `dense_fullft_interstep_transition_passed_pending_review`
- `dense_fullft_interstep_step3_passed_pending_review`
- `dense_fullft_interstep_step10_passed_pending_review`

交付L0派生summary、compiled donation/alias证据、per-boundary memory table、L2/L3/L4报告与允许阶段的checkpoint/restore、全量测试、root/nested clean commit。每个hard gate失败都必须保留raw evidence并停止；不得自动扩权。
