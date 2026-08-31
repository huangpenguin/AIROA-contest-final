# Dense full-FT R4 allocator OOM 审核（2026-09-01）

## 审核结论

`remediation_required`。R3 sharded init 的实现和证据通过审核；R4 确实进入了首个真实 microbatch 计算，但当前不能将该 OOM 分类为“4×32GB 真实物理容量不足”，也不批准立即转比赛 Blackwell。

原因是 R4 明确将 `XLA_CLIENT_MEM_FRACTION`、旧名 `XLA_PYTHON_CLIENT_MEM_FRACTION` 和 `XLA_PYTHON_CLIENT_PREALLOCATE` 全部留为 unset，JAX 使用默认预分配策略。JAX 官方文档说明：首次 JAX GPU 操作默认预分配总显存的 75%；`XLA_CLIENT_MEM_FRACTION` 用来调整该比例，而 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 会改为按需分配且更容易受碎片影响。来源：[GPU memory allocation — JAX documentation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)。

## 核对证据

- root `6239704752dc2966d23b0f6c9d49744efdb152ef`、nested OpenPI `92604d6d07d5bbf636827ee903f758538c8ba015`，审核开始时两者 clean。
- R3：51 leaves、`3,353,433,872` FP32 params、large replicated leaf `0`、sentinel exact，params/Adam/EMA finite；这证明前一轮 replicated-init 问题已修复。
- R4：per-device microbatch `1`、4-GPU accumulation `16`、global batch `64`、FP32/dense/EMA `0.999`均未改变；step-0 checkpoint 完成，optimizer update `0`。
- OOM 发生于 `train_step_accumulated_host` 的首个 `microbatch_grad_fn`，BFC allocator 在每卡请求 `143,327,232 bytes` (`136.69 MiB`) 时失败。
- NVML 峰值在四卡上都是 `24,897 MiB / 32,768 MiB = 75.98%`。75% 的纯显存池为 `24,576 MiB`；多出的约 `321 MiB` 可合理解为 CUDA context、libraries/NCCL 和 allocator 外开销。这是从官方默认值与本地量测一致性得出的推断。
- NVML 显示的剩余约 `7,871 MiB/card` 不等于当前 BFC pool 内有可用连续块。`nvidia-smi` 看的是进程向 driver 保留了多少显存，不是 JAX 池内 live tensors/free chunks 的账本。
- `136.69 MiB` 是失败的单次连续分配，不能解读为“整个训练只差 136.69 MiB”；后续 backward、gradient accumulation、clip、Adam 和 EMA 还可能出现更高峰值。

## 决策

先在本地独占 4×V100 上保持 preallocation，只将 `XLA_CLIENT_MEM_FRACTION` 从默认 `0.75` 提高到 `0.90`；如果在完整 first update 前仍 OOM，且证据显示还是 allocator-limit 而非外部进程，允许唯一次 `0.95` 复测。这只改变 allocator 可用池，不改变 FP32 数值、模型、loss、batch、optimizer 或 EMA。

不使用 `XLA_PYTHON_CLIENT_PREALLOCATE=false`、`platform`、`cuda_malloc_async` 或其他 allocator，避免把内存池容量问题与碎片/分配策略一次性混在同一实验。只有 `0.95` 在独占四卡上仍无法完成 first update，才把结果升级为本地 32GB 容量边界，作为启动比赛 Blackwell 的人工审核依据。
