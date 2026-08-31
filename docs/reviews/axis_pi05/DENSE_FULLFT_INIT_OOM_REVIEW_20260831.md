# Dense full-FT 初始化 OOM 审核（2026-08-31）

## 审核结论

`remediation_required`。执行者正确保留了 FP32、dense 全参数、EMA `0.999`、global batch `64` 和 Axis/canonical LIBERO 输入，并在两卡、四卡首次 OOM 后停止；没有伪造 optimizer update、checkpoint、restore 或 C20。测试 `125 passed, 7 skipped`，root/nested 工作树在输入 commit `dbd4f49` clean。

然而，当前证据不支持“4×V100 无法训练该配方”的结论。两次 OOM 都发生在 `init_train_state`，还没有取 batch，也没有 forward/backward/update。先要修复参数加载时的不必要复制峰值，再判断本地四卡和比赛 Blackwell 的边界。

## 核对事实

- 模型为 `3,353,433,872` 个 FP32 参数；参数、Adam `mu/nu`、EMA 和梯度的理论总下限约 `67.069 GB`，未包括 activation、XLA 临时区和加载峰值。
- 2 GPU 配置为 per-device microbatch `1`、accumulation `32`；4 GPU 为 per-device microbatch `1`、accumulation `16`。两者 effective global batch 都是 `64`。
- microbatch `1` 已经是每卡最小整数样本。降低 accumulation/global batch 只会减少后续累积的次数，不会降低本次在 batch 出现之前的初始化峰值。
- `train.py` 先从 Orbax 恢复约 `12.5 GiB` Axis params，然后用 `PartitionSpec()` 作为 `init` 的统一 `in_shardings`；这会在初始化时将 `partial_params` 整份 replicated 到每张 GPU，而输出 train state 才是 FSDP-sharded。
- 四卡日志明确显示 `Some donated buffers were not usable`，input/output arguments 合计 `26,840,528,788` bytes，并在 `init_train_state` 为每卡再申请 `54 MiB` 时 OOM。这是 initialization transient，不是训练 activation 已经证明超限。
- 现有 FSDP 工具会对 params、Adam `mu/nu` 和 EMA 的大矩阵分片；问题是 weight-loader 输入没有使用与 params 一致的 sharding。

## batch 概念与可调整边界

```text
effective global batch
  = per-device microbatch
  × data-parallel/FSDP devices
  × gradient-accumulation steps

4卡：1 × 4 × 16 = 64
2卡：1 × 2 × 32 = 64
```

- `microbatch` 是每张卡一次 forward/backward 同时处理的样本数，主要决定 activation 峰值。当前为 `1`，无法再降。
- 本项目语境的 `batch=64` 是一次 optimizer update 消化的总样本数。通过 accumulation 可在不改 global batch 的情况下使用 microbatch `1`；这是同一数学目标的梯度累积设计，不降低参数 dtype 精度。但 FP32 加法次序不同可产生最后几个 bit 的差异，因此不声称与“单次同时计算64样本”bitwise exact；梯度、update 和 restore 将按预注册容差验证。
- 无损配方优先顺序：参数从加载开始就 FSDP-shard → 保留/加强 activation rematerialization → 必要时 CPU offload。这些不改 FP32 数值格式或 global-batch 语义，代价是速度或实现复杂度。
- 改 BF16/FP16、去 EMA、冻结参数、缩短 token/图像、改 optimizer 或降 global batch 都会改变精度或训练 recipe，不用于本次第一级修复。

## 资源决策

现在不启动比赛 GPU。单卡 RTX PRO 6000 Blackwell 虽有约 96GB VRAM，但当前初始化会并存 full partial params 和 train-state outputs，加上训练 activation/临时区后仍不是必然稳定。先用本地四卡修复 sharded init，才能区分“代码峰值”和“真实训练峰值”。

只有在 sharded init 已经通过、而 microbatch `1` 的真实 first update 仍在 4×32GB V100 因 activation/optimizer 超限时，才建议把后续 preflight 与长训练合并到同一次 60 小时 Blackwell session，不为单独试跑额外消耗配额。
