# AXIS Pi0.5 E1 JAX Full OOM 审核记录

> 审核日期：2026-08-31（Asia/Tokyo）
> 被审核 commit：`48206a708439b4a08af0ea9f8809eab9950572a5`
> lock commit：`7d545ef`
> 审核结论：`remediation_required`
> 下一授权：只恢复 folded JAX source 的 `340/340` 单样本全量推理；不授权 PyTorch、comparator、closed-loop、E2、FAST、ZIP、训练、push 或上传。

## 1. 审核结论

E1 的 hard stop 正确。CPU input lock 与 JAX sentinel 可以接受，但 E1 尚未通过：JAX full run 在 `microbatch_size=4` 时 CUDA OOM，因而 full coverage 为 `0/340`，LeRobot/PyTorch 与完整 action-response envelope 均未产生。

当前失败更符合执行时显存峰值配置问题，而不是输入、normalization、fold 或模型完整性已经失败：

- 同一 folded JAX source 已用 `microbatch_size=1` 完成 4 observations sentinel；internal/external action 为 `[4,10,32]` / `[4,10,7]`，全部 finite。
- sentinel 的四个单样本 latency 约为 `18.02/12.19/12.08/12.02s`；GPU1 monitor 峰值约 `24717 MiB / 32768 MiB`。
- full run 把 microbatch 提至 `4` 后，在额外申请 `216 MiB` 时 `RESOURCE_EXHAUSTED`，符合 batch 放大 activation 峰值的表现。
- 输入/noise、模型、raw tree、norm 与 processor provenance 均已锁定，没有 hash drift 证据。

这只是根因分级，不等于 E1 获批。只有新的 `microbatch=1` run 完成 `340/340`、finite、完整顺序和 hash 校验后，JAX full 才能进入下一次人工审核。

## 2. 不可变证据

```text
E1 input lock SHA-256              b96acf22e01f81d89c0bebf8adfd3dfd729e485cdc29a1b87c6fc748f67534c4
E1 pair manifest SHA-256           09c9e000135c192b6a3154b82c06f754a047d8006122a888f6c933852887f66e
frozen inputs/noise SHA-256        263a7723d1f866eb73eee9423eda7480d96e1678b4a401bd4d86460b7c37c7fe
JAX sentinel actions SHA-256       f3977cebecbd186c37ee5f4a5de53e7ce43bcef8141ed93586cd16e335422e65
raw Orbax tree SHA-256             d203635f1eb0b006ec27841db6ce5171f74a37035eca4291c8107c05838c276f
folded model SHA-256               5c73e1e8217dbb75a934c2b0165208e851f69e66cfe2ea0430590a29901e6d5b
R6 norm SHA-256                    4c803bf59c52677c36b6b420c614d8a7742e0847708443863e13e67cb80b18f5
locked pair / observation          170 / 340
snapshot / trajectory pair        13 / 157
JAX sentinel / full               4 / 0
PyTorch full / envelope           0 / not generated
```

`git diff 7d545ef..48206a7` 对 lock 与 frozen pair manifest 无差异；因此 action inference 后未改写 lock。GPU0 的外部 PID 未被触碰；GPU1 已释放。审核时 GPU0 仍有外部进程，GPU1–3 空闲。

## 3. 独立验证

```text
root HEAD                         48206a708439b4a08af0ea9f8809eab9950572a5
nested OpenPI                     dafa5a27266996eecf28920cacc32f3629776d91
root/nested at audit start        clean
tests                             116 passed, 7 skipped, 21 warnings
test command                      .venv_axis_pi05/bin/pytest -q tests
```

## 4. 判定

- CPU lock：`approved`。
- JAX sentinel：`approved`。
- JAX full：`failed_due_to_microbatch4_oom`。
- 整体 E1：`remediation_required`。
- R6 候选模型：本审核不作性能判定，也不因 OOM 否决。

下一步必须是独立、可恢复、固定 `inference microbatch=1` 的 JAX full recovery。不得直接续跑旧进程，不得为挤显存改变 dtype、allocator、模型、norm、noise、输入顺序或 GPU 边界；也不得在 recovery 后自动进入 PyTorch。
