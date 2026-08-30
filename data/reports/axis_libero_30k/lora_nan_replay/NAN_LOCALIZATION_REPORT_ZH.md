# AXIS Pi0.5 LoRA NaN 定位与 500-step 恢复报告

## 结论

计划已执行到 R6。`r6_dual_l2_fresh500_r1` 在 GPU1+GPU2 上从 released Axis dense params + exact zero-delta L2 LoRA fresh 初始化，完成 500 个真实 optimizer update；500 条 metrics 和 16000 条 microbatch trace 全部 finite，checkpoint 链 `0,1,10,25,50,100,250,500` 完整，restore checks `1,10,100,250,500` 通过。

B80 `execute8` 也已完成 80/80：48/80 成功（60.0%），step-0 基线为 50/80；paired wins/losses 为 9/11，paired delta 为 `-0.025`，满足计划的最小成功数 `46/80` 与 paired lower bound `-0.05`。因此当前候选状态为：

```text
lora_candidate_500_pending_review
```

按计划到此停止，不自动进入 1K、5K、30K、LIBERO-plus、Track 3、FAST export 或官方提交。

## 先回答 action range 的问题

这个现象本身不是数据损坏。LIBERO 的 7D raw action 是 controller command units，不是把每个分量都表示成米/弧度的同一物理量；因此平移、旋转和 gripper 的原始范围本来就不同。若“位置约 ±0.1”说的是 EEF observation/state 的物理位移，不能直接与 action 字段比较。

本次 R6 使用的 Phase 1 fresh raw action 统计（顺序 `x,y,z,roll,pitch,yaw,gripper`）为：

| 统计量 | x | y | z | roll | pitch | yaw | gripper |
|---|---:|---:|---:|---:|---:|---:|---:|
| q01 | -0.704 | -0.801 | -0.938 | -0.115 | -0.164 | -0.224 | -1 |
| q99 | 0.938 | 0.868 | 0.938 | 0.132 | 0.193 | 0.335 | 1 |
| std | 0.336 | 0.378 | 0.445 | 0.039 | 0.063 | 0.078 | 0.999 |

也就是说，当前 canonical raw action 的前三维并不是 ±0.1 米；它们是控制器命令空间中的值。gripper 的 `{-1,+1}` 则是预期的离散/二值控制信号。

当前 Pi0.5 路径对每个 action 维度使用自己的 q01/q99 做 quantile normalization，训练和部署使用同一套 inverse transform。真正需要防止的是：训练使用 q01/q99，而某个导出/部署路径误用 mean/std（或反之）。这会改变动作幅度；仅凭“各维 range 不一样”不能判定是 NaN 根因。

## NaN 证据与修复

旧的两卡失败发生在第一个累计 optimizer step 内、有效 update 之前，`optimizer_updates_completed=0`；这不是 500-step 行为回归。旧 formal run 与 retry2 使用不同 root/OpenPI commit，不能称为同一 runtime 下的随机复现。

在固定 batch/RNG 的 R3 审计中，D12 replay 暴露出 RNG/noise buffer 在重放计算中被变异；单纯复制 host-side `np.load` view 仍会失败。修复为每个 microbatch 物化私有 sharded time/noise fixture 后，fixed replay gate 为 9/9 finite，U1 矩阵为 16/16 finite，sync/async checkpoint 变体也完成并 finalized。这个结果支持“replay/runtime 路径问题已修复”的受限结论，但没有证据把根因唯一归因给 XLA、分片、硬件或 Orbax。

## R4 / R6 数值门禁

- R4：三个独立 fresh-10 双卡 run 全部完成 10 updates；独立 fresh-100 完成 100 updates；metrics、checkpoint finalize、restore、fixed-forward 均通过。
- R6：500/500 optimizer steps finite；每步 non-finite leaf count 为 0；step-500 dual-topology restore 通过。
- step-500 fixed-forward 输出 finite，external shape 为 `(10,7)`，internal shape 为 `(2,10,32)`。
- 需要人工注意：双 V100 dual-topology 重复 forward 的 `repeat_max_abs_diff=0.28373149`，未达到 bitwise/记录的 FP32 repeatability tolerance。该 warning 已保留在原始报告中，不能静默改写成 deterministic pass。

## R6 B80 行为门禁

主协议为锁定的 B80 `execute8`。两路分片均完成 40/40，合并后：

| 指标 | step-0 baseline | L2 @ 500 |
|---|---:|---:|
| success | 50/80 | 48/80 |
| success rate | 62.5% | 60.0% |
| paired wins/losses | — | 9/11 |
| paired delta | — | -0.025 |
| collision | 19 | 20 |
| timeout | 17 | 19 |
| stall | 0 | 0 |
| CSV/H.264 | 80/80 | 80/80 |

80/80 的结果行、CSV、视频和 failure tags 均已生成；无 episode error。full_chunk 500 没有运行，因为它是 diagnostic-only，不影响本次主门禁判定。

## Provenance 与 GPU 安全

训练 runtime：root `14161dea8b1bf801d57c97b9af2530ec4af52898`，nested OpenPI `dafa5a27266996eecf28920cacc32f3629776d91` / tree `887fa06db36a72288c2db6756c684f85946e3388`。行为 runner 为支持显式 LIBERO config override 使用 root `befff6d`；这与训练 checkpoint provenance 分开记录。数据仍是 canonical `physical-intelligence/libero@a4336d589d589045d1c56423ffdf3b88a0e19b1f` 原生 10 Hz，未切换到 Plus/20 Hz/Track 3 数据。

全程仅使用 GPU1/GPU2，`CUDA_VISIBLE_DEVICES=1,2`；GPU0 外部 PID 未绑定、未终止、未重置，所有相关报告的 `gpu0_touched=false`。

## 证据入口

- [机器可读汇总](./summary.json)
- [R6 training metrics](/home/huang/code/PARC2026_final/outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1/training_metrics.jsonl)
- [R6 checkpoint manifest](/home/huang/code/PARC2026_final/data/reports/axis_libero_30k/lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1/checkpoint_manifest.json)
- [R6 B80 merged summary](/home/huang/code/PARC2026_final/data/reports/axis_libero_30k/phase1/rollout/r6_dual_l2_fresh500_r1_behavior_merge/B80_execute8/summary.json)
