# Official canary100 本地 GPU3 评测交接（2026-09-11）

> 给下一个 agent：评测已结束。延续计划是 `docs/AXIS_PI05_T1_T2_DEDICATED_PANEL_AND_OFFICIAL_250_HORIZON_PLAN_20260911.md`。  
> **不要**从聊天记录重建。GPU3 空闲。长训授权仍全部为 false。  
> Track3 采集仍只走 `docs/AXIS_PI05_TRACK3_FAST_TELEOP_400_AND_SHORT_TRAIN_PLAN.md`，不要改它的 cursor/manifest。

机器 JSON（权威计数与 SHA）：

```text
data/reports/axis_libero_30k/official_canary100_local_gpu3_eval_20260911/EVAL_HANDOFF.json
```

`training_steps = 0`。`official_result = false`。没有开 250/300/500。没有 ZIP/提交。

## 1. 现在的状态

T3 / T1 / T2 的 step100 配对评测 **都已写完**。评测进程不在，GPU3 约 5 MiB。本地 GPU0/1/2 仍被外部 YOLO 占用，不要碰。

评测在本机 Tesla V100 GPU3，不是官方 Blackwell。官方 trainvenv 没有 LIBERO，当时不能在那边评。

## 2. 原始结果文件（不可改写）

| Track | 摘要 | 输出 |
|---|---|---|
| T3 N5 | `data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/n5_official_t3_step100_paired_eval/paired_summary.json` | `outputs/axis_pi05/track3_four_gold_mix_v1/n5_official_t3_step100_paired_eval/` |
| T1 C20 | `data/reports/axis_libero_30k/track12_t1_p20_step100_c20_eval/paired_summary.json` | `outputs/axis_pi05/track12_t1_p20_step100_c20_eval/` |
| T2 C20 | `data/reports/axis_libero_30k/track12_t2_p30_step100_c20_eval/paired_summary.json` | `outputs/axis_pi05/track12_t2_p30_step100_c20_eval/` |

不要写入 `data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q7_gpu3_frozen_eval/`。

## 3. 数字

对照都是本地 R6 step500 params vs 官方 canary step100。协议 20 Hz、chunk10、execute8。

| Track | 面板 | R6 | step100 | Δ | 续训门（当时冻结的） |
|---|---|---:|---:|---:|---|
| T3 | reverse 4 | 0 | 0 | 0 | 要 +1/4 → **未过**，不能 step300 |
| T3 | C20 20 | 14 | 14 | 0 | 掉幅 ≤2 → 过 |
| T1 | C20 20 | 17 | 17 | 0 | 掉幅 ≤2 → 过；专用 Plus 面板未冻 → **不能 step250** |
| T2 | C20 20 | 16 | 15 | −1 | 掉幅 ≤2 → 过（唯一 −1 是 `C20-task12-init0`）；专用 Plus 面板未冻 → **不能 step250** |

T1 C20 持平 **不是**「评测失败」。卡住 T1/T2 的是「专用 Plus 面板 +2」这条额外门，不是 C20。T3 卡住的是 reverse 没有 +1。

## 4. 检查点

本地（gitignored）：

```text
outputs/axis_pi05/official_canary100/t1_p20_canary100/100
outputs/axis_pi05/official_canary100/t2_p30_canary100/100
outputs/axis_pi05/official_canary100/t3_four_gold_canary100/100
outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1/500
```

私有 HF 合同：`configs/axis_pi05/official_canary100_hf.json`。  
官方 persist：`~/data/parc20260911/`（T1/T2/T3 的 `*/100` 与 R6）。

T1/T2 评测锁的是检查点 `assets/**/norm_stats.json`（T1 `f49d37da…`，T2 `39ff8a7c…`），不是训前源文件 SHA。差别是 OpenPI 落盘格式，不是 GPU。

## 5. 门禁去向（2026-09-11 21:14 JST）

用户决定 **不改门**：先在本地 GPU3 冻结并评测 T1/T2 专用面板（T1 lighting+coarse env；T2 camera_pose+sensor_noise+coarse env）。延续仍要求专用面板 +2、各类无净退化、C20 掉幅 ≤2。达不到不长训。面板无法构造时只能再明确授权改门，且最多先批 T1 一条探索性正式训练。

T1 正式训练必须新建 250-horizon runner（R6 fresh optimizer，`stop_after_step=100`），禁止把现有 canary100 `--resume --steps 250`。T3 不训练；继续 Fast400。不开 500。不合并三条。

计划：`docs/AXIS_PI05_T1_T2_DEDICATED_PANEL_AND_OFFICIAL_250_HORIZON_PLAN_20260911.md`

## 6. 设计下一步时不要做的事

- 不要用 `launch_official_blackwell.py` 开 >100 的 fresh canary（`MAX_FRESH_CANARY_STEPS = 100`）。从 step100 续训需要 **新的 resume 入口**，现在还没有。
- 不要动 GPU0/1/2。本地评测只用 GPU3。
- 不要把 LIBERO 装进官方 trainvenv。
- 不要往官方 runtime 上叠 Q7 的 `train.py` / `data_loader.py` / `config.py`。
- 不要发明 T1/T2 lighting/camera 评测集充作已冻结专用面板。先 constructibility/effect test，失败则 `panel_not_constructible`。
- 未过专用面板门、也未收到改门授权前，不要开 250/300/500，不要 ZIP/上传/提交。
- 不要把现有 canary100 `--resume --steps 250`。正式 T1 必须 250-horizon + `stop_after_step=100`。
- 官方实例若已 Stop：NVMe 会没，persist/HF/token 还在。重启步骤见 `docs/AXIS_PI05_OFFICIAL_BLACKWELL_STOP_AND_RESTART_AFTER_CANARY100_20260911.md`。

## 7. `not_evaluated`

专用 T1/T2 Plus 面板、官方机 LIBERO 评测、resume-from-step100 入口、step250/300/500 训练、ZIP/提交。不要把这些写成已经跑过。
