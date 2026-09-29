# AXIS Pi0.5 R6 T2 轻量平滑最后一次推理屏选计划

> 日期：2026-09-17（Asia/Tokyo）  
> 状态：`authorized_inference_only_four_card_screen`  
> 目的：在明日提交前，只用一次低风险闭环试验判断动作接缝平滑是否能改善 T2。

## 结论边界

- 本计划只做 fresh inference / simulator rollout；不训练、不 resume、不改数据、不改 normalization、不改 R6 权重。
- 既有 R6 execute4 官方 ZIP、horizon sweep、H300、attempt_0002 和所有历史 evidence 都是只读输入。
- 固定 `chunk_size=10`、20 Hz、同一 R6 step500 params-only checkpoint、同一 Fast400 QUANTILES norm、同一冻结 4 reverse + 20 C20 case/init/env/policy seed。
- 四个 variant 各自使用独立 fresh worker、独立 GPU、独立 evidence/output namespace；默认不生成 ZIP、不上传、不提交。

## 四卡 variant

| GPU | variant | 实际动作协议 |
|---:|---|---|
| 0 | `execute4_reference` | execute4，无平滑，作为本次 fresh reference |
| 1 | `seam_blend` | 每个新 chunk 的第一帧仅 arm 前六维做 `0.9*new + 0.1*old_chunk_tail`；gripper 原值 |
| 2 | `arm_ema01` | 每帧仅 arm 前六维做 `0.9*new + 0.1*previous_emitted_arm`；gripper 原值 |
| 3 | `execute5` | execute5，无平滑 |

首次 action 没有旧动作时不混合。chunk seam 的旧动作是上一 chunk 实际发出的最后一帧；EMA 的旧动作是上一帧实际发出的平滑结果。所有平滑发生在 simulator 执行前，服务端模型输出、权重、processor、norm 不变。

## 屏选门

以本次 fresh `execute4_reference` 为配对基线。候选（`seam_blend`、`arm_ema01`、`execute5`）必须全部满足：

1. 24/24 case 覆盖，case fingerprint、init/env/policy seed 完全相同，4 reverse + 20 C20；
2. C20 success 不低于 reference，reverse success 不低于 reference；
3. C20 和 reverse collision 数均不高于 reference；
4. 无 infrastructure/model exception，action 与实际 EEF trace 全部 finite 且非零；
5. 24/24 视频都存在，并由独立 `ffprobe` 读出视频流；
6. 由 simulator 观测的实际 EEF 位置计算的离散 jerk RMS 严格下降，且 24-case 平均实际 steps 严格下降。

`EEF jerk RMS = sqrt(mean(||Δ³ eef_position / dt³||²))`，其中 `dt=0.05s`；位置来自每次 `env.step` 后的 `robot0_eef_pos`，不是 action proxy。步数使用 episode 实际 `total_steps`。

任一候选不满足即拒绝，不做补跑、不挑 case、不生成 ZIP。若多个候选通过，选 C20 success 较高、collision 较少、EEF jerk 较低、平均 steps 较少者；若没有通过候选，保留现有 T2 execute4 官方姿态（用户已知官方分数 0.3457），不回退 execute8。

## 条件式发布

只有比较摘要明确 `release_authorized=true` 时，才允许基于只读 R6 staging 构建一个新的 T2 ZIP。新 ZIP 仍不得自动上传或提交，且必须再次通过 static/CRC、真实 health、固定输入 parity、manifest 和独立 ffprobe/trace 审核。失败时 ZIP 必须保持未创建。

## 硬停

本计划不授权 30k、dense、H300 resume、任何训练或数据补采。screen 完成后无论成败都停止，等待用户决定是否手动提交唯一通过审核的 ZIP。
