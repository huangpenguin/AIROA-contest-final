# R6 T2 轻量平滑最后一次推理屏与 ZIP 收尾

> 日期：2026-09-17（Asia/Tokyo）
> 状态：candidate_passed_zip_ready_for_user_manual_submission
> 自动上传/提交：false

## 结论

本次四卡 fresh inference screen 只改变动作执行边界，不训练、不 resume、不改变 R6 权重或 normalization。

唯一通过全部预注册门禁的候选是 seam_blend：

- execute4、chunk_size=10、同一 R6 step500 FAST 权重与 QUANTILES normalization；
- 每个新 chunk 的首帧仅将 arm 前六维按 0.9*new + 0.1*previous_chunk_tail 混合；
- gripper 第七维保持模型原值，不做平滑；
- 24/24 case、4 reverse + 20 C20、case fingerprints 与 reference 完全相同。

可交付的唯一新候选 ZIP 已完成独立打包和静态审计。它不是官方分数，也不保证超过已有 execute4 官方分数；官方提交仍需用户手动完成。

## 四变体结果

| variant | success | collision | C20 | reverse | mean EEF jerk RMS | mean steps | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| execute4_reference | 15/24 | 5 | 15/20, collision 2 | 0/4, collision 3 | 7.224137 | 207.7083 | fresh reference |
| seam_blend | 15/24 | 5 | 15/20, collision 2 | 0/4, collision 3 | 6.848622 | 205.9583 | 通过 |
| arm_ema01 | 15/24 | 6 | 15/20, collision 3 | 0/4, collision 3 | 6.839604 | 207.2083 | 拒绝：collision 增加 |
| execute5 | 15/24 | 7 | 15/20, collision 4 | 0/4, collision 3 | 7.652576 | 199.9167 | 拒绝：collision 增加、jerk 变差 |

seam_blend 相对本次 fresh reference 的实际 EEF jerk 下降约 5.20%，平均实际步数下降约 0.84%；success 和 collision 均不变。它没有解决 reverse（仍为 0/4），只是一个低风险的 T2 推理改进候选。

## ZIP 审计

- ZIP：artifacts/submission_axis_r6_step500_fast_seam_blend_t2_20260917.zip
- SHA256：43f047f7561ac7a4601d691b0c456022b2ca029ab44aad4f4320b1b0853b53a0
- bytes：9383573232
- package_validation：passed
- validate_submission.py --static：passed
- ZIP CRC/testzip：passed
- entry names：与冻结 source 完全相同
- changed entries：仅 policy_server.py、model_weights/config.json、submission_manifest.json
- 未修改 entry 的 size/CRC：全部相同
- 包内 FAST model SHA：e55bf7dcca78b2195993dd7a12b96a4ea13e94761d711c99a82b7411c3f8ceb6
- 包内真实 server smoke：health/reset/6 次 /act、finite/nonzero、跨 execute4 seam、受控退出全部通过
- 新证据目录 JSON：6/6 duplicate-key-reject strict parse
- 无 .partial、无残留 server/训练/评测进程；GPU0–3 均 5MiB / 0%

旧 execute4 ZIP 未触碰，SHA 仍为：

2f2f9c2ed627de50b1af4b9d1501e9f5d4d467e8455ed51f363247a3d7bd3fc9

## 证据入口

- screen comparison：data/reports/axis_libero_30k/r6_t2_light_smoothing_screen_20260917_attempt_0001/comparison.json
- package release：data/reports/axis_libero_30k/r6_t2_seam_blend_submission_20260917_attempt_0001/release.json
- package validation：data/reports/axis_libero_30k/r6_t2_seam_blend_submission_20260917_attempt_0001/package_validation.json
- ZIP delta：data/reports/axis_libero_30k/r6_t2_seam_blend_submission_20260917_attempt_0001/zip_delta_audit.json
- static validator：data/reports/axis_libero_30k/r6_t2_seam_blend_submission_20260917_attempt_0001/validate_submission_static.json
- builder：tools/axis_pi05/build_r6_t2_seam_blend_submission.py

## 明日动作

如果仍有一个 T2 手动提交机会，提交上述 seam-blend ZIP；保留旧 execute4 ZIP 作为明确可回退副本。不要再开 EMA、execute5、300/500、30k、dense、H300 resume 或任何训练。提交后只按官方返回分数落盘，不能把本地 15/24 或 jerk 改善写成官方提升。

