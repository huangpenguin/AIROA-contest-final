# R6 execute4 官方成绩与推理侧建议（2026-09-16）

## 结论

用户报告 execute4 ZIP 的正式分数为 Track1 `0.364`、Track2 `0.3457`。相对既有 R6
execute8 的 Track1 `0.47`、Track2 `0.33`，execute4 在 Track1 下降 `0.106`，在 Track2
上升 `0.0157`。因此 execute4 不是通用升级；当前最优提交姿态是 Track1 保留 execute8，
Track2 使用 execute4。两 Track 按各自最好配置合计为 `0.8157`，高于统一 execute8 的
`0.8000`，也明显高于统一 execute4 的 `0.7097`。

正式规则确认成功与碰撞先作门控，随后还计算归一化后的 time、jerk、SPARC、trajectory
与 EEF rotation；权重和归一化公式不公开。公开仓库的 scorer 只把 success rate 作为
overall score，轨迹指标只是报告项，不能复现正式 Total Score。因此此前 C20 的
success/collision 门只能筛灾难性回归，不能预测正式连续分数。

## 本地轨迹解释

在同一批 20 个 C20 case 上，execute4 相对 execute8：

| 指标 | execute4 | execute8 | 解释 |
|---|---:|---:|---|
| success | 17/20 | 15/20 | execute4 更频繁闭环纠错 |
| collision | 1 | 4 | execute4 的本地安全门更好 |
| timeout | 2 | 1 | 更频繁推理没有缩短执行 |
| 平均 steps | 186.9 | 175.5 | execute4 更慢 |
| arm action 一阶差分 L2 | 0.084144 | 0.077267 | execute4 高约 8.9% |
| arm action 二阶差分 L2 | 0.081478 | 0.057663 | execute4 高约 41.3% |
| gripper sign flips | 3.7 | 2.5 | execute4 切换更频繁 |

动作差分不是正式 EEF jerk/SPARC，但足以表明 execute4 的本地 success/collision 收益伴随
更差的平滑与效率代理。正式 Track2 小幅上升而 Track1 大幅下降，符合“不同 Track 的门控
收益与连续指标折中不同”，不支持一个 horizon 通吃。

## 下一步建议

1. Track1 冻结 R6 execute8，不再用 execute4 覆盖已证明的 `0.47`。
2. Track2 当前以 execute4 的 `0.3457` 为正式最好结果。
3. 若还有一次 Track2 测分机会，只考虑一个轻量、可回退的推理候选：维持 chunk10/execute4，
   只对前六个 arm action 做极轻 EMA 或仅在 chunk seam 混合 1–2 步，gripper 不平滑。
   离线代理中 previous-action 权重 `0.1` 可把 execute4 的一阶差分从 `0.084144` 降到
   `0.080115`、二阶差分从 `0.081478` 降到 `0.071691`，但这不证明闭环 success 不下降；
   必须先 fresh paired rollout。
4. RTC 当前只支持 SmolVLA，Pi0.5 路径会 fail-close；chunk size `10` 是模型输出合同，均不应
   在最后时段修改。全局重 EMA、动作幅度硬裁剪、gripper 平滑或多样本动作平均都容易造成
   接触阶段迟滞，不建议盲交。

机器证据：
`data/reports/axis_libero_30k/r6_execute4_official_trial_20260916_attempt_0001/official_result_user_reported.json`。
分数来源为用户报告，未提供官方分项明细或 submission id，因此不把上述解释写成已证因果。
