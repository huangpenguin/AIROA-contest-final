# R2 本次交接（2026-10-04）

本夜探索已收尾。任务 `R2_NIGHT_VERIFY_20261003T183641Z`，round 2；共享 `done` 仅表示本包结束，**G1 / 正式训练仍未通过**。182 条人工检查已接收，不用重看。

已完成：

- 复用过滤后的 19,493 条候选记录、隔离的 40 条和 train-only norm；此前 4 条完整 episode → 39 个正式预处理 batch → 离线日志的 CPU 交付保留。
- 真实模型连续 4 步、独立分段 2 步完成；前两步一致。NEW Adam 第一步 36 个 moment 计算通过独立核对。
- W&B 离线 SDK 实际记录连续 4 条、分段 2 条、失败恢复 0 条；代码、数据、配方、step 和前后段关联已核对，没有上传。
- continuous step 4 的诊断导出、原容差下 JAX/PyTorch 动作对比、FAST 加载、HTTP 重置一致性通过。
- fresh simulator 完成 80 次策略请求和 90 次环境步；HTTP 动作与实际执行动作完全一致、轨迹有限。**未达成任务，碰撞诊断为 true**。这只证明接口跑通，未评估成功率，也不是 combined 原 episode 的精确回放。

还未完成：

1. 物理来源与 split：缺作者的 episode → 原始 dataset/task/demo、增强父记录映射。当前数值分组不能证明物理示教隔离；正式派生数据与 norm 的准入仍取决于此。
2. 完整模型/Adam 恢复：恢复到 step 2 后，两卡的 `micro_rng` 在 forward 前不一致。小算子已间歇性精确复现；两种候选改法各只通过一次，不能认定修复。成功恢复后的 W&B 指标接续也尚未验收。
3. 恢复成功后的模型仍须重新绑定导出与闭环验证；此次诊断产物明确保留 `model_restore_passed=false`。combined 缺 simulator 初态，保持 `exact_replay_unavailable`。

下一次按[计划](/home/huang/code/PARC2026_final/docs/r2/PLAN.md)先做 RNG 小算子的成对验证，再做无更新的恢复边界检查，最后才考虑一次完整恢复短测。物理映射可独立推进；已有公开入口查找已到界，不重复盲搜。本夜不再追加运行，不自动延伸长训。

实际资源：18:36:41–23:48:34 UTC 执行与交付，约 5 小时 12 分；主审随后收尾。新增 4 次模型尝试、7 次真实 optimizer 更新、9 个已占用更新槽；原 U1 的 2 次/4 更新另外保留。实测 GPU 窗口 4,668.823 秒、2.172804 GPUh；另保守计入未测量的 EGL 初始化 90 秒/0.1 GPUh。产物快照约 134.007 GiB，加少量收尾文档。所有本轮运行进程已退出，GPU 2/3 锁已释放；4 个历史进程保留。官方算力、上传、比赛打包/提交均为 0。

接续只需读取本页及[最终审核](/home/huang/code/PARC2026_final/outputs/r2/agent_handoff/R2_NIGHT_VERIFY_20261003T183641Z.round_0002.review.json)、[最终交付](/home/huang/code/PARC2026_final/outputs/r2/agent_handoff/R2_NIGHT_VERIFY_20261003T183641Z.round_0002.output.json)。代码与完整 nested OpenPI 补丁的恢复入口在[源码清单](/home/huang/code/PARC2026_r2_overnight_restore/outputs/r2/night_verify_20261004/source_B6_final_sealed/MANIFEST.json)；只 checkout 根 commit 不够。[旧交接原文](/home/huang/code/PARC2026_final/outputs/r2/review_r2_night_verify_20261003t183641z/before_final_closeout_docs/MAIN_HANDOFF_20261004.md)完整保留。
