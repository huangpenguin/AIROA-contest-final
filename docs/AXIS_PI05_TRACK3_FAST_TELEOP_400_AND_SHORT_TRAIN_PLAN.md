# AXIS Pi0.5 Track3 快速 Teleop 400-step 与短训计划

> 创建日期：2026-09-09（Asia/Tokyo）
> 唯一活动计划：是
> 前置审核：`docs/reviews/axis_pi05/TRACK3_MG0_MG1_FAST_COLLECTION_AUDIT_20260909.md`
> 前置审核结论：`remediation_required`
> 审核输入：`36e8f269ecd30d4a2a6e555e67bd7315011e5dbf`
> 当前状态：`FT0_ready_to_implement_fast_recorder`
> 下一阶段：`FT0_implement_fast_recorder`

本计划只解决最后一周最重要的问题：快速获得少量真实 reverse gold，立即短训并尽快形成可提交候选。旧 MG0/MG1 证据保持不变；不再用复杂 attempt 配额和每次失败 hard stop 阻塞人工采集。

## 1. FT0 录制器实施要求（尚未执行）

- direct-gold 长度上限从 `300` 改为 `400`。到 400 不退出；401 起 HUD 标记 `postprocess_required`。
- 默认 operator rate 从 4 Hz 实际录制基线提高到 `8 Hz`，墙钟移动速度约 2×；MuJoCo control/video 仍独立记录为 20 Hz。
- UI 提供 `0.5×–2.0×` 动作幅度滑杆，默认 `1.0×`；建议先用 `0.75×–1.0×`，长程搬运可临时调高，抓取/放置再调低。
- Reset 可无限次使用：每次回到同一冻结 start state、清空当前 buffer、重新计步并暂停等待用户继续。Reset 不创建 episode、不消耗正式数据名额。
- Save 时物理门未通过不会退出或写 rejected episode，只暂停并提示；用户可继续或 Reset。只有通过物理门的 Save 才结束并原子保存。
- `<=400` 的成功记录进入 `pending_replay`；`>400` 的成功记录进入 `pending_postprocess`，保留完整 raw。
- 单次 buffer soft cap 为 1600 帧；达到后只暂停，进程仍可 Save、Reset 或 Abort，不作为 gold/失败结论。
- 默认 session timeout 关闭；用户明确 Save 或 Abort 前进程保持。

## 2. 历史 314-step 的即时复用

`official_t3_red_mug_1788921471` 的历史 O1.2 决定永久保持“旧 300 规则下 over-budget、非 gold”。新策略不改旧 metadata，而在新派生 namespace 中允许它进入 fresh exact-action replay：

1. 先在 GPU3 fresh process replay `3/3`；
2. 通过后生成简短双视角审核卡并由用户 accept/reject；
3. 只有 replay `3/3` 且用户 accept 才 promotion 为 accepted/training-eligible。

这一步优先于再录 red mug，最可能最快得到第一条 gold。若 replay 失败，只淘汰这条，不冻结其他题。

## 3. FT1 快速多题目录制

顺序：`official_t3_alphabet_soup` → `official_t3_yellow_book` → 必要时 `official_t3_red_mug` → 可选 `official_t3_two_mugs`。

每个题目只启动一个持续 Teleop session。用户在 session 内可无限 Reset；满意且 HUD gate passed 后 Save。没有按 episode 计的一次失败即停止，也没有旧的 1/1 或 2/2 attempt quota。为了减少审核等待，可先累计 2–4 条 `pending_replay`，再在 GPU3 串行 batch replay。

操作建议：

- 启动后先保持 paused，确认画面、坐标方向、夹爪与 goal box；
- 速度默认 1.0×。接近/夹取/释放可降至 0.5×–0.75×，无遮挡长程运输可升至 1.25×–2.0×；
- 300 帧出现黄警告，350 进入目标窗口，390 红色提醒；400 不杀进程；
- 如果 Save 被阻止，先看 `last_save_blocked_reason`，再继续或 Reset；无需重启服务。

## 4. >400 的后处理边界

允许对完整 raw 派生 training view，但只能做 provenance-aware、obs/action 对齐保持的处理：裁掉明确的起始空等和成功后的尾帧，或把长静止段切成连续子段。不得用样条曲线改 action 后仍配旧 observation/video，也不得改旧 metadata 假装原始轨迹更短。若需要动作平滑，必须用平滑后的 action 从同一 start state 重新执行并生成新 observations。

## 5. FT2 验收与训练门

- 每条 accepted 仍需 machine gate、fresh replay `3/3` 和一次用户视觉 accept；这些是数据质量底线。
- 第一轮 R6 100-step 短训门放宽为至少 `2` 条 accepted、`2` 个 task ID。达到门即冻结 manifest/QUANTILES 并启动，不等额外 family 或第三条。
- 只有 1 条 accepted 时可做 loader/tiny-overfit sanity，但不制作正式候选。
- 100-step 后立即跑最小 reverse dev 与固定 C20 retention；有 reverse `+1` 且 C20 loss `<=2/20` 就当天 fold/FAST/ZIP。
- 100-step 无改善但训练稳定时继续到 300-step；不同时更换模型、rank、LR 和 norm。
- GPU0 禁止。GPU1/2 仅在外部作业自然退出并确认空闲后训练；GPU3 负责 replay/eval。当前 GPU1/2 仍被外部作业占用时，继续采集，不等待。
- 不自动上传或比赛提交；产生有实质变化并通过机械验证的 ZIP 后立刻交给用户决定。

## 6. 简化停止规则

- 只有 raw 覆盖/partial、NaN/Inf、身份或 start-state hash 不明、并发 recorder、GPU 边界不明等基础设施风险才停止整个阶段。
- 抓取失败、goal 未成立、Save 被 gate 阻止、单条 replay 失败都只影响该条/session；继续 Reset 或换题。
- 不为每条 demo 新建大计划、不要求每条采集后跑全量项目测试。录制 raw 的例行检查只做 schema、finite、frame/hash 和 gate；代码或 lifecycle 修改时才按仓库规则跑测试并提交。

## 7. 当前需要用户做什么

当前没有 recorder/MuJoCo 进程。下一次在线操作前先由执行器对 314-step 做 fresh replay；随后你保持 TigerVNC `:3` 在线并回复：

`接管 Teleop：official_t3_alphabet_soup`

进入界面后先做几十步不保存校准；速度过快就把滑杆调到 0.5×–0.75×。正式 session 内可以一直 Reset，不再因为 300/400 或一次不满意操作退出。
