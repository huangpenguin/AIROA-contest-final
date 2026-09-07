# AXIS Pi0.5 Track 3：SR0–SR13 独立审核计划

> 创建日期：2026-09-07（Asia/Tokyo）
> 唯一活动计划：否（已归档）
> 审核对象：SR0–SR13 执行提交、`5c5c746` clean source 与新 SR evidence namespace

本计划只授权只读独立审核。不得启动新 physics、Teleop、GPU replay、generation、accepted promotion、训练、norm、FAST/ZIP、upload 或 submission。旧 SG/CR/BR/AC identity 与本轮 attempt `0001–0004` 均不可覆盖或重跑。

## A0. 输入与执行边界

- 核对 root HEAD、branch/upstream、nested OpenPI、LIBERO-plus 原 dirty digest、SR clean-source commit `5c5c746` 和 tool blob SHA。
- 验证 SR0–SR5 的 attempt 单调、两次 pre-env failure 的原始记录、最终 release pointer 与 clean-source identity。
- 检查 live PID/supervisor/cursor/heartbeat、GPU0–3 compute、`.partial`、root/OpenPI clean；若有 live worker，只观察不碰撞。

## A1. Code/fixture 与 pre-env 审核

- 复核 contact role、dwell acquisition window、lift 后 source release、goal continuous window、stop classification 与 positive/negative fixture。
- 验证 prepare-only 确实使用真实 HDF5/model/state/anchor，且未 import/create `ControlEnv`、未运行 physics。
- 复核 SR5 attempt `0001` 的 tool-SHA failure、attempt `0002` 的 git porcelain failure 和 attempt `0003` pass，不把前两次当 physics identity。

## A2. SR6/SR7 raw 与 family classification

- 对两条 child identity/command/returncode、typed NPZ no-pickle/finite、start state、stage sequence、predicate/contact arrays、action saturation 与 hashes 独立复算。
- 确认 acquisition window 仅看最后四个 grasp-dwell 样本且 required contact 全 false；lift/place/goal 保持 `not_evaluated`。
- 核对 SR7 只因 SR6 为 normal family failure 且达到预冻结 quantified-progress 条件而启动；两条策略/trace/action 必须结构不同且 hash 不同。

## A3. 人工、数据链与禁止项

- 确认 SR8 为 `human_input_required`，agent 未产生键盘输入，Teleop 未开始。
- 验证 SR12 没有把失败 trace 升级、没有 accepted demo 时 loader/norm/training fail-close。
- 检查 GPU/replay/generation/accepted/training/ZIP/upload/submission 均未发生。

## A4. Finalizer、测试与生命周期

- 验证 producer manifest、final manifest、outer hash、closure、current pointer、inactive cursor/final heartbeat 与无 partial。`sr13_closeout.json.not_evaluated` 的空列表不得被解释为 SR9–SR11 已运行；以 raw 文件和顶层 summary 做独立纠正。
- duplicate-key rejection 扫描全部 SR JSON；复跑 targeted、项目规范全量、py_compile、`git diff --check`。
- 输出唯一 `approved` / `remediation_required` / `blocked` 结论；保留旧证据不改写，归档本审核计划，只建立一个新 active plan，并同步 index、Markdown/machine handoff、MEMORY 与 lifecycle tests，clean commit 后 hard stop。
