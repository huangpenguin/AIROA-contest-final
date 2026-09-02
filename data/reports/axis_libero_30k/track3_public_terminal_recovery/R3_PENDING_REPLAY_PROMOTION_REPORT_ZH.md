# Track 3 R3 pending/replay/review promotion 报告

状态：`passed_implementation_only`

人工 candidate 的唯一写入路径是 `pending/`。`tools/axis_pi05/track3_public_replay.py --verify` 会为同一 candidate 启动三个 fresh process，逐帧检查 7D finite action、locked full start state、reset goal false 与最终 reverse predicate true，并写 `replay_summary.json`；任何 `2/3` 或异常都保持 `replay_failed`。

`--promote` 只有在 replay `3/3` 且 `human_review.json` 明确包含 operator、instruction/object、无 teleport/trimming、front/wrist 确认和批准时间时，才将 pending episode 目录 atomic rename 到 `accepted/`，并保留 `promotion.json` provenance。测试验证了 predicate/replay、人审和 `2/3` 三种门禁均不会提前进入 accepted。

本轮没有人工输入，因此没有运行 replay candidate、没有 human review，也没有 promotion；accepted demonstration 仍为 `0`。
