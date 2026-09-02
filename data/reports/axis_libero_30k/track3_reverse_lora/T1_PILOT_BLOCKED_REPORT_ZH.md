# T1 reverse compiler / pilot 执行报告

状态：`track3_reverse_rules_or_pilot_blocked`

## CPU compiler

- 公开 canonical LIBERO 任务目录：`40`
- 由公开 BDDL 的 forward `:init` / `:goal` 编译的候选：`40`
- template split：`32 train / 8 template-holdout`，holdout `20%`
- operator families：containment `12`、support `17`、sequence `9`、articulation `1`、activation `1`
- manifest：`t1_reverse_task_manifest.jsonl`
- manifest SHA-256：`41bbea22eedc5cef9f23e7f0b6bba126f488f5d1e4ba4d3b4fe7b45a75e36856`

这些是候选 reverse specs，不是 accepted demonstrations。编译器只读取公开 LIBERO catalog、公开 BDDL/init 和 T0 已锁定的审计输入；没有读取 hidden Track 3 任务、数量、family、命名或生成机制。

## GPU3 真实环境探测

已在 GPU3、`control_freq=20` 运行受限真实 simulator smoke；GPU0 只做只读观察，GPU1/2 没有启动训练。

1. custom reverse BDDL reset 可以创建环境，action contract 为 `7D OSC_POSE`，初始 `check_success=false`。
2. task10 support reverse 的 object-on-plate 候选在数个零动作后从 plate 落到 table；该状态不满足稳定 reverse initial gate，未接受。
3. task17 support reverse 使用动作-only motion primitive，未形成抓取并未达到 reverse goal；未接受。
4. 公开 episode0 的 robot EEF 首帧与 `.pruned_init[0]` 位置最大差约 `0.02177m`；50 个公开 init 中最近 index14 仍约 `0.00229m`。用该候选做 forward action replay 没有产生可验证 terminal success，因此没有 terminal snapshot 可供 reverse 使用。
5. 先前 forward replay warmup 也保持 `success=false`。所有逆序/取反动作仅作为探测 proposal，从未写入 accepted dataset。

## 门禁结论

当前 accepted episode 为 `0`，未满足 `30–60 episodes`、至少 `3 families`、`6 templates`、`2 scenes`、每 family 两 template，以及每条 `3/3 replay success` 的 T1 gate。原因是现有公开 on-disk 数据没有可直接复用的 full simulator terminal qpos；直接用 BDDL 交换生成的 object-on-object 状态也未通过稳定性 gate。

因此本轮不生成 reverse training dataset，不启动 T2、GPU1/2 LoRA、T3 holdout、FAST、ZIP、closed-loop、E2、C20、训练、比赛 GPU、上传或提交。R6 raw/folded/FAST/正分 ZIP 均未修改。

下一状态固定为 `track3_reverse_rules_or_pilot_blocked`，等待新授权/新计划提供可审计的公开 terminal-state 获取与 teleop/planner 路径；不得把本报告中的候选 specs 当作训练数据。
