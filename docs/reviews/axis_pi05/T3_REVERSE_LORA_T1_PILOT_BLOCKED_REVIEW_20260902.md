# Track 3 T1 reverse compiler / pilot 审核记录

日期：2026-09-02（Asia/Tokyo）
审核状态：`blocked`；执行状态：`track3_reverse_rules_or_pilot_blocked`

## 输入与边界

- 依据活动计划：`docs/AXIS_PI05_TRACK3_REVERSE_LORA_POSITIVE_SCORE_PLAN.md`
- T0 lock：`data/reports/axis_libero_30k/track3_reverse_lora/t0_cpu_lock.json`
- R6 raw/folded/FAST/正分 ZIP 均保持锁定，ZIP 未修改
- 只读取公开 canonical LIBERO catalog、公开 BDDL/init；没有使用 hidden Track 3 task、数量、命名、family 或生成机制
- GPU0 仅做只读状态观察；GPU3 仅运行受限 simulator probe；GPU1/2 未启动训练

## T1 结果

CPU compiler 将公开 40-task catalog 生成 `track3_reverse_pilot_v1` 候选，template split 为 `32 train / 8 holdout`（20%），candidate manifest SHA-256 为：

`41bbea22eedc5cef9f23e7f0b6bba126f488f5d1e4ba4d3b4fe7b45a75e36856`

候选 family 计数为 containment `12`、support `17`、sequence `9`、articulation `1`、activation `1`。它们是 BDDL 候选 spec，不是 demonstration。

GPU3 的真实 20Hz probe 没有产生任何 accepted episode：

- task10 object-on-plate reverse initial 在零动作后失稳；
- task17 action-only motion primitive 没有形成抓取；
- 公开 episode0 的 forward init 与 on-disk robot state 对齐不一致，最近公开 init 仍有约 `2.29mm` EEF 位置差；候选 replay 没有产生可验证 terminal success。

因此 accepted episode=`0`，不满足最低的 30–60 episodes、3 families、6 templates、2 scenes、每 family 两 template 和逐条 3/3 replay gate。不能把合成 reset 或未成功的动作 proposal 记录为 reverse supervision。

## 后续限制

T2、GPU1/2 两卡 LoRA、T3 holdout、FAST、ZIP、closed-loop、E2、C20、训练、比赛 GPU、上传和提交均未执行。状态固定为 `track3_reverse_rules_or_pilot_blocked`，等待新计划明确提供合法且可审计的公开 terminal-state 获取与 teleop/planner 路径。

证据：

- `data/reports/axis_libero_30k/track3_reverse_lora/t1_cpu_compile_summary.json`
- `data/reports/axis_libero_30k/track3_reverse_lora/T1_CPU_COMPILER_REPORT_ZH.md`
- `data/reports/axis_libero_30k/track3_reverse_lora/T1_PILOT_BLOCKED_REPORT_ZH.md`
