# S0 R6 folded release lock audit

状态：`r6_folded_release_locked_not_yet_tested`

- deployment candidate：`r6_step500_lerobot_folded`
- deployment backend：`lerobot_folded_pytorch`
- source model：`5c73e1e8217dbb75a934c2b0165208e851f69e66cfe2ea0430590a29901e6d5b`
- R6 fresh norm：`4c803bf59c52677c36b6b420c614d8a7742e0847708443863e13e67cb80b18f5`，ACTION/STATE=`QUANTILES`
- raw-LoRA research Q2：`n1_q2_policy_sensitivity_failed_raw_lora_fresh_replay`，失败最大差 `1.0728836059570312e-06`，保留且不可覆盖
- deployment runtime：尚未测试；没有生成 `model_fast.safetensors`
- formal winner：`false`
- auto submit：`false`

S0 只锁定输入与 provenance，不宣称 LeRobot/PyTorch、FAST、server 或 ZIP 通过。旧 F2/M1 FAST、processor、norm 被列入拒绝名单，未作为 R6 输入。S1 将实际从磁盘 source model/processors 加载并验证。

机器锁定：[R6_FOLDED_RELEASE_LOCK.json](/home/huang/code/PARC2026_final/data/reports/axis_libero_30k/post_r6_lora/r6_positive_score_fast/R6_FOLDED_RELEASE_LOCK.json)
