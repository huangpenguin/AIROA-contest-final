# Track 3 R0–R6 收尾

状态：`track3_p3_recovery_parent_provenance_failure_hard_stop`。

R0 已完成旧 17 项 evidence closure；R1 已通过独立 atomic result JSON、16 MiB stdout、hang TERM/KILL、malformed/nonzero/exception fail-closed。R2 只启动一次 task10/demo0、seed 3100、attempt 2。child 结果中的 restore、source/goal、contact-v3、camera 41 帧均通过，但 parent wrapper 在发布 GPU provenance 时发生 `NameError`，缺少启动快照、physical UUID 与 peak 显存，故总体按独立 result 合同判定 `failed_closed`。

因此 R3、R4、R5 均未执行；没有 task20/task19 canary、SE(3) executor、generation、replay、pending 或训练。`exact/pending/accepted = 0/0/0`。未修改旧 HDF5/BDDL，未 Teleop，未使用 GPU0/官方 GPU/FAST/ZIP。

本轮 hard stop：如需继续，必须由审核后的新计划修复 parent provenance assembly；本入口不自动恢复或重试 attempt 2。
