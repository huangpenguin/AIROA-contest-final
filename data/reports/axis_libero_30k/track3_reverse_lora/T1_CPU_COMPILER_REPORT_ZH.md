# T1 CPU-only 公共 reverse 编译报告

状态：`track3_reverse_rules_or_pilot_blocked`（编译完成；pilot 未通过）

公共任务：`40`；编译通过：`40`；静态拒绝：`0`。

template split：train `32` / holdout `8`（`20.0%`）。

operator families：`{"activation_inverse": 1, "articulation_inverse": 1, "containment_remove_restore": 12, "sequence_reverse": 9, "support_remove_restore": 17}`。

这些只是由公开 BDDL 的 init/goal 交换生成的候选规格，不是 accepted demonstration；尚未启动 simulator、CUDA 或训练。

下一门禁：GPU3 真实 20Hz pilot，必须达到 30–60 条且每条 3/3 replay predicate success；当前 accepted episodes 为 `0`。

manifest SHA-256：`41bbea22eedc5cef9f23e7f0b6bba126f488f5d1e4ba4d3b4fe7b45a75e36856`
