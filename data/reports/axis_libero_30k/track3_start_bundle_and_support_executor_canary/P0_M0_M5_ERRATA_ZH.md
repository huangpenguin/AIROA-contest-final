# Track 3 Start Bundle P0 Errata

P0 只读锁定 `eef683b` 输入及旧 M0–M5 产物，没有启动 simulator 或 GPU。

旧 M4 的核心物理结论已保留；本轮只修复其工程闭包：逐进程 seed/PID/显存 provenance、step0–40 全窗口 contact gate、可恢复 NPZ state、artifact relative path/bytes/SHA-256，以及 `composed_vs_init_max_abs_diff` 与真实 restore diff 的命名边界。旧报告不原地改写。

P0 的 raw reanalysis 同时保存全部 failed/not_evaluated gate，不能用首个失败字段覆盖其余诊断。reverse demo counts 仍为 `0/0/0`。
