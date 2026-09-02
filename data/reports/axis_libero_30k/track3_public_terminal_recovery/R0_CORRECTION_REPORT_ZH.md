# Track 3 R0 证据与状态修正报告

日期：2026-09-03（Asia/Tokyo）

状态：`passed`

计划文件声明的 root 输入是 `9dac308`，本次实际执行 agent 的 root 输入是其后已审核的 `07de201`；二者关系与 nested OpenPI `b58aff0` 已锁定在 machine-readable summary。历史 P0/P1 evidence 未重写。

- P0 lock 实际 SHA-256：`be12795b3d18d46f774022d817789a14d64a5e16c8e27bee58e256c29acb4ab5`；历史报告中的 `06c2d8...` 只作为 stale 值保留。
- P1 summary SHA-256：`62db697a1ff6afb5d157c9fc786c77eb17a0bdf1054edb060fea50b3f3fe211e`，task 10/19/20 三个 snapshot 仍是 `pending_video_review`，不是 accepted demonstration。
- activation task16 失败摘要未改写，也未计入成功 family。
- accepted demonstration 在 R0 为 `0`；`LIBERO-plus` 既有 dirty 状态未修改、stash、reset 或提交。

机器可读闭环见同目录 `R0_CORRECTION_SUMMARY.json`。
