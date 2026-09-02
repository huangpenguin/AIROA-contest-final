# Track 3 T1 pilot 阻塞与终态来源独立审核

日期：2026-09-02（Asia/Tokyo）
审核输入：root `1e71ec5`，nested OpenPI `b58aff0`
审核结论：`blocked`（只阻塞旧 pilot；允许进入公开终态恢复计划）

## 结论

T1 按 hard gate 停止是正确的。40/40 CPU 编译结果只证明公开任务能转换成候选 reverse spec；它不证明 reverse initial state 物理稳定，也不等于 40 条 demonstration。GPU3 得到 `0 accepted episode`，因此不能启动 LoRA。

本次失败没有否定 reverse supervision 或 LoRA 路线。它否定的是旧的起始状态/动作 proposal：

- `.pruned_init` 是 benchmark reset 候选，不是公开 demonstration 的成功终态；
- LeRobot 的 8D observation state 不是完整 MuJoCo simulator state，不能据此恢复 object、fixture、robot 与 controller 的共同状态；
- 交换 BDDL init/goal predicate 不会自动构造物理稳定的 object-on-support/containment 状态；
- action-only primitive 未抓取成功，不能记为 demonstration；
- 最近公开 init 仍有约 `2.29 mm` EEF 差异，且 replay 未成功；不得放宽该失败或 `3/3 replay` 门禁。

## 可审计的公开恢复路径

官方 LIBERO 公开原始 demonstration HDF5 下载方式；其官方 `check_task_suites.py` 会读取 `data/demo_0/states` 并用 `env.set_init_state(states[-1])` 检查 demonstration 终态。当前仓库只有转换后的 LeRobot/HF 数据，未发现原始完整 simulator-state HDF5。因此下一步先下载并锁定官方 HDF5，而不是继续从 8D state 猜完整状态。

允许两种终态来源，且必须分别标记 provenance：

1. `official_hdf5_terminal_state`：官方公开 HDF5 的完整 `states`；
2. `live_public_forward_success_capture`：当前锁定 LIBERO 环境中，R6 对公开正向模板实际成功时现场保存的完整 `sim.get_state().flatten()`。

来源 1 若因 LIBERO/MuJoCo/XML 版本不匹配而不能 exact restore，必须 fail closed；允许转向来源 2，不允许只手工移植 object qpos 或把 mismatch 当 tolerance。

## 审核边界

- 旧 T0/T1 计划归档；其 manifest、失败视频/日志与 hash 保留。
- 不使用 `check_no_leaks.py` 的内部内容、hidden Track 3 task 或未公开 task 数量。
- `/home/huang/code/PARC2026_final/LIBERO-plus` 的既有 dirty 状态保持原样，不作为本轮实现目录。
- 新计划只做到公开终态、teleop/planner canary、30–60 条 pilot 与 loader smoke；不授权 LoRA、dense、FAST、ZIP、上传或提交。
- GPU0 不得触碰；CPU/网络用于下载，GPU3仅用于 simulator/公开正向捕获，GPU1/2不用于训练。

## 已核对事实

- T1 compiler：40 public candidates，32 train / 8 holdout；manifest SHA-256 `41bbea22eedc5cef9f23e7f0b6bba126f488f5d1e4ba4d3b4fe7b45a75e36856`。
- pilot accepted：`0`；T2 及训练均未启动。
- 测试：`153 passed, 7 skipped, 23 warnings`。
- root 与 nested OpenPI clean；独立 `LIBERO-plus` 目录的既有修改未被触碰。
