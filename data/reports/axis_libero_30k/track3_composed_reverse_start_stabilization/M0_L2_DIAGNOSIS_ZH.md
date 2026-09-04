# Track 3 composed reverse-start M0 诊断

M0 只读锁定上一轮 matched-terminal L0–L5 证据，没有启动 simulator、CPU worker 或 GPU。

- 上一轮 9/9 same-demo model/state identity、forward success、reverse source/goal 语义证据已逐条复算；失败 gate 按 run 完整列出。
- task20 的历史失败主要是 success 末帧后的物体沉降，task19 主要是 forward terminal robot 回稳；positive reward suffix 均只有一帧。
- `contact_contract_v2` 将 operator↔robot/gripper 误归为 background，本计划改用 contact-v3。
- 本轮 composition 只生成 report namespace 中的 candidate，不改变公开 HDF5、BDDL 或历史 evidence。

M0 的固定 demo 顺序为 task10/task20/task19 各 demo_0、demo_1、demo_2；predicate 仍只在加载同一 model XML 的 runtime 中求值。
