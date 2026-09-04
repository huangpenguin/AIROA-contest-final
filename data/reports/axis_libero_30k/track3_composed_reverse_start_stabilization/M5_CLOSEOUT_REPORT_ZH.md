# Track 3 composed reverse-start M5 收尾

状态：`track3_composed_reverse_start_validated_pending_executor_review`

## 结果

- M0/M1 通过；上一轮 matched-terminal evidence 与三份公开 HDF5 的九条固定 demo 已锁定并复核。
- M2 task20/demo0 fresh CPU canary 通过；M3 固定九条 CPU matrix 为 `9/9`，三族各 `3/3`。
- M4 GPU3 camera canary 按 task10→task20→task19 各一次，三条均通过；每条 agentview/wrist 为 41 帧、H.264、20 FPS。
- composition 使用同 demo terminal scene qpos、同 demo init robot/gripper qpos、全零 qvel；hold 为 `float32[7]` 全零。
- `start_state_candidate` 仅保留在 report namespace，数量 `1`；`exact/pending/accepted reverse demo = 0/0/0`。

## 边界

没有修改公开 HDF5、BDDL 或历史 evidence；没有生成 manipulation action/demo，没有 SE(3)/grasp、Teleop、训练、norm、FAST/ZIP、官方 GPU、上传或提交。M4 完成后已 hard stop，后续必须等待新的审核计划。

## 校验

- full tests：`223 passed, 7 skipped, 23 warnings`，returncode `0`
- JSON parse：`passed`；py_compile：`passed`；git diff check：`passed`
- root/OpenPI 与 `LIBERO-plus` 的最终 porcelain 由交接文件记录；`LIBERO-plus` 既有 dirty 状态未触碰。

已 hard stop，等待人工审核。
