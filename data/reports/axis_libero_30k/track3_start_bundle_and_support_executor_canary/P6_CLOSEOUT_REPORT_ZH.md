# Track 3 Start Bundle / Support Executor P6 收尾

状态：`track3_support_executor_canary_failed_worker_result_publication`

## 已完成

- P0 输入锁通过；旧 M0–M5、三份公开 HDF5、9 个 composed hash 与 `0/0/0` reverse-demo 计数未漂移。
- P1 独立 contract 通过；P2 物化并 fresh CPU restore `9/9`，task10/demo0 固定抓取几何提取通过。
- P3 只启动了固定顺序第一条 task10/demo0 GPU3 canary。41 帧 camera 已封存并完成 H.264/20 FPS/ffprobe 闭合。

## 停止原因

P3 worker 在写出完整 checkpoint JSON 时填满父子 stdout 管道，阻塞于结果发布；该次 task10 canary 被终止并记录为 `worker_result_publication` 工具失败。按照计划没有重试 task10，也没有启动 task20/task19、P4 或 P5。新代码已改为文件重定向 stdout，并将后续 checkpoint 输出压缩，供下一份审核计划决定是否允许恢复。

## 边界与计数

没有操控动作、SE(3) executor、demo 生成、pending promotion、Teleop、训练、norm、FAST/ZIP、官方 GPU、上传或提交。`validated_start_state=9`，`exact/pending/accepted=0/0/0`；P4/P5 均 `not_evaluated`，pending 目录未创建。

## 校验

- targeted tests：`8` passed，returncode `0`；full tests：`231` passed，`7` skipped，`23` warnings，returncode `0`。
- NPZ no-pickle：`passed`；旧输入 hash：`passed`；JSON strict parse：`passed`；video ffprobe：`passed`；py_compile：`passed`；git diff check：`passed`。
- root/OpenPI 与 `LIBERO-plus` dirty preservation 记录于 `p6_summary.json`；P6 后 hard stop，等待下一次审核。
