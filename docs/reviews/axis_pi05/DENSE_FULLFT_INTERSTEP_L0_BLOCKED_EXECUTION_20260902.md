# Dense full-FT 跨 optimizer step 生命周期执行记录

日期：2026-09-02（Asia/Tokyo）
执行输入：root `35e67b6`，nested OpenPI `b58aff0`
执行结论：`blocked_external_gpu`

## 1. 已完成门禁

- L0 CPU 语义锁定完成：`restored_internal_step=1`、update transition `1→2`、失败发生于 `optimizer_step_before=2` 的 microbatch `1`；D3 step1 source tree hash 为 `9aeeb1a9b1fdf428486b476930ca9117c28d822dd108b4efaeebf911ca0fafef`。
- L1 CPU/code gate 完成：gradient 与旧 state 一起 donate，device metrics 先 materialize 为 host scalar，再释放 device-side 临时引用；transition probe 和阶段语义解析测试通过。
- 全量测试：`146 passed, 7 skipped, 23 warnings`。

## 2. 资源 hard gate

L0 的只读 GPU snapshot 发现 GPU0 并非自然空闲：外部 PID `1133706` 正在执行 YOLO 训练，进程为 `/mnt/home/moriyama/workspace/ros2_project/mmdetection/.venv-yolo26/bin/python`，占用约 `14.3 GiB`，GPU utilization 非零。GPU1–3 各约 `5 MiB`，但计划要求一次使用完整本地 4×V100，且明确不得触碰外部 PID。

因此 lock 状态为 `dense_fullft_interstep_l0_blocked_external_gpu`。未启动 L2 transition、L3 step3 或 L4 step10；没有终止、暂停或修改外部任务，也没有启动 Blackwell/官方 GPU、C20、长训、FAST/ZIP、上传或提交。

## 3. 停止

当前只等待 GPU0 外部 PID 自然释放。释放后必须重新检查四卡 PID/显存，再从同一 L0 lock 进入 L2；在此之前不得运行任何 GPU 训练或用部分 GPU 绕过门禁。

证据：`data/reports/axis_libero_30k/dense_fullft_interstep_state_lifetime_recovery/l0_cpu_lock.json`。root/nested 工作树保持 clean。
