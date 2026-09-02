# Track 3 公开终态恢复 P0–P2 执行报告

日期：2026-09-02（Asia/Tokyo）

状态：`track3_teleop_input_pending_human`

## P0：官方公开 HDF5 锁定

- 四个公开 suite 各取 10 个与锁定 40-task catalog 对应的原始 HDF5。
- 文件数 `40`；总 bytes `33784856577`；`.partial` 文件 `0`。
- `data/demo_0/states`、`actions`、`rewards`、`dones` 在 40/40 文件中均存在。
- task/scene/BDDL 映射 40/40 通过；reverse BDDL 由当前官方 LIBERO BDDL 与对应 `.pruned_init` 在 CPU 上重新编译，未复用旧 mixed-tree reverse BDDL。
- P0 lock：`p0_lock.json` SHA-256 `06c2d8deb5fbd9363f9ca9cb3412b15d761dbe42940b5909a83f6565bae0e725`。

## P1：完整终态 canary

task 10（support）、task 20（containment）、task 19（articulation）各以官方 HDF5 完整 state 完成 fresh-process `3/3` restore；forward success、20 帧 settle、reverse initial-not-success 均通过。accepted snapshot 数为 `3`，但仍标记为 `pending_video_review`，不是 accepted demonstration。

activation task 16 的独立证据保留：官方 HDF5 state 与公开 demo action 的 live capture 两种来源均在 restore 后 `forward=True`，但 20 帧 no-op settle 后变为 `False`，因此未计入 canary。原始失败摘要：`p1_activation_task16_failed_summary.json`。

## P2：recorder 与人工输入停止点

已实现公开 LIBERO `OSC_POSE` keyboard-first recorder：20 Hz 单调时钟、7D finite action、`{-1,+1}` gripper、pause/abort/save/reset、完整 per-frame state、front/wrist image、timestamp/deadline miss、start/end state、H.264 优先视频封装与 accepted/rejected 分离目录。

CPU self-test 和 py_compile 通过；全量测试为 `155 passed, 7 skipped, 23 warnings`。当前没有人工输入，因此没有生成 accepted/rejected episode，也没有启动 P3 planner、P4 扩充或 P5 loader smoke。

人工启动命令（仅在 GPU3、仅用于公开 task 10/19/20 的 teleop）：

```bash
cd /home/huang/code/PARC2026_final
PYTHONPATH=LIBERO:. CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3 MUJOCO_GL=egl \
  .venv_axis_pi05/bin/python tools/axis_pi05/track3_public_teleop.py \
  --record --task-index 10 --operator <operator-name>
```

每次人工录制结束后必须停止并由审核确认 predicate、回放和视频；不得用随机动作、primitive 或旧失败轨迹替代人工 canary。

## 资源与禁止项

GPU0 外部任务未触碰；GPU1/2 未训练；P0 为 CPU/网络；P1 仅使用 GPU3 simulator。没有启动 LoRA/dense 训练、FAST、ZIP、C20、E2、官方 GPU、上传或提交。`LIBERO-plus` 的既有 dirty 状态保持原样。
