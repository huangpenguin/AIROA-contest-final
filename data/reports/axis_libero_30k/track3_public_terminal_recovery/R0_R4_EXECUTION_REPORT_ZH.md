# Track 3 Teleop Recorder R0–R4 执行收尾报告

日期：2026-09-03（Asia/Tokyo）

最终状态：`track3_teleop_ready_for_human_after_remediation`

输入锁定：root `07de201`（计划记录的 `9dac308` 直接后继）、nested OpenPI `b58aff0`。P0/P1 原始 evidence 保持不变；P0 lock 实际 SHA-256 为 `be12795b3d18d46f774022d817789a14d64a5e16c8e27bee58e256c29acb4ab5`，P1 summary SHA-256 为 `62db697a1ff6afb5d157c9fc786c77eb17a0bdf1054edb060fea50b3f3fe211e`，activation task16 仍排除。

## 门禁结果

- R0 correction/hash closure：通过；历史 stale hash 仅记录，不改写历史 evidence。
- R1 action/reset：通过；CPU 覆盖 50 次 no-key、合法 gripper、pause、双 toggle、camera、reset、save、abort、非法 action。
- R2 preview：通过；仅 `127.0.0.1`，front/wrist 首帧 HTTP gate 与断连安全控制已实现。task10/19/20 各保留 20 帧、20Hz 实际 review asset、codec/hash 与 contact sheet 见 R4 JSON。
- R3 promotion：实现并通过门禁测试；candidate 只能先入 `pending/`，fresh-process replay 3/3 与 human review approved 后才 atomic promotion。此轮没有 candidate/replay/human review，accepted 为 `0`。
- R4：GPU3 非人工 canary 按 `task19 → task10 → task20` 通过；三个任务 reverse goal reset false、20 idle ticks、front/wrist preview 首帧 HTTP 200、gripper 合法，task19 的 motion/pause/camera/reset/abort 通过。R4 原始结果见 `R4_GPU3_NONHUMAN_CANARY.json`。

## CPU/静态校验

- 项目虚拟环境 targeted pytest：`5 passed`。
- 项目虚拟环境全量 pytest：`158 passed, 7 skipped, 23 warnings`。
- 主环境 `python -m pytest`：blocked，环境无 pytest 模块；项目虚拟环境的 pytest 可用。
- 三个工具 `py_compile`：通过。
- R0/R4/handoff/P0/P1 JSON 解析：通过；`git diff --check`：通过。

## 资源与禁止项

R4 只使用物理 GPU3；canary 前后均为 `5 MiB / 32768 MiB`，没有人工输入。GPU0 外部任务未触碰，GPU1/2 未训练；没有旧 teleop、P3/P4/P5、LoRA/dense、FAST、ZIP、C20、E2、官方 GPU、上传或提交。`LIBERO-plus` 原有 dirty 状态未修改、stash、reset 或提交。`accepted/` 与 `pending/` 均不存在，未生成 accepted episode。

## 人工就绪信息（本轮不执行）

固定命令：

```bash
cd /home/huang/code/PARC2026_final
PYTHONPATH=LIBERO:. CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3 MUJOCO_GL=egl .venv_axis_pi05/bin/python tools/axis_pi05/track3_public_teleop.py --record --task-index 19 --operator operator-local --preview-port 8765
```

预览：`http://127.0.0.1:8765/`；远程 tunnel：`ssh -N -L 8765:127.0.0.1:8765 huang@gpu01`。键位：W/S X、A/D Y、R/F Z、U/O Rx、I/K Ry、J/L Rz、G gripper、P pause、C camera、Z reset、Enter save、X abort。顺序只能是 task19→task10→task20。
