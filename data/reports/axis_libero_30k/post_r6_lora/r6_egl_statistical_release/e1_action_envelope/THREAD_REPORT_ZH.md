# E1 EGL Action-Response Envelope

状态：`e1_jax_full_inference_failed`

本 section 按 E1 hard gate 在 `r6_folded_jax_source` 的 `full` 阶段停止：folded JAX full run hit V100 CUDA OOM while microbatch=4; allocation request 216.00 MiB failed。没有重试 OOM，没有启动另一 backend，因此没有把不完整结果写成 170-pair envelope。

## 已完成

- E1.0–E1.1：CPU-only lock，`170 pairs / 340 observations`，snapshot/trajectory `13/157`，lock 中 `actions_observed=false`。
- E1.2：JAX sentinel `4 observations` 通过，输入/model/norm provenance 与 GPU boundary 正常。

## 未执行

- E1.4 LeRobot/PyTorch full inference：未执行。
- E1.5 full comparator：未执行。
- closed-loop、E2、FAST、ZIP、训练、比赛 GPU、push、上传：未执行。

完整失败证据见 `summary.json`；E1.6 只完成失败记录、测试和交接，不宣称 envelope 通过。等待人工审核/最小 remediation 计划。

本 section 已 hard stop。
