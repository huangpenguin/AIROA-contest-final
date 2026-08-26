# Axis Pi0.5 后训练执行实现状态

日期：2026-08-26（Asia/Tokyo）

## 整改续执行状态

- 当前分支：`codex/axis-pi05-audit-remediation`；整改提交已推进至当前 clean commit `a08e76b`，worktree clean。
- v2 formal run 现在在启动前强制要求 clean Git worktree；启动前已产生的 dirty T1/T2 运行仅保留为诊断参考，不进入正式验收。
- R8 训练环境回归为 `61 passed, 7 skipped`；独立 evaluator venv 的 task-filter 为 `3 passed`。
- 独立 evaluator venv 已完成真实 LIBERO 单 task/单 episode/单 step smoke；clean T0 checkpoint 通过新增 OpenPI→HTTP adapter 的 `/health`、`/reset`、`/act` 及真实 rollout；视频链路已改为显式 ffmpeg H.264、yuv420p、20 FPS、faststart，并由 ffprobe 校验通过。
- clean T0/T1/T2/F1/F2 已在 GPU1–3 建立持久队列；GPU0 上已有外部进程，未触碰。checkpoint adapter 已固定；正式全量 model-selection evaluator 仍需 clean T1/T2/F1/F2 checkpoint、固定评测 manifest 和 public T3 holdout 后才能执行。

本轮已把计划中可在 Git 仓库内审计和重建的部分实现为 `tools/axis_pi05`，并在 GPU 服务器完成真实数据、Axis restore、训练与 checkpoint smoke。原始数据、Axis Orbax 参数、OpenPI clone、checkpoint、视频仍由 `.gitignore` 排除。

## 已实现

- `configs/axis_pi05/data_sources.lock.yaml` 锁定 OpenPI、五套标准 LIBERO、LIBERO-plus、Plus coarse-category labels 和 Axis 模型的完整 40 位 SHA。
- `bootstrap.sh`/`bootstrap.py`：独立环境、GPU/磁盘/HF 状态检查、OpenPI checkout、patch replay、HF snapshot 下载、文件/不完整下载校验和 provenance 报告。
- canonical episode catalog 与 frame adapter：只重命名 camera/state/action namespace；fps、维度、双相机、revision 和 Plus task/length join 失败即停止。
- `data_audit`：JSON/CSV/HTML 报告、review flag、无自动删除的人工审阅清单。
- `build_mix`：确定性 split、source-file 泄漏保护、M0/M1/M2 sampling weights、manifest hash。
- `compute_norm`：按有效 sampling weight 做 streaming weighted moments 和确定性 weighted quantiles；明确不读取 Axis 原始 `norm_stats.json`。
- F0/F1/F2 parameter audit、internal-step checkpoint manifest、阶段训练编排、评测适配、失败 H.264、诊断 tag、timeline 和本地 gallery。
- 静态 gates、OpenPI patch replay check 和不依赖 pytest 的纯逻辑回归测试。

## GPU 执行状态

- 服务器：4× Tesla V100-PCIE-32GB；训练使用 GPU 1–3，GPU0 上已有外部进程未触碰。
- 六个锁定数据源、Plus v3 reader、catalog、M0/M1/M2 manifest 和三套 mixture norm stats 已完成并通过数据门禁。
- T0/F0/M1 10 steps 已通过；初始 T1/F0/M1 在 peak lr=5e-5 时从 step 200 起 NaN，未计入有效结果。
- 已加入非有限值门禁；以 `peak_lr=1e-5`、`decay_lr=1e-6` 重跑的 T1/F0/M1 500 steps 全程 finite。
- clean T0 checkpoint：internal step 10，params/train_state Orbax restore smoke 通过，`optimizer_state=present`，norm canonical equality 通过，stable tree SHA 为 `a2fdd3a3f6b35dea7dce4565d57d3c33ed4716ff3bd03f3b74559b764702573c`。此前 dirty T1/F0/M1 step 500 仅作诊断，不计入有效 formal checkpoint；clean T1 正在排队执行。
- F1/M1 在 V100 上仍出现可复现的数值不稳定：bfloat16 兼容路径在 step 20 变为 NaN；降低 LR、收紧 clip 仍在 step 20 失败；实验性 FP32 optimizer-state 路径在 step 10 出现 `grad_norm=inf`，已回退且不计入正式结果。
- 因此没有伪造 F1/F2 checkpoint；上述历史记录是整改前状态。整改后的 F0 clean T1/T2、F1/F2 轨道已排队，正式 checkpoint evaluator 仍保持 gated。
- 最终门禁：`tools.axis_pi05.check` 通过；训练环境测试 `61 passed, 7 skipped`，独立 evaluator venv 的 3 个 task-filter 测试通过。训练环境不安装 evaluator 专用依赖，避免改变环境边界。

GPU 运行使用以下入口；不在没有实际输入或 evaluator 输出时伪造 checkpoint、评分或视频：

```bash
bash tools/axis_pi05/bootstrap.sh --lock configs/axis_pi05/data_sources.lock.yaml
python -m tools.axis_pi05.data_audit --config configs/axis_pi05/data.yaml
python -m tools.axis_pi05.build_mix --all --config configs/axis_pi05/sampling.yaml
python -m tools.axis_pi05.compute_norm --mix M0
python -m tools.axis_pi05.compute_norm --mix M1
python -m tools.axis_pi05.compute_norm --mix M2
```

## 历史来源审计

旧仓库只贡献了计划要求的历史证据：OpenPI HEAD `c23745b5ad24e98f66967ea795a07b2588ed6c79`、旧 diff SHA-256 `8031fa51a1dd2276b2309bb04947254a75e2d6ed838a9dd7f3a56d0b7a3c8412`、Axis revision `c342608c489515efa9ba8281400dc648bf3970e1` 和 Plus revision `f3f49f426d75030177b18778374005bc12ccd588`。新实现没有引用旧仓库绝对路径、旧 checkpoint、旧 norm stats 或 T1 专用 sampler。
