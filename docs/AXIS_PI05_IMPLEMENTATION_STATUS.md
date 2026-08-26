# Axis Pi0.5 后训练执行实现状态

日期：2026-08-26（Asia/Tokyo）

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
- 有效 checkpoint：internal step 500，params/train_state Orbax restore smoke 通过，`optimizer_state=present`，manifest 已写入输出目录。
- F1/M1 在 V100 上仍出现可复现的数值不稳定：bfloat16 兼容路径在 step 20 变为 NaN；降低 LR、收紧 clip 仍在 step 20 失败；实验性 FP32 optimizer-state 路径在 step 10 出现 `grad_norm=inf`，已回退且不计入正式结果。
- 因此没有伪造 F1/F2 checkpoint；T2/T3 与官方 evaluator 暂停在这个数值稳定性门禁之后。完整运行摘要见 `data/reports/gpu_execution_summary.json`。
- 最终门禁：`tools.axis_pi05.check` 通过；训练相关测试 `56 passed, 7 skipped`。正赛 evaluator 的 3 个 task-filter 测试因独立训练环境没有 `libero` 包而阻塞，未安装该依赖以免改变 evaluator 环境边界。

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
