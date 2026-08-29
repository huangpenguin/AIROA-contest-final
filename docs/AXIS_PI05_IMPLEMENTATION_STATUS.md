# Axis Pi0.5 后训练执行实现状态

日期：2026-08-27（Asia/Tokyo）

## 整改续执行状态

- 当前分支：`codex/axis-pi05-audit-remediation`；checkpoint adapter、转换脚本、GPU 说明归总和 Axis 候选包已生成；当前 worktree 包含本轮新增的未提交文档/工具/产物。
- v2 formal run 现在在启动前强制要求 clean Git worktree；启动前已产生的 dirty T1/T2 运行仅保留为诊断参考，不进入正式验收。
- R8 训练环境回归为 `61 passed, 7 skipped`；独立 evaluator venv 的 task-filter 为 `3 passed`。
- 最新回归为 `64 passed, 7 skipped`；固定评测 evaluator 的 task-filter 仍为 `3 passed`。
- 独立 evaluator venv 已完成真实 LIBERO 单 task/单 episode/单 step smoke；clean T0 checkpoint 通过新增 OpenPI→HTTP adapter 的 `/health`、`/reset`、`/act` 及真实 rollout；视频链路已改为显式 ffmpeg H.264、yuv420p、20 FPS、faststart，并由 ffprobe 校验通过。
- clean T0、T1 的 F0/F1/F2 500-step（F1 使用 clean retry）和 T2 的 F0/M0/M1/M2 1000-step 已完成；已完成的 checkpoint 均保留 Orbax `params`、`train_state` 与对应 norm asset。GPU0 上已有外部进程，未触碰；本轮训练/验证使用 GPU1–3。固定评测 manifest、正式本地 model-selection evaluator 和 public T3 manifest 已完成；T3 正式训练仍 gated，但已在显式 diagnostic bypass 下完成独立 F0/M1 3000-step run，500/1000/3000-step checkpoint 均验证并完成 LeRobot 转换。

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
- Axis F2/M1 500-step checkpoint 已通过 JAX Orbax→PyTorch 转换：LoRA 已折叠为 dense 权重，内部 action 32 维与 LIBERO 对外 action 7 维的适配写入配置和处理器统计。
- T3 diagnostic F0/M1 500/1000/3000-step checkpoint 均已通过 params/train_state/optimizer/norm/fixed-forward restore gate；对应 LeRobot 目录为 `outputs/axis_pi05_v2/converted_t3_diag_f0_m1_500_lerobot`、`outputs/axis_pi05_v2/converted_t3_diag_f0_m1_1000_lerobot` 和 `outputs/axis_pi05_v2/converted_t3_diag_f0_m1_3000_lerobot`，三次转换均将 LoRA 10 个模块折叠且模型 key 差异为零。该目录仅为本地 diagnostic artifact，不是官方得分或正式 submission。
- 由该转换结果生成的旧版 LeRobot 提交候选已通过比赛仓库 `validate_submission.py --static`，并在 GPU2 上完成真实 `/health`、`/reset`、`/act` 冒烟；这不是官方隐藏任务评分。
- PARC2026 本选资料 v1.1 的 RTX PRO 6000 Blackwell、60 小时、custom Docker 禁止和 Notebook 启停规则已归总到 `docs/COMPETITION_GPU_GUIDE_SUMMARY_ZH.md`；提交 scorer 的 L4/CUDA13 仅作为另一套运行边界保留。

## 本地固定评测结果（2026-08-27）

- `select_core` manifest 共 82 个 case，manifest SHA-256 为 `0083c46a45f653299f8a8ff8157dc2f4e483bc27958f005e704377418452d91c`；protocol lock SHA-256 为 `3892af7fee26156f1eeb71a87d4585a29aba4952e9750bc55187c4447ee11e83`。
- Axis init、F0/M0、F0/M1、F0/M2 均完成 82/82，coverage complete、runtime/model error 与 infrastructure error 均为 0；所有被引用视频均通过 H.264、yuv420p、20 FPS 的 ffprobe 校验。
- 自动 selector 选出 `f0_m1_step1000`；M0/M1/M2 的 `task×perturbation` macro success 分别为 0.03/0.03/0.01。统计并列后由 collision tie-break 选择 M1；决策文件为 `reviews/axis_pi05/model_selection_v1/winner_decision.json`。
- Axis init（F0/M1、Axis base params + M1 norm）作为 paired baseline 的 micro success 为 0.4634；T2 M0/M1/M2 相对它的 paired success delta 分别为 -0.4268/-0.4268/-0.4512（每项 82 个相同 case，10,000 次 bootstrap）。
- 四组结果的统一 failure gallery 位于 `artifacts/axis_pi05_v2/model_selection_v1/select_core/gallery/index.html`；结果均为 `official_result=false`。`generalization_standard` 1800-case manifest 和 public T3 40-case manifest 已冻结但尚未 rollout；LIBERO-90 长尾当前 unavailable，stress panel 因无通过 effect test 的扰动而 unsupported。
- T3 配置 dry-run 已通过且没有写入/下载/训练；正式启动在写入前曾被 clean-worktree 门禁拦截，随后用显式 diagnostic bypass 完成 `t3_diag_f0_m1_3000_20260827`，500/1000/3000-step 已落盘并通过独立恢复校验，正式 clean-worktree run 仍未宣称完成，记录见 `data/reports/axis_pi05_v2/t3_preflight.json`。

## GPU 执行状态

- 服务器：4× Tesla V100-PCIE-32GB；训练使用 GPU 1–3，GPU0 上已有外部进程未触碰。
- 六个锁定数据源、Plus v3 reader、catalog、M0/M1/M2 manifest 和三套 mixture norm stats 已完成并通过数据门禁。
- T0/F0/M1 10 steps 已通过；初始 T1/F0/M1 在 peak lr=5e-5 时从 step 200 起 NaN，未计入有效结果。
- 已加入非有限值门禁；以 `peak_lr=1e-5`、`decay_lr=1e-6` 重跑的 T1/F0/M1 500 steps 全程 finite。
- clean T0 checkpoint：internal step 10，params/train_state Orbax restore smoke 通过，`optimizer_state=present`，norm canonical equality 通过，stable tree SHA 为 `a2fdd3a3f6b35dea7dce4565d57d3c33ed4716ff3bd03f3b74559b764702573c`。此前 dirty T1/F0/M1 step 500 仅作诊断，不计入有效 formal checkpoint；clean T1 F0/F1/F2 500 已完成，F2 的独立 restore/forward/norm gate 通过。
- F1/M1 在 V100 上仍出现可复现的数值不稳定：bfloat16 兼容路径在 step 20 变为 NaN；降低 LR、收紧 clip 仍在 step 20 失败；实验性 FP32 optimizer-state 路径在 step 10 出现 `grad_norm=inf`，已回退且不计入正式结果。
- 因此没有伪造 F1/F2 checkpoint；上述历史记录是整改前状态。整改后的 F0 clean T1/T2、F1/F2 轨道已完成到计划中的 T1 500-step；T3 正式训练仍保持 gated，诊断 run 已从 Axis base 以 F0/M1 配方完成 3000 steps，500/1000/3000-step 已验证。
- 最终门禁：`tools.axis_pi05.check` 通过；训练环境测试 `64 passed, 7 skipped`，独立 evaluator venv 的 3 个 task-filter 测试通过。训练环境不安装 evaluator 专用依赖，避免改变环境边界。

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
