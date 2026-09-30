# R6 资产恢复与评分刷新计划（2026-09-29）

状态：`partial_official_results_t2_zero_t3_evaluating_t1_pending`

## 授权与边界

用户于 2026-09-29 明确要求恢复历史评测资产并使用当前可用 GPU。当前分支 `codex/axis-pi05-sg0-sg12-support-gold-20260907` 的 HEAD 为 `dc7efaa68bd622be37e3de993410a78da968abe4`。恢复产物放在新的 ignored `outputs/axis_pi05/recovery_20260929/` 下；不覆盖原始数据、历史证据、`third_party/axis_openpi` 的 7 项既有本地改动，也不自动上传或提交比赛结果。此前清理仍作为有效历史记录。

本计划只恢复 R6 推理资产并做可复核的评分对照，不恢复旧训练、续训或 H300。提交 ZIP 的远端二进制不存在于 Git；除非逐项校验通过，不声称恢复成功。GPU 用于推理/本地评测，不用于新训练。

## 锁定来源

- R6 参数：私有 HF `Huangpenguin/parc2026-r6-l2-params-step500`，commit `e788d92131c984f4b4f7914e0b90b244ad014e4c`；历史 params tree SHA256 `d9b7e0d15781060fccfd61dc1890ad98993f0224e4aebba1dd6658d7c011931f`。
- R6 QUANTILES norm：`data/reports/axis_libero_30k/phase1/norm_stats.json`，历史 SHA256 `4c803bf59c52677c36b6b420c614d8a7742e0847708443863e13e67cb80b18f5`；已与清理保留的 R6 `assets/canonical_10hz/norm_stats.json` 作字节比较并相同。
- 原始数据锁：`data/raw/hf/physical-intelligence/libero`，使用 machine handoff 中的 manifest/meta-info hashes；先核验，不改写。
- GPU：4× Tesla V100 32 GiB；每卡 Xorg 约 4 MiB。每个评测启动前重新检查进程、UUID 和空闲显存，保留 Xorg。

## 阶段

1. 从锁定 HF commit 下载 `params/` 和代码快照到新恢复目录；记录下载 PID/heartbeat。固定 commit、HF 文件清单、远端/本地文件数与逐文件尺寸；依赖 pinned revision 与 HF 下载完整性校验。历史 params tree SHA 保留为对照记录，不改写。
2. 核验 QUANTILES norm 字节副本和 raw-data manifest/meta-info 的历史锁。失败时不折叠或打包。
3. 用仓库现有折叠/FAST 导出流程恢复 R6 FAST 权重，检查 FAST SHA256 是否为 `e55bf7dcca78b2195993dd7a12b96a4ea13e94761d711c99a82b7411c3f8ceb6`。不修改 nested OpenPI 工作树。
4. 重建本地候选：T1/Track3 execute8、T2 execute4。每个 ZIP 单独命名并记录 bytes/SHA、配置和来源；若不能复现旧 ZIP SHA，就记录为新构建，不能冒称旧 ZIP。
5. 锁定当前官方公开评测器 revision 与数据版本后运行共同任务对照。官方公开本地配布仅含 Track 1 example tasks，且不包含排行榜总分配置；T2/T3 的新官方总分必须来自官方评测返回，不通过本地指标推断，也不自动上传/提交。

## 恢复进度（2026-09-29）

- HF 参数从锁定 revision 下载完成；Hub 与本地清单为 17/17 文件，逐文件尺寸相同。保留目录为 `outputs/axis_pi05/recovery_20260929/`；原始 LIBERO 数据未改动。
- Orbax 的异步元数据 `gather` 在当前环境挂起；改用同步 TensorStore 读取后恢复 61 个张量、13,502,209,088 bytes，5 个 LoRA 模块折叠，PyTorch 权重映射 missing/unexpected 均为 0。
- 新转换 `model.safetensors` 大小与历史记录同为 14,467,165,872 bytes，但受控 R6 导出器的历史 SHA 锁检查失败。因此新文件不能称为历史 folded 原件；需保留为 pinned checkpoint 派生恢复件，并与历史 best ZIP 区分。
- R6 QUANTILES norm 与保留副本字节一致；处理器文件已从清理保留的 JSON 模板及该 norm 生成。2026-09-11 HF runtime tar 仅含代码快照，不含旧 ZIP/服务器二进制。
- 当前 `matsuolab/PARC2026_pre` 固定到 `eb8a063cf1d69615754b0cbb31b0a9162621ec9b`，公开本地包只有 Track 1 example tasks；环境及 6.4 GB LIBERO-plus 资产准备完毕。
- 评分尚未运行。root CUDA venv 的 LeRobot v0.4.4 推理导入明确缺少 `accelerate`；当前 LeRobot 源与现装 Transformers 4.53.2 也不兼容。尝试在隔离 `/tmp` 运行时包目录安装 `accelerate==1.10.1` 被 Stop That Shit 自动审批以 `DEPENDENCY_NOT_AUTHORIZED` 拒绝。继续官方本地推理需用户明确授权安装所需运行时依赖；期间不触碰 CUDA venv 与 nested OpenPI 改动。
- 按用户本轮要求，已从 pinned checkpoint 派生重建三个 track 的本地候选 ZIP：Track1/Track3 共用 execute8 包，Track2 使用 execute4 包。恢复件是普通 PI0.5 safetensors，不含历史 FAST 权重；导出器锁定的历史 folded SHA 未通过，因此三份都不是历史 ZIP，也没有继承历史分数。
- 三个包均用当前官方公开 `validate_submission.py --static` 检查，通过且 0 errors / 0 warnings；ZIP CRC、15 项结构、动作步数配置和 20 GB 大小限制均通过。仅完成静态检查，未启动服务或评测，官方分数仍为 `not_evaluated`。
- 重新查询 GPU 时，`nvidia-smi` 报告无法与 NVIDIA driver 通信；本轮没有启动 GPU 作业。候选与验证记录位于 ignored `outputs/axis_pi05/recovery_20260929/packages/`，没有上传、提交或 push。

## 停止条件

- HF revision、params tree、norm 或数据哈希不匹配；
- 折叠/FAST 输出身份无法绑定到恢复源；
- GPU 进程检查发现非本任务进程或显存不足；
- 当前官方评测器/数据入口无法锁定。

## 推理修复与 smoke follow-up（2026-09-29）

用户明确授权在 `/tmp/r6_runtime_20260929` 安装依赖并使用 GPU。隔离运行时安装 Python 3.10.19、LeRobot 0.4.4、Transformers 4.53.2、Accelerate 1.10.1、Torch 2.10.0+cu128；四张 Tesla V100 均可见。

实际加载发现两项阻止有效推理的问题：stock Transformers 的 Gemma 缺少 checkpoint 所需 AdaRMSNorm/gated residual 结构，导致严格权重加载报缺失/意外 key 后被 LeRobot 捕获并继续用随机初始化权重；同时 LeRobot 0.4.4 的 denoise loop 多次修改同一个前缀 DynamicCache，复现了 988/978 的 mask/key 长度冲突。修复版包内随附 OpenPI 兼容 Gemma 模块、补回 safetensors 省略的 tied embedding 别名，并在每个 denoise step 复制 cache 容器。候选 requirements 显式加入 `accelerate==1.10.1`。

修复包为 `packages/PARC2026_T1_R6_rebuilt_execute8_cachefix_20260929.zip`、`packages/PARC2026_T2_R6_rebuilt_execute4_cachefix_20260929.zip`；T3 与 T1 为同一 artifact。两个唯一 ZIP 均通过锁定版本 `validate_submission.py --static`，0 errors / 0 warnings。T1/T3 和 T2 的日志均显示 `All keys loaded successfully!`；T1/T3 各完成一条官方本地 Track1 环境 `/act`（HTTP 200），T2 execute4 对同一公开观察/动作 schema 完成 5 条 `/act`（HTTP 200）。公开本地评测器只提供 Track1 示例任务，T2/T3 的官方 track 分数未评测。

详细结果：`outputs/axis_pi05/recovery_20260929/packages/INFERENCE_SMOKE_20260929.json`。用户提供的 SHA256 `f18a32cb6bda6e4629bfd2c58827a6cb711736b07bf92d2f2e35365ce71b5d57` 指向原始 T1 候选；原始 ZIP 保持不变。修复版 ZIP 未生成新的 SHA256，不能用原 SHA 标识修复版。完整 LIBERO 官方分数仍为 `not_evaluated`，无上传、提交或 push。

原有 CPU 转换 PID 2234022 仍在写 `folded_model/`，未被停止或碰触；运行控制 cursor 未被改写。推理服务已关闭，GPU 显存已释放。若继续评分，先重新检查该进程、cursor 与 GPU，并使用官方评测器；不把本次 smoke 分数当作 track 成绩。


## 官方评测状态更新（2026-09-29，用户报告）

- T2 修复版提交结果为 0。此前本地 smoke 只证明模型严格加载并成功返回动作，不代表任务成功率；应以此提交结果为准。
- T3 仍在评测，分数待回报。
- T1 今天没有机会评测，保持 `not_evaluated`。
- 这只是状态记录：没有再次运行 T1、没有修改已提交 ZIP，也没有上传或 push。更新明细在 `outputs/axis_pi05/recovery_20260929/packages/INFERENCE_SMOKE_20260929.json`。
