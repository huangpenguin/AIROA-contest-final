# Axis Pi0.5 审计整改、重跑与验收计划

状态：**执行中：R0–R10 已完成；本地固定 manifest 的 Axis-init/T2 model-selection 已完成，T3 训练与定点评测仍待完成**
日期：2026-08-27（Asia/Tokyo）
执行者：Codex  
基线计划：`docs/AXIS_PI05_POSTTRAIN_EXECUTION_PLAN.md`  
审计基线：2026-08-26 对当前实现、数据产物、日志和 checkpoint 的只读审计

## 执行记录（2026-08-26）

| 阶段 | 状态 | 证据 |
|---|---|---|
| R0 | completed | `data/reports/axis_pi05_v2/pre_remediation_inventory.json` |
| R1 | completed | v2 provenance、clean worktree patch replay、`git diff --check`；formal v2 clean gate 已提交 |
| R2–R4 | completed | `data/manifests_v2/{M0,M1,M2}/`、sampler audit、loader audit、norm provenance |
| R5 | completed | `data/reports/axis_pi05_v2/runtime_parameter_audit_F{0,1,2}.json` |
| R6 | completed for D3 | `data/reports/axis_pi05_v2/f1_diagnostic_D3.json`：50/50 finite |
| R7 | completed | clean T0 step=10 checkpoint 的 `checkpoint_manifest_v2.json` 与 `restore_validation.json`；独立 validator restore/forward/norm gate 通过 |
| R8 | completed in training/evaluator environments | 训练环境 `64 passed, 7 skipped`；独立 evaluator venv 的 task-filter `3 passed` |
| R9 | completed for T0/T1/T2 | pre-commit dirty runs 保留作诊断；clean T0、T1 F0/F1/F2 500 和 T2 F0/M0/M1/M2 1000 已完成并保留 checkpoint；T3 3000 仍未执行 |
| R10 | completed for local panels | clean checkpoint 的 OpenPI→HTTP adapter 与 Axis→LeRobot 候选均完成启动/单步 smoke；提交候选静态 validator 通过；Axis-init 与三个 T2 候选完成固定 `select_core` 82-case rollout、paired baseline、H.264/yuv420p/20 FPS ffprobe 和 failure gallery；官方 E2E 仍 deferred |

旧 v1 产物均保持不变；所有整改训练输出写入 `outputs/axis_pi05_v2/`，不将 pre-commit dirty 诊断 run 晋升为正式证据。当前 clean queue 从最新 clean commit 启动，且 `run.py` 已在创建 v2 run 前拒绝 dirty worktree。

## 1. 目标

本计划用于把当前“工程上真实但实验语义尚未闭环”的 Axis Pi0.5 训练流水线修正为可复现、可比较、可评测的正式流水线。

最终路径保持不变：

```text
axisrobotics/Pi0.5-Axis params
  -> locked LIBERO + LIBERO-plus local snapshots
  -> canonical catalog and leakage-aware split
  -> executable M0/M1/M2 sampler
  -> sampler-matched norm stats
  -> runtime-verified F0/F1/F2 training
  -> internal-step checkpoints 500/1000/3000
  -> fixed evaluator and failure H.264 review package
```

本轮整改必须解决以下审计阻塞项：

1. OpenPI loader 没有消费 episode/task/category 级 sampling weights。
2. 当前 norm 分布与实际训练 sampler 分布不一致。
3. LIBERO-plus 没有 validation episode。
4. F1/F2 parameter audit 没有包含 runtime LoRA 参数。
5. F1 数值失败缺少逐 step、逐参数组诊断和完整 provenance。
6. checkpoint 整体 hash、restore smoke 和 norm asset 还不能独立复核。
7. 实现大部分未进入 Git，现有 checkpoint 无法由 commit 精确重建。
8. 初审时 evaluator 的 3 个 task-filter 测试曾失败；当前 task-filter 测试、固定 82-case evaluator、失败媒体链路和本地 failure review package 均已通过，public T3 仅完成可枚举 manifest。

## 2. 已锁定前提

### 2.1 HF 权限

用户已提供 Hugging Face 权限，执行者可从环境变量读取 `HF_TOKEN`。

执行约束：

- 只检查 token 是否存在，绝不打印 token 内容。
- token 不得写入 YAML、JSON、shell history、日志、W&B、Git 或 checkpoint。
- 所有 dataset/model 继续使用 `configs/axis_pi05/data_sources.lock.yaml` 中的完整 40 位 revision。
- 禁止使用未锁定的 `main`、短 SHA 或隐式 latest。
- 数据和模型 snapshot 完整后，正式训练设置 `HF_HUB_OFFLINE=1`，避免训练期间发生隐式网络读取。

建议的无泄密前置检查：

```bash
python -c 'import os; assert os.getenv("HF_TOKEN"), "HF_TOKEN is missing"; print("HF_TOKEN present")'
huggingface-cli whoami
```

第二条命令只用于确认账号权限；其输出可以进入 bootstrap 日志，但不得包含 token。

### 2.2 当前产物的处置

| 产物 | 当前定级 | 后续处理 |
|---|---|---|
| locked HF snapshots | 可复用 | 校验 revision/tree 后离线复用，不重复下载完整数据 |
| canonical catalog | 条件可复用 | 重新做语义和 grouping 审计后生成 v2 catalog |
| 当前 M0/M1/M2 split/weights | 不可用于正式训练 | 保留作对照，生成 v2 manifests |
| 当前 M0/M1/M2 norm | 不可用于 v2 正式训练 | 保留作对照，按 v2 sampler 重算 |
| `t1_f0_m1_lr1e5/500` | `diagnostic_only` | 不删除、不晋升、不作为 T2/T3 warm start |
| 当前 F1 失败 runs | `diagnostic_only` | 补齐索引和失败原因，不伪装为正式消融 |
| evaluator/gallery 工具 | scaffolding | 在真实 evaluator episode 上验收后才能标为完成 |

所有新产物使用 `v2` 或新的 run ID，禁止覆盖当前证据。

## 3. 执行原则与停机红线

- 所有正式 run 从相同 Axis params 初始化，不从当前 diagnostic checkpoint 续训。
- split 必须先于 sampling weights 和 norm stats 生成。
- norm 必须匹配训练 sampler 的理论 frame distribution。
- trainable set 必须从 weight restore 和 dtype cast 后的 runtime NNX state 导出。
- 非有限检查必须逐 step 执行，不得只在 log interval 聚合时检查。
- 每个命令必须产生 resolved config、provenance、stdout/stderr log 和退出状态。
- 任何失败 run 都保留并标记，不删除日志，不补写虚假成功字段。
- 修复 sampler、split、runtime parameter gate 和 checkpoint validator 之前，禁止启动正式 T1/T2/T3。
- F1/F2 是独立消融轨道；若 F1 仍阻塞，不得阻止已经重新验收的 F0 数据-mixture 主轨继续推进。

## 4. 目标目录

建议在现有结构上新增版本化目录：

```text
configs/axis_pi05/
  sampling_v2.yaml
  train_v2.yaml
contracts/axis_pi05/
  sampler_audit.schema.yaml
  runtime_parameter_audit.schema.yaml
  checkpoint_manifest_v2.schema.yaml
data/processed/libero_axis_mix_v2/
data/manifests_v2/{M0,M1,M2}/
data/reports/axis_pi05_v2/
outputs/axis_pi05_v2/
reviews/axis_pi05_v2/
```

Git 只跟踪代码、配置、lock、patch、contracts、小型 manifests 和报告。raw data、模型、checkpoint、视频继续由 `.gitignore` 排除。

## 5. R0：冻结当前证据

目的：在整改前保留当前事实，防止新代码覆盖旧结论。

执行项：

1. 生成 `data/reports/axis_pi05_v2/pre_remediation_inventory.json`。
2. 记录当前 Git commit/status、patch tree hash、tools/config/contracts tree hash。
3. 索引所有现存 T0/T1/F1 probe 目录，包括没有 `run_state.json` 的 probe。
4. 对每个 run 记录：run ID、scope、mix、LR、precision、clip、最后 finite step、失败类型、是否有完整 provenance。
5. 把 `t1_f0_m1_lr1e5/500` 明确写成 `diagnostic_only_reason=sampler_norm_mismatch`。
6. 不移动、不删除、不重新序列化已有 checkpoint。

验收：旧目录 hash 在整改过程中保持不变。

## 6. R1：Git 与 bootstrap 可重建性

### 6.1 修复 provenance

当前 `git diff` hash 不包含 untracked 文件。新 provenance 必须满足以下二选一条件：

- 推荐：正式 run 只允许在 clean Git commit 上启动；或
- 诊断 run：额外生成包含 `tools/axis_pi05`、`configs/axis_pi05`、`contracts/axis_pi05`、`patches/axis_openpi` 的逐文件 SHA-256 inventory。

正式 run gate：

```text
git_dirty == false
implementation_commit is tracked
patch_series_sha256 is recorded
openpi_revision is full SHA
```

### 6.2 HF/bootstrap

执行者只对缺失或 hash 不符的 snapshot 执行下载；已有 snapshot 先做 revision、expected files、incomplete markers 和 tree hash 检查。

bootstrap 报告必须记录：

- HF repo ID、repo type、完整 revision、local root、file count、total bytes、tree hash。
- token 仅记录 `hf_auth_available: true/false`，不记录值。
- OpenPI base commit、每个 patch SHA、patch 顺序和最终 source tree hash。
- 当前 patch tree，不能复用修复前 bootstrap report 的旧 hash。

验收：在一个新的临时目录，仅凭 Git、HF 权限和 lock 完成 bootstrap dry-run 与实际 replay。

## 7. R2：让训练真正消费 M0/M1/M2 权重

这是本计划的最高优先级阻塞项。

### 7.1 权重语义

`sampling_weights.parquet` 中每行代表一个训练 episode，必须包含：

```text
(source_repo, revision, source_episode_index)
episode_sampling_probability
frame_count
```

定义每个 frame 的理论 sampler 权重：

```text
frame_weight = episode_sampling_probability / observed_frame_count
```

必须满足：

```text
sum(frame_weight for frames in episode) == episode_sampling_probability
sum(frame_weight for all train frames) == 1
```

### 7.2 loader 改造

执行者需要修改 OpenPI patch，而不只修改 ignored 的 `third_party/axis_openpi` 工作树：

1. loader 读取完整 `sampling_weights.parquet`，不能只在 `source_settings()` 中汇总 repo mass。
2. 标准 LIBERO dataset 建立 dataset index -> `(repo, episode)` 映射。
3. Plus v3 dataset 的 `_index` 已包含 episode/frame offset，应直接据此展开 frame weights。
4. `WeightedConcatDataset.sample_weights` 使用每个 frame 的 manifest-derived weight。
5. 禁止在 repo 内重新变为 uniform frame sampling。
6. `WeightedRandomSampler` 显式传入由 `config.seed` 初始化的 `torch.Generator`。
7. JAX 单进程、多 device 情况下 sampler 不得被另一层 distributed sampler 替换。
8. resume 时必须恢复 sampler generator state，或在 manifest 中明确不支持 bit-exact resume 并阻止正式 resume。

### 7.3 sampler audit

新增 `tools/axis_pi05/sampler_audit.py`，输出：

- theoretical episode/source/suite/task/category distribution；
- loader-expanded frame distribution；
- 两者的 total variation distance、最大绝对误差和最大概率比；
- 固定 seed 抽取至少 200,000 个 index 的 empirical bucket frequency；
- 重复运行 hash 和前 N 个 sampled indices hash。

验收阈值：

- theoretical episode distribution TV <= `1e-10`；
- source/suite/task/category theoretical bucket max abs error <= `1e-10`；
- empirical source/suite/category TV <= `0.01`；
- 相同 seed 的 sampled-index prefix hash 完全一致；
- M0/M1/M2 的实际 bucket 分布必须彼此符合各自设计，不能只在 norm 文件上不同。

## 8. R3：重新设计 Plus validation split

### 8.1 先做 grouping EDA

生成 `plus_grouping_audit.csv/json`，统计下列候选字段在每个 task/category 下的 group cardinality：

- recovered `source_file`；
- 数据存储 shard/chunk/file；
- BDDL/task/init-state provenance；
- 可恢复的 base trajectory/demo ID；
- episode index 与 perturbation variant 的映射。

必须明确区分：

- storage shard：多个独立 episode 恰好存放在同一物理文件中；
- provenance group：多个扰动 episode 来源于同一 base trajectory/init state。

storage shard 不能自动当作 leakage group；当前 `source_file` 只有一个 group 的情况下尤其如此。

### 8.2 split unit 决策

按以下优先级选择 Plus split group：

1. 有 100% 覆盖且无冲突的 `base_trajectory_id`：使用它分组。
2. 只能恢复 task/init-state group：使用该组合，并记录 grouping quality。
3. 无法恢复共享来源：退化为 episode-level deterministic hash split，并明确写入 `grouping_quality=episode_only`；不得声称已防止跨扰动同源泄漏。

不得继续用“每个 task/category 只有一个 group”的字段生成零 Plus validation。

### 8.3 split gate

v2 split 必须满足：

- train/validation episode identity overlap 为零；
- 选定 provenance group overlap 为零；
- Plus validation episode 数量大于零；
- 每个 Plus suite/task/category 至少有一个 validation group，无法满足的桶单独列出并阻止正式 model-selection；
- 全局 validation fraction 目标为 10%，允许范围 `[8%, 12%]`；
- split seed 和 split implementation hash 固定；
- 三个 mix 共享同一 train/validation split，只改变 sampling weights。

## 9. R4：重算 sampler-matched norm stats

R2 和 R3 通过后，重新生成 M0/M1/M2 v2 norm。

要求：

1. stats 只读取 train episodes。
2. frame weight 必须调用与训练 sampler 相同的公共函数，避免两份公式漂移。
3. Axis checkpoint 的 `norm_stats.json` 只记录 hash，永不加载。
4. 每个 stats 文件记录 catalog、split、sampling table、lock、实现 commit 和 source revision hash。
5. state 8D、action 7D 的 mean/std/q01/q99 全部 finite，std 必须大于配置的最小阈值。
6. checkpoint asset 中保存的新 stats 必须做 canonical JSON equality 检查。
7. 同时记录 source norm hash、checkpoint asset byte hash 和 canonical semantic hash，避免 `1`/`1.0` 序列化差异造成误报。

验收：`sampler_audit` 的理论 frame distribution hash与 norm provenance 中的 distribution hash一致。

## 10. R5：runtime F0/F1/F2 parameter gate

当前离线 F1/F2 audit 缺少 LoRA。整改后必须在以下时点导出参数：

```text
model create
  -> Axis partial restore
  -> frozen/trainable dtype cast
  -> trainable_filter application
  -> optimizer state creation
  -> runtime_parameter_audit.json
```

报告逐参数记录：path、shape、dtype、numel、trainable、optimizer-state-present、初始化是否 finite。

基于当前模型拓扑的预期数量，用作回归参考而不是硬编码真相：

| scope | 预期 trainable 组成 | 当前参考数量 |
|---|---|---:|
| F0 | 两个 LoRA 分支 | 49,987,584 |
| F1 | F0 + action in/out projections | 50,054,176 |
| F2 | F1 + time MLP in/out | 52,153,376 |

gate 必须验证：

- `paths(F0) ⊂ paths(F1) ⊂ paths(F2)`；
- F1 比 F0 只多四个 action projection tensor；
- F2 比 F1 只多四个 time MLP tensor；
- vision、dense base language 和 dense base expert 全部冻结；
- trainable optimizer state keys 与 trainable path 完全一致；
- runtime audit hash 写入 run provenance 和 checkpoint manifest。

不满足时在 step 0 前退出。

## 11. R6：F1 NaN/Inf 定位轨道

F1 诊断与 F0 主轨分开执行，不得边试边覆盖正式 run。

### 11.1 诊断日志升级

- `log_interval=1`，非有限检查也逐 step 执行。
- 分别记录 loss、global grad norm、global param norm、update norm。
- 额外记录 LoRA、action_in_proj、action_out_proj 三组的 grad/param/update norm 和 max abs。
- 在 optimizer update 前后分别检查 finite。
- 记录首个 non-finite tensor path，而不是只记录聚合 `grad_norm=nan`。
- 固定前 32 个 batch index hash，确保每个 probe 使用同一数据。

### 11.2 诊断矩阵

所有 probe 从 Axis params 开始，使用同一个 v2 M1 sampler/norm/seed/batch：

| probe | trainable | 目的 |
|---|---|---|
| D0 | none, forward only | 验证 restore 后 forward/loss finite |
| D1 | LoRA only | 重现 F0 数值基线 |
| D2 | action projections only | 隔离 action head gradient/update |
| D3 | LoRA + action projections | 重现正式 F1 |
| D4 | F1，action head 独立 LR | 判断是否需要 parameter-group LR |

precision 原则：

- frozen base params 可保持 bfloat16。
- LoRA 和新解冻 action/time 模块优先保持 float32。
- optimizer moments 保持 float32。
- 不再把“全部 trainable cast bfloat16”作为正式解决方案。

诊断决策：

1. update 前 loss/grad 已 non-finite：定位 forward/backward op、输入批次和首个参数组。
2. grad finite、update 后 non-finite：检查 optimizer epsilon、weight decay、moment dtype、update scaling。
3. D2 稳定而 D3 不稳定：检查 LoRA/head 共同 update 和 LR 分组。
4. D2 单独不稳定：action head 使用更低 LR，并对 bias/kernel 分别审计；不能只继续全局降低 LR。
5. D3 连续 50 steps 全部 finite 后，才允许正式 F1 500。
6. F1 未通过前不启动 F2；但已经通过全部数据和 checkpoint gate 的 F0 主轨可以继续 T2。

## 12. R7：checkpoint manifest v2

### 12.1 稳定 hash 规则

`checkpoint_tree_sha256` 使用明确、版本化的文件集合：

- 包含 finalized params、train_state、assets 和 Orbax metadata；
- 排除 `checkpoint_manifest.json` 自身、临时文件、lock 文件和 restore log；
- 必须在 `checkpoint_manager.wait_until_finished()` 后计算；
- manifest 写完后立即由独立 validator 复算。

### 12.2 不接受人工布尔声明

validator 必须实际执行并保存证据：

1. Orbax restore params。
2. Orbax restore train_state。
3. 验证 restored `train_state.step == directory step`。
4. 验证 Adam moments/count 存在且 key 集合匹配 trainable set。
5. 对一个固定样本执行 policy forward，输出 action shape/finite 报告。
6. 验证 checkpoint asset norm 与训练 norm canonical equal。

manifest 中的 `restore_smoke` 和 `optimizer_state` 由 validator 计算，CLI 不允许直接传入 `true/present`。

验收：复制 checkpoint 到新路径后，validator 仍能独立通过。

## 13. R8：测试与干净环境 replay

需要至少三层测试：

### 13.1 纯逻辑测试

- split determinism、Plus holdout coverage、no leakage；
- episode-to-frame weight conservation；
- M0/M1/M2 theoretical distribution；
- sampler seed reproducibility；
- runtime parameter set nesting；
- checkpoint stable hash exclusions。

### 13.2 OpenPI 集成测试

- patched OpenPI clean checkout/replay；
- 六源 dataset loader 构建；
- 每个源各读取多个 episode 的首/中/末 frame；
- 双相机可解码，state/action finite，action horizon padding 正确；
- M0/M1/M2 sampler empirical report。

### 13.3 evaluator 环境测试

训练环境不强行安装 evaluator 依赖；建立单独 evaluator env，并在其中安装/验证 `libero`。

验收命令必须得到 `0 failed`：

```bash
pytest -q tests
python -m tools.axis_pi05.check
python -m tools.axis_pi05.sampler_audit --all
python -m tools.axis_pi05.checkpoint_validator --help
```

pytest 可以有经过解释的 optional skip，但 3 个 task-filter 测试不能继续以 failed 状态存在。

## 14. R9：正式重跑矩阵

所有 run 使用新 ID 和 v2 manifests/norm，均从 Axis params 初始化。

### 14.1 Core F0 主轨

1. `T0-v2`: F0/M1，1 step，随后 10 steps。
2. `T1-v2`: F0/M1，500 steps。
3. 对 step 500 运行 checkpoint validator 和固定小型 evaluator smoke。
4. `T2-v2`: F0/M0、F0/M1、F0/M2，各 1000 steps；三者均从 Axis 开始。
5. 使用固定 evaluator manifest 选择最佳 mix。
6. `T3-v2`: 最佳 F0/mix 从 Axis 开始训练 3000 steps，保留 500/1000/3000。

### 14.2 F1/F2 消融轨

1. D3 50-step finite 后，F1/M1 正式 500。
2. F1 500 finite 后，F2/M1 先 50-step probe，再正式 500。
3. F0/F1/F2 比较保持 data、norm、seed、global batch、precision 和 evaluator 一致。
4. F1/F2 仍阻塞时，报告为未完成消融；不得阻塞 F0 的 M0/M1/M2 核心比较。

### 14.3 默认数值基线

重新验收后的 F0 可从已证明 finite 的设置开始：

```text
peak_lr = 1e-5
decay_lr = 1e-6
clip_grad_norm = 1.0
seed = 1000
global_batch = hardware probe 后固定
```

F1/F2 若使用 parameter-group LR，必须记录每组 LR，并且不能再把该实验简称为“与 F0 同超参数”。

## 15. R10：evaluator、失败 H.264 与目视报告

正式 evaluator 必须使用同一 task/seed/init-state manifest 比较所有候选。

每个 checkpoint 至少输出：

```text
summary.json
tasks.csv
episodes.csv
failure_index.csv
cases/<case_id>/
  failure_h264.mp4
  episode.json
  contact_sheet.jpg
  timeline.json
  timeline.png
gallery.html
```

失败视频 gate：

- front+wrist 并排；
- H.264/libx264、yuv420p、20 FPS、faststart；
- ffprobe 通过且帧数/时长与 episode metadata 一致；
- overlay 包含 run、internal step、task、suite、source、category、label quality、seed、result；
- 保存全部 failure/collision/timeout/environment error，每个 stratum 至少保留一个 success reference；
- failure tag 区分官方 evaluator 事实与本地诊断推断。

当前 gallery 工具只有在真实 episode 输出上通过上述检查后，才能把状态从 scaffolding 改为 complete。

## 16. 工单顺序

| 工单 | 内容 | 依赖 | 完成门禁 |
|---|---|---|---|
| R0 | 冻结当前证据 | 无 | inventory 和旧 hash 固定 |
| R1 | Git/HF/bootstrap 重建 | R0 | clean replay 通过 |
| R2 | manifest-aware sampler | R1 | theoretical/empirical audit 通过 |
| R3 | Plus split v2 | R1 | Plus validation 和 leakage gate 通过 |
| R4 | v2 norm | R2,R3 | distribution hash 对齐 |
| R5 | runtime parameter audit | R1 | F0/F1/F2 nesting 通过 |
| R6 | F1 数值诊断 | R2,R4,R5 | D3 50-step finite 或明确 blocked |
| R7 | checkpoint v2 | R1,R5 | 独立 restore validator 通过 |
| R8 | tests/clean replay | R2-R7 | 0 failed |
| R9 | 正式 T0/T1/T2/T3 | R8 | formal run manifests 完整 |
| R10 | evaluator/media | R7,R8,R9 | 固定评测与视频 gate 通过 |

R2、R3 可以并行实现，但 R4 必须等待两者完成。R5、R7 可以与数据整改并行。R6 不得占用正式训练 run ID。

## 17. 每次交接必须提供的证据

后续 agent 每完成一个工单，必须更新状态文档并列出：

- 修改的 tracked files；
- Git commit 和 clean/dirty 状态；
- 执行的完整命令；
- 测试结果和退出码；
- 新增报告/manifest/checkpoint 的路径与 SHA-256；
- 未完成项和阻塞原因；
- 是否使用 GPU、GPU IDs、开始/结束进程快照；
- 是否访问 HF、访问的 repo/revision；
- 明确声明没有输出的阶段，不得生成占位成功结果。

## 18. 最终 Definition of Done

只有同时满足以下条件，才允许把 v2 pipeline 标记为完成：

- clean Git clone + HF lock 能重建 OpenPI、数据和 Axis params。
- M0/M1/M2 sampler 实际消费 episode/task/category 权重，理论和 empirical audit 通过。
- LIBERO-plus 有合格 validation holdout，split leakage 结论有明确质量等级。
- 三套 norm 与各自训练 sampler distribution hash一致，Axis stats 未加载。
- runtime F0/F1/F2 trainable set 可审计；未完成的 F1/F2 明确标成 blocked，不影响 F0 主轨事实。
- 新 F0/M1 500-step checkpoint 全程 finite、internal step 正确、独立 restore 通过。
- T2 三种 mix 使用相同协议从 Axis 初始化并完成 1000 steps。
- T3 最佳配置完成 3000 steps，500/1000/3000 checkpoint 均保留可恢复。
- tests 为 `0 failed`，官方 evaluator adapter 在独立 evaluator 环境通过。
- 失败 H.264、task/扰动 metadata、timeline 和 gallery 能相互追溯。
- 当前旧 F0/F1 产物继续保留为 diagnostic，不与 v2 正式结果混淆。

## 19. 本计划不授权的动作

本文件本身不授权执行者删除旧数据/checkpoint、上传权重、修改官方 scorer 语义、启动正式训练或替用户选择最终候选。上述动作仍需以后续明确执行任务为准。
