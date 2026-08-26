# Pi0.5-Axis × LIBERO/LIBERO-plus 后训练执行与审核计划

状态：**GPU 执行已暂停；A11 与数据门禁完成，F0/M1 500-step checkpoint 已验收，A12 在 F1 数值稳定性门禁处阻塞**  
日期：2026-08-26（Asia/Tokyo）  
目标仓库：`PARC2026_final`  
旧仓库：`/home/huang/code/AIROA-contest-libero-plus`，只作为迁移来源和历史证据，不作为正式训练事实源。

## 1. 已锁定的目标与边界

正式训练链路固定为：

```text
axisrobotics/Pi0.5-Axis params
→ 两个逻辑数据源落盘到当前仓库 data/raw（Git 忽略）
→ LIBERO + LIBERO-plus 统一 20 Hz schema / catalog
→ 固定 train/validation split 与分层抽样 manifest
→ 按实际有效 mixture 重新计算 norm stats
→ OpenPI/JAX LoRA 分阶段消融
→ checkpoints：internal train step 500 / 1000 / 3000
→ 官方本地 evaluator + 数值诊断 + 失败 H.264 审阅包
```

锁定约束：

- Axis checkpoint 只提供初始化 `params`。原始 `norm_stats.json` 保留并记录哈希，但不参与本次训练或推理。
- 不生成新的仿真示范、不用正式评测观测训练、不根据正式评分结果追加训练。
- 原始数据、模型、checkpoint、视频均实际落在当前仓库目录下，但必须由 `.gitignore` 排除；Git 只提交代码、配置、revision lock、manifest、摘要和小型报告。
- 数据和环境可以在自有 GPU 机器运行；同时要求另一台机器仅凭 Git clone、HF 凭据和 bootstrap 命令可重建同一训练环境。
- 正式 evaluator、任务注册、成功/碰撞判定和提交接口以当前正赛仓库为准，不迁移旧 evaluator 取代它们。
- 500、1000、3000 指 `train_state.step`，不是 OpenPI 当前零基循环目录名。

## 2. 目标目录与 Git 边界

执行者按以下布局实现；不得把大文件混入源码目录：

```text
PARC2026_final/
├── configs/axis_pi05/                 # tracked：训练、抽样、评测 YAML
├── contracts/axis_pi05/               # tracked：canonical schema 与参数树契约
├── data/
│   ├── raw/hf/                        # ignored：HF 数据集真实文件
│   ├── models/axisrobotics/           # ignored：Pi0.5-Axis Orbax 权重
│   ├── processed/libero_axis_mix_v1/  # ignored：统一 catalog/cache，不复制原始视频
│   ├── manifests/                     # tracked：split、mix、抽样与 hash
│   └── reports/                       # tracked：小型 EDA JSON/CSV/Markdown
├── third_party/axis_openpi/           # ignored：按 commit clone 的 OpenPI
├── patches/axis_openpi/               # tracked：最小、可重放 patch
├── tools/axis_pi05/                    # tracked：bootstrap/EDA/sampler/norm/train/review
├── outputs/axis_pi05/                  # ignored：checkpoints、logs、W&B offline 数据
└── reviews/axis_pi05/                  # ignored：H.264、contact sheet、HTML gallery
```

`.gitignore` 至少覆盖：

```text
/data/raw/
/data/models/
/data/processed/
/third_party/axis_openpi/
/outputs/axis_pi05/
/reviews/axis_pi05/
```

不得使用 Git LFS 上传数据或模型。服务器重建依赖 tracked lock，而不是复制开发机绝对路径。

## 3. Phase 0：从旧仓库提取最小可复用能力

### 3.1 只迁移这些来源

| 旧仓库来源 | 新仓库用途 | 处理方式 |
|---|---|---|
| `workspace/training/axis_openpi` 的 OpenPI diff | Axis restore、v3 loader、LoRA config、norm/train launcher | 逐块审计后整理为 tracked patch；删除旧绝对路径和无关 AXIS-clean/DROID 代码 |
| `workspace/docs/PI05_AXIS_POSTTRAIN_RESEARCH.md` | checkpoint、OpenPI、norm 事实依据 | 摘要进入新文档，不整份复制历史结论 |
| `workspace/notebooks/libero_plus_data_qc_zh.ipynb` | EDA 指标和目视检查思路 | 按新 catalog API 重写，不保留旧缓存路径 |
| `workspace/training/build_clean_episode_mix.py` | task/category/length 分层思路 | 重写 sampler；不迁移 T1 专用任务名单和自动质量删除规则 |
| `workspace/training/build_episode_perturbation_map.py` 与 `artifacts/bddl_recovery/` | Plus episode 粗扰动标签恢复 | 保留一对一 `(episode, task, length)` 验证；固定外部 label revision |
| `workspace/eval_pipeline/` | 数值诊断、failure tags、run compare、timeline | 只迁移通用模块，并适配正赛 `pipeline` 数据结构 |
| `workspace/archive/scripts/record_t1_failures_h264.py` | 失败录像与 H.264 转码 | 重写为所有 Track/manifest 通用 recorder |
| `workspace/archive/data_review/*` | clip、contact sheet、本地审阅站点 | 重写路径和 schema，保留 CSV 导入导出能力 |

### 3.2 明确不迁移

- 旧 checkpoint、旧 norm stats、旧 model serving 目录和旧提交 zip。
- 旧 T1 core/related 硬编码、旧分数阈值、旧 Track 专用 sampler。
- 旧 policy vendor、旧打包脚本、旧 evaluator/scorer。
- 旧 OpenPI 中与 `axis_clean_local_droid` continual-pretraining 有关的约 500 行自定义 loader；本项目只做 LIBERO post-training。

### 3.3 验收

- 保存旧 OpenPI HEAD `c23745b5ad24e98f66967ea795a07b2588ed6c79`、旧 diff SHA-256 和 Axis HF revision。
- 迁移 patch 中不得出现 `/home/huang/code/AIROA-contest-libero-plus`。
- `git apply --check` 可在全新 pinned OpenPI clone 上通过。
- 旧 one-step checkpoint 只能作为历史证据；新仓库必须重新完成 restore/forward/backward/save smoke。

## 4. Phase 1：可重建的本机/服务器 bootstrap

### 4.1 数据和模型来源

两个逻辑训练数据源如下。

**LIBERO 标准 20 Hz / no-noops 组：**

1. `IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot`
2. `IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot`
3. `IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot`
4. `IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot`
5. `IPEC-COMMUNITY/libero_90_no_noops_lerobot`

**LIBERO-plus 20 Hz 组：**

6. `lerobot/libero_plus`

**辅助元数据：**

7. `katsukiono/libero-plus-episode-categories`，仅用于恢复该 Plus release 的五个粗类别：`env / lighting / language / camera_pose / sensor_noise`。

**模型：**

8. `axisrobotics/Pi0.5-Axis`

执行阶段先解析每个仓库的完整 commit SHA、许可和文件清单，人工审核后写入 `configs/axis_pi05/data_sources.lock.yaml`；lock 中存在 `main`、短 SHA 或空 revision 时禁止下载和训练。已知 Axis revision 和 Plus revision也要从 API 重新核对后锁定，不依赖本文中的历史值。

### 4.2 Bootstrap 命令契约

实现一个幂等入口：

```bash
bash tools/axis_pi05/bootstrap.sh --lock configs/axis_pi05/data_sources.lock.yaml
```

它必须：

1. 检查 GPU、驱动、磁盘、`git`、`uv`、`ffmpeg`、HF 登录状态；只报告 token 是否存在，不输出 token。
2. clone OpenPI 到 `third_party/axis_openpi`，checkout locked commit，依序执行 `git apply --check` 和 patch。
3. 以 locked revision 下载六个数据仓库到 `data/raw/hf/<org>/<repo>/`。
4. 下载 Axis 到 `data/models/axisrobotics/Pi0.5-Axis/<revision>/`。
5. 校验文件数、总大小、HF/Xet incomplete 文件、关键 metadata 和 SHA-256。
6. 创建独立训练环境和 lock；不修改正赛 evaluator 的 Python 3.10 环境。
7. 生成 `data/reports/bootstrap_report.json`，包含代码/data/model revision、环境版本、GPU、路径和校验结果。

重复运行时，只校验并补齐缺失文件；revision 或 hash 不一致时显式失败，不覆盖现有数据。

### 4.3 服务器等价性

另一台服务器的完整恢复流程必须只有：

```text
git clone
→ checkout experiment commit
→ HF login / 设置 HF_TOKEN
→ bootstrap.sh
→ data audit
→ build manifest
→ compute norm stats
→ train/eval
```

不得要求访问旧仓库、开发机 HF cache 或开发机绝对路径。

## 5. Phase 2：统一 schema、EDA 与数据验收

### 5.1 Canonical sample contract

实现虚拟 unified catalog，不复制两套视频；每个 sample 对外统一为：

```text
observation.images.front  uint8[H,W,3]
observation.images.wrist  uint8[H,W,3]
observation.state         float32[8]
action                    float32[7]
timestamp                 float32
source_fps                20
source                    libero_standard | libero_plus
source_repo/revision/source_episode_index
suite/task/task_index/instruction
perturb_category          base | env | lighting | language | camera_pose | sensor_noise
perturb_label_quality     exact-source | recovered-coarse | unavailable
```

适配器只重命名 camera/key 和 namespace episode/task ID，不改变图像、state、action 或时间顺序。发现任一源不是 20 Hz 时立即停止；不得静默插帧、重复 action 或把 10 Hz metadata 改成 20 Hz。

### 5.2 EDA 必做检查

- 每源/每 suite/task/category 的 episode、frame、时长分布。
- camera key、codec、shape、dtype、range、帧数与 metadata 边界。
- state/action 维度、NaN/Inf、逐维 mean/std/q01/q50/q99/min/max。
- action saturation、零动作比例、gripper 双峰、episode 末端 padding 风险。
- task instruction 重复、suite/task ID 冲突、跨源相同任务映射。
- Plus 标签 CSV 与本地 14,347 episodes 的一对一 join；`task` 或 `length` 任一 mismatch 都失败。
- 确定性抽取每个 source×suite×category×task 的首/中/末帧和双相机 contact sheet。
- 长度/action 异常只产生 `review_flag`，除损坏、NaN/Inf、无法解码外不得自动删除 demonstration。

### 5.3 EDA 输出

```text
data/reports/
├── data_audit.json
├── source_summary.csv
├── task_summary.csv
├── category_summary.csv
├── action_state_quantiles.csv
├── corrupt_or_unreadable.csv
├── manual_review_manifest.csv
└── eda_report.html
```

`eda_report.html` 支持按 source/suite/task/category/review_flag 筛选，展示 front+wrist contact sheet、episode 长度和 action/state 小图。HTML、CSV 和 JSON 可进 Git；原始采样帧/视频默认忽略。

### 5.4 验收门禁

- 所有训练源为 20 Hz、双相机、8D state、7D action。
- 所有数据都能映射到 canonical contract，无 silent fallback。
- train/validation split 之前不得生成 sampling manifest 或 norm stats。
- `data_audit.json` 包含完整 lock hash 和 catalog hash。

## 6. Phase 3：重新设计分层抽样

### 6.1 Split 先于 sampling

用 `sha256(source_repo@revision:source_episode_index:seed)` 在最细可用层级内确定性分割：

- validation：每个 `source × suite × task × perturb_category` 默认 10%，至少 1 条；桶不足 5 条时整桶只进入 train，并在报告中标记 `no_holdout`。
- train/validation episode 必须零重叠。
- 同一个 Plus `source_file` 可同时覆盖多个扰动类别时，优先按 `source_file` 分组，防止同一基础轨迹泄漏；无法满足时输出显式例外清单。
- 公开 T3 示例不进入训练，只作为单独 compositional validation。

### 6.2 三套 sampling 候选

所有 sampler 按 `source → suite → task → category → episode` 层级抽取，使用固定 seed；不通过复制数据文件实现 oversampling。

**M0 `source_suite_balanced`：结构对照**

- 标准 LIBERO 50%，LIBERO-plus 50%。
- 标准五个 suite 各占标准源的 20%。
- Plus 四个 suite 各占 Plus 源的 25%。
- Plus 五个已恢复粗类别在各 suite 内等权。
- suite 内 task 等权，task 内 episode 等权。

**M1 `competition_prior`：默认参考/推荐**

- LIBERO-plus 50%。
- 标准 LIBERO-90 25%。
- 标准 LIBERO-10 15%。
- 标准 Spatial/Object/Goal 合计 10%，三者等权。
- Plus 内先按五个可验证粗类别等权，再按 suite、task 等权。
- 这是参数冻结消融使用的固定 reference mix。

**M2 `natural_capped`：较少人为先验对照**

- 标准 LIBERO 50%，LIBERO-plus 50%。
- 源内按 episode 自然数量采样，但任何 task 的概率不得超过同 suite task 中位概率的 2 倍。
- Plus 每个粗 category 至少获得 Plus 源 15% 的概率，剩余 25% 按自然分布分配。
- 不使用 EDA 图像统计作为真实扰动标签。

### 6.3 Manifest 和 norm 对齐

每套 mix 生成：

```text
data/manifests/<mix_id>/
├── train_episodes.parquet
├── validation_episodes.parquet
├── sampling_weights.parquet
├── distribution_before_after.csv
├── shortfalls_and_exceptions.csv
├── manifest.json
└── manifest.sha256
```

`manifest.json` 必须记录 source revisions、split seed、层级权重、每桶样本数、重复暴露的 p50/p95/p99、catalog hash 和实现 commit。

norm stats 必须针对该 mix 的**有效 sampling distribution**计算。实现方式固定为：按 `sampling_weights` 做 streaming weighted mean/std 和确定性 weighted quantiles；不允许先对原始全集均匀计算再宣称是 mixture stats。每套 sampling 候选对应独立 norm artifact 和 hash。

## 7. Phase 4：OpenPI/Axis 训练实现

### 7.1 固定模型与数据契约

```text
pi05=True
action_horizon=10
discrete_state_input=False
paligemma_variant=gemma_2b_lora
action_expert_variant=gemma_300m_lora
extra_delta_transform=False
ema_decay=None
weight_loader=<repo-local Axis params>/params
assets=<当前 mix 的 fresh norm stats>
```

训练前必须完成：Axis parameter tree restore、缺失 LoRA 参数初始化、完整 key/shape/dtype 对照、one-batch finite loss/gradient 和 checkpoint restore round-trip。

### 7.2 三种可训练参数范围

每次运行保存完整 trainable/frozen parameter path、shape、参数量和 hash；实际集合不符合 regex 预期时停止。

**F0 `lora_only`：第一阶段基线**

- 训练 PaliGemma language branch LoRA。
- 训练 Gemma action-expert branch LoRA。
- 冻结视觉 backbone、两支 base LLM/action expert，以及：
  - `action_in_proj`
  - `action_out_proj`
  - `time_mlp_in`
  - `time_mlp_out`
- 用途：测量 Axis 表征只靠低秩适配能否迁移到 LIBERO mixture。

**F1 `lora_action_proj`：动作接口消融**

- 在 F0 基础上解冻 `action_in_proj`、`action_out_proj`。
- `time_mlp_in/out` 继续冻结。
- 用途：分离动作输入输出映射适配的贡献。

**F2 `lora_action_time`：完整小模块后训练**

- 在 F1 基础上解冻 `time_mlp_in`、`time_mlp_out`。
- 视觉 backbone 和所有 base language/action-expert dense weights继续冻结。
- 这是旧仓库当前 `_freeze_axis_pi05_libero_lora` 思路的正式、可审计版本。

所有 F0/F1/F2 使用相同 LoRA rank、alpha、dropout、optimizer、batch、seed、M1 manifest 和 M1 norm stats。报告中不得把 F1/F2 简写为“纯 LoRA”；必须写明 `LoRA + action projections` 或 `LoRA + action/time adapters`。

### 7.3 默认短程超参数

- optimizer：AdamW，gradient clipping 1.0。
- peak LR：`5e-5`；warmup 300 optimizer steps；之后 cosine decay，到 step 3000 为 `5e-6`。
- precision：自有 GPU 支持时 bfloat16，否则用经过 finite smoke 的 float32/float16 路径；精度一旦选定，全消融锁定。
- global batch：先运行显存 probe，选择可稳定运行的最大值并写入 resolved config；F0/F1/F2 和 M0/M1/M2 比较期间不得改变 global batch。
- seed：模型、sampler、dataloader、eval seed 分开记录，默认主 seed 1000。
- W&B：默认 online，只上传 scalar/table 和小图，不上传 checkpoint、原始视频、数据或 token。

硬件未给定，因此 batch size 不在本文硬编码；执行者只能通过统一 probe 选择一次，不能为不同 arm 单独调 batch。

### 7.4 顺序消融，避免全组合爆炸

**T0：管线门禁**

- F0/M1，1 step；随后 10 steps。
- 验证 restore、loader、new stats、forward/backward、save/resume、推理加载。

**T1：参数范围消融**

- F0/M1、F1/M1、F2/M1 都从同一 Axis params 开始，各训练 500 steps。
- 使用同一 validation manifest 和本地 simulator protocol。
- 根据第 8 节选择一名 trainable-scope winner。

**T2：sampling 消融**

- 用 T1 winner 分别搭配 M0、M1、M2；全部从原始 Axis params 重新开始，各训练 1000 steps。
- 每个 arm 使用自己 mix 对应的 fresh stats。
- 不从 T1 的 500-step checkpoint 续训，以免参数范围阶段污染 sampling 对比。

**T3：最终 3000-step run**

- 选 T2 最优的 `F* + M*`，从原始 Axis params 重跑到 internal step 3000。
- 永久保留 internal steps 500、1000、3000。
- 如果算力允许，额外保留 F0/M1 到 3000 作为最小适配对照；不能用它替代主 run。

### 7.5 Checkpoint 语义修正

OpenPI 当前实现 `max_to_keep=1` 且默认 `keep_period=5000`，会删除短程中间 checkpoint；循环目录名还可能比 internal step 少 1。执行时必须：

- checkpoint 目录以 `int(train_state.step)` 命名。
- `keep_steps={500,1000,3000}` 显式永久保留，而不是依赖 `keep_period=5000`。
- 每个 checkpoint 写入 `checkpoint_manifest.json`：internal step、params hash、norm hash、mix hash、trainable-set hash、optimizer state、source Axis hash。
- 500/1000/3000 均完成独立 restore + one-batch inference，不只检查目录存在。

## 8. Phase 5：评测、失败 H.264 和目视审阅

### 8.1 固定模型选择指标

模型选择不使用公开四题的单一 overall。固定优先级：

1. validation task×perturbation-category macro success。
2. 标准 LIBERO 与 LIBERO-plus source macro 的最差值，防止只提升一侧。
3. compositional/T3 public holdout success。
4. collision rate、goal-reached-but-collision、timeout/stall。
5. jerk、SPARC、trajectory/path 等原始诊断指标。
6. loss、gradient norm 和 action saturation 只作解释，不单独决定胜负。

若候选 macro success 差小于 2 个百分点，选择 collision 更低者；仍持平时选择可训练参数更少者。

### 8.2 每个 checkpoint 的输出结构

```text
reviews/axis_pi05/<run_id>/<step>/
├── summary.json
├── tasks.csv
├── episodes.csv
├── failure_index.csv
├── metrics_by_source_suite_category.csv
├── compare_to_axis_init.json
├── gallery.html
└── cases/<case_id>/
    ├── failure_h264.mp4
    ├── contact_sheet.jpg
    ├── timeline.png
    ├── timeline.json
    ├── episode.json
    └── error.txt                 # 仅环境/服务错误时存在
```

### 8.3 失败视频要求

- 默认只保存 `success=false`、`collided=true`、timeout、policy error 或 simulator error；每个成功 stratum 另保留 1 个 reference case。
- `failure_h264.mp4` 使用 H.264 `libx264`、`yuv420p`、20 FPS、`+faststart`，浏览器/VLC 可播放。
- 视频为 front+wrist 并排，叠加最小可读信息：run/step、task、suite、source、perturb category/quality、variant（若有）、seed、episode、当前 step、success/collision/timeout。
- `episode.json` 保存：完整 instruction、BDDL/task name、init-state/seed、扰动类别和解析来源、checkpoint/mix/norm hash、总步数、首次碰撞步、非目标物体最大位移、goal 首次满足步、failure tags、action chunk 设置。
- `timeline.png/json` 至少展示 reward/goal、collision displacement、action 六轴与 gripper、action norm、replan 边界。
- `gallery.html` 支持按 checkpoint、task、suite、source、category、failure tag、collision、timeout 筛选；点击 card 同时展示视频、contact sheet、timeline 和 metadata。
- 扰动信息未知时写 `unavailable`，不得从画面猜成官方标签；推断型 failure tag 标记 `diagnostic_inference`。

### 8.4 从旧仓库迁移的视觉能力

迁移后至少保留：

- notebook/HTML EDA 总览。
- dataset clip + front/wrist contact sheet。
- 失败 H.264 recorder。
- episode/step 数值诊断和 failure tag。
- checkpoint/run 对比表。
- 本地分页 gallery，支持 CSV 导入导出人工结论。

不得迁移旧页面中的 SmolVLA、T1 专属标题或固定模型路径。

## 9. Phase 6：一键命令和远端执行设计

最终实现以下稳定入口：

```bash
# 1. 环境、模型和六个数据仓库
bash tools/axis_pi05/bootstrap.sh --lock configs/axis_pi05/data_sources.lock.yaml

# 2. 只读数据审计与统一 catalog
python -m tools.axis_pi05.data_audit --config configs/axis_pi05/data.yaml

# 3. 三套 split/mix manifest
python -m tools.axis_pi05.build_mix --all --config configs/axis_pi05/sampling.yaml

# 4. 每套 effective mixture 的 stats
python -m tools.axis_pi05.compute_norm --mix M0
python -m tools.axis_pi05.compute_norm --mix M1
python -m tools.axis_pi05.compute_norm --mix M2

# 5. Dry-run / smoke / ablation / final
python -m tools.axis_pi05.run --stage T0 --dry-run
python -m tools.axis_pi05.run --stage T0
python -m tools.axis_pi05.run --stage T1
python -m tools.axis_pi05.run --stage T2
python -m tools.axis_pi05.run --stage T3

# 6. 固定 evaluator + 失败审阅包
python -m tools.axis_pi05.evaluate --run <run_id> --steps 500,1000,3000
python -m tools.axis_pi05.build_gallery --run <run_id>
```

`run` 命令必须支持 `--dry-run`、resume、单 arm 选择和失败后继续未完成 arm；默认不自动训练下一阶段，必须先写出阶段比较报告并由操作者确认 winner。

所有运行保存：resolved config、Git commit/diff hash、OpenPI commit/patch hash、data lock、catalog/mix/norm/trainable-set hash、GPU/driver/JAX 环境、命令、seed、W&B URL 和输出树 hash。

## 10. 审核标准（后续 Review 必须按此执行）

本节是以后审核实现和实验结果的固定清单。

### 10.1 代码与来源

- [ ] 当前正赛 evaluator/scorer 未被替换或修改语义。
- [ ] OpenPI commit、patch 顺序和 patch SHA 已锁定，可在空目录重放。
- [ ] 实现中没有旧仓库绝对路径、旧提交权重或隐式 fallback。
- [ ] 数据、权重、视频、token 均未进入 Git；tracked lock 能重建它们。
- [ ] 第三方来源、许可、revision 和用途已记录。

### 10.2 数据与 schema

- [ ] 六个 HF 数据 repo 全部按完整 SHA 下载且校验通过。
- [ ] 所有源均为 20 Hz、双相机、8D state、7D action。
- [ ] camera/state/action/gripper/delta 语义逐项检查，不仅 shape 相同。
- [ ] Plus coarse category join 为 one-to-one，task/length mismatch 为零。
- [ ] train/validation 无 episode/source-file 泄漏；例外有清单。
- [ ] 没有插帧、重复 action、仿真合成或静默重写原始 demonstration。

### 10.3 Sampling 与 norm

- [ ] M0/M1/M2 权重与实际抽样频率一致，固定 seed 可复现。
- [ ] task/category 小桶、重复暴露和 shortfall 都有报告。
- [ ] norm 基于该 mix 的有效权重，而不是原始全集均匀统计。
- [ ] Axis 原始 stats 未被误加载；新 stats 同时用于训练与推理 checkpoint assets。
- [ ] norm 文件含 mix/catalog/source hash，所有维度 finite。

### 10.4 模型与训练

- [ ] Axis params restore 后 key/shape/dtype 完整校验。
- [ ] F0/F1/F2 的实际 trainable path 与预期完全一致。
- [ ] F0 确实冻结四个 action/time 模块；F1 只多解冻 action projections；F2 再多解冻 time MLP。
- [ ] 消融间 global batch、LR、seed、precision、数据和评测协议锁定。
- [ ] loss、grad norm、参数 norm、action saturation finite，无 silent OOM batch 变化。
- [ ] 500/1000/3000 以 internal step 计数，checkpoint 未被 retention 删除，均可独立 restore。

### 10.5 评测和可视化

- [ ] 每个候选使用同一固定 task/seed/init-state manifest。
- [ ] success、collision、timeout、environment failure 均计入报告，不删除难例。
- [ ] H.264 为 yuv420p/20 FPS、可播放，front+wrist 与任务/扰动 metadata 对应正确。
- [ ] gallery 能从 CSV/JSON 追溯到 checkpoint、mix、norm、seed 和原始 task。
- [ ] failure tag 明确区分官方事实与诊断推断。
- [ ] 公开 example 结果没有被称为正式 leaderboard 分数。

### 10.6 可复现性与提交前门禁

- [ ] 新机器仅凭 Git + HF 可完成 bootstrap→smoke。
- [ ] `--dry-run` 不创建输出、不下载、不训练、不上传 W&B。
- [ ] 候选不会自动覆盖 submission 目录。
- [ ] 最终 policy server 使用新 checkpoint 内的新 norm stats。
- [ ] 官方 Docker 中 `/health`、`/reset`、`/act`、latency、显存、离线依赖和 zip 大小全部通过。

Review 时发现任一红线即阻止正式训练或候选激活：

- 使用 Axis 原始 stats。
- 把 10 Hz 数据伪装成 20 Hz。
- F0/F1/F2 实际训练参数集合不透明。
- split 后再做会引起泄漏的数据过滤/重编号。
- 500/1000 checkpoint 被覆盖或 step 标签含义不清。
- 失败视频与 JSON metadata 对不上。
- 未锁 revision 的 `main` 下载。
- 评测结果或观测进入后续训练。

## 11. 实施工单顺序与 Definition of Done

执行者按以下工单顺序推进，不能并行跳过门禁：

1. `A1`：Gitignore、目录契约、data/model/OpenPI lock schema。
2. `A2`：bootstrap + pinned OpenPI patch 重放。
3. `A3`：六数据源 adapter + canonical catalog。
4. `A4`：EDA、Plus category join、HTML 数据审阅。
5. `A5`：split + M0/M1/M2 sampler + determinism tests。
6. `A6`：weighted norm stats + train/inference assets contract。
7. `A7`：F0/F1/F2 freeze filters + parameter audit tests。
8. `A8`：checkpoint internal-step/retention 修正 + T0 smoke。
9. `A9`：T1/T2/T3 orchestration、W&B 和 run provenance。
10. `A10`：诊断 evaluator、H.264、timeline、gallery 和 run compare。
11. `A11`：全新目录/server bootstrap rehearsal 与审核报告。
12. `A12`：自有 GPU 正式消融和 3000-step 训练；由操作者逐阶段确认 winner。

当前执行结论：`A11` 已完成。`A12` 已完成 T0、F0/M1 的有效 T1 500-step run 和 checkpoint restore smoke；F1 的正式 500-step checkpoint、F2、T2、T3、官方 evaluator 与审阅包仍未完成，因为 F1 在 V100 上通过多组有限值门禁探针后仍发生可复现数值发散。详见 `data/reports/gpu_execution_summary.json`。

整个计划完成的定义：

- 新仓库能独立下载并审计两组数据和 Axis 模型。
- canonical catalog、三套 sampling manifest 和对应 fresh stats 均可复现且有 hash。
- F0/F1/F2 的 500-step 消融、M0/M1/M2 的 1000-step 消融和最佳 3000-step run 全部有 resolved config 与可恢复 checkpoint。
- 500/1000/3000 每个 checkpoint 都有固定协议指标和可筛选失败审阅包。
- 在另一台 GPU 服务器从 Git clone 开始可重建 smoke；不依赖旧仓库。
- 最终审核按第 10 节全部通过，才允许进入 submission 适配和官方评分流程。

## 12. 当前假设与待执行时确认项

- 自有 GPU 型号、数量和显存尚未写明，因此 batch/precision 通过统一 probe 锁定，不能为不同 arm 单独调整。
- Plus release 只有五个可恢复粗扰动类别，不等同于 LIBERO-plus 环境的全部七轴；报告必须保持这个事实边界。
- IPEC 五套标准数据与 Plus 应为 20 Hz；执行阶段仍必须读取每个 locked revision 的 `meta/info` 逐一确认。
- 本计划不执行数据下载、代码迁移、训练、评测或视频生成；这些由后续执行计划完成。
