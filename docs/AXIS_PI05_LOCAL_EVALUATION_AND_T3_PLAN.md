# AXIS Pi0.5 本地多面板评测与 T3 3000-step 改进计划

> 文档状态：已执行至本地 model-selection；T3 诊断训练已完成 3000 steps，正式训练仍受启动门禁约束
> 编制日期：2026-08-27
> 审计基线：`data/reports/gpu_execution_summary.json` 及当前仓库中的 checkpoint、manifest、evaluator 和可视化实现
> 执行边界：本计划阶段只使用自有本地 GPU；不申请、不占用比赛 GPU，不把本地结果标记为官方分数
> 目标链路：固定评测 manifest → checkpoint-specific policy → 当前 LIBERO rollout → 统一结果 → 自动选择 T3 winner → 从 Axis 基座训练 3000 steps

## 0. 结论与高层决策

1. **先实现完整固定 manifest evaluator，再启动 T3。** 本轮已完成固定 manifest evaluator；`tools/axis_pi05/evaluate.py` 的 legacy external 模式仍只产生 diagnostic 结果。
2. **F0/M0、F0/M1、F0/M2 的 T2 step-1000 checkpoint 都进入候选池。** 不能根据训练 loss 或先验偏好直接指定 F0/M1。
3. **T3 winner 默认由冻结的本地正式 selection panel 自动选出。** 所有候选使用完全相同的任务、初始状态、环境种子、策略种子、最大步数和评分代码，采用成对比较。
4. **扩大本地泛化评测，但严格分离“选胜集”和“诊断集”。** 只允许预先冻结的 `select_core` 决定 winner；标准 LIBERO、LIBERO-90 长尾与扰动压力测试用于衡量泛化、解释风险和处理统计并列，不能在看分后任意挑选有利集合。
5. **允许人工覆盖，但不能静默覆盖。** 必须先生成自动 `winner_decision.json`；若人为改选，另写 `winner_override.json`，记录原 winner、覆盖目标、理由、操作者、时间和涉及风险。覆盖后的候选标记为 `manual_override`，不得称为自动 winner。
6. **T3 从 `axisrobotics/Pi0.5-Axis` 的 `params` 重新初始化。** 使用获胜的 F/M 配方和本项目重新生成的 mixture normalization；不从 T2 step-1000 checkpoint 接着训，除非以后另立实验并修改本计划。
7. **官方 RTX PRO 6000 Blackwell Docker E2E 延后。** 本地完成并不能替代最终官方环境验证；在实际执行该门禁前，状态只能是 `local_model_selection_complete`，不能是 `submission_ready`。

本轮执行结果：`select_core` 82 cases 上 Axis init 与三个 T2 候选均完成 100% coverage，自动 winner 为 `f0_m1_step1000`；T3 dry-run 已通过。由于当前 worktree 非 clean，正式 T3 仍受门禁保护；在显式 `AXIS_PI05_ALLOW_DIRTY_DIAGNOSTIC=1` 与训练命令下完成独立诊断 run `t3_diag_f0_m1_3000_20260827`，`500`、`1000`、`3000` checkpoint 均通过 restore/forward/norm 校验并生成独立 LeRobot 转换目录。`generalization_standard` 与 public T3 manifest 已冻结但尚未 rollout；所有本地结果均保持 `official_result=false`。

## 1. 本次审计判定

### 1.1 已有证据可接受的部分

- T0 F0/M1 10-step、T1 F0/F1/F2-M1 500-step，以及 T2 F0-M0/M1/M2 1000-step 均有 finite 训练和 Orbax restore 证据。
- 三个 T2 候选存在 internal step 1000 checkpoint；checkpoint manifest 已包含参数、训练状态、optimizer、norm、固定样本前向与采样 manifest 哈希等恢复证据。
- 静态门禁和相关测试已通过；单任务、单 episode、单 step 的 evaluator/HTTP/H264 smoke 证明各局部组件可以连通。
- 当前 checkpoint HTTP server 已能把 OpenPI checkpoint 暴露为现有 evaluator 使用的 `/health`、`/reset`、`/act` 接口，是后续 checkpoint-specific policy 的可复用基础。

### 1.2 不能据此宣称完成的部分

- 现有 smoke 不等同于冻结 manifest 的完整评测，也不是官方得分。
- `data/manifests/public_t3_holdout.jsonl` 不存在；`configs/axis_pi05/eval.yaml` 对它的引用当前是悬空配置。
- `validation_episodes.parquet` 是训练数据 split 产物，不能直接当作 simulator rollout manifest。它缺少可复现实验所需的 benchmark、task、BDDL、init-state index、环境/策略种子等运行语义。
- 当前 evaluator 只接受任务数量/统一 seed，不能逐行消费固定 init state；结果记录也缺少 suite、扰动、manifest case id、checkpoint 哈希等字段。
- 当前视频名只由 task 和成功/失败组成，多 episode 时可能覆盖，尚不能生成可靠的 failure review。
- 当前 `evaluate.py` 在没有外部命令产出 `evaluation_result.json` 时会退出；它自身不负责启动策略、执行 manifest、校验覆盖率或进行自动选胜。
- 训练 loss 来自不同 sampling mixture，不是共同验证集指标，不能用于比较 M0/M1/M2。

### 1.3 新发现的审计问题：sampler empirical report 口径错误

`tools/axis_pi05/sampler_audit.py` 当前把“每个 episode 的 `frame_weight`”直接交给 `WeightedRandomSampler`，却没有按 `observed_frame_count` 展开为 frame-level 权重。因此现有 `sampler_audit_M0/M1/M2.json` 中的 `source_empirical_mass` 不是实际训练 loader 的抽样模拟，不能作为 empirical sampler 门禁。

实际 OpenPI loader 在 `third_party/axis_openpi/src/openpi/training/data_loader.py` 中会按每个 episode 的 frame 数展开权重；现有 `loader_audit_M1.json` 也验证了 M1 的权重和为 1、确定性和目标 source mass。这说明目前更像是**审计工具缺陷，而非已经证实的训练采样错误**。但是 M0、M2 尚缺同口径的 actual-loader audit，因此在正式评分前必须补齐：

- 修复 `sampler_audit.py`，改为调用实际 dataset/loader 权重，或按 frame 数进行严格等价展开；
- 重新生成 M0/M1/M2 sampler audit，旧报告标记为 `superseded`，禁止覆盖后无痕消失；
- 为 M0、M1、M2 都生成 actual-loader audit；
- 校验 expected mass、analytic mass、loader mass 与 seeded empirical mass；
- 预设容差，不允许看结果后调整：analytic/loader 绝对误差 `<= 1e-8`，empirical 误差使用基于抽样数的 99% 二项置信界，另加最大绝对误差上限 `0.005`；
- 若 actual loader mass 不通过，则冻结对应 checkpoint 的候选资格，先判断是否必须重训；不得只修报告后继续。

### 1.4 账本一致性问题

部分 `run_state.json` 虽标记 completed，但 checkpoint 列表为空，而 checkpoint manifest 是训练后补生成的。新增只读/可重跑的 reconciler，根据 checkpoint 目录和 manifest 回填统一 inventory，生成差异报告；不得篡改历史 checkpoint 内容。模型选择前要求候选 inventory、internal step 和 manifest 哈希一致。

## 2. 目标产物与目录约定

计划新增以下逻辑组件；文件名可在实现时微调，但职责不可合并成不可审计的单脚本：

```text
configs/axis_pi05/eval_v2.yaml
data/eval_manifests/axis_pi05_eval_v1/
  select_core.jsonl
  generalization_standard.jsonl
  generalization_long_tail.jsonl
  stress_perturbation.jsonl
  public_t3_holdout.jsonl          # 仅在本地确实可枚举 public T3 时生成
  availability.json
  manifest.lock.json
tools/axis_pi05/eval_manifest.py
tools/axis_pi05/eval_runner.py
tools/axis_pi05/eval_result.py
tools/axis_pi05/select_winner.py
tools/axis_pi05/reconcile_runs.py
reviews/axis_pi05/model_selection_v1/
  protocol.lock.json
  candidates.json
  raw/<candidate>/<panel>/
  reports/<candidate>/<panel>/
  comparisons/
  winner_decision.json
  winner_override.json             # 默认不存在；只有人工覆盖才创建
```

现有 `tools/axis_pi05/evaluate.py` 可改为调用上述 runner 的兼容入口，但不得继续把任意 `AXIS_PI05_EVAL_COMMAND` 视为正式评测。环境变量模式只可保留为 `legacy_external`，结果必须标记 `diagnostic_only=true`。

## 3. 固定 simulator evaluation manifest

### 3.1 manifest 不是训练数据 manifest

评测 manifest 必须从当前环境实际注册的 LIBERO benchmark/task/init-state 列表构建，而不是从 demonstration 的 validation parquet 拼接。每行代表一个可执行 rollout case，最少包含：

```json
{
  "schema_version": 1,
  "case_id": "sha256:...",
  "panel": "select_core",
  "selection_eligible": true,
  "benchmark": "libero_t1",
  "suite": "libero_t1",
  "task_id": 0,
  "task_name": "...",
  "language": "...",
  "bddl_path": "...",
  "bddl_sha256": "...",
  "init_state_index": 0,
  "init_state_sha256": "...",
  "env_seed": 1000,
  "policy_seed": 2000,
  "max_steps": 300,
  "perturbation_id": "clean",
  "perturbation_family": "clean",
  "perturbation_params": {},
  "repeat_id": 0
}
```

`case_id` 由所有影响 rollout 的字段规范化后哈希生成。manifest 构建完成后按 `case_id` 排序并锁定 SHA-256。任何任务、seed、init state、最大步数或扰动改变都必须产生新的 manifest version。

### 3.2 本地评测面板

| panel | 建议冻结规模 | 候选范围 | 用途 | 是否参与自动选胜 |
|---|---:|---|---|---|
| `smoke` | 每套件 1 个 task × 1 case | 新接入 checkpoint | 接口、显存、H264 和结果 schema 快检 | 否 |
| `select_core` | 本地可用 public T1/T2/T3 的全部 task × 10 个固定 init states；不足 10 时全部使用 | Axis init + F0/M0/M1/M2 step 1000 | 唯一主选胜集 | 是 |
| `generalization_standard` | LIBERO Spatial/Object/Goal/10 全 40 tasks × 5 init states × 3 固定 seeds | 三个候选；资源足够时加 Axis init | 衡量标准 LIBERO 跨套件泛化 | 否 |
| `generalization_long_tail` | LIBERO-90 中按 task 名哈希固定选 20 tasks × 3 init states × 3 seeds | `select_core` 前二名 | 长尾任务诊断 | 否 |
| `stress_perturbation` | 仅对已实现且通过单元验证的扰动，按 task×family 分层，每层至少 5 cases | `select_core` 前二名 | 位置、物体、观测、动作、组合扰动压力测试 | 否，统计并列时只作预声明 tie-break |

若本地 `libero_t1/t2/t3` 任一套件不可注册，builder 必须在 `availability.json` 写明 `unavailable`、错误和软件版本，不得伪造行。`public_t3_holdout.jsonl` 只有在 public T3 task 与 init state 可由本地环境真实枚举时才生成；否则更新配置使其为 optional，并以现有可用 public 套件完成本地选胜。隐藏官方任务不得推测或重建。

### 3.3 扰动真实性门禁

当前 `TRACK_PERTURBATIONS` 为零参数，部分 environment 方法也未证明实际改变 simulator state。实施 `stress_perturbation` 前必须为每一种扰动加入 effect test：

- 同一 base case 在 clean 与 perturbed 下，目标字段或初始 simulator state 的哈希必须发生预期变化；
- 相同 case/seed 重跑得到相同变化；
- 扰动不得改变任务成功判定或语言，除非该 family 的定义明确要求；
- 未通过 effect test 的扰动不得计入面板，标记 `unsupported`，不能以空配置冒充已评测。

## 4. 固定 manifest evaluator 适配器

### 4.1 checkpoint-specific policy 生命周期

每个候选独占一个顺序评测会话：

1. 校验 checkpoint manifest、internal step、norm 哈希、参数树哈希和 restore validation；
2. 使用显式 checkpoint 路径启动 `checkpoint_http_server.py`，端口动态分配且写入运行记录；
3. 等待 `/health`，执行一次预热和一次 reset/act contract test；
4. 把 checkpoint 路径、配置名、norm 路径、git commit、Python/JAX/CUDA/cuDNN、GPU 型号与 server PID 写入 `run_metadata.json`；
5. 逐行执行冻结 manifest；每个 case 开始前必须 `/reset(instruction, policy_seed)`；
6. 完成后正常关闭 server，验证无遗留进程和显存占用；
7. 不允许一个 server 在运行中切换 checkpoint。

Axis init baseline 必须使用与 post-training 候选相同的 inference transforms 和本项目 mixture norm。若 Axis init 在某一特定 mixture norm 下构成不同基线，必须在 candidate id 中显式编码，不得只写 `axis_init`。

### 4.2 rollout 适配

扩展现有 pipeline，使 runner 可以逐行传入 benchmark、task、init state index、env seed、policy seed、perturbation和 max steps，同时保留原命令行模式。正式 manifest 模式必须：

- 精确加载 manifest 指定的 init state，不再用 `arange(n) % n_available` 隐式选择；
- 环境和 policy seed 分离；
- 每个 case 独立 reset，失败不能污染下一个 case；
- 每条 action 校验 shape、dtype 和 finite；
- 捕获 timeout、server error、environment error、NaN/Inf、video error 等结构化状态；
- 用 case id 生成唯一视频和轨迹文件名，禁止覆盖；
- manifest 行执行顺序可由 protocol 中的固定 order seed 打乱，但所有候选使用同一顺序；为降低热漂移，候选顺序也由预先冻结的 seed 决定并记录。

### 4.3 完整性与失败处理

- 正式结果必须满足 `observed_case_ids == expected_case_ids` 且无重复；否则 panel 状态为 incomplete，不能选胜。
- 模型失败（超时、非 finite action、server 500）按失败 episode 计入，不可只重跑成功部分。
- 基础设施失败允许重试，但要标记 `infra_retry=true`、保留首次失败记录，并对同一 case 的所有候选进行成对重跑，避免候选间条件不一致。
- 评测代码异常不得自动写 0 分并继续选胜；runner 退出非零并保留 partial result。
- 断点续跑只能根据 checkpoint hash + manifest hash + protocol hash 三元组恢复。

## 5. 统一 evaluation result v2

每个候选、每个 panel 生成：

```text
summary.json
tasks.csv
episodes.csv
episodes.parquet
failure_index.csv
trajectories/<case_id>.npz
videos/<case_id>__SUCCESS.h264.mp4
videos/<case_id>__FAIL__<failure_reason>.h264.mp4
gallery/index.html
run_metadata.json
coverage.json
```

`episodes` 至少记录 candidate/checkpoint/manifest/protocol 哈希、case id、suite、task、instruction、init-state index、env/policy seed、扰动 family/params、success、collision、timeout、goal reached、steps、elapsed、failure reason、视频/轨迹相对路径和 error 摘要。

`summary.json` 必须同时提供：

- micro success；
- task macro success；
- task×perturbation macro success；
- suite macro 与 worst-suite success；
- perturbation-family macro 与 worst-family success；
- collision、goal-reached-but-collision、timeout/stall、runtime/model error rate；
- 成功/失败 steps 和 latency 分布；
- 与 Axis init 的同 case paired delta；
- bootstrap 95% CI 和有效 case 数；
- `official_result=false`、`result_scope=local_model_selection|local_diagnostic`。

## 6. 自动选择 T3 winner

### 6.1 候选和统一条件

- 候选：T2 `F0/M0@1000`、`F0/M1@1000`、`F0/M2@1000`。
- 参考基线：`axisrobotics/Pi0.5-Axis` + 与候选一致的 eval transform/norm 口径。
- 三候选必须使用相同 `select_core` manifest、protocol、evaluator commit 和本地硬件类别。
- selection protocol 必须在第一次正式 rollout 前锁定；看过正式结果后改变规则必须升级 version 并整套重跑。

### 6.2 预声明选择规则

依次执行以下规则，不使用训练 loss：

1. **资格门禁：** checkpoint/loader audit、restore、coverage、finite action、结果 schema 全部通过；任一硬门禁失败则候选不合格。
2. **主指标：** `select_core` 的 task×perturbation macro success 最大者获胜。clean-only 时退化为 task macro success。
3. **统计并列：** 对相同 case 进行 paired bootstrap（固定 bootstrap seed，至少 10,000 次）。若第一、第二名差值的 95% CI 包含 0，或绝对差小于 0.02，则视为并列，进入下一条，不能把微小随机差异当确定胜出。
4. **第一 tie-break：** worst-suite success 较高者。
5. **第二 tie-break：** `stress_perturbation` 的 worst-family success 较高者；只有两名都完成相同 stress manifest 时才可使用。
6. **第三 tie-break：** collision rate 较低，再依次比较 timeout/model-error rate、成功 episode 的平均 steps。
7. **仍并列：** 选择参数更新范围更小、运行时更稳定的候选；若三者 F0 相同，则选择 mixture 配方更简单者，并把决定标记为 `statistical_tie`。

不得在选择后再发明加权总分。所有原始指标、置信区间、逐 case 差值与选择路径写入 `winner_decision.json`。该文件至少包含：

```json
{
  "decision": "automatic",
  "winner": "f0_m1_step1000",
  "candidate_checkpoint_sha256": "...",
  "manifest_sha256": "...",
  "protocol_sha256": "...",
  "metric_path": ["qualification", "primary", "tie_break"],
  "runner_git_commit": "...",
  "created_at": "..."
}
```

示例中的 M1 只表示字段格式，不能作为预设结果。

## 7. failure review 与目视诊断

在现有 EDA/gallery 基础上生成 checkpoint 可横向对照的失败页面：

- 顶部显示 candidate、task、完整 instruction、suite、case id、init state、两类 seed、扰动 family 和具体参数；
- 内嵌 H264/yuv420p/20fps 视频，并提供失败时刻、最终 reward、collision、timeout/stall、goal reached 等标签；
- 同一 case 可切换 Axis init、M0、M1、M2 视频，默认并排展示前二候选；
- 支持按 task、suite、扰动、failure reason、candidate、是否相对基线退化筛选；
- 展示动作幅度、gripper、EEF 路径、reward、collision proxy 的时间曲线；
- 输出“共同失败”“仅某候选失败”“相对 Axis 回归”“扰动特有失败”四类索引；
- gallery 只引用结果目录内相对路径，缺视频时显示明确原因，不生成空链接。

失败视频采用唯一命名；每个 task×perturbation 分层至少保存全部失败，并保存一个成功 reference。视频必须经 ffprobe 校验 codec、像素格式、fps 和可解码性。

## 8. T3 3000-step 执行阶段

只有自动 winner 文件和所有上游门禁通过后才能进入 T3：

1. 生成 immutable T3 train config，引用 winner 的 F/M、sampling manifest hash、norm hash、Axis base revision、seed、optimizer/LoRA 超参数；
2. 从 `axisrobotics/Pi0.5-Axis` params 初始化，丢弃其旧训练域 normalization，加载 winner mixture 的新 norm；
3. 使用本地 GPU 训练到 internal step 3000，保留 500/1000/3000 checkpoint；
4. 每个 checkpoint 执行 manifest、params、optimizer、norm、固定前向与 finite restore validation；
5. 训练日志中任何 NaN/Inf 立即停止并写 diagnostic，不生成“有效 checkpoint”标记；
6. T3@500、@1000、@3000 都跑 `select_core`，用于观察过拟合曲线；@1000 和 @3000 跑 `generalization_standard`；最终 @3000 对前期失败高发 strata 跑 `stress_perturbation`；
7. T3 最终报告同时比较 Axis init、T2 winner@1000、T3@500/@1000/@3000；禁止只展示最好点；
8. 若 T3@3000 在 `select_core` 相对 T2 winner 有统计显著回退，保留 checkpoint 但不得自动晋升，进入 learning-rate/early-stop 诊断计划。

本阶段不自动尝试新的合成数据、额外训练数据、未知 normalization 或从 T2 续训；这些属于新实验，必须另开 plan/version。

## 9. 测试、门禁与验收

### 9.1 无 GPU 单元/静态测试

- manifest schema、稳定排序、case id、hash lock 和重复检测；
- unavailable suite 不产生伪造 case；
- fixed init-state index 与两类 seed 精确传递；
- 扰动 effect/determinism test；
- result schema、覆盖率、错误分类和 resume key；
- paired bootstrap 与 winner tie-break 的固定 fixture；
- sampler audit frame expansion 与 actual loader 一致；
- 视频文件唯一命名和 gallery 不覆盖。

### 9.2 本地 GPU 分级门禁

1. 单 checkpoint/single-case smoke；
2. 三 checkpoint/same-case paired smoke；
3. 每个 suite 一个 task 的 mini manifest；
4. 完整 `select_core`；
5. generalization 和 stress panels；
6. T3 训练与定点评测。

每一级只有在报告完整、进程清理和显存释放均通过后进入下一级。不得借用比赛 GPU 补过本地失败门禁。

### 9.3 完成定义

以下条件全部满足后，才可把本地阶段标记为完成：

- [x] M0/M1/M2 actual-loader audit 全部通过，错误的旧 empirical report 已明确 superseded；
- [x] checkpoint/run inventory 一致；
- [x] `select_core` 和诊断 manifests 已冻结并有 lock；
- [x] evaluator 能直接消费 manifest，无需手工外部命令写结果；
- [x] Axis init 和三个 T2 候选的 `select_core` 覆盖率均为 100%；
- [x] 三候选生成统一 summary/tasks/episodes/failure/gallery；
- [x] 自动 `winner_decision.json` 可从原始结果确定性重建；
- [ ] T3 从 Axis base 按 winner 配方训练到有效 internal step 3000；
- [ ] T3 500/1000/3000 的本地横向报告齐全；
- [ ] `gpu_execution_summary.json` 只陈述真实完成项，并明确 official E2E 仍 deferred。

官方 Docker/RTX PRO 6000 Blackwell 验证是后续独立 release gate：需要时再执行相同 checkpoint 的 server contract、单 episode E2E、完整官方允许流程与提交包校验。在那之前所有报告都保持 `official_result=false`。

## 10. 推荐实施顺序与提交边界

每一步单独提交，方便另一 agent 执行和后续 review：

1. `audit: fix frame-level sampler audit and reconcile run inventory`
2. `eval: add immutable simulator manifest builder and locks`
3. `eval: add fixed-case rollout adapter and canonical result v2`
4. `eval: add checkpoint lifecycle runner and coverage gates`
5. `eval: add paired comparison and automatic winner selection`
6. `viz: add case-aligned failure H264 gallery`
7. `tests: cover manifests perturbations metrics retry and selection`
8. `reports: run local T2 model selection panels`
9. `train: launch winner recipe T3 3000-step on local GPU`
10. `reports: evaluate T3 checkpoints and update execution summary`

前七步是代码/CPU 测试阶段；第八步才允许正式本地 rollout；第九步才允许启动 T3 训练。每步提交前记录 `git status --short`，只包含本步文件，避免把当前工作树中的其他未提交资产一起带入。

## 11. 后续 review 时的审核清单

后续审核不要只看“脚本退出 0”或汇总成功率，应至少复核：

1. manifest 是否在看分前冻结，三个候选的 hash 是否完全相同；
2. case 数、唯一 case id、逐候选覆盖率和基础设施重试是否对称；
3. checkpoint、norm、sampling manifest 与 runner commit 是否可追溯；
4. actual loader 的 sample mass，而不是仅看配置中的目标比例；
5. policy server 是否真正加载目标 checkpoint，运行中是否切换或复用错误状态；
6. init-state index、env seed、policy seed、扰动参数是否进入结果行；
7. 扰动是否有可观察 effect，而不是空配置；
8. macro 指标是否按预声明 strata 计算，是否错误用 micro success 代替；
9. paired CI、tie-break 路径和 winner 是否可由原始 episodes 重算；
10. 失败、超时、NaN、server error 是否被保留，是否存在挑选性重跑；
11. H264 视频是否唯一、可解码并与正确 case/task/扰动对应；
12. 本地结果是否始终标记非官方，是否错误宣称通过 Blackwell/官方 E2E；
13. T3 是否从 Axis base params + winner mixture norm 开始，而非意外从 T2 续训；
14. T3 internal step、optimizer、restore、finite 与 500/1000/3000 横向结果是否齐全。

## 12. 本计划明确不授权的事项

- 不使用比赛 GPU，不启动官方 Blackwell Docker；
- 不在计划编写阶段启动 evaluator、训练或批量下载；
- 不预先指定 F0/M1 为 winner；
- 不把 validation demonstration episode 当 simulator holdout；
- 不伪造不可获得的 public/hidden T3 task；
- 不用训练 loss 替代正式 rollout score；
- 不在结果出来后改变 selection manifest 或指标而不升级 protocol；
- 不把本地 smoke、本地分数或 LeRobot API 兼容性测试称为官方比赛结果。
