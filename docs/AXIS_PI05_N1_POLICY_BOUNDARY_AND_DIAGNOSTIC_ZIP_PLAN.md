# AXIS Pi0.5 N1：策略边界确定性恢复与诊断 ZIP 就绪计划

> 制定日期：2026-08-30  
> 输入提交：`a202d6b8eaf865b173f0b279686549549656ac59`  
> 输入状态：`n1_nondeterminism_root_cause_unresolved`  
> 本计划修订 `AXIS_PI05_N1_EVALUATOR_NONDETERMINISM_RECOVERY_PLAN.md` 在 D1 的过严判定，但不覆盖或改写原始报告。  
> 作用范围：完成 N1 的 environment/policy/integration 定位，随后最多生成一个 **R6 step-500 诊断 ZIP**；不得自动提交比赛，不得启动 V160/H160 或任何训练。

> 后续任务说明（2026-08-30）：Q2 已按原 hard gate 在 raw-LoRA fresh-process `1.0728836e-6 > 1e-6` 停止。该失败继续保留；若要验证实际提交运行时，后续以 `AXIS_PI05_R6_FOLDED_DEPLOYMENT_AND_DIAGNOSTIC_SUBMISSION_PLAN.md` 为准，单独验证 on-disk LeRobot folded PyTorch、FAST、server 与 ZIP，不得把 deployment 分支通过回写成 raw-LoRA Q2 通过。

## 0. 审核结论与重新分级

### 0.1 已通过复核的事实

- root worktree 与 `third_party/axis_openpi` 均 clean；root 为 `a202d6b8eaf865b173f0b279686549549656ac59`，nested OpenPI 为 `dafa5a27266996eecf28920cacc32f3629776d91`。
- 测试复核为 `90 passed, 7 skipped`；GPU0 未触碰的结论有 before/active/after monitor evidence。
- baseline same-process 与 fresh-process 均在 E0 simulator state 出现分叉，说明原 evaluator 在 constructor 之后才 seed 的顺序确实不充分。
- seed-before-constructor 后，三例 same-process 为 `0/3 mismatch`；fresh-process 的 simulator、controller、qpos/qvel、8D policy state 均一致。
- seeded fresh-process 唯一报告失败为 `C20-task11-init0`：E1 `agentview_image` 有 9 个 uint8 元素差异，最大绝对差 1；同点 simulator state 与 wrist image exact。
- task11 的 agentview 差异还出现在 E2、warm-up step 1 和 step 7，但在 **warm-up step 10、即首次 policy inference 的 observation 边界**，simulator、agentview、wrist 与 8D policy state 均 bitwise exact。
- task11 的 zero-action 64-step 轨迹中，两路图像 sequence hash 可不同，而 simulator/robot state exact；fixed-action 64-step 轨迹则图像与状态均 exact。这更符合 EGL rasterization 的稀疏低位抖动，而不是已证实的动力学随机性。
- `MUJOCO_GL=osmesa` 在本机缺少可用 binding。OSMesa 只是诊断对照，不是比赛 EGL 路径的前置依赖，不再把安装或修复 OSMesa 作为主线门禁。

### 0.2 原状态为何过于悲观

旧 comparator 在一个 case 的任意 section 首次出现 hash 不同后，就把整例标为 mismatch，并输出：

```text
n1_nondeterminism_root_cause_unresolved
```

它没有分别回答以下三个问题：

1. environment 的数值动力学是否一致；
2. 首次及后续 policy inference 实际接收的 observation 是否一致；
3. 即使 raw renderer 存在 1-LSB 抖动，经过 OpenPI image transform 后是否会造成有意义的 action 差异。

因此本次审核不把 D1 判为通过，但将当前人工状态改写为：

```text
n1_seeded_constructor_effective_renderer_jitter_pending_full_c20
```

这不是正式机器状态迁移；执行 Agent 只有在 Q0 产出新的 summary 后才能写入该状态。

### 0.3 当前不能得出的结论

- 不能只因 warm-up step 10 的三例 exact 就声明完整 C20 evaluator deterministic。
- 不能把 9 个值、最大差 1 自动视为无害；未来闭环帧可能出现更大或更频繁的 renderer jitter。
- 不能用 tolerance、图像 rounding、action rounding、缓存观测或复用 action trace 来伪造 exact。
- 不能把 R6 在 B80 的 `49/80`、相对 step-0 的 `+1/80` 当成已证明的性能提升。

### 0.4 本次执行结果（2026-08-30）

- Q0 已完成 comparator/snapshot 重算；Q1 已完成 full-C20 三副本 environment envelope，数值动力学为 20/20 exact，renderer jitter 已量化。
- Q2.1 在 R6 raw-LoRA fresh-process frozen replay 停止：external action 最大差 `1.0728836059570312e-6`，超过 `1e-6` hard gate；失败位置为 `C20-task04-init0-s0` 的 chunk step 8、gripper 维。
- 因此没有继续 Q2.2、Q3 或 Q4，没有修改 production seed 路径，没有训练，也没有生成或上传 ZIP。详细证据见 [Q2 policy sensitivity report](../data/reports/axis_libero_30k/post_r6_lora/n1_policy_boundary/Q2_POLICY_SENSITIVITY_REPORT_ZH.md)。

## 1. 两种“离 ZIP 还有多远”

### 1.1 R6 step-500 诊断 ZIP

当前 R6 已有 folded LeRobot 权重和 QUANTILES processor，但没有当前 R6 对应的 `model_fast.safetensors`，也没有当前 R6 ZIP。旧的 `submission_axis_f2_m1_500_fast_*.zip` 不是本候选，不得改名冒充。

从现在到一个可本地测试、可人工决定是否提交的 R6 诊断 ZIP，还有五个串行门禁：

```text
Q0 重新表达 D1 证据
  → Q1 full-C20 environment envelope
  → Q2 frozen-observation policy impact
  → Q3 closed-loop N1 恢复
  → Q4 parity + FAST export + Docker + ZIP
```

这里不需要重新训练；folded 权重已经存在。真正耗时来自 C20 多副本 rollout、policy 加载与 FAST 导出/本地 E2E，而不是 optimizer training。任何阶段失败都必须停止，所以不得承诺固定完成时间。

### 1.2 经研究选优的正式比赛 ZIP

正式 winner 仍很远。除上述 N1/FAST 工程门禁外，还需要主计划 N2–N8：V160 基线、30K schedule 语义、1K topology、sampling、第二训练 seed、`1K→3K→10K→30K` 分段、V160 选优、一次性 H160、winner 最终 FAST export。至少还有七个研究阶段和真正的长程训练。

因此本计划的 ZIP 只能标记为：

```text
r6_step500_diagnostic_submission_candidate
```

不得标记为 `formal_winner`、`30k_candidate` 或“性能已验证”。

## 2. 全局执行边界

- 只使用人工确认空闲的批准 GPU；GPU0 或任何存在外部 PID 的 GPU不得使用。
- outer evaluator、MuJoCo/EGL 与 policy subprocess 必须共享同一个受限 `CUDA_VISIBLE_DEVICES` namespace，并记录物理/逻辑 GPU 映射。
- 每次 GPU run 前、中、后采样 PID、显存与利用率；无完整 monitor evidence 时不得写 `gpu0_touched=false`。
- 固定 C20 manifest、native 10 Hz、10-step zero-action warm-up、execute8、300 max steps、相同 env/policy seed、R6 fresh QUANTILES norm。
- root 与 nested worktree 在每个 GPU phase 开始前必须 clean；记录 commit、dataset/model/norm/manifest hash。
- 不安装或修复 OSMesa，不切换 Vulkan，不把更换 renderer 当成竞争部署路径。
- Q0–Q4 的所有新产物写到：

```text
data/reports/axis_libero_30k/post_r6_lora/n1_policy_boundary/
```

- 每个 phase 更新机器 summary 与中文线程报告，满足门禁才进入下一 phase；硬失败后不得自动修改阈值重试。

## 3. Phase Q0：重构 D1 比较语义（CPU，只读复算）

### Q0.1 新 comparator 视图

扩展 probe comparator，但不覆盖旧产物。每个 case 分别输出：

```text
constructor_exact
post_reset_exact
post_init_exact
policy_boundary_exact              # E3 warm-up step 10
dynamics_exact                     # model/controller/sim/qpos/qvel/8D state
renderer_exact_by_camera_and_stage
zero64_dynamics_exact
zero64_renderer_exact_by_camera
fixed64_dynamics_exact
fixed64_renderer_exact_by_camera
```

图像差异不得只比较 hash；必须从无损数组计算并保存：

```text
changed_element_count
changed_pixel_count
max_abs_diff
mean_abs_diff
p99_abs_diff
affected_frame_count / total_frame_count
first_affected_frame
camera
```

`changed_element_count` 与 RGB pixel 数不得继续混称。

### Q0.2 用现有三例重算

对 baseline/seeded、same/fresh 现有 summary 重算新视图，生成：

```text
q0_existing_evidence_reanalysis.json
Q0_AUDIT_REPORT_ZH.md
```

明确保留两个结论：

- baseline dynamics failed；
- seeded constructor 的三例 policy boundary/dynamics 通过，但 renderer trajectory envelope 尚未由完整 C20 界定。

### Q0.3 测试与门禁

新增单测覆盖：中间 E1 image mismatch 但 E3 step10 exact、sim exact 但 image jitter、RGB changed element/pixel 的区别、trajectory 第一差异与最大差异。运行：

```text
.venv_axis_pi05/bin/python -m py_compile tools/axis_pi05/*.py pipeline/*.py
.venv_axis_pi05/bin/pytest -q tests
git diff --check
```

Q0 独立 clean commit 后进入 Q1。

## 4. Phase Q1：seeded constructor 的 full-C20 envelope

### Q1.1 执行矩阵

只运行 environment，不加载 Axis/R6 policy。对 C20 全部 20 个 case 各运行 3 个 fresh process replica：

1. constructor 前设置 Python/NumPy/Torch case-local seed；
2. 构造 `OffScreenRenderEnv`；
3. 调用 `env.seed(env_seed)`；
4. reset 前再次设置同一 case-local seed；
5. 使用 production evaluator 当前的 direct init 路径；
6. 执行 10-step zero-action warm-up；
7. 执行 zero64 与固定 finite-action 64-step trajectory。

不将 case 分配到不同 renderer/GPU；Q1 只允许同一物理 GPU、相同 EGL 配置，以隔离 fresh-process 差异。

### Q1.2 硬门禁：数值动力学

20/20 case、任意 replica pair 必须满足：

```text
model/XML/controller exact
post-init simulator/qpos/qvel/8D state exact
E3 warm-up step 10 simulator/qpos/qvel/8D state exact
zero64 simulator/qpos/qvel/8D trajectory exact
fixed64 simulator/qpos/qvel/8D trajectory exact
```

任一数值动力学 mismatch：状态写为 `n1_environment_dynamics_failed_full_c20`，立即停止，不进入 Q2。

### Q1.3 renderer envelope

单独报告 E1/E2/E3、zero64、fixed64 的两路 raw uint8 image envelope；再用当前 OpenPI production transform（orientation、resize、dtype/scale）处理同一对 raw image，记录 transformed tensor 的 max/mean/p99 diff。禁止人工生成噪声代替真实 EGL pair。

分支：

- 20/20 两路图像全程 exact：状态为 `n1_environment_strict_exact_full_c20`，进入 Q2 strict path。
- dynamics exact，图像仅有观测到的 bounded renderer jitter：状态为 `n1_environment_dynamics_locked_renderer_jitter_measured`，进入 Q2 sensitivity path。
- 图像差异超出已声明 envelope、出现非 finite、shape/dtype/camera 错误：停止人工审核。

Q1 不得仅凭“最大差 1”自动放行；必须报告频率、空间范围、相机、帧位置与 transform 后影响。

## 5. Phase Q2：frozen observation 的 policy 敏感性

此阶段 environment 不运行。分别加载 Axis step-0、R6 raw-LoRA step-500 和 R6 folded-dense step-500。

### Q2.1 exact fixture 重放

从 Q1 policy boundary 保存 8 个 task-diverse exact fixtures × 4 个 policy/noise seed；每格记录 raw observation、OpenPI transformed input、token/mask、RNG/noise、internal 10-action chunk 与 external 10-action chunk。

同进程 reset/replay 必须 bitwise exact；fresh-process transformed input/token/mask 必须 exact，action 最大绝对误差必须 `<=1e-6`。任何 gripper sign flip、shape mismatch、non-finite 或 action diff `>1e-6` 立即停止。

### Q2.2 real EGL pair 敏感性

若 Q1 存在 renderer jitter，选取每个相机/阶段的真实最差 pair 和 p99 pair，在相同 state/prompt/policy seed/noise 下分别推理。报告：

```text
raw/transformed image diff
token/mask mismatch
first internal/external action diff
full chunk max/p99/per-dim diff
gripper sign flip
execute8 prefix diff
```

自动进入 Q3 的必要条件：token/mask mismatch 为 0、action finite、无 gripper sign flip，且任一 real EGL pair 的 external chunk 最大绝对差 `<=1e-6`。

若 action diff 在 `(1e-6, 1e-4]`，停止在人工统计协议审核点；不得自动继续。若 `>1e-4` 或出现离散输出翻转，判定 renderer jitter 对 policy 有实质影响并停止。

### Q2.3 production seed 修复

只有 Q1 数值动力学 20/20 通过、Q2 policy 影响通过后，才把经过验证的最小顺序移植到 `pipeline/environment.py`：constructor 前 seed，constructor 后 `env.seed`，reset 前 case-local reseed。不得同时修改 init、warm-up、control frequency、renderer 或 action contract。

必须新增单测保证 seed 调用顺序，并验证 outer/child GPU namespace。形成独立 clean commit。

## 6. Phase Q3：integration 与严格 N1 恢复

### Q3.1 小面板分级闭环

使用修复后的 production evaluator：

1. C5，Axis step-0，3 个 fresh-process replay；
2. C5，R6 folded step-500，3 个 fresh-process replay；
3. 通过后扩到 C20，两模型各 2 个 fresh-process replay。

保持 episode 只在开始时 `policy_seed` reset，execute8 `chunk_discard` 不 reseed。每个 policy inference 边界保存两路 raw/transformed image、8D state、RNG/noise 和完整 action chunk hash。

### Q3.2 strict path 门禁

若 Q1 是 full strict exact，则 N1.2 仍要求：20/20 coverage、outcome/steps/failure reason exact、action shape/float32 bytes exact、stall=0、action finite。

### Q3.3 bounded renderer path 门禁

若 Q1/Q2 证明存在无实质 policy 影响的 renderer jitter，不再要求原始 uint8 image bytes exact，但仍要求：

```text
20/20 paired cases
policy-boundary simulator/8D state exact
transformed-input diff within Q1 measured envelope
first chunk external action max diff <= 1e-6
all action finite, stall=0
success_flip_count=0
termination_tuple_mismatch_count=0
steps_mismatch_count=0
```

完整 trace 若仍不 byte-exact，必须报告 common prefix 与首次阈值越界；停止在人工批准点，不能由执行 Agent 自行把 `<=1e-6` 改成通过整个 N1。

Q3 通过并经人工批准后的状态才可写为：

```text
r6_evaluator_runtime_reproducibility_locked
```

## 7. Phase Q4：R6 parity、FAST export 与诊断 ZIP

只有 Q3 获得人工批准后执行；仍不训练。

### Q4.1 N1.3 parity grid

- `8 observation/prompt × 4 noise/RNG seed`；
- raw-LoRA 与 folded-dense internal/external action 全 finite；
- 最大绝对误差 `<=1e-3`，输出 max/p99/median/per-dim；
- 另跑 same-seed C20 raw/folded 闭环，诚实报告 outcome/steps/action 差异。

### Q4.2 current R6 FAST export

输入必须是当前 R6 folded 目录：

```text
outputs/axis_pi05/axis_libero_lora_folded/
  r6_dual_l2_fresh500_r1_step500/
```

要求：

- LeRobot preprocessor、postprocessor、ACTION/STATE 均为 `QUANTILES`；norm hash 与 R6 fresh norm exact；
- strict missing/unexpected key audit；特别验证 vision patch embedding bias/weight；
- 导出当前候选的 `model_fast.safetensors` 并记录 sha256、size、source checkpoint/fold/norm/git provenance；
- 禁止把旧 F2/M1 fast 权重或旧 ZIP 复制为 R6。

### Q4.3 本地比赛兼容 Docker E2E

启动日志必须明确：

```text
pi05 FAST load model_fast.safetensors
fast=True
n_action_steps=8
action_ema=0.0
```

不得出现 SLOW fallback、动态 remap warning、vision embedding warning、missing/unexpected key。运行固定 C6 或 C20 canary，要求 POST `/act` 有非零 finite action、无 server timeout/stall，并保存完整日志与 action trace。

### Q4.4 生成 ZIP，但不上传

产物命名必须包含 `r6_step500_diagnostic`，并生成独立 manifest：

```text
artifacts/submission_axis_r6_step500_diagnostic_<timestamp>.zip
artifacts/submission_axis_r6_step500_diagnostic_<timestamp>.manifest.json
```

manifest 至少包含 ZIP/model_fast/config/norm sha256、root/nested commit、source folded checkpoint、FAST/Docker evidence、N1/Q1–Q4 summary 路径和明确声明：

```text
performance_status = local_b80_inconclusive_49_of_80
formal_winner = false
auto_upload_authorized = false
```

ZIP 完成状态：

```text
r6_step500_diagnostic_zip_pending_human_submission_decision
```

## 8. 执行 Agent 的停止条件

出现任一项立即停止当前 phase：

- root/nested dirty 或输入 commit/hash 漂移；
- GPU0/外部 PID GPU 被触碰，或 outer/child GPU namespace 不一致；
- Q1 任一数值动力学 mismatch；
- Q2 exact fixture runtime diff `>1e-6`，或 real EGL pair action diff `>1e-6`；
- gripper sign flip、token/mask mismatch、non-finite、shape/dtype/camera mismatch；
- Q3 success/termination/steps mismatch，或需要放宽未预注册阈值；
- raw/fold parity `>1e-3`；
- FAST 缺失、SLOW fallback、vision key warning、Docker POST `/act` timeout；
- 磁盘空间不足以同时保留 folded source、FAST staging 与 ZIP；不得自动删除旧 checkpoint/ZIP。

每次停止必须写出“观测事实、首次差异、影响范围、未执行阶段”，不得伪造下游产物。

## 9. 本计划结束后的人工决策

若只需要尽快获得可尝试的当前候选，人工可在 Q4 后决定是否提交诊断 ZIP。若目标是高置信性能改进，则即使 Q4 通过，也必须返回 `AXIS_PI05_POST_R6_LORA_SCALING_AND_SELECTION_PLAN.md` 的 N2，从 V160 与长程 LoRA 研究继续；Q4 的 R6 ZIP 不能替代 N2–N8。
