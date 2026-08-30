# AXIS Pi0.5 N1：Evaluator / Runtime 非确定性定位与恢复执行计划

> 制定日期：2026-08-30  
> 输入提交：`ca4a514000abb9226f811dd0b43184d3b4705893`  
> 输入状态：`r6_evaluator_and_fold_reproducibility_failed`  
> 作用范围：只定位并修复 N1.2；不得启动 parity grid、V160/H160、训练、fold/export 或官方评分。  
> 本计划结束时必须再次停止，由人工决定是否恢复主计划 N1.3。

> 后续审核说明（2026-08-30）：D1 的 fresh-process task11 差异在 E1 为 9 个 uint8 元素、最大差 1，但到 warm-up step 10 的首次 policy boundary 已恢复 exact；原 comparator 将任意中间 renderer mismatch 直接等同于整例失败，无法区分 dynamics、policy boundary 与 EGL jitter。后续执行以 `AXIS_PI05_N1_POLICY_BOUNDARY_AND_DIAGNOSTIC_ZIP_PLAN.md` 为准；本文件及原始报告保留为历史证据，不覆盖。

## 执行结果（2026-08-30）

D0 已完成并通过无 GPU 测试；D1 已按“baseline → seeded constructor → fresh process”顺序执行到首个硬失败。seed-before-constructor 修复使 3 个 probe case 的同进程比较达到 exact，但 fresh process 仍在 environment observation 层失败：`C20-task11-init0` 的 E1 `agentview_image` 有 9 个像素、最大绝对差 1；simulator state、wrist image 和其他 case exact。baseline fresh process 则在 `C20-task00-init0` 的 E0 simulator state 已分叉。

因此当前状态锁定为：

```text
n1_nondeterminism_root_cause_unresolved
```

按第 5.3、5.5 和第 8 节，D2 policy-only、D3 integration、D4 N1 恢复均未启动；不得进入主计划 N1.3 或任何 parity/V160/H160/训练阶段。完整产物见 `data/reports/axis_libero_30k/post_r6_lora/n1_nondeterminism/final/summary.json` 和 `N1_NONDETERMINISM_REPORT_ZH.md`。

## 1. 审核结论

### 1.1 已确认

- N0 审计固化和 action trace 落盘有效；项目测试复核为 `82 passed, 7 skipped`。
- `policy_seed` 在 episode 开始设置一次；execute8 的 `chunk_discard` 只清空未执行 action，不重新设置 RNG。
- Axis step-0 与 R6 step-500 都失败，因此当前证据不支持把问题归因于 LoRA、fold 或 checkpoint step。
- `QUANTILES` 与 action range 处理正确；本问题不得通过修改 normalization、action clip、rounding 或 gripper contract 解决。
- 两次 replay 的第一步 action 差异通常只有约 `1e-8～4e-6`；差异随后在闭环内放大：

| 阈值 | Axis 首次超过阈值的 step，中位数 | R6 首次超过阈值的 step，中位数 |
|---|---:|---:|
| `1e-6` | 8 | 8 |
| `1e-4` | 16 | 16 |
| `1e-2` | 32 | 32 |
| `1e-1` | 59 | 56 |
| `1.0` | 112 | 76 |

- 当前报告所称 Axis `15/20`、R6 `16/20` “outcome mismatch”包含 success、collision、goal、timeout、failure reason、steps 中任一字段变化。真正的 success 标签翻转是 Axis `6/20`、R6 `3/20`；aggregate success 分别为 Axis `12→14/20`、R6 `15→14/20`。
- 对已有 H.264 的首帧做只读解码比较，Axis 有 `15/20`、R6 有 `14/20` 首帧不一致。首帧是在 manifest init state 和 10 个 zero-action warm-up 之后、第一次 policy action 之前采集的。这是 environment construction/reset/init/warm-up 路径存在差异的强证据；最终结论仍须由新增的 raw observation/simulator hashes 确认，不能只用有损视频代替原始证据。

### 1.2 当前报告器存在的语义错误

`tools/axis_pi05/evaluator_reproducibility.py` 当前记录：

```text
index        = action trace 中第一个非零差异的位置
max_abs_diff = 同一 case 整段 trace 的最大差异
```

这两个值被放入同一个 `first_action_trace_numeric_mismatch` 对象，容易误读成“该 index 的差异就是约 2”。必须拆成：

```text
first_nonzero_index
abs_diff_at_first_nonzero
case_max_abs_diff
case_max_abs_diff_index
first_step_exceeding_{1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1}
```

同时拆分：

```text
success_flip_count
termination_tuple_mismatch_count
steps_mismatch_count
failure_reason_mismatch_count
trace_shape_mismatch_count
trace_byte_mismatch_count
```

不得继续用单一 `outcome_mismatch_count` 概括这些不同问题。

## 2. 核心假设与定位顺序

按以下顺序定位，后一级不得替代前一级：

```text
D0 报告与观测证据正确
  ↓
D1 environment-only：模型未参与时，init / warm-up / fixed action 是否可重复
  ↓
D2 policy-only：冻结 observation 与 seed 后，same-process / fresh-process 是否可重复
  ↓
D3 integration：environment + policy 的第一 chunk 与闭环何时分叉
  ↓
D4 重新执行 N1.2 或提交“统计评价协议”人工决策
```

优先假设是 environment seed 设置时机。当前 `EnvironmentManager.create_env()` 先构造 `OffScreenRenderEnv`，随后才调用 `env.seed(...)`；LIBERO 的 `seed()` 实际设置全局 `numpy.random`。因此 constructor 中发生的采样可能受前一 case、import 顺序或进程状态影响，并且无法被事后 seed 撤销。

这只是待验证假设。不得未做 D1 probe 就直接修改 seed、XLA flag 或评价阈值。

## 3. 全局边界

- 默认仅使用一张明确空闲的批准 GPU；GPU0 或带外部 PID 的 GPU 不得使用。
- evaluator 外层进程和 policy server 子进程必须使用同一受限 GPU namespace。不得只给 policy subprocess 设置 `CUDA_VISIBLE_DEVICES`，同时让 MuJoCo/EGL evaluator 看见全部 GPU。
- 记录 outer process 与 child process 的：

```text
CUDA_VISIBLE_DEVICES
MUJOCO_GL
MUJOCO_EGL_DEVICE_ID
JAX_PLATFORMS
XLA_FLAGS
CUBLAS_WORKSPACE_CONFIG
PYTHONHASHSEED
OMP_NUM_THREADS
MKL_NUM_THREADS
```

- `gpu0_touched=false` 必须来自运行前/中/后的 GPU PID/显存采样证据，不能在 summary 中硬编码。
- 保持 native 10 Hz、execute8、300 max steps、10-step warm-up、R6 fresh norm、ACTION/STATE `QUANTILES`。
- 禁止通过 action rounding、截断小数、额外 clip、固定输出缓存或复用上次 action trace 伪造确定性。
- 禁止修改 checkpoint、数据、norm、policy seed、init state 或任务顺序。
- 新产物写入：

```text
data/reports/axis_libero_30k/post_r6_lora/n1_nondeterminism/
```

## 4. Phase D0：修复报告与增加 snapshot 能力（无 rollout）

### D0.1 comparator 修复

修复第 1.2 节字段，并为每个 case 输出：

- first-action max diff；
- 每个 chunk 起点的 max diff；
- common-prefix 长度；
- success 是否翻转；
- aggregate success/collision/timeout 的两次值与差；
- shape 不同时只比较公共前缀，不把较短 trace 之后的空缺冒充数值差。

旧 N1 产物不可覆盖；用新 comparator 对旧 trace 生成 `legacy_reanalysis.json`。

### D0.2 原始 snapshot schema

新增只读 probe schema，每个 checkpoint 至少可记录：

```text
case_id / task / BDDL / init hashes
process/env/thread/GPU provenance
model XML or equivalent structural hash
flattened simulator state hash and numeric artifact
qpos / qvel / controller state
raw agentview and wrist image
EEF pos / quat / gripper qpos
serialized HTTP observation hash
OpenPI-transformed image/state/token/mask hashes
policy RNG key before inference
normalized full 10-action chunk
external full 10-action chunk
```

float arrays必须以 dtype/shape 明确的无损格式保存；hash 只作索引，numeric artifact 必须保留以便定位第一处差异。

### D0.3 测试

至少加入：

- first-nonzero diff 与 case-max diff 不得混写；
- success flip 与 steps/failure mismatch 分开计数；
- trace shape 不同仍能安全分析 common prefix；
- `gpu0_touched` 缺少监控证据时不得标为 false；
- snapshot dtype/shape/hash round-trip；
- outer/child GPU namespace 不一致时失败。

执行：

```text
.venv_axis_pi05/bin/python -m py_compile tools/axis_pi05/*.py pipeline/*.py
.venv_axis_pi05/bin/pytest -q tests
git diff --check
```

D0 形成独立 clean commit 后才可进入 D1。

## 5. Phase D1：Environment-only 确定性阶梯

此阶段不加载 Axis/R6 policy，不启动 JAX inference。先使用 `C20-task00-init0`、`task02`、`task11`，分别覆盖旧视频首帧相同和不同的 case；三项通过后再扩到 C20。

### D1.1 记录四个 checkpoint

每次 environment probe 记录：

```text
E0 OffScreenRenderEnv constructor 完成后
E1 env.reset() 后
E2 manifest init state 设置并 sim.forward / observable refresh 后
E3 每个 zero-action warm-up 后，共 10 个
```

每点保存 model/XML hash、sim state、qpos/qvel、controller state、两路 raw image 和 8D policy state。

### D1.2 原样路径基线

在不修改当前代码的情况下运行：

1. 同一进程内创建两个独立 env；
2. 两个 fresh process；
3. 固定 64 步 all-zero action script；
4. 固定一段已保存的 finite action script。

报告第一处发生差异的 checkpoint/field/step，不只给最终 hash。

### D1.3 seed-before-constructor 最小修复

若差异在 E0/E1 已出现，做单变量修复：在构造 env 之前设置 case-local seed：

```text
random.seed(env_seed)
numpy.random.seed(env_seed)
torch.manual_seed(env_seed)
```

然后构造 env，再调用 `env.seed(env_seed)`，在 reset 前重新设置相同 case-local RNG。进程启动时固定 `PYTHONHASHSEED`；不得依赖前一个 case 留下的全局 RNG 状态。

init state 使用 LIBERO wrapper 的 `env.set_init_state(init_state)` / `regenerate_obs_from_state` 官方路径，而不是只调用 `sim.set_state_from_flattened + forward`；保留原 10 个 zero-action warm-up。先用 probe 比较两种 init 调用的观测/controller 差异，再采用与官方参考 evaluator 一致的路径。

### D1.4 case isolation

每个 case 开始时重新建立 case-local RNG state；结束后不得把本 case 的全局 RNG 消耗作为下一个 case 的隐含输入。若库需要 global RNG，保存/恢复 Python、NumPy、Torch RNG state，并在 report 中记录。

### D1.5 D1 门禁

在相同代码、硬件和固定 action 下，三 case 后 C20 必须满足：

```text
model/XML hash                     exact
post-init sim state/qpos/qvel      bitwise exact
10-step warm-up numeric state      bitwise exact
raw uint8 images                   bitwise exact
8D policy state                    bitwise exact
64-step zero-action trajectory     bitwise exact
fixed-action trajectory            bitwise exact
```

如果 numeric state exact 但 image 不 exact，继续定位 renderer/GPU mapping；不得进入 policy-only。若 fresh-process 仍不 exact，比较 constructor/import/RNG/model XML，不能用 tolerance 跳过。

D1 通过状态：

```text
evaluator_environment_determinism_locked
```

## 6. Phase D2：Policy-only 确定性阶梯

仅使用 D1 产出的一个 bitwise-frozen raw observation fixture。environment 不运行。

分别测试 Axis step-0 与 R6 step-500：

### D2.1 同一 server process

在同一加载完成的 policy 中：

1. `episode_reset(seed=S)`；
2. 对同一 observation 请求完整 10-action chunk；
3. 再次 `episode_reset(seed=S)`；
4. 输入完全相同 observation；
5. 比较 RNG/noise、transformed inputs、normalized/external actions。

门禁为所有中间 artifact 与 action chunk bitwise exact。

### D2.2 两个 fresh server process

在同一物理 GPU、同一 env flags 下启动两个 fresh server，输入同一 fixture/seed：

- transformed image/state/token/mask 必须 bitwise exact；
- RNG/noise 必须 bitwise exact；
- 报告 internal/external action 的 max/p99/per-dim diff；
- gripper sign 和所有离散化/掩码必须一致。

若只有 GPU kernel 输出出现约 `1e-7` 差异，先验证本机 JAX/XLA 支持的 deterministic options，再做一次一变量矩阵。任何未知或不受支持的 `XLA_FLAGS` 必须在 probe startup 失败并记录，不能进入正式 evaluator。可审计的候选包括运行时实际支持的 deterministic GPU flag 和 `CUBLAS_WORKSPACE_CONFIG`；不得凭记忆静默追加 flag。

### D2.3 D2 分支

优先通过条件：fresh process full chunk bitwise exact。

若 same-process exact、fresh-process only 有 `<=1e-6` 的 FP32 差异，但所有 input/RNG/noise exact，则记录：

```text
policy_fresh_process_fp32_jitter_bounded
```

这不是 N1.2 通过；进入 D3 前保留该分支，禁止 action rounding。若差异大于 `1e-6`、gripper sign 改变或 input/RNG 不 exact，停止定位，不得进入闭环。

## 7. Phase D3：Integration 分叉定位

### D3.1 first-chunk C5

选择 C20 的固定 5 case；两次 fresh run 分别保存：

- E0–E3 environment snapshots；
- 第一个 HTTP request raw/transformed hashes；
- 第一个完整 action chunk；
- 第一次执行的 8 个 action；
- 第一次 chunk discard 前后的 RNG key。

只有 environment snapshots exact 后，才允许把 action 差异归为 policy/runtime。

### D3.2 open-loop observation replay

从第一 run 保存 observation sequence；第二次不运行环境反馈，只把同一 sequence 喂给 policy。这样区分：

- open-loop 仍差异：policy/runtime；
- open-loop exact、closed-loop 分叉：environment feedback / 数值放大。

### D3.3 closed-loop C5 → C20

先 C5 两次；若 action trace/outcome exact，再扩 C20 两次。失败时报告 divergence curve、success flips 和 aggregate count，禁止只给整段最大值。

## 8. Phase D4：N1.2 恢复判定

### 路径 A：严格确定性恢复

满足以下全部条件：

- D1 environment fresh-process bitwise exact；
- D2 same/fresh policy full chunk bitwise exact；
- D3 C20 两次完整 action traces、success、steps、failure reason exact；
- seed/chunk/norm/provenance/GPU boundary 全部通过。

状态：

```text
r6_evaluator_and_fold_reproducibility_locked
```

写最终报告后停止，等待人工批准恢复主计划 N1.3。

### 路径 B：环境严格确定，但 GPU fresh-process 存在有界 jitter

仅当 D1 exact、D2 input/RNG/noise exact 且 policy action jitter `<=1e-6` 时允许进入本分支。执行同一模型、同一 C20、同一 manifest seed 的至少 5 次独立 replay，分别对 Axis/R6 报告：

```text
aggregate success mean/range/std
per-case success frequency
success label flip rate
collision/timeout frequency
action divergence threshold curve
process/GPU/renderer provenance
```

同时形成一份“从 exact closed-loop 改为 replicated statistical evaluation”的协议提案，包括 V160/H160 的 replicate 数、cluster bootstrap 和最小可识别提升。到此状态必须是：

```text
n1_exact_closed_loop_unavailable_statistical_protocol_pending_human_review
```

不得自动修改主计划门槛、启动 V160/H160 或训练。是否接受统计协议由人工决定。

### 路径 C：environment 或 policy 输入/RNG 不确定

状态：

```text
n1_nondeterminism_root_cause_unresolved
```

立即停止。不能用多次投票掩盖基础环境或 seed contract 错误。

## 9. 必需产物

```text
data/reports/axis_libero_30k/post_r6_lora/n1_nondeterminism/
  D0_LEGACY_REANALYSIS.json
  D0_REPORT_SCHEMA.json
  environment_probe/
    baseline_summary.json
    seeded_constructor_summary.json
    C20_environment_determinism.json
  policy_probe/
    axis_step0_same_process.json
    axis_step0_fresh_process.json
    r6_step500_same_process.json
    r6_step500_fresh_process.json
    deterministic_runtime_matrix.json
  integration/
    C5_first_chunk.json
    C5_open_loop.json
    C5_closed_loop.json
    C20_replay_summary.json
  final/
    summary.json
    N1_NONDETERMINISM_REPORT_ZH.md
```

每个 summary 必须含 root/nested commit、完整环境变量、Python/JAX/jaxlib/CUDA/driver/MuJoCo/robosuite/LIBERO revisions、physical/visible GPU mapping、PID 采样、manifest/checkpoint/norm hashes 和所有输入输出 artifact hashes。

## 10. 停止条件

任一条件出现立即停止，不自动改变阈值或继续主计划：

- root/nested worktree 在 probe 前不 clean；
- outer evaluator 与 policy child GPU namespace 不一致；
- GPU0/外部 PID 所在 GPU 被触碰或缺少监控证据；
- environment model/init/warm-up/fixed-action 任一不 exact；
- policy transformed input、RNG 或 noise 不 exact；
- fresh-process policy jitter 大于 `1e-6`；
- 通过 rounding/clipping/caching 改变官方 action contract；
- 修改 10 Hz、execute8、warm-up、norm、manifest seed 或 checkpoint；
- 覆盖旧 N1 失败证据；
- 未完成人工审核就启动 parity grid、V160/H160、训练或 export。

执行 Agent 应先完成 D0，再按 D1→D2→D3 的证据顺序推进；不得从闭环差异直接猜测 XLA，也不得因为两次 aggregate success 接近就宣布 evaluator 可用于选优。
