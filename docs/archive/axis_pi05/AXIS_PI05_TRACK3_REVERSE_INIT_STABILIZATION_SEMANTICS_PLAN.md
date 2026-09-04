# AXIS Pi0.5 Track 3：Reverse Init Stabilization 语义校准计划

> 制定日期：2026-09-03（Asia/Tokyo）
> 输入执行 commit：root `ed0ec3bff61c09a213300181959491dd107bf778`，nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`
> 当前状态：`track3_reverse_init_stabilization_semantics_planned`
> 唯一授权：执行 I0–I4，随后 hard stop
> 不授权：修改 BDDL/region/predicate、SE(3)、抓取/demo、Teleop、批量生成、训练、norm、FAST/ZIP、官方 GPU、上传或比赛提交
> 生命周期：I0–I4 已由 `0dec0b1` 执行并按 I2 门禁停止；本文件仅作历史审计，不再授权执行

## 0. 目标

本计划只判断：LIBERO sampler 生成的 reverse start，在官方评测式 5 个零动作物理步之后，operator-critical source relation 是否成立且稳定。

不能继续使用“reset 返回瞬间全部 BDDL init atoms 必须 runtime true”的门禁，也不能直接删掉失败 atoms。先以原始 forward BDDL 作为对照，证明 initializer 与 runtime predicate 的实际关系，再对 reverse critical relation 做固定时点验证。

新产物写入：

```text
data/reports/axis_libero_30k/track3_reverse_init_stabilization/
```

## 1. I0：不可变输入锁（CPU only）

输出 `i0_release_lock.json`，锁定：

1. root `ed0ec3b...`、parent `3748c2e...`、nested OpenPI `b58aff0...` 与 root/OpenPI clean；只读记录 `LIBERO-plus` dirty porcelain；
2. R0/R1/R2/R4、R2 per-run、旧 D2/G1、三个 v2 BDDL/spec、原始 task10/task20/task19 forward BDDL 与本地 LIBERO evaluator/source hashes；
3. R2 的 7 条 init truth、1 条 goal truth、关键 object/site pose、worker/schema/status 与 `R3=0/9`；
4. exact/pending/accepted=`0`，无 SE(3)/demo/训练/Teleop/ZIP/上传；
5. 旧 R2/R4 evidence immutability。

任一 hash/count 漂移或 root/OpenPI dirty，状态置为 `track3_reverse_init_stabilization_input_drift` 并停止。

## 2. I1：门禁语义与报告修正（CPU/code only）

### I1.1 predicate 分类

为每族显式注册且测试：

```text
task10 critical source  (On akita_black_bowl_1 plate_1)
task20 critical source  (In orange_juice_1 basket_1_contain_region)
task19 critical source  (Open wooden_cabinet_1_middle_region)
```

- critical source relation 必须在 stabilization 后 runtime true，并在 measurement window 连续 true；
- reverse goal 必须连续 false；
- 其余 BDDL init atoms 标为 background initializer constraints：逐条报告 truth/geometry/contact，但不在没有 source contract 证据时等同于 episode-invalid；
- 任何 critical/background 分类必须来自公开 forward→reverse operator 映射，不得按本次 false 结果动态挑选。

### I1.2 三态报告

所有 gate 使用 `passed/failed/not_evaluated`，同时保留 `executed`：

- 早停前已经判定的 predicate 不丢失；
- settle、controlled destruction 等未运行项必须是 `not_evaluated`，不能填 false；
- `no_grasp_or_demo_generation` 等 provenance 事实独立记录，不依赖 worker 是否走到末尾；
- accepted gate 只接受所有 required gate=`passed`，`not_evaluated` 永远不能被视为通过。

### I1.3 固定 stabilization contract

- action 固定为 7D 全零 OSC_POSE；gripper `0` 是增量 hold，不使用 `-1/+1`；
- official-equivalent checkpoint 固定为 step `5`；只额外观测 `0,1,10,20` 作诊断，不得用后验最优 step 替代 step 5；
- step 5 后的 measurement window 固定 20 steps；controller、20 Hz、threshold 保持 R2 不变；
- R2 的 t=0 failure 永久保留，不重标。

CPU tests 必须覆盖分类不可变、三态 early-stop、required `not_evaluated` fail closed、5-step checkpoint、truth/detail 与 return-code 契约。失败即停止。

## 3. I2：原始/反向初始化时序诊断（CPU/no-renderer）

不修改 BDDL，不使用 GPU。每个 fresh process 都使用 `has_renderer=false`、`has_offscreen_renderer=false`、`use_camera_obs=false`。

### I2.1 task10 配对矩阵

原始 forward BDDL 与 v2 reverse BDDL 各运行 3 个 fresh native resets。对每次的 step `0,1,5,10,20` 记录：

- 全部 init/goal predicate truth 与 detail；
- bowl/plate/table-region pose、bowl↔plate contact pair、COM XY/Z difference；
- robot、target/non-target qpos/qvel 与 scene drift；
- original task 的 critical forward source/goal 与 reverse task 的 critical reverse source/goal。

必须回答：原始官方 BDDL 的 background init atoms 是否也可能在 t=0 或 step5 runtime false，以及 reverse bowl-on-plate 是否在固定 step5 建立接触。

### I2.2 task20/task19 佐证

两族 reverse BDDL 各运行 3 个 fresh resets，同样记录 `0,1,5,10,20`；critical source 分别为 `In orange juice basket` 和 `Open middle drawer`。background false 只作诊断。

I2 通过门禁：

1. 全部 12 个 fresh processes schema/returncode/finite 完整；
2. 每个 reverse family 的 critical source 在 step5 为 true，且 step5→20 连续 true；
3. reverse goal step5→20 连续 false；
4. named scene/robot stability 通过既有阈值；
5. task10 原始 forward 对照足以证明 background initializer truth 不能被当作完整 episode validity gate；否则保持 full conjunction，不得自行放宽。

任一 critical source 在 step5 不成立，状态置为 `track3_reverse_init_critical_relation_failed_<family>` 并直接 I4 收尾；不得调整到 step10/20、修改 sampler 或进入 I3。

## 4. I3：settled-init 动态 release（仅 I2 通过，GPU3 boundary）

运行前检查 GPU PID，只允许自然空闲 GPU3，GPU0 禁止。锁定代码/spec/threshold/stabilization hashes，之后不得修改。

### I3.1 task10 单次 canary

固定一次 task10 fresh process：native reset → 5 个零动作 stabilization → 20-step measurement → critical controlled destruction/restore。通过要求：

- critical `(On bowl plate)`、reverse goal、named settle、controlled destruction、schema/returncode 全部通过；
- background initializer truth 完整报告但只按 I2 已批准的 contract 解释；
- HDF5 state 未作为 predicate oracle；无抓取/demo/pending。

失败立即 I4，禁止 retry。

### I3.2 三族 `3/3`

I3.1 通过后，以完全相同代码/contract按 task10→task20→task19，各 3 个 fresh processes，共最多 9 次。不得对失败族追加运行。全部通过状态：

```text
track3_reverse_semantics_v2_stabilized_dynamic_validated_pending_se3_review
```

失败状态：

```text
track3_reverse_semantics_v2_stabilized_dynamic_failed_<family>_<field>
```

两种结果均进入 I4；通过也不授权 SE(3) 或 demo。

## 5. I4：收尾

- 输出 fail-closed `summary.json`、中文报告、I2 time-series 与 I3 per-run（若执行）；
- 明确区分 t=0 R2 failure、step5 critical result、background initializer truth、not evaluated 与真实失败；
- 更新 plan index/handoff/machine JSON；旧 evidence 不覆盖；
- 小型 code/report/test 纳入 Git，视频/NPZ/model 不入 Git；
- 运行 JSON/YAML parse、`py_compile`、Markdown path、定向测试及：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. .venv_axis_pi05/bin/pytest -q tests
git diff --check
```

- 形成 root/OpenPI clean commit 后 hard stop；不得 push、上传或比赛提交。

## 6. 本计划之后（不授权）

- I3 全通过：下次审核才可恢复 exact-model HDF5 SE(3) mining 与一次 support demo canary。
- task10 critical 在 step5 不成立：下一计划只比较 same-model forward terminal constructor 与 `ObjectBasedSampler` z/contact 语义，不直接 Teleop。
- background initializer 导致真实碰撞/漂移：只修对应 public BDDL init region，不能删除门禁掩盖问题。
- verified reverse demos 建立后才从 R6 step500 分叉 LoRA；不从 Axis init 重训。
