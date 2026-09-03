# Track 3 Native Validator R0–R4 独立审核

> 审核日期：2026-09-03（Asia/Tokyo）
> 审核输入：root `ed0ec3bff61c09a213300181959491dd107bf778`，parent `3748c2ebc2a9e5fc66ca90657cf8a4639488f7d7`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`
> 新状态：`track3_reverse_init_stabilization_semantics_planned`

## 1. 获批事项

- R0 evidence lock、旧 D2 immutability 与计数零通过。
- R1 三族真实 CPU/no-renderer schema probe 全部通过；truth/detail 分层、site wrapper 与 worker return-code 契约已闭合。
- R2 是第一条有效 task10 native-reset 语义结果：worker returncode=`0`、`schema_complete=true`、全部 reverse goal=false，HDF5 state 没有参与 predicate oracle。
- R2 失败后未进入 R3，`0/9`；没有 SE(3)、抓取/demo、pending、Teleop、批量生成、训练、FAST/ZIP、官方 GPU、上传或提交。
- root/OpenPI clean，`LIBERO-plus` 既有 dirty 未触碰；执行记录的全量测试为 `182 passed, 7 skipped, 23 warnings`。

hard stop 与资源边界正确。

## 2. R2 逐 predicate 事实

task10 native reset 的 7 条 reverse initial 中 4 条 true、3 条 false：

```text
false  (On plate_1 main_table_plate_region)
false  (On cream_cheese_1 main_table_cream_cheese_region)
false  (On akita_black_bowl_1 plate_1)
true   其余 4 条 fixture/object placement predicates
```

reverse goal `(On akita_black_bowl_1 main_table_akita_black_bowl_region)` 为 false，符合预期。关键几何为：

```text
bowl xy   = plate xy = [0.0636970, -0.0245182]
bowl z    = 1.08 m
plate z   = 0.97 m
delta z   = 0.11 m
```

因此 `ObjectBasedSampler` 已把 bowl 放到 plate 正上方；reset 瞬间运行时 `On` 仍为 false，最直接缺项是 contact/physics settle，而不是 XY sampler 没执行。

R1 的独立随机 native resets 也显示：task10 的 plate-region/bowl-on-plate、task20 的一个背景 site relation、task19 的 plate-region relation会出现 false。说明“所有 BDDL init atoms 在 reset 返回的瞬间必须是 runtime predicate true”不是已验证的 LIBERO contract。

## 3. 本地源码对照

- BDDL `:init` 中的 `On` 首先用于 `_add_placement_initializer()` 构建 `MultiRegionRandomSampler`、`SiteRegionRandomSampler` 或 `ObjectBasedSampler`。
- 运行时 `On(object, object)` 还要求高度、XY `<0.03 m` 和 MuJoCo contact；sampler placement 与 runtime truth 不是同一个判定函数。
- LIBERO 官方 lifelong evaluator 在 `set_init_state()` 后执行 5 个全零 action，注释为 “simulate the physics without any actions”，然后才开始 policy rollout。
- LIBERO README 示例同样先执行 10 个 dummy actions。

所以 R2 已证明“t=0 全 init conjunction 不成立”，尚未证明“评测等价 stabilization 后 operator-critical reverse relation 不成立”。不能直接改 BDDL，也不能把三个背景 initializer atoms 当作 Track3 行为失败。

## 4. 报告层问题

R2 早停后，外层 `_gate_from_run()` 将未执行项填成 false，导致 summary 同时报告：

- `model_state_predicate_oracle_reconciled=false`，尽管 layout 已明确采用合法的 `incompatible_for_predicate_oracle` 路径；
- `settle/controlled destruction=false`，实际为 not evaluated；
- `no_grasp_or_demo_generation=false`，与本轮明确未抓取的事实矛盾。

这不改变 hard stop，但下一阶段必须使用三态 `passed/failed/not_evaluated`，避免把早停后的未运行项误写成实验失败。

## 5. 审核裁决

审核状态为 `remediation_required`：

1. 批准 R1 validator release 与 R2 的 t=0 事实；原 R2 JSON 保持 immutable。
2. 不批准直接进入 R3，也不批准立即修改 reverse BDDL。
3. 先对原始 forward task 和 reverse task执行评测等价的零动作 stabilization time series，区分 background initializer 与 operator-critical source relation。
4. 只有 task10 的 critical `(On bowl plate)` 在固定 5-step stabilization 后成立并连续稳定，才允许一次新的 settled-init task10 canary；通过后才执行三族 `3/3`。
5. 若 critical relation 在 5 步后仍不成立，停止并转向 same-model terminal constructor / sampler 修复，不得靠延长到任意步数挑成功点。

## 6. 不授权事项

本审核不授权 BDDL/region/predicate 修改、SE(3)、抓取/demo、人工 Teleop、批量生成、训练、norm、FAST/ZIP、官方 GPU、上传或比赛提交。

## 7. 审核验证

- 定向测试：`22 passed, 2 warnings`
- 全量测试：`182 passed, 7 skipped, 23 warnings`
- machine JSON parse、`py_compile` 与 `git diff --check`：通过
