# AXIS Pi0.5 Track 3：动态门禁 Model/State 对齐修复计划

> 制定日期：2026-09-03（Asia/Tokyo）
> 输入执行 commit：root `179045a6d7a7998bf1a54452ef79739c9383c09f`，nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`
> 当前状态：`track3_native_reverse_dynamic_canary_failed_support`
> 唯一授权：执行 D0–D3，随后仅做 D4 收尾并 hard stop
> 不授权：原 G2/G3/G4、任何真实 reverse canary、人工 Teleop、批量生成、训练、norm、FAST/ZIP、官方 GPU、上传或比赛提交

## 0. 目标与决策

本计划只回答一个问题：三个公开 reverse spec 在各自一致的 MuJoCo model 内，初始/目标 predicates 与静置动力学是否成立。

旧 G1.3 不能回答该问题，因为它跨随机 XML 恢复 HDF5 state，并用异量纲整段 proprio 的单一最大差作门禁。不要重做已通过的 S1 v2 或 typed language；不要继续 SE(3) 抓取与 canary。先得到一个可审计、同 model、逐 predicate、逐物理量的动态结论。

## 1. D0：不可变输入锁（CPU only）

输出：

```text
data/reports/axis_libero_30k/track3_dynamic_gate_reconciliation/d0_release_lock.json
```

必须锁定：

1. root `179045a...`、nested OpenPI `b58aff0...`，以及 root/OpenPI clean；只记录 `LIBERO-plus` 既有 dirty porcelain，禁止修改、reset 或 stash；
2. G0、S1 v2、S3 language、旧 G1 dynamic report、三个 v2 BDDL/spec、P1 snapshot 与三份官方 HDF5 的 hash；
3. 每份 HDF5 `model_file` attribute 的 hash、state shape、`states[0] == init_state` 与 time/qpos offset；
4. `pending_autogen_v2=0`、accepted/exact reverse=`0`，没有 G2/G3/G4、训练或新 canary；
5. 旧 `g1_s3_dynamic_gates.json` immutable 回归测试。

任一已锁 evidence 漂移或 root/OpenPI dirty，状态置为 `track3_dynamic_gate_reconciliation_input_drift` 并停止。

## 2. D1：validator 语义与字段修复（CPU/code only）

新实现必须输出到 `track3_dynamic_gate_reconciliation/`，不得覆盖旧 G1 report。

### D1.1 禁止跨 model predicate oracle

- 报告 HDF5 recorded `model_file` 与当前 env XML 的 hash、`nq/nv`、joint/body/site name+order 和关键 body/site pose 差异。
- model/XML 不同即将直接 `set_init_state(hdf5_state)` 标为 `incompatible_for_predicate_oracle`；不得把 predicate true/false 纳入门禁。
- HDF5 state 仍可作为后续同 exact recorded model replay 的参考，但本计划不授权该 replay，更不据此构造 reverse 起点。

### D1.2 权威 reverse 初态

- 每个 v2 reverse BDDL 在自身 fresh env 中执行 native reset，并在同一 env/model 内逐条计算 reverse initial predicates 与 reverse goal predicates；禁止用 `forward.check_success()` 代替 reverse initial。
- 记录每个 predicate 的 boolean、对象/区域/site 名称、支持/包含的几何 margin 或 contact 信息、articulation joint value 与 predicate threshold（可获得时）。
- native reset 必须满足全部 reverse initial、reverse goal=false；否则以具体 predicate 和模型证据 fail closed。
- P1 terminal scene 仅作为 provenance；除非模型兼容性 exact，不能成为动态门禁或生成起点。

### D1.3 同量纲 robot/scene settle

- controller 顺序固定为：restore/native reset → `sim.forward()` → controller `update/reset_goal()` → warm-up；不得 reset controller 后再次覆盖 state。
- 明确定义 gripper hold：保持 native reset 的 gripper target；若 controller API 不能精确 hold，则 arm stability 与 gripper stability 分开，且不得用固定 `gripper=-1` 主动改变起点。
- 至少先 warm-up 20 control steps，再比较后续 20-step observation window；同时报告 warm-up drift，不用它直接否决 settled state。
- 按命名字段分别报告 joint qpos、joint qvel、EEF translation、EEF geodesic rotation、gripper qpos/qvel；各字段使用单位明确的预注册阈值。禁止对拼接 proprio 使用单一 max。
- object translation/rotation、fixture joint、contact/predicate 连续性分别报告；三族不得共用含义不明的 scalar。
- 所有 action/state finite，action contract 为 7D OSC_POSE，control frequency 固定 20 Hz。

### D1.4 CPU 测试门禁

至少覆盖：

- 不同 model hash 时拒绝 HDF5 predicate oracle；
- reverse initial 必须逐 predicate 求值，不能由 forward success 代替；
- controller reset/set-state 调用顺序；
- gripper hold 不等价于固定 `-1`；
- named observation schema、四元数 geodesic、各单位阈值和 missing/shape fail closed；
- warm-up window 与 measurement window 不混淆；
- 旧 G1 evidence immutability。

CPU/schema 任一失败即置为 `track3_dynamic_gate_validator_remediation_failed` 并停止，不得进入 D2。

## 3. D2：最小 GPU3 canary（只验证一个 native reset）

运行前检查四卡 PID；只允许自然空闲的物理 GPU3，禁止 GPU0。固定 task10 support、一个 fresh process、一个 native reverse reset：

1. 输出 model/state compatibility 与逐 predicate 证据；
2. 验证 reverse initial 全 true、goal false；
3. 执行预注册 warm-up + measurement settle；
4. 验证 scene/robot 各命名物理量稳定、predicate 在 measurement window 连续成立；
5. 在同一 model 内构造最小 controlled state，使一个 reverse initial predicate 从 true→false；恢复原 native state 后还须 true；
6. 不移动机械臂去抓取，不生成 demo，不写 pending。

只有 D2 全部通过才能进入 D3。实际 D2 的唯一 fresh process 在 site-state 详情记录处因 validator 把原始 `TargetZone` 当作带 `get_geom_state` 的 state wrapper 而异常；因此没有形成可用 predicate/settle 结论。该次状态记为 `track3_native_reverse_dynamic_canary_failed_support`，修复后不重跑，不调整阈值。

## 4. D3：三族同 model 动态 release（GPU3 only）

对 task10/task20/task19 各运行 `3/3` fresh process，共最多 9 次；输入是各 reverse BDDL 自身 native reset，不是 P1 terminal 或跨 model HDF5 state。由于 D2 未通过，本轮 D3 未运行（`0/9`）。

每次必须满足：

1. reverse initial predicates 全 true，reverse goal=false；
2. model/state compatibility 完整记录，且门禁没有使用 incompatible HDF5 predicate result；
3. warm-up 后 measurement window 中 object/fixture/robot 各命名物理量通过预注册阈值；
4. initial predicates 在 measurement window 连续为 true，goal 连续 false；
5. 同 model controlled destruction 使指定 predicate true→false，恢复后再次为 true；
6. BDDL/XML/controller/init/state/action hash 与 fresh-process stdout/stderr 完整；任何 missing/NaN/shape mismatch fail closed。

全部 `9/9` 通过时状态为：

```text
track3_reverse_semantics_v2_dynamic_validated_pending_se3_review
```

任一失败时状态为：

```text
track3_reverse_semantics_v2_dynamic_failed_<family>_<predicate_or_field>
```

两种结果都立即进入 D4 收尾并 hard stop。即使通过，也不得自动恢复归档计划的 G2–G4；下一轮独立审核才决定是否以 native reverse reset 为起点开展 HDF5 exact-model SE(3) 挖掘与一次 support canary。

## 5. D4：收尾与交接（只记录）

- 写 `summary.json`、中文执行报告、per-run JSON，更新 plan index/handoff/machine JSON；
- 明确区分 `historical_g1_failed`、`validator_reconciled` 与 `reverse_semantics_validated/failed`；
- 小型 code/config/report/test 纳入 Git，视频/NPZ/state 保持 ignored；
- 运行 JSON/YAML parse、`py_compile`、Markdown path 检查、定向测试和：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. .venv_axis_pi05/bin/pytest -q tests
git diff --check
```

- 形成 clean commit；不得 push、上传或比赛提交。

本轮实际收尾：D0 锁定成功；D1 CPU/schema 门禁在修复后为 `15/15`；D2 首次 fresh process 触发 validator exception，D2 不重跑；D3 hard stop，未产生任何三族 release 结果。D4 报告位于 `data/reports/axis_libero_30k/track3_dynamic_gate_reconciliation/`。因此本计划到此结束，等待下一次审核，不把该异常解释为 reverse semantics 失败。

## 6. 本计划之后（不授权）

- D3 通过：新审核可把后续恢复为“exact recorded-model HDF5 抓取模板提取 → native reverse reset 起点 → 一次 support-only SE(3) canary”。
- D3 只因 native BDDL placement 失败：先修 BDDL init predicate/region，不做 Teleop。
- D3 predicate 正确但 controller settle 仍失败：只调查 hold controller，不进入抓取。
- 自动几何路线再次经有效 validator 失败后，才考虑一条 human Teleop reference；当前无需人工录制。
- 有 verified reverse demos 后才从 R6 step500 分叉小步 LoRA；不从 Axis init 重来，也不以 hidden Track3 分数作逐 checkpoint 调参集。
