# AXIS Pi0.5 Track 3：Native Reverse Validator 最小恢复计划

> 制定日期：2026-09-03（Asia/Tokyo）
> 输入执行 commit：root `84c2655f610bb957b0ad1ec2a952776bf023821e`，nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`
> 当前状态：`track3_native_reverse_validator_recovery_planned`
> 唯一授权：执行 R0–R3，随后只做 R4 收尾并 hard stop
> 不授权：SE(3)、任何抓取/demo、Teleop、批量生成、训练、norm、FAST/ZIP、官方 GPU、上传或比赛提交

## 0. 目标

本计划不修改 reverse BDDL、predicate、controller、threshold 或数据。它只修复 D2 中已经定位的 validator runtime schema/进程状态问题，取得第一条有效 task10 native-reset 动态结论；只有该结论通过，才执行原计划的三族 `3/3`。

历史 D2 的 `worker_exception` 不是 task failure，必须保留；新结果写入独立 namespace：

```text
data/reports/axis_libero_30k/track3_native_reverse_validator_recovery/
```

## 1. R0：不可变恢复锁（CPU only）

输出 `r0_release_lock.json`，必须锁定：

1. root `84c2655...`、parent `f0034b9...`、nested OpenPI `b58aff0...` 与 root/OpenPI clean；只读记录 `LIBERO-plus` 既有 dirty porcelain；
2. 原 D0 lock、D2 canary、per-run JSON、D4 summary/report、旧 G1 report、三个 v2 specs 和当前 validator/test 的 SHA-256；
3. 原 D2 精确事实：`worker_exception`、`TargetZone.get_geom_state`、returncode=`0`、有效 predicate/settle evidence=`0`、D3 runs=`0`；
4. exact/pending/accepted reverse demo=`0`，没有 SE(3)/训练/Teleop/ZIP/上传；
5. 原 D2/D4 小型 evidence immutability 回归测试。

任一 hash/count 漂移或 root/OpenPI dirty，状态置为 `track3_native_reverse_validator_recovery_input_drift` 并停止。不得 reset/stash/修改 `LIBERO-plus`。

## 2. R1：runtime schema 与进程门禁闭合（CPU/no-renderer only）

### R1.1 truth 与 telemetry 分层

- 每条 predicate 先独立调用 `_eval_predicate` 并保存 authoritative boolean；随后单独提取 geometry/joint/contact detail。
- detail extraction 异常必须记录 `detail_status=failed`、exception class/message 和具体 atom；不得删除已得到的 truth，也不得静默忽略。
- release 仍 fail closed：任一 required detail 不完整时 `schema_complete=false`，禁止进入 GPU recovery canary；但不得把它伪装成 predicate=false。
- `TargetZone/SiteObject` 必须经 `object_states_dict` 的 `SiteObjectState` wrapper；body/object 走明确的 body/object state 路径。

### R1.2 worker/parent 状态一致

- `worker_exception`、invalid/no output、missing field 必须使 worker 或 parent report 为失败；worker exception 的 OS return code 必须非零。
- parent 同时检查 JSON status 与 return code：`returncode != 0` 不得接受为 passed；`status != passed` 也不得接受。
- stdout 最后一行 JSON 协议、stderr tail、exception provenance 与 atomic report 必须保留。

### R1.3 真实 runtime schema probe

以 `has_renderer=false`、`has_offscreen_renderer=false`、`use_camera_obs=false` 创建 task10/task20/task19 各一个 fresh native reverse env；这是 CPU simulator schema probe，不执行 40-step settle、不使用 GPU。

每族必须验证：

1. env/XML/controller/model layout 可建立；
2. 全部 reverse initial/goal atoms 可求 truth；
3. 所有 atom 的 body/site/articulation details 完整、finite、shape 正确；
4. named robot/scene snapshot、controlled-destruction target resolution 与 gripper hold contract 可构造；
5. HDF5 state 仍未作为 predicate oracle；
6. fresh-process return code/status 一致。

三族必须 `3/3 schema passed`。任一异常/缺字段即置为 `track3_native_reverse_runtime_schema_failed_<family>` 并停止，不得进入 R2。

R1 不得修改 BDDL、threshold、warm-up/measurement 步数、controller action 或 predicate definition。

## 3. R2：唯一一次 task10 recovery canary（GPU3 boundary）

运行前检查所有 GPU PID；只允许自然空闲的物理 GPU3，GPU0 禁止。锁定 R1 code/spec/threshold hashes，固定 task10、run index=`recovery_1`、native reverse reset、20+20 settle 和原 controlled destruction。

输出：

```text
r2_task10_recovery_canary.json
runs/r2_task10_recovery_run_1.json
```

通过必须同时满足：

- worker returncode=`0`、status=`passed`、schema complete；
- native reverse initial predicates 全 true、reverse goal 全 false；
- model compatibility 只作 provenance，HDF5 state 未参与 predicate；
- named robot/scene settle 全通过，predicate measurement continuity 完整；
- controlled predicate true→false→restored true；
- action/state/detail finite，hash/stdout/stderr 完整；
- 无抓取、无 demo/pending、无参数修改。

若出现 exception、schema failure 或任一真实 gate failure，立即 R4 收尾，状态必须区分：

```text
track3_native_reverse_validator_recovery_exception
track3_native_reverse_dynamic_failed_support_<predicate_or_field>
```

不得第二次 recovery retry，不得临时改 threshold/BDDL/controller，也不得进入 R3。

## 4. R3：三族 `3/3` 同 model release（R2 通过后，GPU3 only）

R2 通过后立即锁定 source/spec/threshold hash；R3 前后不得修改代码或配置。按 task10→task20→task19，各 3 个 fresh process，共 9 次。执行语义与 R2 完全一致；原 D2 不计入 9 次。

每次必须满足 R2 的全部 gate。每族要求 `3/3`，missing/exception/nonzero returncode/NaN 均 fail closed。为了保留随机 placement 分布证据，可完成预注册的 9 次；不得对失败族追加运行。

全部通过状态：

```text
track3_reverse_semantics_v2_dynamic_validated_pending_se3_review
```

否则状态：

```text
track3_reverse_semantics_v2_dynamic_failed_<family>_<predicate_or_field>
```

两种结果都进入 R4 并停止。通过不等于 demo 已生成，也不授权恢复旧 SE(3) 计划。

## 5. R4：收尾与交接

- 输出 fail-closed `summary.json`、中文报告与每次 run JSON；明确 historical D2、R2、R3 的计数与不同含义；
- 更新 plan index、handoff Markdown 与 machine JSON；旧 evidence 不覆盖；
- 小型 code/report/test 纳入 Git，视频/NPZ/model 不入 Git；
- 运行 JSON/YAML parse、`py_compile`、Markdown path 检查、定向测试与：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. .venv_axis_pi05/bin/pytest -q tests
git diff --check
```

- root/OpenPI clean commit 后 hard stop；不得 push、上传或提交比赛。

## 6. 本计划之后（不授权）

- `9/9` 通过：下次审核才可恢复 exact recorded-model HDF5 SE(3) template mining，并以 native reverse reset 为生成起点，只做一次 support canary。
- task10 真实 predicate/settle 失败：针对具体 predicate/field 做最小诊断，不做 Teleop 或训练。
- 三族中某族失败：只修该 family 的 BDDL init/predicate/controller 问题；其他族的 pass evidence 保留。
- 只有 verified reverse demos 建立后，才从 R6 step500 分叉小步 LoRA；不从 Axis init 重新训练。
