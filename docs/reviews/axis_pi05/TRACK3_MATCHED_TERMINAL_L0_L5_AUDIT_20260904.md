# Track 3 Matched Terminal-State L0–L5 独立审核

> 审核日期：2026-09-04（Asia/Tokyo）
> 审核输入：closeout `0dc402e2d70e6576c5861a674e0aac04e4427ef0`
> execution：`19cb1eab9556b6e9e0f8b3f9d4b49fb3893b81a0`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`

## 1. 获批准的执行事实

- L0/L1 的输入锁、固定 `task10/task20/task19 × demo_0/demo_1/demo_2`、同 demo `model_file + terminal full-state` 和工具修复均已落盘。
- L2 确实运行了 9 个 fresh CPU/no-renderer process；9 份 raw report 与 `l2_summary.json` 的 SHA-256 全部一致，worker return-code contract 为 `9/9`。
- task10 为 `3/3`；task20 与 task19 均为 `0/3`，这是当前实现下的真实 gate 结果。L2 失败后 L3/L4 保持 `not_evaluated`，没有启动 GPU camera canary。
- 9/9 均通过 same-demo model/state identity、public forward goal、reverse critical source `step5–20=true` 与 reverse goal `step5–20=false`。因此本轮没有出现 reverse 语义消失或 goal 提前成立。
- `contact_ledger.json` 为 189 条，即 9 run × 21 checkpoints；无 `.partial` 残留。reverse demo `exact/pending/accepted=0/0/0`，没有 action/demo、SE(3)、Teleop、训练、norm、FAST/ZIP、官方 GPU、上传或比赛提交。
- root 与 nested OpenPI 在审核输入处 clean；`LIBERO-plus` 的既有 modified/untracked 状态保持原样。审核时 GPU0 仍为外部 PID `1133706`，GPU1–3 约 `5 MiB`，本轮审核未触碰任何 GPU。

## 2. `0/3` 的正确解释

三族的 source/goal 语义均已通过。失败集中在“成功瞬间的 forward terminal robot/scene 仍在释放或沉降”，不是 reverse task 定义本身：

- task20/demo0 仅因 orange juice 在 step5→20 平移 `7.390 mm`；demo1 为 `2.088 mm`，demo2 为 `3.165 mm`。basket 自身漂移仅约 `1e-6 m`。
- task20/demo1 的额外接触是 gripper fingertip↔orange juice，release window 仅 step6 一次，penetration 约 `11.2 µm`；demo1/2 的 robot qpos/qvel gate 也略超线。
- task19/demo0、demo1 只因 forward terminal robot 尚在回稳：arm qpos 最大漂移分别 `0.03380/0.03055 rad`，EEF 平移 `4.763/5.942 mm`；drawer 的 source `Open` 一直成立且 reverse goal `Close` 一直为 false。
- task19/demo2 的 drawer 本体平移与 joint drift 约为 `1e-5` 量级；额外失败来自 right finger↔drawer 的极浅接触，step11–20 最大 penetration 约 `0.263 µm`，以及 robot 稳定性。

9 个 HDF5 demo 的 success reward 都只有最后一帧为正，`positive_count=suffix_length=1`。当前 selection 因而只能取“刚成功的末帧”，没有任何 post-success dwell。继续换 demo、放宽 `2 mm` 或把这些结果解释为缺少 reverse semantics 都没有证据支持。

## 3. Validator 的确定问题

`contact_contract_v2` 的代码把任何包含 operator subject/target、又不属于显式 allowed pair 的接触都加入 `forbidden_operator_background_contacts`。这会把 robot/gripper↔operator 接触也当作“无关 background”。原计划只禁止 operator↔无关 background；机器人不是 background。

因此 task20/demo1 与 task19/demo2 的 `contact_v2` failure 标签不符合预注册语义。旧 raw 与旧失败状态必须保留，但下一版必须把接触分为：

1. operator↔receiver/support；
2. operator↔robot/gripper；
3. operator↔unrelated scene background；
4. non-operator background↔background。

第 3 类始终 hard fail；第 2 类不能伪装成 background，在新构造的 reverse start 中应单独记录并要求测量窗口为零。

另一个较小的报告缺陷是 aggregate `hard_gate.contact_v2` 失败时没有携带 `reason`，虽然逐 checkpoint 的 `contacts_v2.reason` 仍在 raw 中。L2 summary的状态名也只编码首个失败 family `containment_remove`，而task19实际同时为`0/3`。两者不破坏本轮hard stop，但M0必须输出完整逐run/逐gate分解，后续aggregate failed gate必须保留结构化reason。

## 4. 可复现性复核

独立复跑结果：

```text
PYTHONPATH=. .venv_axis_pi05/bin/pytest -q \
  tests/test_track3_matched_terminal_state_release.py \
  tests/test_track3_matched_terminal_state_release_plan.py
10 passed

PYTHONPATH=. JAX_PLATFORMS=cpu .venv_axis_pi05/bin/pytest -q tests
211 passed, 7 skipped, 23 warnings
```

这与执行报告的 pass/skip 数一致；warning 数由 `PYTHONPATH` 是否把既有 dirty `LIBERO-plus` 放在前面而不同。审核使用 clean root import 优先级仍得到相同测试通过数。

## 5. 审核决定

L0–L5 的边界执行、证据闭包和 hard stop 获批准；scientific release 不批准。下一步不重跑旧 L2，也不进入 camera/SE(3)/demo/训练。

新的最小恢复是 same-demo、same-XML 的 composed reverse start：保留成功 terminal state 的 scene/object/fixture qpos，把 robot/gripper 恢复为同 demo `init_state` 的 qpos，清零动力学并使用中性 7D hold。先验证一个 task20 canary，再完成固定 9 条 CPU matrix；通过后最多做三族各一条 GPU3 camera canary。该路径只产出可审计的 start-state candidate，不产出训练 demo。
