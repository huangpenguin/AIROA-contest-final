# Track 3 公开 Primitive / 自动生成 S0–S5 独立审核

> 审核日期：2026-09-03（Asia/Tokyo）
> 审核输入：root `73f675477a41a80241a7549f12f991187ecaaa25`，parent `b0078dc6ffc54036006c732c890f02824ca593a0`，nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`
> 下一唯一活动计划：`docs/AXIS_PI05_TRACK3_SUPPORT_AUTOGEN_SE3_RECOVERY_PLAN.md`

## 1. 结论

本轮正确遵守了单次 support canary 和失败即停的执行边界，失败轨迹、完整 simulator state、policy state、front/wrist H.264、action、接触数与 metadata 均被保留；没有 retry、人工 Teleop、训练、FAST/ZIP、官方 GPU 或提交。S0 的公开来源 lock 和 S2 的选择性 schema/replay 结果可以保留。

但不能认可“S0–S3 全部通过”，也不能进入 containment/articulation、批量生成或训练：

1. S1 的 `identity_counts` 与 `family_counts` 把 `195` 条 task rows 和 `21,961` 条 episode rows 混在一起统计。三种身份计数之和为 `22,156`，不是 episode 数 `21,961`；当前 `identity_classes_separate` 只验证三个 key 存在，没有验证逐行唯一、全集覆盖或和数相等。
2. S3 只验证 `reverse_initial_predicates != reverse_goal_predicates`，没有执行原计划要求的三族 fresh-process `3/3`、20-step 静置、reset goal=false，以及受控 goal true/false。`s3_reverse_semantic_adapter_passed` 因而是过强标签。
3. S3 三条语言包含内部 schema token 或错误指代：`its akita black bowl`、`floor target object`、`wooden cabinet 1 middle`。BDDL goal 可以是合理的原始区域，但这种 instruction 不应进入后续 VLA 训练或策略评测。
4. S4 执行器只控制 xyz，`action[3:6]` 恒为零；抓取目标固定为 object center `+0.025 m`，没有使用物体姿态、抓取 site、夹爪几何或公开成功轨迹中的 object→EEF 相对 6D 位姿。失败不是单纯步数不足。

审核状态固定为 `remediation_required`。当前 accepted/pending reverse demonstrations 仍为 `0/0`。

## 2. 已独立核对的证据

### 2.1 Git、hash 与边界

- root HEAD 与提交记录一致；root 和 nested OpenPI clean。
- `LIBERO-plus` 的既有 dirty porcelain 保持不变，未被本轮修改。
- S0/S1/S2/S3/S5 六个 summary 的当前 SHA-256 与 machine handoff 记录逐项一致。
- 大型 `trajectory.npz`、front/wrist H.264 被 `.gitignore` 排除，但文件在当前 workspace 存在且 metadata hash 可核对。
- S5 仅尝试 `support_remove_task_10`；`pending_autogen=0`，没有隐藏的 accepted promotion。
- 独立复跑定向门禁为 `12 passed`；完整 CPU 回归为 `165 passed, 7 skipped, 23 warnings`。

### 2.2 S0 与 S2

- 固定 source revision、license/schema 证据齐全；未新增大型下载。
- 原始 HDF5 task10/task20 的记录 state/action replay 达到最终 predicate success；报告明确采用 `states[0]` restore 后执行 `actions[1:]`。
- IPEC 样本为 20 FPS、7D action、8D state；Plus 许可未解决时保持只读/不可训练，没有越权升级身份。

S2 的 alignment 目前是成功性证据，不是逐 state exact 证明；下一阶段引用公开示范抓取位姿时必须补一阶 state alignment 误差报告，不能只凭最终 success。

### 2.3 support 失败轨迹

失败资产：`data/reports/axis_libero_30k/track3_public_autogen/diagnostics/support_remove_task_10/`。

```text
总帧数                         132
approach_object                 12，reached=true
approach 最终 EEF error          0.003496823 m
descend_object                 120，reached=false
descend 最小 EEF error           0.027499298 m
descend 最终 EEF error           0.032033743 m
接触数                           20 → 最高 36
action[3:6]                      全零
最终 reverse predicate           false
```

descend 在早期已达到最小误差，之后 100 多步没有继续接近，且接触数明显增加。front/wrist 抽帧也显示夹爪停在碗/盘附近。这支持“目标几何不可达或接触姿态错误”的诊断，反对简单延长 120-step timeout。现有轨迹没有逐帧 object pose、EEF quaternion、target pose 和 contact pair 名称，因此这仍是强证据推断，不是最终根因证明。

`deadline_miss=116/132` 是离线渲染/写视频的 wall-clock 指标；它不改变环境的 `control_freq=20`。后续必须把 simulator control frequency 与实时吞吐分开报告，不能把离线生成未达到 50 ms/step 误判为 20 Hz 数据格式失败。

## 3. 不可进入的阶段

- 不得复用旧 S3 `passed` 标签继续 containment/articulation。
- 不得只提高 descend timeout、盲调 z offset 或做参数网格。
- 不得把 19,704/2,452 当作 episode 分类数量写入训练采样方案。
- 不得把当前三条 malformed instruction 或 generator-reference-only 轨迹用于训练。
- 不得恢复人工 task19、批量 30–60 条、LoRA/dense、norm、FAST/ZIP、官方 GPU、上传或提交。

## 4. 审核决定

旧计划归档。下一计划只授权：不可变证据锁、S1 v2 统计、S3 v2 语言与动态门禁、失败轨迹的 contact/SE(3) 根因定位、从本地公开成功 HDF5 提取 object→EEF 抓取变换，以及一次 support-only 自动 canary。成功也只能进入 pending 并 hard stop；失败必须带增强诊断再次停下。
