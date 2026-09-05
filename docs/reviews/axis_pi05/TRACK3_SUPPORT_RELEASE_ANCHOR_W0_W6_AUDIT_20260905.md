# Track 3 Support Release-Anchor W0–W6 独立审核

> 审核日期：2026-09-05（Asia/Tokyo）
> 审核输入：root `2997c6f5d8c5b05345d74e0e0c4b96698def727e`
> W4 clean source：`1ec308a26e2fad306b221197ab9041d1b7633346`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`
> 用户人工操作：`not_required`

## 1. 获批准的执行事实

- root执行链、W4 source与最终提交均存在且有祖先关系；root/OpenPI在审核开始时clean。`LIBERO-plus`仍只有既有`1 modified + 4 untracked`快照，未被本轮提交触碰。
- W0的15项输入均重新计算为bytes/SHA-256 exact；旧V5 manifest仍为`19/19` exact。W1的task10/demo0 HDF5 transition由原始actions独立复算为close `32`、release `76`；exact-model state75的contact-v3为demonstrator/receiver/unrelated=`3/1/0`。
- 实际Robosuite `OperationalSpaceController`的input max为`1`，translation/rotation output max为`0.05 m / 0.5 rad`；新scale把`0.12 rad`物理rotation clip映射为normalized norm `0.24`。W4 trajectory的controller clip为`0`。
- W4只运行一次CPU/no-renderer task10/demo0、seed3121。worker PID与parent记录一致，return code `0`、无timeout、正常发布`failed_gate`；parent分类`family_failure`而非infrastructure是正确的。两个历史PID均已退出。
- W4共252个action/state frames：approach `41`、descend `12`、close `4`、lift `15`、transport `180`。transport耗尽冻结预算，末帧translation/rotation error为`0.0191271 m / 0.442809 rad`；source为`253/253 true`、goal为`253/253 false`。
- W5按W4失败门禁保持`not_evaluated`，没有启动GPU3。W6正确hard stop，`exact/pending/accepted=0/0/0`。
- 最终namespace有14个strict duplicate-key JSON、2个finite `float64[79]` NPY和1个no-pickle finite NPZ。17-entry manifest与最终应收集合完全相等，全部Git-tracked且bytes/SHA-256 exact；CSV为LF，无`.partial`。
- 审核时GPU0外部PID `1133706`仍在且未触碰，GPU1–3均为空闲。W4之后没有generation、replay、Teleop、训练/norm、FAST/ZIP、官方GPU、上传或提交；已检查执行时间窗内相关输出目录没有新增文件。
- 审核最终树独立复跑targeted为`21 passed, 2 warnings`，全量为`257 passed, 7 skipped, 23 warnings`；排除仓库已知的malformed-result fail-closed fixture后，tracked/new JSON严格duplicate-key parse通过，`py_compile`与`git diff --check`通过。

机器化摘要：`data/reports/axis_libero_30k/track3_support_release_anchor_w0_w6_audit_20260905/summary.json`。

## 2. Finding A：destination object pose 没有转换成 EEF target

W1正确冻结了release-side `T_object^-1·T_eef`，grasp端也使用了`T_world^object·T_object^eef`。但W4从transport开始把same-demo init bowl quaternion直接当作EEF quaternion，并把EEF position直接设为init bowl position加固定z offset；release-side相对pose没有继续参与destination target。

按W1冻结值独立复算，若desired object pose等于same-demo state0 bowl pose，则attached EEF应为：

```text
position [-0.0921273, 0.0434093, 0.892349]
quaternion [-0.684678, -0.656567, 0.256519, -0.185294]  (xyzw)
```

当前place target为`[-0.0768497,-0.00898957,0.953159]`且orientation为object quaternion；与正确组合的exact destination分别相差`0.0817122 m / 1.816425 rad`。transport的抬高waypoint也必须先在object frame定义再组合到EEF，不能丢掉anchor。因此W4没有评估完整的object-frame nominal executor。

## 3. Finding B：stage prerequisite 没有阻止未抓住物体后进入transport

W4的`contact_completion`实际为failed，source `(On bowl plate)`又在t0与全部252个post-step sample上保持true；这说明没有证据表明bowl已离开plate。实现仍只按EEF pose tolerance把close与lift标为reached，然后进入180步transport，直到预算耗尽才写`first_error=stage_timeout:transport`。

所以历史“transport超时”是有效停止位置，但不是唯一或最早的语义失败。后续执行器必须在close/lift边界要求gripper↔object contact、object-follow和source release成立；未满足时立即以grasp/acquisition family gate停止，不得把空手EEF移动解释为lift成功。contact还必须按initial/acquire/transit/destination窗口分角色，不能用t0的receiver contact与后续任意demonstrator contact拼成完成证据。

## 4. Finding C：stall、active cursor和final validation顺序仍不完整

- `stall_timeout`当前只检查是否已有`stage_timeout`和总steps是否小于上限，没有实现计划要求的冻结误差进度窗；W2的internal pass不能释放这一合同。
- W4 supervisor确实执行了identity、PID file、heartbeat与独立result流程，但全局`resume_cursor.json`在运行中从未原子置为`active=true`，只在W6创建inactive hard-stop记录；单实例恢复合同仍缺一环。
- W6先运行namespace validation/full tests，随后才写W6 closeout、CSV、inactive cursor、supervisor closeout和manifest。最终文件已由本审核独立验证通过，但旧W6测试记录本身不是finalized tree的完整证明。
- trajectory的action/state为252帧，而`goal_trace`为253项（t0 + 252 post-step）；数据可解释，但schema必须显式记录alignment，不能默认为逐action等长。

## 5. 阶段判定

```text
W0 immutable lock                         approved
W1 release-side anchor/controller scale   approved
W2 OSC action scaling                      approved
W2 destination/stall contract              remediation required
W3 source identity/PID/heartbeat core      approved
W3 active cursor lifecycle                 remediation required
W4 observed run/category/hard stop         approved
W4 intended object-frame nominal profile   not_evaluated
W5 GPU3 camera canary                      not_evaluated
W6 final artifacts after independent audit approved
exact/pending/accepted                      0/0/0
```

## 6. 审核决定与下一步

本轮不是blocked：已有证据足以定位可自动修复的executor与supervisor问题。旧W4 `family_failure`、全部raw gate和W6 hard stop保持不改；纠正解释只进入本审核namespace。

新唯一活动计划为：

`docs/AXIS_PI05_TRACK3_OBJECT_FRAME_EXECUTOR_AND_BOUNDED_AUTOGEN_LORA_PLAN.md`

它先闭合object→EEF target、acquisition barrier、phase-scoped contact、progress-window stall、active cursor和post-finalization验证；随后在一个计划内条件式继续support双profile、其余两个family、最多12条generation/36次replay、20Hz `80/20` mixture、R6 L2 LoRA 100/500/1000、holdout/retention与本地候选ZIP。每个family物理失败只关闭该lane，共享基础设施/污染/数值失败全局停止；任何分支最终X13 hard stop，不自动上传或提交。
