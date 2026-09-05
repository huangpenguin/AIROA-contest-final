# Track 3 J5 Executor Recovery V0–V5 独立审核

> 审核日期：2026-09-05（Asia/Tokyo）
> 审核输入：root `11483623b7710430a197231b8ed0662c4813c12d`
> V4 clean release commit：`1b704405fa918b5237a50a1219d4a26b61c293de`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`
> 用户人工操作：`not_required`

## 1. 获批准的执行事实

- V0 的 18 个 immutable input/artifact 项独立复算为 `18/18` bytes/SHA-256 exact；旧 J3/J11 manifest failure 保持不改。
- V2 从旧 180 帧 NPZ 离线编码的 agentview/wrist 均为 H.264、20 FPS、128×128、180 帧，无 `.partial` 残留；它们没有被误写为 pilot 成功。
- V3 在 `0b17f66` source 上形成 `1b70440` clean release commit；V4 于该提交之后只启动 task10/demo0 nominal、seed 3111 一次。没有 task10 P3 attempt4 或第二次 canary。
- V4 worker PID `1450749` 与 child result 一致，正常发布 `failed_gate` 后 return code 为 0；parent 正确分类为 `family_failure`，不是 infrastructure。GPU3 UUID、起止空闲快照、159 次采样、459 MiB 峰值、无 timeout 与 telemetry-v2 均通过。
- V4 两路视频各为 H.264、20 FPS、128×128、150 帧，原子发布成功；trajectory 为 `150×7` finite actions、`150×79` finite states，`allow_pickle=False` 可读。
- V5 保留 V4 family failure 并 hard stop。最终 manifest 的 19 项全部 Git tracked 且 `19/19` bytes/SHA-256 exact；cursor inactive，GPU1–3 已释放，GPU0 外部 PID `1133706` 未触碰。
- V4 后没有 conservative、其他 family、generation/replay、Teleop、训练/norm、FAST/ZIP、官方 GPU、上传或提交；`exact/pending/accepted=0/0/0`。
- 用户报告的全量测试已独立复跑：`256 passed, 7 skipped, 23 warnings`。root/OpenPI clean；`LIBERO-plus` 仅保留既有 dirty snapshot。

机器化摘要：`data/reports/axis_libero_30k/track3_j5_executor_recovery_audit_20260905/summary.json`。

## 2. Finding A：OSC rotation scale 仍不符合已声明的物理 profile

`xyzw` compose/log 已修正，V1 的同一时刻 pose 重构误差 `2.2697e-7 rad` 也有效。但 action builder 把物理 rotation error 按 `rotation_clip_rad=0.12` 除到 normalized `[-1,1]`；本地 V4 实际使用的 Robosuite OSC_POSE config 则把 normalized rotation `±1` 映射为 `±0.5 rad`。

所以当前实现对小 rotation error 使用约 `0.5/0.12=4.1667` 倍增益；对已按 0.12 rad 截断的命令仍实际请求 0.5 rad。V1 的 `normalized_7d_builder` 只检查长度/范围，没有验证 controller output scale。V4 rotation error 最终停在 `0.046459 rad`，高于 `0.03 rad` tolerance；150/150 帧均被记录为 saturated。该运行证明当前实现失败，但不能代表计划声称的 `0.12 rad` nominal profile 已被正确执行。

## 3. Finding B：HDF5 object-follow/contact window 没有闭合，抓取 anchor 选错侧

V1 只读取了 forward demo 的 `-1→+1` transition `index=32`，并把该时刻的 object-relative EEF pose 作为 reverse grasp anchor；报告把 object-follow/contact 写成“V4 runtime required”，没有完成计划要求的 HDF5 窗口证明。

独立 exact-model CPU restore 得到同一 demo 的 `+1→-1` release transition 为 `index=76`。紧邻 release 的最后 closed frame `state/action index=75` 有 3 个 demonstrator contacts、1 个 bowl↔plate receiver contact、0 个 unrelated contact，适合作为 reverse acquisition 的 release-side 证据。index 32 到 index 75 的 object-relative EEF 变化为 `0.0387545 m / 0.600317 rad`；将两种 relative pose 重构到同一 terminal bowl pose，target 也相差同样量级。因此 index 32 单点重构正确，并不证明它是 reverse-start 的正确 acquisition anchor。

## 4. Finding C：V4 gate 三态和 supervisor provenance 仍未达到计划合同

V4 在 `approach_grasp` 的 150 帧预算耗尽，translation error 从 `0.271215 m` 最低到 `0.071814 m` 后回升并停在 `0.104612 m`；没有进入 descend、close、lift、transport、place、release 或 retreat。`stall` failure 与 final goal=false 是有效观测，initial source 始终 true。

但当前报告仍把后续阶段才可判定的 `demonstrator contact` 和 `same-demo init pose placement` 写成 executed/failed，并把只有已执行 prefix 的 contact telemetry 标为 `full_contact_window=true`。依照活动计划“未执行 gate=`not_evaluated`”的规则，这两项不能作为独立物理失败；正确状态应为 `not_evaluated`。camera、finite、collision、unrelated contact/movement 的 prefix 证据可保留为 passed。

此外，`launch_identity_gate()` 与 `heartbeat_due()` 只有函数和单元测试，没有被 V4 launcher 调用。V4 parent 也没有记录 root commit/tool SHA；`supervisor_closeout.json` 实际由 V5 命令写出，不是一个执行过 dirty/commit/tool/PID/cursor preflight 的 persistent supervisor。运行仅约 36 秒，所以没有发生 15 分钟 heartbeat gap，但启动 identity 的缺失使 V4 无法满足计划的完整 source-provenance release 条件。

## 5. 阶段判定

```text
V0 immutable input lock                    approved
V1 explicit xyzw                           approved
V1 OSC physical scaling                    remediation required
V1 HDF5 object-follow/contact anchor        not approved
V2 offline atomic video/errata              approved
V3 clean commit                             approved with provenance gap
V4 observed run/category/hard stop          approved
V4 intended corrected nominal profile       not_evaluated
V4 contact/place dependent gates            not_evaluated
V5 closeout/final manifest                  approved
exact/pending/accepted                       0/0/0
```

## 6. 审核决定与下一步

旧计划已完成其授权分支并归档。唯一新计划为：

`docs/AXIS_PI05_TRACK3_SUPPORT_RELEASE_ANCHOR_AND_OSC_SCALE_RECOVERY_PLAN.md`

它先在 CPU 上锁定 release-side HDF5 contact/pose anchor，按真实 OSC output scale 修复 physical-delta→normalized-action 映射和 stage-dependent tri-state，并使 source identity/supervisor 真正进入 launcher。形成 clean commit 后先运行一次 CPU/no-renderer 完整 task10 canary；只有该 canary 全部通过，才允许 GPU3 一次同配置 camera canary。任何失败都 hard stop；仍不授权 conservative、其他 family、demo、generation、训练或 ZIP。
