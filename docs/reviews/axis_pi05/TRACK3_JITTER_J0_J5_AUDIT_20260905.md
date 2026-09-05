# Track 3 Jitter Recovery J0–J5 独立审核

> 审核日期：2026-09-05（Asia/Tokyo）
> 审核输入：root `7ae488b4475169ac1cc3b586158454639e6a2be9`
> J5 clean source：`3a5641956770fa74faaed20f12832489d7f257b7`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`
> 用户人工操作：`not_required`

## 1. 获批准的事实

- `7ae488b`是已审核输入`8c0d296`的9个提交直接后继；审核开始时root和OpenPI clean，`LIBERO-plus`原有dirty文件未改。
- J0锁定的9个文件、R6 folded 14,467,165,872-byte模型、raw checkpoint tree和三份公开HDF5均独立复算hash exact。
- J1的telemetry v2数学门禁正确区分query latency与heartbeat gap；50/148/600 ms query latency均不单独失败，1999/2001 ms heartbeat边界符合预注册阈值。
- J2对task10 attempt3的只读升格可由旧parent/child、113次采样、PID/UUID、正常return code、无timeout和两路H.264重新构造；没有task10 P3 attempt4。
- J3真实运行task20、task19各一次，两个parent/child均passed。四类child hard gate、PID、GPU3 UUID、start/end free snapshot、return code 0、无timeout、telemetry v2和两路41帧20 FPS H.264均通过；核心14个J3产物hash exact。
- J5只启动task10/demo0 nominal一次。child确实写出180帧双相机、180×7 finite actions、180×79 finite states和完整telemetry；reverse goal始终false，阶段停在`approach_grasp`。GPU3峰值459 MiB，230次采样，heartbeat max 252.45 ms，无timeout，结束后释放。
- J5 infrastructure hard stop之后没有运行J6–J10；这些阶段为`not_evaluated`。`exact/pending/accepted=0/0/0`，没有训练、norm、Teleop、FAST、ZIP、官方GPU、上传或提交。
- 基线全量测试独立复跑为`250 passed, 7 skipped, 23 warnings`。当前GPU1–3无compute process；GPU0外部PID `1133706`保持。

机器化审核摘要：`data/reports/axis_libero_30k/track3_j5_audit_20260905/summary.json`。

## 2. Finding A：J4/J5 quaternion约定错误，旧nominal不是有效物理profile结论

Robosuite的`axisangle2quat`、`mat2quat`和`quat2mat`都使用`xyzw`。当前工具却把这些返回值命名为`*_wxyz`，并由`_quat_mul_values`和`quaternion_log_error`按`wxyz`运算。

对旧task10 reference独立复算：使用正确`xyzw`乘法，object-relative grasp重构到HDF5 EEF的geodesic误差仅`2.2760e-7 rad`；当前helper产物若按Robosuite约定解释则偏差`2.21036 rad`。实际J5轨迹180帧全部`action_saturated=true`，最终rotation error仍为`3.12605 rad`。

因此历史child的`failed_gate:stage_timeout:approach_grasp`必须保留，但它只描述带错误orientation contract的运行，不能被晋升为正确HDF5-grounded executor的nominal物理失败。对预期executor profile的结论是`not_evaluated`。

J4报告本身也没有完成计划要求的证明：task10/task20没有从HDF5证明object-follow并锁定same-demo init object pose；task19只证明总state/EEF非零运动，没有提取handle-relative pose、drawer joint方向和关闭目标；contact-v3角色矩阵、collision/stall和unrelated movement hard gate只写成字符串，J5 success path并未实际执行这些门禁。因此J4的历史`passed`保持不改，但本审核不批准其executor release语义。

## 3. Finding B：三处状态/输出基础设施缺陷

1. J5视频原子临时名为`agentview.mp4.partial`。ffmpeg无法据此选择muxer；独立CPU复现return code为`234`，而`agentview.partial.mp4`正常输出。
2. J5 worker对完整发布的物理`failed_gate`返回1；telemetry v2又将任何非零return code判为infrastructure failure。因此即使修复视频，当前parent仍会把正常family failure误报为基础设施错误。正常结果发布与worker exception必须使用不同的进程语义。
3. Supervisor只依据空`generation_profiles`调用`run_j11("J5", "no_family_release")`，覆盖了J5已经明确给出的`infrastructure_failure`。所以历史J5 summary正确，J11/cursor的`no_family_release`错误；只能在新审核/derived namespace纠正，不能回写旧J11。

## 4. Finding C：launcher与manifest closure没有达到J1/J11合同

J1报告声称launcher会校验dirty source、启动commit/tool hash并每15分钟写heartbeat，但实现只写启动/结束cursor并检查PID file，没有上述校验或周期heartbeat。此次J5约52秒即停止，故不影响其单次PID/telemetry证据；它不能支持未来8–12小时无人值守承诺。

旧J3 manifest共21项，当前有5项bytes/hash不一致；旧J11 manifest共37项，当前有4项不一致。漂移来自manifest在最终cursor/closeout/log写完和PID删除之前生成；其中J3 core entries为`14/14`、J11 core execution entries为`25/25` hash exact。旧manifest必须保留为失败证据，新closure必须排除临时PID和仍在写的log，并在所有immutable文件关闭后生成。

另外，J0记录root有未跟踪工具，J1/J2在J3后被重新生成，J3 manifest保存的是其旧hash；J0–J4小型证据直到`3a56419`才统一提交，不满足原计划“GPU前clean intermediate commit、每次运行立即commit evidence”的严格顺序。J3 raw可以独立验证，但旧manifest不能宣称checkout closure通过。

## 5. 阶段判定

```text
J0 immutable inputs/hashes                 approved
J1 telemetry v2 sampling math              approved
J1 persistent launcher contract            remediation required
J2 attempt3 derived P3 release              approved
J3 task20/task19 P3                         approved 2/2
J4 intended HDF5-grounded executor release  not approved
J5 historical run                           infrastructure hard stop approved
J5 corrected nominal profile                not evaluated
J6–J10                                      not_evaluated
exact/pending/accepted                      0/0/0
```

## 6. 审核决定与下一步

旧计划已按J5 hard stop完成其授权分支并归档。唯一新计划是：

`docs/AXIS_PI05_TRACK3_J5_EXECUTOR_CONTRACT_AND_ATOMIC_VIDEO_RECOVERY_PLAN.md`

它先在CPU修复explicit `xyzw`、support HDF5证据、原子MP4、child/parent分类、supervisor heartbeat/identity和最终manifest顺序；再离线重编码旧180帧并形成derived errata。全部通过且clean commit后，只允许GPU3运行一次修正后的task10/demo0 nominal canary，随后hard stop再次审核。暂不授权task20/task19 manipulation、J6 generation、训练或ZIP。
