# AXIS Pi0.5 Track 3：Support Release Anchor 与 OSC Scale 恢复计划

> 制定日期：2026-09-05（Asia/Tokyo）
> 审核输入：root `11483623b7710430a197231b8ed0662c4813c12d`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 当前状态：`track3_support_release_anchor_and_osc_scale_recovery_planned`
> 唯一授权链：`W0_to_W6_then_hard_stop`
> 用户人工操作：`not_required`
> 不授权：修改历史证据、task10 P3 attempt4、conservative profile、task20/task19 manipulation、pending/accepted demo、generation/replay batch、Teleop、训练/norm、GPU0、官方GPU、dense full-FT、FAST/ZIP、上传或比赛提交

## 0. 目标与边界

V4 的 `family_failure` 与 hard stop 保持历史事实，但当前实现仍没有执行计划声称的 nominal profile：reverse acquisition 使用 forward close-side 单点 pose，rotation action 又没有按 Robosuite controller output scale 归一化。本计划只闭合这两个合同、stage-dependent tri-state 和真实 launcher identity，然后先用 CPU/no-renderer 做一次完整 canary。CPU 通过才允许一次 GPU3 camera canary；任意分支随后 W6 hard stop。

## 1. W0：输入、历史状态与审计锁（CPU only）

- 锁定 `1148362`、V0–V5 parent/child/trajectory/frames/videos、19项manifest、本次审核summary、task10 HDF5、9-state bundle、BDDL/model hash、R6/norm hash。
- 保留历史 V4 `family_failure`、原始 gate labels 与 `first_error`；三态修正只写入新 derived namespace，禁止回写旧 parent/child/V5。
- 复算 V0 `18/18`、V5 manifest `19/19`、NPZ/NPY/video schema；记录root/OpenPI、`LIBERO-plus` dirty snapshot、live PID/cursor/heartbeat和GPU0–3。
- W0 不得 import simulator/CUDA。任何 immutable drift、活动 supervisor、未知 dirty source 或 GPU 边界冲突立即 W6 hard stop。

## 2. W1：same-demo release-side HDF5 acquisition contract（CPU/no-renderer）

- 对 task10/demo0 exact model 逐 state 复算 gripper、EEF、bowl、plate 和 contact-v3；锁定两个 action transition：close `32`、release `76`。
- reverse acquisition anchor 固定为 `state[75]`：release 前最后一个 gripper=`+1` 的 state。它必须同时有 demonstrator contact、bowl↔plate receiver contact且无 unrelated contact；预期审计值为 `3/1/0`，不符即停止。
- 物化 index32→75 的 object-relative EEF time series，记录 translation/rotation drift；禁止再用 index32 单点重构代替 object-follow/contact window。
- reverse grasp target 使用 terminal bowl pose 与 `state[75]` 的 `T_object^-1·T_eef`；place target继续使用同 demo `state[0]` bowl pose。所有 quaternion 边界显式 `xyzw`，`q/-q`等价。
- 本阶段只构造/验证 pose，不执行 manipulation；task20/task19 保持 `not_evaluated`。

## 3. W2：OSC physical scale、进度与 tri-state 合同（CPU only）

- 启动前读取实际 controller config并锁定 `input_max=1`、translation `output_max=0.05 m`、rotation `output_max=0.5 rad`；任何配置漂移停止。
- 先按物理 profile 对 error 做 norm clip（translation `0.05 m`、rotation `0.12 rad`），再逐维除以 controller output scale生成normalized action。故最大rotation action norm为`0.24`，不得再把`0.12 rad`映射为`1.0`。
- 回归覆盖 translation/rotation exact scaling、world-frame left-delta orientation、`q/-q`、actual controller config、finite与gripper `{-1,+1}`。
- gate按stage prerequisite三态化：未进入close/lift时grasp/contact-completion为`not_evaluated`；未进入place/release时same-demo destination为`not_evaluated`。initial source、final goal、prefix contact safety、collision、unrelated movement、stall/timeout分别报告，不得合并后把未执行项写false。
- trajectory新增每帧 EEF pose、target pose、physical delta、normalized action、physical-clip/controller-clip标记和stage进度；stall必须基于冻结窗口内的误差进度与stage budget fail closed。

## 4. W3：真实 supervisor、release tests 与 clean commit

- 实现实际被调用的单一 supervisor entrypoint：启动前校验 clean root、expected root commit/tool SHA、OpenPI HEAD、PID不存在/未复用、cursor inactive、精确命令与GPU边界。
- 运行中原子写初始/周期/final heartbeat；15分钟是生产上限，测试用可注入短间隔证明heartbeat确实更新。parent必须记录root commit、tool SHA、supervisor/worker PID与command。
- result publication、failed_gate return 0、exception/timeout TERM→KILL、三类parent mapping、first-error propagation和final-manifest顺序继续回归。
- 运行strict JSON、NPZ/NPY `allow_pickle=False`、CSV LF、py_compile、targeted/full CPU tests与`git diff --check`；形成clean release commit。之后不得修改executor/profile/threshold/anchor。

任一检查失败即W6 hard stop，不得运行W4/W5。

## 5. W4：唯一 CPU/no-renderer 完整 task10 canary

仅允许一个fresh process：

```text
family       support_remove
task/demo    task10/demo0
profile      release_anchor_scaled_nominal
seed         3121
renderer     none
max runs     1
```

- 使用W1 anchor、W2 scaling和原V4 stage budgets/tolerances；不允许看到结果后调参、重跑或切换conservative。
- 记录完整state/action/EEF/pose/contact/gate tri-state与进程provenance，不生成训练demo。
- 任一 infrastructure failure或任何required family gate失败：直接W6 hard stop，W5=`not_evaluated`。
- 只有所有非camera required gates通过，才能冻结action/profile/hash并释放W5。

## 6. W5：条件式唯一 GPU3 camera canary

W4全通过后，才允许GPU3运行一次相同task/demo/profile/seed的fresh camera canary。必须记录PID、GPU3 UUID、起止快照、峰值、query/heartbeat、root/tool identity、完整gate、双相机帧与两路H.264；GPU0外部进程不得触碰。

- 全部通过：`track3_support_release_anchor_scaled_nominal_passed_pending_generation_review`。
- 正常发布的family gate失败：`track3_support_release_anchor_scaled_nominal_failed_gate_pending_review`。
- infrastructure/provenance失败：`track3_support_release_anchor_scale_recovery_stopped_W5_<reason>`。

三种分支均立即W6；不创建pending/accepted demo，不进入其他family或generation。

## 7. W6：统一收尾与 hard stop

- 先关闭phase reports、CSV、cursor、supervisor closeout，再生成排除自身/log/PID/partial的checkout-stable manifest；全部条目必须Git tracked且hash exact。
- 未执行阶段和依赖gate显式`not_evaluated`；历史V4 label只在derived errata中纠正。
- 复跑strict JSON、NPZ/NPY/video/CSV/hash、py_compile、targeted/full tests与`git diff --check`，形成clean commit。
- `exact/pending/accepted`最大仍为`0/0/0`；验证GPU1–3释放、GPU0外部进程保留、root/OpenPI clean、`LIBERO-plus`既有dirty未改。
- hard stop并等待下一次审核；不得自动启动conservative、task20/task19、generation、训练或ZIP。
