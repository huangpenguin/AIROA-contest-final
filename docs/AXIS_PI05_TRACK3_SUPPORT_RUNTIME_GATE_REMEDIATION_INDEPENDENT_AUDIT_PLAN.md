# AXIS Pi0.5 Track 3：SR0–SR13 独立审核计划

> 创建日期：2026-09-07（Asia/Tokyo）
> 唯一活动计划：是
> 审核对象：Support Runtime Gate 修复与 Gold Continuation 的执行闭包
> 前置执行计划：`docs/archive/axis_pi05/AXIS_PI05_TRACK3_SUPPORT_RUNTIME_GATE_REMEDIATION_AND_GOLD_CONTINUATION_PLAN.md`

本计划只授权只读独立审核。不得启动新 physics、Teleop、GPU replay、generation、accepted promotion、训练、norm、FAST/ZIP、upload 或 submission。旧 SG/CR/BR/AC identity 与本轮 attempt `0001–0004` 均不可覆盖或重跑。

## A0：边界与账本

- 冻结 clean-source `5c5c74641ba39e41709e2c51ce576e7c065060fc`、两次 env 前 SR5 失败 attempt、attempt `0003` 通过 freeze、attempt `0004` 两条 CPU identity。
- 核对 nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`、root porcelain、`LIBERO-plus` 原 dirty、无 live worker、GPU compute empty、无 `.partial`。
- 分别记录 launch attempt 与 physics identity；不得把 TypeError/identity freeze 失败算成物理 family failure。

## A1：code/fixture 合同

- 复算 acquisition 只检查 dwell 末 `stable_ticks` 窗口、source release 在 lift 之后、bowl↔main_table 为 goal receiver、goal 连续窗口而非末帧。
- 复算 success-path fixture 可达，以及 unrelated/unknown/late-contact 负例。
- 复算 prepare-only 使用真实 HDF5/XML/start bundle 且未创建 `ControlEnv`。
- 审核两次 pre-env identity 失败是否被诚实保留，以及第三次 freeze 是否使用本模块 tool SHA 且 porcelain 前导空格未被 strip。

## A2：SR6/SR7 physics

- 只读核对两条 child 的 identity、CPU/no-renderer、return 0、typed no-pickle finite trace、stage/contact/source/goal 三态。
- 确认均为正常 `family_failure`，停在 `grasp_dwell` 且 dwell 窗口 demonstrator contact 为 false；lift 后的门为 `not_evaluated`。
- 复算 SR7 仅因量化进展（更晚 stage / 进入 acquisition）启动，不是同策略 retry。
- 确认 success=0，因此 GPU3 replay 与视觉包必须是 `not_evaluated`。

## A3：人工路径与 finalizer

- 确认 SR8 为 `human_input_required`，agent 未产生键盘输入，Teleop 未开始。
- 确认 SR12 未使用失败轨迹、norm/training fail-close。
- 独立重算 manifest/outer/closure、strict JSON、NPZ typed/no-pickle、无 partial、process/GPU。
- `sr13_closeout.json` 的 `not_evaluated` 空列表不得被解释成 SR9–SR11 已执行。

## A4：结论

只能返回 `approved`、`remediation_required` 或 `blocked` 之一。无论结论如何，不得自动授权训练、新 ZIP、上传或比赛提交。
