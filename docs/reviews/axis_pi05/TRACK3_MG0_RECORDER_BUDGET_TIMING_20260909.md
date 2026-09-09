# Track3 MG0 recorder budget/timing remediation（2026-09-09）

## 结论

MG0 的 code-only 与 no-save A/B 已完成，状态为 `ready_for_user_calibration`，并在进入 MG1 前 hard stop。此次没有 formal recording、candidate/pending episode、replay、accepted promotion、training、ZIP、upload 或 submission。

执行源为 root `20fe0014c19b725feb77cf858a1d1f872aea926b`，nested OpenPI 为 `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`。录制器文件 SHA256 为 `dee3cd22abae4adc3e118e7c472642f9e4730bb0ec2e1cc0aa01c62800c83d5a`。

## 不可变输入

- O1.0 preferred seed：`official_t3_red_mug_1788838632`，388 steps；只读使用。
- O1.0 reference seed：`official_t3_red_mug_1788838293`，468 steps；只作参考。
- O1.0 input lock：`data/reports/axis_libero_30k/track3_operations_o1_0_red_mug_retiming_20260908/input_lock.json`，SHA256 `41c2fa144afc20eb2c3b639af3983f2dff6b6b637848fd5cda33b1680da41298`。
- 两条历史 pending raw、metadata、contact 截断事实和 `gold_eligible=false` 均未修改。

## code-only remediation

- formal recording 的真实步数上限为 `300`；预警节点为 `220/260/290`，到 `300` fail-close。
- pause 分支不调用 `env.step`，断线进入 pause。
- keyup 清空 held key、motion key 和 motion pulse；动作立即为零。
- timing 字段分离 `sim_control_hz`、`operator_tick_hz`、operator tick period、wall timestamps 和 video fps；deadline 以 scheduled operator tick 为基准。
- `--no-save/--canary` 只返回诊断 JSON，不写 episode；Save Candidate 事件在该模式被忽略。
- 首次 browser-only pacing 缺陷保留在 attempt 0001，后续修复提交为 `4b697e4`；没有把该失败隐藏或重写为通过。

## no-save A/B

两次都在现有 TigerVNC display `:3` 的 native MuJoCo 进程中执行；事件通过 localhost-only HUD 接口注入，仅用于 no-save calibration。

| attempt | operator tick | frame count | observed interval mean / median | min–max | deadline miss | episode write |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0003 | 2 Hz | 4 | 0.498729 / 0.496550 s | 0.496470–0.503167 s | 0 | false |
| 0004 | 4 Hz | 6 | 0.249058 / 0.248739 s | 0.244150–0.254548 s | 0 | false |

Behavior probes show keyup immediately changed the HUD axis to `none` and emptied `held_keys`. Pause held the simulator step constant (2 Hz: 4→4; 4 Hz: 6→6). Both runs ended with abort, no save event, and `replay=not_evaluated`.

## 资源与停止点

MG0 未启动训练、replay、formal Teleop 或新录制；GPU3 只被用于 canary simulator，GPU0–2 外部占用未触碰。canary 结束后无 Track3/MuJoCo/Teleop 活进程。历史 O1.2 314-step episode 仍为 `physical_success=true`、`gold_eligible=false`、`replay=not_evaluated`，不进入 accepted/training。

下一阶段只释放 MG1 的 red-mug assisted re-execution。它必须从冻结 start state 连续重新执行 retiming blueprint，目标总长 `<=260`、硬上限 `300`，不能离线删帧。正式人工 fallback 仍需用户明确回复：

`接管 Teleop：official_t3_red_mug`
