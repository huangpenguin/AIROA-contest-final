# Track 3 Dynamic Gate Model/State Reconciliation：D0–D4 收尾

- 状态：`track3_native_reverse_dynamic_canary_failed_support`
- D0：输入证据已锁定；三份官方 HDF5 的 `states[0] == init_state` 均为真，state 的 time/qpos offset 为 0/1；旧 G1 report 未覆盖。
- D1：CPU/schema 门禁 15/15 通过。reverse predicate 走 reverse env 的逐条 `_eval_predicate`，controller 顺序和 7D OSC_POSE hold action 已固定。
- D2：GPU3 fresh task10 只执行一次；validator 在 site detail 处遇到 `TargetZone` 缺少 `get_geom_state`，未得到可用的 dynamic predicate/settle 证据。该次已按 hard stop 记录，修复后不重跑。
- D3：因 D2 未通过，未运行三族 3/3。

## 结果边界

本轮不能判定 reverse semantics 成功或失败；它只证明 D2 首次实现仍有一个 validator schema 缺口。没有 SE(3) 抓取、demo、pending、训练、Teleop、FAST/ZIP 或上传。等待下一次审核决定新的最小 remediation。

证据：`d0_release_lock.json`、`d2_task10_canary.json`、`runs/d2_task10_canary_run_1.json`、`summary.json`。
