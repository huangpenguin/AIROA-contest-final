# AXIS Pi0.5 Track 3：Cross-Family Acquisition Canary 独立审核计划

> 创建日期：2026-09-06（Asia/Tokyo）
> 唯一活动计划：是
> 性质：只读独立审核；不授权任何新执行

本计划只审核已完成的 AC0–AC6 attempt-specific evidence。不得启动新的
physics、GPU、ControlEnv、旧 identity、generation/replay、training、ZIP、
upload 或 submission；不得修改历史 ER/EG namespace、公开 HDF5/BDDL 或
当前 AC attempt raw evidence。

## IA0：输入与闭包

- 固定审核 root/source commit、AC `attempt_0001`、`latest_attempt`、
  manifest、outer hash、closed heartbeat、inactive cursor 和 checkout
  closure。
- 逐文件复算 AC evidence bytes/SHA、NPZ `allow_pickle=False`/finite、CSV
  LF、strict JSON、attempt pointer 与 manifest closure；历史 partial
  observation 必须保留，不能被解释为不存在。

## IA1：执行与分类

- 只读核对 AC0–AC6 phase ledger、profile/quota lock、三条实际新 CPU
  identity、固定 support→containment→articulation 顺序、无 retry、family
  isolation、`family_failure`/`infrastructure_failure`/`not_evaluated` 语义。
- 复算 runtime-derived contact、saturation、stage、source/goal、typed
  trace 与 `exact/pending/accepted/training` 计数；成功（如有）只能是
  `pending_replay_review`。

## IA2：边界与独立结论

- 独立核对 parent→child identity、resume hash、singleton、heartbeat、
  timeout TERM→KILL、no-descendant、sticky stop、root/OpenPI/LIBERO-plus
  boundary、Track3 process 和 GPU0–3 compute scan。
- 按仓库规则输出且只输出一个审核分类：`approved`、
  `remediation_required` 或 `blocked`；历史缺失/partial/raw provenance
  不得重构，不得以派生记录冒充 raw。

审核完成后保持 hard stop；若需要任何后续执行，必须由新的审核授权计划
明确释放，不能从本计划自行恢复。
