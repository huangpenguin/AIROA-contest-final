# R0 旧证据勘误

- 旧 P0–P6 证据保持 immutable。
- 旧 manifest 实际包含 `p6_summary`，共 17 项；旧 P6 的 `entries_exclude_self_and_p6_summary` 字段不准确。
- 旧 `source_commit=6cfda92` 只表示执行父提交，不代表本轮修复工具版本。
- 本轮 R0 只闭合 Git tracking 与只读复核，不重写 NPZ 或视频。

状态：`track3_p3_recovery_r0_artifact_closure_passed`
