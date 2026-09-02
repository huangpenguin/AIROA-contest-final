# P0 官方公开 LIBERO HDF5 锁定报告

状态：`track3_public_terminal_source_locked`

- 仅下载公开 `libero_10`、`libero_goal`、`libero_object`、`libero_spatial` 四个 suite，各 10 个与锁定 40-task catalog 对应的 HDF5。
- 所有文件通过 `.partial` 下载、完成后 atomic rename，并记录 bytes、SHA-256、HF revision、来源 URL 与时间。
- 每个文件均通过 HDF5 可读性检查，并确认 `data/demo_0/states`、`actions`、`rewards`、`dones` 存在。
    - task/BDDL 映射为公开 LIBERO task map；reverse BDDL 从本次官方 BDDL 与 `.pruned_init` CPU 编译；候选 spec 未计为 accepted episode。
- P1 尚未执行；GPU、simulator、teleop、训练和外部操作均未启动。

详细 machine-readable 证据见 `p0_lock.json`、`p0_download_manifest.json`、`p0_hdf5_schema_inventory.json` 与 `p0_task_mapping.json`。
