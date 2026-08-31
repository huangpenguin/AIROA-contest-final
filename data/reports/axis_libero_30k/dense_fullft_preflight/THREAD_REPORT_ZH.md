# Dense full-FT preflight execution report

- 状态：`dense_fullft_preflight_failed`。
- Axis params：`c342608c489515efa9ba8281400dc648bf3970e1`，tree SHA-256 `d21ff40cbff9b0435cd6f3fd879479585f4ea19a03d7caab4a88f0c3613aaf14`。
- canonical LIBERO：`a4336d589d589045d1c56423ffdf3b88a0e19b1f`，native 10Hz，1693 episodes / 273465 frames。
- fresh QUANTILES norm：`4c803bf59c52677c36b6b420c614d8a7742e0847708443863e13e67cb80b18f5`；Axis norm 仅记录 hash，未加载。
- dense 参数：3,353,433,872 FP32 params，`nnx.Nothing`，EMA `0.999`，global batch `64`。
- 磁盘门禁：`passed`，可用 894.73 GiB，预留需求 258.709 GiB。
- 两卡 canary：GPU1+GPU2、microbatch=1 / accumulation=32；首个计算申请约 1.12 GiB 时 OOM，未产生 update/checkpoint。
- GPU0 随后自然释放；四卡仅执行一次 microbatch=1 / accumulation=16 的条件预飞行。XLA 将计算降到约 25.00 GiB 后，每卡申请 54 MiB 时 OOM，未产生 1/10/50/100 checkpoint。
- 两次均保留 FP32、EMA、`nnx.Nothing`、global batch 64；没有通过改配方救活。GPU0 外部进程未被触碰，结束后四卡均约 5 MiB。
- 因 F2 无可用拓扑，F3 restore/C20 与 F4 吞吐/60小时预算按 hard gate 未执行；不得启动 1K、官方 GPU、FAST、ZIP 或提交。
- CPU 测试：`.venv_axis_pi05/bin/pytest -q tests` 为 `125 passed, 7 skipped, 21 warnings`；直接无路径运行全仓 `pytest -q` 的 collection 仍会因 vendored OpenPI 缺少 `pynvml` 失败，未安装依赖。
- 详细证据：`two_gpu_canary.json`、`two_gpu_canary.log`、`four_gpu_preflight.json`、`four_gpu_preflight.log`。
