# R6 execute4 官方试分候选交付报告

> 日期：2026-09-16（Asia/Tokyo）
> 状态：`ready_for_user_manual_submission`
> 自动上传/提交：false

## 交付文件

- ZIP：`/home/huang/code/PARC2026_final/artifacts/submission_axis_r6_step500_fast_execute4_official_trial_20260916.zip`
- SHA256：`2f2f9c2ed627de50b1af4b9d1501e9f5d4d467e8455ed51f363247a3d7bd3fc9`
- bytes：`9383572164`（约 8.8 GiB）
- entries：`3401`

## 改动边界

来源为正式得分 R6 FAST ZIP，FAST 权重 SHA256 仍为
`e55bf7dcca78b2195993dd7a12b96a4ea13e94761d711c99a82b7411c3f8ceb6`。

相对原 ZIP，仅以下三个 entry 的 size/CRC 发生变化：

1. `policy_server.py`；
2. `model_weights/config.json`；
3. `submission_manifest.json`。

唯一行为变化为 `n_action_steps: 8→4`。模型 chunk 仍为 10，ACTION_EMA=0，RTC=false；
weights、processor、norm、vendor 和 requirements 均未改变。其余 3398 个 entry 的 size/CRC
与原 ZIP 完全一致。

## 验证

- focused tests：打包器 7 passed；连同 horizon adapter 共 14 passed。
- source ZIP SHA、FAST SHA、计划 SHA、comparator SHA：全部 exact。
- GPU0 packaged-server attempt_0002：
  - FAST 冷启动 332.9 秒；
  - health/reset 通过；
  - 6 次 `/act` 全部成功、finite/nonzero；
  - 日志明确 `chunk=10 execute=4`、`n_action_steps=4 action_ema=0.0 rtc=False`；
  - 受控 SIGTERM，returncode `-15`，无 orphan。
- ZIP CRC/testzip：通过。
- package topology：通过，无 duplicate/symlink/nested archive/slow model。
- `validate_submission --static`：PASS，0 errors、0 warnings。
- 无 `.partial` / `.tmp`；GPU0–3 均回到 5 MiB / 0%。

首次 runtime smoke 因当前 venv 缺少 FastAPI/uvicorn，在模型加载前退出；该失败已原样保留。
attempt_0002 复用了历史成功链已缓存的只读 ASGI runtime，并完整通过，未覆盖 attempt_0001。

## 风险与建议

该候选是用户显式风险覆盖，不改变 horizon screen 的历史结论：execute4 的本地 C20 为
`17/20`，相对 execute8 `15/20` 为 `+2`，paired net wins 也是 `+2`，没有达到原
双 `+3` 门；B80 未执行。因此官方提分不保证。

四个 reverse case 仍全部为 `0/4`，所以本次改动没有 T3 reverse 增益证据。若最后一次机会
可以选择赛道，execute4 更适合尝试 T1/T2 的前向闭环，而不是把它当作 T3 已验证修复。
若排行榜保留历史最高分，此次探索的下行风险较小；若只计最后一次提交，则应明确接受可能低于
既有 T1 R6 `0.47` / T2 R6 `0.33` 的风险。

## 证据

- release：
  `data/reports/axis_libero_30k/r6_execute4_official_trial_20260916_attempt_0001/release.json`
  （SHA256 `30d72dfefce6a57d11ded3c01e04e74f01bdfd172816dd1b7ee377623bd2a67e`）
- runtime smoke attempt_0002：
  `data/reports/axis_libero_30k/r6_execute4_official_trial_20260916_attempt_0001/runtime_smoke_attempt_0002.json`
  （SHA256 `bc61903a7262544a52f1669d7b1209a777dff7c9ee33b24d2eccb217edf1470e`）
- ZIP delta：
  `data/reports/axis_libero_30k/r6_execute4_official_trial_20260916_attempt_0001/zip_delta_audit.json`
  （SHA256 `1dc8de834f6abaed15d0456d870d68d57ebd692a1e7126de24bf60175d64e5c4`）
