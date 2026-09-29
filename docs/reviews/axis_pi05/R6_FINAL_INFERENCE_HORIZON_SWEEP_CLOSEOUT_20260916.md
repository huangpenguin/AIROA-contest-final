# R6 最终 inference-horizon sweep closeout

> 日期：2026-09-16（Asia/Tokyo）  
> 结论：`screen_failed_keep_existing_r6_zip`  
> 发布决定：不进入 B80，不 fold、不生成新 ZIP、不上传、不提交

## 1. 冻结条件

- 模型：R6 step500，weights / processor / QUANTILES norm 均未改变。
- 控制：20 Hz、`chunk_size=10`、RTC off、ACTION_EMA=0、max steps 300。
- 唯一变量：每个 chunk 实际执行的动作数 `n_action_steps=4/6/8/10`。
- 面板：每个 variant 同一组 4 reverse + 20 C20，共 24 cases。
- 四个 worker 均 fresh process、24/24 完成；无 model/infrastructure exception。

## 2. 结果

| execute horizon | C20 success | 相对 execute8 | paired wins/losses/net | C20 collision | C20 timeout | reverse success | 晋级 |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4 | 17/20 | +2 | 3/1/+2 | 1 | 2 | 0/4 | 否 |
| 6 | 15/20 | 0 | 0/0/0 | 4 | 1 | 0/4 | 否 |
| 8 | 15/20 | baseline | 0/0/0 | 4 | 1 | 0/4 | baseline |
| 10 | 12/20 | -3 | 0/3/-3 | 5 | 4 | 0/4 | 否 |

所有 96 个 case 的 trace 均 finite/nonzero，四组 exception 均为 0。fresh execute8 重现
15/20，说明本次同面板 baseline 与冻结参考一致。

execute4 的方向是正向的：success +2、collision 4→1；但它没有达到预注册的
`C20 >= +3/20` 与 `paired net wins >=3`，仅为 `+2/+2`。由于这是从三个候选中筛选、
且用户只剩最后一次官方测分机会，不能把该幅度当成足够稳健的发布证据。

## 3. 门禁决定

- I2：失败；`selected_horizon=null`。
- I3 B80：`not_evaluated`，因为只有 I2 通过才授权。
- inference-only ZIP：`not_generated`。
- fold / upload / submit：均未执行。
- RTC：未启用；当前 Pi0.5 server 明确 fail-close。
- `chunk_size`：未修改；仍为模型输出合同 10。

因此恢复截止日前提交姿态：保留已证明的 T1 R6 `0.47` 与 T2 R6 `0.33`；不使用
T2 H10K `0.13`；T3 不发布 H300，也不以 execute4 替换既有 R6 ZIP。

## 4. 证据与资源闭包

- comparator：
  `data/reports/axis_libero_30k/r6_final_inference_horizon_sweep_20260916_attempt_0001/comparison.json`
- comparator SHA256：
  `32cf8efb3a8100329bae53b3b4be2ed09e252350ec6db617c84333ddf04bd15b`
- evidence root：
  `data/reports/axis_libero_30k/r6_final_inference_horizon_sweep_20260916_attempt_0001/`
- output root：
  `outputs/axis_pi05/r6_final_inference_horizon_sweep_20260916_attempt_0001/`
- 预注册计划 SHA256：
  `10674b6b1d55fec8d8ae99b9523a4a6985370a5fab51b4d98dd0d49403dae643`
- adapter SHA256：
  `a46666ff96d1f74ea3ca8c0e769b95cd21184cf7f57356a1404092676084e24e`
- focused tests：`7 passed`；py_compile、strict JSON、git diff --check 均已通过。
- closeout 时 GPU0–3 均为 5 MiB / 0%；无相关 policy/eval process；无 `.partial`。

## 5. 后续边界

本轮不再做同面板 horizon 搜索；继续尝试会增加选择偏差，且无法在最后一次官方机会前形成
独立确认。若未来有新的时间预算，execute4 可作为研究假设进入预先冻结的更大独立面板，
但不能把本次 17/20 直接升级为发布候选。
