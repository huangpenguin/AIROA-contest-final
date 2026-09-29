# AXIS Pi0.5 R6 最终推理 Horizon Sweep 计划

> 日期：2026-09-16（Asia/Tokyo）  
> 状态：`authorized_inference_only_screen_not_training`  
> 目标：利用最后一次官方测分机会，只改变闭环重规划频率，筛出是否存在足够大的本地增益

## 冻结边界

- 模型固定为 R6 step500；params、folded weights、processor 与 QUANTILES norm 不变。
- `chunk_size=10` 固定；这是模型输出 horizon，不是今晚的调参项。
- `RTC=false` 固定。现有 Pi0.5 submission server 明确拒绝 RTC；RTC 仅实现给 SmolVLA。
- `ACTION_EMA=0`、20 Hz、max 300、case/init/policy seeds 固定。
- 不训练、不 resume H300、不改数据、不 fold；筛选阶段不生成 ZIP、不上传、不提交。

唯一变量是 `n_action_steps`，即每个 10-step chunk 执行多少步后丢弃尾部并重新推理。

## I0：代码与 preflight

新增独立 adapter，不修改既有 paired-eval adapter 或历史 evidence。adapter 必须把自身 SHA、
本计划 SHA、R6 双哈希、Fast400 norm SHA 与 24 个冻结 case fingerprint 写入每个 attempt。

focused tests、py_compile、strict JSON 和四卡 idle gate 通过后，才进入 I1。

## I1：四卡并行 24-case screen

| GPU | variant |
|---:|---:|
| 0 | execute4 |
| 1 | execute6 |
| 2 | execute8 fresh baseline |
| 3 | execute10 |

每个 variant 都跑相同的 4 reverse + 20 C20。任一 worker 出现 infrastructure、NaN/Inf、
全零 trace、case/fingerprint 不一致或不满 24/24，该 variant 失败；不得补挑 case。

## I2：晋级门

候选只有同时满足下列条件才进入确认：

1. C20 相对 fresh execute8 至少 `+3/20`；
2. C20 paired wins-losses 至少 `+3`；
3. C20 collision 数不高于 execute8，timeout 最多高 `1`；
4. reverse 不回退，全部 trace finite/nonzero，无 model/infrastructure exception；
5. 若多个候选通过，依次按 C20 success、paired net wins、较少 collision、较少 timeout、
   与 execute8 距离较近排序，只保留一个。

`+1` 或 `+2` 不足以承担唯一官方机会的选择偏差，按失败处理并保留现有 R6 ZIP。

## I3：条件式确认与发布

只有 I2 通过才新增 20 Hz B80 confirmation adapter，使用四卡把 execute8 与唯一候选各分成
两个 40-case shard。候选需达到 B80 paired net wins `>=4/80`、success 不低于 execute8、
collision 不增加，才允许制作一个 inference-only R6 ZIP。

新 ZIP 只能改变 `n_action_steps` 及与之对应的日志/manifest contract；weights、processor、
norm、prompt aliases、FAST runtime 其余字节保持一致。必须通过 static/CRC、真实 server health、
固定输入 parity 与机械 manifest 审计。不得自动上传或提交；最终是否使用最后一次官方机会由
用户决定。

任一门失败即恢复提交姿态：T1 R6 `0.47`、T2 R6 `0.33`，不使用 H10K `0.13`，T3 不发布
H300。

