# AXIS Pi0.5 当晚正分优先级决策记录

> 决策时间：2026-08-31 18:11（Asia/Tokyo）
> 输入 commit：`58d3fcc46cfa1eaffd5d057be59edf97b03f02eb`
> 决策：暂停非必要统计证据，只保留最短 R6 FAST→server→ZIP critical path。

## 1. 当前到底在哪个阶段

```text
数据下载/审计/manifest/norm          已完成
R6 LoRA 500-step                    已完成
LoRA fold 为 on-disk PyTorch        已完成
本地 direct-policy B80              已完成：49/80
folded PyTorch frozen parity        已完成
FAST model_fast.safetensors         尚未生成
FAST policy server                  尚未验证
R6 ZIP                              尚未生成
官方提交/分数                       尚未执行
```

因此当前不是训练阶段，而是部署转换阶段。距离“有一个 ZIP”主要还差 FAST 导出、行为 parity、server smoke 和打包。

## 2. 为什么今晚不追加 full-FT

Axis checkpoint 具有通用知识，标准 LIBERO 长程 full-FT 是正确的后续研究方向；原计划的 `1K→3K→10K→30K` 仍保留。但它不是今晚正分的最短路径：

- Axis init direct local 为 `48/80`，R6 为 `49/80`，已经有非零行为能力；
- 500-step R6 的统计提升不明确，但没有证据表明必须再训练才能产生动作；
- 官方 0/6 日志直接显示 SLOW path、`fast=False` 和 vision key warning，属于部署链故障；
- 新 full-FT 仍需重新 fold、FAST export、server 和 ZIP，不能绕过当前 blocker；
- 当晚短 full-FT 容易引入 optimizer/NaN/norm/灾难回归风险，且不等价于原团队约 30K 的完整适配规模。

结论：今晚先把已有会做任务的 R6 正确送进 FAST 包；拿到正分后再启动规范 full-FT。

## 3. 旧包核验

```text
submission_track1_pi05_pretrained_base_20260814.zip
  only model.safetensors
  no model_fast.safetensors
  SLOW risk，不作为今晚回退提交

submission_axis_f2_m1_500_fast_20260829T015824.zip
  has model_fast.safetensors
  historical fixed-input behavior parity NOT_READY
  禁止改名冒充 R6 或无门禁提交
```

今晚只做一个候选，避免并行维护多个来源不明的包：`r6_dual_l2_fresh500_r1`。
