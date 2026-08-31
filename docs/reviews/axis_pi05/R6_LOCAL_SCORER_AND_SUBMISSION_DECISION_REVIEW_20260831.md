# R6 本地 scorer 环境与人工提交决策审核

> 审核日期：2026-08-31（Asia/Tokyo）
> 输入 commit：`43aef30`
> 审核结论：`approved_with_score_pending`
> 用户外部动作：已人工提交 Track 1 ZIP；submission ID、完成时间、分数与官方日志尚未写入仓库。

## 1. 提交状态边界

用户明确报告已将下列 ZIP 提交至 Track 1：

```text
assignment    1007 / Track 1
artifact      artifacts/submission_axis_r6_step500_fast_positive_score_20260831.zip
bytes         9383571677
sha256        95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a
```

仓库当前没有 Omnicampus submission ID、提交时间、计分完成状态、score 或官方 log，因此只能记录为：

```text
user_reported_submitted_score_pending
```

不得把该状态写成 `official_result=true`，也不得推测分数或自动重复提交。

## 2. README 的准确含义

仓库 README 明确写明：

- Dockerfile 使用与本番 scorer 相同的 GPU base/dependency 构成；
- `evaluate.py` 和 validator 复现正式评测的处理步骤、成功判定与单请求 10 秒限制；
- 本地只包含公开 example tasks，不包含隐藏任务与 leaderboard score 配置；
- 正式 scorer 是 NVIDIA L4，配布开发 GPU 是 RTX PRO 6000 Blackwell，两者不是同一硬件。

因此“本地可构建官方同构环境”的正确表述是：**依赖、评测流程和接口约束同构，隐藏任务、排行榜分数和 GPU 硬件不等价。** 官方 Blackwell pre-check 是推荐的额外兼容检查，不是提交命令的必需前置条件。

用户已经提交后，再为同一 immutable ZIP 消耗配布 GPU 做 pre-check 不会改变这次提交。将其额度保留给最后训练冲刺是合理决策。

## 3. 本机 Docker 可用性审计

```text
host driver                 580.159.03
host reported CUDA         13.0
local GPUs                  4x V100 32GB
GPU0                        external PID 237527, do not touch
GPU1-3                      free at audit
Docker                      28.1.1
nvidia runtime              available
image                       parc2026:latest
image id                    2481fd635a465c6b9dd38b6101c8307ce97d0bf7a3c2ffb57e487103cec9ccec
image created               2026-08-24T23:22:22.322942931+09:00
image size                  40823788898 bytes
container CUDA              13.0.3
container Python            3.10.12
container PyTorch           2.11.0+cu130
container torch CUDA        13.0
```

核心构建输入一致性：

```text
Dockerfile                  152e62a0c3a03ceec5830c449ea13f78a6a952393ebc2de2122bdce93dcb945c  exact
setup.sh                    28161d18f846a7486f7e9fb49e4033ec48c942ad88ad6f92f6adf5e54e6e4057  exact
evaluate.py                 5c8248db309ab578d1dcecfc44306d565b91ae10e809503ff456fd9c9832d5ca  exact
validate_submission.py      6dd746afa4747d6d21f117ad0c830d3c8f16a705545dbb86ee8d893c51fc47c4  exact
```

README hash 不一致仅来自后续说明文档更新；执行用 Dockerfile/setup/evaluator/validator 均一致，所以没有必要重建 40.8GB image。当前尚未对本次 R6 ZIP 执行真实 `docker run ... evaluate.py`，历史 `official_docker_e2e.json` 仍只是 staged-server equivalent。

## 4. 决策

1. 归档官方 Blackwell pre-check 计划；不使用配布 GPU测试本次已提交 ZIP。
2. 在自有 GPU1 对现有 image 补一次真实 Docker E2E，作为本次计分的诊断对照；不得触碰 GPU0。
3. 等待并归档官方 score/log；未出分前不得重复提交。
4. 配布 GPU 额度冻结给最后训练冲刺。正式使用前，先在自有 V100 完成 full-FT 数值预飞行、recipe选择和可恢复性验证，避免把 60 小时消耗在调试上。

本审核不授权启动训练。训练路线必须依据官方结果区分“部署失败”与“行为能力不足”后再单独落盘。
