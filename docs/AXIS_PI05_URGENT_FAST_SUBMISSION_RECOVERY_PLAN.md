# AXIS Pi0.5 现有微调权重 FAST 提交紧急恢复计划

> 状态：待执行计划；本文不代表已经导出 FAST 权重、通过评测或完成提交
> 日期：2026-08-28（Asia/Tokyo）
> 时间边界：不等待新的 30k 训练；只修复现有自训练 checkpoint 的转换、推理和打包链
> 主候选：`artifacts/submission_axis_f2_m1_500_20260826.zip`
> 目标 Track：先只恢复 Track 1；Track 2/3 不与本次紧急包共用提交决策
> 提交边界：执行者不得自动调用 `omni submit`，正式提交必须由项目负责人单独确认

## 0. 给新执行对话的任务摘要

执行本计划时，只处理“已有 Axis F2/M1@500 微调权重的 FAST 转换与提交链恢复”。不要启动任何训练，不要修改 30k 研究线，不要从 Axis init 重新训练，也不要把旧仓库的 `model_fast.safetensors` 复制进当前候选。

主线如下：

```text
冻结原始 F2/M1@500 submission ZIP
  → 安全解包到全新 staging 目录
  → 移植并加固旧 FAST exporter
  → 从该包自己的 model.safetensors 导出 model_fast.safetensors
  → strict key / vision embedding / fixed-input 等价门禁
  → policy server 禁止 SLOW fallback
  → chunk=10 / execute=8 / EMA=0 / RTC=false
  → 本地 server latency + Track 1 小面板
  → 官方 Docker E2E
  → 生成新的、低于 20GB 的独立 ZIP
  → 等待人工确认是否提交
```

执行者不是仓库中唯一的工作者。不得 `git reset`、`git checkout --`、覆盖其他 Agent 的文件或清理既有 outputs/artifacts。所有新输出使用本文规定的独立 namespace；原 ZIP 和旧仓库只读。

## 1. 已核实的事实与边界

### 1.1 当前自训练候选

主候选：

```text
artifacts/submission_axis_f2_m1_500_20260826.zip
```

已核实内容：

- ZIP 总大小约 14.49GB；
- 内含 `model_weights/model.safetensors`，大小 `14,467,165,872` bytes；
- 不含 `model_weights/model_fast.safetensors`；
- `config.json` 为 `pi05`、`chunk_size=10`、`n_action_steps=8`；
- `policy_server.py` 默认 `PI05_FAST=1`，但 FAST 文件缺失时会静默回退到 `SLOW from_pretrained`；
- 包内有本次 F2/M1@500 的 processors、normalizer、vendor 和 tokenizer 依赖；这些必须与权重一起保留。

该 checkpoint 包含参赛者自己的微调结果，优先级高于未训练的公开 Axis init。是否满足最终规则仍以官方要求和提交报告中的训练说明为准。

### 1.2 官方低分日志说明了什么

用户提供的官方 Track 1 日志显示：

```text
pi05 SLOW from_pretrained
fast=False
Remapped 812 state dict keys
Vision embedding key might need handling
6/6 episodes 均运行到 300 steps 后失败
```

因此官方 0 分有明确的提交/转换链风险，但不能把全部低分都归因于请求超时：server 实际返回了动作，episode 是因行为未成功而达到 300-step 上限。紧急恢复必须同时验证：

1. `/act` 单次调用低于 10 秒；
2. FAST 权重与源 checkpoint 行为一致；
3. Track 1 小面板至少出现真实成功。

只满足第 1 项不能称为恢复。

### 1.3 可借鉴但不可直接复用的旧资产

只读参考：

```text
<OLD_REPO>/workspace/scripts/export_pi05_fast_weights.py
<OLD_REPO>/workspace/artifacts/pi05_replay400_serve/model_fast.safetensors
```

旧 exporter 证明 FAST 路径可行，但存在两个需要修正的问题：

- 通过 `strict=False` 的 SLOW loader 建模，可能把 remap 缺失静默固化进 FAST 文件；
- 会原地改写源 `config.json`。

旧的 9.35GB `model_fast.safetensors` 来源不是当前 F2/M1@500 checkpoint，禁止复制到新包。它只能用于结构、dtype、key-count 和加载时延的只读参考。

## 2. 目标、非目标与成功定义

### 2.1 目标

在不训练的前提下生成一个新 Track 1 候选包，使其：

- 只从当前 F2/M1@500 的 `model.safetensors` 派生；
- 启动日志必定为 `FAST load model_fast.safetensors`；
- 不可能回退到 SLOW；
- vision patch embedding 和全部 state-dict keys 有显式门禁；
- 固定输入和本地 rollout 与源 Orbax/可信转换基线一致；
- `chunk=10, n_action_steps=8, action_ema=0.0, rtc=false`；
- `/act` cold/warm 均小于官方 10 秒限制；
- ZIP 小于 20GB、无嵌套压缩、无 symlink/hardlink；
- 通过 `validate_submission.py` 和官方 `parc2026` Docker E2E。

### 2.2 非目标

- 不执行 30k、3k、500-step 或任何新增 optimizer step；
- 不尝试修复 Track 3 逆向能力；
- 不重新选择 normalization；
- 不改变 action representation、图像预处理或 instruction canonicalization；
- 不用公开 Axis init 冒充自训练提交；
- 不自动消耗当天 Track 1 提交次数。

### 2.3 最终成功门禁

只有以下全部满足，才可把 ZIP 标为 `READY_FOR_HUMAN_SUBMISSION`：

1. 来源 ZIP、源权重、processors、config、converter 和最终 ZIP 均有 SHA-256；
2. FAST key audit 无未解释 missing/unexpected/shape mismatch；
3. vision patch embedding weight/bias 均存在、shape 正确且来源映射可追溯；
4. 固定输入 finite 率 100%，chunk shape `[10, 7]`；
5. FAST 与权威源在预声明数值阈值内，且 20-case paired Track 1 calibration 不显著退化；
6. policy 日志确认 `fast=True`、`n_action_steps=8`、`action_ema=0.0`、`rtc=False`；
7. server 启动、reset、连续 `/act`、episode reset 均无异常；
8. `/act` 最大延迟 `<10s`，并报告 cold/p50/p95/max；
9. Track 1 frozen mini panel 至少有一个真实成功，不能仅以 finite action 代替行为成功；
10. 最终 ZIP `<20GB` 且官方 Docker E2E 通过。

任一失败时输出 `NOT_READY`，不得为了赶时间删除门禁。

## 3. 文件所有权与输出位置

执行对话只拥有以下新文件或目录：

```text
tools/axis_pi05/export_fast_submission_weights.py
tools/axis_pi05/audit_fast_submission.py
tests/test_axis_pi05_fast_submission.py
configs/axis_pi05/urgent_fast_recovery.yaml
outputs/axis_fast_recovery/f2_m1_500_fast/
data/reports/axis_fast_recovery/f2_m1_500_fast/
artifacts/submission_axis_f2_m1_500_fast_<timestamp>.zip
```

可按最小需要修改：

```text
outputs/axis_fast_recovery/f2_m1_500_fast/staging/policy_server.py
```

禁止修改或覆盖：

```text
artifacts/submission_axis_f2_m1_500_20260826.zip
outputs/axis_pi05_v2/converted_f2_m1_500_lerobot/
<OLD_REPO>/
docs/AXIS_PI05_STANDARD_LIBERO_30K_RESEARCH_PLAN.md
现有 checkpoint、评测结果和 winner 文件
```

## 4. Phase U0：冻结源资产与可恢复工作区

1. 记录仓库 commit、dirty status，但不得因此清理其他 Agent 的修改。
2. 对源 ZIP 和其中关键成员生成 inventory：路径、uncompressed bytes、CRC、SHA-256。
3. 使用安全解压器拒绝绝对路径、`..`、symlink、hardlink 和嵌套 ZIP。
4. staging 必须是新目录；若目标存在则停止，不能覆盖。
5. 记录源候选：

```text
source_submission_zip
source_model_sha256
source_config_sha256
source_preprocessor_sha256
source_postprocessor_sha256
source_policy_server_sha256
```

产出：

```text
data/reports/axis_fast_recovery/f2_m1_500_fast/source_inventory.json
outputs/axis_fast_recovery/f2_m1_500_fast/staging/
```

停止条件：源 ZIP 缺成员、CRC 失败、路径不安全或源文件与记录不一致。

## 5. Phase U1：移植并加固 FAST exporter

以旧 exporter 的“建立 PI05Policy → remap → mixed dtype state dict → safetensors”逻辑为参考，在本仓库实现独立 exporter，但必须增加以下门禁。

### U1.1 不可变性

- 不原地修改源 ZIP、源 `model.safetensors` 或源 processors；
- `config.json` 的 CPU/float32 设置只写入 staging 副本；
- 输出先写 `.tmp`，完成 hash 和 reload 后原子 rename；
- export 失败时不得留下看似有效的 `model_fast.safetensors`。

### U1.2 key 与视觉塔审计

exporter 必须输出 source→target key mapping report，并对以下内容 fail closed：

- missing keys 非零；
- unexpected keys 非零；
- shape mismatch 非零；
- duplicate target key；
- 未消费 source key；
- patch embedding weight/bias 缺失；
- vision tower key 只触发 warning 而没有明确映射。

至少显式记录并验证：

```text
paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight
paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias
```

若实际 target 命名不同，报告必须同时写 source 和 target 全名，不能用模糊后缀通过。

### U1.3 dtype 与大小

保留已知 FAST mixed-dtype 策略：

- PaliGemma/action expert 主体使用 fp16 或官方验证过的 mixed dtype；
- `action_in_proj`、`action_out_proj`、`time_mlp_in`、`time_mlp_out` 保持 fp32；
- 输出记录 tensor count、各 dtype tensor/byte count、总 bytes；
- 参考旧 FAST 文件约 9.35GB，但不能把大小接近当作正确性证明。

产出：

```text
model_weights/model_fast.safetensors
model_weights/pi05_fast_meta.json
data/reports/axis_fast_recovery/f2_m1_500_fast/key_mapping.json
data/reports/axis_fast_recovery/f2_m1_500_fast/export_report.json
```

## 6. Phase U2：FAST 数值与行为等价

### U2.1 三路对照

对同一批 frozen fixtures、instruction 和 policy seed 比较：

1. F2/M1@500 原 Orbax checkpoint；
2. 当前 slow-converted LeRobot 模型，仅作诊断；
3. 新 FAST LeRobot 模型。

slow 路径出现 vision warning 时不得作为权威真值；Orbax 是参数行为的第一参考。

### U2.2 fixed-input 报告

至少 20 个 observation fixtures × 3 policy seeds，记录：

- 两张输入图像 hash、resize/rotation 后 hash；
- 8D state 和 normalized state；
- tokenized instruction hash；
- raw/internal action、7D denormalized chunk；
- finite、MAE、max error、cosine、gripper sign agreement；
- 每个候选的单次推理时间。

阈值必须在运行前写入 config，不能看完结果再放宽。紧急默认阈值：

```text
finite_rate = 1.0
median_abs_error <= 0.03
p95_abs_error <= 0.10
gripper_sign_agreement >= 0.95
```

若模型随机采样使逐值阈值不稳定，必须锁定 noise/policy seed；不能改用非成对均值掩盖差异。

### U2.3 20-case calibration rollout

使用既有冻结 Track 1 case，不新增针对官方日志任务的特判：

```text
model_action_horizon=10
executed_action_steps=8
replan_after=8
action_ema=0.0
rtc=false
max_episode_steps=300
```

报告 Orbax 与 FAST 的 paired success、collision、stall/timeout 和 action latency。FAST 相对 Orbax success 下降超过 `0.05` 时停止。

## 7. Phase U3：policy server 最小硬化

只在 staging 副本中修改 `MyPolicy`，不改官方禁止修改的 server/serialization 区域。

### U3.1 禁止 SLOW fallback

对于本紧急候选：

- `model_fast.safetensors` 不存在时立即退出；
- `PI05_FAST=0` 不得启用 SLOW；
- FAST load missing/unexpected 任一非零立即退出；
- 启动日志必须打印唯一标识 `FAST_REQUIRED_AND_VERIFIED`；
- 不再保留“找不到 FAST 就 from_pretrained”的生产路径。

### U3.2 固定执行协议

生产候选锁定：

```text
policy_type=pi05
chunk_size=10
n_action_steps=8
action_ema=0.0
rtc=false
compile_model=false
gradient_checkpointing=false
offline=true
```

环境变量不得在评测时无声改变这些值。启动日志和 package manifest 同时记录它们。

### U3.3 状态与 reset

- 每 8 个动作重新推理，chunk 的第 9、10 步不得进入下一个周期；
- episode reset 清空 policy action queue、EMA、instruction cache 和随机状态；
- gripper 不做 EMA；本候选 EMA 整体关闭；
- 输入 shape/dtype/finite 失败时明确报错，不返回伪造零动作。

### U3.4 latency

分别测量：

```text
process_start_to_health
first_act
warm_act_p50
warm_act_p95
warm_act_max
```

门禁：每个 `/act <10s`；目标 `warm p95 <5s`。启动时间按本地 validator 和官方 Docker 的实际限制报告，不以旧的 182.7 秒作为合格标准。

## 8. Phase U4：重新打包

最终 ZIP 只保留 FAST 权重，不同时装入 14.47GB SLOW 权重：

```text
policy_server.py
requirements.txt
model_weights/config.json
model_weights/model_fast.safetensors
model_weights/pi05_fast_meta.json
model_weights/policy_preprocessor.json
model_weights/policy_postprocessor.json
model_weights/*normalizer*.safetensors
vendor/
submission_manifest.json
```

最终包不得包含：

```text
model_weights/model.safetensors
任何 Orbax train state
另一个 zip/tar
symlink/hardlink
下载 cache
评测视频或日志
```

`submission_manifest.json` 至少包含：

- source F2/M1 checkpoint/ZIP hash；
- converter commit/hash；
- model_fast hash、bytes、key count、dtype summary；
- processors 和 config hash；
- policy server hash；
- action protocol；
- validation report hashes；
- `training_steps=500`、scope=`F2/M1`、norm=`M1`；
- `submission_status=NOT_SUBMITTED`。

压缩后检查：

- `<20GB`；
- 根目录层级正确；
- `unzip -t` 通过；
- `validate_submission.py` 通过；
- 解包后不存在 `model.safetensors`，且存在唯一 `model_fast.safetensors`。

## 9. Phase U5：评测与提交决策

### U5.1 本地顺序

1. 静态 validator；
2. CPU/import smoke；
3. 单 GPU startup、reset、20 次 `/act` latency；
4. fixed-input parity；
5. 20-case Track 1 calibration；
6. 若有时间，再跑现有完整 Track 1 frozen panel；
7. 生成 success/failure H.264 和 episode table，但视频不放入提交 ZIP。

### U5.2 官方 Docker

在比赛 GPU 的官方 `parc2026` 镜像中进行最终 E2E：

- Python 3.10；
- 单 GPU；
- 模拟 L4 24GB 显存边界；
- 外部网络关闭/离线；
- 从全新解压目录安装依赖并启动；
- health、reset、cold/warm act；
- 记录显存峰值、启动时间、act latency 和完整日志。

没有官方 Docker E2E，不得标记 ready。

### U5.3 人工提交门禁

执行者只输出：

```text
READY_FOR_HUMAN_SUBMISSION
或
NOT_READY
```

即使 ready，也不得调用 `omni submit`。负责人确认 ZIP hash、Track 1 当日剩余次数和报告后再提交。

如果官方再次 0/6：

- 日志仍为 SLOW：打包/路径问题，禁止继续提交；
- FAST 且 key/parity 失败：converter 问题；
- FAST、parity、latency 都通过但 300-step 失败：模型行为问题，不再把它称为 policy timeout；
- `/act >10s`：性能问题，优先分析显存、dtype、processor 和依赖，不修改动作语义。

## 10. 候选优先级与回退

### Primary：F2/M1@500 FAST

唯一默认生产候选：当前自训练 F2/M1@500 包派生出的 FAST-only ZIP。

### Diagnostic：Axis init FAST

只用于判断 converter/policy 是否能保持 Axis 能力。公开 Axis init 未包含参赛者自己的新训练，不得默认作为正式提交。

### Conditional fallback：旧 replay400 FAST

只有在以下全部证明后才可成为候选：

- 是本参赛者自己训练所得；
- 训练过程、数据、step、hash 和许可证可报告；
- 对应的完整 submission 资产可重建；
- 在当前官方 Docker 中重新通过；
- 没有把旧权重与当前 processors/norm 混用。

否则它只作结构和时延参考。

## 11. 时间盒

建议紧急执行顺序：

| 时间 | 工作 | 失败时 |
|---:|---|---|
| 0–30 min | 源 ZIP inventory、安全解包、hash | 停止 |
| 30–120 min | exporter 移植、key/vision audit、FAST 导出 | 停止，不用旧 FAST 顶替 |
| 120–180 min | policy FAST-only 硬化、static/server smoke | 修 server，不改模型 |
| 180–240 min | fixed-input parity、latency、20-case Track 1 | 行为为 0 则不提交 |
| 240–300 min | FAST-only ZIP、validator、manifest | 失败则不提交 |
| 之后 | 官方 Docker E2E、人工决定提交 | 不自动提交 |

如果到时间盒结束仍没有通过 key audit、parity 或本地真实成功，则今天不消耗正式提交次数；继续 30k 主线和 converter 修复，而不是把“能启动”误当成“有分数”。

## 12. 必须交付的报告

```text
data/reports/axis_fast_recovery/f2_m1_500_fast/source_inventory.json
data/reports/axis_fast_recovery/f2_m1_500_fast/key_mapping.json
data/reports/axis_fast_recovery/f2_m1_500_fast/export_report.json
data/reports/axis_fast_recovery/f2_m1_500_fast/fixed_input_parity.json
data/reports/axis_fast_recovery/f2_m1_500_fast/server_latency.json
data/reports/axis_fast_recovery/f2_m1_500_fast/track1_calibration/summary.json
data/reports/axis_fast_recovery/f2_m1_500_fast/package_validation.json
data/reports/axis_fast_recovery/f2_m1_500_fast/readiness_decision.json
artifacts/submission_axis_f2_m1_500_fast_<timestamp>.zip
artifacts/submission_axis_f2_m1_500_fast_<timestamp>.zip.sha256
```

最终回复必须明确：

- 实际使用的源 checkpoint；
- FAST 是否真正加载；
- missing/unexpected/vision key 结果；
- source→FAST parity；
- startup 与 `/act` latency；
- Track 1 本地真实成功数；
- ZIP 大小、hash、validator/Docker 结果；
- `READY_FOR_HUMAN_SUBMISSION` 或 `NOT_READY`；
- 明确说明没有自动提交。

## 13. 给新对话的启动指令

将下面内容直接交给新的执行对话：

```text
请完整执行：
docs/AXIS_PI05_URGENT_FAST_SUBMISSION_RECOVERY_PLAN.md

只修复已有 F2/M1@500 自训练 submission 的 FAST 转换、policy server 和打包链；不要训练，不要修改 30k 主线，不要复用旧仓库的 model_fast 权重，不要自动提交。你不是仓库中唯一的工作者，不得回退或覆盖其他人的修改。严格执行 key/vision、fixed-input parity、latency、Track 1 calibration、20GB、validator 和官方 Docker 门禁。最终只在证据完整时输出 READY_FOR_HUMAN_SUBMISSION，否则输出 NOT_READY。
```
