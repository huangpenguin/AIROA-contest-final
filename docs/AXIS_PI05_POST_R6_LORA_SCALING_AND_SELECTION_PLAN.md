# AXIS Pi0.5：R6 后 LoRA 长程训练、分层验证与候选选择执行计划

> 制定日期：2026-08-30  
> 基线提交：`f2056847d9c08e05e8d01808339ce7d9bf28b80d`  
> 当前输入状态：`lora_candidate_500_pending_review`  
> 本计划的判定：R6 step-500 是技术合格锚点，性能提升尚未得到统计确认。  
> 本文件只定义后续执行；落盘本文件时不得启动训练、rollout、导出、比赛 GPU 或提交。

## 1. 计划目的

本计划解决四个问题：

1. 把已经修复的 LoRA restore、fold、seeded evaluator 和 `QUANTILES` 链路固化为不可回退的门禁。
2. 用不重叠的分层 validation/holdout 扩大本地评价样本，避免只凭 B80 的 `+1/80` 选择模型。
3. 在两张 V100、effective global batch 64 的边界内，比较 action-expert LoRA 与 dual LoRA，并将胜者按 `1K → 3K → 10K → 30K` 推进。
4. 仅在 10 Hz 标准 LIBERO 候选通过最终 holdout 后，形成可供后续 20 Hz / Track 3 方案分叉的 checkpoint；本计划不合成 Track 3 数据。

本计划是资源受限下的 LoRA 工程路线，不取代
`docs/AXIS_PI05_STANDARD_LIBERO_30K_RESEARCH_PLAN.md` 中的 canonical full-FT 复现。full-FT 仍等待合适的 4×V100 或单卡 96 GB 资源。

## 执行状态（2026-08-30）

N0 审计固化已通过；N1.1 action trace 已落盘，但 N1.2 同 GPU deterministic replay 未通过。因此当前机器状态为：

```text
r6_evaluator_and_fold_reproducibility_failed
```

按本计划第 5.2 节和第 15 节，当前执行在 N1 停止。不得启动 N1.3 parity grid、N2 V160/H160、N3 长程训练及其后续阶段，直到 evaluator 的 action/outcome 可重复性问题被单独修复并重新通过 N1.2。详细结果见 `data/reports/axis_libero_30k/post_r6_lora/evaluator_reproducibility/summary.json` 和 `THREAD_REPORT_ZH.md`。

下一步只执行专门的定位计划 `docs/AXIS_PI05_N1_EVALUATOR_NONDETERMINISM_RECOVERY_PLAN.md`。该计划必须先验证 environment-only init/warm-up，再验证 frozen-observation policy runtime；不得直接放宽闭环 byte-exact 门禁。若最终只能得到有界 FP32 jitter，则必须停在统计评价协议的人工审核点，不能自动恢复 N1.3。

## 2. 本次审核结论

### 2.1 已接受的事实

- 根仓库 HEAD 为 `f2056847d9c08e05e8d01808339ce7d9bf28b80d`，审核时 root 与 nested OpenPI 工作树均 clean，`git diff --check` 通过。
- R6 `r6_dual_l2_fresh500_r1` 完成 500 个 optimizer update；checkpoint `0,1,10,25,50,100,250,500` 和 fresh restore 证据存在。
- evaluator 使用 `axis_pi05_lora_screen` 恢复完整 `L2_action_lora_heads` 拓扑，不再把 LoRA leaves 静默丢弃。
- step-500 raw LoRA 到 folded dense 共折叠 5 个模块；单卡 fixed-input internal/external 最大绝对误差均为 `9.467601776123047e-4`，在 `atol=rtol=1e-3` 内。
- OpenPI/JAX → folded PyTorch 的最大绝对误差为 `5.662441253662109e-7`；LeRobot pre/postprocessor 使用 R6 fresh norm 和逐维 `QUANTILES`，state/action/token/mask parity 通过。
- baseline 与 candidate 的 B80 CSV 已逐 case 复核；`official_task_index`、task、instruction、init index/hash、BDDL hash、env seed、policy seed 和 max steps 均为 0 mismatch。
- seeded B80 `execute8` 的正确数字是：

| 指标 | Axis step-0 | R6 step-500 | candidate - baseline |
|---|---:|---:|---:|
| success | 48/80 | 49/80 | +1/80 (`+0.0125`) |
| collision | 23 | 20 | -3 |
| timeout | 19 | 15 | -4 |
| stall | 0 | 0 | 0 |

- paired wins/losses/ties 为 `9/8/63`；episode bootstrap 95% CI 为 `[-0.0875, 0.1125]`。CI 跨过 0，不能声称统计显著提升。
- 项目测试的准确表述是 `.venv_axis_pi05/bin/pytest -q tests` 得到 `79 passed, 7 skipped`。直接对整个仓库执行 `pytest -q` 会收集 vendored OpenPI，并因当前环境缺少 `pynvml` 在 collection 阶段失败；不得把前者写成“包括 vendored OpenPI 的全仓测试”。

### 2.2 必须在继续训练前修复的审核项

1. `tools/axis_pi05/phase1_paired_report.py` 目前只比较 case-id 集合和 summary 级 norm/seed 来源，没有在生成报告时强制逐 case 的 init/seed/task/BDDL 字段相同。本次 CSV 实际相同，但工具仍需加入硬断言。
2. paired report 的 stall 统计必须与 rollout summary 使用同一判定函数，并反向核对 CSV 重算计数与 summary 计数相同。
3. episode reset 与 execute8 的“丢弃剩余两步、继续同一 RNG stream”必须在日志和协议中区分。日志应分别记录 `episode_reset(seed=...)` 与 `chunk_discard(no_reseed)`，禁止只打印上一次 seed 造成误解。
4. 当前 raw/folded parity 只有一个 fixed batch，而且误差接近 `1e-3` 门限。它足以接受本次链路修复，但不足以直接批准最终发布模型；长程 winner 必须做多输入、多 seed parity grid 和闭环 rollout 检查。
5. 旧计划中的 `paired delta lower bound -0.05` 是行为非劣点估计门槛，不是“bootstrap CI 下界”。新报告必须分别命名 `point_noninferiority` 与 `ci_noninferiority`，不得混写。

### 2.3 审核后的语义状态

保留历史机器状态 `lora_candidate_500_pending_review`，但后续报告应增加：

```text
r6_500_technical_anchor_accepted_performance_unproven
```

这意味着：不重新训练 R6、不删除 R6、不把 R6 自动打包为 winner；它是新实验的 restore/fold/evaluator 对照锚点。

## 3. 冻结输入与资源边界

以下内容在本计划所有 10 Hz run 中不可改变：

```text
dataset                physical-intelligence/libero
dataset revision       a4336d589d589045d1c56423ffdf3b88a0e19b1f
payload manifest       8ede620026a9196a9d453508dfe2d4820e684d98cda1a5bc706cef68d1583cc3
episodes / frames      1693 / 273465
tasks                   40
fps                     native 10 Hz
resampling              none
Axis init revision      c342608c489515efa9ba8281400dc648bf3970e1
norm path               data/reports/axis_libero_30k/phase1/norm_stats.json
norm sha256             4c803bf59c52677c36b6b420c614d8a7742e0847708443863e13e67cb80b18f5
normalization           ACTION=QUANTILES, STATE=QUANTILES
external/internal act   7 / 32
action horizon          10
primary protocol        execute8
precision               float32; V100 禁止 BF16
effective global batch  64
EMA                     false for LoRA
objective               valid7_temporal
```

资源边界：

- 默认只使用经检查为空闲的 GPU1、GPU2；不得触碰带外部 PID 的 GPU，也不得因 GPU1/2 不可用自动改用 GPU0。
- 每卡 microbatch 1，gradient accumulation 32，两卡 data-parallel，effective global batch 精确为 64。
- 启动任何 GPU 阶段前记录 `nvidia-smi`、PID、显存、磁盘和 git provenance；结束后再次记录。
- 审核时 `/home` 约剩余 914 GB，R6 raw checkpoints 约 95 GB、folded step-500 约 14 GB。每个新阶段启动前要求剩余空间不少于 500 GB；低于门槛立即停止，不自动删除数据或旧 checkpoint。
- 新产物只能写入：

```text
outputs/axis_pi05/axis_libero_lora_long_horizon/
data/reports/axis_libero_30k/post_r6_lora/
```

不得覆盖 R6、旧 B80、旧 norm 或旧 fast-recovery 产物。

## 4. 阶段 N0：无 GPU 的审计固化

### N0.1 起点门禁

执行前保存：

```text
git rev-parse HEAD
git status --porcelain=v1
git -C third_party/axis_openpi rev-parse HEAD
git -C third_party/axis_openpi status --porcelain=v1
git diff --check
git -C third_party/axis_openpi diff --check
df -h /home/huang/code/PARC2026_final
```

`f205684...` 必须是执行分支的祖先，且本计划应先作为独立 docs-only commit 落在其后；进入 N0 实作和任何 GPU run 前，两个工作树必须 clean。后续代码修复必须形成新的 clean commit；不得 amend 或改写 `f205684`。

### N0.2 报告器硬化

修改 paired report，使 candidate/baseline 在每个 case 上强制一致：

```text
case_id
official_task_index
task_name
language
bddl_sha256
init_state_index
init_state_sha256
env_seed
policy_seed
max_steps
execution protocol
```

同时要求：

- candidate/baseline 各自 CSV 重算的 success/collision/timeout/stall 与各自 summary 完全相同；
- case 数、唯一性、manifest coverage、视频存在性均为硬门禁；
- 输出 `pairing_invariants`、`mismatch_count`、两个 CSV hash 和 manifest hash；
- 统计同时输出 episode bootstrap、按 task 聚类 bootstrap、McNemar exact test；
- 报告字段明确区分 point estimate、CI 和选择规则。

### N0.3 seed/reset 协议硬化

- episode 开始必须传入整数 `policy_seed`，重置 JAX RNG；缺失 seed 直接失败。
- execute8 到第 8 步后只能清空服务端未执行 action，不得重新设置 RNG，不得被记录成新的 episode reset。
- 日志写出 `reset_kind`、episode seed、chunk index 和是否 reseed。
- 不依赖“当前 OpenPI 的 `Policy.reset()` 恰好是 no-op”作为隐含契约；用测试锁住行为。

### N0.4 测试与提交

至少新增：

- 一项逐 case seed/init mismatch 必须失败的测试；
- 一项 summary/CSV count mismatch 必须失败的测试；
- 一项 episode reseed 与 chunk-discard 不 reseed 的测试；
- 一项 reversed collision/timeout delta 方向测试。

执行：

```text
.venv_axis_pi05/bin/python -m py_compile tools/axis_pi05/*.py
.venv_axis_pi05/bin/pytest -q tests
git diff --check
```

全部通过后形成一个独立 clean commit，并写：

```text
data/reports/axis_libero_30k/post_r6_lora/N0_AUDIT_LOCK.json
data/reports/axis_libero_30k/post_r6_lora/N0_AUDIT_REPORT_ZH.md
```

N0 未通过，不得进入任何 GPU 阶段。

## 5. 阶段 N1：evaluator 可重复性与 fold 扩展门禁

### N1.1 action trace

rollout 必须为每个 episode 保存 float32 action trace 或无损等价产物，并在 CSV 中记录：

```text
action_trace_path
action_trace_sha256
action_shape
action_finite
```

失败视频、task、instruction、init、env/policy seed、扰动、failure reason 和 action trace 必须能在同一 gallery card 中关联。

### N1.2 同 GPU deterministic replay

在同一张空闲 V100 上分别对 Axis step-0 和 R6 step-500 执行两次相同 C20：

- 20/20 case coverage；
- 每个 case 的 outcome、steps、failure reason 相同；
- 相同 case 的 action shape 相同且 float32 bytes hash 相同；
- episode seed 只设置一次，后续 execute8 chunk discard 不 reseed；
- stall=0，所有 action finite。

如果 action bytes 不完全相同，停止并报告第一处 action/index/max-abs mismatch；不得通过放宽阈值自动继续。

### N1.3 parity grid

对 R6 step-500 构造至少 `8 个 observation/prompt × 4 个 noise/RNG seed` 的 raw-LoRA/folded-dense grid：

- 每格 internal/external action 全 finite；
- 全部格点最大绝对误差 `<=1e-3`；
- 输出 max、p99、median、按 action dim 的误差；
- 另做 raw-LoRA 与 folded-dense 的同 seed C20 闭环诊断，单独报告 outcome/steps 分歧，不用 fixed-forward parity 冒充闭环等价。

N1 通过状态：

```text
r6_evaluator_and_fold_reproducibility_locked
```

## 6. 阶段 N2：冻结更大的本地 validation/holdout

每个官方 task 有 50 个 init states。现有 B80 使用 index `0,1`；新增面板使用不重叠 init：

| 面板 | 组成 | init index | 用途 |
|---|---:|---|---|
| `V160` | 40 tasks × 4 | `2,3,4,5` | checkpoint/配方选择，可重复使用 |
| `H160` | 40 tasks × 4 | `6,7,8,9` | 最终一次性泛化确认，不参与 winner 反复调参 |

要求：

- manifest 固定 BDDL/init-state 内容 hash、task index/name/language、env seed、policy seed、max steps；
- V160、H160、B80 三者 init-state `(task,index)` 零重叠；
- seed namespace 互不重叠；
- task/suite mapping 必须来自官方 benchmark metadata，不从 task 字符串猜测；
- 生成 manifest hash、coverage report、重叠检查和只读 lock；
- 在 N2 只运行 Axis step-0 与 R6 step-500 的 `V160 execute8`。H160 此时只冻结，不评价任何训练候选。

V160 报告必须包含 overall、per-task、官方 suite 分层、collision、timeout、stall、steps、paired wins/losses/ties、task-cluster bootstrap CI 和 failure gallery。

R6 在 V160 的结果只是扩大锚点评价，不改变其训练 schedule，也不自动批准。

## 7. 阶段 N3：为长程 run 修复训练日程语义

R6 的 cosine decay horizon 是 500 steps。不得直接把 R6 step-500 改成 1K/30K 配置继续训练，因为这会在 resume 后改变同一个 optimizer step 对应的 LR，形成未声明的 schedule jump。

长程实验必须：

- 从相同 Axis released params + exact zero-delta LoRA fresh 初始化；
- 将 `schedule_total_steps=30000` 与本次 `stop_at_step` 分离；
- 即使首段只跑到 100 或 1000，LR 仍按从第 0 步就锁定的 30K schedule 计算；
- resume 保留 optimizer state、global optimizer step、data cursor/RNG 和 schedule phase；
- 1000→3000→10000→30000 只改变 stop point，不改变 schedule、data、norm、batch、precision、objective 或 trainable scope。

LoRA 统一 schedule：

```text
warmup optimizer steps   100
cosine horizon           30000
decay LR                 1e-7
L2 peak LR               1e-6
L3 peak LR               3e-7
AdamW                    b1=.9, b2=.95, eps=1e-8, wd=1e-10
gradient clip            0.1
```

这是一条 LoRA 工程配方，不得标成原 AXIS full-FT 配方；原 full-FT 的 `5e-5 / warmup 10K / EMA` 不可混入。

先用固定 fixture 验证 uninterrupted 0→100 与 0→50 + restore→100：

- sample IDs/RNG trace 完全一致；
- optimizer step 与 LR 完全一致；
- trainable parameter leaves 一致；
- 参数 max-abs diff 在既定 FP32 resume tolerance 内；
- checkpoint finalized、restore 和 optimizer state 通过。

同时增加只读诊断，不改变 loss：

- normalized target 与 prediction 的 per-dim loss/variance；
- q01/q99 外 clipped fraction；
- 每个 action dim 的 gradient contribution；
- gripper/rotation 诊断。

不得因为 raw action range 不同而在主臂加入 rotation weight。当前主臂保持逐维 `QUANTILES + valid7_temporal`；任何 action weighting 必须以后作为独立消融。

## 8. 阶段 N4：1K LoRA topology screen

两个 run 顺序执行，每个 run 都使用两张卡和相同 training seed/data order：

| Run ID | Trainable | rank | peak LR | sampling | schedule horizon | 首次 stop |
|---|---|---|---:|---|---:|---:|
| `lh30_l2_native_s0` | action expert LoRA + action in/out + time MLP | action 32 | `1e-6` | native canonical | 30000 | 1000 |
| `lh30_l3_native_s0` | 上述 + PaliGemma LoRA；vision frozen | action 32, PaliGemma 16 | `3e-7` | native canonical | 30000 | 1000 |

执行门禁：

1. 每个 run 先停在 step 1/10/100 做 finite、scope、restore 和 C20 canary。
2. 任一 non-finite、错误 trainable leaf、norm/hash 变化、resume mismatch 立即停止该 arm；不得自动降 LR 重试。
3. 通过后继续到 1000；保存 `0,1,10,100,500,1000`，且 100/500/1000 fresh restore 通过。
4. step-1000 对 V160 运行完整 seeded `execute8`；full-chunk 只作协议诊断，不参与排名。

相对同一 Axis step-0 V160 baseline，arm 的最低晋级门槛：

```text
complete cases                  160/160
stall                           0
point success delta             >= -0.05
task-cluster CI lower bound     >= -0.05
任一官方 suite success delta    >= -0.10
collision/timeout 各自增量      <= +0.05
```

两 arm 都通过时，先按 V160 overall success 排序；差距不超过 `2/160` 时依次用 suite worst-delta、collision、timeout、trainable parameter count 决胜。统计不确定时只写 `provisional_rank`，不写“显著优于”。

两 arm 都失败则停止：

```text
long_horizon_lora_topology_screen_failed
```

不得临时增加新的 rank/LR 搜索。

## 9. 阶段 N5：分层 sampling 消融

仅对 N4 的 provisional winner 运行一个新的 fresh step-1000 对照。除 sampling 外所有 seed、topology、schedule、norm、batch 和 optimizer 完全相同。

分层 sampler 固定为：

1. 官方 suite 等概率；
2. suite 内 task 等概率；
3. task 内 episode 等概率；
4. episode 内合法 temporal anchor 等概率；
5. padding/mask 与 native arm 完全一致。

输出每 100 optimizer steps 的实际 suite/task/episode/frame coverage、重复率和最大/最小采样比。不得用 IPEC 或 LIBERO-plus 替代 canonical source；norm 仍使用同一份 full-corpus fresh `QUANTILES`，以便只隔离 sampling 变量。

只有当 balanced arm 同时满足以下条件才取代 native：

- V160 success delta 相对 native 至少 `+0.025`（4/160）；
- task-cluster CI 不显示超过 `-0.05` 的退化；
- 任一 suite 不退化超过 `0.05`；
- stall=0，collision/timeout 不恶化超过 `0.05`。

否则保留 native，结论写成“没有足够证据更换 sampler”，不得为追求点估计反复改 strata。

## 10. 阶段 N6：训练 seed 复核与 30K 分段推进

### N6.1 seed 复核

用选定 topology/sampling 配方，从 Axis init 另起第二个 training seed 跑到 1000。第二 seed 只用于估计训练方差，不得因为它分数更高就替代预先指定的主 seed。

两个 seed 均须通过数值门禁，V160 overall 差异不得大于 `0.05`，任一 suite 差异不得大于 `0.10`。否则状态为：

```text
lora_recipe_training_seed_unstable
```

停止长程推进并提交人工审核。

### N6.2 主 seed 分段 resume

主 seed 按同一 30K schedule resume：

```text
1000 → 3000 → 10000 → 30000
```

每个 milestone：

- fresh restore、optimizer/schedule/data-cursor continuity、finite/scope 检查；
- C20 canary；
- V160 seeded execute8；
- per-dim normalized diagnostics；
- checkpoint/norm/git/data hashes；
- failure H.264 gallery 与 task/suite 报告。

提前停止规则：

- 任一 non-finite、restore/scope/hash/seed 错误立即停止；
- V160 相对 Axis baseline 的 task-cluster CI 下界小于 `-0.05`，停止；
- 连续两个 milestone 相对该 run 历史最佳 point success 下降超过 `0.05`，停止；
- stall 非零或 collision/timeout 任一相对 baseline 增加超过 `0.10`，停止人工审核；
- 只有“尚未显著提升”但保持非劣，不是自动停止理由，因为原研究目标本来就是长程训练。

不得从某个 milestone 新建不同 LR/schedule 的伪 resume；需要改配方时必须从 Axis init 新建独立 ablation。

## 11. 阶段 N7：checkpoint 选择与一次性 H160

用 V160 在以下 checkpoint 中预先选择一个 winner：

```text
R6 step-500 anchor
long-horizon step-1000
long-horizon step-3000
long-horizon step-10000
long-horizon step-30000
```

选择顺序：技术门禁 → V160 overall → worst-suite delta → collision → timeout → 较早 checkpoint。选择规则必须在读取 H160 candidate 结果前完成并落盘。

之后只对以下两个模型运行 H160：

1. Axis step-0 baseline；
2. V160 预先选出的唯一 winner。

不得因 H160 结果不好再在同一 H160 上轮换 runner-up。H160 最终通过门槛：

```text
coverage                         160/160
stall                            0
point success delta              >= 0
task-cluster noninferiority CI   lower >= -0.05
任一 suite delta                 >= -0.10
collision/timeout 增量            each <= +0.05
```

CI 跨 0 时允许称为“本地非劣候选”，不允许称为“统计显著提升”。只有 CI 下界大于 0 才允许使用“本地显著提升”。

H160 未通过时状态为：

```text
lora_long_horizon_holdout_failed_no_fallback_selection
```

## 12. 阶段 N8：最终转换、FAST 包装前门禁

仅对 H160 通过的唯一 winner：

1. 恢复 raw LoRA 完整拓扑并生成 folded dense；
2. 重做至少 `8×4` parity grid，最大绝对误差 `<=1e-3`；
3. 同 seed C20 raw/folded 闭环比较，报告所有 outcome/steps/action 分歧；
4. OpenPI/JAX → folded PyTorch → LeRobot processor E2E parity，ACTION/STATE 必须为 `QUANTILES`，norm hash 必须是 R6 fresh norm；
5. 严格检查 missing/unexpected keys，特别是 vision patch embedding；
6. 生成 `model_fast.safetensors`，记录 sha256；禁止只打包 `model.safetensors` 后走 SLOW remap；
7. 在本地比赛兼容 Docker 做 policy server E2E，日志必须出现 `fast=True`，不得出现 SLOW fallback 或未处理 vision key；
8. 输出固定 canary 的非零 action、无 timeout/stall 证据以及 submission manifest。

本计划到此停止，不上传官方评分、不启动 20 Hz/Track 3 训练。通过状态为：

```text
axis_pi05_lora_10hz_candidate_ready_for_track3_fork
```

后续 Track 3 必须从完成的 10 Hz checkpoint 分叉，使用独立 20 Hz 官方/合成数据、独立 sampling manifest 和重新计算的 20 Hz norm；禁止将 10 Hz 数据插帧冒充 20 Hz。

## 13. Checkpoint 与磁盘策略

每个长程 run 只计划保留：

```text
0, 1, 10, 100, 500, 1000, 3000, 10000, 30000
```

- optimizer full-state 至少保留 `1000,3000,10000,30000`；其余是否降为 params-only 必须先生成 manifest/hash 并经人工确认。
- 不得自动删除 R6、Axis init、canonical data、V/H manifests、winner 或最后一个可恢复 optimizer checkpoint。
- 任意清理必须先生成 `retention_plan.json`，列出精确路径、大小、hash、保留理由和待删除理由；未经人工批准不执行删除。

## 14. 必需产物

```text
configs/axis_pi05/post_r6_lora_long_horizon.yaml
data/reports/axis_libero_30k/post_r6_lora/
  N0_AUDIT_LOCK.json
  N0_AUDIT_REPORT_ZH.md
  manifests/V160.jsonl
  manifests/H160.jsonl
  manifest_audit.json
  evaluator_reproducibility/summary.json
  parity_grid/summary.json
  topology_screen/summary.json
  sampling_ablation/summary.json
  seed_replication/summary.json
  milestones/{1000,3000,10000,30000}/...
  final_selection/selection_before_holdout.json
  final_selection/H160_summary.json
  final_export/e2e_action_parity_report.json
  final_export/submission_manifest.json
  THREAD_REPORT_ZH.md
outputs/axis_pi05/axis_libero_lora_long_horizon/
  <arm_and_sampling>/<run_id>/<checkpoint>/...
```

所有 summary 至少包含：root/nested commit、git status、dataset/model/norm/manifest hashes、GPU IDs/PIDs、trainable scope、precision、batch/accumulation、optimizer/schedule、checkpoint identity、policy seed contract、case coverage、视频数量和 `gpu0_touched`。

## 15. 全局停止条件

出现以下任一情况立即停止当前阶段，不自动改超参数、不跳样本、不伪造后续结果：

- root 或 nested worktree 在 GPU run 前不 clean；
- 输入 revision/hash/norm/manifest 变化；
- 使用 BF16、global batch 不为 64、未声明的 LR/schedule jump；
- non-finite loss/gradient/update/parameter/action；
- trainable scope 与 arm 不一致或 vision 被意外解冻；
- checkpoint 未 finalize、fresh restore 失败或 optimizer step 不一致；
- paired case seed/init/BDDL/task 不一致；
- evaluator reseed/trace 不可复现；
- GPU0 或任意外部 PID 所在 GPU 被触碰；
- 剩余磁盘低于 500 GB；
- H160 被用于反复选择；
- FAST 包缺失 `model_fast.safetensors`、走 SLOW fallback 或出现 vision key 警告。

执行 Agent 每完成一个 phase 必须更新 `THREAD_REPORT_ZH.md` 和机器 summary，然后按本计划门禁决定继续或停止；不得把“代码跑通”“本地非劣”“统计显著提升”“官方比赛得分”四种结论混为一谈。
