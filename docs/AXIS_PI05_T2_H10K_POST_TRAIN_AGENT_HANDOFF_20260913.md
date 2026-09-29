# T2 H10K 训后移交：官方机剩余用途 + 下一任必须审核的执行合同

> 落盘日期：2026-09-13（Asia/Tokyo）约 16:45
> 读者：训练结束后接手的 agent / 审核者
> 性质：执行合同与审核清单。不是 formal audit 报告，也不改 mix/norm/LR/rank/batch。
> 用户原话：训练可能明天结束，官方服务器之后只剩十多个小时；把本对话所有「执行前需要审核」的内容落盘。
> 机器 JSON：`data/reports/axis_libero_30k/t2_p30_r6_h10k_r1_post_train_handoff_20260913/summary.json`

先读完本文件，再读：

1. `docs/AXIS_PI05_T2_H10K_TOMORROW_PERSIST_FOLD_PLAYBOOK_20260914.md`（明早只读检查 / rsync / fold / Stop 的可执行命令）
2. `docs/AXIS_PI05_FINAL_4DAY_TRAINING_EXECUTION_PLAN_20260912.md`（配方、horizon、门）
3. `docs/AXIS_PI05_OFFICIAL_BLACKWELL_T2_H10K_LAUNCH_ISSUES_20260913.md`（启动失败分类）
4. `MEMORY.md` §26–§28
5. 现场证据：官方 PID、log、checkpoint、GPU；本地 `git status`；本机 GPU0/GPU2 YOLO

**禁止从聊天历史重建状态。** 本文件和官方 live 产物优先。

---

## 0. 一句话

官方 Blackwell 在训练结束后只还需要做三件不可替代的事：**把 Orbax checkpoint 推进 persist、在 JAX 0.11 栈上 fold 成可带走的权重、确认 Stop 后还能恢复。** C20、FAST ZIP、比赛提交都可以离开官方卡。十多个小时不够开 T1/T3/dense。Agent 不自动上传或提交。

---

## 1. 当前冻结事实（接手时先核对，不要改）

```text
run id                 t2_p30_r6_h10k_r1
status                 用户 2026-09-13 ~16:42 JST 确认已成功启动；~17:10 JST 现场 log 为 pid 6950、optimizer_step≈1684、GPU 33439 MiB / 100%
valid_action_elements  640 = 16 micro × 4 × horizon 10（上游 compute_loss 的 (*b,ah).size）
                       不是 Axis valid7 的 64×10×7=4480。恒定 640 不要当成 loader 崩溃，也不要因此停训。
init                   R6 step500 params-only + fresh optimizer
horizon                schedule 10000 / stop 10000（已冻结，不得改成 30k 或 H3K）
batch                  global 64 = microbatch 4 × acc 16
LR                     peak 3e-7, warmup 10, decay 3e-8
topology               L2_action_lora_heads
precision              FP32
T2 manifest SHA        85700f9d85311cdadbacadcecf2b7c9b6617b77e03f51de42a81681ffc770af4
T2 norm SHA            39ff8a7cc2ff2a0be7d43659e708f9ec260686e876cf603c0460972b49746b41
CODE                   /opt/dlami/nvme/parc20260911/code/PARC2026_final
run dir                /opt/dlami/nvme/parc20260911/runs/t2_p30_r6_h10k_r1
log                    /opt/dlami/nvme/parc20260911/runs/t2_p30_r6_h10k_r1.train.log
persist                ~/data/parc20260911
截止                   2026-09-17 23:59 JST（用户 2026-09-12 确认；接手时再看官方页）
```

禁止：

- 第二份 `train.py` / 第二份同名 run
- resume canary100 `train_state`
- 改 mix / norm / LR / rank / batch / EMA
- 30k、T1/T2/T3 merge、dense full-FT
- Stop My Server（在 data-push 完成并核对 persist 之前）
- Restart and run all
- 碰本机 GPU0 / GPU2（外部 YOLO）
- `HfApi.create_tag`
- 自动比赛上传/提交
- 把失败 wandb `971ay3vq` / `etxxxaa3` 当成正式 run

官方机：不能 SSH；格子在用户 Mac 浏览器 Jupyter。留标签防 idle。到期可延長，不要为「清盘」而 Stop。

---

## 2. 训练结束后，还要不要官方服务器？

用户假设：**明天训练完，官方机可能只剩十多个小时。** 按这个预算排，不要再开新训练 identity。

### 2.1 必须留在官方 Blackwell 的（JAX 0.11 / NVMe / persist）

| 优先级 | 事项 | 为什么不能改到本机 | 粗估 |
|---|---|---|---|
| P0 | 确认最新 checkpoint 完整：`params/`、`train_state/`、`_CHECKPOINT_METADATA`、assets/norm | 活动 Orbax 只在 NVMe；Stop 会清 NVMe | 15–30 min |
| P0 | `parc-home-sync data-push` 至少推：1000、3000、10000（若已写出）以及当时最新一步 | persist 才过得了 Stop / 休眠 | 30–90 min，视体积 |
| P0 | 在 **官方 trainvenv + JAX 0.11.1** 上把 LoRA params **fold 成 PyTorch/safetensors** | 本机 V100 是 jax 0.5.3 / orbax 0.11.13，读官方 0.11.25 checkpoint 属于版本墙，不是配方问题 | 1–3 h |
| P0 | 把 folded `model.safetensors` + norm + 身份 JSON 拷进 persist 或 HF，记录 bytes/SHA | 没有这份，Stop 后本机评测/ZIP 没有输入 | 30–60 min |
| P1 | 官方机上对 `train_state` 做一次 fresh restore smoke（resume 同一 run_id，不改 horizon） | 证明 persist 可恢复；本机旧栈不能代替 | 30–60 min |
| P2 | 若 fold 后本机仍无法加载：在官方跑最小 policy smoke（不是完整 C20） | 官方只有 1 卡，完整 C20 会吃光剩余配额 | ≤1 h |

P0 做完且 persist 抽查 hash 通过后，**可以 Stop My Server**。之后官方卡不是评测主场。

### 2.2 明确不要再占用官方卡的

| 事项 | 改到哪里 | 原因 |
|---|---|---|
| 完整 C20 / T2 行为评测 | 本机 GPU3 | canary100 已走这条；fold 后的 PyTorch/FAST 在本机评 |
| FAST ZIP、CRC、布局、大小 | 本机 | 现有 `export_*` / audit 工具；不要在官方现装 robosuite/libero |
| 比赛上传/提交 | 用户 Mac 浏览器 | 用户提交；agent false |
| Fast400 / Teleop / reverse gold | 本机 GPU3 | 与 T2 官方训练并行合同未改 |
| T1 H3K / T3 H300 / dense | **本届不要做** | 十多小时装不下完整 cosine + fold + ZIP + 提交；T1 还要求 T2@3000 比赛分 > 0.33 |
| 再开一条 T2 或把 H10K 改 30k | 禁止 | horizon 已冻 |

### 2.3 十多小时官方配额的推荐顺序

```text
T+0:00   读 log：alive、last step、最新 KEEP 目录是否 1000/3000/10000
T+0:30   未在训则 data-push；仍在训则等当前 KEEP 写完再 push，不要杀进程
T+1:30   fold params → safetensors（官方 trainvenv）
T+4:00   persist 抽查：Orbax tree + safetensors SHA
T+5:00   可选 restore smoke
T+6:00   官方可停。本机 GPU3 开 C20 / FAST / ZIP
剩余     只留给 push 失败重试或 fold 失败，不开新训
```

若训练在 10 小时配额结束前还没到 10000：优先 **push 当时最新完整 step**（3000 或 1000），fold 那一步，不要为了「再挤 2000 step」拖到没时间 fold。候选规则仍是：可归属比赛分 > 0.33 才替换 R6；否则 T2 用 R6 0.33。

### 2.4 官方机死了怎么恢复

1. 不要 `--mode new`。
2. 查 persist 是否有完整 `train_state`。有则同一 `run_id` `--mode resume`，horizon 仍 10000。
3. persist 只有 `params` 没有 `train_state`：不能冒充 resume；那是新 optimizer，等于新 identity，本窗口默认不做。
4. 昨晚 pid 210633 的教训：没 data-push = checkpoint 随 NVMe 消失。

---

## 3. 本对话里所有「执行前必须审核」的内容

下一任在做任何 mutation（第二训、评测启动、fold、ZIP、Stop、commit、上传）之前，逐项给出 `approved` / `blocked` / `not_evaluated`。不要用聊天里的补丁记忆代替现场文件。

### 3.1 身份与配方（已冻，审核只做核对）

- [ ] 现场 log 的 run id 仍是 `t2_p30_r6_h10k_r1`
- [ ] 不是 canary100，params 路径含 `r6_dual_l2_fresh500_r1` 且不是 `train_state` 当 init
- [ ] manifest / norm SHA 仍是上表两个冻结值
- [ ] 没有人改过 LR/batch/rank/EMA/horizon

**未审核不得**：把 H10K 改 30k、从 canary100 续、开 T1/T3 训。

### 3.2 官方 live 树 ≠ 本机 git（拼树，必须现场 diff）

本机 nested OpenPI 有 `TrainState.rng`；官方曾用公开 clone HEAD 补文件，overlay 保住 `train.py`/`data_loader.py`/`config.py`。今日为启动在官方 **现场改过、多半未回灌本机 git** 的内容：

| 位置 | 改了什么 | 审核要点 |
|---|---|---|
| `.../openpi/training/utils.py` | 给 `TrainState` 补 `rng` | 与 overlay `train.py` 是否仍匹配 |
| `.../openpi/training/data_loader.py` | 从 `.bak_ipec_hub` 恢复后再给 `AxisPlusV3Dataset` 加 `episodes` / `episode_data_index` | `ast.parse`；IPEC 走 v3；不要 Hub `LeRobotDatasetMetadata` |
| `data/manifests_track12/T2-P30/source_roots.json` | 6 个 root remap 到 NVMe `dataset/` | 必须是 `/opt/dlami/nvme/parc20260911/dataset/...` 且 `file-*.parquet` |
| `tools/axis_pi05/run_track12_long_horizon_lora.py` | 可能已加 Linux→NVMe remap | 若 overlay 再解一次会丢 |
| Flax site-packages | `jax.extend.core` tracer；去掉 `remat(concrete=)` | 在 trainvenv，不在 git |
| `work/trainvenv_python.sh` | 避免 uv symlink `.resolve()` | 训/fold 都必须走 wrapper |
| Axis 模块 | `axis_lora.py` / `nan_replay.py` / `axis_pi05_contract.py` 写进官方 tree | 缺了会 import 失败 |

**未审核不得**：用 Q7 payload 覆盖 `train.py`/`data_loader.py`/`config.py`；`pip install -e third_party/axis_openpi`；往 trainvenv 装 robosuite/libero；把本机 dirty tree 直接 rsync 盖掉官方 live 补丁。

本机 git 此刻很脏（大量未跟踪 manifests/tools/docs）。**用户未要求则不要 commit。** 官方补丁回灌本机是赛后/ZIP 边界才做的审核项，不是训后第一小时。

### 3.3 数据布局（类 C，不是 pip）

- [ ] 官方 IPEC 五套 + `lerobot/libero_plus` 仍是 v3 `chunk-*/file-*.parquet`
- [ ] Plus 目录名是 `dataset/lerobot/libero_plus` 不是 `libero_plus_lerobot`
- [ ] 视频键：IPEC `image`/`wrist_image`，Plus `front`/`wrist`
- [ ] 不要 `create_tag`

fold/评测若再走 OpenPI loader，仍必须 `AxisPlusV3Dataset`。ZIP 路径应只吃 folded 权重 + 冻结 T2 norm，不再读 IPEC tar。

### 3.4 操作与碰撞

- [ ] `pgrep -af scripts/train.py`；有活进程则只观察
- [ ] 读正式 log，不读 probe
- [ ] 失败残留已归档：`runs/t2_p30_r6_h10k_r1.failed_*` 只读
- [ ] GPU 空才启动 fold/评测；fold 与 train 不能并行（1 卡）
- [ ] 空 `run_dir` 不要 `mkdir` 后 `--mode new`
- [ ] Jupyter 不要 Restart and run all

### 3.5 启动失败分类（写 ZIP/赛后报告前要保持）

权威：`docs/AXIS_PI05_OFFICIAL_BLACKWELL_T2_H10K_LAUNCH_ISSUES_20260913.md`

不要把所有炸点写成版本问题：

1. JAX 0.11 墙（Orbax layout、Flax tracer）
2. 缺 tokenizer / gcsfs / wrapper
3. v3 parquet vs v2.1 loader + Linux `source_roots`
4. 空目录、粘贴、休眠无 push
5. overlay vs clone HEAD（`TrainState.rng`）

失败 pid：5441 Hub/IPEC → 5823 缩进 → 6030 无 `episodes` → 6609 `rng`。成功启动后的 step 以 log 为准，本文不编造。

### 3.6 训后评测 / ZIP 门（四天计划 §6–§11）

在 **fold 产物 SHA 落盘之前**，不得宣称有新 T2 候选。

step 与门（执行前再读现场 checkpoint，缺的标 `not_evaluated`）：

| 产物 | 计划要求 | 十多小时官方配额下 |
|---|---|---|
| step1/10 | finite / restore | 若已过去，只记录，不重跑 |
| step100 | 安全点，不打比赛 ZIP | 同左 |
| step1000 | 有限动作/C20 smoke；C20 掉 >2/20 才停 | C20 改本机 GPU3 |
| step3000 | 第一个外部分数点：C20 + fold/FAST + ZIP + **用户提交** | fold 在官方；C20/ZIP 在本机；提交等用户 |
| step10000 | cosine 终点，同一部署链，不再续 30k | 同 3000；时间不够就用已 push 的最高完整 step |

候选规则（审核不得放宽）：

- 比赛分 **严格高于** T2 R6 0.33 → 新 winner
- ≤ 0.33 → 最终 T2 用 R6
- 回执无法归属 → 不覆盖 R6
- T1 仍用 R6 0.47，除非另有已审核的 T1 新 ZIP（本窗口默认没有）
- T3 无新 accepted gold 则不新训；旧 T3 canary/ZIP 不是正收益

### 3.7 本机 vs 官方评测栈

| | 官方 Blackwell | 本机 |
|---|---|---|
| 训练 JAX | 0.11.1 cuda13，必须 | 0.5.3，不能当官方 restore 真源 |
| 评测 | 仅最小 smoke | GPU3 C20 / reverse；一次一个 simulator |
| 提交包 | 不在官方现装 LIBERO | FAST-only ZIP，已有 R6 机械门 |

**未审核不得**：在官方 trainvenv 装评测依赖；用本机 jax 0.5.3 直接 `restore` 官方 orbax 当「已验证」。

### 3.8 用户权限（agent 不得自行升级）

必须再问用户 / 保持 false：

- 比赛上传、`omni submit`、覆盖当前 winner ZIP
- Stop My Server（persist 未核对前）
- 开 T1 或 T3 训练
- 把 H10K 改更长
- commit 本机脏树
- 删除 R6 / canary ZIP / Fast400 raw

用户已授权且仍有效：

- 官方只训这条 T2 H10K
- 不要休眠、留 Jupyter 标签、到期延長
- 训后把审核清单落盘交给下一 agent（即本文件）

---

## 4. 下一任允许的第一条命令是什么

默认 **只读**。具体 cell/bash 以明早执行稿为准：`docs/AXIS_PI05_T2_H10K_TOMORROW_PERSIST_FOLD_PLAYBOOK_20260914.md`。

1. 官方：PID、GPU、log 尾、KEEP 目录列表、persist 是否已有对应 step
2. 本机：`git status`、GPU0/2 是否仍是 YOLO、不要杀

若训练仍在跑：观察 + 到 KEEP 点再 data-push。不要第二实例。

若训练已停且 GPU 空：第一条占用官方 GPU 的命令只能是 **fold 当前最高完整 checkpoint 的 params**，或 **resume 同一 H10K**（仅当未到 10000 且 persist 有 `train_state` 且剩余配额够跑完+fold）。不要 `--mode new`。

若 persist 已有 safetensors + SHA：官方可以停；第一条占用 **本机** GPU 的命令只能是 GPU3 上针对该 SHA 的冻结 C20，且不碰 GPU0/2。

---

## 5. 给审核者的最短结论栏

接手后先填，再动手：

```text
official_train_alive:           true/false/unknown
last_complete_step:             1|10|100|1000|3000|10000|other
train_state_in_persist:         true/false
params_in_persist:              true/false
folded_safetensors_in_persist:  true/false
jax011_fold_still_needed:       true/false
hours_official_quota_left:      <number or unknown>
next_action:                    observe | data-push | fold | resume-same-run | local-C20 | stop-after-persist | blocked
blocked_reason:                 <or none>
new_training_identity_allowed:  false
auto_submit_allowed:            false
```

`next_action=blocked` 时只汇报，不改官方 live 树、不 Stop、不 ZIP。
