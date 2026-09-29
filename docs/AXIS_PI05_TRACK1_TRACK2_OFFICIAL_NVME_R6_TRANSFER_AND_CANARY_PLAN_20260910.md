# AXIS Pi0.5 Track1 / Track2：官方 NVMe 数据解包、R6 传输与短训启动计划

> 制定日期：2026-09-10（Asia/Tokyo）
> 状态：`track12_official_preflight_ready_not_started`
> 上游计划：`docs/AXIS_PI05_TRACK1_TRACK2_OFFICIAL_BLACKWELL_LORA_PLAN_20260910.md`
> 执行环境：PARC 官方 JupyterLab，RTX PRO 6000 Blackwell Server Edition
> 官方入口：`https://parc-playground.omnicamp.us/user/huangpenguin/lab`
> 当前 session：约 11 小时 32 分剩余；本选 GPU 配额约 58 小时 55 分剩余（用户报告，启动时必须重读页面）
> 当前事实：官方实例已启动；尚未训练；没有把大数据解压进 40 GiB home
> 用户当前指令：不要 Stop My Server；不要点击延长；先完成本计划门禁
> 并行边界：本地 Track3 Q5 正在由另一工作流执行，本计划不得停止其 PID、占用其本地 GPU、修改其 cursor/heartbeat/manifest 或覆盖其 dirty 文件

## 0. 目标与停止点

本计划只负责把 Track1/Track2 的官方训练链推进到以下有界停止点：

1. 在 `/opt/dlami/nvme` 解包并审计唯一选定的 20 Hz combined 数据；
2. 冻结可复现的 Track1/Track2 episode identity、采样权重和 normalization；
3. 把本地 R6 step-500 raw parameters 可靠传到官方机并做 hash/restore/parity 检查；
4. 在 Blackwell 上完成 Track1 100-step canary；
5. Track1 通过行为门禁后，才允许 Track2 100-step canary；
6. 保存 checkpoint、日志、manifest 与 session closeout，等待本地 paired rollout 选优。

本计划不授权：

- Track3 的任何操作；
- dense full fine-tuning；
- 直接跑 500/1000/3000 steps；
- 使用隐藏评测 observation/result 训练；
- 自动上传比赛 ZIP 或自动提交；
- 把官方 `pi05_step005000_submission_py310.zip` 当成 R6；
- 为了追求低 offline loss 绕过 rollout 门禁；
- 把数据、模型或虚拟环境解压到 40 GiB home。

## 1. 已冻结的事实

### 1.1 官方服务器

```text
GPU                 NVIDIA RTX PRO 6000 Blackwell Server Edition
CUDA nvcc           13.2
dataset mount       ~/dataset -> /opt/parc/dataset
persistent data     ~/data，100 GiB，按官方说明手工 data-push/data-pull
ephemeral work      /opt/dlami/nvme，约 1.7 TiB
home                40 GiB；禁止存放或解包大数据/大 checkpoint
```

`nvcc 13.2` 不等于 JAX wheel 已经可用。训练前必须用真实 JAX CUDA smoke 验证 device、matmul、一次 model forward 和一次 optimizer update；不得仅凭 `nvcc --version` 放行。

### 1.2 官方只读数据源

`~/dataset` 当前没有现成目录，只有至少以下 tar/ZIP：

```text
libero_combined_20hz.tar
libero_combined_20hz_v2.tar
lerobot/libero_combined_20hz.tar
IPEC-COMMUNITY/<five suites>.tar
pi05_step005000_submission_py310.zip
```

其中 `pi05_step005000_submission_py310.zip` 是官方 baseline，不是：

```text
r6_dual_l2_fresh500_r1 / step 500 / L2_action_lora_heads
```

### 1.3 已有 Track1/Track2 数据计划

已有本地 20 Hz catalog 生成的 primary mixes：

```text
T1-P20
  train / validation    10149 / 1217 episodes
  sampling mass         80% standard + 10% lighting + 10% env

T2-P30
  train / validation    12696 / 1517 episodes
  sampling mass         70% standard + 10% camera_pose
                        + 10% sensor_noise + 10% env
```

这些不是旧 M0/M1/M2 的约 50% LIBERO-plus 配比。`env` 当前是 coarse label；没有证据时不得称为 Background Texture，也不得拆成 Object Layout / Robot Initial State。

### 1.4 模型锚点

```text
run id                 r6_dual_l2_fresh500_r1
checkpoint             step 500 raw Orbax params
topology               L2_action_lora_heads
raw checkpoint tree    d203635f1eb0b006ec27841db6ce5171f74a37035eca4291c8107c05838c276f
R6 norm sha256         4c803bf59c52677c36b6b420c614d8a7742e0847708443863e13e67cb80b18f5
official scores        Track1 0.47 / Track2 0.33 / Track3 0
```

训练初始化只能使用 R6 raw LoRA topology parameters。folded PyTorch model可用于部署对照，但不能直接充当 raw L2 optimizer continuation。

## 2. 执行总序

```text
O0 远端资源与 tar 只读清单
  -> O1 v2 独立解包至 NVMe
  -> O2 schema / identity / subcategory 审计
  -> O3 冻结 Track1/Track2 effective manifests
  -> O4 传输并验证 R6 params
  -> O5 同步最小代码快照并完成 Blackwell/JAX preflight
  -> O6 Track1 fresh QUANTILES、loader smoke、step-0 restore
  -> O7 Track1 100-step canary
  -> O8 Track2 fresh QUANTILES、loader smoke、step-0 restore
  -> O9 Track2 100-step canary
  -> O10 持久化、session closeout、等待本地 paired rollout
```

不得并行启动 Track1 与 Track2 optimizer。数据清单/传输可并行，但 checkpoint writer 与 optimizer 始终保持单实例。

## 3. O0：官方实例只读资源锁

执行 Agent 首先记录：

- 页面显示的 session 剩余时间与总配额；
- `nvidia-smi` 完整 GPU/driver/memory/process snapshot；
- Python、JAX、CUDA runtime、cuDNN、XLA 版本；
- `readlink -f ~/dataset`；
- `/opt/dlami/nvme`、`~/data`、home 的容量；
- 三份 combined tar 与五个 suite tar 的 bytes、mtime、SHA-256；
- `tar -tf` 的顶层目录、metadata、parquet/video 文件计数；
- tar 是否包含绝对路径、`..` path traversal、异常 symlink 或重复覆盖目标。

建议本轮唯一工作根：

```text
/opt/dlami/nvme/parc_track12_20260910/
  archives_audit/
  datasets/
  code/
  venv/
  cache/
  runs/
  reports/
```

把 `HF_HOME`、`XDG_CACHE_HOME`、`UV_CACHE_DIR`、`TMPDIR` 指到上述 NVMe 子目录。不得重新定义或挪用 `HOME`。

O0 通过条件：NVMe 至少有 300 GiB 可用空间，GPU 无未知训练进程，tar 清单可读且路径安全。否则只写报告并停止。

## 4. O1：选择并解包 combined 20 Hz 数据

### 4.1 主路径

首选只解包：

```text
/opt/parc/dataset/libero_combined_20hz_v2.tar
```

解包目标必须是独立目录，例如：

```text
/opt/dlami/nvme/parc_track12_20260910/datasets/combined_v2/
```

要求：

- `--no-same-owner`、`--no-same-permissions`；
- 解包前后记录 archive hash、文件数、总 bytes；
- 不在 archive 原位置写任何内容；
- 不创建指向 home 的大文件；
- 不把 v2 目录重命名成 v1 来隐藏来源。

### 4.2 v1 fallback

仅当 v2 出现以下任一硬失败时，才允许把 v1 解包到另一个独立目录：

- tar CRC/读取失败；
- 缺少训练所需 observation/action/task/episode metadata；
- schema 与现有 loader 无法对应且不是可审计的小修；
- episode identity 无法与本地 catalog 一一对齐。

fallback 优先顺序：

1. `/opt/parc/dataset/libero_combined_20hz.tar`
2. `/opt/parc/dataset/lerobot/libero_combined_20hz.tar`

不得把多个 combined tar 解压到同一目录，也不得用后解压文件覆盖前一版本。

## 5. O2：数据与细粒度标签审计

对选中的 combined 数据验证：

```text
fps                       20
state dimension           8
external action dimension 7
cameras                   front + wrist（或明确可映射的官方名称）
episode/frame/task count  finite and internally consistent
gripper convention        明确记录并统一 transform
NaN/Inf                   0
episode boundaries        monotonic and in range
```

episode 对齐键固定为：

```text
source_repo / revision / source_episode_index
```

禁止用 combined 全局 `episode_index` 或 archive 中的物理文件顺序猜 source identity/perturbation category。

### 5.1 细粒度标签分支

如果 v2 含有可追溯到源 metadata/file manifest 的精确 subcategory：

```text
Track1-FINE
  80% standard + 10% lighting + 10% background_texture

Track2-FINE
  70% standard + 10% camera_pose + 10% sensor_noise
  + 5% object_layout + 5% robot_initial_state
```

必须生成新的 mix id、mapping report 和 hash，不能静默覆盖 T1-P20/T2-P30。

如果没有精确标签：

- 保留现有 T1-P20/T2-P30 coarse manifests；
- `env` 明确写 `recovered-coarse`；
- fine split 记为 `not_evaluated`；
- 这不阻塞 primary canary。

O2 产物至少包括：`archive_audit.json`、`schema_audit.json`、`episode_join.json`、`subcategory_audit.json`、`source_roots.json`。

## 6. O3：冻结 effective manifests 与 normalization 输入

远端不能直接信任本地 global episode index。执行 Agent必须以 stable identity 重新 join 本地 mix 与远端 combined dataset，并验证：

- T1 train/validation 分别为 `10149/1217`，或 FINE 分支中有明确、可解释的新计数；
- T2 train/validation 分别为 `12696/1517`，或 FINE 分支中有明确、可解释的新计数；
- train/validation stable identity overlap 为 0；
- sampling probability sum 为 1；
- frame weight 等于 episode probability / episode frame count；
- suite/task/category mass 与目标误差小于 `1e-6`；
- 所有 referenced frames 与视频 timestamp 在界内。

每个 track 的 QUANTILES 必须从该 track 的精确 effective sampler 重新计算，不能复用：

- R6 10 Hz norm；
- 官方 baseline norm；
- 旧 M0/M1/M2 norm；
- 另一个 track 的 norm。

先算 Track1 norm；Track1 100-step canary成立后再算 Track2 norm，以减少当前 session 的无效工作。

## 7. O4：R6 参数传输合同

### 7.1 传输内容

推荐创建一个 R6 transfer bundle，只包含续训和身份验证所需内容：

```text
r6_step500_params/params/**
checkpoint/source/resolved config
R6 source manifest and release lock
R6 norm manifest/stats（仅用于身份与 step-0 对照）
fixed input/output reference
SHA256SUMS
bundle_manifest.json
```

不传 optimizer/scheduler state；新训练必须 fresh optimizer/scheduler/EMA。若 OpenPI restore 实际需要 params 之外的 Orbax metadata，则把该 metadata 显式列入 bundle，不得改用完整旧 optimizer continuation。

### 7.2 通道优先级

1. **首选：私有 Hugging Face model/dataset repo**
   - 本地上传 bundle；官方机下载到 NVMe；
   - 使用受限 token，token 不写 notebook、Git、日志或 manifest；
   - 下载完成后立即逐文件 SHA-256 校验。
2. **备选：JupyterLab 上传到 `~/data/incoming/`**
   - `~/data` 是独立持久盘，不是 40 GiB home；
   - 大文件采用带 manifest 的分片 archive；全部分片到齐、hash 通过后才在 NVMe 合并/解包；
   - 不上传到 notebook 当前 home 目录。
3. **其他通道**只有在能提供端到端 bytes/hash 与访问控制时才允许。

不建议用单个约 12 GiB 浏览器上传作为首选，因为中断恢复和完整性较弱。不得创建公开模型链接。

### 7.3 远端放行门禁

- bundle manifest 严格 JSON；
- 每个文件 SHA-256 exact；
- raw parameter leaf names、shapes、dtypes、count 与本地 R6 exact；
- R6 topology 恢复为 `L2_action_lora_heads`；
- trainable/frozen leaves 与合同一致；
- fixed-input forward finite；跨硬件 max absolute drift 不超过 `1e-4`，gripper sign flip 为 0；
- step-0 参数没有被 optimizer update。

任一失败不得用官方 baseline ZIP替代，必须停止并修复传输/restore。

## 8. O5：最小代码快照与 Blackwell preflight

当前本地 root 有 Track3 live writer 和 dirty worktree。禁止直接压缩整个工作目录。应在独立临时目录构造 allowlisted snapshot：

1. 从已记录 root HEAD 导出 tracked code；
2. 只 overlay 已审核的 Track1/Track2 文件；
3. nested OpenPI 固定为记录的 commit/tree；
4. 排除 Track3 cursor、heartbeat、raw data、outputs、logs 和 secrets；
5. 生成 source inventory 与 SHA-256；
6. 在官方 NVMe 解包为只读 source snapshot，run outputs 写到独立目录。

Track1/Track2 allowlist至少覆盖：

```text
configs/axis_pi05/track12_official_lora.yaml
configs/axis_pi05/track1_example_holdout.json
tools/axis_pi05/build_track12_mix.py
tools/axis_pi05/official_blackwell_bootstrap.py
tools/axis_pi05/run_track12_official_lora.py
data/manifests_track12/**
所需 common modules/tests
```

执行前先阅读各工具 `--help` 和配置解析结果；禁止凭计划猜不存在的 CLI flags。

Blackwell preflight顺序：

1. Python import；
2. 单 GPU JAX device 与简单 matmul；
3. dataset 1/32/256 draws，shape/dtype/finite/identity/mass 检查；
4. R6 params restore；
5. fixed forward；
6. step-0 save/restore；
7. 一次真实 optimizer update后 fresh-process restore；
8. 显存峰值与 checkpoint 写盘时间记录。

不得复制本地 V100 venv。优先复用官方支持的 Python/JAX 环境，只把额外依赖和缓存放 NVMe。自定义 Docker不在本计划内。

## 9. O6–O7：Track1 100-step canary

训练配置固定：

```text
init                     R6 step-500 params only
optimizer/scheduler      fresh
topology                 L2_action_lora_heads
precision                FP32 first; no unreviewed BF16 switch
objective                valid7_temporal
peak LR                  3e-7
optimizer                AdamW
gradient clip            0.1
EMA                      off
effective global batch   64
action horizon           10
checkpoints              0,1,10,25,50,100
nonfinite                fail immediately
```

在单张 Blackwell 上先测可承受 microbatch，再用 accumulation 保持 effective global batch 64。不能因提高吞吐同时改 LR、objective 或 normalization。

Track1 放行顺序：

1. fresh QUANTILES finite/hash；
2. loader smoke；
3. step-0 R6 + T1 norm fixed forward；
4. step 1 finite/update/scope/restore；
5. step 10 restore；
6. 继续至 step 100；
7. checkpoint inventory、metrics、sample trace、GPU/time report完整。

offline loss只能用于发现 nonfinite/爆炸/loader 错误。step100是否继续到250/500，必须等待冻结的本地 paired rollout；不能在官方 session 内看 loss 后自行续训。

## 10. O8–O9：Track2 100-step canary

只有 Track1 technical canary 完整、且没有共享 runtime/data/restore 缺陷时才进入 Track2。Track2使用独立：

- manifest/mix id；
- QUANTILES hash；
- run id/output directory；
- optimizer/scheduler；
- fixed-input reference与checkpoint inventory。

训练超参和 Track1 相同，数据改为 T2-P30 或经精确标签审计后的 T2-FINE。Track2 不从 Track1 step100继续；仍从原始 R6 step500 params-only 初始化。

如果当前 session 时间不足：

- 优先保全 Track1 完整 checkpoint/report；
- Track2 标记 `not_evaluated`；
- 不点击延长；
- 不为赶时间跳过 norm/step0/step1 restore 门禁。

## 11. 行为评测与晋级门禁

官方训练机只负责训练与静态/固定输入验证。候选选择必须在冻结的本地 paired rollout上完成，不能训练于隐藏评测 observation/result。

每个 track 比较：R6 step0、step50、step100；相同 task/init/env/policy seeds。

共同晋级门：

- 标准 C20 相对 R6 最多下降 `2/20`；
- 对应 track 面板不能净退化；
- 至少增加 2 个 paired successes 才允许续到 250；
- collision、timeout、stall、goal-reached-but-collision 分开报告；
- Track2 另按 approach/contact/grasp/transport/place-release 分阶段报告；
- 多训练 seed 只对 finalist 做，不用 8-episode leaderboard替代不确定性。

若 step100 未通过，保留原 R6，不扩大 Plus 比例、不延长步数、不根据 failure results临时改 mix。

## 12. O10：持久化与本轮 closeout

训练期间 outputs 首先写 NVMe；关键 checkpoint 完成并 fresh restore 后，再复制到：

```text
~/data/track12_20260910/
```

至少持久化：

- selected source/archive/schema/subcategory audits；
- exact train/validation/sampling manifests与hash；
- Track1/Track2 norm stats/provenance/hash；
- R6 transfer/restore/parity reports；
- resolved configs、source snapshot manifest；
- checkpoint 0/1/10/25/50/100；
- metrics/sample trace/train log/process result/GPU timeline；
- incomplete/not_evaluated phase ledger。

完成后按官方说明执行 `parc-home-sync data-push`，确认返回成功并记录时间。只有在持久化检查完成后，后续执行者才可以决定是否停止服务器；本计划保持用户当前“不要 Stop、不要延长”的指令。

## 13. 当前 session 的建议时间预算

```text
O0 tar/resource inventory            0.5 h
O1 v2 extract                        1.0–2.0 h
O2/O3 data join and audit            1.0–1.5 h
O4 R6 transfer and validation        1.5–3.0 h
O5 environment/JAX preflight         1.0–1.5 h
O6 norm/loader/step0-step1           1.0 h
O7 Track1 100-step canary            remaining bounded window
```

如果 R6 传输或环境门禁占满当前 session，允许只完成 O0–O6 并 closeout；Track1/Track2 optimizer 均记 `not_evaluated`，不得用 baseline checkpoint补位。

## 14. 执行 Agent 交接摘要

```text
official server        running; user says do not Stop and do not Extend
training started       false
large data in home     false
primary archive        libero_combined_20hz_v2.tar -> NVMe only
fallback archive       v1 in separate NVMe directory only after v2 hard failure
T1 mix                 T1-P20, 80/10 lighting/10 env, coarse allowed
T2 mix                 T2-P30, 70/10 camera/10 sensor/10 env, coarse allowed
model init             R6 raw L2 step500 params only
transfer preferred     private Hugging Face + exact SHA-256
first optimizer target Track1 100 steps only
Track2 start           after Track1 technical canary, from original R6 again
Track3                 live parallel workflow; strictly untouched
automatic submission   forbidden
```

本文件是 Track1/Track2 的并行执行计划，不取代仓库中正在运行的 Track3 活动计划，也不授权修改顶层 Track3 cursor/heartbeat/manifest。
