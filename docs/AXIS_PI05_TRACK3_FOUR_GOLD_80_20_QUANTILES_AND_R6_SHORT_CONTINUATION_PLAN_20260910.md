# AXIS Pi0.5 Track3 四条 Gold 80/20、GPU3 单卡测试与官方 Blackwell 训练交接计划

> 创建日期：2026-09-10（Asia/Tokyo）
> 计划性质：用户显式要求落盘、交给另一位 agent 执行的下一步计划
> 当前状态：`Q0_ready`
> 当前可执行范围：本地 `Q0–Q6`；`Q5` 只用 GPU3 做 1→10-step 单卡测试；测试通过并取得官方环境访问后释放 `Q7–Q8`
> 当前停止点：四条 reverse accepted 数据已存在；尚未生成 pre-action 训练视图、80/20 manifest 或新 QUANTILES
> 正式审核：本轮未要求，不自动创建独立审核计划

## 0. 本计划与当前仓库指针的关系

用户在 2026-09-10 明确要求把下一步计划写入仓库，供另一位 agent 执行。

当前 `docs/AXIS_PI05_PLAN_INDEX.md`、Markdown handoff 和 machine JSON 仍指向
`docs/AXIS_PI05_TRACK3_FAST_TELEOP_400_AND_SHORT_TRAIN_PLAN.md`，并停留在
`FT0_implement_fast_recorder`。实际文件系统已经存在四条 accepted episode，状态领先于这些指针。

根仓、`LIBERO` 和 `LIBERO-plus` 当前包含既存未提交修改。本次只新增本计划文件，不归档旧计划，
不改写 plan index/handoff/machine JSON，不提交或覆盖其他人的修改。执行者只有在用户明确指定本文件，
或现有 dirty writer 安全同步 canonical `next_plan` 后，才把本文件作为执行授权；不得自行清理或提交现有脏改动。

## 1. 已完成输入与轻量验收

accepted 根目录：

`data/reports/axis_libero_30k/track3_reverse_pilot_v2/accepted/`

| episode | family | 帧数 | simulator state 维度 |
|---|---|---:|---:|
| `official_t3_yellow_book_1788971728` | support removal | 260 | 58 |
| `official_t3_alphabet_soup_1788946170` | containment removal | 278 | 97 |
| `official_t3_red_mug_1788976069` | support removal | 240 | 84 |
| `official_t3_two_mugs_1788976820` | multi-object support removal | 282 | 84 |

合计：`4 episodes / 4 task IDs / 3 families / 1060 frames`。

2026-09-10 的只读轻量验收已经确认：

- 四条 `metadata.json` 最终状态均为 `accepted`；
- `trajectory.npz` 可在 `allow_pickle=False` 下读取，无 object array；
- 数值数组 finite，action 为 `[T,7]`，gripper 为 `{-1,+1}`；
- state/action/image/contact offsets/frame count 对齐；
- fresh replay 为 `3/3`，人工 review 与 promotion 链存在且 hashes 一致；
- 原始/H.264 双视角视频均为 20 fps 且帧数精确；
- 没有 Track3 recorder、MuJoCo、replay、OpenPI trainer、cursor、PID、heartbeat 或 `.partial`；
- GPU0/1/2 上的 YOLO 进程属于外部用户，禁止终止、迁移或抢占；GPU3 空闲。

历史 `replay_summary.json` 中若仍保留 `human_review_pending` 或
`accepted_promotion=false`，它表示 promotion 前的不可变 replay 快照，不覆盖后续
`human_review.json`、`promotion.json` 和最终 `metadata.status=accepted`。

## 2. 必须修正的训练时序合同

原始 recorder 的顺序是：

1. 执行 `env.step(action_t)`；
2. 保存返回的图像 observation；
3. 保存 `env.get_sim_state()`。

因此 `trajectory.npz["states"][t]` 是 **post-action 完整 MuJoCo simulator state**，
不是 OpenPI 需要的 8D policy state；原始帧图像同样是 post-action observation。
禁止直接把原始 `images[t]`、`states[t]` 与 `actions[t]` 当作同一步监督，否则会产生一步错位。

canonical reverse training sample 必须由 `start_state.npy + actions` fresh replay 派生：

```text
restore start_state
obs_t = current pre-action observation
for action_t in recorded actions:
    emit (pre-action front_t, wrist_t, state8_t, action_t, prompt)
    obs_after = env.step(action_t)
    verify full_sim_state_after == recorded post-action states[t]
    obs_t = obs_after
```

8D state 顺序固定为：

```text
[eef_position_xyz(3), quaternion_to_axis_angle(3), gripper_qpos(2)]
```

不得使用完整 58/84/97D simulator state 代替 8D policy state，也不得把 quaternion 四维直接拼入。

## 3. 固定输入与输出命名

### 输入

- forward 数据：`data/raw/hf/lerobot/libero_plus/`
  - revision：`f3f49f426d75030177b18778374005bc12ccd588`
  - LeRobot v3、20 fps、14347 episodes、2238036 frames、40 tasks
  - state `[8]`、action `[7]`、front/wrist `256×256×3`
- forward catalog：`data/processed/libero_axis_mix_v1/catalog.jsonl`
  - 只选择 `source_repo == "lerobot/libero_plus"`
  - 不把五个 IPEC source 混入本次 forward 80%
- reverse 输入：第 1 节列出的四条 accepted episode
- continuation 初始化：R6 `r6_dual_l2_fresh500_r1` step 500 参数

### 新输出

- 配置：`configs/axis_pi05/track3_four_gold_mix_v1_20260910.yaml`
- reverse materializer：`tools/axis_pi05/materialize_track3_reverse_gold_v1.py`
- 80/20 manifest builder：`tools/axis_pi05/build_track3_four_gold_mix_v1.py`
- multi-root norm 工具：`tools/axis_pi05/compute_track3_mix_norm_v1.py`
- reverse LeRobot repo：`data/processed/axis_pi05_track3_reverse_gold_v1/`
- mixture artifacts：`data/manifests_track3/four_gold_80_20_v1/`
- 机器证据：`data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/`
- 本地单卡测试：`data/reports/axis_libero_30k/track3_four_gold_local_single_gpu_canary_20260910/`
- 官方传输清单：`data/reports/axis_libero_30k/track3_four_gold_official_transfer_v1_20260910/`
- focused tests：
  - `tests/test_track3_reverse_gold_materialization.py`
  - `tests/test_track3_four_gold_mix_and_norm.py`

工具或测试文件若已由另一位 writer 创建，执行者必须先检查和复用，禁止覆盖同名在途实现。

## 4. Q0：执行前碰撞与输入锁

Q0 只读执行，不启动 simulator 或训练。

1. 按 `AGENTS.md` 重新读取 MEMORY、plan index、handoff、machine JSON、active plan 和 latest audit。
2. 严格 duplicate-key 解析 machine JSON 及四条 episode 的 JSON 文件。
3. 记录 root、`third_party/axis_openpi`、`LIBERO`、`LIBERO-plus` 的 HEAD 与 dirty paths。
4. 记录 GPU UUID、PID、显存、利用率和完整命令；不得触碰外部 YOLO。
5. 查找 live trainer/recorder/replay/MuJoCo、cursor/PID/heartbeat/partial。
6. 为四条 episode 生成输入 inventory，至少锁定：
   - `metadata.json`、`trajectory.npz`、`start_state.npy`、`end_state.npy`；
   - `replay_summary.json`、`human_review.json`、`promotion.json`；
   - BDDL、resolved model XML、所用 asset XML；
   - front/wrist 原始视频；
   - bytes 与 SHA-256。

Q0 通过条件：四条目录和 SHA 唯一、accepted 链完整、总帧数 1060、没有相同目标 namespace 的 live writer。

若输入 bytes/hash 变化、出现 `.partial`、存在同目标 writer 或身份不明，停止并报告；不得修复旧 raw。
现有已知 dirty 文件本身不是自动失败条件，但其路径和内容 hash 必须记录，且不得 reset/checkout/clean。

## 5. Q1：实现并验证 reverse pre-action materializer

Q1 可以先完成 CPU code/fixture；真实四条重放仅允许 GPU3，且同一时刻最多一个 simulator process。

实现要求：

1. 从每条 `start_state.npy` fresh restore，不从原始第一帧图像猜测初态。
2. 对每个 action，在 action 执行前采集 front/wrist 和 8D state，然后执行 action。
3. 每一步执行后，把当前完整 simulator state 与原始 `states[t]` 对齐验证；记录 exact 与
   `max_abs_error`。默认 hard gate 为 `max_abs_error <= 1e-8`，若环境确定性实际需要放宽，
   只能停下报告，不能静默改变阈值继续发布数据。
4. 四条 replay 均需重现最终 goal/success；本阶段验证训练时序，不改写原有 replay/promotion 结论。
5. 图像必须保持 20 fps、`uint8[256,256,3]`、front/wrist 两路；不得加入 goal marker、HUD 或人为 overlay。
6. action 保持原始 7D float 值和顺序，不平滑、不插值、不翻转 gripper。
7. 使用 episode 原始 instruction 作为 task prompt：
   - yellow book：`take the yellow book off the top of the shelf and put it back on the table on the left`
   - alphabet soup：`take the alphabet soup out of the tray and place it on the table in front of the ketchup`
   - red mug：`take the red mug off the left plate and put it back on the table behind the other two mugs`
   - two mugs：`take the white mug off the left plate and the yellow and white mug off the right plate, then place the white mug to the left of the red mug and the yellow and white mug to the right of the red mug`
8. 发布标准本地 LeRobot repo，至少含 `meta/info.json`、tasks/episodes metadata、parquet 和两路视频，
   使 `AxisSelectedLeRobotDataset` 能以显式 local root 读取。
9. 写入临时目录，完成 schema/frame/hash 检查后原子 rename；失败时保留诊断但不得发布半成品。
10. 每条 derived episode 记录其 raw episode SHA、start-state SHA、BDDL/model/asset SHA 和 materializer SHA。

Q1 验收：`4/4` episodes、`1060/1060` samples，state `[1060,8]`、action `[1060,7]`，
所有数值 finite，逐 episode frame 数精确，post-step state 对齐通过，generic OpenPI local loader 能读取首/中/末样本。

任何一条失败只停止该条 materialization；不得用其余三条冒充四条完成。最终 80/20 正式 manifest 要求四条全部通过。

## 6. Q2：构建 LIBERO-plus 80% + reverse 20% manifest

本计划中的 80/20 是 **sampler probability mass**，不是通过复制文件形成 80/20 行数。

### Forward 80%

1. 从 catalog 选择 14347 条 `lerobot/libero_plus` episode。
2. 使用既有 `build_mix_v2.py` 的 deterministic split 语义：
   - split seed `1000`；
   - validation fraction `0.10`；
   - episode identity 为 split unit；
   - train/validation episode 不重叠。
3. forward train 内沿用 `M1 competition_prior` 的 task/perturbation 分层逻辑，但只在
   LIBERO-plus 子集内归一化，然后把全部 forward episode mass 乘以 `0.8`。
4. 不得直接缩放现成的总 M1 manifest，因为它还可能包含 IPEC sources。

### Reverse 20%

按 family 先均分 reverse mass，再在 family 内均分 episode：

| reverse episode | episode probability mass |
|---|---:|
| yellow book | `0.2 / 3 / 2 = 0.03333333333333333` |
| red mug | `0.2 / 3 / 2 = 0.03333333333333333` |
| alphabet soup | `0.2 / 3 = 0.06666666666666667` |
| two mugs | `0.2 / 3 = 0.06666666666666667` |

每条 manifest row 必须含：

- `source_repo`、`source_episode_index`；
- `episode_sampling_probability`；
- `observed_frame_count`；
- `frame_weight = episode_sampling_probability / observed_frame_count`；
- source revision/raw episode identity/family/task/prompt provenance。

Q2 hard gates：

- episode probability sum 在 `1e-10` 内等于 1；
- `sum(frame_weight * observed_frame_count)` 在 `1e-10` 内等于 1；
- forward/reverse mass 分别精确为 `0.8/0.2`；
- reverse 三个 family mass 各为 `0.2/3`；
- 所有 probability/frame_weight finite 且大于 0；
- observed frame counts 与实际 parquet 精确一致；
- forward train/validation 无 overlap；四条 reverse train 与 reverse evaluation start identities 无 overlap。

产物至少包括：

- `forward_train_episodes.parquet`
- `forward_validation_episodes.parquet`
- `reverse_train_episodes.parquet`
- `sampling_weights.parquet`
- `distribution_before_after.csv`
- `manifest.json`、`manifest.sha256`
- `source_roots.json`，显式映射 forward raw root 与 reverse derived root

## 7. Q3：从最终有效 manifest 计算新 QUANTILES

不得复用 Axis release norm、旧 R6 norm 或旧 M0/M1/M2 norm。本次 stats 只来自 Q2 的最终有效 80/20 分布。

现有 `compute_norm_v2.py` 假设所有 repo 位于同一个 `raw_root/org/repo` 下，而本计划的 forward 与
derived reverse 位于不同 roots。`compute_track3_mix_norm_v1.py` 应显式读取 Q2 的 `source_roots.json`，
复用已有 `WeightedStats`、vector validation、manifest weight 公式与 exact weighted sort，不应复制出另一套不一致算法。

要求：

- state dim `8`、action dim `7`；
- 使用每帧 `frame_weight` 计算精确加权统计；
- quantile 方法为 deterministic exact weighted sort；
- per-dimension q01/q10/q50/q90/q99（以及现有 norm contract 要求的字段）单调、finite、shape 正确；
- gripper 两端与原始 `{-1,+1}` 合同一致；
- 再次核对实际读取帧数与 manifest `observed_frame_count`；
- provenance 写入 manifest SHA、forward revision、四条 raw SHA、materializer SHA、source roots SHA 和工具 SHA。

输出：

- `norm_stats.json`
- `norm_stats.provenance.json`
- `norm_stats.sha256`
- `sampler_audit.json`

同一份 `norm_stats.json` 与 SHA 必须被 OpenPI training、reverse evaluation、C20 retention、fold、
LeRobot processors 和 FAST export 引用。任何环节出现另一个 norm hash 都是 hard stop。

## 8. Q4：loader 与混合数据轻量验收

本阶段不启动训练，只执行 focused tests/smoke：

1. 用 `local_repo_ids=("lerobot/libero_plus", "axis-pi05/track3_reverse_gold_v1")` 和两个显式
   `local_repo_roots` 实例化 `WeightedConcatDataset`。
2. 指向 Q2 `sampling_weights.parquet`，验证两 source 的 episode selection 和逐帧权重长度。
3. 读取 forward/reverse 各至少一个首帧、中间帧和尾帧；检查：
   - state `(8,)`、action horizon `(10,7)`；
   - 两路 RGB 图像存在且 dtype/shape 正确；
   - prompt 与 task metadata 一致；
   - 尾部 horizon padding/mask 符合当前 OpenPI 合同。
4. 用固定 sampler seed 做统计 smoke，报告观测 forward/reverse mass；统计抽样只作实现检查，
   不用随机近似替代 Q2/Q3 的精确权重验证。
5. 运行新增 focused tests、相关 existing loader/norm tests、strict JSON 和 `git diff --check`。
   本轮不要求自动运行全量项目测试。

Q4 完成状态：`track3_four_gold_80_20_quantiles_ready_for_local_gpu3_single_canary`。

完成后写一份简洁 summary，列出：4/4、1060 frames、manifest/norm SHA、source masses、family masses、
测试 argv/result、root/nested HEAD+dirty paths，以及 GPU/process after snapshot。不得声称已经训练或评估。

## 9. Q5：GPU3 单卡 1→10-step 本地训练测试

Q5 的目标是验证这份新数据、sampler、QUANTILES、R6 restore、梯度累积和 checkpoint 路径可以真实训练，
不是在本地完成正式性能训练。GPU1/2 是否空闲不再是 Q5 前置条件。

只有以下条件全部成立才可启动：

- Q1–Q4 全部通过，manifest/norm/materialized reverse hashes 已冻结；
- GPU3 无 compute PID、显存接近 idle，且没有 simulator/replay/eval；
- GPU0/1/2 及其外部 YOLO PID 完全不触碰；
- 没有另一个 trainer/supervisor/cursor；
- R6 step-500 params identity/hash 正确；
- 磁盘剩余空间不少于 500 GiB。

使用一个 persistent single-instance supervisor，记录 PID、command、root/OpenPI commit、配置/工具/manifest/norm
SHA、GPU3 UUID、atomic cursor、heartbeat 和 NVML peak。SSH/Codex 断开不等于失败，禁止启动第二实例。

固定本地单卡合同：

- device：物理 GPU3 一张；`fsdp_devices=1`；
- 初始化：`r6_dual_l2_fresh500_r1` step 500 **params only**；
- optimizer/scheduler fresh start，不恢复 R6 旧 optimizer；
- topology：`L2_action_lora_heads`；
- loss/action horizon：保持现有 R6 合同；
- precision：FP32，禁止 BF16；
- peak LR：`1e-6`；warmup：100 optimizer steps；
- AdamW：`b1=0.9`、`b2=0.95`、`eps=1e-8`、`weight_decay=1e-10`；
- gradient clip：`0.1`；EMA：off；
- microbatch：1；accumulation：64；effective global batch：64；
- sampling seed/data order 固定；数据和 norm 只能使用 Q2/Q3 的 SHA；
- 连续运行到 step 10，保存 step 0/1/10；不得把 1、10 拆成不同 fresh run。

Q5 通过条件：

1. step1 完成一个真实 global-batch-64 optimizer update；
2. loss、grad、clip、update、trainable params 和 optimizer state 全部 finite；
3. trainable leaves 与 `L2_action_lora_heads` 精确一致；
4. step1 与 step10 checkpoint 均能在 fresh process 单卡 restore；
5. step10 的 step/scheduler/examples 计数正确；
6. loader 实际使用冻结的 80/20 manifest 和 QUANTILES SHA；
7. 无 GPU0/1/2 activity、无 orphan process、无 partial checkpoint。

通过状态：`track3_four_gold_local_single_gpu_canary_passed_ready_for_official_bundle`。

Q5 checkpoint 只作本地可行性证据。官方正式 run 必须重新从 R6 step500 params + fresh optimizer 开始，
不得从本地 step10 继续，以免把本地 canary 的数据顺序和设备差异混入正式结果。

OOM、nonfinite、restore、global-batch 或 hash gate 失败时停止，不上传训练任务，不自动降低 batch、切 BF16 或换 LoRA 范围。

## 10. Q6：官方 GPU 传输包与数据同一性

### 10.1 哪些内容必须传到官方环境

以下是项目自有资产，官方机器不可能自动具备，必须上传或通过受控私有存储下载：

1. Q1 生成的四条 **pre-action aligned** reverse LeRobot repo；
2. Q2/Q3 的 `sampling_weights.parquet`、split、manifest、`source_roots.json`、QUANTILES 和全部 SHA/provenance；
3. R6 step500 的 Orbax **params-only** 目录；本地当前约 12 GiB；
4. 实际训练 code/config、OpenPI identity、依赖锁和 Q5 单卡测试 summary；
5. 一份 transfer inventory：所有文件的相对路径、bytes、SHA-256 和 inventory 自身 SHA。

四条 accepted raw 当前约 168 MiB。官方训练只需要 Q1 materialized repo，不需要原始 post-action 视频/NPZ；
默认不上传 accepted raw，只在 transfer inventory 中保留其来源 SHA。原始数据继续留在本地作为不可变 provenance。

### 10.2 Forward 15 GiB 的两条合法路径

官方 `~/dataset` 是只读配布数据集目录，但不能仅凭目录名假设其中就是本计划的
`lerobot/libero_plus@f3f49f426d75030177b18778374005bc12ccd588`。它可能是不同数据源、不同 revision、
10 Hz、不同 episode split 或不同 state/image schema。

只能二选一：

- **路径 A（优先）**：官方端按固定 revision 下载 `lerobot/libero_plus`，然后与本地 per-file manifest、
  14347 episodes、2238036 frames、20 fps、state8/action7、双相机合同逐项核对；
- **路径 B**：如果官方环境网络/缓存不可用，把本地已锁定的 LIBERO-plus 副本一起传输，当前约 15 GiB。

若 `~/dataset` 中的副本通过完整 identity/hash/schema 检查，可直接只读使用而不重复上传；
只要 revision/hash/frame/schema 任一不一致，就视为不同数据，禁止与本地生成的 QUANTILES 混用。

不能用 similarly named 的 standard LIBERO、10 Hz `physical-intelligence/libero`、IPEC 数据或配布 baseline dataset
替代当前 forward 80%，除非另建 manifest、重新计算 QUANTILES，并把它当成一个新实验。

### 10.3 容量与传输方式

当前已知大小：forward 约 15 GiB、R6 params-only 约 12 GiB、accepted raw 约 168 MiB；
包含 forward 副本的训练输入约 27 GiB 加 derived reverse/code/provenance，低于官方 `~/data` 100 GiB 持久化上限。

官方资料 v1.1 的存储边界：

- `$HOME`：40 GiB 持久化，不作为大数据和多 checkpoint 主目录；
- `~/data`：100 GiB 持久化，重要输入/selected checkpoint 放这里并使用 `parc-home-sync data-push`；
- `~/dataset`：只读，只能在 identity 完全匹配时复用；
- `/opt/dlami/nvme`：约 1.7 TiB 临时工作盘，训练和中间 checkpoint 放这里，停机前必须取回必要产物。

执行者先在本地生成分片传输包和 SHA 清单，不自动上传到任何外部服务，不写入 token/credential。
实际传输需要用户提供官方环境访问，或明确选择 private Hugging Face/官方上传入口。每个分片在官方端重新验 SHA；
解包后再验完整 inventory。上传/下载中断不得使用不完整目录开始训练。

根仓当前包含未提交修改，因此 source bundle 必须按实际训练文件 bytes 建 manifest；不能只记录 root HEAD 并假装 dirty diff 不存在。

Q6 完成状态：`track3_four_gold_official_bundle_ready_pending_user_transfer_or_remote_access`。

## 11. Q7：官方 RTX PRO 6000 Blackwell 正式训练

按仓库内的官方资料 v1.1，Track3 配布开发 GPU 总上限为 120 小时；单次 Notebook/server 最长 12 小时，
完全 idle 1 小时会停止。使用运营方支持的 Notebook/终端与配布环境，不导入 unsupported custom Docker；
训练结束后主动 Stop Server，避免继续计时。

官方 session 启动后的顺序：

1. 记录 GPU/driver/CUDA/JAX/Python/磁盘和剩余配额，不假设本地 V100 环境等价；
2. 把必要输入从 `~/data` staging 到 `/opt/dlami/nvme`；
3. 对 code、R6 params、reverse repo、forward repo、manifest、norm 执行完整 SHA/schema gate；
4. 用与 Q5 相同的 fixed seed、FP32、microbatch1、accumulation64、global64 做官方 step1 compile/update canary；
5. step1 通过后，在同一个 fresh run 连续训练至 step100，保存 0/1/10/100；
6. 记录 finite、throughput、peak VRAM、host RAM、checkpoint restore 和 data/norm hashes；
7. 把 step10、step100 的必要 params/train-state、日志和 summary 同步到 `~/data`，执行 data-push；
8. 主动停止官方 server，并把 selected checkpoint/日志下载回本地后再次验 SHA。

正式 run 仍从 R6 step500 params + fresh optimizer 开始，训练合同与 Q5 相同；不得因为 Blackwell 显存更大就改 BF16、
LoRA rank、LR、norm、80/20 mass 或 global batch。可把 checkpoint 写在临时 NVMe，但不得只留在临时盘。

本地 Q5 通过只证明数据/训练路径可行，不证明 Blackwell 软件栈兼容。因此官方 step1 失败时保存环境和错误证据并停止，
不得在计费 session 内无边界试装/改 recipe。官方端恢复训练前也必须检查是否已有 live notebook/trainer，禁止双实例。

Q7 step100 完成状态：`track3_four_gold_official_step100_ready_for_local_eval`。

## 12. Q8：回传后 reverse dev、C20 与 300-step 决策

四条 accepted episode 全部用于训练，不能把相同 start-state SHA 当作 held-out generalization。

训练前为四个 task 各创建 evaluation-only start identity，要求其 start-state SHA 与四条训练 identity 不同，
并固定 policy/noise seed。若 simulator 只能生成相同 start SHA，则 held-out generalization 标为 `not_evaluated`；
允许 same-start diagnostic，但不得称为泛化结果。

官方 step100 checkpoint 回传后，在本地 GPU3 fresh process restore；先验证 checkpoint、manifest、norm 与官方 summary SHA，
再用与 R6 step0 完全相同的 reverse dev identity/seeds 和固定 C20 retention 评估。继续门：

- reverse success 相对 step0 至少 `+1`；
- C20 success loss 不超过 `2/20`。

决策：

- 达门：选择候选进入 fold/parity/FAST/local ZIP 的后续 release 计划；不自动上传或比赛提交。
- 训练 finite/稳定但 reverse 无改善：可在下一次官方 session 从 step100 **optimizer checkpoint** 恢复到 step300，
  保持同一 manifest/norm/model/rank/LR/seed/global batch；回传 step300 后重复同一评估。
- nonfinite、明显退化或 C20 loss 超门：不延长，不把该 checkpoint 变成候选。

fold、FAST、ZIP 已接近发布边界，应另行执行正式 release 检查。官方 Blackwell 能训练并不替代最终 L4 24 GiB、
`/act < 10s`、server、FAST、ZIP 和 pre-check Docker 门禁。

## 13. 停止规则与交接摘要

### 全局停止

- live writer/supervisor/cursor 或官方 notebook/trainer 冲突；
- raw、start-state、BDDL/model/asset/R6 params 身份不明或 hashes 漂移；
- `.partial`、原子发布失败、NaN/Inf、schema/frame mismatch；
- GPU0/1/2 外部 PID 被干预，或 GPU3 与 simulator/replay/eval 并发；
- local/official 的 code、forward、reverse、manifest 或 norm hash 不一致；
- 官方输入未完整同步、临时盘产物未持久化、配额/12h/storage 边界不明。

### 局部失败

- 单条 reverse materialization mismatch：隔离该条并停止正式四条 manifest 发布；保存诊断，不改 raw。
- focused test 失败：修复对应 code/test 后重跑 focused 范围，不自动扩展为 formal audit。
- GPU3 忙：正常等待，不改用 GPU0/1/2。
- 官方 `~/dataset` 与本地 forward 不同：只切换到固定 revision 下载或上传副本，不修改 QUANTILES 迁就它。
- 单个传输分片失败：重传该分片并验 SHA，不重建或覆盖已经验证的源数据。

### 最小交接内容

执行者完成本地阶段后报告：

1. 四条 materialized episode 与 1060 帧是否全部通过；
2. forward/reverse 与 reverse-family 精确 mass；
3. manifest、norm、source roots、materializer 的绝对路径与 SHA；
4. focused tests 与 loader smoke 结果；
5. Q5 GPU3 单卡 step1/10 finite、update、restore、global64 和峰值显存结果；
6. Q6 transfer inventory、分片、总大小与 SHA；
7. forward 采用官方端 exact cache、固定 revision 下载还是本地上传副本；
8. 官方 step100/300 的 run/checkpoint/log SHA、Notebook 时间和持久化状态；
9. 实际新增/修改文件清单，且声明未覆盖哪些既存 dirty paths。
