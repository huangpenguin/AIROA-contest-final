# 官方 Blackwell T2 H10K 启动问题清单（赛后报告底稿）

> 日期：2026-09-13（Asia/Tokyo）
> 用途：赛后报告 / 赛后总结的事实底稿。不是新的训练授权，不改 mix / norm / LR / rank / batch。
> 对应计划：`docs/AXIS_PI05_FINAL_4DAY_TRAINING_EXECUTION_PLAN_20260912.md`
> 机器摘要：`data/reports/axis_libero_30k/t2_p30_r6_h10k_r1_official_launch_issues_20260913/summary.json`
> 用户确认：2026-09-13 约 16:42 JST，`t2_p30_r6_h10k_r1` **已经成功启动训练**。
> 训后移交：`docs/AXIS_PI05_T2_H10K_POST_TRAIN_AGENT_HANDOFF_20260913.md`

本文只记录已经发生过的失败和分类。不要事后把所有炸点都写成“包版本不对”。本机 V100 绿测通过，只能证明钉死的旧 JAX 栈和本地数据布局能训，不能证明官方 Blackwell 能训。

## 0. 冻结身份（这些都不是根因）

正式 run 仍然是：

```text
run id                 t2_p30_r6_h10k_r1
init                   R6 step500 params-only + fresh optimizer
禁止                    30k；resume canary100 train_state；T1/T2/T3 merge；改 mix/norm/LR/rank/batch
global batch           64（microbatch 4，accumulation 16）
peak LR                3e-7；warmup 10；decay 3e-8
topology               L2_action_lora_heads
precision              FP32
T2 manifest SHA        85700f9d85311cdadbacadcecf2b7c9b6617b77e03f51de42a81681ffc770af4
T2 norm SHA            39ff8a7cc2ff2a0be7d43659e708f9ec260686e876cf603c0460972b49746b41
CODE                   /opt/dlami/nvme/parc20260911/code/PARC2026_final
不要用                  残树 .../code/tools
```

官方机约束：不能 SSH；格子在 Mac 浏览器的 Jupyter 里跑；persist `~/data/parc20260911`；工作盘 `/opt/dlami/nvme/parc20260911`。用户明确不要 Stop My Server、不要 Restart and run all、电脑不要休眠、开训后留 Jupyter 标签防 idle。Agent 不自动上传/提交。

## 1. 分类原则

官方机上的失败至少分成四类。它们可以同时出现，但修复路径完全不同：

| 类 | 一句话 | 换成和本机一样的 pip 能消掉吗 |
|---|---|---|
| A. JAX 栈版本墙 | 官方驱动只能跑 JAX 0.11.1 + CUDA 13，OpenPI 钉的是更旧的 JAX/Flax/Orbax | 不能。本机旧钉死版本在 V100 上绿，官方卡上装不了 |
| B. 缺依赖 / 缺缓存 | 干净环境少文件、少包，不是 minor 不兼容 | 不能单靠“对齐版本”。要把 tokenizer 和 import 清单补齐 |
| C. 数据长得不一样 | 官方 IPEC tar 是 LeRobot v3 `file-*.parquet`，本地 mix 按 v2.1 `episode_000000.parquet` 写 | 不能。JAX/Flax/Orbax 换成和本机完全一样也会炸 |
| D. 操作层 | 空 run 目录、OpenPI 拒覆盖、shell 粘贴、锁屏杀实例、错误补丁缩进 | 不能。这是流程，不是包 |

另外还有一类经常被误写成 A 的：**源码拼树**。今日实例重生后，overlay 的 `train.py`/`data_loader.py` 和公开 `Physical-Intelligence/openpi` clone HEAD 的 `utils.py` 不是同一提交。这不是 pip minor，也不是配方写错。

## 2. 类 A：确实是版本墙（JAX 栈）

官方 Blackwell 只能用：

```text
Python 3.12
jax==0.11.1 + cuda13
flax==0.10.2（源码打 JAX 0.11 补丁）
orbax-checkpoint==0.11.25
transformers==4.53.2
huggingface-hub==0.36.0
```

OpenPI 文档/环境钉的是大约：

```text
jax==0.5.3
flax==0.10.2
orbax==0.11.13
```

本机 V100 训练走旧钉死版本，所以本地 focused tests 看不出来。这是**训练框架和官方 GPU 驱动不匹配**，不是 T2-P30 配方、manifest 或 norm 写错。

已经出现过的具体炸点：

1. Orbax 要 `DeviceLocalLayout`，JAX 0.11 里已经删掉。官方侧把 orbax 放到 `0.11.25`，仍要给 checkpoint 代码接受 `enable_async_checkpointing`。
2. Flax tracer：`jax.core.get_opaque_trace_state` 迁到 `jax.extend.core`；去掉 `jax.remat(concrete=)`。
3. `jaxtyping` 在 0.11 栈上要 `skip_prefixes`；钉 `jaxtyping==0.2.36`。
4. 单卡 mesh：`AxisType.Auto`、gemma `NamedSharding`、`train.py` 1-device `data_sharding`、`AXIS_PI05_DISABLE_ACTIVATION_SHARDING_CONSTRAINT=1`。
5. F2 `StepMetadata` unwrap、F2b augmax unroll。这些是 JAX/Flax 新栈上的控制流/元数据形状，不是数据错。

赛后应写清楚：官方卡的合法训练栈 **不能**等于 OpenPI 上游 pin；要有一份“Blackwell JAX 0.11 运行时补丁清单”，并禁止 `pip install -e third_party/axis_openpi` 把钉死栈冲掉。也不要往 `trainvenv` 装 `robosuite`/`libero`。

## 3. 类 B：缺依赖 / 缺缓存（环境不完整，但不是“改了版本”）

干净官方环境少东西。预先把 tokenizer 放进缓存、把 import 清单跑一遍就能拦住。不是某个包升了 minor 就不兼容。

已经出现过的具体炸点：

1. 没装 `gcsfs`，tokenizer 默认去拉 GCS。
2. 本机/官方 home 都可能没有 `~/.cache/openpi/big_vision/paligemma_tokenizer.model`。缺文件时训练在模型初始化阶段外网失败。
3. 缺 `tqdm-loggable`、`openpi_client` 未拷进 `trainvenv` site-packages。
4. `huggingface-hub` 必须钉 `0.36.0`；这是和 LeRobot/Hub 客户端契约有关，不要和类 A 的 JAX 墙混为一谈。
5. `trainvenv/bin/python` 是 uv symlink。对它做 `Path.resolve()` 会丢掉 site-packages。必须用 wrapper：
   `/opt/dlami/nvme/parc20260911/work/trainvenv_python.sh`
6. Jupyter kernel 是 `/opt/lerobot-venv`（Python 3.10），训练必须是 `trainvenv`（Python 3.12）。格子里可以修文件，但 `train.py` 只能用 trainvenv。
7. Hugging Face runtime 仓库 `Huangpenguin/parc2026-official-runtime-20260911` 今日 401（无 token）。这是凭证/缓存问题，不是 JAX。
8. 官方 persist 的 Q7 payload **不是完整 OpenPI**（有 `data_loader.py`，无 `models/model.py`）。不能拿 Q7 当 runtime 源。

赛后应写清楚：Blackwell 启动清单必须包含“文件存在”而不只是“pip freeze 对齐”。tokenizer、`openpi_client`、wrapper、HF token、wandb `~/.netrc` 都是环境完备性。

## 4. 类 C：不是包版本，数据长得不一样

官方 IPEC tar 是 LeRobot **v3**：

```text
data/chunk-*/file-*.parquet
videos/observation.images.image
videos/observation.images.wrist_image
```

本地 mix / 旧 loader 按 LeRobot **v2.1** 写：

```text
data/episode_000000.parquet
```

还曾经盖过一份 v2.1 的 `info.json`。Plus 官方 tar 解出来是 `libero_plus_lerobot/`，要 rename 成 `dataset/lerobot/libero_plus`。

就算把 JAX/Flax/Orbax 换成和本机完全一样，这个错照样炸。它跟 pip 无关。

今日 T2 H10K 上的直接证据：

```text
pid 5441
huggingface_hub.errors.RevisionNotFoundError
IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot
Your dataset must be tagged with a codebase version.
```

表面像 Hub/版本标签，实际是：

1. overlay 的 `data/manifests_track12/T2-P30/source_roots.json` 仍是 Linux 路径
   `/home/huang/code/PARC2026_final/data/raw/hf/IPEC-COMMUNITY/...`
2. 官方机上这些路径不存在，`_is_lerobot_v3_file_root(local_root)` 为 False。
3. `create_torch_dataset` 掉进 `LeRobotDatasetMetadata(repo_id)`，去问 Hub。
4. 不要执行 `HfApi.create_tag`。IPEC 必须走本地 `AxisPlusV3Dataset`。

随后同一条数据路径上还出现：

- `AxisPlusV3Dataset` 没有公开 `episodes` / `episode_data_index`，官方 overlay 的 `sampling_manifest` 分支仍按 LeRobot v2.1 的 `dataset.episodes` 取值（pid 6030）。
- IPEC 视频键是 `observation.images.image` / `wrist_image`，Plus 是 `front` / `wrist`。F1 `_lerobot_v3_video_dirs` 必须两种都认。

正确修复：

- 把 6 个 `source_roots` remap 到
  `/opt/dlami/nvme/parc20260911/dataset/IPEC-COMMUNITY/...` 和
  `/opt/dlami/nvme/parc20260911/dataset/lerobot/libero_plus`
- 确认每个 root `exists True` 且 `data/chunk-*/file-*.parquet` 为 True。
- 给 `AxisPlusV3Dataset` 补 `episodes` 与 `episode_data_index`，或让 `sampling_manifest` 走 `_index`。
- runner `_load_source_roots()` 应 remap Linux HF 前缀，避免下次 overlay 再把路径打回本机。

赛后应写清楚：官方 tar 的 on-disk layout 是独立契约。本地 preflight 如果只 hash `info.json` / manifest，而 vis 不到 `file-*.parquet`，就会把类 C 漏掉。

## 5. 类 D：也不是包版本，操作层

这些和版本无关。

1. **空输出目录**。OpenPI / runner 在同名 `run_dir` 已存在时拒绝覆盖。失败的空 `t2_p30_r6_h10k_r1` 必须先 rename 成 `*.failed_*`，才能 `--mode new`。禁止预先 `mkdir` 空 RUN。
2. **shell / Jupyter 粘贴**。两行命令黏在一起、三引号在格子里被吃掉、bootstrap 脚本引号损坏，都会造成“看起来像代码错”。
3. **锁屏休眠杀实例**。昨晚 H10K 曾启动成功（pid 210633，wandb online），锁屏后实例被杀，未 `data-push`，checkpoint 随 NVMe 消失，只能 `--mode new`。不要 Stop My Server（清 NVMe）；Mac 不要休眠；留 Jupyter 标签；checkpoint 后 `parc-home-sync data-push`（100/1000/3000/10000）。
4. **看错日志**。确认格必须读正式 log `t2_p30_r6_h10k_r1.train.log`，不要读 probe。pid 5441 的 14633 字节 Hub 栈被重复粘贴时，并不代表新一次失败。
5. **错误补丁**。用 12 空格去匹配 `if sampling_manifest:`，会命中“读 parquet 清单”那一行的后缀，把 v3 分支插进错误缩进（`IndentationError` at `isinstance(dataset, AxisPlusV3Dataset)`）。对 `data_loader.py` 的任何替换必须先 `ast.parse`，失败就恢复 `.bak_ipec_hub`。
6. **Hub create_tag 提示**。那是 LeRobot 的误导性建议，不是修复。
7. **不要 Restart and run all**。笔记本里还有旧 canary cell。
8. **GPU 边界**。本机 GPU0/GPU2 有外部 YOLO，禁止碰。官方机只有 1 张卡，正式 H10K 与 probe 不能并行。

## 6. 源码拼树（常被误写成类 A）

今日实例重生后：

- overlay tar 只是约 19 个文件的增量，不是完整 nested OpenPI。
- 公开 `git clone Physical-Intelligence/openpi` 成功，但 **不能 fetch 本地 nested 提交 `b58aff07`**（那是本仓库 nested commit，不是上游 tag）。
- 用 clone HEAD `215abfb` 填缺失文件，`skip-existing` 保住 overlay 的 `data_loader.py` / `config.py` / `train.py`。
- 结果：overlay `train.py` 仍传 `TrainState(..., rng=...)`，clone HEAD 的 `training/utils.py` 没有 `rng` 字段。

直接证据：

```text
pid 6609
TypeError: TrainState.__init__() got an unexpected keyword argument 'rng'
scripts/train.py init_train_state
```

修复是给官方 `openpi/training/utils.py` 的 `TrainState` 补 `rng: at.KeyArrayLike | None = None`，不是降 JAX。

同类还有：Axis 专用模块 `axis_lora.py`、`nan_replay.py`、`axis_pi05_contract.py` 必须写进官方 tree（Jupyter 三引号，不要 base64）。Q7 payload 不能覆盖已经打过 JAX 0.11 / v3 IPEC 补丁的 live OpenPI。

## 7. 2026-09-13 启动时间线

按失败进程归档，不改历史 log。

| 顺序 | 身份 | 现象 | 类 | 处理 |
|---|---|---|---|---|
| 昨晚 | pid 210633 曾启动 | 锁屏杀实例，无 data-push | D | 只能 `--mode new` |
| 今日重生 | data-pull / overlay / clone | HF runtime 401；Q7 不是完整 OpenPI；不能 fetch `b58aff07` | B + 拼树 | skip-existing + 公开 clone 补文件 |
| 1 | pid 5441 | `RevisionNotFoundError` / `LeRobotDatasetMetadata` | C | remap `source_roots`；禁止 `create_tag` |
| 2 | pid 5823 | `IndentationError` after `elif` | D | 恢复 bak；不要 `elif False` 同级注入 |
| 3 | pid 6030 | `AxisPlusV3Dataset` 无 `episodes` | C | 补 property，或走 `_index` |
| 4 | Cell In[47] | 正则误替换 `if sampling_manifest` | D | 未开训；必须 `ast.parse` |
| 5 | Cell In[49] | 把 fallback `LeRobotDatasetMetadata(repo_id)` 换成 16 空格块 | D | 只恢复 bak + 补 property，不再替换 Hub 调用 |
| 6 | pid 6609 | `TrainState` 无 `rng` | 拼树 | 补字段 |
| 7 | 用户 16:42 JST | 训练已成功启动 | — | 留标签；勿 Stop My Server；checkpoint 后 data-push |

可忽略的失败 wandb run：`971ay3vq`、`etxxxaa3`。正式成功 run 以当时 log / wandb 名为准，本文不编造 step。

失败目录命名（NVMe `runs/`）：

```text
t2_p30_r6_h10k_r1.failed_hf_offline
t2_p30_r6_h10k_r1.failed_ipec_hub
t2_p30_r6_h10k_r1.failed_indent
t2_p30_r6_h10k_r1.failed_v3_episodes
t2_p30_r6_h10k_r1.failed_v3_indent
t2_p30_r6_h10k_r1.failed_hub_indent
t2_p30_r6_h10k_r1.failed_trainstate_rng
```

## 8. 本机绿测为什么拦不住

| 检查 | 本机能过 | 官方仍会炸 |
|---|---|---|
| JAX 0.5.3 + Orbax 0.11.13 import | 是 | 官方驱动装不了这套 |
| T2 manifest / norm SHA | 是 | SHA 对，但 root 路径指向 Linux |
| 本地 `episode_*.parquet` loader | 是 | 官方 tar 是 `file-*.parquet` |
| 空 run_dir `--mode new` | 视情况 | 失败残留目录会拒覆盖 |
| nested OpenPI `b58aff07` 完整树 | 是 | 官方只有 overlay 增量 + 公开 HEAD |

因此赛后报告应把“本地 CI 绿”和“官方可训”写成两个门，而不是一个。

## 9. 下次官方机启动清单

开训前只读检查，全部 True 再 `--mode new`：

1. `nvidia-smi` 0 MiB；无第二份 `train.py`。
2. `trainvenv`：`jax==0.11.1`、Flax 补丁、`orbax-checkpoint==0.11.25`、`huggingface-hub==0.36.0`。
3. wrapper `trainvenv_python.sh` 存在；不要 `Path(venv/bin/python).resolve()`。
4. `~/.cache/openpi/big_vision/paligemma_tokenizer.model` 存在；`import gcsfs` 不是硬依赖路径。
5. 六个 dataset root 在 NVMe，且 `chunk-*/file-*.parquet` 为 True；`source_roots.json` 不是 `/home/huang/...`。
6. `data_loader.py` 对 IPEC/Plus 走 `AxisPlusV3Dataset`；有 `episodes` property 或 v3 `_index` 权重分支；`ast.parse` 通过。
7. `TrainState` 含 `rng`；`train.py` 与 `utils.py` 来自同一逻辑树。
8. `run_dir` 不存在或已归档为 `*.failed_*`。
9. 配方仍是 R6 params-only、H10K、manifest/norm SHA 未改。
10. 准备好 persist `data-push` 时机；Mac 不休眠；不点 Stop My Server。

## 10. 赛后报告可用结论

1. 官方 Blackwell 长训的第一道墙是 **JAX 0.11.1 + CUDA 13 与 OpenPI 旧 pin 不兼容**。这不是数据或配方错误；本机 V100 旧栈绿测无法发现。
2. 第二道墙是 **干净环境缺 tokenizer / 缺客户端包 / uv symlink**。这是完备性，不是 minor 升级。
3. 第三道墙是 **官方 IPEC/Plus 是 LeRobot v3 on-disk layout**。Linux `source_roots` 和 v2.1 `LeRobotDatasetMetadata` 会把它伪装成 Hub `RevisionNotFoundError`。
4. 第四道墙是 **操作契约**：空目录拒覆盖、实例休眠丢 NVMe、错误粘贴和错误正则。
5. 今日重生还暴露 **overlay 与公开 clone HEAD 拼树**。`TrainState.rng` 就是这个，不要写进 pip 版本事故。
6. 配方（R6 params-only、H10K cosine、batch 64、LR 3e-7、冻结 SHA）在上述墙被拆掉之前，没有证据表明它是启动失败原因。

## 11. 现场 log：`valid_action_elements=640`（2026-09-13 ~17:10 JST）

pid 6950，`optimizer_step≈1684`，GPU `33439 MiB / 100%`。identity 冻结 SHA 仍对。

```text
examples_seen=64
micro_steps=16
valid_action_elements=640   # 每步不变
temporal_padding_loss_mass=0
padded25_loss_mass=0
seven_d_loss_mass == loss_sum
loss ≈ 0.14–0.19
loss_sum / 640 ≈ loss
nonfinite_* = 0
post_clip_grad_norm = 0.1000
```

640 = 16×4×10。这是 overlay `train.py` 在 **没有** `model.compute_loss_components` 时的 fallback：`valid_action_elements = compute_loss(...).size`，形状是 `(*batch, horizon)`，不是 Axis valid7 的 64×10×7=4480。fallback 还会把 `padded25` / `temporal_padding` **写死为 0**，所以这两项为 0 不能证明数据没有 pad，只能证明没走 masked 合同。

不要因此停训。W&B 上和昨晚曲线的绝对高度不可比：昨晚若走了 `compute_loss_components`，分母是 4480 量级。今晚这条是 `--mode new`，optimizer 从 0 计，本来也不能接到昨晚 1.8k。

