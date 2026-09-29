# PARC2026 正赛：可复用经验与易错点

## 2026-09-29 收尾结论：为新阶段保留什么

- **先验证正式运行环境。** 本地代理分数、本机 GPU 绿测和服务能启动，都不能替代正式 evaluator 下的动作结果。尽早固定依赖、输入输出契约、推理延迟和实际部署路径。
- **把数据来源当作训练门槛。** Track3 最终 reverse 为 `0/4`，D2 lineage/dev 门槛未通过。脚本能完成动作、文件符合 schema，都不等同于可训练、可用于正式评测的数据；每条样本都要能追到录制过程、初始/目标状态与验收结果。
- **区分局部改善与整体收益。** R6 的推理 horizon screen 未通过预设发布门；execute4 ZIP 是用户覆盖风险门后的单次候选。seam blend 可作为平滑线索，不能据此声称 reverse 或整体任务能力提升。以 [R6 T2 closeout](reviews/axis_pi05/R6_T2_LIGHT_SMOOTHING_SCREEN_AND_SEAM_BLEND_RELEASE_20260917.md) 和 [Track3 最终审查](reviews/axis_pi05/TRACK3_R0_R1_G0_FINAL_INDEPENDENT_AUDIT_20260916.md) 为准。
- **新模型比较先固定共同接口。** 固定 LIBERO 数据版本、train/dev 划分、归一化、观察与动作契约、评测 case 和计时口径，再替换模型及其适配/导出层；否则模型差异会和数据、部署差异混在一起。
- **证据与生成物分开保存。** 原始数据、清单、配置、源码、测试和文本/JSON 结论可支撑复核；checkpoint、视频、NPZ 与 ZIP 属于可再生成的大产物，按空间需要清理时保留 manifest/hash 和复现路径。失败结论也保留，历史记录用新版本修正，不覆写旧 attempt。

这份文档只保留已与正赛仓库契约对齐的通用经验。排行榜权重和归一化公式未公开，因此本地结果只能作为公开 example 的工程门禁，不能当作 Omni 正式分数。

## 先看管线，再看模型

训练、推理、评测、打包是四个独立阶段。模型 loss 下降不代表 server 输入输出正确，也不代表提交包能在单张 L4 上启动。每个候选都按以下顺序检查：

1. contract：两路 128×128 uint8 图像、四类本体状态、7 维 float32 delta action。
2. server：`/health`、`/reset`、`/act`，单请求低于 10 秒。
3. 本地 pipeline：固定 seed、episode 数、任务列表和 max steps。
4. validator：zip 结构、依赖、动态 smoke。
5. 官方 Docker E2E：确认离线加载、显存、启动时间和依赖。

## 官方 Blackwell 启动失败不要都写成版本问题

2026-09-13 T2 H10K 在官方 RTX PRO 6000 上启动时，失败至少分成四类，外加 overlay 与公开 OpenPI clone HEAD 拼树。完整底稿：

`docs/AXIS_PI05_OFFICIAL_BLACKWELL_T2_H10K_LAUNCH_ISSUES_20260913.md`

不要事后把 Hub `RevisionNotFoundError`、缺 tokenizer、空 `run_dir` 或 `TrainState.rng` 都归因于 pip minor。本机 V100 旧 JAX 绿测只能证明本机栈，不能证明官方卡可训。

## 环境必须分开

- 训练使用 `examples/pi05_libero_finetune/scripts/setup_train.sh` 建立的 lerobot 0.4.4 专用 venv。
- 评测使用根目录 `setup.sh`/`env.sh` 对齐 CUDA 13、PyTorch 2.11.0+cu130 和 LIBERO。
- 每个终端先执行 `python -V`、`which python`、`python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'`。
- 训练环境和评测环境不要互相 `pip install`；提交的 `requirements.txt` 也不要随意覆盖评测镜像的 CUDA 组件。

## 一次只改一个变量

固定 seed、数据 revision、mix hash、action chunk、推理步数和评测协议；每次实验只改变 steps、LoRA rank、数据配额或一个推理旋钮。候选权重放在独立 output 下，不自动覆盖 `submission_candidate/model_weights`。

## W&B 与秘密

`WANDB_API_KEY`、`HF_TOKEN` 只能放在 shell 环境或未入库的 `.env`。配置只写 `wandb.mode`：`online`、`offline` 或 `disabled`。runner 会拒绝 online 模式缺少 key；训练参数关闭 model artifact，权重只保存在本地 checkpoint。

## 动作与延迟

`train.steps` 是优化器更新次数，不是机器人动作步数；`action_chunk` 是一次生成的动作数量。pi0.5 示例使用 action chunk 10，且一次 `/act` 可能触发一次重推理。重推理请求也必须低于 10 秒。训练机能运行不等于评测用 NVIDIA L4 24GB 能运行。

## 数据问题不要过度自动化

EDA 应记录 task/episode/frame 数、episode 长度、camera key、图像 dtype/range、state/action shape/range、NaN/Inf、episode 边界和 train/validation 重叠。长度异常与 action spike 只进入人工复核列表，不应凭单一阈值自动删除。

## 提交门禁

提交包必须把推理权重、tokenizer 和 server 代码全部放入 zip，评测阶段不得访问外部网络。先运行静态 validator，再运行动态 smoke，再用官方 Docker `evaluate.py` 做 E2E，最后执行 `omni submit --dry-run`。真实提交每 Track 每日一次，上传完成就计数，错误也计数。

## 2026-08-28 项目状态记忆

- 项目负责人确认上一版计划的“第一部分任务”已经完成；具体完成范围、产物哈希和验收结果以同期 `AXIS_PI05_IMPLEMENTATION_STATUS.md`、执行摘要及对应提交为准，不从这句状态声明扩张推断。
- 当前仍有其他 agent 在修改评测与提交相关代码。后续工作应先检查 worktree，保留并适配并行修改，不覆盖尚未提交的文件。
- 上一版官方/本地评分出现各 Track 为 0 的现象；下面只记录目前提供了逐 episode 证据的 Track 1 日志。其他 Track 的 0 分根因必须分别取证，不能直接套用 Track 1 结论。

## 2026-08-28 项目记忆：Track 3 官方披露

以下内容来自主办方补充说明，是当前项目后续设计必须采用的最新公开事实；它取代仓库中把 Track 3 单纯描述为“已知任务组合／未知域”的旧表述：

- 主办方误将内部检查脚本 `check_no_leaks.py` 包含在参赛者配布物中；该文件泄露了 Track 3 的任务设计方向。
- 为恢复参赛者之间的公平性，主办方公开：**Track 3 由现有 LIBERO 任务的逆向操作任务构成**。例如，相对于“把 A 放到 B”，逆向任务可表现为“把 B 移回 A”一类操作。
- 官方仍不公开具体任务内容和任务数量。项目不得把推测出的单个隐藏任务写成已知真值，也不得声称持有完整 Track 3 清单。
- `check_no_leaks.py` 不会从配布物中删除；本战 1 的 Track 3 内容也维持现状，不再因本次事件调整。
- 这项披露改变了 Track 3 的解释口径：它不再是完全未知的组合泛化测试，但具体 task、数量、对象和初始状态仍然未知。

对本项目的约束：

1. T3 训练和本地诊断可以公开地关注“可逆操作、状态恢复、抓取后放回、逆向语言指令理解”等能力，但只能使用公开数据和本地可构造任务。
2. 可以增加 `reverse_task_generalization` 诊断面板；它必须标记为本地 proxy，不得冒充官方 Track 3 manifest 或正式分数。
3. 冻结评测 manifest 时，正向与逆向 case 应按同一 base task 配对，分别记录语言、初始状态、目标状态和成功条件，避免只改 instruction 文本却没有改变环境目标。
4. 任何使用 `check_no_leaks.py` 的工作只允许依据主办方已公开的“逆向任务”这一高层事实；具体任务与数量仍按未公开信息处理。
5. 项目状态、计划、README 或报告若仍写“Track 3 = 未知任务组合”而未提逆向任务，应在相关 agent 的正常修改范围内更新，并注明这是 2026-08-28 后采用的新口径。

主办方提供的沟通入口：技术问题使用[技术问题表](https://docs.google.com/spreadsheets/d/1QRZQd6wp4j9jz4M-SpQeuyfsozQE9hQH97MSLIvz-GU/edit?usp=sharing)，其他问题使用[事务局联系表](https://forms.gle/62YhtxgqSJazRjAj9)。

## 2026-08-28 项目记忆：Track 1 回退到 0 分的日志证据

当前收到的最新日志只直接对应 Track 1，不能据此推断其他 Track 的逐 episode 行为。已观察事实如下：

- evaluator 明确显示评测目标为 `['track1']`；6 个 episode 均执行满 300 steps 后失败，结果为 0/6、得分 0.000。
- server 日志显示 `pi05 SLOW from_pretrained`、`fast=False`，加载耗时约 182.7 秒；运行配置仍为 `chunk=10`、`n_action_steps=8`、`action_ema=0.0`、`rtc=False`。
- 加载阶段出现视觉 embedding 警告，涉及 `paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias`，并报告动态 remap 了 812 个 state-dict keys。
- 提供的历史对照称此前约 0.395 分的提交使用 `model_fast.safetensors` 和 `fast=True`。该对照需要由历史提交包、权重哈希及原始日志再次确认，不能只凭记忆视为完整因果证据。

当前最高优先级假设是：本次提交包没有包含或没有选择正确导出的 `model_fast.safetensors`，SLOW 动态 remap 又存在视觉塔 key 覆盖或语义不等价，导致策略退化。但日志中的 warning 和 0/6 只说明强相关，**尚未证明机器人确实“失去视觉能力”，也未证明切换 FAST 后一定恢复 0.395**。action convention、图像方向、state/norm、checkpoint 选择、chunk/reset 状态等仍需同时排除。

任何修复版提交在重新评分前必须满足以下门禁：

1. 包内明确存在 `model_fast.safetensors`，记录文件大小和 SHA-256；启动日志必须确认实际走 `FAST load model_fast.safetensors`，不能仅凭文件存在判断。
2. FAST 导出工具输出完整 key coverage 报告；缺失、unexpected、shape mismatch 均为零，视觉塔 patch embedding 的 weight 和 bias 必须单独列入检查。
3. 用固定的真实 LIBERO observation fixtures 比较导出前 OpenPI policy 与 FAST policy：预处理后的输入、首个 action chunk、有限性、shape、符号和量纲均需核对，并设置预声明数值容差。
4. 固定 `n_action_steps=8`、`action_ema=0.0` 和其他已知配置，只切换 FAST/SLOW 权重路径做同 case A/B；不得同时修改多个推理变量后把恢复归因于 FAST。
5. 检查 `/reset` 是否清空 action chunk、两路相机是否方向正确、M1 norm 是否与 checkpoint 匹配、对外 action 是否保持 LIBERO 7 维表示。
6. 先跑单 task paired smoke，再跑完整本地固定 manifest；必须保存失败 H264、task、instruction、init-state、seed、动作轨迹和加载报告。
7. 只有 FAST 路径通过本地等价性和固定 manifest 后才能重新打包；不得把“server 能启动”或“`/act` 返回 200”当作策略正确。
8. 结果报告分别写明 `observed`、`hypothesis`、`verified_cause`。在 FAST/SLOW 严格 A/B 完成前，根因状态保持 `hypothesis`。
