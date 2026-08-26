# PARC2026 正赛仓库部署与迁移计划

## 摘要

当前工作区是 `/home/huang/code`；执行阶段把正赛仓库克隆为 `/home/huang/code/PARC2026_final`，旧仓库作为只读参考保留。SSH 权限已验证，规划时官方 HEAD 为 `2059989571d8a7da3526b994756eb77b748d37af`。

全程遵循“正赛仓库与正赛文件优先、旧仓库只提取通用经验”的原则。先完成只读审计，之后才写入计划、审计文件或迁移内容。

## 1. 克隆与只读审计

- 确认目标目录仍不存在，然后执行：

  ```bash
  GIT_LFS_SKIP_SMUDGE=1 git clone --no-recurse-submodules \
    git@github.com:matsuolab/PARC2026_final.git \
    /home/huang/code/PARC2026_final
  ```

- 记录实际 HEAD、分支、remote 和初始 `git status`。若 HEAD 已不同于规划时提交，只记录变化并按最新官方提交重新审计，不强制回退。
- 不初始化 submodule、不拉取 LFS 权重；只有官方 README 明确要求且完成安全审计后再逐项处理。
- 首先完整阅读根目录及相关子目录中的 `AGENTS.md`，然后审计 README、docs、Docker/依赖文件、训练示例、policy server、pipeline、evaluator、validator、打包及 Omnicampus 文档。
- 以以下顺序处理规则冲突：最新官方 Slack/Omnicampus通知 → 当前正赛仓库 → [正赛 PDF](/home/huang/.codex/attachments/39d516d4-19ba-4bd3-bae8-17ad0e29ed1e/PARC2026開発コンペティション_本選.pdf) Version 1.0 → 对应 TXT → 旧仓库。冲突不得自行猜测，必须进入风险清单。
- 从正赛代码中逐项提取并附文件路径和行号：基线模型结构、dataset/schema/revision、训练入口、state/action 字段、dataset/evaluator/model 三层 camera key、policy server 接口、evaluator、计分、打包、Python/torch/transformers/CUDA 版本。
- 完成只读审计后，首批写入根目录：
  - `MIGRATION_PLAN.md`：保存本计划。
  - `MIGRATION_AUDIT.md`：官方契约表、新仓库已有能力、迁移矩阵、不兼容项、风险、拟迁移清单。
- `MIGRATION_AUDIT.md` 未完成前，禁止从旧仓库复制或改写任何内容。

## 2. 迁移实施

### 文档与经验

- 从旧仓库提炼原则，重写而非整文件复制：
  - `docs/EXPERIENCES_AND_PITFALLS.md`
  - `docs/BEGINNER_WORKFLOW.md`
  - `.cursor/project-context/decisions.md`
  - `.cursor/lessons-learned/`
- 只保留管线先于模型、单变量实验、环境隔离、候选不自动激活、依赖版本一致性、W&B 凭据与布尔参数分离等通用经验。
- 删除旧绝对路径、旧 Track1/Track2 命令、任务表、模型/数据 revision、历史分数和旧环境版本；只有正赛仓库确认仍有效的事实才能重新写入，并标注正赛证据。
- 默认不直接迁移旧 evaluator、scoring、manifest、policy vendor、打包脚本和 Track 专用启动脚本。

### EDA 与抽样

- 新建 `notebooks/00_eda_and_contract_check.ipynb`，以正赛官方 loader/schema 为唯一数据入口，不硬编码本机路径或旧 revision。
- Run All 默认纯只读：不下载、不训练、不启动 server、不生成 mix、不写报告或修改数据；缺少数据时明确失败，不回退到旧路径。
- 完整扫描元数据及 state/action shard；图像检查采用每 episode、每 camera 的首/中/末帧确定性采样，同时核对所有视频文件和帧边界。
- 至少检查任务/episode/frame 数量、episode 长度、camera key、图像 shape/dtype/range、state/action shape、动作范围、gripper 编码、NaN/Inf、episode 边界、task/category 分布和训练/验证泄漏。
- 对 tar 分发数据记录源 tar SHA-256；对 Git/HF 数据记录 revision，并生成相对路径、大小和内容哈希组成的确定性 manifest hash。
- 抽样逻辑从旧代码提取思路后按新 schema 重写：固定 seed、任务×官方类别分层、不跨任务借配额、显式报告不足桶、训练/验证 episode 零重叠。没有官方类别时，派生统计只能标记为 `proxy`，不能伪装成官方标签。
- 长度异常和 action spike 只进入人工复核清单，不能自动判定坏数据。

### 训练配置与 W&B

- 保留正赛官方训练入口，配置 runner 只负责验证和组装命令，不重写训练核心。
- YAML 保存完整实验：模型及 revision/hash、dataset/schema/revision/hash、mix hash、seed、优化器/调度器、batch/gradient accumulation、precision、LoRA/可训练参数、checkpoint、推理 chunk 参数、W&B 和输出目录。
- CLI 覆盖仅允许：`run.tag`、`run.output_dir`、`run.seed`、`train.steps`、`wandb.mode`、`hardware.gpu_ids`；其他字段必须修改 YAML，未知字段直接报错。
- `--dry-run` 必须无写操作，只打印脱敏后的完整命令、环境变量和 resolved config。
- 正式运行拒绝已有输出目录、脏工作树、hash 不匹配和未通过的数据契约；smoke 可由 YAML 显式允许脏树，但必须保存 diff hash。
- 每次运行保存 `resolved_config.yaml` 和 `run_manifest.json`，包含 git commit、dataset/model/mix hash、seed、run tag、环境版本和 W&B URL。
- W&B 默认 online；断网时由 `wandb.mode=offline` 显式切换，不静默降级。记录 loss、learning rate、step、gradient norm、验证指标；关闭 model artifact、checkpoint、权重、视频和大型文件上传，API key 不进入配置、代码、日志或 Git。

### Contract、评测与打包

- 若正赛已有机器可读 contract，测试直接引用；否则新增 `contracts/official_policy_contract.yaml`，记录三层 camera 映射、图像 dtype/range、state/action schema、gripper、chunk、server codec/endpoint/timeout。
- Contract 测试覆盖 `/health`、官方要求的 `/reset`、单次 `/act`、真实 observation fixture、action dtype/shape/finite/range，以及 chunked policy 的 `chunk_size >= n_action_steps`。
- 正赛已确认 action 为 `(7,) float32`，顺序为 `[dx, dy, dz, droll, dpitch, dyaw, gripper]`；state 维度和 camera key 必须等待仓库审计，不从预选赛推断。
- 正式计分只记录官方公式：

  ```text
  smooth = w1*time + w2*jerk + w3*SPARC + w4*trajectory + w5*rotation
  Total Score = mean(success × (1-collision_penalty) × smooth)
  ```

  权重与归一化公式非公开，禁止复制旧公式或把代理分数称为正式分数。三个 Track 的 Total Score 相加。
- 评测 wrapper 只能调用官方 evaluator，不能修改任务、成功/碰撞判定或评分代码。
- 打包只遵循正赛 submission template 和 validator：权重必须同包离线可用、zip 小于 20GB、无嵌套压缩包、软/硬链接、路径穿越、zip bomb 或不必要混淆。
- 实际 `omni submit` 会消耗每日额度，必须先执行官方 validator、Docker 端到端测试和 `omni submit --dry-run`；每个 Track 的真实提交都需要用户单独确认。

## 3. 部署罗盘与门禁

```text
官方规则审计
→ EDA
→ 输入输出 contract
→ 训练/验证数据验收
→ 10–100 step smoke
→ 正式训练
→ W&B 指标确认
→ checkpoint 本地加载与 hash 校验
→ policy server /health
→ /reset + 单次 /act smoke
→ 单 episode 官方 evaluator
→ 固定协议正式本地 evaluator
→ 官方 Docker/validator
→ omni submit --dry-run
→ 用户确认后真实打分/提交
→ 配置、日志、JSON、zip hash 归档
```

任一门禁失败即停止后续阶段；训练不得自动覆盖或激活 submission 目录。评测 observation 不保存为训练数据，server 日志不得记录完整评测图像或 observation payload。

每次评测归档：

- resolved config、git commit/dirty 状态
- model revision及树 hash
- dataset revision、源 tar/hash、mix hash
- seed、evaluator commit/hash
- 推理参数和 contract hash
- server log、评测 JSON
- 环境版本和冷启动/推理耗时
- zip 内容 manifest 与 SHA-256
- W&B URL（如使用）

## 4. 验证与迁移报告

- 运行正赛仓库规定的 lint、import、unit test、训练 dry-run、notebook 执行测试和官方 submission validator。
- Notebook 在 `/tmp` 执行，执行前后确认 Git 状态和 dataset hash/mtime 不变。
- 测试 runner 的未知字段、越权 override、已有输出目录、纯只读 dry-run、秘密脱敏、online/offline W&B 和 artifact 禁用。
- 测试 sampler 的确定性、配额不足、episode 零重叠和 revision/hash 绑定。
- 在官方 CUDA 13/PyTorch 2.11.0+cu130 Docker 中做冷启动验证；目标评测设备是单张 NVIDIA L4 24GB，峰值显存需留安全余量，每次推理低于10秒，依赖安装至 Track 完成低于1小时。
- 迁移前在 `MIGRATION_AUDIT.md` 列出每个源/目标文件、用途、兼容证据和处理方式。目标存在时不覆盖，先在临时目录生成候选并输出 `git diff --no-index`。
- 最终生成 `MIGRATION_REPORT.md`，记录实际新增/改写/未迁移文件、测试结果、未验证假设、人工确认项及所有迁移 diff。

## 假设与已知风险

- 旧仓库当前存在大量用户未提交改动，执行期间只读访问，禁止清理、提交或重置。
- 正赛训练资源为 RTX PRO 6000 Blackwell；正式评分资源为单张 L4 24GB，不能因训练机可运行就判定提交可运行。
- 正赛 PDF 未给出完整 state、camera、Python 和 transformers 契约，这些必须以克隆后的仓库为准。
- PDF 的本选3截止时间与报告截止日期存在明显矛盾；不得把报告日期写入操作日历，需以最新 Slack/Omnicampus通知人工确认。
- 若仓库 README、代码和 Version 1.0 PDF 对 dataset revision、依赖或提交方式存在冲突，暂停相关迁移并列入 `MIGRATION_AUDIT.md`，不沿用预选赛经验补空白。
