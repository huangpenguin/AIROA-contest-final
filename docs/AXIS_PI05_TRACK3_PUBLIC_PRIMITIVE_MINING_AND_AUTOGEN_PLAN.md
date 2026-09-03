# AXIS Pi0.5 Track 3：公开 Primitive 挖掘与自动逆向轨迹 Canary 计划

> 制定日期：2026-09-03（Asia/Tokyo）
> 计划输入：root `bcf7e1c0deffc9c2eaf7595931f7fb2372d05257`，nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 当前状态：`track3_scripted_executor_failed`
> 唯一活动授权：执行 S0–S5，最多生成 3 条非人工自动 canary，完成后 hard stop 等待审核
> 不授权：人工 Teleop、批量 30–60 条生成、LoRA/dense 训练、norm 替换、FAST/ZIP、C20/E2、官方 GPU、上传或比赛提交

## 0. 目标与路线变更

用户账户内的研究对话“Track3逆向训练方案”提出了三个待独立核验的公开方向：

1. `KE7/libero-infinity`：由公开 LIBERO BDDL 构造 reverse initial/goal/instruction；
2. 原始 LIBERO、LIBERO-plus 与 LIBERO-Gen：挖掘真实成功的 close/open、remove-from-container、remove-from-support 等 reverse-relevant primitives；
3. `real-stanford/behavior_prompting` / LIBERO-Gen：用 waypoint/motion-planning 式控制在 20 Hz `OSC_POSE` 环境中重新执行并保存 HDF5。

这些内容是研究线索，不是已经通过的项目事实。S0 必须从 primary source、固定 commit、许可证和实际代码独立复核；不得把聊天结论直接写成可训练数据 provenance。

本计划把旧路线：

```text
task19 人工 canary → 人工三族 canary → 从人工轨迹提取 planner
```

改为：

```text
公开来源核验
  → 现有成功 primitive 语义挖掘
  → 选择性 replay/兼容性验证
  → reverse BDDL adapter
  → scripted 20 Hz 自动执行三族 canary
  → 仅在明确缺口时另行审核人工兜底
```

人工 recorder 与 replay 工具保留，但本计划不授权启动。现有 `accepted=0` 是正确起点；不得为了宣称进展把 forward primitive、候选 reverse spec 或 finite rollout 计作 accepted reverse demonstration。

## 1. 已锁定事实与数据边界

### 1.1 当前模型与正式分数

```text
candidate                 r6_dual_l2_fresh500_r1 / step 500
topology                  action-expert attention/FFN LoRA rank32
                          + action_in/out_proj + time_mlp_in/out
training data             physical-intelligence/libero
training data FPS         metadata 10 Hz, no resampling
training tasks            40
optimizer steps           500
official score provenance user-reported
Track1 / Track2 / Track3  0.47 / 0.33 / 0
```

R6 是本计划后续训练的冻结起点，不在 S0–S5 中加载或修改。Track3=`0` 只证明当前正式提交没有得分；没有逐 episode 官方日志时不得补造隐藏失败类型。

### 1.2 本地公开数据

```text
original LIBERO HDF5      data/raw/public/libero_official_hdf5/
HDF5 files                40
total bytes               33,784,856,577
full-state canary         task10/task19/task20 fresh-process 3/3

IPEC no-noops 20 Hz       spatial/object/goal/10/90
episodes                  5,614
LIBERO-plus 20 Hz         lerobot/libero_plus@f3f49f4...
episodes                  14,347
local 20 Hz total         19,961 episodes / 3,080,750 frames

evaluation denylist       data/eval_manifests/axis_pi05_eval_v1/public_t3_holdout.jsonl
denylist cases            40
```

`physical-intelligence/libero` 10 Hz canonical 与上述 20 Hz 数据不可混称。比赛 `libero_combined_20hz` 的精确 episode/task remap、payload 与 stats 仍未审计；本计划不假设它等于任意本地拼接，也不启动官方 GPU 下载。

### 1.3 公平性与不可越界项

- 只使用官方公开的“Track 3 是现有 LIBERO 任务的逆操作”高层规则；
- 不读取或利用泄漏脚本中的隐藏 task 细节；
- 40-case public T3 manifest 及其可识别 base template/composition/init 全部 evaluation-only；
- 不把 `inverse_view`、文件名含 `inverse` 或 action 倒序自动解释为 task reversal；
- 不从隐藏评测结果反推训练 task；
- 外部数据必须锁定 source URL、commit/revision、license、schema 和下载 hash；许可不清楚只能用于只读研究，不能进入训练候选池。

## 2. 统一数据身份：三种不同资产不得混淆

每条候选必须且只能属于以下一种：

1. `exact_reverse_demo`：reverse initial/goal/instruction 一致，轨迹在 reverse 环境真实执行并成功；未来可进入专项训练池。
2. `reverse_relevant_primitive`：真实成功轨迹包含所需低层技能，但整体 task 不是当前 reverse spec；只能作为 auxiliary skill/retention 数据，禁止重标成 exact reverse。
3. `generator_reference_only`：视频、keypose、reference action、代码或状态可帮助生成，但自身不满足训练 episode 契约。

任何报告都必须分别给出三类数量。只统计 instruction 关键词、有限 action 或最后一帧视觉相似度不能提升身份等级。

## 3. S0：Release lock 与源声明复核（CPU/网络）

### S0.1 输入锁定

写入 `data/reports/axis_libero_30k/track3_public_autogen/s0_release_lock.json`：

- root、nested OpenPI commit/clean；
- `LIBERO-plus` 既有 dirty porcelain 只读快照，禁止修改/stash/reset/commit；
- R6 raw/folded/FAST/ZIP/norm hashes；
- 40-file HDF5 lock、P1 三族 restore summary、R4 recorder canary hash；
- public T3 denylist hash；
- accepted/pending/rejected 当前计数；
- 本地六个 20 Hz 源的 revision、info/stats hash、episodes/frames/tasks/FPS。

若 root 或 nested OpenPI dirty，或历史 evidence/hash 被改写，立即停止。`LIBERO-plus` 的已知独立 dirty 状态只记录，不得作为清理目标。

### S0.2 Primary-source verification

候选来源至少核验：

```text
KE7/libero-infinity
real-stanford/behavior_prompting
Lifelong-Robot-Learning/LIBERO
austinpatel/libero_gen_goal_chain_hdf5
austinpatel/libero_gen_spatial_combination_hdf5
lerobot/libero_plus
```

研究对话给出的 commit/revision 只能作为 lookup hint。执行者必须通过 Git/Hugging Face primary source确认：

- repository/dataset 真实存在且 commit/revision 可解析；
- license 文件与数据卡实际内容；
- reverse 实现是否真的修改 BDDL init/goal/language；
- generator 是否实际创建 20 Hz `OSC_POSE`、7D action；
- HDF5 是否真实包含 `states/actions/model_file`，而不是仅 README 宣称；
- `inverse_view` 的真实语义；
- 是否有可直接使用的 close-drawer primitive；
- 与锁定 LIBERO/robosuite/MuJoCo/controller 版本的兼容要求。

只获取 README、许可证、源码树/metadata；不得在 S0 下载 159–378 GB 等大型数据集。第三方源码若需检查，克隆到被 `.gitignore` 覆盖的 cache `data/raw/cache/track3_public_sources/`，并把 commit/tree/license hash 写入：

```text
configs/axis_pi05/track3_public_sources.lock.yaml
data/reports/axis_libero_30k/track3_public_autogen/s0_source_verification.json
data/reports/axis_libero_30k/track3_public_autogen/S0_SOURCE_REVIEW_ZH.md
```

### S0 hard gate

只有以下全部成立才进入 S1：

- 至少一个 reverse BDDL/semantic generator 可固定版本复现；
- 至少一个 20 Hz scripted/waypoint collection implementation 可审计；
- 原始 LIBERO HDF5 full-state schema 再次确认；
- 许可与比赛数据使用边界不是 `prohibited`；
- public T3 denylist 可映射且 fail-closed；
- 未下载大型数据、未使用 GPU、未创建训练 episode。

否则状态为 `track3_public_source_claims_unverified` 并停止。

## 4. S1：本地成功 Primitive 语义挖掘（CPU only）

实现可复现的 metadata/task parser，不先解码全部视频。对本地 original HDF5、IPEC no-noops 和 LIBERO-plus 建立 episode/task 索引：

```text
data/manifests/track3_public_primitives/task_catalog.jsonl
data/manifests/track3_public_primitives/episode_catalog.jsonl
data/manifests/track3_public_primitives/coverage_matrix.json
data/reports/axis_libero_30k/track3_public_autogen/s1_semantic_mining.json
```

每个 task/episode 至少记录：source/revision、suite、task index、instruction、BDDL hash（若有）、episode index、FPS、action/state/image schema、是否有 full state、候选 operator、身份等级、license status、denylist overlap。

首轮 operator taxonomy：

```text
articulation_close / articulation_open
containment_remove
support_remove
toggle_off / toggle_on
sequence_inverse
```

分类必须基于 instruction + 可用 BDDL predicates；只有 instruction 时置信度最多为 `semantic_candidate`。重点寻找但不预设一定存在：

- `close/open ... drawer/cabinet`；
- `pick ... in/from drawer/container ...`；
- `pick ... on stove/cabinet/box/ramekin ...`；
- 多阶段任务中可独立切分且边界可由 predicate/state 验证的片段。

禁止：

- 将同一 episode 切成多个片段后在统计中冒充多个独立 demo；
- 仅凭 instruction 文本把 forward task 重标为 exact reverse；
- 让 public T3 denylist 的 template/composition/init 进入训练候选；
- 先随机 episode split；split 单位必须至少是 base template。

### S1 hard gate

进入 S2 至少需要：

- articulation、containment、support 三族各有候选；
- 每族至少 2 个 base templates 或明确记录真实 shortfall；
- `exact_reverse_demo`、`reverse_relevant_primitive`、`generator_reference_only` 分开统计；
- denylist overlap 为 0；
- catalog 重跑 hash exact；
- 不解码/复制全量视频，不创建新轨迹。

若三族候选不足，状态为 `track3_public_primitive_coverage_insufficient` 并停止，不自动下载超大 LIBERO-Gen 数据。

## 5. S2：选择性 schema、视频与 replay 资格验证（GPU3 only）

### S2.1 分层抽样

从每族抽取最多：

- original LIBERO HDF5：2 条 full-state demonstrations；
- IPEC/Plus LeRobot：2 条 episode 的 metadata/action/video；
- LIBERO-Gen：只有 S0/S1 显示能补真实缺口时，选择性下载最多 2 条小型 HDF5，不下载整库。

所有下载需 `.partial` + atomic rename + SHA-256。总新增数据上限 `20 GiB`；超过必须停止等待审核。

### S2.2 不同来源使用不同门禁

`full-state HDF5`：

1. 映射到公开 BDDL/XML/controller；
2. 从记录 state fresh-process restore；
3. replay actual actions；
4. 检查 predicate、state drift、碰撞、非目标对象位移；
5. 每个入选 reference 3/3；跨版本不能用放宽 tolerance 冒充 exact。

`LeRobot-only`：

1. 验证 20 FPS、7D action、8D policy state、front/wrist 对齐；
2. 确认 gripper convention；
3. 解码 H.264/AV1 后生成抽样 H.264 和 contact sheet；
4. 只能授予 `reverse_relevant_primitive` 或 `generator_reference_only`，没有 full init state 时不得声称 deterministic replay。

### S2 hard gate

至少三族各有 1 个可审阅 reference，且：

- full-state reference 的 restore/replay 结果如实记录；
- LeRobot-only 不越权升级身份；
- action finite、schema/FPS/gripper 已锁定；
- failure 视频、task详情、operator、扰动/来源和失败原因组合输出；
- GPU0/1/2 和外部进程未触碰。

失败状态为 `track3_public_primitive_replay_or_schema_failed`。

## 6. S3：Reverse semantic adapter canary（GPU3 only）

不得直接把第三方仓库覆盖进 `LIBERO` 或现有 `LIBERO-plus`。在 root 新建 adapter，把已核验的 reverse 规则映射到当前固定环境：

```text
tools/axis_pi05/track3_reverse_semantic_adapter.py
tests/test_track3_reverse_semantic_adapter.py
data/reports/axis_libero_30k/track3_public_autogen/s3_reverse_specs/
```

只选择三个非 denylist public canary，顺序固定为：

1. `support_remove`；
2. `containment_remove`；
3. `articulation_close`。

每个 spec 必须包含 forward BDDL hash、reverse init predicates、reverse goal predicates、instruction、source region/invariants、generator provenance、`official_track3_claim=false`。

每个环境 fresh-process 3 次：

- reset 后 reverse init 为真；
- reverse goal 为假；
- 20 Hz 静置 20 steps 不漂移/掉落/穿模；
- target/non-target objects、fixture joints 和 robot state 可审计；
- goal predicate 可通过受控测试分别达到 true/false，不能是恒真或 reset-success。

### S3 hard gate

三族必须 3/3 stable 且 predicate nontrivial。任一族失败，状态为 `track3_reverse_semantic_adapter_failed_<family>` 并停止；不得跳过该族凑 3 条。

## 7. S4：Scripted/waypoint 自动执行器适配（CPU + GPU3）

只移植并最小化已核验的通用 primitive；禁止把第三方环境 fork 整体替换当前评测环境。统一阶段机：

```text
observe → precontact → approach → contact/grasp
→ remove/fixture-motion → transit/place → release → retreat
```

要求：

- 每一步根据当前 simulator observation/object/fixture/EEF pose 闭环计算；
- 实际向当前环境执行 clipped 7D `OSC_POSE` action，20 Hz；
- 保存完整 sim state、policy state、front/wrist、predicate/subgoal、contact、deadline、action；
- public primitive 只能提供 relative keypose、phase threshold 或 reference action，不得倒放 action；
- privileged pose 的使用写入 generator provenance；
- no grasp、drop、collision、wrong object、timeout、predicate fail 全部进入 rejected/diagnostic；
- 成功前不得创建 accepted；不得裁掉失败前缀/后缀伪造成功。

CPU 单元测试覆盖 phase transition、action clip、gripper、timeout、abort、atomic write、denylist、identity downgrade 和无目标/shape mismatch。

## 8. S5：三族非人工自动 Canary（GPU3 only）

每族最多生成 1 条候选，不做参数网格或无上限 retry。尝试顺序：

```text
support_remove → containment_remove → articulation_close
```

每条候选须满足：

1. reverse reset 非成功，最终 goal 连续稳定满足；
2. 是当前环境 20 Hz 真实执行产生，不是 state teleport 后直接保存；
3. 7D action finite，gripper 统一为训练契约；
4. 保存 start/end/full trajectory states 与 front/wrist H.264；
5. fresh-process action replay 3/3；
6. 非目标对象位移、碰撞、action saturation、deadline miss 通过预注册阈值；
7. denylist overlap 0；
8. 先进入 `pending_autogen/`，不得 promote 到正式 `accepted/`。

若某族自动执行失败，保留完整失败资产并停止；不得在本计划中自动启动人工 Teleop。特别是 `articulation_close` 失败时，允许状态为：

```text
track3_autogen_close_drawer_gap_pending_review
```

后续审核再二选一：补 close primitive，或恢复归档的 task19 单条人工 canary 作为 reference seed。

若三族均通过，输出：

```text
data/reports/axis_libero_30k/track3_public_autogen/s5_canary_summary.json
data/reports/axis_libero_30k/track3_public_autogen/S5_AUTOGEN_CANARY_REPORT_ZH.md
artifacts/track3_public_autogen/failure_gallery/
```

状态为 `track3_autogen_three_family_canary_pending_human_video_review`。必须 hard stop；本计划不得创建 human approval、promote、批量扩充或训练。

## 9. S6：收尾与审核包（不执行后续研究）

无论在哪个 hard gate 结束，都必须：

- 更新 plan index、handoff Markdown 与 machine JSON；
- 小型 config/manifest/report/test 纳入 Git，大型数据、state、视频、checkpoint 保持 ignored；
- `py_compile` 新工具；
- 运行 targeted tests 与完整 `JAX_PLATFORMS=cpu .venv_axis_pi05/bin/pytest -q tests`；
- `git diff --check`、JSON/YAML parse、root/nested clean；
- 形成 clean commit；
- 明确记录未使用官方 GPU、未训练、未生成 ZIP、未上传提交。

## 10. 通过 S5 后的预注册路线（本计划不授权）

后续审核通过三族视频后，另建新活动计划：

### A. 数据扩充

```text
exact reverse pilot       30–60 条
coverage                  >=3 families, >=6 templates, >=2 scenes
replay                    每条 fresh-process 3/3
auxiliary primitive bank  从公开成功数据分层挖掘，数量单独报告
split                     base-template first
```

辅助 primitive 不得在统计或采样中冒充 exact reverse；批量自动生成失败率、retry 数和 rejected 资产必须保留。

### B. 官方 20 Hz retention audit

训练前再用一次短官方环境会话只审计 `libero_combined_20hz` 的 metadata、task remap、stats、files/hash。若与本地来源等价，不复制 payload；若不等价且磁盘门禁通过，只选择性同步该 combined 数据。该动作不是 S0–S5 的授权。

### C. 从 R6 分叉的小步 LoRA

不从 Axis init 重来。先比较：

```text
TR1  action-expert LoRA only
TR2  R6 topology: action-expert LoRA + action heads + time MLP
TR3  仅当失败诊断为对象/顺序理解错误时，增加低 rank PaliGemma LoRA
steps 25 → 100 → 250 → 最多 500
```

每个训练 mixture 使用独立 20 Hz sampling manifest 与 fresh per-dimension QUANTILES；至少比较两个预注册采样臂，明确分开 exact reverse、auxiliary primitive、standard/Plus retention。选优必须同时看 unseen reverse-template holdout 与 Track1/2 retention，不能只看训练模板成功率。

### D. 发布

只有专项候选在 reverse holdout 非零且 Track1/2 retention 通过，才允许 fold → PyTorch/LeRobot parity → FAST → `/act` latency → locked rollout → ZIP。官方 hidden Track3 分数不能作为逐 checkpoint 调参集。

## 11. 允许的本轮最终状态

```text
track3_public_source_claims_unverified
track3_public_primitive_coverage_insufficient
track3_public_primitive_replay_or_schema_failed
track3_reverse_semantic_adapter_failed_<family>
track3_scripted_executor_failed
track3_autogen_close_drawer_gap_pending_review
track3_autogen_three_family_canary_pending_human_video_review
```

除最后两个 review 状态外，不得从失败阶段跳到人工录制或训练。执行者必须停下，把证据交给下一次人工审核。
