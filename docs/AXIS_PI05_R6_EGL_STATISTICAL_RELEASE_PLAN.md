# AXIS Pi0.5 R6：EGL 统计评估、FAST 与诊断 ZIP 续执行计划

> 制定日期：2026-08-31（Asia/Tokyo）
> 输入提交：`be56e8edca77377b57a2941d0483564ce4112f79`
> 输入状态：`s1_real_egl_pairs_failed`
> 前置计划：`AXIS_PI05_R6_FOLDED_DEPLOYMENT_AND_DIAGNOSTIC_SUBMISSION_PLAN.md`
> 目标：保留原 `1e-6` sensitivity failure，不事后放宽或改写；修正 S1.3 证据语义后，预注册统计 closed-loop 协议，再决定是否恢复 FAST/ZIP 部署链。
> 非目标：不训练、不重选 LoRA、不启动 V160/H160/30K、不自动上传或提交比赛。

## 0. 给新 section 的执行摘要

按以下顺序执行，任一 hard gate 失败立即停止：

```text
E0 修正 S1.3 报告语义与 pair coverage（不重写旧 failure）
  → E1 完整 content-unique EGL action-response envelope
  → E2 在读取新 rollout 前锁定统计协议
  → E3 production seed 修复 + source PyTorch 三副本 C20
  → E4 人工/机器统计门禁决定是否恢复部署链
  → E5 R6 FAST export + source↔FAST parity
  → E6 FAST 三副本 C20 非劣验证
  → E7 ZIP/L4/Blackwell pre-check
  → 停止，等待人工提交决定
```

执行者不得从头重跑已经完成的 canonical 下载、R6 500-step 训练、Q0/Q1/Q2 或 S1.2。旧产物只读；所有修正版产物写入新 namespace。

## 1. 不可改写的既有结论

以下状态永久保留：

```text
raw-LoRA Q2:
  n1_q2_policy_sensitivity_failed_raw_lora_fresh_replay
  max_abs_diff = 1.0728836059570312e-6

LeRobot folded S1.3:
  s1_real_egl_pairs_failed
  original gate = external action max <= 1e-6
  observed max = 5.125999450683594e-5
```

新计划不得：

- 将旧 summary 的 status 改为 pass；
- 删除或覆盖旧 pair manifest/E2E artifact；
- 通过修改 seed、重抽 pair、round image/action 或改 normalization 获得“通过”；
- 把新统计协议称为旧 `1e-6` 门禁通过。

S1.3 的全部动作 finite、gripper sign flip 为 0，且 R6 folded JAX 与 on-disk LeRobot folded PyTorch 呈同量级 drift。该结果落在更早 N1 计划已经定义的 `(1e-6, 1e-4]` **人工统计协议审核区间**；`1e-4` 只能用于决定是否值得进入统计 closed-loop，不能替代旧 `1e-6` 结果。

## 2. 当前 S1.3 需要修正的三项证据问题

### 2.1 reference 标签错误

当前 `axis_reference_jax_*` 实际读取 `folded_jax_external_actions`：这是 **R6 folded JAX**，不是 Axis step-0。所有新报告统一改名为：

```text
r6_folded_jax_reference_*
```

若需要 Axis step-0，必须独立加载 Axis params 并写出不同 backend/model hash，禁止只改 label。

### 2.2 agentview representative 与 worst 内容重复

当前两个 agentview pair 的路径/replica 名不同，但左右图像内容 SHA 完全相同。新选择器必须以语义 key 去重：

```text
case_id
phase_or_trajectory_frame
camera
policy_state_hash
prompt_hash
canonical(min(left_image_sha), max(right_image_sha))
```

同一上下文、同一左右内容 pair 只能出现一次。若没有第二个 distinct nonzero pair，报告 `representative_unavailable`，不得复制 worst。

### 2.3 trajectory coverage 缺失

当前选择器只遍历 E1/E2/E3 snapshot，没有读取 Q1 的 zero64/fixed64 trajectory。新工具必须覆盖：

```text
E1 / E2 / E3-1..10
zero64 每个存在图像差异的对应 frame
fixed64 每个存在图像差异的对应 frame
```

只有 simulator/qpos/qvel/8D policy state 在对应 replica/frame exact 的 pair 才能进入 renderer-only envelope；否则单独标为 dynamics mismatch 并 hard stop。

## 3. 全局执行边界

- 输入 root commit 固定为 `be56e8e...`；执行 E0 后每阶段独立 clean commit。
- nested OpenPI 固定 `dafa5a27266996eecf28920cacc32f3629776d91`。
- GPU0 有外部 PID 时不得触碰；GPU run 前/中/后保存 monitor evidence。
- environment、EGL、policy subprocess 必须使用同一受限 GPU namespace；统计 replica 固定在同一物理 GPU 上顺序运行，不用不同 GPU 冒充 fresh-process replica。
- 固定 native 10 Hz、direct init、10-step warm-up、execute8、max 300、C20 manifest、env/policy seed、R6 fresh QUANTILES。
- 不使用 BF16；不切换 OSMesa/Vulkan；不修改 camera、action/state、chunk、EMA、RTC。
- 正确测试入口固定为：

```text
.venv_axis_pi05/bin/python -m py_compile tools/axis_pi05/*.py pipeline/*.py
.venv_axis_pi05/bin/pytest -q tests
git diff --check
```

当前基线为 `99 passed, 7 skipped`。`pynvml` 不是当前测试 collection 依赖；若其他入口报告缺失，先记录 Python/pytest path，不安装包掩盖环境错误。

新产物 namespace：

```text
data/reports/axis_libero_30k/post_r6_lora/r6_egl_statistical_release/
outputs/axis_fast_recovery/r6_step500_egl_statistical/
artifacts/submission_axis_r6_step500_egl_statistical_<timestamp>.zip
```

## 4. Phase E0：报告修正与旧证据只读重分析

### E0.1 工具与 schema 修正

修复 `tools/axis_pi05/r6_s1_egl.py`：

- 将错误的 Axis label 改为 R6 folded JAX；
- 以第 2.2 节 semantic key 去重；
- 加入 trajectory frame 枚举；
- 分开 `snapshot_pair_count`、`trajectory_pair_count`、`content_unique_pair_count`；
- 报告 unavailable，而不是复制 representative；
- 每个 pair 保存 state/prompt/image/replica/frame hash 与 raw/transformed diff。

新增单测：

- 相同图像内容、不同路径/replica 必须去重；
- folded JAX 不得标成 Axis；
- zero64/fixed64 有差异时必须进入 coverage；
- dynamics 不 exact 的 frame 不得被归类为 renderer-only；
- 修正报告不能改变旧 S1.3 status/hash。

### E0.2 只读 reanalysis

不重新运行模型，基于旧 NPZ/JSON 生成：

```text
e0_original_s1_reanalysis.json
E0_AUDIT_REPORT_ZH.md
```

明确记录：旧 failure 仍有效；原四 pair 中 content-unique 数量小于 4；reference 是 R6 folded JAX；trajectory 未完成。

E0 通过状态：

```text
s1_original_failure_preserved_reporting_corrected
```

## 5. Phase E1：完整 EGL action-response envelope

### E1.1 冻结完整 pair manifest

从 Q1 三副本的 1680 项比较中枚举全部 87 个 non-exact comparison group，并展开所有受影响 trajectory frame。按 semantic key 去重后生成只读 manifest；在看到新模型 action 前冻结 commit/hash。

每个 pair 必须满足：

```text
same case/prompt/frame semantics
simulator/qpos/qvel/8D state exact
only one selected camera changes
other camera exact and held fixed
fixed policy seed/noise
raw arrays and transform reproducible
```

若 Q1 数据不足以恢复某 trajectory frame 的对应 state/prompt，标为 `not_evaluable` 并报告数量；不得伪造 zero state 或借用其他 frame state。

### E1.2 批量 policy sensitivity

对全部 evaluable、content-unique pair 同时运行：

1. R6 folded JAX reference；
2. on-disk LeRobot folded PyTorch source。

输出 camera/stage/trajectory/per-dim/per-horizon 的 max、p50/p95/p99、sign flip、finite、JAX↔PyTorch drift correspondence。

进入 E2 的必要条件，不等同于旧 S1.3 pass：

```text
coverage/report completeness                  true
all actions finite                            true
token/mask/state contract mismatch            0
gripper sign flip                             0
任一 backend EGL-induced external max         <= 1e-4
JAX folded ↔ PyTorch source max               <= 1e-3
```

若 EGL-induced max `>1e-4`、出现 gripper sign flip 或 backend divergence，状态为 `egl_policy_sensitivity_unbounded` 并停止。

E1 通过状态只能写：

```text
egl_action_response_bounded_pending_statistical_closed_loop
```

## 6. Phase E2：统计 closed-loop protocol lock（CPU，先于 rollout）

在读取任何新的 source/FAST closed-loop 结果前生成并提交：

```text
EGL_STATISTICAL_PROTOCOL.lock.json
E2_PROTOCOL_REPORT_ZH.md
```

### E2.1 固定面板与 replica

```text
panel                       C20，20 个冻结 case
backend                     R6 folded JAX / LeRobot folded PyTorch source
fresh-process replicas      每 backend 3 个
environment                 同一物理 GPU、相同 EGL/runtime
env/policy seed             原 C20 manifest，不改变
protocol                    execute8
total                       20 × 3 × 2 = 120 episodes
```

replica 是 process/runtime 重复，不是新 seed；不得因结果不好增加第 4/5 个 replica。

### E2.2 预注册指标与门禁

基础门禁：

```text
coverage                    每 backend/replica 20/20
HTTP/runtime error          0
non-finite/stall            0
contract/hash drift         0
```

renderer/runtime 稳定性按 backend 分别报告：

```text
三 replica aggregate success range       <= 3/20
collision range                           <= 3/20
timeout range                             <= 3/20
per-case success flip count               报告，不隐藏
```

PyTorch source 对 R6 folded JAX 的非劣门禁，以 60 个 `(case, replica)` paired observations、task 为 cluster、固定 10,000 bootstrap draws/seed 计算：

```text
point success delta                       >= -0.05
task-cluster bootstrap lower bound        >= -0.10
collision delta                           <= +0.10
timeout delta                             <= +0.10
```

CI 跨 0 只能称为“在当前 renderer variance 下本地非劣”，不能称为显著等价或提升。

## 7. Phase E3：production seed 修复与 source statistical rollout

### E3.1 最小 seed 修复

只移植 Q1 已验证的顺序：constructor 前 Python/NumPy/Torch case-local seed，constructor 后 `env.seed`，reset 前再次设置同一 seed。保持 direct init、warm-up、10 Hz 与 action contract 不变。

新增调用顺序、case isolation、GPU namespace 测试。先运行 C5 environment-only 两 fresh process，数值动力学必须 exact。

### E3.2 真实 source backend adapter

closed-loop 的 PyTorch backend 必须实际读取锁定的 LeRobot `model.safetensors` 与 processors，通过 HTTP policy contract运行；不得用 JAX folded 结果复制或冒充 PyTorch source。

按 E2 固定顺序运行 120 episodes。保存：

```text
summary.json / tasks.csv / episodes.csv
raw/transformed boundary hashes
action traces
server logs and latency
H.264 videos + failure gallery
task/instruction/init/env-policy seed/failure reason
GPU monitor evidence
```

E3 不允许边跑边改 E2 阈值。

## 8. Phase E4：统计判定与部署链恢复点

只使用 E2 预注册脚本生成结果；人工不得挑选有利 replica。

- E2 门禁全部通过：状态 `r6_source_statistically_robust_enough_for_fast_export`，允许进入 E5。
- 基础门禁失败：`source_runtime_or_contract_failed`，停止。
- renderer range 或非劣门禁失败：`source_closed_loop_unstable_under_egl`，停止，不导出 FAST。

无论结论如何，旧 raw-LoRA Q2 与旧 S1.3 status 都保持 failed。

## 9. Phase E5：R6 FAST export 与固定输入门禁

恢复原计划 S3/S4，但使用新 namespace，并满足：

- 唯一 source 是 R6 on-disk folded model SHA `5c73e1e8217dbb75a934c2b0165208e851f69e66cfe2ea0430590a29901e6d5b`；
- 禁止旧 F2/M1/旧仓库 `model_fast.safetensors`、processor、norm；
- strict key、duplicate mapping、vision patch weight/bias、atomic write/reload 全部通过；
- 8×4 source PyTorch↔FAST fixed-noise grid；
- finite、shape/token/mask exact；
- median external error `<=0.03`、p95 `<=0.10`；
- full-chunk gripper sign agreement `>=0.95`，execute8 sign flip `0`；
- 对 E1 content-unique EGL manifest 重新运行 FAST sensitivity，不能只跑原来四 pair。

失败立即停止，不进入 server/ZIP。

## 10. Phase E6：FAST statistical closed-loop

使用 E2 同一 C20、同一三 replica、同一 GPU/runtime；只新增 FAST backend 60 episodes，不重跑或替换已经冻结的 source PyTorch 结果。

基础门禁同 E2。FAST 相对 source PyTorch：

```text
point success delta                       >= -0.05
task-cluster bootstrap lower bound        >= -0.10
collision delta                           <= +0.10
timeout delta                             <= +0.10
HTTP/reset/act >=10s                      0
stall/non-finite/gripper contract error    0
```

server 必须记录 `FAST_REQUIRED_AND_VERIFIED, fast=True, chunk=10, execute=8, EMA=0, RTC=false`，禁止 SLOW/remap/vision warning。

通过状态：

```text
r6_fast_statistically_noninferior_local
```

这仍不是官方得分或正式 winner。

## 11. Phase E7：ZIP、L4 与官方 Blackwell pre-check

沿用原计划 S6/S7：

- ZIP `<20GB`，只有唯一 `model_fast.safetensors`，无 `model.safetensors`；
- CRC/testzip/SHA、root layout、offline import、官方 static validator 通过；
- 正式 scorer 是 L4 24GB，单请求 `<10s`；V100/Blackwell 能运行不能替代 L4 资源预算；
- 配布 Blackwell 只使用官方 pre-check Docker；大 ZIP 外传、HF private upload、启动 Notebook 前必须人工授权；
- Blackwell 通过后仍不得自动调用比赛提交接口。

最终允许状态：

```text
r6_step500_egl_statistical_zip_ready_for_human_submission
```

manifest 必须写明：R6 B80 `49/80` 对 Axis `48/80`、CI 跨 0、raw-LoRA Q2 failed、strict S1.3 failed、统计 EGL 分支通过与否、formal winner=false、official result=false。

## 12. 全局硬停止条件

- root/nested dirty 或输入 artifact/hash 漂移；
- GPU0/外部 PID GPU 被触碰；
- 旧 failure 被覆盖、删除、改阈值或重抽 seed；
- pair 语义重复、trajectory coverage 虚报、JAX backend 标签错误；
- E2 protocol lock 在新 rollout 之后才写；
- action rounding、image rounding、cache action 或复用 trace；
- non-finite、sign flip、contract mismatch、HTTP error/timeout；
- E1/E3/E5/E6 任一预注册门禁失败；
- 使用旧 FAST/processor/norm 或 SLOW fallback；
- 未授权 push/HF upload/Notebook/比赛提交。

停止时必须更新机器 summary、中文报告和 `PROJECT_HANDOFF_20260831`，列出 first failure、未执行阶段与准确状态。

## 13. 新 section 的第一条执行指令

只执行 E0：修复 report schema、pair content dedupe、trajectory coverage 和测试，生成只读 reanalysis，形成 clean commit后停止审核。不得在同一轮启动 GPU、E1 model run、closed-loop、FAST 或 ZIP。
