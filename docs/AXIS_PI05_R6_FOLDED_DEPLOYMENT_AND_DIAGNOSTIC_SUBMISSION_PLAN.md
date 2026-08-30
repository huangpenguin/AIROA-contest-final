# AXIS Pi0.5 R6：Folded 部署链恢复与诊断提交执行任务

> 制定日期：2026-08-30（Asia/Tokyo）  
> 输入提交：`12761d3e6652df8c4c4f59b137282e44a821d085`  
> 输入状态：`n1_q2_policy_sensitivity_failed_raw_lora_fresh_replay`  
> nested OpenPI：`dafa5a27266996eecf28920cacc32f3629776d91`  
> 目标：不重新训练，在保留 raw-LoRA Q2 失败的前提下，验证真正的 `LeRobot folded PyTorch → model_fast → policy server → ZIP` 部署链。  
> 最终边界：只生成 R6 step-500 **诊断提交候选**；不得自动上传比赛、不得声称正式 winner、不得启动 V160/H160 或长程训练。

## 0. 给执行 Agent 的任务摘要

从 clean commit `12761d3` 开始，完成下列串行阶段；每阶段满足门禁并形成 clean commit 后才能继续：

```text
S0 前瞻性 release lock，保留 raw-LoRA failure
  → S1 真实 LeRobot folded 的 frozen/EGL sensitivity
  → S2 production evaluator seed 顺序修复
  → S3 独立 R6 source staging 与 FAST export
  → S4 folded-PyTorch ↔ FAST parity grid
  → S5 FAST policy server + local closed-loop
  → S6 ZIP、静态 validator 与 L4 资源门禁
  → S7 官方配布 Blackwell pre-check Docker
  → 停止，等待人工决定是否提交
```

执行者不是仓库中唯一工作者。不得 reset、stash、覆盖其他人的修改或清理旧 outputs/artifacts。不得以重跑 seed、修改 normalization、action rounding 或放宽旧阈值来把 raw-LoRA Q2 改写为通过。

## 1. 审核事实与本计划的语义修正

### 1.1 Q0–Q2 当前证据

- Q1 full-C20 三个 fresh-process replica 的 simulator/qpos/qvel/8D policy state、zero64/fixed64 数值动力学 20/20 exact。
- Q1 存在 bounded EGL jitter：raw uint8 最大差 2，transformed float32 最大差约 `0.0156863`，但差异稀疏。
- Axis step-0 frozen replay 最大 action 差 `5.9604645e-7`，通过原 `1e-6` 门禁。
- 当前 Q2 `folded_dense` 最大差 `3.5762787e-7`，通过原门禁。
- raw-LoRA 最大差 `1.0728836e-6`，只有一个 float32 元素超过 `1e-6`；位置为 `C20-task04-init0-s0`、horizon 8、gripper 维，无 sign flip、non-finite、shape 或 dtype 错误。
- raw-LoRA 的失败继续保留为：

```text
n1_q2_policy_sensitivity_failed_raw_lora_fresh_replay
```

不得在本计划中将其改成 pass。

### 1.2 当前所谓 folded Q2 不是最终部署运行时

`tools/axis_pi05/n1_q2_policy.py` 中的 `folded_dense` 是从 raw Orbax 恢复参数后，在 JAX 内存中调用 `fold_lora_in_mapping()` 再创建 dense JAX model。文件虽定义：

```text
outputs/axis_pi05/axis_libero_lora_folded/
  r6_dual_l2_fresh500_r1_step500/
```

但该目录没有被 `_load_model()` 实际读取。因此现有 `3.576e-7` 只能证明 **JAX folded fresh-process**，不能证明磁盘上的 PyTorch/LeRobot source 或最终 FAST runtime。

已有单 fixture E2E 是积极证据，但覆盖不足：

```text
JAX folded ↔ PyTorch folded max abs       5.6624413e-7
external action max abs                   4.1723251e-7
token/mask mismatch                       0
normalization                             QUANTILES
```

本任务必须扩成多 task、多 seed、真实 EGL pair 和最终 FAST，不得把单 fixture 结果当作 release gate。

### 1.3 冻结的部署输入

唯一允许的 R6 folded source：

```text
outputs/axis_pi05/axis_libero_lora_folded/
  r6_dual_l2_fresh500_r1_step500/
```

已知锁定值：

```text
model.safetensors sha256   5c73e1e8217dbb75a934c2b0165208e851f69e66cfe2ea0430590a29901e6d5b
norm_stats sha256          4c803bf59c52677c36b6b420c614d8a7742e0847708443863e13e67cb80b18f5
normalization              ACTION=QUANTILES, STATE=QUANTILES
chunk_size                 10
n_action_steps             8
external action            7D
internal action            32D
```

执行 S0 时必须重新计算这些 hash；任一不符立即停止，不得自动采用“最新文件”。

### 1.4 明确禁止的旧资产

以下内容只作失败历史证据，禁止进入 R6 staging：

```text
artifacts/submission_axis_f2_m1_500_fast_*.zip
outputs/axis_fast_recovery/f2_m1_500_fast/staging/model_weights/model_fast.safetensors
任何旧仓库 model_fast.safetensors
M0/M1/M2 旧 processor 或旧 norm
```

旧 F2/M1 FAST readiness 为 `NOT_READY`，source↔FAST parity 曾出现：median `0.0526`、p95 `0.538`、gripper sign agreement `0.70`。它的 key audit/ZIP hash 通过不代表行为正确。

## 2. 全局边界与产物 namespace

- GPU0 有外部 PID 时不得触碰；本地只使用人工确认空闲的 GPU1–3。
- 每次 GPU run 记录 before/active/after PID、显存、利用率；证据不完整时不得写 `gpu0_touched=false`。
- root/nested worktree 在每个阶段开始前必须 clean；保存 root/nested commit、branch、dirty 状态。
- 不训练、不改 checkpoint、不改 norm、不改 action contract、不改 `chunk=10/execute8/EMA=0/RTC=false`。
- 不使用 BF16；本地 V100 使用 FP32/JAX 或经过声明的 PyTorch FP16/FP32 inference。
- 所有新产物只能写入：

```text
data/reports/axis_libero_30k/post_r6_lora/r6_folded_submission/
outputs/axis_fast_recovery/r6_step500_folded_diagnostic/
artifacts/submission_axis_r6_step500_folded_diagnostic_<timestamp>.zip
```

- 旧 Q0–Q2、R6 raw checkpoint、folded source、旧 ZIP 一律只读。
- 不自动 `git push`、上传 Hugging Face、启动官方 Notebook 或调用比赛提交命令。S7 前必须停一次，等待人工授权外部传输与配布 GPU 计时。

## 3. Phase S0：前瞻性 release lock（CPU）

### S0.1 固化两条互不冒充的结论

生成：

```text
R6_FOLDED_RELEASE_LOCK.json
S0_AUDIT_REPORT_ZH.md
```

机器字段至少包含：

```text
research_raw_lora_q2_status = failed_preserved
research_raw_lora_max_abs_diff = 1.0728836059570312e-6
deployment_candidate = r6_step500_lerobot_folded
deployment_runtime_status = not_yet_tested
formal_winner = false
auto_submit_authorized = false
```

记录 source model/config/processors/norm、raw checkpoint、C20 manifest、Q1/Q2 report 和代码的 sha256。

### S0.2 修复测试语义

新增测试，强制：

- `folded_dense_jax` 与 `lerobot_folded_pytorch` 不得共用同一 kind 名称；
- 报告必须记录实际读取的 model path/hash/backend；
- 未读取磁盘 `model.safetensors` 时不得写 `lerobot` 或 `deployment`；
- raw-LoRA failure 不会因 deployment branch 通过而被覆盖；
- release lock 不得引用旧 F2/M1 model/processor/norm hash。

执行：

```text
.venv_axis_pi05/bin/python -m py_compile tools/axis_pi05/*.py pipeline/*.py
.venv_axis_pi05/bin/pytest -q tests
git diff --check
```

S0 独立 clean commit 后进入 S1。

## 4. Phase S1：真实 LeRobot folded 的 fixed/EGL sensitivity

### S1.1 扩展真实 PyTorch backend

扩展或新增只读 parity 工具，使其真正从第 1.3 节目录加载：

```text
config.json
model.safetensors
policy_preprocessor*.json/safetensors
policy_postprocessor*.json/safetensors
```

要求 fixed noise 可注入且与 JAX fixture 完全相同；不得调用会自行产生不同 noise 的便捷 API 后再比较。报告 backend、dtype、device、model hash、processor hash、norm mode、token/mask 和 internal/external action。

### S1.2 8×4 frozen grid

复用已经冻结的 8 task × 4 seed fixture，不重抽 task/seed。比较：

```text
JAX folded-dense
LeRobot folded PyTorch source
LeRobot folded PyTorch same-process replay
LeRobot folded PyTorch fresh-process replay
```

门禁：

```text
coverage                                      32/32
shape/dtype/token/mask mismatch               0
all action finite                             true
processor normalization                       QUANTILES
JAX folded ↔ PyTorch internal/external max    <= 1e-3
PyTorch same-process replay                    bitwise exact
PyTorch fresh-process external max             <= 1e-6
gripper sign flip                              0
```

任一失败立即停止；不得更换 seed 或把 single-fixture 旧报告代替本 grid。

### S1.3 真实 EGL pair

从 Q1 无损数组中预注册并锁定：

- 每个 camera 的 transformed 最大差 pair；
- 每个 camera 至少一个代表性非零 pair；
- E1/E2/E3、zero64/fixed64 中实际存在差异的阶段/trajectory；
- pair manifest 与数组 hash。

由于 Q1 的 p99 为 0（差异很稀疏），不得挑一个 exact 的“p99 pair”冒充 sensitivity；改用“最差 pair + 代表性非零 pair”，并在报告中解释选择规则。

分别运行 Axis reference 与真实 LeRobot folded PyTorch；deployment 放行只由 PyTorch 结果决定，Axis 是 reference。门禁：

```text
token/mask mismatch                 0
state exact                         true
finite                              true
gripper sign flip                   0
external full chunk max             <= 1e-6
execute8 prefix max                 <= 1e-6
```

S1 通过状态：

```text
r6_lerobot_folded_policy_sensitivity_locked
```

## 5. Phase S2：production evaluator seed 顺序修复

Q1 已验证 seed-before-constructor 后 full-C20 数值动力学 exact，因此只移植这一项最小修复：

1. case-local Python/NumPy/Torch seed 在 `OffScreenRenderEnv` constructor 前设置；
2. constructor 后调用 `env.seed(env_seed)`；
3. reset 前再次设置同一 case-local seed；
4. 保持 direct init、10-step warm-up、native 10 Hz、camera、control/action contract 不变。

不得顺便切换 OSMesa/Vulkan、修改 `LIBERO_CONTROL_FREQ` 默认值、改变 init path 或给图像/action rounding。

新增调用顺序、case isolation、outer/child GPU namespace 单测。先做 C5 environment-only 两个 fresh process；数值动力学必须 exact。通过后形成独立 clean commit。

## 6. Phase S3：独立 R6 source staging 与 FAST export

### S3.1 staging provenance

新建独立 staging，禁止原地修改 folded source 或旧 F2 staging。允许复用已经审计的通用 `policy_server.py`、requirements、tokenizer/vendor 模板代码，但必须：

- 记录每个模板文件的来源/hash；
- 删除模板内所有旧 model/config/processor/norm；
- 只复制第 1.3 节 R6 folded model 与 processors；
- staging inventory 检查不得出现旧 F2 model hash、旧 fast hash 或第二份 norm；
- source staging 先只含 `model.safetensors`，不得预放任何 `model_fast.safetensors`。

若现有 exporter 只能接受 source ZIP，可先构造一个 immutable R6 source ZIP，或为 exporter 增加 fail-closed 的 `--source-model-dir` 模式。无论哪种方式，都必须保持“唯一模型输入是当前 R6 folded `model.safetensors`”并写出完整 inventory；不得为省时间使用 `--reuse-staging` 指向旧 F2 staging。

### S3.2 export 硬门禁

导出必须是 atomic write，失败时不留下可误认的 final 文件。要求：

```text
source model sha256                  5c73e1e8217dbb75a934c2b0165208e851f69e66cfe2ea0430590a29901e6d5b
missing source keys                  0
unexpected source keys               0
duplicate target mapping             0
strict reload                        passed
vision patch embedding weight        present + mapped
vision patch embedding bias          present + mapped
FAST key count/dtype/size/hash        recorded
model.safetensors 与 model_fast       不得在最终包共存
```

任何 SLOW dynamic remap warning、vision key warning 或 strict reload 失败立即停止。

## 7. Phase S4：真实 PyTorch folded ↔ FAST parity

### S4.1 8×4 fixed grid

在相同 processor、prompt/state/image、固定 noise 下比较 source PyTorch folded 与导出的 FAST。必须记录 internal 32D 与 postprocessed external 7D。

FAST 采用 mixed dtype，不能把 JAX raw/fold 的 `1e-3` 最大误差门禁不加说明地套用。预注册 release 门禁沿用项目 FAST recovery 的数值口径，并增加 execute8 离散安全门禁：

```text
coverage                              32/32
shape/token/mask mismatch             0
finite                                true
median external abs error             <= 0.03
p95 external abs error                <= 0.10
gripper sign agreement full chunk     >= 0.95
gripper sign flip in execute8         0
```

同时报告 max、p99、per-dim、每个 horizon，不能只输出 aggregate。旧 F2/M1 的 `0.0526/0.538/0.70` 必须被本门禁拒绝。

### S4.2 real EGL pair FAST sensitivity

对 S1 锁定的真实 EGL pairs，使用完全相同 FAST runtime。要求 finite、无 execute8 gripper sign flip，并报告相对 source PyTorch 的 action drift。若 source PyTorch pair通过但 FAST 出现离散翻转或 p95 超过 S4.1 envelope，立即停止。

S4 通过状态：

```text
r6_folded_to_fast_numeric_gate_passed
```

## 8. Phase S5：FAST server 与本地 closed-loop

### S5.1 policy server contract

在兼容 V100 的本地 runtime 启动最终 staging，日志必须明确：

```text
FAST_REQUIRED_AND_VERIFIED
fast=True
chunk=10
n_action_steps=8
action_ema=0.0
rtc=False
```

禁止 SLOW fallback、dynamic remap、vision warning、missing/unexpected key。执行 `/health`、`/reset` 与至少 24 次固定 `/act`：每次 `<10s`，记录 cold/p50/p95/max、peak VRAM、返回 shape/dtype/finite/nonzero。

V100 不是正式 scorer；本地 latency/显存结果只标为 local diagnostic。可以用约 22–23GB 的显存上限做 L4 24GB 压力诊断，但不得据此宣称 L4 已验证。

### S5.2 closed-loop 阶梯

为现有 evaluator 增加实际 FAST HTTP policy adapter，不能再用 direct Orbax/JAX 冒充提交 server。顺序：

1. C5 FAST 两个 fresh-process replay；
2. C5 source PyTorch folded reference；
3. 通过后 C20 FAST；
4. C20 source PyTorch folded reference。

要求：coverage 完整、stall=0、action finite、无 HTTP timeout/server error；C5 两次 success/termination/steps 一致。C20 paired FAST 相对 source PyTorch 的 success drop 不得超过 `1/20`，collision/timeout 各自不得恶化超过 `2/20`。这是防灾难回归门禁，不是统计显著性声明。

保存 summary/tasks/episodes/action trace、H.264 failure gallery、instruction/init/env-policy seed、failure reason 和所有 server logs。

## 9. Phase S6：ZIP、validator 与 L4 资源门禁

最终 ZIP 命名：

```text
artifacts/submission_axis_r6_step500_folded_diagnostic_<timestamp>.zip
artifacts/submission_axis_r6_step500_folded_diagnostic_<timestamp>.zip.sha256
artifacts/submission_axis_r6_step500_folded_diagnostic_<timestamp>.manifest.json
```

必须满足：

- `<20GB`，root layout 正确，无 symlink/hardlink、nested archive、绝对路径、重复 member；
- 包内存在唯一 `model_weights/model_fast.safetensors`；不存在 `model.safetensors`；
- ZIP CRC/testzip、SHA-256、官方 `validate_submission.py --static` 通过；
- offline import 通过，禁止运行时从 HF/网络下载；
- requirements、Python 3.10、CUDA 13/PyTorch 2.11 contract 审计通过；
- L4 24GB peak-memory 预算单独报告，不能用 Blackwell 96GB 或 V100 32GB 的“能运行”替代；
- `/act` 与 `/reset` 的正式限制均为 10 秒。

manifest 必须写：

```text
candidate = r6_step500_folded_diagnostic
research_raw_lora_q2_status = failed_preserved
deployment_source = lerobot_folded_pytorch
performance_status = local_b80_inconclusive_49_of_80
formal_winner = false
official_result = false
auto_upload_authorized = false
```

S6 最多得到：

```text
r6_folded_diagnostic_zip_local_ready_pending_official_precheck
```

## 10. Phase S7：配布 Blackwell 官方 pre-check Docker（人工授权后）

### S7.1 传输边界

代码 branch 与大权重/ZIP 分开处理：

- Git 只传代码、配置、测试和小报告；`outputs/`、`artifacts/` 仍保持 ignore；
- 大 ZIP 只能通过人工批准的私有 artifact 路径传输；不得上传公开 HF repo；
- 若使用私有 HF，先记录 repo id、private visibility、artifact SHA，再上传；没有明确 repo/权限时停止询问；
- 配布环境下载后先核对 ZIP SHA；临时工作放 `/opt/dlami/nvme`，需要保留的小报告再放 `~/data` 并执行 `parc-home-sync data-push`；
- 启动 Notebook 即计入 60 小时，完成后必须主动 Stop Server。

### S7.2 官方 pre-check

只使用运营方提供的 pre-check Docker，不导入 custom Docker。记录 Docker/image、CUDA、PyTorch、GPU、启动/停止时间与剩余额度。执行：

- 官方 static/完整 validator；
- offline server cold start；
- `/health`、`/reset`、至少 24 个 `/act`；
- 每请求 `<10s`，返回 10×7 raw chunk/execute8 contract 正确；
- `fast=True` 且无 SLOW/remap/vision/key warning；
- peak VRAM、cold/p50/p95/max latency；
- 至少一个真实 observation E2E smoke。

配布 Blackwell 是官方支持的 pre-check 环境，但正式 scorer 是单张 L4 24GB；报告必须同时保留这一区别。Blackwell 通过且 L4 静态资源门禁无风险后，状态才可写：

```text
r6_step500_folded_diagnostic_zip_ready_for_human_submission
```

不得自动调用比赛提交接口。

## 11. 全局硬停止条件

出现任一项立即停止，不生成貌似 ready 的下游结论：

- root/nested dirty、输入 hash/revision 漂移；
- GPU0/外部 PID GPU 被触碰或 GPU monitor evidence 不完整；
- raw-LoRA Q2 failure 被覆盖、删除或重新抽 seed；
- S1 工具没有实际读取 on-disk LeRobot model/processors；
- norm 不是 R6 fresh QUANTILES，或 state/action/camera/chunk contract 漂移；
- non-finite、shape/dtype/token/mask mismatch、execute8 gripper sign flip；
- S1/S4 预注册 parity 门禁失败；
- source/staging 出现旧 F2 model/fast/norm hash；
- vision patch embedding key 缺失、SLOW fallback、dynamic remap warning；
- `/act` 或 `/reset` 耗时 `>=10s`，HTTP 500、stall 或 server crash；
- local C20 FAST 灾难性回归超过 S5 门禁；
- ZIP ≥20GB、CRC/static validator/offline import 失败；
- L4 24GB 资源预算不安全；
- 未经授权执行 Git push、HF upload、启动配布 Notebook 或比赛 submission。

每次停止都必须写 `status/first_failure/affected_artifact/not_started_phases`，更新中文报告与机器 summary，并形成 clean commit。

## 12. 必需产物与最终交接

```text
data/reports/axis_libero_30k/post_r6_lora/r6_folded_submission/
  R6_FOLDED_RELEASE_LOCK.json
  S0_AUDIT_REPORT_ZH.md
  s1_lerobot_frozen_grid/summary.json
  s1_real_egl_pairs/summary.json
  s2_seed_order/summary.json
  s3_fast_export/source_inventory.json
  s3_fast_export/key_mapping.json
  s3_fast_export/export_report.json
  s4_folded_vs_fast/summary.json
  s5_server/server_latency.json
  s5_closed_loop/summary.json
  s6_package/package_validation.json
  s7_official_precheck/summary.json
  THREAD_REPORT_ZH.md
outputs/axis_fast_recovery/r6_step500_folded_diagnostic/
  staging/
  fixtures/
artifacts/
  submission_axis_r6_step500_folded_diagnostic_<timestamp>.zip
  submission_axis_r6_step500_folded_diagnostic_<timestamp>.zip.sha256
  submission_axis_r6_step500_folded_diagnostic_<timestamp>.manifest.json
```

最终交接必须明确区分：

1. raw-LoRA research reproducibility 仍失败；
2. folded/FAST deployment 是否通过；
3. 本地 C20/B80 不是官方得分；
4. Blackwell pre-check 不是 L4 hidden-task score；
5. ZIP 是否仅等待人工提交，还是仍有明确 blocker。

若 S0–S6 完成而 S7 未获人工授权，正确状态是 `local_ready_pending_official_precheck`，不是 `submission_ready`。
