# Track 3 TZ ER0–ER5 独立审核

> 审核日期：2026-09-06（Asia/Tokyo）
> 审核输入 HEAD：`c2d9f23bb0bbe13a0860aeb4fecf3a75c932ae0d`
> ER worker source：`04eef24161bfe62adc9a51378235f73b34e6ac51`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 唯一审核结论：`remediation_required`

## 1. 获得批准的事实

- 审核开始时 root 在 `c2d9f23` clean，OpenPI 在 `b58aff0` clean；`LIBERO-plus` 仍是既有 1 项 modified 和 4 项 untracked。ER cursor 为 inactive，记录 PID 不存活，无 live Track3 worker，GPU0–3 无 compute app。
- ER0 对旧 Tonight 证据的派生事实成立。23 个旧 JSON 全部 duplicate-key strict；6 个旧 NPZ 的 object-array/no-pickle failure、首条 infrastructure failure、后六条诊断性 family failure、三组 byte-identical trace 与旧 aggregate null 矛盾均保留。ER0 manifest 的 30 项当前 bytes/SHA exact；其中 24 项存在于 `2ae08e9`，6 个旧 NPZ 本来就未进入该提交，不能改称 Git-tracked。
- fallback/source 是同 inode 的只读 hardlink。独立重算两者均为 `9,383,571,677` bytes、SHA-256 `95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a`；两次全包 CRC 通过，3401 entries，无 duplicate/unsafe path，恰好一个 `model_fast.safetensors`，无 slow 权重。它只获批为 `fallback_no_track3_improvement`。
- ER4 确实只消费三条新 identity，顺序为 support→containment→articulation，child PID/PGID 各自唯一，均运行于 root `04eef24`、tool SHA `15e53f436457ed427a0e7564ad4c85841d404fe2fc3efbdda0ee97c268481f09`、OpenPI `b58aff0`、CPU/no-renderer，return code 0、无 timeout/descendant/retry，seed 记录为实际应用。
- 三条 `family_failure` 的物理分类有效。support 在 approach 15 步后 descend 140 步失败，总计 155；containment 为 9+140=149；articulation 在 approach 180 步失败。三条 source 最终仍 true、goal false，并分别有 action saturation `10/5/10` 次，相应 controller-clip gate 已失败。三份 trace bytes/SHA exact，全部 typed、finite、`allow_pickle=False` 可读，且 `training_eligible=false`。
- 没有成功 family，accepted reverse 为 0，training/new candidate 为 false；提交差异只含执行器、测试和交接文件，没有 GPU、generation、replay、Teleop、dense full-FT、上传或比赛提交。生命周期更新前独立复跑 targeted 为 `15 passed`，项目全量为 `308 passed, 7 skipped, 23 warnings`。

## 2. 必须修复的合同缺口

### 2.1 ER evidence 没有进入任何提交

ER namespace 现有 29 个文件，包括三条 raw/NPZ、26-entry manifest、outer hash 与 final validation；`git ls-files` 结果为 `0/29`，全部受顶层 ignore 规则排除。`04eef24` 只提交工具和测试，`d412c86`/`c2d9f23` 只提交交接文本。因此当前本机 bytes 可复核，manifest 26/26 与 outer hash也 exact，但这些证据不具备 checkout/reclone 可复现性，所谓 clean evidence commit 不成立。

### 2.2 resume 和 clean-state identity 没有按声明 fail-close

`run_er4()` 的 existing-result 分支只比较 result content hash 与 profile ID，随后直接采用 payload；没有对 payload 内的 root/tool/OpenPI/profile/command、clean state、trace hash再调用 identity gate。worker CLI 又用 child 当下的 porcelain 构造 expected clean/OpenPI/`LIBERO-plus` 值，无法检测 parent→child 之间的 dirty baseline drift。现有三条 fresh run 的记录身份可以批准，但 `resume_requires_result_hash_and_identity=true` 的一般合同不能批准。

### 2.3 pre-physics 回归和 finalizer 主要仍是声明值

ER3 要求在 physics 前真实覆盖 sticky stop、周期 heartbeat、TERM→KILL/no-descendant、manifest/outer hash和inactive closeout；现有定向测试只覆盖 helper/flag/静态 handoff，没有 fixture child 的 timeout、heartbeat序列、sticky resume或完整 finalizer 回归。`er1_executor_contract.json` 中这些能力主要来自 `cli_contract()` 的常量布尔值。

`run_er5()` 没有执行 targeted/full pytest；它只把事先生成的 `tests_before_er5.json` 纳入 manifest。ZIP inventory在 manifest/outer hash之前运行，final validation 的 pass 条件不包含 expected ZIP hash/CRC/layout、GPU free、真实 process scan或测试 return code；`live_worker_pids`直接写空列表。收尾 heartbeat 至今仍为 `state=finalizing`，与 inactive cursor/hard stop不一致。因此“按冻结顺序完成 post-output final validation”的声明不能批准。

### 2.4 contact-completeness 名称与物理语义不一致

worker 的 `contact_complete` 实际等于 `contact_matrix.complete`，只证明 MuJoCo contact telemetry 枚举成功；即使没有 demonstrator↔object/handle 接触也会 passed。三条 run 已因 stage/source/goal/clip失败，故这不会改变当前 family-failure 分类，但未来 success gate 必须把 telemetry-schema completeness 与 required physical contact completeness 分开。

## 3. 当前真实状态

```text
R6 fallback                 approved as honest mechanical fallback
ER0 old-evidence derivation approved on current local bytes
ER4 CPU attempts            3 consumed; 3 valid diagnostic family failures
successful family           0
accepted/train/candidate    0 / false / false
executor/finalizer contract remediation_required
```

## 4. 后续授权

旧 ER 计划已归档。唯一活动计划为 `docs/AXIS_PI05_TRACK3_ER_EVIDENCE_ANCHOR_AND_FINALIZER_VERIFICATION_PLAN.md`：只允许把现存 ER bytes 原样锚定、修复完整 identity/resume、用无 physics fixture 实测 supervisor、补全 required-contact gate，并实现真实 finalizer/checkout closure。

三条 ER4 identity和旧七条 TZ identity都不得重跑；不授权任何新 physics、GPU0–3、官方GPU、Teleop、generation/replay、accepted数据、训练/norm、新candidate ZIP、上传或比赛提交。完成 EF0–EF4 后再次 hard stop。

## 5. 审核生命周期验证

本报告及派生 summary 保留原始失败标签，不修改 ER/TZ evidence。生命周期更新后 targeted 为 `19 passed`，项目全量为 `312 passed, 7 skipped, 23 warnings`；strict duplicate-key JSON、归档/引用、py_compile 和 `git diff --check` 在最终提交前再次复核。
