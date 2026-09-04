# AXIS Pi0.5 Track 3：Background Predicate Convergence Release 计划

> 制定日期：2026-09-04（Asia/Tokyo）
> 输入 commits：root `f969393f3e16dc94985675083d09daa712cc7c04`，execution `bc2419e62ff05a9584d1d6033d22b0e55d03a240`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`
> 当前状态：`track3_background_convergence_gpu_failed_task19_canary_hard_stop_waiting_review`
> 唯一授权：执行K0–K4，随后hard stop
> 不授权：修改BDDL/region/predicate、SE(3)、抓取/demo、Teleop、批量生成、训练/norm、FAST/ZIP、官方GPU、上传或比赛提交

## 0. 目标

本计划修复两件事：

1. 将J2的15个raw 21-step worker reports闭包到Git，使结论可在新clone复算；
2. 用role-aware monotonic contract替代“完整truth vector必须逐帧完全相同”：严格保护operator-critical source和reverse goal，同时允许background initializer朝其BDDL预期状态单调收敛。

旧J1/J2/J4、`8/9`和唯一transition永久只读，不修改或重标。新产物写入：

```text
data/reports/axis_libero_30k/track3_background_predicate_convergence/
```

## 1. K0：raw evidence closure（CPU/filesystem only）

### K0.1 输入锁

输出`k0_input_lock.json`，锁定：

- root `f969393...`、parent/execution `bc2419e...`、audit parent `3acbde8...`、nested OpenPI；
- J0/J1/J2/J4 summary与旧I2 hash；
- J3=`not_evaluated`、reverse-demo exact/pending/accepted=`0/0/0`；
- root/OpenPI clean，`LIBERO-plus` dirty只读，GPU0外部PID只读。

### K0.2 raw manifest与Git closure

从`j2_summary.json.raw_fresh_process_reports`解析恰好15个raw文件；禁止用glob静默补文件。每个文件必须满足：

- schema=`track3_operator_scoped_stability_j2_worker_v1`、status=`passed`；
- task/kind/run identity恰好覆盖reverse `3×3`和forward `2×3`；
- checkpoints恰好为整数`0..20`，truth/detail/contact均完整、无NaN/Inf；
- 记录relative path、bytes、SHA-256、task/kind/run/native-state hash。

输出`k0_raw_manifest.json`，manifest自身使用canonical SHA-256。只将这15个raw `j2_task*.json`用`git add -f`显式纳入Git；不纳入15个可重建的`j2_reanalysis_*.json`。完成一个clean evidence-closure commit，再从fresh checkout/临时worktree验证15/15存在且hash一致。

缺少任一raw文件、identity/checkpoint/hash不一致或无法clean commit，状态置为`track3_background_convergence_evidence_closure_failed`并执行K4；不得重跑J2替代原证据。

## 2. K1：冻结monotonic predicate contract（CPU/code gate）

新增独立分析器与测试，不修改旧J工具、J1 contract或J2 reports。运行K2前输出并hash锁定`k1_monotonic_contract.json`。

step5–20固定规则：

1. `operator_critical` source：每步必须true；任何false立即失败。
2. `operator_critical_goal` / reverse goal：每步必须false；任何true立即失败。
3. `background_initializer`：truth序列必须单调非递减；允许全false、全true或最多一次`false→true`；任何`true→false`以及`false→true→false`均失败。
4. 未识别role、缺atom、顺序/重复atom、truth/detail未执行、非bool value均fail-closed。
5. robot、participant、contact、finite、scene-name与原J1 family-specific gates完全不变；不得因predicate规则修复而放宽物理/contact门禁。
6. 旧exact-vector结果和transition ledger同时报告，不能隐藏旧failure。

单测至少覆盖全false、全true、一次false→true、true→false、回摆、critical变化、goal变化、unknown role、missing step/atom以及旧J2 immutability。

## 3. K2：15个raw只读复算（CPU only）

只能读取K0 manifest中的15个raw文件，不执行simulator，不读取ignored corrected reports作为oracle。

输出：

- `k2_transition_ledger.json`：逐atom序列、role、所有transition step/direction；
- `k2_reverse_release.json`：9条reverse的monotonic + 原物理/contact gates；
- `k2_forward_controls.json`：6条forward schema/contact及background regression诊断；
- `k2_summary.json`。

硬门禁：

- raw manifest/fresh-checkout hash 15/15；
- reverse 9/9，forward 6/6，worker exception 0；
- critical transition 0、goal transition 0、background `true→false` 0、background回摆0；
- 允许的background `false→true`只作为计数，不要求必须出现；
- old I2/J1/J2/J4 hashes保持。

任一失败即`track3_background_convergence_cpu_release_failed_<reason>`并执行K4。K2通过也不得直接启动SE(3)。

K2通过后必须将K1 contract/tool/tests和K2 reports提交为一个clean CPU-release commit，记录commit与contract/tool hash；K3只能执行该commit中的冻结代码。K3开始后不得再修改contract、工具、BDDL、threshold或seed选择逻辑。

## 4. K3：GPU3 production-like release

仅在K2全通过、K0 evidence commit与K1 contract/tool hash冻结、root/OpenPI clean、GPU3自然空闲时执行。只允许GPU3；GPU0/1/2均禁止。

### K3.1 task19单次remediation canary

使用真实camera/EGL、20Hz、7D zero hold，执行reset→5 warm-up→step5–20 measurement；生成逐帧JSON和诊断H.264。应用完整K1 contract，并直接记录背景predicate monotonic ledger。随后只用direct simulator-state perturbation验证`Open`可置false/restore true；不得把perturbation写成action/demo。

失败即`track3_background_convergence_gpu_failed_task19_<reason>`并执行K4，不retry、不换seed、不调contract。

### K3.2 三族matrix

K3.1通过后，在相同代码/contract/spec/hash下运行task10→task20→task19各`3/3` fresh process，共最多9次。要求：

- 9/9完整monotonic + physical/contact gate；
- predicate sensitivity false/restore true为9/9；
- JSON/H.264完整、finite、20Hz；
- reverse-demo仍为`0/0/0`。

任何一条失败立即停止剩余任务。全部通过的唯一状态：

```text
track3_predicate_convergence_dynamic_validated_pending_se3_review
```

## 5. K4：收尾与hard stop

无论成功或失败：

1. 输出`summary.json`、`K4_CLOSEOUT_REPORT_ZH.md`，未执行项使用`not_evaluated`；
2. 并列记录旧J exact-vector `8/9`与新monotonic结果，禁止覆盖旧证据；
3. 记录raw manifest/hash、contract/tool commit、GPU UUID/PID/峰值、process counts、视频/JSON hashes；
4. 重验old I2/J1/J2/J4、reverse-demo`0/0/0`、root/OpenPI clean与`LIBERO-plus`未触碰；
5. 运行targeted tests、`JAX_PLATFORMS=cpu PYTHONPATH=. .venv_axis_pi05/bin/pytest -q tests`、JSON parse、`py_compile`、`git diff --check`；
6. 更新计划索引、Markdown/机器交接，生成clean commit并hard stop。

## 6. 明确禁止

- 不得原地修改或重标旧J1/J2/J4、旧I2或统一阈值；
- 不得通过延长window、挑seed/retry或删除background atom过门；
- 不得修改BDDL/region/predicate或使用hidden task细节；
- 不得运行SE(3)、grasp/action demo、Teleop、pending/accepted promotion、批量生成；
- 不得启动LoRA/dense/30K、norm、FAST/ZIP、官方GPU、上传或比赛提交；
- 不得触碰/终止GPU0外部任务，也不得因GPU3忙改用其他GPU。

K4后必须由下一次审核决定是否授权一条support-only SE(3) canary。
