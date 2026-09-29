# AXIS π0.5 Track3：Q5 step-10 冻结评测至官方训练交接计划（2026-09-10）

## 0. 决策与范围

本计划选择用户给出的 **方案 1**：

> 使用现有 Q5 step 10 在本地 GPU3 做冻结评测；不重新训练、不创建新的短训 canary。

原因：Q5 10-step 已通过，重复训练没有新增信息；当前本地磁盘约 490.8 GiB，低于训练 runner 的 500 GiB 启动门；现有 Q5 namespace 为一次性运行。冻结评测只恢复既有 checkpoint 做 inference，不触发训练磁盘门，也不改写 Q5 cursor/checkpoint。

本计划是
`docs/AXIS_PI05_TRACK3_POST_Q5_SEMANTIC_SIGNOFF_AND_OFFICIAL_STEP100_PLAN_20260910.md`
的执行补充。它新增“Q5 step 0/10 配对冻结评测”和“官方 Track3 专用 runner”两段；不改写已被 Q6 证据引用的旧计划字节。

当前 Track1/Track2 正在官方 RTX PRO 6000 Blackwell 实例运行。其 supervisor/trainer 存活期间，本计划只允许本地 GPU3 评测和本地传输准备，不允许启动官方 Track3 训练。

## 1. 当前完成状态

### 已完成

- Q5：`10/10` optimizer steps，通过；step `1/10` fresh-process restore 通过。
- Q6 shifted pre-action semantic signoff：通过。
- Q6 证据：
  `data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q6_semantic_signoff.json`
- Q6 状态：`passed_shifted_pre_action_semantic_signoff`。
- Q6 publication decision：`reuse_v1_for_q6_transfer_preparation_only`。
- Reverse dev：4 个与训练 start state 不重合的冻结 case。
- Retention：固定 C20 20-case manifest。
- GPU3：当前空闲；GPU0/1/2 外部作业保持不动。

### 尚未执行

- Q5 step 0/10 的 GPU3 冻结配对评测。
- Track3 官方传输 manifest/bundle。
- Track3 官方专用 step-1/step-100 runner。
- 官方 Track3 训练。

## 2. 冻结输入

| 项目 | 冻结值 |
|---|---|
| baseline checkpoint | `outputs/axis_pi05/track3_four_gold_mix_v1/q5_local_gpu3/0` |
| diagnostic checkpoint | `outputs/axis_pi05/track3_four_gold_mix_v1/q5_local_gpu3/10` |
| OpenPI config | `axis_pi05_track3_q5` |
| norm stats | `data/manifests_track3/four_gold_80_20_v1/norm_stats.json` |
| norm SHA256 | `faf3f20f04756c8f85eeea7b72b893cd6a712d37d91cb5c0144004323cc994ab` |
| reverse manifest | `data/eval_manifests/axis_pi05_eval_v1/public_t3_holdout.jsonl` |
| reverse manifest SHA256 | `64a1124ef17a866e259ba7e8bce96351cd183dff1ebf76c1afbd5694dfe0ec95` |
| C20 manifest | `data/reports/axis_libero_30k/phase1/rollout/C20.jsonl` |
| C20 manifest SHA256 | `1dad183301e255e4287348470d962b9016420c2365f0ac8a8c2eef7ed0052231` |
| protocol | chunk 10 / execute first 8 / discard 2 |
| control frequency | 20 Hz for both paired checkpoints |
| local GPU | physical GPU3 only |

Q5 step 0 是同一 Q5 run 在 optimizer update 前保存的 R6 params-only 基线，适合与 step 10 使用完全相同的 config、norm、inference adapter 和环境做配对比较。评测前仍须验证 step 0 参数树与 Q5 锁中的 R6 初始化身份一致。

## 3. 为什么需要 Track3 专用评测入口

当前没有可直接运行的 post-Q5 Track3 evaluator：

- `tools/axis_pi05/phase1_rollout.py` 只接受 `C20/B80`，并固定旧 phase1 的 10 Hz source/control 元数据，不能把 reverse holdout 改名后冒充 C20。
- `tools/axis_pi05/eval_runner.py` 的内置 server 路径从候选名推导旧 `F*/M*` norm 目录，并要求旧 checkpoint sidecar identity；Q5 checkpoint 是直接 Orbax 目录，不应为了适配而修改历史 checkpoint。
- `tools/axis_pi05/checkpoint_http_server.py` 可以正确恢复指定 config + direct Orbax checkpoint，也支持外部注入 `AXIS_PI05_NORM_STATS`，应作为底层 policy server 复用。

因此新增一个窄入口：

`tools/axis_pi05/run_track3_q5_step10_frozen_eval.py`

不要修改通用 phase1 evaluator 的历史默认值；专用入口负责 Track3 manifest、20 Hz、checkpoint identity、resume/cursor 和配对汇总。

## 4. E0 — 实现一次性冻结评测 runner

### 4.1 归属与输出

执行 agent 只负责以下新文件/namespace；不得回退其他 agent 的 dirty changes：

- runner：`tools/axis_pi05/run_track3_q5_step10_frozen_eval.py`
- focused test：`tests/test_track3_q5_step10_frozen_eval.py`
- 小型 evidence：
  `data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q7_gpu3_frozen_eval/`
- 视频、action trace 和较大运行输出：
  `outputs/axis_pi05/track3_four_gold_mix_v1/q7_gpu3_frozen_eval/`

### 4.2 runner 合同

runner 必须：

1. 拒绝覆盖已完成的 attempt；使用 attempt lock、PID、atomic cursor 和 heartbeat。
2. 启动前严格解析 Q5/Q6/evaluation-freeze JSON 并拒绝 duplicate keys。
3. 复核本计划第 2 节全部 SHA256、step `0/10` Orbax metadata/params、Q6 passed 状态。
4. 只允许 `CUDA_VISIBLE_DEVICES=3`；启动前后记录四卡 UUID/memory/process，禁止 kill 或触碰 GPU0/1/2。
5. 每次只启动一个 `checkpoint_http_server`，依次恢复 step 0 和 step 10；server 必须使用：
   - config `axis_pi05_track3_q5`；
   - 本计划冻结的 Track3 norm；
   - JAX FP32/cuda；
   - chunk `10`、execute8；
   - 每个 episode 使用 manifest 内的固定 policy seed 重置。
6. simulator 使用冻结 BDDL/init state、固定 env seed、20 Hz control、每 case 300-step 上限。
7. 每 case 原子写出 success、collision、timeout/stall、steps、failure class、有限 action 统计、checkpoint/norm/manifest hashes、视频和 action-trace 索引。
8. 中断后只能从同一输入 hash 的最后完整 case 续跑；partial case 重新运行并保留旧 partial 证据，不把异常记成真实 policy failure。
9. `finally` 终止本 runner 自己启动的 server/simulator，检查无 orphan；不得操作外部进程。

### 4.3 focused tests

不启动 GPU/MuJoCo，至少覆盖：

- step 只能为 `0` 或 `10`；
- hash mismatch / Q6 非 passed 时 fail closed；
- checkpoint/norm 注入不落回旧 `M0/M1/M2` 目录；
- execute8 chunk reset 不重新 seed episode RNG；
- cursor resume 不重复完整 case；
- partial、server failure 和 policy nonfinite 分类分离；
- paired summary 必须有相同 case 顺序与 seeds；
- GPU boundary 和 output namespace 不允许指向 Q5 历史目录。

本阶段只跑新 focused tests、`py_compile` 和 `git diff --check`；不自动跑全量 suite。

## 5. E1 — GPU3 两 case restore/inference canary

在完整冻结评测前，每个 checkpoint 各运行两个 case：

1. reverse dev 的 alphabet-soup holdout；
2. C20 的首个 case。

顺序：step 0 reverse → step 0 C20 → 完全关闭 server → step 10 reverse → step 10 C20。

硬门：

- 两个 checkpoint 均在 GPU3 成功 restore；
- 4/4 case 均产生完整结果，不能有 infrastructure/model exception；
- action shape 始终为 7D、全部 finite；
- execute8 的 chunk/reset/seed 日志正确；
- checkpoint server 和 simulator 结束后无 orphan；
- GPU0/1/2 的外部 PID 未变化且未被本任务占用。

评测不受训练 runner 的 500 GiB 门限制，但启动前应估算视频/trace 上限并保证不复制 36 GiB checkpoint。若评测预计新增超过 5 GiB，先报告而不运行。

## 6. E2 — GPU3 完整配对冻结评测

E1 通过后，在同一冻结协议下完成：

| checkpoint | reverse dev | C20 | 合计 |
|---|---:|---:|---:|
| Q5 step 0 / R6 baseline | 4 | 20 | 24 |
| Q5 step 10 | 4 | 20 | 24 |
| 总计 | 8 | 40 | 48 |

推荐顺序：先完成两个 checkpoint 的 reverse 4+4，再完成 C20 20+20。所有配对必须使用完全相同的 case order、env seed、policy seed、20 Hz 和 max steps。

产出 `paired_summary.json`，至少包括：

- 48/48 coverage 与每个 case 的配对关系；
- reverse success delta；
- C20 success delta；
- collision/timeout/stall delta；
- action norm 与 gripper sign/finite 汇总；
- checkpoint、config、norm、manifest、runner、root/nested source hashes；
- 运行前后 GPU/process/disk 快照；
- 明确 `official_result=false`、`diagnostic_only=true`。

### E2 通过门

- coverage `48/48`，无 infrastructure/model exception、nonfinite 或 restore failure；
- step 10 相对 step 0 的 C20 success 下降不超过 `2/20`；
- 没有系统性全零动作、全程同一 gripper sign 或新增大面积 stall；
- reverse dev 不要求 step 10 已有 `+1/4`：10 steps 太短，`0 → 0` 但技术稳定仍允许进入官方 100-step；
- 若 step 10 已有 `+1/4`，只记为支持性诊断，不把 step 10 升格为训练初始化或候选。

若 C20 下降超过 2、出现 nonfinite/interface failure 或动作塌缩，停止官方 Track3；先诊断 config/norm/inference contract。不得通过再训一个本地 canary 掩盖失败。

## 7. E3 — 官方传输准备

E2 通过后生成 content-addressed transfer manifest。除上一计划列出的 Q1–Q6、数据、mix、norm、source snapshot 外，再加入 E0–E2 runner/test/summary hashes。

### 官方数据路径规则

- 只使用 `/opt/dlami/nvme`，不向 40 GiB `$HOME` 解压大数据。
- 不假定官方现有 `libero_combined_20hz_v2.tar` 等于本地 Track3 forward source。
- 在使用前核对官方解压源是否满足：40 tasks、14,347 episodes、2,238,036 frames，以及 meta/info 和 source-root 身份与当前 manifest 一致。
- 若不一致，停止；选择传输冻结的本地 forward source，或以官方源新建不同 manifest/norm/run ID。不得让当前 `cc6c...` manifest 指向不同数据字节。
- R6 初始化若已由 Track1/2 放在官方 NVMe，可复用同一 params bytes，但必须重算 tree hash；不能用官方 baseline ZIP 或 Track1/2 训练后 checkpoint 替代。

不要把 Q5 step 10 传为官方训练初始化；它只作为本地诊断证据。官方 Track3 始终从原始 R6 params-only 新开 fresh optimizer。

## 8. E4 — 实现官方 Track3 专用 runner

在不改 Track1/2 runner namespace 的前提下新增：

`tools/axis_pi05/run_track3_official_blackwell_lora.py`

runner 至少提供独立的 `preflight`、`step1`、`step100`、`validate` 子命令：

- 独立 run ID、output/evidence/PID/cursor/heartbeat，不复用 Track1/2 或本地 Q5。
- preflight 验证 Blackwell GPU、JAX device、R6 tree hash、数据/meta、manifest、sampling weights、norm、source code 和 transfer manifest。
- 训练只允许 `axis_pi05_track3_q5` 等价冻结配置：L2 action LoRA heads、FP32、LR `1e-6`、warmup `100`、clip `0.1`、EMA off、microbatch `1`、accumulation `64`、global batch `64`、80/20 sampler。
- step 1 从 R6 params-only + fresh optimizer 开始；保存并 fresh-process restore 后才能继续同一 run 到 step 100。
- metrics 必须严格连续、finite，并记录每步 microsteps/examples、sampler 来源和显存。
- hash mismatch、OOM、NaN/Inf、loader provenance、restore failure 都 fail closed；改超参必须使用新计划与新 run ID。

只为新增入口运行 focused tests；不要改动正在运行的 Track1/2 runner/cursor。

## 9. E5 — 官方训练启动门与停止点

只有以下条件全部成立，才允许启动 Track3：

1. Q6 semantic signoff passed；
2. E2 step 0/10 冻结配对评测通过；
3. transfer manifest 在官方机 hash exact；
4. 官方 Track1/Track2 trainer/supervisor 均已自然结束，final checkpoint/summary 已保存；
5. Blackwell GPU 无其他训练进程；
6. 官方 session 剩余时间足够完成 step 100、保存和 fresh restore；
7. Track3 专用 output/cursor 不存在，或是 hash 相同且明确可恢复的未完成 attempt。

启动顺序：

`preflight` → `step1` → fresh restore/finite/sampler 检查 → **同一 run** `step100` → fresh restore → hard stop。

本计划在 step 100 完成并可恢复处停止。随后才进入上一计划定义的 reverse-dev/C20 配对评测和是否继续 step 300 的决策。

## 10. 当前禁止事项

- 不选择方案 2，不重跑或扩展本地 Q5 训练。
- 不用 Q5 step 10 初始化官方 Track3。
- 不因评测使用 GPU3 而修改 500 GiB 训练安全门。
- 不删除 Q5 step `0/1/10` 来释放空间。
- 不在 Track1/2 运行期间启动官方 Track3 或修改其运行环境。
- 不把本地冻结评测写成官方比赛结果。
- 不更新当前 dirty 的 plan index/machine handoff 指针；待其他 writer 停止后由单一交接者统一更新，避免覆盖并发修改。

## 11. 给执行 agent 的最短交接

当前直接从 **E0** 开始：实现专用冻结评测 runner 和 focused tests；随后在 GPU3 执行 E1/E2。E2 通过后做 E3/E4，但必须等待 Track1/Track2 自然完成，才允许 E5 启动官方 Track3 step 1/100。
