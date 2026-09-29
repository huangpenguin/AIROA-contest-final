# AXIS π0.5 Track3：Q5 后语义签字、官方 step-100 与评测计划（2026-09-10）

## 0. 当前结论

- 本计划是日常轻量迭代计划，不是 formal audit。
- Track3 的本地 Q5 **没有训练中断**：supervisor 与 trainer 都正常退出，10/10 optimizer steps 完成，step 1 与 step 10 均通过 fresh-process restore，状态为 `passed`。
- 当前不应重跑 Q5，也不应立刻在官方服务器启动 Track3。
- Track1/Track2 已在官方 RTX PRO 6000 Blackwell 实例上重启并运行。Track3 不得停止、延长、覆盖或争用该实例上的运行、输出目录、数据目录和端口。
- Track3 下一步的唯一前置门是：对 `post-action raw -> shifted pre-action training row` 的语义做一次短签字。通过后准备传输包；等待 Track1/Track2 完成后才启动官方 step-100。

## 1. 已完成且冻结的 Track3 事实

### 1.1 四条 reverse gold

| task | frames |
|---|---:|
| alphabet soup | 278 |
| red mug | 240 |
| two mugs | 282 |
| yellow book | 260 |
| 合计 | 1060 |

训练派生仓库：`data/processed/axis_pi05_track3_reverse_gold_v1`。

### 1.2 80/20 mix 与 QUANTILES

- forward train episodes：12,832；validation episodes：1,515。
- reverse train episodes：4。
- sampler mass：forward `0.8`，reverse `0.2`；三个 reverse family 各 `0.066666...`。
- manifest SHA256：`cc6c7206458277512a4f8a33a7e43a49044d8d845cea796f70f1fc292b1d8244`。
- sampling weights SHA256：`2276a6bf2fa5b5741cb11f69c3076a865b7aa4b6ff7800baffa40897f8bd1cb9`。
- source roots SHA256：`30676c6035f6a1633db1733e2c2f5271366d235b93b99a3e8b2417a8ba8bb52a`。
- fresh norm stats SHA256：`faf3f20f04756c8f85eeea7b72b893cd6a712d37d91cb5c0144004323cc994ab`。

### 1.3 Q4/Q5

- Q4 loader frame count：forward `1,998,905` + reverse `1,060` = `1,999,965`。
- 20k sampler smoke：forward `0.7942`，reverse `0.2058`。
- Q5 config：`axis_pi05_track3_q5`。
- Q5 recipe：R6 params-only 初始化；fresh AdamW；FP32；LR `1e-6`；warmup `100`；clip `0.1`；EMA off；microbatch `1`；accumulation `64`；global batch `64`。
- Q5 共 10 条连续、有限的 optimizer metrics；trainable scope 为 `L2_action_lora_heads`，18 leaves / 24,284,192 parameters。
- Q5 证据：`data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q5_local_gpu3/`。
- Q5 checkpoints：`outputs/axis_pi05/track3_four_gold_mix_v1/q5_local_gpu3/{0,1,10}`，合计约 36 GiB。
- Q5 完成后本地剩余空间约 `490.94 GiB`，低于 runner 的 `500 GiB` 启动阈值；未经重新评估或释放空间，不再启动新的本地训练。

### 1.4 已冻结评测

- Reverse dev：四个 official T3 task 各一个 holdout start state，均与四条训练轨迹的 start-state SHA256 不同。
- Reverse dev manifest：`data/eval_manifests/axis_pi05_eval_v1/public_t3_holdout.jsonl`，SHA256 `64a1124ef17a866e259ba7e8bce96351cd183dff1ebf76c1afbd5694dfe0ec95`。
- C20 retention：`data/reports/axis_libero_30k/phase1/rollout/C20.jsonl`，20 cases，SHA256 `1dad183301e255e4287348470d962b9016420c2365f0ac8a8c2eef7ed0052231`。
- 冻结记录：`data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/evaluation_freeze.json`，SHA256 `2c8d2eca6b5138ddc59d9d00837c95e774f4a2e14687cd656662ee6e9dd77b77`。

## 2. 为什么还需要一个短语义门

录制器的真实顺序已经由代码确认：先执行 `env.step(action)`，随后把同一步的 `action`、post-action `obs` 图像和 `env.get_sim_state()` 一起写入 raw trajectory。因此，对训练行 `t >= 1`，用 raw row `t-1` 的 observation/state 作为 action `t` 的 pre-action observation，是正确的时间左移。

但是，旧计划曾把“从 start state 连续重放所有 action，full simulator state 误差必须不超过 `1e-8`”写成 publication hard gate。实际四条轨迹的连续 replay 都出现了非零漂移：

| task | first mismatch | max full-state error |
|---|---:|---:|
| alphabet soup | 263 | `1.1444953420092219e-05` |
| red mug | 143 | `0.07338490577769499` |
| two mugs | 91 | `0.751217292276802` |
| yellow book | 0 | `0.0004411426881052163` |

现有 materializer 后来把连续 replay 降为 diagnostic-only，并通过逐帧恢复 raw post-action state 生成 8D state。这个改变在进入官方长训前必须显式留证，不能用 Q5 的技术通过替代数据语义证明。

## 3. 执行阶段

### N0 — 只读 closeout，禁止重跑

1. 核对 Q5 `q5_attempt.lock.json`、`cursor.json`、`heartbeat.json`、`q5_summary.json` 一致为结束/通过。
2. 核对没有 Q5 supervisor、trainer 或 orphan process。
3. 核对 step `0/1/10` 完整、无 partial checkpoint，step `1/10` 可 fresh restore。
4. 记录 GPU3 与磁盘快照；不删除 36 GiB Q5 checkpoints，不启动新训练。

通过条件：已有 Q5 事实未变化。若 hash 或 checkpoint inventory 变化，停止并报告，不自行修补历史证据。

### N1 — shifted pre-action 语义签字

新增一个只读/验证型脚本或复用 `tools/axis_pi05/materialize_track3_reverse_gold_v1.py` 的验证逻辑，产出：

`data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q6_semantic_signoff.json`

必须逐项证明：

1. **Recorder ordering**：引用实际代码位置，证明 raw row `i` 是执行 `action[i]` 后同步取得的图像、full state、contact/predicate 与 success。
2. **Index mapping**：
   - output row `0` = 冻结 start state 恢复得到的 pre-action observation + raw action `0`；
   - output row `t>=1` = raw post-action observation/state `t-1` + raw action `t`；
   - 最后一条 raw post-action observation 仅作 terminal evidence，不错误配给额外 action。
3. **无内容篡改**：四条 raw action 数组逐元素进入派生 action；不平滑、不插值、不删帧、不改 gripper；所有值 finite、shape 正确。
4. **逐帧 state 对齐**：对全部 1060 行或至少每 task 的 start、pre-contact、grasp、首次 replay divergence、release、terminal 邻域，证明派生 8D state 来自对应冻结 full state，而不是来自已经漂移的连续 replay state。
5. **图像对齐**：对 `t>=1`，派生 front/wrist 必须与 raw row `t-1` byte/pixel 等价；row 0 必须可追溯到冻结 start state 的一次恢复与渲染。
6. **训练窗口语义**：再次验证 action chunk、episode boundary 和 `action_is_pad`，不得跨 episode；padding mask 必须对应真实动作长度。
7. **漂移解释**：把上表四条连续 replay 误差原样写入签字文件，明确它只说明保存的 simulator state 不含可完全复现接触求解器历史的全部隐状态；不得声称达到 `1e-8`。

本计划显式把 publication gate 修正为“真实 logged transition 的时间和内容可追溯”，而不是“从不完整 simulator snapshot 连续重放必须 bit-exact”。只有 1–7 全部通过，才允许复用当前派生仓库、mix 和 QUANTILES。

若任一项失败：不要上传或官方训练；保留现有 raw/Q1–Q5 为历史证据，升级 recorder 为原生 pre-action observation/state 记录，重新采集相同四 task 的合格轨迹，并使用新 dataset/mix/norm/run ID 从 Q1 重做，禁止覆盖 v1。

### N2 — 冻结 Q6 传输清单

N1 通过后，创建 content-addressed manifest，至少包括：

- 原始 R6 params-only 初始化；tree SHA256 必须为 `d9b7e0d15781060fccfd61dc1890ad98993f0224e4aebba1dd6658d7c011931f`。
- `data/processed/axis_pi05_track3_reverse_gold_v1`。
- `data/manifests_track3/four_gold_80_20_v1/` 全部文件。
- `configs/axis_pi05/track3_four_gold_mix_v1_20260910.yaml`。
- Q1–Q5 summaries、evaluation freeze、N1 semantic signoff。
- 本次训练所需的 root 与 nested OpenPI 精确 source snapshot/patch，尤其是 `axis_pi05_track3_q5` 和 temporal `action_is_pad` 支持。
- 官方启动/监控/恢复脚本及其 SHA256。

不要把 Q5 step 10 当作官方初始化；不要为了传输再复制一份 36 GiB Q5 checkpoint。优先逐文件 hash + 上传/同步，避免本地制作同尺寸 tar 导致磁盘继续下降。

### N3 — 等待 Track1/Track2 释放官方实例

在 Track1/Track2 任一 supervisor/trainer 仍存活时：

- 不启动 Track3 JAX 进程；
- 不改它们的 CUDA/JAX 环境、dataset cache、output path、PID/cursor/heartbeat；
- 不点击 Stop My Server 或延长；
- 可做 N0–N2 的本地只读验证和传输准备，但大文件传输若会争用当前训练 I/O，也延后。

Track1/Track2 全部结束后，先保存它们的最终 checkpoint/summary 和进程/GPU 快照，再重新核对本 session 剩余时间与配额。若不足以完成 step 100、保存和 fresh restore，则不要启动 Track3。

### N4 — 官方 Blackwell step 1 → step 100

在独立的新 output/evidence 目录运行，绝不复用 Track1/Track2 或 Q5 的 cursor：

1. 官方机重新计算传输清单全部 SHA256；任何不一致都停止。
2. 只从 R6 params-only 初始化，创建 fresh optimizer；不得从 Q5 step 10 续训。
3. 保持 Q5 冻结 recipe：L2 action LoRA heads、FP32、LR `1e-6`、warmup `100`、clip `0.1`、EMA off、global batch `64`。
4. 先完成 step 1，检查 batch 来源、80/20 sampler、finite loss/grad、GPU memory、checkpoint 和 fresh-process restore。
5. step 1 通过后，在同一官方 run/cursor 继续到 step 100；每个 optimizer step 的 metrics 必须连续、有限，禁止悄悄降低 batch 或更换数据/norm。
6. step 100 保存 params + train state，并在全新进程 restore；产出原子 summary、cursor、heartbeat 和 source manifest。

任何 OOM、NaN/Inf、hash mismatch、loader provenance mismatch、checkpoint restore 失败都停止在 step 1/最近完整 checkpoint；不得自动改超参后继续使用同一 run ID。

### N5 — 配对评测与是否继续到 step 300

对同一冻结评测集、同一 seeds 和预算，配对比较 R6 baseline 与官方 step 100：

- Reverse dev：4 个 disjoint start states。
- C20 retention：20 cases。
- 保存逐 case success、步数、终止原因、视频/日志索引与汇总。

继续门：

- reverse dev 至少比 R6 baseline 增加 `+1/4` success；并且
- C20 success 相对 R6 baseline 下降不超过 `2/20`；并且
- 没有新的系统性异常、nonfinite 或恢复失败。

只有全部满足，才新建独立计划从 step 100 续到 step 300。否则停止在 step 100，保留结果，调整数据而不是盲目拉长训练。

## 4. 当前明确禁止事项

- 不把 Q5 的正常退出误判成训练中断并重跑。
- 不在 Track1/Track2 运行期间启动 Track3 官方训练。
- 不用官方 baseline ZIP 冒充 R6 初始化。
- 不用 Q5 step 10 作为官方 step-100 的初始权重。
- 不覆盖 v1 raw、派生仓库、manifest、norm、evaluation freeze 或历史报告。
- 不因本地磁盘低于 500 GiB 擅自删除 checkpoint；如需释放空间，先向用户报告明确目录、大小与恢复性。
- 不在未通过 N1 前声称四条连续 replay 达到 `1e-8` 或数据已经完成 publication signoff。

## 5. 下一位执行 agent 的最短路线

`N0 只读确认 Q5 passed` → `N1 生成 semantic signoff` → `N2 生成传输 manifest` → `等待 Track1/2 完成` → `N4 官方 step 1/100` → `N5 paired eval`。

当前应立即执行的是 N0–N1；当前不应执行 N4。
