# Track 3 Background Convergence K0–K4 独立审核

> 审核日期：2026-09-04（Asia/Tokyo）
> 审核输入：root `8490f8dde1148aec0aec4d3d105082b44cfbb286`
> K2 frozen commit：`cb8e898640a66e9baf5a587f21d5f44a84494a8e`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`

## 1. 获批准的部分

- K0 的 15 份 raw J2 reports 已被 Git 跟踪，总计 `16,072,261` bytes；fresh checkout 为 `15/15` bytes/hash 一致。
- K1 的 role-aware monotonic contract 与 K2 的只读复算程序真实执行。K2 为 reverse `9/9`、forward `6/6`，critical/goal transition 均为 `0`，旧 exact-vector `8/9` 保留。
- K3 只运行了一次 GPU3 task19 canary，失败后没有 retry、换卡或继续 matrix；K4 hard stop、GPU 边界与禁止项执行正确。
- 执行提交 ancestry、旧 I2 hash、reverse-demo `0/0/0`、root/OpenPI clean 与 `LIBERO-plus` 既有 dirty preservation 均可核对。

## 2. K3 的真实发现

K3 不是单纯的 renderer 抖动。task19 canary 的背景 bowl 从 step5 到 step20 持续与 middle drawer 接触：step5 有 `39` 个 bowl↔drawer geom contacts，最深 penetration 约 `-0.00680 m`；到 step20 仍有 `39` 个、最深约 `-0.00550 m`。bowl 的 `On(table region)` 在 step19→20 从 true 变为 false。

旧 contact gate 把 step5 已存在的 target↔background body pair 全部 grandfather 为 diagnostic，所以在明显穿透时仍报告 physical/contact passed。这证明当前本地 reverse-BDDL native reset 不能直接作为合成数据初态；它不证明 Track 3 的 close operator 无效。

## 3. 三个确定的观测工具缺陷

1. `OffScreenRenderEnv` 的 renderer flags 在内部 `env.env`，当前 schema helper只读外层 wrapper，因而报告 `use_camera_obs=false`、`has_offscreen_renderer=false`；实际两路均获得 `21/21` frames。
2. ffmpeg 输出名为 `*.mp4.partial`，无法按扩展名选择 muxer；应使用 `*.partial.mp4` 或显式 `-f mp4`，并以 ffprobe 校验 H.264、20 FPS 与帧数。
3. task19 controlled destruction 把 drawer qpos 从 `-0.1464` 减到 `-0.2664`。WoodenCabinet 的 `Open` 条件是 qpos `< -0.14`，`Close` 条件是 qpos `> 0`，所以 raw evidence 实为 `true/true/true`，不是收尾报告所写的 `true/false/true`。下一版必须设置到公开 close range 的绝对值后再 exact restore。

此外 K3 没有显式记录或设置随机 seed；`no_seed_change=true` 不能替代可复现的 initialization identity。K4 也未按原计划记录 GPU3 peak/PID，且中文报告对 sensitivity 的陈述与 raw JSON 不一致。旧 K3/K4 证据必须保留，不能原地修饰。

K4 声称的最终 `201 passed` 也不能在其最终 commit 上直接成立：`8490f8d` 已把 machine handoff 改为 `...hard_stop_waiting_review` / `await_next_audit...`，但同一 commit 中两个既有测试仍断言旧的 `...release_planned` / `K0_to_K4...`。这表明全量测试是在最终 handoff 写入前运行，或结果未在最终 commit 上复验。下一次 closeout 必须以最终待提交 tree 重新跑全量测试，不能沿用写文件前的计数。

## 4. 审核判断

K0–K2 与“失败后停止”的程序行为批准；K3 的 overall failure 保留。但它是“一个真实的初态碰撞 + 三个工具缺陷”的混合结果，不能据此继续扩大 native reverse-BDDL matrix，也不能据此进入 SE(3) 或训练。

下一步不再随机重试同一手写 reverse init。使用公开 HDF5 每个 demo 自带的 `model_file` 与同 demo 的成功 terminal full-state，先构造 exact-model matched reverse-start 候选；这是先前 P1 已证明可行但尚未做批量、匹配模型和碰撞清理的路径。

## 5. 资源复核

- 审核时 GPU0 仍由外部 PID `1133706` 占用，未触碰；GPU1–3 空闲约 `5 MiB`。
- 本轮审核不启动 simulator/GPU、训练、FAST/ZIP、官方 GPU、上传或提交。
- `LIBERO-plus` 保持既有 modified/untracked 状态，未纳入本轮修改。
- 审核后的最终 tree：定向测试 `21 passed`；全量测试 `204 passed, 7 skipped, 23 warnings`；JSON、`py_compile`、`git diff --check` 通过。
