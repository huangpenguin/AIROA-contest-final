# AXIS Pi0.5 E1 Microbatch-1 Canary 审核记录

> 审核日期：2026-08-31（Asia/Tokyo）
> 被审核 commit：`222e5942d8274a10322ebf4679aaeb6d4f989420`
> 审核结论：`remediation_required`
> 发布判断：bitwise hard gate 确实失败；数值差异符合有界 FP32 fresh-process replay drift，允许新计划以预注册数值门禁继续，但不得改写旧 failure。

## 1. 核验结果

```text
canary coverage                       16/16
microbatch                            1
internal / external shape             [16,10,32] / [16,10,7]
finite                                true
input/model/norm/raw-tree provenance  exact
first-four internal exact             281/1280
first-four external exact             226/280
internal/external max abs diff         1.1920928955078125e-7
internal p99                           5.960464477539063e-8
external p99                           1.1920928955078125e-7
elements > 1e-6                       0
all-action sign mismatch              0
external gripper sign mismatch        0
GPU1 peak                             24706 MiB
GPU0 touched                          false
```

旧计划要求 bitwise exact，因此执行者 hard stop 正确；不能把旧状态改成 passed。另一方面，最大差等于一个约 unit-scale FP32 ULP，且没有 sign flip、non-finite、shape 或 provenance 漂移。这不支持“模型或 normalization 已损坏”的判断。

## 2. 新数值 replay 门禁

下一次 fresh-process replay 必须在观察新结果前固定为：

```text
first-four internal max abs diff       <= 2.384185791015625e-7
first-four external max abs diff       <= 2.384185791015625e-7
all-action sign mismatch               0
external execute8 gripper sign flip    0
finite/shape/provenance                exact
```

`2.384185791015625e-7` 是两个 unit-scale FP32 epsilon，只用于 fresh-process runtime replay 接受度，不替代旧 EGL `1e-6` failure，也不作为模型性能或 source↔FAST parity 阈值。

旧 `[0,16)` shard 的 metadata 已声明 `usable_as_full_prefix=false`，不得事后改标签或作为 full prefix。新 full run 必须从 observation 0 在新 namespace 开始；计算前四个 action 后立即应用上述数值门禁，失败则停止。

## 3. 独立验证

```text
root/nested at audit start             clean
nested OpenPI                          dafa5a27266996eecf28920cacc32f3629776d91
tests                                  121 passed, 7 skipped, 21 warnings
test command                           .venv_axis_pi05/bin/pytest -q tests
GPU1 released                          true
```

## 4. 当日 ZIP 判断

今天生成本地诊断 ZIP 仍有现实机会。JAX 340/340 是 action-envelope 证据，不是 FAST 文件转换本身的输入；因此可与 FAST staging/export 并行，不能继续让 bitwise replay 门禁串行阻塞全部打包工作。

但 ZIP 只能在以下 critical path 通过后生成：R6 source identity、FAST key/vision/strict reload、真实 folded-PyTorch↔FAST action parity、FAST-only server contract、`/act` latency/finite、C5 灾难回归 smoke、ZIP CRC/static/offline checks。任一 critical gate 失败则不得打包。

在完整 E1 envelope、C20 与官方 Docker/L4 pre-check 未完成时，即使文件生成，也只能标为 `diagnostic_only_not_submission_ready`，禁止自动上传或提交。
