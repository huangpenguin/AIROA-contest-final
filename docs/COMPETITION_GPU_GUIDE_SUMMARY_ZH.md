# PARC2026 本选 GPU / 开发环境归总（资料 v1.1）

日期：2026-08-26（Asia/Tokyo）
依据：[PARC2026 本选参加者资料 v1.1（Google Slides）](https://docs.google.com/presentation/d/1ndimq0yx2taFzqG6BVA9sDNi1N9keEkUmpFfGNV2-lo/edit?usp=sharing)，重点核对 p.14、15、35–42。

本文将“运营方配布的本选开发 GPU”与“提交包的本地检查/正式采分环境”分开记录。此前文档把 L4 误写成配布开发 GPU，已在本版修正。

## 1. 本选配布开发环境

v1.1 公告给本选参赛者提供的 GPU 为：

| 项目 | 规则 |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell |
| 本选 1 / 2 / 3 使用上限 | 60 / 75 / 120 小时 |
| 使用方法 | 以资料 p.35 以后“配布 GPU 环境的利用方法”为准 |
| 外部环境 | 可以按需并用外部付费云环境 |

这台 GPU 是训练/开发资源，不应据此推断正式 scorer 的 GPU 型号、容器镜像或 CUDA 版本。

每个本选阶段的配额独立计算；进入下一阶段时上限会重置，剩余时间不会结转。

配布实例的存储边界如下：

| 路径 | 容量/属性 | 使用注意 |
|---|---|---|
| `$HOME`（`/home/<account name>`） | 40GiB，持久化 | 每 10 分钟及实例结束时保存，启动时恢复最近一次保存状态 |
| `~/data` | 100GiB，持久化但需手动同步 | `parc-home-sync data-push` 保存，`parc-home-sync data-pull` 恢复 |
| `~/dataset` | 只读 | 配布数据集与 baseline model 所在位置 |
| `/opt/dlami/nvme` | 约 1.7TB，临时实例存储 | 停止/结束实例可能丢失，重要 checkpoint 必须提前同步到持久化存储 |

## 2. v1.1 新增的 Docker 规则（p.14）

运营方已经将与采分环境同一构成（CUDA 13.0、PyTorch `2.11.0+cu130` 等）的 Docker 镜像预先放置在配布 GPU 环境中，并建议在这个 Docker 内执行提交 zip 的端到端检查。

在运营方提供的 Omnicampus 环境中，只支持配布的“提交前检查用 Docker”。不支持参加者自行导入其他 custom Docker；资料明确说明配布环境的存储结构和容量不是按 custom Docker 使用场景设计的，运营方也不承诺为这类用法提供支持。

因此当前仓库的执行约束是：

- 配布 GPU 上只使用运营方提供的 pre-check Docker 和资料 p.35 以后的启动方法。
- 不在配布环境中自行替换基础镜像、挂载结构或 Docker runtime。
- 使用外部付费云时，可以单独维护自己的环境，但不能把它当作运营方配布环境的等价支持路径。

## 3. Notebook 启停规则（p.15）

- 关闭浏览器标签页不会立即停止 Notebook。
- 使用完毕后必须在 Notebook 内主动执行 shutdown。
- 完全没有 cell 执行或处理的 idle 状态持续 1 小时后会自动停止。
- 即使不是 idle，Notebook 从启动起经过 12 小时也会自动停止。
- 本选 1 当前不能使用“延长”按钮；不要依赖延长功能规划训练。

这意味着训练脚本必须可恢复，启动前记录 run/checkpoint 路径，并在每次使用结束后确认 kernel/Notebook 已停止；60 小时配额和 12 小时单次自动停止是两层不同限制。

## 4. p.35–42 的实际使用边界

具体账号登录、Notebook 启动、GPU 终端进入、数据路径和停止操作以 Slides p.35–42 为准。本仓库只固化与代码运行直接相关的原则：

```text
配布 RTX PRO 6000 Blackwell
  -> 使用运营方提供的 pre-check Docker
  -> 在 Notebook/终端中执行训练或验证
  -> 主动 shutdown，避免继续占用时间
```

Slides 中给出的操作入口是 `https://parc-playground.omnicamp.us/hub/login`：选择
“Sign in with Auth0”，使用 Omnicampus 账号登录，等待 GPU 分配和实例启动完成。使用结束后从
`File > Hub Control Panel` 进入控制面板，点击 `Stop Server`；只有停止完成后配布时间才停止消耗。

不要把当前 SSH 服务器的 4×V100 视为配布环境，也不要把当前服务器上的 CUDA/PyTorch 版本复制成运营方 GPU 环境的假定配置。

## 5. 提交检查环境与配布开发 GPU 的区别

当前仓库 README 中的“採点環境”仍描述提交包本地检查/正式 scorer 使用的 GPU 容器：单张 NVIDIA L4、CUDA 13、PyTorch `2.11.0+cu130` 等。这是提交运行时边界，不是 v1.1 公告中的配布开发 GPU。

因此要分别验证：

1. 在 RTX PRO 6000 Blackwell 配布环境上训练、恢复 checkpoint、做训练侧性能检查。
2. 在运营方提供的 pre-check Docker 中检查提交包的 Python 3.10、依赖、server 和 GPU 推理。
3. 不以 V100 上能加载或运行就断言 L4 scorer 一定能运行。

## 6. Axis Pi0.5 对这些规则的影响

Axis 当前训练结果是 OpenPI/JAX Orbax checkpoint，不是比赛旧 LeRobot server 直接读取的 PyTorch `model.safetensors`。Axis 内部 action head 为 32 维，而 LIBERO 对外 action contract 是 7 维，导出/推理时才保留前 7 维。因此：

- 只把 `params/` 压成 zip 不能称为可提交包；仍缺少 server、依赖、tokenizer、processor 和 action 适配。
- Axis→LeRobot 转换必须折叠 JAX LoRA，并独立验证 state=8、action=7、双相机、归一化和推理精度。
- 提交候选必须在运营方支持的 pre-check Docker 中重新做 `/health`、`/reset`、`/act` 和真实 observation smoke。
- 静态 validator 通过不等于正式采分通过；还要确认启动时间、显存、推理时延和离线依赖。

## 7. 最小记录清单

每次使用配布 GPU 后至少保留：

- GPU 型号、Notebook 启动/停止时间和剩余配额；
- run id、checkpoint 路径、恢复点和训练日志；
- Docker 是否为运营方配布的 pre-check Docker；
- 提交 zip 的路径、大小、静态 validator 结果；
- server 的 `/health`、`/reset`、`/act` smoke 结果。

当前 Axis 候选包和转换工具状态见：

- `docs/AXIS_PI05_IMPLEMENTATION_STATUS.md`
- `tools/axis_pi05/convert_for_lerobot.py`
- `tools/axis_pi05/prepare_lerobot_assets.py`
