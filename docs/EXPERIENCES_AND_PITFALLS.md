# PARC2026 正赛：可复用经验与易错点

这份文档只保留已与正赛仓库契约对齐的通用经验。排行榜权重和归一化公式未公开，因此本地结果只能作为公开 example 的工程门禁，不能当作 Omni 正式分数。

## 先看管线，再看模型

训练、推理、评测、打包是四个独立阶段。模型 loss 下降不代表 server 输入输出正确，也不代表提交包能在单张 L4 上启动。每个候选都按以下顺序检查：

1. contract：两路 128×128 uint8 图像、四类本体状态、7 维 float32 delta action。
2. server：`/health`、`/reset`、`/act`，单请求低于 10 秒。
3. 本地 pipeline：固定 seed、episode 数、任务列表和 max steps。
4. validator：zip 结构、依赖、动态 smoke。
5. 官方 Docker E2E：确认离线加载、显存、启动时间和依赖。

## 环境必须分开

- 训练使用 `examples/pi05_libero_finetune/scripts/setup_train.sh` 建立的 lerobot 0.4.4 专用 venv。
- 评测使用根目录 `setup.sh`/`env.sh` 对齐 CUDA 13、PyTorch 2.11.0+cu130 和 LIBERO。
- 每个终端先执行 `python -V`、`which python`、`python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'`。
- 训练环境和评测环境不要互相 `pip install`；提交的 `requirements.txt` 也不要随意覆盖评测镜像的 CUDA 组件。

## 一次只改一个变量

固定 seed、数据 revision、mix hash、action chunk、推理步数和评测协议；每次实验只改变 steps、LoRA rank、数据配额或一个推理旋钮。候选权重放在独立 output 下，不自动覆盖 `submission_candidate/model_weights`。

## W&B 与秘密

`WANDB_API_KEY`、`HF_TOKEN` 只能放在 shell 环境或未入库的 `.env`。配置只写 `wandb.mode`：`online`、`offline` 或 `disabled`。runner 会拒绝 online 模式缺少 key；训练参数关闭 model artifact，权重只保存在本地 checkpoint。

## 动作与延迟

`train.steps` 是优化器更新次数，不是机器人动作步数；`action_chunk` 是一次生成的动作数量。pi0.5 示例使用 action chunk 10，且一次 `/act` 可能触发一次重推理。重推理请求也必须低于 10 秒。训练机能运行不等于评测用 NVIDIA L4 24GB 能运行。

## 数据问题不要过度自动化

EDA 应记录 task/episode/frame 数、episode 长度、camera key、图像 dtype/range、state/action shape/range、NaN/Inf、episode 边界和 train/validation 重叠。长度异常与 action spike 只进入人工复核列表，不应凭单一阈值自动删除。

## 提交门禁

提交包必须把推理权重、tokenizer 和 server 代码全部放入 zip，评测阶段不得访问外部网络。先运行静态 validator，再运行动态 smoke，再用官方 Docker `evaluate.py` 做 E2E，最后执行 `omni submit --dry-run`。真实提交每 Track 每日一次，上传完成就计数，错误也计数。
