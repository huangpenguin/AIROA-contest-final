# 实验归档与可复现性清理（2026-09-21）

## 当前状态

比赛阶段已暂时结束。本仓库进入“只读归档、禁止训练/评测/上传/提交”状态。清理保留了实验结论、参数记录、配置、代码、文本型证据、ZIP 的 manifest/SHA256 sidecar；可重新生成的大型模型、评测原始媒体和 ZIP 已删除。

机器级清理结果： data/reports/axis_libero_30k/cleanup_20260921/cleanup_result.json
严格 JSON 复核对剩余记录通过；仅跳过两个专门用于测试 fail-closed 解析器的 malformed negative fixture：data/reports/axis_libero_30k/track3_overnight_autogen_lora/o1_io_tests/malformed/child_result.json 与 data/reports/axis_libero_30k/track3_p3_io_recovery_and_support_executor_canary/r1_io_tests/malformed/result.json。它们是保留的测试输入，不是本次清理产生的损坏。

结果状态为 completed_with_permission_denials：主体清理完成，但当前用户无法删除少量由 root 创建的旧 Python 缓存。它们不属于实验结论，也不影响归档记录。

## 保留内容

- data/raw/：原始数据集与 canonical payload。
- configs/、源码、测试、环境和全部 docs/。
- data/reports/axis_libero_30k/ 下的 JSON、JSONL、CSV、YAML、Markdown、日志和摘要。
- artifacts/ 下的 .manifest.json 与 .sha256 sidecar；ZIP 本体不保留。
- 输出目录中的小型参数/身份/manifest 文本副本：
  data/reports/axis_libero_30k/cleanup_20260921/retained_output_metadata/
- .venv_axis_pi05/，用于未来需要时复现工具链。

## 已删除内容

本次清理脚本的最终结果记录：

| 类别 | 文件数 | 字节数 |
|---|---:|---:|
| artifacts/*.zip / .zip.partial | 8 | 75,068,571,608 |
| 报告原始二进制（NPY/NPZ/MP4/PNG/JPG/partial） | 44,226 | 2,751,170,089 |
| 报告中的上述二进制 symlink | 320 | 0 |
| 已清理的 generated output 条目 | 3,221 | 84,170,797 |
| Python/test cache 条目 | 720 | 29,162,823 |

首次删除阶段已清空绝大部分历史 outputs；最终核验中 outputs/ 仅剩约 2.3 MiB 的 root-owned 缓存残留。当前用户权限下无法删除的实际文件为 178 个、1,911,309 bytes，分布在 16 个 __pycache__ 目录中；这些路径和错误信息已完整写入 cleanup_result.json。如需彻底清零，管理员只需在仓库根目录删除 outputs/axis_fast_recovery/ 中列出的 root-owned 缓存。

## 复现边界

保留的记录可以复核“做了什么、使用什么参数、得到什么分数和哈希”。但由于 raw checkpoint、folded model、FAST staging 和 ZIP 本体已删除，未来不能只运行打包器直接恢复旧 ZIP；若需重新提交，必须先按历史计划重新生成或训练所需的大文件，再运行对应的 fold/FAST/ZIP 流程。

历史 T2 seam-blend 记录：

- 计划：docs/AXIS_PI05_R6_T2_LIGHT_SMOOTHING_SCREEN_PLAN_20260917.md
- closeout：docs/reviews/axis_pi05/R6_T2_LIGHT_SMOOTHING_SCREEN_AND_SEAM_BLEND_RELEASE_20260917.md
- 保留的 ZIP 指纹：artifacts/submission_axis_r6_step500_fast_seam_blend_t2_20260917.zip.manifest.json 和 .sha256
- 官方提交：本仓库记录为未自动上传、未自动提交。

历史失败/成功判断、实验参数和数据门禁仍以对应审核报告及 data/reports/axis_libero_30k/ 中的机器摘要为准；不要用缺失的二进制路径推断“未运行”。

## 未来恢复前检查

1. 先阅读 AGENTS.md、MEMORY.md、计划索引、本清理说明和 machine handoff。
2. 验证原始数据与配置哈希，确认新的执行授权。
3. 重新生成 checkpoint/folded/FAST 资产，并在新目录中运行 focused checks。
4. 只有用户明确授权且正式门禁通过后，才重新生成 ZIP；默认不上传、不提交。
