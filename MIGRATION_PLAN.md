# PARC2026 正赛仓库部署与迁移计划

本文件保存本次执行所依据的附件计划。正赛仓库与正赛官方文件优先，旧仓库只用于提取通用工程经验；不复制旧仓库的 evaluator、scoring、manifest、policy vendor 或 Track 专用启动脚本。

执行顺序：

1. 克隆正赛仓库并记录 HEAD、分支、remote、初始状态。
2. 只读审计 README、Docker/依赖、训练示例、Policy server、pipeline、evaluator、validator、打包与 Omnicampus 文档。
3. 先完成 `MIGRATION_AUDIT.md`，再加入通用经验与本地编排工具。
4. 以官方训练入口为核心，增加可验证配置、dry-run、训练 smoke、官方本地评测、Docker/validator、提交 dry-run 与归档门禁。
5. 真实 Omnicampus 评分提交保留为用户确认后的手动动作；每 Track 每日一次，错误也会消耗次数。

## 门禁

```text
官方契约审计
→ 数据与环境检查
→ 训练 dry-run / 10–100 step smoke
→ 正式训练
→ checkpoint / hash 归档
→ /health + /reset + /act smoke
→ 官方 pipeline 本地评测
→ 官方 validator + Docker E2E
→ omni submit --dry-run
→ 用户确认后真实提交
→ 结果、日志、manifest、zip hash 归档
```

## 允许的 CLI 覆盖

runner 只允许覆盖 `run.tag`、`run.output_dir`、`run.seed`、`train.steps`、`wandb.mode`、`hardware.gpu_ids`。其余配置必须写入 YAML；未知字段直接报错。

## 安全边界

- 正式训练拒绝已有输出目录、脏工作树和未通过检查的数据；smoke 必须显式允许并记录 diff hash。
- dry-run 不创建目录、不写配置、不启动训练、不上传 W&B、不提交 Omnicampus。
- 不改动官方 evaluator、scorer、pipeline 的任务、成功/碰撞判定或正式分数。
- 评测 observation 不写入训练数据；日志不记录完整 observation payload。
