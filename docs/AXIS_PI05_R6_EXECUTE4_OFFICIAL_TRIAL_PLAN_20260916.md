# AXIS Pi0.5 R6 execute4 最后一次官方试分计划

> 日期：2026-09-16（Asia/Tokyo）
> 状态：`user_risk_override_authorized_package_only`
> 自动上传/提交：false

## 用户覆盖与理由

预注册 horizon screen 要求候选相对 execute8 同时达到 C20 `+3/20` 和 paired net wins
`>=3`。execute4 实测为 C20 `17/20`，相对 fresh execute8 的 `15/20` 为 `+2`；
paired wins/losses/net 为 `3/1/+2`，collision `4→1`。它没有通过统计发布门，但用户明确指出
今晚尚有一次会过期的官方试分机会，并选择承担不确定性。

本计划把该选择记录为显式 user risk override：不把 I2 改写成通过，不补称 B80 已验证；
只制作一个清楚标注的 execute4 R6 候选，供用户人工提交。

## P0：冻结输入

- 来源固定为已验证 R6 FAST staging 与正式得分 ZIP。
- FAST 权重 SHA256 固定为
  `e55bf7dcca78b2195993dd7a12b96a4ea13e94761d711c99a82b7411c3f8ceb6`。
- 源 ZIP SHA256 固定为
  `95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a`。
- weights、processor、norm、vendor、requirements、chunk size、EMA、RTC 均不得改变。
- 唯一行为变化为 `n_action_steps: 8→4`。

## P1：隔离 staging

建立全新 staging；大文件和未修改 vendor 使用只读硬链接，历史 staging 保持不变。
只复制并修改：

1. `model_weights/config.json`：`n_action_steps=4`；
2. `policy_server.py`：接受/固定 execute4，并同步启动日志；
3. `submission_manifest.json`：声明 user override、execute4 与 screen evidence。

所有其他 ZIP entry 必须与正式 R6 ZIP 在 size/CRC 上完全一致。

## P2：验证

1. focused tests、py_compile、strict JSON、source hash lock；
2. GPU0 fresh server：health → reset → 真实 `/act`，日志必须声明 chunk10/execute4；
3. ZIP topology、无 symlink/duplicate/nested archive、CRC/testzip；
4. `validate_submission --static`；
5. unchanged entry size/CRC exact，FAST 权重 SHA exact。

任一项失败即不生成可提交 release。

## P3：交付边界

机械与 runtime gate 全通过后，只交付 ZIP 路径、SHA256、bytes 和报告。不得自动上传或提交。
官方分数返回后按事实落盘；不得把本地 `17/20` 写成官方增益保证。该 ZIP 是一次高方差
探索，不替代已经证明的 T1 R6 `0.47` / T2 R6 `0.33` 事实记录。
