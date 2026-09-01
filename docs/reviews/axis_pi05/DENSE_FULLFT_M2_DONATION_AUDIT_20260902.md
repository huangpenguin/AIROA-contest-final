# Dense full-FT M2 Gradient Merge 审核

日期：2026-09-02（Asia/Tokyo）
审核输入：root `ecff24f`，nested OpenPI `e31c2bf`
审核结论：`remediation_required`

## Findings

1. M2不是microbatch index1的forward/backward OOM；compute indexes`[0,1]`均完成，失败发生在index1的`_add_gradients(grad_sum, grads)`。
2. M1移除了审计`values`长期引用并显式释放临时对象，但`_add_gradients`仍是普通逐叶`x+y`，未实现计划预期的donated/aliased accumulator。普通加法需要旧accumulator、current grads和新output三者同时驻留；`26.42GB` live之后的`72MiB`短缺与此一致。
3. 因而当前结果不证明稳定梯度累积或单microbatch超过32GB；直接转Blackwell仍属过度扩展。下一最小动作是JIT donated tree add，并先做两份真实microbatch merge-only canary。
4. M0交付仍不完整：新summary顶层`optimizer_updates/next_action`缺失、M1为null；machine handoff顶层commit仍为`c353ff0/81d8220`，而execution子树已是`ecff24f/e31c2bf`。
5. execution子树的`first_uncompleted_microbatch_index=1`语义模糊：index1 compute已经完成，未完成的是index1 merge；index2 compute从未尝试。下一计划必须拆字段。

## 通过与边界

- M0 raw correction、step0 immutable copy/restore/sentinel有效；
- CPU回归`137 passed, 7 skipped, 23 warnings`；
- optimizer update=0，未生成step1，M3未启动；
- 未使用Blackwell/比赛GPU，未执行C20、长训、FAST、ZIP、上传或提交；
- 四卡退出后约5MiB，审核开始时root/nested clean。

## 决策

不批准Blackwell、降精度/降batch或继续M3。归档旧计划，唯一授权新的donated accumulator D0–D4，最多本地step10；D2两份merge canary失败即停止。

审核后定向测试为`5 passed, 2 warnings`；全量测试为`137 passed, 7 skipped, 23 warnings`。为保持计划生命周期fail-closed，旧测试中硬编码的已归档`next_plan`断言已改为验证当前donated计划，同时继续核对历史allocator execution与本次M2阶段字段。审核过程未启动GPU或训练。
