# Interstep L0/L1 阻塞与 Track 3 并行改道审核

日期：2026-09-02（Asia/Tokyo）
审核输入：root `88796b2`，nested OpenPI `b58aff0`
审核结论：`blocked`（仅 dense full-FT 四卡路径）；项目级工作不阻塞

## 1. L0/L1 审核

L0 的语义锁和 source checkpoint 锁有效：D4 从 internal step 1 恢复并完成 update `1→2`，失败发生在下一轮 `optimizer_step_before=2` 的 microbatch `1`；D3 step1 tree SHA-256 为 `9aeeb1a9b1fdf428486b476930ca9117c28d822dd108b4efaeebf911ca0fafef`。

L1 只改变生命周期边界：optimizer state 与mean gradient同时donate；device metrics同步成host scalar后释放；新增transition probe。未发现修改FP32、loss、AdamW、EMA、ordered accumulation、global batch或数据/norm的证据。CPU全量测试为执行方报告的`146 passed, 7 skipped, 23 warnings`。

资源hard gate正确：GPU0上的外部YOLO PID `1133706`占用约`14.3GiB`且utilization非零；GPU1–3空闲。执行没有绑定、暂停、终止或影响该进程，也没有启动L2–L4。dense full-FT的审核状态为`blocked`，不是代码失败。

## 2. 为什么不能把同一 dense 配方直接改成三卡

当前锁定配方是4卡FSDP、microbatch/device `1`、accumulation `16`、effective global batch `64`，并且step1 checkpoint与transition memory gate都绑定4-device mesh。三卡不能保持`1×devices×integer accumulation=64`；改成global batch `60/66`、非均匀microbatch、不同partition或重新编译checkpoint topology都会成为新实验，不是无损续跑。它可以以后单列研究，但不应在Track 3仍为0分时抢占优先级。

## 3. Track 3 决策依据

保留用户提供的官方公开说明作为规则边界：Track 3核心为“现有LIBERO任务的逆向操作”。只使用这一公开高层规则和公开LIBERO资产；不从泄漏脚本提取、推测或利用内部task family、具体隐藏任务、数量、命名或生成机制。

现有R6 FAST已由用户报告取得Track1=`0.47`、Track2=`0.33`，但Track3=`0`。这说明R6已经具备视觉、server、FAST和一般正向控制能力，Track 3主要缺口更像“逆向initial state/goal/instruction对应的行为分布”，而不是再做少量纯正向LIBERO LoRA就能可靠解决。GPU0长期被占时，优先用GPU1+2复用已经通过的R6两卡LoRA路径、GPU3做20Hz reverse rollout，是当前最有信息量且最接近正分的替代路线。

## 4. 审核决定

1. 当前interstep计划归档为“L0/L1完成、L2因外部资源阻塞”；其代码与step1 hash保留，GPU0释放后须重新制定恢复授权，不能从归档尾部自动启动L2。
2. 唯一活动计划改为`docs/AXIS_PI05_TRACK3_REVERSE_LORA_POSITIVE_SCORE_PLAN.md`。
3. 首先生成并验证通用reverse pilot，不允许把正向视频/动作简单倒放后直接训练；只有真实环境执行、predicate成功且3/3 replay通过的20Hz轨迹可进入训练。
4. 训练从正式正分的R6 raw step500分叉，首轮只比较保守action-expert LoRA与现有R6 LoRA+heads/time scope，使用GPU1+2；GPU3只做冻结proxy/holdout rollout。GPU0和外部PID不可触碰。
5. 标准LIBERO-only继续LoRA、LIBERO-plus-only LoRA或三卡dense均不作为Track3正分首选；它们没有直接提供reverse supervision。

## 5. 审核复验

- 定向CPU测试：`17 passed, 23 warnings`；
- 首次未锁platform的全量测试误选CUDA，在外部任务占用GPU0时因cuBLAS handle资源不足得到`1 failed, 145 passed, 7 skipped`，失败用例是小型LoRA einsum初始化，不是断言或数值不一致；
- 随后显式`JAX_PLATFORMS=cpu`全量复跑：`146 passed, 7 skipped, 23 warnings`；
- 本审核未启动训练、rollout、数据生成、Blackwell或官方GPU，也未触碰外部PID。
