# Track 3 Start Bundle / Support Executor P0–P6 独立审核

> 审核日期：2026-09-05（Asia/Tokyo）
> 审核输入：root `ad2d4ef7d6603e7b0e79a8519155de9cfdefbde2`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`
> 用户人工操作：`not_required`

## 1. 获批准的结果

- P0/P1输入锁、full-window contact-v3、state/manifest/provenance基础合同通过；旧M0–M5及三份HDF5没有被改写。
- P2的9个`float64[79]` composed states与旧hash逐项相符。独立no-pickle读取确认9个key、dtype/shape/finite与逐state SHA-256全部通过。
- 9个fresh CPU restore均通过，最大restore误差`2.220446049250313e-16`；truth与41-step contact gate均为passed。task10/demo0固定抓取几何报告通过。
- P3只启动task10/demo0一次；失败后没有启动task20/task19、P4/P5或任何操作动作。两路41帧H.264存在且hash/bytes/ffprobe与manifest一致。
- hard stop与禁止项成立：没有SE(3)、demo/pending/accepted、Teleop、训练/norm、FAST/ZIP、官方GPU、上传或比赛提交。

## 2. P3的解释

本轮P3不能解释为环境、state或reverse语义失败。子进程已经生成41帧camera NPZ与两路视频，但在父进程取得最终result JSON前被终止；`p3_task10_demo0.json`中的truth、contact、restore和peak因此没有真实worker结果，只能保持`worker_exception`。

stdout管道阻塞诊断与症状一致。最终commit中的工具已经改为stdout/stderr文件重定向并压缩最终telemetry，但失败时实际运行的pre-fix dirty-worktree版本没有独立commit/tree hash，不能把当前修复代码当作已验证。下一轮必须先用独立result file、大输出压力与超时终止合同验证，再授权一次明确标为attempt 2的task10恢复运行。

## 3. 需要闭合的问题

1. `artifact_manifest.json`的17个entries均在本机存在且bytes/SHA-256匹配，但其中4个未被Git跟踪：`start_states.npz`、task10 camera NPZ及两路MP4。fresh checkout无法恢复P2 bundle或复核P3媒体。
2. `p6_summary.json`声明`entries_exclude_self_and_p6_summary=true`且`entry_count_before_manifest=16`，实际manifest包含P6 summary并共有17项。旧文件保持immutable，新closure manifest必须写准。
3. P3 pre-fix执行代码没有可复现commit；manifest中的`source_commit=6cfda92...`是执行父提交，而该提交中还没有本轮工具，不能当作工具源码版本。
4. 当前post-fix worker仍以stdout文件最后一行作为result protocol且没有wall-clock timeout。它消除了bounded pipe，但还需独立原子result JSON和TERM→KILL超时路径。
5. `p0_release_lock.json`约4.9MB、`p2_summary.json`约30.6MB，主要重复嵌入raw telemetry。它们保留为历史证据；后续报告只写引用、hash与紧凑gate摘要，禁止继续成倍复制。

这些缺口不推翻P2 `9/9`，但在闭合前不得继续P3剩余两族或P4/P5。

## 4. 独立复核摘要

```text
artifact manifest entries           17
existing/hash-exact entries          17/17
untracked manifest entries            4
state bundle keys/hash-exact           9/9
fresh CPU restore                     9/9
max restore abs diff           2.22e-16
valid P3 worker result                  0
P3 attempts started                     1
P4/P5 attempts                          0/0
exact/pending/accepted                  0/0/0
```

审核时root与nested OpenPI clean；`LIBERO-plus`既有dirty状态保持不变。下一阶段不需要用户Teleop、视频选择或人工轨迹批准。
