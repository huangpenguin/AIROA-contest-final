# Track 3 P3 I/O Recovery R0–R6 独立审核

> 审核日期：2026-09-05（Asia/Tokyo）
> 审核输入：root `309ccafe9e6e2cddd0c9f16db111fc054478eb8b`
> GPU阶段clean source：`ad0a428c76f2c597109855f4d67f00df78184514`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`
> 用户人工操作：`not_required`

## 1. 获批准的结果

- R0旧17-entry closure为`17/17`，新manifest为`33/33`；独立复核全部文件存在、bytes/SHA-256 exact且被Git跟踪。
- R1独立atomic result JSON、16MiB stdout、hang TERM/KILL、malformed/nonzero/exception fail-closed合同与clean intermediate commit成立。
- R2确实只启动task10/demo0 attempt 2一次。child返回`passed`：restore误差`2.220446049250313e-16`，source/goal、step0–40 contact-v3、camera schema全部passed，两路camera均41帧，NPZ与H.264/20FPS闭合。
- parent按fail-closed没有把child成功伪造为完整P3通过；R3–R5未执行，generation/replay/pending/accepted/training均为0。
- GPU0/1/2、官方GPU、Teleop、HDF5/BDDL、FAST/ZIP、上传/提交边界成立；GPU3已释放。

## 2. 失败根因与解释

R2不是机器人、camera、state或reverse语义失败，而是parent provenance assembly的确定性代码缺陷。GPU执行所用`ad0a428`中的`_run_gpu_canary()`调用`gpu_provenance_gate(...)`，但该commit没有定义该函数，因此child完成并原子写出result后，parent触发`NameError`，未能发布start snapshot、physical UUID与peak。

最终`309ccaf`已加入该函数，并保留attempt 2 child证据；但该修复尚未经历一次真实parent+GPU端到端运行。下一轮必须先把provenance组装提取为纯函数并单测全部required fields，再允许task10 attempt 3。attempt 2不可追溯补齐缺失的parent采样，也不能计入P3 `3/3`。

## 3. 当前阶段

项目不在训练阶段，而在Track3 reverse-data pipeline的最后基础设施门禁：

```text
公开reverse任务/起始状态          已完成
9-state restore与静态物理稳定性    已完成
worker I/O与child camera gate       已完成
parent GPU provenance E2E           待恢复
manipulation executor               未开始
reverse demonstrations              0
Track3 LoRA/FAST ZIP                 未开始
```

因此可以保留P2与R1，不应回退到AXIS init或重新构造state。今晚适合执行一条更长但分层隔离的流水线：先修P3，然后按family独立尝试公开逆向executor；只有产生严格replay通过的数据后才进入20Hz混合LoRA和本地候选ZIP。

## 4. 独立复核摘要

```text
old artifacts tracked/hash-exact       17/17
new artifacts tracked/hash-exact       33/33
task10 attempt 2 child result          passed
task10 attempt 2 parent result         failed_closed
valid complete P3 parent results       0
R3/R4/R5                               not_evaluated
validated start states                 9
exact/pending/accepted                  0/0/0
reported full tests                    239 passed, 7 skipped
```

本审核不要求用户Teleop、逐帧看视频或手工选择轨迹。
