# Track 3 Operator-Scoped J0–J4 独立审核

> 审核日期：2026-09-04（Asia/Tokyo）
> 审核输入：execution `bc2419e62ff05a9584d1d6033d22b0e55d03a240`，record commit `f969393f3e16dc94985675083d09daa712cc7c04`
> parent：`3acbde8046563e3e550f78feaf19ad97cb391b79`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 审核结论：`remediation_required`
> 新状态：`track3_background_predicate_convergence_release_planned`

## 1. 获批事项

- J0 锁定旧 I2 hash `e72ae24d...cc17dab`、旧失败、BDDL/source 和 reverse-demo `0/0/0`；J4 重验通过。
- J1 在新 namespace 冻结 operator participant、contact 与 step5–20 contract，没有改旧 I2、BDDL 或 predicate。
- J2.1 对旧 I2 的已保存字段复算为 reverse `9/9`；旧 contact 缺失正确记录为 `not_evaluated`，没有伪造 release。
- J2.2 真实完成15个CPU/no-renderer fresh processes：reverse 9、forward 6；21个step、full contact matrix、schema/finite/return-code均完整。
- reverse `8/9`、forward `6/6`；J2失败后J3未运行，GPU0/GPU3均未触碰。
- 未执行BDDL修改、SE(3)、抓取/demo、Teleop、批量生成、训练、FAST/ZIP、官方GPU、上传或比赛提交。
- 执行记录测试为`194 passed, 7 skipped, 26 warnings`；root/OpenPI clean，`LIBERO-plus`既有dirty未触碰。

因此，J0–J4执行和hard stop本身获批准。

## 2. 唯一 truth transition 的语义

独立逐atom、逐step复算15个raw worker reports，只有一个truth transition：

```text
task19 reverse run1
role       background_initializer
atom       (On akita_black_bowl_1 main_table_akita_black_bowl_region)
step5–19  false
step20    true
direction false → true
```

同一run中：

- operator-critical `(Open wooden_cabinet_1_middle_region)` 在step5–20始终true；
- reverse goal `(Close wooden_cabinet_1_middle_region)` 始终false；
- target drawer site最大translation约`3.43e-7 m`，target joint最大drift约`3.52e-7`；
- robot、operator participant、contact-matrix/family rule全部通过；
- 其余8个reverse和6个forward没有truth transition。

背景bowl持续下落并靠近目标site：相对site XY从step5约`0.01502 m`降至step20约`0.01244 m`，高度差从约`0.07381 m`降至`0.07195 m`，在step20首次满足LIBERO runtime `On`。这是background initializer向其BDDL预期状态单调收敛，不是关键任务拓扑回退。

旧J1要求完整truth vector逐帧字节等值，因此执行者按合同判fail是正确的；但这项合同与更早已获批的事实冲突：background initializer在runtime step5可以为false，且不能自动升级为行为失败门禁。不能把本轮`8/9`解释成reverse start或task19语义失败。

## 3. Evidence / handoff 缺口

J2的6个summary/closeout文件已提交，但`j2_summary.json`引用的15个raw worker reports和15个corrected reports都被`.gitignore`排除；`git ls-files .../runs/*`为0。raw 15文件共`16,072,261 bytes`，每份包含21个step，是独立复算truth transition和contact的必要证据。corrected reports可由raw确定性重建，无需重复入库。

机器handoff顶层仍保留上一轮：`latest_execution_root_commit=0dec0b1`、`next_plan`指向已完成J计划、tests=`189`；虽然尾部已有J execution对象，但当前字段不一致。下一阶段必须先做evidence closure并修正current pointers。

## 4. 审核裁决

结论为`remediation_required`：

1. 批准J0–J4执行、15个raw本地结果和J3 hard stop。
2. 永久保留旧J1 exact-vector failure与J2 `8/9`，不得原地重标。
3. 新合同按role/direction预注册：critical source逐帧true；reverse goal逐帧false；background init允许最多一次`false→true`，禁止任何`true→false`或回摆。
4. 先把15个raw reports及manifest显式纳入Git并锁hash；没有证据闭包不得放行GPU。
5. 新合同只读复算达到reverse `9/9`后，才允许GPU3先做一次task19 canary，再做三族各`3/3`。

## 5. 不授权事项

本审核不授权修改BDDL/region/predicate、SE(3)、抓取或reverse demo、人工Teleop、批量生成、训练/norm、FAST/ZIP、官方GPU、上传或比赛提交。

## 6. 审核验证

- `bc2419e`、`f969393` commit关系与`git show --check`：通过。
- J1/J2/J4 SHA-256：`b2b61930...e5e2fe` / `5e4bc547...a5013` / `4d299bd5...bd7bf`。
- 15个raw reports：process status、21 checkpoints和JSON parse通过；truth-transition总数1，且仅为上述background `false→true`。
- 审核时GPU0为外部YOLO PID `1133706`占用，未触碰；GPU1–3空闲。
- 定向测试：`16 passed, 2 warnings`。
- 全量测试：`194 passed, 7 skipped, 23 warnings`。
- machine JSON parse、`py_compile`与`git diff --check`：通过。
