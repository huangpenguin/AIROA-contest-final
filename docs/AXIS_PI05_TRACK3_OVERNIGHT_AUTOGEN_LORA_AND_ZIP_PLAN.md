# AXIS Pi0.5 Track 3：Overnight Autogen、LoRA 与候选 ZIP 计划

> 制定日期：2026-09-05（Asia/Tokyo）
> 审核输入：root `309ccafe9e6e2cddd0c9f16db111fc054478eb8b`
> nested OpenPI：`b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`
> 当前状态：`track3_overnight_autogen_lora_zip_planned`
> 用户人工操作：`not_required`
> 预计上限：约8–12小时；任一全局基础设施门禁失败立即停止，单个task family失败只关闭该lane
> 唯一授权：O0–O10；随后hard stop
> 不授权：修改公开HDF5/BDDL、人工Teleop、使用GPU0或官方GPU、dense full-FT、无限retry/在线调参、失败轨迹入训、自动上传或比赛提交

## 0. 今晚目标与资源调度

目标不是重新从AXIS init训练，而是从已有正式Track1=`0.47`、Track2=`0.33`的R6 step-500继续：尽量生成至少一个Track3 reverse family的严格可回放20Hz数据，完成小规模LoRA continuation、本地holdout选择和FAST候选ZIP。

资源固定：

- GPU3：P3、executor generation/replay与rollout evaluation；同一时刻只允许一个simulator process。
- GPU1+GPU2：LoRA训练，固定两卡；不得与GPU3 rollout同时争抢CPU/内存时启动。
- GPU0：无论是否变空闲都禁止使用或终止其外部进程。
- 每阶段写heartbeat、原子summary与resume cursor；进程重启后只从已完成artifact继续，不重复成功episode。

新产物：

```text
data/reports/axis_libero_30k/track3_overnight_autogen_lora/
data/generated/track3_reverse_autogen_v1/
data/manifests/track3_reverse_autogen_v1/
outputs/axis_pi05/track3_reverse_lora_overnight/
artifacts/submission_axis_r6_track3_reverse_overnight_*.zip
```

## 1. O0：release lock与现状复核（CPU/filesystem）

1. 锁定`309ccaf`、`ad0a428`、旧17/17、新33/33 artifacts、attempt1/2、9-state bundle、三份HDF5、R6 raw/folded checkpoint、R6 norm与本地`lerobot/libero_plus`revision/hash。
2. 复核attempt2 child的四类hard gate、41帧、NPZ/视频hash；parent仍保持failed_closed，不追溯伪造provenance。
3. 复核`gpu_provenance_gate`在`ad0a428`缺失而在`309ccaf`存在；输出symbol/code SHA与调用图。
4. 检查GPU、磁盘、root/OpenPI clean和`LIBERO-plus`既有dirty preservation。O0不得import simulator/GPU。

输入漂移、artifact不全或R6 checkpoint/norm缺失即全局hard stop。

## 2. O1：parent provenance修复与overnight watchdog（CPU/code only）

1. 把parent result/provenance assembly提取为纯函数；required fields为seed、parent/child PID一致、physical UUID、CUDA mapping、start/end snapshots、peak、poll count/interval、deadline与return code。
2. 新增回归：完整child result能组装passed；缺任一字段、`gpu_provenance_gate`不可调用、NameError、malformed result、PID mismatch、超时均fail-closed。
3. 保留独立result JSON；stdout/stderr仅诊断。wall deadline `300 s`，TERM grace `10 s`，随后KILL。
4. 新增overnight supervisor：每phase/episode原子写cursor；禁止无限loop；重复identity自动skip；连续两个基础设施错误全局停止；simulator超时释放GPU3。
5. 静态检查、targeted/full CPU tests通过后创建clean intermediate commit。后续GPU只能记录并运行该commit/tool SHA。

## 3. O2：P3完整恢复

### O2.1 task10 attempt 3

只运行task10/demo0、seed`3100`一次，标为attempt 3，不覆盖attempt1/2。要求child全部hard gate、parent provenance、两路视频、NPZ与artifact tracking完整通过。

任何child/parent/资源错误均全局hard stop；不得attempt 4。

### O2.2 task20/task19

O2.1通过后，依次运行task20/demo0 seed`3200`、task19/demo0 seed`3190`各一次。任一基础设施错误全局停止；物理/predicate family失败则关闭该family的manipulation lane，但继续复核其他family。P3完整状态要求三条均通过；若只有部分family通过，可继续的executor lane仅限对应通过family。

## 4. O3：三族HDF5-grounded executor合同（CPU/code only）

从公开forward HDF5固定提取几何，不根据运行结果搜索waypoint：

- task10 `support_remove`：bowl从plate抓起并回到same-demo init object pose。
- task20 `containment_remove`：目标物从container/cabinet抓出并回到same-demo init pose。
- task19 `articulation_close`：从forward open轨迹提取handle-relative EEF pose与drawer joint终点，执行reverse close。

对task10/task20必须用action/state证明gripper sign：首次`-1→+1`后object随EEF运动，后续`+1→-1`对应release；不能证明则关闭该lane。task19必须证明handle contact/fixture joint方向，不以object relocation逻辑代替。

每族预注册两个profile，之后禁止在线修改：

- `nominal`：原始clip/tolerance/stable ticks。
- `conservative`：较小translation/rotation clip、较长stable ticks；所有数值在GPU前写入contract。

统一阶段、step budget、SE(3) quaternion-log、gripper sign、合法contact矩阵、`>2 mm` unrelated-object movement、stall/timeout门禁。O3不得启动simulator/GPU；三族代码和tests相互隔离，一个family的实现不得包含hidden-task特判。

## 5. O4：family pilot grid（GPU3，独立lane）

固定顺序task10→task20→task19。每族只用demo0，最多运行`nominal`与`conservative`各一次；profile在O3已冻结。

- 首个成功profile成为该family frozen profile；若两个都成功，两者均可用于数据多样性。
- 两个都失败只关闭该family，保存阶段误差/contact/视频，不阻断其他family。
- parent protocol或GPU资源错误属于全局错误，立即停止全部lane。
- success必须reverse goal true、finite、无collision/stall/timeout、无unrelated contact/movement，并保存首个success frame。

至少一个family释放才允许O5；否则跳到O10收尾，不训练。

## 6. O5：可回放reverse数据生成与清洗

对每个released family使用固定`demo_0/1/2` start states：

- `demo_0/1`为train source，`demo_2`永久holdout，identity/hash不得交叉。
- 对每个train source运行该family已通过的profile，最多2个profile，即每族最多4条train trace。
- 每条generation成功后冻结action trace，并从同state做`3/3` fresh-process replay；全部通过才标为`machine_accepted`。
- demo2只生成holdout reference/视频，不进入训练，不参与norm。
- 失败轨迹、pilot失败、未完成trace均只进入diagnostics，禁止混入dataset。

每条accepted episode必须有20Hz时间轴、front/wrist、state、7D OSC_POSE、gripper`{-1,+1}`、完整sim state/source hash、task/instruction/family/profile、collision/background/predicate ledger和H.264。去重按`start_state + action_trace + instruction`hash。

计数上限：3 families、6 profiles、12 train generations、36 replay、6 holdout references；没有retry扩张。至少一个family有`≥2`条machine-accepted train traces才允许O6。

## 7. O6：20Hz统一数据、分层采样与fresh norm

1. 锁定本地`data/raw/hf/lerobot/libero_plus`为20Hz forward replay source；不得回退混用10Hz canonical。
2. 把machine-accepted reverse traces转换为同一LeRobot/OpenPI schema；holdout与diagnostics显式denylist。
3. 建立实际采样manifest：forward/reverse=`80/20`；forward按task/扰动分层，reverse先按family等权、再按demo/profile等权。单family模式也保持reverse 20%，并在报告标为`support_only`或对应family-only。
4. 按有效采样manifest重新计算逐维`QUANTILES`；训练、OpenPI rollout、folded PyTorch与FAST必须使用同一artifact。禁止MEAN_STD/IDENTITY或复用Axis原始norm。
5. 质量门禁：fps、schema、finite、action range/gripper、episode边界、frame/action alignment、train/holdout hash隔离、抽样权重和norm provenance全部通过。

O6失败不得训练。

## 8. O7：两卡LoRA overnight training

初始化固定为R6 raw step-500参数，不回到Axis init；optimizer/scheduler/EMA重新初始化。拓扑固定`L2_action_lora_heads`，core vision/language/expert dense weights冻结；不启用dense full-FT。

资源/数值：GPU1+GPU2、FP32、per-device microbatch`1`、accumulation`32`、global batch`64`、clip norm`1.0`、EMA沿用R6合同。禁止V100 BF16。

### O7.1 两个100-step学习率screen

顺序运行，不并发共享两卡：

- Arm A：peak LR `3e-7`
- Arm B：peak LR `1e-6`（R6已验证量级）

checkpoint `0/1/10/100`；每步loss/grad/update/params/EMA finite，checkpoint restore和optimizer state通过。任一arm失败只关闭该arm；两个都失败则O10。

### O7.2 winner continuation

O8对step100选出winner后继续至step500；step500仍通过提升/保留门禁才继续step1000。保存`250/500/750/1000`。不得无评估直跑超过1000步。

## 9. O8：本地选优与遗忘门禁

在训练前step0、两臂step100、winner step500/1000运行：

- reverse holdout：每个released family固定demo2与固定policy seeds，execute8为主；executor reference不计policy成功。
- retention C20：既有标准LIBERO固定manifest、execute8；必要时增加LIBERO-plus扰动P20，但不使用holdout反向数据选超参以外的信息。
- 记录success/collision/timeout/stall、paired wins/losses/ties、任务级CSV和失败H.264。

step100 winner排序：reverse success优先，其次C20 retention、collision/timeout。必须满足reverse至少比step0多1个成功，且C20相对step0下降不超过`2/20`；否则不继续长训。step500/1000也必须满足同一遗忘门禁，最后选择最早达到最佳reverse且retention合格的checkpoint，避免无收益过训。

## 10. O9：fold、FAST与本地候选ZIP

仅对O8 winner：

1. LoRA fold dense，raw/folded fixed parity最大绝对误差`≤1e-3`。
2. OpenPI/JAX→folded PyTorch→LeRobot processor parity，token/mask exact、action误差`≤1e-5`；QUANTILES hash一致。
3. 导出`model_fast.safetensors`，16-case source↔FAST envelope、`/act 12/12 <10s`、C5 coverage与source-success replay通过。
4. 构建并验证候选ZIP：FAST路径优先、slow权重排除、CRC/布局/大小/SHA-256通过。

只有reverse holdout至少1个成功且retention门禁通过才生成ZIP；文件仅本地落盘，不上传、不提交比赛。

## 11. O10：统一收尾与早晨交接

无论在哪个lane/phase停止，都生成：

- `summary.json`、phase/family/task/episode/checkpoint CSV；
- success/failure H.264 gallery、失败阶段与第一异常；
- 数据manifest/norm/checkpoint/FAST/ZIP hashes及GPU/provenance；
- resume cursor与所有未执行阶段`not_evaluated`；
- targeted/full tests、strict JSON、NPZ/parquet、py_compile、ffprobe、`git diff --check`；
- root/OpenPI clean commit，`LIBERO-plus`既有dirty状态不改。

允许的最终状态：

- `track3_overnight_candidate_zip_ready_for_review`
- `track3_overnight_lora_candidate_ready_without_zip`
- `track3_overnight_autogen_ready_without_training`
- `track3_overnight_stopped_<phase>_<reason>`

所有分支都hard stop等待审核；不自动上传或提交。用户今晚无需保持终端交互，也无需Teleop/逐帧审核。
