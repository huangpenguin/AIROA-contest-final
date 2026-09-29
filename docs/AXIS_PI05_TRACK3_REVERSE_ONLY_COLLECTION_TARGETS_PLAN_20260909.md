# AXIS Pi0.5 Track3 reverse-only 快速采集目标计划

> 日期：2026-09-09（Asia/Tokyo）  
> 状态：`ready_for_assisted_collection`  
> 前序：fast400 Teleop 已实现，用户已实际录制并确认效果很好  
> 自动独立审核：关闭  
> 机器任务清单：`configs/axis_pi05/track3_reverse_collection_targets_20260909.json`

## 1. 当前起点

上一步按轻量方式判定为完成：fast400 录制器已具备 400-step direct-gold、超过 400 后继续、无限 Reset、8 Hz operator rate、速度滑杆和 Save 失败不退出；当前没有 Track3 Teleop/MuJoCo 进程。

现有官方四题数据：

| 题目 | 当前最好结果 | 下一动作 |
| --- | --- | --- |
| red mug | 314-step，物理 gate 通过，`pending_replay`；388/468 仅作备用 | 先 batch replay，不优先重录 |
| alphabet soup | 278-step，物理 gate 通过，`pending_replay` | 先 batch replay，不优先重录 |
| yellow book | 4 条 rejected timeout，其中一条 1200 steps | 第一优先补录 1 条 |
| two mugs | 没有 episode | 第二优先补录 1 条；属于较难多物体题 |

因此下一轮不是继续反复录 red mug，而是先兑现两条已有候选，再补 yellow book 与 two mugs，并同时准备一批原始 LIBERO 题目的明确反向版本。

## 2. reverse 题目生成准则

本地源池已扫描：标准 LIBERO 共 130 个 BDDL；LIBERO-plus 共 16,330 个 BDDL 文件，分类表含 10,030 个位置、语言、相机、光照、纹理、传感器和机器人初态扰动条目。本轮只取原始任务语义，**不启用任何 LIBERO-plus 扰动**。

采用以下准则：

1. reverse start 是原题成功终态，reverse goal 是原题中与被操作对象有关的明确初始谓词；不是把 action 数组倒序。
2. 第一批优先单物体、单关系：`On→On`、`In→On`、`Open→Close`、`Turnon→Turnoff`。多物体任务最后录。
3. prompt 直接写动作、对象和可见落点，不加 `Reverse the public LIBERO operation:`，不使用 `back at its bowl` 这类错误或含糊表达。
4. 每个 reset 必须满足 reverse source=true、reverse goal=false；保存时 goal 连续成立、动作 finite、视频与轨迹帧数一致。
5. 当前只录 canonical reset。Scenic 的约束采样和 LIBERO-PRO 的位置/任务扰动只作为下一轮扩展方法；本轮不做光照、摄像头、背景、纹理、传感器、语言改写或对象替换。

方法参考：

- LIBERO-Infinity 支持从 BDDL 自动生成并以 `--reverse` 运行反向任务：[evaluation pipeline](https://github.com/KE7/libero-infinity/blob/main/docs/evaluation_pipeline.md)
- Scenic 用带约束的概率分布生成合法场景：[scene generation](https://docs.scenic-lang.org/en/2.x/reference/scene_generation.html)
- LIBERO-PRO 将任务、位置、语言和环境扰动分开评估：[paper](https://arxiv.org/abs/2510.03827)

## 3. 本轮采集目标

机器清单中冻结 14 个目标：4 个官方题和 10 个由标准 LIBERO 原题导出的 reverse 题。

### P0：先把已有好数据变成可训练候选

1. `official_t3_red_mug`：优先使用 `official_t3_red_mug_1788921471`（314 steps）。
2. `official_t3_alphabet_soup`：优先使用 `official_t3_alphabet_soup_1788946170`（278 steps）。

两条先各 replay 1 次。通过后进入候选集；在训练 manifest 冻结前再各补 2 次，总计 3/3。任何一条 replay 失败只淘汰该 episode，并触发该题补录 1 条，不冻结其他题。

### P1：补齐官方四题的缺口

1. `official_t3_yellow_book`：目标 1 条，理想 2 条。
2. `official_t3_two_mugs`：目标 1 条；先白杯后黄白杯，分别放回红杯左右两侧的原始区域。

每题一个持续 Teleop session，用户可无限 Reset。成功且满意后 Save；单次失败不退出、不记为整题失败。

### P2：最先编译和采集的四个简单 reverse 题

1. `rev_black_bowl_plate_to_table_center`：从盘子拿下黑碗，放到桌面中央。
2. `rev_alphabet_soup_basket_to_floor_front`：从篮子取出 alphabet soup，放到篮子前方的原始地面区域。
3. `rev_close_middle_drawer`：关闭木柜中层抽屉。
4. `rev_turn_off_stove`：关闭炉灶。

每题先做一个 canonical BDDL/reset，目标各 1 条。四题不必全部完成才训练；完成多少就把合格候选送入 batch replay。

### P3：第二批单物体与 articulation 覆盖

1. `rev_black_bowl_stove_to_table`
2. `rev_wine_bottle_rack_to_table`
3. `rev_black_book_caddy_to_table`
4. `rev_black_bowl_top_drawer_to_table`
5. `rev_open_microwave`

这些题各目标 1 条；某题 BDDL/reset 不稳定就标记 `needs_spec_fix` 并跳到下一题，不阻塞整晚。

### P4：最后做多物体

`rev_two_moka_pots_stove_to_table`：把两个 moka pot 从炉灶移回左右原始桌面区域，目标 1 条。它只在至少 4 个单物体目标已产生候选后执行。

## 4. 执行步骤

### RC0 — 已完成

- 确认 fast400 实现存在、用户实际效果良好、当前无录制进程。
- 冻结本计划和 JSON 任务清单。

### RC1 — 现有候选 batch replay

- 在 GPU3 串行 replay red-mug 314 和 alphabet-soup 278，各 1 次。
- 输出 episode ID、replay pass/fail 与短视频/联系图供用户快速 accept/reject。
- pass 的候选先保留；不等待正式审计。

### RC2 — 官方缺口录制

- 用户接管 `official_t3_yellow_book`，无限 Reset，保存 1 条满意数据。
- 然后接管 `official_t3_two_mugs`，无限 Reset，保存 1 条满意数据。
- red/soup 仅在 RC1 对应 replay 失败时补录。

### RC3 — reverse BDDL 编译

- 其他 agent 可并行准备 P2/P3 的 reverse BDDL、自然语言、source/goal predicate 和 canonical reset。
- 编译检查只需：环境可启动、reset source=true、goal=false、目标谓词可在手动放置后成立。
- 不为每题跑项目全量测试；改到共享 recorder 时才跑对应 focused test。

### RC4 — P2/P3 采集

- 顺序：黑碗下盘 → 关中抽屉 → 关炉灶 → soup 出篮；然后按 P3 顺序。
- 每获得 4 条 saved candidate，或当晚结束时，触发一次 batch replay。
- 单题长时间不顺手时由用户主动换题；计划没有强制失败 hard stop。

### RC5 — 快速验收与短训

- 采集阶段先做 replay 1/1；准备进入训练 manifest 的 episode 再补到 3/3。
- 用户只需回复 `接受 <episode_id>` 或 `拒绝 <episode_id>：<原因>`。
- 达到至少 2 条 accepted、覆盖至少 2 个 task ID，即可启动 R6 continuation 100-step 短训；理想首批为 4–6 条、3–4 个 task ID。
- 训练不等待 14 个目标全部完成。GPU1/2 被外部任务占用时继续采集；GPU0 不使用。

## 5. 失败时怎么继续

- 操作不满意或夹取失败：直接 Reset，session 保持。
- Save 被 gate 阻止：显示原因，继续修正或 Reset；不退出。
- 成功但超过 400 steps：保存为 `pending_postprocess`；只允许裁明确空等/尾帧，禁止样条修改 action 后沿用旧 observation。
- replay 失败：排除该 episode，同题最多补录 1 条，然后换题。
- reverse BDDL/reset 不成立：标 `needs_spec_fix`，其他题继续。
- raw partial、NaN/Inf、同目录并发 writer 或不明 start-state hash：只暂停相关 session，先修数据安全问题。

## 6. 多 agent 分工与人工接管格式

- 任务编译 agent：只负责 BDDL、prompt、source/goal predicate 和 reset smoke check。
- 录制 agent：唯一可写当前 episode 目录的进程；同一时刻只开一个 MuJoCo recorder。
- replay agent：GPU3 串行 batch replay，不与 recorder 同时启动 simulator。
- 训练 agent：只消费已冻结 manifest；GPU1/2 空闲后启动短训。

交给采集 agent 时使用：

```text
按 docs/AXIS_PI05_TRACK3_REVERSE_ONLY_COLLECTION_TARGETS_PLAN_20260909.md 执行。
先读取 configs/axis_pi05/track3_reverse_collection_targets_20260909.json。
不要自动独立审核；只汇报当前 target 的 saved/replay 状态并继续队列。
```

需要你接管时，agent 必须给出：

```text
接管 Teleop：<task_id>
目标：<一句自然语言>
成功条件：<明确 source→goal>
建议速度：运输 1.0×–1.5×，抓取/放置 0.5×–0.75×
```

本计划的下一人工目标是：

```text
接管 Teleop：official_t3_yellow_book
```
