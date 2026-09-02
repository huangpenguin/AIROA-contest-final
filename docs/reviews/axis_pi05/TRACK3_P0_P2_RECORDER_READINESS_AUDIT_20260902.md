# Track 3 P0–P2 终态与 Teleop Recorder 就绪性审核

日期：2026-09-02（Asia/Tokyo）
审核输入：root `9dac308`，nested OpenPI `b58aff0`
审核结论：`remediation_required`

## 可保留成果

- P0 官方公开HDF5路径成立：40个文件、总计`33784856577` bytes、`.partial=0`；40份task mapping、schema inventory及required dataset gate通过。
- P1 task10 support、task19 articulation、task20 containment的完整simulator-state restore/20帧settle/reverse-not-success均为fresh-process `3/3`。
- activation task16两种公开来源均在settle后失稳，正确排除，没有伪装成成功family。
- 没有训练、FAST、ZIP、官方GPU、上传或提交；GPU0未触碰，独立`LIBERO-plus`既有dirty状态未修改。

这些结果证明公开完整终态路径已经跨过旧T1的`0 snapshot`阻塞，但P1 snapshot仍是`pending_video_review`，不是accepted demonstration。

## 必须先修复的问题

### 1. Recorder空闲帧会违反action contract

`record_episode()`在没有键盘输入时执行`action = np.zeros(7)`，gripper因而为`0`；紧接着`validate_action()`要求gripper只能为`{-1,+1}`。真实录制很可能在首个无按键tick直接异常。现有CPU self-test调用的是`key_to_action("")`，没有覆盖真实loop的no-key分支。

### 2. 当前headless环境无法目视操作

审核环境的`DISPLAY`、`WAYLAND_DISPLAY`均为空，session为TTY。recorder只采集offscreen image，没有`imshow`、web preview或其他实时画面；`camera`字段只被toggle，从未参与显示。操作者按当前命令会盲操作。

### 3. 报告宣称的控制与实现不一致

- `KeyboardState.reset`存在，但没有reset key分支，也没有重新加载P1 start state；
- camera switch没有可见效果；
- 报告命令含`--operator <operator-name>`，若原样粘贴会被shell解释为重定向；
- recorder没有GPU3占用/外部PID fail-closed门禁。

### 4. Accepted语义不合格

只要当前进程末尾predicate成功，轨迹就直接写入`accepted/`，但metadata仍是`replay=pending_fresh_process_3_of_3`、`human_review=pending`。这会让未replay/未审核的candidate混入后续manifest。当前也没有fresh-process replay verifier或promotion步骤。

### 5. 证据与测试状态不闭合

- `p0_lock.json`实际SHA-256为`be12795b3d18d46f774022d817789a14d64a5e16c8e27bee58e256c29acb4ab5`；执行报告和machine handoff记录的`06c2d8...`不匹配。
- P1九个attempt均写`video_codec=not_yet_recorded`；PNG存在，但原计划要求的settle H.264/人工目视门禁未完成。
- 在clean commit `9dac308`上复跑两个相关测试文件得到`1 failed, 6 passed`；失败是machine handoff已前进到P2，而旧回归测试仍断言P0–P5旧状态。因此提交所称全量`155 passed`不能作为该commit的可复现结果。

## 审核决定

不要运行执行报告中的teleop命令。旧计划归档；新计划只修recorder语义、headless实时预览、candidate→3/3 replay→human-review promotion与证据闭合。修复完成并通过GPU3无输入/重置/预览canary前，accepted episode必须保持`0`，不得进入planner、数据扩充或训练。
