# Track 3 Teleop R0–R4 人工就绪审核

日期：2026-09-03（Asia/Tokyo）
审核输入：root `15791f9`，nested OpenPI `b58aff0`
审核结论：`approved`，批准范围仅为task19单条人工pending canary与其3/3 replay

## 通过项

- R0锁定的P0/P1 hash正确；历史stale hash没有覆盖raw evidence。
- R1统一action state machine覆盖50个no-key ticks、pause、gripper、camera、reset、save与abort；所有实际action路径保持7D finite且gripper为`{-1,+1}`。
- R2 preview只绑定`127.0.0.1`，browser未连接并拉取首帧前不会进入record loop；front/wrist、task、instruction、goal、step、lag、gripper和success均可见。
- R3录制成功只进入`pending/`；代码与测试均要求fresh-process replay `3/3`以及显式human review，才允许atomic promotion到`accepted/`。
- R4在物理GPU3按task19→10→20通过no-key/preview/reset/abort canary；reverse goal reset false，GPU前后均为5MiB，accepted/pending均未创建。
- 独立复跑全量测试：`158 passed, 7 skipped, 23 warnings`。
- root/nested clean；GPU0未触碰，GPU1/2未训练，独立`LIBERO-plus`既有dirty状态未修改。

task19 front contact sheet已目视检查：中间抽屉处于打开状态，robot与cabinet可见，20帧settle无明显漂移；wrist视角信息有限但front足以作为开始操作的主画面。实际review asset使用`mp4v`并同时保存exact `raw_rgb_npz`，codec如实记录，不冒充H.264。

## 保留风险与授权限制

R4直接调用共同state machine/preview/reset组件，但没有执行完整的人工`record_episode → pending write → fresh-process replay`链；R3 promotion只做了实现和fixture测试。因此task19仍是第一个真实端到端canary。

本审核不批准：

- task10或task20人工录制；
- task19 replay通过后的自动promotion；
- P3 planner、30–60条扩充、P5 loader、LoRA/dense、FAST、ZIP、C20/E2、官方GPU、上传或提交。

只有task19产生一个合法pending candidate且fresh-process replay `3/3`，才能停在视频人工审核；human review未明确批准前，accepted必须继续为0。
