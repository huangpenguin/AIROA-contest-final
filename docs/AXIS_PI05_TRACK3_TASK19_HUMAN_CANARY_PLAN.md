# AXIS Pi0.5 Track 3 Task19 单条人工 Teleop Canary 计划

> 制定日期：2026-09-03（Asia/Tokyo）
> 执行输入：root `15791f9`，nested OpenPI `b58aff0`
> 当前状态：`track3_task19_human_canary_ready`
> 目标：人工完成“关闭木柜中间抽屉”一条task19轨迹，写入pending并fresh-process replay 3/3；随后停在人审前。
> 不授权：promotion到accepted、task10/20、planner、P4/P5、训练、FAST、ZIP、C20/E2、官方GPU、上传或提交。

## H0. 启动前门禁

1. 锁定root/nested commit、P0/P1/R4 hashes；确认root/nested clean，`LIBERO-plus`既有dirty内容不动。
2. 确认`track3_reverse_pilot_v2/accepted`与`pending`均不存在或episode count均为0；历史diagnostic/rejected可保留。
3. 只读检查GPU3；显存超过1024MiB或存在外部任务就停止，不终止外部进程。GPU0/1/2不得启动CUDA。
4. 预览端口8765必须空闲；不得把服务绑定到`0.0.0.0`。
5. 若浏览器不在gpu01本机，先在浏览器所在机器开启：

```bash
ssh -N -L 8765:127.0.0.1:8765 huang@gpu01
```

该tunnel保持前台；若已在gpu01本机浏览器访问，则不需要此命令。

## H1. 唯一人工录制

在gpu01的交互式TTY执行一次：

```bash
cd /home/huang/code/PARC2026_final
PYTHONPATH=LIBERO:. CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3 MUJOCO_GL=egl \
  .venv_axis_pi05/bin/python tools/axis_pi05/track3_public_teleop.py \
  --record --task-index 19 --operator operator-local --preview-port 8765
```

在45秒内打开`http://127.0.0.1:8765/`，只在浏览器页面操作，不同时使用终端键盘。开始前必须目视确认：

- task index为19；
- instruction是关闭木柜中间抽屉；
- goal predicate是middle drawer close；
- `success=false`；front和wrist都有画面。

控制：W/S为X，A/D为Y，R/F为Z；U/O、I/K、J/L为三个旋转轴；G切换夹爪；P暂停；C切换相机状态；Z重置；X中止。尽量短按而非长按；方向不确定或接触异常立即P暂停或Z重置。

只有网页显示`success=true`时才按Enter/Save Candidate。保存后程序必须退出并返回`status=pending_replay`与实际`episode_dir`。若timeout、abort、preview断开、predicate fail或异常：保留rejected/diagnostic并hard stop，不自动重试。

H1门禁：pending恰好1条、accepted仍为0；metadata task=19、start goal false、end goal true、7D actions finite、gripper仅±1，trajectory/front/wrist/start/end state及视频文件齐全。

## H2. Task19 fresh-process replay

由执行agent读取H1返回的实际pending目录，禁止“取任意latest”或手工猜路径。GPU3重新通过空闲门禁后执行：

```bash
PYTHONPATH=LIBERO:. CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3 MUJOCO_GL=egl \
  .venv_axis_pi05/bin/python tools/axis_pi05/track3_public_replay.py \
  --verify /实际的/pending/task_19_episode目录
```

要求三个fresh process均为start success false、end success true、return code 0、action finite；`replay_summary.json`必须为`replay_3_of_3_passed_human_review_pending`。任一失败即状态`track3_task19_replay_failed_pending_review`并停止，不重录、不降低3/3门禁。

## H3. 停在人审前

生成task19 canary summary，锁定candidate目录、metadata/trajectory/start/end/front/wrist/replay hashes、deadline miss、frame count、GPU边界和失败/成功状态。把front/wrist视频交给用户目视审核，但本计划不得创建`human_review.json`或调用`--promote`。

若H1/H2通过，状态为`track3_task19_replay_passed_pending_human_video_review`；accepted仍必须为0。等待下一次审核明确记录用户是否批准视频，之后才决定promotion或修复。

## 允许的最终状态

- `track3_task19_human_recording_failed`
- `track3_task19_replay_failed_pending_review`
- `track3_task19_replay_passed_pending_human_video_review`
