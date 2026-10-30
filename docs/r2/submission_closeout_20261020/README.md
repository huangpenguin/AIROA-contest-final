# 最终提交清单与课程报告

2026-10-20，停止模型改进。使用现存R6 FAST step500权重重新封装以下两个ZIP；T1和T3共用同一文件。没有自动上传或提交。

| 提交入口 | 选择的文件 | 固定配置 | 历史记录（不属于新包） |
| --- | --- | --- | --- |
| Track 1 | [PARC2026_T13_R6_FAST_v2_execute8_20261020.zip](/home/huang/code/PARC2026_r2_submit_inventory_20261020/outputs/r2/final_submission_20261020/PARC2026_T13_R6_FAST_v2_execute8_20261020.zip) | chunk10 / execute8，EMA0、RTC off | 原包用户报告0.47 |
| Track 2 | [PARC2026_T2_R6_FAST_v2_execute4_20261020.zip](/home/huang/code/PARC2026_r2_submit_inventory_20261020/outputs/r2/final_submission_20261020/PARC2026_T2_R6_FAST_v2_execute4_20261020.zip) | chunk10 / execute4，EMA0、RTC off，无seam | 9月16日原包用户报告0.3457 |
| Track 3 | 与Track 1相同的T13文件，在Track 3入口提交 | 同execute8；没有已审更优替代模型 | 旧R6用户报告0；没有新提分证据 |

两文件分别9,438,690,086与9,438,690,071 bytes，各约8.79GiB。原最高分ZIP已经删除；这两个文件具有新身份，不能继承历史分数。9月29日non-FAST恢复包不是这次选定的包；不采用9月17日seam包或10月20日被拒绝的seam01候选。

## 文件身份与验证边界

- T13 ZIP SHA-256：`38d9317bb0496767c00dbbb7dbff71cab973c96688ee6b1a30fe43592807571e`
- T2 ZIP SHA-256：`30c4d880f1b6aaf3746b9bfd979db3706a9e5b8a25835ae5fbf7bc3b587561a1`
- 共同FAST权重SHA-256：`e55bf7dcca78b2195993dd7a12b96a4ea13e94761d711c99a82b7411c3f8ceb6`；在ZIP写入流中复核，没有再次全文件扫描。
- 每包3197个普通文件，均无链接、嵌套压缩包、重复entry或越界路径，根目录有policy_server.py、requirements.txt、model_weights、vendor。默认用一次固定合成输入预热并清空状态，然后开放health。
- 聚焦CPU检查覆盖实际处理器与tokenizer、spy模型队列/回合重置/HTTP、离线约束与预热调用顺序。主审另查最终ZIP目录与小文件；10月20日前序真实v2加载与推理证据复用。
- 官方环境依赖全新安装、安装总时间、新合成预热的真实耗时、L4显存与新包官方分数均未验收。CPU结果不替代这些项目；此次收尾没有再次启动GPU实验。

## 日文课程报告

[5页PDF](../../course_final_20261020/REPORT_JA.pdf) · [可编辑日文原稿](../../course_final_20261020/REPORT_JA.md) · [要求核对表](../../course_final_20261020/REQUIREMENTS.md) · [匿名证据索引](../../course_final_20261020/EVIDENCE_PUBLIC.md)。

正文涵盖背景、R6训练与数据、执行长度、最终15对平滑化实验、否定结果与局限。数值、日文字体嵌入、文本提取、PDF元数据和匿名检查已核对，主审查看首页与结果页。报告PDF SHA：`4b214d37e6fd4d1795b371cf95931cdac319a9a6d5f9be4dbee101a77480d841`。

这是Physical AI应用篇1课程最终课题报告，不是PARC本选2比赛报告（本选2无需报告）。课程截止2026-11-02 10:00 JST，[课程官方公告](https://physicalai2025.slack.com/archives/C0ARM7WEUGL/p1792396824899139)。已遵守公告中的匿名要求；Quri详细课题/模板需要登录，指定页数、必填栏目及视频/代码要求尚未取得。因此正文与PDF已完成，模板最终符合性待核，不把PDF单独标为整个课程作业已完成。用户补充该段要求后可只改文档，不恢复模型实验。

内部原始路径/校验日志放在outputs中，不随匿名报告提交。竞争模型ZIP只用于对应竞赛入口，不能当课程匿名代码附件或公开权重包。

## 内部审阅入口

[计划](PLAN.md) · [主审结论](REVIEW.md)。执行与原始证据：[工作树交付目录](/home/huang/code/PARC2026_r2_submit_inventory_20261020/outputs/r2/final_submission_20261020/)。共享交接使用唯一身份`R2_SUBMISSION_CLOSEOUT_20261020`，旧轮次及unprefixed别名保持原状。
