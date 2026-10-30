# 主审结论：接受本地收尾交付，保留未验项目

`R2_SUBMISSION_CLOSEOUT_20261020` round0；主审/root，唯一执行交付者/root/submission_inventory。审核output SHA-256 `afb1a489f965d2e39339aa6e6825b3eb706597d9595d0191170f70cbb0d52124`。两个执行worker均使用用户指定的gpt-6.1-sol、worker角色、独立上下文与codex分支工作树。

## 接受范围

- T1/T3的execute8 ZIP及T2的execute4 ZIP实际存在，各约9.44GB。主审独立读取两包目录与小文件：每包3197个普通文件、无链接/嵌套压缩/重复名/越界路径；policy、config、tokenizer和normalizer路径闭合。流式写入SHA及共同权重身份与历史FAST SHA吻合，源stat未变化。
- 源码diff已审，仅固定T2执行长4、删除本地metadata依赖、固定recipe校验和默认一次合成预热。没有加入seam/RTC/指令alias；实际HTTP序列化、health/act等五函数与模板AST一致。CPU测试在最终ZIP小文件的独立展开目录执行，各5项通过，使用真实处理器/tokenizer与明确神经模型spy；没有加载大模型或使用GPU。
- 报告正文、5页PDF、匿名证据索引和要求映射已完成。主审核对训练/最终15对结果与历史成绩身份，查看PDF首页及结果页，长表格单元格换行已修正。日本语字体嵌入、可提取文本、作者元数据为空，未发现真实姓名/邮箱/私人路径或账号链接。
- 首次打包因缓存文件被检查器拦截，发生在ZIP创建前；改为排除缓存后完成，原失败证据保留。最初numpy版本草案与实际已测版本不一致，最终固定2.2.6，保留首份检查记录。

## 结论的边界

这是原FAST权重的新封装本地候选，不是已删除的计分原ZIP。历史T1 0.47、T2 0.3457均为保存的用户报告，不能继承到新包。T3没有已审更优替代，使用同execute8作最终fallback，历史0，没有涨分证据。

未验项目：官方环境全新依赖安装/ABI/时间、官方L4显存与真实默认合成预热延迟、新官方任务分数、全ZIP二次CRC扫描、课程Quri详细模板。CPU通过不等于这些项目已通过。默认预热是在服务开放前执行，但新合成输入的GPU时限未测，不借用前序实输入预热的时限证据。禁止将该结论表述为官方环境完整验收或课程所有要求已满足。

接受本包结束和文件交付；不批准新的模型实验，不自动上传、公开模型、推送或比赛提交。课程模板待用户提供登录后具体格式/栏目，再做文档调整即可。

## 资源与状态

打包交付阶段墙钟781.3秒；专有工作树保守计入旧checkout后的磁盘分配19,289,980,928B，低于24GiB。两个ZIP合计18,877,380,157B。CPU秒数与峰值RSS未计量；GPU秒、真实模型初始化、优化器更新、模拟步、安装、下载、上传、提交均为0。

构建和测试已结束，不留后台执行。受保护CPU PID2234022/2236818/2237769保留，GPU compute列表为空。root HEAD仍0a71e1f、嵌套OpenPI HEAD仍b58aff0，原修改未撤销。新prefix持既有.handoff.lock发布，旧aliases与旧轮次快照保持不变；done只代表此收尾交付关闭。

审阅证据：`outputs/r2/submission_closeout_20261020/ROOT_ARTIFACT_INSPECTION.json`、`REPORT_VALIDATION.json`以及共享前缀output所列全部收据。完整包清单见[README](README.md)。
