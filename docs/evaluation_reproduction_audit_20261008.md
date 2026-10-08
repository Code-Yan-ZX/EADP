# EADP 多模型评测和论文复现差距审计

审计对象为 `EADP_amp` 的最新多模型结果，起始提交 `194562d`，日期为 2026 年 10 月 8 日。
当前差距同时包含评测错误、复现协议差异和方法在小预算下的低分，不能归为一个原因。
本轮保留历史预测，用独立产物重算和验证；方法的 alpha、beta、lambda 均未调整。

最显著的两项复现差距已有直接解释：MME 的本地答案文件有 162 个错误标注，
修正后 FULL 距论文只剩 −6.47 分；ScienceQA 原包题面全量对照使 FULL
从 64.85% 升至 69.51%，距论文只剩 −0.09 点。两者分别是评分答案错误和输入格式差异。

## 已确认的错误和修复

| 问题 | 影响 | 本轮处理 |
|---|---|---|
| MME 本地重建 GT 与官方模板有 162 个 Yes/No 相反 | FULL 和复现 EADP 被低估约 86–90 分 | 按类别、图名、规范化题面匹配官方 GT，保留原预测与错误 GT 留档，全部 8 臂独立重评 |
| 通用驱动继续调用旧 VizWiz、POPE、GQA 评分路径 | 今天的离线修复无法阻止后续恢复任务再次生成错误分数 | 新增统一在线评分入口，驱动写独立 `.official.score.json`，保存源文件哈希 |
| VizWiz 旧规则省略逐标注留一 | 绝对分数约高 1.4 点 | 在线入口与正确快照逐题一致 |
| POPE random 顺序对齐错误及 popular 单类冒充主列 | random 错位、汇总指标不一致 | 按图像和题面匹配，主列为三类平均 F1；保留用户已选的 2910 道 random 题集 |
| GQA 在构建字典时覆盖重复 ID，静默跳过未知题 | 缺题或重复题也可能得到分数 | 在构字典前检查行数、唯一性、题集完整性；失败不输出成绩 |
| SQA 旧 QCM 输入被恢复驱动继续调用 | 新旧提示协议混用 | 停用旧入口；本地重建 CQMI 专用入口校验完整题集和题面身份 |
| 最新 SQA K128 差值记错分母 | Overall 被误写为 −0.15 | 正确为 Overall −0.070738、IMG −0.148736，均少 3 道正确题 |
| NeXT TextVQA FULL 用错论文参照 | 60.24 被记成比论文低 4.8，并用于解释 SQA | 对应论文值为 60.3，差 −0.06；撤回同源归因 |
| Qwen EADP 复现被概括为全部在 ±1.3 内 | 隐藏 DocVQA 和 Hallusion 的剩余偏差 | DocVQA −1.66、Hallusion −1.76，逐项列差 |
| NeXT 预算清单漏算全局图并把 tqdm 刷新计数当题号 | 5 个输入 crop 被记成 4，部分图题的 token 数被错记为 0 | 重导出独立 v2，验证 1 全局＋4 局部；未对齐的日志字段隔离，实际保留 token 数标未验证 |

22 份完整预测的验证覆盖 VizWiz 6 份、GQA 8 份、POPE 8 份。
在线新评分与今天正确快照的逐题成绩或各类混淆矩阵一致。
另外验证了重复、缺失、未知 ID、FAILED 答案拒绝，以及合法全答 no 的 POPE F1=0。
SQA 9 份本地重建 CQMI 预测通过完整性和题面复用门。
详细验证见 `outputs/stage1_roundtrip_pilot/legacy_full/audit_20261008/live_scorer_validation_20261008.json`。

## 论文参照必须与本地复现分开

[EADP 论文表 1 和表 2](https://arxiv.org/html/2607.02484v1) 的 LLaVA Avg 使用九项任务，
排除 VizWiz。VizWiz 的带星号数值为 val，与当前 val 可对照；未带星号的 test 数值不能混用。
本地缺失 VQAv2 或 MMVet 时，子集均分不能充当论文 Avg。

Qwen 的统一十任务表中，本地完整方法为 68.0244，本地 EADP 为 67.6636，
同本地协议增益为 +0.3608。论文 EADP 为 68.2，因此直接跨论文的差为 −0.1756。
这两种差回答不同问题：前者衡量同一复现底座的增量，后者还包含本地底座的 −0.5364 差距。
Hallusion 的论文列采用 fAcc 仍是基于代码与数值的推断，不能改写成作者已明确声明。

Qwen native 与 official legacy 是两条执行路径。此前找到的官方全量预测已经复现
TextVQA 71.042、DocVQA 61.1356、OCRBench 623；native 更高的结果包含 DeepStack 和
3D mRoPE 的恢复效应。最新论文对照使用 legacy，同 native 的增益不能算作新方法贡献。

LLaVA 的 MME 论文列为 perception，项目同时保留 cognition 分列。
使用 perception+cognition 总和比较论文单列，或将该总和放进同名平均，会造成口径不一致。
这一点与 Qwen 表 4 的 MME 总分口径不同，应按模型分别记录。

## MME 标注错误：已用官方答案重评

本地 `MME_Benchmark_release_version` 的重建答案，以及从它转换的所有 8 份评测输入，
均有 162 个标签翻转（97 个 no→yes、65 个 yes→no）。错误集中于 artwork 71、posters 42、
scene 34；其余为 existence 5、count 1、commonsense 2、numerical 2、code 5。
另外 7 条题面的差异仅为空白，不影响规范化后配对。

以 [MME 官方 Evaluation 分支](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)
的 `tools/eval_tool.zip/Your_Results` 为答案来源。它与包内 LaVIN 模板、当前
`/media/disk2/YZX/LMUData/MME.tsv` 的全部 2374 道题和标签完全一致；
本地和本轮下载的官方 zip SHA256 同为 `b8125e2a…`。
因此错误发生在本机 LLaVA 答案重建/落盘后的数据，不是当前 TSV 的标注错误。
无法仅由现存工件确定历史上造成翻转的具体写入操作。

| 模型 / 臂 | 错误 GT 的 perception | 官方 GT 的 perception | 论文对应值 | 修正后距论文 |
|---|---:|---:|---:|---:|
| v1.5 FULL | 1420.43 | 1506.93 | 1513.4 | −6.47 |
| v1.5 EADP K128 | 1339.10 | 1429.40 | 1439.0 | −9.60 |
| v1.5 AnchorZip K128 | 1350.35 | 1445.69 | 1439.0 (EADP) | +6.69 |
| v1.5 AnchorZip K64 | 1312.89 | 1400.37 | 1403.6 (EADP) | −3.23 |
| v1.5 AnchorZip K32 | 1042.97 | 1092.63 | 1347.0 (EADP) | −254.37 |
| NeXT AnchorZip K640 | — | 1467.14 | 1494.7 (EADP) | −27.56 |
| NeXT AnchorZip K320 | — | 1439.71 | 1491.3 (EADP) | −51.59 |
| NeXT AnchorZip K160 | — | 1369.50 | 1419.5 (EADP) | −50.00 |

表中 AnchorZip 与论文 EADP 的差仍包含实现/输入残差，不能直接当作同底座方法增量。
本地同协议 v1.5 K128 AnchorZip−EADP 的 perception 为 **+16.2941**，
cognition 为 −2.5；不能混用二者的总和与论文 perception。
原包自带 LLaVA-1.5-13B 预测用该 GT 得到 1531.3187，精确对齐其论文公布的 1531.3，
进一步支持答案来源及评分流程。

逐题翻转、逐类分数和输入哈希见
`outputs/audit_followup_20261008/mme_official_gt_rescore.json`；
可直接交给官方 calc_scores.py 的完整转换结果另存于
`outputs/anchor_completion_validation/records_20261003/audit_followup_20261008/mme_canonical_gt/`。
历史 MME 分数和使用该 GT 的配对置信区间须由新 GT 重算，不能继续视为有效成绩。
新 canonical scorer/converter 已接入两条恢复驱动，校验官方 archive SHA256、
完整题集、题面和每图两问，不再读取错误重建 GT 或复用旧 `.score.txt`。
官方 calculator 对 8 本地臂和 13B 参考共 9 份转换结果与独立重算一致（<1e−9），
6 个损坏输入/旧目录覆盖负例通过。

Qwen 的评分依赖单独复核：EADP256、AnchorZip256/128/64 各 2374 条 GT、
auxmatch 和逐题分数全部与官方模板一致，变化为 0。
Qwen P1 的 MME 差 −9.770908、已修复重复簇的 CI 和 Avg10 不受 LLaVA 错标影响。
本轮未找到实际 Avg7 计算工件，不能据不存在的公式宣称已重导；
依赖盘点见 `outputs/audit_followup_20261008/mme_dependency_impact.json`。

剩余 MME 缺口还有配置身份边界。官方 MME/TextVQA shell 的 β 默认值为1，
本机冻结β2；官方主架构实际 entropy keep 比例硬编码0.2，环境变量只由 ablation
架构读取。[论文表7](https://arxiv.org/html/2607.02484v1#S5.SS6) 的 β2/β10 面板
MME 分别为1412.4/1439.0，但没有公开主表每任务的原始运行参数。
不能仅由 ablation 单元格唯一还原主表配置，也不能认定改β10就能复现1439。
本轮保留冻结参数、记录未验证的论文配置身份，不调参追分；证据与源码哈希为
`outputs/audit_followup_20261008/llava_paper_configuration_ambiguities.json`。

## SQA 输入身份和验证

当前被称为“官方 CQMI”的输入是本地从转换器重建的文件。
[EADP 的 sqa.sh](https://github.com/SJTU-DeepVisionLab/EADP/blob/e1a08801461c181871b9ff1cb0803b8e9966f083/LLaVA/scripts/v1_5/eval/sqa.sh)
只给出 CQM-I 文件名，未提供其实际文件，因此重建输入尚未通过作者输入身份检查。

[LLaVA 官方 sqa.sh](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/eval/sqa.sh)
使用 eval.zip 提供的 CQM-A。本地 CQM-A 与原始压缩包内文件字节一致，
与重建 CQMI 的题集和图像路径相同，但 4241/4241 个用户题面不同：
原始文件用分行 `A. ...` 选项，本地重建用一行 `Options: (A) ...`，还改变了
Question 标签、空 Context 的处理和图片标记的存放位置。

本地重建 CQMI 的 v1.5 FULL IMG=64.85，论文表 1 为 69.6，差 −4.75。
该 FULL 仅 17/2017 个图题答案抽取失败，最多解释 0.84 点，不能解释全部差距。
NeXT FULL IMG=66.93，论文表 2 为 67.6，差仅 −0.67，不能沿用 v1.5 的差距归因。

本轮固定同一 checkpoint、图像、vicuna_v1、greedy、FULL=576 和官方评分器，
将输入换为原始 CQM-A，对全部 2017 个图题运行配对验证。
正确题数从 1308 升至 1402，IMG **64.8488→69.5092（+4.6604 点）**；
论文为 69.6，残差降至 −0.0908。196 道原错误题被救回、102 道原正确题变错，
解析失败从 17 降为 0。该实验覆盖全部图题，没有按答案筛选。
结果与完整性证据存入 `outputs/audit_followup_20261008/sqa_input_comparison.json`；
其归因范围是题面格式整体替换，不拆解为某一个标签或选项排版的贡献。
它也未证明作者未发布的 CQM-I 与原包 CQM-A 相同；原 CQMI 各剪枝臂成绩保留独立标注。
原包题面下 EADP K128 与 AnchorZip K128 的完整配对也已完成：

| 臂（原包 CQM-A，图题2017） | 正确数 | IMG | 相对同机 EADP |
|---|---:|---:|---:|
| FULL | 1402 | 69.5092 | — |
| EADP K128 | 1403 | 69.5588 | 0 |
| AnchorZip K128 | 1396 | 69.2117 | −0.3471 |

三臂解析失败均0。AnchorZip 对 EADP 救回15题、变错22题，净少7题；
配对 bootstrap 95% CI 为 [−0.9420,+0.2479] 点。
论文 EADP K128 为69.0，原包控制中的 AnchorZip 高0.2117点，
但同机配对略低且 CI 跨0，不能据跨论文数值宣布本任务方法增益。
新评分、逐题身份、冻结参数和 source hash 见
`outputs/audit_followup_20261008/sqa_shipped_control_analysis.json` 及
`sqa_shipped_control/v15/`；旧 CQMI 两臂的−3题差值保持独立记录。

## TextVQA 剪枝指导文本差异

论文官方 [textvqa.sh](https://github.com/SJTU-DeepVisionLab/EADP/blob/e1a08801461c181871b9ff1cb0803b8e9966f083/LLaVA/scripts/v1_5/eval/textvqa.sh)
走 model_vqa_loader。当前历史预测走 model_vqa：两者 LLM 题面相同，
但给 EADP 的打分文本不同。官方 loader 先删除“用单词或短语回答”的后缀；
历史 model_vqa 保留该后缀。5000/5000 道题的打分文本不同，CLIP 增加约 10 个 token，
其中 298 道题增加一个文本分段，会改变 dense relevance 和 entropy 的输入。

历史生成上限为 1024，官方 loader 为 128。EADP 输出复分词超过或达到 128 的题数很少：
v1.5 为 2/5000，NeXT 为 17/5000，提示该因素涉及的输出比例很小。
这是解码文本复分词估计，不是原始生成长度的严格上界；仍需用实际生成验证。
打分文本的影响须用固定面板，只切该文本输入验证，不能据此预设全量分数会升高。

本轮面板按 CLIP 文本长度分层固定 16 道题，seed=20261008，不按答案对错筛选。
保持 LLM 提示、图像和生成参数不变，记录两种打分文本的 keep-set、配额和预测，
并核对历史 EADP 输出身份；产物为 `outputs/audit_followup_20261008/textvqa_guidance_*`。
面板已完成：两模型 current 都 16/16 对齐历史 EADP 回答；
去掉 guidance 后缀后，各 16/16 题的保留集合改变，平均 Jaccard 分别约 0.650/0.645。
v1.5 仅 1/16 个答案改变（该题正确→错误），NeXT 0/16 改变。
该差异确实影响剪枝，但这组分层面板不能估计全量准确率效应，
也不能支持“修后必然更接近论文”。严格官方入口 `run_textvqa_official.py`
已验证全 5000 道 guidance 转换与官方 loader 一致，独立输出名和拒绝旧预测复用通过；
此前审计阶段未启动 TextVQA 全量官方入口重生成；用户于10月8日要求立即并发重跑后，已启动 NeXT 官方 K128 的 EADP/AnchorZip 配对，后续两模型各预算配对已接入持久队列，进度见文末。

## 剩余低分和运行稳定性

正确评分后的 v1.5 K32 POPE 平均 F1 为 80.008，论文 EADP K32 为 86.7。
本地 popular recall 从 K128 的 83.667 降至 K32 的 70.0，precision 仍为 94.851，
说明小预算出现大量存在对象被回答为不存在的漏检。评分修复没有消除这个低分。
但本机同预算 EADP K32 的全量对照尚未完成，当前只能确定低分及其错误形态，
不能把整个 6.69 点跨论文差距都归给新方法。

K128→K32 新增错误 701 道，其中 642 道（91.6%）为 FN；
MME 同预算变化则是双向损坏：新增错误 349 道，FN212/FP137，
Yes 比例仅 55.60%→54.42%，没有全 no 崩塌。
日志显示全程预算32、每任务一次模型加载，历史驱动冻结 α0.5/β2，
未变 port 常量 λ0.25；`LRMAIN00625` 是输出名，不是解析得到的 λ。
历史运行没有保存这些超参的直接 runtime trace，应保留证据边界。

重复性仍需核实：POPE 有 2334 组相同图像、完整题面重复题，
EADP128、AnchorZip128/64 组内答复均一致，但 AnchorZip32 有182组分歧。
这不能仅由低准确率解释，应核查输入张量、token选择、合并和解码的重复性；
目前不能认定是具体随机算子、状态污染或历史运行条件。
只读错误诊断见 records 的 `audit_followup_20261008/v15_k32_failure_diagnosis.json`，
完整正确→错误清单另存，固定重复题面板用于运行问题复现，不用于估计准确率。

固定16分歧组×3次 GPU 重放已完成，48次全部回答 Yes；实际图像、input_ids、
vision/text embeddings、RTG权重、importance、keep/quota、Completion和生成IDs
等16阶段hash均重复一致。模块均eval，实测预算32、α0.5、β2、λ0.25。
因此短重放未复现历史分歧，不能把已存 K32 headline 直接视为已验证可重复的方法成绩；
也不能用该面板重算总体 F1。记录为 `outputs/audit_followup_20261008/pope_k32_repeat_smoke_v15.json`。

源码候选风险：loader 的 non_blocking CUDA 传输后，CLIP 图像分支新 stream
未显式等待输入所属 stream。[PyTorch CUDA 文档](https://docs.pytorch.org/docs/stable/notes/cuda#cuda-streams)
要求跨 stream 消费建立同步依赖；末尾 synchronize 不等价于事先建立读写顺序。
这是读码推断，未通过实验确认它导致了历史分歧。历史 lane 设计允许并发任务，
实际并发与运行时输入/模块哈希证据缺失；本轮串行短重放不能替代历史长序列条件。
按预设条件没有运行 streamwait 分支，也没有据候选风险修改生产代码。

先完成输入协议和官方基线校验，再在同一协议下比较完整方法。
SQA 需要分别标注重建 CQMI 和原始 CQM-A；TextVQA 的严格官方复现需要新输出名，
以免恢复脚本复用历史打分文本。MMBench 的偏差需对照模型、指标和提示格式，
不得通过换指标或选子集使数字接近论文。固定协议后仍落后的任务，再作为方法失败分析。

## 工件与复核入口

`outputs/audit_followup_20261008/latest_method_comparison.csv` 和同名 JSON
分别列当前方法、同机 EADP（缺失则留空）、对应论文行及两种差值，
每项保存分数来源与 SHA256。旧 CQMI、原包 CQMA 控制和 Qwen native/legacy
各自标注，不补不存在的同预算基线、不构造缺任务的论文 Avg。
v1.5 K32 POPE 单元格另标注历史重复不一致和全量可重复性未验证。
`latest_method_comparison_validation.json` 验证源哈希、行数和缺项处理。

以下从仓库根目录运行，仅生成 CPU 计划或重评分，不启动模型生成：

```bash
/home/dell/miniconda3/envs/llava_pruner/bin/python \
  Qwen_vl/scripts/stage1_roundtrip_pilot/mme_canonical_score.py \
  --result-file LLaVA/playground/data/eval/anchorzip_p3/mme/FULL.jsonl \
  --question-file LLaVA/playground/data/eval/MME/llava_mme.jsonl \
  --gt-archive LLaVA/playground/data/eval/MME/eval_tool.zip \
  --out Qwen_vl/outputs/audit_followup_20261008/mme_FULL_recheck.canonical_gt.score.json
/home/dell/miniconda3/envs/llava_pruner/bin/python \
  Qwen_vl/scripts/stage1_roundtrip_pilot/run_sqa_shipped_control.py \
  --model v15 --arms E_GATHER_K64,AZ_K64
/home/dell/miniconda3/envs/llava_pruner/bin/python \
  Qwen_vl/scripts/stage1_roundtrip_pilot/run_textvqa_official.py --model both
```

两个新生成入口默认只打印计划；显式 `--generate` 才会顺序生成，已有预测/协议文件
拒绝覆盖。Python13入口及shell8入口语法检查、`git diff --check` 已通过；
MME9份官方calculator复核、22份live scoring对照和完整性负例见上述独立验证工件。

## 10月8日下午启动的完整重跑

用户授权立即并发，截止为北京时间10月9日12:00。两个GPU通路已运行：
v1.5 POPE AZ32全8910，以及NeXT官方TextVQA K128配对。后续已持久排队，
包含TextVQA两模型各7臂（FULL、EADP/AnchorZip各128/64/32）和SQA原包题面
v1.5剩4臂、NeXT全部7臂；总共26个新GPU实验、101097条新预测。

`rerun_batch/rerun_schedule_manifest.json` 记录任务量；
`lane1_v15_plan.json`、`lane2_next_plan.json` 对应实际队列，各自的state文件
记录启动/评分/完成/失败。两GPU子进程峰显存约42325/46068MiB，禁止盲开第三模型。
每臂保留原冻结参数与生成协议，不为赶时间降低精度或抽样；与旧预测分目录。
队列不依赖对话在线，逐项完整性检查后运行官方评分。

调度审查另修正了TextVQA完整性键：question_id是image_id，5000题只有3166唯一
ID；完整 `(question_id,全文prompt)` 才是5000唯一题。真实全量正例通过，缺题、
重复复合键、未知键及runtime错位负例均拒绝。此为新调度入口修复，不能倒推旧
官方评分存在按ID去重。SQA保持2017唯一题ID。初始已启动NeXT K128两臂没有
后来新增的逐题token runtime trace，需明确保留这一证据边界。

暂按双路实际吞吐保守估计完整批次于10月9日凌晨3–6点完成；该时间不是结果承诺。
准确率只汇总完整评分工件，未完成臂不计算前缀分数。原始历史结果、修复输入协议
结果及论文对照各自标注，等待新配对后更新方法优劣结论。

本次as-is POPE重放在15:37生成4596题后，845个已经重复的同图同题面组中有
77组Yes/No分歧；实际预算全部32，输入题面逐字一致。这证明运行不一致并未因
换一次进程自动消失，仍不证明缺少CUDA依赖就是全部原因。原全8910继续保留并
评分，额外完整8910控制仅增加CLIP跨流等待，原生产代码保持冻结。该额外控制
不包含在上面的26主实验数量中。v1.5队列已在其前设置等待门，控制完整评分后
无论分歧是否消失都续跑主实验，禁止以取得期望分数作为放行条件。

额外控制最终决定为AZ32 wait与同机EADP32 wait两完整8910臂；这是事先决定的
公平基线，不根据准确率挑选臂。两者输入、参数与wait协议一致，唯一差别为
AnchorZip安装；两项完整评分后才释放v15队列。总量更新为28臂/118917新预测，
主26实验的统计仍保留单列。K128配对估至傍晚—19点左右；完整批次目标Oct9
凌晨3–6点，保守预留8点，临时预测随实际工件更新。

## 2026-10-08 16:10 北京时间独立巡检

本次完整读取交接，确认分支 `codex/anchor-completion-validation`、HEAD `194562d`，
并执行 CPU 只读 `monitor_repair_hourly.py --once` 两次。巡检证据保存在
`outputs/audit_followup_20261008/rerun_batch/hourly_monitor/20261008T080726Z_independent/`。
两次快照间隔153.63秒：NeXT EADP128 TextVQA 从3137增至3310/5000；
v1.5 AZ32 streamwait POPE 从2940增至3578/8910。两者输出和生成日志的最后
更新时间均不足1秒，GPU worker PID分别为1504969、1559829，启动身份未变。
两lane监督器、控制监督器和汇总器均存活；未发现僵尸、Traceback、OOM或FAILED。
单张A40恰好两个GPU模型进程，抽查总显存42687/46068MiB、利用率100%。
lane1合法等待两POPE控制完整评分后释放marker；lane2等待仍在生成的初始NeXT
K128配对父进程。控制当前实际阶段为AZ，不将控制日志/state未刷新误判为卡死。

as-is POPE已有完整8910题合法评分（random2910、popular/adversarial各3000），
三类平均F1为78.541，较历史80.008低1.467点；2334个重复输入组中226组分歧
（历史182组），实际保留数8910/8910均为32。完整重跑仍有不一致，不能声称
已修复或由stream风险解释。AZ与EADP streamwait两臂尚未完整评分，marker未释放。
汇总器当时仅包含此前v1.5原包SQA三臂完整2017题结果，`errors=[]`；
新NeXT配对与后续队列尚未完成。两lane及控制计划冻结的来源/输入SHA均匹配。
本次仅保存巡检证据和事实记录，未重启进程或修改生成源码/参数/预测。

## 2026-10-08 16:45 北京时间：新增完整结果与等待故障

完整AZ32 streamwait控制于16:32结束：8910题、输入全文逐字一致、实际保留数
全部32、退出码0，三类均F1 **83.084**。较同轮as-is78.541为**+4.543点**，
image-cluster配对95%CI[3.984684,5.129300]；较历史80.008为+3.076点。
2334组重复输入的Yes/No分歧从本次as-is226组降至**0组**。这是完整同步干预
改善运行结果的证据，每配置一次全量运行仍不足以证明独立重跑可重复性，生产
CLIP文件没有改变。同预算EADP32控制尚无完整结果，禁止把83.084相对论文86.7
的−3.616点当作公平方法增量。证据为
`rerun_fast/pope_v15_AZ32_streamwait.{official.score,comparison}.json`。

NeXT官方TextVQA EADP K128（名义640）完整5000题，独立CPU复算**57.958%**；
比旧协议E58.136低0.178点、比论文59.2低1.242点。改变guidance并未自动提高
基线。同协议AnchorZip仍在生成，完整评分后再计算方法差。原包v15 SQA128
完整同预算配对仍为E69.5588/AZ69.2117（净−7题）；Qwen legacy Avg10同机
+0.3608点、canonical MME-P128同机+16.2941分属于此前已核验结果，不能据此
宣称每模型/任务整体占优。

本次状态检查另发现POPE控制阶段2被外部Ollama的0 MiB CUDA上下文阻塞，
原并发门把所有NVML条目一律当GPU模型。禁止杀外部Ollama；恢复只接已预定
EADP32控制，保留第一阶段工件，冻结原worker与全部实验超参。只读监控已增加
明确0 MiB Ollama上下文豁免和阶段间等待超过10分钟告警，相关9项CPU边界验证
通过；未知0 MiB进程仍保守计数。实际恢复结果以控制state/resume登记为准。

16:52:47恢复程序已经实际启动EADP32 worker1692442（监督器1692429），与NeXT
1643948两个GPU模型并发，模型正在加载；外部Ollama未动。旧空闲监督器确认无
子进程才终止，旧state、AZ完整预测/协议/评分及原worker源码SHA均保留。并发
门7项CPU分支验证通过；实际free19608MiB满足实测19134+400MiB余量。恢复
记录为 `rerun_fast/pope_streamwait_stage2_recovery.json`，巡检已识别新监督程序。
后续完整E32评分仍需等待，不能把“进程已接续”写成“实验已完成”。

本次吞吐更新：关键K128配对预计今晚19–20点齐全，完整批次预计Oct9凌晨
03:00–07:00，保守预留08:00。NeXT第一E128完整生成用时约75分钟，后续仍受
并发竞争及FULL时长影响；首份v15 Text运行后再校准工期。

## 2026-10-08 18:12 北京时间独立巡检：控制闭环与新完整结果

EADP32 streamwait完整8910题评分均F1 **84.073**，AZ32同协议为83.084，
方法差 **−0.989点**，image-cluster配对95%CI[−1.817871,−0.163651]。
两臂完整性通过、random仍2910题、退出码均0、2334重复输入组分歧均0；
17:32:33的合法success marker已释放，lane1于17:32:37自动接续TextVQA E128。
证据为 `rerun_fast/pope_v15_EADP32_streamwait.{official.score,fair_comparison}.json`
及 `pope_v15_AZ32.stream_control.finished.json`。公平方法差低于0应保留；
控制success与方法是否胜出无关，每配置一次全量仍不构成独立重复性证明。

NeXT官方TextVQA K128两臂各5000复合题键和官方完整评分已核验，CPU精确
EADP **57.958%**、AnchorZip **57.588%**，方法差 **−0.370点**。该初始配对
无新增runtime trace。NeXT原包SQA E128新完成2017题、1365正确，
**67.6747645%**；18:07:50评分后自动接续AZ128，其配对尚未完成。
汇总器正常运行，6完整行与CSV一致，`errors=[]`；新分数只来自完整预测。

巡检证据目录为 `outputs/audit_followup_20261008/rerun_batch/hourly_monitor/check_20261008T100930Z/`。
两次快照18:09:30和18:11:21显示v15 TextE128完整行4045→4241/5000、
NeXT SQA AZ128为150→340/2017，输出与日志同时推进；/proc完整命令、
父子和启动身份通过，PID1779765/1829368保持不变。两lane监督器和汇总器
存活，已完成的控制监督器正常退出，A40实际两个GPU模型。28实验产物路径、
25相关日志未发现Traceback/OOM/FAILED；两lane来源SHA22/22、21/21、
两控制各8/8及launcher、as-is来源均匹配，完整预测/评分来源SHA通过。
两lane仍在生成，不能报全部完成；截止尚余约17小时48分。本次只保存CPU
巡检证据并追加事实记录，无进程重启、冻结源码或超参数修改。

## 2026-10-08 19:15 北京时间独立巡检：新完整配对与持续接续

v15官方TextVQA K128 E/AZ各5000复合题键、完整题面/顺序/runtime及官方
评分通过，独立CPU精确复算 **56.390% / 56.604%**，方法差 **+0.214点**。
AZ官方打印56.60是四舍五入；方法差用精确成绩计算。两臂分别18:18:20和
19:06:00完成，lane1自动接续SQA E64。NeXT原包SQA K128 E/AZ各2017题，
正确数 **1365 / 1373**、准确率 **67.6747645% / 68.0713932%**，净
**+8题/+0.3966287点**；18:27:39评分后lane2正常接续TextVQA E64。
未为这些新配对重复bootstrap，不据点估计宣称显著或整体稳定增益。
NeXT Text128仍为AZ−E **−0.370点**，POPE公平控制仍为 **−0.989点**。

19:07:54→19:14:54，v15 SQA E64完整行307→1579/2017、NeXT Text E64
3931→4618/5000，两worker PID1908673/1856362保持启动身份，输出和日志
同时推进；/proc完整argv、父子关系及两lane监督器的前轮启动ticks均匹配。
前轮worker已退出且完整评分后正常接续，A40实际仅两个本计划GPU模型；
控制监督器已正常完成退出，合法两8910/exit0/评分success marker通过。

28新实验已完整评分9个，50764/118917条完整新预测；2个生成中、17个未
启动。summary.completed_rows=9包含6新Text/SQA与3历史v15 SQA，三POPE
另行完整审计，不将该汇总行数直接用于新实验计数。JSON/CSV一致且errors=[]。
来源/输入/协议/完整预测与评分SHA实际988/988匹配，两lane冻结来源22/22、
21/21通过，25相关日志无Traceback/OOM/FAILED。已重算三POPE混淆矩阵、
重复输入组与所有完整Text/SQA精确成绩，独立audit_errors=[]；random仍
2910题，未为初始NeXT Text128补称runtime trace。

本次证据在 `outputs/audit_followup_20261008/rerun_batch/hourly_monitor/check_20261008T110754Z/`：
`inspection_record.json`、`score_integrity_audit.json`、`source_log_audit.json`、
`process_identity_audit.json`、`active_prefix_integrity.json`和三次CPU只读快照。
只追加事实与本机记录，未操作GPU/外部进程、修改冻结生成代码/参数/预测或
提交推送。两lane仍generating，整体未完成；截止尚余约16小时45分。

## 2026-10-08 20:15 北京时间独立巡检：新增三组完整配对与正常接续

v15原包SQA K64完整2017题配对：EADP1388正确/**68.8150719%**、
AnchorZip1394正确/**69.1125434%**，净 **+6题/+0.2974715点**；K32为
1387/**68.7654933%** 对1402/**69.5091720%**，净 **+15题/+0.7436787点**。
NeXT官方TextVQA K64各完整5000复合题键，独立精确成绩 **55.352% / 56.026%**，
方法差 **+0.674点**（官方打印55.35/56.03）。完整分母、题面/顺序/runtime、
官方解析及精确评分通过；未计算这三组新配对的bootstrap，不声称显著或整体稳定增益。

20:08:34→20:15:19，v15 TextE64 PID1969775完整行 **2285→3189/5000**，
输出与日志同时推进。NeXT TextAZ64于20:12:46.016923完整评分后.067979
自动启动TextE32（约0.051秒）；原PID1927134已退出，新PID2003306的实际
完整命令、父子/启动身份、协议和来源门通过，20:13:36→20:15:19为
**71→256/5000**，末次来源补证20:15:21为260合法预测/runtime。A40实际两个
本计划GPU模型，两lane监督器及汇总器正常；控制complete、两8910完整评分、
直接exit0与合法marker再次核验通过。队列任务exit0依据冻结queue非零退出
硬门和completed状态推断，未捏造独立逐任务exit字段。

新实验完整评分 **15/28**，对应 **68832条完整评分预测**；独立评分审计
12:13:50 UTC的已写完整JSONL行数为71897/118917（含活跃前缀）。2个生成中、
11个未启动。summary20:13:39更新为15完整行=12新Text/SQA+3历史SQA，
三POPE另行完整审计；JSON/CSV一致且errors=[]，独立audit_errors=[]。
SHA核验 **1138/1138**（571唯一文件），新接续独立补证 **42/42**；两lane
冻结来源22/22、21/21无漂移，35份相关日志及4份接续日志无异常。

证据目录为`outputs/audit_followup_20261008/rerun_batch/hourly_monitor/check_20261008T120800Z/`，
含五次CPU只读快照、评分/来源/接续、进程身份、活跃前缀/runtime与进度审计。
本轮仅保存本机审计与追加事实，未重启进程、改冻结生成代码/参数/模型/预测、
提交推送或发外部消息。两lane仍generating，整体未完成；20:15:19距
Oct9 12:00截止约15小时45分，旧工期预测不是完成证明。

## 2026-10-08 21:12 北京时间独立巡检：Text64完整配对与新worker推进

v15官方TextVQA K64各5000复合题键、完整prompt/顺序/runtime、prediction
SHA及官方完整评分通过，独立精确EADP **54.918%** / AnchorZip **54.978%**，
方法差 **+0.060点**；未计算新bootstrap，不声称显著或整体稳定胜出。
NeXT Text E32完整5000题、精确 **51.996%**，AZ32仍生成，不评分前缀。

v15 AZ64于21:10:35.183352完整评分后.241644自动接TextE32（0.058秒）；
旧PID2026207退出，新PID2084699完整argv/父子/启动身份与冻结来源、协议
通过，21:11:19→21:12:31预测 **77→247/5000**，输出与日志推进。NeXT
AZ32同PID2064425于21:08:02→21:12:31为 **1175→1674/5000**，输出与
日志推进。A40两计划内模型，显存21662/19342MiB；两lane监督器、汇总器
正常，控制complete、两8910完整评分/direct exit0及合法marker通过。

新实验完整评分 **18/28、83832/118917条预测**，2个生成中、8个未启动。
独立评分审计21:11:10写出85411完整行（含前缀）；summary在正常60秒刷新
间隔内由17增至18完整行=15新Text/SQA+3历史SQA，三POPE独立审计。
JSON/CSV一致、summary.errors与audit_errors均空；来源1265/1265 SHA
（575唯一比较文件），43份唯一相关日志无异常，两lane冻结来源22/22、
21/21通过。21:12:53补证新E32的298预测/runtime、NeXT AZ32的1717
预测/runtime身份通过，NeXT名义预算与实际几何总保留数继续区分。

证据目录`outputs/audit_followup_20261008/rerun_batch/hourly_monitor/check_20261008T130802Z/`
保留五次CPU快照、独立完整评分及来源接续补证、进程身份、进度与前缀runtime。
Git分支仍`codex/anchor-completion-validation`、HEAD `194562d`；本次只保存
本机巡检与追加事实，无进程重启/额外GPU/冻结代码参数预测修改/提交推送或
外部消息。两lane仍generating，整体未完成；21:12:31距截止约14小时47分，
工期预测继续不是完成证明。
