# EADP Visual Token Pruning 项目交接记录

> **2026-11-15 代码与历史实验证据归档**：用户明确授权提交并推送本轮及此前未提交的实验结果，当前分支 `codex/detail-allocation-oracle`。14族实验源码、22份此前未跟踪文档及本handoff一起纳入；第三方CatchUp固定 `18374ec779df0eaab07b788b777a75e1ecd812a3`、LayoutLite固定 `7e398a2ff4f23f0cb836e36e0e81d21d379d8fc2`，以 `.gitmodules` 和gitlink登记，不复制嵌套Git目录或权重。历史证据入口 `Qwen_vl/outputs/research_archive/20261115/README.md`，清单 `index.json`（SHA256 `c70aa9f62ffca0b9ac5b868480935c49278019883f26e91f1e42f479996b4333`），归档脚本 `Qwen_vl/scripts/archive_research.py`。19族/2790个原始证据文件（2079100043字节）无损压缩为24个≤32MiB分片（合计279365730字节），另有9份原字节Markdown报告副本供浏览；所有分片SHA、完整gzip SHA/CRC、逐成员SHA/大小和报告副本校验全部PASS。完整保留5个SQLite主DB：student_full 37443 QA、paired 4416 QA/690计时、transfer79443/1512、broad203252/3360、detail14208/216；全部WAL为空、quick_check正常、普通只读与immutable主文件计数相同，未执行checkpoint或修改原DB。协议、原始JSON/JSONL、manifest、freeze、gate、训练/选点摘要、日志、表格以及失败/作废历史均保留原字节。权重、特征缓存、源图、parquet和SQLite临时文件仍留本地，排除项路径/大小/原因记入index；这不是模型/数据的完整迁移包。恢复先运行 `python3 -B Qwen_vl/scripts/archive_research.py verify`，再按README将某族分片拼接解包到独立新目录，勿覆盖冻结输入/结果。归档不改变科学结论，不补跑GPU或训练；下方历史启动状态以各完成条目为准。

> **2026-11-14 单/双细节区域诊断已全部完成：停止本版几何，不启动区域选择器训练**：`Qwen_vl/outputs/detail_allocation/run_20261114/` 于北京时间16:53:56结束，实际启动后75分48秒；14208/14208 QA、216/216计时齐全，91/91 GPU门禁PASS，missing/failed/empty/events均0，1条128-token截断保留主分母。自动报告及独立只读build_report(write=False)均核验714冻结文件与原生预算/DS/位置/cache/答案依赖；另用锁定官方process_line/hit_calculate独立复算14208条原答案分数，0处不一致。SQLite独立quick_check=ok，实验进程已结束。InfoVQA/TextVQA各192新TRAIN主图、每图1人工问题，仅固定首parquet分片的小面板，不能与旧全量benchmark直接比较或称底座预训练未见。Info：full70.837、B1 52.924、R-res57.305、旧bag三seed均值53.136；Text：86.667/83.177/71.771/82.604。答案依赖的有限12候选池择优：Info single78.707、dual77.216，dual−single −1.491pp，95%父图配对CI[−4.779,+1.665]；Text single93.333、dual93.542，+0.208pp，CI[−2.188,+2.760]。两集等权差−0.642pp，CI[−2.688,+1.383]，均未显示稳定双区增益。full≥.8/R-res<.5的目标失败共55题（26/29），并非不足信息的HOLD；其中54题已有single候选≥.8（25/29），single<.5且dual≥.8的目标候选在替换控制前即为0，两集严格候选均0，不能将零计数归因于替换判据太严。按预注册判定STOP_THIS_GEOMETRY，仅否定本有限窗口方案的推进价值，不能否定一般多区域策略。全样本另有8个single<.5/dual≥.8局部案例（两集各4），5个也通过六替换低分；但7个full本身<.8，另1个full/R-res均成功，全部不在目标失败子集，且个别涉及ANLS措辞或人工票分歧，不能写完全无双区修复或反过来当作R-res修复机制成立。4dev图的固定mask计时：首token中位R-res170.438ms、B1 409.168ms、single330.458ms、dual325.760ms；固定16token分别788.167/1026.547/943.746/943.414ms。高清臂仍支付全1024-token视觉编码，且未计oracle搜索/未来selector，不能把择优分数或固定mask成本写成可部署效果/效率。报告 `evaluation/report.md/json`；未训练新学生，未改冻结源文件或旧结果，未启动后续GPU实验。

> **2026-11-14 固定预算的单/双细节区域诊断已实现并后台启动（91项GPU门禁全PASS，正式计时中）**：用户授权思考下一步并实际跑实验。已完整读handoff、检查旧结果与原始相关论文后，新建分支 `codex/detail-allocation-oracle`，HEAD仍a724122d，保留所有既有改动。新隔离代码 `Qwen_vl/scripts/detail_allocation/`；当前run `Qwen_vl/outputs/detail_allocation/run_20261114/`，后台PID3106269，默认命令 `CUDA_VISIBLE_DEVICES=0 bash /media/disk2/YZX/research/EADP/Qwen_vl/scripts/detail_allocation/run.sh`。北京时间15:38启动，15:42:54完成91/91项GPU门禁；已实际开始正式端到端计时，当前147/216条、QA 0/14208条、错误事件0，阶段timing。独立只读检查SQLite quick_check=ok，已存计时均为正、实际token数等于cap；后台将自动继续全部计时、QA和报告，尚无效果结论。35项CPU测试通过，含实际run→SQLite→report合成端到端与续跑、geometry边际严格匹配、真实factorial替换选集、答案依赖审计及所有决策分支。714文件冻结，hash `7948724112e9e263b88f83cc4ad016731293ab963955d040351b2688f907113a`；新manifest `6212fe2defc357fea737fd5a6ae16e2fcf6b81c9936565efe585b739d33c6d5d`。InfoVQA TRAIN/TextVQA TRAIN各192独立主图+2dev，每图1原人工题；只用固定首parquet分片（Info998题/211图、Text1731题/1116图），不能称全TRAIN均匀采样/底座预训练未见。当前学生全部217页和旧13集源/物化图字节、RGB、dHash排重，访问候选无重合/嫌疑，Text10票完整。CPU初稿96主扩至192以避免旧Text约9pp full/Rres差距导致不足12个可分析失败；未看新模型结果，原96manifest备份保留，原96是192前缀、dev与图hash不变。Info首片仅78.9MB按固定HF revision下载，未下载整库。固定N1024原生V/三DS/3D：G64均匀锚+两块8×13窗口非锚各96=K256，12相邻single与12分离dual通过同一B名册置换，A/B边际和逐token频数完全一致，覆盖988/1024、同36位置未覆盖。每题7基线(full/B1/Rres/均匀256/原bag3seed)+24几何+6控制=37输出，共14208 QA。先固定候选生成，再用原QA score选最佳dual（同分action_id字典序），对其做2组预固定donor的00/10/01替换；控制实际选集依赖答案，明确audit.answer_in_selection=True，生成输入仍无答案。max12 oracle只是有限bank诊断，非deployable结果/可学习性/新颖性。所有高清臂仍支付完整1024-token ViT，非实际crop/多图编码，不能继承Rres速度或推出原图因果局域性。主判据dual−single连续分数、每集独立父图bootstrap及等权汇总；full≥.8/Rres<.5各集不足12则HOLD，只有两集差>0、池差≥3pp且95CI下界>0、严格joint候选≥8且每集≥2才GO_DIAGNOSIS_ONLY，否则STOP_THIS_GEOMETRY；GO不自动训练。严格候选另需最佳single<.5、dual≥.8且6控制全<.5。4dev图9固定臂cap1/16、warm2/block3共216计时，仅固定mask成本不含oracle搜索。GPU先stock/native identity、32×32真实位置、全部固定臂共享/新鲜特征输出、DS/预算、EOS及冻结参数门禁。启动曾遇ollama占约27GB，现已空闲；脚本等待物理GPU0 free≥24576MiB且无其他compute，默认30秒/两次稳定、最长24h，不结束他人进程；只支持CUDA_VISIBLE_DEVICES=0。协议 `docs/detail_allocation_prereg.md`；状态status.json、run.log、最终evaluation/report.md/json。未训练新模型、未修改旧冻结源码或结果，尚无本轮效果结论。

> **2026-11-14 中文汇报与创新边界复核（讨论，未启动新实验）**：用户提出R-res已有效，是否加入其他创新。合并旧3集与新10集，学生三seed均值相对B1有7/13正点估计，相对R-res仅TextVQA和InfoVQA更高（2/13），不能称逐项都显著。新10集计时按paired median_speedup s换算耗时节省1−1/s，学生对B1首token节省17.59–21.16%、固定16token节省7.23–8.06%；相对R-res首token耗时约1.89–2.08倍、16token多19.1–21.8%，限每集8图面板。公开原始论文复核发现简单组合已有直接先例：[Q-Zoom](https://arxiv.org/abs/2604.06912)已有低清足够则跳过高清、query区域补看；[GapSight](https://arxiv.org/abs/2608.21762)已有global-vs-crop答案loss/margin增益监督gate、ROI及预期收益；[Foveated Compression v2](https://arxiv.org/html/2610.07729v2)已有固定预算粗细混合、预算匹配区域干预teacher，且其学习方案仍不及强整图缩放，视觉encoder仍跑完整高清。故不能把R-res+学生/gate/crop/条件增益teacher直接写成新贡献。潜在研究问题尚未证实：①低清失败中区分证据采样已消失与证据仍可辨而模型读错，再决定改采样或做可辨识性约束的蒸馏；普通learned resizer/KD本身也已有先例。②InfoVQA是否存在足量必须联合保留多个分离区域才修复的错误，在同总token及真实encoder成本下单区域方案不够；多crop本身不新，必须先做独立小面板预算匹配诊断，有明确收益才继续。当前没有证明上述机制、新颖性或部署收益，不再为了A+B拼模块，也不根据已看过的测试集选路由。所有GPU实验保持结束，未改冻结源码/配置/结果。

> **2026-11-14 其余10集扩评全部完成：R-res更强，当前学生未成为通用修复方案**：`Qwen_vl/outputs/student_broad/run_20261113_v2/` 于北京时间今天08:32:07结束，历时15h28m40s，203252/203252 QA、3360/3360计时，149/149 GPU门禁PASS。自动报告完成并核验8937冻结文件；本次独立只读CPU核对SQLite quick_check=ok、10集×7臂全部题数及各48条计时完整，missing/error/empty/events均0，DB/status/gates/report冻结hash一致，GPU实验进程已结束。本次独立核查没有重复全量答案评分。261条128-token截断保留主分母，主要集中Hallusion，不能写无截断。按各集原生主指标点估计、所有3seed均报告，学生均值仅InfoVQA超过R-res（ANLS48.611 vs45.653，B1 46.827，full66.267），其余9集低于R-res；不等同已做原生分组指标差值的显著性检验。ChartQA学生71.080超过B1 66.880、仍低于R-res76.920/full82.320；AI2D学生78.681低于B1 81.606/R-res84.035/full84.974，MMStar55.844低于60.933/64.533/66.333，显示不能从旧Doc/Text/OCR直接推广为通用收益。其他原生结果（B1/R-res/学生均值）：MMBench EN circular83.824/83.591/81.089，CN81.579/83.127/80.650，RealWorldQA66.275/66.275/64.488，Hallusion有图951题aAcc67.508/71.609/67.893，MME原生点数2331.855/2440.729/2355.770，POPE F1 89.051/88.546/85.822。Hallusion1129完整原表扩展仍单独报告，不能与951常规图模式混用；MME点数不与百分制宏平均。R-res在10个配对图像计时面板的cap1/16均快于三个学生；样本每集8图，不能写全量吞吐。新的主要结论是保留R-res强参考、针对InfoVQA等具体损失做诊断，不能将当前学生直接宣称为已解决R-res跨任务掉点，更不能测试后按答案择优拼成方法。完整报告 `Qwen_vl/outputs/student_broad/run_20261113_v2/evaluation/report.md/json`；原始SQLite、日志与协议同run目录。源代码与冻结配置未改，无需补跑，尚未启动任何新修复或路由实验。

> **2026-11-13 R-res与冻结学生的其余10集扩评已通过149项GPU门禁，正式计时中**：用户要求全面检查R-res在其他任务的下降，以及冻结学生是否互补。新代码隔离于 `Qwen_vl/scripts/student_broad/`，当前有效run为 `Qwen_vl/outputs/student_broad/run_20261113_v2/`，PID891200、单A40 GPU0、北京时间17:03:26启动；北京时间17:07:02已通过149/149项GPU门禁，正式计时已保存21/3360条、错误事件0，QA尚为0/203252；后台将自动继续计时、全量QA与报告，不宣称新效果。RealWorldQA765、ChartQA_TEST2500、InfoVQA_VAL2801、AI2D_TEST3088、MMStar1500、HallusionBench1129、MME2374、POPE5127、MMBench_DEV_EN_V11和CN各4876，共29036题，full/B1/B2/R-res512/旧bag_rank seed11/23/37七臂共203252条QA；旧Doc/Text/OCR不重复。没有新训练、teacher或目标参数适配，三个checkpoint保持源validation选点。41项CPU测试通过，真实pandas原行→原生Qwen提示与runtime完整消息29036/29036逐项一致，原生指标全量CPU聚合烟测通过；fit+val精确与dHash近重复嫌疑均0，不能据此声称底座预训练未见或项目盲测。全部输入/代码/权重溯源冻结8937文件，freeze为 `ee11eb298f4e4b0ff20c2d3d472724462e20b000a35a5edf078468d53fd0aee8`。长MCQ实际最大479token：仅提高无位置编码bag的输入长度保护至4096，学习参数不变、无截断，逐seed原短输入选集等价门禁。保留Hallusion178无图控制，1129为完整原表扩展、951有图才对应VLMEvalKit默认skip_noimg图模式；MMStar2条空正确选项不删；AI2D85条NA/None错误干扰选项按pandas默认NA口径省略，全量metadata逐行核验通过。MME官方1187两题组保持，统计区间按1177源像素cluster。计时每集8图、3block、2warmup、cap1/16共3360正式调用，fresh端到端、单请求非吞吐。主报告使用原生MMBench循环、MME点数、POPE F1与Hallusion a/f/q，不跨量纲混算。首次run_20261113仅门禁fixture将缺失hint变空字符串、使官方MCQ额外加空Hint行而5项失败（其余144/149通过），无正式QA/计时；原推理提示本已正确，修复仅门禁fixture，旧日志/失败门禁/12份源码协议快照保留。v2引用初始run图片，勿删除初始目录。协议 `docs/student_broad_prereg.md`；当前状态 `status.json`、日志 `run.log`、最终 `evaluation/report.md/json`。同冻结续跑命令 `CUDA_VISIBLE_DEVICES=0 bash /media/disk2/YZX/research/EADP/Qwen_vl/scripts/student_broad/run.sh`。本轮只诊断互补和失败范围，没有新路由/修复算法；不能测试后按数据集或答案择优当统一方法，未来输入路由应另设训练/dev并计入全部高分辨率成本。

> **2026-11-13复核：冻结基础学生跨数据集评测已完成，相对EADP收益复现，尚未全面超过R-res**：`Qwen_vl/outputs/student_transfer/run_20261112_v2/` 于北京时间11月12日22:51:37完成（约7小时15分），79443/79443真实QA、1512/1512多图计时，missing/error/empty均0，43项GPU门禁PASS。无目标重训，复用run_20261111的bag_rank seed11/23/37，未按测试选seed。百分制：DocVQA5349题 full94.537/B1 72.118/B2 68.315/R-res80.307，学生75.441/74.517/76.473，均值75.477（对B1 +3.359pp）；TextVQA5000题 full83.740/B1 77.788/B2 77.286/R-res74.544，学生79.852/78.886/80.614，均值79.784（+1.996pp）；OCRBench1000题 full85.100/B1 74.200/B2 73.400/R-res79.600，学生76.7/77.4/77.8，均值77.300（+3.100pp）。9项seed×dataset对B1点估计均正，8项父文档配对95%CI严格高于0，OCR seed11下界恰0；对B2全部区间高于0。相对R-res，Doc三seed均显著差、Text三seed均显著好，OCR点估计均差但两seed区间跨0；R-res在全部计时面板更快。36不同图（每集12）单A40配对计时：学生对B1首token延迟低约11.7–12.4%，固定16token总耗时低约4.3–5.0%；对B2约2.6–3.2%和0.9–1.4%，不是全量benchmark时延或吞吐。fit+val精确训练图/父文档重合0；Doc排除19题/4父文档dHash嫌疑后5330题学生均值75.471，结论不变。三公开集合均有项目历史曝光，不能称项目未观察盲测或底座预训练未见；支持冻结学生未经目标适配的迁移。9条截断全部保留（Doc6/Text1/OCR2）。独立只读CPU全量复算原答案分数、native预算/DS/位置/选集/trace及SQLite quick_check全通过，242新冻结文件、69源冻结文件、3学生hash和4底座weight stat均未变，无需补跑。当前值得继续的是基础梯度蒸馏学生的监督归因和强参考竞争力，不把该收益转记给已失败的ordered_pair/multimode，不宣称新颖性已成立。报告 `evaluation/report.md/json`，协议 `docs/student_transfer_prereg.md`，原始SQLite与日志同run目录。首轮run_20261112只因门禁sibling oracle导入失败，无目标QA/计时，失败记录和源码快照保留；v2仍引用其隔离图片，勿删除。所有GPU实验进程已结束。

> **2026-11-12 两条 paired-pruning 实验已完成，新增机制均未显现稳定收益**：用户启动的 `Qwen_vl/outputs/paired_pruning/run_20261111/` 于北京时间11月12日01:23结束，历时3小时6分；18个学生各900更新，DocVQA TRAIN父文档独立test64页/192题，23臂共4416/4416生成、690/690计时齐全，错误事件0；仅B2有1条截断，主分母保留。原生GPU门禁PASS；本次只读复核69项冻结文件及模型metadata/权重stat与协议一致，未启动新GPU工作。ANLS百分制：full97.592、B1 facility78.742、B2 block8 75.713、R-res81.626；三seed均值bag_rank85.006、ordered_rank81.870、ordered_pair81.479、canonical79.846、token_marginal80.731、multimode78.168。第一条ordered_pair相对bag三个seed均下降（−2.751/−3.465/−4.366pp），相对ordered_rank两降一升；第二条multimode相对canonical均值−1.678pp、相对token_marginal−2.563pp，无稳定增益。不要把训练对照bag_rank的改善算作两个新增机制的成功。bag相对B1三个seed点估计均正，其中两个父文档配对95%CI高于0，一个跨0；相对R-res三个区间均跨0，不能认定超过R-res。仅开发页TTFT：bag约403ms、B1 443ms、R-res253ms，非跨页效率结论。validation离线gold-mask own对swap为+41.790pp，支持问题条件差异存在；68个低重合成功集合对中交叉后8组未达预设成功阈值（其中4题ANLS=0）、60组仍成功，只支持局部整集相容性现象，非部署收益。当前应暂停本版两种新增目标，保留基础学生作为需更大独立面板复测的候选；不能否定一般query语义或多充分集方向。完整结果、三seed及配对区间见 `Qwen_vl/outputs/paired_pruning/run_20261111/evaluation/report.md`、`report.json`，机制在 `diagnostics/mechanisms.json`。这不是此前5349题官方VAL，不能直接比较两轮绝对分数；本轮数字子组仅8题，不能推广其单独结果。

> **2026-11-11 两条新实验已实现，等待用户启动 GPU**：用户选择“同页问题变化监督”和“多个充分选集的整集蒸馏”，代码位于 `Qwen_vl/scripts/paired_pruning/`，完整运行协议见 `docs/paired_pruning_experiments.md`。统一脚本 `run_experiments.sh` 依次完成 CPU 测试、数据冻结、原生 GPU 门禁、真实 K256 候选标签、validation 机制诊断、两组三臂各三个 seed 的训练、完整 held-out QA 和计时、配对统计报告。默认新 TRAIN 父文档 fit128/val24/test64/dev1，每页3道不同人工题且至少1道非数字；所有学生保留原生 V/三DS/3D 位置。实现阶段只运行了 CPU 测试、合成端到端与本地环境/数据检查，没有启动真实 8B GPU、teacher 或训练实验。用户将自行启动，之后回来分析结果；不要把代码完成写成方法有效。命令：`CUDA_VISIBLE_DEVICES=0 bash /media/disk2/YZX/research/EADP/Qwen_vl/scripts/paired_pruning/run_experiments.sh`。默认结果目录 `Qwen_vl/outputs/paired_pruning/run_20261111/`，状态 `status.json`，最终 `evaluation/report.md` 与 `report.json`；原命令可续跑，源码或协议变化必须使用新的 `--run-dir`。两个实验是独立比较，不强制合并为最终方法。

> **实现验证补充**：52 项 CPU 测试通过，含两组训练→checkpoint 故障恢复→真实评估接口的合成端到端→报告→再次续跑。真实原始数据仅 CPU 核验已足额取得217独立父文档/651题；所有 question+suffix 最长43 token，fit/val/dev首答案最长24 token，未对test答案生成teacher标签。数字子组 fit/val/test/dev 仅13/3/8/0题，后续不得把该子组当作充分功效的结论。正式启动仍会重新冻结自己的数据及源码并执行 GPU 门禁。

> **2026-11-09 DocVQA_VAL 全量评测已完成**：5349题/938源父文档，full、native EADP facility(b1)、B2、R-res512、magnitude V教师学生seed11/23/37共37443条全部成功，七臂均覆盖全量，无error events。统一原生Qwen3自定义VQA提示、DS/3D、生成128上限；官方ANLS（百分制）分别为94.537、72.118、68.315、80.307、66.094/59.816/67.520。学生相对B2分别−2.221 [−3.890,−0.530]、−8.499 [−10.219,−6.782]、−0.794 [−2.464,+0.863]个百分点（按938源父文档配对bootstrap 10000次）；均低于B1和R-res。旧32题数值exact优势没有在此全量ANLS与提示设置下复现，二者指标、题型和提示不同，不能直接归因于特定因素。full/B2/B1/R-res/学生各seed截断数0/1/1/2/2/11/6，主结果保留所有样本；独立逐题官方ANLS与实现复算一致、输入及冻结完整性通过。学生fit/val/test/dev与VAL父文档及原始像素无重合，但部分VAL已被以往项目观察，无法排除pretraining及相似重编码污染。16项双问题缓存/新鲜GPU门禁通过；VLMEvalKit PIL图片重编码在隔离目录复现并校验。协议 `docs/student_full_prereg.md`；报告 `docs/student_full_results.md`；全量结果和复现日志 `Qwen_vl/outputs/student_full/`、分析与截断敏感性 `Qwen_vl/outputs/student_full_aux/postprocess_20261109T120758.272152Z_k4f0tlgm/`。

> **2026-11-09 前向学生蒸馏探索已完成**：DocVQA TRAIN按独立父文档与原图划分fit120/val16/test32/dev1，排除所有旧面板；原生Qwen3-VL-8B N1024→K256。四组学生均输入V/三DS及问题embedding（无答案或backward），共享1,311,105参数结构，仅离线teacher标签不同，每组3 seed、各900步，只按val Top8 recall选点，12个seed均做真实QA。测试full32/32、B2(EADP scoring+block8)22/32、R-res512 19/32、random15/32；magnitude V-only teacher学生为31/31/32，joint teacher为29/32/29；signed V为29/29/28，joint为30/25/29。magnitude V学生相对B2三个seed分别净增9/9/10题，各配对bootstrap 95%CI分别[+12.5,+46.875]、[+12.5,+43.75]、[+15.625,+46.875]个百分点；N=32、数值QA及多项比较，不能推广到跨数据集或原版EADP官方facility。四路teacher未显现稳定增益；注意学生**全部都使用四路视觉输入**，并非证明多路输入无用。全部640问答完整无截断，31项native、52项student输入门禁和独立冻结/数据/选集审计通过；gold teacher仅在训练冻结后作离线对照。单A40开发页10组配对TTFT中位：学生seed11 359.48ms、B2 359.44ms、R-res214.42ms、full488.47ms；学生与B2耗时相近，R-res更快，不代表跨页效率。训练120小样本与DocVQA TRAIN不保证预训练未见/跨版式泛化；拟在大规模独立官方benchmark、不同问题与budget以及EADP完整facility上复测，不按测试挑seed。详见 `docs/student_distill_results.md`、`docs/student_distill_prereg.md`、`Qwen_vl/outputs/student_distill/`。

> **2026-11-09 多流梯度选集探索已完成**：新DocVQA TRAIN 32个独立父文档、每页1数值QA，Qwen3-VL-8B原生N1024→K256；每题使用原始标准答案的完整模型单次四流梯度，仅作离线、标签依赖机制诊断，非可部署方法或严格上界。Full 31/32、R-res512 23/32、B2 23/32、random 18/32；坐标绝对梯度×特征magnitude的V-only/joint各31/32（无正确性变化）；signed V-only/joint为22→27/32（7修复2破坏，+15.625pp，配对95%CI[0,+34.375]pp），DS关闭时22→23/32，交互+12.5pp（CI[−6.25,+34.375]pp）。有符号多流可能有局部信号但不足以证明稳定性或DS独特因果作用；magnitude不支持多流增量。不得比较成部署方法胜过B2或R-res。首次入口在第一题teacher forward前因inference-mode加载权重报错，已保留旧manifest/log，修复后重跑31项CPU/GPU门禁并重新冻结；32题480输出完整无截断。仅开发页10组配对计时：R-res/B2 TTFT中位数175.39/320.65ms；gold teacher主面板离线总耗时中位0.563秒/题且峰值约21.8GiB。详情见 `docs/multistream_oracle_results.md`、`docs/multistream_oracle_prereg.md` 和 `Qwen_vl/outputs/multistream_oracle/`。

> **2026-11-09 attention校准探索已完成**：原生Qwen3-VL-8B、32个新DocVQA TRAIN父文档各1数值QA，N1024→K256；固定VisionZip D205/C51选集、V/三DS/位置。显式mask零偏置20/32，普通簇大小count21/32，相似度有效簇大小effective19/32（相对count 0修复2破坏，−6.25pp，95%CI[−15.625,0]pp），RESTORE式prefill21/32；普通count相对maskzero 2修复1破坏（+3.125pp，CI[−6.25,+12.5]pp）。不推进这版effective，不据此否定一般内容感知校准。R-res20/32；B2硬剪枝zero/constant/RESTORE为17/17/16。显式mask本身改变1条VisionZip答案的正确性，故不能直接将相对native的差额归因于校准。单A40开发页端到端TTFT中位数：native VisionZip 304.83ms、maskzero328.35ms、count330.64ms、effective329.76ms、R-res137.30ms；每臂3预热、10配对block，当前Python hook/mask的16-token总耗时较高，非优化内核结论。134项门禁通过，主问答448输出完整、无截断，原始数据及结果独立审计通过。首次启动的模块导入冲突发生在冻结/任何QA前，已修复并重跑门禁，失败日志保留。详见 `docs/attention_calibration_results.md`、协议 `docs/attention_calibration_prereg.md` 和 `Qwen_vl/outputs/attention_calibration/`。

> **2026-11-08 最新实验**：CatchUp“先修正候选、再选择”做完Qwen3-VL-8B两阶段探索性真实QA。16个此前未用的DocVQA TRAIN父文档、各1个固定数值QA，完整模型15/16、R-res512 7/16、两阶段stale基线2/16、离线dense-key全池1/16、dense-key限Top512 2/16、提出的LTC限Top512 0/16；相对基线0修复2破坏。LTC平均改变59.8/256个选择；dense参考选择在stale Top512的平均覆盖94.19%。基线偏弱且N=16，结果不支持推进此配置，不能否定一般reranking或官方CatchUp。此为Qwen3 L2/L7各K256适配，不是官方四阶段LLaVA复现；被跳过token未接收第三条DeepStack，dense对照同时包含这部分差异。一张开发页TTFT三次中位数：基线349.26ms、LTC352.07ms、R-res175.11ms。CPU 25项算子检查和8B identity/缓存/位置门禁通过；GPU首轮mask内核差异失败和修复过程保留。完整报告见 `docs/catchup_rerank_results.md`、协议 `docs/catchup_rerank_prereg.md`、数据/日志/预测在本地 `Qwen_vl/outputs/catchup_rerank/`（该目录受gitignore影响），脚本位于 `Qwen_vl/scripts/catchup_rerank/`。

> **2026-11-08 最新实验已完成**：LayoutLite局部奖励无训练诊断完成32个新TRAIN父文档、80个人工数值QA、192/192条OCR。2页完整参考截断，30有效页；主比较只有5文档/10对，未达预注册12文档/24对门槛，正式判定 **HOLD_INSUFFICIENT_INFORMATION**。整页/局部均值/tail20排序一致率点估计90%/80%/50%，tail20相对局部均值−30pp、95%CI[−80,+20]pp；离线择优方案的页面平均证据覆盖98.33%对96.67%，无推进训练信号。原生与批处理门禁及2527项产物审计通过；GPU已结束。注意：这是官方FireRed-OCR-2B+LayoutLite head的等K50%动作诊断，不是8B/25%保留率或原阈值推理复现；普通QA smoke0/6，**只测字面证据覆盖代理，不能称真实QA准确率或训练效果**。不训练这版tail20，不据此否定一般local reward。报告 `docs/local_reward_diagnostic_results.md`，协议 `docs/local_reward_diagnostic_prereg.md`，原始产物 `Qwen_vl/outputs/local_reward_diagnostic/`；串行吞吐预跑已归档作废。

> **2026-11-08 最新实验已完成**：用户要求转向改进已有强方法，当前分支为 `codex/visionzip-binding-oracle`。VisionZip/PPE 的区域合并 oracle 已完成62个独立文档候选，56对通过完整模型四问验证。**同K256、同dominant/anchors，限制跨区域合并与不限区域对照均为22/56四问全对；单问149→156/224，净增7问（+3.125pp，文档bootstrap95%CI[+0.45,+6.25]pp）。存在局部信号，但没有新增完整区分对，按预注册停止此具体机制，不训练边界预测器。** 降分辨率参考为25/56，未编辑QA为38/48，优于合并oracle的30/48。报告见 `docs/binding_oracle_results.md`，协议见 `docs/binding_oracle_prereg.md`。原生SDPA、PPE单例、真实K256及三条DeepStack gates均通过。前300个几何近似版本的压缩回答已归档作废，全部按精确smart_resize→square映射重跑；不要引用旧几何结果。GPU进程已结束。不能把这轮写成“完全无效”或“绑定保持一般不可行”，也不能把oracle的全部增益算作合并收益。

> **2026-11-08 新进展**：本文下方 M1 状态是历史记录，已被 M2–M14 及新一轮实验取代。
> 当前分支为 `codex/cf-semantic-mining-pilot`。最新候选是“由当前压缩模型的语义区分失败驱动编辑样本挖掘与持续训练”，协议见 `docs/cf_mining_pilot_prereg.md`，相关工作边界见 `docs/cf_mining_prior_art.md`。
> 这轮初筛及六组训练对照、补充参考均已完成。可挖出真实数字差异盲区，但当前训练载体严重过拟合，所有训练组的保留样本双图正确率和原始页面精确正确率均为零。不可把“发现盲区”写成“方法有效”，也不能据此否定一般的闭环思想。完整证据见 [本轮报告](cf_mining_pilot_results.md) 与 `Qwen_vl/outputs/cf_mining/`。

> 最后更新：2026-09-23  
> 当前工作分支：`m1-data-scaling`  
> 最新结果提交：`2f39b68 Correct two numbers in the M1 report`

## 0. 给下一个对话的说明

这是本项目的研究状态和决策依据。继续工作前，应先阅读本文，再按需查阅各阶段的完整报告。

特别注意：用户之前提到的“A+B”只是对常见科研方法组合方式的举例，**不是要求最终方法必须由两个模块拼接而成**。不要为了形成方法而强行缝合模块。最终方法的结构应由实验结果决定；相关论文中的模块和思想只作为候选工具与灵感来源。

## 1. 最终目标

基于已有数据和后续实验，发现并验证一个适合本项目的、可部署的 visual token pruning 方法。它可以被表述为对 EADP 的实质改进，也可以在此基础上形成一个新的完整方法。

目标优先级如下：

1. 在固定视觉 token 预算下获得可靠的下游效果，当前核心设置是 Qwen3-VL-8B、`1024 -> 256` visual tokens。
2. 最好在 TextVQA、DocVQA、OCRBench 等任务上超过 EADP；如果准确率与 EADP 相当，但速度、显存、复杂度或泛化性明显更好，也可以接受。
3. 推理必须是 forward-only。训练阶段可以使用梯度 teacher、离线标签或知识蒸馏，但不能把昂贵 backward teacher 带到部署推理中。
4. 最终评价必须包括真实 downstream generation/accuracy 和端到端效率；proxy 指标只能用于分析和筛选，不能代替最终结论。
5. 不预先规定方法必须是单模块、双模块或某种论文式结构，让实验决定最终形态。

Visual token pruning 属于 visual token compression 的一个子方向；本项目当前聚焦的是 hard token selection/pruning，而不是 token merging、quantization 或通用压缩。

## 2. EADP 的结构，以及我们实际在改什么

EADP 可粗分为两部分：

1. **Importance scoring**：在进入语言模型前，由 global/dense text-visual similarity、entropy filtering、分数融合、归一化、平滑和 polarization 等操作生成 token importance。
2. **Coverage-aware selection**：根据视觉 token 相似度，用 facility-location greedy 在 1024 个 token 中选择 256 个，然后把它们送入完整 LLM。

现有证据表明，EADP 的主要精度瓶颈是 **importance scoring**，不是 facility-location 本身。我们当前实质上是在寻找一种更好的、可部署的 token utility estimator，并重新判断强 scorer 出现后是否仍需要昂贵的 coverage selector。

当前最有希望的部署形式是 early-layer pruning：先让全部 visual tokens 经过少量 LLM 层（目前是 L4），用轻量 scorer 打分并剪枝，然后复用已有前缀状态，让剩余层只处理保留 token。

## 3. 已确认的实验事实

### 3.1 Stage 1：EADP 审计

- Facility-location 占 EADP pruning overhead 的约 92--98%，但 error decomposition 表明它只对应约 10% 的错误；约 55% 的失败来自 scoring。
- 官方 `facility @256` 在完整 450 样本评估上的 macro accuracy 为 **66.88**，selector cost 为 **42.91 ms**。
- `block8` 是 block-parallel facility greedy：macro **64.87**，相对 facility 为 `-2.01`，95% CI `[-5.40, +1.37]`，selector cost **8.49 ms**。它是有效的工程加速，但单独作为论文创新较弱。
- 使用 EADP 原分数时，plain Top-K 会严重掉点，说明 facility 当时主要是在补偿差的 importance map。

完整报告：`docs/eadp_method_discovery.md`。

### 3.2 S2-A/S2-B：梯度 teacher 证明了评分上限

- P1-G2 gradient sensitivity teacher 是目前最强的 token ranking 信号。
- `gradient + facility` 在完整评估上达到约 **78.81**，相对 EADP facility 的 **66.88** 提高约 **11.92** 点。
- 使用强 gradient score 后，selector 差异大幅缩小：facility **78.806**、block8 **79.148**、Top-K **77.372**。
- 这说明强 score 可能让昂贵 facility 不再必要；但 gradient teacher 本身需要 backward，不能直接部署。

完整报告：`docs/scoring_search_s2a.md`、`docs/scoring_search_s2b.md`。

### 3.3 S2-C0/S2-C1：简单 forward proxy 不够

- 多种 attention/forward proxy 没有复现 gradient teacher 的价值。
- 从 early hidden state 学习 token scorer 可以恢复一部分 teacher 信号，L4 通常优于或不差于 L2。
- 在 held-out 150 的公平 split 上：EADP facility 为 **61.097**，P1-G2 teacher Top-K 为 **75.150**，早期线性/查询 student 约为 **57--58**。
- 注意：这里的 **61.097** 与完整 450 样本上的 **66.88** 来自不同评估集合，后续不得混用或直接比较。

完整报告：`docs/scoring_search_s2c0.md`、`docs/scoring_search_s2c1.md`。

### 3.4 S2-C2：真正的损失集中在少量高价值漏选 token

- 学生与 teacher 的 bulk overlap 不是正确的价值货币。
- 在 student 的 256-token set 中替换少量 token：
  - teacher 排名最高的 8 个漏选 token 可恢复 student-to-teacher accuracy gap 的约 **53%**；
  - 16 个恢复约 **77%**；
  - 32 个恢复约 **97%**。
- 价值来自“加入了哪些 token”，而不是“移除了哪些 token”；removal identity 没有可测量贡献。
- 普通 student score、hidden-state norm、L2->L4 movement、冗余度、位置、邻域相似度等便宜属性都不能单独识别这些高价值 token；teacher gradient ranking 可以。
- 这是重要机制发现，但不等于最终方法必须采用 rescue 模块。

完整报告：`docs/scoring_search_s2c2.md`。

### 3.5 S2-C3/C4：改 loss 有用，但共享方向不够

- 把监督目标改到 teacher ranking head，能显著提高 teacher Top-8 token 的 retention；HEAD_RANK 将相关 recall 提高约 3 点。
- 但是 proxy/head recall 的改善没有稳定转化为平均 downstream accuracy，存在 rescue/break churn。
- per-image linear oracle 很强，说明 L4 representation 中包含信息；但跨图像共享方向和低维方向不能充分迁移。
- 因此“信息完全不在 L4”是错误结论，但“一个共享线性方向足够”同样不成立。

完整报告：`docs/scoring_search_s2c3.md`、`docs/scoring_search_s2c4.md`。

### 3.6 S2-C5/C5A：局部非线性 scorer 是目前最强可部署方向

- Token-local MLP 明显优于共享线性 scorer。
- 在正确的 H8 checkpoint selection 下：
  - LOCAL-MLP：held-out teacher Top-8 recall@256 = **0.8047**；
  - GLOBAL-CTX：**0.7978**；
  - SET-CTX：**0.8019**；
  - LOCAL-MLP-WIDE：**0.8081**。
- 简单 global/set context 与加宽网络都没有得到独立、稳定的收益。
- 因此当前不应继续无依据地堆 context 或增加模型宽度。

完整报告：`docs/scoring_search_s2c5.md`、`docs/scoring_search_s2c5a.md`。

### 3.7 S2-C6：trajectory/query 没有独立收益，数据路线仍然开放

LOCAL-MLP 的 fit-data scaling：

| fit images | held-out teacher Top-8 recall@256 |
|---:|---:|
| 60 | 0.7644 |
| 120 | 0.7853 |
| 180 | 0.7922 |
| 240 | 0.8047 |

- 60->240 总增益为 **+0.0403**。
- 180->240 为 **+0.0125**，CI `[+0.0006, +0.0256]`；一个 seed 为 `-0.0008`，因此没有满足预注册的“全部 seed 为正”条件。
- 报告按预注册规则写作 `DATA-PLATEAU`，但这不是科学意义上的数据饱和。更准确的表述是：**data-responsive, endpoint unresolved**。
- L2、DELTA、L2+L4、L4+DELTA 均未超过 L4；没有证据支持 trajectory 是残差信号来源。
- 两种 query arm 均只比 L4 高约 0.003，置信区间跨零。Token-dependent bilinear query 确实改变了约 1.5% 的选择，但对 teacher Top-8 retention 的净作用为 0。
- 最好 arm 为 0.8081，未过 0.82 内部门槛，因此 C6 没有运行 downstream generation。

完整报告和数据：

- `docs/scoring_search_s2c6.md`
- `docs/scoring_search_s2c6_prereg.md`
- `Qwen_vl/outputs/discovery/s2c6_audit.json`

### 3.8 M1：数据效应成立，超过 480 后的边际收益未决

M1 已在服务器分支 `m1-data-scaling` 完成，并把 LOCAL-MLP 数据规模扩展到 960，同时用每个规模固定 900 optimizer updates 的 primary protocol 消除了 C6 的 step-count 混杂。

Fixed-step primary ladder：

| fit images | held-out teacher Top-8 recall@256 |
|---:|---:|
| 240 | 0.8047 |
| 480 | 0.8189 |
| 960 | 0.8214 |

- 60->240 在 fixed-step 下提高 **+0.0408**，与 C6 fixed-epoch 的 **+0.0403** 几乎相同；此前的上升不是更多 optimizer steps 造成的。
- 240->480：**+0.0142**，CI `[+0.0014,+0.0272]`。均值和 CI 支持真实数据收益，但一个 seed 为负，因此没有满足预注册的全-seed resolved 条件。
- 480->960：**+0.0025**，CI `[-0.0103,+0.0158]`，无法区分小幅收益与零收益。
- 240->960：**+0.0167**，CI `[+0.0031,+0.0308]`；三个 seed 的增量为 `+0.0183/0.0000/+0.0317`。
- Fixed-epoch control 在 n=960 达到 0.8242，与 primary 的结论一致。
- M1 的预注册总判定为 **INCONCLUSIVE**：不能声称 480 后继续 data-responsive，也不能声称已经 saturation。
- 960 的均值首次越过内部 0.82 deploy line，因此现在应进入真实 downstream，而不是继续只优化 proxy 或盲目扩大数据。

M1 的重要实现审计：

- `m1_downstream.py` 当前按 held-out `test_h8` 选择单个 seed checkpoint，存在 test leakage。下游不得沿用这种选择方式；应运行所有三个 seed 并报告均值/范围，或在运行前预注册只基于 validation 的选择规则。
- `m1_downstream.py --offline/--generate` 当前仍是 guard，会直接退出，并没有真正实现 selector comparison 或 generation。报告中的“prepared”只表示 artifact 检查完成，M2 harness 仍需实现。
- 服务器上的 `.pt`、6 GB feature cache 和新增 teacher maps 被 gitignore，本地 clone 不包含；后续 M2 应继续在同一服务器工作区运行。

完整报告和数据：

- `docs/scoring_search_m1.md`
- `docs/scoring_search_m1_prereg.md`
- `Qwen_vl/outputs/discovery/m1_audit.json`

## 4. 对 C6 和现有分析的必要限定

1. **不要把 `DATA-PLATEAU` 当成真实结论。** 未证明 `DATA-LIMITED` 不等于证明 plateau；曲线在最大数据点仍然上升。
2. **数据量与优化步数混杂已由 M1 排除。** Fixed-step 的 60->240 增益为 +0.0408，几乎等于原 fixed-epoch 的 +0.0403；数据而非额外更新次数产生了主要增益。但 480->960 仍因统计功效不足而未决。
3. **`L2+L4` 与 `L4+DELTA` 不是严格相同的 hypothesis class。** 输入对之间可逆线性等价，但 DualTokMLP 在每个 channel 上分别投影并先做 GELU，网络函数族不一定对跨通道线性变换保持不变。二者的相近结果只能作为经验一致性，不能作为严格数学等价。
4. **Wrong-image trajectory control 只证明模型依赖第二输入。** 替换 channel 会制造跨图像冲突，不能单独证明模型读取了“轨迹”而不是第二份图像内容。
5. **Query 结论必须限定范围。** 当前结果只说明在 240 张训练图像、现有 rank-8 query 结构和 HEAD_RANK objective 下，没有发现可利用的增量 query 信息；不能泛化为“query 永远无用”。
6. **Top-8 recall 不是 downstream accuracy。** C3 已显示 proxy 改善可能伴随 answer rescue 和 breakage 同时增加。0.82 是内部筛选门槛，不是方法是否有效的自然定律。

## 5. 当前最有证据支持的中心假设

目前不应预设最终架构。最值得检验的假设是：

> L4 token hidden state 中包含可预测的 gradient-defined utility，增加独立 teacher-labelled 数据可将 LOCAL-MLP 从 0.8047 提高到约 0.821--0.824；现在最关键的问题不是继续提升 proxy，而是验证这 1.7--1.9 pt 的 teacher-head recall 增益能否转化为 downstream accuracy，并确定强 scorer 是否仍需要 facility coverage。

对应的候选部署流程是：

1. 全部 visual tokens 运行到早期 LLM layer（当前 L4）；
2. 轻量 token-local scorer 预测由离线 gradient teacher 定义的 utility；
3. 用实验选出的 selector 将 1024 token 减到 256；
4. 后续 LLM layers 只运行保留 token，并复用前缀计算。

这只是当前最合理的候选形态，不是已经确定的最终方法。

## 6. 推荐的下一阶段实验

不要继续无目标地做 architecture search，也不要马上扩大到更多 fit data。M1 已完成；下一阶段应进入 M2/M3 method-forming evaluation。

### M1：已完成

M1 已证明 60->240 的收益来自数据而不是 optimizer steps；240->480 仍有均值级真实收益，480->960 未决。详见 §3.8。当前没有理由在做 downstream translation 前继续扩到更大的 fit pool。

### M2：真实 downstream translation

不再仅依赖 `R@8 >= 0.82` gate。至少对以下两项运行同一 held-out split 的 generation：

1. 240-data LOCAL-MLP 三个 seed；
2. 960-data fixed-step LOCAL-MLP 三个 seed。

不能根据 held-out test R@8 挑 seed。优先让三个 seed 全部生成并报告均值、范围和 paired CI；如果计算预算只允许一个 checkpoint，必须在看到 downstream 前预注册纯 validation 选择规则。报告 TextVQA、DocVQA、OCRBench、macro，以及相对 EADP和 n=240 scorer 的 rescue/break counts。只有这里才能判断 proxy 提升是否产生实际价值。

### M3：让数据决定 selector

用 M1/M2 的最佳 scorer，在同一 token budget 256 下比较：

- Top-K；
- block8；
- official facility-location。

比较必须在 240/960 scorer 的同一 seed 协议下进行。先实现真正的 offline selector comparison，再接入现有 EADP-256 generation pipeline；不能把当前 guard-only 的 `m1_downstream.py` 当作已完成 harness。

根据结果决定最终方法：

- **Top-K ≈ facility**：最终方法可以是最简单的 `L4 learned utility + Top-K`，同时删除 EADP 最昂贵的 selector。
- **block8 最好或与 facility 等价**：保留轻量 coverage approximation。
- **facility 显著最好**：保留 EADP selector，将方法创新集中在 gradient-distilled scoring/early pruning。
- **所有 deployable scorer 均不能达到 EADP**：说明需要改变 teacher/objective 或整体 compression mapping，而不是继续增加 LOCAL-MLP 宽度。

## 7. 结果驱动的后续分支

只有在 M1--M3 暴露具体失败模式后，才从相关论文中选择对应灵感：

- **L4 scorer 准确但 early-layer full-token latency 太高**：参考 MAP，将已验证的 L4 utility 再蒸馏到 vision encoder 或 pre-LLM predictor。
- **Gradient rank proxy 提升但 downstream 不提升**：参考 ZOO-Prune 的 perturbation/sensitivity 思路，重新定义更接近答案变化的 teacher；不要继续盲目优化 R@8。
- **Teacher label 太贵、无法扩大数据**：参考 IF-Prune，用较小 VLM 或廉价 teacher 扩大 supervision。
- **Head 聚合掩盖有效信号**：参考 HAWK，研究 head-aware teacher aggregation；前提是先证明 head choice 与 downstream 有关。
- **Token-wise scorer 明确达到上限**：再考虑 MetaCompress 一类直接学习 compression mapping 的路线。
- **Trajectory 类方法**（如 TransPrune、V²Drop）：C6 已给出负证据，除非新的实验表明不同 layer/transition 定义可能改变结论，否则不是当前优先方向。

文献只是候选工具箱，不是待拼装模块列表。正式写论文前应重新核验最新版本、实验设置和准确引用。

## 8. 成功标准与论文边界

### 最低成功

- 在严格相同 held-out split 上，与 EADP accuracy 相当或达到预设 non-inferiority margin；
- 同时在端到端延迟、显存、实现复杂度或跨设置泛化中至少有一项明确更好。

### 理想成功

- 在相同 token budget 下显著超过 EADP；
- 保持相当或更低的端到端成本；
- 对不同 budget、benchmark，最好还有另一模型规模，结论一致。

### 不能提前声称的内容

- C6 已经产生最终方法；
- 0.80 是 L4/local function 的表示上限；
- query 或 trajectory 在所有形式下都无用；
- teacher Top-8 recall 的提升必然带来 accuracy 提升；
- 方法必须由两个模块组成；
- 仅凭当前 Qwen3-VL-8B、三个 benchmark 就已证明普适性。

## 9. 当前操作状态

- 工作目录：`F:\doct\EADP`
- 当前分支：`m1-data-scaling`，跟踪 `origin/m1-data-scaling`；最新提交不在主分支。
- C6 的训练 JSON、audit、代码和文档已从服务器拉取。
- M1 的代码、报告和 JSON 已拉取；服务器上的 checkpoint、feature cache、teacher maps 等大型 gitignored artifact 不在本地 clone 中。
- `s2c6_localization.png` 未被 Git 跟踪，因为 `Qwen_vl/outputs/discovery/*` 默认忽略非 JSON 输出；已可由 audit JSON 重建。
- `s2c6_figure.py` 通过 `s1_audit.OUT` 使用服务器绝对路径，跨机器运行时需要覆盖或改成 repo-relative path。
- 在继续新实验前先确认 Git 状态、服务器 artifact 和可用 teacher cache，不要默认本地包含被 `.gitignore` 排除的模型权重与 feature arrays。

## 10. 新对话的推荐开场方式

用户只需说：

> 请先阅读 `docs/project_handoff.md`，继续 EADP method discovery。

读取后应优先检查最新 Git 提交和服务器新结果，再根据本文第 6--7 节推进；不要要求用户重新复述项目背景。

## 11. 2026-11-08：语义差异挖掘 pilot（已完成）

- 读取完整 handoff 后确认实际 HEAD 为 `a724122d`、分支 `exp/m14-facility-lossless-accel`，保留了原有未跟踪的 `Qwen_vl/scripts/m14/`，在新分支开展实验。
- 使用 E0 修复后的原生 Qwen3-VL-8B、DeepStack 与 3-D mRoPE；1024 输入视觉 token、256 压缩视觉 token。旧 M1/M2 legacy 训练路径未用于新结论。
- DocVQA_VAL 内部按源文档重新划分，仅用于 pilot，不是官方训练/测试结果，也不是未见版式证据。180 对可核验数字编辑候选，首轮筛选120对：101对完整模型两侧正确，其中B2有62对至少一侧错误、18对严格同答案塌缩。原始未编辑24问：完整模型24/24、B2 12/24。
- 三次 carrier 训练检查全部保留：全局池化120步0/8→0/8；加全局低秩读取400步0/8→0/8且8/8塌缩；局部读取+答案序列对比400步0/8→1/8，但保留样本0/32、未编辑页面0/24，载体泛化明显失败。
- 六组对照为ordinary、random、fixed、online、replay、replay_ce，每组400更新/800正答案曝光；关系损失组另有800负答案前向，**不是总计算量匹配**。固定与在线共享初始难例；两种replay使用在线组的精确曝光多重集。六组均为保留样本0/32、未编辑页面0/24。B2同一集合为10/32与12/24；R-res512原生256-token参考为17/32与19/24。全零组间的bootstrap `[0,0]`不是算法等价证据。
- Online 训练后32个候选中的18个初始正确对全部损坏，刷新主要追逐训练引入的失败。这只能支持“当前实现不合格”，不能推翻一般的持续挖掘思想。
- 本轮仅测试数字替换与变化答案监督；未训练无关字段不变性约束，未测试字段交换/求和，未做正式跨版式泛化。不要继续堆新的小型 carrier 后把失败归因于候选算法；下一阶段应先建立能保持真实QA质量的成熟可训练压缩基线。
- 补充不变问题检查8/8对通过完整模型验证，B2双图正确3/8，online为0/8。固定前8个训练原图的事后诊断中ordinary生成8/8正确、teacher-forced NLL≤0.0361，online为0/8；这支持过拟合/学习失败解释，未发现全局训练与生成接口失效的证据。事后诊断未用于选模型。
- 原生零初始化身份、梯度、冻结参数、源文档划分、实际更新次数、相同数据回放及R-res的256-token预算均已核查；所有GPU实验进程已结束。代码位于 `Qwen_vl/scripts/cf_mining/`，README记录重现命令。前两次carrier失败、最终六组原始日志和预测均已保留。最终结论为“盲区挖掘信号成立；当前训练实现不可用；闭环相对静态难例的价值尚未有效验证”。

## 12. 2026-11-08：改进强方法的字段绑定 oracle（已完成）

- 用户选择先从VisionZip/PPE的内容—位置绑定切口入手。新增隔离目录 `Qwen_vl/scripts/binding_oracle/`，没有训练或改基础模型权重，也未开始备用的LayoutLite奖励路线。
- 62候选独立源文档，VAL22+TRAIN40；额外TRAIN排除所有VAL父文档，全部排除前轮CF父文档。完整模型四问验证通过56对（VAL20、TRAIN36），无基于压缩失败的数据筛选。独立像素/QA/源文档审计通过。
- 主配置D205/C51、K256；数值框oracle不是语义字段金标准。anchor覆盖对照与合并约束共享同一dominant/修复后anchor集合，所有主V/三DS使用同聚合矩阵，无静默丢token。PPE是Qwen3原生4slot位置算子适配，未复现作者SFT权重。
- VisionZip及其PPE版18/56；anchor覆盖对照22/56，合并oracle+PPE22/56，随机18/56、空间17/56、R-res25/56。相对VisionZip增加的4个完整正确页面，anchor覆盖对照已全部恢复。更宽OCR行的对照与约束也同为19/50。
- 单问仍有小幅独立收益：相同位置下149→156/224，8修复1破坏；+3.125pp，源文档bootstrap95%CI[+0.45,+6.25]pp。原始真实QA主面板30/48，对照28/48，R-res38/48。必须保留此局部正信号，不能用主指标不变来声称完全无效。
- 机制诊断112图全部完整：52图确实有数值context合并，23图数值token归属改变，PPE双区域位置覆盖105→112/112；约束真实生效，仍未产生新增四问全对页面。PPE代表位置也随簇变化，不能仅归因于内容绑定。
- 历史238/18和DS修复在预定11个有效审计页均4/11；205/51在这小面板反而2/11，不能声称当前配置已是最佳VisionZip。原生接口gates通过不等于复现原论文全部训练/模型设定。
- 预注册判定STOP_THIS_MECHANISM：不启动新边界预测器/合并模块训练；结论仅适用于这版region oracle/native配置。完整证据、所有预测、几何纠错及局限见 `docs/binding_oracle_results.md`、`Qwen_vl/outputs/binding_oracle/analysis.md`。

## 13. 2026-11-08：LayoutLite局部奖励排序诊断（已完成）

- 用户要求在强方法上继续找方向并实际尝试。选择检验LayoutLite整页OCR一致性是否掩盖关键数字遗漏，先做无训练动作排序诊断，未另造carrier。
- 官方FireRed-OCR基于Qwen3-VL-2B，revision `7bf5e9d91d3bb6bf91a3b6bc66d163096b696e50`，已下载到 `/media/disk2/YZX/research/model/FireRed-OCR`。官方LayoutLite源码固定 `7e398a2`，head SHA256 `92e25d31ffae823b04b2b97be2259446b656afb81eb67f959fad16ac7d07e73d`。5.6源码的评分头/提示词通过隔离适配在本地Transformers4.57.6原生网络运行，未复制decoder。
- 32主页面来自新的TRAIN父文档，80个人工数值QA；另16候补未用。排除4957个父文档（VAL、前两轮、历史fit roster合并），只按已有注释/OCR银标定位/40–450词筛选，不按模型失败挑页。原始全页OCR不是人工gold。32页所有source QA共221个，也仅作为未过滤字符串覆盖辅助统计。
- 每页完整OCR+5个同K方案：Top-K与4个固定seed Gumbel-Top-K，K=round(.5N)，全部保留原生V/3DS及mRoPE索引。这个动作池不等于原方法全局平均预算的两簇阈值推理，也不等于项目8B1024→256。全模型普通QA smoke0/6，仍输出转写，故未启用真实QA终点。
- 192输出完整、无重复或待补。2页完整参考截断→30有效页；全32页full覆盖72/80个selected numeric答案，有效30页提供70个teacher可覆盖目标。压缩输出7条截断，主分析按预注册保留，并做全未截断敏感性。
- 近整页奖励相同（差≤.01）且独立答案覆盖不同的主比较仅5文档/10对，未达12/24门槛，HOLD_INSUFFICIENT_INFORMATION。整页/归一化整页/普通片段均值/长度加权/tail20的文档宏排序一致率为90/85/80/85/50%；tail20对均值−30pp，95%CI[−80,+20]。所有分歧辅助10文档/46对：整页81.7%、均值78.3%、tail73.3%，tail相对均值−5pp，CI[−17.5,+5]。
- 离线按奖励在同5个候选中择优：整页及两种均值页面宏覆盖98.33%，tail96.67%，差−1.67pp、CI[−5,0]；不是部署或训练收益。固定TopK已保留68/70答案，当前50%压缩的字面覆盖接近天花板，遗漏信息量有限。32/128字符分片、反向编辑路径及全未截断敏感性均未给出正向点估计；不能把小样本点估计当确定负效应。
- 事后反例ynxh0227_4原图CalciumCarbonate=181.1：tail选出的方案漏掉右栏数值，tail分反而更高；整页选出的方案保留181.1。另固定5页视觉/文本复核显示局部编辑损失会混入HTML重组、teacher重复去除、teacher错字纠正等，不能把每个字符差异都解释为真实遗漏。反例没有用于改规则或排除样本。
- 原生全保留V/三DS/prefill/next-decode逐位一致；stock8token生成和五mask batch16token门禁通过。串行吞吐预跑已完整归档并作废，正式32页统一full串行+五方案batch5，耗时1298秒。2527项最终审计全部通过；审计脚本一次参数名冲突修复，firstfail单独保留，未改模型输出或冻结配置。GPU实验进程已全部结束。
- 决定暂停这版tail20训练，不继续调尾部比例或放宽本轮信息门槛。一般局部证据保全、强压缩以及真实QA仍未被有效验证。详见 `docs/local_reward_diagnostic_results.md`、`Qwen_vl/outputs/local_reward_diagnostic/analysis.json`、`audit.json`、`manual_alignment_audit.json`、`counterexample.json`；复现入口 `Qwen_vl/scripts/local_reward_diagnostic/README.md`。
