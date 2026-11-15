# R-res与冻结学生：剩余本地数据集扩评（2026-11-13）

用户授权把R-res和刚完成的学生方法在其他数据集上全部运行。范围是本地已有、去除同集_local别名后的10个新集合：RealWorldQA765、ChartQA_TEST2500、InfoVQA_VAL2801、AI2D_TEST3088、MMStar1500、HallusionBench1129、MME2374、POPE5127、MMBench_DEV_EN_V11和CN_V11各4876，共29036题。旧DocVQA/TextVQA/OCRBench已有完整冻结评估，不重复运行。

## 方法与输入

7臂：full1024、原生EADP facility B1、block8 B2、R-res512、源run_20261111的bag_rank seed11/23/37，共203252条QA。学生仅在原DocVQA TRAIN训练，保持源validation选择的checkpoint；无新训练、teacher标签、目标学习参数适配或测试选seed。R-res是同一冻结Qwen3-VL-8B的512固定分辨率原生256-token输入，不调用学生或高分辨率编码。其他压缩臂1024输入、保留256；全部保留原生V、三条DeepStack对齐及三维位置。

提示使用本地Qwen3官方适配：MCQ含原hint、问题及全部非缺失选项，要求选项字母；Y/N要求yes/no；VQA沿用原简短回答suffix。学生query embedding来自同一个完整公开提示（含选项），add_special_tokens=True，不截断。新任务的长提示统一上限4096；仅对无位置编码的bag评分器提高原256长度保护上限，不改变任何学习参数或原256长度position buffer，短输入选集须逐seed通过等价门禁；这属于冻结权重的输入适配，不声称训练时见过长MCQ。

HallusionBench的178条原始无图控制保留，原生text-only、N=K=0，DS关闭；7臂均正常执行相同文本路径，学生选择器不参与。1129题报告称为完整原表扩展评测；951题有图子集对应VLMEvalKit默认skip_noimg图模式，分别报告原生aAcc/fAcc/qAcc，不能把扩展分母冒充常规图模式分数；文本控制不得算作剪枝收益。MMStar有2条原始正确选项文本缺失的题，保持题和答案，公共选项按VLMEvalKit默认NaN语义省略空值，官方exact_matching失败记0，主分母不删，另报annotation_clean敏感性。

公共选项和hint严格按本地pandas 2.3.3默认NA语义读入。冻结前输入审计发现AI2D有85题的错误干扰选项字符串为NA/None，已按默认读取行为省略；原题、答案、图片及全部题目不变，原始元数据与正规化前manifest保留。全10集逐行交叉核题号、问题、答案、选项、hint通过，记录pandas_metadata_audit.json，正规化没有参考目标模型输出。

输入白名单只有图像、题号、数据集名、问题、选项、hint及has_image；原答案、评分元数据、历史预测和污染标记不进入选择器或生成。原生greedy/EOS，QA最多128新token；截断保留。全部原题和答案、图片引用链、PIL默认物化、输入RGB/字节hash及源TSV冻结；与当前fit+val做精确/启发式近重复审计，主全量及去重子集分开。历史曝光未知不等于未见，基础模型预训练曝光未知。

## 评分与效率

使用锁定本地VLMEvalKit实现：MCQ deterministic exact_matching（不能解析记错，不随机猜、不调用外部LLM裁判）；MMBench circular grouping各旋转全对才得分，逐行accuracy只是辅助。ChartQA relaxed accuracy、InfoVQA ANLS；POPE F1/accuracy/precision/recall，保留原category拆分加权；MME类别acc+成对acc+并给原生总分；Hallusion aAcc/fAcc/qAcc。表格明确指标量纲，不把MME点数和百分制直接宏平均。原指标函数入口与边界行为需CPU对照测试。

每集在无精确训练重合、有图样本中按图像SHA排序预固定8张不同图，3配对block、每臂每图每cap预热2次，cap1/16，共3360正式计时。每次fresh处理图像、视觉编码、选择和生成；停止在decoder同步完成后、位置/hash审计前，模型加载不计。仅计时忽略EOS强制固定token数，QA保持EOS。按源图像像素/父文档聚合配对区间，不能把同图多问或重复计时当独立样本；MME有10张源图跨类别重复，置信区间合并相同源像素，官方两题分组仍按原category+image。单A40单请求，不声称服务吞吐。

CPU测试、全元数据/提示长度检查和新源码/数据/checkpoint冻结完成后，GPU先检查各任务官方prompt、stock全保留identity、共享/新鲜视觉与各臂生成相同、DS/位置/预算、无图控制、原短问题student长度适配等价。通过后先计时、再全部QA、最后独立审计/原生聚合报告。逐条SQLite提交并支持同冻结续跑；遇异常记录退出，不静默少报分母。冻结后修改源码/协议必须新run目录。

## 研究解释

本轮用于定位R-res与学生各自的优劣与失效范围，没有新增路由或修复算法。不能按测试数据集名称或答案挑赢家再当统一方法。后续若研究低分辨率失败修复/逐题路由，必须独立训练与开发数据选规则，测试冻结；决策仅用上线可得输入，若计算高分辨率特征才能路由，则计入全部成本。答案依赖的逐题择优只能是离线诊断上限。最终方法是否组合、如何组合，依实验决定。

启动：`CUDA_VISIBLE_DEVICES=0 bash Qwen_vl/scripts/student_broad/run.sh`；默认输出 `Qwen_vl/outputs/student_broad/run_20261113/`，状态 `status.json`，日志 `run.log`，最终 `evaluation/report.md/json`。
