# 前向学生蒸馏：实验记录

**状态：训练、32 个独立测试文档上的 640 条回答及单开发页计时已完成。** magnitude V-only 教师训练的三个前向学生分别答对 31、31、32 题，B2 为 22 题、R-res512 为 19 题。三个学生均输入四路视觉特征和问题，V-only 指离线教师标签；结果支持在这个小面板继续验证，尚不能推断跨任务泛化或官方基准收益。预注册见 [student_distill_prereg.md](student_distill_prereg.md)。

## 研究问题与实验设置

检验学生能否只用图片和问题，在前向推理时预测由答案条件梯度定义的视觉 token 排序。主模型为原生 Qwen3-VL-8B-Instruct，bf16、SDPA，1024 个 merged visual tokens 中保留 K=256；每个保留索引同时作用于主视觉特征 V 和三路 DeepStack，并保留原始 3-D mRoPE 位置。模型快照 revision 为 `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`。

四个教师只在离线阶段使用题目原始首个答案及完整模型的 answer-only NLL backward。对每一路特征 `x_s`，`magnitude_s = Σ_d |∂L/∂x_s,d · x_s,d|`，`signed_s = −Σ_d (∂L/∂x_s,d · x_s,d)`；V-only 取 V 的分数，joint 对 V 和三路 DeepStack 的原始分数直接求和。梯度是局部近似，不是大量 token 同时删除的精确影响。

四组学生始终使用同样的 V、三路 DeepStack 和当前问题文本原生 embedding 均值，训练配置也相同，仅教师标签不同：`magnitude_v`、`magnitude_joint`、`signed_v`、`signed_joint`。因此这一设计检验教师标签差异，**不单独识别学生输入中 DeepStack 的贡献**。部署评分器不接收答案、教师图或 backward；梯度教师自身不是部署方法，也不是严格准确率上界。

## 数据与训练

- 使用 DocVQA TRAIN 中互不重叠的 **120 个 fit、16 个 validation、32 个 test 和 1 个 development 父文档**，每页一个原始数值问答。采样由注释及预先固定的哈希顺序决定，没有按模型是否答对或压缩是否失败筛选。
- 从候选中排除项目先前使用的父文档集合（并集 5,071 个），同时检查历史页面的图像哈希。169 条所选原始题目、答案别名、父文档与图像字节均有独立回查记录；训练与校验、测试和开发按父文档及图像哈希不重叠。这不证明模型预训练时没有见过这些文档。
- 所有学生采用相同的 token-local 64 隐层评分器，参数量 **1,311,105**。每路 token 特征逐 token 做 RMS 归一化，保留四个 log RMS；问题向量同样做 RMS 归一化。输出在原空间稳定排序，取 Top256 后恢复 raster 顺序。
- 每个教师分别训练 seed **11、23、37**。训练目标为 Top256 membership 的 BCE（正类权重 3），加上 0.5 倍 Top32 对排名 257–1024 的 margin-1 排序损失，Top32 按 `1/(rank+1)` 加权并归一化。AdamW 学习率 0.001、weight decay 0.0001，batch 8，**每组 900 次更新**；同 seed 各教师的初始化及数据顺序一致。
- 每 30 步只看独立 validation 的对应教师 Top8 recall@256；选择最高点，平分取最早。全部 12 个 checkpoint 都应进入同一测试面板，不以测试结果挑种子或教师。

下表是**用于选 checkpoint 的 validation proxy，不是测试准确率**；不同教师的 Top8 是不同参照集合，不能把它们的数值高低解释为下游优劣。

| 教师标签 | seed | 所选更新步 | validation teacher Top8 recall@256 |
| --- | ---: | ---: | ---: |
| magnitude_v | 11 | 390 | 0.8672 |
| magnitude_v | 23 | 420 | 0.8672 |
| magnitude_v | 37 | 420 | 0.8594 |
| magnitude_joint | 11 | 600 | 0.7891 |
| magnitude_joint | 23 | 660 | 0.8125 |
| magnitude_joint | 37 | 810 | 0.8203 |
| signed_v | 11 | 60 | 0.7734 |
| signed_v | 23 | 60 | 0.7422 |
| signed_v | 37 | 60 | 0.7656 |
| signed_joint | 11 | 60 | 0.6797 |
| signed_joint | 23 | 30 | 0.6719 |
| signed_joint | 37 | 30 | 0.6875 |

## 测试问答

全部 **32×20 条真实生成**完整且无截断。主指标是预注册的 numeric exact，容许标准化货币符号、空白与数值格式；辅助报告 DocVQA ANLS 和原始别名 literal exact。B2 是本项目的 **EADP 分数 + block8 selector** 原生 Qwen3 适配，不能称为官方 facility 选择器。

| 前向参考 | numeric exact | ANLS | literal exact |
| --- | ---: | ---: | ---: |
| 完整 1024 tokens | 32/32 | 0.9965 | 31/32 |
| B2，K256 | 22/32 | 0.7958 | 21/32 |
| R-res512，原生 256 tokens | 19/32 | 0.8387 | 19/32 |
| 固定随机，K256 | 15/32 | 0.6524 | 15/32 |

| 学生教师标签 | seed 11 | seed 23 | seed 37 | 三 seed 平均 |
| --- | ---: | ---: | ---: | ---: |
| magnitude V-only | 31/32 | 31/32 | 32/32 | 97.92% |
| magnitude joint | 29/32 | 32/32 | 29/32 | 93.75% |
| signed V-only | 29/32 | 29/32 | 28/32 | 89.58% |
| signed joint | 30/32 | 25/32 | 29/32 | 87.50% |

这四行的学生全部读取 V、三路 DeepStack 和问题，只有训练用教师的分数定义不同。对于 magnitude V-only 学生，seed 11/23/37 的 ANLS 分别为 **0.9840/0.9653/0.9965**，literal exact 分别为 **30/30/31 题**。主指标与 literal 的差异来自一题输出 `$750.00`，原始别名含 `750.00` 和 `$ 750.00`：numeric 标准化后匹配，literal 严格字符串不匹配。

以下每行在相同 32 个父文档上配对，使用 10,000 次共享父文档 bootstrap；区间单位为百分点。三个 seed 都报告，没有从测试集中挑选。

| magnitude V-only 学生 − 对照 | seed | 修复 / 破坏 | 差值 | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| 相对 B2 | 11 | 10 / 1 | +28.125 | [+12.5, +46.875] |
| 相对 B2 | 23 | 9 / 0 | +28.125 | [+12.5, +43.75] |
| 相对 B2 | 37 | 10 / 0 | +31.250 | [+15.625, +46.875] |
| 相对 R-res512 | 11 | 13 / 1 | +37.500 | [+18.75, +56.25] |
| 相对 R-res512 | 23 | 13 / 1 | +37.500 | [+18.75, +56.25] |
| 相对 R-res512 | 37 | 13 / 0 | +40.625 | [+25, +59.375] |

同 seed 下 magnitude joint 教师学生减 magnitude V-only 教师学生，差值分别为 **−6.25、+3.125、−9.375** 个百分点，区间分别为 **[−18.75,+6.25]、[0,+9.375]、[−18.75,0]**。这一面板未显示 joint 教师稳定优于 V-only 教师；不能从区间包含或触及零推断等价，也不能据此判断学生输入侧 DeepStack 是否有益。比较较多、样本较小，区间应按探索性结果解释。

四个答案知情教师选集的正确数分别为 magnitude V-only **32/32**、magnitude joint **32/32**、signed V-only **27/32**、signed joint **26/32**。它们在冻结学生后使用标准答案和梯度，只作离线诊断，不能作为前向方法效果或严格准确率上界。

## 成本

独立 development 页面、A40、bf16/SDPA、batch 1：每臂 3 次预热及 10 个轮换顺序的配对 block，fresh 预处理与视觉编码，不含模型加载；表中是中位数，16-token 总时间强制忽略 EOS。

| 方法 | 首 token ms | 16-token 总时间 ms | 问题编码 + 选择 ms |
| --- | ---: | ---: | ---: |
| 完整模型 | 488.47 | 1023.41 | — |
| R-res512 | 214.42 | 744.74 | — |
| B2 | 359.44 | 889.61 | 9.64 |
| 随机 | 355.43 | 890.28 | 0.19 |
| magnitude V-only 学生，seed 11 | 359.48 | 892.73 | 2.11 |
| magnitude joint 学生，seed 11 | 359.63 | 895.29 | 2.15 |
| signed V-only 学生，seed 11 | 363.36 | 901.42 | 2.16 |
| signed joint 学生，seed 11 | 358.62 | 887.86 | 2.11 |

该页上 magnitude V-only 学生与 B2 的首 token 中位时间几乎相同，R-res512 明显更快。所有臂计时时 **12 个学生均驻留显存**，仅相应学生臂运行评分器；不能将这些峰值直接当作单 checkpoint 生产部署的内存。单页面不代表跨数据集时延分布。

离线部分单独计算：136 题 fit/validation 特征与教师缓存过程记录总耗时 **129.64 秒**，其中教师标签步骤累计 **75.95 秒**；缓存共约 **4.26 GiB**。12 次学生训练的记录耗时合计 **160.67 秒**，另有特征预载 **1.74 秒**。这些成本不包含在前向部署计时中。

## 门禁和审计

- 原生模型与离线教师门禁 **31/31 通过**，涵盖 stock 完整视觉/DeepStack 特征和 prefill/forced decode 对齐、位置、四路梯度、共同选择及缓存检查。详见 `Qwen_vl/outputs/student_distill/gates.json`。
- 学生推理门禁 **52/52 通过**：在一张 fit 页面上检查缓存与实时 bf16 输入一致、只修改答案不改变特征与问题表示，并对全部 12 个 checkpoint 检查缓存/实时分数和选集一致、改答案后分数和选集不变。该门禁覆盖的是一张页面，不等于跨页面的准确率评估。
- 只读 CPU 独立审计验证了训练 freeze、门禁、冻结源码、fit/val manifest、cache index、全部 136 个 fit/val 缓存和 12 个 checkpoint 的哈希链；逐组重算 30 个 validation 时点的每页均值，所选步均为最高值且同分最早。记录为每组 900 次更新；未保存逐步优化器轨迹，不能声称已独立重放每一步更新。
- 独立复算全部 136 个 fit/val 教师缓存：四种教师在两个划分中，全相等分数图、Top32 边界并列和 Top256 边界并列均为 **0**。这不排除其他排名处的并列或梯度近零。
- 另对前 5 个完整 fit 缓存，独立核验文件/源图和四流输入字节哈希、问题向量、原始答案经冻结 tokenizer 编码后的 token IDs；将教师/答案对象替换为不可读取对象后，学生前向输出不变。合成数据上验证了四流汇总函数的 V 和 raw joint 定义。
- **审计边界：** fit/val 缓存保留了四张汇总教师图，但没有保留每一路逐 token 梯度图；因此无法仅从这些缓存逐样本数值重算四路和。真实样本的汇总定义由冻结源码、门禁及部分必要条件支持，不应描述为逐样本独立重算通过。
- 测试结果独立 CPU 审计重算所有 640 个数值评分、ANLS、literal exact 与 token 解码；检查 12 组学生的保存分数确实对应稳定 Top256、四个教师的逐流分数和 joint 汇总及选集、全部视觉/DeepStack/cache 长度与随机选集。用冻结 tokenizer、原生 `get_rope_index` 及对应图像网格重建全部 **640 个位置张量的 SHA**，均与实际记录相同。
- 384 个学生选集没有一个与其对应的答案知情教师选集完全相同；平均交集约 87–136/256。测试时学生源码只接收图像和问题。32 个问题中原始答案别名未以完整词边界字面出现在问题中，此检查不能排除语义暗示或模型预训练见过材料。测试页未保存供学生使用的完整四流输入，因此不能仅凭测试产物重新计算全部学生分数。

## 局限与产物

这是较小的 DocVQA TRAIN 数值问答探索；测试集只有 32 个独立父文档、比较较多。validation proxy 被用于选择 checkpoint，其结果不能替代真实问答，更不能证明跨任务泛化。当前特征输入相同，教师比较不说明四路学生输入本身带来的增益；结果应结合同面板强参考与端到端成本判断。

主要产物：

- 数据、分组和独立回查：`Qwen_vl/outputs/student_distill/data/`。
- 预教师冻结、fit/val 缓存及索引：`Qwen_vl/outputs/student_distill/train_freeze.json`、`cache/`、`cache_index.json`。
- 训练记录和权重：`Qwen_vl/outputs/student_distill/checkpoints/train_complete.json` 与同目录 12 个 checkpoint。
- 预测试冻结、推理门禁、逐题测试结果、分析及计时：`Qwen_vl/outputs/student_distill/eval_freeze.json`、`student_inference_gates.json`、`test_predictions.json`、`test_analysis.json`、`latency.json`。
- 可复现入口：`Qwen_vl/scripts/student_distill/data.py`、`audit_data.py`、`extract_features.py`、`train_student.py`、`student_eval.py`、`analyze_student.py`、`student_benchmark.py`。训练和测试冻结文件使用只写一次保护；复跑应使用独立输出目录，不能覆盖既有实验。
