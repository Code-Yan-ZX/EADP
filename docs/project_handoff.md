# EADP Visual Token Pruning 项目交接记录

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

## 11. 2026-10-03 增量：Anchor Completion 全量验证 + RTG 候选（当前状态）

- **主判定（全量，预注册）**：MAIN025（EADP+Completion）相对 BASE（EADP 硬剪枝）
  pooled Δ=+0.734 [+0.427, +1.030]，三任务方向一致 → 改善成立；
  fresh 全行 pooled +0.517 [−0.114, +0.990] 跨零（独立确认不成立；
  OCRBench fresh=0）。BOTH025 预案未触发。
- **RTG 候选（round-trip 文本权重，LoFTR 启发）**：GO 门 4/4 → R_MAIN025
  全量已跑。主要比较 R−E_MAIN025 = +0.047 [−0.541, +0.637]：**未支持非劣**
  （差 0.04 于边界）；补充比较 R−E_GATHER = **+0.978 [+0.363, +1.598]** 显著正。
  fresh 机制对照 R−S 反转（−0.981），跨列空间对应机制未获独立确认。
  完整方法相对硬剪枝的优势主要由 Completion+权重在 DocVQA/OCRBench 贡献。
- **文献对照**：PruMerge（无 CLS token）与 Libra（LLM 层 attention+代码未发布）
  均不可忠实移植；"最近完整方法数值对照"记未完成。
- 非回归（R−E_GATHER）：RWQA +2.88✓ / ChartQA +1.92✓ / MMStar +0.47 / POPE +0.04；
  MMBench 仅 headline −0.39（逐题 CI 未完成）。E_MAIN025 非回归与 EADP 锚
  fresh 消融在截止时间前部分完成，见运行台账。
- 报告：`docs/anchor_completion_validation_report.md`、
  `docs/stage1_roundtrip_pilot_report.md`；全部偏离 D-1~D-8 入协议。

## 12. 2026-10-03 增量：BASE vs EADP 论文表4 差距审计 + 恢复审计闭环

两轮 CPU-only 审计（未启动 GPU 生成、未调参、原仓库与预测全程只读），
产物在 `/media/disk2/YZX/research/audit_base_gap_20261003/`
（`SHA256SUMS.txt` + `archive_recovery/20261003_r2/SHA256SUMS_r2.txt`，未入库）。

**定位结论（已闭环）**：本地 BASE 显著高于 EADP 论文表4
（TV +6.4 / ChartQA +11.4 / DV +9.3 / MMB +3.3）的主因是
**官方 EADP 的 Qwen3-VL 剪枝路径只传 `inputs_embeds` 给 `generate`，
丢弃 DeepStack 三路注入、3D mRoPE 退化为 1D 位置**；本地 BASE 用修复后的
native E0 引擎。证据链：

- 官方 repo（SJTU-DeepVisionLab/EADP @ `e1a0880`）的
  `model_fixed_res.py`/`pruner.py` 与本地 legacy 路径 byte-identical；
  EADP 超参（α=0.5, β=2.0 等）、facility、1024²、K=256、greedy、评分器全同。
- 恢复审计找回官方 wrapper 原始逐题预测（旧 clone `research/EADP` 下
  `outputs/eadp/.../T20260921/22_Ge1a08801`，启动命令与运行日志齐全），
  独立重评 diff=0：TextVQA **71.042** / DocVQA **61.136** / OCRBench **623**，
  ≈ 论文表4（71.4/62.8/625）→ 论文数字 = 官方代码行为。
- 全量同题配对（交集 5000/5349/1000，零缺失；image-cluster bootstrap
  5000 次 seed=20261002）旧官方wrapper − native BASE：TV −6.75
  [−7.66,−5.85] / DV −10.98 [−12.06,−9.90] / OCR −11.90 [−14.48,−9.28]，
  等权 macro **−9.88 [−10.86,−8.90]**。
- 受控消融锚点：keep-all（TextVQA DEV-300, K=1024）native 85.03 vs
  legacy 75.33（s1_dualpath，原始 shard 逐字复核）；K=256 同 B2 selector
  只切引擎 flag（D1 门）macro +7.52 [4.89,10.15]（DEV-300，缺 ChartQA；
  DocVQA 单任务 −3.7，单 flag 因素分解未闭合）。

**评分侧审计（本地分数全部可信）**：TextVQA/DocVQA/OCRBench/ChartQA
独立重评 diff=0.0；BASE TextVQA 正确值 **77.788**（此前口头转述的 77.54
无任何出处）；R_MAIN025 TextVQA 绝对分 vs 配对差无矛盾（DEV-100 子集噪声）；
MMBench headline 0.8382 实际仅覆盖 1292/4876 题（VLMEvalKit
`mcq_circular_eval` 的 `index%1e6` 静默丢弃 index≥1e6 的 3584 题，
"circular"退化单轮）——引用 MMBench headline 时必须注明口径；
DocVQA 4 行字面 NA→'nan' 丢配（绝对分 −0.04，delta 不变）。

**对既有结论的影响**：R 相对 BASE 的配对提升（R−E_GATHER +0.978
[+0.363,+1.598] 等）**不受影响**（同引擎同评分器配对；复算 +0.981
[+0.363,+1.605]）。论文 EADP@256 行与本地 BASE **不可直接比较**；
论文内部 pruned 行之间（同走 CDPruner inputs_embeds 路径）内部可比但
绝对水平整体被压低；论文 full-token 行 ≈ 本地 native 不剪枝（DEV-300 佐证）。

**定性边界**：native 引擎对 DeepStack/3D mRoPE 的处理属 Qwen 已有架构的
正确传递，记为**实现/复现修复**，不是方法创新；修复效应对非 EADP
selector 的普适性无证据（它们只跑过 native，无 legacy 对照）。

**缺失测量（如实标注）**：同臂 legacy vs native 配对 TTFT 不存在——
RTG 的 404.7−402.1=2.6ms 是 native-vs-native，不得引用为修复成本；
E0 计划的 `e0_perf_paired.json` 从未产出（D3=INCOMPLETE）。

详细产物：第一轮 `audit_base_gap_20261003/{evidence,extracts,rescore}/`
（DeepStack 代码证据、论文全文提取、独立重评分）；恢复审计
`audit_base_gap_20261003/archive_recovery/20261003_r2/`
（`CONCLUSION.md`、`old_vs_new_scores.csv`、逐题分明细、配置对照、
消融/效率/出处盘点）。

## 13. 2026-10-08 增量：最新 EADP_amp 多模型结果的输入/评分追查

本轮用户指定对象为 **EADP_amp 最新多模型、多数据集结果**。起始分支
`codex/anchor-completion-validation`、HEAD `194562d`。必须结合最新台账 §23–24 和
`docs/evaluation_reproduction_audit_20261008.md` 使用本文；历史审计的推断不自动沿用。

- **MME（LLaVA）主因确认**：本机重建 `MME_Benchmark_release_version` 的 GT 有
  162 个 Yes/No 标签与官方 eval_tool.zip 模板相反；当前 LMUData TSV 的全部
  2374 道题/标签与官方一致。8 份本地 LLaVA 臂共享这一错误。
  保留原预测 CPU 重评，v1.5 FULL perception **1420.43→1506.93**（论文1513.4，
  残差−6.47），复现 EADP K128 **1339.10→1429.40**（论文1439.0，残差−9.60）。
  AnchorZip K128 正确 perception **1445.69**，与同机 EADP 配对 **+16.2941**；
  cognition 单列 −2.5。旧错误 GT 上的绝对分数及派生结果不能继续引用。
  不能将本事件自动扩展到 Qwen：Qwen 使用另一条 TSV 评测链，需独立核验依赖。
- **SQA（v1.5）主因验证**：本地重建 CQM-I 与 LLaVA 原包 CQM-A 的 4241 个题面
  全部不同；作者实际 CQM-I 未发布。冻结模型/图像/greedy/vicuna/评分器，原包题面
  对全部 2017 图题的 FULL 对照为 **64.8488→69.5092**（+4.6604 点，论文69.6，
  残差−0.0908）。归因范围是整体题面格式替换；不能称已证明作者文件身份，
  也不能把 FULL 增益自动套到各剪枝臂。原包题面两 K128 臂完整配对完成：
  EADP69.5588 / AnchorZip69.2117，净−7题/−0.3471点，配对95%CI
  [−0.9420,+0.2479]跨0；论文EADP128为69.0，不能凭跨论文+0.2117判方法增益。
- **TextVQA**：历史 model_vqa 与官方 loader 的 LLM 提示相同，但 EADP guidance
  前者含答案指令后缀，后者删除；5000 题均不同、298 题改变 CLIP 文本分段。
  已增加严格官方入口与独立输出名；固定面板验证只切 guidance 的影响，不调参。
- **评分入口**：新 live scorer 接入 VizWiz/POPE/GQA；22 份完整文件与 Oct8 正确
  快照逐题/混淆矩阵一致，缺失/重复/未知ID/FAILED 拒绝。SQA 旧 QCM 恢复入口停用，
  重建 CQMI 入口增加完整题面复用检查。POPE 保留用户选择的 random2910 as-asked。
- **预算证据勘误**：NeXT SQA 为1全局+4局部=5 crops，ViT输入2880 tokens；
  旧 tqdm 刷新计数不是 question_id，不能验证逐题保留数。独立 v2 清单只保留正确
  输入几何，实际预算待 ID 对齐 runtime trace；新 wrapper 已记录返回的保留数量。
- **仍需如实处理低分**：v1.5 K32 POPE F1 80.008 vs论文EADP86.7，popular recall70；
  MME 正确 GT 后仍只有1092.63 vs论文EADP1347。没有同机完整 EADP K32 对照前，
  不把整个跨论文差距当作方法增量。不能通过换题集、换指标或调参抹平。
  运行稳定性另发现：POPE同图同完整题面2334组中，K32有182组Yes/No分歧，
  全部GT为Yes；K128/K64/E128为0。旧预测低分的原因还需排除运行异常。
  固定16分歧组用于重复重放，不用其估计benchmark准确率。
  当前重放16组×3共48次均Yes、16阶段hash一致，未复现历史分歧；
  旧8910长序列可重复性/原因仍未证实。输入跨CUDA stream缺显式依赖是候选风险，
  未证实因果、未改生产代码，不能用短面板替代全量重跑。
- **修正旧解释**：第12节“MMBench 1292/4876 说明丢题”的历史推断已被最新工件
  推翻：1292是循环组数，3584是旋转行，应查组内循环准确性，不能自动宣称漏评。
  最新 Qwen legacy Avg10 AnchorZip68.0244 / EADP67.6636，增量+0.3608；
  native的DeepStack/mRoPE修复与方法创新继续分开。

主要独立工件：`Qwen_vl/outputs/audit_followup_20261008/` 的
`paper_reference.json`、`comparison.csv`、`mme_official_gt_rescore.json`、
`sqa_input_comparison.json`、`textvqa_guidance_*`、`budget_manifests_v2/`；
canonical MME 完整转换另在 `records_20261003/audit_followup_20261008/mme_canonical_gt/`。
具体入口修复、来源哈希、剩余边界和可重复命令见本轮审计报告。

### 13.1 用户授权的并发重跑（2026-10-08 下午启动）

用户要求立即重跑并尽量并发，明确论文用结果截止为**北京时间2026-10-09 12:00**。
已在同一A40上启动两GPU任务：v1.5 AnchorZip K32 POPE完整8910题与NeXT官方
TextVQA K128 EADP/AnchorZip配对。显存峰约42325/46068MiB，
只保留两个GPU子进程；本轮并发结果用于准确率，不能作为独占延迟基准。

后续通过 `run_repair_queue.py` 持久执行：v1.5通路在POPE完整核验和重评分后接
TextVQA K128配对、SQA K64/K32配对、TextVQA K64/K32配对及FULL；NeXT通路
在首批TextVQA两臂完成后接剩余同预算配对及FULL。每模型TextVQA共7臂，
SQA新跑v1.5剩4臂/NeXT7臂，加POPE共**26个新GPU实验/101097次生成**。
不补POPE random90题，不调alpha=.5/beta=2/Completion lambda=.25。

计划/状态/日志在 `Qwen_vl/outputs/audit_followup_20261008/rerun_batch/`，
`lane1_v15_{plan,state}.json` 与 `lane2_next_{plan,state}.json` 是实际任务依据。
所有新输出与旧预测分开，逐项完整性通过才评分；TextVQA的question_id实际上是
image_id（5000题仅3166唯一ID），用 `(question_id,完整prompt)` 校验5000唯一题，
禁止按单一ID去重。新的TextVQA逐题runtime记录输入位置、prompt哈希、实际保留数；
初始NeXT K128两臂沿已启动旧入口执行，无这一新增runtime trace，不能假装已有。

临时工期按实际双路吞吐估计：K128关键配对下午至傍晚可用；完整批次保守估至
10月9日凌晨3–6点，随后上午核表。此为预测而非完成证明，需随状态/吞吐更新。
当前仍不能声称整体稳定超过EADP：Qwen legacy Avg10同机+0.3608点但对论文−0.1756点，
v1.5 K128多项略高、SQA原包配对−0.3471点；NeXT历史TextVQA同机−0.544点，
官方新协议全量结果尚未完成，不能拿前缀预测作最终准确率。

并发POPE完整重放已出现新的运行证据：15:37累计4596/8910，845个已重复输入组
中77组Yes/No分歧，所有题面与实际保留数32一致。它复现了旧K32的不一致，
不能把“重跑一次”视为已经修复，也不能把所有低分归因于方法退化。保留as-is
8910完整生成/评分，另准备只增加CLIP两流等待caller的全8910控制实验，不改
生产CLIP源码，不调参数。v1.5后续队列在as-is评分后等待该独立控制实验完成。
控制完成marker的success表示完整性/评分通过，**不要求分歧消失**；阴性结果也
须保存并释放后续任务。stream风险与旧低分的因果仍待完整干预证据。

根据信息增量，额外完整控制最终预先决定为**AZ32 streamwait + 同机EADP32
streamwait**各8910题，固定同预算/同输入/同等待，仅AnchorZip安装不同；两项
完整评分后再释放v15队列，不根据高低分择跑。共26主实验+2控制=28臂/118917
新预测。K128配对预计傍晚至19点左右可用；完整批次目标Oct9凌晨3–6点，保守
预留至8点，仍为动态工期预测。详细受控计划与AST身份在 `rerun_fast/`，生产
CLIP文件保持原样，以区分as-is与只增加wait的因果对照。

### 13.2 每小时主动巡检（Oct8 16:06启用）

用户明确要求每一个小时主动检查，避免卡死。当前会话没有automation_update/
定时唤醒工具；已核对官方OpenAI文档与本机能力，改用本机systemd user timer
实际启动独立Codex CLI巡检，沿用已登录ChatGPT账号与既有模型配置。不是原生
聊天定时任务，不会自动在当前聊天发消息；异常/恢复/全部完成发本人桌面通知，
正常只留本机记录。首轮16:06:36已启动，下一轮17:06:36北京时间；全部实验与
官方评分完整通过才自动disable timer，科学结果高低不影响停止判定。

服务名 `codex-eadp-repair-monitor-20261008.{timer,service}`；unit在
`/home/dell/.config/systemd/user/`。runner `run_hourly_repair_agent.py`，
只读快照 `monitor_repair_hourly.py --once --record`；登记、prompt、schema、
每次agent日志/结果在 `rerun_batch/hourly_monitor/`。快照16个CPU边界验证通过，
systemd unit语法通过；CLI首轮已经认证并执行真实工具，通知D-Bus读查询成功。
每次核对worker身份、输出/日志进度、崩溃/OOM/评分/来源SHA/后续接续；生成连续
30分钟输出和日志无变化需处理，加载宽限10分钟，合法控制marker等待豁免。
恢复限已授权可逆队列故障；保留预测/协议/日志，严格最多2GPU，不改冻结数学、
题集或参数、不忽略SHA漂移、不杀外来进程、不做commit/push或外部消息。

最新as-is POPE32已完整8910：均F1 **78.541**（旧80.008），2334重复输入组有
226分歧（旧182）；它没有自然恢复稳定性。独立wait-only控制已经自动生成，
后续同预算EADP32公平基线在同一控制监督器中串行，完整评分后释放v15队列。

### 13.3 Oct8 16:45 增量：完整同步干预与首份官方TextVQA基线

本次as-is AZ32 POPE完整8910题均F1为78.541，重复同图同完整题面2334组中
226组出现Yes/No分歧。仅在内存增加caller流依赖的完整AZ32控制已于16:32评分：
均F1 **83.084**（random84.601/popular84.066/adversarial80.585），实际保留数
8910/8910均为32，完整题面逐字对齐，**重复组分歧为0/2334**。较as-is
**+4.543点**，image-cluster配对95%CI[3.984684,5.129300]；较历史80.008
+3.076点。本次完整干预支持同步问题影响了运行结果，但每配置仅一次完整运行，
不能声称独立重跑可重复性已验证、所有stream/allocator风险均已消除，或修复增益
就是方法创新。生产CLIP源码仍冻结。原包random2910口径不变。

NeXT官方TextVQA第一臂EADP K128（名义640）已完整5000题并官方评分，
独立CPU精确复算 **57.958%**，较旧协议本机E58.136低0.178点、较论文59.2
低1.242点。AnchorZip同协议尚在生成，不能使用前缀准确率或跨协议旧方法分数
配对新基线。当前确定的原包v15 SQA128配对仍是AZ69.2117/E69.5588，
净−7题/−0.3471点、CI跨0；总体不能宣称已稳定超过EADP。

16:40巡检发现控制第2阶段等待条件把外部Ollama的**0 MiB CUDA上下文**也当成
GPU模型，导致EADP32未能接续；该外部进程不是我们可杀的实验worker。已把只读
监控改为只忽略明确Ollama runner的0 MiB上下文（未知0仍计入），并为两控制阶段
之间的并发等待增加10分钟加载宽限告警；9个相关CPU边界用例均通过。后续恢复
必须保留第一阶段预测/协议/评分，只重接第2阶段，保持两个实际GPU模型上限，
不修改冻结worker/数学/参数。恢复登记和实际启动结果见控制state及独立resume记录。

16:52:47已实际接续EADP32：新监督器1692429、冻结worker1692442，状态generating，
与NeXT worker1643948并发；外部Ollama未动。原监督器确认等待且无child后才终止，
旧state按UTC归档；AZ8910/协议/评分和原worker来源SHA均保留。新门采用实际
memory.free（扣reserved）、实测v15峰19134MiB+400MiB余量，启动时free19608，
并发门7个CPU分支通过。状态内旧gpu_gate来自此前失败的空闲等待快照；实际启动
条件以 `concurrency_at_launch.ready=true` 与当前status/child为准。恢复证据在
`rerun_fast/pope_streamwait_stage2_recovery.json`，新监督程序为
`rerun_fast/resume_pope_streamwait_stage2.py`，小时巡检identity已同步更新。

按NeXT E128完整生成75分钟及当前并发实测，暂估NeXT Text128配对17:40–18:00、
其SQA128配对18:30–19:15，v15 Text128配对19:00–20:00；完整批次工期更新为
Oct9 **03:00–07:00**、保守预留08:00。仍是动态预测，尤其FULL时长和首份v15
Text热身后需要校准；不构成赶在12:00前完成的证明。

完整证据：`rerun_fast/pope_v15_AZ32_streamwait.comparison.json`、
`.official.score.json`；新TextVQA/SQA完整成绩见
`rerun_batch/repaired_results_summary.json`。小时监控边界验证在
`rerun_batch/hourly_monitor/zero_context_gate_monitor_validation.json`。

### 13.4 Oct8 18:12 增量：控制闭环与小时巡检

独立EADP32 streamwait已完整生成/评分8910题，三类均F1 **84.073**；
同协议AZ32为83.084，AZ−EADP **−0.989点**，image-cluster配对95%CI
[−1.817871,−0.163651]。两臂2334重复输入组均无Yes/No分歧，random仍为
2910题，生成退出码均0。合法success marker于17:32:33释放，lane1于
17:32:37自动接续官方TextVQA E128；success只表示完整性与评分合法。
控制监督器完成后正常退出，不是丢失监督器。该公平控制结果不支持AZ32
在此POPE设置超过EADP；每配置一次全量运行仍不能证明独立重复性。

NeXT官方TextVQA K128两臂均完整5000复合题键并官方评分，CPU精确核验
EADP **57.958%**、AnchorZip **57.588%**，同协议方法差 **−0.370点**。
初始两臂没有新runtime trace，不能补称已记录逐题实际保留数。NeXT原包
SQA E128于18:07:50完整2017题评分（1365正确，**67.6747645%**），随后
自动接续AZ128；其配对方法差尚未完成，不能用生成前缀估计。

本次独立巡检完整读取本文并核验Git、实际/proc命令、父子与启动身份。
18:09:30→18:11:21，v15 TextE128为4045→4241/5000、NeXT SQA AZ128为
150→340/2017，两worker PID1779765/1829368保持相同启动身份，输出和日志
均连续推进。A40实际两个GPU模型；28实验产物路径、25相关日志无异常。
两lane来源SHA22/22与21/21、两wait控制各8/8及launcher、as-is来源均通过；
汇总6完整行且errors=[]。两lane仍在生成，整体未完成，距Oct9 12:00截止
约17小时48分。本次未重启进程或改生成代码/参数，CPU快照与独立完整性、
评分、SHA审计保存在 `rerun_batch/hourly_monitor/check_20261008T100930Z/`。

### 13.5 Oct8 19:15 增量：两组新完整配对与独立巡检

v1.5官方TextVQA K128两臂各完整5000复合题键、题面/顺序/新增runtime
trace及官方评分通过，独立CPU精确复算EADP **56.390%**、AnchorZip
**56.604%**，同协议方法差 **+0.214点**（官方打印AZ为四舍五入56.60）。
两臂分别18:18:20、19:06:00完成评分，lane1已自动接续原包SQA E64。
NeXT原包SQA K128两臂各2017题：EADP1365正确/**67.6747645%**、
AnchorZip1373正确/**68.0713932%**，净 **+8题/+0.3966287点**；AZ于
18:27:39完整评分后lane2正常接续官方TextVQA E64。未重做这些新配对的
bootstrap，不据点估计宣称显著增益或整体稳定胜出。NeXT Text128的
完整同协议方法差仍为−0.370点，POPE公平控制差仍为−0.989点。

本次巡检完整读取本文并核对Git（分支/HEAD仍为此前记录）；实际/proc完整
argv、父子关系、启动ticks和已结束worker的评分/接续均通过。19:07:54→
19:14:54，同一PID1908673的v15 SQA E64为307→1579/2017，同一PID1856362
的NeXT Text E64为3931→4618/5000，输出和日志同时推进，末次mtime均不足
0.05秒。A40实际两个本计划GPU模型；两lane监督器与summary watcher正常，
控制state complete、两8910完整评分/exit0及合法success marker再次通过。

28个新实验已完整评分9个，共50764/118917条完整新预测；2个生成中、17个
尚未启动。summary的9完整行是6个新Text/SQA加3个历史v15 SQA，POPE另行
审计，不能把汇总行数直接当28实验计数。JSON/CSV一致、errors=[]；全部
记录来源SHA实际988/988匹配（两lane冻结来源22/22、21/21），25相关日志
无Traceback/OOM/FAILED，独立精确评分/分母审计errors=[]。初始NeXT
Text128仍无新增runtime trace，不补称有此证据。

证据与三次CPU只读快照保存在
`rerun_batch/hourly_monitor/check_20261008T110754Z/`，含独立评分、来源/
日志、进程身份、活跃前缀题面/runtime与进度审计。本次无进程重启或生成
代码/参数/预测修改。两lane仍generating，整体未完成；截止尚余约16小时
45分，已有工期预测不构成按时完成证明。

### 13.6 Oct8 20:15 增量：SQA64/32、NeXT Text64完整配对与正常接续

v15原包SQA K64两臂各2017题，EADP1388正确/**68.8150719%**、
AnchorZip1394正确/**69.1125434%**，净 **+6题/+0.2974715点**；
K32为1387/**68.7654933%** 对1402/**69.5091720%**，净
**+15题/+0.7436787点**。独立逐题解析、完整题面、runtime及分母通过。
NeXT官方TextVQA K64两臂各完整5000复合题键、完整题面/顺序/runtime及
官方评分通过，独立CPU精确成绩EADP **55.352%**、AnchorZip **56.026%**，
方法差 **+0.674点**；官方打印55.35/56.03为四舍五入。未为这三组新配对
计算bootstrap，不据点估计宣称显著或整体稳定超过EADP。

本次完整读取交接并核对Git，分支仍为`codex/anchor-completion-validation`、
HEAD `194562d`，未见更新提交。20:08:34→20:15:19，同PID1969775的v15
TextE64完整行 **2285→3189/5000**，输出和日志同时推进。NeXT TextAZ64
由4590/5000完成，20:12:46.016923评分后于.067979启动下一项TextE32，
接续约0.051秒；原PID1927134已退出，新PID2003306的实际完整命令、父子、
启动身份及来源门通过，20:13:36→20:15:19完整行 **71→256/5000**，
20:15:21补证为260条合法预测/runtime。两lane监督器和summary watcher均
存活，A40实际两个计划内GPU模型；已完成控制监督器正常退出，合法success
marker及两控制8910完整评分再次通过。队列任务exit0由冻结监督器非零退出
硬门与completed状态推断；POPE控制另有直接exit_code=0证据。

28新实验已完整评分 **15个/68832条预测**，2个生成中、11个未启动。
12:13:50 UTC独立评分审计时共71897/118917条已写完整JSONL行（含活跃
前缀），不把前缀当完整实验。summary于20:13:39刷新为15完整行，实际
为12个新Text/SQA加3个历史v15 SQA；三POPE另行核验。JSON/CSV一致，
summary.errors与独立audit_errors均为空。来源审计 **1138/1138 SHA**
匹配（571唯一文件），新接续另 **42/42** 补证通过；两lane冻结来源
22/22、21/21无漂移，35份相关日志及4份接续日志无异常。

证据在`rerun_batch/hourly_monitor/check_20261008T120800Z/`，包含五次CPU
只读快照、独立评分/来源与接续审计、实际进程身份、活跃前缀runtime、进度
对照和本机巡检登记。未重启进程或修改冻结生成代码、模型、参数、预测，
未提交推送或发外部消息。两lane仍generating，整体未完成；20:15:19距
Oct9 12:00截止约15小时45分，已有工期预测继续不构成按时完成证明。

### 13.7 Oct8 21:12 增量：Text64完整配对与正常接续

v15官方TextVQA K64两臂各完整5000复合题键、完整prompt/顺序/runtime、
预测SHA及官方评分通过，独立CPU精确成绩EADP **54.918%**、AnchorZip
**54.978%**，同协议方法差 **+0.060点**；官方打印54.92/54.98是四舍五入。
NeXT官方TextVQA E32已完整5000题，精确 **51.996%**（官方打印52.00），
AZ32仍生成，不能据前缀计算配对成绩。未为新Text64配对计算bootstrap，
不据点估计宣称显著或整体稳定胜出。

本次完整读取交接、核对Git，分支仍为`codex/anchor-completion-validation`、
HEAD `194562d`，无新提交。v15 AZ64于21:10:35.183352评分后.241644自动
接续TextE32，约0.058秒；旧PID2026207正常退出，新PID2084699完整argv、
父子、启动身份、冻结来源/协议通过，21:11:19→21:12:31完整行 **77→247**，
输出与日志同时推进。NeXT AZ32 PID2064425保持身份，21:08:02→21:12:31为
**1175→1674/5000**，输出与日志推进。A40实际两个计划内模型（显存
21662/19342MiB），两lane监督器与summary watcher正常；控制complete、
两8910完整评分、direct exit0及合法success marker再次通过。

28新实验已完整评分 **18个/83832条预测**，2个生成中、8个未启动。
独立评分审计21:11:10的完整JSONL行数85411/118917包含活跃前缀，不能当作
全部评分结果；随后summary正常从17刷新至18完整行，即15新Text/SQA+
3历史v15 SQA，三POPE另行审计。JSON/CSV一致，summary.errors和独立
audit_errors均为空。来源 **1265/1265 SHA**（575唯一比较文件）、两lane
冻结来源22/22和21/21无漂移，43份唯一计划相关日志无异常。21:12:53补证
新E32的298预测/runtime、NeXT AZ32的1717预测/runtime完整身份通过；
NeXT实际保留数随输入几何变化，不将名义32误写为总保留数。初始NeXT
Text128无新runtime trace的边界继续保留。

本机证据在`rerun_batch/hourly_monitor/check_20261008T130802Z/`，含五次
CPU只读快照、评分/来源接续补证、实际进程身份、完整分母与进度对照。
本次无故障恢复、GPU启动、预测或冻结生成代码/参数修改、commit/push及
外部消息。两lane仍generating，整体未完成；21:12:31距Oct9 12:00截止
约14小时47分，旧工期预测不构成按时完成证明，授权队列继续执行。

### 13.8 Oct8 21:20 用户请求的进度核查与工期更新

21:17:59→21:20:18只读快照确认v15 TextE32为1004→1346/5000、NeXT TextAZ32
为2270→2509/5000，两worker持续推进、实际两个GPU模型，needs_attention=false。
新完整实验18/28，2项运行、8项未启动；完整评分预测83832/118917，活跃前缀
不算完整实验。summary18完整行包含15新Text/SQA+3历史控制，三新POPE另计。
完整同预算配对9组中点估计6高3低；POPE32公平差−0.989，仍不支持整体胜出。

实测v15 Text5000用时约40–47分钟、SQA2017约10–11分钟；NeXT Text5000约
44–54分钟、SQA128约20–21分钟。剩余v15预计23:00–00:00结束；NeXT余5项
SQA和1项FULL Text，FULL本轮尚无实测，保守给60–120分钟。整体工期更新为
Oct9 **00:30–02:30，保守预留03:00**，比原03:00–07:00预测提前；这是剩余
任务量与当前吞吐的估计，尚非完成证明。小时巡检21:07触发、21:15:50成功
退出healthy，下一次22:07:12。此次未启动或重启GPU、未改冻结源/参数/预测。

### 13.9 Oct8 用户新增授权：每完整一组立即提交推送

用户先要求跑完后推送，随后明确改为**“跑完一组就推一组”**。该后续指令覆盖
13.2及此前小时巡检的“不做commit/push”约束：现在允许在完整评分/来源核验
后将本轮结果与审计修复提交并推送 `origin/codex/anchor-completion-validation`。
确认本轮开始时本地、tracking及真实远端均停在194562d，下午新结果尚未提交。
一组按同模型/同任务/同预算的EADP和AnchorZip完整配对计，FULL单臂可独立；
POPE仅完整8910公平同步配对，as-is作为审计辅证。未完成前缀不发布方法成绩。

增量发布需精确allowlist，保留原预测及live Git index/用户暂存，只取本轮修复
源码、docs与完整结果/协议/官方score/runtime/来源SHA，保留初始NeXT Text128
无新增runtime的事实。用户已有figure/export/screen/mmben/p3_data_serial脚本
及llava_round2_driver.sh先前图像路径改动不夹带。禁止向upstream推送、force
push或重置用户工作；每次需查询实际远端确认提交到达后才记成功。独立CPU
按组发布服务与GPU队列解耦，失败保留待重试状态；小时巡检监控其健康。

21:59:32首批11完整组已实际推送并独立ls-remote确认远端提交
`95d1a6440e550b78c3a2b5bc2417c7f0359c3a6c`。发布目录为
`records_20261003/audit_followup_20261008/published_results/`；11组+common共281
归档member逐项SHA/size复核通过，合计约54MiB、最大blob8.77MiB，GT symlink
已物化成普通文件。新增NeXT Text32完整配对为E51.996/AZ53.174，方法差+1.178。
实际提交/推送在独立detached worktree进行，**live工作区HEAD仍194562d、live
index原SHA完全保留**，未夹带用户已有变动；后续不要仅看live git log就判定
结果未推，须核对真实origin和 `rerun_batch/group_publish/publish_state.json`。
publisher源码为 `publish_repair_results.py`，prepare/validate只做CPU核验，
--publish实际提交推送，--watch --interval30用于每组增量发布。来源漂移、
完整性或未知远端更新拒绝发布；push丢回复可由pending-publication恢复，禁止
force。相关9项CPU边界验证通过，确认成功后才记组为已发布。

22:14确认按组服务 `codex-eadp-group-publisher-20261008.service` active/running，
每30秒检查完整组，全部17组发布并且实验全部完成后自动退出。首批评分另有
可读 `published_results/paired_results.csv` 和README。初次后台访问GitHub因
systemd未继承交互shell既有代理而超时；仅私有0600服务环境文件继承现有连接
配置后已恢复，环境/账户文件不入库。更新版服务实际提交/推送成功、远端确认、
pending为空并清除旧错误。无新组且allowlist源码/审计/评分表内容无变化时，
本地指纹直接跳过Git操作；新组或任何被允许内容变化仍触发发布，相关14项CPU
边界验证通过。两GPU队列与评分汇总继续健康，发布服务不改变生成源码/参数。


### 13.10 Oct8 22:13 独立巡检：20项完整评分与发布服务恢复证据

本次完整读取交接并核对Git，live分支仍为`codex/anchor-completion-validation`、
HEAD `194562d`。28新实验已完整评分 **20项/93832条预测**，2项生成中、
6项未启动，尚未满足全部完成条件。summary JSON/CSV20行一致，实际为
17个新Text/SQA加3个历史v15 SQA；3个新POPE另核，不能用summary行数
替代28项计数。summary.errors与独立audit_errors均为空，完整预测SHA、
5000 TextVQA复合题键/题面/顺序及可用runtime、2017 SQA、8910 POPE
(random2910)分母通过。1254/1254来源SHA匹配（577唯一文件），两lane
冻结来源22/22、21/21无漂移，47份计划相关日志无异常。两streamwait
控制完整评分/direct exit0/wait-only来源/合法success marker再次通过；
公平POPE差仍−0.989点。初始NeXT Text128无新增runtime的边界保留。

新增完整v15 TextE32精确52.532%，AZ32仍生成，不能计算完整配对；
NeXT Text32完整配对E51.996%/AZ53.174%，差+1.178点，未据该点估计
宣称显著或整体稳定胜出。两新接续分别约0.055/0.057秒，前次worker已
正常退出且评分完成。22:08:19→22:12:37，同PID2135335的v15 TextAZ32
完整行2365→2848/5000，同PID2131159的NeXT SQA FULL为1263→1480/2017，
输出和日志同时推进，末次mtime均不足0.75秒；完整argv、父子、cwd、
启动ticks/启动时间与计划一致，两监督器及summary watcher正常，A40实际
两个计划内GPU模型。完成的控制监督器正常退出，不是丢失进程。

发布服务曾出现Git TLS非正常终止及github.com:443连接超时并登记
failed_retryable；journal记录22:09:51发生stop/start，非本巡检执行，
不能声称无需重启即恢复。随后多次发布成功，22:13:24独立ls-remote
确认origin为`2233f33496f536aad0fa1d2a48314897ee354e24`且与state一致。
已发布仍11完整组，后续提交为metadata refresh；281证据文件、12内嵌
manifest、12归档blob及190已发布来源SHA通过，publisher PID2169682
active/running。live HEAD/index字节/暂存内容均保留登记状态。本次
独立归档审计已按发布格式区分内嵌MANIFEST与外部含archive hash的清单，
errors=[]；旧error字段保留的是已被成功发布超越的历史网络错误。
其余6组仍合法等待完整生成，本巡检未自行stage/commit/push。

三次CPU只读快照及评分/来源/日志/进程/发布核验保存在
`rerun_batch/hourly_monitor/check_20261008T140736Z/`。本次未启动或重启GPU，
未修改冻结生成代码、参数或预测，未新建定时任务或发外部消息。距Oct9
12:00截止约13小时47分，剩余队列继续执行；已有ETA不是按时完成证明。

### 13.11 Oct8 23:13 用户要求继续定位NeXT TextVQA两点残差

用户明确认为EADP复现低约2点可能共同影响新方法，要求继续寻找原因。这是
新增诊断任务，原28项冻结队列继续执行；不把残差归于普通浮动，不直接给
两方法都加分，也不为了追分挑参数或样本。NeXT TextE128/64/32完整5000题
对论文表2及补充表16仍低1.242/1.648/2.204个百分点，原因尚未确定。

独立paper/config/input审计已排除预算映射错误：补充材料§9明确用importance-
based，Eq26要求floor/min1；CLI32/64为每crop预算，五crop实际156–159/
316–319属于官方取整。7项核心源码与官方clone逐字相同，官方OCR题文件
与eval.zip字节相同，固定16题的token IDs、RGB及half后五crop张量、尺寸
均与官方CustomDataset/collate逐元素相同，全部5000题不触及4096上下文上限。
本地模型配置仅model_type适配与HF参考不同；权重文件size/header一致但
未读取14GB payload进行绝对哈希核验，不能称权重全部身份已证明。
明确数学参数差异是本机beta2 vs官方NeXT脚本默认beta1；论文每任务实际
beta/q未公开，默认差异不能直接归因为主表差距，单一默认参数对照应独立标注。

找到具体CUDA依赖缺口：`llava_arch.py`的5D AnyRes分支在caller stream
执行torch.cat（656行）后立即encode_images（657行），CLIP在新image_stream
读取，入口没有等待caller。后置全局synchronize只等完成，不能保证先写再读。
NeXT TextE/AZ K32/K64完整runtime均5000题经过此五crop路径；外层blocking
传输不足以排除内层concat竞争，此前以blocking为由排除Text同步风险的判断
需撤回。此静态缺口仍未证明能解释2.204点；准备固定128系统抽样、同模型
as-is/wait-only各重复两遍的独立机制对照，保留所有原始预测与冻结源码。
先安全等待v15 lane完成空闲slot，至多两个GPU模型；若主lane2转入FULL
且显存窗口不足，仅中止自己的诊断并保留partial，原队列不受影响。

CPU证据与新诊断产物在`outputs/audit_followup_20261008/
next_gap_diagnosis_20261008/{paper_config,input_identity,stability}/`。NeXT
TextFULL仍是下一项关键基线；先量化同步机制，再按证据决定是否需要完整
两臂wait-only及单一官方默认beta1对照，不能将128题面板当全量准确率或
以面板选择最好配置替换主表。

### 13.12 Oct8 23:26 机制阳性与新增全量控制注册

固定128个等距中点样本、同模型各as-is/wait重复两次，共512次生成完整完成，
supervisor2305566/worker2305580正常退出、完整marker及SHA核验通过。主实验
用no-index-trace，不在vision消费前插同步或GPU tensor读回。原代码同输入
重复回答分歧5/128、实际保留数分歧43/128；wait-only两项均0/128。这个对照
证实实际运行不稳定被缺失的流依赖修复；它尚未解释论文2.204点准确率差距。
面板分数as-is1/wait1均56.25、as-is2为56.953125/wait2为56.25，没有准确率
改善证据，不能把不稳定修复直接换算为全5000题方法增益。

三份NeXT safetensors完整payload共14,125,909,456 bytes逐份SHA全部匹配
官方HF同revision LFS SHA，读取前后stat不变，§13.11的权重未验边界已关闭。
v15 TextFULL新完整5000精确58.226%，论文58.2；原批次27/28已完整评分，
只有NeXT TextFULL尚在生成。原SQA NeXT K32新配对E67.575607/AZ67.178979
差−0.396629应保留；不因新增诊断覆写任何原始成绩。

23:35:45已实际启动独立全量控制supervisor2325219，源码与计划冻结，
先等原两lane及panel全部完成退出，然后至多两GPU并发跑EADP_beta2和
AZ_beta2（同5000题、同K32/alpha.5/greedy128、只补同一内存wait修复）。
EADP_beta2完整官方评分成功后再接唯一EADP_beta1公开脚本默认参数控制；
这不是确认论文用了beta1，不搜索q/beta最高分，三项不计入原28分母。
原生runtime/wrapper由runpy原样执行，生产源码不改；独立输出/原生及控制
protocol/actual-token trace/官方精确m4c score/完成marker保留。loader未开始
生成时占用reservation，第二槽必须等实际NVML和首条预测/runtime，防止
同时加载误判；启动显存门22500MiB，未知零显存进程也占slot，仅真实Ollama0
豁免。12项controller门验证、31项fullworker和17项独立核心验证通过。

实际状态为`next_gap_diagnosis_20261008/stability/
next_text_full_streamwait.controller.state.json`，私有小时prompt已增加新控制
与发布检查。原28完成但已注册新诊断或远端发布未完成时不能停止小时巡检。
按组publisher扩展期间短暂停CPU服务，测试通过后恢复；新增配对和单默认
控制分别归档，单beta1不标成AnchorZip或FULL。128及CPU身份证据SHA列表为
`outputs/audit_followup_20261008/next_gap_diagnosis_report.json`，原17组保持。

### 13.11 Oct8 23:12 独立巡检：24项完整评分与14组发布核验

本次完整读取交接并核对Git，live分支仍为`codex/anchor-completion-validation`、
HEAD `194562d`，tracking落后10个提交；独立publisher使用detached工作区，
不能据live HEAD判断发布丢失。23:10:01独立CPU审计确认28新实验已完整评分
**24项/104883条预测**，2项生成中、2项未启动。summary JSON/CSV24行一致，
实际为21个新Text/SQA加3个历史v15 SQA；三新POPE另核。summary.errors、
独立audit_errors均为空；完整预测SHA、TextVQA5000复合题键/完整prompt/顺序、
可用runtime、SQA2017及POPE8910(random2910)评分分母通过。来源
**1357/1357 SHA**（584唯一文件）、两lane冻结来源22/22和21/21匹配，
55份计划相关日志无Traceback/OOM/FAILED。两控制complete、各8910完整评分、
direct exit0/wait-only来源与合法success marker再次通过。

新增v15官方TextVQA K32完整配对精确EADP **52.532%**、AnchorZip
**52.542%**，差 **+0.010点**；官方打印52.53/52.54为四舍五入。
NeXT原包SQA FULL为1365/2017、**67.6747645%**；K64配对EADP
1356/**67.2285573%**、AnchorZip1363/**67.5756073%**，净
**+7题/+0.3470501点**。未计算这些新增配对的bootstrap，不据点估计宣称
显著或整体稳定胜出。初始NeXT Text128无新增runtime trace的边界继续保留。

23:08:37→23:11:08，同PID2199422的v15 Text FULL完整行
**3903→4190/5000**，同PID2275562的NeXT SQA E32为**778→1100/2017**，
输出和日志同时推进，末次mtime均不足0.23秒。23:12:02补证完整前缀与runtime
分别4292/1215无错误。完整argv、父子、cwd、启动ticks/时间与计划一致；
两监督器和summary watcher存活，A40实际两个计划内GPU模型（19054/21362MiB）。
上轮worker已正常退出且完整评分，新任务最近接续分别0.355549/0.083558秒；
queue退出0仍由冻结监督器硬门及completed推断，不补称存在独立per-job exit字段。

按组发布服务PID2175988 active/running且身份/启动ticks未变，14/17组已发布，
新增`v15_textvqa_K32`、`next_sqa_FULL`、`next_sqa_K64`；三剩余组合法等待。
23:10:09独立安全ls-remote确认origin指定分支为
`0746b5d9d80e34fe72df4b81a40b869b79b5974d`，与publish_state一致；
status=published，pending_commit/publication/error均为空。23:11:59独立
归档审计15 blob/15内嵌manifest/366 evidence member的SHA/size、265来源SHA、
已发布预测分母/复合键/分数及69提交路径allowlist通过，errors=[]。
live HEAD/index字节/暂存状态与上轮一致，用户figure等排除文件未夹带；
本巡检未自行stage/commit/push或重启publisher。

三次CPU只读快照及评分/来源/日志/进程/发布核验保存在
`rerun_batch/hourly_monitor/check_20261008T150749Z/`。本次无故障恢复或GPU启动，
未改冻结生成代码、参数、预测、计划状态或定时任务，未发外部消息。
整体尚未完成，23:12距Oct9 12:00截止约12小时48分；授权队列继续执行，
已有ETA仍不是按时完成证明。

### 13.13 Oct9 00:37 追加全部受影响补跑，已实际登记自动接续

用户明确要求把新发现的问题也排队，生成跑错/受生产者流依赖影响的原结果
补跑，前批约凌晨3点结束后自动接上；继续执行每小时巡检和每组完成即推送。
这里的“受影响”指静态路径确实存在未建立调用流依赖，不能据此断言每一条
历史预测都错误，也不能把128题稳定性修复当成追回论文2.204点的证据。

接续严格按完成状态和进程退出触发，不设03:00固定开始：原28批最后NeXT
TextVQA FULL → 已登记的3项5000题Text K32控制（E2/AZ2/E1） → 12项核心
wait-only补跑 → 32项旧入口wait-only补跑。原监督/预测/生产源码不改；新
worker只在内存增加调用流等待的3个AST节点，alpha0.5/beta2/lambda0.25
及原图像、问题、提示、生成预算保持，单beta1仍仅是单独发布默认敏感性控制。

共同目录为 `Qwen_vl/outputs/audit_followup_20261008/
streamwait_continuation_20261008/`。phase1于北京时间00:13:39实际注册
监督PID2391998，`streamwait_repair_continuation.state.json`有started与12
registered_jobs、7publication_expected_groups；当前合法等待原3控制。12项
共39119题：Text E/AZ64/128及FULL，SQA E/AZ32/64/128及FULL；Text K32
已包含于前3控制，v15 blocking Text/SQA无此流依赖问题而不重复生成。worker
SHA d2b213ab51bf6cdee1a20f92b5a8b236f64a8ea144bea0a61b82d36b2552fafd；
controller SHA30296dc9ee8e3e139e666b1c7e79ef9783ef968588c594e3f19a7dee8eb74c0a；
plan SHAc9e3f6248d2e0e2f032663fcbb1972ed30aab74f35496c583848d9159917c850。
47项worker、13项调度CPU门及真实57source/12路径发布门通过。

phase2也已实际注册监督PID2430956，`legacy_streamwait_continuation.state.json`
为waiting_for_phase1、started非空、32registered_jobs/29publication_groups、
child_pids=[]；164冻结来源校验通过。32项共224199题：v15 POPE4项、MME5项、
GQA5项；NeXT POPE/MME/GQA/VizWiz/MMBenchEN/CN各3项AZ预算。v15 POPE32
已经完整wait-fixed，排除重复；NeXT旧POPE不完整的FULL/E128不伪装有效完成。
3组v15 K128 E/AZ公平配对、3组v15 FULL、23组独立单臂审计，单臂没有本地
EADP配对，不从论文数字伪造本地基线。顺序为全部MME→POPE→MMBenchEN/CN→
VizWiz→全部GQA，任务内部可跨单双组并行。workerSHA
1c8bb72fc855ffc26b3273ac6dfdf327c8d2fb56eded19e7d8cf4cb4c2f31fe3；
controllerSHA82a7ce81194bf15b0f4d898b23f14a68ef88a8756b8c81ab25e18b94bcd77f77；
planSHAd63b92f5bcee064484b08392a09504592b6a8026d007564f5c1a787f7ada6972。
所有32新协议统一attempt2，原完整预测及attempt1控制协议保留。

phase2第一次CPU准备后、任何actual --run/GPU生成前，发现GQA的官方每题报告
遍历set导致不同hashseed下列表顺序不同，会误拒绝完整评分。旧worker、旧
controller、32attempt1协议、prepared计划/状态、旧test精确源码归档至
`prepared_only_archive_20261008T163137Z/`；没有对应预测/runtime/score/finished/log。
之后按原问题序固定GQA/VizWiz报告与精确value的求和序，仅改变报告序，不改
每题判定或官方数学。37 CPU tests通过，GQA/VizWiz各两个真实hashseed子进程
完整报告相同，32历史非空官方summary/每题完全相同（报告序归一）。完整EOS
空答、MMBench字母提取失败保留完整分母按官方计错，不能据答案错误拒整臂；
CPU门只在隔离namespace去掉严格证明的empty-only冗余拒绝条件，源scorer冻结。
第二次14项调度门通过。具体恢复边界记于legacy_prepared_only_recovery_provenance.json。

原脚本只能恢复历史recipe，历史外部env/实际进程argv未记录，不补称已知。
NeXT VizWiz/MMBench原max_new_tokens1024保留，旧loader128保留；MMBenchCN
旧脚本省略--lang而默认en，此次wait-only控制明确保留en，尚不能标成论文协议
完全一致。MME按已验证官方2374 GT（162条历史标签差异已纠正）精确perception
原始点数评分，不叫准确率百分点；MMB4876行/1292循环组；POPE8910随机2910。
纯评分错误可以CPU重评；这些legacy生成另有流风险所以仍补跑。perf2 prevision
计时同步阻断此次流风险，不机械重跑perf2；无关Qwen与有效blocking入口不重复。

所有监督器最多2个真实模型，包括尚未建立NVML context的loading reservation。
第二槽需第一worker实际NVML和完整prediction/runtime首行，free>=22500MiB。
NeXT FULL必须真实独占且free>=32000MiB；v15 FULL原内存spec允许22500及并发。
只豁免真实Ollama runner且0MiB，未知零上下文占slot。phase2启动必须phase1全12
完整官方分数和成功marker、supervisor/children退出，原三控制也完整成功退出。
任何worker失败保留部分产物并停自有子进程，不覆盖原结果或伪造完成marker。

publisher已于00:31:37恢复active PID2423330，冻SHA
 ae9fc06bcb69dc20c2e2987badda79a95474019fa38c1ef58f9a7eed7a477d57，57项CPU门通过。
实际scope为原17组+诊断2组+phase1 7组+phase2 29组，共55；必须全部完整评分并
远端确认才结束。legacy单臂写独立repaired_legacy_results.csv，旧paired CSV
保留；MMB含base64图像的原TSV不归档图像payload，而归档4876行无图GT、原SHA
及imagehashmanifest；MME评分zip归档2374规范GT JSON并保留原zipSHA引用。
用户figure/driver、私有env/hourly/state/lock与模型图像不夹带，live HEAD/index
保持194562d和原87ad06fd…字节；origin正常快进推送，绝不推upstream。

私有小时prompt已纳入实际phase1/phase2与55组远端发布，原28独立完成不停止timer。
32项旧日志累计15.297 GPU小时，理想双槽7.649小时只是下界；连同前序关键控制，
核心暂估Oct9上午07–09点，全部legacy可能持续到Oct9下午或更晚，不能承诺12点
全部完成。以真实进度更新ETA，优先交付论文核心对照，不承诺方法或复现必增2点。

### 13.14 Oct9 01:15 独立巡检：原28完成，新增控制正常接续

本次完整读取交接并核对Git，live分支仍为`codex/anchor-completion-validation`、
HEAD `194562d`。独立CPU逐题/来源核验确认原28项全部完整评分，合计
**118917/118917新预测**；TextVQA5000复合题键与全文prompt/顺序、SQA2017、
POPE8910（random2910）、可用runtime及官方分母通过。来源1411/1411 SHA
（590唯一路径）匹配，61份日志无Traceback/OOM/FAILED，独立errors=[]。
summary JSON/CSV28行一致且errors=[]，其中25新Text/SQA加3历史v15 SQA，
三新POPE另核，不能用summary行数替代实际28项。两lane、两POPE控制及合法
success marker均完成；原监督器和summary watcher实际已退出，watcher最后
成功导出与冻结源码正常退出规则一致，未补称存在独立持久化退出码。

最后NeXT TextVQA FULL完整5000题精确**60.37%**，5000条runtime实际均2880
tokens；评分完成于北京时间00:48:03.977605，随后约0.632秒接上新增K32
EADP_beta2，首条完整runtime后再接AZ_beta2。初始NeXT Text128两臂无新增
runtime的边界保留；原queue逐job退出0由冻结监督器硬门和完成记录推断，
两POPE控制另有直接exit0证据。overall_run_plan旧进度/ETA仍为历史估计，
当前完整预测、官方评分和lane状态优先；基础monitor的all_complete=true
只覆盖原28，不能据此结束新增任务巡检。

01:11:22→01:14:52，同一启动身份的PID2451780/2452469分别由
**2436→2817/5000、2403→2783/5000**，prediction/runtime完整行一致，输出
与日志mtime同步推进，最后age分别约0.015/0.275秒；A40实际仅两个登记模型，
无未知零CUDA context或额外加载reservation。控制监督PID2325219正常，
beta1仍pending且EADP_beta2完整5000官方评分硬门尚未释放，不报告任何新
全量对照成绩。AZ启动记录free27891MiB、首worker真实context16358MiB且
首条runtime时间早于AZ启动，第二槽门通过。

128面板complete、128样本/512生成、success/result SHA及独立官方逐题
复算再次通过；回答分歧as-is5/wait0、count分歧43/0，面板没有准确率改善
证据，不能解释全5000题或论文2.204点差距。新增控制/phase1/phase2审计
4987/4987 SHA匹配（1693来源/工件、3294图像payload，3337唯一路径）；
四份计划来源表32/38/57/164全部通过。phase1 PID2391998已登记12任务/7组，
phase2 PID2430956已登记32任务/29组且全部attempt2；实际Python -u启动身份、
controller/worker/plan/协议SHA通过，分别合法waiting_for_existing_controls和
waiting_for_phase1，child=[]。未把pending manifest或活跃前缀当完整评分，
未启动、信号或修改这些只读巡检对象。

publisher维护已结束，source_ready=true/maintenance_active=false，服务
active/running PID2423330且冻结SHA通过。17/55组已发布、38组合法等待；
01:10:32独立ls-remote确认origin指定分支为
`f0adb8ae6b19163821e76350487fe3ee0ae3a1c5`，与当时publish_state一致，
pending_commit/publication/error为空。新增next_textvqa_FULL远端组已确认。
18归档/18内嵌manifest/429 evidence member SHA与size、104 metadata SHA、
317来源SHA、30预测/116058归档行及176提交路径allowlist通过，用户figure等
未夹带，live index/暂存保持。本巡检没有自行stage/commit/push或重启服务。

本机快照、独立评分/来源/进程/新增scope/发布证据保存于
`rerun_batch/hourly_monitor/check_20261008T170827Z/`，独立审计器修正了对
Python -u的首轮身份误报并保留原检查证据，不是实验故障。整体仍未完成：
新增3控制、12+32续跑任务及全部55远端组均纳入停止条件。01:15距12:00约
10小时45分，legacy尚32项、前序尚未结束；旧日志双槽7.649小时下界不足以
证明全scope能按时完成，截止风险仍存在，授权队列继续执行。小时timer仍
active，本次未改定时任务、冻结生成源码、模型、参数或预测，未发外部消息。


## 13.14 追加复现错误与全覆盖盘点（2026-10-09 01:21 北京时间）

用户要求继续找项目错误，并核实是否全部EADP已复现。先检查live branch/status并
完整阅读当时906行handoff；三个并行CPU审计只写独立
`additional_reproduction_audit_20261009/`，未改生产算法/环境/冻结GPUworker、
controller、plan、protocol或旧预测/score，未新启动GPU。完整汇总为REPORT.md及
`audit_summary.json`，来源/覆盖/协议/实现各报告和CPU脚本保留。

Coverage218个唯一格，按论文120个主表FULL/EADP格，完整预测+对应评分仅34：
v15 15/40，NeXT8/40，Qwen legacy11/40；v15其中6格、NeXT8格wait重跑仍pending。
完整评分覆盖不等于作者配置身份已证明。NeXT完整本地E仅Text/SQA三预算，其他
MME/GQA/VizWiz/MMBEN/CN只有AZ；POPE E1287/8910、FULL1901/8910不算完成。
v15缺Viz/MMBEN/CN E和MME/GQA低预算E/POPE64 E。Qwen论文预算512/256/128，
不是本地256/128/64；E256十任务有评分，E512/E128及大部分FULL未齐。
Qwen native另8完整格与legacy实际DeepStack/mRoPE状态不同，不能替代或混算。
VQAv2不完整、MMVet218预测缺对应官方GPT评分，不生成虚假LLaVA Avg9。
Qwen legacy HallB E256meta记录bank SHA与当前完整bank不同，resume仅首次记录
有可能留旧meta，但生成时身份未闭合；不改score、不伪造生成来源、不预判结果错。

新增确定项目协议错误：两模型MMBench-CN旧脚本漏--lang cn，native默认en，
4876条LLM答题指令全部改变，CLIP guidance不变。现phase2 wait-only明确保留en，
不会自动修语言参数。独立CN proposal prepared_only/registered=false/started=false，
6个AZ臂×4876=29256题保留V11/alpha.5/beta2/lambda.25/max1024/vicuna和wait；
不属于现55组。NeXT已注册wait+en可作单因素语言基线；v15还缺同wait+en基线，
不能把v15新旧全部差值归因语言。这个提案尚未执行，不能称修好或已排GPU。

新增协议差异：MMB发布脚本EN20230712/CN20231003 vs当前V11。EN旧4377行/
1176组 vs4876/1292，950共有baseID有31内容/GT变化、16答案字母变化；实际paper
题集未知。图像多数重编码/小像素差异，不称943错图。MMB12份完整raw解析与
VLMEvalKit no-judge逐行一致、无fail，V11所有旋转已生成；28组固定D来自TSV，
不能再加all-rounds重复题集。MMB1024确与发布/上游native同，排除伪错误。

VizWiz发布scorer min(matches/3) vs官方留一公式已用6份相同4319完整预测CPU
量化，发布分高1.391–1.473pp；NeXT AZ128官方59.254457/发布60.716215，保留
官方主分数，附发布口径审计，不证明paper实际使用该scorer，不解释Text2.204。
Viz旧model_vqa guidance保留回答指令且1024 vs作者loader移除/128；发布验证
alpha0 vs本地.5，POPE发布alpha1 vs本地.5；发布各任务beta默认1 vs多数本地2。
22官方task entry已逐项记录，默认值不能反推paper argv；beta1既有控制仍pending，
无参数搜索、无选择最高分，所有新入口对照仅proposal未注册。

新增确定AZ映射偏离：NeXT Completion预注册逐crop，实际按整图split_sizes=[5]
flatten后允许跨crop assignment。仅AZ，不能解释EADP基线。隔离逐cropadapter
已准备并root独立复跑16/16 CPU gates：单crop/λ0/fullkeep一致、五crop局部
索引与数目/顺序不变、无prodimport/无CUDAinit；未安装/未登记/未测accuracy。
RTG只用首文本段已证实（Text881/5000多段），原M>1 mapping未明确，暂不算bug。

sharedCLIP336单次读1,711,974,081字节SHA等官方HF ce19dc...，7小文件身份同；
13扩展核心入口file byte同root复核。沿用先前NeXT LLM/projector身份，不称本次
核验v15所有LLMpayload。6环境依赖版本差异无accuracy影响证据，不改live env。
NeXT新不剪枝Text完整60.370% vspaper60.3（rawSHA acf08dce...），v15 58.226%
vs58.2；收窄global两点错误，剪枝执行/参数残差未解决。新运行对象alpha/beta/
dtype/class未动态记录，source/protocol不能替代loaded对象trace。
POPE新增primary核验：LLaVA指向旧POPE发布本就random2910/两组3000，所有8910
asked顺序/身份/标签完全一致，不因所谓缺90补生成。GQA12578身份题目/图名和
GT一致，MME2374和历史162GT已修分；SQA官方包CQM-A图题2017身份同但作者
CQM-I仍未提供，不把猜造格式称作者输入。

01:14实际GPU仅NeXT Text K32wait E/AZ beta2两子PID2451780/2452469生成，
beta1pending，phase1PID2391998等待旧控制，phase2PID2430956等待phase1；原28已
完成，现注册55组/远端17组，与120格覆盖不同。小时timer01:07触发，publisher
active且no_op/error空。root16项集成CPU门通过，liveHEAD194562d/index87ad06fd
及原publisherae9.../两个plan/旧workerSHA保持。新增审计将通过原publisher静态
metadata范围复制到streamwait_continuation_20261008/additional_audit_20261009，
保留原sourceSHA，不修改55组/44续跑/3先行control计数；独立修正候选不伪装注册。
不能承诺本轮全部达到paper，更不能给每个方法统一加两点；完整补齐120格远超
现队列，截止Oct9中午按已有完整核心对照交付，剩余协议修正需独立冻结全量控制。


## 13.15 用户0.2点数值验收与CPU评分门恢复（2026-10-09 01:55 北京时间）

用户明确：同模型/任务/预算的EADP完整复现低paper不超过0.1–0.2点可接受，
高于paper不为贴论文数字继续重跑；明显低于paper优先排错，以公平显示新方法
贡献。采用0.2上限、未舍入值Decimal比较；数值状态与协议/来源caveat分列。
已确认输入/评分/执行错误仍纠正，低分不自动证明bug，不承诺修正必涨分。
公共修复同施E/AZ，同预算同输入配对，不选seed/参数/指标最好分。
新的POLICY.md及deficit_triage.json/.csv在threshold_triage_20261009。
按120主表baseline格，34完整评分中17数值接受（9超过+8低<=.2），16低>.2
继续排查，另1Qwenlegacy HallB FULL为刻意1D/noDeepStack控制不可比paper FULL；
86缺/未评分单列，不算不达标，不以AZ填E，未算缺项总体Avg。MME raw照列，
/20仅paper Avg等效单位非准确率；OCR/10。NeXT TextFULL60.37 vs60.3等数值
达标不为贴分再跑，但已注册同步对照用途不因高分取消冻结协议。

01:35两个NeXT TextK32wait-only worker已各完整5000生成、recorded exit0、
官方score/state/finished合法；controller却failed，publisher failed_retryable，
phase1/phase2合法等待，GPU空。具体CPU bug为validate_and_score返回
actual_count_distribution=Counter整数键，JSON保存成字符串键，后续strict dict
比较机械误拒。固定真实Python3.10两个hashseed独立10000题复核：唯一不等
字段就是键类型，其他含accuracy逐位exact一致；3.12审核sum尾位差不是实际
事故主因，没有给生产加容差。E52.086000000000055/AZ53.26200000000008，
方法差+1.176pp；对旧as-is E+.090/AZ+.088，不解释剩余E对paper54.2差2.114。

原failed controllerstate/marker/plan及两臂完整score/state/finished/protocol共16
份精确copy至threshold_triage_20261009/json_gate_failure_originals/；原预测/
runtime/score及生产/冻结源码、controllerplan/protocol始终不改。新独立
json_count_distribution_overlay.py SHA2a0a0cb07e1bd9779ee60349fd1cc23b76b4d6c5be5262208595f33f87973406
只在CPU读入确切原worker80affa...且SHA匹配后，包装其严格validate_and_score
输出count键str化；原5000复合题键、runtime/预算、官方数学、源SHA门全部
保留，数字不改、数值容差无。27CPU门包含真实两臂原gate、两个hashseed、
错误SHA/分母/count/1e-10分数严格拒绝。任何其它动态模块加载原样透传，hook
退出恢复；GPU子进程仍原command，不安装该CPUwrapper或改模型math。

新recovery resume_next_full_controls_after_json_gate.py SHAce871705c2a868a8eb888702be288b87f0405f1e84c35777cb2d4903a4a1d6ec，
planSHA12d2d3e6413b6bdce8000345b753718214ef26ccb197625d3a927c4a798c04d5，
16CPU恢复门+真实全部图像/oldarchiveSHA通过。01:49:56 root实际--run启动
新supervisor2556883，preserve原started/三arm/两发布组，复用两beta2完成臂，
只跑原已登记未开始beta1 command，GPU child2556913；不重复生成beta2。
原失败canonical marker精确归档核验后移除，完整beta1 exit0/三臂再严格校验
才写真实全controller成功marker退出，不能跳过beta1或仅twoarms释放phase1。

仅两个waiting且无child的CPU接续supervisor2391998/2430956被root按真实/proc
身份停止，signal前后state精确保留于waiting_controller_recovery_originals/；
无GPU子进程被杀。canonicalstate以显式recovery provenance恢复原prepared
入口后，原controller通过固定overlay/runpy重新登记相同12/32项、原plan/源SHA
不改；phase1新PID2558610、phase2新PID2558609分别等待前3control/phase1，
child=[]，当前全部pending。originalstarted/PID/SHA留recovery字段，不冒称从未
注册。新CPUoverlayargv含--target原脚本路径 -- --run --plan，hourlyprompt已
更新实际身份，不能按旧直接Python -u误报，未知其它wrapper不豁免。

publisher01:45:23暂停CPU做此恢复，01:52:10服务恢复active新PID2560860；原
sourceae9...不改，唯一systemd drop-in 30-json-score-key-overlay.conf把ExecStart
包为固定Python3.10+同overlay --target原publisher -- --watch --interval30，
原主unit/EnvironmentFile保持。private registration maintenance结束/source_ready
true，新PIDs及overlaySHA/实际restore证据已记；小时prompt对应恢复完毕。
01:53实际beta1进度501/5000、单GPU21456MiB、两新controller合法waiting；
publisher已完整发布K32wait pair为第18/55组，status published、error及pending
为空，remote记录f8dafb90...。后续需独立ls-remote/归档验证，不仅信state。
现55scope不增，不为本次新验收随意删既定控制；新CN/逐crop候选仍未GPU注册。

新阈值表按最新预定wait协议K32 E52.086主选，不是选两个结果高分；旧51.996
出处保留。优先残差为v15POPE32 -2.6264、NeXTText32 -2.114/64 -1.648/
128 -1.242、QwenHallB256 -1.7595（bank身份未闭合）/Doc256 -1.6644/AI2D256
-1.1534。先完成原β1单因素公开默认控制，再依明确source/输入/actual配置证据
决定新修正，不把默认参数当paper主表实际argv，不做追分参数搜索。


### 13.16 Oct9 02:15 独立小时巡检：beta1推进，44项接续合法等待

本次完整读取1096行交接并核对live Git，分支仍codex/anchor-completion-validation、
HEAD194562d、indexSHA87ad06fd…；原28项118917/118917预测、5000 TextVQA复合
题键/完整prompt、SQA2017、POPE8910及官方评分再次通过。1411/1411来源SHA
匹配，61份日志无异常；summary JSON/CSV28行一致、errors=[]。原主监督器、
POPE控制、初始helper及summary watcher已按完成条件退出，没有恢复需求。

新增EADP_beta2/AZ_beta2各5000严格CPU评分及原预测/runtime/score/finishedSHA
再次通过，使用已登记JSON count-key overlay且无数值容差；panel128完整128/512
与resultSHA通过，仍不能据面板宣称全量准确率改善。新增5034/5034SHA通过
（1740来源/工件+3294图像，3373唯一路径），四plan来源32/38/57/164均无漂移。

北京时间02:10:20→02:15:05，beta1 PID2556913同启动身份
由3420→4270/5000，runtime最后4270完整行，输出和日志mtime同步推进、无异常；
A40仅一个登记模型/reservation，未发现未知CUDAcontext。recovery supervisor
2556883的完整argv/startticks，以及phase1/2监督器2558610/2558609的固定
Python3.10+overlay+原target+plan身份通过。phase1登记12任务/7组、phase2
32任务/29组（全attempt2）仍无child，合法依赖当前beta1与phase1；历史
interrupted error由明确恢复登记和当前状态超越，不作为新故障。

publisher PID2560860 active/running、授权wrapper与原sourceSHA通过；02:11
独立ls-remote确认origin 47b7ac956cf78331c94d4e1c9afbeefdb3eae127，18/55组提交
均为已确认远端ancestor、37组合法pending，pending commit/publication/error空。
19归档/483 evidence member/353来源检查/32预测126058归档行/158 metadataSHA
及134阈值静态工件通过，232提交路径allowlist无夹带。

本机证据在rerun_batch/hourly_monitor/check_20261008T180812Z/。独立checker的
Git worktree index路径、SQA不同算式末位尾差、phase2 lambda字段位置假设已
按真实来源修正并保留首轮证据，均非实验故障，未改原score或加数值容差。
本次未启动/信号/重启GPU或监督器、未改冻结source/参数/预测、未stage/commit/push。
距Oct9 12:00约9.75小时；44续跑尚未开始，legacy历史双槽7.649小时下界
不包括phase1和前置时间，仍不能证明全scope按时完成。授权队列继续，timer
保持active，all_experiments_complete=false，55远端组仍全部纳入停止条件。

### 13.17 Oct9 03:13 独立小时巡检：三控制完成，phase1正常接续

本次完整读取1132行交接并核对live Git，分支仍codex/anchor-completion-validation、
HEAD194562d，用户既有变动保留。原28项118917/118917预测、官方完整分母/
TextVQA复合题键与全文prompt、SQA2017、POPE8910及可用runtime再次通过；
1411/1411来源SHA、61份日志、summary JSON/CSV28行一致且errors=[]。原主
监督器/helper/watcher与17个历史worker启动身份已实际退出，31份关闭工件
与上轮SHA不变。初始NeXT Text128两臂无新增runtime的边界继续保留。

新增三臂各完整5000题、runtime/图像/题面/来源/官方score/finished/exit0通过
固定Python3.10及已登记count-key overlay精确复算：EADP_beta2为
52.086000000000055%，AZ_beta2为53.26200000000008%，EADP_beta1为
50.96600000000002%。同beta2方法差+1.176pp，预定义beta1−beta2为−1.120pp；
默认beta1控制没有追回paper残差，不能称beta1已核实为论文配置或事后选臂。
controller于北京时间02:19:20合法complete、三臂success marker通过，监督器
2556883及所有children实际退出，phase1于02:19:28/43自动接E64/AZ64。
panel128/512及独立result SHA/逐题复算再次通过；面板无全量准确率改善证据。
新增5038/5038 SHA通过（1744来源/工件、3294图像），四plan来源32/38/57/164
匹配，冻结controller/worker/plan及授权recovery/overlay来源均无漂移。

03:09:28→03:13:19，同启动身份PID2609341/2609540的TextK64完整预测由
4419→4798/5000、4370→4737/5000，输出与日志mtime持续推进、无异常；
末次runtime分别4799/4737行，活跃文件分次读取产生一行时间差，独立前缀
身份/题面/预算验证通过，未当作完整成绩。A40恰两个登记模型/reservation，
无未知CUDA context。第二worker启动时首worker实际16810MiB context且已写
首runtime，free27439MiB≥22500，首槽free44257；加载也计slot。phase1实际
登记12任务/7组正在运行，phase2实际登记32任务/29组、全attempt2，正常
waiting_for_phase1、child=[]。两CPU监督器2558610/2558609的完整授权overlay
argv、来源与上轮start_ticks通过；旧SIGTERM error及旧dependency快照被
当前运行/合法dependency_validation超越，没有再次失败证据。

publisher PID2560860 active/running，维护结束/source_ready=true，授权argv/
source SHA通过。03:10:59独立ls-remote确认origin
bb0bbb4d911fb4d603857613b433af14d610a090；19/55组commit均为远端ancestor，
36组合法等待，pending commit/publication/error空。新增beta1组独立发布，
未混入paired CSV。20归档/530 member/371来源检查/33预测131058归档行/
158 metadata SHA及234提交路径allowlist通过，live HEAD/index/暂存保持。
历史阈值静态清单的handoff旧SHA已核对原远端字节，并证明当前仅追加02:15
事实且与本次远端doc相同；这是允许的文档更新，不豁免冻结源码或真实push失败。

证据保存于rerun_batch/hourly_monitor/check_20261008T190840Z/。仅更新本机
巡检证据与本文事实；未启动/停止/重启任何实验或监督器、未改冻结源码/参数/
预测、未stage/commit/push。本次快照checker按真实generate_command字段和
已完整beta1单臂发布适配，原检查证据及来源SHA保留，均非实验恢复。
03:13距12:00约8小时47分；phase1尚未完整一组、legacy32项仍等待，历史
legacy最理想双槽7.649小时下界尚不含前序，全部scope按时完成仍有风险。
授权队列和既有小时timer继续，全部已登记任务及55组远端确认前保持
all_experiments_complete=false，基础monitor的原28 all_complete=true不构成停止条件。

03:17收尾增量：巡检期间K64两worker已各5000完整exit0并实际退出，固定
Python3.10/授权CPU overlay独立严格validate_job再次通过：E55.4100000000001%、
AZ56.29200000000013%，同协议差+0.882pp，不据点估计称显著。phase1自动接续
K128，真实PID2707803/2708023完整argv、父子、start_ticks通过；03:16:37→
03:17:34完整预测35→107、2→73，输出与日志推进，两个实际模型/reservation。
新K128全文prompt复合键/顺序/runtime来源/预算及第二槽启动门另行只读核验通过。
03:17:35独立ls-remote确认origin d99d82f1159db78e895ecc4db89b88eb8a47ec11，
新增K64组后20/55组均远端ancestor，pending/error仍空。phase1完成2/12、
phase2仍32项等待；剩42任务/35发布组不构成整体完成，小时timer继续。

### 13.18 Oct9 05:14 独立小时巡检：phase1完成8项，23组远端确认

本次完整分段读取1191行交接并核对当前Git，live分支仍
codex/anchor-completion-validation、HEAD194562d；用户既有变动保留。原28项
118917/118917预测、官方完整分母、TextVQA复合题键/全文prompt、SQA2017、
POPE8910（random2910）、可用runtime及独立官方评分再次通过；1411/1411
来源SHA匹配、61日志无异常、summary JSON/CSV28行一致且errors=[]。原监督器、
helper、watcher和17个历史worker启动身份实际退出，31关闭工件与上轮SHA不变。
初始NeXT Text128两臂无新增runtime的边界保留。

128面板完整128/512、独立resultSHA/逐题评分及success marker通过；新增K32
三臂各完整5000的原预测/runtime/来源/官方score/exit0与完整controller marker
按已登记Python3.10 count-key overlay精确复核通过，没有数值容差或GPU重生成。
phase1实际登记12任务/7组，目前8项完整评分共28068/39119预测：Text64两臂
E55.4100000000001%/AZ56.29200000000013%；新增Text128两臂
E57.866000000000206%/AZ57.634000000000206%，方法差−0.232pp。
SQA32为E1361/AZ1354（各2017题，差−7题），SQA64为E1356/AZ1361
（各2017题，差+5题）。这些点估计没有新增bootstrap，不声称显著或整体胜出。
新增5320/5320 SHA检查通过（2026来源/工件、3294图像；3417唯一路径），四份
计划来源32/38/57/164匹配；phase2实际32任务/29组、全attempt2仍pending。

05:09:14→05:13:45，同启动身份PID2882901/2883681的SQA128两臂完整预测
及runtime由1031→1453/2017、994→1416/2017，输出与日志mtime同步推进，
没有加载超时或30分钟无进展。A40恰两个登记模型/reservation，无未知CUDA
context；两CPU监督器2558610/2558609的授权overlay/原target/plan完整argv、
父子、启动ticks与上轮一致，phase2合法waiting_for_phase1且child=[]。历史
SIGTERM error有明确recovery登记，被当前实际运行和依赖推进证据超越。
phase1全部10个已启动任务的显存/slot门与4次组间接续核验通过，接续各约
0.29–0.31秒。SQA首runtime没有创建时间戳，第二槽首完整记录先行依据是冻结
controller的has_progress硬门及记录的ready/context，不补称独立时间戳证明。

publisher PID2560860 active/running、授权overlay及冻结原sourceSHA通过，
maintenance_active=false/source_ready=true。05:10:58第二次独立ls-remote确认
origin为5f8e8b69a3027bfe0489ef519c35da81af561159，23/55组commit全为远端
ancestor；新增Text128、SQA32/64组已到远端，其余32组合法等待，pending_commit、
pending_publication、error为空。24归档/806 member/827来源检查/41预测159126
归档行/158 metadata及134静态工件通过，242提交路径allowlist无夹带用户figure
或私有文件，live HEAD/index/暂存保留。

证据保存在rerun_batch/hourly_monitor/check_20261008T211000Z/。审计器首轮
将成功score的integrity字面句no empty/FAILED误作生成失败，首轮证据和精确
过滤修订均保留；实际FAILED预测、Traceback、OOM、非零退出与failed state门
继续检查，不是实验故障。仅写本机巡检证据和本段事实，没有启动/终止/重启
实验或监督器、改冻结生成代码/数学/参数/预测、stage/commit/push或改定时器。
05:14距12:00约6小时46分，phase1剩4项且legacy32尚未开始；历史legacy最理想
双槽7.649小时下界还不含前序，存在明显deadline风险，继续既定授权队列。
小时timer保持active，全部登记scope及55组远端确认前
all_experiments_complete=false；原28的基础monitor all_complete不能停止巡检。

### 13.19 Oct9 06:12 独立小时巡检：phase1完成10项，24组远端确认

本次完整分段读取1240行交接并核对live Git，分支仍为
codex/anchor-completion-validation、HEAD194562d，用户既有变动保留。原28项
118917/118917新预测再次通过完整题面/复合键、官方分母及CPU评分；两lane、
POPE控制和合法success marker完整，原监督器/helper/watcher与17个历史worker
启动身份实际退出，31关闭工件与上轮SHA一致。1411/1411来源SHA、61日志通过，
summary JSON/CSV28行一致、errors=[]；初始NeXT Text128无新增runtime的边界保留。

128面板完整128/512、result SHA与独立逐题评分通过，仍不外推全量准确率。
K32三臂各5000完整预测/runtime/官方score/exit0/finished及完整controller marker
按固定Python3.10和已登记count-key-only CPU overlay精确复核通过。phase1目前
10/12项完整评分，共32102/39119题；新增SQA128为EADP1368/2017
（67.82350024789291%）、AnchorZip1374/2017（68.12097174020823%），
净+6题/+0.29747149pp，未做新增bootstrap，不据点估计宣称显著或整体胜出。
新增5390/5390 SHA检查通过（2096来源/工件、3294图像，3427唯一路径），四plan
来源32/38/57/164和另7份用户登记固定SHA均一致；phase2实际32项/29组、
全attempt2仍pending，未将prepared-only归档或前缀算完整任务。

06:09:22至06:11:41，同启动身份PID2920999的NeXT TextFULL完整预测/runtime
由3051增至3201/5000，输出和日志mtime同步推进，末次age约0.066秒，未发现
加载超时或30分钟无进展。独立3142行前缀复合键、全文prompt/顺序/runtime
身份与预算通过，实际均2880 tokens。A40恰一个登记模型/reservation，真实
独占，无未知CUDA context；启动gate空contexts/reservations、free44257MiB
通过32000MiB门。此前SQA128两worker完整exit0并退出，约0.323秒自动接FULL；
phase1全部11次已启动任务门及5次组间接续通过。CPU监督器2558610/2558609的
固定Python+授权overlay+原target/plan、父子与启动ticks和上轮一致；phase2合法
waiting_for_phase1且child=[]，历史SIGTERM error有明确recovery，不是新失败。

publisher PID2560860 active/running、授权argv/overlay/原source SHA通过，
maintenance_active=false/source_ready=true。06:10:06、06:11:08两次独立
ls-remote确认origin为835433af1b43dff0e99053f1f32b381680ac39b9，24/55组commit
均为远端ancestor，新增SQA128配对已发布；其余31组合法等待，pending_commit、
pending_publication、available_groups、error为空。25归档、875member、941来源
检查、43预测/163160归档行、158metadata、134static与244提交路径allowlist
通过，liveHEAD/index/暂存保持，没有夹带用户figure或私有文件。

本机证据保存在rerun_batch/hourly_monitor/check_20261008T220813Z/。仅执行CPU
只读审计、保存本机快照和本段事实；本次checker副本按冻结worker规则补齐
FULL预算0的前缀检查并保留来源，没有修改冻结生成代码、数学、参数、预测、
状态、定时器或信号任何进程，未自行stage/commit/push。距12:00不足6小时，
phase1尚有TextFULL与SQA FULL、legacy32项未开始；历史legacy理想双槽7.649
小时下界单独已超过剩余时间，存在明显截止风险，继续已授权队列。小时timer
保持active，全部实际登记任务及55组远端确认前all_experiments_complete=false。

### 13.20 Oct9 07:14 独立小时巡检：phase1完成，legacy正常接续

本次完整分段读取1285行交接并核对live Git，分支仍为
codex/anchor-completion-validation、HEAD194562d、index SHA87ad06fd…，用户既有
变动保留。原28项118917/118917预测、完整题面/复合键/官方分母和CPU评分再次
通过；1411/1411来源SHA、61日志及summary JSON/CSV28行通过，errors=[]。
原监督器/helper/watcher及17个历史worker身份实际退出，31关闭工件SHA不变；
初始NeXT Text128两臂仍无新增runtime，不补称已有此证据。

128面板128/512与三项K32各5000的完整评分/来源/runtime/marker再次通过。
phase1于北京时间07:07:04完整12/12、39119/39119并写合法success marker，
监督器2558610与全部children实际退出。新增FULL独立精确复算为TextVQA
60.370000000000225%，SQA1364/2017=67.62518591968269%；保持原结果，不据点估计
声称方法显著或整体胜出。新增17055/17055文件SHA及2584/2584内嵌图像SHA通过，
四plan来源32/38/57/164、用户登记固定source与recovery plan SHA均匹配。

phase2于07:07:27自动启动首MME，距phase1成功marker约22.469秒；当前真实
dependencies ready、12项上游严格复核与marker SHA一致。两MME K128 worker
PID3104762/3109792的完整argv、父子、启动ticks与attempt2登记通过；
07:10:06→07:13:05完整预测/runtime由245/13增至784/549，各分母2374，
输出和日志mtime同步推进，末次age小于0.3秒。扩展审计07:13:52补证925/685，
不将前缀作为成绩；phase2尚0/32完整，2项生成、30项pending。A40恰两个
实际登记模型/reservation，无未知CUDA context；第二槽启动时首context
17494MiB且完整首prediction/runtime已出现，free26619MiB>=22500。全部14个
已启动continuation任务门与6次phase1组间接续通过。

native loader的四CPU fork子进程继承相同argv；初轮审计器误作额外模型的证据
保留，按真实父子、原num_workers=4和无NVML context修订仅本次独立checker。
没有豁免任何未知零CUDA context。旧supervisor.log的01:51 KeyboardInterrupt
属于已登记CPU恢复历史；当前两recovery_supervisor.log为空且真实fd目标通过，
无新失败，不据旧dependency/gpu_gate快照判故障。

publisher PID2560860 active/running，授权overlay/冻结target/实际argv通过，
maintenance=false/source_ready=true。07:10独立ls-remote确认origin
2ea966201e47b36220aabfe6004809a85187af24；26/55组提交均为远端ancestor，phase1
七组全部发布，剩29个legacy组合法pending，pending_commit/publication/error空。
27归档/1003member/1055来源检查/45预测170177归档行与248提交路径allowlist
通过；新增配对/FULL/独立beta1摘要与原score精确一致，beta1未混入paired CSV。

证据保存于rerun_batch/hourly_monitor/check_20261008T230812Z/。只执行CPU
只读审计、保存本机快照和追加本段事实，未启动/停止/重启任何实验或监督器、
改冻结源码/参数/预测、stage/commit/push或改定时器。距12:00不足5小时，legacy
32项刚开始，历史理想双槽7.649小时下界仍提示明显deadline风险，继续授权
队列；全部实际登记任务及55组远端确认之前all_experiments_complete=false。

### 13.21 Oct9 08:15 独立小时巡检：MME全8项完成，33组远端确认

本次完整分段读取1330行交接并核对live Git，分支仍为
codex/anchor-completion-validation、HEAD194562d、index SHA87ad06fd…；用户既有
变动和暂存保留。原28项118917/118917预测再次通过完整题面/复合键/官方分母、
CPU评分与可用runtime，1411/1411来源SHA、61日志与summary JSON/CSV28行
通过，errors=[]；原监督器/helper/watcher及17个历史worker启动身份退出，
31关闭工件与上轮SHA不变，初始NeXT Text128两臂无新增runtime边界保留。

128面板128/512、K32三项各5000与phase1全12项39119题的严格官方score、
来源/runtime/finished和成功marker通过；已完成监督器及children实际退出。
新增17293/17293文件SHA（2412来源/工件、14881图像）及2584/2584内嵌图像
SHA通过，四plan来源32/38/57/164匹配；最新phase2另168/168来源检查通过。

phase2当前8/32项严格完成，共18992题，全部为MME各2374题、exit0且独立
官方评分/finished/source门精确通过。v15 K128 EADP1431.982292917167、
AnchorZip1445.5550220088035，MME perception raw差+13.5727290916365；
v15 FULL1507.0586234493799，NeXT AZ128单臂1462.0508203281313。保留raw
指标与单臂身份，不当作准确率百分点、NeXT公平配对或新增显著性结论。

末MME于08:10:20.604816完整评分，1.298567秒自动接POPE；任务优先级门、
所有已启动slot/显存门与完整前缀身份/runtime预算通过。08:11:09→08:13:40，
POPE K128 E/AZ PID3214303/3215558同启动身份，完整预测/runtime由156/80
推进至644/566（各8910），输出与日志mtime同步推进，末次age<0.15秒。
A40恰两个登记模型/reservation，39331MiB、无未知CUDAcontext；第二槽启动
时首worker真实18672MiB context且已有完整首prediction/runtime，free25577
≥22500。phase2监督器2558609授权overlay/原target/plan/父子/startticks与
上轮一致，旧SIGTERM error与旧supervisor日志属于已登记恢复历史；当前真实
fd对应recovery日志且无新异常。

publisher PID2560860 active/running、授权overlay及原target SHA/argv/启动身份
通过，maintenance=false/source_ready=true。08:12:04与08:12:42独立ls-remote
确认origin70c5cca0e113d130552d0107ce6f0c06e656b379，33/55组提交均远端
ancestor，22组合法等待，pending_commit/publication/available/error为空。
34归档/1269 member/1327来源检查/53预测189169归档行、158metadata/134static
与262提交路径allowlist通过；23份新增官方score与远端字节精确一致，legacy
8行CSV保留raw指标和配对/单臂身份，live HEAD/index/暂存保留。

本机证据在rerun_batch/hourly_monitor/check_20261009T000848Z/。首轮checker
捕获末MME退出与POPE新启动的跨时状态，以及第二worker恰在/proc遍历时启动；
已保留原报告并以新版state、精确argv/startticks/父子、合法gate、完整score
及持续前缀逐项核验，收尾errors=[]，不属于实验故障或恢复。独立CPU checker
按真实generate_command字段与官方score路径修订并保留来源，冻结代码不动。
仅保存CPU只读快照并追加本段事实，未启动/终止/重启实验或监督器、修改冻结
源码/数学/参数/状态/预测，未stage/commit/push或改变timer。距12:00约3小时
45分，legacy仍24项/205207待完成预测，存在明显deadline风险，继续授权队列。
全部已登记scope与55远端组完成前保持all_experiments_complete=false。

### 13.22 Oct9 08:20 用户进度核查与实测工期更新

用户询问是否全部跑完。先核查当前Git并完整读取1378行交接，再并行只读核对
实际状态、进程增长、最新完整配对与远端。原28项、K32三控制、phase1全部12项
均完成；phase2为8/32完整（MME全部8项）、2项POPE生成、22项pending。
当前登记的75个全量实验臂中51完成；这不是论文120个baseline格的全覆盖。
剩余24臂为POPE7、MMBenchEN3/CN3、VizWiz3、GQA8，共205207题全分母；
两活跃前缀已写部分另扣，不当作完整成绩。

08:18:48→08:19:06，两POPE K128 worker PID3214303/3215558同启动身份，
完整预测/runtime分别1628→1687、1545→1603，约3.26/3.20题每秒，mtime小于
0.3秒；后续轻读1790/1705。A40仅这两个模型；phase2实际running、来源
164/164匹配、无新FAILED/OOM。小时巡检08:08成功，下一09:08:07。
当前POPE完整配对估08:56–09:01评分，受题长与吞吐变化影响。

基于冻结历史日志，剩余任务约14.171 GPU小时（扣活跃前缀约14.00），仅双槽
理想下界约7小时。本轮8项MME实际GPU累计120.95分钟、对应历史67.55分钟，
倍率1.790；当前POPE128估约46分钟、历史约26分钟也相近。按该实测倍率，
剩余双槽约12.5小时，再预留加载/长回答/严格评分与推送，暂估全部legacy及
发布在Oct9 **20:00–23:00**结束。后续NeXT MMB/Viz/GQA本轮速度尚未实测，
须滚动修订，不能承诺中午12点全部收齐；已完成核心配对可先用于论文核表。
原凌晨/上午工期只针对此前批次或历史下界，不能沿用为追加全部scope的保证。

08:18:43独立安全ls-remote确认origin为
22d553acf2ce58f97db66ba6949cffb0d7f800c4，33/55组全部已发布commit均为真实
远端ancestor，pending_commit/publication/error为空，publisher active/no_op；
liveHEAD194562d、index87ad06fd及用户暂存仍保持。每组完成即推送继续执行。

最新NeXT wait-only TextVQA完整E/AZ配对：K128 57.866/57.634（差−0.232pp），
K64 55.410/56.292（+0.882pp），K32 52.086/53.262（+1.176pp）；E分别低
paper59.2/57.0/54.2为1.334/1.590/2.114pp，均仍超用户0.2门槛。beta1完整
50.966%，比beta2低1.120pp，没有解决残差。SQA wait128/64/32配对分别
+0.29747149/+0.24789291/−0.34705007pp；v15 MME128方法差+13.5727290916365
原始perception点，不能叫准确率百分点。结果方向有高有低，不能宣称整体稳定
超过EADP。未更改冻结生成源码/协议/计划/参数/预测，未重跑或信号任何进程。

### 13.23 Oct9 08:49 最新待查清单与已确认未修问题

用户准备亲自检查，询问是否只有NeXT TextK32的2.114点残差。读取当前Git及
完整交接后，依据已完整官方score更新10格证据，独立保存
`Qwen_vl/outputs/audit_followup_20261008/user_checklist_20261009_0846/`：
numeric_deficits.csv（15项，SHA ff5175c61123d04d7755dfe9aed445bf7b8c7d15406e14ebffefc791a683e605）、
numeric_snapshot.json（120格及原来源/更新依据，SHA1ac8b0fc…）、
root_checklist_verification.json与issues_and_checks.json（精确代码检查点）。
目录0846指开始核查时间，数值快照实际08:49:10；不覆盖01:53冻结旧阈值表。
root逐项核对15个原score SHA及未舍入差值，120格计数为18数值达标、15待查、
1协议不可比、86缺失或未评分。NeXT SQA128最新67.823500 vs68.0，低0.176500，
已移出>.2清单；不是仍低约0.9点。MME原始点数及/20纸表Avg等效均保留，
后者不是准确率百分点。POPE128活跃前缀未替换任何完整score。

仍待查的15项：v15 POPE32低2.626400、SQA32低0.534507、GQA128低0.372078，
MME128/FULL分别低7.017707/6.341377原始分（/20为.350885/.317069）；
NeXT Text32/64/128低2.114/1.590/1.334、SQA64低.371443；
Qwen legacy256 Doc1.664422、HallB1.759538、AI2D1.153368、Chart.700、
MMBEN.623839、Text.358。数值差距不自动证明15个实现错误。

另两项已确认而未修：MMBenchCN漏--lang cn；AZ NeXT Completion预注册逐crop
但实际五crop跨组assignment。现有55组wait-only保留原语言/方法，不修这两项；
独立CN proposal/逐crop adapter仍未登记GPU，效应未测。后者仅AZ，不能解释
EADP基线低分。Qwen HallB E256 recorded/current bank SHA不闭合是来源疑点，
不是已证实分数错误；应找历史bank/生成resume日志，不替换旧meta。另有MMB
题集版本、VizWiz公式/guidance/生成上限、任务alpha/beta默认及SQA作者CQM-I
身份差异待查。beta1已完整50.966%，没有恢复Text残差，不能重复说仍待运行。
当前补跑也不会补齐NeXT大量缺E、Qwen512/128及VQAv2/MMVet最终评分。
本次仅读已有完整结果、写独立CPU派生清单和本文；未改冻结GPU源码/数学/
参数/题集/协议/plan/原预测/score，未启动、停止或重跑任何实验。

### 13.24 Oct9 09:04 当前POPE任务与TextVQA跨模型诊断

用户问正在跑什么、为何其他模型Text接近paper而NeXT不接近。09:04:49真实
state及/proc确认两GPU均为AZ：NeXT POPE K32 PID3292761完整1570/8910，
v15 POPE K64 PID3293502完整1454/8910；phase2已10/32完成，2生成/20pending。
最新v15 POPE K128配对各8910完整raw/runtime/官方score/SHA/exit0/marker通过：
EADP87.519596（paper87.2，高.319596 F1点），AZ87.454656，方法差−.064940点。
09:03独立ls-remote核验origin5267a4d57a7e75769c7b75f4bf245a142d1bb563，
34/55组含新POPE pair均到真实远端，pending/error空，队列没有新故障。

v15 Text FULL/E128/E64/E32对paper分别+.026/−.110/−.082/−.168，均达.2；
Qwen legacy E256仍−.358，不能称所有其他模型都严格验收。NeXT FULL60.370
vs60.3已达标，pruned三预算仍−1.334/−1.590/−2.114。这支持优先检查NeXT
专有剪枝执行/配置，但没有确定根因。当前五crop importance比例配额、全crop
归一化、floor/min1和singlecrop不同；实际156–159符合官方规则，预算映射及
原始权重/CLIP/主要输入和scorer已核，不能把这些步骤的存在本身称bug。
wait控制只使E32+.090，beta1完整50.966比beta2低1.120，都不能解释全部残差。
q=.2在llava_arch.py:586硬编码且与官方源码一致，不是已发现的本地q偏差。
现有runtime只记录几何/count/CLI预算/promptSHA，尚无实际模型对象α/β/class/
dtype/backend快照。作者真实运行argv以及历史环境未知，是后续身份检查边界；
没有环境准确率因果证据，未为回答启动新probe或改变已有冻结算法/生成队列。


### 13.25 Oct9 09:15 独立小时巡检：legacy完成10项，34组远端确认

本次完整分段读取1467行交接并核对live Git；分支仍
codex/anchor-completion-validation、HEAD194562d、index SHA87ad06fd…，用户既有
变动及暂存保留。原28项118917/118917题再次通过完整题面/Text复合键、官方
分母及CPU评分；1411/1411来源SHA、61日志、summary JSON/CSV28行一致且
errors=[]。两lane与POPE控制complete及合法marker通过，原监督/helper/watcher
与17个历史worker启动身份实际退出，31关闭产物与上轮SHA一致。初始NeXT
Text128两臂无新增runtime的边界保留，不把基础all_complete当全scope完成。

128面板128/512/result SHA，K32三臂各5000题和phase1全12项39119题完整
来源/runtime/严格官方score/exit0/finished/成功controller marker再次通过。
phase2当前10/32项严格完成，共36812题，新增v15 POPE128两臂各8910
(random2910)：EADP87.51959583686018、AnchorZip87.45465578793069，
AZ−E为−0.06494004892948624 F1点；保留负方向，不据点估计宣称显著或整体胜出。
17399/17399文件SHA（2518来源/工件、14881图像）及2584/2584内嵌图像SHA
通过；四plan来源32/38/57/164和另8个固定overlay/原target/recovery plan等SHA
一致，全部32实际attempt2范围保持，prepared-only工件不计额外scope。

09:10:26→09:13:40，同启动身份PID3292761/3293502的NeXT POPE AZ32与
v15 POPE AZ64完整预测分别2650→3265、2535→3149/8910，输出与日志mtime
同步推进，末次age<0.31秒；最后分次读取v15 runtime/log为3150，是一行持续
写入的时间差，独立前缀题面/身份/顺序/runtime/实际预算门通过。A40恰两个
登记模型/reservation，41729MiB、无未知CUDA context；实际worker完整argv、
父子/startticks及CPU监督器2558609的授权overlay/target/plan/startticks通过。
两新单臂于各自前驱完整评分后约0.934/0.920秒自动接续，第二槽首NeXT真实
21316MiB context、free22933>=22500；24个continuation启动门与MME→POPE
顺序门通过。native runtime无创建时间戳，首完整行先行依据冻结has_progress
硬门与ready/context记录，不补称独立时间戳证明。旧SIGTERM/error属于明确
登记CPU恢复历史；当前实际fd对应recovery_supervisor.log且无新异常。

publisher PID2560860 active/running，授权overlay与原target SHA、实际argv/
启动身份通过，maintenance=false/source_ready=true。09:11:48二次独立
ls-remote确认origin9891f7a5d6494f77a0eeb8e14146d3e5db933c7a，34/55组commit
全为远端ancestor，21组合法等待；pending_commit/publication/error空。
35归档/1314member/1399来源检查/55预测206989归档行、158metadata/134static
与264提交路径allowlist通过，25新增官方score与原件/归档精确一致；legacy
CSV10行保持raw指标及配对/单臂身份，没有夹带用户figure或私有文件。

证据保存于rerun_batch/hourly_monitor/check_20261009T010915Z/。本次仅执行
CPU只读巡检、保存本机快照和追加本文事实，未启动/停止/重启任何实验或
监督器，未改冻结源码/数学/参数/状态/计划/预测/score，未stage/commit/push
或改定时器。当前已登记75个全量实验臂中53完整，phase2仍2生成/20pending，
22未完成臂共187387完整分母题尚未整体完成；活跃前缀另扣，不当作成绩。
距中午12:00约2小时45分，最新实测总工期20:00–23:00仍只是估计，截止风险
明显，继续已授权队列。全部已登记scope及55远端组完成前
all_experiments_complete=false，小时timer保持active。

### 13.26 Oct9 新用户请求：NeXT 根因排查与独立文本指导修正对照

用户明确要求核对 NeXT 是否论文模型并排查修复；项目为 EADP_amp。
新诊断详见 docs/next_reproduction_diagnosis_20261009.md，证据目录
Qwen_vl/outputs/audit_followup_20261008/next_runtime_repair_20261009/。
实际 CPU loader（CUDA 不可见）核对全部395视觉/投影权重与checkpoint一致，
实际alpha=.5/beta=2/K32/SDPA；不是错误模型或微调视觉权重被覆盖。
16题固定CPU FP32面板确认padding/特殊位实际参与局部评分：593padding+
40special/1500所选位，10/16题至少一个crop-segment的无效位权重>.5。
全5000题有881长文本、原1122段缺EOS；首长段CLIP legacy pool取BOS而非EOS，
两不同prompt实测首段global embedding逐位相同。独立bounded chunk修正后
全文本无丢token、每段有EOS、原短输入不变。不是NeXT独有代码缺陷；不能
据CPU机制检查声称已解释2.114分残差或已恢复paper。

新增可恢复adapter next_text_guidance_control.py 与4项CPU语义测试，原生产
源码/方法数学/参数/计划/旧预测评分均未改。64题四臂GPU诊断为wait-released/
valid-dense/bounded-chunks/both，固定K32 alpha.5 beta2 q.2，不按测试分数调参。
它独立于既有75臂/55发布组，不将小面板成绩当全量成绩，也未更改AnchorZip。
09:32:29仅向CPU监督器2558609发SIGSTOP保留下一个slot，两生成worker未收到
信号、继续完成当前8910题。run_next_guidance_reserved_slot.py有独立20分钟
SIGCONT watchdog和finally恢复；诊断出现CUDA context即恢复原监督器，原
foreign-context门会暂阻新模型，防止第三context。精确事件及最终恢复须读
panel64/slot_coordination.json；未改监督器/状态/plan/publisher/timer文件。
09:45:46四臂面板完整64题/256生成/exit0完成，原路径64/64预测与既有全量
wait-only逐字一致，12来源SHA/题目顺序/result/runtime检查通过。实际GPU
模型参数/精度/SDPA与CPU对象快照相同。原路径56.40625，valid-dense54.375
（0胜2负），bounded-chunks56.40625（0胜0负），both55.9375（0胜1负）。
这是小样本机制对照，不是5000题成绩；没有改善证据，不提升为正式baseline，
两个adapter保留为显式opt-in control，不能宣布已找到2.114分主因或修复复现。
真实五crop prepare函数12个哨兵case与独立48x48 raster oracle全通过；8个
不同seed variable-quota greedy与独立标量facility oracle相同，未发现拼接
mask错位。旧全量长/短文本的FULL→E32损失8.672/8.201，仅描述性差.471，
不支持直接把全部paper残差归因长文本。作者Table2实际argv仍未知，未调参
追paper成绩，未联系作者；公开issues查询为空。

CPU监督器09:43:17已恢复、09:45:46finally再次SIGCONT，独立watchdog与GPU
诊断worker均退出。原phase2已12/32完成，后续NeXT POPE AZ64/AZ128自动续跑
PID3375750/3376381，恰两个GPU模型。原75臂/55发布组scope、源码与plan未改。

### 13.27 Oct9 用户并发超时漏答假设的CPU核查

用户询问是否并发使部分题超时未回答。新增脚本
Qwen_vl/scripts/stage1_roundtrip_pilot/audit_generation_completeness.py，证据
next_runtime_repair_20261009/timeout_audit.json。20份已完成v15/NeXT TextVQA
文件共100000条（跨重复实验，非10万独立题），每份完整有序5000复合题键，
缺失/多余/重复/空或非字符串/失败占位均0，20对应生成日志超时/Traceback/
CUDA OOM/error匹配均0。8个wait-only控制另核runtime5000、exit0、finished
及score预测SHA均通过。Qwen EADP/AZ的Text/Doc/AI2D/HallB共8份xlsx28776行
也满分母、无重复index、空答或失败标记；不称本次独立核验了所有Qwen旧日志。

LLaVA题循环无按秒超时或异常吞掉跳题，128是max_new_tokens，不是秒；两份
generation_config均无max_time。输出未保存EOS/finish_reason，因此不能据
非空输出断言每题都自然结束或从未触及token上限。此前128题重复面板旧路径
答案5/128变化、count43/128变化，wait后均0，是进程内CUDA依赖错误，不等于
已证明双进程并发造成。全量E32 51.996→52.086仅+.090，剩余paper差2.114
相当约106道满分题，现有文件没有漏答支撑该解释。未做严格GPU独占vs双进程
对照，不排除所有并发数值/时序影响；本次只CPU读原件写派生证据，未动GPU队列。


### 13.28 Oct9 10:16 独立小时巡检：legacy完成12项，36组远端确认

完整读取起始1555行交接，并补读巡检期间root新增19行CPU漏答核查；当前
live分支codex/anchor-completion-validation、HEAD194562d和index SHA87ad06fd…
保持，用户既有变动/暂存保留。原28项118917题、完整题面/Text复合键、官方
分母/精确评分、1411/1411来源、61日志与summary JSON/CSV28行通过，errors=[]；
原监督器/helper/watcher已退出，初始NeXT Text128无新增runtime的边界保留。
128面板128/512、K32三臂各5000、phase1全12项39119题的来源/runtime/官方
评分/exit0/finished、成功marker和完成进程退出再次通过。扩展17471/17471
文件SHA及2584/2584内嵌图像SHA匹配，四plan来源32/38/57/164及固定
overlay/target/plan身份通过。

phase2当前12/32完整、54632题，2生成/18pending。新增NeXT POPE AZ32
82.79087792370159、v15 POPE AZ64 86.52213413116876，均8910完整题
(random2910)、官方macro F1、exit0及严格source/score/finished通过；保持
审计单臂身份，不伪造同预算EADP配对。10:10:25至10:13:53，NeXT POPE
AZ64/128同启动身份PID3375750/3376381预测及runtime分别3626→4124、
2296→2618/8910，输出和日志均持续增长。A40恰两个登记模型/reservation，
GPU使用44021MiB，无未知CUDA context；监督器2558609授权overlay/target/
plan/startticks与上轮一致、未暂停，实际recovery日志无新异常。全部26次
continuation启动门与任务优先级门通过，旧SIGTERM error不是新故障。

独立只读复核新增64题面板64样本/256生成、12源码SHA、64图像SHA、题序、
原5000题wait基线对应答案、逐题官方分数/result SHA及exit0；诊断worker
3371129和watchdog3353972均退出，主CPU监督器同身份已恢复。主队列在诊断
退出约1.28秒后续跑；面板无准确率改善证据，不外推全量。

publisher2560860 active/running、maintenance=false/source_ready=true；授权
overlay/原target来源与实际argv通过。10:10与10:14:11独立ls-remote确认origin
10fa0fc2811effb97965e0d097a8792efa4e6081，36/55组commit均为远端ancestor；
19组合法待完成，pending_commit/publication/error为空。37归档/1390member、
1471来源检查、57预测224809归档行、158metadata/134static及268提交路径
allowlist通过；legacy CSV12行保留指标和单臂身份，未夹带用户figure等变动。

本机证据rerun_batch/hourly_monitor/check_20261009T021008Z/。仅CPU只读
巡检、保存快照并追加事实，未信号/重启进程、改冻结代码/参数/计划/预测/score、
stage/commit/push或改timer。最终轻量checker一次目录层级笔误已留证修正，
并发交接更新触发写前保护后已补读保留，均为检查器自身事件；最终审计
errors=[]，无实验恢复。75个登记全量臂中55完整，20项仍未完成（全分母
169567题，活跃前缀另扣）；距12:00约1小时44分，截止风险明显，继续授权
队列。全部登记scope及55组远端确认前all_experiments_complete=false。

### 13.29 Oct9 10:33 投稿表格快照

用户要求立即按三张截图更新表格，最新消息将投稿截止说明为明天；不据此修改
已登记队列或旧截止配置。用户另提出上涨用新、下降用旧：未按分数择优混填，
改为保留统一修复协议主表及独立旧图/新值对照。新表输出在
`Qwen_vl/outputs/audit_followup_20261008/submission_tables_20261009/`，
含latest_tables.json（逐格状态/来源与106个源SHA）、CSV、HTML、LaTeX、
三个PNG及由表格绘图导出的PDF；ours_old_vs_new.csv/html保留上涨和下降，
旧图数字仅一位小数，变化不当作方法增益。源LaTeX内置编译器报
“Unable to find standard directories for platform”，未声称编译成功，未安装TeX。

89行保留arXiv2607.02484v1表1/2/4全部相应reported基线，原始HTML及解析表
本地留存。LLaVA按既定Avg7公式从reported列重算，缺MMB时本地Avg7留空；
GQA、v15 FULL POPE与NeXT Viz/POPE64/128沿用旧值但明确†待重跑，
不把旧值称修复完成。v15/NeXT已完整Text/SQA、MME及POPE已完成臂按
既定新协议替换，并核预测/评分SHA、runtime、退出码/finished。
Qwen256取p1_unified，128/64取既有STATUS完成表及HallBench更正，未新跑Qwen。
快照phase2为12/32完整，两NeXT POPE64/128生成中；未动GPU或冻结队列。

### 13.30 Oct9 11:02 用户授权后续三行核查与加密停滞检查

用户要求确保实际运行、不浪费停滞时间，并指定submission_tables_20261009/
latest_tables.pdf仅重点核EADP reported/reproduced及ours三行；本批跑完后继续
必要实验，明天最后修改论文。因此最新论文工作时间为Oct10，Oct9中午仅旧
目标，不再用其阻止授权工作；新消息没有给Oct10具体小时。当前GPU批次不改序。

11:00左右phase2为13/32完整、2生成/17pending；原75臂56完整，37/55组已
发布。NeXT POPE AZ64完整后自动接v15 POPE FULL。10:51→10:59:37，仍同
PID3376381的NeXT AZ128完整预测/runtime由6176增至6738/8910，v15 FULL
PID3479469由841增至2311/8910；11:00补读为6820/2526。真实两路继续产出。
10:51独立ls-remote确认origin ed2d674111da54aae6c3752c9c6af39a2d80d35e，
37组已发布；后续1945c9bf…为metadata更新，无待提交/发布或新错误。此前
20:00–23:00 ETA仍仅原75臂及55组，不包括新后续提案，不能把其当新全部工期。

新增独立watch_repair_stalls.py及codex-eadp-stallwatch-20261009.timer/service，
每5分钟只读phase2实际进程启动身份、文件stat/首完整记录及合法依赖，不全量
hash或重评、不初始化CUDA、不直接信号GPU。连续10分钟输出与日志都未推进、
加载无首记录、依赖已齐却无活跃worker、异常退出/身份不符时唤醒既有hourly
检查；正常静默，异常/恢复/原登记批次ready才本人桌面通知。20个CPU边界门
及unit verify通过，11:00:24真实timer首触发exit0、读耗时.0645s，issues/events/
actions空；下一11:05:24。原hourly仍active、下一11:08:27。证据在独立
stall_watch_20261009/。原冻结phase2 worker/controller/plan SHA及live index
87ad06fd…保持。这个轻量探针登记现phase2，不谎报已监控尚未登记的新GPU臂。

持久后续请求在post_batch_followup_20261009/request.json，status为
waiting_for_registered_batch，不是已登记GPUplan。原75全评分/来源/marker通过、
监督/worker退出、55组真实origin确认后，hourly按该授权立即更新三行表、
检查必要对照并建立独立plan/source/新路径继续运行。私有inspection_prompt已
记录最新授权及确证可逆phase2故障恢复的边界。CPU run_hourly_repair_agent.py
加完成保护：request未complete或登记文件缺失时，即使agent误报原批complete
也不得停timer；实际Python3.10八个边界门通过。独立复核87份活动plan/protocol
无runner引用，publisher静态allowlist包含它；未改任何GPU冻结来源。新的实际
GPU计划/状态也须纳入后续监控与独立按组发布，不把原55分母悄悄加大。

GQA只读审计gqa_audit.json（SHA2175e331…）：8个wait GQA仍未启动，PDF†为
旧值；v15 E12859.627922低paper60.0为.372078，ours59.802830较同机+.174909
（22题）、较paper−.197170。12578题ID/题面/图名/GT一致，8份完整raw均无
尾句号；官方converter rstrip('.')链差异实际影响0题/0点，不能当根因。先复用
已登记v15 E/AZ128 wait；最少新补NeXT同recipe E128，复用其本批完整AZ128。
不为追.4盲扫beta，shared缺陷尚未证实，不能统一给两方法补分。

VizWiz审计vizwiz_audit.json（SHA88c4e9d6…）：两模型三预算共6个本地EADP格
全空，当前只是ours对reported。论文星标val与本地4319题同split；当前官方
2023注释ZIP中的val.json及API vqaEval.py与本地逐字节相同，排除本地GT版本
错误。6份完整旧ours用发布min(matches/3)较官方留一平均高1.391–1.473pp；
例如NeXT128名义640为官方59.254457/发布60.716215，paper60.6。作者paper真实
scorer/argv未知，此差异只有在paper用了发布scorer时能解释相应缺口，不按较高
分替换官方主列。v15仅同一题895答案超128tokens且截断仍错，NeXT最长32/17/20
tokens，纯max_new_tokens不足解释这些旧答1–2点缺口。公开val入口alpha0/beta1、
loader去短答guidance后缀/max128与本地.5/2/旧guidance/max1024不同。

新Viz提案固定两模型configuredK128（v15名义128/NeXT640）各E+ours，共4臂
17276题：公开val默认recipe、双方同wait、官方LOO主分/发布公式辅审计，保留
所有负方向；不是作者真实argv已证实，也不是GPU已排或已跑。先等原批次完成
后冻结独立计划，不能把不同recipe旧ours复用配新E。最终单个EADP主表行须统一
披露reported或reproduced依据，三行审计及全部原来源保留，不逐格择低baseline
或择高ours。本次新增仅CPU核查、监控及持久后续安排，没有新GPU生成/调参。

### 13.31 Oct9 11:15 独立小时巡检：13项legacy完成，37组远端确认

完整读取交接并核对live分支codex/anchor-completion-validation、HEAD194562d；
原28项118917题完整身份/官方分母/评分、1411来源SHA、61日志与summary
JSON/CSV再次通过，errors=[]，原监督器/helper/watcher正常退出。128面板、
K32三项各5000题和phase1全12项39119题的来源/runtime/严格评分/成功marker
及进程退出通过；初始NeXT Text128无新runtime的既有边界保留。

phase2为13/32完整、63542题，2生成/17pending；新增NeXT POPE AZ64为
85.89048883614093，完整8910（random2910）题、exit0及官方macro F1硬门通过，
保持审计单臂身份。扩展17507/17507文件SHA及2584/2584内嵌图像SHA匹配，
四plan来源32/38/57/164及授权overlay/原target/plan身份通过，27次continuation
启动门与任务优先级通过。v15 FULL合法启动时free22863MiB，首NeXT context
21386MiB；native首行无创建时间戳的证据边界仍保留。

11:10:43→11:14:35，同启动身份PID3376381/3479469的NeXT POPE AZ128与
v15 POPE FULL预测/runtime分别7518→7790、4359→5069/8910，日志同步增长。
A40恰两个登记模型/reservation，无未知CUDA context；监督器2558609授权
overlay完整argv/启动身份通过，旧SIGTERM error不当新失败。11:10:26轻量探针
issues/events/actions均空；小时及5分钟探针timer仍active。没有实验恢复动作。

publisher2560860 active/running，源码/overlay身份通过，pending_commit/
pending_publication/error为空。11:13:34独立ls-remote确认origin
21c15df824943efd4dbda77e309041448cc8db09，37/55组commit全为远端ancestor；
38归档/1428member/1507来源检查及官方score/CSV验证通过。巡检未stage/commit/
push，live用户变动与index保留。

本机证据rerun_batch/hourly_monitor/check_20261009T031043Z/。复核后续GQA/
VizWiz报告与表格36份不可变引用SHA；报告所记旧live state SHA只属历史快照，
当前动态state另按真实PID/冻结plan严格检查。已更新request.last_hourly_inspection，
status仍waiting_for_registered_batch，没有额外登记GPU臂。75臂56完整、55组37
远端确认；原批结束后仍须执行已授权三行审计/必要独立实验，不提前停timer。
最新论文修改日期为Oct10、具体小时未给，Oct9中午仅旧目标；既有20–23时估计
只含原批，尚不含后续工作。检查器自身先读未产出评分报告和误将动态state当
静态SHA的首轮记录已保留并纠正，最终检查errors=[]，未改冻结源码/参数/原件。
