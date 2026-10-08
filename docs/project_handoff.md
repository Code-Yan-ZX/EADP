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
