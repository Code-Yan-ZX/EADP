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
