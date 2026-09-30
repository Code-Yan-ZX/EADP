# Anchor-Merge Pilot — 固定 anchors 的视觉 token 内容聚合（机制验证轮）

**Branch**: `codex/anchor-merge-pilot`（base = `a724122`,worktree
`/media/disk2/YZX/research/EADP_amp`) · **Date**: 2026-10-01（本文件在任何本轮
准确率数字产生之前写定并冻结）

## 0. 状态与边界

- **性质**：机制验证 pilot,不宣称新颖性。"选择后合并"是已有机制（cf. M4 REC
  在同一项目里的负结果）,本轮检验的是一个 M4 未测过的变体：**保留全部 256 个
  anchor 槽位、直接修改 anchor 表示**（M4 是驱逐 anchor 换 capsule）。
- 不训练;不加 OT / PPE / residual rescue / 新 scorer;不改分辨率、prompt、
  生成参数、评分方法。推理 forward-only,pre-LLM。
- 本轮是**执行轮**：实现 → 正确性 → 真实生成评估 → 分析 → 报告。

## 1. 模型与执行路径

- 模型：`Qwen3-VL-8B-Instruct-1024`（VLMEvalKit fixed-res wrapper,1024×1024）,
  与 E0/M12/M13 完全一致。
- 执行：native E0 engine（`model/native_qwen3.py`）—— DeepStack 三路注入开启、
  3D mRoPE、贪心 decode（argmax,`do_sample=False` 等价）。SDPA。
- 压缩点：pre-LLM。视觉编码一次 → 主特征 V [1024, D] + 三路 DS [1024, D] →
  聚合为 256 → prefill 建 KV → 正常 decode。
- 环境：`qwen3vl_clean`（python 3.10.20, torch 2.10.0+cu128, transformers
  4.57.6）,单卡 A40。

## 2. Selector：官方 EADP facility-location（命名澄清）

- 主 selector = **official facility**：`instrumented.TimedEADPPruner._score` +
  `_similarity` + `instrumented.SELECTORS["facility"]`,K=256。在
  `e0_selectors.py` 里对应 arm 名 **`b1`**。
- 这与项目 incumbent **`b2`（block8** = block-parallel facility greedy）**不同**,
  两者不得混写。历史 150-panel 数字（b1 66.88 / b2 64.87）来自旧执行路径,本轮
  不与之直接相减。
- official facility 路径在本轮**可用**（gates 会再次验证）。
- 分数与 anchor 集均用**原始视觉特征**（encode 输出,未聚合）计算。

## 3. 冻结的实验定义

对每张图（N=1024 merged tokens,K=256）：

1. S = official-facility(V, K)（原始特征空间）。
2. S 冻结;顺序 = 原始序列顺序（升序 raster）。
3. **所有 arm 共用完全相同的 S**：由 bank 脚本一次性离线计算、落盘、带哈希;
   arm 只读 bank,不得重算选点。
4. 每个 dropped token i 分配到 argmax_j cos(x_i, a_j)（**主视觉特征空间**,
   FP32 余弦）；平分时取**更小的 anchor 序号**（argmax 首最大语义）。
5. 组 G_j = {a_j} ∪ {分配给 j 的 dropped tokens}。
6. 分组不使用答案、teacher 梯度或生成反馈。

**聚合（每一路特征流独立聚合自身值,但对应关系与权重只由主特征决定一次）**：

- 均匀：m_j = mean(x_i, i∈G_j)（含 anchor）
- 相似度：w_ij = softmax(cos(x_i, a_j)/0.1),仅在 G_j 内归一化（含 anchor,
  anchor 的 logit = 1/0.1）
- y_j = a_j + λ·(m_j − a_j)
- pooling 用 FP32 累加（确定性 sort+cumsum 组和,无 atomics）,结果转回模型 dtype。
- **位置近似（本轮明确声明）**：compressed token 使用对应 anchor 原有的 3D
  位置;文本 token、顺序、其余位置处理与 native 路径一致;prefill 前完成聚合,
  再正常建 KV。不声称保留组内所有位置。
- DeepStack：三路流使用与主路**相同的 token 对应和权重**。

### Arms（DEV,5 个,全部预注册）

| arm | 定义 | 角色 |
|---|---|---|
| BASE | b1 选择,恒等 gather,λ=0（直接走 identity gather 路径,不经聚合代码） | 对照 |
| U025 | 均匀聚合 λ=0.25 | **primary** |
| U050 | 均匀聚合 λ=0.5 | |
| U100 | 均匀聚合 λ=1（完整组均值） | |
| S025 | 相似度聚合 λ=0.25,τ=0.1 | |

本轮不加新的 λ、温度、分组器或位置规则。

## 4. 数据与 manifest

复用 E0 的 image-disjoint 分区（`e0_plan.json`,seed 20260929）。

- **DEV**：每个数据集取 `dev_rows` **前 100** 行（TextVQA_VAL / DocVQA_VAL /
  OCRBench）,共 300。取子集规则在任何本轮分数之前冻结（前缀,不重排）。
  **历史暴露声明**：E0 的 300-row DEV 已被 E0 各 arm 与 M12/M13 的生成评估
  使用过（b0/b2/baselines/retinagate 等）;本轮 5 个 arm 的定义在看到本轮分数
  前冻结,但 DEV 不能称为 untouched,只能称 reuse evaluation set。
- **CONFIRM**：每数据集 200 行,取 `confirm_rows` 前缀（E0 分区本身与 bank
  150、M1 extension、S2-A causal、SAGE confirmation 按 image 排除;E0 记录
  CONFIRM 此前从未被读取;M12–M14 未使用 confirm 行）。OCRBench confirm 池
  仅 164 行 → 不足 200,补 36 行来自 DEV 池中**未被本轮 DEV 使用的后缀**
  （`dev_rows[100:200]` 中与 DEV 前 100 image-disjoint 的前 36 行）。此规则在
  看到任何本轮分数之前冻结;OCRBench CONFIRM 相应标注为部分复用。
- 冻结 manifest：样本 ID、dataset row index、image identity（沿用 e0_plan 的
  image-key 规则）、sha256,落盘 `outputs/anchor_merge_pilot/manifest_*.json`。
- 数量不足时,在看到分数前记录实际数量;不放回凑题。

## 5. 正确性 gates（全部通过后才允许正式准确率运行）

在约 12 个 DEV 样本上：

1. **G1 keep-all 恒等**：engine K=1024/identity 与 stock 模型的 prefill
   logits（full lm_head）+ 固定 32 步 forced decode logits 一致;非零差异须
   定位解释（已知 A2：last-position-only lm_head ~1 bf16 ulp,用 full_lm_head
   对齐后应 bit-exact）。
2. **G2 λ=0 恒等**：合并代码 λ=0 的输出特征与 b1 纯 gather 逐元素相等;
   两路径 prefill/decode logits 与预测一致。
3. **G3 参考实现一致**：聚合结果与逐组 Python 参考实现一致（FP32 容差
   ≤1e-5 相对）。
4. **G4 退化与不变量**：N=K（无 dropped）→ 原样保留;单成员组 → y_j = a_j;
   空 dropped 集恒等;每个 dropped token 恰属一组;输出恰 256 个视觉 token;
   索引/特征流/mask/位置一一对应;layer_calls、cache 长度不变量通过。
5. **G5 可重复性**：同 arm 同样本重复运行（新进程）,聚合特征 bitwise 一致、
   预测一致（M4 曾因 index_add_ 原子归约出现 1.1–1.3 macro 的 run-to-run
   漂移;本轮使用确定性归约并实测验证）。

外加 smoke：每 arm 30 题（3 数据集 × 10）,记录空答案率、截断率、异常重复。

## 6. 执行规模与判定规则（预冻结）

- 先 smoke 估计耗时并记录。
- DEV：5 arms × 300 = 1500 次生成。三任务等权 macro;从 4 个候选
  （U025/U050/U100/S025）选 winner：macro 最高;并列时先 U025,再取实现更
  简单者。**DEV winner 的分数不作为确认结果。**
- **停止规则**：若全部 4 个候选的 DEV macro 点估计均不优于 BASE,本轮不消耗
  CONFIRM,报告记为"未发现值得确认的信号"。
- 若存在正向候选：冻结 winner,CONFIRM 上跑 4 arms × 600：
  BASE / winner / **NORM-ONLY** / **SHUFFLED-DELTA**。看完 CONFIRM 后不得
  换 winner 或调参。

### 机制对照定义（winner 的 Δ_j = y_j − a_j）

- **NORM-ONLY**：每路流独立,y′_j = a_j · (‖y_j‖/‖a_j‖);‖a_j‖=0 时保留 y_j
  （无法缩放零向量,记录计数）。它检验收益可否由范数变化解释;仍可能携带组内
  标量信息,不称完全无内容控制。
- **SHUFFLED-DELTA**：固定种子（20261001）在同图 anchors 间固定随机置换 π;
  y′_j = a_j + Δ_π(j) · (‖Δ_j‖/‖Δ_π(j)‖);Δ 为零向量时 Δ′=0。各流用同一 π。
  报告零 Δ 计数（=单成员组数）与置换后非零率。注意：打乱同时制造分布外扰动,
  winner 胜过它**不能单独**证明"补回答案证据"。

## 7. 评分与统计

- 每题存原始预测、原始答案、官方得分、截断标记,以及 dataset index / row
  index / image identity / sample ID 映射。
- 官方评分语义：VLMEvalKit 官方 evaluator;DocVQA 用正确 ANLS（per-question
  hit = 1 − min-ANLS-distance,阈值 0.5,M12 已修复的规则）;OCRBench 按**实际
  评估题数**为分母（per-question hit 规则重算）,不用 1000 固定分母;headline
  统一 0–100;逐题聚合必须能复现 headline（写进 analyze 的自检）。
- macro 仅在完整三任务面板上计算。
- 不与旧 150/450 或其他执行路径的数字直接相减。
- 统计：同题配对、**数据集内按 image cluster bootstrap**（5000 次,seed
  20261001）重算三任务等权 macro,报告 95% CI;报告每任务得分、macro、相对
  BASE 的 Δ 及 CI、winner 相对两个机制对照的 Δ 及 CI、rescue/break（定义：
  逐题分数跨 0.5 阈值的方向计数;DocVQA 连续 ANLS 同阈值）、空答案率、截断率。
- 300/600 题不保证分辨 0.5–1 点;CI 跨零时写"不确定",不写"相同"。

## 8. 效率

对 BASE 和最终 winner（若存在）配对测量：selector 时间、分组+pooling 时间、
端到端 TTFT、固定 64-token 输出的总生成时间、每 arm 独立重置后的峰值显存。
同样本、交错顺序、15 次预热、CUDA 同步;**每次生成重新构建 state/cache**
（本 harness 每次 `generate` 从 prepare 开始,天然满足）;只在 GPU 无其他任务
的窗口计时。报告实际模型调用与视觉 token 数。

## 9. 交付

- 可复现代码与完整运行命令（`Qwen_vl/scripts/anchor_merge_pilot/`）。
- 冻结 protocol / manifest / 配置 / commit 哈希。
- 正确性、smoke、逐题预测、评分、效率产物（`Qwen_vl/outputs/anchor_merge_pilot/`）。
- `docs/anchor_merge_pilot_report.md`：**全部** DEV arms 与（若运行）CONFIRM
  结果,不只报最好一项;运行总耗时、失败/未执行项、偏离记录。
- 只提交本轮文件到 `codex/anchor-merge-pilot`;不合并 main、不强推、不提交
  权重或数据集。

## 10. 偏离记录

（运行中追加;当前条目均发生在任何本轮准确率数字产生之前,且不改变实验定义。）

1. **D-1(2026-10-01,首次 smoke 前)**:`amp_correctness.py` G3 的 real-sample
   检查初版把 bf16 输出与 FP32 参考直接比较(1.8e-3 差异即 bf16 输出量化,
   非数学错误);已改为在 FP32 域比较数学、并单独要求 bf16 输出 cast 与生产
   路径 bitwise 一致。修复后 G3 rel_real = 3.2e-8,cast bitwise 相等。
2. **D-2(2026-10-01,首次 smoke)**:`amp_accuracy.py` 分片持久化 bug——shard
   在循环内被重复从磁盘加载,内存中累积的记录被丢弃,每次只落盘最后一条;
   已修复(每个 (arm, ds) 加载一次,定期原子保存)。首次 smoke 生成的单条
   记录有效,已并入续跑。
3. **D-3(2026-10-01,首次 smoke 评分)**:VLMEvalKit 的 Text/DocVQA evaluator
   在本环境返回嵌套 headline `{'Overall': {0: 86.0}}`,`e0_analyze.official_main`
   会 KeyError。评分自检改用本地 `_headline()` 解包一层嵌套 dict 取数值。
   macro 一律由逐题分数均值聚合,不依赖该 headline。
4. **D-4(2026-10-01,confirm BASE 评分自检失败后)**:OCRBench 逐题重算规则
   (沿用 m12_analyze 的版本)与官方 `vlmeval/dataset/utils/ocrbench.py` 有两处
   偏差:① prediction 漏做 `\n`→`' '` 归一化;② math 分支错误地对 prediction
   做了 lowercase(官方 math 分支双方均不 lowercase),导致 Latex 大小写不匹配
   而漏计。confirm/BASE 上官方 145 vs 重算 141,4 题漏计。已改为与官方逐字符
   一致的分支结构并重新评分;DEV 与 CONFIRM 全部 27 个评分单元的逐题均值
   均复现官方 headline(自检全过)。此 bug 同样存在于 m12_analyze 的历史
   版本;历史 OCRBench 逐题分数若受影响方向为偏低,paired 对比两侧同规则
   相互抵消,但 headline 复现检查会失败。
