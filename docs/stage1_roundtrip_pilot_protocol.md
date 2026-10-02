# Stage 1 Round-trip Textual Guidance Pilot (RTG) — 协议

**冻结日期**: 2026-10-02（写于本轮任何新臂下游得分存在之前；此刻核心
Anchor-Completion 主面板仍在生成，其分数未读取）。
**分支**: `codex/anchor-completion-validation`（base 含 `60a40c3` 纠错及后续
提交）。**不切换被运行中作业占用的 checkout**——本 pilot 与核心全量共用
worktree `/media/disk2/YZX/research/EADP_amp`，代码/产物独立目录
`Qwen_vl/scripts/stage1_roundtrip_pilot/` 与
`Qwen_vl/outputs/stage1_roundtrip_pilot/`，不覆盖任何旧结果。
**调度**: 本 pilot 是对 Anchor-Completion 全量任务的调度补充。核心三项
BASE/MAIN025 全量继续保留；尚未开始的非回归五项、fresh 消融与 perf
**暂缓**，pilot 结束或达 GPU 上限后立即恢复，不等待用户再次批准。

## 0. 状态与边界

- **唯一候选**: RTG（Round-trip Textual Guidance）。只替换 EADP 的
  entropy filtering + textual weighting；Facility 与 MAIN025 完全不变。
  不再搜索其他 Stage 1；不跑新 λ/温度/符号变体；不训练。
- 思路受 LoFTR 双向匹配置信度启发（仅思想借鉴，不加载其模型、不加
  encoder/LLM 前向）：
  - https://arxiv.org/abs/2104.00680
  - https://github.com/zju3dv/LoFTR/blob/master/src/loftr/utils/coarse_matching.py
- 双向匹配、归一化与 chi-square 恒等式均为标准构造，**不声称新发明**；
  创新性只能围绕具体适配与实验贡献评估。目前无真实 VLM 正结果，
  不预设胜出或首次提出。
- 数据：旧 anchor_merge_pilot manifest 的 **DEV**（TextVQA_VAL/DocVQA_VAL/
  OCRBench 各前 100 题，共 300；manifest SHA256
  `8d55e4e491abee2876217c2b10d2d17d47d75b4dbff657f64eefe8dce9c2fc97`）。
  **已观察的探索集**（E0 / anchor-merge r1-2 / cross_stream / grounding /
  visual-calibration 均用过），只作筛选，不作独立确认；本轮不重抽样。

## 1. 模型与执行路径（不变）

与 anchor-merge r1-2 完全一致：Qwen3-VL-8B-Instruct-1024（fixed-res
1024×1024，逐图记录实际 N）、native E0 engine（DeepStack 三路、3D mRoPE、
SDPA、greedy、max_new_tokens=2048）、K=256、pre-LLM、forward-only、
qwen3vl_clean 环境、单卡 A40。

## 2. RTG 精确定义（不扫超参）

逐图执行，FP32，no_grad。使用与当前 native EADP 相同的 instruction
提取、tokenizer 与有效文本 token 集合；不额外过滤词/停用词/special
tokens、不改 prompt；有 padding 时按原真实长度 mask，pad 不参与归一化。

设 `A[N,L]` 为当前 EADP `local_sim_all` 中该图的二维矩阵（dim0=视觉,
dim1=文本），`g[N]` 为当前 `global_sim`。**沿用当前实现原始符号（已知
代码保存负 cosine）；不翻转符号**，不把该响应称为已验证的语义定位概率。

固定 `inverse_temperature=100`（原 EADP 视觉维温度）。对所有有效文本列：

```python
# A.shape == [N, L]; dim 0 visual, dim 1 text
logP = torch.log_softmax(100.0 * A.float(), dim=0)     # P(i|t), 视觉维归一
logB = logP - torch.logsumexp(logP, dim=1, keepdim=True)  # B(t|i)
c = torch.exp(logP + logB).sum(dim=0)                  # [L] 文本往返概率
text_weight = c / c.sum()                              # 和为 1
local = (A.float() * text_weight.unsqueeze(0)).sum(dim=1)
fused = 0.5 * g.float() + 0.5 * local
```

- **不调用** entropy filter、不保留 20% entropy 筛选、不计算
  softmax(-H/.01)、不对 c 做 softmax/z-score/floor/幂变换。
- fused 之后**严格复用现有 EADP 原操作与顺序**：min-max（含原 eps）→
  原 grid 上 3×3 σ=1 Gaussian smoothing → β=2.0；global/local α=0.5
  （fused 式即官方 alpha*global+(1-alpha)*local）。这些是继承组件，
  如实归属 EADP；本候选只替换熵引导文本权重。
- Stage 2：原主路相似度矩阵、原 official facility、原 tie-breaking、
  选完按原索引排序，K=256。**RTG scorer 重新选 anchors**，不沿用 EADP
  bank 的 keep。
- Completion：对本 scorer 的 anchors，dropped token 按原主特征 FP32
  cosine argmax 分配（并列取更小 anchor 序号），组均值含 anchor 自身，
  y_a = v_a + 0.25·(mean(G_a) − v_a)；只聚合主路；全部 DS[s] 保持
  DS[s][keep] 原值；native DeepStack/3D mRoPE/cache/位置不破坏；
  每图输出 min(N,K) 个 token；N≤K 整图保留；多图禁止跨图评分/分组。

## 3. 两个机制对照

- **FLAT**: 仅把 text_weight 替换为 1/L；global、postprocess、Facility、
  MAIN025 全同。这不是 uniform visual importance，禁止与旧 U_MAIN025 混用。
- **SHUF**: 每个文本列分别独立打乱视觉行索引得到 A_shuf，仅从 A_shuf
  计算 text_weight；随后用该权重加权**真实未打乱**的 A，再走相同
  global/postprocess/Facility/MAIN025。每列的熵/top-k 值不变，仅列间
  空间对应改变。所有列共用同一排列是**错误对照**（会保留对应关系）。
  - 种子（冻结）: 对字符串
    `20261002|dataset|question_id|image_index|text_index`
    取 SHA256 前 8 字节小端无符号整数 mod 2**63 为 seed，创建**局部 CPU
    torch.Generator** 生成 randperm(N)；不改全局 RNG。question_id = 数据集
    行 idx；image_index = 图内序号（本轮全部单图，恒 0）；text_index =
    文本列序号。定义与版本随产物保存。
  - 不把 SHUF 解释成完全隔离语义的因果实验；它检验依赖真实跨列空间
    对应的效果。

## 4. 六组主表（DEV；四个新臂全量生成）

| arm | Stage 1 | Completion | 来源 |
|---|---|---|---|
| E_GATHER | official EADP | 无 | **复用**逐题预测（严格验证后） |
| E_MAIN025 | official EADP | MAIN025 | **复用**逐题预测（严格验证后） |
| R_GATHER | RTG | 无 | 本轮新生成 |
| R_MAIN025 | RTG | MAIN025 | 本轮新生成（唯一主要候选） |
| F_MAIN025 | FLAT | MAIN025 | 本轮新生成 |
| S_MAIN025 | SHUF | MAIN025 | 本轮新生成 |

- E 两臂复用条件（全部满足才可复用）：配置/输入/代码路径/manifest 与
  本轮完全一致；E_GATHER/E_MAIN025 逐题分数复现 **81.1574 / 83.2673**、
  逐题键完整一致；9 题 keep/分组/特征/prefill logits/输出与旧路径复现。
  预测复用自 `outputs/stage1_cross_stream_pilot/acc/E_*`（该轮已验证与
  round-1 bitwise 一致）；历史 bank/分数仅在 provenance 完全吻合时复用。
- 新四臂每臂 300 题全量生成，smoke（每数据集/臂前 10 题）包含在内；
  不跑新变体。
- λ=0 恒等单独验证，不能替代正式 λ=0.25 的复现检查。

## 5. 正确性门（小门，全过才读下游得分；不扩成新审计工程）

1. 独立逐元素参考实现对照向量化实现：P 按视觉维列和=1；B 按文本维
   行和=1；c 按视觉维求和；text_weight 和=1。
2. 小规模 FP64 测例：1/L ≤ c_t ≤ 1；验证恒等式
   `pbar_i = mean_t P_it`，`c_t = (1 + Σ_i (P_it − pbar_i)²/pbar_i) / L`
   （标准 chi-square divergence，不当新理论）；FP32 给出合理误差而非
   不适用的 bitwise 要求。
3. 各列完全相同 ⇒ 等权；L=1 ⇒ 权重 1；零输入有限；真实 N>K 剪枝测试
   （N=64, K=16）；多图隔离；N≤K 恒等。
4. SHUF：每列排序后的值不变（熵不变）；在特制非退化例子中权重确实
   改变（含协议预核验例：三列同熵、前两列共享一峰、第三列峰在另一
   位置，权重约 [.2511, .2511, .4978]；这仅证明算子行为，不是模型效果）。
   统计实际样本 anchors/权重变化比例，不把概率不变或 anchors 不变的
   真实样本误判为实现错误。
5. E 正式配置 9 题（每数据集 3 题）keep/分组/特征/prefill logits/输出
   与旧路径复现；λ=0 恒等单独验证。
6. 新 scorer：现场与 bank keep/gid 全等；MAIN scope 的 DS 恒等
   （DS==DS[keep]）；输出数量与 native 不变量；每数据集/臂先 10 题
   smoke（空答/截断/重复），再继续。

## 6. 统计、效率、决策

- 评分：正确官方评分与逐题 scorer（D-4 修正后 OCRBench 规则、DocVQA
  ANLS、S0 硬失败）；严格 headline 复现；OCRBench 子集用实际题数归一化。
- 统计：配对、数据集内 image-cluster bootstrap **20000 次、seed=20261002**；
  三任务等权 macro；分数 0-100。所有预设比较共享重采样索引；报告每任务
  与 macro 的 delta/95%CI：
  1. R_MAIN025 − E_MAIN025（主要）
  2. R_GATHER − E_GATHER
  3. R_MAIN025 − F_MAIN025
  4. R_MAIN025 − S_MAIN025
  5. R_MAIN025 − R_GATHER
  6. 与 E 两臂的 2×2 交互
- 诊断：各组文本权重有效 token 数（1/Σw²）、与熵权重差异、anchors
  overlap、变化样本数。**效应必须由真实生成衡量**；权重/热图变化不是
  成功标准。
- 效率：同 DEV 每任务前 10 题共 30 题，现场全路径配对 E_MAIN025 /
  R_MAIN025；15 次预热，交错顺序，无 GPU 争用，固定 decode64 +
  ignore_eos + 断言长度。报告含预处理/真实 Stage1/Facility/Completion 的
  TTFT、Stage1 成本、decode、峰值显存；不以 bank lookup 替代推理；
  测量前清理上一轮 out/state/KV 引用并 reset peak。
- **GO 探索门**（跑前冻结；只决定是否值得扩大，不是统计非劣证明）：
  1. R_MAIN025 − E_MAIN025 macro 点估计 ≥ −0.5；
  2. DocVQA 点估计 ≥ −1.0；
  3. R_MAIN025 macro 严格高于 F_MAIN025 与 S_MAIN025（只称机制线索，
     CI 跨零仍必须写不确定）；
  4. 现场配对 TTFT 中位数增量 ≤ 5%，无正确性/生成异常。
  任一不满足：停止本候选，不调温度/权重/符号，不派生候选，恢复核心
  BASE/MAIN025 全量并如实报告负结果。
  全部满足：R_MAIN025 加入同一核心三项全量队列，复用已验证 E 两臂
  结果，不扩其他新臂到全量；全量分析单独预先记录（R_MAIN025−E_MAIN025
  等权 macro 为主、cluster bootstrap 20000/seed 20261002、非劣 margin
  0.5：CI 下界 > −0.5 才支持非劣；每任务尤其 DocVQA 同时报告）。
  **不修改原 BASE/MAIN025 全量协议的主要判定。**
- **GPU 预算上限 2 小时**（含 gates/新 bank/生成/性能测试；不计等待原
  作业与 CPU 实现）。达到上限即保存并标明未完成，不从缺失面板作 GO
  判定，恢复核心全量。GO 后的全量不计入本 pilot 预算，但必须先报告
  实际吞吐与 ETA。

## 7. 交付

`docs/stage1_roundtrip_pilot_report.md` + 协议；`Qwen_vl/scripts/
stage1_roundtrip_pilot/`（rtg_common/rtg_bank/rtg_correctness/rtg_accuracy/
rtg_analyze/rtg_perf）；`Qwen_vl/outputs/stage1_roundtrip_pilot/`
（correctness、bank、四新臂逐题预测/评分、E 臂复用 provenance 记录、
联合 bootstrap、perf、诊断）。报告只使用真实已完成数据，分开回答：
RTG 是否接近 EADP；是否优于简单文本平均（FLAT）；是否有真实空间对应
的增量（SHUF）；与 MAIN025 组合效果如何。引用借鉴来源，不因算法有新
名字就声称首次或足够录用。台账记录调度调整（暂缓项、恢复状态、实际
GPU 时间）与完整续跑命令。提交并推送本分支，不合并 main、不加 AI 署名。
本轮结束后**不自动寻找第五种 Stage 1**。

## 8. 偏离记录

（运行中追加；当前为空。）

## 9. 偏离记录（续）

6. **D-7（2026-10-03 04:0x，任何 fresh 消融分析发布前）**：fresh 面板
   F_MAIN025/S_MAIN025 首次生成误用 RTG bank（`load_or_build_bank` 硬编码
   scorer），且旧分片的首次清理因相对路径错误静默失败，导致 F≡S≡R 的
   错误评分（发现方式：R−F 与 R−S 的 Δ 恒为 0）。已改为 per-arm bank、
   绝对路径清理并重新生成全部 F/S 分片；错误数据未进入任何已交付分析表
   （首次 fresh 对比表在发现异常后未对外交付）。R_GATHER 分片同样重建。
7. **D-8（2026-10-03 04:5x）**：ChartQA 首评硬失败暴露 `_headline100` 取
   首个数值键 `test_human` 而非 `Overall` 的 bug；POPE 首评硬失败暴露
   `rtg_accuracy_full._score_one_full` 缺 category-explode 复算分支。
   两处修复后重评 diff=0；S0 硬失败语义按设计拦截了错误分。
