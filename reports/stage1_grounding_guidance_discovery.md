# Stage-1 Grounding Guidance Discovery — Report

Date: 2026-10-01 · Worktree `/media/disk2/YZX/research/EADP_amp` · branch
`codex/anchor-merge-pilot` · env `qwen3vl_clean` · single A40.

**Verdict: KILL.** 在与 EADP 官方 Stage 1（entropy filtering + inverse-entropy
aggregation）完全相同的 facility selector / K=256 / 面板 / 解码配置下，
所有 7 个 grounding-guidance 变体在 DEV panel 上**全部不优于 BASE**
（最好 −0.32，最差 −1.37 macro）。按冻结决策规则（§7：所有新方法 ≤ EADP
即停止），Stage 1 discovery 终止。DCC pipeline 维持
**EADP Stage 1 + Facility Location + Anchor Completion (λ=0.25)**。

原始产物：`Qwen_vl/outputs/stage1_grounding/`（本仓库 `Qwen_vl/outputs/`
不入库时见各 shard json；分析 JSON 与 perf json 在该目录）。

---

## 1. Motivation

EADP Stage 1 判断 text token 是否参与 visual scoring 的唯一维度是
**响应的空间集中度**（entropy）。本轮检验第二个维度：**grounding
confidence**——该 token 的最高视觉响应是否真正显著高于背景。假设是二者
互补：一个 token 可能空间弥散但有真视觉证据（people/numbers/windows），
或空间集中但无可靠证据（please/using/the）。

## 2. EADP Stage 1 baseline（BASE）

官方 `VisualTokenPruner.compute_importance` 路径
（`Qwen_vl/model/pruner.py`，经 `instrumented.TimedEADPPruner`，
α=0.5、β=2.0、entropy T=100、keep_ratio=0.2、M_temp=0.01、smoothing
k=3 σ=1.0、官方 facility selector）。

**实现事实**：官方聚合权重 `softmax(-H/0.01)` 数值上近似 one-hot（entropy
为 nats 量级，/0.01 后 logits 跨度数百）——Stage 1 聚合实际≈"保留最低
entropy 20% token，然后基本只取其中 argmin-entropy 单个 token 的相似度行"。
因此字面的加性 grounding 融合是数学 no-op；经确认采用
**entropy 过滤 + 集合内 grounding 加权** 作为 ARM B 的融合机制。

BASE 数字直接复用 anchor-merge-pilot round-2 的 BASE shards（同一冻结
manifest；gate G2 证明 online b1 == 冻结 bank keep，9/9 样本）。
参考：CONFIRM-600 上 BASE = 77.442 macro，U025（Anchor Completion）=
78.838（round-2 报告）。

## 3. Grounding Confidence 定义（复用已有 text×visual cosine，零额外 forward）

对每个 text token m 的视觉响应行 s_m ∈ R^N（实际 cosine，非官方取负存储）：
G1 = mean(top5%) − mean(all)；G2 = mean(top5%) − mean(bottom10%)；
G3 = (mean(top5%) − mean(all)) / std(all)；peak = frac(s_m > mean+1·std)。
总体常数冻结（top 5% ≥8、bottom 10% ≥8、c=1.0），不搜参。

ARM（用户确认的 7-arm 网格）：
A_*（grounding-only，无 entropy，softmax(λ·z(g)) 全 token 加权）；
B_*（官方 entropy 20% 过滤保留，集合内权重换为 softmax(λ·z(g_kept))）；
C_G1（B 机制 + peak 统计）。

## 4. 校准与冻结

λ∈{0.25,0.5,1.0} 在 10/ds smoke 上校准：keep 集对 λ 完全不敏感
（离线验证 3 样本 + smoke 预测逐位相同）→ λ 在允许网格内无信息量，
冻结 **λ=1.0**。smoke 30 题不足以下结论（arm 移动 ~50/256 kept tokens
但 30 题答案全稳定），arm 选择在完整 DEV panel 进行。

## 5. 正确性门（PASS 后才跑精度）

- G1: 变体打分器的 base 路径 == 官方 `_score`，逐位（max|diff| = 0.0）。
- G2: online b1 keep == 冻结 amp bank keep（9/9 DEV 样本）。
- G3: 7 arms 全部端到端 smoke，selector ~80 ms，diag 捕获正常。

**过程记录**：首轮 DEV 曾因 `stage1_local_sim` 返回形状错误
（(1,1,N) 与 (1,N,1) 广播成 (1,N,N)，把所有 only/cal 变体洗成同一结果）
而无效；发现（各 arm 预测 100/100 逐位相同）→ 修复 → 重新过 G1–G3 →
重跑。无效 shards 保留在
`outputs/stage1_grounding/invalid_bug_bcast/`，未删除。

## 6. DEV panel 结果（n=300 = 100/数据集，λ=1.0）

| arm | TextVQA | DocVQA | OCRBench | macro | Δ vs BASE, 95% CI (paired cluster bootstrap) |
|---|---|---|---|---|---|
| **BASE (EADP)** | 78.70 | 78.77 | 86.00 | **81.157** | — |
| A_G1 (only, margin) | 80.60 | 76.87 | 83.00 | 80.157 | −1.001 [−3.752, +1.731] |
| A_G2 (only, peak-bg) | 79.00 | 76.52 | 87.00 | 80.840 | −0.317 [−2.811, +2.213] |
| A_G3 (only, standardized) | 79.90 | 72.68 | 87.00 | 79.860 | −1.298 [−4.053, +1.522] |
| B_G1 (entropy+G1) | 79.40 | 74.96 | 85.00 | 79.787 | −1.371 [−3.777, +1.014] |
| B_G2 (entropy+G2) | 81.70 | 73.31 | 86.00 | 80.335 | −0.822 [−3.476, +1.800] |
| B_G3 (entropy+G3) | 79.40 | 75.31 | 87.00 | 80.570 | −0.587 [−2.814, +1.674] |
| C_G1 (entropy+multi-peak) | 80.20 | 74.47 | 87.00 | 80.556 | −0.602 [−3.212, +2.089] |

质量检查：所有 arm × 数据集 empty=0、truncated=0、degenerate_repeat=0。

**逐数据集模式高度一致**：所有 7 个 arm 都是 TextVQA 微升
（+0.3…+3.0）、DocVQA 下降（−1.9…−6.1）、OCRBench 基本持平或微升。
没有任何 arm 在三个数据集同时不劣于 BASE。

## 7. 机制分析（TextVQA DEV diag，100 sets / 378 kept tokens）

- entropy 与 grounding 的相关性：corr(H, G3) = **−0.109**（B_G3）、
  corr(H, peak) = **+0.182**（C_G1）——假设中的强反相关并不存在，
  两个信号大部分时候给出一致的排序。
- **blind-spot 假设未兑现**：在整个 diag 语料中，
  "高 entropy 且强 grounding"（z>1 且 H > median+0.5σ）的 token 数为
  **0**。用户假设的 people/numbers/windows 类 blind-spot token 在本
  backbone 的 LLM 空间相似度中不构成可测现象。
- grounding 会翻盘但翻错：集合内 grounding argmax 与官方 argmin-entropy
  的选择在 **67%（B_G3）–77%（C_G1）** 的样本上不同，DEV 净效应为负
 （broken ≥ rescued）。
- **"低 entropy 但弱 grounding" 的 token 恰恰是问题词**：
  what(14 次)、conc(isely)(8)、the(7)、phrases(7)、?(5)、when(4)。
  它们视觉弥散但对任务关键——grounding 加权系统性削弱它们，正对应
  所有 arm 的 DocVQA 大幅下降（文档题最依赖问题词引导）。
- token 级案例（idx 43，"what type of liquor is displayed?"）：
  entropy 保留集 {of, type, isely, .} 漏掉了 "liquor"（grounding 第 2 强、
  H 中位）——grounding 在 token 层面确实"看到"了内容词，但该优势
  不能转化为选择/答案层面的收益。

## 8. 失败案例（配对翻转）

A_G2：rescued=10 / broken=12；B_G3：6/7；C_G1：9/9（全部数据集合计）。
逐题明细见 `outputs/stage1_grounding/diag/cases_*.json` 的 outcome_flips。
方向性结论：翻转集里 broken 一侧集中在 DocVQA——与 §7 的问题词机制一致。

## 9. Latency（selector 阶段，配对硬件）

BASE（online b1，n=30 独立测量）：mean 89.7 ms / p50 80.3 ms。
各 arm（n=300/臂，生成 shard 内记录）：A_G1 80.8、A_G2 80.1、A_G3 80.3、
B_G1 78.9、B_G2 79.2、B_G3 81.1、C_G1 81.5 ms——**新增开销 ≈ 0**
（新统计量只是对已算好的相似度矩阵做归约；与 BASE online 均值的
−8…−11 ms 差异在测量噪声内，p50 层面基本相等）。

## 10. GO / HOLD / KILL

**KILL**（冻结规则 §7 的第三种结果）。三个候选答案：
1. grounding-only 能否独立达到 entropy 水平？——不能（最好 −0.32，
   且 DocVQA 结构性受损）。
2. entropy+grounding 校准是否互补？——不互补（最好 −0.59；flip 率
   高但净负）。
3. multi-peak 是否救回 entropy blind spot？——blind spot 在语料中
   不存在（0 token），multi-peak 变体 −0.60。

诚实性说明：DEV n=300 的 CI 约 ±2.5–3 点，检测 +0.5 级效应的 power
有限；但 (a) 冻结规则要求 DEV winner > BASE 才进 CONFIRM，无一满足；
(b) 7 个 arm 全部呈 "+TextVQA / −DocVQA" 的同号模式，方向一致性
进一步反对存在被噪声掩盖的正效应。

## 11. 若 GO 的冻结配置

不适用（KILL）。**无新 Stage-1 配置可冻结；保留官方 EADP Stage 1。**

## 12. Stage-1 + Anchor Completion 组合结果

**NOT_RUN**——按协议 §8，组合实验仅在 Stage-1 winner 冻结后进行；
本轮无 winner。DCC 最终 pipeline 冻结为：
EADP Stage 1（官方 entropy 路径）+ Facility Location（b1, K=256）+
Anchor Completion（uniform, λ=0.25, 主特征），即 round-2 的 U025：
CONFIRM-600 macro 78.838（+1.396 [+0.324, +2.569] vs EADP 77.442）。

## 13. 复现

```bash
cd Qwen_vl/scripts/stage1_grounding
python s1g_gate.py --gate-samples 3 --lam 1.0        # G1-G3, PASS
bash dev_panel.sh                                     # 7 arms x 300 + analyze
python s1g_perf.py --arms ... --samples 10            # 配对延迟
python s1g_cases.py --diag <diag.jsonl> --arm <ARM>   # 机制案例
```

代码：`Qwen_vl/model/eadp_stage1.py`（变体打分器）、
`Qwen_vl/scripts/stage1_grounding/`（驱动/分析）。
面板 = `outputs/anchor_merge_pilot/manifest.json`（冻结，DEV 100/ds）。
