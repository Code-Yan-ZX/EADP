# RTG pilot — GO 触发记录与 R_MAIN025 全量预注册（补充）

**日期**: 2026-10-02 · 触发依据: `docs/stage1_roundtrip_pilot_protocol.md` §6
GO 分支（冻结于任何 RTG 得分之前）。本文件只记录机械判定与全量执行规则，
不改变任何已冻结的统计定义。

## 1. GO 判定（2026-10-02 21:41，四条件全部满足）

| 条件 | 阈值 | 实测 | 判定 |
|---|---|---|---|
| 1. macro 点估计 | ≥ −0.5 | **−0.254** [−2.566, +2.168] | ✓ |
| 2. DocVQA 点估计 | ≥ −1.0 | **−0.761** [−5.502, +4.369] | ✓ |
| 3. R macro 严格高于 F 与 S | 点估计 | R−F **+0.138** / R−S **+1.093** | ✓ |
| 4. TTFT 中位数增量 | ≤ 5% | **+1.26%**（399.1→404.2 ms） | ✓ |

条件 3 的两处 CI 均跨零 → 按协议只称**机制线索**，不写"已区分"。
主对比 R_MAIN025−E_MAIN025 在 DEV-300 上为 −0.254 [−2.566, +2.168]：
**不确定、未支持非劣，也无正信号**；GO 仅因机械门通过而触发。
Stage-1 成本：RTG 评分 2.12 ms + facility 78.07 ms，与 EADP 路径
（s1+facility 80.56 ms）相当；官方 Instruct-embedding 步骤免除
（RTG 不需要 instruction embeds 的额外一遍——其评分直接使用文本序列）。
有效文本 token 数：RTG p50 ≈ 19.6 / L 均值 21.6（FLAT 21.0 为对照上限）。
anchors 与 EADP 的 Jaccard ≈ 0.65–0.66（300/300 样本）。

## 2. R_MAIN025 全量预注册（加入核心三项队列）

- **范围**：TextVQA_VAL / DocVQA_VAL / OCRBench 官方完整 split，
  R_MAIN025（RTG + MAIN025，全部参数与 DEV 阶段逐位一致，不再扫任何
  超参）。E 两臂**复用**已验证全量结果（BASE ≡ E_GATHER、
  MAIN025 ≡ E_MAIN025，同配置同 split，不重新生成）。
- **bank**：RTG scorer 全量冻结 bank（keep/gid/gsize/w_diag），先于生成
  落盘并哈希；每题 RTG 重新选 anchors。
- **评分**：与本轮完全一致的官方评分 + 逐题抽取 + S0 硬失败 + 严格
  headline 复现；OCRBench Final Score 仅全量上报。
- **统计（预冻结）**：R_MAIN025 − E_MAIN025（= − MAIN025−BASE 已有 paired
  基础上的同题配对），数据集内 image-cluster bootstrap **20000 次、
  seed=20261002**，报告每任务与等权 macro Δ/95%CI；**主要判定为等权
  macro**；非劣 margin 0.5：CI 下界 > −0.5 才支持非劣，其余不确定或更差；
  **每任务尤其 DocVQA 同时报告，不能只看宏平均**。
- **不修改**原 BASE/MAIN025 全量协议的主判定（pooled 改善已成立，
  照常报告）；R_MAIN025 全量是 GO 触发的**附加臂**，其结果不影响原判定。
- **预算**：不计入 pilot 2h 上限（协议明文）。实际吞吐（DEV 实测：
  生成 ~0.5–1.3 s/q，bank ~0.35 s/q）→ 全量 ETA：bank ≈ 1.1 h +
  生成 ≈ 3.2 h + 评分 ≈ 0.3 h ≈ **4.5–5 h** 单卡 A40。执行顺序排在
  核心恢复（g3 verify → 非回归 → fresh → perf → analyze）完成之后，
  串行不抢占。

## 3. 交付

`Qwen_vl/scripts/stage1_roundtrip_pilot/rtg_full_*`（bank/生成/评分/分析），
产物入 `outputs/stage1_roundtrip_pilot/full/`；报告
`docs/stage1_roundtrip_pilot_report.md`（含 DEV 六组主表、机制对照、
GO 记录、全量结果、负/不确定结果如实报告）。
