# Anchor Completion — 全量 official benchmark 预注册

**冻结日期**: 2026-10-02（早于任何全量评测数字的存在）
**前置**: round 1/2 报告（`anchor_merge_pilot_report.md` /
`anchor_merge_pilot_round2_report.md`）+ Stage 1 grounding KILL。
**背景**: DEV 300 / CONFIRM 600 面板分辨率约 ±1–2 点，不足以承载论文主
claim；MAIN025 的 CONFIRM CI 下界贴零（+0.005）。

## 1. 冻结的正式方法

official EADP Stage 1（entropy importance，不再改动）+ official Facility
Location selector + **MAIN025**：仅主特征流的 Anchor Completion，
λ = 0.25 均匀修正，λ 来自 DEV 预注册网格 {0.25, 0.5, 1.0}（如实披露，
非调参产物）。

BOTH(U025，主特征 + DeepStack 双流聚合) **不是**正式方法，仅作 ablation；
round 2 已证 BOTH − MAIN 不可分辨（CONFIRM +0.259 CI [−0.109, +0.720]）。

## 2. 全量评测设置

- 引擎 / 解码：native E0 engine，greedy，与 round 1/2 完全一致；
- 臂：`BASE`（facility 纯 gather）vs `MAIN025`，配对；
- 数据集与规模：主面板 = TextVQA val、DocVQA val、OCRBench **全量**；
  非回归面板 = ChartQA test、MMBench dev、MMStar、RealWorldQA、POPE 全量；
- 评分：官方规则；**OCRBench official Final Score 只在全量集上报**
  （子集上官方类别聚合公式无效，S2 结论）；子集/逐题分析用
  amp_accuracy 的逐题 scorer（27/27 复现官方 headline）。

## 3. 主判定（预注册，跑前冻结）

- 统计：逐 benchmark 的 image-cluster paired bootstrap（5000 次，
  seed 20261002，双侧 α=0.05）；
- **主效应成立** = 主面板三个 benchmark 的逐题配对差拼接后的 pooled
  cluster-bootstrap CI 不跨零，且逐 benchmark 方向一致（不要求每个
  单独显著）；
- 非回归面板：逐 benchmark 报告 Δ 与 CI；任何 benchmark 出现 CI 排除零
  的**负** Δ 时如实报告并讨论，不因机制叙事而淡化。

## 4. 预案（用户 2026-10-02 采纳）

若主面板 pooled CI 跨零且点估计 < +0.5：

1. 触发预案臂 `BOTH025` 全量（其 CONFIRM CI 下界 +0.32 更稳）；
2. 论文如实报告：流范围在统计上不可分辨，主效应以 BOTH025 的全量结果
   确认；MAIN025 保留为 ablation 与最省实现。

预案只允许触发一次，且触发与否由本文件的规则机械判定，不看结果后改规则。

## 5. 与 S0 加固的关系

全量评分一律使用含 S0 assertion 补丁的 `e0_accuracy.py`（空 official
dict / evaluate 异常 = 硬失败，写入 `score_failures.jsonl` 并非零退出）与
含 coverage 报告的 `e0_analyze.py`（不完整面板上的 macro 不允许进入任何
决策规则）。E0 基线主表（FastV/PDrop/… 端口）在 S1–S3 root-cause 销案前
维持 do-not-cite。
