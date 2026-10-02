# Stage-1 Visual Calibration Discovery — Report

Date: 2026-10-02 · Worktree `/media/disk2/YZX/research/EADP_amp` · branch
`codex/anchor-merge-pilot` · env `qwen3vl_clean` · single A40.

**Verdict: KILL（DEV 通过 → CONFIRM 否证）.** Visual novelty /
coverage-potential 校准在 DEV 300 上产生了唯一的通过冻结 GO 阈值的 arm
（C: kNN-novelty β=0.5，macro +0.47），按协议冻结后进入 CONFIRM 600——
**未能确认**：CONFIRM macro **−1.37 [−3.56, +0.83]**，DocVQA −4.57。
按用户指令（§9：Stage1 KILL 后不再找新 Stage1），Stage 1 discovery
**就此终止**（这是 grounding 轮之后的第二次、也是最后一次 Stage-1
discovery）。DCC 方法冻结不变：**official EADP Stage 1 + Facility
Location + Anchor Completion MAIN025 (λ=0.25)**。

原始产物：`Qwen_vl/outputs/stage1_visual_calibration/`；脚本
`Qwen_vl/scripts/stage1_novelty/`（协议见其 `PROTOCOL_NOTES.md`）。

---

## 1. Hypothesis

一个视觉 token 是否值得保留，除 query relevance（官方 EADP Stage-1）外，
还取决于其信息是否容易被其他视觉 token 替代（visual novelty / uniqueness /
redundancy）。Facility 仍负责最终 coverage；本轮只让 importance map 不
完全由文本 relevance 决定：

> query relevance tells us what the question wants;
> visual novelty tells us what the image cannot afford to lose.

与上一轮（grounding confidence，文本侧维度，KILL，d50da38）正交；本轮是
**视觉侧**维度，不是对上一轮的调参重试。

## 2. Exact signal definitions（冻结于任何 accuracy 运行之前）

对每张图（N > K，主视觉特征，官方 sim matrix `0.5*(cos+1)`，cos = 2*sim−1）：

- **v1_knn（kNN novelty）**：`novelty_i = 1 − mean(top-16 off-diagonal
  cosine)`。k=16 冻结（brief 允许 8 或 16，取一，不扫参）。高 = 特征
  邻域无法替代该 token。
- **v3_cov（coverage potential）**：`mass_i = Σ_{j≠i} relu(cos_ij − τ)`，
  `τ` = off-diagonal cosine 均值（自适应阈值，不调参）。高 = 该 token
  代表较多其他 token。这是 per-token 表示效用先验，**不是**重跑 facility。
- v2_local_global 按 brief 允许**跳过**（实现成本与 v1/v3 的方向覆盖
  不成比例；v1/v3 已覆盖 anti-redundancy vs representativeness 两个方向）。

融合（官方 importance 为 min-max→smoothing k3 σ1→^2 之后的 verbatim 输出，
天然有界 [0,1]）：

```
importance' = importance + β · minmax(visual_signal)
```

β=0 时逐位等于官方（gate G1）。Facility 选择器 verbatim 不变。

## 3. Gates（全部 PASS，`outputs/stage1_visual_calibration/gate.json`）

- **G1** PASS：β=0 fused selector keep == online b1 keep，逐位一致
  （9/9 gate 样本）——融合管线在 β=0 时不改变任何东西。
- **G2** PASS：online b1 keep == 冻结 amp bank keep（9/9）——BASE 路径
  与 anchor-merge round-2 逐位同源。
- **G3** PASS：4 个 arm × 3 数据集端到端 smoke，keep finite，selector_ms
  正常。
- **G4** 信号有效性：signal 与官方 importance 的 Pearson 相关近零
  （v1: −0.11/−0.01/−0.17 于 Text/Doc/OCR；v3: +0.13/−0.03/+0.07）——
  与假设一致（近乎正交的新维度）；keep 集相对官方变化 42–134/256 token
  （非 no-op）。离线 sanity：构造 50-token 近重复块 → 该块 novelty→0、
  coverage 显著高于其余 token，方向正确。

## 4. DEV panel（300 题；BASE 复用 amp round-2 shards）

| arm | 定义 | TextVQA | DocVQA | OCRBench | macro | ΔBASE |
|---|---|---|---|---|---|---|
| BASE | official EADP (b1) | 78.70 | 78.77 | 86.00 | 81.157 | — |
| A | v1_knn β=0.10 | 81.40 | 71.89 | 86.00 | 79.765 | −1.39 |
| B | v1_knn β=0.25 | 79.10 | 73.95 | 87.00 | 80.016 | −1.14 |
| C | v1_knn β=0.50 | 80.70 | 76.19 | 88.00 | **81.629** | **+0.47** |
| D | v3_cov β=0.25 | 76.90 | 77.21 | 87.00 | 80.370 | −0.79 |

Paired cluster bootstrap vs BASE（macro，95% CI）：

- A: −1.39 [−3.69, +1.11]，rescued 7 / broken 11
- B: −1.14 [−3.65, +1.46]，rescued 8 / broken 12
- **C: +0.47 [−2.32, +3.50]，rescued 12 / broken 10**
- D: −0.79 [−3.14, +1.39]，rescued 9 / broken 7

Selected-set overlap vs BASE（arm C）：0.730 / 0.748 / 0.722
（Text/Doc/OCR，200/300 样本集每样本均有变化——非 identity）。

**GO 判定（冻结规则逐条）**：C 满足 macro ≥ +0.3（+0.47）✓；无 benchmark
崩（最差 DocVQA −2.58）✓；paired 点估计为正 ✓（**但 CI 跨零，须如实
报告**）；额外 latency ≈ 0 ✓；keep 集非 identity ✓ → **GO**。
Winner 冻结一次：`v1_knn, k=16, β=0.5, minmax 归一化, additive`，此后
未做任何调参。

诚实备注：β 单调性不成立（0.1→−1.39，0.25→−1.14，0.5→+0.47），C 的
DEV 优势主要来自 OCRBench +2 与 TextVQA +2、部分被 DocVQA −2.58 抵消；
CONFIRM 事先就是对这个弱信号的检验。

## 5. CONFIRM panel（600 题；BASE 复用 amp round-2 confirm shards）

| arm | TextVQA | DocVQA | OCRBench | macro | ΔBASE |
|---|---|---|---|---|---|
| BASE | 83.30 | 76.53 | 72.50 | 77.442 | — |
| C（frozen winner） | 82.75 | 71.96 | 73.50 | 76.069 | **−1.37** |

C_vs_BASE: **delta −1.373 [−3.555, +0.834]**，rescued 20 / broken 30，
per_ds: TextVQA −0.55 / **DocVQA −4.57** / OCRBench +1.00。
Overlap vs BASE ≈ 0.73–0.74（200/200 样本变化）。

**未确认。** DEV 的 +0.47 在 CONFIRM 上翻转为 −1.37，且 DocVQA 跌幅
扩大（−2.58 → −4.57）。rescued<broken（20 vs 30）。DEV 信号判为噪声/
选择效应。

## 6. 2×2 Stage1/Stage2 combination

**NOT_RUN** —— 协议门控：仅当 Stage-1 CONFIRM 通过才组合 MAIN025
（§8）。CONFIRM KILL，cell C/D 不运行；BASE 与 MAIN025 的 CONFIRM 数字
沿用 anchor-merge round-2（MAIN025 +1.40 [CI +0.32, +2.57] 面板 /
[+0.005, +2.34] 全量预注册口径，见该轮报告）。分析脚本已就绪未使用
（`s1n_combo.py` / `s1n_combo_analyze.py`）。

## 7. Latency

Selector（含 signal 计算）vs 单独测量的 online b1（30 样本参考 87.5 ms
均值）：

| arm | mean ms | n | overhead |
|---|---|---|---|
| A | 81.5 | 300 | ≈0（参考测量含 warm-up，差异为负） |
| B | 80.2 | 300 | ≈0 |
| C | 80.0 | 300 | ≈0 |
| D | 80.5 | 300 | ≈0 |

signal 的额外开销（N×N top-k 与阈值计数，N≈10²–10³）在 CUDA 计时噪声内
（~0 ms）。**latency 不是本轮的 GO/KILL 决定因素。**

## 8. GO / KILL

**KILL**（CONFIRM 未通过；DEV GO 被否证）。依据用户指令 §9：

- 不再寻找新的 Stage-1（grounding 轮、visual-calibration 轮各 KILL 一次，
  Stage-1 brainstorm 按约定终止）。
- DCC 方法冻结：**official EADP Stage 1 + official Facility (b1, K=256)
  + MAIN025 Anchor Completion (λ=0.25)**，training-free / 0 新参数 /
  同 token budget。
- 下一步 = 已冻结的全量评测预注册（`docs/anchor_merge_full_eval_prereg.md`）
  与 DCC 写稿；无新的 run plan。

## 9. 最终冻结方法配置（未变）

```
Stage 1:  official EADP scorer (α=0.5, β=2.0, entropy T=100 keep 0.2,
          M_temp=0.01, smoothing k=3 σ=1.0)  —— 无任何修改
Stage 2:  official Facility Location (b1), K=256
Gather:   Anchor Completion MAIN025 (uniform, λ=0.25, main stream only)
训练:     training-free / 0 new parameters
Budget:   与 official EADP 完全相同
```
