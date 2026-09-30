# M13 — Hierarchy-on-Demand & Context-then-Compress Oracle

**Branch**: `exp/m13-hierarchy-compression-oracle` · **Date**: 2026-09-30
**Model**: Qwen3-VL-8B-Instruct (native E0 engine, SDPA, 1024×1024 fixed res)
**Question**: ① Qwen3-VL 的 DeepStack hierarchy 上是否存在可动态跳过的冗余计算？② "先完整 contextualize 再压缩"能否解决 M12 暴露的 pre-ViT irregular deletion failure？

---

## 1. Motivation

M12 证明 Qwen3-VL 的 vision tower 有巨大真实加速空间（1024→512 merged tokens：ViT
110.3→55.8 ms，TTFT 405→256 ms），但 **pre-ViT** 的 irregular spatial deletion 严重
破坏精度，且所有 pixel-space cheap score 对"后续真正重要的 token"的预测接近随机。
两条剩余假设未检验：

- **A. 深度方向冗余**：DeepStack 三个 branch（DS@8/16/24）可能存在"很贵但没贡献"
  的分支，可按需跳过（Hierarchy-on-Demand）。
- **B. 压缩时机**：M12 的失败也许不是"删错了 token"，而是"删得太早"——在 patch 尚未
  contextualize 时就做 irregular deletion。如果先让全部 token 过 L 层 dense ViT，
  再压缩，性能是否随 L 单调恢复（Context-then-Compress）？

## 2. M12 negative evidence recap

| ds | b0 | rres@512 | variance@512 | retinagate@512 | uniform@512 | random@512 |
|---|---|---|---|---|---|---|
| TextVQA_VAL | 85.03 | 80.93 | 77.87 | 69.90 | 37.10 | 42.30 |
| DocVQA_VAL | 93.86 | 89.78 | 89.93 | 71.55 | 34.42 | 48.54 |
| OCRBench | 143 | 141 | 137 | 115 | 70 | 71 |

Pre-encoder irregular deletion（任何 cheap selector）都显著落后于同预算的 resolution
reduction（rres）；真 sparse ViT 的 wall-clock 收益非常大（K512 → ViT 55.8 ms）。

## 3. DeepStack implementation audit

架构事实（transformers 4.57.3, `modeling_qwen3_vl.py`）：

- Vision tower depth=27；`deepstack_visual_indexes = [8, 16, 24]`；每个 branch 是一个
  独立的 `Qwen3VLVisionPatchMerger`（与主 merger 同构，post-shuffle-norm）。
- DS merger 输出 [N_merged=1024, D_llm=4096]，在 LLM 的 layer 0/1/2 之后经
  `_deepstack_process` 按视觉位掩码加进残差流。
- 关闭一个 branch = **真正跳过**该 DS merger 的 vision-side 计算
  （`m13_common.visual_forward_ds`，disabled branch 不跑 merger），并在 LLM 侧
  patch `_deepstack_process` 直接跳过注入（不是加零）。

Gate（`m13_smoke.py`，全部通过）：

1. all-on 自定义 forward 与 stock `visual(pv, gthw)` **bitwise 一致**（V 与三条 DS 流）；
2. 关 branch 不改主路 V（bitwise）、不改序列长度（K=1024 恒定）；
3. L=0 drop 的 depth forward 与 M12 `rg.sparse_visual_forward` **bitwise 一致**，
   单样本预测与 M12 variance@512 shard 完全相同；
4. L8-merge prefill/decode 不变量（layer_calls、cache 长度、n_vis_kept=K）全部通过。

## 4. Per-component latency profile

30 测量 blocks × 12 个 OCR-panel 分层样本，15 warm-up；同步 wall-clock per stage。

| stage | median ms | mean ms | std |
|---|---|---|---|
| patch_embed + pos prep | 6.36 | 6.48 | 0.38 |
| ViT blocks 0–7 | 30.40 | 30.45 | 0.14 |
| **DS merger @8** | **0.90** | 0.90 | 0.01 |
| ViT blocks 8–15 | 30.07 | 30.17 | 0.42 |
| **DS merger @16** | **0.89** | 0.89 | 0.01 |
| ViT blocks 16–23 | 29.99 | 29.98 | 0.12 |
| **DS merger @24** | **0.88** | 0.89 | 0.01 |
| ViT blocks 24–26 | 11.23 | 11.25 | 0.08 |
| main patch merger | 0.87 | 0.88 | 0.03 |
| **vision total** | **111.93** | 112.04 | 0.62 |
| LLM prefill (1024 vis) | 207.53 | 207.58 | 0.71 |

**三个 DS merger 合计 2.67 ms = vision 的 2.4%、E2E TTFT（~405 ms）的 ~0.7%。**
DS merger 与主 merger 成本相同（同为一次 4·1152→4096 投影）；成本主体完全在
27 个 ViT block（~101.7 ms）。LLM 侧注入只是一个 masked add，无可观成本。

## 5. DeepStack branch ablation table

<!-- ABLATION_TABLE -->

## 6. DeepStack GO/HOLD/KILL

<!-- ABLATION_VERDICT -->

## 7. Intermediate ViT compression implementation

设置：固定 selector（M12 variance，pixel-space，确定性），**keep-set 在所有深度间
完全相同**（跨深度严格配对，差异只来自压缩时机）；L ∈ {0,4,8,12,16,20}；K=512（50%）。

- **Drop**：removed group 的 4 个 patch 直接删除。
- **Merge**（scale-preserving union-mean）：removed group 按组网格欧氏最近距离分配给
  survivor；survivor 的每个 patch 吸收对应 intra-slot 的 removed patch 特征，
  `h'_i = (h_i + Σ h_j)/(1+m)`（α=1/(1+m)，保尺度；fp32 累加）。
- 压缩以 native 2×2 merge group 为原子（4096 patch → 512 group ×4），cu_seqlens 按
  kept per-image 重建；survivor 保留自己的 rotary/position；DS merger 在 L 之前的
  按 full 序列算再按 group 子集，在 L 之后的直接在压缩序列上算——与
  `eng.prefill(keep_idx, V_sel, DS_sel)` 的 M12 约定完全对齐；mRoPE3D 位置取
  full-sequence `get_rope_index` 的子集。

## 8. Compression-depth accuracy table

<!-- DEPTH_ACC_TABLE -->

## 9. Compression-depth latency table

<!-- DEPTH_LAT_TABLE -->

## 10. Drop vs Merge

<!-- DROP_VS_MERGE -->

## 11. Dataset-specific behavior

<!-- DATASET_BEHAVIOR -->

## 12. Paired sample recovery/failure analysis

<!-- PAIRED_ANALYSIS -->

## 13. Pareto summary

<!-- PARETO -->

## 14. Verdicts

<!-- VERDICTS -->

## 15. Next single mechanism (if GO)

<!-- NEXT_STEP -->

---

## Reproduction

- Scripts: `Qwen_vl/scripts/m13/`（`m13_smoke.py` → `m13_profile.py` →
  `m13_ds_ablation.py` → `m13_depth_oracle.py` → `m13_analyze.py`；runner
  `run_m13.sh`，日志 `outputs/m13/logs/`）
- Raw shards (per-sample predictions + timings, resumable):
  `Qwen_vl/outputs/m13/acc/<arm>/K*/<dataset>.json`
- Profiling: `Qwen_vl/outputs/m13/m13_ds_profile.json`; consolidated:
  `Qwen_vl/outputs/m13/m13_analysis.json`
