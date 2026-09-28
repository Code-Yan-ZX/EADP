# Research Contract: SAGE — Set-Conditioned Answer-Gain Exchange

> Focused working document for the selected idea (`idea.detail.en.md`). Session
> recovery reads this, not the full proposal.

## Selected Idea

- **Description**: In fixed-budget (K=256 of N=1024) hard visual-token pruning
  for Qwen3-VL-8B, propose equal-budget group exchanges around the incumbent
  EADP B2 set, train a small pre-LLM critic conditioned on the surviving set to
  predict the paired official-answer-score change of each complete exchange, and
  deploy the single best edge only above a validation-calibrated threshold —
  one decoder pass, no gold answer, no gradient, no extra generation online.
- **Source**: `idea.detail.en.md` (Phase-1 idea discovery output).
- **Selection rationale**: the project's M2–M9 panels show cheap pre-LLM and
  interaction-based proxies never converted to an answer+TTFT win; this idea
  changes the target from token/coverage proxies to the paired final-answer
  effect of whole equal-budget exchanges — the one decision the deployment
  actually makes.

## Core Claims

1. **Main claim**: a survivor-set-conditioned pre-LLM critic trained on paired
   official answer-score differences selects beneficial equal-budget group
   exchanges around B2 on fresh OCR-heavy Qwen3-VL images.
2. **Supporting claim**: the gain survives the measured end-to-end TTFT
   constraint and the controls (matched-rate random exchange, same-label unary
   scorer, within-image label permutation).
3. **Scope claim**: tested on TextVQA/DocVQA/OCRBench with the frozen EADP
   harness (N=1024, K=256, 1024×1024, α=0.5, β=2.0); no generality claim beyond
   this setting.

## Method Summary

Frozen harness: weight-accessible Qwen3-VL-8B, official VLMEvalKit scoring,
B2 (block8@256) incumbent, B1 (facility@256) secondary baseline. The edge
generator ranks removal seeds by low EADP importance inside S0 and insertion
seeds by high unretained EADP + M6 cos_s0c nomination, expands each seed to a
size-g group of grid-nearest eligible tokens (Chebyshev distance, index tie
breaks), and builds Se = (S0 \ G-) ∪ G+ with |Se| = K. On image-disjoint fit
images, sampled edges are labeled by generating once for S0 and once per Se with
the same frozen decoder and scoring both answers with the official scorer
(Δ ∈ [-1, 1]). A small MLP on survivor/group projected-vector means, the mean
question embedding, and group-to-survivor cosine summaries is fit by squared
error to Δ; capacity, g, and the acceptance threshold τ are chosen on
image-disjoint validation under a measured TTFT constraint. Deployment evaluates
E(x) before the first decoder block and delivers one ordered K-token set.

## Experiment Design

- **Datasets**: TextVQA_VAL, DocVQA_VAL, OCRBench (VLMEvalKit TSVs, local).
  Fit = bank 240, val = bank 60, confirmation = fresh 240/benchmark outside
  bank ∪ M1-extension ∪ S2-A causal cases (`sage_plan.json`, PR decision 11).
- **Baselines**: B2 (identity reference), B1, matched-rate random exchange,
  same-label unary scorer, within-image label permutation, hindsight best edge
  (ceiling only). M10 routing: blocked, no definition in repo.
- **Metrics**: official macro accuracy (mean of per-benchmark acc_pct), paired
  bootstrap CIs, per-task rescue/break counts, end-to-end TTFT.
- **Key hyperparameters**: pre-registered in `refine-logs/EXPERIMENT_PLAN.md`
  decisions 1–13 (g ∈ {1,2,4,8}, m=16, 8 sampled edges/image, MLP width
  {256,1024}, τ grid, TTFT ≤ 1.10× B2 median).
- **Compute budget**: ~4–6 A40 GPU-hours total (measured throughput ≈0.8 s per
  generation at MAX_NEW=2048).

## Baselines (reproduced, this machine)

| Method | Dataset | Metric | Score | Source |
|--------|---------|--------|-------|--------|
| EADP B2 (block8@256) | TextVQA_VAL | VQA ×100 | 71.04 | official repro 2026-09-21 |
| EADP B2 | DocVQA_VAL | ANLS ×100 | 61.14 | official repro 2026-09-22 |
| EADP B2 | OCRBench | Final Score | 623 | official repro 2026-09-22 |

## Current Results

| Method | Dataset | Metric | Score | Notes |
|--------|---------|--------|-------|-------|
| B2 (reference) | fresh 720 (3 tasks) | macro acc | 63.62 | 240/benchmark locked split |
| SAGE (frozen) | fresh 720 | macro acc | 62.32 | −1.30 [−2.3, −0.4] paired bootstrap — REFUTED |
| UNARY control | fresh 720 | macro acc | 62.57 | set-conditioning adds nothing |
| RND matched-rate | fresh 720 | macro acc | 61.95 | ordering signal worth ≈+0.4 only |
| Hindsight (4 edges/img) | fresh 720 | macro acc | 68.83† | +5.2 lower-bound ceiling, 8.2% of images |

† best sampled edge per image, realized — not deployable.

Full details: `refine-logs/EXPERIMENT_RESULTS.md`; verdict JSON:
`Qwen_vl/outputs/discovery/sage_verdict.json`.

## Key Decisions

- g = K endpoint excluded (critic features undefined there — the idea's own
  blocker); g grid capped at 8.
- One candidate edge deployed per instance ⇒ τ is sweepable offline from cached
  generations (huge compute saving, exact under eq. 5).
- M10 control not fabricated; recorded as blocked-on-definition.
- PERM control found degenerate under argmax deployment (≡ SAGE); amended to
  PERM-X (cross-image rotation) before its result was seen.
- Single A40, sequential runs, atomic JSON dumps (M3 lesson), gates abort
  rather than record (M7 convention).

## Status

- [x] Idea selected
- [x] Baseline reproduced (official B2/B1 numbers above)
- [x] Main method implemented
- [x] Representative dataset results
- [x] Full confirmation results
- [x] Verdict against pre-registered criteria: **REFUTED**
