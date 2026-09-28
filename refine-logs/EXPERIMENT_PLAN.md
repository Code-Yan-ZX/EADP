# EXPERIMENT PLAN — SAGE (Set-Conditioned Answer-Gain Exchange)

**Source**: `idea.detail.en.md` (research proposal, unvalidated).
**Date frozen**: 2026-09-28. **Hardware reality**: single A40 46 GB (not 80GB-class);
throughput measured from M7 bank arms (~0.8 s/instance end-to-end generation at
T=256, MAX_NEW=2048) puts the whole campaign at ~4–6 GPU-hours, far under the
150 GPU-day ceiling. Budget is therefore not the binding constraint; statistical
resolution is.

## What SAGE claims (falsification target)

A survivor-set-conditioned pre-LLM critic f_theta(x, e), trained on paired
official-answer-score differences Delta(x, e) of complete equal-budget group
exchanges around B2, can pick exchanges that raise paired macro accuracy on a
fresh locked split while satisfying a measured end-to-end TTFT constraint.

## Frozen pre-registration decisions

The idea's "Open questions" are resolved here, BEFORE any number is read.
Each decision is marked PR (pre-registered).

1. **PR — g grid**: g in {1, 2, 4, 8}, selected on image-disjoint validation.
   The g = K endpoint is excluded: the survivor mean and group-to-survivor
   cosines are undefined there (the idea itself flags this as a blocker), and
   g > 16 would evict >6% of the core in one step.
2. **PR — seeds**: m = 16 removal seeds = the 16 lowest-`imp` tokens inside S0
   (ties by token index ascending). Insertion seeds = the top-2m = 32 unretained
   tokens by the mean of within-instance percentiles of `imp` and `cos_s0c`
   (both oriented high = keep; ties by token index). This is exactly the idea's
   "high unretained EADP and existing M6 cosine nomination scores".
3. **PR — grid distance**: Chebyshev distance on the 32x32 token grid; a
   size-g group is the seed plus its (g-1) nearest eligible tokens; distance
   ties by original token index (as the source fixes). Removal groups draw
   eligible = S0 \ {seed}; insertion groups eligible = unretained.
4. **PR — E(x) construction**: for each (removal seed, insertion seed) pair,
   G- = removal group of the removal seed, G+ = insertion group of the
   insertion seed; dedupe by (G-, G+) identity; drop edges with G- ∩ G+ ≠ ∅
   (impossible by construction: proper sets) and cap E(x) at 256 edges/image
   (never reached with m=16: ≤ 16×32 = 512 pre-dedup, dedup collapses
   overlapping groups).
5. **PR — sampled edges**: 8 edges per image, uniform over deduped E(x),
   `np.random.default_rng(20260928)`, for BOTH fit labeling and val RMSE
   labeling. Confirmation hindsight uses 4 edges/image, same seed protocol.
6. **PR — critic features** (phi, eq. 3 of the idea): concat of fp32 means of
   L2-normalised projected vectors of survivors (S0 \ G-), G-, G+; the mean
   question instruction embedding q̄ (4096); and 4 cosine summaries — for each
   exchange group, the mean and max over its tokens of cos(token, survivor
   centroid), zero-norm clamped at 1e-8 (m5_common convention). Total 16388
   inputs. The survivor set is never empty for g ≤ 8 << 256.
7. **PR — critic architecture**: 2-hidden-layer GELU MLP, width grid
   {256, 1024}, squared error on Delta, Adam lr 1e-3, weight decay 1e-4,
   batch 256, 100 epochs, seed 0, inputs standardised by fit-split stats.
   Capacity chosen by RMSE on val edges. The UNARY control gets the same
   training protocol on 8 unary scalars (mean/max of `imp` and `cos_s0c` over
   each group) — no survivor conditioning, no question embedding.
8. **PR — calibration**: for each (g, capacity), deploy the val argmax edge
   once, record y(S0) and y(S_e*) hits, then sweep tau OFFLINE over
   {0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5} ∪ {deciles of the val
   predicted-gain distribution}. Only ONE candidate edge is deployed per
   instance (eq. 5), so every tau shares the same generations. Selection:
   highest official val macro subject to the TTFT constraint; ties → lower
   deployed exchange rate, then larger tau.
9. **PR — TTFT constraint**: median wall end-to-end TTFT (prompt-build → first
   emitted token, CUDA-event bracketed for the added edge-construction + critic
   window) of the deployed arm ≤ 1.10 × median B2 wall TTFT on the same val
   pass. The multiplier is declared here, before measurement.
10. **PR — tie rules at deployment**: argmax ties among edges → lexicographically
    smallest (sorted G-, sorted G+); empty E(x) → retain S0; acceptance is
    strict f_theta(x, e*) > tau.
11. **PR — confirmation split**: 240 instances per benchmark (720 total) drawn
    from outside the frozen bank (150 evenly-spaced rows) ∪ the M1 extension
    (720 rows) ∪ the S2-A causal cases, one fixed permutation per benchmark
    under `EXTRASET_SEED + 202*(i+1)` (same protocol as m1_common.build_extension,
    new seed). Locked in `sage_plan.json` before any confirmation number exists.
12. **PR — confirmation arms**: B2 (identity reference), B1 (facility), SAGE
    (frozen critic + tau), RND (uniform random edge from E(x) on exactly the
    instances SAGE exchanged — matched rate), UNARY (its own argmax edge at its
    own tau, selected on val identically), PERM (within-image permutation of
    critic scores across E(x), argmax, on exactly the instances SAGE exchanged),
    HINDSIGHT (best realized edge among the 4 sampled — ceiling only).
13. **PR — success / rejection criteria** (from the idea's Minimal falsification,
    made numeric):
    - PASS requires: paired macro(SAGE) − macro(B2) > 0 with paired bootstrap
      95% CI excluding 0 (2000 resamples, seed 20260926); no benchmark drops
      more than 1.0 macro point vs B2 (the "clear OCRBench loss" veto); median
      TTFT within the constraint of decision 9.
    - Mechanism checks: PERM must erase the gain (its CI vs B2 includes 0 or
      is below); UNARY must recover less than SAGE (paired CI of SAGE − UNARY
      excluding 0, or a point estimate ≥ 50% of SAGE's gain missing).
    - Falsification of the early-observability premise: hindsight ceiling
      clearly positive while SAGE fails → premise falsified (report, per idea).
14. **NOT RUNNABLE — M10 control**: "local M10 whole-set routing" has no
    definition in this repo (no code, no report, no doc). Not fabricated;
    recorded as blocked-on-definition in the tracker.
15. **Evaluation integrity**: every arm's hits come from `scoring.per_sample_hits`
    against the dataset's own gold answers (TextVQA VQA score, DocVQA ANLS,
    OCRBench containment incl. HMER rule). Gold answers never enter anything
    online; they label offline fit/val/hindsight only.

## Milestones

### M0 — Sanity (MUST-RUN)
Edge-generator validity gates (equal budget, proper sets, dedup, determinism
under fixed seed, tie rules) on bank instances; end-to-end labeling of 3
instances (S0 + 2 edges each) producing plausible Delta in [-1, 1]; the
SetPruner injection gate re-verified (identity == stored B2 on the bank test
150, already proven by M7 — re-checked on 5 instances).

### M1 — Paired labels (S3) (MUST-RUN)
`y(S0)` for all 300 fit+val images; 8 edges/image × 4 g values labeled on fit
(240) and val (60). ≈ 7680 + 1920 + 300 generations ≈ 2.2 h on the A40.

### M2 — Critic (S4) (MUST-RUN)
Train per g on fit; capacity by val RMSE. Minutes on GPU.

### M3 — Calibration (S5) (MUST-RUN)
Val deployment per (g, capacity): y(S_e*) generations + offline tau sweep +
TTFT measurement. ≈ 30 min. Output: frozen (g*, cap*, tau*) + measured TTFT.

### M4 — Confirmation (S7) (MUST-RUN)
Fresh 720 (decision 11), arms of decision 12. ≈ 2 h. Paired stats per
`sage_analyze.py`.

### M5 — Analysis & report (MUST-RUN)
`EXPERIMENT_RESULTS.md` with the verdict against decision 13.

## Budget
~4–6 A40 GPU-hours total. All outputs to `Qwen_vl/outputs/discovery/sage_*.json|npz`,
JSON with atomic writes (m6_common.dump_json convention).
