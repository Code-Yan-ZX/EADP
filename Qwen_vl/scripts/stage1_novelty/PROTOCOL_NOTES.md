# Stage-1 Visual Calibration Discovery — frozen protocol

Date frozen: 2026-10-02.  Worktree `/media/disk2/YZX/research/EADP_amp`,
branch `codex/anchor-merge-pilot`.  Final discovery round for Stage 1:
the previous grounding round was KILL (commit d50da38); this round tests a
DIFFERENT, visual-side dimension and is not a re-tune of it.

## Hypothesis (user brief)

Whether a visual token is worth keeping should depend not only on query
relevance (official EADP Stage-1) but also on whether its information is
easily substituted by other visual tokens (visual novelty / redundancy).
Facility keeps responsibility for final coverage; this round only stops
the importance map from being decided purely by text relevance.

> query relevance tells us what the question wants;
> visual novelty tells us what the image cannot afford to lose.

## Fixed (Stage-2 freeze carries over; user directive)

- Backbone Qwen3-VL-8B-Instruct-1024, native E0 engine, greedy decode,
  max_new_tokens=2048, protocol v0.3.
- Official EADP scorer (alpha=0.5, beta=2.0, entropy T=100 keep 0.2,
  M_temp=0.01, smoothing k=3 sigma=1.0) -> official facility-location
  selector (b1), K=256, identity gather — **no Anchor Completion anywhere
  in the discovery stage**.
- Training-free; no new forward passes; no new parameters; no changes to
  Facility or Anchor Completion.
- Panels: the FROZEN anchor-merge-pilot manifest (DEV 100/ds = 300;
  CONFIRM 200/ds = 600).  BASE = the round-2 amp BASE shards (gate G2
  must prove online b1 keep == bank keep; gate G1 proves the beta=0
  fused selector is bitwise identical to b1).

## Fusion rule (frozen BEFORE any accuracy run)

For each image (N > K):

    importance' = importance + beta * minmax(visual_signal)

- `importance` = the verbatim official map AFTER smoothing and ^2
  (already bounded [0,1]); beta=0 is therefore bitwise official (G1).
- `visual_signal` computed on the OFFICIAL visual-visual similarity
  matrix (cos = 2*sim-1), main visual features only:
  - **v1_knn** (kNN novelty): `novelty_i = 1 - mean(top-16 off-diagonal
    cosine)`.  k=16 frozen (brief allows 8 or 16; one value, not swept).
  - **v3_cov** (coverage potential): `mass_i = sum_{j!=i} relu(cos_ij -
    tau)`, `tau` = mean off-diagonal cosine (self-adaptive, not tuned).
    This is a per-token representation-utility prior, NOT a re-run of
    facility.
- v2_local_global SKIPPED per brief (implementation cost not justified
  for a discovery round; v1 and v3 bracket the anti-redundancy vs
  representativeness directions).

## Arms (frozen; 5 incl. BASE — brief cap 5-6)

| arm | variant | beta |
|-----|---------|------|
| BASE | official EADP (b1), reused amp round-2 shards | — |
| A | v1_knn | 0.10 |
| B | v1_knn | 0.25 |
| C | v1_knn | 0.50 |
| D | v3_cov | 0.25 |

## Decision rules (user brief section 6, frozen)

- GO: best arm macro >= BASE + 0.3 AND no benchmark collapses AND paired
  trend positive AND extra selector latency negligible AND selected set
  is not identity.
- KILL: best arm <= +0.2, or only TextVQA rises while DocVQA/OCRBench
  clearly drop.  No re-tuning after the DEV readout either way.
- On GO: freeze the winner (variant, k, beta, normalization) once, run
  CONFIRM 600, then the 2x2 Stage1/Stage2 combination (BASE / +MAIN025 /
  NEW-S1 / NEW-S1+MAIN025).  On KILL: no further Stage-1 exploration;
  DCC method stays official EADP Stage 1 + Facility + MAIN025.

## Gates (must PASS before accuracy runs)

G1: beta=0 fused selector keep == online b1 keep, bitwise, on real DEV
    samples (fusion plumbing is a no-op at beta=0).
G2: online b1 keep == frozen amp bank keep on gate samples.
G3: all 4 arms end-to-end smoke, finite keep, selector_ms recorded.
G4: importance correlation + keep-set delta vs official on gate images
    (catches a no-op signal: if keep_delta == 0 everywhere, FAIL fast).

## Verdict (2026-10-02)

DEV: arm C (v1_knn, beta=0.5) macro +0.47 [+0.3 threshold met; CI
[-2.32,+3.50] crosses zero — reported as-is] -> GO per the frozen
conjunction; winner frozen ONCE (v1_knn, k=16, beta=0.5, minmax, additive).
CONFIRM 600: C vs BASE macro **-1.37 [-3.56, +0.83]**, DocVQA -4.57,
rescued 20 / broken 30 -> **NOT CONFIRMED -> KILL**.  2x2 combination
NOT_RUN (gated on Stage-1 confirm).  Stage-1 discovery terminates per the
user directive (no further Stage-1 exploration); the DCC method stays
official EADP Stage 1 + Facility + MAIN025.
Report: reports/stage1_visual_calibration_discovery.md.  Evidence JSON in
outputs/stage1_visual_calibration/ (analysis_dev.json, analysis_confirm.json,
gate.json, perf_selector.json; shards local).
