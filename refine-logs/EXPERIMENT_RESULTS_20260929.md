# Initial Experiment Results — SAGE (Set-Conditioned Answer-Gain Exchange)

**Date**: 2026-09-29
**Plan**: `refine-logs/EXPERIMENT_PLAN.md` (pre-registered decisions 1–15)
**Verdict file**: `Qwen_vl/outputs/discovery/sage_verdict.json`

## Verdict: **REFUTED** (pre-registered criteria)

On a fresh, locked, image-disjoint confirmation split (720 instances, 240 per
benchmark, drawn outside every panel any earlier stage used), the frozen SAGE
critic **fails to beat B2** — it is significantly *worse* — while the measured
hindsight ceiling shows real opportunity the critic cannot reach.

## Results by milestone

### M0 — Sanity: PASSED
Edge support valid (equal budget, proper sets, dedup, deterministic tie rules)
for g ∈ {1,2,4,8}; end-to-end labeling of 3 images, 96 edges, 0 errors.

### M1 — Paired labels (S3): DONE
9,900 generations, 0 errors (fit 240: 7,680 edges; val 60: 1,920 edges).
Label structure: mean Δ negative at every g (−0.003 → −0.012 at g=8); nonzero
Δ on 2.4–6.9% of sampled edges; conditional-nonzero mean ≈ −0.13…−0.20.
Gold answers used for offline labeling only.

### M2 — Critic (S4): DONE, with a warning sign
Set-conditioned critic overfits hard (fit RMSE 0.016–0.03 vs val RMSE
0.16–0.25); val RMSE is at or above the predict-zero baseline; the 8-scalar
UNARY control matches or beats the set critic's val RMSE at every g.

### M3 — Calibration (S5): DONE
Frozen: **g=8, width=256, τ=0.207, val macro 66.07 (+0.64 over B2-identity),
exchange rate 0.50**; unary: g=8, w=256, τ=0.107, val macro 67.20 (+1.77).
TTFT: all 12 configs within the 318.9 ms bound (B2 median 289.9 ms; the added
edge-construction + critic window costs 4 ms at g=1, 25–28 ms at g≥2).
These val gains did not transfer (val n=60; the whole τ grid was selectable
offline from cached generations — decision 8 held).

### M4 — Confirmation (S7): DONE, 8 arms × 720, 0 errors anywhere

| Arm | TextVQA | DocVQA | OCRBench | Macro | Δ vs B2 (paired bootstrap 95% CI) | TTFT med |
|-----|---------|--------|----------|-------|-----------------------------------|----------|
| B2  | 74.21   | 54.56  | 62.08    | **63.62** | — | 293.8 ms |
| B1  | —       | —      | —        | 63.30 | −0.003 [−0.028, +0.020] | 330.0 ms |
| **SAGE** | 71.75 | 53.13 | 62.08 | **62.32** | **−0.013 [−0.023, −0.004]** | 307.9 ms |
| UNARY | — | — | — | 62.57 | −0.010 [−0.021, −0.000] | 311.2 ms |
| RND | — | — | — | 61.95 | −0.017 [−0.027, −0.006] | — |
| PERM-X | — | — | — | 62.72 | −0.009 [−0.020, +0.001] | — |
| PERM | — | — | — | 62.32 | −0.013 [−0.023, −0.004] | — |
| Hindsight ceiling | — | — | — | 68.83† | +0.052 mean best Δ on 8.2% of images | — |

† lower bound: best of only 4 sampled edges per image.

SAGE rescue/break: **6 / 15** (ties 699).

## Criteria (decision 13)

| # | Criterion | Result |
|---|-----------|--------|
| C1 | paired macro > B2, CI excluding 0 | **FAIL** (CI excludes 0 on the wrong side) |
| C2 | no benchmark loses > 1 pt | **FAIL** (TextVQA −2.46, DocVQA −1.44) |
| C3 | TTFT within constraint | PASS (307.9 ≤ 318.9 ms) |
| C4 | PERM-X erases the gain | PASS (SAGE−PERM-X −0.004 [−0.015, +0.008]) |
| C5 | UNARY recovers less | PASS in the vacuous direction (SAGE−UNARY −0.002 [−0.016, +0.012]) |

## What the numbers actually say

1. **Every way of exchanging ~8 tokens around B2 loses.** Critic, unary,
   cross-image-rotated, and random edge choice land at −0.9…−1.7 macro; the
   entire ordering signal (SAGE − RND = +0.004, and vs PERM-X +0.004, both
   CIs straddling 0) is worth a fraction of the ~1.3-point cost of exchanging.
   The incumbent's own margin is what the critic is spending.
2. **Survivor conditioning adds nothing over a unary scorer**
   (SAGE−UNARY −0.002 [−0.016, +0.012]) — the same conclusion M3-v2 reached
   for a query-conditioned auditor: conditioning on the surviving set does not
   create answer-relevant signal that the group's own aggregate lacks.
3. **Val calibration did not transfer** — val (60 images) showed +0.6/+1.8 for
   the frozen configs; fresh data shows −1.3/−1.0. Consistent with the repo's
   standing noise floor (M3-v0: seed swing 3.3 macro > every effect).
4. **The opportunity is real but the premise is falsified.** Hindsight over 4
   sampled edges/image reaches +5.2 macro (lower bound on the E(x) ceiling)
   while the critic is at −1.3: per the idea's own falsification clause, the
   early-observability premise — that a pre-LLM critic can predict the paired
   answer effect of a complete exchange — is falsified for this feature/architecture
   class. This extends the project's arc (S2-C1, M3-v2, M6, M7, M9): the value
   sits in whole-set identity, not in any pre-LLM per-edge score this family
   can express.

## Deviations & amendments (all declared before the affected numbers were seen)

- **M10 routing control**: no definition exists in the repo; blocked, not fabricated.
- **PERM degeneracy**: the pre-registered within-image score permutation is
  provably degenerate under eq. 5 deployment (argmax of a permuted list returns
  the same element → PERM ≡ SAGE set-for-set; confirmed: Δ exactly 0.000
  [0, 0]). Amended to **PERM-X** (cross-image score rotation, M2's C1-SHUF
  precedent), declared before its result was seen.
- **B1 harness bug**: the first B1 arm silently ran block8 (m7's `install()`
  hardwires `BASE_SELECTOR`); caught via B1≡B2 on 720/720 predictions with 0/720
  identical sets, fixed, rerun (−0.3 macro, CI includes 0 — the facility/block8
  gap on this panel is within noise).
- **UNARY `family` bug**: first UNARY pass ran the set-critic path (16388-dim φ
  vs 8-dim stats); fixed, rerun clean.

## Compute

~6.5 A40 GPU-hours total (labeling 1.6 h, calibration 0.7 h, confirmation ~4 h,
critic training 0.2 h) — far under the 35–70 GPU-day estimate, which was
dominated by an assumed 2m²-edge labeled support; the pre-registered 8-sampled-edges
protocol kept it bounded.

## Next step

Negative result is conclusive under the pre-registration. → `/auto-review-loop` or
archive per the project's falsification-log convention; the hindsight-ceiling
figure (+5.2 on 8.2% of images) is the reusable number for any successor proposal.
