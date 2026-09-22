# S1 — Offline scoring search on EADP's existing relevance signal

**Scope.** S1 of Stage 2. The question: *can EADP's existing global/dense
relevance signal, re-fused / re-normalised / re-smoothed more sensibly, rank the
answer-bearing tokens materially better?*

**What was done.** A correctness audit of the scoring chain, a verification that
the official numbers reproduce offline, and a zero-model-call sweep over the
rank-changing part of the scoring stage (branch normalization, fusion weight
`alpha`, smoothing strength `lambda`). No gradient or backward-pass scoring, no
full-split run, no change to the selector.

**Status.** Complete. **Verdict: No-Go** — no configuration reaches the required
effect size, and the whole family's ceiling is at chance. Recommendation:
proceed to S2 (causal / gradient scoring).

---

## 0. Executive summary

1. The scoring chain was audited against the saved intermediates and reproduces
   bit-for-bit offline. The official necessary-token mean rank percentile
   (**0.5251**) is reproduced exactly, as are all 15 per-instance values.
2. **Two stages of the official pipeline are provably rank no-ops** — the
   per-image min-max normalisation and the `x**beta` polarisation. Both were
   confirmed rank-neutral empirically on 15/15 instances. `beta` was therefore
   dropped from the sweep entirely, as the brief anticipated.
3. Consequently the only genuinely rank-changing levers downstream of the two
   branches are (a) **branch balance / fusion** and (b) **smoothing strength**.
   Per-branch normalization turns out to be empirically irrelevant (all six
   variants land within 0.004 of each other at matched `alpha`).
4. The sweep's best configuration is `alpha = 1.0, lambda = 0` — i.e. **use the
   global branch only and apply no smoothing**. It improves the primary tier by
   **-0.0552** (i.e. necessary tokens move up), 3/4 localized cases improved.
5. **This is below the required 0.08 and is not a calibration result: it is an
   ablation.** Every gain in the family comes from *removing* a component. The
   dense branch and the Gaussian smoothing both *degrade* answer-relevance
   ranking monotonically; nothing in the fusion/calibration/normalization space
   *adds* information.
6. The ceiling of the entire family is **block AUROC 0.52–0.54** on the three
   localized tiers (official: 0.37–0.41, i.e. *below* chance); the configuration
   that is best on the primary rank metric reaches 0.500. Recall@256 of the
   necessary tokens rises from 0.129 to 0.191 — still four fifths of the
   answer-bearing evidence outside the budget.
7. **Conclusion: the existing similarity-based relevance signal cannot be
   recovered by fusion / calibration / smoothing.** It is at chance for
   answer-relevance at the token level, and the search space defined by the
   brief contains no mechanism that changes this. S1 stops here.

**One correction to the Stage-1 report.** §6.3 states the localized sub-mode is
"7 of 15" cases. It is **4 of 15** (`nNec = 64`); the diffuse count of 7 is
correct. The remaining 4 cases (nNec = 128, 128, 256, 256) are in neither bucket.
This matters because the brief's Go/No-Go criterion "at least 5/7 localized
cases improved" cannot be applied as written to a set of 4. Section 6 evaluates
the criterion under four definitions of "localized"; the verdict is No-Go under
all of them. See §2.3.

---

## 1. Correctness audit

### 1.1 The authoritative chain (from code)

`VisualTokenPruner.compute_importance()` in `Qwen_vl/model/pruner.py:219`, matched
against the instrumented copy in `scripts/discovery/instrumented.py:395`:

| # | Stage | Operation | Code |
|---|---|---|---|
| 1 | global | `global_sim = -einsum(bnc,mc->bnm)` | `_sim_cross` |
| 2 | dense | `local_sim_all = -einsum(bnc,mlc->bnml)` | `_sim_cross` |
| 3 | dense | entropy filter, `T=100`, keep-ratio `q=0.2` | `_entropy_filter_impl` |
| 4 | dense | aggregation, `M_temp=0.01`, `negative_entropy` | `_local_aggregation_impl` |
| 5 | fusion | `text_sim_all = alpha*global + (1-alpha)*local_sim` | inline |
| 6 | fusion | `.mean(dim=-1)` over M | inline |
| 7 | norm | per-image min-max to `[0,1]` | inline |
| 8 | smooth | 3x3 Gaussian, `sigma=1.0`, reflect padding | `_spatial_smoothing_impl` |
| 9 | polar | `importance = importance ** beta` (`beta=2.0`) | inline |

Selector (`_greed_select_impl`) consumes only stages 1–9 and is out of scope here.

### 1.2 Which stages can change the token ranking

| Stage | Rank-changing? | Why |
|---|---|---|
| 1–2 similarity kernels | **no** | affine-invariant under the facility-location marginal gain (Stage-1 §5.5, H2). Also fixed here by construction. |
| 3–4 entropy filter + aggregation | **yes** | this *defines* the dense branch; not a transform of an existing ranking. Not swept (would require re-running the vision tower). |
| 5 **fusion weight `alpha`** | **yes** | combines two branches on different scales; changes the ordering. **Swept.** |
| 6 `.mean(dim=-1)` | no-op here | `M = 1` in this configuration (single global instruction embedding). |
| 7 **per-image min-max** | **no** | positive affine `(x-a)/(b-a)`, `a,b` per-image scalars. Applied *before* the linear smoother it is still rank-neutral, because a linear operator maps a constant field to itself: `G((x-a)/(b-a)) = (G(x)-a)/(b-a)`, a positive affine function of `G(x)`. Applied after, it is trivially rank-neutral. **In every position it could occupy, it is a rank no-op.** |
| 8 **smoothing strength `lambda`** | **yes** | `(1-lambda)*S + lambda*G(S)` is not a uniform monotone transform of `S`. **Swept.** |
| 9 `x**beta` | **no** | strictly monotone on the non-negative smoothed score. |

**Consequence for the search design.** The brief's requested normalization axis
("official / no per-image branch normalization / dataset-level robust scaling")
collapses in the official code: there *is* no per-branch normalization, and the
only normalization present (stage 7) is rank-neutral. To give the axis any
meaning at all it was redefined as **normalization applied to each branch
separately before fusion**, which does change the ranking because it changes the
effective branch balance per image. That is what §4 sweeps.

### 1.3 Numerical verification (no model calls)

`s1_audit.py`:

| Check | Result |
|---|---|
| A. `probe_*.npz` importance == `<ds>_<idx>_b256.npz` importance (same forward pass) | max abs diff `0.000e+00` |
| B. `fused == 0.5*global + 0.5*local` on the raw saved branches | max abs diff `0.000e+00` |
| C. independent numpy reimplementation of stages 7–9 reproduces saved `importance_post_smooth` | max abs diff `1.1e-05` (float32 storage) |
| C. same, reproduces saved `importance` (`**beta`) | max abs diff `1.8e-05` |
| D. `rank(smooth(raw fused)) == rank(saved importance)` | **15/15 instances** |
| D. min-max rank-neutral | `True` (all 15) |
| D. `x**beta` rank-neutral | `True` (all 15) |
| E. official mean necessary-token rank percentile | **0.5251** (Stage-1 report: 0.5251) |
| E. recomputed vs stored `fa_classification.json` per-instance values | max abs diff `0.000e+00` |

Check C is the load-bearing one: it means the numpy reimplementation of
min-max → 3x3 reflect-pad Gaussian → `**beta` is exact, so every sweep arm is
scored by the same operators the official pipeline uses.

---

## 2. Saved artifacts: what actually carries causal labels

### 2.1 Inventory

`outputs/discovery/fa_maps/` holds 171 `.npz` files in two disjoint families:

| Family | n | Keys | Causal labels? |
|---|---|---|---|
| `<ds>_<idx>_b256.npz` | 150 | `global_sim`, `local_sim`, `fused`, `importance_pre_smooth`, `importance_post_smooth`, `importance`, `select_idx`, `text_entropy`, `top_entropy_vals` | **no** |
| `probe_<ds>_<idx>.npz` | 21 | `importance`, `select_idx`, `needed_blocks` | **yes** |

**The 150 saved maps carry no occlusion-derived labels** and must not be treated
as causal cases. Only the 21 probe files do, and of those **6 have an empty
`needed_blocks`** (occluding any single block did not flip the unpruned answer —
no causally necessary region found at this granularity). That leaves **15
instances with usable causal labels**, matching the Stage-1 report's n = 15.

### 2.2 The join is sound

The probe instances are drawn from the same error set as the 150 saved maps, so
every probe instance has a branch-map counterpart. Verified: **21/21** probe
files have a matching `<ds>_<idx>_b256.npz`, and check A confirms both files come
from the same forward pass (bit-identical importance). This is what makes the
zero-model-call sweep possible: `needed_blocks` from the probe file, `global_sim`
/ `local_sim` from the branch-map file.

### 2.3 The 15 causal cases, and the tier definitions

| Case | nNec | blocks | official mean rank pct |
|---|---|---|---|
| DocVQA_VAL_241 | 64 | 1 | 0.6255 |
| TextVQA_VAL_105 | 64 | 1 | 0.5431 |
| TextVQA_VAL_285 | 64 | 1 | 0.5247 |
| TextVQA_VAL_400 | 64 | 1 | 0.5566 |
| DocVQA_VAL_341 | 128 | 2 | 0.4018 |
| TextVQA_VAL_505 | 128 | 2 | 0.6231 |
| OCRBench_222 | 256 | 4 | 0.6253 |
| TextVQA_VAL_609 | 256 | 4 | 0.4729 |
| DocVQA_VAL_119 | 384 | 6 | 0.4880 |
| DocVQA_VAL_593 | 384 | 6 | 0.5554 |
| OCRBench_214 | 448 | 7 | 0.5594 |
| TextVQA_VAL_181 | 448 | 7 | 0.4661 |
| OCRBench_206 | 576 | 9 | 0.5026 |
| DocVQA_VAL_738 | 704 | 11 | 0.4389 |
| DocVQA_VAL_8 | 832 | 13 | 0.4926 |

Tier sizes: `nNec = 64` → **4**; `<= 128` → **6**; `<= 256` → **8**; all → **15**.

**Discrepancy with the Stage-1 report.** §6.3 claims "Localised (nNec = 64, a
single block; 7 of 15)". The data gives 4. The diffuse claim ("nNec >= 384;
7 of 15") is correct, and 7 + 4 = 11, so the report's two buckets also did not
sum to 15. The brief's criterion "at least 5/7 localized cases improved"
therefore cannot be applied literally; §6 evaluates it under all four tier
definitions.

### 2.4 Not available offline

The brief asks, if the saved data permit, for the **necessary-token coverage at
T = 256 under the official facility selector** for each candidate. It does not
permit it: `_greed_select_impl` needs the visual similarity matrix
`_sim_visual_impl(image_features)`, and neither the image features nor the
`N x N` similarity matrix were saved (the 150 files contain only the nine keys
listed above). Re-deriving them requires the vision tower — a model call.
**This item is therefore unavailable by construction, and was not worked around
by re-running the model.** The official coverage number (0.244 for EADP-256,
from the saved `select_idx`) is unchanged and already reported in Stage-1 §6.3.

---

## 3. Baseline sanity check

Reproduced offline from the saved branch maps, before any sweeping:

| Quantity | Value | Reference |
|---|---|---|
| Mean necessary-token rank percentile, all 15 causal cases | **0.5251** | Stage-1 report: 0.5251 |
| Per-instance values vs `fa_classification.json` | max diff `0.0` | exact |
| `fused` from raw branches | max diff `0.0` | exact |
| Official anchor, primary tier (n = 4) | mean 0.5625 | — |
| Official block AUROC, primary tier | 0.367 | below chance |
| Official Recall@256, primary tier | 0.129 | — |

The official pipeline is reproduced exactly, so all deltas below are paired and
attributable to the swept operator alone.

---

## 4. Search space

All configurations are pure functions of the saved `global_sim` / `local_sim`.
No model call, no generation, no selector change.

```
g', d'  = normalize(global_sim), normalize(local_sim)      # per branch
S       = alpha * g' + (1 - alpha) * d'                    # fusion
score   = (1 - lambda) * S + lambda * smooth3x3(S)         # smoothing strength
```

| Axis | Values | Notes |
|---|---|---|
| `branch_norm` | `none` (official), `img_mm`, `img_z`, `ds_robust`, `ds_z`, `img_rank` | `img_mm`/`img_z` per-image; `ds_robust` = shared median/IQR over all 150 saved maps; `ds_z` = shared z-score; `img_rank` = per-image rank normalization (**non-affine**, outside the brief's list, reported separately) |
| `alpha` | 0, 0.25, 0.5, 0.75, 1.0 | 0.5 = official. 0 = dense only, 1.0 = global only |
| `lambda` | 0, 0.25, 0.5, 0.75, 1.0 | 1.0 = official. 0 = no smoothing |
| `beta` | — | **dropped**: provably and empirically rank-neutral (§1.2, §1.3) |
| learned calibration | — | not introduced, per the brief |

Degenerate combinations (single-branch with per-branch scaling) were pruned.
Effective grid: 145 configurations, of which 25 are the non-affine `img_rank`
arm.

**Metrics.** Computed per instance, then aggregated across instances; tokens are
never pooled as independent samples. Block-level metrics use one score per 8x8
block (mean of its 64 tokens) to match the occlusion granularity, with
Mann-Whitney AUROC and AP under tie handling.

---

## 5. Results

### 5.1 The alpha / lambda surface (`norm = none`)

Mean necessary-token rank percentile, lower is better; in parentheses, delta vs
official. **Primary tier (nNec = 64, n = 4), official = 0.5625:**

| | `lam=0` | `lam=0.25` | `lam=0.5` | `lam=0.75` | `lam=1.0` |
|---|---|---|---|---|---|
| `alpha=0` | 0.5423 (-0.020) | 0.5516 (-0.011) | 0.5644 (+0.002) | 0.5807 (+0.018) | 0.5958 (+0.033) |
| `alpha=0.25` | 0.5385 (-0.024) | 0.5465 (-0.016) | 0.5573 (-0.005) | 0.5682 (+0.006) | 0.5810 (+0.018) |
| `alpha=0.5` (official) | 0.5312 (-0.031) | 0.5372 (-0.025) | 0.5448 (-0.018) | 0.5535 (-0.009) | 0.5625 (0.000) |
| `alpha=0.75` | 0.5188 (-0.044) | 0.5229 (-0.040) | 0.5277 (-0.035) | 0.5338 (-0.029) | 0.5401 (-0.022) |
| **`alpha=1.0`** | **0.5072 (-0.055)** | 0.5087 (-0.054) | 0.5115 (-0.051) | 0.5163 (-0.046) | 0.5198 (-0.043) |

The surface is **monotone in both axes, everywhere**:

* **`alpha` → 1.0 (global only) is best at every value of `lambda` and in every
  tier.** The entropy-filtered dense branch does not merely fail to help; it
  *dilutes* the global branch monotonically.
* **`lambda` → 0 (no smoothing) is best at every value of `alpha` and in every
  tier.** Gaussian smoothing monotonically *degrades* the ranking of
  answer-bearing tokens. This is a direct confirmation of Stage-1 §6.4: the
  structure the 3x3 kernel manufactures is not answer-relevant structure.

Both axes point at the same conclusion from opposite directions: the useful
signal is what survives *before* the dense branch and the smoothing are applied,
and both of those stages subtract from it.

### 5.2 Same surface, other tiers (best row per tier)

| Tier | best config | official | candidate | delta | improved / worsened |
|---|---|---|---|---|---|
| nNec = 64 (n=4) | `alpha=1.0, lam=0` | 0.5625 | 0.5072 | **-0.0552** | 3 / 1 |
| nNec <= 128 (n=6) | `alpha=1.0, lam=0.5` | 0.5458 | 0.4966 | -0.0492 | 4 / 2 |
| nNec <= 256 (n=8) | `alpha=1.0, lam=1.0` | 0.5466 | 0.4929 | -0.0537 | 6 / 2 |
| all causal (n=15) | `alpha=1.0, lam=0` | 0.5251 | 0.5011 | -0.0240 | 8 / 7 |

The effect shrinks as the necessary set grows, and on the full causal set it is
8 improved / 7 worsened — a sign test gives two-sided **p = 1.000**.

### 5.3 The normalization axis is flat

At matched `alpha = 0.75, lambda = 0`, primary tier:

| `branch_norm` | delta | block AUROC |
|---|---|---|
| `none` (official) | -0.0436 | 0.483 |
| `img_mm` | -0.0418 | 0.483 |
| `img_z` | -0.0431 | 0.483 |
| `ds_robust` | -0.0438 | 0.483 |
| `ds_z` | -0.0438 | 0.483 |
| `img_rank` (non-affine) | -0.0453 | 0.500 |

Spread across all six: **0.0035**. Per-branch normalization — including a
non-affine rank transform, and including shared dataset-level statistics — makes
essentially no difference once the fusion weight is fixed. This is consistent
with §1.2: since the rank order of a per-branch *affine* rescaling is fully
absorbed by `alpha`, the normalization axis had no independent room to begin
with. **There is nothing here to search.**

### 5.4 Per-case detail, primary tier, best configuration

`alpha = 1.0, lambda = 0` (global branch only, no smoothing):

| Case | nNec | official mean | candidate mean | **delta** | official med | cand med | official AUROC | cand AUROC | official R@256 | cand R@256 |
|---|---|---|---|---|---|---|---|---|---|---|
| DocVQA_VAL_241 | 64 | 0.6255 | 0.5564 | **-0.0691** | 0.5952 | 0.5571 | 0.400 | 0.400 | 0.016 | 0.141 |
| TextVQA_VAL_105 | 64 | 0.5431 | 0.4384 | **-0.1048** | 0.5249 | 0.4644 | 0.333 | 0.733 | 0.156 | 0.266 |
| TextVQA_VAL_285 | 64 | 0.5247 | 0.5400 | **+0.0154** | 0.5464 | 0.5527 | 0.467 | 0.467 | 0.219 | 0.156 |
| TextVQA_VAL_400 | 64 | 0.5566 | 0.4942 | **-0.0624** | 0.5469 | 0.4946 | 0.267 | 0.400 | 0.125 | 0.203 |

Three of four improve; one (TextVQA_VAL_285) regresses slightly. With n = 4 a
paired bootstrap CI on the mean delta is `[-0.0942, -0.0057]` — nominally
excluding zero, but this rests on four instances and should not be read as
evidence of a usable effect. The per-case spread (-0.105 to +0.015) is larger
than the mean.

**Read the block AUROC column with care.** For the `nNec = 64` cases there is
exactly one necessary block out of 16, so block AUROC is nothing more than the
rank of that single block among the 16, quantized to steps of 1/15 — every value
in the table is an exact multiple (0.267 = 4/15, 0.400 = 6/15, 0.733 = 11/15).
It is a far coarser statistic than the 64-token mean rank percentile, and it is
why AUROC is *unchanged* for two of the four cases while the mean rank still
moves. The orderings the two statistics give are not in conflict; the percentile
is simply the finer instrument.

The two statistics also **disagree about which cases are helped**. DocVQA_VAL_241
and TextVQA_VAL_285 have *identical* block AUROC before and after (0.400 and
0.467 respectively), yet the candidate moves them in opposite directions at the
token level (-0.069 and +0.015). A reweighting that redistributes mass across the
token grid — which is precisely what dropping the dense branch and the smoothing
does — can move a 64-token average without changing the standing of the one
necessary block. With n = 4 there is no way to tell a genuine localisation
improvement from that redistribution, and the all-causal tier (8 improved /
7 worsened) is where the mean effect disappears. This is a further reason to
treat the primary-tier delta as indicative at best.

### 5.5 Block-level AUROC / AP and Recall@k

Best configuration (`alpha=1.0, lambda=0`) vs official, per tier:

| Tier | n | block AUROC | block AP | R@128 | R@256 | R@512 |
|---|---|---|---|---|---|---|
| nNec = 64 | 4 | 0.367 → **0.500** | 0.070 → 0.086 | 0.035 → 0.090 | 0.129 → 0.191 | 0.383 → 0.488 |
| nNec <= 128 | 6 | 0.405 → 0.512 | 0.191 → 0.254 | 0.046 → 0.091 | 0.142 → 0.204 | 0.421 → 0.510 |
| nNec <= 256 | 8 | 0.385 → 0.519 | 0.210 → 0.256 | 0.051 → 0.099 | 0.151 → 0.216 | 0.425 → 0.513 |
| all causal | 15 | 0.449 → 0.504 | 0.398 → 0.448 | 0.088 → 0.113 | 0.198 → 0.230 | 0.463 → 0.500 |

**This is the decisive table.** The official score is *below chance* at block
level on every localized tier (0.367 / 0.405 / 0.385 against 0.5), and the
configuration that is best on the primary rank metric reaches **0.500** —
exactly chance. The best block AUROC anywhere in the family is **0.517 / 0.529 /
0.540** on the three localized tiers and 0.515 across all 15 causal cases. Two
different optima, the same conclusion: the family tops out at a coin flip.
Recall@256 for the necessary tokens rises from 0.129 to 0.191 — even the best
calibration leaves **four fifths** of the answer-bearing evidence outside a
256-token budget.

No amount of further search inside this family changes that, because the search
space contains only reweightings of a signal that is already at chance.

---

## 6. Go / No-Go

The brief's criteria, evaluated literally:

| Criterion | Requirement | Measured | Verdict |
|---|---|---|---|
| Primary-tier mean improvement | >= 0.08 absolute | **0.0552** | **fail** |
| Primary-tier cases improved | >= 5 of 7 | 3 of 4 (nNec=64 set has 4 members, not 7) | **fail** |
| No material regression on all causal cases | — | delta -0.024, 8 improved / 7 worsened, sign test p = 1.000 | not a regression, but not a gain either |

Re-evaluated under the other tier definitions, no reading passes: `<= 128` gives
-0.049 (4/6); `<= 256` gives -0.054 (6/8); all 15 gives -0.024 (8/15). **No tier
reaches 0.08, and no tier is unanimous.**

Beyond the numeric threshold, three structural facts make the No-Go the right
call rather than a near-miss:

1. **Every gain is an ablation, not a calibration.** The best arm discards the
   dense branch (`alpha = 1.0`) and discards smoothing (`lambda = 0`). A family
   whose optimum is "do strictly less" has no headroom left to search.
2. **The ceiling is chance.** Block AUROC tops out at 0.500–0.540 on the
   localized tiers and 0.515 across all 15 causal cases, against an official
   0.367–0.449. Even a hypothetical perfect optimiser over this space would hand
   the selector a coin flip.
3. **The proposed gain is a necessary condition, not a sufficient one.** This
   metric asks whether the *score* ranks necessary tokens highly. Stage-1 showed
   34.7 % of errors are ones the unpruned model also makes (no ranking can fix
   them) and that the diffuse cases need up to 832 of 1024 tokens (no 256-token
   budget can hold them). A -0.055 ranking shift would not have translated into
   accuracy without an accuracy run, which by design was not performed.

### Verdict: **NO-GO for S1**

> **Conclusion.** EADP's existing similarity-based relevance signal cannot be
> restored to answer-relevance by fusion, calibration, or smoothing. The
> rank-changing degrees of freedom downstream of the two branches are only
> (i) how the branches are balanced and (ii) how much the map is smoothed; the
> optimum of both is to *remove* a stage, and what remains is statistically
> indistinguishable from random at the block granularity the causal labels are
> defined on. Recommend proceeding to **S2 (causal / gradient scoring)**.

S1 stops here. No further normalization or fusion tuning is warranted; the brief
explicitly rules out unbounded normalization search, and §5.3 shows the axis is
flat to within 0.004 anyway.

---

## 7. Top-3 for the record

Recorded for completeness; **these are not promoted to an accuracy validation**,
because the Go/No-Go failed and none of them clears the effect-size bar.

| Rank | Config | Primary-tier delta | All-causal delta | Note |
|---|---|---|---|---|
| 1 | `norm=none, alpha=1.0, lambda=0` | -0.0552 (3/1) | -0.0240 (8/7) | global branch only, no smoothing |
| 2 | `norm=img_rank, alpha=1.0, lambda=0` | -0.0552 (3/1) | -0.0240 (8/7) | identical outcome via a non-affine rescale, i.e. the same ablation |
| 3 | `norm=none, alpha=1.0, lambda=0.25` | -0.0537 (3/1) | -0.0238 (8/7) | as #1 with residual smoothing |

All three are the same mechanism: **stop using the dense branch.** The
`img_rank` arm tying exactly with `none` at `alpha = 1.0` is itself informative —
with one branch, normalization is provably irrelevant, which is the §5.3 result
in miniature.

*Minimal frozen-set accuracy validation (not triggered).* Had a candidate passed,
the cheapest decisive test would have been an accuracy run on the frozen
150-instance-per-benchmark set already used in Stage-1 Part 2, at budget 256,
swapping only the importance map (keeping the official facility selector) so the
comparison is paired against the existing `facility @256` arm (66.88 mean). Any
such run is gated on also confirming that the map change does not degrade
per-instance coverage, which §2.4 shows cannot be checked offline.

---

## 8. Limitations

* **n = 4 on the primary tier.** The primary set the brief specifies
  (`nNec = 64`) has four members, not seven. Every primary-tier number above
  rests on four instances; the per-case spread exceeds the mean. The all-causal
  tier (n = 15) is more stable and shows the effect collapsing to 8/7.
* **The causal labels are block-level and coarse.** A "necessary block" may be a
  layout anchor rather than the answer glyph (Stage-1 §6.3 states this
  limitation). Block-level AUROC/AP inherit it.
* **The probe subset is not a representative draw** — the first 7 error
  instances per dataset. Unchanged from Stage-1.
* **Ranking is not accuracy.** Nothing here measures model output quality; the
  brief explicitly defers accuracy to a later stage.
* **Coverage under the facility selector is unavailable offline** (§2.4). The
  candidates in §5 therefore cannot be checked for selector-level coverage.
* `beta` and the per-image min-max were removed from the sweep on provable
  rank-neutrality grounds. If a future stage evaluates a *non-rank* consumer of
  the score (e.g. a thresholded sufficiency statistic for D5), they become
  live variables again.

---

## 9. Reproduce

```
source ~/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
cd /media/disk2/YZX/research/EADP
python Qwen_vl/scripts/discovery/s1_audit.py           # audit + baseline sanity check
python Qwen_vl/scripts/discovery/scoring_search_s1.py  # the sweep
python Qwen_vl/scripts/discovery/s1_report_tables.py   # detail tables
python Qwen_vl/scripts/discovery/s1_significance.py    # bootstrap / sign test / ceiling
```

Outputs: `outputs/discovery/s1_scoring_search.json` (all 145 configurations x 15
instances, per-instance metrics). No model call is made by any of the four
scripts.
