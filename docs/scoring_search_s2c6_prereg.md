# S2-C6 — Residual Signal Localization (pre-registration)

**Written before any S2-C6 training run.** Frozen, dated, and not edited after the
first ladder result is read. If a rule below turns out to be wrong, it is amended
in `scoring_search_s2c6.md` with the amendment dated and the pre-registered
version quoted, not silently overwritten here.

---

## 0. What S2-C6 inherits, verbatim

Everything not named as a degree of freedom in this stage is the S2-C5A
head-selected configuration, unchanged:

| item | value |
|------|-------|
| model | Qwen3-VL-8B, frozen, never loaded — all features cached |
| features | `s2c1_feats_L{2,4}.npy`, (465, 1024, 4096) fp16 |
| query | `s2c1_query_L{4}.npy`, (465, 4096) fp16, the last prompt position at layer L |
| split | fit 240 / val 60 / held-out 150 / causal 15, image-level, 80/20/50 per benchmark |
| teacher | S2-B P1-G2 gradient maps, `s2b_gradient_scores.npz` |
| target | `HEAD_RANK`: positives = teacher Top-32, pair weights `32/(r+1)` mean-normalised, `pos_weight = 31` |
| loss | balanced BCE + `0.5 ×` weighted all-pairs margin ranking, margin 1.0 |
| optimizer | AdamW, lr 1e-3, wd 1e-4, batch 8 |
| budget | `MAX_EPOCHS = 80`, halt at `PATIENCE = 20` of validation Top-256 overlap |
| preprocessing | fit-split per-dimension mean/std, computed on the fit rows the arm is allowed to see |
| seeds | 0, 1, 2 |
| kernels | deterministic forced (`torch.use_deterministic_algorithms(True)`) |

**Primary metric.** Held-out (150) teacher **Top-8 recall@256**.

**Checkpoint selection.** Argmax over epochs of **validation (60) teacher Top-8
recall@256** — the S2-C5A `H8` rule. The halt rule stays S2-C5's overlap patience,
so the trajectory is the same trajectory S2-C5A audited; both the `H8` and the
`OV` index are tracked inside the one trajectory and the `OV` reading is reported
as a selection-invariance check, but **all verdicts below are computed on `H8`**.

**The held-out 150 is never used for any selection.** No held-out quantity is
computed inside an epoch loop anywhere in this stage. It is measured once per arm
after the checkpoint is frozen.

---

## 1. The three questions

S2-C5A left the token-local family at **0.8047** (LOCAL-MLP, head-selected)
against an S2-C4 oracle of **0.95** for a direction fitted on the image itself.
That oracle is *in sample* -- it fits 4 096 free parameters to the 1 024 tokens of
the same image it is then scored on -- so it is a statement about how much a
shared function has to learn, not about what the L4 snapshot forward-contains,
and it is not evidence that the gap is closing. S2-C5A declined to call ~0.80 a
ceiling of the family, because two explanations were never separated:

* the 240 training images are the limit (**data**), or
* the single L4 snapshot is the limit, and the missing signal sits in the layer
  trajectory or in the query (**representation**).

S2-C6 separates them. It is mechanism discovery: no final method, no architecture
search, no hyperparameter search, no layer sweep beyond the two layers S2-C1
already cached.

---

## 2. Part A — teacher-data scaling

Fixed: the S2-C5A `LOCAL-MLP` architecture, target, loss and every training
setting. The only degree of freedom is `n`, the number of **fit** images:

```
n ∈ {60, 120, 180, 240}     20 / 40 / 60 / 80 per benchmark
```

* Subsets are **nested** (`S(60) ⊂ S(120) ⊂ S(180) ⊂ S(240) = fit`) and
  **benchmark-stratified**, built as the prefix of one fixed random permutation
  *within each benchmark*, drawn once from `SUBSET_SEED` and never re-drawn.
* Validation stays the frozen 60, held-out stays the frozen 150, at every `n`.
* Standardization mean/std is recomputed **on the n images the arm is allowed to
  see**. Reusing the 240-image statistics for the n=60 arm would leak the 180
  images the arm is pretending not to have, so it is not done. This is
  preprocessing that follows the fit set, not a second degree of freedom.
* 3 seeds per `n`; primary metric held-out R@8, supplementary R@16/R@32.
* Paired bootstrap CI on every segment of the curve.

### Pre-registered reading

Let `Δ(n₁→n₂)` be the seed-averaged paired held-out R@8 difference between the
two arms, with its 95 % paired-bootstrap CI over the 150 held-out images.

```
DATA-LIMITED   Δ(180→240) is a RESOLVED GAIN (§4)
               -> the curve is still materially rising at the largest n this
                  stage can train, so ~0.80 may not be read as a representation
                  ceiling, and the representation arms in Part B are read
                  against a moving floor.
DATA-PLATEAU   Δ(180→240) is not a RESOLVED GAIN
               -> the token-local family is not obviously data-limited at 240;
                  the Part B arms are read against a settled floor.
```

Whatever the verdict, the whole curve and all three segments are reported; a
non-monotone curve is reported as such rather than smoothed.

Note this stage cannot *prove* a plateau — four points cannot distinguish a
plateau from a very slowly rising curve. `DATA-PLATEAU` is a statement about
180→240 at the resolution this design has, and is worded that way.

---

## 3. Part B / Part C — representation localization

### 3.1 Arms and parameter matching

Every arm shares one read-out shape, `s = w2 · GELU(a + b)` with `a ∈ R^128`,
`w2: 128 → 1`, `b ∈ R^128`.

| arm | `a` | input projections | params |
|-----|-----|-------------------|--------|
| **L4** (reference) | `GELU(P h₄)` | `P: 4096 → 128` | 524 673 |
| **L2** | `GELU(P h₂)` | `P: 4096 → 128` | 524 673 |
| **DELTA** | `GELU(P Δ)` | `P: 4096 → 128`, `Δ = h₄ − h₂` | 524 673 |
| **L2+L4** | `[GELU(P₂ h₂) ; GELU(P₄ h₄)]` | `P₂, P₄: 4096 → 64` | **524 673** |
| **L4+DELTA** | `[GELU(P₄ h₄) ; GELU(P_d Δ)]` | `P₄, P_d: 4096 → 64` | **524 673** |
| **L4+L4** (architecture control) | `[GELU(P_a h₄) ; GELU(P_b h₄)]` | `P_a, P_b: 4096 → 64` | **524 673** |
| **L4+QUERY** (Part C) | `GELU(P h₄) + B u` | `P: 4096 → 128`, `u = W_q q ∈ R^8` | 558 465 |

The dual-input arms are **exactly** parameter-matched to the single-input arms:
`2 × (4096 × 64 + 64) = 4096 × 128 + 128 = 524 416`. There is no 8192 → 128 arm
in this stage and no reading of a dual-input gain as a representation gain unless
it survives the `L4+L4` control below. `L4+QUERY` is **not** capacity-matched to
the rest: it adds 33 792 parameters (6.44 %) against `GLOBAL-CTX`'s 32 768
(6.25 %) in S2-C5, and its parameter count is reported with every number it
produces.

Each channel is standardized by its own fit-split per-dimension mean/std:
`h₂` by `(μ₂, σ₂)`, `h₄` by `(μ₄, σ₄)`, `Δ` by `(μ_Δ, σ_Δ)`, all computed on the
240 fit rows. The per-dimension σ of `Δ` is checked against the fp16 storage
quantization before any arm is trained; if any dimension's σ approached the
quantization floor the delta arm would be measuring storage noise, so that
check is reported rather than assumed away.

### 3.2 Controls

* **L4+L4** (trained). Same dual architecture, same parameter count, *no second
  snapshot*. This is the optimization/capacity control: if the dual architecture
  is better than the single one on its own, the trajectory arms must be read
  against `L4+L4` and not against `L4`. Pre-registered: **if `L4+L4` is itself a
  RESOLVED GAIN over `L4`, then TRAJECTORY-INFORMATIVE requires the trajectory
  arm to be a RESOLVED GAIN over `L4+L4`.**
* **WRONG-TRAJECTORY** (inference, no training). For each dual arm, re-score each
  held-out image with *another* image's second channel substituted (a
  derangement). This is S2-C5's wrong-image context control in its trajectory
  form: a gain that survives its own trajectory being replaced was never reading
  the matched pair.
* **QUERY-SHUFFLE** (inference, no training). For `L4+QUERY`, re-score each
  held-out image with another image's query, same derangement construction.

### 3.3 Part C scope

One arm, `L4+QUERY`. It exists only to answer whether query conditioning carries
independent information under the correct ranking-head objective; S2-C1's `QRY`
conclusion was reached under the old objective and does not transfer. If — and
only if — `L4+QUERY` is a RESOLVED GAIN over `L4`, one follow-up arm is run,
`L4+QUERY(shuffled-train)`, in which every training image reads another training
image's query, to separate "the query's content carries information" from "the
query path is extra capacity". No cross-attention, no query-conditioned
architecture search, no third arm unless that condition holds.

---

## 4. The decision rule

```
MARGIN = 0.01 held-out head_recall@8        (the S2-C5 / S2-C5A constant)

RESOLVED GAIN (A over B)
    mean paired delta >= MARGIN
    AND paired-bootstrap 95 % CI lower bound > 0   (10 000 draws, over the 150
                                                    held-out images, on the
                                                    seed-averaged difference)
    AND all three seed-matched deltas > 0

BELOW-MARGIN
    paired CI lower bound > 0 but not all RESOLVED conditions met
    -> reported as a directional result; carries no verdict
```

`BELOW-MARGIN` exists so that a real but sub-margin effect is not silently
recorded as "no effect". It also never becomes a verdict, so a 0.4 pt difference
cannot be promoted into a mechanism by adding adjectives.

### Verdicts

```
A. DATA-LIMITED            Δ(180→240) is a RESOLVED GAIN.
B. SNAPSHOT-SUFFICIENT     none of {L2, DELTA, L2+L4, L4+DELTA} is a RESOLVED
                           GAIN over L4 (or, when the L4+L4 control fires, none
                           is a RESOLVED GAIN over the control that governs it).
C. TRAJECTORY-INFORMATIVE  L2+L4 or L4+DELTA is a RESOLVED GAIN over L4 — and,
                           if L4+L4 is itself a RESOLVED GAIN over L4, over
                           L4+L4 instead.
D. QUERY-INFORMATIVE       L4+QUERY is a RESOLVED GAIN over L4 AND the
                           QUERY-SHUFFLE control shows the gain depends on the
                           image's own query: honest − shuffled >= MARGIN with a
                           paired CI lower bound > 0.
```

A is independent of B/C/D and may hold alongside any of them. B/C/D are mutually
exclusive by construction except that C and D may co-occur (a trajectory arm and
a query arm can both clear), in which case both are reported.

If a trajectory arm lands at `BELOW-MARGIN`, the stage reports
**INDETERMINATE-TRAJECTORY** — neither B nor C — and says so explicitly rather
than resolving it in the direction of the null.

### Downstream gate

Generation is run **once** only if some deployable forward-only arm satisfies
*all* of:

1. held-out R@8 clearly above `L4` (paired CI lower bound > 0),
2. consistent across all or most seeds,
3. held-out `R@8 >= 0.82` (aspirational 0.85).

If the gate opens, the run is exactly three arms — the best arm, the `L4`
baseline, and the gradient teacher reference — with no sweep. If it does not
open, no generation is run and the stage stops at the offline verdict.

---

## 5. What S2-C6 will not claim

* Not "token transition is novel". TransPrune already prunes on representation
  transition magnitude/direction. If the trajectory arm wins, the only claim
  licensed is a *mechanism* one: gradient-defined high-value ranking-head tokens
  may carry a learnable forward trajectory signature that a single L4 snapshot
  does not expose.
* Not that the tested arms bound every possible function of the trajectory or of
  the query. A null here is a null over this arm set, this target and this split.
* Not a final method. S2-C6 ends at the mechanism verdict; no S2-C7 is designed
  from it.
