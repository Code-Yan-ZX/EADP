# S2-C6 — Residual Signal Localization

**Scope.** S2-C5A left the nonlinear token-local family at **0.8047** held-out
teacher Top-8 recall@256 and explicitly refused to call ~0.80 a ceiling, because
it had never separated two explanations: the 240 fit images are the limit
(**data**), or the single L4 snapshot is the limit and the missing signal lives in
the layer trajectory or in the query (**representation**). S2-C6 separates them.
It trains 27 scorers, changes no architecture outside its arm table, sweeps no
hyperparameter, and ends at a mechanism verdict.

**Status.** Complete. Scripts `s2c6_{common,models,train,consolidate,tables,
figure,query_control}.py`, runner `run_s2c6.sh`. Pre-registration
`scoring_search_s2c6_prereg.md` (written before the first training run; amended
once, in §7). Deliverables: `outputs/discovery/s2c6_audit.json` (main),
`s2c6_train_{A,B,C}.json`, `s2c6_perimage_{A,B,C}.npz`, `s2c6_scores_{A,B,C}.npz`,
`s2c6_query_control.json`, `s2c6_delta_noise.json`, `s2c6_tables.md`,
`figures/s2c6_localization.png`.

**No generation was run.** The downstream gate stayed closed (§6).

---

## HEADLINE

```
S2-C6 -- the residual gap is not in the layer trajectory and not in the query.

(a) THE L4 REFERENCE IS THE PUBLISHED ONE.  Part A's n=240 arm reproduces
    S2-C5A's LOCAL-MLP to the last digit: per-seed 0.7992 / 0.8133 / 0.8017 at
    selected epochs 0 / 1 / 0, exactly the published run.  Part A's n=240 and
    Part B's L4 are the same computation and agree to max |per-image diff| = 0.
    Everything below is measured against that reference, not a reimplementation.

(b) THE FAMILY IS STILL DATA-RESPONSIVE, AND THE PRE-REGISTERED BINARY FITS THE
    EVIDENCE BADLY.  60 -> 240 fit images buys +0.0403 (0.7644 -> 0.8047).  The
    last segment, 180 -> 240, is +0.0125 with a paired CI of [+0.0006, +0.0256]
    that excludes zero and a mean LARGER than the middle segment's +0.0069.  The
    pre-registered DATA-LIMITED flag is nevertheless NOT set, because one of the
    three seeds is flat (-0.0008) and the rule requires all three positive.  So
    the formal verdict is DATA-PLATEAU, and that word must not be read as "the
    family is saturated": the curve has not flattened, the last increment is at
    the margin, and four points cannot resolve it.  Under the overlap-selected
    checkpoints the last segment is +0.0150, the largest of the three.

(c) THE TRAJECTORY CARRIES NOTHING BEYOND THE SNAPSHOT.  L2+L4 and L4+DELTA are
    exactly parameter-matched to L4 (524 673 = 524 673) and land at -0.0008
    [-0.0078, +0.0061] and -0.0039 [-0.0183, +0.0103].  The dual architecture's
    own null, L4+L4 (same architecture, same parameter count, no second
    snapshot), is +0.0019 [-0.0064, +0.0100] -- it does not fire, so L4 remains
    the governing reference.  The arms do read their second input: replacing it
    with another image's costs +0.096 to +0.178 head recall.  It just carries no
    teacher-head information that h4 does not already carry.

(d) THE QUERY CARRIES NOTHING EITHER, AND THIS TIME THE ARM COULD HAVE SHOWN IT.
    Both query arms sit ~0.003 above L4 with CIs spanning zero.  The informative
    one is the post-hoc token-dependent bilinear arm: its query term is 89-97 %
    across-image variation (a genuinely image-conditioned term, not a constant),
    replacing the query moves 1.5 % of the selected 256 tokens, and the teacher's
    Top-8 retention moves by +0.0000.  The pre-registered additive arm turned out
    to be uninformative for a different reason -- the model spent its 33 792
    parameters on a per-image *constant* (1.8-4.6 % across-image variation) --
    which is itself the finding that forced the amendment.

(e) VERDICTS  A DATA-PLATEAU (with the caveat under (b)) | B SNAPSHOT-SUFFICIENT
    | C not supported | D not supported.  The two representation hypotheses this
    design can test are both refuted; the data route demonstrably still pays and
    is left open.
```

---

## 1. The open item, and what would settle it

`S2-C5A` §6 closed with an explicit refusal:

> "The evidence supports: *under the current L4 representation, training setup and
> tested model family, adding width or simple global/set context yields no further
> gain.*" — not "*~0.80 is the ceiling of every possible token-local function*".

S2-C4's oracle is what makes that refusal bite: a direction fitted on the image's
own teacher map keeps 0.95 of the Top-8. That oracle is **in sample** — it fits
4 096 free parameters to the 1 024 tokens of the image it is then scored on — so
without S2-C6 it could not be told apart from "the representation has more in it
than a shared function can read". S2-C6 asks whether the missing signal is
forward-readable from something the current scorer does not look at.

Two candidates were named in the brief, and one bound:

| candidate | why it could carry signal | arm |
|-----------|---------------------------|-----|
| the layer trajectory | h4 is one snapshot of an evolving residual stream; the update Δ = h4-h2 has norm 40 % of ‖h₄‖ and cos(Δ, h₂) = 0.22 (77° apart), so it is a large, largely-distinct direction rather than a small correction | `L2`, `DELTA`, `L2+L4`, `L4+DELTA` |
| the query | the teacher is a *question-conditioned* gradient map, and no scorer in S2-C1→S2-C5 sees the question | `L4+QUERY`, `L4+QRY-BILIN` |
| (bound) the training set | 240 images may simply be too few | Part A |

---

## 2. Design

Frozen from S2-C5A and re-tuned nowhere: features `s2c1_feats_L{2,4}.npy` (465,
1024, 4096, fp16), query `s2c1_query_L4.npy`, split fit 240 / val 60 / held-out
150 (80/20/50 per benchmark), target `HEAD_RANK`, loss = balanced BCE + 0.5 ×
weighted all-pairs margin ranking, AdamW 1e-3 / 1e-4, batch 8, `MAX_EPOCHS 80`,
`PATIENCE 20`, seeds 0/1/2, deterministic kernels.

* **Primary metric** held-out teacher Top-8 recall@256.
* **Checkpoint selection** argmax over epochs of validation teacher Top-8
  recall@256 (`H8`). The `OV` index (validation Top-256 overlap) is tracked inside
  the same trajectory and reported as a robustness check; it selects nothing.
* **The held-out 150 is never used to select anything.** No held-out quantity is
  computed inside an epoch loop. It is measured once per frozen checkpoint.
* **`RESOLVED GAIN`** = mean paired delta ≥ 0.01 **and** paired-bootstrap 95 % CI
  lower bound > 0 **and** all three seed-matched deltas > 0. Anything with a CI
  excluding zero that fails the rest is reported as `BELOW-MARGIN` and carries no
  verdict.

**Parameter matching is exact, and it is the point.** Every arm shares the read-out
`s = w2 · GELU(a + b)`, `a ∈ R^128`, and the input projections satisfy

```
single-input   P: 4096 -> 128                     4096*128 + 128 = 524 416
dual-input     2 x (4096 -> 64)                   2*(4096*64 + 64) = 524 416
```

so `L4`, `L2`, `DELTA`, `L2+L4`, `L4+DELTA` and the control `L4+L4` are all
**524 673** parameters. There is deliberately no 8192 → 128 arm: that would double
the projection and make any gain uninterpretable.

**Two equivalences that shape how the results must be read.**

1. Since `h4 = h2 + Δ`, the input pairs `{h2, h4}` and `{h4, Δ}` are related by an
   invertible linear map. `L2+L4` and `L4+DELTA` are therefore **the same
   hypothesis in two parameterizations**, not two independent tests. Their
   agreement (-0.0008 and -0.0039) is a consistency check that passed;
   disagreement would have been optimization noise, not a mechanism.
2. The pre-registered `L4+QUERY` adds one 128-vector *per image* to **every**
   token. Its query contribution is identical across the image's 1 024 tokens, so
   it can reorder tokens only through the final GELU's nonlinearity and cannot
   express "this token matters for this question" — which is what query
   conditioning means. This is why §7 amends the pre-registration with
   `L4+QRY-BILIN`, a token-dependent rank-8 bilinear term at an **identical**
   added budget (33 792 parameters, 558 465 total for both).

**Controls.** `L4+L4` (trained) is the dual architecture's own null. At held-out
inference, each dual arm is re-scored with another image's channel substituted —
twice, borrowing channel 1 and channel 0 separately, so an arm that reads the
*pairing* is hurt by both and an arm that merely reads one channel hard is hurt by
exactly one. Each query arm is re-scored with another image's query. All
substitutions are derangements. The honest pass is re-run through the same code
path as a plumbing self-check (`max|Δscore| = 0` everywhere).

**Storage-noise check on `DELTA`.** The features are fp16, so `h4 - h2` could in
principle be a quantisation-noise detector. Measured before any delta arm was
trained: per-dimension σ has minimum 0.188 against an fp16 absolute floor of
2.3e-4 — a ratio of **812×** — with **zero** dimensions below 1e-3 and a maximum
standardisation gain of 5.3×. The delta arm is not measuring storage noise.

---

## 3. Part A — teacher-data scaling

Only `n` changes. Subsets are nested prefixes of one fixed benchmark-stratified
permutation (`SUBSET_SEED = 20260923`), so `S(60) ⊂ S(120) ⊂ S(180) ⊂ S(240)`.
Validation stays 60 and held-out stays 150 at every `n`. The standardisation
mean/std is recomputed **on the n images the arm may see** — reusing the 240-image
statistics for the n=60 arm would leak the 180 images that arm is pretending not
to have.

| n | R@8 | R@16 | R@32 | ov256 | per-seed R@8 | init R@8 | epoch |
|---|-----|------|------|-------|--------------|----------|-------|
| 60 | 0.7644 | 0.7362 | 0.6874 | 0.5130 | 0.7675 / 0.7608 / 0.7650 | 0.2269 | 1, 2, 2 |
| 120 | 0.7853 | 0.7581 | 0.7047 | 0.5152 | 0.7900 / 0.7800 / 0.7858 | 0.2247 | 2, 1, 1 |
| 180 | 0.7922 | 0.7621 | 0.7067 | 0.5072 | 0.7767 / 0.7975 / 0.8025 | 0.2256 | 0, 1, 1 |
| 240 | **0.8047** | 0.7751 | 0.7135 | 0.5078 | 0.7992 / 0.8133 / 0.8017 | 0.2258 | 0, 1, 0 |

| segment | delta | 95 % CI | per-seed | resolved |
|---------|-------|---------|----------|----------|
| 60 → 120 | +0.0208 | [+0.0067, +0.0350] | +0.022 / +0.019 / +0.021 | **yes** |
| 120 → 180 | +0.0069 | [-0.0061, +0.0197] | -0.013 / +0.018 / +0.017 | no |
| **180 → 240** | **+0.0125** | **[+0.0006, +0.0256]** | +0.022 / +0.016 / **-0.001** | **no** |

**The n=240 row is the published run.** Per-seed 0.7992 / 0.8133 / 0.8017 at
epochs 0 / 1 / 0 is S2-C5A's `LOCAL-MLP` under `H8` selection, to the last digit.
That is not a coincidence being asserted: `channel_stats("h4", fit)` returns
bit-identical statistics to `s2c1_train.standardize_stats` (verified, max |diff| =
0), the training loop draws its RNG in the same order as `s2c5a_train.main`, and
deterministic kernels are forced. Part A's n=240 and Part B's `L4` are the same
computation and agree at max |per-image diff| = **0.00e+00**.

### What the curve says

Total 60 → 240 is **+0.0403**. The increments are +0.0208, +0.0069, +0.0125 —
they do **not** decay monotonically; the last is nearly twice the middle one. Its
CI excludes zero. Under `OV` selection the same curve gives +0.0147, +0.0119,
+0.0150, so the last segment is again the largest.

The pre-registered rule asks for mean ≥ 0.01, CI lower bound > 0 **and** all three
seed-matched deltas > 0. The first two hold (+0.0125, lower bound +0.0006); the
third does not, because seed 2 is flat at -0.0008. So:

> **Formal verdict: DATA-PLATEAU. The `DATA-LIMITED` flag is not set.**

**This must not be read as "the family is data-saturated at 240."** The
pre-registration's own wording for `DATA-PLATEAU` is a statement *about that
segment at this design's resolution*, not proof of a plateau, and the numbers sit
exactly where the design runs out of resolution: a +1.25 pt mean on 150 images
with a CI lower bound of +0.0006. The substance and the label disagree, and the
honest report is both: the rule did not fire, and the curve has not flattened.
Consequently ~0.80 still cannot be called a representation ceiling — which was
S2-C5A's position and is now directly supported rather than merely withheld.

**Everything in Part A is learned, not architecture.** The untrained model scores
0.2247-0.2269 held-out R@8 at every `n`, against a chance level of 0.25 for a
random 256-of-1024 subset. The architecture contributes nothing on its own; the
whole ~0.80 is fitted.

---

## 4. Part B — representation localization

| arm | R@8 | R@16 | ov256 | vs governing ref | 95 % CI | per-seed | resolved | params | init |
|-----|-----|------|-------|------------------|---------|----------|----------|--------|------|
| **L4** (reference) | **0.8047** | 0.7751 | 0.5078 | — | — | 0.7992 / 0.8133 / 0.8017 | — | 524 673 | 0.2258 |
| L2 | 0.7906 | 0.7588 | 0.4980 | **-0.0142** | [-0.0250, -0.0036] | -0.013 / -0.018 / -0.012 | no | 524 673 | 0.2117 |
| DELTA | 0.7975 | 0.7660 | 0.5039 | -0.0072 | [-0.0264, +0.0111] | +0.007 / -0.018 / -0.011 | no | 524 673 | 0.2850 |
| L2+L4 | 0.8039 | 0.7742 | 0.5070 | -0.0008 | [-0.0078, +0.0061] | -0.005 / -0.002 / +0.004 | no | 524 673 | 0.2244 |
| L4+DELTA | 0.8008 | 0.7743 | 0.5130 | -0.0039 | [-0.0183, +0.0103] | +0.003 / -0.016 / +0.002 | no | 524 673 | 0.2225 |
| L4+L4 (control) | 0.8067 | 0.7756 | 0.5102 | +0.0019 | [-0.0064, +0.0100] | +0.005 / -0.001 / +0.002 | no | 524 673 | 0.2308 |

**The architecture control does not fire.** `L4+L4` — the dual architecture with no
second snapshot — is +0.0019 [-0.0064, +0.0100] over `L4`. So the dual
construction is worth nothing by itself and `L4` remains the governing reference
for `L2+L4` and `L4+DELTA`. (Its two substitution controls, which are the same
manipulation applied to the same channel, return 0.671 and 0.673 — the control
implementation agrees with itself.)

**No snapshot or trajectory arm clears the reference.** The two trajectory arms
land at -0.0008 and -0.0039, i.e. on top of `L4`, and — as §2 predicted, since
they are one hypothesis in two parameterizations — they agree with each other. The
best arm in the table is the control.

**The arms do use their second input.** The wrong-image substitution controls are
all strongly resolved:

| arm | borrow channel 1 | borrow channel 0 |
|-----|------------------|------------------|
| L2+L4 | +0.1781 [+0.1542, +0.2028] | +0.0961 [+0.0783, +0.1142] |
| L4+DELTA | +0.1469 [+0.1244, +0.1694] | +0.1158 [+0.0969, +0.1356] |

(positive = honest − wrong, i.e. how much head recall is lost when that half of
the input is another image's). Both halves matter to both arms, so "the extra
channel was ignored" is ruled out. The asymmetry — borrowing `h4` costs more than
borrowing `h2` — says the arms lean on the current state more than on the history,
which is what one would expect if `h2` acts as a second, noisier view of the same
visual content rather than as trajectory information. That is consistent with the
arm failing to beat `L4`, and it is the mechanism, not a separate result.

**`L2` is reliably worse, not merely not better.** -0.0142 with all three
seed-matched deltas negative and a CI excluding zero. S2-C1 found `LIN_L2 ≈
LIN_L4` for the *linear* probe; for the nonlinear token-local scorer the layer
choice is not neutral, and the later snapshot is the better one. `DELTA` alone
recovers most of `L4` (0.7975) — unsurprising given ‖Δ‖ is 40 % of ‖h₄‖ and
cos(Δ, h₂) = 0.22, so the change retains a good part of `h₄`'s direction — but it
does not exceed it.

---

## 5. Part C — query re-check under the current objective

| arm | R@8 | R@16 | vs L4 | 95 % CI | per-seed | resolved | params (query) |
|-----|-----|------|-------|---------|----------|----------|----------------|
| L4+QUERY (pre-registered) | 0.8075 | 0.7785 | +0.0028 | [-0.0050, +0.0106] | +0.007 / +0.006 / -0.004 | no | 558 465 (33 792) |
| L4+QRY-BILIN (amendment) | 0.8081 | 0.7756 | +0.0033 | [-0.0031, +0.0097] | +0.015 / -0.007 / +0.003 | no | 558 465 (33 792) |

Neither arm clears `L4`. The interesting part is *why*, and the two arms fail for
opposite reasons — which is exactly why the second one was needed.

**The pre-registered arm never learned to condition.** Its query term grows from
exactly zero at init to a norm of 2.0-4.4 at the selected checkpoint, so the
branch trained and is not dead. But only **1.8-4.6 %** of that norm is
across-image variation; the rest is a per-image constant. Given 33 792 free
parameters and a real additive budget, the optimiser on 240 images chose a static
bias. A null from an arm that never conditioned says nothing about whether the
query carries information — so this arm does not close the brief's question.

**The token-dependent arm did condition, and it still did not help.** Its query
term is **89-97 %** across-image variation — a genuinely image-specific term — and
replacing the query moves held-out scores by up to 1.08. What that does to the
selection (`s2c6_query_control.py`):

| arm | changed fraction of the selected 256 | head R@8 honest | wrong | delta |
|-----|--------------------------------------|-----------------|-------|-------|
| L4+QUERY | 0.0014 | 0.8075 | 0.8078 | -0.0003 |
| **L4+QRY-BILIN** | **0.0149** | **0.8081** | **0.8081** | **+0.0000** |

So the question's content *does* reach the scorer, *does* re-rank ~1.5 % of the
selected tokens, and changes the teacher's Top-8 retention by **+0.0000**. The
conditioning is real and it lands on tokens that do not matter. That is the
informative null, and it is the answer to the brief's question: under the
`HEAD_RANK` objective, query conditioning provides no information about which
tokens the teacher's gradient map holds to be important.

S2-C1's `QRY` conclusion (`QueryConditioned`, old objective) is therefore
**reproduced under the current objective**, and now with a control that separates
"the query is uninformative" from "the arm ignored the query".

---

## 6. Verdicts, and the downstream gate

```
A. DATA-LIMITED              NOT SET   (formal verdict DATA-PLATEAU; see §3 --
                                        the rule missed on one flat seed, and
                                        the curve has not flattened)
B. SNAPSHOT-SUFFICIENT       SET       no arm in {L2, DELTA, L2+L4, L4+DELTA}
                                        is a RESOLVED GAIN over L4, and the
                                        L4+L4 control that would have governed
                                        the dual arms did not fire
C. TRAJECTORY-INFORMATIVE    NOT SET   L2+L4 -0.0008, L4+DELTA -0.0039
D. QUERY-INFORMATIVE         NOT SET   L4+QUERY +0.0028, L4+QRY-BILIN +0.0033,
                                        both CIs spanning zero, and the
                                        token-dependent arm's shuffled-query
                                        control is exactly null (+0.0000)
INDETERMINATE-TRAJECTORY     NOT SET   no trajectory arm is BELOW-MARGIN; they
                                        are flat, not directionally positive
```

A is independent of B/C/D and the pre-registration allowed them to co-occur. Here
B is set and C, D are not.

**Selection-invariance.** Every arm is at or below `L4` under `OV` selection too
(`L2` 0.7953, `DELTA` 0.7933, `L2+L4` 0.7956, `L4+DELTA` 0.7822, `L4+L4` 0.7950,
`L4+QUERY` 0.7975, `L4+QRY-BILIN` 0.7908, against `L4` 0.7972). The verdicts do
not depend on which checkpoint rule is used.

**Downstream gate: closed.** It required a deployable forward-only arm with
held-out R@8 ≥ 0.82, a paired CI lower bound > 0 against `L4`, and a consistent
seed pattern. The best arm in the stage is `L4+QRY-BILIN` at 0.8081 with a CI
lower bound of -0.0031; no arm reaches 0.82. **No generation was run**, as
pre-registered.

---

## 7. Amendment record

The pre-registration fixed `Part C` at one arm, `L4+QUERY`, with a conditional
`L4+QUERY(shuffled-train)` follow-up if it won. Neither the follow-up condition
nor the arm survived contact with the result, and one change was made:

> **Amendment 1 (post-hoc, after the 27-run ladder was read).** Added
> `L4+QRY-BILIN`: a token-dependent rank-8 bilinear query term,
> `s_i = w2·GELU(a_i + b) + (Wv a_i)·(Wq q)/√r`, at an **identical** added budget
> to the pre-registered arm (33 792 parameters; 558 465 total for both).
>
> **Reason.** The pre-registered arm adds one 128-vector per image to every token,
> so its query contribution is constant across the image's tokens and it cannot
> express token-specific query relevance. Its measured behaviour — the model
> learned a per-image constant from it (§5) — showed the arm does not answer the
> question it was built to answer. The amendment closes that hole rather than
> opening a search: a *single* arm, fixed in advance of running it, at a matched
> budget, and reported under the same pre-registered rule as everything else.
>
> **Effect on the verdict.** None in direction, decisive in strength. The
> token-dependent arm is what makes `D: NOT SET` an informative null instead of an
> inconclusive one. Had it cleared `L4` with its shuffled-query control firing, D
> would have been SET — it did not.
>
> The pre-registered arm is retained and reported alongside it; nothing was
> removed or relabelled.

The `L4+L4` architecture control and the dual arms' two substitution controls are
not amendments: `L4+L4` is named in the pre-registration's arm table, and the
two-variant substitution is the pre-registered "WRONG-TRAJECTORY" control
implemented symmetrically.

---

## 8. What this changes, and what it does not

**Established by S2-C6:**

* The residual gap is **not** locatable in the layer trajectory. Two
  parameter-matched dual-snapshot arms, one token-local function of the change
  alone, and a same-architecture control that would have explained away a dual
  gain — none of them moves held-out head recall above the single L4 snapshot.
* The residual gap is **not** locatable in the query, under a query-conditioned
  rank-8 term that provably reaches the scores and provably re-ranks part of the
  selection.
* The token-local family is **data-responsive**, not saturated: +4.0 pt from
  60 → 240 fit images, with the final increment at the margin of what 150
  held-out images and 3 seeds can resolve.

**Not established, and not claimed here:**

* That the residual gap *is* data-limited. S2-C6 measured four points; the curve
  has not flattened at the largest one, so the stage cannot say where it ends.
  It can only say the data route is still open while the two representation
  routes it could test are closed.
* That ~0.80 is a ceiling of every token-local function. `LOCAL-MLP-WIDE` in S2-C5
  and `L4+QRY-BILIN` here both sit marginally above the reference without
  resolving, and no arm in either stage reaches 0.82.
* That representation transition is novel. TransPrune already prunes on
  transition magnitude/direction. Nothing here is a novelty claim about
  transitions; the licensed claim is the mechanism one — the gradient-defined
  high-value ranking-head tokens do not carry a forward trajectory signature, in
  either the earlier snapshot or the change, that a single L4 snapshot lacks.
* Anything about layers beyond L2 and L4. Only the two layers S2-C1 cached were
  available, and this stage added no forward pass.

---

## 9. Open items

1. **Where does the data curve end?** The one live route. All features are cached,
   so extending to `n > 240` needs either more fit images (a new feature-cache
   pass over additional dataset instances) or a different data source. The
   standardisation-follows-the-fit-set rule would have to be maintained.
2. **Is the H8 checkpoint rule the right comparator?** S2-C5A flagged that
   validation `head_recall@8` peaks at epoch 0-2, so every arm here is frozen near
   the start of training. The verdicts are selection-invariant under `OV` (§6), so
   this does not change the conclusions — but both rules pick early checkpoints,
   and neither stage has tested a rule that trains to convergence.
3. **The `BELOW-MARGIN` class was never exercised.** No arm in this stage landed
   there, so the pre-registered distinction between "resolved", "below-margin" and
   "no effect" remains untested in practice.
4. **`L2` being reliably worse than `L4`** for the nonlinear family, where S2-C1
   found the linear probes tied, is unexplained. It is not load-bearing for any
   verdict here.

**S2-C6 stops here.** No S2-C7 and no final method are designed from it.
