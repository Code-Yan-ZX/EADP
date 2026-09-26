# M5 — Safe Removal (SafeTrim)

**Stage:** M5, first round of a **new method line**. MissGuard / gradient-head
distillation (M3), residual capsules (M4), conditional utility (S3-A) and bundle
prediction (S3-B) are all closed and this document does not reopen any of them.
**Status:** pilot, held-out 150, forward-only, pre-LLM.

**The question this document answers, and the only one.**

> Instead of predicting *which tokens are most important and should be kept*,
> can we predict *which tokens of an already-chosen retained set can be removed
> safely* — and is that strictly easier?

Four stages closed the importance question: the teacher's head is not
recoverable from pre-LLM features (S2-C2, S3-B, M3-v0, M3-v2 — `Top16
recall@16 ≈ 0.236`). Safe Removal changes the question rather than the model
class. It never looks at the 768 tokens B2 dropped and never ranks value. It
asks the opposite question about the 256 tokens B2 already kept.

**Why this line exists.** M4's grid produced one lead it could not use.
`EVICT-r8` — delete the 8 most redundant retained anchors, put **nothing**
back, end at 248 tokens — was the largest positive effect in the whole grid
(+3.77 macro, CI [−0.2, +8.0], 8 fixed / 3 broken, 0.63 ms). It was not the
only time: M3-v0's `RND-maxred-r8` = 61.82 and `MG-maxred-r8` = 64.98 both sit
above B2. A prescribed budget of 256 is not obviously the optimal retained size,
and "safe to remove" may be a far more learnable predicate than "critical to
keep".

**The answer, in one line.** _[PENDING — filled from `m5_analysis.json`.]_

---

## 1. What is new, and what is deliberately unchanged

**Unchanged, deliberately.** The base selector is B2 verbatim (official EADP
importance + `block8` coverage greedy at T=256), run inside the same
`m2_gdep.GDEPEngine` in `prellm` mode, on the same frozen held-out 150, with the
same greedy decode and the same scorer as B0/B1/B2. The identity arm reproduces
the stored B2 record, which is what makes every other row comparable to a number
the project already trusts.

**Unchanged, and this is the binding one.** SafeTrim is pre-LLM and
forward-only. Every feature is a function of quantities the incumbent's own
pruner already produces — the EADP score and its components, the selector's own
similarity matrix, the post-merger vision feature the tower already emitted —
plus the retained set `S0` itself. No gradient, no decoder layer, no backward
pass, no teacher at inference, no generation feedback.

**New.** What happens to `S0` *after* B2 chose it: `k` of its own tokens are
deleted, and nothing is put back.

```
S0 = B2(vis)                        base selection,        |S0| = 256
r_i = P(deleting i hurts)           probe, i in S0
S* = S0 \ {the k lowest-risk i}                            |S*| = 256 - k
```

## 2. The teacher: deletion intervention, not gradient ranking

The label is the one the brief demands and it is **not** the gradient ranking
four earlier stages used:

```
d_i = L(y | S0 \ {i}) - L(y | S0)
```

`L` is the teacher-forced gold-answer NLL under PRE-LLM delivery — the same
harness, the same prompt construction and the same convention S3-A used, so a
`d_i` here and a `leave`-marginal there are the same quantity measured on a
different set. `d_i > 0`: deleting `i` hurts. `d_i <= 0`: free, or better than
free.

### 2.1 The two noise floors, and why only one of them is the right one

S3-A measured a per-instance "bf16 resolution floor" as the largest change in a
single gold's NLL when the *batch composition* changes. Inherited unchanged it
is **the wrong quantity for `d_i`**, and M5 measures both so the gap is visible
rather than assumed:

| quantity | what it moves | median over the 300 |
|---|---|---|
| `floor_max` | one gold's NLL, under re-batching | _[PENDING]_ |
| `floor_mean` | the **mean over golds**, under the same re-batching | _[PENDING]_ |
| `floor_rep` | two recomputations of the *same* mean | _[PENDING]_ |

`d_i` is a difference of two means computed with the **same** batching, so the
per-gold perturbation is common-mode and cancels; the resolvability that
actually applies is `floor_mean`, and it is 5–20× smaller. `floor_rep` is the
strict determinism check and is 0.

### 2.2 Sampling

`|S0| = 256`, so an exhaustive sweep is 256 forwards per instance. Each
instance is measured on a stratified sample of 54:

| stratum | n | what it covers |
|---|---:|---|
| `rand` | 24 | uniform draws from `S0` — the **unbiased** view of the population |
| `maxred` | 6 | highest `red_s0` — what the MAXRED rule would delete |
| `minred` | 6 | lowest `red_s0` — the contrast end |
| `highimp` | 6 | highest EADP importance |
| `lowimp` | 6 | lowest EADP importance — what the LOWIMP rule would delete |
| `recon` | 6 | lowest `nn4_recon` — what the RECON rule would delete |

The tail strata are enriched on purpose, so the strata are **not** independent:
`rand` is drawn first and each tail then walks its own score order skipping what
`rand` already took. The naive inclusion probability `k/|S0|` is therefore wrong
for a token sitting just past a tail's quota — a token ranked 7th by redundancy
is measured whenever `rand` happened to take one of the top 6, which is ~44 %,
not 9.4 %. `m5_probe.inclusion_weights` recovers the true probabilities by
re-running the sampler, and every population-level statistic is reported with
inverse-inclusion weights beside its unweighted twin.

### 2.3 Group deletion control (brief §2B)

For `k ∈ {4, 8}` and `rule ∈ {maxred, lowimp, random}` the whole group is
deleted at once and `D(G) = L(y | S0 \ G) - L(y | S0)` recorded, with a real
greedy generation per group arm. The question is one only: do the single-token
labels predict the group's harm? No Shapley values, no combinatorial search.

## 3. The label definition is conservative (brief §4)

The task is not to regress `d_i` precisely. It is to find a few tokens that are
*confidently* removable, so the problem is selective rather than a regression:

```
SAFE       d_i <= +eps       deleting it is not resolvably harmful
HARMFUL    d_i >= +delta     deleting it clearly costs
IGNORE     eps < d_i < delta
```

`eps` and `delta` are fixed from the noise floor, never from an accuracy number:

```
eps   = median over instances of floor_mean
delta = p90    over instances of floor_mean
```

The probe's **pre-registered primary target** is `harmful` (`d >= delta`), the
literal "risk = P(removal harms the answer)". A second target, `notsafe`
(`d > eps`) — the complement of the class the method acts on — is fitted
alongside so the gate can show whether the two disagree.

## 4. The features

27 columns, every one a function of what the incumbent's pruner already
produces. `m5_common.safe_features` is **the same function** called by the
offline bank builder and by the live pruner, so there is no train/serve skew by
construction; `m5_accuracy` re-derives the drop set from the bank and the saved
probe and requires it to equal the live one.

| group | columns |
|---|---|
| EADP score and components | `imp`, `imp_pre`, `fused`, `glob`, `loc`, `loc_max`, `loc_top5`, `loc_std` |
| redundancy inside `S0` | `red_s0`, `red_s0_top8`, `resid_u`, `nn4_recon`, `red_all`, `nn_drop`, `nn_drop_min` |
| local structure | `nb_mean`, `nb_max`, `nb_std`, `nb_mean_f`, `nb_red`, `nb_cos` |
| geometry and scale | `x`, `y`, `vis_norm`, `rank_imp`, `cos_s0c`, `cos_top64c` |

`red_s0` is the max cosine to **another** retained token (self excluded);
`nn4_recon` is `‖v_i − proj_span(4 nearest retained)‖ / ‖v_i‖`, a genuine
reconstruction error rather than a nearest-neighbour distance.

## 5. The probes

Three families, all deliberately small (brief §3 forbids an architecture
search):

| kind | input | params |
|---|---|---|
| `lr` | logistic regression on the 27 scalars | 28 |
| `mlp` | one hidden layer of 64 over the 27 scalars | ~6k |
| `vis` | the same head, plus one linear view of the 4096-d vision feature | ~262k |

**Pre-registered deployable arm: `mlp`, seed 0.** It is the small MLP the brief
asks for and it sits between the linear probe and the vision-view probe in
capacity. All three are reported in the gate, and the accuracy grid runs `lr`
and `vis` at k=8 beside it so the probe's capacity and its input are priced
rather than assumed.

Early stopping splits the fit split **by instance** (20 % of fit instances held
out, checked every 10 epochs, patience 10). The `vis` family reaches a train
loss of 0.016 unaided, so without a stopping rule its val number would measure
the stopping rule rather than the features; all three families get the same rule
so the comparison between them stays like-for-like, and val is never touched.

## 6. Protocol and controls

Every arm runs the full held-out 150 (TextVQA 50 / DocVQA 50 / OCRBench 50),
one greedy generation per instance, through the same engine and the same scorer
as the stored baselines.

| arm | k | tokens | what it is for |
|---|---:|---:|---|
| `B2` | 0 | 256 | **identity gate** — must equal the stored B2 |
| `SAFE-k*` | 4/8/12/16 | 252/248/244/240 | **the method** (primary: `SAFE-k8`) |
| `MAXRED-k*` | 4/8/12/16 | " | delete the k most redundant retained tokens |
| `LOWIMP-k*` | 4/8/12/16 | " | delete the k lowest-importance retained tokens |
| `RECON-k*` | 4/8/12/16 | " | delete the k best-reconstructed retained tokens |
| `RANDOM-k*` | 4/8/12/16 | " | delete k at random (3 seeds at k=8, 1 elsewhere) |
| `SAFELR-k8`, `SAFEVIS-k8` | 8 | 248 | the same method, other probe families |

`SAFE-k8` is the **pre-registered primary arm**: k=8 is where M4's `EVICT-r8`
lead was measured, and that lead came from a `maxred` eviction — so `MAXRED-k8`
is the exact training-free rule the method has to beat.

**Every arm returns `256 − k` tokens.** That is the point of the round — the
brief's §9 asks whether the prescribed budget is the optimal retained size — so
the token count is printed beside every score and no arm is ever compared to B2
without it.

**On resolution, stated before the numbers.** M3-v0 measured the paired macro SE
on this 150 at ~2–4 points and the MDE at 80 % power at **7.0**; M4 measured it
at **2.9 and 8.2**. A +2 point estimate is therefore *not* evidence on its own,
and every delta below is printed with its paired bootstrap CI and the McNemar p
on discordant pairs.

---

## 7. Results

_[PENDING]_

## 8. The trim curve

_[PENDING]_

## 9. What the supervision is worth (brief §12)

_[PENDING]_

## 10. The two formulations, side by side

_[PENDING]_

## 11. Gates and verdict

_[PENDING]_

## 12. Latency

_[PENDING]_

## 13. What this does and does not establish

_[PENDING]_

## 14. The six questions

_[PENDING]_
