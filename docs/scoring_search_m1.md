# M1 — Is the L4 LOCAL-MLP gain data, or optimizer steps?

**Scope.** S2-C6 Part A reported the `LOCAL-MLP` curve 0.7644 / 0.7853 / 0.7922 /
0.8047 at n = 60 / 120 / 180 / 240 fit images, returned the formal verdict
`DATA-PLATEAU` while saying the curve had not flattened, and named *where does the
data curve end?* as its one live open item. M1 asks a narrower and prior question:
in that protocol one "epoch" is `ceil(n/8)` optimizer updates, so `n` moves **both**
the number of independent teacher-labelled images and the number of optimizer
updates executed before the checkpoint is frozen. M1 separates the two, extends
the fit ladder to **n = 960**, and changes nothing else — same arm (the S2-C6 `L4`
reference), same `HEAD_RANK` target, same loss, same optimizer, same `H8`
selection rule. No query, no trajectory, no global or set context, no rescue path,
no wider model.

**Status.** Complete. Scripts
`m1_{common,plan,features,teacher,teacher_floor,train,consolidate,tables,pool_check,downstream,verify}.py`,
runner `run_m1.sh`. Pre-registration `scoring_search_m1_prereg.md` (written before
the first cache was built and before the first training run; amended once, in §5.2).
Main artifact `outputs/discovery/m1_audit.json`.

**Every number below is recomputed from the raw per-image artifacts by
`m1_verify.py`** — the headline, the paired segments (all three fields of all
five, over 10 000 bootstrap draws), the gate-G3 equality, the ladder membership
and the verdict rule. Worst disagreement: **0.00e+00**.

**No generation was run.** §9.

---

## HEADLINE

```
M1 -- the 60 -> 240 rise is data, not optimizer steps. The curve past 240 is
      positive but unresolved, and the L4 family clears the deploy line for the
      first time.

(a) THE CONFOUND IS REAL AND IT IS NOT WHAT PRODUCED THE GAIN.  At the
    H8-selected checkpoint S2-C6's protocol had executed 21.3 / 35.0 / 38.3 /
    40.0 mean updates at n = 60 / 120 / 180 / 240 -- 1.875x more at 240 than at
    60 -- and had shown the model 160 / 280 / 300 / 320 images.  Under a fixed
    budget of 900 updates for every n the same ladder gives 0.7639 / 0.7817 /
    0.7911 / 0.8047: a 60 -> 240 rise of +0.0408 [+0.0225, +0.0600], RESOLVED,
    against the published +0.0403.  The step-attributable residual is -0.0005,
    i.e. -1.3 % of the published rise, with the wrong sign to be an effect.

(b) THE CURVE KEEPS RISING PAST 240, AND THE LAST DOUBLING IS UNRESOLVED.
    240 -> 480 -> 960 at a fixed update budget: 0.8047 -> 0.8189 -> 0.8214.
    The first doubling is +0.0142 [+0.0014, +0.0272] -- a real but sub-margin
    effect (BELOW-MARGIN) whose CI excludes zero.  The second is +0.0025
    [-0.0103, +0.0158], which spans both zero and the margin.  240 -> 960 is
    +0.0167 [+0.0031, +0.0308].

(c) THE PRE-REGISTERED ORDERED RULE RETURNS INCONCLUSIVE, AND THAT IS THE HONEST
    WORD.  It is not EVIDENCE-OF-SATURATION: that label needs the 480 -> 960 CI
    upper bound below the 1 pt margin, and the upper bound is +0.0158.  It is not
    DATA-RESPONSIVE either, because the lower bound is -0.0103.  Nor is it
    ENDPOINT-UNRESOLVED: no primary run selects its checkpoint at the final
    validation point, so the fixed budget was not the binding constraint.

(d) THE GAP IS A GENERALISATION GAP AND IT CLOSES.  train R@8 minus held-out
    R@8 falls 0.2056 -> 0.1496 -> 0.1436 -> 0.0824 -> 0.0898 -> 0.0765.  Most of
    the closure is in 60 -> 240, the same range that carries the published gain.

(e) THE DEPLOY GATE S2-C6 LEFT CLOSED NOW OPENS ON A LITERAL READING -- and M1
    still runs no generation, by its own pre-registration.  The best arm reaches
    held-out R@8 0.8214 against the 0.82 threshold, with a CI lower bound of
    +0.0031 against the 240 reference.  One of the three seeds is exactly flat,
    so S2-C6's second condition ("consistent across all or most seeds") is met
    2 of 3.  Whether that opens the gate is the M2/M3 decision, and §9 says what
    is already on disk to make it.

(f) THE REFERENCE IS REPRODUCED BIT-EXACTLY.  The fixed-step protocol at n = 240
    is a strict prefix of S2-C6's n=240 trajectory -- one n=240 epoch is 30
    updates, which is exactly one M1 validation interval and exactly one
    reshuffled queue -- and it returns per-seed 0.7992 / 0.8133 / 0.8017 at
    selected epochs 0 / 1 / 0 with worst validation-curve |diff| = 0.0e+00 over
    all 68 compared epochs.  Everything above is measured against that reference.
```

---

## 1. The question

`S2-C6` §9.1:

> **Where does the data curve end?** The one live route. All features are cached,
> so extending to `n > 240` needs either more fit images (a new feature-cache pass
> over additional dataset instances) or a different data source.

M1 answers a question that has to be settled before that one can be read at all,
and states it as two:

1. **How much of the published 60 → 240 rise is an optimizer-step effect?** In
   S2-C6's protocol, `n` is not the only thing that changes.
2. **Under a fixed update budget, does 240 → 480 → 960 keep improving, and does
   the largest scale beat the 240 reference?**

---

## 2. Read-only audit (done before the pre-registration was frozen)

### 2.1 What exists

| artifact | shape | contents |
|----------|-------|----------|
| `s2c1_feats_L4.npy` | (465, 1024, 4096) fp16, 3.9 GB | L4 visual-token hidden states |
| `s2c1_query_L4.npy` | (465, 4096) fp16 | query position — **unused by M1** |
| `s2c1_features.json` | 465 plan entries | split assignment |
| `s2b_gradient_scores.npz` | 450 × (1024,) float32 | P1-G2 teacher maps |
| `s2c6_L4_s{0,1,2}__{H8,OV}.pt` | 6 files | the S2-C6 L4 reference checkpoints |
| `s2c6_train_{A,B,C}.json`, `s2c6_perimage_{A,B,C}.npz`, `s2c6_scores_{A,B,C}.npz` | — | S2-C6 records |

The instance set is the Stage-1 frozen bank of 150 per benchmark plus 15 S2-A
causal cases. Per benchmark the bank is `sample_indices(n_total, 150)`, and within
it `test = bank[::3]` (50), `val = pool[::5]` (20), `fit` = the remaining 80, so
**fit 240 / val 60 / test 150 / causal 15 = 465**. The teacher archive has 450
keys — fit + val + test; the 15 causal keys are deliberately absent and M1 never
touches them.

**The held-out 150 is never used to select anything in M1.** No held-out quantity
is computed inside a training loop; it is measured once per frozen checkpoint.

### 2.2 The extension is possible, with room to spare

Every row outside the frozen bank — and outside that benchmark's causal cases —
is free:

| benchmark | rows | frozen bank | causal | **free** | needed for n=960 (320/ds) |
|-----------|------|-------------|--------|----------|---------------------------|
| TextVQA_VAL | 5000 | 150 | 6 | **4844** | 240 |
| DocVQA_VAL | 5349 | 150 | 6 | **5193** | 240 |
| OCRBench | 1000 | 150 | 3 | **847** | 240 |

**n = 960 is reachable with 3.5× headroom on the tightest benchmark.** No fallback
scale was needed and **no val or test row is reused**.

### 2.3 The confound, measured from the published records

From `s2c6_train_A.json`, at the `H8`-selected checkpoint:

| n | steps / epoch | mean steps executed at the selected checkpoint | mean images seen |
|---|---------------|------------------------------------------------|------------------|
| 60 | 8 | 21.3 | 160 |
| 120 | 15 | 35.0 | 280 |
| 180 | 23 | 38.3 | 300 |
| 240 | 30 | 40.0 | 320 |

Going 60 → 240 multiplies the update count at the selected checkpoint by
**1.875×** and the images seen by **2.0×**, while multiplying the image *pool* by
4×. The published +0.0403 could therefore have been an optimizer-step effect.
That is the thing M1 removes.

### 2.4 What had to be generated

`n = 960` needs 240 new fit rows per benchmark = **720 new instances**.

| new artifact | shape | measured cost | wall clock |
|--------------|-------|---------------|------------|
| `m1_feats_L4_extra.npy` | (720, 1024, 4096) fp16, 6.0 GB | 206 ms / image | 335 s |
| `m1_gradient_scores_extra.npz` | 720 maps | 216 ms fwd + 296 ms bwd | 657 s |

Both come from the **existing** generators with the instance list replaced and
nothing else changed. The original caches are never written to: the combined row
space is `rows 0..464 → s2c1_feats_L4.npy` and `rows 465..1184 →
m1_feats_L4_extra.npy`, dispatched by row index, so the n = 240 rung of every
ladder reads the same bytes S2-C6 read.

---

## 3. The extension pool, and whether it is comparable

**The draw** (fixed before running, `EXTRASET_SEED = 20260924`): for benchmark `i`,
a single fixed permutation of the complement of `sample_indices(n_total, 150)`
and of that benchmark's causal cases, taking the first 240. The ladders are
then

| n | per benchmark | composition |
|---|---------------|-------------|
| 240 | 80 | the S2-C6 fit set, byte-identical |
| 480 | 160 | `F_i` + `extra_i[:80]` |
| 960 | 320 | `F_i` + `extra_i[:240]` |

so `S(240) ⊂ S(480) ⊂ S(960)`, benchmark-stratified at every rung, with val
frozen at 60 and held-out frozen at 150. Standardisation mean/std is recomputed on
the rows each arm may see.

**Comparability** (`m1_pool_check.py`, `m1_pool_check.json`). Extending the fit set
with 720 images that are systematically harder or differently supervised would
produce a flat curve for a reason unrelated to data quantity, so the extension is
compared against the 240 rows it extends on instance and teacher-map properties:

| quantity | original (240) | extension (720) | rel. shift | KS | KS crit. 5 % |
|----------|----------------|-----------------|-----------|-----|--------------|
| `g2_sum` | 1032 | 995.5 | −3.6 % | 0.043 | 0.101 |
| `g2_max` | 39.22 | 38.79 | −1.1 % | 0.050 | 0.101 |
| `g2_top32_mass` | 0.2842 | 0.2829 | −0.4 % | 0.042 | 0.101 |
| `h4_norm` | 0.4280 | 0.4273 | −0.2 % | 0.069 | 0.101 |

**No quantity differs significantly at the 5 % level, pooled or within any
benchmark.** The largest divergence is OCRBench `g2_max` (−23 %, KS 0.154 against
a critical value of 0.176) — reported because it is the largest, not because it is
significant. The representation itself (`h4_norm`) is flat to ±0.6 % everywhere.

---

## 4. The two protocols

### 4.1 Primary — fixed steps

Identical for every n and every seed: `L4` arm (524 673 params), batch 8, AdamW
1e-3 / 1e-4, constant LR (there is no schedule), **900 optimizer updates**,
validation every 30 updates → 30 validation points, selection = argmax of
validation `head_recall8` (`H8`), **no early halt**. Data order is one reshuffled
queue refilled whenever it empties; because 60, 120, 180, 240, 480 and 960 are all
divisible by 8, the refill lands exactly on the epoch boundaries S2-C6 drew its
own permutation on. Grid: `n ∈ {60,120,180,240,480,960}` × seeds `{0,1,2}` = 18 runs.

* **240 → 480 → 960 is the pre-registered primary ladder.**
* **60 → 120 → 180 → 240 is the step-controlled counterpart of S2-C6 Part A**,
  and it is in the grid because Q1 cannot be answered without it.

### 4.2 Secondary — fixed epoch (S2-C6's own protocol, reproduced)

`MAX_EPOCHS = 80`, halt at `PATIENCE = 20` epochs after the best validation
overlap, validation once per epoch, `H8` selection. Run at `n ∈ {240, 480, 960}`
× 3 seeds. It re-entangles steps with n by construction, which is why it is the
control and not the primary.

### 4.3 One declared asymmetry

At `n = 60` the reshuffling cycle draws `ceil(60/8) = 8` batches, the last of
which holds 4 images, so per-update data volume is 1.5 % lower than at other n.
This is inherited from S2-C6's epoch loop, is identical in both protocols, and
touches only the C6-alignment points. It is reported, not corrected, because
correcting it would break the reproduction gate.

### 4.4 A limitation of the fixed-step design, stated rather than buried

At a fixed update count the number of *passes* over the pool varies inversely with
n: 900 updates × 8 images = 7200 samples is 120 passes at n = 60 and 7.5 passes at
n = 960. The fixed-step ladder therefore isolates "same number of updates, larger
and more diverse pool", not "same number of passes". That is the correct
isolation for the question asked — it is exactly what removes the confound — but
the small-n rungs are heavily repeated, and that is a property of the design, not
a property of the data.

---

## 5. The three gates

### 5.1 G1 — feature generator

Eight already-cached instances were re-derived through the extension generator
before anything was written. **Worst max |Δ| = 0.000e+00, 100 % of elements
exactly equal, on all eight** (`m1_feature_repro.json`). The L4 forward is
bit-reproducible here, so the extension rows are produced by a generator that
demonstrably reproduces the published cache rather than one asserted to.

### 5.2 G2 — teacher generator, and amendment 1

The pre-registration fixed the criterion *before* the quantity's reproducibility
was measured:

> **G2 teacher** `... reproduces s2b_gradient_scores.npz to float32 rounding
> (max |Δ| <= 1e-4 relative)`

That is unachievable. The P1-G2 score is a **backward** pass and does not
reproduce *itself* to 1e-4.

> **Amendment 1 (before the extension maps were generated).** Replace the fixed
> 1e-4 with two criteria measured in the same run: **(a)** worst
> `|published − recomputed| / max|published|` ≤ 2 × worst
> `|recomputed − recomputed| / max|published|`, and **(b)** the Top-32 change
> against the published map ≤ the Top-32 change the identical code path makes
> against itself.
>
> **Reason.** `m1_teacher_floor.py` measured the floor first: over 8 instances ×
> 3 repeats the code path disagrees with itself by up to **9.6e-3** relative. A
> 1e-4 threshold would have failed the published pipeline against itself, so it
> tested nothing. Criterion (b) is the one that carries the weight — `HEAD_RANK`
> supervises exactly the teacher's Top-32 and their rank order, so the question
> is whether *the target* moves more than the generator's own noise moves it.
>
> **Effect on the verdict.** None. G2 is a generator self-check; the M1 decision
> rule in §7 is untouched.

**Result** (`m1_teacher_repro.json`): passed. Worst self 1.76e-2, worst published
3.37e-2. `top1_match = True` and **Top-32 symmetric difference = 0 against the
published map on all 8 instances**, with the self comparison also 0 — the target
does not move.

**A caveat about criterion (a), reported because it weakens it.** The per-instance
published/self ratios are 1.00, 1.33, 1.60, 2.00, 2.00, 2.00, 2.00, 5.50 — a
median of exactly 2.00. That is not a coincidence: `self` is a max over one
comparison while `published` is a max over two recomputations, so a heavy-tailed
maximum is inflated ~2× by construction, and the factor of 2 in criterion (a)
cancels that bias almost exactly. **Criterion (a) should therefore be read as
close to vacuous, and criterion (b) as the load-bearing one.** (b) passed exactly,
on every instance. A properly matched estimator would compare like with like;
that is recorded as an open item rather than patched after the result.

**Consequence for the data.** The extension rows' teacher maps come from this
generator while the original 240 rows' maps come from the published file. G2 plus
§3's pool comparison are what license treating the two halves as one dataset: the
difference is within the generator's own noise, the Top-32 is unchanged, and the
per-benchmark map statistics are statistically indistinguishable.

### 5.3 G3 — the training reference, and the decisive gate

The pre-registration required the fixed-step protocol at n = 240 to reproduce
S2-C6's `A240` arm, and to stop the chain if it did not. It is a strict prefix by
construction: one n=240 epoch is 30 updates = one validation interval = one
reshuffled queue.

| seed | epochs compared | C6 epochs run | C6 H8 epoch | M1 H8 point | epoch match | M1 held-out R@8 | published | match |
|------|-----------------|---------------|-------------|-------------|-------------|-----------------|-----------|-------|
| 0 | 24 | 24 | 0 | 0 | yes | 0.7992 | 0.7992 | yes |
| 1 | 22 | 22 | 1 | 1 | yes | 0.8133 | 0.8133 | yes |
| 2 | 22 | 22 | 0 | 0 | yes | 0.8017 | 0.8017 | yes |

**Worst validation-curve |diff| over all 68 compared epochs: 0.0e+00.** Passed
(`m1_gate_g3.json`). The fixed-step protocol is a controlled re-parameterisation
of the published run, not a new configuration — which is what makes the
step/data decomposition below interpretable rather than merely suggestive.

---

## 6. Results

### 6.1 Primary — fixed steps, 900 updates for every n

| n | R@8 | R@16 | R@32 | ov256 | per-seed R@8 | selected point | OV-selected R@8 | train R@8 | val R@8 |
|---|-----|------|------|-------|--------------|----------------|-----------------|-----------|---------|
| 60 | **0.7639** | 0.7371 | 0.6899 | 0.5162 | 0.7658 / 0.7592 / 0.7667 | 0 / 0 / 0 | 0.7639 | 0.9694 | 0.7319 |
| 120 | **0.7817** | 0.7561 | 0.7027 | 0.5129 | 0.7792 / 0.7800 / 0.7858 | 0 / 0 / 0 | 0.7731 | 0.9313 | 0.7472 |
| 180 | **0.7911** | 0.7624 | 0.7076 | 0.5158 | 0.7917 / 0.7858 / 0.7958 | 1 / 0 / 1 | 0.7892 | 0.9347 | 0.7472 |
| 240 | **0.8047** | 0.7751 | 0.7135 | 0.5078 | 0.7992 / 0.8133 / 0.8017 | 0 / 1 / 0 | 0.7972 | 0.8872 | 0.7806 |
| 480 | **0.8189** | 0.7889 | 0.7299 | 0.5159 | 0.8275 / 0.8042 / 0.8250 | 3 / 1 / 3 | 0.8181 | 0.9087 | 0.7937 |
| 960 | **0.8214** | 0.7925 | 0.7392 | 0.5262 | 0.8175 / 0.8133 / 0.8333 | 5 / 4 / 7 | 0.8192 | 0.8979 | 0.8139 |

| segment | delta | 95 % CI | per-seed | label |
|---------|-------|---------|----------|-------|
| 60 → 120 | +0.0178 | [+0.0031, +0.0328] | +0.0133 / +0.0208 / +0.0192 | **RESOLVED** |
| 120 → 180 | +0.0094 | [−0.0042, +0.0231] | +0.0125 / +0.0058 / +0.0100 | AMBIGUOUS |
| 180 → 240 | +0.0136 | [−0.0008, +0.0289] | +0.0075 / +0.0275 / +0.0058 | AMBIGUOUS |
| **240 → 480** | **+0.0142** | **[+0.0014, +0.0272]** | +0.0283 / **−0.0092** / +0.0233 | **BELOW-MARGIN** |
| **480 → 960** | **+0.0025** | **[−0.0103, +0.0158]** | **−0.0100** / +0.0092 / +0.0083 | **AMBIGUOUS** |
| 240 → 960 | +0.0167 | [+0.0031, +0.0308] | +0.0183 / **0.0000** / +0.0317 | BELOW-MARGIN |

**Selection-invariance.** Under `OV` selection the same ladder gives 0.7639 /
0.7731 / 0.7892 / 0.7972 / 0.8181 / 0.8192. The `H8` and `OV` readings are close
at every rung and the ordering is unchanged, so the verdict does not depend on the
checkpoint rule.

### 6.2 Secondary — fixed epoch, S2-C6's own budget

| n | R@8 | R@16 | ov256 | per-seed R@8 | selected epoch | epochs run | steps run |
|---|-----|------|-------|--------------|----------------|------------|-----------|
| 240 | **0.8047** | 0.7751 | 0.5078 | 0.7992 / 0.8133 / 0.8017 | 0 / 1 / 0 | 24 / 22 / 22 | 720 / 660 / 660 |
| 480 | **0.8189** | 0.7889 | 0.5159 | 0.8275 / 0.8042 / 0.8250 | 1 / 0 / 1 | 23 / 23 / 22 | 1380 / 1380 / 1320 |
| 960 | **0.8242** | 0.7943 | 0.5241 | 0.8133 / 0.8258 / 0.8333 | 2 / 0 / 1 | 23 / 22 / 22 | 2760 / 2640 / 2640 |

| segment | delta | 95 % CI | per-seed | label |
|---------|-------|---------|----------|-------|
| 240 → 480 | +0.0142 | [+0.0014, +0.0272] | +0.0283 / −0.0092 / +0.0233 | BELOW-MARGIN |
| 480 → 960 | +0.0053 | [−0.0067, +0.0178] | −0.0142 / +0.0217 / +0.0083 | AMBIGUOUS |

The two protocols agree, and their agreement is structural rather than lucky:

* at **n = 240** the validation grids coincide exactly (one epoch = 30 updates),
  which is gate G3;
* at **n = 480** one epoch is 60 updates, so the fixed-epoch grid hits every other
  M1 validation point, and both protocols select the *same* checkpoints
  (steps 60 / 120) — hence identical numbers;
* at **n = 960** one epoch is 120 updates and the grids separate. The fixed-epoch
  protocol selects later checkpoints (steps 360 / 120 / 240) than the fixed-step
  one (180 / 150 / 240) and lands +0.0028 higher, well inside noise. **That
  coarser validation cadence at large n is itself part of the confound the
  fixed-step protocol removes** — the original protocol becomes progressively
  blinder to the early `H8` optimum as `n` grows.

### 6.3 Per-benchmark (50 held-out images each, descriptive)

| segment | TextVQA_VAL | DocVQA_VAL | OCRBench |
|---------|-------------|------------|----------|
| 60 → 120 | +0.0150 [−0.0058, +0.0375] | −0.0033 [−0.0300, +0.0217] | **+0.0417 [+0.0142, +0.0708] RESOLVED** |
| 120 → 180 | −0.0050 [−0.0267, +0.0158] | +0.0275 [−0.0000, +0.0558] | +0.0058 [−0.0142, +0.0258] |
| 180 → 240 | **+0.0333 [+0.0075, +0.0608] RESOLVED** | −0.0025 [−0.0283, +0.0233] | +0.0100 [−0.0133, +0.0350] |
| 240 → 480 | +0.0100 [−0.0033, +0.0233] | +0.0283 [+0.0000, +0.0592] BELOW-MARGIN | +0.0042 [−0.0150, +0.0242] |
| 480 → 960 | +0.0125 [−0.0033, +0.0275] | −0.0042 [−0.0325, +0.0258] | −0.0008 [−0.0217, +0.0200] |

Only three of fifteen benchmark-segments resolve, and they are in three different
segments and three different benchmarks — no benchmark carries the ladder, and the
patterns do not line up across segments. At 50 images per benchmark these readings
are descriptive, not verdicts.

### 6.4 The train/validation gap

| n | train R@8 | val R@8 | held-out R@8 | train − val | train − held-out |
|---|-----------|---------|--------------|-------------|------------------|
| 60 | 0.9694 | 0.7319 | 0.7639 | +0.2375 | +0.2056 |
| 120 | 0.9313 | 0.7472 | 0.7817 | +0.1840 | +0.1496 |
| 180 | 0.9347 | 0.7472 | 0.7911 | +0.1875 | +0.1436 |
| 240 | 0.8872 | 0.7806 | 0.8047 | +0.1066 | +0.0824 |
| 480 | 0.9087 | 0.7937 | 0.8189 | +0.1149 | +0.0898 |
| 960 | 0.8979 | 0.8139 | 0.8214 | +0.0840 | +0.0765 |

The train − held-out gap falls by **0.129**, from 0.2056 to 0.0765, and about
two-thirds of that closure happens between n = 180 and n = 240. Train R@8 itself
is roughly flat (0.887–0.969) while held-out rises — the extra data is buying
generalisation, not fit capacity. That is the expected signature of a data effect
rather than an optimization effect, and it is consistent with §8 Q1.

---

## 7. The verdict

Pre-registered ordered rule, first match wins:

```
1. ENDPOINT-UNRESOLVED   any primary run selects H8 at the final validation point
2. DATA-RESPONSIVE       CI lower bound of Δ(480→960) > 0
3. EVIDENCE-OF-SATURATION  CI upper bound of Δ(480→960) < MARGIN
4. INCONCLUSIVE          otherwise
```

```
1. not triggered    no primary run selects at point 29 of 30
                    (selected points 0/0/0, 0/0/0, 1/0/1, 0/1/0, 3/1/3, 5/4/7)
2. not triggered    lower bound is -0.0103
3. not triggered    upper bound is +0.0158, above MARGIN = 0.01
4. VERDICT: INCONCLUSIVE
```

> **INCONCLUSIVE.** The 480 → 960 interval spans both zero and the margin, so the
> primary contrast is unresolved. That is not evidence of a plateau and is not
> reported as one.

**What the numbers do say, stated separately from the label.** The curve has not
flattened: 240 → 480 is +0.0142 with a CI excluding zero, and the mean is positive
at every rung. But the *last* doubling is +0.0025 and one seed is negative, while
240 → 960 is +0.0167 with one seed exactly flat at 0.0000. The honest summary is
that the family is still data-responsive in the 240 → 480 range, that nothing
resolves at 480 → 960, and that three seeds and 150 held-out images cannot
separate "a small real gain" from "no gain" at that segment.

**The binary reading is refused.** `EVIDENCE-OF-SATURATION` required a bounded
interval below the margin, and it was not obtained; `INCONCLUSIVE` is the label
pre-registered for exactly this case and it is reported as the verdict rather than
being folded into either neighbour.

---

## 8. The six questions

**Q1 — how much of S2-C6's 60 → 240 rise could be an optimizer-step effect?**

**Essentially none.** Under a fixed budget of 900 updates for every n the curve is
0.7639 / 0.7817 / 0.7911 / 0.8047, a 60 → 240 rise of **+0.0408
[+0.0225, +0.0600], RESOLVED, all three seed deltas positive**. The published
fixed-epoch rise is **+0.0403**. The step-attributable residual is **−0.0005**,
i.e. **−1.3 %** of the published rise — and its sign is wrong for an effect, since
the fixed-step protocol *removes* the step advantage large n had and the rise came
out marginally *larger*.

The two curves are near-identical rung by rung (0.7644/0.7639, 0.7853/0.7817,
0.7922/0.7911, 0.8047/0.8047). The 1.875× update multiplier that 60 → 240
incidentally carried was not doing the work; the extra independent images were.

**Q2 — under fixed steps, does 240 → 480 → 960 keep improving?**

**On the mean, yes, monotonically: 0.8047 → 0.8189 → 0.8214.** But only the first
doubling has a CI excluding zero (+0.0142 [+0.0014, +0.0272], BELOW-MARGIN, one
seed negative), and the second does not resolve at all (+0.0025
[−0.0103, +0.0158]). The secondary protocol says the same thing
(0.8047 → 0.8189 → 0.8242; +0.0142 BELOW-MARGIN, then +0.0053 AMBIGUOUS).

**Q3 — does the largest scale significantly exceed the 240 reference?**

**It clears zero but not the margin.** Δ(240 → 960) = **+0.0167
[+0.0031, +0.0308]**, BELOW-MARGIN: the CI lower bound is positive, so this is a
directional result and not a null, but the per-seed deltas are +0.0183 / 0.0000 /
+0.0317 and one seed is exactly flat, so the pre-registered RESOLVED conditions
(mean ≥ 0.01 **and** CI lower bound > 0 **and** all three seeds positive) are not
met. It carries no verdict.

**Q4 — is any gain consistent across seeds and benchmarks?**

**No.**
* *By seed:* 240 → 480 is +0.0283 / **−0.0092** / +0.0233; 480 → 960 is
  **−0.0100** / +0.0092 / +0.0083; 240 → 960 is +0.0183 / **0.0000** / +0.0317.
  Every segment has at least one non-positive seed, and they are different seeds.
* *By benchmark:* no segment resolves in more than one benchmark, and the three
  resolved benchmark-segments (60→120 OCRBench, 180→240 TextVQA, 240→480 DocVQA)
  fall in three different places. At 50 images per benchmark these are descriptive.

The mean-level gains are real but they are carried unevenly, which is part of why
nothing clears the pre-registered RESOLVED bar.

**Q5 — does the train/validation gap shrink?**

**Yes, substantially, and mostly in the published range.** train − held-out R@8:
0.2056 → 0.1496 → 0.1436 → 0.0824 → 0.0898 → 0.0765, a fall of **0.129**, with
about two-thirds of it between n = 180 and n = 240. train − val falls 0.2375 →
0.0840. Train R@8 stays roughly flat across the whole ladder while held-out rises,
so the extra data is buying generalisation rather than capacity.

**Q6 — is a downstream generation run worth it?**

**The gate condition is met on a literal reading, and M1 still runs nothing.**
The best primary arm is n = 960 at held-out R@8 **0.8214**, above S2-C6's 0.82
threshold for the first time in the project, with a CI lower bound of +0.0031
against the 240 reference. S2-C6's gate asks for three things: CI lower bound > 0
against the reference (met), held-out R@8 ≥ 0.82 (met), and consistency across
"all or most seeds" (2 of 3, with one seed exactly flat).

So the honest answer is: **the gate opens on the letter of its first two
conditions and is arguable on the third.** That is an M2/M3 decision, and §9 lists
what is already on disk so it can be taken without regenerating anything. M1's
pre-registration §6 commits M1 to running no generation, and M1 kept that
commitment.

---

## 9. Downstream: prepared, verified, not run

`m1_downstream.py --check` resolves every artifact the comparison needs and
reports `ready: True` (`m1_downstream_ready.txt`). The comparison it prepares holds
the selector fixed and swaps only the importance vector, at budget 256, for
`topk`, `block8` and `facility`, against the EADP importance baseline:

| what | where | state |
|------|-------|-------|
| best checkpoint | `m1_fixed-step_n960_L4_s2__H8.pt`, 524 673 params, loads clean | on disk |
| baseline importance + facility selection, all 150 held-out keys | `s2b_official_selection.npz` (`imp__`, `sel__`, budget 256) | on disk |
| teacher maps for the held-out 150 | `s2b_gradient_scores.npz` | on disk |
| L4 features for the held-out 150 | combined row space | on disk |
| selector implementations | `instrumented.SELECTORS`, `_sim_visual_impl` | importable |

**The one missing piece is named rather than discovered later:** the coverage
selectors (`block8`, `facility`) also need the token similarity matrix, which comes
from the post-merger vision features (`_sim_visual_impl`, sim mode `rebound`). It
is **not cached anywhere** in the S2 outputs and needs one vision-tower pass per
instance — minutes, not a generation run. The offline half of the comparison
therefore costs no generation; only the accuracy half does.

```
python scripts/discovery/m1_downstream.py --offline    # selector-level, no generation
python scripts/discovery/m1_downstream.py --generate   # accuracy, needs the EADP-256 pipeline
```

Both are wired and guarded; neither was executed.

---

## 10. What this changes, and what it does not

**Established by M1:**

* The published 60 → 240 rise in the L4 `LOCAL-MLP` family is a **data** effect,
  not an optimizer-step artefact: under a fixed update budget the rise is
  +0.0408 against the published +0.0403, and the step-attributable residual is
  −1.3 % of it with the wrong sign.
* The data route is **still open at 240 → 480** (+0.0142, CI excluding zero) and
  **unresolved at 480 → 960**.
* The L4 scorer reaches held-out R@8 **0.8214** at n = 960, above the 0.82 deploy
  line for the first time in this project, with a positive CI lower bound against
  the 240 reference.
* The train/held-out gap falls by 0.129 across the ladder while train R@8 stays
  flat — a generalisation effect, not a capacity effect.
* The S2-C6 reference is reproducible **bit-exactly** under a different but
  equivalent protocol (gate G3), which is what makes the decomposition above a
  measurement rather than an argument.

**Not established, and not claimed here:**

* **That the family has saturated.** The pre-registered verdict is `INCONCLUSIVE`,
  not `EVIDENCE-OF-SATURATION`; the 480 → 960 upper bound is +0.0158, above the
  1 pt margin. Nothing here licenses the word plateau.
* **That ~0.82 is a ceiling, or that the deploy gate is genuinely open.** One of
  three seeds is exactly flat on 240 → 960 and every segment has a non-positive
  seed. The `BELOW-MARGIN` class is again exercised and again carries no verdict.
* **That the extension draw is a random sample of a population.** It is one fixed
  draw per benchmark from each benchmark's complement set, with fixed seeds, and
  it is reported as such. §3 shows it is distributionally comparable to the pool
  it extends; that is weaker than random sampling and is the correct claim.
* **Anything about a representation route.** No query, trajectory, global or set
  context, rescue path or wider model was trained. S2-C6's B/C/D verdicts are
  untouched.
* **That a fixed-step null would mean saturation.** The rule was written so that
  `EQUIVALENT-NULL` and `AMBIGUOUS` are different labels, and the outcome here was
  the latter.

---

## 11. Open items

1. **Is the deploy gate open?** §8 Q6 leaves it arguable: two conditions met, the
   seed-consistency condition met 2 of 3 with one seed exactly flat. Deciding it
   is the M2/M3 entry point, and §9 lists everything already on disk.
2. **480 → 960 is unresolved at 3 seeds and 150 held-out images.** The paired CI
   on that segment is ±0.013 wide, which is larger than the effect it is trying to
   detect. Resolving it needs either more seeds or a larger held-out set, not more
   data — the fit pool already has headroom to 4844 rows on the tightest benchmark.
3. **Validation is frozen at 60 images.** Kept for comparability with S2-C6, but
   `H8` selection at n = 960 is choosing among checkpoints on a 60-image split
   while the fit set is 16× larger. The `H8` and `OV` readings agree closely
   (§6.1), so this is not load-bearing for the verdict — but no stage has yet
   tested a selection rule that scales validation with the fit set.
4. **Gate G2's criterion (a) is close to vacuous** (§5.2): the factor of 2 cancels
   a max-of-2-versus-max-of-1 estimator bias. Criterion (b) carried the weight and
   passed exactly. A properly matched estimator should be used if the gate is
   reused.
5. **The `H8` rule freezes an early checkpoint at every n** (selected points 0–7 of
   30). Part of what the larger fit sets buy may be *less fit-split overfitting*
   rather than a better-chosen epoch, exactly as S2-C5A flagged. §6.4's gap
   closure is consistent with that reading and does not separate it from a
   straightforward data effect.
6. **The teacher map's Top-32 is stable but not frozen**: on 2 of 8 gate-G2
   instances the recomputation and the published map each differ from the code
   path's own re-run by the same 2 tokens out of 32. This affects the extension
   rows and the published rows identically and the Top-32 never moved *beyond* the
   self floor, but it is a floor on how precisely the target is defined.

**M1 stops here.** No final method is designed from it and no generation was run.
