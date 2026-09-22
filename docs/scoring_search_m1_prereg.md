# M1 — Is the L4 LOCAL-MLP gain data, or optimizer steps? (pre-registration)

**Written before any new feature cache, any new teacher map and any M1 training
run.** Frozen and dated. If a rule below turns out to be wrong it is amended in
`scoring_search_m1.md` with the amendment dated and the pre-registered version
quoted, not silently overwritten here.

---

## 0. What M1 inherits, verbatim

Everything not named as a degree of freedom below is the S2-C6 / S2-C5A
configuration, unchanged. M1 re-tunes nothing, widens nothing, adds no query, no
trajectory, no global context, no set context, no rescue path.

| item | value |
|------|-------|
| arm | **`L4`** — the S2-C6 `LOCAL-MLP` reference, `s = w2 · GELU(P h₄ + b)`, 524 673 params |
| features | `s2c1_feats_L4.npy`, (465, 1024, 4096) fp16, extended (below) |
| teacher | S2-B P1-G2 gradient maps, `s2b_gradient_scores.npz` |
| target | `HEAD_RANK` — positives = teacher Top-32, pair weights `32/(r+1)` mean-normalised, `pos_weight = 31` |
| loss | balanced BCE + `0.5 ×` weighted all-pairs margin ranking, margin 1.0 |
| optimizer | AdamW, lr 1e-3, wd 1e-4, batch 8 |
| preprocessing | per-dimension mean/std computed on **the fit rows that arm may see**, never inherited |
| seeds | 0, 1, 2 |
| kernels | deterministic forced (`torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`) |
| primary metric | held-out (150) teacher **Top-8 recall@256** |
| checkpoint selection | argmax over validation (60) teacher Top-8 recall@256 — the S2-C5A **`H8`** rule |
| `OV` | validation Top-256 overlap, recorded as a robustness reading, selects nothing |

**The held-out 150 is never used for any selection.** No held-out quantity is
computed inside a training loop anywhere in M1. It is measured once per frozen
checkpoint, exactly as in S2-C6.

---

## 1. The question

S2-C6 Part A reported the `LOCAL-MLP` curve 0.7644 / 0.7853 / 0.7922 / 0.8047 at
n = 60 / 120 / 180 / 240 fit images, and formally returned `DATA-PLATEAU` while
saying the curve had not flattened. It then named the open item itself (§9.1):
*where does the data curve end?*

**M1 asks a narrower and prior question.** In S2-C6's protocol one "epoch" is
`ceil(n/8)` optimizer updates, so `n` changes **both** the number of independent
teacher-labelled images *and* the number of optimizer updates executed before the
checkpoint is frozen. At the `H8`-selected checkpoint the two are entangled:

| n | steps / epoch | mean steps executed at the selected checkpoint | mean images seen |
|---|---------------|------------------------------------------------|------------------|
| 60 | 8 | 21.3 | 160 |
| 120 | 15 | 35.0 | 280 |
| 180 | 23 | 38.3 | 300 |
| 240 | 30 | 40.0 | 320 |

(from `s2c6_train_A.json`, seeds 0/1/2). Going 60 → 240 multiplies the update
count at the selected checkpoint by **1.875×** and the images seen by **2.0×**,
while the image *pool* is multiplied by 4×. So part — possibly all — of the
published +0.0403 could be an optimizer-step effect rather than a
data-diversity effect.

**M1 separates them, and nothing else.** It does not ask whether ~0.80 is a
ceiling, does not test a representation hypothesis, and does not design a method.

---

## 2. Read-only audit (completed before this document was frozen)

### 2.1 What exists

| artifact | shape / size | contents |
|----------|--------------|----------|
| `outputs/discovery/s2c1_feats_L4.npy` | (465, 1024, 4096) fp16, 3.9 GB | L4 visual-token hidden states |
| `outputs/discovery/s2c1_query_L4.npy` | (465, 4096) fp16 | query position — **not used by M1** |
| `outputs/discovery/s2c1_features.json` | 465 plan entries | split assignment |
| `outputs/discovery/s2b_gradient_scores.npz` | 450 keys, (1024,) float32 | P1-G2 teacher maps |
| `outputs/discovery/s2c6_L4_s{0,1,2}__{H8,OV}.pt` | 6 files | S2-C6 L4 reference checkpoints |
| `outputs/discovery/s2c6_train_{A,B,C}.json`, `s2c6_perimage_{A,B,C}.npz`, `s2c6_scores_{A,B,C}.npz` | — | S2-C6 records |

**Split (frozen, unchanged by M1).** The instance set is the Stage-1 frozen bank
of 150 per benchmark plus 15 S2-A causal cases. Per benchmark the bank is
`sample_indices(n_total, 150)`, and within it `test = bank[::3]` (50),
`val = pool[::5]` (20), `fit = the remaining 80`. So fit 240 / val 60 / test 150
/ causal 15 = 465.

**Teacher coverage.** `s2b_gradient_scores.npz` has 450 keys = fit + val + test.
The 15 causal keys are deliberately absent (`teacher_orders` returns an unusable
order for them and nothing in M1 touches them).

### 2.2 Can the fit set be extended to 480 and 960 without touching val/test?

**Yes.** Each benchmark's frozen bank is 150 of `n_total` rows; every other row
is free except the handful used as causal cases:

| benchmark | rows | frozen bank | causal | **free** | needed for n=960 (320/ds) |
|-----------|------|-------------|--------|----------|---------------------------|
| TextVQA_VAL | 5000 | 150 | 6 | **4844** | 240 |
| DocVQA_VAL | 5349 | 150 | 6 | **5193** | 240 |
| OCRBench | 1000 | 150 | 3 | **847** | 240 |

`n = 960` is reachable with 3.5× headroom on the tightest benchmark. **No
fallback scale is needed and no val/test row is reused.**

### 2.3 What must be newly generated

`n = 960` needs 320 fit rows per benchmark; 80 already exist, so **240 new rows
per benchmark = 720 new images**. `n = 480` needs the first 80 of those 240.

| new artifact | size | cost at measured rates | est. wall clock |
|--------------|------|------------------------|-----------------|
| `m1_feats_L4_extra.npy` | (720, 1024, 4096) fp16, 6.0 GB | 206 ms / image (S2-C1 measured) | ~2.5 min |
| `m1_gradient_scores_extra.npz` | 720 keys | 512 ms / image = 216 fwd + 296 bwd (S2-B measured) | ~6 min |

Both are produced by re-running the **existing** generators
(`s2c1_features.py`'s forward, `s2b_gradient_scores.py`'s P1-G2 objective) with
the instance list replaced by the extension set. Neither generator is modified in
any other way. **Both new caches are verified against the existing ones** by
recomputing a sample of already-cached instances and requiring the results to
match (see §3.3).

---

## 3. The extension set and the fit ladders

### 3.1 Extension draw (fixed before running)

```
EXTRASET_SEED = 20260924                     # one draw, never re-drawn
for i, ds in enumerate([TextVQA_VAL, DocVQA_VAL, OCRBench]):
    used_i  = set(sample_indices(n_total_i, 150)) | causal_indices_i
    free_i  = sorted(set(range(n_total_i)) - used_i)
    rng_i   = np.random.default_rng(EXTRASET_SEED + 101 * (i + 1))
    extra_i = [free_i[j] for j in rng_i.permutation(len(free_i))[:240]]
```

The draw is **uniform over the complement of the frozen bank**. It is one random
permutation per benchmark with a fixed, benchmark-specific stream, so `extra_i`
is reproducible and `n` remains the only thing that changes across the ladders.

### 3.2 Fit ladders (nested by construction, benchmark-stratified)

Let `F_i` be benchmark `i`'s 80 frozen fit rows.

| n | per benchmark | composition |
|---|---------------|-------------|
| 240 | 80 | `F_i` — **the S2-C6 fit set, byte-identical** |
| 480 | 160 | `F_i` + `extra_i[:80]` |
| 960 | 320 | `F_i` + `extra_i[:240]` |

`S(240) ⊂ S(480) ⊂ S(960)`. Validation stays the frozen 60 and held-out stays the
frozen 150 at every n. Standardisation mean/std is recomputed on the n rows the
arm may see, per the inherited rule.

### 3.3 Reproduction gates (must pass before any M1 arm is run)

Three gates, all checked and reported, and **any failure stops the run**:

```
G1  data   m1_feats_L4_extra row ordering and the combined row space reproduce
            the original plan: rows 0..464 of the combined cache are
            bit-identical to s2c1_feats_L4.npy rows 0..464.
G2  teacher the P1-G2 objective recomputed on >= 8 already-cached instances
            reproduces s2b_gradient_scores.npz to float32 rounding
            (max |Δ| <= 1e-4 relative), which is what licenses applying the
            same generator to the 720 new images.
G3  train   the primary fixed-step protocol at n = 240, which is a strict prefix
            of S2-C6's n=240 trajectory (one n=240 epoch = 30 updates = exactly
            one M1 validation interval), reproduces s2c6_train_A.json's A240
            validation history for the overlapping epochs to 1e-9, and selects
            the same H8 epoch and the same per-seed held-out R@8 as the
            published run (0.7992 / 0.8133 / 0.8017 at epochs 0 / 1 / 0).
```

If G3 fails, the fixed-step protocol is **not** a controlled re-parameterisation
of the published run, the data/step decomposition would be uninterpretable, and
M1 stops and reports the failure instead of running the rest.

---

## 4. Protocols

### 4.1 Primary — FIXED STEPS

Holds the optimizer budget constant across n and lets only the fit pool vary.
Every setting below is identical for every n and every seed:

```
arm              L4 (LOCAL-MLP), 524 673 params
batch size       8
optimizer        AdamW, lr 1e-3, weight decay 1e-4
LR schedule      none (constant) — identical to S2-C6
MAX_STEPS        900
validation       every 30 optimizer updates -> 30 validation points
                 (step 30, 60, ..., 900); step 0 (untrained) recorded separately
selection        argmax over the 30 validation points of validation H8
early halt       NONE — the budget is fixed in advance, so every arm executes
                 exactly 900 updates; patience-based halting would reintroduce a
                 step-count difference between n and is therefore forbidden here
data order       reshuffling cycle: draw a fresh permutation of the n fit rows,
                 consume it in consecutive batches of 8, redraw when exhausted
```

`n ∈ {60, 120, 180, 240, 480, 960}` × seeds `{0,1,2}` = 18 runs.

* **240 → 480 → 960 is the pre-registered primary ladder** (the M1 question).
* **60 → 120 → 180 → 240 is the step-controlled counterpart of S2-C6 Part A.**
  It is in the grid because question Q1 cannot be answered without it: the
  published 60 → 240 rise can only be decomposed against a curve that held the
  update count fixed.

Every n divides evenly by 8 (60 → 7.5 is the exception: `ceil(60/8) = 8` batches
with a 4-image final batch; see §4.3), so at n = 240 one permutation is consumed
by exactly 30 batches and the reshuffling cycle is *identical* to S2-C6's
one-permutation-per-epoch draw. That is what makes gate G3 exact rather than
approximate.

### 4.2 Secondary — FIXED EPOCH (S2-C6's own protocol, reproduced)

S2-C6's protocol verbatim: `MAX_EPOCHS = 80`, halt at `PATIENCE = 20` epochs after
the best validation overlap, validate once per epoch, select the `H8` argmax.
Run at `n ∈ {240, 480, 960}` × seeds `{0,1,2}`.

* `n = 240` here is exactly S2-C6 Part A's `A240` arm and must reproduce it
  bit-for-bit; that is gate G3's second half.
* `n = 480` and `n = 960` extend S2-C6's learning curve to the new scales, under
  the *old* protocol, so the M1 curve can be placed next to the published one.
  It is a **control, not the primary**: it re-entangles steps with n by
  construction, which is the whole reason §4.1 exists.

### 4.3 One declared asymmetry

At `n = 60` the reshuffling cycle draws 8 batches (`ceil(60/8)`), the last of
which holds 4 images, so the per-update data volume is 1.5 % lower than at other
n. This is inherited from S2-C6's epoch loop (`range(0, n_fit, BATCH)`), is
identical in both protocols, and touches only the `60/120/180` C6-alignment
points — it is reported, not corrected, because correcting it would break the
G3 reproduction.

---

## 5. Estimators and the decision rule

### 5.1 Estimators

Identical to S2-C6, so every M1 number is comparable to S2-C6's:

* Per-image held-out `head_recall@8` at the frozen `H8` checkpoint, per seed.
* A segment `Δ(n₁→n₂)` is the **seed-matched** difference of per-image vectors,
  averaged over seeds, with a **10 000-draw paired percentile bootstrap** over
  the 150 held-out images. Per-seed deltas are always reported alongside.
* `MARGIN = 0.01` — the S2-C5 / S2-C5A / S2-C6 constant.
* Secondary: `R@16`, `R@32`, `overlap256`, and the train/val gap
  (`train R@8 − val R@8` and `train R@8 − held-out R@8`, both from the same
  frozen checkpoint through the same code path).

### 5.2 Segment labels

```
RESOLVED GAIN     mean >= MARGIN  AND  CI lower bound > 0  AND  all 3 seed deltas > 0
BELOW-MARGIN      CI lower bound > 0  but not all RESOLVED conditions
EQUIVALENT-NULL   CI upper bound < MARGIN   (the data exclude a >= 1 pt gain)
AMBIGUOUS         none of the above (the CI spans both 0 and MARGIN)
```

`BELOW-MARGIN` and `EQUIVALENT-NULL` are reported as measurements and carry no
verdict on their own.

### 5.3 The M1 conclusion

**One ordered rule over the primary fixed-step ladder, first match wins.** Let
`Δ_ab = Δ(240→480)`, `Δ_bc = Δ(480→960)`, `Δ_ac = Δ(240→960)`, all under §4.1.

```
1. ENDPOINT-UNRESOLVED
   any run in the primary grid selects its H8 checkpoint at the FINAL validation
   point (index 29 of 30), for any (n, seed).
   -> the fixed budget, not the data, is the binding constraint; no ordering
      claim about the ladder is licensed, and the stage says so.

2. DATA-RESPONSIVE
   CI lower bound of Δ_bc > 0.
   -> doubling the fit pool from 480 to 960 still buys a measurable held-out
      gain at a fixed update budget. Report whether it also clears MARGIN.

3. EVIDENCE-OF-SATURATION
   CI upper bound of Δ_bc < MARGIN.
   -> the data are inconsistent with 480 → 960 buying even 1 pt at a fixed
      update budget. Reported together with Δ_ac and the secondary curve.

4. INCONCLUSIVE
   otherwise (Δ_bc is AMBIGUOUS).
```

**`EVIDENCE-OF-SATURATION` requires a bounded interval, not a failed test.**
Failing to resolve `Δ_bc` is not evidence of a plateau and is never reported as
one; the binary "not data-limited, therefore saturated" reading is explicitly
rejected. An `INCONCLUSIVE` result is a legitimate outcome and is reported as
such.

### 5.4 The six questions, and which estimator answers each

| # | question | estimator |
|---|----------|-----------|
| Q1 | how much of S2-C6's 60 → 240 rise could be an optimizer-step effect? | fixed-step Δ(60→240) minus the C6 fixed-epoch Δ(60→240) = +0.0403; the fixed-step curve is the step-controlled counterpart |
| Q2 | under fixed steps, does 240 → 480 → 960 keep improving? | `Δ_ab`, `Δ_bc` under §4.1 |
| Q3 | does the largest scale significantly beat the 240 reference? | `Δ_ac` under §4.1 |
| Q4 | is any gain consistent across seeds and benchmarks? | the 3 per-seed deltas of §5.1, plus per-benchmark `Δ_bc` (paired bootstrap within each benchmark's 50 held-out images) |
| Q5 | does the train/validation gap shrink? | §5.1's gap columns across the ladder |
| Q6 | is a downstream generation run worth it? | the S2-C6 downstream gate, re-applied unchanged in §6 |

Q4's per-benchmark reading uses 50 images per benchmark and is reported as
descriptive; the verdict of §5.3 is computed on the full 150.

---

## 6. The downstream gate (unchanged from S2-C6, and not opened by M1)

M1 **runs no generation.** It only prepares the comparison. The gate is S2-C6's,
verbatim:

```
held-out R@8 with paired CI lower bound > 0 against the L4 reference
AND consistent across all or most seeds
AND held-out R@8 >= 0.82 (aspirational 0.85)
```

If the gate does not open, no generation is run and M1 stops at the offline
verdict. If it does open, the run is exactly three arms — the best M1 checkpoint,
the `L4` reference, and the gradient teacher — with no sweep, and the checkpoint
is compared against the existing `TopK`, `block8` and `facility` selector arms on
the same held-out instances. M1 records the checkpoints, the instance keys and
the command required to do that; it does not execute it.

---

## 7. What M1 will not claim

* Not that ~0.80 is or is not a ceiling. M1 varies the fit-set size at a fixed
  update budget and stops there.
* Not that a fixed-step null means the family is saturated. §5.3 makes that
  explicit and it is the reason `EQUIVALENT-NULL` is a separate label from
  `AMBIGUOUS`.
* Not that the extension draw is a random sample of a population. It is a fixed
  draw from each benchmark's complement set, with fixed seeds, and it is reported
  as such.
* Not anything about a representation route. No query, trajectory, global or set
  context, rescue path or wider model is trained in M1.
* Not that the validation split's size is adequate at n = 960. Validation stays
  frozen at 60 for comparability with S2-C6; that the H8 rule then selects from a
  noisier relative position at large n is reported as a limitation, not patched.
