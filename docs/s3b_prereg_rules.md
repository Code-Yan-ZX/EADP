# S3-B — Are accuracy-critical teacher tokens small *bundles*? Protocol and decision rules

**Frozen 2026-09-24, before any S3-B number was computed.** Nothing in the arm
list, the class definitions, the null families or the verdict thresholds below
was written after seeing a result. The stage is mechanism discovery: no new
scorer, no new training, no new pruning method.

## 0. Question and why it is asked this way

S3-A closed `u(i | S)` as a *per-token* programme and left one positive fact
standing: the answer is decided by *which 256 tokens together* (context main
effects 5–12 nats; a `strong` context drives OCRBench gold loss to 0.000),
while single-token marginals are mostly zero and chains of single swaps die
after step 1. S2-C2 independently found that an accuracy rescue needs a small
GROUP (no single token necessary in 87 % of block ablations; median transition
block 4) and that the group came from the *teacher ordering*.

So the object of study is a **set**, and the question is whether successful
sets have inspectable structure:

> Are the teacher-missed tokens that flip a wrong answer into a right one
> organised as small evidence bundles with repeatable spatial / feature / rank
> / content structure, or is "the teacher's ordering" the whole story, or is
> even the successful bundle an accident of set composition?

Three terminal verdicts (defined numerically in §5):

| label | claim |
|---|---|
| **A. STRUCTURED-BUNDLE** | successful bundles have a cheap, pre-LLM-observable structure that distinguishes them from same-window alternatives — a future method could select *sets* by a rule |
| **B. SCORE-ORDER-ONLY** | bundles work, but carry no inspectable structure beyond "these ranks"; the teacher ordering itself is the selection rule |
| **C. CHAOTIC-SET** | even successful bundles are not distinguishable from arbitrary same-size draws; only the full teacher-set composition matters |

## 1. Frozen objects and the two bases

All instances are the frozen held-out 150 (50 per benchmark) of
`m1_common.load_m1_plan()["test"]` == `m2_accuracy.json` keys.

* **Teacher** `T` — P1-G2 gradient maps (`s2b_gradient_scores.npz`), top-256 in
  descending score, ties by ascending index (`argsort(-g, kind="stable")`).
  Verified identical (0/150 mismatch) to the `s2c0_pilot.json` arm `P1G2`
  `select_idx` lists.
* **Bank G** (primary, deployment) — student `S` = GDEP scorer
  (`m1_fixed-step_n960_L4_s2__H8.pt`, LOCAL-MLP n960 seed 2) top-256, ranked by
  that score. This is the current student; gate B1 re-proves it equals the live
  GDEP engine's `select_idx`.
* **Bank L** (powered, discovery) — student `S` = LIN_L4 top-256 from
  `s2c1_pilot.json`, whose *list order is the student rank order* (S2-C2
  convention). Chosen because S2-C2 already established that this base has
  ~23 real rescue bundles with `k* ≤ 32`: bank G's per-instance bundle count is
  not known in advance, and the structure questions need bundles to exist.
  Bank L is a retired student: **its rates are never quoted as deployment
  facts**, only its bundle structure.

Shared notation per instance: `T_only = T ∖ S` (teacher-rank order, length
`m`), `S_only = S ∖ T` (student-rank order, length `m`), `C = T ∩ S`.

## 2. The intervention protocol (unchanged from S2-C2, budget-frozen)

```
F(k) = (S ∖ S_only[m−k :])  ∪  T_only[:k]        |F(k)| = 256 for every k
```

Additions are the first `k` teacher-missed tokens in teacher-rank order; the
`k` dropped tokens are the student-rank-tail of `S_only`. Delivery is the S3-A
direct splice of `vision_tower(image)[F(k)]` in raster order between the prompt
prefix and suffix, which gate S3A-G2 proved bit-identical to the S2-C2
indicator-map delivery through the incumbent pruner. No L0–L4 pass runs over
dropped tokens; the LLM never sees them.

Endpoints recorded for every arm: **official per-instance accuracy**
(`scoring.per_sample_hits`, the metric code of the M2 runs; `hit ≥ 0.5` is the
hard right/wrong convention used since Stage 1) **and** teacher-forced gold
answer NLL `L(y | S)` (mean over deduplicated golds ≤ 6, answer tokens ≤ 64).
NLL is secondary evidence only: the per-instance bf16 floor (0–0.29 nats,
measured per instance by the batched-vs-per-gold re-scoring probe) is carried
into every NLL comparison and a difference counts as *resolved* only at ≥ 3×
that floor.

## 3. Part A — minimal rescue groups

* **A0 (GPU)** base points: `F(0)` for all 150 instances of both banks,
  generation + NLL. This is the honest denominator: "wrong" in what follows
  means *wrong in this regime*, not the frozen M2 engine outcome (the two
  disagree on ~15 % of instances by construction).
* **A1 (GPU, bank G)** every instance with `hit(F(0)) < 0.5` is swept
  `k = 1, 2, 3, 4, 6, 8, 12, 16, 24, 32` in ascending order, **stopping at the
  first `k` with `hit ≥ 0.5`**; that `k` is `k*` and `G = T_only[:k*]` is the
  minimal rescue group. If nothing rescues by 32, the sweep extends to
  `k = 48, 64` and the instance is recorded `k* = nil` (with whatever the
  extension produced).
* **A1-L (mostly reuse)** bank L's rescue inventory is taken from the frozen
  `s2c2_rescue.json` / `s2c2_rescue_plan.json` (fine grid `1..96`, 28 rescuable
  instances, 23 of them rescued at `k* ≤ 32`). No new generations are run for
  it — gate B2 re-generates 6 of those arms with the S3-B harness and requires
  the recorded prediction text back. NLL along the `k` grid is added cheaply.

Per-k bookkeeping: added tokens, dropped tokens, hit, NLL, prediction text.

## 4. Part B — is the bundle actually indivisible?

For every bundle (`k* ≤ 32`, both banks) the following arms are measured
(same budget, same removal rule unless stated). `W = min(m, max(2k*, k*+16))`
is the *rank window* the bundle lives in.

| arm | definition | what it asks |
|---|---|---|
| `full` | `F(k*)` | reference (= Part A endpoint) |
| `single_j` | `(S ∖ {S_only[m−1]}) ∪ {T_only[j]}`, `j < min(k*, 8)` | is one member sufficient alone? |
| `loo_j` | `F(k*) ∖ {T_only[j]} ∪ {S_only[m−1−j]}` for every `j < k*` | is member `j` necessary? (the displaced student token returns) |
| `top1/top2` | `F(1)`, `F(2)` | is the head of the order the whole payload? |
| `randwin_r` | `k*` tokens drawn uniformly from `T_only[:W] ∖ G`, `r = 1..6` | is *this* set special among same-rank-window sets? (the key null) |
| `randT_r` | `k*` tokens from `T_only ∖ G`, `r = 1..4` | same-size draw from the whole missed pool |
| `randO_r` | `k*` tokens from image tokens ∉ `S ∪ T`, `r = 1..4` | "any k* extra pixels" baseline |
| `shift_a` | additions `= T_only[a : a+k*]`, `a ∈ {1,2,4}` where legal | is it the bundle or just the next `k*` ranks? |
| `blockperm_r` | keep `T_only[:k*−b]`, replace the last `b = max(1, k*//2)` by a random draw from `T_only[k* : k*+2b]`, `r = 1..4` | membership vs window-position inside the ordering |
| `remrand_r` | additions `= G`, but drop `k*` *random* members of `S`, `r = 1..3` | removal-side confound (S3-A: resolvable on OCRBench) |

Derived quantities per bundle: `n_suff` = #`single_j` with `hit ≥ 0.5`;
`n_nec` = #`loo_j` with `hit < 0.5`; **group-only** = `full` hit ∧
`n_suff = 0` ∧ `n_nec = k*` (nothing alone is enough, every member is needed).
Null draws are scored at full size only.

## 5. Part C — structure of the successful bundles, and the verdict statistics

Every set measured in Parts A/B (bundles *and* their nulls, which are
size-matched, instance-matched, window-matched) gets the same feature vector.

**Spatial** (32×32 merged grid): mean pairwise Chebyshev and Euclidean
distance; min pairwise distance; bbox area ÷ `k`; bbox aspect; #connected
components under 8-connectivity; fraction of pairs sharing a row; sharing a
column; fraction of cells adjacent (Chebyshev 1) to a retained `C` member;
distance from set centroid to the evidence-backbone centroid (top-32 teacher
members of `C`); centroid radius from image centre.

**Feature** (from the frozen caches, pre-LLM-observable): mean pairwise cosine
of `s2c1_feats_L4`; mean cosine of members to `S`; mean L4 norm; mean L2→L4
displacement (`s2c1_feats_L2`); mean cosine in the merged vision-embedding
space (`s3a_vis`).

**Rank / score**: mean and max teacher rank inside `T_only`; teacher-score span
within the set; gap from the set's weakest member to the next candidate outside
it; mean student score of members; mean teacher−student score z-gap.
(For prefix bundles rank-contiguity is 1 by construction, so this family is
reported for the *nulls* as the covariate that must be matched, never as
evidence for A.)

**Content**: per-cell image statistics of the source pixels (luminance mean/std
and edge energy of the 64×64 block each merged token covers, and the fraction
of high-edge cells); same-text-line indicator (all cells within one 64-px row
band of the 1024×1024 frame); and the **crop probe** (GPU, bank L bundles and
their `randwin` nulls): the union-bbox of the set is cropped from the image,
resized to 1024×1024, and the *same question* is answered from that crop
unpruned. If a bundle is a legible evidence unit, its crop answers correctly
far more often than a same-window null's crop.

**Statistics.**
* *Prefix-vs-null*: rescue rate of the exact bundle vs the pooled `randwin` /
  `randT` / `randO` draws at matched `(instance, k)`; Fisher exact on pooled
  counts plus a per-instance paired bootstrap.
* *Additivity*: distributions of `n_suff`, `n_nec`, group-only rate, per
  benchmark and bank; the NLL version of the same (is `ΔL(full)` super-additive
  over members' `ΔL(single_j)`?).
* *Structure separation*: for each feature `P`, the paired difference
  `P(bundle) − mean P(randwin nulls)` per instance. A feature is a **candidate
  structure variable** iff a leave-one-instance-out single-threshold classifier
  on that difference reaches AUC ≥ 0.65 with bootstrap CI above 0.5, the sign is
  consistent on ≥ 2 of 3 benchmarks, and the Wilcoxon signed-rank over bundles
  gives `p < 0.05`. Only pre-LLM-observable features qualify.

**Verdict thresholds** (evaluated on bank G first; bank L must agree in
direction on any clause quoted as a structure claim):

1. **Gate 0 — bundles exist.** `median(k*) ≤ 16` over Part-A rescued instances
   AND the exact bundle beats its `randwin` nulls (Fisher `p < 0.05`, pooled
   over both banks). If Gate 0 fails → **C. CHAOTIC-SET**, and no structure
   claim is made regardless of what Part C shows.
2. **A. STRUCTURED-BUNDLE** iff Gate 0 holds AND ≥ 1 pre-LLM-observable
   feature is a candidate structure variable (above) AND that feature also
   separates *successful* from *unsuccessful* size-matched sets across
   instances at `p < 0.05`.
3. **B. SCORE-ORDER-ONLY** iff Gate 0 holds AND no feature reaches the
   candidate bar AND the ordering still assembles working sets:
   `shift_a`/`blockperm` arms degrade relative to `full` (i.e. *which* ranks,
   not just *how many*).
4. **C. CHAOTIC-SET** iff Gate 0 fails, OR the bundles' advantage is fully
   explained by rank depth (`shift`/`blockperm` ≈ `full` ≈ `randT` once `k` is
   matched) and successful-bundle membership is not reproducible
   (cross-bank Jaccard of `G` on shared instances ≤ 0.2 AND `randwin ≈ full`).
5. Any split outcome reports the clause that failed. No clause may be
   re-thresholded after seeing the data; a threshold change would be recorded
   as an amendment with its reason.

## 6. Reuse, cost and gates

Nothing is re-trained or re-scored: teacher maps, GDEP checkpoint, LIN_L4
pilot sets, L2/L4 feature caches, `s3a_vis` vision cache, M2 outcomes and the
whole S2-C2 rescue grid are read from disk. New GPU work: A0 300 generations,
A1 ~55 instances × ≤10 prefix points with early stop, B ~23–35 bundles × ~20
arms, crop probe ~120 generations. Estimated 2.5–3.5 h on one A40.

| gate | statement |
|---|---|
| B1 | recomputed bank-G `S` == live GDEP engine `select_idx` (topk@256), 3 probes, symdiff 0 |
| B2 | bank-L reuse: 6 arms of `s2c2_rescue.json` re-generated with the S3-B harness return the recorded prediction text |
| B3 | `F(0)` bank-G == the frozen `s3a_nll.json` base predictions on the 72 shared instances |
| B4 | determinism: one arm twice, bit-identical |
| B5 | every arm delivers exactly 256 tokens and its recorded set (set-equality on the delivered splice) |

## 7. Non-goals

No conditional scorer, no pair-interaction network, no support-token model, no
training of any kind (S3-A stop rule). No claim that any *method* follows from
this stage; the deliverable is a verdict about bundle structure.

## 8. Amendments recorded after freezing (2026-09-24, none touches a threshold)

Each of these either *supplies* a measurement a frozen clause requires, or fixes
an implementation bug that made a frozen test unrunnable. Verdict-bearing
thresholds (§5) are unchanged throughout; the report's §8 lists the same items
with their consequences.

1. **Additional measurement stages** — `s3b_nulls` (matched nulls on
   non-rescued instances, required by §5 clause 2's "separates successful from
   unsuccessful size-matched sets across instances"), `s3b_subst` (trades one
   bundle member for a deeper teacher token at constant bundle size: the frozen
   `randwin` null keeps zero bundle members and `loo` returns a student token,
   so the frozen pair cannot separate membership from rank depth),
   `s3b_bands` (the rank-band ladder), `s3b_crop` + `s3b_poscheck` (the content
   axis and its calibration).
2. **Distinct-set counting** — the zero-overlap window null is forced unique
   once `k* ≥ 16` (`T_only[:W] ∖ G` holds exactly `k*` tokens), so pooled rates
   count each distinct added set once (`s3b_analysis.json: dedup`,
   `window_degeneracy`).
3. **`single` arms of monadic bundles** — a `k*=1` bundle's `single0` arm *is*
   the bundle; it is re-labelled `single_trivial` and excluded from the
   "one member alone" rate.
4. **Classifier direction (§5's "candidate structure variable")** — the frozen
   wording is "a leave-one-instance-out single-threshold classifier", whose
   first fitted parameter is the comparison direction. The implementation
   initially pinned it to "higher is better", which reads every
   lower-is-better property (bbox area per token, connected components,
   distance to retained evidence) as a null result. Fixed: the direction is
   estimated once from the pooled within-group sign (T1) or fitted on the
   training folds (T3), and both the signed and oriented AUC are reported. This
   can only make clause A *easier* to satisfy, so it cannot have produced a
   harsher verdict than the frozen text intends.
4b. **The frozen sign-consistency requirement is now actually enforced** —
    clause A's "the sign is consistent on >= 2 of 3 benchmarks" was in the frozen
    text but absent from the first implementation; `T1` now reports
    `auc_by_ds` per benchmark x base panel and `n_ds_beyond_chance`, and the
    candidate test requires >= 2. This is a tightening, not a relaxation.

5. **T3's fold structure** — at a fixed `k` each instance contributes exactly
   one prefix row, so a within-fold AUC is undefined (the first implementation
   returned NaN for every feature). Replaced by the correct leave-one-instance-out
   estimate: pooled out-of-fold predictions, plus the Mann-Whitney separation
   test that §5 clause 2 actually names.
6. **Gate B1** compares sets, not list orders (bank-G `S` is stored in student
   rank order; the engine returns sorted indices). **Gate B2** probes were
   re-selected to arms that S2-C2 scored *correct*, since reproducing a failure
   proves less than reproducing a success.
7. **Generation cap 256** for every arm of both banks (S2-C2 used 2048, S3-A 64);
   reconciled by gates B2/B2b/B3.
