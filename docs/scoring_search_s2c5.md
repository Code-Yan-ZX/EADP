# S2-C5 — Token-Local vs Set-Dependent Factorization Diagnosis

**Scope.** Stage-2 S2-C5. One mechanism question, and deliberately not a design
step:

> The gradient teacher's high-value ranking-head signal — is it a **shared
> nonlinear function of the token alone**, or does identifying it **require the
> image's other visual tokens**?

S2-C4 closed the previous reading. The per-image oracle direction is strong
(0.950 of the teacher's Top-8, against 0.766 for the trained shared direction)
but **not transferable**: every retrieval and regression rule that tries to pick
a direction per image from cheap forward statistics lands at or below the
no-adaptation direction. So the adaptation variable is not recoverable as a
per-image *direction*. That leaves the factorization question this stage settles:
is the missing information inside the token's own hidden state (a nonlinearity
the linear probe cannot express), inside an image-level summary, or inside the
*token set* — the configuration of which other tokens are present?

No method is designed here. No architecture is proposed, no downstream generation
is run, and no query conditioning is added anywhere. The LLM and the vision tower
are never loaded: every number is a read-out of the cached S2-C1 layer-4 hidden
states plus the S2-B P1-G2 teacher ranking, plus 12 tiny trained scorers
(0.52 M – 2.36 M parameters).

**Status.** Complete. Scripts:
`Qwen_vl/scripts/discovery/s2c5_{common,models,c0_diagnosis,train,controls,consolidate,figure}.py`,
runner `run_s2c5.sh`. Deliverables:
`outputs/discovery/s2c5_factorization.json` (main), `s2c5_c0_diagnosis.json`,
`s2c5_train.json`, `s2c5_controls.json`, `s2c5_scores.npz` (5 580 score vectors),
`s2c5_perimage.npz`, `figures/s2c5_factorization.png`.

---

## VERDICT

```
S2-C5 A -- NONLINEAR-LOCAL.

The teacher's head signal is a shared nonlinear function of the token's own
layer-4 hidden state. Nothing about the image's other tokens is needed to
identify it.

LOCAL-MLP, a per-token MLP with no access to any other token, any pooled image
feature or the query, reaches 0.7972 held-out Top-8 recall@256 against the
linear reference's 0.7656 -- +0.0317 [+0.0189, +0.0444], reproduced in all
three seeds (+0.019/+0.036/+0.040). A 4.5x wider token-local MLP (2 360 833
parameters) reaches 0.7947, i.e. no better: the local family is saturated, so
the gain is a property of nonlinearity, not of capacity.

Neither contextual arm adds anything on top. GLOBAL-CTX (permutation-invariant
image pooling, parameter-matched) is 0.7786 -- 0.0186 BELOW the local arm
[-0.0311, -0.0061], with a CI against the reference that includes zero.
SET-CTX (one-head, one-round token-token attention, parameter-matched) is
0.7978, which is +0.0006 over LOCAL-MLP [-0.0075, +0.0086] -- indistinguishable.

And the wrong-image control says why. Replacing every held-out image's context
with another image's costs GLOBAL-CTX +0.0012 and SET-CTX -0.0001, while moving
the scores substantially (mean |delta score| 0.28-0.55). The contextual arms
were never reading sample-specific token configuration. The reason is visible in
the mechanism read-out: SET-CTX's interaction term is large (68 % of the
residual it is added to) but its attention is nearly uniform -- effective support
534 of 1024 tokens -- so it computes a diffuse permutation-invariant average
rather than a pairwise configuration. GLOBAL-CTX's context term is 4x the
residual's norm but varies across images by only 6 % of it, i.e. it is a shared
bias.

The wall S2-C3 hit was linearity, not locality. It is now partly paid: the
token-local factor moves the number by +3.2 pt. It is not fully paid -- the
oracle direction still reaches 0.95, and no arm here clears the 0.82 downstream
gate.
```

The single most useful number is the ordering
**LOCAL-MLP ≈ SET-CTX > GLOBAL-CTX**. If token-token configuration mattered, the
attention arm would lead; if image-level conditioning mattered, the pooled arm
would; instead the arm that sees the least wins, and the arm that sees the most
does worst.

---

## 0. Executive summary

1. **The misses are not at the frontier.** Of the 1 200 teacher-Top-8 tokens on
   the held-out 150, HEAD_RANK drops 271 (22.6 %); 70.7 % of images lose at least
   one. Their median student rank is **417** — 161 places past the 256 frontier.
   Only **13 %** are within 32 places of it, 25 % within 64, 42 % within 128. The
   per-image *worst* miss has median rank 418 and p90 771. A frontier-local
   architecture (contextualize the top-256) is therefore not justified, and none
   was built (§2).
2. **But a modest pool still covers the head.** 65.7 % of misses are inside
   rank 512, so the candidate-oracle coverage is **0.922 at C = 512** and 0.983 at
   C = 768, against 0.774 at C = 256. A two-stage design (cheap pool of 512, then
   contextual re-ranking inside it) retains a 0.92 ceiling and is not excluded by
   the geometry — it is excluded by §4, not by §2.
3. **A nonlinear token-local function is sufficient and saturated.** LOCAL-MLP
   (524 673 parameters, no context, no query) reaches **0.7972** against the
   linear reference's 0.7656, **+0.0317 [+0.0189, +0.0444]**, positive in all
   three seeds. The 4.5x wider token-local ceiling (2 360 833 parameters) reaches
   0.7947, **-0.0025 [-0.0147, +0.0094]** versus the matched local arm. The local
   family is not capacity-limited; it is at its ceiling (§3).
4. **Image-global context does not help; it hurts.** GLOBAL-CTX is **0.7786**,
   which is **-0.0186 [-0.0311, -0.0061]** against parameter-matched LOCAL-MLP and
   +0.0131 [-0.0031, +0.0292] against the reference — a CI that includes zero. It
   gains only on OCRBench, and is *negative* on TextVQA (-0.0108) (§3.2).
5. **Genuine token-token interaction adds nothing.** SET-CTX is **0.7978**,
   **+0.0006 [-0.0075, +0.0086]** over LOCAL-MLP. It does beat GLOBAL-CTX
   (+0.0192 [+0.0044, +0.0333]), which is exactly what one expects if the useful
   part of both is a diffuse average and the pooling arm's max-pool adds noise —
   not if pairwise configuration were the missing factor (§3.3).
6. **The wrong-image control does not eliminate a contextual gain, because there
   is none.** Honest minus wrong-image is **+0.0012** for GLOBAL-CTX and
   **-0.0001** for SET-CTX, averaged over five derangements per seed, with the
   substituted context drawn from other held-out images and (separately) from fit
   images. The substitution is not inert — mean |delta score| is 0.28–0.55 and the
   top-256 Jaccard drops to 0.86–0.93 — the read-out moves and the metric does
   not (§4).
7. **SET-CTX's interaction term is active, not switched off — and that is the
   mechanism finding.** Its token-token term is 68 % of the residual stream's
   norm, so the arm did not collapse to the local form by zeroing the term. Its
   attention entropy is 85.6 % of maximum, giving an **effective support of 534
   of 1 024 tokens** with a mean top-1 weight of 0.056. A near-uniform average
   over half the token set is a permutation-invariant summary in all but name,
   which is why it lands where the pooling arm lands (§5).
8. **GLOBAL-CTX's context is a shared bias.** Its context term has 4x the norm of
   the residual it modulates (63.4 vs 16.4), but its per-dimension spread *across
   images* is 1.00 — **6.1 %** of the residual — so the term is large, nearly
   constant, and carries no sample-specific signal (§5).
9. **The gate does not fire.** No capacity-matched arm reaches 0.82
   (best: SET-CTX 0.7978). The pre-registered downstream rule requires 0.82 with
   a positive CI, so **no generation was run** and none should be run from this
   stage (§6).
10. **Every headline gain is positive on all three benchmarks.** LOCAL-MLP:
    TextVQA +0.0258 [+0.0042, +0.0500], DocVQA +0.0292 [+0.0067, +0.0533],
    OCRBench +0.0400 [+0.0217, +0.0583]. SET-CTX: +0.0317 / +0.0317 / +0.0333,
    the most uniform profile in the ladder. The gain is not a benchmark artifact
    (§3.4).

---

## 1. What is held fixed, and what the objects are

Nothing about the S2-C1/C2/C3/C4 configuration is re-tuned. The target, the loss
family, the split, the preprocessing, the optimizer and the early-stopping
criterion are S2-C3's, verbatim.

| object | value | source |
|---|---|---|
| model | Qwen3-VL-8B, vision tower + LLM frozen, never loaded | S2-C1 |
| feature cache | layer-4 visual hidden states, 1024 × 4096 fp16 | `s2c1_feats_L4.npy` |
| tokens | 1 024 visual tokens per image, selection budget T = 256 | S2-B / S2-C1 |
| preprocessing | one fixed step: fit-split per-dimension mean/std | S2-C1 |
| teacher | P1-G2 gradient map (`s2b_gradient_scores.npz`) | S2-B |
| split (image-level) | fit 240 / val 60 / **held-out 150** / causal 15 | `s2c1_features.json` |
| target | `HEAD_RANK`: positives = teacher Top-32, weights 32/(r+1) mean-normalised | S2-C3 |
| loss | balanced BCE + 0.5 × weighted all-pairs margin ranking, margin 1.0 | S2-C3 |
| early stopping | validation Top-256 overlap, patience 20, max 80 epochs | S2-C1/C3 |
| seeds | 3 (0, 1, 2) | S2-C3 |
| **primary metric** | **held-out teacher Top-8 recall@256** | S2-C2/C3/C4 |
| supplementary | Top-16/32 recall, exact Top-8/16/32 agreement, Top-256 overlap | S2-C3 |

**Top-256 overlap never vetoes a head-recovery result.** S2-C3 recorded HEAD_RANK
at overlap256 0.513 against BASE's 0.561 while *gaining* head recall, and S2-C2
established the downstream value is paid in head recall. All four trained arms do
carry overlap256 *below* the bulk-target linear probe (LOCAL-MLP 0.5220,
GLOBAL-CTX 0.5258, SET-CTX 0.5222, WIDE 0.5185, against LIN_L4's 0.5618), though
all four are *above* HEAD_RANK's 0.5132 — the same trade S2-C3 saw. It is
recorded, not treated as a cost.

**The n = 8 causal tier is not used anywhere in this stage**, for arm selection or
for the verdict.

### 1.1 C5-0: the candidate accessibility diagnosis, and what it decided

Before any scorer was trained, the frozen baselines were asked one question:
*when HEAD_RANK or LIN_L4 keeps a teacher-head token out of the selected 256,
where does that token actually sit?* (Script `s2c5_c0_diagnosis.py`; no training,
pure read-out of the cached score banks.)

The answer decided the architecture of arm D. Because the misses are **not**
frontier-local (§2), no frontier-only contextual architecture was built. The
alternative permitted by the brief — confine token-token interaction to a top-384
or top-512 candidate set — was rejected on the same evidence, and arm D instead
reads the full 1 024-token set, so that the comparison against LOCAL-MLP is a
comparison of *information*, not of scope. The candidate-oracle coverage is
reported anyway (§2.3), because it bounds what a two-stage design could reach.

### 1.2 The ladder

The design constraint that makes the ladder interpretable is that the arms must
differ in **information access, not capacity**. So all four arms share one trunk

```
a_i = GELU(P h_i)                    P: (4096, 128)      524 416 parameters
```

and differ only in what reaches the read-out:

| arm | score | extra parameters | reads |
|---|---|---|---|
| **`LOCAL-MLP`** | `w₂ · GELU(a_i + b)` | 0 | the token alone |
| **`GLOBAL-CTX`** | `w₂ · GELU(a_i + V c + b)`, `c = [mean_j a_j ; max_j a_j]` | V (128×256) → +32 768 (+6.2 %) | token + permutation-invariant image pool |
| **`SET-CTX`** | `w₂ · GELU(a_i + Σ_j α_ij V_v a_j)`, one head, rank 32 | Q,K,V_v → +24 576 (+4.7 %) | token + the whole token set |
| **`LOCAL-MLP-WIDE`** | `w₂ · GELU(W_mid GELU(P₂ h_i))`, `P₂: (4096, 512)` | 2 360 833 total (+350 %) | the token alone, 4.5× capacity |

`LOCAL-MLP-WIDE` is deliberately **not** capacity-matched and is not a deployable
arm: it is the ceiling of the token-local hypothesis. If a 4.5× wider token-local
MLP also fails, "the head signal is a nonlinear function of the token alone" is
dead regardless of where the matched arm lands. It does not fail — it matches.

No arm reads the query. No arm reads another image at inference. `SET-CTX` is one
attention head, one round, no layer norm, no feed-forward, no depth: it is
deliberately below the threshold at which "a Transformer was trained" would be an
honest description. Latency is negligible in every case (scorer 0.149 ms local,
0.224 ms global, 0.285 ms set, against a 32.5 ms layer-4 prefix).

The two required mechanism controls are built in:

* **wrong-image context** — `context` and the attention key/value set are
  computed by separate methods and passed in explicitly, so the control is a
  one-line intervention: score image *i*'s queries against image *j*'s context or
  token set. Run as a derangement (no image keeps its own), redrawn five times per
  seed, and separately with the substitute drawn from fit images.
* **capacity control** — structural: the shared trunk is identical across arms,
  so the additions are +6.2 % / +4.7 % and cannot explain a 1 pt difference.

### 1.3 The decision rule, fixed before the ladder was read

The rule lives in the docstring of `s2c5_consolidate.py`, in the file that applies
it. At the time it was written only C5-0 and a 2-epoch plumbing smoke test had
been seen.

* `MARGIN` = **0.01** of head_recall@8. One point. S2-C3's entire target move was
  0.021, so a smaller difference is not an architecture result.
* `CLEARS(arm)` — seed-averaged delta ≥ MARGIN, paired-bootstrap 95 % CI lower
  bound > 0, **and** all three seed-matched deltas (arm_s − reference_s) positive.
  A mean that rides on one lucky seed is not a mechanism. This mattered: the
  reference's own seed sd is 0.0085 and the arms' is 0.010–0.014.
* `USES_CTX(arm)` — contextual arms only: swapping the context for another
  image's must cost ≥ MARGIN with the paired CI excluding zero.

Cascade, first match wins: `UNRESOLVED` (nothing clears) → `NONLINEAR-LOCAL`
(LOCAL-MLP clears, no contextual arm beats it) → `SET-DEPENDENT` (SET-CTX beats
both matched arms *and* USES_CTX) → `IMAGE-GLOBAL` (GLOBAL-CTX beats LOCAL-MLP,
SET-CTX adds nothing) → `UNRESOLVED` (the arms do not separate).

Worked example of the rule catching something the headline would hide:
GLOBAL-CTX's per-seed deltas are **+0.034 / +0.001 / +0.004**. Its mean, +0.0131,
is positive and would read as a small win; the reproducibility clause shows it is
one seed.

---

## 2. C5-0 — where the missed head tokens are

Reference: HEAD_RANK, the S2-C3 arm, seed-averaged scores, held-out 150 (50 per
benchmark), head_recall@8 = 0.774.

### 2.1 The headline: the misses are far from the frontier

| quantity | teacher Top-8 | Top-16 | Top-32 |
|---|---|---|---|
| head tokens examined | 1 200 | 2 400 | 4 800 |
| missed (outside the 256) | 271 (22.6 %) | 597 | 1 419 |
| rank median | **417** | 436 | 454 |
| rank mean | 466 | 492 | 503 |
| rank p75 / p90 / p95 | 580 / 717 / 800 | 621 / 789 / 884 | 646 / 801 / 886 |
| ≤ 320 | 25.1 % | 22.2 % | 21.4 % |
| ≤ 384 | 42.1 % | 38.2 % | 37.3 % |
| ≤ 512 | 65.7 % | 61.2 % | 58.2 % |
| ≤ 768 | 92.3 % | 89.2 % | 87.4 % |

The median missed head token sits **161 places past the frontier**. Expressing the
same thing as distance past the boundary: **13 %** of misses are within +32,
25 % within +64, 42 % within +128, 66 % within +256.

Per image (150 images):

| quantity | value |
|---|---|
| images losing ≥ 1 head token | 106 / 150 = **70.7 %** |
| head tokens missed per image | mean 1.81, median 2, p90 4 |
| **worst** missed rank per image | median **418**, p75 634, p90 771, p95 853, max 999 |
| mean missed rank per image | median 428 |

So even the *best-case* image statistic — the worst miss it suffers — has a median
of 418. This is the number that rules out the frontier-local architecture: a
scorer that only contextualizes the student's top-256 would be looking at a region
that contains a minority of the tokens it is missing.

**The miss profile rises with teacher rank but is not concentrated at the
marginal head.** Miss rate by the token's position in the teacher's own Top-8:
0.15, 0.16, 0.22, 0.22, 0.23, 0.27, 0.33, 0.23. The teacher's single best token is
still lost 15 % of the time, so this is not "only the marginal head members are
hard".

### 2.2 LIN_L4 shows the same geometry

LIN_L4 (0.7358) drops 317 head tokens; median rank 416, ≤ 384 42.1 %, ≤ 512 69.4 %,
per-image worst median 465. The linear and rank-targeted arms fail on the same
tokens in the same place, which is why C5-0's conclusion applies to both.

### 2.3 Candidate-oracle coverage — the ceiling of a pool-restricted scorer

Fraction of the teacher's Top-8 inside the student's top-C, HEAD_RANK:

| C | 256 | 320 | 384 | 448 | 512 | 640 | 768 | 896 |
|---|---|---|---|---|---|---|---|---|
| recall@C | 0.774 | 0.831 | **0.869** | 0.902 | **0.922** | 0.958 | 0.983 | 0.996 |

Read this as a ceiling, not a plan. A scorer allowed to re-rank only the top-512
of a cheap first stage could reach at most 0.922 for the head, and 0.869 at 384 —
both above the 0.82 gate. So the *geometry* does not forbid a two-stage design.
What forbids it is §4: the contextual machinery that would do the re-ranking adds
nothing even when it can see every token, so restricting its scope cannot help.

### 2.4 Per benchmark

| benchmark | head_recall@8 | missed | median miss rank | ≤ 384 | ≤ 512 | worst miss (median) |
|---|---|---|---|---|---|---|
| TextVQA_VAL | 0.833 | 67 | 390 | 45 % | 70 % | 481 |
| DocVQA_VAL | 0.723 | 111 | 438 | 40 % | 62 % | 640 |
| OCRBench | 0.767 | 93 | 421 | 43 % | 67 % | 514 |

DocVQA is hardest at the headline (0.723) and its worst miss is deepest (median
640), but the *shape* of the failure is the same in all three: a median miss
between 390 and 438, well past 256 everywhere. There is no benchmark for which the
frontier story would have worked.

---

## 3. The ladder

Held-out 150, three seeds, seed-mean of per-seed metrics. Reference: **HEAD_RANK
0.7656** (per-seed 0.7608 / 0.7775 / 0.7583, seed sd 0.0085).

| arm | params | recall@8 | sd | **Δ vs ref** | 95 % CI | per-seed Δ | clears |
|---|---|---|---|---|---|---|---|
| `HEAD_RANK` (ref) | 4 097 | 0.7656 | 0.0085 | — | — | — | — |
| `S2C1_LIN_L4` | 4 097 | 0.7358 | — | −0.0297 | — | — | no |
| **`LOCAL-MLP`** | 524 673 | **0.7972** | 0.0136 | **+0.0317** | **[+0.0189, +0.0444]** | +0.019/+0.036/+0.040 | **yes** |
| `GLOBAL-CTX` | 557 441 | 0.7786 | 0.0133 | +0.0131 | [−0.0031, +0.0292] | +0.034/+0.001/+0.004 | no |
| **`SET-CTX`** | 549 121 | **0.7978** | 0.0104 | **+0.0322** | **[+0.0186, +0.0461]** | +0.042/+0.030/+0.025 | **yes** |
| `LOCAL-MLP-WIDE` | 2 360 833 | 0.7947 | 0.0113 | +0.0292 | [+0.0119, +0.0456] | +0.047/+0.019/+0.022 | yes |

Supplementary metrics (seed-mean):

| arm | recall@16 | recall@32 | agree@8 | agree@16 | agree@32 | overlap256 |
|---|---|---|---|---|---|---|
| `HEAD_RANK` | 0.7408 | 0.6941 | 0.2656 | 0.2401 | 0.2444 | **0.5132** |
| `S2C1_LIN_L4` | 0.7188 | 0.6813 | 0.1917 | 0.1900 | 0.2085 | **0.5618** |
| `LOCAL-MLP` | 0.7719 | 0.7188 | **0.2875** | 0.2556 | 0.2637 | 0.5220 |
| `GLOBAL-CTX` | 0.7538 | 0.7115 | **0.2886** | **0.2607** | **0.2658** | 0.5258 |
| `SET-CTX` | **0.7722** | **0.7204** | 0.2786 | 0.2528 | 0.2587 | 0.5222 |
| `LOCAL-MLP-WIDE` | 0.7669 | 0.7156 | 0.2792 | 0.2517 | 0.2652 | 0.5185 |

Every nonlinear arm shifts the whole head profile, not just recall@8: recall@16
+0.031, recall@32 +0.025, exact Top-8 agreement +0.022 over the linear reference.
The gain is not a Top-8-only artifact.

### 3.1 LOCAL-MLP — the local hypothesis, and its ceiling

LOCAL-MLP sees `h_i` and nothing else. It has no context, no query, no other
token, and it clears the reference by **+0.0317 [+0.0189, +0.0444]** with all
three seed-matched deltas positive.

The wide arm settles whether that is a capacity story. `LOCAL-MLP-WIDE` has
**4.5× the parameters** (2 360 833 vs 524 673) and two hidden layers, and reaches
0.7947 — **−0.0025 [−0.0147, +0.0094]** against the matched local arm, with
per-seed deltas of −0.017/−0.018 on two of three seeds. **More capacity in the
local family buys nothing.** The 128-wide token-local MLP is at the family's
ceiling of ≈ 0.795.

That ceiling is the honest bound on this factorization: whatever the remaining
0.795 → 0.95 gap is, it is not "a bigger nonlinear function of the token alone",
and by the next two subsections it is not context either.

### 3.2 GLOBAL-CTX — image-level conditioning does not help

GLOBAL-CTX is parameter-matched to LOCAL-MLP (+6.2 %) and adds a mean-and-max
permutation-invariant summary of the image's own token set. It lands **0.7786**:

* vs LOCAL-MLP: **−0.0186 [−0.0311, −0.0061]** — worse, with a CI excluding zero;
* vs the reference: +0.0131 [−0.0031, **+0.0292**] — a CI that includes zero, and
  per-seed deltas +0.034/+0.001/+0.004, i.e. carried by seed 0 alone.

Adding image-level global information to a token-local function makes it *worse*.
The per-benchmark profile agrees: GLOBAL-CTX is the only arm in the ladder that is
negative on any benchmark (TextVQA −0.0108 [−0.0408, +0.0183]).

### 3.3 SET-CTX — real token-token interaction adds nothing

SET-CTX is the smallest module that can express a genuine pairwise interaction:
one head, rank 32, one round, parameter-matched to LOCAL-MLP (+4.7 %).

* vs LOCAL-MLP: **+0.0006 [−0.0075, +0.0086]** — indistinguishable. Per-seed
  +0.023/−0.006/−0.015.
* vs GLOBAL-CTX: +0.0192 [+0.0044, +0.0333] — it does beat the pooling arm.

That second number is worth reading carefully, because it is easy to mistake for
evidence of set-dependence. It is not. If pairwise configuration carried the
missing signal, SET-CTX would beat *both* local and global arms. What actually
happens is that SET-CTX ≈ LOCAL-MLP > GLOBAL-CTX, which is the ordering you get
when (a) the useful content of the interaction term is a diffuse average, and
(b) the pooling arm is handicapped by its max-pool branch. §5 confirms (a) by
measurement.

### 3.4 Is the gain a benchmark artifact?

No. LOCAL-MLP and SET-CTX both gain on all three benchmarks with CIs excluding
zero:

| arm | TextVQA_VAL | DocVQA_VAL | OCRBench |
|---|---|---|---|
| `LOCAL-MLP` | +0.0258 [+0.0042, +0.0500] | +0.0292 [+0.0067, +0.0533] | +0.0400 [+0.0217, +0.0583] |
| `SET-CTX` | +0.0317 [+0.0067, +0.0575] | +0.0317 [+0.0100, +0.0542] | +0.0333 [+0.0117, +0.0567] |
| `GLOBAL-CTX` | **−0.0108** [−0.0408, +0.0183] | +0.0150 [−0.0133, +0.0450] | +0.0350 [+0.0133, +0.0575] |
| `LOCAL-MLP-WIDE` | +0.0075 [−0.0258, +0.0408] | +0.0383 [+0.0108, +0.0675] | +0.0417 [+0.0183, +0.0675] |

SET-CTX has the flattest profile in the ladder (+0.0317 / +0.0317 / +0.0333), but
it is statistically the same arm as LOCAL-MLP, so this is a remark about
consistency, not a separate finding.

---

## 4. The wrong-image context control

Protocol. For each contextual arm and each seed the model is reloaded from its
checkpoint and the held-out set is scored twice: honestly, and with every image
reading a *different* image's context (a derangement of the 150, so no image keeps
its own). Redrawn 5 times per seed. A third pass substitutes context drawn from
the **fit** split, i.e. from images the scorer never evaluated on. The honest pass
is also re-run through the borrowed-context code path with an identity
permutation; it reproduces the direct pass to **max |Δ| = 0.0** in every case, so
the control measures the substitution and not the plumbing.

**Result: swapping the context costs nothing.**

| arm | honest rec@8 | wrong-image rec@8 | honest − wrong | mean \|Δscore\| | top-256 Jaccard | wrong-fit |
|---|---|---|---|---|---|---|
| `GLOBAL-CTX` s0 | 0.7950 | 0.7922 | +0.0028 | 0.382 | 0.891 | 0.7925 |
| `GLOBAL-CTX` s1 | 0.7783 | 0.7760 | +0.0023 | 0.477 | 0.862 | 0.7825 |
| `GLOBAL-CTX` s2 | 0.7625 | 0.7640 | −0.0015 | 0.550 | 0.864 | 0.7625 |
| `SET-CTX` s0 | 0.8025 | 0.8015 | +0.0010 | 0.283 | 0.929 | 0.8042 |
| `SET-CTX` s1 | 0.8075 | 0.8063 | +0.0012 | 0.284 | 0.928 | 0.8067 |
| `SET-CTX` s2 | 0.7833 | 0.7857 | −0.0023 | 0.304 | 0.920 | 0.7867 |

Pooled: **GLOBAL-CTX +0.0012, SET-CTX −0.0001**, against a 0.01 attribution margin.
`USES_CTX` is false for both, and by a factor of ten.

Two things make this a mechanism result rather than a failed manipulation check.

1. **The substitution is not inert.** Replacing the context moves the raw scores
   by 0.28–0.55 on average and up to 3.0 at the maximum, and changes the top-256
   *membership* by 7–14 % (Jaccard 0.86–0.93). The control demonstrably changes
   what the scorer computes.
2. **It changes the scores without changing which head tokens win.** The teacher's
   Top-8 tokens keep their relative positions under either image's context. That
   is the precise statement of "this information is not sample-specific for this
   task": the context modulates the read-out, but the modulation does not
   discriminate the tokens the teacher cares about.

The fit-split substitution agrees (0.7825–0.8067 vs honest 0.7625–0.8075), so the
conclusion is not an artifact of drawing the substitute from the same split.

---

## 5. Mechanism read-out: why the contextual arms degenerate

The controls say the context is not *used* sample-specifically. These measurements
say why — and rule out the trivial explanation that the contextual machinery was
switched off.

**SET-CTX is not collapsed.** Its token-token term is large:

| quantity | value |
|---|---|
| residual norm `‖a_i‖` | 18.64 |
| interaction term `‖Σ_j α_ij V_v a_j‖` | **0.681 × the residual** |
| attention entropy | 5.93 nats = **0.856 of log 1024** |
| **effective support** `exp(H)` | **534 of 1 024 tokens** |
| mean top-1 attention weight | 0.056 |
| term magnitude when the key/value set is another image's | **1.048 ×** its own |

The interaction term is 68 % of the residual it is added to, so the arm is doing
real work with it. But the attention is **nearly uniform over half the token set**,
with no token receiving more than ~6 % of the mass. A near-uniform average over
534 tokens of `V_v a_j` is a permutation-invariant pooled summary; the arm is
computing a *global average* through an attention-shaped implementation. That is
why it lands exactly where the pooling arm's information content puts it, and why
the ordering is SET-CTX ≈ LOCAL-MLP rather than SET-CTX ≫ LOCAL-MLP.

**GLOBAL-CTX's context is a large shared constant.**

| quantity | value |
|---|---|
| residual norm `‖a_i‖` | 16.40 |
| context term norm `‖V c‖` | 63.45 (≈ 4× the residual) |
| per-dimension spread of `V c` **across images** | 1.00 |
| **variation / residual** | **0.061** |
| fraction of context dims with any cross-image variance | 1.00 |

The context term is large — four times the residual it modulates — but its
image-to-image variation is 6 % of that residual. It is a big, nearly constant
vector, i.e. a shared bias. LOCAL-MLP already carries a learnable 128-dim bias of
its own, so GLOBAL-CTX is LOCAL-MLP plus a redundant constant plus an extra
gradient path — which is consistent with it landing *below* the local arm.

Neither arm is broken. Both are doing what they were built to do; the information
they were given access to is simply not where the head signal lives.

---

## 6. The six questions

**1. How far from the rank-256 frontier are the missed teacher-head tokens?**
Far, and not uniformly. The median missed Top-8 token has student rank **417**
(161 past the frontier); the p90 is 717. Only 13 % are within +32 places, 25 %
within +64, 42 % within +128. Per image, the worst miss has median rank 418 and
p90 771. A frontier-local architecture was therefore not built. The one piece of
good news for a pool design is coverage: 92.2 % of the head sits inside rank 512,
so a two-stage scorer would have a 0.92 ceiling — but §4 removes the reason to
build one.

**2. Is a nonlinear token-local function sufficient?**
**Yes.** LOCAL-MLP — no context, no query, no other token — takes held-out Top-8
recall from 0.7656 to **0.7972** (+0.0317 [+0.0189, +0.0444], positive in all
three seeds). And it is *sufficient*, not merely helpful: a 4.5× wider token-local
MLP gains nothing further (−0.0025 [−0.0147, +0.0094]), so ≈ 0.795 is the ceiling
of the entire token-local family, linear or not. The S2-C3 wall was **linearity**:
the head signal is nonlinear in the token's own hidden state, and a linear probe
underfits it.

**3. Is image-global context sufficient?**
**No.** GLOBAL-CTX, parameter-matched, is 0.7786 — **below** the local arm by
0.0186 [−0.0311, −0.0061] — and its CI against the reference includes zero
(+0.0131 [−0.0031, +0.0292], per-seed +0.034/+0.001/+0.004). A permutation-invariant
image summary is not the missing factor, and adding it actively hurts.

**4. Does genuine token-token / set interaction provide independent gain?**
**No.** SET-CTX is 0.7978, **+0.0006 [−0.0075, +0.0086]** over parameter-matched
LOCAL-MLP. It is not that the interaction term was disabled: it carries 68 % of
the residual, but with an effective support of 534/1024 tokens it is a diffuse
average, i.e. a permutation-invariant summary implemented as attention.

**5. Does wrong-image context eliminate the contextual gain?**
**There is no contextual gain to eliminate.** Honest minus wrong-image is
**+0.0012** for GLOBAL-CTX and **−0.0001** for SET-CTX (five derangements per seed;
fit-split substitution agrees). The substitution is not inert — it moves scores by
0.28–0.55 on average and changes 7–14 % of the top-256 membership — but it does
not change which head tokens win. So the control does not refute a positive
finding here; it *explains* the null one: the contextual arms were never reading
sample-specific token configuration.

**6. Which mechanism does the data support?**
**A. NONLINEAR-LOCAL.** The teacher's high-value head signal is a shared,
nonlinear, token-local function of the L4 hidden state. Context — image-global or
set-level — adds nothing. The pre-registered cascade returns `NONLINEAR-LOCAL`:
LOCAL-MLP clears the reference reproducibly and no contextual arm beats it by the
1 pt margin.

**Downstream gate: not met, no generation run.** Best capacity-matched arm is
SET-CTX at 0.7978; the gate requires ≥ 0.82 with a positive paired CI (aspirational
0.85). `deployable_gate.downstream_run = false`.

---

## 7. What this stage does and does not establish

**Established.** (i) The head signal is nonlinear in the token's own L4 state —
a per-token MLP recovers +3.2 pt of Top-8 recall over the linear probe, on all
three benchmarks, in all three seeds. (ii) That family is saturated at ≈ 0.795;
4.5× capacity adds nothing. (iii) Neither permutation-invariant image context nor
token-token interaction adds anything, and the wrong-image control plus the
attention-entropy measurement explain why rather than leaving it as a null.
(iv) The missed head tokens are not at the selection frontier, so frontier-local
architectures are unsupported by the geometry as well as by the ladder.

**Not established, and not attempted.** This stage says nothing about *what* the
nonlinear function computes, whether a better input representation (another layer,
multiple layers, pre-merger features) would help, or whether query conditioning
would help — the S2-C1 `QRY` arm (528 K parameters) measured it as a no-op against
`LIN` at the overlap-256 objective, and no query-conditioned arm was trained here.
It also does not attempt to close the remaining 0.795 → 0.95 gap, which by this
stage's evidence lies outside both the token-local factor and the context factor
as they were operationalized here. Per the brief, no final method is designed and
no scorer is proposed for deployment.

**One caveat on the size of the local gain.** The gain is +0.0317 with a CI of
[+0.0189, +0.0444], and the arms' seed sd is 0.010–0.014 — so the effect is real
but only ~2–3× the retraining noise. The reason it is nonetheless a firm result is
the reproducibility clause (all three seed-matched deltas positive in every arm
that clears) and the fact that the *same* ceiling is reached independently by
three different architectures (LOCAL-MLP 0.7972, SET-CTX 0.7978, WIDE 0.7947) —
which is what a genuine information ceiling looks like, as opposed to a lucky
optimization trajectory.

---

## 8. Reproduction

```bash
cd Qwen_vl && bash scripts/discovery/run_s2c5.sh      # full stage, ~40 min on one A40
```

Steps: `s2c5_c0_diagnosis.py` (CPU, ~1 min) → `s2c5_train.py` (12 runs, 3 seeds ×
4 arms) → `s2c5_controls.py` (wrong-image and mechanism controls) →
`s2c5_consolidate.py` (verdict) → `s2c5_figure.py`.

Frozen inputs (never recomputed): `s2c1_feats_L4.npy`, `s2b_gradient_scores.npz`,
`s2c1_scores.npz`, `s2c3_scores.npz`, `s2c3_headmetrics.npz`. The linear reference
is the cached S2-C3 `HEAD_RANK` artifact, not a retrained arm, and its per-image
metrics are read from `s2c3_headmetrics.npz` aligned **by key** (that file uses a
different key order from this stage's split enumeration; aligning by position
would silently shuffle the held-out set). The alignment reproduces S2-C3's
published per-seed recall exactly: 0.7608 / 0.7775 / 0.7583 against 0.761 / 0.777
/ 0.758.
