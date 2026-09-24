# M3-v0 — MissGuard: critical-miss correction on top of B2

**Stage:** M3, first method-building stage (the project leaves mechanism
discovery here). **Working name:** MissGuard. **Status:** pilot, held-out 150.

**The question this document answers, and the only one.** Can

> *keep B2's 256 tokens, find a few of the ones B2 dropped that the gradient
> teacher would have kept, and pay for them by evicting the most expendable
> tokens B2 kept*

be a real method — i.e. does the formulation have a ceiling worth chasing, and
can a forward-only student reach it? It does not attempt SOTA, does not touch
the base selector, and does not compare against HAWK / SCoRe / TransPrune.

**The answer, in one line.** **The formulation has a large, real, monotone
ceiling (+7.7 at r=8, +14.9 at r=32); the forward-only pre-LLM student reaches
none of it — its rescue sits at mean teacher rank 74–132 where the oracle's
sits at 3.5–15.5, and every learned arm is matched by a content-free random
rescue under the same eviction rule. Verdict: WEAK (brief §10).**

---

## 1. Exact method definition

```
S0  = B2(vis)                       base selection: EADP importance + block8 coverage greedy, |S0| = 256
D   = [1024] \ S0                   the 768 dropped tokens
R   = top_r  s_theta(i)             the audit: student scores every i in D, r best are rescued
E   = evict_rule(S0, r)             r retained tokens chosen for removal
S*  = (S0 \ E) ∪ R                  |S*| = 256 exactly, by construction
```

Nothing else changes. The pruned tensor handed to the decoder is `vis[S*]` in
ascending index order — the same shape, dtype and position policy (the pre-LLM
branch renumbers) as B2's own output.

Three properties are enforced by assertion, not by convention:

* `|S*| == 256` **and** `|unique(S*)| == 256` — a rescue that lands inside `S0`
  or an eviction that repeats an index raises instead of silently shrinking the
  budget;
* `r = 0` or `source = "none"` takes the incumbent's own `forward()` unchanged,
  so the identity arm is byte-identical to B2 by construction — and the gate
  checks that against the *stored* B2 predictions, not against a re-run;
* the audit runs inside the pruner call, i.e. inside the measured region, before
  any decoder layer sees the sequence.

## 2. Training labels

Teacher: the frozen **P1-G2** gradient saliency of S2-B
(`outputs/discovery/s2b_gradient_scores.npz`) — `score_i = Σ_d |∂J/∂v_i,d · v_i,d|`
with `J` the argmax logit at the first answer position. One prompt forward and
one backward per instance, offline, once.

| positive definition | what it is | positives per image |
|---|---|---|
| `strict` | teacher Top-64 \ S0 | 50.2 |
| `loose` | teacher Top-256 \ S0 | 196.8 |

The label tensor is built in the full 1024-token index space and sliced to the
768 dropped tokens at pooling time.

**Separation.** The teacher labels `fit` (240) and `val` (60) rows only. The
held-out 150 contributes no label to any training run; the trainer asserts the
split it reads, and model selection is on `val` alone. The teacher's only other
role is the **oracle** arm of §6, which is a deliberately-labelled ceiling
measurement, never a deployable path.

**How much is there to find.** On the held-out 150, B2's set contains only
**59.2 of the teacher's Top-256** (23 %) and holds **33.6 %** of that set's
teacher mass. The miss set `M = teacher Top-256 \ S0` averages **197 tokens per
image** — the correction is not short of candidates, it is short of a way to
rank them.

## 3. Inference features

All hand features are functions of quantities the incumbent's pre-LLM pruner
already computes, on the device, before any decoder layer runs. The first 18 are
the v0 set; the last 4 were added for the v1 follow-up of §8.8.

| group | features |
|---|---|
| EADP score | `imp` (the B2 score), `imp_pre` (pre-`^β`), `fused` (`α·global + (1-α)·local`), `glob`, `loc` |
| redundancy | `red_s0` (max cosine to a retained token), `red_all`, `nn_drop` (max cosine to another dropped token) |
| geometry | `x`, `y` (grid coords), `nb_mean`, `nb_max`, `nb_std`, `nb_mean_f` (3×3 neighbourhood of `imp`/`fused`) |
| vision-derived | `vis_norm`, `cos_s0c` (cosine to the `S0` centroid), `cos_top64c` (cosine to the top-64-by-`imp` centroid) |
| rank | `rank_imp` |
| *v1 only* | `loc_max`, `loc_top5`, `loc_std` (best single instruction-token match, before EADP's entropy filter and weighted mean collapse it) · `nb_cos` (mean cosine to the 3×3 neighbourhood in vision space) |

Plus the **vision feature itself**: the vision tower's own post-merger
(1024 × 4096) output — the tensor the incumbent already materialises and hands
to the pruner. This is the pre-LLM analogue of the token-local family S2-C5/S2-C6
studied on layer-4 hidden states, and it is the student's only non-scalar input.

**Banned and absent.** No L2/L4 hidden state, no decoder activation of any kind,
no backward pass at inference, no cached teacher score at inference, no
generation feedback. The feature builder reads only `TimedEADPPruner.last_gpu`,
which `_score` and `forward` fill from the vision tower's output and the
similarity kernel.

**Train/serve identity.** The bank stores the vision feature at fp16 and the
live path quantises to fp16 before use, so the student sees exactly the tensor
its standardiser was fitted on; the hand features come from *one* function
(`m3_common.handcrafted_t`) called by both the bank builder and the live pruner.
A bank records the feature *names* it holds, so appending to `FEATURES` cannot
silently reinterpret an older bank, and a checkpoint resolves its own feature
list to column indices so an older student keeps working.

## 4. Predictor architecture and parameters

```
score_i = f( [ GELU(W_v v_i + b_v) ; GELU(W_h x_i + b_h) ] )
W_v : 4096 → 64      W_h : 18 → 64      then 128 → 64 → 32 → 1, GELU
```

**273 793 parameters**, no attention, no context, no cross-token interaction —
every token is scored in isolation. `x` is standardised by statistics fitted on
the fit rows' dropped tokens; `v` by per-dimension statistics over the fit rows'
visual tokens (245 760 rows). An optional per-instance z-scoring of `x` was
tested and rejected (§5).

## 5. Training and model selection

Six configs, trained once each (`m3_train.py`, 120 epochs, AdamW, early stop on
the selection metric):

| config | val overlap@8/16/32 | val recall@16 |
|---|---|---|
| `strict` (Top-64), global z | 0.260 / **0.236** / 0.263 | 0.478 |
| `strict`, per-instance z | 0.258 / 0.229 / 0.248 | 0.455 |
| `loose` (Top-256), global z | 0.237 / 0.202 / 0.215 | 0.405 |
| `loose`, per-instance z | 0.248 / 0.198 / 0.221 | 0.417 |
| `head` (listwise), global z | 0.000 / 0.002 / 0.005 | 0.005 |
| `head`, per-instance z | 0.000 / 0.006 / 0.010 | 0.020 |

`overlap@r` = |student top-r ∩ oracle top-r| / r over the 768 dropped tokens;
**chance is 0.0208**. Selected: **`strict`, global z, val overlap@16 = 0.2365**.

Three things this table says, and one it refuses to.

* The strict positive definition wins on the metric that matters: a diffuse
  Top-256 target is dominated by easy mid-ranked tokens, exactly the failure
  mode S2-C5A caught when it found "GLOBAL-CTX below local" was a selection
  artefact.
* The **vision feature is worth ~+0.04 overlap** over the 17 hand features
  alone (0.2365 vs 0.202, same target and protocol). Real but modest.
* **The listwise head objective collapsed** (overlap ≈ 0.00–0.01, *below*
  chance). A geometric-decay target over 768 tokens with a temperature-1 softmax
  is far too diffuse to train against 240 images; recorded as a negative result,
  not tuned away. The v0 student is the BCE one.
* The table does **not** say the student is good. 0.2365 is 11× chance and finds
  ~3.8 of the oracle's 16 tokens. §8 shows what that is worth downstream.

**No architecture search.** Widths 64/32 and one linear view of the vision
feature were fixed before training; the six configs above are the whole grid.

## 6. Eviction rule and the oracle ceiling

```
lowimp   r retained tokens with the lowest EADP importance
maxred   r retained tokens with the largest cosine to the REST of S0
combo    z(imp) − λ·z(red), ascending (λ = 1)
teacher  lowest teacher score — NOT deployable; exists only for the double oracle
```

**Pre-registered primary arm: `MG-lowimp-r16`**, frozen before the grid ran, on
two grounds that do not depend on any accuracy number: `lowimp` is the cheapest
deployable rule (it reads the score the base selector already computed and never
touches the similarity submatrix), and the first pilot's oracle pair put
`lowimp` eviction **0.20 macro** from the *teacher* eviction oracle (73.28 vs
73.47) — the eviction axis is saturated there. Every other arm is secondary;
§11's verdict reads the primary.

**Oracle-Miss** replaces the student with the teacher: rescue = the r
highest-teacher-score dropped tokens, eviction rule identical to the arm it is
the ceiling for. **Double oracle** (`OR-teacher-r16`) additionally evicts by the
teacher, separating "what the miss predictor can add" from "what the eviction
rule costs".

## 7. Protocol, controls and gates

Held-out 150 (TextVQA 50 / DocVQA 50 / OCRBench 50), greedy, one generation per
instance, scored by `scoring.per_sample_hits`. Every arm runs through
`m2_gdep.GDEPEngine` in `prellm` mode with `block8` at T=256 — the same engine,
the same measured region, the same selector as the stored B0/B1/B2.

Controls, all of which move exactly r tokens and hold the budget at 256:

| control | rescue | eviction | what it bounds |
|---|---|---|---|
| `MG0` | — | — | the harness: must equal stored B2 hit-for-hit |
| `RND-<rule>-r<r>` | r random dropped tokens | same rule | everything that is *not* content: the position shift the swap induces, and the eviction damage |
| `OR-teacher-r16` | teacher top-r | teacher | the eviction half's cost |

Gates (`m3_accuracy._gates`), all three passing on the final grid:

* **`MG0` is B2** — 150/150 hit agreement *and* 150/150 prediction-string
  agreement against `m2_accuracy.json`'s stored B2 arm. `cfg_key`/`cfg_hash`
  cannot witness `alpha`, `beta`, `sim_mode` or `visual_token_num`, so this is
  the only gate that can.
* **live `S0` equals bank `S0`** — 1350/1350 instance checks across all
  non-identity arms.
* **oracle rescue is the teacher top-r** — 600/600 teacher-optimal. 16 of those
  are *tie-broken*: four instances have bit-identical teacher scores at the r-th
  boundary, so numpy's and torch's arg-sorts may pick either token. The gate
  compares the teacher-score multiset and records the tie count rather than
  pretending the two orders agree.
* **a crashed instance is not a wrong answer** — an arm with any raising
  instance is recorded as `error`, never as a low macro. (The first pilot
  persisted six crashed arms as 0.2 macro; fixed here.)

**Resolution, stated up front.** The paired macro SE on this 150 is ~2.5 points;
the minimum detectable effect at 80 % power is **7.0 points**. Every bar in §8
is therefore reported with its paired CI and a McNemar p on the discordant
pairs, and a point estimate below the MDE is never treated as evidence on its
own.

## 8. Results

### 8.1 The grid

Δ is the paired macro difference against B2; the CI resamples images within
benchmark (the project's standard paired bootstrap); *p* is McNemar on the
discordant pairs.

| arm | TextVQA | DocVQA | OCRBench | macro | Δ vs B2 | 95 % CI | *p* |
|---|---:|---:|---:|---:|---:|---|---:|
| B0 full 1024 | 73.60 | 85.27 | 68.00 | **75.62** | — | — | — |
| B1 official EADP facility | 59.40 | 61.89 | 62.00 | **61.10** | — | — | — |
| B2 block8 | 65.40 | 60.24 | 54.00 | **59.88** | — | — | — |
| MG0 (identity) | 65.40 | 60.24 | 54.00 | 59.88 | +0.00 | [0.0, 0.0] | 1.00 |
| **MG-lowimp-r16 (primary)** | 67.40 | 58.13 | 58.00 | **61.18** | **+1.28** | [−3.5, +6.2] | 1.00 |
| MG-maxred-r16 | 67.60 | 64.99 | 58.00 | 63.53 | +3.65 | [−1.5, +9.1] | 0.21 |
| MG-combo-r16 | 65.40 | 63.03 | 62.00 | 63.48 | +3.59 | [−1.3, +8.7] | 0.48 |
| MG-lowimp-r8 | 65.40 | 56.53 | 64.00 | 61.98 | +2.06 | [−2.5, +6.7] | 0.80 |
| MG-lowimp-r32 | 69.40 | 54.97 | 62.00 | 62.12 | +2.21 | [−3.5, +8.0] | 1.00 |
| MG-maxred-r8 | 70.00 | 60.93 | 64.00 | **64.98** | +5.06 | [+1.0, +9.5] | 0.077 |
| MG-maxred-r32 | 63.40 | 65.50 | 52.00 | 60.30 | +0.42 | [−5.5, +6.3] | 1.00 |
| RND-lowimp-r8 | 61.40 | 54.88 | 56.00 | 57.43 | −2.46 | [−6.9, +1.8] | 0.48 |
| RND-lowimp-r16 | 67.40 | 57.27 | 58.00 | 60.89 | +1.00 | [−4.5, +6.7] | 1.00 |
| RND-lowimp-r32 | 62.80 | 56.88 | 58.00 | 59.23 | −0.63 | [−6.7, +5.5] | 0.86 |
| RND-maxred-r8 | 67.40 | 60.05 | 58.00 | 61.82 | +1.94 | [−2.7, +6.7] | 0.42 |
| RND-maxred-r16 | 68.20 | 63.21 | 60.00 | 63.80 | +3.95 | [−1.9, +10.0] | 0.10 |
| RND-maxred-r32 | 63.60 | 65.20 | 52.00 | 60.27 | +0.43 | [−6.0, +6.8] | 0.74 |
| RND-combo-r8 | 61.40 | 58.72 | 58.00 | 59.37 | −0.53 | [−4.9, +3.8] | 1.00 |
| RND-combo-r16 | 67.40 | 58.46 | 62.00 | 62.62 | +2.75 | [−3.0, +8.6] | 0.25 |
| RND-combo-r32 | 65.00 | 66.01 | 62.00 | **64.34** | +4.49 | [−1.5, +10.5] | 0.08 |
| OR-lowimp-r8 | 69.40 | 73.53 | 60.00 | **67.64** | **+7.73** | [+2.2, +13.5] | 0.007 |
| OR-lowimp-r16 | 75.40 | 76.43 | 68.00 | **73.28** | **+13.42** | [+7.5, +19.5] | <0.0001 |
| OR-lowimp-r32 | 73.40 | 78.75 | 72.00 | **74.72** | **+14.87** | [+8.5, +21.6] | <0.0001 |
| OR-maxred-r16 | 73.40 | 79.70 | 66.00 | 73.03 | +13.17 | [+6.4, +20.3] | 0.0001 |
| OR-combo-r16 | 77.40 | 77.47 | 64.00 | 72.96 | +13.09 | [+7.0, +19.4] | <0.0001 |
| OR-teacher-r16 | 75.40 | 79.02 | 66.00 | 73.47 | +13.61 | [+7.6, +19.8] | <0.0001 |

### 8.2 The ceiling is real, large, and monotone in r

The oracle is the only family in this table whose CI excludes zero at every r,
whose p is below 0.01 at every r, and — the part no significance test replaces —
whose effect **grows monotonically with the number of swapped tokens**:

| r | oracle Δ | 95 % CI | rescued of B2's 53 wrong | broken |
|---|---|---|---|---|
| 8 | +7.73 | [+2.2, +13.5] | 16 | 5 |
| 16 | +13.42 | [+7.5, +19.5] | 22 | 4 |
| 32 | +14.87 | [+8.5, +21.6] | 24 | 5 |

At r=32 the oracle reaches **74.72**, within 0.9 points of the *unpruned* model
(75.62) and above the official facility selector by 13.6 — while keeping the
budget at 256 tokens and adding only the eviction's own cost. The eviction rule
is nearly irrelevant inside the oracle (73.28 / 73.03 / 72.96 for lowimp /
maxred / combo; the double oracle is 73.47), consistent with S2-C2's "the
removal half of a swap contributes almost nothing". **The formulation's ceiling
is not the question.**

### 8.3 The learned arm has no dose-response

| r | learned Δ (lowimp) | learned Δ (maxred) |
|---|---|---|
| 8 | +2.06 | +5.06 |
| 16 | +1.28 | +3.65 |
| 32 | +2.21 | +0.42 |

Where the oracle climbs by 7 points from r=8 to r=32, the learned arm is flat
(lowimp: 2.06 → 1.28 → 2.21) or backwards (maxred: 5.06 → 3.65 → 0.42). If the
student were finding valuable tokens, more of them would help more. It does not.

### 8.4 The content-free controls span the same range as the learned arms

This is the measurement that settles it. Every `RND` arm moves exactly r tokens
under exactly the same eviction rule; only the rescue is random.

| family | Δ vs B2, range | CI excluding zero |
|---|---|---|
| random rescue (9 arms) | **−2.46 … +4.49** | 0 of 9 |
| learned rescue (7 arms) | **+0.42 … +5.06** | 1 of 7 (`MG-maxred-r8`, p=0.077) |
| teacher rescue (6 arms) | **+7.73 … +14.87** | 6 of 6 |

The learned family sits entirely **inside** the random family's range. The one
learned arm whose CI excludes zero, `MG-maxred-r8` (+5.06 [+1.0,+9.5],
p=0.077), is the maximum of 7 learned arms, does not replicate at r=16 (+3.65)
or r=32 (+0.42) under its own rule, and its matched random control
(`RND-maxred-r8` = +1.94) removes most of it. It is reported because it exists,
not because it is a result.

### 8.5 The decomposition

Content = learned − random under the identical eviction rule and r:

| r | rule | learned Δ | random Δ | **content** |
|---|---|---|---|---|
| 8 | lowimp | +2.06 | −2.46 | +4.52 |
| 8 | maxred | +5.06 | +1.94 | +3.24 |
| 16 | lowimp | +1.28 | +1.00 | +0.28 |
| 16 | maxred | +3.65 | +3.95 | **−0.30** |
| 16 | combo | +3.59 | +2.75 | +0.84 |
| 32 | lowimp | +2.21 | −0.63 | +2.84 |
| 32 | maxred | +0.42 | +0.43 | **−0.01** |

Mean +1.63, SD 1.89 over seven non-independent arms (same student, nested rescue
sets, shared base) — a naive SE of 0.71 that no correction for the seven arms
survives. The content effect changes sign, has no r-dependence, and its largest
values come from the pair with the largest random-control noise. **The rescue
content contributes nothing measurable.**

### 8.6 Why: the student finds the teacher's middle, not its head

Teacher-mass `F(S)` is the share of the teacher's Top-256 mass that a set holds;
`rescue rank` is the mean rank of the rescued tokens *within the dropped set's
teacher ranking* (0 = the teacher's best dropped token).

| family | F(S) | mean rescue rank | fraction of rescue in teacher top-16 | measured macro |
|---|---|---|---|---|
| B2 (no swap) | 0.336 | — | — | 59.88 |
| random, all 9 arms | 0.324–0.341 | 377–384 | 0.019–0.026 | 57.43–64.34 |
| learned, r=8 | 0.408–0.417 | 73.7 | 0.383 | 61.98–64.98 |
| learned, r=16 | 0.423–0.437 | 108.2 | 0.238 | 61.18–63.53 |
| learned, r=32 | 0.455–0.471 | 131.9 | 0.160 | 60.30–62.12 |
| oracle, r=8 | 0.500 | **3.5** | **1.000** | 67.64 |
| oracle, r=16 | 0.563–0.594 | **7.5** | 0.998 | 72.96–73.47 |
| oracle, r=32 | 0.641 | **15.5** | 0.500 | 74.72 |

Three facts, in order of importance:

1. **The student is not random.** It gains real teacher mass (0.336 → 0.41–0.47)
   and 16–38 % of its rescue lands in the teacher's top-16 against a 2 % chance
   rate. The miss-prediction signal exists.
2. **It is 10–20× too deep.** Its rescue sits at mean teacher rank 74–132; the
   oracle's sits at 3.5–15.5. The formulation's value is concentrated at the very
   head — S2-C2 measured that 8 tokens close 53 % of the teacher gap and tokens
   past rank 32 are worth nothing — so rank-74 tokens are worth approximately
   nothing while rank-4 tokens are worth everything.
3. **Teacher mass is therefore not fungible.** The linear mass forecast
   (`61.10 + (F−F(B2))/(1−F(B2)) · 16.70`, calibrated on S2-B's two measured
   endpoints on these same instances) predicts 62.9–64.5 for the learned arms —
   which they roughly hit — and 65.2–68.8 for the oracle, which the oracle
   *exceeds by 2.4–6.5 points*. The link from mass to accuracy is convex, and
   convexity is exactly what makes head rank, not mass, the thing a student must
   predict.

### 8.7 Per-benchmark profile — no collapse

The brief asked whether MissGuard reproduces the GDEP failure profile
(TextVQA ↑, OCRBench ↓). It does not, on any learned arm:

| arm | TextVQA | DocVQA | OCRBench |
|---|---|---|---|
| B2 | 65.40 | 60.24 | 54.00 |
| MG-lowimp-r16 | **+2.00** | −2.11 | **+4.00** |
| MG-maxred-r16 | +2.20 | **+4.75** | +4.00 |
| MG-combo-r16 | 0.00 | +2.79 | **+8.00** |
| MG-maxred-r8 | **+4.60** | +0.69 | **+10.00** |
| OR-lowimp-r32 | +8.00 | +18.51 | +18.00 |

The learned arms lift TextVQA and OCRBench together, and the oracle lifts all
three by 8–19 points. Whatever is happening, it is not a benchmark-profile
collapse.

### 8.8 The v1 follow-up: more features, three seeds — and the resolution of the experiment

Four features were added (`loc_max`, `loc_top5`, `loc_std` — the best single
instruction-token match before EADP's entropy filter and weighted mean collapse
it — and `nb_cos`, the mean cosine to the 3×3 neighbourhood in vision space),
the bank was rebuilt, and three students were trained at seeds 0/1/2.

**The features changed nothing.**

| student | features | val overlap@16 |
|---|---|---|
| v0 | 18 | 0.2365 |
| v1 seed 0 | 22 | 0.2385 |
| v1 seed 1 | 22 | 0.2323 |
| v1 seed 2 | 22 | 0.2323 |

All four land within 0.006 of each other, and the `head` listwise objective
collapsed to 0.002–0.004 in every v1 run exactly as it did in v0. The audit
proposed that a student trained to the *head* might do better; it does not, and
neither do the four extra features. **The feature set is not the bottleneck.**

**And that gives the experiment its resolution.** Four independently trained
students of statistically identical head precision produce, on the
pre-registered primary arm, macros from **60.84 to 64.18**:

| arm | v0 | v1 s0 | v1 s1 | v1 s2 | seed mean | range |
|---|---:|---:|---:|---:|---:|---:|
| MG-lowimp-r16 (primary) | 61.18 | 60.84 | 62.73 | 64.18 | 62.23 | **3.34** |
| MG-maxred-r16 | 63.53 | 60.46 | 62.87 | 61.12 | 62.00 | **3.07** |

Not one of those eight runs has a CI excluding zero (every CI spans ±5 points),
and the seed swing of **3.3 macro points is larger than every learned-arm
effect measured in this document**. The v0 learned arms' spread of +1.28 to
+5.06 is what this swing looks like when it is read as a method effect.

The matched content-free controls at r=16 — `RND-lowimp-r16` +1.00,
`RND-maxred-r16` +3.95, `RND-combo-r16` +2.75, mean **+2.56** — sit on the same
number as the learned arms' seed means (+2.35 and +2.12). At the seed-mean level
the content effect is −0.2 to −0.3. It remains a null.

## 9. Latency

Measured with M2's paired protocol (`m2_perf_paired.py`, 10 warm-up + 60
measured blocks, every arm measured once per block in a randomised order on the
same fixed input, block-bootstrap CIs). 60/60 blocks retained;
`no_recompute` True in every block of every arm; `n_kept` = 256 for all three.

| arm | TTFT med | model-only | preprocessing | vision | EADP scoring | selector | **miss** | LLM prefill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B2 | 279.0 | 193.6 | 81.9 | 110.7 | 1.42 | 8.62 | 0.00 | 72.6 |
| **MG (learned r=16)** | **283.7** | 197.8 | 82.0 | 110.5 | 1.39 | 8.62 | **4.50** | 72.5 |
| OR (teacher r=16) | 280.2 | 194.0 | 81.5 | 110.5 | 1.38 | 8.62 | **0.84** | 72.6 |

Within-block paired contrasts:

| contrast | TTFT Δ (median) | mean CI | blocks | decode 32 Δ |
|---|---:|---|---|---:|
| **MG − B2** | **+4.81 ms** | **[+0.34, +4.96]** | 13 : 47 for B2 | +0.28 |
| OR − B2 | +1.53 ms | [−1.09, +2.19] | 21 : 39 for B2 | +1.33 |

**The audit costs +4.8 ms over B2**, i.e. 279.0 → 283.7 ms, or 1.7 % of TTFT —
inside the brief's 3–5 ms target, at the top of it. It decomposes cleanly: the
oracle's swap machinery alone is +0.84 ms (its paired median, +1.53 ms), so the
student's feature build plus MLP is **+3.66 ms**.

That is what the on-device design buys. The audit's first implementation moved
the 1024 × 4096 vision tensor to the host and ran numpy over it — measured at
24–148 ms per image depending on load, against B2's entire 8.5 ms selector.
Doing every reduction where the tensors already live, and letting only the r
chosen indices cross to the host, removed a factor of 5–30.

Two accounting notes, both the M2 amendment's lesson:

* the miss window opens **after** the base pass returns, so it contains no part
  of `selector_ms` or `eadp_scoring_ms`. Bracketing the base call as well would
  have nested one window inside another and made the stage sum exceed TTFT —
  the double-count artefact in mirror image. The residual
  `TTFT − Σ(stages)` is **+3.6 / +3.9 / +4.3 ms** for B2 / MG / OR, the same
  un-bracketed glue as every other arm in the project.
* no arm's `miss_ms` is read inside the timed region: the pruner hands over an
  unread CUDA-event pair and `prefill._finish()` reads it under the single sync
  the engine already performs.

Against the incumbents: MissGuard r=16 lands at **283.7 ms**, 26.4 ms faster
than B1 (310.1 ms) and 5.1 ms slower than B2 (278.6 ms). Memory is unchanged —
the audit allocates three 1024-vectors and one 1024 × 18 matrix, and the
compacted sequence is still 290 positions.

## 10. Examples

**What the oracle fixes** — near-miss OCR and partial reads, on all three
benchmarks:

| instance | B2 | Oracle r16 |
|---|---|---|
| TextVQA_VAL_3925 | `rebook` | `reebok` |
| TextVQA_VAL_2214 | `plsun` | `tecsun` |
| TextVQA_VAL_2818 | `dynasty warriors gundam` | `dynasty warriors gundam reborn` |
| TextVQA_VAL_1812 | `55%` | `80%` |
| DocVQA_VAL_538 | `New Family Practice` | `Annals Family Practice` |
| DocVQA_VAL_1723 | `Life Sciences Center` | `Bowman Technical Center` |

**What the learned arm fixes** — the same *kinds* of near-miss, but a smaller
and non-overlapping subset: `MG-maxred-r8` rescues TextVQA_VAL_2214, _2818,
_3925 and OCRBench_201 (`10556` → `1056`), and breaks TextVQA_VAL_4932
(`mastercard` → `asia big loyalty programme`) and DocVQA_VAL_4199.

**The tell.** `RND-combo-r32` — random rescue, no student at all — rescues
TextVQA_VAL_1812 (`55%` → `80%`), TextVQA_VAL_4831 (`15` → `8`) and
DocVQA_VAL_538 (`New Family Practice` → `Annals Family Practice`): *the same
instances the oracle fixes*. There is a set of flippable instances that any
budget-neutral perturbation of the token set can flip, and the learned arms are
drawing from that pool rather than from the teacher's head.

**Break cases.** Against B2's 53 wrong instances the oracle rescues 16–25 and
breaks 4–7 of B2's correct ones, on answers like `mastercard` → `airasia` and
`DOROTHY C. CLENNY` → `CARTER`; the learned arms rescue 9–13 and break 2–11, and
the random arms rescue 4–15 and break 5–14. No arm is break-free, and the
learned and random families overlap completely on both counts — which is why the
verdict rests on the paired CI and not on the rescue count.

## 11. Verdict and the six questions

**Verdict: WEAK** (brief §10), read on the pre-registered primary arm.

> `MG-lowimp-r16` = 61.18 macro, **+1.28** over B2, CI [−3.5, +6.2], McNemar
> p = 1.00, against a minimum detectable effect of 7.0. The oracle reaches
> **+14.87** (CI [+8.5, +21.6]). The formulation has headroom; the forward-only
> student does not reach it.

The controls sharpen this past what the gate alone says: the learned arms'
range is contained in the content-free random arms' range, the content effect
changes sign across rules, and the learned rescue sits at mean teacher rank
74–132 where the oracle's sits at 3.5–15.5. The student is above chance at
finding tokens the teacher *mildly* prefers and nowhere near finding the tokens
the teacher's value is *concentrated* in.

This is the third independent confirmation in this project that membership-level
selection from cheap forward-only features is dead — after S2-C2 ("no cheap
property identifies the valuable tokens") and S3-B ("membership-level selection
is dead, again"). M3 adds the part those two could not: **the ceiling of the
correction formulation, measured, and it is large.**

---

**Q1. Does the critical-miss correction formulation have a high oracle
ceiling?**
Yes, and it is the strongest result in this document. +7.73 at r=8, +13.42 at
r=16, +14.87 at r=32, all with CIs excluding zero and p ≤ 0.007, monotone in r,
reaching 74.72 macro at r=32 — within 0.9 of the unpruned model and 13.6 above
the official selector — at an unchanged 256-token budget. The eviction rule
barely matters inside the oracle (73.28 / 73.03 / 72.96 / 73.47 across four
rules), so the ceiling is bought almost entirely by the rescue.

**Q2. How much of the oracle gain does the forward-only student recover?**
Essentially none, and the honest estimate is a range rather than a number.
Against its matched content-free control the content effect is +4.52, +3.24,
+0.28, −0.30, +0.84, +2.84, −0.01 across seven (rule, r) cells — mean +1.63 with
a naive SE of 0.71 over non-independent arms, changing sign, with no
r-dependence. At the seed-mean level (§8.8) it is −0.2 to −0.3. Against the
oracle's +7.7 to +14.9, that is between 0 % and ~15 % of the ceiling, and not
distinguishable from zero at this n. The mechanism is in §8.6: the student finds
rank-74 tokens, the value lives at rank 4.

**Q3. What is the most effective r?**
For the oracle, larger is monotonically better: 8 → 16 → 32 gives +7.73 → +13.42
→ +14.87, with diminishing returns after 16. For the learned arm there is no
effective r: lowimp is flat (2.06 / 1.28 / 2.21) and maxred is backwards
(5.06 / 3.65 / 0.42), and the single best learned cell (maxred r=8) is the max
of seven arms and does not replicate. If the student is ever fixed, the oracle's
curve says r=16 is where the marginal token starts to pay less than it costs.

**Q4. Does MissGuard at least approach or beat official EADP accuracy?**
On point estimates, yes — every learned arm at r=16 and above clears B1's 61.10
(lowimp 61.18, maxred 63.53, combo 63.48; `MG-maxred-r8` reaches 64.98), and the
primary arm's four-student seed mean is **62.23** against B1's 61.10. But this
cannot be credited to the method: the same arm ranges over 60.84–64.18 across
four students of identical head precision (§8.8), the content-free controls
reach 63.80 and 64.34 under the same rules with a mean of +2.56 against the
learned arms' +2.12/+2.35, and no learned arm's CI excludes zero. The correct
statement is **"a budget-neutral swap of 16–32 tokens under a redundancy-aware
eviction rule matches B1 at B2's latency"** — a systems observation about the
eviction half, not evidence for MissGuard.

**Q5. How much extra computation does MissGuard cost over B2?**
**+4.81 ms of TTFT** (paired within-block median, mean CI [+0.34, +4.96], B2
wins 47 of 60 blocks), i.e. 279.0 → 283.7 ms. The oracle's swap machinery alone
is +1.53 ms, so the student's feature build plus MLP is +3.66 ms. That is inside
the brief's 3–5 ms target at the top of it, and it is 26.4 ms faster than B1.
The design that makes it possible: the whole audit — feature build, student and
eviction — runs on the device the tensors already live on, and only the r chosen
indices cross to the host. The first implementation, which copied the
1024 × 4096 vision tensor to the host and ran numpy over it, cost 24–148 ms per
image; that version was never run on the grid.

**Q6. Is MissGuard worth upgrading into a paper method?**
**Not in this form, and not by improving the predictor.** The formulation is
worth keeping — the ceiling is real, large, monotone, and cheap to reach if the
head can be found — but three independent attempts in this project have now
failed to find the teacher's head from cheap forward-only features, and M3 shows
*why*: the mass a cheap student can recover sits at rank 74–132 where it is
worth nothing. Four added features and three training seeds moved head precision
by 0.006 overlap points (§8.8), so the gap is not a feature-engineering problem.
Closing it needs either an inference-time signal that is not pre-LLM (which the
brief bans) or a different objective — predicting the head directly rather than
the ranking — and the second has now been tried twice (`head` in §5, and the
strict Top-64 target that won) without moving the number.

There is also a measurement problem standing in front of any upgrade: with four
students of identical head precision spanning 3.3 macro points, **n=150 cannot
resolve a +2 method effect at all** (MDE80 = 7.0). A next stage that wants to
claim anything about a miss predictor needs a larger held-out set before it
needs a better predictor.

The one thing M3-v0 does hand to a paper is the negative result itself, stated
quantitatively: **a 256-token set can be worth +14.9 macro over the incumbent
selector, and the entire difficulty is that the value is concentrated in the
first ~16 ranks of a signal no pre-LLM feature predicts.**

---

## Appendix — scripts, artefacts, and how to re-run

| file | what it is |
|---|---|
| `scripts/discovery/m3_common.py` | the method: features, student, eviction, `MissGuardPruner`, `install_missguard` |
| `scripts/discovery/m3_features.py` | one vision-tower forward per instance → the pre-LLM audit bank |
| `scripts/discovery/m3_train.py` | the 6-config student grid, selection on val overlap@16 |
| `scripts/discovery/m3_accuracy.py` | the held-out grid, the three gates, crash-is-not-a-wrong-answer |
| `scripts/discovery/m3_analyze.py` | tables, paired CIs, McNemar, teacher mass and rescue rank, the verdict |
| `scripts/discovery/m3_forecast.py` | pre-GPU forecast of the learned/oracle gain from the teacher cache |
| `scripts/discovery/m3_merge_arms.py` | copies student-independent arms between tags (provenance recorded) |
| `run_m3_grid.sh` / `_b.sh` / `_b2.sh` / `_b3.sh` | the v0 sessions |
| `run_m3_v1.sh` | re-extract + 3 seeds + v1 grid |
| `run_m3_perf.sh` | the paired latency measurement |

| artefact | contents |
|---|---|
| `m3_bank.npz` / `m3_bank_v1.npz` | 450 instances × (X, vis, s0, g2, split, feature_names) |
| `m3_miss.pt` / `m3_miss_v1_s{0,1,2}.pt` | the frozen students |
| `m3_accuracy.json` | the v0 grid (26 arms), gates, per-instance hits and predictions |
| `m3_accuracy_v1.json` | the v1 grid (seed axis), student-independent arms merged with provenance |
| `m3_analysis.json` / `m3_analysis_v1.json` | the derived tables and the verdict |
| `m3_perf_paired.json` | 60 paired blocks, per-arm stage decomposition, within-block contrasts |
| `m3_pilot1_evidence.json` | the pre-fix pilot's oracle pair (73.28 / 73.47), kept as the first ceiling measurement |

```bash
cd Qwen_vl && source ~/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
python scripts/discovery/m3_features.py --tag m3_bank          # ~10 min (model load dominates)
python scripts/discovery/m3_train.py                            # ~4 min
bash   scripts/discovery/run_m3_grid.sh                         # ~25 min
python scripts/discovery/m3_analyze.py
bash   scripts/discovery/run_m3_perf.sh                         # ~17 min
```

**Engine change made for M3, and its guard.** `m2_gdep.prefill._finish()` reads
the pruner's pending miss event pair under its own single sync and writes
`timings["miss_ms"]`; `m2_perf_paired.STAGE_KEYS` gained `miss_ms`. The change
touches no computation and no returned value — it only lets the miss stage
appear in the decomposition at all. `TimedEADPPruner` also gained an additive
`keep_gpu` flag (default off) that keeps the scoring intermediates on the device
instead of copying them to the host; with it off the class behaves exactly as
before. `dump_json` now writes through a temp file and `os.replace`, after a
crash mid-write destroyed 13 completed arms of session B.
