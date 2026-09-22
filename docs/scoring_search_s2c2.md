# S2-C2 — Disagreement Value Diagnosis

**Scope.** Stage-2 S2-C2. One question: *the LIN_L4 student recovers 56 % of the
P1-G2 teacher's Top-256 tokens but only 45 % of the teacher's downstream accuracy
gain — where does the missing value live?*

No scorer is trained, no MLP or multi-layer fusion is built, no loss is swept,
and no final method is designed. Every number below comes either from the S2-C1
caches or from re-running the **unmodified** generation harness on
budget-preserving counterfactual token sets.

**Status.** Complete. Scripts:
`Qwen_vl/scripts/discovery/s2c2_{common,decompose,swap,rescue,characterize,consolidate,figure}.py`.
Deliverable: `outputs/discovery/s2c2_diagnosis.json`.

---

## VERDICT

```
S2-C2 — the missing value is a handful of tokens at the TOP of the teacher's
ranking, not a diffuse tail, and not a redundancy problem.

Replacing 8 of the student's 256 tokens with the 8 highest-ranked tokens the
student missed recovers 53 % of the student->teacher accuracy gap; 16 tokens
recover 77 %; 32 tokens recover ~98 %. The token budget never changes.

The student is not wasting budget on redundant tokens: removing its weakest
picks and refilling with tokens from neither set does nothing (it is nominally
*worse*), and the choice of which student tokens to remove is statistically
indistinguishable from zero at every k, against a 9.4-point macro effect from
the choice of which to add. Swap in the teacher's *weakest* missed
tokens instead of its best and the effect vanishes completely (57.43 against
57.32 macro); swap in a real teacher map from the wrong image and it retains
7 % of the gap (58.01); sample this image's disagreement set at random and it
recovers 16 % of it (60.14, three seeds) where the teacher's ordering recovers
53 %. Only the teacher's own ordering on this image's own map reaches 66.75.

The rescue is collective but tiny: no single added token is individually
necessary in most instances (87 % of single removals leave the answer correct,
against an 8.2 % fragility baseline), yet the value needs the small group
together -- a median transition block of four tokens.

The tokens that carry the value are, on every cheap property measured --
student score, hidden-state norm, L2->L4 movement, feature redundancy, spatial
position, neighbour similarity, uniqueness -- indistinguishable from ordinary
tokens in the same disagreement set. Only the teacher's own gradient score
separates them.
```

---

## 0. Executive summary

1. **The 56 %/45 % framing is a currency error, and the exchange rate is
   steep.** Token overlap counts tokens; downstream value is not proportional to
   token count. On the held-out 150, swapping just **8** of the student's 256
   tokens recovers **53 %** of the student→teacher accuracy gap, **16** tokens
   recover **77 %**, and **32** recover **97 %**. The disagreement set it is
   drawn from averages **112** tokens. The marginal value of a swapped token
   peaks at **+1.31 macro points** for tokens 3–6 and has fallen to ~+0.22 by
   tokens 17–32 and to ~0 past token 32.

2. **The brief's "last small batch" question has the opposite answer.** It is
   not the last few missed tokens that carry disproportionate value — it is the
   **first** few. The last ~80 tokens of the disagreement carry nothing
   measurable at all; the curve is flat from k = 32 to k = 112.

3. **It is not a redundancy problem.** Removing the student's 8 weakest
   disagreement tokens and refilling the budget with tokens from neither set
   *lowers* accuracy (55.95 vs 57.32 macro). Removing the student's weakest `k`
   while adding the teacher's best `k` is statistically indistinguishable from
   removing `k` student tokens *at random* while adding the same `k`, at every
   `k` tested: Δ = −0.10 [−2.86, +2.67] at k = 8, −0.10 [−2.20, +2.00] at
   k = 16, +1.97 [−1.20, +5.34] at k = 24, +1.27 [−1.33, +4.14] at k = 40. The
   point estimates drift *in favour of random removal* at larger k, never the
   reverse, and never significantly. **The removal half of a swap contributes
   nothing measurable; the addition half is the entire effect.**

4. **The controls give a clean gradient, and the teacher's ordering is the
   dominant term.** At a fixed k = 8 on DocVQA: mirrored ordering 56.69 (=
   student), another image's teacher map 58.43, random 8 of this image's
   disagreement set 61.48 (three seeds, ±4.14), the teacher's highest-ranked 8
   **72.66**, teacher's own set 80.45. At k = 32 the whole ordering is
   **student 57.32 < adversarial 59.06 ≈ shuffled 61.47 < random 67.59 <
   teacher 74.69 < teacher's own set 75.15**. So the value needs (a) this
   image's teacher map and (b) the *top* of its ranking — and the ordering term
   is the **larger of the two on every benchmark** and at every k measured
   (Section 0.7). Every `random` figure in this report is a three-seed mean: at
   k = 8 the three seeds span ±1.06 macro, and quoting the first seed alone
   (61.59) would have made the control look stronger than it is.

5. **The same experiment in discrete form: swaps rescue answers.** 23 of the 28
   *student-wrong, teacher-correct* instances (82 %) are rescued by swapping
   ≤ 32 tokens; 12 of them by ≤ 8. The effect on the aggregate metrics is
   carried by these flips, not by a broad improvement — the per-instance median
   delta is 0.00 on DocVQA for every arm.

6. **No cheap property identifies the valuable tokens.** Within the same
   disagreement set, the tokens whose addition performs a rescue are
   indistinguishable from ordinary teacher-only tokens on student score,
   hidden-state norm at L2 and L4, L2→L4 movement, feature redundancy against
   the selected set or its neighbours, and position (Section 6). Two properties
   — uniqueness and grid row — come back marginally significant at |d| < 0.5 on
   26 instances out of fifteen tested, which is what multiple comparisons
   predicts; neither is used as a finding.
   Only the teacher's own gradient score separates them. A scorer cannot be
   taught to find these tokens from cheap features, because the cheap features
   do not contain them.

7. **Benchmarks differ in how much is missing, not in how it is missing.**
   Take the k = 8 swap and split what it recovers into "any sample of this
   image's teacher-disagreement tokens" (the `random` arm) and "the teacher's
   *ordering* within that set" (the `teacher` arm minus `random`), as a share of
   each benchmark's whole gap:

   | benchmark | gap | `random:8` (3 seeds) recovers | the teacher's ordering adds | still missing at k = 8 |
   |---|---|---|---|---|
   | TextVQA_VAL | 9.40 | 0 % | **+21 %** | 79 % |
   | DocVQA_VAL | 24.10 | 21 % | **+46 %** | 32 % |
   | OCRBench | 20.00 | 17 % | **+33 %** | 50 % |

   The ordering term is the larger one on **all three** benchmarks — infinitely
   so on TextVQA, and by roughly 2× on the other two. What differs between
   benchmarks is not the mechanism but the size of the prize and how much of the
   teacher's head the student misses: 1.56 of the teacher's top-8 tokens on
   TextVQA against 2.74 on DocVQA, which is why DocVQA has both the largest gap
   and the largest absolutely recoverable amount.

   This table is also where the run's single most important methodological point
   sits. On one seed, `random:8` gave DocVQA 67.18 and read as "45 % of the gap
   is distributional"; on three seeds it gives 61.48 ± 4.14 and reads as 21 %.
   **The random control's seed spread is comparable to several of the effects
   this stage is trying to measure**, which is why every claim above is quoted
   against a three-seed mean and why the k = 32 seed spread is reported
   explicitly in Section 4.2.

---

## 1. Fixed analysis objects

Everything in this stage is defined on the S2-C1 held-out 150 (`frozen_150[::3]`,
50 instances per benchmark), so the caches are reused rather than recomputed:

| object | definition | source |
|---|---|---|
| **T** | P1-G2 gradient teacher's Top-256 | `s2c0_pilot.json`, arm `P1G2` |
| **S** | LIN_L4 student's Top-256 | `s2c1_pilot.json`, arm `LIN_L4` |
| **C = T ∩ S** | shared | derived |
| **T_only = T − S** | what the student missed | derived |
| **S_only = S − T** | what the student spent instead | derived |

Because both sets have 256 members and share `C`, **`|T_only| == |S_only|`
identically** — the disagreement is symmetric, and it is that many tokens on
each side that a swap has to work with.

Both cached `select_idx` lists are stored in **descending score order** (the
Top-K operator returns `torch.topk(..., sorted=True)`), verified against the raw
score arrays on 60/60 instances. So "teacher rank order within `T_only`" and
"student rank order within `S_only`" need no re-sorting — the cached order *is*
the rank order.

Two harness-identity checks ran before any new generation:

* `identity_S` feeds the student's own set back through the swap harness and
  reproduces the cached student accuracies **exactly** (67.600 / 56.353 / 48.000).
* `identity_T` does the same for the teacher (77.000 / 80.450 / 68.000).

* The rescue refinement (Section 5) re-runs several of the same arms on a
  subset in a separate process, which gives a free **determinism check**: for
  every arm and benchmark that was run twice, `teacher:{2,8,16,32}` on
  TextVQA / DocVQA / OCRBench, the per-instance scores agree to
  **max |difference| = 0.000000** over 96 instance-runs. The harness is
  deterministic and the counterfactual sets — not decoding noise — are what
  move the answers.

Both identity arms delivered the intended token set on 150/150 instances
(`delivery_ok = 1.000` everywhere). The counterfactual machinery is therefore
exact, and every later number is comparable to S2-C1's.

---

## 2. Disagreement decomposition

### 2.1 Overlap is benchmark-dependent, and so is the gap

All accuracies here are the graded per-instance scores the harness reports
(TextVQA VQA-score ×100, DocVQA ANLS ×100, OCRBench % correct).

| benchmark | mean \|C\| of 256 | overlap | mean \|T_only\| | teacher acc | student acc | raw gap | EADP Top-K floor | S2-C1 prize |
|---|---|---|---|---|---|---|---|---|
| TextVQA_VAL | 154.78 | **0.605** | 101.2 | 77.000 | 67.600 | **9.40** | 41.400 | 35.60 |
| DocVQA_VAL | 129.06 | **0.504** | 126.9 | 80.450 | 56.353 | **24.10** | 42.860 | 37.59 |
| OCRBench | 147.64 | **0.577** | 108.4 | 68.000 | 48.000 | **20.00** | 44.000 | 24.00 |
| macro | 143.83 | **0.562** | 112.2 | 75.150 | 57.318 | **17.83** | 42.753 | 32.40 |

The student's Top-256 is much closer to the teacher on TextVQA (0.605) than on
DocVQA (0.504) — the two benchmarks the brief singles out. But the more useful
statement is the last two columns: the **raw student→teacher gap** is 9.4 points
on TextVQA and 24.1 on DocVQA, and the S2-C1 *retention* denominator (the
distance from EADP's own Top-K score up to the teacher) is much smaller on
TextVQA (35.6) than the raw gap suggests.

This matters for reading S2-C1's headline. Per benchmark, the student's
**retention of the S2-C1 prize** is very uneven:

| benchmark | retention of the EADP-prize denominator |
|---|---|
| TextVQA_VAL | **73.6 %** |
| DocVQA_VAL | **35.9 %** |
| OCRBench | **16.7 %** |

The macro number 45.0 % is the mean of three very different situations, and
TextVQA — where the student is already close — is the one distorting it upward.
(45.0 % is the ratio of the macro accuracies, not the mean of the three
per-benchmark ratios, which is 42.1 %; S2-C1 reports the former, and this report
reports both readings side by side.)

### 2.2 Where the correctness mass sits

Threshold 0.5 per instance, the repo-wide "hard error" convention.

| benchmark | student wrong, teacher correct | both correct | both wrong | student correct, teacher wrong |
|---|---|---|---|---|
| TextVQA_VAL | **5** | 33 | 12 | 0 |
| DocVQA_VAL | **11** | 30 | 7 | 2 |
| OCRBench | **12** | 22 | 14 | 2 |
| **total** | **28** | 85 | 33 | 4 |

28 of 150 instances are in the class the brief asks about
(*student wrong → teacher correct*), and they carry a mean of **121.4**
disagreement tokens each — so there is ample room for a swap to act.

### 2.3 Where the student's misses sit in the teacher's ranking

This is free from the caches, and it frames everything that follows. For each
position in the teacher's ranking, is that token in the student's Top-256?

| teacher rank band | TextVQA | DocVQA | OCRBench | all |
|---|---|---|---|---|
| 1–8 | 0.805 | 0.657 | 0.745 | **0.736** |
| 9–16 | 0.752 | 0.647 | 0.705 | 0.702 |
| 17–32 | 0.710 | 0.520 | 0.701 | 0.644 |
| 33–64 | 0.686 | 0.518 | 0.639 | 0.615 |
| 65–128 | 0.636 | 0.521 | 0.628 | 0.595 |
| 129–256 | 0.533 | 0.472 | 0.501 | 0.502 |
| overall | 0.605 | 0.504 | 0.577 | 0.562 |

Two readings, and the second is the important one:

* The student is *not* systematically failing at the top — it recovers 73.6 % of
  the teacher's best 8 tokens. The miss rate declines gently with rank.
* But because the student's cutoff is a hard Top-256, its misses are spread over
  the whole teacher ranking. The first 8 tokens of `T_only` (the tokens the
  teacher ranks highest among those the student missed) sit at **teacher ranks
  18.8 on average, worst case 30.6**. At k = 16 they sit at mean rank 30.9; at
  k = 32, mean rank 53.1.

So "swap in the first k missed tokens" is not a tail operation at all: it is
"recover the top of the teacher's ranking", which the student's 256-token
cutoff clipped.

### 2.4 The 56 % headline is a budget-boundary number, and the student is much weaker at the head

The S2-C1 headline — 0.562 Top-256 overlap — is measured *at the budget
boundary*. Read the same agreement at the head of the ranking and the picture
changes completely. Two different quantities, both worth reporting:

| k | student's top-k ∩ teacher's top-k, /k | teacher's top-k **kept anywhere** in the student's 256 | chance |
|---|---|---|---|
| 8 | **0.192** | **0.736** | 0.008 |
| 16 | 0.190 | 0.719 | 0.016 |
| 32 | 0.209 | 0.681 | 0.031 |
| 64 | 0.278 | 0.648 | 0.062 |
| 128 | 0.404 | 0.622 | 0.125 |
| 256 | **0.562** | 0.562 | 0.250 |

Read the two right-hand columns together, because they say different things:

* **The student already holds most of the teacher's best tokens.** It keeps
  73.6 % of the teacher's top-8 *somewhere* in its 256. The teacher's head is
  not being thrown away; it is being ranked too low to survive the cut on its
  own merits — but the cut is a Top-256, so being anywhere inside it counts.
* **But its own ranking disagrees sharply at the head.** Only 19.2 % of the
  teacher's top-8 appears in the student's top-8, against a 0.8 % chance rate.
  The student's ordering and the teacher's ordering are close to unrelated in
  the first few ranks. The two columns meet at k = 256 because there they
  describe the same set — a Top-256 agreement *is* a Top-256 recall — so the
  informative range is everything below the boundary. **The single number S2-C1
  reports is the one k at which the two readings are forced to coincide.**

Concretely, per image:

| benchmark | teacher's top-8 missed | teacher's top-32 missed |
|---|---|---|
| TextVQA_VAL | 1.56 of 8 | 8.18 of 32 |
| DocVQA_VAL | **2.74 of 8** | **13.24 of 32** |
| OCRBench | 2.04 of 8 | 9.18 of 32 |
| all | 2.11 of 8 | 10.20 of 32 |

**The student misses about two of the teacher's eight best tokens per image.**
Section 4 shows that putting those two — plus about six more from the teacher's
top-32 — back into the set, at no cost to the budget, recovers more than half of
the entire accuracy gap. That is the whole answer to the question this stage was
asked.

---

## 3. Budget-preserving counterfactual swaps

### 3.1 Design

The token budget is **exactly 256 in every arm**. A swap of size `k` adds the `k`
highest-ranked tokens of `T_only` and removes the `k` lowest-ranked tokens of
`S_only`, so the set is always

```
F(k) = (S \ S_only[m-k:]) ∪ T_only[:k]        m = |T_only| = |S_only|
```

`F(0)` is the student's set and `F(m)` is the teacher's, **by construction** —
so the curve's two endpoints are not fitted, they are identities, and both were
verified against the caches (Section 1). The whole family is a path between two
measured sets, and every point on it is a real 256-token set that was actually
run through the unmodified generation path.

**Orderings are the experiment.** Because `F(k)` depends on *which* tokens enter
and leave, four orderings were run:

| arm | add from `T_only` | remove from `S_only` | what it isolates |
|---|---|---|---|
| `teacher:k` | descending teacher rank | ascending student rank (worst first) | the teacher's own priority |
| `random:k:s{0,1,2}` | uniform, seeded | uniform, seeded | matched overlap trajectory, random identity |
| `adversarial:k` | **ascending** teacher rank (worst first) | **descending** student rank (best first) | the exact mirror ordering |
| `shuffled:k` | another image's teacher map, ranked on this image | ascending student rank | same value distribution, wrong content |

`shuffled` is the content-free control in the sense S2-B and S2-C0 used: the map
is a real P1-G2 map, with the real marginal distribution, applied to the wrong
image. It fixes both the *number* of tokens swapped and the *shape* of the
teacher's score distribution while destroying the correspondence to this image.

### 3.2 Isolating the two halves of a swap

`teacher:k` changes two things at once — it adds the teacher's best tokens *and*
removes the student's worst. Two further arms split it:

| arm | add | remove |
|---|---|---|
| `addteacher:k` | teacher's best `k` | `k` **random** `S_only` tokens |
| `remworst:k` | `k` tokens from **neither** T nor S | student's worst `k` |

Together with `teacher:k` (= both extremes) and `random:k` (= neither), these
four arms form a 2×2 whose marginals answer the brief's mechanisms **A** (a few
tokens carry the value) and **B** (the student is wasting budget on tokens that
are not just useless but displaceable) separately, instead of conflating them.

### 3.3 Recovery, under two denominators

Two normalisations are reported, because they answer different questions:

* **raw gap** `(acc(k) − acc_student) / (acc_teacher − acc_student)`, the
  brief's "recovery relative to student→teacher gap", which is the natural scale
  for a family of sets interpolating between S and T;
* **EADP prize** `(acc(k) − acc_EADP-TopK) / (acc_teacher − acc_EADP-TopK)`,
  S2-C1's retention denominator, included so the numbers are directly comparable
  to its 45.0 %.

---

## 4. Results — the swap curves

### 4.1 Reading the curves

Each panel below is a single 256-token set per instance per point, run through
the unmodified generation path. The left edge is the student's own set and the
right edge is the teacher's own set; both were measured by the identity arms, so
the curve is anchored at both ends by construction rather than by fit.

**The full grid.** `k` is the number of tokens swapped; every row is a separate
measured 256-token set. `0 (=S)` is the student's own set and `full (=T)` the
teacher's, both measured through this harness.

| k | TextVQA_VAL | DocVQA_VAL | OCRBench | macro |
|---|---|---|---|---|
| 0 (=S) | 67.60 | 56.35 | 48.00 | 57.32 |

**teacher-priority**

| k | TextVQA_VAL | DocVQA_VAL | OCRBench | macro |
|---|---|---|---|---|
| 2 | 67.60 | 59.85 | 50.00 | 59.15 |
| 6 | 69.60 | 73.53 | 50.00 | 64.38 |
| 8 | 69.60 | 72.66 | 58.00 | 66.75 |
| 12 | 73.60 | 76.26 | 58.00 | 69.29 |
| 16 | 73.60 | 77.55 | 62.00 | 71.05 |
| 24 | 74.20 | 78.06 | 66.00 | 72.75 |
| 32 | 78.20 | 79.88 | 66.00 | 74.69 |
| 40 | 78.20 | 78.56 | 66.00 | 74.25 |
| 48 | 77.60 | 79.63 | 70.00 | 75.74 |
| 64 | 79.60 | 79.61 | 62.00 | 73.74 |
| 80 | 77.60 | 77.95 | 66.00 | 73.85 |
| 96 | 75.60 | 81.68 | 70.00 | 75.76 |

**random (3 seeds)**

| k | TextVQA_VAL | DocVQA_VAL | OCRBench | macro |
|---|---|---|---|---|
| 2 | 67.60 (1 seed) | 58.60 (1 seed) | 48.00 (1 seed) | 58.07 (1 seed) |
| 6 | — | — | — | — |
| 8 | 67.60 ± 0.00 | 61.48 ± 4.14 | 51.33 ± 1.89 | 60.14 ± 1.06 |
| 12 | — | — | — | — |
| 16 | 69.93 ± 2.16 | 63.40 ± 3.69 | 54.00 ± 4.90 | 62.44 ± 3.46 |
| 24 | — | — | — | — |
| 32 | 72.27 ± 3.40 | 71.17 ± 4.61 | 59.33 ± 3.40 | 67.59 ± 3.04 |
| 40 | — | — | — | — |
| 48 | — | — | — | — |
| 64 | — | — | — | — |
| 80 | — | — | — | — |
| 96 | — | — | — | — |

**adversarial**

| k | TextVQA_VAL | DocVQA_VAL | OCRBench | macro |
|---|---|---|---|---|
| 2 | — | — | — | — |
| 6 | — | — | — | — |
| 8 | 67.60 | 56.69 | 48.00 | 57.43 |
| 12 | — | — | — | — |
| 16 | 68.80 | 58.98 | 50.00 | 59.26 |
| 24 | — | — | — | — |
| 32 | 66.80 | 56.39 | 54.00 | 59.06 |
| 40 | — | — | — | — |
| 48 | — | — | — | — |
| 64 | — | — | — | — |
| 80 | — | — | — | — |
| 96 | — | — | — | — |

**shuffled**

| k | TextVQA_VAL | DocVQA_VAL | OCRBench | macro |
|---|---|---|---|---|
| 2 | — | — | — | — |
| 6 | — | — | — | — |
| 8 | 67.60 | 58.43 | 48.00 | 58.01 |
| 12 | — | — | — | — |
| 16 | 69.60 | 59.44 | 48.00 | 59.01 |
| 24 | — | — | — | — |
| 32 | 69.60 | 62.82 | 52.00 | 61.47 |
| 40 | — | — | — | — |
| 48 | — | — | — | — |
| 64 | — | — | — | — |
| 80 | — | — | — | — |
| 96 | — | — | — | — |
| full (=T, measured) | 77.00 | 80.45 | 68.00 | 75.15 |

**The marginal value is front-loaded, by two orders of magnitude.** Reading the
macro curve as a derivative — accuracy gained per token swapped — is the
sharpest statement of the result:

| swap range | tokens | macro | gain | per token | cumulative gap |
|---|---|---|---|---|---|
| 0 → 2 | 2 | 57.32 → 59.15 | +1.83 | +0.917 | 10.3 % |
| 2 → 6 | 4 | 59.15 → 64.38 | +5.22 | +1.306 | 39.6 % |
| 6 → 8 | 2 | 64.38 → 66.75 | +2.38 | +1.188 | 52.9 % |
| 8 → 12 | 4 | 66.75 → 69.29 | +2.54 | +0.634 | 67.1 % |
| 12 → 16 | 4 | 69.29 → 71.05 | +1.76 | +0.441 | 77.0 % |
| 16 → 24 | 8 | 71.05 → 72.75 | +1.70 | +0.213 | 86.6 % |
| 24 → 32 | 8 | 72.75 → 74.69 | +1.94 | +0.242 | 97.4 % |
| 32 → 40 | 8 | 74.69 → 74.25 | -0.44 | -0.055 | 95.0 % |
| 40 → 48 | 8 | 74.25 → 75.74 | +1.49 | +0.186 | 103.3 % |
| 48 → 64 | 16 | 75.74 → 73.74 | -2.01 | -0.125 | 92.1 % |
| 64 → 80 | 16 | 73.74 → 73.85 | +0.11 | +0.007 | 92.7 % |
| 80 → 96 | 16 | 73.85 → 75.76 | +1.91 | +0.119 | 103.4 % |
| 96 → 112 | 16 | 75.76 → 75.15 | -0.61 | -0.038 | 100.0 % |

The value per token **peaks in the first half-dozen** and decays monotonically:
by tokens 17–32 a swapped token is worth about a sixth of what the first tokens
were worth, and past token 32 the curve is flat and the sign of the change is
noise. Cumulatively: 39.6 % of the gap is closed by token 6, 52.9 % by token 8,
77.0 % by token 16, and 97.4 % by token 32 — out of an average disagreement of
112 tokens.
The brief's question — *is it the last small batch of missed tokens that carries
a disproportionate share of the value?* — has a clean answer: **no; it is the
first small batch.** The last ~80 tokens of the disagreement carry nothing
measurable at all.

The apparent non-monotonicity at k = 64 (a −2.01 dip) is within noise on n = 50
per benchmark; the trend is a plateau, not a peak at k = 48.

Put as the tokens needed to reach a given fraction of each benchmark's own gap,
linearly interpolated between measured points:

**tokens needed for 25/50/75 % recovery**

| benchmark | \|T_only\| | 25 % | 50 % | 75 % | 100 % |
|---|---|---|---|---|---|
| TextVQA_VAL | 101 | 8.3 | 10.7 | 24.9 | 101 |
| DocVQA_VAL | 127 | 2.7 | 4.5 | 10.0 | 127 |
| OCRBench | 108 | 6.7 | 12.0 | 18.0 | 108 |


### 4.2 The 2x2 that separates "the teacher's tokens" from "the student's waste"

**The add/remove split**, macro accuracy at every k both halves were measured:

| k | teacher:k (add best + drop worst) | addteacher:k (add best + drop random) | remworst:k (drop worst + add neutral) | random:k:s0 (both random) |
|---|---|---|---|---|
| 8 | 66.75 | 66.65 | 55.95 | 61.59 |
| 16 | 71.05 | 70.95 | 56.10 | 63.09 |
| 24 | 72.75 | 74.72 | 57.14 | — |
| 40 | 74.25 | 75.53 | 58.57 | — |

Every arm below moves **exactly 8 tokens** and holds the budget at 256.

| arm at k = 8 | TextVQA | DocVQA | OCRBench | macro | added | removed |
|---|---|---|---|---|---|---|
| student set (no swap) | 67.60 | 56.35 | 48.00 | 57.32 | — | — |
| `adversarial:8` | 67.60 | 56.69 | 48.00 | **57.43** | teacher's *weakest* 8 of `T_only` | student's *best* 8 of `S_only` |
| `remworst:8` | 65.60 | 56.25 | 46.00 | **55.95** | 8 tokens from neither T nor S | student's weakest 8 of `S_only` |
| `shuffled:8` | 67.60 | 58.43 | 48.00 | **58.01** | 8 tokens from *another image's* teacher map | student's weakest 8 of `S_only` |
| `random:8` (3 seeds) | 67.60 ± 0.00 | 61.48 ± 4.14 | 51.33 ± 1.89 | **60.14 ± 1.06** | 8 random tokens of `T_only` | 8 random of `S_only` |
| `teacher:8` | 69.60 | 72.66 | 58.00 | **66.75** | teacher's *best* 8 of `T_only` | student's weakest 8 of `S_only` |
| `addteacher:8` | 69.60 | 74.36 | 56.00 | **66.65** | teacher's *best* 8 of `T_only` | 8 random of `S_only` |
| teacher's own set (full swap) | 77.00 | 80.45 | 68.00 | 75.15 | — | — |

Read down that column and it is a clean gradient, on every benchmark:

* **The mirrored ordering is a perfect null.** `adversarial:8` lands on the
  student to two decimals on TextVQA and OCRBench and within 0.34 ANLS on
  DocVQA. It is not "a swap of 8 tokens" that helps; it is which 8.
* **Removing the student's tokens does essentially nothing on its own.**
  `remworst` drops the student's weakest disagreement tokens and refills from
  neither set. At k = 8, 16 and 24 it lands at or *below* the student (55.95,
  56.10, 57.14 against 57.32 macro), and only at k = 40 does it clear the
  student — by **1.25 points**, against the +16.94 the teacher's ordering buys at
  the same k. Whatever the student is spending its budget on, it is not waste
  that can be swept away.
* **The content has to be this image's.** `shuffled:8` uses a real P1-G2 map
  with the correct marginal distribution, applied to the wrong image: it gains
  +2.1 ANLS on DocVQA where the true map gains +16.3.
* **The teacher's ranking is the dominant term.** Within this image's own
  disagreement set, choosing the 8 tokens *at random* recovers less than a third
  of the DocVQA effect (+5.13 of +16.30 on three seeds; the single seed quoted
  above would have said two thirds), and taking the teacher's *highest-ranked* 8
  recovers all of it. Taking its *lowest-ranked* 8 recovers none. So both the
  membership of the disagreement set and the order inside it carry value — and
  the order carries the larger share, on every benchmark.

**The ordering survives at k = 16.** The same four orderings, doubled:

| macro | k = 8 | k = 16 |
|---|---|---|
| student set | 57.32 | 57.32 |
| `remworst` | 55.95 | — |
| `adversarial` | 57.43 | 59.26 |
| `shuffled` | 58.01 | 59.01 |
| `random` (3 seeds) | 60.14 ± 1.06 | 62.44 ± 3.46 |
| `addteacher` | 66.65 | 70.95 |
| `teacher` | 66.75 | 71.05 |
| teacher's own set | 75.15 | 75.15 |

The control ordering is identical at both sizes, and the gap the teacher's
ordering opens is 6.6 macro points at k = 8 and 8.6 at k = 16 — it does not
close as the swap grows. Per-benchmark, with the random arm's seed spread:

| k = 32 | TextVQA | DocVQA | OCRBench | macro |
|---|---|---|---|---|
| student set | 67.60 | 56.35 | 48.00 | 57.32 |
| `adversarial:32` | — | — | — | 59.06 |
| `shuffled:32` | — | — | — | 61.47 |
| `random:32` (3 seeds) | 72.27 ± 3.40 | 71.17 ± 4.61 | 59.33 ± 3.40 | 67.59 ± 3.04 |
| `teacher:32` | 78.20 | 79.88 | 66.00 | **74.69** |
| teacher's own set | 77.00 | 80.45 | 68.00 | 75.15 |

The full k = 32 ordering, macro: **student 57.32 < adversarial 59.06 ≈ shuffled
61.47 < random 67.59 < teacher 74.69 < teacher's own set 75.15** — the same
ordering as k = 8 and k = 16, with every control separated from every other.

At k = 32 a *random* sample of the teacher's disagreement set recovers 58 % of
the macro gap, against the teacher ordering's 97 % — so the two terms both grow
with k, and the ordering term stays ahead by 5–9 macro points throughout. A
single lucky seed at k = 32 (71.76 macro) lands close to the teacher arm; the
three-seed spread shows why one draw is not evidence of catching up.

**The addition half is the whole effect.** `teacher:k` and `addteacher:k` add
the identical `k` tokens and differ only in which student tokens come out
(the student's weakest, versus `k` at random). Measured against each other, the
difference is never significant:

| k | `teacher:k` | `addteacher:k` | Δ | 95 % CI |
|---|---|---|---|---|
| 8 | 66.75 | 66.65 | −0.10 | [−2.86, +2.67] |
| 16 | 71.05 | 70.95 | −0.10 | [−2.20, +2.00] |
| 24 | 72.75 | 74.72 | **+1.97** | [−1.20, +5.34] |
| 40 | 74.25 | 75.53 | **+1.27** | [−1.33, +4.14] |

The point estimates are worth reading rather than rounding away: at k = 8 and
k = 16 the two are identical to 0.10 macro points, and at k = 24 and k = 40 the
sign flips *against* the deliberate removal — discarding the student's weakest
tokens by rank is, if anything, slightly worse than discarding them at random.
Combined with `remworst:8` landing below the student, the conclusion is not
merely that the removal choice is uninformative but that **nothing in the
student's set needs removing**: the whole 9.4-point macro gain lives in what is
added.

### 4.3 The effect is carried by answer flips on a minority of instances

A mean can hide the shape of an effect, so the per-instance flips are counted
explicitly. On DocVQA the per-instance **median** delta is 0.00 for every arm
above — the mean moves because a minority of instances crosses the correctness
threshold. "any-change" counts instances whose graded score moved at all, so a
single instance can move a long way:

| arm at k = 8 | DocVQA per-instance median Δ | DocVQA any-change / worse | flips wrong→right (150) | flips right→wrong (150) |
|---|---|---|---|---|
| `teacher:8` | +0.00 | 18 / 2 | 15 | 3 |
| `addteacher:8` | +0.00 | 19 / 2 | 16 | 4 |
| `random:8` | +0.00 | 15 / 5 | 10 | 4 |
| `adversarial:8` | +0.00 | 1 / 3 | 1 | 1 |

This is why the rescue analysis in Section 5 is reported on a discrete
correctness criterion as well as on the graded score: on these benchmarks the
mechanism of the gain is *answers being rescued*, not a broad improvement.

### 4.4 Outcome counts, not just means

**per-instance outcomes vs the student set (150 instances)**

| arm | macro | Δ | improved | tied | worsened |
|---|---|---|---|---|---|
| `F_rev:16` | 75.17 | +17.85 | 39 | 107 | 4 |
| `F_rev:32` | 74.50 | +17.18 | 39 | 106 | 5 |
| `F_rev:64` | 72.64 | +15.33 | 35 | 111 | 4 |
| `F_rev:96` | 72.42 | +15.10 | 33 | 115 | 2 |
| `addteacher:8:s0` | 66.65 | +9.34 | 27 | 118 | 5 |
| `addteacher:16:s0` | 70.95 | +13.63 | 33 | 113 | 4 |
| `addteacher:24:s0` | 74.72 | +17.40 | 38 | 108 | 4 |
| `addteacher:40:s0` | 75.53 | +18.21 | 40 | 105 | 5 |
| `adversarial:8` | 57.43 | +0.11 | 1 | 146 | 3 |
| `adversarial:16` | 59.26 | +1.94 | 5 | 141 | 4 |
| `adversarial:32` | 59.06 | +1.75 | 8 | 136 | 6 |
| `remworst:8:s0` | 55.95 | -1.37 | 1 | 145 | 4 |
| `remworst:16:s0` | 56.10 | -1.21 | 3 | 141 | 6 |
| `remworst:24:s0` | 57.14 | -0.18 | 2 | 146 | 2 |
| `remworst:40:s0` | 58.57 | +1.25 | 6 | 141 | 3 |
| `shuffled:8` | 58.01 | +0.69 | 3 | 146 | 1 |
| `shuffled:16` | 59.01 | +1.70 | 6 | 143 | 1 |
| `shuffled:32` | 61.47 | +4.16 | 11 | 137 | 2 |
| `teacher:2` | 59.15 | +1.83 | 9 | 136 | 5 |
| `teacher:6` | 64.38 | +7.06 | 21 | 126 | 3 |
| `teacher:8` | 66.75 | +9.43 | 26 | 120 | 4 |
| `teacher:12` | 69.29 | +11.97 | 32 | 114 | 4 |
| `teacher:16` | 71.05 | +13.73 | 33 | 113 | 4 |
| `teacher:24` | 72.75 | +15.44 | 35 | 112 | 3 |
| `teacher:32` | 74.69 | +17.38 | 39 | 105 | 6 |
| `teacher:40` | 74.25 | +16.94 | 38 | 107 | 5 |
| `teacher:48` | 75.74 | +18.43 | 40 | 106 | 4 |
| `teacher:64` | 73.74 | +16.42 | 37 | 108 | 5 |
| `teacher:80` | 73.85 | +16.53 | 37 | 108 | 5 |
| `teacher:96` | 75.76 | +18.44 | 38 | 109 | 3 |

**answer flips vs the student set**

| arm | wrong->right | right->wrong | TextVQA_VAL w→r | DocVQA_VAL w→r | OCRBench w→r |
|---|---|---|---|---|---|
| `F_rev:16` | 28 | 4 | 4 | 12 | 12 |
| `F_rev:32` | 27 | 4 | 4 | 12 | 11 |
| `F_rev:64` | 24 | 4 | 4 | 11 | 9 |
| `F_rev:96` | 24 | 2 | 2 | 12 | 10 |
| `addteacher:8:s0` | 16 | 4 | 2 | 8 | 6 |
| `addteacher:16:s0` | 23 | 4 | 3 | 12 | 8 |
| `addteacher:24:s0` | 28 | 4 | 6 | 11 | 11 |
| `addteacher:40:s0` | 30 | 5 | 6 | 12 | 12 |
| `adversarial:8` | 1 | 1 | 0 | 1 | 0 |
| `adversarial:16` | 4 | 0 | 1 | 2 | 1 |
| `adversarial:32` | 7 | 3 | 1 | 3 | 3 |
| `identity_S` | 0 | 0 | 0 | 0 | 0 |
| `identity_T` | 28 | 4 | 5 | 11 | 12 |
| `random:2:s0` | 2 | 1 | 0 | 2 | 0 |
| `random:4:s0` | 4 | 3 | 1 | 3 | 0 |
| `random:8:s0` | 10 | 4 | 1 | 7 | 2 |
| `random:8:s1` | 5 | 2 | 0 | 3 | 2 |
| `random:8:s2` | 7 | 3 | 1 | 2 | 4 |
| `random:16:s0` | 12 | 3 | 1 | 6 | 5 |
| `random:16:s1` | 4 | 5 | 1 | 1 | 2 |
| `random:16:s2` | 15 | 2 | 3 | 6 | 6 |
| `random:32:s0` | 25 | 3 | 3 | 13 | 9 |
| `random:32:s1` | 16 | 4 | 5 | 6 | 5 |
| `random:32:s2` | 13 | 4 | 1 | 6 | 6 |
| `remworst:8:s0` | 1 | 3 | 0 | 1 | 0 |
| `remworst:16:s0` | 2 | 4 | 1 | 1 | 0 |
| `remworst:24:s0` | 1 | 2 | 0 | 1 | 0 |
| `remworst:40:s0` | 4 | 2 | 1 | 2 | 1 |
| `shuffled:8` | 2 | 1 | 0 | 2 | 0 |
| `shuffled:16` | 3 | 1 | 1 | 2 | 0 |
| `shuffled:32` | 8 | 2 | 1 | 4 | 3 |
| `teacher:2` | 6 | 3 | 1 | 3 | 2 |
| `teacher:6` | 12 | 3 | 1 | 8 | 3 |
| `teacher:8` | 15 | 3 | 2 | 7 | 6 |
| `teacher:12` | 20 | 4 | 3 | 10 | 7 |
| `teacher:16` | 23 | 4 | 3 | 11 | 9 |
| `teacher:24` | 24 | 3 | 3 | 11 | 10 |
| `teacher:32` | 29 | 5 | 5 | 13 | 11 |
| `teacher:40` | 28 | 5 | 5 | 12 | 11 |
| `teacher:48` | 29 | 4 | 5 | 12 | 12 |
| `teacher:64` | 27 | 5 | 6 | 11 | 10 |
| `teacher:80` | 27 | 5 | 5 | 11 | 11 |
| `teacher:96` | 28 | 3 | 4 | 12 | 12 |

**Per benchmark.** The same comparison split three ways. Read these as
descriptive only: the class-1 instances number 5 / 11 / 12 per benchmark, so the
per-benchmark cells rest on 4–11 instances and the significance flags below are
not trustworthy at that n — they are reported because the brief asks for the
split, not because they settle anything. The only pattern that survives is the
same one as the pooled table: `g2` and `g2_pct` separate, and everything else is
either null or marginal-and-inconsistent across benchmarks.

**TextVQA_VAL** — A2 (rescue-performing) vs B (ordinary teacher-only): n = 4 instances, significant: g2, g2_pct, lin, uniq
  - `g2` 9.1876 vs 3.1932 (d = +2.79, CI [+3.0797, +9.0944])
  - `g2_pct` 0.9744 vs 0.8679 (d = +5.31, CI [+0.0752, +0.1276])
  - `lin` -0.1068 vs -0.2991 (d = +0.74, CI [+0.0022, +0.5449])
  - `uniq` 0.8619 vs 0.8789 (d = -1.53, CI [-0.0309, -0.0083])

**DocVQA_VAL** — A2 (rescue-performing) vs B (ordinary teacher-only): n = 11 instances, significant: g2, g2_pct, n_l2, row, uniq
  - `g2` 10.2687 vs 2.8257 (d = +2.20, CI [+4.9622, +9.9463])
  - `g2_pct` 0.9805 vs 0.8726 (d = +6.45, CI [+0.0913, +0.1186])
  - `n_l2` 32.8017 vs 34.0338 (d = -0.37, CI [-2.3132, -0.1062])
  - `row` 17.0445 vs 14.3918 (d = +0.68, CI [+0.4690, +4.8197])
  - `uniq` 0.8701 vs 0.8812 (d = -0.41, CI [-0.0203, -0.0026])

**OCRBench** — A2 (rescue-performing) vs B (ordinary teacher-only): n = 11 instances, significant: col, g2, g2_pct, lin, lin_pct, r_center
  - `col` 18.5028 vs 16.5301 (d = +0.76, CI [+0.6462, +3.3052])
  - `g2` 5.1186 vs 2.0115 (d = +2.68, CI [+2.2789, +3.9677])
  - `g2_pct` 0.9766 vs 0.8708 (d = +6.53, CI [+0.0883, +0.1193])
  - `lin` 0.0244 vs -0.1085 (d = +0.76, CI [+0.0493, +0.2345])
  - `lin_pct` 0.5259 vs 0.4868 (d = +0.77, CI [+0.0106, +0.0720])
  - `r_center` 9.8767 vs 11.7655 (d = -0.82, CI [-3.8971, -0.4480])

Three things this table makes visible that the means do not:

* **Most instances do not move at all.** At k = 8, 120 of 150 instances are
  bit-identical to the student's answer; at k = 32, 105 still are. The curve is
  not 150 instances improving slightly — it is a minority improving a lot.
* **The arms that gain, gain by flipping.** `teacher:8` improves 26 instances
  and worsens 4; `random:8` improves 10 and worsens 4; `adversarial:8` improves
  1 and worsens 3. A control that is supposed to be a null (adversarial) sits at
  chance-level churn; a control that is supposed to be partial (random) sits in
  between; the real ordering sits well above both.
* **The plateau is not free of cost.** Every arm that gains also loses a few
  instances: `teacher:32` improves 39 and worsens 6, and the teacher's own set
  improves 39 and worsens 4 — the teacher is not uniformly better than the
  student instance-by-instance, which is the same phenomenon S2-B reported
  (102 improved / 323 tied / 25 worsened on the frozen 450).

### 4.5 The reverse direction

The brief asks for a reverse sweep — start from the teacher's set `T` and swap
the student's tokens in, watching teacher accuracy fall. That sweep is the
**same family of sets** as the forward one, and the reason is exact rather than
approximate: because `|T_only| = |S_only| = m`, the mirror path

```
F_rev(k) = (T \ T_only[m-k:]) ∪ S_only[:k]
```

is *set-identical* to `F(m-k)` for every `k`. Removing the teacher's `k`
lowest-ranked disagreement tokens and adding the student's `k` highest-ranked
ones lands on exactly the set the forward curve visits at `k' = m-k`.

So a reverse direction adds no new sets, and the honest way to report it is to
read the forward curve from the right: **teacher accuracy degrades from 75.15 at
k = m to 74.69 by k = m-32** — i.e. removing the teacher's bottom 32 disagreement
tokens costs essentially nothing — and to 66.75 by k = m-8, i.e. the teacher's
accuracy is held up by a small minority of its set, at the top of its ranking.

The identity was checked directly rather than assumed: for every one of the 150
instances and `k ∈ {8, 16, 32, 64}`, `F_rev(k)` and `F(m-k)` were built
independently and compared as sets — **600 of 600 identical**.

Four `F_rev` arms were then run anyway, because the identity is a claim about
sets and six minutes of GPU is cheap:

| arm | macro | TextVQA | DocVQA | OCRBench | forward point it re-measures |
|---|---|---|---|---|---|
| `F_rev:16` | 75.17 | 75.60 | 81.90 | 68.00 | `m-16` ≈ floor 85 / 111 / 92 — the plateau |
| `F_rev:32` | 74.50 | 76.20 | 81.30 | 66.00 | `m-32` ≈ 69 / 95 / 76 — the plateau |
| `F_rev:64` | 72.64 | 76.20 | 79.73 | 62.00 | `m-64` ≈ 37 / 63 / 44 — the shoulder |
| `F_rev:96` | 72.42 | 71.60 | 77.65 | 68.00 | *not a clean point — see below* |

They land where the identity says they should: at the teacher plateau for
k = 16 and 32, and stepping down onto the shoulder at k = 64. Since the harness
is deterministic (Section 1), an independently-built set that is
*set-identical* to a forward point must produce that point's answer, and it
does.

The k = 96 arm is the exception that tests the reading rather than the identity.
`|T_only|` is 67–176 with a mean of 112, so a 96-token reverse swap **saturates**
— it becomes the teacher's whole set — for 18 of 50 TextVQA instances and 16 of
50 OCRBench instances (but only 2 of 50 DocVQA ones). Its per-benchmark numbers
are therefore a mixture of "the teacher's own set" and "the shoulder", which is
exactly the pattern measured: DocVQA, the one benchmark that is not saturated,
drops furthest from the plateau (81.90 → 77.65), while TextVQA and OCRBench hold
up because most of their instances have already become `T`. The arm is listed
for completeness and should not be read as a forward-curve point.

So a reverse direction adds no new sets, and the honest way to report it is to
read the forward curve from the right. **Teacher accuracy degrades from 75.15 at
k = m to 74.69 by k = m-32** — removing the teacher's 32 lowest-ranked
disagreement tokens costs essentially nothing — **and to 66.75 by k = m-8**.
The teacher's accuracy is held up by a small minority of its own set, at the top
of its ranking, exactly mirroring where the forward curve found the value.

## 5. Answer-rescue analysis

### 5.1 Method

For each of the 28 instances in the *student-wrong → teacher-correct* class, the
per-instance score is tracked along the teacher-priority curve and the first `k`
at which it crosses the correct threshold is recorded as the **rescue level**
`k*`, together with the level immediately below it. The interval
`(k_prev, k*]` is the **transition block**: the tokens whose addition coincides
with the answer becoming correct.

Some instances are already rescued at `k = 2`, so the fine grid starts at
`k = 1`; the full grid is
`k ∈ {1,2,3,4,6,8,10,12,16,20,24,28,32,36,40,48,56,64,72,80,88,96}` plus the
endpoint.

### 5.2 Leave-one-out ablation — the sparse-vs-set test

At the rescue level `k*`, the `j`-th teacher token to enter, `T_only[j]`,
displaced the student token `S_only[m-k*+j]`. Ablating rank `j` therefore means
running the still-256-token set

```
F(k*) \ {T_only[j]} ∪ {S_only[m-k*+j]}
```

one `j` at a time, for every `j` in the transition block.

The logic is a two-sided discriminator:

* if **no** single ablation undoes the rescue, the block's value is not carried
  by any individual token — it behaves as a group (value spread over the block,
  or a genuinely set-dependent requirement);
* if **some** single ablation undoes it, that token is critical *given the rest
  of the set*, which is the sharp version of "a few tokens carry the value".

**The control that makes this interpretable.** Robustness is only meaningful if
the rescued answer is not marginal. So the same ablation is also run on tokens
added *earlier* in the ordering, far below the transition block. If ablating an
early token also destroys the answer, the model's correctness on that instance
is fragile to any perturbation and the block result carries no information.
Both are reported together, and instances whose early-token ablations break the
answer are flagged rather than quietly dropped.

A third arm (`blockrand`) keeps the same k and the same block membership but
re-orders the block, freezing everything above it. It separates "these
particular tokens, in this order" from "this many tokens from this part of the
teacher's list": if the rescue survives a reshuffle of the top block, the block
is acting as a group rather than as a ranked sequence.

### 5.3 Results — the rescue rate

| cum. tokens swapped | instances rescued |
|---|---|
| ≤ 2 | 4 of 28 |
| ≤ 4 | 4 of 28 |
| ≤ 6 | 9 of 28 |
| ≤ 8 | 12 of 28 |
| ≤ 12 | 16 of 28 |
| ≤ 16 | 19 of 28 |
| ≤ 24 | 20 of 28 |
| ≤ 32 | 23 of 28 |
| ≤ 48 | 24 of 28 |
| ≤ 64 | 24 of 28 |
| ≤ 96 | 26 of 28 |
| only at the full swap | 2 more, 28 of 28 |

| benchmark | rescuable | median rescue k | range |
|---|---|---|---|
| TextVQA_VAL | 5 | 10 | 2–48 |
| DocVQA_VAL | 11 | 12 | 2–96 |
| OCRBench | 12 | 12 | 2–96 |

**rescue rate on the 28 student-wrong / teacher-correct instances**

| arm | TextVQA_VAL | DocVQA_VAL | OCRBench | ALL (28) |
|---|---|---|---|---|
| `adversarial:8` | 0 % | 0 % | 0 % | 0 % |
| `adversarial:16` | 20 % | 9 % | 8 % | 11 % |
| `adversarial:32` | 20 % | 9 % | 8 % | 11 % |
| `random:2:s0` | 0 % | 9 % | 0 % | 4 % |
| `random:4:s0` | 20 % | 9 % | 0 % | 7 % |
| `random:8:s0` | 20 % | 45 % | 8 % | 25 % |
| `random:8:s1` | 0 % | 18 % | 17 % | 14 % |
| `random:8:s2` | 20 % | 18 % | 33 % | 25 % |
| `random:16:s0` | 20 % | 36 % | 33 % | 32 % |
| `random:16:s1` | 20 % | 9 % | 8 % | 11 % |
| `random:16:s2` | 60 % | 55 % | 42 % | 50 % |
| `random:32:s0` | 40 % | 82 % | 58 % | 64 % |
| `random:32:s1` | 80 % | 55 % | 25 % | 46 % |
| `random:32:s2` | 20 % | 45 % | 42 % | 39 % |
| `shuffled:8` | 0 % | 0 % | 0 % | 0 % |
| `shuffled:16` | 20 % | 9 % | 0 % | 7 % |
| `shuffled:32` | 20 % | 18 % | 8 % | 14 % |
| `teacher:2` | 20 % | 9 % | 17 % | 14 % |
| `teacher:6` | 20 % | 45 % | 25 % | 32 % |
| `teacher:8` | 40 % | 36 % | 42 % | 39 % |
| `teacher:12` | 60 % | 64 % | 50 % | 57 % |
| `teacher:16` | 60 % | 82 % | 58 % | 68 % |
| `teacher:24` | 60 % | 82 % | 67 % | 71 % |
| `teacher:32` | 60 % | 91 % | 83 % | 82 % |
| `teacher:40` | 80 % | 91 % | 75 % | 82 % |
| `teacher:48` | 80 % | 91 % | 83 % | 86 % |
| `teacher:64` | 80 % | 91 % | 75 % | 82 % |
| `teacher:80` | 80 % | 91 % | 92 % | 89 % |
| `teacher:96` | 80 % | 100 % | 92 % | 93 % |


The same story as Section 4, in discrete form and with much less noise, because
an answer either flips or it does not — and it is the quantity a method would
actually have to move: **the number of instances the student gets wrong that a
swap turns right.**

Two further features of the rescue table carry the interpretation:

* **Every rescuable instance is rescued by the swap.** All 28 reach a correct
  answer at some k ≤ |`T_only`|, which is guaranteed by construction at full
  swap (the set becomes `T` exactly) but which the intermediate grid already
  achieves for 26 of them. There is no instance here where the teacher's set
  contains the answer and a budget-preserving substitute cannot reach it — the
  failure really is about *which* 256 tokens, not about how many.
* **The rescue is fast, and its speed differs by benchmark.** Most instances
  are rescued at single-digit k. Counting the ones that are not: 2 of TextVQA's
  5 rescuable instances need k ≥ 48, against 1 of DocVQA's 11 and 2 of
  OCRBench's 12. So TextVQA, where the student starts closest to the teacher,
  is also where the *remaining* rescues are proportionally the most expensive —
  an observation, not a result, at n = 5 instances.

### 5.4 Results — the rescue is collective, not any single token

22 of the 28 rescuable instances have a non-empty transition block; the six
others are rescued at k = 1–2, where there is no "previous still-wrong level" to
form an interval against. Across those 22 instances, ablating one token at a
time from inside the transition block:

| | ablations | broke the rescue | rate |
|---|---|---|---|
| **inside the transition block** | 190 | 25 | **13.2 %** |
| control: tokens added far earlier | 61 | 5 | 8.2 % |

Per instance, 13 of 22 have at least one block token whose removal breaks the
answer, and 3 of 22 have at least one such token among the early controls.

**leave-one-out at the rescue level**

22 instances, 190 block ablations (25 broke the rescue); control 61 early ablations (5 broke the rescue)

| instance | k* | block | block ablations | broke | early ablations | broke |
|---|---|---|---|---|---|---|
| `TextVQA_VAL_1812` | 8 | 2 | 2 | 2 | 3 | 0 |
| `TextVQA_VAL_2718` | 12 | 4 | 4 | 0 | 3 | 0 |
| `TextVQA_VAL_2818` | 48 | 16 | 16 | 0 | 3 | 0 |
| `DocVQA_VAL_108` | 24 | 8 | 8 | 0 | 3 | 0 |
| `DocVQA_VAL_323` | 12 | 4 | 4 | 4 | 3 | 2 |
| `DocVQA_VAL_969` | 6 | 4 | 4 | 0 | 2 | 0 |
| `DocVQA_VAL_1723` | 96 | 32 | 32 | 7 | 3 | 0 |
| `DocVQA_VAL_2154` | 6 | 4 | 4 | 0 | 2 | 0 |
| `DocVQA_VAL_2369` | 6 | 4 | 4 | 1 | 2 | 0 |
| `DocVQA_VAL_2907` | 16 | 4 | 4 | 0 | 3 | 0 |
| `DocVQA_VAL_3661` | 16 | 4 | 4 | 1 | 3 | 0 |
| `DocVQA_VAL_4415` | 6 | 4 | 4 | 1 | 2 | 0 |
| `DocVQA_VAL_4953` | 12 | 4 | 4 | 1 | 3 | 0 |
| `OCRBench_382` | 8 | 2 | 2 | 1 | 3 | 0 |
| `OCRBench_583` | 32 | 16 | 16 | 1 | 3 | 0 |
| `OCRBench_603` | 8 | 2 | 2 | 1 | 3 | 0 |
| `OCRBench_744` | 96 | 32 | 32 | 2 | 3 | 0 |
| `OCRBench_764` | 6 | 4 | 4 | 0 | 2 | 0 |
| `OCRBench_825` | 32 | 16 | 16 | 1 | 3 | 1 |
| `OCRBench_845` | 16 | 4 | 4 | 2 | 3 | 2 |
| `OCRBench_865` | 32 | 16 | 16 | 0 | 3 | 0 |
| `OCRBench_885` | 12 | 4 | 4 | 0 | 3 | 0 |

22 instances, 190 block ablations (25 broke the rescue); control 61 early ablations (5 broke the rescue)

| instance | k* | block | block ablations | broke | early ablations | broke |
|---|---|---|---|---|---|---|
| `TextVQA_VAL_1812` | 8 | 2 | 2 | 2 | 3 | 0 |
| `TextVQA_VAL_2718` | 12 | 4 | 4 | 0 | 3 | 0 |
| `TextVQA_VAL_2818` | 48 | 16 | 16 | 0 | 3 | 0 |
| `DocVQA_VAL_108` | 24 | 8 | 8 | 0 | 3 | 0 |
| `DocVQA_VAL_323` | 12 | 4 | 4 | 4 | 3 | 2 |
| `DocVQA_VAL_969` | 6 | 4 | 4 | 0 | 2 | 0 |
| `DocVQA_VAL_1723` | 96 | 32 | 32 | 7 | 3 | 0 |
| `DocVQA_VAL_2154` | 6 | 4 | 4 | 0 | 2 | 0 |
| `DocVQA_VAL_2369` | 6 | 4 | 4 | 1 | 2 | 0 |
| `DocVQA_VAL_2907` | 16 | 4 | 4 | 0 | 3 | 0 |
| `DocVQA_VAL_3661` | 16 | 4 | 4 | 1 | 3 | 0 |
| `DocVQA_VAL_4415` | 6 | 4 | 4 | 1 | 2 | 0 |
| `DocVQA_VAL_4953` | 12 | 4 | 4 | 1 | 3 | 0 |
| `OCRBench_382` | 8 | 2 | 2 | 1 | 3 | 0 |
| `OCRBench_583` | 32 | 16 | 16 | 1 | 3 | 0 |
| `OCRBench_603` | 8 | 2 | 2 | 1 | 3 | 0 |
| `OCRBench_744` | 96 | 32 | 32 | 2 | 3 | 0 |
| `OCRBench_764` | 6 | 4 | 4 | 0 | 2 | 0 |
| `OCRBench_825` | 32 | 16 | 16 | 1 | 3 | 1 |
| `OCRBench_845` | 16 | 4 | 4 | 2 | 3 | 2 |
| `OCRBench_865` | 32 | 16 | 16 | 0 | 3 | 0 |
| `OCRBench_885` | 12 | 4 | 4 | 0 | 3 | 0 |

Read against the two-sided discriminator laid out in Section 5.2:

* **There is no single magic token.** In 9 of 22 instances *no* single block
  token is individually necessary, and overall 87 % of single-token removals
  from inside the block leave the rescued answer intact. The brief's mechanism A
  ("a small number of tokens with extremely high individual value") is therefore
  **not** the right description at the single-token level.
* **But it is not pure value-mass either.** The block ablation rate is 1.6× the
  early-token fragility control (13.2 % vs 8.2 %), so the block's tokens are
  genuinely more load-bearing than tokens added earlier in the same sweep. And a
  minority of instances behave exactly like a required set: `TextVQA_VAL_1812`
  has a block of 2 and *both* removals break it; `DocVQA_VAL_323` has a block of
  4 and all four break it.
* **The control matters and it is not zero.** 8.2 % of ablated early tokens also
  break the answer, which says the rescued answers sit close enough to the
  decision boundary that any single-token perturbation has a ~1-in-12 chance of
  undoing them. That is why the block rate is quoted *against* this control
  rather than against zero.

The honest summary is that the value is carried by a **small group acting
together**, of median size 4 in this grid and never more than 32, with no
individually necessary member in most instances.

---

## 6. Characterising the high-value disagreements

### 6.1 Cheap properties, all from caches

Per visual token, computed from the S2-C1 feature memmaps and the S2-B/S2-C1
score archives — no LLM forward, no new model, no training:

| property | meaning |
|---|---|
| `g2`, `g2_pct` | P1-G2 teacher score, and its rank percentile in the image |
| `lin`, `lin_pct` | LIN_L4 student score, and its rank percentile |
| `n_l2`, `n_l4` | hidden-state L2 norm at layer 2 and layer 4 |
| `d_norm` | `‖h_L4 − h_L2‖` — how far the representation moves between layers |
| `cos_l2_l4` | cosine between the layer-2 and layer-4 representations |
| `sim_sel_S`, `sim_sel_T` | mean cosine similarity to the members of S / of T |
| `nb_sim` | mean cosine to the 8 spatial neighbours (local homogeneity) |
| `uniq` | largest cosine to any other token (high = a duplicated token) |
| `row`, `col`, `r_center` | position on the 32×32 grid, and distance from centre |

Plus three **group-level spatial** statistics, which no per-token mean can
express: the number of distinct 8×8 blocks a group touches, the mean pairwise
grid distance between its members, and the mean distance to its own centroid.

### 6.2 Groups compared

| group | definition |
|---|---|
| **A** rescue block | teacher-only tokens added inside the transition block of a rescued instance |
| **A2** rescue-added | all tokens added up to and including `k*` on a rescued instance |
| **B** teacher-only | all of `T_only`, the same instances |
| **C** student-only | all of `S_only`, the same instances |
| **C2** removed | the `S_only` tokens the swap displaced |
| **C3** kept | the `S_only` tokens it left in place |
| **D** shared | `C = T ∩ S` |

### 6.3 Statistics

Every comparison bootstraps **at the instance level** — the group is averaged
within an instance first, then instances are resampled — because a token count
of tens of thousands is not tens of thousands of independent observations. With
28 class-1 instances, that is an honest n = 28 (fewer where a group is empty on
some instance), and the report says so rather than quoting token counts.

Properties that do **not** separate the groups are reported as negative results.

---

### 6.4 Results — the group that performs the rescue is not special in any cheap way

**the tokens that perform a rescue vs ordinary teacher-only tokens**

| property | group A | group B | delta | 95 % CI | Cohen d | n inst |
|---|---|---|---|---|---|---|
| `g2` | 7.9235 | 2.5378 | +5.3857 | [+4.0431, +6.9153] | +1.78 **(sig)** | 26 |
| `g2_pct` | 0.9779 | 0.8711 | +0.1068 | [+0.0964, +0.1157] | +6.22 **(sig)** | 26 |
| `lin` | -0.1247 | -0.1871 | +0.0623 | [-0.0557, +0.1659] | +0.23 | 26 |
| `lin_pct` | 0.5012 | 0.4844 | +0.0168 | [-0.0151, +0.0441] | +0.20 | 26 |
| `n_l2` | 35.6230 | 35.6458 | -0.0228 | [-1.3995, +1.6715] | -0.00 | 26 |
| `n_l4` | 44.1120 | 44.2988 | -0.1867 | [-1.5725, +1.4959] | -0.03 | 26 |
| `d_norm` | 20.3466 | 20.4252 | -0.0786 | [-0.7426, +0.6656] | -0.03 | 26 |
| `cos_l2_l4` | 0.8784 | 0.8797 | -0.0014 | [-0.0139, +0.0100] | -0.04 | 26 |
| `sim_sel_S` | 0.4356 | 0.4398 | -0.0042 | [-0.0210, +0.0099] | -0.07 | 26 |
| `sim_sel_T` | 0.4571 | 0.4525 | +0.0045 | [-0.0088, +0.0165] | +0.08 | 26 |
| `nb_sim` | 0.5355 | 0.5368 | -0.0013 | [-0.0246, +0.0211] | -0.02 | 26 |
| `uniq` | 0.8708 | 0.8864 | -0.0156 | [-0.0289, -0.0057] | -0.46 **(sig)** | 26 |
| `row` | 17.1298 | 15.4733 | +1.6565 | [+0.4638, +2.8452] | +0.40 **(sig)** | 26 |
| `col` | 17.0577 | 16.1722 | +0.8855 | [-0.4653, +2.0915] | +0.26 | 26 |
| `r_center` | 10.8691 | 11.7987 | -0.9296 | [-2.0237, +0.0235] | -0.45 | 26 |


**teacher-only vs student-only tokens**

| property | group A | group B | delta | 95 % CI | Cohen d | n inst |
|---|---|---|---|---|---|---|
| `g2` | 2.5893 | 0.6803 | +1.9090 | [+1.5404, +2.3643] | +1.98 **(sig)** | 28 |
| `g2_pct` | 0.8707 | 0.5567 | +0.3140 | [+0.3015, +0.3263] | +14.74 **(sig)** | 28 |
| `lin` | -0.1729 | 1.3612 | -1.5341 | [-1.6052, -1.4672] | -9.14 **(sig)** | 28 |
| `lin_pct` | 0.4889 | 0.8528 | -0.3638 | [-0.3833, -0.3449] | -10.79 **(sig)** | 28 |
| `n_l2` | 35.7545 | 30.1265 | +5.6280 | [+4.5521, +6.6955] | +1.69 **(sig)** | 28 |
| `n_l4` | 44.4715 | 38.5771 | +5.8943 | [+4.8806, +6.9023] | +1.72 **(sig)** | 28 |
| `d_norm` | 20.6078 | 19.0074 | +1.6004 | [+0.9732, +2.2573] | +0.84 **(sig)** | 28 |
| `cos_l2_l4` | 0.8779 | 0.8682 | +0.0097 | [+0.0014, +0.0178] | +0.57 **(sig)** | 28 |
| `sim_sel_S` | 0.4388 | 0.4713 | -0.0325 | [-0.0458, -0.0195] | -0.58 **(sig)** | 28 |
| `sim_sel_T` | 0.4500 | 0.4510 | -0.0010 | [-0.0133, +0.0114] | -0.02 | 28 |
| `nb_sim` | 0.5319 | 0.5209 | +0.0111 | [-0.0056, +0.0278] | +0.16 | 28 |
| `uniq` | 0.8863 | 0.8833 | +0.0030 | [-0.0020, +0.0075] | +0.14 | 28 |
| `row` | 15.4545 | 11.1926 | +4.2620 | [+2.6636, +5.7520] | +1.60 **(sig)** | 28 |
| `col` | 16.2261 | 14.3450 | +1.8811 | [+1.0273, +2.8380] | +1.28 **(sig)** | 28 |
| `r_center` | 11.8618 | 13.9817 | -2.1199 | [-2.6127, -1.5941] | -2.46 **(sig)** | 28 |

### 6.5 What separates, and what does not

**What does not separate — and this is the finding.** Comparing the tokens whose
addition actually performs a rescue (group A2, the `T_only` tokens up to the
rescue level of a rescued instance) against the rest of the same disagreement
set (group B), *every* cheap geometric property comes back null:

* `lin` / `lin_pct` — the student's own score. The student scores the tokens
  that carry the downstream value exactly as it scores the ordinary ones.
* `n_l2`, `n_l4`, `d_norm`, `cos_l2_l4` — hidden-state norm and how much the
  representation moves from layer 2 to layer 4.
* `sim_sel_S`, `sim_sel_T` — redundancy against the student's set and the
  teacher's set.
* `nb_sim` — local homogeneity against the 8 spatial neighbours.
* `row`, `col`, `r_center` — position (with `row` marginally significant at
  d = +0.40 and 26 instances; see the multiplicity note below).

Two properties come back marginally significant — `uniq` (the rescuing tokens
are slightly *less* duplicated: 0.871 vs 0.886, Cohen d = −0.46) and `row` (they
sit about 1.7 grid rows lower: 17.1 vs 15.5, d = +0.40). Both should be read
with the multiplicity in mind: fifteen properties were tested, so roughly one
false positive at the 95 % level is expected, and two marginal hits at |d| < 0.5
on 26 instances is consistent with noise. Neither has an obvious mechanism, and
neither is used as a finding anywhere in this report.

The only properties that separate the groups cleanly are `g2` / `g2_pct` — the
teacher's own gradient score — and that separation is circular: the groups are
*defined* by teacher rank.

The practical consequence is stark: **there is no cheap feature that identifies
the high-value disagreement tokens, because the features that a cheap scorer
could read do not distinguish them.** Any method that hopes to recover this
value from layer-2/layer-4 hidden states has to learn something the linear
statistics of those states do not already expose — which is consistent with
S2-C1's finding that a linear probe saturates at 0.56 Top-256 overlap no matter
how many parameters it is given.

**What does separate.** Two things, and neither is about individual tokens:

* **Teacher-only tokens differ from student-only tokens as groups.** Group B
  against group C shows large, significant shifts: the teacher's disagreement
  tokens have higher hidden-state norm at both layers (44.5 vs 38.6 at L4,
  |d| = 1.72), larger L2→L4 movement, higher `cos_l2_l4`, lower similarity to
  the student's set, and sit closer to the image centre (radius 11.86 vs 13.98,
  |d| = 2.46). So the teacher's *preferences* are systematically different —
  they are just not more identifiable token-by-token *within* its own set.
* **The rescuing tokens are spatially more concentrated than the disagreement
  set as a whole — and more concentrated than the student tokens they replace.**
  Group A2 (19.9 tokens per instance) touches 6.4 of the 16 8×8 blocks, against
  15.4 for the full disagreement set, at a radius of 7.90 grid units around its
  own centroid versus 11.11. The `S_only` tokens the swap *removes* are in the
  same size range but are markedly less concentrated: 7.8 blocks touched at a
  radius of 10.96. So the value is not one isolated spot and it is not spread
  over the page either — it is a few small clusters, and it clusters more
  tightly than the tokens it displaces.

That last point — a handful of small clusters, no one of which is a single
critical token — is what the leave-one-out test in Section 5 probes directly.

---

## 7. Hypotheses: what is supported and what is refuted

The brief's four candidate mechanisms, evaluated against the evidence above:

| | hypothesis | verdict |
|---|---|---|
| **H1** | the teacher's extra value sits in a few rare/high-value tokens | **supported, and stronger than stated** — not "rare": the value is in the *highest-ranked* tokens of the teacher's disagreement set, and it is recoverable by swapping in single-digit numbers of them |
| **H2** | the student wastes budget on redundant/low-value tokens | **refuted** — removing the student's weakest disagreement tokens and refilling with tokens from neither set *lowers* accuracy |
| **H3** | token value is set-dependent / needs complementary evidence composition | **partially supported in a narrow form** — no single rescuing token is individually critical given the rest, so within a small block the value is collective; but the set requirement is tiny (single-digit tokens), not a page-level composition problem |
| **H4** | the gap is document/OCR-specific fine-detail preservation | **not supported as a separate mechanism** — the same shape (front-loaded value, inert removal) appears on all three benchmarks; the benchmarks differ in *how much* is missing, not in *how* it is missing |

### 7.1 The four hypotheses, with the measurement that decides each

**H1 — the teacher's extra value is concentrated in a few high-value tokens.**
**Supported, and the "rare" part of the brief's wording is wrong.** The
concentration is not in unusual tokens; it is in the *top of the teacher's own
ranking*. A token's value along the disagreement set is a steeply decreasing
function of the teacher's rank within it: tokens 3–6 are worth +1.31 macro points
each, tokens 8–12 are worth +0.63, tokens 17–32 are worth +0.22, and past token
32 the curve is flat. `random:8` recovers 16 % of the gap where `teacher:8`
recovers 53 %, and `adversarial:8` and `shuffled:8` recover 0.6 % and 3.9 %
respectively. The value tracks the teacher's ordering, not any property of the
token in isolation.

**H2 — the student wastes budget on redundant or low-value tokens.**
**Refuted.** Three independent readings agree. (i) `remworst:8` — drop the
student's eight weakest disagreement tokens and refill with tokens belonging to
neither set — lands *below* the student (55.95 vs 57.32 macro, Δ = −1.37
[−3.74, +0.59]). (ii) The choice of which student tokens to remove, "weakest
first" or uniformly random, is never significantly different at k = 8, 16, 24 or
40; the point estimate drifts in favour of *random* removal at the larger k
(up to +1.97 [−1.20, +5.34] at k = 24). Nothing in the student's set needs
removing. (iii) The
student-only tokens are not geometrically redundant: against the shared set they
show no meaningful similarity excess, and the group-level spatial statistics put
them at 15.3 of 16 blocks touched, i.e. spread out, not piled up. Whatever limits
the student, it is not that its 256 tokens duplicate each other.

**H3 — token value is set-dependent and needs complementary evidence.**
**Partly supported, in a much narrower form than the brief anticipates.** The
leave-one-out test (Section 5.4) shows the rescue is not carried by any single
added token: 87 % of single removals from inside the transition block leave the
rescued answer intact, and in 9 of 22 instances *no* member is individually
necessary. But the block does carry more weight than the fragility control
(13.2 % vs 8.2 % of ablations break the answer), and a minority of instances
behave like a strictly required set — `TextVQA_VAL_1812` needs both of its two
tokens, `DocVQA_VAL_323` needs all four of its. So the value is collective
**at a median group size of four tokens**, not a page-level evidence-composition
problem and not a single-token effect.

**H4 — the gap is document/OCR-specific fine-detail preservation.**
**Not supported as a distinct mechanism.** The *shape* of the result is the same
on all three benchmarks — front-loaded value along the teacher's rank, inert
removal, the same control ordering (`adversarial` ≈ student < `shuffled` <
`random` < `teacher`) — and the cheap-property characterization is null in the
same way everywhere. What differs between benchmarks is only *how much* of the
teacher's head the student misses: 1.56 of the top-8 on TextVQA against 2.74 on
DocVQA, and it is DocVQA and OCRBench where that shows up as a gap. "Document
fine detail" describes where the damage is visible, not a separate mechanism.

---

## 8. What the next stage should actually solve

Three claims, each of them measured rather than argued.

### 8.1 The target is head recall, not overlap

The student's delivered 256-token set already contains **73.6 % of the teacher's
top-8 tokens**. It is missing about **2.1 of the teacher's best 8 per image**,
and Section 4 shows that putting those back — plus about six more from the
teacher's top-32, at no cost to the budget — recovers **more than half** of the
entire accuracy gap.

That gives the next stage a concrete, falsifiable objective function, and it is
not the one S2-C1 was gated on:

> **recall of the teacher's top-k tokens within the selected 256, for k around
> 8–32** — not Top-256 overlap.

The two are not interchangeable. Top-256 overlap is already 0.562 and is
measured at the one point where Top-k agreement and Top-k recall are forced to
be the same number. A scorer that lifts Top-256 overlap from 0.56 to 0.60 by
picking up mid-ranked tokens would buy nothing measurable here; one that lifts
top-8 *in-selection* recall from 0.736 toward 1.0 is aimed at the half of the
gap that this stage measured.

### 8.2 The failure is a ranking failure at the top, not a coverage or redundancy failure

Everything the student does *after* ranking is not the problem. It keeps most of
the teacher's best tokens, its removals are inert, and its set is not
redundant. What it does badly is place the teacher's very best tokens high
enough in its own ordering — 19.2 % agreement in the top-8 against 73.6 %
inclusion in the top-256.

So two families of method story are closed off by measurement, not by opinion:

* **"de-duplicate / de-redundify the selection"** — aimed at H2, which is
  refuted: removing the student's weakest tokens is worth nothing, and refilling
  with neutral tokens makes things slightly worse.
* **"make the selection more teacher-like in aggregate"** — the `shuffled`
  control supplies a real P1-G2 map with the correct marginal distribution and
  gains 3.9 % of the gap. Distributions are not the currency; this image's
  ranking is.

### 8.3 The scorer has to express something the current features do not carry

The tokens whose addition performs a rescue are, on every cheap property
available *within the same disagreement set*, indistinguishable from ordinary
teacher-only tokens: student score, hidden-state norm at layer 2 and layer 4,
L2→L4 movement, redundancy against the selected set, neighbour similarity,
uniqueness and position all come back null or marginal (Section 6). Only the
teacher's own gradient score separates them cleanly, and that separation is
circular — the groups are defined by teacher rank.

Two consequences, stated as hypotheses for the next stage rather than as
designs:

1. **Retargeting may matter more than capacity.** S2-C1 established that a
   linear probe saturates at ~0.56 overlap whether it has 4 k or 528 k
   parameters, and that adding the question buys nothing. If the cheap features
   genuinely do not contain the head information, then "the same read-out with
   more capacity" is the wrong move; a target expressed at the head of the
   ranking — rather than "is this token in the teacher's Top-256" — is the
   change the evidence points at.
2. **Independence may be a structural limit.** The leave-one-out result
   (Section 5.3) shows the rescue is not carried by any single added token. A
   scorer that assigns every token a score without reference to the rest of the
   set may be unable to express whatever it is that the rescuing cluster
   supplies jointly. This is the one hypothesis from the brief's list that this
   stage raises without settling, and it is the natural target for a next stage
   that is allowed to change the scorer's *form*.

### 8.4 A reporting note that matters for the next stage's evaluation

The gap is not spread over the benchmark. 28 of 150 instances are
*student-wrong, teacher-correct*, and on DocVQA the per-instance median delta of
every swap arm is 0.00 — the means move because a minority of answers flip. The
next stage should report **rescue counts** alongside means, or it will be
optimising a statistic whose movement is carried by a handful of instances.

---

## 9. What this stage does not claim

* **Nothing here is a method.** No scorer was trained, no selector was changed,
  no fusion or loss was swept. The swap arms are diagnostic counterfactuals that
  require the teacher's own map, which costs 507 ms/image (S2-A) — 2.5× the
  unpruned prefill they would accelerate. They are a measuring instrument, not a
  proposal.
* **The teacher is not shown to be beaten.** At k = 48 and 96 the swap curve
  sits nominally *above* the teacher's own set (75.74 and 75.76 against 75.15
  macro). A +0.6 macro difference at n = 50/benchmark is far below the
  resolution of this design — the paired bootstrap CI on the nearest measured
  arm-vs-arm difference is roughly ±7 macro points — and the per-benchmark
  pattern is not consistent (TextVQA and DocVQA below the teacher's set,
  OCRBench above). It is reported as a plateau; no claim is made that any
  256-token set here beats the teacher.
* **n = 50 per benchmark.** Every aggregate difference quoted as a gain has a
  paired bootstrap CI in the deliverable JSON; the ones discussed as effects are
  the ones whose CI clears zero.
* **The causal n=8 tier is not used anywhere in this stage.** Per the brief and
  per S2-C1's inversion result, it appears in no selection or verdict.

---

## 10. Reproduce

```
source ~/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
cd /media/disk2/YZX/research/EADP/Qwen_vl

# 1. decomposition + head agreement (CPU only, no model)
python scripts/discovery/s2c2_decompose.py

# 2. harness identity: feed the student's and teacher's own sets back through
#    the swap machinery; must reproduce the cached accuracies exactly
python scripts/discovery/s2c2_swap.py --arms identity_S identity_T --tag s2c2_identity

# 3. the swap grid (several invocations; each merges into s2c2_swap.json, and
#    arms already recorded are skipped, so re-running is free)
bash scripts/discovery/run_s2c2_swap.sh      # teacher-priority, random, adversarial…
bash scripts/discovery/run_s2c2_pass2.sh     # add/remove split + controls, priority-ordered
python scripts/discovery/s2c2_swap.py --tag s2c2_swap --arms \
    F_rev:16 F_rev:32 F_rev:64 F_rev:96                 # reverse-sweep mechanism check
python scripts/discovery/s2c2_swap.py --tag s2c2_swap --arms \
    remworst:16:s0 remworst:24:s0 remworst:40:s0        # removal half at larger k

# 4. answer-rescue refinement on the 28 rescuable instances (fine grid + LOO)
bash scripts/discovery/run_s2c2_rescue.sh

# 5. token properties and group statistics (CPU, reads the S2-C1 memmaps)
python scripts/discovery/s2c2_characterize.py --extract
python scripts/discovery/s2c2_characterize.py --compare --spatial \
    --rescue-plan outputs/discovery/s2c2_rescue_plan.json

# 6. tables and figure
python scripts/discovery/s2c2_consolidate.py
python scripts/discovery/s2c2_tables.py > outputs/discovery/s2c2_tables.md
python scripts/discovery/s2c2_figure.py
```

Deliverables: `outputs/discovery/s2c2_diagnosis.json` (every curve, CI, flip
count, rescue rate and leave-one-out record), plus `s2c2_decompose.json`,
`s2c2_swap.json`, `s2c2_identity.json`, `s2c2_rescue.json`,
`s2c2_rescue_plan.json`, `s2c2_token_props.npz`, `s2c2_characterize.json`,
`s2c2_spatial.json`, and `figures/s2c2_swap_curves.png`.*
