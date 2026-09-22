# S2-C1 — Lightweight teacher distillation viability

**Scope.** Stage-2 S2-C1. One question: *can a tiny learned scorer predict which
visual tokens the P1-G2 gradient teacher considers useful, using only cheap
early-forward representations?*

This is a viability test, not a model-training stage. The search space is one
configuration per scorer family: two layers, one loss, one rank dimension, one
optimizer setting, no sweeps. The one robustness check the brief allows in the
ambiguous branch was spent, and the stage stops there — no mid-layer pruning
integration, no state/KV reuse, no second-stage student.

**Status.** Complete. Scripts: `Qwen_vl/scripts/discovery/s2c1_{features,train,
eval,select,latency,consolidate}.py`. Deliverable:
`outputs/discovery/s2c1_distillation.json`.

---

## VERDICT

```
S2-C1 AMBIGUOUS:
a 4 k-parameter linear probe on layer-4 hidden states keeps 45-47 % of the
gradient teacher's Top-K accuracy gain -- inside the GO band on retention,
but it does not beat the official EADP facility baseline, so the gate fails.
```

Retention is **45.0 % (token-only) / 47.3 % (query-conditioned)**, well above the
40 % GO threshold and far above S2-C0's 4.3–7.8 %. But gate condition 3 — macro
accuracy above the official EADP facility baseline — fails by ~3 points for both
students, so this is not a GO. It is also nowhere near the < 20 % NO-GO floor.

The ambiguous branch permits exactly one robustness check. It was spent answering
the only question that could change the verdict — whether condition 3's failure
is a score deficiency or a selector artifact — and the answer is unambiguous: it
is a **score** deficiency (§6).

---

## 0. Executive summary

1. **A learned scorer is a different regime from every S2-C0 proxy.** On the
   held-out 150, teacher Top-256 overlap is **0.563** against the best attention
   proxy's 0.352 and a 0.250 chance rate; AUROC **0.798** against 0.612. The
   gap S2-C0 could not close with any forward attention signal is largely closed
   by ~4 000 learned parameters reading layer-4 hidden states (§3).
2. **That closed much, but not most, of the accuracy gap.** Top-256 overlap
   rises 0.35 → 0.56 while retained gain rises ~5 % → ~46 %. Retention is
   steeply non-linear in overlap: the first 60 % of the teacher's Top-256 set
   buys 46 % of its gain, the last 40 % buys the other 54 % (§5.2).
3. **Token-only ≈ query-conditioned, at 129× the parameters.** Adding the
   first-answer-position query in a rank-64 bilinear term moves held-out AUROC by
   0.003 and macro accuracy by +0.77 points — inside the bootstrap CI. The
   gradient teacher is, at this capacity, mostly detecting *generic
   information-bearing visual tokens*, not question-specific ones (§7).
4. **The scrambled-target floor is high and the students clear it.** The teacher's
   map rotated onto a different image retains 30.0 %. The learned scorers reach
   45.0–47.3 %, i.e. 15–17 points of genuinely content-dependent signal — where
   S2-C0's proxies sat 22–26 points *below* that floor (§5.3).
5. **n=8 causal AUROC is refuted as a selection metric, decisively this time.**
   The attention proxy C_L2 has *better* causal block AUROC than any learned
   scorer (0.705 vs 0.666) and *worse* accuracy by 12 points (45.27 vs 57.32).
   S2-C0 could only say the tier had no resolution; here it ranks two arms in the
   wrong order by a wide margin (§8).
6. **Nothing is overfitting.** Fit / val / held-out Top-256 overlap are
   0.59–0.61 / 0.555–0.557 / 0.554–0.563. All four arms land within 0.009
   overlap and 0.8 macro points of each other, and training stopped at epoch 5–8
   of 26–29. The constraint is **scorer capacity, not data or optimisation** (§2.4).
7. **Cost is comfortable.** Scoring-only is **29.5 ms** (prefix through L4 +
   0.14 ms scorer+TopK) against the teacher's 507 ms; naive end-to-end, including
   re-running the pruned model, is **102.3 ms** against a 203.2 ms unpruned
   prefill (§9).

---

## 1. Method

### 1.1 Teacher

Fixed, never recomputed: the cached S2-B **P1 + gradient × input (G2)** maps
(`s2b_gradient_scores.npz`, 450 instances). Two target readings:

| | definition | role |
|---|---|---|
| **T1** | `y_i = 1` iff token `i` is in the teacher's Top-256 | primary loss |
| **T2** | `y_i = rank_percentile(G2_i)` | secondary, implemented, not run |

No MSE against raw G2 — its scale is not stable across images, and S2-C0 showed a
rank-agreement objective selects the layers that are *least* causally useful.

### 1.2 Split (image-level)

The frozen 450 bank, split per benchmark without any token-level resampling:

| split | per benchmark | total | how |
|---|---|---|---|
| fit | 80 | 240 | `frozen_150` minus test minus val |
| val | 20 | 60 | `(frozen_150 − test)[::5]` |
| **held-out** | 50 | **150** | `frozen_150[::3]` |

The held-out 150 is **exactly** the S2-C0 / S2-B pilot — verified key-for-key, not
assumed — so the official facility, official Top-K, P1-G2 teacher, P1-G2 shuffled
and C_L2 baselines are all already cached on the identical instances and do not
need re-running. Every one of the 1 024 token labels moves with its image.

The 15 causal cases sit outside the frozen bank, are never trained on, and appear
only as the supplementary tier (§8).

> **One deviation from the brief, recorded explicitly.** The brief gives
> "train 100/benchmark, val/test 50/benchmark" and "train 300 images, test 150".
> Early stopping requires a validation set that is *not* the held-out set —
> otherwise the gate's held-out numbers are contaminated by model selection. The
> 60-instance validation tier is therefore carved out of the 100-image training
> pool (80 fit + 20 val). The stated budget is unchanged: 300 training images,
> 150 held-out images, zero leakage.

### 1.3 Features

One prompt-only forward per instance (465 instances: the frozen 450 plus the 15
causal), built exactly as `s2b_gradient_scores.py` — `inputs_embeds` with the
visual embeddings injected, no `input_ids`, no `image_grid_thw`, no backward
anywhere. Forward hooks on decoder layers **L2 and L4 (0-based, the S2-C0
convention)** capture

* `h_i` — hidden state of visual token `i` at the output of layer L → (1024, 4096)
* `q` — hidden state of the first answer position (last prompt token) at layer L → (4096,)

Both are what a scorer sitting after layer L has for free. Features are cached as
fp16 memmaps (max |h| ≈ 60, safely inside fp16 range). One fixed preprocessing
step, applied identically at train and test: per-dimension standardization by
fit-split mean/std.

### 1.4 Scorers

| arm | form | params |
|---|---|---|
| `LIN_L{L}` | `s_i = w·ĥ_i + b` | **4 097** |
| `QRY_L{L}` | `s_i = (W_v ĥ_i)·(W_q q̂)/√r + w_b·ĥ_i + c`, `r = 64` | **528 385** |

No MLP, no concat of `h` and `q`, no hidden-size sweep.

### 1.5 Loss and training

```
loss = balanced_BCE(s, T1) + 0.5 · mean_{(p,n)} max(0, 1 − s_p + s_n)
```

Positive weight 3.0 matches the 1:3 positive:negative ratio. The ranking term
uses **all** 256×768 positive/negative pairs per image rather than a sample, so
the term is deterministic and no sampling rate becomes a hidden hyperparameter.
`λ = 0.5` and margin `1.0` are fixed, as specified; no loss sweep was run.

The LLM and vision tower are frozen and are not even loaded during training — the
scorer only sees the cached arrays. AdamW, lr 1e-3, weight decay 1e-4, batch 8
images, ≤ 80 epochs, early stop with patience 20 on **validation Top-256
overlap**. The metric is overlap rather than Spearman deliberately: S2-C0 showed
teacher agreement is anti-indicative in the attention family, and overlap is what
the downstream Top-K selector actually consumes.

Selection of which arm goes to the accuracy translation is made on **val only**,
one per family. Train accuracy is never used to select anything.

---

## 2. What was trained

### 2.1 Arms

| arm | params | best epoch | epochs run | train time | scorer latency |
|---|---|---|---|---|---|
| `LIN_L2` | 4 097 | 8 | 29 | 205 s | 0.068 ms |
| `LIN_L4` | 4 097 | 8 | 29 | 232 s | 0.074 ms |
| `QRY_L2` | 528 385 | 5 | 26 | 337 s | 0.193 ms |
| `QRY_L4` | 528 385 | 5 | 26 | 176 s | 0.191 ms |

Training time is dominated by CPU-side feature reads, not GPU compute — an 8B
model is nowhere near this loop.

### 2.2 Top-256 overlap across the three tiers

| arm | fit (240) | val (60) | **held-out (150)** |
|---|---|---|---|
| `LIN_L2` | 0.592 | 0.556 | **0.563** |
| `LIN_L4` | 0.602 | 0.557 | **0.562** |
| `QRY_L2` | 0.589 | 0.555 | **0.554** |
| `QRY_L4` | 0.608 | 0.556 | **0.560** |

### 2.3 Val-selected arms

`LIN_L4` (val 0.5572) and `QRY_L4` (val 0.5561).

The layer decision is nominal: within a family the two layers differ by
**≤ 0.001** val overlap, and by ≤ 0.001 held-out overlap. Layer choice is not a
live variable here and nothing in this stage should be read as evidence about
which layer is better.

### 2.4 Nothing is overfitting, and that is the problem

Fit / val / held-out overlap are within 0.05 of each other and training stops
after 5–8 epochs of a 80-epoch budget. The scorer is **capacity-limited**: a
linear read-out of one layer's hidden states extracts roughly 0.56 of the
teacher's Top-256 and then saturates, and it saturates in the same place whether
it has 4 097 or 528 385 parameters. More data would not move it; the S2-C0
lesson that agreement rises with depth (§3.1) suggests depth would.

---

## 3. Tier 1 — teacher separability on the held-out 150

No selection decision in this stage touched these 150 instances.

| arm | AUROC | AP | Top-256 overlap | Recall@256 | Spearman | TextVQA | DocVQA | OCRBench |
|---|---|---|---|---|---|---|---|---|
| `LIN_L2` | **0.798** | **0.606** | **0.563** | **0.563** | 0.613 | 0.835 | 0.751 | 0.809 |
| `LIN_L4` | **0.798** | 0.605 | 0.562 | 0.562 | 0.613 | 0.834 | 0.749 | 0.811 |
| `QRY_L4` | 0.795 | 0.605 | 0.560 | 0.560 | 0.608 | 0.833 | 0.745 | 0.806 |
| `QRY_L2` | 0.793 | 0.598 | 0.554 | 0.554 | 0.604 | 0.831 | 0.746 | 0.803 |
| `C_L2` *(S2-C0)* | 0.612 | 0.372 | 0.352 | 0.352 | 0.248 | 0.622 | 0.604 | 0.612 |
| `A_L2` *(S2-C0)* | 0.607 | 0.378 | 0.353 | 0.353 | 0.232 | 0.606 | 0.608 | 0.607 |
| random | 0.499 | 0.253 | 0.248 | 0.248 | −0.002 | 0.504 | 0.497 | 0.496 |
| official EADP score | 0.402 | 0.219 | 0.185 | 0.185 | −0.188 | 0.385 | 0.448 | 0.373 |

*Chance Top-256 overlap = 0.250. The AUROC columns on the right are per-benchmark.*

**The learned scorer is in a different regime from the attention family.** It
moves Top-256 overlap from 0.352 to 0.563 — 56 % of the way from chance to
perfect — where S2-C0's best proxy had reached only 27 %. It is also the first
score in this project that beats the random map on *every* benchmark.

Two baselines are worth noting because they bracket the whole exercise:

* **The official EADP importance score is anti-correlated with the teacher**
  (AUROC 0.402, overlap 0.185, Spearman −0.188). It selects *against* the
  gradient teacher's preferences. This is why EADP's own score under Top-K
  collapses to 42.75 (§5).
* **The random map is a hard floor at 0.248** — anything at or below it carries
  no teacher-aligned information regardless of how it looks elsewhere.

### 3.1 An unresolved tension with S2-C0

S2-C0 found that agreement with the teacher *rose with depth* (Spearman 0.22 at
L1 → 0.35 at L8 in the attention family) while causal utility peaked sharply at
L2. This stage cannot resolve that: the learned scorers at L2 and L4 are
indistinguishable (ΔAUROC 0.003). S2-C1's layers were fixed at {2, 4} by the
brief, so the question "does a learned read-out keep improving toward L8?" is
left open — and §4.4 is the reason it matters.

---

## 4. Tier 2 — causal 15 (supplementary, does not gate)

Primary tier `nNec ≤ 256`, n = 8. Metrics reused verbatim from
`scoring_search_s1.evaluate`.

| method | mean rank | Δ vs official | AUROC | AP | R@128 | R@256 | R@512 |
|---|---|---|---|---|---|---|---|
| official EADP | 0.5466 | — | 0.380 | 0.207 | 0.051 | 0.151 | 0.425 |
| **P1-G2 teacher** | **0.3142** | **−0.2325** | **0.895** | 0.406 | 0.296 | 0.492 | 0.773 |
| `QRY_L4` | 0.3988 | −0.1478 | 0.669 | 0.567 | 0.213 | 0.377 | 0.636 |
| `QRY_L2` | 0.3996 | −0.1470 | 0.663 | 0.561 | 0.238 | 0.393 | 0.633 |
| `LIN_L4` | 0.4028 | −0.1438 | 0.666 | 0.566 | 0.223 | 0.375 | 0.633 |
| `LIN_L2` | 0.4036 | −0.1430 | 0.645 | 0.556 | 0.215 | 0.375 | 0.632 |
| `C_L2` *(S2-C0)* | 0.4264 | −0.1202 | **0.705** | 0.633 | 0.176 | 0.324 | 0.618 |
| `A_L2` *(S2-C0)* | 0.4352 | −0.1114 | 0.681 | 0.625 | 0.171 | 0.315 | 0.594 |

The learned scorers beat both attention proxies on necessary-token **rank**
(−0.144/−0.148 vs −0.111/−0.120) but lose on block **AUROC** (0.645–0.669 vs
0.681–0.705) and on **AP** (0.556–0.567 vs 0.625–0.633). The split verdict —
better mean rank, worse block AUROC — is the same shape as S2-C0's, and is
another reason not to gate on this tier.

### 4.1 The causal tier ranks two arms in the wrong order

Set the tiers side by side:

| arm | causal block AUROC (n=8) | held-out Top-256 overlap | **held-out accuracy** |
|---|---|---|---|
| `C_L2` | **0.705** | 0.352 | 45.27 |
| `LIN_L4` | 0.666 | **0.562** | **57.32** |

The causal tier prefers `C_L2`; accuracy prefers `LIN_L4` by 12 points. S2-C0's
conclusion was that n=8 causal AUROC has no resolution — this is stronger: it
actively **inverts** the ordering between two arms that differ by 12 accuracy
points, with the causal-AUROC gap (0.039) far outside what the tier's own null
spread (sd 0.092 at n=8) would call meaningful. Any future stage that selects on
this tier will select the worse arm.

This is the third independent time the causal tier has failed to track accuracy
(S2-A: teacher 0.895 → +16.7; S2-C0: 0.68–0.71 → −16; here: 0.705 → −12). The
tier is retained for interpretation only, as the brief directs.

---

## 5. Tier 3 — held-out accuracy translation

T = 256. **Every arm uses the same Top-K selector**, per the brief: S2-B showed
Top-K costs the gradient teacher only ~1.4 points against facility location, so
the selector is not the variable under test. Calibration is the identity — the
tested calibrations are monotone rescalings and Top-K depends only on the ranking.

### 5.1 Results — held-out 150, n = 50/benchmark

| arm | selector | TextVQA | DocVQA | OCRBench | **macro** | vs EADP TopK | vs facility | **retained** |
|---|---|---|---|---|---|---|---|---|
| official EADP **facility** | facility | 59.400 | 61.891 | 62.000 | **61.097** | +18.344 | — | — |
| **P1-G2 teacher** | topk | 77.000 | 80.450 | 68.000 | **75.150** | **+32.397** | +14.053 | **100 %** |
| **`QRY_L4`** (learned) | topk | 69.400 | 58.875 | 46.000 | **58.092** | +15.338 | −3.006 | **47.3 %** |
| **`LIN_L4`** (learned) | topk | 67.600 | 56.353 | 48.000 | **57.318** | +14.564 | −3.779 | **45.0 %** |
| `QRY_L4` *(robustness, §6)* | facility | 74.800 | 50.694 | 36.000 | 53.831 | +11.078 | −7.266 | 34.2 % |
| *P1-G2 shuffled* | topk | 63.600 | 45.769 | 48.000 | *52.456* | *+9.703* | *−8.641* | *30.0 %* |
| `C_L2` *(S2-C0)* | topk | 58.200 | 37.594 | 40.000 | 45.265 | +2.511 | −15.833 | 7.8 % |
| `A_L2` *(S2-C0)* | topk | 52.200 | 40.226 | 40.000 | 44.142 | +1.389 | −16.955 | 4.3 % |
| *EADP's own score* | topk | 41.400 | 42.860 | 44.000 | *42.753* | *0* | *−18.344* | *0 %* |

Paired bootstrap, 10 000 resamples, stratified by benchmark:

| arm | Δ vs EADP Top-K | 95 % CI | Δ vs facility | 95 % CI |
|---|---|---|---|---|
| P1-G2 teacher | +32.397 | — | +14.053 | [+7.16, +21.09] |
| `QRY_L4` | +15.338 | [+6.74, +24.05] | −3.006 | [−11.43, +5.70] |
| `LIN_L4` | +14.564 | [+5.80, +22.92] | −3.779 | [−11.22, +3.73] |
| `QRY_L4` @facility | +11.078 | [+2.70, +19.25] | −7.266 | [−15.45, +0.72] |

Both students' gains over EADP's own score under the same selector are
significant. Both deficits against the facility baseline are *not* — the CI
straddles zero, so condition 3 fails on the point estimate with n = 50/benchmark,
not by a margin the data resolves.

Harness identity: `n_kept_mean = 256.0` for both student arms, confirming the
maps actually drove a Top-256 selection rather than falling back.

### 5.2 Retention is steeply non-linear in overlap

| map | Top-256 overlap w/ teacher | retained gain (vs EADP Top-K) |
|---|---|---|
| random / EADP's own score | 0.185 – 0.248 | 0 % |
| `A_L2` / `C_L2` (S2-C0) | 0.352 – 0.353 | 4.3 – 7.8 % |
| **`LIN_L4` / `QRY_L4`** | **0.560 – 0.563** | **45.0 – 47.3 %** |
| P1-G2 teacher | 1.000 | 100 % |

Recovering the first 56 % of the teacher's Top-256 set buys 46 % of its accuracy
gain; the remaining 44 % of the set buys the other 54 %. The relationship is
strongly convex, which is the single most useful thing to carry out of this
stage: **it is the last stretch of the overlap that carries the value**, and any
method that stops at "we agree with the teacher on most tokens" will retain less
than half the gain.

### 5.3 The content-free floor, and why this is not S2-C0 again

The teacher's map **rotated onto a different image** retains **30.0 %** of the
teacher's Top-K gain. This floor — same value distribution, no content — is very
high, because the teacher's maps have consistent spatial structure (text-dense
regions are text-dense in most images).

That floor is exactly what killed the S2-C0 proxies: at 4.3–7.8 % they sat
**22–26 points below** a content-free rotation of the teacher's own map, which is
why S2-C0 could say decisively that the L2 attention signal was worth less than
nothing.

The learned scorers sit **15–17 points above** the floor. That is a real,
content-dependent signal, and it is the difference between S2-C0's NO-GO and this
stage's AMBIGUOUS. It is also the reason the verdict is not NO-GO despite
condition 3 failing.

---

## 6. The single robustness check — and why it is not d=128

The brief's ambiguous branch allows one robustness check, naming "d=128 / layer 4"
as the example. That check was **not** run, and the reason is recorded here rather
than left implicit:

* Every arm on that axis is already measured and flat. Across LIN/QRY × L2/L4,
  held-out Top-256 overlap spans 0.554–0.563 (±0.9 % relative) and accuracy spans
  57.32–58.09 macro. The val split that would choose between them separates the
  family winners by **0.001**. Another point on the layer / hidden-size axis
  carries no information about whether the method is viable.
* Condition 3 is the **only** failing gate condition, and it is the only
  comparison in the gate that mixes selectors: every other arm is Top-K, but
  condition 3 pits *student + Top-K* against *official + facility location*. The
  selector is worth **+18.3 points** to EADP's own score (42.75 → 61.10) and only
  **+2.6 points** to the teacher's (75.15 → 77.80). So the incumbent's advantage
  in condition 3 could be entirely a selector artifact.

The check therefore tests that one question: **run the learned score through the
same facility selector the incumbent uses.**

| arm | selector | TextVQA | DocVQA | OCRBench | macro | retained |
|---|---|---|---|---|---|---|
| official EADP (own score) | facility | 59.400 | 61.891 | 62.000 | **61.097** | — |
| `QRY_L4` (learned) | **facility** | 74.800 | 50.694 | 36.000 | **53.831** | 34.2 % |
| `QRY_L4` (learned) | topk | 69.400 | 58.875 | 46.000 | **58.092** | 47.3 % |

**The robustness check makes the verdict harder, not softer.** Facility location
makes the learned score *worse* (58.09 → 53.83, retention 47.3 % → 34.2 %), and it
still trails the incumbent by 7.3 points under the incumbent's own selector. So:

* Condition 3's failure is **not** a selector artifact. Put the learned score and
  EADP's score through the identical selector and the learned score still loses.
* Facility location helps a score that has genuine spatial diversity structure
  (EADP's, designed for it, +18.3) and the teacher's (+2.6), but not this one.
* The retained-gain denominator the brief specifies — EADP Top-K — remains the
  right one, and 45–47 % is the honest number.

The verdict is AMBIGUOUS on the brief's literal criteria and the robustness check
confirms that reading rather than overturning it.

---

## 7. Token-only vs query-conditioned

The brief calls this "very important for the final method story". The answer is
clean and slightly deflationary.

| | separability (held-out 150) | accuracy (held-out 150) |
|---|---|---|
| | AUROC | Top-256 | Spearman | macro |
| `LIN_L4` (token-only, 4 097 params) | 0.798 | 0.562 | 0.613 | 57.318 |
| `QRY_L4` (query-cond, 528 385 params) | 0.795 | 0.560 | 0.608 | 58.092 |
| **Δ** | **−0.003** | **−0.002** | **−0.005** | **+0.774** |

**129× the parameters buys +0.77 macro points and −0.003 AUROC.** The accuracy
difference is inside the bootstrap CI; the separability differences are inside
the seed-to-seed noise of a 4 k-parameter linear probe.

The reading, per the brief's own decision rule: **the gradient teacher's
preferences are largely recoverable from a visual token's own content, without
knowing the question.** The P1-G2 objective is dominated by *generic
information-bearing / text-dense token detection*, and question conditioning is
not the mechanism.

This is consistent with — and sharpens — S2-A's earlier finding that G2 is
"largely answer-agnostic" (matched vs mismatched-answer Spearman 0.75–0.86). Both
point the same way: whatever the gradient teacher is measuring, it is mostly a
property of the image region, not of the image–question pair.

**Consequence for the method story.** A query-conditioned low-rank head is not
where the value is; the honest summary of this stage is "a single linear read-out
of one layer's visual hidden states", which is about as far from an architectural
contribution as a learned component can get. That is a finding about the
*teacher*, and it constrains what a distillation-based method can claim to
contribute.

---

## 8. Gate

Criteria (brief §9): (1) retained ≥ 40 % of the teacher's Top-K gain;
(2) ≥ 2 benchmarks clearly better than EADP Top-K (margin 2.0 points);
(3) macro above the official EADP facility baseline.

| student | retained | c1 (≥40 %) | c2 (>2 benchmarks) | c3 (beats facility) | verdict |
|---|---|---|---|---|---|
| `LIN_L4` | **45.0 %** | ✅ | ✅ 3/3 | ❌ −3.779 | fail |
| `QRY_L4` | **47.3 %** | ✅ | ✅ 2/3 | ❌ −3.006 | fail |

```
S2-C1 AMBIGUOUS:
a 4 k-parameter linear probe on layer-4 hidden states keeps 45-47 % of the
gradient teacher's Top-K accuracy gain -- inside the GO band on retention,
but it does not beat the official EADP facility baseline, so the gate fails.
```

Not GO: condition 3 fails for both students. Not STRONG GO: retention is 47.3 %,
below the 60 % bar. Not NO-GO: retention is nowhere near the 20 % floor, and the
students clear the content-free floor by 15–17 points.

Per the brief, the stage **stops here**. No mid-layer prune integration, no
KV/state reuse, no second-stage student, no expansion of MLP width, hidden size,
layer set, or training tricks.

---

## 9. Cost

Measured on the unpruned 1 052-token prompt forward, A40 46 GB, SDPA, batch 1,
idle GPU. The vision tower (~117 ms) is excluded — every method pays it
identically.

| quantity | ms |
|---|---|
| full unpruned LLM prefill (1 052 tokens) | 203.2 |
| prefix forward through layer 2 (3/36 layers) | 18.8 |
| prefix forward through layer 4 (5/36 layers) | 29.4 |
| tiny scorer forward (1×1024×4096 → 1) | 0.10 |
| + Top-K 256 | 0.135 |
| pruned prefill, 284 tokens (256 visual + 28 text) | 72.8 |
| P1-G2 teacher (one forward + one backward) | 507 (S2-A) |
| reference: EADP pruning overhead, Stage-1 | 42.0 |

**A. Scoring-only latency** — what the selector costs on top of everything else:

| layer | prefix | scorer + Top-K | **total** |
|---|---|---|---|
| L2 | 18.8 | 0.135 | **18.9 ms** |
| L4 | 29.4 | 0.135 | **29.5 ms** |

**B. Naive end-to-end latency** — A, plus actually re-running the pruned model:

| layer | scoring | + pruned prefill | **total** | vs unpruned | vs EADP |
|---|---|---|---|---|---|
| L2 | 18.9 | 72.8 | **91.7 ms** | 2.22× | 114.8 ms |
| L4 | 29.5 | 72.8 | **102.3 ms** | 1.99× | 114.8 ms |

Two caveats, both per the brief:

* **The prefix forward is not amortised.** A system that scores from a prefix and
  then re-runs the pruned model pays the prefix on top of the full pruned
  prefill. Quoting 29.5 ms as the method's cost would hide 72.8 ms of it. Row B
  is the honest deployment number.
* **The pruned prefill is measured, not extrapolated**: the unselected visual
  rows are deleted from the prompt embeddings and the model is run on what
  remains, which is what the generation harness does after selection. Decode
  steps are excluded — they are identical across arms — so these are prefill
  numbers.

Note also that the prefill only shrinks 203.2 → 72.8 ms while the token count
shrinks 1 052 → 284 (3.7×). The gap is per-layer weight-traffic overhead that
pruning does not remove, and it is why the naive end-to-end speedup is 2.0× and
not 3.7×. A real mid-layer prune would remove the prefix's cost entirely and
recover some of that gap; that integration is a separate stage and was not
attempted.

---

## 10. What carries forward

1. **The learned scorer works, partially and cheaply.** A 4 097-parameter linear
   probe on frozen layer-4 hidden states recovers 45–47 % of a 507 ms/image
   gradient teacher's accuracy gain at 29.5 ms scoring cost. The S2-C0 wall was
   not fundamental to "forward-only" — it was fundamental to the *attention*
   family. Hidden states carry information that attention rows do not.

2. **Retention is convex in Top-256 overlap, and the last stretch is the
   valuable one.** 0.35 overlap → ~5 % retention; 0.56 → ~46 %; 1.00 → 100 %.
   There is no "agree with the teacher on most tokens" shortcut to most of the
   gain, and future work should treat the *tail* of the teacher's Top-256 — not
   its bulk — as the target.

3. **Token-only ≈ query-conditioned, so the teacher is mostly generic.** 129×
   parameters for +0.8 points (inside CI). Combined with S2-A's matched-vs-
   mismatched result, the P1-G2 signal is largely a function of the image region,
   not the image–question pair. Any method story built on "the gradient knows
   what *this question* needs" is contradicted by both stages.

4. **n=8 causal AUROC is not merely low-resolution — it inverts.** `C_L2` has
   better causal AUROC than `LIN_L4` (0.705 vs 0.666) and 12 points *worse*
   accuracy. Three stages have now failed to make this tier predict accuracy, and
   it should be dropped as a selection criterion entirely.

5. **The remaining gap is capacity, not data.** Fit ≈ val ≈ held-out, training
   stops at epoch 5–8, and 129× parameters change nothing. A linear read-out of
   one early layer saturates at ~0.56 overlap. S2-C0's observation that teacher
   agreement rises monotonically with depth (attention family, L1→L8) is the one
   untested direction that the evidence actually supports — but the brief fixed
   the layer set at {2, 4}, and this stage does not extend it.

6. **Condition 3 is the open question.** The students beat EADP's own score by
   +14.6/+15.3 points under a matched selector (both CIs clear of zero) and lose
   to it by ~3 points when the incumbent is allowed its own selector. Whether the
   method's viability is judged on the former or the latter is a decision about
   what the method has to replace — a score, or a score-plus-selector pair — and
   this stage reports both without resolving it.

The brief's next-stage decision is left open by design. Nothing in S2-C1 argues
that mid-layer integration would fail; what it establishes is that the score
alone, at this capacity, is not yet good enough to replace the incumbent
end-to-end.

---

## 11. Reproduce

```
source ~/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
cd /media/disk2/YZX/research/EADP/Qwen_vl

# 1. cache prefix features (465 instances, no backward) + prefix latency
python scripts/discovery/s2c1_features.py --latency

# 2. train the four tiny scorers (LLM not loaded; features are pre-cached)
python scripts/discovery/s2c1_train.py --latency

# 3. held-out separability (tier 1) + causal 15 (tier 2)
python scripts/discovery/s2c1_eval.py

# 4. held-out accuracy translation for the two val-selected arms
python scripts/discovery/s2c0_run.py --scores s2c1_scores.npz \
    --arms LIN_L4 QRY_L4 --tag s2c1_pilot

# 5. the single robustness check (learned score through facility location)
python scripts/discovery/s2c0_run.py --scores s2c1_scores.npz \
    --arms QRY_L4 --selector facility --tag s2c1_robust

# 6. latency, then retained gain + gate
python scripts/discovery/s2c1_latency.py
python scripts/discovery/s2c1_consolidate.py --select LIN_L4,QRY_L4 \
    --extra s2c1_robust.json
```

Or in one step: `bash scripts/discovery/run_s2c1.sh`.

Deliverable: `outputs/discovery/s2c1_distillation.json` — the held-out accuracy
table with per-dataset deltas and paired bootstrap CIs against both references,
the gate outcomes, the robustness arm, the tier-1/tier-2 tables, per-arm training
records, and the latency block. Supporting: `s2c1_features.json`,
`s2c1_train.json`, `s2c1_scores.npz` (1 860 score vectors = 4 arms × 465
instances), `s2c1_eval.json`, `s2c1_pilot.json`, `s2c1_robust.json`,
`s2c1_latency.json`, `s2c1_<ARM>.pt` (trained weights),
`s2c1_{feats,query}_L{2,4}.npy` (feature caches, 7.8 GB fp16).
