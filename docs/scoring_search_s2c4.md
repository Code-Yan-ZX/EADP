# S2-C4 — Per-Image Direction Transfer Diagnosis

**Scope.** Stage-2 S2-C4. One question, and it is a mechanism question rather than
a design step:

> The teacher's high-value ranking direction — is it **image-dependent**, and is
> the per-image variation **predictable from cheap forward information**?

S2-C3 left exactly this object behind. A direction fitted **on the image itself**
recovers 98.2 % of the teacher's Top-8; the best direction fitted across 240
images reaches 76.6 %. S2-C3 showed the loss was aimed wrong, fixed the aim, and
bought three points of head recall — not the twenty-odd the exchange rate needs.
So the remaining explanation is either that the right direction differs per image
and could be *chosen* per image, or that it does not and only set-dependent
scoring is left.

No method is designed here. Nothing is trained, no architecture is changed, no
scorer is fitted, no query conditioning is added, no multi-layer feature is
built. Every quantity is a closed-form read-out of the cached S2-C1 layer-4
hidden states plus the P1-G2 teacher ranking, and the whole stage is CPU-only.

**Status.** Complete. Scripts:
`Qwen_vl/scripts/discovery/s2c4_{common,build,transfer,geometry,pca,verify,consolidate,figure}.py`.
Deliverables: `outputs/discovery/s2c4_direction.json` (main),
`s2c4_transfer.json`, `s2c4_geometry.json`, `s2c4_pca.json`, `s2c4_build.json`,
`s2c4_dirs.npz` (the 450 oracle directions), `s2c4_transfer.npz` (the transfer
matrix), `s2c4_geometry.npz`, `s2c4_pca.npz`,
`figures/s2c4_direction_transfer.png`.

---

## VERDICT

```
S2-C4 C -- NON-TRANSFERABLE.

The per-image oracle direction is strong: fitted on the image itself it keeps
0.950 of the teacher's Top-8 inside the selected 256, against 0.712 for the best
direction that adapts to nothing at all. So the value-carrying direction really
does differ from image to image, by 0.238 recall.

And that variation is not recoverable. Retrieving a direction by image
similarity, at every descriptor and every neighbourhood size tested, lands
between 0.639 and 0.734. The best of them -- the mean of the 8 nearest fit
directions under the mean+std+delta descriptor -- reaches 0.734, which beats the
no-adaptation direction by 0.0225 [+0.0008, +0.0433] and is still 0.0442 BELOW
the shared direction S2-C3 trained [-0.0750, -0.0142]. Predicting the direction
from a cheap descriptor with a ridge regressor does worse still: 0.651-0.660,
i.e. below the no-adaptation direction.

The geometry says why. The 240 fit directions are not low-dimensional -- mean
pairwise cosine 0.180, effective rank 43.5 of 240, 96 components for 80 % of the
variance, PC1 carrying 10.4 % -- and even an ORACLE coefficient vector projected
onto that basis retains only 0.832 of the in-sample direction's recall at k = 32
and 0.872 at the full k = 240. The per-image direction has a large component that
lives outside anything the other images span.

So: the right direction exists for almost every image (95 % of held-out images
have some candidate direction above 0.85) and the descriptors identify it only
weakly -- enough to beat chance, nowhere near enough to beat a shared learned
direction. The failure is identification, not existence, and it is not fixable by
choosing a direction per image.
```

The single most useful number is the sign of the last comparison: **the trained
shared direction beats the best per-image adaptive rule.** A stage that found a
predictable per-image component would show the opposite ordering.

---

## 0. Executive summary

1. **The oracle is strong and it is image-specific.** On the held-out 150, a
   mean-difference direction fitted on the image's own teacher Top-32 keeps
   **0.950** of the teacher's Top-8 and **0.9025** of its Top-32. The
   best single direction that adapts to nothing (the mean of all 240 fit
   directions) keeps 0.712 and 0.625. The gap, **+0.238 [+0.203, +0.273]**, is
   the per-image structure this stage was built to chase (§2).
2. **Cross-image transfer is weakly positive and far too small.** The full
   transfer matrix — every (held-out image, fit direction) pair, 150 × 240 — has
   mean **0.5435**, median 0.50, sd 0.258. A randomly chosen other-image
   direction is therefore *better than nothing* (0.544 vs 0.25 chance) and
   *worse than no adaptation at all* (0.712). Retrieval improves on the random
   column by +0.096 and on the no-adaptation direction by at most +0.0225 (§2).
3. **No deployable arm reaches the shared trained direction.** `HEAD_RANK`
   (S2-C3, seed-averaged direction) sits at **0.778**; the best of the 16
   deployable arms sits at **0.734**, with a paired CI of
   **[−0.0750, −0.0142]** against it. The trained shared direction wins by a
   margin that excludes zero (§4).
4. **The directions are not low-dimensional.** Effective rank 43.5 of 240; 25
   components for 50 % of the variance, 96 for 80 %, 144 for 90 %. PC1 carries
   10.4 % and **60.5 % of PC1's own variance is benchmark identity** — the
   leading axis of variation across images is "which benchmark is this", not
   anything image-specific (§3).
5. **Even the optimistic low-dimensional reconstruction stops at 0.83.** Give the
   held-out image's true PCA coefficients (an upper bound that reads its teacher
   map) and truncate to k components fitted on other images: 0.791 at k = 32,
   0.806 at k = 64, **0.828 at the full k = 240**. That is 0.872 of the in-sample
   direction's power: 0.122 recall points go missing to the subspace truncation
   alone, with no prediction error involved at all (§5).
6. **Predicting those coefficients costs another 0.17.** A ridge regression from
   a cheap descriptor to the PCA coefficients reaches **0.651–0.660**, below the
   no-adaptation direction's 0.712. The two failures compound: the low-dimensional
   family is already lossy, and the predictable part of it is lossy again (§5).
7. **The right direction exists for almost every image.** For 95.3 % of held-out
   images some one of the 240 candidate directions keeps ≥ 0.85 of the teacher's
   Top-8, and for 73.3 % some candidate keeps ≥ 0.95; the median per-image best
   is a perfect 1.000. The median per-image *mean* is 0.556. So the candidates
   contain the answer and the descriptor cannot find it (§2.3).
8. **The one resolved positive is a tenth of the gap.** KNN improves on
   no-adaptation by +0.0225 out of a 0.238 gap, and nearest-neighbour retrieval
   *hurts* — single-nearest is 0.639, significantly below the 0.712 it was
   supposed to beat. Whatever the descriptors carry, it is only visible after
   averaging, which is the signature of noise reduction rather than adaptation
   (§4.2).

---

## 1. What is held fixed, and what the objects are

Nothing is trained and nothing is tuned. The objects:

| object | value | source |
|---|---|---|
| feature cache | layer-4 visual hidden states, 1024 × 4096 fp16 | `s2c1_feats_L4.npy` |
| preprocessing | one fixed step: fit-split per-dimension mean/std | matches S2-C1/C2/C3 |
| teacher | P1-G2 gradient map (`s2b_gradient_scores.npz`) | S2-B |
| split (image-level) | fit 240 / val 60 / **held-out 150** / causal 15 | `s2c1_features.json` |
| instances with a teacher map | **450** (the causal 15 have none and drop out) | S2-C1's rule |
| budget | Top-K @ 256, identity calibration | S2-B / S2-C1 |

For image `i`, with `h_i` the standardised layer-4 block (1024 × 4096) and `pos`
the teacher's Top-k indices:

```
w_i  =  mean(h_i[pos]) − mean(h_i[¬pos])         mean-difference / centroid
w_i ←  w_i / ‖w_i‖₂                              unit direction
```

**Top-32 is the primary oracle**, not Top-8, and the reason is S2-C3's: at Top-8
a direction fitted on the teacher's own head beats one fitted on an arbitrary 8
by only 0.6 of that control's own spread (0.982 vs 0.921 ± 0.107), while at
Top-32 the separation is 1.6 spreads (0.903 vs 0.691 ± 0.127). The Top-8
direction is carried as a supplementary.

A direction is used only to score: `score_{j,t} = w_i · h_{j,t}`. Every direction
is unit-norm, so scores are on a common scale and no calibration is needed —
S2-B established calibration is a monotone no-op for Top-K.

**The identity check comes first.** `s2c4_build.py` recomputes S2-C3 §6 on the
held-out 150 from a completely separate code path and reproduces it to four
decimals: Top-8 direction → recall 0.9817 (S2-C3: 0.9817), Top-32 direction →
recall@32 0.9025 (S2-C3: 0.9025), Top-8 agreement 0.4267 (S2-C3: 0.4267). The
objects this stage diagnoses are the objects S2-C3 measured. `s2c4_verify.py`
additionally recomputes 25 random transfer-matrix cells with a literal
argsort-and-sets implementation and requires exact agreement before the verdict
is computed — it caught one real bug during development (an early version scored
`head_agree@k` against the teacher's Top-256 instead of its Top-k).

### 1.1 The arms

Every arm is a rule for turning image `j`'s own cheap information into a
direction. Nothing below reads image `j`'s teacher map except the two arms
marked optimistic.

| arm | direction used for image j | reads j's teacher map |
|---|---|---|
| **`SELF`** | `w_j` itself | **yes — upper bound** |
| `SELF_TOP8DIR` | `w_j` from the Top-8 target | **yes — upper bound** |
| `GLOBAL` | mean of the 240 fit directions | no |
| `GLOBAL_DS__<b>` | mean of the fit directions in j's benchmark | no |
| `TRAINED_*` | S2-C3's trained shared direction | no |
| `NN__<desc>` | the single nearest fit image by descriptor | no |
| `KNN{k}__<desc>` | mean of the k nearest fit directions, k ∈ {1, 4, 8} | no |
| `PCA_RIDGE__<desc>` | ridge from descriptor → PCA coefficients → direction | no |

The descriptors are the cheap forward statistics the brief allows, all computed
from `h_j` alone, z-scored on fit-set statistics before any distance is taken:

```
mean(h_L4)        4096-d
std(h_L4)         4096-d
mean+std          8192-d
meanstd+delta    12288-d      delta = mean(h_L4) − mean(h_L2), S2-C1 cache
```

No query conditioning. No pooling over the question. No hidden layer anywhere.

### 1.2 The decision rule, fixed before the numbers were read

The rule lives in the docstring of `s2c4_consolidate.py`, in the file that
applies it.

* `resolved(x, y)` — paired bootstrap (10 000 resamples, resampling the 150
  held-out instances) on per-instance `head_recall@8`; the 95 % CI on the mean
  difference excludes zero.
* `material(x)` — mean `head_recall@8` ≥ **0.80**. S2-C3's best shared direction
  is 0.766 and the brief names 0.82–0.85 as the level at which a downstream run
  becomes worth paying for, so 0.80 is the point below which nothing has changed
  regime.

| outcome | condition |
|---|---|
| **A** LOW-DIMENSIONAL IMAGE-ADAPTIVE | `BEST_DEPLOY` ≥ 0.80, resolved above `GLOBAL`, **and** the PCA basis is genuinely low-dimensional (oracle reconstruction retains ≥ 0.90 of `SELF` at k ≤ 64) |
| **B** TRANSFERABLE CLUSTERS | `BEST_DEPLOY` ≥ 0.80, resolved above `GLOBAL`, and the winning arm is a retrieval arm |
| **C** NON-TRANSFERABLE | `BEST_DEPLOY` < 0.80, or it fails to beat `GLOBAL` with a resolved positive CI |
| **D** AMBIGUOUS | anything else |

The brief's downstream gate is applied here too and recorded either way
(§6).

---

## 2. The cross-image transfer matrix

The stage's central object is the full matrix, not a mean:

```
M[j, i] = head_recall@8 of  w_i  scored on image j
          j ∈ held-out 150      i ∈ fit 240
```

| statistic | value |
|---|---|
| mean | **0.5435** |
| median | 0.500 |
| sd | 0.258 |
| p05 / p95 | 0.125 / 1.000 |
| by benchmark | TextVQA 0.593 · OCRBench 0.552 · DocVQA 0.485 |
| chance (random 256 of 1024) | 0.250 |

A randomly chosen other-image direction keeps 0.544 of the teacher's head — above
the 0.25 chance rate, and **well below the 0.712 that no adaptation at all
achieves**. Averaging directions is worth more than choosing one.

### 2.1 The distribution is lumpy, with a real spike at the top

The matrix is not a spread around a middle — `head_recall@8` takes only nine
values, so the distribution is discrete and lumpy, with a real spike at 1.0. Two
readings, and the stage needs both:

* the *existence* of a direction that works is common. **6.2 %** of the 36 000
  pairs keep the teacher's entire Top-8; at a 256/1024 budget the chance rate for
  8 of 8 is 0.25⁸ ≈ 1.5 × 10⁻⁵, so that spike is far beyond coincidence;
* the *identity* of that direction is not predictable. If it were, retrieval
  would find it (§4).

### 2.2 The arms

Seed-mean-free: each arm produces one score vector per held-out image.

| arm | R@8 | R@16 | R@32 | A@8 | ov256 | Δ vs `GLOBAL` | 95 % CI |
|---|---|---|---|---|---|---|---|
| **`SELF`** *(optimistic)* | **0.9500** | 0.9304 | 0.9025 | 0.2475 | 0.5512 | +0.2383 | [+0.2033, +0.2733] |
| `SELF_TOP8DIR` *(optimistic)* | 0.9817 | 0.8788 | 0.7881 | 0.4267 | 0.5081 | +0.2700 | [+0.2367, +0.3042] |
| `HEAD_RANK` *(S2-C3)* | **0.7783** | 0.7567 | 0.7071 | 0.2708 | 0.5220 | +0.0667 | [+0.0392, +0.0942] |
| `HEAD_BIN` *(S2-C3)* | 0.7692 | 0.7454 | 0.7015 | 0.2633 | 0.5250 | +0.0575 | [+0.0308, +0.0842] |
| `S2C1_LIN_L4` | 0.7358 | 0.7188 | 0.6813 | 0.1917 | 0.5618 | +0.0242 | [−0.0025, +0.0500] |
| **`KNN8__meanstd+delta`** | **0.7342** | 0.6846 | 0.6304 | 0.2000 | 0.4777 | **+0.0225** | **[+0.0008, +0.0433]** |
| `KNN4__std` | 0.7333 | 0.6800 | 0.6271 | 0.1817 | 0.4708 | +0.0217 | [−0.0017, +0.0458] |
| `KNN8__std` | 0.7325 | 0.6804 | 0.6275 | 0.2017 | 0.4839 | +0.0208 | [−0.0017, +0.0433] |
| `KNN8__mean` | 0.7300 | 0.6783 | 0.6256 | 0.2008 | 0.4771 | +0.0183 | [−0.0033, +0.0392] |
| `KNN8__mean+std` | 0.7300 | 0.6833 | 0.6300 | 0.2033 | 0.4778 | +0.0183 | [−0.0008, +0.0375] |
| `KNN4__mean` | 0.7258 | 0.6725 | 0.6190 | 0.1900 | 0.4693 | +0.0142 | [−0.0083, +0.0367] |
| `KNN4__meanstd+delta` | 0.7175 | 0.6687 | 0.6165 | 0.1942 | 0.4686 | +0.0058 | [−0.0158, +0.0267] |
| **`GLOBAL`** | **0.7117** | 0.6758 | 0.6246 | 0.2300 | 0.4879 | — | — |
| `KNN4__mean+std` | 0.7058 | 0.6596 | 0.6121 | 0.1867 | 0.4683 | −0.0058 | [−0.0300, +0.0183] |
| `GLOBAL_DS__OCRBench` | 0.6983 | 0.6617 | 0.6027 | 0.2325 | 0.4680 | −0.0133 | [−0.0283, +0.0017] |
| `GLOBAL_DS__TextVQA` | 0.6967 | 0.6467 | 0.5992 | 0.2033 | 0.4522 | −0.0150 | [−0.0408, +0.0092] |
| `NN__std` | 0.6408 | 0.5983 | 0.5577 | 0.1317 | 0.4333 | −0.0708 | [−0.1058, −0.0375] |
| `NN__mean+std` | 0.6408 | 0.5938 | 0.5542 | 0.1417 | 0.4344 | −0.0708 | [−0.1092, −0.0342] |
| `NN__mean` | 0.6392 | 0.6008 | 0.5669 | 0.1600 | 0.4432 | −0.0725 | [−0.1108, −0.0358] |
| `NN__meanstd+delta` | 0.6392 | 0.5921 | 0.5577 | 0.1508 | 0.4367 | −0.0725 | [−0.1117, −0.0350] |
| `GLOBAL_DS__DocVQA` | 0.6092 | 0.5763 | 0.5373 | 0.2142 | 0.4594 | −0.1025 | [−0.1300, −0.0767] |

Four things to read off it:

1. **Nothing deployable beats the trained shared direction.** The best is 0.7342
   against `HEAD_RANK`'s 0.7783, a **−0.0442 [−0.0750, −0.0142]** deficit.
2. **Nearest-neighbour retrieval is actively harmful.** All four single-nearest
   arms land at 0.639–0.641, *significantly below* the 0.712 of adapting to
   nothing. The descriptor's notion of "similar image" points at directions that
   are worse than average.
3. **The neighbourhood must be large to help at all.** k = 1 is −0.071, k = 4 is
   ≈ +0.02, k = 8 is ≈ +0.02. The improvement from 1 to 8 is the signature of
   averaging away selection noise rather than of finding a better direction.
4. **Benchmark conditioning does not help either.** `GLOBAL_DS` — the direction
   restricted to the image's own benchmark — is *worse* than the global mean for
   all three benchmarks (−0.013, −0.015, −0.103). The per-image signal is not
   "which benchmark am I".

### 2.3 The right direction exists but is not identifiable

Per held-out image, over the 240 candidate directions:

| | median | range |
|---|---|---|
| mean over candidates | 0.556 | p05 0.330 – p95 0.726 |
| **best** over candidates | **1.000** | min 0.750 |

* **95.3 %** of held-out images have some candidate direction above 0.85;
  **73.3 %** have one above 0.95.
* The best that any descriptor-based rule achieves on the same images is 0.734.

The `best` column is a maximum over 240 draws and is inflated by selection — but
the inflation is bounded and small here, because the underlying per-pair
distribution is genuinely bimodal (§2.1) rather than a narrow unimodal spread
whose max would be dominated by tail noise. The honest statement is: for nearly
every image, *a* direction in the fit pool recovers the teacher's head, and the
cheap descriptors cannot tell which one.

---

## 3. Direction geometry

Fit on the 240 fit directions (unit vectors in R⁴⁰⁹⁶).

| statistic | value |
|---|---|
| mean pairwise cosine | **0.180** |
| cosine p05 / p95 | −0.030 / 0.415 |
| within benchmark / across benchmark | 0.2455 / 0.1481 |
| mean cosine to the global mean direction | 0.4286 |
| **effective rank** (participation ratio) | **43.5** of 240 |
| components for 50 / 80 / 90 / 95 / 99 % variance | 25 / **96** / 144 / 179 / 222 |
| PC1 / PC2 / PC3 explained variance | 10.40 % / 6.15 % / 4.39 % |
| **fraction of PC1's variance = benchmark identity** | **60.5 %** |
| k-means purity wrt benchmark (k = 3 / k = 8) | 0.663 / 0.713 |

Three readings:

* **Not low-dimensional.** A 240-point sample in R⁴⁰⁹⁶ has at most rank 240; an
  effective rank of 43.5 and 96 components for 80 % of the variance means the
  directions are spread across the space rather than confined to a small
  subspace. PC1 at 10.4 % is not a dominant axis.
* **The leading axis is the benchmark, not the image.** 60.5 % of PC1's variance
  is between-benchmark, within-benchmark cosine (0.246) is 1.7× across-benchmark
  cosine (0.148), and a 3-means partition already achieves 0.66 purity against
  the benchmark label. The most salient thing about an image's oracle direction
  is which dataset it came from — and §2.2 shows exploiting exactly that
  (`GLOBAL_DS`) makes things worse, because it throws away the other two
  benchmarks' 160 images of averaging.
* **No clustering claim is made.** Purity 0.663 at k = 3 against a 3-class label
  with 80/80/80 instances is above chance (≈ 0.33) but this is k-means on a
  projection chosen for variance, not a cluster structure discovered
  independently, and it is reported as a descriptor of the geometry rather than
  as evidence of routing structure. The brief asks explicitly not to manufacture
  clustering, and this is not one.

---

## 4. Can cheap image descriptors predict transfer?

§2.2 answers it, but the mechanism deserves separating out, because two different
failures both produce a flat result and they license different next stages.

### 4.1 The descriptor does carry something

* `NN__mean` (0.639) beats the expected score of a randomly chosen fit direction
  (the matrix mean, 0.544) by **+0.096**. If the descriptor were uninformative,
  its nearest neighbour would be a random column.
* `KNN8__meanstd+delta` (0.734) beats `GLOBAL` (0.712) by **+0.0225
  [+0.0008, +0.0433]**, marginally clearing zero.

So "the descriptors contain no information about which direction to use" is
false, and should not be written.

### 4.2 …but not enough, and not the right kind

Three facts bound the effect, and together they are what makes the verdict C
rather than D:

1. **The resolved gain is one tenth of the gap it is chasing.** +0.0225 against a
   per-image gap of 0.238 (§2).
2. **Single-nearest retrieval is below no-adaptation** (−0.0725, CI clear of
   zero). A rule that identifies *the* most similar image does worse than not
   choosing at all, which is the opposite of what cluster/routing structure would
   predict.
3. **The trained shared direction beats every deployable arm**
   (−0.0442 [−0.0750, −0.0142]). This is the decisive comparison. A shared
   direction learned from 240 images extracts more of the per-image structure
   than any descriptor-based selection rule over those same 240 directions.

The natural reading: the descriptor's similarity is real but correlates with the
*benchmark and style* of the image, not with the component of `w_i` that carries
the head — and after that component is averaged over enough neighbours the
selection noise cancels and what is left is a slightly denoised version of the
global direction.

One caveat stated plainly: with 16 deployable arms tested and one of them
clearing zero at [+0.0008, +0.0433], a single marginal rejection is exactly what
multiple comparisons would produce by chance. The KNN gain is reported as
"barely resolved" and nothing in the verdict depends on it.

---

## 5. The low-dimensional reconstruction test

Two levels, and the gap between them is the point.

**`PCA_ORACLE(k)` — optimistic, reads image j's teacher map.** Project `w_j` onto
the fit-set PCA basis truncated to k components, reconstruct, score. This is an
upper bound: it assumes the coefficients are known exactly.

**`PCA_RIDGE(k)` — deployable.** Predict those k coefficients from image j's
cheap descriptor with a ridge regression fitted on the fit split. k and α are
chosen on the **validation** 60, never on the held-out 150.

| k | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 240 |
|---|---|---|---|---|---|---|---|---|---|
| `PCA_ORACLE` R@8 | 0.476 | 0.566 | 0.647 | 0.700 | 0.746 | **0.791** | 0.806 | 0.819 | **0.828** |
| retention of `SELF` | 0.501 | 0.596 | 0.681 | 0.737 | 0.785 | 0.832 | 0.848 | 0.862 | 0.872 |

| descriptor | dim | chosen k | `PCA_RIDGE` R@8 | vs `GLOBAL` 0.712 |
|---|---|---|---|---|
| mean | 4096 | 240 | 0.6508 | −0.061 |
| std | 4096 | 240 | 0.6517 | −0.060 |
| mean+std | 8192 | 128 | 0.6525 | −0.059 |
| meanstd+delta | 12288 | 240 | **0.6600** | −0.052 |

Three results:

1. **The low-dimensional family is already lossy before any prediction.** Even
   with the true coefficients, a 240-component basis fitted on other images caps
   at 0.828 against `SELF`'s 0.950. The 0.122 that goes missing is the part of
   `w_j` orthogonal to the span of the other 239 directions — pure
   image-specificity, with zero estimation error involved.
2. **The deployable predictor is worse than not adapting.** Every ridge arm sits
   at 0.651–0.660, below `GLOBAL`'s 0.712. Predicting per-image coefficients from
   a cheap descriptor actively hurts.
3. **The two losses compound.** 0.950 → 0.828 (truncation) → 0.660 (prediction).
   Neither half is adequate and the product is far from it.

The oracle curve also confirms §3 from a second direction: if the directions had
a genuine k ≈ 32 structure, the oracle curve would be flat after k = 32. It keeps
climbing to k = 240, which is the same statement as "effective rank 43.5, 96
components for 80 % variance".

---

## 6. Downstream: the gate was not met, and was not run

The brief is explicit that held-out generation is paid for only if a deployable
arm lifts `head_recall@8` clearly above S2-C3's ~0.766, ideally to 0.82–0.85.
The gate is implemented in `s2c4_consolidate.py` and its outcome is recorded in
the JSON either way:

```
downstream_gate:
  threshold 0.766   ideal 0.82
  best_deployable  KNN8__meanstd+delta
  best_deployable_recall8  0.7342
  clears_s2c3   false
  reaches_ideal false
  downstream_run  false
```

**No held-out generation was run**, so this stage reports no accuracy table, by
design and per the brief. This is a real saving rather than a gap: 150 instances
× 3 benchmarks × 12 arms is the most expensive part of an S2 stage, and the
head-retention half of the experiment had already excluded every arm that would
have been run. S2-C3 measured the same design's downstream null at ±2–3 macro
points, so even an arm that cleared the gate by 0.02 would have been unreadable.

The references the doc would have reported against, unchanged from S2-C3 and
quoted here for continuity: S2-C1 `LIN_L4` macro **57.318**, official EADP
facility **61.097**, P1-G2 teacher **75.150**, EADP's own score under Top-K
**42.753**.

---

## 7. Decision

Applying the pre-registered rule of §1.2:

| quantity | value |
|---|---|
| `SELF` (optimistic upper bound) | 0.9500 |
| `GLOBAL` (no adaptation) | 0.7117 |
| `BEST_TRAINED` (`HEAD_RANK`, S2-C3) | 0.7783 |
| `BEST_DEPLOY` (`KNN8__meanstd+delta`) | **0.7342** |
| `BEST_DEPLOY` vs `GLOBAL` | +0.0225 [+0.0008, +0.0433] — resolved |
| `BEST_DEPLOY` vs `BEST_TRAINED` | −0.0442 [−0.0750, −0.0142] |
| `BEST_DEPLOY` ≥ 0.80 (material)? | **no** |
| PCA basis low-dimensional? | **no** (effective rank 43.5; oracle retains 0.832 of `SELF` at k = 32, below the 0.90 bar) |

`BEST_DEPLOY` < 0.80 and does not reach the trained shared direction, so the rule
returns **C**.

```
S2-C4 C -- NON-TRANSFERABLE.
```

**What this licenses, and what it does not.** The brief's C branch says: self
oracle strong, no stable structure between directions, cheap descriptors cannot
predict, cross-image transfer near the shared baseline — and gates the next stage
on token-context / set-dependent interaction. The measurement supports the gate
and sharpens the reason in one respect worth carrying forward:

* It is **not** that there is nothing to adapt to. The per-image gap is 0.238 and
  it is large.
* It is **not** that the descriptors are empty. Nearest-neighbour beats a random
  column by 0.096 and 8-NN beats no-adaptation by 0.0225.
* It **is** that the adaptable part is small, the identifiable part is smaller,
  and **a single shared learned direction already captures more of it than any
  per-image selection rule over the same pool** (§4.2). The per-image component
  that carries the head is largely orthogonal to what other images span (§5) and
  is not visible in the image's own mean or std.

So the next stage is not licensed to look for a better *rule for choosing a
direction*. It is licensed to change what a token's score may depend on — which
is the other half of S2-C2's §8.3.

---

## 8. What this stage does not claim

* **Nothing here is a method.** No scorer was trained, no selector was changed,
  no architecture was built, no query information was used, no hidden layer
  exists in any arm. The two ridge regressors are closed-form and are reported as
  diagnostics; the best of them loses to a constant direction.
* **The oracle is not a proposal.** `SELF` and `SELF_TOP8DIR` read the held-out
  image's own teacher map and are labelled optimistic in every table. The teacher
  costs 507 ms/image (S2-A), 2.5× the unpruned prefill they would accelerate; an
  arm that needs it is a measuring instrument.
* **`best over candidates` is selection-inflated.** §2.3's "95 % of images have a
  candidate above 0.85" is a maximum over 240 draws. It is reported because the
  gap between it (1.000) and what any rule extracts (0.734) is the stage's
  subject, not because a max is an achievable score.
* **One marginal rejection is not a mechanism.** The KNN gain's CI is
  [+0.0008, +0.0433]; with 16 deployable arms this is within what multiple
  comparisons produce. The verdict does not rest on it — it rests on the
  comparison against `BEST_TRAINED`, which is comfortably negative.
* **Cluster purity is descriptive.** §3's 0.663 purity at k = 3 is offered as a
  property of the geometry, not as discovered routing structure.

---

## 9. Cost

Measured end to end: the stage's artifacts span **20:53 → 21:14**, about
**21 minutes** of CPU wall clock for a full run. Per-step figures below are the
dominant cost, not independently instrumented timings.

| step | dominant cost |
|---|---|
| `s2c4_build.py` | two streaming passes over the 3.9 GB feature memmap (layer 4 and layer 2) |
| `s2c4_transfer.py` | 150 × 240 direction-token products over a (150, 1024, 240) block |
| `s2c4_geometry.py` | one SVD of 240 × 4096, plus the k-means partitions |
| **`s2c4_pca.py`** | **the stage's bottleneck**: 4 descriptors × 9 k × 6 α ridge fits. In dual form each fit is n³ with n = 240; in primal form the 12 288-d descriptor costs d³ and the run does not finish in hours |
| `s2c4_verify.py` | 25 brute-force cells, each a full 1024 × 4096 product |
| `s2c4_consolidate.py`, `s2c4_figure.py` | negligible |

**The LLM is never loaded.** No GPU was used; the whole stage ran on CPU. The
generation harness — the expensive part of every previous stage — was not
invoked, because the gate in §6 was not met.

---

## 10. What carries forward

Five findings, in the order a next stage should rely on them.

1. **The per-image oracle direction is real and large.** 0.950 against 0.712 for
   no adaptation, on the same features and the same teacher. The value-carrying
   direction genuinely differs per image. This is the cleanest positive object
   the project has produced since S2-B, and it is the ceiling any method is
   aiming at.
2. **It is not identifiable from cheap forward statistics.** Sixteen deployable
   arms, four descriptors, three neighbourhood sizes, two oracle-coefficient
   predictors; the best reaches 0.734 and loses to a constant learned direction
   by 0.044. Do not build a direction-selection rule on `mean(h)`/`std(h)`.
3. **The directions are not low-dimensional.** Effective rank 43.5, 96
   components for 80 % of the variance, and an oracle projection onto the
   fit-set basis caps at 0.828. A mixture/routing design over a small number of
   direction prototypes is therefore not supported either — not because routing
   is unexplored, but because the thing being routed to does not live in a small
   subspace.
4. **The leading axis of variation across images is the benchmark.** 60.5 % of
   PC1 is between-benchmark and exploiting it makes things worse. Useful as a
   warning: a method that adapts on easily-visible image style will be adapting
   to the wrong thing.
5. **The head is preserved by the teacher's own top-32 direction but only at
   recall 0.90, and the S2-C3 shared scorer gets 0.71.** The 0.19 between them
   is not closable by any per-image linear direction; by §5 it is not even
   *representable* in a basis fitted on other images.

The natural next object is therefore the one S2-C2 raised and this stage leaves
standing: a score that depends on the rest of the selected set. Nothing here
speaks against it, and two things speak for it — the per-image component is not
expressible as a shared direction, and S2-C2's leave-one-out result already
showed the rescue is collective rather than carried by any single token.

No S2-C5 is designed, started, or implied by this document.

---

## 11. Reproduce

```
source ~/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
cd /media/disk2/YZX/research/EADP/Qwen_vl

# 1. oracle directions + cheap descriptors + standardisation stats
python scripts/discovery/s2c4_build.py --tag s2c4

# 2. the cross-image transfer matrix and every fit-set-derived arm
python scripts/discovery/s2c4_transfer.py --tag s2c4

# 3. direction geometry: cosine structure, PCA spectrum, benchmark conditioning
python scripts/discovery/s2c4_geometry.py --tag s2c4

# 4. the low-dimensional test: oracle coefficient bound vs deployable ridge
python scripts/discovery/s2c4_pca.py --tag s2c4

# 5. identity check: vectorised metrics vs brute force, plus the two S2-C3 §6
#    calibration points. Exits non-zero on any disagreement.
python scripts/discovery/s2c4_verify.py --tag s2c4

# 6. apply the pre-registered decision rule and write the main JSON
python scripts/discovery/s2c4_consolidate.py --tag s2c4

# 7. figure
python scripts/discovery/s2c4_figure.py
```

Or in one step: `bash scripts/discovery/run_s2c4.sh`. CPU only; the LLM is never
loaded. Step 8 of that script is the commented-out downstream command, to be used
only if `s2c4_direction.json → downstream_gate.downstream_run` is ever true.

Deliverables: `s2c4_direction.json` (verdict, the decision rule's inputs, the
geometry and transfer blocks, all 35 arm rows, the ridge curves, the downstream
gate), `s2c4_transfer.json` / `.npz` (the 150 × 240 matrix and per-arm
per-instance metrics), `s2c4_geometry.json` / `.npz` (spectrum, cosine structure,
basis), `s2c4_pca.json` / `.npz`, `s2c4_build.json`, `s2c4_dirs.npz` (450 oracle
directions, 60 MB), and `figures/s2c4_direction_transfer.png`.
