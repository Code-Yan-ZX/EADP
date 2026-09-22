# S2-B — Gradient sensitivity → actual pruning accuracy

**Scope.** Stage-2 S2-B. One question: *does the very strong gradient-sensitivity
ranking from S2-A, actually used to prune visual tokens, improve downstream
answer accuracy?* No latency work, no partial backward, no surrogate, no new
scoring objective.

**Setting.** Frozen Stage-1 Part 2 sample bank, `T = 256`, score = **P1 +
gradient×input (G2)** — no answer, no GT, no generated continuation, one prompt
forward + one backward.

**Status.** Complete.

---

## VERDICT

```
S2-B GO
```

Both the pilot and the full frozen set clear the gate by a wide margin, and the
effect survives two controls that were run specifically to try to kill it: an
exact harness-identity reproduction of the Stage-1 baseline, and a shuffled-score
control in which the gain collapses to zero.

```
Best teacher configuration:
  P1-G2 minmax^2 (C3) + official facility location
  (nominally higher but statistically indistinguishable: C3 + block8, 79.15)
Accuracy gain:
  +11.92 macro points   (TextVQA +6.80, DocVQA +21.64, OCRBench +7.33)
  95% CI [+8.09, +15.76], 102 improved / 323 tied / 25 worsened of 450
Oracle latency:
  757 ms/image = 216 ms gradient forward + 296 ms gradient backward
                 + 245 ms untouched EADP-256 end-to-end
```

---

## 0. Executive summary

1. **The gradient score translates.** At `T = 256` on the frozen 450,
   `P1-G2 + facility` reaches macro **78.81** against the official EADP facility
   baseline of **66.88** — **+11.92 points**, 95 % CI [+8.09, +15.76].
2. **It is not an artifact of the harness.** Running *official* EADP scoring
   through the exact same runner reproduces the Stage-1 numbers to three
   decimals on all three benchmarks (70.267 / 64.380 / 66.000), Δ = **+0.000**,
   with 450/450 tied instances. The only thing that changes in the gradient arms
   is the importance map.
3. **It is not an artifact of score magnitude distribution.** Rotating the
   gradient map onto the wrong image — identical statistics, identical
   calibration, identical selector — gives macro Δ **+0.97** with a CI spanning
   zero, and actively *hurts* DocVQA (−6.9). The gain is content-specific.
4. **No benchmark regresses**; DocVQA gains the most (+21.6), reaching 86.0 ANLS
   at 256 tokens against 77.7 for the paper's EADP-512 row.
5. **The expensive selector becomes nearly unnecessary.** With the gradient
   score, plain Top-256 reaches +10.49 — within 1.4 macro points of the full
   facility-location arm, where with EADP's own score the same swap costs −19.8
   (Stage-1). The gradient score is already well distributed.
6. **The Stage-1 efficiency fix still applies**: `block8` + gradient is nominally
   the best arm (79.15, +12.27) at 8.8 ms of selection instead of 41.7 ms.
7. **Gains concentrate exactly where the Stage-1 diagnosis said the damage was**
   (brief §9): OCRBench KIE +23.3, DocVQA long answers +41.0, table/list +19.3.
8. **The oracle is not deployable**: 757 ms/image, 3.1× the EADP-256 end-to-end
   it would accelerate and 1.45× the unpruned model. That is the next problem,
   not this one's.

---

## 1. Setup

### 1.1 Frozen set

The exact Stage-1 Part 2 bank: `sample_indices(len(dataset.data), 150, offset=0)`
per benchmark, n = 450 (TextVQA_VAL / DocVQA_VAL / OCRBench). The rebuilt indices
were asserted **bit-identical** to the `idx` field stored in
`diag_selectors_b256.json` before anything was run.

The official `facility @256` baseline is **reused from that file, not
recomputed** — same frozen indices, same generation code, same model
configuration (`eadp_model_name(256, 0.5, 2.0)`, `max_new_tokens=2048`,
`sim_mode=rebound`).

### 1.2 Score

Identical to S2-A P1, no answer and no GT:

```
k        = argmax(logits at the first answer position)     # prompt-only forward
J        = logits[k]
score_i  = sum_d | dJ/dv_i,d * v_i,d |
```

computed once for all 450 instances (~9 min on the A40) and cached to
`s2b_gradient_scores.npz`. Per the S2-A finding the signal is largely
answer-agnostic, so it is described as **gradient sensitivity**, not
answer-conditioned attribution.

### 1.3 Score injection

The precomputed map is injected by overriding `TimedEADPPruner._score`. Nothing
else changes: the similarity kernel, the selector, `_build_pruned_inputs` and the
generation call are the Stage-1 code verbatim (`diag_selectors.generate_prediction`).
`n_kept` is 256 on every arm.

### 1.4 Calibrations

Only the three the brief specifies. Facility location consumes score *magnitude*,
not just ranking, so the calibration is the interface between the gradient map and
the existing objective:

| | definition |
|---|---|
| C1 | `s = g2 / mean(g2)` |
| C2 | `s = (g2 − min) / (max − min)` |
| C3 | `s = minmax(g2)²` |

No larger sweep, no learned calibration. C3 was promoted from the pilot by the
brief's rule (highest pilot mean); C1 was carried to the full set as a
calibration-robustness arm.

---

## 2. Two controls

Both were run because a +12-point macro gain at fixed token budget is not
something to accept on the strength of one experiment.

### 2.1 Harness identity — exact

Running **official** EADP scoring (no override at all) through this same runner,
on the full frozen 450:

| Benchmark | Stage-1 `diag_selectors_b256.json` | this runner, official scoring | Δ |
|---|---|---|---|
| TextVQA_VAL | 70.267 | **70.267** | +0.000 |
| DocVQA_VAL | 64.380 | **64.380** | +0.000 |
| OCRBench | 66.000 | **66.000** | +0.000 |
| **macro** | 66.882 | **66.882** | **+0.000** |

450 tied instances, 0 improved, 0 worsened, bootstrap CI [+0.000, +0.000]. The
runner is provably neutral: the only difference between the baseline and the
gradient arms is the importance map.

### 2.2 Shuffled-score control — the gain collapses

The same C2 calibration, the same facility selector, the same pipeline — but the
gradient map is assigned to a **different** image (indices rotated by 7). Every
statistic of the map is preserved: same distribution, same dynamic range, same
sparsity, same block structure, same selector. Only the correspondence to the
image is destroyed.

Pilot subset (n = 50/dataset):

| arm | TextVQA | DocVQA | OCRBench | macro | Δ vs official | 95 % CI |
|---|---|---|---|---|---|---|
| official EADP facility | 59.400 | 61.891 | 62.000 | 61.097 | — | — |
| C2, correct assignment | 73.600 | 80.978 | 76.000 | **76.859** | **+15.762** | [+9.22, +22.73] |
| C2, **rotated** assignment | 65.200 | **55.014** | 66.000 | 62.071 | **+0.974** | **[−5.87, +7.76]** |

24 improved / 101 tied / 25 worsened. The rotated map is statistically
indistinguishable from EADP's own score, and it makes DocVQA **worse** than the
baseline.

This is the decisive control. It rules out the strongest alternative explanation —
that any map with the gradient score's magnitude distribution would reshape
facility location's behaviour favourably, regardless of what it points at. It
does not: the image-specific content of the score is doing the work.

---

## 3. Stage 1 — pilot (n = 50 per benchmark)

Pilot indices are every 3rd index of the frozen 150, so that the official
facility baseline is already available for exactly those instances and the
comparison is paired and free.

*(This is not the 50 indices Stage-1 profiling used: those have no saved accuracy
baseline, so a paired pilot is impossible on them.)*

| arm | TextVQA | DocVQA | OCRBench | macro | Δ | 95 % CI | win/tie/loss |
|---|---|---|---|---|---|---|---|
| official EADP facility | 59.400 | 61.891 | 62.000 | 61.097 | — | — | — |
| facility \| C1 | 73.600 | 82.077 | 76.000 | 77.226 | **+16.129** | [+9.61, +22.65] | 41/103/6 |
| facility \| C2 | 73.600 | 80.978 | 76.000 | 76.859 | +15.762 | [+9.22, +22.73] | 39/105/6 |
| facility \| C3 | 77.000 | 82.396 | 74.000 | **77.799** | **+16.702** | [+9.82, +23.80] | 42/101/7 |

Note the official baseline on this subset (61.10 macro) is much *harder* than the
full frozen set (66.88) — the every-3rd draw is enriched in difficult instances on
all three benchmarks. Headroom is therefore larger here, and the full-set gain is
correspondingly smaller; this is why the brief's pilot→full protocol matters.

**Pilot diagnostics.** Selection overlap with the official EADP selection is only
**0.25–0.34** of 256 tokens — these are substantially different token sets, not a
perturbation. The gradient score's top-256 is *less* spatially concentrated than
EADP's own map (top block 14.4 % vs 16.3 % of the budget; both far above the
8.7 % a random draw gives), while the resulting facility *selection* is slightly
more concentrated (C3 top block 13.9 % vs official 10.4 %).

### Pilot decision

Best calibration C3 → **Strong GO** (macro gain ≥ +3.0, no benchmark regressing
by more than 1 point) → promoted to the full n = 450 per the brief.

---

## 4. Stage 2 — full frozen n = 450

| arm | TextVQA | DocVQA | OCRBench | macro | Δ macro | 95 % CI | win/tie/loss |
|---|---|---|---|---|---|---|---|
| **official EADP facility** | 70.267 | 64.380 | 66.000 | 66.882 | — | — | — |
| **facility \| C3** | 77.067 | 86.018 | 73.333 | **78.806** | **+11.924** | [+8.09, +15.76] | 102/323/25 |
| facility \| C1 | 75.333 | 86.277 | 75.333 | 78.981 | +12.099 | [+8.45, +15.86] | 101/324/25 |
| topk \| C3 | 76.133 | 84.649 | 71.333 | 77.372 | +10.490 | [+6.64, +14.30] | 97/325/28 |
| block8 \| C3 | 77.000 | 86.443 | 74.000 | **79.148** | **+12.265** | [+8.51, +16.15] | 104/322/24 |

Per-benchmark deltas for the primary arm (`facility | C3`): TextVQA **+6.800**,
DocVQA **+21.638**, OCRBench **+7.333**. No benchmark regresses. Paired bootstrap
is stratified over datasets, 10 000 resamples, macro = mean of the three dataset
accuracies.

For reference, the paper's own EADP-512 row (twice the token budget) reports
TextVQA 75.2, DocVQA 77.7, OCRBench 673/1000. The gradient-scored 256-token
configuration exceeds the 512-token EADP result on both VQA benchmarks.

**Calibration robustness.** All three calibrations are monotone rescalings of the
same ranking. On the pilot the full three-way spread is 0.94 macro points
(C3 77.799, C1 77.226, C2 76.859); on the full set C1 and C3 land **0.18** apart
(78.981 vs 78.806). Once the ranking is good, the exact magnitude reshaping is
not critical — a useful negative result about how much of the effect is
"calibration" (very little) versus "ranking" (essentially all of it). C2 was not
carried to the full set: it was neither best nor worst on the pilot and the
spread is inside the noise.

### 4.1 The selector barely matters any more

| selector (C3) | macro | Δ vs official | selection cost |
|---|---|---|---|
| facility (official greedy) | 78.806 | +11.924 | 41.7 ms |
| block8 (Stage-1 approximation) | 79.148 | +12.265 | 8.8 ms |
| topk (no coverage at all) | 77.372 | +10.490 | 0.73 ms |

With EADP's own score, replacing facility location with plain Top-K cost **−19.8
points** (Stage-1 §5.2). With the gradient score the same substitution costs
**−1.43** points, and dropping to `block8` costs nothing measurable. The
diagnostic question the brief posed — *does the gradient score already provide
sufficiently distributed evidence that expensive coverage selection becomes
unnecessary?* — is answered **essentially yes**. The coverage objective's value
was compensating for a bad importance map.

### 4.2 Error transitions

`correct = hit ≥ 0.5`, the Stage-1 hard-error cut. Primary arm `facility | C3`:

| | fixed (off wrong → arm correct) | broken (off correct → arm wrong) | both correct | both wrong |
|---|---|---|---|---|
| TextVQA_VAL | 16 | 6 | 99 | 29 |
| DocVQA_VAL | 29 | 5 | 103 | 13 |
| OCRBench | 22 | 11 | 88 | 29 |
| **total** | **67** | **22** | **290** | **71** |

Net +45 instances. The gain is not a reshuffle: 67 hard failures are repaired
against 22 newly broken, and the ratio is favourable on every benchmark.

### 4.3 Where the gains land (brief §7 / §9)

Stage-1's error composition said the damage is concentrated on precise reading of
small, dense, repetitive text. The transitions were broken down with the
**Stage-1 bucketing only** (`category`, `question_types`, `[-1,3,8,20,1000]`
answer-length cut); no new annotations were introduced.

**DocVQA by ground-truth answer length** — the gain rises monotonically with
answer length:

| chars | n | fixed | broken | Δ acc | official | gradient |
|---|---|---|---|---|---|---|
| 1-3 | 19 | 2 | 1 | +9.65 | 78.9 | 84.2 |
| 4-8 | 39 | 7 | 2 | +15.79 | 69.2 | 82.1 |
| 9-20 | 69 | 11 | 1 | +21.80 | 76.8 | 91.3 |
| **21+** | **23** | **9** | **1** | **+40.98** | **56.5** | **91.3** |

**DocVQA by question type** — gains are broad, largest on the categories Stage-1
flagged: `table/list` +19.26 (n=37), `layout` +19.40 (n=42), `form` +20.77
(n=19), `free_text` +21.15 (n=18).

**OCRBench by category**:

| category | n | fixed | broken | Δ acc | official | gradient |
|---|---|---|---|---|---|---|
| **Key Information Extraction** | 30 | 9 | 2 | **+23.33** | 60.0 | 83.3 |
| Digit String Recognition | 8 | 2 | 1 | +12.50 | 12.5 | 25.0 |
| Doc-oriented VQA | 30 | 7 | 4 | +10.00 | 50.0 | 60.0 |
| Scene Text-centric VQA | 30 | 2 | 1 | +3.33 | 90.0 | 93.3 |
| Non-Semantic Text Recognition | 7 | 1 | 2 | −14.29 | 71.4 | 57.1 |
| Regular / Artistic / Irregular / Handwriting | 37 | 0 | 0 | ±0.00 | 85.7–100 | 85.7–100 |

**TextVQA by answer length**: +9.68 (1-3, n=31), +7.81 (4-8, n=64), +6.53 (9-20,
n=49), −16.67 (21+, n=6 — one broken, zero fixed, small n).

**Observation, not a mechanism claim.** Five of the brief's six predicted
concentration points hold — OCRBench KIE is the largest OCRBench gain, DocVQA
`table/list` and long answers are large, digit strings and doc-oriented VQA
improve. That is *consistent with* the S2-A hypothesis that the sensitivity map
finds dense information-bearing regions. It is also consistent with several other
stories, and the one counterexample (TextVQA 21+, n=6, −16.7) is small enough to
be noise. **Recorded as an observation only; no mechanism is claimed.**

---

## 5. Timing

Measured per instance on the full run, A40 46 GB, SDPA:

| quantity | ms |
|---|---|
| gradient forward (prompt-only) | **216.1** |
| gradient backward | **296.2** |
| peak memory | 23 085 MB |
| facility selection | 41.7 |
| block8 selection | 8.8 |
| topk selection | 0.7 |

**Oracle total cost** (brief §10) — gradient forward + backward + selection +
pruned prefill + generation:

```
216.1  gradient forward
296.2  gradient backward
244.8  untouched EADP-256 end-to-end (selection + pruned prefill + generation)
------
757.1 ms per image
```

Reference points from Stage-1: EADP-256 end-to-end 244.8 ms, unpruned end-to-end
522.1 ms, EADP pruning overhead 41.96 ms, EADP scoring stages 0.92 ms.

So the oracle is **3.1× the EADP-256 pipeline it would accelerate** and **1.45×
the unpruned model** — it would be slower than not pruning at all. Per the brief
this is recorded and not used for the GO/NO-GO gate, but it must never be
presented as deployable: the accuracy value of the score is established here, and
the cost of approximating it is the entire next problem.

---

## 6. Final gate

Brief §8: promote only if `P1-G2 + facility` at `T = 256` achieves **either**
macro gain ≥ +3.0, **or** macro gain ≥ +2.0 with a paired 95 % CI excluding 0 and
no substantial benchmark regression.

| criterion | requirement | measured | pass |
|---|---|---|---|
| macro gain ≥ +3.0 | +3.0 | **+11.924** | ✓ |
| (alternative) ≥ +2.0 with CI excluding 0 | — | +11.924, CI [+8.09, +15.76] | ✓ |
| no substantial benchmark regression | — | TextVQA +6.80, DocVQA +21.64, OCRBench +7.33 | ✓ |

Both routes pass, with roughly 4× the required margin, on a set where the harness
is provably neutral and a shuffled-score control returns zero.

---

## 7. What this does not establish

* **Nothing about deployment.** The measured configuration is an oracle costing
  757 ms/image. No approximation of the gradient was tested — that was explicitly
  out of scope.
* **Only the frozen 450, only T = 256.** No 128/512 sweep, no other benchmark, no
  full split beyond the frozen 150/benchmark.
* **The frozen set is 150 evenly-spaced instances per benchmark**, not the whole
  split. The official baseline reproduces the Stage-1 measurement exactly, so the
  comparison is valid, but the absolute numbers are subset numbers.
* **The mechanism is unidentified.** §4.3's concentration of gains is consistent
  with the dense-information-region story and with others; the S2-A
  answer-agnosticism result still stands and no text-density annotation was built.
* **The calibration choice is not a contribution.** C1/C2/C3 differ by ≤0.5 macro
  points; none is claimed as novel.
* **`block8` being nominally best (79.15 vs 78.81) is not a finding** — the
  difference is far inside the noise. It only shows the Stage-1 efficiency fix
  remains compatible with the new score.

---

## 8. Reproduce

```
source ~/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
cd /media/disk2/YZX/research/EADP/Qwen_vl
python scripts/discovery/s2b_gradient_scores.py                 # P1-G2 for the frozen 450 (~9 min GPU)
python scripts/discovery/s2b_run.py --pilot --cals C1 C2 C3 --selectors facility --tag s2b_pilot
python scripts/discovery/s2b_run.py --cals official --selectors facility --tag s2b_identity   # control
python scripts/discovery/s2b_run.py --pilot --cals C2 --selectors facility --shuffle-scores 7 --tag s2b_ctrl
python scripts/discovery/s2b_run.py --cals C3 --selectors facility topk block8 --tag s2b_full
python scripts/discovery/s2b_run.py --cals C1 --selectors facility --tag s2b_full
for t in pilot identity ctrl full; do
  python scripts/discovery/s2b_analysis.py --arms s2b_$t.json --tag s2b_$t; done
python scripts/discovery/s2b_transitions.py --arms s2b_full.json --arm "facility|C3" --tag s2b_full
python scripts/discovery/s2b_concentration.py
python scripts/discovery/s2b_consolidate.py
```

Deliverable: `outputs/discovery/s2b_accuracy_translation.json`. Supporting:
`s2b_full.json`, `s2b_pilot.json`, `s2b_identity.json`, `s2b_ctrl.json`,
`s2b_gradient_scores.npz`, `s2b_official_selection.npz`, the four
`*_analysis.json`, `s2b_full_transitions.json`, `s2b_concentration.json`.

---

```
S2-B GO:
gradient sensitivity is a valid high-quality teacher score.
Next problem is approximating it cheaply.
```

```
Best teacher configuration:
  P1-G2 minmax^2 (C3) + official facility location
Accuracy gain:
  +11.92 macro points (TextVQA +6.80, DocVQA +21.64, OCRBench +7.33)
  95% CI [+8.09, +15.76]
Oracle latency:
  757 ms/image (216 ms forward + 296 ms backward + 245 ms untouched EADP-256 e2e)
```
