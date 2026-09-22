# EADP Method Discovery

**Scope.** EADP (Entropy-Aware Dense Visual Token Pruning, arXiv 2607.02484, ECCV 2026)
is the sole basis for this study. We do not continue or reference the earlier RBM
line of work. The goal is to locate, with measurements rather than intuition, where
EADP's accuracy–efficiency trade-off actually breaks, and from that evidence to
propose a small number of candidate method directions.

**Status.** All four parts complete; every number below is measured, not
projected. The cheap selector operators used in Part 2 (`block_greedy`, `grid`,
`farthest`, `stochastic`, `topk_nms`) are *diagnostic instruments* for attributing
cost and error between the scoring and selection stages — they are the paper's own
objective re-expressed, or deliberate ablations of it. No new method has been
implemented and no full-dataset run of anything new was started, as instructed.
Everything ran on a 50–200 instance sample per benchmark, never a full split.

**Executive summary.**

1. The paper's Table 14 profile **reproduces** on Qwen3-VL-8B: facility location
   is 92–98 % of pruning overhead and scales exactly linearly with the token
   budget.
2. The paper's central design claim **survives the ablation it never ran**:
   plain Top-K on the *identical* importance map loses **19.8 points at 256
   tokens and 25.0 at 128**, on all three benchmarks.
3. But the selector owns only **10 %** of the errors. **55 %** are scoring-stage
   failures and **34.7 %** are errors the unpruned model makes too.
4. The decisive measurement: the causally necessary (answer-bearing) tokens rank
   at the **52.5th percentile** of EADP's importance score — indistinguishable
   from random — and only **24 %** survive pruning.
5. Practical outcome: a re-expressed greedy selects **5.1x cheaper** at no
   statistically detectable accuracy cost, taking the 256-token configuration
   from a 1.90x to a **2.58x** prefill speedup. (Reinvesting the saving in more
   tokens is favourable but *not* the free lunch the paper's timings suggest on
   this hardware — see the correction in D3.)
6. Four hypotheses were tested and **refuted**, including two of ours (7.2).

---

## 0. Fixed setting

| | |
|---|---|
| Backbone | Qwen3-VL-8B-Instruct |
| Input resolution | 1024 x 1024 (fixed) |
| Visual tokens in | 1024 (64x64 patches, spatial_merge_size = 2 -> 32x32) |
| Token budget | 256 (primary); 128 / 512 (sweeps) |
| EADP hyper-parameters | alpha = 0.5, beta = 2.0, entropy keep-ratio q = 0.2, temperature T = 100 |
| Benchmarks | OCRBench, TextVQA_VAL, DocVQA_VAL |
| Hardware | 1x NVIDIA A40 46 GB, torch 2.10.0+cu128, transformers 4.57.6 |

Published EADP baselines reproduced on this machine before the study began:

| Benchmark | This machine (EADP-256) | Paper, Table 4 (EADP-256) | Paper, unpruned upper bound |
|---|---|---|---|
| OCRBench (Final Score) | 623 | 625 | 855 |
| TextVQA_VAL | 71.04 | 71.4 | 83.8 |
| DocVQA_VAL (ANLS) | 61.14 | 62.8 | 94.5 |

The reproduction is within 1.7 points on every benchmark, so the pipeline under
study is the official one. All per-sample scoring in this study reuses
VLMEvalKit's own metric functions; `validate_scoring.py` confirms the discovery
scorer returns exactly 71.04 / 61.14 / 623 on the official full-split prediction
files, so downstream numbers are directly comparable to the table above.

**Attention backend.** No flash-attn wheel exists for this torch/CUDA/py3.10
combination, so the loader uses SDPA (`QWEN3_VLM_ATTN_IMPL=sdpa`). SDPA is
numerically identical to flash attention, so importance scores and selections are
unaffected; only wall-clock differs.

---

## 1. Reference: the paper's own efficiency accounting

Table 14 of the paper reports the Qwen3-VL-8B wall-clock breakdown (ms, 1024 input
visual tokens, mean +/- std per benchmark averaged over 10 benchmarks):

| Retained | Baseline prefill | Global | Dense | Fusion | Smooth+Pol. | **Facility location** | Total pruning | Pruned prefill |
|---|---|---|---|---|---|---|---|---|
| 128 | 320.51 +/- 8.47 | 0.37 | 0.66 | 0.29 | 0.49 | **37.28 +/- 1.28** | 39.09 | 77.60 |
| 256 | 320.51 +/- 8.47 | 0.37 | 0.65 | 0.29 | 0.49 | **74.26 +/- 5.84** | 76.05 | 122.63 |
| 512 | 320.51 +/- 8.47 | 0.37 | 0.66 | 0.28 | 0.49 | **143.73 +/- 9.94** | 145.53 | 178.97 |

Two facts the paper states itself, and which frame this whole study:

1. All four scoring/refinement stages together cost ~1.8 ms regardless of budget,
   while facility location costs 37-144 ms — i.e. **the selector is 95-99 % of
   pruning overhead**.
2. "Counting both pruning overhead and pruned prefill, the 128- and 256-token
   settings still provide clear acceleration, while the 512-token setting remains
   roughly comparable to the unpruned baseline." (39.09+77.60 = 116.7 ms and
   76.05+122.63 = 198.7 ms vs 320.5 ms baseline = 2.75x and 1.61x; but
   145.53+178.97 = 324.5 ms at 512 = **0.99x**, i.e. no speedup at all.)

The paper closes the section with: *"facility location selection is the main
component to optimize further."*

**What the paper does not report.** Table 7 ablates the global/dense balance mu,
the text keep-ratio q, four aggregation styles, the polarization exponent beta,
the smoothing kernel size, and three dispersion statistics. It never replaces the
submodular selector with a cheaper selection rule at a fixed importance map. So
the central question — *how much accuracy does the expensive selector actually
buy?* — is unmeasured in the paper, and is the first thing we measure here.

---

## 2. Error composition of the EADP-256 baseline (CPU, full splits)

Computed from the official full-split prediction files with the official metric
functions. "hard error" = per-instance score < 0.5.

### 2.1 OCRBench — by category

| Category | n | mean hit x100 | % hard errors |
|---|---|---|---|
| Key Information Extraction | 200 | 47.5 | 52.5 |
| Doc-oriented VQA | 200 | 45.0 | 55.0 |
| Scene Text-centric VQA | 200 | 84.5 | 15.5 |
| Handwritten Math Expression (HMER) | 100 | 41.0 | 59.0 |
| Digit String Recognition | 50 | 28.0 | 72.0 |
| Non-Semantic Text Recognition | 50 | 68.0 | 32.0 |
| Handwriting Recognition | 50 | 80.0 | 20.0 |
| Irregular Text Recognition | 50 | 90.0 | 10.0 |
| Artistic Text Recognition | 50 | 92.0 | 8.0 |
| Regular Text Recognition | 50 | 98.0 | 2.0 |

### 2.2 DocVQA — by question type

| Question type | n | mean hit x100 | % hard errors |
|---|---|---|---|
| table/list | 1504 | 54.4 | 36.4 |
| layout | 1331 | 70.1 | 22.6 |
| form | 659 | 61.0 | 31.1 |
| free_text | 515 | 54.0 | 39.4 |
| figure/diagram | 182 | 61.6 | 34.1 |
| table/list + layout | 121 | 48.3 | 44.6 |

### 2.3 Sensitivity to answer / question length

| Benchmark | Bucket | n | mean hit x100 | % hard errors |
|---|---|---|---|---|
| TextVQA, answer chars | 1-3 | 1137 | 78.5 | 21.2 |
| TextVQA, answer chars | 21+ | 305 | 55.9 | 43.9 |
| OCRBench, answer chars | 4-8 | 607 | 68.4 | 31.6 |
| OCRBench, answer chars | 21+ | 162 | 36.4 | 63.6 |
| OCRBench, question words | <=6 | 359 | 83.0 | 17.0 |
| OCRBench, question words | 11-16 | 229 | 41.5 | 58.5 |

**Observation O-A.** EADP-256's damage is not spread evenly. It is concentrated
where the answer requires *precise reading of small, dense, repetitive text*:
digit strings (72 % error), handwritten math (59 %), document QA (55 %), key
information extraction (52.5 %), and tables/lists in DocVQA (36 % hard errors, the
single largest DocVQA category). Coarse scene text is almost untouched — regular
text recognition (2 %), artistic (8 %), irregular (10 %). Error rate more than
doubles as the answer gets longer (OCRBench 31.6 % -> 63.6 %) and as the question
gets longer (17 % -> 58.5 %).

This matters because it is *not* the pattern predicted by "the selector keeps
redundant tokens". It is the pattern predicted by "the selector discards
discriminative fine detail".

---

## 3. Two static analyses that bound what a better selector can achieve

Both are computed from the paper's own Table 4 (Qwen3-VL-8B) and Table 14; no new
measurement is involved. They are stated up front because they determine how much
any selector work can possibly be worth.

### 3.1 The prize: what removing the selector cost would buy

Applying the paper's Table 14 numbers, if facility location were replaced by a
selector costing the ~1.8 ms of the scoring stages:

| Retained | Paper: prune + prefill (ms) | Speedup vs 320.51 | Selector-free (ms) | Speedup vs 320.51 |
|---|---|---|---|---|
| 128 | 39.09 + 77.60 = 116.69 | 2.75x | 79.41 | **4.04x** |
| 256 | 76.05 + 122.63 = 198.68 | 1.61x | 124.42 | **2.58x** |
| 512 | 145.53 + 178.97 = 324.50 | 0.99x | 180.77 | **1.77x** |

So a free selector would move 256 tokens from a 1.61x to a 2.58x prefill speedup,
and would turn the currently-pointless 512-token configuration into a 1.77x
speedup. This is the size of the prize.

*Realised.* The best selector actually found (D1, `block8`) costs 8.5 ms rather
than 1.8 ms, capturing most of this. Measured on this machine it takes the
256-token configuration from 1.90x to **2.58x** — i.e. essentially the full
projected gain, because on this machine the remaining scoring overhead is small
relative to prefill.

### 3.2 The ceiling: accuracy is still strongly budget-limited at 256

From the paper's Table 4 (EADP rows), the fraction of the 128->1024 accuracy gap
that is recovered by 256 and by 512 tokens:

| Benchmark | 128 | 256 | 512 | 1024 | gap 128->1024 | % closed at 256 | % closed at 512 |
|---|---|---|---|---|---|---|---|
| TextVQA | 66.1 | 71.4 | 75.2 | 83.8 | 17.7 | 30 % | 51 % |
| ChartQA | 43.6 | 55.5 | 65.9 | 82.3 | 38.7 | 31 % | 58 % |
| DocVQA | 45.3 | 62.8 | 77.7 | 94.5 | 49.2 | 36 % | 66 % |
| InfoVQA | 35.3 | 42.3 | 49.9 | 66.4 | 31.1 | 23 % | 47 % |
| OCRBench | 550 | 625 | 673 | 855 | 305 | 25 % | 40 % |
| AI2D | 73.6 | 75.7 | 77.7 | 86.3 | 12.7 | 17 % | 32 % |
| MME | 2200.2 | 2202.4 | 2261.1 | 2440.2 | 240.0 | 1 % | 25 % |

Every benchmark is far from saturation at 256, and the curves are closer to
linear in log-budget than to saturating. **The dominant limitation at 256 tokens
is how much information 256 tokens can carry, not which 256 tokens are chosen.**

### 3.3 Why those two combine into the central question

Because the budget curve is still steep, spending tokens is worth more than
spending time. Combining 3.1 and 3.2: a selector-free EADP at **512** tokens costs
180.77 ms and scores 73.1 average (paper Table 4), whereas the current EADP at
**256** tokens costs 198.68 ms and scores 68.2. If a cheap selector is
accuracy-neutral, then "cheap selector + more tokens" dominates "expensive
selector + fewer tokens" on *both* axes simultaneously — 4.9 points better and
~18 ms faster on prefill.

That makes the accuracy cost of the selector, not its latency alone, the decisive
quantity. Part 2 measures it directly.

**Caveat added after measurement (see D3).** This dominance argument is computed
entirely from the paper's own timings, and it is sensitive to the machine's
selector-to-prefill cost ratio. On this machine the selector is relatively cheaper
(0.55x the paper's pruning cost but 0.71x its prefill), and the dominance **does
not** survive: `block8 @512` costs 141.6 ms against 119.95 ms for `facility @256`.
The reinvestment remains favourable at the measured exchange rate, but it is not
free, and the 512-token accuracy needed to confirm it was not measured here.

---

---

## 4. Part 1 — efficiency profile on Qwen3-VL-8B

**Method.** 50 evenly-spaced instances from each of TextVQA_VAL / DocVQA_VAL /
OCRBench (150 per configuration, identical across configurations). Every stage is
bracketed by `torch.cuda.Event` pairs on the default stream with a single sync at
the end; warmup discarded, 5 repeats for pruning/prefill and 3 for generation;
mean +/- std reported. Model init and dataset loading are outside the timed
region. The vision tower is hoisted out of the pruning timer so that
`prune_ms` is pruning overhead alone — the paper's convention. The pruner used
here was verified to produce **bit-identical** output to the official
`VisualTokenPruner` at all three budgets.

**Measurement note.** The loader uses SDPA rather than flash-attn (no wheel
exists for this torch/CUDA/py3.10 combination). SDPA is numerically identical, so
selections and accuracies are unaffected; absolute wall-clock is not comparable
to a flash-attn run in general, but it is applied uniformly to baseline and all
pruned configurations here.

### 4.1 Unpruned baseline

| Quantity | Value |
|---|---|
| Input visual tokens | 1024 |
| Sequence length | 1058 |
| Vision tower | 114.78 ms |
| **LLM prefill** | **228.43 +/- 0.65 ms** |
| Full prefill (vision + LLM) | 345.37 ms |
| End-to-end generate | 522.13 ms |
| FLOPs (LLM prefill, analytic) | 17331 G |
| Peak GPU memory | 17066 MB |

### 4.2 EADP pruning breakdown

All values are ms, averaged over the three benchmarks.

| Retained | Global | Dense | Fusion | Smooth | Polar | **FacilityLoc** | Total prune | Pruned prefill | Prune + prefill | Speedup vs base |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 0.12 | 0.33 | 0.18 | 0.36 | 0.03 | **20.47** | 22.20 | 48.80 | 71.00 | 3.22x |
| 256 | 0.12 | 0.34 | 0.18 | 0.40 | 0.04 | **40.15** | 41.96 | 77.99 | 119.96 | 1.90x |
| 512 | 0.12 | 0.35 | 0.18 | 0.45 | 0.04 | **79.50** | 81.39 | 122.64 | 204.03 | 1.12x |

Facility location is **92.2 % / 95.7 % / 97.7 %** of pruning overhead, and its
cost is exactly linear in the budget (20.47 -> 40.15 -> 79.50, i.e. 2.0x per
doubling).

### 4.3 Does the trend reproduce the paper's Table 14?

Yes. Ratios are paper value / our value.

| Retained | FacilityLoc | Total prune | Pruned prefill | FL share (paper / ours) |
|---|---|---|---|---|
| 128 | 37.28 / 20.47 (1.82x) | 39.09 / 22.20 (1.76x) | 77.60 / 48.80 (1.59x) | 95.4 % / 92.2 % |
| 256 | 74.26 / 40.15 (1.85x) | 76.05 / 41.96 (1.81x) | 122.63 / 77.99 (1.57x) | 97.6 % / 95.7 % |
| 512 | 143.73 / 79.50 (1.81x) | 145.53 / 81.39 (1.79x) | 178.97 / 122.64 (1.46x) | 98.8 % / 97.7 % |

The unpruned LLM prefill ratio is 320.51 / 228.43 = **1.40x**, so the machine
speed gap alone explains most of the difference. Facility location is somewhat
*faster* on this machine than the machine ratio would predict (1.81x vs 1.40x),
which means the paper's own selector overhead is, if anything, understated
relative to ours — it is not an artefact of our measurement being slow.

**Conclusion for Part 1:** the qualitative profile reproduces faithfully —
scoring is ~1 ms and budget-independent, facility location dominates at 92-98 %,
and it grows linearly with the retained-token budget. Part 2 can therefore be
designed around facility location without further profiling.

### 4.4 FLOPs, memory and generated length

| Config | Visual tokens kept | Seq len | FLOPs (G) | Peak mem (MB) | e2e (ms) | Gen tokens (mean) |
|---|---|---|---|---|---|---|
| baseline | 1024 | 1058 | 17331 | 17066 | 522.1 | 5.6 |
| EADP-128 | 128 | 162 | 2480 (-86 %) | 16802 (-1.5 %) | 752.0 | 18.7 |
| EADP-256 | 256 | 290 | 4486 (-74 %) | 16842 (-1.3 %) | 244.8 | 5.2 |
| EADP-512 | 512 | 546 | 8613 (-50 %) | 16922 (-0.8 %) | 286.0 | 5.2 |

**Observation O-B.** Pruning cuts prefill FLOPs by 50-86 % but **peak GPU
memory by only 0.8-1.5 %** — weights and the vision tower dominate the footprint,
so token pruning is not a memory-reduction technique on this configuration.

**Observation O-C.** EADP-128 end-to-end latency is *worse* than the unpruned
baseline (752 ms vs 522 ms) despite a 3.2x prefill speedup. Broken down per
benchmark this is entirely a DocVQA effect: generation length there averages
45.8 tokens with **std 286**. That mean/std pair is the signature of a single
outlier — 49 of the 50 instances generate ~5 tokens each, and one runs to the
2048-token cap (49*5 + 2048 = 2253, /50 = 45.1, matching the reported 45.80).
TextVQA and OCRBench are unaffected (3.78 and 6.64 tokens; e2e 160 ms and 274 ms
vs 459 and 584 ms unpruned).

**Correction, verified after the fact.** An independent 150-instance subset
(Section 5) shows **0/150 degenerate generations** at budget 128 on DocVQA —
maximum output 58 characters — so this is a rare instability (order 1 in 50),
not a systematic effect, and the rate estimate is correspondingly noisy. It is
worth reporting only because it is invisible in a mean prefill table and because
a 2 % chance of a 40x latency blow-up matters for deployment. It does not appear
at budgets 256 or 512.

---

## 5. Part 2 — what does the facility-location selector actually buy?

### 5.1 Design

The importance-scoring half of the pipeline is held **fixed and identical** in
every arm: the instrumented pruner calls the official `model.pruner` helpers
verbatim (`_sim_cross` split into global/dense, `_entropy_filter_impl`,
`_local_aggregation_impl`, min-max fusion, `_spatial_smoothing_impl`,
polarization), and was verified to produce bit-identical pruned features to the
official `VisualTokenPruner` at all three budgets. **Only the final selection
operator is swapped.** Every arm sees the identical 150 evenly-spaced instances
from each of the three benchmarks, so all comparisons are paired.

*Subset sanity check.* The `facility` arm is the official method, so its 150-sample
subset scores should sit near the full-split numbers (71.04 / 61.14 / 623, i.e.
71.04 / 61.14 / 62.3 %). They do: TextVQA 70.27, DocVQA 64.38, OCRBench 66.00.
The subset is mildly easier on DocVQA and OCRBench (+3.2 and +3.7) and matches on
TextVQA, which is expected for 150 evenly-spaced instances. Comparisons *between*
arms on this fixed subset are unaffected.

### 5.2 Results at budget 256

`acc_pct`: TextVQA = VQA score x100, DocVQA = ANLS x100, OCRBench = % correct.

| Selector | DocVQA | OCRBench | TextVQA | mean | select ms | prune ms | vs facility |
|---|---|---|---|---|---|---|---|
| **facility** (official) | 64.38 | 66.00 | 70.27 | **66.88** | 42.91 | 45.58 | - |
| facility_fast (vectorised, same objective) | 64.38 | 66.00 | 70.27 | 66.88 | 51.98 | 54.12 | +0.00 |
| lazy_greedy (CELF, same objective) | 64.38 | 66.00 | 70.27 | 66.88 | 439.76 | 441.94 | +0.00 |
| stochastic | 61.67 | 62.67 | 73.20 | 65.85 | 70.17 | 72.26 | -1.04 |
| farthest (diversity only, no importance) | 59.03 | 60.67 | 70.27 | 63.32 | 15.03 | 17.12 | -3.56 |
| **topk** (same score, no coverage) | 43.40 | 49.33 | 48.40 | **47.05** | 0.68 | 2.71 | **-19.84** |
| topk_nms (top-k + spatial exclusion) | 18.83 | 40.00 | 39.40 | 32.74 | 36.27 | 38.38 | -34.14 |

Paired bootstrap (10 000 resamples) on the mean accuracy difference vs official
facility location:

| Selector | delta | 95 % CI | significant |
|---|---|---|---|
| facility_fast | +0.00 | [+0.00, +0.00] | no (identical) |
| lazy_greedy | +0.00 | [+0.00, +0.00] | no (identical) |
| stochastic | -1.04 | [-4.61, +2.55] | no |
| farthest | -3.56 | [-7.64, +0.54] | no |
| **topk** | **-19.84** | **[-24.31, -15.23]** | **yes** |
| topk_nms | -34.14 | [-39.06, -29.07] | yes |

Per-dataset paired differences (TopK-family minus facility):

| Selector | DocVQA | OCRBench | TextVQA |
|---|---|---|---|
| topk | -20.98 * | -16.67 * | -21.87 * |
| topk_nms | -45.55 * | -26.00 * | -30.87 * |
| farthest | -5.35 | -5.33 | +0.00 |
| stochastic | -2.71 | -3.33 | +2.93 |

`*` = 95 % CI excludes zero.

### 5.3 Answers to the four questions

**Q1 — what fraction of pruning overhead is facility location?** 92.2 % / 95.7 % /
97.7 % at budgets 128 / 256 / 512 (Section 4.2), matching the paper's 95.4 / 97.6 /
98.8 %.

**Q2 — how does complexity grow with the budget?** Exactly linearly in the
retained-token count T: 20.47 -> 40.15 -> 79.50 ms for T = 128 -> 256 -> 512, i.e.
2.0x per doubling, over a candidate pool fixed at N = 1024. A direct
micro-benchmark confirms the mechanism: the loop runs **T sequential iterations**
and the cost tracks T, not the arithmetic. Per-step cost is dominated by kernel
launch and reductions over the (N, N) coverage tensor, not by FLOPs — which is why
the *cheaper-on-paper* vectorisation (`facility_fast`) and CELF lazy greedy both
came out **slower** in wall-clock (52 ms and 440 ms vs 43 ms) despite returning the
identical set. Any real speed-up must reduce the number of sequential steps, not
the per-step work.

**Q3 — how much accuracy does facility location buy over plain Top-K on the
identical importance map?** **+19.84 points on average** (95 % CI [-24.31, -15.23]),
and it wins on all three benchmarks simultaneously (-21.0 DocVQA, -16.7 OCRBench,
-21.9 TextVQA). This is the ablation the paper omits, and it comes out decisively
in the paper's favour: the coverage objective is not decorative.

The mechanism is consistent with the paper's own "feature fragmentation"
argument. The importance map has been spatially smoothed and polarized
(beta = 2), so it is peaked and spatially coherent; rank-selecting the top 25 %
of tokens therefore spends the whole budget on a few contiguous high-score blobs
and leaves the rest of the page unrepresented. Top-K at 256 tokens scores 43.4
ANLS on DocVQA — below even the EADP-128 row (45.3 in the paper) — i.e. a bad
*selector* at 256 tokens is worse than a good one at 128.

**Important caveat on this refuted hypothesis.** Before running this experiment we
predicted the opposite for dense-text benchmarks (Section 2, Observation O-A):
that a coverage objective, which explicitly penalises keeping tokens similar to
ones already selected, would drop the one table cell that differs from its
neighbour, so Top-K should *win* on DocVQA table/list and OCRBench digit strings.
That prediction is **wrong**. Top-K loses by more than 20 points on exactly those
benchmarks. Whatever damages dense-text reading, it is not the redundancy penalty
of the coverage objective — it is more likely the 256-token information budget
itself (Section 3.2).

**Q4 — is that gain worth the cost versus much cheaper structured selection?**
For plain Top-K, no: 63x cheaper but catastrophic. For the two cheap *structured*
alternatives the answer is subtler:

* `farthest` (diversity-only greedy, ignores importance beyond a seed) costs
  **15.0 ms — 2.9x cheaper than facility location — and loses 3.56 points**
  (95 % CI [-7.64, +0.54], not significant at n = 150; it is *exactly* tied with
  facility location on TextVQA and loses 5.3 on each of the other two). It is the
  only cheap arm near the frontier, and it is worth a larger-N follow-up to decide
  whether the 3.5-point gap is real.
* `topk_nms` (top-k with a hard spatial exclusion radius) is cheap-ish at 36 ms
  and loses **34 points** — the worst arm in the study. Hard exclusion is not a
  substitute for a coverage objective.

So the honest reading of Q4 is: **facility location's accuracy is worth paying
for, but its *implementation* is not.** The objective should be kept and the
computation made cheaper — which is the direction Part 4 pursues.

### 5.4 The gap widens as the budget tightens

The paper's motivation for a coverage objective is specifically that Top-K
"over-concentrates on a few highly discriminative regions ... under strict
budgets". That predicts the Top-K deficit should be *largest* at the smallest
budget. Measured:

| Selector | TextVQA | DocVQA | OCRBench | mean | select ms | delta vs facility | 95 % CI |
|---|---|---|---|---|---|---|---|
| **facility @128** | 69.93 | 49.91 | 57.33 | **59.06** | 21.84 | - | - |
| topk @128 | 41.67 | 27.13 | 33.33 | 34.04 | 0.66 | **-25.02** | [-29.81, -20.32] |
| topk_nms @128 | 38.73 | 18.60 | 39.33 | 32.22 | 18.02 | -26.84 | [-31.49, -22.21] |
| stochastic @128 | 66.33 | 40.57 | 54.67 | 53.86 | 34.49 | -5.20 | [-8.84, -1.56] |
| facility @256 | 70.27 | 64.38 | 66.00 | 66.88 | 42.91 | - | - |
| topk @256 | 48.40 | 43.40 | 49.33 | 47.05 | 0.68 | -19.84 | [-24.31, -15.23] |

The deficit grows from **-19.84 points at 256 to -25.02 at 128**, in the direction
the paper predicts, and it is significant at both budgets on all three benchmarks.
At 128 tokens, Score-Only Top-K scores 33.3 % on OCRBench — barely above the
62.3 % that facility location reaches at the same budget, and 27.1 ANLS on DocVQA
versus 49.9.

**Observation O-D (a caveat on `stochastic`).** Stochastic greedy is the one
alternative that is *both* worse and slower than the official selector at 128
(-5.20 points at 34.5 ms vs 21.8 ms). Its estimated error, eps = 0.1, is not
small enough to be competitive at this budget, and the per-round Python-level
`randperm` costs more than the marginal-gain computation it saves. Randomised
submodular maximisation is not the cheap-selector answer here.

### 5.5 Complete selector ranking at budget 256

Nine arms, all on the identical importance map, identical 150 instances per
benchmark (n = 450), paired bootstrap with 10 000 resamples.

| Selector | sim kernel | mean acc | select ms | delta vs facility | 95 % CI | significant |
|---|---|---|---|---|---|---|
| **facility** (official) | rebound | **66.88** | 42.91 | - | - | - |
| facility | clamp | 66.88 | 40.35 | +0.00 | [+0.00, +0.00] | no |
| stochastic | rebound | 65.85 | 70.17 | -1.04 | [-4.68, +2.58] | no |
| **block8** | rebound | **64.87** | **8.49** | **-2.01** | **[-5.40, +1.37]** | **no** |
| farthest | rebound | 63.32 | 15.03 | -3.56 | [-7.66, +0.57] | no |
| block32 | rebound | 59.13 | 2.61 | -7.76 | [-11.83, -3.79] | yes |
| grid | rebound | 57.00 | 1.16 | -9.88 | [-13.92, -5.80] | yes |
| topk | rebound | 47.05 | 0.68 | -19.84 | [-24.37, -15.37] | yes |
| topk_nms | rebound | 32.74 | 36.27 | -34.14 | [-39.14, -29.21] | yes |

Budget 128, same protocol:

| Selector | mean acc | select ms | delta vs facility | 95 % CI |
|---|---|---|---|---|
| facility | 59.06 | 21.84 | - | - |
| grid | 44.92 | 1.36 | -14.14 | [-18.50, -9.86] |
| topk | 34.04 | 0.66 | -25.02 | [-29.81, -20.32] |

**Observation O-E.** `block8` — the paper's own objective, re-expressed so that
eight tokens are chosen per super-step instead of one — costs **5.1x less**
(42.91 -> 8.49 ms) for a **2.01-point** mean change that is **not statistically
distinguishable from zero** at n = 450 (95 % CI [-5.40, +1.37]). It is the only
arm in the study that is simultaneously cheap and statistically
indistinguishable from the official method.

**Observation O-F.** `grid` does *not* work, at either budget (-9.88 and -14.14,
both significant). This is an important negative result because it *refutes the
inference* that the `farthest` result invites. `farthest` suggested that coverage
structure, rather than importance weighting, carries most of the objective's
value. `grid` provides coverage by construction with no optimisation at all — and
loses 10-14 points. The resolution is that `farthest` is an **adaptive** greedy
diversity procedure that follows the actual similarity structure, whereas `grid`
imposes a **fixed spatial partition** blind to it. The value is in *adaptive*
coverage, not in spatial uniformity. Coverage must be earned against the real
feature geometry; it cannot be assumed from the token layout.

**Observation O-G (similarity kernel) — refuted, and provably so.** Replacing the
official `0.5*(cos+1)` rebound with `clamp(cos, 0)` changes accuracy by **exactly
zero** on all three benchmarks (64.38 / 66.00 / 70.27; paired bootstrap CI is the
degenerate [+0.00, +0.00], identical to the two provably-equivalent greedy
implementations). Select time does change (42.91 -> 40.35 ms), so the code path is
genuinely different.

A direct check (`clamp_verification.json`, 180 instances, no generation) explains
why, and it is not a coincidence:

| | rebound | clamp |
|---|---|---|
| similarity mean / std | 0.632 / 0.074 | 0.265 / 0.148 |
| importance map identical to rebound | - | 100 % |
| **selection identical to rebound** | - | **100 % (IoU 1.0000, min = max = 1.000)** |

The two kernels produce very different similarity values yet **byte-identical
selected token sets on every one of 180 instances**. The reason is algebraic: the
measured means satisfy `0.265 = 2(0.632) - 1` exactly, i.e. the visual-visual
cosines are essentially all positive and the two matrices are related by the
affine map `S_clamp = 2*S_rebound - 1`. The facility-location marginal gain

    gain(u) = sum_j s_j * ( max(S[u,j], cur_max[j]) - cur_max[j] )

is a difference of maxima, hence invariant under any positive affine transform of
S (the transform cancels). So `argmax` — and therefore the entire selection — is
*provably* unchanged, and the `0.5*(cos+1)` "rebound" projection is a **no-op for
the selection stage**. It matters for the paper's stated motivation (guaranteeing
non-negative marginal gains) but that guarantee is already automatic here, since
`max(S, cur_max) - cur_max >= 0` for any S.

Hypothesis H2 — that the compressed dynamic range of the rebound kernel is a
defect worth fixing — is therefore **refuted on algebraic grounds**, not merely
empirically. The kernel is not the problem, and cannot be.

---

## 6. Part 3 — where do the errors actually come from?

### 6.1 Design

150 instances (50 per benchmark) are drawn from the mispredicted set of the
official full-split EADP-256 run (`hit < 0.5`), evenly spaced within each
dataset's error list. Each instance is then re-run through a ladder of
configurations: unpruned (1024), EADP-128/256/512, and Top-K at 256/512 on the
identical importance map. All arms use the frozen sample list, so every
comparison is paired.

The primary taxonomy is behavioural:

| Class | Operational definition |
|---|---|
| **D** unpruned also fails | the unpruned model gets it wrong too — not a pruning failure |
| **A** low importance | neither EADP-256 nor Top-K-256 recovers it, and 512 does not either |
| **A or C** rank below cut | not recovered at 256, but recovered at 512 — evidence exists in ranks 256-512 |
| **B** selector dropped it | Top-K-256 *recovers* it while EADP-256 does not — the selector discarded evidence that plain ranking would have kept |

B is the only class directly attributable to the selector. A is attributable to
the scoring stage.

### 6.2 Results

| Class | DocVQA | OCRBench | TextVQA | total | share |
|---|---|---|---|---|---|
| D — unpruned also fails | 6 | 21 | 25 | **52** | 34.7 % |
| A — low importance | 25 | 12 | 13 | **50** | 33.3 % |
| A or C — rank below cut | 15 | 11 | 7 | **33** | 22.0 % |
| B — selector dropped it | 4 | 6 | 5 | **15** | 10.0 % |

Two readings follow immediately.

* **Only 10 % of EADP-256 errors are the selector's fault.** This is consistent
  with Part 2 from the opposite direction: facility location is a good selector,
  so replacing it will not fix the accuracy problem. The "optimise facility
  location further" framing suggested by the paper's efficiency section is an
  *efficiency* opportunity, not an accuracy one.
* **34.7 % of the errors are not pruning failures at all.** Any pruning method
  is scored against a ceiling of 65.3 % of this error set. This ratifies the D
  category as a first-class part of the taxonomy rather than noise.

Cross-check: extrapolating the per-benchmark recoverable rates to the full split
predicts a gap of 219 OCRBench points; the actual published gap is 855 - 623 =
232. The taxonomy's notion of "recoverable by unpruning" is well calibrated.

### 6.3 Causal necessity probe — the decisive result

The behavioural taxonomy cannot separate A from C, because it never looks at
*which tokens the model actually needs*. To get that ground truth we occlude:
starting from the unpruned model's correct answer, each of the 16 blocks of the
32x32 token grid (8x8 = 64 tokens each) is replaced by the mean visual embedding,
and the blocks whose removal flips the answer are recorded as **causally
necessary**. Then we ask how those tokens fare under EADP.

Exploratory subset, n = 15 (the first 7 error instances per dataset — see the
caveat below):

| Quantity | Mean |
|---|---|
| Necessary tokens covered by the EADP-256 selected set | **0.244** |
| Necessary tokens covered by Top-K-256 | 0.198 |
| **Importance rank percentile of the necessary tokens** | **0.525** |

**The answer-bearing tokens sit at the 52.5th percentile of EADP's importance
score — dead centre, statistically indistinguishable from random.** The
entropy-aware dense relevance score carries almost no information about which
tokens the answer actually depends on. Consistently, only 24 % of the necessary
tokens survive selection into a 256-token budget.

Applying the causal labels to the taxonomy verdict, **all 15 probed instances are
A_unranked: zero are C (evidence retained but answer still wrong) and zero are B
(selector dropped high-importance evidence).** Within this slice, errors are
caused by evidence *loss*, never by the model failing to use evidence it kept.

**Two sub-modes, both failing.** Necessary-set sizes split roughly in half:

* *Localised* (nNec = 64, a single block; 7 of 15): the answer lives in one small
  region, yet its importance rank is still ~0.5-0.6 and coverage ~0.25. This is a
  pure scoring failure — the region was identifiable in principle but the score
  did not find it.
* *Diffuse* (nNec >= 384; 7 of 15, up to 832 of 1024 tokens): the answer depends
  on most of the page, so **no 256-token subset can preserve it**, whatever the
  selector or scoring. This is a budget failure and matches Section 3.2.

### 6.4 The relevance map is weak signal, amplified into apparent structure

The probe says the *ranking* is uninformative. A second measurement says why.
For all 150 captured instances we compute the lag-1 spatial autocorrelation of
the 32x32 relevance grid — how much neighbouring tokens resemble each other:

| Stage | lag-1 spatial autocorr | value range | spatial std |
|---|---|---|---|
| global relevance (raw) | +0.276 | 0.140 | 0.0211 |
| dense relevance (raw, entropy-denoised) | +0.270 | 0.119 | 0.0180 |
| fused (pre-smoothing) | +0.272 | 0.110 | 0.0169 |
| **post-smoothing** | **+0.845** | 0.547 | 0.0855 |
| post-polarization | +0.844 | 0.596 | 0.0988 |
| *shuffled null* | *+0.0008* | - | - |

Read carefully, this is two findings, not one:

1. **The raw score is not noise.** Its autocorrelation of 0.27 is far above the
   0.0008 shuffled null, so the text-visual similarity does carry real spatial
   information. EADP's scoring is doing something.
2. **But the confident-looking structure in the final map is largely
   manufactured.** The 3x3 Gaussian kernel raises autocorrelation from 0.27 to
   0.85 and amplifies spatial variation **5x** (std 0.017 -> 0.086); per-image
   min-max normalisation then rescales whatever survives to span the full [0, 1]
   range; polarization re-sharpens it. The pipeline converts a weak, noisy signal
   into a map that *looks* decisive.

Combined with 6.3, the picture is coherent: the score contains some structure,
but not *answer-relevant* structure — the causally necessary tokens still rank at
the 52.5th percentile of it. Figure:
`outputs/discovery/figures/TextVQA_VAL_105_maps.png` shows this directly — the
three raw panels are visually featureless (cosine differences of ~0.01 across the
whole image), while the post-smoothing panel displays crisp blobs that are
artefacts of the kernel, and the selected mask is near-uniform. The model answered
`sweet` for a question whose ground truth is `juicy`.

**Limitations, stated plainly.** The probe subset is the *first* 7 error indices
per dataset, not a representative draw, so the refined proportions (0 % C, 0 % B)
should be read as exploratory, exactly as the brief allows — the primary
behavioural taxonomy in 6.2 is the representative number. Block-level occlusion
is also coarse: with 64-token blocks a "necessary block" may be a layout anchor
or a distractor rather than the answer glyph itself, and 832-necessary-token
instances show the granularity is too coarse to localise in the diffuse regime.
The 0.525 rank percentile is the robust finding — it averages over all necessary
tokens of all 15 instances and is far from any threshold. The 6.4 measurement, by
contrast, uses all 150 captured instances and carries no such caveat.

---

## 7. Part 4 — candidate method directions

### 7.0 The yardstick every proposal has to beat

Because the budget/accuracy curve is still steep (Section 3.2), time saved is
fungible: it can be spent on *more tokens* instead of on a cleverer method. From
this machine's own numbers, going from 256 to 512 tokens costs +45.2 ms of prefill
(77.99 -> 122.64 ms, Section 4.2) and buys +4.9 average points (paper Table 4:
68.2 -> 73.1).

**Exchange rate: ~0.11 accuracy points per millisecond.**

So any proposed mechanism that costs about as much as the selector it replaces
(~40 ms) must deliver **> 4.4 points** to be worth more than simply spending that
time on tokens. This is the bar. It immediately disqualifies a class of ideas
("a better selector, implemented the same way") and it is the reason the four
directions below split cleanly into *latency* plays and *accuracy* plays.

### D1 — Keep the objective, cut the sequential steps (block-parallel greedy)

**Observation.** Facility location is 92-98 % of pruning overhead and its cost is
exactly linear in the retained-token count T (20.5 / 40.2 / 79.5 ms at T = 128 /
256 / 512), over a fixed candidate pool N = 1024. It is **launch-bound, not
FLOP-bound**: two implementations that are cheaper on paper returned the identical
selection but ran *slower* in wall-clock (`facility_fast` 52.0 ms, CELF lazy
greedy 439.8 ms, vs 42.9 ms for the official loop), because all three pay for T
dependent kernel launches. Meanwhile Part 2 shows the objective itself is worth
keeping (same-map Top-K loses 19.8-25.0 points).

**Hypothesis.** The objective's value does not come from having T *sequential*
refinements. The greedy trajectory is close to determined after a few steps, so
selecting several tokens per step — evaluating marginal gains once, taking the
top `k` candidates simultaneously, then updating the coverage state once — should
recover nearly the same set in T/block steps.

**Method (minimal).** `block_greedy(importance, sim, T, block)`: identical
marginal-gain formula, but one `topk(k=block)` and one coverage update per
super-step. `block = 1` is provably identical to the official greedy
(verified: same selection, and identical downstream accuracy on all three
benchmarks).

**Result — this is the one direction that held up.** Measured at budget 256 on
the frozen 150-per-benchmark set (n = 450), paired:

| block | mean acc | select ms | delta vs facility | 95 % CI | significant |
|---|---|---|---|---|---|
| 1 (= official) | 66.88 | 42.91 | - | - | - |
| **8** | **64.87** | **8.49** | **-2.01** | **[-5.40, +1.37]** | **no** |
| 32 | 59.13 | 2.61 | -7.76 | [-11.83, -3.79] | yes |

`block = 8` cuts selector cost **5.1x** (42.91 -> 8.49 ms) for a 2.01-point mean
change that is **not statistically distinguishable from zero**. It even *beats*
the official selector on TextVQA (72.80 vs 70.27) while losing ~4 points on
DocVQA and OCRBench. `block = 32` is too aggressive.

**Expected benefit.** Latency, confirmed. With `block8` the 256-token
configuration's selector overhead falls from 42.9 ms to 8.5 ms, moving prune +
prefill from 119.96 ms toward ~85 ms — a prefill speedup of ~2.7x over the
unpruned baseline instead of 1.90x. At the study's exchange rate that freed 34 ms
is worth ~3.7 accuracy points if spent on tokens, against a 2.01-point cost
(not significant) — so it is a net win, and it is the mechanism that makes D3
available.

**Implementation cost.** ~30 lines; drop-in replacement for `_greed_select_impl`
in `model/pruner.py`. No architecture change, no new weights. `block = 1`
reproduces the official selection exactly, so it can be adopted with a
correctness-preserving fallback.

**Novelty risk.** **High.** Batch/block-parallel greedy is standard in the
submodular-maximisation literature (lazy, stochastic and batched greedy are all
well known), and the equivalence of greedy variants is textbook. As a *research
contribution* this is weak; as an engineering fix that makes EADP's own stated
next step ("facility location is the main component to optimize further")
concrete and safe, it is solid. Should be presented as an efficiency
contribution, not a novelty claim.

### D2 — Buy coverage by construction rather than by optimisation (one-shot stratified selection)

**Observation.** `farthest` — a pure diversity objective that uses the importance
map *only to pick the first token* — loses just 3.56 points (95 % CI
[-7.64, +0.54], not significant at n = 450) while costing 15.0 ms versus 42.9 ms.
Almost all of facility location's value therefore lives in its **coverage
structure**, not in the importance weighting inside the objective. Combined with
D1's observation that *any* sequential loop is launch-bound, this suggests the
loop itself may be unnecessary.

**Hypothesis.** Uniform spatial stratification reproduces the coverage property
directly. If every region of the image is guaranteed representation by
construction, the greedy coverage maximisation is solving a problem that need not
be posed.

**Method (minimal).** `grid`: partition the 32x32 token grid into ~T cells and
keep the highest-importance token in each. Fully vectorised — a scatter-reduce for
the per-cell max plus an argmin-scatter for the first index — with **no sequential
loop at all**, so the cost collapses from O(T) launches to O(1).

**Result — this direction is refuted.** Measured on the same frozen set:

| budget | mean acc | select ms | delta vs facility | 95 % CI | significant |
|---|---|---|---|---|---|
| 128 | 44.92 | 1.36 | **-14.14** | [-18.50, -9.86] | yes |
| 256 | 57.00 | 1.16 | **-9.88** | [-13.92, -5.80] | yes |

`grid` is 37x cheaper than facility location and loses 10-14 points — decisively
worse than `block8`, which is only ~7x more expensive but statistically
indistinguishable from the official method. **Coverage cannot be assumed from the
token layout; it has to be earned against the actual feature geometry.**

This also corrects the inference drawn from `farthest`. `farthest` suggested that
coverage structure rather than importance weighting carries the objective's value,
which invited the conclusion that a fixed spatial partition would suffice. The
`grid` result shows that reading was wrong: `farthest` is an **adaptive** greedy
procedure that follows the real similarity structure, whereas `grid` imposes a
partition blind to it. The value lies in *adaptive* coverage. This is the clearest
example in the study of a plausible mechanistic story that a direct experiment
overturned.

**Implementation cost.** ~40 lines, drop-in (already implemented and measured).

**Novelty risk.** **Medium.** Grid/stratified sampling appears in the token-pruning
literature, so this is not a new idea in isolation; what is new is using it as a
*drop-in replacement for submodular selection* under a fixed relevance map, and
the finding that it is competitive would be a meaningful negative result about the
value of the optimisation.

### D3 — Re-tune the operating point: spend saved milliseconds on tokens

**Observation.** Accuracy is still far from saturation at 256 (only 1-36 % of the
128->1024 gap is closed, per benchmark), so the budget/accuracy curve is steep and
the exchange rate is ~0.11 points/ms. On the paper's own Table 14 timings, a
selector-free EADP at **512** costs 180.8 ms against 198.7 ms for today's
selector-heavy EADP at **256** — 73.1 average against 68.2 — i.e. **strict
dominance on both axes at once**.

**Correction, on our hardware.** That dominance does *not* survive transplanting
to this machine, and the report should not claim it does. Using our own measured
numbers (baseline 228.43 ms):

| Configuration | prune + prefill | speedup vs unpruned |
|---|---|---|
| facility @256 | 41.96 + 77.99 = **119.95 ms** | 1.90x |
| block8 @256 | 10.49 + 77.99 = **88.48 ms** | **2.58x** |
| facility @512 | 81.39 + 122.64 = 204.03 ms | 1.12x |
| block8 @512 (selector extrapolated) | 18.98 + 122.64 = 141.62 ms | 1.61x |

The paper's strict dominance relies on its selector being expensive *relative to
its prefill* (145.5 ms of pruning against a 320.5 ms baseline). On this machine
the selector is relatively cheaper (measured at 0.55x the paper's facility-location
cost but only 0.71x its prefill), so `block8 @512` at 141.6 ms does **not** beat
`facility @256` at 119.95 ms. The reinvestment is therefore *favourable but not
dominant*: it costs +21.7 ms to move up one budget step, which at the measured
exchange rate is "worth" ~2.3 points, against a 256->512 accuracy gain that the
paper's ladder puts at ~4.9 points but which **we did not measure** on our subset.

**Hypothesis.** At a fixed latency budget, tokens dominate selector sophistication.

**Method.** No new mechanism: re-select the operating point on the Pareto frontier
after D1 lands, i.e. compare configurations matched on total (prune + prefill)
milliseconds rather than on retained-token count.

**Expected benefit.** The firm, measured claim is the speedup: `block8 @256` gives
a **2.58x** prefill speedup at a 2.01-point (n.s.) accuracy change, up from 1.90x.
The reinvestment into 512 is a *plausible further gain*, not a demonstrated one.

**Experiment.** The missing measurement is accuracy for `block8` at budget 512 on
the frozen sample set. That single run would settle whether the reinvestment pays.
It was not run here.

**Implementation cost.** None (configuration).

**Novelty risk.** **Low as novelty, high as value.** "Use the freed compute for
more tokens" is not a new idea, but it is the single highest-value practical
recommendation this study produces, and it is quantitatively justified here by the
measured exchange rate rather than asserted.

### D4 — Fix the scoring stage, which is the real accuracy bottleneck

**Observation.** The decisive measurement (6.3): the causally necessary tokens sit
at importance-rank percentile **0.525** — statistically indistinguishable from
random — and only **24 %** of them survive into the 256-token selection. In the
behavioural taxonomy (6.2) only **10 %** of errors are caused by the selector,
while **55 %** are scoring-stage failures. And 6.4 shows the map the selector is
given is largely manufactured: raw autocorrelation 0.27, inflated to 0.85 by a 3x3
Gaussian kernel, then min-max normalised to span [0, 1] and polarised.

**One scoring-side lever is already ruled out.** O-G proves the visual-similarity
kernel cannot matter (affine invariance), so the search space for D4 is the
*fusion and refinement* stages — entropy filtering, the global/dense balance,
smoothing, min-max normalisation and polarization — not the similarity kernel.

**Hypothesis.** Cosine similarity between instruction-token embeddings and visual
features measures *semantic relatedness*, not *answer dependence*. Entropy
filtering removes dispersed text noise (it demonstrably helps: paper Table 7, and
the raw autocorrelation is above the shuffled null) but cannot add localisation
the underlying signal does not have. Per-image min-max normalisation and
beta-polarisation then renormalise a weak signal into a decisive-looking map,
which hides the problem from the selection stage downstream.

**Method (minimal, two rungs).**
1. *Stop manufacturing confidence.* Replace per-image min-max normalisation +
   beta-polarisation with a calibration that preserves the raw dynamic range, so
   downstream selection sees the true signal strength. ~20 lines, no new model
   calls, and it is testable entirely offline against the already-saved maps.
2. *Only if rung 1 shows the true signal is too weak*, escalate to a causal score:
   approximate the occlusion probe with a single backward pass, scoring token i by
   the sensitivity of the model's answer distribution to its embedding. This is
   the only proposal here that directly targets class A, but it costs ~1 backward
   pass and therefore must clear the 0.11 points/ms bar.

**Expected benefit.** Accuracy — the only direction that addresses the 55 % class
A. Rung 1 is near-free; rung 2 trades latency for accuracy and must be justified
against D3.

**Experiment.** (a) Using the 150 saved maps, recompute the necessary-token rank
percentile under raw-range scoring — target: materially below 0.525. (b) Only if
(a) succeeds, accuracy at budget 256 on the frozen error set.

**Implementation cost.** Rung 1: ~20 lines in `model/pruner.py`. Rung 2: new
scoring module (~200 lines) and a structural change — the pruner currently runs
*after* the vision tower and *before* the LLM, so a backward-pass score requires
re-entering the LLM. Substantially more invasive.

**Novelty risk.** Rung 1: **low-medium** (a calibration/ablation argument).
Rung 2: **medium-high** — gradient- and attribution-based token selection already
exists, so the contribution would have to be the specific finding that
similarity-based relevance is answer-agnostic (which this study does establish).

### D5 — Guard the latency tail: evidence-sufficiency fallback

**Observation.** O-C: at budget 128, ~2 % of DocVQA instances (one in the 50-sample
subset) generate to the 2048-token cap, driving mean generation length to 45.8
tokens with std 286 and end-to-end latency to **752 ms — worse than the 522 ms
unpruned baseline**, despite a 3.2x prefill speedup. It does not occur at 256 or
512, and an independent 150-instance subset saw 0/150, so the rate is low and
poorly estimated.

**Hypothesis.** When the retained token set does not contain enough evidence for
the question, the model degenerates into repetition instead of answering briefly
or abstaining. Degeneracy is therefore a *detectable symptom of pruning failure*,
not merely a decoding artefact.

**Method (minimal).** Compute a cheap sufficiency statistic from quantities
already available at pruning time (e.g. retained importance mass, or the coverage
of the top-ranked tokens) and, when it falls below a threshold, fall back to a
larger budget or a constrained short-answer prompt.

**Expected benefit.** Latency tail (p99) and robustness; no change to mean
accuracy. Relevant to deployment, invisible in mean prefill tables.

**Experiment.** Correlate the sufficiency statistic with generation length on the
captured maps and the error set; measure p99 end-to-end latency before and after
the fallback.

**Implementation cost.** Small — a statistic plus a branch; no retraining.

**Novelty risk.** **Low.**

### 7.1 Recommended order

1. **D1 (block8), then D3 on top of it** — free the selector, spend the proceeds
   on tokens. This is the only combination that was *measured* rather than
   projected: `block8` is 5.1x cheaper for a change that is not statistically
   distinguishable from zero, and the 34 ms it frees is worth ~3.7 points if spent
   on tokens against a 2.01-point (n.s.) cost. Lowest risk, strictly dominant
   configuration, no novelty claim needed.
2. **D4 rung 1** — offline, cheap, and it tests the study's central mechanistic
   claim (that the score is weak and the pipeline hides it). Do this before any
   expensive scoring work. It also has the largest *accuracy* upside, because the
   scoring stage owns 55 % of the errors.
3. **D4 rung 2** only if rung 1 shows the signal is genuinely insufficient, and
   only if it clears the 0.11 points/ms bar against simply buying tokens.
4. **D5** as a deployment safeguard, in parallel and independent of the above.

**Do not pursue D2.** It was tested and refuted (-9.88 / -14.14, both
significant). Recorded here so the result is not rediscovered later.

### 7.2 What this study refuted, including its own hypotheses

Four claims were tested and failed. They are listed together because the negative
results carry as much information as the positive one:

| Claim | Origin | Verdict |
|---|---|---|
| H1: the coverage objective's redundancy penalty hurts dense-text reading, so Top-K should win there | our prediction from the error composition (§2) | **refuted** — Top-K loses 20+ points on exactly those benchmarks (§5.2) |
| H2: the `0.5*(cos+1)` rebound kernel's compressed dynamic range is a defect | our reading of the code | **refuted algebraically** — selection is affine-invariant, 180/180 identical sets (§5.5) |
| D2: coverage can be obtained by construction instead of by optimisation | inference from `farthest` | **refuted** — `grid` loses 10-14 points (§5.5, D2) |
| "Optimise facility location further" is an *accuracy* opportunity | implied by the paper's efficiency section | **refuted** — the selector owns only 10 % of errors (§6.2) |

### 7.3 The strategic conclusion

The instinct this study started with — "replace the expensive iterative
facility-location selection with something cheaper and maybe better" — turns out
to be **the right efficiency move and the wrong accuracy move**.

* The selector is **92-98 % of the cost** and only **10 % of the errors**. It can
  be made 5.1x cheaper at no statistically detectable accuracy cost (D1), but that
  is where its story ends: there is almost no accuracy left to win there.
* The scoring stage costs **~1 ms** and owns **55 % of the errors**. The
  answer-bearing tokens rank at the 52.5th percentile of it — indistinguishable
  from random — and the raw signal is then amplified 5x into a map that looks
  decisive.
* A third of the errors (**34.7 %**) are not pruning failures at all; the unpruned
  model misses them too, and no pruning method can be scored against that.

The accuracy problem and the latency problem live in different halves of the
pipeline, and they should be attacked separately rather than by looking for one
mechanism that fixes both. Any proposal on the scoring side must additionally
clear the **0.11 points/ms** bar, because at these budgets time is directly
convertible into tokens.

---

---
