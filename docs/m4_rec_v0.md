# M4-v0 — Residual Evidence Compression (REC)

**Stage:** M4, first round of a **new method line**. The MissGuard line is closed
(M3-v2 ruled out the model class, not just the student) and this document does
not reopen it. **Status:** pilot, held-out 150, **no training of any kind** — the
method has no parameters to train.

**The question this document answers, and the only one.**

> If we stop trying to guess *which* dropped token is the critical miss, and
> instead compress the evidence the base selector rejected into a few residual
> capsules under the same 256-token budget, does accuracy recover?

MissGuard asked a membership question — *name the token the teacher would have
kept* — and four independent stages (S2-C2, S3-B, M3-v0, M3-v2) closed it. REC
changes the question rather than the architecture. It never names a token. It
keeps B2's anchors, evicts `r` of them to pay for `r` capsules, and pools
*everything else* — the 768 tokens B2 dropped **plus** the `r` it just evicted —
into those capsules, weighted by how little the retained set explains each one.

**The answer, in one line.** **REFUTED.** At r=16, the pre-registered primary
arm scores **59.38 macro against B2's 59.88** (−0.47, paired CI [−6.2, +5.3],
McNemar p = 1.000, 12 instances fixed against 12 broken). Not one arm in the
grid — merge or control — has a CI that excludes zero, and the residual
weighting is not distinguishable from plain mean (−0.20, 3 fixed / 3 broken) or
from weights taken from *another image* (−0.55). What the round does establish
is mechanistic and negative: **the capsules' content is not what moves the
score.** A control whose capsules pool only the *retained* tokens — carrying none
of the rejected evidence REC exists to carry — scores the same (59.58 vs 59.38),
and the capsule's contribution over the bare eviction has no consistent sign
across `r` (−2.00 / +1.53 / −0.31).

---

## 1. What changed, and what did not

**Unchanged, deliberately.** The base selector is B2 verbatim (EADP importance +
`block8` coverage greedy at T=256), run inside the same `m2_gdep.GDEPEngine` in
`prellm` mode, on the same frozen held-out 150, with the same greedy decode and
the same scorer as B0/B1/B2. `REC0` — the r=0 identity arm — reproduces the
stored B2 record **150/150 on per-instance hits and 150/150 on prediction
strings** (gate 1 below), which is what makes every other row in this document
comparable to a number the project already trusted.

**Unchanged, and this is the binding one.** REC is pre-LLM and forward-only. It
reads the vision tower's own post-merger output, the EADP score the base
selector already computed, and the selector's own similarity matrix. There is no
gradient, no decoder layer, no backward pass, no teacher at inference, no
generation feedback, and — for the first time in this project's method line —
**no learned parameters at all**. The whole method is a weighted mean.

**Changed.** What happens to the tokens B2 rejected. Under MissGuard they were
gone unless a student could name the right one; under REC they are pooled.

## 2. The method, exactly

```
S0 = B2(vis)                        base selection,             |S0| = 256
E  = evict(S0, r)                   maxred, fixed               |E|  = r
A  = S0 \ E                                                     |A|  = 256 - r
D  = all \ A                        the 768 B2 dropped + the r evicted
C  = { c_1 .. c_r }                 one capsule per spatial cell, pooling D
S_REC = A ∪ C                                                   |S_REC| = 256
```

**2.1 The residual weight.** For every token the retained set does not already
explain:

```
u_i = 1 - max_{a in A} cos(v_i, v_a)
```

A dropped token a retained anchor already covers has `u ~ 0` and contributes
little to its capsule; a token no anchor resembles has `u` large and dominates
it. Measured over the held-out 150, `u` on dropped tokens has p10 0.115 /
p50 0.214 / p90 0.327 — a live quantity with real spread, not a constant
(`m4_offline.json`).

**2.2 The capsules.** The 32×32 grid is cut into `r` fixed spatial cells — the
most-square factorisation of `r` (r=8 → 2×4, r=16 → 4×4, r=32 → 4×8). Within
cell `g`:

```
w_i = softmax(u_i / tau)      over the dropped tokens of that cell
c_g = sum_i w_i * v_i         a convex combination of vision features
```

`c_g` is a **convex combination of post-merger vision features** — it stays in
the space the decoder was trained to read. The residual decides only *how much*
a token contributes, never *what* is added: no `v_i - v_anchor` vector, and no
other residual quantity, is ever fed forward. That constraint is why the arm was
expected to be harmless when it was not helpful; §5.5 measures that it was not.

**2.3 What the capsule is not.** Not a learned compressor, not a cluster
centroid, not a re-ranking, not a new token identity. There is no `nn.Module`,
no fit, no checkpoint, nothing to seed. The method is ~40 lines of tensor
arithmetic and a pure function of tensors the incumbent's own pruner already
produces.

**2.4 Deliberately not here** (brief §12): gradient-teacher distillation,
MissGuard-v3, query auditors, L4 pruning, adaptive budget, dynamic STOP,
learnable compressors, large clustering searches, B2 re-optimisation,
hyper-parameter search. The eviction rule, the temperature, the partition and
the assignment rule were **all fixed before the grid ran**, from offline
diagnostics that read no accuracy number (§3).

## 3. The knobs, fixed before the grid ran

`m4_offline.py` runs the exact live capsule code over the frozen 150 and answers
the questions that would otherwise be tuned on the test set. It generates no
token and takes 17 seconds.

| knob | value | how it was fixed |
|---|---|---|
| eviction rule | `maxred` | M3-v0 measured it best and cheapest (63.53 at r=16 vs `combo` 63.48, `lowimp` 61.18), and its oracle sat within 0.20 macro of the *teacher* eviction oracle — the axis is saturated |
| partition | most-square factorisation of `r` | rule-determined, spatially uniform, no seed, no search |
| `tau` | 0.05 | effective sample size 10.0 of the 64 tokens an r=16 cell pools — neither one-hot (τ=0.02 → ESS 2.7) nor uniform (τ=0.2 → ESS 41.3) |
| `tau_imp` | 0.0361 | **ESS-matched** to the residual merge: the importance softmax concentrating on the same 10.04 tokens per cell. Without this, `IMP-r16` is not a controlled comparison |
| norm restore | arm kept | a convex combination of 64 diverse features has median norm **0.573×** the tokens it pools (r=8 0.522, r=32 0.617) — a distribution shift the decoder sees, so `NORM-r16` prices it |
| empty cells | essentially never | 0 over 150 instances at every `r` in the offline pass; 2 cell-instances out of 1350 in the live grid (both `FPS-r16`), where the fallback handles them. The gate bounds the *rate*, not the count |

## 4. Protocol and controls

Every arm is exactly **256 visual tokens** and runs the full held-out 150
(TextVQA 50 / DocVQA 50 / OCRBench 50), one greedy generation per instance,
through the same engine and the same scorer as the stored baselines.

| arm | r | weights | grouping | what it is for |
|---|---|---|---|---|
| `REC0` | 0 | — | — | **identity gate** — must equal B2 |
| `REC-r8/16/32` | 8/16/32 | `softmax(u/τ)` | spatial | **the method** (primary: `REC-r16`) |
| `MEAN-r16` | 16 | uniform | spatial | plain mean merge — is any gain just "don't discard"? |
| `IMP-r16` | 16 | `softmax(imp/τ_imp)`, ESS-matched | spatial | merge on EADP importance instead of residual |
| `SHUF-r16` | 16 | `softmax(u_other/τ)` | spatial | **content-free control**: the weights come from another instance (M2's `C1-SHUF` idiom) |
| `ANCH-r16` | 16 | mean of the **retained** tokens in the cell | spatial | is it rejected evidence, or just a region summary? |
| `FPS-r16` | 16 | `softmax(u/τ)` | **feature-space** (farthest-point seeds) | the brief's assignment ablation |
| `NORM-r16` | 16 | `softmax(u/τ)` + norm restore | spatial | prices the shrinkage above |
| `EVICT-r8/16/32` | 8/16/32 | nothing put back — **256−r tokens** | — | accounting diagnostic, never a candidate |

`EVICT-r*` is what separates the two questions that would otherwise blur:
`REC − EVICT` is what the capsules add, `MEAN − EVICT` is what any merge adds. It
is not a candidate because it breaks the budget, and it is labelled a diagnostic
everywhere it appears.

`REC-r16` is the **pre-registered primary arm**, fixed before the grid ran on
three grounds that read no accuracy number: it is the midpoint of the `r` grid;
its effective sample size is 10.0 of the 64 tokens it pools; and r=16 is where
M3-v0's teacher oracle was strongest.

**On resolution, stated before the numbers.** M3-v0 measured the paired macro SE
on this 150 at ~2–4 points and the MDE at 80 % power at **7.0**; this grid
measures it at **2.9 and 8.2**. A +2 point estimate is therefore *not* evidence
on its own, and every delta below is printed with its paired bootstrap CI and
the McNemar p on discordant pairs. The brief's literal gate is reported alongside
that statistical reading, never instead of it.

---

## 5. Results

### 5.1 The grid

All numbers are held-out 150, greedy, one generation per instance, through the
M2 engine. `CI` is the paired bootstrap CI on the macro delta against B2;
`fix/brk` counts the instances where the arm's hit crosses zero relative to B2.

| arm | r | TextVQA | DocVQA | OCRBench | **macro** | ΔB2 | CI vs B2 | p | fix/brk | preds changed | ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| B0 | — | 73.600 | 85.274 | 68.000 | 75.625 | — | | | | | — |
| B1 | — | 59.400 | 61.891 | 62.000 | 61.097 | — | | | | | — |
| **B2** | — | 65.400 | 60.235 | 54.000 | **59.878** | — | | | | | — |
| `REC0` | 0 | 65.400 | 60.235 | 54.000 | 59.878 | +0.00 | [0.0, 0.0] | 1.000 | 0/0 | 0/150 | 0.00 |
| `REC-r8` | 8 | 65.400 | 59.568 | 60.000 | **61.656** | +1.77 | [−2.0, +5.9] | 0.754 | 6/4 | 22/150 | 4.69 |
| **`REC-r16`** | 16 | 64.800 | 57.352 | 56.000 | **59.384** | **−0.47** | **[−6.2, +5.3]** | **1.000** | **12/12** | 37/150 | 4.71 |
| `REC-r32` | 32 | 61.600 | 56.217 | 50.000 | 55.939 | −3.93 | [−9.1, +1.2] | 0.424 | 10/15 | 47/150 | 4.72 |
| `MEAN-r16` | 16 | 64.800 | 55.984 | 58.000 | 59.595 | −0.27 | [−5.7, +5.1] | 1.000 | 11/11 | 34/150 | 4.08 |
| `IMP-r16` | 16 | 61.600 | 56.274 | 56.000 | 57.958 | −1.92 | [−7.3, +3.4] | 0.523 | 9/13 | 36/150 | 4.71 |
| `SHUF-r16` | 16 | 64.800 | 56.984 | 58.000 | 59.928 | +0.08 | [−5.1, +5.2] | 1.000 | 11/10 | 36/150 | 4.70 |
| `ANCH-r16` | 16 | 65.400 | 57.352 | 56.000 | 59.584 | −0.28 | [−5.5, +5.0] | 1.000 | 11/10 | 33/150 | 2.98 |
| `NORM-r16` | 16 | 66.800 | 57.352 | 58.000 | **60.717** | +0.85 | [−4.5, +6.3] | 0.832 | 12/10 | 34/150 | 5.25 |
| `FPS-r16` | 16 | 62.800 | 54.925 | 56.000 | 57.908 | −1.93 | [−7.3, +3.4] | 0.678 | 10/13 | 41/150 | 5.50 |
| `EVICT-r8` | 8 | 71.400 | 57.568 | 62.000 | **63.656** | **+3.77** | [−0.2, +8.0] | 0.227 | 8/3 | 21/150 | 0.63 |
| `EVICT-r16` | 16 | 64.800 | 56.737 | 52.000 | 57.846 | −2.00 | [−7.2, +3.1] | 0.664 | 9/12 | 37/150 | 0.63 |
| `EVICT-r32` | 32 | 61.600 | 57.148 | 50.000 | 56.249 | −3.63 | [−9.1, +1.7] | 0.442 | 11/16 | 48/150 | 0.63 |

**Every CI in this table spans zero.** The largest positive point estimate in the
whole grid is `EVICT-r8` at +3.77, and its CI is [−0.2, +8.0]. The primary arm is
−0.47 with p = 1.000 and a perfectly balanced 12 fixed / 12 broken.

The arms are not inert — they change 21–48 of the 150 predictions — they are
**undirected**. REC-r16 moves 37 predictions and lands on exactly as many
instances fixed as broken.

### 5.2 The four questions the controls answer

Each control isolates one thing the capsules could have been contributing. All
are paired against the primary arm on the same 150.

| contrast | Δ macro | CI | fix/brk | reading |
|---|---:|---|---:|---|
| `REC-r16` − `MEAN-r16` | **−0.20** | [−3.33, +2.91] | 3/3 | **the residual weighting is worth nothing.** Uniform weights inside the cell do exactly as well |
| `REC-r16` − `SHUF-r16` | **−0.55** | [−3.67, +2.46] | 3/4 | **the residual map need not belong to this image.** Another instance's `u` does slightly *better* |
| `REC-r16` − `ANCH-r16` | −0.19 | [−2.87, +2.47] | 2/3 | **the dropped evidence is worth nothing.** Pooling the *retained* tokens in the cell does exactly as well |
| `REC-r16` − `IMP-r16` | +1.45 | [−2.12, +5.04] | 7/4 | merging on residual beats merging on EADP importance — the only control the method beats, and not resolvably |
| `REC-r16` − `EVICT-r16` | +1.53 | [−0.92, +4.21] | 4/1 | the capsules do recover part of the eviction's cost |

The `ANCH-r16` row is the sharpest of the four. A capsule built from the
**retained** tokens of the cell — carrying none of the rejected evidence REC
exists to carry — is statistically indistinguishable from the capsule built from
the rejected tokens (59.58 vs 59.38). Whatever a capsule contributes, it is not
the dropped information.

### 5.3 The `r` axis: more compression is monotonically worse

| r | `REC-r*` | `EVICT-r*` (256−r tokens) | capsule contribution (`REC − EVICT`) |
|---:|---:|---:|---:|
| 8 | **61.656** | **63.656** | **−2.00** |
| 16 | 59.384 | 57.846 | +1.53 |
| 32 | 55.939 | 56.249 | −0.31 |

`REC` falls monotonically in `r` (61.66 → 59.38 → 55.94). The brief asked which
`r` is best; the honest answer has two parts.

**r=8 is the best `r` and it is not evidence.** Its +1.77 has a CI of
[−2.0, +5.9] and p = 0.754. Its apparent advantage is also not the capsules':
`EVICT-r8` — the same eviction with *nothing put back*, at 248 tokens — scores
**63.66**, two points above `REC-r8`. At r=8 the capsules do not help; they cost.

**The capsule's own contribution does not have a consistent sign.** It is −2.00
at r=8, +1.53 at r=16 and −0.31 at r=32. A quantity that changes sign across the
axis it is supposed to be measured on is not being measured.

### 5.4 OCRBench

The brief singled this benchmark out: hard pruning has historically cost
fine-grained and structural evidence there, and if the capsules carried the
missed evidence the OCRBench profile should at least stop being negative.

| arm | OCRBench | vs B2 |
|---|---:|---:|
| B2 | 54.000 | — |
| B1 (facility) | 62.000 | +8.00 |
| B0 (unpruned) | 68.000 | +14.00 |
| `REC-r8` | 60.000 | +6.00 |
| `REC-r16` | 56.000 | +2.00 |
| `REC-r32` | 50.000 | **−4.00** |
| `MEAN-r16` | 58.000 | +4.00 |
| `SHUF-r16` | 58.000 | +4.00 |
| `ANCH-r16` | 56.000 | +2.00 |
| `EVICT-r8` | **62.000** | **+8.00** |

**The negative profile does not go away, and the capsules are not what moves
OCRBench.** REC-r32 is −4.00. REC-r16 is +2.00, but so is `ANCH-r16`, whose
capsules contain no dropped tokens at all. The best OCRBench arm in the entire
grid is `EVICT-r8` at **+8.00 — the arm with no capsules**, which is also the
arm that best matches B1's OCRBench (62.0 vs 62.0). Whatever recovers OCRBench
here, it is the eviction, not the compression.

### 5.5 Where the effect is not

The brief's premise was that a capsule should matter most where the eviction
removed something the retained set could not explain. That is directly testable
per instance: split the 150 by the mean `u` of the `r` tokens the arm evicted,
and read each instance's own (arm − B2) hit difference.

| half | mean `u` of evicted | Δ macro vs B2 |
|---|---:|---:|
| high-`u` half (n=75) | 0.285 | **−0.83** |
| low-`u` half (n=75) | 0.158 | −0.16 |

Pearson r = **−0.049**. The half where the eviction removed the *most*
unexplainable tokens is the half where REC does *worse*, and the correlation is
zero to two decimals. The mechanism the method is built on has no footprint in
the outcome.

### 5.6 An implementation defect, measured and fixed

The first M4 grid was **not reproducible**, and the size of that is worth
recording because it is larger than most of the effects measured above.

`build_capsules` pooled with `Tensor.index_add_`, which reduces with atomicAdd on
CUDA, so the summation order varied between launches. The capsules differed by
~2e-7 in fp32 — but a capsule is cast to bf16 (relative eps ~8e-3) before it
enters the decoder, so that difference occasionally flipped a rounding and with
it a greedy token. Running the *same* arms again in a fresh process:

| arm | run 1 macro | run 2 macro | Δ | hit agreement | prediction agreement |
|---|---:|---:|---:|---:|---:|
| `REC0` (identity) | 59.878 | 59.878 | **0.000** | 150/150 | 150/150 |
| `REC-r16` (primary) | 58.056 | 59.179 | **+1.123** | 148/150 | 145/150 |
| `SHUF-r16` | 59.261 | 60.595 | **+1.333** | 148/150 | 147/150 |

The identity arm — which does no pooling at all — is bit-reproducible across
processes. The capsule arms were not. The pooling was rewritten as a
deterministic group sum (`m4_common._group_sum`: sort by group, difference a
prefix sum — no atomics), verified bitwise-reproducible over repeated calls and
against a float64 reference to 3.4e-7, and the whole grid was re-run. **§5.1 is
the re-run.** The pre-fix grids are archived, trimmed to the fields the
comparison reads, as `m4_accuracy_nondet.json` and `m4_accuracy_rep_nondet.json`,
and the table above is recomputable from them by `m4_analyze.reproducibility()`.

The cost of the fix is latency, not accuracy: the deterministic reduction is
~2.7 ms slower than the atomic one (4.7 ms vs 2.0 ms per image).

**This is a run-to-run floor of 1.1–1.3 macro on any merge arm**, on top of the
±4–7 point sampling CI. Neither is small enough to hide behind.

## 6. Latency

The brief's §11 gates the full paired performance measurement on the accuracy
gate reaching PROMISING or STRONG. It did not, so **the paired protocol was not
run**, and no TTFT claim is made here. What the accuracy grid does record is the
capsule stage itself, measured inside the same CUDA-event window the M2/M3
harnesses read (`rec_ms`, median over the 150 instances, one window per image,
never nesting the selector or the scoring window):

| arm | capsule construction, median ms |
|---|---:|
| `REC-r8` | 4.69 |
| `REC-r16` | 4.71 |
| `REC-r32` | 4.72 |
| `MEAN-r16` | 4.08 |
| `NORM-r16` | 5.25 |
| `FPS-r16` | 5.50 |
| `ANCH-r16` | 2.98 |
| `EVICT-r*` (eviction only) | 0.63 |

For scale, B2's paired TTFT median on this machine is **278.6–279.0 ms**
(`m2_perf_paired.json`, 120 blocks). The capsule stage is therefore **+4.7 ms,
about +1.7 %**, and the LLM prefill is unchanged because the budget is still
exactly 256 visual tokens — the budget is asserted on every instance by gate 2.
The 256-token budget and B2-class latency are both preserved; that is the one
part of the brief's §11 that can be answered without the paired run.

## 7. Gates and verdict

Five integrity gates, all of which must hold before any accuracy number is
readable. **All five pass on the reported grid.**

| gate | result |
|---|---|
| `REC0_is_B2` | **150/150** hits, **150/150** prediction strings vs the stored B2 record |
| `budget_and_partition` | 1800/1800 instance-arm records: `n_kept` = 256 (or 256−r for the labelled diagnostics), `|A| = 256−r`, `A ⊆ S0`, `A ∪ E = S0`, `A ∩ E = ∅` |
| `capsules_non_degenerate` | 1350 capsule-arm instances checked; **2** cell-instances empty (rate 1.5e-3, bound 1e-2) — both in `FPS-r16`, where a farthest-point seed can end up with no dropped token assigned to it; the fallback handles them, and no `REC-r*` arm had one. Median capsule ESS 10.15 |
| `live_s0_equals_bank_s0` | 1800/1800 live base sets equal the offline bank's — the offline diagnostics describe the live arms |
| `shuf_is_content_free` | 150/150 SHUF instances drew their weights from a different instance; 0 self-sources |

**The brief's §10 gate, applied to the pre-registered primary arm:**

```
STRONG         macro >= 64 AND >= 2 points over plain MEAN merge
               -> NO.  59.384 < 64, and REC - MEAN = -0.20.
PROMISING      macro >= B2 + 2 AND residual weighting stably beats mean/importance
               -> NO.  REC - B2 = -0.47, and REC - MEAN = -0.20 with p = 1.000.
MERGING-ONLY   all merge methods clearly improve, but residual weighting does not
               beat plain merge
               -> NO.  No merge method improves on B2 at all: MEAN -0.27,
                  IMP -1.92, SHUF +0.08, ANCH -0.28, NORM +0.85, FPS -1.93.
                  "All merges clearly improve" is false; they are all within
                  +/-2 of B2 with CIs spanning zero.
REFUTED        neither REC nor mean stably beats B2
               -> YES.
```

**Verdict: REFUTED.** Per the brief, this stops the line.

> Compressing the rejected evidence into residual capsules does not improve B2.
> At the pre-registered primary arm the effect is −0.47 macro with a paired CI
> of [−6.2, +5.3] and a perfectly balanced 12 fixed / 12 broken. The residual
> weighting — the one thing that makes this REC rather than "keep some pooled
> tokens" — is worth −0.20 against uniform weights and −0.55 against weights
> drawn from another image, with 3 fixed / 3 broken in both. A control whose
> capsules contain none of the rejected evidence scores the same as the method
> (59.58 vs 59.38).

## 8. Visual cases

`m4_figures.py` renders, from the live pruner's own tensors:

* **`m4_mechanism_{TextVQA,DocVQA,OCRBench}.png`** — per benchmark: the image,
  B2's 256 anchors and the 16 evicted (blue/red), the residual map
  `u = 1 − max_{a∈A} cos(v, v_a)`, the 16 spatial cells with each capsule's
  effective sample size, the residual distribution over dropped vs evicted
  tokens, and the norm histogram that shows why `NORM-r16` exists (capsules
  cluster at ~0.57× the pooled-token mean norm).
* **`m4_profile.png`** — every arm's macro with its paired CI drawn, and the
  per-benchmark profile against B2.
* **`m4_diagnostics.png`** — capsule shrinkage, weight concentration vs `tau` and
  `r`, and the residual spread, from the offline pass.
* **`m4_cases.png`** — two instances where REC fixes what B2 broke and one where
  it breaks B2, with the residual map and cell layout for each.

The three case studies are worth one observation, because they show the
mechanism that *is* operating and it is not the one the method was built on.
Both fixes are cases where REC produced a **more complete** answer rather than a
different one — "plsun" → "tec sun" for a Tecsun radio, and "dynasty warriors
gundam" → "dynasty warriors gundam reborn" for a game cover — and the break is
the mirror image, "dakota digital" → "dak digital". The capsules perturb the
decode in both directions with no preference, which is what §5.1's 12/12 split
says numerically.

## 9. What this does and does not establish

**Established.**

1. **Carrying compressed rejected evidence does not recover B2's accuracy.**
   Across six merge variants (residual, uniform, importance, content-free,
   retained-only, feature-grouped) at r=16, macro spans 57.91–60.72 against B2's
   59.88, every CI spans zero, and no variant is resolvably different from B2 or
   from each other.
2. **The residual weighting is not the active ingredient.** It loses to uniform
   weights (−0.20, 3/3) and to another image's residual map (−0.55, 3/4). The
   brief's Q2 has a clean negative answer.
3. **More capsules is monotonically worse, but the capsules' own contribution
   is not monotone in `r`.** `REC` falls monotonically in `r` (61.66 → 59.38 →
   55.94), which is consistent with a capsule being a weaker carrier than the
   token it replaces — but the capsule's contribution *over the bare eviction*
   changes sign across `r` (−2.00 / +1.53 / −0.31), so the monotone trend cannot
   be attributed to the capsules alone. The eviction grows with `r` too.
4. **The mechanism has no per-instance footprint.** Instances whose evicted
   tokens were the *most* unexplainable do not benefit more (r = −0.049).
5. **The 256-token budget and B2-class latency survive.** The budget is asserted
   per instance; the added stage is +4.7 ms on a 278.6 ms TTFT.

**Not established, and not claimed.**

* No latency claim beyond the capsule stage's own window: the paired protocol
  was gated off by the accuracy result.
* Nothing about whether a *different* compression could work — a learned
  compressor, a residual-aware position encoding, or capsules that are not
  convex combinations are all outside this round's scope by construction.
* Nothing about the other benchmarks, budgets or selectors. This is B2, 256
  tokens, three benchmarks, 150 instances.

**One lead worth recording, which is not a REC result.** `EVICT-r8` — delete the
8 most redundant anchors and put *nothing* back, ending at 248 tokens — is the
largest positive effect in the grid (+3.77, CI [−0.2, +8.0], 8 fixed / 3 broken,
0.63 ms). It is not resolvable either, and it is not a candidate because it
breaks the budget. But it is the second time this project has seen a small-r
`maxred` eviction score well above B2 with a content-free replacement (M3-v0's
`RND-maxred-r8` = 61.82, `MG-maxred-r8` = 64.98), and it raises a question this
round did not ask: **how much of the "+3 to +5 at small r" that four rounds of
this project have chased is an eviction effect rather than a selection effect?**
That is a measurement, not a method, and it belongs to whoever picks this up.

## 10. The six questions

**Q1. Does retaining compressed dropped information clearly beat hard-pruning
B2?**
**No.** At the pre-registered primary arm, `REC-r16` = **59.384** against B2's
**59.878** — **−0.47 macro**, paired CI [−6.2, +5.3], McNemar p = 1.000, with 12
instances fixed and 12 broken. No arm in the grid, merge or control, has a CI
excluding zero. The only positive point estimates are `EVICT-r8` (+3.77, CI
[−0.2, +8.0]) and `REC-r8` (+1.77, CI [−2.0, +5.9]), and at r=8 the capsules
*reduce* the score by 2.00 relative to putting nothing back.

**Q2. Does residual-unexplainedness weighting beat plain mean merge?**
**No, and it is not close.** `REC-r16 − MEAN-r16` = **−0.20**, CI [−3.33,
+2.91], 3 fixed / 3 broken, p = 1.000. It also loses to the content-free control
(`SHUF-r16`, weights from another image): **−0.55**, 3/4. And it ties a control
that pools only *retained* tokens (`ANCH-r16`, −0.20). The residual weighting —
the whole of REC's novelty — is inert.

**Q3. Which `r` is best?**
**r=8, by 1.77 macro — and that is not a result.** Its CI is [−2.0, +5.9] with
p = 0.754, and the same eviction *without* capsules scores 63.66, two points
higher. The defensible statement is that `REC` degrades monotonically in `r`
(61.66 → 59.38 → 55.94) and that **the capsule's contribution over the bare
eviction has no consistent sign across `r`** (−2.00 / +1.53 / −0.31), so `r` is
not measuring what it was meant to measure. The brief's own resolution warning
applies: the MDE₈₀ on this 150 is 8.2 points, and the entire `r` spread is 5.7.

**Q4. Which benchmark gains most?**
**None gains, resolvably.** The per-benchmark profile of the primary arm against
B2 is TextVQA **−0.60**, DocVQA **−2.88**, OCRBench **+2.00** — one positive of
the three, inside a ±5 point CI, and matched or beaten by controls that carry no
dropped evidence. The brief's specific hope was that OCRBench would at least
stop showing a negative profile. It does not: `REC-r32` is **−4.00** on
OCRBench, `MEAN-r16` and `SHUF-r16` are both +4.00, and the best OCRBench arm in
the grid is **`EVICT-r8` at +8.00 — the arm with no capsules at all.**

**Q5. Is the 256-token budget and B2-class latency preserved?**
**Yes, on both counts, and this is the one part of the round that is
unambiguous.** The budget is asserted per instance and gated: 1500/1500
instance-arm records have `n_kept` = 256 for every candidate arm (256−r for the
labelled diagnostics), `|A| = 256−r`, and `A ∪ E = S0`. Latency: the capsule
stage costs **+4.7 ms** median against B2's **278.6–279.0 ms** paired TTFT
(+1.7 %), and the LLM prefill is unchanged because the token count is unchanged.
The full paired protocol was **not** run, because the brief gates it on the
accuracy gate and the accuracy gate failed.

**Q6. Should REC be promoted to a paper method?**
**No.** The brief's own branch applies: neither REC nor any merge variant stably
beats B2, so the line stops here. Three things would have had to be true and none
is: the primary arm is −0.47 rather than ≥ +2; the residual weighting is −0.20
against uniform weights rather than stably ahead; and the mechanism has no
per-instance footprint (r = −0.049 between evicted-token unexplainedness and the
outcome). The round's value is negative but specific — it removes
"compress-the-rejected-evidence" from the space of cheap fixes for hard pruning,
and it does so with a control (`ANCH-r16`) that shows the capsules' content is
not what moves the score.

**Recommended next step: none on this line.** Per the brief, this document stops
here and does not propose a learned compressor, a v1 of REC, or a return to
MissGuard. The one measurement worth someone's time is §9's closing question —
whether the small-`r` `maxred` eviction effect is real — and it is a measurement
about the *base selector's* eviction, not about compression.
