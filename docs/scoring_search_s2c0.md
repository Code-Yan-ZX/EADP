# S2-C0 — Forward-only proxy viability

**Scope.** Stage-2 S2-C0. One question: *can a forward-only visual-influence
signal taken from the first few LLM layers approximate the S2-B P1-G2 gradient
teacher, which costs 507 ms/image?*

Search space was held deliberately small, per the brief: two proxy families
(plus one exact-contribution extra), four layers, one query position, one
selector, no calibration sweep, no rollout, no multi-layer fusion, no learned
weights, no facility-location selector.

**Status.** Complete. Scripts: `Qwen_vl/scripts/discovery/s2c0_{sanity,
forward_proxy,analysis,run,consolidate}.py`. Deliverable:
`outputs/discovery/s2c0_forward_proxy.json`.

---

## VERDICT

```
S2-C0 NO-GO:
simple early-layer forward influence does not approximate
the gradient sensitivity teacher.
```

No proxy/layer combination passes the gate (§7). The failure is not marginal and
is not a threshold artifact: on the pilot, **every** forward proxy scores *below*
both the official EADP facility baseline and a content-free control, and all
three benchmarks regress (§5).

---

## 0. Executive summary

1. **The hand-computed attention row is the official one.** A bf16 mirror of the
   hand-rolled last-query attention matches the official
   `eager_attention_forward` row to **≤ 3.8e-6** on 8 layer×case checks (mostly
   exactly 0); the fp32 residual of 3.3e-3 is dtype rounding, not a derivation
   error (§1). GQA (`repeat_interleave`) and the M-RoPE rotary application are
   the model's own.
2. **Teacher agreement is weak.** Best per-image Spearman against P1-G2 is
   **0.346** (C_L8); best Top-256 overlap **0.413** (C_L8) against a 0.250
   chance rate. At the layers that matter causally (L2) agreement is 0.248 /
   0.352 (§3).
3. **Agreement and causal utility peak at *different* layers.** Spearman rises
   monotonically with depth (0.22→0.35 from L1 to L8) while causal AUROC peaks
   sharply at **L2** (0.68–0.71) and falls back to ~0.55–0.58 at L4/L8 (§3, §4).
4. **L2 clears the causal bar, but only at ~2 sd.** A_L2/B_L2/C_L2 reach block
   AUROC 0.673–0.705 and rank gain 0.111–0.120. The empirical null at n=8 is
   0.501 ± 0.092 (p95 0.651), so this is **+1.9 to +2.2 sd** — against the
   teacher's **+4.28 sd** (§4).
5. **The causal signal did not translate into accuracy at all.** Under the
   brief's fixed Top-256 selector, the best proxy reaches macro **45.27** on the
   pilot against the teacher's **75.15** and the official facility baseline's
   **61.10** (§5).
6. **Two controls make that negative interpretable, and both are unflattering.**
   (a) EADP's *own* score through the same Top-256 selector scores 42.75, so
   Top-K is not selectively punishing the proxies — but the proxies only beat it
   by 1.4–2.5 points. (b) The teacher's map **rotated onto a different image**
   scores 52.46, i.e. **7.2 points above the best proxy**. The proxies do not
   clear a content-free floor (§5.3).
7. **attention-only ≈ attention×value.** Variant A (≈ FastV) and variant B differ
   by ~0.01 AUROC and ~0.08 macro points everywhere. There is no
   attention×value advantage to report as a separate finding (§6).
8. **Cost is nonetheless attractive**: a prefix forward through layer 2 costs
   **23.6 ms** against the teacher's 507 ms (`§8`). The proxy idea was cheap and
   failed on signal, not on cost — which is the opposite of S2-B's failure mode.

---

## 1. Correctness — the attention row is the official one

The model runs SDPA, so the attention matrix is never materialised. A forward
pre-hook on `layers[L].self_attn` re-derives q/k/v with the layer's own
`q_proj`/`q_norm`/`k_proj`/`k_norm`/`v_proj` and the rotary embedding the model
passed in, then evaluates **only the last query row** against the 1024 visual
keys — a `[heads, 1024]` tensor, never a `seq × seq` matrix. The model's own
forward is untouched and stays on SDPA.

`--sanity` / `s2c0_sanity.py` checks the derivation against the official code
path: one layer's `config._attn_implementation` is switched to `eager` and a
forward hook captures the attention weights the official
`eager_attention_forward` actually returns. 2 cases × 4 layers:

| Check | Result |
|---|---|
| bf16 mirror of the hand-rolled row vs official row, max abs diff | **≤ 3.8e-6** (5 of 8 checks exactly 0) |
| fp32 hand-rolled row vs official row, max abs diff | ≤ 3.3e-3 |
| per-head Spearman (fp32 vs official), worst head, worst case | **0.999815** |
| official row sums to 1 over the visible prefix | 0.9973 – 1.0025 (bf16 jitter, not masking) |
| visual-slice attention mass, official vs hand-rolled | agrees to 1e-4 |

The bf16 mirror is the load-bearing column: it reproduces the official row
almost exactly, so the 3e-3 fp32 residual is the *reference's* bf16 rounding, not
a logic error. A wrong rotary application, a missing GQA repeat or a wrong
scaling would destroy the correlation, not perturb it at the 1e-3 level.

`config._attn_implementation` is restored after each check; nothing in the
official implementation is modified.

---

## 2. Proxies

Query position is the **first answer position** — the last token of the
prompt-only forward, exactly as S2-A's P1 objective. The prompt is built with the
same construction as `s2b_gradient_scores.py` (`inputs_embeds`, no `input_ids`,
no `image_grid_thw`), so the position encoding the proxy sees is identical to the
one the teacher saw.

With `a_{h,i}` the last query's attention onto visual token `i` at head `h`, and
`v_{h,i}` the corresponding value vector:

| variant | score |
|---|---|
| **A** | `mean_h a_{h,i}` — attention only |
| **B** | `mean_h a_{h,i} · ‖v_{h,i}‖₂` — attention × value norm |
| **C** | `‖ W_o [a_{1,i}v_{1,i} ; … ; a_{H,i}v_{H,i}] ‖₂` — exact output contribution |

Variant C was recorded because the brief permits it if cheap, and it is: the
heads are already materialised. One implementation trap is worth recording — the
official path does `attn_output.transpose(1, 2).reshape(*input_shape, -1)`, so
`o_proj` consumes heads in **head-major** order. A plain reshape of the
`[H, n_vis, D]` contribution tensor silently interleaves heads and tokens; it
must be `permute(1, 0, 2)` first. The first C run had this bug and scored *worse
than chance* (AUROC 0.255 at L2); after the fix it is the best proxy at L2
(0.705). The bug is noted here because a below-chance result is easy to mistake
for "this signal is anti-correlated" rather than "this code is wrong".

Layers `L ∈ {1, 2, 4, 8}` (0-based). Instances: the S2-B pilot (every 3rd of the
frozen 150 → 50/benchmark) plus the 15 S2-A causal cases, **165 instances**, one
forward each, no backward anywhere. **The 15 causal cases are disjoint from the
frozen 150** — they carry occlusion labels but have no frozen-bank accuracy
baseline, which is why the causal and accuracy tiers use different instance sets.

---

## 3. Teacher agreement vs P1-G2 (diagnostic)

Per image over the 150 pilot instances, then aggregated.

| arm | Spearman | median | p10 | p90 | Top-128 | Top-256 | Top-512 |
|---|---|---|---|---|---|---|---|
| A_L1 | 0.223 | 0.224 | 0.040 | 0.401 | 0.233 | 0.358 | 0.575 |
| B_L1 | 0.238 | 0.241 | 0.050 | 0.418 | 0.232 | 0.360 | 0.581 |
| C_L1 | 0.259 | 0.265 | 0.084 | 0.436 | 0.234 | 0.368 | 0.589 |
| A_L2 | 0.232 | 0.237 | 0.051 | 0.401 | 0.226 | 0.353 | 0.576 |
| B_L2 | 0.244 | 0.248 | 0.065 | 0.403 | 0.224 | 0.355 | 0.580 |
| C_L2 | 0.248 | 0.259 | 0.095 | 0.380 | 0.221 | 0.352 | 0.582 |
| A_L4 | 0.329 | 0.353 | 0.127 | 0.521 | 0.274 | 0.408 | 0.614 |
| B_L4 | 0.337 | 0.362 | 0.127 | 0.536 | 0.273 | 0.409 | 0.618 |
| C_L4 | 0.322 | 0.333 | 0.131 | 0.515 | 0.262 | 0.399 | 0.612 |
| A_L8 | 0.319 | 0.337 | 0.113 | 0.524 | 0.271 | 0.406 | 0.609 |
| B_L8 | 0.343 | 0.359 | 0.131 | 0.548 | 0.271 | 0.412 | 0.619 |
| **C_L8** | **0.346** | 0.363 | 0.127 | 0.543 | **0.267** | **0.413** | **0.621** |

*Chance Top-K overlap is k/1024: 0.125 / 0.250 / 0.500.*

Two things to read off:

* The agreement is **weak in absolute terms**. Top-256 overlap of 0.41 against a
  0.25 chance rate means the proxy's 256 tokens and the teacher's 256 tokens
  share only ~40 of 256; the remaining 60 % are different tokens. Spearman 0.35
  is a real but modest rank correlation.
* **Agreement increases with depth while causal utility peaks at L2** (§4). The
  proxy that best approximates the teacher (C_L8) is one of the *worst* at
  localizing the causally necessary blocks. Teacher agreement is therefore not a
  usable selection criterion for this family — worth recording, since it is the
  intuitive thing to optimize.

This tier is diagnostic only, as the brief specifies, and does not gate.

---

## 4. Causal evaluation

Identical metrics to S2-A, reused verbatim from `scoring_search_s1.evaluate`.
Primary tier `nNec <= 256`, n = 8.

| method | mean rank | Δ vs official | impr/wors | AUROC | AP | R@128 | R@256 | R@512 |
|---|---|---|---|---|---|---|---|---|
| official EADP | 0.5466 | — | — | 0.380 | 0.207 | 0.051 | 0.151 | 0.425 |
| S1-best | 0.4959 | −0.0507 | 6/2 | 0.519 | 0.256 | 0.099 | 0.216 | 0.513 |
| **P1-G2 teacher** | **0.3142** | **−0.2325** | **8/0** | **0.895** | 0.406 | 0.296 | 0.492 | 0.773 |
| P2-G2 | 0.3062 | −0.2405 | 8/0 | 0.878 | 0.486 | 0.315 | 0.526 | 0.779 |
| A_L1 | 0.5013 | −0.0453 | 6/2 | 0.466 | 0.318 | 0.109 | 0.232 | 0.512 |
| B_L1 | 0.5006 | −0.0460 | 6/2 | 0.483 | 0.313 | 0.099 | 0.226 | 0.511 |
| C_L1 | 0.4940 | −0.0526 | 6/2 | 0.479 | 0.321 | 0.107 | 0.251 | 0.518 |
| **A_L2** | 0.4352 | −0.1114 | 6/2 | **0.681** | 0.625 | 0.171 | 0.315 | 0.594 |
| **B_L2** | 0.4331 | −0.1136 | 6/2 | 0.673 | 0.619 | 0.175 | 0.317 | 0.594 |
| **C_L2** | **0.4264** | **−0.1202** | 7/1 | **0.705** | 0.633 | 0.176 | 0.324 | **0.618** |
| A_L4 | 0.4604 | −0.0862 | 7/1 | 0.556 | 0.520 | 0.137 | 0.289 | 0.552 |
| B_L4 | 0.4610 | −0.0857 | 7/1 | 0.547 | 0.508 | 0.134 | 0.283 | 0.543 |
| C_L4 | 0.4687 | −0.0779 | 6/2 | 0.513 | 0.500 | 0.124 | 0.273 | 0.538 |
| A_L8 | 0.4676 | −0.0790 | 6/2 | 0.550 | 0.606 | 0.134 | 0.287 | 0.555 |
| B_L8 | 0.4677 | −0.0789 | 7/1 | 0.577 | 0.621 | 0.136 | 0.292 | 0.548 |
| C_L8 | 0.4621 | −0.0846 | 7/1 | 0.573 | 0.631 | 0.143 | 0.299 | 0.554 |

### 4.1 The causal bar is cleared at L2 — at ~2 sd

Against the empirical null from S2-A (random block-AUROC at n=8: mean 0.501,
sd 0.092, p95 0.651):

| score | AUROC | distance from null |
|---|---|---|
| P1-G2 teacher | 0.895 | **+4.28 sd** |
| C_L2 | 0.705 | +2.22 sd |
| A_L2 | 0.681 | +1.96 sd |
| B_L2 | 0.673 | +1.87 sd |
| best non-L2 (B_L8) | 0.577 | +0.83 sd |
| official EADP | 0.380 | −1.32 sd |

So gate condition 1 (`AUROC ≥ 0.65` **or** rank gain ≥ 0.10) is passed by
A_L2/B_L2/C_L2 — and the brief's 0.65 bar sits at almost exactly the null's p95,
as S2-A noted. The teacher is 4.28 sd out; the proxies are 2 sd out. The
difference between "clears the bar" and "is the bar" is the whole story here.

**L1 is at chance (0.466–0.483) and L4/L8 recover to only 0.51–0.58.** The effect
is confined to a single layer, L2, which is a thin basis for a method. For
context, the mean fraction of the last query's attention mass that lands on the
1024 visual tokens is 0.114 at L1, **0.125 at L2**, 0.067 at L4 and 0.090 at L8
(seq len ≈ 1057) — L2 is not distinguished by visual mass, so the L2 peak is not
explained by "L2 looks at the image more".

### 4.2 Per-case, primary tier

| case | nNec | official | P1-G2 | A_L2 | B_L2 | C_L2 |
|---|---|---|---|---|---|---|
| DocVQA_VAL_241 | 64 | 0.6255 | **0.1849** | 0.3635 | 0.3545 | 0.3441 |
| DocVQA_VAL_341 | 128 | 0.4018 | 0.3186 | 0.4262 | 0.4291 | 0.4632 |
| OCRBench_222 | 256 | 0.6253 | **0.3044** | 0.5183 | 0.5192 | 0.4902 |
| TextVQA_VAL_105 | 64 | 0.5431 | 0.4176 | 0.4595 | 0.4619 | 0.4537 |
| TextVQA_VAL_285 | 64 | 0.5247 | **0.2616** | 0.2810 | 0.2709 | 0.2771 |
| TextVQA_VAL_400 | 64 | 0.5566 | **0.3099** | 0.4782 | 0.4734 | 0.4584 |
| TextVQA_VAL_505 | 128 | 0.6231 | **0.3389** | 0.4760 | 0.4701 | 0.4529 |
| TextVQA_VAL_609 | 256 | 0.4729 | 0.3776 | 0.4792 | 0.4854 | 0.4720 |

The proxies beat the teacher on **no** case and beat the official score on 6 of 8
for A_L2/B_L2 — but the two regressions are the two cases where the official
score already had the least headroom, and their margins over official are small
(0.02–0.10 rank) where the teacher's are large (0.15–0.44).

---

## 5. Accuracy translation

Selector **Top-K @ 256** for every arm, per the brief: S2-B already established
that Top-K costs the gradient teacher only ~1.4 points against facility
location, so the selector is not the variable under test and facility location
must not leak into the proxy comparison. Calibration is the identity — the tested
calibrations are monotone rescalings and Top-K depends only on the ranking.

Layer choice: the brief asks for the accuracy run only on each family's best
layer, selected by causal AUROC. **All three families peak at L2**
(A 0.681, B 0.673, C 0.705), so A_L2 / B_L2 / C_L2 were run. The other nine
layer×variant arms are causal-only.

### 5.1 Results — pilot, n = 50/benchmark, T = 256

| arm | TextVQA | DocVQA | OCRBench | macro | Δ vs official | retained (brief) | retained (selector-matched) |
|---|---|---|---|---|---|---|---|
| official EADP **facility** @256 | 59.400 | 61.891 | 62.000 | 61.097 | — | — | — |
| *gradient + facility (S2-B ref)* | *77.000* | *82.396* | *74.000* | *77.799* | *+16.702* | *118.8 %* | — |
| **P1-G2 teacher + TopK** | 77.000 | 80.450 | 68.000 | **75.150** | **+14.053** | 100 % | 100 % |
| *P1-G2 shuffled + TopK* | *63.600* | *45.769* | *48.000* | *52.456* | *−8.641* | *−61.5 %* | *+29.9 %* |
| C_L2 + TopK | 58.200 | 37.594 | 40.000 | 45.265 | −15.833 | −112.7 % | +7.8 % |
| B_L2 + TopK | 52.400 | 40.260 | 40.000 | 44.220 | −16.877 | −120.1 % | +4.5 % |
| A_L2 + TopK | 52.200 | 40.226 | 40.000 | 44.142 | −16.955 | −120.7 % | +4.3 % |
| *EADP's own score + TopK* | *41.400* | *42.860* | *44.000* | *42.753* | *−18.344* | *−130.5 %* | *0 %* |

`retained (brief)` = `(proxy − official_facility) / (teacher_topk − official_facility)`,
the brief's formula. `retained (selector-matched)` = the same with every map
scored through the *same* Top-K selector, which separates score quality from
selector quality; the brief's formula charges the proxy for Top-K's own cost.

### 5.2 Two harness identity checks

Both pass exactly, so the numbers above are attributable to the score maps and
not to the runner:

1. **EADP's own score through this runner's Top-K path reproduces Stage-1's
   `b256|topk|` arm on the pilot indices** — 41.400 / 42.860 / 44.000, identical
   to 1e-9 on all three benchmarks.
2. **The teacher arm re-run here reproduces S2-B's own `topk|C3` arm** —
   77.000 / 80.450 / 68.000, identical on all three.

### 5.3 Controls — why this is a signal failure, not a selector artifact

The obvious objection to §5.1 is "Top-K is the problem". It is not, and the two
controls pin that down:

* **Top-K is not selectively punishing the proxies.** Under the identical
  selector, EADP's *own* importance score scores 42.75 — *below* every forward
  proxy. So the proxies do carry a little real information (+1.4 to +2.5 points
  over the incumbent score under the same selector). It is just very little.
* **But they do not clear a content-free floor.** The teacher's map **rotated
  onto a different image** — same value distribution, no content — scores
  **52.46**, which is **7.2 points above the best proxy**. So the proxies are
  worse than using the teacher's magnitude distribution on the wrong image. This
  is the decisive control: whatever signal the L2 proxies carry is worth less
  than nothing once it displaces the teacher's value structure.

Under Top-K, the gradient teacher gains +32.4 points over the incumbent score
(75.150 vs 42.753). The forward proxies keep **4.3 % (A_L2), 4.5 % (B_L2) and
7.8 % (C_L2)** of that. The gate asks for ≥ 50 %.

### 5.4 Run-to-run context

S2-B's `facility|C3` arm is included for reference: 77.799 macro. Its presence in
the same table is a reminder that the *same* score, through a *different*
selector, retains 118.8 % of the teacher's gain. Selector quality is worth far
more than the entire difference between these proxies.

---

## 6. The FastV control the brief asked for

There is **no FastV (or equivalent early-attention scoring) implementation in
this repository** — a repo-wide case-insensitive search for `fastv`, `fast_v`,
`attention prun`, `token merg` matches only the S2-C0 script itself. Nothing was
reused or compared against directly.

However, **variant A is the FastV construction by definition** (prune by the
attention the last query receives, before/inside the LLM), so the comparison the
brief asks for is available internally: does attention × value beat
attention-only?

| comparison | A (attention only) | B (attention × ‖V‖) | C (exact output contribution) |
|---|---|---|---|
| causal AUROC @L2 | 0.681 | 0.673 | 0.705 |
| causal rank gain @L2 | 0.1114 | 0.1136 | 0.1202 |
| teacher Spearman @L2 | 0.232 | 0.244 | 0.248 |
| pilot macro @L2 | 44.142 | 44.220 | 45.265 |

**A and B are indistinguishable** (ΔAUROC 0.008, Δmacro 0.08). The value-norm
weighting buys nothing. C is marginally ahead of both on every causal axis, but
the margin (ΔAUROC 0.024 over A at n=8) is well inside the null's spread.

Per the brief, no novelty claim is made, and there is no A-vs-B gap worth
reporting as a separate finding. That itself is a result: the reason attention×V
does not help is presumably the same reason the whole family fails — the L2
attention row is not aligned with the causally necessary regions to begin with.

---

## 7. Gate

Criteria (brief §8): (1) primary causal block AUROC ≥ 0.65 **or** necessary-rank
improvement ≥ 0.10; (2) pilot accuracy retains ≥ 50 % of the P1-G2 Top-K gain;
(3) no more than one benchmark regressing by > 2 points.

| arm | rank gain | AUROC | AUROC vs teacher | Spearman | Top-256 | macro Δ | retained (brief) | retained (sel-matched) | regressions >2pt | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| A_L1 | 0.0453 | 0.466 | 0.52 | 0.223 | 0.358 | — | — | — | — | not run |
| B_L1 | 0.0460 | 0.483 | 0.54 | 0.238 | 0.360 | — | — | — | — | not run |
| C_L1 | 0.0526 | 0.479 | 0.54 | 0.259 | 0.368 | — | — | — | — | not run |
| A_L2 | 0.1114 | 0.681 | 0.76 | 0.232 | 0.353 | −16.955 | −120.7 % | +4.3 % | **3** | **fail** |
| B_L2 | 0.1136 | 0.673 | 0.75 | 0.244 | 0.355 | −16.877 | −120.1 % | +4.5 % | **3** | **fail** |
| C_L2 | 0.1202 | **0.705** | 0.79 | 0.248 | 0.352 | −15.833 | −112.7 % | +7.8 % | **2** | **fail** |
| A_L4 | 0.0862 | 0.556 | 0.62 | 0.329 | 0.408 | — | — | — | — | not run |
| B_L4 | 0.0857 | 0.547 | 0.61 | 0.337 | 0.409 | — | — | — | — | not run |
| C_L4 | 0.0779 | 0.513 | 0.57 | 0.322 | 0.399 | — | — | — | — | not run |
| A_L8 | 0.0790 | 0.550 | 0.61 | 0.319 | 0.406 | — | — | — | — | not run |
| B_L8 | 0.0789 | 0.577 | 0.64 | 0.343 | 0.412 | — | — | — | — | not run |
| C_L8 | 0.0846 | 0.573 | 0.64 | 0.346 | 0.413 | — | — | — | — | not run |

Condition 1 passes for A_L2/B_L2/C_L2 only. **Conditions 2 and 3 fail for every
arm**, and condition 2 fails by two orders of magnitude in the wrong direction:
the retained gain is *negative* under the brief's formula and ≤ 7.8 % under the
selector-matched one. C_L2 has the best of everything and is still 15.8 points
below the official facility baseline.

```
S2-C0 NO-GO:
simple early-layer forward influence does not approximate
the gradient sensitivity teacher.
```

Per the brief: no mid-layer pruning integration, no student training, no
gradient-teacher distillation. Stopping here.

---

## 8. Cost

Measured on the unpruned 1024-token forward, A40 46 GB, SDPA, batch 1.

| quantity | ms |
|---|---|
| prompt-only full forward, 36 layers | **221.6** |
| prefix forward through layer 1 (2/36 layers) | **14.1** |
| prefix forward through layer 2 (3/36 layers) | **23.6** |
| prefix forward through layer 4 (5/36 layers) | **39.0** |
| prefix forward through layer 8 (9/36 layers) | **65.7** |
| P1-G2 teacher (one forward + one backward) | 507 (S2-A) |
| reference: EADP pruning overhead total | 42.0 (Stage-1) |

The prefix latencies are measured with an early-exit hook (a forward hook that
aborts the layer loop), so they are real measurements of "embeddings + layers
0..L", not `(L+1)/36` extrapolations. They exclude the vision tower (~117 ms),
which every method and the unpruned baseline pay identically.

**An L2 proxy would cost 23.6 ms — 21× less than the teacher and 0.56× the entire
official EADP pruning overhead.** So unlike S2-B, cost is not the blocker here:
the cheap signal exists, it is simply not good enough. That inverts S2-B's
conclusion and is the most useful thing to carry out of this stage.

Two caveats on that number, per the brief:

* These are **scoring** costs, not deployable end-to-end latency. A real system
  that scores from a prefix forward and then re-runs the pruned model still pays
  the prefix forward on top of the full pruned prefill; the prefix is not
  amortised. Do not quote 23.6 ms as the method's cost.
* The prefix forward is over the **unpruned** 1024-token sequence, which is the
  regime a pruner must operate in — it cannot be shortened by pruning first.

---

## 9. What carries forward

1. **The negative is decisive and cheap to state**: under a fixed selector, the
   best forward proxy keeps 4.3–7.8 % of the gradient teacher's accuracy gain
   and sits *below* a content-free rotation of the teacher's own map. This is not
   a threshold or a tuning question.
2. **Causal block AUROC at n=8 did not predict 150-instance accuracy at all.**
   The arm with the best causal AUROC (C_L2, 0.705) and the arm with the best
   accuracy (C_L2) coincide here, but all three L2 arms are within 1.1 macro
   points of each other while spanning 0.673–0.705 AUROC; more importantly, the
   *teacher* at 0.895 is 30 points better in accuracy than arms at 0.68–0.71.
   With a null sd of 0.092 at n=8, the causal tier has essentially no resolution
   between "clears the bar" and "noise". Any future stage should not select on
   n=8 AUROC.
3. **Teacher agreement is anti-indicative here.** Agreement rose with depth while
   causal utility and accuracy both peaked at L2. If a distillation stage is ever
   attempted, its loss should not be "match the teacher's ranking" — that is
   satisfied best by the layers that are least useful.
4. **attention × value ≈ attention only.** The FastV-style baseline is not
   beaten by the value-weighted variant; if a FastV comparison is ever written
   up, these are the numbers, and the honest statement is "no difference".
5. **The L2-only localisation is thin.** A single early layer shows the effect;
   its neighbours do not. Whatever L2 is doing, it is not a property that
   survives moving one layer in either direction, and the visual-attention mass
   does not explain the peak.

The brief's next-stage decision — whether to attempt gradient-teacher
distillation — is left open. Nothing in S2-C0 argues that a forward-only
*richer* signal (multi-layer, rollout, learned combination) would fail; what it
rules out is that the simplest early-layer influence signal, at the layers and
query position tested, substitutes for the teacher.

---

## 10. Reproduce

```
source ~/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
cd /media/disk2/YZX/research/EADP/Qwen_vl

python scripts/discovery/s2c0_sanity.py          # attention-row correctness (GPU: 2 cases)
python scripts/discovery/s2c0_forward_proxy.py --prefix-latency   # 165 instances, no backward
python scripts/discovery/s2c0_analysis.py        # teacher agreement + causal metrics

# accuracy: proxy arms, then the teacher upper bound and the two controls
python scripts/discovery/s2c0_run.py --arms A_L2 B_L2 C_L2 --tag s2c0_pilot
python scripts/discovery/s2c0_run.py --arms P1G2 official      --tag s2c0_pilot
python scripts/discovery/s2c0_run.py --arms P1G2 --shuffle 7   --tag s2c0_pilot

python scripts/discovery/s2c0_consolidate.py     # retained gain + gate + deliverable
```

Or in one step: `bash scripts/discovery/run_s2c0.sh` (sanity + proxy only).

Deliverable: `outputs/discovery/s2c0_forward_proxy.json` — pilot accuracy per
arm with per-dataset deltas and bootstrap CIs, gate outcomes, the causal and
agreement tables, the sanity record, and timings. Supporting:
`s2c0_forward_proxy.npz` (all 1980 score vectors), `s2c0_forward_proxy_meta.json`,
`s2c0_sanity.json`, `s2c0_proxy_analysis.json`, `s2c0_pilot.json`.
