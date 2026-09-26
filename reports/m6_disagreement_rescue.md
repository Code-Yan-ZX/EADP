# M6 — Disagreement-Rescue Composition

**Stage:** M6, a new formulation, first round. MissGuard (M3), residual capsules
(M4), SafeTrim (M5), conditional utility (S3-A) and bundle prediction (S3-B) are
closed and this document does not reopen any of them. **Status:** Phase 1
complete; **Phase 2/3/4 not run**, because the pre-registered Phase-1 gate
failed. Everything here is forward-only and training-free.

**The question this document answers, and the only one.**

> If a token B2 dropped ranks low in EADP importance but high under some *other*
> cheap forward-only proxy, does that cross-proxy **disagreement** enrich the set
> of tokens the gradient teacher would have kept — more than random, more than
> EADP-next, and more than the best single proxy?

**The answer, in one line.** **No. Every disagreement rule is *worse* than the
best single proxy at every rescue size that matters (k = 8/16/32), with paired
bootstrap CIs excluding zero; at k = 64 a small win appears (+0.015 to +0.021
recall) but k = 64 is a quarter of the budget. Verdict: REFUTED.**

---

## 1. Executive Summary

- **Hypothesis refuted.** Cross-proxy disagreement does **not** enrich
  oracle-critical misses. At k = 8 the best disagreement rule (D4) reaches
  **0.1595** head recall against **0.2089** for the single best proxy — a
  **−0.049** paired deficit, 95 % CI [−0.063, −0.036], losing on 86 instances
  and winning on 20. The same holds at k = 16 (−0.017) and k = 32 (−0.002, a
  tie). At k = 64 disagreement wins by +0.015 to +0.021.
- **Best method found is not the hypothesis.** The strongest cheap signal in the
  audit is `cos_s0c` **inverted** — *low* cosine to the retained-set centroid —
  a single proxy, not a composition. It finds 20.1× the chance rate of teacher
  head tokens at k = 8 (0.2089 vs 0.0102) and beats EADP-next by +0.198.
- **But even that is predicted not to translate.** Its rescue sits at mean
  teacher rank **131.0** (median 75) — *deeper* than the M3-v0 learned student's
  73.7, which was measured downstream at ≈ 0 value, and far from the oracle's
  3.5–15.5 where the +7.7/+14.9 macro ceiling lives.
- **Downstream was not run.** The brief gates Phase 2/3 on Phase 1 passing; it
  did not pass. No macro number is claimed, and no TTFT for a method that was
  never built.
- **Cost of the audit:** +**1.73 ms** per instance marginal (nn4_recon 1.48 ms +
  cos_s0c 0.25 ms); the other 25 columns are free.
- **The precise failure mechanism.** The proxies' *union* does hold far more head
  tokens than any single proxy — an oracle combiner would reach 0.2931 vs 0.2069
  at k = 8. But disagreement selects a **70 %-different** token set from the best
  proxy and the swap is harmful on 50.9 % of instances against helpful on 14.0 %.
  The extra head tokens other proxies contribute are not identifiable by any
  cheap score.

**Verdict: REFUTED. Should we continue this line into a paper method? NO.**

---

## 2. Existing Setup

| item | value |
|---|---|
| backbone | Qwen3-VL-8B-Instruct, `attn_implementation=sdpa` |
| resolution / visual tokens | 1024 × 1024 → **1024** merged visual tokens (32 × 32 grid) |
| token budget | **256** throughout |
| base selector | B2 = official EADP importance + `block8` block-parallel coverage greedy |
| engine | `m2_gdep.GDEPEngine`, `prellm` mode |
| datasets | TextVQA_VAL, DocVQA_VAL, OCRBench |
| decode | greedy, one generation per instance, `max_new_tokens=2048` |
| scorer | `scoring.per_sample_hits` (VQA score / ANLS / substring) |

**Reference arms** (M2, stored, reproduced 150/150 hit-for-hit):

| arm | TextVQA | DocVQA | OCRBench | macro | TTFT (ms) |
|---|---:|---:|---:|---:|---:|
| B0 full-1024 | 73.60 | 85.27 | 68.00 | **75.62** | 397.9 |
| B1 official EADP facility@256 | 59.40 | 61.89 | 62.00 | **61.10** | 310.1 |
| B2 EADP block8@256 | 65.40 | 60.24 | 54.00 | **59.88** | 278.6 |

**Panels.** The frozen bank is **450 instances** (fit 240 / val 60 / test 150;
150 per benchmark). Phase 1 is a pure *ranking* measurement, so it uses:

| panel | n | role |
|---|---:|---|
| `holdout210` | 210 | val + test — **every headline number in this report** |
| `all450` | 450 | used only for the per-benchmark split and the pool control |
| `test150` | 150 | the historical panel of M2/M3/M4/M5, reported for continuity |

**Teacher / oracle.** The frozen **P1-G2** gradient saliency
(`s2b_gradient_scores.npz`): `score_i = Σ_d |∂J/∂v_i,d · v_i,d|`, `J` the argmax
logit at the first answer position; one forward and one backward per instance,
offline, once. Verified **bit-identical** to the bank's `g2` column on all 450
(worst |Δ| = 0). It is an offline label only and never enters a deployable path.

**Definitions**, inherited from M3-v0 so numbers are directly comparable:

```
D        = [1024] \ S0, the 768 tokens B2 dropped          (per instance)
tr(i)    = teacher rank of i within D, 0 = teacher's best dropped token
head_q   = { i in D : tr(i) < q }                          the oracle-critical set
recall@k = |pick_k ∩ head_k| / k, per instance, then averaged
```

`recall@k` against `head_k` **is** M3-v0's `overlap@r`. Chance is **k/768**. No
metric anywhere in this stage pools a global top-k across instances.

---

## 3. Proxy Audit

The M5 bank (`m5_bank.npz`) already caches **27 cheap forward-only columns for
all 1024 tokens** on all 450 instances — the single most useful asset this round
inherited. Per the brief, **6 are declared** and the rest are reported only as a
supplementary ranking (T4 lists every one). Declaration was by mechanism
coverage, not by score.

| proxy | definition | source | forward-only | cached | marginal cost |
|---|---|---|---|---|---|
| `imp` | EADP importance, post-smoothing ^β — the B2 score | pruner score pass | yes | yes | 0 (already computed) |
| `cos_s0c` | cosine of the token to the centroid of S0's unit features | tower output + S0 | yes | yes | **0.25 ms** (measured) |
| `nn4_recon` | ‖v − proj_span(4 nearest retained)‖ / ‖v‖ | tower output + S0 | yes | yes | **1.48 ms** (measured) |
| `red_s0_top8` | mean of the 8 largest cosines to other retained tokens | selector's sim matrix | yes | yes | 0 (reduction over sim) |
| `loc_std` | spread of the EADP local match across instruction tokens | pruner score pass | yes | yes | 0 (already computed) |
| `vis_norm` | ‖v‖ of the post-merger tower feature | tower output | yes | yes | 0 (already computed) |

Measured total marginal cost: **1.73 ms/instance** (`m6_cost.json`), i.e.
**+0.62 %** of B2's 278.6 ms TTFT. Only `nn4_recon` and `cos_s0c` do work the
incumbent's pruner does not already do; the other 25 columns are reductions over
tensors it already produces.

**Orientations** (the only fitted quantity in this stage — one sign per proxy,
fitted on the **fit** split, never on the split it is reported on):

| proxy | fitted sign | fit recall@16 | reading |
|---|---:|---:|---|
| `imp` | **−1** | 0.1190 | the teacher's head is EADP-**unimportant** |
| `cos_s0c` | **−1** | 0.1802 | low cosine to the retained centroid = representationally novel |
| `nn4_recon` | **+1** | 0.0820 | high residual = not explained by S0 |
| `red_s0_top8` | **−1** | 0.1279 | low redundancy with S0 |
| `loc_std` | **+1** | 0.0555 | the EADP local match is diffuse |
| `vis_norm` | **−1** | 0.0651 | smaller magnitude |

### 3.1 Headroom: EADP's own ranking is anti-correlated with the teacher's head

This is the fact that makes EADP-next a weak baseline rather than a strong one.

| teacher head | median EADP percentile | frac ≥ 90th | frac ≥ 75th | frac inside EADP-next-64 | chance |
|---|---:|---:|---:|---:|---:|
| top-8 dropped | **30.2** | 0.0589 | 0.1481 | **0.0639** | 0.0833 |
| top-16 dropped | 32.7 | 0.0612 | 0.1518 | 0.0657 | 0.0833 |
| top-32 dropped | 34.4 | 0.0645 | 0.1604 | 0.0658 | 0.0833 |
| top-64 dropped | 36.1 | 0.0657 | 0.1669 | 0.0687 | 0.0833 |

The teacher's best dropped tokens sit **below the median** of EADP importance,
and the incumbent's own next-64 captures them at **below chance**. There is
therefore genuine headroom for *some* other signal — which is exactly what made
this hypothesis worth testing.

### 3.2 Are the proxies real, or restatements of each other?

5 equal-count strata per instance; `ratio` = observed recall@8 over the exact
random-within-stratum expectation. A ratio near 1 means the proxy is a
restatement of the stratifying variable.

| proxy | dir | stratified by `vis_norm` | by `cos_s0c` | by `imp` |
|---|---|---:|---:|---:|
| `cos_s0c` | low | **6.29** | 4.41 | 5.54 |
| `red_s0_top8` | low | 4.69 | 3.52 | 3.90 |
| `imp` | low | 4.62 | 3.64 | 2.51 |
| `nn4_recon` | high | 3.13 | 2.35 | 2.38 |
| `loc_std` | high | 2.56 | 2.18 | 2.05 |
| `vis_norm` | low | 1.24 | 2.32 | 2.08 |

Every declared proxy survives the others as a control, and `cos_s0c` is not a
magnitude artefact (6.29× within `vis_norm` strata). These are **distinct,
genuine** signals — which is what makes the negative result below about
*combination*, not about signal absence.

---

## 4. Critical-Miss Enrichment

All values on the **held-out 210**. `recall` = mean over instances of
|pick ∩ teacher-top-k dropped| / k. `meanTR` = mean teacher rank of picked
tokens (0 = the teacher's best dropped token). Δ columns are paired bootstrap on
the same 210; `*` marks a CI excluding zero.

### k = 8

| rule | recall | enrich | meanTR | medTR | Δ vs random | Δ vs EADP-next | Δ vs best single |
|---|---:|---:|---:|---:|---|---|---|
| R0 random (mean of 20 seeds) | 0.0102 | 1.00 | — | — | — | — | — |
| R1 EADP-next | 0.0113 | 1.09 | 400.1 | 408.0 | +0.0011 | — | −0.1976* |
| **R2 `cos_s0c` (best single)** | **0.2089** | **20.06** | **131.0** | 75.0 | +0.1987* | +0.1976* | — |
| R2 `red_s0_top8` | 0.1482 | 14.23 | 217.8 | 149.0 | +0.1380* | +0.1369* | −0.0607* |
| R2 `imp` | 0.0923 | 8.86 | 198.4 | 138.5 | +0.0821* | +0.0810* | −0.1167* |
| R2 `nn4_recon` | 0.0750 | 7.20 | 283.2 | 253.0 | +0.0648* | +0.0637* | −0.1339* |
| R2 `vis_norm` | 0.0423 | 4.06 | 186.6 | 142.0 | +0.0321* | +0.0310* | −0.1667* |
| R2 `loc_std` | 0.0393 | 3.77 | 282.9 | 241.5 | +0.0291* | +0.0280* | −0.1696* |
| R3 union, 2 strongest | 0.2024 | 19.43 | 160.8 | 86.5 | +0.1922* | +0.1911* | −0.0065 |
| R3 union, 3 strongest | 0.1940 | 18.63 | 163.3 | 92.0 | +0.1839* | +0.1827* | −0.0149* |
| R3 union, all 6 | 0.1750 | 16.80 | 185.0 | 112.0 | +0.1648* | +0.1637* | −0.0339* |
| R4 **D4** binary disagreement | 0.1595 | 15.31 | 197.2 | 120.5 | +0.1493* | +0.1482* | **−0.0494*** |
| R5 max-fusion | 0.1577 | 15.14 | 200.3 | 125.0 | +0.1476* | +0.1464* | −0.0512* |
| R4 **D1** max_aux − EADP pct | 0.1357 | 13.03 | 157.9 | 99.5 | +0.1255* | +0.1244* | **−0.0732*** |
| R4 D1, 2 strongest aux only | 0.1488 | 14.29 | 154.5 | 95.0 | +0.1386* | +0.1375* | −0.0601* |
| R4 D2 EADP_rank − aux_rank | 0.1357 | 13.03 | 157.9 | 99.5 | +0.1255* | +0.1244* | −0.0732* |
| R5 min-fusion (agreement) | 0.0304 | 2.91 | 243.0 | 199.0 | +0.0202* | +0.0190* | −0.1786* |
| R4 D3 rank range | 0.0679 | 6.51 | 220.4 | 162.0 | +0.0577* | +0.0565* | −0.1411* |
| **R6 ORACLE-union (ceiling)** | **0.3012** | 28.91 | 26.9 | 19.0 | +0.2910* | +0.2899* | **+0.0923*** |
| R6b random draw from union pool | 0.0676 | — | — | — | — | — | — |

### k = 16 / 32 / 64 — the same table, condensed

| k | best single | best single recall | best disagreement | disagreement recall | Δ vs best single | verdict |
|---:|---|---:|---|---:|---|---|
| 8 | `cos_s0c` | 0.2089 | D4 | 0.1595 | **−0.0494*** | disagreement loses |
| 16 | `cos_s0c` | 0.1693 | D4 | 0.1524 | **−0.0170*** | disagreement loses |
| 32 | `cos_s0c` | 0.1659 | D4 | 0.1640 | −0.0019 [−0.0121, +0.0091] | **tie** |
| 64 | `cos_s0c` | 0.1949 | D4 | 0.2153 | **+0.0204*** | small win |

Per-benchmark (all 450, 150 per benchmark), k = 8:

| rule | TextVQA | DocVQA | OCRBench | macro |
|---|---:|---:|---:|---:|
| R1 EADP-next | 0.0058 | 0.0067 | 0.0133 | 0.0086 |
| R2 `cos_s0c` | 0.1833 | 0.2767 | 0.1608 | **0.2069** |
| R3 union, all 6 | 0.1600 | 0.2233 | 0.1283 | 0.1706 |
| R4 D1 | 0.1350 | 0.1817 | 0.1083 | 0.1417 |
| R4 D4 | 0.1492 | 0.2017 | 0.1117 | 0.1542 |
| R5 max-fusion | 0.1483 | 0.1975 | 0.1100 | 0.1519 |
| R6 ORACLE-union | 0.2967 | 0.3400 | 0.2425 | 0.2931 |

The ordering is **identical on all three benchmarks**: disagreement sits between
EADP-next and the best single proxy everywhere, and never above the best single.
The result is not a benchmark-profile artefact.

### 4.1 Paired bootstrap vs the best single proxy (held-out 210)

| k | rule | Δ | 95 % CI | wins | losses | ties |
|---:|---|---:|---|---:|---:|---:|
| 8 | D1 | −0.0732 | [−0.0887, −0.0577] | 20 | 117 | 73 |
| 8 | D4 | −0.0494 | [−0.0631, −0.0363] | 20 | 86 | 104 |
| 8 | max-fusion | −0.0512 | [−0.0649, −0.0381] | 19 | 87 | 104 |
| 8 | union-top2 | −0.0065 | [−0.0149, +0.0012] | 17 | 25 | 168 |
| 8 | ORACLE-union | **+0.0923** | [+0.0786, +0.1065] | 118 | 0 | 92 |
| 16 | D1 | −0.0321 | [−0.0432, −0.0217] | 38 | 108 | 64 |
| 16 | ORACLE-union | **+0.1470** | [+0.1330, +0.1625] | 186 | 0 | 24 |
| 32 | D4 | −0.0019 | [−0.0121, +0.0091] | 94 | 84 | 32 |
| 32 | ORACLE-union | **+0.2207** | [+0.2076, +0.2341] | 209 | 0 | 1 |
| 64 | D4 | **+0.0204** | [+0.0096, +0.0309] | 127 | 70 | 13 |
| 64 | ORACLE-union | **+0.3378** | [+0.3261, +0.3500] | 210 | 0 | 0 |

**Answer to the brief's key question** — *is disagreement rescue clearly better
than EADP-next, random, and any single proxy?* It is clearly better than
EADP-next and random (which are both at chance), and clearly **worse than the
best single proxy** at every k except 64, where it is marginally better.

---

## 5. Downstream Accuracy

**Not run. No macro number is claimed.**

The brief gates Phase 2 (composition) and Phase 3 (generation) on Phase 1
passing: *"如果 disagreement rule 在 critical recall 上没有稳定优势 … 则直接停止
这条路线。不要继续 downstream."* The rule failed on all three of the brief's own
stopping criteria (below the best single proxy; small advantage where it exists;
and the advantage that does exist is confined to k = 64). Phase 2/3/4 were
therefore not started, and the table below is given only as the reference frame
a passing result would have been measured against.

| Method | Tokens | TextVQA | DocVQA | OCRBench | Macro | Δ vs B2 | Δ vs EADP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full 1024 (B0) | 1024 | 73.60 | 85.27 | 68.00 | 75.62 | +15.74 | +14.52 |
| Official EADP facility (B1) | 256 | 59.40 | 61.89 | 62.00 | 61.10 | +1.22 | — |
| B2 EADP block8 | 256 | 65.40 | 60.24 | 54.00 | 59.88 | — | −1.22 |
| *Ours (disagreement rescue)* | — | — | — | — | **not run** | — | — |

**Why not run anyway, stated plainly.** The one thing a downstream run could
have added is a *measured* confirmation of the M3-v0 depth argument in §8. That
argument is not speculation: M3-v0 measured four students of statistically
identical head precision, at mean rescue rank 73.7–131.9, against an oracle at
3.5–15.5, and every one of them was matched by a content-free random control.
`cos_s0c`'s rescue sits at mean rank **131.0** — the *deep* end of that already
measured dead band. Running 150 generations to re-confirm a measured null is the
"chase +1 macro" the brief forbids.

---

## 6. Efficiency

No method was built, so **no TTFT was measured**. What is decision-relevant is
the cost of the *audit*, which a future round would pay on top of B2.

| quantity | value | source |
|---|---:|---|
| `nn4_recon` | 1.479 ms | measured, `m6_cost.json` |
| `cos_s0c` | 0.247 ms | measured, `m6_cost.json` |
| **total marginal feature cost** | **1.726 ms** | measured |
| other 25 columns | 0 ms | reductions over tensors the pruner already produces |
| B2 TTFT (reference) | 278.6 ms | M2 paired profile |
| B2 selector (reference) | ~9 ms | M2 |

Marginal overhead would be **+0.62 %** of B2's TTFT — small. **This is not the
reason the line failed**; the line failed on signal, and it is worth being
explicit that a cheap method that does not work is still a method that does not
work.

Forward-only: **yes** — every column is a function of the EADP score and its
components, the selector's own similarity matrix, the post-merger tower feature,
and S0. No gradient, no backward, no decoder layer, no teacher at inference.

---

## 7. Ablation

The rule family **is** the ablation; no knobs were tuned. The interpretable
comparisons, at k = 8:

| comparison | recall | what it isolates |
|---|---:|---|
| no disagreement — best single proxy (`cos_s0c`) | **0.2089** | the signal |
| single proxy, next best (`red_s0_top8`) | 0.1482 | one proxy is not enough |
| union, 2 strongest (round-robin) | 0.2024 | combination *without* a disagreement term |
| union, all 6 | 0.1750 | adding weak proxies dilutes |
| max-fusion over aux percentiles | 0.1577 | fusion *is* a disagreement form, and loses |
| min-fusion (agreement) | 0.0512 | consensus is the worst combiner |
| **D1** max_aux_pct − EADP_pct | 0.1357 | the brief's primary disagreement rule |
| **D4** binary top-10 % disagreement | 0.1595 | the best disagreement rule |
| D3 rank range | 0.0679 | dispersion alone is near-useless |
| **random from the union pool** | 0.0676 | the pool is enriched 6.6× over global random |
| **ORACLE-union (ceiling)** | **0.3012** | what a perfect combiner could extract |

**Eviction / redundancy ablation was not run** — the brief makes redundancy-aware
eviction part of Phase 2, which the gate closed. No claim about eviction is made.

**D2 is a degenerate rule and is reported as such.** As literally specified
("best rank among auxiliary proxies minus EADP rank") the raw-rank difference is
dominated by the EADP term, whose range is 0–1 while the min over five aux ranks
concentrates near 1/6; D2 therefore collapses onto the EADP term. Its numbers are
numerically identical to D1's at k = 8 and k = 16, which is how the degeneracy is
visible. It is kept in the tables rather than removed.

---

## 8. Failure / Success Analysis

### 8.1 Did disagreement catch catastrophic misses? No — and the reason is precise

At k = 8, over all 450:

| rule | mean head hits (of 8) | P(≥1 hit) | P(0 hits) | meanTR | overlap with `cos_s0c`'s picks |
|---|---:|---:|---:|---:|---:|
| `cos_s0c` (best single) | **1.656** | 0.849 | 0.151 | 132.9 | 1.000 |
| D4 | 1.233 | 0.773 | 0.227 | 193.1 | 0.296 |
| D1 | 1.133 | 0.711 | 0.289 | 155.5 | **0.298** |
| union-top2 | 1.609 | 0.827 | 0.173 | 162.9 | 0.630 |
| EADP-next | 0.069 | 0.062 | 0.938 | 408.8 | 0.007 |

D1 is a **real** disagreement rule: it picks a **70 %-different** token set from
the best single proxy. The swap is simply harmful — per instance, D1 beats
`cos_s0c` on **14.0 %** of instances and loses on **50.9 %** (35.1 % tie). So the
failure is not "disagreement changes nothing"; it is "**disagreement changes the
right things into the wrong things**".

### 8.2 The pool is enriched; the *ranking inside it* is what fails

This is the sharpest form of the result and the reason it is REFUTED rather than
"no signal exists":

| k | best single | random draw from union pool | ORACLE-union | oracle − best single |
|---:|---:|---:|---:|---:|
| 8 | 0.2069 | 0.0676 | 0.2931 | **+0.0861** |
| 16 | — | 0.0748 | 0.3190 | +0.1470 |
| 32 | — | 0.0978 | 0.3956 | +0.2207 |
| 64 | — | 0.1437 | 0.5400 | +0.3378 |

(all 450). The union of the six proxies' top-k **does** contain substantially
more head tokens than `cos_s0c` alone — 42 % more at k = 8, 84 % at k = 16. A
perfect combiner confined to that pool would capture them. But no cheap score
ranks inside the pool better than `cos_s0c` ranks the whole 768-token set, and
the disagreement scores rank it **worse**. The information is present in the
proxies *collectively* and is not extractable by any of the combiners tested.

### 8.3 Is one proxy doing all the work? Yes — `cos_s0c`

`cos_s0c` is the best single proxy at every k, on every benchmark, and the
oracle-union at k = 8/16/32 is only reachable by adding tokens that no score can
rank. Restricting D1 to the two strongest auxiliaries (removing the weak proxies
that might be diluting it) does **not** rescue it: 0.1488 vs 0.1357 at k = 8 —
still −0.060 below the best single. The failure is not dilution by weak proxies.

### 8.4 Is it redundancy or random replacement in disguise? No

- Random replacement at k = 8 gives 0.0102; `cos_s0c` gives 0.2089 — **20×**.
- `cos_s0c` survives the stratified controls (§3.2): 6.29× within `vis_norm`
  strata, 5.54× within `imp` strata. It is not a magnitude or an EADP artefact.
- The 20-seed random control's spread is [0.0064, 0.0139] at k = 8 — an order of
  magnitude smaller than every effect discussed here.

### 8.5 Which benchmark? DocVQA is easiest for every rule, and nothing inverts

`cos_s0c` reaches 0.2767 on DocVQA vs 0.1833 TextVQA / 0.1608 OCRBench; the
ordering of rules is identical across all three. There is no benchmark on which
disagreement wins.

### 8.6 The depth argument: why even the winner would not have translated

The brief warns against treating old-panel noise as success; the same standard
applies to the *single-proxy* result, which is the strongest thing found here.

| family | mean teacher rank of the rescue | measured downstream value |
|---|---:|---|
| M3-v0 oracle, r = 8 | **3.5** | **+7.73 macro** |
| M3-v0 oracle, r = 32 | 15.5 | +14.87 macro |
| M3-v0 learned student, r = 8 | 73.7 | ≈ 0 (matched by random control) |
| M3-v0 learned student, r = 32 | 131.9 | ≈ 0 |
| **M6 `cos_s0c`, k = 8** | **131.0** | **not run — predicted ≈ 0** |
| M6 EADP-next, k = 8 | 400.1 | not run — predicted ≈ 0 |

`cos_s0c` lands at the **deep end** of the band M3-v0 measured as worthless. It
finds more head tokens than the M3-v0 student, but it finds them *deeper* — and
S2-C2 measured that value is convex in rank (8 tokens close 53 % of the gap, 32
close 97 %, tokens past rank 32 worth ~nothing). Head recall and head *depth*
are different quantities, and only the second one converts.

*(Panel note: M3-v0's ranks are on the test-150 with a trained 240-image
student; M6's are on the held-out 210 training-free. The comparison is of
magnitudes, not of like-for-like arms.)*

### 8.7 Representative cases

- **Success (k = 8, `cos_s0c`)**: 84.9 % of instances get ≥ 1 teacher-head token,
  mean 1.66 of 8. It is a real, cheap, forward-only signal that recovers about a
  sixth of the teacher's head.
- **Failure (k = 8, D1)**: on 28.9 % of instances D1 finds *nothing* from the
  head, against 15.1 % for `cos_s0c`. The disagreement term actively displaces
  good picks: D1's 70 %-different set is worse on half the panel.
- **The structural case**: no rule in this stage, and no proxy in the bank,
  places the teacher's head tokens at a mean rank shallower than ~130. The M3-v0
  ceiling needs rank 3.5–15.5.

---

## 9. Statistical / Reproducibility Checks

| check | result |
|---|---|
| per-instance metrics | every recall is computed per instance then averaged; no pooled global top-k anywhere |
| chance baseline | analytic k/768, and an empirical 20-seed random control |
| random-control spread | k = 8 [0.0064, 0.0139]; k = 32 [0.0383, 0.0444]; k = 64 [0.0808, 0.0875] |
| orientation fitting | one sign per proxy, fitted on **fit (240)** only; never re-fitted on val/test |
| paired bootstrap | 2000 resamples, paired over instances, on the held-out 210 |
| determinism, analysis | re-ran the full grid: **2016 numeric comparisons, worst abs diff 0.000e+00** |
| determinism, EADP-next | `m6_eapd_order.py` re-run in a **separate process**: order rows bit-identical |
| identity gates | G-NEXT (first 256 greedy picks == live selection order) 450/450; G-S0 (sorted == bank `s0`) 450/450 |
| teacher integrity | bank `g2` vs `s2b_gradient_scores.npz`: worst \|Δ\| = 0 on all 450 |
| sample counts | held-out 210 (val 60 + test 150); all-450 and test-150 reported alongside |
| resolution | M3-v0 measured MDE80 = 7.0 macro on the old 150; **this stage never quotes a macro number**, so that limit does not bind |

---

## 10. Final Verdict

# REFUTED

> **Should we continue developing this into the final paper method? NO.**

The three reasons, in order of importance:

1. **Disagreement is worse than the best single proxy where it matters.** At
   k = 8/16 it loses by −0.049 / −0.017 with CIs excluding zero; at k = 32 it
   ties. The only wins (k = 64, +0.015 to +0.021) come at a quarter of the
   budget, where the rescue is no longer a rescue. A method whose combining rule
   underperforms simply not combining is not a method.

2. **The failure is not fixable by better disagreement definitions, because the
   ceiling is the ranking, not the pool.** An oracle combiner confined to the
   proxies' own union reaches 0.3012 vs 0.2069 for the best single at k = 8 —
   the extra head tokens are *there* — yet every tested combiner (D1, D2, D3, D4,
   max-fusion, min-fusion, round-robin union at three widths, and D1 restricted
   to the strongest auxiliaries) lands at or below the best single. The
   information is present collectively and is not extractable by any cheap
   score. Tightening the rule family further is the "调几十个参数" the brief
   forbids.

3. **Even the winner lands in a depth band already measured as worthless.** The
   best single proxy, `cos_s0c`, reaches mean teacher rank **131.0** — deeper
   than the M3-v0 student at 73.7 that four seeds and a content-free control
   already showed converts to nothing downstream. Pursuing it would repeat M3-v0
   with a cheaper scorer.

**What would change the verdict.** Not a better disagreement rule — §8.2 bounds
that. The binding constraint is the *depth* of what cheap forward-only signals
can reach: nothing in the 27-column bank places the teacher's head shallower
than ~rank 130, while the ceiling needs 3.5–15.5. If this line were ever
reopened, the two experiments worth running are:

1. **Measure the depth wall directly.** For each of the 27 columns, the mean
   teacher rank of its top-k — the quantity in §8.6 — and the best achievable by
   any *monotone* function of them. If no monotone combination reaches rank < 50,
   the whole pre-LLM family is closed and the effort belongs elsewhere.
2. **Test whether depth is a property of the teacher or of the features.** S2-C2
   found value convex in rank; if a *different* teacher definition (e.g.
   answer-conditioned rather than argmax-conditioned saliency) has a flatter
   head, the target itself changes and the depth wall may move.

Neither is started here.

---

## Appendix — files

| file | what |
|---|---|
| `scripts/discovery/m6_common.py` | proxies, rules, metrics, bootstrap |
| `scripts/discovery/m6_eapd_order.py` | the incumbent's exact greedy continuation (GPU, 128 s) |
| `scripts/discovery/m6_phase1.py` | the Phase-1 grid |
| `scripts/discovery/m6_analyze.py` | regenerates every table above |
| `scripts/discovery/m6_cost.py` | the marginal feature-cost measurement |
| `outputs/discovery/m6_eapd_order.npz` | (450, 1024) selector order + gates |
| `outputs/discovery/m6_phase1.json` | the full grid |
| `outputs/discovery/m6_tables.md` | every table above, generated (untracked, like all `outputs/discovery/*.md`) |
| `outputs/discovery/m6_cost.json` | measured costs |

Reused unchanged from earlier stages: `m5_bank.npz` (the 27-column feature bank,
450 × 1024 × 27), `s2b_gradient_scores.npz` (the P1-G2 teacher), `m2_gdep.py`
(engine and B2 selector), `instrumented.py` (selector swap), `m5_common.py`
(feature definitions and bank loader).
