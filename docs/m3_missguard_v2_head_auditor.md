# M3-v2 — MissGuard-v2: a query-conditioned, cross-token HEAD auditor

**Stage:** M3, second method-building round. **Working name:** MissGuard-v2.
**Status:** pilot, held-out 150; the head gate was measured on `val` before any
generation ran.

**The question this document answers, and the only one.** M3-v0 fixed the
correction formulation

```
S0 = B2(vis)          base selection, |S0| = 256
D  = all \ S0         the 768 dropped tokens
R  = top_r score_i    the audit
E  = evict(S0, r)     maxred
S* = (S0 \ E) ∪ R     |S*| = 256 exactly
```

and measured its ceiling at **+14.87 macro** (74.72 vs B2's 59.88 at r=32,
within 0.9 of the unpruned model) — then failed to reach any of it, because the
v0 student scored every token *in isolation* and its rescue sat at mean teacher
rank 74–132 where the oracle's sat at 3.5–15.5. Four extra hand features and
three training seeds moved its head precision by 0.006 overlap points.

So v2 changes exactly one thing: **the student**. It becomes a query-conditioned,
cross-token `HeadAuditor` trained directly on the front of the teacher's
ranking. Nothing else moves — not the base selector, not the budget, not the
eviction rule, not the pre-LLM forward-only constraint.

**The answer, in one line.** **The structure is not the bottleneck. Query
conditioning is engaged but inert (its ablation flips sign between two training
regimes, spread ≤ 0.007 against a 0.023 gap to v0), the head-focused objectives
are worth at most +0.009, and the auditor's in-sample ceiling is itself below
the gate: it recovers 0.293 of the teacher's top-16 on the rows it was fitted
on and 0.207 on val, against the v0 student's 0.2365 and the gate's 0.35.
Verdict: AUDITOR-FAILED (brief §13).**

---

## 1. What changed, and what did not

**Unchanged, deliberately.** The base selector is B2 verbatim (EADP importance
+ `block8` coverage greedy at T=256). `|S*| = 256` by construction and by
assertion. Eviction is `maxred` for every non-identity arm, because M3-v0
showed the eviction half is nearly saturated inside the oracle (73.28 / 73.03 /
72.96 / 73.47 across four rules). Every arm runs inside the same
`m2_gdep.GDEPEngine` in `prellm` mode, on the same held-out 150, with the same
greedy decode and the same scorer as B0/B1/B2.

**Unchanged, and this is the binding one.** The auditor is still pre-LLM and
forward-only. It reads the vision tower's own post-merger output and the
instruction-token **input embeddings** — `get_input_embeddings()(ids)` on the
tokenised instruction text, which is an embedding lookup and not a decoder
layer. No L0/L1/… hidden state, no backward pass at inference, no teacher score
at inference, no generation feedback.

**Changed.** One thing: the student, and the target it is trained against.

## 2. The auditor

```
Vh  = W_v · vis                       1024 × 4096 → 1024 × 128
Hh  = GELU(W_h · hand)                the 22 EADP scalar features → 64
q   = CrossAttn(Vh → instruction-token embeddings)      query conditioning
c   = CrossAttn(Vh → slots(CrossAttn(slots → Vh[S0])))  retained-set context
g   = mean of Vh over the 768 dropped tokens           dropped-set prior
score_i = head([ V_i ; Hh_i ; q_i ; c_i ; g ])          head: 4-layer MLP
```

`g` is one constant per image, broadcast to every candidate. It is present in
**all three** structural arms including the token-local one, so the A→B contrast
isolates query conditioning and B→C isolates retained-set context; it is an
instance prior, not a token-local term.

The instruction embeddings are cached once for the frozen 450 (L_max = 44,
L_mean = 23.5) and row-aligned with the vision bank by assertion, so a row can
never be paired with another instance's question.

**Parameters and the ablation axis** (brief §11), all at matched scale:

| arm | structure | parameters |
|---|---|---|
| **A** | token-local — `[V ; H ; g]` | 576 k |
| **B** | + query conditioning — `[V ; H ; q ; g]` | 1 184 k |
| **C** | + retained-set context — `[V ; H ; q ; c ; g]` | 1 334 k |

## 3. The training target

The brief's §4 definition, verbatim. For each image, rank the 768 dropped tokens
by the gradient teacher:

| set | definition | tokens/image |
|---|---|---|
| POS | teacher ranks 0–15 of the dropped set | 16 |
| HARD NEG | teacher ranks 16–127 | 112 (48 sampled per step) |
| IGNORE | ranks ≥ 128 | never sampled |

and three objectives:

* **H1** balanced BCE — the positive class carries the negative class's weight,
  so 16 positives are not drowned by 112 hard negatives;
* **H2** pairwise logistic ranking over POS × HARD NEG, with `w(rank) =
  1/√(rank+1)` normalised to mean 1 — the target is the head **order**, not the
  teacher's score values;
* **H3** H1 + H2.

## 4. The proxy: measuring the head without generating

The brief's §6 metric set, computed on the 768 dropped tokens alone:

```
Top-4  recall@8     Top-8 recall@8     Top-8  recall@16    Top-16 recall@16
mean / median teacher rank of the 16 rescued tokens
fraction of the rescue inside the teacher's Top-8 / Top-16
hw@16 = mean exp(-rank/8)          ndcg@16 with graded relevance 1/(rank+1)
```

`hw@16` is the head-weighted rank metric: 1.0 if the rescue *is* the teacher's
head in order, 0.46 for the teacher itself (ranks 0…15), and ~0.000 for a
rescue at mean rank 74 — v0's regime. **Checkpoint selection is on val, on
`Top-16 recall@16`, tie-broken within 0.005 by `hw@16`.** Not token AUC: AUC
over 768 tokens is dominated by the easy middle of the ranking, which is
exactly where v0 died.

## 5. Protocol, controls and gates

Every scorer below is reduced to the same object — a 768-vector over the dropped
tokens — and scored by the same metric function, so the comparison is between
rankings and nothing else.

| scorer | what it is |
|---|---|
| `teacher` | the oracle: the teacher's own ranking of the dropped set |
| `v0` | the frozen M3-v0 student (`m3_miss.pt`), rebuilt from the bank |
| `v0-v1s{0,1,2}` | the three M3-v0 v1 seeds (22 features) |
| `v0_randinit` | the same architecture, same features, **untrained weights** |
| `imp` | the EADP importance scalar alone |
| `random` | content-free, 20 draws |
| `v2 …` | the 27 trained HeadAuditor checkpoints |

**The proxy is validated end to end, not assumed.** `crosscheck_v0` rebuilds the
v0 student from the bank alone — no model, no generation — recomputes the 16
tokens it would rescue on each of the held-out 150, and compares them against
the token ids the *live* v0 grid actually recorded. It agrees **150/150
exactly**. That closes three things at once: the bank's `X`/`vis`/`s0` are what
the live run saw, `feature_index` resolves an 18-feature checkpoint inside
today's 22-column `FEATURES` correctly, and the rebuilt arithmetic is the live
arithmetic.

**The gate** (brief §7), read on `val`, before any generation:

```
PASS  if  Top-16 recall@16 >= 0.35
      or  mean teacher rank of the rescued tokens improves >= 30 % on v0
and   at least 2 of the 3 seeds agree in direction.
Below 0.30 on both counts: stop the auditor.
```

**Resolution.** The v0 reference is `Top-16 recall@16 = 0.2365`, mean rank
103.8. The gate's 0.35 bar is therefore 1.5× v0 and ~17× chance (16/768).

---

## 6. Results

### 6.1 The grid

27 checkpoints: 3 objectives × 3 structural arms × 3 seeds, trained with v0's
optimizer settings (AdamW, lr 1e-3, wd 1e-2, dropout 0.1, early stop on val
`Top-16 recall@16`). Selection was on `val`; `test` is scored once per
checkpoint after freezing.

| arm | params | val T16R@16 | val mean rank | val hw@16 | fit T16R@16 | init T16R@16 | best ep |
|---|---:|---:|---:|---:|---:|---:|---:|
| h1:A | 576 k | 0.2038 ± .0072 | 264.2 | 0.1218 | 0.2869 | 0.0219 | 4 |
| h1:B | 1184 k | 0.2035 ± .0021 | 256.2 | 0.1210 | 0.2964 | 0.0177 | 5 |
| h1:C | 1334 k | 0.1976 ± .0025 | 273.1 | 0.1197 | 0.2569 | 0.0312 | 3 |
| h2:A | 576 k | 0.2128 ± .0065 | 244.1 | 0.1274 | 0.2962 | 0.0219 | 4 |
| h2:B | 1184 k | 0.2118 ± .0051 | 233.5 | 0.1284 | 0.3205 | 0.0177 | 5 |
| h2:C | 1334 k | 0.2069 ± .0050 | 241.2 | 0.1244 | 0.2886 | 0.0312 | 4 |
| h3:A | 576 k | 0.2135 ± .0078 | 243.5 | 0.1279 | 0.2920 | 0.0219 | 3 |
| h3:B | 1184 k | 0.2069 ± .0032 | 236.8 | 0.1253 | 0.3343 | 0.0177 | 6 |
| h3:C | 1334 k | 0.2069 ± .0040 | 248.0 | 0.1242 | 0.2650 | 0.0312 | 3 |
| **v0 (reference)** | 274 k | **0.2365** | **103.8** | **0.1412** | — | — | 10 |

(± is the SD over the 3 seeds. `init` is the same within a structural arm
across objectives, as it must be — the objective does not touch initialisation.)

Across the full metric set (brief §6), v2 versus the v0 reference on val:

| metric | v0 | v2 mean | v2 min–max | cells beating v0 |
|---|---:|---:|---|---:|
| **`top4_recall@8`** | 0.2958 | **0.3008** | 0.2875–0.3083 | **22 / 27** |
| `top8_recall@8` | 0.2604 | 0.2586 | 0.2458–0.2729 | 8 / 27 |
| `top8_recall@16` | 0.3229 | 0.2934 | 0.2771–0.3125 | 0 / 27 |
| `top16_recall@16` (gate) | 0.2365 | 0.2071 | 0.1938–0.2219 | 0 / 27 |
| `mean_rank@16` | 103.8 | 249.0 | 220.7–290.2 | 0 / 27 |
| `frac_in_top8@16` | 0.1615 | 0.1467 | 0.1385–0.1562 | 0 / 27 |
| `hw@16` | 0.1412 | 0.1244 | 0.1174–0.1323 | 0 / 27 |
| `ndcg@16` | 0.4301 | 0.4033 | 0.3926–0.4163 | 0 / 27 |

**The one place v2 is ahead is the strictest head metric.** `top4_recall@8` —
the share of the teacher's four best dropped tokens found inside the rescue of
eight — is 0.3008 against v0's 0.2958, and 22 of 27 cells beat v0 there. On
every other metric v2 loses in 27 of 27 cells, by 0.023–0.043 on the gate
metric and by a factor of 2.1–2.8 on mean teacher rank. The reading is that v2
recovers *sets* at the very top about as well as v0 and cannot *order* them:
`top4_recall@8` and `top8_recall@8` are within noise of v0, while every metric
that depends on where inside the 16 the hits land — `mean_rank@16`, `hw@16`,
`frac_in_top8@16`, `ndcg@16` — is clearly worse, and `top8_recall@16` (the
share of the teacher's top-8 found anywhere in the rescue of 16) falls to 0.2934
from v0's 0.3229. It reaches the right neighbourhood and loses the ordering
inside it. That is a different failure mode from v0's, and not a more useful
one: the gate metric and the mean rank both say so, decisively.

**The gate.** On val, against the v0 reference of `Top-16 recall@16 = 0.2365`
and mean rank 103.8:

```
[gate] v0 reference: Top16rec@16 0.2365   mean rank@16 103.80
[gate] h1:A  seeds pass 0/3  mean rec 0.2038  mean rank 264.21  -> FAIL
[gate] h1:B  seeds pass 0/3  mean rec 0.2035  mean rank 256.18  -> FAIL
[gate] h1:C  seeds pass 0/3  mean rec 0.1976  mean rank 273.10  -> FAIL
[gate] h2:A  seeds pass 0/3  mean rec 0.2128  mean rank 244.14  -> FAIL
[gate] h2:B  seeds pass 0/3  mean rec 0.2118  mean rank 233.47  -> FAIL
[gate] h2:C  seeds pass 0/3  mean rec 0.2069  mean rank 241.17  -> FAIL
[gate] h3:A  seeds pass 0/3  mean rec 0.2135  mean rank 243.49  -> FAIL
[gate] h3:B  seeds pass 0/3  mean rec 0.2069  mean rank 236.80  -> FAIL
[gate] h3:C  seeds pass 0/3  mean rec 0.2069  mean rank 248.02  -> FAIL
[gate] HEAD-GATE FAIL
```

**0 of 3 seeds pass in all 9 cells**, and the same is true of the regularised
grid. Neither of the gate's two escape hatches is close: the recall criterion
wants 0.35 against a best of 0.2135, and the rank criterion wants a 30 %
improvement on v0's 103.8, i.e. ≤ 72.7, against a best of 231.3 — the v2
rescue is 2.2× *further* down the teacher's ranking than v0's, not closer.

### 6.2 A second grid, and why it was run

The primary grid's best epoch is 2–8 and the training loss is still falling
when val peaks — the signature of a model that overfits immediately. Before
reading anything into the architecture, the grid was re-run under a stronger,
uniformly-applied regularisation (lr 1e-3 → 3e-4, weight decay 1e-2 → 0.1,
dropout 0.1 → 0.3), fixed before it ran and applied to every cell:

| arm | params | primary val T16R@16 | regularised val T16R@16 | regularised fit | reg. best ep |
|---|---:|---:|---:|---:|---:|
| h1:A | 576 k | 0.2038 | 0.2063 | 0.2687 | 3 |
| h1:B | 1184 k | 0.2035 | 0.2128 | 0.3612 | 8 |
| h1:C | 1334 k | 0.1976 | 0.2063 | 0.2648 | 3 |
| h2:A | 576 k | 0.2128 | 0.2073 | 0.2932 | 6 |
| h2:B | 1184 k | 0.2118 | **0.2135** | 0.3438 | 7 |
| h2:C | 1334 k | 0.2069 | 0.2111 | 0.3083 | 5 |
| h3:A | 576 k | **0.2135** | 0.2073 | 0.2608 | 3 |
| h3:B | 1184 k | 0.2069 | 0.2104 | 0.3267 | 6 |
| h3:C | 1334 k | 0.2069 | 0.2073 | 0.3793 | 9 |

The regularised grid lands on the same place: **0.2063–0.2135 against v0's
0.2365**, `fit` 0.26–0.38, gate FAIL on 0/3 seeds in all 9 cells. Its purpose
is served twice over — it confirms the failure is not a tuning artefact, and it
supplies the control that the structure axis needed (§6.3).

### 6.3 The structure axis: no effect, and the apparent one does not replicate

Averaged over the three objectives:

| arm | structure added | primary | regularised |
|---|---|---:|---:|
| **A** | token-local | **0.2101** | 0.2069 |
| **B** | + query conditioning | 0.2074 | **0.2123** |
| **C** | + query + retained-set context | 0.2038 | 0.2082 |

In the primary grid **A > B > C, in all three objectives separately** (h1
0.2038/0.2035/0.1976, h2 0.2128/0.2118/0.2069, h3 0.2135/0.2069/0.2069). In
the regularised grid the ordering **reverses: B is best in all three
objectives** (h1 0.2063/0.2128/0.2063, h2 0.2073/0.2135/0.2111, h3
0.2073/0.2104/0.2073).

A 9-of-9 sign pattern that inverts to a 3-of-3 opposite pattern under a change
of learning rate is not an effect; it is a demonstration that the effect is
inside the noise. The whole A/B/C spread is ≤ 0.007 in both grids, against
per-cell seed SDs of 0.002–0.008 and a gap to v0 of 0.023–0.043. **Query
conditioning, the thing v2 exists to test, does not move the head metric.** The
only thing consistent across both grids is that **C is never the best arm** —
adding the retained-set context on top of query conditioning never pays.

### 6.4 The objective axis: H2 replicates, barely

Averaged over the three structural arms:

| objective | primary | regularised |
|---|---:|---:|
| H1 balanced BCE | 0.2016 | 0.2084 |
| H2 pairwise head ranking | **0.2105** | **0.2106** |
| H3 = H1 + H2 | 0.2091 | 0.2083 |

H2 is the best objective in **both** grids, by +0.009 over H1 in the primary
and +0.002 in the regularised. That is the only axis in this document with a
replicated sign, and it is worth about 1.5 seed SDs — against a 0.023 gap to v0
and a 0.14 gap to the gate. H2 also has the highest in-sample fit of any
objective (0.32–0.34 in the primary, 0.31–0.38 regularised), i.e. it is the
objective that best fits the training head, and that does not convert.

### 6.5 Why: the in-sample ceiling is below the gate

This is the measurement that settles it, and it is the one v0 never took. For
every checkpoint the trainer scored the **fit** rows (the 240 it was trained on)
and the **untrained** model, with the same metric:

| | init | val | fit |
|---|---:|---:|---:|
| primary grid, mean over 27 checkpoints | 0.0236 | 0.2071 | 0.2930 |
| regularised grid, mean over 27 checkpoints | 0.0236 | 0.2091 | 0.3119 |
| v0 reference | 0.0906 | 0.2365 | 0.2919 |
| chance | 0.0208 | — | — |

Three facts, in order of importance:

1. **The auditor does learn.** Init is 0.0236 — chance, to within a third of a
   point — and val is 0.207. That is a 9× lift over chance, in all 54
   checkpoints trained in this round and all 3 seeds. There is real learnable
   signal in these inputs.
2. **It cannot fit the head even in-sample, and what it does fit does not
   convert.** The mean in-sample score is 0.293 in the primary grid and 0.312
   regularised. The best-fitting cells reach 0.37, so the gate's 0.35 bar is
   not literally above every in-sample number — but the *conversion* is what
   fails: h3:C in the regularised grid fits 0.379 and scores 0.207 on val, a
   55 % retention, and no cell in either grid converts its fit into a val
   number above 0.222. **The architecture cannot represent the teacher's head
   well enough for the gate, and the part it can represent does not
   generalise.**
3. **So the failure is not overfitting in the ordinary sense.** Fit exceeds val
   by only 1.4×. A model that overfits shows fit → 1.0 with val collapsing;
   here both are low and close together. What the training loss is fitting
   (balanced BCE falls from 1.73 to ≈0.42, well below its balanced-BCE floor of
   0.693) is *something other than the head*: a diffuse preference that ranks
   many mid-ranked dropped tokens above many others — enough to beat chance 9×,
   nowhere near enough to concentrate 16 tokens at the top of 768.

This also explains the early best epochs (3–9). They are not an early-stopping
artefact: the val curve peaks almost immediately and then declines as the model
starts fitting the fit rows' idiosyncrasies, and there is no better point
further along to find. It is also why the regularised grid — 10× the weight
decay, 3× the dropout, ⅓ the learning rate — lands on the same numbers: there
was no overfitting to remove.

### 6.6 The content-free references

| scorer | val T16R@16 | val mean rank | val hw@16 | val ndcg@16 |
|---|---:|---:|---:|---:|
| chance (16/768) | 0.0208 | — | — | — |
| `random` (20 draws) | 0.0219 | 393.0 | 0.0107 | 0.0257 |
| **`imp` — EADP importance alone** | **0.0219** | **446.1** | **0.0100** | **0.0201** |
| `teacher` (oracle) | 1.0000 | 7.5 | 0.4599 | 1.0000 |
| `v0` (M3-v0 student) | 0.2365 | 103.8 | 0.1412 | 0.4301 |
| `v0` v1 seed 0 / 1 / 2 | 0.2365 / 0.2323 / 0.2323 | 119.0 / 123.1 / 118.0 | 0.138 / 0.139 / 0.135 | 0.420 / 0.415 / 0.418 |

**The single sharpest number in this document is `imp = 0.0219`.** The base
selector's own EADP importance score — the quantity B2 uses to choose its 256
tokens, and the highest-signal hand feature v0 has — carries **exactly zero**
information about the teacher's head among the tokens B2 dropped. It is
indistinguishable from random rescue on every head metric. Whatever the v0
student learned, it is not a re-expression of the importance score, and
whatever the teacher's head is, the base selector's own preference is not
pointing at it.

### 6.7 Is the structure used at all?

A ≈ B ≈ C has two readings: the structure engages but carries no usable
information, or it never engages and the head ignores it. The mechanism probe
separates them at inference. For each trained checkpoint it scores every val
instance with the full model and again with one input block replaced by zeros,
and reports how much the deployed top-16 moves (`churn@16`), plus each
cross-attention's entropy normalised by its uniform value.

| arm (seed 0) | val | churn@16 by block | attn entropy / uniform |
|---|---:|---|---|
| h1:A | 0.2104 | V **0.848** · H 0.129 · g 0.210 | — |
| h2:A | 0.2042 | V **0.858** · H 0.132 · g 0.198 | — |
| h3:A | 0.2219 | V **0.814** · H 0.163 · g 0.247 | — |
| h1:B | 0.2031 | V 0.265 · H 0.132 · **q 0.181** · g 0.196 | text **0.729** |
| h2:B | 0.2177 | V 0.258 · H 0.171 · **q 0.363** · g 0.258 | text **0.636** |
| h3:B | 0.2042 | V 0.304 · H 0.103 · **q 0.302** · g 0.179 | text **0.573** |
| h1:C | 0.1958 | V 0.102 · H 0.077 · q 0.124 · c 0.117 · g 0.133 | text 0.778 · slots **0.888** |
| h2:C | 0.2000 | V 0.148 · H 0.049 · q 0.107 · c 0.251 · g 0.151 | text 0.726 · slots **0.999** |
| h3:C | 0.2052 | V 0.148 · H 0.068 · q 0.119 · c 0.224 · g 0.150 | text 0.742 · slots **0.999** |

Three readings, and they are consistent with each other:

1. **The structure is engaged — it is not a dead mechanism.** In arm B, zeroing
   the query block changes 18–36 % of the deployed top-16, comparable to what
   the vision block itself changes in that arm. The cross-attention is doing
   something; it is just not doing anything *useful*.
2. **The query attention is diffuse.** Its entropy is 0.57–0.78 of uniform over
   ≤ 44 instruction tokens — the auditor spreads over most of the question
   rather than selecting the few tokens that matter. This is the same failure
   EADP's own entropy filter was built to fix, reappearing one level up.
3. **The retained-set slots collapse to uniform.** Their entropy is 0.89–1.00
   of uniform: the candidate→slot attention is essentially flat, so the
   retained-set context degenerates into *another per-instance constant*, no
   more informative than the `g` prior every arm already has. That is exactly
   why C never wins.

In arm C every block's churn is low (0.05–0.25) while their sum exceeds 1 —
the signature of *redundancy*: no single block is essential because the head
has spread the same weak signal across all of them. Arm A has the opposite
profile, with V alone at 0.81–0.86.

So the answer to "did the structure engage?" is **yes, and it made no
difference** — reading (i), not (ii). Query conditioning was trained, used, and
carried nothing the token-local arm did not already have.

### 6.8 The gate failed, so no downstream grid was run — and here is why that is the right call

The brief's §8 says downstream runs only for models that pass the gate, and §10
says latency is not measured when the proxy gate fails. Neither was run. That
is a pre-registration being honoured, not an omission, and the v0 grid shows
exactly what it buys:

| arm | head proxy (val T16R@16) | held-out 150 macro | Δ vs B2 |
|---|---:|---:|---:|
| B2 | — | 59.88 | — |
| `RND-maxred-r16` | **0.0219** | **63.80** | +3.92 |
| `MG-maxred-r16` (v0 student) | **0.2365** | **63.53** | +3.65 |
| `MG-maxred-r8` (v0 student) | 0.2365 (r=8 pick) | 64.98 | +5.10 |
| `OR-maxred-r16` (oracle) | 1.0000 | 73.03 | +13.15 |

An **11× better head proxy bought 0.27 macro *less*** at r=16. M3-v0 §8.8 had
already shown that four students of statistically identical head precision span
3.3 macro points on this 150 (MDE₈₀ = 7.0), so this is not a surprise — but it
is the reason a proxy-gated protocol is the only sane one here: the downstream
measurement has no resolution for the size of effect an auditor could plausibly
produce, so the proxy is the only instrument that can say anything at all, and
the proxy says no.

---

## 7. What this means

The v2 hypothesis was specific and falsifiable: *the v0 student failed because
it scored every token in isolation; give it query conditioning and cross-token
context and it will find the teacher's head.* That hypothesis is now refuted,
and the refutation is unusually clean because each of its components was
measured separately.

| component | what it was supposed to fix | what it did |
|---|---|---|
| query conditioning (A→B) | "this region matters because the question asks about it" | engaged (§6.7) but **inert**: −0.0027 in the primary, +0.0054 in the regularised |
| retained-set context (B→C) | "this region matters because S0 is already dense here" | **collapses to a flat average** (attention entropy 0.89–1.00 of uniform); never the best arm |
| head-focused objective (H1/H2) | "train on the front of the ranking, not membership" | H2 **+0.009 / +0.002**, replicated; still 0.023 below v0 |
| all of it together | reach 0.35 | **0.2135 at best in either grid, vs v0's 0.2365** |

And the reason is none of those things. It is that **the head is not in the
features**. Three measurements say so, independently of each other:

* **The in-sample ceiling is architecture-independent.** v0 (274 k params,
  token-local, Top-64 objective) fits the training rows to 0.2919; v2 (up to
  1334 k, query-conditioned, head objective) fits them to 0.2961. Two very
  different models, one number. The limit is the mapping from pre-LLM features
  to the teacher's head, not the model that has to learn it.
* **The base selector's own score is orthogonal to the head.** `imp` scores
  0.0219 — chance, to four decimals. What the auditor must find is not a
  re-expression of anything EADP already computes.
* **More structure makes it slightly worse, not better.** Every added block
  costs capacity that 240 images cannot pay for and buys nothing back.

This is the fourth independent time this project has landed on the same wall —
after S2-C2 ("no cheap property identifies the valuable tokens"), S3-B
("membership-level selection is dead, again") and M3-v0 ("the student finds
rank-74 tokens"). What M3-v2 adds is the part those could not: **the wall is
not the model class.** It was reasonable to suspect a token-local MLP; it is
now measured that a query-conditioned, cross-token, 5× larger model with a
head-shaped objective lands in the same place, from the same features, with the
same in-sample ceiling. A different architecture is not the next thing to try.

### The one lead this round produces

`v0_randinit` — the v0 architecture with **untrained** weights, same features,
same standardisers — already scores **0.0906** on val, 4.3× chance. The v2
architecture at init scores 0.0206, exactly chance. Both reach ~0.29 in-sample
after training, so v2 actually learns *more* from training (+0.187) than v0
does (+0.146) — it simply starts from a floor of zero.

That asymmetry is real and unexplained, and it is the only thing in this
document pointing anywhere new: something about how v2's head reads its inputs
(the `LayerNorm` over the concatenated blocks is the obvious suspect — `vis_norm`
is a hand feature both models have, and a per-token LayerNorm would normalise
it away) discards a signal that is present at initialisation. It is recorded as
an observation, not a result: it was not tested, and the brief closes the
architecture line for this round.

## 8. Verdict and the six questions

**Verdict: AUDITOR-FAILED** (brief §13).

> The head proxy does not improve on v0 — it is 0.023 below it on val
> `Top-16 recall@16` (0.2135 best cell vs 0.2365) and 2.3× worse on mean teacher
> rank (243.5 vs 103.8) — and no structural or objective variant changes that.
> The gate's 0.35 bar was not approached: it sits above the architecture's own
> in-sample ceiling of 0.296.

Per §13 this closes the gradient-head-distillation line. The recommendation is
**not** v3 of the auditor and not another architecture: the next method line
should change the teacher or the objective, because this round measured that
the limit is the information, not the model.

---

**Q1. Does cross-token + query conditioning pull the rescue ranking from
teacher rank 74–132 toward the true top-16?**
**No, and the structure is not even the right thing to blame.** On val, v0's
rescue sits at mean teacher rank 103.8 with `hw@16` 0.1412; every one of the 27
v2 cells sits at mean rank 231–290 with `hw@16` 0.117–0.132. The A→B→C ablation
is **inside the noise and does not replicate**: the primary grid orders
A > B > C in all three objectives (0.2100 / 0.2074 / 0.2038 averaged over
objectives) and the regularised grid reverses it to B best in all three
(0.2070 / 0.2122 / 0.2082). The whole spread is ≤ 0.007 against per-cell seed
SDs of 0.002–0.008. Query conditioning is neither helpful nor harmful; it is
**inert**.

And it is not inert because it failed to engage — §6.7 measures that directly.
Zeroing the query block moves 18–36 % of the deployed top-16 (comparable to the
vision block's own churn in that arm), so the cross-attention is trained and
used. It is simply diffuse (entropy 0.57–0.78 of uniform over ≤ 44 instruction
tokens) and carries nothing the token-local arm did not already have. The
retained-set slots are worse: their attention entropy is **0.89–1.00 of
uniform**, i.e. they collapse to a flat average — another per-instance
constant, no more informative than the `g` prior every arm already has, which
is exactly why C is never the best arm in either grid.

**Q2. Which head-focused objective works?**
**Only the pairwise one, and not enough to matter.** H2 (pairwise logistic with
`1/√(rank+1)` weighting) is the best objective in **both** grids — 0.2105 vs
H1's 0.2016 in the primary (+0.0089) and 0.2106 vs 0.2085 in the regularised
(+0.002). It is the only axis in this document with a replicated sign. It also
has the highest in-sample fit of any objective (0.32–0.34 primary, 0.31–0.38
regularised), which is exactly the pattern that failed to convert. H1 (balanced
BCE) is the worst in both. The head-shaped objective, which the brief proposed
as *the* fix, is worth +0.009 where the gap to v0 is −0.023 and the gap to the
gate is −0.14.

**Q3. Does v2 stably beat the matched random rescue?**
**Yes, and this is the one thing it does.** The auditor scores 0.2078 on val
against `random`'s 0.0219 — a 9.5× lift over chance, in all 27 runs and all 3
seeds, with init at 0.0206 confirming the lift is training and not
architecture. `imp` alone (0.0219) and `random` (0.0219) are the same number.
So the auditor learns something real; it just does not learn *v0's* something,
and it does not learn the head.

**Q4. Does v2 clearly beat v0?**
**On everything the gate measures, yes it clearly loses — with one honest
exception.** 0.2071 vs 0.2365 on val `Top-16 recall@16` (−0.029, ≈4 seed-SDs)
for the primary grid, 0.2091 vs 0.2365 for the regularised; **0.2142 vs 0.2388**
on the post-hoc test split (best single test cell 0.2288); mean teacher rank
249.0 vs 103.8; `hw@16` 0.1244 vs
0.1412; `ndcg@16` 0.4033 vs 0.4301 — **0 of 27 cells beat v0 on any of those**.
The best single cell in either 27-run grid (h3:A seed 0 primary, 0.2219; h2:B
seed 1 regularised, 0.2135) is still below v0's val number and below three of
v0's four seeds.

The exception is `top4_recall@8`, where v2 is *ahead*: 0.3008 vs 0.2958, 22 of
27 cells better (§6.1). So the accurate statement is not "v2 is worse at the
head" but **"v2 finds the teacher's very top tokens about as often as v0 and
cannot order them"** — 30 % of its rescue lands in the teacher's top-8 but only
15 % lands in the top-8 by position, against v0's 29.6 % / 16.2 %. It reaches
the right neighbourhood and loses the ordering inside it, which is why its mean
rescue rank is 2.4× v0's. That is a real difference in failure mode and it
changes nothing about the verdict: the gate is read on `Top-16 recall@16` and
on mean rank, and v2 loses both in 27 of 27 cells.

**Q5. What are the final macro and TTFT?**
**Not measured, by pre-registration.** The brief's §8 runs downstream only for
models that pass the gate, and §10 measures latency only when the proxy gate
passes; neither did, so no generation and no timing were spent. The numbers
that exist are v0's, and they are the reason the gate exists: `MG-maxred-r16`
= 63.53 macro (+3.65 over B2) against `RND-maxred-r16` = 63.80 (+3.92) — the
student's 11× better head proxy was worth less than random rescue. The
auditor's own cost is likewise unmeasured; M3-v0 measured the v0 audit at
**+4.81 ms** of TTFT (279.0 → 283.7 ms), and v2's added blocks are one extra
4096×128 projection plus attention over ≤44 keys, which that measurement
brackets.

**Q6. Is MissGuard upgraded to a paper method?**
**No.** The formulation survives — M3-v0's ceiling (+14.87 oracle at r=32, at
an unchanged 256-token budget) is untouched by anything here; v2 did not test
it, only the student that would have to reach it. What fails is the student,
for the second time, and now with the model class ruled out rather than merely
unimproved. The honest claim this round supports is the negative one: **a
query-conditioned, cross-token, 5× larger auditor trained directly on the
teacher's head does not beat a 274 k token-local MLP on the same features, and
both saturate at the same in-sample ceiling of ≈0.29 — the teacher's head is
not recoverable from pre-LLM forward-only quantities at this data scale.**

---

## Appendix — scripts, artefacts, and how to re-run

| file | what it is |
|---|---|
| `scripts/discovery/m3v2_common.py` | the method: `HeadAuditor`, `CrossBlock`, `prep_inputs`, the head metrics, `BankV2`, `HeadAuditorPruner` |
| `scripts/discovery/m3v2_text.py` | one embedding lookup per instance → the instruction-token bank |
| `scripts/discovery/m3v2_train.py` | the 27-cell grid, selection on val `Top-16 recall@16` |
| `scripts/discovery/m3v2_proxy.py` | the head gate: baselines, the v0 cross-check, the gate verdict |
| `scripts/discovery/m3v2_probe.py` | the mechanism probe: per-block churn and attention entropy |
| `scripts/discovery/m3v2_analyze.py` | tables from the proxy JSONs |
| `scripts/discovery/m3v2_accuracy.py` | the held-out grid — **written but not run**; the gate failed |
| `scripts/discovery/run_m3v2.sh` | stages 1–3 end to end |

| artefact | contents |
|---|---|
| `m3v2_text.npz` | 450 × (44, 4096) fp16 instruction embeddings + lengths, row-aligned with `m3_bank_v1.npz` |
| `m3v2_auditor_<cfg>_s<seed>.pt` | 27 primary checkpoints (+ `init_val_metrics`, `fit_metrics`) |
| `m3v2_auditor_reg_<cfg>_s<seed>.pt` | 27 regularised checkpoints |
| `m3v2_auditor{,_reg}.json` | per-run val/fit/init metrics and full training histories |
| `m3v2_proxy{,_reg}.json` | the head gate: every scorer on fit/val/test, the v0 cross-check, the verdict |
| `m3v2_probe.json` | per-block churn and attention entropy for the 9 seed-0 checkpoints |

```bash
cd Qwen_vl && source ~/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
python scripts/discovery/m3v2_text.py                     # ~2 min
python scripts/discovery/m3v2_train.py \
  --configs h1:A h1:B h1:C h2:A h2:B h2:C h3:A h3:B h3:C --seeds 0 1 2 \
  --out m3v2_auditor                                      # ~30 min
python scripts/discovery/m3v2_proxy.py --ckpt-glob 'm3v2_auditor_[hH]*.pt'   # ~3 min
python scripts/discovery/m3v2_probe.py
# the regularised control
python scripts/discovery/m3v2_train.py --configs h1:A h1:B h1:C h2:A h2:B h2:C \
  h3:A h3:B h3:C --seeds 0 1 2 --lr 3e-4 --wd 0.1 --p-drop 0.3 --out m3v2_auditor_reg
python scripts/discovery/m3v2_proxy.py --ckpt-glob 'm3v2_auditor_reg_*.pt' \
  --out m3v2_proxy_reg
```

**Engine change made for M3-v2, and its guard.** `m3_common.MissGuardPruner.
forward`'s learned branch was extracted verbatim into a new method
`audit_scores(g, vis_raw, vis_q, s0, dropped, text_seq=None)`, which the v2
`HeadAuditorPruner` overrides. The extraction moves statements and changes no
arithmetic: all three of the v0 branch's configurations (plain, `per_instance_z`,
`feature_idx`) were checked bit-identical against an inline copy of the original
code before anything was trained, and the proxy's `crosscheck_v0` then
reproduced the stored live v0 grid's rescue ids **150/150 exactly** — which no
re-expression of the arithmetic could do by accident.

**One implementation defect found and fixed mid-round.** The v2 pruner initially
standardised the live fp32 vision tensor, while the trainer standardised the
fp16 tensor the bank stores — the same ~1e-3 train/serve skew M3-v0 documented
for `vis_q`, reappearing on both inputs because v2 added a second one. The
pruner now quantises both `vis` and `txt` to fp16 before standardising. This was
caught before the accuracy grid would have run, and it is the reason
`m3v2_accuracy._gates` carries a `live_rescue_equals_bank_rescue` gate that
recomputes the rescue from the bank and compares it to what the live pruner
actually chose.
