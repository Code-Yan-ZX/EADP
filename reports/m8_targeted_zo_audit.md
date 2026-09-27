# M8 — Targeted Zeroth-Order Audit

**Stage:** M8, a new formulation and the first stage in this project that
*measures the model* rather than reading a feature off it. MissGuard (M3),
residual capsules (M4), SafeTrim (M5), conditional utility (S3-A), bundle
prediction (S3-B), disagreement rescue (M6) and candidate-union hedging (M7) are
closed and this document reopens none of them. **Status:** Phase 0 and Phase 1
run; **Phases 2–5 not run**, because the pre-registered Phase-1 depth gate
failed. Everything here is forward-only and training-free.

> **AMENDMENT (2026-09-27, found during M9).** The cheap-rule cells of the
> Phase-1 grid (`cos_s0c`, `imp`, `nn4_recon`, `red_s0_top8`, D4, max-fusion)
> were computed **misaligned**: `m8_phase1.picks_for` passed the positional
> panel index `j` to `r_cheap_in_pool` / `m6_score_in_pool`, which index
> `bank["X"]` with it, while the pool and the teacher rank belonged to global
> bank row `hold[j]` — a different instance. Oracle, random and ZO-P cells were
> always aligned, and the pool audit was unaffected. The bug was fixed in
> `m8_phase1.py`, the grid regenerated (`m8_phase1.json`,
> `reports/m8_tables.md`) and the affected figures corrected in place below.
> The headline correction: in pool P1 at `r = 8`, `cos_s0c` is
> **131.0** (recall 0.2089), not 118.9 (0.2333); D4/max-fusion are
> **127.8/127.5**, not 133.4/133.7 — so in-pool the M6 composition rules are
> *marginally better* than the best single proxy, reversing that ordering in
> this table. Every gate clause still fails, by wider margins than reported:
> the best ZO-P cell (184.2) is ~57 ranks worse than the best cheap rule, and
> the "worse on 13 of 15 pool×r cells" count is unchanged (the two exceptions
> are exact ties from pool exhaustion at r = 32). **The verdict is unaffected.**

**The question this document answers, and the only one.**

> Cheap forward-only proxies leave the tokens that actually carry the answer at
> a mean teacher rank of ~131, and M6 showed that no cheap score can rank inside
> an enriched candidate pool. A zeroth-order perturbation estimator is not a
> cheap score — it *measures* the model. Can it break that depth wall?

**The answer, in one line.** **No, and not for want of trying: the estimator
that could in principle break it does not exist at the deployed precision, and
the estimator that does exist — ZOO-Prune's own, reproduced bit-exactly — is a
rank restatement of a feature the incumbent already computes for free, is
*anti*-correlated with the gradient teacher, and ranks *worse* inside every
candidate pool than the cheap proxy that built the pool. Verdict: REFUTED.**

---

## 1. Executive Summary

- **The hypothesis.** M8's bet is that a strong sensitivity estimator is too
  expensive to apply to every token but cheap enough to apply to a *nominated*
  pool: `cheap nomination → targeted zeroth-order audit → small rescue →
  fixed-budget correction`. That is a strictly different bet from M6's ("can a
  cheap score rank inside the pool?") and from M7's ("can the pool be kept
  whole?"). It does not ask for a new cheap score; it asks for a *measurement*.
- **The pre-registered gate, and its outcome.** The brief fixes the gate before
  any number is read: at `r = 8`, mean teacher rank **< 40**, ideally < 25,
  clearly better than `cos_s0c` and than the best M6 cheap scorer. The best
  zeroth-order cell anywhere in the grid is **mean teacher rank 184.2** (pool
  P1, ZO-P m=2) against `cos_s0c`'s **131.0** in the same pool and an oracle's
  **31.3**. The gate fails by a factor of **4.6** on its own primary metric, and
  the estimator loses to the cheap proxy it was meant to beat.
- **Why — the mechanism, and it is not subtle.** ZOO-Prune's estimator, run at
  the projector as published, agrees with the token's post-merger feature norm
  at **Spearman 0.986** (positive on **100 %** of instances; **0.9876** at the
  published `m = 64`, which changes nothing). It is, to 99 % rank agreement,
  `vis_norm` — a column the incumbent's pruner already produces for free (M6:
  marginal cost 0). And it is **anti**-correlated with the gradient teacher:
  ρ = **−0.236** inside the dropped set, positive on **0.5 %** of instances. A
  score that is a free column and points the wrong way cannot rescue anything.
  An fp32 reverse-mode control rules out the obvious objection — that the
  published step `h = 0.01` is lost to bf16 rounding: the **exact** Jacobian
  norm, computed with no finite difference at all, tracks the token's output
  norm at ρ = **0.973**. `S ≈ ‖v‖` is what this merger does.
- **The estimator the hypothesis actually needs does not exist.** The
  zeroth-order estimator that *could* break the depth wall is the same idea one
  level up — perturb a token's visual embedding, run the LLM, read the change in
  the P1-G2 objective. It is the central-difference approximation of the very
  gradient the teacher computes. It is also **below the deployed model's
  numerical resolution**, measured three ways on 16 held-out instances: the
  objective's bf16 quantum is **0.25** while the average visual token's share of
  the whole-block effect is **0.031 quanta**; the per-token score takes **2.9
  distinct values** across a 32-token pool (36 % of tokens return exactly zero);
  and its **split-half reliability across independent direction draws is ≈ 0**
  (best of nine h × response cells: 0.096). Run through the gate on 40
  instances it recovers **1.21× chance** inside the pool, at mean teacher rank
  200 against `cos_s0c`'s 129, with a teacher agreement of **+0.014**. It does
  not reproduce, so no ranking from it can mean anything — and none does.
- **And it would not have been affordable anyway.** One perturbed forward at the
  1024-token context costs **205 ms** under this harness — **74 %** of B2's whole
  278.6 ms TTFT — and per-token sensitivity needs one forward *per token*. A
  32-token pool with central differences is **13.1 s**, i.e. **47× B2's TTFT**.
  The shared-direction (Hadamard) design that would cut this to `K` forwards does
  not help either: its aggregate response sits at **0.44 quanta**, below the
  quantum on **58 %** of draws, and does not improve at 768 tokens. Two
  independent reasons, either one fatal, and no third route around them.
- **Nomination works; auditing does not.** This is the one genuinely new
  positive finding, and it sharpens the negative. The cheap pools are strongly
  enriched (P1: `cov@8` = 0.2994 against chance 0.0312, and its in-pool oracle
  at `r = 8` sits at mean teacher rank **31.3** against `cos_s0c`'s 131.0 from
  the same pool). The pool contains the answer. The
  audit cannot find it. M6's "no cheap score ranks inside the enriched pool" is
  now joined by "and neither does the published zeroth-order estimator".
- **The global ZOO-Prune arm is at chance, and B2 is below it.** Applied to all
  1024 tokens with the incumbent's budget, ZO-P keeps **0.328** of the teacher's
  top-8 head, against a chance rate of **0.250** (256 of 1024) and B2's own S0
  at **0.228**. So the estimator is 1.31× chance and the incumbent is 0.91×
  chance — a real gap, but neither is a mechanism, and it says nothing about the
  depth wall, which is measured *inside* the dropped set.
- **No Phase 2, 3, 4 or 5.** The brief gates all of them on Phase 1 passing; it
  did not. No generation was run, no macro is claimed, and no TTFT is reported
  for a method that was never built. What *is* priced is the estimator's cost,
  in §7, because that is the number the M8 hypothesis turns on.

**Verdict: REFUTED. Should targeted zeroth-order auditing become the final paper
method? NO.**

---

## 2. ZOO-Prune Audit

### 2.1 What the paper actually does

> **ZOO-Prune: Training-Free Token Pruning via Zeroth-Order Gradient Estimation
> in Vision-Language Models.** Youngeun Kim, Youjia Zhang, Huiling Liu, Aecheon
> Jung, Sunwoo Lee, Sungeun Hong. **CVPR 2026**, pp. 39572–39582.
> arXiv:2509.24837. Code: `github.com/AIM-SKKU/ZOO-Prune`.

| item | value | source |
|---|---|---|
| perturbation site | the **pre-projector** vision-encoder output `X ∈ R^{N_v × d_v}` | paper §3.2, Algorithm 1 |
| response read at | the **projector output** `Z = M(X) ∈ R^{N_v × d_l}` | code `llava_arch.py` |
| direction | `u_j ~ N(0, I)`, then `u_j ← u_j / ‖u_j‖₂` (Gaussian, not Rademacher) | Eq. 2 |
| perturbation | `x_i ± h·u_j`, `h = 0.01` | Eq. 2, `NOISERECOV_INTENS` |
| **direction sharing** | **one bank of `m` directions applied to every token at once** | code: `u.unsqueeze(1).expand(-1, N_v, -1)` |
| score | `δ_{i,j} = [M(x_i + h u_j) − M(x_i − h u_j)] / 2h`; `S(i) = (1/m) Σ_j ‖δ_{i,j}‖₂` | Eq. 3–4 |
| aggregation | **mean** over directions (a stale docstring says "var"; the code uses `coeff.mean(dim=0)`), **L2** over the projected dimension, **no sign** | code |
| defaults | `m = 64` (robust over 16–160), `h = 0.01` | §4 |
| extra full-model forwards | **zero** — two projector forwards on `m·N_v` tokens | §4.3, App. D.2 |
| reported overhead | ~**0.3 %** of baseline prefill FLOPs (with a rank-128 projector); prefill **2.59×**, E2E **2.30×** on LLaVA-NeXT-7B/POPE at 160 tokens | App. D.2, Fig. 7 |
| selection | **SADS**: greedy max-min cosine diversity on the *projected* features, gated multiplicatively by the min-max-normalised `Ŝ` | §3.3 |
| lineage | central-difference **Randomized Gradient Estimator** (Duchi et al. 2015; Nesterov & Spokoiny 2017). **Not** SPSA | §3.2, App. B |

Three facts about it matter for everything below, and none is stated plainly in
the paper text:

1. **It probes the projector, not the LLM.** The LLM is never run. The estimator
   is the Jacobian magnitude of the modality-alignment MLP.
2. **The direction bank is shared across tokens.** This is in the code and
   contradicted by the paper's own cost example; the code is authoritative.
   Sharing is what makes `m = 64` cost two batched projector calls instead of
   `2·m·N_v` sequential ones.
3. **It is query-free and token-local.** `S(i)` is a function of `x_i` alone:
   no instruction, no text, no other token.

### 2.2 What we reproduced, and the gate that proves it

`m8_zop.py` reproduces Eq. 2–4 at the equivalent site on Qwen3-VL: the
perturbation is applied to the vision tower's pre-merger output `(4096, 1152)`,
and the response is read at `Qwen3VLVisionPatchMerger`'s output `(1024, 4096)` —
which *is* the `image_features` the incumbent pruner selects over. Gate
**G-MERGER** requires `merger(captured input) == tower output` **bit-for-bit**;
it passes on **450/450** instances with `max |Δ| = 0.0`.

Cost, measured on the same A40 under the same harness:

| direction budget `m` | 1 | 2 | 4 | 8 | **64** (published) |
|---|---:|---:|---:|---:|---:|
| ZO-P, median | 2.4 ms | 4.7 ms | 8.3 ms | 15.9 ms | **127.0 ms** |
| as % of B2's 278.6 ms TTFT | 0.9 % | 1.7 % | 3.0 % | 5.7 % | **45.6 %** |

The grid runs `m ≤ 8` for the depth analysis and `m = 64` as well, so the
published default is covered rather than argued around — §8 shows it changes
nothing measurable.

### 2.3 M8 is not ZOO-Prune, and the difference is the point

| | ZOO-Prune | M8 |
|---|---|---|
| role of the estimator | **replaces** the selector | **audits** an existing selector |
| scope | global: every visual token | targeted: a nominated pool only |
| final set | the estimator's own top-k × diversity | the incumbent's set, corrected at the margin |
| estimator site | projector (cheap, query-free) | projector **and** LLM (expensive, query-conditioned) |
| what it is asked to find | which tokens are sensitive | which *dropped* tokens carry the answer |

The last row is the whole difficulty. ZOO-Prune asks "which token's perturbation
moves the projector's output", a question with a cheap and well-conditioned
answer. M8 asks "which dropped token would have carried the answer", which is a
question about the LLM's behaviour, and §5 shows it has no answer at the
deployed precision.

---

## 3. Candidate Pool Audit

Four pools, frozen before any number was read, plus the global control. All
numbers on the held-out 210 (`val` + `test`), per instance then averaged.

| pool | definition | mean \|pool\| | p10 | p90 | `cov@8` | `yield@8` | chance | meanTR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **P0** | all 768 dropped tokens | 768.0 | 768 | 768 | 1.0000 | 0.0104 | 0.7500 | 383.5 |
| **P1** | `cos_s0c` top-32 | 32.0 | 32 | 32 | **0.2994** | **0.0749** | 0.0312 | 210.5 |
| **P2** | `cos_s0c` top-64 | 64.0 | 64 | 64 | 0.3595 | 0.0449 | 0.0625 | 250.6 |
| **P3** | 4 strong proxies × top-8, unioned | 25.0 | 21 | 28 | 0.2649 | 0.0863 | 0.0244 | 229.5 |
| **P4** | 4 strong proxies × top-16, unioned | 48.8 | 43 | 54 | 0.3262 | 0.0541 | 0.0476 | 255.7 |

`cov@8` = |pool ∩ teacher-top-8 dropped| / 8. `yield@8` = the same intersection
over the slots the pool spends. `meanTR` = mean teacher rank of the pool's
members (0 = the teacher's best dropped token).

Two implementation notes. The ZOO-Prune score is **token-local by construction**
(§2.1), so scoring the pool by subsetting the global score is *exactly* the same
as scoring the pool directly — no approximation is involved. And ZOO-Prune's own
selection rule (SADS: sensitivity × diversity greedy) is **not** reimplemented
here: this stage asks about the estimator, and every method arm is a rank inside
a fixed pool, which SADS never is.

**The pools are genuinely enriched.** P1 holds 9.6× chance coverage of the
teacher's top-8 head in 32 slots, and its in-pool oracle at `r = 8` reaches mean
teacher rank **31.3** — 4.2× shallower than what `cos_s0c` actually delivers
(131.0) from the same pool. The ceiling falls as the pool grows, exactly as it
must: **3.5** for the full 768 (P0 — which reproduces M3-v0's global oracle band
of 3.5–15.5, where a rescue was measured at +7.7 to +14.9 macro), **20.1** for
64 slots (P2), **22.7** for 48.8 (P4), **31.3** for 32 (P1), **39.6** for 25
(P3). A 25–64-slot cheap pool therefore caps a perfect auditor somewhere between
teacher rank **20 and 40**.

Read against M3-v0's two anchors — rank 73.7–131.9 measured as worth ≈ 0 macro,
rank 3.5–15.5 measured as worth +7.7 to +14.9 — a 32–64-slot pool's ceiling sits
*between* them: clearly better placed than the students that failed, and not
proven to convert. So the honest statement of what nomination buys is this: it
turns an unrankable 768-token problem into a 25–64-token problem whose ceiling
is plausibly inside the converting band, and it is the only step in the M8
pipeline that does anything. **The nomination step is not the bottleneck; the
gap between the pool's ceiling and what any real auditor achieves is** — and
that gap is the whole of M6, M7 and now M8.

---

## 4. Sensitivity Quality

The decisive table. Every rule is confined to its pool; `r` is the rescue size;
`head recall@r` is |pick ∩ teacher-top-r dropped| / r; mean and median teacher
rank are over the picked tokens (0 = the teacher's best dropped token).

### P1 — `cos_s0c` top-32

Method | r | head recall@r | mean teacher rank | median teacher rank
|---|---:|---:|---:|---:|
random-in-pool (20 seeds) | 8 | 0.0731 | — | —
`cos_s0c` (best single) | 8 | **0.2089** | **131.0** | 84.8
M6 D4 disagreement | 8 | 0.2149 | **127.8** | 76.8
M6 max-fusion | 8 | 0.2155 | **127.5** | 76.0
ZO-P m=1 | 8 | 0.1607 | 187.6 | 149.0
ZO-P m=2 | 8 | 0.1702 | **184.2** | 147.5
ZO-P m=4 | 8 | 0.1637 | 185.6 | 144.5
ZO-P m=8 | 8 | 0.1655 | 186.6 | 148.0
**oracle in pool** | 8 | **0.2994** | **31.3** | 23.0

*(the remaining pool × r cells are in `reports/m8_tables.md`, T3)*

**Reading it.** Three things, in order of importance:

1. **The oracle inside the pool is at mean teacher rank 31.3.** The pool
   contains what a rescue needs. This is the sharpest possible statement that
   the *nomination* step is sound.
2. **ZO-P does not reach it, and does not reach the cheap proxy either.** At
   `r = 8` ZO-P's best budget (m=2) sits at mean teacher rank **184.2** against
   `cos_s0c`'s **131.0** — a **53-rank** deficit — and against the oracle's
   31.3.
   It beats the matched random draw (0.170 vs 0.073 recall) but loses to the
   free column by **55 teacher ranks** and to both M6 composition rules by ~57.
3. **More directions do not help.** `m = 1, 2, 4, 8` are within 4 ranks of each
   other on every pool, and the published `m = 64` lands inside the same band.
   The estimator is not variance-limited; it is *bias*-limited, and §8
   identifies the bias.

### ZO-L through the same gate

The brief asks for *the* zeroth-order estimator inside the pool. ZO-P is the
published one, but the estimator the M8 hypothesis actually needs is the
LLM-level one (§5), so it is put through the identical gate: 40 held-out
instances, the same P1 pool of 32 tokens, two directions per token, central
differences, `h_rel = 0.1`, at the 1024-token context the teacher uses.

Method | r | head recall@r | mean teacher rank | median teacher rank
|---|---:|---:|---:|---:|
random-in-pool (20 seeds) | 8 | 0.0748 | — | —
`cos_s0c` | 8 | **0.2219** | **129** | 94
`imp` | 8 | 0.1531 | 140 | 92
**ZO-L, \|ΔJ\|** | 8 | **0.0906** | **200** | 143
ZO-L, signed ΔJ | 8 | 0.1031 | 186 | 147
ZO-L, ‖Δ logits‖ | 8 | 0.0781 | 221 | 176
oracle in pool | 8 | 0.2812 | 37 | 31
random-in-pool (20 seeds) | 16 | 0.0971 | — | —
`cos_s0c` | 16 | 0.1625 | 173 | 127
**ZO-L, \|ΔJ\|** | 16 | 0.1141 | 209 | 160
oracle in pool | 16 | 0.1953 | 88 | 83

**ZO-L ranks at chance.** At `r = 8` it recovers 0.0906 of the teacher's head
against the matched random draw's 0.0748 — **1.21× chance**, where `cos_s0c`
gets 2.97× and the pool's oracle 3.76× — and its picks sit at mean teacher rank
**200** against `cos_s0c`'s 129. Its rank agreement with the teacher across the
pool is **+0.014** (and +0.023 for the ‖Δ logits‖ response): indistinguishable
from zero, exactly as the split-half reliability of 0.069 predicted. **Over a
third of the pool's tokens (36 %) return a response of exactly zero**, and
`|dJ|` takes 3.9 distinct values across the 32 tokens. (At `r = 32` the pool is
exhausted and every rule returns the whole pool — 0.1562 for all of them — so
that row carries no information.)

This is the cleanest possible confirmation of §5.4: an estimator whose
split-half reliability is 0.07 produces a ranking with no teacher information,
and the depth gate fails for the LLM-level estimator on its own terms — before
its cost is even considered.

### The global pool says it plainly

On **P0** — the full 768 dropped tokens, no nomination at all — ZO-P's best
`r = 8` pick sits at mean teacher rank **448**, against a chance value of 384
(768/2) and `cos_s0c`'s **131.0**. The estimator is *worse than chance* when it is
allowed to choose from everything, and it is worse than chance for the same
reason it is weak inside a pool: it is ranking by a quantity the teacher's head
is anti-correlated with (§8).

### Where the gate stands

| gate (fixed before any number was read) | outcome |
|---|---|
| `r = 8` mean teacher rank **< 40**, ideally < 25 | ❌ **184.2** (best cell anywhere in the grid) |
| clearly better than `cos_s0c` | ❌ **worse** — 184.2 vs 131.0 in the same pool |
| clearly better than the best M6 cheap scorer | ❌ D4 = 127.8, max-fusion = 127.5, both better than ZO-P |
| same direction on several pools | ❌ ZO-P is worse than `cos_s0c` on **P0, P1, P2, P3 and P4** at `r = 8`, and on **13 of 15** pool × r cells using its best budget |
| depth improves, not only recall | ❌ neither improves |

**The gate fails on all five clauses. Phases 2–5 are not run.**

### The global arm, for completeness

Applied to all 1024 tokens and cut to the incumbent's own budget of 256:

| selector | teacher head-8 kept | head-16 kept | head-32 kept |
|---|---:|---:|---:|
| **chance** (256 of 1024) | 0.2500 | 0.2500 | 0.2500 |
| ZO-P m=1 | 0.3190 | 0.3012 | 0.2705 |
| ZO-P m=8 | **0.3280** | 0.3027 | 0.2723 |
| B2's own S0 | 0.2280 | 0.2295 | 0.2387 |
| oracle top-256 | 1.0000 | 1.0000 | 1.0000 |

ZO-P as a *global selector* keeps 0.328 of the teacher's head-8 against a chance
rate of 0.250 — **1.31× chance** — while EADP's own S0 keeps 0.228, i.e. **0.91×
chance**. Both facts are real and neither is a mechanism. M6 §3.1 already
established that EADP's ranking is *anti*-correlated with the teacher's head, so
"better than EADP on this metric" is a bar set below chance. ZO-P clears chance
by a third; an oracle clears it four-fold. And the metric is about the *kept*
set, not about the depth wall, which §4 measures inside the dropped set.

---

## 5. The LLM-level estimator, and why it does not exist

### 5.1 What the hypothesis actually needs

ZO-P is query-free and token-local (§2.1). The estimator that could break the
depth wall is the same zeroth-order idea applied to the objective the teacher
differentiates:

```
J(v)    = max logit at the last prompt position            (P1-G2's own J)
s(i)    = [ J(v + h·u_i) − J(v − h·u_i) ] / 2h             u supported on token i
ZO-L(i) = mean_j | s(i) |
```

This is a *targeted* estimator by construction: it needs one full forward per
token, so it can only ever be afforded on a nominated pool. It is also the only
estimator in this stage whose failure or success would be informative about the
hypothesis, because it is a finite-difference approximation of the teacher
itself. Everything in this section is a measurement on the frozen bank.

### 5.2 Control 1 — the harness is deterministic

Four identical rows in one batch produce **bit-identical** logits
(`max |Δ| = 0.0`, same argmax on all four). The forward is not the noise source.

### 5.3 Control 2 — the response floor, and where the per-token effect sits

One batch, one fixed shape, 16 arms, on a `fit` instance. The response is the
top logit at the last prompt position, `J₀ = 41.5`. bf16 carries 7 stored
mantissa bits, so the quantum of a value in `[32, 64)` is `2⁻²` = **0.25**, not
`J₀·2⁻⁸` = 0.162 — an error this stage made first and corrected before
reporting.

| arm | ΔJ | ‖Δ logits‖₂ |
|---|---:|---:|
| identity | 0.000 | 0.000 |
| perturb one token, h=0.01 | −0.250 | 21.6 |
| perturb one token, h=0.1 | −0.250 | 22.9 |
| perturb one token, h=1.0 | 0.000 | 20.2 |
| perturb one token, h=3.0 | 0.000 | 27.2 |
| scale one token ×2 | 0.000 | 28.7 |
| **zero one token entirely** | **−0.250** | 39.8 |
| copy another token over it | 0.000 | 33.7 |
| zero 24 tokens | −0.750 | 166.6 |
| zero 256 tokens | **−8.250** | 846.1 |
| zero all 1024 | −16.375 | 1649.6 |

**A single token, perturbed by 300 % or deleted outright, moves the objective by
at most one quantum.** This table is one instance; the `k`-curve on the full
held-out panel (n = 16) is in T8, and it says the same thing more precisely:

| k tokens zeroed | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean ΔJ | −0.438 | −0.562 | −1.141 | −1.336 | −1.523 | −1.750 | −1.539 | −2.008 |
| in quanta | −1.75 | −2.25 | −4.56 | −5.34 | −6.09 | −7.00 | −6.16 | −8.03 |

The curve saturates almost immediately — deleting 256 of 1024 tokens (a quarter
of the image) moves the objective by **8 quanta**, and the average token's share
of that is **0.031 quanta**. The audit's small-`h` perturbation is weaker than a
deletion, so for a typical token the true effect is a small fraction of one
quantum. §5.4 measures what that does to the estimator.

The same holds for the hidden state. `‖Δh_last‖` is **14.8** when a whole token
is deleted and **15.7** when *one component of one token is moved by a single
bf16 ulp* — the response is a fixed numerical floor of ≈ 15, not a function of
the perturbation's size. (It is not saturation of a smooth map: zeroing 16 tokens gives
70.8 and 256 gives 182.9, so the floor is ≈ 15 and one token's true effect is
below it.)

### 5.4 Control 3 — the estimator's own reliability, and the end of the argument

The controls above bound the *effect*. This one measures the *estimator*: on the
held-out panel (n = **16**), the P1 pool of 32 tokens, three step sizes and three
responses, `m8_zol.py` runs the ZO-L score twice with independent direction draws
(seed 0 vs seed 1) and asks whether the two agree.

**Split-half reliability** (Spearman over the 32 pool tokens, per instance then
averaged) and **agreement with the teacher**:

| h | response | split-half ρ | ρ vs \|teacher\| | spread (sd/mean) |
|---:|---|---:|---:|---:|
| 0.01 | \|dJ\| | 0.069 | +0.004 | 1.194 |
| 0.01 | ‖Δ logits‖ | −0.029 | +0.028 | 0.235 |
| 0.01 | ‖Δ h_last‖ | 0.065 | +0.034 | 0.222 |
| 0.1 | \|dJ\| | −0.014 | −0.011 | 1.229 |
| 0.1 | ‖Δ logits‖ | 0.004 | +0.048 | 0.217 |
| 0.1 | ‖Δ h_last‖ | 0.049 | −0.031 | 0.249 |
| 0.5 | \|dJ\| | 0.018 | +0.033 | 1.173 |
| 0.5 | ‖Δ logits‖ | 0.096 | +0.009 | 0.223 |
| 0.5 | ‖Δ h_last‖ | 0.086 | +0.007 | 0.229 |

**Every split-half reliability is ≈ 0.** The best of the nine cells is 0.096;
the three `|dJ|` cells — the estimator the brief names — are 0.069, −0.014 and
0.018. A ranking built on a score with reliability 0.07 is noise, and no step
size or response fixes it. **The teacher correlations are zero too**: |ρ| ≤ 0.048
in all nine cells, against the cheap columns' agreement with the same teacher on
the same instances — oriented `vis_norm` **+0.239**, oriented `imp` **+0.202**,
oriented `cos_s0c` **+0.145** (all measured here, in the dropped set, per
instance then averaged). The estimator is not merely weak; it carries less
teacher information than any column the pruner already computes.

**Why.** `|dJ|` takes **2.9 distinct values** across a 32-token pool (min 2, max
4) — the response is quantised to the logits' bf16 step, so the "score" is a
2-bit indicator of whether the perturbation happened to move the response by
zero, one or two quanta. The `k`-curve of §5.3 explains why that indicator
carries no per-token information: the average token's true effect is 0.031
quanta, so which of the three values a token lands on is set by rounding, not by
the token.

### 5.5 What this means

The deployed model is bf16, and a single visual token carries ~1/1024 of the
last position's attention. Its influence on any downstream quantity is below the
quantisation of that quantity, and the measurement confirms it three ways: the
response is constant in `h`, it takes three values, and it does not reproduce
across independent direction draws. This is not a tuning problem — raising `h`
does not help, adding directions does not help (there is no signal to average),
and measuring a different response does not help (the hidden state has the same
floor, and its reliability is 0.049–0.086).

And it is not a prediction. §4 runs ZO-L through the identical Phase-1 gate and
it lands where a reliability of 0.07 says it must: **1.21× chance** inside the
pool at `r = 8` (recall 0.0906 against random's 0.0748), mean teacher rank
**200** against `cos_s0c`'s 129, and a rank agreement with the teacher of
**+0.014**.

That is the bottleneck, stated exactly: **the targeted zeroth-order estimator
requires a measurement resolution the deployment does not have, and a
per-token forward count the deployment cannot afford.** Either one alone would
kill the M8 hypothesis; both hold.

### 5.6 The one design that could evade both, and why it does not

There is a third option, and it is ZOO-Prune's own trick applied one level up.
Perturb *every* pool token at once with a **shared** direction, and use a
structured (orthogonal / Hadamard) sign pattern across `K` forwards to decode
the per-token values by inversion:

```
forward k perturbs token i by  h * H[k,i] * u        (u shared, unit)
dJ_k / h = sum_i H[k,i] * (g_i . u)
=>  (g_i . u) = (1/K) sum_k H[k,i] * (dJ_k / h)      for K >= P
```

This costs `K` forwards instead of `2P` — a genuine saving — and it works iff
the **aggregate** response rises above the output's quantum. `m8_zol_group.py`
measures that directly, one batch per instance, 8 instances:

| arm | mean \|ΔJ\| | in quanta | below half a quantum |
|---|---:|---:|---:|
| **zero the whole 32-token pool** (positive control) | **1.594** | **6.38** | — |
| perturb the 32-token pool, 8 shared directions | 0.109 | **0.44** | **58 %** of draws |
| perturb all 768 dropped tokens, 8 shared directions | 0.102 | **0.41** | **62 %** of draws |

The positive control proves the setup can see a group effect: *deleting* the
pool moves the objective by 6.4 quanta. But a shared **unit-direction**
perturbation of the same pool — the thing the Hadamard design has to invert —
sits at **0.44 quanta**, below the quantum on 58 % of draws, and it does not
improve when the group grows from 32 tokens to all 768. The per-token values the
inversion would recover are therefore reconstructed from sub-quantum
measurements, and the design fails for the same reason the per-token one does.
**There is no structured-direction escape.**

---

## 6. Fixed-Budget Results and Independent Confirmation

**Not run.** The brief gates Phases 2–5 on the Phase-1 depth gate, and the gate
failed on all five of its clauses (§4). Running a 256-token generation grid for
an auditor whose picks sit at mean teacher rank **184.2** — deeper than the 131
that M6 already measured as worth ≈ 0 downstream, and 5.9× deeper than the same
pool's oracle at 31.3 — would be measuring a number the depth analysis has
already bounded. The LLM-level estimator is worse still (mean rank **200**,
1.21× chance), so it does not rescue the case for a generation run either.

This is the same discipline M6 applied when its Phase-1 gate failed, and the
opposite of the practice M7's own report identifies as the project's worst
habit: *"This is the second time in the project that a 150-instance panel
produced a headline the larger panel reversed."* No screening panel was run,
because a screening panel is for deciding whether to run the confirmation panel,
and the depth gate has already decided.

---

## 7. Efficiency

Reported because the M8 hypothesis *is* a cost claim — "spend strong estimation
only on a small suspicious subset" — and the cost is what kills it.

| arm | visual tokens | cost | vs B2 TTFT |
|---|---:|---:|---:|
| B0 full-1024 | 1024 | 397.9 ms TTFT | 1.43× |
| B1 official EADP facility@256 | 256 | 310.1 ms TTFT | 1.11× |
| **B2 EADP block8@256** (incumbent) | 256 | **278.6 ms TTFT** | 1.00× |
| global ZOO-Prune, estimator only (m=8) | 1024 | **+16.0 ms** | **+5.6 %** |
| global ZOO-Prune, estimator (m=64, published default) | 1024 | **+127.0 ms** | **+46 %** |
| **M8 targeted ZO-L, 32-token pool, central differences** | 288–1024 | **13.1 s** | **47×** |

The ZO-L figure is arithmetic from a measured per-row cost, not an estimate:
a batched perturbed forward costs **205 ms/row** at the 1024-token context and
**55.8 ms/row** at the 288-token `S0 ∪ pool` context, and **flat in batch size**
(206 ms/row at batch 8, 205 at batch 16, 203 at batch 32 — the forward is
weight-bandwidth-bound, so batching buys 4 % at the 1024-token context and 1.5×
at 256). There is no batching trick left to play.

A 32-token pool with central differences is 64 rows = **13.1 s** at 1024 context
or **3.6 s** at 288. For reference, `r = 8` needs only 16 rows, still **1.8 s**.

**The targeted-versus-global comparison the brief asks for:**

| | perturbations | context | measured cost |
|---|---:|---|---:|
| global ZO-Prune (projector level, published) | 2 × 1024 tokens × m=8 | 4096 patches | **16 ms** |
| global ZO-L (LLM level, all 1024 tokens) | 2048 forwards | 1024 | **420 s** |
| **M8 targeted ZO-L (32-token pool)** | 64 forwards | 1024 | **13.1 s** |

Targeting buys **32×** over the global LLM-level estimator — and still lands
**47× above** the incumbent it is supposed to improve. The reason is structural:
ZOO-Prune is cheap because the projector is cheap, and moving the same estimator
to the LLM moves it from a two-call MLP probe to a per-token 8-billion-parameter
forward. **Targeting does not make an unaffordable estimator affordable; it only
divides an unaffordable number by the pool size.**

---

## 8. Mechanism Analysis

> **Did targeted zeroth-order auditing solve the ranking-inside-the-pool problem
> exposed by M6?**

**No. It reproduced it, and this stage explains why.**

**What the published estimator is a function of.** ZO-P's rank agreement with
each of the 27 cheap columns the M5 bank already caches, per instance then
averaged, m = 8:

| cheap column | mean ρ | frac. of instances positive |
|---|---:|---:|
| **`vis_norm`** — ‖post-merger feature‖ | **+0.9863** | **1.00** |
| `glob` | +0.3439 | 1.00 |
| `loc_top5` | +0.3102 | 1.00 |
| `fused` | +0.2905 | 0.99 |
| `loc_max` | +0.2484 | 1.00 |
| `cos_s0c` | −0.2368 | 0.05 |
| `imp` | +0.1459 | 0.94 |

**The published estimator is `vis_norm`.** It is a rank restatement, at ρ =
0.986, of a quantity the incumbent's pruner already computes for free — the
column M6 measured at marginal cost 0 and at fit recall@16 = 0.0651, one of the
weakest in the bank. That is not a reproduction failure: §2.1 shows why it is
*expected*. The merger is `LayerNorm → Linear → GELU → Linear`; its Jacobian is
a function of the normalised input direction, and the GELU's activation mask
drives both `‖J u‖` and `‖M(x)‖`.

**And it is a property of the model, not of the arithmetic** — the one thing
that had to be ruled out, because `h = 0.01` on pre-merger features of norm
~3741 is a step of 2.9e-4 per component and only **16 %** of it survives the
bf16 cast. `m8_zop_validate.py` computes the same quantity three ways on the
same merger (n = 12, `test` split):

| comparison | mean Spearman | reading |
|---|---:|---|
| finite difference (bf16) vs finite difference (fp32) | **0.982** | the deployed reproduction is not corrupted by the dtype |
| finite difference (fp32) vs **exact** reverse-mode `‖Jᵀv‖` | **0.991** | the finite-difference estimator is faithful |
| **exact** `‖Jᵀv‖` vs the token's output norm | **0.973** | the Jacobian norm tracks the output norm — a fact about the merger |
| finite difference (bf16) vs the bank's `vis_norm` | **0.989** | what the deployed score actually ranks |
| split-half, two direction banks (bf16) | **0.997** | the estimator is not variance-limited |

The exact control uses no step size and no finite difference at all, so it cannot
inherit a quantisation artefact from either. **At ρ = 0.973, `S ≈ ‖v‖` is what
this merger does.**

**Why it points the wrong way — and the arithmetic closes.** ZO-P agrees with
the gradient teacher at ρ = **−0.227** over all 1024 tokens and **−0.236**
inside the dropped set, and it is positive on **0.5 %** of instances. High ZO-P
marks teacher-*unimportant* tokens. This is not a sign convention — both signs
are reported. And it is exactly what the rest of §8 predicts: the **raw**
`vis_norm` column correlates with the teacher at **−0.239**, ZO-P tracks raw
`vis_norm` at **+0.986**, and ZO-P's own teacher correlation is **−0.236**. The
three numbers agree to within 0.003, which is the whole mechanism in one line:
the published estimator inherits `vis_norm`'s ranking *and its sign*.

**Why more directions and more compute do not fix it.** `m = 1, 2, 4, 8` are
within 4 mean-teacher-rank of each other on every pool, and the published
default `m = 64` changes nothing measurable: teacher agreement **−0.2361**
against −0.2360 at `m = 8`, `vis_norm` agreement 0.9876 against 0.9863,
top-256 overlap with S0 0.2719 against 0.2721. The estimator has converged — at
`m = 1` it is already a split-half-ρ = 0.997 estimate of its own target. Adding
directions reduces variance around a target that is itself uninformative.
**This is a bias problem, and the bias is the choice of what to measure — the
projector, rather than the model.**

**And the model cannot be measured instead** (§5): the LLM-level estimator's
per-token signal is 0.031 quanta, its score takes 2.9 distinct values, its
split-half reliability across direction draws is ≈ 0 (best cell 0.096), and when
it is finally run through the gate it lands at 1.21× chance with a teacher
agreement of +0.014. The two halves of the M8 hypothesis therefore fail for
*different* reasons — the affordable estimator is a restatement of a free
column, and the informative one is not measurable — and no combination of them
is a rescue.

**The one thing that works.** Nomination. The pools hold the answer: P1's
in-pool oracle at `r = 8` is mean teacher rank **31.3**, and the whole 768-token
pool's is **3.5** — the same global oracle M3-v0 priced at +7.7 to +14.9 macro.
The bottleneck is not *finding candidates* and never was.
It is *ranking them*, and this stage adds a third independent confirmation of
that, after M6's cheap-score audit and M7's keep-them-all hedge.

### The brief's four questions, answered

**Q1 — Does ZO sensitivity break the cheap-feature depth wall?** **No — both
estimators were run and both failed.** ZO-P, the published one: best cell
anywhere is mean teacher rank **184.2** at `r = 8` (gate: < 40), worse than
`cos_s0c` (131.0) in the same pool, worse on 13 of 15 pool × r cells, worse than
both M6 composition rules. ZO-L, the LLM-level one that *could* have: mean
teacher rank **200** at `r = 8` against `cos_s0c`'s 129, **1.21× chance**, teacher
agreement **+0.014** — a ranking with no teacher information, exactly as its
split-half reliability of 0.069 predicted.

**Q2 — Does candidate nomination have value?** **Yes, and it is the only part of
the pipeline that does.** ZO-P restricted to the 32-token P1 pool scores 0.1655
recall / mean rank 186.6 at `r = 8` (m=8); the same estimator over all 768
dropped tokens (P0) scores 0.0170 / mean rank **447.7**. Nomination moves the
estimator from below chance (447.7 of 768, against a chance value of 384) to a
real signal (186.6) — a 2.4× depth gain.
But the cheap score that *built* the pool, applied inside it, is better still
(131.0), so the audit does not repay the nomination it depends on.

**Q3 — Does the gain come from the ZO audit or from ordinary replacement?**
**From neither, because there is no gain to attribute.** ZO-P does beat the
matched random-in-pool draw (P1 `r = 8`: 0.1655 vs 0.0731 recall, 186.6 vs —), so
the audit is not worthless in isolation. It is simply dominated: a free column
inside the same pool beats it by 55 teacher ranks. Since no generation was run
(§6), no macro claim is made in either direction, and the matched random control
remains the bar any future rescue formulation must clear.

**Q4 — Targeted ZO versus global ZOO-Prune: what is the difference, and what is
the efficiency value?** The difference is *which model* is probed. ZOO-Prune
probes the projector — 2 batched MLP calls, 16 ms at `m = 8`, 5.6 % of B2's
TTFT, and it
never runs the LLM. M8's targeted estimator probes the LLM — one 8B forward per
token, 205 ms/row, 13.1 s for a 32-token pool. **Targeting divides the global
LLM-level cost (420 s) by the pool size, giving 13.1 s — a 32× saving that still
lands 47× above the incumbent.** The efficiency value of targeting is real and
insufficient: it converts an impossible cost into an unaffordable one. The
genuinely cheap zeroth-order estimator is cheap *because it avoids the LLM*, and
avoiding the LLM is exactly what makes it query-free and token-local — i.e.
exactly what makes it `vis_norm` and unable to see the teacher's head.

---

## 9. Final Verdict

# REFUTED

> **Should targeted zeroth-order auditing become the final paper method? NO.**

Against the brief's own gate:

| criterion | outcome |
|---|---|
| fixed 256 tokens | n/a — no method arm was built |
| `r = 8` mean teacher rank < 40, ideally < 25 | ❌ **184.2** |
| clearly better than `cos_s0c` / best M6 cheap scorer | ❌ **worse** than both |
| multiple pools, same direction | ❌ worse on 5 of 5 pools at `r = 8` |
| depth improves, not only head recall | ❌ neither improves |
| overhead acceptable | ❌ 47× B2's TTFT, and not measurable anyway |

**Where the bottleneck is, exactly.** There are two, and they are independent:

1. **The affordable zeroth-order estimator is a restatement of a free column.**
   ZOO-Prune's estimator, reproduced bit-exactly at the right site, agrees with
   `vis_norm` at ρ = 0.986 and with the gradient teacher at ρ = −0.236. It is
   query-free and token-local *by construction*, and it cannot see what the
   teacher sees. More directions do not help because the limitation is bias —
   and an fp32 reverse-mode control shows the agreement is a property of the
   merger (the **exact** Jacobian norm tracks the output norm at ρ = 0.973),
   not an artefact of the published step being lost to bf16 rounding.
2. **The informative zeroth-order estimator is below the deployment's numerical
   resolution.** The objective's bf16 quantum is 0.25; the average visual token's
   share of the whole-block effect is 0.031 quanta; a one-ulp input change and a
   whole-token deletion produce the same response; the per-token score takes 2.9
   distinct values; and its split-half reliability across direction draws is
   ≈ 0 (best of nine cells: 0.096), with |ρ| ≤ 0.048 against the teacher — and
   run through the gate it recovers 1.21× chance inside the pool at `r = 8`.
   Even if it were resolvable, it costs one 8B forward per token — 47× the
   incumbent for a 32-token pool.

**What is left standing**, all reusable:

1. **Nomination is solved; ranking is not.** P1's in-pool oracle at `r = 8` is
   mean teacher rank 31.3 against `cos_s0c`'s 131.0 from the same 32 slots, and
   the 768-token pool's oracle is 3.5. This is now the
   third independent measurement of the same asymmetry (M6 cheap scores, M7
   keep-the-union, M8 zeroth-order), and it should be stated as the project's
   central negative: *the candidates are there and nothing forward-only can
   order them.*
2. **ZOO-Prune's estimator does not transfer to a query-conditioned LLM pruning
   problem.** It transfers *mechanically* — we reproduce it exactly — but what
   it computes at the projector is a token-local magnitude statistic, and a
   token-local magnitude statistic is not what distinguishes a critical miss.
   A paper claiming otherwise on our benchmark would be measuring `vis_norm`.
3. **A numerical caution for the field.** Zeroth-order estimators are usually
   validated at the projector, where the response is read at full precision from
   a small MLP. Moving the same estimator to the LLM is not a matter of paying
   more FLOPs: at bf16 the per-token response is below the representation's own
   quantum, so the estimator returns a quantisation pattern rather than a
   derivative. The cheap diagnostic is **split-half reliability across two
   independent direction draws** — it costs one extra pass, it needs no labels
   and no teacher, and it would have caught this in an afternoon. Any future ZO
   work at the LLM level should report it before reporting a ranking.

**What would change the verdict.** Not a better pool (§3: the pool is already
right), not more directions (§8: bias, not variance — `m = 64` is
indistinguishable from `m = 1`), not a larger budget (§5.3: the response is
constant in `h`), and not the shared-direction Hadamard design that would make
per-token auditing affordable (§5.6: its aggregate sits at 0.44 quanta, below
the quantum on 58 % of draws, and does not improve at 768 tokens). It would take
either a *query-conditioned* forward-only quantity with genuine resolution —
which S2-C0, S2-C5, M3-v0, M3-v2, M6 and now M8 have each failed to find by a
different route — or a precision at which the model is not deployed. Neither is
a knob.

---

## Appendix — files

| file | what |
|---|---|
| `scripts/discovery/m8_common.py` | pools P0–P4, in-pool rules, depth metrics |
| `scripts/discovery/m8_zop.py` | the ZOO-Prune reproduction (Eq. 2–4) on the 450 |
| `scripts/discovery/m8_phase1.py` | the depth-wall grid and the paired bootstraps |
| `scripts/discovery/m8_zop_diag.py` | what the score is a function of |
| `scripts/discovery/m8_zop_validate.py` | fp32 reverse-mode control for the reproduction |
| `scripts/discovery/m8_zol.py` | the LLM-level estimator and its resolution floor |
| `scripts/discovery/m8_zol_group.py` | the shared-direction / Hadamard escape hatch |
| `scripts/discovery/m8_zol_phase1.py` | ZO-L through the same Phase-1 gate |
| `scripts/discovery/m8_analyze.py` | regenerates `reports/m8_tables.md` |
| `scripts/discovery/run_m8.sh` | the whole stage, end to end |
| `outputs/discovery/m8_zop.npz`, `m8_zop_meta.json` | ZO-P scores (450 × 1024 × 5 budgets) + gate + cost |
| `outputs/discovery/m8_phase1.json` | the grid, the pool audit, the significances |
| `outputs/discovery/m8_zop_diag.json` | teacher / cheap-column / selection diagnostics |
| `outputs/discovery/m8_zop_validate.json` | the fp32 reverse-mode control |
| `outputs/discovery/m8_zol.json` | the k-curve, the per-token effect, the split-half reliabilities |
| `outputs/discovery/m8_zol_group.json` | the group / shared-direction measurement |
| `outputs/discovery/m8_zol_phase1.json` | ZO-L's depth-wall grid on the same pools |
| `reports/m8_tables.md` | every table above, generated |

Reused unchanged: `m5_bank.npz` and `m5_vis.npy` (the 27-column feature bank and
the bf16 vision cache), `m6_eapd_order.npz`, `m2_gdep.py`, `instrumented.py`,
`m5_common.py`, `m6_common.py`.
