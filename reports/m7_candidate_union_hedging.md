# M7 — Candidate-Union Hedging

**Stage:** M7, a new formulation. MissGuard (M3), residual capsules (M4),
SafeTrim (M5), conditional utility (S3-A), bundle prediction (S3-B) and
disagreement rescue (M6) are closed and this document does not reopen any of
them. **Status:** all four phases run; **REFUTED at the confirmation panel.**

**The question this document answers, and the only one.**

> M6 showed that the cheap proxies' *union* holds more teacher-head tokens than
> any single proxy, and that no cheap rule can rank inside it. If the union
> cannot be ranked, can it simply be **kept whole** — paying for it by shrinking
> the incumbent core — so that `S_final = core(256 − |U|) ∪ U`?

The bet is deliberately different from M6's. M6 asked *which token in the union
matters* and failed. M7 never asks: it keeps every candidate and buys the slots
by giving up the incumbent's own weakest picks.

**The answer, in one line.** **No. On the 720-instance independent confirmation
panel the method is significantly *worse* than B2 (−2.69 macro, 95 % CI
[−5.21, −0.15]), and the complete union is statistically indistinguishable from
both the matched single-proxy control (+0.77, n.s.) and a matched random draw
(+0.59, n.s.). The core shrink costs more than the union's extra coverage is
worth. Verdict: REFUTED.**

---

## 1. Executive Summary

- **The screening panel lied, and the confirmation panel caught it.** On the
  historical test 150, `U12` scored **63.748** against B2's 59.878 — **+3.87
  macro**, clearing the brief's +3 screening bar and beating its matched
  single-proxy control. On the independent 720-instance panel the same arm
  scores **61.555** against B2's **64.241**: **−2.69**, with a paired CI
  excluding zero. The test-150 number was inside that panel's noise floor
  (MDE₈₀ ≈ 7.0), exactly as the brief warned.
- **The union's mechanism does not survive either.** On the 720, `U12` beats
  matched `cos_s0c` by **+0.77** (CI [−1.46, +2.91]) and matched random by
  **+0.59** (CI [−1.76, +2.94]). Neither is significant, and random is
  indistinguishable from the single proxy. All three rescue sources — union,
  single proxy, random — land within ~0.2 macro of each other.
- **The core shrink is the dominant term, and it is negative.** On the
  confirmation panel at |U| ≈ 59, shrinking the core and adding *nothing* costs
  **−5.13 macro**. Adding 59 tokens back recovers only +1.68 to +2.44 of it: the
  union +2.44, **random +1.85**, the best single proxy +1.68. A *random* rescue
  recovers 76 % of what the union recovers — and **more than the single proxy
  does**. The rescue content is nearly irrelevant; what matters is only that
  some tokens are added back.
- **Flip structure on the 720**: `U12` fixes 40 instances and breaks 60 (net
  −20); matched random fixes 31 and breaks 52 (net −21). The union's entire
  advantage over random is one net instance in 720.
- **Coverage and conversion disagree.** Phase 1 confirms the union genuinely
  holds more teacher-head tokens at large |U| (0.4161 vs 0.3851 at 78 slots),
  but those tokens sit at mean teacher rank ~359 where M3-v0 and M6 both
  measured the value to be ≈ 0.
- **No Phase 4.** The method loses to B2, so no efficiency campaign was run. The
  measured selection overhead is a few tenths of a millisecond plus the 1.73 ms
  feature computation M6 already priced.

**Verdict: REFUTED. Should we continue this into a paper method? NO.**

---

## 2. Existing Setup

| item | value |
|---|---|
| backbone | Qwen3-VL-8B-Instruct, `attn_implementation=sdpa` |
| resolution / visual tokens | 1024 × 1024 → **1024** merged visual tokens (32 × 32) |
| token budget | **256**, hard-asserted on every arm |
| base selector | B2 = official EADP importance + `block8` coverage greedy |
| engine | `m2_gdep.GDEPEngine`, `prellm` mode, unchanged |
| teacher | P1-G2 gradient saliency, **offline label only** — it selects nothing |

**Reference arms** (M2, stored and gated 150/150 hit-for-hit):

| arm | TextVQA | DocVQA | OCRBench | macro |
|---|---:|---:|---:|---:|
| B0 full-1024 | 73.60 | 85.27 | 68.00 | **75.62** |
| B1 official EADP facility@256 | 59.40 | 61.89 | 62.00 | **61.10** |
| B2 EADP block8@256 | 65.40 | 60.24 | 54.00 | **59.88** |

*(B0/B1/B2 are quoted from the stored M2 record on the historical test 150; the
B2 arm is re-run and gated inside every M7 generation run below.)*

**Two panels, and the whole result turns on the difference:**

| panel | n | source | role |
|---|---:|---|---|
| `bank` test150 | 150 | the frozen 450's test split | **screening only** |
| `ext` | **720** | `m1_plan.json` extension rows, 240 per benchmark, asserted disjoint from the bank and from the S2-A causal cases | **confirmation** |

The extension panel carries no teacher scores, which costs nothing here: M7
selects nothing with the teacher. Its B2 baseline is 64.241 macro against the
bank panel's 59.878 — the extension sample is easier, so **every comparison in
this document is within-panel.**

---

## 3. Phase 1 — Candidate-union budget audit (no generation)

Per instance: each proxy nominates its own top-`m` from the 768 dropped tokens;
the union is the set union, **never re-ranked**. `coverage@q` = |pick ∩
teacher-top-q dropped| / q, per instance then averaged. `yield@q` is the same
intersection divided by the **slots spent** — the quantity that decides whether
the union is worth its budget. Chance = |pick|/768.

### 3.1 The union's size and coverage

| proxies | m | mean \|U\| | p10 | p90 | cov@8 | cov@16 | cov@32 | yield@8 | meanTR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A cos | 8 | 8.0 | 8 | 8 | 0.2089 | 0.1339 | 0.0890 | 0.2089 | 244.9 |
| B cos+red | 8 | 13.9 | 12 | 16 | 0.2339 | 0.1548 | 0.1064 | 0.1369 | 289.5 |
| C cos+red+imp | 8 | 20.1 | 17 | 24 | 0.2554 | 0.1744 | 0.1275 | 0.1041 | 297.6 |
| D +nn4 | 8 | 25.0 | 21 | 30 | 0.2649 | 0.1845 | 0.1357 | 0.0863 | 313.9 |
| E all6 | 8 | 39.8 | 32 | 48 | 0.3196 | 0.2414 | 0.1897 | 0.0647 | 345.7 |
| E all6 | 12 | 59.0 | 47 | 71 | 0.3685 | 0.2943 | 0.2418 | 0.0503 | 353.2 |
| E all6 | 16 | 77.6 | 62 | 93 | 0.4161 | 0.3420 | 0.2897 | 0.0432 | 359.0 |
| D +nn4 | 16 | 49.0 | 39 | 59 | 0.3262 | 0.2458 | 0.1951 | 0.0541 | 338.1 |

Adding proxies raises raw coverage and **lowers yield**, monotonically.

### 3.2 The decisive comparison: union vs matched-slot single proxy

Same instance, same number of slots. This is what the stage turns on.

| config | slots | union cov@8 | `cos_s0c` cov@8 | Δ | paired 95 % CI (held-out 210) |
|---|---:|---:|---:|---:|---|
| D +nn4, m=8 | 25.0 | 0.2649 | 0.2821 | **−0.0173** | [−0.0321, −0.0030] * |
| C cos+red+imp, m=8 | 20.1 | 0.2554 | 0.2661 | −0.0107 | [−0.0232, +0.0012] |
| C cos+red+imp, m=16 | 40.1 | 0.3149 | 0.3202 | −0.0054 | [−0.0220, +0.0113] |
| D +nn4, m=16 | 49.0 | 0.3262 | 0.3393 | −0.0131 | [−0.0310, +0.0054] |
| E all6, m=8 | 39.8 | 0.3196 | 0.3202 | −0.0006 | [−0.0226, +0.0214] |
| E all6, m=12 | 59.0 | 0.3685 | 0.3530 | **+0.0155** | [−0.0089, +0.0417] |
| E all6, m=16 | 77.6 | 0.4161 | 0.3851 | **+0.0310** | [+0.0048, +0.0583] * |

**Answer to the brief's question — *can 20–50 union slots buy materially wider
catastrophic-miss coverage than a single proxy?* No.** Across the entire 20–50
slot range the union is at or *below* the matched single proxy. It wins only
from ~59 slots upward, and only the 78-slot point is significant.

**And the tokens it wins with are deep.** The union's mean teacher rank is
**353–359** at those sizes against the single proxy's **347–355** — the union
buys coverage by adding *deeper* tokens. M3-v0 measured a rescue at mean rank
73.7–131.9 as worth ≈ 0 downstream, against an oracle at 3.5–15.5 worth +7.7 to
+14.9. Rank ~355 is far past that.

### 3.3 The mechanism claim, at matched slots

M6's own best rules, re-run at the union's slot count:

| subset | m | slots | rule | cov@8 | cov@16 | cov@32 | meanTR |
|---|---:|---:|---|---:|---:|---:|---:|
| E all6 | 8 | 39.8 | single `cos_s0c` | **0.3202** | 0.2378 | 0.1844 | 336.3 |
| E all6 | 8 | 39.8 | M7 complete union | 0.3196 | 0.2414 | 0.1897 | 345.7 |
| E all6 | 8 | 39.8 | M6 D4 disagreement | 0.3161 | 0.2426 | 0.1897 | 349.1 |
| E all6 | 8 | 39.8 | M6 max-fusion | 0.3161 | 0.2402 | 0.1878 | 351.2 |
| E all6 | 12 | 59.0 | single `cos_s0c` | 0.3530 | 0.2735 | 0.2198 | 346.8 |
| E all6 | 12 | 59.0 | M6 D4 disagreement | 0.3637 | 0.2938 | 0.2420 | 359.3 |
| E all6 | 12 | 59.0 | M6 max-fusion | 0.3661 | 0.2943 | 0.2409 | 360.7 |
| E all6 | 12 | 59.0 | **M7 complete union** | **0.3685** | **0.2943** | **0.2418** | 353.2 |
| E all6 | 16 | 77.6 | single `cos_s0c` | 0.3851 | 0.3054 | 0.2531 | 355.3 |
| E all6 | 16 | 77.6 | M6 D4 disagreement | 0.3982 | 0.3333 | 0.2826 | 363.3 |
| E all6 | 16 | 77.6 | M6 max-fusion | 0.4006 | 0.3342 | 0.2842 | 364.9 |
| E all6 | 16 | 77.6 | **M7 complete union** | **0.4161** | **0.3420** | **0.2897** | 359.0 |
| D +nn4 | 16 | 49.0 | **single `cos_s0c`** | **0.3393** | 0.2554 | 0.2010 | 341.7 |
| D +nn4 | 16 | 49.0 | M6 D4 disagreement | 0.3327 | 0.2652 | 0.2135 | 354.7 |
| D +nn4 | 16 | 49.0 | M6 max-fusion | 0.3321 | 0.2631 | 0.2116 | 355.8 |
| D +nn4 | 16 | 49.0 | M7 complete union | 0.3262 | 0.2458 | 0.1951 | 338.1 |

**The claim `single < fusion/disagreement < complete union` holds only at
|U| ≥ 59, and inverts below it.** At 39.8 and 49.0 slots the single proxy wins
outright. Random matched is 0.05–0.11 throughout, an order of magnitude below
every real rule.

**Phase 1 verdict, stated before Phase 3 ran:** the union has no coverage
advantage in the useful 20–50 slot range, and where it does win it wins with
deeper tokens. Phase 3 was nonetheless run, because the brief asks for it
whenever the complete union shows coverage value and because the *core-shrink
cost* — the one thing coverage cannot price — is only measurable by generation.

---

## 4. Phase 2 — Fixed-budget set construction

```
S_final = core(256 − |U|)  ∪  U          |S_final| = 256 exactly
```

Core constructions, none trained:

| core | definition | role |
|---|---|---|
| **C0** | the first `256 − |U|` of the incumbent's **own greedy selection order** | **primary** |
| C1 | evict the `|U|` most redundant retained tokens (max cosine to the rest of S0) | auxiliary |
| C2 | evict the `|U|` lowest-importance retained tokens | auxiliary |

C0 is the cleanest possible construction: the evicted tokens are the *last* ones
the incumbent's own marginal gain picked, so the shrink discards exactly what
the selector itself rated least. The prefix is read from the greedy order, which
the live pruner already produces — nothing extra is computed at inference.

Matched controls, identical per-instance slot count and identical core:

| control | rescue |
|---|---|
| `COS` | `cos_s0c`'s own top-\|U_i\| — **the arm the stage turns on** |
| `RND` | \|U_i\| uniformly random dropped tokens |
| `CORE` | nothing; the shrunk core alone, at `256 − |U|` tokens (deliberately breaks the budget; diagnostic only) |

Gates: `order[:256]` sorted equals the bank's `s0` on all 450 and all 720 rows;
every delivered set is exactly 256, duplicate-free, and disjoint between core
and rescue; a crashed instance is recorded as `error`, never as a low macro.

---

## 5. Phase 3 — Real generation

Greedy decoding, one generation per instance, `max_new_tokens=2048`, scored by
`scoring.per_sample_hits`. The **B2 identity arm is re-run inside every panel**
and gated: on the bank it reproduces the stored M2 predictions **150/150** and
hits **150/150**.

### 5.1 Screening panel — bank test 150

| arm | TextVQA | DocVQA | OCRBench | macro | Δ vs B2 | 95 % CI | win/loss |
|---|---:|---:|---:|---:|---:|---|---|
| B2 | 65.400 | 60.235 | 54.000 | **59.878** | — | — | — |
| U8 | 64.000 | 57.235 | 64.000 | 61.745 | +1.867 | [−2.98, +7.21] | 14/12 |
| **U12** | 70.000 | 59.245 | 62.000 | **63.748** | **+3.870** | [−2.50, +10.40] | 21/16 |
| U16 | 68.000 | 57.488 | 60.000 | 61.829 | +1.951 | [−4.39, +8.46] | 19/18 |
| COS16 | 67.600 | 54.531 | 56.000 | 59.377 | −0.501 | [−7.71, +6.73] | 21/26 |
| RND16 | 62.000 | 54.353 | 52.000 | 56.118 | −3.761 | [−10.60, +2.94] | 17/23 |
| CORE16 *(178 tok)* | 66.600 | 43.647 | 52.000 | 54.082 | −5.796 | [−12.63, +0.71] | 12/26 |

`U12` cleared the brief's +3 screening bar and beat the matched single-proxy
control, so the confirmation panel was run, as the brief requires.

**Decomposition at m = 16** (|U| ≈ 77, core ≈ 179):

| step | macro | effect |
|---|---:|---:|
| B2, 256 tokens | 59.878 | — |
| CORE16, 179 tokens (shrink only) | 54.082 | **shrink −5.796** |
| RND16 = shrink + 77 random | 56.118 | content **+2.036** |
| COS16 = shrink + 77 from `cos_s0c` | 59.377 | content **+5.295** |
| U16 = shrink + 77 from the union | 61.829 | content **+7.747** |

Content ordering is a clean dose-response: **union > single proxy > random >
nothing.** On this panel.

### 5.2 Confirmation panel — M1 extension, 720 instances

| arm | TextVQA | DocVQA | OCRBench | macro | Δ vs B2 | 95 % CI | win/loss |
|---|---:|---:|---:|---:|---:|---|---|
| B2 | 72.250 | 60.889 | 59.583 | **64.241** | — | — | — |
| **U12** | 69.458 | 56.040 | 59.167 | **61.555** | **−2.686** | **[−5.21, −0.15]** \* | 61/84 |
| COS12 | 68.833 | 55.618 | 57.917 | 60.789 | −3.451 | [−5.93, −0.96] \* | 53/84 |
| RND12 | 71.125 | 54.685 | 57.083 | 60.964 | −3.276 | [−5.72, −0.95] \* | 47/86 |

**The screening result did not replicate.** `U12` is **significantly worse than
B2** on 720 instances, on all three benchmarks.

### 5.3 The contrasts that decide the mechanism

| contrast | Δ | 95 % CI (720) | significant? |
|---|---:|---|---|
| U12 − B2 | −2.686 | [−5.21, −0.15] | **yes, negative** |
| U12 − COS12 (matched single proxy) | +0.766 | [−1.46, +2.91] | no |
| U12 − RND12 (matched random) | +0.591 | [−1.76, +2.94] | no |
| RND12 − COS12 | +0.175 | [−1.93, +2.37] | no |

**The complete union is statistically indistinguishable from both controls.**
Random rescue and single-proxy rescue are indistinguishable from each other.
Whatever the union contributes, it is smaller than this panel can resolve — and
the panel is 720 instances, four times the screening panel.

### 5.4 Why: the shrink does the damage, not the rescue

**Decomposition at |U| ≈ 59, core ≈ 197 (720 instances).** Every arm below has
the same core construction and the same slot count; only what fills the 59
rescue slots changes.

| step | macro | effect |
|---|---:|---:|
| B2, 256 tokens | 64.241 | — |
| CORE12, 197 tokens (shrink only) | 59.113 | **shrink −5.127** |
| COS12 = shrink + 59 from `cos_s0c` | 60.789 | content **+1.676** |
| RND12 = shrink + 59 **random** | 60.964 | content **+1.851** |
| U12 = shrink + 59 from the union | 61.555 | content **+2.441** |

This is the decisive table of the whole stage, and it is worse for the
hypothesis than the screening panel suggested. On 150 instances the content
ordering looked like a clean dose-response (union +7.75 > single proxy +5.30 >
random +2.04). On 720 it is **union +2.44 > random +1.85 > single proxy +1.68**:
a random rescue recovers 76 % of the union's content value, and **more than the
best proxy in a 27-column bank does**. The single proxy's advantage over random
on the screening panel (+3.26) is gone — it is *below* random here.

The core shrink costs 5.13 macro for 59 tokens, and no rescue source recovers
half of it. That is the whole failure.

**Flip structure on the 720:**

| arm | fixed | broken | net | instances changed |
|---|---:|---:|---:|---:|
| U12 | 40 | 60 | **−20** | 145/720 |
| COS12 | 34 | 57 | −23 | 137/720 |
| RND12 | 31 | 52 | −21 | 133/720 |

The union's entire advantage over a random draw is **+9 fixes and +8 breaks** —
one net instance in 720. Per benchmark, `U12`'s net is TextVQA −8, DocVQA −11,
OCRBench −1.

---

## 6. Efficiency

No Phase 4 campaign was run: the method loses to B2, so pricing its latency
would answer a question nobody is asking. What is measurable and relevant:

| quantity | value | source |
|---|---:|---|
| feature computation (6 proxy columns) | **1.73 ms**/instance | M6 `m6_cost.json`, measured |
| set construction (prefix + union + gather) | O(256) index ops | by inspection |
| selection overhead, event-measured | 0.28–4.8 ms | **unreliable — see below** |
| extra model forwards | **none** | the base pruner pass is unchanged |

**Caveat, stated because it would otherwise be misread.** The generation wall
time per injected arm (~115 s) exceeds B2's (73 s), but that is a **harness
artefact**: `read_hedge_ms` calls `a.synchronize()` once per instance, forcing a
pipeline drain that the identity path never incurs. The event-measured
`hedge_ms` values scatter over an order of magnitude (0.28–4.8 ms) across arms
that do identical work, which is the signature of measurement noise rather than
a real cost difference. The honest figure for the method's marginal inference
cost is the **1.73 ms** feature computation plus a 256-element index gather.

M7 is forward-only and training-free at inference: every nomination reads the
EADP score and its components, the selector's similarity matrix, the tower
feature and S0 — all of which the incumbent's own pruner already produces.

---

## 7. Statistical / Reproducibility Checks

| check | result |
|---|---|
| per-instance metrics | every coverage and every accuracy is computed per instance, then averaged; no pooled global top-k |
| panels | screening 150; confirmation **720** (240/benchmark), disjoint from the bank, drawn by `m1_plan.py` with seed 20260924 |
| B2 identity gate | bank panel: **150/150** predictions and **150/150** hits against the stored M2 record |
| set gates | exactly 256 tokens, duplicate-free, core ∩ rescue = ∅, on every instance of every arm |
| order gate | greedy `order[:256]` sorted equals the recorded `s0` on all 450 and all 720 rows |
| paired bootstrap | 4000–6000 resamples, paired over instances, per panel |
| determinism | the analysis re-runs to 0 difference; the set builder is pure numpy with a fixed seed |
| crashed arms | recorded as `error`, never persisted as a score (two were: `CORE16` on the first pass, fixed by an explicit diagnostic flag) |

---

## 8. Failure / Success Analysis

**Did the union catch catastrophic misses?** At the coverage level, yes, at
large |U| — and that is real, not an artefact (§3.2). At the generation level,
no: on the 720 the union is indistinguishable from random.

**Which benchmark?** The damage is worst on DocVQA (U12 net −11 of 720, and
DocVQA is where the bare shrink collapses: CORE16 scores 43.6 against B2's
60.2). TextVQA −8, OCRBench −1. There is no benchmark where the union wins.

**Is one proxy doing all the work?** Yes — `cos_s0c` is the best single proxy at
every size on every benchmark (M6 established this, Phase 1 re-confirms it).
The union's edge over it is +0.77 on 720, not significant.

**Is the coverage gain just deeper tokens?** Yes. The union's mean teacher rank
at the sizes where it wins is 353–359, against the single proxy's 347–355, and
against the M3-v0 oracle's 3.5–15.5 where the +7.7/+14.9 ceiling lives. M7 buys
*more* head tokens by buying *deeper* ones, and S2-C2 measured value to be convex
in rank.

**Why the screening panel said otherwise.** `U12`'s +3.87 on 150 instances is
7 instances net. M3-v0 measured MDE₈₀ = 7.0 macro on that panel and M4 measured
8.2. A +3.87 point estimate with a CI of [−2.50, +10.40] is not evidence, and
the brief says so explicitly. This is the second time in the project that a
150-instance panel produced a headline the larger panel reversed; the practice
of screening on 150 and confirming on the extension set is what caught it.

**What is left standing.** Three things, all negative but reusable:
1. The union's coverage advantage over a single proxy is real but only exists at
   |U| ≥ 59, i.e. ≥ 23 % of the budget, and it is bought in tokens at teacher
   rank ~355.
2. The core shrink is expensive: **−5.13 macro for 59 tokens**, −5.80 for 77.
   Any future rescue formulation that pays for itself by shrinking the incumbent
   core inherits that bill. (M4's `EVICT-r8` — deleting 8 tokens and adding
   nothing — looked cheap at r = 8; at r = 59 it is not.)
3. **The rescue content is worth far less than the slots it occupies.** At
   |U| = 59 the best content in a 27-column bank recovers +2.44 macro against a
   −5.13 bill, and random recovers +1.85. A rescue formulation must therefore
   beat random by a wide margin *and* be worth more than ~half the core it
   displaces; nothing in this bank is close. This is the bar M6's oracle-union
   ceiling (0.2931 vs 0.2069 coverage at k = 8) should have been read against.

---

## 9. Final Verdict

# REFUTED

> **Should we continue developing this into the final paper method? NO.**

Against the brief's own criteria:

| criterion | outcome |
|---|---|
| fixed 256 tokens | ✅ satisfied on every method arm |
| M7 > B2 by ~+3 macro | ❌ **−2.69** on the 720 confirmation panel |
| beats official EADP | ❌ not reached (B1 = 61.10; U12 on 720 = 61.56 on an easier sample) |
| complete union > same-budget strongest single proxy | ❌ +0.77, CI includes 0 |
| matched random clearly worse | ❌ **random is statistically indistinguishable — and beats the single proxy** |
| overhead acceptable | ✅ ~1.7 ms, but moot |

The brief's REFUTED condition is met on both of its clauses: *the union's extra
coverage is cancelled by the core-shrink loss*, **and** *the complete union does
not beat the matched single-proxy or matched-random controls*.

The three reasons, in order:

1. **The effect does not replicate.** +3.87 on 150 instances became −2.69 on 720,
   with the confirmation CI excluding zero on the *negative* side. The screening
   panel's resolution (MDE₈₀ ≈ 7.0) never supported the claim.
2. **The union is not the active ingredient — and neither is the proxy.** At
   matched slots and matched core shrink on 720 instances, union (+0.77 over
   single, +0.59 over random) is within noise of both, and **random beats the
   single best proxy** in content value (+1.85 vs +1.68 against a −5.13 shrink
   bill). The only real effect in the grid is the core shrink, and it is
   negative.
3. **The mechanism's coverage advantage is in the wrong place.** The union's
   extra head tokens sit at teacher rank ~355. Three prior stages (S2-C2,
   M3-v0, M6) each measured that band as worth approximately nothing, and this
   stage reproduces that conclusion by a fourth route.

**What would change the verdict.** Not a better core construction — §5.4 shows
the rescue content is not what moves the number, and that a *random* fill is
already within +0.6 macro of the best available signal. The binding constraint
is the ratio between what the rescue content is worth and what the displaced
core slots cost. At |U| = 59 that ratio is **2.44 / 5.13 ≈ 0.48 for the union
and 1.85 / 5.13 ≈ 0.36 for random**; a winning formulation needs it above 1.
That is a much sharper bar than M6's, and it is the useful thing this stage
leaves behind: **before proposing another rescue source, check what fraction of
the displaced core it recovers, and check it against random.**

---

## Appendix — files

| file | what |
|---|---|
| `scripts/discovery/m7_common.py` | proxies, nomination, union, coverage metrics |
| `scripts/discovery/m7_phase1.py` | the budget audit |
| `scripts/discovery/m7_sets.py` | Phase-2 set construction, both panels |
| `scripts/discovery/m7_ext.py` | the 720-row confirmation panel (features + greedy order) |
| `scripts/discovery/m7_accuracy.py` | the generation harness with the explicit-set pruner |
| `scripts/discovery/m7_analyze.py` | regenerates every table above |
| `outputs/discovery/m7_phase1.json` | the budget audit grid |
| `outputs/discovery/m7_sets_bank.json`, `m7_sets_ext.json` | the delivered index sets |
| `outputs/discovery/m7_accuracy_bank.json`, `m7_accuracy_ext.json` | generation results + predictions |
| `outputs/discovery/m7_ext.npz` | the confirmation panel's features and greedy order |
| `outputs/discovery/m7_tables.md` | every table above, generated |

Reused unchanged: `m5_bank.npz` (the 27-column feature bank), `m6_eapd_order.npz`
(the incumbent's greedy order), `m2_gdep.py` (engine and B2 selector),
`instrumented.py` (selector swap), `m5_common.py` (`safe_features`, `bank_items`),
`m1_plan.json` (the extension panel).
