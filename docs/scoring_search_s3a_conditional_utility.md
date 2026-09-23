# S3-A — Conditional utility: does a visual token's value depend on the retained set?

**Stage:** method-discovery, mechanism test (no new method, no new training).
**Date:** 2026-09-24. **Branch:** `m2-system-pareto`.
**Verdict: WEAK / TASK-SPECIFIC** — conditionality is real, large and
resolvable on ALL three benchmarks (not OCRBench-specific), but it is
chaotic rather than semantic, does not explain the TextVQA/OCRBench flip,
and does not convert into answer rescue. Per the brief's own stop rule,
the conditional-utility route has no demonstrated value beyond the unary
teacher. Details and the frozen decision rules below.

---

## 0. The hypothesis under test

> visual token utility may not be a fixed unary score `u_i`; what matters may
> be the conditional marginal utility `u(i | S)` of token `i` given the
> already-retained set `S`.

This stage measures exactly one thing: **for the same candidate token `i`, does
its measured downstream utility change when the retained context `S` changes —
and does it change more on OCRBench than on TextVQA?** Nothing is designed on
top of the result inside this document.

Why this hypothesis now (frozen facts, all previously established):

| Fact | Where |
|---|---|
| The gradient teacher is a near-oracle Top-256 scorer: its set, delivered pre-LLM at budget 256, scores 77.0 / 80.4 / 68.0 (macro 75.1) vs B1 61.1 and B0 75.6; a shuffled-identity control collapses it | S2-B/S2-C0 pilot |
| The student→teacher accuracy gap is front-loaded: swapping 8/16/32 missed teacher tokens recovers ≈53 % / 77 % / 97 % of it; the remove half of a swap is inert | S2-C2 |
| No single token is individually necessary for an accuracy rescue; value needs a small GROUP (median block 4) — an existing fingerprint of set-dependence | S2-C2 |
| GDEP (LOCAL-MLP n960) beats B1 on TextVQA (+12.9) and loses on OCRBench (−11.3) | M2 + amendment |
| L4 pruning costs 286 ms TTFT — too late; the eventual method must be pre-LLM / vision-side | M2 perf |
| A parameter-matched context-aware scorer never reads sample-specific context; the teacher's *predictable ranking* is a saturated token-local nonlinearity | S2-C5 |

S2-C5 bounded what is *learnable about the teacher*. It did not measure
whether a token's TRUE downstream effect is itself context-dependent.
S3-A measures the latter directly, causally, at the answer.

## 1. Protocol (frozen before results)

All artifacts: scripts `Qwen_vl/scripts/discovery/s3a_*.py`, runner
`run_s3a.sh`; frozen inputs `outputs/discovery/s3a_cases.json`; measurements
`s3a_nll.json`, `s3a_rescue.json`; statistics `s3a_analysis.json`; gates
`s3a_gates.json`; decision rules `docs/s3a_prereg_rules.md` fixed on
2026-09-24 **before the first Δ number was computed**.

### 1.1 Regime: pre-LLM delivery

A retained set S (exactly 256 of the 1024 merged visual tokens) is realised
by splicing `vision_tower(image)[S]` (raster order) into the prompt between
the text prefix and suffix — precisely the sequence the incumbent's pruned
path hands the LLM (proved bit-identical to the S2-C2 indicator-map
delivery, gate G2). Dropped tokens never enter the LLM, so a difference
`u(i|S1) − u(i|S2)` can only come from the LLM's computation over the kept
set: the conditional effect proper, in the deployment regime the final
method must live in. The GDEP engine is used exactly once, in gate G1, to
prove S_base is the engine's real retained set.

Reuse, not rebuild: instance bank = the frozen held-out 50-per-benchmark
(`load_m1_plan()` test-150 ≡ `m2_accuracy.json` keys, verified identical);
gradient teacher = cached P1-G2 maps (`s2b_gradient_scores.npz`); student =
the GDEP C1 scorer (LOCAL-MLP `n960_L4_s2`) re-applied to the published M1
L4 feature cache.

### 1.2 Utility definition (answer-level, no hidden proxy)

```
L(y | S)  = mean teacher-forced per-token NLL of the gold answer(s) under S;
            golds deduplicated, ≤ 6 per instance; ≤ 64 answer tokens (the
            gradient teacher's own cap).
add-marginal   Δ(i | S) = L(y | S) − L(y | S ⊕ {i} ⊖ {r})    budget 256
leave-marginal Δ(j | S) = L(y | S) − L(y | S ⊖ {j} ⊕ {f})    budget 256
```

`r` = the current context's lowest student-score token; `f` = a fixed neutral
filler. Positive Δ = improves the answer. Two fidelity controls ride along:
`add_norem` (budget 257, no removal) and `ri` (6 candidates × 3 RANDOM
removals instead of `r`) — S2-C2's "remove side is inert" is re-measured,
not assumed.

**Numerical resolution.** bf16 single-forward CE carries an
instance-dependent noise floor, re-measured per instance (batched vs
per-gold re-scoring of the same set): `noise|floor` ∈ [0, 0.29] nats, mean
0.084 TextVQA / 0.047 DocVQA / 0.008 OCRBench. The prereg rule counts a Δ
as *resolved* only at ≥ 3× the instance floor. This floor is the main
measurement caveat of the stage: on TextVQA half of all candidates sit
under it; on OCRBench almost none do (single short golds → tiny floors).

### 1.3 Contexts (all exactly 256 tokens, per instance; W = 32 churn)

| ctx | construction | reading |
|---|---|---|
| base | S_base | the student's own set |
| weak | − 32 highest-teacher S-members, + 32 neutral (outside S∪T) | backbone evidence removed |
| strong | − 32 lowest-teacher S-members, + top-32 of T_only | moved toward the teacher |
| rand | −32 random members, +32 random outside tokens | churn-per-se control |
| rand2 | rand, independent draw | seed-robustness of the control |

base∩weak = base∩strong = 224/256 by construction.

### 1.4 Candidate pools (per instance, outside every context)

Add-pool P (12): `hi` ×4 = T_only ranks 33–96 (high gradient utility, clear
of the strong refill); `mid` ×4 = global teacher rank 257–600 outside S;
`lo` ×4 = teacher rank ≥ 600 outside S; even spread inside each band, fixed
seed. Leave-pool J (8): S-members ranked 33–40 by teacher score within S
(survive the weak and strong drops, probed under ≥ 3 genuinely different
contexts).

### 1.5 Subset (24 per benchmark, stratified on the frozen M2 outcomes)

`break` = GDEP wrong where B1 right; `rescue` = GDEP right where B1 wrong;
`gw_b0` = both wrong, full model right; `bw` = all wrong; `ok` = all right.
Quotas TextVQA 1/7/1/8/7, DocVQA 9/4/7/4/0, OCRBench 10/6/3/3/2 — every
seat filled by its own stratum (zero spill). In the injection regime the
S_base generation scores (mean per-instance hit) 0.575 / 0.402 / 0.333 on
the deliberately failure-enriched subset; its right/wrong verdict agrees
with the GDEP run's on 84.7 % of instances (the regimes differ by
construction — the injected path never runs L0–L4 over dropped tokens).

### 1.6 Gates (all PASS before any measurement; `s3a_gates.json`)

| gate | statement | result |
|---|---|---|
| S3A-G1 | recomputed student top-256 == live GDEP engine `select_idx` (n960 s2, topk@256) | PASS, 3/3 symdiff 0 |
| S3A-G2 | direct splice == S2-C2 indicator-map delivery through the incumbent pruner | PASS, embedding max-abs-diff = 0.0 (bit-exact) |
| S3A-G3 | teacher forcing reproduces greedy generation at every compared position | PASS, 4/4, 3/3, 2/2 argmax |
| S3A-G3b | batched-gold NLL == per-gold NLL within the bf16 floor | PASS, 8.3e-2 (tol 2.5e-1) |
| S3A-G4 | determinism | PASS, bit-identical |

### 1.7 Cost

~184 teacher-forced forwards + 1 generation per instance × 72 instances =
31 min; rescue (Part D) 30 min; both on one A40.

## 2. Part A — is conditionality real?  **Yes, and it is large.**

Per instance we get a 12-candidate × 5-context matrix
`M[i,S] = L(y|S) − L(y|S⊕i⊖r)`. Each entry already subtracts its own
context's reference, so a context-level shift cannot leak into the
comparison; the two-way decomposition `M = a_i + b_S + c_{iS}` isolates the
token×context interaction `c_{iS}` — the conditional effect proper.

| statistic (resolved candidates only) | TextVQA | DocVQA | OCRBench |
|---|---|---|---|
| instances with ≥4 resolved candidates | 14/24 | 21/24 | 23/24 |
| interaction share of Var(Δ) | **0.449** [0.361, 0.533] | **0.395** [0.303, 0.491] | **0.435** [0.362, 0.511] |
| within-token Δ range (nats) | 0.288 | 0.539 | 0.238 |
| ÷ 3×noise floor | 1.1× | 3.8× | 9.9× |
| range ÷ mean\|Δ(base)\| | **4.60** | **3.12** | **2.39** |
| Kendall rank-reversal base↔weak | 0.37 | 0.37 | 0.46 |
| Kendall base↔strong | 0.46 | 0.43 | 0.51 |
| Spearman ρ base↔weak | 0.32 [0.13, 0.51] | 0.33 [0.20, 0.45] | 0.10 [−0.03, 0.23] |
| Spearman ρ base↔strong | 0.08 | 0.21 | −0.04 |
| top-3 Jaccard base↔weak | 0.36 | 0.33 | 0.21 |
| leave-marginal interaction share (held tokens) | 0.453 | 0.434 | 0.538 |

The same token's marginal utility swings, across retained sets of identical
size, by 2.4–4.6× the utility's own magnitude; roughly 40 % of the variance
of Δ is pure interaction; the "best candidate" list is almost a different
list in a different set (top-3 overlap 0.21–0.36); the ranking of tokens
*inside* the set moves just as much (leave-share 0.43–0.54). Resolved
sign-flips — a token measurably helps under one retained set and measurably
hurts under another, ≥3× floor both sides — occur **505 times** (TextVQA
78 / DocVQA 162 / OCRBench 265 out of 576 possible candidate×context tests
per benchmark: 13.5 % / 28.1 % / **46.0 %**).

**But the same thing happens under meaningless churn.** ρ(rand, rand2) —
two independently randomized churns, no semantic content at all — is
0.25 / 0.13 / 0.09, statistically indistinguishable from the semantic
perturbations above. What Part A establishes is that utility depends on S
*strongly and chaotically*: swapping 32 essentially arbitrary tokens is as
destabilizing as swapping the teacher's best/worst 32. There is, in these
data, no additional *semantic* structure in how the ranking moves —
`weak` is not systematically more disruptive than `rand`.

Controls, honestly reported:
* **Remove-side identity is NOT inert on OCRBench.** The `ri` arm moves L
  by ~0.10 nats on every benchmark; relative to each benchmark's floor that
  is 0.9× (TextVQA, indistinguishable), 2.2× (DocVQA), **12.1× (OCRBench)**.
  At OCRBench resolution, *which* token you drop to keep the budget matters
  about as much as the small add-effects themselves. S2-C2's "removal is
  inert" survives in effect size but not at OCRBench's resolution.
  (Δ vs removed-token teacher rank: ρ = −0.15, weak but nonzero.)
* **Fixed budget ≠ free budget.** add vs add_norem differ by 0.31 mean nats
  — larger than the utility scale itself. Swap-based conditional utility is
  a different quantity from inclusion-based utility; both are reported
  separately and the fixed-budget one is the deployment-relevant one.
* The bulk of marginals is ZERO: across all 480 candidate×context deltas per
  benchmark the median is 0.000 and P(Δ>0) ∈ [0.45, 0.55] for every band.
  Conditionality lives in a heavy tail, not in the typical token
  (resolved-positive counts per benchmark, hi/mid/lo: TVQA 35/21/19,
  DocVQA 22/28/18, OC 9/7/5 out of 160 each).

## 3. Part B — benchmark difference: **no. OCRBench is NOT more conditional.**

The motivating story was "conditional/support-token effects might explain
GDEP's +12.9 TextVQA / −11.3 OCRBench flip". The measured contrast says no:

| contrast (per-instance bootstrap CI) | value |
|---|---|
| interaction share, TextVQA − OCRBench | **+0.014 [−0.097, +0.127]** |
| within-token Δ range, TextVQA − OCRBench | +0.144 [−0.347, +0.829] |
| interaction share, DocVQA − OCRBench | −0.039 [−0.157, +0.078] |

Conditionality is uniform across benchmarks in share and rank-instability
(in fact OCRBench's range/|Δ| ratio is the *lowest*, 2.4). What does differ:
OCRBench deltas are the most *resolvable* (floor 0.008 → most sign flips,
most resolved entries) and its remove-side is the least inert. Neither of
those is "more support dependence"; they are measurement-resolution effects
of short single-gold instances. The TextVQA↔OCRBench inversion is **not** a
conditionality-asymmetry phenomenon at this granularity.

## 4. Part C — unary (direct) score vs conditional utility

Rank correlation between the teacher's unary ordering and the measured
marginal at the student's own set: **ρ(unary, Δ(·|base)) = 0.050**
(ρ(·|weak) = 0.051). The P1-G2 score, whose deployment success is
tremendous at the SET level (macro 75.1), carries essentially **no rank
information about the answer-NLL marginal of swapping one of its tokens in**
at budget 256. This is fully consistent with S2-A's "largely
answer-agnostic" — saliency toward the model's own next token is not
support for the gold answer — and it means the two utilities are close to
independent quantities.

Do "low-direct-utility / high-conditional-support" tokens exist? Yes:
defined as teacher rank ≥ 257 (outside the teacher's own top-256) yet
beating every high-band candidate's best marginal under some context, they
are **9.4 % / 10.1 % / 11.1 %** of the candidate pool on TextVQA / DocVQA /
OCRBench (27/29/32 of 288). Their spatial signature: mean grid distance to
the strongly-evidenced retained region is 4.1–4.3 cells for these tokens vs
3.0–3.7 for the hi band — i.e. slightly *farther* from the evidence
backbone, with no benchmark-specific clustering (no OCRBench excess). The
category is real but thin (~1 in 10 candidates), not spatially coherent,
and — decisively for the route — none of them is findable in advance: their
conditional value is discoverable only by measuring the intervention.

## 5. Part D — the rescue control: **conditional utility does NOT convert.**

23 base-wrong *and* rescuable instances (TextVQA 2, DocVQA 9, OCRBench 12;
B1 or B0 right, injection-regime base wrong). Four arms, one identical
sequential fixed-budget rule (add one, drop the student-min), only the pick
differs: `unary` = best remaining teacher score; `cond` = greedy on the
MEASURED marginal Δ(i|S_cur) — given the gold-answer loss itself, an oracle
advantage no other arm has; `spatial` = farthest grid point; `random`.

Mean ΔL = L(base) − L(arm set), and answers rescued (official metric):

| | TextVQA (n=2) | DocVQA (n=9) | OCRBench (n=12) |
|---|---|---|---|
| unary k8 | +0.133 / 0 | +0.116 / 1 | −0.081 / 0 |
| cond k8 | +0.128 / 0 | +0.135 / 1 | **+0.190 / 0** |
| spatial k8 | −0.107 / 0 | +0.171 / 1 | −0.024 / 0 |
| random k8 | +0.069 / 0 | −0.120 / 1 | +0.019 / 1 |
| unary k16 | −0.190 / 0 | +0.064 / 0 | **+1.038 / 1** |
| cond k16 | +0.051 / 0 | +0.023 / 0 | +0.111 / 0 |
| spatial k16 | +0.019 / 0 | −0.041 / 1 | −0.001 / 0 |
| random k16 | −0.131 / 0 | **+0.005 / 2** | +0.125 / 0 |

Paired cond−unary per instance: k8 dL = −0.005 [−0.104,+0.095] (TVQA,
n=2, uninformative), +0.020 [−0.199,+0.209] (DocVQA),
**+0.271 [+0.077,+0.548]** (OCRBench); rescue difference 0 at every cell
and k except OCRBench k16: **−0.083 [−0.25, 0.00] — conditional greedy
rescues strictly fewer answers than the unary order there** (unary's one
big rescue plus a +1.04 nats mean, driven by a single instance).

The trajectory analysis (per-step gains telescope exactly to ΔL) is the
cleanest single fact of this section: **greedy conditional utility is one
token deep.** On all 23 instances the first swap has a large positive gain
(mean +1.71 … +1.90 nats, all 23 > 0), and on average exactly **1.0**
step per instance clears the 3×-floor bar; steps 2–16 are negative or zero
at every benchmark (means −0.36 … −0.01). The oracle-stopped dL (k\*=1 for
23/23) is +1.7 to +1.9 nats — yet at that stopping point nothing was
rescued either (accuracy conversion of the best single swap needs more
than one swap, which S2-C2 already showed: rescue needs groups of ~4, and
S2-C2's groups came from the TEACHER ORDER, which is exactly as good here:
the greedy first pick lies inside the unary top-8 in 12/23 instances, and
never equals its first token, but the two arms' k=8 rescues tie at
0–1).

Per the brief's stop rule — *if conditional tokens do not clearly beat
unary gradient or spatial diversity, this route's value is very limited* —
that is what happened: on answers, cond never beats unary (tie or loss);
on loss it wins only on OCRBench at k=8 by +0.27 nats with zero answer
consequence; random matches it more often than not at these n's.

## 6. Cases

The extremes (all ≥3× floor):

| instance | stratum | token (teacher rank) | Δ(base) | Δ(other ctx) | reading |
|---|---|---|---|---|---|
| DocVQA_VAL_1723 | bw | 98 (80) | −0.247 | **+8.785** (strong) | harmful alone, essential once the teacher's top-32 are retained — a 9-nat sign+scale flip |
| DocVQA_VAL_3015 | rescue | 12 (114) | +2.996 | +0.026 (weak) | its entire value was carried by the removed evidence backbone — a pure support token |
| DocVQA_VAL_2692 | break | 83 (74) | +1.337 | −1.776 (rand2) | +1.3 to −1.8 from random churn alone |
| TextVQA_VAL_1711 | bw | 429 (83) | −1.081 | −4.155 (rand) | both harmful, magnitude 4× by context |
| OCRBench_624 | break | 642 (72) | +0.901 | −0.717 (weak) | high-unary token flips sign when backbone removed ("Inspired" vs base "Lonely") |
| OCRBench_543 | break | 151 (106) | −1.583 | +0.000 (strong) | harmful alone, neutral with the teacher's best |
| OCRBench_624 | break | 550 (677) | +0.189 | −1.615 (weak) | LOW-unary, helps at base, hurts without backbone |

Context main effects are the other half of the story and they dwarf the
token effects: the `strong` context (adding the teacher's top-32 misses)
drops L to **0.000** on OCRBench_543 and _624 — the gold answer becomes
certain — while the student base set sits at 5.98 / 11.54 nats. Set
composition decides the answer; individual swap margins are almost always
zero and when they aren't, they depend on the set. That is the exact
statement of conditional utility — and the reason a token-level
`u(i|S)`-scored selector has nothing to grab: the informative variance is
in b_S (which 256 together), not in the marginal ranking of any 256+1.

## 7. Verdict and answers

Against the frozen rules (`docs/s3a_prereg_rules.md`):

* **Rule 1 (REFUTED) does not trigger** — conditionality clears the noise
  bar decisively on every benchmark (interaction share CIs ≥ 0.30;
  ranges 1.1–9.9× the 3×-floor threshold; 505 resolved sign flips).
* **Rule 2 (SUPPORTED) fails its conversion clause (c)** — the paired
  rescue test cond−unary is 0 everywhere and negative in its only
  resolved accuracy cell; the loss-only OCRBench k8 win (+0.27 nats,
  CI>0) has no answer consequence and is beaten by unary at k16.
* **Rule 3 applies: WEAK / TASK-SPECIFIC** — the frozen wording's second
  branch, "visible in NLL but not converting into rescue wins", is exactly
  the observed shape; and the "task-specific" half of the label is
  *absent*: the conditional effect is uniform across the three benchmarks
  (Part B CIs all straddle 0), so there is not even an OCRBench niche to
  retreat to.

### Q1. Does token utility significantly depend on the retained set S?
**Yes.** ~40 % of Var(Δ) is token×context interaction (CI lower bounds
0.30–0.36 on all three benchmarks); the same token's marginal swings by
2.4–4.6× the utility's own scale across equal-size sets; 46 % of OCRBench
candidate×context tests are resolved sign flips; the best-token list has
top-3 Jaccard 0.15–0.36 across contexts. The dependency is real — but it
is chaotic: two independent RANDOM churns destabilize the ranking as much
as semantically-targeted ones.

### Q2. Is OCRBench more conditional-support-dependent than TextVQA?
**No.** Interaction share difference +0.014 [−0.097, +0.127]; range
difference CI straddles 0; OCRBench's apparent excesses (most sign flips,
most resolved entries) are resolution artifacts of its ~10× smaller noise
floor, and it has the *lowest* range/|Δ| ratio (2.4). The
TextVQA(+12.9)/OCRBench(−11.3) inversion is not a conditionality-asymmetry
phenomenon at token granularity; Part C's low-unary/high-conditional class
likewise shows no OCRBench concentration (9.4/10.1/11.1 %).

### Q3. Does conditional marginal utility rescue GDEP failures better than the unary gradient score?
**No.** On answer rescue: ties at 0–1 rescues everywhere, and a strict
loss (−0.083 [−0.25,0]) at OCRBench k16, despite the conditional arm
optimizing the gold loss directly. Its convertible content is exactly one
swap deep (step-1 gain +1.8 nats, all 23>0; k\*=1 at 23/23; every
subsequent step ≈0 or negative), and even that first swap converts to
nothing — rescue needs groups of ~4 (S2-C2) and the teacher order already
supplies comparable groups (cond's first pick ∈ unary top-8: 12/23).

### Q4. Is the evidence sufficient to design a pre-LLM conditional-utility selector next?
**No — stop this route.** The selector would have to (i) predict a
quantity that is nearly uncorrelated with every unary score available
(ρ=0.05), (ii) whose informative variance is chaotic under arbitrary
32-token perturbations, (iii) whose recoverable part (support of a single
swap) does not survive budget-preserving continuation and (iv) does not
convert to accuracy any better than the incumbent teacher ordering. This
sits alongside S2-C5 (a trained context-aware scorer never reads the
context) and S2-C0/C1 (forward proxies keep 5–45 % of the teacher, none
predictably). What S3-A ADDS to that picture is a positive structural
fact worth reusing later: the answer is decided at the level of *which 256
tokens together* (context main effects of 5–12 nats, `strong` → L=0.000 on
OCRBench break cases), and the first fixed-budget swap matters while the
2nd–16th are dead — i.e. the gap between student and teacher is not an
ordering problem at token granularity that any u(i|S) score can mine;
any successor must change how SETS are chosen, not sharpen per-token
numbers. Per the brief, no S3-B method design is started here.

---

### Threats to validity (self-reported)

* n=23 rescue instances (2 TextVQA) — rescue-rate CIs are wide; the k16
  OCRBench unary advantage (+1.04 dL) rides on one instance; the
  conclusion rests on the paired trajectory result (k\*=1 at 23/23), not
  on rates alone.
* Gold-answer NLL ≠ official accuracy (ANLS/VQA-soft/OCBench rules);
  Part D closes that gap with the real metric at fixed k and finds no
  conversion.
* The 12-candidate pool per benchmark-band is small (even-spread by
  design); P(Δ>0)≈0.5 medians=0 say the pool is mostly typical tokens —
  the heavy tail (e.g. +8.8 nats) is where all signal lives and it is
  under-sampled by construction. A denser hi-band pool could add more
  first-swap examples; nothing in Part A suggests it would change the
  step-2-onwards collapse.
* The swap-removal rule (student-min) is one natural "lowest-value"
  definition; the `ri` arm shows its identity is resolvable on OCRBench,
  so per-token conclusions there carry that caveat (quantified, not
  assumed away).
* bf16 resolution floor (per-instance, up to 0.29 nats) makes TextVQA
  half-unresolved by the 3× rule; its statistics are reported
  resolved-only with n=14 instances and flagged throughout.

---

## Appendix: reuse map

| component | reused from |
|---|---|
| held-out 150 bank / strata | `m1_common.load_m1_plan`, `m2_accuracy.json` |
| gradient teacher maps | `s2b_gradient_scores.npz` (P1-G2, S2-B) |
| student scorer | `m1_fixed-step_n960_L4_s2__H8.pt` + `m1_stats_n960.npz` + `s2c1_feats_L4.npy`, via `m2_gdep.build_scorer` |
| set delivery through the incumbent path | S2-C2 indicator-map mechanism (G2 gate) |
| official per-instance scoring | `scoring.per_sample_hits` |
| teacher-forcing harness | `s2a_gradient_probe.py` A1 objective (NLL without the gradient) |
| greedy-decode equivalence | `diag_selectors.generate_prediction` conventions |
