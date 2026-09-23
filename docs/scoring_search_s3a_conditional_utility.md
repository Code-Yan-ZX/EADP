# S3-A — Conditional utility: does a visual token's value depend on the retained set?

**Stage:** method-discovery, mechanism test (no new method, no new training).
**Date:** 2026-09-24. **Branch:** `m2-system-pareto`.
**Verdict:** *TBD — filled from the frozen analysis.*

---

## 0. The hypothesis under test

> visual token utility may not be a fixed unary score `u_i`; what matters may
> be the conditional marginal utility `u(i | S)` of token `i` given the
> already-retained set `S`.

This stage measures exactly one thing: **for the same candidate token `i`, does
its measured downstream utility change when the retained context `S` changes —
and does it change more on OCRBench than on TextVQA?** If the conditional
effect is weak, this route stops here; if strong and it explains the TextVQA /
OCRBench flip, it becomes the basis of the next design step. Nothing is
designed on top of the result inside this document.

Why this hypothesis now (frozen facts, all previously established):

| Fact | Where |
|---|---|
| The gradient teacher is a strong Top-256 scorer (macro 64.6 vs B1 61.1 vs B0 75.6 at budget 256) | S2-B |
| The student→teacher accuracy gap is front-loaded: swapping 8/16/32 missed teacher tokens recovers ≈53 % / 77 % / 97 % of it; the remove half of a swap is inert | S2-C2 |
| No single token is individually necessary for a rescue; the value of the missing tokens needs a small GROUP together (median block 4) — an existing fingerprint of set-dependence | S2-C2 |
| LOCAL-MLP(n960) beats every prior student on TextVQA (+12.9 over B1) but loses on OCRBench (−11.3) — the score is unary and task-inverted | M2 (+amendment) |
| L4 pruning costs 286 ms TTFT — too late; the eventual method must be pre-LLM / vision-side | M2 perf |
| A parameter-matched context-aware scorer never reads sample-specific context (wrong-image control) | S2-C5 |

S2-C5 said the *teacher's ranking* is learnable only as a token-local
nonlinearity. That is a statement about PREDICTING the teacher. The teacher's
ranking being nearly context-free does not by itself say whether the token's
TRUE downstream effect — measured causally on the answer — is context-free.
S3-A measures the latter directly.

## 1. Protocol (frozen before results)

All artifacts: scripts `Qwen_vl/scripts/discovery/s3a_*.py`, runner
`run_s3a.sh`; frozen inputs `outputs/discovery/s3a_cases.json`; measurements
`s3a_nll.json`, `s3a_rescue.json`; statistics `s3a_analysis.json`; gates
`s3a_gates.json`. Decision rules in `docs/s3a_prereg_rules.md`, fixed
2026-09-24 BEFORE the first Δ number was computed.

### 1.1 Regime: pre-LLM delivery

A retained set S (exactly 256 of the 1024 merged visual tokens) is realised by
splicing `vision_tower(image)[S]` (raster order) into the prompt between the
text prefix and suffix — precisely the sequence the incumbent's pruned path
hands the LLM. Dropped tokens never reach the LLM, so a difference
`u(i|S1) − u(i|S2)` can only come from the LLM computation over the kept set:
the conditional effect proper, measured in the deployment regime the final
method must live in. (The GDEP engine itself is only used once, in gate G1, to
prove that S_base below is the engine's real retained set.)

Reuse, not rebuild: instance bank = the frozen held-out 50-per-benchmark
(test-150 of `load_m1_plan()` ≡ the `m2_accuracy.json` key list, verified
identical); gradient teacher = the cached P1-G2 maps
(`s2b_gradient_scores.npz`); student = the GDEP C1 scorer (LOCAL-MLP
`n960_L4_s2`) re-applied to the published M1 L4 feature cache, whose top-256
is verified against the live engine's selection (G1).

### 1.2 Utility definition (answer-level, no hidden proxy)

```
L(y | S)  = mean teacher-forced per-token NLL of the gold answer(s) under S,
            golds deduplicated, capped at 6 per instance and 64 answer tokens
            (the gradient teacher's own cap).
add-marginal   Δ(i | S) = L(y | S) − L(y | S ⊕ {i} ⊖ {r})   budget stays 256
leave-marginal Δ(j | S) = L(y | S) − L(y | S ⊖ {j} ⊕ {f})   budget stays 256
```
`r` = the current context's lowest student-score token ("lowest value" under
the frozen student); `f` = a fixed neutral filler outside S. Positive Δ =
improves the answer. Every measurement records `r`'s identity, student score
and teacher rank, because S2-C2's "remove side is inert" is re-checked, not
assumed. Two controls quantify the intervention's arithmetic:
- `add_norem`: Δ with budget 257 (no removal) — is the fixed-budget version
  faithful to the free one?
- `ri`: for 6 candidates × 3 alternative removed tokens, Δ with a RANDOM
  removal instead of `r` — how much does remove-side identity move L?

**Numerical resolution.** bf16 single-forward CE carries an instance-dependent
noise floor (batched vs per-gold re-measurement of the same set): recorded
per instance as `noise|floor` (median ≈ 0.08 nats, up to 0.29). The prereg rule
counts a Δ as *resolved* only at ≥ 3× that instance's floor.

### 1.3 Contexts (all exactly 256 tokens, per instance)

Derived from S_base by a W=32 churn, refills never touching the candidate pool:

| ctx | construction | reading |
|---|---|---|
| base | S_base | the student's own set |
| weak | − the 32 S-members with the highest teacher score, + 32 neutral (outside S∪T, low-teacher) | backbone evidence removed |
| strong | − the 32 lowest-teacher S-members, + top-32 of T_only | moved toward the teacher |
| rand | −32 random members + 32 random outside tokens | churn-per-seed control |
| rand2 | rand with independent seeds | seed-robustness of the control |

base∩weak = base∩strong = 224/256 by construction.

### 1.4 Candidate pool (12 per instance, outside every context)

- `hi` ×4: teacher-set misses at T_only rank 33–96 (high gradient utility,
  clear of the strong refill);
- `mid` ×4: global teacher rank 257–600, outside S (medium utility);
- `lo` ×4: global teacher rank ≥ 600, outside S (low/random control).
Chosen by even spread inside each band under a fixed seed. Leave-candidates
`J` ×8: the S-members ranked 33–40 by teacher score inside S (they survive the
weak and strong drops, so they are probed under ≥ 3 genuinely different
contexts).

### 1.5 Subset (24 per benchmark, stratified on the frozen M2 outcomes)

`break` = GDEP wrong where B1 right (the OCRBench pain); `rescue` = GDEP right
where B1 wrong; `gw_b0` = GDEP wrong, B1 wrong, full model right; `bw` = all
wrong; `ok` = all right. Quotas TextVQA 1/7/1/8/7, DocVQA 9/4/7/4/0,
OCRBench 10/6/3/3/2 — all seats filled by their own stratum (no spill).

### 1.6 Gates (all PASS before any measurement; `s3a_gates.json`)

| gate | statement | result |
|---|---|---|
| S3A-G1 | recomputed student top-256 == live GDEP engine `select_idx` (n960 s2, topk, 256) | PASS 3/3 symdiff 0 |
| S3A-G2 | direct splice == S2-C2 indicator-map delivery through the incumbent pruner, embeddings max-abs-diff | PASS 0.0 (bit-exact) |
| S3A-G3 | teacher forcing reproduces greedy generation at every compared position | PASS 4/4, 3/3, 2/2 |
| S3A-G3b | batched-gold NLL == per-gold NLL within the bf16 floor | PASS (8.3e-2, tol 2.5e-1) |
| S3A-G4 | determinism | PASS (bit-identical) |

### 1.7 Measurement cost

~184 teacher-forced forwards + 1 generation per instance (5 refs + noise probe
+ 60 add + 60 add_norem + ~40 leave + 18 ri), 72 instances, single A40.

## 2. Part A — is conditionality real?

*TBD*

## 3. Part B — benchmark difference

*TBD*

## 4. Part C — direct (unary) vs conditional support

*TBD*

## 5. Part D — the rescue control

*TBD*

## 6. Cases

*TBD*

## 7. Verdict

*TBD — one of CONDITIONAL-UTILITY-SUPPORTED / WEAK·TASK-SPECIFIC /
CONDITIONAL-UTILITY-REFUTED, per the frozen rules.*

### Q1. Does token utility significantly depend on the retained set S?
*TBD*

### Q2. Is OCRBench more conditional-support-dependent than TextVQA?
*TBD*

### Q3. Does conditional marginal utility rescue GDEP failures better than the unary gradient score?
*TBD*

### Q4. Is the evidence sufficient to design a pre-LLM conditional-utility selector next?
*TBD*

---

## Appendix: reuse map

| component | reused from |
|---|---|
| held-out 150 bank / strata | `m1_common.load_m1_plan`, `m2_accuracy.json` |
| gradient teacher maps | `s2b_gradient_scores.npz` (P1-G2, S2-B) |
| student scorer | `m1_fixed-step_n960_L4_s2__H8.pt` + `m1_stats_n960.npz` + `s2c1_feats_L4.npy` (M1 cache), `m2_gdep.build_scorer` |
| set delivery through the incumbent path | S2-C2 indicator-map mechanism (G2 gate) |
| official per-instance scoring | `scoring.per_sample_hits` |
| teacher-forcing harness | `s2a_gradient_probe.py` A1 objective (NLL without the gradient) |
