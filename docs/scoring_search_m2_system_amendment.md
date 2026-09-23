# M2 — implementation-defect amendment (frozen-RoPE PRESERVE decode)

**What this document is.** An amendment to the M2 pre-registration
(`scoring_search_m2_system_prereg.md`) and its interim audit
(`scoring_search_m2_system_interim.md`, commit 6a74c0b). It does **not**
rewrite the pre-registration: the original text stands, frozen, and this file
records (a) the implementation defect that invalidates part of the grid the
prereg produced, (b) the correction of that defect and the engine-versioning
that keeps the two generations of artefacts apart, (c) the replacement
measurement protocol, and (d) a decision rule set **frozen before the
corrected measurement** (§5 below, frozen 2026-09-23 21:15 local, before the
fix gates ran and before any corrected arm executed).

**The finding, in one line.** `GDEPEngine.decode` recomputed the PRESERVE
branch's RoPE position from the *frozen prefill tensor* on every step
(`Qwen_vl/scripts/discovery/m2_gdep.py:553` pre-fix:
`nxt = (int(pos1.max().item()) + 1) if preserve else cur_len`), so every
generated token was issued the same position; `cache_position` advanced
correctly and the RENUMBER branch was never touched by it. The interim audit
D2/D5 localised the defect and isolated its cost on three instances.

---

## 1. Invalidation scope (directive 一)

The pre-registered accuracy grid was stopped on 2026-09-23 ~21:05 local, mid
way through `C2|s2`, after archiving its artefacts
(`outputs/discovery/m2_pre_stop_snapshot/`, and the final grid log as
`m2_run_progress_attempt6_stopped.log`). No contaminated result was deleted;
each is annotated in place in `m2_accuracy.json` with

```
"validity": "INVALID_DUE_TO_FROZEN_ROPE_POSITION"
```

| run | status |
|---|---|
| `B0`, `B1`, `B2` | **valid** — never take the PRESERVE decode branch; B1 reproduces the published official-facility numbers 150/150 hit-for-hit (gate G-F re-pass at 21:1x) |
| `C1-R|s0..s2` | **valid** — RENUMBER decode advances via `cache_position`; ran after the stop, 21:06–21:14 (§2) |
| `C0|s0..s2`, `C1|s0..s2`, `C2|s0..s1` | **INVALID** — PRESERVE arms of the defective engine; retained in the artefact, excluded from every macro, CI, Pareto reading and verdict by `m2_consolidate.valid()` and `m2_amend_report.valid()` |
| `C2|s2` (partial), `C3|s*`, `C1-SHUF|s*` | **never completed** — further measurement of the contaminated configuration has no scientific value and was stopped |

The invalid arms' numbers (~33–40 macro, 13–17 % capped generations) remain
in the file as evidence of the defect's size, not of the method.

## 2. The valid arm first: C1-R (directive 二)

C1-R (GDEP n=960 topk@256, RENUMBER) ran immediately after the stop, on the
pre-fix code — which is legitimate for this arm because the fix is provably
inert on the RENUMBER branch (§3), and gate HG-8 (§4) re-verifies v1≡v2
bit-identically: scores, selected token sets and 24-step force-decoded RENUMBER
token ids are identical between the git-v1 engine and the fixed v2 engine.
C1-R's `cfg_key` carries no `ev` segment: it is recorded as a v1-built
artefact.

Results: §6 below (full table in `outputs/discovery/m2_amend_report.md`).

## 3. The fix (directive 三)

`GDEPEngine.decode`, PRESERVE branch only:

* the first generated token takes `max(prefill position_ids) + 1`;
* every later decode step takes **one more than the step before** (a running
  `next_pos`, incremented inside the loop);
* the original token-position gaps of the compacted prefill are untouched —
  the fix is entirely in the decode continuation;
* `cache_position` keeps following the compacted cache length, as before;
* RENUMBER and the non-gdep modes are byte-identical to v1 (the `else` arm of
  the same line).

Engine versioning: `m2_gdep.ENGINE_VERSION = "gdep_preserve_v2_advancing_rope"`
is embedded in `GDEPConfig.key()` **for gdep-mode arms** (full/prellm keys and
hashes are unchanged so the stored baselines stay reproducible). A corrected
PRESERVE run therefore has both a new arm key (`C1-P`) and a new `cfg_hash`;
pre/post-fix PRESERVE artefacts cannot overwrite or mix with each other.

A second, independent timing defect was fixed in the same pass (directive 六
required its explanation): in `prefill()`'s MODE_PRELLM branch the pruner call
was bracketed by an outer CUDA event *and* its internal CudaTimer wrote
`selector_ms`, and `_finish()` then **added** the bracket to the assignment —
double-counting the selector window. This is the whole of the interim report's
"selector cost is load-unstable on this host: 85.8 ms (B1) vs 41.4 ms (C3)"
observation: `m2_perf.json` shows B1's reported 85.85 ms = internal
44.76 ms + outer bracket ≈ 41 ms, i.e. the true facility cost **is** ~41–45 ms
and agrees with C3; and it is also why B1's "model prefill median" (352.7 ms,
the event sum) exceeded its own wall-clock TTFT median (313.3 ms). The outer
bracket was removed; the pruner-internal windows are now the single selector
window for prellm arms. (TTFT figures themselves — wall-clock — were never
affected; only the event decomposition and the "load drift" interpretation
were.)

## 4. Post-fix correctness gates (directive 四)

`scripts/discovery/m2_fixgates.py` → `outputs/discovery/m2_fixgates.json`.
Nine checks, all required to pass before any corrected number is used:

| gate | requirement | result (2026-09-23 21:3x, `m2_fixgates.json`, all_passed=**True**) |
|---|---|---|
| HG-1 | K=1024 **multi-step** (24 forced steps, past EOS) generation token-for-token identical to stock HF greedy, on the `full` path and the gdep K=1024 identity path, 8 held-out instances | **PASS 8/8 exact.** (First run FAILed 0/8 on a *gate* bug: `min_new_tokens` installs a logits processor that MASKS the real EOS, so the forced reference diverged from the engine exactly where EOS is greedy — step 4 of `dakota`. With the forcing convention fixed to `eos_token_id=-1` the two paths are identical. Recorded so the reader knows why.) |
| HG-2 | corrected PRESERVE: decode positions start at `max+1`, advance +1 per step, all distinct | **PASS** 6/6, e.g. rope 1055..1078 (24/24 distinct) |
| HG-3 | RENUMBER: decode positions advance contiguously (fix did not disturb the control) | **PASS** 6/6, e.g. rope 287..310 |
| HG-4 | 64 force-decoded steps on the D5 trio: 64 distinct positions — no frozen position survives | **PASS** (v1 would show 1 distinct position; observed 1054..1117 etc.) |
| HG-5 | D5 trio under corrected PRESERVE reproduces the audit's P′ token ids exactly | **PASS** (prefix compare — D5 recorded only `first_ids` = ids[:12]; corrected engine reproduces `preventent` / `FEB 1, 1972` / `(03) 341 566…`) |
| HG-6 | every previously-capped instance (≥20 required; union of the invalid records) re-decoded corrected: ≥90 % terminate by EOS | **PASS** 50 distinct instances (196 invalid runs), **48/50 = 96 %** terminate by EOS. The 2 exceptions are position-independent: `OCRBench_925` caps identically under corrected **and** RENUMBER on scorer seeds 1/2 and not on seed 0 (`C1-P` and `C1-R` agree), and `OCRBench_80` (a `c c c…` loop, re-run seeds only) caps for the n=240 scorer sets while the corrected n=960 set answers in 4 tokens — both track the scorer's token set, not the position policy |
| HG-7 | EOS rate on that set vs the baselines' 100 % | **PASS** 96 % vs baseline 100 % |
| HG-8 | v1 (git 6a74c0b) vs v2: prefill scores **bit-identical**, `select_idx`/`keep_full` exactly equal (8 instances), RENUMBER decode ids identical (4 instances, 24 forced steps) — the decode fix changes no prefill quantity | **PASS** all bit/exact — this is what licenses keeping the existing C1-R records as fixed-engine measurements |
| HG-9 | no extra forward/recompute: `layer_calls == 36 × (1 + decode_steps)` exactly, and prefill still 36 layer calls | **PASS** 900/900 on every HG-2/3 instance; asserted every paired-perf block |

The original G-A…G-E gates are additionally re-run unchanged against the v2
engine (`m2_correctness_v2.json`) alongside the paired benchmark, so every
corrected number is produced by a fully gate-passed binary.

## 5. Minimal corrected grid and the frozen decision (directive 五)

Phase 1 runs exactly two arms, 150 held-out instances, greedy, seeds 0/1/2:

* **C1-P** — corrected PRESERVE (n=960, topk, T=256), the P′ of the audit;
* **C1-R** — the existing valid arm (§2), not re-run.

C0, C2, C3 and C1-SHUF are **not** run in phase 1.

The decision below was frozen with the directive (2026-09-23 ~20:50 local),
before any corrected measurement existed. Macro = mean over the three
benchmarks of hit-rate; B1 = official EADP facility@256 pre-LLM (61.10);
"the cached-proxy baseline" is the S2-B/S2-C1 translation of the gradient
teacher into this engine's regime (~57 macro expected at T=256, interim §6;
the S2-B measurement itself: 78.81 vs 66.88 official on the 450 bank,
+11.92 pts, CI [+8.09,+15.76]).

| | condition | consequence |
|---|---|---|
| **A** | mean macro of P′ **or** R ≥ B1 − 1.0 | that position policy enters the selector comparison; continue block8/facility |
| **B** | both < B1 − 1.0, but either ≥ 1.0 pt above the cached-proxy baseline | early-pruning graph/objective mismatch; do **not** run the selector grid |
| **C** | both far below B1 (and not above the proxy) | stop the GDEP L4 method; no SHUF/selector sweep |
| **D** | \|P′ − R\| > 1.0 pt | the position policy is a method design variable: analysed as its own object; the higher of the two is never reported as the sole result |

## 6. Results (corrected grid, 2026-09-23 21:35–21:5x)

Held-out 150, greedy, seeds 0/1/2; full table + CIs in
`outputs/discovery/m2_amend_report.md`.

**C1-R (valid, v1 build, fix proven inert on it by HG-8):**

| seed | TextVQA | DocVQA | OCRBench | macro | Δ vs B1 [CI] | EOS | capped |
|---|---|---|---|---|---|---|---|
| 0 | 73.40 | 57.58 | 48.00 | 59.66 | −1.44 [−9.02, +5.96] | 100.0 % | 0 |
| 1 | 75.40 | 58.52 | 50.00 | 61.31 | +0.21 [−7.70, +7.76] | 99.3 % | 1 |
| 2 | 65.40 | 67.76 | 52.00 | 61.72 | +0.62 [−6.65, +7.76] | 98.7 % | 2 |

seed mean **60.90** (59.66–61.72); vs B1 seed-avg **−0.20 pts [−7.36, +6.71]**;
rescued 19 / newly broken 21.

**C1-P (corrected PRESERVE, v2 build, cfg carries the engine version):**

| seed | TextVQA | DocVQA | OCRBench | macro | Δ vs B1 [CI] | EOS | capped |
|---|---|---|---|---|---|---|---|
| 0 | 70.00 | 54.88 | 48.00 | 57.63 | −3.47 [−10.94, +3.98] | 100.0 % | 0 |
| 1 | 75.40 | 57.46 | 52.00 | 61.62 | +0.52 [−7.23, +8.06] | 98.7 % | 2 |
| 2 | 71.40 | 64.99 | 52.00 | 62.80 | +1.70 [−5.96, +9.07] | 99.3 % | 1 |

seed mean **60.68** (57.63–62.80); vs B1 seed-avg **−0.42 pts [−7.54, +6.51]**;
rescued 17 / newly broken 22. The whole 23-point interim gap was the defect:
on the identical token sets (same seed ⇒ same scores ⇒ same Top-256), the
corrected engine moves C1 from ~38 to 60.7.

**Head-to-head P′ − R: −0.21 pts [−2.57, +2.02]** (rescued 3, newly broken 6).

**Frozen decision reading (amendment §5):**

* **A fires for BOTH policies**: P′ 60.68 ≥ 60.10 = B1 − 1.0, and R 60.90 ≥
  60.10. Both position policies enter the selector comparison (block8 /
  facility) as the next phase.
* **D does not fire**: |P′ − R| = 0.21 < 1.0. At the resolution of this
  measurement the position policy is NOT a method design variable; both arms
  are reported, neither is chosen post hoc as "the" result.
* B/C not applicable (neither arm trails B1 by more than 1.0).
* Honest note: P′ 60.68 clears the 1.0-point bar by only 0.42 with a per-seed
  range of 5.2 points — the margin is within one seed's swing. Reading A is
  the preregistered reading, but it is marginal for P′ and comfortable for R.
* Paired performance: §7 — GDEP is **−22.5 ms (P′) / −23.0 ms (R) vs B1 TTFT**
  paired, but **+10 ms behind B2**; accuracy is statistically level with the
  incumbent, not above it.
* Gates G-A…G-E re-run on the v2 engine: `m2_correctness_v2.json`,
  **all_passed=True** (2026-09-23 22:0x).

## 7. Paired performance protocol (directive 六)

`scripts/discovery/m2_perf_paired.py` → `m2_perf_paired.json`:

* arms B0, B1, B2, C1-P, C1-R; 15 warm-up + ≥100 measured **blocks**; each
  block measures every arm once on the same fixed input inside one process,
  arm order randomised per block (seed recorded);
* windows: **TTFT** = sync→(image preprocess + vision tower + scoring/
  selection + LLM prefill + first-token logits)→sync, wall clock;
  **model-only prefill** = TTFT minus the image-preprocess window (the part
  pruning can act on); **selector** = the arm's selection machinery, each
  stage counted exactly once after §3's timing fix; **decode** = fixed 32
  steps, `ignore_eos`, CUDA-event;
* saved raw per block; reported per contrast: within-block median difference
  and block-bootstrap 95 % CI, plus same-arm first-half/second-half drift so
  the old "load instability" claim can be tested directly;
* `no_recompute` (layer-call accounting) asserted every block — the fixed
  decode path must add no forward pass.

**Result (120/120 blocks retained, seed 20260923, `no_recompute` True in every
block of every arm):**

| arm | TTFT med (ms) | p10–p90 | model-only (ms) | preprocessing (ms) | vision (ms) | prune machinery (ms) | LLM prefill (ms) | decode 32t (ms) |
|---|---|---|---|---|---|---|---|---|
| B0 | 397.9 | 396.2–404.7 | 312.3 | 82.6 | 110.2 | — | 202.1 | 1276.6 |
| B1 | 310.1 | 308.5–318.5 | 224.2 | 82.4 | 110.1 | 1.4 + 40.8 (scoring + facility) | 71.9 | 1278.9 |
| B2 | 278.6 | 276.3–285.0 | 192.2 | 82.4 | 110.2 | 1.4 + 8.5 (scoring + block8) | 72.0 | 1280.0 |
| C1-P | 288.3 | 285.9–294.5 | 202.1 | 82.8 | 110.3 | 28.0 + 0.3 + 0.1 (L0–L4 + scorer + compaction, sel 0.07) | 63.4 | 1281.7 |
| C1-R | 287.8 | 285.7–293.9 | 201.8 | 82.6 | 110.2 | as C1-P | 63.3 | 1280.3 |

Paired within-block contrasts (median Δ ms, block-bootstrap 95 % CI of the
mean, block win-count):

| contrast | TTFT Δ | CI | blocks | decode32 Δ |
|---|---|---|---|---|
| C1-P − B1 | **−22.50** | [−23.16, −20.79] | P′ wins 118 : 2 | +1.51 |
| C1-R − B1 | **−22.95** | [−23.77, −20.95] | R wins 118 : 2 | +1.65 |
| C1-P − B2 | **+9.98** | [+7.61, +11.26] | B2 wins 115 : 5 | +1.02 |
| C1-R − B2 | **+9.35** | [+7.17, +11.04] | B2 wins 112 : 8 | +0.83 |
| B1 − B2 | +31.89 | [+29.68, +33.04] | B2 wins 119 : 1 | −0.20 |

Findings:

* The old "load drift" story is dead: same-arm first-half vs second-half TTFT
  medians move by ≤ 2.3 ms across the whole run (C1-P: 289.7 → 287.4). The
  B1-vs-C3 selector gap was the double count (§3), and B1's selector now
  measures 40.8 ms — the same facility loop C3 measured at 41.4.
* Component accounting now explains itself: TTFT − Σ(stages) = +3.0…+3.6 ms
  for **every** arm (un-bracketed glue: norm + lm_head + launches + sync
  jitter), so the tables are internally consistent and "model prefill median
  > TTFT median" cannot recur.
* The −22.5 ms vs B1 is real but is a **selector-cost story**, not a pruning
  story: it is 32 ms of B1's expensive facility loop, of which GDEP's
  structure saves only 8.6 ms of suffix LLM. Against B2 — the incumbent with
  the cheap selector — GDEP **loses** 10 ms: the 28.0 ms full-token L0–L4
  prefix plus 0.3 + 0.1 of scoring/compaction is not repaid by the
  8.6 ms suffix saving.
* TTFT reduction vs B1 = 7.2–7.4 %, vs the pre-registered primary bar of
  15 %: **not met**; against B2 the sign is wrong. Decode: no arm moves TPOT
  (all within 4 ms of each other over 32 steps). Memory: unchanged conclusion
  (weights-owned footprint).

## 8. Method reading (directive 七) — final

The corrected facts (paired benchmark, §7; valid accuracy, §6):

* Accuracy: P′ 60.68, R 60.90 vs B1 61.10, vs B2 59.88. Statistically
  **level** with the incumbent (every CI spans zero; the point estimates are
  −0.2 and −0.4 below B1, +0.8/+1.0 above B2). GDEP is NOT "accuracy clearly
  better than EADP": the preregistered accuracy-led criterion (CI lower bound
  above B1) fails, as do the primary (≥15 % TTFT cut) and strong-compression
  (≥25 % cut or ≥20 % memory) criteria — the 15 % TTFT bar is missed at
  7.2–7.4 % versus B1, and the sign is wrong versus B2.
* The TTFT win over B1 is bought from B1's *selector* (facility 40.8 ms), not
  from the pruning point: the L4 prune pays 28.0 ms of full-token L0–L4
  prefix and repays only 8.6 ms of suffix. The honest speed reference is B2 —
  the incumbent with the cheap selector — and GDEP is 10 ms behind it.
* Memory flat (weights-owned); TPOT flat (decode is weight-bandwidth bound).

**Reading: the L4 pruning point is structurally too late.** With a corrected
engine the method does exactly what the cheapest incumbent configuration does
on accuracy, at a mid-stack cost no pre-LLM prune pays. Per the directive's
own rule, the next research direction is therefore to **distil the validated
L4 / gradient utility into a vision-encoder or pre-LLM scorer**, keeping the
scorer's accuracy but eliminating the 28 ms full-token prefix — i.e. the M1/S2
distillation line re-opened at the *pre-LLM* interface, not further sweep of
selectors at L4.

Status of the selector comparison that reading A authorised: it stays open as
a measurement (block8/facility maps at L4 would say whether the +0.8/+1.0 pt
over B2 survives a cheap selector — GDEP-topk already costs 0.07 ms to select,
so a GDEP+block8 arm is structurally the only way GDEP could still overtake
B2's TTFT *and* hold accuracy), but the §7 rule says that cannot resurrect
the L4 pruning point as a method result: the best case is parity with the
incumbent on both axes, which is not a Pareto point. Recording the decision
for the user rather than executing it unilaterally: **recommended next step
is the pre-LLM distillation pivot**; a GDEP-block8 arm is a 40-minute check
that can ride along if wanted.

What the amendment established beyond the defect itself:

1. The interim's headline mystery — GDEP at 38 vs ~57 expected — is fully
   explained by the decode defect; the method's honest accuracy is ≈ B1
   (measured twice, under both position policies, on identical token sets).
2. The position policy is not a design variable at this resolution
   (|P′ − R| = 0.21 pts [−2.57, +2.02]), and the earlier P-vs-R contrast on
   contaminated data ("renumber − preserve = +22") was entirely the defect.
3. The performance story of M2 is B2, not B1: a cheap selector is worth
   32 ms TTFT and ~1 accuracy point, and every future method comparison has
   to be against block8, not the facility incumbent (proposing that
   comparison-baseline change for the next stage's prereg; it does not amend
   this one's frozen criteria, which B1 remains).

## 9. Artefact map

| artefact | produced by | note |
|---|---|---|
| `m2_accuracy.json` | grid + amendment runs | contaminated records kept, marked `INVALID_DUE_TO_FROZEN_ROPE_POSITION`; valid: B0/B1/B2, `C1-R|s*`, (after §4) `C1-P|s*` with v2 hash |
| `m2_pre_stop_snapshot/` | this amendment | pre-stop copy of the four live artefacts |
| `m2_run_progress_attempt6_stopped.log` | grid | the stopped grid's log |
| `m2_fixgates.json` | `m2_fixgates.py` | §4 |
| `m2_correctness_v2.json` | `m2_correctness.py --tag m2_correctness_v2` | G-A…G-E on the fixed engine |
| `m2_perf_paired.json` | `m2_perf_paired.py` | §7 |
| `m2_amend_report.json/.md` | `m2_amend_report.py` | valid-arms-only tables + frozen-decision reading |
| `m2_gdep.py` (v2) | fixed engine, `ENGINE_VERSION` stamped | git diff vs 6a74c0b is the whole defect record |
