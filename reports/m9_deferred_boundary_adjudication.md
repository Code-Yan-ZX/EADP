# M9 — Deferred Boundary Adjudication

**Stage:** M9, a new formulation. MissGuard (M3), residual capsules (M4),
SafeTrim (M5), conditional utility (S3-A), bundle prediction (S3-B),
disagreement rescue (M6), candidate-union hedging (M7) and targeted
zeroth-order auditing (M8) are closed and this document reopens none of them.
Everything here is forward-only and training-free unless stated otherwise.

**The question this document answers, and the only one.**

> Six stages established that cheap pre-LLM signals nominate but cannot rank,
> and that no pre-LLM quantity — including a measured zeroth-order estimator —
> can order the candidates inside an enriched pool (M6, M8). M7 showed that
> keeping the whole pool is unaffordable. So: **defer the irreversible
> decision.** Carry a small rejected-candidate reserve *into* the first LLM
> layers alongside the incumbent core, let real text→visual attention happen,
> and only then re-adjudicate the selector's boundary — `S_final = S_core ∪
> top-r(boundary)`, |S_final| = 256 always.

**Status:** Phases 0, 0.5, 1, 2, 3 run; **Phase 4 (720 confirmation) not run**,
because the pre-registered screening gate failed (§8). Everything here is
forward-only and training-free at inference except the ORACLE arm, which is an
offline ceiling label.

**The answer, in one line.** **Deferred adjudication is the first rescue
mechanism in the project that works — a post-interaction attention signal
ranks the boundary at teacher rank 79 where every pre-LLM feature stops at
131, and the adjudication beats its random control at generation (+2.9 vs
+0.6 over D-B2) — but the formulation's own generation ceiling, measured by
the matched teacher-perfect oracle, is +2.70 over D-B2 and the attention
baseline already sits on it. Net over original B2 after the −1.8 early-
exposure cost: +1.0 macro, not significant, below B1. The screening gate
fails on its pre-registered clauses; the stage stops and reports. Verdict:
REFUTED as a paper method, with the project's first positive mechanism
result.**

---

## 1. Executive Summary

- **The formulation's own ceiling exists (Phase 0.5).** A perfect adjudicator
  confined to `boundary = tail-r ∪ reserve-32` places the rescue at mean
  teacher rank **10.0 (r=4)** and **26.9 (r=8)** — inside or near the band
  M3-v0 measured as converting (3.5–15.5 → +7.7..+14.9 macro), far above the
  dead band (73.7–131.9 → ≈ 0). At r=16 the ceiling degrades to 58.4 — the
  boundary is too large a budget for the pool; r=4/8 are the working points.
- **Phase 0 — the mechanism is reachable at negligible cost.** The question
  sits *after* the visual span in the prompt, so every decoder layer's causal
  self-attention carries text→visual interaction. SDPA returns no weights, so
  the capture recomputes, per layer, only the `~31 text query rows × 48
  boundary key rows` slice by hand from hooked q/k/v projections (read-only
  hooks; the forward path is untouched). Gate G-CAP: the whole decoder layer
  reconstructed in fp32 from the retained raw q/k/v matches the module's own
  bf16 SDPA output at relative error 3.6e-3 — the expected bf16↔fp32
  tolerance. Overhead: +73 ms per instance for 8 captured layers at S≈319,
  no S×S matrix ever materialised.

---

## 2. Phase 0 — Architecture / feasibility audit

All facts verified on the deployed stack (transformers 4.57.3, sdpa,
`Qwen/Qwen3-VL-8B-Instruct`, snapshot `0c351dd0`), live:

| item | value | consequence |
|---|---|---|
| decoder layers | **36**, 0-based 0..35 | L ∈ {1,2,4,8} = layers {0}, {0–1}, {0–3}, {0–7} |
| Q / KV heads | **32 / 8** (GQA group 4), head_dim 128 | CMC needs the KV-head mapping `v_h = v[h//4]` and per-head `o_proj` slices |
| q/k norm | per-head RMSNorm, **before** rope | captured raw q/k must be re-normalised before use |
| rope | mrope (interleaved, sections [24,20,20]); the harness feeds 1-D positions expanded to 3 identical rows | reuse the layer's own cos/sin + `apply_rotary_pos_emb`; slicing rows and slicing cos/sin commute |
| attention | SDPA, weights not returned | recompute only the text-query × boundary-key slice by hand |
| prompt layout | `<|im_start|>user\n<|vision_start|>` + 1024 visual + `<|vision_end|>` + **question** + `<|im_end|>\n<|im_start|>assistant\n` | **the question sits after the visual span and attends to it causally at every layer** — the premise of the method holds |
| KV cache | `DynamicCache`; GDEP already compacts layers 0..L by `index_select` | Phase 2 reuses the proven compaction path |

**The capture (G-CAP gate).** No eager-attention switch and no S×S matrix:
read-only forward hooks record each early layer's raw q/k/v and its
`position_embeddings`; after the layer returns, the scores for the
`~31 text query rows × 48 boundary key rows` slice are recomputed by hand
(q/k norm → rope → logits → causal softmax → probs; CMC additionally projects
`a_h·v_{kv(h)}` through `o_proj`'s per-head slice). Gate: an fp32 eager
reconstruction of the **whole decoder layer** from the retained raw q/k/v
matches the module's own bf16 SDPA output at relative error **3.6e-3**
(median, text rows) — the expected bf16↔fp32 tolerance. **PASS.**
Overhead: **+73 ms** per instance for 8 captured layers at S≈319 (plain
forward 17 ms), ~26 s for the whole 210-instance Phase-1 panel.

**Declared conventions.** Position policy RENUMBER (arange) for the early
sequence — the incumbent prellm path's own implicit convention, like-for-like
with B2 and with the D-B2 control. Layer indices are 0-based throughout
("L4" = layers 0–3 captured, scored at the output of layer 3). The boundary
`tail-16 ∪ reserve-32` is captured once and covers r = 4/8/16; scores are
r-independent, so one pass per instance serves every condition.

## 3. Phase 0.5 — Oracle boundary ceiling (stop-or-go)

Computed offline from the frozen banks (`m9_oracle_boundary.py`); no model.
Oracle = the r boundary members with the highest P1-G2 teacher score (tail
tokens compete directly); headline panel held-out 210.

| r | frac any swap | mean swap teacher rank | median | head recall@r | reading |
|---:|---:|---:|---:|---:|---|
| 4 | 1.00 | **10.0** | 5.0 | 0.387 | inside M3-v0's converting band (3.5–15.5 → +7.7..+14.9) |
| 8 | 1.00 | **26.9** | 18.0 | 0.299 | near M8's P1 pool oracle (31.3); plausibly converting |
| 16 | 1.00 | **58.4** | 45.0 | 0.215 | at the edge of the dead band (73.7–131.9 → ≈ 0) |

The perfect adjudicator swaps essentially every slot (`frac_full_swap` 0.92 /
0.62 / 0.06 at r = 4/8/16): the tail is always the weakest part of the
boundary. Cross-check: head recall@8 = 0.2994 reproduces M8's P1 `cov@8`
exactly — the reserve construction is the M8 pool P1 by construction.

**Verdict: GO.** The restricted formulation has a real ceiling at r = 4/8.
r = 16 is capped in the dead band and is demoted to a secondary budget.

## 4. Phase 1 — Does early LLM interaction break the ranking wall?

Frozen banks + one capture pass over the 210 held-out instances
(`m9_phase1_scores.npz`); the teacher never enters the run. Every arm keeps
exactly r of `boundary = tail-r ∪ reserve-32`; two framings — **adj** (rank
the whole boundary by the score) and **resc** (always evict the tail, rescue
the top-r reserves; the M6/M8-comparable framing). Panel held-out 210.

**Baselines (r = 8, mean swap teacher rank / head recall@8):**

| arm | rank | recall |
|---|---:|---:|
| oracle in boundary | **26.9** | 0.299 |
| **CMC best cell** (`cmc_mn\|A2_last4\|L4\|adj`) | **78.6** | **0.243** |
| cos_s0c (adj = resc here) | 131.0 | 0.209 |
| random in boundary | 210.6 | 0.060 |
| incumbent tail (no swap) | — (0 swaps) | 0 |

(The `cos_resc` cell reproduces M8's construction; M8's *reported* 118.9 was a
misaligned computation — see the amendment in `reports/m8_targeted_zo_audit.md`.
The correctly aligned value is 131.0, and this panel reproduces it.)

**The grid** (r = 8, top cells; full grid in `m9_phase1_eval.json`):

| signal | agg | L | rank | recall |
|---|---|---:|---:|---:|
| avn (a·‖v‖) | A3_all | 4 | 78.1 | 0.243 |
| cmc_mn | A2_last4 | 4 | 78.6 | 0.242 |
| cmc_mn | A3_all | 4 | 78.6 | 0.243 |
| cmc_nm | A2_last4 | 4 | 80.9 | 0.241 |
| att_mean | A3_all | 4 | 81.3 | 0.242 |
| cmc_nm | A3_all | 8 | 85.5 | 0.241 |
| cmc_mn | A2_last4 | 1 | 119.6 | 0.196 |
| cmc_mn | A2_last4 | 2 | 138.8 | 0.197 |

Findings, in order:

1. **The wall is broken, not shattered.** 78.6 vs 131.0 — the first
   query-conditioned signal in the project's history that beats the best
   pre-LLM proxy by a wide margin, at a cost of ~9 ms of extra attention
   arithmetic per layer. ρ(reserve ranking, teacher) = **+0.565** vs cos's
   +0.331. CMC recovers **81 %** of the oracle's head-recall advantage over
   cos_s0c (0.243 vs 0.209, oracle 0.299).
2. **All of it happens at L4, and almost all of it is plain attention.** L4 is
   the best depth for every signal, non-monotonically (L8 ≈ +5–10 ranks, L1 ≈
   +41, L2 ≈ +60 — back at cos_s0c level); avn ≈ cmc_mn ≈ att_mean within
   3 ranks — the `o_proj` projection and even the value norm add almost nothing
   over routing probability. The signal is *where the question looks*, not what
   the look is worth.
3. **Same direction on all three benchmarks** (mean swap rank, r = 8, adj):
   TextVQA 132.3 → **73.5**, DocVQA 119.1 → **82.2**, OCRBench 140.3 →
   **80.3**. Paired bootstrap (4000 resamples): rank diff −51.9,
   95 % CI **[−60.5, −43.8]**; the resc framing agrees (−49.0 [−56.8, −41.4]).
4. **Pre-registered primary cell**, declared before D1–D3 were read:
   `cmc_mn | A2_last4 | L4 | adj` (tied best in the grid). All downstream
   phases use it.
5. **Positive-case identification**: on the 200/210 instances where the oracle
   swaps in at least one teacher-top-8 dropped token, CMC lands ≥ 1 such token
   on **93.0 %** (cos 90.0 %) — and its hits are 51 ranks shallower on average.

**Phase-1 gate reading.** The brief's ideal (r = 8 mean rank < 60, ideally
40–50) is **not** met; the REFUTED band (≈ 100–150) is **not** entered. The
outcome is the brief's middle band (60–80) with its condition satisfied:
multiple benchmarks in the same direction and a decisive, paired-significant
improvement over cos_s0c. **Proceed to the small-scale generation gate
(Phase 2/3/4 on bank150), with D-B2 as the primary baseline.**

---

## 5. Phase 2 — The deferred-pruning engine, and its identity gates

`m9_accuracy.py` implements the real inference path. One code path for every
arm: assemble `S_early = sorted(S0 ∪ reserve) = 288` visual tokens (positions
RENUMBER) → decoder layers 0..L−1 (ATTN/CMC capture text→boundary scores at
layer L−1 inside the same pass) → adjudicate the boundary to exactly r →
compact the layer-0..L−1 KV cache and hidden states → layers L..35 → greedy
decode.

**Identity gates, both passing:**

| gate | check | result |
|---|---|---|
| G-B2 | the incumbent path reproduces the stored M2 B2 predictions | **150/150** predictions, 150/150 hits |
| G-L0 | the deferred path with **L=0** (zero layers before compaction) is the incumbent path by construction | **150/150** predictions identical, macro identical (59.878) |

So every difference below is attributable to the early 288-token exposure and
the boundary adjudication — not to the engine.

## 6. Phase 3 — bank150 screening (the only generation panel run)

All arms share the identical early 288-token pass at L=4, identical reserve-32,
identical schedule; only which r boundary members survive differs. D-B2 (keep
the incumbent tail) is the primary baseline.

**T1 — main table.** ΔOrig = Δ vs original B2 (59.878); ΔD-B2 = the
adjudication effect (vs 58.035).

| Method | r | TextVQA | DocVQA | OCRBench | Macro | ΔOrig-B2 | ΔD-B2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ORIGINAL-B2 | — | 65.400 | 60.235 | 54.000 | **59.878** | — | +1.843 |
| **D-B2** | — | 63.400 | 58.705 | 52.000 | **58.035** | **−1.843** | — |
| RANDOM | 4 | 67.400 | 57.188 | 54.000 | 59.529 | −0.349 | +1.494 |
| COS | 4 | 61.400 | 57.790 | 54.000 | 57.730 | −2.148 | −0.305 |
| ATTN | 4 | 61.400 | 58.252 | 56.000 | 58.551 | −1.328 | +0.516 |
| CMC | 4 | 65.400 | 58.252 | 54.000 | 59.217 | −0.661 | +1.182 |
| ORACLE | 4 | 61.400 | 57.790 | 54.000 | 57.730 | −2.148 | **−0.305** |
| RANDOM | 8 | 63.400 | 58.590 | 54.000 | 58.663 | −1.215 | +0.628 |
| COS | 8 | 65.400 | 58.335 | 58.000 | 60.578 | +0.700 | +2.543 |
| **ATTN** | 8 | 65.400 | 57.118 | 60.000 | **60.839** | **+0.961** | **+2.804** |
| **CMC** | 8 | 65.400 | 57.318 | 60.000 | **60.906** | **+1.027** | **+2.871** |
| ORACLE | 8 | 65.400 | 58.790 | 58.000 | 60.730 | +0.852 | +2.695 |

**T3 — paired bootstrap (macro points, 6000 resamples).** On 150 instances
nothing is individually significant (M3-v0's MDE₈₀ ≈ 7 applies): CMC8−D-B2
+2.87 [−1.7, +7.2]; CMC8−RND8 +2.24 [−1.5, +6.2]; ATTN8−RND8 +2.18
[−1.5, +6.2]; CMC8−B2 +1.03 [−3.2, +5.0]; D-B2−B2 −1.84 [−4.7, +0.7].

**T4 — flips vs D-B2.** CMC8 9 fixed / 5 broken (net +4); ATTN8 8/5 (+3);
COS8 and ORC8 8/3 (+5); RND8 5/4 (+1). At r=4 everything nets ≤ +3 with
random at +3.

## 7. Temporary Exposure vs Boundary Adjudication

The decomposition the brief requires, in order:

| step | macro | effect | what it answers |
|---|---:|---:|---|
| ORIGINAL-B2 (no early pass) | 59.878 | — | the incumbent |
| → D-B2 (288 tokens through L4, keep tail) | 58.035 | **−1.843** | Q1: temporary exposure |
| → RANDOM-r8 | 58.663 | +0.628 | the swap mechanics alone |
| → COS-r8 | 60.578 | +1.915 | pre-LLM ranking |
| → ATTN-r8 | 60.839 | +0.261 | post-interaction routing |
| → CMC-r8 | 60.906 | +0.067 | post-interaction contribution |
| → ORACLE-r8 | 60.730 | −0.176 | Q4: the generation ceiling |

**Q1 — does carrying reserve-32 through the first 4 layers help by itself?
No.** D-B2 is **−1.84** below original B2 (n.s.). The early exposure is a
small cost, not a free lunch; nothing was hidden inside it.

**Q2 — does actually re-choosing the boundary add value? Yes, at r=8.**
Content-bearing adjudication gains **+2.5 to +2.9** over D-B2 while a random
boundary draw gains only **+0.63** — the content of the swap matters
(≈ +2.2 for ATTN/CMC over random), and the net result (+0.96..+1.03 over
original B2) is entirely adjudication, partly cancelling the exposure cost.
At r=4 there is no effect to explain: every arm nets ≤ +1.5 with random the
best, and even the oracle lands at −0.31 — four slots are not enough for
this boundary.

**Q3 — is ordinary attention enough? Yes.** ATTN8 ≈ CMC8: macro Δ +0.07,
identical surviving sets on **85/150** instances, per-benchmark profiles
nearly identical. By the brief's simplicity rule the method of record is
**ATTN** (post-interaction text→boundary attention, layer 3, last-4-question
aggregation). CMC's extra machinery (value norms, o_proj projection) buys
nothing downstream.

**Q4 — how high is the real generation ceiling? +2.70 over D-B2 at r=8 —
and the oracle does NOT beat CMC.** The teacher-perfect boundary adjudicator
(60.730) sits *below* ATTN/CMC (60.84/60.91). The restricted formulation is
**saturated**: no better boundary ranker can extract more from it, because
the ceiling itself is the binding constraint — not the ranking (Phase 1's
teacher-rank gain of 131 → 79 converts to ≈ 0 here, the fourth independent
confirmation of the S2-C2 depth-convexity pattern). At r=4 the ceiling is
negative.

## 8. Screening gate verdict — STOP

| brief's condition | outcome |
|---|---|
| 情况 1: ATTN or CMC ≥ ~+3 over D-B2 | ❌ best is **+2.87** (CMC8) |
| 情况 2a: ATTN/CMC > RANDOM | ✅ +2.2 (not individually significant on 150) |
| 情况 2b: > COS | ⚠️ only +0.33 |
| 情况 2c: all three benchmarks same direction | ❌ DocVQA is negative for every r=8 arm (−1.3 to −1.6) |
| 情况 2d: oracle shows remaining headroom | ❌ **ORC8 (+2.70) < CMC8 (+2.87) — ceiling already reached** |
| 情况 2e: paired flips clearly support the selector | ⚠️ 9/5 vs 5/4 — direction only |
| STOP: ATTN/CMC ≈ RANDOM | ✅ avoided (+2.2) |
| STOP: gain entirely from temporary exposure | ✅ avoided — exposure is *negative*; the gain is adjudication |

**情况 1 fails; 情况 2 fails on two of its five clauses (2c, 2d). The 720
confirmation panel is NOT run. Per the brief, the stage stops here and
reports.**

---

## 9. Final Verdict

# REFUTED (as a paper method) — with the project's first positive mechanism result

> **Should deferred boundary adjudication become the final paper method? NO.**

| criterion | outcome |
|---|---|
| fixed 256 tokens | ✅ every arm |
| ~+3 macro over D-B2 (screening) | ❌ +2.87 at best, CI includes 0 |
| beats official EADP (B1 = 61.10 on this panel) | ❌ best arm 60.91 |
| 720 confirmation | not run — gate failed |

**Why, exactly.** M9 achieved what six stages said was impossible: a
query-conditioned signal (plain text→visual attention at layer 3) that ranks
the candidates far inside what any pre-LLM feature can reach (teacher rank
78.6 vs 131.0), at negligible cost, with a clean engine (G-L0 bit-identity).
And it converts — the adjudication effect (+2.9, +2.2 over random) is the
first rescue mechanism in the project's history that beats its random control
at the generation level. **But the formulation's own generation ceiling is
+2.70 over D-B2 / +1.03 over B2 (n.s.), and the oracle arm proves it is
saturated: a perfect boundary ranker does no better than the attention
baseline.** The value that remains after paying the −1.8 early-exposure cost
is +1.0 macro — below B1, below significance, below the paper bar. The depth
wall was real but it was never the last wall; the last wall is that the
boundary's convertible value is small no matter how well it is ranked.

**What is left standing, all reusable:**

1. **The pre-LLM ranking wall is broken in principle.** Post-interaction
   text→visual attention at layer 3 ranks a 32-token reserve pool at mean
   teacher rank 79 where the best pre-LLM feature reaches 131 — with
   read-only hooks, ~9 ms/layer, no S×S matrix, and a bit-exact identity
   path (G-L0). Any future formulation that needs query-conditioned token
   ranking should start from this signal, not from the pre-LLM bank.
2. **Deferred re-adjudication works as a mechanism**: swapping the greedy
   tail for nominated reserves after real cross-modal attention is worth
   ≈ +2.2 over a random swap — the first content-sensitive rescue in eight
   stages. But the same measurement prices its ceiling: ≈ +2.7 over D-B2,
   already captured. A future method must find a boundary whose *oracle*
   is deeper than ±3 macro (e.g., a larger reserve that the early layers
   can afford, or adjudication without the early-exposure cost), or accept
   that r ∈ {4, 8} over a 32-reserve pool has ≤ 1 point of net value.
3. **The r=4 caution**: with 4 slots, the oracle itself cannot help
   (−0.31); small-budget boundary adjudication on this pool is noise.
4. **A correction M9 contributed to M8**: the cheap-rule cells of M8's
   Phase-1 grid were misaligned (features of instance j read against the
   pool of instance hold[j]); corrected values are in the M8 amendment.
   Oracle/random/ZO-P cells were unaffected; M8's verdict stands.

**What would change the verdict.** Not a better aggregator (§6-Q3, §7-Q4:
saturated), and not r=16 (Phase 0.5 caps it in the dead band). It would take
a boundary whose oracle ceiling is materially larger than +3 macro — i.e.
more slots, a cheaper way to carry the reserve (the −1.8 exposure bill), or
adjudication at a depth where the teacher head is still convertible. Nothing
in this stage suggests such a boundary exists within the B2 + 32-reserve
design; proposing one would be a new stage, not a fix of this one.

---

## Appendix — files

| file | what |
|---|---|
| `scripts/discovery/m9_capture.py` | `EarlyLayerCapture` — read-only q/k/v hooks + slice recomputation (att_mean, avn, cmc_mn, cmc_nm) |
| `scripts/discovery/m9_common.py` | boundary construction (tail-16 ∪ reserve-32), early-sequence assembly |
| `scripts/discovery/m9_oracle_boundary.py` | Phase 0.5 oracle boundary ceiling (offline) |
| `scripts/discovery/m9_phase0.py` | Phase 0 audit + G-CAP equivalence gate |
| `scripts/discovery/m9_phase1.py` | Phase 1 capture over the held-out 210 |
| `scripts/discovery/m9_phase1_eval.py` | Phase 1 adjudication grid (adj/resc framings) |
| `scripts/discovery/m9_phase1_diag.py` | per-benchmark direction, bootstrap, positive-case ID |
| `scripts/discovery/m9_accuracy.py` | Phase 2/3 deferred engine + bank150 screening, G-B2/G-L0 gates |
| `scripts/discovery/m9_analyze.py` | T1–T4, decomposition, bootstraps, flips |
| `outputs/discovery/m9_oracle_boundary.json` | Phase 0.5 numbers |
| `outputs/discovery/m9_phase1_scores.npz` | captured scores (210 × 8 layers × 4 metrics × Q × 48) |
| `outputs/discovery/m9_phase1_eval.json`, `m9_phase1_diag.json` | Phase 1 grids and diagnostics |
| `outputs/discovery/m9_phase0.json` | Phase 0 audit record |
| `outputs/discovery/m9_accuracy_bank.json` | bank150 generation record (13 arms, predictions + per-instance sets) |
| `outputs/discovery/m9_analyze.json` | screening tables |

Reused unchanged: `m5_bank.npz`, `m6_eapd_order.npz` (greedy order),
`s2b_gradient_scores.npz` (teacher, offline label only), `m2_gdep.py`
(engine base: prepare/decode/cache compaction), `m5_common.py`/`m2_accuracy.py`
(panel and scoring), `instrumented.py` (incumbent pruner).
