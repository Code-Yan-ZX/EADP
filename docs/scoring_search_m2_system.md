# M2 — Gradient-Distilled Early Pruning (GDEP): the system Pareto

**Scope.** M2 is the first stage in this project that implements and measures a
**real method prototype end to end**, on one code path for accuracy and for
latency. Everything before it — S1, S2-A…S2-C6, M1 — was a proxy diagnostic:
token sets scored offline, or a cached importance map injected into the
incumbent's pre-LLM harness. M1 (`scoring_search_m1.md` §8 Q6) opened the deploy
gate on a literal reading of its own conditions; M2 is the stage that gate was
for.

`GDEP` is an engineering identifier for the configuration under test. It is not a
novelty claim. Whether it is worth anything is decided by the accuracy–efficiency
Pareto and by nothing else.

**Status.** COMPLETE. Measurement finished under an implementation-defect
amendment (§2.1 and `scoring_search_m2_system_amendment.md`): the original
PRESERVE accuracy grid was stopped after the interim audit localised a frozen
RoPE position in the decode branch, and the corrected grid re-measured the
candidate arms. Where this report and the amendment overlap, the amendment's
numbers supersede the original grid's; the contaminated records are retained
in `m2_accuracy.json` marked `INVALID_DUE_TO_FROZEN_ROPE_POSITION` and enter
no number below.

**Pre-registration** `scoring_search_m2_system_prereg.md`, written before the
first line of the engine and before any measurement, with all three success
thresholds frozen. The amendment did not touch the preregistration's criteria;
it corrected the implementation so the criteria could be evaluated honestly.

---

## HEADLINE

**The mechanism works and it is not enough.** GDEP — a real mid-stack layer-4
token prune, 524 673 online parameters, one forward pass, no recomputation —
holds EADP-level accuracy: seed-mean macro 60.90 (RENUMBER) / 60.68
(corrected PRESERVE) against the incumbent's 61.10, paired CIs spanning zero
(−0.20 [−7.36, +6.71] / −0.42 [−7.54, +6.51]). Its TTFT edge over the
facility incumbent is **real but small and not its own**: 120 interleaved
paired blocks give −22.5/−23.0 ms (118/120 block wins), but essentially all of
that is B1 paying 40.8 ms for the expensive facility selector. Against the
incumbent with the cheap selector — **B2, block8@256 — GDEP is 10 ms SLOWER
(288.3/287.8 vs 278.6), 7.2–7.4 % short of the pre-registered 15 % primary
bar, with no memory or TPOT gain.** The 28.0 ms of full-token L0–L4 prefix is
repaid by only 8.6 ms of suffix saving, and no selector choice inside the
architecture changes that arithmetic: at L4 the block8 and facility variants
measured 299.1/323.8 ms in the (timing-valid) first grid.

**Verdict: none of the three pre-registered bars is met (the ladder's "no
candidate clears a Pareto bar" branch), and per the amendment's directive-7
rule a merely-equivalent accuracy means the L4 pruning point is structurally
too late.** The lesson is not that the score is bad —
M1's LOCAL-MLP transfers intact into the real graph (§3) — but that *where*
the decision is made costs more than what it buys. The next direction the
evidence points at is distilling the validated L4/gradient utility into a
**vision-encoder / pre-LLM scorer**, keeping the accuracy profile at zero
prefix cost, rather than any further sweep at layer 4.

Two secondary findings the measurement forced out (full detail in the
amendment): the interim "GDEP collapses to 38 macro" was 100 % a frozen-RoPE
decode defect — flipping only the position advance moves 38 → 60.7 on
identical token sets; and the perf table's "facility selector load drift,
85.8 vs 41.4 ms" was a selector-window **double count**, not drift — the
drifted claim is dead, the true facility cost is 40.8 ms, and component
accounting now closes to +3.1…+3.6 ms of un-bracketed glue on every arm.

---

## 1. What M2 implements

The method under test, in one sentence: **run the prompt at full length through
the first five decoder layers, score the 1024 visual tokens there with the
524 673-parameter M1 `LOCAL-MLP`, keep 256 of them, and run the remaining 31
layers over the compacted sequence.**

Concretely, in `m2_gdep.GDEPEngine.prefill`, mode `gdep`:

| step | what happens |
|------|--------------|
| 1 | vision tower produces the 1024 merged visual embeddings, as always |
| 2 | they are spliced into the prompt at the image-token positions |
| 3 | decoder layers **0…4** run over the **full 1058-token** sequence, once |
| 4 | `h₄`, the visual slice of the layer-4 residual stream, `(1024, 4096)`, is read off |
| 5 | the **online** scorer computes `s = w₂·GELU(P ĥ₄ + b)`, 524 673 parameters |
| 6 | the selector picks 256 of the 1024 indices |
| 7 | hidden states, the layer-0…4 KV cache, the attention mask and the position ids are compacted |
| 8 | decoder layers **5…35** run over the kept 256 visual + all text tokens |
| 9 | greedy generation continues, 36 layers per step, over the compacted cache |

L0–L4 run exactly once. Nothing is recomputed, no prefix is thrown away and
re-run, and no cached score vector is read at inference time. (The offline
teacher maps are used only to *train* the scorer, offline, in M1; their cost is
reported in §7 and is excluded from every deployment number.)

### 1.1 One engine, three modes, and why that matters

`B1` and `B2` are the incumbent and cannot be measured with a different harness
than the candidates, or the Pareto is comparing two programs. So the engine has
three modes and every arm goes through the same manual layer loop, the same
decode loop and the same prompt construction:

| mode | arms | what runs |
|------|------|-----------|
| `full` | B0 | 36 layers over 1058 tokens, no pruning |
| `prellm` | B1, B2 | the incumbent's own pruner over the vision embeddings, then 36 layers over the compacted sequence |
| `gdep` | C0–C3 | the nine steps above |

Gate **G-F** is what licenses quoting B1/B2 from this engine: with the official
importance map and selector, `prellm` mode must reproduce the published
Stage-1 held-out numbers.

### 1.2 Position policy

Two policies exist and both are implemented; the pre-registration fixes
**PRESERVE** as primary.

* **PRESERVE** — a kept token carries the position id it had in the full
  sequence, so the compacted `position_ids` are `sorted(kept_indices)` and are
  **not contiguous**; the next generated token takes `max + 1`. The kept tokens
  are seen by layers 5…35 at exactly the relative distances the full model would
  have seen them at, and are never renumbered.
* **RENUMBER** — `arange(0, L')`. This is what the incumbent's pre-LLM path does
  implicitly, so it is carried as a declared control on C1 only (`C1-R`) to make
  the GDEP-vs-EADP positional comparison like-for-like.

### 1.3 KV-cache policy

Every arm has exactly one policy and it is recorded per arm in every result
record.

| arm | early-layer cache (L0–L4) | late-layer cache (L5–L35) |
|-----|---------------------------|---------------------------|
| B0 | — (36 layers × 1058) | — |
| B1, B2 | **does not exist** — no LLM layer runs at 1024 tokens | 36 layers × 290 |
| GDEP | 5 layers allocated at 1058, **compacted to 290** | 31 layers × 290 |

The GDEP policy is **compact-L0–L4**: after selection the keys and values of the
dropped visual tokens are removed from the layer-0…4 caches, so a kept token's
late-layer representation never attends to a dropped token. The transient
1058-token allocation of layers 0…4 is measured separately from the
steady-state 290-token footprint, because it is a real cost of pruning late
rather than early.

The consequence is stated up front and is not a defect: GDEP's layer-0…4
attention for the kept tokens was computed over the full 1058-token context,
whereas B1/B2's was computed over 290 tokens from the start. **GDEP and B1/B2 are
different computation graphs; their logits are not expected to agree, and no
gate asks them to.**

### 1.4 Two inherited conventions

* **No deepstack.** `s2c1_features.py`, `m1_features.py` and the incumbent's
  pruned path all hand the model `inputs_embeds` with no `pixel_values`, so
  `Qwen3VLModel` never builds `deepstack_visual_embeds` and the deepstack
  features are not injected. GDEP keeps that convention in every mode. Gate G-A
  is what forces it: the online `h₄` must match the published M1 cache, and that
  cache was built without deepstack. **This is a property of the whole project,
  not of M2, and it is a limitation of every number in it.**
* **1-D text positions.** With `inputs_embeds` and no `image_grid_thw`,
  `get_rope_index` takes its no-image branch and every position id is a plain
  running index. Same in every mode.

---

## 2. Correctness gates

All five machine-checked gates passed, on the code that then produced every
performance and accuracy number (`m2_correctness.json`, environment torch
2.10.0+cu128 / CUDA 12.8 / A40, SDPA). The gates were not decorative: two of the
three attempts at a green run failed, and each failure was a real property of
the implementation that the report now records.

| gate | question | measured | verdict |
|------|----------|----------|---------|
| **G-A** | online L4 hidden states vs the published M1 cache, 8 held-out instances | worst **max\|Δ\| = 0.000e+00**, 100.00 % of elements bit-exact, against a 4.715e-03 threshold (1 % of mean \|h\| = 0.4715) | **PASS** |
| **G-B** | online scorer vs cached-h₄ scorer | worst **max\|Δs\| = 0.000e+00**; Top-256 index sets identical on **8/8** instances | **PASS** |
| **G-C** | K = 1024 identity vs the stock HF forward | worst max\|Δlogits\| = **0.000e+00** in all three arms (full mode, no-op compaction, forced split-and-rejoin), tolerance 1e-3; selection is the identity permutation, argmax matches 8/8 | **PASS** |
| **G-D** | K = 256 invariants | **0 violations** on 12 instances × both policies: exactly 256 kept, no duplicates/out-of-range, all text kept, mask/position/cache lengths all 256+n_text, PRESERVE ids strictly increasing and a subsequence of the full range | **PASS** |
| **G-E** | generation sanity | 12/12 complete, non-empty, `layer_calls` exactly `36·(1+decode_steps)` on every instance — one pass per layer per step, no implicit recomputation | **PASS** |

G-A and G-C landing at **exact zero** rather than "inside tolerance" is worth
stating precisely: the engine's manual layer loop, causal-mask construction and
cache handling are the same computation as `Qwen3VLModel.forward`, not merely
a numerically-close re-implementation, and the online extraction point is the
same tensor M1 trained on, bit for bit. The whole M2 accuracy claim therefore
rests on the engine being the model, and on the whole distillation claim resting
on the same `h₄` the scorer was fitted to.

**The two real failures the gates caught, before any number was reported.**

1. *A mask-sentinel bug.* The first version of G-C measured the K = 1024 arm at
   max|Δlogits| = 3.125e-01 ≈ **0.8–1.0 bf16 ulp** — nonzero where full mode was
   exactly zero. The cause was not the compaction at all: `create_causal_mask`
   legitimately returns `None` when SDPA can use its `is_causal` fast path (an
   empty cache), and the second loop was written with `attn=None` as the
   "caller supplied nothing" default, so it **silently rebuilt a materialised
   additive mask against a now-populated cache**. One ulp at logit scale 40 is
   below what any downstream argmax would notice; "nothing changed" is exactly
   the kind of difference this project is supposed to catch. Fixed with an
   explicit `_AUTO` sentinel plus an identity fast path that carries the mask
   through an unchanged shape; the forced-split recheck then measured
   **0.000e+00** in all three columns, so the pre-registered 1e-3 tolerance was
   met outright and **no tolerance amendment was needed**.
2. *An accounting error in the recompute gate.* The first G-E version predicted
   `36·(1+n_decode)` layer calls, but the greedy loop breaks on EOS *after*
   appending the token, so an EOS-terminated generation runs `n_decode − 1`
   forwards, not `n_decode`. All 12 rows showed `got = expected − 36·(EOS
   break)` — a wrong formula, not a wrong engine. G-E now counts its own decode
   steps inside the engine, and the check is exact.

Both fixes predate the green run whose gates are tabulated above, and the same
green run then continued directly into the performance grid (one process, one
model load), so the measured system is the gated system.

**G-F, the gate that mattered most, fired during the accuracy grid.** The
pre-LLM harness-identity check is the only gate that compares *behaviour*, not
numerics, and it failed its first campaign: the engine's B1 arm produced
71.40 / 70.92 / 58.00 against the published official-facility reference of
59.40 / 61.89 / 62.00 on the held-out instances. The cause was in the engine,
not the incumbent: the pruner's text-side input had been assembled as the full
chat-templated prompt minus the visual span, while the official harness scores
against the **instruction text alone** (`_tokenize_instruction`: question
string with special tokens, no scaffolding). A longer, scaffolding-laden text
sequence changes the dense-guidance entropy filter, hence the importance map,
hence the selection — an incumbent with the wrong importance map is not the
incumbent. Fixed by calling the wrapper's own
`_get_instruction_sequence_embedding` verbatim. The GDEP arms are untouched by
this bug (their scores come from `h₄`; only B1/B2 consume the text side), but
because the headline comparison *is* C1-vs-B1, the entire performance and
accuracy grid was restarted after the fix rather than spliced.

Two implementation notes from the same episode, recorded because the
pre-registration's wording deserves a precise audit trail:

* the first G-F reference had been taken from `diag_selectors_b256.json`,
  which reports the whole 150-per-benchmark bank — *not* the held-out 50-per-
  benchmark subset. A ±0.5 test against a differently-composed mean can fail
  for arithmetic reasons alone. The reference now comes from
  `s2b_identity.json`, the S2-B §2.1 arm recorded **per instance** on the bank,
  sliced `[::3]` to exactly the held-out rows, so G-F is a paired per-instance
  comparison. This fixes which numbers are compared; it does not move the
  ±0.5 threshold, which was frozen beforehand.
* the first accuracy campaign also measured the two pre-LLM baseline arms
  under the bug; those records were purged with an explicit provenance note
  (`purged.reason`) in the JSON rather than silently overwritten.

### 2.1 The gate the gates missed, and the gate set that caught it

G-A…G-E all passed on the code that ran the original PRESERVE grid — and that
grid was still wrong, by ~23 macro points, because **every gate was a prefill
gate**. The decode loop recomputed the PRESERVE branch's RoPE position from the
frozen prefill tensor every step (`max+1` each time, `cache_position` correct),
so every generated token shared one position. The interim audit (D2/D5,
commit 6a74c0b) localised it; the amendment (`scoring_search_m2_system_amendment.md`)
stopped the contaminated grid, marked its records
`INVALID_DUE_TO_FROZEN_ROPE_POSITION` (retained, excluded from every number
here), fixed the engine to advance the position (`ENGINE_VERSION
gdep_preserve_v2_advancing_rope`, hashed into gdep config keys), and added
**HG-1…HG-9** decode gates: K = 1024 *multi-step* (24 forced steps, past EOS)
generation token-for-token identical to stock HF; position advance under both
policies with no repeat over 64 forced steps; the D5 trio de-looped to the
audit's P′ ids; all 50 previously-capped instances re-run (48 terminate by
EOS — the 2 exceptions track the scorer's token set, not the position, and hit
the RENUMBER arm identically); and a v1-vs-v2 invariance check — scores
**bit-identical**, selected sets exactly equal, RENUMBER decode identical —
which is what licenses the RENUMBER control's records as fixed-engine
measurements. G-A…G-E were then re-run whole on the v2 engine and pass
(`m2_correctness_v2.json`). The same amendment pass removed the prellm
selector-window double count described in §4.3.

---

## 3. Accuracy

Held-out 150 (50/benchmark), greedy, `max_new_tokens = 2048`, one generation
per (arm, seed, instance); hit ≥ 0.5 is the Stage-1 hard-error cut. Tables and
bootstrap CIs from `m2_amend_report.py` (`m2_amend_report.md/.json`); the
pre-registration's arm definitions are unchanged, and the only two valid GDEP
arms are the RENUMBER control (defect-inert, proven in §2.1) and the corrected
PRESERVE rerun. The n=240 / block8 / facility / SHUF arms were never re-run:
the reading they were preregistered to support is settled by §5 (P4) and by
the amendment's frozen decision, whose reading **A fires for both position
policies and whose reading D — |P′ − R| = 0.21 pts [−2.57, +2.02] — says the
position policy is not a method design variable at this resolution**.

| arm | TextVQA | DocVQA | OCRBench | macro | vs B1 [95 % paired CI] | EOS | capped |
|---|---|---|---|---|---|---|---|
| B0 full-1024 | 73.60 | 85.27 | 68.00 | 75.62 | +14.53 [+8.23, +20.86] | 100 % | 0 |
| B1 EADP facility | 59.40 | 61.89 | 62.00 | 61.10 | — (G-F: 150/150 hits) | 100 % | 0 |
| B2 EADP block8 | 65.40 | 60.24 | 54.00 | 59.88 | −1.22 [−7.57, +5.29] | 100 % | 0 |
| C1-R GDEP renumber | 71.40 | 61.29 | 50.00 | 60.90 | −0.20 [−7.36, +6.71] | 99.3 % | 3/450 |
| C1-P GDEP preserve (corrected) | 72.27 | 59.11 | 50.67 | 60.68 | −0.42 [−7.54, +6.51] | 99.3 % | 3/450 |

(seed-averaged per-benchmark hit rates; per-seed macro ranges: C1-R
59.66–61.72, C1-P 57.63–62.80. C0/C1/C2 contaminated values — 33.6–40.1
macro, 13–17 % capped — are in `m2_accuracy.json` marked invalid and are not
accuracy evidence of anything but the defect.)

What the valid arms say:

1. **The M1 scorer's accuracy survives the real graph.** The gradient-distilled
   L4 score selects 256 of 1024 tokens and lands statistically level with the
   official EADP map inside a genuinely different computation graph (mid-LLM
   prune vs pre-LLM prune), on both position policies, with normal generation
   behaviour once the decode is correct. The M1→M2 transfer that S2-C0/S2-C1
   could only proxy — and which S2-C0's *forward-attention* proxy failed — holds
   for the trained LOCAL-MLP.
2. **"Level" hides a real, reproducible trade.** Against B1, GDEP gives back
   ~12 points of OCRBench (50.0/50.7 vs 62.0) and takes back ~12 of TextVQA
   (71.4/72.3 vs 59.4) — TextVQA nearly at the *unpruned* 73.6. The pattern is
   the LOCAL-MLP's token-local saliency profile: it keeps the small text
   glyphs the official importance map drops, and loses the layout/global
   context the dense text-guidance preserves. A macro point estimate cannot
   see this; the paired per-benchmark deltas in `m2_amend_report.json` can,
   and any successor that inherits the score inherits the profile.
3. Seed noise is ±2 macro even at n=960 (P′ s0 at 57.63 is a −3.06 outlier vs
   B1), the same order as the whole B1-vs-B2 gap. Every accuracy claim below
   is made on seed means with paired CIs, never on one seed.

---

## 4. Performance

Batch 1, A40, SDPA, 20 warm-up + 100 measured iterations per arm, every
measured region bracketed by both CUDA events and synchronised wall-clock;
decode benchmark fixed at 32 tokens (`ignore_eos`, so every arm steps exactly 32
times). Input: one fixed held-out instance (`TextVQA_VAL_0`, 1058 prompt
tokens). TTFT here is the **whole per-request cycle** — image preprocessing,
vision tower, whatever token machinery the arm has, and the LLM pass to the
first token. All eight arms measured in one session window after the G-F fix.

| arm | TTFT median (ms) | TTFT P90 | std | peak mem (GB) | 32-tok decode | TPOT (ms) | req/s | TTFT vs B1 |
|-----|-----------------:|---------:|----:|--------------:|--------------:|----------:|------:|-----------:|
| B0 full-1024 | 394.4 | 404.7 | 5.3 | 16.61 | 1287.5 | 40.39 | 2.54 | +25.9 % |
| B1 EADP facility@256 | 313.3 | 321.2 | 6.2 | 16.54 | 1282.9 | 40.27 | 2.82 | — |
| B2 EADP block8@256 | 275.6 | 278.7 | 2.0 | 16.54 | 1279.7 | 40.30 | 3.53 | −12.0 % |
| C0 GDEP n=240 topk | 291.8 | 296.6 | 4.2 | 16.54 | 1289.5 | 40.59 | 3.46 | −6.9 % |
| **C1 GDEP n=960 topk** | **286.7** | 290.6 | 2.7 | 16.54 | 1284.1 | 40.35 | 3.52 | **−8.5 %** |
| C2 GDEP n=960 block8 | 299.1 | 308.4 | 6.5 | 16.54 | 1289.0 | 40.51 | 3.37 | −4.5 % |
| C3 GDEP n=960 facility | 323.8 | 332.5 | 9.8 | 16.54 | 1283.1 | 40.41 | 3.09 | +3.4 % |
| C1-R (RENUMBER control) | 285.5 | 289.6 | 3.2 | 16.54 | 1282.4 | 40.26 | 3.53 | −8.9 % |

### 4.1 Stage decomposition (mean ms)

| arm | preproc | vision | EADP score | selector | L0–L4 @1058 | scorer | compact | L5+ @290 | LLM @1058 |
|-----|--------:|-------:|-----------:|---------:|------------:|-------:|--------:|---------:|----------:|
| B0 | ~81 | ~110 | — | — | — | — | — | — | ~203 |
| B1 | ~81 | ~110 | 1.4 | ~~86~~ **~41** † | — | — | — | — | ~72 |
| B2 | ~81 | ~110 | 1.3 | ~~19~~ **~9** † | — | — | — | — | ~72 |
| C1 | ~81 | ~110 | — | 0.1 | 28.1 | 0.3 | 0.3 | ~63 | — |
| C2 | ~81 | ~110 | — | ~9 | 28.1 | 0.3 | 0.3 | ~63 | — |
| C3 | 81.0 | 110.5 | — | 41.4 | 28.1 | 0.30 | 0.30 | 62.1 | — |

(per-instance stage means; the full-precision per-stage tables live in
`m2_perf.json`. B0's LLM column is the uninterrupted 36-layer pass at 1058
positions; GDEP's is the split 5+31. **†** the B1/B2 selector cells as first
published doubled the internal pruner timer into the event bracket for the
same call (§2.1, §4.3); the struck values were artefacts, the bold values are
the single-count windows, and only the pre-LLM arms were affected.)

### 4.2 What this performance picture says

1. **The fixed cost dominates and pruning does not touch it.**
   Preprocessing (84 ms) plus vision tower (111 ms) is 195 ms — 49 % of B0's
   TTFT and ~67 % of C1's — and is byte-identical across arms. Any TTFT claim at
   this resolution must be read against that floor; the Stage-1 exchange-rate
   arguments were prefill-only and are more favourable than end-to-end looks.
2. **GDEP's win over the incumbent is real but small end to end:
   −8.5 % TTFT vs B1 in this table, −7.2 % measured paired later (§4.3)** —
   against the pre-registered −15 % bar. And it does not beat B2
   (286.7/285.5 vs 275.6 ms here; +9.4…+10.0 ms slower paired): EADP with the
   cheap block8 selector on the pre-LLM path remains faster than a mid-LLM
   prune, because the prune pays 28 ms of full-length prefix that the pre-LLM
   path never pays. The L4 decision point structurally forfeits most of the
   budget it saves, and no selector or scorer tuning inside GDEP recovers it.
3. ~~**Selector timing is load-unstable on this machine**~~ — **this claim was
   wrong and §4.3 replaces it.** The 85.8-vs-41.4 ms gap was a timing-window
   double count in the pre-LLM arms (pruner-internal CudaTimer + an outer CUDA
   bracket summed into one field), not host drift: paired measurement puts B1's
   facility at 40.8 ms, in agreement with C3's 41.4 and Stage-1's 42.9, with
   same-arm first/second-half drift ≤ 2.3 ms across the whole run. What the
   claim was reaching for is still true — the −7.1 % vs B1 is bought almost
   entirely at B1's selector cost, not by the prune itself — but it is now a
   paired, 118:2-block measurement rather than a bound read off unstable data.
4. **Memory is a dead axis for pruning here**, as Stage-1 O-B already said:
   16.54 GB peak for every pruned arm against 16.61 unpruned. The KV ledger is
   reported per the cache policy — GDEP early layers transiently allocate the
   1058-token cache (21.4 MB) and compact to 5.88 MB before layer 5; steady
   state is 5.88 + 42.32 = 48.2 MB vs the incumbent's flat 42.3 MB vs the
   hypothetical 155.6 MB unpruned. Weights, not tokens, own the gigabytes.
5. **Decode does not care about pruning at all**: TPOT 40.3–40.6 ms across
   every arm, identical within noise — the per-token weight traffic of an 8B
   model at batch 1 on an A40 is the whole story. Token pruning is a prefill
   technique, and end-to-end latency claims that include long generations
   inherit that.

### 4.3 The paired re-measurement (amendment §7, supersedes the sequential numbers)

The sequential grid above measured each arm in its own block minutes apart,
which is what let a double-counted window masquerade as host drift. The
amendment's harness (`m2_perf_paired.py`) runs all five live arms — B0, B1,
B2, C1-P, C1-R — once per block, randomised order within block, 15 warm-up +
**120 measured blocks**, raw per-block latencies retained, `layer_calls`
accounting asserted every block (no recompute anywhere), and explicit timing
windows (TTFT = sync-to-first-token whole request cycle; model-only = TTFT
minus the CPU image-preprocess window; selector/prune machinery = one single-
counted window per arm). Result:

| arm | TTFT med | p10–p90 | model-only | prune machinery | LLM prefill | decode 32t |
|---|---|---|---|---|---|---|
| B0 | 397.9 | 396.2–404.7 | 312.3 | — | 202.1 | 1276.6 |
| B1 | 310.1 | 308.5–318.5 | 224.2 | 1.4 + 40.8 | 71.9 | 1278.9 |
| B2 | 278.6 | 276.3–285.0 | 192.2 | 1.4 + 8.5 | 72.0 | 1280.0 |
| C1-P | 288.3 | 285.9–294.5 | 202.1 | 28.0 + 0.3 + 0.1 (sel 0.07) | 63.4 | 1281.7 |
| C1-R | 287.8 | 285.7–293.9 | 201.8 | as C1-P | 63.3 | 1280.3 |

Paired within-block contrasts: **C1-P − B1 = −22.50 ms [−23.16, −20.79]**
(118/120 blocks), **C1-R − B1 = −22.95 [−23.77, −20.95]** (118/120), but
**C1-P − B2 = +9.98 [+7.61, +11.26]** and **C1-R − B2 = +9.35 [+7.17,
+11.04]** — both GDEP arms *lose* to the cheap-selector incumbent;
B1 − B2 = +31.89 [29.68, 33.04]. Same-arm first/second-half drift ≤ 2.3 ms;
TTFT − Σ(stages) = +3.0…+3.6 ms for **every** arm, so the component accounting
now explains itself end to end.

---

## 5. The Pareto

The pre-registered criteria (§5 of the preregistration), evaluated on the
valid arms with paired bootstrap CIs (`m2_amend_report.json`; the original
grid's C0/C1/C2 are marked invalid and never enter this table):

| criterion (frozen) | requirement | C1-P | C1-R |
|---|---|---|---|
| primary | CI lower > −1.0 pt **and** TTFT ≥ 15 % below B1 | −7.54 / −7.4 % ✗ | −7.36 / −7.4 % ✗ |
| strong-compression | ≤ 2.0 pt drop **and** (≥ 25 % TTFT **or** ≥ 20 % memory) | drop ✗ (CI spans −7.5)…TTFT ✗, mem ✗ | ✗ (same) |
| accuracy-led | CI lower > 0 vs B1 **and** TTFT ≤ 1.10× B1 | lower = −7.54 ✗ | lower = −7.36 ✗ |
| content-free control (SHUF) | honest arm must beat shuffled control | not re-run (see below) | — |

No GDEP arm clears any bar. Under the pre-registration's own conclusion
ladder the run lands in its final branch — *"no candidate clears a Pareto bar:
GDEP is faster than B1 on TTFT (7.2–7.4 % median, below the 15 % primary
threshold) while its accuracy sits within noise of the incumbent"* — and the
literal P4 antecedent (model-side saving ≤ 0 end to end) does **not** fire
against B1, because the −22.5 ms is a real saving. What the saving is not is
the prune's: decomposed, the L0–L4 prefix costs 28.4 ms and the suffix
saving repays 8.7 ms of it (63.3 vs 72.0); the rest of the edge is B1's
40.8 ms facility selector, which any cheaper incumbent configuration cancels.
Measured against that honest frontier, **B2, GDEP never lands on the
Pareto front at all** — it is behind on speed (by ~10 ms with topk, by ≥ 20 ms
with block8/facility per the first grid's position-independent timing rows:
C2 299.1, C3 323.8) and level-or-below on accuracy. The amendment's directive-
7 rule then settles the reading: corrected accuracy merely equivalent to the
incumbent ⇒ **the L4 pruning point is structurally too late**, and the
research direction is the pre-LLM distillation pivot, not another L4 sweep.

The selector comparison reading A authorised (block8/facility at L4) is
therefore closed analytically rather than with more GPU time: GDEP's own
selector already costs 0.07 ms, so the prefix — not the selection — is what
separates 288 from 278. The SHUF control was stopped as contaminated; the
attribution surrogate evidence for the score is the S2-B shuffled-map control
(significant there) plus §3's reproducible benchmark-profile shift, and a
rebuilt content-free control inside the fixed engine is listed as optional
follow-up in §8, not claimed as evidence here.

---

## 6. Conclusion

**M2 built the thing it promised and the thing answered its question.** One
engine, one manual layer loop, one decode loop, one measured code path for
accuracy and latency; five prefill gates plus G-F, then — after the decode
defect was caught by the interim audit — nine decode gates (HG-1…9) and a
v2-stamped engine whose whole history is hash-separated from v1's. The method
question the stage existed to answer is answered: a gradient-distilled,
524 673-parameter L4 scorer *can* prune at layer 4 inside a real graph and
hold the incumbent's accuracy (60.7–60.9 vs 61.1, CIs spanning zero, both
position policies) — the M1 distillation line transfers. The systems question
is answered too, and it is what kills the configuration: the pruning point
itself is the cost. 28.4 ms of unavoidable full-token prefix buys back 8.7 ms
of suffix, so the only arms that beat B2's 278.6 ms are ones that never pay a
mid-stack pass at all — and B1's 40.8 ms facility selector, the whole source
of GDEP's measured edge over it, is already gone in B2.

The stage's contribution going forward is therefore not GDEP-the-deployable
but three transferable numbers and a direction: (i) the prefix arithmetic that
rules out *any* prune deeper than the vision tower as a TTFT mechanism at
budget 256 on this host; (ii) the score's validated benchmark-profile trade
(TextVQA ≈ unpruned, OCRBench −12 vs the official map — §3.2), which any
pre-LLM successor inherits and should target; (iii) the lesson that a green
gate set on the prefill path says nothing about the decode path, which is why
every later stage harness in this repo now includes generation-level identity
checks. The pre-registration's own fallback question — what would make us
stop — fired as written: accuracy equivalent to the incumbent ⇒ stop sweeping
L4; distil the same utility into the vision-encoder / pre-LLM scorer.

---

## 7. Costs that are not deployment costs

Per the pre-registration, the offline gradient-teacher cost is reported once
and excluded from every deployment figure: 512.3 ms GPU per image
(216.1 forward + 296.2 backward) and 912.0 ms wall per map, 1 170 maps for the
n = 960 fit (450 frozen + 720 extension) — ≈ 10 GPU-minutes (≈ 18 min wall on
this host) for the *training* of the scorer, amortised to zero at inference.
Inference pays:
2 MB of frozen weights (524 673 parameters), 0.30 ms of online scoring,
0.10 ms of compaction, 0.07 ms of selection — the scorer is not where the
method's cost is; the full-token prefix is. The amendment's extra measurement
cost (fix gates ≈ 25 min of GPU, paired grid ≈ 19 min, and the ~5.5 h of the
stopped contaminated grid whose partial work was discarded) is measurement
overhead, not system cost.

---

## 8. What this does and does not establish

**Establishes.** (1) A real mid-stack prune executes correctly end to end —
bit-exact prefill identity at K = 1024, exact position/cache bookkeeping, no
recomputation in all 600 timed arm-calls of the paired grid (and every G-E /
HG-9 instance). (2) The M1
LOCAL-MLP's accuracy transfers into the real engine on both position policies;
this was *not* given — S2-C0's forward-attention proxy failed at 4–8 %
retention, and the trained scorer's 45–57 % carried through. (3) The L4 TTFT
arithmetic, measured paired: the prune cannot out-run the pre-LLM path with
the cheap selector at this budget/model/host. (4) The position policy is not
the accuracy axis (|P′ − R| = 0.21 [−2.57, +2.02]).

**Does not establish.** (1) Any deployable GDEP configuration — all three
frozen bars missed and the architecture's headroom is measured out (§5–6).
(2) Within-engine score-specificity: the SHUF control was stopped contaminated
and not re-run; the surrogate is S2-B's shuffled-map control plus the
reproducible profile shift, not a fresh CI. (3) The profile trade's
significance — +12/−12 benchmark deltas sit in ±7–8 pt paired CIs at n = 150;
they survive across 6 runs × 2 policies but have not been tested with a
wider bank. (4) Anything at budgets other than 256/1024, at pruning depths
other than L4, for other models, or batch > 1 (the compaction is
per-sequence; batched GDEP remains unimplemented and unmeasured). (5) The
peak-memory and TPOT axes: flat here, but that is a statement about this
8B/A40/batch-1 regime, not a general law.

**Open by decision, not by oversight:** the pre-LLM distillation successor
(carries the score, drops the prefix — the natural M3 candidate); a rebuilt
content-free control inside the fixed engine if score attribution is ever
needed again; and the B2-as-baseline convention (§5) that future stages
should pre-register explicitly.

---

## 9. Reproduce

```bash
cd /media/disk2/YZX/research/EADP/Qwen_vl
bash scripts/discovery/run_m2.sh     # the ORIGINAL stage: gates -> perf -> accuracy
```

The amendment sequence, in its actual order (each step's artefact is the
input to the decision documented above):

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
cd /media/disk2/YZX/research/EADP/Qwen_vl

# 0. the original grid ran to C2|s2 and was stopped; its file keeps the
#    contaminated records marked INVALID_DUE_TO_FROZEN_ROPE_POSITION.

# 1. valid pre-fix control (RENUMBER; ran on the v1 build, justified by HG-8)
python -u scripts/discovery/m2_accuracy.py --arms C1-R --seeds 0 1 2 \
    --resume --tag m2_accuracy

# 2. post-fix decode gates (HG-1..HG-9) and full prefill-gate re-pass on v2
python -u scripts/discovery/m2_fixgates.py
python -u scripts/discovery/m2_correctness.py --gates A B C D E \
    --tag m2_correctness_v2

# 3. corrected PRESERVE arm (new key + engine-versioned cfg hash)
python -u scripts/discovery/m2_accuracy.py --arms C1-P --seeds 0 1 2 \
    --resume --tag m2_accuracy

# 4. paired interleaved timing (120 blocks, randomised within-block order)
python -u scripts/discovery/m2_perf_paired.py --blocks 120 --warmup-blocks 15

# 5. valid-arms-only report + frozen-decision reading + figure
python scripts/discovery/m2_amend_report.py --arms C1-P C1-R \
    --proxy-baseline-macro 57.0
python scripts/discovery/m2_amend_figure.py
```

Deliverables: `docs/scoring_search_m2_system_prereg.md` (frozen, untouched by
the amendment), `docs/scoring_search_m2_system_interim.md` (the audit that
found the defect), `docs/scoring_search_m2_system_amendment.md` (this stage's
invalidation + fix + re-measurement record),
`outputs/discovery/m2_accuracy.json` (valid + marked-invalid records),
`m2_fixgates.json`, `m2_correctness_v2.json`, `m2_perf_paired.json` (raw
per-block latencies), `m2_amend_report.json/.md`,
`figures/m2_pareto.png`; the original grid's `m2_correctness.json`,
`m2_perf.json` (its event-stage columns corrected by §4.3's footnote), and
the stopped-grid log `m2_run_progress_attempt6_stopped.log`.
`m2_consolidate.py` (the original grid's evaluator) is validity-filtered but
otherwise superseded by `m2_amend_report.py`.
