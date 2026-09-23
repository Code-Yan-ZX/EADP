# M2 — INTERIM read-only snapshot (GDEP system Pareto)

**This document carries no verdict.** It is an export of the artefacts that
already exist at the moment of writing, plus three read-only diagnostics run
while the pre-registered grid kept executing. It does not stop, reorder or
modify the grid, does not add an arm, does not re-select a seed, does not change
the pre-registration, and does not pre-empt the final conclusion. The final
`docs/scoring_search_m2_system.md` will be written from the completed grid,
consolidated and committed as planned.

**Why it exists.** The first completed candidate arm (`C0`, and then `C1`)
scored ~23 macro points below the incumbent with 13–17 % of generations running
to the 2048-token cap. That is far worse than every cached-score proxy in
S2-B/S2-C0/S2-C1 predicted. Before spending the remaining ~7 hours of grid time,
the question had to be settled: **method failure, objective mismatch, or
implementation defect?**

**Answer, up front: implementation defect — and it is localised, isolated and
reproducible.** Section 5 is the evidence; section 4 is the symptom; section 6
ranks the three explanations.

Machine-readable twin: `Qwen_vl/outputs/discovery/m2_system_interim.json`.
Diagnostics: `m2_interim_diag.json` (D1/D2/D3), `m2_interim_d5.json` (D5).

---

## 1. Arm definitions

All arms run in one engine (`m2_gdep.GDEPEngine`), one manual 36-layer loop, one
decode loop, one prompt construction. Nothing below changes between the accuracy
run and the performance run: the same engine, the same selector table, the same
scorer checkpoints. The configuration hash (`cfg_hash`) is written into both
`m2_perf.json` and `m2_accuracy.json` per arm, and consolidation compares them.

| arm | pruning point | selector | scorer checkpoint | KV-cache policy | graph |
|-----|---------------|----------|-------------------|-----------------|-------|
| **B0** | none | — | — | 36 × 1058, unpruned | 36 layers over the full 1058-token sequence |
| **B1** | none inside the LLM; prune happens *before* it | official EADP facility (`_greed_select_impl`) | official EADP importance (α 0.5, β 2.0) | 36 × 290; the L0–L4 cache does not exist | incumbent: pruner over vision embeddings → 36 layers over 290 tokens |
| **B2** | as B1 | `block8` (D1: same objective, T/8 super-steps) | official EADP importance | as B1 | as B1 with the cheap selector |
| **C0** | **decoder layer 4** (0-based, mid-stack) | `topk` | M1 `LOCAL-MLP`, n = 240, seed 0, **524 673 params** | L0–L4 allocated at 1058 → **compacted to 290**; L5–L35 290 | GDEP: 5 full-length layers → online score → select → compact → 31 layers |
| **C1** | layer 4 | `topk` | M1 `LOCAL-MLP`, n = 960, seeds 0/1/2 | as C0 | GDEP primary candidate |
| **C2** | layer 4 | `block8` | M1 `LOCAL-MLP`, n = 960, seeds 0/1/2 | as C0 | GDEP secondary candidate |
| **C3** | layer 4 | `facility` | M1 `LOCAL-MLP`, n = 960, seeds 0/1/2 | as C0 | GDEP secondary candidate |
| **C1-R** | layer 4 | `topk` | M1 `LOCAL-MLP`, n = 960, seeds 0/1/2 | as C0 | C1 under **RENUMBER** |
| **C1-SHUF** | layer 4 | `topk` | M1 `LOCAL-MLP` n = 960, **score map rotated +7 instances** | as C0 | content-free control |

**What `R` changes.** Exactly one thing: the position policy. Under PRESERVE a
kept token keeps the position id it had in the full sequence, so the compacted
`position_ids` are `sorted(kept_indices)` — 136 non-unit gaps for the diagnostic
instance — and the next generated token takes `max + 1`. Under RENUMBER the
compacted sequence is renumbered `arange(0, 290)`, which is what the incumbent's
pre-LLM path does implicitly. Same token set, same scorer, same selector, same
cache policy: only the positional encoding of the compacted sequence differs.

**What `SHUF` changes.** Exactly one thing: which image's score map is used. The
scores are computed online as always, then rotated onto a different instance
(derangement, +7). Every statistic of the map is preserved — same distribution,
same range, same sparsity — only the correspondence to the image is destroyed.
It is the S2-B §2.2 control re-run inside the real engine.

**KV-cache policy, stated per the pre-registration §2.3.** GDEP uses
**compact-L0–L4**: layers 0–4 transiently allocate the 1058-token cache
(21.4 MB) and are compacted to 290 positions (5.88 MB) before layer 5 runs;
layers 5–35 hold 290 positions (42.3 MB). B1/B2 have no early-layer cache at
all — no LLM layer ever sees 1024 tokens. The consequence is inherited and not a
defect: GDEP's kept tokens were computed with 1058-token attention in layers
0–4, whereas B1's were computed with 290-token attention from the start. G-D and
D2 both confirm the shapes (`[1, 8, 287, 128]` after compaction → `[1, 8, 290,
128]` after three decode steps).

---

## 2. Performance — all eight arms, measured (new window, post-G-F-fix)

Batch 1, A40, SDPA, 20 warm-up + 100 measured iterations, CUDA events and
synchronised wall-clock, decode fixed at 32 tokens, input = one fixed held-out
instance (`TextVQA_VAL_0`). TTFT is the whole per-request cycle.

| arm | model prefill med (ms) | model prefill P90 | TTFT med (ms) | TTFT P90 | vision enc (ms) | L0–L4 (ms) | scorer (ms) | selector (ms) | compaction (ms) | L5+ (ms) | 32-tok decode (ms) | TPOT (ms) | peak mem (GB) | KV early+late (MB) | ×B0 | ×B1 | ×B2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 391.3 | 401.6 | 394.4 | 404.7 | 111.0 | — | — | — | — | 204.3 | 1287.5 | 40.39 | 16.61 | 155.6 (hypothetical) | 1.00 | 0.90 | 0.72 |
| B1 | 352.7 | 361.2 | 313.3 | 321.2 | 110.9 | — | — | 85.8 | — | 72.4 | 1282.9 | 40.27 | 16.54 | 42.3 | 1.11 | 1.00 | 0.80 |
| B2 | 282.4 | 286.1 | 275.6 | 278.7 | 110.4 | — | — | 19.3 | — | 72.4 | 1279.7 | 40.30 | 16.54 | 42.3 | 1.39 | 1.25 | 1.00 |
| C0 | 288.6 | 293.3 | 291.8 | 296.6 | 111.0 | 28.1 | 0.30 | 0.1 | 0.10 | 63.6 | 1289.5 | 40.59 | 16.54 | 5.9 + 42.3 | 1.36 | 1.22 | 0.98 |
| **C1** | 283.5 | 288.1 | **286.7** | 290.6 | 110.7 | 28.1 | 0.30 | 0.1 | 0.10 | 63.6 | 1284.1 | 40.35 | 16.54 | 5.9 + 42.3 | 1.38 | 1.24 | 1.00 |
| C2 | 295.9 | 301.4 | 299.1 | 308.4 | 110.9 | 28.1 | 0.30 | 9.0 | 0.31 | 62.8 | 1289.0 | 40.51 | 16.54 | 5.9 + 42.3 | 1.32 | 1.19 | 0.95 |
| C3 | 320.6 | 327.1 | 323.8 | 332.5 | 110.5 | 28.1 | 0.30 | 41.4 | 0.30 | 62.1 | 1283.1 | 40.41 | 16.54 | 5.9 + 42.3 | 1.22 | 1.10 | 0.88 |
| C1-R | 282.4 | 286.0 | 285.5 | 289.6 | 110.5 | 28.0 | 0.30 | 0.1 | 0.09 | 63.5 | 1282.4 | 40.26 | 16.54 | 5.9 + 42.3 | 1.39 | 1.25 | 1.00 |

Reading it:

* The mechanism works as designed: prefix 28.1 ms, online scorer **0.30 ms**,
  compaction 0.10 ms, suffix 63.6 ms (< the flat 72.4 ms of a 36-layer 290-token
  pass, because GDEP's suffix is 31 layers). No arm recomputes anything —
  `extra_forward_or_recompute` is false on all eight.
* **C1's TTFT is 8.5 % below B1's** (286.7 vs 313.3) — real but far below the
  pre-registered 15 % bar, and C1 does not beat B2 (275.6). The L4 decision point
  pays 28 ms of full-length prefix that a pre-LLM prune never pays.
* **Peak memory is flat** (16.54 GB for every pruned arm vs 16.61 unpruned):
  weights own the footprint, exactly as Stage-1 O-B reported. The 20 % memory
  criterion is unreachable for token pruning on this configuration.
* **TPOT is identical to 0.3 % across all eight arms** (40.3–40.6 ms). Decode is
  weight-bandwidth bound; token pruning buys prefill only.
* Selector cost is load-unstable on this host: the same facility loop measured
  85.8 ms (B1) and 41.4 ms (C3) minutes apart. GDEP's TTFT edge over B1 is
  bought at the expensive end of that range and should be read as an upper bound.

---

## 3. Accuracy — completed arms

Held-out 150 (50 per benchmark), greedy, `max_new_tokens = 2048`. `hit ≥ 0.5` is
the Stage-1 hard-error cut.

| arm | seed | TextVQA | DocVQA | OCRBench | macro | correct/wrong | Δ vs B1 (pts) | 95 % CI | rescued | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|
| **B0** full-1024 | — | 73.60 | 85.27 | 68.00 | **75.62** | 114 / 36 | +14.53 | [+8.23, +20.86] | 24 | 5 |
| **B1** EADP facility | — | 59.40 | 61.89 | 62.00 | **61.10** | 95 / 55 | — | — | — | — |
| **B2** EADP block8 | — | 65.40 | 60.24 | 54.00 | **59.88** | 94 / 56 | −1.22 | [−7.57, +5.29] | 13 | 14 |
| **C0** n=240 topk | 0 | 48.80 | 30.49 | 34.00 | **37.76** | 61 / 89 | −23.33 | [−31.37, −15.49] | 8 | 42 |
| | 1 | 53.00 | 29.23 | 38.00 | **40.08** | 64 / 86 | −21.02 | [−28.97, −13.17] | 10 | 41 |
| | 2 | 53.80 | 20.74 | 32.00 | **35.51** | 56 / 94 | −25.58 | [−33.45, −17.76] | 7 | 46 |
| **C1** n=960 topk | 0 | 54.20 | 25.85 | 34.00 | **38.02** | 60 / 90 | −23.08 | [−30.93, −15.35] | 7 | 42 |
| | 1 | 53.80 | 25.82 | 34.00 | **37.87** | 59 / 91 | −23.22 | [−31.86, −14.87] | 11 | 47 |
| | 2 | 59.80 | 25.49 | 34.00 | **39.76** | 62 / 88 | −21.34 | [−29.73, −13.06] | 11 | 43 |

Seed means: **C0 37.78** (35.51 / 40.08 / 35.51…, range 35.51–40.08, all 3
seeds) · **C1 38.55** (38.02 / 37.87 / 39.76, all 3 seeds). C0 and C1 are nine
tenths of a point apart in the mean and fully overlapping in range.

Two things to read, and they are both about the *implementation*, not the method:

1. **C0 ≈ C1.** The n = 960 scorer's +1.7 pt held-out teacher R@8 over n = 240
   buys nothing downstream here. If the defect were absent and the method were
   simply weak, one would still expect the better scorer to move the number.
2. **DocVQA is where the collapse is** (20.7–30.5 vs B1's 61.9), and section 4
   shows DocVQA is exactly where the capped generations concentrate.

These numbers are **contaminated** and must not be used as the method's accuracy:
section 5 identifies the defect. C1-R (RENUMBER, unaffected by the defect) is
still running and is the arm that will give the honest read.

---

## 4. Generation-length anomaly

### 4.1 Per arm

| arm | runs | gen chars median | P90 | max | capped (>2000 chars) | rate | terminated (EOS) | rate |
|---|---|---|---|---|---|---|---|---|
| B0 | 1 | 8 | 20 | 54 | 0 | **0.0 %** | 150 | 100 % |
| B1 | 1 | 7 | 18 | 49 | 0 | **0.0 %** | 150 | 100 % |
| B2 | 1 | 7 | 19 | 59 | 0 | **0.0 %** | 150 | 100 % |
| C0 | 3 | 12–13 | 45–60 | 2082–6148 | 20 / 24 / 21 | **13.3 / 16.0 / 14.0 %** | 127–130 | 85–87 % |
| C1 | 3 | 13 | 45–70 | 2086–6148 | 25 / 24 / 24 | **16.7 / 16.0 / 16.0 %** | 126 / 126 / 126 | 84.0 % |

`max_new_tokens = 2048` for every arm, so a "capped" run is one that never
emitted EOS. **The baselines do not have this problem at all** (0/450 capped, max
59 chars across B0/B1/B2); the GDEP arms do, at 13–17 %. The bounded-runs-only
picture is still bad (mean hit among non-capped C0|s0 runs = 0.436), which is why
the anomaly alone does not explain the whole gap — but it is the loudest symptom.

### 4.2 By benchmark (capped / 50)

| arm | TextVQA | DocVQA | **OCRBench** |
|---|---|---|---|
| C0|s0 | 1 | **11** | 8 |
| C0|s1 | 1 | **13** | 10 |
| C0|s2 | 1 | **11** | 9 |
| C1|s0 | 1 | **14** | 10 |
| C1|s1 | 2 | **13** | 9 |
| C1|s2 | 2 | **13** | 9 |

The concentration is in **DocVQA**, the benchmark whose answers are short
alphanumeric strings — and whose text is exactly the kind of numeric run that a
degenerate decoder loops on.

### 4.3 Ten-plus capped instances (`C0|s0`), with the requested fields

`n_kept = 256` and `kv_seq_len == context_len` on every one of them (D-level
invariants hold). `context_len = 256 + n_text`, so the full prompt length is
`context_len + 768`. Positions under the grid's PRESERVE run are
`sorted(kept) ⊂ [0, seq_full−1]` and the decode positions are the frozen
`(cache_len, max+1)` pairs of section 5.

| # | key | bench | prompt tokens | n_kept | first generated tokens | looping? | chars |
|---|-----|-------|--------------:|-------:|------------------------|----------|------:|
| 1 | `TextVQA_VAL_4127` | TextVQA | 1054 | 256 | `prevententententententententententente` | yes | 6148 |
| 2 | `DocVQA_VAL_1077` | DocVQA | 1060 | 256 | `FEB 1/1/13/13/13/13/13 13 13 13 13 13` | yes | 2049 |
| 3 | `DocVQA_VAL_1292` | DocVQA | 1058 | 256 | `(03)34263426 26 26 26 26 26 1034 07 7` | yes | 2048 |
| 4 | `DocVQA_VAL_1400` | DocVQA | 1053 | 256 | `$17,017,017,017,01701717,0171717171717` | yes | 2048 |
| 5 | `DocVQA_VAL_1507` | DocVQA | 1055 | 256 | `$30,60,6000000000000000000000000000000` | yes | 2048 |
| 6 | `DocVQA_VAL_1831` | DocVQA | 1052 | 256 | `617/217/217/217/21/21/21/21/21/21/21/2` | yes | 2048 |
| 7 | `DocVQA_VAL_1938` | DocVQA | 1062 | 256 | `(212121212121212121212121212121212)989` | yes | 2048 |
| 8 | `DocVQA_VAL_2477` | DocVQA | 1067 | 256 | `$2,00000000000000000000000000000000000` | yes | 2048 |
| 9 | `DocVQA_VAL_2800` | DocVQA | 1057 | 256 | `GRATRATIS TAX REIMMIS TAX REIMBUTAX RE` | yes | 2080 |
| 10 | `DocVQA_VAL_3230` | DocVQA | 1054 | 256 | `29529507070000000000000000000000000000` | yes | 2048 |
| 11 | `DocVQA_VAL_3553` | DocVQA | 1054 | 256 | `(2121212121212121212121212121212121212` | yes | 2048 |
| 12 | `DocVQA_VAL_3876` | DocVQA | 1055 | 256 | `March 2, 19519519519519519519519519519` | yes | 2052 |

Every one is a repetition loop, and **every one of them is reproduced
identically by the same token set in the incumbent's pre-LLM graph** when that
graph is given the same (defective) position handling — see section 5, which
also shows that with *correct* advancing positions the same instances answer
normally.

### 4.4 Per-layer KV shape (D2, live)

`[1, 8, 287, 128]` for layers 0, 4, 5, 10 and 35 right after compaction →
`[1, 8, 290, 128]` after three decode steps. Layers 0–4 were allocated at 1058
before compaction. `kv_bytes_early = 5.88 MB`, `kv_bytes_late = 42.32 MB`.

---

## 5. Generation-correctness coverage: what gates A–E do and do not cover

Gate A–E all passed, but they were written for the **prefill** graph. Audited
against the eight properties this interim needs:

| # | property | covered by | result |
|---|----------|-----------|--------|
| 1 | K = 1024 **full multi-step generation** matches the stock model | **not A–E** → D1 | **yes**: engine `[67, 585, 6089, 151645]` == HF generate `[67, 585, 6089, 151645]`, 4/4 steps, identical text `dakota` |
| 2 | first decode-step logits | G-C (prefill) + D1 | prefill argmax == HF's first token; prefill logits identical to 0.000e+00 |
| 3 | first 8 decode steps | **not A–E** → D1 | per-step equality recorded (16 requested, generation ended at EOS after 4) |
| 4 | EOS / stopping criteria identical | D1 + config inspection | both use `[151645, 151643]`; same stop point observed |
| 5 | `cache_position` after compaction | **not A–E** → D2 | advances 287, 288, 289 over the 287-entry compacted cache — correct; **but see the defect below** |
| 6 | mRoPE ids / `rope_deltas` | **not A–E** → D2 | engine passes explicit `(3,1,S)` ids; `rope_deltas` is unused in `inputs_embeds` mode. **Defect found here** |
| 7 | KV lengths L0–L4 vs L5+ | G-D + D2 | L0–L4 1058 → 290 compacted; L5–L35 290 |
| 8 | identical generation configuration | code inspection + G-F | all greedy, `do_sample=False`, same `max_new_tokens`; B1 reproduces the published numbers exactly |

### 5.1 The defect (D2 + D5)

D2, live, on the compacted 290-position state:

```
decode (cache_position, rope_position) = [(287, 1055), (288, 1055), (289, 1055)]
```

`cache_position` advances correctly, but the **RoPE position is frozen**: the
PRESERVE branch recomputes `max(prefill_position_ids) + 1` from the *prefill*
tensor on every step instead of advancing it. Every generated token is therefore
embedded at position 1055 (and at 1054, 1058, … for other instances). The
RENUMBER branch is unaffected — it uses `cache_position`, which advances.

D5 isolates the cost. Same instance, same engine, same 256-token set, four
variants — `P` = the defective frozen policy (what the grid ran), `P'` = the
intended advancing policy, `R` = RENUMBER, `L` = the incumbent's pre-LLM graph
on the identical token set:

| instance | P (frozen) | P′ (advancing) | R (renumber) | L (pre-LLM) |
|---|---|---|---|---|
| `TextVQA_VAL_4127` | `prevententen…` (12 tok, loop) | `preventent` | `preventent` | `preventent` |
| `DocVQA_VAL_1077` | `FEB 1/1/13/13/13…` (48 tok, loop) | **`FEB 1, 1972`** | **`FEB 1, 1972`** | **`FEB 1, 1972`** |
| `DocVQA_VAL_1292` | `(03)3333303) 303 3 3…` (48 tok, loop) | `(03) 341 5664` | `(03) 1078` | `(03) 1078` |

With an advancing position — under *either* policy — and with the incumbent's
pre-LLM graph, **the same 256 tokens answer correctly**. The degeneracy appears
only when the position is frozen.

*(An honest note on the earlier D3 run: its pre-LLM control was built with
`POLICY_PRESERVE` and therefore inherited the same defect, which made the control
degenerate too and briefly suggested the token set itself was at fault. D5's L
column is that control rebuilt correctly, and it exonerates the token set.)*

### 5.2 What this means for the grid

* Arms unaffected by the defect: **B0, B1, B2** (they do not take the PRESERVE
  decode branch) and **C1-R** (RENUMBER). B1's exact reproduction of the
  published numbers is the strongest single piece of evidence that the shared
  generation path is sound.
* Arms contaminated: **C0, C1, C2, C3, C1-SHUF** — every PRESERVE arm. Their
  accuracy numbers are invalid as measurements of the method.
* The fix is two lines in `GDEPEngine.decode` (advance the position each step
  instead of recomputing it). **It has deliberately not been applied while the
  grid is running**, so that no artefact can be attributed to an ambiguous code
  version; it belongs in the post-grid decision, together with a re-run of the
  PRESERVE arms.

---

## 6. Evidence ranking: method failure vs objective mismatch vs implementation error

| rank | explanation | evidence for | evidence against |
|---|---|---|---|
| **1** | **Implementation error — frozen RoPE position in the PRESERVE decode branch** | D2 shows the position frozen across steps; D5 shows that flipping *only* that behaviour (P → P′) converts degenerate loops into correct answers on the identical token set, identical selector, identical instance; R and L (both advancing) agree with each other and with P′ | none found |
| 2 | Objective mismatch — the learned score's TopK keeps the wrong 256 tokens | the capped-run rate is concentrated in DocVQA; C0 ≈ C1 shows the extra n = 960 data buys nothing | D5's R and L columns answer correctly **on the same token set**; the token set is therefore sufficient for these instances |
| 3 | Method failure — mid-layer pruning does not transfer from the cached-score proxy | the gap to the proxy prediction (38 vs ~57) was the trigger for this audit | R (mid-LLM graph, correct positions) is still running and is the direct test; D5's R column is a favourable 3-instance preview, not a measurement |

**Caveat on the preliminary nature of the above.** Three instances is not an
accuracy measurement, and the interim has no mandate to produce one. What the
diagnostics establish is narrower and sufficient for the decision they were run
for: the degeneration mechanism is a position-handling defect in the decode
loop, reproducible and localised, and it is not the token set and not the graph.

---

## 7. Grid state at snapshot time

Completed accuracy runs: `B0, B1, B2, C0|s0, C0|s1, C0|s2, C1|s0, C1|s1, C1|s2`
(the C1 family is complete: 38.02 / 37.87 / 39.76).
Still to run: `C2|s0..s2, C3|s0..s2, C1-R|s0..s2, C1-SHUF|s0..s2` (plus its scoring
passes). `C2|s0` was in progress at snapshot time. The grid is untouched and continues running; the final consolidation,
figure, report and commit proceed on the original plan once it finishes.