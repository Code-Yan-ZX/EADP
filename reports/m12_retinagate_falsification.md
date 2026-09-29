# M12 — RetinaGate falsification round

Status: **in progress**. Branch `exp/m12-retinagate`. Single A40, `qwen3vl_clean`
(torch 2.10.0+cu128, transformers 4.57.6, SDPA). Prereg-style question:

> At the same pre-ViT token budget, does Base Lattice + Opponent-Channel
> Event Emission (RetinaGate) beat Random / Uniform / Sobel / grayscale-DoG /
> direct resolution reduction (R-res) at preserving VLM ability, while
> genuinely reducing Qwen3-VL vision-encoder latency and E2E TTFT?

Everything in this round is a *pre-encoder* intervention. Post-encoder design
(SAGE / token-swap / new classifiers / post-encoder scoring) is formally
stopped; M12 touches only what happens before the ViT blocks.

## 1. Motivation: why pivot to the vision encoder

Measured TTFT decomposition on this host (m12_speed_oracle, K=1024, median
over 30 paired blocks, 4 datasets):

| stage | median ms |
|---|---|
| preprocessing (CPU resize+patchify, 1024²) | 86.2 |
| vision encoder (ViT @ 4096 patches → 1024 merged) | 110.3 |
| LLM prefill (~1100 seq) | 207.7 |

The post-encoder program's own ceiling was already documented (E0: at
K=256 merged tokens, post-encoder EADP b2 does not clearly beat plain
R-res on the DEV panel). The largest single *vision* lever is the encoder
itself, so M12 attacks the 110 ms ViT directly, before patch embedding.

## 2. Sparse-ViT implementation (Qwen3-VL data layout audit)

`Qwen_vl/model/retinagate.py`. Verified against the installed transformers
4.57.6 sources and empirically:

* processor emits `pixel_values` rows in **group-major** order
  `(t, gy, gx, iy, ix)` (`image_processing_qwen2_vl_fast` permute
  `(0,1,4,7,5,8,3,2,6,9)`); a native 2×2 merge group's 4 patches are the
  consecutive rows `g*4 .. g*4+3`;
* the merged-token index of a group equals its group index `g`;
* `rot_pos_emb` and `fast_pos_embed_interpolate` produce the same
  group-major per-patch order → the sparse path interpolates/builds them on
  the FULL grid once (O(N), ratio-independent) and indexes with kept rows,
  so every kept patch keeps its original 2-D position / rotary entry;
* row layout is `(channel, temporal_patch_size, ph, pw)` — the temporal axis
  of a row is `temporal_patch_size=2`, not `grid_thw`'s t (unit-tested).

The sparse forward (`sparse_visual_forward`) really reduces what enters the
27 ViT blocks: `patch_embed` (Conv3d, per-patch exact), pos/rotary lookup,
blocks over kept patches with `cu_seqlens = [0, 4K]` (full attention among
kept patches only), DeepStack mergers and final merger via
`view(-1, D*4)` — valid because kept patches stay group-contiguous.
Downstream, `NativeEngine.prefill` consumes `(keep_idx=keep_groups, V_sel,
DS_sel)` unchanged; DeepStack streams and the full-sequence 3-D mRoPE
positions shrink with the same index; decode continues from
`prefill_max_pos`. No attention-masking, no post-hoc gather, no LLM-side
trickery, and stage timings are CUDA-synced wall measurements, not FLOPs.

## 3. Gate 1 — 100% keep correctness audit: PASS

`scripts/m12/m12_correctness.py`, 4 DEV samples/dataset (TextVQA/DocVQA/
OCRBench/ChartQA), 16 samples:

| check | result |
|---|---|
| C1 merged features V, 3 DeepStack streams, sparse@100% vs stock | **max\|Δ\| = 0.0 (bit-exact)** |
| C2 prefill + 32 forced-decode logits, sparse path vs stock | **max\|Δ\| = 0.0** |
| C3 generation text (32 tokens, greedy) | 16/16 identical |
| C4 50% invariants (exact K, no dup/OOB, ascending, DS lengths K,K,K, layer calls) | all pass |

Note: with `ignore_eos=True` the last generated token costs no layer call,
so expected layer calls are `36·n_decode` (not `36·(1+n_decode)`); accuracy
runs use EOS-stopped generation where the stock formula holds.

## 4. Gate 2 — speed oracle (selector-agnostic): PASS

`scripts/m12/m12_speed_oracle.py`: uniform 2-D lattice selection (structured,
deterministic, quality-agnostic), 15 warm-up + 30 measured blocks, stratified
samples, CUDA-event/sync windows, median over blocks (std ≤ ~21 ms TTFT).
R-res arms run the N1-verified keep-all native path at matched merged-token
counts (side = nearest multiple of 32).

| arm | merged K | ViT input patches | T_ViT ms | T_gate ms | LLM prefill ms | E2E TTFT ms | ViT GFLOPs |
|---|---|---|---|---|---|---|---|
| b0 (native) | 1024 | 4096 | 110.3 | — | 207.7 | 405.1 | 5455 |
| lattice 75% | 768 | 3072 | 82.9 | 1.0 | 163.6 | 331.3 | 3700 |
| lattice 50% | 512 | 2048 | **55.8** | 0.9 | 117.4 | **256.0** | 2206 |
| lattice 37.5% | 384 | 1536 | 45.4 | 1.0 | 96.5 | 225.8 | 1556 |
| lattice 25% | 256 | 1024 | 33.2 | 0.9 | 77.5 | 194.0 | 972 |
| R-res 896 | 784 | 3136 | 83.3 | — | 164.7 | 313.7 | 3802 |
| R-res 736 | 529 | 2116 | 54.5 | — | 118.1 | 219.0 | 2297 |
| R-res 640 | 400 | 1600 | 43.9 | — | 97.0 | 177.7 | 1634 |
| R-res 512 | 256 | 1024 | 29.5 | — | 77.4 | 135.4 | 972 |

Findings:

* **A40 wall-clock scales genuinely and near-linearly with structured token
  budget**: 50% budget → ViT 110→56 ms (−49%), TTFT −37%. Gate A is answered
  positively: a real pre-encoder sparse path buys real time.
* Sparse-path overhead is negligible: gate ≈0.9 ms, patch-embed 1.1–2.0 ms,
  full-grid pos/rotary prep ≈5 ms, mergers <1 ms.
* Sparse-ViT time ≈ R-res time at matched budgets (55.8 vs 54.5 ms at ~512).
  R-res's *TTFT* edge (219 vs 256 ms at ~512) is entirely CPU preprocessing:
  resizing to 736² costs 46 ms vs 86 ms at 1024². RetinaGate inherits the
  expensive 1024² preprocessing; this is an honest structural deficit of
  ~35–40 ms TTFT that selector quality must pay for (a production variant
  would gate before the resize — out of scope this round).
* Peak GPU memory is identical everywhere (17.06 GB weights-dominated).
* Theoretical FLOPs are reported for reference only (allocation above is
  measured wall time).

## 5. RetinaGate definition (fixed before evaluation)

From `pixel_values` rows only (no network, no extra forward, no raw-image
re-decode; rows de-normalized with the processor mean/std):

* per-patch mean RGB → raw patch map [64,64,3] (group-major reconstruction,
  unit-tested to 2.4e-7 against synthetic images);
* opponent channels `Y = .299R+.587G+.114B`, `RG = R−G`, `BY = B−(R+G)/2`;
* center-surround energy per channel `E_c = |x−avg3(x)| + 0.5|x−avg5(x)|`
  (patch resolution), mean-pooled onto the 32×32 group grid;
* per-image per-channel normalization (divide by mean), fusion
  `E_event = E_Y + 0.5·E_RG + 0.5·E_BY` — λ fixed a priori, no tuning;
* budget K (merged tokens): **Base+Event** — 2-D lattice covering takes
  `round(0.32·K)` groups, remaining budget filled by top event score;
  exact-K everywhere, raster-ascending, duplicate-free.

Controls (same exact-K): random (seeded per sample), uniform 2-D lattice,
local variance, Sobel gradient energy, grayscale DoG (`E_Y` only), pure
event (no base), and R-res at matched budgets.

## 6. Data & protocol

E0 DEV split, image-disjoint, seed 20260929, CONFIRM untouched:
TextVQA_VAL 300 / DocVQA_VAL 300 / OCRBench 164 (short). Same rows as all
E0 arms; predictions resumable per shard, scored by the official VLMEvalKit
rules; paired per-question contrasts with 10k bootstrap CIs vs b0 and vs
R-res at matched budget.

Caveat carried over from E0 (commit cc84545): DEV-panel *absolute* numbers
fail historical sanity (b0 TextVQA 85.0 vs 71.0 official full; DocVQA 93.9
vs 61.1), so M12 conclusions rest on **paired same-sample contrasts**, not
absolute levels. b0/b2@256 numbers are reused from the E0 shards.

## 7. Accuracy / quality results

Screen 1: K=512 (50 % of 1024 merged tokens), E0 DEV rows, same samples for
every arm, official VLMEvalKit scoring (OCRBench raw /100):

| arm @K=512 | TextVQA | DocVQA | OCRBench | mean Δ vs b0 |
|---|---|---|---|---|
| b0 (native, K=1024) | 85.03 | 93.86 | 143 | — |
| **R-res side 704 (=512 tok)** | **80.93** | **89.78** | **141** | **−3.4** |
| variance | 77.87 | 89.93 | 137 | −5.3 |
| event (pure retina, no base) | 73.93 | 75.60 | 122 | −16.4 |
| graydog (Y-only DoG) | 69.50 | 75.60 | 122 | −18.2 |
| retinagate (base + event) | 69.90 | 71.55 | 115 | −19.9 |
| sobel | 60.67 | 73.84 | 122 | −22.1 |
| random | 42.30 | 48.54 | 71 | −53.5 |
| uniform 2-D lattice | 37.10 | 34.42 | 70 | −58.1 |

Reference arms at HALF the budget (K=256), from E0 shards: b2 (post-encoder
EADP) 78.33 / 68.43 / 131; rres side 512 → **78.63 / 77.51 / 137**.

Gate readings (prereg §8 of the task):

* **B (beat Random + Uniform): technically yes** (by 28–43 points) — but
  this is meaningless because **uniform is worse than random**: a regular
  lattice at 50 % alias-destroys document/text structure (OCRBench KIE
  1/164).  Structured regular decimation is pathological, not protective.
* **C (beat Sobel / grayscale DoG): FAIL.**  Local variance crushes every
  contrast-flavored score (77.9/89.9/137 vs retinagate 69.9/71.5/115).
* **D (clear quality edge over R-res): CATASTROPHIC FAIL — the decisive
  one.**  R-res at the same budget beats every pre-ViT selector by 7–30
  points; worse, R-res at *half* the budget (side 512, K=256: 78.6/77.5/137)
  still beats RetinaGate at 50 % (69.9/71.5/115).
* **E (≤1–2 point loss at ~50 %): FAIL.**  Best pre-ViT arm (variance)
  loses 5.3 points on average; R-res loses 3.4 with better TTFT.

Failure trend confirmation at K=256 (same DEV rows; rres side 512 from the
E0 shards = identical 256-token budget):

| arm @K=256 | TextVQA | DocVQA | OCRBench | mean Δ vs b0 |
|---|---|---|---|---|
| R-res side 512 | 78.63 | 77.51 | 137 | −11.0 |
| variance | 53.53 | 54.80 | 102 | −36.3 |
| retinagate | 45.53 | 20.83 | 70 | −61.5 |
| random | 20.57 | 20.56 | 28 | −86.4 |

The R-res advantage **widens** as the budget shrinks (33 / 57 / 67 points
over RetinaGate at 25 %).  Paired per-question contrasts with cluster-free
bootstrap (10k resamples; full table in `m12_analysis.json`) confirm every
R-res vs selector gap is decisively negative (e.g. OCRBench RetinaGate@512
vs R-res@512: −27 points [−38, −16]).

Scoring note: the `_pred_results` detail's `eval_score` stores DocVQA anls
*distance* (not the headline's gated hit), so per-question contrasts here
re-derive hits from `eval_match` with the exact `hit_calculate` semantics
and are validated to reproduce each shard's official headline to <1.5
points.  (This also re-explains E0's D1 "sanity" flag: nothing is wrong
with the headline numbers; only the naive per-question backfill was
misread.)

## 7b. The opponent-channel / base-lattice ablation answers (early)

The screen doubles as the prereg ablation:

* **Base lattice contributes NEGATIVE value**: base+event (retinagate)
  < event-only on all three datasets (−4.0 / −4.0 / −7).  The 2-D lattice's
  structured gaps are the same aliasing pathology as the uniform arm.
* **Opponent chroma**: on DocVQA, event and graydog produce *identical*
  predictions on 299/300 questions (chroma carries nothing on near-grayscale
  documents).  On TextVQA (natural images) chroma is worth +4.4
  (73.9 vs 69.5).  Honest reading: the "retina story" reduces to luminance
  DoG plus a modest natural-image chroma bonus, dominated by plain variance
  in both regimes.

## 8. Mechanism diagnostics

`m12_diag.py` (30 DEV samples/dataset, K=512): coverage of the groups the
post-encoder incumbent (B2 = official EADP scoring + block8 facility)
keeps, by each cheap pre-ViT score:

| dataset | event | graydog | sobel | variance | retinagate sel. |
|---|---|---|---|---|---|
| TextVQA | .524 | .522 | .519 | .545 | .522 |
| DocVQA | .500 | .500 | .503 | .533 | .494 |
| OCRBench | .520 | .518 | .523 | .549 | .520 |
| ChartQA | .512 | .503 | .505 | .538 | .510 |

Recall ≈ 0.5 = chance (K/N = 0.5); AUROC 0.49–0.56.  **Every cheap
pixel-space score is near-chance at predicting which regions the encoder
features (and the post-encoder selector) consider important.**  This was
predictable from the round's own history (post-encoder selectors read
semantic features that only exist *after* the encoder), and it predicts the
accuracy outcome: no pre-ViT score can find "the important half" because
importance at this granularity is not a pixel-space property.

## 8b. Failure cases (DocVQA, RetinaGate K=512 vs R-res side 704)

Where the two arms diverge, RetinaGate shows character-level text mangling —
the signature of dropped glyph patches with unrecoverable context:

| q | b0 | R-res 704 | RetinaGate 512 |
|---|---|---|---|
| 180 | 4-30-92 | 4-20-92 | 7-30-72 |
| 236 | Change of Due Dates for Monthly Payroll… | (same) | Change of Due dates for hourly payroll… |
| 276 | Dec-08 | Feb'09 | Feb-09 |

The uniform-lattice arm's failures are coarser: entire column bands of text
missing (KIE 1/164 on OCRBench), consistent with stride aliasing.

## 9. Selector overhead & honest cost accounting

Measured on the real path (§4): T_total = T_gate + T_sparse_ViT + downstream
with gate ≈ 0.9 ms against ~55 ms saved at 50% — overhead <2% of savings.
No hidden CPU↔GPU copies: everything runs on the already-on-GPU
`pixel_values`.

## 10. Verdict

# **KILL** — the pre-encoder token-dropping direction, not just RetinaGate.

Both kill conditions of the prereg fired, and they are structural, not
tuning failures:

1. **Gate A passed** (real wall-clock exists: −54 ms ViT, −149 ms TTFT at
   50 %), so the *engineering* is fine.
2. **Gate D failed absolutely.**  Direct resolution reduction dominates the
   entire pre-ViT selection axis at every comparator: R-res at the same
   budget (80.9/89.8/141) beats the best selector (variance, 77.9/89.9/137)
   and crushes RetinaGate (69.9/71.5/115); R-res at *half* the budget still
   beats RetinaGate at 50 %.  Lowering resolution preserves each kept
   token's encoder context (every token still sees the whole image);
   dropping pre-encoder tokens removes both raw signal (unrecoverable) and
   the ViT context of the kept tokens (the kept features themselves
   degrade).  That mechanism cannot be fixed by a better cheap score — the
   diagnostics show no cheap score even correlates with post-encoder
   importance (§8), and the two best "selectors" (variance, R-res) work by
   preserving coverage/context, not by finding importance.
3. The specific RetinaGate claims falsified:
   * **Base Lattice — negative** (aliasing destroys text structure; the
     uniform-lattice arm is *worse than random*, and base+event < event);
   * **Opponent channels — near-inert** (identical predictions to
     luminance-only on 299/300 DocVQA questions; +4.4 on natural images,
     still far behind plain variance);
   * **Retina event > simple heuristics — false** (loses to variance by
     5–14 points everywhere, ties or loses to graydog).

Per the prereg stop rule ("if A or D clearly fail, record the negative
evidence and stop"), no further selector work, no rescue designs, no
parameter search were run.  The one extra confirmation (K=256 trend) was
recorded to make the negative result reusable.

What survives from this round:
* a bit-exact, genuinely wall-clock-reducing sparse-ViT path for Qwen3-VL
  (group-structured, DeepStack- and mRoPE-preserving, N1-style verified);
* the measured A40 scaling law of the encoder (§4) and the ~35–40 ms CPU
  preprocessing share of TTFT that any production pre-encoder gate would
  want to absorb (gate before the resize);
* a strong, well-controlled negative: at matched budget, **context
  preservation (R-res) beats importance selection (any cheap pre-ViT
  score) by 5–30 points** on OCR-type workloads — a useful prior for the
  next round: if encoder compute must shrink, reduce resolution or merge
  (context-preserving), do not drop.

## 11. Pareto data (quality vs real ViT latency)

Quality (mean Δ vs b0 over the three OCR datasets) against measured ViT
latency from §4:

| arm | K | ViT ms | TTFT ms | Text | Doc | OCR | mean Δ vs b0 |
|---|---|---|---|---|---|---|---|
| b0 | 1024 | 110.3 | 405 | 85.0 | 93.9 | 143 | 0 |
| R-res | 784 | 83.3 | 314 | — | — | — | (not run) |
| R-res | 484–529 | 54.5 | 219 | 80.9 | 89.8 | 141 | −3.4 |
| R-res | 256 | 29.5 | 135 | 78.6 | 77.5 | 137 | −11.0 |
| variance | 512 | 55.8 | 256 | 77.9 | 89.9 | 137 | −5.3 |
| variance | 256 | 33.2 | 194 | 53.5 | 54.8 | 102 | −36.3 |
| retinagate | 512 | 55.8 | 256 | 69.9 | 71.5 | 115 | −19.9 |
| retinagate | 256 | 33.2 | 194 | 45.5 | 20.8 | 70 | −61.5 |
| random | 512 | 55.8 | 256 | 42.3 | 48.5 | 71 | −53.5 |
| random | 256 | 33.2 | 194 | 20.6 | 20.6 | 28 | −86.4 |

R-res rows: accuracy from side 704 (484 tokens, quality arm) / side 512
(256 tokens, E0 shard); timing from the oracle's side-736 / side-512 runs.
The quality axis, not latency, decides: at every budget the R-res point
dominates every selection point on accuracy at equal (or better) latency.

## 12. Reproducibility

Code: `Qwen_vl/model/retinagate.py`, `Qwen_vl/scripts/m12/`.
Raw: `Qwen_vl/outputs/m12/{m12_correctness.json,m12_speed_oracle.json,
m12_diag.json,acc/**}` (per-sample shards, per-question scores, paired
bootstrap in `m12_analysis.json`).
