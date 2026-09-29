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

(filled after screen 1)

## 8. Mechanism diagnostics

(filled after diag)

## 9. Selector overhead & honest cost accounting

Measured on the real path (§4): T_total = T_gate + T_sparse_ViT + downstream
with gate ≈ 0.9 ms against ~55 ms saved at 50% — overhead <2% of savings.
No hidden CPU↔GPU copies: everything runs on the already-on-GPU
`pixel_values`.

## 10. Verdict

(pending accuracy)

## 11. Reproducibility

Code: `Qwen_vl/model/retinagate.py`, `Qwen_vl/scripts/m12/`.
Raw: `Qwen_vl/outputs/m12/{m12_correctness.json,m12_speed_oracle.json,
m12_diag.json,acc/**}` (per-sample shards, per-question scores, paired
bootstrap in `m12_analysis.json`).
