# MosaicPrune Phase 1 — adaptive spatial resolution instead of token selection

Status: **RUNNING / PENDING VERDICT** (numbers filled at the end of the run)

- Branch: `mosaic-prune-phase1`
- Date: 2026-09-29
- Code: `Qwen_vl/model/mosaic.py`, `Qwen_vl/VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py` (`Qwen3VLChatMosaic`), drivers `Qwen_vl/scripts/discovery/run_mosaic.sh`, `mosaic_vis.py`, `mosaic_efficiency.py`, `mosaic_tokenstats.py`, `mosaic_analyze.py`
- Raw outputs: `Qwen_vl/outputs/mosaic/` (accuracy), `Qwen_vl/outputs/discovery/mosaic/` (stats, efficiency, visualizations)

## 1. Motivation

Every method-search iteration so far (S1 scoring search, S2-B, M3–M9, SAGE) scored
*individual* visual tokens and decided "which tokens survive". The recurring
findings were:

- critical-token prediction is unstable; learned scorers saturate at the same
  in-sample ceiling as cheap proxies (M3-v2), and complex selection rarely beats
  simple rules (M7, M8, SAGE);
- the *identity* of individual tokens appears to be the wrong variable to learn
  (S2-C1/C5: token-local structure retains signal only weakly);
- what survives robustly across all negative results is spatial structure:
  rescue value is depth-structured (S3-B), early-layer attention is spatially
  coherent (M9), facility-location coverage beats top-k (the incumbent B2).

Phase 1 of MosaicPrune tests the orthogonal framing: instead of choosing
tokens, choose **resolution**. Keep full spatial coverage of the image, and
spend the token budget by refining high-complexity regions and coarsening
low-complexity ones. No token is ever deleted without a replacement covering
its area.

## 2. Hypothesis

**H (core):** feature dispersion of spatial blocks predicts which regions need
more tokens; an adaptive-resolution quadtree allocation driven by dispersion
beats (a) uniform spatial compression and (b) an equally-shaped content-blind
partition, at the same token budget K.

**Falsification gate (pre-registered, from the task spec):**

- PASS if any of: (1) Dispersion beats both Uniform and Random clearly on ≥2 of
  3 benchmarks; (2) stable advantage on DocVQA/OCRBench even without TextVQA;
  (3) accuracy parity with clearly cheaper compression overhead than the
  incumbent selector; (4) marked robustness advantage at K=128.
- FAIL if Dispersion ≈ Random, Uniform wins, the partition shows no task
  correlation, position handling breaks the model, or accuracy loss is large
  with no efficiency upside.

No tuning of the dispersion rule is allowed after the first fixed-protocol
run; at most two cheap variants (normalized dispersion, area-corrected score)
if PASS.

## 3. Method

Operating grid: the post-merger visual token grid of Qwen3-VL at 1024×1024 —
patch 16, spatial merge 2 → 64×64 patches → **32×32 = 1024 merged tokens**
(`grid_thw` → grid_h = h/2, grid_w = w/2). One cell is already a 2×2 patch
aggregate; the finest leaf is the native LLM-consumed resolution.

Partition to exactly K leaves (default K = 256):

- **uniform** — regular per-axis pooling with exact division (32×32 → 2×2
  pooling → 16×16 = 256). Pure locality, zero adaptivity.
- **random** — the same quadtree split mechanics as dispersion, but the leaf to
  split is chosen uniformly at random (seed 0). Because the split sequence
  does not read features, every image receives the *same* content-blind
  partition with (approximately) the same leaf-size distribution as
  dispersion — isolating "does the hierarchy shape help" from "does the
  signal's spatial placement help".
- **dispersion** — greedy refinement: repeatedly split the leaf with the
  highest within-block dispersion D(B) = mean_i ‖x_i − mean(x_B)‖² (computed
  from float32 integral images of the post-merger features, O(1) per query).

Budget mechanics: each quadrant split adds 3 leaves, so K = 256 is reached
exactly with 85 splits (1 + 3·85). For K = 128, 42 quadrant splits reach 127
leaves and one half-split (+1) of the highest-dispersion splittable leaf
finishes at exactly 128. Leaf → representation: mean pooling of its post-merger
tokens (single-token leaves pass through). Leaves are emitted in raster order,
so sequence order preserves reading order.

First version is deliberately pure: no learned scorer, no attention, no query
guidance, no residual tokens, no training.

## 4. Implementation details & position/grid handling

- **Insertion point.** Identical to every pruned baseline in this repo
  (EADP / CDPruner / DivPrune / HiPrune): the post-merger image embeds
  `model.visual(pixel_values, grid_thw)` are compressed, then spliced into the
  LLM input via `inputs_embeds` (`_build_pruned_inputs`), and generation runs
  `model.generate(inputs_embeds=..., attention_mask=...)`.
- **Position handling.** With this pathway `input_ids=None`, so transformers'
  `get_rope_index` falls back to plain sequential (arange) positions broadcast
  across the 3 M-RoPE axes; no per-image grid geometry reaches the decoder.
  This is a known shared approximation of *all* pruned arms in this repo, not
  something Mosaic introduces; the unpruned fixed-res baseline keeps its full
  M-RoPE path. Comparisons Mosaic-vs-EADP-vs-Uniform are therefore
  like-for-like; Mosaic-vs-unpruned conflates token count with position
  degradation, and is labelled accordingly in the tables.
- **Deepstack features.** The `inputs_embeds` pathway drops the multi-layer
  deepstack visual injection, exactly as for EADP and the other pruned
  baselines. Same caveat as above: shared by all pruned arms.
- **Grid/shape log.** 1024×1024 input → pixel grid 64×64 → token grid 32×32 →
  leaves partition {32×32} → K tokens, each LLM-dim (4096 for Qwen3-VL-8B).
  Single-image messages only in these benchmarks; the code handles
  `grid_thw` with multiple images by per-image partitioning.
- **Numerics.** Dispersion accumulates in float32 integral images; the split
  loop runs entirely on CPU (numpy) against a one-time ~16 MB integral-image
  copy, so it never synchronises with the GPU; pooling is a handful of
  integral-image gather kernels on GPU. Bitwise deterministic per (features,
  mode, seed): verified identical leaves and outputs across repeats.
- **Compression cost** (bf16, 1024 tokens, single A40, K=256):
  uniform ≈ 3.1 ms, random ≈ 9 ms, dispersion ≈ 18 ms per image — all below
  the incumbent EADP selector (~42 ms at K=256 from
  `outputs/discovery/efficiency_profile.json`).

## 5. Experimental protocol

- Backbone: Qwen3-VL-8B-Instruct, fixed 1024×1024, greedy decoding
  (temperature 0.01, top_p 0.001, top_k 1, max_new_tokens 2048) — identical to
  the EADP baseline runs of 2026-09-21.
- Benchmarks: DocVQA_VAL (ANLS), OCRBench (final score), TextVQA_VAL (VQA
  score) via the bundled VLMEvalKit `run.py`; full splits.
- Arms at K = 256 (primary) and K = 128 (secondary): uniform, random,
  dispersion. References: unpruned fixed-res baseline (1024 tokens, run in
  this sweep) and the reproduced official EADP-256 (α=0.5, β=2.0).
- Token statistics per arm/dataset: leaf side-length histogram
  (1×1 / 2×2 / 4×4 / 8×8 / 16×16 / 32×32) collected offline over 150
  deterministic samples per dataset (`mosaic_tokenstats.py`), since the eval
  driver resets the per-run JSONL.
- Efficiency: `mosaic_efficiency.py` mirrors the protocol that produced the
  existing baseline/EADP profile (50 samples/dataset, CUDA events, warmup):
  vision forward, compression, pruned prefill, e2e generate, FLOPs, peak mem.
- Visualization: 2 samples per dataset × 3 modes rendered as partition
  overlays (`outputs/discovery/mosaic/vis/`).

## 6. Main table

(filled when the sweep completes)

| arm | DocVQA_VAL | OCRBench | TextVQA_VAL |
|---|---|---|---|
| baseline-1024 | — | — | — |
| EADP-256 (incumbent) | 61.14 | 623 | 71.04 |
| Mosaic-256 uniform | — | — | — |
| Mosaic-256 random | — | — | — |
| Mosaic-256 dispersion | — | — | — |
| Mosaic-128 uniform | — | — | — |
| Mosaic-128 random | — | — | — |
| Mosaic-128 dispersion | — | — | — |

## 7. Efficiency table

(filled when the sweep completes)

## 8. Token statistics

(filled when the sweep completes)

## 9. Visualization

Diagnostic overlays for DocVQA / OCRBench / TextVQA samples are committed
under `outputs/discovery/mosaic/vis/` (gitignored output dir; referenced
here). First renders show the intended behaviour: blank bands and dark margins
collapse to 8×8/16×16 blocks, while titles, axis labels, plot lines, camera
text ("DAKOTA DIGITAL") and lens/flash edges refine to 1×1 leaves.

## 10. PASS / FAIL verdict

(pending)

## 11. Failure analysis & mechanism interpretation

(pending)

## 12. Worth developing into a paper method?

(pending)

## 13. Recommended next experiment

(pending)
