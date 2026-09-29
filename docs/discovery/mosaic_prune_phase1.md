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

Baseline / EADP-256 numbers: baseline from this sweep (`outputs/mosaic/`), EADP-256
reproduced 2026-09-21 (`outputs/eadp/`). All pruned arms share the
`inputs_embeds` pathway (no M-RoPE grid geometry, no deepstack); the baseline
keeps both, so the baseline row conflates token count with position/deepstack
degradation.

| arm | DocVQA_VAL (ANLS) | OCRBench | TextVQA_VAL (VQA) |
|---|---|---|---|
| baseline-1024 (full model) | **94.48** | **851** | **83.74** |
| EADP-256 (incumbent selector) | 61.14 | 623 | 71.04 |
| Mosaic-256 uniform | 56.78 | 539 | 67.58 |
| Mosaic-256 random | 42.99 | 493 | 61.95 |
| Mosaic-256 dispersion | 45.32 | 492 | 52.29 |
| Mosaic-128 uniform | (pending) | (pending) | (pending) |
| Mosaic-128 random | (pending) | (pending) | (pending) |
| Mosaic-128 dispersion | (pending) | (pending) | (pending) |

OCRBench sub-scores (K=256): dispersion keeps Text Recognition high (227 vs
uniform 177, EADP 228) but collapses Scene-text VQA (103 vs 161/169) and KIE
(53 vs 73/95) — refinement lands on glyph texture while the coarse blocks
destroy scene/document context. Baseline-1024 sub-scores: 265/193/151/178/64.

## 7. Paired diagnostic: pooling vs selection

`mosaic_diag_pooling.py`, 30 DocVQA samples (offset 0, easier-than-average;
full-1024 scores 86.1 there), all four arms on the identical wrapper/pathway:

| arm | ANLS |
|---|---|
| full-1024 | 86.08 |
| rand-sel-256 (random 256-token selection, original embeddings) | 50.18 |
| uniform-256 (2×2 mean pooling) | 52.03 |
| disp-256 (dispersion quadtree pooling) | 42.05 |

Readings: (1) mean-pooling is *not* extra-harmful — uniform pooling ≈ random
selection, so the off-manifold concern is refuted; (2) content-blind 256-token
compression of any kind is catastrophic (−34 vs full on these samples); (3)
dispersion pooling is 10 points *below* uniform pooling, and the loss is
heavy-tailed: median per-sample delta = 0, P(disp<uniform) = 0.30, driven by
catastrophic collapses — e.g. three different questions ('Men', '7', '.97')
all answered '6/20', i.e. when the answer region is absorbed into a coarse
block, the model grabs unrelated surviving text.

## 8. Efficiency

Compression wall-clock (microbenchmark + in-pipeline stats, A40, bf16,
1024→256): uniform ≈ 3.1 ms, random ≈ 9 ms, dispersion ≈ 18–21 ms per image.
EADP's full selector ≈ 42 ms at K=256 (`outputs/discovery/efficiency_profile.json`),
so the mosaic compressor is ~2–14× cheaper than the incumbent selector, and
the prefill gain (232 ms at 1024 tokens → ~230/180 ms at 256/128 kept tokens,
same profile) applies to all arms equally. Full profiler output for the mosaic
arms: (pending, `mosaic/efficiency_mosaic.json`).

## 9. Token statistics

Offline over 120 deterministic samples per dataset (`mosaic/tokenstats.json`),
K=256 leaf side-length distributions (percent of leaves):

| dataset | mode | 1×1 | 2×2 | 4×4 | 8×8 | 16×16 |
|---|---|---|---|---|---|---|
| DocVQA | dispersion | 71.9 | 20.6 | 5.6 | 1.8 | 0.4 |
| DocVQA | random | 75.0 | 15.6 | 7.0 | 2.3 | — |
| OCRBench | dispersion | 69.3 | 21.8 | 7.1 | 1.8 | 0.4 |
| OCRBench | random | 75.0 | 15.6 | 7.0 | 2.3 | — |
| TextVQA | dispersion | 70.0 | 21.8 | 6.3 | 1.8 | 0.4 |
| TextVQA | random | 75.0 | 15.6 | 7.0 | 2.3 | — |
| all | uniform | — | 100.0 | — | — | — |

The dispersion and random partitions have nearly identical *marginal*
distributions (by construction random matches the achievable quadtree shape
space) — so the accuracy differences in the main table are attributable to
*placement*, not shape. K=128 numbers in the same file; dispersion shifts
mass to coarser leaves (e.g. DocVQA 1×1 → ~62%).

## 10. Visualization

Overlays under `outputs/discovery/mosaic/vis/` (referenced, gitignored).
Confirmed behaviour: blank bands and dark margins collapse to 8×8/16×16
blocks; titles, axis labels, plot lines, camera text and lens/flash edges
refine to 1×1. The partition does what the hypothesis asked — the hypothesis
about what the LLM needs was wrong.

## 11. PASS / FAIL verdict

**FAIL** — pre-registered fail conditions met:

1. *"Dispersion 与 Random 基本一样"*: OCRBench 492 vs 493 (tie), TextVQA
   52.29 vs 61.95 (dispersion −9.7 **worse**), DocVQA 45.32 vs 42.99 (+2.3).
   No consistent dispersion-over-random edge.
2. *"Uniform 更好"*: uniform beats both adaptive arms on all three benchmarks.
3. *"adaptive partition 没有稳定 task correlation"*: only 1 of 3 benchmarks
   shows a dispersion>random edge, and it flips sign on TextVQA.

Gate conditions 1/2/4 unreachable; gate 3 (accuracy parity with cheaper
overhead) is unreachable at −15.8/−131/−18.7 vs EADP-256.

## 12. Failure analysis & mechanism interpretation

1. **The adaptive hierarchy itself is the primary damage.** Both adaptive
   partitions (dispersion, random) allow arbitrarily large leaves (8×8, 16×16
   cells → one token per up to 512×512 px); uniform never loses a region below
   2×2-cell granularity. The −11/−14 gap of adaptive arms vs uniform on DocVQA
   is the cost of those bets. Answer: *question 6 of the mechanism list — the
   hierarchy has negative value at this budget; only spatial locality (uniform)
   is safely exploitable.*
2. **Low feature dispersion ≠ low information.** Form fields, faint print and
   small labels live in visually flat areas; a variance-seeking allocator
   spends the budget on edges/texture (it even *wins* the raw Text Recognition
   sub-score) and collapses the context needed for scene-text VQA / KIE.
3. **Where dispersion beat random (DocVQA +2.3) it did so by placing 1×1
   leaves on glyphs** — a low-level, query-independent win that does not
   transfer to TextVQA (−9.7), where texture variance is decorrelated from
   semantics.
4. **Cheap content-agnostic signals and answer relevance**: this reproduces the
   project-wide negative (S1: necessary-token rank 0.525; S2-A: gradient
   saliency largely answer-agnostic; M8: nomination solved, ranking not) in a
   *spatial* form. Question 8 of the mechanism list: yes — Mosaic is the
   spatial analogue of the scorer search, and it fails for the same root
   reason: nothing about single-image feature statistics tells you which
   regions the *question* needs.
5. **"Large regions can be coarse-merged but not deleted" (question 7): NOT
   supported.** Uniform coarse-merging (2×2) is fine, but 8×8/16×16 merging —
   the whole point of an adaptive map — is consistently punished. At this
   pathway/budget the LLM's usable information is spread at ≤2×2-cell
   granularity across the image; there is no large "free" region class.
6. **Budget context**: the whole 256-token regime sits 33–51 points below the
   full model (94.48/851/83.74) — EADP included. Any resolution-map method
   would first have to beat its own uniform floor at 256 tokens; none here
   does.

## 13. Worth developing into a paper method?

**No, not in this form.** The phase-1 falsification is clean: a
training-free, query-free dispersion map cannot allocate resolution better
than uniform pooling, and adaptive spatial allocation *hurts* relative to
uniform at the same budget. The only salvageable observation is that plain
uniform 2×2 pooling is only −4.4/−84/−3.5 from EADP-256 at ~14× lower
selector cost — a useful floor/sanity baseline for future methods, and a
negative result worth one paragraph in the paper's ablation ("why not
adaptive spatial maps").

The two pre-registered follow-up variants (normalized dispersion, area
correction) are **not pursued**: they only reshape the allocation, and the
dominant failure (any large coarse leaf is dangerous; texture variance is
task-irrelevant) survives both.

## 14. Recommended next experiment

If the "resolution, not identity" framing is worth one more shot, the phase-1
evidence points at the missing ingredient: **the signal must be
query-conditioned, not content-agnostic** — every successful retained point
(EADP's dense instruction scoring, S3-B's depth structure) involves the text.
A minimal next probe would keep the quadtree mechanics but score leaves with
the *existing* EADP importance map (query-aware, already computed in this
repo): does query-aware refinement of a coverage-preserving partition beat
EADP's pure top-k selection at equal K? That isolates "coverage-preserving
allocation with the right signal" from "coverage-preserving allocation with a
wrong signal", at near-zero implementation cost (the pruner already returns
per-token importance; replace dispersion with importance in the same split
loop). If that also loses to EADP, the resolution framing is dead on this
pipeline and the token-selection line (S3-B depth structure) remains the only
live direction.
