# Four-stream local minimax: feature-only 32-image diagnostic

All 32 previously selected DEV images and all 8,192 2×2 blocks were processed on CPU, without model generation or opening ground truth. The 32 archive byte hashes matched their previously accepted identities; the four real streams and 256 EADP positions were checked against the prior independent component acceptance. The complete feature calculation was independently recomputed. Each archive was deserialized in full, although this analysis used only the four visual streams, grid and prior selected IDs.

The radius is the largest per-stream normalized L2 distance from one selected representative to any of the four positions in its block. Lower is better **only for this constructed feature objective**. Each image's 256 block radii were reduced to a median and a maximum, then the 32 image summaries were reduced to medians below. The final column is the median number, out of 256, also selected by EADP K256.

| One-per-block selection | Median of image median radii | Median of image maximum radii | Median EADP set overlap / 256 |
|---|---:|---:|---:|
| Four-stream minimax | 1.452 | 4.673 | 64 |
| Main-stream minimax | 1.485 | 5.052 | 65 |
| Top-left | 1.562 | 5.797 | 62.5 |
| Block random seed 20261104 | 1.580 | 5.543 | 64 |
| Block random seed 20261105 | 1.561 | 5.586 | 64 |
| Block random seed 20261106 | 1.565 | 5.616 | 65 |

The four-stream objective was strictly lower than the main-only choice in 2,626 of 8,192 blocks, with at least one such block in every image. Its ≤ relationship against each one-per-block control is guaranteed by construction, rather than independent evidence of utility. The predeclared stop rule of *no* strict improvement on all 32 images was not triggered; this does not authorize or imply a later model-quality test. There were 294 exact minimum-radius tie blocks and no all-zero streams; ties and their candidate IDs are retained in the per-image records.

EADP is not constrained to one per block. Across all 8,192 blocks, its 256 selections per image left 2,865 blocks empty, and occupied 3,124/1,622/500/81 blocks with one/two/three/four selected positions respectively. No one-per-block radius comparison is claimed for EADP's empty blocks. Set overlap and block occupancy describe different selection rules; neither is an answer-quality result.

This is posthoc exploration on 32 DEV images, with no model execution, logits, accuracy, inference latency, resource advantage or novelty claim. The block objective cannot bound original decoder output because dropping tokens changes its attention normalization. A single block can contain multiple task-critical tokens; the fixed one-token allocation can delete them. All four streams here were produced by a full visual forward and stored, unlike PixelPrune's pre-ViT reduction. A causal claim, deployment advantage, or quality conclusion would require a separate preregistered experiment with real generation, strong baselines, full costs and independent data.

Evidence: `plan001.json`, `inventory001.json`, `rows001.jsonl`, `result001.json`, and `monitor_run001`. Independent review: `e1_independent_review/local_minimax_v2_result_review.json`, SHA-256 `c693756dfef1e79ffdb1291166332d6a321ca2eace579b0cdddfd7284accbcf7`; its supervisor completed/0. Version 2 fixes the first candidate's space-order, finite-distance and tie-inventory defects before any real feature analysis; the original source and byte-hash inventory remain preserved.
