# PixelPrune dynamic-budget DEV comparison

The complete 384-question DEV panel (128 per task) passed no-GT acceptance and independent review. All 384 generations ended with EOS. The official and independent per-question scores and all task-level paired bootstrap intervals were independently recomputed.

| DEV task | Pixel dynamic | E1 P3-D1 | E4 K256 D1 | E4 K1024 D1 | Pixel K min / median / max |
|---|---:|---:|---:|---:|---:|
| DocVQA | 91.62 | 67.85 | 67.85 | 91.95 | 161 / 666 / 992 |
| TextVQA | 83.75 | 80.78 | 80.78 | 85.55 | 462 / 759.5 / 1024 |
| AI2D | 82.03 | 82.03 | 82.03 | 82.03 | 187 / 493.5 / 950 |

Scores are percentages under the accepted local author-compatible scoring convention; AI2D uses deterministic local scoring. Differences below are Pixel minus the named reference, in percentage points, with paired 95% image-group percentile bootstrap intervals (10,000 draws; seed 20261104).

| DEV task | vs E1 P3-D1 | vs E4 K256 D1 | vs E4 K1024 D1 |
|---|---:|---:|---:|
| DocVQA | +23.77 [16.53, 31.09] | +23.77 [16.53, 31.09] | −0.33 [−2.88, 2.06] |
| TextVQA | +2.97 [−0.70, 6.87] | +2.97 [−0.70, 6.87] | −1.80 [−4.14, 0.00] |
| AI2D | 0.00 [−3.91, 3.91] | 0.00 [−3.91, 3.91] | 0.00 [−2.34, 2.34] |

This is an exploratory, post-hoc DEV comparison with multiple unadjusted intervals; intervals that include zero do not establish equivalence. PixelPrune uses an input-dependent visual budget, so fixed K256/K1024 are not equal-compute controls. This instrumented run supplies no latency claim. The retained visual/DeepStack and KV diagnostics include runtime checks and hashes/shapes; full high-dimensional tensors were not archived for independent reconstruction. No CONFIRM or FINAL answers were parsed.

Formal no-GT acceptance: `e6/pixelprune_dev384_accept/verification001.json` (SHA-256 `8f84e387bf818777e3fcc1336c6edbc53c116de8347699b9f78d7103f1da6ba0`); independent run review: `e1_independent_review/pixel_dev384_result_review.json` (`74bb252af44eacc1b8a04116a692abbe9b744eb27f83a7b4b3436d9cd3454aa5`). Scoring analysis SHA-256 `f128394c829b9bad8d91d59f177e17dd9149891b91f7b512d43c0809b9544e9f`; independent score review: `e1_independent_review/pixel_dev384_scores_review.json` (`d19949b8236b7d8cd95ca2e6df70dc7ace1c8360dff64f243f84b1cf676a45f4`). All corresponding supervised processes exited successfully.
