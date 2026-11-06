# R-res 512/K256 — exploratory DEV quality

The 384-question natural generation, formal no-GT acceptance, independent result review, scoring, and independent score review all completed. Each task has 128 questions and 128 image groups; all 384 R-res outputs ended with EOS, none were truncated or empty.

Scores are percentages; DocVQA uses ANLS, TextVQA VQA consensus, and AI2D accuracy.

**Correction:** The historical field `e1_full` denotes E1 P1-D1, an EADP K256 run with compact one-dimensional positions and DeepStack on. It is not an uncompressed baseline. Earlier descriptions calling it uncompressed were incorrect; all scores and intervals below are unchanged. See `PLAN_CORRECTION.md`.

| Arm | DocVQA | TextVQA | AI2D |
|---|---:|---:|---:|
| R-res 512/K256 | 80.98 | 72.81 | 82.03 |
| EADP K256 | 67.85 | 80.78 | 82.03 |
| EADP K256 P1/DS-on (compact 1D positions) | 64.62 | 79.45 | 75.78 |
| Random seed 20261104 | 52.61 | 73.52 | 72.66 |
| Random seed 20261105 | 56.96 | 72.19 | 75.78 |
| Random seed 20261106 | 56.72 | 73.44 | 73.44 |
| Random three-seed mean | 55.43 | 73.05 | 73.96 |

Difference is R-res minus the stated arm in percentage points; intervals are paired image-group percentile bootstrap 95% intervals (10,000 draws, seed 20261104).

| Comparator | DocVQA | TextVQA | AI2D |
|---|---:|---:|---:|
| EADP K256 | +13.14 [+4.91, +21.49] | -7.97 [-14.38, -1.88] | +0.00 [-5.47, +5.47] |
| EADP K256 P1/DS-on (compact 1D positions) | +16.36 [+7.97, +24.75] | -6.64 [-13.28, -0.08] | +6.25 [-1.56, +14.06] |
| Random seed 20261104 | +28.38 [+19.40, +37.37] | -0.70 [-8.36, +7.27] | +9.38 [+3.12, +15.62] |
| Random seed 20261105 | +24.02 [+15.41, +32.95] | +0.62 [-6.17, +7.11] | +6.25 [+0.78, +11.72] |
| Random seed 20261106 | +24.26 [+15.80, +33.04] | -0.62 [-8.52, +7.19] | +8.59 [+2.34, +14.84] |
| Random three-seed mean | +25.55 [+18.36, +32.95] | -0.23 [-6.77, +6.09] | +8.07 [+3.12, +13.02] |

R-res did not archive its actual runtime `generate` kwargs or full generation configuration. Its six reported generation-control scalars were copied from previously accepted E1 metadata. The outer `panel_core.checked_generation` hook checked all six actual `generate` kwargs and EOS against that metadata during each request, but the individual observed kwargs were not archived separately. Frozen code and checkpoint configuration provide indirect evidence for the remaining controls, but full runtime recipe equality with EADP or any other comparator is not independently established.

This is a posthoc analysis on the same DEV data after earlier results were observed. Intervals are uncorrected for multiple comparisons and do not establish confirmation or noninferiority. Same K does not mean equal compute: R-res uses a 512px input, while EADP and random start from the 1024-token visual input. The three random seeds were averaged within each question; the interval is conditional on those seeds. The random adapter executes and discards the EADP greedy pass, so its timings are not a random-selector speed comparison. Full visual, DeepStack and KV tensors were not archived; identity rests on runtime checks, recorded summaries and cache lengths.

Evidence: `e6/rres_k256_dev384/verification001.json`, `e1_independent_review/rres_k256_dev384_result_review.json`, `e6/rres_k256_full_analysis_v2/score001/analysis.json`, `e1_independent_review/rres_k256_dev384_scores_review.json`; all five coordinator steps completed.
