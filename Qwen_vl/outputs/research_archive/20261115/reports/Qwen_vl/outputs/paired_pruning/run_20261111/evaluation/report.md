# Paired pruning evaluation

Status: **complete**; QA: **complete**.

DocVQA TRAIN heldout by source parent; not official test or official VAL results.

Original denominator: 192 questions / 64 source parents; outputs 4416/4416.

Scores below are percentages. Incomplete arms have no primary score; see JSON for missing keys and bounds.

| Arm | ANLS | Strict | Truncated | Outputs |
| --- | ---: | ---: | ---: | ---: |
| full | 97.592 | 77.604 | 0 | 192/192 |
| b1 | 78.742 | 52.604 | 0 | 192/192 |
| b2 | 75.713 | 51.562 | 1 | 192/192 |
| rres512 | 81.626 | 56.250 | 0 | 192/192 |
| random | 63.233 | 32.292 | 0 | 192/192 |
| bag_rank__seed_11 | 85.186 | 65.104 | 0 | 192/192 |
| bag_rank__seed_23 | 83.711 | 63.021 | 0 | 192/192 |
| bag_rank__seed_37 | 86.123 | 64.583 | 0 | 192/192 |
| ordered_rank__seed_11 | 85.510 | 64.062 | 0 | 192/192 |
| ordered_rank__seed_23 | 74.664 | 57.292 | 0 | 192/192 |
| ordered_rank__seed_37 | 85.437 | 65.625 | 0 | 192/192 |
| ordered_pair__seed_11 | 82.435 | 61.458 | 0 | 192/192 |
| ordered_pair__seed_23 | 80.245 | 60.417 | 0 | 192/192 |
| ordered_pair__seed_37 | 81.757 | 63.021 | 0 | 192/192 |
| canonical__seed_11 | 78.220 | 59.375 | 0 | 192/192 |
| canonical__seed_23 | 79.909 | 57.292 | 0 | 192/192 |
| canonical__seed_37 | 81.408 | 60.938 | 0 | 192/192 |
| token_marginal__seed_11 | 78.493 | 57.812 | 0 | 192/192 |
| token_marginal__seed_23 | 79.433 | 58.333 | 0 | 192/192 |
| token_marginal__seed_37 | 84.268 | 63.542 | 0 | 192/192 |
| multimode__seed_11 | 78.486 | 54.167 | 0 | 192/192 |
| multimode__seed_23 | 80.446 | 61.458 | 0 | 192/192 |
| multimode__seed_37 | 75.573 | 53.646 | 0 | 192/192 |

## All training seeds

No seed is selected by test performance.

| Arm | Mean ANLS | Seed range |
| --- | ---: | ---: |
| bag_rank | 85.006 | 83.711–86.123 |
| ordered_rank | 81.870 | 74.664–85.510 |
| ordered_pair | 81.479 | 80.245–82.435 |
| canonical | 79.846 | 78.220–81.408 |
| token_marginal | 80.731 | 78.493–84.268 |
| multimode | 78.168 | 75.573–80.446 |

## Paired comparisons

Delta is candidate minus reference in percentage points; 10000 paired parent bootstrap draws.

| Comparison | Δ ANLS | 95% CI | Improved / worsened |
| --- | ---: | --- | ---: |
| bag_rank__seed_11_vs_b1 | 6.443 | [0.628, 12.380] | 54 / 27 |
| bag_rank__seed_11_vs_b2 | 9.472 | [3.339, 15.681] | 54 / 26 |
| bag_rank__seed_11_vs_rres512 | 3.559 | [-2.275, 9.537] | 49 / 26 |
| bag_rank__seed_23_vs_b1 | 4.968 | [-1.491, 11.470] | 51 / 30 |
| bag_rank__seed_23_vs_b2 | 7.998 | [1.068, 14.825] | 55 / 31 |
| bag_rank__seed_23_vs_rres512 | 2.085 | [-4.184, 8.411] | 47 / 34 |
| bag_rank__seed_37_vs_b1 | 7.380 | [1.155, 13.849] | 56 / 26 |
| bag_rank__seed_37_vs_b2 | 10.410 | [3.575, 17.239] | 57 / 24 |
| bag_rank__seed_37_vs_rres512 | 4.497 | [-1.081, 10.288] | 50 / 21 |
| ordered_rank__seed_11_vs_b1 | 6.767 | [0.992, 12.802] | 51 / 28 |
| ordered_rank__seed_11_vs_b2 | 9.796 | [3.044, 16.622] | 56 / 28 |
| ordered_rank__seed_11_vs_rres512 | 3.884 | [-2.488, 10.505] | 49 / 28 |
| ordered_rank__seed_23_vs_b1 | -4.079 | [-11.405, 3.438] | 42 / 39 |
| ordered_rank__seed_23_vs_b2 | -1.050 | [-9.679, 7.607] | 48 / 40 |
| ordered_rank__seed_23_vs_rres512 | -6.962 | [-14.724, 0.784] | 42 / 40 |
| ordered_rank__seed_37_vs_b1 | 6.695 | [0.894, 12.840] | 53 / 26 |
| ordered_rank__seed_37_vs_b2 | 9.724 | [3.543, 15.965] | 54 / 24 |
| ordered_rank__seed_37_vs_rres512 | 3.811 | [-2.018, 9.921] | 48 / 24 |
| ordered_pair__seed_11_vs_b1 | 3.692 | [-2.104, 9.972] | 48 / 29 |
| ordered_pair__seed_11_vs_b2 | 6.722 | [0.358, 13.384] | 50 / 28 |
| ordered_pair__seed_11_vs_rres512 | 0.809 | [-5.511, 7.559] | 48 / 30 |
| ordered_pair__seed_23_vs_b1 | 1.503 | [-5.158, 8.245] | 48 / 32 |
| ordered_pair__seed_23_vs_b2 | 4.532 | [-3.366, 12.256] | 54 / 33 |
| ordered_pair__seed_23_vs_rres512 | -1.381 | [-8.457, 5.908] | 48 / 33 |
| ordered_pair__seed_37_vs_b1 | 3.014 | [-3.523, 9.772] | 51 / 31 |
| ordered_pair__seed_37_vs_b2 | 6.044 | [-0.628, 12.788] | 52 / 27 |
| ordered_pair__seed_37_vs_rres512 | 0.131 | [-6.506, 7.072] | 45 / 30 |
| canonical__seed_11_vs_b1 | -0.523 | [-7.499, 6.600] | 43 / 38 |
| canonical__seed_11_vs_b2 | 2.506 | [-4.653, 9.658] | 45 / 35 |
| canonical__seed_11_vs_rres512 | -3.406 | [-9.295, 2.390] | 46 / 33 |
| canonical__seed_23_vs_b1 | 1.167 | [-4.951, 7.558] | 48 / 37 |
| canonical__seed_23_vs_b2 | 4.196 | [-2.610, 11.032] | 51 / 32 |
| canonical__seed_23_vs_rres512 | -1.717 | [-7.325, 4.034] | 47 / 36 |
| canonical__seed_37_vs_b1 | 2.666 | [-3.553, 9.122] | 50 / 32 |
| canonical__seed_37_vs_b2 | 5.695 | [-1.350, 12.788] | 49 / 27 |
| canonical__seed_37_vs_rres512 | -0.218 | [-6.222, 5.948] | 45 / 27 |
| token_marginal__seed_11_vs_b1 | -0.249 | [-7.116, 6.836] | 50 / 37 |
| token_marginal__seed_11_vs_b2 | 2.780 | [-4.644, 10.258] | 51 / 33 |
| token_marginal__seed_11_vs_rres512 | -3.133 | [-9.844, 3.723] | 45 / 34 |
| token_marginal__seed_23_vs_b1 | 0.691 | [-6.104, 7.721] | 49 / 39 |
| token_marginal__seed_23_vs_b2 | 3.720 | [-3.385, 10.930] | 48 / 33 |
| token_marginal__seed_23_vs_rres512 | -2.193 | [-8.422, 4.066] | 46 / 35 |
| token_marginal__seed_37_vs_b1 | 5.525 | [-0.149, 11.636] | 51 / 28 |
| token_marginal__seed_37_vs_b2 | 8.554 | [1.966, 15.222] | 51 / 26 |
| token_marginal__seed_37_vs_rres512 | 2.642 | [-3.469, 8.961] | 52 / 28 |
| multimode__seed_11_vs_b1 | -0.257 | [-6.606, 6.374] | 47 / 48 |
| multimode__seed_11_vs_b2 | 2.772 | [-4.680, 10.231] | 51 / 47 |
| multimode__seed_11_vs_rres512 | -3.140 | [-9.541, 3.376] | 44 / 47 |
| multimode__seed_23_vs_b1 | 1.703 | [-4.318, 8.041] | 44 / 31 |
| multimode__seed_23_vs_b2 | 4.732 | [-2.030, 11.368] | 47 / 30 |
| multimode__seed_23_vs_rres512 | -1.180 | [-7.754, 5.734] | 48 / 35 |
| multimode__seed_37_vs_b1 | -3.170 | [-9.819, 3.815] | 44 / 47 |
| multimode__seed_37_vs_b2 | -0.141 | [-7.569, 7.478] | 45 / 43 |
| multimode__seed_37_vs_rres512 | -6.053 | [-13.227, 1.087] | 47 / 44 |
| ordered_rank__seed_11_vs_bag_rank__seed_11 | 0.324 | [-3.798, 4.724] | 17 / 16 |
| ordered_pair__seed_11_vs_ordered_rank__seed_11 | -3.075 | [-7.749, 1.463] | 21 / 26 |
| ordered_pair__seed_11_vs_bag_rank__seed_11 | -2.751 | [-7.157, 1.522] | 19 / 22 |
| token_marginal__seed_11_vs_canonical__seed_11 | 0.274 | [-4.111, 4.734] | 25 / 21 |
| multimode__seed_11_vs_canonical__seed_11 | 0.266 | [-4.562, 5.233] | 26 / 36 |
| multimode__seed_11_vs_token_marginal__seed_11 | -0.008 | [-4.995, 4.983] | 20 / 33 |
| ordered_rank__seed_23_vs_bag_rank__seed_23 | -9.047 | [-15.213, -3.310] | 12 / 33 |
| ordered_pair__seed_23_vs_ordered_rank__seed_23 | 5.582 | [0.338, 10.914] | 26 / 16 |
| ordered_pair__seed_23_vs_bag_rank__seed_23 | -3.465 | [-9.490, 2.606] | 22 / 29 |
| token_marginal__seed_23_vs_canonical__seed_23 | -0.476 | [-3.658, 2.580] | 19 / 16 |
| multimode__seed_23_vs_canonical__seed_23 | 0.536 | [-3.683, 4.776] | 27 / 23 |
| multimode__seed_23_vs_token_marginal__seed_23 | 1.012 | [-3.252, 5.313] | 22 / 23 |
| ordered_rank__seed_37_vs_bag_rank__seed_37 | -0.686 | [-3.558, 2.194] | 11 / 11 |
| ordered_pair__seed_37_vs_ordered_rank__seed_37 | -3.680 | [-8.388, 0.699] | 20 / 23 |
| ordered_pair__seed_37_vs_bag_rank__seed_37 | -4.366 | [-9.534, 0.530] | 20 / 24 |
| token_marginal__seed_37_vs_canonical__seed_37 | 2.860 | [-0.821, 6.556] | 20 / 16 |
| multimode__seed_37_vs_canonical__seed_37 | -5.835 | [-10.459, -1.213] | 15 / 38 |
| multimode__seed_37_vs_token_marginal__seed_37 | -8.695 | [-13.272, -4.270] | 10 / 41 |

## Numeric and non-numeric questions

| Arm | Numeric ANLS (N) | Non-numeric ANLS (N) |
| --- | ---: | ---: |
| full | 100.000 (8) | 97.488 (184) |
| b1 | 100.000 (8) | 77.818 (184) |
| b2 | 62.500 (8) | 76.288 (184) |
| rres512 | 60.938 (8) | 82.526 (184) |
| random | 75.000 (8) | 62.721 (184) |
| bag_rank__seed_11 | 87.500 (8) | 85.085 (184) |
| bag_rank__seed_23 | 95.312 (8) | 83.207 (184) |
| bag_rank__seed_37 | 87.500 (8) | 86.063 (184) |
| ordered_rank__seed_11 | 87.500 (8) | 85.423 (184) |
| ordered_rank__seed_23 | 87.500 (8) | 74.106 (184) |
| ordered_rank__seed_37 | 87.500 (8) | 85.348 (184) |
| ordered_pair__seed_11 | 75.000 (8) | 82.758 (184) |
| ordered_pair__seed_23 | 82.812 (8) | 80.134 (184) |
| ordered_pair__seed_37 | 62.500 (8) | 82.594 (184) |
| canonical__seed_11 | 97.917 (8) | 77.363 (184) |
| canonical__seed_23 | 100.000 (8) | 79.036 (184) |
| canonical__seed_37 | 87.500 (8) | 81.143 (184) |
| token_marginal__seed_11 | 87.500 (8) | 78.102 (184) |
| token_marginal__seed_23 | 87.500 (8) | 79.083 (184) |
| token_marginal__seed_37 | 87.500 (8) | 84.127 (184) |
| multimode__seed_11 | 85.938 (8) | 78.162 (184) |
| multimode__seed_23 | 87.500 (8) | 80.139 (184) |
| multimode__seed_37 | 85.938 (8) | 75.122 (184) |

## Fresh end-to-end timing on fixed DEV questions

Status: complete; 690/690 measured invocations.

Each invocation includes image preparation, vision encoding, selection and generation; model loading is excluded. TTFT generates one token; the 16-token measurement disables EOS only for timing. QA retains native EOS. These small dev-page timings are not benchmark-wide latency claims.

| Arm | Median TTFT ms | Median 16-token total ms |
| --- | ---: | ---: |
| full | 529.034 | 1070.364 |
| b1 | 443.348 | 985.033 |
| b2 | 411.645 | 957.115 |
| rres512 | 252.660 | 796.009 |
| random | 401.460 | 944.760 |
| bag_rank__seed_11 | 403.818 | 944.232 |
| bag_rank__seed_23 | 402.484 | 943.243 |
| bag_rank__seed_37 | 404.040 | 949.634 |
| ordered_rank__seed_11 | 403.657 | 944.080 |
| ordered_rank__seed_23 | 401.695 | 942.029 |
| ordered_rank__seed_37 | 403.377 | 943.829 |
| ordered_pair__seed_11 | 403.346 | 944.786 |
| ordered_pair__seed_23 | 402.727 | 943.837 |
| ordered_pair__seed_37 | 404.155 | 950.071 |
| canonical__seed_11 | 403.699 | 948.554 |
| canonical__seed_23 | 403.284 | 943.695 |
| canonical__seed_37 | 402.440 | 945.017 |
| token_marginal__seed_11 | 403.809 | 941.624 |
| token_marginal__seed_23 | 402.901 | 946.606 |
| token_marginal__seed_37 | 403.972 | 942.614 |
| multimode__seed_11 | 403.540 | 945.305 |
| multimode__seed_23 | 402.798 | 946.264 |
| multimode__seed_37 | 403.311 | 943.914 |

## Mechanism and training context

Mechanism diagnostics: **complete**. These fit/validation checks do not establish test-method effectiveness.

Query interventions: 72 validation questions; own minus swapped-mask ANLS: 41.79023042556157 percentage points.

Distinct successful pairs: 68; crossed-set failures: 8.

fit: 366/384 questions have at least two successful masks; 16/384 have no successful mask (fallback fraction 4.167%).

val: 68/72 questions have at least two successful masks; 2/72 have no successful mask (fallback fraction 2.778%).

No-success fallbacks must not be counted as successful-set supervision. Multiple successful sets alone do not show that mixing their members fails.

| Checkpoint | Updates | Validation-selected step | Fallback exposures / question exposures |
| --- | ---: | ---: | ---: |
| bag_rank__seed_11 | 900 | 300 | 0 / 5400 |
| bag_rank__seed_23 | 900 | 300 | 0 / 5400 |
| bag_rank__seed_37 | 900 | 240 | 0 / 5400 |
| ordered_rank__seed_11 | 900 | 390 | 0 / 5400 |
| ordered_rank__seed_23 | 900 | 480 | 0 / 5400 |
| ordered_rank__seed_37 | 900 | 240 | 0 / 5400 |
| ordered_pair__seed_11 | 900 | 750 | 0 / 5400 |
| ordered_pair__seed_23 | 900 | 540 | 0 / 5400 |
| ordered_pair__seed_37 | 900 | 840 | 0 / 5400 |
| canonical__seed_11 | 900 | 390 | 213 / 5400 |
| canonical__seed_23 | 900 | 690 | 186 / 5400 |
| canonical__seed_37 | 900 | 630 | 230 / 5400 |
| token_marginal__seed_11 | 900 | 780 | 213 / 5400 |
| token_marginal__seed_23 | 900 | 690 | 186 / 5400 |
| token_marginal__seed_37 | 900 | 660 | 230 / 5400 |
| multimode__seed_11 | 900 | 870 | 213 / 5400 |
| multimode__seed_23 | 900 | 690 | 186 / 5400 |
| multimode__seed_37 | 900 | 810 | 230 / 5400 |

## Interpretation limits

- All questions, including truncations and full-model errors, stay in the denominator.
- No test-based checkpoint, seed or question selection.
- Confidence intervals are exploratory and not adjusted for multiple comparisons.
- Document bootstrap covers sample uncertainty, not the full training-seed distribution.
- A complete evaluation is not evidence that either proposed mechanism succeeds.
