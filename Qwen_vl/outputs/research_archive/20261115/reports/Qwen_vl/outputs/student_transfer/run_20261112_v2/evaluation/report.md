# Frozen student cross-dataset evaluation

Overall: **complete**. QA 79443/79443; timing 1512/1512.

Fixed checkpoints trained only on DocVQA TRAIN; no target-data training, tuning or test-seed selection. Missing predictions remain missing; failed attempts score zero if stored, truncations stay in the primary denominator. All score differences are paired by source document; intervals are exploratory pointwise intervals without multiplicity correction.

Official validation benchmarks have been observed in earlier project experiments. Decontamination means no exact training image/parent overlap under the recorded checks, not absence from backbone pretraining or proof against near-duplicate contamination.

Scores below are percentages using each dataset's own metric. No across-dataset ANLS or cherry-picked seed is reported.

Subsets: decontaminated excludes recorded exact fit/validation image or source-parent overlap; conservative also excludes dHash near-duplicate suspects. no_documented_history additionally excludes indices in configured prior prediction tables. Missing history evidence is not proof that a benchmark is untouched.

## DocVQA_VAL: ANLS

5349 questions / 938 source parents. History exposure: {"exposed": 5349}.

Project observation, not current student training; no score-based selection.

Overlap audit: {"training_overlap_count": 0, "near_duplicate_suspected_count": 19}.

### all (n=5349)

| Arm | Score | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 94.537 | 5349/5349 | 0 | 0 |
| b1 | 72.118 | 5349/5349 | 1 | 0 |
| b2 | 68.315 | 5349/5349 | 1 | 0 |
| rres512 | 80.307 | 5349/5349 | 2 | 0 |
| bag_rank__seed_11 | 75.441 | 5349/5349 | 0 | 0 |
| bag_rank__seed_23 | 74.517 | 5349/5349 | 1 | 0 |
| bag_rank__seed_37 | 76.473 | 5349/5349 | 1 | 0 |

bag_rank: three-seed mean 75.477; range 74.517–76.473. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Parent bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 3.323 | [1.828, 4.851] | 1325 / 989 / 3035 |
| bag_rank__seed_11_vs_b2 | 7.126 | [5.602, 8.639] | 1513 / 889 / 2947 |
| bag_rank__seed_11_vs_rres512 | -4.866 | [-6.272, -3.412] | 1033 / 1149 / 3167 |
| bag_rank__seed_23_vs_b1 | 2.399 | [0.905, 3.851] | 1294 / 1063 / 2992 |
| bag_rank__seed_23_vs_b2 | 6.202 | [4.636, 7.713] | 1497 / 960 / 2892 |
| bag_rank__seed_23_vs_rres512 | -5.790 | [-7.241, -4.330] | 1009 / 1271 / 3069 |
| bag_rank__seed_37_vs_b1 | 4.356 | [2.866, 5.893] | 1357 / 979 / 3013 |
| bag_rank__seed_37_vs_b2 | 8.159 | [6.572, 9.698] | 1563 / 866 / 2920 |
| bag_rank__seed_37_vs_rres512 | -3.834 | [-5.185, -2.453] | 1032 / 1123 / 3194 |

### decontaminated (n=5349)

| Arm | Score | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 94.537 | 5349/5349 | 0 | 0 |
| b1 | 72.118 | 5349/5349 | 1 | 0 |
| b2 | 68.315 | 5349/5349 | 1 | 0 |
| rres512 | 80.307 | 5349/5349 | 2 | 0 |
| bag_rank__seed_11 | 75.441 | 5349/5349 | 0 | 0 |
| bag_rank__seed_23 | 74.517 | 5349/5349 | 1 | 0 |
| bag_rank__seed_37 | 76.473 | 5349/5349 | 1 | 0 |

bag_rank: three-seed mean 75.477; range 74.517–76.473. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Parent bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 3.323 | [1.828, 4.851] | 1325 / 989 / 3035 |
| bag_rank__seed_11_vs_b2 | 7.126 | [5.602, 8.639] | 1513 / 889 / 2947 |
| bag_rank__seed_11_vs_rres512 | -4.866 | [-6.272, -3.412] | 1033 / 1149 / 3167 |
| bag_rank__seed_23_vs_b1 | 2.399 | [0.905, 3.851] | 1294 / 1063 / 2992 |
| bag_rank__seed_23_vs_b2 | 6.202 | [4.636, 7.713] | 1497 / 960 / 2892 |
| bag_rank__seed_23_vs_rres512 | -5.790 | [-7.241, -4.330] | 1009 / 1271 / 3069 |
| bag_rank__seed_37_vs_b1 | 4.356 | [2.866, 5.893] | 1357 / 979 / 3013 |
| bag_rank__seed_37_vs_b2 | 8.159 | [6.572, 9.698] | 1563 / 866 / 2920 |
| bag_rank__seed_37_vs_rres512 | -3.834 | [-5.185, -2.453] | 1032 / 1123 / 3194 |

### conservative (n=5330)

| Arm | Score | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 94.540 | 5330/5330 | 0 | 0 |
| b1 | 72.107 | 5330/5330 | 1 | 0 |
| b2 | 68.284 | 5330/5330 | 1 | 0 |
| rres512 | 80.367 | 5330/5330 | 2 | 0 |
| bag_rank__seed_11 | 75.434 | 5330/5330 | 0 | 0 |
| bag_rank__seed_23 | 74.512 | 5330/5330 | 1 | 0 |
| bag_rank__seed_37 | 76.468 | 5330/5330 | 1 | 0 |

bag_rank: three-seed mean 75.471; range 74.512–76.468. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Parent bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 3.327 | [1.804, 4.870] | 1322 / 987 / 3021 |
| bag_rank__seed_11_vs_b2 | 7.150 | [5.615, 8.691] | 1508 / 886 / 2936 |
| bag_rank__seed_11_vs_rres512 | -4.933 | [-6.363, -3.553] | 1028 / 1148 / 3154 |
| bag_rank__seed_23_vs_b1 | 2.406 | [0.905, 3.895] | 1291 / 1060 / 2979 |
| bag_rank__seed_23_vs_b2 | 6.228 | [4.667, 7.791] | 1492 / 956 / 2882 |
| bag_rank__seed_23_vs_rres512 | -5.855 | [-7.320, -4.376] | 1005 / 1271 / 3054 |
| bag_rank__seed_37_vs_b1 | 4.361 | [2.837, 5.905] | 1353 / 977 / 3000 |
| bag_rank__seed_37_vs_b2 | 8.184 | [6.647, 9.767] | 1558 / 863 / 2909 |
| bag_rank__seed_37_vs_rres512 | -3.899 | [-5.294, -2.541] | 1027 / 1122 / 3181 |

### no_documented_history (n=0)

| Arm | Score | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 未完成 | 0/0 | 0 | 0 |
| b1 | 未完成 | 0/0 | 0 | 0 |
| b2 | 未完成 | 0/0 | 0 | 0 |
| rres512 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_11 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_23 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_37 | 未完成 | 0/0 | 0 | 0 |

bag_rank: three-seed mean 未完成; range 未完成. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Parent bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 未完成 | — | — |
| bag_rank__seed_11_vs_b2 | 未完成 | — | — |
| bag_rank__seed_11_vs_rres512 | 未完成 | — | — |
| bag_rank__seed_23_vs_b1 | 未完成 | — | — |
| bag_rank__seed_23_vs_b2 | 未完成 | — | — |
| bag_rank__seed_23_vs_rres512 | 未完成 | — | — |
| bag_rank__seed_37_vs_b1 | 未完成 | — | — |
| bag_rank__seed_37_vs_b2 | 未完成 | — | — |
| bag_rank__seed_37_vs_rres512 | 未完成 | — | — |

## TextVQA_VAL: VQA consensus

5000 questions / 3166 source parents. History exposure: {"exposed": 5000}.

Project observation, not current student training; no score-based selection.

Overlap audit: {"training_overlap_count": 0, "near_duplicate_suspected_count": 0}.

### all (n=5000)

| Arm | Score | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 83.740 | 5000/5000 | 0 | 0 |
| b1 | 77.788 | 5000/5000 | 0 | 0 |
| b2 | 77.286 | 5000/5000 | 0 | 0 |
| rres512 | 74.544 | 5000/5000 | 1 | 0 |
| bag_rank__seed_11 | 79.852 | 5000/5000 | 0 | 0 |
| bag_rank__seed_23 | 78.886 | 5000/5000 | 0 | 0 |
| bag_rank__seed_37 | 80.614 | 5000/5000 | 0 | 0 |

bag_rank: three-seed mean 79.784; range 78.886–80.614. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Parent bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 2.064 | [1.211, 2.932] | 419 / 288 / 4293 |
| bag_rank__seed_11_vs_b2 | 2.566 | [1.665, 3.485] | 451 / 293 / 4256 |
| bag_rank__seed_11_vs_rres512 | 5.308 | [4.256, 6.361] | 657 / 358 / 3985 |
| bag_rank__seed_23_vs_b1 | 1.098 | [0.214, 1.987] | 412 / 331 / 4257 |
| bag_rank__seed_23_vs_b2 | 1.600 | [0.670, 2.523] | 434 / 333 / 4233 |
| bag_rank__seed_23_vs_rres512 | 4.342 | [3.264, 5.402] | 643 / 390 / 3967 |
| bag_rank__seed_37_vs_b1 | 2.826 | [1.978, 3.676] | 436 / 259 / 4305 |
| bag_rank__seed_37_vs_b2 | 3.328 | [2.421, 4.224] | 466 / 268 / 4266 |
| bag_rank__seed_37_vs_rres512 | 6.070 | [5.031, 7.102] | 666 / 318 / 4016 |

### decontaminated (n=5000)

| Arm | Score | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 83.740 | 5000/5000 | 0 | 0 |
| b1 | 77.788 | 5000/5000 | 0 | 0 |
| b2 | 77.286 | 5000/5000 | 0 | 0 |
| rres512 | 74.544 | 5000/5000 | 1 | 0 |
| bag_rank__seed_11 | 79.852 | 5000/5000 | 0 | 0 |
| bag_rank__seed_23 | 78.886 | 5000/5000 | 0 | 0 |
| bag_rank__seed_37 | 80.614 | 5000/5000 | 0 | 0 |

bag_rank: three-seed mean 79.784; range 78.886–80.614. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Parent bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 2.064 | [1.211, 2.932] | 419 / 288 / 4293 |
| bag_rank__seed_11_vs_b2 | 2.566 | [1.665, 3.485] | 451 / 293 / 4256 |
| bag_rank__seed_11_vs_rres512 | 5.308 | [4.256, 6.361] | 657 / 358 / 3985 |
| bag_rank__seed_23_vs_b1 | 1.098 | [0.214, 1.987] | 412 / 331 / 4257 |
| bag_rank__seed_23_vs_b2 | 1.600 | [0.670, 2.523] | 434 / 333 / 4233 |
| bag_rank__seed_23_vs_rres512 | 4.342 | [3.264, 5.402] | 643 / 390 / 3967 |
| bag_rank__seed_37_vs_b1 | 2.826 | [1.978, 3.676] | 436 / 259 / 4305 |
| bag_rank__seed_37_vs_b2 | 3.328 | [2.421, 4.224] | 466 / 268 / 4266 |
| bag_rank__seed_37_vs_rres512 | 6.070 | [5.031, 7.102] | 666 / 318 / 4016 |

### conservative (n=5000)

| Arm | Score | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 83.740 | 5000/5000 | 0 | 0 |
| b1 | 77.788 | 5000/5000 | 0 | 0 |
| b2 | 77.286 | 5000/5000 | 0 | 0 |
| rres512 | 74.544 | 5000/5000 | 1 | 0 |
| bag_rank__seed_11 | 79.852 | 5000/5000 | 0 | 0 |
| bag_rank__seed_23 | 78.886 | 5000/5000 | 0 | 0 |
| bag_rank__seed_37 | 80.614 | 5000/5000 | 0 | 0 |

bag_rank: three-seed mean 79.784; range 78.886–80.614. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Parent bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 2.064 | [1.211, 2.932] | 419 / 288 / 4293 |
| bag_rank__seed_11_vs_b2 | 2.566 | [1.665, 3.485] | 451 / 293 / 4256 |
| bag_rank__seed_11_vs_rres512 | 5.308 | [4.256, 6.361] | 657 / 358 / 3985 |
| bag_rank__seed_23_vs_b1 | 1.098 | [0.214, 1.987] | 412 / 331 / 4257 |
| bag_rank__seed_23_vs_b2 | 1.600 | [0.670, 2.523] | 434 / 333 / 4233 |
| bag_rank__seed_23_vs_rres512 | 4.342 | [3.264, 5.402] | 643 / 390 / 3967 |
| bag_rank__seed_37_vs_b1 | 2.826 | [1.978, 3.676] | 436 / 259 / 4305 |
| bag_rank__seed_37_vs_b2 | 3.328 | [2.421, 4.224] | 466 / 268 / 4266 |
| bag_rank__seed_37_vs_rres512 | 6.070 | [5.031, 7.102] | 666 / 318 / 4016 |

### no_documented_history (n=0)

| Arm | Score | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 未完成 | 0/0 | 0 | 0 |
| b1 | 未完成 | 0/0 | 0 | 0 |
| b2 | 未完成 | 0/0 | 0 | 0 |
| rres512 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_11 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_23 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_37 | 未完成 | 0/0 | 0 | 0 |

bag_rank: three-seed mean 未完成; range 未完成. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Parent bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 未完成 | — | — |
| bag_rank__seed_11_vs_b2 | 未完成 | — | — |
| bag_rank__seed_11_vs_rres512 | 未完成 | — | — |
| bag_rank__seed_23_vs_b1 | 未完成 | — | — |
| bag_rank__seed_23_vs_b2 | 未完成 | — | — |
| bag_rank__seed_23_vs_rres512 | 未完成 | — | — |
| bag_rank__seed_37_vs_b1 | 未完成 | — | — |
| bag_rank__seed_37_vs_b2 | 未完成 | — | — |
| bag_rank__seed_37_vs_rres512 | 未完成 | — | — |

## OCRBench: OCRBench substring accuracy

1000 questions / 930 source parents. History exposure: {"exposed": 1000}.

Project observation, not current student training; no score-based selection.

Overlap audit: {"training_overlap_count": 0, "near_duplicate_suspected_count": 0}.

### all (n=1000)

| Arm | Score | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 85.100 | 1000/1000 | 0 | 0 |
| b1 | 74.200 | 1000/1000 | 0 | 0 |
| b2 | 73.400 | 1000/1000 | 0 | 0 |
| rres512 | 79.600 | 1000/1000 | 0 | 0 |
| bag_rank__seed_11 | 76.700 | 1000/1000 | 1 | 0 |
| bag_rank__seed_23 | 77.400 | 1000/1000 | 1 | 0 |
| bag_rank__seed_37 | 77.800 | 1000/1000 | 0 | 0 |

bag_rank: three-seed mean 77.300; range 76.700–77.800. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Parent bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 2.500 | [0.000, 5.030] | 86 / 61 / 853 |
| bag_rank__seed_11_vs_b2 | 3.300 | [0.810, 5.823] | 90 / 57 / 853 |
| bag_rank__seed_11_vs_rres512 | -2.900 | [-5.427, -0.395] | 63 / 92 / 845 |
| bag_rank__seed_23_vs_b1 | 3.200 | [0.897, 5.616] | 85 / 53 / 862 |
| bag_rank__seed_23_vs_b2 | 4.000 | [1.608, 6.483] | 95 / 55 / 850 |
| bag_rank__seed_23_vs_rres512 | -2.200 | [-4.592, 0.202] | 63 / 85 / 852 |
| bag_rank__seed_37_vs_b1 | 3.600 | [1.329, 5.917] | 84 / 48 / 868 |
| bag_rank__seed_37_vs_b2 | 4.400 | [2.096, 6.786] | 94 / 50 / 856 |
| bag_rank__seed_37_vs_rres512 | -1.800 | [-4.225, 0.591] | 64 / 82 / 854 |

### decontaminated (n=1000)

| Arm | Score | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 85.100 | 1000/1000 | 0 | 0 |
| b1 | 74.200 | 1000/1000 | 0 | 0 |
| b2 | 73.400 | 1000/1000 | 0 | 0 |
| rres512 | 79.600 | 1000/1000 | 0 | 0 |
| bag_rank__seed_11 | 76.700 | 1000/1000 | 1 | 0 |
| bag_rank__seed_23 | 77.400 | 1000/1000 | 1 | 0 |
| bag_rank__seed_37 | 77.800 | 1000/1000 | 0 | 0 |

bag_rank: three-seed mean 77.300; range 76.700–77.800. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Parent bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 2.500 | [0.000, 5.030] | 86 / 61 / 853 |
| bag_rank__seed_11_vs_b2 | 3.300 | [0.810, 5.823] | 90 / 57 / 853 |
| bag_rank__seed_11_vs_rres512 | -2.900 | [-5.427, -0.395] | 63 / 92 / 845 |
| bag_rank__seed_23_vs_b1 | 3.200 | [0.897, 5.616] | 85 / 53 / 862 |
| bag_rank__seed_23_vs_b2 | 4.000 | [1.608, 6.483] | 95 / 55 / 850 |
| bag_rank__seed_23_vs_rres512 | -2.200 | [-4.592, 0.202] | 63 / 85 / 852 |
| bag_rank__seed_37_vs_b1 | 3.600 | [1.329, 5.917] | 84 / 48 / 868 |
| bag_rank__seed_37_vs_b2 | 4.400 | [2.096, 6.786] | 94 / 50 / 856 |
| bag_rank__seed_37_vs_rres512 | -1.800 | [-4.225, 0.591] | 64 / 82 / 854 |

### conservative (n=1000)

| Arm | Score | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 85.100 | 1000/1000 | 0 | 0 |
| b1 | 74.200 | 1000/1000 | 0 | 0 |
| b2 | 73.400 | 1000/1000 | 0 | 0 |
| rres512 | 79.600 | 1000/1000 | 0 | 0 |
| bag_rank__seed_11 | 76.700 | 1000/1000 | 1 | 0 |
| bag_rank__seed_23 | 77.400 | 1000/1000 | 1 | 0 |
| bag_rank__seed_37 | 77.800 | 1000/1000 | 0 | 0 |

bag_rank: three-seed mean 77.300; range 76.700–77.800. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Parent bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 2.500 | [0.000, 5.030] | 86 / 61 / 853 |
| bag_rank__seed_11_vs_b2 | 3.300 | [0.810, 5.823] | 90 / 57 / 853 |
| bag_rank__seed_11_vs_rres512 | -2.900 | [-5.427, -0.395] | 63 / 92 / 845 |
| bag_rank__seed_23_vs_b1 | 3.200 | [0.897, 5.616] | 85 / 53 / 862 |
| bag_rank__seed_23_vs_b2 | 4.000 | [1.608, 6.483] | 95 / 55 / 850 |
| bag_rank__seed_23_vs_rres512 | -2.200 | [-4.592, 0.202] | 63 / 85 / 852 |
| bag_rank__seed_37_vs_b1 | 3.600 | [1.329, 5.917] | 84 / 48 / 868 |
| bag_rank__seed_37_vs_b2 | 4.400 | [2.096, 6.786] | 94 / 50 / 856 |
| bag_rank__seed_37_vs_rres512 | -1.800 | [-4.225, 0.591] | 64 / 82 / 854 |

### no_documented_history (n=0)

| Arm | Score | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 未完成 | 0/0 | 0 | 0 |
| b1 | 未完成 | 0/0 | 0 | 0 |
| b2 | 未完成 | 0/0 | 0 | 0 |
| rres512 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_11 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_23 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_37 | 未完成 | 0/0 | 0 | 0 |

bag_rank: three-seed mean 未完成; range 未完成. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Parent bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 未完成 | — | — |
| bag_rank__seed_11_vs_b2 | 未完成 | — | — |
| bag_rank__seed_11_vs_rres512 | 未完成 | — | — |
| bag_rank__seed_23_vs_b1 | 未完成 | — | — |
| bag_rank__seed_23_vs_b2 | 未完成 | — | — |
| bag_rank__seed_23_vs_rres512 | 未完成 | — | — |
| bag_rank__seed_37_vs_b1 | 未完成 | — | — |
| bag_rank__seed_37_vs_b2 | 未完成 | — | — |
| bag_rank__seed_37_vs_rres512 | 未完成 | — | — |

### Task categories

| Category (n) | full | b1 | b2 | rres512 | bag_rank__seed_11 | bag_rank__seed_23 | bag_rank__seed_37 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Artistic Text Recognition (50) | 92.000 | 92.000 | 94.000 | 98.000 | 92.000 | 92.000 | 92.000 |
| Digit String Recognition (50) | 76.000 | 78.000 | 76.000 | 86.000 | 74.000 | 70.000 | 76.000 |
| Doc-oriented VQA (200) | 75.500 | 51.500 | 49.000 | 65.500 | 58.500 | 58.000 | 59.000 |
| Handwriting Recognition (50) | 84.000 | 86.000 | 88.000 | 84.000 | 84.000 | 84.000 | 80.000 |
| Handwritten Mathematical Expression Recognition (100) | 64.000 | 62.000 | 63.000 | 65.000 | 54.000 | 57.000 | 55.000 |
| Irregular Text Recognition (50) | 92.000 | 92.000 | 92.000 | 94.000 | 92.000 | 92.000 | 92.000 |
| Key Information Extraction (200) | 89.000 | 65.500 | 60.500 | 72.500 | 73.500 | 76.500 | 76.500 |
| Non-Semantic Text Recognition (50) | 88.000 | 86.000 | 90.000 | 94.000 | 84.000 | 84.000 | 88.000 |
| Regular Text Recognition (50) | 98.000 | 98.000 | 98.000 | 100.000 | 98.000 | 98.000 | 98.000 |
| Scene Text-centric VQA (200) | 96.500 | 90.000 | 91.500 | 88.500 | 93.500 | 94.000 | 94.500 |

Original OCRBench reports the sum of correct items (out of 1000 for the full benchmark) and sum/10 as percentage. Here score is mean correctness over the explicit frozen roster, with per-category denominators.

## Latency

fresh image preparation+vision+selection+native generation; no image/KV cache; model load excluded

Median over parent-level medians; each item median includes all repeated paired blocks. Confidence intervals resample parents, not individual repeats.

Fixed benchmark image panel and single-device latency; not serving throughput. Cap 1 is TTFT, cap 16 is fixed-length total generation time.

### DocVQA_VAL: 12 independent parents

| Arm | Cap | Median ms | Observed/expected |
|---|---:|---:|---:|
| full | 1 | 432.947 | 36/36 |
| full | 16 | 966.109 | 36/36 |
| b1 | 1 | 347.756 | 36/36 |
| b1 | 16 | 882.706 | 36/36 |
| b2 | 1 | 315.000 | 36/36 |
| b2 | 16 | 854.203 | 36/36 |
| rres512 | 1 | 165.046 | 36/36 |
| rres512 | 16 | 689.140 | 36/36 |
| bag_rank__seed_11 | 1 | 303.556 | 36/36 |
| bag_rank__seed_11 | 16 | 843.797 | 36/36 |
| bag_rank__seed_23 | 1 | 307.847 | 36/36 |
| bag_rank__seed_23 | 16 | 832.847 | 36/36 |
| bag_rank__seed_37 | 1 | 307.072 | 36/36 |
| bag_rank__seed_37 | 16 | 834.339 | 36/36 |

| Paired candidate − reference | Median Δ ms [95% CI] | Median speedup [95% CI] |
|---|---:|---:|
| bag_rank__seed_11_vs_b1__cap_1 | -39.642 [-41.451, -38.420] | 1.136 [1.122, 1.140] |
| bag_rank__seed_11_vs_b1__cap_16 | -39.857 [-43.360, -35.953] | 1.048 [1.043, 1.051] |
| bag_rank__seed_11_vs_b2__cap_1 | -8.745 [-10.328, -8.028] | 1.029 [1.026, 1.034] |
| bag_rank__seed_11_vs_b2__cap_16 | -8.558 [-15.991, -5.651] | 1.010 [1.007, 1.019] |
| bag_rank__seed_11_vs_rres512__cap_1 | 145.471 [137.003, 151.153] | 0.540 [0.496, 0.552] |
| bag_rank__seed_11_vs_rres512__cap_16 | 151.023 [141.524, 153.307] | 0.823 [0.812, 0.831] |
| bag_rank__seed_23_vs_b1__cap_1 | -39.278 [-40.865, -38.361] | 1.132 [1.118, 1.140] |
| bag_rank__seed_23_vs_b1__cap_16 | -44.214 [-46.528, -36.071] | 1.053 [1.043, 1.056] |
| bag_rank__seed_23_vs_b2__cap_1 | -9.299 [-10.457, -7.185] | 1.030 [1.024, 1.037] |
| bag_rank__seed_23_vs_b2__cap_16 | -7.677 [-14.937, -5.292] | 1.009 [1.006, 1.018] |
| bag_rank__seed_23_vs_rres512__cap_1 | 145.969 [141.854, 149.779] | 0.540 [0.496, 0.555] |
| bag_rank__seed_23_vs_rres512__cap_16 | 139.999 [137.770, 153.427] | 0.830 [0.818, 0.834] |
| bag_rank__seed_37_vs_b1__cap_1 | -39.412 [-40.963, -38.864] | 1.137 [1.119, 1.143] |
| bag_rank__seed_37_vs_b1__cap_16 | -41.760 [-48.776, -39.172] | 1.050 [1.046, 1.059] |
| bag_rank__seed_37_vs_b2__cap_1 | -8.873 [-10.274, -7.278] | 1.029 [1.024, 1.036] |
| bag_rank__seed_37_vs_b2__cap_16 | -10.367 [-13.694, -4.267] | 1.012 [1.005, 1.016] |
| bag_rank__seed_37_vs_rres512__cap_1 | 144.302 [140.011, 150.928] | 0.539 [0.498, 0.553] |
| bag_rank__seed_37_vs_rres512__cap_16 | 145.144 [136.074, 147.914] | 0.829 [0.821, 0.833] |

### TextVQA_VAL: 12 independent parents

| Arm | Cap | Median ms | Observed/expected |
|---|---:|---:|---:|
| full | 1 | 409.913 | 36/36 |
| full | 16 | 941.270 | 36/36 |
| b1 | 1 | 322.828 | 36/36 |
| b1 | 16 | 846.908 | 36/36 |
| b2 | 1 | 291.625 | 36/36 |
| b2 | 16 | 817.634 | 36/36 |
| rres512 | 1 | 135.831 | 36/36 |
| rres512 | 16 | 662.987 | 36/36 |
| bag_rank__seed_11 | 1 | 282.507 | 36/36 |
| bag_rank__seed_11 | 16 | 811.128 | 36/36 |
| bag_rank__seed_23 | 1 | 284.130 | 36/36 |
| bag_rank__seed_23 | 16 | 809.261 | 36/36 |
| bag_rank__seed_37 | 1 | 283.303 | 36/36 |
| bag_rank__seed_37 | 16 | 810.783 | 36/36 |

| Paired candidate − reference | Median Δ ms [95% CI] | Median speedup [95% CI] |
|---|---:|---:|
| bag_rank__seed_11_vs_b1__cap_1 | -39.599 [-40.653, -38.158] | 1.140 [1.137, 1.152] |
| bag_rank__seed_11_vs_b1__cap_16 | -37.224 [-38.852, -32.956] | 1.046 [1.040, 1.049] |
| bag_rank__seed_11_vs_b2__cap_1 | -8.848 [-10.513, -8.213] | 1.033 [1.030, 1.037] |
| bag_rank__seed_11_vs_b2__cap_16 | -7.016 [-11.498, -4.068] | 1.009 [1.005, 1.014] |
| bag_rank__seed_11_vs_rres512__cap_1 | 147.340 [143.778, 147.978] | 0.482 [0.478, 0.485] |
| bag_rank__seed_11_vs_rres512__cap_16 | 148.279 [140.413, 150.276] | 0.817 [0.816, 0.827] |
| bag_rank__seed_23_vs_b1__cap_1 | -38.852 [-40.130, -37.630] | 1.140 [1.133, 1.145] |
| bag_rank__seed_23_vs_b1__cap_16 | -38.507 [-40.698, -32.046] | 1.048 [1.039, 1.050] |
| bag_rank__seed_23_vs_b2__cap_1 | -7.950 [-9.281, -7.424] | 1.030 [1.026, 1.033] |
| bag_rank__seed_23_vs_b2__cap_16 | -8.362 [-15.901, -2.938] | 1.011 [1.004, 1.020] |
| bag_rank__seed_23_vs_rres512__cap_1 | 148.007 [145.304, 149.153] | 0.480 [0.477, 0.485] |
| bag_rank__seed_23_vs_rres512__cap_16 | 147.698 [144.535, 149.455] | 0.818 [0.816, 0.822] |
| bag_rank__seed_37_vs_b1__cap_1 | -39.856 [-40.229, -38.344] | 1.141 [1.139, 1.145] |
| bag_rank__seed_37_vs_b1__cap_16 | -36.635 [-39.398, -28.877] | 1.045 [1.036, 1.049] |
| bag_rank__seed_37_vs_b2__cap_1 | -8.264 [-11.216, -7.053] | 1.031 [1.025, 1.039] |
| bag_rank__seed_37_vs_b2__cap_16 | -7.081 [-10.510, -0.511] | 1.009 [1.001, 1.013] |
| bag_rank__seed_37_vs_rres512__cap_1 | 146.954 [144.312, 149.636] | 0.480 [0.478, 0.487] |
| bag_rank__seed_37_vs_rres512__cap_16 | 148.115 [140.975, 162.192] | 0.817 [0.804, 0.826] |

### OCRBench: 12 independent parents

| Arm | Cap | Median ms | Observed/expected |
|---|---:|---:|---:|
| full | 1 | 411.540 | 36/36 |
| full | 16 | 955.537 | 36/36 |
| b1 | 1 | 325.442 | 36/36 |
| b1 | 16 | 860.545 | 36/36 |
| b2 | 1 | 290.013 | 36/36 |
| b2 | 16 | 834.484 | 36/36 |
| rres512 | 1 | 133.732 | 36/36 |
| rres512 | 16 | 667.534 | 36/36 |
| bag_rank__seed_11 | 1 | 282.340 | 36/36 |
| bag_rank__seed_11 | 16 | 816.085 | 36/36 |
| bag_rank__seed_23 | 1 | 282.637 | 36/36 |
| bag_rank__seed_23 | 16 | 813.581 | 36/36 |
| bag_rank__seed_37 | 1 | 282.888 | 36/36 |
| bag_rank__seed_37 | 16 | 826.073 | 36/36 |

| Paired candidate − reference | Median Δ ms [95% CI] | Median speedup [95% CI] |
|---|---:|---:|
| bag_rank__seed_11_vs_b1__cap_1 | -39.593 [-40.514, -38.077] | 1.141 [1.124, 1.149] |
| bag_rank__seed_11_vs_b1__cap_16 | -38.004 [-42.654, -32.102] | 1.046 [1.038, 1.053] |
| bag_rank__seed_11_vs_b2__cap_1 | -7.738 [-9.306, -6.651] | 1.027 [1.023, 1.030] |
| bag_rank__seed_11_vs_b2__cap_16 | -8.513 [-14.068, -6.990] | 1.011 [1.008, 1.017] |
| bag_rank__seed_11_vs_rres512__cap_1 | 148.689 [146.770, 152.631] | 0.478 [0.463, 0.522] |
| bag_rank__seed_11_vs_rres512__cap_16 | 147.629 [142.850, 155.993] | 0.818 [0.809, 0.825] |
| bag_rank__seed_23_vs_b1__cap_1 | -39.955 [-44.129, -38.578] | 1.139 [1.131, 1.154] |
| bag_rank__seed_23_vs_b1__cap_16 | -38.268 [-44.810, -34.217] | 1.047 [1.041, 1.056] |
| bag_rank__seed_23_vs_b2__cap_1 | -8.217 [-10.112, -7.376] | 1.028 [1.025, 1.035] |
| bag_rank__seed_23_vs_b2__cap_16 | -11.289 [-19.561, -7.705] | 1.014 [1.009, 1.024] |
| bag_rank__seed_23_vs_rres512__cap_1 | 147.842 [144.846, 151.921] | 0.480 [0.465, 0.521] |
| bag_rank__seed_23_vs_rres512__cap_16 | 147.992 [144.909, 153.146] | 0.820 [0.815, 0.824] |
| bag_rank__seed_37_vs_b1__cap_1 | -39.127 [-41.148, -37.934] | 1.141 [1.124, 1.150] |
| bag_rank__seed_37_vs_b1__cap_16 | -37.677 [-38.773, -22.619] | 1.046 [1.027, 1.048] |
| bag_rank__seed_37_vs_b2__cap_1 | -7.921 [-11.428, -6.782] | 1.027 [1.024, 1.038] |
| bag_rank__seed_37_vs_b2__cap_16 | -7.845 [-14.200, 0.493] | 1.010 [0.999, 1.017] |
| bag_rank__seed_37_vs_rres512__cap_1 | 147.942 [145.338, 152.764] | 0.481 [0.462, 0.527] |
| bag_rank__seed_37_vs_rres512__cap_16 | 151.365 [145.235, 166.656] | 0.812 [0.808, 0.820] |

## Audit

Freeze: verified. Stored scores independently recomputed against original answers. Native token budgets, DeepStack alignment, original positions and token caps checked for all successful outputs.

Missing outputs are never silently removed from denominators. See report.json for full-denominator bounds, category comparisons, source hashes and every missing key.
