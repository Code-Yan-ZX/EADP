# Frozen student and resolution reference: broad benchmark evaluation

Overall: **complete**. QA 203252/203252; timing 3360/3360.

Fixed checkpoints trained only on DocVQA TRAIN; no target-data training, tuning or test-seed selection. Missing predictions remain missing; failed attempts and unparsed answers score zero, truncations stay in the primary denominator. Native grouped benchmark scores are primary. Paired bootstrap intervals describe auxiliary per-row scores only, not native grouped metrics; no multiplicity correction. Resampling and timing use source-pixel clusters, so duplicate pixels across source paths/categories remain dependent. No cross-benchmark scalar average is reported.

Public benchmarks may have prior project exposure. Decontamination means no exact training image/parent overlap under the recorded checks, not absence from backbone pretraining or proof against near-duplicate contamination. MCQ and yes/no use deterministic local exact matching without an external judge. This must be distinguished from published scores obtained with LLM answer matching. HallusionBench all 1129 is an extended complete-original-table evaluation, including 178 image-free controls, NOT the conventional image-mode denominator. Its image-only 951 subset matches the pinned VLMEvalKit default skip_noimg=True image-mode roster; both are reported separately.

Native benchmark aggregates are primary. Fraction metrics display as percentages; MME displays native points. No across-benchmark scalar average or cherry-picked seed is reported. Native aggregate confidence intervals are not estimated by the auxiliary row bootstrap.

Subsets: decontaminated excludes recorded exact fit/validation image or source-parent overlap; conservative also excludes dHash near-duplicate suspects. no_documented_history additionally excludes indices in configured prior prediction tables. Missing history evidence is not proof that a benchmark is untouched.

## AI2D_TEST: exact-matching accuracy

3088 questions / 1201 source parents / 1201 independent pixel/parent clusters. History exposure: {"unknown": 3088}.

source_pixel_sha256 first, then pixel_sha256 for images; parent_id fallback for no-image controls. Source parent identity and native benchmark group identities remain unchanged.

No configured historical full-prediction identity file; false flags do not prove no exposure.

Overlap audit: {"training_overlap_count": 0, "near_duplicate_suspected_count": 0}.

### Primary native benchmark metric

| Arm | Native score | Auxiliary row score (%) | Observed/expected | Unparsed |
|---|---:|---:|---:|---:|
| full | 84.974 | 84.974 | 3088/3088 | 54 |
| b1 | 81.606 | 81.606 | 3088/3088 | 48 |
| b2 | 81.120 | 81.120 | 3088/3088 | 44 |
| rres512 | 84.035 | 84.035 | 3088/3088 | 62 |
| bag_rank__seed_11 | 78.627 | 78.627 | 3088/3088 | 51 |
| bag_rank__seed_23 | 78.465 | 78.465 | 3088/3088 | 49 |
| bag_rank__seed_37 | 78.951 | 78.951 | 3088/3088 | 53 |

bag_rank: native three-seed mean 78.681; range 78.465–78.951. The range is not a confidence interval.

### Auxiliary per-row diagnostics

Auxiliary per-row correctness/ANLS differences only. These intervals are NOT confidence intervals for circular accuracy, MME points, POPE F1 or Hallusion grouped accuracy.

#### all (n=3088)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 84.974 | 3088/3088 | 0 | 0 |
| b1 | 81.606 | 3088/3088 | 1 | 0 |
| b2 | 81.120 | 3088/3088 | 1 | 0 |
| rres512 | 84.035 | 3088/3088 | 0 | 0 |
| bag_rank__seed_11 | 78.627 | 3088/3088 | 0 | 0 |
| bag_rank__seed_23 | 78.465 | 3088/3088 | 0 | 0 |
| bag_rank__seed_37 | 78.951 | 3088/3088 | 0 | 0 |

bag_rank: three-seed mean 78.681; range 78.465–78.951. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | -2.979 | [-4.247, -1.732] | 139 / 231 / 2718 |
| bag_rank__seed_11_vs_b2 | -2.494 | [-3.771, -1.226] | 156 / 233 / 2699 |
| bag_rank__seed_11_vs_rres512 | -5.408 | [-6.697, -4.133] | 111 / 278 / 2699 |
| bag_rank__seed_23_vs_b1 | -3.141 | [-4.389, -1.901] | 125 / 222 / 2741 |
| bag_rank__seed_23_vs_b2 | -2.655 | [-3.888, -1.438] | 138 / 220 / 2730 |
| bag_rank__seed_23_vs_rres512 | -5.570 | [-6.858, -4.293] | 98 / 270 / 2720 |
| bag_rank__seed_37_vs_b1 | -2.655 | [-3.893, -1.433] | 133 / 215 / 2740 |
| bag_rank__seed_37_vs_b2 | -2.170 | [-3.396, -0.931] | 152 / 219 / 2717 |
| bag_rank__seed_37_vs_rres512 | -5.084 | [-6.313, -3.839] | 100 / 257 / 2731 |

#### image_only (n=3088)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 84.974 | 3088/3088 | 0 | 0 |
| b1 | 81.606 | 3088/3088 | 1 | 0 |
| b2 | 81.120 | 3088/3088 | 1 | 0 |
| rres512 | 84.035 | 3088/3088 | 0 | 0 |
| bag_rank__seed_11 | 78.627 | 3088/3088 | 0 | 0 |
| bag_rank__seed_23 | 78.465 | 3088/3088 | 0 | 0 |
| bag_rank__seed_37 | 78.951 | 3088/3088 | 0 | 0 |

bag_rank: three-seed mean 78.681; range 78.465–78.951. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | -2.979 | [-4.247, -1.732] | 139 / 231 / 2718 |
| bag_rank__seed_11_vs_b2 | -2.494 | [-3.771, -1.226] | 156 / 233 / 2699 |
| bag_rank__seed_11_vs_rres512 | -5.408 | [-6.697, -4.133] | 111 / 278 / 2699 |
| bag_rank__seed_23_vs_b1 | -3.141 | [-4.389, -1.901] | 125 / 222 / 2741 |
| bag_rank__seed_23_vs_b2 | -2.655 | [-3.888, -1.438] | 138 / 220 / 2730 |
| bag_rank__seed_23_vs_rres512 | -5.570 | [-6.858, -4.293] | 98 / 270 / 2720 |
| bag_rank__seed_37_vs_b1 | -2.655 | [-3.893, -1.433] | 133 / 215 / 2740 |
| bag_rank__seed_37_vs_b2 | -2.170 | [-3.396, -0.931] | 152 / 219 / 2717 |
| bag_rank__seed_37_vs_rres512 | -5.084 | [-6.313, -3.839] | 100 / 257 / 2731 |

#### decontaminated (n=3088)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 84.974 | 3088/3088 | 0 | 0 |
| b1 | 81.606 | 3088/3088 | 1 | 0 |
| b2 | 81.120 | 3088/3088 | 1 | 0 |
| rres512 | 84.035 | 3088/3088 | 0 | 0 |
| bag_rank__seed_11 | 78.627 | 3088/3088 | 0 | 0 |
| bag_rank__seed_23 | 78.465 | 3088/3088 | 0 | 0 |
| bag_rank__seed_37 | 78.951 | 3088/3088 | 0 | 0 |

bag_rank: three-seed mean 78.681; range 78.465–78.951. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | -2.979 | [-4.247, -1.732] | 139 / 231 / 2718 |
| bag_rank__seed_11_vs_b2 | -2.494 | [-3.771, -1.226] | 156 / 233 / 2699 |
| bag_rank__seed_11_vs_rres512 | -5.408 | [-6.697, -4.133] | 111 / 278 / 2699 |
| bag_rank__seed_23_vs_b1 | -3.141 | [-4.389, -1.901] | 125 / 222 / 2741 |
| bag_rank__seed_23_vs_b2 | -2.655 | [-3.888, -1.438] | 138 / 220 / 2730 |
| bag_rank__seed_23_vs_rres512 | -5.570 | [-6.858, -4.293] | 98 / 270 / 2720 |
| bag_rank__seed_37_vs_b1 | -2.655 | [-3.893, -1.433] | 133 / 215 / 2740 |
| bag_rank__seed_37_vs_b2 | -2.170 | [-3.396, -0.931] | 152 / 219 / 2717 |
| bag_rank__seed_37_vs_rres512 | -5.084 | [-6.313, -3.839] | 100 / 257 / 2731 |

#### conservative (n=3088)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 84.974 | 3088/3088 | 0 | 0 |
| b1 | 81.606 | 3088/3088 | 1 | 0 |
| b2 | 81.120 | 3088/3088 | 1 | 0 |
| rres512 | 84.035 | 3088/3088 | 0 | 0 |
| bag_rank__seed_11 | 78.627 | 3088/3088 | 0 | 0 |
| bag_rank__seed_23 | 78.465 | 3088/3088 | 0 | 0 |
| bag_rank__seed_37 | 78.951 | 3088/3088 | 0 | 0 |

bag_rank: three-seed mean 78.681; range 78.465–78.951. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | -2.979 | [-4.247, -1.732] | 139 / 231 / 2718 |
| bag_rank__seed_11_vs_b2 | -2.494 | [-3.771, -1.226] | 156 / 233 / 2699 |
| bag_rank__seed_11_vs_rres512 | -5.408 | [-6.697, -4.133] | 111 / 278 / 2699 |
| bag_rank__seed_23_vs_b1 | -3.141 | [-4.389, -1.901] | 125 / 222 / 2741 |
| bag_rank__seed_23_vs_b2 | -2.655 | [-3.888, -1.438] | 138 / 220 / 2730 |
| bag_rank__seed_23_vs_rres512 | -5.570 | [-6.858, -4.293] | 98 / 270 / 2720 |
| bag_rank__seed_37_vs_b1 | -2.655 | [-3.893, -1.433] | 133 / 215 / 2740 |
| bag_rank__seed_37_vs_b2 | -2.170 | [-3.396, -0.931] | 152 / 219 / 2717 |
| bag_rank__seed_37_vs_rres512 | -5.084 | [-6.313, -3.839] | 100 / 257 / 2731 |

#### no_documented_history (n=0)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 未完成 | 0/0 | 0 | 0 |
| b1 | 未完成 | 0/0 | 0 | 0 |
| b2 | 未完成 | 0/0 | 0 | 0 |
| rres512 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_11 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_23 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_37 | 未完成 | 0/0 | 0 | 0 |

bag_rank: three-seed mean 未完成; range 未完成. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
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
| atomStructure (8) | 75.000 | 75.000 | 75.000 | 62.500 | 75.000 | 62.500 | 75.000 |
| eclipses (31) | 90.323 | 93.548 | 90.323 | 90.323 | 87.097 | 87.097 | 83.871 |
| faultsEarthquakes (28) | 71.429 | 60.714 | 67.857 | 71.429 | 64.286 | 64.286 | 57.143 |
| foodChainsWebs (1162) | 92.513 | 89.845 | 89.071 | 92.427 | 89.071 | 88.726 | 89.157 |
| lifeCycles (417) | 80.815 | 77.698 | 76.019 | 79.616 | 70.504 | 72.422 | 72.662 |
| moonPhaseEquinox (277) | 73.646 | 67.509 | 68.231 | 72.924 | 66.065 | 65.343 | 64.621 |
| partsOfA (487) | 84.189 | 80.903 | 81.109 | 83.368 | 77.207 | 76.386 | 77.618 |
| partsOfTheEarth (52) | 84.615 | 76.923 | 76.923 | 82.692 | 73.077 | 75.000 | 73.077 |
| photosynthesisRespiration (79) | 83.544 | 82.278 | 82.278 | 83.544 | 77.215 | 73.418 | 77.215 |
| rockCycle (67) | 74.627 | 73.134 | 73.134 | 70.149 | 64.179 | 73.134 | 70.149 |
| rockStrata (41) | 82.927 | 78.049 | 75.610 | 82.927 | 73.171 | 70.732 | 70.732 |
| solarSystem (36) | 97.222 | 91.667 | 91.667 | 94.444 | 88.889 | 86.111 | 91.667 |
| typesOf (343) | 78.134 | 75.219 | 73.761 | 76.093 | 70.554 | 69.388 | 70.262 |
| volcano (16) | 93.750 | 81.250 | 87.500 | 93.750 | 81.250 | 93.750 | 93.750 |
| waterCNPCycle (44) | 72.727 | 65.909 | 70.455 | 63.636 | 68.182 | 63.636 | 68.182 |

## MMBench_DEV_EN_V11: circular exact-matching accuracy

4876 questions / 1250 source parents / 1250 independent pixel/parent clusters. History exposure: {"unknown": 4876}.

source_pixel_sha256 first, then pixel_sha256 for images; parent_id fallback for no-image controls. Source parent identity and native benchmark group identities remain unchanged.

No configured historical full-prediction identity file; false flags do not prove no exposure.

Overlap audit: {"training_overlap_count": 0, "near_duplicate_suspected_count": 0}.

### Primary native benchmark metric

| Arm | Native score | Auxiliary row score (%) | Observed/expected | Unparsed |
|---|---:|---:|---:|---:|
| full | 86.146 | 90.012 | 4876/4876 | 9 |
| b1 | 83.824 | 88.413 | 4876/4876 | 12 |
| b2 | 82.895 | 88.208 | 4876/4876 | 11 |
| rres512 | 83.591 | 88.720 | 4876/4876 | 9 |
| bag_rank__seed_11 | 81.115 | 86.833 | 4876/4876 | 18 |
| bag_rank__seed_23 | 80.960 | 86.690 | 4876/4876 | 14 |
| bag_rank__seed_37 | 81.192 | 87.039 | 4876/4876 | 9 |

bag_rank: native three-seed mean 81.089; range 80.960–81.192. The range is not a confidence interval.

### Auxiliary per-row diagnostics

Auxiliary per-row correctness/ANLS differences only. These intervals are NOT confidence intervals for circular accuracy, MME points, POPE F1 or Hallusion grouped accuracy.

#### all (n=4876)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 90.012 | 4876/4876 | 2 | 0 |
| b1 | 88.413 | 4876/4876 | 5 | 0 |
| b2 | 88.208 | 4876/4876 | 4 | 0 |
| rres512 | 88.720 | 4876/4876 | 4 | 0 |
| bag_rank__seed_11 | 86.833 | 4876/4876 | 5 | 0 |
| bag_rank__seed_23 | 86.690 | 4876/4876 | 1 | 0 |
| bag_rank__seed_37 | 87.039 | 4876/4876 | 1 | 0 |

bag_rank: three-seed mean 86.854; range 86.690–87.039. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | -1.579 | [-2.688, -0.476] | 118 / 195 / 4563 |
| bag_rank__seed_11_vs_b2 | -1.374 | [-2.546, -0.242] | 129 / 196 / 4551 |
| bag_rank__seed_11_vs_rres512 | -1.887 | [-3.040, -0.704] | 122 / 214 / 4540 |
| bag_rank__seed_23_vs_b1 | -1.723 | [-2.964, -0.502] | 144 / 228 / 4504 |
| bag_rank__seed_23_vs_b2 | -1.518 | [-2.742, -0.323] | 148 / 222 / 4506 |
| bag_rank__seed_23_vs_rres512 | -2.030 | [-3.171, -0.880] | 127 / 226 / 4523 |
| bag_rank__seed_37_vs_b1 | -1.374 | [-2.523, -0.205] | 132 / 199 / 4545 |
| bag_rank__seed_37_vs_b2 | -1.169 | [-2.318, -0.041] | 135 / 192 / 4549 |
| bag_rank__seed_37_vs_rres512 | -1.682 | [-2.808, -0.556] | 125 / 207 / 4544 |

#### image_only (n=4876)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 90.012 | 4876/4876 | 2 | 0 |
| b1 | 88.413 | 4876/4876 | 5 | 0 |
| b2 | 88.208 | 4876/4876 | 4 | 0 |
| rres512 | 88.720 | 4876/4876 | 4 | 0 |
| bag_rank__seed_11 | 86.833 | 4876/4876 | 5 | 0 |
| bag_rank__seed_23 | 86.690 | 4876/4876 | 1 | 0 |
| bag_rank__seed_37 | 87.039 | 4876/4876 | 1 | 0 |

bag_rank: three-seed mean 86.854; range 86.690–87.039. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | -1.579 | [-2.688, -0.476] | 118 / 195 / 4563 |
| bag_rank__seed_11_vs_b2 | -1.374 | [-2.546, -0.242] | 129 / 196 / 4551 |
| bag_rank__seed_11_vs_rres512 | -1.887 | [-3.040, -0.704] | 122 / 214 / 4540 |
| bag_rank__seed_23_vs_b1 | -1.723 | [-2.964, -0.502] | 144 / 228 / 4504 |
| bag_rank__seed_23_vs_b2 | -1.518 | [-2.742, -0.323] | 148 / 222 / 4506 |
| bag_rank__seed_23_vs_rres512 | -2.030 | [-3.171, -0.880] | 127 / 226 / 4523 |
| bag_rank__seed_37_vs_b1 | -1.374 | [-2.523, -0.205] | 132 / 199 / 4545 |
| bag_rank__seed_37_vs_b2 | -1.169 | [-2.318, -0.041] | 135 / 192 / 4549 |
| bag_rank__seed_37_vs_rres512 | -1.682 | [-2.808, -0.556] | 125 / 207 / 4544 |

#### decontaminated (n=4876)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 90.012 | 4876/4876 | 2 | 0 |
| b1 | 88.413 | 4876/4876 | 5 | 0 |
| b2 | 88.208 | 4876/4876 | 4 | 0 |
| rres512 | 88.720 | 4876/4876 | 4 | 0 |
| bag_rank__seed_11 | 86.833 | 4876/4876 | 5 | 0 |
| bag_rank__seed_23 | 86.690 | 4876/4876 | 1 | 0 |
| bag_rank__seed_37 | 87.039 | 4876/4876 | 1 | 0 |

bag_rank: three-seed mean 86.854; range 86.690–87.039. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | -1.579 | [-2.688, -0.476] | 118 / 195 / 4563 |
| bag_rank__seed_11_vs_b2 | -1.374 | [-2.546, -0.242] | 129 / 196 / 4551 |
| bag_rank__seed_11_vs_rres512 | -1.887 | [-3.040, -0.704] | 122 / 214 / 4540 |
| bag_rank__seed_23_vs_b1 | -1.723 | [-2.964, -0.502] | 144 / 228 / 4504 |
| bag_rank__seed_23_vs_b2 | -1.518 | [-2.742, -0.323] | 148 / 222 / 4506 |
| bag_rank__seed_23_vs_rres512 | -2.030 | [-3.171, -0.880] | 127 / 226 / 4523 |
| bag_rank__seed_37_vs_b1 | -1.374 | [-2.523, -0.205] | 132 / 199 / 4545 |
| bag_rank__seed_37_vs_b2 | -1.169 | [-2.318, -0.041] | 135 / 192 / 4549 |
| bag_rank__seed_37_vs_rres512 | -1.682 | [-2.808, -0.556] | 125 / 207 / 4544 |

#### conservative (n=4876)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 90.012 | 4876/4876 | 2 | 0 |
| b1 | 88.413 | 4876/4876 | 5 | 0 |
| b2 | 88.208 | 4876/4876 | 4 | 0 |
| rres512 | 88.720 | 4876/4876 | 4 | 0 |
| bag_rank__seed_11 | 86.833 | 4876/4876 | 5 | 0 |
| bag_rank__seed_23 | 86.690 | 4876/4876 | 1 | 0 |
| bag_rank__seed_37 | 87.039 | 4876/4876 | 1 | 0 |

bag_rank: three-seed mean 86.854; range 86.690–87.039. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | -1.579 | [-2.688, -0.476] | 118 / 195 / 4563 |
| bag_rank__seed_11_vs_b2 | -1.374 | [-2.546, -0.242] | 129 / 196 / 4551 |
| bag_rank__seed_11_vs_rres512 | -1.887 | [-3.040, -0.704] | 122 / 214 / 4540 |
| bag_rank__seed_23_vs_b1 | -1.723 | [-2.964, -0.502] | 144 / 228 / 4504 |
| bag_rank__seed_23_vs_b2 | -1.518 | [-2.742, -0.323] | 148 / 222 / 4506 |
| bag_rank__seed_23_vs_rres512 | -2.030 | [-3.171, -0.880] | 127 / 226 / 4523 |
| bag_rank__seed_37_vs_b1 | -1.374 | [-2.523, -0.205] | 132 / 199 / 4545 |
| bag_rank__seed_37_vs_b2 | -1.169 | [-2.318, -0.041] | 135 / 192 / 4549 |
| bag_rank__seed_37_vs_rres512 | -1.682 | [-2.808, -0.556] | 125 / 207 / 4544 |

#### no_documented_history (n=0)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 未完成 | 0/0 | 0 | 0 |
| b1 | 未完成 | 0/0 | 0 | 0 |
| b2 | 未完成 | 0/0 | 0 | 0 |
| rres512 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_11 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_23 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_37 | 未完成 | 0/0 | 0 | 0 |

bag_rank: three-seed mean 未完成; range 未完成. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
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
| action_recognition (311) | 90.997 | 91.961 | 93.569 | 87.781 | 92.283 | 91.961 | 91.318 |
| attribute_comparison (163) | 92.638 | 87.730 | 86.503 | 87.730 | 82.209 | 76.687 | 81.595 |
| attribute_recognition (218) | 98.624 | 97.248 | 97.248 | 97.248 | 93.119 | 94.495 | 91.743 |
| celebrity_recognition (396) | 98.485 | 97.980 | 97.980 | 98.737 | 97.980 | 98.232 | 98.232 |
| function_reasoning (235) | 94.468 | 95.745 | 94.894 | 94.468 | 94.894 | 94.894 | 94.894 |
| future_prediction (177) | 84.746 | 86.441 | 83.051 | 86.441 | 80.791 | 80.791 | 82.486 |
| identity_reasoning (203) | 100.000 | 100.000 | 100.000 | 98.030 | 100.000 | 100.000 | 100.000 |
| image_emotion (240) | 89.583 | 89.583 | 88.750 | 90.833 | 90.833 | 92.083 | 91.250 |
| image_quality (275) | 63.636 | 61.455 | 61.455 | 59.636 | 50.182 | 51.273 | 47.636 |
| image_scene (382) | 93.455 | 92.670 | 92.670 | 93.194 | 93.455 | 94.503 | 93.717 |
| image_style (248) | 92.742 | 92.742 | 93.145 | 93.145 | 94.355 | 94.355 | 93.548 |
| image_topic (236) | 95.763 | 97.034 | 96.186 | 94.915 | 97.881 | 97.458 | 97.881 |
| nature_relation (207) | 89.855 | 86.957 | 87.440 | 90.338 | 89.372 | 89.372 | 89.855 |
| object_localization (274) | 88.686 | 81.022 | 79.927 | 87.226 | 76.642 | 74.453 | 78.102 |
| ocr (240) | 99.167 | 99.583 | 98.333 | 98.333 | 99.167 | 97.500 | 99.167 |
| physical_property_reasoning (165) | 86.667 | 81.818 | 81.818 | 81.818 | 76.364 | 78.788 | 72.727 |
| physical_relation (183) | 81.967 | 80.874 | 78.142 | 78.689 | 67.213 | 72.678 | 72.678 |
| social_relation (247) | 93.522 | 91.903 | 91.903 | 93.117 | 92.713 | 93.117 | 93.522 |
| spatial_relationship (193) | 78.238 | 75.648 | 73.057 | 76.166 | 75.130 | 71.503 | 76.166 |
| structuralized_imagetext_understanding (283) | 81.272 | 73.145 | 77.739 | 78.445 | 77.385 | 74.558 | 79.859 |

## MMBench_DEV_CN_V11: circular exact-matching accuracy

4876 questions / 1250 source parents / 1250 independent pixel/parent clusters. History exposure: {"unknown": 4876}.

source_pixel_sha256 first, then pixel_sha256 for images; parent_id fallback for no-image controls. Source parent identity and native benchmark group identities remain unchanged.

No configured historical full-prediction identity file; false flags do not prove no exposure.

Overlap audit: {"training_overlap_count": 0, "near_duplicate_suspected_count": 0}.

### Primary native benchmark metric

| Arm | Native score | Auxiliary row score (%) | Observed/expected | Unparsed |
|---|---:|---:|---:|---:|
| full | 84.443 | 89.110 | 4876/4876 | 3 |
| b1 | 81.579 | 87.203 | 4876/4876 | 2 |
| b2 | 82.198 | 87.387 | 4876/4876 | 2 |
| rres512 | 83.127 | 88.351 | 4876/4876 | 4 |
| bag_rank__seed_11 | 81.579 | 86.813 | 4876/4876 | 2 |
| bag_rank__seed_23 | 80.108 | 86.034 | 4876/4876 | 1 |
| bag_rank__seed_37 | 80.263 | 86.608 | 4876/4876 | 0 |

bag_rank: native three-seed mean 80.650; range 80.108–81.579. The range is not a confidence interval.

### Auxiliary per-row diagnostics

Auxiliary per-row correctness/ANLS differences only. These intervals are NOT confidence intervals for circular accuracy, MME points, POPE F1 or Hallusion grouped accuracy.

#### all (n=4876)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 89.110 | 4876/4876 | 1 | 0 |
| b1 | 87.203 | 4876/4876 | 0 | 0 |
| b2 | 87.387 | 4876/4876 | 1 | 0 |
| rres512 | 88.351 | 4876/4876 | 1 | 0 |
| bag_rank__seed_11 | 86.813 | 4876/4876 | 0 | 0 |
| bag_rank__seed_23 | 86.034 | 4876/4876 | 0 | 0 |
| bag_rank__seed_37 | 86.608 | 4876/4876 | 0 | 0 |

bag_rank: three-seed mean 86.485; range 86.034–86.813. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | -0.390 | [-1.441, 0.674] | 139 / 158 / 4579 |
| bag_rank__seed_11_vs_b2 | -0.574 | [-1.675, 0.480] | 139 / 167 / 4570 |
| bag_rank__seed_11_vs_rres512 | -1.538 | [-2.691, -0.391] | 128 / 203 / 4545 |
| bag_rank__seed_23_vs_b1 | -1.169 | [-2.316, -0.041] | 135 / 192 / 4549 |
| bag_rank__seed_23_vs_b2 | -1.354 | [-2.478, -0.245] | 132 / 198 / 4546 |
| bag_rank__seed_23_vs_rres512 | -2.317 | [-3.533, -1.146] | 123 / 236 / 4517 |
| bag_rank__seed_37_vs_b1 | -0.595 | [-1.793, 0.611] | 160 / 189 / 4527 |
| bag_rank__seed_37_vs_b2 | -0.779 | [-1.930, 0.368] | 152 / 190 / 4534 |
| bag_rank__seed_37_vs_rres512 | -1.743 | [-2.859, -0.612] | 132 / 217 / 4527 |

#### image_only (n=4876)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 89.110 | 4876/4876 | 1 | 0 |
| b1 | 87.203 | 4876/4876 | 0 | 0 |
| b2 | 87.387 | 4876/4876 | 1 | 0 |
| rres512 | 88.351 | 4876/4876 | 1 | 0 |
| bag_rank__seed_11 | 86.813 | 4876/4876 | 0 | 0 |
| bag_rank__seed_23 | 86.034 | 4876/4876 | 0 | 0 |
| bag_rank__seed_37 | 86.608 | 4876/4876 | 0 | 0 |

bag_rank: three-seed mean 86.485; range 86.034–86.813. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | -0.390 | [-1.441, 0.674] | 139 / 158 / 4579 |
| bag_rank__seed_11_vs_b2 | -0.574 | [-1.675, 0.480] | 139 / 167 / 4570 |
| bag_rank__seed_11_vs_rres512 | -1.538 | [-2.691, -0.391] | 128 / 203 / 4545 |
| bag_rank__seed_23_vs_b1 | -1.169 | [-2.316, -0.041] | 135 / 192 / 4549 |
| bag_rank__seed_23_vs_b2 | -1.354 | [-2.478, -0.245] | 132 / 198 / 4546 |
| bag_rank__seed_23_vs_rres512 | -2.317 | [-3.533, -1.146] | 123 / 236 / 4517 |
| bag_rank__seed_37_vs_b1 | -0.595 | [-1.793, 0.611] | 160 / 189 / 4527 |
| bag_rank__seed_37_vs_b2 | -0.779 | [-1.930, 0.368] | 152 / 190 / 4534 |
| bag_rank__seed_37_vs_rres512 | -1.743 | [-2.859, -0.612] | 132 / 217 / 4527 |

#### decontaminated (n=4876)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 89.110 | 4876/4876 | 1 | 0 |
| b1 | 87.203 | 4876/4876 | 0 | 0 |
| b2 | 87.387 | 4876/4876 | 1 | 0 |
| rres512 | 88.351 | 4876/4876 | 1 | 0 |
| bag_rank__seed_11 | 86.813 | 4876/4876 | 0 | 0 |
| bag_rank__seed_23 | 86.034 | 4876/4876 | 0 | 0 |
| bag_rank__seed_37 | 86.608 | 4876/4876 | 0 | 0 |

bag_rank: three-seed mean 86.485; range 86.034–86.813. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | -0.390 | [-1.441, 0.674] | 139 / 158 / 4579 |
| bag_rank__seed_11_vs_b2 | -0.574 | [-1.675, 0.480] | 139 / 167 / 4570 |
| bag_rank__seed_11_vs_rres512 | -1.538 | [-2.691, -0.391] | 128 / 203 / 4545 |
| bag_rank__seed_23_vs_b1 | -1.169 | [-2.316, -0.041] | 135 / 192 / 4549 |
| bag_rank__seed_23_vs_b2 | -1.354 | [-2.478, -0.245] | 132 / 198 / 4546 |
| bag_rank__seed_23_vs_rres512 | -2.317 | [-3.533, -1.146] | 123 / 236 / 4517 |
| bag_rank__seed_37_vs_b1 | -0.595 | [-1.793, 0.611] | 160 / 189 / 4527 |
| bag_rank__seed_37_vs_b2 | -0.779 | [-1.930, 0.368] | 152 / 190 / 4534 |
| bag_rank__seed_37_vs_rres512 | -1.743 | [-2.859, -0.612] | 132 / 217 / 4527 |

#### conservative (n=4876)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 89.110 | 4876/4876 | 1 | 0 |
| b1 | 87.203 | 4876/4876 | 0 | 0 |
| b2 | 87.387 | 4876/4876 | 1 | 0 |
| rres512 | 88.351 | 4876/4876 | 1 | 0 |
| bag_rank__seed_11 | 86.813 | 4876/4876 | 0 | 0 |
| bag_rank__seed_23 | 86.034 | 4876/4876 | 0 | 0 |
| bag_rank__seed_37 | 86.608 | 4876/4876 | 0 | 0 |

bag_rank: three-seed mean 86.485; range 86.034–86.813. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | -0.390 | [-1.441, 0.674] | 139 / 158 / 4579 |
| bag_rank__seed_11_vs_b2 | -0.574 | [-1.675, 0.480] | 139 / 167 / 4570 |
| bag_rank__seed_11_vs_rres512 | -1.538 | [-2.691, -0.391] | 128 / 203 / 4545 |
| bag_rank__seed_23_vs_b1 | -1.169 | [-2.316, -0.041] | 135 / 192 / 4549 |
| bag_rank__seed_23_vs_b2 | -1.354 | [-2.478, -0.245] | 132 / 198 / 4546 |
| bag_rank__seed_23_vs_rres512 | -2.317 | [-3.533, -1.146] | 123 / 236 / 4517 |
| bag_rank__seed_37_vs_b1 | -0.595 | [-1.793, 0.611] | 160 / 189 / 4527 |
| bag_rank__seed_37_vs_b2 | -0.779 | [-1.930, 0.368] | 152 / 190 / 4534 |
| bag_rank__seed_37_vs_rres512 | -1.743 | [-2.859, -0.612] | 132 / 217 / 4527 |

#### no_documented_history (n=0)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 未完成 | 0/0 | 0 | 0 |
| b1 | 未完成 | 0/0 | 0 | 0 |
| b2 | 未完成 | 0/0 | 0 | 0 |
| rres512 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_11 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_23 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_37 | 未完成 | 0/0 | 0 | 0 |

bag_rank: three-seed mean 未完成; range 未完成. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
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
| action_recognition (311) | 89.389 | 90.354 | 90.032 | 89.711 | 88.746 | 89.711 | 90.997 |
| attribute_comparison (163) | 90.798 | 82.209 | 82.209 | 90.184 | 87.117 | 71.779 | 78.528 |
| attribute_recognition (218) | 97.248 | 97.248 | 96.789 | 95.413 | 94.037 | 94.037 | 90.367 |
| celebrity_recognition (396) | 97.980 | 97.727 | 97.727 | 97.980 | 98.232 | 97.980 | 97.980 |
| function_reasoning (235) | 95.319 | 95.745 | 94.043 | 95.745 | 96.596 | 96.170 | 96.170 |
| future_prediction (177) | 81.356 | 80.791 | 80.226 | 79.096 | 79.096 | 78.531 | 75.141 |
| identity_reasoning (203) | 99.507 | 99.507 | 99.507 | 98.030 | 100.000 | 100.000 | 99.507 |
| image_emotion (240) | 86.667 | 88.750 | 87.917 | 87.917 | 87.083 | 87.500 | 87.500 |
| image_quality (275) | 61.091 | 57.818 | 60.000 | 54.545 | 49.455 | 51.636 | 52.364 |
| image_scene (382) | 92.670 | 92.932 | 92.932 | 92.147 | 92.147 | 92.932 | 92.932 |
| image_style (248) | 91.129 | 92.339 | 92.742 | 91.935 | 93.952 | 93.145 | 93.145 |
| image_topic (236) | 96.186 | 97.881 | 98.305 | 94.915 | 98.729 | 96.610 | 97.881 |
| nature_relation (207) | 93.720 | 86.957 | 85.990 | 90.821 | 91.787 | 89.855 | 90.338 |
| object_localization (274) | 87.956 | 79.197 | 78.832 | 87.226 | 76.277 | 72.993 | 78.467 |
| ocr (240) | 98.750 | 98.750 | 98.750 | 97.500 | 98.750 | 97.917 | 98.750 |
| physical_property_reasoning (165) | 81.818 | 80.606 | 81.212 | 83.636 | 84.848 | 84.242 | 77.576 |
| physical_relation (183) | 76.503 | 75.956 | 75.956 | 79.235 | 67.760 | 65.574 | 66.667 |
| social_relation (247) | 95.951 | 93.522 | 93.522 | 96.356 | 95.547 | 95.951 | 97.166 |
| spatial_relationship (193) | 79.275 | 74.611 | 75.130 | 78.756 | 69.948 | 73.057 | 73.057 |
| structuralized_imagetext_understanding (283) | 80.919 | 70.671 | 74.558 | 78.799 | 76.678 | 75.618 | 79.505 |

## MMStar: exact-matching accuracy

1500 questions / 1430 source parents / 1430 independent pixel/parent clusters. History exposure: {"unknown": 1500}.

source_pixel_sha256 first, then pixel_sha256 for images; parent_id fallback for no-image controls. Source parent identity and native benchmark group identities remain unchanged.

No configured historical full-prediction identity file; false flags do not prove no exposure.

Overlap audit: {"training_overlap_count": 0, "near_duplicate_suspected_count": 0}.

### Primary native benchmark metric

| Arm | Native score | Auxiliary row score (%) | Observed/expected | Unparsed |
|---|---:|---:|---:|---:|
| full | 66.333 | 66.333 | 1500/1500 | 11 |
| b1 | 60.933 | 60.933 | 1500/1500 | 8 |
| b2 | 61.200 | 61.200 | 1500/1500 | 10 |
| rres512 | 64.533 | 64.533 | 1500/1500 | 12 |
| bag_rank__seed_11 | 56.467 | 56.467 | 1500/1500 | 8 |
| bag_rank__seed_23 | 56.133 | 56.133 | 1500/1500 | 9 |
| bag_rank__seed_37 | 54.933 | 54.933 | 1500/1500 | 12 |

bag_rank: native three-seed mean 55.844; range 54.933–56.467. The range is not a confidence interval.

### Auxiliary per-row diagnostics

Auxiliary per-row correctness/ANLS differences only. These intervals are NOT confidence intervals for circular accuracy, MME points, POPE F1 or Hallusion grouped accuracy.

#### all (n=1500)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 66.333 | 1500/1500 | 0 | 0 |
| b1 | 60.933 | 1500/1500 | 1 | 0 |
| b2 | 61.200 | 1500/1500 | 1 | 0 |
| rres512 | 64.533 | 1500/1500 | 1 | 0 |
| bag_rank__seed_11 | 56.467 | 1500/1500 | 1 | 0 |
| bag_rank__seed_23 | 56.133 | 1500/1500 | 1 | 0 |
| bag_rank__seed_37 | 54.933 | 1500/1500 | 1 | 0 |

bag_rank: three-seed mean 55.844; range 54.933–56.467. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | -4.467 | [-6.622, -2.340] | 93 / 160 / 1247 |
| bag_rank__seed_11_vs_b2 | -4.733 | [-6.867, -2.674] | 87 / 158 / 1255 |
| bag_rank__seed_11_vs_rres512 | -8.067 | [-10.258, -5.930] | 80 / 201 / 1219 |
| bag_rank__seed_23_vs_b1 | -4.800 | [-6.994, -2.651] | 101 / 173 / 1226 |
| bag_rank__seed_23_vs_b2 | -5.067 | [-7.210, -2.931] | 93 / 169 / 1238 |
| bag_rank__seed_23_vs_rres512 | -8.400 | [-10.645, -6.142] | 87 / 213 / 1200 |
| bag_rank__seed_37_vs_b1 | -6.000 | [-8.182, -3.880] | 93 / 183 / 1224 |
| bag_rank__seed_37_vs_b2 | -6.267 | [-8.399, -4.144] | 85 / 179 / 1236 |
| bag_rank__seed_37_vs_rres512 | -9.600 | [-11.843, -7.397] | 79 / 223 / 1198 |

#### image_only (n=1500)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 66.333 | 1500/1500 | 0 | 0 |
| b1 | 60.933 | 1500/1500 | 1 | 0 |
| b2 | 61.200 | 1500/1500 | 1 | 0 |
| rres512 | 64.533 | 1500/1500 | 1 | 0 |
| bag_rank__seed_11 | 56.467 | 1500/1500 | 1 | 0 |
| bag_rank__seed_23 | 56.133 | 1500/1500 | 1 | 0 |
| bag_rank__seed_37 | 54.933 | 1500/1500 | 1 | 0 |

bag_rank: three-seed mean 55.844; range 54.933–56.467. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | -4.467 | [-6.622, -2.340] | 93 / 160 / 1247 |
| bag_rank__seed_11_vs_b2 | -4.733 | [-6.867, -2.674] | 87 / 158 / 1255 |
| bag_rank__seed_11_vs_rres512 | -8.067 | [-10.258, -5.930] | 80 / 201 / 1219 |
| bag_rank__seed_23_vs_b1 | -4.800 | [-6.994, -2.651] | 101 / 173 / 1226 |
| bag_rank__seed_23_vs_b2 | -5.067 | [-7.210, -2.931] | 93 / 169 / 1238 |
| bag_rank__seed_23_vs_rres512 | -8.400 | [-10.645, -6.142] | 87 / 213 / 1200 |
| bag_rank__seed_37_vs_b1 | -6.000 | [-8.182, -3.880] | 93 / 183 / 1224 |
| bag_rank__seed_37_vs_b2 | -6.267 | [-8.399, -4.144] | 85 / 179 / 1236 |
| bag_rank__seed_37_vs_rres512 | -9.600 | [-11.843, -7.397] | 79 / 223 / 1198 |

#### annotation_clean (n=1498)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 66.422 | 1498/1498 | 0 | 0 |
| b1 | 61.015 | 1498/1498 | 1 | 0 |
| b2 | 61.282 | 1498/1498 | 1 | 0 |
| rres512 | 64.619 | 1498/1498 | 1 | 0 |
| bag_rank__seed_11 | 56.542 | 1498/1498 | 1 | 0 |
| bag_rank__seed_23 | 56.208 | 1498/1498 | 1 | 0 |
| bag_rank__seed_37 | 55.007 | 1498/1498 | 1 | 0 |

bag_rank: three-seed mean 55.919; range 55.007–56.542. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | -4.473 | [-6.613, -2.403] | 93 / 160 / 1245 |
| bag_rank__seed_11_vs_b2 | -4.740 | [-6.836, -2.661] | 87 / 158 / 1253 |
| bag_rank__seed_11_vs_rres512 | -8.077 | [-10.294, -5.933] | 80 / 201 / 1217 |
| bag_rank__seed_23_vs_b1 | -4.806 | [-6.972, -2.658] | 101 / 173 / 1224 |
| bag_rank__seed_23_vs_b2 | -5.073 | [-7.190, -2.897] | 93 / 169 / 1236 |
| bag_rank__seed_23_vs_rres512 | -8.411 | [-10.643, -6.158] | 87 / 213 / 1198 |
| bag_rank__seed_37_vs_b1 | -6.008 | [-8.206, -3.898] | 93 / 183 / 1222 |
| bag_rank__seed_37_vs_b2 | -6.275 | [-8.466, -4.158] | 85 / 179 / 1234 |
| bag_rank__seed_37_vs_rres512 | -9.613 | [-11.854, -7.385] | 79 / 223 / 1196 |

#### decontaminated (n=1500)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 66.333 | 1500/1500 | 0 | 0 |
| b1 | 60.933 | 1500/1500 | 1 | 0 |
| b2 | 61.200 | 1500/1500 | 1 | 0 |
| rres512 | 64.533 | 1500/1500 | 1 | 0 |
| bag_rank__seed_11 | 56.467 | 1500/1500 | 1 | 0 |
| bag_rank__seed_23 | 56.133 | 1500/1500 | 1 | 0 |
| bag_rank__seed_37 | 54.933 | 1500/1500 | 1 | 0 |

bag_rank: three-seed mean 55.844; range 54.933–56.467. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | -4.467 | [-6.622, -2.340] | 93 / 160 / 1247 |
| bag_rank__seed_11_vs_b2 | -4.733 | [-6.867, -2.674] | 87 / 158 / 1255 |
| bag_rank__seed_11_vs_rres512 | -8.067 | [-10.258, -5.930] | 80 / 201 / 1219 |
| bag_rank__seed_23_vs_b1 | -4.800 | [-6.994, -2.651] | 101 / 173 / 1226 |
| bag_rank__seed_23_vs_b2 | -5.067 | [-7.210, -2.931] | 93 / 169 / 1238 |
| bag_rank__seed_23_vs_rres512 | -8.400 | [-10.645, -6.142] | 87 / 213 / 1200 |
| bag_rank__seed_37_vs_b1 | -6.000 | [-8.182, -3.880] | 93 / 183 / 1224 |
| bag_rank__seed_37_vs_b2 | -6.267 | [-8.399, -4.144] | 85 / 179 / 1236 |
| bag_rank__seed_37_vs_rres512 | -9.600 | [-11.843, -7.397] | 79 / 223 / 1198 |

#### conservative (n=1500)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 66.333 | 1500/1500 | 0 | 0 |
| b1 | 60.933 | 1500/1500 | 1 | 0 |
| b2 | 61.200 | 1500/1500 | 1 | 0 |
| rres512 | 64.533 | 1500/1500 | 1 | 0 |
| bag_rank__seed_11 | 56.467 | 1500/1500 | 1 | 0 |
| bag_rank__seed_23 | 56.133 | 1500/1500 | 1 | 0 |
| bag_rank__seed_37 | 54.933 | 1500/1500 | 1 | 0 |

bag_rank: three-seed mean 55.844; range 54.933–56.467. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | -4.467 | [-6.622, -2.340] | 93 / 160 / 1247 |
| bag_rank__seed_11_vs_b2 | -4.733 | [-6.867, -2.674] | 87 / 158 / 1255 |
| bag_rank__seed_11_vs_rres512 | -8.067 | [-10.258, -5.930] | 80 / 201 / 1219 |
| bag_rank__seed_23_vs_b1 | -4.800 | [-6.994, -2.651] | 101 / 173 / 1226 |
| bag_rank__seed_23_vs_b2 | -5.067 | [-7.210, -2.931] | 93 / 169 / 1238 |
| bag_rank__seed_23_vs_rres512 | -8.400 | [-10.645, -6.142] | 87 / 213 / 1200 |
| bag_rank__seed_37_vs_b1 | -6.000 | [-8.182, -3.880] | 93 / 183 / 1224 |
| bag_rank__seed_37_vs_b2 | -6.267 | [-8.399, -4.144] | 85 / 179 / 1236 |
| bag_rank__seed_37_vs_rres512 | -9.600 | [-11.843, -7.397] | 79 / 223 / 1198 |

#### no_documented_history (n=0)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 未完成 | 0/0 | 0 | 0 |
| b1 | 未完成 | 0/0 | 0 | 0 |
| b2 | 未完成 | 0/0 | 0 | 0 |
| rres512 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_11 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_23 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_37 | 未完成 | 0/0 | 0 | 0 |

bag_rank: three-seed mean 未完成; range 未完成. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
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
| coarse perception (250) | 74.400 | 75.200 | 75.600 | 75.200 | 70.000 | 68.800 | 67.200 |
| fine-grained perception (250) | 64.800 | 50.000 | 51.600 | 60.000 | 44.400 | 44.000 | 43.200 |
| instance reasoning (250) | 74.400 | 70.000 | 71.200 | 72.000 | 67.200 | 66.800 | 66.400 |
| logical reasoning (250) | 67.200 | 61.600 | 60.000 | 66.400 | 56.000 | 54.400 | 53.200 |
| math (250) | 67.200 | 60.400 | 61.200 | 64.400 | 60.000 | 61.600 | 60.800 |
| science & technology (250) | 50.000 | 48.400 | 47.600 | 49.200 | 41.200 | 41.200 | 38.800 |

## RealWorldQA: exact-matching accuracy

765 questions / 762 source parents / 762 independent pixel/parent clusters. History exposure: {"unknown": 765}.

source_pixel_sha256 first, then pixel_sha256 for images; parent_id fallback for no-image controls. Source parent identity and native benchmark group identities remain unchanged.

No configured historical full-prediction identity file; false flags do not prove no exposure.

Overlap audit: {"training_overlap_count": 0, "near_duplicate_suspected_count": 0}.

### Primary native benchmark metric

| Arm | Native score | Auxiliary row score (%) | Observed/expected | Unparsed |
|---|---:|---:|---:|---:|
| full | 71.895 | 71.895 | 765/765 | 0 |
| b1 | 66.275 | 66.275 | 765/765 | 0 |
| b2 | 66.667 | 66.667 | 765/765 | 0 |
| rres512 | 66.275 | 66.275 | 765/765 | 0 |
| bag_rank__seed_11 | 65.621 | 65.621 | 765/765 | 0 |
| bag_rank__seed_23 | 63.137 | 63.137 | 765/765 | 0 |
| bag_rank__seed_37 | 64.706 | 64.706 | 765/765 | 0 |

bag_rank: native three-seed mean 64.488; range 63.137–65.621. The range is not a confidence interval.

### Auxiliary per-row diagnostics

Auxiliary per-row correctness/ANLS differences only. These intervals are NOT confidence intervals for circular accuracy, MME points, POPE F1 or Hallusion grouped accuracy.

#### all (n=765)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 71.895 | 765/765 | 0 | 0 |
| b1 | 66.275 | 765/765 | 0 | 0 |
| b2 | 66.667 | 765/765 | 0 | 0 |
| rres512 | 66.275 | 765/765 | 0 | 0 |
| bag_rank__seed_11 | 65.621 | 765/765 | 0 | 0 |
| bag_rank__seed_23 | 63.137 | 765/765 | 0 | 0 |
| bag_rank__seed_37 | 64.706 | 765/765 | 0 | 0 |

bag_rank: three-seed mean 64.488; range 63.137–65.621. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | -0.654 | [-3.281, 2.092] | 53 / 58 / 654 |
| bag_rank__seed_11_vs_b2 | -1.046 | [-3.916, 1.830] | 57 / 65 / 643 |
| bag_rank__seed_11_vs_rres512 | -0.654 | [-3.801, 2.490] | 74 / 79 / 612 |
| bag_rank__seed_23_vs_b1 | -3.137 | [-5.898, -0.391] | 49 / 73 / 643 |
| bag_rank__seed_23_vs_b2 | -3.529 | [-6.405, -0.652] | 50 / 77 / 638 |
| bag_rank__seed_23_vs_rres512 | -3.137 | [-6.405, 0.131] | 68 / 92 / 605 |
| bag_rank__seed_37_vs_b1 | -1.569 | [-4.421, 1.307] | 55 / 67 / 643 |
| bag_rank__seed_37_vs_b2 | -1.961 | [-4.849, 0.916] | 57 / 72 / 636 |
| bag_rank__seed_37_vs_rres512 | -1.569 | [-4.818, 1.569] | 72 / 84 / 609 |

#### image_only (n=765)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 71.895 | 765/765 | 0 | 0 |
| b1 | 66.275 | 765/765 | 0 | 0 |
| b2 | 66.667 | 765/765 | 0 | 0 |
| rres512 | 66.275 | 765/765 | 0 | 0 |
| bag_rank__seed_11 | 65.621 | 765/765 | 0 | 0 |
| bag_rank__seed_23 | 63.137 | 765/765 | 0 | 0 |
| bag_rank__seed_37 | 64.706 | 765/765 | 0 | 0 |

bag_rank: three-seed mean 64.488; range 63.137–65.621. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | -0.654 | [-3.281, 2.092] | 53 / 58 / 654 |
| bag_rank__seed_11_vs_b2 | -1.046 | [-3.916, 1.830] | 57 / 65 / 643 |
| bag_rank__seed_11_vs_rres512 | -0.654 | [-3.801, 2.490] | 74 / 79 / 612 |
| bag_rank__seed_23_vs_b1 | -3.137 | [-5.898, -0.391] | 49 / 73 / 643 |
| bag_rank__seed_23_vs_b2 | -3.529 | [-6.405, -0.652] | 50 / 77 / 638 |
| bag_rank__seed_23_vs_rres512 | -3.137 | [-6.405, 0.131] | 68 / 92 / 605 |
| bag_rank__seed_37_vs_b1 | -1.569 | [-4.421, 1.307] | 55 / 67 / 643 |
| bag_rank__seed_37_vs_b2 | -1.961 | [-4.849, 0.916] | 57 / 72 / 636 |
| bag_rank__seed_37_vs_rres512 | -1.569 | [-4.818, 1.569] | 72 / 84 / 609 |

#### decontaminated (n=765)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 71.895 | 765/765 | 0 | 0 |
| b1 | 66.275 | 765/765 | 0 | 0 |
| b2 | 66.667 | 765/765 | 0 | 0 |
| rres512 | 66.275 | 765/765 | 0 | 0 |
| bag_rank__seed_11 | 65.621 | 765/765 | 0 | 0 |
| bag_rank__seed_23 | 63.137 | 765/765 | 0 | 0 |
| bag_rank__seed_37 | 64.706 | 765/765 | 0 | 0 |

bag_rank: three-seed mean 64.488; range 63.137–65.621. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | -0.654 | [-3.281, 2.092] | 53 / 58 / 654 |
| bag_rank__seed_11_vs_b2 | -1.046 | [-3.916, 1.830] | 57 / 65 / 643 |
| bag_rank__seed_11_vs_rres512 | -0.654 | [-3.801, 2.490] | 74 / 79 / 612 |
| bag_rank__seed_23_vs_b1 | -3.137 | [-5.898, -0.391] | 49 / 73 / 643 |
| bag_rank__seed_23_vs_b2 | -3.529 | [-6.405, -0.652] | 50 / 77 / 638 |
| bag_rank__seed_23_vs_rres512 | -3.137 | [-6.405, 0.131] | 68 / 92 / 605 |
| bag_rank__seed_37_vs_b1 | -1.569 | [-4.421, 1.307] | 55 / 67 / 643 |
| bag_rank__seed_37_vs_b2 | -1.961 | [-4.849, 0.916] | 57 / 72 / 636 |
| bag_rank__seed_37_vs_rres512 | -1.569 | [-4.818, 1.569] | 72 / 84 / 609 |

#### conservative (n=765)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 71.895 | 765/765 | 0 | 0 |
| b1 | 66.275 | 765/765 | 0 | 0 |
| b2 | 66.667 | 765/765 | 0 | 0 |
| rres512 | 66.275 | 765/765 | 0 | 0 |
| bag_rank__seed_11 | 65.621 | 765/765 | 0 | 0 |
| bag_rank__seed_23 | 63.137 | 765/765 | 0 | 0 |
| bag_rank__seed_37 | 64.706 | 765/765 | 0 | 0 |

bag_rank: three-seed mean 64.488; range 63.137–65.621. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | -0.654 | [-3.281, 2.092] | 53 / 58 / 654 |
| bag_rank__seed_11_vs_b2 | -1.046 | [-3.916, 1.830] | 57 / 65 / 643 |
| bag_rank__seed_11_vs_rres512 | -0.654 | [-3.801, 2.490] | 74 / 79 / 612 |
| bag_rank__seed_23_vs_b1 | -3.137 | [-5.898, -0.391] | 49 / 73 / 643 |
| bag_rank__seed_23_vs_b2 | -3.529 | [-6.405, -0.652] | 50 / 77 / 638 |
| bag_rank__seed_23_vs_rres512 | -3.137 | [-6.405, 0.131] | 68 / 92 / 605 |
| bag_rank__seed_37_vs_b1 | -1.569 | [-4.421, 1.307] | 55 / 67 / 643 |
| bag_rank__seed_37_vs_b2 | -1.961 | [-4.849, 0.916] | 57 / 72 / 636 |
| bag_rank__seed_37_vs_rres512 | -1.569 | [-4.818, 1.569] | 72 / 84 / 609 |

#### no_documented_history (n=0)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 未完成 | 0/0 | 0 | 0 |
| b1 | 未完成 | 0/0 | 0 | 0 |
| b2 | 未完成 | 0/0 | 0 | 0 |
| rres512 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_11 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_23 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_37 | 未完成 | 0/0 | 0 | 0 |

bag_rank: three-seed mean 未完成; range 未完成. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
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

## ChartQA_TEST: relaxed accuracy

2500 questions / 1509 source parents / 1509 independent pixel/parent clusters. History exposure: {"unknown": 2500}.

source_pixel_sha256 first, then pixel_sha256 for images; parent_id fallback for no-image controls. Source parent identity and native benchmark group identities remain unchanged.

No configured historical full-prediction identity file; false flags do not prove no exposure.

Overlap audit: {"training_overlap_count": 0, "near_duplicate_suspected_count": 0}.

### Primary native benchmark metric

| Arm | Native score | Auxiliary row score (%) | Observed/expected | Unparsed |
|---|---:|---:|---:|---:|
| full | 82.320 | 82.320 | 2500/2500 | 0 |
| b1 | 66.880 | 66.880 | 2500/2500 | 0 |
| b2 | 67.800 | 67.800 | 2500/2500 | 0 |
| rres512 | 76.920 | 76.920 | 2500/2500 | 0 |
| bag_rank__seed_11 | 71.840 | 71.840 | 2500/2500 | 0 |
| bag_rank__seed_23 | 70.920 | 70.920 | 2500/2500 | 0 |
| bag_rank__seed_37 | 70.480 | 70.480 | 2500/2500 | 0 |

bag_rank: native three-seed mean 71.080; range 70.480–71.840. The range is not a confidence interval.

### Auxiliary per-row diagnostics

Auxiliary per-row correctness/ANLS differences only. These intervals are NOT confidence intervals for circular accuracy, MME points, POPE F1 or Hallusion grouped accuracy.

#### all (n=2500)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 82.320 | 2500/2500 | 0 | 0 |
| b1 | 66.880 | 2500/2500 | 0 | 0 |
| b2 | 67.800 | 2500/2500 | 0 | 0 |
| rres512 | 76.920 | 2500/2500 | 0 | 0 |
| bag_rank__seed_11 | 71.840 | 2500/2500 | 0 | 0 |
| bag_rank__seed_23 | 70.920 | 2500/2500 | 0 | 0 |
| bag_rank__seed_37 | 70.480 | 2500/2500 | 0 | 0 |

bag_rank: three-seed mean 71.080; range 70.480–71.840. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 4.960 | [3.085, 6.815] | 323 / 199 / 1978 |
| bag_rank__seed_11_vs_b2 | 4.040 | [2.132, 5.922] | 306 / 205 / 1989 |
| bag_rank__seed_11_vs_rres512 | -5.080 | [-6.727, -3.415] | 144 / 271 / 2085 |
| bag_rank__seed_23_vs_b1 | 4.040 | [2.134, 5.967] | 321 / 220 / 1959 |
| bag_rank__seed_23_vs_b2 | 3.120 | [1.322, 4.958] | 299 / 221 / 1980 |
| bag_rank__seed_23_vs_rres512 | -6.000 | [-7.673, -4.367] | 142 / 292 / 2066 |
| bag_rank__seed_37_vs_b1 | 3.600 | [1.683, 5.477] | 318 / 228 / 1954 |
| bag_rank__seed_37_vs_b2 | 2.680 | [0.761, 4.513] | 304 / 237 / 1959 |
| bag_rank__seed_37_vs_rres512 | -6.440 | [-8.097, -4.798] | 140 / 301 / 2059 |

#### image_only (n=2500)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 82.320 | 2500/2500 | 0 | 0 |
| b1 | 66.880 | 2500/2500 | 0 | 0 |
| b2 | 67.800 | 2500/2500 | 0 | 0 |
| rres512 | 76.920 | 2500/2500 | 0 | 0 |
| bag_rank__seed_11 | 71.840 | 2500/2500 | 0 | 0 |
| bag_rank__seed_23 | 70.920 | 2500/2500 | 0 | 0 |
| bag_rank__seed_37 | 70.480 | 2500/2500 | 0 | 0 |

bag_rank: three-seed mean 71.080; range 70.480–71.840. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 4.960 | [3.085, 6.815] | 323 / 199 / 1978 |
| bag_rank__seed_11_vs_b2 | 4.040 | [2.132, 5.922] | 306 / 205 / 1989 |
| bag_rank__seed_11_vs_rres512 | -5.080 | [-6.727, -3.415] | 144 / 271 / 2085 |
| bag_rank__seed_23_vs_b1 | 4.040 | [2.134, 5.967] | 321 / 220 / 1959 |
| bag_rank__seed_23_vs_b2 | 3.120 | [1.322, 4.958] | 299 / 221 / 1980 |
| bag_rank__seed_23_vs_rres512 | -6.000 | [-7.673, -4.367] | 142 / 292 / 2066 |
| bag_rank__seed_37_vs_b1 | 3.600 | [1.683, 5.477] | 318 / 228 / 1954 |
| bag_rank__seed_37_vs_b2 | 2.680 | [0.761, 4.513] | 304 / 237 / 1959 |
| bag_rank__seed_37_vs_rres512 | -6.440 | [-8.097, -4.798] | 140 / 301 / 2059 |

#### decontaminated (n=2500)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 82.320 | 2500/2500 | 0 | 0 |
| b1 | 66.880 | 2500/2500 | 0 | 0 |
| b2 | 67.800 | 2500/2500 | 0 | 0 |
| rres512 | 76.920 | 2500/2500 | 0 | 0 |
| bag_rank__seed_11 | 71.840 | 2500/2500 | 0 | 0 |
| bag_rank__seed_23 | 70.920 | 2500/2500 | 0 | 0 |
| bag_rank__seed_37 | 70.480 | 2500/2500 | 0 | 0 |

bag_rank: three-seed mean 71.080; range 70.480–71.840. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 4.960 | [3.085, 6.815] | 323 / 199 / 1978 |
| bag_rank__seed_11_vs_b2 | 4.040 | [2.132, 5.922] | 306 / 205 / 1989 |
| bag_rank__seed_11_vs_rres512 | -5.080 | [-6.727, -3.415] | 144 / 271 / 2085 |
| bag_rank__seed_23_vs_b1 | 4.040 | [2.134, 5.967] | 321 / 220 / 1959 |
| bag_rank__seed_23_vs_b2 | 3.120 | [1.322, 4.958] | 299 / 221 / 1980 |
| bag_rank__seed_23_vs_rres512 | -6.000 | [-7.673, -4.367] | 142 / 292 / 2066 |
| bag_rank__seed_37_vs_b1 | 3.600 | [1.683, 5.477] | 318 / 228 / 1954 |
| bag_rank__seed_37_vs_b2 | 2.680 | [0.761, 4.513] | 304 / 237 / 1959 |
| bag_rank__seed_37_vs_rres512 | -6.440 | [-8.097, -4.798] | 140 / 301 / 2059 |

#### conservative (n=2500)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 82.320 | 2500/2500 | 0 | 0 |
| b1 | 66.880 | 2500/2500 | 0 | 0 |
| b2 | 67.800 | 2500/2500 | 0 | 0 |
| rres512 | 76.920 | 2500/2500 | 0 | 0 |
| bag_rank__seed_11 | 71.840 | 2500/2500 | 0 | 0 |
| bag_rank__seed_23 | 70.920 | 2500/2500 | 0 | 0 |
| bag_rank__seed_37 | 70.480 | 2500/2500 | 0 | 0 |

bag_rank: three-seed mean 71.080; range 70.480–71.840. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 4.960 | [3.085, 6.815] | 323 / 199 / 1978 |
| bag_rank__seed_11_vs_b2 | 4.040 | [2.132, 5.922] | 306 / 205 / 1989 |
| bag_rank__seed_11_vs_rres512 | -5.080 | [-6.727, -3.415] | 144 / 271 / 2085 |
| bag_rank__seed_23_vs_b1 | 4.040 | [2.134, 5.967] | 321 / 220 / 1959 |
| bag_rank__seed_23_vs_b2 | 3.120 | [1.322, 4.958] | 299 / 221 / 1980 |
| bag_rank__seed_23_vs_rres512 | -6.000 | [-7.673, -4.367] | 142 / 292 / 2066 |
| bag_rank__seed_37_vs_b1 | 3.600 | [1.683, 5.477] | 318 / 228 / 1954 |
| bag_rank__seed_37_vs_b2 | 2.680 | [0.761, 4.513] | 304 / 237 / 1959 |
| bag_rank__seed_37_vs_rres512 | -6.440 | [-8.097, -4.798] | 140 / 301 / 2059 |

#### no_documented_history (n=0)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 未完成 | 0/0 | 0 | 0 |
| b1 | 未完成 | 0/0 | 0 | 0 |
| b2 | 未完成 | 0/0 | 0 | 0 |
| rres512 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_11 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_23 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_37 | 未完成 | 0/0 | 0 | 0 |

bag_rank: three-seed mean 未完成; range 未完成. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
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

## HallusionBench: HallusionBench aAcc (exact matching)

1129 questions / 167 source parents / 403 independent pixel/parent clusters. History exposure: {"unknown": 1129}.

source_pixel_sha256 first, then pixel_sha256 for images; parent_id fallback for no-image controls. Source parent identity and native benchmark group identities remain unchanged.

No configured historical full-prediction identity file; false flags do not prove no exposure.

Overlap audit: {"training_overlap_count": 0, "near_duplicate_suspected_count": 0}.

HallusionBench all=1129 is the complete-original-table extended evaluation (951 image questions + 178 text-only controls). It is not the conventional image-mode denominator. The separate image_only=951 panel matches the pinned VLMEvalKit default skip_noimg=True image-mode roster.

### Primary native benchmark metric

| Arm | Native score | Auxiliary row score (%) | Observed/expected | Unparsed |
|---|---:|---:|---:|---:|
| full | 71.036 | 71.036 | 1129/1129 | 13 |
| b1 | 67.759 | 67.759 | 1129/1129 | 13 |
| b2 | 67.759 | 67.759 | 1129/1129 | 6 |
| rres512 | 71.213 | 71.213 | 1129/1129 | 18 |
| bag_rank__seed_11 | 68.468 | 68.468 | 1129/1129 | 22 |
| bag_rank__seed_23 | 67.671 | 67.671 | 1129/1129 | 20 |
| bag_rank__seed_37 | 68.113 | 68.113 | 1129/1129 | 27 |

bag_rank: native three-seed mean 68.084; range 67.671–68.468. The range is not a confidence interval.

| Arm | aAcc | fAcc | qAcc |
|---|---:|---:|---:|
| full | 71.036 | 48.139 | 40.659 |
| b1 | 67.759 | 41.687 | 35.385 |
| b2 | 67.759 | 42.432 | 35.604 |
| rres512 | 71.213 | 47.395 | 41.538 |
| bag_rank__seed_11 | 68.468 | 45.658 | 38.242 |
| bag_rank__seed_23 | 67.671 | 44.169 | 36.484 |
| bag_rank__seed_37 | 68.113 | 45.161 | 38.022 |

Detailed native category/group metrics are preserved in report.json.

### Native image-only sensitivity (image-free controls excluded)

This is the conventional HallusionBench image-mode roster (951 questions with the pinned VLMEvalKit default skip_noimg=True). The all=1129 extended panel above retains all original questions, including 178 text-only controls.

| Arm | Native score (%) | Questions |
|---|---:|---:|
| full | 71.399 | 951/951 |
| b1 | 67.508 | 951/951 |
| b2 | 67.508 | 951/951 |
| rres512 | 71.609 | 951/951 |
| bag_rank__seed_11 | 68.349 | 951/951 |
| bag_rank__seed_23 | 67.403 | 951/951 |
| bag_rank__seed_37 | 67.928 | 951/951 |

### Auxiliary per-row diagnostics

Auxiliary per-row correctness/ANLS differences only. These intervals are NOT confidence intervals for circular accuracy, MME points, POPE F1 or Hallusion grouped accuracy.

#### all (n=1129)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 71.036 | 1129/1129 | 19 | 0 |
| b1 | 67.759 | 1129/1129 | 36 | 0 |
| b2 | 67.759 | 1129/1129 | 26 | 0 |
| rres512 | 71.213 | 1129/1129 | 19 | 0 |
| bag_rank__seed_11 | 68.468 | 1129/1129 | 34 | 0 |
| bag_rank__seed_23 | 67.671 | 1129/1129 | 32 | 0 |
| bag_rank__seed_37 | 68.113 | 1129/1129 | 37 | 0 |

bag_rank: three-seed mean 68.084; range 67.671–68.468. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 0.709 | [-1.218, 2.684] | 71 / 63 / 995 |
| bag_rank__seed_11_vs_b2 | 0.709 | [-1.344, 2.788] | 70 / 62 / 997 |
| bag_rank__seed_11_vs_rres512 | -2.746 | [-4.774, -0.696] | 47 / 78 / 1004 |
| bag_rank__seed_23_vs_b1 | -0.089 | [-2.044, 1.885] | 67 / 68 / 994 |
| bag_rank__seed_23_vs_b2 | -0.089 | [-2.172, 2.017] | 68 / 69 / 992 |
| bag_rank__seed_23_vs_rres512 | -3.543 | [-5.530, -1.600] | 40 / 80 / 1009 |
| bag_rank__seed_37_vs_b1 | 0.354 | [-1.670, 2.423] | 66 / 62 / 1001 |
| bag_rank__seed_37_vs_b2 | 0.354 | [-1.879, 2.632] | 72 / 68 / 989 |
| bag_rank__seed_37_vs_rres512 | -3.100 | [-5.198, -0.995] | 41 / 76 / 1012 |

#### image_only (n=951)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 71.399 | 951/951 | 19 | 0 |
| b1 | 67.508 | 951/951 | 36 | 0 |
| b2 | 67.508 | 951/951 | 26 | 0 |
| rres512 | 71.609 | 951/951 | 19 | 0 |
| bag_rank__seed_11 | 68.349 | 951/951 | 34 | 0 |
| bag_rank__seed_23 | 67.403 | 951/951 | 32 | 0 |
| bag_rank__seed_37 | 67.928 | 951/951 | 37 | 0 |

bag_rank: three-seed mean 67.893; range 67.403–68.349. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 0.841 | [-1.421, 3.188] | 71 / 63 / 817 |
| bag_rank__seed_11_vs_b2 | 0.841 | [-1.613, 3.348] | 70 / 62 / 819 |
| bag_rank__seed_11_vs_rres512 | -3.260 | [-5.708, -0.842] | 47 / 78 / 826 |
| bag_rank__seed_23_vs_b1 | -0.105 | [-2.408, 2.236] | 67 / 68 / 816 |
| bag_rank__seed_23_vs_b2 | -0.105 | [-2.534, 2.411] | 68 / 69 / 814 |
| bag_rank__seed_23_vs_rres512 | -4.206 | [-6.597, -1.810] | 40 / 80 / 831 |
| bag_rank__seed_37_vs_b1 | 0.421 | [-1.959, 2.812] | 66 / 62 / 823 |
| bag_rank__seed_37_vs_b2 | 0.421 | [-2.249, 3.175] | 72 / 68 / 811 |
| bag_rank__seed_37_vs_rres512 | -3.680 | [-6.174, -1.240] | 41 / 76 / 834 |

#### text_only_control (n=178)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 69.101 | 178/178 | 0 | 0 |
| b1 | 69.101 | 178/178 | 0 | 0 |
| b2 | 69.101 | 178/178 | 0 | 0 |
| rres512 | 69.101 | 178/178 | 0 | 0 |
| bag_rank__seed_11 | 69.101 | 178/178 | 0 | 0 |
| bag_rank__seed_23 | 69.101 | 178/178 | 0 | 0 |
| bag_rank__seed_37 | 69.101 | 178/178 | 0 | 0 |

bag_rank: three-seed mean 69.101; range 69.101–69.101. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 0.000 | [0.000, 0.000] | 0 / 0 / 178 |
| bag_rank__seed_11_vs_b2 | 0.000 | [0.000, 0.000] | 0 / 0 / 178 |
| bag_rank__seed_11_vs_rres512 | 0.000 | [0.000, 0.000] | 0 / 0 / 178 |
| bag_rank__seed_23_vs_b1 | 0.000 | [0.000, 0.000] | 0 / 0 / 178 |
| bag_rank__seed_23_vs_b2 | 0.000 | [0.000, 0.000] | 0 / 0 / 178 |
| bag_rank__seed_23_vs_rres512 | 0.000 | [0.000, 0.000] | 0 / 0 / 178 |
| bag_rank__seed_37_vs_b1 | 0.000 | [0.000, 0.000] | 0 / 0 / 178 |
| bag_rank__seed_37_vs_b2 | 0.000 | [0.000, 0.000] | 0 / 0 / 178 |
| bag_rank__seed_37_vs_rres512 | 0.000 | [0.000, 0.000] | 0 / 0 / 178 |

#### decontaminated (n=1129)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 71.036 | 1129/1129 | 19 | 0 |
| b1 | 67.759 | 1129/1129 | 36 | 0 |
| b2 | 67.759 | 1129/1129 | 26 | 0 |
| rres512 | 71.213 | 1129/1129 | 19 | 0 |
| bag_rank__seed_11 | 68.468 | 1129/1129 | 34 | 0 |
| bag_rank__seed_23 | 67.671 | 1129/1129 | 32 | 0 |
| bag_rank__seed_37 | 68.113 | 1129/1129 | 37 | 0 |

bag_rank: three-seed mean 68.084; range 67.671–68.468. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 0.709 | [-1.218, 2.684] | 71 / 63 / 995 |
| bag_rank__seed_11_vs_b2 | 0.709 | [-1.344, 2.788] | 70 / 62 / 997 |
| bag_rank__seed_11_vs_rres512 | -2.746 | [-4.774, -0.696] | 47 / 78 / 1004 |
| bag_rank__seed_23_vs_b1 | -0.089 | [-2.044, 1.885] | 67 / 68 / 994 |
| bag_rank__seed_23_vs_b2 | -0.089 | [-2.172, 2.017] | 68 / 69 / 992 |
| bag_rank__seed_23_vs_rres512 | -3.543 | [-5.530, -1.600] | 40 / 80 / 1009 |
| bag_rank__seed_37_vs_b1 | 0.354 | [-1.670, 2.423] | 66 / 62 / 1001 |
| bag_rank__seed_37_vs_b2 | 0.354 | [-1.879, 2.632] | 72 / 68 / 989 |
| bag_rank__seed_37_vs_rres512 | -3.100 | [-5.198, -0.995] | 41 / 76 / 1012 |

#### conservative (n=1129)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 71.036 | 1129/1129 | 19 | 0 |
| b1 | 67.759 | 1129/1129 | 36 | 0 |
| b2 | 67.759 | 1129/1129 | 26 | 0 |
| rres512 | 71.213 | 1129/1129 | 19 | 0 |
| bag_rank__seed_11 | 68.468 | 1129/1129 | 34 | 0 |
| bag_rank__seed_23 | 67.671 | 1129/1129 | 32 | 0 |
| bag_rank__seed_37 | 68.113 | 1129/1129 | 37 | 0 |

bag_rank: three-seed mean 68.084; range 67.671–68.468. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 0.709 | [-1.218, 2.684] | 71 / 63 / 995 |
| bag_rank__seed_11_vs_b2 | 0.709 | [-1.344, 2.788] | 70 / 62 / 997 |
| bag_rank__seed_11_vs_rres512 | -2.746 | [-4.774, -0.696] | 47 / 78 / 1004 |
| bag_rank__seed_23_vs_b1 | -0.089 | [-2.044, 1.885] | 67 / 68 / 994 |
| bag_rank__seed_23_vs_b2 | -0.089 | [-2.172, 2.017] | 68 / 69 / 992 |
| bag_rank__seed_23_vs_rres512 | -3.543 | [-5.530, -1.600] | 40 / 80 / 1009 |
| bag_rank__seed_37_vs_b1 | 0.354 | [-1.670, 2.423] | 66 / 62 / 1001 |
| bag_rank__seed_37_vs_b2 | 0.354 | [-1.879, 2.632] | 72 / 68 / 989 |
| bag_rank__seed_37_vs_rres512 | -3.100 | [-5.198, -0.995] | 41 / 76 / 1012 |

#### no_documented_history (n=0)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 未完成 | 0/0 | 0 | 0 |
| b1 | 未完成 | 0/0 | 0 | 0 |
| b2 | 未完成 | 0/0 | 0 | 0 |
| rres512 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_11 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_23 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_37 | 未完成 | 0/0 | 0 | 0 |

bag_rank: three-seed mean 未完成; range 未完成. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
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
| VD (591) | 65.144 | 63.283 | 64.467 | 67.005 | 63.959 | 62.775 | 62.944 |
| VS (538) | 77.509 | 72.677 | 71.375 | 75.836 | 73.420 | 73.048 | 73.792 |

## MME: MME perception + reasoning points

2374 questions / 1187 source parents / 1177 independent pixel/parent clusters. History exposure: {"unknown": 2374}.

source_pixel_sha256 first, then pixel_sha256 for images; parent_id fallback for no-image controls. Source parent identity and native benchmark group identities remain unchanged.

No configured historical full-prediction identity file; false flags do not prove no exposure.

Overlap audit: {"training_overlap_count": 0, "near_duplicate_suspected_count": 0}.

### Primary native benchmark metric

| Arm | Native score | Auxiliary row score (%) | Observed/expected | Unparsed |
|---|---:|---:|---:|---:|
| full | 2433.495 | 89.638 | 2374/2374 | 2 |
| b1 | 2331.855 | 88.837 | 2374/2374 | 0 |
| b2 | 2316.224 | 88.332 | 2374/2374 | 0 |
| rres512 | 2440.729 | 89.343 | 2374/2374 | 2 |
| bag_rank__seed_11 | 2339.451 | 88.964 | 2374/2374 | 2 |
| bag_rank__seed_23 | 2339.841 | 88.880 | 2374/2374 | 2 |
| bag_rank__seed_37 | 2388.018 | 89.132 | 2374/2374 | 2 |

bag_rank: native three-seed mean 2355.770; range 2339.451–2388.018. The range is not a confidence interval.

| Arm | perception | reasoning |
|---|---:|---:|
| full | 1767.781 | 665.714 |
| b1 | 1734.355 | 597.500 |
| b2 | 1693.724 | 622.500 |
| rres512 | 1770.729 | 670.000 |
| bag_rank__seed_11 | 1695.522 | 643.929 |
| bag_rank__seed_23 | 1693.769 | 646.071 |
| bag_rank__seed_37 | 1722.303 | 665.714 |

Detailed native category/group metrics are preserved in report.json.

### Auxiliary per-row diagnostics

Auxiliary per-row correctness/ANLS differences only. These intervals are NOT confidence intervals for circular accuracy, MME points, POPE F1 or Hallusion grouped accuracy.

#### all (n=2374)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 89.638 | 2374/2374 | 2 | 0 |
| b1 | 88.837 | 2374/2374 | 0 | 0 |
| b2 | 88.332 | 2374/2374 | 0 | 0 |
| rres512 | 89.343 | 2374/2374 | 2 | 0 |
| bag_rank__seed_11 | 88.964 | 2374/2374 | 2 | 0 |
| bag_rank__seed_23 | 88.880 | 2374/2374 | 2 | 0 |
| bag_rank__seed_37 | 89.132 | 2374/2374 | 2 | 0 |

bag_rank: three-seed mean 88.992; range 88.880–89.132. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 0.126 | [-0.841, 1.096] | 64 / 61 / 2249 |
| bag_rank__seed_11_vs_b2 | 0.632 | [-0.295, 1.560] | 70 / 55 / 2249 |
| bag_rank__seed_11_vs_rres512 | -0.379 | [-1.306, 0.549] | 56 / 65 / 2253 |
| bag_rank__seed_23_vs_b1 | 0.042 | [-0.923, 1.008] | 64 / 63 / 2247 |
| bag_rank__seed_23_vs_b2 | 0.548 | [-0.379, 1.471] | 72 / 59 / 2243 |
| bag_rank__seed_23_vs_rres512 | -0.463 | [-1.388, 0.463] | 54 / 65 / 2255 |
| bag_rank__seed_37_vs_b1 | 0.295 | [-0.629, 1.221] | 62 / 55 / 2257 |
| bag_rank__seed_37_vs_b2 | 0.800 | [-0.084, 1.723] | 71 / 52 / 2251 |
| bag_rank__seed_37_vs_rres512 | -0.211 | [-1.094, 0.675] | 53 / 58 / 2263 |

#### image_only (n=2374)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 89.638 | 2374/2374 | 2 | 0 |
| b1 | 88.837 | 2374/2374 | 0 | 0 |
| b2 | 88.332 | 2374/2374 | 0 | 0 |
| rres512 | 89.343 | 2374/2374 | 2 | 0 |
| bag_rank__seed_11 | 88.964 | 2374/2374 | 2 | 0 |
| bag_rank__seed_23 | 88.880 | 2374/2374 | 2 | 0 |
| bag_rank__seed_37 | 89.132 | 2374/2374 | 2 | 0 |

bag_rank: three-seed mean 88.992; range 88.880–89.132. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 0.126 | [-0.841, 1.096] | 64 / 61 / 2249 |
| bag_rank__seed_11_vs_b2 | 0.632 | [-0.295, 1.560] | 70 / 55 / 2249 |
| bag_rank__seed_11_vs_rres512 | -0.379 | [-1.306, 0.549] | 56 / 65 / 2253 |
| bag_rank__seed_23_vs_b1 | 0.042 | [-0.923, 1.008] | 64 / 63 / 2247 |
| bag_rank__seed_23_vs_b2 | 0.548 | [-0.379, 1.471] | 72 / 59 / 2243 |
| bag_rank__seed_23_vs_rres512 | -0.463 | [-1.388, 0.463] | 54 / 65 / 2255 |
| bag_rank__seed_37_vs_b1 | 0.295 | [-0.629, 1.221] | 62 / 55 / 2257 |
| bag_rank__seed_37_vs_b2 | 0.800 | [-0.084, 1.723] | 71 / 52 / 2251 |
| bag_rank__seed_37_vs_rres512 | -0.211 | [-1.094, 0.675] | 53 / 58 / 2263 |

#### decontaminated (n=2374)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 89.638 | 2374/2374 | 2 | 0 |
| b1 | 88.837 | 2374/2374 | 0 | 0 |
| b2 | 88.332 | 2374/2374 | 0 | 0 |
| rres512 | 89.343 | 2374/2374 | 2 | 0 |
| bag_rank__seed_11 | 88.964 | 2374/2374 | 2 | 0 |
| bag_rank__seed_23 | 88.880 | 2374/2374 | 2 | 0 |
| bag_rank__seed_37 | 89.132 | 2374/2374 | 2 | 0 |

bag_rank: three-seed mean 88.992; range 88.880–89.132. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 0.126 | [-0.841, 1.096] | 64 / 61 / 2249 |
| bag_rank__seed_11_vs_b2 | 0.632 | [-0.295, 1.560] | 70 / 55 / 2249 |
| bag_rank__seed_11_vs_rres512 | -0.379 | [-1.306, 0.549] | 56 / 65 / 2253 |
| bag_rank__seed_23_vs_b1 | 0.042 | [-0.923, 1.008] | 64 / 63 / 2247 |
| bag_rank__seed_23_vs_b2 | 0.548 | [-0.379, 1.471] | 72 / 59 / 2243 |
| bag_rank__seed_23_vs_rres512 | -0.463 | [-1.388, 0.463] | 54 / 65 / 2255 |
| bag_rank__seed_37_vs_b1 | 0.295 | [-0.629, 1.221] | 62 / 55 / 2257 |
| bag_rank__seed_37_vs_b2 | 0.800 | [-0.084, 1.723] | 71 / 52 / 2251 |
| bag_rank__seed_37_vs_rres512 | -0.211 | [-1.094, 0.675] | 53 / 58 / 2263 |

#### conservative (n=2374)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 89.638 | 2374/2374 | 2 | 0 |
| b1 | 88.837 | 2374/2374 | 0 | 0 |
| b2 | 88.332 | 2374/2374 | 0 | 0 |
| rres512 | 89.343 | 2374/2374 | 2 | 0 |
| bag_rank__seed_11 | 88.964 | 2374/2374 | 2 | 0 |
| bag_rank__seed_23 | 88.880 | 2374/2374 | 2 | 0 |
| bag_rank__seed_37 | 89.132 | 2374/2374 | 2 | 0 |

bag_rank: three-seed mean 88.992; range 88.880–89.132. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 0.126 | [-0.841, 1.096] | 64 / 61 / 2249 |
| bag_rank__seed_11_vs_b2 | 0.632 | [-0.295, 1.560] | 70 / 55 / 2249 |
| bag_rank__seed_11_vs_rres512 | -0.379 | [-1.306, 0.549] | 56 / 65 / 2253 |
| bag_rank__seed_23_vs_b1 | 0.042 | [-0.923, 1.008] | 64 / 63 / 2247 |
| bag_rank__seed_23_vs_b2 | 0.548 | [-0.379, 1.471] | 72 / 59 / 2243 |
| bag_rank__seed_23_vs_rres512 | -0.463 | [-1.388, 0.463] | 54 / 65 / 2255 |
| bag_rank__seed_37_vs_b1 | 0.295 | [-0.629, 1.221] | 62 / 55 / 2257 |
| bag_rank__seed_37_vs_b2 | 0.800 | [-0.084, 1.723] | 71 / 52 / 2251 |
| bag_rank__seed_37_vs_rres512 | -0.211 | [-1.094, 0.675] | 53 / 58 / 2263 |

#### no_documented_history (n=0)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 未完成 | 0/0 | 0 | 0 |
| b1 | 未完成 | 0/0 | 0 | 0 |
| b2 | 未完成 | 0/0 | 0 | 0 |
| rres512 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_11 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_23 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_37 | 未完成 | 0/0 | 0 | 0 |

bag_rank: three-seed mean 未完成; range 未完成. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
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
| OCR (40) | 97.500 | 97.500 | 95.000 | 97.500 | 95.000 | 97.500 | 97.500 |
| artwork (400) | 83.250 | 84.000 | 83.500 | 82.750 | 83.750 | 84.250 | 84.500 |
| celebrity (340) | 91.765 | 93.824 | 93.824 | 92.647 | 94.706 | 92.941 | 92.647 |
| code_reasoning (40) | 92.500 | 80.000 | 87.500 | 95.000 | 90.000 | 92.500 | 95.000 |
| color (60) | 95.000 | 95.000 | 93.333 | 100.000 | 85.000 | 86.667 | 88.333 |
| commonsense_reasoning (140) | 84.286 | 87.857 | 87.857 | 85.714 | 85.000 | 85.714 | 84.286 |
| count (60) | 93.333 | 86.667 | 86.667 | 91.667 | 83.333 | 75.000 | 83.333 |
| existence (60) | 100.000 | 98.333 | 96.667 | 100.000 | 98.333 | 98.333 | 98.333 |
| landmark (400) | 94.000 | 91.250 | 90.750 | 92.500 | 93.000 | 94.000 | 93.500 |
| numerical_calculation (40) | 87.500 | 75.000 | 80.000 | 87.500 | 85.000 | 82.500 | 87.500 |
| position (60) | 85.000 | 85.000 | 80.000 | 85.000 | 85.000 | 81.667 | 86.667 |
| posters (294) | 93.537 | 91.837 | 90.816 | 91.837 | 92.177 | 92.857 | 92.857 |
| scene (400) | 85.750 | 85.250 | 84.500 | 85.500 | 84.750 | 84.750 | 84.250 |
| text_translation (40) | 90.000 | 87.500 | 85.000 | 87.500 | 87.500 | 87.500 | 87.500 |

## InfoVQA_VAL: ANLS

2801 questions / 500 source parents / 500 independent pixel/parent clusters. History exposure: {"unknown": 2801}.

source_pixel_sha256 first, then pixel_sha256 for images; parent_id fallback for no-image controls. Source parent identity and native benchmark group identities remain unchanged.

No configured historical full-prediction identity file; false flags do not prove no exposure.

Overlap audit: {"training_overlap_count": 0, "near_duplicate_suspected_count": 0}.

### Primary native benchmark metric

| Arm | Native score | Auxiliary row score (%) | Observed/expected | Unparsed |
|---|---:|---:|---:|---:|
| full | 66.267 | 66.267 | 2801/2801 | 0 |
| b1 | 46.827 | 46.827 | 2801/2801 | 0 |
| b2 | 44.493 | 44.493 | 2801/2801 | 0 |
| rres512 | 45.653 | 45.653 | 2801/2801 | 0 |
| bag_rank__seed_11 | 48.737 | 48.737 | 2801/2801 | 0 |
| bag_rank__seed_23 | 48.192 | 48.192 | 2801/2801 | 0 |
| bag_rank__seed_37 | 48.905 | 48.905 | 2801/2801 | 0 |

bag_rank: native three-seed mean 48.611; range 48.192–48.905. The range is not a confidence interval.

### Auxiliary per-row diagnostics

Auxiliary per-row correctness/ANLS differences only. These intervals are NOT confidence intervals for circular accuracy, MME points, POPE F1 or Hallusion grouped accuracy.

#### all (n=2801)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 66.267 | 2801/2801 | 2 | 0 |
| b1 | 46.827 | 2801/2801 | 3 | 0 |
| b2 | 44.493 | 2801/2801 | 2 | 0 |
| rres512 | 45.653 | 2801/2801 | 2 | 0 |
| bag_rank__seed_11 | 48.737 | 2801/2801 | 1 | 0 |
| bag_rank__seed_23 | 48.192 | 2801/2801 | 3 | 0 |
| bag_rank__seed_37 | 48.905 | 2801/2801 | 2 | 0 |

bag_rank: three-seed mean 48.611; range 48.192–48.905. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 1.910 | [0.134, 3.667] | 554 / 483 / 1764 |
| bag_rank__seed_11_vs_b2 | 4.244 | [2.190, 6.319] | 623 / 469 / 1709 |
| bag_rank__seed_11_vs_rres512 | 3.083 | [1.110, 5.073] | 551 / 439 / 1811 |
| bag_rank__seed_23_vs_b1 | 1.365 | [-0.356, 3.116] | 526 / 498 / 1777 |
| bag_rank__seed_23_vs_b2 | 3.699 | [1.842, 5.565] | 588 / 468 / 1745 |
| bag_rank__seed_23_vs_rres512 | 2.538 | [0.526, 4.578] | 529 / 450 / 1822 |
| bag_rank__seed_37_vs_b1 | 2.078 | [0.139, 3.985] | 550 / 476 / 1775 |
| bag_rank__seed_37_vs_b2 | 4.412 | [2.301, 6.496] | 621 / 467 / 1713 |
| bag_rank__seed_37_vs_rres512 | 3.251 | [1.267, 5.219] | 547 / 417 / 1837 |

#### image_only (n=2801)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 66.267 | 2801/2801 | 2 | 0 |
| b1 | 46.827 | 2801/2801 | 3 | 0 |
| b2 | 44.493 | 2801/2801 | 2 | 0 |
| rres512 | 45.653 | 2801/2801 | 2 | 0 |
| bag_rank__seed_11 | 48.737 | 2801/2801 | 1 | 0 |
| bag_rank__seed_23 | 48.192 | 2801/2801 | 3 | 0 |
| bag_rank__seed_37 | 48.905 | 2801/2801 | 2 | 0 |

bag_rank: three-seed mean 48.611; range 48.192–48.905. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 1.910 | [0.134, 3.667] | 554 / 483 / 1764 |
| bag_rank__seed_11_vs_b2 | 4.244 | [2.190, 6.319] | 623 / 469 / 1709 |
| bag_rank__seed_11_vs_rres512 | 3.083 | [1.110, 5.073] | 551 / 439 / 1811 |
| bag_rank__seed_23_vs_b1 | 1.365 | [-0.356, 3.116] | 526 / 498 / 1777 |
| bag_rank__seed_23_vs_b2 | 3.699 | [1.842, 5.565] | 588 / 468 / 1745 |
| bag_rank__seed_23_vs_rres512 | 2.538 | [0.526, 4.578] | 529 / 450 / 1822 |
| bag_rank__seed_37_vs_b1 | 2.078 | [0.139, 3.985] | 550 / 476 / 1775 |
| bag_rank__seed_37_vs_b2 | 4.412 | [2.301, 6.496] | 621 / 467 / 1713 |
| bag_rank__seed_37_vs_rres512 | 3.251 | [1.267, 5.219] | 547 / 417 / 1837 |

#### decontaminated (n=2801)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 66.267 | 2801/2801 | 2 | 0 |
| b1 | 46.827 | 2801/2801 | 3 | 0 |
| b2 | 44.493 | 2801/2801 | 2 | 0 |
| rres512 | 45.653 | 2801/2801 | 2 | 0 |
| bag_rank__seed_11 | 48.737 | 2801/2801 | 1 | 0 |
| bag_rank__seed_23 | 48.192 | 2801/2801 | 3 | 0 |
| bag_rank__seed_37 | 48.905 | 2801/2801 | 2 | 0 |

bag_rank: three-seed mean 48.611; range 48.192–48.905. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 1.910 | [0.134, 3.667] | 554 / 483 / 1764 |
| bag_rank__seed_11_vs_b2 | 4.244 | [2.190, 6.319] | 623 / 469 / 1709 |
| bag_rank__seed_11_vs_rres512 | 3.083 | [1.110, 5.073] | 551 / 439 / 1811 |
| bag_rank__seed_23_vs_b1 | 1.365 | [-0.356, 3.116] | 526 / 498 / 1777 |
| bag_rank__seed_23_vs_b2 | 3.699 | [1.842, 5.565] | 588 / 468 / 1745 |
| bag_rank__seed_23_vs_rres512 | 2.538 | [0.526, 4.578] | 529 / 450 / 1822 |
| bag_rank__seed_37_vs_b1 | 2.078 | [0.139, 3.985] | 550 / 476 / 1775 |
| bag_rank__seed_37_vs_b2 | 4.412 | [2.301, 6.496] | 621 / 467 / 1713 |
| bag_rank__seed_37_vs_rres512 | 3.251 | [1.267, 5.219] | 547 / 417 / 1837 |

#### conservative (n=2801)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 66.267 | 2801/2801 | 2 | 0 |
| b1 | 46.827 | 2801/2801 | 3 | 0 |
| b2 | 44.493 | 2801/2801 | 2 | 0 |
| rres512 | 45.653 | 2801/2801 | 2 | 0 |
| bag_rank__seed_11 | 48.737 | 2801/2801 | 1 | 0 |
| bag_rank__seed_23 | 48.192 | 2801/2801 | 3 | 0 |
| bag_rank__seed_37 | 48.905 | 2801/2801 | 2 | 0 |

bag_rank: three-seed mean 48.611; range 48.192–48.905. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 1.910 | [0.134, 3.667] | 554 / 483 / 1764 |
| bag_rank__seed_11_vs_b2 | 4.244 | [2.190, 6.319] | 623 / 469 / 1709 |
| bag_rank__seed_11_vs_rres512 | 3.083 | [1.110, 5.073] | 551 / 439 / 1811 |
| bag_rank__seed_23_vs_b1 | 1.365 | [-0.356, 3.116] | 526 / 498 / 1777 |
| bag_rank__seed_23_vs_b2 | 3.699 | [1.842, 5.565] | 588 / 468 / 1745 |
| bag_rank__seed_23_vs_rres512 | 2.538 | [0.526, 4.578] | 529 / 450 / 1822 |
| bag_rank__seed_37_vs_b1 | 2.078 | [0.139, 3.985] | 550 / 476 / 1775 |
| bag_rank__seed_37_vs_b2 | 4.412 | [2.301, 6.496] | 621 / 467 / 1713 |
| bag_rank__seed_37_vs_rres512 | 3.251 | [1.267, 5.219] | 547 / 417 / 1837 |

#### no_documented_history (n=0)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 未完成 | 0/0 | 0 | 0 |
| b1 | 未完成 | 0/0 | 0 | 0 |
| b2 | 未完成 | 0/0 | 0 | 0 |
| rres512 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_11 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_23 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_37 | 未完成 | 0/0 | 0 | 0 |

bag_rank: three-seed mean 未完成; range 未完成. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
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

## POPE: POPE F1

5127 questions / 500 source parents / 500 independent pixel/parent clusters. History exposure: {"unknown": 5127}.

source_pixel_sha256 first, then pixel_sha256 for images; parent_id fallback for no-image controls. Source parent identity and native benchmark group identities remain unchanged.

No configured historical full-prediction identity file; false flags do not prove no exposure.

Overlap audit: {"training_overlap_count": 0, "near_duplicate_suspected_count": 0}.

### Primary native benchmark metric

| Arm | Native score | Auxiliary row score (%) | Observed/expected | Unparsed |
|---|---:|---:|---:|---:|
| full | 89.607 | 92.101 | 5127/5127 | 0 |
| b1 | 89.051 | 91.476 | 5127/5127 | 0 |
| b2 | 88.780 | 91.320 | 5127/5127 | 0 |
| rres512 | 88.546 | 91.476 | 5127/5127 | 0 |
| bag_rank__seed_11 | 86.631 | 91.633 | 5127/5127 | 0 |
| bag_rank__seed_23 | 84.533 | 90.677 | 5127/5127 | 0 |
| bag_rank__seed_37 | 86.302 | 91.633 | 5127/5127 | 0 |

bag_rank: native three-seed mean 85.822; range 84.533–86.631. The range is not a confidence interval.

| Arm | Overall | acc | precision | recall |
|---|---:|---:|---:|---:|
| full | 89.607 | 90.033 | 93.609 | 85.933 |
| b1 | 89.051 | 89.500 | 93.028 | 85.400 |
| b2 | 88.780 | 89.267 | 92.993 | 84.933 |
| rres512 | 88.546 | 89.100 | 93.284 | 84.267 |
| bag_rank__seed_11 | 86.631 | 87.778 | 95.601 | 79.200 |
| bag_rank__seed_23 | 84.533 | 86.156 | 95.754 | 75.667 |
| bag_rank__seed_37 | 86.302 | 87.578 | 96.177 | 78.267 |

Detailed native category/group metrics are preserved in report.json.

### Auxiliary per-row diagnostics

Auxiliary per-row correctness/ANLS differences only. These intervals are NOT confidence intervals for circular accuracy, MME points, POPE F1 or Hallusion grouped accuracy.

#### all (n=5127)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 92.101 | 5127/5127 | 0 | 0 |
| b1 | 91.476 | 5127/5127 | 0 | 0 |
| b2 | 91.320 | 5127/5127 | 0 | 0 |
| rres512 | 91.476 | 5127/5127 | 0 | 0 |
| bag_rank__seed_11 | 91.633 | 5127/5127 | 0 | 0 |
| bag_rank__seed_23 | 90.677 | 5127/5127 | 0 | 0 |
| bag_rank__seed_37 | 91.633 | 5127/5127 | 0 | 0 |

bag_rank: three-seed mean 91.314; range 90.677–91.633. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 0.156 | [-0.430, 0.739] | 142 / 134 / 4851 |
| bag_rank__seed_11_vs_b2 | 0.312 | [-0.253, 0.877] | 138 / 122 / 4867 |
| bag_rank__seed_11_vs_rres512 | 0.156 | [-0.449, 0.762] | 146 / 138 / 4843 |
| bag_rank__seed_23_vs_b1 | -0.800 | [-1.409, -0.176] | 136 / 177 / 4814 |
| bag_rank__seed_23_vs_b2 | -0.644 | [-1.264, -0.039] | 142 / 175 / 4810 |
| bag_rank__seed_23_vs_rres512 | -0.800 | [-1.457, -0.137] | 144 / 185 / 4798 |
| bag_rank__seed_37_vs_b1 | 0.156 | [-0.449, 0.744] | 142 / 134 / 4851 |
| bag_rank__seed_37_vs_b2 | 0.312 | [-0.292, 0.899] | 145 / 129 / 4853 |
| bag_rank__seed_37_vs_rres512 | 0.156 | [-0.468, 0.780] | 148 / 140 / 4839 |

#### image_only (n=5127)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 92.101 | 5127/5127 | 0 | 0 |
| b1 | 91.476 | 5127/5127 | 0 | 0 |
| b2 | 91.320 | 5127/5127 | 0 | 0 |
| rres512 | 91.476 | 5127/5127 | 0 | 0 |
| bag_rank__seed_11 | 91.633 | 5127/5127 | 0 | 0 |
| bag_rank__seed_23 | 90.677 | 5127/5127 | 0 | 0 |
| bag_rank__seed_37 | 91.633 | 5127/5127 | 0 | 0 |

bag_rank: three-seed mean 91.314; range 90.677–91.633. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 0.156 | [-0.430, 0.739] | 142 / 134 / 4851 |
| bag_rank__seed_11_vs_b2 | 0.312 | [-0.253, 0.877] | 138 / 122 / 4867 |
| bag_rank__seed_11_vs_rres512 | 0.156 | [-0.449, 0.762] | 146 / 138 / 4843 |
| bag_rank__seed_23_vs_b1 | -0.800 | [-1.409, -0.176] | 136 / 177 / 4814 |
| bag_rank__seed_23_vs_b2 | -0.644 | [-1.264, -0.039] | 142 / 175 / 4810 |
| bag_rank__seed_23_vs_rres512 | -0.800 | [-1.457, -0.137] | 144 / 185 / 4798 |
| bag_rank__seed_37_vs_b1 | 0.156 | [-0.449, 0.744] | 142 / 134 / 4851 |
| bag_rank__seed_37_vs_b2 | 0.312 | [-0.292, 0.899] | 145 / 129 / 4853 |
| bag_rank__seed_37_vs_rres512 | 0.156 | [-0.468, 0.780] | 148 / 140 / 4839 |

#### decontaminated (n=5127)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 92.101 | 5127/5127 | 0 | 0 |
| b1 | 91.476 | 5127/5127 | 0 | 0 |
| b2 | 91.320 | 5127/5127 | 0 | 0 |
| rres512 | 91.476 | 5127/5127 | 0 | 0 |
| bag_rank__seed_11 | 91.633 | 5127/5127 | 0 | 0 |
| bag_rank__seed_23 | 90.677 | 5127/5127 | 0 | 0 |
| bag_rank__seed_37 | 91.633 | 5127/5127 | 0 | 0 |

bag_rank: three-seed mean 91.314; range 90.677–91.633. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 0.156 | [-0.430, 0.739] | 142 / 134 / 4851 |
| bag_rank__seed_11_vs_b2 | 0.312 | [-0.253, 0.877] | 138 / 122 / 4867 |
| bag_rank__seed_11_vs_rres512 | 0.156 | [-0.449, 0.762] | 146 / 138 / 4843 |
| bag_rank__seed_23_vs_b1 | -0.800 | [-1.409, -0.176] | 136 / 177 / 4814 |
| bag_rank__seed_23_vs_b2 | -0.644 | [-1.264, -0.039] | 142 / 175 / 4810 |
| bag_rank__seed_23_vs_rres512 | -0.800 | [-1.457, -0.137] | 144 / 185 / 4798 |
| bag_rank__seed_37_vs_b1 | 0.156 | [-0.449, 0.744] | 142 / 134 / 4851 |
| bag_rank__seed_37_vs_b2 | 0.312 | [-0.292, 0.899] | 145 / 129 / 4853 |
| bag_rank__seed_37_vs_rres512 | 0.156 | [-0.468, 0.780] | 148 / 140 / 4839 |

#### conservative (n=5127)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 92.101 | 5127/5127 | 0 | 0 |
| b1 | 91.476 | 5127/5127 | 0 | 0 |
| b2 | 91.320 | 5127/5127 | 0 | 0 |
| rres512 | 91.476 | 5127/5127 | 0 | 0 |
| bag_rank__seed_11 | 91.633 | 5127/5127 | 0 | 0 |
| bag_rank__seed_23 | 90.677 | 5127/5127 | 0 | 0 |
| bag_rank__seed_37 | 91.633 | 5127/5127 | 0 | 0 |

bag_rank: three-seed mean 91.314; range 90.677–91.633. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
|---|---:|---:|---:|
| bag_rank__seed_11_vs_b1 | 0.156 | [-0.430, 0.739] | 142 / 134 / 4851 |
| bag_rank__seed_11_vs_b2 | 0.312 | [-0.253, 0.877] | 138 / 122 / 4867 |
| bag_rank__seed_11_vs_rres512 | 0.156 | [-0.449, 0.762] | 146 / 138 / 4843 |
| bag_rank__seed_23_vs_b1 | -0.800 | [-1.409, -0.176] | 136 / 177 / 4814 |
| bag_rank__seed_23_vs_b2 | -0.644 | [-1.264, -0.039] | 142 / 175 / 4810 |
| bag_rank__seed_23_vs_rres512 | -0.800 | [-1.457, -0.137] | 144 / 185 / 4798 |
| bag_rank__seed_37_vs_b1 | 0.156 | [-0.449, 0.744] | 142 / 134 / 4851 |
| bag_rank__seed_37_vs_b2 | 0.312 | [-0.292, 0.899] | 145 / 129 / 4853 |
| bag_rank__seed_37_vs_rres512 | 0.156 | [-0.468, 0.780] | 148 / 140 / 4839 |

#### no_documented_history (n=0)

| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |
|---|---:|---:|---:|---:|
| full | 未完成 | 0/0 | 0 | 0 |
| b1 | 未完成 | 0/0 | 0 | 0 |
| b2 | 未完成 | 0/0 | 0 | 0 |
| rres512 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_11 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_23 | 未完成 | 0/0 | 0 | 0 |
| bag_rank__seed_37 | 未完成 | 0/0 | 0 | 0 |

bag_rank: three-seed mean 未完成; range 未完成. Every seed is reported; this range is not a CI.

| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |
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
| adversarial (690) | 88.841 | 86.812 | 86.812 | 87.536 | 92.754 | 92.464 | 93.333 |
| adversarial,popular (753) | 91.766 | 92.032 | 92.032 | 92.032 | 94.821 | 95.883 | 96.016 |
| adversarial,random (22) | 100.000 | 95.455 | 100.000 | 100.000 | 100.000 | 95.455 | 100.000 |
| adversarial,random,popular (1535) | 86.124 | 85.603 | 85.147 | 84.430 | 79.544 | 76.091 | 78.632 |
| popular (684) | 94.737 | 94.152 | 94.152 | 95.906 | 98.246 | 97.515 | 98.392 |
| random (1415) | 99.081 | 98.657 | 98.516 | 98.728 | 99.293 | 99.435 | 99.223 |
| random,popular (28) | 85.714 | 82.143 | 82.143 | 78.571 | 85.714 | 92.857 | 89.286 |

## Latency

fresh image preparation+vision+selection+native generation; no image/KV cache; model load excluded

Median over image-cluster medians; each item median includes all repeated paired blocks. Clusters prefer source_pixel_sha256, then pixel_sha256; image-free fallback is source parent. Duplicate pixels across different source categories/paths remain one cluster. Confidence intervals resample clusters, not individual repeats. Source parent identity and official MME question-pair groups are unchanged.

Fixed benchmark image panel and single-device latency; not serving throughput. Cap 1 is TTFT, cap 16 is fixed-length total generation time.

### AI2D_TEST: 8 independent pixel/parent clusters (8 source parents)

| Arm | Cap | Median ms | Observed/expected |
|---|---:|---:|---:|
| full | 1 | 467.135 | 24/24 |
| full | 16 | 1095.434 | 24/24 |
| b1 | 1 | 394.737 | 24/24 |
| b1 | 16 | 1014.240 | 24/24 |
| b2 | 1 | 331.523 | 24/24 |
| b2 | 16 | 949.870 | 24/24 |
| rres512 | 1 | 153.754 | 24/24 |
| rres512 | 16 | 778.308 | 24/24 |
| bag_rank__seed_11 | 1 | 318.541 | 24/24 |
| bag_rank__seed_11 | 16 | 935.749 | 24/24 |
| bag_rank__seed_23 | 1 | 315.720 | 24/24 |
| bag_rank__seed_23 | 16 | 935.137 | 24/24 |
| bag_rank__seed_37 | 1 | 315.022 | 24/24 |
| bag_rank__seed_37 | 16 | 939.654 | 24/24 |

| Paired candidate − reference | Median Δ ms [95% CI] | Median speedup [95% CI] |
|---|---:|---:|
| bag_rank__seed_11_vs_b1__cap_1 | -78.259 [-79.355, -70.748] | 1.248 [1.210, 1.256] |
| bag_rank__seed_11_vs_b1__cap_16 | -75.069 [-80.487, -74.094] | 1.081 [1.079, 1.086] |
| bag_rank__seed_11_vs_b2__cap_1 | -10.215 [-12.334, -3.008] | 1.032 [1.010, 1.038] |
| bag_rank__seed_11_vs_b2__cap_16 | -11.417 [-17.215, -8.188] | 1.012 [1.009, 1.018] |
| bag_rank__seed_11_vs_rres512__cap_1 | 162.773 [159.227, 170.165] | 0.486 [0.480, 0.497] |
| bag_rank__seed_11_vs_rres512__cap_16 | 157.727 [150.847, 165.570] | 0.831 [0.827, 0.839] |
| bag_rank__seed_23_vs_b1__cap_1 | -79.324 [-81.263, -76.414] | 1.256 [1.237, 1.260] |
| bag_rank__seed_23_vs_b1__cap_16 | -76.102 [-80.211, -73.202] | 1.082 [1.078, 1.086] |
| bag_rank__seed_23_vs_b2__cap_1 | -10.890 [-12.259, -9.525] | 1.035 [1.030, 1.038] |
| bag_rank__seed_23_vs_b2__cap_16 | -10.814 [-14.660, -9.763] | 1.011 [1.010, 1.016] |
| bag_rank__seed_23_vs_rres512__cap_1 | 159.312 [153.053, 167.021] | 0.490 [0.482, 0.497] |
| bag_rank__seed_23_vs_rres512__cap_16 | 158.715 [148.498, 160.476] | 0.831 [0.828, 0.839] |
| bag_rank__seed_37_vs_b1__cap_1 | -79.596 [-81.567, -77.699] | 1.255 [1.242, 1.265] |
| bag_rank__seed_37_vs_b1__cap_16 | -73.493 [-79.987, -70.730] | 1.078 [1.076, 1.085] |
| bag_rank__seed_37_vs_b2__cap_1 | -11.560 [-12.605, -9.496] | 1.037 [1.030, 1.039] |
| bag_rank__seed_37_vs_b2__cap_16 | -10.012 [-16.826, -4.249] | 1.011 [1.005, 1.018] |
| bag_rank__seed_37_vs_rres512__cap_1 | 158.207 [152.100, 165.736] | 0.490 [0.484, 0.497] |
| bag_rank__seed_37_vs_rres512__cap_16 | 158.943 [150.912, 168.065] | 0.831 [0.822, 0.838] |

### MMBench_DEV_EN_V11: 8 independent pixel/parent clusters (8 source parents)

| Arm | Cap | Median ms | Observed/expected |
|---|---:|---:|---:|
| full | 1 | 464.423 | 24/24 |
| full | 16 | 1089.297 | 24/24 |
| b1 | 1 | 387.478 | 24/24 |
| b1 | 16 | 1013.854 | 24/24 |
| b2 | 1 | 321.892 | 24/24 |
| b2 | 16 | 941.948 | 24/24 |
| rres512 | 1 | 152.194 | 24/24 |
| rres512 | 16 | 775.523 | 24/24 |
| bag_rank__seed_11 | 1 | 311.282 | 24/24 |
| bag_rank__seed_11 | 16 | 932.147 | 24/24 |
| bag_rank__seed_23 | 1 | 310.687 | 24/24 |
| bag_rank__seed_23 | 16 | 932.929 | 24/24 |
| bag_rank__seed_37 | 1 | 310.041 | 24/24 |
| bag_rank__seed_37 | 16 | 935.703 | 24/24 |

| Paired candidate − reference | Median Δ ms [95% CI] | Median speedup [95% CI] |
|---|---:|---:|
| bag_rank__seed_11_vs_b1__cap_1 | -79.537 [-82.404, -77.407] | 1.258 [1.235, 1.284] |
| bag_rank__seed_11_vs_b1__cap_16 | -76.703 [-82.395, -73.315] | 1.083 [1.078, 1.088] |
| bag_rank__seed_11_vs_b2__cap_1 | -10.611 [-12.110, -9.111] | 1.034 [1.030, 1.037] |
| bag_rank__seed_11_vs_b2__cap_16 | -9.341 [-10.395, -2.771] | 1.010 [1.003, 1.011] |
| bag_rank__seed_11_vs_rres512__cap_1 | 157.543 [135.753, 161.695] | 0.499 [0.487, 0.516] |
| bag_rank__seed_11_vs_rres512__cap_16 | 153.645 [135.030, 160.777] | 0.835 [0.828, 0.851] |
| bag_rank__seed_23_vs_b1__cap_1 | -80.140 [-81.942, -75.200] | 1.268 [1.224, 1.286] |
| bag_rank__seed_23_vs_b1__cap_16 | -78.153 [-82.244, -71.610] | 1.084 [1.076, 1.088] |
| bag_rank__seed_23_vs_b2__cap_1 | -11.225 [-16.234, -6.177] | 1.035 [1.020, 1.053] |
| bag_rank__seed_23_vs_b2__cap_16 | -10.382 [-11.472, -4.692] | 1.011 [1.005, 1.013] |
| bag_rank__seed_23_vs_rres512__cap_1 | 155.902 [135.920, 166.465] | 0.499 [0.484, 0.517] |
| bag_rank__seed_23_vs_rres512__cap_16 | 156.598 [133.037, 169.478] | 0.832 [0.821, 0.852] |
| bag_rank__seed_37_vs_b1__cap_1 | -78.028 [-84.619, -76.265] | 1.261 [1.235, 1.289] |
| bag_rank__seed_37_vs_b1__cap_16 | -76.403 [-79.792, -74.756] | 1.083 [1.077, 1.085] |
| bag_rank__seed_37_vs_b2__cap_1 | -10.192 [-17.088, -8.182] | 1.031 [1.026, 1.055] |
| bag_rank__seed_37_vs_b2__cap_16 | -8.730 [-11.180, -3.534] | 1.010 [1.004, 1.012] |
| bag_rank__seed_37_vs_rres512__cap_1 | 154.822 [135.145, 163.245] | 0.502 [0.484, 0.518] |
| bag_rank__seed_37_vs_rres512__cap_16 | 159.390 [132.761, 165.735] | 0.830 [0.824, 0.853] |

### MMBench_DEV_CN_V11: 8 independent pixel/parent clusters (8 source parents)

| Arm | Cap | Median ms | Observed/expected |
|---|---:|---:|---:|
| full | 1 | 461.418 | 24/24 |
| full | 16 | 1090.269 | 24/24 |
| b1 | 1 | 388.347 | 24/24 |
| b1 | 16 | 1006.086 | 24/24 |
| b2 | 1 | 322.511 | 24/24 |
| b2 | 16 | 943.923 | 24/24 |
| rres512 | 1 | 153.359 | 24/24 |
| rres512 | 16 | 775.004 | 24/24 |
| bag_rank__seed_11 | 1 | 308.890 | 24/24 |
| bag_rank__seed_11 | 16 | 933.464 | 24/24 |
| bag_rank__seed_23 | 1 | 312.003 | 24/24 |
| bag_rank__seed_23 | 16 | 934.497 | 24/24 |
| bag_rank__seed_37 | 1 | 312.962 | 24/24 |
| bag_rank__seed_37 | 16 | 931.355 | 24/24 |

| Paired candidate − reference | Median Δ ms [95% CI] | Median speedup [95% CI] |
|---|---:|---:|
| bag_rank__seed_11_vs_b1__cap_1 | -79.877 [-83.348, -77.087] | 1.266 [1.240, 1.285] |
| bag_rank__seed_11_vs_b1__cap_16 | -75.995 [-81.831, -71.088] | 1.084 [1.077, 1.086] |
| bag_rank__seed_11_vs_b2__cap_1 | -11.131 [-13.224, -8.456] | 1.036 [1.029, 1.042] |
| bag_rank__seed_11_vs_b2__cap_16 | -9.651 [-15.382, -7.356] | 1.010 [1.008, 1.017] |
| bag_rank__seed_11_vs_rres512__cap_1 | 153.960 [136.033, 160.873] | 0.502 [0.487, 0.515] |
| bag_rank__seed_11_vs_rres512__cap_16 | 156.567 [137.142, 165.625] | 0.832 [0.825, 0.848] |
| bag_rank__seed_23_vs_b1__cap_1 | -78.505 [-80.920, -73.760] | 1.252 [1.234, 1.289] |
| bag_rank__seed_23_vs_b1__cap_16 | -77.538 [-80.873, -75.523] | 1.085 [1.079, 1.087] |
| bag_rank__seed_23_vs_b2__cap_1 | -10.312 [-10.751, -8.780] | 1.034 [1.031, 1.037] |
| bag_rank__seed_23_vs_b2__cap_16 | -9.359 [-15.824, -5.929] | 1.010 [1.006, 1.017] |
| bag_rank__seed_23_vs_rres512__cap_1 | 157.133 [135.834, 168.918] | 0.496 [0.484, 0.516] |
| bag_rank__seed_23_vs_rres512__cap_16 | 159.493 [136.497, 167.295] | 0.829 [0.823, 0.848] |
| bag_rank__seed_37_vs_b1__cap_1 | -78.080 [-79.862, -72.745] | 1.248 [1.224, 1.285] |
| bag_rank__seed_37_vs_b1__cap_16 | -78.518 [-82.770, -73.343] | 1.084 [1.079, 1.091] |
| bag_rank__seed_37_vs_b2__cap_1 | -8.598 [-10.107, -6.633] | 1.028 [1.023, 1.034] |
| bag_rank__seed_37_vs_b2__cap_16 | -10.943 [-17.609, -7.202] | 1.012 [1.008, 1.019] |
| bag_rank__seed_37_vs_rres512__cap_1 | 158.032 [135.195, 167.628] | 0.495 [0.480, 0.517] |
| bag_rank__seed_37_vs_rres512__cap_16 | 156.351 [140.334, 164.686] | 0.832 [0.826, 0.845] |

### MMStar: 8 independent pixel/parent clusters (8 source parents)

| Arm | Cap | Median ms | Observed/expected |
|---|---:|---:|---:|
| full | 1 | 462.107 | 24/24 |
| full | 16 | 1092.191 | 24/24 |
| b1 | 1 | 390.204 | 24/24 |
| b1 | 16 | 1011.476 | 24/24 |
| b2 | 1 | 319.811 | 24/24 |
| b2 | 16 | 944.848 | 24/24 |
| rres512 | 1 | 154.820 | 24/24 |
| rres512 | 16 | 776.021 | 24/24 |
| bag_rank__seed_11 | 1 | 309.465 | 24/24 |
| bag_rank__seed_11 | 16 | 931.403 | 24/24 |
| bag_rank__seed_23 | 1 | 312.903 | 24/24 |
| bag_rank__seed_23 | 16 | 934.617 | 24/24 |
| bag_rank__seed_37 | 1 | 311.796 | 24/24 |
| bag_rank__seed_37 | 16 | 931.447 | 24/24 |

| Paired candidate − reference | Median Δ ms [95% CI] | Median speedup [95% CI] |
|---|---:|---:|
| bag_rank__seed_11_vs_b1__cap_1 | -79.638 [-87.151, -76.590] | 1.259 [1.243, 1.294] |
| bag_rank__seed_11_vs_b1__cap_16 | -78.273 [-81.698, -74.139] | 1.085 [1.079, 1.088] |
| bag_rank__seed_11_vs_b2__cap_1 | -9.748 [-17.566, -5.405] | 1.032 [1.016, 1.057] |
| bag_rank__seed_11_vs_b2__cap_16 | -10.852 [-14.898, -9.096] | 1.012 [1.010, 1.016] |
| bag_rank__seed_11_vs_rres512__cap_1 | 157.341 [143.878, 159.386] | 0.498 [0.490, 0.524] |
| bag_rank__seed_11_vs_rres512__cap_16 | 154.132 [151.258, 157.942] | 0.834 [0.831, 0.839] |
| bag_rank__seed_23_vs_b1__cap_1 | -79.150 [-84.787, -76.105] | 1.255 [1.243, 1.298] |
| bag_rank__seed_23_vs_b1__cap_16 | -75.437 [-82.806, -72.544] | 1.081 [1.077, 1.089] |
| bag_rank__seed_23_vs_b2__cap_1 | -11.373 [-13.095, -6.054] | 1.037 [1.019, 1.043] |
| bag_rank__seed_23_vs_b2__cap_16 | -13.406 [-15.693, -6.050] | 1.014 [1.006, 1.017] |
| bag_rank__seed_23_vs_rres512__cap_1 | 154.892 [148.820, 161.850] | 0.500 [0.488, 0.528] |
| bag_rank__seed_23_vs_rres512__cap_16 | 155.678 [148.855, 165.055] | 0.833 [0.825, 0.841] |
| bag_rank__seed_37_vs_b1__cap_1 | -81.921 [-85.514, -78.019] | 1.267 [1.250, 1.296] |
| bag_rank__seed_37_vs_b1__cap_16 | -79.514 [-86.646, -73.638] | 1.085 [1.079, 1.095] |
| bag_rank__seed_37_vs_b2__cap_1 | -10.621 [-16.325, -4.113] | 1.035 [1.013, 1.055] |
| bag_rank__seed_37_vs_b2__cap_16 | -13.518 [-16.350, -8.440] | 1.014 [1.009, 1.018] |
| bag_rank__seed_37_vs_rres512__cap_1 | 155.028 [146.681, 157.115] | 0.500 [0.495, 0.527] |
| bag_rank__seed_37_vs_rres512__cap_16 | 152.478 [142.937, 159.341] | 0.836 [0.829, 0.844] |

### RealWorldQA: 8 independent pixel/parent clusters (8 source parents)

| Arm | Cap | Median ms | Observed/expected |
|---|---:|---:|---:|
| full | 1 | 487.905 | 24/24 |
| full | 16 | 1113.311 | 24/24 |
| b1 | 1 | 413.521 | 24/24 |
| b1 | 16 | 1034.794 | 24/24 |
| b2 | 1 | 343.254 | 24/24 |
| b2 | 16 | 965.448 | 24/24 |
| rres512 | 1 | 174.601 | 24/24 |
| rres512 | 16 | 795.205 | 24/24 |
| bag_rank__seed_11 | 1 | 332.930 | 24/24 |
| bag_rank__seed_11 | 16 | 953.953 | 24/24 |
| bag_rank__seed_23 | 1 | 333.483 | 24/24 |
| bag_rank__seed_23 | 16 | 957.191 | 24/24 |
| bag_rank__seed_37 | 1 | 333.846 | 24/24 |
| bag_rank__seed_37 | 16 | 958.368 | 24/24 |

| Paired candidate − reference | Median Δ ms [95% CI] | Median speedup [95% CI] |
|---|---:|---:|
| bag_rank__seed_11_vs_b1__cap_1 | -80.272 [-82.317, -78.418] | 1.243 [1.233, 1.247] |
| bag_rank__seed_11_vs_b1__cap_16 | -79.868 [-83.650, -77.212] | 1.084 [1.080, 1.088] |
| bag_rank__seed_11_vs_b2__cap_1 | -14.413 [-18.207, -3.366] | 1.043 [1.010, 1.055] |
| bag_rank__seed_11_vs_b2__cap_16 | -14.072 [-15.339, -10.396] | 1.015 [1.011, 1.016] |
| bag_rank__seed_11_vs_rres512__cap_1 | 155.886 [153.493, 159.527] | 0.529 [0.524, 0.539] |
| bag_rank__seed_11_vs_rres512__cap_16 | 153.002 [151.041, 159.927] | 0.840 [0.832, 0.842] |
| bag_rank__seed_23_vs_b1__cap_1 | -78.832 [-81.769, -77.024] | 1.236 [1.227, 1.246] |
| bag_rank__seed_23_vs_b1__cap_16 | -78.578 [-81.430, -69.488] | 1.082 [1.073, 1.085] |
| bag_rank__seed_23_vs_b2__cap_1 | -10.929 [-12.913, -7.949] | 1.033 [1.024, 1.038] |
| bag_rank__seed_23_vs_b2__cap_16 | -9.657 [-14.866, -6.073] | 1.010 [1.006, 1.016] |
| bag_rank__seed_23_vs_rres512__cap_1 | 158.816 [154.174, 163.285] | 0.526 [0.516, 0.532] |
| bag_rank__seed_23_vs_rres512__cap_16 | 157.382 [153.915, 161.568] | 0.835 [0.831, 0.839] |
| bag_rank__seed_37_vs_b1__cap_1 | -80.534 [-85.875, -74.795] | 1.241 [1.220, 1.261] |
| bag_rank__seed_37_vs_b1__cap_16 | -76.794 [-84.187, -69.776] | 1.080 [1.073, 1.089] |
| bag_rank__seed_37_vs_b2__cap_1 | -11.066 [-17.776, -5.449] | 1.033 [1.016, 1.053] |
| bag_rank__seed_37_vs_b2__cap_16 | -11.161 [-14.916, -4.685] | 1.012 [1.005, 1.016] |
| bag_rank__seed_37_vs_rres512__cap_1 | 157.461 [154.220, 163.150] | 0.525 [0.516, 0.538] |
| bag_rank__seed_37_vs_rres512__cap_16 | 157.159 [153.922, 161.723] | 0.836 [0.831, 0.839] |

### ChartQA_TEST: 8 independent pixel/parent clusters (8 source parents)

| Arm | Cap | Median ms | Observed/expected |
|---|---:|---:|---:|
| full | 1 | 464.858 | 24/24 |
| full | 16 | 1091.580 | 24/24 |
| b1 | 1 | 391.936 | 24/24 |
| b1 | 16 | 1011.151 | 24/24 |
| b2 | 1 | 322.817 | 24/24 |
| b2 | 16 | 943.854 | 24/24 |
| rres512 | 1 | 154.831 | 24/24 |
| rres512 | 16 | 775.884 | 24/24 |
| bag_rank__seed_11 | 1 | 312.987 | 24/24 |
| bag_rank__seed_11 | 16 | 934.503 | 24/24 |
| bag_rank__seed_23 | 1 | 312.826 | 24/24 |
| bag_rank__seed_23 | 16 | 933.468 | 24/24 |
| bag_rank__seed_37 | 1 | 313.719 | 24/24 |
| bag_rank__seed_37 | 16 | 933.621 | 24/24 |

| Paired candidate − reference | Median Δ ms [95% CI] | Median speedup [95% CI] |
|---|---:|---:|
| bag_rank__seed_11_vs_b1__cap_1 | -79.105 [-82.316, -78.327] | 1.254 [1.250, 1.271] |
| bag_rank__seed_11_vs_b1__cap_16 | -77.687 [-79.216, -76.629] | 1.083 [1.082, 1.085] |
| bag_rank__seed_11_vs_b2__cap_1 | -10.153 [-12.880, -9.012] | 1.032 [1.029, 1.041] |
| bag_rank__seed_11_vs_b2__cap_16 | -9.944 [-12.258, -6.948] | 1.011 [1.007, 1.013] |
| bag_rank__seed_11_vs_rres512__cap_1 | 157.137 [155.309, 159.809] | 0.494 [0.492, 0.500] |
| bag_rank__seed_11_vs_rres512__cap_16 | 158.760 [155.388, 161.983] | 0.830 [0.827, 0.832] |
| bag_rank__seed_23_vs_b1__cap_1 | -79.355 [-82.287, -77.826] | 1.253 [1.251, 1.258] |
| bag_rank__seed_23_vs_b1__cap_16 | -77.950 [-79.946, -75.929] | 1.083 [1.081, 1.087] |
| bag_rank__seed_23_vs_b2__cap_1 | -9.921 [-12.231, -6.982] | 1.032 [1.022, 1.039] |
| bag_rank__seed_23_vs_b2__cap_16 | -11.031 [-12.231, -8.416] | 1.012 [1.009, 1.013] |
| bag_rank__seed_23_vs_rres512__cap_1 | 157.575 [154.916, 159.712] | 0.495 [0.489, 0.500] |
| bag_rank__seed_23_vs_rres512__cap_16 | 157.619 [154.969, 158.297] | 0.831 [0.830, 0.834] |
| bag_rank__seed_37_vs_b1__cap_1 | -79.117 [-81.492, -78.429] | 1.253 [1.250, 1.262] |
| bag_rank__seed_37_vs_b1__cap_16 | -77.773 [-79.875, -68.193] | 1.083 [1.072, 1.086] |
| bag_rank__seed_37_vs_b2__cap_1 | -9.098 [-10.228, -7.307] | 1.029 [1.023, 1.033] |
| bag_rank__seed_37_vs_b2__cap_16 | -10.322 [-11.460, -3.822] | 1.011 [1.004, 1.012] |
| bag_rank__seed_37_vs_rres512__cap_1 | 159.104 [154.092, 161.810] | 0.492 [0.488, 0.501] |
| bag_rank__seed_37_vs_rres512__cap_16 | 157.783 [155.819, 166.477] | 0.831 [0.823, 0.833] |

### HallusionBench: 8 independent pixel/parent clusters (7 source parents)

| Arm | Cap | Median ms | Observed/expected |
|---|---:|---:|---:|
| full | 1 | 475.762 | 24/24 |
| full | 16 | 1101.167 | 24/24 |
| b1 | 1 | 401.556 | 24/24 |
| b1 | 16 | 1027.022 | 24/24 |
| b2 | 1 | 335.383 | 24/24 |
| b2 | 16 | 954.783 | 24/24 |
| rres512 | 1 | 168.207 | 24/24 |
| rres512 | 16 | 789.595 | 24/24 |
| bag_rank__seed_11 | 1 | 324.335 | 24/24 |
| bag_rank__seed_11 | 16 | 945.565 | 24/24 |
| bag_rank__seed_23 | 1 | 322.938 | 24/24 |
| bag_rank__seed_23 | 16 | 944.276 | 24/24 |
| bag_rank__seed_37 | 1 | 324.842 | 24/24 |
| bag_rank__seed_37 | 16 | 945.310 | 24/24 |

| Paired candidate − reference | Median Δ ms [95% CI] | Median speedup [95% CI] |
|---|---:|---:|
| bag_rank__seed_11_vs_b1__cap_1 | -78.203 [-78.813, -75.010] | 1.237 [1.200, 1.250] |
| bag_rank__seed_11_vs_b1__cap_16 | -83.419 [-87.212, -78.029] | 1.085 [1.083, 1.092] |
| bag_rank__seed_11_vs_b2__cap_1 | -10.665 [-11.680, -9.611] | 1.032 [1.028, 1.036] |
| bag_rank__seed_11_vs_b2__cap_16 | -11.299 [-12.072, -9.064] | 1.012 [1.009, 1.013] |
| bag_rank__seed_11_vs_rres512__cap_1 | 157.827 [155.409, 176.287] | 0.515 [0.496, 0.556] |
| bag_rank__seed_11_vs_rres512__cap_16 | 154.517 [153.262, 166.996] | 0.835 [0.824, 0.837] |
| bag_rank__seed_23_vs_b1__cap_1 | -78.647 [-79.483, -75.967] | 1.241 [1.202, 1.257] |
| bag_rank__seed_23_vs_b1__cap_16 | -85.016 [-88.165, -80.743] | 1.088 [1.085, 1.093] |
| bag_rank__seed_23_vs_b2__cap_1 | -11.497 [-14.287, -10.276] | 1.034 [1.030, 1.046] |
| bag_rank__seed_23_vs_b2__cap_16 | -11.699 [-12.099, -9.558] | 1.012 [1.010, 1.013] |
| bag_rank__seed_23_vs_rres512__cap_1 | 156.208 [154.377, 174.588] | 0.518 [0.498, 0.554] |
| bag_rank__seed_23_vs_rres512__cap_16 | 155.836 [151.334, 165.066] | 0.834 [0.828, 0.839] |
| bag_rank__seed_37_vs_b1__cap_1 | -78.846 [-79.427, -76.841] | 1.238 [1.211, 1.250] |
| bag_rank__seed_37_vs_b1__cap_16 | -80.079 [-83.561, -77.761] | 1.085 [1.081, 1.088] |
| bag_rank__seed_37_vs_b2__cap_1 | -11.359 [-12.632, -10.212] | 1.033 [1.030, 1.041] |
| bag_rank__seed_37_vs_b2__cap_16 | -9.126 [-9.501, -5.467] | 1.010 [1.005, 1.010] |
| bag_rank__seed_37_vs_rres512__cap_1 | 157.654 [154.160, 167.918] | 0.516 [0.498, 0.557] |
| bag_rank__seed_37_vs_rres512__cap_16 | 161.022 [155.370, 167.643] | 0.833 [0.826, 0.837] |

### MME: 8 independent pixel/parent clusters (8 source parents)

| Arm | Cap | Median ms | Observed/expected |
|---|---:|---:|---:|
| full | 1 | 461.735 | 24/24 |
| full | 16 | 1090.286 | 24/24 |
| b1 | 1 | 393.958 | 24/24 |
| b1 | 16 | 1009.359 | 24/24 |
| b2 | 1 | 322.266 | 24/24 |
| b2 | 16 | 943.118 | 24/24 |
| rres512 | 1 | 152.252 | 24/24 |
| rres512 | 16 | 774.208 | 24/24 |
| bag_rank__seed_11 | 1 | 310.188 | 24/24 |
| bag_rank__seed_11 | 16 | 930.024 | 24/24 |
| bag_rank__seed_23 | 1 | 313.914 | 24/24 |
| bag_rank__seed_23 | 16 | 930.689 | 24/24 |
| bag_rank__seed_37 | 1 | 312.960 | 24/24 |
| bag_rank__seed_37 | 16 | 930.756 | 24/24 |

| Paired candidate − reference | Median Δ ms [95% CI] | Median speedup [95% CI] |
|---|---:|---:|
| bag_rank__seed_11_vs_b1__cap_1 | -80.478 [-84.261, -78.324] | 1.265 [1.246, 1.276] |
| bag_rank__seed_11_vs_b1__cap_16 | -79.656 [-81.493, -78.097] | 1.086 [1.084, 1.087] |
| bag_rank__seed_11_vs_b2__cap_1 | -9.696 [-16.048, -8.545] | 1.031 [1.026, 1.052] |
| bag_rank__seed_11_vs_b2__cap_16 | -12.036 [-14.288, -9.839] | 1.013 [1.010, 1.016] |
| bag_rank__seed_11_vs_rres512__cap_1 | 158.934 [151.862, 167.853] | 0.492 [0.475, 0.499] |
| bag_rank__seed_11_vs_rres512__cap_16 | 156.431 [153.402, 160.542] | 0.833 [0.828, 0.835] |
| bag_rank__seed_23_vs_b1__cap_1 | -78.921 [-80.332, -77.752] | 1.251 [1.250, 1.259] |
| bag_rank__seed_23_vs_b1__cap_16 | -80.240 [-81.621, -78.353] | 1.087 [1.085, 1.088] |
| bag_rank__seed_23_vs_b2__cap_1 | -9.901 [-10.622, -3.795] | 1.033 [1.012, 1.034] |
| bag_rank__seed_23_vs_b2__cap_16 | -12.649 [-14.569, -10.063] | 1.014 [1.011, 1.016] |
| bag_rank__seed_23_vs_rres512__cap_1 | 160.686 [157.184, 169.733] | 0.488 [0.473, 0.498] |
| bag_rank__seed_23_vs_rres512__cap_16 | 156.154 [153.046, 160.619] | 0.832 [0.828, 0.835] |
| bag_rank__seed_37_vs_b1__cap_1 | -79.360 [-82.697, -77.251] | 1.255 [1.247, 1.271] |
| bag_rank__seed_37_vs_b1__cap_16 | -79.729 [-81.863, -78.799] | 1.087 [1.084, 1.088] |
| bag_rank__seed_37_vs_b2__cap_1 | -10.082 [-10.633, -8.530] | 1.033 [1.027, 1.036] |
| bag_rank__seed_37_vs_b2__cap_16 | -12.425 [-15.354, -10.605] | 1.013 [1.011, 1.017] |
| bag_rank__seed_37_vs_rres512__cap_1 | 158.846 [156.463, 168.596] | 0.488 [0.474, 0.501] |
| bag_rank__seed_37_vs_rres512__cap_16 | 156.367 [152.999, 160.172] | 0.832 [0.828, 0.835] |

### InfoVQA_VAL: 8 independent pixel/parent clusters (8 source parents)

| Arm | Cap | Median ms | Observed/expected |
|---|---:|---:|---:|
| full | 1 | 522.050 | 24/24 |
| full | 16 | 1152.020 | 24/24 |
| b1 | 1 | 446.440 | 24/24 |
| b1 | 16 | 1066.999 | 24/24 |
| b2 | 1 | 379.774 | 24/24 |
| b2 | 16 | 998.471 | 24/24 |
| rres512 | 1 | 189.018 | 24/24 |
| rres512 | 16 | 812.521 | 24/24 |
| bag_rank__seed_11 | 1 | 367.027 | 24/24 |
| bag_rank__seed_11 | 16 | 989.514 | 24/24 |
| bag_rank__seed_23 | 1 | 367.959 | 24/24 |
| bag_rank__seed_23 | 16 | 991.872 | 24/24 |
| bag_rank__seed_37 | 1 | 370.188 | 24/24 |
| bag_rank__seed_37 | 16 | 989.851 | 24/24 |

| Paired candidate − reference | Median Δ ms [95% CI] | Median speedup [95% CI] |
|---|---:|---:|
| bag_rank__seed_11_vs_b1__cap_1 | -79.412 [-85.649, -76.600] | 1.220 [1.193, 1.250] |
| bag_rank__seed_11_vs_b1__cap_16 | -79.676 [-83.187, -76.290] | 1.081 [1.077, 1.084] |
| bag_rank__seed_11_vs_b2__cap_1 | -13.204 [-17.818, -12.719] | 1.036 [1.035, 1.050] |
| bag_rank__seed_11_vs_b2__cap_16 | -11.371 [-14.560, -7.768] | 1.012 [1.008, 1.014] |
| bag_rank__seed_11_vs_rres512__cap_1 | 177.034 [157.781, 190.186] | 0.515 [0.495, 0.523] |
| bag_rank__seed_11_vs_rres512__cap_16 | 180.741 [151.914, 197.556] | 0.824 [0.808, 0.839] |
| bag_rank__seed_23_vs_b1__cap_1 | -77.595 [-81.696, -74.450] | 1.213 [1.191, 1.233] |
| bag_rank__seed_23_vs_b1__cap_16 | -78.410 [-80.293, -75.875] | 1.078 [1.075, 1.081] |
| bag_rank__seed_23_vs_b2__cap_1 | -11.815 [-15.762, -10.575] | 1.032 [1.028, 1.036] |
| bag_rank__seed_23_vs_b2__cap_16 | -9.147 [-10.253, -3.793] | 1.009 [1.004, 1.010] |
| bag_rank__seed_23_vs_rres512__cap_1 | 178.179 [158.906, 191.393] | 0.505 [0.489, 0.521] |
| bag_rank__seed_23_vs_rres512__cap_16 | 180.243 [160.682, 193.893] | 0.824 [0.807, 0.835] |
| bag_rank__seed_37_vs_b1__cap_1 | -79.878 [-82.402, -73.769] | 1.222 [1.196, 1.238] |
| bag_rank__seed_37_vs_b1__cap_16 | -77.586 [-82.768, -75.317] | 1.080 [1.075, 1.087] |
| bag_rank__seed_37_vs_b2__cap_1 | -12.559 [-16.126, -7.882] | 1.032 [1.023, 1.043] |
| bag_rank__seed_37_vs_b2__cap_16 | -9.968 [-11.828, -6.783] | 1.010 [1.007, 1.013] |
| bag_rank__seed_37_vs_rres512__cap_1 | 180.195 [160.003, 185.842] | 0.510 [0.489, 0.524] |
| bag_rank__seed_37_vs_rres512__cap_16 | 177.865 [153.789, 203.212] | 0.821 [0.809, 0.834] |

### POPE: 8 independent pixel/parent clusters (8 source parents)

| Arm | Cap | Median ms | Observed/expected |
|---|---:|---:|---:|
| full | 1 | 466.346 | 24/24 |
| full | 16 | 1093.719 | 24/24 |
| b1 | 1 | 393.845 | 24/24 |
| b1 | 16 | 1013.422 | 24/24 |
| b2 | 1 | 324.978 | 24/24 |
| b2 | 16 | 945.735 | 24/24 |
| rres512 | 1 | 152.580 | 24/24 |
| rres512 | 16 | 774.636 | 24/24 |
| bag_rank__seed_11 | 1 | 315.260 | 24/24 |
| bag_rank__seed_11 | 16 | 935.197 | 24/24 |
| bag_rank__seed_23 | 1 | 316.476 | 24/24 |
| bag_rank__seed_23 | 16 | 934.656 | 24/24 |
| bag_rank__seed_37 | 1 | 315.024 | 24/24 |
| bag_rank__seed_37 | 16 | 937.601 | 24/24 |

| Paired candidate − reference | Median Δ ms [95% CI] | Median speedup [95% CI] |
|---|---:|---:|
| bag_rank__seed_11_vs_b1__cap_1 | -78.318 [-81.842, -77.175] | 1.249 [1.239, 1.262] |
| bag_rank__seed_11_vs_b1__cap_16 | -79.799 [-84.779, -74.940] | 1.085 [1.080, 1.091] |
| bag_rank__seed_11_vs_b2__cap_1 | -10.360 [-13.313, -6.491] | 1.033 [1.020, 1.043] |
| bag_rank__seed_11_vs_b2__cap_16 | -14.331 [-21.282, -7.526] | 1.015 [1.008, 1.023] |
| bag_rank__seed_11_vs_rres512__cap_1 | 162.499 [160.836, 164.532] | 0.484 [0.481, 0.487] |
| bag_rank__seed_11_vs_rres512__cap_16 | 160.561 [155.411, 164.285] | 0.828 [0.825, 0.833] |
| bag_rank__seed_23_vs_b1__cap_1 | -79.124 [-80.567, -76.996] | 1.249 [1.244, 1.255] |
| bag_rank__seed_23_vs_b1__cap_16 | -80.330 [-80.782, -74.776] | 1.086 [1.080, 1.087] |
| bag_rank__seed_23_vs_b2__cap_1 | -10.068 [-11.229, -6.956] | 1.032 [1.022, 1.035] |
| bag_rank__seed_23_vs_b2__cap_16 | -11.354 [-14.379, -9.484] | 1.012 [1.010, 1.015] |
| bag_rank__seed_23_vs_rres512__cap_1 | 163.689 [160.908, 166.572] | 0.481 [0.478, 0.488] |
| bag_rank__seed_23_vs_rres512__cap_16 | 160.334 [157.189, 170.942] | 0.828 [0.819, 0.831] |
| bag_rank__seed_37_vs_b1__cap_1 | -79.737 [-83.085, -77.200] | 1.252 [1.244, 1.265] |
| bag_rank__seed_37_vs_b1__cap_16 | -74.522 [-83.083, -67.111] | 1.079 [1.071, 1.090] |
| bag_rank__seed_37_vs_b2__cap_1 | -11.568 [-13.169, -7.949] | 1.037 [1.025, 1.043] |
| bag_rank__seed_37_vs_b2__cap_16 | -10.602 [-22.769, -1.697] | 1.011 [1.002, 1.024] |
| bag_rank__seed_37_vs_rres512__cap_1 | 161.881 [157.810, 166.099] | 0.485 [0.478, 0.490] |
| bag_rank__seed_37_vs_rres512__cap_16 | 162.921 [159.135, 171.607] | 0.826 [0.819, 0.830] |

## Audit

Freeze: verified. Stored scores independently recomputed against original answers. Native token budgets, DeepStack alignment, original positions and token caps checked for all successful outputs.

Missing outputs are never silently removed from denominators. See report.json for full-denominator bounds, category comparisons, source hashes and every missing key.
