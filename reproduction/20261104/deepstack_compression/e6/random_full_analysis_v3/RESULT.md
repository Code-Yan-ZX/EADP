# Random K256 versus EADP K256: exploratory DEV

The complete 384-question × three fixed-seed random panel passed formal and independent no-GT acceptance before scoring. `score001/analysis.json` and the 1,152 individual random scores were independently recomputed by `e1_independent_review/random_full_v2_scores_review.json`; all related supervisors exited successfully. The score methodology was fixed before full generation; its E1 identity gate and task-by-seed XLSX type-handling repairs were frozen and independently reviewed after generation began but without reading that panel's predictions or labels.

| Task (128 questions/128 image groups) | EADP K256 | Random seed 20261104 | Random seed 20261105 | Random seed 20261106 | Random three-seed within-question mean | Random mean − EADP (pp; 95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| DocVQA | 67.85 | 52.61 | 56.96 | 56.72 | 55.43 | −12.42 [−19.84, −4.84] |
| TextVQA | 80.78 | 73.52 | 72.19 | 73.44 | 73.05 | −7.73 [−12.92, −2.71] |
| AI2D | 82.03 | 72.66 | 75.78 | 73.44 | 73.96 | −8.07 [−13.28, −3.12] |

Per-seed paired differences and unadjusted 95% image-group bootstrap intervals are:

| Task | 20261104 | 20261105 | 20261106 |
|---|---:|---:|---:|
| DocVQA | −15.24 [−24.79, −5.57] | −10.88 [−19.60, −2.21] | −11.12 [−19.69, −2.46] |
| TextVQA | −7.27 [−14.06, −0.94] | −8.59 [−14.61, −2.97] | −7.34 [−14.22, −0.63] |
| AI2D | −9.38 [−15.63, −3.12] | −6.25 [−12.50, −0.78] | −8.59 [−14.84, −2.34] |

Each interval resamples image groups 10,000 times, pairing the random and EADP scores. The three-seed comparison averages scores *within each question* before bootstrapping; uncertainty is conditional on those fixed seeds and does not estimate a population of seeds. This is a posthoc DEV exploration following earlier results, with no multiplicity correction; it is not CONFIRM or FINAL. All 1,152 random generations ended with EOS, with zero truncations or empty answers. Native XLSX conversion was done separately for each task and seed; author and independent scoring matched for each answer.

Same retained K does not prove equal compute. The random adapter actually executes the original greedy selector and discards its selections before injecting random indices, so its elapsed time cannot be compared with EADP or treated as the random selector's cost. Full ordered selected indices are recorded, while full visual/DeepStack/KV tensors are represented by runtime checks and fingerprints rather than complete archived arrays; transient weight replacement cannot be excluded by endpoint stat checks.
