# Fixed-budget regional allocation diagnostic

Status: **complete**; decision: **STOP_THIS_GEOMETRY**. QA 14208/14208; timing 216/216.

Finite, balanced candidate-bank oracle diagnostic. Geometry generation is answer-free; best-over-bank reporting and selected-dual factorial controls use ground-truth scores. Not a deployable selector, training gain, global single-region upper bound, proof of causal locality, or novel method. All QA including truncations stay in denominators. STOP applies only to these geometries.

New source-training panels, not untouched benchmarks or proof of absence from base pretraining. Full encoder context can carry information across selected regions. Donors are alternative spatial units from the same encoded image, not edited pixels or an independent image. Two fixed donor pairs reduce one-filler accidents without proving causal necessity or excluding broader geometric effects. No deployment claim follows from an oracle win.

HOLD if either dataset has fewer than 12 full>=0.8 and R-res<0.5 cases. GO only if both dual-minus-single point estimates>0, equal-dataset delta>=0.03 with CI lower>0, strict total>=8 and each dataset>=2; otherwise STOP_THIS_GEOMETRY.

full>=0.8; R-res<0.5; best of 12 single geometries<0.5; QA-max of 12 dual geometries>=0.8; all 00/10/01 controls in both donor replicates<0.5. Every record must be present and successful.

## InfoVQA_TRAIN (ANLS)

| Arm | Score (%) | Observed/expected | Empty | Truncated | Failed |
|---|---:|---:|---:|---:|---:|---:|
| full | 70.837 | 192/192 | 0 | 0 | 0 |
| b1 | 52.924 | 192/192 | 0 | 0 | 0 |
| rres512 | 57.305 | 192/192 | 0 | 0 | 0 |
| uniform256 | 47.256 | 192/192 | 0 | 0 | 0 |
| bag_rank__seed_11 | 54.878 | 192/192 | 0 | 0 | 0 |
| bag_rank__seed_23 | 52.085 | 192/192 | 0 | 0 | 0 |
| bag_rank__seed_37 | 52.444 | 192/192 | 0 | 0 | 0 |
| single_vertical_r0_c0 | 36.565 | 192/192 | 0 | 1 | 0 |
| single_vertical_r0_c16 | 44.620 | 192/192 | 0 | 0 | 0 |
| single_vertical_r1_c0 | 37.544 | 192/192 | 0 | 0 | 0 |
| single_vertical_r1_c16 | 46.391 | 192/192 | 0 | 0 | 0 |
| single_vertical_r2_c0 | 39.896 | 192/192 | 0 | 0 | 0 |
| single_vertical_r2_c16 | 43.388 | 192/192 | 0 | 0 | 0 |
| single_horizontal_r0_c0 | 35.293 | 192/192 | 0 | 0 | 0 |
| single_horizontal_r0_c16 | 43.940 | 192/192 | 0 | 0 | 0 |
| single_horizontal_r1_c0 | 36.759 | 192/192 | 0 | 0 | 0 |
| single_horizontal_r1_c16 | 41.333 | 192/192 | 0 | 0 | 0 |
| single_horizontal_r2_c0 | 39.776 | 192/192 | 0 | 0 | 0 |
| single_horizontal_r2_c16 | 42.404 | 192/192 | 0 | 0 | 0 |
| dual_vertical_r0_c0 | 38.306 | 192/192 | 0 | 0 | 0 |
| dual_vertical_r0_c16 | 43.649 | 192/192 | 0 | 0 | 0 |
| dual_vertical_r1_c0 | 38.848 | 192/192 | 0 | 0 | 0 |
| dual_vertical_r1_c16 | 46.598 | 192/192 | 0 | 0 | 0 |
| dual_vertical_r2_c0 | 47.168 | 192/192 | 0 | 0 | 0 |
| dual_vertical_r2_c16 | 38.982 | 192/192 | 0 | 0 | 0 |
| dual_horizontal_r0_c0 | 39.849 | 192/192 | 0 | 0 | 0 |
| dual_horizontal_r0_c16 | 40.526 | 192/192 | 0 | 0 | 0 |
| dual_horizontal_r1_c0 | 38.555 | 192/192 | 0 | 0 | 0 |
| dual_horizontal_r1_c16 | 39.556 | 192/192 | 0 | 0 | 0 |
| dual_horizontal_r2_c0 | 39.950 | 192/192 | 0 | 0 | 0 |
| dual_horizontal_r2_c16 | 40.024 | 192/192 | 0 | 0 | 0 |
| factorial__rep_0__00 | 31.711 | 192/192 | 0 | 0 | 0 |
| factorial__rep_0__10 | 49.980 | 192/192 | 0 | 0 | 0 |
| factorial__rep_0__01 | 49.370 | 192/192 | 0 | 0 | 0 |
| factorial__rep_1__00 | 34.275 | 192/192 | 0 | 0 | 0 |
| factorial__rep_1__10 | 48.946 | 192/192 | 0 | 0 | 0 |
| factorial__rep_1__01 | 50.923 | 192/192 | 0 | 0 | 0 |

Balanced 12-vs-12 finite-bank oracle dual − single: -1.491 pp; 95% parent-bootstrap CI ['-4.779', '1.665'].

Answer-dependent finite-bank best-single mean 78.707%; best-dual mean 77.216%. Full − R-res reference gap 13.532 pp.

Full-good/R-res-bad: 26; strict cases: 0.

Strict item IDs: 

Per-question scores for all 24 candidates, the selected single/dual, all six controls, original answers, and raw generated evidence are saved in report.json.

## TextVQA_TRAIN (VQA consensus (ten annotators))

| Arm | Score (%) | Observed/expected | Empty | Truncated | Failed |
|---|---:|---:|---:|---:|---:|---:|
| full | 86.667 | 192/192 | 0 | 0 | 0 |
| b1 | 83.177 | 192/192 | 0 | 0 | 0 |
| rres512 | 71.771 | 192/192 | 0 | 0 | 0 |
| uniform256 | 77.240 | 192/192 | 0 | 0 | 0 |
| bag_rank__seed_11 | 82.760 | 192/192 | 0 | 0 | 0 |
| bag_rank__seed_23 | 83.073 | 192/192 | 0 | 0 | 0 |
| bag_rank__seed_37 | 81.979 | 192/192 | 0 | 0 | 0 |
| single_vertical_r0_c0 | 59.479 | 192/192 | 0 | 0 | 0 |
| single_vertical_r0_c16 | 65.312 | 192/192 | 0 | 0 | 0 |
| single_vertical_r1_c0 | 59.167 | 192/192 | 0 | 0 | 0 |
| single_vertical_r1_c16 | 66.979 | 192/192 | 0 | 0 | 0 |
| single_vertical_r2_c0 | 60.677 | 192/192 | 0 | 0 | 0 |
| single_vertical_r2_c16 | 65.156 | 192/192 | 0 | 0 | 0 |
| single_horizontal_r0_c0 | 64.844 | 192/192 | 0 | 0 | 0 |
| single_horizontal_r0_c16 | 63.854 | 192/192 | 0 | 0 | 0 |
| single_horizontal_r1_c0 | 70.208 | 192/192 | 0 | 0 | 0 |
| single_horizontal_r1_c16 | 70.052 | 192/192 | 0 | 0 | 0 |
| single_horizontal_r2_c0 | 66.302 | 192/192 | 0 | 0 | 0 |
| single_horizontal_r2_c16 | 65.885 | 192/192 | 0 | 0 | 0 |
| dual_vertical_r0_c0 | 67.083 | 192/192 | 0 | 0 | 0 |
| dual_vertical_r0_c16 | 70.729 | 192/192 | 0 | 0 | 0 |
| dual_vertical_r1_c0 | 68.177 | 192/192 | 0 | 0 | 0 |
| dual_vertical_r1_c16 | 68.750 | 192/192 | 0 | 0 | 0 |
| dual_vertical_r2_c0 | 70.938 | 192/192 | 0 | 0 | 0 |
| dual_vertical_r2_c16 | 65.365 | 192/192 | 0 | 0 | 0 |
| dual_horizontal_r0_c0 | 58.021 | 192/192 | 0 | 0 | 0 |
| dual_horizontal_r0_c16 | 64.531 | 192/192 | 0 | 0 | 0 |
| dual_horizontal_r1_c0 | 67.656 | 192/192 | 0 | 0 | 0 |
| dual_horizontal_r1_c16 | 68.177 | 192/192 | 0 | 0 | 0 |
| dual_horizontal_r2_c0 | 71.094 | 192/192 | 0 | 0 | 0 |
| dual_horizontal_r2_c16 | 71.042 | 192/192 | 0 | 0 | 0 |
| factorial__rep_0__00 | 58.958 | 192/192 | 0 | 0 | 0 |
| factorial__rep_0__10 | 69.531 | 192/192 | 0 | 0 | 0 |
| factorial__rep_0__01 | 72.344 | 192/192 | 0 | 0 | 0 |
| factorial__rep_1__00 | 62.500 | 192/192 | 0 | 0 | 0 |
| factorial__rep_1__10 | 71.823 | 192/192 | 0 | 0 | 0 |
| factorial__rep_1__01 | 74.896 | 192/192 | 0 | 0 | 0 |

Balanced 12-vs-12 finite-bank oracle dual − single: 0.208 pp; 95% parent-bootstrap CI ['-2.188', '2.760'].

Answer-dependent finite-bank best-single mean 93.333%; best-dual mean 93.542%. Full − R-res reference gap 14.896 pp.

Full-good/R-res-bad: 29; strict cases: 0.

Strict item IDs: 

Per-question scores for all 24 candidates, the selected single/dual, all six controls, original answers, and raw generated evidence are saved in report.json.

## Equal-dataset oracle comparison

Dual − single: -0.642 pp; 95% CI ['-2.688', '1.383'].

Equal dataset weights; independent parent bootstrap within each dataset.

## Development latency

Fresh image processing, complete vision encoding, fixed geometry selection and native generation. Model loading and offline oracle search excluded.

Four development images only; fixed-mask diagnostic costs, not a deployable oracle or selector latency. Every high-resolution geometry still runs the full 1024-token visual encoder. No R-res speed claim transfers to these geometries.

| Arm | Cap | Median across independent image medians (ms) | Images |
|---|---:|---:|---:|
| full | 1 | 488.508 | 4 |
| full | 16 | 1103.465 | 4 |
| b1 | 1 | 409.168 | 4 |
| b1 | 16 | 1026.547 | 4 |
| rres512 | 1 | 170.438 | 4 |
| rres512 | 16 | 788.167 | 4 |
| uniform256 | 1 | 327.786 | 4 |
| uniform256 | 16 | 943.757 | 4 |
| bag_rank__seed_11 | 1 | 330.065 | 4 |
| bag_rank__seed_11 | 16 | 957.669 | 4 |
| bag_rank__seed_23 | 1 | 335.266 | 4 |
| bag_rank__seed_23 | 16 | 947.806 | 4 |
| bag_rank__seed_37 | 1 | 332.459 | 4 |
| bag_rank__seed_37 | 16 | 951.344 | 4 |
| single_vertical_r0_c0 | 1 | 330.458 | 4 |
| single_vertical_r0_c0 | 16 | 943.746 | 4 |
| dual_vertical_r0_c0 | 1 | 325.760 | 4 |
| dual_vertical_r0_c0 | 16 | 943.414 | 4 |

## Audit

Freeze: verified (714 files). Missing QA: 0; failed: 0; empty: 0; truncated: 1; error events: 0.

Original-answer scores independently recomputed. Native token/DeepStack/position/cache traces and answer-dependence declarations audited. Missing records remain missing; no incomplete-panel success declaration.
