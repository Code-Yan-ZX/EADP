# Div/CD DEV: independent quality comparison

The full 384-question DEV panel contains 1,536 completed natural generations, four per question. The 1,536 author and independent scores and all 12 paired image-group bootstrap contrasts were independently recomputed. CONFIRM and FINAL answer values were not parsed. Source, run, and review identities are recorded in `e1_independent_review/e6_selected_scores_review.json`.

| Task | Div stock | Div adapted | CD stock | CD adapted |
|---|---:|---:|---:|---:|
| DocVQA | 49.88 | 66.11 | 53.28 | 57.18 |
| TextVQA | 71.25 | 75.47 | 69.61 | 78.28 |
| AI2D | 66.41 | 77.34 | 69.53 | 75.78 |

Scores are percentages. Intervals below are unadjusted 95% percentile image-group bootstrap intervals for paired differences, in percentage points.

| Task | Contrast | Difference [95% interval] | Improved / declined / unchanged |
|---|---|---:|---:|
| DocVQA | div_adapted_minus_stock | +16.23 [+8.93, +23.69] | 42 / 10 / 76 |
| DocVQA | cd_adapted_minus_stock | +3.90 [-0.78, +8.65] | 25 / 9 / 94 |
| DocVQA | adapted_div_minus_cd | +8.94 [+0.49, +17.73] | 35 / 28 / 65 |
| DocVQA | stock_div_minus_cd | -3.39 [-11.81, +5.19] | 29 / 33 / 66 |
| TextVQA | div_adapted_minus_stock | +4.22 [-2.11, +10.55] | 15 / 9 / 104 |
| TextVQA | cd_adapted_minus_stock | +8.67 [+3.20, +14.45] | 15 / 4 / 109 |
| TextVQA | adapted_div_minus_cd | -2.81 [-8.52, +2.89] | 8 / 12 / 108 |
| TextVQA | stock_div_minus_cd | +1.64 [-4.38, +7.97] | 11 / 9 / 108 |
| AI2D | div_adapted_minus_stock | +10.94 [+3.91, +17.97] | 18 / 4 / 106 |
| AI2D | cd_adapted_minus_stock | +6.25 [-1.56, +14.06] | 17 / 9 / 102 |
| AI2D | adapted_div_minus_cd | +1.56 [-3.91, +7.03] | 8 / 6 / 114 |
| AI2D | stock_div_minus_cd | -3.12 [-10.16, +3.12] | 7 / 11 / 110 |

Each method compares its author selection with two simultaneous inference settings: stock uses P1 sequential positions and DS off; adapted uses P3 original positions and all three DS streams on. Therefore adapted-minus-stock is a combined protocol difference; this panel cannot attribute the difference to positions or DS separately. The comparison is exploratory and intervals are not corrected for the twelve task/contrast looks; intervals crossing zero do not establish equivalence. AI2D uses the deterministic local parser. Full-panel arrays are summarized, while the earlier three-task smoke archived full tensors. No latency or memory claim follows from this quality panel.

Each task has 128 questions from 128 distinct image groups. All 1,536 outputs reached EOS; none were empty or truncated. Bootstrap uses 10,000 resamples with seed 20261104. The independent score review proof `e1_independent_review/e6_selected_scores_review.json` has SHA256 `a76d68f1b429be2980a16bfc4fb98de4f232757f114420b0d9c5f7e195099b1a`.
