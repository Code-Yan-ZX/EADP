# DocVQA VAL full-run local report

All 5,349 questions were included in each primary arm. ANLS and differences below are on a 0–100 scale.

| Method | ANLS | Truncated |
| --- | ---: | ---: |
| full | 94.537 | 0 |
| b2 | 68.315 | 1 |
| b1 | 72.118 | 1 |
| rres512 | 80.307 | 2 |
| magnitude_v__seed_11 | 66.094 | 2 |
| magnitude_v__seed_23 | 59.816 | 11 |
| magnitude_v__seed_37 | 67.520 | 6 |

Paired differences use the same question rows, source-parent bootstrap, 10,000 draws.

| Comparison | Difference (points) | 95% interval |
| --- | ---: | ---: |
| magnitude_v__seed_11 - full | -28.443 | [-29.878, -26.983] |
| magnitude_v__seed_11 - b2 | -2.221 | [-3.890, -0.530] |
| magnitude_v__seed_11 - b1 | -6.023 | [-7.671, -4.378] |
| magnitude_v__seed_11 - rres512 | -14.213 | [-15.971, -12.429] |
| magnitude_v__seed_23 - full | -34.721 | [-36.202, -33.202] |
| magnitude_v__seed_23 - b2 | -8.499 | [-10.219, -6.782] |
| magnitude_v__seed_23 - b1 | -12.301 | [-14.033, -10.609] |
| magnitude_v__seed_23 - rres512 | -20.491 | [-22.355, -18.624] |
| magnitude_v__seed_37 - full | -27.017 | [-28.400, -25.607] |
| magnitude_v__seed_37 - b2 | -0.794 | [-2.464, +0.863] |
| magnitude_v__seed_37 - b1 | -4.597 | [-6.255, -2.973] |
| magnitude_v__seed_37 - rres512 | -12.787 | [-14.515, -11.080] |
| b2 - b1 | -3.803 | [-4.822, -2.799] |

Truncation sensitivity is secondary: it removes questions for which either arm truncated; denominators can differ between pairs. The primary result retains every question.

Frozen run SHA256: `61605b39f40ee38d2c6d2b2ba1a1ab24929688efd7b6b664d185ce673bfae53c`
Ordered result snapshot SHA256: `f34c45331c9f32f3d43330a84e5cd7cbd8183707fcb9bd9e95eec41b96941281`
Detailed per-question primary scores and all sensitivity denominators are in the adjacent JSON files.
