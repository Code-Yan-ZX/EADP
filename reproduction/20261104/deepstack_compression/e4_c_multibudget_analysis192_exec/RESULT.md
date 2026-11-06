# Exploratory 192-image four-budget timing

The complete 7,920-job run and frozen analysis passed formal and independent no-ground-truth result checks, and the full statistics were independently recomputed. Each task has 64 image groups; each image/protocol/budget has five measured repeats. Reduce each set of five to its median, then summarize the 64 image medians. There are 240 additional warmup jobs. Intervals are unadjusted 95% image-group bootstrap percentiles (10,000 resamples), conditional on this posthoc DEV selection and shared machine environment.

## Natural generation: first-token time

| Task | K | Median ms [95% CI] | Paired K − 256 ms [95% CI] | Paired K / 256 [95% CI] |
|---|---:|---:|---:|---:|
| DocVQA | 128 | 350.53 [348.74, 352.05] | -42.14 [-43.03, -41.40] | 0.892 [0.890, 0.894] |
| DocVQA | 256 | 391.65 [390.28, 393.83] | 0.00 [0.00, 0.00] | 1.000 [1.000, 1.000] |
| DocVQA | 512 | 463.52 [463.04, 466.41] | 72.37 [72.00, 73.17] | 1.186 [1.184, 1.188] |
| DocVQA | 1024 | 481.32 [479.23, 482.70] | 88.84 [88.44, 89.42] | 1.227 [1.225, 1.230] |
| TextVQA | 128 | 301.02 [300.22, 301.73] | -42.29 [-42.53, -41.59] | 0.876 [0.875, 0.879] |
| TextVQA | 256 | 343.02 [342.47, 344.09] | 0.00 [0.00, 0.00] | 1.000 [1.000, 1.000] |
| TextVQA | 512 | 416.34 [415.67, 416.82] | 72.84 [72.18, 73.34] | 1.213 [1.211, 1.214] |
| TextVQA | 1024 | 431.92 [430.97, 433.15] | 88.71 [88.38, 89.13] | 1.259 [1.258, 1.261] |
| AI2D | 128 | 307.72 [302.30, 312.42] | -42.14 [-43.15, -41.28] | 0.879 [0.876, 0.885] |
| AI2D | 256 | 347.26 [344.87, 352.04] | 0.00 [0.00, 0.00] | 1.000 [1.000, 1.000] |
| AI2D | 512 | 418.08 [417.03, 424.72] | 72.19 [71.42, 72.65] | 1.207 [1.204, 1.210] |
| AI2D | 1024 | 435.11 [433.05, 438.18] | 88.11 [87.60, 88.54] | 1.254 [1.249, 1.256] |

## Fixed 32 generated tokens: first-token time

| Task | K | Median ms [95% CI] | Paired K − 256 ms [95% CI] | Paired K / 256 [95% CI] |
|---|---:|---:|---:|---:|
| DocVQA | 128 | 349.71 [348.32, 351.72] | -42.50 [-43.21, -41.26] | 0.891 [0.890, 0.894] |
| DocVQA | 256 | 392.16 [390.25, 394.98] | 0.00 [0.00, 0.00] | 1.000 [1.000, 1.000] |
| DocVQA | 512 | 464.93 [463.63, 466.93] | 72.42 [71.71, 73.55] | 1.187 [1.184, 1.189] |
| DocVQA | 1024 | 480.37 [478.96, 482.70] | 88.22 [87.60, 89.74] | 1.227 [1.224, 1.230] |
| TextVQA | 128 | 301.23 [300.85, 302.16] | -41.89 [-42.30, -41.36] | 0.878 [0.877, 0.879] |
| TextVQA | 256 | 343.29 [342.56, 343.73] | 0.00 [0.00, 0.00] | 1.000 [1.000, 1.000] |
| TextVQA | 512 | 416.22 [415.28, 416.86] | 72.93 [72.35, 73.63] | 1.214 [1.212, 1.215] |
| TextVQA | 1024 | 431.69 [431.08, 432.53] | 88.80 [88.19, 89.34] | 1.260 [1.258, 1.262] |
| AI2D | 128 | 306.06 [302.51, 312.41] | -42.32 [-43.19, -41.37] | 0.880 [0.878, 0.884] |
| AI2D | 256 | 347.54 [345.79, 351.52] | 0.00 [0.00, 0.00] | 1.000 [1.000, 1.000] |
| AI2D | 512 | 418.51 [416.80, 423.12] | 72.20 [71.30, 72.70] | 1.207 [1.204, 1.210] |
| AI2D | 1024 | 434.89 [433.15, 439.84] | 88.24 [87.79, 88.60] | 1.251 [1.248, 1.255] |

## Generation completion

Median milliseconds across image medians. Completion is synchronized generate return from immediately before image read; it excludes output token copying, fsync and answer text decoding. Natural output lengths vary; fixed32 generates exactly 32 tokens, with 31 later decode forwards. All natural runs ended in EOS and none were truncated.

| Task / protocol | K128 | K256 | K512 | K1024 |
|---|---:|---:|---:|---:|
| DocVQA / natural | 484.94 | 567.36 | 684.89 | 704.52 |
| DocVQA / fixed32 | 1724.08 | 1771.54 | 1843.72 | 1866.45 |
| TextVQA / natural | 390.48 | 433.68 | 506.93 | 522.50 |
| TextVQA / fixed32 | 1676.68 | 1716.72 | 1795.77 | 1818.10 |
| AI2D / natural | 356.10 | 395.16 | 470.10 | 486.02 |
| AI2D / fixed32 | 1684.74 | 1727.93 | 1799.47 | 1821.65 |

## Observation and limits

All 7,680 measured jobs are retained. Recorded intersections with first-token / generation intervals are: model-supervisor health 1/3, watcher-supervisor health 2/3, process query through first fsync 60/154, and manual health 5/5. These categories can overlap and are not summed. The observer second fsync, manual-log fsync, marker write, light polling and all unsampled background interference remain unknown. Other services were active; results are not an isolated-machine guarantee.

Manual health checks were actually completed at 19:07 and 20:04 local time, within one hour of the prior check. The separately preplanned coverage check uses a ±5-minute window around each elapsed hour of the measured jobs; neither manual entry falls within those specific windows, so its manual coverage is correctly marked unknown while supervisor health was observed at both elapsed-hour targets. That coverage label must not be read as proof that no manual checks occurred.

TTFT starts immediately before image read and ends after selection of the first output token and device synchronization. The completion measure excludes downstream text decoding and persistence, so neither metric is a complete user-facing end-to-end latency. Same K does not imply equal compute between different methods. Bootstrap intervals do not measure differences between hardware, other days or unobserved common load. These are exploratory posthoc DEV timings and not confirmatory results.

Evidence: `analysis001.json`; `monitor_analysis001/terminal.json`; `../e4_c_multibudget_worker192_v6_accept/verification001.json`; `../e1_independent_review/e4c_multibudget_worker192_v6_result_review.json`; `../e1_independent_review/e4c_multibudget_analysis192_review.json`.
