# E4-C 32-image four-budget exploratory timing pilot

The frozen analysis `analysis004.json` completed under `monitor_analysis004`, then `e1_independent_review/e4c_multibudget_analysis004_review.json` independently recomputed it and exited successfully. This sample and the decision to expand were chosen after earlier DEV/latency observations. Each task has only 11, 11 and 10 measurement image groups (DocVQA, TextVQA, AI2D). These data do not establish performance on a population, other hardware, or production workloads. Each entry is the median across images of that image's five repetitions; 30 other images served only as warmups.

Natural greedy first-selected-token TTFT, ms:

| Task | K128 | K256 | K512 | K1024 |
|---|---:|---:|---:|---:|
| DocVQA (11) | 351.0 | 395.1 | 468.6 | 481.5 |
| TextVQA (11) | 310.5 | 352.6 | 425.6 | 438.0 |
| AI2D (10) | 306.2 | 350.0 | 423.7 | 437.0 |

Natural K1024 minus K256 paired TTFT differences are 86.8 ms [84.3, 87.6], 86.2 [84.7, 87.0], and 86.4 [85.8, 87.3] for Doc/Text/AI, respectively. Intervals are unadjusted 95% percentile bootstraps of the **median per-image paired difference**, using 10,000 fixed-seed image-group resamples (NumPy 2.2.6); they describe this small exploratory sample and are not a population or multiplicity-corrected guarantee. Medians of differences need not equal differences of table medians.

The naturally terminated generation-completion medians for K128/K256/K512/K1024 are Doc 522.0/536.4/611.0/616.9 ms, Text 400.0/443.4/517.7/529.3 ms and AI 357.9/398.2/470.9/483.2 ms; varying output length affects these numbers. All natural runs reached EOS; see JSON for token counts. The fixed-32 protocol generates 32 tokens with 31 later decode forwards; it and all 1,520 individual job windows, token counts, pairwise contrasts, and raw resampling summaries are retained in the JSON. Generation completion ends at synchronized `generate` return before token copy, fsync and text decoding: it is not full end-to-end user latency. TTFT begins immediately before reading the input image and ends at synchronized selection of the first token.

All 1,280 measured jobs and 240 warmups were retained. Of measured jobs, recorded process-snapshot query and query-through-first-fsync windows overlap 12 TTFT and 24 generation windows; recorded model/watcher health and manual-check windows overlap none. Watcher second-fsync duration, manual-log fsync, light polling and other residual background activity remain unknown: zero recorded overlap does not establish an interference-free run. No samples were excluded. Environment, exact source, observer, verification and reference hashes are in the analysis JSON and associated proof artifacts.
