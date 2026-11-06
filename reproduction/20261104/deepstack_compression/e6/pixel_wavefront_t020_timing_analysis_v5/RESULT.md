# PixelPrune τ=0.2 wavefront: three-anchor model TTFT

The full 24-process run, formal acceptance, independent result review, frozen analysis and independent statistics recomputation all completed successfully. There are three posthoc task anchors and two implementations; each arm has one warmup process and three measured processes. Each measured process independently loads the model before the timing boundary. The observer remains active; no samples are dropped.

| Task anchor | Author ms (three samples) | Wavefront ms (three samples) | Author → wavefront median ms | Median change | Same-index wavefront/author ratios |
|---|---:|---:|---:|---:|---:|
| AI2D_TEST / 368100 | 1637.8, 1646.1, 1649.9 | 1517.8, 1546.1, 1509.3 | 1646.1 → 1517.8 | -7.79% | 0.927, 0.939, 0.915 |
| DocVQA_VAL / 16539 | 1715.6, 1719.4, 1714.6 | 1577.4, 1591.6, 1656.6 | 1715.6 → 1591.6 | -7.23% | 0.919, 0.926, 0.966 |
| TextVQA_VAL / 39421 | 1726.9, 1781.7, 1719.8 | 1682.6, 1636.2, 1642.6 | 1726.9 → 1642.6 | -4.88% | 0.974, 0.918, 0.955 |

TTFT begins immediately before image read and ends when the first token is selected and the GPU synchronized. Model loading occurs before that boundary, but each measured process is fresh; the separate warmup processes do not establish in-process steady state. Selection-observation instrumentation is included in both arms. Same-index comparisons use preassigned repeat labels: the author and wavefront processes ran at different times, not simultaneously. These samples are descriptive, not a confidence interval or a population estimate. Do not compare directly with warm-process 192-image EADP timing, t=0 Pixel or equal-compute baselines.

The supplemental observer covered the measured range and recorded 10 snapshots. Its bounded ps, Docker and query-through-first-fsync windows did not intersect any measured TTFT window in the 24 jobs. This does not establish an interference-free run: the second fsync, ordinary observer-supervisor health windows, polling and all unsampled external activity remain unknown, and observer queries themselves can affect timing. Other services were active. All 24 jobs are retained.

Evidence: `analysis001.json`; `../pixel_wavefront_t020_timing_accept_v5/verification001.json`; `../../e1_independent_review/pixel_wavefront_t020_timing_result_review.json`; `../../e1_independent_review/pixel_wavefront_t020_timing_analysis_review.json`; `../pixel_wavefront_t020_timing_coverage_v3/coverage001.json`. Original three-anchor selector, DeepStack and output equivalence was validated only for the recorded cases; full intermediate tensors were not all archived.
