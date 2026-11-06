# PixelPrune threshold 0.1: exploratory complete DEV panel

Threshold 0.1 was selected after inspecting previous DEV results and calibration. All 384 questions generated successfully, with 128 per task and EOS for every generation. Formal and independent no-ground-truth acceptance, scoring, and an independent per-question and bootstrap recomputation all passed. The score review is `../../e1_independent_review/pixel_t010_dev384_scores_review.json`; the machine-readable results are `score001/analysis.json`.

| Task | t=0.1 score | t=0 score | t=0.1 − t=0 (percentage points, 95% CI) | t=0.1 K median (range) |
| --- | ---: | ---: | ---: | ---: |
| DocVQA | 91.69 | 91.62 | +0.07 [−1.55, +2.14] | 510.5 (152–830) |
| TextVQA | 82.50 | 83.75 | −1.25 [−4.14, +1.56] | 625 (231–952) |
| AI2D | 82.03 | 82.03 | 0.00 [−3.12, +3.12] | 400.5 (157–857) |

EADP K=256 scores are 67.85, 80.78 and 82.03, respectively. The paired differences for t=0.1 minus EADP K=256 are +23.85 [16.56, 31.20], +1.72 [−1.56, 5.08] and 0.00 [−4.69, 4.69] percentage points. Each interval is a 10,000-replicate paired image-group percentile bootstrap, unadjusted for multiple comparisons. These intervals do not establish the preset 0.5-point noninferiority to t=0, and the post-hoc DEV result is not a confirmatory test. Dynamic K differs across methods and tasks and does not establish equal compute or latency. Runtime visual/DeepStack/KV checks were recorded, but the complete high-dimensional arrays were not archived.

The separately frozen paired comparison of t=0.1 minus t=0.2 gives +0.19 [−0.69, +0.96], +0.16 [−1.87, +2.11], and +0.78 [0.00, +2.34] percentage points, respectively. `../../e1_independent_review/pixel_t010_t020_comparison_review.json` independently checks all 384 pairs and the bootstrap. These are exploratory, unadjusted intervals; t=0.1 has a higher median K on each task, and neither K nor these scores establish latency or equal compute.
