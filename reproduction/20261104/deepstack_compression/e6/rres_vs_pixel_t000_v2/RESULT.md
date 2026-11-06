# R-res 512/K256 versus PixelPrune τ=0 — posthoc DEV comparison

Both complete 384-question natural-generation panels and their scores passed formal and independent review. This comparison was frozen after observing each method's DEV results, then independently recomputed. All 384 task/qid/group and original image/message identities matched; each task has 128 questions in 128 image groups. Both methods ended all 384 outputs with EOS and had no truncations.

| Task | R-res score | Pixel τ=0 score | R-res − Pixel (pp), paired 95% CI | R-res higher / lower / tied |
|---|---:|---:|---:|---:|
| DocVQA | 80.98 | 91.62 | −10.64 [−15.79, −5.75] | 3 / 35 / 90 |
| TextVQA | 72.81 | 83.75 | −10.94 [−16.56, −5.55] | 2 / 18 / 108 |
| AI2D | 82.03 | 82.03 | 0.00 [−3.91, 3.91] | 3 / 3 / 122 |

Scores use the already accepted task metrics: DocVQA ANLS, TextVQA consensus and AI2D accuracy. The 95% intervals are question-weighted, paired image-group percentile bootstraps with 10,000 draws and fixed seed 20261104. All questions are retained. The intervals are uncorrected across these exploratory comparisons; They do not establish noninferiority or confirmation.

R-res records copied six generation scalars from E1; its runtime outer hook checked all six actual `generate` kwargs and EOS before each request, but did not separately archive actual per-request kwargs or the complete effective generation configuration. Pixel explicitly recorded `num_beams` and `use_cache` in addition. Thus matching fully effective runtime configuration cannot be independently asserted. R-res uses a 512px input and K=256; Pixel uses 1024 visual input tokens with dynamic K. This is not an equal-compute comparison or a speed result.

Evidence: `result001.json`, `monitor_compare001`, `e1_independent_review/rres_vs_pixel_t000_v2_review.json` and `monitor_rres_vs_pixel_t000_v2`. The independent review did not open additional ground truth; existing mixed prediction/scored files were opened for pairing. Version 1 was frozen but not executed; version 2 corrects its inaccurate description of the six online control checks.
