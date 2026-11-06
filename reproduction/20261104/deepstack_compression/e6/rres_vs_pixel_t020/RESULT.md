# R-res 512/K256 versus PixelPrune τ=0.2 — exploratory DEV comparison

Both 384-question natural-generation panels, their scored outputs, and this paired comparison passed independent review. All task/qid/group and original image/message identities matched; each task has 128 questions in 128 image groups. Both methods ended all 384 outputs with EOS, with no truncations.

| Task | R-res score | Pixel τ=0.2 score | R-res − Pixel (pp), paired 95% CI | R-res higher / lower / tied |
|---|---:|---:|---:|---:|
| DocVQA | 80.98 | 91.50 | −10.52 [−15.81, −5.52] | 5 / 34 / 89 |
| TextVQA | 72.81 | 82.34 | −9.53 [−15.08, −4.22] | 4 / 18 / 106 |
| AI2D | 82.03 | 81.25 | +0.78 [−3.91, +5.47] | 5 / 4 / 119 |

Scores use accepted task metrics: DocVQA ANLS, TextVQA consensus and AI2D accuracy. Intervals are paired, question-weighted image-group percentile bootstraps (10,000 draws, seed 20261104). The analysis was planned after seeing DEV results, including the τ=0 comparison; intervals are uncorrected for multiple comparisons and do not establish confirmation or noninferiority.

R-res records copied six generation scalars from E1; its outer runtime hook checked all six actual `generate` kwargs and EOS on each request, but did not separately archive the actual kwargs or the complete effective generation configuration. Pixel explicitly recorded `num_beams` and `use_cache` too. Fully effective runtime configuration equality cannot therefore be independently asserted. R-res uses a 512px input and K=256; Pixel uses a 1024-token visual input and dynamic K. This is neither an equal-compute comparison nor a speed result.

Evidence: `result001.json`, `monitor_compare001`, `e1_independent_review/rres_vs_pixel_t020_prelaunch_review.json`, `e1_independent_review/rres_vs_pixel_t020_review.json`, and `monitor_rres_vs_pixel_t020_result`. No additional ground truth was opened; existing mixed prediction/scored files were opened for pairing.
