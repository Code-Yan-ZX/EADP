# Multi-stream budget-gain diagnostic: exploratory 32-image screen

For each existing DEV anchor, the previously fixed outcome is its already-scored K512-D1 minus K128-D1, joined to fixed features from the accepted full main-vision and three DeepStack streams. The table reports every planned Spearman correlation (no significance test).

| Feature | DocVQA (n=11) | TextVQA (n=11) | AI2D (n=10) |
|---|---:|---:|---:|
| main_cv | -0.243 | -0.500 | +0.174 |
| ds0_cv | +0.163 | -0.300 | +0.058 |
| ds1_cv | +0.121 | +0.200 | +0.406 |
| ds2_cv | -0.555 | +0.300 | -0.174 |
| main_ds0_rank_corr | -0.354 | -0.200 | -0.174 |
| main_ds1_rank_corr | -0.131 | +0.500 | -0.174 |
| main_ds2_rank_corr | +0.415 | +0.500 | +0.406 |
| four_stream_rank_disagreement | +0.238 | -0.400 | -0.058 |

The outcome is tied heavily: the K512−K128 change is zero on 4/11 DocVQA, 10/11 TextVQA and 9/10 AI2D images. In the latter two tasks the correlation is driven by a single nonzero outcome, so the displayed values cannot establish a useful or generalizable predictor. The eight features are four per-stream token-norm coefficients of variation, three main-vs-DeepStack token-norm rank correlations, and mean four-stream per-token rank disagreement. They were fixed before this diagnostic read its individual tensors and scores, but after other DEV results had been observed; all 24 associations are posthoc and exploratory.

No evaluation answer was used to train a selector or fit a threshold. This diagnostic uses existing scores and captures, makes no new model predictions, and does not establish causality, quality improvement, an online decision policy, equal compute or measured latency. Computing the full-feature statistics itself has a cost. Do not choose a rule based on these 10–11-image task samples; meaningful next steps would require a separately frozen hypothesis and independent data, with all feature and selection costs measured.

Sources: `result001.json`, `../e4_multistream_budget_diagnostic/PLAN.md`, `../e1_independent_review/e4_multistream_budget_v2_prelaunch_review.json`, and `../e1_independent_review/e4_multistream_budget_v2_result_review.json`. The independent review reloaded all 32 captures and recomputed feature values, joins and all 24 associations.
