# Multi-stream association on complete DEV

Post hoc descriptive Spearman correlation of each fixed stream feature with the same-question D1 score difference K512 − K128. All 384 questions are included: 128 per task, each with a unique within-task image group. Every feature and outcome is defined for all 128 task cases. Average ranks are used for ties.

| Feature | DocVQA | TextVQA | AI2D |
|---|---:|---:|---:|
| main_cv | -0.162 | -0.094 | 0.167 |
| ds0_cv | -0.150 | -0.013 | 0.062 |
| ds1_cv | -0.153 | 0.030 | 0.114 |
| ds2_cv | -0.194 | 0.060 | 0.073 |
| main_ds0_rank_corr | -0.116 | 0.021 | 0.044 |
| main_ds1_rank_corr | -0.157 | 0.037 | 0.100 |
| main_ds2_rank_corr | -0.033 | 0.081 | 0.059 |
| four_stream_rank_disagreement | 0.218 | -0.072 | -0.037 |

For all 24 entries, paired n = 128 and undefined features = 0. Feature tie excess (n minus number of distinct values) is 0 for each entry; outcome tie excess is 97, 118 and 125 for DocVQA, TextVQA and AI2D respectively. These numbers are duplicate excesses, not counts of all observations belonging to tied groups. Cross-task group overlaps in this cohort are DocVQA_VAL|TextVQA_VAL: 0, DocVQA_VAL|AI2D_TEST: 0, TextVQA_VAL|AI2D_TEST: 0.

The four CVs summarize variation in each stream’s 1,024 token norms; the next three features correlate main and DS token-norm ranks, while the last is mean position-wise variance of four percentile ranks. The original 32 and pilot nine had already been inspected, all data are the same DEV set, and these correlations are exploratory: no held-out, causal, feature-selection, router, threshold or significance claim follows. The 343-image run had completed before the mechanical analysis-gate repair; the repair was independently reviewed before the analysis, with the earlier failed attempt retained.

For the nine and 343 captures, independently recomputed float64 norm vectors support the feature values, but full visual tensors were not archived; their linkage to the model is runtime capture/identity evidence. E1/E4 mixed prediction records were parsed only for identity and existing independently checked DEV scores; no reference answers were opened by this analysis.

Analysis: `analysis002.json` SHA256 `6b3852aeff322c537cd0fed9fd28bc0d9ea7b0569a3aab1a091a1ee674ecfaae`. Independent statistical review: `../e1_independent_review/e4_multistream_expansion343_analysis_review.json` SHA256 `7d3ac4a3ef6f628199f48d34aeb303d664d6d8a60a5721334368a8184daf72ab`. Full 343 result review: `../e1_independent_review/e4_multistream_expansion343_result_review.json` SHA256 `4360e3dad55bbac3f8cd988b96d5470c300ac946db3d54324219dcdfabfd68fc`. Source/exit evidence and all unrounded per-question data remain in those artifacts.
