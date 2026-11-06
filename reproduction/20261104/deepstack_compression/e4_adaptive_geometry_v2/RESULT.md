# Selected-position geometry routing: exploratory DEV diagnostic

Each task has 128 previously evaluated image groups. For each rule, rank the batch using only the K128 selection geometry or a fixed random seed, allocate 42 images K512 and 86 images K128 (mean K254), and reuse their previously scored predictions. Compare with the same images at fixed K256. All six rules are shown; changes are percentage points.

| Rule | DocVQA mean / change | TextVQA mean / change | AI2D mean / change |
|---|---:|---:|---:|
| nearest_low | 60.84 / -7.01 | 80.47 / -0.31 | 77.34 / -4.69 |
| occupied8_high | 62.91 / -4.94 | 79.30 / -1.48 | 78.12 / -3.91 |
| entropy4_high | 59.58 / -8.26 | 79.45 / -1.33 | 77.34 / -4.69 |
| random_0 | 62.87 / -4.98 | 78.44 / -2.34 | 78.12 / -3.91 |
| random_1 | 60.12 / -7.73 | 77.97 / -2.81 | 77.34 / -4.69 |
| random_2 | 60.05 / -7.79 | 79.14 / -1.64 | 78.91 / -3.12 |

Fixed K256 means are 67.85, 80.78 and 82.03, respectively. The three geometry heuristics were nearest-neighbor distance (low first), occupied cells in an 8×8 grid (high first), and entropy over a 4×4 grid (high first). The results are descriptive, posthoc and unadjusted; no uncertainty interval is calculated here. The batch rank requires the other images and task identity, so it is not an online policy. Equal mean K does not establish equal compute: extracting a K128 selection and potentially generating again at K512 adds cost that was not measured. The independently reviewed results do not establish that all adaptive strategies fail.

The rule definitions preceded extraction. The final analysis admission correction was made after feature extraction and independent inspection of the features; its provenance correction is preserved in `analysis_freeze005.json`. The implementation did not open scores while making that correction. The extractor parsed mixed existing records containing predictions but used only identity and selected-position fields.

Sources: `analysis001/analysis.json`, `analysis001/assignments.jsonl`, and `../e1_independent_review/e4_adaptive_geometry_v2_analysis_review.json`. The independent review recomputed all 2,304 assignments and scores.
