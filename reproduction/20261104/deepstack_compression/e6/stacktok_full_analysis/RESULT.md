# StackTok local Qwen3 adaptation: full DEV comparison

The local Qwen3 adaptation of the pinned StackTok selector completed all 384 frozen DEV questions at K=256, three DeepStack streams on and retained original 3-axis positions. The author repository supplies a Qwen2.5 integration; this Qwen3 integration is local. All 384 answers terminated with EOS. The complete run passed no-GT formal and independent acceptance before scoring. The model, prompt, fixed-resolution inputs and generation recipe were matched to accepted EADP and PixelPrune references, but K and computational cost are not interchangeable.

| Task, 128 images each | StackTok K256 | EADP K256 | Pixel t0 | Pixel t045 |
|---|---:|---:|---:|---:|
| DocVQA | 69.07 | 67.85 | 91.62 | 79.16 |
| TextVQA | 77.89 | 80.78 | 83.75 | 67.19 |
| AI2D | 83.59 | 82.03 | 82.03 | 82.03 |

Scores are task-specific percentages under the project’s accepted scoring convention. StackTok minus each comparator, in percentage points with paired image-group percentile 95% intervals, is:

| Task | vs EADP K256 | vs Pixel t0 | vs Pixel t045 |
|---|---:|---:|---:|
| DocVQA | +1.23 [−7.94, +10.78] | −22.55 [−29.46, −15.73] | −10.09 [−18.81, −1.28] |
| TextVQA | −2.89 [−8.20, +2.11] | −5.86 [−11.02, −0.86] | +10.70 [+3.44, +18.05] |
| AI2D | +1.56 [−3.12, +6.25] | +1.56 [−2.34, +5.47] | +1.56 [−2.34, +6.25] |

Intervals use 10,000 paired image-group bootstrap draws, seed 20261104, and no multiple-comparison correction. This is exploratory DEV after earlier investigations; all comparisons are reported and no conclusion of equivalence follows from an interval crossing zero. The three comparisons against EADP do not establish a quality advantage or the prespecified 0.5-point noninferiority. Pixel has a variable K and K256 methods may have different visual/selector costs. This run was instrumented for correctness and is not a latency or GPU-memory comparison.

The full generation used single-call author selector semantics and local Qwen3 position/DeepStack adaptation. The three-anchor real-model identity diagnostic archived high-dimensional values; the full panel retains runtime checks and summaries, not complete selected DeepStack or KV tensors, so their contents cannot be independently reconstructed offline. Full-run no-GT evidence is `../stacktok_full/run_001_verification.json` and `../../e1_independent_review/stacktok_full_run001_review.json`. The scoring source preflight is `../../e1_independent_review/stacktok_full_score_prelaunch_review.json`. All 384 new scores, 1,152 reference scores and nine intervals were separately rechecked by `../../e1_independent_review/stacktok_scores_independent_review.json`, using the previously accepted scoring implementations and direct image-level bootstrap; the original same-author reconstruction remains marked as such. Raw score and analysis artifacts are in `score001/`.
