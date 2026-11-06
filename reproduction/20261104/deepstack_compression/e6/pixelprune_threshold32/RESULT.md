# PixelPrune threshold calibration: 32-image CPU probe

The frozen, distinct-group sample contains 11 DocVQA, 11 TextVQA and 10 AI2D DEV images. On their already independently validated processor pixels, the author's `pred_2d`/`max` selector was run on CPU at 10 fixed thresholds. All 320 cases completed, and a separately supervised process reran every selector call and reproduced every selected index. No answers, new predictions, model, or GPU were used. The grid was chosen after earlier DEV quality results and a one-image CPU feasibility probe, so any choice and subsequent quality experiment are exploratory.

| Threshold | All K min / median / max | Doc median | Text median | AI2D median | Images with K=1 | Images with 224≤K≤288 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 292 / 694.5 / 1022 | 697 | 706 | 435.5 | 0 | 0 |
| 0.40 | 1 / 367 / 723 | 405 | 359 | 369 | 1 | 6 |
| 0.45 | 1 / 327 / 680 | 320 | 321 | 365.5 | 2 | 6 |
| 0.50 | 1 / 1 / 667 | 1 | 1 | 1 | 30 | 1 |
| 0.55 | 1 / 1 / 648 | 1 | 1 | 1 | — | 1 |
| 0.60 | 1 / 1 / 621 | 1 | 1 | 1 | — | 0 |
| 0.65 | 1 / 1 / 582 | 1 | 1 | 1 | — | 0 |
| 0.70 | 1 / 1 / 536 | 1 | 1 | 1 | — | 0 |
| 0.75 | 1 / 1 / 506 | 1 | 1 | 1 | — | 0 |
| 0.80 | 1 / 1 / 451 | 1 | 1 | 1 | — | 0 |

Threshold **0.45** is a reasonable candidate to freeze for an exploratory quality test because its observed median K (327) is closest to 256 among these tested settings without the widespread K=1 collapse at 0.50. It still has wide, task-dependent budgets and 2/32 examples with K=1: it is not an equal-K or equal-compute K256 baseline. The interval 224–288 is a descriptive ±32-token neighborhood, not a formal acceptance criterion. None of these 32 observed image sequences increased K as threshold increased; this finite check does not prove monotonicity on other inputs. At threshold 0 the selector uses its fast path; positive thresholds use the author's anchored Python path. CPU selections have not been shown equal to GPU selections, and CPU wall times here do not establish GPU or end-to-end latency. Before any quality test, verify the chosen setting on the target GPU with the actual processor, DeepStack/position handling, full outputs, bounded runtime and no dropped failures. A later DEV quality comparison after this threshold selection must remain explicitly post-hoc and exploratory.

Independent review: `e1_independent_review/pixel_threshold32_review.json`, SHA-256 `2dba41861611ba99533140dfa2058ec3cb29019e56a76de6b0766b9c5f43252e`; its `monitor_pixel_threshold32` and the original `monitor_run001` both ended completed/0 with empty process groups. The exact plan, full per-image counts and indices are preserved in `plan001.json`, `run001/` and the review proof.
