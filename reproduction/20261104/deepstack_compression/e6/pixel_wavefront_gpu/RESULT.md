# Anchored PixelPrune: antidiagonal GPU component diagnostic

The author’s anchored `pred_2d` recurrence depends only on left, upper and upper-left reconstructed cells. The local candidate processes one antidiagonal at a time, keeping that dependency order. This is an implementation optimization of the pinned PixelPrune selector, not a new pruning algorithm.

Four deliberately selected DEV images were run at threshold 0.45 on one A40. For all four, the candidate, pinned author GPU implementation, and the previously validated CPU reference returned identical **complete ordered indices**. Their K values were 568, 499, 500, and 1. An earlier separate CPU study matched complete order on 32 images × three thresholds, plus synthetic boundary cases. Neither finite check proves equality for all inputs or other devices.

| Component | Four individual synchronized host-wall calls, ms | Median, ms |
|---|---|---:|
| Author anchored selector | 196.864, 191.831, 193.202, 184.419 | 192.516 |
| Antidiagonal candidate | 30.638, 29.687, 29.971, 29.848 | 29.910 |

Each implementation got one tiny 3×3 warmup and one measured call on each full 32×32 input. Within each image, measurement order alternated. The timer synchronizes immediately before and after each selector; input normalization, host-to-device transfer and the entire model are outside it. The author’s internal `.item()` synchronizations are included. The worker persisted the elapsed value, not the two raw clock timestamps, so the elapsed arithmetic cannot be independently reconstructed. No larger warmup or repeated-call stability study was performed. Host-process snapshots exist only before and after the run; other services were active and their precise overlap is unknown. These descriptive measurements are neither population estimates nor model TTFT or total-memory results.

Raw results and all ordered indices: `run_smoke001/observations/complete.json` and `progress.jsonl`. The source/CPU/no-GPU preflight review is `independent_prelaunch_review.json`; result/exit/source verification by a reviewer independent of the GPU worker author is `../../e1_independent_review/pixel_wavefront_gpu_result_review.json`; that reviewer also authored the antidiagonal candidate. Its supervised review and the GPU supervisor both exited completed/0; the device and container were subsequently observed idle. Full-model output/DeepStack/position/KV identity and natural TTFT remain to be tested before this candidate can support an end-to-end claim.
