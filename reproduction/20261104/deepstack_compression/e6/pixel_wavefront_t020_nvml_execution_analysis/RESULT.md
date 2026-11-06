# Pixel nine-process sampled memory pilot

Nine isolated model processes cover three previously selected DEV anchors and three arms per anchor. `K` is the actual selected token count; runs have different workloads. Every process completed and all nine are reported. This is descriptive memory evidence, not TTFT, equal-compute comparison, or a population conclusion.

For the phase rows, `process` is the maximum known sampled target-process NVML usage; `device` is the maximum known sampled whole-device usage. Both are MiB, accompanied by known/unknown observation counts. A target absent from NVML is unknown, not zero. Observations whose query windows overlap a phase can extend beyond that phase.

| Task | Arm | K | Phase | Overlap samples | Process max MiB (known/unknown) | Target absent | Device used max MiB (known/unknown) | Graphics PIDs observed |
|---|---|---:|---|---:|---:|---:|---:|---:|
| DocVQA_VAL | t020_wavefront | 598 | pre_cuda | 1 | unknown (0/1) | 1 | 1169.00 (1/0) | 10 |
| DocVQA_VAL | t020_wavefront | 598 | model_load | 33 | 17002.00 (32/1) | 1 | 18176.00 (33/0) | 10 |
| DocVQA_VAL | t020_wavefront | 598 | request | 4 | 17270.00 (4/0) | 0 | 18444.00 (4/0) | 10 |
| DocVQA_VAL | t020_wavefront | 598 | teardown | 1 | 5170.00 (1/0) | 0 | 6604.00 (1/0) | 10 |
| TextVQA_VAL | t0_author | 770 | pre_cuda | 1 | unknown (0/1) | 1 | 1169.00 (1/0) | 10 |
| TextVQA_VAL | t0_author | 770 | model_load | 9 | 17002.00 (8/1) | 1 | 18176.00 (9/0) | 10 |
| TextVQA_VAL | t0_author | 770 | request | 3 | 17156.00 (3/0) | 0 | 18466.00 (3/0) | 10 |
| TextVQA_VAL | t0_author | 770 | teardown | 1 | 17032.00 (1/0) | 0 | 18466.00 (1/0) | 10 |
| AI2D_TEST | t020_wavefront | 549 | pre_cuda | 1 | unknown (0/1) | 1 | 1169.00 (1/0) | 10 |
| AI2D_TEST | t020_wavefront | 549 | model_load | 9 | 17002.00 (8/1) | 1 | 18176.00 (9/0) | 10 |
| AI2D_TEST | t020_wavefront | 549 | request | 3 | 17270.00 (3/0) | 0 | 18444.00 (3/0) | 10 |
| AI2D_TEST | t020_wavefront | 549 | teardown | 1 | 364.00 (1/0) | 0 | 1538.00 (1/0) | 10 |
| TextVQA_VAL | t020_author | 703 | pre_cuda | 1 | unknown (0/1) | 1 | 1169.00 (1/0) | 10 |
| TextVQA_VAL | t020_author | 703 | model_load | 9 | 17002.00 (8/1) | 1 | 18176.00 (9/0) | 10 |
| TextVQA_VAL | t020_author | 703 | request | 3 | 17312.00 (3/0) | 0 | 18486.00 (3/0) | 10 |
| TextVQA_VAL | t020_author | 703 | teardown | 1 | 344.00 (1/0) | 0 | 1518.00 (1/0) | 10 |
| DocVQA_VAL | t020_author | 598 | pre_cuda | 1 | unknown (0/1) | 1 | 1169.00 (1/0) | 10 |
| DocVQA_VAL | t020_author | 598 | model_load | 9 | 17002.00 (8/1) | 1 | 18176.00 (9/0) | 10 |
| DocVQA_VAL | t020_author | 598 | request | 4 | 17270.00 (4/0) | 0 | 18444.00 (4/0) | 10 |
| DocVQA_VAL | t020_author | 598 | teardown | 1 | 364.00 (1/0) | 0 | 1624.00 (1/0) | 10 |
| TextVQA_VAL | t020_wavefront | 703 | pre_cuda | 1 | unknown (0/1) | 1 | 1169.00 (1/0) | 10 |
| TextVQA_VAL | t020_wavefront | 703 | model_load | 9 | 17002.00 (8/1) | 1 | 18176.00 (9/0) | 10 |
| TextVQA_VAL | t020_wavefront | 703 | request | 3 | 17156.00 (3/0) | 0 | 18366.00 (3/0) | 10 |
| TextVQA_VAL | t020_wavefront | 703 | teardown | 1 | 16892.00 (1/0) | 0 | 18366.00 (1/0) | 10 |
| AI2D_TEST | t020_author | 549 | pre_cuda | 1 | unknown (0/1) | 1 | 1169.00 (1/0) | 10 |
| AI2D_TEST | t020_author | 549 | model_load | 9 | 17002.00 (8/1) | 1 | 18176.00 (9/0) | 10 |
| AI2D_TEST | t020_author | 549 | request | 3 | 17250.00 (3/0) | 0 | 18384.00 (3/0) | 10 |
| AI2D_TEST | t020_author | 549 | teardown | 1 | 1590.00 (1/0) | 0 | 2964.00 (1/0) | 10 |
| AI2D_TEST | t0_author | 692 | pre_cuda | 1 | unknown (0/1) | 1 | 1169.00 (1/0) | 10 |
| AI2D_TEST | t0_author | 692 | model_load | 10 | 17002.00 (9/1) | 1 | 18176.00 (10/0) | 10 |
| AI2D_TEST | t0_author | 692 | request | 3 | 17250.00 (3/0) | 0 | 18424.00 (3/0) | 10 |
| AI2D_TEST | t0_author | 692 | teardown | 1 | 7812.00 (1/0) | 0 | 9286.00 (1/0) | 10 |
| DocVQA_VAL | t0_author | 667 | pre_cuda | 1 | unknown (0/1) | 1 | 1169.00 (1/0) | 10 |
| DocVQA_VAL | t0_author | 667 | model_load | 9 | 17002.00 (8/1) | 1 | 18176.00 (9/0) | 10 |
| DocVQA_VAL | t0_author | 667 | request | 3 | 17292.00 (3/0) | 0 | 18466.00 (3/0) | 10 |
| DocVQA_VAL | t0_author | 667 | teardown | 1 | 9012.00 (1/0) | 0 | 10446.00 (1/0) | 10 |

PyTorch allocator values below are in MiB and are separate from both NVML columns. Model-load peaks are taken at load completion; request peaks follow a peak-stat reset before the request. Allocator measurements do not equal total process or device memory.

| Task | Arm | K | Total NVML samples | Load allocated peak | Load reserved peak | Request allocated peak | Request reserved peak |
|---|---|---:|---:|---:|---:|---:|---:|
| DocVQA_VAL | t020_wavefront | 598 | 42 | 16722.03 | 16742.00 | 16924.80 | 16946.00 |
| TextVQA_VAL | t0_author | 770 | 17 | 16722.03 | 16742.00 | 16972.80 | 17008.00 |
| AI2D_TEST | t020_wavefront | 549 | 17 | 16722.03 | 16742.00 | 16918.27 | 16946.00 |
| TextVQA_VAL | t020_author | 703 | 17 | 16722.03 | 16742.00 | 16953.91 | 16988.00 |
| DocVQA_VAL | t020_author | 598 | 19 | 16722.03 | 16742.00 | 16924.80 | 16946.00 |
| TextVQA_VAL | t020_wavefront | 703 | 17 | 16722.03 | 16742.00 | 16953.91 | 16988.00 |
| AI2D_TEST | t020_author | 549 | 18 | 16722.03 | 16742.00 | 16918.27 | 16946.00 |
| AI2D_TEST | t0_author | 692 | 18 | 16722.03 | 16742.00 | 16958.58 | 16988.00 |
| DocVQA_VAL | t0_author | 667 | 18 | 16722.03 | 16742.00 | 16944.25 | 16968.00 |

NVML observations are approximately every 0.5 seconds, rounded to MiB, and can miss short-lived peaks. Graphics activity was present in the recorded windows; unobserved background remains unknown. Instrumentation, CPU copies and synchronization affect memory. Only three post-hoc anchors were studied; full prior visual intermediates were not all archived and identity relies partly on online capture checks. The raw records retain per-window indices, exact query boundaries, process/device inventories and all unknown counts.

Complete machine-readable analysis: `analysis001.json` SHA256 `5308ea66f1c5c41ad3da79809f7838d085acd9e3577fc3c07ca751962aa59ca8`. Independent complete 36-phase recomputation: `../../e1_independent_review/pixel_t020_nvml_execution_analysis_review.json` SHA256 `d2f548da548279dabf72d22925c330840cea7790ab7c839f022ce502966be9a2`. Independent nine-process raw result audit: `../../e1_independent_review/pixel_t020_nvml_execution_result_review.json` SHA256 `27ba3c825eb216fe4b5cba0625680c606148be0cc47325ba0f51113c065f8a32`.
