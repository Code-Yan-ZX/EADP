# 32-image component diagnostic

The original frozen DEV schedule was run on all 32 images (DocVQA 11, TextVQA 11, AI2D 10): three warmups and 20 recorded CUDA-event measurements per component per image. For each image, calculate the median of its 20 measurements; the table shows the median and range of those 32 image-level medians. The image is the observational unit; the 1,920 calls are not 1,920 independent images.

| Component | Median (ms) | Image-level median range (ms) |
|---|---:|---:|
| Original selector | 38.367 | 38.203–39.168 |
| Three DeepStack mergers | 2.596 | 2.585–2.703 |
| Three-stream selection and mask staging | 0.183 | 0.153–0.285 |

Selector image-level medians by task were DocVQA 38.358 ms, TextVQA 38.368 ms and AI2D 38.387 ms. `result_001.json` contains all per-image medians, component ranges, task breakdown and source hashes. The full run passed formal acceptance and a separate independent review: 32 diagnostic captures and identity checks, restored hooks, 288 warmups and 1,920 output-checked measurements, with clean container and supervisor exits. The 32-image extension was decided after the single-image pilot was observed; it is exploratory even though the 32-image schedule itself was fixed previously.

These are isolated warm-cache component intervals, not end-to-end TTFT or an equal-compute comparison. The staging component excludes the decoder's actual DeepStack addition. Each output was fully compared with its reference at runtime, but per-measurement output tensors were not archived. One light Python file-stat operation occurred in monotonic interval `[13974385382685808, 13974385382692250]` ns; Python startup and other light-process overhead have no precise boundaries. Measurement start/stop monotonic timestamps were not recorded, so individual overlap and any impact cannot be established. All measurements remain included.

Evidence: `../e4_c_microbench_full/full_001_verification.json`, `../e1_independent_review/component_full001_review.json`, and `monitor_summary001/terminal.json`. The summary source is `summarize.py`; the summary computation was requested after the experimental run and before inspecting the group-level results. No new GT or scoring was used.
