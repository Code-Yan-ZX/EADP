# R-res versus EADP: descriptive image geometry on DEV

All 384 accepted input images were re-hashed against the accepted E1 manifest and R-res design; width and height were read from the same bytes. Original encoded dimensions are used, without EXIF rotation. Existing independently reviewed per-question scores were joined by task, qid and image group; no additional ground truth was opened.

| Task | Group | Questions | R-res | EADP | R-res − EADP (pp) |
|---|---|---:|---:|---:|---:|
| DocVQA_VAL | all | 128 | 80.98 | 67.85 | 13.14 |
| DocVQA_VAL | aspect: portrait | 105 | 80.55 | 65.06 | 15.49 |
| DocVQA_VAL | aspect: middle | 7 | 82.14 | 73.21 | 8.93 |
| DocVQA_VAL | aspect: landscape | 16 | 83.33 | 83.76 | -0.43 |
| DocVQA_VAL | area: small | 4 | 88.75 | 93.75 | -5.00 |
| DocVQA_VAL | area: large | 124 | 80.73 | 67.01 | 13.72 |
| TextVQA_VAL | all | 128 | 72.81 | 80.78 | -7.97 |
| TextVQA_VAL | aspect: portrait | 26 | 80.00 | 83.46 | -3.46 |
| TextVQA_VAL | aspect: middle | 14 | 56.43 | 75.00 | -18.57 |
| TextVQA_VAL | aspect: landscape | 88 | 73.30 | 80.91 | -7.61 |
| TextVQA_VAL | area: small | 118 | 74.83 | 81.27 | -6.44 |
| TextVQA_VAL | area: large | 10 | 49.00 | 75.00 | -26.00 |
| AI2D_TEST | all | 128 | 82.03 | 82.03 | 0.00 |
| AI2D_TEST | aspect: portrait | 11 | 90.91 | 90.91 | 0.00 |
| AI2D_TEST | aspect: middle | 50 | 82.00 | 84.00 | -2.00 |
| AI2D_TEST | aspect: landscape | 67 | 80.60 | 79.10 | 1.49 |
| AI2D_TEST | area: small | 112 | 82.14 | 82.14 | 0.00 |
| AI2D_TEST | area: large | 16 | 81.25 | 81.25 | 0.00 |

Fixed bins: portrait w/h < 0.8; middle 0.8 ≤ w/h ≤ 1.25; landscape > 1.25. Small area < 1,000,000 pixels; large ≥ 1,000,000. The two groupings overlap and must not be added together.

This is a posthoc descriptive analysis after the overall DEV results were known. The groups were fixed before extracting individual dimensions or joining scores. Multiple subgroups and task differences are considered, some cells are small, and image geometry can be confounded with content and task. There is no statistical significance claim, routing rule or causal conclusion. R-res uses a 512px input with 256 visual tokens; EADP starts from 1024 visual tokens and selects 256, so this is not equal compute. Six R-res generation kwargs and EOS were checked online, but the full effective runtime configuration was not separately archived.

First run processed individual records and wrote rows but failed to write the result because of a variable-shadowing bug. Version 2 only repairs that output bug and adds a full 384-row synthetic main regression; group rules were unchanged, and no individual values were inspected before that repair. The first attempt remains archived.

Independent recomputation: `e1_independent_review/rres_eadp_image_geometry_v2_review.json`; `monitor_rres_eadp_image_geometry_v2` completed/0. Source and result: `plan001.json`, `result001.json`, `rows001.jsonl`, `monitor_run001`.
