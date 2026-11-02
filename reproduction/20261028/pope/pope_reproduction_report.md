All8910canonicalquestions were generated and scored with the unchanged author evaluator and exact source-pinned labels. No partial-run scores or higher-result selection were used.

| Run | POPE mean F1 (%) | Evidence boundary |
|---|---:|---|
| Paper Table1 | 86.700000 | reported |
| Historical alpha.5/beta2, wait | 84.073600 | preserved |
| Fresh alpha1/beta1, source as-is | 77.393686 | 231/2334 same-input conflicts |
| Fresh alpha1/beta1, wait only | 82.220237 | 0/2334 same-input conflicts |

The current released task shell sets alpha1/beta1; generic loader/README example setsalpha.5/beta1. The paper does not publish complete main-table argv. Both fresh outcomes are retained; a close macro alone is insufficient if deterministic same-input consistency fails.

Model/tokenizer/preprocessor payloads match17officialHFfiles. All500image RGB/FP32/FP16 transforms and8910full LLM/CLIP prompt IDs match across tested current/author package pins. Full raw predictions, stdout, source hashes, exact runtime patch proof and per-category TP/FP/TN/FN remain beside this report.
