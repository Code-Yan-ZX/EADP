# LayoutLite local-reward diagnostic

Verdict: **HOLD_INSUFFICIENT_INFORMATION**

Inference only. External target is literal coverage of original human DocVQA answers in OCR text, not actual QA accuracy, field binding, or a trained improvement.

Frozen primary pages: 32. OCR outputs: 192/192. Pending pages: 0.
Selected annotated numeric QA: 80; covered by full OCR: 72. Valid reward-ranking pages: 30.
Full references truncated: 2; compressed outputs truncated: 7; excluded pages: 2.

## Registered near-global-tie ranking

Contributing independent documents: 5; Q-discordant pairs within global-ratio difference <=0.01: 10. Gates: >=12 documents and >=24 pairs.

| Reward | Document-macro concordance | Tail minus reward [95% CI] |
|---|---:|---:|
| global_raw | 0.9000 | -0.4000 [-0.9000, 0.1000] |
| global_normalized | 0.8500 | -0.3500 [-0.8500, 0.1500] |
| span_mean | 0.8000 | -0.3000 [-0.8000, 0.2000] |
| span_length_weighted | 0.8500 | -0.3500 [-0.8500, 0.1500] |
| tail20 | 0.5000 | — |

## Coverage of fixed actions

| Action | Retained eligible answer evidence | All-targets-retained pages |
|---|---:|---:|
| topk | 68/70 | 28/30 |
| gumbel_2026110801 | 65/70 | 26/30 |
| gumbel_2026110802 | 67/70 | 27/30 |
| gumbel_2026110803 | 62/70 | 23/30 |
| gumbel_2026110804 | 64/70 | 26/30 |

All selected numeric QA (without teacher filtering): full 72/80; compressed topk 72/80, gumbel_2026110801 69/80, gumbel_2026110802 69/80, gumbel_2026110803 66/80, gumbel_2026110804 70/80.
All original source-page QA strings (unfiltered descriptive proxy): full 202/221; compressed topk 201/221, gumbel_2026110801 190/221, gumbel_2026110802 198/221, gumbel_2026110803 189/221, gumbel_2026110804 196/221.

## Reward-selected actions (same five actions, no training)

| Reward | Page-macro evidence coverage |
|---|---:|
| global_raw | 0.9833 |
| global_normalized | 0.9833 |
| span_mean | 0.9833 |
| span_length_weighted | 0.9833 |
| tail20 | 0.9667 |

## Fixed sensitivities

| Analysis | Documents / near-tie pairs | Tail minus span mean | Tail minus length mean |
|---|---:|---:|---:|
| width_32 | 5 / 10 | -0.1000 | -0.1500 |
| width_128 | 5 / 10 | -0.2500 | -0.4000 |
| reverse_alignment | 5 / 10 | -0.4000 | -0.4000 |
| all_untruncated | 4 / 9 | -0.1250 | -0.1875 |

## Limits

- Official released FireRed-OCR/score head; stock Transformers 4.57.6 execution adapted from upstream 5.6 code. All-token native interface gates are recorded separately.
- Exact per-page 50% masks probe equal-cost actions. This does not reproduce the original globally calibrated clustered allocation rule, nor the project's 8B/25%-retention setting.
- The panel is compact numeric DocVQA TRAIN pages, selected without compressed outcomes. Fresh source parents do not establish unseen-layout generalization.
- Full OCR can contain errors; literal answer presence can miss formatting changes or credit a value in the wrong field. Provided OCR is a silver reference, not human whole-page gold.
- No reward, seed, tie window or tail fraction was fitted to this panel. A ranking signal would still require controlled training and real-QA evaluation.
