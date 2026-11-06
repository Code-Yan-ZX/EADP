# Exploratory 0.1 and 0.2 selected-index overlap

For all 384 matched DEV images, we compared the actual selected indices from the formally accepted runs. The analysis plan and source were frozen before reading these per-image indices, after observing earlier DEV scores. The independent recomputation is `../../e1_independent_review/pixel_t010_t020_selection_review.json`. The mixed per-image observation files contain predictions, but the analysis used only input identity and selected indices; it did not read ground truth or score predictions.

| Task | Images | 0.2 set is not a subset of 0.1 | Images where K(0.1) < K(0.2) | Median Jaccard |
| --- | ---: | ---: | ---: | ---: |
| DocVQA | 128 | 63 | 0 | 0.951 |
| TextVQA | 128 | 113 | 0 | 0.839 |
| AI2D | 128 | 65 | 1 | 0.974 |

For the one AI2D image, K at 0.1 is two tokens lower. The observed selections at these thresholds are therefore not generally nested. This descriptive result does not establish a cause, quality effect, latency, or generalization to other images.
