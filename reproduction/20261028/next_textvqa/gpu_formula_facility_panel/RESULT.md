# NeXT native AnyRes formula diagnostic — complete

All16 images and320 planned cases were measured; ragged cases0. Independent CPU verification passed16 native source anchors,80 identity controls, all1744 finite arrays, native/internal/effective ID and final-mask consistency, the frozen source/reference/implementation guards,20 model-stat guards,16 image hashes and root additional source guards. Worker2633433/startticks1332520592 actually exited; root exec34131 returned0. No answers, GT labels or benchmark scores were used.

| Existing complete recipe | Eq6 changed importance | Eq6 changed crop quotas | Eq6 changed effective greedy order | Eq6 changed actual final mask | Eq10 changed actual final mask |
|---|---:|---:|---:|---:|---:|
| K128_alpha05_beta1 | 6/16 | 0/16 | 1/16 | 0/16 | 0/16 |
| K32_alpha05_beta1 | 6/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| K128_alpha05_beta2 | 6/16 | 0/16 | 1/16 | 1/16 | 0/16 |
| K64_alpha05_beta2 | 6/16 | 0/16 | 1/16 | 0/16 | 0/16 |
| K32_alpha05_beta2 | 6/16 | 0/16 | 1/16 | 0/16 | 0/16 |

The single actual final-mask change is panel image11, original full-question position3632, `b60dc6dee9c060de.jpg`, Eq6 at CLI128/alpha.5/beta2. Zero-based crop3 replaces visual position330 with115. Native and variant crop quotas remain `[146,162,106,130,93]`, total637. Crop3 is the fourth crop, corresponding to the bottom-left local crop under the fixed2×2 native spatial layout. These are crop-local24×24-grid positions, not final LLM sequence offsets.

The first effective greedy divergence occurs at zero-based step15 (the16th choice) with the entire first15 choices identical. Source gains are144=.35757192969322205 and142=.3574090898036957, so it selects144. Eq6 gains are144=.35707080364227295 and142=.3573876619338989, so it selects142. Later coverage updates reconcile most differences; the final valid set differs only by330→115. The source advantage of144 is0.0001628398895263672; the Eq6 advantage of142 is0.00031685829162597656.

This image has3 real CLIP segments, dense shape5×576×3×77. In crop3, Eq6 adds original text position55 in segment0,19 in segment1 and64 in segment2, preserving every original top15 order; none is an exact threshold tie. Those membership additions were changed together. The recorded gains identify the combined formula-to-selection effect, not one uniquely causal segment. Exact members/thresholds are saved in `causal_selection_case.json`; no masking or new q/temperature was introduced.

No image changes any quota under either formula. Thus the proposed quota-rounding mediator did not trigger on this panel; the positive case follows membership→importance→marginal-gain ordering→a different final visual position. Other Eq6 greedy-order changes return the same final mask. Eq10 changes16/16 importance maps for every recipe but no effective prefix or final mask; one internal max_T sequence changes only beyond a crop quota and is discarded. Native spatial input reconstruction uses the mask, so neither a discarded-suffix change nor merely reordered greedy choices with the same mask is called a model-input change.

This is one real selection sensitivity among16 fixed images under one existing recipe. It does not establish how often full5000 inputs change, whether any answer changes, an accuracy gain, the cause of the paper gap, or the unpublished main-table argv. All source/baseline/negative results remain preserved; no full-accuracy variant was scheduled by this diagnostic.

Evidence: `execution/summary.json`,16 `execution/image_XX.npz`, `execution/per_image.jsonl`, `independent_verification.json`, `root_terminal_process_proof.json`.
