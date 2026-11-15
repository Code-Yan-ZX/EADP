# Semantic-difference mining: feasibility pilot results

2026-11-08 (Asia/Shanghai).

**Blind-spot mining produced verified failures, but the tested training pipeline failed to generalize.** Every completed control arm (6 arms) scored zero both-correct held-out pairs; the B2 baseline scored 10/32. This implementation is not a viable method on the tested held-out pool. The all-zero primary outcomes cannot rank the mining algorithms or establish equivalence, and do not reject the broader proposal. All trained arms also scored zero on the untouched real-question pool.

The full model verified 101/120 attempted pairs. Within those teacher-valid pairs, B2 failed on 62/101 and returned the same normalized answer on 18/101. This establishes the presence of candidate compression-specific training signals; training and transfer are assessed separately below.

The [preregistered protocol](cf_mining_pilot_prereg.md) defines the gates and controls. The [primary-source related-work review](cf_mining_prior_art.md) bounds the novelty claim; this pilot does not establish novelty.

## Setup and accounting

Model: `Qwen3-VL-8B-Instruct-1024`; 1024 square input, 1024 → 256 visual tokens, native DeepStack and 3-D mRoPE. Edits change one numeric digit on a real page; both members have the same local rerendering. Untouched pages remain a separate real-document evaluation.

CPU candidate construction scanned 600 numeric questions on 396 pages and produced 180 pairs. Model screening attempted the 120 pairs shown below. Unscreened candidates are not counted as model-verified failures.

| Split | Attempted | Teacher valid | Teacher invalid | Broad pair failure / valid | Strict collapse / valid | Student both correct / valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fit | 68 | 55 | 13 | 29/55 | 8/55 | 26/55 |
| validation | 35 | 30 | 5 | 22/30 | 8/30 | 8/30 |
| test | 17 | 16 | 1 | 11/16 | 2/16 | 5/16 |
| all | 120 | 101 | 19 | 62/101 | 18/101 | 39/101 |

Broad failure means at least one incorrect member. Strict collapse is the subset with the same normalized student output. Only pairs with correct full-model answers on both members enter these failure denominators. Correctness uses exact numeric value normalization with percent signs preserved; changing the answer is not sufficient to count as correct.

## Three observed strict collapses

| Split / pair | Question | Correct original → edit | Full teacher original → edit | Compressed original → edit |
| --- | --- | --- | --- | --- |
| test / docvqa_16999_389 | What is the Budget Estimate for Pharmaceutical Compendia Surveillance? | $100,000 → 100100 | $100,000 → $100,100 | $100,000 → $100,000 |
| test / docvqa_21126_2914 | Whats the amount in ESTIMATED UNOBLIGATED BALANCE FOR Personnel? | 359.62 → 359.32 | 359.62 → 359.32 | 359 → 359 |
| validation / docvqa_1022_582 | What is the serial number of ACTOS? | 501 → 571 | 501 → 571 | 490 → 490 |

## Untouched real pages

| Split | Questions / source documents | Full teacher accuracy | B2 accuracy |
| --- | ---: | ---: | ---: |
| test | 9 / 9 | 9/9 (100.0%) | 4/9 (44.4%) |
| validation | 15 / 14 | 15/15 (100.0%) | 8/15 (53.3%) |
| all | 24 / 23 | 24/24 (100.0%) | 12/24 (50.0%) |

These denominators include teacher errors; they are not restricted to teacher-valid edited pairs.

## All carrier attempts

Every discovered completed carrier gate and the active attempt are retained below. The [preregistered amendments](cf_mining_pilot_prereg.md#trainability-amendment-after-the-first-carrier-failed) first add separate zero-initialized low-rank value/output readouts to the four visual streams. The final amendment combines a fixed Gaussian local-attention bias (sigma 1.5 merged-token units) with a weight-one sequence margin `softplus(1 - log p(correct|image) + log p(opposite answer|image))`. These changes follow failed fit gates; held-out performance did not select them. Because locality and the margin changed together, this screen cannot attribute any recovery to either alone. The carriers are experimental training interfaces, not architectural novelty claims.

| Carrier / attempt | Recorded trainable parameters | Logged / requested updates | Fit both correct before → after | Strict collapse after | Gate verdict | Untouched QA before → after |
| --- | ---: | ---: | --- | ---: | --- | --- |
| pool / previous | not recorded | 120 / 120 | 0/8 → 0/8 | 3/8 | UNRESOLVED_CARRIER_OR_OPTIMIZATION | 50.0% → 58.3% |
| readout / previous | not recorded | 400 / 400 | 0/8 → 0/8 | 8/8 | UNRESOLVED_CARRIER_OR_OPTIMIZATION | 50.0% → 0.0% |
| local_readout / active | 1310721 | 400 / 400 | 0/8 → 1/8 | 7/8 | PASS | 50.0% → 0.0% |

The active carrier records rank 32, 400 requested gate updates and refresh interval 50. The gate pair IDs match the archived attempt: `True`. Comparison-arm budgets are reported from the actual final logs below.

The archived pool gate made 0/8 fit pairs both correct before and 0/8 after 120 logged updates; 3/8 pairs collapsed afterward (last recorded loss 10.825870037078857).

The archived readout gate made 0/8 fit pairs both correct before and 0/8 after 400 logged updates; 8/8 pairs collapsed afterward (last recorded loss 0.28007420897483826).

Fewer identical outputs or falling average loss do not count as recovery: both answers must be correct. A failed carrier gate does not by itself reject the broader semantic-difference mining idea. No training-control comparison is run unless the final carrier passes its fit gate.

## Carrier training gate

Observed verdict: **PASS**. Fit both-correct pairs changed from 0/8 to 1/8; the selected gate contained 5 strict collapses and 3 other pair failures.

Held-out paired change: -31.2 pp [95% CI -50.0, -16.1]; 32 questions / 25 documents. Untouched-page paired change: -50.0 pp [95% CI -70.8, -29.2]; 24 questions / 23 documents.

Frozen model version audit: `True`; sampled decoder checksum unchanged: `True`. Observed logged optimizer updates: 400.

The gate tests whether this carrier and optimizer can repair fit failures. A pass does not establish generalization; an unresolved gate does not falsify the general mining proposal.

## Implementation audit

Two launches stopped before any optimizer update: a float32 checksum-index interpolation exceeded a large embedding bound, and a VLMEvalKit import disabled global autograd. Exact integer index interpolation and an explicit training autograd context fixed these issues. These are infrastructure failures before training, not failed method experiments. Mining predictions and the experimental protocol were retained.

The completed active gate records native zero-gate checks `{"first_argmax_match": true, "first_logit_max_absolute_delta": 0.0625, "first_logit_mean_absolute_delta": 1.5385827282443643e-05, "forced_first_token": 16, "frozen_decoder_sample_sha256": "848284bd41b69e6a4417fa841907f481b674f009154376bfbcc5b7b8935616fa", "native_first_token": 16, "pair_id": "docvqa_51771_1743", "zero_gate_deepstack_bitwise": true, "zero_gate_main_bitwise": true}`; 400/400 logged updates had nonzero compressor gradients. Frozen-weight and checksum checks are reported with the gate above.

## Post-hoc training-generation diagnostic

After the comparison, a diagnostic checked the first eight fit original images. These are individual training answers, not held-out cases or both-correct edit pairs. The diagnostic did not choose a model or checkpoint.

| Arm | Correct training originals | Teacher-forced answer NLL range |
| --- | ---: | ---: |
| ordinary | 8/8 | 1.19209e-06–0.0360823 |
| online | 0/8 | 0.635381–3.52977 |

Ordinary training can generate its memorized fit answers with low teacher-forced NLL. This rules out an across-the-board inability of the interface to turn learned fit answers into generation, while leaving other implementation limits possible. The observed held-out collapse is consistent with severe memorization/generalization failure. Adapter state restored after the diagnostic: `True`.

## Matched training controls

The controls use the recorded carrier, optimizer-update budget and positive-answer exposure budget. Ordinary uses original values, random uses teacher-valid edited pairs, fixed uses initial failures, online refreshes fit examples using the current compressor, and replay shuffles the same online exposure multiset. Final checkpoints are evaluated without held-out checkpoint selection.

Observed audits: shared initialization hash = `True`; equal logged update counts = `True`; exact online-data exposure multisets = `{'replay': True, 'replay_ce': True}`.

| Arm | Held-out both correct | Pair accuracy | Change from B2, document-paired 95% CI | Untouched QA accuracy | Untouched change from B2 | Logged updates |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| B2 baseline | 10/32 | 31.2% | not available | 50.0% | not available | — |
| ordinary | 0/32 | 0.0% | -31.2 pp [95% CI -50.0, -16.1]; 32 questions / 25 documents | 0.0% | -50.0 pp [95% CI -70.8, -29.2]; 24 questions / 23 documents | 400 |
| random | 0/32 | 0.0% | -31.2 pp [95% CI -50.0, -16.1]; 32 questions / 25 documents | 0.0% | -50.0 pp [95% CI -70.8, -29.2]; 24 questions / 23 documents | 400 |
| fixed | 0/32 | 0.0% | -31.2 pp [95% CI -50.0, -16.1]; 32 questions / 25 documents | 0.0% | -50.0 pp [95% CI -70.8, -29.2]; 24 questions / 23 documents | 400 |
| online | 0/32 | 0.0% | -31.2 pp [95% CI -50.0, -16.1]; 32 questions / 25 documents | 0.0% | -50.0 pp [95% CI -70.8, -29.2]; 24 questions / 23 documents | 400 |
| replay | 0/32 | 0.0% | -31.2 pp [95% CI -50.0, -16.1]; 32 questions / 25 documents | 0.0% | -50.0 pp [95% CI -70.8, -29.2]; 24 questions / 23 documents | 400 |
| replay_ce | 0/32 | 0.0% | -31.2 pp [95% CI -50.0, -16.1]; 32 questions / 25 documents | 0.0% | -50.0 pp [95% CI -70.8, -29.2]; 24 questions / 23 documents | 400 |

| Online contrast | Held-out both-correct change | Untouched QA change |
| --- | --- | --- |
| online_minus_ordinary | +0.0 pp [95% CI +0.0, +0.0]; 32 questions / 25 documents | +0.0 pp [95% CI +0.0, +0.0]; 24 questions / 23 documents |
| online_minus_random | +0.0 pp [95% CI +0.0, +0.0]; 32 questions / 25 documents | +0.0 pp [95% CI +0.0, +0.0]; 24 questions / 23 documents |
| online_minus_fixed | +0.0 pp [95% CI +0.0, +0.0]; 32 questions / 25 documents | +0.0 pp [95% CI +0.0, +0.0]; 24 questions / 23 documents |
| online_minus_replay | +0.0 pp [95% CI +0.0, +0.0]; 32 questions / 25 documents | +0.0 pp [95% CI +0.0, +0.0]; 24 questions / 23 documents |
| online_minus_replay_ce | +0.0 pp [95% CI +0.0, +0.0]; 32 questions / 25 documents | +0.0 pp [95% CI +0.0, +0.0]; 24 questions / 23 documents |

| Arm | Actual updates | Positive-answer forwards | Negative-answer forwards | Mining-answer forwards | Training wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| ordinary | 400 | 800 | 0 | 0 | 164.3761248588562 |
| random | 400 | 800 | 800 | 0 | 314.92105650901794 |
| fixed | 400 | 800 | 800 | 0 | 310.3873541355133 |
| online | 400 | 800 | 800 | 448 | 434.14910078048706 |
| replay | 400 | 800 | 800 | 0 | 316.19846057891846 |
| replay_ce | 400 | 800 | 0 | 0 | 166.95544123649597 |

Ordinary and `replay_ce` use CE plus bucket KL. Random, fixed, online, and replay additionally use the same sequence-margin objective. `replay_ce` uses the exact online positive-data multiset with CE plus KL only. Contrast arms perform an additional negative-answer forward per positive exposure. Equal optimizer steps and positive exposures therefore do **not** imply equal total computation. The ordinary contrast confounds edited data, loss, and extra forwards; the replay_ce contrast also includes a loss/compute difference. Random/fixed/online/replay share the loss, while online adds refreshed search cost.

The table separates observed training and mining forwards. One initial current-model hard-set search is shared by fixed and online; its recorded plan is `{"candidate_pairs": 32, "extra_online_search_steps": [50, 100, 150, 200, 250, 300, 350], "pair_ids": ["docvqa_47460_4140", "docvqa_3077_1986", "docvqa_51771_1743", "docvqa_64487_4576", "docvqa_1959_1198", "docvqa_44965_3133", "docvqa_2209_1412", "docvqa_65065_4999"], "search_step": 0, "shared_by": ["fixed", "online"]}`. Per-arm objective-name unions may include previously executed arms; negative-forward counts identify actual contrast computation.

### What the current-model search finds

Observed search rounds: 1 initial round shared by fixed/online and 7 online refresh rounds (8 total). The same fit pairs are joined to their B2-initialization outcomes. Baseline-correct → current-failure counts training breakage; baseline-failure → current-both-correct counts rescue. Repeated screening rounds are not independent new examples.

| Search phase / step | Scored / labelled candidates | Current both correct | Current failures | Current strict collapse | New failure / baseline correct | Rescue / baseline failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| initial_shared_by_fixed_online / 0 | 32 / 32 | 18 | 14 | 5 | 0/18 | 0/14 |
| online_refresh / 50 | 32 / 32 | 0 | 32 | 23 | 18/18 | 0/14 |
| online_refresh / 100 | 32 / 32 | 0 | 32 | 28 | 18/18 | 0/14 |
| online_refresh / 150 | 32 / 32 | 0 | 32 | 26 | 18/18 | 0/14 |
| online_refresh / 200 | 32 / 32 | 0 | 32 | 22 | 18/18 | 0/14 |
| online_refresh / 250 | 32 / 32 | 0 | 32 | 23 | 18/18 | 0/14 |
| online_refresh / 300 | 32 / 32 | 0 | 32 | 28 | 18/18 | 0/14 |
| online_refresh / 350 | 32 / 32 | 0 | 32 | 28 | 18/18 | 0/14 |

The refreshed candidate failures include pairs that the initial compressed model answered correctly. Thus the search is partly reacting to training-induced breakage. A plausible interpretation is that the refreshed hard pool can track damage caused by the updating carrier; these logs alone do not establish the cause of the generalization failure.


Intervals resample whole source documents and preserve paired question outcomes. A confidence interval crossing zero does not resolve an advantage. Online-versus-replay differences address ordering/refresh, while similar replay performance can attribute benefit to selected data. Any relation-loss comparison is subject to the extra-forward compute difference.

Every arm is at the primary accuracy floor. Degenerate paired intervals such as `[0, 0]` describe identical observed zero outcomes; they do not demonstrate algorithmic equivalence or a sufficiently sensitive comparison. The carrier/generalization failure takes precedence over any ranking among these mining schedules.

## Same-source quality references

Same-pair Rres512 status: **complete**. The 32 held-out pairs (25 source documents) were selected by teacher validity before the resolution-reference predictions, with no filtering by compressed-model success.

| Same held-out pairs | Both correct / pairs | Pair accuracy | Strict answer collapse / pairs |
| --- | ---: | ---: | ---: |
| teacher | 32/32 | 100.0% | — |
| b2 | 10/32 | 31.2% | 6/32 |
| rres512 | 17/32 | 53.1% | 3/32 |

Rres512 minus B2 in both-correct accuracy: +21.9 pp [95% CI +0.0, +45.5]; 32 questions / 25 documents; rescued 10, broke 3, and repaired 4 B2 strict collapses to two correct answers. Observed per-member input/decoder token counts: [256]/[256].

This reference tests whether the same answer-critical differences remain difficult for an alternative compression baseline. Failures mined from B2 alone do not establish a weakness shared by stronger compressors.

| Trained arm minus Rres512 | Paired both-correct change on matching pairs |
| --- | --- |
| ordinary | -53.1 pp [95% CI -75.0, -35.0]; 32 questions / 25 documents |
| random | -53.1 pp [95% CI -75.0, -35.0]; 32 questions / 25 documents |
| fixed | -53.1 pp [95% CI -75.0, -35.0]; 32 questions / 25 documents |
| online | -53.1 pp [95% CI -75.0, -35.0]; 32 questions / 25 documents |
| replay | -53.1 pp [95% CI -75.0, -35.0]; 32 questions / 25 documents |
| replay_ce | -53.1 pp [95% CI -75.0, -35.0]; 32 questions / 25 documents |

Rres512 status: **complete**. This keeps all 256 visual tokens produced at 512×512, whereas B2 starts from 1024 tokens and keeps 256. Both use native DeepStack and 3-D positions. These are accuracy references; no end-to-end latency claim follows.

| Same untouched source pool | Correct / questions | Accuracy |
| --- | ---: | ---: |
| teacher | 24/24 | 100.0% |
| b2 | 12/24 | 50.0% |
| rres512 | 19/24 | 79.2% |

Rres512 minus B2: +29.2 pp [95% CI +8.3, +52.2]; 24 questions / 23 documents; rescued 8 and broke 1 questions. Observed input/decoder token counts: [256]/[256]; resolution restored to 1024.

| Trained arm minus Rres512 | Paired accuracy change on matching questions |
| --- | --- |
| ordinary | -79.2 pp [95% CI -95.7, -59.3]; 24 questions / 23 documents |
| random | -79.2 pp [95% CI -95.7, -59.3]; 24 questions / 23 documents |
| fixed | -79.2 pp [95% CI -95.7, -59.3]; 24 questions / 23 documents |
| online | -79.2 pp [95% CI -95.7, -59.3]; 24 questions / 23 documents |
| replay | -79.2 pp [95% CI -95.7, -59.3]; 24 questions / 23 documents |
| replay_ce | -79.2 pp [95% CI -95.7, -59.3]; 24 questions / 23 documents |

Unrelated sibling-question check: **complete**; 8/8 attempted pairs passed full-teacher verification on both images (0 rejected). These are invariance checks, not field-swap or arithmetic interventions.

| Teacher-verified invariant pairs | Both correct / accepted | Accuracy |
| --- | ---: | ---: |
| teacher | 8/8 | 100.0% |
| b2 | 3/8 | 37.5% |
| online | 0/8 | 0.0% |

Online minus B2: -37.5 pp [95% CI -75.0, -12.5]; 8 questions / 8 documents; rescued 0, broke 3; adapter state restored after the check: `True`. These 8 sibling-question pairs were evaluated only; unrelated-question invariance constraints were not trained.

## Secondary DocVQA ANLS

Exact numeric correctness remains primary: a one-digit error fails even when the text is nearly identical. Secondary ANLS uses the repository's official `anls_compute` and `hit_calculate` functions against all available original/edited aliases, with their own case/whitespace/string rules and no Decimal normalization. `anls_compute` returns normalized edit distance; `hit_calculate` converts the best alias to similarity and applies the official threshold. For example, `1997` versus `1996` has distance 0.25 and ANLS 0.75, while primary exact correctness is zero. The scores below average individual answers, not both-correct pairs, and are not full-benchmark DocVQA results.

| Evaluation | Model / arm | Scored individual answers | Secondary ANLS (0–100) |
| --- | --- | ---: | ---: |
| Teacher-valid mined pair members | teacher | 202 | 99.49 |
| Teacher-valid mined pair members | student | 202 | 64.19 |
| Untouched real questions | teacher | 24 | 98.58 |
| Untouched real questions | b2 | 24 | 56.77 |
| pool fit members before | student | 16 | 28.75 |
| pool fit members after | student | 16 | 25.62 |
| readout fit members before | student | 16 | 28.75 |
| readout fit members after | student | 16 | 77.45 |
| local_readout fit members before | student | 16 | 28.75 |
| local_readout fit members after | student | 16 | 82.45 |
| Fixed held-out pair members | B2 baseline | 64 | 58.96 |
| Fixed held-out pair members | ordinary | 64 | 8.98 |
| Untouched real questions | ordinary | 24 | 18.65 |
| Fixed held-out pair members | random | 64 | 8.59 |
| Untouched real questions | random | 24 | 14.48 |
| Fixed held-out pair members | fixed | 64 | 12.19 |
| Untouched real questions | fixed | 24 | 19.24 |
| Fixed held-out pair members | online | 64 | 8.99 |
| Untouched real questions | online | 24 | 13.96 |
| Fixed held-out pair members | replay | 64 | 9.78 |
| Untouched real questions | replay | 24 | 6.25 |
| Fixed held-out pair members | replay_ce | 64 | 11.23 |
| Untouched real questions | replay_ce | 24 | 18.33 |
| Reference untouched questions | teacher | 24 | 98.58 |
| Reference untouched questions | b2 | 24 | 56.77 |
| Reference untouched questions | rres512 | 24 | 88.06 |
| Reference pair members | teacher | 64 | 99.61 |
| Reference pair members | b2 | 64 | 58.96 |
| Reference pair members | rres512 | 64 | 84.21 |
| Teacher-verified invariant members | teacher | 16 | 100.00 |
| Teacher-verified invariant members | b2 | 16 | 51.99 |
| Teacher-verified invariant members | online | 16 | 7.50 |

Official-function source SHA-256: `f63c9fd539a4a2750cfb4d17ca654e9871b58358bbb943549a71c947839de3ae`. The analysis loads only those pure function definitions, avoiding model/package imports. ANLS partial credit must not be interpreted as preservation of the answer-critical digit distinction.

## Limits and next decision

The original mining-stage invariant check was not run; supplementary checks are reported separately above.

- DocVQA VAL custom pilot partitions; not an official benchmark train/test result.
- Image/document-disjoint partitions do not establish unseen-layout transfer.
- 3 carrier attempts were screened; the six-arm comparison uses the final carrier and one seed. This does not cover the full method-design space.
- Only 32 teacher-valid fit pairs trained the comparison carrier; this tiny adaptation pool is insufficient to establish a strong pretrained compressor.
- Unrelated-question invariance constraints were not trained; sibling-question pairs were evaluated only, with the actual count reported separately.
- No field swaps, arithmetic/sum interventions, or unseen-layout evaluation were implemented. The full proposed method was not tested.
- Data-selection contrasts require the same objective: inspect each arm's positive/negative forwards and loss settings.
- Locality and the answer-relation objective changed together in the final carrier screen; any recovery cannot be attributed to either alone.

Continue only on evidence from real untouched QA, matched controls and additional sources/layouts. Strict-collapse yield alone supports mining availability, not publication value or a final architecture.

Machine-readable accounting and all observed denominators are in `Qwen_vl/outputs/cf_mining/report/summary.json`; the compact table figure is `overview.png` alongside it.
