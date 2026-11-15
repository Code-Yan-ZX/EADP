# VisionZip / PPE binding oracle analysis

Decision: **STOP_THIS_MECHANISM**. adequately sampled oracle has no net repair or fails to beat a simple control; does not refute binding preservation generally

Active geometry: `native_smart_resize_then_square_v2`. Historical invalidated answers: 300; currently excluded entries with incompatible/missing referenced diagnostics: 0.

Teacher 4/4 verification: 56 accepted, 6 rejected, 0 pending / 62 planned pages; 62 pages attempted. Accepted pages span 56 source documents.

Missing predictions remain pending. Primary correctness requires all four field questions on both image members.

| Arm | Status | 4/4 pages | Individual answers | Any side swaps two recognized values | Four-answer multiset complete | Untouched 2/2 pages |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| teacher | complete (56/56) | 56/56 | 224/224 | 0 | 56 | 21/22 |
| visionzip | complete (56/56) | 18/56 | 136/224 | 0 | 21 | 8/22 |
| visionzip_ppe | complete (56/56) | 18/56 | 136/224 | 0 | 21 | 8/22 |
| field_anchor_control_ppe | complete (56/56) | 22/56 | 149/224 | 0 | 24 | 11/22 |
| field_oracle | complete (56/56) | 22/56 | 157/224 | 0 | 24 | 10/22 |
| field_oracle_ppe | complete (56/56) | 22/56 | 156/224 | 0 | 24 | 10/22 |
| random_oracle_ppe | complete (56/56) | 18/56 | 135/224 | 0 | 21 | 9/22 |
| spatial_oracle_ppe | complete (56/56) | 17/56 | 131/224 | 0 | 20 | 9/22 |
| rres512 | complete (56/56) | 25/56 | 145/224 | 0 | 25 | 17/22 |
| historical | complete (11/11) | 4/11 | 27/44 | 0 | 4 | pending |
| historical_ds_fixed | complete (11/11) | 4/11 | 27/44 | 0 | 4 | pending |
| line_anchor_control_ppe | complete (50/50) | 19/50 | 134/200 | 0 | 20 | pending |
| line_oracle_ppe | complete (50/50) | 19/50 | 136/200 | 0 | 20 | pending |

| Paired contrast | Complete matched pages | Rescued / broken | Net repairs | Page accuracy difference | Document bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| restricted_aggregation_same_anchors (complete) | 56/56 | 0 / 0 | 0 | +0.0 pp | [+0.0, +0.0] pp |
| oracle_minus_visionzip_ppe (complete) | 56/56 | 4 / 0 | 4 | +7.1 pp | [+1.8, +14.3] pp |
| oracle_minus_random (complete) | 56/56 | 4 / 0 | 4 | +7.1 pp | [+1.8, +14.3] pp |
| oracle_minus_spatial (complete) | 56/56 | 5 / 0 | 5 | +8.9 pp | [+1.8, +17.9] pp |
| oracle_minus_rres512 (complete) | 56/56 | 7 / 10 | -3 | -5.4 pp | [-19.6, +8.9] pp |
| anchor_allocation_only (complete) | 56/56 | 4 / 0 | 4 | +7.1 pp | [+1.8, +14.3] pp |
| ordinary_ppe_only (complete) | 56/56 | 0 / 0 | 0 | +0.0 pp | [+0.0, +0.0] pp |
| oracle_ppe_only (complete) | 56/56 | 0 / 0 | 0 | +0.0 pp | [+0.0, +0.0] pp |
| historical_ds_repair (complete) | 11/11 | 0 / 0 | 0 | +0.0 pp | [+0.0, +0.0] pp |
| contextual_budget_change (complete) | 11/11 | 0 / 2 | -2 | -18.2 pp | [-45.5, +0.0] pp |
| line_restricted_aggregation (complete) | 50/50 | 0 / 0 | 0 | +0.0 pp | [+0.0, +0.0] pp |

Secondary identical-anchor question endpoint: 224 matched questions on 56 pages, 8 repaired / 1 broken; exact question accuracy difference +3.1 pp, document bootstrap 95% CI [+0.4, +6.2] pp. This records partial-answer gains and does not replace the registered all-four decision.

The untouched source panel keeps the globally planned first-page prefix. Its unfiltered results include pages rejected by the edited-pair teacher; the accepted-subset endpoint above uses the same teacher-valid pair IDs as the primary experiment.

| Untouched source arm, unfiltered planned prefix | Status | 2/2 pages | Exact individual answers |
| --- | --- | ---: | ---: |
| teacher | complete (24/24) | 23/24 | 47/48 |
| visionzip | complete (24/24) | 8/24 | 21/48 |
| visionzip_ppe | complete (24/24) | 8/24 | 22/48 |
| field_anchor_control_ppe | complete (24/24) | 11/24 | 28/48 |
| field_oracle | complete (24/24) | 10/24 | 30/48 |
| field_oracle_ppe | complete (24/24) | 10/24 | 30/48 |
| random_oracle_ppe | complete (24/24) | 9/24 | 22/48 |
| spatial_oracle_ppe | complete (24/24) | 9/24 | 23/48 |
| rres512 | complete (24/24) | 18/24 | 38/48 |

## TRAIN source stratum

Teacher: 36 accepted, 4 rejected, 0 pending / 40 planned; 40 attempted. Accepted pages span 36 source documents. Decision: STOP_THIS_MECHANISM.

| Arm | Complete pages / expected | 4/4 correct | Exact individual answers | Untouched 2/2, unfiltered prefix |
| --- | ---: | ---: | ---: | ---: |
| teacher | 36/36 | 36/36 | 144/144 | 1/2 |
| visionzip | 36/36 | 15/36 | 95/144 | 1/2 |
| visionzip_ppe | 36/36 | 15/36 | 95/144 | 1/2 |
| field_anchor_control_ppe | 36/36 | 17/36 | 100/144 | 1/2 |
| field_oracle | 36/36 | 17/36 | 106/144 | 1/2 |
| field_oracle_ppe | 36/36 | 17/36 | 105/144 | 1/2 |
| random_oracle_ppe | 36/36 | 15/36 | 93/144 | 1/2 |
| spatial_oracle_ppe | 36/36 | 15/36 | 91/144 | 1/2 |
| rres512 | 36/36 | 16/36 | 95/144 | 2/2 |
| historical | 0/0 | pending | pending | pending |
| historical_ds_fixed | 0/0 | pending | pending | pending |
| line_anchor_control_ppe | 33/33 | 16/33 | 91/132 | pending |
| line_oracle_ppe | 33/33 | 16/33 | 95/132 | pending |

Primary identical-anchor contrast (complete): 36/36 matched pages; net repairs 0; document bootstrap 95% CI [+0.0, +0.0] pp.

Secondary identical-anchor exact-question difference: +3.5 pp; document bootstrap 95% CI [+0.0, +7.6] pp, 144 matched questions.

## VAL source stratum

Teacher: 20 accepted, 2 rejected, 0 pending / 22 planned; 22 attempted. Accepted pages span 20 source documents. Decision: STOP_THIS_MECHANISM.

| Arm | Complete pages / expected | 4/4 correct | Exact individual answers | Untouched 2/2, unfiltered prefix |
| --- | ---: | ---: | ---: | ---: |
| teacher | 20/20 | 20/20 | 80/80 | 22/22 |
| visionzip | 20/20 | 3/20 | 41/80 | 7/22 |
| visionzip_ppe | 20/20 | 3/20 | 41/80 | 7/22 |
| field_anchor_control_ppe | 20/20 | 5/20 | 49/80 | 10/22 |
| field_oracle | 20/20 | 5/20 | 51/80 | 9/22 |
| field_oracle_ppe | 20/20 | 5/20 | 51/80 | 9/22 |
| random_oracle_ppe | 20/20 | 3/20 | 42/80 | 8/22 |
| spatial_oracle_ppe | 20/20 | 2/20 | 40/80 | 8/22 |
| rres512 | 20/20 | 9/20 | 50/80 | 16/22 |
| historical | 11/11 | 4/11 | 27/44 | pending |
| historical_ds_fixed | 11/11 | 4/11 | 27/44 | pending |
| line_anchor_control_ppe | 17/17 | 3/17 | 43/68 | pending |
| line_oracle_ppe | 17/17 | 3/17 | 41/68 | pending |

Primary identical-anchor contrast (complete): 20/20 matched pages; net repairs 0; document bootstrap 95% CI [+0.0, +0.0] pp.

Secondary identical-anchor exact-question difference: +2.5 pp; document bootstrap 95% CI [-2.5, +7.5] pp, 80 matched questions.

Anchor allocation is field_anchor_control_ppe minus visionzip_ppe. Restricted aggregation is field_oracle_ppe minus its identical-anchor unrestricted control. These effects must not be combined into a claim about aggregation alone.

PPE target-position coverage is reported per image, distinguishing position slots from unique original token coordinates:

| Arm | Images with diagnostics | Both value regions represented in PPE | Missing a value region | Anchor replacements |
| --- | ---: | ---: | ---: | ---: |
| visionzip | 112 | 102 | 10 | 0 |
| visionzip_ppe | 112 | 102 | 10 | 0 |
| field_anchor_control_ppe | 112 | 105 | 7 | 96 |
| field_oracle | 112 | 112 | 0 | 96 |
| field_oracle_ppe | 112 | 112 | 0 | 96 |
| random_oracle_ppe | 112 | 102 | 10 | 192 |
| spatial_oracle_ppe | 112 | 100 | 12 | 45 |
| historical | 22 | 19 | 3 | 0 |
| historical_ds_fixed | 22 | 19 | 3 | 0 |
| line_anchor_control_ppe | 100 | 94 | 6 | 78 |
| line_oracle_ppe | 100 | 98 | 2 | 78 |

The manifest has 3 leading-zero fields on 2 pairs. Primary Decimal exactness evaluates numeric value; it does not establish character-exact identifier preservation. A separate literal check removes currency/comma/space but preserves leading zeros and decimal places:

| Leading-zero target sensitivity | Available / expected questions | Decimal value correct | Literal correct | Extra primary credits |
| --- | ---: | ---: | ---: | ---: |
| teacher | 6/6 | 6 | 6 | 0 |
| visionzip | 6/6 | 2 | 2 | 0 |
| visionzip_ppe | 6/6 | 2 | 2 | 0 |
| field_anchor_control_ppe | 6/6 | 2 | 2 | 0 |
| field_oracle | 6/6 | 3 | 3 | 0 |
| field_oracle_ppe | 6/6 | 3 | 3 | 0 |
| random_oracle_ppe | 6/6 | 2 | 2 | 0 |
| spatial_oracle_ppe | 6/6 | 2 | 2 | 0 |
| rres512 | 6/6 | 3 | 3 | 0 |
| line_anchor_control_ppe | 6/6 | 2 | 2 | 0 |
| line_oracle_ppe | 6/6 | 3 | 3 | 0 |

The numeric-set endpoints check recognition without proving correct field assignment. PPE representatives may change when clusters change; a positive oracle supports joint content/position preservation. A content-only claim requires an additional crossed-position control.

Interface audit: 0 violations, 0 pending requirements. Missing gates/diagnostics cannot produce ADVANCE.

Intervals use 10,000 source-document bootstrap draws, seed 20261108. Degenerate zero intervals do not establish algorithmic equivalence. Fewer than 20 accepted independent pages cannot produce ADVANCE.

- QA-grounded value boxes and OCR lines are not gold semantic fields
- same decoder K is not equal total computation; Rres512 also reduces encoder tokens
- same-anchor restriction can change PPE position representatives, so it is a joint oracle
- VAL and fresh TRAIN are reported separately; their combined endpoint is a mixed-population pilot
- Decimal primary preserves numeric value rather than identifier characters; leading zeros and decimal formatting are checked only in a separate descriptive sensitivity
- teacher rejection/pending records are retained; missing arms are not scored as zero
- a null or tied result only limits this tested mechanism and cannot refute binding preservation generally
