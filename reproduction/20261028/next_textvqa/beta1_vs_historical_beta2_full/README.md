# Full NeXT TextVQA pairing: new beta1 versus preserved beta2

The new public-default alpha.5/beta1/K128 run scores **56.944%**, compared
with **57.866%** for the qualified historical alpha.5/beta2/K128 run located
through `comparison.json`: a difference of **−0.922 percentage points**.
Both contain the same5000 ordered composite IDs and full prompts and match
the same5000 fresh official GT entries. The fresh author TextVQA evaluator
re-scores both complete sets; its native CLI stdout is preserved separately.

| Paired observation, new beta1 minus historical beta2 | Count |
|---|---:|
| Raw answers changed | 534 |
| Normalized answers changed | 499 |
| Per-question official scores improved | 93 |
| Per-question official scores declined | 138 |
| Per-question official scores unchanged | 4769 |

The improvements contribute73.1 summed per-question score units, while the
declines contribute−119.2, giving−46.1/5000 and therefore−0.922pp. Among534
raw-answer changes,303 leave the official score unchanged. All5000 paired
answers, normalized forms, official scores and total retained-token counts
are recorded in `per_question_pairs.jsonl`.

Both runtime traces are available and qualified: their SHA256 values match
their respective original official-score records, and all5000 trace positions,
IDs, image names, prompt hashes, nominalK128 and five-crop shapes align.
Total retained counts differ on2941 questions and agree on2059. The old
mean count is637.5054 and the new mean is637.5106; both range636–639.
The traces do not provide comparable selected-token indices or per-crop
quotas, so equal counts do not establish equal visual content.

The historical raw prediction SHA is
`522301b3e543833e264c02b74ff123101b5bc2e5dece2bc1fc2d7b5806d55cfc`.
The new raw SHA is
`5a3da1c410a6a044809f7cf0892762d9c59f90ec660dc1ccfd992678ddfe94c2`.
The comparison also confirms the same3166 recorded image hashes, core source
hashes and three caller-stream waits. Full source/score/input/trace fingerprints
and unchanged-before/after checks are in `comparison_report.json`.

This quantifies these two complete executions. Historical and current entry
wrappers, observations, process execution and low-level scheduling cannot be
replayed or equated by this CPU analysis; the difference does not prove beta
alone causes every changed answer. The paper main-table argv remains unknown,
and56.944 is below its59.2 reported value. This lower result does not establish
paper reproduction or completion of the broader goal. No GPU, new parameter
run, fixed plan, worker or prediction was changed.
