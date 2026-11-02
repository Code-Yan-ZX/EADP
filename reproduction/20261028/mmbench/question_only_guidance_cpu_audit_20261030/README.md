# MMB question-only guidance: CPU input audit

This is an input audit, not a new MMB score or a GPU run plan. It does not read
answers, GT columns or current Qwen K256 outputs. No vision/text model, selector,
generation or evaluator is run. The saved rows contain hashes and token counts.

Reproduce from the V2 root:

```bash
/home/dell/miniconda3/envs/llava_pruner/bin/python -B reproduction/20261028/mmbench/question_only_guidance_cpu_audit_20261030/audit_inputs.py
```

The script disables CUDA and network model access before importing libraries.
It uses the native cached CLIPTokenizerFast and the unmodified source option/null
helpers. NumPy reproduces the native consecutive-77/zero-padding transform; the
full-guidance IDs and attention-mask hashes must match all 4377 existing accepted
input fingerprints before results are written. Only this directory is written.

## Public source and measured input difference

At author commit `e1a08801461c181871b9ff1cb0803b8e9966f083`,
[`model_vqa_mmbench.py:92–100`](https://github.com/SJTU-DeepVisionLab/EADP/blob/e1a08801461c181871b9ff1cb0803b8e9966f083/LLaVA/llava/eval/model_vqa_mmbench.py#L92-L100)
constructs the question with any hint and every option. Lines118–119 then choose
that full prompt as CLIP guidance, immediately above an inactive alternative
`question = row['question']`; line130 passes it as `texts`. The complete LLM
prompt is constructed earlier and is independent of this late assignment.
The [paper appendix](https://arxiv.org/html/2607.02484v1#S10.SS1) describes circular
option evaluation, but does not uniquely identify this CLIP guidance boundary.
An inactive comment is evidence for a concrete input hypothesis, not evidence
that the paper main table used it or that the active implementation is wrong.

The CPU execution saved in `report.json` and `token_fingerprints.jsonl` finds:

| Input property | Active full guidance | Question-only alternative |
| --- | ---: | ---: |
| Rows | 4377 | 4377 |
| Minimum / maximum CLIP tokens | 13 / 582 | 6 / 79 |
| Rows longer than 77 | 703 | 4 |
| Rows with changed IDs or attention mask | — | 4377 |
| Circular groups with constant guidance within rotations | 0 / 1176 | 1176 / 1176 |

All1176 groups have the same question and accepted image-payload hash within
their rotations. The active guidance changes with options in every group.
Question-only also removes hints on322 rows; it changes semantics and segment
boundaries together. No effect on actual importance, masks, LLM inputs, answers
or accuracy has been measured. The tokenizer's normal >77 warning is expected:
the native encoder explicitly splits these inputs into77-slot segments.

## Existing evidence and cache boundary

The bounded check read the current V2 MMB Python/controller/plan/protocol/audit/
README/RESULT metadata and the existing old-version protocol diagnosis and
prediction inventory under `EADP_amp/Qwen_vl/outputs/audit_followup_20261008/`.
No documented MMB question-only execution was identified in that scope. This
does not prove that no historical experiment exists elsewhere.

The completed V2 MMB as-is, three-wait and beta2 controls all retain full guidance;
their ordered guidance checks explicitly verify it. FULL CLI0 bypasses pruning.
The prior NeXT question-only experiment removes OCR guidance on a different
task; it does not test MMB option-rotation-dependent pruning. Existing long/short
MMB loss stratification is an observation, not an input intervention.

In the checked V2 MMB tree there are image/processor/CLIP/LLM input fingerprints,
predictions, runtime records and metadata, but no saved NPZ/NPY/PT numerical
pruner-feature/similarity cache. The historical prediction inventories are not
such caches, and Qwen/NeXT/POPE caches cannot substitute for the matching MMB
image/model/input chain. Consequently this CPU input audit cannot produce a
source-anchored mask or answer comparison without a future model execution.

## Smallest useful subsequent test, not scheduled

Use one model and the same oldEN input, K128, alpha.5/beta1/q.2, three waits,
greedy1024 and vicuna_v1. Select16 complete circular groups solely by sorted base
ID at positions `floor((2*i+1)*1176/32)`; the IDs are72,217,370,498,632,767,909,
1030,1143,1255,1364,1480,1591,1706,1821,1936. Keep all their rotations:60 rows,
120 native generations across two arms, with no GT or accuracy evaluation.

First run and preserve all60 source calls. Anchor their raw texts to the accepted
full-source run, ordered inputs/image hashes and actual current source/model
identity; do not invent old mask or output-ID anchors where not recorded.
Record new native features, global/dense scores, importance, ordered selected
IDs/gains/mask, actual spatial visual/nonvisual embeddings and input/output IDs.
Only after the whole source arm passes should the second arm change the late
CLIP `texts` assignment to `row['question']`. The full LLM hints/options/IDs,
image/features, model and generation parameters stay fixed; record any true
selection/visual-input/output changes and circular-rotation consistency.

This tests whether that exact source-supported boundary can affect MMB pruning
and downstream generation. It cannot establish the direction or size of full
accuracy change, isolate hint removal from option/segmentation removal, confirm
paper argv, or make an efficiency claim. No variant tuning, result-based panel
selection, automatic retry, GPU allocation or full4377 follow-up is created here.

## POPE and already excluded repeats

The public repository/issue page still exposes the same three-commit release and
the single PapersWithCode issue, with no additional recipe identified in this
check. No new POPE-specific discriminating evidence was found. Existing actual
wait, alpha/beta, q, Eq6/Eq10, data/model/input and scorer controls remain accepted;
the question/option boundary above does not apply to POPE. Do not repeat them as
new evidence or infer paper parameters from a nearby score. CDPruner's public
post-projector geometry supports the current choice, so this check does not
renew a pre-projector or temperature/grid proposal.
