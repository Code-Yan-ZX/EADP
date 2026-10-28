# NeXT reproduction diagnosis — 2026-10-09

Scope: the user explicitly requested investigation and repair in **EADP_amp**.
This is independent of the existing frozen 75-arm / 55-publication-group queue.
No frozen model source, plan, prediction, score, or existing method definition
has been overwritten. New code lives in `Qwen_vl/scripts/stage1_roundtrip_pilot/`;
evidence lives in `Qwen_vl/outputs/audit_followup_20261008/next_runtime_repair_20261009/`.

## Model identity and remaining accuracy gap

The local checkpoint is `liuhaotian/llava-v1.6-vicuna-7b`, the model selected by
the [official EADP NeXT TextVQA script](https://github.com/SJTU-DeepVisionLab/EADP/blob/main/LLaVA/scripts/v1_6/eval/textvqa.sh).
Earlier audits matched all three safetensors payloads to the Hugging Face LFS
hashes. Today's actual CPU loader inspection also matched all **395 embedded
vision/projector tensors**, after the intended BF16-to-FP16 conversion, to the
loaded state. The finetuned vision tower was not accidentally discarded.

Actual loaded class is `LlavaLlamaForCausalLM`, alpha=.5, beta=2, CLI budget=32,
attention implementation=SDPA. Vision parameters are FP16, CLIP text and visual
projection are FP32. This is a CPU placement audit using the unchanged loader;
the separate GPU panel records its own actual runtime snapshot.

Full TextVQA scores already available before this diagnosis:

| NeXT budget (nominal total / CLI per crop) | Paper EADP | Local wait-only EADP | Local minus paper |
|---|---:|---:|---:|
| FULL | 60.3 | 60.370 | +0.070 |
| 640 / 128 | 59.2 | 57.866 | -1.334 |
| 320 / 64 | 57.0 | 55.410 | -1.590 |
| 160 / 32 | 54.2 | 52.086 | -2.114 |

The already completed beta=1 control scored 50.966, worse than beta=2; it did
not fix the residual. The caller-stream wait repair improved EADP K32 by .090
points, insufficient to explain the full gap. Other models are not uniformly
reproduced: v1.5 TextVQA is close across budgets, but v1.5 POPE K32 and several
Qwen tasks still have deficits. Missing/scoring-incomplete cells remain missing.

## New findings and opt-in repairs

1. **Invalid dense-guidance tokens.** Released `clip_encoder.py` pads text to
   multiples of 77. Although the text encoder receives an attention mask,
   returned hidden states at padding query positions remain nonzero. Released
   `entrpy_filter` ranks all 77 positions, including padding, BOS and EOS.
   The [paper's dense-guidance description](https://arxiv.org/html/2607.02484v1#S4.SS2)
   uses non-EOS text tokens. In a fixed 16-question midpoint CPU FP32 panel,
   593 of 1500 selected positions were padding and 40 were special tokens.
   Invalid positions carried over half the local weight in 17/100 crop-segment
   pairs, affecting 10/16 questions. This is a mechanism observation, not a
   population accuracy estimate or a GPU precision measurement.

2. **Long chunks lack EOS.** Original whole-prompt tokenization adds BOS/EOS
   once, then blindly reshapes to chunks of 77. In the full 5000-question file,
   881 prompts exceed 77 positions and 1122 original segments have no EOS.
   With this CLIP's legacy pooling rule, the first long segment pools at BOS
   (position zero). Two distinct real long prompts gave bit-identical first-
   segment global embeddings; corrected per-chunk EOS pooling distinguished
   them. Later EADP averaging does not make the entire prompt embedding constant,
   so this finding must not be overstated.

`next_text_guidance_control.py` supplies two independent, reversible controls:

- `mask_content`: excludes padding/BOS/EOS before entropy selection; computes
  the existing q=.2 fraction over actual content; pads ragged selections with
  zero aggregation weight; falls back to existing global guidance for empty
  segments. Excluding BOS is an explicit content-token interpretation.
- `bounded_chunks`: splits content into at most 75 tokens and adds BOS/EOS to
  **every** CLIP chunk. All 5000 inputs preserve all original content tokens,
  every resulting chunk has EOS, and every originally short input is unchanged.

Neither control changes the LLM prompt, images, alpha, beta, q, pruning budget,
facility selection, crop allocation, smoothing, or answer scoring. They are
**alternative corrected text-guidance protocols**, not byte-identical replay
of the released code and not proof of the author's unpublished configuration.
These shared CLIP issues also exist on v1.5; they are not inherently NeXT-only.
No accuracy cause is established until measured.

## Validation and GPU panel

Four CPU semantic gates pass: exact released filtering when all positions are
valid; invariance to arbitrary invalid-position scores; finite empty-segment
handling; and BOS/EOS/content preservation across chunk boundaries through 500
content tokens. Corpus-level token checks also pass with zero content loss,
zero missing EOS after repair and zero changes to short inputs.

The independently registered GPU panel uses 64 fixed midpoint questions and
four rotated arms: wait-only released, valid dense tokens only, bounded chunks
only, and both. All use EADP K32, alpha=.5, beta=2, q=.2, greedy decoding and the
same official scorer. This panel diagnoses execution and direction; it cannot
replace the full 5000-question scores or establish recovery of paper accuracy.

At 09:32:29 the identified CPU supervisor PID 2558609 was temporarily SIGSTOPed
to reserve the next model slot. **Neither generation worker was signalled.**
The coordinator has a 20-minute independent resume watchdog and a finally
resume block. It resumes the supervisor as soon as the diagnostic creates its
CUDA context; the existing foreign-context gate then prevents a third model.
The coordinator stops only its own diagnostic on failure. See
`panel64/slot_coordination.json` for exact events. No existing scheduler source,
state, plan, publisher, or timer was changed.

GPU panel finished at 09:45:46 with exit 0. All 64 baseline predictions match
the corresponding existing wait-only full-run predictions exactly; all 12
registered source hashes, ordered sample identities, runtime and result hashes
passed. The GPU object snapshot confirms the CPU audit's alpha/beta/dtypes/SDPA.

| Fixed 64-question arm | Panel accuracy | Delta vs released | Scored wins / losses |
|---|---:|---:|---:|
| wait-only released | 56.40625 | 0 | 0 / 0 |
| valid dense tokens | 54.37500 | -2.03125 | 0 / 2 |
| bounded chunks | 56.40625 | 0 | 0 / 0 |
| both | 55.93750 | -0.46875 | 0 / 1 |

**Decision: do not promote either control into the production baseline.** The
semantic defects are real, but this small paired panel does not show accuracy
recovery, does not identify them as the cause of the paper residual, and cannot
determine their full-corpus effect. The adapters remain explicit opt-in research
controls. No successful reproduction fix is claimed, and the full benchmark
scores in the first table remain unchanged.

Additional exclusion checks: the actual five-crop preparation function passed
12 independent sentinel cases (global prefix plus 48x48 raster, heterogeneous
masks and three image aspect ratios). Eight seeded variable-quota selectors
matched an independent scalar facility-gain oracle. No feature-mask permutation
bug was found. Existing full-corpus pruning loss is 8.201 points on short prompts
and 8.672 on long prompts, a descriptive difference of only .471; difficulty is
confounded and this does not attribute the paper gap to prompt length.

The CPU supervisor resumed at 09:43:17 and was resumed again in the coordinator's
finally block at 09:45:46. The watchdog exited, the diagnostic GPU process exited,
and the original queue resumed generation of `next_pope_LRMAIN0125` and
`next_pope_LRMAIN025` with exactly two registered CUDA models. Phase2 has 12/32
completed arms at this checkpoint. The original publication scope is unchanged.

Remaining evidence gap: the authors' actual Table 2 run arguments/configuration
and environment cannot be inferred from the paper scores. Public-code replay,
model identity, actual loaded parameters, stream ordering and crop alignment
have been checked, but the 1.334–2.114 point NeXT TextVQA residual is unresolved.
Changing hyperparameters to select the highest test-set result would not prove
a faithful reproduction. The public issue endpoint returned no issues at this
check; no message or issue was posted to the authors.

## Follow-up: concurrency, timeouts and missing answers

The user specifically asked whether concurrent runs timed out on individual
questions. A new CPU audit (`audit_generation_completeness.py`, run using
`qwen3vl_clean` for its spreadsheet reader) inspected 20 completed v1.5/NeXT
TextVQA prediction files: 100,000 records across runs, each file containing the
same complete ordered 5000 composite question/prompt keys. There were **zero
missing, extra, duplicate, empty/non-string or failure-marker predictions**.
All 20 associated generation logs had zero matches for TimeoutError, timed out,
timeout exceeded, CUDA OOM/error or traceback. The eight current wait-only
controls additionally have 5000 runtime rows, exit 0, complete/finished markers,
and score-to-prediction hashes matching the audited bytes.

Eight Qwen prediction spreadsheets (EADP and AnchorZip, TextVQA/DocVQA/AI2D/
HallusionBench; 28,776 rows) also have the expected row counts, unique indices,
zero empty predictions and zero failure markers. This checks prediction cells;
it does not independently revalidate every historical Qwen runtime log or timer.
Evidence: `next_runtime_repair_20261009/timeout_audit.json`.

The LLaVA TextVQA generation loop has no per-question wall-clock timeout or
catch-and-skip fallback. `max_new_tokens=128` is a token limit, not seconds;
neither checked LLaVA generation_config declares `max_time`. A nonempty answer
can still be wrong or hit a token limit; the stored files lack EOS/finish-reason
traces, so this audit does not claim every answer ended naturally.

Concurrency-related correctness remains a distinct question. In the earlier
128-question repeated-input stream panel, the unpatched path changed decoded
answers on 5 questions and retained-token counts on 43; the wait-patched path
changed neither. This is an intra-process stream dependency defect, not evidence
of timeout omissions, and does not establish that running two model processes
caused it. The full K32 EADP wait repair changed 51.996 to 52.086 (+.090), leaving
2.114 points below the reported 54.2. Explaining that residual solely through
omissions would require roughly 106 fully-correct-answer equivalents, while
the inspected file has zero omissions/empty answers.

No strict isolated-GPU versus two-process-GPU factorial test was run in this
follow-up. The result rules against missing/failed answers as the explanation
for these completed runs; it does not prove that all numerical or scheduling
effects of concurrent execution are impossible. No experiments were started,
stopped or modified by this CPU audit.
