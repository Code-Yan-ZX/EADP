# CatchUp-style pre-selection correction: exploratory pilot

**Result: do not advance this specific Qwen3 two-stage configuration to
training or a larger study yet.** On 16 independently selected DocVQA TRAIN
questions, the sparse baseline answered 2 correctly and the proposed
pre-selection LTC reranker answered 0. There were no repairs and two
breakages. An offline dense-key reference also failed to improve the baseline.
The routed baseline is itself weak on this numeric-question panel, so this
pilot does not determine whether the idea could help a stronger implementation,
another task, a different budget, or the official CatchUp configuration.

## Scope and protocol

The motivation is a concrete execution order in
[CatchUp](https://arxiv.org/html/2610.23324v1): score and select the active
tokens, then apply local transition correction (LTC) to re-admitted tokens.
The [released code](https://github.com/pej0918/CatchUp) revision
`18374ec779df0eaab07b788b777a75e1ecd812a3` supports LLaVA in its patch entry
point, not this project's native Qwen3-VL. This experiment implements an
explicit Qwen3-VL-8B two-routing-stage adaptation rather than reproducing the
authors' LLaVA four-stage results. Both decisions retain exactly 256 of 1024
visual tokens, before decoder layers 2 and 7 (zero-based). All text tokens
remain active, with native positions and the three DeepStack additions.

The implementation sends all 1024 tokens through the first two DeepStack
additions but only the first selected 256 through the third addition, after
layer 2. Accordingly, the offline dense-key intervention captures the effect
of both a missing third DeepStack addition and skipped decoder evolution.
The other arms share the same active-token policy. It is incorrect to ascribe
the comparison exclusively to state evolution under identical DS inputs.

All 16 independently selected source parents came from an unused reserve pool
of DocVQA TRAIN, with the first existing numeric human QA per page. This is a
compact, numeric-question-selected pilot, not an official held-out benchmark,
unseen-layout test, or evidence that the pretrained model never saw those
documents. No questions were selected or excluded using any model result.
All 16 are the primary denominator, including the full-model error and
truncations. Greedy generation has a 24-token cap. Strict numeric exact is the
primary measure; ANLS is a secondary measure and can award partial credit to
an incorrect digit. Frozen details: [protocol](catchup_rerank_prereg.md),
`Qwen_vl/outputs/catchup_rerank/run_manifest.json`.

## Accuracy

| Arm | Exact / 16 | ANLS | Truncated |
| --- | ---: | ---: | ---: |
| Full native 1024 | 15 | 0.938 | 1 |
| Native 512-resolution, 256 tokens | 7 | 0.713 | 1 |
| Two-stage baseline, stale-score selection | 2 | 0.370 | 0 |
| Offline dense-key full-pool reference | 1 | 0.270 | 0 |
| Same dense-key scores, stale Top-512 shortlist | 2 | 0.263 | 0 |
| Pre-selection LTC, stale Top-512 shortlist | 0 | 0.290 | 0 |

Versus the baseline, LTC's paired change is **0 rescues, 2 breaks**;
the source-document bootstrap 95% interval for the exact-accuracy change is
`[-31.25, 0]` percentage points. Dense-key full-pool reference has 0 rescues
and 1 break; dense-key shortlist has 1 rescue and 1 break. These very small
samples and percentile intervals do not establish equivalence, a general harm,
or statistical generalization. All individual answers and aliases, literal
exact sensitivity, official ANLS and complete paired statistics are in
`Qwen_vl/outputs/catchup_rerank/analysis.json` and `predictions.json`.

On average, LTC changed **59.8 of 256** second-stage selections, so the lack
of benefit is not because the reranker left selections unchanged. The dense
full-reference selection changed 88.1 tokens on average. On this sample the
stale Top-512 contains **94.19%** of the dense full-reference Top-256, averaged
per page. Among dense-reference-selected deferred tokens specifically, it
contains 1,330/1,501 (**88.61%**). This shows that the shortlist missed some reference choices; it does
not establish that the omitted choices would have improved QA. Dense-key
scores use the same compressed-query state and replace only deferred visual
keys with same-depth states from a separate dense trajectory. Those keys are
never inserted as execution states and are unavailable without expensive
offline computation; this is a reference intervention, not an attainable
accuracy upper bound. Both reference arms use identical full-population scores,
with only their eligible final selection pool differing.

The sparse baseline's 2/16 versus the 512-resolution reference's 7/16 makes
this experiment an especially weak test of a final deployable method. The
selection schedule, lack of third-DS updates to deferred tokens, and the
selected numeric-document distribution can all matter. No single cause has
been established. The result does not test the official LLaVA setup, all
CatchUp schedules, grounding, or unseen layouts.

## Runtime

Measured on one outside-panel development document, batch 1, NVIDIA A40,
with model loading excluded and image preprocessing, vision, routing and
generation of the first token included. Three alternating synchronized
repeats after one warm-up per arm gave median wall times:

| Arm | First-token wall time, median of 3 |
| --- | ---: |
| Two-stage baseline | 349.26 ms |
| Pre-selection LTC | 352.07 ms |
| 512-resolution reference | 175.11 ms |

For the same sample, the second routing operation itself was approximately
3.37–3.42 ms for baseline and 5.80–6.01 ms for LTC. These are measurements of
this unoptimized adaptation on one image, not official CatchUp numbers or a
reliable deployment benchmark. `latency.json` retains every repeat. The first
attempt to start the timing script failed on an ambiguous Python `run` import;
the script was corrected before any timing run and that failed log was kept.
The frozen run manifest records the earlier timing-script hash; the final
timing-script hash differs solely because of this later import fix, and did not
change the frozen QA runner or its predictions.

## Validation and reproduction

- Independent CPU audit: 25 checks pass, including numerical LTC comparison
  with the official code and native tiny-Qwen3 per-head attention checks.
- The full-token 8B manual walk matches native prefill and the tested greedy
  decode logits bit for bit (three generated tokens before EOS, cap eight).
  Repeated baseline is deterministic; all four arms
  pass the token count, selection, original-position, DeepStack, and per-layer
  cache/greedy-decode gates on the development image.
- The first GPU gate failed due to an explicit attention mask selecting a
  numerically different SDPA kernel; restoring the native causal path made the
  full-token comparison exact. In the independent one-shot-pruning gate, the
  unmodified older engine produced a maximum 0.25 logit difference on GPU.
  Its expanded mask is verified after SDPA slicing to equal the causal lower
  triangle. Temporarily aligning that unpadded test to the causal SDPA kernel
  produces exact independent prefill equality; the test restores the function
  immediately. Both earlier failed gate artifacts are retained. The older
  engine compacts early-layer caches, so its decode is not equated to CatchUp's
  layer-specific-cache decode; the latter is checked separately.
- The independent final artifact audit passes **1,809 checks**, including all
  16 source documents, token selections, cache lengths, decode positions and
  DeepStack injections. Its initial failed hash check and the separately
  disclosed post-freeze timing-script correction are both retained. The gate
  and scorer are not a reproduction of original LLaVA benchmarks.

Reproduction commands from the project root (model files are local):

```bash
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
$PY Qwen_vl/scripts/catchup_rerank/data.py
$PY Qwen_vl/scripts/catchup_rerank/audit_ltc.py
$PY Qwen_vl/scripts/catchup_rerank/audit_mask.py
$PY Qwen_vl/scripts/catchup_rerank/gates.py
$PY Qwen_vl/scripts/catchup_rerank/run.py
$PY Qwen_vl/scripts/catchup_rerank/analyze.py --input Qwen_vl/outputs/catchup_rerank/predictions.json --output Qwen_vl/outputs/catchup_rerank/analysis.json
$PY Qwen_vl/scripts/catchup_rerank/benchmark.py
```

`run.py` requires the matching frozen data, gate, and implementation hashes
and refuses changed configurations. It resumes only records already present
under the same frozen configuration. Run a single GPU process at a time.
`freeze.py` is deliberately write-once to avoid silently overwriting the
pre-outcome protocol; the existing freeze need not be regenerated.
