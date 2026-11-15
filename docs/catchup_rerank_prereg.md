# CatchUp reranking: exploratory real-QA pilot

This protocol is a **draft until the model runner freezes its exact algorithm and
generation configuration before the first panel prediction**. The data selection
below is fixed independently of all new model outputs. This is a 16-question
exploratory diagnostic; either a positive or a negative result only informs the
next study and cannot establish general effectiveness or rule out the method.

## Question and scope

Test whether CatchUp-style reranking changes an existing compression method's
selection in a way that improves actual native Qwen3-VL-8B question answering.
The endpoint is a generated answer to a human-annotated question, not literal
coverage in an OCR transcript. This experiment introduces no trained carrier,
new reward training, or gradient-based deployment computation.

Read `docs/project_handoff.md` in full before interpreting this pilot. Previous
CF carrier failures, the stopped binding-merger mechanism, and the insufficient
LayoutLite reward panel remain separate evidence. They do not establish the
outcome of the present reranking test.

## Frozen data selection

Run `Qwen_vl/scripts/catchup_rerank/data.py` with the clean Python environment.
It uses **all 16 `role=reserve` pages**, in their original order, from
`Qwen_vl/outputs/local_reward_diagnostic/data/manifest.json`. For each page,
select exactly `qa[0]`: the first existing human numeric QA in that source
manifest. The result is 16 independent source parents and 16 questions. Keep
the original image bytes, questions, and every answer alias unchanged. Source
and image SHA256 values, source parquet revision/row/hash, parent identity,
and selection indices are recorded in
`Qwen_vl/outputs/catchup_rerank/data/manifest.json`.

The earlier reserve pool was selected by fixed seeded ordering, provided-OCR
length and silver numeric localization, without model-output selection. Its
scope is compact numeric DocVQA TRAIN pages with 40–450 provided-OCR words and
at least two numeric QA fields. These pages exclude every VAL parent, prior CF
and binding parents, and conservatively the historical fit roster. They were
not used in the previous 192-output OCR panel; the builder audits parent
overlap and fails rather than replacing a sample if this assertion changes.
This project-level independence is not proof that the pretrained model never
encountered DocVQA TRAIN data, nor a claim about unseen-layout generalization.

Do **not** filter the panel by full-model correctness, any compressed-model
failure, truncation, or agreement among algorithms. Report full-model mistakes.
Do not replace a question or page after generation. Provided-OCR regions are
silver localization and provenance only; no gold answer box enters selection.

## Planned comparisons and freeze

The frozen pilot uses 1024 input visual tokens and a final K=256 visual-token
budget. Each compressed arm uses the same native main-visual stream, three
DeepStack streams and original native mRoPE coordinates. Candidate size 2K is
512; it is not a larger final decoder token budget.

Before any pilot answer, the implemented scope is fixed as follows. It is a
two-stage Qwen3 adaptation, with routing at zero-based decoder layers 2 and 7,
each retaining K=256, not a reproduction of the authors' four-stage LLaVA
schedule. Main V has N=1024 initially; the three DeepStack injections occur
after layers 0, 1 and 2 and therefore respectively cover N, N and K active
tokens. A deferred token misses the third DeepStack injection; the dense-key
reference therefore combines that omission with skipped decoder evolution.
Both the baseline and deployable reranker use the same policy. Original 3-D
positions are retained for each active token.

At the second route the baseline selects stale-score Top-K. Freeze its
intersection with the first selection as the common anchor set for every arm.
Each arm's selected deferred tokens receive the same LTC rule once after
selection. `dense_full` substitutes dense same-depth states only for deferred
scoring keys, without injecting them into execution. `dense_2k` uses exactly
those same full-population dense scores but selects within the stale Top-512
shortlist. `ltc_2k` instead temporarily corrects only deferred shortlist keys
using the common anchors. Attention uses per-head softmax over all prompt keys,
then averages heads. Dense reference uses no answer tokens, is offline and is
not a downstream upper bound. The generation and code hashes are recorded in
`run_manifest.json` before panel generation.

| Arm | Purpose |
| --- | --- |
| `full` | Native 1024-token reference, no pruning |
| `rres512` | Native 512-square input, 256 tokens, no pruning |
| `baseline` | Original compression/selection without the new reranking |
| `dense_full` | Exact dense-reference reranking over the full candidate population |
| `dense_2k` | The same exact-reference reranking restricted to the frozen 2K shortlist |
| `ltc_2k` | The proposed lightweight reranking on exactly that same 2K shortlist |

Before generation, the runner must write `run_manifest.json` recording the
precise baseline, candidate ranking and tie-breaking, dense-reference formula,
LTC formula and all hyperparameters, exact arm names, seed, K, candidate counts,
prompt, generation cap, greedy/EOS rule, model snapshot, software versions,
source hashes and code hashes. If any distinction above is not implemented,
report the missing arm rather than invent its result or silently rename it.
If a formula is an adaptation rather than the authors' released implementation,
state this explicitly; this pilot is not a full paper reproduction.

The full and 2K dense references separate candidate-shortlist loss from the
lightweight approximation's loss. All arm outputs must be retained. No
hyperparameter search or panel-outcome-based checkpoint/seed choice is allowed.

## Interface gates

Before panel generation, verify the following on fixed outside-panel smoke
data and record the exact sample/configuration used:

1. Native keep-all prefill and greedy decode agree with stock Qwen3-VL, using
   the established engine and a declared numerical tolerance.
2. Main V and **all three** DeepStack streams share the same kept original
   indices, cardinality and ordering; every final compressed arm has exactly
   K tokens, with no duplicates or out-of-range indices.
3. Kept 3-D positions equal the corresponding full-sequence native positions;
   decode starts at the correct full-position maximum plus one and continues
   consistently through the native cache.
4. If reranking concerns greedy selection order, preserve this trace separately
   while restoring raster order for decoder insertion. If it instead changes
   decoder sequence order, the existing `NativeEngine.prefill` sorts indices
   and cannot demonstrate such a change without an audited separate path.
5. Dense and approximate scorers' shapes, finite values, deterministic tie rules,
   shortlist identity, and the actual resulting sets must be auditable. A
   no-op or unexpected identity is reported rather than treated as a gain.

A failed interface gate is an implementation failure, not evidence against
the research idea. Correct it before generating interpretable results and
retain the failed audit. Do not overwrite incompatible panel outputs.

## Outcomes and reporting

The primary denominator is **all 16 preselected questions**. Use the original
human answer aliases, report exact answer correctness under a frozen numerical
normalization rule, and add official DocVQA ANLS as a separate secondary
endpoint. Numeric normalization must preserve percent units and signs; disclose
its treatment of currencies, commas and leading-zero identifiers. A literal
alias exact sensitivity can expose identifier-format effects. Do not replace
primary exact correctness with ANLS, since a one-digit error can still have a
high ANLS. Reuse official ANLS semantics: `anls_compute` returns distance;
`hit_calculate` converts it to the final similarity score and threshold.

For every arm, report count/16, every generated answer and ground truth, length
and truncation flag. Report paired rescues and breaks versus baseline, and LTC
versus dense-2K. Summarize dense-full versus dense-2K separately. Bootstrap
whole independent source documents for paired 95% intervals and show the
underlying integer counts. With 16 questions, intervals are exploratory and
wide; degenerate zero-difference intervals do not prove equivalence.

All questions and full-model errors stay in the primary analysis. Any
teacher-correct subset is secondary, explicitly labeled and accompanied by its
changed denominator; it never changes the frozen panel. Keep truncations in
the main analysis and report them per arm, optionally adding a clearly marked
all-untruncated paired sensitivity. Do not infer missing answers as measured
correctness or silently discard them; an incomplete run is pending.

Speed claims require **actual measurements** under a recorded, shared timing
protocol. Report selector/reranking time and end-to-end TTFT separately, account
for every dense-reference or shortlist operation used, synchronize GPU timing
properly and disclose amortized/shared work. Quality-only results cannot
establish a speedup; a reduced theoretical operation count is not a measured
latency result. R-res is a quality/cost reference with a different vision input.

This pilot can justify a larger independent panel only if it identifies a
repeatable, concrete benefit worth testing. A zero, negative or uncertain result
is a limitation of this small exploratory configuration, not a general
rejection of CatchUp, reranking, or compression. Do not publish a training or
benchmark-level success claim from these 16 questions.

## CPU-only data reproduction

From repository root:

```bash
/home/dell/miniconda3/envs/qwen3vl_clean/bin/python \
  Qwen_vl/scripts/catchup_rerank/data.py
```

The script does not load a model or use a GPU. It refuses to change an existing
frozen manifest; a separate data-only reproduction uses `--out PATH`. The
native QA runner and its execution command will be recorded when its actual
implementation is complete. No GPU results exist at the time of this draft.
