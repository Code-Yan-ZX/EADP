# LayoutLite local-reward diagnostic: preregistration

Registered 2026-11-08 on `codex/visionzip-binding-oracle`, base `a724122d`.
Freeze this protocol, the page manifest, prompts, checkpoint revisions and mask
configuration before inspecting the 32-page masked-output panel. Interface smoke
on other pages may determine whether a separate real-QA endpoint is usable;
record that decision before generating the panel. A positive diagnostic alone
does not establish a training improvement; this registered run is inference-only.

## Question and scope

At the **same visual-token budget**, does the whole-page OCR consistency reward
rank masks poorly when they lose a small but independently annotated answer?
Does a lower-tail local consistency reward improve that ranking beyond simply
giving every fragment equal weight?

The prior binding oracle had zero incremental complete-page repairs but a
positive secondary question signal, +3.125 percentage points. It does not justify
more field-constrained merging. This experiment studies reward ranking on an
existing trained hard-pruning baseline, without another carrier, token-merging
rule, gradient teacher, learned proxy, or synthetic image edit.

This is a **no-training diagnostic**. It can support a reward-ranking hypothesis;
it cannot establish better policy gradients, training stability, generalization
of a trained pruner, deployment accuracy, or a new algorithm's novelty.

## Baseline and execution population

Primary carrier: official FireRed-OCR, based on Qwen3-VL-2B, with the official
LayoutLite `firered_600.pt` score head. The intended backbone revision is
`7bf5e9d91d3bb6bf91a3b6bc66d163096b696e50`; record the actual downloaded revision,
head SHA256, source-code commit and package versions in the run artifact.
LayoutLite source inspected before this experiment was
`7e398a2ff4f23f0cb836e36e0e81d21d379d8fc2`.

Use the official backbone's native image processing and projected tokens. For a
page with N visual tokens, every compressed mask keeps exactly
`K = round(0.5 * N)` tokens, sorted in native raster order. Keep main embeddings,
all three DeepStack streams and original spatial positions aligned. Preserve
the original unpruned prompt's native decode-position continuation. Use the
same full-page OCR prompt, greedy decoding, a **2048-new-token cap** and greedy
EOS stopping for full and compressed generation; record their exact
strings/settings before the panel. Do not increase the cap after observing
truncation. No test-time QA supervision, reward-based mask search or extra tokens.

Required gates before the panel: stock full model versus integrated keep-all
first logits and generated output agree; K is exact and unique; all DeepStack
lengths equal K; retained spatial coordinates and decode positions match the
native path; official head's feature dimensions and layer-merger inputs match
the backbone. A failed gate gives INADEQUATE_INTERFACE, not negative scientific
evidence. Fixing an interface gate must not involve picking a better reward,
mask seed or model checkpoint from panel results.

The official inference rule uses per-page score clustering with a globally
calibrated **average** compression ratio. Exact per-page K here is an explicitly
declared action-probing protocol, not a reproduction of its original allocation
rule. It is also not the project's 8B `1024 -> 256` target. A successful 2B,
50%-retention diagnostic would require later 8B/25%-retention validation. If the
official baseline cannot be loaded, pause the primary experiment; an 8B port
must be separately labeled exploratory and must not silently replace it.

The released training path appears to use different score transformations for
mask sampling and its recomputed Bernoulli log probability. This run does not
train or depend on that log probability. Any later training comparison must
first correct and document this baseline issue in every applicable arm.

### Technical amendment: throughput-only pre-run invalidation

Before inspecting any reward, answer-coverage score, QA ordering or mask-ranking
result, a serial execution probe measured approximately 37 generated tokens per
second and approximately 147 seconds for the first page's six OCR outputs. The
projected serial panel runtime was approximately 40 minutes. The run was stopped
based only on token lengths and execution times. All its predictions, run
manifest and logs are retained under
`Qwen_vl/outputs/local_reward_diagnostic/serial_throughput_probe/` and are excluded
wholesale from the primary analysis.

Use the same frozen model, head, 32 pages, mask IDs/indices, seeds, K, prompts,
2048-token OCR cap, stopping rules and reward definitions. Re-run **all 32 main
pages**, including the already probed page. Generate the full reference
separately and decode that page's five equal-K compressed actions together in
one batch using a new batch runner. Do not mix pre-run serial predictions or
references into the new primary predictions file. Archive old manifests and
write a new execution manifest/hash after this explicit amendment. This is an
execution-throughput correction, not a response to scientific outcomes or a
new pruning method.

Before restarting the primary panel, require two additional gates on a page
outside the new panel:

* Compare **eight tokens** of the integrated full-token serial generation with
  stock `model.generate` under exactly matched greedy settings, including EOS,
  token cap and prompt. Record raw token IDs, not just normalized decoded text.
* Compare **16 tokens** of each of the same five compressed masks under serial
  and batched decoding. Require identical generated token IDs through the cap
  or per-sequence EOS; record all five comparisons. Verify equal-K V/DeepStack
  indexing, retained positions, native decode continuation, independent cache
  rows and per-sequence EOS handling. Finished rows must not append later tokens
  into their scored outputs while other rows continue.

Native keep-all, stock-generation and batch-parity gates must all be explicitly
passed and saved before the new panel. Missing/failed gate evidence yields an
incomplete/inadequate execution result, never a reward-ranking SIGNAL. Do not
repair parity by changing seeds, prompts, rewards or the main sample, and do
not choose a favorable outside-page comparison after seeing main-panel scores.

## Pages and independent targets

Use 32 real, unedited DocVQA training pages from 32 distinct parent documents,
selected from the locally available dataset before compressed outcomes are
known. Exclude parent documents from the previous CF and binding panels where
identifiable. Freeze ordered page IDs, source-image hashes, parent IDs, all
existing questions/answer aliases, OCR references and curation reasons.

Selection may use only existing annotations and image properties: readable
pages, pre-existing QA, and answer localization/ambiguity checks. Do not select
pages, questions or masks using the candidate rewards, teacher fragment-tail
values, compressed failures or observed reward disagreements. Keep every
teacher rejection and every generation failure visible. No replacement pages
based on compressed outcomes.

The data-curation target is at least two existing numeric human-annotated QA
answers per candidate, each uniquely localizable in the provided automatic OCR,
with **40--450 provided-OCR words**. These are silver localization conditions,
not human bounding-box gold. Freeze 32 main pages and an ordered 16-page reserve
list. The primary panel is the fixed first 32 pages; the 16 reserve pages are
not consumed in this registered run. A full OCR missing an annotated answer makes that answer
ineligible for primary retention and remains in the yield report. Do not use
compressed failures, candidate rewards or a desire for more discordant pairs
to replace a main page or activate additional reserve evaluations.

Primary external target: retention of **existing human-annotated DocVQA answer
evidence in the compressed OCR output**. Questions and answer aliases come from
the original dataset, never from teacher text or low-scoring fragments. Freeze
which answers are eligible using annotations and the full-token OCR only:

* The answer is unambiguously localized in the dataset's OCR reference, or its
  ambiguity is explicitly flagged. Use the unambiguous subset as primary.
* The correct answer alias must actually occur in the full-token OCR. This is
  the retention denominator. Report all annotated QA, full-reference-covered
  QA, ambiguous QA and excluded QA separately, including a secondary score over
  all annotated QA.
* Literal matching uses Unicode NFKC, case folding and whitespace collapse to a
  single space (trimmed at the ends), preserving boundaries between distinct
  numbers. This differs intentionally from reward normalization, which removes
  whitespace. Thus `12 34` does not count as evidence for `1234`.
  Keep punctuation, thousands separators, leading zeros, signs, decimal digits
  and units significant; no fuzzy or numerical-value equivalence. Use numeric
  boundaries so answer `128` does not match `1280` or a decimal continuation.
  Extremely short/common aliases require an explicit ambiguity flag; do not
  retrospectively relax matching to raise the yield.

For each mask, score each eligible annotated answer as retained or absent in
its OCR output, accepting legitimate dataset aliases. Page quality Q is the
fraction of that page's eligible answer evidence retained. Higher Q is better.
A true omission is thus defined independently of the candidate reward and its
teacher fragments. A page with no full-reference-covered unambiguous answer is
not a zero-quality page: it has no primary external target and is reported as
ineligible/pending for the ranking endpoint.

**Evidence coverage is a proxy, not real QA accuracy.** It does not prove that
the model can bind the value to the queried field, reason about it, or answer
the question; a correct string may occur in the wrong field. Dataset-provided
whole-page OCR is typically automatically generated, so its token/line recall
is an auxiliary reference, not human gold transcription. Any descriptive
numeric-omission audit must retain the original aliases and image context.

Optional real-QA endpoint: first confirm that the interface accepts a question
without breaking the native model path. Run a full-token smoke using **three
previous binding-panel pages, two existing questions each**, outside the new
32-page panel. The question prompt is fixed as
`question + '\nAnswer with the answer only, without explanation.'`, with greedy
EOS decoding and a **32-new-token cap**. Run secondary real QA only if at least
**4/6 answers are exact-correct and no answer is truncated**. Record all six
outputs and the pass/fail decision before panel generation. Otherwise retain
only the evidence-coverage proxy; do not search for a more favorable prompt or
smoke-page subset.

If enabled, use existing QA ground truth and restrict real-QA retention analysis
to questions the full model answers correctly; also report all-question
accuracy and full-model yield. Freeze the same QA prompt/cap across full and
compressed masks. Real-QA results remain secondary to the already registered
primary coverage endpoint and may not be substituted after seeing panel results.

## Five masks per page

Obtain the official head's sigmoid-before raw logits once per page. Use:

1. deterministic Top-K of those logits;
2. four Gumbel-Top-K samples, each retaining exactly K unique tokens.

For sampling, standardize logits by that page's mean and standard deviation,
then use temperature 1 and independent Gumbel noise with seeds
`2026110801`, `2026110802`, `2026110803`, `2026110804`. Define an explicit numeric
epsilon for a degenerate standard deviation in code; a constant score vector
must be flagged. Record every kept-index array and mask hash. Do not resample
duplicate masks to obtain a more useful result; report the number of distinct
masks. These samples probe actions and are not a claimed new sampler.

Generate each mask's OCR with identical inference settings. The five masks are
not five independent documents. Inference errors/missing outputs remain
pending. Do not score them as zeros or silently omit the affected mask from a
reward's comparison set. Truncated compressed outputs are recorded and count
as observed outputs; also report a sensitivity restricted to pages whose full
and all five compressed outputs are untruncated. If the full reference itself
is truncated, the page is not valid for complete-page reward interpretation;
retain it in the yield table and report its exclusion. Freeze the token cap
before panel generation.

## Rewards and alignment

All candidate rewards compare a compressed OCR output with the **same full
teacher OCR**, using no QA labels, OCR GT or question-dependent weights.

Global consistency `R_global`: the official Levenshtein-ratio convention applied
to the full strings. Save the exact library version, normalization and scoring
implementation. Every mask has the same K on a page, so the original soft-budget
penalty is constant and cannot change within-page rankings.

Define the teacher's fragments once, before masked outputs. For each original
line, apply Unicode NFKC, case folding and whitespace removal, and split its
result into successive chunks of at most **64 normalized characters**. Retain
short lines/chunks; omit only empty units. Concatenating the units must equal
the normalized full teacher string. Record each fragment's text and character
offsets. Do not redefine units
using compressed errors, answer spans, QA relevance, or tail membership. Pages
with fewer than five nonempty usable fragments do not support the registered
20% tail analysis and are reported as ineligible, without fallback q tuning.

Apply the same NFKC/case/whitespace normalization to the compressed text. Align
full-reference characters monotonically to compressed-output characters using
the installed `Levenshtein.editops` minimum-edit path with unit edit costs.
Substitutions/deletions belong to the source character's fragment; an insertion
at a fragment boundary belongs to the right fragment, and an insertion after
the final source character belongs to the last fragment. Record the library
version and implementation hash before the panel. Do not independently search
for the best matching substring for each
fragment: a repeated number or phrase could otherwise be credited repeatedly.
Each prediction character may belong to at most one aligned fragment.

Let e_j be that fragment's assigned edit count and l_j its normalized reference
length. Define d_j = min(e_j/l_j, 1) and c_j = 1-d_j. Thus deleting a whole
fragment is penalized, and very long insertions are capped in the local rewards.
Save offsets, edit counts, lengths, scores and alignment coverage for every
fragment. Report the following rewards/controls:

* `R_global`: full-string consistency;
* `R_global_normalized`: full-string Levenshtein ratio after the same
  NFKC/case/whitespace normalization, controlling for superficial formatting;
* `R_span_mean`: arithmetic mean of all fragment scores;
* `R_length_weighted`: reference-character-length-weighted mean of fragment
  scores, controlling for segmentation and weighting changes;
* `R_tail20`: arithmetic mean of the lowest `ceil(0.20 * number_of_fragments)`
  fragment scores. This is the only registered tail fraction.

Also report the unclipped length-normalized control `1-sum(e_j)/sum(l_j)` to
make the effect of clipping large insertions visible. It may fall below zero.
The official Levenshtein ratio and unit-edit local scores use different
normalizations/cost conventions, another reason the local mean controls matter.
Predeclared width-32 and width-128 sensitivity results must both be reported;
neither replaces the width-64 primary or is chosen for a better outcome.

This comparison is essential: a tail benefit over the global score alone could
come entirely from equal weighting of short spans or a different alignment.
It would not establish a benefit from focusing on the lower tail.

Repeated content can give multiple equally good alignments. Use the library's
fixed path for the primary analysis and align reversed strings, mapping edits
back to original offsets, as the predeclared alternative-path sensitivity.
Report the fraction of masks whose local scores/rankings change; do not choose
whichever alignment favors the tail. If the sign of the tail's incremental
comparison changes
under this sensitivity, label the result ALIGNMENT_SENSITIVE/HOLD. Inspect a
fixed seed-selected small sample of alignments regardless of reward or QA
outcome; no outcome-conditioned deletion of difficult alignments.

Teacher OCR is a reference, not infallible truth. Requiring the annotated answer
to occur in the full OCR reduces reference error for the primary retention
target, but does not validate every teacher fragment. Report teacher/reference
disagreement against dataset OCR, and retain examples where a compressed output
corrects a teacher error. Candidate reward must not override the external QA
annotation when scoring those cases.

## Ranking analysis

Compute all ten unordered pairs of five masks within a page. The primary panel
is the registered near-global-tie subset:

`abs(R_global(mask_a) - R_global(mask_b)) <= 0.01`.

Within that subset, use pairs with different external page quality Q. No
threshold on tail separation is used to pick pairs. For each reward, concordance
is 1 when its ranking agrees with Q, 0 when reversed, and 0.5 for an exact reward
tie (fixed floating comparison epsilon recorded in code). Report ties and
abstentions explicitly. Each reward is evaluated on the **same mask pairs**;
missing/ineligible records remain pending for all compared rewards.

For each independent document, average concordance over its eligible pairs;
then average documents equally. Primary incremental quantities are
`C_tail20 - C_global`, `C_tail20 - C_global_normalized`, `C_tail20 - C_span_mean`, and
`C_tail20 - C_length_weighted`. Use document-level paired bootstrap, 10,000
draws, seed `20261108`, keeping all masks, questions and pairs of a sampled
document together. Report raw correct/reversed/tied counts as well as document
macro estimates and 95% percentile intervals. Ten pairs from one page never
become ten independent samples.

Secondary, fixed analyses: the same comparison on all Q-discordant mask pairs;
per-page best-mask selection by each reward, with ties resolved by the fixed
mask order, evaluated using Q; numerical versus other answer aliases; literal
answer evidence versus dataset-OCR-reference recall; the optional genuine QA
endpoint if frozen before generation. The best achievable Q among the five
masks may be shown as an explicitly labeled action-pool oracle, never used to
pick a reward, tail fraction, temperature or pages. No train/test tuning split
is needed because there are no fitted parameters; this also means the panel
cannot justify post-hoc reward optimization.

## Information gates and decisions

Do not interpret a partially generated panel. Planned pages, teacher-reference
yield, fragment eligibility, valid mask counts, Q-discordant pairs, near-tie
pairs and contributing independent documents must all be reported.

The primary analysis needs at least **12 independent documents and 24
Q-discordant, near-global-tie mask pairs**. These are pilot information gates,
not a power guarantee. Below either threshold: HOLD_INSUFFICIENT_INFORMATION;
do not expand the tie window or regenerate masks to manufacture the signal.

With adequate information:

* A promising reward-ranking signal requires positive incremental concordance
  over all four registered controls, with each paired 95% interval excluding
  zero on its positive side, and the same directional conclusion under the
  alignment sensitivity. This is an intersection gate, not selection of the
  most favorable control. Report any external-Q damage in reward-selected masks.
  The verdict is SIGNAL_FOR_FURTHER_STUDY, **not** permission to claim a trained
  improvement or to start unrestricted training.
* If the tail does not improve the point estimate over `R_span_mean` or
  `R_length_weighted`, stop pursuing the special tail mechanism. A mean-score
  improvement, if present, belongs to reweighting/alignment and must be reported
  under that narrower interpretation.
* If tail is positive versus the controls but the intervals include zero, report
  HOLD_UNRESOLVED and stop at this registered pilot budget. This is not evidence
  of equivalence or of a reliable benefit.
* If apparent benefit is only toward teacher fragments and not the independent
  annotated-answer target, stop the reward-ranking claim. If alignment ambiguity,
  unusable references or interface failures dominate, report the specific
  inadequate-diagnostic reason rather than reject local fidelity in general.

Do not select new tail fractions, error types, prompts or weights after viewing
the panel. Any later amendment/new experiment must preserve this completed
panel and be registered as new evidence.

## Related-work boundary

[LayoutLite](https://arxiv.org/html/2607.22200v1) already trains from full-output
OCR consistency plus layout supervision and uses adaptive clustered inference
thresholds. [Official code](https://github.com/dpxudong/LayoutLite) supplies the
baseline, training and public heads. Fixed-budget sampling is already present
in [STP-GUI](https://arxiv.org/html/2610.20390v1); equal-budget token-exchange
reward differences and local preference training already appear in
[AVEX-Prune](https://arxiv.org/html/2606.10533v1). Better credit assignment and
document-element rewards also have adjacent prior work. This diagnostic alone
cannot establish novelty for fixed-K sampling, local rewards, CVaR, counterfactual
credit, or task-aware pruning. Its narrow contribution, if positive, is evidence
that a specified local reward ranks an existing pruner's equal-cost actions
better on an independently annotated retention target.
