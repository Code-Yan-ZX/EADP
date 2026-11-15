# Native four-stream utility diagnostic: preregistered exploratory oracle

Status: **frozen before this round's main outputs, 2026-11-09**. Only data-only
preparation, literature inspection and implementation/interface development
precede the freeze. The runner must save the data, code, model, prompt and
generation hashes before calculating any main-panel teacher labels or answers.
This is a TRAIN-only, answer-conditioned mechanism diagnostic. It is not a
deployable method, an exact hard-deletion oracle, a trained scorer result, or a
GeoPrune reproduction.

## 1. Question and limits of the experiment

At K=256, does an offline token score that accounts for the main visual stream
**and all three native DeepStack injections** choose a better common set than
the corresponding main-stream-only score? Does the difference depend on whether
DeepStack is enabled during native answer generation?

Read `docs/project_handoff.md` completely before interpreting this pilot. The
experiment uses the repaired native Qwen3-VL execution graph, not historical
legacy teacher labels with missing DS/ordinary 1-D positions. It does not use
the observed attention-calibration pilot to tune data selection, targets,
teacher definitions, thresholds, seeds, or sample exclusions. Previous results
motivate examining another mechanism, not the outcome expected here.

Native-compatible compression preserves DS tensor alignment, but its selection
score need not actually use DS information. That is a hypothesis to test, not
an established defect. The present comparison tests two explicitly defined
answer-gradient scores; a negative result cannot rule out all multi-stream
selection mechanisms. A positive result cannot establish that cheap DS scores
can predict the answer-conditioned teacher or that training will generalize.

## 2. Provenance and GeoPrune boundary

[GeoPrune, Findings of EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.1913/)
computes importance from residual L2 energy after shallow spectral deflation
of final ViT output states. The paper reports T=10 power iterations, eigengap
threshold 2, cap 3, and beta=1.5 for its importance/diversity variant. Its
diversity term uses the original space; applying diversity in the residual
space is a published unfavorable Qwen3 ablation. These components must not be
silently replaced by residual-cosine diversity or presented as a new idea.

At registration, targeted searches and the ACL/PDF inspection did **not locate
a verifiable author-released GeoPrune implementation**. The paper does not
uniquely specify the local 4096-patch to 1024-merged-token score/diversity
mapping or every stopping/tie detail required to reproduce its native Qwen3
path. Thus this round does not claim an official GeoPrune reproduction or a
measured improvement over it. The strong compatible references here are native
B2 and native resolution reduction. A later GeoPrune-inspired port must be
named as an adaptation and frozen/audited separately before comparison.

This is also not a claim that using multiple visual sources or conditioning
deletion effects is new: [METEOR](https://arxiv.org/abs/2507.20842) already
studies collaborative multi-encoder pruning, [Hi-Lo's official selector](https://github.com/sealost/Hi-Lo_Prune/blob/main/hilo_prune/selection.py)
uses low/high-level features, and [CoRePrune](https://arxiv.org/html/2609.26484v1)
models depth/set-conditioned removability. The narrow empirical question is
the **same spatial keep-set across Qwen3's four differently injected streams**.
Novelty requires further review even if the diagnostic is positive.

## 3. Frozen data, labels and generation

Use 32 new independent DocVQA TRAIN parent documents, exactly one preselected
existing human numeric QA per parent, and one separate development parent.
Exclude every VAL parent, historical fit roster, prior CF/binding/LayoutLite/
CatchUp parents, and **all 33 attention-calibration parents including its
development page**. Preserve original image bytes, question strings and all
answer aliases. Source revision/row/hash, parent, image SHA256, ordering,
question index and exclusion provenance must be frozen in
`Qwen_vl/outputs/multistream_oracle/data/manifest.json` and the separate
development manifest before any main teacher or QA run.

Selection is based only on frozen source annotations and a seeded deterministic
ordering, never on full-model correctness, compression failure, gradients,
losses, truncation or a desired baseline score. All 32 questions remain in
every primary denominator. No replacement after teacher or generation results.
This project-level independence is not proof that pretraining excluded these
TRAIN images or that their layouts are unseen.

For the teacher target, use the first original answer alias, `answers[0]`,
exactly as stored. Tokenize it with `add_special_tokens=False`; do not append
EOS, generate a replacement answer with the full model, strip the alias into
another target, or select the easiest alias by model loss. Concatenate those
target IDs to the frozen native prompt IDs. If the target token sequence is
empty, fail the data/teacher gate before main execution. All original aliases
are still used to score generated QA answers.

Use the cached native Qwen3-VL-8B-Instruct, batch size 1, 1024-square input
yielding N=1024 merged visual tokens, greedy normal-EOS generation and at most
32 new tokens. Prompt construction is the existing native DocVQA path with
the fixed suffix `\nAnswer with the value only.`. Freeze actual rendered token
IDs, model/software revision, dtype and SDPA implementation. The R-res512
reference uses exactly 256 native tokens at 512-square, without pruning.

## 4. Native feature alignment and teacher construction

The frozen native vision encoder has 27 blocks and captures DS at zero-based
blocks 8, 16 and 24. At 1024-square, its post-ViT/pre-main-merger tensor is
4096 patches by 1152 channels. Input patches are in native group-major order:
four consecutive rows are the four spatial slots consumed by one 2x2 merger
group. The main merger produces V with shape [1024,4096]; each independent
DS merger likewise produces D1/D2/D3 with shape [1024,4096]. Group g corresponds
to rows 4g:4g+4 before a merger and row g after it. Do not reinterpret ordinary
raster flattening or individual patch scores as merged-token indices.

The teacher operates **after these native mergers**, on four detached floating
leaf tensors F0=V, F1=D1, F2=D2, F3=D3, all requiring gradients. The vision
encoder remains frozen and is not differentiated. All model weights are frozen;
gradients are requested only for the four leaf inputs. Use the stock language
model, native prompt features, DS additions and 3-D positions, with all 1024
visual tokens and **DS on**. Answer positions continue from the full prompt's
native maximum+1, identically to native cached generation. Prompt tokens are
not target labels. The objective is mean causal negative log-likelihood over
the preselected answer tokens only, with predictions shifted by one position.
No EOS loss or sampled continuation is included.

`NativeEngine.prefill/decode` and encoding helpers use no-grad decorators;
the teacher must therefore construct a separately audited *differentiable
input path to the same stock language model*, not call a no-grad entry point
and assume gradients exist. Features originating under inference mode must
be cloned into ordinary leaf tensors in an enabled-grad context. No copied
decoder, untracked attention rewrite or changed architecture is accepted.

One full-native teacher forward/backward supplies gradients Gs for **all four
streams in the same DS-on context**. Main-only and joint scores use this same
pass, same loss and same leaves. Do not recompute a DS-off teacher for the
main-only score, or normalize streams independently to manufacture equal
contributions.

## 5. Two fixed teacher families and common K256 selection

For original merged spatial index i, define the following four-stream scores:

1. **Magnitude sensitivity:**
   `m_s(i) = sum_d abs(G_s(i,d) * F_s(i,d))`.
   Main-only `m_V(i)=m_0(i)`; joint `m_joint(i)=sum_{s=0}^3 m_s(i)`.
2. **Signed first-order removal estimate:**
   `t_s(i) = -sum_d G_s(i,d) * F_s(i,d)`.
   Main-only `t_V(i)=t_0(i)`; joint `t_joint(i)=sum_{s=0}^3 t_s(i)`.

Calculate the product and reductions in float32. Signed scores remain signed:
no absolute value, positive clipping, per-stream rank normalization or per-image
hyperparameter search. Magnitude is a coordinate-wise absolute sum, **not**
the absolute value of a signed sum. Record this distinction in artifacts.

For each of the four scores select exactly the largest 256 values, stable
descending score order with ties broken by original raster index. Preserve the
selection trace, then sort selected indices into original raster order before
native execution. A selected spatial index retains its V row, all three DS
rows and original native 3-D position together; no independent per-stream
indices, merged features, added tokens or altered positions are allowed.

These are **approximate gradient-defined scores**, not exact hard-pruning
utilities. A gate that scales a feature toward zero preserves sequence length
and attention normalization, whereas physically removing its token changes
both. Four-stream magnitude also discards signs/cancellations. Even the signed
sum is a local derivative around the full sequence and does not capture finite
deletion, interactions among 768 removals, or generated-answer discontinuities.
Consequently gold-selected QA is not a rigorous accuracy upper bound.

## 6. Planned generated-answer arms and DS interaction

Each teacher-selected set is evaluated twice, holding its indices, original
V/positions, text and generation rule fixed: once with the three native DS
additions enabled, once with all three disabled. When DS is on, injections
occur once after native decoder layers 0/1/2; there is no extra full-token early
decoder stage. No attention calibration from the previous experiment is used.

| Family | DS on | DS off |
| --- | --- | --- |
| Native full N1024 | `full` | `full_no_ds` |
| Magnitude V-only K256 | `mag_v_on` | `mag_v_off` |
| Magnitude joint K256 | `mag_joint_on` | `mag_joint_off` |
| Signed V-only K256 | `signed_v_on` | `signed_v_off` |
| Signed joint K256 | `signed_joint_on` | `signed_joint_off` |

Additional frozen forward-only references are `b2`/`b2_no_ds` (native EADP
scoring + block8, the same K256 set with DS on/off), `rres512` (native unpruned
512-square, DS on), and `random_on`/`random_off` (the same exact 256 indices
sampled without replacement by one fixed per-item seed derived from seed
20261109 and item ID, evaluated with DS on/off). Record the seed
derivation and realized random indices in the run freeze, not a favorable seed
chosen after QA. No extra random trials are searched for a better comparator.
The exact arm list is frozen as:

`full, full_no_ds, rres512, b2, b2_no_ds, mag_v_on, mag_v_off, mag_joint_on,
mag_joint_off, signed_v_on, signed_v_off, signed_joint_on, signed_joint_off,
random_on, random_off`.

Freeze paired contrast order as (before, after):

- `(mag_v_on, mag_joint_on)`, `(mag_v_off, mag_joint_off)`,
  `(signed_v_on, signed_joint_on)`, `(signed_v_off, signed_joint_off)`;
- `(full_no_ds, full)`, `(b2_no_ds, b2)`, `(random_off, random_on)`;
- `(reference, teacher)` for each of the four DS-on teacher arms and each
  reference in `b2, rres512, random_on`;
- `(reference, teacher)` for each of the four DS-off teacher arms and each
  reference in `b2_no_ds, random_off`.

For each family report joint-minus-V-only paired QA changes with DS on and
with DS off, and their interaction: `delta_on - delta_off`. Report full-on/off
separately to show the effect of DS removal without compression. The selected
sets are constructed under the DS-on teacher, so DS-off also changes the
teacher/execution distribution. An interaction is diagnostic support, not
sufficient proof that DS supplies unique causal answer evidence. A benefit
that remains equally large without DS cannot be attributed specifically to
preserving DS injections by this experiment.

Known metadata caveat: the current `NativeEngine` sets `PrefillState.n_vis=0`
when DS is disabled because it derives the field from the absent DS mask.
The physical sequence still contains the selected visual rows. Preserve that
reported field and separately record `actual_visual_count=len(keep_idx)`;
verify physical embeddings, positions and cache lengths. Never interpret the
metadata zero as genuine removal of all visual tokens or silently alter model
execution to make the metadata look correct.

## 7. Development gates and recorded diagnostics

Before main teacher labels or answers, use only the one outside-panel
development page and CPU synthetic fixtures. Save all failed attempts and
source hashes; a gate failure is an implementation problem, not a method
result. Required checks:

- Native full-on prefill and forced decode agree with stock, including native
  DS/3-D positions. Full-off genuinely omits the three injections while keeping
  the full visual sequence. Main/DS merger capture, native group ordering and
  all N/K counts are checked against stock features.
- The differentiable teacher reproduces the same answer-token loss/logits as
  the equivalent no-grad native teacher-forced path with identical input shape,
  positions, mask and DS. Record a predeclared dtype-appropriate tolerance,
  CPU float32 exact-reference error and GPU maximum/relative error. No tolerance
  is relaxed based on a useful teacher ranking or QA answer.
- All four leaf gradients exist and are finite; all model-weight gradients
  remain absent and all model weights remain unchanged. Verify answer-token
  target alignment and count, first/last target IDs, no appended EOS, and
  absence of prompt targets. A synthetic scalar/finite-difference fixture
  checks the sign of `-sum(G*F)` and the distinction from `sum(abs(G*F))`.
- Stable top-K, ties and raster restoration are tested on synthetic equal and
  negative signed scores. Actual K256 is unique/in-range. Identical score
  definitions produce identical sets. V/DS/position input hashes agree for a
  given set across its on/off pair, with DS execution status recorded.
- Each generated arm starts from a fresh native cache, uses the full-position
  decode delta, and has correct cache length and layer-call count. On/off is
  not implemented by dropping base visual tokens, reindexing, or changing
  prompts. Hooks and temporary requires-grad states restore on exceptions.

For all 32 pages save target text/IDs and loss, four per-stream feature/gradient
norm distributions, four magnitude/signed score maps, finite/zero/negative
counts, quantiles and relative stream contributions. Save every selected set,
V-only/joint overlap and Jaccard, random overlap, and all stream/position
hashes. Describe how concentrated scores are using
`N_eff=(sum(abs(score)))^2/sum(score^2)` (undefined for an all-zero map, reported
as such); this is a score participation statistic, not a number of truly useful
tokens. Large DS scores can reflect scale or Jacobian differences; distributions
alone do not establish unique information or downstream value.

The development gate also records the time and peak allocated/reserved GPU
memory for feature extraction, teacher forward/backward and each generated
arm. If native full-N backward does not fit, do not silently reduce resolution,
remove DS, use a smaller model, shorten the target, or use old legacy labels.
Report the blocked teacher setup; any checkpointing/offload implementation
must first pass the same identity/gradient gates and be frozen before main work.

## 8. Primary outcomes and fixed interpretation

The primary endpoint is actual generated-answer numeric exact correctness on
all 32 original questions, scored using the frozen `normalize`/`correct`
functions from `scripts/cf_mining/cf_mine.py`. Preserve percent/sign semantics;
disclose currency/comma/whitespace normalization and leading-zero collapse.
Literal original-alias exact and official DocVQA ANLS on original strings are
secondary. ANLS distance must be converted using official `hit_calculate`
semantics; a one-digit-error similarity is not numeric exact correctness.

Report counts/32, every answer/alias, length/EOS/truncation and paired
rescue/break/tie counts. Predeclare both joint-minus-main comparisons under
both DS states, both interactions, full-on/off, each DS-on teacher versus B2,
R-res and random. Report paired source-document bootstrap 95% intervals with
10,000 shared draws, seed 20261109. Multiple exploratory comparisons and weak
sample resolution are disclosed; do not call a favorable small-panel difference
statistically established or turn a degenerate interval into equivalence.
Teacher-correct or all-untruncated subsets are explicitly secondary and retain
their changed denominators. Operationally missing answers are pending, not
manufactured observations or silently discarded documents.

The score uses the **gold answer of each evaluated TRAIN question**. Even
though parents are new and output generation itself is native, this is label-aware
oracle QA, not held-out deployment accuracy. No student has been trained or
validated. A future deployment claim requires a forward-only predictor trained
on disjoint fit parents and tested on new parents without gold answers.

Freeze the following decision interpretation before outputs:

- Joint not improving V-only in both families gives no evidence to train a
  multi-stream scorer from these labels. Stop this pair of first-order teacher
  hypotheses on this configuration; do not infer that DS is useless or that
  every common-set utility estimator is impossible.
- Proxy/overlap gains without generated-QA gains do not justify training or a
  method claim. If both teachers are poor versus random/B2, interpret that as
  a weak teacher diagnostic rather than a representation impossibility result.
- A positive joint-minus-V-only DS-on estimate remains tentative; report
  breaks, intervals, both families and DS-off. Stronger DS-on than DS-off
  effects support the proposed interpretation but do not prove it causally.
- If an apparent teacher gain still loses to R-res/B2, it is local mechanism
  evidence, not a competitive compression method. Do not choose the favorable
  score family and relabel its observed accuracy as an unbiased estimate.

No score-sign switch, stream normalization, target-alias search, top-K rescue
sweep, exponent tuning, training, or new module is added after seeing main
outputs. A follow-up requires a separate protocol and new data. There is no
automatic success threshold or benchmark-level conclusion from N=32.

## 9. Cost reporting and artifacts

Only the genuine forward-only references (`full`, `full_no_ds`, B2 on/off, R-res and
random) can be compared for deployable end-to-end TTFT. Use the same development
page/device/dtype/backend with fresh preprocessing/vision/selection in each
call, synchronized timing, 3 warm-ups and 10 paired rotated-order blocks, and
report raw block times and medians. A fixed-16-token generation diagnostic uses
the same ignore-EOS policy across timed references. One page does not establish
a benchmark-wide speedup.

Report gold-target tokenization, native feature extraction, backward teacher
time, ranking time, peak GPU memory and label/storage cost **separately**. A
teacher-selected generation can be timed as an isolated diagnostic, but
excluding the gold/backward computation cannot produce a deployable TTFT
claim. The minimum label budget is one full DS-on forward/backward per page,
shared by all four scores; no student training is authorized by this pilot.

Save immutable data/run manifests, code/model/environment hashes, target IDs,
teacher maps, all generated predictions, development gates/failures, physical
sequence/DS/cache traces and timing under
`Qwen_vl/outputs/multistream_oracle/`. This document is not revised after main
teacher/QA outputs; any implementation deviation is dated and reported
separately without overwriting prior artifacts.
