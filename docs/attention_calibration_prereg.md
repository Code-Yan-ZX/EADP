# Fixed-selection attention calibration: exploratory native-Qwen3 pilot

Status: **frozen before the first main-panel prediction, 2026-11-09**. This
document fixes the scientific comparisons and formulas; the runner must also
write its model, data, code, prompt and environment hashes before generation.
This is a 32-document exploratory real-QA pilot, not a benchmark reproduction,
training study, proof of novelty, or confirmatory efficacy test.

## 1. Question and evidence boundary

Does changing attention normalization improve generated numeric answers when
the visual selection, merging, positions and all three DeepStack streams are
held fixed? In particular, can a similarity-weighted effective cluster size
avoid over-weighting heterogeneous VisionZip clusters relative to raw member
counts? A secondary hard-pruning carrier checks whether a calibration effect
exists without merging. Read `docs/project_handoff.md` in full before using
this result; the historical legacy Qwen3 path is not used.

Previous binding results do not prove that normalization is the remaining
bottleneck. RESTORE already proposes size- and distance-based calibration.
The effective-size variant below is an untrained heuristic, not an exact
attention-mass reconstruction and not an established novel method.

No full-model attention distribution, answer, gradient, dense decoder pass or
teacher-computed correction enters any compressed arm. Full and R-res are
separate measured references. Oracle analyses, if later added, must be labeled
offline and cannot be counted as deployable accuracy or timing.

## 2. Frozen data and generation

Use **32 new DocVQA TRAIN source parents, one existing human numeric QA per
parent**, plus **one independent development parent** for interface gates and
timing. The data-only builder fixes ordering, annotation rule and all original
answer aliases without consulting any new model output. It must audit exclusion
of VAL parents, previous CF/binding/LayoutLite/CatchUp parents and the historical
fit roster; duplicate parents or image bytes fail the audit rather than trigger
model-dependent replacement. Save source revision/row/hash, original image
SHA256, parent identity, selected QA index, question and every answer alias in
`Qwen_vl/outputs/attention_calibration/data/manifest.json` before generation.

Do not filter or replace samples using full-model correctness, compressed-model
failure, truncation, method disagreement, or a desired baseline accuracy. All
32 questions remain in every primary denominator. The development page is not
one of these 32 and its answers never determine calibration hyperparameters.
The dataset is an exploratory project-level split, not proof that pretrained
weights never saw these documents or that layouts are unseen.

Use the cached native Qwen3-VL-8B-Instruct model, batch size 1, fixed
1024-square input yielding N=1024 merged visual tokens, and K=256 for every
compressed arm. R-res uses a 512-square image yielding exactly 256 native visual
tokens, with no pruning. Use the existing native DocVQA prompt construction,
greedy argmax, normal EOS stopping, maximum **32 new tokens**, no sampling and
no post-generation answer correction. Save exact rendered prompt/token IDs and
generation arguments in the run manifest. A cap hit remains in the primary
analysis and is marked truncated. No cap or prompt change after main-panel
generation starts.

## 3. Frozen carriers and shared-input requirement

The primary VisionZip carrier uses the stock-SDPA vision pass and the existing
last-vision-block received-attention/key capture in
`Qwen_vl/scripts/binding_oracle/runtime.py:encode_vz`. Use
`select_binding(..., variant='baseline_fixed')` with **205 dominant tokens and
51 contextual anchors**. Context anchors use the existing deterministic raster
spacing and nearest-anchor cosine assignment; no field/OCR oracle, anchor
coverage repair, PPE, changed anchors, alternative positions or new clustering
is enabled. Each contextual feature is `anchor + mean(other assigned members)`
for **V and every DS stream**, with an empty assignee set contributing zero.
This deliberately fixes the corrected native carrier; it is not the legacy
`historical_exact` arm whose DS rule omitted the anchor.

Save keep indices, dominant/anchor indices, every cluster membership and its
count, all input-stream hashes and position hashes. Across VisionZip arms,
these inputs must be identical; only the additive attention bias differs.
Original anchor positions are retained as the corresponding subset of the
full-sequence native 3-D positions. Reordering for selection traces does not
change raster order at decoder insertion.

The secondary hard-pruning carrier is the native **B2 = EADP scoring + block8
facility selection**, K=256. Its baseline and calibrated arms share exactly the
same selected V rows, DS rows, indices and positions; there is no merging and
therefore no cluster-size weighting. Record the selector implementation/hash
and existing fixed parameters. It cannot replace the primary VisionZip result
or be used to select a favorable carrier after observing answers.

## 4. Bias formulas and exact intervention scope

Let M be the ordinary causal additive mask, z the stock scaled QK logits
**after native Q/K normalization and mRoPE**, and j a key position. Every arm
uses `softmax(z + M + b)`; the bias is added after the QK scale, never divided
by sqrt(head_dim) again. All 36 decoder layers and all attention heads use the
same prescribed bias. Visual-only bias means the K visual-key columns only,
not image-boundary special tokens or other text columns. It affects every
causally permitted query row, including visual queries; causal exclusions
always take precedence.

For contextual anchor a with assigned *other* members A(a), define
`n(a) = 1 + |A(a)|`. Dominant tokens have n=1. Counts include the anchor once;
all 1024 original tokens are accounted for once, so the K counts sum to 1024.
Do not use clamped non-anchor denominators as these source counts.

For the new candidate, use the exact merged-token keys returned by `encode_vz`:
post-RoPE last-block keys, averaged across heads and then within each native
2x2 patch group, followed by L2 normalization with denominator clamped at
1e-8. For assigned i and its anchor a, let
`u(i,a) = clamp(cos(key_i,key_a), 0, 1)`. Freeze

`n_eff(a) = 1 + sum_{i in A(a)} u(i,a)^8`.

Dominant tokens have n_eff=1. The exponent **8** and all other constants are
fixed without a sweep. No weighting of V/DS, query dependence, learned
parameters, renormalization to total 1024, or additional correction is applied.
Sizes are computed once before decoder execution and reused for prefill/decode.
Require `1 <= n_eff <= n`, with singleton identity. This quantity is a proxy
for feature homogeneity, not a mathematical effective sample size guarantee.

| Arm | Bias and stage |
| --- | --- |
| `full` | Native N=1024, no calibration |
| `rres512` | Native N=K=256 at 512-square, no calibration |
| `baseline` | Fixed VisionZip carrier, native attention path |
| `vz_maskzero` | Same carrier, explicit causal mask with zero calibration, prefill and decode |
| `count` | b=log n(j) on visual keys only, prefill and decode |
| `effective` | b=log n_eff(j) on visual keys only, prefill and decode |
| `bias_const` | b=log(N/K)=log 4 on all visual keys only, prefill and decode |
| `restore_prefill` | log-size plus native-mRoPE distance term below, prefill only |
| `restore_continuous` | Same formula, prefill and decode; auxiliary adaptation |
| `b2_baseline` | Fixed native B2 hard-pruning carrier, native attention path |
| `b2_maskzero` | Same B2 carrier, explicit causal mask with zero calibration, prefill and decode |
| `b2_bias_const` | b=log 4 on B2 visual keys only, prefill and decode |
| `b2_restore_prefill` | Distance term below with every size=1, prefill only |
| `b2_restore_continuous` | Same B2 formula, prefill and decode; auxiliary |

Non-visual keys have size 1. Generated text keys also have size 1 and their
native positions are appended exactly once at each decode step. No answer
token is used to calculate a size or change a calibration parameter.

For RESTORE-style calibration, take the actual decoder rotary `inv_freq` and
the actual Qwen3 interleaved axis assignment. With F=head_dim/2 frequency
pairs, let g(f) be the temporal/height/width axis actually selected by the
installed `apply_interleaved_mrope`. For native 3-D positions p, define

`D(q,j) = (1/F) sum_f cos((p[g(f),q] - p[g(f),j]) * inv_freq[f])`,

`b_restore(q,j) = log s(j) + log(2 - D(q,j))`.

Use s=n for the VisionZip carrier and s=1 for B2. The distance term acts on
**all key columns**, including text, as in the official Qwen2.5 implementation;
only log-size is visual-specific. Use float32 phase/bias calculation, preserve
the original position coordinates, and do not substitute sequence-index or
Euclidean grid distance. Do not use Qwen2.5's contiguous frequency slices on
Qwen3's interleaved frequencies. No clipping/rounding of original integer
distances is required in this pilot. D lies in [-1,1], hence 2-D in [1,3].
Under the frozen model configuration no extra RoPE attention scaling should be
folded into D; verify actual rotary config/scaling and fail the RESTORE gate if
the assumed pure phase construction is not valid.

`restore_prefill`/`b2_restore_prefill` turn **both** size and distance calibration
off during decode. The continuous versions recompute only the current query's
distance row against cached native key positions; they never repeat prefill
or update old cache states. The four count/effective/constant arms instead
reuse their fixed visual-key sizes in both stages as stated above.

## 5. RESTORE provenance, limitations and fallback

Primary sources checked before the pilot:

- [RESTORE v2, Eq. 3–4 and Appendix A](https://arxiv.org/html/2606.01711v2),
  ICML 2026 as identified by the [author project page](https://cvlab.yonsei.ac.kr/projects/RESTORE/).
- [Official repository](https://github.com/cvlab-yonsei/RESTORE), HEAD
  `38b952518ca5aa8f2e6e3770796cff8c7d32258e`, checked 2026-11-09.
- [Official Qwen2.5 calibration source](https://github.com/cvlab-yonsei/RESTORE/blob/38b952518ca5aa8f2e6e3770796cff8c7d32258e/qwen25vl/qwen/tore/restore.py)
  and [reduction caller](https://github.com/cvlab-yonsei/RESTORE/blob/38b952518ca5aa8f2e6e3770796cff8c7d32258e/qwen25vl/qwen/tore/reduction.py).

The code resolves Eq. 4's compact notation as **log s + log(2-D)**, not
`log(s) * (2-D)` and not count-only. Paper default c=2 corresponds to code
`CALIBRATION_CONST=1` in `log(const+1-D)`. Full RESTORE also changes anchors,
uses gamma=0.5 retained/merged budget allocation and mean source positions;
this experiment intentionally fixes the VisionZip selection/merger/positions
instead. Official Qwen2.5 uses contiguous M-RoPE axis frequencies, prefill
calibration and float merged positions with a distance lookup. The tested
Qwen3 adaptation uses its native interleaved frequencies and integer anchor
positions. The continuous variants additionally extend calibration to decode.
The LLaVA code has a different decode treatment (count continues while distance
is prefill-only), so it is not an interchangeable implementation reference.
The official Qwen2.5 README warns that current code scores differ from paper
Table 3; no published score is treated as a locally reproduced baseline.

All arms are forward-only and training-free. RESTORE-style arms are an
attributed calibration-only adaptation, **not an official full RESTORE
reproduction**. No author SFT checkpoint or new training is used.

If native phase/axis, SDPA-bias, cache or position checks fail, retain the failed
audit and withhold the affected RESTORE arm **before** main-panel generation.
Continue the zero/count/effective/constant study only if its independent gates
pass, report RESTORE as unavailable with the precise failed gate, and do not
rename a 1-D surrogate, log-size-only arm, or uniform visual bias as RESTORE.
No main-panel QA result decides this fallback. Fixes after any main output
must retain old files and restart all affected arms under a new code hash.

## 6. Interface and operator gates before any main-panel prediction

Use CPU synthetic fixtures and the one outside-panel development page. Record
every gate result, first failures, fixes and final source hashes.

1. **Native identity and mask control.** Native keep-all prefill and forced
   decode match the established stock Qwen3 path, including full-shape lm_head
   comparison when claiming bit identity. Test explicit maskzero versus native
   with a recorded dtype-appropriate numerical tolerance; record any greedy
   change rather than assuming zero bias implies identical CUDA output.
   `vz_maskzero` and `b2_maskzero` are the primary zero-calibration controls
   because an explicit additive mask can change SDPA kernel selection.
2. **Matched operator.** On CPU small random Q/K/V, compare calibrated SDPA
   against explicit `softmax(QK^T/sqrt(d)+M+b)V`, with prefill, cache-like
   non-square decode, future-token exclusions and GQA fixtures; float32 maximum
   absolute output error must be <=1e-5. Bias must not reveal future keys.
   On GPU, check implementation's actual dtype/backend against a same-backend
   explicit-mask reference and record maximum/relative errors; no unexplained
   mismatch is waived based on a favorable QA answer. Preserve stock Q/K norm,
   projections, native rotary, GQA repetition, value path and cache updates.
3. **Count/effective identity.** Verify source-count sum 1024, singleton n=1,
   correct anchor inclusion, 1<=n_eff<=n, exponent 8, exact log4 constant, and
   that setting all sizes to 1 reproduces maskzero at the bias-tensor level.
   Identical non-anchor keys give n_eff=n; nonpositive similarities contribute
   zero; empty-assignee anchors have both sizes 1.
4. **Native mRoPE distance.** Independently construct the phase vector through
   the installed rotary's axis-interleaving routine and compare D with the
   frequency/axis formula on synthetic positions and real prefill/decode
   positions (float32 max error <=1e-5). Check diagonal D=1, symmetry, [-1,1]
   range and shape/order. Uniform 3-axis positions must reduce to the native
   1-D frequency average. Test official Qwen2.5 contiguous-axis formula on an
   integer synthetic fixture for formula agreement, without assuming Qwen3
   shares that axis map. Do not silently clip a formula discrepancy.
5. **Selection/streams/positions.** Every compressed sample has exactly 256
   unique, ascending, in-range visual indices, V/DS lengths 256/256/256/256,
   matching visual mask and cache bookkeeping. Hashes must establish identical
   selected V, all three DS inputs, positions and cluster membership across the
   corresponding carrier's arms. DS additions remain at native layers 0/1/2;
   calibration never skips or duplicates one. Kept mRoPE positions are the
   full-sequence subset; decode starts at full maximum+1 and advances normally.
6. **Intervention trace.** Capture the actual bias shape, finite allowed entries,
   visual/text support, layer count and stage for the development page. Verify
   count/effective/constant biases persist on each decode, and prefill-only
   RESTORE becomes exactly zero there. Hooks restore original methods/state
   after every arm, including exceptions; arm order must not leak state.

The stock decoder is reused with a narrowly scoped attention-mask/interface
wrapper; no separately copied decoder is accepted as an unaudited replacement.
An implementation failure is not a scientific failure of calibration.

## 7. Outcomes, comparisons and fixed decision interpretation

The primary endpoint is **generated-answer numeric exact correctness**, using
the existing `normalize`/`correct` semantics from `scripts/cf_mining/cf_mine.py`
as extracted by `scripts/catchup_rerank/analyze.py`. Freeze/hash those functions:
strip outer quotes, lowercase, remove whitespace/currency/comma, canonicalize
Unicode minus and Decimal; preserve sign and percent suffix. Leading numeric
zeros are not preserved, so report literal-alias exact as a separate
sensitivity. Do not extract a favorable number from free prose. Official
DocVQA ANLS is secondary on original strings: `anls_compute` returns distance;
use `hit_calculate` for its final similarity/threshold semantics.

For all 32 questions report every prediction, exact count/32, ANLS, literal
exact, token length, EOS and truncation. Primary paired contrasts are:

- each VisionZip calibrated arm versus `vz_maskzero`;
- `effective` versus `count` and versus `bias_const`;
- `restore_prefill` versus `count` and `bias_const`;
- continuous versus prefill-only RESTORE, separately labeled auxiliary;
- each calibrated B2 arm versus `b2_maskzero`, separately labeled secondary;
- both maskzero controls versus their native baseline, as numerical/backend
  diagnostics; every compressed arm versus `rres512` as the strong reference.

Report paired rescue/break/tie counts, net correctness differences and 95%
paired percentile bootstrap intervals (10,000 draws, seed 20261109, independent
source parent as unit, shared draws across contrasts). All 32 preselected
parents stay in primary analysis; a teacher-correct or all-untruncated subset
is secondary only and shows its changed denominator. Missing runs are pending,
not fabricated mistakes or excluded questions. Degenerate intervals do not
establish equivalence. Multiple exploratory comparisons are disclosed and no
best-arm selection is presented as an unbiased confirmatory estimate.

No hypothesis is declared established from N=32. Freeze the following
interpretation before results: a positive point estimate with observed
rescues is a candidate signal; its interval and breaks determine uncertainty,
and it still requires an independent larger study. `effective` cannot claim
added value unless it improves both the count and constant-bias controls; a
count-only or constant-only gain does not validate the homogeneity heuristic.
A zero/negative effective-minus-count result supplies no reason to train or
tune this exponent on this panel. A positive calibration-minus-maskzero result
that still loses to R-res is a local repair signal, not a competitive method.
There is no post-hoc relaxation, substitute proxy endpoint, learned parameter,
alpha/exponent sweep or new rescue module in this pilot. Any follow-up must
be a separately frozen experiment with new data. No automatic GO threshold,
non-inferiority claim or broad method-rejection rule is inferred from these
small counts.

## 8. Timing and artifacts

Time all deployable arms on the **same outside-panel development image and
question, device, dtype, backend and generation cap**, with full and R-res
included. Use 3 warm-up complete calls per arm followed by 10 paired blocks,
rotating the arm order deterministically by block to reduce ordering bias.
Synchronize CUDA before/after wall-clock windows. Report median, individual
block times and range; a single-page measurement is a local timing diagnostic.
Measure preprocessing, vision, score/selection/clustering/size preparation,
bias preparation, decoder prefill and true end-to-end TTFT through first-token
logits/choice. Report fixed-length 16-token generation separately from
EOS-stopped QA; use the same ignore-EOS rule for every timed fixed-length arm.
Account for per-decode bias construction and any explicit-mask backend cost.

Shared vision/selection caches may be used to make accuracy-arm inputs exact,
but warm cached isolated decoder times are not end-to-end deployment times.
Rerun the required vision/selection work within each full timed call; disclose
amortized/shared work separately. Report native and maskzero latency because
changing the SDPA mask/kernel can dominate a tiny bias's arithmetic cost. Do
not infer speedup from FLOPs or equal K. R-res reduces encoder/preprocessing
work as well as decoder tokens and can erase the practical motivation.

Keep data/run manifests, config/source hashes, CPU gates, GPU gates, all failed
audits, all predictions and paired timing blocks under
`Qwen_vl/outputs/attention_calibration/`. Record exact software/model revision,
device, actual attention backend, native rotary config, calibration support,
random seeds and source code hash before main generation. This protocol file
is not amended after main QA outcomes are read; implementation deviations and
their timing are recorded separately in the results report.
