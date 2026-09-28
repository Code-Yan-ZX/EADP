# Set-Conditioned Answer-Gain Exchange for Fixed-Budget Visual Token Pruning

**Method.** Set-Conditioned Answer-Gain Exchange (SAGE)

## Motivation
**Problem framing.** In fixed-budget hard visual-token pruning, selecting K original visual tokens is a complete-set decision. The final answer may change when one piece of evidence is removed alongside another, so cheap pre-LLM salience, coverage, or token-wise training signals need not rank equal-budget sets by their actual answer effects. Phase 1 identifies this mismatch at the problem-class level. CoverPruner (openalex:W7208716036) represents discarded projected tokens through query-weighted coverage, while CoRePrune (openalex:W7214127050) recomputes set-conditioned perturbation only after visual–text interaction. Neither supplied result establishes an early, answer-level ordering of complete hard sets with matched end-to-end latency.

The measured instance is OCR-heavy Qwen3-VL-8B with the existing EADP B2 selector as incumbent. Local M2–M9 tests show that cheap pre-LLM and later interaction-based proxies have not yielded a reliable answer-and-TTFT win, and the B2/U12 candidate union can lose accuracy when it shrinks the B2 core. AutoSelect (openalex:W7134841662), MAP (openalex:W7202121322), CRISP (openalex:W7169880607) and TOPS (openalex:W7166090344) each improve a selector or proxy, yet the specified residue is whether an early signal can predict the answer effect of replacing one whole equal-budget set with another. The old local panels motivate the hypothesis and cannot certify it; fresh image-disjoint confirmation is required.

**Why now.** Recent CoRePrune (2026-09; openalex:W7214127050) makes set dependence explicit but pays for post-interaction evidence, while CoverPruner (2026-09; openalex:W7208716036) and MAP (2026-08; openalex:W7202121322) demonstrate increasingly targeted early proxies without the proposed paired official-answer target. The project already has a weight-accessible frozen Qwen3-VL-8B harness, official TextVQA/DocVQA/OCRBench scoring, the B2 incumbent and M6 nomination scores, so sampled hard-set counterfactuals and matched TTFT can now be measured with the same decoder and split discipline. That availability makes this narrow empirical test possible; it does not imply that the early critic will generalize.

**Why prior work stopped.**
- `openalex:W7214127050` (arXiv (Cornell University) 2026): CoRePrune (openalex:W7214127050) refreshes vision-depth deletion perturbations and recomputes rescue marginals from question-token attention outputs under the current deletion set.
  - _Did not do_: It does not establish pre-LLM prediction of paired final-answer changes for complete equal-budget hard exchanges with matched end-to-end TTFT.
  - _Structural reason_: Its conditional measurement uses visual–text interaction and an attention-output distortion objective rather than an early official-answer difference.
- `openalex:W7208716036` (arXiv (Cornell University) 2026): CoverPruner (openalex:W7208716036) selects original projected tokens by query-weighted coverage using a lightweight first-layer attention probe.
  - _Did not do_: It does not directly label competing complete retained sets by the frozen decoder's paired official answer-score change.
  - _Structural reason_: Its optimized object is representation coverage in the chosen feature space, which remains a proxy for downstream answer value.
- `openalex:W7134841662` (arXiv (Cornell University) 2026): AutoSelect (openalex:W7134841662) trains a visual-token scorer through noise-gated full-sequence next-token loss and deploys hard Top-K selection.
  - _Did not do_: Its deployed scorer does not condition a question-specific exchange prediction on the surviving hard set.
  - _Structural reason_: A token-wise, text-agnostic deployed score and soft training gate do not directly supervise hard-set substitution effects.
- `openalex:W7202121322` (arXiv (Cornell University) 2026): MAP (openalex:W7202121322) distills sample-selected middle-layer text-to-visual attention into a pre-LLM predictor and combines predicted attention with diversity.
  - _Did not do_: It does not train its early predictor on paired official answer differences between complete fixed-budget sets in the named Qwen3-VL setting.
  - _Structural reason_: Its learning target remains an attention distribution, leaving a possible teacher-proxy mismatch with answer-level set value.
- `openalex:W7169880607` (arXiv (Cornell University) 2026): CRISP (openalex:W7169880607) selects pre-LLM text-aligned visual evidence and then adds semantic context.
  - _Did not do_: It does not estimate whether one specified whole hard-set exchange improves the frozen decoder's scored answer.
  - _Structural reason_: Its relevance and diversity heuristics have no paired answer-change supervision for the proposed exchange decision.
- `openalex:W7166090344` (arXiv (Cornell University) 2026): TOPS (openalex:W7166090344) forms a training-free preservation set from task relevance, information coverage and semantic diversity.
  - _Did not do_: It does not establish answer-level ordering of equal-budget complete sets under the project's matched latency protocol.
  - _Structural reason_: Its preservation criteria are proxy principles rather than measurements of the frozen decoder's paired final-answer response.

**What changes when the gap closes.** If fresh-split results support the prediction, a fixed-budget selector can use a pre-LLM estimate of the final-answer effect of a specific hard-set substitution while retaining one decoder pass. This would provide an answer-level alternative to the coverage targets of EADP and CoverPruner (openalex:W7167223567; openalex:W7208716036) and to CoRePrune's post-interaction distortion target (openalex:W7214127050) for the tested Qwen3-VL setting. The result would not prove general recovery or transfer beyond the documented proposal edge family and task mixture.

## Method
**Pipeline.** The frozen Qwen3-VL-8B processor and projector turn an image and question into projected visual tokens and the existing EADP B2 retained set. A pre-LLM generator proposes equal-size group removals and insertions around that set, then a shared critic reads the surviving set, both groups and the question to predict each complete exchange's official answer-score change. Fit labels come from paired frozen-decoder generations and official task scoring; separate image-disjoint validation selects proposal and critic settings under an end-to-end TTFT constraint. At deployment, only the best edge above the validation threshold replaces B2, and the unchanged decoder sees one ordered K-token set. Fresh locked TextVQA, DocVQA and OCRBench generation, paired intervals and mechanism controls determine whether the hypothesized gain survives.

### Background
*Fix the existing model, B2 baseline and official scoring setup used by every paired comparison.*

1. **Freeze the incumbent harness** (`S1`)
   - Fix the weight-accessible Qwen3-VL-8B processor, projector, decoder, B2 selector, decoding settings and official task scorers. For each image and question, obtain projected visual tokens V and B2 set S0; run the incumbent decoder to get baseline answers where paired fit or validation labels require them, without running a baseline answer at deployment.
   - _Why:_ A frozen, shared decoder and scoring harness make a proposed hard-set exchange the only intervention measured by the paired outcome.

### Equal-budget exchange candidates
*Propose finite group replacements around B2 while keeping exactly K original visual tokens.*

2. **Generate feasible group exchanges** (`S2`)
   - Rank removal seeds from low EADP scores within S0 and insertion seeds from high unretained EADP and existing M6 cosine nomination scores. Around each seed, use its nearest eligible token-grid neighbors to form a size-g group, restrict each side to its proper set, deduplicate the Cartesian product, and construct Se by equal-size replacement; m and g are chosen on image-disjoint validation under the latency constraint. The source specifies index ties for neighbor distance but has not defined score-ranking ties or the grid distance, so literal determinism remains an implementation-audit obligation.

*A feasible removal and insertion edge replaces equally many original visual tokens in the B2 set, so the selected set retains K tokens; g is selected on image-disjoint validation from the source's admitted range, whose endpoint with an empty survivor remains unresolved for critic computation.*
$$ e=(G_{-},G_{+}),\quad G_{-}\subseteq S_0,\quad G_{+}\subseteq {1,\ldots,N}\setminus S_0,\quad |G_{-}|=|G_{+}|=g,\quad S_e=(S_0\setminus G_{-})\cup G_{+},\quad |S_e|=K \tag{1} $$

   - _Why:_ The finite E(x) support makes answer-level counterfactuals measurable while equal removal and insertion preserve the hard K-token budget.

### Set-conditioned answer-gain critic
*Learn paired official answer-score changes from surviving and exchanged token groups.*

3. **Label paired answer effects** (`S3`)
   - On image-disjoint fit images, uniformly sample edges from E(x), generate once for S0 and once per sampled Se with the same frozen decoder and decoding configuration, score both generated answers with the task's official TextVQA, DocVQA or OCRBench scorer scaled to [0,1], and record their paired difference Delta(x,e). Gold answers are used only for offline labeling.

*On fit images, the same frozen decoder and task-specific official scorer compare the generated answer from a complete exchanged set with the B2 answer; z denotes the gold answer and is never available at deployment.*
$$ \Delta(x,e)=a_b\bigl(y(S_e),z\bigr)-a_b\bigl(y(S_0),z\bigr) \tag{2} $$

   - _Why:_ The training target reflects the actual final-answer effect of each complete hard exchange rather than an attention, coverage or caption proxy.
4. **Fit the survivor-conditioned critic** (`S4`)
   - Concatenate projected-vector means of the surviving S0 minus Gminus set, removed group and added group, the mean question input embedding, and mean/max cosine summaries from each exchange group to survivors. Fit one small MLP $f_{theta}$ by squared error against Delta on the uniformly sampled fit edges; choose its capacity on validation. The source has not settled the cosine aggregation convention, zero-denominator behavior, or the admitted g=K empty-survivor case, so these are blockers for exact feature construction rather than grounds to invent a replacement operator.

*The fixed-length critic uses projected means of survivors and both exchange groups, mean question input embedding, and mean/max group-to-survivor cosine summaries. The source leaves the cosine aggregation convention and zero-vector handling unspecified, and the admitted empty-survivor endpoint makes these features undefined.*
$$ \phi(x,e)=\operatorname{concat}\!\left(\overline v_{S_0\setminus G_{-}},\overline v_{G_{-}},\overline v_{G_{+}},\overline q,\mu_{-},\max_{-},\mu_{+},\max_{+}\right),\quad f_{\theta}(x,e)=\operatorname{MLP}_{\theta}(\phi(x,e)) \tag{3} $$


*The shared critic fits paired official-score differences on uniformly sampled feasible proposal edges. At a population optimum, a sufficiently expressive squared-loss model targets the conditional mean under the stated fit mixture and edge sampler, not arbitrary K-subsets.*
$$ \theta^{\star}\in\operatorname*{arg\,min}_{\theta}\;\mathbb{E}_{x,\,e\sim\operatorname{Unif}(E(x))}\!\left[\bigl(f_{\theta}(x,e)-\Delta(x,e)\bigr)^2\right] \tag{4} $$

   - _Why:_ Conditioning on survivors and both exchange groups lets the predictor represent set-dependent answer effects that a unary added-minus-removed score cannot encode by construction; whether it learns useful effects remains empirical.

### Calibrated early selection
*Choose a latency-aware threshold on validation and make one predecoder exchange decision.*

5. **Calibrate on separate images** (`S5`)
   - Use image-disjoint validation to select critic capacity, seed count m, exchange size g and strict acceptance threshold tau by official macro accuracy subject to the measured end-to-end TTFT deployment constraint; freeze the resulting proposal generator and critic before confirmation.
   - _Why:_ The deployment decision must account for generator and critic overhead while avoiding selection on the locked confirmation split.
6. **Select once before decoding** (`S6`)
   - At inference, construct E(x) before the first decoder block, score each feasible edge with $f_{theta}$, emit the best Se only if its prediction strictly exceeds tau, otherwise retain S0, sort selected original-token indices and run the unchanged Qwen3-VL-8B decoder once. No gold answer, teacher gradient or extra decoder pass is used online; the source still needs rules for tied best edges and an empty E(x).

*Deployment evaluates the finite proposal set before the first decoder block, accepts its best edge only above the validation-selected threshold tau, and otherwise keeps B2; ties among maximal edges and empty proposal lists still need source-level rules.*
$$ e^{\star}\in\operatorname*{arg\,max}_{e\in E(x)}f_{\theta}(x,e),\qquad S_{\mathrm{out}}=\begin{cases}S_{e^{\star}},&f_{\theta}(x,e^{\star})>\tau,\\ S_0,&\text{otherwise.}\end{cases} \tag{5} $$

   - _Why:_ This is the load-bearing early exchange intervention whose actual official answer and TTFT impact the confirmation test must measure.

### Confirmation and controls
*Test official answer quality, TTFT and the mechanism against locked-split controls.*

7. **Confirm and try to falsify** (`S7`)
   - On a newly locked image-disjoint split, run official TextVQA, DocVQA and OCRBench generation and end-to-end TTFT for the candidate against B2, B1, matched-rate random exchange, the same-label unary exchange scorer and local M10 whole-set routing. Report task-level rescue/break counts and paired uncertainty intervals; include within-image edge-label permutation at matched deployed exchange rate and, only as a ceiling, hindsight best edge. Reject if paired macro accuracy fails to beat B2, a clear OCRBench loss appears, or the TTFT constraint fails; a high hindsight ceiling with a failed critic rejects the early-observability premise.
   - _Why:_ The controls distinguish a useful survivor-conditioned answer-gain model from exchange-rate effects, unary scoring and hindsight opportunity; fresh data adjudicate the actual contribution.

## Reviewer concerns
- **Concern [non_blocking]:** Paper-pointed threat: openalex:W7164123516 (lit_table). AVEX-Prune already uses swap-based reward estimates for audio-visual token pruning, so exchange supervision by itself cannot carry this proposal's novelty. The supplied record does not show AVEX deploying a predictor conditioned on the surviving hard visual-token set and trained on paired official answer-score changes from complete fixed-budget exchanges; that is the candidate's narrower claim. If AVEX in fact deploys that same conditional whole-edge predictor, the claimed method distinction falls, but the supplied evidence does not establish such overlap.
  - **Response:** The defense is anchored in core_mechanism, what_step_was_missed and differentiation_from_lit: AVEX-Prune supplies exchange-reward precedent, but the supplied record gives it an additive deployed token score rather than a survivor-conditioned predictor trained on paired official answer changes for whole visual-only fixed-budget exchanges. Phase 3.2's verdict_rationale treats this as a narrowed, testable distinction, not a demonstrated gain; if the fuller AVEX record shows the same deployed conditional whole-edge mechanism, the novelty claim must be revised.
- **Concern [non_blocking]:** Un-retrieved mechanism family flagged by the audit (parametric knowledge, not in the retrieved pool): Offline contextual bandit action-value regression for structured or slate actions is an older conceptual family not resolved by the retrieved online combinatorial-bandit hits; query 'slate reward model full-action feedback' and 'offline contextual combinatorial policy improvement' before investing. — novelty vs this family is UNVERIFIED; run a targeted scoop-check on that vocabulary before investing.
  - **Response:** A targeted search of offline contextual bandit and slate full-action reward modeling must determine whether an earlier method already combines survivor-conditioned whole-edge answer feedback, fixed-budget pre-LLM visual selection and the one-decoder-pass deployment constraint. Conceptual overlap may narrow the claimed domain contribution; the supplied record does not establish that this scoop-check has passed.

## Research status

Research proposal — not experimentally validated. Pipeline checks do not prove novelty or mathematical correctness.

## Main contribution

A survivor-set-conditioned pre-LLM predictor of paired official answer-score changes may select beneficial equal-budget group exchanges around EADP B2 on fresh OCR-heavy Qwen3-VL images while satisfying the measured end-to-end TTFT constraint.

## Minimal falsification

Freeze Qwen3-VL-8B and the edge generator after image-disjoint fitting and validation, then run TextVQA, DocVQA and OCRBench generation on a fresh locked confirmation split at N=1024 and K=256, recording official macro accuracy, per-task rescue/break counts and end-to-end TTFT against B2 and B1. If the mechanism works, macro accuracy rises over B2 while TTFT satisfies the validation-set deployment constraint; any clear OCRBench loss or failure to beat B2 on paired macro accuracy vetoes the method. The load-bearing variable is the set-conditioned predicted answer gain f_theta(x,e): permuting its edge labels within each fit image while matching the deployed exchange rate should erase the downstream macro-accuracy gain, whereas a unary added-minus-removed score trained on the same paired labels should recover less of it. A hindsight best edge may be reported only as a ceiling, and if it is high while the critic fails on fresh images, the early observability premise is falsified.

## Resources and feasibility

Original estimate: Estimated full falsification campaign: 35-70 80GB-class GPU-days for paired frozen-model generations, critic training, validation and fresh confirmation; \$0 paid API spend because the named model is self-hosted and official benchmark scorers are local. This fits the stated approximately 150 GPU-day envelope, with the final run count governed by a predeclared compute cap rather than by a mechanism constant.

Current estimate: Provisional 35–70 80GB-class GPU-days from the canonical estimate; actual full-campaign cost is unknown and could exceed the $\approx 150$ GPU-day ceiling without a fixed sampled-edge and validation run count. Paid API spend is expected to be zero for the stated self-hosted harness.

User ceiling: 80GB-class GPUs (A100/H100), up to 8 concurrent (one node), $\approx 150$ GPU-days over 5 months, plus ~\$10k inference/API budget for the full falsification campaign

Status: tight. For m removal seeds and at most 2m distinct insertion seeds, E(x) can contain up to $2m^{2}$ exchanges before deduplication. Each sampled fit edge needs an additional frozen-decoder generation, and $capacity/m/g/\tau $ validation plus B1, B2, random, unary, M10, permutation, confirmation and TTFT measurements add runs. Neither sampled edges per image nor generation throughput or validation grid size is fixed, so the source's 35–70 GPU-day estimate cannot be verified against $\approx 150$ GPU-days.

Change: The original estimate is preserved verbatim. No completed cost measurement or changed scope is present; confidence is downgraded because the dominant paired-generation count and throughput are unspecified.

## Open questions

token-grid distance and score-ranking tie rule: The source only fixes distance ties by original token index; g>1 and tied nomination scores need deterministic ordering.

sampled edges per fit image: Uniform sampling is specified but its count drives label cost and edge coverage.

For the admitted g=K endpoint, S0 minus Gminus is empty, so its projected mean and group-to-survivor cosine maximum are undefined. For nonempty survivors, cosine aggregation and zero-norm behavior are also unspecified.: The normal g=1 toy has computable means and cosines, but the source explicitly admits g=K=256. The trace cannot construct its prescribed feature vector at that endpoint; no critic was trained.

MLP width: The source already selects width on image-disjoint validation.

TTFT bound and no-feasible-setting rule: The source names a deployment TTFT constraint but not its numerical reference or fallback.

argmax tie and empty E(x) behavior: Neither case has a source-level decision rule.

confirmation sample sizes, paired interval procedure and clear OCRBench-loss criterion: These are pre-registration choices needed for a reproducible verdict, not new mechanism operations.

