# Semantic-difference mining: pilot protocol

Frozen before new model predictions, 2026-11-08 (Asia/Shanghai).

This tests the user's proposal: find answer-critical visual changes that a
full model distinguishes and its compressed counterpart fails to preserve,
then train on the current failures. It does not presuppose a module combination.

## Execution and scope

Use cached Qwen3-VL-8B-Instruct, native E0 DeepStack and 3-D mRoPE, a single
A40, 1024 visual input tokens and 256 output tokens. The first trainable
carrier is B2 anchor selection plus a learned shared attention pooling map
for the main feature stream and all three DeepStack streams. Its zero gate
must reproduce B2. This architecture is an experimental carrier, not the
novelty claim. Historical native DocVQA evidence favors resolution reduction
and VisionZip over B2; beating this carrier alone cannot establish superiority
to strong compression baselines.

Generate one-digit changes on real DocVQA pages using unique OCR matches to
numeric answer strings. Rerender both old and new values in the same box and
font. Preserve untouched pages for real-document evaluation. Check complete
model generation on BOTH members, so OCR matching is not treated as proof of
the answer. Split by source image/document before teacher/student screening.
This is a pilot constructed from DocVQA VAL, not an official train/val result.

## Gates and outcomes

1. Report all attempted edits, exclusions, teacher-valid pairs, compression
   failures (not both correct), and strict answer collapses (same output).
   Correctness uses normalized exact numeric strings, with no tolerance that
   can turn the single-digit edit back into the same answer. ANLS is secondary.
2. Verify initial identity, nonzero training gradients, and unchanged frozen
   decoder weights. Attempt to overfit 8–16 mined pairs. Failure to learn is
   an unresolved carrier/optimization issue, not rejection of the general idea.
3. If the carrier learns, compare equal initialization and update budgets:
   ordinary teacher distillation on old values; random teacher-valid edited
   pairs; fixed initial failure pairs; periodically refreshed current failures.
   Distillation/answer loss stays identical across these arms. Replay the same
   mined multiset using ordinary training to locate the source of any benefit.
4. Evaluate generation, both-correct pair accuracy, untouched held-out real
   pages, and unrelated-field questions when available. Keep held-out results
   out of mining/checkpoint selection. An image-disjoint pilot is not evidence
   of unseen-layout generalization; that requires a separate future dataset.

Use fixed final checkpoints, log seeds, actual optimizer updates, candidates
searched, mining frequency, wall time, and per-example results. Bootstrap
paired differences by source image where sample size permits. No success
claim based only on training loss or average visual-feature error.

Implementation details fixed before training: seed 20261108, up to 120
screened edit pairs, 32 teacher-valid fit candidates, eight selected hard
pairs per round, a 120-update learnability gate, then 160 updates per control
arm with two answer exposures per update and refresh every 20 updates. Use
up to 32 held-out valid pairs and 24 untouched held-out pages. Fixed-hard and
online must start with the SAME failure-ranked initial hard set; only online
refreshes it. Prioritize exact collapsed outputs, and distinguish those from
the broader criterion of at least one incorrect member. Use common CE plus
top-64-plus-remainder bucket KL to the complete teacher for all fit answers,
including EOS. The replay uses the exact online exposure multiset shuffled
from the common initialization. These are single-seed pilot budgets, not a
claim of fully converged benchmark training.

### Trainability amendment after the first carrier failed

The first carrier completed its 120 updates: 0/8 training pairs both correct
before and after, with nonzero gradients and unchanged frozen weights. Its
complete artifacts are retained in `outputs/cf_mining/training_global_gate/`.
This failed learnability gate prevents an interpretable mining comparison.

Before running another gate, add a generic rank-32 low-rank value/output
readout to each of the four visual streams. Output projections start at zero,
so B2 initialization remains exact; nonzero value projections allow immediate
output gradients. Attention and native anchor coordinates remain shared. This
is an increased-capacity experimental carrier, not a method claim. Run the
same eight training pairs for 400 updates, selecting no checkpoint by held-out
performance. If trainable, use 400 updates for every comparison arm, keeping
the same data, loss, evaluation, and search protocol (refresh every 50 updates
to retain eight rounds). Retain and report the first carrier's failure.

### Locality / answer-relation trainability amendment

The global readout carrier also failed: after 400 updates it scored 0/8
both-correct, while all eight training pairs collapsed to one answer each.
Its artifacts remain in `outputs/cf_mining/training_global_readout/`. Falling
average loss did not imply learning the required image distinction.

Before the next gate, localize the shared attention with a fixed Gaussian
raster-distance bias (sigma 1.5 merged-token units), retaining the same rank,
anchors and zero-output B2 initialization. Add a sequence contrast loss
`softplus(1 - log p(correct|image) + log p(opposite_pair_answer|image))`, weight
one, to the existing CE+KL. Reuse the positive logits; one additional negative
answer forward is required per image. This combines two carrier/objective
changes for learnability and cannot attribute improvement between them.

Run the same eight training pairs for 400 updates. If trainable, compare
ordinary CE+KL on original values; random, fixed, and online edited pairs
with the same contrast objective; shuffled exact-data replay with that
objective; and an additional exact-data `replay_ce` using only CE+KL. All
have 400 optimizer updates and 800 positive-answer exposures. Contrast arms
also have 800 negative-answer forwards; disclose this extra computation and
do not call this an equal-total-compute trial. This is the final trainability
carrier screen in this pilot. Failure still leaves the general idea unresolved.

## Interpretation

Stable teacher-valid compression failures support the proposed training
signal. Overfit recovery supports trainability. Held-out gains against random
editing and fixed hard cases support model-dependent search/refresh only if
they survive uncertainty and matched budgets. Same-data replay can explain
gains as data selection rather than a special relation objective. This pilot
cannot establish novelty relative to ETC, VC-OPD, or CIRA without separate
primary-source review, or publication value without real QA/layout transfer.
