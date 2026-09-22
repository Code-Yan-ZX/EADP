# S2-C5A — Checkpoint / Metric Consistency Audit

**Scope.** A methodological audit of S2-C5, run before S2-C5's verdict is allowed
to stand. It trains nothing new, changes no architecture, adds no feature and
re-tunes no hyperparameter. It answers two questions:

1. S2-C5 supervised on `HEAD_RANK` and reports **held-out teacher Top-8
   recall@256** as its primary metric, but selected every checkpoint on
   **validation Top-256 overlap**. S2-C2 and S2-C3 both recorded those two
   quantities in tension. Is S2-C5's ~0.795 landing point a property of the
   token-local family, or an artefact of the selection rule?
2. S2-C4 quotes `HEAD_RANK = 0.7783` and S2-C5 quotes `HEAD_RANK = 0.7656` for
   the same arm. Which aggregation produced which, and are they comparable?

**Status.** Complete. Scripts:
`Qwen_vl/scripts/discovery/s2c5a_{train,consolidate,aggregation}.py`, runner
`run_s2c5a.sh`. Deliverables:
`outputs/discovery/s2c5a_audit.json` (main), `s2c5a_train.json`,
`s2c5a_aggregation.json`, `s2c5a_perimage.npz`, `s2c5a_scores.npz`.

---

## HEADLINE

```
S2-C5A -- the verdict survives, one of its side-claims does not.

(a) THE CHECKPOINT OBJECTION IS REAL AND SMALL.  Selecting on validation
    Top-256 overlap rather than on the primary metric (validation teacher
    Top-8 recall@256) picks a checkpoint 0.7-3.3 epochs later, and it costs
    every arm held-out head_recall@8: +0.0042 to +0.0194 when re-selected on
    the primary metric.  The pre-registered verdict is unchanged under both
    rules: OV -> NONLINEAR-LOCAL, H8 -> NONLINEAR-LOCAL.

(b) THE CORE PRACTICAL CLAIM DOES NOT DEPEND ON THE RULE.  Under head
    selection nonlinear token-local still clears the shared linear reference
    reproducibly (LOCAL-MLP 0.8047, +0.0392 [+0.0228, +0.0567]) and no
    contextual arm beats it by the margin.  Head-selected LOCAL-MLP (0.8047)
    and SET-CTX (0.8019) remain below 0.82.

(c) THE STRONGER ORDERING CLAIM DOES NOT SURVIVE.  S2-C5 states that
    GLOBAL-CTX lands 0.0186 BELOW the local arm with a CI against the
    reference that includes zero.  Both halves of that are selection
    artefacts.  Under head selection GLOBAL-CTX reaches 0.7978 -- +0.0322
    [+0.0183, +0.0467] over the reference, i.e. it CLEARS the reference --
    and it is statistically indistinguishable from LOCAL-MLP (-0.0069
    [-0.0194, +0.0050]).  Selecting on overlap, which for GLOBAL-CTX picked
    an epoch 3.3 earlier on average than the head criterion, is what made it
    look worse than it is.

(d) THE AGGREGATION QUESTION IS SETTLED AND BENIGN.  S2-C4's 0.7783 and
    S2-C5's 0.7656 are the same three checkpoints on the same 150 images under
    two different aggregation protocols, neither of which is wrong.  S2-C5
    never mixed them.
```

---

## 1. Why a rerun was needed

`S2-C5` stored each run's full per-epoch validation curve (`val_history` in
`s2c5_train.json`) but saved only the weights at the **overlap** argmax. The
head-argmax epoch was therefore already readable from the published artifacts;
what was missing was the weights at that epoch. Since no per-epoch checkpoint was
kept, the audit re-runs the ladder with the training loop byte-identical to
`s2c5_train.main` -- same target, loss family, optimizer, batch order, RNG draw
order, `PATIENCE = 20` and `MAX_EPOCHS = 80` -- and tracks **two** checkpoints
inside the single resulting trajectory:

| tag | criterion | what S2-C5 did |
|-----|-----------|----------------|
| **OV** | argmax over epochs of validation `overlap256` | this one |
| **H8** | argmax over epochs of validation `head_recall8` | the audit's |

Because both are indices into one trajectory, the OV/H8 comparison is exactly
paired: weights path, data order and epoch budget are held fixed and only the
selection rule moves. The *training* loop was not re-stopped, and it does not
need to be -- validation `head_recall@8` peaks at epoch 0-2 in every run while
the runs halt at epoch 22-27, so `best_H8_epoch + PATIENCE <= epochs_run` holds
in all 12 runs (recorded per run as `head_patience_reached_within_run`). A
head-patience halt would have returned the same argmax.

**Validation only.** Both criteria read the 60-image validation split. No
held-out quantity is computed inside the epoch loop; the 150 held-out images are
scored once, after the checkpoint is frozen.

**Reproduction.** Deterministic kernels are forced for the audit, so it is
exactly re-runnable (S2-C5's own trajectory drifts at ~1e-8 per epoch from GPU
reduction order). The OV-selected arm reproduces the published run's held-out
metrics to a worst-case absolute difference of **0.0017** in `head_recall@8` and
**0** in the selected epoch, across all 12 runs -- so the trajectory audited here
is the trajectory S2-C5 measured.

---

## 2. Both selections, every arm

`head_recall@8/16/32` and `overlap256` are held-out (150 images), seed-averaged
over seeds 0/1/2. `val R@8` is validation `head_recall@8` at the selected epoch.

| arm | sel | epoch (s0,s1,s2) | val R@8 | held-out R@8 | R@16 | R@32 | ov256 |
|-----|-----|------------------|---------|--------------|------|------|-------|
| LOCAL-MLP | OV | 3, 1, 1 | 0.7569 | 0.7972 | 0.7719 | 0.7188 | 0.5220 |
| LOCAL-MLP | **H8** | 0, 1, 0 | 0.7806 | **0.8047** | 0.7751 | 0.7135 | 0.5078 |
| GLOBAL-CTX | OV | 3, 5, 6 | 0.7556 | 0.7783 | 0.7533 | 0.7117 | 0.5257 |
| GLOBAL-CTX | **H8** | 1, 2, 1 | 0.7729 | **0.7978** | 0.7668 | 0.7152 | 0.5203 |
| SET-CTX | OV | 1, 1, 1 | 0.7632 | 0.7978 | 0.7722 | 0.7204 | 0.5222 |
| SET-CTX | **H8** | 0, 1, 0 | 0.7750 | **0.8019** | 0.7738 | 0.7152 | 0.5145 |
| LOCAL-MLP-WIDE | OV | 1, 2, 2 | 0.7590 | 0.7947 | 0.7669 | 0.7156 | 0.5185 |
| LOCAL-MLP-WIDE | **H8** | 0, 0, 0 | 0.7778 | **0.8081** | 0.7747 | 0.7107 | 0.5038 |

Per-seed held-out `head_recall@8` under each selection, and the paired
`H8 - OV` difference (10 000-draw paired bootstrap over the 150 images,
seed-averaged, positive = head selection is better):

| arm | per-seed OV | per-seed H8 | H8 - OV | 95 % CI | all seeds + |
|-----|-------------|-------------|---------|---------|-------------|
| LOCAL-MLP | 0.7800 / 0.8133 / 0.7983 | 0.7992 / 0.8133 / 0.8017 | **+0.0075** | [-0.0050, +0.0200] | no |
| GLOBAL-CTX | 0.7958 / 0.7783 / 0.7608 | 0.8000 / 0.7917 / 0.8017 | **+0.0194** | [+0.0061, +0.0322] | yes |
| SET-CTX | 0.8025 / 0.8075 / 0.7833 | 0.8000 / 0.8075 / 0.7983 | **+0.0042** | [-0.0050, +0.0133] | no |
| LOCAL-MLP-WIDE | 0.8075 / 0.7967 / 0.7800 | 0.8125 / 0.8117 / 0.8000 | **+0.0133** | [-0.0058, +0.0328] | yes |

Two things to read here. First, head selection moves **every** arm up, so the
S2-C5 numbers are mildly pessimistic rather than optimistic -- the selection rule
was not inflating the result. Second, the size of the move is arm-dependent:
GLOBAL-CTX gains 0.0194 (the only arm whose gain has a CI excluding zero) and
0.0133 for the wide arm, against 0.0075 and 0.0042 for LOCAL-MLP and SET-CTX.
GLOBAL-CTX is the arm whose overlap-argmax sits furthest from its head-argmax
(3.3 epochs), and it is the arm whose published number was most affected.

The tension shows up directly in the last column: under head selection every
arm's held-out `overlap256` **falls** (LOCAL-MLP 0.5220 → 0.5078, SET-CTX
0.5222 → 0.5145), while its `head_recall@8` rises. That is exactly the trade
S2-C3 recorded -- checkpoint selection was buying bulk overlap with head recall
-- and it is why an overlap-selected checkpoint cannot be assumed to be the
head-recall optimum.

**The two criteria are correlated but not the same.** Spearman correlation
between the validation overlap and validation `head_recall@8` curves, computed
over epochs within each run, is +0.65 to +0.76 across the four arms -- strongly
positive, which is why the damage is small, but far from 1, which is why the
objection was worth testing. Overlap is not a proxy good enough to select on
when the primary metric is head recall, and it is now measured rather than
assumed either way.

**Caveat on the head-selected checkpoints.** Validation `head_recall@8` peaks
early -- epoch 0-2 in all 12 runs -- and declines monotonically after, while
overlap keeps climbing, its argmax landing at epoch 1-6. So H8 selection is, in
effect,
choosing a nearly-untrained checkpoint, and part of what it buys may be
*less fit-split overfitting* rather than a better-chosen epoch. The selection is
also noisy: the validation split has 60 images and `head_recall@8` per image is
granular at 1/8, so the curve's resolution is 1/480 and its argmax is not a
precise estimate. This is visible in the result -- the H8 gains for LOCAL-MLP
(+0.0075) and SET-CTX (+0.0042) have CIs spanning zero, and only GLOBAL-CTX's
(+0.0194) excludes it. The direction of the effect is consistent across all four
arms and all twelve runs, which is why it is reported as a real effect on the
level, but it should be read as "S2-C5's checkpoint rule was mildly
pessimistic", not as "head selection is a reliable +1 pt".

---

## 3. The 0.7783 vs 0.7656 discrepancy

Both numbers are `HEAD_RANK` on the held-out 150, from the same three S2-C3
checkpoints. They differ only in **where the mean over seeds is taken**.
Per-seed `head_recall@8` is 0.7608 / 0.7775 / 0.7583; the per-seed score vectors
were verified to reconstruct exactly from the checkpoints (max abs error
2.1e-06), so all three protocols below score identical rankings per seed.

| protocol | definition | value |
|----------|------------|-------|
| **A** metric-then-mean | `mean_i mean_s head_recall@8(score_{s,i}, teacher_i)` | **0.7656** |
| **B_raw** mean-of-raw-weights | `mean_i head_recall@8(mean_s score_{s,i}, teacher_i)` | 0.7742 |
| **B_unit** mean-of-unit-dirs | `mean_i head_recall@8((mean_s ŵ_s) · h_i, teacher_i)` | **0.7783** |

- **S2-C5 quotes A = 0.7656.** Its arms are also averaged the A way, and its
  reference is `ref_pi_mean`, the A reading. Verified against
  `s2c5_controls.json`, which recorded both readings explicitly
  (`reference_seed_mean_head_recall8 = 0.7656`,
  `reference_seedavg_scorer_head_recall8 = 0.7742`) and compared on the A one.
  **S2-C5 did not mix protocols.**
- **S2-C4 quotes B_unit = 0.7783.** Its `HEAD_RANK_seedavg` arm averages the
  three *unit* directions and re-scores, which is the protocol that treats the
  seeds symmetrically for a direction comparison.
- **B_raw = 0.7742** is what `s2c5_common.seed_mean_scores` computes, because
  the cached score vectors are `w_s · h + b_s` with seed-dependent `‖w_s‖`
  (0.9237 / 0.6502 / 0.8387). The raw mean is therefore a norm-weighted average
  of the three directions. `s2c5_common`'s docstring says score-averaging and
  direction-averaging "agree to within the seed spread"; they agree to 0.0042,
  which is the whole correction.

Decomposition of the 0.0128 total gap:

| component | value | 95 % CI |
|-----------|-------|---------|
| metric order (A − B_raw) | −0.0086 | [−0.0161, −0.0006] |
| unit normalisation (B_raw − B_unit) | −0.0042 | [−0.0100, +0.0025] |
| **total (A − B_unit)** | **−0.0128** | **[−0.0214, −0.0039]** |

So the discrepancy is **mostly** "seed-specific metric then average" versus
"seed-averaged weights/direction then re-scoring" -- that component is resolved
-- with a smaller, unresolved component from unit-normalising the directions
before averaging.

**Apples-to-apples.** The two stages answer different questions and their
headline `HEAD_RANK` numbers are not comparable to each other:

- against S2-C4's protocol, the S2-C5 ladder's reference is **0.7783**;
- against S2-C5's own protocol -- and the one this audit uses throughout, so
  that every delta is computed the same way -- the reference is **0.7656**.

Every number in §2 and §4 is protocol A. Under protocol A the reference is
0.7656, which matches the value S2-C5's rule was written against.

---

## 4. Verdict re-application

The S2-C5 rule was re-applied unchanged (`MARGIN = 0.01`, paired bootstrap over
the 150 held-out images, all three seed-matched deltas positive) to both
selections. The wrong-image context control was recomputed with its own paired
CI rather than inherited.

| | OV (what S2-C5 used) | H8 (primary-metric selection) |
|---|---|---|
| LOCAL-MLP | 0.7972 | **0.8047** |
| SET-CTX | 0.7978 | **0.8019** |
| GLOBAL-CTX | 0.7783 | **0.7978** |
| LOCAL-MLP-WIDE | 0.7947 | **0.8081** |
| HEAD_RANK reference | 0.7656 | 0.7656 |
| **verdict** | **NONLINEAR-LOCAL** | **NONLINEAR-LOCAL** |

Under **OV**, the pairwise deltas reproduce S2-C5: GLOBAL-CTX is −0.0189
[−0.0311, −0.0064] below LOCAL-MLP and does *not* clear the reference
(+0.0128 [−0.0031, +0.0286]); SET-CTX is +0.0006 [−0.0075, +0.0086] over
LOCAL-MLP; the wide arm is −0.0025 below it.

Under **H8** every one of those separations dissolves:

```
GLOBAL-CTX - LOCAL-MLP     -0.0069 [-0.0194, +0.0050]   not resolved
SET-CTX    - LOCAL-MLP     -0.0028 [-0.0103, +0.0042]   not resolved
SET-CTX    - GLOBAL-CTX    +0.0042 [-0.0064, +0.0150]   not resolved
WIDE       - LOCAL-MLP     +0.0033 [-0.0044, +0.0111]   not resolved
```

All four arms land in 0.798-0.808 (a 1.0 pt band), all clear the 0.7656
reference, and none separates from another. The mechanism controls still fail to
detect context use under either selection: honest − wrong-image `head_recall@8`
is −0.0008 [−0.0092, +0.0069] for GLOBAL-CTX and −0.0006 [−0.0094, +0.0078] for
SET-CTX. Replacing each image's context with another image's changes nothing,
so the contextual arms were not reading sample-specific token configuration --
that conclusion is selection-invariant.

### Requirement-5 antecedent

| condition | under OV | under H8 |
|-----------|----------|----------|
| LOCAL-MLP clears the reference | yes | yes |
| LOCAL-MLP ≈ SET-CTX | yes | yes |
| no contextual arm beats LOCAL-MLP by ≥ 1 pt | yes | yes |
| SET-CTX > GLOBAL-CTX (resolved) | yes | **no** |
| GLOBAL-CTX clears the reference | no | **yes** |
| best matched arm < 0.82 | yes | yes |

**The practical verdict is maintained**, and the two conditions that matter for
it hold under both selections: nonlinear token-local improves reproducibly over
shared linear scoring, and the tested contextual arms provide no independent
gain. Head-selected LOCAL-MLP (0.8047) and SET-CTX (0.8019) remain below 0.82.

**But the antecedent is not *fully* met, and the difference is reported rather
than rounded away.** The strict ordering `LOCAL-MLP ≈ SET-CTX > GLOBAL-CTX`
held under OV and does not hold under H8: the ">" is gone. GLOBAL-CTX is level
with the token-local arms, not below them.

---

## 5. What this changes in S2-C5, and what it does not

**Corrected in S2-C5:**

> "GLOBAL-CTX ... is 0.7786 -- 0.0186 BELOW the local arm [−0.0311, −0.0061],
> with a CI against the reference that includes zero."

Both halves are selection artefacts. Under primary-metric selection GLOBAL-CTX
reaches 0.7978, clears the reference by +0.0322 [+0.0183, +0.0467], and is
within noise of LOCAL-MLP. S2-C5's number for this arm should be read as a
lower bound produced by an overlap-argmax checkpoint, not as the arm's level.

*(S2-C5's own run quotes 0.7786 for this arm where the audit's reproduction of
its OV setting gives 0.7783; the 0.0003 is the GPU-reduction drift noted in §1,
and it is largest on this arm because GLOBAL-CTX_s2 is the one run whose
selected epoch sits late enough for the drift to perturb a few token orderings.)*

**Unchanged in S2-C5 (survives the audit):**

- the pre-registered verdict, **NONLINEAR-LOCAL**, under both selection rules;
- nonlinear token-local beats the shared linear probe: +0.0317 (OV) and +0.0392
  (H8) over the reference, CI excluding zero, all three seed-matched deltas
  positive;
- no contextual arm adds an independent gain over the token-local arm;
- width is not the binding constraint: the 4.5× wide arm is within noise of the
  matched local arm under both rules;
- the contextual arms do not read sample-specific context (wrong-image control
  null under both rules);
- the token-local family lands below 0.82 under both rules, so the downstream
  gate stays closed and no generation is run.

**Not claimed, here or by S2-C5 on this evidence:**

- that context is unnecessary *in general*;
- that ~0.80 is the ceiling of every possible token-local function.

The evidence supports: *under the current L4 representation, training setup and
tested model family, adding width or simple global/set context yields no further
gain.*

---

## 6. Open item this audit did not close

Under H8 the four arms are mutually indistinguishable while all beating the
reference by 3.2-4.3 pt. That is consistent with the S2-C5 reading (a shared
nonlinear token-local function is what the linear probe was missing) but it is
also consistent with the contextual arms being *capable* of the same gain and
simply not using their context -- which is exactly what the null wrong-image
control shows. This audit does not separate "context is not needed" from
"context was available and unused", and neither does S2-C5. That distinction is
left open.
