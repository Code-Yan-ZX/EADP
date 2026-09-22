# S2-C3 — Head-Ranking Retargeting Test

**Scope.** Stage-2 S2-C3. One question, and it is a fork rather than a design
step:

> **H-A** — layer-4 hidden states already carry what is needed, and S2-C1 simply
> optimised the wrong target (the bulk of the teacher's Top-256 instead of the
> head of its ranking);
> **H-B** — even with supervision aimed squarely at the teacher's ranking head, a
> token-local read-out of layer-4 hidden states cannot recover those tokens.

No final method is designed, no architecture is changed, no multi-layer fusion,
attention feature, mid-layer pruning integration or query-conditioning expansion
is built. The scorer is the S2-C1 `LIN_L4` — 4 097 parameters, one layer, one dot
product per token — and it stays that way for every arm. What moves is only the
**teacher reading the loss is computed against**.

**Status.** Complete. Scripts:
`Qwen_vl/scripts/discovery/s2c3_{common,train,eval,oracle,consolidate,figure}.py`.
Deliverables: `outputs/discovery/s2c3_retarget.json` (main), `s2c3_train.json`,
`s2c3_eval.json`, `s2c3_headmetrics.npz`, `s2c3_oracle.json`, `s2c3_pilot.json`,
`s2c3_scores.npz`, `figures/s2c3_head_retargeting.png`.

---

## VERDICT

```
S2-C3 GO-B -- head recall moves, downstream accuracy does not follow it.

The supervision question is answered YES and is resolved: aiming the loss at the
head of the teacher's ranking really does buy head retention. HEAD_BIN and
HEAD_RANK raise recall of the teacher's Top-8 inside the selected 256 from 0.736
to 0.763 / 0.766 (+2.75 [+0.81,+4.75] and +2.97 [+0.86,+5.17] against the exact
S2-C1 baseline, both clearing zero against two independent references), lift
exact Top-8 agreement from 0.192 to 0.261 / 0.266, and cut the mean number of
missed teacher Top-8 tokens per image from 2.11 to 1.89 / 1.88.

And it is paid for in the currency S2-C1 was gated on: Top-256 overlap *falls*
from 0.562 to 0.519 / 0.513, and the separability AUROC falls from 0.798 to 0.760
/ 0.754. The two objectives trade against each other, and the retargeted scorer
moves cleanly from one end of that frontier to the other.

Downstream, at a fixed Top-K@256 on the held-out 150, nothing moves. All three
head targets land between 56.53 and 57.03 macro against the baseline's 57.32;
none differs from it by more than 0.79 points and every paired-bootstrap CI
straddles zero. The same-configuration seed null on this design is +/-2 to 3
macro points, so the 3-point head-recall gain is an order of magnitude below what
this exchange rate needs in order to be visible.

So: the objective was misaligned, correcting it works on its own terms, and
correcting it is not worth anything measurable at this capacity. The bottleneck
is not "the loss was aimed at the wrong tokens".
```

Two readings that must be kept apart, because they point to different next steps:

* **The optimistic half.** This is the first time in the project that head
  retention has moved *from* the S2-C1 membership probe. S2-C0's attention
  proxies sat at 0.50–0.51 recall of the teacher's Top-8; a linear probe trained
  on Top-256 membership sat at 0.736; a probe trained *on the head* sits at 0.766
  with 7 points more exact Top-8 agreement. The representation is not indifferent
  to how it is supervised, and until now nothing had moved it.
* **The deflationary half.** The achievable movement is +2.9 points of head
  recall, which by S2-C2's own measured exchange rate (~+1.3 macro per recovered
  teacher Top-8 token in the first few) is worth about +0.2 to +0.3 macro. It was
  measured at -0.79 to -0.29. Both numbers are inside the noise floor. The
  verdict is therefore not "head-aware supervision fails" but "head-aware
  supervision at this capacity changes too little to matter".

---

## 0. Executive summary

1. **Retargeting works on the metric it targets — for two of the three head
   targets.** `HEAD_BIN` (positives = teacher Top-32) and `HEAD_RANK` (same
   positives, pair weights `∝ 1/(r+1)`) raise `head_recall@8` by **+2.75** and
   **+2.97** points against the cached S2-C1 `LIN_L4`, with CIs clearing zero
   against both the exact baseline and a seed-matched retrained `BASE`, and
   margins exceeding the metric's own 3-seed spread (1.8–1.9 points). Exact
   Top-8 agreement rises from **0.192 to 0.261 / 0.266** — a +7-point move whose
   CI is [+5.3, +8.6] / [+5.7, +9.1] (§3).
2. **`HEAD_MULTI` does nothing.** Grading the *wide* target by rank band —
   the same Top-256 positives as `BASE`, weighted 4.0/2.0/1.5/1.0 by tier —
   moves held-out `head_recall@8` by **+0.67 [−0.36, +1.72]**, inside seed noise.
   Combined with (1), the operative variable is the **positive set**, not the
   weighting: `HEAD_BIN` and `HEAD_RANK` differ only in pair weighting and are
   indistinguishable on every metric; `HEAD_MULTI` has `BASE`'s positive set and
   behaves exactly like `BASE` (§3.3).
3. **The gain is paid for out of Top-256 overlap, and the exchange is real.**
   `overlap256` falls 0.562 → **0.519 / 0.513** (CIs [−4.94,−3.63] and
   [−5.53,−4.21]) and AUROC falls 0.798 → 0.760 / 0.754. `head_recall@64` is
   flat (0.648 → 0.643) while `@8`/`@16` rise, so the scorer has not found *more*
   of the teacher's ranking — it has moved its budget from the middle of the
   ranking to the top of it. This is the first direct measurement in the project
   that the two objectives are in tension rather than nested (§3.4).
4. **Downstream, nothing moves.** On the held-out 150 at a fixed Top-K@256:
   `BASE` 55.75, `HEAD_MULTI` 56.76, `HEAD_RANK` 56.53, `HEAD_BIN` 57.03, against
   the cached baseline's 57.32. No arm differs from the baseline by more than
   **0.79 macro points**, every CI straddles zero, and the 3-seed ranges
   (2.4–4.1 points) overlap completely (§4.1).
5. **The design cannot resolve an effect of the size S2-C2's exchange rate
   predicts.** Running the *same* target under two seeds — a true null — moves
   macro by −0.22 [−4.28, +3.78] and +2.26 [−2.03, +6.67] macro points. The
   measured head gain is worth ~+0.2–0.3 macro by S2-C2's per-token value. A
   ±5-point CI cannot see that, and neither can this stage. The honest statement
   is "no downstream gain is detectable", not "the head gain is proven
   worthless" (§4.3).
6. **The arms change which answers are right without changing how many.** On the
   S2-C2 class-1 set (28 instances that `LIN_L4` gets wrong and the teacher gets
   right), `HEAD_RANK` rescues 9.0 against `BASE`'s 2.7 — and newly breaks 11.3
   against `BASE`'s 6.3. `HEAD_BIN` rescues 7.7 and breaks 8.7. Counted as
   instances the head arms finish +0.0 to +1.0 above the baseline's 89 correct of
   150, against a within-arm seed spread of ±2 — a wash, which is exactly the
   S2-C2 warning that these means are carried by a minority of flips (§4.2).
7. **Both halves of the trade are individually near-zero, which is why the sum
   is.** The head arms recover ~0.23 more teacher Top-8 tokens per image and give
   up ~4.4 points of Top-256 overlap, i.e. ~11 mid-tail tokens. S2-C2 measured
   the marginal value of a mid-tail swapped token at ~0 past position 32 and the
   value of the first recovered head tokens at ~+1.3 macro each — so ~+0.3 from
   the first half and ~0 from the second. The measured −0.29 to −0.79 is that
   prediction landing inside noise (§5.2).
8. **The head is linearly expressible from L4; the difficulty is transferring one
   direction.** A supplementary closed-form diagnostic (no training — a
   mean-difference direction fitted on the image itself) recovers **98.2 %** of
   the teacher's Top-8, against **92.1 %** for an arbitrary group of 8 fitted the
   same way and **23.7 %** for a direction fitted on an arbitrary group. So
   "the information is not in L4" is false. But the best *shared* direction the
   trained arms reach is 0.766. The 0.22 gap between those two numbers is where
   this stage's failure lives (§6).

---

## 1. What is held fixed

Everything except the target. The stage is a supervision experiment, so every
other degree of freedom is pinned to S2-C1's values:

| object | value | source |
|---|---|---|
| feature cache | layer-4 visual hidden states, 1 024 × 4 096 fp16, 465 instances | `s2c1_feats_L4.npy` |
| preprocessing | one fixed step: fit-split per-dimension mean/std | `s2c1_train.standardize_stats` |
| split (image-level) | fit 240 / val 60 / **held-out 150** / causal 15 | `s2c1_features.json` |
| scorer | `s_i = w·ĥ_i + b` — **4 097 parameters** | `TokenOnly` |
| loss form | balanced BCE + `λ·` all-pairs margin ranking (no sampling) | `s2c1_train.compute_loss` |
| `λ`, margin | 0.5, 1.0 | S2-C1 |
| optimizer | AdamW, lr 1e-3, wd 1e-4, batch 8 images | S2-C1 |
| early stop | max 80 epochs, patience 20, **on val Top-256 overlap** | S2-C1 |
| selector | Top-K @ 256, calibration = identity | S2-C1 / S2-B |

The baseline is the **cached S2-C1 `LIN_L4` weights**, not a re-implementation.
`BASE` is also retrained here as a *procedural control*: if the retrained arm
lands on the cached one within seed spread, a difference between a head arm and
the baseline is attributable to the target and not to the re-implementation.
It does — `BASE` reaches `overlap256` 0.560 / `head_recall@8` 0.734 /
`head_agree@8` 0.194 against the cached 0.562 / 0.736 / 0.192, and its downstream
macro of 55.75 is 1.57 below the cached arm's 57.32 for a scorer whose token
metrics differ by ≤0.003 (§4.3).

Each target is trained with **three seeds** (0, 1, 2). This is a variance
control, not a hyperparameter search — nothing is tuned, and the seeds differ
only in weight initialisation and batch order. Every headline difference is
quoted against the retraining noise as well as against zero.

The causal-15 tier appears nowhere in this stage. S2-C1 showed n = 8 causal AUROC
inverts the ordering of two arms 12 accuracy points apart; S2-C2 did not use it;
nothing here does either.

### 1.1 The four targets

No sweep, no tuning — four fixed readings, each changing one thing. Let `r_i` be
token `i`'s rank in the image's P1-G2 map (0 = the teacher's best); ranks below
256 are the teacher's negatives.

| arm | positives | pair / token weighting | `pos_weight` | what it changes vs `BASE` |
|---|---|---|---|---|
| **`BASE`** | teacher Top-256 | uniform | 3.0 | — (exact S2-C1 config) |
| **`HEAD_BIN`** | teacher **Top-32** | uniform | 31.0 | the positive set |
| **`HEAD_MULTI`** | teacher Top-256 | **tier-graded**: 4.0 / 2.0 / 1.5 / 1.0 / 0.25 for ranks <8 / <32 / <128 / <256 / ≥256 | 3.0 | the weighting |
| **`HEAD_RANK`** | teacher **Top-32** | **rank-graded**: pair weight `∝ 32/(r+1)`, i.e. 7.88 → 0.25 across the 32 positives | 31.0 | both |

`HEAD_BIN` is `BASE` with the positive set moved to the head. `HEAD_MULTI` is
`BASE` with the wide target graded by rank band. `HEAD_RANK` is `HEAD_BIN` with
the pairing weighted so the teacher's very best tokens dominate the ordering
term.

Both weight tensors are normalised to **mean 1** in every arm — the token weights
over the 1 024 tokens, the pair weights over the positives — so `λ = 0.5` keeps
the same meaning and the arms differ in the target, not in the loss scale. With
uniform weights the loss is *byte-identical* to `s2c1_train.compute_loss`
(verified on three random teacher orders, max |Δ| = 0.000e+00).

### 1.2 The metrics change, and that is the point

S2-C2 established that Top-256 overlap is measured at the one point where Top-k
agreement and Top-k recall are forced to coincide, and that the downstream value
is concentrated in the first few tokens of the teacher's ranking. So overlap is
kept but demoted, and two head-oriented families become primary:

| metric | definition | chance |
|---|---|---|
| **`head_recall@k`** | fraction of the teacher's Top-k that survives **inside the selected 256** ("kept anywhere") — k ∈ {8, 16, 32, 64} | 0.250 for every k |
| **`head_agree@k`** | `\|student Top-k ∩ teacher Top-k\| / k` — k ∈ {8, 16, 32} | `k/1024` |
| `overlap256` | **secondary** — the S2-C1 headline, reported for continuity | 0.250 |
| `auroc`, `ap`, `spearman` | separability diagnostics; never used to select | 0.500 / 0.25 |

The baseline row is recomputed from the cached `LIN_L4` scores through the same
code path, and reproduces S2-C2's measurements exactly: `overlap256` 0.5618
(0.562), `head_recall@8` 0.7358 (0.736), `head_agree@8` 0.1917 (0.192), mean
missed teacher Top-8 tokens 2.1133 (2.11). Every arm sees the identical 150
instances.

One metric the brief did not ask for turns out to matter, and is reported
throughout because it separates two things that both look like "head recall":
`head_recall@k` says how many of the teacher's best `k` are **in the set**, and
`head_agree@k` says how highly the student **orders** them. They are not the same
axis, and S2-C2's own reference arms prove it: `C_L2`, the attention proxy, has
`head_agree@8` = 0.211 — *above* `LIN_L4`'s 0.192 — while its `head_recall@8` is
0.506, far below it. An arm can rank the teacher's very best tokens better than
the linear probe does while holding half as many of them.

### 1.3 The decision rule, fixed before the numbers were read

Every comparison is made against **two** references: the cached S2-C1 `LIN_L4`,
and the retrained `BASE` at the *same seed* — a seed-matched control that cancels
initialisation and batch order and leaves only the target. An effect has to clear
zero against both.

* **head gain** — paired bootstrap (10 000 resamples, stratified by benchmark,
  resampling instances) on per-instance `head_recall@8` excludes zero against both
  references, **and** the margin exceeds the metric's own seed spread.
* **accuracy gain** — the same test on per-instance accuracy excludes zero against
  both references, and the seed-matched macro difference vs `BASE` is positive.
* **meaningful** — a head gain of at least **+5.0 points** of `head_recall@8`:
  the student misses 2.11 of the teacher's best 8 per image, so 5 points is about
  0.4 tokens per image.

| outcome | condition |
|---|---|
| **GO-A** | some arm has **both** a head gain and an accuracy gain |
| **GO-B** | (a) `head_recall@8` never improves, or (b) head recall improves but no arm's accuracy gain clears zero |
| **AMBIGUOUS** | no head gain clears the bar, but the CIs are too wide to exclude a meaningful one either |

The rule lives in the docstring of `s2c3_consolidate.py`, in the file that
computes it, so that it cannot be fitted to the result afterwards.

---

## 2. What was trained

Twelve runs — four targets × three seeds — 37–66 s each (the LLM is never loaded;
the scorer only reads the cached arrays).

| arm | seeds | best epoch (of run) | val `overlap256` | val `head_recall@8` | held-out `overlap256` | held-out `head_recall@8` |
|---|---|---|---|---|---|---|
| `BASE` | 0/1/2 | 8 (29) / 14 (35) / 8 (29) | 0.5559 / 0.5566 / 0.5570 | 0.7021 / 0.6937 / 0.6833 | 0.561 / 0.558 / 0.561 | 0.745 / 0.739 / 0.719 |
| `HEAD_BIN` | 0/1/2 | 8 (29) / 2 (23) / 6 (27) | 0.5060 / 0.5025 / 0.5116 | 0.7292 / 0.7146 / 0.7229 | 0.521 / 0.515 / 0.521 | 0.765 / 0.772 / 0.753 |
| `HEAD_MULTI` | 0/1/2 | 10 (31) / 14 (35) / 13 (34) | 0.5496 / 0.5489 / 0.5507 | 0.6833 / 0.6917 / 0.6917 | 0.555 / 0.553 / 0.556 | 0.749 / 0.738 / 0.741 |
| `HEAD_RANK` | 0/1/2 | 8 (29) / 1 (22) / 6 (27) | 0.4969 / 0.4965 / 0.5068 | 0.7250 / 0.7375 / 0.7292 | 0.514 / 0.508 / 0.517 | 0.761 / 0.777 / 0.758 |

Three things this table settles:

* **The head arms are better at the held-out head on the validation split too.**
  `HEAD_BIN`/`HEAD_RANK` sit at 0.715–0.738 val `head_recall@8` against `BASE`'s
  0.683–0.702. No selection decision in this stage touched the held-out 150, so
  this is not a held-out artifact.
* **The stopping rule did not truncate them.** Early stopping tracks val Top-256
  overlap, which is the *wrong* metric for a head arm — so this was checked
  directly. Val `head_recall@8` peaks at epochs 1–11 for every run and is within
  0.01–0.03 of its peak at the epoch the overlap criterion picked; the runs stop
  at 22–35 epochs, long after the head metric has flattened. Re-selecting the
  epoch on val head recall would move val `head_recall@8` by at most +0.02 and
  does not change which arms are separated. Recorded as a diagnostic in
  `s2c3_train.json` (`val_head32_epoch`), never used for selection.
* **Nothing is overfitting**, again: train ≈ val ≈ held-out on every arm, and all
  twelve runs converge in 22–35 epochs of an 80-epoch budget.

---

## 3. Head retention on the held-out 150

Seed mean over three seeds; `R@k` is the fraction of the teacher's Top-k inside
the selected 256, `A@k` the exact Top-k agreement, `miss8` the mean number of the
teacher's best 8 that are dropped.

| arm | ov256 | R@8 | R@16 | R@32 | R@64 | A@8 | A@16 | A@32 | miss8 | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| `P1G2_teacher` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00 | 1.000 |
| **`HEAD_RANK`** | 0.513 | **0.766** | **0.741** | **0.694** | 0.643 | **0.266** | **0.240** | **0.244** | **1.88** | 0.754 |
| **`HEAD_BIN`** | 0.519 | **0.763** | 0.738 | 0.694 | 0.643 | 0.261 | 0.238 | 0.244 | 1.89 | 0.760 |
| `HEAD_MULTI` | 0.554 | 0.743 | 0.720 | 0.676 | 0.642 | 0.196 | 0.197 | 0.208 | 2.06 | 0.794 |
| *`BASE`* | *0.560* | *0.734* | *0.714* | *0.675* | *0.644* | *0.194* | *0.194* | *0.208* | *2.12* | *0.796* |
| **`S2C1_LIN_L4`** | **0.562** | 0.736 | 0.719 | 0.681 | 0.648 | 0.192 | 0.190 | 0.209 | 2.11 | 0.798 |
| `C_L2` *(S2-C0)* | 0.352 | 0.506 | 0.448 | 0.406 | 0.385 | 0.211 | 0.172 | 0.140 | 3.95 | 0.612 |
| `A_L2` *(S2-C0)* | 0.353 | 0.497 | 0.445 | 0.401 | 0.381 | 0.229 | 0.176 | 0.149 | 4.02 | 0.607 |
| *random* | *0.248* | *0.247* | *0.240* | *0.242* | *0.247* | *0.010* | *0.018* | *0.033* | *6.02* | *0.499* |
| *EADP's own score* | *0.185* | *0.150* | *0.156* | *0.167* | *0.168* | *0.006* | *0.010* | *0.020* | *6.80* | *0.402* |

### 3.1 The change, against two independent references

Δ in points, paired bootstrap over the 150 instances, stratified by benchmark.
`LIN_L4` is the exact cached baseline; `BASE` is the seed-matched retrained
control; the last column is the three per-seed paired differences.

| arm | metric | Δ vs `LIN_L4` | 95 % CI | Δ vs `BASE` | 95 % CI | per-seed Δ vs `BASE_s*` | 3-seed spread |
|---|---|---|---|---|---|---|---|
| `HEAD_BIN` | `head_recall@8` | **+2.75** | **[+0.81, +4.75]** | **+2.89** | **[+1.08, +4.69]** | +2.00, +3.25, +3.42 | 1.83 |
| `HEAD_BIN` | `head_recall@32` | +1.26 | [−0.06, +2.60] | **+1.90** | **[+0.62, +3.20]** | — | — |
| `HEAD_BIN` | `head_agree@8` | **+6.97** | **[+5.31, +8.58]** | **+6.72** | **[+5.08, +8.33]** | — | 1.58 |
| `HEAD_BIN` | `overlap256` | **−4.28** | **[−4.94, −3.63]** | **−4.09** | **[−4.73, −3.46]** | — | 0.57 |
| `HEAD_RANK` | `head_recall@8` | **+2.97** | **[+0.86, +5.17]** | **+3.11** | **[+1.19, +5.03]** | +1.58, +3.83, +3.92 | 1.92 |
| `HEAD_RANK` | `head_recall@32` | +1.28 | [−0.15, +2.72] | **+1.92** | **[+0.54, +3.33]** | — | — |
| `HEAD_RANK` | `head_agree@8` | **+7.39** | **[+5.67, +9.11]** | **+7.14** | **[+5.44, +8.83]** | — | 1.42 |
| `HEAD_RANK` | `overlap256` | **−4.86** | **[−5.53, −4.21]** | **−4.68** | **[−5.32, −4.04]** | — | 0.91 |
| `HEAD_MULTI` | `head_recall@8` | +0.67 | [−0.36, +1.72] | +0.81 | [+0.11, +1.50] | +0.42, −0.17, +2.17 | 1.17 |
| `HEAD_MULTI` | `head_recall@32` | −0.55 | [−1.24, +0.15] | +0.09 | [−0.42, +0.60] | — | — |
| `HEAD_MULTI` | `head_agree@8` | +0.39 | [−0.47, +1.22] | +0.14 | [−0.56, +0.83] | — | — |
| `HEAD_MULTI` | `overlap256` | **−0.75** | **[−1.05, −0.44]** | **−0.56** | **[−0.79, −0.34]** | — | 0.26 |
| *`BASE`* | `head_recall@8` | −0.14 | [−1.06, +0.81] | — | — | — | 2.58 |
| *`BASE`* | `overlap256` | −0.19 | [−0.39, +0.02] | — | — | — | 0.31 |

**The supervision question is answered, and answered yes.** `HEAD_BIN` and
`HEAD_RANK` clear zero against both references on `head_recall@8`, by margins
(2.75, 2.97) larger than their own 3-seed spread (1.83, 1.92), and their
`head_agree@8` gain is +7 points with a CI bounded well away from zero. It is
also the first scorer in this project to move retention of the teacher's *best*
tokens away from where S2-C1's membership probe left it.

The `head_recall@32` row is the honest caveat and shows why the two references
were both required. Against the cached baseline the gain (+1.26, +1.28) straddles
zero; against the seed-matched control it does not (+1.90 [+0.62, +3.20],
+1.92 [+0.54, +3.33]). The pre-registered rule demands both, so `HEAD_BIN` and
`HEAD_RANK` are **not** credited with a Top-32 gain — but the discrepancy is
itself informative: the retrained `BASE` is 0.64 points *below* the cached
baseline on this metric (`head_recall@32` −0.64 [−1.17, −0.11]), so part of the
head arms' apparent Top-32 advantage is the control's own deficit. The resolved
movement is at the very top of the ranking (Top-8/16), not through Top-32, and by
Top-64 it is gone.

### 3.2 The gain is concentrated where S2-C2 said the value was — and on the benchmark that had the most to recover

Per-benchmark `head_recall@8`, seed mean:

| arm | TextVQA_VAL | DocVQA_VAL | OCRBench | macro |
|---|---|---|---|---|
| `S2C1_LIN_L4` | 0.8050 | 0.6575 | 0.7450 | 0.7358 |
| `BASE` | 0.8075 | 0.6600 | 0.7358 | 0.7344 |
| `HEAD_BIN` | 0.8100 | **0.7158** | 0.7642 | 0.7633 |
| `HEAD_MULTI` | 0.8092 | 0.6825 | 0.7358 | 0.7425 |
| `HEAD_RANK` | 0.8183 | **0.7192** | 0.7592 | 0.7656 |

The retargeting does almost nothing on TextVQA (+0.5 points), where the student
was already recovering 80.5 % of the head, and moves DocVQA by **+5.8 / +6.2**
points — the benchmark S2-C2 measured as missing 2.74 of the teacher's best 8 per
image against TextVQA's 1.56, and the one with the largest raw gap (24.1 points).
The scorer improves where the head was actually being lost. That is a coherent
mechanism, not a diffuse metric shift.

### 3.3 It is the positive set, not the weighting

`HEAD_BIN` and `HEAD_RANK` differ *only* in how the 32 positives' pairs are
weighted (uniform versus `∝ 1/(r+1)`, a 32-fold spread), and they are
indistinguishable on every metric in the table: `head_recall@8` 0.7633 vs 0.7656
(Δ = 0.2 points, inside both seed spreads), `head_agree@8` 0.2614 vs 0.2656,
`overlap256` 0.5191 vs 0.5132. Whatever the head objective is doing, the fine
grading of the head's ordering is not doing it.

`HEAD_MULTI` changes the weighting *without* moving the positive set, and it
behaves exactly like `BASE`: `head_recall@8` 0.7425 against `BASE`'s 0.7344 and
`LIN_L4`'s 0.7358, `head_agree@8` 0.1956 against 0.1942 / 0.1917, `overlap256`
0.5543 against 0.5599 / 0.5618. Its only significant movement is a 0.75-point
*loss* of overlap.

So of the four targets, the two that change the positive set move the head and
the two that do not, do not. "Grade the target by teacher rank band" — which
reads like the natural way to express S2-C2's finding that value is front-loaded
— is measurably inert, because the tier weights are applied to a positive set
(256 tokens) that is 74 % of the way to the head already.

### 3.4 The two objectives trade against each other

This is the stage's most transferable structural result.

| arm | `overlap256` | `head_recall@8` | `head_recall@64` | AUROC |
|---|---|---|---|---|
| `S2C1_LIN_L4` | 0.562 | 0.736 | 0.648 | 0.798 |
| `HEAD_MULTI` | 0.554 | 0.743 | 0.642 | 0.794 |
| `HEAD_BIN` | 0.519 | 0.763 | 0.643 | 0.760 |
| `HEAD_RANK` | 0.513 | 0.766 | 0.643 | 0.754 |
| *random* | *0.248* | *0.247* | *0.247* | *0.499* |

`head_recall@64` is **flat across every trained arm** (0.648 → 0.643) while `@8`
and `@16` rise and `overlap256` falls by 4.3–4.9 points. The head arms have not
found more of the teacher's ranking; they have **moved their budget from the
middle of the ranking to the top of it**, and the Top-64 total is unchanged.

The AUROC column says the same thing in the separability language S2-C1 was gated
on: 0.798 → 0.754, i.e. the head-trained scores are *worse* global predictors of
Top-256 membership, by a margin (0.044) that dwarfs anything S2-C1's layer or
capacity axes produced. **Any selection rule that scores a candidate on
teacher-agreement would rank `BASE` above `HEAD_RANK` on AUROC and overlap, and
would be selecting the arm with less head retention.** It is one more instance of the
pattern S2-C0 and S2-C1 reported — teacher-agreement metrics pointing away from
usefulness — but the first time it is visible *within* a single fixed scorer
family, where the features, the layer, the capacity and the training pipeline
are all identical and only the objective differs.

---

## 4. Downstream translation

All twelve runs through the **unmodified** generation harness (`s2c0_run.py`),
held-out 150, Top-K @ 256, identity calibration — the same selector, the same
instances and the same code path as every S2-B/S2-C0/S2-C1 arm quoted here.
`n_kept_mean` = 256.0 on all 36 benchmark-runs, confirming the maps actually
drove the selection.

### 4.1 Results

Seed mean over three seeds.

| arm | TextVQA | DocVQA | OCRBench | **macro** | 3-seed range | Δ vs `LIN_L4` | 95 % CI | retained | rescued / still / broken |
|---|---|---|---|---|---|---|---|---|---|
| **`S2C1_LIN_L4`** *(cached)* | 67.600 | 56.353 | 48.000 | **57.318** | — | — | — | 45.0 % | — |
| **`HEAD_BIN`** | 68.000 | 55.083 | 48.000 | **57.028** | 54.67–58.73 | −0.290 | [−5.34, +4.84] | 44.1 % | 7.7 / 20.3 / 8.7 |
| **`HEAD_MULTI`** | 70.333 | 55.943 | 44.000 | **56.759** | 56.50–57.22 | −0.559 | [−5.02, +3.91] | 43.2 % | 5.0 / 23.0 / 7.3 |
| **`HEAD_RANK`** | 68.267 | 55.334 | 46.000 | **56.534** | 55.39–57.79 | −0.784 | [−6.47, +4.93] | 42.5 % | 9.0 / 19.0 / 11.3 |
| **`BASE`** *(retrained control)* | 69.333 | 53.901 | 44.000 | **55.745** | 54.84–57.33 | −1.573 | [−5.48, +2.31] | 40.1 % | 2.7 / 25.3 / 6.3 |

Reference points, identical instances and harness: official EADP **facility**
61.097, P1-G2 teacher 75.150, EADP's own score under Top-K 42.753.

**Not one head target beats the baseline, and not one loses to it in a way the
design can resolve.** The three head arms sit within 0.79 macro points of
`LIN_L4`, with CIs 10–11 points wide straddling zero. Two of three are nominally
*worse*; `BASE` retrained is nominally worse still (−1.57), for a scorer whose
token metrics are within 0.003 of the baseline's.

The retained-gain column is the S2-C1 denominator (gain over EADP's own score
under the matched selector): 45.0 % for the cached baseline, 44.1 / 43.2 / 42.5 %
for the head arms, 40.1 % for the retrained control. The head arms are, if
anything, marginally *ahead* of their own seed-matched control and marginally
behind the cached baseline, and both gaps are noise.

### 4.2 The arms change which answers are right without changing how many

On the S2-C2 class-1 set — the 28 instances `LIN_L4` gets wrong and the teacher
gets right — the head targets do move the answers, and then undo it:

| arm | rescued (of 28) | still wrong | newly broken (of 150) | correct count (of 150) | Δ vs `LIN_L4` |
|---|---|---|---|---|---|
| *`S2C1_LIN_L4`* | — | — | — | *89* | — |
| `BASE` | **2.7** | 25.3 | 6.3 | 87.3 (86/86/90) | −1.7 |
| `HEAD_MULTI` | 5.0 | 23.0 | 7.3 | 89.0 (90/88/89) | +0.0 |
| `HEAD_BIN` | **7.7** | 20.3 | 8.7 | 90.0 (93/90/87) | **+1.0** |
| `HEAD_RANK` | **9.0** | 19.0 | 11.3 | 89.7 (89/91/89) | +0.7 |

`HEAD_RANK` rescues **9.0** of the 28 against `BASE`'s 2.7 — a 3.3× increase in
the number of instances the retargeted scorer gets right that the baseline gets
wrong. It also newly breaks **11.3** instances the baseline got right, against
`BASE`'s 6.3. Counted as instances, the three head arms end up +0.0, +0.7 and
+1.0 above the baseline and +1.7 to +2.7 above the retrained control — all
inside a within-arm seed spread of ±2 instances.

This is exactly the S2-C2 warning (§8.4) landing: the aggregate metric is carried
by a minority of answer flips, and an intervention that reshuffles *which*
instances flip without increasing the count produces a flat mean. The per-dataset
split shows the same churn — on OCRBench, `HEAD_RANK` rescues 4.0 of 12 and
breaks 5.3.

**One nuance the two columns expose.** The head arms are nominally *below* the
baseline on macro (−0.29 to −0.78) while being nominally *above* it on the count
of correct answers (+0.7 to +1.0). The two metrics disagree because macro averages
graded per-instance scores (ANLS, VQA-score) and the threshold count does not:
the retargeted scorer wins on instances that cross the 0.5 line and loses partial
credit on instances that do not. Both numbers are inside their respective noises,
but they are not the same number, and a stage that reported only one of them
would have reported a sign.

### 4.3 Why the accuracy half of the test cannot resolve this — and what it can still say

The design's own noise floor has to be measured, not assumed. Two seeds of the
**same target** are a true null: the scorer has the same objective and differs
only in initialisation.

| null comparison (same target, two seeds) | Δ macro | 95 % CI |
|---|---|---|
| `BASE_s0` vs `BASE_s1` | −0.22 | [−4.28, +3.78] |
| `BASE_s0` vs `BASE_s2` | +2.26 | [−2.03, +6.67] |
| `HEAD_BIN_s0` vs `HEAD_BIN_s1` | −1.04 | [−5.92, +3.57] |
| `HEAD_RANK_s0` vs `HEAD_RANK_s1` | +1.36 | [−3.99, +6.69] |
| `HEAD_MULTI_s0` vs `HEAD_MULTI_s1` | −0.72 | [−4.78, +3.47] |

**Running the identical configuration twice moves macro by 1–2 points and
produces CIs ±4–5 points wide.** The whole of the S2-C3 downstream effect —
0.79 points — sits an order of magnitude inside that.

The seed-matched per-seed differences do carry a consistent *sign*: six of the
nine (head arm − `BASE_s<same seed>`) are positive, ranging −2.65 to +3.66, with
means +1.28 (`HEAD_BIN`), +1.01 (`HEAD_MULTI`) and +0.79 (`HEAD_RANK`). Every one
of those nine CIs straddles zero. Read as a sign test this is weak evidence of a
small positive effect; read against the null above it is indistinguishable from
one.

What this stage can say is therefore bounded, and should be stated that way:

* **Not supported:** any claim that head-aligned supervision improves downstream
  accuracy on this benchmark. The point estimate is −0.29 to −0.78 against the
  baseline.
* **Not refuted either:** an effect of +0.2 to +0.3 macro points — the size the
  head-recall gain is worth by S2-C2's exchange rate — is invisible at this
  resolution. The verdict is "nothing detectable", not "nothing there".
* **Firmly established:** the ~3-point head-recall gain and ~7-point head-
  agreement gain are real, resolved, and reproducible across seeds and against
  two independent references. The failure is in the conversion, not in the
  measurement.

---

## 5. The key analysis: does head recall translate into accuracy?

The question the brief asks is whether a scorer that improves head retention
moves up the accuracy axis. `figures/s2c3_head_retargeting.png` panel A plots
exactly that, for `head_recall@8`, `@16` and `@32`.

| | `head_recall@8` | Δ vs baseline | macro accuracy | Δ vs baseline |
|---|---|---|---|---|
| `S2C1_LIN_L4` | 0.736 | — | 57.318 | — |
| `BASE` | 0.734 | −0.14 | 55.745 | −1.573 |
| `HEAD_MULTI` | 0.743 | +0.67 | 56.759 | −0.559 |
| `HEAD_BIN` | 0.763 | **+2.75** | 57.028 | −0.290 |
| `HEAD_RANK` | 0.766 | **+2.97** | 56.534 | −0.784 |
| `P1G2_teacher` | 1.000 | +26.4 | 75.150 | +17.83 |
| `random` | 0.247 | −48.9 | 42.753 | −14.57 |

**The points lie on a horizontal line.** Across a 2.9-point span of head recall
the four trained arms move 1.3 macro points in a direction uncorrelated with the
head axis; the whole of the measured accuracy variation is the seed noise
documented in §4.3.

### 5.1 The one positive result the brief asked to check for, and its answer

> *"If Top-256 overlap does not improve much, but Top-8/32 recall and accuracy both
> improve significantly, this is a very important positive result."*

The first half happened and the second half did not. Top-256 overlap did not
improve — it fell 4.3–4.9 points, significantly. Top-8 recall improved
significantly (+2.75/+2.97, CIs clear of zero). Accuracy did **not** improve; it
is flat to within noise and nominally negative.

That combination is the GO-B branch of §1.3, and it is worth being precise about
what it does and does not imply, because the brief's GO-B gloss — "the token-local
L4 scorer cannot express the structure S2-C2 found" — is **not** what the data
show. The scorer expressed more of that structure than any previous arm in this
project; it just did not express enough of it to move a 150-instance benchmark.

### 5.2 The size of the prize, by S2-C2's own exchange rate

The expected size follows from S2-C2's measured curve alone, with no input from
this stage's accuracy numbers, so it can be stated independently of them.

The head arms recover 0.23 more of the teacher's best-8 tokens per image
(2.11 → 1.88). S2-C2's marginal-value table prices the first few recovered tokens
at **+1.31 macro per token** (its 2→6 range) and **+0.92** for the first two
(0→2). 0.23 tokens is therefore worth roughly **+0.2 to +0.3 macro**.

The head arms also give up 4.4 points of Top-256 overlap, i.e. ~11 mid-tail
tokens. S2-C2 measured the marginal value of a swapped token past position 32 at
**~0** (its 32→112 range is flat, `teacher:40` is nominally *below* `teacher:32`,
and its removal arms are inert or negative). Those 11 tokens are worth
**~0**.

So the predicted downstream change is **+0.2 to +0.3 macro, on a metric whose
own seed-to-seed spread is ±2 to 3**. The measured values are −0.29, −0.56 and
−0.78.

The result is not a surprise and not a contradiction: it is the arithmetic of
S2-C2's own exchange rate applied to a head-recall gain that is three points
where the exchange rate needs tens. **The reason head-retargeting does not pay is
that a shared token-local linear scorer can only move head retention by ~3 points,
and 3 points of head retention is worth ~0.3 macro.**

---

## 6. Supplementary: is the head linearly expressible from L4 at all?

`GO-B` as the brief defines it carries a specific mechanism claim — that the
token-local representation cannot express the head. It is worth testing that
directly, cheaply, and without training, because the trained arms confound two
different failures: *no direction exists* versus *one shared direction does not
transfer*.

The test is the simplest closed-form linear read-out there is: the mean-difference
(one-shot centroid) direction, `w = mean(h over the target set) − mean(h over the
rest)`, fitted **on the same image** and therefore deliberately optimistic. Three
quantities, on the held-out 150:

| | read-out | recall@256 |
|---|---|---|
| **A** | teacher's Top-8, direction centred on the teacher's head | **0.9817** |
| **B** | an arbitrary 8 tokens, direction centred on that set (5 seeds) | 0.9205 ± 0.1068 |
| **C** | teacher's Top-8, direction centred on an arbitrary 8 (5 seeds) | 0.2370 ± 0.2471 |
| **A32** | teacher's Top-32, direction centred on the teacher's head | 0.9025 |
| **B32** | an arbitrary 32, direction centred on that set | 0.6914 ± 0.1271 |

`A_agree8` — the top-8 agreement of that in-sample direction — is 0.4267.

Three readings:

* **The head is linearly present in L4.** A in-sample direction recovers 98.2 %
  of the teacher's best 8. "The information is not in the layer-4 token
  representation" is false, and so is the strong form of the brief's GO-B gloss.
* **It is not *specially* present at the top-8.** A = 0.982 against
  B = 0.921 ± 0.107: a direction fitted on the teacher's own best eight recovers
  0.061 more of it than a direction fitted on an arbitrary eight, which is 0.6 of
  B's own spread. At top-8 there is no strong sense in which the teacher's head
  sits in a linearly distinctive place in L4 space. At top-32 it does — A32 =
  0.903 against B32 = 0.691 ± 0.127, 1.6 spreads apart — so the *wider* head is
  more linearly self-consistent than an arbitrary group of the same size, while
  the very best eight are barely more so. That is the same shape as the trained
  result, where the gain is at Top-8/16 and dissolves by Top-64.
* **C is the control that makes this interpretable.** A direction fitted on a
  random 8 recovers 23.7 % of the teacher's head — chance (0.25). The direction
  has to be aligned with the head; alignment is not free.

The gap that matters: the best **shared** direction the trained arms reach is
`head_recall@8` = 0.766 with 0.266 top-8 agreement, against 0.982 and 0.427 for a
direction fitted on the image itself. **Retargeting the loss closed 0.031 of the
0.247 recall gap and 0.071 of the 0.233 agreement gap.** The failure is neither
"the loss was aimed wrong" nor "the representation lacks the information": it is
that a single linear direction fitted across 240 images transfers only a fraction
of what a per-image direction can express.

This diagnostic is interpretive only. It trained nothing, selected nothing, and
the verdict in §7 does not rest on it.

---

## 7. Decision

Applying the pre-registered rule of §1.3:

| arm | head gain | accuracy gain | Δ macro vs `LIN_L4` | 95 % CI | seed-matched Δ macro |
|---|---|---|---|---|---|
| `HEAD_BIN` | **yes** (+2.75, both refs, > spread) | **no** | −0.290 | [−5.34, +4.84] | +1.28 |
| `HEAD_MULTI` | no (+0.67, CI straddles zero) | no | −0.559 | [−5.02, +3.91] | +1.01 |
| `HEAD_RANK` | **yes** (+2.97, both refs, > spread) | **no** | −0.784 | [−6.47, +4.93] | +0.79 |

Head recall improves; no arm's accuracy gain clears zero against either
reference. That is **GO-B (b)**.

```
S2-C3 GO-B -- head recall moves, downstream accuracy does not follow it.
```

The per-arm detail is reported in full in `s2c3_retarget.json` (`verdicts`),
including the intermediate quantities the rule reads — both references' CIs, the
seed spreads, and the seed-matched per-seed differences — so the rule can be
re-applied to different thresholds without re-running anything.

**What GO-B licenses, and what it does not.** The brief's GO-B is worded as
"the current token-local L4 scorer cannot express the high-value structure S2-C2
found", and gates the next stage on moving to contextual / group-aware scoring.
The measurement supports the *gate* but corrects the *reason*:

* It is **not** that the structure is inexpressible. A per-image linear direction
  recovers 98 % of the head (§6), and the retargeted shared scorer does express
  measurably more of it than the membership-trained one — 3 points of recall and
  7 points of top-8 agreement.
* It is that the expressible movement is **too small**. Three points of head
  recall is worth ~0.3 macro by S2-C2's own exchange rate, which no 150-instance
  comparison can resolve.
* And it is bought by **giving up 4.4 points of Top-256 overlap**, which S2-C2
  prices at ~0. The trade is therefore free in both directions, which is another
  way of saying that within this scorer family, *the position along the
  head-recall / overlap frontier does not matter*.

That last statement is the transferable result, and it is the one a next stage
should be built against. Anything that hopes to reach S2-C2's measured recoverable
value must move head retention by tens of points, not three — which means changing
what a token's score can depend on, not which tokens the loss calls positive.

---

## 8. What this stage does not claim

* **Nothing here is a method.** No selector changed, no architecture changed, no
  fusion, no attention feature, no hidden size, no layer sweep, no loss sweep. The
  four targets are four fixed readings of the same teacher, and the scorer is
  S2-C1's 4 097-parameter linear probe throughout.
* **The negative is not a proof of absence.** §4.3 measures the design's own null
  at ±2–3 macro points with ±5-point CIs. The stage establishes "no downstream
  gain is detectable"; it does not establish that the true effect is zero, and it
  explicitly cannot see the +0.2–0.3 the exchange rate predicts.
* **`head_recall@32` did not clear zero against both references.** It clears the
  seed-matched control (+1.90 [+0.62, +3.20], +1.92 [+0.54, +3.33]) but not the
  cached baseline (+1.26 [−0.06, +2.60], +1.28 [−0.15, +2.72]), and the rule
  requires both. The resolved gain is at Top-8 and Top-16 only. Text claiming
  "the head" moved should say "the first eight to sixteen tokens of the teacher's
  ranking".
* **The oracle diagnostic is in-sample and optimistic.** §6 fits the direction on
  the same image it is evaluated on, with a matched random-target control. It
  bounds what a linear read-out *could* express, not what any deployable scorer
  achieves, and it is used for interpretation only.
* **The causal-15 tier is not used.** Per the brief and per S2-C1's inversion
  result, it appears in no selection, metric or verdict.
* **n = 50 per benchmark.** Every aggregate difference quoted as a gain has a
  paired bootstrap CI in the deliverable JSON; the ones discussed as effects are
  the ones whose CI clears zero.

---

## 9. Cost

The scorer has exactly the `LIN_L4` form — one `Linear(4 096, 1)` — so only its
weights differ and its cost is unchanged: measured here at **0.071 ms** for the
1×1024×4096 forward and **0.102 ms** for the Top-256 selection. Against S2-C1's measured prefix forward through layer 4 (29.4 ms) and
pruned prefill (72.8 ms), the scoring-only cost stays ~29.5 ms and the naive
end-to-end cost ~102 ms — the retargeting changes which tokens are selected, not
what selecting costs.

---

## 10. What carries forward

1. **The objective was misaligned, and realigning it works — on its own metric.**
   Positive set Top-256 → Top-32 raises teacher Top-8 retention by 2.9 points and
   exact Top-8 agreement by 7 points, with CIs clear of zero against two
   independent references. This is the first movement in head retention
   since S2-C1's membership probe, and it costs the same 4 097 parameters.

2. **The two objectives are in tension, not nested.** Head recall up, Top-256
   overlap down 4.4 points, AUROC down 0.044, `head_recall@64` flat. A scorer
   cannot be pushed toward the head without giving up the middle of the ranking,
   and within this family the position on that frontier is **not** worth anything
   downstream over the range explored.

3. **Grading beats membership is false; membership beats grading is true.**
   `HEAD_BIN` ≈ `HEAD_RANK` despite a 32-fold difference in pair weighting;
   `HEAD_MULTI`, which re-weights the wide target, is indistinguishable from
   `BASE`. What matters is *which tokens are called positive*.

4. **The head gain concentrates where the head was actually being lost.** +5.8 to
   +6.2 points of `head_recall@8` on DocVQA, whose baseline was 0.658 and which
   missed 2.74 of the teacher's best 8 per image; +0.5 on TextVQA, whose baseline
   was 0.805.

5. **The answers churn without moving the mean.** `HEAD_RANK` rescues 9.0 of the
   28 S2-C2 class-1 instances against `BASE`'s 2.7, and newly breaks 11.3 against
   `BASE`'s 6.3. Any future stage reporting a head-retargeting result must report
   rescue *and* breakage counts, or it will report a net that does not exist.

6. **The information is in L4; one shared direction cannot get to it.** An
   in-sample per-image linear read-out recovers 98.2 % of the teacher's Top-8
   (versus 23.7 % for a direction fitted on a random 8). The trained shared
   direction reaches 76.6 %. That 0.247 recall / 0.233 agreement gap is the object
   the next stage has to attack, and it is a transfer problem, not an
   expressibility problem — which is a different problem from the one the brief's
   GO-B branch anticipated.

7. **Three points of head recall is worth three-tenths of a macro point.** The
   stage can be summarised as an arithmetic result: S2-C2's exchange rate (~+1.3
   macro per recovered teacher Top-8 token) times the head retention a shared
   token-local linear scorer can actually deliver (+0.23 tokens/image) is
   +0.3 macro, and +0.3 macro is not measurable on 150 instances. Any method
   aimed at this value has to move head retention by an order of magnitude more
   than supervision retargeting can.

---

## 11. Reproduce

```
source ~/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
cd /media/disk2/YZX/research/EADP/Qwen_vl

# 1. train 4 targets x 3 seeds (12 runs; the LLM is not loaded -- the scorer
#    only reads the cached S2-C1 layer-4 features)
python scripts/discovery/s2c3_train.py --arms BASE HEAD_BIN HEAD_MULTI HEAD_RANK \
    --seeds 0 1 2 --tag s2c3

# 2. held-out head retention + the identity check against S2-C2's baselines
python scripts/discovery/s2c3_eval.py --tag s2c3

# 3. accuracy translation through the unmodified harness, every arm and seed
python scripts/discovery/s2c0_run.py --scores s2c3_scores.npz \
    --arms BASE_s0 BASE_s1 BASE_s2 HEAD_BIN_s0 HEAD_BIN_s1 HEAD_BIN_s2 \
           HEAD_MULTI_s0 HEAD_MULTI_s1 HEAD_MULTI_s2 \
           HEAD_RANK_s0 HEAD_RANK_s1 HEAD_RANK_s2 --tag s2c3_pilot

# 4. the supplementary linear-expressibility diagnostic (CPU, no training)
python scripts/discovery/s2c3_oracle.py --tag s2c3

# 5. consolidate (verdict) and plot
python scripts/discovery/s2c3_consolidate.py --tag s2c3
python scripts/discovery/s2c3_figure.py
```

Or in one step: `bash scripts/discovery/run_s2c3.sh`.

Deliverables: `outputs/discovery/s2c3_retarget.json` (head-retention table with
both references' CIs and seed spreads, the per-run and per-arm downstream tables,
the class-1 rescue block, the pre-registered verdicts, training records and the
latency block), plus `s2c3_train.json`, `s2c3_eval.json`, `s2c3_headmetrics.npz`
(per-instance metric arrays, 13 arms × 150 instances), `s2c3_oracle.json`,
`s2c3_pilot.json`, `s2c3_scores.npz` (5 580 score vectors = 12 arms × 465
instances), `s2c3_<ARM>_s<seed>.pt`, and
`figures/s2c3_head_retargeting.png`.
