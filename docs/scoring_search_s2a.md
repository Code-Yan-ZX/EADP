# S2-A — Causal-gradient scoring viability probe

**Scope.** Stage-2 S2-A. One question: *does gradient saliency w.r.t. the 1024
visual token embeddings entering the LLM actually identify the causally
necessary regions that the Stage-1 occlusion probe already established?*

This is a viability probe, not a method implementation. No full-split run, no
accuracy run, no selector integration, no S2-B. The S1 Top-3 were not run for
accuracy.

**Status.** Complete.

---

## VERDICT

```
S2-A: GO
deployment proxy promising
```

Gate A and Gate B both pass, in both cases by a wide margin (§8, §9).

**But the probe also produced a result that reframes what the method is, and it
belongs beside the verdict rather than buried under it.** A control that
teacher-forces a *deliberately wrong* answer — same image, same prompt, same
visual tokens, only the target continuation changed — still localizes the
necessary blocks at AUROC **0.907** (matched answer: 0.928). The two saliency
maps correlate at Spearman **0.85**. So:

> **The gradient saliency is a strong localisation signal (+4.6 sd above the
> random null), but only ~0.02 AUROC of it is answer-specific. The bulk is a
> general "visual tokens the LM's output distribution is sensitive to" signal
> that survives having the answer replaced.**

This is consistent with, and explains, the fact that the answer-free proxies
P1/P2 match the answer-conditioned upper bound (§5). It does **not** invalidate
Gate A — the gate asks whether gradient saliency localizes causal evidence, and
it does, decisively. It does mean the novelty story cannot be "answer-conditioned
attribution", and that any S2-B framing should be built on the sensitivity
signal itself. See §3 and §11.

---

## 0. Executive summary

1. **The gradient hook is verified exact.** Differentiating the tensor
   `model.visual(pixel_values, grid_thw)` — shape `[1024, hidden_dim]` — and the
   official pruner, run on that same tensor, reproduces the Stage-1 saved
   `select_idx` bit-for-bit on all 15 causal cases (§1).
2. **Teacher forcing is verified valid.** Prompt ends with
   `<|im_start|>assistant\n`; matched-answer NLL is 0.00–1.46 total; and the
   answer-free P1 top-1 token equals the true answer's first token in **15/15**
   cases (§2).
3. **Gate A passes**: answer-conditioned gradient×input reaches mean rank
   improvement **0.2435** (bar 0.10), block AUROC **0.928** (bar 0.65),
   **8/8** cases improved (bar 6/8).
4. **Gate B passes**: the answer-free top-1-logit proxy reaches AUROC **0.895**
   and rank improvement **0.2325**, 8/8 improved. The answer-free proxy is
   *nearly as good as the answer-conditioned oracle*.
5. **gradient×input (G2) beats gradient-norm (G1) consistently**, by 0.04–0.05
   rank and 0.05–0.07 AUROC, on every tier and every objective.
6. **Two obvious confounds are refuted**: `||v_i||` alone scores at chance
   (AUROC 0.443), and is uncorrelated with G2 (r = −0.07); necessary blocks are
   spread over 12 of 16 blocks on the primary set, so no position prior explains
   the result.
7. **One confound is not refuted**: the signal is largely answer-agnostic (§3).
8. **Cost**: one forward + one backward ≈ **507 ms** per image, against a 228 ms
   unpruned prefill and 40 ms EADP selector overhead. Recorded, not used to
   reject, per the brief. It is nonetheless the thing standing between this and
   a deployable method (§7).

---

## 1. Gradient hook correctness

### 1.1 The tensor, and why it is the right one

The EADP path is `Qwen3VLChatEADP._get_pruned_image_features`:

```python
image_embeds = unwrap_visual_output(self.model.visual(pixel_values, grid_thw=image_grid_thw))
... pruner(image_embeds, text_embeds_llm, text_embeds_seq_llm, image_grid_thw)
```

`image_embeds` is `[1024, hidden_dim]` — the post-merger visual tokens — and EADP's
`select_idx` indexes *its rows*. So differentiating this tensor gives a
per-visual-token gradient in exactly the token index space the Stage-1 occlusion
labels live in.

The probe differentiates a leaf built from it:

```python
with torch.no_grad():
    vis0 = unwrap_visual_output(model.visual(pixel_values, grid_thw=image_grid_thw))
visual_tokens = vis0.detach().clone().requires_grad_(True)     # [1024, hidden_dim]
```

Detaching at the visual-tower output (rather than back-propagating through the
tower) is both cheaper and semantically correct: the saliency is *with respect to
the embedding*, which is the quantity a pruner selects. All LLM parameters are set
`requires_grad_(False)` so only `visual_tokens` accumulates gradient.

### 1.2 Verified checks (all 15 causal cases)

| Check | Result |
|---|---|
| **H1** `visual_tokens.shape == [1024, hidden_dim]` | ✓ 15/15 (hidden_dim = 4096) |
| **H2** prompt has exactly 1024 image-token positions, contiguous | ✓ 15/15 |
| **H3** official pruner on this tensor reproduces Stage-1 saved `select_idx` | ✓ **15/15, exact set equality** |
| **H4** differentiable prompt reconstruction == official `_build_pruned_inputs` at budget 1024 | ✓ max abs diff **0.0** |
| **H5** block geometry: block *b* = rows/cols `[r*8:(r+1)*8] x [c*8:(c+1)*8]` of the 32x32 grid | ✓ matches `failure_analysis.py` |

H3 is the load-bearing one. Re-running the official scoring chain
(`_sim_cross` → entropy filter → aggregation → `alpha=0.5` fusion → min-max →
3x3 Gaussian → `**beta`) and the official greedy selector on the hooked tensor
reproduces the saved selection **exactly**, so the tensor being differentiated is
provably the tensor EADP scores.

H5 spot-check, three blocks per case (first token index, last token index, count):

```
block 0  -> (0,   231, 64)      block 5  -> (264, 495, 64)
block 1  -> (8,   239, 64)      block 10 -> (528, 759, 64)
block 2  -> (16,  247, 64)      block 12 -> (768, 999, 64)
block 4  -> (256, 487, 64)      block 15 -> (792, 1023, 64)
```

### 1.3 One environment trap worth recording

Importing `vlmeval.config` calls `torch.set_grad_enabled(False)` **globally**, so
`is_grad_enabled()` is `False` well before any model is constructed. Every
earlier Stage-1/S1 script therefore ran with autograd silently off — harmless
there (nothing needed gradients) but fatal here: the first probe attempt produced
`logits.requires_grad == False` and a `backward()` error. The probe re-enables it
explicitly and asserts on it. Any future gradient work in this repo must do the
same.

---

## 2. Objective definitions

Let `v_i` be visual token *i*'s embedding (`i = 0..1023`), `g_i = dJ/dv_i`, and
`d` the hidden dimension.

### A1 — answer-conditioned (diagnostic upper bound)

Teacher-forcing on the correct answer, appended after the prompt:

```
J_A1 = sum_{t} log p(a_t | image, question, a_<t)
```

* Target `a` = `ref_prediction` from `fa_probe.json` — the answer the **unpruned
  model itself generated** at Stage-1 probe time, which by construction is the
  correct one (the probe only kept instances where the unpruned model was right).
* Capped at **64 answer tokens**. A no-op for 13/15 cases (2–18 tokens); OCRBench_214
  and OCRBench_222 hit the generation cap at probe time and produced 2048-token
  degenerate digit repetitions, which would otherwise dominate the objective.
* Verified: prompt tail is `<|im_start|>assistant\n`; matched NLL is **0.00–1.46**
  total (per-token 0.000–0.237).

### A2 — answer-free (deployment-oriented)

One forward on the prompt only, take `logits` at the last prompt position (the
first answer position):

```
P1:  k = argmax(logits);  J_P1 = logits[k]
P2:  J_P2 = logits[top1] - logits[top2]
```

No ground truth, no answer token, one backward each.

*Verified:* the P1 argmax decodes to the true answer's first token in **15/15**
cases — e.g. `'ju'`↔"juicy", `'game'`↔"game time", `'our'`↔"our onward day of
celebration", `'Meta'`↔"Meta-analysis of phase III studies…".

### Saliency maps

```
G1   score_i = ||g_i||_2
G1L1 score_i = ||g_i||_1
G2   score_i = sum_d |g_i,d * v_i,d|
```

`G1L1` was recorded as the brief permits; it tracks `G1` to within 0.001 rank in
every tier, so it is reported once and not discussed further. No other saliency
methods were tried.

---

## 3. Controls — and the one that matters

### 3.1 Refuted: "G2 is just feature magnitude"

If the gradient were roughly uniform across the hidden dimension, `G2 = sum_d
|g_i,d * v_i,d|` would degenerate into a function of `||v_i||`, and the "saliency"
would just be "big features win".

| Score | mean rank pct (loc256) | block AUROC | block AP |
|---|---|---|---|
| A1 G2 | 0.3031 | **0.928** | 0.425 |
| `\|\|v_i\|\|` only | 0.5024 | **0.443** | 0.236 |
| shuffled `\|\|v_i\|\|` | 0.4933 | 0.529 | 0.304 |

and `corr(A1_G2, ||v||)` = **−0.067** (range −0.134 to +0.058 across cases);
`corr(A1_G1, ||v||)` = −0.286. So `||v||` alone is at chance and essentially
uncorrelated with the saliency. **Refuted.**

### 3.2 Refuted: "a position prior explains it"

| Tier | distinct blocks hit | most frequent block's share |
|---|---|---|
| loc256 (n=8, primary) | **12 / 16** | 25 % |
| loc64 (n=4) | 2 / 16 | 75 % |

On the primary set the necessary blocks are spread across 12 of the 16 blocks, so
no fixed spatial prior competes. **But note the n=4 tier**: three of its four
cases have block 1 as a necessary block, so on `nNec = 64` a position prior is
partially informative and that tier's headline numbers are correspondingly less
trustworthy. This is an additional reason — beyond sample size — to treat
`nNec <= 256` as primary, as the brief does.

### 3.3 The random null, calibrated

Block AUROC with one positive block per 16 is a coarse, high-variance statistic, so
the null was calibrated empirically rather than assumed to be 0.5 (3 000 random
score draws per tier):

| Tier | n | random block-AUROC mean | sd | p95 | max |
|---|---|---|---|---|---|
| loc64 | 4 | 0.497 | **0.151** | 0.750 | 0.967 |
| loc256 | 8 | **0.501** | **0.092** | 0.651 | 0.793 |
| all | 15 | 0.501 | 0.056 | 0.592 | 0.690 |

Two consequences. First, the brief's Gate A threshold of AUROC ≥ 0.65 sits at
almost exactly the null's 95th percentile at n = 8 — it is a sensible bar, and
0.928 clears it by 4.6 sd. Second, a *single* n=4 AUROC value is nearly
uninformative on its own (sd 0.151): random draws at n=4 ranged up to 0.967. Any
n=4 AUROC number in this report must be read alongside that.

### 3.4 NOT refuted: the signal is largely answer-agnostic

The control the brief did not ask for, and the most informative thing in this
probe. The **identical** A1 objective is run with a mismatched answer — the next
case's answer, cyclically. Same image, same question, same prompt, same visual
tokens, same position; only the teacher-forced continuation changes. The model is
genuinely forced: mismatched NLL is **10.7–104.6** against matched NLL of
0.00–1.46.

| Tier | objective | matched | mismatched | matched AUROC | mismatched AUROC |
|---|---|---|---|---|---|
| loc256 | G2 | 0.3031 | **0.3329** | 0.928 | **0.907** |
| loc256 | G1 | 0.3522 | 0.3809 | 0.863 | 0.773 |
| loc64 | G2 | 0.2800 | 0.3328 | 0.967 | 0.950 |
| all | G2 | 0.3658 | 0.3836 | 0.835 | 0.812 |

Paired over the primary 8 cases:

```
mean dAUROC (matched - mismatched) = +0.021   95% CI [-0.003, +0.046]
mean dRank  (matched - mismatched) = -0.0298  95% CI [-0.0601, +0.0050]
mean Spearman(matched map, mismatched map)     =  0.849
```

Against the empirical null (mean 0.501, sd 0.092 at n=8):

| Score | AUROC | distance from null |
|---|---|---|
| A1 G2 (matched answer) | 0.928 | **+4.64 sd** |
| mismatched-answer G2 | 0.907 | **+4.42 sd** |
| P1 G2 (answer-free) | 0.895 | **+4.28 sd** |
| official EADP | 0.380 | −1.32 sd |

**Reading.** The saliency localizes the causally necessary regions strongly —
4.6 sd above chance — but it does so *whether or not the model is being made to
say the right thing*. Teacher-forcing a completely different answer moves the map
hardly at all (Spearman 0.85) and costs only 0.02 AUROC. The dominant component of
the signal is a general sensitivity of the LM's output distribution to each
visual token, not answer-conditioned attribution.

This is **good news for deployment** and **bad news for novelty**:

* Good: it explains why the answer-free P1/P2 proxies in §5 perform essentially as
  well as the answer-conditioned oracle — they are measuring nearly the same
  quantity, so nothing is lost by not having the answer. The deployment path is
  therefore not a compromise.
* Bad: a contribution framed as "we use the answer to attribute visual tokens"
  would not survive this control. The honest description is a
  **gradient/attribution token-importance signal**, closer in spirit to
  sensitivity-based saliency than to answer-conditioned causal attribution.

One further implication is recorded but **not tested** here: since all 15 causal
cases are error instances whose answers live in dense text, the sensitivity
signal may largely be finding *text-bearing / information-dense* regions. That
would still be a useful pruning signal and would still be far above what EADP's
similarity score achieves, but it is a different claim from "finds the answer
evidence". Testing it needs a text-density annotation this study does not have,
and the brief forbids expanding the probe. It is flagged for S2-B.

---

## 4. Per-case results

Mean necessary-token rank percentile (lower = better). Primary set, `nNec <= 256`,
n = 8.

| Case | nNec | official | S1-best | A1 G2 | A1 G1 | P1 G2 | P1 G1 | P2 G2 | null (mismatch G2) |
|---|---|---|---|---|---|---|---|---|---|
| DocVQA_VAL_241 | 64 | 0.6255 | 0.5564 | **0.1501** | 0.2033 | 0.1849 | 0.2479 | 0.1503 | 0.2138 |
| DocVQA_VAL_341 | 128 | 0.4018 | 0.4716 | **0.3449** | 0.4120 | 0.3186 | 0.3971 | 0.2684 | 0.2793 |
| OCRBench_222 | 256 | 0.6253 | 0.5660 | **0.2459** | 0.2965 | 0.3044 | 0.3443 | 0.2567 | 0.3172 |
| TextVQA_VAL_105 | 64 | 0.5431 | 0.4384 | **0.4378** | 0.4589 | 0.4176 | 0.4504 | 0.4600 | 0.5052 |
| TextVQA_VAL_285 | 64 | 0.5247 | 0.5400 | **0.2325** | 0.2902 | 0.2616 | 0.3196 | 0.2821 | 0.2333 |
| TextVQA_VAL_400 | 64 | 0.5566 | 0.4942 | **0.2996** | 0.3564 | 0.3099 | 0.3690 | 0.2997 | 0.3789 |
| TextVQA_VAL_505 | 128 | 0.6231 | 0.4854 | **0.3380** | 0.3775 | 0.3389 | 0.3830 | 0.3613 | 0.3308 |
| TextVQA_VAL_609 | 256 | 0.4729 | 0.4151 | **0.3757** | 0.4230 | 0.3776 | 0.4247 | 0.3707 | 0.4044 |

Six of the eight arms — A1 G2, P1 G2, P1 G1, P2 G2, P2 G1 and the mismatched-answer
null — improve on the official score on **all 8** cases. The only regressions in
the table are A1 G1 and A1 G1L1 on **DocVQA_VAL_341** (0.4120 vs 0.4018), which is
also the case with the least headroom: official already scores 0.4018 there, the
best of the eight. Note that the null column is below the official score on every
case too — §3.4 seen at per-case granularity.

---

## 5. Aggregates

### 5.1 Primary — `nNec <= 256`, n = 8

| Method | mean rank | median | Δ vs official | impr/wors | AUROC | AP | R@128 | R@256 | R@512 |
|---|---|---|---|---|---|---|---|---|---|
| official EADP | 0.5466 | 0.5386 | — | — | 0.380 | 0.207 | 0.051 | 0.151 | 0.425 |
| S1-best (global only, no smooth) | 0.4959 | 0.4967 | −0.0507 | 6/2 | 0.519 | 0.256 | 0.099 | 0.216 | 0.513 |
| **A1 G2** | **0.3031** | 0.2582 | **−0.2435** | **8/0** | **0.928** | 0.425 | 0.333 | 0.538 | 0.773 |
| A1 G1 | 0.3522 | 0.3351 | −0.1944 | 7/1 | 0.863 | 0.345 | 0.253 | 0.422 | 0.708 |
| P1 G2 | 0.3142 | 0.2683 | −0.2325 | 8/0 | 0.895 | 0.406 | 0.296 | 0.492 | 0.773 |
| P1 G1 | 0.3670 | 0.3393 | −0.1796 | 8/0 | 0.831 | 0.331 | 0.215 | 0.399 | 0.705 |
| P2 G2 | 0.3062 | 0.2618 | −0.2405 | 8/0 | 0.878 | 0.486 | 0.315 | 0.526 | 0.779 |
| P2 G1 | 0.3576 | 0.3359 | −0.1890 | 8/0 | 0.814 | 0.355 | 0.241 | 0.418 | 0.713 |
| *null: mismatched-answer G2* | *0.3329* | *0.2901* | *−0.2137* | *8/0* | *0.907* | *0.412* | *0.274* | *0.456* | *0.755* |
| *control: `\|\|v\|\|` only* | *0.5024* | *0.5049* | *−0.0442* | *6/2* | *0.443* | *0.236* | *0.123* | *0.244* | *0.477* |
| *control: random (null dist.)* | *0.501* | — | — | — | *0.501 ± 0.092* | — | — | — | — |

### 5.2 Secondary — `nNec = 64`, n = 4

| Method | mean rank | Δ | impr/wors | AUROC | AP | R@256 |
|---|---|---|---|---|---|---|
| official EADP | 0.5625 | — | — | 0.367 | 0.069 | 0.129 |
| S1-best | 0.5072 | −0.0552 | 3/1 | 0.500 | 0.086 | 0.191 |
| **A1 G2** | **0.2800** | **−0.2824** | **4/0** | **0.967** | 0.396 | 0.594 |
| A1 G1 | 0.3272 | −0.2352 | 4/0 | 0.900 | 0.267 | 0.441 |
| P1 G2 | 0.2935 | −0.2690 | 4/0 | 0.900 | 0.396 | 0.523 |
| *null: mismatched G2* | *0.3328* | *−0.2296* | *4/0* | *0.950* | *0.417* | *0.473* |

Caveats: n = 4, and 3 of the 4 cases share block 1 as necessary (§3.2), so the
AUROC values here carry a null sd of 0.151 (§3.3). Treat as directional only.

### 5.3 Secondary — all 15 causal cases

| Method | mean rank | Δ | impr/wors | AUROC | AP | R@128 | R@256 | R@512 |
|---|---|---|---|---|---|---|---|---|
| official EADP | 0.5251 | — | — | 0.449 | 0.389 | 0.088 | 0.198 | 0.463 |
| S1-best | 0.5011 | −0.0240 | 8/7 | 0.504 | 0.448 | 0.113 | 0.230 | 0.500 |
| **A1 G2** | **0.3658** | **−0.1593** | **13/2** | **0.835** | 0.542 | 0.255 | 0.434 | 0.688 |
| A1 G1 | 0.3993 | −0.1257 | 12/3 | 0.782 | 0.490 | 0.207 | 0.366 | 0.641 |
| P1 G2 | 0.3769 | −0.1481 | 12/3 | 0.776 | 0.538 | 0.232 | 0.405 | 0.681 |
| P2 G2 | 0.3713 | −0.1538 | 12/3 | 0.803 | 0.579 | 0.241 | 0.424 | 0.686 |
| *null: mismatched G2* | *0.3836* | *−0.1415* | *12/3* | *0.812* | *0.526* | *0.224* | *0.388* | *0.675* |

The diffuse cases (nNec up to 832) dilute the effect relative to the primary set,
as expected: a 256-token budget cannot hold 832 necessary tokens regardless of
ranking quality. Recall@512 for A1 G2 reaches 0.688 across all 15.

---

## 6. Head-to-head: official EADP vs S1-best vs gradient

Primary set, `nNec <= 256`, n = 8:

| | official EADP | S1-best ablation | gradient (A1 G2) | gradient (P1 G2) |
|---|---|---|---|---|
| mean necessary rank pct | 0.5466 | 0.4959 | **0.3031** | 0.3142 |
| block AUROC | 0.380 | 0.519 | **0.928** | 0.895 |
| block AP | 0.207 | 0.256 | 0.425 | 0.406 |
| Recall@256 (necessary tokens) | 0.151 | 0.216 | **0.538** | 0.492 |
| cost per image | ~0.9 ms (scoring stages) | ≪1 ms (reuses the global branch) | ~515 ms (fwd+bwd) | ~507 ms |

The gradient signal is in a different regime from both baselines. EADP's
similarity score is *below* the random null at block level (−1.3 sd); the
gradient signal is 4.6 sd above it. Recall@256 rises from 0.151 to 0.538 — the
fraction of answer-bearing evidence surviving a 256-token budget improves
**3.6x**.

The S1-best ablation is included only as a stronger scoring-only baseline, as the
brief specifies. It is a reweighting of the same similarity signal and cannot
leave that regime; S2-A confirms that from the opposite direction.

---

## 7. Efficiency

Measured per case on the unpruned 1024-token configuration, A40 46 GB, SDPA:

| Quantity | ms (mean over 15) |
|---|---|
| prompt-only forward (P1/P2) | **215.8** |
| one backward | **291.5** |
| A1 forward (prompt + answer) | 224.3 |
| A1 backward | 293.6 |
| peak memory | 23 094 MB |

Reference points from Stage 1:

| Component | ms |
|---|---|
| EADP scoring stages (global/dense/fusion/smooth/polar) | ~0.9 |
| EADP facility-location selector | ~40 |
| EADP pruning overhead total | 42.0 |
| unpruned LLM prefill | 228.4 |
| unpruned full prefill (vision + LLM) | 345.4 |

**The deployable unit (P1 = one forward + one backward) costs ≈ 507 ms.** That is
**12x** the entire EADP pruning overhead it would replace, and **2.2x** the
unpruned prefill it is supposed to accelerate. On these numbers a gradient-scored
pruner would be slower end-to-end than not pruning at all.

Per the brief this is recorded and **not** used to reject: signal value was to be
judged first. But the gap is not a tuning gap — it is an algorithmic one, and it
is the first thing S2-B has to solve. Note also that the forward here is the
*unpruned* 1024-token forward; the cost is dominated by the backward through 8B
parameters, so most of it is irreducible without approximations (e.g. a partial
backward through the last *k* LLM layers, or a single-layer surrogate).

---

## 8. Gate A — does the gradient carry causal information?

Criterion: answer-conditioned gradient must reach **all** of Δrank ≥ 0.10,
block AUROC ≥ 0.65, ≥ 6/8 cases improved, on `nNec <= 256`.

| Method | Δrank | AUROC | improved | Verdict |
|---|---|---|---|---|
| A1 grad-norm (G1) | **+0.1944** | 0.863 | 7/8 | **PASS** |
| A1 grad×input (G2) | **+0.2435** | **0.928** | **8/8** | **PASS** |
| A1 grad-L1 | +0.1942 | 0.858 | 7/8 | **PASS** |

All three clear every bar with substantial margin. **Gate A passes.** The brief's
NO-GO condition — "even answer-conditioned gradient sensitivity fails to localize
causally necessary visual evidence" — does not apply.

---

## 9. Gate B — is there a usable answer-free proxy?

Criterion: answer-free gradient must reach block AUROC ≥ 0.60 **or** Δrank ≥ 0.08,
**and** ≥ 6/8 cases improved, on `nNec <= 256`.

| Method | Δrank | AUROC | improved | Verdict |
|---|---|---|---|---|
| P1 grad×input (G2) | **+0.2325** | **0.895** | **8/8** | **PROMISING** |
| P1 grad-norm (G1) | +0.1796 | 0.831 | 8/8 | **PROMISING** |
| P2 grad×input (G2) | +0.2405 | 0.878 | 8/8 | **PROMISING** |
| P2 grad-norm (G1) | +0.1890 | 0.814 | 8/8 | **PROMISING** |

Every answer-free arm clears both alternatives of the criterion, with 8/8 cases
improved. **Gate B passes.**

The brief's third outcome — "causal signal exists, but current answer-free
objective fails" — does not apply either.

---

## 10. Final verdict

```
S2-A GO
```

Both gates pass; the answer-free deployment proxy is not merely viable but nearly
as strong as the answer-conditioned oracle. Per the brief, the next stage is to
measure actual pruning accuracy with a gradient-derived score.

### What must be carried forward, unresolved

1. **The signal is largely answer-agnostic (§3.4).** Matched and
   mismatched-answer saliency maps correlate at Spearman 0.85 and differ by 0.02
   AUROC; the mismatched version alone is +4.4 sd above the null. The method
   should be described as a gradient/attribution *sensitivity* score, not as
   answer-conditioned attribution.
2. **The cost is prohibitive as measured (§7).** ~507 ms per image against a
   228 ms unpruned prefill. Signal value was the question here; cost is the
   question next.
3. **n = 8, all error cases, single benchmark family.** The 15 causal cases are
   Stage-1 error instances only — by construction the hardest ones. Nothing here
   says what the score does on instances where EADP already succeeds.
4. **Block-level granularity limits block AUROC** (16 blocks, one positive in the
   localized cases; null sd 0.151 at n = 4). The token-level rank percentile is
   the finer instrument and behaves consistently with it.
5. **The text-density hypothesis is untested (§3.4).** If the sensitivity signal
   is largely a text detector, that is still useful but is a different claim, and
   it would need a control this study does not have.

Task stops here. No full-split accuracy, no selector integration, no S2-B.

---

## 11. Reproduce

```
source ~/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
cd /media/disk2/YZX/research/EADP/Qwen_vl
python scripts/discovery/s2a_hook_sanity.py      # hook correctness (GPU: vision tower only)
python scripts/discovery/s2a_gradient_probe.py    # A1 / P1 / P2 saliency + timing
python scripts/discovery/s2a_input_norm.py        # ||v|| control data
python scripts/discovery/s2a_null_control.py      # mismatched-answer control
python scripts/discovery/s2a_controls.py          # confound tables
python scripts/discovery/s2a_analysis.py          # gates A/B
python scripts/discovery/s2a_consolidate.py       # writes the deliverable JSON
```

Deliverable: `outputs/discovery/s2a_gradient_viability.json` — per-case records,
all 1024-dim saliency maps, per-case metrics, per-tier aggregates, controls, gate
outcomes and timings. Supporting: `s2a_hook_sanity.json`,
`s2a_null_control.{json,npz}`, `s2a_input_norm.npz`, `s2a_metrics.json`.
