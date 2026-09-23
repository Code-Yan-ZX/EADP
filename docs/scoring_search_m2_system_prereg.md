# M2 — Gradient-Distilled Early Pruning (GDEP): the system Pareto (pre-registration)

**Written before the first line of the GDEP inference engine, before any
correctness test, before any timing and before any generation.** Frozen and
dated 2026-09-23. If a rule below turns out to be wrong it is amended in
`scoring_search_m2_system.md` with the amendment dated and the pre-registered
version quoted, not silently overwritten here.

`GDEP` is an **engineering identifier for the configuration under test**. It is
not a novelty claim. Whether the configuration is worth anything is decided by
the accuracy–efficiency Pareto, and by nothing else.

---

## 0. What M2 is, and what it is not

M2 is the **first stage in this project that implements and measures a real
method prototype end to end**, on the same code path for accuracy and for
latency. Every stage before it (S1, S2-A…S2-C6, M1) was a *proxy diagnostic*:
it scored token sets offline, or it injected a cached importance map into the
official pre-LLM pruning harness. M1 (§8 Q6) opened the deploy gate on a literal
reading; M2 is the stage the gate was for.

M2 **re-tunes nothing, trains nothing, adds no module and searches no
architecture**. It takes exactly the artefacts M1 left on disk, wires the
existing 524 673-parameter `LOCAL-MLP` scorer into a real early-layer pruning
graph, and measures both axes of the Pareto.

**Out of scope, explicitly:** no query, no trajectory, no global or set context,
no rescue path, no new scorer, no wider scorer, no second-stage student, no
hyper-parameter sweep, no new teacher map. If the result is bad, M2 stops and
reports why the Pareto failed — it does not open a search.

---

## 1. Fixed setting (inherited, unchanged)

| item | value |
|------|-------|
| Backbone | `Qwen3-VL-8B-Instruct`, VLMEvalKit wrapper `Qwen3VLChatEADP` |
| Input resolution | 1024 × 1024 fixed (`QWEN3_FIXED_RESOLUTION`) |
| Visual tokens | 1024 merged (64×64 patches, `spatial_merge_size = 2` → 32×32) |
| Text prompt | the benchmark's own VLMEvalKit prompt, unmodified |
| Budget `T` | 256 retained visual tokens |
| Attention backend | SDPA (`QWEN3_VLM_ATTN_IMPL=sdpa`); no flash-attn wheel exists for this torch/CUDA/py3.10 |
| Decoding | greedy (`do_sample=False`), `max_new_tokens = 2048`, identical for every arm |
| Hardware | 1× NVIDIA A40 46 GB |
| Torch / transformers | 2.10.0+cu128 / 4.57.6 |
| Scorer | `L4` / `LOCAL-MLP`, `s = w₂·GELU(P h₄ + b)`, **524 673 params**, unchanged from S2-C6/M1 |
| Teacher (offline only) | S2-B P1-G2 gradient maps — **never computed in the deployment path** |
| Held-out | the frozen 150: `bank[::3]`, 50 per benchmark × {TextVQA_VAL, DocVQA_VAL, OCRBench} |

The scorer's per-dimension standardisation mean/std **follows the fit set**: the
n = 960 arms read `m1_stats_n960.npz`, the n = 240 control reads
`m1_stats_n240.npz`. Neither is recomputed here.

---

## 2. The method under test

At layer **L4** (0-based decoder-layer index 4, the S2-C0/S2-C1/M1 convention —
the output of the 5th of 36 decoder layers), with the **full 1024 visual tokens
still present**:

1. the vision tower produces 1024 merged visual embeddings, as always;
2. visual embeddings are spliced into the prompt at the image-token positions,
   and the prompt runs through decoder layers 0…4 **once**, at full length;
3. the visual slice of the layer-4 hidden state, `h₄ ∈ R^{1024×4096}`, is read
   off the residual stream;
4. the **online** scorer produces `s = w₂·GELU(P ĥ₄ + b)` — 524 673 parameters,
   one GEMM and one GELU;
5. a selector consumes `(s, similarity(h₄))` and chooses 256 of the 1024 visual
   tokens;
6. all text tokens are kept **unconditionally**; the hidden states, the
   layer-0…4 KV cache and the position ids are compacted to the kept set;
7. decoder layers 5…35 run **only** over the kept 256 visual tokens plus the
   full text;
8. generation continues normally from the compacted state.

**L0–L4 run exactly once. No recomputation, no second forward, no prefix that is
thrown away and re-run.** The scorer runs online: no cached score vector is read
at inference time. (The offline teacher maps are used *only* to train the
scorer, offline, in M1, and are not part of any measured configuration. Their
generation cost is reported as an amortised training cost and is **excluded from
every latency number**.)

### 2.1 Selectors

The selector is a *system* choice, not a method variable, and all three are the
Stage-1/D1 instruments, unchanged:

| tag | selector | source |
|-----|----------|--------|
| `topk` | plain Top-256 on `s` | `diag_selectors.select_topk` |
| `block8` | block-parallel facility-location greedy, block = 8 | `diag_selectors.select_block_greedy` |
| `facility` | official EADP facility-location greedy | `pruner._greed_select_impl` |

`block8` and `facility` also consume a visual-visual similarity matrix. Inside
GDEP that matrix is computed from **`h₄`**, the same tensor the scorer reads
(`_sim_visual_impl`, `sim_mode = rebound`, the official kernel). This is stated
because it is a degree of freedom: the incumbent computes it from the vision
tower's post-merger embeddings, which GDEP never materialises. It is not a
tuned choice and no alternative was tried.

### 2.2 Position policy (pre-registered, and why)

**PRESERVE is the primary policy.** A kept token carries the position id / RoPE
angle it had in the full-length sequence; the compacted sequence's `position_ids`
are `sorted(kept_indices)` and are therefore **not contiguous**. The next
generated token takes `max(position_ids) + 1`. The kept tokens are consequently
seen by layers 5…35 at exactly the relative distances the full model would have
seen them at. Renumbering is what the brief forbids and it is also the thing
that would make the pruned run a different positional problem rather than a
sub-sequence of the same one.

**RENUMBER is carried as a declared control, not as a candidate.** Under
RENUMBER the compacted sequence is assigned `arange(0, L')`. This is what the
incumbent pre-LLM harness does (it hands HF a short `inputs_embeds` and HF
assigns `0…L'-1`), so RENUMBER is the policy that makes a GDEP-vs-EADP
comparison like-for-like on positional encoding. It is run on the primary
candidate only (C1), and reported separately. **It is not a second method.**

Both policies are implemented in the same engine and the policy in force is
recorded in every result record. No other arm (B0/B1/B2) has a position policy
to vary: B0 is unpruned and B1/B2 are pre-LLM.

### 2.3 KV-cache policy (pre-registered)

Three configurations exist in this comparison and each has exactly one policy.
**Every policy is reported per arm; none is silently shared.**

| arm | early-layer cache (L0–L4) | late-layer cache (L5–L35) |
|-----|---------------------------|---------------------------|
| B0 full-1024 | 36 layers × 1058 tokens | — |
| B1/B2 (pre-LLM prune) | **does not exist** — no LLM layer runs at 1024 tokens | 36 layers × 290 tokens |
| **GDEP (C0/C1/C2/C3)** | 5 layers allocated at **1058** tokens, then **compacted to 290** | 31 layers × 290 tokens |

The GDEP policy is **compact-L0–L4**: after selection, the keys and values of
the dropped visual tokens are removed from the layer-0…4 caches, so layers 5…35
attend only over the kept 290 positions. This is the policy that makes the
pruning real (a kept token's late-layer representation never attends to a
dropped token), and it is the policy under which pruning actually reduces KV
footprint. The transient 1058-token allocation of layers 0…4 is measured and
reported separately from the steady-state 290-token footprint, because it is a
real cost of pruning late rather than early.

The consequence — stated here so it is not discovered later — is that GDEP's
layer-0…4 attention for the kept tokens was computed over the full 1058-token
context, whereas B1/B2's was computed over 290 tokens from the start. **GDEP and
B1/B2 are therefore different computation graphs and their logits are not
expected to agree.** No test in §4 asks them to.

---

## 3. Arms

`n` is the M1 fit-set size the scorer was trained on. Seeds are M1 seeds 0/1/2,
each a separately trained checkpoint; the architecture and the compute graph are
identical across seeds, so seeds are an accuracy-only axis.

### Baselines

| id | arm | budget | description |
|----|-----|--------|-------------|
| **B0** | Full model | 1024 | no pruning at all |
| **B1** | Official EADP `facility` | 256 | the incumbent, official scoring + facility location, pre-LLM |
| **B2** | EADP `block8` | 256 | the D1 efficiency fix on the incumbent's own importance map, pre-LLM |

### Candidates

| id | arm | n | selector | policy |
|----|-----|---|----------|--------|
| **C1** | M1 L4 LOCAL-MLP + TopK | 960 | `topk` | PRESERVE |
| **C2** | M1 L4 LOCAL-MLP + block8 | 960 | `block8` | PRESERVE |
| **C3** | M1 L4 LOCAL-MLP + facility | 960 | `facility` | PRESERVE |
| **C0** | M1 L4 LOCAL-MLP + TopK | 240 | `topk` | PRESERVE |

### Declared controls (reported, not candidates, carrying no Pareto verdict)

| id | arm | why it exists |
|----|-----|---------------|
| **C1-R** | C1 under RENUMBER | isolates the position policy; makes the B1/B2 comparison like-for-like |
| **C1-ID** | C1 with `T = 1024` | the K = 1024 identity arm of §4.3; must reproduce B0 |
| **C1-SHUF** | C1 with the importance map rotated onto another image | the S2-B content-free control, re-run inside the real engine |

`C1-SHUF` is the control that decides whether any measured Pareto movement is
attributable to the *learned score* or merely to *pruning to 256 tokens by some
map with the right statistics*. It was the decisive control in S2-B (§2.2) and
it is the decisive control here.

---

## 4. Correctness gates — all must pass before any accuracy or timing number is reported

Six gates. **Any failure stops M2 and the failure is the report.**

### 4.1 G-A — online L4 extraction matches the M1 cache

For 8 held-out instances: run the full-length prompt through decoder layers 0…4
inside the GDEP engine, take the visual slice, and compare against the
corresponding rows of `s2c1_feats_L4.npy` (fp16 ground truth produced by
`s2c1_features.py`'s hook). Report **max and mean absolute difference** and the
reference scale. Pass: worst `max|Δ|` below 1 % of the cache's mean `|h|` — the
same tolerance `m1_features.py` gate G1 used, chosen there to exclude a
*different computation* rather than an fp16 ulp. A wrong layer index, a wrong
token slice or a different attention mask lands orders of magnitude above it.

### 4.2 G-B — online scorer matches the cached scorer

On the same 8 instances, feed the *cached* `h₄` rows through the same
`LOCAL-MLP` with the same standardisation, and compare the two score vectors.
Pass: (i) report max/mean `|Δs|` relative to the score's own scale; (ii) the
**Top-256 index sets must be identical**. If they are not, every differing token
is listed with its two scores and the cause is explained; an unexplained
difference is a failure.

### 4.3 G-C — K = 1024 identity

With `T = 1024` the selector keeps everything and the compaction is the identity.
The engine's layer-by-layer forward must then reproduce the stock
`Qwen3VLModel.forward` logits **exactly enough to be the same computation**.
Pass: `max|Δlogits|` over the checked instances below an explicit tolerance
fixed here as **1e-3 absolute** on the pre-softmax logits, against a logit scale
of order 10. This gate is what certifies that the engine's manual layer loop,
causal mask, RoPE and cache handling are the model's own.

### 4.4 G-D — K = 256 invariants

For every instance of the accuracy run, the engine asserts and records:

* exactly **256** visual tokens retained (never 255, never 257);
* **no duplicate** index, **no index out of `[0, 1024)`**;
* **all** text tokens retained, count unchanged;
* `attention_mask` length `= position_ids` length `= hidden_states` length, all
  equal to `256 + n_text`;
* KV-cache sequence length `= 256 + n_text` at every layer, after compaction;
* under PRESERVE, `position_ids` is strictly increasing and a subsequence of
  `arange(n_full)`;
* under RENUMBER, `position_ids == arange(256 + n_text)`.

Pass: zero violations across all instances and arms. Any violation is reported
with the offending instance key.

### 4.5 G-E — generation sanity

On ≥ 10 instances: greedy generation completes, produces non-empty output, and
the run records **no NaN**, no shape fallback, no exception-driven path, and no
implicit full recomputation (asserted by instrumenting the engine to count
decoder-layer invocations and comparing against the expected
`5 + 31 × (1 + n_decode)` per token budget).

### 4.6 G-F — harness identity against the published pre-LLM baselines

The engine's **pre-LLM mode** at budget 256, with the official importance map
and facility selection, must reproduce the published held-out numbers
`TextVQA 70.267 / DocVQA 64.380 / OCRBench 66.000` (`diag_selectors_b256.json`,
`S2-B` §2.1) to **±0.5 points per benchmark**. This is the check that the
engine's decode loop, prompt construction and scoring are the incumbent's, and
it is the reason B1/B2 can be quoted from the same engine as C1/C2/C3.

---

## 5. Pre-registered success criteria (thresholds frozen before measurement)

All accuracy criteria are evaluated on the **held-out 150**, macro over the three
benchmarks, three seeds, **paired bootstrap with 10 000 resamples**. The
efficiency criteria are evaluated on the profiling run of §6.3.

### 5.1 Primary Pareto success

```
C1 (n=960 TopK) vs B1 (official EADP facility@256):
  paired 95 % CI lower bound of the macro accuracy difference  >  -1.0 point
  AND
  median end-to-end TTFT reduction vs B1                        >=  15 %
```

### 5.2 Strong compression success

```
any candidate (C0/C1/C2/C3) vs B1:
  macro accuracy drop                                          <=  2.0 points
  AND at least one of
      median TTFT reduction vs B1                              >=  25 %
      peak memory reduction vs B1                              >=  20 %
```

### 5.3 Accuracy-led Pareto success

```
any candidate vs B1:
  macro accuracy difference significantly > 0 (paired 95 % CI lower bound > 0)
  AND
  median TTFT                                                <=  1.10 × B1's
```

These three are **independent** and more than one may hold. If none holds, that
is the result and §8's four failure readings are applied instead.

### 5.4 What is not a success

Teacher-recall improvement, Top-256 overlap, AUROC, selection cost, FLOPs, or
any offline token-set metric. **A candidate that wins on any of those and loses
on the Pareto has lost.** Every table in the report carries quality and
performance side by side; no table reports a token-set metric alone.

---

## 6. Measurement protocol

### 6.1 Accuracy

Same engine, same code path as the timing run. Held-out 150, greedy.

| arm family | n | selector | seeds |
|-----------|---|----------|-------|
| C0 | 240 | topk | 0/1/2 |
| C1 | 960 | topk | 0/1/2 |
| C2 | 960 | block8 | 0/1/2 |
| C3 | 960 | facility | 0/1/2 |
| C1-R | 960 | topk, RENUMBER | 0/1/2 |
| B1, B2 | — | facility / block8 | seed-independent |
| C1-SHUF | 960 | topk, rotated score | 0/1/2 |

Reported: per seed, seed mean and range, per benchmark, macro, paired bootstrap
against B1, and the **error transitions** (rescued / still-wrong / newly-broken)
against B1 and against B0.

**No test split, and no downstream accuracy, selects the seed, the checkpoint,
the policy or the threshold.** The held-out 150 is measured once per frozen
configuration and is never used to choose anything.

### 6.2 The profiling checkpoint

The model, the scorer and the selector all have to be fixed for a latency
measurement to mean anything, and three seeds give three different scorers.
**One checkpoint is used for profiling, selected by a validation-only rule
fixed here:**

> Among the three n = 960 `H8` checkpoints, take the one whose **validation
> (60-image) `head_recall@8` at its selected checkpoint** is highest; ties break
> to the lower seed index.

From `m1_train_fixed-step.json` this is **seed 2** (validation `head_recall@8`
0.8167 / 0.8125 / 0.8125 for seeds 2 / 0 / 1). No held-out quantity is read to
make that choice, and the rule is applied to the record on disk, not re-derived.
The n = 240 control uses the same rule, which selects **seed 0** (0.7854 /
0.7833 / 0.7729). All three seeds are still run for accuracy; the rule only
picks which checkpoint the stopwatch sees, and it is applied once, before any
timing.

### 6.3 Performance

Primary hardware: the A40 already in use, same CUDA/PyTorch/dtype. **Batch size 1
is primary**; batch size 4 is secondary if the memory and the time allow, and if
it is not run that is stated rather than implied.

Per arm: **≥ 20 warm-up iterations, ≥ 100 measured iterations**, a
`torch.cuda.synchronize()` around every measured region, and **both CUDA-event
timing and synchronised wall-clock** recorded. Reported: mean, median, P90 and
standard deviation — never the mean alone.

Input control: fixed image resolution (1024×1024), fixed prompt construction,
**no variable-length generation inside the prefill benchmark**. The prefill
benchmark ends at the first generated token. The decode benchmark generates
**exactly 32 tokens** for every arm with the same generation configuration.

Mandatory decomposition, per arm, in this order:

```
 1  image preprocessing
 2  vision encoder
 3  EADP scoring                      (B1/B2 only; 0 for others)
 4  facility / block8 / TopK selector
 5  L0-L4 full-token forward          (GDEP only; 0 for B0/B1/B2)
 6  LOCAL-MLP scorer                  (GDEP only)
 7  token + KV compaction             (GDEP only)
 8  L5+ pruned forward
 9  total model prefill               (= 5+6+7+8 for GDEP, = 8 for B0/B1/B2)
10  end-to-end TTFT                   (1+2+3+4+9)
11  32-token decode total
12  TPOT                              (11 / 32)
13  peak torch.cuda.max_memory_allocated
14  visual-token KV-cache memory, split early (L0-L4) and late (L5+)
15  throughput / requests per second
```

Also reported: speedups relative to B0, to B1 and to B2; the scorer's parameter
count; **the offline teacher-label generation cost, amortised over the n = 960
training images and explicitly excluded from every deployment latency figure**;
and a boolean per arm for **did an extra forward or a recomputation occur**.

### 6.4 Accuracy and performance must be the same implementation

The engine that produced §6.1's predictions is the engine that produced §6.3's
timings. This is checked mechanically, not asserted: the accuracy runner and the
benchmark runner import the same module, and the engine's configuration hash
(model revision, scorer checkpoint, selector, budget, position policy, cache
policy) is written into both result files and compared at consolidation.

---

## 7. Deliverables

| path | contents |
|------|----------|
| `docs/scoring_search_m2_system_prereg.md` | this document |
| `docs/scoring_search_m2_system.md` | the report |
| `Qwen_vl/outputs/discovery/m2_system_audit.json` | machine-readable main record |
| `Qwen_vl/outputs/discovery/m2_correctness.json` | G-A…G-F |
| `Qwen_vl/outputs/discovery/m2_perf.json` | the 15-field breakdown, per arm |
| `Qwen_vl/outputs/discovery/m2_accuracy.json` | per-image accuracy, all arms × seeds |
| `Qwen_vl/outputs/discovery/figures/m2_pareto.png` | accuracy vs TTFT, accuracy vs peak memory, stage breakdown |
| `Qwen_vl/scripts/discovery/m2_gdep.py` | the engine |
| `Qwen_vl/scripts/discovery/m2_{correctness,perf,accuracy,consolidate,figure}.py` | the runners |
| `Qwen_vl/scripts/discovery/run_m2.sh` | the reproducible command sequence |

Large artefacts (model weights, feature caches, KV dumps) stay gitignored, as in
every previous stage.

---

## 8. The only conclusions M2 is allowed to draw

Exactly one of these, chosen by the pre-registered criteria and the measurements:

```
P1  GDEP is a new Pareto point.
        §5.2 or §5.3 holds, and the gain survives C1-SHUF.
P2  GDEP has better accuracy but the performance cost is too large.
        accuracy is at or above B1, but no §5.1/§5.2 efficiency bar is met.
P3  GDEP is faster but the accuracy loss is unacceptable.
        an efficiency bar is met, but the accuracy drop exceeds 2.0 points
        (or the §5.1 CI lower bound falls below -1.0).
P4  The L4 pruning point is too late; the distillation has to move into the
    vision encoder / pre-LLM stage.
        the L0-L4 full-token forward plus compaction costs so much of the
        saved prefill that TTFT does not move, i.e. the mechanism is
        structurally unable to pay for itself at this depth.
P5  The learned score works, but the selector / system implementation is the
    bottleneck.
        C1-SHUF shows the score is carrying real content-specific signal
        (C1 materially above C1-SHUF), while no candidate clears a §5 bar.
```

**If the result is bad, M2 stops.** No new module, no architecture search, no
rescue path, no extra hyper-parameter. M2 reports which of P1–P5 applies, names
the measured quantity that decided it, and states what the next stage would have
to change. Nothing is designed here from a failed result.

---

## 9. What M2 will not claim

* Not that GDEP is novel. Gradient-distilled token selection and early-layer
  pruning both exist in the literature; the contribution under test is a
  *measured configuration*, and the report says so.
* Not that the L4 scorer's teacher-recall (0.8214 at n = 960) implies downstream
  accuracy. M1's gate opened on teacher recall; **M2 is the stage that tests
  whether that quantity means anything downstream**, and a failure of that
  translation is a result about the teacher metric, not about the engine.
* Not that a Pareto loss on this hardware generalises to another. The
  selector-to-prefill cost ratio is machine-specific and the M1/Stage-1 reports
  already record that it does not transplant.
* Not that the held-out 150 is a full split. It is 50 per benchmark, and every
  absolute number is a subset number.
* Not that the incumbent has been beaten on a full split, on other benchmarks,
  or at other budgets. Only T = 256 is measured.
