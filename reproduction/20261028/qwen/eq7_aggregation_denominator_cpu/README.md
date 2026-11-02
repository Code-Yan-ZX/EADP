# Fixed16 Eq7 aggregation denominator: CPU cache reference

This consumes only the sixteen completed `gpu_formula_facility_panel_run02`
caches. It does not load a model, access ground truth or historical answers,
generate text, calculate accuracy, use CUDA, or edit the original experiment.

The two arms retain q=.2, alpha=.5, beta=2 and K256. The only numerical delta is
the `negative_entropy` denominator: source `M_temp + 1e-6` versus literal Eq7
`M_temp`, with M_temp=.01. Both source AST inverse checks and post-call observer
inverse checks run before arithmetic. Eq6 membership/order and Eq10 numerator
and denominator retain the original source behavior.

The graph starts with the saved native GPU FP32 global/dense cross-modal
similarities and visual similarity matrix. Real BF16 projected features and
text embeddings are checked from the same cache; their encoder and cross-modal
GEMMs are not rerun. The unchanged source `compute_importance` AST consumes
these explicit cached similarity inputs. CPU entropy filtering, aggregation,
smoothing, polarization and greedy selection then execute the original source
arithmetic. Observers keep references to logits/weights/local similarity after
their computation and clone gain arrays before argmax; a separate unobserved
source greedy invocation must match the observed CPU source.

The report first compares CPU source importance, all256 ordered IDs, sorted
spatial IDs and gain traces against the archived native GPU source. Any failure
of bitwise closure makes facility comparisons **conditional CPU references**,
even if final sets happen to match. No CPU variant is a GPU measurement. Gain
trace deltas after a first selection divergence use different histories and
are explicitly marked descriptive. Compressed outputs contain small numeric
arrays, rather than another copy of the BF16 image/text feature payloads.

Run once with the author interpreter:

```sh
CUDA_VISIBLE_DEVICES='' /media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python -B run_cpu.py run
```

`plan.json` freezes source, worker, parent report and each NPZ SHA. Output is
exclusive-create and existing partial evidence remains after any failure.
`comparison.json` and all referenced sources are checked at the end. No retry,
GPU launcher, model loading or automatic expansion is provided.

A later separately authorized GPU-only feature-cache replay could run pruning
without loading8B. It would still need the same pinned math/environment and an
exact source anchor (importance, all ordered IDs and gain traces) before any
variant result could be called an actual GPU diagnostic. This directory does
not implement or schedule that future replay.
