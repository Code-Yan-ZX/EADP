# Eq7 denominator CPU result

All sixteen fixed cached questions completed once with author torch2.8.0,
CUDA_VISIBLE_DEVICES empty, two CPU threads, no initialized CUDA and no model.
Worker2804985/startticks1333248456 and execution77094 exited successfully.
Original source, parent reports and all sixteen NPZ hashes remained unchanged.

The source CPU replay did **not** reproduce GPU importance bitwise on any
question (maximum absolute difference2.2172928e-5). Nevertheless, all256 ordered
IDs and the sorted spatial set matched the archived GPU source on every
question. Gain traces also differed numerically (maximum0.00592041). The Eq7
results below are therefore conditional CPU references, not GPU measurements.

| Same-cache source .010001 versus literal Eq7 .01 | Result |
|---|---:|
| Questions with changed entropy weights |16/16|
| Maximum absolute weight change |2.1755695e-5|
| Questions with changed local aggregate |13/16|
| Maximum absolute local aggregate change |1.5925616e-6|
| Questions with changed importance |11/16|
| Maximum absolute importance change |9.4771385e-6|
| Questions with changed ordered256 |1/16|
| Questions with changed final spatial256 |0/16|

Only position4847/index6963 changed selection order. At zero-based step197,
both arms still share the same previous selection history: source picks144
with gains[0.08160069585,0.08160036057] at candidates[144,837]; Eq7 picks837
with corresponding gains[0.08159997314,0.08160009980]. Both candidates occur in
the eventual same256-token set. Sorting for the native spatial output removes
this order difference: both CPU arms reconstruct exactly the archived native
BF16 projected feature sequence on all sixteen questions.

The maximum gain perturbation over valid unselected candidates, while the
selection histories remain identical, is0.0026397705 across the panel. The
whole-trace maximum1e9 in the raw comparison is the selected-mask sentinel at
different positions after that one order divergence; it is not overflow or a
meaningful same-history numerical error. The supplementary boundary report
keeps these two scopes separate.

No answer, GT, benchmark score, logit or generation was read or computed.
This does not establish a paper configuration, a cause of the full-run gap,
or a negative result for all5349 questions.

A later separately authorized GPU feature-cache replay is technically possible
without loading8B: this pruner has no learned parameters and the caches already
contain real BF16 image/text features plus FP32 similarities. A replay starting
after the similarity graph cut could use the original source arithmetic and
one case at a time. It must reproduce saved source importance, all256 ordered
IDs and the entire source gain trace **bitwise** before accepting any GPU
variant result; otherwise preserve the failure as unmeasured. Recomputing the
original cross-similarity/visual-similarity from cached BF16 values would require
separate bitwise gates at those earlier cuts. This is not an end-to-end
native_get or LLM test. No GPU launcher or GPU allocation was prepared here.

Authoritative numeric evidence is `comparison.json`,
`gain_and_final_sequence_boundary.json` and `CPU_completion.json`; each has
frozen file/array hashes. The compressed new CPU arrays total4,060,351bytes.
