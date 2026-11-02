# Full K64 question-only pruning guidance (prepared, not launched)

This is one full5000 TextVQA input hypothesis supported by the inactive public
loader comment at model_vqa_loader.py:113. It is not known paper argv and is not
labelled a bug fix. The completed fixed16 source/variant experiment passed its
source anchors and changed actual visual input; that terminal proof is frozen
as a prerequisite. No accuracy or closeness to paper decides whether to run.

Parameters remain CLI64/nominal320, alpha.5, beta1, three caller-stream waits,
SDPA FP16, greedy max128, vicuna_v1, all5000 original questions and images.
Only the text passed as `texts=` changes: insert
`question=question.split('\nReference OCR token')[0]` before the already active
answer-suffix removal. Full LLM OCR conversation/input IDs, CustomDataset,
image processing, native pruning arithmetic and generate kwargs are untouched.
The source AST insertion is removed in an inverse check that restores the
entire original entry AST. The original source file is never edited.

`worker.py` imports the frozen completed K64 adapter, changes isolated output
paths, and binds a local runpy object solely in that framework's namespace.
`guidance_patch.py` executes the transformed native entry and records both its
AST SHA and actual eval_model code SHA. The old entry file SHA is provenance,
not a claim that the executed eval_model remained unmodified.

`observe.py` reuses the old budget/input-ID observer and additionally hashes
the actual `texts` string only after native generation returns. All5000 hashes
must match the already measured question-only guidance references. The prior
LLM input SHA, actual crop quota/mask/effective-prefix and output-token
re-decoding gates remain. Native floor/min1 allocation is preserved; the actual
total is not forced to320.

`verify.py` retains the complete-only original author score, independent
pointwise consensus, fresh author CLI and displayed-score gates. It additionally
requires the full5000 guidance hash checks before reading GT/scoring. It never
scores a running prefix. Both positive and negative full results are retained;
there is no parameter sweep, best-score selection, automatic retry or queue.

Default `run.py check` is CPU-only and imports no torch. A later explicit root
allocation bound to the final plan is required for the single attempt; exact
other-process PID/starttick/boot identities and the two-model global maximum
are checked. The worker must actually exit before the one CPU terminal
verification. Release records the worker exit; root must separately confirm
the controller execution returned. No GPU launch is authorized by preparation,
and the accepted ten-row ledger is not modified here.
