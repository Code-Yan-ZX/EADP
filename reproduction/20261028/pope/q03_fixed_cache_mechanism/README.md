# Fixed16 POPE entropy retention q=.2 to .3

Prepared only. This is one public-ablation-supported mechanism hypothesis,
not a main-table default, identified paper argv, confirmed repair or full
benchmark. No GPU allocation, automatic queue or new model is created by
preparation. A separate root note bound to the final plan is required.

The fixed original16 image/question cache records are reused. Parameters remain
alpha.5/beta1/K32. The sole mathematical change is the actual
`entropy_keep_ratio` keyword .2 to .3 in the source `encode_images` importance
AST. Its inverse restores the entire original importance AST. The original
filter's softmax, entropy, floor/topk/minimum-one rule and topk ordering operation remain:
77 dense positions retain15 for q=.2 and23 for q=.3. No quantile-threshold Eq6,
Eq7 denominator, Eq10 epsilon, ENV ENTROPY flag, ablation class or other recipe
is changed.

The graph starts BEFORE entropy filtering at saved full FP32 global/dense77/
visual similarities. Each arm independently runs the original filter, local
aggregation, fusion, normalization, smoothing, power and facility arithmetic.
Actual original FP16 projected visual features are reused losslessly. This
does not recalculate similarities, extract features, load7B/CLIP/FA2, read GT or
saved answers, generate with an LLM or compute accuracy.

The old native encode anchor was alpha1/beta1 and supplies the common real
features. The targeted .5/beta1 source importance/ordered32/full32-step gains/
effective mask are same-run GPU helper results. They are not a separate native
encode or actual LLM prepare anchor. A spatial FP16 sequence is derived exactly
from the saved projected array and effective mask; the old cache did not save
an actual prepare/LLM embedding sequence.

For each image, source importance, ordered32 IDs, the entire gain trace and
effective final mask must match those old .5/1 helper tensors byte-for-byte.
The derived spatial FP16 sequence must also match its old-cache derivation.
Float comparisons include dtype, shape and bytes, including signed zero. Only
after all anchors pass may q=.3 be measured for that image. Failed source
anchors are saved as explicitly unmeasured with diagnostic deltas; their
variant is skipped. Audit success may include such unmeasured cases. No CPU
approximation, flag tuning, new cache selection or automatic retry fills them.

The original helper greedy body is preserved; the gain observer is one
post-computation clone and its removal exactly restores the original AST. The
traced output must equal an unobserved original greedy call. Both arms preserve
source final-mask construction. The CPU terminal checker audits NumPy cache
bytes, full16 coverage, gain argmax/history, source gates, actual worker exit,
all frozen SHA and output-array metadata; it never replays pruning or scores.

Gain differences are separately reported along a common selection history.
Whole-trace differences after histories diverge may include the selected-mask
-1e9 sentinel at different positions and are not described as numerical
perturbations. Internal greedy order, final set/mask and derived spatial FP16
sequence differences are distinct observations, not answer-quality claims.

Runtime uses the original torch2.1.2+cu118/CUDA11.8 environment. Pure CPU import
has been checked with CUDA hidden; no CUDA device was queried. Original runtime
recorded FP32 default/autocast false/matmul TF32 false/cuDNN TF32 true, but not
every cuDNN or workspace flag. Current defaults are recorded without tuning;
strict source byte anchors decide whether the variant is measurable.

`run.py` reuses the reviewed cache-only single-attempt controller framework.
Default check is stdlib-only. A GPU run checks the exact allowed other GPU
PID/starttick/boot identities, root note/plan/parent identity and maximum2
concurrent GPU contexts including loading. An O_EXCL claim permits one attempt;
the controller waits for the real worker exit before one CPU terminal audit.
Every failure/partial result is retained. If a controller fails while its child
is still alive, that is not proof that its GPU slot is released; root must
check the real original identities. Root separately confirms the controller's
actual PTY return after release. No benchmark ledger is changed here.
