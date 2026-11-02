This is a prepared bounded mechanism diagnostic, with no allocated GPU or container.
The panel is exactly floor((2*i+1)*5349/32), i=0..15, in the verified complete Doc
canonical order. Position and canonical question ID are recorded separately. No
answer/GT/score influences selection. The worker reads only the frozen question,
message, image and input fingerprints, not the canonical GT or prediction files.
The canonical file's SHA is preparation-only provenance. It is excluded from
runtime hash guards, so even its GT-containing file bytes are not opened by the
worker; the immutable selected question/message/input chain is checked instead.

Only the already completed native K256/alpha.5/beta2 recipe is used. Once per
question, the unchanged author tokenizer/input embedding path and
_get_pruned_image_features call supply the real post-merger BF16 visual features,
BF16 instruction sequence and its native BF16 global mean. The native pruner uses
FP32 negative cross-modal cosine; visual similarity retains the original BF16
normalization before FP32 matmul. No CLIP adapter, contextual text LLM forward,
DeepStack restoration or position change is introduced. Both outer and visual
model attention backends must actually be FA2 in the established author runtime.

Post-call observers retain references and return the original objects. The native
visual call and pruning each execute once. Afterward, the four cases reuse those
same real tensors: source, strict source-members/order identity, Eq6 threshold
membership only, and Eq10 numerator-epsilon only. The actual LLM text length L is
used, not 77. The threshold convention is torch.quantile(q=.2, linear), including
all ties. This convention is explicit; the paper did not certify a specific
interpolation implementation. Source top-floor members/order are retained and
new members are appended in original token-position order. Ragged membership is
explicitly unmeasured, never padded or processed with a substitute reduction.

The source anchor requires exact native importance, ordered256 greedy IDs and
the final spatially sorted BF16 feature sequence. The identity case requires all
gains/importance/IDs/final sequence identical. The trace only clones an already
computed gain, and must match unobserved greedy IDs. No efficiency or entirely
unobserved scheduling identity is claimed. An anchor mismatch is preserved as
failure/unmeasured, never reported as a formula outcome.

Real BF16 arrays are stored as uint16 bit patterns with their actual dtype,
shape and raw-byte SHA; FP32 global/dense/importance/sim arrays remain FP32.
Each variant records ordered IDs and final spatial IDs separately, plus final
sequence fingerprints. These are feature/selection results, not benchmark
scores: model.generate is explicitly forbidden. No answer, accuracy or API is
produced, and the plan has no automatic extension to a full dataset or recipe.

Historical numeric caches exist; see cache_inventory.json. Missing qualification
means incomplete fixed-question/native-source/precision evidence, not absence of
all local numeric data. Zero fixed-question position overlap does not establish
zero shared-image overlap. A shared visual cache would still need qualified
BF16 values, correct source/runtime and the matching complete text/global/dense
native anchor. Older cache rows must not replace this predetermined panel.

The default diagnose.py or run_panel.py check is CPU-only and imports no torch.
A future run requires a new exact root allocation note with authorization,
run_id, run_plan_sha256, GPU_slot, allowed_other_gpu_pids and corresponding
allowed_other_gpu_processes [{pid,start_ticks,boot_id}]. The note's SHA is shared
between host and worker. Current GPU PIDs, process identities and fewer than two
loaded models are checked before launch; loading counts. The exclusive claim
permits one attempt. Fixed Docker image/driver mounts, host init PID/startticks,
cgroup/container ID and actual exit are recorded. No queue/retry is created.

failure.json does not mean a live container or model has exited. Root must check
actual processes/GPU state before releasing a failed slot. Successful controller
state verifies the native container exit; root confirms the controller exit
after its command returns. Existing author source, environments, full-generation
plans, predictions, official scores and frozen audits remain unchanged.
