# Author scatter CUDA graph: 32-image component expansion

The question is whether a single-statement mask-scatter variant of the author
greedy loop, captured once and replayed with changing input, preserves every
ordered selected index and quota on the 32 already archived DEV image inputs at
K=128,256,512. This is a component identity experiment, not model accuracy,
end-to-end TTFT or an assumption of a final method. The fixed 32 images include
image_017, where the separate fused candidate differed at K512.

Reuse `selector_real32/run_001/image_NNN/case_K.pt` in image index order 000–031
for each K. Formal and independent real32 evidence must bind every archive SHA;
the preceding scatter single-case GPU experiment must have passed formal and
independent evidence gates. Do not open GT or new predictions. Each K runs in a
fresh GPU subprocess with a hard limit, captures once from image_000, and replays
each of 32 inputs after copying both tensors. Compare the full graph order and
quota against an immediate unmodified author GPU call for the same input, and
compare that author order against the independently verified archive. Persist
each case immediately, including all complete orders and failure states; never
drop an input or choose a subset based on its outcome.

Before allocations require 12 GiB free (K128/K256) or 32 GiB free (K512) and
cap the PyTorch allocator at 60% of total device memory. This does not bound
driver/context/graph metadata. Record actual free/total and capture setup wall
time and allocator allocated/reserved/peak before/after, and measure replay on
the stream with input copies included separately from graph-only replay. These
are exploratory component timings, not isolated NVML peak or model TTFT. Each K
has its own fresh process and graph pool. A failed/OOM/timed-out K is preserved
as such and is not conflated with a numerical selection mismatch.

Nominal subprocess budget is 600 seconds per K and 2400 seconds internal, 2520
seconds external, 90 seconds termination grace, and 720 seconds without a
progress event. The small precursor showed all five K captures completed for one
image; its capture/setup and replay observations are exploratory and cannot
bound 32 images, longer environments or predict future performance. Abort on
insufficient user time or unknown GPU occupancy. Independent prelaunch and
post-run reviews remain required; GPU scheduling belongs to the root task.

Only after this component identity passes may a separate three-anchor fixed32
model experiment be designed. It must validate actual prompt/processor, all
ordered selections and output tokens versus the ordinary author path, record
capture setup and graph-pool lifetime separately, and treat any mismatch as
failed identity. The result would remain a small TTFT diagnostic, not a full
benchmark or an evidence-free assumption of deployability.
