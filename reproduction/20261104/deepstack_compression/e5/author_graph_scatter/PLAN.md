# One-statement mask-scatter hypothesis

The unmodified author graph probe failed at the mask assignment during stream
capture for K=1,16,128,256,512. CUDA reports may surface asynchronously, so the
reported line does not by itself establish the failing operation. Retain that
attempt unchanged and diagnose capture in separate isolated GPU work only after
review and scheduling.

For a batch-row index `b`, the existing assignment sets `mask[b,best[b]]` to
true. `mask.scatter_(1,best.unsqueeze(1),True)` sets the same one position per
row, including repeated index values across distinct rows. It does not alter any
floating operation, summation order, argmax, tie handling or loop order in the
source. This is a mathematical statement for valid in-range indices, not proof
that the GPU runtime is bitwise identical or capture-compatible.

Build the candidate only by replacing that one exact author AST statement; all
other author nodes remain untouched, and the author file stays unchanged. CPU
tests compare full ordered selections and quota, including batch >1, ties,
zeros, near ties, seeded inputs and the archived real K512 failure input.
Real input CPU versus saved GPU order is reported separately because CPU and GPU
reductions may differ; CPU original versus CPU scatter must match exactly.

The planned GPU diagnosis isolates the two mask writes, indexed row read, and
boolean mask assignment in separate CUDA graph processes; then captures the
whole candidate for K=1,16,128,256,512 in separately bounded processes. Each
whole run compares against direct unmodified EADP on the same GPU, retaining all
ordered choices, same-process A/B/A, tie, zero, random and the real near-tie
case. Each subprocess is limited to 300 seconds. K512 requires at least 32 GiB
free before allocations; other full captures require at least 12 GiB free.
Each GPU child sets the PyTorch allocator to at most 60% of total device memory;
CUDA context, driver and graph metadata can consume additional memory, so this
is not a device-wide peak guarantee. Record actual free/total, allocator
allocated/reserved and capture time for each child. Preserve failure/OOM/timeout
without treating it as an identity mismatch. A positive microprobe is not a
model-level identity, model TTFT or final-method claim.
