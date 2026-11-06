# Author-op CUDA graph feasibility probe

This is a local, fixed-shape execution experiment. It captures the unchanged
author `model.pruner._greed_select_impl` directly; it does not use the Triton
selector, rewrite a reduction, change the author source, or claim semantic
identity before actual replay. CPU preparation does not initialize CUDA and
there is no authorized GPU entry in this draft.

The archived DEV input is real32 `image_017/case_512.pt`: the verified record
reports the author and E4 endpoints equal, and fused and fused-graph ordered
selections unequal at K=512. Its float32 `importance` and `similarity`, plus
its stored author order, are fixed by SHA. This is an exploratory input, not a
new prediction or an accuracy experiment.

When separately reviewed and scheduled, run the unmodified author function on
identical inputs on the same GPU as the reference for every comparison. Probe
small captures K=1,16 before K=128,256,512; each graph has a separate pool and
is released before the next. For each successful capture compare all ordered
indices and quota, not just the final set. Include ties, zero scores, seeded
input, the archived real K512 input and a shape-preserving changed input; for
one captured shape run A, B, A and check each replay against a fresh author
call, including the restored A, to detect stale state and copy failures. Use
exact dtype/device/shape checks and save every outcome, mismatch and exception.

Record capture/setup wall time, allocator allocated/reserved immediately before
and after capture, and end-to-end replay host/device time including both input
copies as distinct from graph-only replay. Captured outputs alias persistent
graph storage and must be copied before the next replay when archived. Pools,
other allocations and pre-existing GPU/graphics use are reported separately;
allocator deltas are not device-wide peak memory. CUDA graphs may be unsupported
for some author operations, numerical ordering may still differ, and the
unrolled K512 graph can exhaust memory. Run every step with hard deadlines and
bounded stop/kill/inspect; preserve all partial work and failures. Do not treat
a successful microprobe as a model-level, accuracy or TTFT result.
