# C-ON isolated-memory smoke preparation

One future C-ON worker runs 24 jobs derived from the accepted E1 DEV schedule,
with a fresh container and a separately supervised process sampler. The host
must first validate the reviewed source and allocation, budget, one A40 UUID,
and no other compute processes. Existing graphics use is separately recorded.
The worker performs real CPU-only input/weight preparation before publishing
its PID-1 CUDA-uninitialized ready event. It may load the model only after the
host has bound the actual Docker and host PID/cgroup, a live sampler, and the
first sample with no target compute process. Sampling continues beyond stopped
container inspection. Timeout and signal cleanup must stop/kill/inspect, and
unknown state is failure.

NVML target-process peak, total device used/capacity, background graphics,
PyTorch allocated/reserved peaks and logical KV sizes are different measures.
Interval samples can miss transient peaks, and missing values stay unknown.
GT is never read; existing accepted E1 DEV prediction journal is parsed only
for input identity. No execution entry currently grants GPU authorization.
