# Minimal real GPU sampling handshake, no model

One retained same-image container, GPU0 only, private nonce and PID-1 worker.
Before CUDA initialization the worker publishes ready; host confirms container
image, user, command, mounts, device nodes, cgroup/PID/start ticks/boot and one
idle A40 UUID, starts a separately supervised real `nvidia-smi -q -x` sampler,
and releases only after a pre-CUDA sample. Worker allocates/touches exactly
16 MiB of uint8 CUDA tensor, holds for 8 seconds, frees it, synchronizes and
exits. Host requires real Docker stopped/PID0/exit0, sampler terminal and
post-exit sample window, and verifies every sanitized device metadata window
against each sample. Compute concurrency aborts; graphics occupancy reported.
This tests lifecycle and sampled memory only, not model performance, FLOPs,
per-request memory or an exact peak. MiB rounding and sample gaps can miss
transients; target absence is unknown, not zero. CUDA context and allocator
make process bytes different from 16 MiB. Fake 1 MiB sample never enters this
run. No labels, predictions, model or model weights. CPU-only preflight has no
GPU device or allocation. Real GPU requires independent reviewed proof plus
root scheduling, single run, deadline/budget, hard timeouts and stop/kill.
