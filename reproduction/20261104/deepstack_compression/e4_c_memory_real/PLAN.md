# Real NVML isolated-memory controller preparation

The eventual run is a separate process per configuration, initially a bounded
24-job C-ON smoke. The controller must independently validate the no-GPU fake
handshake review, frozen sources, supervisor and budget, host A40 UUID and empty
compute process list before creating its private Docker container. It must check
actual image/entrypoint/arguments/environment/network/devices/mounts, host PID,
start tick and cgroup before publishing identity; CUDA remains uninitialized
until worker readiness and a real host sampler's first exclusive sample. The
real sampler must cover through a query that begins after stopped Docker inspect.

All subprocesses need bounded timeouts. On any failure or signal preserve raw
evidence, stop then kill if needed and inspect ownership; an unknown state is a
failure. Keep the sampler under a separate supervisor and bind both terminals.
NVML is MiB-rounded periodic process sampling and may miss transient peaks;
PyTorch allocator peaks and logical KV accounting are separate quantities.

This CPU preparation parses the accepted E1 DEV prediction journal to extract
input identities; it does not open GT or generate new predictions.
No CLI in this directory can authorize a real
worker, model load, CUDA or GPU work. Root schedules any eventual GPU smoke;
the previous fake 1 MiB has no physical interpretation.
