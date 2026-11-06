# No-GPU controller integration rehearsal

Exercise the existing controller, real Docker container, worker PID-1 barrier,
real supervisor, stopped-container inspection and sampling state machine using
an explicitly synthetic sampling source. This is not GPU memory measurement.
The fake source returns no process before release, a fabricated 1 MiB entry
while the worker is alive after release, and no process after exit. No CUDA,
model, NVML, nvidia-smi, ground truth or predictions are used.

Use a fresh private nonce and output directory, a retained container, no GPU
devices, no network and a bounded worker/sampler/supervisor. The existing
supervisor may collect its ordinary host GPU health snapshot; that snapshot is
not used as the fake sampler's data. Save actual Docker
inspect, supervisor terminal, handshake files, complete synthetic journal and
failure cleanup. Check all identities and the sampled time-window contract;
never report the fabricated byte count as hardware evidence. Before execution,
independently review source, tests, actual Docker command and termination path.
Root retains scheduling authority. Existing runtime files are immutable.
