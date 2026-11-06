# Separate isolated-process memory measurements (CPU preparation only)

The interleaved latency process must not run this sampler. A future memory run uses one fresh, exclusively assigned GPU process for each configuration and records PyTorch allocator/KV observations inside that process; host sampling reports one MiB-rounded, whole-process NVML maximum per configuration, which can miss transient peaks. It is not a per-question maximum and is not a TTFT measurement.

`contract.py` checks a pre-release barrier and completed sample chronology; `sampler.py` is an injectable sampling engine. The tests simulate identities, gaps, exits and GPU contention without executing nvidia-smi or loading a GPU model. These files do not create a Docker container, establish that a host PID/cgroup is the real container worker, start a supervised sampler, or release a worker. All of those capabilities, plus independent review, a retained Docker exit chain and the real E4-C identity proof, remain required before use. The current production memory entry remains locked.

The first CPU-test invocation was mistakenly launched from the parent directory and exited 2 before tests, as recorded in `initial_command_failure.txt`. Subsequent supervised tests are retained under distinct `monitor_cpu*` directories.
