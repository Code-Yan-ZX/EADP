# Process and device memory sampling

Each future sample uses one bounded `nvidia-smi -q -x` query on the reviewed GPU.
The XML is parsed in memory and never persisted: process names may contain
sensitive command-line arguments. The persisted, sanitized metadata contains
UUID, device framebuffer total/used/free/reserved, and each visible PID, type
(compute, graphics, or both) and process memory. The process memory for the
worker is separately passed into the existing controller/sampler contract;
other compute processes reject the run, while graphics processes are recorded
as background rather than silently described as an idle device.

Each XML query records monotonic start/end; the independent verifier must bind
each metadata record to the enclosing existing sampler query window. Counts,
device total/used and worker PID memory are separate measurements: no device
total or used value may be labeled as the worker process memory. Values are
MiB-rounded sampled observations; transient peaks between queries may be
missed, and a sampled maximum is not an exact physical peak. Missing values
are explicitly unknown. Device graphics or other usage may change at any time.

CPU preparation only: no CLI here authorizes a GPU job. No process names,
full command lines, predictions or GT are logged by this sampler.
