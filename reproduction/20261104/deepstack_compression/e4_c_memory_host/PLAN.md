# Future real C-ON memory smoke host controller

Draft only. A future root-scheduled 24-job run requires a reviewed worker,
real-sampler source, CPU preflight and retained-container review, a bound
supervisor and time budget, a fresh allocation, and one available A40 with no
other compute PIDs. Existing graphics processes are recorded, not stopped.
The worker must remain pre-CUDA until a verified host PID/container identity,
worker-ready event, sampler supervisor and first exclusive compute observation
are bound. The worker, sampler and controller each have finite limits.

On failure or signal stop, then kill if needed, inspect the owned nonce-labeled
container, retain unknowns and stop the sampler. Do not label an unknown state
as a clean exit. After Docker stop, require the sampler's final query to start
after the independently recorded stop event and validate every XML metadata
window against its enclosing process sample. A zero or unknown GPU process
observation is not the device's total VRAM, and sampled peaks can miss spikes.

This directory grants no permission and contains no GPU entry point. Existing
DEV predictions are parsed only for input identity; no GT or new prediction
is read or generated during CPU checks.
