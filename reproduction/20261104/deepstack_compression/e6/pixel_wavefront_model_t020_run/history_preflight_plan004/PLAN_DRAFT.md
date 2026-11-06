# Threshold 0.2 full-model identity run draft

Six processes, one fresh retained container per arm. Three previously examined DEV anchors, two arms each, with alternating order from the independent no-GPU plan. For each process preserve raw returned sequence before any validation, actual full ordered selector result, prefill embeddings, masks, P3 and three DeepStack inputs, and per-step 36-layer KV fingerprints. The latter are fingerprints only, not archived KV tensors. Compare each output with its already accepted t020 same-arm model smoke and compare author vs wavefront; a failed arm ends the run and is preserved.

The no-GPU import proof is prerequisite only. The eventual execution plan must bind all running sources and upstream results, actual retained container identity, exclusive GPU, source/weight invariance, deadline, parent supervisor/PID/group/argv, timeouts and stop/kill cleanup. Each arm must pass its actual PID/cgroup handshake before model loading. A separately reviewed formal verifier and independent final reviewer are required before any timing. Candidate and preflight reviewer provenance must be explicitly disclosed.

Current GPU gate is unconditional rejection. The controller, plan, no-GPU execution-chain validation and final review below are work in progress: no GPU entry is authorized.
