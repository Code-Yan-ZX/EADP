# Four-budget natural TTFT pilot

Use the same three frozen, distinct DEV anchors as the 32-generation
four-budget identity diagnostic. For K128, K256, K512 and K1024 use the
author selector and reviewed P3/DeepStack-on adapter. K1024 is identity
keep-all and must not call greedy. Three samples per budget and anchor are
exploratory, not a panel-level latency estimate. These anchors and accepted
E4 quality outputs are already known; this is not a blinded or independent
choice of inputs. The prior three-anchor Pixel/EADP timing pilot is also
already known and is not combined as a paired measurement.

Use one loaded model, 12 warmups and 36 measured samples in deterministic
hash-randomized complete four-budget blocks (`contract.py`). Each actual
natural output, EOS and scalar recipe must equal the already accepted E4
same-arm output. Include image open/decode, preprocessing, vision, selector,
P3/DeepStack and the decoder in every timing. T0 follows a GPU synchronization
and precedes image read; the first-token timestamp follows a synchronization
inside the Transformers stopping criterion after argmax and token append. The
adapter hooks have real cost and remain enabled for all four budgets. Record
the observed hook path and use of actual K. No per-layer probes, tensor hashes
or NVML polling inside the latency worker.

Raw token outputs are fsynced immediately after `generate` returns. That is
after the first-token boundary but before the worker's final boundary, so the
latter contains journal overhead and is not clean end-to-end latency. Warmup
and measurement journals, all failed attempts, health-query windows, host
background and overlapping/unknown cases must be kept; no samples removed.
This pilot does not claim an isolated-process or whole-device memory peak.

GPU remains locked until both formal and independent real-model 32-generation
identity proofs, the actual performance worker/launcher source and container
preflight, user budget, GPU exclusivity, and an independent prelaunch review
are bound. A separate memory experiment needs independent process sampling.
