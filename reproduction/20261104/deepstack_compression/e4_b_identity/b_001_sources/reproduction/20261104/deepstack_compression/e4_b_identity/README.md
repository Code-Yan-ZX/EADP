# E4-B identity preparation

Status: CPU preparation only. The `b_gate.py` lock refuses execution while the
approved panel, independent B review and full container preflight are absent;
there are no B GPU results, timing measurements, or accuracy scores.

The predetermined diagnostic uses the first E4 smoke anchor from each task,
which is the same deterministic three-group selection as the accepted native
diagnostic. Each anchor has C-DROP and C-SKIP natural arms. The first anchor
also has the same two arms using a fixed 33-token prefix (32 actual decode
steps). Outputs must be journaled immediately after `generate` returns.

The record contract compares both natural arms to the accepted E4 K256 ON and
OFF anchors on inputs, visual backbone output, selector order, retained IDs,
P3 position and prefill. DROP and SKIP must match each other on actual prefill
and logits, every natural output token, and actual positions/cache fingerprints;
all visual blocks and the main merger must run, while the three dedicated DS
merger results must be real tensors in DROP and `None` in SKIP. The forced
pair must have identical prefill and 32-step positions and KV fingerprints.
Fingerprints do not permit reconstruction of raw KV cache arrays.

The separate CPU verifier reads back diagnostic image/processor, main visual,
pruned visual, actual prefill, logits and available DS arrays, checks their
hashes and reconstructs the selected 256 visual rows and actual prefill slice.
No labels are needed. Instrumentation uses hooks, cloning, and KV hashing and
therefore cannot be treated as a performance measurement.

Before any B GPU execution, a bounded launcher, externally supervised
same-container preflight, exact source/weight bundle, completed E4-A acceptance,
independent B review, exclusive GPU check, three-day deadline and real
container-exit chain remain required. The fail-closed gate itself also requires
independent review. Never run this while E4-A is using the GPU.
