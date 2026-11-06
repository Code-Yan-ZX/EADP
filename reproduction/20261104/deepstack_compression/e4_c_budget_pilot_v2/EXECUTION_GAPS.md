# Work required before GPU scheduling

The current successful preflight does not authorize GPU use. A complete
diagnostic worker and retained Docker controller still need to be implemented
and independently reviewed. The worker must bind the accepted v2 preflight,
its independent review and their actual supervisor exits; the fixed 120-source
inventory; accepted E1 and E4 proofs, exits, inputs and records; all four
weight payloads; and the exact author recipe and actual imported code. The
same checks must fail closed before model loading when any proof is missing or
changed. Its actual container command, image, user, devices, mounts, started
process identity, source hashes, deadline and supervisor must be frozen and
testable without a GPU.

The worker then has to execute the already prescribed 16 paired jobs and
capture the full tensors and first logits. The forced token is independently
derived from the fixed tokenizer, and each actual `generate` call is guarded
against changes in all six accepted scalar controls, EOS and the exact forced
processor/token. The raw generated sequence is fsynced immediately; partial
or failed arms, comparison errors and cleanup outcomes remain observable.
All source and weight identities must be checked again at the end.

The controller needs an outer supervisor, a generous but finite inner wall
limit, stage progress and stop/kill/final-inspect closure, including signal and
client-early-exit failures. Limits must fit the remaining user deadline and
GPU remains exclusively scheduled by root. A fresh no-GPU full-worker
preflight and independent source/container review are required after the
worker and controller are frozen; the current preflight cannot stand in for
that test.

After 32 diagnostic generations actually complete, a separate verifier and
independent reviewer must check full tensor archives, all position/delta,
selector/DeepStack and cache traces, output sequences, Docker and supervisor
termination before enabling any latency pilot. The separately planned latency
worker needs its own review, actual first-token boundary and monitoring
disclosure. No true performance inference is made from this diagnostic.
