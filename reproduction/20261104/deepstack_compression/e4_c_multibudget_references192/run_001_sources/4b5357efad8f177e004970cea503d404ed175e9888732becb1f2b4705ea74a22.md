# Four-budget fixed32 references for the 192-image expansion

Generate fresh references for every input: 192 DEV measurement images and 30
distinct warmup images, each at K=128, 256, 512 and 1024, for 888 arms. The
frozen proposal and its independent review are the input authority. No older
fixed32 outputs are reused. Each image and prompt is bound to accepted E1/E4
identity and processor tensors; same-budget natural prefill, P3 positions,
three DeepStack streams, selection and the actual generation recipe are
checked. Each run generates exactly 32 tokens, with 31 decode steps following
prefill. Records contain selected tensor summaries, prefill KV fingerprints,
and per-layer cache lengths/positions; they do not archive full KV contents.

This is an exploratory reference generation run, not timing or scoring. The
previously accepted E1/E4 DEV mixed prediction files are parsed only to bind
input and prefill identities. No GT or new scoring is permitted. Record each
arm's start and raw generated sequence immediately with fsync, retain failures,
and reject partial runs for acceptance. A separately frozen CPU test/source
plan, independent preflight review, real no-GPU processor/weight preflight,
independent GPU prelaunch review, complete supervisor/container exit, formal
no-GT verification and independent result review are required. GPU scheduling
is reserved to the project controller.

The earlier 248-arm diagnostic completed in about 596 seconds including its
preflight and load; a linear extrapolation is speculative and may miss output,
I/O and host variation. Proposed hard bounds are 7200 seconds in-container,
7320 seconds outside, 1200 seconds progress stall and 90 seconds cleanup grace,
with 3600 seconds additionally reserved before the user's deadline for
acceptance. Every timeout/failure remains recorded, and these diagnostics are
not used to claim latency or speed.
