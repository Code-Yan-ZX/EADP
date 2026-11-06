# Four-budget DEV endpoint diagnostic: independent review scope

GPU execution is locked. This directory is an unadmitted diagnostic candidate,
not a latency runner and not a conclusion about real-model identity.

The intended paired identity check uses three accepted, distinct DEV image
anchors, four P3/DeepStack-on budgets and both an instrumented E4 budget
adapter and the new lightweight hook. It therefore has 12 natural pairs and
four additional forced 33-token pairs on the first anchor: 16 pairs, 32
generations. Each generated token sequence must be fsynced before subsequent
observation or comparison. Every pair must agree in actual selected indices,
prefill and 3 DS arrays, position IDs, masks, output IDs, first logits and KV
fingerprints. Natural reference records must also agree with accepted same-arm
E4 panel output. Forced pairs check 32 actual subsequent decode steps and all
36 KV layers. The force token must be established independently from the fixed
tokenizer by the future GPU worker and verifier.

The full-position delta assertion is valid only for batch 1 without padding:
`full_delta = max(full_position_ids) + 1 - original_length`, then
`adjusted_delta = full_delta + 1024 - K`. `preflight.py` independently checks
that each actual anchor attention mask is all ones. K1024 is identity ordered
indices 0..1023 with zero calls to greedy, rather than calling greedy at
full budget. Both are CPU contracts until model-level verification.

`monitor_cpu006` completed with return code zero and 18 tests. The candidate
worker checks all six actual generation parameters and EOS against the
accepted same-arm E4 record at the real `generate` boundary; forced calls also
require fixed-length processor and EOS settings, the processor code object of
the respective frozen runner, and a captured token equal to the fixed
tokenizer encoding. Incorrect arguments are rejected before the underlying
model call. The previous
`e4_c_budget_pilot` directory and its real Docker preflight are immutable
evidence for version 1 only; this version changes the comparator to permit an
absent initial cache as well as 36 zero-length layers. It needs a new frozen
source inventory, true no-GPU Docker preflight, independent review and its
own admitted, bounded GPU worker/launcher before execution. The additional
`worker_candidate.py` has no executable GPU entry and is not covered by the
version 1 source inventory; its new pre-generation gate also needs review and
coverage in the next frozen source inventory.

The preflight reads already-accepted E1 mixed DEV prediction journal and E4
DEV predictions for input and same-arm identity. It does not open labels or
score. Model-level data, no-GPU container exit and source proof are separate
from the independently reviewed GPU prelaunch proof; no partial evidence is
an authorization to load a model or use the GPU.
