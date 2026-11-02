# Qwen fixed 16: prefill DeepStack toggle (CPU candidate)

CPU preparation only. Controller and independent terminal verifier are provided, and the candidate requires independent code review, real CPU container preflight, and explicit root GPU allocation before any run; no GPU has been started or queued here.

A single native FA2/BF16 model and the fixed16 previously measured inputs are
used. First rerun the exact 16 plain and 16 EADP K1024 endpoints; save outputs
before testing byte-level prefill/features/position/cache/rope/output anchors
against the completed earlier panel. A failed endpoint preserves its artifacts
and stops before the two new corners. Only after all32 endpoint anchors pass,
run two more native generations per question: plain with only prefill
`visual_pos_masks` and `deepstack_visual_embeds` removed, and EADP K1024 with
those fields supplied from the same run's actual image-token mask and three
actual native vision tower DeepStack outputs. A missing prerequisite leaves the
original call unchanged, records that it was not measured and stops after
preserving any generated result. No weights, source, native position IDs, rope_deltas, explicit KV-cache arguments or generation parameters are directly changed. Altering DeepStack can naturally change the computed KV contents and later generated tokens. The mask cardinality check synchronizes and reads the actual mask before prefill, adding observation overhead.

This is two controlled within-path DeepStack toggles under two distinct native
input APIs. The two paths have distinct position construction, mRoPE deltas,
input representation and potentially other generation state. Even when position
and DeepStack are displayed as a 2×2 grid, the two native APIs are not a fully
controlled position-factor intervention; results cannot identify an isolated
3D-position causal effect. Capture prefill inputs, selected DeepStack fields, masks, position IDs, cache positions and rope_deltas at the final registered PyTorch pre-hook (not inside attention), and every later forward position/cache/delta and DeepStack/mask presence record; after outputs diverge, their subsequent trajectory steps are not equivalent conditions; compare generated output IDs without ground truth
or scoring. Native snapshots and injection are observation/intervention work
and add memory, time and potentially scheduling differences.

Earlier M13 DeepStack tests used an older environment and different samples;
the present candidate concerns only these 16 fixed inputs and the current
FA2/native K1024 paths. It is not a full DocVQA metric or a paper main-table
recipe. Source and previously completed results remain unchanged.
