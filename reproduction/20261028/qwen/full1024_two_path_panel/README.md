# Qwen fixed 16: native full multimodal vs EADP 1024 paths

CPU-prepared candidate only. No GPU allocation, model launch, queue or result is included. The controller, CPU container preflight and independent terminal verifier are prepared for independent review before any possible execution.

The accepted fixed16 DocVQA questions, exact source/model/input guards and
original FA2 environment are reused. One new Qwen3VLChatEADP instance has
`visual_token_num=1024`, `alpha=.5`, `beta=2`; 1024 is not an author-registered
EADP benchmark configuration. Fixed 1024×1024 input yields 1024 image tokens;
the author's `token_num >= N_i` branch bypasses feature selection. Explicitly
call the original FixedRes method and the original inherited CDPruner method
on this same instance and its same native model and processor, once each per
question, in fixed order. Both retain the original greedy generation kwargs.

The first path calls `model.generate(**inputs)` and processes multimodal inputs
inside Qwen3. The second calls the author EADP feature builder then
`model.generate(inputs_embeds=..., attention_mask=...)`. Their DeepStack and
mRoPE treatment may differ. Observe each actual forward with hooks that clone
GPU tensors at vision, processor, pruner and LLM prefill boundaries; cloning
adds measurement overhead but does not replace native arithmetic. Serialize only after the original generation returns or raises. Observation errors and repeated hooks are recorded without interrupting native calls; raw response and available arrays are saved before validation. Hook cloning is an observation operation and can affect timing. Save native full output IDs and
comparable generated suffix separately, the actual features, embedding,
positions, masks, DeepStack tensors and raw decoded response. Keep artifacts
before measuring pair differences. The original cached 16 native visual
features are an additional reference, not a fabricated full-generation anchor.

This is an observational two-path comparison, not an intervention isolating
DeepStack or position semantics, and is not a benchmark or accuracy estimate.
No ground-truth labels, scoring, paper parameter claim or score selection is
involved. Source, weights, existing artifacts and original model classes remain
unchanged. Any mismatch is recorded as a measurement, not silently corrected.
