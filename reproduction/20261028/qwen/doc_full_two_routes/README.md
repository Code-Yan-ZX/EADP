# Qwen DocVQA full-route calibration, CPU preparation

This candidate runs all 5349 original ordered DocVQA questions in the existing
FA2/BF16 environment. It creates a single `Qwen3VLChatEADP` instance with
visual_token_num=1024 and explicitly invokes the unchanged native FixedRes
multimodal method and the unchanged inherited CDPruner/EADP inputs_embeds
method in this fixed order for every question. Each method independently runs
the original processor and generate; the model and weights are common. The
1024 setting is not an author-registered configuration, and this is a full-path
calibration, not the EADP paper's main-table recipe.

Preparation reads the canonical dataset (including answer fields) to extract the full ordered input manifest, but does not copy answers or use them to select questions or score. The manifest contains question, image identity and previously qualified processor checks only; it contains no answers. Native predictions are saved and
flushed before recording or checking observations. Actual processor tensors and
output token IDs are observed; processor tensors are cloned for identity checks,
which adds memory and time overhead, and no per-layer visual tensors are saved.
The original wrapper has post_process=False, which the worker verifies and records; the independent CPU verifier reconstructs all responses from the output IDs.
Native source, model payload and all1286 images are checked; every raw outcome
and full denominator is retained. Controller waits for the GPU container's real
exit, then runs the non-scoring verification. Root must separately confirm the
actual controller/PTY exit and write its proof. Only after that proof may the
separate CPU scoring operation read ground truth, call the author's original
DocVQA evaluator and calculate independent per-question ANLS and paired scores.
No automatic GPU queue, auto-retry, scores on prefixes or source edits exist.

The fixed16 predecessor showed equal visual features and initial embeddings,
while position IDs and DeepStack handling differed, with 3/16 changed output
sequences. This full comparison tests the combined paths, not a single-cause
DeepStack or position intervention. The original author main-table argv remains
unknown. Historical FULL results are not an independent same-environment baseline.
