# PixelPrune τ=.2 wavefront: full DEV comparison

The full 384-question run, source snapshot, formal verification and independent result review completed successfully. The original τ=.2 and wavefront runs have the same ordered selected positions and natural output token IDs in all 384 questions; decoded output, EOS termination and output length also match in all 384. Each task contributes 128 questions, and all 384 wavefront generations ended in EOS. No row was dropped for a mismatch.

| Task | Questions | Ordered selection equal | Natural output tokens equal | EOS termination equal |
|---|---:|---:|---:|---:|
| DocVQA | 128 | 128 | 128 | 128 |
| TextVQA | 128 | 128 | 128 | 128 |
| AI2D | 128 | 128 | 128 | 128 |

This is an exploratory comparison on previously observed DEV, not an independent general equivalence, quality or timing conclusion. The result covers the recorded prompts, backend and generation configuration. Full historical visual/DeepStack and KV tensors and first logits were not archived; online runtime identity flags and KV shapes cannot establish their elementwise equality. No reference answers were opened or scoring performed.

After the GPU run, the initial snapshot failed because its guard shadowed a dictionary with an imported module. The failed supervisor/log and original run source are retained. A separate independently reviewed v2 repair reconstructed the real stopped container and snapshotted the original frozen source bytes before complete formal and independent verification. Snapshotting after the run and endpoint hashes cannot exclude a transient change during execution.

Evidence: `verification002.json`, `snapshot002/manifest.json`, and `../../e1_independent_review/pixel_wavefront_full_dev_result_review.json`. All 768 new record/raw files and per-question comparisons remain in the linked machine-readable evidence.
