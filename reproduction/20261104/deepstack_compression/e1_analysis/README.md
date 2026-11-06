# E1 DEV completeness and scoring preparation

This directory provides CPU-only code for a future **completed** E1 full DEV run. No experimental predictions or reference answers have been read or scored by its preparation/tests. This code must not be run on an ongoing journal or used on CONFIRM/FINAL.

## Gate before reference answers

`completion.py` first requires an actual successful external supervisor exit, the corresponding launcher PID/start identity, verified stopped Docker container, clean GPU state, an uninterrupted full run, frozen 384-question panel/input proofs and current source/tokenizer hashes. It checks all 384 × 4 natural records, quartet validation, EOS/truncation accounting and independently decodes every generated token sequence with the locally frozen tokenizer. It opens no reference answers. It emits an acceptance JSON only if every requirement succeeds.

The future full launcher must save its exact command with one `--stage full`, `--out <run-dir>` and Docker `--name`, controller identity, plus an exit record with real container ID, `returncode=0`, `timed_out=false`, `interrupted=null`, `client_exited_while_container_running=false`, `container_stopped_verified=true` and an empty `gpu_pids_after`. An early-exiting Docker client followed by a forced container stop must be recorded as failure even when the client returned zero. The supervisor terminal must report the same launcher PID, `reason=completed`, `worker_returncode=0` and no remaining process group. The exact interface must be confirmed with the full-run implementation before launch; no such accepted full run exists yet.

Illustrative invocation only, with actual completed artifact paths substituted:

```bash
python monitor/supervise.py --run-dir ACCEPTANCE_SUPERVISOR --timeout 1800 --snapshot-seconds 3600 -- \
  /media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python \
  e1_analysis/completion.py --run-dir FULL_RUN --panel data_manifest/panel_manifest.jsonl \
  --launch FULL_LAUNCH.json --terminal FULL_SUPERVISOR/terminal.json --exit FULL_EXIT.json \
  --inputs e1/input_preparation_002/inputs.jsonl \
  --input-proof e1/input_preparation_002/independent_verification.json --out ACCEPTANCE.json
```

`score.py` checks that the completion verifier itself finished cleanly under its supervisor, then rechecks acceptance, artifact hashes, process identity/exit and the complete natural-arm inventory **before** reading any source dataset answers. It then rechecks the source TSV SHA, selects only the 384 frozen DEV questions, writes to a new directory and rechecks source/code identities at the end. An incomplete run, changed artifact, missing task/arm, decoder mismatch, unresolved exit or score disagreement fails closed. Empty output strings and 2048-token truncations remain in the denominator and are separately counted.

```bash
CUDA_VISIBLE_DEVICES='' /media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python \
  e1_analysis/score.py --acceptance ACCEPTANCE.json --acceptance-started ACCEPTANCE_SUPERVISOR/started.json \
  --acceptance-terminal ACCEPTANCE_SUPERVISOR/terminal.json --dataset-summary data_manifest/manifest_summary.json --out NEW_SCORE_DIRECTORY
```

## Metric provenance

- DocVQA: unchanged fixed author `process_line(method='anls')` and `hit_calculate`; independent two-row edit distance and normalized similarity reproduce the same raw-length denominator and 0.5 threshold. Do not confuse author `eval_score` (mean edit distance) with the final ANLS hit.
- TextVQA: fixed author `process_line(method='vqa_score')` and `hit_calculate`; independent normalization implementation and closed-form leave-one-out consensus compare each item. The independent normalizer reuses only the fixed source's article, digit and contraction tables, which are recorded as shared constants; it is not independent provenance for those tables.
- AI2D: fixed author `can_infer` and `exact_matching` behavior; unresolved answers map to `Z` and score 0 unless the label itself is Z. Independent code separately reconstructs both option and text fallback logic. No external judge/API is used, so this does not claim paper judge reproduction.

Primary scoring reproduces the author's `xlsxwriter` → pandas `read_excel` prediction-file round trip independently for each task and arm, including pandas column typing and NA conversion; empty strings may become `nan`. The per-row index is retained through the round trip to prevent dropped blank rows. Raw generated strings remain intact, and a separate literal-answer score is emitted per item. This is a scoring-compatibility choice, not permission to drop empty answers. `scores.jsonl` includes all four arms, raw answers, official/independent scores, literal diagnostic, EOS and truncation. Any official/independent mismatch aborts rather than silently choosing one.

`analysis.json` reports each task's four means, DS effect at P1/P3, position effect at DS off/on, the difference-in-differences interaction, improvement/decline/unchanged and answer-change counts. It uses 10,000 deterministic percentile bootstrap resamples of image groups and retains every question within each resampled group. These are exploratory DEV 95% intervals, not multiplicity-adjusted FINAL evidence. The original split and the conservative old-anchor-excluded sensitivity lists remain distinct; only DEV is scored here.

`python -m unittest -v test_analysis` runs synthetic acceptance, adversarial incomplete-run, answer-normalization, exact-matching and factorial/bootstrap tests with no GPU, predictions or actual reference answers. Additional 10,000 synthetic normalizer and 10,000 option-parser comparisons matched the author functions during preparation. Exact runtime compatibility and all cited gates must still be verified against the eventual full-run artifacts.
