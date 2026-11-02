This directory independently checks the unchanged author Qwen recipe and its
preserved outputs. It never loads a model, uses a GPU/API, changes source or raw
predictions, or retries generation.

`preparation_audit.json` verifies all5349 DocVQA and3088 AI2D canonical TSV records,
their entire-byte hashes, all2490 unique current cached images against the
author-regeneration audit, the four predeclared smoke prompts/indices/images,
and the frozen public launcher. No four-question accuracy is computed.

The first public launch failed before generation because the author repository
omits `vlmeval.utils.result_transfer`. Run02 preserves that failure and supplies
only this missing import through `import_overlay`. Its3496-byte helper was
independently downloaded from official VLMEvalKit commit
`401916c00a4769884aeca865a77674abe0c62a26`, SHA256
`d6225256ec0afa2ac9857b139df2f910b23b693749d2f2eb58cf7e696b9b6974`.
The hook returns no override for other imports. The helper's only public-entry
calls are guarded by MMMU_TEST/MMT-Bench_ALL, so neither selected task calls it.

After generation, run this CPU-only command from the V2 root:

```sh
CUDA_VISIBLE_DEVICES='' /home/dell/miniconda3/envs/llava_pruner/bin/python \
  reproduction/20261028/qwen/independent_audit/audit_qwen.py verify
```

The default plan/run are `public_fa2_full_run02_plan.json` and
`public_fa2_full_run02`; other isolated attempts require explicit `--plan` and
`--run-dir` and a prior `prepare` audit. The verifier rejects missing, duplicated,
reordered or modified IDs/questions/GT/options,
changed sources and changed image/input identity. It reads raw XLSX/TSV/JSON,
executes unchanged author metric helpers, independently recomputes each score,
and compares author `_acc` artifacts when available. All audit outputs stay here.

Empty answers can be legitimate EOS-only generation. They remain in the full
denominator, are counted separately, and do not invalidate complete coverage.
`coverage_complete`, `generation_valid`, `empty_count`, and
`planned_no_empty_gate_passed` are distinct fields. An empty answer means the
original planned "No empty" gate did not pass; this fact is never rewritten.
Actual API/runtime failure placeholders make `generation_valid=false`, while
all rows and a clearly diagnostic score remain. Missing IDs still reject any
full benchmark score. The original generation plan/source/predictions are not
changed or retried.

DocVQA's primary ANLS reproduces the native pandas default-NA load followed by
`str()`: literal predictions such as `None`/`NA` score as `nan`. Raw text remains
preserved; the literal-text ANLS is separately labelled auxiliary. This is a
fixed loader protocol, not selection of a higher score. The native pandas2.3.3
NA set was checked in an isolated CPU-only container.
The separately labelled `auxiliary_empty_answers_zero_score_percent` forces
only genuinely blank raw answers to zero. Native pandas' `nan` can otherwise
receive nonzero ANLS on some GT strings; the primary native score is unchanged.

This policy was fixed before inspecting any new complete score, using synthetic
cases and preserved old complete outputs only. `empty_answer_policy.json`
records the reason and frozen generation-plan SHA. The superseded auditor and
its pending/no-score report are preserved under `protocol_history/v1_reject_empty/`;
the archived auditor SHA exactly matches that report's original source SHA.

AI2D's public entry requests an API judge by default. Its failure or exact-match
fallback remains in the original logs. This verifier computes the released
deterministic `can_infer` policy without API access and independently duplicates
that extraction; it does not claim unavailable judge matching was reproduced.

`complete` means both full prediction sets passed their gates. The separate
`author_score_artifacts_complete` and `public_launcher_exited_successfully`
fields must also be reported. A CPU score recovery does not erase a launcher or
API failure, and none of these fields alone proves the paper argv.

Eleven CPU tests pass, including empty/failure/full-denominator distinctions and
full preserved old5349/3088 parser/input/GT and
pointwise metric regressions. They recover old official scores
61.135577708229846 and74.54663212435233; these regression results are not new FA2
experimental results. `cpu_tests.log` contains the validation output.

`supervise_terminal_audit.py` is a separately prepared CPU watcher. Its default
`check` is read-only; `run --root-review-note ...` requires the root agent to
start it after review. It waits for the original `run_packaged` controller,
Docker client, fixed container/init and observed DocVQA/AI2D workers to exit.
DocVQA worker exit alone cannot trigger the audit. Only then does it call the
existing verifier once, retaining native exit codes, full original log hashes,
all error lines and any CPU-verifier exception. Model payload identity reuses
the prior complete-file SHA proof with current stat guards; no second large
weight-file hash pass is added.

Stable output prefix is `public_fa2_full_run02.terminal_audit`. Its `report.json`
separates `full_prediction_audit_passed`, `coverage_complete`, `generation_valid`,
`empty_count`, `planned_no_empty_gate_passed`, native launcher exit and author
score-artifact coverage. `state.json` becoming `complete` means only that the
CPU audit terminated; it never claims the paper score or original No-empty
gate was reproduced. A failure preserves `failure.json` and stops without
generation or automatic retry. Eight small CPU control-plane tests cover the
terminal/identity/no-prefix boundaries; the existing eleven metric tests are
unchanged.
