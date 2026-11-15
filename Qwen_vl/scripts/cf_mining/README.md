# Current-compressor semantic difference pilot

Run from the repository root with the cached native Qwen3-VL environment:

```bash
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
$PY Qwen_vl/scripts/cf_mining/cf_data.py
$PY Qwen_vl/scripts/cf_mining/cf_mine.py --limit 120 --source-limit 24
$PY Qwen_vl/scripts/cf_mining/cf_compare.py --stage gate --carrier local_readout --relation-weight 1 --steps 400 --round-steps 50 --prepare-teacher-kl
# Continue after gate_result.json reports PASS:
$PY Qwen_vl/scripts/cf_mining/cf_finish.py --stage compare --carrier local_readout --relation-weight 1 --steps 400 --round-steps 50 --prepare-teacher-kl
$PY Qwen_vl/scripts/cf_mining/cf_analyze.py --write-report
```

`cf_data.py` needs CPU Tesseract, its English data, and the paths given by
`--tesseract`, `--tessdata`, `--library-path`. On this host these were installed
without changing system packages: `apt-get download tesseract-ocr
libtesseract4 liblept5 tesseract-ocr-eng`, then extract the downloaded `.deb`
files with `dpkg-deb -x` beneath `/tmp/eadp_cf_tools/root`. The manifest records
the exact tool paths, seed, boxes, labels and source images. Image/feature
caches and checkpoints remain local and are ignored by Git.

The miner can resume per pair. The training runner intentionally refuses to
append to an existing optimizer trajectory; use a fresh `--out` for a new
training run, or preserve the old run before retrying a failed configuration.
`--dry-run` inspects data and partitions without loading a model. A gate
failure concerns the tested carrier/optimization, not the general hypothesis.

The first 262,145-parameter pooling carrier failed the 120-update overfit
gate (0/8 both-correct before and after); its results remain in
`outputs/cf_mining/training_global_gate`. The `readout` carrier adds
four low-rank visual reading interfaces, for 1,310,721 trainable parameters.
All comparison arms use the same rank, exact B2 initialization, frozen
decoder and update budget. Global readout also failed after 400 updates
(0/8 both-correct, all eight collapsed); its results remain in
`outputs/cf_mining/training_global_readout`. The final `local_readout` screen
adds a Gaussian spatial attention bias and opposing-answer margin loss.
Contrast arms use additional negative-answer forwards; `ordinary` and
`replay_ce` use plain CE+KL. This is not an equal-total-computation trial.
The complete teacher's verified
response provides the training answer; numeric ground truth remains the
evaluation target. Distillation uses top-64 soft probabilities plus an
aggregate remaining-mass bucket, not full-vocabulary KL. Fixed and online
start with the same hard set. Online refresh prioritizes current same-answer
collapses; replay preserves its exact individual exposure multiset.

The current search repeatedly screens a finite pool of one-digit edits. It
does not yet synthesize new edits after each update, perform field swaps or
sum interventions, train unrelated-question invariance constraints, or
demonstrate unseen-layout transfer. Sibling-question invariance is evaluated
only as a supplementary check. Partitions come
from DocVQA_VAL and support only a feasibility pilot. Consult
`docs/cf_mining_pilot_prereg.md` and `docs/cf_mining_prior_art.md` for the
decision rules and claim boundaries.
