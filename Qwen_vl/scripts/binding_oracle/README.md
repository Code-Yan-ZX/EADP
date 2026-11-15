# Content–position binding oracle pilot

The predeclared protocol is `docs/binding_oracle_prereg.md`. This is an
inference-only diagnostic, not a trained or deployable new compression method.
Native Qwen3-VL-8B features, all three DeepStack streams, and native interleaved
mRoPE are retained. The PPE position operator is an explicit native adaptation,
not the published Qwen2.5 checkpoint or its SFT benchmark reproduction.

From repository root:

```bash
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
$PY Qwen_vl/scripts/binding_oracle/smoke.py
$PY Qwen_vl/scripts/binding_oracle/gpu_gates.py
$PY Qwen_vl/scripts/binding_oracle/run.py --manifest Qwen_vl/outputs/binding_oracle/data/combined_manifest.json --limit 62 --stage all
$PY Qwen_vl/scripts/binding_oracle/analyze.py --manifest Qwen_vl/outputs/binding_oracle/data/combined_manifest.json --include-anls
$PY Qwen_vl/scripts/binding_oracle/mechanism_audit.py
$PY Qwen_vl/scripts/binding_oracle/plot.py
```

`run.py` resumes per question/arm and refuses a changed annotation for an
already predicted pair. Run only one GPU process at a time. `integration_gates`
tests actual K256 selector/PPE paths before the comparison. Four-way teacher
acceptance is based solely on full-model correctness; student failures never
select the evaluation panel. Untouched real source pages are evaluated separately.

Per-image `.pt` features and PNG images stay local; prediction JSONs, diagnostic
cluster membership, manifests, gates and compact visual audits are retained.
Every counterpart image is separately encoded. Value-box and OCR-line regions
are localization oracles, not gold semantic field boundaries. Shared-token
collisions are resolved by larger pixel intersection area and counted explicitly.

`data_build.py` creates raw/curated VAL candidates and `data_train.py` extends
them with supplied-OCR TRAIN pages; their `--help` records the local source and
Tesseract defaults. Data QA/provenance is in `outputs/binding_oracle/data/`.
The combined manifest's first 22 VAL entries are immutable, followed by 40
TRAIN entries from disjoint parent documents. Builder source-to-square masks
are idealized; `run.py` recomputes actual native masks from raw pixel boxes
through qwen-vl-utils smart-resize, then padding and the final resize.

`archive_geometry_v1` is invalidated evidence and must not be analyzed as a
model comparison. A runner does not automatically invalidate answers when
code changes: use a fresh output location or explicitly archive and clear
affected answers, as recorded in the protocol, before recomputing them.

Prior `cf_mining/` artifacts are independent and remain unchanged. No new LoRA,
adapter, optimization trajectory, or teacher gradient is introduced here.
