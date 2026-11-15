# Fixed-selection attention calibration pilot

Scientific protocol: `docs/attention_calibration_prereg.md`. This directory's
data and analysis utilities run on CPU and do not import the VLM. Native-model
gates, exact attention formulas and the GPU runner follow that protocol.

## Frozen data

`Qwen_vl/outputs/attention_calibration/data/manifest.json` contains 32 original
DocVQA TRAIN pages, one original numeric QA per independent source parent.
`development_manifest.json` contains one additional parent used only for
interface gates and timing. All original answer aliases and source image bytes
are preserved. No OCR localization, teacher correctness, compression failure or
model output enters selection. The source split is TRAIN; this is an
exploratory project-level pilot, not an official benchmark or unseen-layout
claim.

The builder examined 39,463 TRAIN annotations. After excluding 5,005 source
parents, 440 eligible numeric QA rows remained across 243 independent parents.
Selection uses seed 2026110907, SHA256 ordering of parents and pages, then the
first numeric question by question ID on the selected page. The first selected
parent is the development page; the next 32 are primary. Eligibility requires
an original pure numeric/currency/percentage alias with at least two digits;
all eligible numeric aliases must represent one value. All original aliases,
including any other valid string forms, remain in the manifest.

The exclusion union covers all VAL parents, every parent in the CF, binding,
local-reward and CatchUp manifests (including reserves), and the conservative
historical fit roster/earlier exclusion union. The recorded overlap with every
pool is zero. The pool sizes are VAL 938, binding 62, CF 133, local-reward 48,
CatchUp 16 and historical fit roster 4,494; these pools overlap each other.

An additional CPU audit read actual prior original-image files, rather than
only trusting recorded hashes: binding 62, CF 141, local-reward 48 and CatchUp
16 distinct source images. Each pool's SHA256 intersection with the new 32
primary images was empty. Parent exclusion, rather than image hashing alone,
also prevents another page of a previously used source document from entering.

Frozen SHA256 values:

| Artifact | SHA256 |
| --- | --- |
| Primary manifest | `2aa3e4366d838b5b910704f9a4c0e92d4fa3fc89a3481088dc6197cd3db3b5dc` |
| Development manifest | `2f4263bbf2e797c0228df1de964e7914325d3f50e5c46dff8e8bbc80e5f2cf9c` |
| `data.py` | `72a008072f9f8d3c3f6995ed838b3bcedefdeec192072290ff091b437bb1d145` |
| `analysis.py` | `ff97475753e6dcc82a4ddbf66e804adc4db668fb9fdd33e227da0b329221cccf` |
| Historical CF exact-scoring source | `9c7b10dcbd39714e340938f1c40cd958d7a05b3ad095df35ac4d3c4043014e97` |

From the project root, this reconstructs/verifies the same immutable manifests
from the local data. It scans local parquet metadata, reads needed image
columns and hashes source shards; no download or GPU is used. Existing files
must match exactly or the builder fails without replacing samples.

```bash
CAL_PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
$CAL_PY Qwen_vl/scripts/attention_calibration/data.py
```

Quick audit without re-reading all parquet images:

```bash
$CAL_PY - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('Qwen_vl/outputs/attention_calibration/data')
audit = json.loads((root / 'data_audit.json').read_text())
primary = json.loads((root / 'manifest.json').read_text())
development = json.loads((root / 'development_manifest.json').read_text())
assert hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest() == audit['primary_manifest_sha256']
assert hashlib.sha256((root / 'development_manifest.json').read_bytes()).hexdigest() == audit['development_manifest_sha256']
assert len(primary['items']) == len({r['parent_id'] for r in primary['items']}) == 32
assert len(development['items']) == 1
parents = {r['parent_id'] for r in primary['items']}
assert not parents & {r['parent_id'] for r in development['items']}
assert not parents & set(primary['provenance']['excluded_parent_documents'])
for item in primary['items'] + development['items']:
    assert item['answers'] == item['answers_original']
    assert hashlib.sha256(Path(item['image_path']).read_bytes()).hexdigest() == item['image_sha256']
assert not any(audit['data_validation']['overlap_by_exclusion_pool'].values())
print(json.dumps(audit['data_validation'], indent=2))
PY
```

Independent actual-image audit, including CF sources without a recorded image
hash in their old manifest:

```bash
$CAL_PY - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('Qwen_vl/outputs')
new = json.loads((root / 'attention_calibration/data/manifest.json').read_text())['items']
new_hashes = {r['image_sha256'] for r in new}
for name in ['binding_oracle/data/combined_manifest.json', 'cf_mining/manifest.json',
             'local_reward_diagnostic/data/manifest.json', 'catchup_rerank/data/manifest.json']:
    manifest = json.loads((root / name).read_text())
    rows = manifest.get('records', manifest.get('items'))
    paths = {r.get('source_image', r.get('image_path')) for r in rows}
    assert None not in paths
    hashes = {hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    overlap = new_hashes & hashes
    assert not overlap
    print(name, 'source_images=', len(hashes), 'overlap=', len(overlap))
PY
```

## Analysis contract and outcomes

The runner writes `predictions.json` with `complete=true`, a config containing
`manifest_sha256`, `planned_arms`, `baseline_arm` (default `baseline`) and the
explicit `contrasts` below. Each record contains `item_id`, `parent_id`,
`question`, original `answers`, and `predictions[arm]`. A prediction contains
`text` (including a genuinely completed empty response), boolean `truncated`,
`token_ids`, and boolean `eos`. The analyzer preserves provided EOS metadata;
the runner determines EOS from actual stopping tokens. If `correct` is also
provided, it must agree with the exact scorer.

All 32 frozen primary records and every declared arm must be present. Missing
runs and operational errors remain pending: they prevent primary analysis,
instead of being invented incorrect answers or excluded from the denominator.
Completed wrong answers, empty answers and cap hits stay in the primary
analysis. All-untruncated analysis is a separately labeled sensitivity with
explicit included/excluded item IDs and its changed denominator.

Numeric exact uses the original `normalize` and `correct` function ASTs from
`scripts/cf_mining/cf_mine.py`, without importing its model code. The complete
source hash and extracted function names are reported. This preserves numeric
sign and percent suffix; leading zeros and format differences may normalize.
Literal alias exact is reported separately. ANLS uses the checked-in official
VLMEvalKit `anls_compute` distance followed by `hit_calculate`'s DocVQA
similarity/threshold, on the original strings.

Paired differences and rescue/break/tie counts use the same questions. The
95% percentile bootstrap uses **10,000 draws, seed 20261109**, resampling original
source parents with shared draws across all arms and contrasts. There is one
QA per parent in this panel, so document macro and question micro accuracy
coincide. The implementation also checks document grouping on a synthetic
multiple-QA-per-parent example. A degenerate CI does not establish equivalence.

With all 14 arms enabled, the preregistered groups require these **28 distinct
before/after pairs**. Continuous RESTORE and the B2 comparisons remain
auxiliary/secondary, and maskzero-versus-native comparisons are backend
diagnostics. The analysis code accepts explicit contrasts; it does not infer
the scientific set from arm names. A gate-based RESTORE fallback must be
frozen before main QA and remove the unavailable arm and its comparisons
explicitly, with the failed gate retained.

```json
[
  ["vz_maskzero", "count"],
  ["vz_maskzero", "effective"],
  ["vz_maskzero", "bias_const"],
  ["vz_maskzero", "restore_prefill"],
  ["vz_maskzero", "restore_continuous"],
  ["count", "effective"],
  ["bias_const", "effective"],
  ["count", "restore_prefill"],
  ["bias_const", "restore_prefill"],
  ["restore_prefill", "restore_continuous"],
  ["b2_maskzero", "b2_bias_const"],
  ["b2_maskzero", "b2_restore_prefill"],
  ["b2_maskzero", "b2_restore_continuous"],
  ["b2_restore_prefill", "b2_restore_continuous"],
  ["baseline", "vz_maskzero"],
  ["b2_baseline", "b2_maskzero"],
  ["rres512", "baseline"],
  ["rres512", "vz_maskzero"],
  ["rres512", "count"],
  ["rres512", "effective"],
  ["rres512", "bias_const"],
  ["rres512", "restore_prefill"],
  ["rres512", "restore_continuous"],
  ["rres512", "b2_baseline"],
  ["rres512", "b2_maskzero"],
  ["rres512", "b2_bias_const"],
  ["rres512", "b2_restore_prefill"],
  ["rres512", "b2_restore_continuous"]
]
```

```bash
$CAL_PY Qwen_vl/scripts/attention_calibration/analysis.py --self-check
$CAL_PY Qwen_vl/scripts/attention_calibration/analysis.py
```

`analysis_selfcheck_v2.json` records a CPU synthetic integration check using
all 32 manifest records, all 14 arms and all 28 contrasts. It verifies numeric
and official ANLS boundaries, paired rescue/break accounting, source-parent
bootstrap, truncation handling and EOS propagation; it rejects missing or
duplicate questions, altered aliases, a wrong manifest hash, undeclared arms
and operational failures. These are synthetic outputs, not QA results.
`analysis_selfcheck.json` is the preserved earlier check with data seed
2026110907; the bootstrap was aligned to preregistered seed 20261109 and exact
CF function extraction **before any main-panel QA**, and the v2 artifact
records the final code hash.

Report all declared outcomes and exploratory comparisons; selecting the best
arm on these 32 pages does not provide an unbiased confirmation. Accuracy
analysis does not infer speed: use separately measured paired timing artifacts.
