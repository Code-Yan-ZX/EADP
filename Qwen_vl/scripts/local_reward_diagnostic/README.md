# LayoutLite local-reward diagnostic

This is an **inference-only reward-ranking diagnostic**. The external endpoint is literal coverage of original human DocVQA answer variants in an OCR transcript: an **evidence coverage proxy**, not actual QA accuracy, correct field binding, full-page OCR gold, or an improvement from training.

The frozen protocol is [local_reward_diagnostic_prereg.md](../../../docs/local_reward_diagnostic_prereg.md). The experiment uses the released LayoutLite head on FireRed-OCR/Qwen3-VL-2B, with five exact-cardinality masks per page: Top-K and four predeclared Gumbel-Top-K masks. Every mask keeps `round(0.5*N)` native visual tokens. This probes equal-cost actions; it does not reproduce LayoutLite's globally calibrated clustered allocation, its training policy, or this project's 8B/25%-retention target.

## Environment and pinned inputs

Run from `/media/disk2/YZX/research/EADP` with:

```bash
LR_PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
```

- Downloaded backbone: `/media/disk2/YZX/research/model/FireRed-OCR`, revision `7bf5e9d91d3bb6bf91a3b6bc66d163096b696e50`. The released weights are already stored there; no model download is part of these reproduction commands.
- Official source: `Qwen_vl/third_party/LayoutLite`, commit `7e398a2ff4f23f0cb836e36e0e81d21d379d8fc2`.
- Official head: `checkpoints/firered_600.pt`, SHA256 `92e25d31ffae823b04b2b97be2259446b656afb81eb67f959fad16ac7d07e73d`.
- Execution: stock Transformers 4.57.6; the head definition and OCR prompt are extracted from the pinned upstream source. Native vision/text decoding and all three DeepStack streams are retained. Exact package versions and source hashes belong in `run_manifest.json`.
- Main artifact directory: `Qwen_vl/outputs/local_reward_diagnostic/`.

GPU execution belongs to one coordinated runner. Data preparation, analysis, and plotting use CPU only.

## Data

`data_probe.py` uses already-local pinned DocVQA TRAIN parquet. It makes a seeded annotation-only selection of 32 main pages and 16 ordered reserve pages. The registered run does not consume reserves or replace main pages after seeing outputs.

Each page has 2–4 independently annotated numeric QA answers, each uniquely localized through format-tolerant matching in provided OCR. Provided OCR supplies silver regions and a fixed 40–450-word length filter; original QA questions and all answer variants remain unchanged. The primary set has 80 selected QA fields; the reserve set has 47. Exclusions cover all 938 VAL parents, prior CF/binding parents, and conservatively all 4,494 parents in the historical fit roster. There are 48 independent selected parents, with no excluded-parent overlap.

For a fresh reproduction, build data before freezing the runner:

```bash
$LR_PY Qwen_vl/scripts/local_reward_diagnostic/data_probe.py
```

The canonical manifest is `outputs/local_reward_diagnostic/data/manifest.json` with `items` entries containing `doc_id`, `parent_id`, `image_path`, `role`, and `qa`. `records` is an equivalent alias. Source-image and source-parquet SHA256 values, repository revision, raw QA variants, silver regions, and exclusion provenance are recorded. See `data/data_inventory.json` and the manifest's `data_validation` audit.

Do not regenerate a frozen manifest during a resumed panel: the runner rejects changes to frozen data, code, protocol, or head hashes. Use `--out` with a separate directory for a data-only reproduction check.

## Native runner and outside-panel smoke

A standalone native all-token check and OCR generation:

```bash
$LR_PY Qwen_vl/scripts/local_reward_diagnostic/layoutlite_runner.py \
  --image Qwen_vl/outputs/local_reward_diagnostic/data/images/sqxg0023_3.png \
  --identity-gate \
  --max-new-tokens 2048
```

The preregistered optional QA smoke uses three fixed **old binding-panel** source pages, outside the new 32-page population. It records all six questions and would enable genuine secondary QA only if at least 4/6 were exact-correct and none truncated:

```bash
$LR_PY Qwen_vl/scripts/local_reward_diagnostic/run.py --smoke
```

The frozen run does **not** enable secondary real QA. Read `smoke.json` for all six outputs and the pre-panel decision. Its conclusions must remain about answer-evidence coverage in OCR text, not actual QA performance.

The all-token identity gate, exact K, all three DeepStack lengths, original mRoPE positions, and native decode continuation must pass. A failed interface gate is not a scientific negative result. The OCR prompt comes from the released code; generation is greedy with native EOS and a frozen 2048-new-token cap. QA smoke uses the registered suffix and 32-token cap.

## Batch execution

`batch_runner.py` implements same-page, same-K native prefill/generation. It preserves each mask's embeddings, DeepStack order, mRoPE positions, and native first-EOS stopping. Full-token and compressed rows have different K and are not one mixed-K batch. Shared batch elapsed time must not be summed across row records.

The real outside-panel batch gate passed before the registered panel. It records native full-token agreement, serial/batched compressed prefixes, decode positions, and any BF16 logit/token differences in `batch_gates.json`:

```bash
$LR_PY Qwen_vl/scripts/local_reward_diagnostic/gates.py
```

No batch size or numerical-equivalence decision is inferred from main-panel rewards. The frozen execution runs the full reference serially and the five same-K compressed masks together with `generate_many` at batch size 5.

The exact final panel launch is:

```bash
$LR_PY -u Qwen_vl/scripts/local_reward_diagnostic/run.py \
  > Qwen_vl/outputs/local_reward_diagnostic/run.log 2>&1
```

`run_manifest.json` freezes the actual prompt, 2048-token cap, mask seeds, model/head revisions, source hashes and data-manifest hash. Resume uses the same command and rejects any frozen-hash mismatch.

`serial_throughput_probe/` is an archived, invalid-for-panel preliminary run. Its outputs are excluded from the registered population and final analysis. Do not merge them with the formal `predictions.jsonl` or use them to select pages, rewards, masks, or settings.

## Main-panel outputs

The expected OCR population is **192 outputs**: 32 fixed pages × (one full reference + five masks). Questions and answer labels do not enter mask selection or candidate rewards. The five masks and their hashes are fixed before reward analysis. Missing outputs remain pending; compressed truncations are observed and reported, while truncated full references cannot support the primary complete-reference reward interpretation.

Expected artifacts:

- `run_manifest.json`: checkpoint/source/data hashes, prompt, cap, seeds, execution mode.
- `predictions.jsonl`: append-only page/arm OCR records; secondary QA appears only if its outside-panel smoke passed.
- `masks/*.json`: official scores, exact kept indices, seeds and cardinality audits.
- `identity_gate.json`, `tiny_interface_gate.json`, `tiny_batch_gate.json`, and `batch_gates.json`: interface evidence; the real-model batch gate is required before the main run.

A source/data hash mismatch on resume requires investigation; do not silently mix incompatible outputs.

## Final analysis and figure

After all 192 registered OCR outputs exist:

```bash
$LR_PY Qwen_vl/scripts/local_reward_diagnostic/analyze.py
$LR_PY Qwen_vl/scripts/local_reward_diagnostic/plot.py
```

Final analysis writes `analysis.json` and `analysis.md`; plotting writes `results.png` and `results.pdf`. `plot.py` refuses incomplete analysis. `analyze.py --partial`, if used for bookkeeping, creates explicitly partial artifacts and cannot establish a verdict.

The left figure panel shows document-macro reward concordance on Q-discordant mask pairs with raw global-reward difference ≤0.01, with 95% document-bootstrap intervals. The right panel compares external evidence coverage for the five fixed masks and masks selected by each registered reward from that same action pool. Fixed-mask intervals are reconstructed from the final analysis's valid per-document Q values and checked against its reported means; reward-selection intervals come directly from final analysis.

The registered information gate requires at least 12 independent documents and 24 Q-discordant near-global-tie pairs. `HOLD_INSUFFICIENT_INFORMATION` must appear prominently in the figure title if either condition fails. Null estimates remain “Not estimable”; they are never plotted as zeros. Width/alignment sensitivities remain separate from the primary width-64, 20%-tail analysis.

All figures explicitly say **“Evidence coverage proxy; no training.”** A positive diagnostic would require a separate controlled training study and real-QA evaluation before making a deployment or method claim.
