# Research evidence archive — 2026-11-15

This preserves existing experiment evidence. It does not rerun experiments, change their conclusions, establish a new method, or make oracle scores deployable results.

Code and project reports remain at their normal repository paths. The archives preserve original JSON/JSONL, reports, logs, CSV/TSV, SQLite, historical source snapshots, selected result figures, and opaque result XLSX/PKL bytes. Pickles are never loaded or executed.

Weights, feature caches, source images, raw dataset shards, bytecode and transient SQLite WAL/SHM/locks are excluded. index.json records every included original relative path, size and SHA256, all excluded paths/types/sizes, original database schemas/counts/read-only checks, recorded statuses, and every compressed part hash. This is evidence preservation, not a self-contained model/data backup.

discovery, e0 and m12 are already tracked historical output families; native_detail_sources contains external raw data; research_archive is never recursively archived. These roots are intentionally excluded.

## Latest browsable reports

- [detail_allocation](reports/Qwen_vl/outputs/detail_allocation/run_20261114/evaluation/report.md)
- [student_broad](reports/Qwen_vl/outputs/student_broad/run_20261113_v2/evaluation/report.md)
- [student_transfer](reports/Qwen_vl/outputs/student_transfer/run_20261112_v2/evaluation/report.md)
- [paired_pruning](reports/Qwen_vl/outputs/paired_pruning/run_20261111/evaluation/report.md)
- [student_full_aux](reports/Qwen_vl/outputs/student_full_aux/postprocess_20261109T120758.272152Z_k4f0tlgm/report.md)

Other preserved Markdown:

- [binding_oracle/analysis.md](reports/Qwen_vl/outputs/binding_oracle/analysis.md)
- [detail_allocation/run_20261114/evaluation/report.md](reports/Qwen_vl/outputs/detail_allocation/run_20261114/evaluation/report.md)
- [local_reward_diagnostic/analysis.md](reports/Qwen_vl/outputs/local_reward_diagnostic/analysis.md)
- [paired_pruning/run_20261111/evaluation/report.md](reports/Qwen_vl/outputs/paired_pruning/run_20261111/evaluation/report.md)
- [student_broad/run_20261113/source_snapshot/student_broad_prereg.md](reports/Qwen_vl/outputs/student_broad/run_20261113/source_snapshot/student_broad_prereg.md)
- [student_broad/run_20261113_v2/evaluation/report.md](reports/Qwen_vl/outputs/student_broad/run_20261113_v2/evaluation/report.md)
- [student_full_aux/postprocess_20261109T120758.272152Z_k4f0tlgm/report.md](reports/Qwen_vl/outputs/student_full_aux/postprocess_20261109T120758.272152Z_k4f0tlgm/report.md)
- [student_transfer/run_20261112/failed_source_snapshot/student_transfer_prereg.md](reports/Qwen_vl/outputs/student_transfer/run_20261112/failed_source_snapshot/student_transfer_prereg.md)
- [student_transfer/run_20261112_v2/evaluation/report.md](reports/Qwen_vl/outputs/student_transfer/run_20261112_v2/evaluation/report.md)

## Verification and recovery

Run from the repository root using Python 3 (standard library only):

```bash
python3 -B Qwen_vl/scripts/archive_research.py verify
```

verification.json records the completed full streaming verification. Parts are 32 MiB maximum byte ranges of ONE gzip stream per family; an individual part is not a standalone gzip archive.

To recover a family into a new empty directory, concatenate its parts in the exact index.json order, then unpack. For example:

```bash
mkdir -p /tmp/eadp-evidence-restore
cat Qwen_vl/outputs/research_archive/20261115/parts/detail_allocation.tar.gz.part* | tar -xz -C /tmp/eadp-evidence-restore
```

Archive members use original repository-relative paths. Recover into a separate directory; do not overwrite live experiment inputs/results. Verify first. Original database bytes are preserved exactly; all five database WALs were empty, quick_check passed, and immutable main-file counts matched ordinary read-only counts. Recovered JSON paths still refer to original external datasets/models and need those dependencies for fresh computation.

## Completed, failed, and superseded evidence

Read the original reports and docs/project_handoff.md for scientific interpretation. The index status_sources retain the recorded state locations; historical_boundaries identify failed attempts, superseded geometry, probes and preflight versions. Presence in this archive does not make a failed gate, old geometry or preflight panel an accepted experiment.

In particular: student_transfer/run_20261112 and student_broad/run_20261113 precede their completed v2 runs; their failed gates/snapshots remain historical evidence. binding_oracle/archive_geometry_v1 predates corrected geometry. local_reward_diagnostic/serial_throughput_probe was discarded. detail_allocation/manifest_panel96_preflight.json is the earlier panel; the final 192-per-dataset run concluded STOP_THIS_GEOMETRY. CF carrier failures, the old student's full-DocVQA failure, and negative paired-pruning targets remain negative results. No teacher/oracle result is relabeled as learned or deployable performance.

| Family | Files | Original MiB | gzip MiB | Parts |
|---|---:|---:|---:|---:|
| _root_files | 4 | 0.009 | 0.001 | 1 |
| attention_calibration | 19 | 8.012 | 0.389 | 1 |
| binding_oracle | 1561 | 80.271 | 11.598 | 1 |
| catchup_rerank | 21 | 4.006 | 0.340 | 1 |
| cf_mining | 482 | 8.324 | 1.637 | 1 |
| detail_allocation | 10 | 117.387 | 9.316 | 1 |
| eadp | 16 | 3.596 | 3.317 | 1 |
| local_reward_diagnostic | 109 | 17.383 | 2.456 | 1 |
| logs | 28 | 12.019 | 1.574 | 1 |
| m13 | 285 | 10.479 | 3.227 | 1 |
| m14 | 1 | 0.032 | 0.005 | 1 |
| mosaic | 107 | 23.649 | 21.777 | 1 |
| multistream_oracle | 19 | 46.193 | 11.224 | 1 |
| paired_pruning | 24 | 25.209 | 3.386 | 1 |
| student_broad | 32 | 1153.801 | 131.816 | 5 |
| student_distill | 32 | 47.857 | 12.495 | 1 |
| student_full | 15 | 24.050 | 4.973 | 1 |
| student_full_aux | 9 | 4.868 | 0.223 | 1 |
| student_transfer | 16 | 395.639 | 46.672 | 2 |
