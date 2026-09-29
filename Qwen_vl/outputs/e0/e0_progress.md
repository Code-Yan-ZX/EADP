# E0 progress note

Branch: `e0-native-baselines`. Spec: `docs/e0_native_baselines_prereg.md` (frozen).

## Environment (recorded 2026-09-29, M0)

| item | value |
|---|---|
| GPU | NVIDIA A40, 46 GB (1 card, index 0) |
| driver | 560.35.05 |
| conda env | `qwen3vl_clean` (python 3.10.20) |
| torch | 2.10.0+cu128 |
| CUDA (torch build) | 12.8 |
| transformers | 4.57.6 |
| VLMEvalKit | 0.2rc1 (editable, `Qwen_vl/VLMEvalKit`) |
| attention | SDPA (`QWEN3_VLM_ATTN_IMPL=sdpa`; no flash-attn wheel for this combo) |
| env conventions | `HF_HUB_OFFLINE=1`, `LMUData=/media/disk2/YZX/LMUData`, `HF_DATASETS_CACHE=/media/disk2/YZX/hf-datasets-cache` (from `Qwen_vl/scripts/discovery/common.py`) |

transformers 4.57.6 `modeling_qwen3_vl.py` was read before implementation:
`get_rope_index`, `_deepstack_process` (inject after layers 0–2 via
`visual_pos_masks`), the decode branch (`position_ids=None` → `arange +
cache_position + self.rope_deltas`) and `prepare_inputs_for_generation`
(forcing `position_ids=None`) all behave exactly as prereg §3.2 assumes.
No amendment needed on this account.

## M0 — data partition (2026-09-29)

Script `Qwen_vl/scripts/e0/e0_plan.py` → `Qwen_vl/outputs/e0/e0_plan.json`.
Seed `E0_SEED=20260929`; exclusions at image level; sources exactly the frozen
list (all `*_plan.json`, SAGE fit/val labels, `sample_indices(n,150)` bank,
S2-A causal cases).

| dataset | total rows | excluded imgs (rows touched) | pool rows / imgs | DEV | CONFIRM |
|---|---:|---:|---:|---:|---:|
| TextVQA_VAL | 5000 | 608 | 3958 / 2558 | 300 | 1949 |
| DocVQA_VAL | 5349 | 490 | 2767 / 796 | 300 | 1362 |
| OCRBench | 1000 | 606 | 328 / 324 | **164** | 164 |
| ChartQA_TEST | 2500 | 0 | 2500 / 1509 | 300 | 1245 |
| MMBench_DEV_EN_V11 | 4876 | 0 | 4876 / 1310 | 300 | 2472 |
| MMStar | 1500 | 0 | 1500 / 1430 | 300 | 749 |
| RealWorldQA | 765 | 0 | 765 / 762 | 300 | 381 |
| POPE | 5127 | 0 | 5127 / 5127 | 300 | 2563 |

- OCRBench is short exactly as prereg §2.5 predicted: 164 DEV / 164 CONFIRM
  (606 of 1000 images touched by history). Reported as-is; nothing borrowed
  from CONFIRM.
- DEV caps at 300; surplus images were dropped, not moved to CONFIRM
  (per-dataset `dev_dropped_for_cap` in the plan JSON).
- Text-only MMBench rows (no image) are each their own group.
- Split hashes: global DEV `35214185be3a73814fbb335d0dc61c8227ba9ae99dcac86e83be9096794783c7`,
  global CONFIRM `b81f9135e8cec3cd571c1c2818b3a74251164a07e69673b1f75a2bee3cb8a7ec`.
  Per-dataset hashes inside `e0_plan.json`. CONFIRM answers are NOT read; only
  indices + sha256 are archived.
