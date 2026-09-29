# E0 — Native Qwen3-VL repair, baseline reproduction and encoder-axis audit

Status: **in progress**. Spec: `docs/e0_native_baselines_prereg.md` (frozen);
amendments: `docs/e0_native_baselines_prereg_amendment.md` (A1–A5, all frozen
before the affected numbers existed). Branch `e0-native-baselines`.
Environment: single A40 (46 GB), driver 560.35.05, `qwen3vl_clean`
(python 3.10.20, torch 2.10.0+cu128, CUDA 12.8, transformers 4.57.6, SDPA —
no flash-attn wheel exists for this combo). Same host as M2.

## 1. Gates (M1) — all passed

`Qwen_vl/scripts/e0/e0_gates.py` → `Qwen_vl/outputs/e0/e0_gates.json`,
`all_passed=true` (2026-09-29):

| gate | result |
|---|---|
| N1 identity | 16 DEV samples, K=1024: prefill and all 32 forced-decode logits **bit-exact** (max\|Δ\| = 0.0), 0 token mismatches |
| N2 DeepStack live | min \|Δ\| = 2.88 ≫ 0.1 — the injection is real, not silently skipped |
| N3 3-D positions | kept visual tokens carry their full-sequence (t,h,w); kept text positions strictly monotone; decode step n sits at prefill_max+1+n |
| N4 invariants | 12 samples × 8 local arms: exact K, no duplicates/OOB, DS lengths K, mask/position/cache lengths consistent, `layer_calls = 36·(1+decode_steps)` exactly |
| N5 legacy parity | new engine in legacy mode (deepstack off, 1-D positions, B2) reproduces the archived SAGE-confirmation B2 predictions **32/32 exact** (bar: ≥ 95 %) |

Notes on the two issues the gates surfaced (both recorded in the amendment):

- **A2**: the stock forward computes `lm_head` over all prefill positions;
  the engine's last-position call differs by ~1 bf16 ulp (measured 6.25e-02).
  Gates compare with the stock-shaped call → bit-exact. Generation arms keep
  the last-position call (argmax-identical).
- The eager-attention capture used by in-LLM arms builds an explicit causal
  mask (the SDPA fast path passes `mask=None`, under which the eager kernel
  would not apply causal masking).

## 2. Data partition (M0)

Image-disjoint DEV/CONFIRM split, seed 20260929; OCRBench short exactly as
predicted (164/164); details and hashes in `Qwen_vl/outputs/e0/e0_progress.md`
and `e0_plan.json`. CONFIRM was never read.

## 3. Baseline ports (M2)

Upstream repos cloned into `_upstream/` (HEADs pinned in the port headers):
FastV `d1659729`, PyramidDrop `6444f304`, VisionZip `8f86b55c`, SparseVLMs
`a9e71427`, PACE `240b2206`. Ports in `Qwen_vl/model/baselines/`; the local
pruners (cdpruner/hipruner) received index-exposing wrappers. Every ported
arm passed the N4 invariants on 12 DEV samples (SparseVLM under the
amendment-A4 recycling bound: realized visual count K + n_recycled).

CIVIC: no public code exists (two recorded searches, 2026-09-29:
[arXiv:2605.28115](https://arxiv.org/abs/2605.28115) is the only artifact).
Per prereg §4.2 E0 does not reimplement it: it requires training, and its
compressed-KV anchor attention differs structurally. Listed as an
un-reproduced near neighbour; its own evaluation (Qwen3-VL-2B; MMMU/
MathVision/ODinW-13/RealWorldQA/VideoMME) does not overlap this setting.

## 4. PACE reproduction (§4.1)

(Qwen2.5-VL-7B, official `reproduce_10_percent.sh` settings with
`attn_implementation=sdpa` — the only available attention backend on this
host. Reference scores: RealWorldQA 68.37, MMStar 59.08, ChartQA 73.52.)

STATUS: reproduction run launched; results to be filled in.

## 5. Accuracy (§5.1) — K=256 main table

(To be filled by `e0_analyze.py` from `outputs/e0/acc/*/_score.json`.)

## 6. Efficiency (§5.2) and resolution sweep (§5.3)

(To be filled from `e0_perf_paired.json` / `e0_res_sweep.json`.)

## 7. D1–D4 verdicts

(Mechanical evaluation in `e0_verdict.json` by `e0_analyze.py`.)

## 8. Deviations, refusals and port concerns

- Amendments A1–A5 (budget alignment for FastV/PDrop/SparseVLM/VisionZip;
  N1 lm_head shape; engine/greedy-loop choice; SparseVLM recycling
  bookkeeping; PACE port decisions) — all frozen before the affected
  numbers existed.
- **CIVIC not run** (no code; see §3).
- The host runs an Ollama service that intermittently reserves ~27 GB of the
  46 GB GPU; all E0 GPU scripts carry OOM-retry loops. This is an
  infrastructure hazard for the timing sections, not a method effect.
- R-res runs through the N1-verified keep-all native path rather than a
  separate stock-`generate` call (bit-identical by N1).
- PDrop per-stage retention is budget-scaled (A1); the realized per-stage
  token counts are reported with the results.
