# Stage-1 Grounding Guidance Discovery — frozen protocol (round 1)

Date frozen: 2026-10-01.  Worktree `/media/disk2/YZX/research/EADP_amp`,
branch `codex/anchor-merge-pilot`.

## Fixed (Stage-2 freeze carries over; user directive)

- Backbone Qwen3-VL-8B-Instruct-1024, native E0 engine, greedy decode,
  max_new_tokens=2048, protocol v0.3.
- Official EADP facility-location selector (b1), K=256, identity gather —
  **no Anchor Completion anywhere in this round**.
- Training-free; no new forward passes; no new parameters.
- Panels: the FROZEN anchor-merge-pilot manifest (DEV 100/ds = 300;
  CONFIRM 200/ds = 600 incl. OCRBench top-up).  BASE = the round-2 amp
  BASE shards (gate G2 must prove bank keep == online b1 keep).

## Stage-1 variants (user-approved via Q&A, 2026-10-01)

Official aggregation weight softmax(-H/0.01) is numerically near-one-hot;
a literal additive grounding fusion would be a no-op.  Approved design:

- **ARM A (grounding-only)** `mode="only"`: no entropy; text weight
  softmax(lam * z(g)) over ALL instruction tokens.
- **ARM B (entropy + grounding)** `mode="cal"`: official entropy 20%
  filter kept verbatim; within the kept set the near-one-hot entropy
  weight is REPLACED by softmax(lam * z(g_kept)).
- **ARM C (multi-peak)**: ARM B with g = strong-response fraction
  (share of visual tokens > mean + 1 std).

Grounding statistics (on the existing text-token x visual-token cosine,
actual sign; populations frozen, not swept):
  g1 = mean(top 5%) - mean(all)                     (top-k margin)
  g2 = mean(top 5%) - mean(bottom 10%)              (peak-to-background)
  g3 = (mean(top 5%) - mean(all)) / std(all)        (standardized peak)
  peak = frac(visual > mean + 1*std)                (multi-peak share)

Arm grid (user-approved: run all 7 on DEV):
  A_G1 A_G2 A_G3 | B_G1 B_G2 B_G3 | C_G1

## Calibration rule (frozen BEFORE any full panel)

lam in {0.25, 0.5, 1.0}, calibrated ONCE on a 10/ds smoke (A_G1 + B_G1,
macro), then frozen for all arms and both splits.  No CONFIRM tuning.

Calibration outcome (2026-10-01): keep sets are IDENTICAL across the grid
(checked offline on DEV samples and via smoke predictions), so lam is
uninformative within the approved range; FROZEN lam = 1.0 (largest of the
grid).  Smoke tables also showed the 10/ds subset is underpowered: arms
move ~50/256 kept tokens but the 30 smoke answers were all stable, so
paired smoke deltas are exactly 0 — arm selection therefore happens on
the full DEV panel, not the smoke.

## Decision rules (user directive §7)

- GO: grounding-only within ~0.5 of EADP AND/OR entropy+grounding > EADP.
- KILL: all variants <= EADP -> freeze EADP Stage 1 + Facility + Anchor
  Completion as the DCC pipeline.

## Verdict (2026-10-01)

KILL — all 7 arms negative vs BASE on DEV (best A_G2 −0.317
[−2.811, +2.213]); no CONFIRM run (gated on a DEV winner); combination
experiment NOT_RUN.  See reports/stage1_grounding_guidance_discovery.md.

## Gates (must PASS before accuracy runs)

G1: variant base-path scorer == official TimedEADPPruner._score bitwise.
G2: online b1 keep == frozen amp bank keep on gate samples.
G3: all 7 arms end-to-end smoke, finite, diag captured.
